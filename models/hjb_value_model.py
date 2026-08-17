from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1).float()
    else:
        x = x.float()
    if x.size(-1) == dim:
        return x
    if x.size(-1) > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.size(-1), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HJBValueModel(nn.Module):
    """Small differentiable value model for sampling-time HJB guidance.

    The model intentionally avoids detached risk features in the default forward
    path.  It uses ligand coordinates, atom-type features, protein-pocket
    features, and time, so ``autograd.grad(s.sum(), ligand_pos)`` gives a usable
    coordinate-space direction.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 27,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        output_dim: int = 1,
        head_names: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        if isinstance(head_names, str):
            head_names = [x.strip() for x in head_names.split(",") if x.strip()]
        self.head_names = list(head_names) if head_names is not None else [f"head{i}" for i in range(self.output_dim)]
        self.ligand_encoder = MLP(3 + self.ligand_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.protein_encoder = MLP(3 + self.protein_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        # ligand pool, protein pool, interaction features, time
        self.value_head = MLP(2 * hidden_dim + 5 + 1, hidden_dim, self.output_dim, num_layers=num_layers, dropout=dropout)

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().view(-1).clamp(min=0, max=self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def _protein_features(self, protein_v: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(protein_v, self.protein_feature_dim)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
        lig_feat = self._ligand_features(ligand_v)
        prot_feat = self._protein_features(protein_v)

        lig_center = scatter_mean(ligand_pos, batch_ligand, dim=0, dim_size=num_graphs)
        prot_center = scatter_mean(protein_pos, batch_protein, dim=0, dim_size=num_graphs)
        lig_rel = ligand_pos - lig_center.index_select(0, batch_ligand)
        prot_rel = protein_pos - prot_center.index_select(0, batch_protein)

        lig_node = self.ligand_encoder(torch.cat([lig_rel, lig_feat], dim=-1))
        prot_node = self.protein_encoder(torch.cat([prot_rel, prot_feat], dim=-1))
        lig_pool = scatter_mean(lig_node, batch_ligand, dim=0, dim_size=num_graphs)
        prot_pool = scatter_mean(prot_node, batch_protein, dim=0, dim_size=num_graphs)

        interaction = []
        for graph_idx in range(num_graphs):
            lm = batch_ligand == graph_idx
            pm = batch_protein == graph_idx
            if bool(lm.any()) and bool(pm.any()):
                d = torch.cdist(ligand_pos[lm], protein_pos[pm])
                min_d = d.min()
                soft_min = -torch.logsumexp(-d.reshape(-1), dim=0)
                clash = torch.relu(torch.as_tensor(1.6, dtype=d.dtype, device=d.device) - d).pow(2).mean()
                contact = torch.exp(-((d - 3.5) / 1.0).pow(2)).mean()
            else:
                min_d = ligand_pos.new_tensor(0.0)
                soft_min = ligand_pos.new_tensor(0.0)
                clash = ligand_pos.new_tensor(0.0)
                contact = ligand_pos.new_tensor(0.0)
            com_dist = (lig_center[graph_idx] - prot_center[graph_idx]).norm()
            interaction.append(torch.stack([min_d, soft_min, clash, contact, com_dist]))
        inter = torch.stack(interaction, dim=0)

        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        x = torch.cat([lig_pool, prot_pool, inter, time_fraction], dim=-1)
        out = self.value_head(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


class PairwiseHJBValueModel(nn.Module):
    """Local protein-ligand pairwise value model for coordinate guidance.

    Unlike :class:`HJBValueModel`, this model makes the scalar value depend
    directly on differentiable ligand-protein atom-pair terms.  The intended
    use is not only state ranking, but a more local coordinate gradient:
    ``autograd.grad(S.sum(), ligand_pos)`` receives signal from nearby protein
    atoms through distance RBF features.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 27,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        output_dim: int = 1,
        head_names: list[str] | tuple[str, ...] | None = None,
        rbf_dim: int = 24,
        cutoff: float = 6.0,
        cutoff_temperature: float = 0.5,
    ):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.rbf_dim = int(rbf_dim)
        self.cutoff = float(cutoff)
        self.cutoff_temperature = float(cutoff_temperature)
        if isinstance(head_names, str):
            head_names = [x.strip() for x in head_names.split(",") if x.strip()]
        self.head_names = list(head_names) if head_names is not None else [f"head{i}" for i in range(self.output_dim)]
        self.ligand_encoder = MLP(3 + self.ligand_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.protein_encoder = MLP(3 + self.protein_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        pair_dim = self.ligand_feature_dim + self.protein_feature_dim + self.rbf_dim + 4
        self.pair_head = MLP(pair_dim, hidden_dim, self.output_dim, num_layers=num_layers, dropout=dropout)
        self.global_head = MLP(2 * hidden_dim + 6 + 1, hidden_dim, self.output_dim, num_layers=num_layers, dropout=dropout)
        centers = torch.linspace(0.0, self.cutoff, self.rbf_dim)
        self.register_buffer("rbf_centers", centers, persistent=False)
        self.rbf_gamma = 1.0 / max((self.cutoff / max(self.rbf_dim - 1, 1)) ** 2, 1e-6)

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().view(-1).clamp(min=0, max=self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def _protein_features(self, protein_v: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(protein_v, self.protein_feature_dim)

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.to(dtype=d.dtype, device=d.device)
        return torch.exp(-self.rbf_gamma * (d.unsqueeze(-1) - centers).pow(2))

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
        lig_feat = self._ligand_features(ligand_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        prot_feat = self._protein_features(protein_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)

        lig_center = scatter_mean(ligand_pos, batch_ligand, dim=0, dim_size=num_graphs)
        prot_center = scatter_mean(protein_pos, batch_protein, dim=0, dim_size=num_graphs)
        lig_rel = ligand_pos - lig_center.index_select(0, batch_ligand)
        prot_rel = protein_pos - prot_center.index_select(0, batch_protein)
        lig_node = self.ligand_encoder(torch.cat([lig_rel, lig_feat], dim=-1))
        prot_node = self.protein_encoder(torch.cat([prot_rel, prot_feat], dim=-1))
        lig_pool = scatter_mean(lig_node, batch_ligand, dim=0, dim_size=num_graphs)
        prot_pool = scatter_mean(prot_node, batch_protein, dim=0, dim_size=num_graphs)

        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(dtype=ligand_pos.dtype, device=ligand_pos.device)

        pair_values = []
        global_scalars = []
        eps = ligand_pos.new_tensor(1e-8)
        for graph_idx in range(num_graphs):
            lm = batch_ligand == graph_idx
            pm = batch_protein == graph_idx
            if bool(lm.any()) and bool(pm.any()):
                lig_pos_g = ligand_pos[lm]
                prot_pos_g = protein_pos[pm]
                lig_feat_g = lig_feat[lm]
                prot_feat_g = prot_feat[pm]
                d = torch.cdist(lig_pos_g, prot_pos_g)
                n_lig = max(int(lig_pos_g.size(0)), 1)
                n_pair = max(int(d.numel()), 1)
                rbf = self._rbf(d)
                smooth_cut = torch.sigmoid((self.cutoff - d) / max(self.cutoff_temperature, 1e-6))
                close = torch.relu(ligand_pos.new_tensor(2.4) - d).pow(2)
                clash = torch.relu(ligand_pos.new_tensor(1.6) - d).pow(2)
                contact = torch.exp(-((d - 3.5) / 1.0).pow(2))
                lig_pair = lig_feat_g[:, None, :].expand(-1, prot_feat_g.size(0), -1)
                prot_pair = prot_feat_g[None, :, :].expand(lig_feat_g.size(0), -1, -1)
                pair_extra = torch.stack(
                    [
                        d / self.cutoff,
                        smooth_cut,
                        close,
                        contact,
                    ],
                    dim=-1,
                )
                pair_input = torch.cat([lig_pair, prot_pair, rbf, pair_extra], dim=-1).reshape(n_pair, -1)
                pair_raw = self.pair_head(pair_input).reshape(lig_feat_g.size(0), prot_feat_g.size(0), self.output_dim)
                pair_value = (pair_raw * smooth_cut.unsqueeze(-1)).sum(dim=(0, 1)) / float(n_lig)
                nearest_lig = d.min(dim=1).values
                nearest_prot = d.min(dim=0).values
                scalar = torch.stack(
                    [
                        d.min() / self.cutoff,
                        nearest_lig.mean() / self.cutoff,
                        smooth_cut.sum() / float(n_lig),
                        close.sum() / float(n_lig),
                        clash.sum() / float(n_lig),
                        contact.sum() / float(n_lig),
                    ]
                )
                _ = nearest_prot  # Kept for readability; protein coverage may be added later.
            else:
                pair_value = ligand_pos.new_zeros((self.output_dim,))
                scalar = ligand_pos.new_zeros((6,))
            pair_values.append(pair_value)
            global_scalars.append(scalar)
        pair_value = torch.stack(pair_values, dim=0)
        global_scalar = torch.stack(global_scalars, dim=0)
        global_input = torch.cat([lig_pool, prot_pool, global_scalar, time_fraction], dim=-1)
        out = pair_value + self.global_head(global_input)
        return out.squeeze(-1) if self.output_dim == 1 else out


class PhysicalPairwiseHJBValueModel(PairwiseHJBValueModel):
    """Pairwise value model with explicit ligand-ligand distance features.

    The original pairwise model exposes protein-ligand distances to the value
    head, while intraligand geometry only enters through pooled atom features.
    This variant adds a differentiable ligand-ligand branch so augmented
    structural labels can shape coordinate gradients during sampling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ligand_pair_dim = 2 * self.ligand_feature_dim + self.rbf_dim + 5
        self.ligand_pair_head = MLP(
            ligand_pair_dim,
            self.hidden_dim,
            self.output_dim,
            num_layers=3,
            dropout=0.0,
        )

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        base = super().forward(
            ligand_pos,
            ligand_v,
            protein_pos,
            protein_v,
            batch_ligand,
            batch_protein,
            time_fraction,
        )
        if base.dim() == 1:
            base = base.unsqueeze(-1)

        lig_feat = self._ligand_features(ligand_v).to(
            dtype=ligand_pos.dtype,
            device=ligand_pos.device,
        )
        num_graphs = int(batch_ligand.max().item()) + 1
        pair_values = []
        for graph_idx in range(num_graphs):
            mask = batch_ligand == graph_idx
            pos = ligand_pos[mask]
            feat = lig_feat[mask]
            n_lig = int(pos.size(0))
            if n_lig < 2:
                pair_values.append(ligand_pos.new_zeros((self.output_dim,)))
                continue

            pair_idx = torch.triu_indices(n_lig, n_lig, offset=1, device=pos.device)
            i, j = pair_idx[0], pair_idx[1]
            distance = (pos[i] - pos[j]).norm(dim=-1).clamp_min(1e-6)
            smooth_cut = torch.sigmoid(
                (self.cutoff - distance) / max(self.cutoff_temperature, 1e-6)
            )
            short = torch.relu(ligand_pos.new_tensor(1.4) - distance).pow(2)
            near = torch.exp(-((distance - 1.5) / 0.45).pow(2))
            mid = torch.exp(-((distance - 2.5) / 0.75).pow(2))
            same_type = (feat[i].argmax(dim=-1) == feat[j].argmax(dim=-1)).to(distance.dtype)
            pair_extra = torch.stack(
                [distance / self.cutoff, smooth_cut, short, near + mid, same_type],
                dim=-1,
            )
            pair_input = torch.cat(
                [feat[i], feat[j], self._rbf(distance), pair_extra],
                dim=-1,
            )
            raw = self.ligand_pair_head(pair_input)
            pair_value = (raw * smooth_cut.unsqueeze(-1)).sum(dim=0) / float(n_lig)
            pair_values.append(pair_value)

        out = base + torch.stack(pair_values, dim=0)
        return out.squeeze(-1) if self.output_dim == 1 else out


class TriangleAwareHJBValueModel(PhysicalPairwiseHJBValueModel):
    """Physical pair model with explicit three-body molecular reasoning.

    Ligand triangles expose angle and local-shape consistency, while mixed
    protein-ligand-ligand triangles expose orientation-dependent pocket
    contacts.  Nearest-neighbour truncation keeps the three-body branches
    practical during coordinate-gradient sampling.
    """

    def __init__(
        self,
        *args,
        triangle_ligand_neighbors: int = 8,
        triangle_pocket_neighbors: int = 6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.triangle_ligand_neighbors = int(triangle_ligand_neighbors)
        self.triangle_pocket_neighbors = int(triangle_pocket_neighbors)
        ligand_triangle_dim = 3 * self.ligand_feature_dim + 3 * self.rbf_dim + 5
        pocket_triangle_dim = (
            2 * self.ligand_feature_dim
            + self.protein_feature_dim
            + 3 * self.rbf_dim
            + 5
        )
        self.ligand_triangle_head = MLP(
            ligand_triangle_dim,
            self.hidden_dim,
            self.output_dim,
            num_layers=3,
            dropout=0.0,
        )
        self.pocket_triangle_head = MLP(
            pocket_triangle_dim,
            self.hidden_dim,
            self.output_dim,
            num_layers=3,
            dropout=0.0,
        )

    def _ligand_triangle_value(self, pos, feat):
        n_lig = int(pos.size(0))
        if n_lig < 3:
            return pos.new_zeros((self.output_dim,))
        distance = torch.cdist(pos, pos)
        values = []
        for center in range(n_lig):
            candidates = torch.cat(
                [
                    torch.arange(center, device=pos.device),
                    torch.arange(center + 1, n_lig, device=pos.device),
                ]
            )
            if candidates.numel() < 2:
                continue
            keep = min(int(candidates.numel()), self.triangle_ligand_neighbors)
            nearest = candidates[torch.topk(distance[center, candidates], keep, largest=False).indices]
            pairs = torch.triu_indices(keep, keep, offset=1, device=pos.device)
            i, k = nearest[pairs[0]], nearest[pairs[1]]
            j = torch.full_like(i, center)
            d_ji = (pos[i] - pos[j]).norm(dim=-1).clamp_min(1e-6)
            d_jk = (pos[k] - pos[j]).norm(dim=-1).clamp_min(1e-6)
            d_ik = (pos[i] - pos[k]).norm(dim=-1).clamp_min(1e-6)
            vi = pos[i] - pos[j]
            vk = pos[k] - pos[j]
            cosine = (vi * vk).sum(dim=-1) / (d_ji * d_jk).clamp_min(1e-8)
            area = torch.cross(vi, vk, dim=-1).norm(dim=-1) / (d_ji * d_jk).clamp_min(1e-8)
            smooth = (
                torch.sigmoid((self.cutoff - d_ji) / max(self.cutoff_temperature, 1e-6))
                * torch.sigmoid((self.cutoff - d_jk) / max(self.cutoff_temperature, 1e-6))
            )
            endpoint_sum = feat[i] + feat[k]
            endpoint_diff = (feat[i] - feat[k]).abs()
            extra = torch.stack(
                [
                    d_ji / self.cutoff,
                    d_jk / self.cutoff,
                    d_ik / self.cutoff,
                    cosine.clamp(-1.0, 1.0),
                    area,
                ],
                dim=-1,
            )
            tri_input = torch.cat(
                [
                    feat[j],
                    endpoint_sum,
                    endpoint_diff,
                    self._rbf(d_ji),
                    self._rbf(d_jk),
                    self._rbf(d_ik),
                    extra,
                ],
                dim=-1,
            )
            values.append((self.ligand_triangle_head(tri_input) * smooth.unsqueeze(-1)).sum(dim=0))
        if not values:
            return pos.new_zeros((self.output_dim,))
        return torch.stack(values, dim=0).sum(dim=0) / float(n_lig)

    def _pocket_triangle_value(self, lig_pos, lig_feat, prot_pos, prot_feat):
        n_lig = int(lig_pos.size(0))
        n_prot = int(prot_pos.size(0))
        if n_lig < 2 or n_prot < 1:
            return lig_pos.new_zeros((self.output_dim,))
        ll_distance = torch.cdist(lig_pos, lig_pos)
        lp_distance = torch.cdist(lig_pos, prot_pos)
        values = []
        for center in range(n_lig):
            lig_candidates = torch.cat(
                [
                    torch.arange(center, device=lig_pos.device),
                    torch.arange(center + 1, n_lig, device=lig_pos.device),
                ]
            )
            lig_keep = min(int(lig_candidates.numel()), self.triangle_ligand_neighbors)
            prot_keep = min(n_prot, self.triangle_pocket_neighbors)
            lig_near = lig_candidates[
                torch.topk(ll_distance[center, lig_candidates], lig_keep, largest=False).indices
            ]
            prot_near = torch.topk(lp_distance[center], prot_keep, largest=False).indices
            k = lig_near[:, None].expand(-1, prot_keep).reshape(-1)
            p = prot_near[None, :].expand(lig_keep, -1).reshape(-1)
            j = torch.full_like(k, center)
            d_jk = (lig_pos[k] - lig_pos[j]).norm(dim=-1).clamp_min(1e-6)
            d_jp = (prot_pos[p] - lig_pos[j]).norm(dim=-1).clamp_min(1e-6)
            d_kp = (prot_pos[p] - lig_pos[k]).norm(dim=-1).clamp_min(1e-6)
            v_lig = lig_pos[k] - lig_pos[j]
            v_prot = prot_pos[p] - lig_pos[j]
            cosine = (v_lig * v_prot).sum(dim=-1) / (d_jk * d_jp).clamp_min(1e-8)
            close = torch.relu(lig_pos.new_tensor(2.4) - d_jp).pow(2)
            smooth = (
                torch.sigmoid((self.cutoff - d_jk) / max(self.cutoff_temperature, 1e-6))
                * torch.sigmoid((self.cutoff - d_jp) / max(self.cutoff_temperature, 1e-6))
            )
            extra = torch.stack(
                [
                    d_jk / self.cutoff,
                    d_jp / self.cutoff,
                    d_kp / self.cutoff,
                    cosine.clamp(-1.0, 1.0),
                    close,
                ],
                dim=-1,
            )
            tri_input = torch.cat(
                [
                    lig_feat[j],
                    lig_feat[k],
                    prot_feat[p],
                    self._rbf(d_jk),
                    self._rbf(d_jp),
                    self._rbf(d_kp),
                    extra,
                ],
                dim=-1,
            )
            values.append((self.pocket_triangle_head(tri_input) * smooth.unsqueeze(-1)).sum(dim=0))
        return torch.stack(values, dim=0).sum(dim=0) / float(n_lig)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        base = super().forward(
            ligand_pos,
            ligand_v,
            protein_pos,
            protein_v,
            batch_ligand,
            batch_protein,
            time_fraction,
        )
        if base.dim() == 1:
            base = base.unsqueeze(-1)
        lig_feat = self._ligand_features(ligand_v).to(ligand_pos.device, ligand_pos.dtype)
        prot_feat = self._protein_features(protein_v).to(protein_pos.device, protein_pos.dtype)
        num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item())) + 1
        triangle_values = []
        for graph_idx in range(num_graphs):
            lm = batch_ligand == graph_idx
            pm = batch_protein == graph_idx
            ligand_value = self._ligand_triangle_value(ligand_pos[lm], lig_feat[lm])
            pocket_value = self._pocket_triangle_value(
                ligand_pos[lm], lig_feat[lm], protein_pos[pm], prot_feat[pm]
            )
            triangle_values.append(ligand_value + pocket_value)
        out = base + torch.stack(triangle_values, dim=0)
        return out.squeeze(-1) if self.output_dim == 1 else out


def select_hjb_value(output: torch.Tensor, component: str = "total", head_names: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
    """Select or combine scalar values from scalar or multi-head HJB output."""
    if output.dim() == 1:
        return output
    names = list(head_names or ["safe", "dock", "drug"][: output.size(-1)])
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    component = str(component or "total")
    aliases = {
        "safe_only": "safe",
        "dock_only": "dock",
        "drug_only": "drug",
        "all": "total",
        "safe+dock+drug": "total",
        "constrained_dock": "dock_constrained",
        "projected_multi": "dock_projected_multi",
        "multi_projected": "dock_projected_multi",
    }
    component = aliases.get(component, component)
    if component in {"total", "sum", "auto"}:
        return output.sum(dim=-1)
    if component == "safe+dock":
        return output[:, name_to_idx.get("safe", 0)] + output[:, name_to_idx.get("dock", min(1, output.size(-1) - 1))]
    if component in {"dock_constrained", "dock_constrained_soft", "dock_constrained_strong"}:
        dock_idx = name_to_idx.get("dock", min(0, output.size(-1) - 1))
        weight = {
            "dock_constrained_soft": 0.25,
            "dock_constrained": 0.50,
            "dock_constrained_strong": 1.00,
        }[component]
        value = output[:, dock_idx]
        for name in ("min", "score", "valid"):
            idx = name_to_idx.get(name)
            if idx is not None and idx < output.size(-1):
                value = value + float(weight) * F.relu(output[:, idx])
        return value
    if component == "dock_projected_multi":
        return output[:, name_to_idx.get("dock", min(0, output.size(-1) - 1))]
    if component in name_to_idx:
        return output[:, name_to_idx[component]]
    raise ValueError(f"Unknown HJB value component {component!r}; available heads={names}")


def build_hjb_value_model_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> HJBValueModel:
    args = ckpt.get("args", {})
    model_type = str(args.get("model_type", ckpt.get("model_type", "")))
    model_arch = str(args.get("model_arch", ckpt.get("model_arch", "")))
    if model_type == "frozen_paflow_scalar_potential":
        from models.paflow_feature_potential import FrozenPAFlowHJBValue

        model = FrozenPAFlowHJBValue.from_checkpoint(
            checkpoint_path=ckpt.get("paflow_checkpoint", args.get("paflow_checkpoint")),
            device=device,
            hidden_dim=int(ckpt.get("hidden_dim", args.get("hidden_dim", 96))),
            dropout=float(ckpt.get("dropout", args.get("dropout", 0.05))),
            sampler_steps=int(ckpt.get("sampler_steps", 50)),
            paflow_timesteps=int(ckpt.get("paflow_timesteps", 1000)),
            fix_x=bool(ckpt.get("fix_x", False)),
        ).to(device)
        model.head.load_state_dict(ckpt["head_state_dict"])
        model.head_names = list(ckpt.get("head_names", ["total"]))
        return model
    if model_type == "enhanced" or ckpt.get("use_interaction_features") or ckpt.get("use_physical_branch"):
        from scripts.train_final_pose_reranker import EnhancedFinalPoseReranker

        class SamplingEnhancedFinalPoseReranker(EnhancedFinalPoseReranker):
            def __init__(self, *m_args, feature_mean=None, feature_std=None, **m_kwargs):
                super().__init__(*m_args, **m_kwargs)
                self.register_buffer(
                    "sampling_feature_mean",
                    torch.as_tensor(feature_mean, dtype=torch.float32) if feature_mean is not None else torch.empty(0),
                    persistent=False,
                )
                self.register_buffer(
                    "sampling_feature_std",
                    torch.as_tensor(feature_std, dtype=torch.float32) if feature_std is not None else torch.empty(0),
                    persistent=False,
                )
                self.sampling_max_ligand_types = int(m_kwargs.get("max_ligand_types", 23))
                self.sampling_protein_element_dim = int(args.get("protein_element_dim", 6))
                self.sampling_pair_hist_bins = int(args.get("pair_hist_bins", 0))
                self.sampling_pair_hist_max_dist = float(args.get("pair_hist_max_dist", 6.0))

            def _batched_interaction_features(
                self,
                ligand_pos,
                ligand_v,
                protein_pos,
                protein_v,
                batch_ligand,
                batch_protein,
            ):
                features = []
                num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                max_ligand_types = int(self.sampling_max_ligand_types)
                protein_element_dim = int(self.sampling_protein_element_dim)
                pair_hist_bins = int(self.sampling_pair_hist_bins)
                pair_hist_max_dist = float(self.sampling_pair_hist_max_dist)
                dtype = ligand_pos.dtype
                for graph_idx in range(num_graphs):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    lig_pos_g = ligand_pos[lm]
                    prot_pos_g = protein_pos[pm]
                    lig_v_g = ligand_v[lm]
                    prot_v_g = protein_v[pm]
                    if lig_pos_g.numel() == 0 or prot_pos_g.numel() == 0:
                        dim = 16 + max_ligand_types + protein_element_dim + max_ligand_types * protein_element_dim * max(pair_hist_bins, 0)
                        features.append(ligand_pos.new_zeros((dim,)))
                        continue

                    d = torch.cdist(lig_pos_g, prot_pos_g)
                    nearest_lig = d.min(dim=1).values
                    nearest_prot = d.min(dim=0).values
                    n_lig = max(int(lig_pos_g.size(0)), 1)
                    n_prot = max(int(prot_pos_g.size(0)), 1)
                    shell = torch.exp(-((d - 3.5) / 0.75).pow(2))
                    contact45 = (d < 4.5).to(dtype)
                    contact40 = (d < 4.0).to(dtype)
                    close24 = (d < 2.4).to(dtype)
                    clash20 = (d < 2.0).to(dtype)

                    lig_center = lig_pos_g.mean(dim=0)
                    prot_center = prot_pos_g.mean(dim=0)
                    lig_radius = (lig_pos_g - lig_center).norm(dim=-1).mean()
                    contact_mask = nearest_prot < 4.5
                    if bool(contact_mask.any()):
                        prot_contact_radius = (prot_pos_g[contact_mask] - prot_center).norm(dim=-1).mean()
                    else:
                        prot_contact_radius = ligand_pos.new_tensor(0.0)

                    scalar = torch.stack(
                        [
                            torch.log1p(ligand_pos.new_tensor(float(n_lig))),
                            torch.log1p(ligand_pos.new_tensor(float(n_prot))),
                            nearest_lig.min() / 5.0,
                            nearest_lig.mean() / 5.0,
                            nearest_lig.std(unbiased=False) / 5.0,
                            (nearest_lig < 3.5).to(dtype).mean(),
                            (nearest_lig < 4.5).to(dtype).mean(),
                            (nearest_lig > 6.0).to(dtype).mean(),
                            contact40.sum() / float(n_lig),
                            contact45.sum() / float(n_lig),
                            shell.sum() / float(n_lig),
                            close24.sum() / float(n_lig),
                            clash20.sum() / float(n_lig),
                            (nearest_prot < 4.5).to(dtype).mean(),
                            (lig_center - prot_center).norm() / 10.0,
                            (lig_radius + prot_contact_radius) / 10.0,
                        ]
                    )

                    if lig_v_g.dim() == 1 or (lig_v_g.dim() == 2 and lig_v_g.size(-1) == 1):
                        lig_ids = lig_v_g.long().view(-1).clamp(min=0, max=max_ligand_types - 1)
                    else:
                        lig_ids = lig_v_g[..., :max_ligand_types].argmax(dim=-1).long().clamp(min=0, max=max_ligand_types - 1)
                    lig_type_contact = ligand_pos.new_zeros((max_ligand_types,))
                    per_lig_shell = shell.sum(dim=1) / max(float(n_prot), 1.0)
                    for atom_type in range(max_ligand_types):
                        mask = lig_ids == atom_type
                        if bool(mask.any()):
                            lig_type_contact[atom_type] = per_lig_shell[mask].sum() / float(n_lig)

                    if prot_v_g.dim() == 2 and prot_v_g.size(1) > 0:
                        elem = prot_v_g[:, :protein_element_dim].float()
                        if elem.size(1) < protein_element_dim:
                            elem = F.pad(elem, (0, protein_element_dim - elem.size(1)))
                    else:
                        elem = ligand_pos.new_zeros((n_prot, protein_element_dim))
                    per_prot_shell = shell.sum(dim=0) / float(n_lig)
                    prot_element_contact = (elem.to(dtype) * per_prot_shell.unsqueeze(-1)).sum(dim=0) / float(n_prot)

                    parts = [scalar, lig_type_contact, prot_element_contact]
                    if pair_hist_bins > 0:
                        bins = torch.linspace(2.0, pair_hist_max_dist, pair_hist_bins, dtype=dtype, device=ligand_pos.device)
                        width = max((pair_hist_max_dist - 2.0) / max(pair_hist_bins - 1, 1), 0.5)
                        lig_onehot = F.one_hot(lig_ids, num_classes=max_ligand_types).to(dtype)
                        hist_parts = []
                        for center in bins:
                            w = torch.exp(-((d - center) / width).pow(2))
                            hist = torch.einsum("il,jp,ij->lp", lig_onehot, elem.to(dtype), w) / float(n_lig)
                            hist_parts.append(hist.reshape(-1))
                        parts.append(torch.cat(hist_parts, dim=0))
                    features.append(torch.nan_to_num(torch.cat(parts, dim=0), nan=0.0, posinf=0.0, neginf=0.0))
                feat = torch.stack(features, dim=0)
                if self.sampling_feature_mean.numel() and self.sampling_feature_std.numel():
                    feat = (feat - self.sampling_feature_mean.to(feat.device, feat.dtype)) / self.sampling_feature_std.to(feat.device, feat.dtype).clamp_min(1e-6)
                return feat

            def forward(
                self,
                ligand_pos,
                ligand_v,
                protein_pos,
                protein_v,
                batch_ligand,
                batch_protein,
                time_fraction,
                interaction_features=None,
            ):
                if self.use_interaction_features and interaction_features is None:
                    interaction_features = self._batched_interaction_features(
                        ligand_pos,
                        ligand_v,
                        protein_pos,
                        protein_v,
                        batch_ligand,
                        batch_protein,
                    )
                return super().forward(
                    ligand_pos,
                    ligand_v,
                    protein_pos,
                    protein_v,
                    batch_ligand,
                    batch_protein,
                    time_fraction,
                    interaction_features=interaction_features,
                )

        model = SamplingEnhancedFinalPoseReranker(
            hidden_dim=int(args.get("hidden_dim", 192)),
            num_layers=int(args.get("num_layers", 4)),
            dropout=float(args.get("dropout", 0.0)),
            feature_dim=int(ckpt.get("feature_dim", 0)),
            use_interaction_features=bool(args.get("use_interaction_features", ckpt.get("use_interaction_features", False))),
            use_physical_branch=bool(args.get("use_physical_branch", ckpt.get("use_physical_branch", False))),
            max_ligand_types=int(args.get("max_ligand_types", ckpt.get("ligand_feature_dim", 23))),
            protein_feature_dim=int(ckpt.get("protein_feature_dim", 27)),
            physical_rbf_dim=int(args.get("physical_rbf_dim", 32)),
            physical_cutoff=float(args.get("physical_cutoff", 6.0)),
            physical_gamma_init=float(args.get("physical_gamma_init", 0.03)),
            feature_mean=ckpt.get("feature_mean"),
            feature_std=ckpt.get("feature_std"),
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        return model

    if model_arch in {"pairwise", "physical_pairwise", "triangle_aware"} or model_type in {
        "hjb_value_pairwise",
        "hjb_value_pairwise_multihead",
        "hjb_value_physical_pairwise",
        "hjb_value_physical_pairwise_multihead",
        "hjb_value_triangle_aware",
        "hjb_value_triangle_aware_multihead",
    }:
        model_cls = {
            "pairwise": PairwiseHJBValueModel,
            "physical_pairwise": PhysicalPairwiseHJBValueModel,
            "triangle_aware": TriangleAwareHJBValueModel,
        }.get(model_arch, PairwiseHJBValueModel)
        triangle_kwargs = {}
        if model_cls is TriangleAwareHJBValueModel:
            triangle_kwargs = {
                "triangle_ligand_neighbors": int(
                    args.get("triangle_ligand_neighbors", ckpt.get("triangle_ligand_neighbors", 8))
                ),
                "triangle_pocket_neighbors": int(
                    args.get("triangle_pocket_neighbors", ckpt.get("triangle_pocket_neighbors", 6))
                ),
            }
        model = model_cls(
            ligand_feature_dim=int(args.get("ligand_feature_dim", ckpt.get("ligand_feature_dim", 16))),
            protein_feature_dim=int(args.get("protein_feature_dim", ckpt.get("protein_feature_dim", 27))),
            hidden_dim=int(args.get("hidden_dim", 128)),
            num_layers=int(args.get("num_layers", 3)),
            dropout=float(args.get("dropout", 0.0)),
            output_dim=int(args.get("output_dim", ckpt.get("output_dim", 1))),
            head_names=args.get("head_names", ckpt.get("head_names")),
            rbf_dim=int(args.get("pair_rbf_dim", ckpt.get("pair_rbf_dim", 24))),
            cutoff=float(args.get("pair_cutoff", ckpt.get("pair_cutoff", 6.0))),
            cutoff_temperature=float(args.get("pair_cutoff_temperature", ckpt.get("pair_cutoff_temperature", 0.5))),
            **triangle_kwargs,
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state)
        return model

    model = HJBValueModel(
        ligand_feature_dim=int(args.get("ligand_feature_dim", ckpt.get("ligand_feature_dim", 16))),
        protein_feature_dim=int(args.get("protein_feature_dim", ckpt.get("protein_feature_dim", 27))),
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_layers=int(args.get("num_layers", 3)),
        dropout=float(args.get("dropout", 0.0)),
        output_dim=int(args.get("output_dim", ckpt.get("output_dim", 1))),
        head_names=args.get("head_names", ckpt.get("head_names")),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    return model

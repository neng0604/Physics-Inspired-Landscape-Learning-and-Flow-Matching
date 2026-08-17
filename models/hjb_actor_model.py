from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.egnn import EGNN


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


class HJBActorModel(nn.Module):
    """Value-conditioned coordinate residual policy.

    The actor predicts an atom-wise coordinate correction from the frozen value
    landscape signal. It is intentionally small and local: the sampler can still
    rescale the returned correction to a target correction/FM ratio.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        scalar_dim: int = 4,
    ):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.scalar_dim = int(scalar_dim)
        input_dim = 3 + self.ligand_feature_dim + 3 + 3 + self.scalar_dim
        layers = []
        dim = input_dim
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 3))
        self.net = nn.Sequential(*layers)
        # Start close to zero so the actor must earn its deviation from FM,
        # but avoid an exactly zero vector because ratio-rescaling would
        # otherwise have no gradient at initialization.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(last.bias)

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().view(-1).clamp(min=0, max=self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        neg_grad_s: torch.Tensor,
        fm_dx: torch.Tensor,
        graph_scalars: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(batch_ligand.max().item()) + 1
        lig_center = scatter_mean(ligand_pos, batch_ligand, dim=0, dim_size=num_graphs)
        lig_rel = ligand_pos - lig_center.index_select(0, batch_ligand)
        lig_feat = self._ligand_features(ligand_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        scalars = graph_scalars.to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(-1)
        if scalars.size(-1) != self.scalar_dim:
            scalars = _pad_or_trim(scalars, self.scalar_dim)
        node_scalars = scalars.index_select(0, batch_ligand)
        x = torch.cat([lig_rel, lig_feat, neg_grad_s, fm_dx, node_scalars], dim=-1)
        return self.net(x)


class EquivariantHJBActorModel(nn.Module):
    """E(3)-equivariant residual controller over protein-ligand geometry.

    The model builds a mixed protein-ligand graph and uses the existing EGNN
    coordinate update as the residual action.  It keeps the same conceptual
    inputs as ``HJBActorModel`` but additionally consumes protein nodes, so the
    action can depend on local pocket geometry while remaining equivariant.
    """

    uses_protein_context = True

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        scalar_dim: int = 4,
        num_r_gaussian: int = 20,
        k: int = 24,
        cutoff: float = 8.0,
        cutoff_mode: str = "hybrid",
    ):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.scalar_dim = int(scalar_dim)
        self.num_r_gaussian = int(num_r_gaussian)
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.cutoff_mode = str(cutoff_mode)
        lig_input_dim = self.ligand_feature_dim + 3 + 3 + self.scalar_dim + 2
        prot_input_dim = self.protein_feature_dim + 3 + 3 + self.scalar_dim + 2
        self.lig_proj = nn.Sequential(nn.Linear(lig_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.prot_proj = nn.Sequential(nn.Linear(prot_input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.egnn = EGNN(
            num_layers=int(num_layers),
            hidden_dim=hidden_dim,
            edge_feat_dim=4,
            num_r_gaussian=int(num_r_gaussian),
            k=int(k),
            cutoff=float(cutoff),
            cutoff_mode=str(cutoff_mode),
            update_x=True,
            act_fn="silu",
            norm=False,
        )
        self.action_scale = nn.Parameter(torch.tensor(1.0))

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().view(-1).clamp(min=0, max=self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def _protein_features(self, protein_v: torch.Tensor) -> torch.Tensor:
        if protein_v.dim() == 1 or (protein_v.dim() == 2 and protein_v.size(-1) == 1):
            ids = protein_v.long().view(-1).clamp(min=0, max=self.protein_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.protein_feature_dim).float()
        return _pad_or_trim(protein_v, self.protein_feature_dim)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        neg_grad_s: torch.Tensor,
        fm_dx: torch.Tensor,
        graph_scalars: torch.Tensor,
        *,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(batch_ligand.max().item()) + 1
        scalars = graph_scalars.to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(-1)
        if scalars.size(-1) != self.scalar_dim:
            scalars = _pad_or_trim(scalars, self.scalar_dim)
        lig_scalars = scalars.index_select(0, batch_ligand)
        prot_scalars = scalars.index_select(0, batch_protein)
        lig_feat = self._ligand_features(ligand_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        prot_feat = self._protein_features(protein_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        prot_zero_vec = torch.zeros(protein_pos.size(0), 3, dtype=ligand_pos.dtype, device=ligand_pos.device)
        lig_role = torch.tensor([1.0, 0.0], dtype=ligand_pos.dtype, device=ligand_pos.device).view(1, 2).expand(ligand_pos.size(0), -1)
        prot_role = torch.tensor([0.0, 1.0], dtype=ligand_pos.dtype, device=ligand_pos.device).view(1, 2).expand(protein_pos.size(0), -1)
        lig_h = self.lig_proj(torch.cat([lig_feat, neg_grad_s, fm_dx, lig_scalars, lig_role], dim=-1))
        prot_h = self.prot_proj(torch.cat([prot_feat, prot_zero_vec, prot_zero_vec, prot_scalars, prot_role], dim=-1))
        h = self.dropout(torch.cat([lig_h, prot_h], dim=0))
        x = torch.cat([ligand_pos, protein_pos], dim=0)
        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        mask_ligand = torch.cat(
            [
                torch.ones(ligand_pos.size(0), dtype=ligand_pos.dtype, device=ligand_pos.device),
                torch.zeros(protein_pos.size(0), dtype=ligand_pos.dtype, device=ligand_pos.device),
            ],
            dim=0,
        )
        out = self.egnn(h, x, mask_ligand=mask_ligand, batch=batch, return_all=False)
        action = out["x"][: ligand_pos.size(0)] - ligand_pos
        return action * self.action_scale


def build_hjb_actor_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> HJBActorModel:
    args = ckpt.get("args", {})
    actor_arch = str(args.get("actor_arch", ckpt.get("actor_arch", "mlp")))
    if actor_arch in {"egnn", "equivariant"}:
        model = EquivariantHJBActorModel(
            ligand_feature_dim=int(args.get("ligand_feature_dim", ckpt.get("ligand_feature_dim", 16))),
            protein_feature_dim=int(args.get("protein_feature_dim", ckpt.get("protein_feature_dim", 32))),
            hidden_dim=int(args.get("hidden_dim", ckpt.get("hidden_dim", 128))),
            num_layers=int(args.get("num_layers", ckpt.get("num_layers", 4))),
            dropout=float(args.get("dropout", ckpt.get("dropout", 0.0))),
            scalar_dim=int(args.get("scalar_dim", ckpt.get("scalar_dim", 4))),
            num_r_gaussian=int(args.get("num_r_gaussian", ckpt.get("num_r_gaussian", 20))),
            k=int(args.get("k", ckpt.get("k", 24))),
            cutoff=float(args.get("cutoff", ckpt.get("cutoff", 8.0))),
            cutoff_mode=str(args.get("cutoff_mode", ckpt.get("cutoff_mode", "hybrid"))),
        ).to(device)
    else:
        model = HJBActorModel(
            ligand_feature_dim=int(args.get("ligand_feature_dim", ckpt.get("ligand_feature_dim", 16))),
            hidden_dim=int(args.get("hidden_dim", ckpt.get("hidden_dim", 128))),
            num_layers=int(args.get("num_layers", ckpt.get("num_layers", 3))),
            dropout=float(args.get("dropout", ckpt.get("dropout", 0.0))),
            scalar_dim=int(args.get("scalar_dim", ckpt.get("scalar_dim", 4))),
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

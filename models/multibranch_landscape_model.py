from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hjb_value_model import MLP, _pad_or_trim


class InvariantDistributionalLandscape(nn.Module):
    """SE(3)-invariant scalar landscape with mean and uncertainty heads.

    Coordinates enter only through protein--ligand and ligand--ligand distances.
    Consequently the scalar is invariant to a joint rigid transformation and its
    coordinate gradient is equivariant.  The two outputs parameterize the mean
    cost and the log standard deviation of future continuation cost.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 27,
        hidden_dim: int = 128,
        rbf_dim: int = 24,
        cutoff: float = 7.0,
        cutoff_temperature: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.rbf_dim = int(rbf_dim)
        self.cutoff = float(cutoff)
        self.cutoff_temperature = float(cutoff_temperature)

        self.ligand_encoder = MLP(
            self.ligand_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        self.protein_encoder = MLP(
            self.protein_feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        cross_dim = 2 * hidden_dim + self.rbf_dim + 4
        ligand_pair_dim = 2 * hidden_dim + self.rbf_dim + 4
        self.cross_energy = MLP(cross_dim, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.ligand_pair_energy = MLP(
            ligand_pair_dim, hidden_dim, hidden_dim, num_layers=3, dropout=dropout
        )
        self.output_head = MLP(
            3 * hidden_dim + 5,
            hidden_dim,
            2,
            num_layers=3,
            dropout=dropout,
        )
        centers = torch.linspace(0.0, self.cutoff, self.rbf_dim)
        self.register_buffer("rbf_centers", centers, persistent=False)
        spacing = self.cutoff / max(self.rbf_dim - 1, 1)
        self.rbf_gamma = 1.0 / max(spacing * spacing, 1e-6)

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def _protein_features(self, protein_v: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(protein_v, self.protein_feature_dim)

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.to(distances.device, distances.dtype)
        return torch.exp(-self.rbf_gamma * (distances.unsqueeze(-1) - centers).pow(2))

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ligand_features = self._ligand_features(ligand_v).to(ligand_pos.device, ligand_pos.dtype)
        protein_features = self._protein_features(protein_v).to(protein_pos.device, protein_pos.dtype)
        ligand_nodes = self.ligand_encoder(ligand_features)
        protein_nodes = self.protein_encoder(protein_features)
        num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item())) + 1
        graph_features = []

        for graph_index in range(num_graphs):
            ligand_mask = batch_ligand == graph_index
            protein_mask = batch_protein == graph_index
            lig_pos = ligand_pos[ligand_mask]
            prot_pos = protein_pos[protein_mask]
            lig_node = ligand_nodes[ligand_mask]
            prot_node = protein_nodes[protein_mask]
            n_ligand = max(int(lig_pos.size(0)), 1)

            cross_dist = torch.cdist(lig_pos, prot_pos).clamp_min(1e-6)
            smooth = torch.sigmoid(
                (self.cutoff - cross_dist) / max(self.cutoff_temperature, 1e-6)
            )
            lig_cross = lig_node[:, None, :].expand(-1, prot_node.size(0), -1)
            prot_cross = prot_node[None, :, :].expand(lig_node.size(0), -1, -1)
            cross_extra = torch.stack(
                [
                    cross_dist / self.cutoff,
                    smooth,
                    torch.relu(lig_pos.new_tensor(2.0) - cross_dist).pow(2),
                    torch.exp(-((cross_dist - 3.5) / 1.0).pow(2)),
                ],
                dim=-1,
            )
            cross_input = torch.cat(
                [lig_cross, prot_cross, self._rbf(cross_dist), cross_extra], dim=-1
            )
            cross_value = self.cross_energy(cross_input)
            cross_pool = (cross_value * smooth.unsqueeze(-1)).sum(dim=(0, 1)) / float(n_ligand)

            if int(lig_pos.size(0)) >= 2:
                pair_index = torch.triu_indices(
                    lig_pos.size(0), lig_pos.size(0), offset=1, device=lig_pos.device
                )
                first, second = pair_index[0], pair_index[1]
                pair_dist = (lig_pos[first] - lig_pos[second]).norm(dim=-1).clamp_min(1e-6)
                pair_smooth = torch.sigmoid(
                    (self.cutoff - pair_dist) / max(self.cutoff_temperature, 1e-6)
                )
                pair_extra = torch.stack(
                    [
                        pair_dist / self.cutoff,
                        pair_smooth,
                        torch.relu(lig_pos.new_tensor(1.2) - pair_dist).pow(2),
                        torch.exp(-((pair_dist - 1.5) / 0.5).pow(2)),
                    ],
                    dim=-1,
                )
                pair_input = torch.cat(
                    [
                        lig_node[first],
                        lig_node[second],
                        self._rbf(pair_dist),
                        pair_extra,
                    ],
                    dim=-1,
                )
                pair_value = self.ligand_pair_energy(pair_input)
                pair_pool = (pair_value * pair_smooth.unsqueeze(-1)).sum(dim=0) / float(n_ligand)
            else:
                pair_pool = lig_pos.new_zeros(self.hidden_dim)

            ligand_pool = lig_node.mean(dim=0)
            nearest = cross_dist.min(dim=1).values
            scalar = torch.stack(
                [
                    nearest.min() / self.cutoff,
                    nearest.mean() / self.cutoff,
                    smooth.sum() / float(n_ligand),
                    torch.relu(lig_pos.new_tensor(2.0) - cross_dist).pow(2).sum()
                    / float(n_ligand),
                ]
            )
            graph_features.append(torch.cat([cross_pool, pair_pool, ligand_pool, scalar]))

        graph_tensor = torch.stack(graph_features, dim=0)
        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        output = self.output_head(
            torch.cat(
                [graph_tensor, time_fraction.to(graph_tensor.device, graph_tensor.dtype)], dim=-1
            )
        )
        mean = output[:, 0]
        log_std = output[:, 1].clamp(-4.0, 2.0)
        return mean, log_std

    @staticmethod
    def risk(mean: torch.Tensor, log_std: torch.Tensor, risk_lambda: float = 0.5) -> torch.Tensor:
        return mean + float(risk_lambda) * log_std.exp()

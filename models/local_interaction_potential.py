from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_sum

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class LocalInteractionPotential(nn.Module):
    """Invariant scalar with an atom-resolved pocket--ligand readout.

    PAFlow's mixed EGNN supplies invariant node features.  A learned attention
    distribution then retains the few ligand atoms whose local environment
    changed under a small branch, instead of diluting them in mean pooling.
    The returned scalar remains invariant to joint E(3) transforms, so its
    ligand-coordinate gradient is an equivariant guidance field.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 27,
        hidden_dim: int = 96,
        num_layers: int = 4,
        num_r_gaussian: int = 20,
        k: int = 24,
        cutoff: float = 8.0,
        cutoff_mode: str = "hybrid",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.ligand_projection = nn.Sequential(
            nn.Linear(self.ligand_feature_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_feature_dim + 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.message_passing = EGNN(
            num_layers=int(num_layers), hidden_dim=self.hidden_dim, edge_feat_dim=4,
            num_r_gaussian=int(num_r_gaussian), k=int(k), cutoff=float(cutoff),
            cutoff_mode=str(cutoff_mode), update_x=False, act_fn="silu", norm=False,
        )
        self.local_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.local_energy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 1, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.mix = nn.Parameter(torch.tensor(0.0))

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

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
        dtype, device = ligand_pos.dtype, ligand_pos.device
        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(device=device, dtype=dtype)
        ligand_feature = self._ligand_features(ligand_v).to(device=device, dtype=dtype)
        protein_feature = _pad_or_trim(protein_v, self.protein_feature_dim).to(
            device=device, dtype=dtype
        )
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).expand(ligand_pos.size(0), 2)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).expand(protein_pos.size(0), 2)
        ligand_h = self.ligand_projection(torch.cat([
            ligand_feature, time_fraction.index_select(0, batch_ligand), ligand_role,
        ], dim=-1))
        protein_h = self.protein_projection(torch.cat([
            protein_feature, time_fraction.index_select(0, batch_protein), protein_role,
        ], dim=-1))
        h = self.dropout(torch.cat([ligand_h, protein_h], dim=0))
        x = torch.cat([ligand_pos, protein_pos], dim=0)
        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        mask_ligand = torch.cat([
            torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
            torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
        ])
        output = self.message_passing(
            h, x, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        ligand_output = output[: ligand_pos.size(0)]
        protein_output = output[ligand_pos.size(0):]
        graph_count = int(time_fraction.size(0))
        ligand_pool = scatter_mean(
            ligand_output, batch_ligand, dim=0, dim_size=graph_count
        )
        protein_pool = scatter_mean(
            protein_output, batch_protein, dim=0, dim_size=graph_count
        )
        global_value = self.global_head(torch.cat([
            ligand_pool, protein_pool, time_fraction,
        ], dim=-1)).squeeze(-1)
        attention = softmax(
            self.local_attention(ligand_output).squeeze(-1),
            batch_ligand, num_nodes=graph_count,
        )
        local_value = scatter_sum(
            attention * self.local_energy(ligand_output).squeeze(-1),
            batch_ligand, dim=0, dim_size=graph_count,
        )
        value = global_value + torch.sigmoid(self.mix) * local_value
        return value, torch.zeros_like(value)

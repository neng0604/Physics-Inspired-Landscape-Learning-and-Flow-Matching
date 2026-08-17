from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class EquivariantCoordinatePotential(nn.Module):
    """E(3)-invariant scalar from atomwise equivariant pocket-ligand messages.

    The mixed graph and edge types are identical in spirit to the EGNN path
    supported by PAFlow.  Scalar node features are updated from interatomic
    distances; the final graph scalar is invariant to joint rigid transforms,
    while its derivative with respect to ligand coordinates is equivariant.
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
        self.num_layers = int(num_layers)
        self.num_r_gaussian = int(num_r_gaussian)
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.cutoff_mode = str(cutoff_mode)
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
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            edge_feat_dim=4,
            num_r_gaussian=self.num_r_gaussian,
            k=self.k,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            update_x=False,
            act_fn="silu",
            norm=False,
        )
        self.output_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = ligand_pos.dtype
        device = ligand_pos.device
        ligand_features = self._ligand_features(ligand_v).to(device=device, dtype=dtype)
        protein_features = self._protein_features(protein_v).to(device=device, dtype=dtype)
        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(device=device, dtype=dtype)
        ligand_time = time_fraction.index_select(0, batch_ligand)
        protein_time = time_fraction.index_select(0, batch_protein)
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).view(1, 2).expand(
            ligand_pos.size(0), -1
        )
        protein_role = protein_pos.new_tensor([0.0, 1.0]).view(1, 2).expand(
            protein_pos.size(0), -1
        )
        ligand_h = self.ligand_projection(
            torch.cat([ligand_features, ligand_time, ligand_role], dim=-1)
        )
        protein_h = self.protein_projection(
            torch.cat([protein_features, protein_time, protein_role], dim=-1)
        )
        h = self.dropout(torch.cat([ligand_h, protein_h], dim=0))
        x = torch.cat([ligand_pos, protein_pos], dim=0)
        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        mask_ligand = torch.cat(
            [
                torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
                torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
            ]
        )
        output = self.message_passing(
            h, x, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        ligand_output = output[: ligand_pos.size(0)]
        protein_output = output[ligand_pos.size(0) :]
        number_graphs = int(time_fraction.size(0))
        ligand_pool = scatter_mean(
            ligand_output, batch_ligand, dim=0, dim_size=number_graphs
        )
        protein_pool = scatter_mean(
            protein_output, batch_protein, dim=0, dim_size=number_graphs
        )
        value = self.output_head(
            torch.cat([ligand_pool, protein_pool, time_fraction], dim=-1)
        ).squeeze(-1)
        # Keep the interface shared with the previous two-head critic.  This
        # experiment is deterministic and never trains or uses this zero head.
        return value, torch.zeros_like(value)

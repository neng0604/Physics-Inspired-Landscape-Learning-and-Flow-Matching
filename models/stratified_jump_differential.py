from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class StratifiedJumpDifferential(nn.Module):
    """E(3)-invariant, action-conditioned differential between graph strata.

    A categorical edit is an oriented edge (source, target) of a jump graph.
    Its future advantage is represented by a low-rank tensor contraction
    between global pocket--ligand context, the edited atom's local context,
    and that edge.  Unlike an additive state/action regressor, the ordering of
    edits can therefore change with the molecular state.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 13,
        protein_feature_dim: int = 27,
        hidden_dim: int = 48,
        rank: int = 32,
        num_layers: int = 2,
        num_r_gaussian: int = 20,
        k: int = 16,
        cutoff: float = 8.0,
        cutoff_mode: str = "hybrid",
        scalar_dim: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.scalar_dim = int(scalar_dim)
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
        self.encoder = EGNN(
            num_layers=int(num_layers), hidden_dim=self.hidden_dim,
            edge_feat_dim=4, num_r_gaussian=int(num_r_gaussian), k=int(k),
            cutoff=float(cutoff), cutoff_mode=str(cutoff_mode),
            update_x=False, act_fn="silu", norm=False,
        )
        graph_dim = 2 * self.hidden_dim
        self.global_factor = nn.Sequential(
            nn.LayerNorm(graph_dim), nn.Linear(graph_dim, self.rank), nn.Tanh()
        )
        self.local_factor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.rank), nn.Tanh()
        )
        # A categorical jump is an oriented edge, not a Euclidean difference
        # between two category scores.  The off-diagonal transition kernel can
        # therefore represent non-separable source--target effects.
        self.type_factor = nn.Linear(
            self.ligand_feature_dim * self.ligand_feature_dim, self.rank, bias=False
        )
        self.scalar_factor = nn.Sequential(
            nn.LayerNorm(self.scalar_dim), nn.Linear(self.scalar_dim, self.rank), nn.Tanh()
        )
        self.scalar_residual = nn.Linear(self.scalar_dim, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.readout = nn.Parameter(torch.ones(self.rank) / math.sqrt(self.rank))

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def forward(
        self,
        *,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        decision_time: torch.Tensor,
        action_to_anchor: torch.Tensor,
        action_atom_index: torch.Tensor,
        transition_edge: torch.Tensor,
        scalar_features: torch.Tensor,
        center_by_proposal: bool = True,
    ) -> torch.Tensor:
        dtype, device = ligand_pos.dtype, ligand_pos.device
        decision_time = decision_time.reshape(-1, 1).to(device=device, dtype=dtype)
        ligand_time = decision_time.index_select(0, batch_ligand)
        protein_time = decision_time.index_select(0, batch_protein)
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).view(1, 2).expand(ligand_pos.size(0), -1)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).view(1, 2).expand(protein_pos.size(0), -1)
        ligand_h = self.ligand_projection(torch.cat((
            self._ligand_features(ligand_v).to(device=device, dtype=dtype),
            ligand_time, ligand_role,
        ), dim=-1))
        protein_h = self.protein_projection(torch.cat((
            _pad_or_trim(protein_v, self.protein_feature_dim).to(device=device, dtype=dtype),
            protein_time, protein_role,
        ), dim=-1))
        node_h = torch.cat((ligand_h, protein_h), dim=0)
        coordinates = torch.cat((ligand_pos, protein_pos), dim=0)
        batch = torch.cat((batch_ligand, batch_protein), dim=0)
        mask_ligand = torch.cat((
            torch.ones(ligand_pos.size(0), device=device, dtype=dtype),
            torch.zeros(protein_pos.size(0), device=device, dtype=dtype),
        ))
        encoded = self.encoder(
            node_h, coordinates, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        ligand_encoded = encoded[:ligand_pos.size(0)]
        protein_encoded = encoded[ligand_pos.size(0):]
        graphs = int(decision_time.size(0))
        global_context = torch.cat((
            scatter_mean(ligand_encoded, batch_ligand, dim=0, dim_size=graphs),
            scatter_mean(protein_encoded, batch_protein, dim=0, dim_size=graphs),
        ), dim=-1)
        action_to_anchor = action_to_anchor.long()
        factors = (
            self.global_factor(global_context).index_select(0, action_to_anchor)
            * self.local_factor(ligand_encoded.index_select(0, action_atom_index.long()))
            * self.type_factor(transition_edge.to(device=device, dtype=dtype))
            * self.scalar_factor(scalar_features.to(device=device, dtype=dtype))
        )
        score = (
            (self.dropout(factors) * self.readout).sum(dim=-1)
            + self.scalar_residual(scalar_features.to(device=device, dtype=dtype)).squeeze(-1)
        )
        if center_by_proposal:
            score = score - scatter_mean(
                score, action_to_anchor, dim=0, dim_size=graphs
            ).index_select(0, action_to_anchor)
        return score

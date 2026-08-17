from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class StratifiedStateValue(nn.Module):
    """E(3)-invariant policy value shared across molecular graph strata."""

    def __init__(self, ligand_feature_dim=13, protein_feature_dim=27,
                 hidden_dim=48, num_layers=2, num_r_gaussian=20, k=16,
                 cutoff=8.0, cutoff_mode="hybrid", dropout=0.05):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.ligand_projection = nn.Sequential(
            nn.Linear(self.ligand_feature_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_feature_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder = EGNN(
            num_layers=int(num_layers), hidden_dim=int(hidden_dim), edge_feat_dim=4,
            num_r_gaussian=int(num_r_gaussian), k=int(k), cutoff=float(cutoff),
            cutoff_mode=str(cutoff_mode), update_x=False, act_fn="silu", norm=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(), nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def _ligand_features(self, ligand_v):
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            index = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(index, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def forward(self, *, ligand_pos, ligand_v, protein_pos, protein_v,
                batch_ligand, batch_protein, decision_time):
        dtype, device = ligand_pos.dtype, ligand_pos.device
        decision_time = decision_time.reshape(-1, 1).to(device=device, dtype=dtype)
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).view(1, 2).expand(ligand_pos.size(0), -1)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).view(1, 2).expand(protein_pos.size(0), -1)
        ligand_h = self.ligand_projection(torch.cat((
            self._ligand_features(ligand_v).to(device=device, dtype=dtype),
            decision_time.index_select(0, batch_ligand), ligand_role,
        ), -1))
        protein_h = self.protein_projection(torch.cat((
            _pad_or_trim(protein_v, self.protein_feature_dim).to(device=device, dtype=dtype),
            decision_time.index_select(0, batch_protein), protein_role,
        ), -1))
        coordinates = torch.cat((ligand_pos, protein_pos), 0)
        batch = torch.cat((batch_ligand, batch_protein), 0)
        mask_ligand = torch.cat((
            torch.ones(ligand_pos.size(0), device=device, dtype=dtype),
            torch.zeros(protein_pos.size(0), device=device, dtype=dtype),
        ))
        encoded = self.encoder(
            torch.cat((ligand_h, protein_h), 0), coordinates,
            mask_ligand=mask_ligand, batch=batch, return_all=False,
        )["h"]
        ligand_encoded = encoded[:ligand_pos.size(0)]
        protein_encoded = encoded[ligand_pos.size(0):]
        graphs = int(decision_time.size(0))
        graph = torch.cat((
            scatter_mean(ligand_encoded, batch_ligand, dim=0, dim_size=graphs),
            scatter_mean(protein_encoded, batch_protein, dim=0, dim_size=graphs),
        ), -1)
        return self.head(graph).squeeze(-1)

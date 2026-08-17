from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class ProjectedBranchField(nn.Module):
    """State-only E(3)-equivariant atom-wise field distilled from branch probes."""

    def __init__(self, ligand_feature_dim=16, protein_feature_dim=27, hidden_dim=64,
                 num_layers=3, num_r_gaussian=20, k=16, cutoff=8.0,
                 cutoff_mode="hybrid", dropout=0.05):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_r_gaussian = int(num_r_gaussian)
        if self.num_r_gaussian != 20:
            raise ValueError("The repository EGNN uses the fixed 20-bin Gaussian basis")
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.cutoff_mode = str(cutoff_mode)
        self.dropout_rate = float(dropout)
        self.ligand_projection = nn.Sequential(
            nn.Linear(self.ligand_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.encoder = EGNN(
            num_layers=self.num_layers, hidden_dim=self.hidden_dim, edge_feat_dim=4,
            num_r_gaussian=self.num_r_gaussian, k=self.k, cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode, update_x=True, act_fn="silu", norm=False,
        )
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def _ligand_features(self, value):
        if value.dim() == 1 or (value.dim() == 2 and value.size(-1) == 1):
            ids = value.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(value, self.ligand_feature_dim)

    def forward(self, ligand_pos, ligand_v, protein_pos, protein_v,
                batch_ligand, batch_protein, time_fraction):
        dtype, device = ligand_pos.dtype, ligand_pos.device
        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(device=device, dtype=dtype)
        lt = time_fraction.index_select(0, batch_ligand)
        pt = time_fraction.index_select(0, batch_protein)
        lr = ligand_pos.new_tensor([1.0, 0.0]).view(1, 2).expand(ligand_pos.size(0), -1)
        pr = protein_pos.new_tensor([0.0, 1.0]).view(1, 2).expand(protein_pos.size(0), -1)
        lh = self.ligand_projection(torch.cat([
            self._ligand_features(ligand_v).to(device=device, dtype=dtype), lt, lr,
        ], dim=-1))
        ph = self.protein_projection(torch.cat([
            _pad_or_trim(protein_v, self.protein_feature_dim).to(device=device, dtype=dtype), pt, pr,
        ], dim=-1))
        h = self.dropout(torch.cat([lh, ph], dim=0))
        coordinates = torch.cat([ligand_pos, protein_pos], dim=0)
        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        mask = torch.cat([
            torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
            torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
        ])
        output = self.encoder(h, coordinates, mask_ligand=mask, batch=batch, return_all=False)
        return (output["x"][:ligand_pos.size(0)] - ligand_pos) * self.output_scale

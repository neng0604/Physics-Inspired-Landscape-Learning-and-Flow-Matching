from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class BranchCloudHodgeLandscape(nn.Module):
    """An integrable finite landscape reconstructed from branch comparisons.

    The edge network predicts an antisymmetric comparison field on the complete
    branch-state graph.  Averaging incident comparisons is the closed-form
    least-squares projection onto centered node potentials for a complete
    graph, so the returned scores are mutually consistent rather than
    independent pairwise votes.
    """

    branch_cloud_model = True

    def __init__(self, ligand_feature_dim=39, protein_feature_dim=27,
                 hidden_dim=64, num_layers=3, num_r_gaussian=20, k=20,
                 cutoff=8.0, dropout=0.05, nodes_per_context=15):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.nodes_per_context = int(nodes_per_context)
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
            cutoff_mode="hybrid", update_x=False, act_fn="silu", norm=False,
        )
        graph_dim = 2 * int(hidden_dim)
        edge_dim = 4 * graph_dim + 1
        self.edge_score = nn.Sequential(
            nn.LayerNorm(edge_dim), nn.Linear(edge_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.edge_score[-1].weight, gain=0.05)
        nn.init.zeros_(self.edge_score[-1].bias)

    def _encode(self, ligand_pos, ligand_v, protein_pos, protein_v,
                batch_ligand, batch_protein, decision_time):
        dtype, device = ligand_pos.dtype, ligand_pos.device
        time = decision_time.reshape(-1, 1).to(device=device, dtype=dtype)
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).expand(ligand_pos.size(0), 2)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).expand(protein_pos.size(0), 2)
        ligand_h = self.ligand_projection(torch.cat((
            _pad_or_trim(ligand_v, self.ligand_feature_dim).to(device=device, dtype=dtype),
            time.index_select(0, batch_ligand), ligand_role,
        ), -1))
        protein_h = self.protein_projection(torch.cat((
            _pad_or_trim(protein_v, self.protein_feature_dim).to(device=device, dtype=dtype),
            time.index_select(0, batch_protein), protein_role,
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
        graphs = int(time.size(0))
        return torch.cat((
            scatter_mean(encoded[:ligand_pos.size(0)], batch_ligand, dim=0, dim_size=graphs),
            scatter_mean(encoded[ligand_pos.size(0):], batch_protein, dim=0, dim_size=graphs),
        ), -1), time

    def forward(self, *, ligand_pos, ligand_v, protein_pos, protein_v,
                batch_ligand, batch_protein, decision_time):
        embedding, time = self._encode(
            ligand_pos, ligand_v, protein_pos, protein_v,
            batch_ligand, batch_protein, decision_time,
        )
        if embedding.size(0) % self.nodes_per_context:
            raise ValueError("Branch-cloud graph count is not divisible by cloud size")
        contexts = embedding.size(0) // self.nodes_per_context
        h = embedding.reshape(contexts, self.nodes_per_context, -1)
        left = h[:, :, None, :].expand(-1, -1, self.nodes_per_context, -1)
        right = h[:, None, :, :].expand(-1, self.nodes_per_context, -1, -1)
        delta = left - right
        context_time = time.reshape(contexts, self.nodes_per_context, 1)[:, :1]
        edge_time = context_time[:, :, None].expand(-1, self.nodes_per_context,
                                                    self.nodes_per_context, -1)
        feature = torch.cat((left, right, delta, delta.square(), edge_time), -1)
        directed = self.edge_score(feature).squeeze(-1)
        # r_ij estimates value(i)-value(j).  Antisymmetry removes an arbitrary
        # symmetric edge component; row averaging solves the complete-graph
        # centered least-squares potential reconstruction in closed form.
        comparison = 0.5 * (directed - directed.transpose(1, 2))
        value = comparison.mean(dim=2)
        return value.reshape(-1)

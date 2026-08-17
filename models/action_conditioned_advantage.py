from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class ActionConditionedAdvantage(nn.Module):
    """Invariant future-effect scorer for an explicit equivariant action.

    The action is represented by an inference-available virtual coordinate
    update.  A shared invariant encoder compares the current state, the frozen
    PAFlow next state, and the action-probed state.  The model predicts a direct
    branch advantage; it is not a scalar state potential and is never
    differentiated with respect to coordinates.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        protein_feature_dim: int = 27,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_r_gaussian: int = 20,
        k: int = 16,
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
        self.dropout_rate = float(dropout)

        # One scalar is the decision time.  The two remaining entries encode
        # ligand/protein role without exposing a coordinate frame.
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
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.encoder = EGNN(
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
        graph_dim = 2 * self.hidden_dim
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(3 * graph_dim + 2),
            nn.Linear(3 * graph_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().reshape(-1).clamp(0, self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def _protein_features(self, protein_v: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(protein_v, self.protein_feature_dim)

    def encode(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        decision_time: torch.Tensor,
    ) -> torch.Tensor:
        dtype = ligand_pos.dtype
        device = ligand_pos.device
        if decision_time.dim() == 1:
            decision_time = decision_time.unsqueeze(-1)
        decision_time = decision_time.to(device=device, dtype=dtype)
        ligand_time = decision_time.index_select(0, batch_ligand)
        protein_time = decision_time.index_select(0, batch_protein)
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).view(1, 2).expand(
            ligand_pos.size(0), -1
        )
        protein_role = protein_pos.new_tensor([0.0, 1.0]).view(1, 2).expand(
            protein_pos.size(0), -1
        )
        ligand_h = self.ligand_projection(
            torch.cat(
                [
                    self._ligand_features(ligand_v).to(device=device, dtype=dtype),
                    ligand_time,
                    ligand_role,
                ],
                dim=-1,
            )
        )
        protein_h = self.protein_projection(
            torch.cat(
                [
                    self._protein_features(protein_v).to(device=device, dtype=dtype),
                    protein_time,
                    protein_role,
                ],
                dim=-1,
            )
        )
        h = self.dropout(torch.cat([ligand_h, protein_h], dim=0))
        coordinates = torch.cat([ligand_pos, protein_pos], dim=0)
        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        mask_ligand = torch.cat(
            [
                torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
                torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
            ]
        )
        output = self.encoder(
            h, coordinates, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        ligand_output = output[: ligand_pos.size(0)]
        protein_output = output[ligand_pos.size(0) :]
        graphs = int(decision_time.size(0))
        return torch.cat(
            [
                scatter_mean(ligand_output, batch_ligand, dim=0, dim_size=graphs),
                scatter_mean(protein_output, batch_protein, dim=0, dim_size=graphs),
            ],
            dim=-1,
        )

    def forward(
        self,
        *,
        base_graph: dict[str, torch.Tensor],
        flow_graph: dict[str, torch.Tensor],
        action_graph: dict[str, torch.Tensor],
        action_to_anchor: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        base_embedding = self.encode(**base_graph)
        flow_embedding = self.encode(**flow_graph)
        action_embedding = self.encode(**action_graph)
        action_to_anchor = action_to_anchor.long().to(base_embedding.device)
        selected_base = base_embedding.index_select(0, action_to_anchor)
        selected_flow = flow_embedding.index_select(0, action_to_anchor)
        action_features = torch.cat(
            [
                selected_base,
                action_embedding - selected_base,
                selected_flow - selected_base,
            ],
            dim=-1,
        )
        action_time = base_graph["decision_time"].reshape(-1).index_select(
            0, action_to_anchor
        )
        horizons = horizons.to(device=action_features.device, dtype=action_features.dtype)
        number_actions = int(action_features.size(0))
        number_horizons = int(horizons.numel())
        repeated_features = action_features[:, None, :].expand(
            number_actions, number_horizons, -1
        )
        time_feature = action_time[:, None, None].expand(
            number_actions, number_horizons, 1
        )
        horizon_feature = horizons.view(1, number_horizons, 1).expand(
            number_actions, number_horizons, 1
        )
        head_input = torch.cat(
            [repeated_features, time_feature, horizon_feature], dim=-1
        )
        return self.advantage_head(head_input).squeeze(-1)

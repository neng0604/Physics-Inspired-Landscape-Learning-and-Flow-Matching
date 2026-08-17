from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class StructuredActionResponseOperator(nn.Module):
    """Nonlinear finite-response operator for an explicit equivariant action.

    A shared invariant encoder compares the current state with the virtual
    local geometry ``x + u``.  Subtracting the zero-action head evaluation
    makes the action residual exactly zero at ``u=0`` while retaining both odd
    and even finite-response components.
    """

    direct_action_response = True

    def __init__(
        self,
        outcomes: tuple[str, ...] = ("vina_dock", "qed", "sa", "completion"),
        ligand_feature_dim: int = 13,
        protein_feature_dim: int = 27,
        hidden_dim: int = 96,
        num_layers: int = 3,
        num_r_gaussian: int = 20,
        k: int = 24,
        cutoff: float = 8.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.outcomes = tuple(outcomes)
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.ligand_projection = nn.Sequential(
            nn.Linear(self.ligand_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.encoder = EGNN(
            num_layers=int(num_layers), hidden_dim=self.hidden_dim, edge_feat_dim=4,
            num_r_gaussian=int(num_r_gaussian), k=self.k, cutoff=self.cutoff,
            cutoff_mode="hybrid", update_x=False, act_fn="silu", norm=False,
        )
        graph_dim = 2 * self.hidden_dim
        self.offset = nn.Sequential(
            nn.LayerNorm(graph_dim + 1),
            nn.Linear(graph_dim + 1, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, len(self.outcomes)),
        )
        # The response head is evaluated at the probed and zero-action
        # embeddings; their difference is a nonlinear finite increment.
        self.response = nn.Sequential(
            nn.LayerNorm(3 * graph_dim + 2),
            nn.Linear(3 * graph_dim + 2, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, len(self.outcomes)),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        nn.init.xavier_uniform_(self.response[-1].weight, gain=0.05)
        nn.init.zeros_(self.response[-1].bias)

    def _encode(
        self,
        ligand_pos: torch.Tensor,
        ligand_feature: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_feature: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        dtype, device = ligand_pos.dtype, ligand_pos.device
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).expand(ligand_pos.size(0), 2)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).expand(protein_pos.size(0), 2)
        ligand_h = self.ligand_projection(torch.cat((
            ligand_feature,
            time_fraction.index_select(0, batch_ligand), ligand_role,
        ), dim=-1))
        protein_h = self.protein_projection(torch.cat((
            protein_feature,
            time_fraction.index_select(0, batch_protein), protein_role,
        ), dim=-1))
        h = self.dropout(torch.cat((ligand_h, protein_h), dim=0))
        x = torch.cat((ligand_pos, protein_pos), dim=0)
        batch = torch.cat((batch_ligand, batch_protein), dim=0)
        mask_ligand = torch.cat((
            torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
            torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
        ))
        encoded = self.encoder(
            h, x, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        graphs = int(time_fraction.size(0))
        return torch.cat((
            scatter_mean(
                encoded[: ligand_pos.size(0)], batch_ligand, dim=0, dim_size=graphs
            ),
            scatter_mean(
                encoded[ligand_pos.size(0):], batch_protein, dim=0, dim_size=graphs
            ),
        ), dim=-1)

    def forward_actions(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
        actions: list[torch.Tensor],
    ) -> torch.Tensor:
        dtype, device = ligand_pos.dtype, ligand_pos.device
        if time_fraction.dim() == 1:
            time_fraction = time_fraction.unsqueeze(-1)
        time_fraction = time_fraction.to(device=device, dtype=dtype)
        ligand_feature = _pad_or_trim(
            ligand_logits.to(device=device, dtype=dtype), self.ligand_feature_dim
        )
        protein_feature = _pad_or_trim(
            protein_v.to(device=device, dtype=dtype), self.protein_feature_dim
        )
        number_graphs = int(time_fraction.size(0))
        number_actions = int(actions[0].size(0))
        if any(int(value.size(0)) != number_actions for value in actions):
            raise ValueError("All graphs must expose the same action count")

        repeated_ligand_pos = []
        repeated_ligand_feature = []
        repeated_protein_pos = []
        repeated_protein_feature = []
        repeated_batch_ligand = []
        repeated_batch_protein = []
        action_norm = []
        for graph in range(number_graphs):
            local_ligand = batch_ligand == graph
            local_protein = batch_protein == graph
            coordinate = ligand_pos[local_ligand]
            field = actions[graph].to(device=device, dtype=dtype)
            for action in range(number_actions):
                expanded_graph = graph * number_actions + action
                repeated_ligand_pos.append(coordinate + field[action])
                repeated_ligand_feature.append(ligand_feature[local_ligand])
                repeated_protein_pos.append(protein_pos[local_protein])
                repeated_protein_feature.append(protein_feature[local_protein])
                repeated_batch_ligand.append(torch.full(
                    (int(local_ligand.sum()),), expanded_graph,
                    dtype=torch.long, device=device,
                ))
                repeated_batch_protein.append(torch.full(
                    (int(local_protein.sum()),), expanded_graph,
                    dtype=torch.long, device=device,
                ))
                action_norm.append(torch.sqrt(field[action].square().sum(-1).mean()))
        repeated_time = time_fraction[:, None, :].expand(
            number_graphs, number_actions, 1
        ).reshape(-1, 1)
        embedding = self._encode(
            torch.cat(repeated_ligand_pos), torch.cat(repeated_ligand_feature),
            torch.cat(repeated_protein_pos), torch.cat(repeated_protein_feature),
            torch.cat(repeated_batch_ligand), torch.cat(repeated_batch_protein),
            repeated_time,
        ).reshape(number_graphs, number_actions, -1)
        base = embedding[:, 0]
        delta = embedding - base[:, None, :]
        base_repeated = base[:, None, :].expand_as(embedding)
        norm = torch.stack(action_norm).reshape(number_graphs, number_actions, 1)
        time = time_fraction[:, None, :].expand(number_graphs, number_actions, 1)
        feature = torch.cat((
            base_repeated, delta, delta.square(), norm, time,
        ), dim=-1)
        probed_response = self.response(feature)
        # One shared stochastic forward pass preserves the exact zero-action
        # origin even when dropout is active during fitting.
        residual = probed_response - probed_response[:, :1]
        offset = self.offset(torch.cat((base, time_fraction), dim=-1))
        return offset[:, None, :] + residual

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

from models.common import GaussianSmearing, batch_hybrid_edge_connection
from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class StructuredActionCotangent(nn.Module):
    """Equivariant cotangent paired with an explicit atom-wise action field.

    The state encoder produces invariant node features.  A vector readout
    forms one E(3)-equivariant cotangent field per outcome from relative
    pocket--ligand vectors.  For a structured action ``u``, the response is

        b(s) + sum_i <omega_i(s), u_i>.

    Thus the model sees the actual state-dependent chart frame.  It neither
    differentiates a scalar potential nor predicts an unconstrained action.
    """

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
        self.distance_expansion = GaussianSmearing(
            stop=self.cutoff, num_gaussians=int(num_r_gaussian)
        )
        edge_input = 2 * self.hidden_dim + int(num_r_gaussian) + 4
        self.cotangent_edge = nn.Sequential(
            nn.Linear(edge_input, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, len(self.outcomes), bias=False),
        )
        self.offset = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 1, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, len(self.outcomes)),
        )
        nn.init.xavier_uniform_(self.cotangent_edge[-1].weight, gain=0.01)
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
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
        ligand_feature = _pad_or_trim(
            ligand_logits.to(device=device, dtype=dtype), self.ligand_feature_dim
        )
        protein_feature = _pad_or_trim(
            protein_v.to(device=device, dtype=dtype), self.protein_feature_dim
        )
        ligand_role = ligand_pos.new_tensor([1.0, 0.0]).expand(ligand_pos.size(0), 2)
        protein_role = protein_pos.new_tensor([0.0, 1.0]).expand(protein_pos.size(0), 2)
        ligand_h = self.ligand_projection(torch.cat((
            ligand_feature, time_fraction.index_select(0, batch_ligand), ligand_role,
        ), dim=-1))
        protein_h = self.protein_projection(torch.cat((
            protein_feature, time_fraction.index_select(0, batch_protein), protein_role,
        ), dim=-1))
        h = self.dropout(torch.cat((ligand_h, protein_h), dim=0))
        x = torch.cat((ligand_pos, protein_pos), dim=0)
        batch = torch.cat((batch_ligand, batch_protein), dim=0)
        mask_ligand = torch.cat((
            torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
            torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
        ))
        h = self.encoder(h, x, mask_ligand=mask_ligand, batch=batch, return_all=False)["h"]
        edge_index = batch_hybrid_edge_connection(
            x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True
        )
        src, dst = edge_index
        relative = x[dst] - x[src]
        distance = relative.square().sum(-1, keepdim=True).add(1e-8).sqrt()
        edge_type = torch.zeros(src.numel(), dtype=torch.long, device=device)
        source_ligand = mask_ligand[src] == 1
        target_ligand = mask_ligand[dst] == 1
        edge_type[source_ligand & ~target_ligand] = 1
        edge_type[~source_ligand & target_ligand] = 2
        edge_type[~source_ligand & ~target_ligand] = 3
        edge_feature = torch.cat((
            h[dst], h[src], self.distance_expansion(distance),
            F.one_hot(edge_type, num_classes=4).to(dtype=dtype),
        ), dim=-1)
        weight = self.cotangent_edge(edge_feature)
        vector_message = (
            relative / (distance + 1.0)
        ).unsqueeze(1) * weight.unsqueeze(-1)
        covector = scatter_sum(
            vector_message, dst, dim=0, dim_size=x.size(0)
        )[: ligand_pos.size(0)]
        graphs = int(time_fraction.size(0))
        ligand_pool = scatter_mean(
            h[: ligand_pos.size(0)], batch_ligand, dim=0, dim_size=graphs
        )
        protein_pool = scatter_mean(
            h[ligand_pos.size(0):], batch_protein, dim=0, dim_size=graphs
        )
        offset = self.offset(torch.cat((ligand_pool, protein_pool, time_fraction), dim=-1))
        return offset, covector

    @staticmethod
    def pair_actions(
        offset: torch.Tensor, covector: torch.Tensor, actions: list[torch.Tensor],
        batch_ligand: torch.Tensor,
    ) -> torch.Tensor:
        responses = []
        for graph, action in enumerate(actions):
            local = covector[batch_ligand == graph]
            action = action.to(device=local.device, dtype=local.dtype)
            responses.append(offset[graph, :, None] + torch.einsum(
                "npc,anc->pa", local, action
            ))
        return torch.stack(responses)

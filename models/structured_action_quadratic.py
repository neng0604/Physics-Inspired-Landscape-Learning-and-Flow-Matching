from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

from models.common import GaussianSmearing, batch_hybrid_edge_connection
from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class StructuredActionQuadratic(nn.Module):
    """E(3)-invariant low-rank quadratic response of an explicit action.

    The response is an offset plus a cotangent pairing and a signed low-rank
    quadratic form.  This represents local curvature without constructing a
    dense 3N by 3N Hessian.
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
        quadratic_rank: int = 4,
    ) -> None:
        super().__init__()
        self.outcomes = tuple(outcomes)
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.quadratic_rank = int(quadratic_rank)
        if self.quadratic_rank < 1:
            raise ValueError("quadratic_rank must be positive")
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
        number_fields = len(self.outcomes) * (1 + self.quadratic_rank)
        self.response_edge = nn.Sequential(
            nn.Linear(edge_input, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, number_fields, bias=False),
        )
        graph_input = 2 * self.hidden_dim + 1
        self.offset = nn.Sequential(
            nn.Linear(graph_input, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, len(self.outcomes)),
        )
        self.curvature_eigenvalue = nn.Sequential(
            nn.Linear(graph_input, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, len(self.outcomes) * self.quadratic_rank),
        )
        nn.init.xavier_uniform_(self.response_edge[-1].weight, gain=0.02)
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        nn.init.xavier_uniform_(self.curvature_eigenvalue[-1].weight, gain=0.2)
        nn.init.zeros_(self.curvature_eigenvalue[-1].bias)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        weights = self.response_edge(edge_feature).reshape(
            -1, len(self.outcomes), 1 + self.quadratic_rank
        )
        vector_message = (
            relative / (distance + 1.0)
        )[:, None, None, :] * weights[..., None]
        fields = scatter_sum(
            vector_message, dst, dim=0, dim_size=x.size(0)
        )[: ligand_pos.size(0)]
        linear = fields[:, :, 0]
        modes = fields[:, :, 1:]
        graphs = int(time_fraction.size(0))
        ligand_pool = scatter_mean(
            h[: ligand_pos.size(0)], batch_ligand, dim=0, dim_size=graphs
        )
        protein_pool = scatter_mean(
            h[ligand_pos.size(0):], batch_protein, dim=0, dim_size=graphs
        )
        graph_feature = torch.cat((ligand_pool, protein_pool, time_fraction), dim=-1)
        offset = self.offset(graph_feature)
        eigenvalue = self.curvature_eigenvalue(graph_feature).reshape(
            graphs, len(self.outcomes), self.quadratic_rank
        )
        return offset, linear, modes, eigenvalue

    @staticmethod
    def pair_actions(
        offset: torch.Tensor,
        linear: torch.Tensor,
        modes: torch.Tensor,
        eigenvalue: torch.Tensor,
        actions: list[torch.Tensor],
        batch_ligand: torch.Tensor,
    ) -> torch.Tensor:
        responses = []
        for graph, action in enumerate(actions):
            local_linear = linear[batch_ligand == graph]
            local_modes = modes[batch_ligand == graph]
            action = action.to(device=local_linear.device, dtype=local_linear.dtype)
            first_order = torch.einsum("npc,anc->pa", local_linear, action)
            projection = torch.einsum("nprc,anc->pra", local_modes, action)
            second_order = 0.5 * (
                eigenvalue[graph, :, :, None] * projection.square()
            ).sum(dim=1)
            responses.append(offset[graph, :, None] + first_order + second_order)
        return torch.stack(responses)

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.egnn import EGNN
from models.hjb_actor_model import _pad_or_trim


class BranchResponseCovector(nn.Module):
    """E(3)-invariant local response coefficients in an equivariant chart.

    For every requested outcome, the network returns an isotropic fixed-radius
    intervention term and two invariant coefficients in the semantic
    affinity/steric chart.  If ``d`` is a unit chart coefficient, the predicted
    response is ``b + <g,d>``.  The graph never predicts an unconstrained 3N
    vector and never differentiates a scalar value with respect to coordinates.
    """

    def __init__(
        self,
        outcomes: tuple[str, ...] = ("vina_dock", "qed", "sa", "completion"),
        ligand_feature_dim: int = 13,
        protein_feature_dim: int = 27,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_r_gaussian: int = 20,
        k: int = 24,
        cutoff: float = 8.0,
        dropout: float = 0.05,
        coefficient_dim: int = 3,
    ) -> None:
        super().__init__()
        self.outcomes = tuple(outcomes)
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.protein_feature_dim = int(protein_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.coefficient_dim = int(coefficient_dim)
        self.ligand_projection = nn.Sequential(
            nn.Linear(self.ligand_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_feature_dim + 3, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.message_passing = EGNN(
            num_layers=int(num_layers), hidden_dim=self.hidden_dim, edge_feat_dim=4,
            num_r_gaussian=int(num_r_gaussian), k=int(k), cutoff=float(cutoff),
            cutoff_mode="hybrid", update_x=False, act_fn="silu", norm=False,
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 1, self.hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.coefficient_dim * len(self.outcomes)),
        )
        # Start from an identity controller.  Training must earn every response
        # coefficient from branch supervision.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode_state(
            ligand_pos, ligand_logits, protein_pos, protein_v,
            batch_ligand, batch_protein, time_fraction,
        )
        return self.readout(state).reshape(
            state.size(0), len(self.outcomes), self.coefficient_dim
        )

    def encode_state(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        """Return a rigid-motion invariant pocket--ligand state embedding."""
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
        ligand_h = self.ligand_projection(torch.cat([
            ligand_feature, time_fraction.index_select(0, batch_ligand), ligand_role,
        ], dim=-1))
        protein_h = self.protein_projection(torch.cat([
            protein_feature, time_fraction.index_select(0, batch_protein), protein_role,
        ], dim=-1))
        h = self.dropout(torch.cat((ligand_h, protein_h), dim=0))
        x = torch.cat((ligand_pos, protein_pos), dim=0)
        batch = torch.cat((batch_ligand, batch_protein), dim=0)
        mask_ligand = torch.cat((
            torch.ones(ligand_pos.size(0), dtype=dtype, device=device),
            torch.zeros(protein_pos.size(0), dtype=dtype, device=device),
        ))
        node = self.message_passing(
            h, x, mask_ligand=mask_ligand, batch=batch, return_all=False
        )["h"]
        count = int(time_fraction.size(0))
        ligand_pool = scatter_mean(
            node[: ligand_pos.size(0)], batch_ligand, dim=0, dim_size=count
        )
        protein_pool = scatter_mean(
            node[ligand_pos.size(0):], batch_protein, dim=0, dim_size=count
        )
        return torch.cat((ligand_pool, protein_pool, time_fraction), dim=-1)

    @staticmethod
    def action_response(coefficients: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate ``b + <g,d>`` for one or more two-dimensional actions."""
        if coefficients.dim() != 3 or action.dim() != 2:
            raise ValueError("Expected coefficients [batch,outcome,3] and actions [action,2]")
        return coefficients[:, :, :1] + torch.einsum(
            "bpc,ac->bpa", coefficients[:, :, 1:], action
        )


class BranchResponseGeometry(BranchResponseCovector):
    """Gradient--Hessian response geometry in a two-dimensional action chart.

    Coefficients are ordered as ``(g1, g2, h11, h12, h22)`` and define
    ``g^T d - 1/2 d^T H d``.  The zero action therefore has exactly zero
    response, while the learned symmetric Hessian represents angular
    curvature at the reference intervention radius.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["coefficient_dim"] = 5
        super().__init__(*args, **kwargs)

    @staticmethod
    def action_response(coefficients: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if coefficients.dim() != 3 or coefficients.size(-1) != 5 or action.dim() != 2:
            raise ValueError("Expected coefficients [batch,outcome,5] and actions [action,2]")
        first, second = action[:, 0], action[:, 1]
        design = torch.stack((
            first, second, -0.5 * first.square(),
            -first * second, -0.5 * second.square(),
        ), dim=-1)
        return torch.einsum("bpc,ac->bpa", coefficients, design)


class HybridChartJetResponseGeometry(BranchResponseCovector):
    """Second-order future response in a 3-D coordinate--categorical chart.

    The shared E(3)-invariant encoder is evaluated at the center and at small
    forward/reverse virtual perturbations of two equivariant coordinate axes
    and one categorical-logit tangent axis.  Center, odd, and even embedding
    differences form a chart-aligned local jet.  These evaluations never
    advance the frozen flow and therefore are not terminal rollouts.

    Output coefficients are ordered as
    ``(g1,g2,g3,q11,q12,q13,q22,q23,q33)`` and parameterize
    ``g^T a + 1/2 a^T Q a``.
    """

    coefficient_dim = 9

    def __init__(self, *args, num_classes: int = 13, **kwargs) -> None:
        jet_dropout = float(kwargs.get("dropout", 0.05))
        self.num_classes = int(num_classes)
        kwargs["ligand_feature_dim"] = 2 * self.num_classes
        kwargs["coefficient_dim"] = self.coefficient_dim
        super().__init__(*args, **kwargs)
        # All virtual states must see the same deterministic encoder. Independent
        # dropout masks would contaminate the odd/even finite differences.
        self.dropout = nn.Identity()
        state_dim = 2 * self.hidden_dim + 1
        self.readout = nn.Sequential(
            nn.Linear(7 * state_dim, 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(jet_dropout) if jet_dropout > 0 else nn.Identity(),
            nn.Linear(
                2 * self.hidden_dim,
                self.coefficient_dim * len(self.outcomes),
            ),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def encode_hybrid_state(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        categorical_reference: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        # Categorical logits have an additive gauge. Encoding probabilities
        # makes the state representation invariant to per-atom logit shifts.
        probability = torch.softmax(ligand_logits, dim=-1)
        if categorical_reference.dim() == 1:
            reference = torch.nn.functional.one_hot(
                categorical_reference.long(), self.num_classes
            ).to(dtype=probability.dtype)
        elif categorical_reference.shape == probability.shape:
            reference = categorical_reference.to(dtype=probability.dtype)
        else:
            raise ValueError(
                "categorical_reference must contain hard types or one-hot features"
            )
        feature = torch.cat((probability, reference), dim=-1)
        return BranchResponseCovector.encode_state(
            self, ligand_pos, feature, protein_pos, protein_v,
            batch_ligand, batch_protein, time_fraction,
        )

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_logits: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
        categorical_reference: torch.Tensor,
        coordinate_basis: torch.Tensor,
        categorical_basis: torch.Tensor,
        probe_radius: float | torch.Tensor = 0.05,
    ) -> torch.Tensor:
        if coordinate_basis.shape != (ligand_pos.size(0), 2, 3):
            raise ValueError(
                "coordinate_basis must have shape [num_ligand_atoms,2,3]"
            )
        if categorical_basis.shape != ligand_logits.shape:
            raise ValueError("categorical_basis must match ligand_logits")
        if categorical_reference.size(0) != ligand_pos.size(0):
            raise ValueError("categorical_reference must have one row per ligand atom")
        count = int(time_fraction.numel())
        if isinstance(probe_radius, torch.Tensor):
            radius = probe_radius.to(device=ligand_pos.device, dtype=ligand_pos.dtype).reshape(-1)
            if radius.numel() == 1:
                radius = radius.expand(count)
            if radius.numel() != count:
                raise ValueError("probe_radius must be scalar or contain one value per graph")
        else:
            radius = ligand_pos.new_full((count,), float(probe_radius))
        if bool((radius <= 0).any()):
            raise ValueError("probe_radius must be positive")
        atom_radius = radius.index_select(0, batch_ligand).unsqueeze(-1)

        def encode(position: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            return self.encode_hybrid_state(
                position, logits, categorical_reference, protein_pos, protein_v,
                batch_ligand, batch_protein, time_fraction,
            )

        center = encode(ligand_pos, ligand_logits)
        odd, even = [], []
        for axis in range(3):
            if axis < 2:
                displacement = atom_radius * coordinate_basis[:, axis]
                forward = encode(ligand_pos + displacement, ligand_logits)
                reverse = encode(ligand_pos - displacement, ligand_logits)
            else:
                displacement = atom_radius * categorical_basis
                forward = encode(ligand_pos, ligand_logits + displacement)
                reverse = encode(ligand_pos, ligand_logits - displacement)
            graph_radius = radius.unsqueeze(-1)
            odd.append((forward - reverse) / (2.0 * graph_radius))
            even.append((forward + reverse - 2.0 * center) / graph_radius.square())
        jet = torch.cat((center, *odd, *even), dim=-1)
        return self.readout(jet).reshape(
            count, len(self.outcomes), self.coefficient_dim
        )

    @staticmethod
    def action_response(coefficients: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if coefficients.dim() != 3 or coefficients.size(-1) != 9:
            raise ValueError("Expected coefficients [batch,outcome,9]")
        if action.dim() != 2 or action.size(-1) != 3:
            raise ValueError("Expected action [num_actions,3]")
        first, second, third = action.unbind(dim=-1)
        design = torch.stack((
            first,
            second,
            third,
            0.5 * first.square(),
            first * second,
            first * third,
            0.5 * second.square(),
            second * third,
            0.5 * third.square(),
        ), dim=-1)
        return torch.einsum("bpc,ac->bpa", coefficients, design)

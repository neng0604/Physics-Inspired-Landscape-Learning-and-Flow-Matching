from __future__ import annotations

import math

import torch

from models.intended_action_response import intended_action_features


COEFFICIENTS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (math.sqrt(0.5), math.sqrt(0.5)),
    (math.sqrt(0.5), -math.sqrt(0.5)),
)

GEOMETRY_PROBE_SPECS = (
    {"name": "near_geometry", "ll_center": 1.45, "ll_width": 0.35,
     "pl_center": 2.25, "pl_width": 0.45, "ll_weight": 1.0, "pl_weight": 0.5},
    {"name": "wide_geometry", "ll_center": 2.10, "ll_width": 0.55,
     "pl_center": 3.20, "pl_width": 0.70, "ll_weight": 0.5, "pl_weight": 1.0},
)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.reshape(-1).norm().clamp_min(1e-12)


def remove_flow_parallel(value: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    flat_flow = flow.reshape(-1)
    coefficient = torch.dot(value.reshape(-1), flat_flow) / flat_flow.square().sum().clamp_min(1e-12)
    return value - coefficient * flow


def rigidity_metric_tangent(
    affinity: torch.Tensor,
    flow: torch.Tensor,
    ligand_pos: torch.Tensor,
    strength: float = 4.0,
    bond_center: float = 1.50,
    bond_width: float = 0.32,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the flow-transverse natural gradient of an affinity field.

    The local metric penalizes the first-order change of likely covalent
    distances.  Its null space retains rigid translation and rotation, while
    soft constraints still permit torsional and other non-rigid changes.  The
    constrained solve is the closed-form solution of

        max_d <affinity, d> - 0.5 <d, H d>  subject to <flow, d> = 0,

    with ``H = I + strength * R.T R`` and a smooth distance-weighted rigidity
    Jacobian ``R``.
    """
    atoms = int(ligand_pos.size(0))
    dimensions = 3 * atoms
    if atoms < 2:
        value = remove_flow_parallel(affinity, flow)
        return _normalize(value), {"pairs": 0.0, "strain_ratio": 0.0}

    rows = []
    weights = []
    for first in range(atoms):
        for second in range(first + 1, atoms):
            delta = ligand_pos[first] - ligand_pos[second]
            distance = delta.norm().clamp_min(1e-6)
            weight = torch.exp(
                -0.5 * ((distance - float(bond_center)) / float(bond_width)) ** 2
            )
            # Vanishingly small weights add numerical work but no geometry.
            if float(weight.detach().cpu()) < 1e-4:
                continue
            row = ligand_pos.new_zeros(dimensions)
            unit = delta / distance
            row[3 * first:3 * first + 3] = unit
            row[3 * second:3 * second + 3] = -unit
            rows.append(row)
            weights.append(weight)

    gradient = affinity.reshape(-1)
    base_flow = flow.reshape(-1)
    identity = torch.eye(dimensions, dtype=ligand_pos.dtype, device=ligand_pos.device)
    if rows:
        rigidity = torch.stack(rows)
        root_weight = torch.stack(weights).sqrt().unsqueeze(1)
        weighted = root_weight * rigidity
        gram = weighted.T @ weighted
        # Normalize the non-Euclidean term so ``strength`` is comparable over
        # different ligand sizes and branch states.
        scale = torch.trace(gram) / max(dimensions, 1)
        metric = identity + float(strength) * gram / scale.clamp_min(1e-8)
    else:
        weighted = ligand_pos.new_zeros((0, dimensions))
        metric = identity

    natural_gradient = torch.linalg.solve(metric, gradient)
    natural_flow = torch.linalg.solve(metric, base_flow)
    denominator = torch.dot(base_flow, natural_flow).clamp_min(1e-12)
    direction = natural_gradient - natural_flow * (
        torch.dot(base_flow, natural_gradient) / denominator
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    euclidean = remove_flow_parallel(affinity, flow).reshape(-1)
    euclidean = euclidean / euclidean.norm().clamp_min(1e-12)
    strain_metric = lambda value: (weighted @ value).norm() if rows else value.new_zeros(())
    before = strain_metric(euclidean)
    after = strain_metric(direction)
    diagnostics = {
        "pairs": float(len(rows)),
        "metric_strength": float(strength),
        "euclidean_strain": float(before.detach().cpu()),
        "natural_strain": float(after.detach().cpu()),
        "strain_ratio": float((after / before.clamp_min(1e-12)).detach().cpu()),
        "affinity_alignment": float(torch.dot(gradient, direction).detach().cpu()),
        "flow_inner_product": float(torch.dot(base_flow, direction).detach().cpu()),
    }
    return direction.reshape_as(affinity), diagnostics


def equivariant_geometry_probe(
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    spec: dict,
) -> torch.Tensor:
    ll_delta = ligand_pos[:, None, :] - ligand_pos[None, :, :]
    ll_dist = ll_delta.square().sum(dim=-1).sqrt()
    ll_mask = ~torch.eye(ligand_pos.size(0), dtype=torch.bool, device=ligand_pos.device)
    ll_weight = torch.exp(
        -0.5 * ((ll_dist - float(spec["ll_center"])) / float(spec["ll_width"])) ** 2
    ) * ll_mask
    ll_field = (
        ll_weight.unsqueeze(-1) * ll_delta / ll_dist.clamp_min(1e-6).unsqueeze(-1)
    ).sum(dim=1)

    pl_delta = ligand_pos[:, None, :] - protein_pos[None, :, :]
    pl_dist = pl_delta.square().sum(dim=-1).sqrt()
    pl_weight = torch.exp(
        -0.5 * ((pl_dist - float(spec["pl_center"])) / float(spec["pl_width"])) ** 2
    )
    pl_field = (
        pl_weight.unsqueeze(-1) * pl_delta / pl_dist.clamp_min(1e-6).unsqueeze(-1)
    ).sum(dim=1)
    value = float(spec["ll_weight"]) * ll_field + float(spec["pl_weight"]) * pl_field
    return value - value.mean(dim=0, keepdim=True)


def _construct_basis(
    affinity: torch.Tensor,
    steric: torch.Tensor,
    flow: torch.Tensor,
    fallback_fields: tuple[tuple[str, torch.Tensor], ...] = (),
) -> tuple[torch.Tensor, tuple[str, ...], dict[str, float]]:
    """Construct two route-changing directions, using state geometry if a gradient vanishes."""
    candidates = (("affinity", affinity),) + fallback_fields
    basis, sources = [], []
    residual_norms = {}
    for name, candidate in candidates:
        residual = remove_flow_parallel(candidate, flow)
        for previous in basis:
            residual = residual - torch.dot(residual.reshape(-1), previous.reshape(-1)) * previous
        residual = remove_flow_parallel(residual, flow)
        norm = residual.reshape(-1).norm()
        residual_norms[f"first:{name}"] = float(norm.detach().cpu())
        if float(norm.detach().cpu()) > 1e-8:
            basis.append(residual / norm)
            sources.append(name)
            break
    if not basis:
        empty = torch.empty(0, *affinity.shape, dtype=affinity.dtype, device=affinity.device)
        return empty, tuple(sources), residual_norms

    candidates = (("steric", steric),) + fallback_fields
    for name, candidate in candidates:
        residual = remove_flow_parallel(candidate, flow)
        for previous in basis:
            residual = residual - torch.dot(residual.reshape(-1), previous.reshape(-1)) * previous
        residual = remove_flow_parallel(residual, flow)
        norm = residual.reshape(-1).norm()
        residual_norms[f"second:{name}"] = float(norm.detach().cpu())
        if float(norm.detach().cpu()) > 1e-8:
            basis.append(residual / norm)
            sources.append(name)
            break
    if len(basis) != 2:
        empty = torch.empty(0, *affinity.shape, dtype=affinity.dtype, device=affinity.device)
        return empty, tuple(sources), residual_norms
    return torch.stack(basis), tuple(sources), residual_norms


def transverse_orthonormal_basis(
    affinity: torch.Tensor,
    steric: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    basis, _, _ = _construct_basis(affinity, steric, flow)
    return basis


def state_transverse_orthonormal_basis(
    affinity: torch.Tensor,
    steric: torch.Tensor,
    flow: torch.Tensor,
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...], dict[str, float]]:
    fallbacks = tuple(
        (str(spec["name"]), equivariant_geometry_probe(ligand_pos, protein_pos, spec))
        for spec in GEOMETRY_PROBE_SPECS
    )
    return _construct_basis(affinity, steric, flow, fallbacks)


def transverse_candidates(basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or int(basis.size(0)) != 2:
        return torch.empty(0, *basis.shape[1:], dtype=basis.dtype, device=basis.device)
    values = []
    for first, second in COEFFICIENTS:
        values.append(_normalize(float(first) * basis[0] + float(second) * basis[1]))
    return torch.stack(values)


def transverse_features(
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    action: torch.Tensor,
    basis: torch.Tensor,
    time_fraction: float | torch.Tensor,
    horizon: float | torch.Tensor,
) -> torch.Tensor:
    return intended_action_features(
        ligand_pos, protein_pos, action, basis, time_fraction, horizon
    )


def predict_probability_ensemble(
    checkpoint: dict,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = []
    max_abs_z = features.new_zeros(())
    temperature = float(checkpoint.get("calibration_temperature", 1.0))
    for model in checkpoint["models"]:
        mask = torch.as_tensor(model["feature_mask"], dtype=torch.bool, device=features.device)
        scale = torch.as_tensor(model["feature_scale"], dtype=features.dtype, device=features.device)
        weight = torch.as_tensor(model["weight"], dtype=features.dtype, device=features.device)
        selected = features[mask]
        z = selected / scale.clamp_min(1e-6)
        max_abs_z = torch.maximum(max_abs_z, z.abs().max())
        z = z.clamp(-5.0, 5.0)
        logit = torch.dot(z, weight) / max(temperature, 1e-6)
        probabilities.append(torch.sigmoid(logit))
    stacked = torch.stack(probabilities)
    return stacked.mean(), stacked.std(unbiased=False), max_abs_z

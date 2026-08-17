from __future__ import annotations

import math

import torch


PROPERTIES = ("gnina_affinity", "posecheck_clashes")


def _odd_statistics(value: torch.Tensor) -> torch.Tensor:
    """Odd summaries: every output changes sign when the action is reversed."""
    value = value.reshape(-1)
    if value.numel() == 0:
        return value.new_zeros(4)
    cubic = torch.sign(value.pow(3).mean()) * value.pow(3).abs().mean().pow(1.0 / 3.0)
    fifth = torch.sign(value.pow(5).mean()) * value.pow(5).abs().mean().pow(1.0 / 5.0)
    return torch.stack((value.mean(), cubic, value.max() + value.min(), fifth))


def orthonormalize_fields(fields: list[torch.Tensor]) -> torch.Tensor:
    basis = []
    for field in fields:
        vector = field.reshape(-1).clone()
        for previous in basis:
            vector = vector - torch.dot(vector, previous) * previous
        norm = vector.norm()
        if float(norm.detach().cpu()) <= 1e-8:
            return torch.empty(0, *field.shape, dtype=field.dtype, device=field.device)
        basis.append(vector / norm)
    return torch.stack(basis).reshape(len(basis), *fields[0].shape)


def intended_action_features(
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    action: torch.Tensor,
    basis_fields: torch.Tensor,
    time_fraction: float | torch.Tensor,
    horizon: float | torch.Tensor,
) -> torch.Tensor:
    """Invariant, deployment-available, sign-equivariant action features.

    No post-intervention coordinates or terminal values enter this map.
    """
    dtype, device = ligand_pos.dtype, ligand_pos.device
    action = action.to(device=device, dtype=dtype)
    action = action / torch.sqrt(action.square().sum(-1).mean()).clamp_min(1e-8)
    basis_fields = basis_fields.to(device=device, dtype=dtype)
    semantic = torch.einsum("nd,mnd->m", action, basis_fields)
    semantic = semantic / semantic.norm().clamp_min(1e-8)

    distance = torch.cdist(ligand_pos, protein_pos).clamp_min(1e-5)
    nearest_count = min(3, int(protein_pos.size(0)))
    nearest_distance, nearest_index = torch.topk(distance, k=nearest_count, largest=False)
    source = ligand_pos[:, None, :].expand(-1, nearest_count, -1)
    target = protein_pos.index_select(0, nearest_index.reshape(-1)).reshape_as(source)
    radial = source - target
    radial_unit = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pocket_derivative = (action[:, None, :] * radial_unit).sum(-1)
    pocket_weighted = pocket_derivative * torch.exp(-nearest_distance / 2.0)

    ligand_distance = torch.cdist(ligand_pos, ligand_pos)
    ligand_distance.fill_diagonal_(float("inf"))
    neighbor_count = min(3, max(int(ligand_pos.size(0)) - 1, 1))
    neighbor_index = torch.topk(ligand_distance, k=neighbor_count, largest=False).indices
    ligand_source = torch.arange(ligand_pos.size(0), device=device)[:, None].expand_as(neighbor_index)
    pair_vector = ligand_pos[ligand_source] - ligand_pos[neighbor_index]
    pair_unit = pair_vector / pair_vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pair_derivative = ((action[ligand_source] - action[neighbor_index]) * pair_unit).sum(-1)

    ligand_center = ligand_pos.mean(0)
    protein_center = protein_pos.mean(0)
    center_axis = (ligand_center - protein_center)
    center_axis = center_axis / center_axis.norm().clamp_min(1e-8)
    translation_derivative = torch.dot(action.mean(0), center_axis).reshape(1)
    odd = torch.cat((
        semantic,
        _odd_statistics(pocket_derivative),
        _odd_statistics(pocket_weighted),
        _odd_statistics(pair_derivative),
        translation_derivative,
    ))

    finite_distance = distance.min(dim=1).values
    time_value = torch.as_tensor(time_fraction, dtype=dtype, device=device).reshape(1)
    horizon_value = torch.as_tensor(horizon, dtype=dtype, device=device).reshape(1)
    state = torch.cat((
        torch.ones(1, dtype=dtype, device=device),
        time_value,
        horizon_value / 5.0,
        finite_distance.mean().reshape(1) / 5.0,
        finite_distance.min().reshape(1) / 3.0,
        (finite_distance < 2.0).float().mean().reshape(1),
    ))
    # Tensor product gives state-conditioned response while preserving
    # F(x,-d)=-F(x,d), hence zero response for zero control.
    return torch.einsum("i,j->ij", odd, state).reshape(-1)


def predict_ensemble(checkpoint: dict, prop: str, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    models = checkpoint["models"][prop]
    values = []
    for model in models:
        scale = torch.as_tensor(model["feature_scale"], dtype=features.dtype, device=features.device)
        weight = torch.as_tensor(model["weight"], dtype=features.dtype, device=features.device)
        values.append((features / scale.clamp_min(1e-8)) @ weight)
    stacked = torch.stack(values)
    return stacked.mean(), stacked.std(unbiased=False)

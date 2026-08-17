from __future__ import annotations

import torch

from models.intended_action_response import intended_action_features


def _signed_summaries(plus: torch.Tensor, minus: torch.Tensor) -> torch.Tensor:
    """Antisymmetric summaries of matched quantities under two controls."""
    plus = plus.reshape(-1)
    minus = minus.reshape(-1)
    if plus.numel() == 0:
        return plus.new_zeros(8)
    difference = plus - minus
    return torch.stack(
        (
            difference.mean(),
            difference.abs().mean() * difference.mean().sign(),
            torch.sqrt(difference.square().mean()) * difference.mean().sign(),
            difference.min() + difference.max(),
            plus.mean() - minus.mean(),
            plus.min() - minus.min(),
            torch.quantile(plus, 0.10) - torch.quantile(minus, 0.10),
            plus.max() - minus.max(),
        )
    )


def _upper_distances(position: torch.Tensor) -> torch.Tensor:
    count = int(position.size(0))
    if count < 2:
        return position.new_zeros(0)
    distance = torch.cdist(position, position)
    indices = torch.triu_indices(count, count, offset=1, device=position.device)
    return distance[indices[0], indices[1]]


def transition_pulled_features(
    ligand_pos: torch.Tensor,
    frozen_next_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    action: torch.Tensor,
    basis_fields: torch.Tensor,
    time_fraction: float | torch.Tensor,
    horizon: float | torch.Tensor,
    control_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Deployment-available response features after a frozen-flow proposal.

    ``frozen_next_pos`` is the proposal produced by one frozen PAFlow step
    before accepting a control.  Both candidate states are then constructed
    analytically; no realized branch state or continuation endpoint is read.
    Every appended feature changes sign if ``action`` is reversed.
    """
    dtype, device = ligand_pos.dtype, ligand_pos.device
    frozen_next_pos = frozen_next_pos.to(device=device, dtype=dtype)
    protein_pos = protein_pos.to(device=device, dtype=dtype)
    action = action.to(device=device, dtype=dtype)
    action = action / action.reshape(-1).norm().clamp_min(1e-8)
    scale = torch.as_tensor(control_scale, dtype=dtype, device=device)
    plus = frozen_next_pos + scale * action
    minus = frozen_next_pos - scale * action

    intended = intended_action_features(
        ligand_pos, protein_pos, action, basis_fields, time_fraction, horizon
    )

    plus_nearest = torch.cdist(plus, protein_pos).min(dim=1).values
    minus_nearest = torch.cdist(minus, protein_pos).min(dim=1).values
    pocket = [_signed_summaries(plus_nearest, minus_nearest)]
    for cutoff in (1.6, 1.8, 2.0, 2.2, 2.5):
        plus_overlap = torch.relu(plus_nearest.new_tensor(cutoff) - plus_nearest).square()
        minus_overlap = torch.relu(minus_nearest.new_tensor(cutoff) - minus_nearest).square()
        pocket.append((plus_overlap.mean() - minus_overlap.mean()).reshape(1))
    for width in (1.0, 2.0, 3.0):
        pocket.append(
            (torch.exp(-plus_nearest / width).mean() - torch.exp(-minus_nearest / width).mean()).reshape(1)
        )

    plus_pairs = _upper_distances(plus)
    minus_pairs = _upper_distances(minus)
    ligand = [_signed_summaries(plus_pairs, minus_pairs)]
    for cutoff in (0.7, 0.9, 1.1, 1.3, 1.6, 2.0):
        plus_short = torch.relu(plus_pairs.new_tensor(cutoff) - plus_pairs).square()
        minus_short = torch.relu(minus_pairs.new_tensor(cutoff) - minus_pairs).square()
        ligand.append((plus_short.mean() - minus_short.mean()).reshape(1))

    current_pairs = _upper_distances(ligand_pos)
    if current_pairs.numel():
        bond_mask = (current_pairs >= 0.65) & (current_pairs <= 2.0)
    else:
        bond_mask = torch.zeros_like(current_pairs, dtype=torch.bool)
    if bool(bond_mask.any()):
        reference = current_pairs[bond_mask]
        plus_stretch = (plus_pairs[bond_mask] - reference).abs()
        minus_stretch = (minus_pairs[bond_mask] - reference).abs()
        ligand.append(_signed_summaries(plus_stretch, minus_stretch))
    else:
        ligand.append(ligand_pos.new_zeros(8))

    base_step = frozen_next_pos - ligand_pos
    plus_centered = plus - plus.mean(0, keepdim=True)
    minus_centered = minus - minus.mean(0, keepdim=True)
    transition = torch.stack(
        (
            torch.sum(action * base_step),
            torch.sum((action - action.mean(0, keepdim=True)) * (base_step - base_step.mean(0, keepdim=True))),
            plus_centered.square().sum(-1).mean().sqrt()
            - minus_centered.square().sum(-1).mean().sqrt(),
        )
    )
    return torch.cat((intended, *pocket, *ligand, transition))

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


ACTION_TYPES = ["zero", "grad_e2", "grad_e6", "clash_avoid", "fm_aligned", "contact"]


@dataclass
class BranchActionConfig:
    action_ratio: float = 0.01
    projection_mode: str = "remove_negative_parallel"
    clash_cutoff: float = 2.0
    clash_eps: float = 1e-6
    include_contact_action: bool = False
    contact_distance: float = 3.5
    contact_sigma: float = 0.8
    contact_repulsion_cutoff: float = 2.0
    contact_repulsion_weight: float = 0.5


def atomwise_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).sum(dim=-1).sum().sqrt().clamp_min(eps)


def clip_by_fm_ratio(action: torch.Tensor, v_fm: torch.Tensor, ratio: float, eps: float = 1e-8) -> torch.Tensor:
    """Clip an atom-wise residual velocity to ``ratio * ||v_fm||``."""
    max_norm = float(ratio) * atomwise_norm(v_fm, eps=eps)
    norm = atomwise_norm(action, eps=eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return action * scale


def flow_compatible_projection(action: torch.Tensor, v_fm: torch.Tensor, mode: str = "remove_negative_parallel", eps: float = 1e-8) -> torch.Tensor:
    """Project a residual correction against the base FM direction.

    ``positive_only`` drops the whole correction when it is anti-aligned.
    ``remove_negative_parallel`` removes only the component that points against
    FM, keeping lateral motion and any positive parallel component.
    """
    if mode == "none":
        return action
    dot = (action * v_fm).sum()
    v_norm_sq = v_fm.pow(2).sum().clamp_min(eps)
    if mode == "positive_only":
        return torch.zeros_like(action) if dot < 0 else action
    if mode == "remove_negative_parallel":
        coeff = dot / v_norm_sq
        if coeff >= 0:
            return action
        return action - coeff * v_fm
    raise ValueError(f"Unknown projection mode: {mode}")


def clash_avoid_direction(ligand_pos: torch.Tensor, protein_pos: torch.Tensor, cutoff: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """Simple protein-ligand steric repulsion direction.

    For protein atoms closer than ``cutoff``, push ligand atoms away with a
    ReLU-weighted inverse-distance direction. This is intentionally simple and
    deterministic so it can serve as a branch action, not as a learned force.
    """
    diff = ligand_pos[:, None, :] - protein_pos[None, :, :]
    dist = diff.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
    weight = torch.relu(torch.as_tensor(float(cutoff), dtype=dist.dtype, device=dist.device) - dist)
    direction = diff / dist[..., None]
    repulse = (weight[..., None] * direction).sum(dim=1)
    return repulse


def contact_attraction_direction(
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    distance: float = 3.5,
    sigma: float = 0.8,
    repulsion_cutoff: float = 2.0,
    repulsion_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Simple contact-forming branch direction.

    This is intentionally a proposal action, not a learned chemistry force.  It
    attracts ligand atoms toward nearby protein atoms around a soft contact
    shell, while retaining a short-range repulsion term so the action does not
    simply bury atoms into the pocket.
    """
    diff_lp = protein_pos[None, :, :] - ligand_pos[:, None, :]
    dist = diff_lp.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
    unit_to_protein = diff_lp / dist[..., None]

    contact_center = torch.as_tensor(float(distance), dtype=dist.dtype, device=dist.device)
    contact_sigma = torch.as_tensor(float(max(sigma, eps)), dtype=dist.dtype, device=dist.device)
    attract_weight = torch.exp(-0.5 * ((dist - contact_center) / contact_sigma).pow(2))
    not_too_close = torch.sigmoid((dist - float(repulsion_cutoff)) / 0.25)
    attract = (attract_weight * not_too_close)[..., None] * unit_to_protein

    repel_weight = torch.relu(torch.as_tensor(float(repulsion_cutoff), dtype=dist.dtype, device=dist.device) - dist)
    repel = repel_weight[..., None] * (-unit_to_protein)
    return attract.sum(dim=1) + float(repulsion_weight) * repel.sum(dim=1)


def normalize_to_fm_ratio(direction: torch.Tensor, v_fm: torch.Tensor, ratio: float, eps: float = 1e-8) -> torch.Tensor:
    """Scale a direction to exactly ``ratio * ||v_fm||`` unless it is zero."""
    d_norm = atomwise_norm(direction, eps=eps)
    if float(d_norm.detach().cpu()) <= eps:
        return torch.zeros_like(direction)
    return direction / d_norm * (float(ratio) * atomwise_norm(v_fm, eps=eps))


def make_branch_actions(
    *,
    ligand_pos: torch.Tensor,
    protein_pos: torch.Tensor,
    v_fm: torch.Tensor,
    grad_e2: Optional[torch.Tensor] = None,
    grad_e6: Optional[torch.Tensor] = None,
    config: BranchActionConfig,
) -> Dict[str, torch.Tensor]:
    """Create the fixed branch action set for one state.

    Returned tensors are residual velocity corrections with the same shape as
    ``ligand_pos``. They are already projected and clipped/scaled to the
    configured trust region.
    """
    actions: Dict[str, torch.Tensor] = {}
    actions["zero"] = torch.zeros_like(ligand_pos)

    for name, grad in (("grad_e2", grad_e2), ("grad_e6", grad_e6)):
        if grad is None:
            raw = torch.zeros_like(ligand_pos)
        else:
            raw = -grad
        proj = flow_compatible_projection(raw, v_fm, mode=config.projection_mode)
        actions[name] = clip_by_fm_ratio(proj, v_fm, config.action_ratio)

    clash = clash_avoid_direction(ligand_pos, protein_pos, cutoff=config.clash_cutoff, eps=config.clash_eps)
    clash = flow_compatible_projection(clash, v_fm, mode=config.projection_mode)
    actions["clash_avoid"] = clip_by_fm_ratio(clash, v_fm, config.action_ratio)

    actions["fm_aligned"] = normalize_to_fm_ratio(v_fm, v_fm, config.action_ratio)
    if config.include_contact_action:
        contact = contact_attraction_direction(
            ligand_pos,
            protein_pos,
            distance=config.contact_distance,
            sigma=config.contact_sigma,
            repulsion_cutoff=config.contact_repulsion_cutoff,
            repulsion_weight=config.contact_repulsion_weight,
            eps=config.clash_eps,
        )
        contact = flow_compatible_projection(contact, v_fm, mode=config.projection_mode)
        actions["contact"] = clip_by_fm_ratio(contact, v_fm, config.action_ratio)
    else:
        actions["contact"] = torch.zeros_like(ligand_pos)
    return actions

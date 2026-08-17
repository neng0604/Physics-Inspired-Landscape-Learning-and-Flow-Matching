from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hjb_value_model import (
    HJBValueModel,
    PairwiseHJBValueModel,
    PhysicalPairwiseHJBValueModel,
    TriangleAwareHJBValueModel,
    select_hjb_value,
)


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = rankdata(x[mask])
    yr = rankdata(y[mask])
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = math.sqrt(float((xr * xr).sum() * (yr * yr).sum()))
    return float((xr * yr).sum() / denom) if denom > 1e-12 else float("nan")


class HJBBankDataset(Dataset):
    def __init__(
        self,
        bank,
        indices,
        target_key="G_t",
        target_keys=None,
        hjb_u_key="U_t",
        boundary_key="",
        branch_soft_action_key="",
        terminal_only=False,
    ):
        self.traj = bank["trajectories"]
        self.target_key = target_key
        self.target_keys = list(target_keys or [])
        self.hjb_u_key = str(hjb_u_key)
        self.boundary_key = str(boundary_key or "")
        self.branch_soft_action_key = str(branch_soft_action_key or "")
        self.terminal_only = bool(terminal_only)
        self.items = []
        for traj_idx in indices:
            if bool(self.traj[traj_idx].get("boundary_only", False)) and not self.terminal_only:
                continue
            T = int(self.traj[traj_idx]["ligand_pos_traj"].size(0))
            time_indices = [T - 1] if self.terminal_only and T > 0 else range(T)
            for t in time_indices:
                if self.target_keys:
                    ok = True
                    for key in self.target_keys:
                        target = self.traj[traj_idx].get(key)
                        if target is None or not math.isfinite(float(target[t])):
                            ok = False
                            break
                    if ok:
                        self.items.append((traj_idx, t))
                else:
                    target = self.traj[traj_idx].get(target_key)
                    if target is None:
                        continue
                    value = float(target[t])
                    if math.isfinite(value):
                        self.items.append((traj_idx, t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        traj_idx, t = self.items[idx]
        tr = self.traj[traj_idx]
        T = int(tr["ligand_pos_traj"].size(0))
        pos_traj = tr["ligand_pos_traj"].float()
        time_series = tr.get("time_fraction_traj")
        if time_series is not None:
            time_series = torch.as_tensor(time_series, dtype=torch.float32).reshape(-1)
            if int(time_series.numel()) != T:
                raise ValueError(f"Trajectory {traj_idx} has mismatched time_fraction_traj length.")
            time_fraction = float(time_series[t])
            if T <= 1:
                dt = 1.0
            elif t + 1 < T:
                dt = max(float(time_series[t + 1] - time_series[t]), 1e-8)
            else:
                dt = max(float(time_series[t] - time_series[t - 1]), 1e-8)
        else:
            dt = 1.0 / max(T - 1, 1)
            time_fraction = float(t / max(T - 1, 1))
        if T <= 1:
            v_fm = torch.zeros_like(pos_traj[t])
        elif t + 1 < T:
            v_fm = (pos_traj[t + 1] - pos_traj[t]) / dt
        else:
            v_fm = (pos_traj[t] - pos_traj[t - 1]) / dt
        if self.target_keys:
            primary_target = tr[self.target_keys[0]]
        else:
            primary_target = tr[self.target_key]
        next_t = min(t + 1, T - 1)
        boundary_value = float(primary_target[T - 1])
        if self.boundary_key:
            raw_boundary = tr.get(self.boundary_key)
            if raw_boundary is not None:
                if torch.is_tensor(raw_boundary):
                    boundary_value = float(raw_boundary.reshape(-1)[-1])
                else:
                    boundary_value = float(raw_boundary)
        u_series = tr.get(self.hjb_u_key)
        if u_series is None:
            u_series = tr.get("U_t")
        if u_series is None:
            raise KeyError(f"Trajectory {traj_idx} has neither {self.hjb_u_key!r} nor 'U_t'.")
        item = {
            "traj_idx": traj_idx,
            "t_idx": t,
            "group_id": int(tr.get("result_index", traj_idx)),
            "time_fraction": time_fraction,
            "protein_pos": tr["protein_pos"].float(),
            "protein_v": tr["protein_v"].float(),
            "ligand_pos": pos_traj[t],
            "ligand_v": tr["ligand_v_traj"][t].long(),
            "next_ligand_pos": pos_traj[next_t],
            "next_ligand_v": tr["ligand_v_traj"][next_t].long(),
            "v_fm": v_fm,
            "dt": dt,
            "cost_dt": 1.0 / max(T, 1),
            "U_t": float(u_series[t]),
            "G_t": float(primary_target[t]),
            "G_next": float(primary_target[next_t]),
            "is_terminal": float(t == T - 1),
            "boundary_value": boundary_value,
            "boundary_only": float(bool(tr.get("boundary_only", False))),
            "relaxed_good": float(tr.get("relaxed_good", 0.0)),
            "severe_bad": float(tr.get("severe_bad", 0.0)),
        }
        if self.target_keys:
            item["targets"] = torch.tensor([float(tr[key][t]) for key in self.target_keys], dtype=torch.float32)
        if "branch_actions" in tr and "branch_Y" in tr:
            item["branch_actions"] = tr["branch_actions"][t].float()
            item["branch_Y"] = tr["branch_Y"][t].float()
            item["branch_best_action"] = int(tr.get("branch_best_action", torch.full((T,), -1))[t])
            if "branch_weights" in tr:
                item["branch_weights"] = tr["branch_weights"][t].float()
        if "branch_actions_multi" in tr and "branch_Y_multi" in tr:
            item["branch_actions_multi"] = tr["branch_actions_multi"][t].float()
            item["branch_Y_multi"] = tr["branch_Y_multi"][t].float()
            if "branch_direction_scale_multi" in tr:
                item["branch_direction_scale_multi"] = tr["branch_direction_scale_multi"][t].float()
        if self.branch_soft_action_key:
            action = tr.get(self.branch_soft_action_key)
            valid_key = self.branch_soft_action_key.replace("branch_soft_action_", "branch_soft_action_valid_")
            if action is not None:
                item["branch_soft_action"] = torch.as_tensor(action, dtype=torch.float32)
                item["branch_soft_action_valid"] = float(bool(tr.get(valid_key, False)))
        if "contrastive_clean_pos_traj" in tr:
            item["contrastive_clean_pos"] = tr["contrastive_clean_pos_traj"][t].float()
            weight = tr.get("contrastive_weight_traj")
            if weight is None:
                item["contrastive_weight"] = 1.0
            elif torch.is_tensor(weight):
                item["contrastive_weight"] = float(weight.reshape(-1)[t])
            else:
                item["contrastive_weight"] = float(weight)
        return item


def _traj_pocket_key(traj, fallback_idx):
    for key in ("result_index", "dataset_index", "subset_index", "data_id", "pocket_id"):
        if key in traj:
            value = traj[key]
            if torch.is_tensor(value):
                value = value.reshape(-1)[0].item()
            return str(value)
    return str(fallback_idx)


def _traj_gamma_key(traj):
    value = traj.get("gamma", traj.get("guidance_gamma", traj.get("source_gamma", 0.0)))
    if torch.is_tensor(value):
        value = value.reshape(-1)[0].item()
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value_f - round(value_f)) < 1e-6:
        return str(int(round(value_f)))
    return f"{value_f:.6g}"


def split_trajectory_indices(bank, val_fraction, seed, balance_mode="none"):
    n_traj = len(bank["trajectories"])
    order = list(range(n_traj))
    rng = random.Random(int(seed))
    if any("split_bundle_id" in traj for traj in bank["trajectories"]):
        bundles = {}
        for idx, traj in enumerate(bank["trajectories"]):
            bundle = str(traj.get("split_bundle_id", f"unbundled-{idx}"))
            bundles.setdefault(bundle, []).append(idx)
        bundle_order = list(bundles)
        rng.shuffle(bundle_order)
        n_val_bundles = max(1, int(round(len(bundle_order) * float(val_fraction))))
        val_bundles = set(bundle_order[:n_val_bundles])
        train_idx = [idx for bundle in bundle_order if bundle not in val_bundles for idx in bundles[bundle]]
        val_idx = [idx for bundle in bundle_order if bundle in val_bundles for idx in bundles[bundle]]
        return train_idx, val_idx
    if str(balance_mode) == "none":
        rng.shuffle(order)
        n_val = max(1, int(round(n_traj * float(val_fraction))))
        return order[n_val:], order[:n_val]

    if str(balance_mode) == "pocket_holdout":
        pockets = {}
        for idx, traj in enumerate(bank["trajectories"]):
            pockets.setdefault(_traj_pocket_key(traj, idx), []).append(idx)
        pocket_order = list(pockets)
        rng.shuffle(pocket_order)
        n_val_pockets = max(1, int(round(len(pocket_order) * float(val_fraction))))
        val_pockets = set(pocket_order[:n_val_pockets])
        train_idx = [idx for pocket in pocket_order if pocket not in val_pockets for idx in pockets[pocket]]
        val_idx = [idx for pocket in pocket_order if pocket in val_pockets for idx in pockets[pocket]]
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    groups = {}
    for idx, traj in enumerate(bank["trajectories"]):
        key = (_traj_pocket_key(traj, idx), _traj_gamma_key(traj))
        groups.setdefault(key, []).append(idx)

    train_idx, val_idx = [], []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        n_group = len(members)
        n_val = int(round(n_group * float(val_fraction)))
        if n_group > 1:
            n_val = max(1, min(n_group - 1, n_val))
        else:
            n_val = 1
        val_idx.extend(members[:n_val])
        train_idx.extend(members[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def pocket_bootstrap_indices(bank, indices, seed):
    """Resample complete training pockets with replacement.

    Validation indices must be split before this function is called.  Repeating
    an index is intentional: HJBBankDataset then exposes every state from that
    pocket once for each bootstrap draw.
    """
    pockets = {}
    for idx in indices:
        pocket = _traj_pocket_key(bank["trajectories"][idx], idx)
        pockets.setdefault(pocket, []).append(idx)
    pocket_keys = sorted(pockets)
    if not pocket_keys:
        raise ValueError("Cannot pocket-bootstrap an empty training split.")

    rng = random.Random(int(seed))
    selected_pockets = [rng.choice(pocket_keys) for _ in pocket_keys]
    bootstrapped = [idx for pocket in selected_pockets for idx in pockets[pocket]]
    rng.shuffle(bootstrapped)
    multiplicity = {pocket: 0 for pocket in pocket_keys}
    for pocket in selected_pockets:
        multiplicity[pocket] += 1
    metadata = {
        "seed": int(seed),
        "pool_pockets": len(pocket_keys),
        "drawn_pockets": len(selected_pockets),
        "unique_drawn_pockets": sum(count > 0 for count in multiplicity.values()),
        "out_of_bag_pockets": sum(count == 0 for count in multiplicity.values()),
        "original_train_trajectories": len(indices),
        "bootstrapped_train_trajectories": len(bootstrapped),
        "pocket_multiplicity": multiplicity,
    }
    return bootstrapped, metadata


def balance_validation_items(dataset, bank, max_items, seed, time_bins=5):
    if int(max_items) <= 0 or not dataset.items:
        return
    rng = random.Random(int(seed))
    time_bins = max(1, int(time_bins))
    grouped = {}
    for traj_idx, t_idx in dataset.items:
        traj = bank["trajectories"][traj_idx]
        T = int(traj["ligand_pos_traj"].size(0))
        time_series = traj.get("time_fraction_traj")
        if time_series is not None:
            time_frac = float(torch.as_tensor(time_series).reshape(-1)[t_idx])
        else:
            time_frac = float(t_idx / max(T - 1, 1))
        time_bin = min(time_bins - 1, int(time_frac * time_bins))
        key = (_traj_pocket_key(traj, traj_idx), _traj_gamma_key(traj), time_bin)
        grouped.setdefault(key, []).append((traj_idx, t_idx))
    for items in grouped.values():
        rng.shuffle(items)

    keys = list(grouped.keys())
    rng.shuffle(keys)
    balanced = []
    while keys and len(balanced) < int(max_items):
        next_keys = []
        for key in keys:
            items = grouped[key]
            if items:
                balanced.append(items.pop())
                if len(balanced) >= int(max_items):
                    break
            if items:
                next_keys.append(key)
        keys = next_keys
    dataset.items = balanced


def collate_graphs(batch, device):
    lig_pos, lig_v, next_lig_pos, next_lig_v, lig_vfm, prot_pos, prot_v = [], [], [], [], [], [], []
    batch_lig, batch_prot = [], []
    branch_actions, branch_y, branch_best, branch_weights = [], [], [], []
    c_lig_pos, c_lig_v, c_prot_pos, c_prot_v = [], [], [], []
    c_batch_lig, c_batch_prot, c_time, c_item_idx, c_weight = [], [], [], [], []
    has_targets = all("targets" in item for item in batch)
    has_branch = all("branch_actions" in item and "branch_Y" in item for item in batch)
    has_branch_multi = all("branch_actions_multi" in item and "branch_Y_multi" in item for item in batch)
    has_branch_direction_scale = all("branch_direction_scale_multi" in item for item in batch)
    has_soft_action = all("branch_soft_action" in item for item in batch)
    for i, item in enumerate(batch):
        lp = item["ligand_pos"]
        pp = item["protein_pos"]
        lig_pos.append(lp)
        lig_v.append(item["ligand_v"])
        next_lig_pos.append(item["next_ligand_pos"])
        next_lig_v.append(item["next_ligand_v"])
        lig_vfm.append(item["v_fm"])
        prot_pos.append(pp)
        prot_v.append(item["protein_v"])
        batch_lig.append(torch.full((lp.size(0),), i, dtype=torch.long))
        batch_prot.append(torch.full((pp.size(0),), i, dtype=torch.long))
        if has_branch:
            branch_actions.append(item["branch_actions"].to(device))
            branch_y.append(item["branch_Y"].to(device))
            branch_best.append(int(item.get("branch_best_action", -1)))
            branch_weights.append(item.get("branch_weights"))
        if "contrastive_clean_pos" in item:
            ci = len(c_time)
            cp = item["contrastive_clean_pos"]
            c_lig_pos.append(cp)
            c_lig_v.append(item["ligand_v"])
            c_prot_pos.append(pp)
            c_prot_v.append(item["protein_v"])
            c_batch_lig.append(torch.full((cp.size(0),), ci, dtype=torch.long))
            c_batch_prot.append(torch.full((pp.size(0),), ci, dtype=torch.long))
            c_time.append(float(item["time_fraction"]))
            c_item_idx.append(i)
            c_weight.append(float(item.get("contrastive_weight", 1.0)))
    out = {
        "ligand_pos": torch.cat(lig_pos, dim=0).to(device),
        "ligand_v": torch.cat(lig_v, dim=0).to(device),
        "next_ligand_pos": torch.cat(next_lig_pos, dim=0).to(device),
        "next_ligand_v": torch.cat(next_lig_v, dim=0).to(device),
        "v_fm": torch.cat(lig_vfm, dim=0).to(device),
        "protein_pos": torch.cat(prot_pos, dim=0).to(device),
        "protein_v": torch.cat(prot_v, dim=0).to(device),
        "batch_ligand": torch.cat(batch_lig, dim=0).to(device),
        "batch_protein": torch.cat(batch_prot, dim=0).to(device),
        "time_fraction": torch.tensor([b["time_fraction"] for b in batch], dtype=torch.float32, device=device),
        "dt": torch.tensor([b["dt"] for b in batch], dtype=torch.float32, device=device),
        "cost_dt": torch.tensor([b["cost_dt"] for b in batch], dtype=torch.float32, device=device),
        "group_id": torch.tensor([b.get("group_id", b["traj_idx"]) for b in batch], dtype=torch.long, device=device),
        "U_t": torch.tensor([b["U_t"] for b in batch], dtype=torch.float32, device=device),
        "G_t": torch.tensor([b["G_t"] for b in batch], dtype=torch.float32, device=device),
        "G_next": torch.tensor([b["G_next"] for b in batch], dtype=torch.float32, device=device),
        "is_terminal": torch.tensor([b["is_terminal"] for b in batch], dtype=torch.bool, device=device),
        "boundary_value": torch.tensor([b["boundary_value"] for b in batch], dtype=torch.float32, device=device),
        "boundary_only": torch.tensor([b.get("boundary_only", 0.0) for b in batch], dtype=torch.bool, device=device),
        "relaxed_good": np.asarray([b["relaxed_good"] for b in batch], dtype=float),
        "severe_bad": np.asarray([b["severe_bad"] for b in batch], dtype=float),
    }
    if has_targets:
        out["targets"] = torch.stack([b["targets"] for b in batch], dim=0).to(device)
    if has_branch:
        out["branch_actions"] = branch_actions
        out["branch_Y"] = torch.stack(branch_y, dim=0)
        out["branch_best_action"] = torch.tensor(branch_best, dtype=torch.long, device=device)
        if all(w is not None for w in branch_weights):
            out["branch_weights"] = torch.stack([w.to(device) for w in branch_weights], dim=0)
    if has_branch_multi:
        out["branch_actions_multi"] = [item["branch_actions_multi"].to(device) for item in batch]
        out["branch_Y_multi"] = torch.stack([item["branch_Y_multi"].to(device) for item in batch], dim=0)
        if has_branch_direction_scale:
            out["branch_direction_scale_multi"] = torch.stack(
                [item["branch_direction_scale_multi"].to(device) for item in batch], dim=0
            )
    if has_soft_action:
        out["branch_soft_action"] = torch.cat([b["branch_soft_action"] for b in batch], dim=0).to(device)
        out["branch_soft_action_valid"] = torch.tensor(
            [b.get("branch_soft_action_valid", 0.0) for b in batch], dtype=torch.bool, device=device
        )
    if c_time:
        out["contrastive_ligand_pos"] = torch.cat(c_lig_pos, dim=0).to(device)
        out["contrastive_ligand_v"] = torch.cat(c_lig_v, dim=0).to(device)
        out["contrastive_protein_pos"] = torch.cat(c_prot_pos, dim=0).to(device)
        out["contrastive_protein_v"] = torch.cat(c_prot_v, dim=0).to(device)
        out["contrastive_batch_ligand"] = torch.cat(c_batch_lig, dim=0).to(device)
        out["contrastive_batch_protein"] = torch.cat(c_batch_prot, dim=0).to(device)
        out["contrastive_time_fraction"] = torch.tensor(c_time, dtype=torch.float32, device=device)
        out["contrastive_item_index"] = torch.tensor(c_item_idx, dtype=torch.long, device=device)
        out["contrastive_weight"] = torch.tensor(c_weight, dtype=torch.float32, device=device)
    return out


def per_head_huber_losses(output: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
    if output.dim() != 2 or targets.dim() != 2:
        return [F.huber_loss(output, targets, delta=1.0)]
    n_heads = min(int(output.size(-1)), int(targets.size(-1)))
    return [F.huber_loss(output[:, idx], targets[:, idx], delta=1.0) for idx in range(n_heads)]


def flatten_grads(grads, params):
    flats = []
    for grad, param in zip(grads, params):
        if grad is None:
            flats.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
        else:
            flats.append(grad.detach().reshape(-1))
    return torch.cat(flats) if flats else torch.empty(0)


def pcgrad_merge_losses(losses, params, eps=1e-12):
    """Merge per-task gradients with PCGrad-style conflict removal."""
    if not losses or not params:
        return torch.empty(0), {"mean_cos": float("nan"), "conflict_rate": float("nan")}
    grads = []
    for loss in losses:
        raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        grads.append(flatten_grads(raw, params))
    if len(grads) == 1:
        return grads[0], {"mean_cos": float("nan"), "conflict_rate": 0.0}

    cosines = []
    conflicts = 0
    pairs = 0
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            denom = grads[i].norm() * grads[j].norm()
            if float(denom.item()) > eps:
                cos = float((grads[i] @ grads[j]).item() / max(float(denom.item()), eps))
                cosines.append(cos)
                conflicts += int(cos < 0.0)
                pairs += 1

    adjusted = [g.clone() for g in grads]
    for i in range(len(adjusted)):
        order = list(range(len(grads)))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            denom = grads[j] @ grads[j]
            if float(denom.item()) <= eps:
                continue
            dot = adjusted[i] @ grads[j]
            if float(dot.item()) < 0.0:
                adjusted[i] = adjusted[i] - dot / denom * grads[j]
    merged = torch.stack(adjusted, dim=0).mean(dim=0)
    stats = {
        "mean_cos": float(np.mean(cosines)) if cosines else float("nan"),
        "conflict_rate": float(conflicts / max(pairs, 1)),
    }
    return merged, stats


def add_flat_grad_to_params(params, flat_grad, scale=1.0):
    if flat_grad.numel() == 0:
        return
    offset = 0
    for param in params:
        n = param.numel()
        chunk = flat_grad[offset : offset + n].view_as(param).to(dtype=param.dtype, device=param.device)
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.add_(chunk, alpha=float(scale))
        offset += n


def directional_branch_loss(grad_x, data, margin=0.0, improve_margin=1e-6):
    """Encourage the value gradient to descend along improving branch actions."""
    if "branch_actions" not in data or "branch_Y" not in data:
        return grad_x.new_tensor(0.0)
    losses = []
    for gi, actions in enumerate(data["branch_actions"]):
        mask = data["batch_ligand"] == gi
        if not bool(mask.any()):
            continue
        grad_g = grad_x[mask]
        y = data["branch_Y"][gi]
        y0 = y[0]
        if not torch.isfinite(y0):
            continue
        for ki in range(1, y.numel()):
            if not torch.isfinite(y[ki]) or not (y[ki] + float(improve_margin) < y0):
                continue
            u = actions[ki]
            denom = u.pow(2).sum(dim=-1).sum().sqrt().clamp_min(1e-8)
            if float(denom.detach().cpu()) <= 1e-7:
                continue
            direction = u / denom
            deriv = (grad_g * direction).sum(dim=-1).mean()
            losses.append(torch.relu(torch.as_tensor(float(margin), dtype=deriv.dtype, device=deriv.device) + deriv))
    return torch.stack(losses).mean() if losses else grad_x.new_tensor(0.0)


def branch_alignment_terms(grad_x, data, improve_margin=1e-6, weighted=False):
    """Return graph-level alignment terms for improving branch actions.

    ``directional_branch_loss`` only asks the directional derivative to be
    negative.  These terms measure the stronger property needed by sampling:
    whether ``-grad S`` points in the same coordinate direction as an observed
    improving branch action.
    """

    if "branch_actions" not in data or "branch_Y" not in data:
        return [], [], []
    cos_terms = []
    deriv_terms = []
    weights = []
    for gi, actions in enumerate(data["branch_actions"]):
        mask = data["batch_ligand"] == gi
        if not bool(mask.any()):
            continue
        neg_grad_g = -grad_x[mask]
        grad_g = grad_x[mask]
        y = data["branch_Y"][gi]
        y0 = y[0]
        if not torch.isfinite(y0):
            continue
        for ki in range(1, y.numel()):
            if not torch.isfinite(y[ki]) or not (y[ki] + float(improve_margin) < y0):
                continue
            u = actions[ki]
            u_norm = u.pow(2).sum().sqrt().clamp_min(1e-8)
            g_norm = neg_grad_g.pow(2).sum().sqrt().clamp_min(1e-8)
            if float(u_norm.detach().cpu()) <= 1e-7 or float(g_norm.detach().cpu()) <= 1e-7:
                continue
            cos_terms.append((neg_grad_g * u).sum() / (g_norm * u_norm))
            deriv_terms.append((grad_g * (u / u_norm)).sum())
            if weighted and "branch_weights" in data:
                w = data["branch_weights"][gi, ki]
            else:
                w = (y0 - y[ki]).clamp_min(0.0)
            weights.append(w.detach().clamp_min(0.0))
    return cos_terms, deriv_terms, weights


def branch_cosine_loss(grad_x, data, margin=0.1, improve_margin=1e-6, weighted=False):
    """Encourage ``-grad S`` to align with improving branch actions."""

    cos_terms, _, weights = branch_alignment_terms(
        grad_x,
        data,
        improve_margin=improve_margin,
        weighted=weighted,
    )
    if not cos_terms:
        return grad_x.new_tensor(0.0)
    cos = torch.stack(cos_terms)
    raw = torch.relu(torch.as_tensor(float(margin), dtype=cos.dtype, device=cos.device) - cos)
    if not weighted:
        return raw.mean()
    w = torch.stack(weights).to(dtype=raw.dtype, device=raw.device)
    w = w / w.mean().clamp_min(1e-8)
    return (raw * w).mean()


def branch_pairwise_directional_loss(
    grad_x,
    data,
    margin=0.05,
    improve_margin=1e-6,
    weighted=False,
    max_pairs_per_graph=64,
):
    """Rank branch actions by the local directional derivative of ``S``.

    If branch rollout ``i`` has lower future cost than ``j``, the executable
    value direction should prefer ``i``:

        (-grad S) dot u_i > (-grad S) dot u_j.

    The zero/base branch is kept with score 0, so an improving nonzero branch is
    also trained to beat doing nothing at the anchor state.
    """

    if "branch_actions" not in data or "branch_Y" not in data:
        return grad_x.new_tensor(0.0)
    losses = []
    for gi, actions in enumerate(data["branch_actions"]):
        mask = data["batch_ligand"] == gi
        if not bool(mask.any()):
            continue
        neg_grad_g = -grad_x[mask]
        y = data["branch_Y"][gi]
        scores, costs = [], []
        for ki in range(int(y.numel())):
            if not torch.isfinite(y[ki]):
                continue
            u = actions[ki]
            u_norm = u.pow(2).sum().sqrt()
            if float(u_norm.detach().cpu()) <= 1e-8:
                score = neg_grad_g.new_tensor(0.0)
            else:
                score = (neg_grad_g * (u / u_norm.clamp_min(1e-8))).sum()
            scores.append(score)
            costs.append(y[ki])
        if len(scores) < 2:
            continue
        score_t = torch.stack(scores)
        cost_t = torch.stack(costs).detach()
        better = cost_t[:, None] + float(improve_margin) < cost_t[None, :]
        pairs = torch.nonzero(better, as_tuple=False)
        if pairs.numel() == 0:
            continue
        if pairs.size(0) > int(max_pairs_per_graph):
            perm = torch.randperm(pairs.size(0), device=pairs.device)[: int(max_pairs_per_graph)]
            pairs = pairs[perm]
        i = pairs[:, 0]
        j = pairs[:, 1]
        raw = torch.relu(float(margin) - (score_t[i] - score_t[j]))
        if weighted:
            w = (cost_t[j] - cost_t[i]).clamp_min(0.0)
            w = w / w.mean().clamp_min(1e-8)
            raw = raw * w
        losses.append(raw.mean())
    return torch.stack(losses).mean() if losses else grad_x.new_tensor(0.0)


def branch_pairwise_directional_metrics(grad_x, data, improve_margin=1e-6):
    """Compute branch action ranking diagnostics for ``-grad S``."""

    if "branch_actions" not in data or "branch_Y" not in data:
        return [], [], []
    pair_accs = []
    top1_hits = []
    ranked_counts = []
    for gi, actions in enumerate(data["branch_actions"]):
        mask = data["batch_ligand"] == gi
        if not bool(mask.any()):
            continue
        neg_grad_g = -grad_x[mask].detach()
        y = data["branch_Y"][gi].detach()
        scores, costs = [], []
        for ki in range(int(y.numel())):
            if not torch.isfinite(y[ki]):
                continue
            u = actions[ki].detach()
            u_norm = u.pow(2).sum().sqrt()
            if float(u_norm.cpu()) <= 1e-8:
                score = 0.0
            else:
                score = float((neg_grad_g.cpu() * (u.cpu() / u_norm.cpu().clamp_min(1e-8))).sum())
            scores.append(score)
            costs.append(float(y[ki].cpu()))
        if len(scores) < 2:
            continue
        correct = 0
        total = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if costs[i] + float(improve_margin) < costs[j]:
                    total += 1
                    correct += int(scores[i] > scores[j])
                elif costs[j] + float(improve_margin) < costs[i]:
                    total += 1
                    correct += int(scores[j] > scores[i])
        if total > 0:
            pair_accs.append(float(correct / total))
        pred_best = int(np.argmax(scores))
        true_best = int(np.argmin(costs))
        top1_hits.append(float(pred_best == true_best))
        ranked_counts.append(float(len(scores)))
    return pair_accs, top1_hits, ranked_counts


def branch_counterfactual_loss(
    model,
    data,
    component="total",
    head_names=None,
    lookahead=1,
    max_nonbase_branches=1,
    loss_type="huber",
):
    """Match finite value differences to observed branch-outcome differences.

    ``branch_actions[g, k]`` is the coordinate displacement from an anchor to
    the lookahead state of branch ``k``.  Branch zero is the base continuation.
    The loss asks the learned potential difference between branch endpoints to
    reproduce the corresponding empirical future-cost difference.
    """

    if "branch_actions" not in data or "branch_Y" not in data:
        zero = data["ligand_pos"].sum() * 0.0
        return zero, zero.detach(), zero.detach(), 0

    endpoint_pos, endpoint_lig_v = [], []
    endpoint_prot_pos, endpoint_prot_v = [], []
    endpoint_batch_lig, endpoint_batch_prot, endpoint_time = [], [], []
    groups = []
    endpoint_graph = 0
    for gi, actions in enumerate(data["branch_actions"]):
        y = data["branch_Y"][gi]
        if y.numel() < 2 or not torch.isfinite(y[0]):
            continue
        candidates = torch.where(torch.isfinite(y[1:]))[0] + 1
        if candidates.numel() == 0:
            continue
        if int(max_nonbase_branches) > 0 and candidates.numel() > int(max_nonbase_branches):
            perm = torch.randperm(candidates.numel(), device=candidates.device)[: int(max_nonbase_branches)]
            candidates = candidates.index_select(0, perm)
        selected = torch.cat([torch.zeros(1, dtype=torch.long, device=y.device), candidates])

        lig_mask = data["batch_ligand"] == gi
        prot_mask = data["batch_protein"] == gi
        if not bool(lig_mask.any()) or not bool(prot_mask.any()):
            continue
        anchor_pos = data["ligand_pos"][lig_mask].detach()
        ligand_v = data["ligand_v"][lig_mask]
        protein_pos = data["protein_pos"][prot_mask]
        protein_v = data["protein_v"][prot_mask]
        next_time = (
            data["time_fraction"][gi].detach()
            + float(max(int(lookahead), 1)) * data["dt"][gi].detach()
        ).clamp(max=1.0)

        start = endpoint_graph
        for branch_idx in selected.tolist():
            pos = anchor_pos + actions[int(branch_idx)].to(anchor_pos)
            endpoint_pos.append(pos)
            endpoint_lig_v.append(ligand_v)
            endpoint_prot_pos.append(protein_pos)
            endpoint_prot_v.append(protein_v)
            endpoint_batch_lig.append(torch.full((pos.size(0),), endpoint_graph, dtype=torch.long, device=pos.device))
            endpoint_batch_prot.append(
                torch.full((protein_pos.size(0),), endpoint_graph, dtype=torch.long, device=protein_pos.device)
            )
            endpoint_time.append(next_time)
            endpoint_graph += 1
        groups.append((start, endpoint_graph, y.index_select(0, selected).detach()))

    if not groups:
        zero = data["ligand_pos"].sum() * 0.0
        return zero, zero.detach(), zero.detach(), 0

    out = model(
        torch.cat(endpoint_pos, dim=0),
        torch.cat(endpoint_lig_v, dim=0),
        torch.cat(endpoint_prot_pos, dim=0),
        torch.cat(endpoint_prot_v, dim=0),
        torch.cat(endpoint_batch_lig, dim=0),
        torch.cat(endpoint_batch_prot, dim=0),
        torch.stack(endpoint_time),
    )
    endpoint_value = select_hjb_value(out, component, head_names)
    losses, abs_errors, correct = [], [], []
    pair_count = 0
    for start, end, y_selected in groups:
        pred = endpoint_value[start:end]
        pred_delta = pred[1:] - pred[0]
        target_delta = y_selected[1:] - y_selected[0]
        losses.append(squared_or_huber_loss(pred_delta, target_delta, loss_type, reduction="none"))
        abs_errors.append((pred_delta.detach() - target_delta).abs())
        non_tied = target_delta.abs() > 1e-6
        if bool(non_tied.any()):
            correct.append(((pred_delta.detach()[non_tied] * target_delta[non_tied]) > 0).float())
        pair_count += int(target_delta.numel())

    loss = torch.cat(losses).mean()
    mae = torch.cat(abs_errors).mean() if abs_errors else loss.detach().new_tensor(float("nan"))
    sign_acc = torch.cat(correct).mean() if correct else loss.detach().new_tensor(float("nan"))
    return loss, mae, sign_acc, pair_count


def branch_counterfactual_directional_loss(
    model,
    data,
    component="total",
    head_names=None,
    max_nonbase_branches=1,
    create_graph=True,
    loss_type="huber",
    normalize_derivative=False,
):
    """Match anchor directional derivatives to relative branch outcomes.

    For branch ``k`` and the matched base continuation ``0``, the first-order
    target is

        grad S(z) dot (delta_x_k - delta_x_0) ~= Y_k - Y_0.

    Unlike endpoint-value matching, this directly supervises the coordinate
    gradient that is converted into the sampling residual.
    """

    has_single = "branch_actions" in data and "branch_Y" in data
    has_multi = "branch_actions_multi" in data and "branch_Y_multi" in data
    if not has_single and not has_multi:
        zero = data["ligand_pos"].sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach(), 0

    pos = data["ligand_pos"].detach().requires_grad_(True)
    t = data["time_fraction"].detach()
    out = model(
        pos,
        data["ligand_v"],
        data["protein_pos"],
        data["protein_v"],
        data["batch_ligand"],
        data["batch_protein"],
        t,
    )
    value = select_hjb_value(out, component, head_names)
    grad_x = torch.autograd.grad(
        value.sum(),
        pos,
        create_graph=bool(create_graph),
        retain_graph=bool(create_graph),
    )[0]

    losses, abs_errors, correct, aligned_cosines = [], [], [], []
    pair_count = 0
    num_graphs = int(value.numel())
    for gi in range(num_graphs):
        atom_mask = data["batch_ligand"] == gi
        if not bool(atom_mask.any()):
            continue
        grad_g = grad_x[atom_mask]
        if has_multi:
            action_sets = data["branch_actions_multi"][gi]
            y_sets = data["branch_Y_multi"][gi]
            scale_sets = data.get("branch_direction_scale_multi")
            scale_sets = scale_sets[gi] if scale_sets is not None else None
        else:
            action_sets = data["branch_actions"][gi].unsqueeze(0)
            y_sets = data["branch_Y"][gi].unsqueeze(0)
            scale_sets = None

        for pair_idx, (actions, y) in enumerate(zip(action_sets, y_sets)):
            if y.numel() < 2 or not torch.isfinite(y[0]):
                continue
            candidates = torch.where(torch.isfinite(y[1:]))[0] + 1
            if candidates.numel() == 0:
                continue
            if int(max_nonbase_branches) > 0 and candidates.numel() > int(max_nonbase_branches):
                perm = torch.randperm(candidates.numel(), device=candidates.device)[: int(max_nonbase_branches)]
                candidates = candidates.index_select(0, perm)

            base_action = actions[0]
            for branch_idx in candidates.tolist():
                relative_action = actions[int(branch_idx)] - base_action
                action_norm = relative_action.pow(2).sum().sqrt()
                if float(action_norm.detach().cpu()) <= 1e-8:
                    continue
                target_delta = (y[int(branch_idx)] - y[0]).detach()
                predicted_delta = (grad_g * relative_action).sum()
                if bool(normalize_derivative):
                    derivative_denom = action_norm
                    if scale_sets is not None:
                        derivative_denom = derivative_denom * scale_sets[pair_idx].clamp_min(1e-8)
                    predicted_delta = predicted_delta / derivative_denom
                    target_delta = target_delta / derivative_denom
                losses.append(squared_or_huber_loss(predicted_delta, target_delta, loss_type, reduction="none"))
                abs_errors.append((predicted_delta.detach() - target_delta).abs())
                if float(target_delta.abs().cpu()) > 1e-6:
                    correct.append(((predicted_delta.detach() * target_delta) > 0).float())
                    desired_sign = -torch.sign(target_delta)
                    grad_norm = grad_g.detach().pow(2).sum().sqrt().clamp_min(1e-8)
                    cosine = ((-grad_g.detach()) * relative_action).sum() / (grad_norm * action_norm.clamp_min(1e-8))
                    aligned_cosines.append(cosine * desired_sign)
                pair_count += 1

    if not losses:
        zero = value.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach(), 0
    loss = torch.stack(losses).mean()
    mae = torch.stack(abs_errors).mean()
    sign_acc = torch.stack(correct).mean() if correct else loss.detach().new_tensor(float("nan"))
    aligned_cos = (
        torch.stack(aligned_cosines).mean() if aligned_cosines else loss.detach().new_tensor(float("nan"))
    )
    return loss, mae, sign_acc, aligned_cos, pair_count


def filter_counterfactual_items(dataset, bank):
    """Keep states with a finite base branch and at least one comparison branch."""

    filtered = []
    for traj_idx, time_idx in dataset.items:
        trajectory = bank["trajectories"][traj_idx]
        branch_y = trajectory.get("branch_Y_multi", trajectory.get("branch_Y"))
        if branch_y is None or time_idx >= int(branch_y.size(0)):
            continue
        row = branch_y[time_idx]
        if row.dim() > 1:
            valid = bool(torch.isfinite(row[:, 0]).any()) and bool(torch.isfinite(row[:, 1:]).any())
        else:
            valid = row.numel() >= 2 and bool(torch.isfinite(row[0])) and bool(torch.isfinite(row[1:]).any())
        if valid:
            filtered.append((traj_idx, time_idx))
    dataset.items = filtered
    return dataset


def eval_counterfactual_model(
    model,
    loader,
    device,
    component="total",
    head_names=None,
    lookahead=1,
    max_batches=50,
    loss_type="huber",
):
    if loader is None:
        return {"loss": float("nan"), "mae": float("nan"), "sign_acc": float("nan"), "pairs": 0}
    model.eval()
    losses, maes, accuracies = [], [], []
    pairs = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            data = collate_graphs(batch, device)
            loss, mae, sign_acc, count = branch_counterfactual_loss(
                model,
                data,
                component=component,
                head_names=head_names,
                lookahead=lookahead,
                max_nonbase_branches=0,
                loss_type=loss_type,
            )
            if count > 0:
                losses.append(float(loss.detach().cpu()))
                maes.append(float(mae.detach().cpu()))
                if torch.isfinite(sign_acc):
                    accuracies.append(float(sign_acc.detach().cpu()))
                pairs += int(count)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "sign_acc": float(np.mean(accuracies)) if accuracies else float("nan"),
        "pairs": int(pairs),
    }


def eval_counterfactual_directional_model(
    model,
    loader,
    device,
    component="total",
    head_names=None,
    max_nonbase_branches=0,
    max_batches=50,
    loss_type="huber",
    normalize_derivative=False,
):
    if loader is None:
        return {"loss": float("nan"), "mae": float("nan"), "sign_acc": float("nan"), "aligned_cos": float("nan"), "pairs": 0}
    model.eval()
    losses, maes, accuracies, cosines = [], [], [], []
    pairs = 0
    with torch.enable_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= int(max_batches):
                break
            data = collate_graphs(batch, device)
            loss, mae, sign_acc, aligned_cos, count = branch_counterfactual_directional_loss(
                model,
                data,
                component=component,
                head_names=head_names,
                max_nonbase_branches=max_nonbase_branches,
                create_graph=False,
                loss_type=loss_type,
                normalize_derivative=bool(normalize_derivative),
            )
            if count > 0:
                losses.append(float(loss.detach().cpu()))
                maes.append(float(mae.detach().cpu()))
                if torch.isfinite(sign_acc):
                    accuracies.append(float(sign_acc.detach().cpu()))
                if torch.isfinite(aligned_cos):
                    cosines.append(float(aligned_cos.detach().cpu()))
                pairs += int(count)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "sign_acc": float(np.mean(accuracies)) if accuracies else float("nan"),
        "aligned_cos": float(np.mean(cosines)) if cosines else float("nan"),
        "pairs": int(pairs),
    }


def branch_soft_action_loss(grad_x, data, margin=1.0):
    if "branch_soft_action" not in data:
        return grad_x.new_tensor(0.0), grad_x.new_empty(0)
    num_graphs = int(data["G_t"].numel())
    action = data["branch_soft_action"]
    neg_grad = -grad_x
    dot = graph_mean_dot(neg_grad, action, data["batch_ligand"], num_graphs)
    grad_norm = graph_mean_squared_norm(neg_grad, data["batch_ligand"], num_graphs).sqrt()
    action_norm = graph_mean_squared_norm(action, data["batch_ligand"], num_graphs).sqrt()
    cosine = dot / (grad_norm * action_norm).clamp_min(1e-8)
    valid = data.get("branch_soft_action_valid", torch.ones_like(cosine, dtype=torch.bool))
    valid = valid & (action_norm > 1e-8) & torch.isfinite(cosine)
    if not bool(valid.any()):
        return grad_x.new_tensor(0.0), cosine.detach()
    return (float(margin) - cosine[valid]).mean(), cosine.detach()


def policy_bellman_loss(
    model,
    s,
    data,
    component="total",
    head_names=None,
    policy_rho=0.05,
    cost_scale=1.0,
    control_cost_weight=0.0,
    policy_steps=1,
    loss_type="huber",
):
    """One-step Bellman consistency under the sampling-time residual policy."""

    steps = max(int(policy_steps), 1)
    dt_graph = data["dt"].clamp_min(1e-8)
    dt_atom = dt_graph.index_select(0, data["batch_ligand"]).unsqueeze(-1)
    next_pos = data["ligand_pos"].detach()
    next_t = data["time_fraction"].detach()
    v_fm = data["v_fm"].detach()
    control_cost = s.new_zeros(s.shape)
    for _ in range(steps):
        cur_pos = next_pos.detach().requires_grad_(True)
        cur_t = next_t.detach()
        out_cur = model(
            cur_pos,
            data["ligand_v"],
            data["protein_pos"],
            data["protein_v"],
            data["batch_ligand"],
            data["batch_protein"],
            cur_t,
        )
        s_cur = select_hjb_value(out_cur, component, head_names)
        grad_cur = torch.autograd.grad(s_cur.sum(), cur_pos, create_graph=False, retain_graph=False)[0]
        residual_control = -float(policy_rho) * grad_cur.detach()
        if float(control_cost_weight) > 0:
            control_cost = control_cost + graph_mean_squared_norm(
                residual_control,
                data["batch_ligand"],
                int(s.shape[0]),
            )
        next_pos = cur_pos.detach() + dt_atom * (v_fm + residual_control)
        next_t = (cur_t + dt_graph).clamp(max=1.0)
    out_next = model(
        next_pos.detach(),
        data["ligand_v"],
        data["protein_pos"],
        data["protein_v"],
        data["batch_ligand"],
        data["batch_protein"],
        next_t.detach(),
    )
    s_next = select_hjb_value(out_next, component, head_names)
    target = (
        float(steps) * float(cost_scale) * data["U_t"]
        + float(control_cost_weight) * control_cost.detach()
        + s_next.detach()
    )
    return squared_or_huber_loss(s, target, loss_type)


def flow_bellman_loss(model, s, data, component="total", head_names=None, loss_type="huber"):
    """Exact discrete Bellman consistency on stored hybrid coordinate/type transitions."""

    dt_graph = data["dt"].clamp_min(1e-8)
    next_pos = data["next_ligand_pos"].detach()
    next_t = (data["time_fraction"].detach() + dt_graph).clamp(max=1.0)
    out_next = model(
        next_pos.detach(),
        data["next_ligand_v"],
        data["protein_pos"],
        data["protein_v"],
        data["batch_ligand"],
        data["batch_protein"],
        next_t.detach(),
    )
    s_next = select_hjb_value(out_next, component, head_names)
    target = data["cost_dt"] * data["U_t"] + s_next.detach()
    nonterminal = ~data["is_terminal"]
    if not bool(nonterminal.any()):
        return s.sum() * 0.0
    return squared_or_huber_loss(s[nonterminal], target[nonterminal], loss_type)


def fm_alignment_loss(grad_x, data, margin=0.05, improve_margin=1e-6):
    """Make ``-grad S`` usable by the sampler on states where FM lowers the target.

    The scalar value can rank states while its coordinate gradient remains
    nearly orthogonal to the FM trajectory.  This loss only activates when the
    stored next state has lower target cost, then asks ``-grad S`` to have a
    positive cosine with the finite-difference FM direction.
    """

    num_graphs = int(data["G_t"].numel())
    dot = graph_mean_dot(data["v_fm"], -grad_x, data["batch_ligand"], num_graphs)
    fm_norm = graph_mean_squared_norm(data["v_fm"], data["batch_ligand"], num_graphs).sqrt()
    grad_norm = graph_mean_squared_norm(grad_x, data["batch_ligand"], num_graphs).sqrt()
    cos = dot / (fm_norm * grad_norm).clamp_min(1e-8)
    improves = data["G_next"] + float(improve_margin) < data["G_t"]
    finite = torch.isfinite(cos) & improves
    if not bool(finite.any()):
        return grad_x.new_tensor(0.0), cos.detach()
    loss = torch.relu(torch.as_tensor(float(margin), dtype=cos.dtype, device=cos.device) - cos[finite]).mean()
    return loss, cos.detach()


def graph_mean_squared_norm(vec, batch_ligand, num_graphs):
    out = torch.zeros(num_graphs, dtype=vec.dtype, device=vec.device)
    for gi in range(num_graphs):
        mask = batch_ligand == gi
        if bool(mask.any()):
            out[gi] = vec[mask].pow(2).sum(dim=-1).mean()
    return out


def graph_mean_dot(a, b, batch_ligand, num_graphs):
    out = torch.zeros(num_graphs, dtype=a.dtype, device=a.device)
    for gi in range(num_graphs):
        mask = batch_ligand == gi
        if bool(mask.any()):
            out[gi] = (a[mask] * b[mask]).sum(dim=-1).mean()
    return out


def local_geometry_terms(protein_pos, ligand_pos, batch_protein, batch_ligand):
    """Differentiable graph-level local geometry proxies."""
    num_graphs = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
    clash_terms, center_terms, contact_terms = [], [], []
    for gi in range(num_graphs):
        lm = batch_ligand == gi
        pm = batch_protein == gi
        if bool(lm.any()) and bool(pm.any()):
            lig = ligand_pos[lm]
            prot = protein_pos[pm]
            d = torch.cdist(lig, prot)
            clash = torch.relu(torch.as_tensor(1.6, dtype=d.dtype, device=d.device) - d).pow(2).mean()
            contact = torch.exp(-((d - 3.5) / 1.0).pow(2)).mean()
            center = (lig.mean(dim=0) - prot.mean(dim=0)).norm()
        else:
            clash = ligand_pos.new_tensor(0.0)
            contact = ligand_pos.new_tensor(0.0)
            center = ligand_pos.new_tensor(0.0)
        clash_terms.append(clash)
        center_terms.append(center)
        contact_terms.append(contact)
    return torch.stack(clash_terms), torch.stack(center_terms), torch.stack(contact_terms)


def action_geometry_direction_loss(
    grad_s,
    data,
    pos,
    clash_weight=1.0,
    center_weight=0.05,
    contact_weight=0.5,
    margin=0.0,
    mode="raw",
    active_clash_threshold=-1.0,
    eps=1e-8,
):
    """Encourage the sampling direction ``-grad S`` to reduce local geometry cost.

    Let

        C_geo = w_c clash + w_r center_distance - w_i contact.

    A usable value-gradient action should satisfy the local directional
    derivative ``d C_geo(x + eps*(-grad S)) / d eps <= 0``.  This loss uses that
    derivative directly, so the value model is trained to orient its coordinate
    gradient toward lower local geometry cost rather than only toward lower
    predicted value.
    """

    clash, center, contact = local_geometry_terms(
        data["protein_pos"],
        pos,
        data["batch_protein"],
        data["batch_ligand"],
    )
    geom_cost = float(clash_weight) * clash + float(center_weight) * center - float(contact_weight) * contact
    grad_geom = torch.autograd.grad(geom_cost.sum(), pos, create_graph=False, retain_graph=True)[0].detach()
    num_graphs = int(geom_cost.numel())
    direction = -grad_s
    deriv = graph_mean_dot(grad_geom, direction, data["batch_ligand"], num_graphs)
    geom_norm = graph_mean_squared_norm(grad_geom, data["batch_ligand"], num_graphs).sqrt()
    s_norm = graph_mean_squared_norm(direction, data["batch_ligand"], num_graphs).sqrt()
    cos = deriv / (geom_norm * s_norm).clamp_min(float(eps))

    if float(active_clash_threshold) >= 0.0:
        active = clash > float(active_clash_threshold)
    else:
        active = torch.ones_like(clash, dtype=torch.bool)

    if str(mode) == "raw":
        violation = deriv
    elif str(mode) == "cosine":
        violation = cos
    else:
        raise ValueError(f"Unknown action geometry loss mode: {mode}")

    if bool(active.any()):
        loss = torch.relu(violation[active] + float(margin)).mean()
    else:
        loss = direction.sum() * 0.0
    return loss, deriv.detach(), cos.detach(), active.float().detach()


def within_group_rank_loss(pred, target, group_id, margin=0.1, max_pairs_per_group=256):
    losses = []
    for group in torch.unique(group_id):
        idx = torch.where(group_id == group)[0]
        if idx.numel() < 2:
            continue
        p = pred[idx]
        y = target[idx]
        better = y[:, None] < y[None, :]
        pairs = torch.nonzero(better, as_tuple=False)
        if pairs.numel() == 0:
            continue
        if pairs.size(0) > int(max_pairs_per_group):
            perm = torch.randperm(pairs.size(0), device=pairs.device)[: int(max_pairs_per_group)]
            pairs = pairs[perm]
        i = pairs[:, 0]
        j = pairs[:, 1]
        losses.append(torch.relu(float(margin) + p[i] - p[j]).mean())
    return torch.stack(losses).mean() if losses else pred.new_zeros(())


def augmented_clean_bad_rank_loss(model, s, data, component="total", head_names=None, margin=0.25):
    """Rank an augmented bad state above its paired clean parent state."""
    if "contrastive_ligand_pos" not in data:
        return s.new_zeros(())
    out_clean = model(
        data["contrastive_ligand_pos"],
        data["contrastive_ligand_v"],
        data["contrastive_protein_pos"],
        data["contrastive_protein_v"],
        data["contrastive_batch_ligand"],
        data["contrastive_batch_protein"],
        data["contrastive_time_fraction"],
    )
    s_clean = select_hjb_value(out_clean, component, head_names)
    s_bad = s.index_select(0, data["contrastive_item_index"])
    raw = torch.relu(float(margin) + s_clean - s_bad)
    weight = data.get("contrastive_weight")
    if weight is not None:
        weight = weight.to(dtype=raw.dtype, device=raw.device)
        weight = weight / weight.mean().clamp_min(1e-8)
        raw = raw * weight
    return raw.mean()


def eval_model(
    model,
    loader,
    device,
    max_batches=50,
    component="total",
    head_names=None,
    multi_head=False,
    hjb_mode="full",
    dir_improve_margin=1e-6,
):
    model.eval()
    preds, targets, all_preds, all_targets, good, bad = [], [], [], [], [], []
    grad_norms, dot_vfm_grads, hjb_losses, boundary_losses = [], [], [], []
    branch_cosines, branch_derivs, branch_soft_cosines = [], [], []
    branch_pair_accs, branch_top1_hits, branch_ranked_counts = [], [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        data = collate_graphs(batch, device)
        pos = data["ligand_pos"].detach().requires_grad_(True)
        t = data["time_fraction"].detach().requires_grad_(True)
        out = model(pos, data["ligand_v"], data["protein_pos"], data["protein_v"], data["batch_ligand"], data["batch_protein"], t)
        s = select_hjb_value(out, component, head_names)
        grad = torch.autograd.grad(s.sum(), pos, create_graph=False, retain_graph=True)[0]
        ds_dt = torch.autograd.grad(s.sum(), t, create_graph=False, retain_graph=False, allow_unused=True)[0]
        if ds_dt is None:
            ds_dt = torch.zeros_like(s)
        num_graphs = int(s.size(0))
        grad_sq = graph_mean_squared_norm(grad, data["batch_ligand"], num_graphs)
        dot_vfm_grad = graph_mean_dot(data["v_fm"], grad, data["batch_ligand"], num_graphs)
        if str(hjb_mode) == "residual":
            residual = ds_dt + dot_vfm_grad - 0.5 * grad_sq + data["U_t"]
        elif str(hjb_mode) == "fixed_policy":
            residual = ds_dt + dot_vfm_grad + data["U_t"]
        else:
            residual = ds_dt - 0.5 * grad_sq + data["U_t"]
        hjb_losses.extend(residual.detach().pow(2).cpu().numpy().tolist())
        if bool(data["is_terminal"].any()):
            boundary_loss = F.huber_loss(
                s[data["is_terminal"]],
                data["boundary_value"][data["is_terminal"]],
                delta=1.0,
                reduction="none",
            )
            boundary_losses.extend(boundary_loss.detach().cpu().numpy().tolist())
        target_for_s = data["G_t"]
        if multi_head and "targets" in data:
            target_for_s = select_hjb_value(data["targets"], component, head_names)
        preds.extend(s.detach().cpu().numpy().tolist())
        targets.extend(target_for_s.detach().cpu().numpy().tolist())
        if multi_head and "targets" in data:
            all_preds.append(out.detach().cpu().numpy())
            all_targets.append(data["targets"].detach().cpu().numpy())
        good.extend(data["relaxed_good"].tolist())
        bad.extend(data["severe_bad"].tolist())
        gn = torch.zeros_like(s)
        dot = torch.zeros_like(s)
        for gi in range(s.size(0)):
            mask = data["batch_ligand"] == gi
            gn[gi] = grad[mask].pow(2).sum(dim=-1).mean().sqrt()
            dot[gi] = (data["v_fm"][mask] * grad[mask]).sum(dim=-1).mean()
        grad_norms.extend(gn.detach().cpu().numpy().tolist())
        dot_vfm_grads.extend(dot.detach().cpu().numpy().tolist())
        cos_terms, deriv_terms, _ = branch_alignment_terms(grad, data, improve_margin=float(dir_improve_margin))
        if cos_terms:
            branch_cosines.extend(torch.stack(cos_terms).detach().cpu().numpy().tolist())
        if deriv_terms:
            branch_derivs.extend(torch.stack(deriv_terms).detach().cpu().numpy().tolist())
        pair_accs, top1_hits, ranked_counts = branch_pairwise_directional_metrics(
            grad,
            data,
            improve_margin=float(dir_improve_margin),
        )
        branch_pair_accs.extend(pair_accs)
        branch_top1_hits.extend(top1_hits)
        branch_ranked_counts.extend(ranked_counts)
        _, soft_cos = branch_soft_action_loss(grad, data)
        if soft_cos.numel() and "branch_soft_action_valid" in data:
            valid_soft = data["branch_soft_action_valid"] & torch.isfinite(soft_cos)
            branch_soft_cosines.extend(soft_cos[valid_soft].detach().cpu().numpy().tolist())
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    errors = preds - targets
    low = preds <= np.nanquantile(preds, 0.25) if len(preds) else np.asarray([], dtype=bool)
    if len(preds):
        high_pred = preds >= np.nanquantile(preds, 0.75)
        high_target = targets >= np.nanquantile(targets, 0.75)
        high_target_count = int(high_target.sum())
        high_cost_top25_recall = (
            float(np.logical_and(high_pred, high_target).sum() / high_target_count)
            if high_target_count > 0
            else float("nan")
        )
    else:
        high_cost_top25_recall = float("nan")
    result = {
        "spearman_s_G": spearman(preds, targets),
        "spearman_neg_s_neg_G": spearman(-preds, -targets),
        "mae_s_G": float(np.nanmean(np.abs(errors))) if len(errors) else float("nan"),
        "rmse_s_G": float(np.sqrt(np.nanmean(errors**2))) if len(errors) else float("nan"),
        "bias_s_G": float(np.nanmean(errors)) if len(errors) else float("nan"),
        "error_gt_1_rate": float(np.nanmean(np.abs(errors) > 1.0)) if len(errors) else float("nan"),
        "high_cost_top25_recall": high_cost_top25_recall,
        "low_s_relaxed_good": float(np.mean(np.asarray(good)[low])) if low.any() else float("nan"),
        "low_s_severe_bad": float(np.mean(np.asarray(bad)[low])) if low.any() else float("nan"),
        "grad_norm_mean": float(np.nanmean(grad_norms)) if grad_norms else float("nan"),
        "dot_vfm_grad_mean": float(np.nanmean(dot_vfm_grads)) if dot_vfm_grads else float("nan"),
        "hjb_residual_mean": float(np.nanmean(hjb_losses) ** 0.5) if hjb_losses else float("nan"),
        "boundary_loss": float(np.nanmean(boundary_losses)) if boundary_losses else float("nan"),
        "branch_cos_mean": float(np.nanmean(branch_cosines)) if branch_cosines else float("nan"),
        "branch_deriv_mean": float(np.nanmean(branch_derivs)) if branch_derivs else float("nan"),
        "branch_action_n": int(len(branch_cosines)),
        "branch_pair_acc": float(np.nanmean(branch_pair_accs)) if branch_pair_accs else float("nan"),
        "branch_top1_rate": float(np.nanmean(branch_top1_hits)) if branch_top1_hits else float("nan"),
        "branch_ranked_count_mean": float(np.nanmean(branch_ranked_counts)) if branch_ranked_counts else float("nan"),
        "branch_soft_action_cos_mean": float(np.nanmean(branch_soft_cosines)) if branch_soft_cosines else float("nan"),
        "n": int(len(preds)),
    }
    if multi_head and all_preds and all_targets:
        pred_mat = np.concatenate(all_preds, axis=0)
        targ_mat = np.concatenate(all_targets, axis=0)
        for idx, name in enumerate(list(head_names or [])[: pred_mat.shape[1]]):
            result[f"spearman_{name}"] = spearman(pred_mat[:, idx], targ_mat[:, idx])
    return result


def metric_is_better(value, best, mode):
    if not math.isfinite(float(value)):
        return False
    if str(mode) == "min":
        return float(value) < float(best)
    return float(value) > float(best)


def squared_or_huber_loss(pred, target, loss_type="huber", reduction="mean"):
    if str(loss_type) == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    return F.huber_loss(pred, target, delta=1.0, reduction=reduction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Optional compatible checkpoint used only to initialize model weights.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs; zero disables early stopping.",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=0,
        help="Minimum number of epochs completed before early stopping is allowed.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model_arch",
        choices=["pooled", "pairwise", "physical_pairwise", "triangle_aware"],
        default="pooled",
    )
    parser.add_argument("--pair_rbf_dim", type=int, default=24)
    parser.add_argument("--pair_cutoff", type=float, default=6.0)
    parser.add_argument("--pair_cutoff_temperature", type=float, default=0.5)
    parser.add_argument("--triangle_ligand_neighbors", type=int, default=8)
    parser.add_argument("--triangle_pocket_neighbors", type=int, default=6)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Maximum optimization batches per epoch; zero uses the full training set.",
    )
    parser.add_argument(
        "--value_loss_type",
        choices=["huber", "mse"],
        default="huber",
        help="Regression loss for fitting the trajectory cost-to-go target.",
    )
    parser.add_argument("--lambda_value", type=float, default=1.0)
    parser.add_argument("--lambda_hjb", type=float, default=0.0)
    parser.add_argument("--lambda_boundary", type=float, default=0.0)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--lambda_aug_rank", type=float, default=0.0)
    parser.add_argument("--aug_rank_margin", type=float, default=0.25)
    parser.add_argument("--boundary_batch_size", type=int, default=0)
    parser.add_argument("--boundary_every", type=int, default=1)
    parser.add_argument(
        "--hjb_mode",
        choices=["full", "residual", "fixed_policy"],
        default="full",
        help="Dynamic consistency equation. fixed_policy evaluates the frozen base flow without a quadratic control term.",
    )
    parser.add_argument("--target_key", default="G_t", help="Trajectory tensor to use as value target, e.g. G_t or G_tilde.")
    parser.add_argument("--boundary_key", default="", help="Scalar terminal boundary key, e.g. U_d_terminal. Empty uses target_key at final step.")
    parser.add_argument("--multi_head", action="store_true")
    parser.add_argument("--target_keys", default="G_safe,G_dock,G_drug", help="Comma-separated target tensors for multi-head training.")
    parser.add_argument("--head_names", default="safe,dock,drug", help="Comma-separated output head names.")
    parser.add_argument("--hjb_component", default="total", help="Head/component used for HJB and directional losses.")
    parser.add_argument("--hjb_u_key", default="U_t", help="Trajectory tensor used as local U in the HJB residual, e.g. U_t or U_dock.")
    parser.add_argument(
        "--value_grad_strategy",
        choices=["standard", "pcgrad"],
        default="standard",
        help="How to combine multi-head value-regression gradients. PCGrad applies conflict removal to per-head value losses.",
    )
    parser.add_argument("--lambda_dir", type=float, default=0.0)
    parser.add_argument("--dir_margin", type=float, default=0.0)
    parser.add_argument("--dir_improve_margin", type=float, default=1e-6)
    parser.add_argument("--lambda_branch_cos", type=float, default=0.0)
    parser.add_argument("--branch_cos_margin", type=float, default=0.1)
    parser.add_argument("--branch_cos_weighted", action="store_true")
    parser.add_argument("--lambda_branch_pair_rank", type=float, default=0.0)
    parser.add_argument("--branch_pair_margin", type=float, default=0.05)
    parser.add_argument("--branch_pair_weighted", action="store_true")
    parser.add_argument("--branch_pair_max_pairs", type=int, default=64)
    parser.add_argument("--lambda_cf", type=float, default=0.0)
    parser.add_argument("--lambda_cf_directional", type=float, default=0.0)
    parser.add_argument(
        "--cf_directional_normalize",
        action="store_true",
        help="Fit directional derivatives after action-norm and direction/time scale normalization.",
    )
    parser.add_argument(
        "--aux_loss_type",
        choices=["huber", "mse"],
        default="huber",
        help="Regression loss used by Bellman and counterfactual costate objectives.",
    )
    parser.add_argument("--cf_batch_size", type=int, default=4)
    parser.add_argument("--cf_lookahead", type=int, default=0, help="Branch lookahead; zero reads it from bank metadata.")
    parser.add_argument("--cf_max_nonbase_branches", type=int, default=1)
    parser.add_argument("--branch_soft_action_key", default="")
    parser.add_argument("--lambda_branch_soft_action", type=float, default=0.0)
    parser.add_argument("--lambda_policy_bellman", type=float, default=0.0)
    parser.add_argument("--policy_rho", type=float, default=0.05)
    parser.add_argument("--policy_cost_scale", type=float, default=1.0)
    parser.add_argument("--policy_control_cost_weight", type=float, default=0.0)
    parser.add_argument("--policy_steps", type=int, default=1)
    parser.add_argument("--lambda_flow_bellman", type=float, default=0.0)
    parser.add_argument("--lambda_fm_align", type=float, default=0.0)
    parser.add_argument("--fm_align_margin", type=float, default=0.05)
    parser.add_argument("--fm_align_improve_margin", type=float, default=1e-6)
    parser.add_argument("--lambda_action_geom", type=float, default=0.0)
    parser.add_argument("--action_geom_clash_weight", type=float, default=1.0)
    parser.add_argument("--action_geom_center_weight", type=float, default=0.05)
    parser.add_argument("--action_geom_contact_weight", type=float, default=0.5)
    parser.add_argument("--action_geom_margin", type=float, default=0.0)
    parser.add_argument("--action_geom_mode", choices=["raw", "cosine"], default="raw")
    parser.add_argument(
        "--action_geom_active_clash_threshold",
        type=float,
        default=-1.0,
        help="Apply the geometry loss only to graphs above this clash value; a negative value keeps all graphs active.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=-1,
        help="Seed used only for the train/validation split; a negative value uses --seed.",
    )
    parser.add_argument(
        "--pocket_bootstrap_train",
        action="store_true",
        help="After splitting, resample complete training pockets with replacement.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=-1,
        help="Pocket-bootstrap seed; a negative value uses --seed.",
    )
    parser.add_argument("--val_max_batches", type=int, default=50)
    parser.add_argument(
        "--val_balance_mode",
        choices=["none", "pocket_gamma_time", "pocket_holdout"],
        default="none",
        help="Validation split. pocket_holdout keeps complete pockets disjoint; pocket_gamma_time stratifies trajectories within each pocket/gamma.",
    )
    parser.add_argument("--val_time_bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--best_metric", default="val_spearman_s_G")
    parser.add_argument("--best_mode", choices=["max", "min"], default="max")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    cf_lookahead = int(args.cf_lookahead)
    if cf_lookahead <= 0:
        cf_lookahead = int(bank.get("args", {}).get("branch_action_lookahead", 1))
    target_keys = [x.strip() for x in str(args.target_keys).split(",") if x.strip()] if args.multi_head else []
    head_names = [x.strip() for x in str(args.head_names).split(",") if x.strip()] if args.multi_head else ["total"]
    if args.multi_head and len(target_keys) != len(head_names):
        raise ValueError(f"--target_keys and --head_names must have the same length, got {target_keys} vs {head_names}")
    n_traj = len(bank["trajectories"])
    split_seed = int(args.seed) if int(args.split_seed) < 0 else int(args.split_seed)
    train_idx, val_idx = split_trajectory_indices(
        bank,
        args.val_fraction,
        split_seed,
        balance_mode=args.val_balance_mode,
    )
    original_train_idx = list(train_idx)
    bootstrap_metadata = None
    if bool(args.pocket_bootstrap_train):
        bootstrap_seed = int(args.seed) if int(args.bootstrap_seed) < 0 else int(args.bootstrap_seed)
        train_idx, bootstrap_metadata = pocket_bootstrap_indices(bank, train_idx, bootstrap_seed)
    split_manifest = {
        "split_seed": split_seed,
        "val_fraction": float(args.val_fraction),
        "val_balance_mode": str(args.val_balance_mode),
        "original_train_indices": original_train_idx,
        "validation_indices": list(val_idx),
        "pocket_bootstrap_train": bool(args.pocket_bootstrap_train),
        "bootstrap": bootstrap_metadata,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    train_ds = HJBBankDataset(
        bank,
        train_idx,
        target_key=args.target_key,
        target_keys=target_keys,
        hjb_u_key=args.hjb_u_key,
        boundary_key=args.boundary_key,
        branch_soft_action_key=args.branch_soft_action_key,
    )
    terminal_train_ds = HJBBankDataset(
        bank,
        train_idx,
        target_key=args.target_key,
        target_keys=target_keys,
        hjb_u_key=args.hjb_u_key,
        boundary_key=args.boundary_key,
        branch_soft_action_key=args.branch_soft_action_key,
        terminal_only=True,
    )
    val_ds = HJBBankDataset(
        bank,
        val_idx,
        target_key=args.target_key,
        target_keys=target_keys,
        hjb_u_key=args.hjb_u_key,
        boundary_key=args.boundary_key,
        branch_soft_action_key=args.branch_soft_action_key,
    )
    cf_train_ds = filter_counterfactual_items(
        HJBBankDataset(
            bank,
            train_idx,
            target_key=args.target_key,
            target_keys=target_keys,
            hjb_u_key=args.hjb_u_key,
            boundary_key=args.boundary_key,
            branch_soft_action_key=args.branch_soft_action_key,
        ),
        bank,
    )
    cf_val_ds = filter_counterfactual_items(
        HJBBankDataset(
            bank,
            val_idx,
            target_key=args.target_key,
            target_keys=target_keys,
            hjb_u_key=args.hjb_u_key,
            boundary_key=args.boundary_key,
            branch_soft_action_key=args.branch_soft_action_key,
        ),
        bank,
    )
    if str(args.val_balance_mode) == "pocket_gamma_time":
        balance_validation_items(
            val_ds,
            bank,
            max_items=max(1, int(args.val_max_batches)) * max(1, int(args.batch_size)),
            seed=int(args.seed) + 17,
            time_bins=int(args.val_time_bins),
        )
    print(
        json.dumps(
            {
                "event": "dataset_split",
                "n_traj": n_traj,
                "original_train_traj": len(original_train_idx),
                "train_traj": len(train_idx),
                "val_traj": len(val_idx),
                "train_states": len(train_ds),
                "val_states": len(val_ds),
                "cf_train_anchors": len(cf_train_ds),
                "cf_val_anchors": len(cf_val_ds),
                "cf_lookahead": int(cf_lookahead),
                "val_fraction": float(args.val_fraction),
                "split_seed": split_seed,
                "pocket_bootstrap_train": bool(args.pocket_bootstrap_train),
                "bootstrap": bootstrap_metadata,
                "val_balance_mode": str(args.val_balance_mode),
                "val_time_bins": int(args.val_time_bins),
                "val_max_batches": int(args.val_max_batches),
            }
        )
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=0)
    terminal_train_loader = None
    if int(args.boundary_batch_size) > 0 and len(terminal_train_ds) > 0:
        terminal_train_loader = DataLoader(
            terminal_train_ds,
            batch_size=max(1, int(args.boundary_batch_size)),
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x, num_workers=0)
    cf_train_loader = None
    if (float(args.lambda_cf) > 0 or float(args.lambda_cf_directional) > 0) and len(cf_train_ds) > 0:
        cf_train_loader = DataLoader(
            cf_train_ds,
            batch_size=max(1, int(args.cf_batch_size)),
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )
    cf_val_loader = None
    if len(cf_val_ds) > 0:
        cf_val_loader = DataLoader(
            cf_val_ds,
            batch_size=max(1, int(args.cf_batch_size)),
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    model_cls = {
        "pooled": HJBValueModel,
        "pairwise": PairwiseHJBValueModel,
        "physical_pairwise": PhysicalPairwiseHJBValueModel,
        "triangle_aware": TriangleAwareHJBValueModel,
    }[str(args.model_arch)]
    model_kwargs = {}
    if str(args.model_arch) in {"pairwise", "physical_pairwise", "triangle_aware"}:
        model_kwargs.update(
            rbf_dim=int(args.pair_rbf_dim),
            cutoff=float(args.pair_cutoff),
            cutoff_temperature=float(args.pair_cutoff_temperature),
        )
    if str(args.model_arch) == "triangle_aware":
        model_kwargs.update(
            triangle_ligand_neighbors=int(args.triangle_ligand_neighbors),
            triangle_pocket_neighbors=int(args.triangle_pocket_neighbors),
        )
    model = model_cls(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=len(target_keys) if args.multi_head else 1,
        head_names=head_names if args.multi_head else None,
        **model_kwargs,
    ).to(device)
    if str(args.init_checkpoint).strip():
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(init_payload["model_state_dict"], strict=True)
        print(
            json.dumps(
                {
                    "event": "initialized_from_checkpoint",
                    "checkpoint": str(init_path),
                    "source_best_epoch": init_payload.get("best_epoch"),
                    "source_best_metric": init_payload.get("best_metric"),
                    "source_best_metric_value": init_payload.get("best_metric_value"),
                }
            )
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = []
    best_metric = float("inf") if str(args.best_mode) == "min" else -float("inf")
    best_path = output_dir / "best.pt"
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, l_values, l_hjbs, l_boundaries, l_ranks, l_aug_ranks, l_dirs, l_branch_coss, l_branch_pairs, l_cfs, l_cf_dirs, l_policies, l_flow_bellmans, l_fm_aligns, l_action_geoms, grad_norms, dot_vfm_grad_means, fm_align_cosines, action_geom_derivs, action_geom_cosines, action_geom_active_rates = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        cf_maes, cf_sign_accs, cf_pair_counts = [], [], []
        cf_dir_maes, cf_dir_sign_accs, cf_dir_cosines, cf_dir_pair_counts = [], [], [], []
        pcgrad_conflict_rates, pcgrad_mean_cosines = [], []
        skipped_nonfinite_batches = 0
        skipped_nonfinite_grad_batches = 0
        terminal_iter = iter(terminal_train_loader) if terminal_train_loader is not None else None
        cf_iter = iter(cf_train_loader) if cf_train_loader is not None else None
        for step_idx, batch in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and step_idx >= int(args.max_train_batches):
                break
            if (
                terminal_train_loader is not None
                and int(args.boundary_batch_size) > 0
                and int(args.boundary_every) > 0
                and step_idx % int(args.boundary_every) == 0
            ):
                try:
                    terminal_batch = next(terminal_iter)
                except StopIteration:
                    terminal_iter = iter(terminal_train_loader)
                    terminal_batch = next(terminal_iter)
                batch = list(batch) + list(terminal_batch)
            data = collate_graphs(batch, device)
            pos = data["ligand_pos"].detach().requires_grad_(True)
            t = data["time_fraction"].detach().requires_grad_(True)
            out = model(pos, data["ligand_v"], data["protein_pos"], data["protein_v"], data["batch_ligand"], data["batch_protein"], t)
            head_value_losses = []
            if args.multi_head:
                head_value_losses = per_head_huber_losses(out, data["targets"])
                l_value = torch.stack(head_value_losses).mean()
                s = select_hjb_value(out, args.hjb_component, head_names)
            else:
                s = out
                if args.value_loss_type == "mse":
                    l_value = F.mse_loss(s, data["G_t"])
                else:
                    l_value = F.huber_loss(s, data["G_t"], delta=1.0)
            if bool(data["is_terminal"].any()):
                l_boundary = F.huber_loss(s[data["is_terminal"]], data["boundary_value"][data["is_terminal"]], delta=1.0)
            else:
                l_boundary = s.new_zeros(())
            target_for_rank = select_hjb_value(data["targets"], args.hjb_component, head_names) if args.multi_head and "targets" in data else data["G_t"]
            l_rank = within_group_rank_loss(
                s,
                target_for_rank,
                data["group_id"],
                margin=float(args.rank_margin),
            )
            if float(args.lambda_aug_rank) > 0:
                l_aug_rank = augmented_clean_bad_rank_loss(
                    model,
                    s,
                    data,
                    component=args.hjb_component,
                    head_names=head_names,
                    margin=float(args.aug_rank_margin),
                )
            else:
                l_aug_rank = s.new_zeros(())
            grad_x = torch.autograd.grad(s.sum(), pos, create_graph=True, retain_graph=True)[0]
            ds_dt = torch.autograd.grad(s.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
            if ds_dt is None:
                ds_dt = torch.zeros_like(s)
            num_graphs = int(s.size(0))
            grad_sq = graph_mean_squared_norm(grad_x, data["batch_ligand"], num_graphs)
            dot_vfm_grad = graph_mean_dot(data["v_fm"], grad_x, data["batch_ligand"], num_graphs)
            if args.hjb_mode == "residual":
                residual = ds_dt + dot_vfm_grad - 0.5 * grad_sq + data["U_t"]
            elif args.hjb_mode == "fixed_policy":
                residual = ds_dt + dot_vfm_grad + data["U_t"]
            else:
                residual = ds_dt - 0.5 * grad_sq + data["U_t"]
            hjb_mask = ~data["boundary_only"]
            l_hjb = residual[hjb_mask].pow(2).mean() if bool(hjb_mask.any()) else residual.sum() * 0.0
            l_dir = directional_branch_loss(
                grad_x,
                data,
                margin=float(args.dir_margin),
                improve_margin=float(args.dir_improve_margin),
            )
            l_branch_cos = branch_cosine_loss(
                grad_x,
                data,
                margin=float(args.branch_cos_margin),
                improve_margin=float(args.dir_improve_margin),
                weighted=bool(args.branch_cos_weighted),
            )
            l_branch_pair = branch_pairwise_directional_loss(
                grad_x,
                data,
                margin=float(args.branch_pair_margin),
                improve_margin=float(args.dir_improve_margin),
                weighted=bool(args.branch_pair_weighted),
                max_pairs_per_graph=int(args.branch_pair_max_pairs),
            )
            l_branch_soft, _ = branch_soft_action_loss(grad_x, data)
            if float(args.lambda_policy_bellman) > 0:
                l_policy = policy_bellman_loss(
                    model,
                    s,
                    data,
                    component=args.hjb_component,
                    head_names=head_names,
                    policy_rho=float(args.policy_rho),
                    cost_scale=float(args.policy_cost_scale),
                    control_cost_weight=float(args.policy_control_cost_weight),
                    policy_steps=int(args.policy_steps),
                    loss_type=str(args.aux_loss_type),
                )
            else:
                l_policy = s.new_zeros(())
            if float(args.lambda_flow_bellman) > 0:
                l_flow_bellman = flow_bellman_loss(
                    model,
                    s,
                    data,
                    component=args.hjb_component,
                    head_names=head_names,
                    loss_type=str(args.aux_loss_type),
                )
            else:
                l_flow_bellman = s.new_zeros(())
            l_fm_align, fm_align_cos = fm_alignment_loss(
                grad_x,
                data,
                margin=float(args.fm_align_margin),
                improve_margin=float(args.fm_align_improve_margin),
            )
            if float(args.lambda_action_geom) > 0:
                l_action_geom, action_geom_deriv, action_geom_cos, action_geom_active = action_geometry_direction_loss(
                    grad_x,
                    data,
                    pos,
                    clash_weight=float(args.action_geom_clash_weight),
                    center_weight=float(args.action_geom_center_weight),
                    contact_weight=float(args.action_geom_contact_weight),
                    margin=float(args.action_geom_margin),
                    mode=str(args.action_geom_mode),
                    active_clash_threshold=float(args.action_geom_active_clash_threshold),
                )
            else:
                l_action_geom = s.new_zeros(())
                action_geom_deriv = torch.empty(0, device=device)
                action_geom_cos = torch.empty(0, device=device)
                action_geom_active = torch.empty(0, device=device)
            if cf_train_loader is not None:
                try:
                    cf_batch = next(cf_iter)
                except StopIteration:
                    cf_iter = iter(cf_train_loader)
                    cf_batch = next(cf_iter)
                cf_data = collate_graphs(cf_batch, device)
                if float(args.lambda_cf) > 0:
                    l_cf, cf_mae, cf_sign_acc, cf_pairs = branch_counterfactual_loss(
                        model,
                        cf_data,
                        component=args.hjb_component,
                        head_names=head_names,
                        lookahead=cf_lookahead,
                        max_nonbase_branches=int(args.cf_max_nonbase_branches),
                        loss_type=str(args.aux_loss_type),
                    )
                else:
                    l_cf = s.new_zeros(())
                    cf_mae = s.detach().new_tensor(float("nan"))
                    cf_sign_acc = s.detach().new_tensor(float("nan"))
                    cf_pairs = 0
                if float(args.lambda_cf_directional) > 0:
                    l_cf_dir, cf_dir_mae, cf_dir_sign_acc, cf_dir_cos, cf_dir_pairs = branch_counterfactual_directional_loss(
                        model,
                        cf_data,
                        component=args.hjb_component,
                        head_names=head_names,
                        max_nonbase_branches=int(args.cf_max_nonbase_branches),
                        create_graph=True,
                        loss_type=str(args.aux_loss_type),
                        normalize_derivative=bool(args.cf_directional_normalize),
                    )
                else:
                    l_cf_dir = s.new_zeros(())
                    cf_dir_mae = s.detach().new_tensor(float("nan"))
                    cf_dir_sign_acc = s.detach().new_tensor(float("nan"))
                    cf_dir_cos = s.detach().new_tensor(float("nan"))
                    cf_dir_pairs = 0
            else:
                l_cf = s.new_zeros(())
                cf_mae = s.detach().new_tensor(float("nan"))
                cf_sign_acc = s.detach().new_tensor(float("nan"))
                cf_pairs = 0
                l_cf_dir = s.new_zeros(())
                cf_dir_mae = s.detach().new_tensor(float("nan"))
                cf_dir_sign_acc = s.detach().new_tensor(float("nan"))
                cf_dir_cos = s.detach().new_tensor(float("nan"))
                cf_dir_pairs = 0
            loss = (
                float(args.lambda_value) * l_value
                + float(args.lambda_hjb) * l_hjb
                + float(args.lambda_boundary) * l_boundary
                + float(args.lambda_rank) * l_rank
                + float(args.lambda_aug_rank) * l_aug_rank
                + float(args.lambda_dir) * l_dir
                + float(args.lambda_branch_cos) * l_branch_cos
                + float(args.lambda_branch_pair_rank) * l_branch_pair
                + float(args.lambda_cf) * l_cf
                + float(args.lambda_cf_directional) * l_cf_dir
                + float(args.lambda_branch_soft_action) * l_branch_soft
                + float(args.lambda_policy_bellman) * l_policy
                + float(args.lambda_flow_bellman) * l_flow_bellman
                + float(args.lambda_fm_align) * l_fm_align
                + float(args.lambda_action_geom) * l_action_geom
            )
            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                if skipped_nonfinite_batches <= 5:
                    components = {
                        "event": "skip_nonfinite_loss",
                        "epoch": int(epoch),
                        "step": int(step_idx),
                        "loss": float(loss.detach().cpu()) if torch.isfinite(loss.detach()).all() else float("nan"),
                        "value": float(l_value.detach().cpu()) if torch.isfinite(l_value.detach()).all() else float("nan"),
                        "hjb": float(l_hjb.detach().cpu()) if torch.isfinite(l_hjb.detach()).all() else float("nan"),
                        "boundary": float(l_boundary.detach().cpu()) if torch.isfinite(l_boundary.detach()).all() else float("nan"),
                        "rank": float(l_rank.detach().cpu()) if torch.isfinite(l_rank.detach()).all() else float("nan"),
                        "aug_rank": float(l_aug_rank.detach().cpu()) if torch.isfinite(l_aug_rank.detach()).all() else float("nan"),
                        "dir": float(l_dir.detach().cpu()) if torch.isfinite(l_dir.detach()).all() else float("nan"),
                        "branch_cos": float(l_branch_cos.detach().cpu()) if torch.isfinite(l_branch_cos.detach()).all() else float("nan"),
                        "branch_pair": float(l_branch_pair.detach().cpu()) if torch.isfinite(l_branch_pair.detach()).all() else float("nan"),
                        "counterfactual": float(l_cf.detach().cpu()) if torch.isfinite(l_cf.detach()).all() else float("nan"),
                        "counterfactual_directional": float(l_cf_dir.detach().cpu()) if torch.isfinite(l_cf_dir.detach()).all() else float("nan"),
                        "action_geom": float(l_action_geom.detach().cpu()) if torch.isfinite(l_action_geom.detach()).all() else float("nan"),
                    }
                    print(json.dumps(components), flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            if str(args.value_grad_strategy) == "pcgrad" and args.multi_head and len(head_value_losses) > 1:
                params = [p for p in model.parameters() if p.requires_grad]
                merged_value_grad, pcgrad_stats = pcgrad_merge_losses(head_value_losses, params)
                add_flat_grad_to_params(params, merged_value_grad, scale=float(args.lambda_value))
                aux_loss = (
                    float(args.lambda_hjb) * l_hjb
                    + float(args.lambda_boundary) * l_boundary
                    + float(args.lambda_rank) * l_rank
                    + float(args.lambda_aug_rank) * l_aug_rank
                    + float(args.lambda_dir) * l_dir
                    + float(args.lambda_branch_cos) * l_branch_cos
                    + float(args.lambda_branch_pair_rank) * l_branch_pair
                    + float(args.lambda_cf) * l_cf
                    + float(args.lambda_cf_directional) * l_cf_dir
                    + float(args.lambda_branch_soft_action) * l_branch_soft
                    + float(args.lambda_policy_bellman) * l_policy
                    + float(args.lambda_flow_bellman) * l_flow_bellman
                    + float(args.lambda_fm_align) * l_fm_align
                    + float(args.lambda_action_geom) * l_action_geom
                )
                if float(aux_loss.detach().abs().item()) > 0.0:
                    aux_loss.backward()
                pcgrad_conflict_rates.append(float(pcgrad_stats["conflict_rate"]))
                pcgrad_mean_cosines.append(float(pcgrad_stats["mean_cos"]))
            else:
                loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(total_grad_norm):
                skipped_nonfinite_grad_batches += 1
                if skipped_nonfinite_grad_batches <= 5:
                    print(
                        json.dumps(
                            {
                                "event": "skip_nonfinite_gradient",
                                "epoch": int(epoch),
                                "step": int(step_idx),
                                "grad_norm": float("nan"),
                            }
                        ),
                        flush=True,
                    )
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            losses.append(float(loss.item()))
            l_values.append(float(l_value.item()))
            l_hjbs.append(float(l_hjb.item()))
            l_boundaries.append(float(l_boundary.item()))
            l_ranks.append(float(l_rank.item()))
            l_aug_ranks.append(float(l_aug_rank.item()))
            l_dirs.append(float(l_dir.item()))
            l_branch_coss.append(float(l_branch_cos.item()))
            l_branch_pairs.append(float(l_branch_pair.item()))
            l_cfs.append(float(l_cf.item()))
            l_cf_dirs.append(float(l_cf_dir.item()))
            if torch.isfinite(cf_mae):
                cf_maes.append(float(cf_mae.item()))
            if torch.isfinite(cf_sign_acc):
                cf_sign_accs.append(float(cf_sign_acc.item()))
            cf_pair_counts.append(int(cf_pairs))
            if torch.isfinite(cf_dir_mae):
                cf_dir_maes.append(float(cf_dir_mae.item()))
            if torch.isfinite(cf_dir_sign_acc):
                cf_dir_sign_accs.append(float(cf_dir_sign_acc.item()))
            if torch.isfinite(cf_dir_cos):
                cf_dir_cosines.append(float(cf_dir_cos.item()))
            cf_dir_pair_counts.append(int(cf_dir_pairs))
            l_policies.append(float(l_policy.item()))
            l_flow_bellmans.append(float(l_flow_bellman.item()))
            l_fm_aligns.append(float(l_fm_align.item()))
            l_action_geoms.append(float(l_action_geom.item()))
            grad_norms.append(float(grad_sq.detach().mean().sqrt().item()))
            dot_vfm_grad_means.append(float(dot_vfm_grad.detach().mean().item()))
            if fm_align_cos.numel():
                fm_align_cosines.append(float(torch.nanmean(fm_align_cos.float()).item()))
            if action_geom_deriv.numel():
                action_geom_derivs.append(float(torch.nanmean(action_geom_deriv.float()).item()))
            if action_geom_cos.numel():
                action_geom_cosines.append(float(torch.nanmean(action_geom_cos.float()).item()))
            if action_geom_active.numel():
                action_geom_active_rates.append(float(action_geom_active.float().mean().item()))
        val_metrics = eval_model(
            model,
            val_loader,
            device,
            max_batches=int(args.val_max_batches),
            component=args.hjb_component,
            head_names=head_names,
            multi_head=bool(args.multi_head),
            hjb_mode=args.hjb_mode,
            dir_improve_margin=float(args.dir_improve_margin),
        )
        val_cf_metrics = eval_counterfactual_model(
            model,
            cf_val_loader,
            device,
            component=args.hjb_component,
            head_names=head_names,
            lookahead=cf_lookahead,
            max_batches=int(args.val_max_batches),
            loss_type=str(args.aux_loss_type),
        )
        val_cf_dir_metrics = eval_counterfactual_directional_model(
            model,
            cf_val_loader,
            device,
            component=args.hjb_component,
            head_names=head_names,
            max_nonbase_branches=0,
            max_batches=int(args.val_max_batches),
            loss_type=str(args.aux_loss_type),
            normalize_derivative=bool(args.cf_directional_normalize),
        )
        row = {
            "epoch": epoch,
            "hjb_mode": args.hjb_mode,
            "loss": float(np.mean(losses)),
            "value": float(np.mean(l_values)),
            "hjb": float(np.mean(l_hjbs)),
            "boundary": float(np.mean(l_boundaries)),
            "rank": float(np.mean(l_ranks)),
            "aug_rank": float(np.mean(l_aug_ranks)),
            "dir": float(np.mean(l_dirs)),
            "branch_cos": float(np.mean(l_branch_coss)),
            "branch_pair": float(np.mean(l_branch_pairs)),
            "counterfactual": float(np.mean(l_cfs)),
            "counterfactual_mae": float(np.mean(cf_maes)) if cf_maes else float("nan"),
            "counterfactual_sign_acc": float(np.mean(cf_sign_accs)) if cf_sign_accs else float("nan"),
            "counterfactual_pairs": int(np.sum(cf_pair_counts)),
            "counterfactual_directional": float(np.mean(l_cf_dirs)),
            "counterfactual_directional_mae": float(np.mean(cf_dir_maes)) if cf_dir_maes else float("nan"),
            "counterfactual_directional_sign_acc": float(np.mean(cf_dir_sign_accs)) if cf_dir_sign_accs else float("nan"),
            "counterfactual_directional_aligned_cos": float(np.mean(cf_dir_cosines)) if cf_dir_cosines else float("nan"),
            "counterfactual_directional_pairs": int(np.sum(cf_dir_pair_counts)),
            "policy_bellman": float(np.mean(l_policies)),
            "flow_bellman": float(np.mean(l_flow_bellmans)),
            "fm_align": float(np.mean(l_fm_aligns)),
            "action_geom": float(np.mean(l_action_geoms)),
            "grad_norm": float(np.mean(grad_norms)),
            "hjb_residual_mean": float(np.mean(l_hjbs)) ** 0.5,
            "dot_vfm_grad_mean": float(np.mean(dot_vfm_grad_means)) if dot_vfm_grad_means else float("nan"),
            "fm_align_cos_mean": float(np.mean(fm_align_cosines)) if fm_align_cosines else float("nan"),
            "action_geom_deriv_mean": float(np.mean(action_geom_derivs)) if action_geom_derivs else float("nan"),
            "action_geom_cos_mean": float(np.mean(action_geom_cosines)) if action_geom_cosines else float("nan"),
            "action_geom_active_rate": float(np.mean(action_geom_active_rates)) if action_geom_active_rates else float("nan"),
            "pcgrad_conflict_rate": float(np.nanmean(pcgrad_conflict_rates)) if pcgrad_conflict_rates else float("nan"),
            "pcgrad_mean_cos": float(np.nanmean(pcgrad_mean_cosines)) if pcgrad_mean_cosines else float("nan"),
            "skipped_nonfinite_batches": int(skipped_nonfinite_batches),
            "skipped_nonfinite_grad_batches": int(skipped_nonfinite_grad_batches),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_cf_{k}": v for k, v in val_cf_metrics.items()},
            **{f"val_cf_directional_{k}": v for k, v in val_cf_dir_metrics.items()},
        }
        val_spearman = float(row.get("val_spearman_s_G", float("nan")))
        val_branch_cos = float(row.get("val_branch_cos_mean", float("nan")))
        row["val_phys_composite"] = (
            val_spearman + 0.25 * val_branch_cos
            if math.isfinite(val_spearman) and math.isfinite(val_branch_cos)
            else val_spearman
        )
        history.append(row)
        metric = row.get(str(args.best_metric), float("nan"))
        improved = metric_is_better(metric, best_metric, args.best_mode)
        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "bank": args.bank,
                    "target_key": args.target_key,
                    "target_keys": target_keys,
                    "model_type": (
                        "hjb_value_triangle_aware_multihead"
                        if args.multi_head and str(args.model_arch) == "triangle_aware"
                        else "hjb_value_triangle_aware"
                        if str(args.model_arch) == "triangle_aware"
                        else "hjb_value_physical_pairwise_multihead"
                        if args.multi_head and str(args.model_arch) == "physical_pairwise"
                        else "hjb_value_physical_pairwise"
                        if str(args.model_arch) == "physical_pairwise"
                        else "hjb_value_pairwise_multihead"
                        if args.multi_head and str(args.model_arch) == "pairwise"
                        else "hjb_value_pairwise"
                        if str(args.model_arch) == "pairwise"
                        else "hjb_value_multihead"
                        if args.multi_head
                        else "hjb_value"
                    ),
                    "model_arch": str(args.model_arch),
                    "output_dim": model.output_dim,
                    "head_names": model.head_names,
                    "hjb_mode": args.hjb_mode,
                    "hjb_u_key": args.hjb_u_key,
                    "lambda_value": float(args.lambda_value),
                    "lambda_boundary": float(args.lambda_boundary),
                    "lambda_rank": float(args.lambda_rank),
                    "rank_margin": float(args.rank_margin),
                    "lambda_aug_rank": float(args.lambda_aug_rank),
                    "aug_rank_margin": float(args.aug_rank_margin),
                    "lambda_dir": float(args.lambda_dir),
                    "lambda_branch_cos": float(args.lambda_branch_cos),
                    "branch_cos_margin": float(args.branch_cos_margin),
                    "branch_cos_weighted": bool(args.branch_cos_weighted),
                    "lambda_branch_pair_rank": float(args.lambda_branch_pair_rank),
                    "branch_pair_margin": float(args.branch_pair_margin),
                    "branch_pair_weighted": bool(args.branch_pair_weighted),
                    "branch_pair_max_pairs": int(args.branch_pair_max_pairs),
                    "lambda_cf": float(args.lambda_cf),
                    "lambda_cf_directional": float(args.lambda_cf_directional),
                    "cf_directional_normalize": bool(args.cf_directional_normalize),
                    "aux_loss_type": str(args.aux_loss_type),
                    "cf_batch_size": int(args.cf_batch_size),
                    "cf_lookahead": int(cf_lookahead),
                    "cf_max_nonbase_branches": int(args.cf_max_nonbase_branches),
                    "branch_soft_action_key": str(args.branch_soft_action_key),
                    "lambda_branch_soft_action": float(args.lambda_branch_soft_action),
                    "lambda_policy_bellman": float(args.lambda_policy_bellman),
                    "policy_rho": float(args.policy_rho),
                    "policy_cost_scale": float(args.policy_cost_scale),
                    "policy_control_cost_weight": float(args.policy_control_cost_weight),
                    "policy_steps": int(args.policy_steps),
                    "lambda_flow_bellman": float(args.lambda_flow_bellman),
                    "boundary_batch_size": int(args.boundary_batch_size),
                    "boundary_every": int(args.boundary_every),
                    "boundary_key": str(args.boundary_key),
                    "val_fraction": float(args.val_fraction),
                    "val_max_batches": int(args.val_max_batches),
                    "val_balance_mode": str(args.val_balance_mode),
                    "val_time_bins": int(args.val_time_bins),
                    "best_metric": str(args.best_metric),
                    "best_mode": str(args.best_mode),
                    "lambda_fm_align": float(args.lambda_fm_align),
                    "fm_align_margin": float(args.fm_align_margin),
                    "lambda_action_geom": float(args.lambda_action_geom),
                    "action_geom_clash_weight": float(args.action_geom_clash_weight),
                    "action_geom_center_weight": float(args.action_geom_center_weight),
                    "action_geom_contact_weight": float(args.action_geom_contact_weight),
                    "action_geom_margin": float(args.action_geom_margin),
                    "action_geom_mode": str(args.action_geom_mode),
                    "action_geom_active_clash_threshold": float(args.action_geom_active_clash_threshold),
                    "ligand_feature_dim": model.ligand_feature_dim,
                    "protein_feature_dim": model.protein_feature_dim,
                    "pair_rbf_dim": int(args.pair_rbf_dim),
                    "pair_cutoff": float(args.pair_cutoff),
                    "pair_cutoff_temperature": float(args.pair_cutoff_temperature),
                    "best_epoch": epoch,
                    "best_metric_value": best_metric,
                    "best_val_spearman_s_G": row.get("val_spearman_s_G", float("nan")),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        print(json.dumps(row))
        if (
            int(args.early_stop_patience) > 0
            and epoch >= int(args.early_stop_min_epochs)
            and epochs_without_improvement >= int(args.early_stop_patience)
        ):
            stopped_early = True
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_metric": str(args.best_metric),
                        "best_metric_value": best_metric,
                        "patience": int(args.early_stop_patience),
                    }
                )
            )
            break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if history:
        xs = [h["epoch"] for h in history]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, [h["value"] for h in history], label="value")
        plt.plot(xs, [h["hjb"] for h in history], label="hjb")
        plt.plot(xs, [h["boundary"] for h in history], label="boundary")
        plt.plot(xs, [h["rank"] for h in history], label="rank")
        plt.plot(xs, [h["aug_rank"] for h in history], label="aug rank")
        plt.plot(xs, [h["branch_cos"] for h in history], label="branch cos")
        plt.plot(xs, [h["branch_pair"] for h in history], label="branch pair")
        plt.plot(xs, [h["policy_bellman"] for h in history], label="policy")
        plt.plot(xs, [h["flow_bellman"] for h in history], label="flow")
        plt.plot(xs, [h["action_geom"] for h in history], label="action geom")
        plt.plot(xs, [h["val_spearman_s_G"] for h in history], label="val spearman")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=160)
        plt.close()
    summary = {
        "best_checkpoint": str(best_path),
        "best_metric": str(args.best_metric),
        "best_mode": str(args.best_mode),
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history),
        "stopped_early": bool(stopped_early),
        "init_checkpoint": str(args.init_checkpoint),
        "split_seed": split_seed,
        "pocket_bootstrap_train": bool(args.pocket_bootstrap_train),
        "bootstrap": bootstrap_metadata,
        "max_train_batches": int(args.max_train_batches),
        "final": history[-1] if history else {},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "summary.md").write_text(
        "# HJB Value Training\n\n"
        f"- Bank: `{args.bank}`\n"
        f"- Target key: `{args.target_key}`\n"
        f"- Multi-head: `{bool(args.multi_head)}`\n"
        f"- Target keys: `{','.join(target_keys) if target_keys else args.target_key}`\n"
        f"- HJB component: `{args.hjb_component}`\n"
        f"- HJB U key: `{args.hjb_u_key}`\n"
        f"- Boundary key: `{args.boundary_key or args.target_key + '[-1]'}`\n"
        f"- Validation balance: `{args.val_balance_mode}`\n"
        f"- Split seed: {split_seed}\n"
        f"- Pocket-bootstrap training: {bool(args.pocket_bootstrap_train)}\n"
        f"- Bootstrap seed: {bootstrap_metadata['seed'] if bootstrap_metadata else 'none'}\n"
        f"- Bootstrap unique/OOB pockets: {bootstrap_metadata['unique_drawn_pockets'] if bootstrap_metadata else 'n/a'}/{bootstrap_metadata['out_of_bag_pockets'] if bootstrap_metadata else 'n/a'}\n"
        f"- Validation max batches: {args.val_max_batches}\n"
        f"- Validation time bins: {args.val_time_bins}\n"
        f"- Value grad strategy: `{args.value_grad_strategy}`\n"
        f"- Lambda value: {args.lambda_value}\n"
        f"- Lambda HJB: {args.lambda_hjb}\n"
        f"- Lambda boundary: {args.lambda_boundary}\n"
        f"- Lambda rank: {args.lambda_rank}\n"
        f"- Rank margin: {args.rank_margin}\n"
        f"- Lambda augmented rank: {args.lambda_aug_rank}\n"
        f"- Augmented rank margin: {args.aug_rank_margin}\n"
        f"- Boundary batch size: {args.boundary_batch_size}\n"
        f"- Boundary every: {args.boundary_every}\n"
        f"- Lambda dir: {args.lambda_dir}\n"
        f"- Lambda branch cos: {args.lambda_branch_cos}\n"
        f"- Branch cos margin: {args.branch_cos_margin}\n"
        f"- Branch cos weighted: {bool(args.branch_cos_weighted)}\n"
        f"- Lambda branch pair rank: {args.lambda_branch_pair_rank}\n"
        f"- Branch pair margin: {args.branch_pair_margin}\n"
        f"- Branch pair weighted: {bool(args.branch_pair_weighted)}\n"
        f"- Branch pair max pairs: {args.branch_pair_max_pairs}\n"
        f"- Branch soft action key: `{args.branch_soft_action_key}`\n"
        f"- Lambda branch soft action: {args.lambda_branch_soft_action}\n"
        f"- Lambda counterfactual value: {args.lambda_cf}\n"
        f"- Lambda costate: {args.lambda_cf_directional}\n"
        f"- Normalize costate derivatives: {bool(args.cf_directional_normalize)}\n"
        f"- Auxiliary loss: `{args.aux_loss_type}`\n"
        f"- Lambda policy Bellman: {args.lambda_policy_bellman}\n"
        f"- Policy rho: {args.policy_rho}\n"
        f"- Policy cost scale: {args.policy_cost_scale}\n"
        f"- Policy control cost weight: {args.policy_control_cost_weight}\n"
        f"- Policy steps: {args.policy_steps}\n"
        f"- Lambda flow Bellman: {args.lambda_flow_bellman}\n"
        f"- Lambda FM align: {args.lambda_fm_align}\n"
        f"- Lambda action geom: {args.lambda_action_geom}\n"
        f"- Value regression loss: `{args.value_loss_type}`\n"
        f"- Initialization checkpoint: `{args.init_checkpoint or 'none'}`\n"
        f"- Maximum train batches per epoch: {args.max_train_batches}\n"
        f"- Early stopping patience: {args.early_stop_patience}\n"
        f"- Early stopping minimum epochs: {args.early_stop_min_epochs}\n"
        f"- Epochs completed: {len(history)} of {args.epochs}\n"
        f"- Stopped early: {stopped_early}\n"
        f"- Action geom weights: clash={args.action_geom_clash_weight}, center={args.action_geom_center_weight}, contact={args.action_geom_contact_weight}\n"
        f"- Action geom margin: {args.action_geom_margin}\n"
        f"- Action geom mode: {args.action_geom_mode}\n"
        f"- Action geom active clash threshold: {args.action_geom_active_clash_threshold}\n"
        f"- HJB mode: {args.hjb_mode}\n"
        f"- Best metric: `{args.best_metric}` ({args.best_mode})\n"
        f"- Best checkpoint: `{best_path}`\n"
        f"- Best metric value: {best_metric:.4f}\n"
    )


if __name__ == "__main__":
    main()

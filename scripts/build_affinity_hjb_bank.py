from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import utils.misc as misc
from models.molopt_score_model_energy_guide import ScorePosNet3D_guided_flow, center_pos
from scripts.sample_flow_VP_split import load_sampling_data


def safe_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        value = float(x)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def safe_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def zscore(values, eps=1e-6):
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if int(mask.sum()) == 0:
        return out, {"mean": 0.0, "std": 1.0}
    mean = float(arr[mask].mean())
    std = max(float(arr[mask].std()), float(eps))
    out[mask] = ((arr[mask] - mean) / std).astype(np.float32)
    return out, {"mean": mean, "std": std}


def affinity(vina, key):
    value = (vina or {}).get(key)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, dict):
        return safe_float(value.get("affinity"))
    return safe_float(value)


def metric_signature(metric_row: dict):
    pos = np.asarray(metric_row.get("pred_pos"), dtype=np.float32)
    v = np.asarray(metric_row.get("pred_v"), dtype=np.int64)
    if pos.ndim != 2 or v.ndim != 1:
        return None
    return (int(pos.shape[0]), tuple(v.tolist()), np.round(pos, 3).tobytes())


def result_signature(result: dict, sample_idx: int):
    pos = np.asarray(result["pred_ligand_pos"][sample_idx], dtype=np.float32)
    v = np.asarray(result["pred_ligand_v"][sample_idx], dtype=np.int64)
    return (int(pos.shape[0]), tuple(v.tolist()), np.round(pos, 3).tobytes())


def metric_row_to_labels(item: dict):
    vina = item.get("vina") or {}
    chem = item.get("chem_results") or {}
    return {
        "has_eval_metrics": 1.0,
        "vina_score": affinity(vina, "score_only"),
        "vina_min": affinity(vina, "minimize"),
        "vina_dock": affinity(vina, "dock"),
        "qed": safe_float(chem.get("qed")),
        "sa": safe_float(chem.get("sa")),
        "complete": float(item.get("mol") is not None),
        "smiles": item.get("smiles", ""),
    }


def load_metric_map(result_dir: Path):
    metrics_path = result_dir / "eval_results" / "metrics_-1.pt"
    if not metrics_path.exists():
        return {}, {}
    metrics = torch.load(metrics_path, map_location="cpu", weights_only=False)
    source_map = {}
    signature_map = {}
    for item in metrics.get("all_results", []):
        labels = metric_row_to_labels(item)
        rf = item.get("source_result_file")
        si = item.get("source_sample_idx")
        if rf is not None and si is not None:
            source_map[(str(rf), int(si))] = labels
        sig = metric_signature(item)
        if sig is not None:
            signature_map[sig] = labels
    return source_map, signature_map


def state_components(protein_pos: torch.Tensor, ligand_pos_traj: torch.Tensor):
    raw = {k: [] for k in ["clash", "severe_clash", "overburied", "contact", "center_drift", "step_norm"]}
    prot_center = protein_pos.float().mean(dim=0)
    T = int(ligand_pos_traj.size(0))
    for t in range(T):
        pos = ligand_pos_traj[t].float()
        d = torch.cdist(pos, protein_pos.float()).clamp_min(1e-4)
        n_lig = max(int(pos.size(0)), 1)
        raw["clash"].append(float(torch.relu(torch.as_tensor(2.0) - d).pow(2).sum().item() / n_lig))
        raw["severe_clash"].append(float(torch.relu(torch.as_tensor(1.55) - d).pow(2).sum().item() / n_lig))
        raw["overburied"].append(float(torch.relu(torch.as_tensor(2.4) - d.min(dim=1).values).pow(2).mean().item()))
        raw["contact"].append(float(torch.exp(-((d - 3.6) / 0.75).pow(2)).sum().item() / n_lig))
        raw["center_drift"].append(float((pos.mean(dim=0) - prot_center).norm().item()))
        if t + 1 < T:
            raw["step_norm"].append(float((ligand_pos_traj[t + 1].float() - pos).pow(2).sum(dim=-1).mean().sqrt().item()))
        else:
            raw["step_norm"].append(0.0)
    return raw


def result_index(path: Path):
    return int(path.stem.split("_")[-1])


def parse_gamma_dir(item: str):
    if "=" not in item:
        raise ValueError(f"Expected GAMMA=DIR, got {item!r}")
    gamma_str, dir_str = item.split("=", 1)
    return float(gamma_str), Path(dir_str)


def build_model(config_path: Path, subset: str, device: torch.device):
    config = misc.load_config(str(config_path))
    if subset:
        config.data.subset = subset
    ckpt, _, _, protein_featurizer, ligand_featurizer = load_sampling_data(config, config.data.subset)
    model = ScorePosNet3D_guided_flow(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=str(device),
    ).to(device)
    expert_ckpt = torch.load(config.model.checkpoint, map_location=device, weights_only=False)
    for key, value in expert_ckpt["model"].items():
        if key.startswith("expert_pred"):
            ckpt["model"][key] = value
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def score_expert_trajectory(model, protein_pos, protein_v, pos_traj, v_traj, device, batch_size: int):
    protein_pos0 = protein_pos.float().to(device)
    protein_v0 = protein_v.float().to(device)
    scores = []
    T = int(pos_traj.size(0))
    for start in range(0, T, max(int(batch_size), 1)):
        end = min(start + max(int(batch_size), 1), T)
        pos_parts, v_parts, bp_parts, bv_parts, bl_parts, bpr_parts, steps = [], [], [], [], [], [], []
        for graph_idx, t in enumerate(range(start, end)):
            pos = pos_traj[t].float().to(device)
            ids = v_traj[t].long().to(device).clamp(min=0, max=model.num_classes - 1)
            pos_parts.append(pos)
            v_parts.append(F.one_hot(ids, model.num_classes).float())
            bl_parts.append(torch.full((pos.size(0),), graph_idx, dtype=torch.long, device=device))
            bp_parts.append(protein_pos0)
            bv_parts.append(protein_v0)
            bpr_parts.append(torch.full((protein_pos0.size(0),), graph_idx, dtype=torch.long, device=device))
            remaining = max((T - 1) - int(t), 0)
            steps.append(int(round(remaining * 1000.0 / float(max(T, 1)))))
        ligand_pos = torch.cat(pos_parts, dim=0)
        ligand_v = torch.cat(v_parts, dim=0)
        batch_ligand = torch.cat(bl_parts, dim=0)
        protein_pos_b = torch.cat(bp_parts, dim=0)
        protein_v_b = torch.cat(bv_parts, dim=0)
        batch_protein = torch.cat(bpr_parts, dim=0)
        time_step = torch.as_tensor(steps, dtype=torch.long, device=device)
        protein_pos_b, ligand_pos, _ = center_pos(
            protein_pos_b,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=model.center_pos_mode,
        )
        with torch.no_grad():
            pred = model(
                protein_pos=protein_pos_b,
                protein_v=protein_v_b,
                batch_protein=batch_protein,
                ligand_xt=ligand_pos,
                ligand_vt=ligand_v,
                batch_ligand=batch_ligand,
                time_step=time_step,
            )["final_affinity_pred"].detach().cpu().float()
        scores.append(pred)
    return torch.cat(scores, dim=0)


def read_result_set(gamma: float, result_dir: Path, model, device, args):
    result_files = sorted(result_dir.glob("result_*.pt"), key=result_index)
    if args.max_results > 0:
        result_files = result_files[: args.max_results]
    source_map, signature_map = load_metric_map(result_dir)
    metrics_available = (result_dir / "eval_results" / "metrics_-1.pt").exists()
    trajectories = []
    for result_path in tqdm(result_files, desc=f"read gamma={gamma:g}"):
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        data = result["data"]
        protein_pos = safe_tensor(data.protein_pos)
        protein_v = getattr(data, "protein_atom_feature", None)
        if protein_v is None:
            protein_v = getattr(data, "protein_element", None)
        protein_v = safe_tensor(protein_v).float()
        pos_list = result["pred_ligand_pos_traj"]
        v_list = result.get("pred_ligand_v_traj", result.get("pred_ligand_v"))
        n_samples = len(pos_list)
        if args.max_samples_per_result > 0:
            n_samples = min(n_samples, args.max_samples_per_result)
        for sample_idx in range(n_samples):
            pos_traj = safe_tensor(pos_list[sample_idx]).float()
            ligand_v_traj = safe_tensor(v_list[sample_idx], dtype=torch.long)
            if ligand_v_traj.dim() == 1:
                ligand_v_traj = ligand_v_traj.unsqueeze(0).repeat(pos_traj.size(0), 1)
            labels = source_map.get((result_path.name, sample_idx))
            if labels is None:
                labels = signature_map.get(result_signature(result, sample_idx))
            if labels is None:
                if args.missing_metric_is_incomplete and metrics_available:
                    labels = {
                        "has_eval_metrics": 1.0,
                        "complete": 0.0,
                        "vina_score": float("nan"),
                        "vina_min": float("nan"),
                        "vina_dock": float("nan"),
                        "qed": float("nan"),
                        "sa": float("nan"),
                        "smiles": "",
                    }
                else:
                    labels = {
                        "has_eval_metrics": 0.0,
                        "complete": float("nan"),
                        "vina_score": float("nan"),
                        "vina_min": float("nan"),
                        "vina_dock": float("nan"),
                        "qed": float("nan"),
                        "sa": float("nan"),
                        "smiles": "",
                    }
            expert_score = score_expert_trajectory(
                model,
                protein_pos,
                protein_v,
                pos_traj,
                ligand_v_traj,
                device,
                args.expert_batch_size,
            )
            trajectories.append(
                {
                    "gamma": float(gamma),
                    "result_dir": str(result_dir),
                    "result_file": str(result_path),
                    "result_index": result_index(result_path),
                    "sample_index": int(sample_idx),
                    "protein_pos": protein_pos,
                    "protein_v": protein_v,
                    "ligand_pos_traj": pos_traj,
                    "ligand_v_traj": ligand_v_traj,
                    "expert_affinity_score": expert_score,
                    "raw_state_costs": state_components(protein_pos, pos_traj),
                    "terminal_metrics": labels,
                }
            )
    return trajectories


def positive_mean(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return float(np.maximum(arr, 0.0).mean()) if arr.size else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma_result", action="append", required=True, help="GAMMA=/path/to/result_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="./configs/sampling_landscape_test100.yml")
    parser.add_argument("--subset", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_results", type=int, default=-1)
    parser.add_argument("--max_samples_per_result", type=int, default=-1)
    parser.add_argument("--expert_batch_size", type=int, default=10)
    parser.add_argument("--w_vina_dock", type=float, default=0.50)
    parser.add_argument("--w_vina_min", type=float, default=0.30)
    parser.add_argument("--w_vina_score", type=float, default=0.20)
    parser.add_argument("--w_qed", type=float, default=0.15)
    parser.add_argument("--w_sa", type=float, default=0.05)
    parser.add_argument("--w_incomplete", type=float, default=1.50)
    parser.add_argument("--w_local_expert", type=float, default=1.00)
    parser.add_argument("--w_local_safety", type=float, default=0.10)
    parser.add_argument("--w_local_step", type=float, default=0.05)
    parser.add_argument(
        "--terminal_source",
        choices=("metrics", "expert_affinity", "metrics_or_expert"),
        default="metrics",
        help="Terminal boundary source. expert_affinity is useful for non-test pockets without receptor files.",
    )
    parser.add_argument("--w_local_physics", type=float, default=0.0)
    parser.add_argument("--w_terminal_physics", type=float, default=0.0)
    parser.add_argument("--w_physics_clash", type=float, default=1.0)
    parser.add_argument("--w_physics_severe_clash", type=float, default=2.0)
    parser.add_argument("--w_physics_overburied", type=float, default=0.5)
    parser.add_argument("--w_physics_center_drift", type=float, default=0.0)
    parser.add_argument("--w_physics_contact", type=float, default=0.25)
    parser.add_argument("--terminal_property_mode", choices=("none", "hinge"), default="none")
    parser.add_argument("--terminal_qed_threshold", type=float, default=0.45)
    parser.add_argument("--terminal_sa_threshold", type=float, default=0.60)
    parser.add_argument("--terminal_qed_scale", type=float, default=0.10)
    parser.add_argument("--terminal_sa_scale", type=float, default=0.10)
    parser.add_argument("--w_terminal_qed_hinge", type=float, default=0.15)
    parser.add_argument("--w_terminal_sa_hinge", type=float, default=0.40)
    parser.add_argument("--w_terminal_incomplete_hinge", type=float, default=1.00)
    parser.add_argument(
        "--missing_metric_is_incomplete",
        action="store_true",
        help="When eval metrics exist, treat missing sample metrics as reconstruction/evaluation failures.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if (str(args.device) == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model(Path(args.config), args.subset, device)
    gamma_dirs = [parse_gamma_dir(item) for item in args.gamma_result]
    trajectories = []
    for gamma, result_dir in gamma_dirs:
        trajectories.extend(read_result_set(gamma, result_dir, model, device, args))
    if not trajectories:
        raise RuntimeError("No trajectories were loaded.")

    terminal = {k: [] for k in ["vina_dock", "vina_min", "vina_score", "qed", "sa", "complete", "expert_terminal_cost"]}
    state_values = {k: [] for k in ["expert_cost", "clash", "severe_clash", "overburied", "contact", "center_drift", "step_norm"]}
    for tr in trajectories:
        metrics = tr["terminal_metrics"]
        for key in terminal:
            if key == "expert_terminal_cost":
                terminal_score = tr["expert_affinity_score"].float()[-1].clamp(1e-6, 1.0 - 1e-6)
                terminal[key].append(float(-torch.log(terminal_score).item()))
            else:
                terminal[key].append(safe_float(metrics.get(key)))
        expert_score = tr["expert_affinity_score"].float().clamp(1e-6, 1.0 - 1e-6)
        state_values["expert_cost"].extend((-torch.log(expert_score)).numpy().tolist())
        for key in ["clash", "severe_clash", "overburied", "contact", "center_drift", "step_norm"]:
            state_values[key].extend(tr["raw_state_costs"][key])

    terminal_z, terminal_norm = {}, {}
    for key, values in terminal.items():
        terminal_z[key], terminal_norm[key] = zscore(values)
    state_z, state_norm = {}, {}
    for key, values in state_values.items():
        state_z[key], state_norm[key] = zscore(values)

    complete = np.asarray([0.0 if not math.isfinite(v) else v for v in terminal["complete"]], dtype=np.float32)
    metrics_terminal_cost = (
        float(args.w_vina_dock) * terminal_z["vina_dock"]
        + float(args.w_vina_min) * terminal_z["vina_min"]
        + float(args.w_vina_score) * terminal_z["vina_score"]
        - float(args.w_qed) * terminal_z["qed"]
        - float(args.w_sa) * terminal_z["sa"]
        + float(args.w_incomplete) * (1.0 - complete)
    ).astype(np.float32)
    expert_terminal_cost = terminal_z["expert_terminal_cost"].astype(np.float32)
    if args.terminal_source == "expert_affinity":
        terminal_cost = expert_terminal_cost
    elif args.terminal_source == "metrics_or_expert":
        has_metrics = np.asarray(
            [safe_float(tr["terminal_metrics"].get("has_eval_metrics"), 0.0) for tr in trajectories],
            dtype=np.float32,
        )
        terminal_cost = np.where(has_metrics > 0.5, metrics_terminal_cost, expert_terminal_cost).astype(np.float32)
    else:
        terminal_cost = metrics_terminal_cost

    terminal_property_penalty = np.zeros_like(terminal_cost, dtype=np.float32)
    if args.terminal_property_mode == "hinge":
        qed_raw = np.asarray(terminal["qed"], dtype=np.float32)
        sa_raw = np.asarray(terminal["sa"], dtype=np.float32)
        complete_raw = np.asarray(terminal["complete"], dtype=np.float32)

        qed_scale = max(float(args.terminal_qed_scale), 1e-6)
        sa_scale = max(float(args.terminal_sa_scale), 1e-6)
        qed_gap = np.zeros_like(terminal_property_penalty)
        sa_gap = np.zeros_like(terminal_property_penalty)
        incomplete_gap = np.zeros_like(terminal_property_penalty)

        qed_mask = np.isfinite(qed_raw)
        sa_mask = np.isfinite(sa_raw)
        complete_mask = np.isfinite(complete_raw)
        qed_gap[qed_mask] = np.maximum((float(args.terminal_qed_threshold) - qed_raw[qed_mask]) / qed_scale, 0.0) ** 2
        sa_gap[sa_mask] = np.maximum((float(args.terminal_sa_threshold) - sa_raw[sa_mask]) / sa_scale, 0.0) ** 2
        incomplete_gap[complete_mask] = np.maximum(1.0 - complete_raw[complete_mask], 0.0)

        terminal_property_penalty = (
            float(args.w_terminal_qed_hinge) * qed_gap
            + float(args.w_terminal_sa_hinge) * sa_gap
            + float(args.w_terminal_incomplete_hinge) * incomplete_gap
        ).astype(np.float32)
        terminal_cost = (terminal_cost + terminal_property_penalty).astype(np.float32)

    offset = 0
    gamma_summary = {}
    for traj_idx, tr in enumerate(trajectories):
        T = int(tr["ligand_pos_traj"].size(0))
        zseg = {key: state_z[key][offset : offset + T] for key in state_z}
        local_safety = (
            np.maximum(zseg["clash"], 0.0)
            + 1.5 * np.maximum(zseg["severe_clash"], 0.0)
            + 0.5 * np.maximum(zseg["overburied"], 0.0)
            + 0.25 * np.maximum(zseg["center_drift"], 0.0)
        ).astype(np.float32)
        local_physics = (
            float(args.w_physics_clash) * np.maximum(zseg["clash"], 0.0)
            + float(args.w_physics_severe_clash) * np.maximum(zseg["severe_clash"], 0.0)
            + float(args.w_physics_overburied) * np.maximum(zseg["overburied"], 0.0)
            + float(args.w_physics_center_drift) * np.maximum(zseg["center_drift"], 0.0)
            - float(args.w_physics_contact) * zseg["contact"]
        ).astype(np.float32)
        local_cost = (
            float(args.w_local_expert) * zseg["expert_cost"]
            + float(args.w_local_safety) * local_safety
            + float(args.w_local_step) * zseg["step_norm"]
        ).astype(np.float32)
        local_physics_cost = (local_cost + float(args.w_local_physics) * local_physics).astype(np.float32)
        terminal_physics_cost = float(terminal_cost[traj_idx] + float(args.w_terminal_physics) * local_physics[-1])
        tr["U_affinity_local"] = torch.from_numpy(local_cost)
        tr["trajectory_affinity_cost"] = float(terminal_cost[traj_idx])
        tr["trajectory_affinity_score"] = float(-terminal_cost[traj_idx])
        tr["G_affinity_direct"] = torch.from_numpy(
            (np.cumsum(local_cost[::-1])[::-1] / max(T, 1) + terminal_cost[traj_idx]).astype(np.float32)
        )
        tr["local_safety_cost"] = torch.from_numpy(local_safety)
        tr["local_physics_cost"] = torch.from_numpy(local_physics)
        tr["U_affinity_physics_local"] = torch.from_numpy(local_physics_cost)
        tr["trajectory_affinity_physics_cost"] = terminal_physics_cost
        tr["trajectory_affinity_physics_score"] = float(-terminal_physics_cost)
        tr["G_affinity_physics_direct"] = torch.from_numpy(
            (np.cumsum(local_physics_cost[::-1])[::-1] / max(T, 1) + terminal_physics_cost).astype(np.float32)
        )
        gamma_key = f"{float(tr['gamma']):g}"
        item = gamma_summary.setdefault(
            gamma_key,
            {
                "n": 0,
                "terminal_cost": [],
                "terminal_physics_cost": [],
                "terminal_property_penalty": [],
                "complete": [],
                "qed": [],
                "sa": [],
                "vina_dock": [],
                "expert_score": [],
                "local_safety": [],
                "local_physics": [],
            },
        )
        item["n"] += 1
        item["terminal_cost"].append(float(terminal_cost[traj_idx]))
        item["terminal_physics_cost"].append(float(terminal_physics_cost))
        item["terminal_property_penalty"].append(float(terminal_property_penalty[traj_idx]))
        item["complete"].append(float(complete[traj_idx]))
        item["qed"].append(safe_float(tr["terminal_metrics"].get("qed")))
        item["sa"].append(safe_float(tr["terminal_metrics"].get("sa")))
        item["vina_dock"].append(safe_float(tr["terminal_metrics"].get("vina_dock")))
        item["expert_score"].append(float(tr["expert_affinity_score"].float().mean().item()))
        item["local_safety"].append(positive_mean(local_safety))
        item["local_physics"].append(float(np.nanmean(local_physics)))
        offset += T

    for item in gamma_summary.values():
        for key in list(item):
            if key == "n":
                continue
            arr = np.asarray(item[key], dtype=float)
            mask = np.isfinite(arr)
            item[key] = float(arr[mask].mean()) if int(mask.sum()) else float("nan")

    bank = {
        "trajectories": trajectories,
        "normalization": {"terminal": terminal_norm, "state": state_norm},
        "args": vars(args),
        "note": (
            "Affinity-HJB bank. U_affinity_local/G_affinity_direct preserves the original dense expert-affinity "
            "target. U_affinity_physics_local/G_affinity_physics_direct adds a configurable steric/contact "
            "running cost and optional terminal physics boundary term."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    summary = {
        "output": str(output),
        "num_trajectories": len(trajectories),
        "num_states": int(sum(int(t["ligand_pos_traj"].size(0)) for t in trajectories)),
        "terminal_source": str(args.terminal_source),
        "gamma_summary": gamma_summary,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Affinity-HJB Bank",
        "",
        f"- Output: `{output}`",
        f"- Terminal source: `{args.terminal_source}`",
        f"- Trajectories: {summary['num_trajectories']}",
        f"- States: {summary['num_states']}",
        "",
        "| gamma | n | terminal cost | terminal physics cost | complete | QED | SA | VinaDock | expert score | local safety | local physics |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gamma_key, item in sorted(gamma_summary.items(), key=lambda kv: float(kv[0])):
        lines.append(
            f"| {gamma_key} | {item['n']} | {item['terminal_cost']:.3f} | {item['terminal_physics_cost']:.3f} | {item['complete']:.3f} | "
            f"{item['qed']:.3f} | {item['sa']:.3f} | {item['vina_dock']:.3f} | "
            f"{item['expert_score']:.3f} | {item['local_safety']:.3f} | {item['local_physics']:.3f} |"
        )
    lines += [
        "",
        "Cost sign note: lower terminal cost is better; Vina values are lower-is-better before z-scoring.",
        "",
    ]
    output.with_suffix(".summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean, scatter_sum
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.hjb_actor_model import build_hjb_actor_from_checkpoint
from models.hjb_value_model import build_hjb_value_model_from_checkpoint
from models.molopt_score_model_guide import ScorePosNet3D_guided_flow
from sample_atom_num import predict_atom_num
from utils.evaluation import atom_num


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]])
    return [np.stack(step_v) for step_v in all_step_v]


def unbatch_v_logits_traj(ligand_v_logits_traj, n_data, ligand_cum_atoms):
    all_step_logits = [[] for _ in range(n_data)]
    for logits in ligand_v_logits_traj:
        logits_array = logits.cpu().numpy().astype(np.float32)
        for k in range(n_data):
            all_step_logits[k].append(
                logits_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
            )
    return [np.stack(step_logits) for step_logits in all_step_logits]


def unbatch_velocity_trace(velocity_trace, n_data):
    if not velocity_trace:
        return []
    sample_traces = [[] for _ in range(n_data)]
    graph_keys = {
        "pred_affinity",
        "fm_norm",
        "binding_norm",
        "total_norm",
        "binding_to_fm_ratio",
        "fm_binding_cos",
        "fm_total_cos",
        "base_norm",
        "hjb_score",
        "hjb_schedule",
        "hjb_late_taper",
        "hjb_raw_norm",
        "hjb_projected_norm",
        "hjb_projected_to_raw_ratio",
        "hjb_raw_scale",
        "hjb_scale",
        "hjb_scale_clamped",
        "hjb_actor_raw_norm",
        "hjb_actor_neggrad_cos",
        "hjb_actor_base_cos",
        "hjb_norm",
        "hjb_to_base_ratio",
        "hjb_effective_target_fraction",
        "hjb_raw_base_cos",
        "hjb_base_cos",
        "hjb_mobility_delta_norm",
        "hjb_mobility_to_raw_ratio",
        "hjb_barrier_delta_norm",
        "hjb_barrier_to_hjb_ratio",
        "hjb_fc_delta_norm",
        "hjb_fc_active_count",
        "hjb_fc_violation_sum",
        "hjb_gate_choice",
        "hjb_gate_locked",
        "hjb_gate_mean_d_hjb_base_cos",
        "hjb_gate_d_hjb_base_cos",
        "hjb_gate_e_hjb_base_cos",
        "hjb_gate_d_to_base_ratio",
        "hjb_gate_e_to_base_ratio",
        "physics_score",
        "physics_schedule",
        "physics_raw_norm",
        "physics_projected_norm",
        "physics_projected_to_raw_ratio",
        "physics_raw_scale",
        "physics_scale",
        "physics_scale_clamped",
        "physics_norm",
        "physics_to_base_ratio",
        "physics_effective_target_fraction",
        "physics_target_base_ratio",
        "physics_raw_base_cos",
        "physics_base_cos",
        "physics_hjb_cos",
        "local_affinity_schedule",
        "local_affinity_raw_norm",
        "local_affinity_projected_norm",
        "local_affinity_raw_scale",
        "local_affinity_scale",
        "local_affinity_scale_clamped",
        "local_affinity_norm",
        "local_affinity_to_base_ratio",
        "local_affinity_effective_target_fraction",
        "local_affinity_raw_base_cos",
        "local_affinity_base_cos",
        "branch_response_gate",
        "branch_response_schedule",
        "branch_response_affinity",
        "branch_response_clash",
        "branch_response_affinity_std",
        "branch_response_clash_std",
        "branch_response_norm",
        "flow_sde_D_mean",
        "flow_sde_score_norm",
        "flow_sde_drift_norm",
        "flow_sde_sigma_mean",
    }
    for step in velocity_trace:
        for k in range(n_data):
            item = {}
            for key, value in step.items():
                if key == "branch_response_coefficients":
                    item[key] = None if value is None else value[k].tolist()
                elif key in graph_keys:
                    item[key] = None if value is None else float(value[k].item())
                else:
                    item[key] = value
            sample_traces[k].append(item)
    return sample_traces


def load_sampling_data(config, subset_name):
    ckpt = torch.load(config.model.checkpoint, map_location="cpu", weights_only=False)
    ckpt["config"]["data"]["path"] = config.data.path
    ckpt["config"]["data"]["split"] = config.data.split
    if "name" in config.data:
        ckpt["config"]["data"]["name"] = config.data.name

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose(
        [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
        ]
    )
    dataset, subsets = get_dataset(config=ckpt["config"].data, transform=transform)
    if subset_name not in subsets:
        raise KeyError(f"Subset {subset_name!r} not found. Available: {list(subsets.keys())}")
    return ckpt, dataset, subsets[subset_name], protein_featurizer, ligand_featurizer


def load_pocket_info(data, subset_name, device):
    dataset_path = Path("./data/atom_num_dataset.pkl")
    with dataset_path.open("rb") as f:
        atom_num_dataset = pickle.load(f)
    if subset_name in atom_num_dataset:
        subset_info = atom_num_dataset[subset_name]
    elif subset_name == "val" and "train" in atom_num_dataset:
        subset_info = atom_num_dataset["train"]
    else:
        subset_info = atom_num_dataset.get("test", {})
    key = data["ligand_filename"][0:-4]
    if key not in subset_info:
        return None
    item = subset_info[key]
    return torch.tensor(
        [item["pocket_atom_num"], item["volume"], item["area"], item["pocket_size"]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def sample_ligands(
    model,
    data,
    num_samples,
    batch_size,
    device,
    num_steps,
    pos_only,
    center_pos_mode,
    sample_num_atoms,
    noise,
    pocket_info,
    atom_predictor_ckpt,
    atom_num_std,
    pos_grad_w,
    v_grad_w,
    trace_velocity,
    hjb_guidance,
    physics_guidance,
    local_affinity_guidance,
    branch_response_guidance,
    flow_sde,
    categorical_velocity_mode,
    categorical_state_mode,
    categorical_temperature_start,
    categorical_temperature_end,
    categorical_velocity_seed,
):
    all_pred_pos, all_pred_v, all_initial_v = [], [], []
    all_pred_pos_traj, all_pred_v_traj, all_pred_v_logits_traj = [], [], []
    all_velocity_trace = []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch), desc="sample batches"):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == "prior":
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == "range":
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == "ref":
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == "predict":
                ligand_num_atoms = predict_atom_num(pocket_info, device, atom_predictor_ckpt, n_data, atom_num_std)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            else:
                raise ValueError(f"Unknown sample_num_atoms: {sample_num_atoms}")

            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            init_ligand_pos = center_pos[batch_ligand] + torch.randn(len(batch_ligand), 3, device=device)
            init_ligand_v = (
                batch.ligand_atom_feature_full
                if pos_only
                else torch.randint(0, model.num_classes, (len(batch_ligand),), device=device)
            )

            result = model.sample_guided_flow_VP(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                noise=noise,
                pos_grad_w=pos_grad_w,
                v_grad_w=v_grad_w,
                trace_velocity=trace_velocity,
                hjb_guidance=hjb_guidance,
                physics_guidance=physics_guidance,
                local_affinity_guidance=local_affinity_guidance,
                branch_response_guidance=branch_response_guidance,
                flow_sde={**flow_sde, "seed": int(flow_sde["seed"]) + i * 100003},
                categorical_transition={
                    "velocity_mode": categorical_velocity_mode,
                    "state_mode": categorical_state_mode,
                    "temperature_start": categorical_temperature_start,
                    "temperature_end": categorical_temperature_end,
                    "velocity_seed": categorical_velocity_seed,
                    "stream_ids": list(range(current_i, current_i + n_data)),
                },
            )

            ligand_pos, ligand_v = result["pos"], result["v"]
            ligand_pos_traj, ligand_v_traj = result["pos_traj"], result["v_traj"]
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [
                ligand_pos_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]])
            all_pred_pos_traj += [np.stack(step_pos) for step_pos in all_step_pos]

            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]] for k in range(n_data)]
            initial_v_array = result["categorical_reference_v"].cpu().numpy()
            all_initial_v += [
                initial_v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]
            all_pred_v_traj += unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_logits_traj += unbatch_v_logits_traj(
                result.get("v_logits_traj", []), n_data, ligand_cum_atoms
            )
            all_velocity_trace += unbatch_velocity_trace(result.get("velocity_trace", []), n_data)
        time_list.append(time.time() - t1)
        current_i += n_data
    return (
        all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj,
        all_pred_v_logits_traj, all_initial_v, time_list, all_velocity_trace,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-i", "--data_id", type=int, default=0)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--flow_sde_mode", choices=("none", "equivalent"), default="none")
    parser.add_argument("--flow_sde_dmax", type=float, default=0.0)
    parser.add_argument(
        "--categorical_velocity_mode", type=str, default="sample",
        choices=("sample", "rao_blackwell", "stateless_gumbel"),
        help="Monte Carlo or Rao--Blackwellized categorical endpoint field.",
    )
    parser.add_argument(
        "--categorical_state_mode", type=str, default="hard",
        choices=("hard", "simplex"),
        help="Intermediate hard atom labels or continuous simplex states.",
    )
    parser.add_argument("--categorical_temperature_start", type=float, default=1.0)
    parser.add_argument("--categorical_temperature_end", type=float, default=1.0)
    parser.add_argument("--flow_sde_seed", type=int, default=None)
    parser.add_argument("--v_grad_w", type=float, default=0.0)
    parser.add_argument("--pos_grad_w", type=float, default=350.0)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace_velocity_components", action="store_true")
    parser.add_argument("--hjb_value_checkpoint", type=str, default=None)
    parser.add_argument("--hjb_value_component", type=str, default="total")
    parser.add_argument(
        "--hjb_value_time_mode",
        type=str,
        default="vp_time",
        choices=("vp_time", "generation_progress"),
        help="Time coordinate passed to trajectory-trained value models. vp_time preserves legacy behavior.",
    )
    parser.add_argument("--hjb_target_base_ratio", type=float, default=0.0)
    parser.add_argument("--hjb_target_base_ratio_max_scale", type=float, default=100.0)
    parser.add_argument(
        "--hjb_control_mode",
        type=str,
        default="normalized",
        choices=("normalized", "control_cost"),
        help="How to convert the value gradient into a residual velocity.",
    )
    parser.add_argument(
        "--hjb_control_cost_weight",
        type=float,
        default=0.0,
        help="Quadratic control-cost coefficient c for control_cost mode.",
    )
    parser.add_argument(
        "--hjb_control_cap_ratio",
        type=float,
        default=None,
        help="Trust-region cap for control_cost mode relative to the base velocity. Defaults to --hjb_target_base_ratio.",
    )
    parser.add_argument(
        "--hjb_score_ratio_mode",
        type=str,
        default="none",
        choices=("none", "sigmoid"),
        help="Optionally set the HJB residual ratio from the predicted value score.",
    )
    parser.add_argument("--hjb_score_ratio_min", type=float, default=0.0)
    parser.add_argument("--hjb_score_ratio_max", type=float, default=0.0)
    parser.add_argument("--hjb_score_ratio_center", type=float, default=0.0)
    parser.add_argument("--hjb_score_ratio_temperature", type=float, default=0.5)
    parser.add_argument(
        "--hjb_actor_checkpoint",
        type=str,
        default="",
        help="Optional residual actor checkpoint. If set, the actor replaces raw -grad S before projection and normalization.",
    )
    parser.add_argument(
        "--hjb_multi_head_mode",
        type=str,
        default="none",
        choices=("none", "weighted", "pcgrad"),
        help="Combine gradients from multiple value heads at sampling time.",
    )
    parser.add_argument(
        "--hjb_multi_head_components",
        type=str,
        default="",
        help="Comma-separated value heads used by --hjb_multi_head_mode.",
    )
    parser.add_argument(
        "--hjb_multi_head_weights",
        type=str,
        default="",
        help="Comma-separated weights for --hjb_multi_head_components.",
    )
    parser.add_argument("--hjb_atom_value_checkpoint", type=str, default="")
    parser.add_argument("--hjb_atom_value_component", type=str, default="total")
    parser.add_argument("--hjb_atom_target_base_ratio", type=float, default=0.0)
    parser.add_argument("--hjb_atom_target_base_ratio_max_scale", type=float, default=10.0)
    parser.add_argument("--hjb_feas_checkpoint", type=str, default="")
    parser.add_argument("--hjb_feas_component", type=str, default="total")
    parser.add_argument("--hjb_feas_target_base_ratio", type=float, default=0.0)
    parser.add_argument("--hjb_feas_target_base_ratio_max_scale", type=float, default=100.0)
    parser.add_argument(
        "--hjb_feas_projection_mode",
        type=str,
        default="remove_negative_parallel",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument(
        "--hjb_feas_gate_mode",
        type=str,
        default="none",
        choices=("none", "soft", "soft_geom"),
        help="Use a learned feasibility value to gate the main residual and/or a feasibility residual.",
    )
    parser.add_argument("--hjb_feas_gate_threshold", type=float, default=0.0)
    parser.add_argument("--hjb_feas_gate_temperature", type=float, default=0.50)
    parser.add_argument("--hjb_feas_gate_strength", type=float, default=0.0)
    parser.add_argument("--hjb_feas_geom_clash_threshold", type=float, default=0.02)
    parser.add_argument("--hjb_feas_geom_overburied_threshold", type=float, default=0.03)
    parser.add_argument("--hjb_feas_geom_temperature", type=float, default=0.02)
    parser.add_argument(
        "--hjb_disable_normalization",
        action="store_true",
        help="Use the projected/raw -grad S direction directly instead of rescaling it to a base-velocity ratio.",
    )
    parser.add_argument(
        "--hjb_raw_gradient_scale",
        type=float,
        default=1.0,
        help="Multiplier used only with --hjb_disable_normalization.",
    )
    parser.add_argument(
        "--hjb_projection_mode",
        type=str,
        default="remove_negative_parallel",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument("--hjb_t0", type=float, default=0.65)
    parser.add_argument("--hjb_sigmoid_k", type=float, default=14.0)
    parser.add_argument(
        "--hjb_late_taper_start",
        type=float,
        default=1.0,
        help="Generation progress at which a smooth terminal taper reaches 0.5; 1.0 disables tapering.",
    )
    parser.add_argument("--hjb_late_taper_k", type=float, default=30.0)
    parser.add_argument(
        "--local_affinity_target_base_ratio",
        type=float,
        default=0.0,
        help="Add a normalized local affinity-gradient residual with this target ratio to the base velocity.",
    )
    parser.add_argument("--local_affinity_target_base_ratio_max_scale", type=float, default=100.0)
    parser.add_argument(
        "--local_affinity_projection_mode",
        type=str,
        default="none",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument("--local_affinity_t0", type=float, default=0.50)
    parser.add_argument("--local_affinity_sigmoid_k", type=float, default=14.0)
    parser.add_argument("--branch_response_checkpoint", type=str, default="")
    parser.add_argument("--branch_response_target_base_ratio", type=float, default=0.0)
    parser.add_argument("--branch_response_t0", type=float, default=0.50)
    parser.add_argument("--branch_response_uncertainty_k", type=float, default=0.50)
    parser.add_argument("--branch_response_min_improvement", type=float, default=0.05)
    parser.add_argument("--branch_response_max_clash_increase", type=float, default=0.25)
    parser.add_argument(
        "--branch_response_affinity_only",
        action="store_true",
        help="Use only the deployment-validated affinity response; do not read or gate on a learned clash head.",
    )
    parser.add_argument("--branch_response_contact_barrier", action="store_true")
    parser.add_argument("--branch_response_bond_projection", action="store_true")
    parser.add_argument(
        "--branch_response_active_steps",
        type=str,
        default="",
        help="Comma-separated zero-based solver steps at which a calibrated transverse controller may act.",
    )
    parser.add_argument("--branch_response_min_probability", type=float, default=0.60)
    parser.add_argument("--branch_response_softmax_temperature", type=float, default=0.15)
    parser.add_argument("--branch_response_max_probability_std", type=float, default=0.15)
    parser.add_argument("--branch_response_max_abs_z", type=float, default=5.0)
    parser.add_argument(
        "--branch_response_pure_safety_projection",
        action="store_true",
        help="Project only an existing branch residual; never add an independent barrier force.",
    )
    parser.add_argument("--branch_response_barrier_min_dist", type=float, default=1.60)
    parser.add_argument("--branch_response_barrier_active_dist", type=float, default=2.10)
    parser.add_argument("--branch_response_barrier_kappa", type=float, default=0.50)
    parser.add_argument("--branch_response_barrier_strength", type=float, default=1.00)
    parser.add_argument(
        "--hjb_mobility_mode",
        type=str,
        default="none",
        choices=("none", "protein_normal"),
        help="Optional preconditioner applied to the raw value-gradient residual proposal before projection.",
    )
    parser.add_argument("--hjb_mobility_strength", type=float, default=0.0)
    parser.add_argument("--hjb_mobility_radius", type=float, default=2.40)
    parser.add_argument("--hjb_mobility_softness", type=float, default=0.35)
    parser.add_argument(
        "--hjb_mobility_all_normals",
        action="store_true",
        help="Dampen both inward and outward protein-normal components instead of only inward components.",
    )
    parser.add_argument("--hjb_barrier_filter", action="store_true")
    parser.add_argument("--hjb_barrier_min_dist", type=float, default=1.60)
    parser.add_argument("--hjb_barrier_active_dist", type=float, default=2.10)
    parser.add_argument("--hjb_barrier_kappa", type=float, default=0.50)
    parser.add_argument("--hjb_barrier_strength", type=float, default=1.00)
    parser.add_argument("--hjb_feasible_corridor_filter", action="store_true")
    parser.add_argument(
        "--hjb_fc_mode",
        type=str,
        default="clash",
        choices=("clash", "clash_overburied", "overburied", "full"),
        help="Physical constraints used by the feasible-corridor residual filter.",
    )
    parser.add_argument("--hjb_fc_eta", type=float, default=0.25)
    parser.add_argument("--hjb_fc_iters", type=int, default=2)
    parser.add_argument("--hjb_fc_clash_radius", type=float, default=1.75)
    parser.add_argument("--hjb_fc_severe_radius", type=float, default=1.45)
    parser.add_argument("--hjb_fc_overburied_radius", type=float, default=2.20)
    parser.add_argument("--hjb_fc_drift_radius", type=float, default=8.0)
    parser.add_argument(
        "--hjb_phys_ratio_gate_mode",
        type=str,
        default="none",
        choices=("none", "inverse"),
        help="Scale the HJB residual ratio by a physical-risk gate.",
    )
    parser.add_argument("--hjb_phys_ratio_gate_strength", type=float, default=0.0)
    parser.add_argument("--hjb_phys_gate_clash_weight", type=float, default=0.0)
    parser.add_argument("--hjb_phys_gate_severe_weight", type=float, default=0.0)
    parser.add_argument("--hjb_phys_gate_overburied_weight", type=float, default=0.0)
    parser.add_argument("--hjb_phys_gate_drift_weight", type=float, default=0.0)
    parser.add_argument("--hjb_phys_gate_clash_radius", type=float, default=1.75)
    parser.add_argument("--hjb_phys_gate_severe_radius", type=float, default=1.45)
    parser.add_argument("--hjb_phys_gate_overburied_radius", type=float, default=2.20)
    parser.add_argument("--hjb_phys_gate_drift_radius", type=float, default=8.0)
    parser.add_argument("--hjb_adaptive_de_gate", action="store_true")
    parser.add_argument("--hjb_gate_warmup_steps", type=int, default=15)
    parser.add_argument("--hjb_gate_threshold", type=float, default=0.22946724712673572)
    parser.add_argument("--hjb_gate_d_ratio", type=float, default=0.10)
    parser.add_argument("--hjb_gate_e_ratio", type=float, default=0.20)
    parser.add_argument(
        "--hjb_gate_d_projection",
        type=str,
        default="remove_negative_parallel",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument(
        "--hjb_gate_e_projection",
        type=str,
        default="tangent_remove_negative_parallel",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument("--hjb_gate_d_physics_ratio", type=float, default=0.05)
    parser.add_argument("--hjb_gate_e_physics_ratio", type=float, default=0.0)
    parser.add_argument("--physics_target_base_ratio", type=float, default=0.0)
    parser.add_argument("--physics_target_base_ratio_max_scale", type=float, default=100.0)
    parser.add_argument(
        "--physics_projection_mode",
        type=str,
        default="remove_negative_parallel",
        choices=(
            "none",
            "positive_only",
            "remove_negative_parallel",
            "tangent",
            "tangent_remove_negative_parallel",
            "molecular_constraint",
            "molecular_constraint_remove_negative_parallel",
        ),
    )
    parser.add_argument("--physics_t0", type=float, default=0.50)
    parser.add_argument("--physics_sigmoid_k", type=float, default=14.0)
    parser.add_argument("--physics_clash_radius", type=float, default=1.75)
    parser.add_argument("--physics_severe_radius", type=float, default=1.45)
    parser.add_argument("--physics_overburied_radius", type=float, default=2.20)
    parser.add_argument("--physics_contact_dist", type=float, default=3.50)
    parser.add_argument("--physics_contact_sigma", type=float, default=0.75)
    parser.add_argument("--physics_steric_weight", type=float, default=1.0)
    parser.add_argument("--physics_severe_weight", type=float, default=2.0)
    parser.add_argument("--physics_overburied_weight", type=float, default=0.5)
    parser.add_argument("--physics_contact_weight", type=float, default=0.25)
    args = parser.parse_args()

    config = misc.load_config(args.config)
    if args.num_samples is not None:
        config.sample.num_samples = int(args.num_samples)
    effective_seed = config.sample.seed if args.seed is None else int(args.seed)
    misc.seed_all(effective_seed)
    flow_sde = {
        "mode": args.flow_sde_mode,
        "dmax": float(args.flow_sde_dmax),
        "seed": effective_seed + 7919 if args.flow_sde_seed is None else int(args.flow_sde_seed),
    }
    if args.noise and args.flow_sde_mode != "none":
        raise ValueError("--noise and --flow_sde_mode cannot be enabled together")
    subset_name = args.subset or config.data.subset
    result_path = args.result_path or config.output.result_path
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger("sampling_paflow_prior", log_dir=result_path)
    logger.info(config)
    logger.info(args)

    ckpt, _, subset, protein_featurizer, ligand_featurizer = load_sampling_data(config, subset_name)
    model = ScorePosNet3D_guided_flow(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device,
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    logger.info(f"Loaded original PAFlow prior-guided model: {config.model.checkpoint}")

    hjb_guidance = None
    if args.hjb_adaptive_de_gate and float(args.hjb_target_base_ratio) <= 0:
        args.hjb_target_base_ratio = float(args.hjb_gate_d_ratio)
    if args.hjb_value_checkpoint and float(args.hjb_target_base_ratio) > 0:
        hjb_ckpt = torch.load(args.hjb_value_checkpoint, map_location=args.device, weights_only=False)
        hjb_model = build_hjb_value_model_from_checkpoint(hjb_ckpt, args.device).to(args.device)
        hjb_model.eval()
        for param in hjb_model.parameters():
            param.requires_grad_(False)
        feas_model = None
        if args.hjb_feas_checkpoint:
            feas_ckpt = torch.load(args.hjb_feas_checkpoint, map_location=args.device, weights_only=False)
            feas_model = build_hjb_value_model_from_checkpoint(feas_ckpt, args.device).to(args.device)
            feas_model.eval()
            for param in feas_model.parameters():
                param.requires_grad_(False)
        atom_model = None
        if args.hjb_atom_value_checkpoint and float(args.hjb_atom_target_base_ratio) > 0:
            atom_ckpt = torch.load(args.hjb_atom_value_checkpoint, map_location=args.device, weights_only=False)
            atom_model = build_hjb_value_model_from_checkpoint(atom_ckpt, args.device).to(args.device)
            atom_model.eval()
            for param in atom_model.parameters():
                param.requires_grad_(False)
        actor_model = None
        if args.hjb_actor_checkpoint:
            actor_ckpt = torch.load(args.hjb_actor_checkpoint, map_location=args.device, weights_only=False)
            actor_model = build_hjb_actor_from_checkpoint(actor_ckpt, args.device).to(args.device)
            actor_model.eval()
            for param in actor_model.parameters():
                param.requires_grad_(False)
        hjb_guidance = {
            "model": hjb_model,
            "component": args.hjb_value_component,
            "value_time_mode": args.hjb_value_time_mode,
            "target_base_ratio": float(args.hjb_target_base_ratio),
            "max_scale": float(args.hjb_target_base_ratio_max_scale),
            "control_mode": args.hjb_control_mode,
            "control_cost_weight": float(args.hjb_control_cost_weight),
            "control_cap_ratio": (
                float(args.hjb_target_base_ratio)
                if args.hjb_control_cap_ratio is None
                else float(args.hjb_control_cap_ratio)
            ),
            "score_ratio_mode": args.hjb_score_ratio_mode,
            "score_ratio_min": float(args.hjb_score_ratio_min),
            "score_ratio_max": float(args.hjb_score_ratio_max),
            "score_ratio_center": float(args.hjb_score_ratio_center),
            "score_ratio_temperature": float(args.hjb_score_ratio_temperature),
            "actor_model": actor_model,
            "actor_checkpoint": args.hjb_actor_checkpoint,
            "multi_head_mode": args.hjb_multi_head_mode,
            "multi_head_components": args.hjb_multi_head_components,
            "multi_head_weights": args.hjb_multi_head_weights,
            "atom_model": atom_model,
            "atom_checkpoint": args.hjb_atom_value_checkpoint,
            "atom_component": args.hjb_atom_value_component,
            "atom_target_base_ratio": float(args.hjb_atom_target_base_ratio),
            "atom_max_scale": float(args.hjb_atom_target_base_ratio_max_scale),
            "feas_model": feas_model,
            "feas_checkpoint": args.hjb_feas_checkpoint,
            "feas_component": args.hjb_feas_component,
            "feas_target_base_ratio": float(args.hjb_feas_target_base_ratio),
            "feas_max_scale": float(args.hjb_feas_target_base_ratio_max_scale),
            "feas_projection_mode": args.hjb_feas_projection_mode,
            "feas_gate_mode": args.hjb_feas_gate_mode,
            "feas_gate_threshold": float(args.hjb_feas_gate_threshold),
            "feas_gate_temperature": float(args.hjb_feas_gate_temperature),
            "feas_gate_strength": float(args.hjb_feas_gate_strength),
            "feas_geom_clash_threshold": float(args.hjb_feas_geom_clash_threshold),
            "feas_geom_overburied_threshold": float(args.hjb_feas_geom_overburied_threshold),
            "feas_geom_temperature": float(args.hjb_feas_geom_temperature),
            "disable_normalization": bool(args.hjb_disable_normalization),
            "raw_gradient_scale": float(args.hjb_raw_gradient_scale),
            "projection_mode": args.hjb_projection_mode,
            "t0": float(args.hjb_t0),
            "sigmoid_k": float(args.hjb_sigmoid_k),
            "late_taper_start": float(args.hjb_late_taper_start),
            "late_taper_k": float(args.hjb_late_taper_k),
            "mobility_mode": args.hjb_mobility_mode,
            "mobility_strength": float(args.hjb_mobility_strength),
            "mobility_radius": float(args.hjb_mobility_radius),
            "mobility_softness": float(args.hjb_mobility_softness),
            "mobility_inward_only": not bool(args.hjb_mobility_all_normals),
            "barrier_filter": bool(args.hjb_barrier_filter),
            "barrier_min_dist": float(args.hjb_barrier_min_dist),
            "barrier_active_dist": float(args.hjb_barrier_active_dist),
            "barrier_kappa": float(args.hjb_barrier_kappa),
            "barrier_strength": float(args.hjb_barrier_strength),
            "feasible_corridor_filter": bool(args.hjb_feasible_corridor_filter),
            "fc_mode": args.hjb_fc_mode,
            "fc_eta": float(args.hjb_fc_eta),
            "fc_iters": int(args.hjb_fc_iters),
            "fc_clash_radius": float(args.hjb_fc_clash_radius),
            "fc_severe_radius": float(args.hjb_fc_severe_radius),
            "fc_overburied_radius": float(args.hjb_fc_overburied_radius),
            "fc_drift_radius": float(args.hjb_fc_drift_radius),
            "phys_ratio_gate_mode": args.hjb_phys_ratio_gate_mode,
            "phys_ratio_gate_strength": float(args.hjb_phys_ratio_gate_strength),
            "phys_ratio_gate_clash_weight": float(args.hjb_phys_gate_clash_weight),
            "phys_ratio_gate_severe_weight": float(args.hjb_phys_gate_severe_weight),
            "phys_ratio_gate_overburied_weight": float(args.hjb_phys_gate_overburied_weight),
            "phys_ratio_gate_drift_weight": float(args.hjb_phys_gate_drift_weight),
            "phys_ratio_gate_clash_radius": float(args.hjb_phys_gate_clash_radius),
            "phys_ratio_gate_severe_radius": float(args.hjb_phys_gate_severe_radius),
            "phys_ratio_gate_overburied_radius": float(args.hjb_phys_gate_overburied_radius),
            "phys_ratio_gate_drift_radius": float(args.hjb_phys_gate_drift_radius),
            "adaptive_de_gate": bool(args.hjb_adaptive_de_gate),
            "gate_warmup_steps": int(args.hjb_gate_warmup_steps),
            "gate_threshold": float(args.hjb_gate_threshold),
            "gate_d_ratio": float(args.hjb_gate_d_ratio),
            "gate_e_ratio": float(args.hjb_gate_e_ratio),
            "gate_d_projection": args.hjb_gate_d_projection,
            "gate_e_projection": args.hjb_gate_e_projection,
        }
        logger.info(
            "Loaded HJB residual guidance: checkpoint=%s component=%s time=%s ratio=%.4f projection=%s control=%s c=%.6g cap=%.4f t0=%.3f k=%.3f late_taper=(%.3f,%.1f) score_ratio=%s",
            args.hjb_value_checkpoint,
            args.hjb_value_component,
            args.hjb_value_time_mode,
            args.hjb_target_base_ratio,
            args.hjb_projection_mode,
            args.hjb_control_mode,
            args.hjb_control_cost_weight,
            float(args.hjb_target_base_ratio) if args.hjb_control_cap_ratio is None else float(args.hjb_control_cap_ratio),
            args.hjb_t0,
            args.hjb_sigmoid_k,
            args.hjb_late_taper_start,
            args.hjb_late_taper_k,
            args.hjb_score_ratio_mode,
        )
        if args.hjb_multi_head_mode != "none":
            logger.info(
                "Using multi-head value residual: mode=%s components=%s weights=%s",
                args.hjb_multi_head_mode,
                args.hjb_multi_head_components,
                args.hjb_multi_head_weights,
            )
        if actor_model is not None:
            logger.info(
                "Loaded HJB residual actor: checkpoint=%s",
                args.hjb_actor_checkpoint,
            )
        if feas_model is not None:
            logger.info(
                "Loaded feasibility value guidance: checkpoint=%s component=%s ratio=%.4f gate=%s threshold=%.4f temp=%.4f strength=%.3f projection=%s",
                args.hjb_feas_checkpoint,
                args.hjb_feas_component,
                args.hjb_feas_target_base_ratio,
                args.hjb_feas_gate_mode,
                args.hjb_feas_gate_threshold,
                args.hjb_feas_gate_temperature,
                args.hjb_feas_gate_strength,
                args.hjb_feas_projection_mode,
            )
        if atom_model is not None:
            logger.info(
                "Loaded atom-type value residual: checkpoint=%s component=%s ratio=%.4f max_scale=%.3f",
                args.hjb_atom_value_checkpoint,
                args.hjb_atom_value_component,
                args.hjb_atom_target_base_ratio,
                args.hjb_atom_target_base_ratio_max_scale,
            )
        if args.hjb_mobility_mode != "none" or args.hjb_barrier_filter or args.hjb_feasible_corridor_filter:
            logger.info(
                "Enabled HJB execution filters: mobility=(mode=%s strength=%.3f radius=%.3f softness=%.3f inward_only=%s) "
                "barrier=(enabled=%s min=%.3f active=%.3f kappa=%.3f strength=%.3f) "
                "fc=(enabled=%s mode=%s eta=%.3f iters=%d)",
                args.hjb_mobility_mode,
                args.hjb_mobility_strength,
                args.hjb_mobility_radius,
                args.hjb_mobility_softness,
                not bool(args.hjb_mobility_all_normals),
                bool(args.hjb_barrier_filter),
                args.hjb_barrier_min_dist,
                args.hjb_barrier_active_dist,
                args.hjb_barrier_kappa,
                args.hjb_barrier_strength,
                bool(args.hjb_feasible_corridor_filter),
                args.hjb_fc_mode,
                args.hjb_fc_eta,
                args.hjb_fc_iters,
            )
        if args.hjb_adaptive_de_gate:
            logger.info(
                "Enabled adaptive D/E HJB gate: warmup=%d threshold=%.6f D=(ratio %.3f, %s) E=(ratio %.3f, %s)",
                args.hjb_gate_warmup_steps,
                args.hjb_gate_threshold,
                args.hjb_gate_d_ratio,
                args.hjb_gate_d_projection,
                args.hjb_gate_e_ratio,
                args.hjb_gate_e_projection,
            )
    physics_guidance = None
    if args.hjb_adaptive_de_gate and float(args.physics_target_base_ratio) <= 0:
        args.physics_target_base_ratio = max(float(args.hjb_gate_d_physics_ratio), float(args.hjb_gate_e_physics_ratio))
    if float(args.physics_target_base_ratio) > 0:
        physics_guidance = {
            "target_base_ratio": float(args.physics_target_base_ratio),
            "max_scale": float(args.physics_target_base_ratio_max_scale),
            "projection_mode": args.physics_projection_mode,
            "t0": float(args.physics_t0),
            "sigmoid_k": float(args.physics_sigmoid_k),
            "clash_radius": float(args.physics_clash_radius),
            "severe_radius": float(args.physics_severe_radius),
            "overburied_radius": float(args.physics_overburied_radius),
            "contact_dist": float(args.physics_contact_dist),
            "contact_sigma": float(args.physics_contact_sigma),
            "steric_weight": float(args.physics_steric_weight),
            "severe_weight": float(args.physics_severe_weight),
            "overburied_weight": float(args.physics_overburied_weight),
            "contact_weight": float(args.physics_contact_weight),
            "adaptive_de_gate": bool(args.hjb_adaptive_de_gate),
            "gate_d_ratio": float(args.hjb_gate_d_physics_ratio),
            "gate_e_ratio": float(args.hjb_gate_e_physics_ratio),
        }
        logger.info(
            "Enabled physics residual: ratio=%.4f projection=%s t0=%.3f steric=%.3f severe=%.3f overburied=%.3f contact=%.3f",
            args.physics_target_base_ratio,
            args.physics_projection_mode,
            args.physics_t0,
            args.physics_steric_weight,
            args.physics_severe_weight,
            args.physics_overburied_weight,
            args.physics_contact_weight,
        )

    local_affinity_guidance = None
    if float(args.local_affinity_target_base_ratio) > 0:
        local_affinity_guidance = {
            "target_base_ratio": float(args.local_affinity_target_base_ratio),
            "max_scale": float(args.local_affinity_target_base_ratio_max_scale),
            "projection_mode": args.local_affinity_projection_mode,
            "t0": float(args.local_affinity_t0),
            "sigmoid_k": float(args.local_affinity_sigmoid_k),
        }
        logger.info(
            "Enabled normalized local-affinity residual: ratio=%.4f projection=%s t0=%.3f k=%.3f",
            args.local_affinity_target_base_ratio,
            args.local_affinity_projection_mode,
            args.local_affinity_t0,
            args.local_affinity_sigmoid_k,
        )

    branch_response_guidance = None
    if args.branch_response_checkpoint and float(args.branch_response_target_base_ratio) > 0:
        response_checkpoint = torch.load(args.branch_response_checkpoint, map_location="cpu", weights_only=False)
        if args.branch_response_affinity_only:
            if not bool(response_checkpoint.get("affinity_gate", False)):
                raise RuntimeError("Branch-response checkpoint did not pass its preregistered affinity gate")
        elif not bool(response_checkpoint.get("online_gate", False)):
            raise RuntimeError("Branch-response checkpoint did not pass its preregistered joint online gate")
        response_models = response_checkpoint["models"]
        if isinstance(response_models, dict):
            response_models = [model for models in response_models.values() for model in models]
        for response_model in response_models:
            response_model["feature_scale"] = torch.as_tensor(
                response_model["feature_scale"], device=args.device
            )
            response_model["weight"] = torch.as_tensor(
                response_model["weight"], device=args.device
            )
        branch_response_guidance = {
            "checkpoint": response_checkpoint,
            "target_base_ratio": float(args.branch_response_target_base_ratio),
            "t0": float(args.branch_response_t0),
            "uncertainty_k": float(args.branch_response_uncertainty_k),
            "min_improvement": float(args.branch_response_min_improvement),
            "max_clash_increase": float(args.branch_response_max_clash_increase),
            "affinity_only": bool(args.branch_response_affinity_only),
            "contact_barrier": bool(args.branch_response_contact_barrier),
            "bond_projection": bool(args.branch_response_bond_projection),
            "active_steps": [
                int(value) for value in args.branch_response_active_steps.split(",") if value.strip()
            ],
            "min_probability": float(args.branch_response_min_probability),
            "softmax_temperature": float(args.branch_response_softmax_temperature),
            "max_probability_std": float(args.branch_response_max_probability_std),
            "max_abs_z": float(args.branch_response_max_abs_z),
            "pure_safety_projection": bool(args.branch_response_pure_safety_projection),
            "barrier_min_dist": float(args.branch_response_barrier_min_dist),
            "barrier_active_dist": float(args.branch_response_barrier_active_dist),
            "barrier_kappa": float(args.branch_response_barrier_kappa),
            "barrier_strength": float(args.branch_response_barrier_strength),
        }
        logger.info(
            "Enabled intended-action response controller: checkpoint=%s ratio=%.3f t0=%.3f k=%.3f",
            args.branch_response_checkpoint, args.branch_response_target_base_ratio,
            args.branch_response_t0, args.branch_response_uncertainty_k,
        )

    data = subset[args.data_id]
    dataset_index = subset.indices[args.data_id] if hasattr(subset, "indices") else args.data_id
    effective_sample_num_atoms = config.sample.sample_num_atoms
    pocket_info = None
    if effective_sample_num_atoms == "predict":
        pocket_info = load_pocket_info(data, subset_name, args.device)
        if pocket_info is None:
            logger.warning("Atom-number metadata missing; falling back to ref atom count.")
            effective_sample_num_atoms = "ref"

    (
        pred_pos, pred_v, pred_pos_traj, pred_v_traj,
        pred_v_logits_traj, initial_v, time_list, velocity_trace,
    ) = sample_ligands(
        model=model,
        data=data,
        num_samples=config.sample.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=effective_sample_num_atoms,
        noise=args.noise,
        pocket_info=pocket_info,
        atom_predictor_ckpt=config.model.atom_predictor_ckpt if effective_sample_num_atoms == "predict" else None,
        atom_num_std=config.sample.num_atoms_std if effective_sample_num_atoms == "predict" else None,
        pos_grad_w=args.pos_grad_w,
        v_grad_w=args.v_grad_w,
        trace_velocity=args.trace_velocity_components,
        hjb_guidance=hjb_guidance,
        physics_guidance=physics_guidance,
        local_affinity_guidance=local_affinity_guidance,
        branch_response_guidance=branch_response_guidance,
        flow_sde=flow_sde,
        categorical_velocity_mode=args.categorical_velocity_mode,
        categorical_state_mode=args.categorical_state_mode,
        categorical_temperature_start=args.categorical_temperature_start,
        categorical_temperature_end=args.categorical_temperature_end,
        categorical_velocity_seed=effective_seed + int(args.data_id) * 1000003,
    )

    result = {
        "data": data,
        "pred_ligand_pos": pred_pos,
        "pred_ligand_v": pred_v,
        "pred_ligand_pos_traj": pred_pos_traj,
        "pred_ligand_v_traj": pred_v_traj,
        "pred_ligand_v_logits_traj": pred_v_logits_traj,
        "pred_ligand_initial_v": initial_v,
        "time": time_list,
        "velocity_trace": velocity_trace,
        "split_name": subset_name,
        "subset_index": int(args.data_id),
        "dataset_index": int(dataset_index),
        "sample_num_atoms_mode": effective_sample_num_atoms,
        "paflow_prior_guidance": {
            "pos_grad_w": float(args.pos_grad_w),
            "v_grad_w": float(args.v_grad_w),
            "trace_velocity": bool(args.trace_velocity_components),
            "paper_form": "VP_field + para_x * grad log p(y=1|m_t) * pos_grad_w",
        },
        "flow_sde": flow_sde,
        "categorical_velocity_mode": args.categorical_velocity_mode,
        "categorical_state_mode": args.categorical_state_mode,
        "categorical_temperature_start": float(args.categorical_temperature_start),
        "categorical_temperature_end": float(args.categorical_temperature_end),
        "categorical_velocity_seed": effective_seed + int(args.data_id) * 1000003,
        "hjb_residual_guidance": None
        if hjb_guidance is None
        else {
            "checkpoint": args.hjb_value_checkpoint,
            "component": args.hjb_value_component,
            "value_time_mode": args.hjb_value_time_mode,
            "actor_checkpoint": args.hjb_actor_checkpoint,
            "target_base_ratio": float(args.hjb_target_base_ratio),
            "target_base_ratio_max_scale": float(args.hjb_target_base_ratio_max_scale),
            "control_mode": args.hjb_control_mode,
            "control_cost_weight": float(args.hjb_control_cost_weight),
            "control_cap_ratio": (
                float(args.hjb_target_base_ratio)
                if args.hjb_control_cap_ratio is None
                else float(args.hjb_control_cap_ratio)
            ),
            "score_ratio_mode": args.hjb_score_ratio_mode,
            "score_ratio_min": float(args.hjb_score_ratio_min),
            "score_ratio_max": float(args.hjb_score_ratio_max),
            "score_ratio_center": float(args.hjb_score_ratio_center),
            "score_ratio_temperature": float(args.hjb_score_ratio_temperature),
            "multi_head_mode": args.hjb_multi_head_mode,
            "multi_head_components": args.hjb_multi_head_components,
            "multi_head_weights": args.hjb_multi_head_weights,
            "atom_checkpoint": args.hjb_atom_value_checkpoint,
            "atom_component": args.hjb_atom_value_component,
            "atom_target_base_ratio": float(args.hjb_atom_target_base_ratio),
            "atom_target_base_ratio_max_scale": float(args.hjb_atom_target_base_ratio_max_scale),
            "feas_checkpoint": args.hjb_feas_checkpoint,
            "feas_component": args.hjb_feas_component,
            "feas_target_base_ratio": float(args.hjb_feas_target_base_ratio),
            "feas_target_base_ratio_max_scale": float(args.hjb_feas_target_base_ratio_max_scale),
            "feas_projection_mode": args.hjb_feas_projection_mode,
            "feas_gate_mode": args.hjb_feas_gate_mode,
            "feas_gate_threshold": float(args.hjb_feas_gate_threshold),
            "feas_gate_temperature": float(args.hjb_feas_gate_temperature),
            "feas_gate_strength": float(args.hjb_feas_gate_strength),
            "feas_geom_clash_threshold": float(args.hjb_feas_geom_clash_threshold),
            "feas_geom_overburied_threshold": float(args.hjb_feas_geom_overburied_threshold),
            "feas_geom_temperature": float(args.hjb_feas_geom_temperature),
            "disable_normalization": bool(args.hjb_disable_normalization),
            "raw_gradient_scale": float(args.hjb_raw_gradient_scale),
            "projection_mode": args.hjb_projection_mode,
            "t0": float(args.hjb_t0),
            "sigmoid_k": float(args.hjb_sigmoid_k),
            "late_taper_start": float(args.hjb_late_taper_start),
            "late_taper_k": float(args.hjb_late_taper_k),
            "mobility_mode": args.hjb_mobility_mode,
            "mobility_strength": float(args.hjb_mobility_strength),
            "mobility_radius": float(args.hjb_mobility_radius),
            "mobility_softness": float(args.hjb_mobility_softness),
            "mobility_inward_only": not bool(args.hjb_mobility_all_normals),
            "barrier_filter": bool(args.hjb_barrier_filter),
            "barrier_min_dist": float(args.hjb_barrier_min_dist),
            "barrier_active_dist": float(args.hjb_barrier_active_dist),
            "barrier_kappa": float(args.hjb_barrier_kappa),
            "barrier_strength": float(args.hjb_barrier_strength),
            "feasible_corridor_filter": bool(args.hjb_feasible_corridor_filter),
            "fc_mode": args.hjb_fc_mode,
            "fc_eta": float(args.hjb_fc_eta),
            "fc_iters": int(args.hjb_fc_iters),
            "fc_clash_radius": float(args.hjb_fc_clash_radius),
            "fc_severe_radius": float(args.hjb_fc_severe_radius),
            "fc_overburied_radius": float(args.hjb_fc_overburied_radius),
            "fc_drift_radius": float(args.hjb_fc_drift_radius),
            "adaptive_de_gate": bool(args.hjb_adaptive_de_gate),
            "gate_warmup_steps": int(args.hjb_gate_warmup_steps),
            "gate_threshold": float(args.hjb_gate_threshold),
            "gate_d_ratio": float(args.hjb_gate_d_ratio),
            "gate_e_ratio": float(args.hjb_gate_e_ratio),
            "gate_d_projection": args.hjb_gate_d_projection,
            "gate_e_projection": args.hjb_gate_e_projection,
            "form": "dx = PAFlow_prior_dx + schedule * scaled_projected(-grad S)",
        },
        "physics_residual_guidance": None
        if physics_guidance is None
        else {
            "target_base_ratio": float(args.physics_target_base_ratio),
            "target_base_ratio_max_scale": float(args.physics_target_base_ratio_max_scale),
            "projection_mode": args.physics_projection_mode,
            "t0": float(args.physics_t0),
            "sigmoid_k": float(args.physics_sigmoid_k),
            "clash_radius": float(args.physics_clash_radius),
            "severe_radius": float(args.physics_severe_radius),
            "overburied_radius": float(args.physics_overburied_radius),
            "contact_dist": float(args.physics_contact_dist),
            "contact_sigma": float(args.physics_contact_sigma),
            "steric_weight": float(args.physics_steric_weight),
            "severe_weight": float(args.physics_severe_weight),
            "overburied_weight": float(args.physics_overburied_weight),
            "contact_weight": float(args.physics_contact_weight),
            "adaptive_de_gate": bool(args.hjb_adaptive_de_gate),
            "gate_d_ratio": float(args.hjb_gate_d_physics_ratio),
            "gate_e_ratio": float(args.hjb_gate_e_physics_ratio),
            "form": "dx += schedule * scaled_projected(-grad physical_potential)",
        },
        "local_affinity_residual_guidance": None
        if local_affinity_guidance is None
        else {
            "target_base_ratio": float(args.local_affinity_target_base_ratio),
            "target_base_ratio_max_scale": float(args.local_affinity_target_base_ratio_max_scale),
            "projection_mode": args.local_affinity_projection_mode,
            "t0": float(args.local_affinity_t0),
            "sigmoid_k": float(args.local_affinity_sigmoid_k),
            "form": "dx += schedule * normalized_projected(grad log p(y=1|m_t))",
        },
        "branch_response_guidance": None
        if branch_response_guidance is None
        else {
            "checkpoint": args.branch_response_checkpoint,
            "target_base_ratio": float(args.branch_response_target_base_ratio),
            "t0": float(args.branch_response_t0),
            "uncertainty_k": float(args.branch_response_uncertainty_k),
            "min_improvement": float(args.branch_response_min_improvement),
            "max_clash_increase": float(args.branch_response_max_clash_increase),
            "affinity_only": bool(args.branch_response_affinity_only),
            "contact_barrier": bool(args.branch_response_contact_barrier),
            "bond_projection": bool(args.branch_response_bond_projection),
            "active_steps": [
                int(value) for value in args.branch_response_active_steps.split(",") if value.strip()
            ],
            "min_probability": float(args.branch_response_min_probability),
            "softmax_temperature": float(args.branch_response_softmax_temperature),
            "max_probability_std": float(args.branch_response_max_probability_std),
            "max_abs_z": float(args.branch_response_max_abs_z),
            "pure_safety_projection": bool(args.branch_response_pure_safety_projection),
            "barrier_min_dist": float(args.branch_response_barrier_min_dist),
            "barrier_active_dist": float(args.branch_response_barrier_active_dist),
            "barrier_kappa": float(args.branch_response_barrier_kappa),
            "barrier_strength": float(args.branch_response_barrier_strength),
            "form": "continuous affinity-response coefficients in affinity/steric/flow basis, optionally projected by analytic geometry constraints",
        },
    }
    Path(result_path).mkdir(parents=True, exist_ok=True)
    torch.save(result, Path(result_path) / f"result_{args.data_id}.pt")
    logger.info(f"Saved {Path(result_path) / f'result_{args.data_id}.pt'}")


if __name__ == "__main__":
    main()

import argparse
import csv
import heapq
import os
import shutil
import sys

for _env_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env_name, '1')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.landscape_encoder import PAFlowLandscapeEncoder
from models.landscape_model import build_landscape_model
from models.molopt_score_model_energy_guide import ScorePosNet3D_guided_flow


class _SamplingPHController(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, mode='diag'):
        super().__init__()
        self.mode = str(mode)
        output_dim = 3 if self.mode == 'diag' else 9
        layers = []
        dim = int(input_dim)
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.mode == 'diag':
            return F.softplus(out)
        if self.mode == 'matrix':
            return out.view(out.shape[0], 3, 3)
        raise ValueError(f'Unsupported PH controller mode: {self.mode}')


class _SamplingResidualPathValue(nn.Module):
    def __init__(self, base_model, residual_model, residual_scale):
        super().__init__()
        self.base_model = base_model
        self.residual_model = residual_model
        self.residual_scale = float(residual_scale)

    def forward(self, x):
        return self.base_model(x) + self.residual_scale * self.residual_model(x)


class _ReplayGateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        dim = int(input_dim)
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ReplayRatioSelectorMLP(_ReplayGateMLP):
    pass


def _parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _zscore_tensor(values):
    values = values.float()
    finite = torch.isfinite(values)
    if finite.any():
        finite_values = values[finite]
        values = values.clone()
        values[~finite] = finite_values.max()
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    return torch.zeros_like(values)


def _load_pairwise_potential_tensors(
    topology_state_csv,
    quality_potential_csv,
    pair_potential_csv,
    num_topology_codes,
    num_quality_codes,
    device,
):
    topology_values = torch.full((int(num_topology_codes),), float('inf'), dtype=torch.float32)
    with open(topology_state_csv, newline='') as f:
        for row in csv.DictReader(f):
            code = int(row['code'])
            if 0 <= code < int(num_topology_codes) and row.get('value_to_go') not in {None, ''}:
                value = float(row['value_to_go'])
                if value < float(topology_values[code].item()):
                    topology_values[code] = value
    topology_values = _zscore_tensor(topology_values)

    quality_values = torch.zeros((int(num_quality_codes),), dtype=torch.float32)
    with open(quality_potential_csv, newline='') as f:
        for row in csv.DictReader(f):
            code = int(row['quality_code'])
            if 0 <= code < int(num_quality_codes):
                value_key = 'normalized_quality_value' if row.get('normalized_quality_value') not in {None, ''} else 'raw_quality_value'
                quality_values[code] = float(row[value_key])

    pair_values = torch.zeros((int(num_topology_codes), int(num_quality_codes)), dtype=torch.float32)
    with open(pair_potential_csv, newline='') as f:
        for row in csv.DictReader(f):
            topo_code = int(row['topology_code'])
            quality_code = int(row['quality_code'])
            if 0 <= topo_code < int(num_topology_codes) and 0 <= quality_code < int(num_quality_codes):
                pair_values[topo_code, quality_code] = float(row['pair_value'])

    return {
        'topology_values': topology_values.to(device),
        'quality_values': quality_values.to(device),
        'pair_values': pair_values.to(device),
    }


def _build_prob_transition_tensors(
    topology_state_csv,
    num_topology_codes,
    device,
    stride=20,
    smoothing=1e-4,
):
    """Build empirical code transition probabilities from saved FM trajectories."""
    num_codes = int(num_topology_codes)
    stride = max(int(stride), 1)
    counts = torch.zeros((num_codes, num_codes), dtype=torch.float32)
    trajectories = {}
    code_values = torch.full((num_codes,), float('inf'), dtype=torch.float32)
    with open(topology_state_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('code') in {None, ''}:
                continue
            code = int(row['code'])
            if not (0 <= code < num_codes):
                continue
            traj_key = (
                row.get('trajectory_id', ''),
                row.get('example_idx', ''),
                row.get('sample_idx', ''),
            )
            time_idx = int(float(row.get('time_index', len(trajectories.get(traj_key, [])))))
            trajectories.setdefault(traj_key, []).append((time_idx, code))
            if row.get('value_to_go') not in {None, ''}:
                value = float(row['value_to_go'])
                if value < float(code_values[code].item()):
                    code_values[code] = value

    transition_pairs = 0
    for items in trajectories.values():
        items = sorted(items, key=lambda x: x[0])
        for idx, (_, src) in enumerate(items):
            dst_idx = idx + stride
            if dst_idx >= len(items):
                continue
            dst = items[dst_idx][1]
            counts[src, dst] += 1.0
            transition_pairs += 1

    if smoothing > 0.0:
        counts += float(smoothing)
    row_sums = counts.sum(dim=-1, keepdim=True)
    empty_rows = row_sums.squeeze(-1) <= 0
    if bool(empty_rows.any()):
        counts[empty_rows] = torch.eye(num_codes, dtype=torch.float32)[empty_rows]
        row_sums = counts.sum(dim=-1, keepdim=True)
    transition = counts / row_sums.clamp_min(1e-8)
    code_values = _zscore_tensor(code_values)
    entropy = -(transition.clamp_min(1e-12) * transition.clamp_min(1e-12).log()).sum(dim=-1)
    info = {
        'topology_state_csv': topology_state_csv,
        'stride': stride,
        'smoothing': float(smoothing),
        'num_trajectories': len(trajectories),
        'transition_pairs': int(transition_pairs),
        'nonzero_edges': int((counts > float(smoothing)).sum().item()) if smoothing > 0.0 else int((counts > 0).sum().item()),
        'mean_entropy': float(entropy.mean().item()),
        'self_loop_ratio': float(torch.diag(transition).mean().item()),
    }
    return {
        'transition': transition.to(device),
        'code_values': code_values.to(device),
        'info': info,
    }


def _build_prob_transition_displacement_tensors(
    topology_state_csv,
    pose_bank_path,
    num_topology_codes,
    device,
    stride=20,
    smoothing=1e-4,
):
    num_codes = int(num_topology_codes)
    stride = max(int(stride), 1)
    counts = torch.zeros((num_codes, num_codes), dtype=torch.float32)
    disp_sum = torch.zeros((num_codes, num_codes, 3), dtype=torch.float32)
    code_values = torch.full((num_codes,), float('inf'), dtype=torch.float32)
    code_by_state = {}
    with open(topology_state_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('code') in {None, ''}:
                continue
            key = (
                int(row.get('trajectory_id', 0)),
                int(float(row.get('time_index', 0))),
            )
            code = int(row['code'])
            if 0 <= code < num_codes:
                code_by_state[key] = code
                if row.get('value_to_go') not in {None, ''}:
                    value = float(row['value_to_go'])
                    if value < float(code_values[code].item()):
                        code_values[code] = value

    bank = torch.load(pose_bank_path, map_location='cpu', weights_only=False)
    transition_pairs = 0
    displacement_norms = []
    for traj in bank.get('trajectories', []):
        trajectory_id = int(traj['trajectory_id'])
        states = traj.get('states', [])
        for idx, src_state in enumerate(states):
            dst_idx = idx + stride
            if dst_idx >= len(states):
                continue
            src_time = int(src_state.get('time_index', idx))
            dst_state = states[dst_idx]
            dst_time = int(dst_state.get('time_index', dst_idx))
            src_code = code_by_state.get((trajectory_id, src_time))
            dst_code = code_by_state.get((trajectory_id, dst_time))
            if src_code is None or dst_code is None:
                continue
            src_pos = src_state.get('ligand_pos')
            dst_pos = dst_state.get('ligand_pos')
            if not torch.is_tensor(src_pos) or not torch.is_tensor(dst_pos):
                continue
            delta_com = dst_pos.float().mean(dim=0) - src_pos.float().mean(dim=0)
            counts[src_code, dst_code] += 1.0
            disp_sum[src_code, dst_code] += delta_com
            displacement_norms.append(float(delta_com.norm().item()))
            transition_pairs += 1

    mean_disp = disp_sum / counts.clamp_min(1.0).unsqueeze(-1)
    transition_counts = counts.clone()
    if smoothing > 0.0:
        counts = counts + float(smoothing)
    row_sums = counts.sum(dim=-1, keepdim=True)
    empty_rows = row_sums.squeeze(-1) <= 0
    if bool(empty_rows.any()):
        counts[empty_rows] = torch.eye(num_codes, dtype=torch.float32)[empty_rows]
        row_sums = counts.sum(dim=-1, keepdim=True)
    transition = counts / row_sums.clamp_min(1e-8)
    code_values = _zscore_tensor(code_values)
    entropy = -(transition.clamp_min(1e-12) * transition.clamp_min(1e-12).log()).sum(dim=-1)
    info = {
        'topology_state_csv': topology_state_csv,
        'pose_bank_path': pose_bank_path,
        'stride': stride,
        'smoothing': float(smoothing),
        'transition_pairs': int(transition_pairs),
        'nonzero_edges': int((transition_counts > 0).sum().item()),
        'mean_entropy': float(entropy.mean().item()),
        'self_loop_ratio': float(torch.diag(transition).mean().item()),
        'mean_delta_com_norm': float(sum(displacement_norms) / max(len(displacement_norms), 1)),
    }
    return {
        'transition': transition.to(device),
        'code_values': code_values.to(device),
        'mean_com_displacement': mean_disp.to(device),
        'transition_counts': transition_counts.to(device),
        'info': info,
    }


def _label_bool(labels, key, default=False):
    if labels is None:
        return default
    return bool(labels.get(key, default))


def _label_float(labels, key):
    if labels is None or key not in labels or labels[key] is None:
        return None
    try:
        return float(labels[key])
    except (TypeError, ValueError):
        return None


def _compute_code_indices(model, x, device, batch_size):
    codes = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch_x = x[start:start + batch_size].to(device)
            outputs = model(batch_x)
            codes.append(outputs['code_indices'].detach().cpu())
    return torch.cat(codes, dim=0)


def _build_value_to_go(
    model,
    train_config,
    device,
    features_path=None,
    dataset_path=None,
    input_key=None,
    batch_size=8192,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
    transition_cost='one_minus_prob',
    good_vina_key='vina_min',
    good_vina_quantile=0.25,
    normalize='zscore',
):
    features_path = features_path or train_config.data.train_features
    dataset_path = dataset_path or getattr(train_config.data, 'train_landscape_dataset', None)
    input_key = input_key or train_config.model.input_key
    features = torch.load(features_path, map_location='cpu', weights_only=False)
    if dataset_path is None:
        raise ValueError('A landscape dataset is required to define good basins for value-to-go guidance.')
    landscape_dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)

    x = features[input_key].float()
    code_indices = _compute_code_indices(model, x, device=device, batch_size=int(batch_size))
    num_codes = int(model.num_codes)

    transition_index = features['transition_index']
    src_rows = transition_index['src_row'].long()
    dst_rows = transition_index['dst_row'].long()
    src_codes = code_indices.index_select(0, src_rows)
    dst_codes = code_indices.index_select(0, dst_rows)
    counts = torch.zeros((num_codes, num_codes), dtype=torch.float32)
    counts.index_put_((src_codes, dst_codes), torch.ones_like(src_codes, dtype=torch.float32), accumulate=True)
    row_sums = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    transition_prob = counts / row_sums

    trajectories = landscape_dataset.get('trajectories', [])
    labels_by_traj = {
        int(traj['trajectory_id']): (traj.get('final_labels') or {})
        for traj in trajectories
        if 'trajectory_id' in traj
    }
    traj_ids = features['trajectory_id'].long()
    is_anchor = features.get('is_anchor_state', torch.zeros_like(traj_ids, dtype=torch.bool)).bool()

    candidate_rows = []
    candidate_vina = []
    for row_idx in torch.nonzero(is_anchor, as_tuple=False).view(-1).tolist():
        labels = labels_by_traj.get(int(traj_ids[row_idx].item()), {})
        if not _label_bool(labels, 'complete', False):
            continue
        if not _label_bool(labels, 'recon_success', True):
            continue
        vina_value = _label_float(labels, good_vina_key)
        if vina_value is None:
            continue
        candidate_rows.append(row_idx)
        candidate_vina.append(vina_value)

    if len(candidate_rows) == 0:
        raise ValueError('No good-basin candidates found for value-to-go guidance.')

    if good_vina_quantile is not None and float(good_vina_quantile) > 0:
        sorted_vina = sorted(candidate_vina)
        q_index = min(
            len(sorted_vina) - 1,
            max(0, int(round((len(sorted_vina) - 1) * float(good_vina_quantile)))),
        )
        vina_threshold = sorted_vina[q_index]
    else:
        vina_threshold = float('inf')

    good_rows = [
        row for row, vina in zip(candidate_rows, candidate_vina)
        if vina <= vina_threshold
    ]
    good_codes = torch.unique(code_indices.index_select(0, torch.tensor(good_rows, dtype=torch.long)))
    if good_codes.numel() == 0:
        raise ValueError('Good-basin code set is empty after filtering.')

    energies = model.code_energies.detach().cpu().float()
    costs = torch.full((num_codes, num_codes), float('inf'), dtype=torch.float32)
    has_edge = counts > 0
    uphill = torch.relu(energies.view(1, -1) - energies.view(-1, 1))
    if transition_cost == 'one_minus_prob':
        transition_penalty = 1.0 - transition_prob
    elif transition_cost == 'neg_log_prob':
        transition_penalty = -torch.log(transition_prob.clamp_min(1e-8))
    else:
        raise ValueError(f'Unsupported transition cost: {transition_cost}')

    if float(gamma) != 0.0:
        codebook = model.quantizer.codebook.detach().cpu().float()
        prototype_distance = torch.cdist(codebook, codebook, p=2)
        finite_dist = prototype_distance[has_edge]
        if finite_dist.numel() > 0:
            prototype_distance = prototype_distance / finite_dist.mean().clamp_min(1e-6)
    else:
        prototype_distance = torch.zeros_like(uphill)

    edge_cost = (
        float(alpha) * uphill
        + float(beta) * transition_penalty
        + float(gamma) * prototype_distance
    )
    costs[has_edge] = edge_cost[has_edge]

    reverse_edges = [[] for _ in range(num_codes)]
    edge_src, edge_dst = torch.nonzero(torch.isfinite(costs), as_tuple=True)
    for src, dst in zip(edge_src.tolist(), edge_dst.tolist()):
        reverse_edges[dst].append((src, float(costs[src, dst].item())))

    values = torch.full((num_codes,), float('inf'), dtype=torch.float32)
    heap = []
    for code in good_codes.tolist():
        values[int(code)] = 0.0
        heapq.heappush(heap, (0.0, int(code)))
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > float(values[node].item()):
            continue
        for prev, cost in reverse_edges[node]:
            new_dist = dist + cost
            if new_dist < float(values[prev].item()):
                values[prev] = new_dist
                heapq.heappush(heap, (new_dist, prev))

    finite = torch.isfinite(values)
    if not finite.all():
        replacement = values[finite].max() + float(beta) if finite.any() else torch.tensor(1.0)
        values[~finite] = replacement

    raw_values = values.clone()
    next_codes = torch.arange(num_codes, dtype=torch.long)
    for src in range(num_codes):
        dst_candidates = torch.nonzero(torch.isfinite(costs[src]), as_tuple=False).view(-1)
        if dst_candidates.numel() == 0 or float(raw_values[src].item()) == 0.0:
            continue
        candidate_cost = costs[src].index_select(0, dst_candidates) + raw_values.index_select(0, dst_candidates)
        best_idx = int(torch.argmin(candidate_cost).item())
        next_codes[src] = dst_candidates[best_idx]

    num_policy_moves = int((next_codes != torch.arange(num_codes, dtype=torch.long)).sum().item())

    if normalize == 'zscore':
        values = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    elif normalize == 'minmax':
        values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    elif normalize in {'none', None}:
        pass
    else:
        raise ValueError(f'Unsupported value normalization: {normalize}')

    return values, next_codes, {
        'features_path': features_path,
        'dataset_path': dataset_path,
        'input_key': input_key,
        'num_candidate_rows': len(candidate_rows),
        'num_good_rows': len(good_rows),
        'num_good_codes': int(good_codes.numel()),
        'good_vina_key': good_vina_key,
        'good_vina_quantile': float(good_vina_quantile),
        'good_vina_threshold': float(vina_threshold),
        'alpha': float(alpha),
        'beta': float(beta),
        'gamma': float(gamma),
        'transition_cost': transition_cost,
        'normalize': normalize,
        'num_observed_code_edges': int(has_edge.sum().item()),
        'num_reachable_codes': int(finite.sum().item()),
        'num_policy_moves': num_policy_moves,
        'num_policy_self_loops': int(num_codes - num_policy_moves),
        'value_min': float(values.min().item()),
        'value_max': float(values.max().item()),
        'value_mean': float(values.mean().item()),
        'value_std': float(values.std(unbiased=False).item()),
    }


def load_sampling_data(config, subset_name):
    ckpt = torch.load(config.model.checkpoint, map_location='cpu', weights_only=False)
    ckpt['config']['data']['path'] = config.data.path
    ckpt['config']['data']['split'] = config.data.split
    if 'name' in config.data:
        ckpt['config']['data']['name'] = config.data.name

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform,
    )
    if subset_name not in subsets:
        raise KeyError(f'Subset {subset_name} not found in split file {config.data.split}. Available: {list(subsets.keys())}')

    subset = subsets[subset_name]
    return ckpt, dataset, subset, protein_featurizer, ligand_featurizer


def load_landscape_guidance(
    config_path,
    checkpoint_path,
    generator_checkpoint,
    device,
    late_start_fraction,
    strength,
    soft_tau,
    energy_mode,
    value_features_path=None,
    value_dataset_path=None,
    value_batch_size=8192,
    value_alpha=1.0,
    value_beta=1.0,
    value_gamma=0.0,
    value_transition_cost='one_minus_prob',
    value_good_vina_key='vina_min',
    value_good_vina_quantile=0.25,
    value_normalize='zscore',
    value_cache_path=None,
    gate_mode='none',
    gate_min_cos=0.0,
    gate_min_ratio=0.0,
    gate_max_ratio=float('inf'),
    gate_softness=0.05,
    risk_gate_mode='none',
    risk_gate_energy_threshold=0.0,
    risk_gate_energy_quantile=0.75,
    risk_gate_min_pl_distance=0.0,
    target_fm_ratio=0.0,
    target_fm_ratio_max_scale=float('inf'),
    clash_guidance_weight=0.0,
    clash_guidance_cutoff=2.0,
    clash_guidance_late_start_fraction=0.0,
    clash_guidance_target_fm_ratio=0.0,
    clash_guidance_target_fm_ratio_max_scale=float('inf'),
    pair_topology_state_csv=None,
    pair_quality_potential_csv=None,
    pair_potential_csv=None,
    pair_lambda=0.1,
    prob_transition_stride=20,
    prob_transition_lambda=1.0,
    prob_transition_smoothing=1e-4,
    prob_transition_pose_bank=None,
    prob_transition_cache=None,
    path_value_checkpoint=None,
    path_value_component='auto',
    path_value_projection='none',
    hjb_value_checkpoint=None,
    hjb_t0=0.5,
    hjb_sigmoid_k=12.0,
    hjb_sampling_mode='residual_guidance',
    hjb_blend_rho=0.5,
    hjb_projection_mode='none',
    hjb_value_component='total',
    hjb_control_cost_weight=0.0,
    hjb_action_max_fm_ratio=float('inf'),
    hjb_replay_gate_checkpoint=None,
    hjb_replay_gate_mode='none',
    hjb_replay_gate_threshold=0.5,
    hjb_replay_gate_temperature=1.0,
    hjb_ratio_selector_checkpoint=None,
    hjb_ratio_selector_mode='none',
    hjb_ratio_candidates='0,0.025,0.05,0.10,0.15',
    hjb_value_gradient_checkpoint=None,
    hjb_actor_checkpoint=None,
    hjb_actor_mode='none',
    hjb_actor_output_sign=1.0,
    hjb_actor_output_projection='none',
    controller_checkpoint=None,
):
    train_config = misc.load_config(config_path)
    encoder = PAFlowLandscapeEncoder.from_checkpoint(
        checkpoint_path=generator_checkpoint,
        device=device,
        pooling='mean',
        graph_context='joint',
        graph_emb_dim=None,
        freeze_backbone=True,
        center_pos_mode='protein',
    ).to(device)
    encoder.eval()

    input_dim = int(encoder.graph_emb_dim)
    landscape_model = build_landscape_model(train_config, input_dim=input_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    landscape_model.load_state_dict(ckpt['model'])
    landscape_model.eval()
    for param in landscape_model.parameters():
        param.requires_grad_(False)

    value_info = None
    needs_value_graph = str(energy_mode) in {
        'value',
        'value_local',
        'value_gated_local',
        'prototype',
        'value_proto',
        'value_proto_local',
    }
    needs_code_policy = str(energy_mode) in {'prototype', 'value_proto', 'value_proto_local'}
    if needs_value_graph:
        code_values = None
        code_next = None
        cache_hit = False
        if value_cache_path and os.path.exists(value_cache_path):
            cached = torch.load(value_cache_path, map_location='cpu', weights_only=False)
            code_values = cached['code_values']
            code_next = cached.get('code_next')
            value_info = cached.get('value_info', {})
            cache_hit = True
        if code_values is None or (needs_code_policy and code_next is None):
            code_values, code_next, value_info = _build_value_to_go(
                landscape_model,
                train_config=train_config,
                device=device,
                features_path=value_features_path,
                dataset_path=value_dataset_path,
                input_key=getattr(train_config.model, 'input_key', None),
                batch_size=value_batch_size,
                alpha=value_alpha,
                beta=value_beta,
                gamma=value_gamma,
                transition_cost=value_transition_cost,
                good_vina_key=value_good_vina_key,
                good_vina_quantile=value_good_vina_quantile,
                normalize=value_normalize,
            )
            if value_cache_path:
                os.makedirs(os.path.dirname(value_cache_path) or '.', exist_ok=True)
                torch.save({
                    'code_values': code_values.cpu(),
                    'code_next': code_next.cpu(),
                    'value_info': value_info,
                }, value_cache_path)
            cache_hit = False
        value_info = {**(value_info or {}), 'cache_path': value_cache_path, 'cache_hit': cache_hit}
        if code_values is not None:
            landscape_model.set_code_values(code_values)
        if code_next is not None:
            landscape_model.set_code_policy_next(code_next)

    pairwise_potential = None
    if str(energy_mode) in {'pairwise', 'frozen_h_controller', 'frozen_h_relative', 'prob_transition'}:
        if not hasattr(landscape_model, 'quality_num_codes'):
            if str(energy_mode) in {'pairwise', 'frozen_h_controller', 'frozen_h_relative'}:
                raise ValueError(f'{energy_mode} landscape guidance requires a dual-codebook landscape checkpoint.')
        missing = [
            name for name, path in {
                'pair_topology_state_csv': pair_topology_state_csv,
                'pair_quality_potential_csv': pair_quality_potential_csv if str(energy_mode) != 'prob_transition' else 'optional',
                'pair_potential_csv': pair_potential_csv if str(energy_mode) != 'prob_transition' else 'optional',
            }.items()
            if not path
        ]
        if missing:
            raise ValueError(f'{energy_mode} landscape guidance is missing required files: {missing}')
        if str(energy_mode) in {'pairwise', 'frozen_h_controller', 'frozen_h_relative'} or (pair_quality_potential_csv and pair_potential_csv and hasattr(landscape_model, 'quality_num_codes')):
            pairwise_potential = _load_pairwise_potential_tensors(
                topology_state_csv=pair_topology_state_csv,
                quality_potential_csv=pair_quality_potential_csv,
                pair_potential_csv=pair_potential_csv,
                num_topology_codes=int(landscape_model.num_codes),
                num_quality_codes=int(landscape_model.quality_num_codes),
                device=device,
            )

    prob_transition = None
    if str(energy_mode) in {'prob_transition', 'prob_transition_disp'}:
        if not pair_topology_state_csv:
            raise ValueError(f'{energy_mode} guidance requires --landscape_pair_topology_state_csv.')
        if prob_transition_cache and os.path.exists(prob_transition_cache):
            prob_transition = torch.load(prob_transition_cache, map_location=device, weights_only=False)
            for key, value in list(prob_transition.items()):
                if torch.is_tensor(value):
                    prob_transition[key] = value.to(device)
        elif str(energy_mode) == 'prob_transition_disp':
            if not prob_transition_pose_bank:
                raise ValueError('prob_transition_disp guidance requires --landscape_prob_transition_pose_bank.')
            prob_transition = _build_prob_transition_displacement_tensors(
                topology_state_csv=pair_topology_state_csv,
                pose_bank_path=prob_transition_pose_bank,
                num_topology_codes=int(landscape_model.num_codes),
                device=device,
                stride=prob_transition_stride,
                smoothing=prob_transition_smoothing,
            )
            if prob_transition_cache:
                os.makedirs(os.path.dirname(prob_transition_cache) or '.', exist_ok=True)
                torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prob_transition.items()}, prob_transition_cache)
        else:
            prob_transition = _build_prob_transition_tensors(
                topology_state_csv=pair_topology_state_csv,
                num_topology_codes=int(landscape_model.num_codes),
                device=device,
                stride=prob_transition_stride,
                smoothing=prob_transition_smoothing,
            )
            if prob_transition_cache:
                os.makedirs(os.path.dirname(prob_transition_cache) or '.', exist_ok=True)
                torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prob_transition.items()}, prob_transition_cache)
        if pairwise_potential is not None:
            prob_transition['code_values'] = pairwise_potential['topology_values'].to(device)

    path_model = None
    path_correction_scale = 0.0
    controller = None
    controller_info = None
    path_runtime_info = None
    if str(energy_mode) in {'frozen_h_controller', 'frozen_h_relative'}:
        if not path_value_checkpoint:
            raise ValueError(f'{energy_mode} requires --landscape_path_value_checkpoint.')
        if str(energy_mode) == 'frozen_h_controller' and not controller_checkpoint:
            raise ValueError('frozen_h_controller requires --landscape_controller_checkpoint.')
        from scripts.train_path_value_potential import DualHeadRelativePathValueModel, PathValueMLP, RelativePathValueModel, TripleHeadRelativePathValueModel

        path_ckpt = torch.load(path_value_checkpoint, map_location=device, weights_only=False)
        path_args = path_ckpt.get('args', {})
        if path_ckpt.get('model_type') == 'residual_path_value':
            base_model = PathValueMLP(
                input_dim=int(path_ckpt['input_dim']),
                hidden_dim=int(path_args.get('hidden_dim', 256)),
                num_layers=int(path_args.get('num_layers', 3)),
                dropout=float(path_args.get('dropout', 0.0)),
            ).to(device)
            base_model.load_state_dict(path_ckpt['base_model_state_dict'])
            residual_model = PathValueMLP(
                input_dim=int(path_ckpt['input_dim']),
                hidden_dim=int(path_args.get('residual_hidden_dim', 64)),
                num_layers=int(path_args.get('residual_num_layers', 2)),
                dropout=float(path_args.get('residual_dropout', 0.0)),
            ).to(device)
            residual_model.load_state_dict(path_ckpt['residual_model_state_dict'])
            path_model = _SamplingResidualPathValue(
                base_model,
                residual_model,
                float(path_args.get('residual_scale', 0.1)),
            ).to(device)
        elif path_ckpt.get('model_type') == 'dual_relative_path_value':
            input_info = path_ckpt['input_info']
            aux_dim = int(path_ckpt['input_dim']) - int(input_info['state_dim']) - int(input_info['pocket_dim'])
            path_model = DualHeadRelativePathValueModel(
                state_input_dim=int(input_info['state_dim']),
                pocket_input_dim=int(input_info['pocket_dim']),
                aux_input_dim=int(aux_dim),
                hidden_dim=int(path_args.get('hidden_dim', 256)),
                num_layers=int(path_args.get('num_layers', 3)),
                dropout=float(path_args.get('dropout', 0.0)),
            ).to(device)
            path_model.load_state_dict(path_ckpt['model_state_dict'])
        elif path_ckpt.get('model_type') == 'triple_relative_path_value':
            input_info = path_ckpt['input_info']
            aux_dim = int(path_ckpt['input_dim']) - int(input_info['state_dim']) - int(input_info['pocket_dim'])
            path_model = TripleHeadRelativePathValueModel(
                state_input_dim=int(input_info['state_dim']),
                pocket_input_dim=int(input_info['pocket_dim']),
                aux_input_dim=int(aux_dim),
                hidden_dim=int(path_args.get('hidden_dim', 256)),
                num_layers=int(path_args.get('num_layers', 3)),
                dropout=float(path_args.get('dropout', 0.0)),
            ).to(device)
            path_model.load_state_dict(path_ckpt['model_state_dict'])
        elif path_ckpt.get('model_type') == 'relative_path_value':
            input_info = path_ckpt['input_info']
            aux_dim = int(path_ckpt['input_dim']) - int(input_info['state_dim']) - int(input_info['pocket_dim'])
            path_model = RelativePathValueModel(
                state_input_dim=int(input_info['state_dim']),
                pocket_input_dim=int(input_info['pocket_dim']),
                aux_input_dim=int(aux_dim),
                hidden_dim=int(path_args.get('hidden_dim', 256)),
                num_layers=int(path_args.get('num_layers', 3)),
                dropout=float(path_args.get('dropout', 0.0)),
            ).to(device)
            path_model.load_state_dict(path_ckpt['model_state_dict'])
        else:
            path_model = PathValueMLP(
                input_dim=int(path_ckpt['input_dim']),
                hidden_dim=int(path_args.get('hidden_dim', 256)),
                num_layers=int(path_args.get('num_layers', 3)),
                dropout=float(path_args.get('dropout', 0.0)),
            ).to(device)
            path_model.load_state_dict(path_ckpt['model_state_dict'])
        path_model.eval()
        for param in path_model.parameters():
            param.requires_grad_(False)
        path_correction_scale = float(path_args.get('base_correction_scale', path_args.get('correction_scale', 0.5)))

        extra_feature_columns = _parse_csv_list(path_args.get('extra_feature_columns', ''))
        extra_feature_stats = {}
        if extra_feature_columns:
            if not path_args.get('path_value_csv'):
                raise ValueError(f'{energy_mode} needs args.path_value_csv in the path-value checkpoint to recover feature normalization.')
            df_stats = pd.read_csv(path_args['path_value_csv'], usecols=extra_feature_columns)
            for col in extra_feature_columns:
                values = pd.to_numeric(df_stats[col], errors='coerce').to_numpy(dtype=float)
                extra_feature_stats[col] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(max(np.nanstd(values), 1e-6)),
                }
        path_runtime_info = {
            'model_type': path_ckpt.get('model_type', 'path_value_mlp'),
            'input_info': path_ckpt.get('input_info'),
            'extra_feature_columns': extra_feature_columns,
            'extra_feature_stats': extra_feature_stats,
            'append_prev_state_emb': bool(path_args.get('append_prev_state_emb', False)),
            'append_delta_state_emb': bool(path_args.get('append_delta_state_emb', False)),
            'append_history_scalar_features': bool(path_args.get('append_history_scalar_features', False)),
            'use_pocket_baseline_decomposition': bool(path_args.get('use_pocket_baseline_decomposition', False)),
            'input_u_mode': str(path_args.get('input_u_mode', 'base')),
        }

        if str(energy_mode) == 'frozen_h_controller':
            ctrl_ckpt = torch.load(controller_checkpoint, map_location=device, weights_only=False)
            ctrl_args = ctrl_ckpt.get('args', {})
            controller = _SamplingPHController(
                input_dim=int(ctrl_ckpt['controller_input_dim']),
                hidden_dim=int(ctrl_args.get('hidden_dim', 128)),
                num_layers=int(ctrl_args.get('num_layers', 3)),
                mode=str(ctrl_args.get('controller_mode', 'diag')),
            ).to(device)
            controller.load_state_dict(ctrl_ckpt['controller_state_dict'])
            controller.eval()
            for param in controller.parameters():
                param.requires_grad_(False)
            controller_info = {
                'path_value_checkpoint': path_value_checkpoint,
                'controller_checkpoint': controller_checkpoint,
                'controller_mode': str(ctrl_args.get('controller_mode', 'diag')),
                'path_correction_scale': path_correction_scale,
            }

    hjb_value_model = None
    hjb_replay_gate_model = None
    hjb_replay_gate_info = None
    hjb_ratio_selector_model = None
    hjb_ratio_selector_info = None
    hjb_value_gradient_model = None
    hjb_value_gradient_info = None
    hjb_actor_model = None
    hjb_actor_info = None
    if str(energy_mode) == 'hjb_value':
        if not hjb_value_checkpoint:
            raise ValueError('hjb_value guidance requires --hjb_value_checkpoint.')
        from models.hjb_value_model import build_hjb_value_model_from_checkpoint

        hjb_ckpt = torch.load(hjb_value_checkpoint, map_location=device, weights_only=False)
        hjb_value_model = build_hjb_value_model_from_checkpoint(hjb_ckpt, device=device)
        hjb_value_model.eval()
        for param in hjb_value_model.parameters():
            param.requires_grad_(False)
        if hjb_replay_gate_checkpoint and str(hjb_replay_gate_mode) != 'none':
            gate_ckpt = torch.load(hjb_replay_gate_checkpoint, map_location=device, weights_only=False)
            gate_args = gate_ckpt.get('args', {})
            feature_names = list(gate_ckpt['feature_names'])
            hjb_replay_gate_model = _ReplayGateMLP(
                input_dim=len(feature_names),
                hidden_dim=int(gate_ckpt.get('hidden_dim', gate_args.get('hidden_dim', 32))),
                num_layers=int(gate_ckpt.get('num_layers', gate_args.get('num_layers', 2))),
                dropout=float(gate_ckpt.get('dropout', 0.0)),
            ).to(device)
            hjb_replay_gate_model.load_state_dict(gate_ckpt['model_state_dict'])
            hjb_replay_gate_model.eval()
            for param in hjb_replay_gate_model.parameters():
                param.requires_grad_(False)
            hjb_replay_gate_info = {
                'checkpoint': str(hjb_replay_gate_checkpoint),
                'feature_names': feature_names,
                'feature_mean': gate_ckpt['feature_mean'].to(device).float(),
                'feature_std': gate_ckpt['feature_std'].to(device).float().clamp_min(1e-6),
                'mode': str(hjb_replay_gate_mode),
                'threshold': float(hjb_replay_gate_threshold),
                'temperature': float(hjb_replay_gate_temperature),
            }
        if hjb_ratio_selector_checkpoint and str(hjb_ratio_selector_mode) != 'none':
            selector_ckpt = torch.load(hjb_ratio_selector_checkpoint, map_location=device, weights_only=False)
            selector_args = selector_ckpt.get('args', {})
            selector_features = list(selector_ckpt['feature_names'])
            hjb_ratio_selector_model = _ReplayRatioSelectorMLP(
                input_dim=len(selector_features),
                hidden_dim=int(selector_ckpt.get('hidden_dim', selector_args.get('hidden_dim', 64))),
                num_layers=int(selector_ckpt.get('num_layers', selector_args.get('num_layers', 3))),
                dropout=float(selector_ckpt.get('dropout', 0.0)),
            ).to(device)
            hjb_ratio_selector_model.load_state_dict(selector_ckpt['model_state_dict'])
            hjb_ratio_selector_model.eval()
            for param in hjb_ratio_selector_model.parameters():
                param.requires_grad_(False)
            candidates = [
                float(item)
                for item in str(hjb_ratio_candidates).split(',')
                if str(item).strip()
            ]
            hjb_ratio_selector_info = {
                'checkpoint': str(hjb_ratio_selector_checkpoint),
                'feature_names': selector_features,
                'feature_mean': selector_ckpt['feature_mean'].to(device).float(),
                'feature_std': selector_ckpt['feature_std'].to(device).float().clamp_min(1e-6),
                'mode': str(hjb_ratio_selector_mode),
                'candidates': candidates,
            }
        if hjb_value_gradient_checkpoint:
            from models.hjb_value_gradient_model import build_hjb_value_gradient_from_checkpoint

            vg_ckpt = torch.load(hjb_value_gradient_checkpoint, map_location=device, weights_only=False)
            hjb_value_gradient_model = build_hjb_value_gradient_from_checkpoint(vg_ckpt, device=device)
            hjb_value_gradient_model.eval()
            for param in hjb_value_gradient_model.parameters():
                param.requires_grad_(False)
            hjb_value_gradient_info = {
                'checkpoint': str(hjb_value_gradient_checkpoint),
            }
        if hjb_actor_checkpoint and str(hjb_actor_mode) != 'none':
            from models.hjb_actor_model import build_hjb_actor_from_checkpoint

            actor_ckpt = torch.load(hjb_actor_checkpoint, map_location=device, weights_only=False)
            hjb_actor_model = build_hjb_actor_from_checkpoint(actor_ckpt, device=device)
            hjb_actor_model.eval()
            for param in hjb_actor_model.parameters():
                param.requires_grad_(False)
            hjb_actor_info = {
                'checkpoint': str(hjb_actor_checkpoint),
                'mode': str(hjb_actor_mode),
            }

    return {
        'encoder': encoder,
        'model': landscape_model,
        'late_start_fraction': float(late_start_fraction),
        'strength': float(strength),
        'soft_tau': float(soft_tau),
        'energy_mode': str(energy_mode),
        'value_info': value_info,
        'train_config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'gate_mode': str(gate_mode),
        'gate_min_cos': float(gate_min_cos),
        'gate_min_ratio': float(gate_min_ratio),
        'gate_max_ratio': float(gate_max_ratio),
        'gate_softness': float(gate_softness),
        'risk_gate_mode': str(risk_gate_mode),
        'risk_gate_energy_threshold': float(risk_gate_energy_threshold),
        'risk_gate_energy_quantile': float(risk_gate_energy_quantile),
        'risk_gate_min_pl_distance': float(risk_gate_min_pl_distance),
        'target_fm_ratio': float(target_fm_ratio),
        'target_fm_ratio_max_scale': float(target_fm_ratio_max_scale),
        'clash_guidance_weight': float(clash_guidance_weight),
        'clash_guidance_cutoff': float(clash_guidance_cutoff),
        'clash_guidance_late_start_fraction': float(clash_guidance_late_start_fraction),
        'clash_guidance_target_fm_ratio': float(clash_guidance_target_fm_ratio),
        'clash_guidance_target_fm_ratio_max_scale': float(clash_guidance_target_fm_ratio_max_scale),
        'pairwise_potential': pairwise_potential,
        'pair_lambda': float(pair_lambda),
        'pair_topology_state_csv': pair_topology_state_csv,
        'pair_quality_potential_csv': pair_quality_potential_csv,
        'pair_potential_csv': pair_potential_csv,
        'prob_transition': prob_transition,
        'prob_transition_stride': int(prob_transition_stride),
        'prob_transition_lambda': float(prob_transition_lambda),
        'prob_transition_smoothing': float(prob_transition_smoothing),
        'prob_transition_pose_bank': prob_transition_pose_bank,
        'prob_transition_cache': prob_transition_cache,
        'path_model': path_model,
        'path_correction_scale': float(path_correction_scale),
        'path_value_component': str(path_value_component),
        'path_value_projection': str(path_value_projection),
        'path_runtime_info': path_runtime_info,
        'controller': controller,
        'controller_info': controller_info,
        'hjb_value_model': hjb_value_model,
        'hjb_value_checkpoint': hjb_value_checkpoint,
        'hjb_t0': float(hjb_t0),
        'hjb_sigmoid_k': float(hjb_sigmoid_k),
        'hjb_sampling_mode': str(hjb_sampling_mode),
        'hjb_blend_rho': float(hjb_blend_rho),
        'hjb_projection_mode': str(hjb_projection_mode),
        'hjb_value_component': str(hjb_value_component),
        'hjb_control_cost_weight': float(hjb_control_cost_weight),
        'hjb_action_max_fm_ratio': float(hjb_action_max_fm_ratio),
        'hjb_replay_gate_model': hjb_replay_gate_model,
        'hjb_replay_gate_info': hjb_replay_gate_info,
        'hjb_ratio_selector_model': hjb_ratio_selector_model,
        'hjb_ratio_selector_info': hjb_ratio_selector_info,
        'hjb_value_gradient_model': hjb_value_gradient_model,
        'hjb_value_gradient_info': hjb_value_gradient_info,
        'hjb_actor_model': hjb_actor_model,
        'hjb_actor_info': hjb_actor_info,
        'hjb_actor_output_sign': float(hjb_actor_output_sign),
        'hjb_actor_output_projection': str(hjb_actor_output_projection),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('-i', '--data_id', type=int, default=0, help='index within the selected split subset')
    parser.add_argument('--subset', type=str, default=None, help='override config.data.subset')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=None, help='override config.sample.num_samples')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--v_grad_w', type=float, default=0.0)
    parser.add_argument('--pos_grad_w', type=float, default=350.0)
    parser.add_argument(
        '--energy_guidance_gamma',
        type=float,
        default=None,
        help='Override model.config.energy_guidance_gamma. Use 0 to disable the original PAFlow binding-energy velocity term.',
    )
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, help='override sampling seed from config')
    parser.add_argument('--landscape_config', type=str, default=None, help='landscape training config for velocity guidance')
    parser.add_argument('--landscape_checkpoint', type=str, default=None, help='landscape checkpoint for velocity guidance')
    parser.add_argument('--landscape_late_start_fraction', type=float, default=0.8)
    parser.add_argument('--landscape_guidance_strength', type=float, default=0.0)
    parser.add_argument('--landscape_guidance_tau', type=float, default=1.0)
    parser.add_argument(
        '--landscape_energy_mode',
        type=str,
        default='code',
        choices=[
            'code',
            'local',
            'total',
            'value',
            'value_local',
            'value_gated_local',
            'prototype',
            'value_proto',
            'value_proto_local',
            'pairwise',
            'prob_transition',
            'prob_transition_disp',
            'frozen_h_controller',
            'frozen_h_relative',
            'hjb_value',
        ],
    )
    parser.add_argument('--landscape_value_features', type=str, default=None, help='feature file used to build value-to-go code potentials')
    parser.add_argument('--landscape_value_dataset', type=str, default=None, help='landscape dataset with final labels used to define good basins')
    parser.add_argument('--landscape_value_batch_size', type=int, default=8192)
    parser.add_argument('--landscape_value_alpha', type=float, default=1.0)
    parser.add_argument('--landscape_value_beta', type=float, default=1.0)
    parser.add_argument('--landscape_value_gamma', type=float, default=0.0, help='codebook prototype-distance weight for value-to-go edge cost')
    parser.add_argument('--landscape_value_transition_cost', type=str, default='one_minus_prob', choices=['one_minus_prob', 'neg_log_prob'])
    parser.add_argument('--landscape_value_good_vina_key', type=str, default='vina_min')
    parser.add_argument('--landscape_value_good_vina_quantile', type=float, default=0.25)
    parser.add_argument('--landscape_value_normalize', type=str, default='zscore', choices=['zscore', 'minmax', 'none'])
    parser.add_argument('--landscape_value_cache', type=str, default=None, help='optional cache path for computed code value-to-go potentials')
    parser.add_argument('--landscape_gate_mode', type=str, default='none', choices=['none', 'alignment', 'soft_alignment'])
    parser.add_argument('--landscape_gate_min_cos', type=float, default=0.0)
    parser.add_argument('--landscape_gate_min_ratio', type=float, default=0.0)
    parser.add_argument('--landscape_gate_max_ratio', type=float, default=float('inf'))
    parser.add_argument('--landscape_gate_softness', type=float, default=0.05)
    parser.add_argument(
        '--landscape_risk_gate_mode',
        type=str,
        default='none',
        choices=['none', 'energy_threshold', 'energy_quantile', 'contact', 'energy_or_contact'],
        help='Additional graph-level risk gate. This is independent of alignment gating.',
    )
    parser.add_argument('--landscape_risk_gate_energy_threshold', type=float, default=0.0)
    parser.add_argument('--landscape_risk_gate_energy_quantile', type=float, default=0.75)
    parser.add_argument('--landscape_risk_gate_min_pl_distance', type=float, default=0.0)
    parser.add_argument('--landscape_target_fm_ratio', type=float, default=0.0, help='if >0, rescale landscape dx per graph to this ||landscape|| / ||FM|| ratio after gating')
    parser.add_argument('--landscape_target_fm_ratio_max_scale', type=float, default=float('inf'), help='optional cap on the per-graph landscape rescale factor')
    parser.add_argument('--sampling_clash_guidance_weight', type=float, default=0.0)
    parser.add_argument('--sampling_clash_guidance_cutoff', type=float, default=2.0)
    parser.add_argument('--sampling_clash_guidance_late_start_fraction', type=float, default=0.0)
    parser.add_argument('--sampling_clash_guidance_target_fm_ratio', type=float, default=0.0, help='if >0, rescale sampling clash/soft-steric dx per graph to this ||clash|| / ||FM|| ratio')
    parser.add_argument('--sampling_clash_guidance_target_fm_ratio_max_scale', type=float, default=float('inf'), help='optional cap on the per-graph clash/soft-steric rescale factor')
    parser.add_argument('--landscape_pair_topology_state_csv', type=str, default=None)
    parser.add_argument('--landscape_pair_quality_potential_csv', type=str, default=None)
    parser.add_argument('--landscape_pair_potential_csv', type=str, default=None)
    parser.add_argument('--landscape_pair_lambda', type=float, default=0.1)
    parser.add_argument('--landscape_prob_transition_stride', type=int, default=20)
    parser.add_argument('--landscape_prob_transition_lambda', type=float, default=1.0)
    parser.add_argument('--landscape_prob_transition_smoothing', type=float, default=1e-4)
    parser.add_argument('--landscape_prob_transition_pose_bank', type=str, default=None)
    parser.add_argument('--landscape_prob_transition_cache', type=str, default=None)
    parser.add_argument('--landscape_path_value_checkpoint', type=str, default=None)
    parser.add_argument(
        '--landscape_path_value_component',
        type=str,
        default='auto',
        choices=['auto', 'pred', 'relative', 'correction', 'safety', 'dock', 'raw', 'relative_safety', 'relative_dock', 'relative_raw'],
        help='Select which path-value component to differentiate for frozen_h_relative guidance.',
    )
    parser.add_argument(
        '--landscape_path_value_projection',
        type=str,
        default='none',
        choices=['none', 'dock_projected', 'raw_projected', 'raw_dock_projected', 'vfm_orthogonal', 'double_projected', 'raw_dock_vfm_projected'],
        help='Optional constrained correction mode for multi-head path-value guidance.',
    )
    parser.add_argument('--landscape_controller_checkpoint', type=str, default=None)
    parser.add_argument('--hjb_value_checkpoint', type=str, default=None)
    parser.add_argument('--hjb_t0', type=float, default=0.5)
    parser.add_argument('--hjb_sigmoid_k', type=float, default=12.0)
    parser.add_argument(
        '--hjb_sampling_mode',
        type=str,
        default='residual_guidance',
        choices=['residual_guidance', 'bellman_consistent', 'direct_full', 'direct_scaled', 'blended_full', 'blended_schedule'],
    )
    parser.add_argument('--hjb_blend_rho', type=float, default=0.5)
    parser.add_argument(
        '--hjb_projection_mode',
        type=str,
        default='none',
        choices=['none', 'positive_only', 'remove_negative_parallel', 'rigid_body'],
    )
    parser.add_argument(
        '--hjb_value_component',
        type=str,
        default='total',
        choices=[
            'auto',
            'total',
            'sum',
            'safe',
            'dock',
            'drug',
            'min',
            'score',
            'valid',
            'safe_only',
            'dock_only',
            'drug_only',
            'safe+dock',
            'safe+dock+drug',
            'constrained_dock',
            'dock_constrained',
            'dock_constrained_soft',
            'dock_constrained_strong',
            'dock_projected_multi',
            'projected_multi',
            'multi_projected',
        ],
    )
    parser.add_argument(
        '--hjb_control_cost_weight',
        type=float,
        default=0.0,
        help='Control-cost coefficient c_u for bellman_consistent HJB sampling.',
    )
    parser.add_argument(
        '--hjb_action_max_fm_ratio',
        type=float,
        default=float('inf'),
        help='Optional trust-region cap for bellman_consistent action norm relative to FM norm.',
    )
    parser.add_argument('--hjb_replay_gate_checkpoint', type=str, default=None)
    parser.add_argument(
        '--hjb_replay_gate_mode',
        type=str,
        default='none',
        choices=['none', 'continuous', 'hard', 'thresholded'],
    )
    parser.add_argument('--hjb_replay_gate_threshold', type=float, default=0.5)
    parser.add_argument('--hjb_replay_gate_temperature', type=float, default=1.0)
    parser.add_argument('--hjb_ratio_selector_checkpoint', type=str, default=None)
    parser.add_argument(
        '--hjb_ratio_selector_mode',
        type=str,
        default='none',
        choices=['none', 'min_cost'],
        help='Use a replay action selector to choose a per-graph target FM ratio.',
    )
    parser.add_argument(
        '--hjb_ratio_candidates',
        type=str,
        default='0,0.025,0.05,0.10,0.15',
        help='Comma-separated candidate target-FM ratios evaluated by the replay action selector.',
    )
    parser.add_argument(
        '--hjb_value_gradient_checkpoint',
        type=str,
        default=None,
        help='Optional learned value-gradient field used as actor input instead of raw -grad S.',
    )
    parser.add_argument('--hjb_actor_checkpoint', type=str, default=None)
    parser.add_argument(
        '--hjb_actor_mode',
        type=str,
        default='none',
        choices=['none', 'replace_neg_grad'],
        help='Use a frozen value-conditioned actor in place of direct -grad S for HJB residual guidance.',
    )
    parser.add_argument(
        '--hjb_actor_output_sign',
        type=float,
        default=1.0,
        help='Diagnostic multiplier applied to frozen actor output before residual guidance; use -1 to test reward-gradient sign.',
    )
    parser.add_argument(
        '--hjb_actor_output_projection',
        type=str,
        default='none',
        choices=['none', 'positive_only', 'remove_negative_parallel'],
        help='Diagnostic projection applied to frozen actor output after sign transform.',
    )
    parser.add_argument('--trace_velocity_components', action='store_true', help='save per-step FM / landscape velocity norms and alignment')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    effective_seed = config.sample.seed if args.seed is None else int(args.seed)
    misc.seed_all(effective_seed)
    subset_name = args.subset or config.data.subset
    result_path = args.result_path or config.output.result_path
    os.makedirs(result_path, exist_ok=True)

    logger = misc.get_logger('sampling_split', log_dir=result_path)
    logger.info(config)
    logger.info(args)
    logger.info(f'Using subset: {subset_name}')
    logger.info(f'Results will be saved to: {result_path}')
    logger.info(f'Using sampling seed: {effective_seed}')
    if args.num_samples is not None:
        config.sample.num_samples = int(args.num_samples)
        logger.info(f'Override config.sample.num_samples={config.sample.num_samples}')

    ckpt, dataset, subset, protein_featurizer, ligand_featurizer = load_sampling_data(config, subset_name)
    logger.info(f'Successfully loaded subset {subset_name} (size: {len(subset)})')

    if args.data_id < 0 or args.data_id >= len(subset):
        raise IndexError(f'data_id {args.data_id} out of range for subset {subset_name} of size {len(subset)}')

    model = ScorePosNet3D_guided_flow(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device,
    ).to(args.device)

    expert_ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    for key, value in expert_ckpt['model'].items():
        if key.startswith('expert_pred'):
            ckpt['model'][key] = value
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully loaded model checkpoint: {config.model.checkpoint}')
    if args.energy_guidance_gamma is not None:
        model.config.energy_guidance_gamma = float(args.energy_guidance_gamma)
        logger.info(f'Override model.config.energy_guidance_gamma={model.config.energy_guidance_gamma}')

    data = subset[args.data_id]
    dataset_index = subset.indices[args.data_id]
    protein_name = data['protein_filename'].split('/')[0]
    logger.info(f'Generating ligands for protein: {protein_name}')
    logger.info(f'Subset index: {args.data_id} | dataset index: {dataset_index}')

    landscape_guidance = None
    if args.landscape_checkpoint:
        if not args.landscape_config:
            raise ValueError('--landscape_config is required when --landscape_checkpoint is provided')
        landscape_guidance = load_landscape_guidance(
            config_path=args.landscape_config,
            checkpoint_path=args.landscape_checkpoint,
            generator_checkpoint=config.model.checkpoint,
            device=args.device,
            late_start_fraction=args.landscape_late_start_fraction,
            strength=args.landscape_guidance_strength,
            soft_tau=args.landscape_guidance_tau,
            energy_mode=args.landscape_energy_mode,
            value_features_path=args.landscape_value_features,
            value_dataset_path=args.landscape_value_dataset,
            value_batch_size=args.landscape_value_batch_size,
            value_alpha=args.landscape_value_alpha,
            value_beta=args.landscape_value_beta,
            value_gamma=args.landscape_value_gamma,
            value_transition_cost=args.landscape_value_transition_cost,
            value_good_vina_key=args.landscape_value_good_vina_key,
            value_good_vina_quantile=args.landscape_value_good_vina_quantile,
            value_normalize=args.landscape_value_normalize,
            value_cache_path=args.landscape_value_cache,
            gate_mode=args.landscape_gate_mode,
            gate_min_cos=args.landscape_gate_min_cos,
            gate_min_ratio=args.landscape_gate_min_ratio,
            gate_max_ratio=args.landscape_gate_max_ratio,
            gate_softness=args.landscape_gate_softness,
            risk_gate_mode=args.landscape_risk_gate_mode,
            risk_gate_energy_threshold=args.landscape_risk_gate_energy_threshold,
            risk_gate_energy_quantile=args.landscape_risk_gate_energy_quantile,
            risk_gate_min_pl_distance=args.landscape_risk_gate_min_pl_distance,
            target_fm_ratio=args.landscape_target_fm_ratio,
            target_fm_ratio_max_scale=args.landscape_target_fm_ratio_max_scale,
            clash_guidance_weight=args.sampling_clash_guidance_weight,
            clash_guidance_cutoff=args.sampling_clash_guidance_cutoff,
            clash_guidance_late_start_fraction=args.sampling_clash_guidance_late_start_fraction,
            clash_guidance_target_fm_ratio=args.sampling_clash_guidance_target_fm_ratio,
            clash_guidance_target_fm_ratio_max_scale=args.sampling_clash_guidance_target_fm_ratio_max_scale,
            pair_topology_state_csv=args.landscape_pair_topology_state_csv,
            pair_quality_potential_csv=args.landscape_pair_quality_potential_csv,
            pair_potential_csv=args.landscape_pair_potential_csv,
            pair_lambda=args.landscape_pair_lambda,
            prob_transition_stride=args.landscape_prob_transition_stride,
            prob_transition_lambda=args.landscape_prob_transition_lambda,
            prob_transition_smoothing=args.landscape_prob_transition_smoothing,
            prob_transition_pose_bank=args.landscape_prob_transition_pose_bank,
            prob_transition_cache=args.landscape_prob_transition_cache,
            path_value_checkpoint=args.landscape_path_value_checkpoint,
            path_value_component=args.landscape_path_value_component,
            path_value_projection=args.landscape_path_value_projection,
            hjb_value_checkpoint=args.hjb_value_checkpoint,
            hjb_t0=args.hjb_t0,
            hjb_sigmoid_k=args.hjb_sigmoid_k,
            hjb_sampling_mode=args.hjb_sampling_mode,
            hjb_blend_rho=args.hjb_blend_rho,
            hjb_projection_mode=args.hjb_projection_mode,
            hjb_value_component=args.hjb_value_component,
            hjb_control_cost_weight=args.hjb_control_cost_weight,
            hjb_action_max_fm_ratio=args.hjb_action_max_fm_ratio,
            hjb_replay_gate_checkpoint=args.hjb_replay_gate_checkpoint,
            hjb_replay_gate_mode=args.hjb_replay_gate_mode,
            hjb_replay_gate_threshold=args.hjb_replay_gate_threshold,
            hjb_replay_gate_temperature=args.hjb_replay_gate_temperature,
            hjb_ratio_selector_checkpoint=args.hjb_ratio_selector_checkpoint,
            hjb_ratio_selector_mode=args.hjb_ratio_selector_mode,
            hjb_ratio_candidates=args.hjb_ratio_candidates,
            hjb_value_gradient_checkpoint=args.hjb_value_gradient_checkpoint,
            hjb_actor_checkpoint=args.hjb_actor_checkpoint,
            hjb_actor_mode=args.hjb_actor_mode,
            hjb_actor_output_sign=args.hjb_actor_output_sign,
            hjb_actor_output_projection=args.hjb_actor_output_projection,
            controller_checkpoint=args.landscape_controller_checkpoint,
        )
        landscape_guidance['trace_velocity'] = bool(args.trace_velocity_components)
        logger.info(
            'Enabled landscape velocity guidance: '
            f'ckpt={args.landscape_checkpoint}, '
            f'late_start={args.landscape_late_start_fraction}, '
            f'strength={args.landscape_guidance_strength}, '
            f'tau={args.landscape_guidance_tau}, '
            f'energy_mode={args.landscape_energy_mode}, '
            f'gate={args.landscape_gate_mode}, '
            f'risk_gate={args.landscape_risk_gate_mode}, '
            f'path_projection={args.landscape_path_value_projection}, '
            f'hjb_sampling_mode={args.hjb_sampling_mode}, '
            f'hjb_value_component={args.hjb_value_component}, '
            f'hjb_blend_rho={args.hjb_blend_rho}, '
            f'hjb_projection_mode={args.hjb_projection_mode}, '
            f'hjb_control_cost_weight={args.hjb_control_cost_weight}, '
            f'hjb_action_max_fm_ratio={args.hjb_action_max_fm_ratio}, '
            f'hjb_replay_gate={args.hjb_replay_gate_mode}:{args.hjb_replay_gate_checkpoint}, '
            f'hjb_ratio_selector={args.hjb_ratio_selector_mode}:{args.hjb_ratio_selector_checkpoint}, '
            f'hjb_actor={args.hjb_actor_mode}:{args.hjb_actor_checkpoint}, '
            f'hjb_actor_output_sign={args.hjb_actor_output_sign}, '
            f'hjb_actor_output_projection={args.hjb_actor_output_projection}, '
            f'target_fm_ratio={args.landscape_target_fm_ratio}, '
            f'trace_velocity={args.trace_velocity_components}'
        )
        if landscape_guidance.get('value_info') is not None:
            logger.info(f"Value-to-go guidance info: {landscape_guidance['value_info']}")

    pocket_info = None
    effective_sample_num_atoms = config.sample.sample_num_atoms
    if config.sample.sample_num_atoms == 'predict':
        from sample_atom_num import predict_atom_num  # local import to match existing sampling behavior
        import pickle

        dataset_path = './data/atom_num_dataset.pkl'
        with open(dataset_path, 'rb') as f:
            atom_num_dataset = pickle.load(f)
        if subset_name in atom_num_dataset:
            subset_info = atom_num_dataset[subset_name]
        elif subset_name == 'val' and 'train' in atom_num_dataset:
            # The new validation split is carved out from the original train pool.
            subset_info = atom_num_dataset['train']
        else:
            subset_info = atom_num_dataset.get('test')
        key = data['ligand_filename'][0:-4]
        if key not in subset_info:
            logger.warning(
                f"Key '{key}' not found in atom number dataset subset '{subset_name}'. "
                "Falling back to sample_num_atoms='ref' for this example."
            )
            effective_sample_num_atoms = 'ref'
        else:
            item = subset_info[key]
            pocket_info = torch.tensor([
                item['pocket_atom_num'],
                item['volume'],
                item['area'],
                item['pocket_size'],
            ]).float().to(args.device).unsqueeze(0)

    # This legacy sampler is imported lazily so bank construction can reuse
    # load_sampling_data without pulling in unrelated training modules.
    from sample_flow_VP_guide import sample_diffusion_ligand

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, velocity_trace = sample_diffusion_ligand(
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
        atom_predictor_ckpt=config.model.atom_predictor_ckpt if effective_sample_num_atoms == 'predict' else None,
        atom_num_std=config.sample.num_atoms_std if effective_sample_num_atoms == 'predict' else None,
        pos_grad_w=args.pos_grad_w,
        v_grad_w=args.v_grad_w,
        landscape_guidance=landscape_guidance,
    )

    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'velocity_trace': velocity_trace,
        'split_name': subset_name,
        'subset_index': args.data_id,
        'dataset_index': int(dataset_index),
        'sample_num_atoms_mode': effective_sample_num_atoms,
        'landscape_velocity_guidance': None if landscape_guidance is None else {
            'checkpoint': landscape_guidance['checkpoint_path'],
            'config': landscape_guidance['train_config_path'],
            'late_start_fraction': float(args.landscape_late_start_fraction),
            'strength': float(args.landscape_guidance_strength),
            'tau': float(args.landscape_guidance_tau),
            'energy_mode': str(args.landscape_energy_mode),
            'value_info': landscape_guidance.get('value_info'),
            'gate_mode': str(args.landscape_gate_mode),
            'gate_min_cos': float(args.landscape_gate_min_cos),
            'gate_min_ratio': float(args.landscape_gate_min_ratio),
            'gate_max_ratio': float(args.landscape_gate_max_ratio),
            'gate_softness': float(args.landscape_gate_softness),
            'risk_gate_mode': str(args.landscape_risk_gate_mode),
            'risk_gate_energy_threshold': float(args.landscape_risk_gate_energy_threshold),
            'risk_gate_energy_quantile': float(args.landscape_risk_gate_energy_quantile),
            'risk_gate_min_pl_distance': float(args.landscape_risk_gate_min_pl_distance),
            'target_fm_ratio': float(args.landscape_target_fm_ratio),
            'target_fm_ratio_max_scale': float(args.landscape_target_fm_ratio_max_scale),
            'clash_guidance_target_fm_ratio': float(args.sampling_clash_guidance_target_fm_ratio),
            'clash_guidance_target_fm_ratio_max_scale': float(args.sampling_clash_guidance_target_fm_ratio_max_scale),
            'pos_grad_w': float(args.pos_grad_w),
            'energy_guidance_gamma': None if args.energy_guidance_gamma is None else float(args.energy_guidance_gamma),
            'pair_lambda': float(args.landscape_pair_lambda),
            'pair_topology_state_csv': args.landscape_pair_topology_state_csv,
            'pair_quality_potential_csv': args.landscape_pair_quality_potential_csv,
            'pair_potential_csv': args.landscape_pair_potential_csv,
            'prob_transition_info': None if landscape_guidance.get('prob_transition') is None else landscape_guidance['prob_transition'].get('info'),
            'prob_transition_stride': int(args.landscape_prob_transition_stride),
            'prob_transition_lambda': float(args.landscape_prob_transition_lambda),
            'prob_transition_smoothing': float(args.landscape_prob_transition_smoothing),
            'prob_transition_pose_bank': args.landscape_prob_transition_pose_bank,
            'prob_transition_cache': args.landscape_prob_transition_cache,
            'path_value_checkpoint': args.landscape_path_value_checkpoint,
            'path_value_component': args.landscape_path_value_component,
            'path_value_projection': args.landscape_path_value_projection,
            'hjb_value_checkpoint': args.hjb_value_checkpoint,
            'hjb_sampling_mode': args.hjb_sampling_mode,
            'hjb_value_component': args.hjb_value_component,
            'hjb_blend_rho': float(args.hjb_blend_rho),
            'hjb_projection_mode': args.hjb_projection_mode,
            'hjb_t0': float(args.hjb_t0),
            'hjb_sigmoid_k': float(args.hjb_sigmoid_k),
            'hjb_control_cost_weight': float(args.hjb_control_cost_weight),
            'hjb_action_max_fm_ratio': float(args.hjb_action_max_fm_ratio),
            'hjb_replay_gate_checkpoint': args.hjb_replay_gate_checkpoint,
            'hjb_replay_gate_mode': args.hjb_replay_gate_mode,
            'hjb_replay_gate_threshold': float(args.hjb_replay_gate_threshold),
            'hjb_replay_gate_temperature': float(args.hjb_replay_gate_temperature),
            'hjb_ratio_selector_checkpoint': args.hjb_ratio_selector_checkpoint,
            'hjb_ratio_selector_mode': args.hjb_ratio_selector_mode,
            'hjb_ratio_candidates': args.hjb_ratio_candidates,
            'hjb_value_gradient_checkpoint': args.hjb_value_gradient_checkpoint,
            'hjb_actor_checkpoint': args.hjb_actor_checkpoint,
            'hjb_actor_mode': args.hjb_actor_mode,
            'hjb_actor_output_sign': float(args.hjb_actor_output_sign),
            'hjb_actor_output_projection': args.hjb_actor_output_projection,
            'controller_checkpoint': args.landscape_controller_checkpoint,
            'controller_info': landscape_guidance.get('controller_info'),
        },
    }

    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
    logger.info('Sample done!')
    logger.info(f'Result saved to {os.path.join(result_path, f"result_{args.data_id}.pt")}')

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from models.common import compose_context
from models.molopt_score_model import ScorePosNet3D_flow
from models.molopt_score_model_energy_guide import ScorePosNet3D_guided_flow


def center_joint_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = protein_pos.new_zeros((batch_protein.max().item() + 1, protein_pos.size(-1)))
        return protein_pos, ligand_pos, offset
    if mode != 'protein':
        raise NotImplementedError(f'Unsupported center_pos_mode: {mode}')

    offset = scatter_mean(protein_pos, batch_protein, dim=0)
    protein_pos = protein_pos - offset[batch_protein]
    ligand_pos = ligand_pos - offset[batch_ligand]
    return protein_pos, ligand_pos, offset


def infer_backbone_kind_from_state_dict(state_dict):
    if any(k.startswith('expert_pred.') for k in state_dict.keys()):
        return 'guided_flow'
    return 'flow'


def infer_feature_dims_from_state_dict(state_dict):
    protein_atom_feature_dim = state_dict['protein_atom_emb.weight'].shape[1]
    ligand_atom_feature_dim = state_dict['v_inference.2.weight'].shape[0]
    return protein_atom_feature_dim, ligand_atom_feature_dim


def build_backbone(backbone_kind, config, protein_atom_feature_dim, ligand_atom_feature_dim, device=None):
    if backbone_kind == 'flow':
        return ScorePosNet3D_flow(
            config,
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
            device=device,
        )
    if backbone_kind in {'guided_flow', 'energy_guided_flow'}:
        return ScorePosNet3D_guided_flow(
            config,
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
            device=device,
        )
    raise ValueError(f'Unknown backbone_kind: {backbone_kind}')


class PAFlowLandscapeEncoder(nn.Module):
    """
    Wrap a pretrained PAFlow backbone and expose graph-level latent states for
    PESLA-style trajectory landscape learning.

    This encoder reuses the shared protein-ligand backbone, pools the final
    hidden states into a graph embedding, and provides a helper interface for
    states produced by `scripts/build_landscape_dataset.py`.
    """

    def __init__(
        self,
        backbone,
        pooling='mean',
        graph_context='joint',
        graph_emb_dim=None,
        freeze_backbone=True,
        center_pos_mode=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.graph_context = graph_context
        self.hidden_dim = backbone.hidden_dim
        self.center_pos_mode = center_pos_mode or getattr(backbone, 'center_pos_mode', 'protein')

        if graph_context not in {'joint', 'ligand', 'protein'}:
            raise ValueError(f'Unsupported graph_context: {graph_context}')
        if pooling not in {'mean', 'sum', 'max'}:
            raise ValueError(f'Unsupported pooling mode: {pooling}')

        if graph_context == 'joint':
            readout_dim = self.hidden_dim * 2
        else:
            readout_dim = self.hidden_dim
        self.readout_dim = readout_dim
        self.graph_emb_dim = graph_emb_dim or readout_dim
        if self.graph_emb_dim == self.readout_dim:
            self.graph_projector = nn.Identity()
        else:
            self.graph_projector = nn.Linear(self.readout_dim, self.graph_emb_dim)

        if freeze_backbone:
            self.freeze_backbone()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        device='cpu',
        backbone_kind=None,
        pooling='mean',
        graph_context='joint',
        graph_emb_dim=None,
        freeze_backbone=True,
        center_pos_mode=None,
        strict=False,
    ):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'config' not in ckpt or 'model' not in ckpt:
            raise ValueError('Expected checkpoint with keys `config` and `model`.')

        state_dict = ckpt['model']
        config = ckpt['config'].model
        backbone_kind = backbone_kind or infer_backbone_kind_from_state_dict(state_dict)
        protein_atom_feature_dim, ligand_atom_feature_dim = infer_feature_dims_from_state_dict(state_dict)
        backbone = build_backbone(
            backbone_kind=backbone_kind,
            config=config,
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
            device=device,
        )

        load_info = backbone.load_state_dict(state_dict, strict=strict)
        unexpected = set(load_info.unexpected_keys)
        missing = set(load_info.missing_keys)

        if backbone_kind == 'flow':
            unexpected = {k for k in unexpected if not k.startswith('expert_pred.')}
        if unexpected or missing:
            raise RuntimeError(
                f'Checkpoint mismatch when loading {checkpoint_path}. '
                f'Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}.'
            )

        model = cls(
            backbone=backbone,
            pooling=pooling,
            graph_context=graph_context,
            graph_emb_dim=graph_emb_dim,
            freeze_backbone=freeze_backbone,
            center_pos_mode=center_pos_mode,
        )
        model.backbone_kind = backbone_kind
        model.checkpoint_path = checkpoint_path
        return model

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        return self

    def _pool_nodes(self, node_h, batch, num_graphs):
        if self.pooling == 'mean':
            return scatter_mean(node_h, batch, dim=0, dim_size=num_graphs)
        if self.pooling == 'sum':
            return scatter_sum(node_h, batch, dim=0, dim_size=num_graphs)
        if self.pooling == 'max':
            return scatter_max(node_h, batch, dim=0, dim_size=num_graphs)[0]
        raise ValueError(f'Unsupported pooling mode: {self.pooling}')

    def _ensure_device_tensor(self, x, dtype=None):
        device = next(self.graph_projector.parameters(), next(self.backbone.parameters())).device
        if isinstance(x, torch.Tensor):
            if dtype is None:
                return x.to(device)
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    def _prepare_batch_index(self, batch_index, num_nodes, device):
        if batch_index is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)
        if isinstance(batch_index, torch.Tensor):
            return batch_index.to(device=device, dtype=torch.long)
        return torch.as_tensor(batch_index, device=device, dtype=torch.long)

    def _prepare_time_step(self, time_step, num_graphs, device):
        if time_step is None:
            return torch.zeros(num_graphs, dtype=torch.long, device=device)
        if isinstance(time_step, (int, float)):
            return torch.full((num_graphs,), int(time_step), dtype=torch.long, device=device)

        time_step = torch.as_tensor(time_step, dtype=torch.long, device=device)
        if time_step.ndim == 0:
            return torch.full((num_graphs,), int(time_step.item()), dtype=torch.long, device=device)
        if time_step.numel() == 1:
            return time_step.expand(num_graphs)
        if time_step.numel() != num_graphs:
            raise ValueError(f'time_step has {time_step.numel()} values but num_graphs={num_graphs}.')
        return time_step

    def _prepare_ligand_atom_feature(self, ligand_v):
        if isinstance(ligand_v, torch.Tensor):
            ligand_v = ligand_v.to(next(self.backbone.parameters()).device)
        else:
            ligand_v = torch.as_tensor(ligand_v, device=next(self.backbone.parameters()).device)

        if ligand_v.ndim == 1:
            return F.one_hot(ligand_v.long(), num_classes=self.backbone.num_classes).float()
        if ligand_v.ndim == 2 and ligand_v.shape[-1] == self.backbone.num_classes:
            return ligand_v.float()
        raise ValueError(
            f'ligand_v must be index tensor of shape [N] or one-hot/logit tensor '
            f'of shape [N, {self.backbone.num_classes}], got {tuple(ligand_v.shape)}.'
        )

    def _prepare_ligand_input(self, ligand_feat, time_step, batch_ligand):
        if self.backbone.time_emb_dim <= 0:
            return ligand_feat

        if self.backbone.time_emb_mode == 'simple':
            time_feat = (time_step.float() / self.backbone.num_timesteps)[batch_ligand].unsqueeze(-1)
            return torch.cat([ligand_feat, time_feat], dim=-1)

        if self.backbone.time_emb_mode == 'sin':
            time_feat = self.backbone.time_emb(time_step.float())
            if time_feat.size(0) != ligand_feat.size(0):
                time_feat = time_feat[batch_ligand]
            return torch.cat([ligand_feat, time_feat], dim=-1)

        raise NotImplementedError(f'Unsupported time embedding mode: {self.backbone.time_emb_mode}')

    def _build_graph_embedding(self, ligand_graph_h, protein_graph_h):
        if self.graph_context == 'ligand':
            graph_h = ligand_graph_h
        elif self.graph_context == 'protein':
            graph_h = protein_graph_h
        else:
            graph_h = torch.cat([protein_graph_h, ligand_graph_h], dim=-1)
        return self.graph_projector(graph_h)

    def encode_nodes(
        self,
        protein_pos,
        protein_atom_feature,
        ligand_pos,
        ligand_v,
        batch_protein=None,
        batch_ligand=None,
        time_step=None,
        center_pos_mode=None,
        return_all=False,
        fix_x=False,
    ):
        device = next(self.backbone.parameters()).device
        protein_pos = self._ensure_device_tensor(protein_pos, dtype=torch.float32)
        protein_atom_feature = self._ensure_device_tensor(protein_atom_feature, dtype=torch.float32)
        ligand_pos = self._ensure_device_tensor(ligand_pos, dtype=torch.float32)
        ligand_feat = self._prepare_ligand_atom_feature(ligand_v)

        batch_protein = self._prepare_batch_index(batch_protein, len(protein_pos), device)
        batch_ligand = self._prepare_batch_index(batch_ligand, len(ligand_pos), device)
        num_graphs = int(max(batch_protein.max().item(), batch_ligand.max().item())) + 1
        time_step = self._prepare_time_step(time_step, num_graphs, device)

        center_mode = center_pos_mode or self.center_pos_mode
        protein_pos_centered, ligand_pos_centered, offset = center_joint_pos(
            protein_pos=protein_pos,
            ligand_pos=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            mode=center_mode,
        )

        h_protein = self.backbone.protein_atom_emb(protein_atom_feature)
        input_ligand_feat = self._prepare_ligand_input(ligand_feat, time_step, batch_ligand)
        h_ligand = self.backbone.ligand_atom_emb(input_ligand_feat)

        if self.backbone.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1, device=device)], dim=-1)
            h_ligand = torch.cat([h_ligand, torch.ones(len(h_ligand), 1, device=device)], dim=-1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos_centered,
            pos_ligand=ligand_pos_centered,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.backbone.refine_net(
            h_all,
            pos_all,
            mask_ligand,
            batch_all,
            return_all=return_all,
            fix_x=fix_x,
        )

        final_h, final_pos = outputs['h'], outputs['x']
        protein_mask = ~mask_ligand
        final_ligand_h = final_h[mask_ligand]
        final_protein_h = final_h[protein_mask]
        ligand_graph_h = self._pool_nodes(final_ligand_h, batch_all[mask_ligand], num_graphs)
        protein_graph_h = self._pool_nodes(final_protein_h, batch_all[protein_mask], num_graphs)
        graph_emb = self._build_graph_embedding(ligand_graph_h, protein_graph_h)

        result = {
            'graph_emb': graph_emb,
            'ligand_graph_emb': ligand_graph_h,
            'protein_graph_emb': protein_graph_h,
            'final_h': final_h,
            'final_pos': final_pos,
            'final_ligand_h': final_ligand_h,
            'final_protein_h': final_protein_h,
            'mask_ligand': mask_ligand,
            'batch_all': batch_all,
            'offset': offset,
            'time_step': time_step,
        }

        if return_all:
            layer_graph_embs = []
            layer_ligand_graph_embs = []
            layer_protein_graph_embs = []
            for layer_h in outputs['all_h']:
                layer_ligand_h = layer_h[mask_ligand]
                layer_protein_h = layer_h[protein_mask]
                layer_ligand_graph_h = self._pool_nodes(layer_ligand_h, batch_all[mask_ligand], num_graphs)
                layer_protein_graph_h = self._pool_nodes(layer_protein_h, batch_all[protein_mask], num_graphs)
                layer_graph_embs.append(self._build_graph_embedding(layer_ligand_graph_h, layer_protein_graph_h))
                layer_ligand_graph_embs.append(layer_ligand_graph_h)
                layer_protein_graph_embs.append(layer_protein_graph_h)
            result.update({
                'layer_graph_embs': layer_graph_embs,
                'layer_ligand_graph_embs': layer_ligand_graph_embs,
                'layer_protein_graph_embs': layer_protein_graph_embs,
                'all_h': outputs['all_h'],
                'all_x': outputs['all_x'],
            })

        return result

    def forward(self, *args, **kwargs):
        return self.encode_nodes(*args, **kwargs)

    def trajectory_index_to_time_step(self, state_index, num_steps, descending=True):
        if num_steps <= 1:
            return 0
        step_size = max(float(self.backbone.num_timesteps) / float(num_steps), 1.0)
        if descending:
            raw_time = (num_steps - 1 - state_index) * step_size
        else:
            raw_time = state_index * step_size
        raw_time = min(max(raw_time, 0.0), float(self.backbone.num_timesteps - 1))
        return int(round(raw_time))

    def encode_trajectory_state(
        self,
        trajectory: Dict[str, Any],
        state: Dict[str, Any],
        time_step: Optional[Union[int, torch.Tensor]] = None,
        return_all=False,
        fix_x=False,
    ):
        device = next(self.backbone.parameters()).device
        protein_pos = trajectory['protein_pos'].to(device)
        protein_atom_feature = trajectory['protein_atom_feature'].to(device)
        ligand_pos = state['ligand_pos'].to(device)
        ligand_v = state['ligand_v'].to(device)

        batch_protein = torch.zeros(len(protein_pos), dtype=torch.long, device=device)
        batch_ligand = torch.zeros(len(ligand_pos), dtype=torch.long, device=device)

        if time_step is None:
            time_step = self.trajectory_index_to_time_step(
                state_index=int(state['time_index']),
                num_steps=int(trajectory['num_ligand_steps']),
                descending=True,
            )

        return self.encode_nodes(
            protein_pos=protein_pos,
            protein_atom_feature=protein_atom_feature,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            time_step=time_step,
            return_all=return_all,
            fix_x=fix_x,
        )

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.hjb_value_model import select_hjb_value
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils import reconstruct, transforms
from energy.energy import compute_binding_energy

def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        # betas = np.linspace(-10, 10, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def to_torch(x, device):
    x = torch.from_numpy(x).float().to(device)
    return x

def to_torch_var(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=True)
    return x


def _zscore_scalar_tensor(value, mean, std):
    return (value - float(mean)) / max(float(std), 1e-6)


def _compute_sampling_state_risk_metrics(protein_pos, ligand_pos, batch_protein, batch_ligand):
    num_graphs = int(batch_ligand.max().item()) + 1 if ligand_pos.numel() > 0 else 0
    min_dist = torch.full((num_graphs,), float('nan'), dtype=ligand_pos.dtype, device=ligand_pos.device)
    clash_count = torch.zeros((num_graphs,), dtype=ligand_pos.dtype, device=ligand_pos.device)
    severe_clash_count = torch.zeros((num_graphs,), dtype=ligand_pos.dtype, device=ligand_pos.device)
    contact_count = torch.zeros((num_graphs,), dtype=ligand_pos.dtype, device=ligand_pos.device)
    center_drift = torch.full((num_graphs,), float('nan'), dtype=ligand_pos.dtype, device=ligand_pos.device)
    overburied = torch.zeros((num_graphs,), dtype=ligand_pos.dtype, device=ligand_pos.device)
    for graph_idx in range(num_graphs):
        lig_mask = batch_ligand == graph_idx
        prot_mask = batch_protein == graph_idx
        if not bool(lig_mask.any()) or not bool(prot_mask.any()):
            continue
        lig = ligand_pos[lig_mask]
        prot = protein_pos[prot_mask]
        dist = torch.cdist(lig, prot)
        graph_min = dist.min()
        min_dist[graph_idx] = graph_min
        clash_count[graph_idx] = (dist < 2.0).float().sum()
        severe_clash_count[graph_idx] = (dist < 1.5).float().sum()
        contact_count[graph_idx] = (dist < 4.0).float().sum()
        center_drift[graph_idx] = torch.linalg.norm(lig.mean(dim=0) - prot.mean(dim=0))
        overburied[graph_idx] = (graph_min < 2.4).float()
    risk_clash = torch.maximum((clash_count / 5.0).clamp(0.0, 1.0), (severe_clash_count / 1.0).clamp(0.0, 1.0))
    risk_overburied = overburied
    risk_drift = torch.sigmoid((center_drift - 4.0) / 1.0)
    contact_density = contact_count
    state_risk = risk_clash + risk_overburied + 0.5 * risk_drift
    return {
        'current_min_pl_distance': min_dist,
        'current_clash_count': clash_count,
        'current_severe_clash_count': severe_clash_count,
        'current_contact_count': contact_count,
        'current_center_drift': center_drift,
        'current_overburied': overburied,
        'current_risk_clash': risk_clash,
        'current_risk_overburied': risk_overburied,
        'current_risk_drift': risk_drift,
        'current_contact_density': contact_density,
        'current_state_risk': state_risk,
    }


def _sampling_relative_model_outputs(model, x, pred_base, correction_scale):
    if hasattr(model, 'forward_components'):
        comps = model.forward_components(x)
        if isinstance(comps, (list, tuple)):
            if len(comps) >= 3:
                total, pocket_base, relative = comps[:3]
            else:
                raise ValueError(f"Unexpected forward_components output length: {len(comps)}")
        else:
            raise ValueError("forward_components must return a tuple/list")
        out = {
            'pred': total,
            'relative': relative,
            'pocket_base': pocket_base,
            'correction': relative,
        }
        if isinstance(comps, (list, tuple)) and len(comps) >= 5:
            out['relative_safety'] = comps[3]
            out['relative_dock'] = comps[4]
        if isinstance(comps, (list, tuple)) and len(comps) >= 6:
            out['relative_raw'] = comps[5]
        return out
    correction = model(x)
    pred = pred_base + float(correction_scale) * correction
    zeros = torch.zeros_like(pred)
    return {
        'pred': pred,
        'relative': zeros,
        'pocket_base': zeros,
        'correction': correction,
    }


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits, uniform=None):
    if uniform is None:
        uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))




# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model

class ScorePosNet3D_guided_flow(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device=None):
        super().__init__()
        self.config = config

        # ===== energy regularization config =====
        self.use_energy_loss = getattr(config, 'use_energy_loss', False)
        self.lambda_energy = getattr(config, 'lambda_energy', 0.0)
        self.energy_cutoff = getattr(config, 'energy_cutoff', 1.5)
        self.use_ph_velocity_loss = getattr(config, 'use_ph_velocity_loss', False)
        self.lambda_ph_velocity = getattr(config, 'lambda_ph_velocity', 0.0)
        self.ph_velocity_delta_t = getattr(config, 'ph_velocity_delta_t', 1.0e-3)
        self.ph_loss_on_noisy_state = getattr(config, 'ph_loss_on_noisy_state', True)
        self.ph_conservative_mode = getattr(config, 'ph_conservative_mode', 'none')
        self.ph_conservative_alpha = getattr(config, 'ph_conservative_alpha', 0.0)
        self.ph_conservative_axis = getattr(config, 'ph_conservative_axis', 'z')
        print('Energy config:', self.use_energy_loss, self.lambda_energy, self.energy_cutoff)
        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        # self.v_mode = config.v_mode
        # assert self.v_mode == 'categorical'
        # self.v_net_type = getattr(config, 'v_net_type', 'mlp')
        # self.bond_loss = getattr(config, 'bond_loss', False)
        # self.bond_net_type = getattr(config, 'bond_net_type', 'pre_att')
        # self.loss_bond_weight = getattr(config, 'loss_bond_weight', 0.)
        # self.loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        # self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        # print(f'Loss pos mode {self.loss_pos_type} applied!')
        # print(f'Loss bond net type: {self.bond_net_type} '
        #       f'bond weight: {self.loss_bond_weight} non bond weight: {self.loss_non_bond_weight}')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        alphas_cumprod_v = np.cumprod(alphas_v, axis=0)
        alphas_cumprod_v_prev = np.append(1., alphas_cumprod_v[:-1])
        self.alphas_cumprod_v_prev = to_torch(alphas_cumprod_v_prev, device)
        self.alphas_cumprod_v = to_torch(alphas_cumprod_v, device)
        self.one_minus_alphas_cumprod_v = to_torch(1. - alphas_cumprod_v, device)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
        self.expert_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_pos, protein_v, batch_protein, ligand_xt, ligand_vt, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1

        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    ligand_vt,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([ligand_vt, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = ligand_vt

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=ligand_xt,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
        final_affinity_pred = scatter_mean(atom_affinity, batch_ligand)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'atom_affinity': atom_affinity,
            'final_affinity_pred': final_affinity_pred
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch, uniform):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0, uniform)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError


    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def loss_func(self, a, b):
        return (a - b)**2

    def VP_path_pos(self, alpha, x_0, x_1):
        x_t = alpha.sqrt() * x_1 + (1.0 - alpha).sqrt() * x_0
        return x_t

    def VP_path_v(self, v_1, t, batch):
        alphas_cumprod_v = extract(self.alphas_cumprod_v, t, batch)
        one_minus_alphas_cumprod_v = extract(self.one_minus_alphas_cumprod_v, t, batch)
        v_1_onehot = F.one_hot(v_1, self.num_classes)
        v_perturbed = v_1_onehot * alphas_cumprod_v + one_minus_alphas_cumprod_v * (1. / self.num_classes)
        uniform = torch.rand_like(v_perturbed)
        gumbel_noise = -torch.log(uniform + 1e-30) + 1e-30
        v_t = (v_perturbed/gumbel_noise).argmax(dim=-1)
        log_vt = index_to_log_onehot(v_t, self.num_classes)
        return v_t, log_vt, uniform


    def sqrt_a_bar_hat(self, delta_t, t, batch_ligand):
        a_bar_t = self.alphas_cumprod.index_select(0, t.int())
        if (t-delta_t*1000)[0] < 0:
            a_bar_t_sub_1 = self.alphas_cumprod_prev.index_select(0, t.int())
        else:
            a_bar_t_sub_1 = self.alphas_cumprod.index_select(0, (t-delta_t*1000).int())
        sqrt_a_bar_hat = -(torch.sqrt(a_bar_t_sub_1)-torch.sqrt(a_bar_t))/delta_t # delta_t = 0.001
        return sqrt_a_bar_hat

    def a_bar_hat(self, delta_t, t, batch_ligand): # 对alphas_cumprod求导
        a_bar_t = self.alphas_cumprod.index_select(0, t.int())
        if (t-delta_t*1000)[0] < 0:
            a_bar_t_sub_1 = self.alphas_cumprod_prev.index_select(0, t.int())
        else:
            a_bar_t_sub_1 = self.alphas_cumprod.index_select(0, (t-delta_t*1000).int())
        a_bar_hat = (a_bar_t_sub_1-a_bar_t)/delta_t # delta_t = 0.001
        return a_bar_hat

    def VP_field(self, x1, xt, t, delta_t, batch_ligand):
        a_bar_t = self.alphas_cumprod.index_select(0, t.int())
        # a_bar_t_sub_1 = self.alphas_cumprod_prev.index_select(0, t.int())

        sqrt_a_bar_hat = self.sqrt_a_bar_hat(delta_t, t, batch_ligand)
        M_para = sqrt_a_bar_hat[batch_ligand] / (1 - a_bar_t[batch_ligand] + 1e-5)
        vector = (
            torch.sqrt(a_bar_t[batch_ligand])[:, None] * xt - x1
        ) * M_para[:, None]

        return vector

    def q_v_posterior_ot(self, ligand_v1, v_0_onehot, ligand_vt_one_hot, delta_t, time_step, batch):
        ligand_v1_one_hot = F.one_hot(ligand_v1, num_classes=self.num_classes)
        dv = ligand_v1_one_hot - v_0_onehot
        nonzero_mask = (1 - (time_step == 0).float())[batch].unsqueeze(-1)
        ligand_logits_v_next = ligand_vt_one_hot + (dv * delta_t) * nonzero_mask
        ligand_v_prob = F.softmax(ligand_logits_v_next, dim=-1)
        return ligand_v_prob

    def log_p_y_posterior(self, pred_score, target):
        l = self.loss_func(pred_score, target)
        posterior = (0-l).exp()
        return torch.log(posterior)


    def get_VP_ba_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, affinity, batch_ligand, time_step=None,
            landscape_guidance=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        ligand_x1 = ligand_pos
        ligand_v1 = ligand_v

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        ligand_x0 = torch.zeros_like(ligand_x1)
        ligand_x0.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_xt = self.VP_path_pos(a_pos, ligand_x0, ligand_x1)
        use_landscape_ph = (
            landscape_guidance is not None and
            self.use_ph_velocity_loss and
            self.lambda_ph_velocity > 0
        )
        if use_landscape_ph and self.ph_loss_on_noisy_state:
            ligand_xt = ligand_xt.detach().requires_grad_(True)
        # xt_diff = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * ligand_x0  # x0 * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v1 = index_to_log_onehot(ligand_v1, self.num_classes)
        ligand_vt, log_ligand_vt, uniform = self.VP_path_v(v_1 = ligand_v1, t = time_step, batch = batch_ligand)
        # ligand_v_perturbed, log_ligand_vt_diff = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, uniform)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            ligand_xt=ligand_xt,
            ligand_vt=F.one_hot(ligand_vt, self.num_classes).float(),
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos_all = loss_pos
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v1, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v1,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # ba precditor loss
        pred_ba = preds['final_affinity_pred']
        loss_ba = F.mse_loss(pred_ba, affinity)

        # energy regularization
        loss_energy = torch.zeros((), device=protein_pos.device)
        E_steric = torch.zeros((), device=protein_pos.device)

        if self.use_energy_loss:
            energy_dict = compute_binding_energy(
                protein_pos=protein_pos,
                ligand_pos=pred_ligand_pos,   # 關鍵：用模型預測的 ligand 位置
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                cutoff=self.energy_cutoff,
            )
            loss_energy = energy_dict['E_total']
            E_steric = energy_dict['E_steric']

        loss_ph_velocity = torch.zeros((), device=protein_pos.device)
        ph_alpha = torch.zeros((), device=protein_pos.device)
        if use_landscape_ph:
            ph_state_pos = ligand_xt if self.ph_loss_on_noisy_state else ligand_pos
            ph_state_v = F.one_hot(ligand_vt, self.num_classes).float()
            ph_time_step = time_step
            encoder = landscape_guidance['encoder']
            landscape_model = landscape_guidance['model']
            soft_tau = float(landscape_guidance.get('soft_tau', 1.0))
            energy_mode = str(landscape_guidance.get('energy_mode', 'code'))
            ph_alpha = landscape_guidance.get('train_alpha', 0.0)
            if not torch.is_tensor(ph_alpha):
                ph_alpha = torch.tensor(float(ph_alpha), device=protein_pos.device)
            else:
                ph_alpha = ph_alpha.to(protein_pos.device)

            encoded = encoder.encode_nodes(
                protein_pos=protein_pos,
                protein_atom_feature=protein_v,
                ligand_pos=ph_state_pos,
                ligand_v=ph_state_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                time_step=ph_time_step,
                center_pos_mode=self.center_pos_mode,
                fix_x=True,
            )
            graph_emb = encoded['graph_emb']
            if hasattr(landscape_model, 'guidance_energy'):
                soft_energy = landscape_model.guidance_energy(graph_emb, tau=soft_tau, mode=energy_mode)
            else:
                z_e = landscape_model.projector(graph_emb)
                codebook = landscape_model.quantizer.codebook
                z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
                code_sq = torch.sum(codebook ** 2, dim=-1)
                distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, codebook.t())
                soft_assign = torch.softmax(-distances / max(soft_tau, 1e-6), dim=-1)
                soft_energy = torch.sum(soft_assign * landscape_model.code_energies.unsqueeze(0), dim=-1)
            landscape_pos_guidance = -torch.autograd.grad(
                soft_energy.sum(),
                ph_state_pos,
                create_graph=False,
                retain_graph=True,
            )[0]

            conservative_guidance = torch.zeros_like(landscape_pos_guidance)
            ph_conservative_alpha = torch.tensor(
                float(self.ph_conservative_alpha),
                device=protein_pos.device,
                dtype=landscape_pos_guidance.dtype,
            )
            if self.ph_conservative_mode != 'none' and float(self.ph_conservative_alpha) != 0.0:
                if self.ph_conservative_axis == 'x':
                    axis = torch.tensor([1.0, 0.0, 0.0], device=protein_pos.device, dtype=landscape_pos_guidance.dtype)
                elif self.ph_conservative_axis == 'y':
                    axis = torch.tensor([0.0, 1.0, 0.0], device=protein_pos.device, dtype=landscape_pos_guidance.dtype)
                elif self.ph_conservative_axis == 'xyz':
                    axis = torch.tensor([1.0, 1.0, 1.0], device=protein_pos.device, dtype=landscape_pos_guidance.dtype)
                    axis = axis / axis.norm().clamp_min(1e-6)
                else:
                    axis = torch.tensor([0.0, 0.0, 1.0], device=protein_pos.device, dtype=landscape_pos_guidance.dtype)

                # Conservative PH component: J grad(H). The existing guidance is -grad(H),
                # so recover grad(H) before applying the fixed skew rotation.
                grad_h = -landscape_pos_guidance
                conservative_guidance = torch.cross(axis.expand_as(grad_h), grad_h, dim=-1)

            pred_displacement = pred_ligand_pos - ligand_xt
            target_displacement = (
                (ligand_x1 - ligand_xt.detach())
                + ph_alpha * landscape_pos_guidance.detach()
                + ph_conservative_alpha * conservative_guidance.detach()
            )
            loss_ph_velocity = scatter_mean(
                ((pred_displacement - target_displacement) ** 2).sum(-1),
                batch_ligand,
                dim=0,
            ).mean()

        loss = (
            loss_pos +
            loss_ba +
            loss_v * self.loss_v_weight +
            self.lambda_energy * loss_energy +
            self.lambda_ph_velocity * loss_ph_velocity
        )

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'loss_ba': loss_ba,
            'x1': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_ba': pred_ba,
            # 'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
            # egergy
            'loss_energy': loss_energy,
            'E_total': loss_energy,
            'E_steric': E_steric,
            'loss_ph_velocity': loss_ph_velocity,
            'ph_alpha': ph_alpha.detach() if torch.is_tensor(ph_alpha) else torch.tensor(float(ph_alpha), device=protein_pos.device),
            'ph_conservative_alpha': torch.tensor(float(self.ph_conservative_alpha), device=protein_pos.device),
        }


    def sample_guided_flow_VP(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False,
                         noise=False, pos_grad_w=1.0, v_grad_w=0,
                         landscape_guidance=None):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
        protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)
        pos_traj, v_traj = [], []
        v1_pred_traj, vt_pred_traj = [], []
        velocity_trace = []
        ligand_xt, ligand_vt = init_ligand_pos, init_ligand_v
        prev_ligand_xt_for_trace = None
        prev_graph_emb_for_relative_h = None
        init_ligand_v_onehot = F.one_hot(ligand_vt, self.num_classes)
        ligand_vt_logits = init_ligand_v_onehot

        # time sequence
        time_seq = list(reversed(range(0, 50)))
        delta_t = 1/50

        def compute_soft_landscape_guidance(ligand_pos, ligand_v, time_step, current_progress, fm_dx=None):
            nonlocal prev_graph_emb_for_relative_h
            if landscape_guidance is None:
                return None, None, None
            if current_progress < float(landscape_guidance.get('late_start_fraction', 0.8)):
                return None, None, None

            encoder = landscape_guidance['encoder']
            landscape_model = landscape_guidance['model']
            soft_tau = float(landscape_guidance.get('soft_tau', 1.0))
            energy_mode = str(landscape_guidance.get('energy_mode', 'code'))

            encoded = encoder.encode_nodes(
                protein_pos=protein_pos,
                protein_atom_feature=protein_v,
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                time_step=time_step,
                center_pos_mode=center_pos_mode,
                fix_x=False if energy_mode in {'frozen_h_controller', 'frozen_h_relative', 'hjb_value'} else True,
            )
            graph_emb = encoded['graph_emb']
            protein_graph_emb = encoded.get('protein_graph_emb')
            if energy_mode == 'hjb_value':
                hjb_model = landscape_guidance.get('hjb_value_model')
                if hjb_model is None:
                    raise ValueError('hjb_value guidance requires a loaded HJB value model.')
                if fm_dx is None:
                    raise ValueError('hjb_value guidance requires fm_dx.')
                time_fraction = torch.full(
                    (int(batch_ligand.max().item()) + 1,),
                    float(current_progress),
                    dtype=ligand_pos.dtype,
                    device=ligand_pos.device,
                )
                value_out = hjb_model(
                    ligand_pos=ligand_pos,
                    ligand_v=ligand_v,
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_ligand=batch_ligand,
                    batch_protein=batch_protein,
                    time_fraction=time_fraction,
                )
                hjb_value_component = str(landscape_guidance.get('hjb_value_component', 'total'))
                score = select_hjb_value(
                    value_out,
                    hjb_value_component,
                    getattr(hjb_model, 'head_names', None),
                )
                grad_s = torch.autograd.grad(
                    score.sum(),
                    ligand_pos,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                grad_s = torch.nan_to_num(grad_s, nan=0.0, posinf=0.0, neginf=0.0)
                neg_grad_s = -grad_s
                schedule = torch.sigmoid(
                    torch.as_tensor(
                        float(landscape_guidance.get('hjb_sigmoid_k', 12.0))
                        * (float(current_progress) - float(landscape_guidance.get('hjb_t0', 0.5))),
                        dtype=ligand_pos.dtype,
                        device=ligand_pos.device,
                    )
                )

                def graph_norm_for_hjb(vec):
                    return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                def graph_dot_for_hjb(a, b):
                    return scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)

                multi_project_info = {}
                projected_neg_grad_s = neg_grad_s
                if hjb_value_component in {'dock_projected_multi', 'projected_multi', 'multi_projected'}:
                    if value_out.dim() == 1:
                        raise ValueError('dock_projected_multi requires a multi-head HJB value model.')
                    head_names = list(getattr(hjb_model, 'head_names', None) or [])
                    if not head_names:
                        head_names = ['dock', 'min', 'score', 'valid'][: value_out.size(-1)]
                    name_to_idx = {name: idx for idx, name in enumerate(head_names)}
                    if 'dock' not in name_to_idx:
                        raise ValueError(f'dock_projected_multi requires a dock head; available heads={head_names}')
                    projected_action = neg_grad_s
                    eps = 1e-8
                    for constraint_name in ('min', 'score', 'valid'):
                        idx = name_to_idx.get(constraint_name)
                        if idx is None or idx >= value_out.size(-1):
                            continue
                        grad_constraint = torch.autograd.grad(
                            value_out[:, idx].sum(),
                            ligand_pos,
                            create_graph=False,
                            retain_graph=True,
                        )[0]
                        harmful_dot = graph_dot_for_hjb(grad_constraint, projected_action)
                        grad_sq = graph_dot_for_hjb(grad_constraint, grad_constraint).clamp_min(eps)
                        remove_alpha = torch.clamp(harmful_dot / grad_sq, min=0.0)
                        projected_action = (
                            projected_action
                            - remove_alpha[batch_ligand].unsqueeze(-1) * grad_constraint
                        )
                        multi_project_info[f'hjb_project_{constraint_name}_active'] = (
                            harmful_dot > 0.0
                        ).to(dtype=ligand_pos.dtype).detach()
                        multi_project_info[f'hjb_project_{constraint_name}_alpha'] = remove_alpha.detach()
                    projected_neg_grad_s = projected_action

                def rigid_body_project_for_hjb(vec):
                    projected = torch.zeros_like(vec)
                    num_graphs_hjb = int(batch_ligand.max().item()) + 1
                    eye = torch.eye(3, dtype=vec.dtype, device=vec.device)
                    for graph_idx in range(num_graphs_hjb):
                        mask = batch_ligand == graph_idx
                        if not bool(mask.any()):
                            continue
                        pos_g = ligand_pos[mask]
                        vec_g = vec[mask]
                        trans = vec_g.mean(dim=0, keepdim=True)
                        rel_pos = pos_g - pos_g.mean(dim=0, keepdim=True)
                        rel_vec = vec_g - trans
                        if int(rel_pos.size(0)) < 2:
                            projected[mask] = trans.expand_as(vec_g)
                            continue
                        a_mat = (
                            rel_pos.pow(2).sum(dim=1).sum() * eye
                            - rel_pos.transpose(0, 1).matmul(rel_pos)
                        )
                        rhs = torch.cross(rel_pos, rel_vec, dim=1).sum(dim=0)
                        omega = torch.linalg.solve(
                            a_mat + 1e-6 * eye,
                            rhs.unsqueeze(-1),
                        ).squeeze(-1)
                        rot = torch.cross(omega.unsqueeze(0).expand_as(rel_pos), rel_pos, dim=1)
                        projected[mask] = trans + rot
                    return projected

                fm_norm_hjb = graph_norm_for_hjb(fm_dx)
                raw_neg_grad_norm_hjb = graph_norm_for_hjb(neg_grad_s)
                raw_dot_fm_neggrad_hjb = graph_dot_for_hjb(fm_dx, neg_grad_s)
                raw_cos_fm_neggrad_hjb = raw_dot_fm_neggrad_hjb / (
                    fm_norm_hjb * raw_neg_grad_norm_hjb
                ).clamp_min(1e-8)
                projection_mode = str(landscape_guidance.get('hjb_projection_mode', 'none'))
                projection_gate = torch.ones_like(raw_cos_fm_neggrad_hjb)
                if projection_mode == 'rigid_body':
                    projected_neg_grad_s = rigid_body_project_for_hjb(projected_neg_grad_s)
                elif projection_mode == 'positive_only':
                    projection_gate = (raw_cos_fm_neggrad_hjb >= 0.0).to(dtype=neg_grad_s.dtype)
                    projected_neg_grad_s = projected_neg_grad_s * projection_gate[batch_ligand].unsqueeze(-1)
                elif projection_mode == 'remove_negative_parallel':
                    fm_sq = fm_norm_hjb.pow(2).clamp_min(1e-8)
                    alpha = graph_dot_for_hjb(fm_dx, projected_neg_grad_s) / fm_sq
                    negative_alpha = torch.minimum(alpha, torch.zeros_like(alpha))
                    negative_parallel = negative_alpha[batch_ligand].unsqueeze(-1) * fm_dx
                    projected_neg_grad_s = projected_neg_grad_s - negative_parallel
                    projection_gate = (alpha >= 0.0).to(dtype=neg_grad_s.dtype)

                neg_grad_norm_hjb = graph_norm_for_hjb(projected_neg_grad_s)
                dot_fm_grad_hjb = graph_dot_for_hjb(fm_dx, grad_s)
                cos_fm_neggrad_hjb = graph_dot_for_hjb(fm_dx, neg_grad_s) / (
                    fm_norm_hjb * raw_neg_grad_norm_hjb
                ).clamp_min(1e-8)
                projected_cos_fm_neggrad_hjb = graph_dot_for_hjb(fm_dx, projected_neg_grad_s) / (
                    fm_norm_hjb * neg_grad_norm_hjb
                ).clamp_min(1e-8)
                feature_map = {
                    'hjb_score': score,
                    'hjb_neg_grad_norm': neg_grad_norm_hjb,
                    'hjb_raw_neg_grad_norm': raw_neg_grad_norm_hjb,
                    'hjb_fm_neggrad_cos': cos_fm_neggrad_hjb,
                    'hjb_projected_fm_neggrad_cos': projected_cos_fm_neggrad_hjb,
                    'hjb_fm_grad_dot': dot_fm_grad_hjb,
                    'hjb_schedule': torch.full_like(score, float(schedule.detach().item())),
                    'hjb_blend_rho': torch.full_like(score, float(landscape_guidance.get('hjb_blend_rho', 0.5))),
                }
                selected_fm_ratio = None
                ratio_selector_info = landscape_guidance.get('hjb_ratio_selector_info')
                ratio_selector_model = landscape_guidance.get('hjb_ratio_selector_model')
                if ratio_selector_info is not None and ratio_selector_model is not None:
                    candidates = torch.as_tensor(
                        ratio_selector_info.get('candidates', [0.0, 0.025, 0.05, 0.10, 0.15]),
                        dtype=ligand_pos.dtype,
                        device=ligand_pos.device,
                    )
                    if candidates.numel() == 0:
                        raise ValueError('HJB ratio selector requires at least one candidate ratio.')
                    feature_rows = []
                    for candidate in candidates:
                        values = []
                        for name in ratio_selector_info['feature_names']:
                            if name == 'rho':
                                values.append(torch.full_like(score, float(candidate.item())))
                            elif name == 'steps':
                                values.append(torch.ones_like(score))
                            else:
                                if name not in feature_map:
                                    raise ValueError(f'Unknown HJB ratio selector feature {name!r}')
                                values.append(feature_map[name])
                        feature_rows.append(torch.stack(values, dim=-1))
                    selector_features = torch.cat(feature_rows, dim=0)
                    selector_features = (
                        selector_features
                        - ratio_selector_info['feature_mean'].to(selector_features.device, selector_features.dtype)
                    ) / ratio_selector_info['feature_std'].to(selector_features.device, selector_features.dtype).clamp_min(1e-6)
                    selector_cost = ratio_selector_model(selector_features).view(candidates.numel(), -1).transpose(0, 1)
                    best_idx = selector_cost.argmin(dim=-1)
                    selected_fm_ratio = candidates.index_select(0, best_idx).to(dtype=projected_neg_grad_s.dtype)

                replay_gate = None
                replay_gate_info = landscape_guidance.get('hjb_replay_gate_info')
                replay_gate_model = landscape_guidance.get('hjb_replay_gate_model')
                if replay_gate_info is not None and replay_gate_model is not None:
                    gate_features = torch.stack(
                        [feature_map[name] for name in replay_gate_info['feature_names']],
                        dim=-1,
                    )
                    gate_features = (
                        gate_features
                        - replay_gate_info['feature_mean'].to(gate_features.device, gate_features.dtype)
                    ) / replay_gate_info['feature_std'].to(gate_features.device, gate_features.dtype).clamp_min(1e-6)
                    temperature = max(float(replay_gate_info.get('temperature', 1.0)), 1e-6)
                    replay_prob = torch.sigmoid(replay_gate_model(gate_features) / temperature)
                    replay_mode = str(replay_gate_info.get('mode', 'continuous'))
                    threshold = float(replay_gate_info.get('threshold', 0.5))
                    if replay_mode == 'hard':
                        replay_gate = (replay_prob >= threshold).to(projected_neg_grad_s.dtype)
                    elif replay_mode == 'thresholded':
                        replay_gate = replay_prob * (replay_prob >= threshold).to(projected_neg_grad_s.dtype)
                    else:
                        replay_gate = replay_prob
                    projected_neg_grad_s = projected_neg_grad_s * replay_gate[batch_ligand].unsqueeze(-1)

                actor_model = landscape_guidance.get('hjb_actor_model')
                actor_info = landscape_guidance.get('hjb_actor_info')
                value_gradient_model = landscape_guidance.get('hjb_value_gradient_model')
                value_gradient_guidance = None
                actor_guidance = None
                actor_input_direction = neg_grad_s.detach()
                if actor_model is not None and actor_info is not None and str(actor_info.get('mode', 'none')) != 'none':
                    actor_scalars = torch.stack(
                        [
                            time_fraction,
                            score.detach(),
                            raw_cos_fm_neggrad_hjb.detach(),
                            (raw_neg_grad_norm_hjb / fm_norm_hjb.clamp_min(1e-8)).detach(),
                        ],
                        dim=-1,
                    )
                    if value_gradient_model is not None:
                        value_gradient_guidance = value_gradient_model(
                            ligand_pos.detach(),
                            ligand_v.detach(),
                            batch_ligand,
                            neg_grad_s.detach(),
                            fm_dx.detach(),
                            actor_scalars,
                        )
                        value_gradient_guidance = torch.nan_to_num(
                            value_gradient_guidance,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        actor_input_direction = value_gradient_guidance.detach()
                        value_gradient_norm_hjb = graph_norm_for_hjb(value_gradient_guidance)
                        value_gradient_cos_fm_hjb = graph_dot_for_hjb(fm_dx, value_gradient_guidance) / (
                            fm_norm_hjb * value_gradient_norm_hjb
                        ).clamp_min(1e-8)
                        actor_scalars = torch.stack(
                            [
                                time_fraction,
                                score.detach(),
                                value_gradient_cos_fm_hjb.detach(),
                                (value_gradient_norm_hjb / fm_norm_hjb.clamp_min(1e-8)).detach(),
                            ],
                            dim=-1,
                        )
                    actor_kwargs = {}
                    if bool(getattr(actor_model, 'uses_protein_context', False)):
                        actor_kwargs = {
                            'protein_pos': protein_pos.detach(),
                            'protein_v': protein_v.detach(),
                            'batch_protein': batch_protein,
                        }
                    actor_guidance = actor_model(
                        ligand_pos.detach(),
                        ligand_v.detach(),
                        batch_ligand,
                        actor_input_direction,
                        fm_dx.detach(),
                        actor_scalars,
                        **actor_kwargs,
                    )
                    actor_guidance = torch.nan_to_num(actor_guidance, nan=0.0, posinf=0.0, neginf=0.0)
                    actor_guidance = float(landscape_guidance.get('hjb_actor_output_sign', 1.0)) * actor_guidance
                    actor_output_projection = str(landscape_guidance.get('hjb_actor_output_projection', 'none'))
                    if actor_output_projection == 'positive_only':
                        actor_norm = graph_norm_for_hjb(actor_guidance)
                        actor_cos = graph_dot_for_hjb(fm_dx, actor_guidance) / (
                            fm_norm_hjb * actor_norm
                        ).clamp_min(1e-8)
                        actor_gate = (actor_cos >= 0.0).to(dtype=actor_guidance.dtype)
                        actor_guidance = actor_guidance * actor_gate[batch_ligand].unsqueeze(-1)
                    elif actor_output_projection == 'remove_negative_parallel':
                        fm_sq_actor = fm_norm_hjb.pow(2).clamp_min(1e-8)
                        actor_alpha = graph_dot_for_hjb(fm_dx, actor_guidance) / fm_sq_actor
                        actor_negative_alpha = torch.minimum(actor_alpha, torch.zeros_like(actor_alpha))
                        actor_guidance = actor_guidance - actor_negative_alpha[batch_ligand].unsqueeze(-1) * fm_dx
                    elif actor_output_projection != 'none':
                        raise ValueError(f'Unsupported hjb_actor_output_projection: {actor_output_projection}')
                    if str(actor_info.get('mode', 'none')) == 'replace_neg_grad':
                        projected_neg_grad_s = actor_guidance

                bellman_action = None
                sampling_mode = str(landscape_guidance.get('hjb_sampling_mode', 'residual_guidance'))
                def cap_action_to_fm_ratio(action):
                    max_ratio = float(landscape_guidance.get('hjb_action_max_fm_ratio', float('inf')))
                    if max_ratio >= float('inf'):
                        return action
                    action_norm = graph_norm_for_hjb(action)
                    target_norm = max_ratio * fm_norm_hjb
                    action_scale = target_norm / action_norm.clamp_min(1e-8)
                    action_scale = torch.where(
                        action_norm > target_norm,
                        action_scale,
                        torch.ones_like(action_scale),
                    )
                    return action * action_scale[batch_ligand].unsqueeze(-1)

                if sampling_mode == 'bellman_consistent':
                    control_cost = float(landscape_guidance.get('hjb_control_cost_weight', 0.0))
                    if control_cost <= 0.0:
                        raise ValueError('bellman_consistent HJB sampling requires --hjb_control_cost_weight > 0.')
                    next_progress = min(1.0, float(current_progress) + float(delta_t))
                    base_next_pos = (ligand_pos + float(delta_t) * fm_dx).detach().requires_grad_(True)
                    next_time_fraction = torch.full(
                        (int(batch_ligand.max().item()) + 1,),
                        next_progress,
                        dtype=ligand_pos.dtype,
                        device=ligand_pos.device,
                    )
                    next_value_out = hjb_model(
                        ligand_pos=base_next_pos,
                        ligand_v=ligand_v.detach(),
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_ligand=batch_ligand,
                        batch_protein=batch_protein,
                        time_fraction=next_time_fraction,
                    )
                    next_score = select_hjb_value(
                        next_value_out,
                        str(landscape_guidance.get('hjb_value_component', 'total')),
                        getattr(hjb_model, 'head_names', None),
                    )
                    next_grad_s = torch.autograd.grad(
                        next_score.sum(),
                        base_next_pos,
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                    next_grad_s = torch.nan_to_num(next_grad_s, nan=0.0, posinf=0.0, neginf=0.0)
                    # One-step quadratic Bellman minimizer:
                    # argmin_u S(t+dt, x+dt(v_FM+u)) + c_u ||u||^2
                    # under a first-order expansion around the FM next state.
                    bellman_action = -(float(delta_t) / (2.0 * control_cost)) * next_grad_s
                    if projection_mode == 'rigid_body':
                        bellman_action = rigid_body_project_for_hjb(bellman_action)
                    elif projection_mode == 'positive_only':
                        bellman_norm = graph_norm_for_hjb(bellman_action)
                        bellman_cos = graph_dot_for_hjb(fm_dx, bellman_action) / (
                            fm_norm_hjb * bellman_norm
                        ).clamp_min(1e-8)
                        bellman_gate = (bellman_cos >= 0.0).to(dtype=bellman_action.dtype)
                        bellman_action = bellman_action * bellman_gate[batch_ligand].unsqueeze(-1)
                    elif projection_mode == 'remove_negative_parallel':
                        fm_sq = fm_norm_hjb.pow(2).clamp_min(1e-8)
                        bellman_alpha = graph_dot_for_hjb(fm_dx, bellman_action) / fm_sq
                        negative_alpha = torch.minimum(bellman_alpha, torch.zeros_like(bellman_alpha))
                        bellman_action = bellman_action - negative_alpha[batch_ligand].unsqueeze(-1) * fm_dx
                    if replay_gate is not None:
                        bellman_action = bellman_action * replay_gate[batch_ligand].unsqueeze(-1)
                    bellman_action = cap_action_to_fm_ratio(bellman_action)

                if sampling_mode == 'direct_full':
                    # The outer sampler adds landscape_dx to v_FM.  Returning
                    # (-grad s - v_FM) makes the final position velocity equal
                    # to the full-control HJB velocity when strength=1.
                    pos_guidance = projected_neg_grad_s - fm_dx
                elif sampling_mode == 'bellman_consistent':
                    pos_guidance = schedule * bellman_action
                elif sampling_mode == 'direct_scaled':
                    # Same full replacement idea, but use hjb_blend_rho as a
                    # step-size multiplier so the final velocity is
                    # rho * (-grad S) without retaining FM.
                    rho = float(landscape_guidance.get('hjb_blend_rho', 0.5))
                    pos_guidance = rho * projected_neg_grad_s - fm_dx
                elif sampling_mode == 'blended_full':
                    rho = float(landscape_guidance.get('hjb_blend_rho', 0.5))
                    pos_guidance = rho * (projected_neg_grad_s - fm_dx)
                elif sampling_mode == 'blended_schedule':
                    rho = float(landscape_guidance.get('hjb_blend_rho', 0.5))
                    pos_guidance = (rho * schedule) * (projected_neg_grad_s - fm_dx)
                else:
                    rho = float(landscape_guidance.get('hjb_blend_rho', 1.0))
                    residual_action = cap_action_to_fm_ratio(rho * projected_neg_grad_s)
                    pos_guidance = schedule * residual_action

                pos_guidance = torch.nan_to_num(pos_guidance, nan=0.0, posinf=0.0, neginf=0.0)

                neg_grad_norm_hjb = graph_norm_for_hjb(projected_neg_grad_s)
                projected_cos_fm_neggrad_hjb = graph_dot_for_hjb(fm_dx, projected_neg_grad_s) / (
                    fm_norm_hjb * neg_grad_norm_hjb
                ).clamp_min(1e-8)
                return pos_guidance, score.detach(), {
                    'hjb_score': score.detach(),
                    'hjb_value_component': hjb_value_component,
                    'hjb_schedule': torch.full_like(score.detach(), float(schedule.detach().item())),
                    'hjb_sampling_mode': sampling_mode,
                    'hjb_projection_mode': projection_mode,
                    'hjb_actor_output_sign': torch.full_like(score.detach(), float(landscape_guidance.get('hjb_actor_output_sign', 1.0))),
                    'hjb_blend_rho': torch.full_like(score.detach(), float(landscape_guidance.get('hjb_blend_rho', 0.5))),
                    'hjb_neg_grad_norm': neg_grad_norm_hjb.detach(),
                    'hjb_raw_neg_grad_norm': raw_neg_grad_norm_hjb.detach(),
                    'hjb_fm_neggrad_cos': cos_fm_neggrad_hjb.detach(),
                    'hjb_projected_fm_neggrad_cos': projected_cos_fm_neggrad_hjb.detach(),
                    'hjb_fm_grad_dot': dot_fm_grad_hjb.detach(),
                    'hjb_projection_gate': projection_gate.detach(),
                    'hjb_replay_gate': None if replay_gate is None else replay_gate.detach(),
                    'hjb_selected_fm_ratio': None if selected_fm_ratio is None else selected_fm_ratio.detach(),
                    'hjb_value_gradient_norm': None if value_gradient_guidance is None else graph_norm_for_hjb(value_gradient_guidance).detach(),
                    'hjb_actor_norm': None if actor_guidance is None else graph_norm_for_hjb(actor_guidance).detach(),
                    'hjb_bellman_action_norm': None if bellman_action is None else graph_norm_for_hjb(bellman_action).detach(),
                    **multi_project_info,
                }
            if energy_mode == 'frozen_h_controller':
                if fm_dx is None:
                    raise ValueError('frozen_h_controller guidance requires fm_dx.')
                pairwise_potential = landscape_guidance.get('pairwise_potential')
                path_model = landscape_guidance.get('path_model')
                controller = landscape_guidance.get('controller')
                if pairwise_potential is None or path_model is None or controller is None:
                    raise ValueError('frozen_h_controller requires pairwise potential, path model, and controller.')
                outputs = landscape_model(graph_emb)
                if 'quality_soft_assign' not in outputs:
                    raise ValueError('frozen_h_controller requires dual-codebook soft assignments.')
                topo_soft = outputs['soft_assign']
                quality_soft = outputs['quality_soft_assign']
                topo_values = pairwise_potential['topology_values'].to(graph_emb.device)
                quality_values = pairwise_potential['quality_values'].to(graph_emb.device)
                pair_values = pairwise_potential['pair_values'].to(graph_emb.device)
                topo_energy = torch.sum(topo_soft * topo_values.unsqueeze(0), dim=-1)
                quality_energy = torch.sum(quality_soft * quality_values.unsqueeze(0), dim=-1)
                pair_energy = torch.sum(torch.matmul(topo_soft, pair_values) * quality_soft, dim=-1)
                u_base = topo_energy + quality_energy + float(landscape_guidance.get('pair_lambda', 0.1)) * pair_energy
                time_fraction = (time_step.float() / 980.0).clamp(0.0, 1.0).unsqueeze(-1)
                h_input = torch.cat([graph_emb, protein_graph_emb, time_fraction, u_base.unsqueeze(-1)], dim=-1)
                h_corr = path_model(h_input)
                soft_energy = u_base + float(landscape_guidance.get('path_correction_scale', 0.5)) * h_corr
                neg_grad_h = -torch.autograd.grad(
                    soft_energy.sum(),
                    ligand_pos,
                    create_graph=False,
                    retain_graph=True,
                )[0]

                def graph_norm_for_controller(vec):
                    return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                def graph_cos_for_controller(a, b):
                    dot = scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)
                    return dot / (graph_norm_for_controller(a) * graph_norm_for_controller(b)).clamp_min(1e-8)

                raw_cos = graph_cos_for_controller(fm_dx, neg_grad_h)
                raw_ratio = graph_norm_for_controller(neg_grad_h) / graph_norm_for_controller(fm_dx).clamp_min(1e-8)
                controller_input = torch.cat(
                    [
                        graph_emb.detach(),
                        protein_graph_emb.detach(),
                        time_fraction.detach(),
                        soft_energy.detach().unsqueeze(-1),
                        u_base.detach().unsqueeze(-1),
                        raw_cos.detach().unsqueeze(-1),
                        raw_ratio.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
                params = controller(controller_input)
                if getattr(controller, 'mode', 'diag') == 'diag':
                    controller_guidance = neg_grad_h * params[batch_ligand]
                else:
                    controller_guidance = torch.bmm(params[batch_ligand], neg_grad_h.unsqueeze(-1)).squeeze(-1)
                return controller_guidance, soft_energy.detach(), {
                    'raw': neg_grad_h,
                    'controller': controller_guidance,
                    'u_base': u_base.detach(),
                    'h_corr': h_corr.detach(),
                    'topo_energy': topo_energy.detach(),
                    'quality_energy': quality_energy.detach(),
                    'pair_energy': pair_energy.detach(),
                }
            if energy_mode == 'frozen_h_relative':
                pairwise_potential = landscape_guidance.get('pairwise_potential')
                path_model = landscape_guidance.get('path_model')
                runtime_info = landscape_guidance.get('path_runtime_info')
                if pairwise_potential is None or path_model is None or runtime_info is None:
                    raise ValueError('frozen_h_relative requires pairwise potential, path model, and runtime metadata.')
                outputs = landscape_model(graph_emb)
                if 'quality_soft_assign' not in outputs:
                    raise ValueError('frozen_h_relative requires dual-codebook soft assignments.')
                topo_soft = outputs['soft_assign']
                quality_soft = outputs['quality_soft_assign']
                topo_values = pairwise_potential['topology_values'].to(graph_emb.device)
                quality_values = pairwise_potential['quality_values'].to(graph_emb.device)
                pair_values = pairwise_potential['pair_values'].to(graph_emb.device)
                topo_energy = torch.sum(topo_soft * topo_values.unsqueeze(0), dim=-1)
                quality_energy = torch.sum(quality_soft * quality_values.unsqueeze(0), dim=-1)
                pair_energy = torch.sum(torch.matmul(topo_soft, pair_values) * quality_soft, dim=-1)
                u_base = topo_energy + quality_energy + float(landscape_guidance.get('pair_lambda', 0.1)) * pair_energy
                time_fraction = (time_step.float() / 980.0).clamp(0.0, 1.0).unsqueeze(-1)
                chunks = [graph_emb, protein_graph_emb, time_fraction]
                input_u_mode = str(runtime_info.get('input_u_mode', 'base'))
                if input_u_mode == 'base':
                    chunks.append(u_base.unsqueeze(-1))
                elif input_u_mode == 'split':
                    chunks.append(torch.stack([topo_energy, quality_energy, pair_energy], dim=-1))
                elif input_u_mode == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown path-value input_u_mode: {input_u_mode}')
                metrics = _compute_sampling_state_risk_metrics(protein_pos, ligand_pos, batch_protein, batch_ligand)
                extra_cols = runtime_info.get('extra_feature_columns', [])
                extra_stats = runtime_info.get('extra_feature_stats', {})
                if extra_cols:
                    extra_stack = []
                    for col in extra_cols:
                        stat = extra_stats.get(col, {'mean': 0.0, 'std': 1.0})
                        extra_stack.append(
                            _zscore_scalar_tensor(metrics[col], stat.get('mean', 0.0), stat.get('std', 1.0))
                        )
                    chunks.append(torch.stack(extra_stack, dim=-1))

                prev_graph_emb = prev_graph_emb_for_relative_h
                if prev_graph_emb is None:
                    prev_graph_emb = torch.zeros_like(graph_emb)
                    has_prev = 0.0
                else:
                    has_prev = 1.0

                if bool(runtime_info.get('append_prev_state_emb', False)):
                    chunks.append(prev_graph_emb)
                if bool(runtime_info.get('append_delta_state_emb', False)):
                    chunks.append(graph_emb - prev_graph_emb)
                if bool(runtime_info.get('append_history_scalar_features', False)):
                    delta_state = graph_emb - prev_graph_emb
                    scalar_stack = torch.stack(
                        [
                            delta_state.norm(dim=-1),
                            graph_emb.norm(dim=-1),
                            prev_graph_emb.norm(dim=-1),
                            torch.full((graph_emb.shape[0],), float(has_prev), dtype=graph_emb.dtype, device=graph_emb.device),
                        ],
                        dim=-1,
                    )
                    chunks.append(scalar_stack)

                h_input = torch.cat(chunks, dim=-1)
                out_h = _sampling_relative_model_outputs(
                    path_model,
                    h_input,
                    u_base,
                    float(landscape_guidance.get('path_correction_scale', 0.5)),
                )
                component = str(landscape_guidance.get('path_value_component', 'auto'))
                if component == 'auto':
                    energy_key = 'relative' if bool(runtime_info.get('use_pocket_baseline_decomposition', False)) else 'pred'
                elif component == 'safety':
                    energy_key = 'relative_safety'
                elif component == 'dock':
                    energy_key = 'relative_dock'
                elif component == 'raw':
                    energy_key = 'relative_raw'
                else:
                    energy_key = component
                if energy_key not in out_h:
                    raise ValueError(f'Path-value component `{component}` resolved to missing output `{energy_key}`.')
                projection_mode = str(landscape_guidance.get('path_value_projection', 'none'))

                def graph_project_not_harm(delta, grad_constraint, eps=1e-8):
                    dot = scatter_sum((delta * grad_constraint).sum(dim=-1), batch_ligand, dim=0)
                    denom = scatter_sum(grad_constraint.pow(2).sum(dim=-1), batch_ligand, dim=0).clamp_min(float(eps))
                    coeff = torch.where(dot > 0, dot / denom, torch.zeros_like(dot))
                    return delta - coeff[batch_ligand].unsqueeze(-1) * grad_constraint

                def graph_project_perp(delta, axis, eps=1e-8):
                    dot = scatter_sum((delta * axis).sum(dim=-1), batch_ligand, dim=0)
                    denom = scatter_sum(axis.pow(2).sum(dim=-1), batch_ligand, dim=0).clamp_min(float(eps))
                    coeff = dot / denom
                    return delta - coeff[batch_ligand].unsqueeze(-1) * axis

                if projection_mode == 'none':
                    relative_energy = out_h[energy_key]
                    neg_grad_h = -torch.autograd.grad(
                        relative_energy.sum(),
                        ligand_pos,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                else:
                    if 'relative_safety' not in out_h or 'relative_dock' not in out_h:
                        raise ValueError(
                            f'Path-value projection `{projection_mode}` requires a dual-head checkpoint '
                            'with relative_safety and relative_dock outputs.'
                        )
                    if projection_mode in {'raw_projected', 'raw_dock_projected', 'raw_dock_vfm_projected'} and 'relative_raw' not in out_h:
                        raise ValueError(
                            f'Path-value projection `{projection_mode}` requires a triple-head checkpoint '
                            'with relative_raw output.'
                        )
                    grad_safe = torch.autograd.grad(
                        out_h['relative_safety'].sum(),
                        ligand_pos,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    grad_dock = torch.autograd.grad(
                        out_h['relative_dock'].sum(),
                        ligand_pos,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                    grad_raw = None
                    if 'relative_raw' in out_h:
                        grad_raw = torch.autograd.grad(
                            out_h['relative_raw'].sum(),
                            ligand_pos,
                            create_graph=False,
                            retain_graph=True,
                        )[0]
                    neg_grad_h = -grad_safe
                    if projection_mode in {'raw_projected', 'raw_dock_projected', 'raw_dock_vfm_projected'}:
                        neg_grad_h = graph_project_not_harm(neg_grad_h, grad_raw)
                    if projection_mode in {'dock_projected', 'double_projected', 'raw_dock_projected', 'raw_dock_vfm_projected'}:
                        neg_grad_h = graph_project_not_harm(neg_grad_h, grad_dock)
                    if projection_mode in {'vfm_orthogonal', 'double_projected', 'raw_dock_vfm_projected'}:
                        if fm_dx is None:
                            raise ValueError(f'Path-value projection `{projection_mode}` requires fm_dx.')
                        neg_grad_h = graph_project_perp(neg_grad_h, fm_dx)
                    relative_energy = out_h['relative_safety']
                prev_graph_emb_for_relative_h = graph_emb.detach()
                return neg_grad_h, relative_energy.detach(), {
                    'raw': neg_grad_h,
                    'u_base': u_base.detach(),
                    'h_pred': out_h['pred'].detach(),
                    'h_relative': out_h['relative'].detach(),
                    'h_safety': None if 'relative_safety' not in out_h else out_h['relative_safety'].detach(),
                    'h_dock': None if 'relative_dock' not in out_h else out_h['relative_dock'].detach(),
                    'h_raw': None if 'relative_raw' not in out_h else out_h['relative_raw'].detach(),
                    'h_component': relative_energy.detach(),
                    'topo_energy': topo_energy.detach(),
                    'quality_energy': quality_energy.detach(),
                    'pair_energy': pair_energy.detach(),
                }
            if energy_mode == 'prob_transition':
                prob_transition = landscape_guidance.get('prob_transition')
                if prob_transition is None:
                    raise ValueError('prob_transition guidance requires transition tensors.')
                z_e = landscape_model.projector(graph_emb)
                codebook = landscape_model.quantizer.codebook
                z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
                code_sq = torch.sum(codebook ** 2, dim=-1)
                distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, codebook.t())
                soft_assign = torch.softmax(-distances / max(soft_tau, 1e-6), dim=-1)
                hard_code = torch.argmax(soft_assign, dim=-1)
                transition = prob_transition['transition'].to(graph_emb.device)
                code_values = prob_transition['code_values'].to(graph_emb.device)
                base_prob = transition.index_select(0, hard_code).clamp_min(1e-12)
                prob_lambda = float(landscape_guidance.get('prob_transition_lambda', 1.0))
                logits = torch.log(base_prob) - prob_lambda * code_values.unsqueeze(0)
                next_prob = torch.softmax(logits, dim=-1)
                expected_code = torch.matmul(next_prob, codebook)
                prob_energy = 0.5 * (z_e - expected_code.detach()).pow(2).sum(dim=-1)
                topo_energy = torch.sum(soft_assign * code_values.unsqueeze(0), dim=-1)
                pos_guidance = -torch.autograd.grad(
                    prob_energy.sum(),
                    ligand_pos,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                return pos_guidance, prob_energy.detach(), {
                    'u_base': topo_energy.detach(),
                    'topo_energy': topo_energy.detach(),
                    'prob_transition_energy': prob_energy.detach(),
                    'prob_transition_entropy': (-(next_prob.clamp_min(1e-12) * next_prob.clamp_min(1e-12).log()).sum(dim=-1)).detach(),
                    'prob_transition_expected_value': torch.sum(next_prob * code_values.unsqueeze(0), dim=-1).detach(),
                }
            if energy_mode == 'prob_transition_disp':
                prob_transition = landscape_guidance.get('prob_transition')
                if prob_transition is None or 'mean_com_displacement' not in prob_transition:
                    raise ValueError('prob_transition_disp guidance requires displacement transition tensors.')
                z_e = landscape_model.projector(graph_emb)
                codebook = landscape_model.quantizer.codebook
                z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
                code_sq = torch.sum(codebook ** 2, dim=-1)
                distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, codebook.t())
                soft_assign = torch.softmax(-distances / max(soft_tau, 1e-6), dim=-1)
                hard_code = torch.argmax(soft_assign, dim=-1)
                transition = prob_transition['transition'].to(graph_emb.device)
                code_values = prob_transition['code_values'].to(graph_emb.device)
                mean_com_displacement = prob_transition['mean_com_displacement'].to(graph_emb.device)
                base_prob = transition.index_select(0, hard_code).clamp_min(1e-12)
                prob_lambda = float(landscape_guidance.get('prob_transition_lambda', 1.0))
                logits = torch.log(base_prob) - prob_lambda * code_values.unsqueeze(0)
                next_prob = torch.softmax(logits, dim=-1)
                disp_by_current = mean_com_displacement.index_select(0, hard_code)
                expected_disp = torch.sum(next_prob.unsqueeze(-1) * disp_by_current, dim=1)
                pos_guidance = expected_disp.index_select(0, batch_ligand)
                topo_energy = torch.sum(soft_assign * code_values.unsqueeze(0), dim=-1)
                entropy = -(next_prob.clamp_min(1e-12) * next_prob.clamp_min(1e-12).log()).sum(dim=-1)
                return pos_guidance, topo_energy.detach(), {
                    'u_base': topo_energy.detach(),
                    'topo_energy': topo_energy.detach(),
                    'prob_transition_entropy': entropy.detach(),
                    'prob_transition_expected_value': torch.sum(next_prob * code_values.unsqueeze(0), dim=-1).detach(),
                    'prob_transition_disp_norm': expected_disp.norm(dim=-1).detach(),
                }
            if energy_mode == 'pairwise':
                pairwise_potential = landscape_guidance.get('pairwise_potential')
                if pairwise_potential is None:
                    raise ValueError('pairwise landscape guidance requires pairwise_potential tensors.')
                outputs = landscape_model(graph_emb)
                if 'quality_soft_assign' not in outputs:
                    raise ValueError('pairwise landscape guidance requires dual-codebook soft assignments.')
                topo_soft = outputs['soft_assign']
                quality_soft = outputs['quality_soft_assign']
                topo_values = pairwise_potential['topology_values'].to(graph_emb.device)
                quality_values = pairwise_potential['quality_values'].to(graph_emb.device)
                pair_values = pairwise_potential['pair_values'].to(graph_emb.device)
                topo_energy = torch.sum(topo_soft * topo_values.unsqueeze(0), dim=-1)
                quality_energy = torch.sum(quality_soft * quality_values.unsqueeze(0), dim=-1)
                pair_energy = torch.sum(torch.matmul(topo_soft, pair_values) * quality_soft, dim=-1)
                soft_energy = (
                    topo_energy
                    + quality_energy
                    + float(landscape_guidance.get('pair_lambda', 0.1)) * pair_energy
                )
            elif energy_mode == 'value_gated_local':
                if not hasattr(landscape_model, 'guidance_energy'):
                    raise ValueError('value_gated_local guidance requires a landscape model with guidance_energy().')
                value_energy = landscape_model.guidance_energy(graph_emb, tau=soft_tau, mode='value')
                local_energy = landscape_model.guidance_energy(graph_emb, tau=soft_tau, mode='local')
                value_guidance = -torch.autograd.grad(
                    value_energy.sum(),
                    ligand_pos,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                local_guidance = -torch.autograd.grad(
                    local_energy.sum(),
                    ligand_pos,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                return (
                    value_guidance + local_guidance,
                    (value_energy + local_energy).detach(),
                    {'value': value_guidance, 'local': local_guidance},
                )
            elif hasattr(landscape_model, 'guidance_energy'):
                soft_energy = landscape_model.guidance_energy(graph_emb, tau=soft_tau, mode=energy_mode)
            else:
                z_e = landscape_model.projector(graph_emb)
                codebook = landscape_model.quantizer.codebook
                z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
                code_sq = torch.sum(codebook ** 2, dim=-1)
                distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, codebook.t())
                soft_assign = torch.softmax(-distances / max(soft_tau, 1e-6), dim=-1)
                soft_energy = torch.sum(soft_assign * landscape_model.code_energies.unsqueeze(0), dim=-1)
            pos_guidance = -torch.autograd.grad(
                soft_energy.sum(),
                ligand_pos,
                create_graph=False,
                retain_graph=True,
            )[0]
            return pos_guidance, soft_energy.detach(), None

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            with torch.enable_grad():
                t = torch.full(size=(num_graphs,), fill_value=delta_t*i*1000, dtype=torch.long, device=protein_pos.device)
                ligand_xt = ligand_xt.detach().requires_grad_(True)
                ligand_vt = F.one_hot(ligand_vt, self.num_classes).float().detach().requires_grad_(True)
                preds = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,

                    ligand_xt=ligand_xt,
                    ligand_vt=ligand_vt,
                    batch_ligand=batch_ligand,
                    time_step=t
                )

                x1_from_e = preds['pred_ligand_pos']
                v1_from_e = preds['pred_ligand_v']

                energy_dict = compute_binding_energy(
                    protein_pos=protein_pos,
                    ligand_pos=ligand_xt,
                    batch_protein=batch_protein,
                    batch_ligand=batch_ligand,
                    cutoff=self.energy_cutoff,
                )
                E_total = energy_dict['E_total']
                E_steric = energy_dict['E_steric']
                pos_guidance = -torch.autograd.grad(
                    E_total.sum(),
                    ligand_xt,
                    create_graph=True
                )[0]
                clash_pos_guidance = None
                clash_energy = None
                clash_weight = 0.0
                clash_late_start = 0.0
                if landscape_guidance is not None:
                    clash_weight = float(landscape_guidance.get('clash_guidance_weight', 0.0))
                    clash_late_start = float(landscape_guidance.get('clash_guidance_late_start_fraction', 0.0))
                if clash_weight > 0.0:
                    clash_cutoff = float(landscape_guidance.get('clash_guidance_cutoff', 2.0))
                    clash_dict = compute_binding_energy(
                        protein_pos=protein_pos,
                        ligand_pos=ligand_xt,
                        batch_protein=batch_protein,
                        batch_ligand=batch_ligand,
                        cutoff=clash_cutoff,
                    )
                    clash_energy = clash_dict['E_total']
                    clash_pos_guidance = -torch.autograd.grad(
                        clash_energy.sum(),
                        ligand_xt,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
                v_guidance = torch.zeros_like(ligand_vt)
                progress = 1.0 - (float(i) / float(max(num_steps - 1, 1)))
                landscape_pos_guidance = None
                soft_landscape_energy = None
                landscape_guidance_parts = None


            a_bar_hat = self.a_bar_hat(delta_t, t, batch_ligand)
            para_x = a_bar_hat / (2 * (self.alphas_cumprod.index_select(0, t.int())))

            from torch_scatter import scatter
            molecule_norms = scatter(pos_guidance.pow(2).sum(dim=1), batch_ligand, dim=0, reduce='sum').sqrt()

            # 加一個 gamma 參數控制 energy guidance 強度（可從 config 讀）
            energy_gamma = getattr(self.config, 'energy_guidance_gamma', 1.0)  # 預設 1.0
            with torch.enable_grad():
                fm_dx = self.VP_field(x1=x1_from_e, xt=ligand_xt, t=t, delta_t = delta_t, batch_ligand=batch_ligand)
                landscape_pos_guidance, soft_landscape_energy, landscape_guidance_parts = compute_soft_landscape_guidance(
                    ligand_pos=ligand_xt,
                    ligand_v=ligand_vt,
                    time_step=t,
                    current_progress=progress,
                    fm_dx=fm_dx,
                )
            binding_dx = para_x[batch_ligand].unsqueeze(1) * pos_guidance * pos_grad_w + energy_gamma * (-pos_guidance)
            clash_dx = torch.zeros_like(fm_dx)
            if clash_pos_guidance is not None and progress >= clash_late_start:
                clash_dx = clash_weight * clash_pos_guidance
                clash_target_fm_ratio = float(landscape_guidance.get('clash_guidance_target_fm_ratio', 0.0))
                if clash_target_fm_ratio > 0.0:
                    clash_norm_rescale = scatter_sum(clash_dx.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()
                    fm_norm_rescale = scatter_sum(fm_dx.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()
                    target_norm = clash_target_fm_ratio * fm_norm_rescale
                    clash_rescale = target_norm / clash_norm_rescale.clamp_min(1e-8)
                    clash_max_scale = float(landscape_guidance.get('clash_guidance_target_fm_ratio_max_scale', float('inf')))
                    if clash_max_scale < float('inf'):
                        clash_rescale = clash_rescale.clamp(max=clash_max_scale)
                    clash_rescale = torch.where(clash_norm_rescale > 1e-8, clash_rescale, torch.zeros_like(clash_rescale))
                    clash_dx = clash_dx * clash_rescale[batch_ligand].unsqueeze(-1)
            landscape_dx = torch.zeros_like(fm_dx)
            dx = fm_dx + binding_dx + clash_dx  # 往低能量方向
            landscape_gate = None
            landscape_raw_norm = None
            landscape_raw_ratio = None
            landscape_raw_cos = None
            if landscape_pos_guidance is not None:
                landscape_gamma = float(landscape_guidance.get('strength', 0.0))
                energy_mode = str(landscape_guidance.get('energy_mode', 'code'))
                if energy_mode == 'hjb_value':
                    # HJB value guidance returns a fully scaled residual
                    # velocity from compute_soft_landscape_guidance().
                    landscape_dx = landscape_pos_guidance
                    value_dx = None
                    local_dx = None
                elif landscape_guidance_parts is not None and energy_mode == 'value_gated_local':
                    value_dx = landscape_gamma * landscape_guidance_parts['value']
                    local_dx = landscape_gamma * landscape_guidance_parts['local']
                    landscape_dx = value_dx + local_dx
                else:
                    value_dx = None
                    local_dx = None
                    landscape_dx = landscape_gamma * landscape_pos_guidance

                def graph_norm_for_gate(vec):
                    return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                def graph_cos_for_gate(a, b):
                    dot = scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)
                    return dot / (graph_norm_for_gate(a) * graph_norm_for_gate(b)).clamp_min(1e-8)

                def graph_min_pl_distance():
                    num_graphs_gate = int(graph_norm_for_gate(fm_dx).size(0))
                    mins = torch.full(
                        (num_graphs_gate,),
                        float('inf'),
                        dtype=ligand_xt.dtype,
                        device=ligand_xt.device,
                    )
                    for graph_idx in range(num_graphs_gate):
                        lig_mask = batch_ligand == graph_idx
                        prot_mask = batch_protein == graph_idx
                        if bool(lig_mask.any()) and bool(prot_mask.any()):
                            d = torch.cdist(ligand_xt[lig_mask], protein_pos[prot_mask])
                            mins[graph_idx] = d.min()
                    return mins

                gate_mode = str(landscape_guidance.get('gate_mode', 'none'))
                fm_norm_gate = graph_norm_for_gate(fm_dx)
                landscape_raw_norm = graph_norm_for_gate(landscape_dx)
                landscape_raw_ratio = landscape_raw_norm / fm_norm_gate.clamp_min(1e-8)
                landscape_raw_cos = graph_cos_for_gate(fm_dx, landscape_dx)
                if gate_mode != 'none':
                    min_cos = float(landscape_guidance.get('gate_min_cos', 0.0))
                    min_ratio = float(landscape_guidance.get('gate_min_ratio', 0.0))
                    max_ratio = float(landscape_guidance.get('gate_max_ratio', float('inf')))
                    gate_target_dx = local_dx if local_dx is not None else landscape_dx
                    hard_gate = (
                        (graph_cos_for_gate(fm_dx, gate_target_dx) >= min_cos) &
                        ((graph_norm_for_gate(gate_target_dx) / fm_norm_gate.clamp_min(1e-8)) >= min_ratio) &
                        ((graph_norm_for_gate(gate_target_dx) / fm_norm_gate.clamp_min(1e-8)) <= max_ratio)
                    ).float()
                    if gate_mode == 'soft_alignment':
                        softness = max(float(landscape_guidance.get('gate_softness', 0.05)), 1e-6)
                        gate_cos = graph_cos_for_gate(fm_dx, gate_target_dx)
                        gate_ratio = graph_norm_for_gate(gate_target_dx) / fm_norm_gate.clamp_min(1e-8)
                        cos_gate = torch.sigmoid((gate_cos - min_cos) / softness)
                        ratio_gate = (gate_ratio >= min_ratio).float() * (gate_ratio <= max_ratio).float()
                        landscape_gate = cos_gate * ratio_gate
                    else:
                        landscape_gate = hard_gate
                    if local_dx is not None:
                        landscape_dx = value_dx + local_dx * landscape_gate[batch_ligand].unsqueeze(-1)
                    else:
                        landscape_dx = landscape_dx * landscape_gate[batch_ligand].unsqueeze(-1)

                risk_gate_mode = str(landscape_guidance.get('risk_gate_mode', 'none'))
                if risk_gate_mode != 'none':
                    num_graphs_risk = int(fm_norm_gate.size(0))
                    risk_gate = torch.zeros(num_graphs_risk, dtype=landscape_dx.dtype, device=landscape_dx.device)
                    if risk_gate_mode == 'energy_threshold' and soft_landscape_energy is not None:
                        energy_threshold = float(landscape_guidance.get('risk_gate_energy_threshold', 0.0))
                        risk_gate = torch.maximum(risk_gate, (soft_landscape_energy >= energy_threshold).float())
                    if risk_gate_mode in {'energy_quantile', 'energy_or_contact'} and soft_landscape_energy is not None:
                        q = float(landscape_guidance.get('risk_gate_energy_quantile', 0.75))
                        q = min(max(q, 0.0), 1.0)
                        threshold = torch.quantile(soft_landscape_energy.detach(), q)
                        risk_gate = torch.maximum(risk_gate, (soft_landscape_energy >= threshold).float())
                    if risk_gate_mode in {'contact', 'energy_or_contact'}:
                        min_pl_distance = float(landscape_guidance.get('risk_gate_min_pl_distance', 0.0))
                        if min_pl_distance > 0.0:
                            pl_min = graph_min_pl_distance()
                            risk_gate = torch.maximum(risk_gate, (pl_min <= min_pl_distance).float())
                    if landscape_gate is None:
                        landscape_gate = risk_gate
                    else:
                        landscape_gate = landscape_gate * risk_gate
                    if local_dx is not None:
                        landscape_dx = value_dx + local_dx * landscape_gate[batch_ligand].unsqueeze(-1)
                    else:
                        landscape_dx = landscape_dx * landscape_gate[batch_ligand].unsqueeze(-1)

                target_fm_ratio = float(landscape_guidance.get('target_fm_ratio', 0.0))
                selected_target_fm_ratio = None
                if (
                    landscape_guidance_parts is not None
                    and landscape_guidance_parts.get('hjb_selected_fm_ratio') is not None
                ):
                    selected_target_fm_ratio = landscape_guidance_parts['hjb_selected_fm_ratio'].to(
                        device=ligand_xt.device,
                        dtype=ligand_xt.dtype,
                    )
                if target_fm_ratio > 0.0 or selected_target_fm_ratio is not None:
                    def graph_norm_for_rescale(vec):
                        return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                    fm_norm_rescale = graph_norm_for_rescale(fm_dx)
                    landscape_norm_rescale = graph_norm_for_rescale(landscape_dx)
                    if selected_target_fm_ratio is None:
                        target_norm = target_fm_ratio * fm_norm_rescale
                    else:
                        target_norm = selected_target_fm_ratio * fm_norm_rescale
                    rescale = target_norm / landscape_norm_rescale.clamp_min(1e-8)
                    max_scale = float(landscape_guidance.get('target_fm_ratio_max_scale', float('inf')))
                    if max_scale < float('inf'):
                        rescale = rescale.clamp(max=max_scale)
                    rescale = torch.where(landscape_norm_rescale > 1e-8, rescale, torch.zeros_like(rescale))
                    landscape_dx = landscape_dx * rescale[batch_ligand].unsqueeze(-1)
                landscape_dx = torch.nan_to_num(landscape_dx, nan=0.0, posinf=0.0, neginf=0.0)
                dx = dx + landscape_dx

            if landscape_guidance is not None and bool(landscape_guidance.get('trace_velocity', False)):
                def graph_norm(vec):
                    return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                def graph_cos(a, b):
                    dot = scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)
                    return dot / (graph_norm(a) * graph_norm(b)).clamp_min(1e-8)

                fm_norm = graph_norm(fm_dx).detach().cpu()
                binding_norm = graph_norm(binding_dx).detach().cpu()
                clash_norm = graph_norm(clash_dx).detach().cpu()
                landscape_norm = graph_norm(landscape_dx).detach().cpu()
                total_norm = graph_norm(dx).detach().cpu()

                def graph_contact_and_geometry_metrics():
                    num_graphs_trace = int(fm_norm.size(0))
                    pl_min = torch.full((num_graphs_trace,), float('nan'), dtype=ligand_xt.dtype, device=ligand_xt.device)
                    contact_risk = torch.full_like(pl_min, float('nan'))
                    center_drift = torch.full_like(pl_min, float('nan'))
                    geom_step_rms = torch.full_like(pl_min, float('nan'))
                    lig_min = torch.full_like(pl_min, float('nan'))
                    cutoff = float(landscape_guidance.get('risk_gate_min_pl_distance', 2.0))
                    for graph_idx in range(num_graphs_trace):
                        lig_mask = batch_ligand == graph_idx
                        prot_mask = batch_protein == graph_idx
                        if bool(lig_mask.any()):
                            lig_pos = ligand_xt[lig_mask]
                            if lig_pos.size(0) > 1:
                                lig_dist = torch.pdist(lig_pos)
                                if lig_dist.numel() > 0:
                                    lig_min[graph_idx] = lig_dist.min()
                            if prev_ligand_xt_for_trace is not None:
                                prev_pos = prev_ligand_xt_for_trace[lig_mask]
                                if lig_pos.size(0) > 1 and prev_pos.size(0) == lig_pos.size(0):
                                    cur_d = torch.pdist(lig_pos)
                                    prev_d = torch.pdist(prev_pos)
                                    if cur_d.numel() == prev_d.numel() and cur_d.numel() > 0:
                                        geom_step_rms[graph_idx] = (cur_d - prev_d).pow(2).mean().sqrt()
                        if bool(lig_mask.any()) and bool(prot_mask.any()):
                            lig_pos = ligand_xt[lig_mask]
                            prot_pos = protein_pos[prot_mask]
                            d = torch.cdist(lig_pos, prot_pos)
                            pl_min[graph_idx] = d.min()
                            contact_risk[graph_idx] = torch.relu(torch.as_tensor(cutoff, dtype=d.dtype, device=d.device) - d).pow(2).mean()
                            center_drift[graph_idx] = (lig_pos.mean(dim=0) - prot_pos.mean(dim=0)).pow(2).sum().sqrt()
                    return {
                        'pl_min_distance': pl_min.detach().cpu(),
                        'contact_risk': contact_risk.detach().cpu(),
                        'center_drift': center_drift.detach().cpu(),
                        'geometry_step_rms': geom_step_rms.detach().cpu(),
                        'ligand_min_distance': lig_min.detach().cpu(),
                    }

                process_metrics = graph_contact_and_geometry_metrics()
                if landscape_gate is None:
                    num_graphs_trace = int(fm_norm.size(0))
                    landscape_gate_cpu = torch.ones(num_graphs_trace)
                    landscape_raw_ratio_cpu = (landscape_norm / fm_norm.clamp_min(1e-8)).detach().cpu()
                    landscape_raw_cos_cpu = graph_cos(fm_dx, landscape_dx).detach().cpu()
                else:
                    landscape_gate_cpu = landscape_gate.detach().cpu()
                    landscape_raw_ratio_cpu = landscape_raw_ratio.detach().cpu()
                    landscape_raw_cos_cpu = landscape_raw_cos.detach().cpu()
                velocity_trace.append({
                    'step_index': int(len(velocity_trace)),
                    'time_index': int(i),
                    'progress': float(progress),
                    'landscape_active': bool(landscape_pos_guidance is not None),
                    'soft_landscape_energy': None if soft_landscape_energy is None else soft_landscape_energy.detach().cpu(),
                    'u_global': None if not (landscape_guidance_parts is not None and 'u_base' in landscape_guidance_parts) else landscape_guidance_parts['u_base'].detach().cpu(),
                    'hjb_score': None if not (landscape_guidance_parts is not None and 'hjb_score' in landscape_guidance_parts) else landscape_guidance_parts['hjb_score'].detach().cpu(),
                    'hjb_schedule': None if not (landscape_guidance_parts is not None and 'hjb_schedule' in landscape_guidance_parts) else landscape_guidance_parts['hjb_schedule'].detach().cpu(),
                    'hjb_neg_grad_norm': None if not (landscape_guidance_parts is not None and 'hjb_neg_grad_norm' in landscape_guidance_parts) else landscape_guidance_parts['hjb_neg_grad_norm'].detach().cpu(),
                    'hjb_raw_neg_grad_norm': None if not (landscape_guidance_parts is not None and 'hjb_raw_neg_grad_norm' in landscape_guidance_parts) else landscape_guidance_parts['hjb_raw_neg_grad_norm'].detach().cpu(),
                    'hjb_fm_neggrad_cos': None if not (landscape_guidance_parts is not None and 'hjb_fm_neggrad_cos' in landscape_guidance_parts) else landscape_guidance_parts['hjb_fm_neggrad_cos'].detach().cpu(),
                    'hjb_projected_fm_neggrad_cos': None if not (landscape_guidance_parts is not None and 'hjb_projected_fm_neggrad_cos' in landscape_guidance_parts) else landscape_guidance_parts['hjb_projected_fm_neggrad_cos'].detach().cpu(),
                    'hjb_fm_grad_dot': None if not (landscape_guidance_parts is not None and 'hjb_fm_grad_dot' in landscape_guidance_parts) else landscape_guidance_parts['hjb_fm_grad_dot'].detach().cpu(),
                    'hjb_projection_gate': None if not (landscape_guidance_parts is not None and 'hjb_projection_gate' in landscape_guidance_parts) else landscape_guidance_parts['hjb_projection_gate'].detach().cpu(),
                    'hjb_replay_gate': None if not (landscape_guidance_parts is not None and landscape_guidance_parts.get('hjb_replay_gate') is not None) else landscape_guidance_parts['hjb_replay_gate'].detach().cpu(),
                    'hjb_selected_fm_ratio': None if not (landscape_guidance_parts is not None and landscape_guidance_parts.get('hjb_selected_fm_ratio') is not None) else landscape_guidance_parts['hjb_selected_fm_ratio'].detach().cpu(),
                    'hjb_blend_rho': None if not (landscape_guidance_parts is not None and 'hjb_blend_rho' in landscape_guidance_parts) else landscape_guidance_parts['hjb_blend_rho'].detach().cpu(),
                    'h_path_correction': None if not (landscape_guidance_parts is not None and 'h_corr' in landscape_guidance_parts) else landscape_guidance_parts['h_corr'].detach().cpu(),
                    'topo_energy': None if not (landscape_guidance_parts is not None and 'topo_energy' in landscape_guidance_parts) else landscape_guidance_parts['topo_energy'].detach().cpu(),
                    'quality_energy': None if not (landscape_guidance_parts is not None and 'quality_energy' in landscape_guidance_parts) else landscape_guidance_parts['quality_energy'].detach().cpu(),
                    'pair_energy': None if not (landscape_guidance_parts is not None and 'pair_energy' in landscape_guidance_parts) else landscape_guidance_parts['pair_energy'].detach().cpu(),
                    **process_metrics,
                    'fm_norm': fm_norm,
                    'binding_norm': binding_norm,
                    'clash_norm': clash_norm,
                    'landscape_norm': landscape_norm,
                    'total_norm': total_norm,
                    'fm_landscape_cos': graph_cos(fm_dx, landscape_dx).detach().cpu(),
                    'fm_total_cos': graph_cos(fm_dx, dx).detach().cpu(),
                    'landscape_to_fm_ratio': (landscape_norm / fm_norm.clamp_min(1e-8)).detach().cpu(),
                    'binding_to_fm_ratio': (binding_norm / fm_norm.clamp_min(1e-8)).detach().cpu(),
                    'clash_to_fm_ratio': (clash_norm / fm_norm.clamp_min(1e-8)).detach().cpu(),
                    'landscape_gate': landscape_gate_cpu,
                    'landscape_gate_active_rate': float((landscape_gate_cpu > 0).float().mean().item()),
                    'landscape_raw_fm_cos': landscape_raw_cos_cpu,
                    'landscape_raw_to_fm_ratio': landscape_raw_ratio_cpu,
                })

            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            prev_ligand_xt_for_trace = ligand_xt.detach().clone()
            if noise:
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_xt)
            else:
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask
            ligand_xt = ligand_pos_next

            if (t-delta_t*1000)[0] < 0:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v_prev.index_select(0, t.int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            else:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            v1_from_e_prob = F.softmax(v1_from_e, dim=-1)
            v1_from_e_index = torch.multinomial(v1_from_e_prob, num_samples=1).squeeze(-1)
            # 将索引转换为 one-hot 编码
            v1_from_e_one_hot = F.one_hot(v1_from_e_index, num_classes=self.num_classes)
            para_v = alpha_v_cumprod_hat * ((1-self.alphas_cumprod_v.index_select(0, t.int()))/self.alphas_cumprod_v.index_select(0, t.int()))

            dv = (v1_from_e_one_hot - init_ligand_v_onehot) * alpha_v_cumprod_hat[batch_ligand][:, None] + para_v[batch_ligand].unsqueeze(1) * v_guidance * v_grad_w


            ligand_logits_v_next = ligand_vt_logits + (dv * delta_t) * nonzero_mask

            ligand_vt_prob = F.softmax(ligand_logits_v_next, dim=-1)
            ligand_v_next = ligand_vt_prob.argmax(dim=-1)
            ligand_vt = ligand_v_next
            ligand_vt_logits = ligand_logits_v_next

            ori_ligand_pos = ligand_xt + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_vt.clone().cpu())

        ligand_x1 = ligand_xt
        ligand_v1 = ligand_vt
        ligand_x1 = ligand_x1 + offset[batch_ligand]
        return {
            'pos': ligand_x1,
            'v': ligand_v1,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'velocity_trace': velocity_trace,
            # 'v1_traj': v1_pred_traj,
            # 'vt_traj': vt_pred_traj
        }


    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()
        if (time_step == self.num_timesteps).all():
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)
            return kl_pos_prior, kl_v_prior

        # perturb pos and v
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        if self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        # t = [T-1, ... , 0]
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        return kl_pos, kl_v

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            fix_x=True
        )
        return preds

    @torch.no_grad()
    def sample_flow_VP(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False, noise=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v1_pred_traj, vt_pred_traj = [], []
        ligand_xt, ligand_vt = init_ligand_pos, init_ligand_v
        # ligand_vt_onehot = (torch.ones(len(batch_ligand), self.num_classes) * (1 / self.num_classes)).to(protein_pos.device)
        init_ligand_v_onehot = F.one_hot(ligand_vt, self.num_classes)
        ligand_vt_logits = init_ligand_v_onehot
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        # time_seq = list(reversed(range(0, 500)))
        delta_t = 0.001
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=delta_t*i*1000, dtype=torch.long, device=protein_pos.device)
            ligand_vt = F.one_hot(ligand_vt, self.num_classes).float()
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                ligand_xt=ligand_xt,
                ligand_vt=ligand_vt,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            # pos0_from_e = preds['pred_ligand_pos']
            # v0_from_e = preds['pred_ligand_v']
            x1_from_e = preds['pred_ligand_pos']
            v1_from_e = preds['pred_ligand_v']

            dx = self.VP_field(x1=x1_from_e, xt=ligand_xt, t=t, delta_t = delta_t, batch_ligand=batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            if noise:
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_xt)
            else:
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask
            ligand_xt = ligand_pos_next

            if (t-delta_t*1000)[0] < 0:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v_prev.index_select(0, t.int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            else:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            v1_from_e_prob = F.softmax(v1_from_e, dim=-1)
            v1_from_e_index = torch.multinomial(v1_from_e_prob, num_samples=1).squeeze(-1)
            # 将索引转换为 one-hot 编码
            v1_from_e_one_hot = F.one_hot(v1_from_e_index, num_classes=self.num_classes)

            dv = (v1_from_e_one_hot - init_ligand_v_onehot) * alpha_v_cumprod_hat[batch_ligand][:, None]

            ligand_logits_v_next = ligand_vt_logits + (dv * delta_t) * nonzero_mask

            ligand_vt_prob = F.softmax(ligand_logits_v_next, dim=-1)
            ligand_v_next = ligand_vt_prob.argmax(dim=-1)
            ligand_vt = ligand_v_next
            ligand_vt_logits = ligand_logits_v_next

            ori_ligand_pos = ligand_xt + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_vt.clone().cpu())

        ligand_x1 = ligand_xt
        ligand_v1 = ligand_vt
        ligand_x1 = ligand_x1 + offset[batch_ligand]
        return {
            'pos': ligand_x1,
            'v': ligand_v1,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
        }



class AtomCountPredictor(nn.Module):

    def __init__(self, dropout_rate=0.2):

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),    # input
            nn.BatchNorm1d(128),
            ShiftedSoftplus(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            ShiftedSoftplus(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            ShiftedSoftplus(),

            nn.Linear(128, 1)    # output
        )

    def forward(self, x):
        return self.fc(x)

    def get_loss(self, pred, target):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target)
        return loss


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

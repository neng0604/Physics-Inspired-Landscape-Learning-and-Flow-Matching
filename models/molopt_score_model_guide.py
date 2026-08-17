import math

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


def compute_hjb_schedule(
    reverse_progress,
    t0=0.65,
    sigmoid_k=14.0,
    late_taper_start=1.0,
    late_taper_k=30.0,
):
    """Return the usual activation schedule and an optional smooth late taper."""
    activation = torch.sigmoid((reverse_progress - float(t0)) * float(sigmoid_k))
    if float(late_taper_start) < 1.0:
        if not 0.0 <= float(late_taper_start) <= 1.0:
            raise ValueError('HJB late taper start must be in [0, 1].')
        taper = torch.sigmoid(
            (float(late_taper_start) - reverse_progress) * float(late_taper_k)
        )
    else:
        taper = torch.ones_like(reverse_progress)
    return activation * taper, taper

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
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, affinity, batch_ligand, time_step=None
    ):
        num_graphs = int(batch_protein.max().item() + 1)
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

        loss = loss_pos + loss_ba + loss_v * self.loss_v_weight

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
        }


    def sample_guided_flow_VP(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         init_ligand_v_logits=None,
                         categorical_reference_v=None,
                         num_steps=None, center_pos_mode=None, pos_only=False,
                         noise=False, pos_grad_w=1.0, v_grad_w=0, trace_velocity=False,
                         hjb_guidance=None, physics_guidance=None, local_affinity_guidance=None,
                         resume_after_step_index=None, flow_sde=None, branch_control=None,
                         branch_response_guidance=None,
                         chemistry_control=None, max_resume_steps=None,
                         categorical_replay=None, categorical_transition=None):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
        protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)
        branch_reference_pos = init_ligand_pos.detach().clone()
        pos_traj, v_traj, v_logits_traj = [], [], []
        velocity_trace = []
        v1_pred_traj, vt_pred_traj = [], []
        ligand_xt, ligand_vt = init_ligand_pos, init_ligand_v
        initial_ligand_v = ligand_vt.detach().clone()
        reference_v = (
            ligand_vt if categorical_reference_v is None
            else torch.as_tensor(
                categorical_reference_v, dtype=torch.long, device=ligand_vt.device
            )
        )
        if tuple(reference_v.shape) != tuple(ligand_vt.shape):
            raise ValueError('categorical_reference_v must match init_ligand_v shape')
        init_ligand_v_onehot = F.one_hot(reference_v, self.num_classes).float()
        if init_ligand_v_logits is None:
            ligand_vt_logits = F.one_hot(
                ligand_vt, self.num_classes
            ).to(dtype=ligand_xt.dtype)
        else:
            ligand_vt_logits = torch.as_tensor(
                init_ligand_v_logits, dtype=ligand_xt.dtype, device=ligand_xt.device,
            )
            expected_shape = (int(ligand_vt.numel()), int(self.num_classes))
            if tuple(ligand_vt_logits.shape) != expected_shape:
                raise ValueError(
                    f'init_ligand_v_logits must have shape {expected_shape}, '
                    f'got {tuple(ligand_vt_logits.shape)}'
                )
        categorical_replay = dict(categorical_replay or {})
        categorical_replay_states = categorical_replay.get('states')
        categorical_replay_active_steps = int(categorical_replay.get('active_steps', 0))
        if categorical_replay_states is not None:
            categorical_replay_states = torch.as_tensor(
                categorical_replay_states,
                dtype=torch.long,
                device=ligand_vt.device,
            )
            if categorical_replay_states.ndim != 2:
                raise ValueError('categorical_replay.states must have shape [steps, atoms]')
            if int(categorical_replay_states.size(1)) != int(ligand_vt.numel()):
                raise ValueError('categorical_replay.states atom count must match ligand atoms')
            if categorical_replay_active_steps <= 0:
                categorical_replay_active_steps = int(categorical_replay_states.size(0))
            if categorical_replay_active_steps > int(categorical_replay_states.size(0)):
                raise ValueError('categorical_replay.states does not cover active_steps')
            if categorical_replay_states.numel() and (
                int(categorical_replay_states.min()) < 0
                or int(categorical_replay_states.max()) >= int(self.num_classes)
            ):
                raise ValueError('categorical_replay.states contains an invalid class index')
        categorical_transition = dict(categorical_transition or {})
        categorical_transition_mode = str(
            categorical_transition.get('mode', 'argmax') or 'argmax'
        )
        if categorical_transition_mode not in {'argmax', 'coupled_gumbel'}:
            raise ValueError(
                f'Unknown categorical transition mode: {categorical_transition_mode}'
            )
        categorical_velocity_mode = str(
            categorical_transition.get('velocity_mode', 'sample') or 'sample'
        )
        if categorical_velocity_mode not in {
            'sample', 'rao_blackwell', 'stateless_gumbel'
        }:
            raise ValueError(
                f'Unknown categorical velocity mode: {categorical_velocity_mode}'
            )
        categorical_velocity_seed = int(
            categorical_transition.get('velocity_seed', 2021)
        )
        categorical_stream_ids = categorical_transition.get('stream_ids')
        if categorical_velocity_mode == 'stateless_gumbel':
            if categorical_stream_ids is None:
                categorical_stream_ids = torch.arange(
                    num_graphs, device=protein_pos.device, dtype=torch.long
                )
            else:
                categorical_stream_ids = torch.as_tensor(
                    categorical_stream_ids, device=protein_pos.device, dtype=torch.long
                ).reshape(-1)
            if int(categorical_stream_ids.numel()) != int(num_graphs):
                raise ValueError('categorical stream_ids must contain one id per graph')
        categorical_state_mode = str(
            categorical_transition.get('state_mode', 'hard') or 'hard'
        )
        if categorical_state_mode not in {'hard', 'simplex'}:
            raise ValueError(f'Unknown categorical state mode: {categorical_state_mode}')
        categorical_temperature_start = float(
            categorical_transition.get('temperature_start', 1.0)
        )
        categorical_temperature_end = float(
            categorical_transition.get(
                'temperature_end', categorical_temperature_start
            )
        )
        if categorical_temperature_start <= 0.0 or categorical_temperature_end <= 0.0:
            raise ValueError('Categorical state temperatures must be positive')
        if categorical_state_mode == 'simplex':
            ligand_vt = F.softmax(ligand_vt_logits, dim=-1)
        chemistry_control = dict(chemistry_control or {})
        chemistry_type_anchor_ratio = float(chemistry_control.get('type_anchor_ratio', 0.0))
        chemistry_type_max_scale = float(chemistry_control.get('type_max_scale', 10.0))
        chemistry_active_steps = int(chemistry_control.get('active_steps', 0))
        chemistry_type_anchor_mask = chemistry_control.get('type_anchor_mask')
        if chemistry_type_anchor_mask is not None:
            chemistry_type_anchor_mask = torch.as_tensor(
                chemistry_type_anchor_mask,
                dtype=ligand_xt.dtype,
                device=ligand_xt.device,
            ).reshape(-1)
            if int(chemistry_type_anchor_mask.numel()) != int(ligand_xt.size(0)):
                raise ValueError('chemistry_control.type_anchor_mask size must match ligand atoms')
        # Optional categorical component of a branch action.  The action lives
        # in logit coordinates, while its magnitude is defined intrinsically
        # by an exact KL trust-region budget at every active sampler step.
        branch_type_action = None if branch_control is None else branch_control.get('type_logit_action')
        branch_type_coefficient = (
            0.0 if branch_control is None else float(branch_control.get('type_coefficient', 0.0))
        )
        branch_type_target_base_ratio = (
            0.0 if branch_control is None
            else float(branch_control.get('type_target_base_ratio', 0.0))
        )
        branch_type_active_steps = (
            0 if branch_control is None else int(branch_control.get('type_active_steps', 0))
        )
        branch_type_max_scale = (
            100.0 if branch_control is None else float(branch_control.get('type_max_scale', 100.0))
        )
        if branch_type_action is not None:
            branch_type_action = torch.as_tensor(
                branch_type_action, dtype=ligand_xt.dtype, device=ligand_xt.device,
            )
            expected_shape = (int(ligand_xt.size(0)), int(self.num_classes))
            if tuple(branch_type_action.shape) != expected_shape:
                raise ValueError(
                    'branch_control.type_logit_action shape must be '
                    f'{expected_shape}, got {tuple(branch_type_action.shape)}'
                )
            # Per-atom constants do not alter a categorical distribution.
            branch_type_action = branch_type_action - branch_type_action.mean(
                dim=-1, keepdim=True
            )

        # time sequence
        time_seq = list(reversed(range(0, 50)))
        if resume_after_step_index is not None:
            start = max(0, min(len(time_seq), int(resume_after_step_index) + 1))
            time_seq = time_seq[start:]
        if max_resume_steps is not None:
            max_resume_steps = int(max_resume_steps)
            if max_resume_steps <= 0:
                raise ValueError('max_resume_steps must be positive when provided')
            time_seq = time_seq[:max_resume_steps]
        delta_t = 1/50
        flow_sde = dict(flow_sde or {})
        flow_sde_mode = str(flow_sde.get('mode', 'none') or 'none')
        flow_sde_dmax = float(flow_sde.get('dmax', 0.0))
        flow_sde_generator = None
        if flow_sde_mode != 'none':
            if flow_sde_mode != 'equivalent':
                raise ValueError(f'Unknown flow SDE mode: {flow_sde_mode}')
            if flow_sde_dmax <= 0.0:
                raise ValueError('flow_sde.dmax must be positive in equivalent mode')
            flow_sde_generator = torch.Generator(device=protein_pos.device)
            flow_sde_generator.manual_seed(int(flow_sde.get('seed', 2021)))
        hjb_gate_enabled = bool(hjb_guidance is not None and hjb_guidance.get('adaptive_de_gate', False))
        hjb_gate_choice = None
        hjb_gate_cos_sum = None
        hjb_gate_cos_count = None
        if hjb_gate_enabled:
            hjb_gate_choice = torch.full((num_graphs,), -1, dtype=torch.long, device=protein_pos.device)
            hjb_gate_cos_sum = torch.zeros(num_graphs, dtype=protein_pos.dtype, device=protein_pos.device)
            hjb_gate_cos_count = torch.zeros(num_graphs, dtype=protein_pos.dtype, device=protein_pos.device)

        for guidance_step_index, i in enumerate(tqdm(time_seq, desc='sampling', total=len(time_seq))):
            with torch.enable_grad():
                t = torch.full(size=(num_graphs,), fill_value=delta_t*i*1000, dtype=torch.long, device=protein_pos.device)
                ligand_xt = ligand_xt.detach().requires_grad_(True)
                if categorical_state_mode == 'simplex':
                    generation_progress = float(49 - int(i)) / 49.0
                    log_temperature = (
                        (1.0 - generation_progress)
                        * math.log(categorical_temperature_start)
                        + generation_progress * math.log(categorical_temperature_end)
                    )
                    state_temperature = math.exp(log_temperature)
                    ligand_vt = F.softmax(
                        ligand_vt_logits / state_temperature, dim=-1
                    ).detach().requires_grad_(True)
                else:
                    ligand_vt = F.one_hot(
                        ligand_vt, self.num_classes
                    ).float().detach().requires_grad_(True)
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

                pred_affinity = preds['final_affinity_pred']

                log_p_y_posterior = self.log_p_y_posterior(pred_affinity, 1)
                pos_guidance = torch.autograd.grad(log_p_y_posterior, ligand_xt, grad_outputs=torch.ones_like(log_p_y_posterior),retain_graph=True)[0]
                v_guidance = torch.autograd.grad(log_p_y_posterior, ligand_vt, grad_outputs=torch.ones_like(log_p_y_posterior),retain_graph=True)[0]


            a_bar_hat = self.a_bar_hat(delta_t, t, batch_ligand)
            para_x = a_bar_hat / (2 * (self.alphas_cumprod.index_select(0, t.int())))

            from torch_scatter import scatter
            molecule_norms = scatter(pos_guidance.pow(2).sum(dim=1), batch_ligand, dim=0, reduce='sum').sqrt()

            fm_dx = self.VP_field(x1=x1_from_e, xt=ligand_xt, t=t, delta_t = delta_t, batch_ligand=batch_ligand)
            binding_dx = para_x[batch_ligand].unsqueeze(1) * pos_guidance * pos_grad_w
            flow_sde_D = None
            flow_sde_score = None
            flow_sde_drift = None
            if flow_sde_mode == 'equivalent':
                a_pos = extract(self.alphas_cumprod, t, batch_ligand)
                one_minus_a = (1.0 - a_pos).clamp_min(1e-6)
                flow_sde_D = flow_sde_dmax * one_minus_a
                flow_sde_score = -(ligand_xt - a_pos.sqrt() * x1_from_e) / one_minus_a
                flow_sde_drift = flow_sde_D * flow_sde_score
            base_dx = fm_dx + binding_dx
            if flow_sde_drift is not None:
                base_dx = base_dx + flow_sde_drift
            dx = base_dx
            hjb_stats = {}
            physics_stats = {}
            local_affinity_stats = {}
            branch_response_stats = {}
            hjb_dx = None
            hjb_gate_active_choice = None
            atom_raw_dv = None
            atom_target_base_ratio = 0.0
            atom_schedule = None
            def graph_norm(vec):
                return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

            def graph_dot(a, b):
                return scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)

            def remove_inward_normal(vec):
                out = vec.clone()
                num_graphs_local = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                for graph_idx in range(num_graphs_local):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    if not (bool(lm.any()) and bool(pm.any())):
                        continue
                    lig_pos_g = ligand_xt[lm]
                    prot_pos_g = protein_pos[pm]
                    d = torch.cdist(lig_pos_g, prot_pos_g)
                    nearest = prot_pos_g.index_select(0, d.argmin(dim=1))
                    normal = lig_pos_g - nearest
                    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    inward = (out[lm] * normal).sum(dim=-1, keepdim=True).clamp(max=0.0)
                    out[lm] = out[lm] - inward * normal
                return out

            def apply_protein_normal_mobility(
                vec,
                strength=0.60,
                radius=2.40,
                softness=0.35,
                inward_only=True,
            ):
                out = vec.clone()
                num_graphs_local = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                for graph_idx in range(num_graphs_local):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    if not (bool(lm.any()) and bool(pm.any())):
                        continue
                    lig_pos_g = ligand_xt[lm]
                    prot_pos_g = protein_pos[pm]
                    d = torch.cdist(lig_pos_g, prot_pos_g)
                    nearest_dist, nearest_idx = d.min(dim=1)
                    nearest = prot_pos_g.index_select(0, nearest_idx)
                    normal = lig_pos_g - nearest
                    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    normal_comp = (out[lm] * normal).sum(dim=-1, keepdim=True)
                    if inward_only:
                        normal_comp = normal_comp.clamp(max=0.0)
                    weight = torch.sigmoid((float(radius) - nearest_dist).unsqueeze(-1) / float(softness))
                    out[lm] = out[lm] - float(strength) * weight * normal_comp * normal
                return out

            def apply_contact_barrier_filter(
                proposal_dx,
                base_vec,
                min_dist=1.60,
                active_dist=2.10,
                kappa=0.50,
                strength=1.00,
            ):
                out = proposal_dx.clone()
                num_graphs_local = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                for graph_idx in range(num_graphs_local):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    if not (bool(lm.any()) and bool(pm.any())):
                        continue
                    lig_pos_g = ligand_xt[lm]
                    prot_pos_g = protein_pos[pm]
                    d = torch.cdist(lig_pos_g, prot_pos_g)
                    nearest_dist, nearest_idx = d.min(dim=1)
                    nearest = prot_pos_g.index_select(0, nearest_idx)
                    normal = lig_pos_g - nearest
                    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    total_normal_speed = ((base_vec[lm] + out[lm]) * normal).sum(dim=-1, keepdim=True)
                    h = nearest_dist.unsqueeze(-1) - float(min_dist)
                    lower_bound = -float(kappa) * h
                    active = nearest_dist.unsqueeze(-1) < float(active_dist)
                    correction = (lower_bound - total_normal_speed).clamp_min(0.0) * normal
                    out[lm] = out[lm] + float(strength) * active.to(out.dtype) * correction
                return out

            def apply_feasible_corridor_filter(
                proposal_dx,
                base_vec,
                target_ratio_graph,
                schedule_graph,
                mode='clash',
                eta=0.25,
                iters=2,
                clash_radius=1.75,
                severe_radius=1.45,
                overburied_radius=2.20,
                drift_radius=8.0,
            ):
                out = proposal_dx.clone()
                num_graphs_local = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                names = ['clash', 'severe']
                if mode in {'clash_overburied', 'overburied'}:
                    names += ['overburied']
                elif mode == 'full':
                    names += ['overburied', 'drift']
                elif mode != 'clash':
                    raise ValueError(f'Unknown feasible-corridor mode: {mode}')

                cost_lists = {name: [] for name in names}
                with torch.enable_grad():
                    for graph_idx in range(num_graphs_local):
                        lm = batch_ligand == graph_idx
                        pm = batch_protein == graph_idx
                        zero = ligand_xt.new_tensor(0.0)
                        if bool(lm.any()) and bool(pm.any()):
                            lig_pos_g = ligand_xt[lm]
                            prot_pos_g = protein_pos[pm]
                            d = torch.cdist(lig_pos_g, prot_pos_g).clamp_min(1e-4)
                            if 'clash' in cost_lists:
                                cost_lists['clash'].append(
                                    torch.relu(ligand_xt.new_tensor(clash_radius) - d).pow(2).mean()
                                )
                            if 'severe' in cost_lists:
                                cost_lists['severe'].append(
                                    torch.relu(ligand_xt.new_tensor(severe_radius) - d).pow(2).mean()
                                )
                            if 'overburied' in cost_lists:
                                nearest_lig = d.min(dim=1).values
                                cost_lists['overburied'].append(
                                    torch.relu(ligand_xt.new_tensor(overburied_radius) - nearest_lig).pow(2).mean()
                                )
                            if 'drift' in cost_lists:
                                lig_center = lig_pos_g.mean(dim=0)
                                pocket_center = prot_pos_g.mean(dim=0)
                                center_dist = (lig_center - pocket_center).norm()
                                cost_lists['drift'].append(
                                    torch.relu(center_dist - ligand_xt.new_tensor(drift_radius)).pow(2)
                                )
                        else:
                            for name in names:
                                cost_lists[name].append(zero)

                    constraints = []
                    for name in names:
                        cost_vec = torch.stack(cost_lists[name], dim=0)
                        grad = torch.autograd.grad(
                            cost_vec.sum(),
                            ligand_xt,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is None:
                            grad = torch.zeros_like(ligand_xt)
                        constraints.append((name, cost_vec.detach(), grad.detach()))

                active_count = ligand_xt.new_zeros(num_graphs_local)
                violation_sum = ligand_xt.new_zeros(num_graphs_local)
                for _, cost_vec, grad in constraints:
                    g_norm_sq = graph_dot(grad, grad)
                    active = (cost_vec > 1e-8) & (g_norm_sq > 1e-8)
                    active_f = active.to(out.dtype)
                    active_count = active_count + active_f
                    violation_sum = violation_sum + cost_vec.to(out.dtype) * active_f

                before = out
                n_iters = max(int(iters), 1)
                base_norm_local = graph_norm(base_vec).clamp_min(1e-8)
                radius = (target_ratio_graph * base_norm_local * schedule_graph).clamp_min(1e-8)
                for _ in range(n_iters):
                    for _, cost_vec, grad in constraints:
                        g_norm_sq = graph_dot(grad, grad).clamp_min(1e-8)
                        violation = cost_vec.to(out.dtype).clamp_min(0.0)
                        active = ((cost_vec > 1e-8) & (g_norm_sq > 1e-8)).to(out.dtype)
                        # Enforce grad C · (base + u) <= -eta [C]_+ where possible.
                        bound = -float(eta) * violation - graph_dot(grad, base_vec)
                        dot = graph_dot(grad, out)
                        excess = (dot - bound).clamp_min(0.0) * active
                        out = out - (excess / g_norm_sq)[batch_ligand].unsqueeze(-1) * grad
                    out_norm = graph_norm(out).clamp_min(1e-8)
                    cap = (radius / out_norm).clamp(max=1.0)
                    out = out * cap[batch_ligand].unsqueeze(-1)

                delta_norm = graph_norm(out - before)
                return out, {
                    'hjb_fc_delta_norm': delta_norm.detach().cpu(),
                    'hjb_fc_active_count': active_count.detach().cpu(),
                    'hjb_fc_violation_sum': violation_sum.detach().cpu(),
                }

            def remove_ligand_bond_stretch(vec, cutoff=1.85, max_pairs=256):
                out = vec.clone()
                num_graphs_local = int(batch_ligand.max().item() + 1)
                for graph_idx in range(num_graphs_local):
                    lm = batch_ligand == graph_idx
                    if not bool(lm.any()):
                        continue
                    pos_g = ligand_xt[lm]
                    vec_g = out[lm]
                    if int(pos_g.size(0)) < 2:
                        continue
                    dist = torch.cdist(pos_g, pos_g)
                    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
                    pair_mask = upper & (dist > 0.6) & (dist < float(cutoff))
                    pair_idx = pair_mask.nonzero(as_tuple=False)
                    if int(pair_idx.numel()) == 0:
                        continue
                    if int(pair_idx.size(0)) > int(max_pairs):
                        pair_idx = pair_idx[: int(max_pairs)]
                    i_idx, j_idx = pair_idx[:, 0], pair_idx[:, 1]
                    bond_vec = pos_g[i_idx] - pos_g[j_idx]
                    bond_unit = bond_vec / bond_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    stretch = ((vec_g[i_idx] - vec_g[j_idx]) * bond_unit).sum(dim=-1, keepdim=True)
                    correction = 0.5 * stretch * bond_unit
                    accum = torch.zeros_like(vec_g)
                    counts = torch.zeros(vec_g.size(0), 1, dtype=vec_g.dtype, device=vec_g.device)
                    accum.index_add_(0, i_idx, correction)
                    accum.index_add_(0, j_idx, -correction)
                    counts.index_add_(0, i_idx, torch.ones_like(stretch))
                    counts.index_add_(0, j_idx, torch.ones_like(stretch))
                    out[lm] = vec_g - accum / counts.clamp_min(1.0)
                return out

            def project_residual(raw_dx, projection_mode):
                raw_dot_base = graph_dot(raw_dx, base_dx)
                if projection_mode == 'positive_only':
                    keep = (raw_dot_base >= 0).to(raw_dx.dtype)[batch_ligand].unsqueeze(-1)
                    projected = raw_dx * keep
                elif projection_mode in {
                    'remove_negative_parallel',
                    'tangent_remove_negative_parallel',
                    'molecular_constraint_remove_negative_parallel',
                }:
                    base_sq = graph_dot(base_dx, base_dx).clamp_min(1e-8)
                    coeff = (raw_dot_base / base_sq).clamp(max=0.0)
                    negative_parallel = coeff[batch_ligand].unsqueeze(-1) * base_dx
                    projected = raw_dx - negative_parallel
                elif projection_mode in {'none', 'tangent', 'molecular_constraint'}:
                    projected = raw_dx
                else:
                    raise ValueError(f'Unknown residual projection mode: {projection_mode}')
                if projection_mode in {'tangent', 'tangent_remove_negative_parallel'}:
                    projected = remove_inward_normal(projected)
                if projection_mode in {'molecular_constraint', 'molecular_constraint_remove_negative_parallel'}:
                    projected = remove_ligand_bond_stretch(projected)
                return projected, raw_dot_base

            def feasibility_geometry_gate(clash_threshold, overburied_threshold, temperature):
                gates = []
                temp = max(float(temperature), 1e-6)
                for graph_idx in range(num_graphs):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    if bool(lm.any()) and bool(pm.any()):
                        lig_pos_g = ligand_xt[lm]
                        prot_pos_g = protein_pos[pm]
                        d = torch.cdist(lig_pos_g, prot_pos_g).clamp_min(1e-4)
                        n_lig = max(int(lig_pos_g.size(0)), 1)
                        clash = torch.relu(ligand_xt.new_tensor(1.75) - d).pow(2).sum() / float(n_lig)
                        nearest = d.min(dim=1).values
                        overburied = torch.relu(ligand_xt.new_tensor(2.20) - nearest).pow(2).mean()
                        clash_gate = torch.sigmoid((clash - float(clash_threshold)) / temp)
                        over_gate = torch.sigmoid((overburied - float(overburied_threshold)) / temp)
                        gate = torch.maximum(clash_gate, over_gate)
                    else:
                        gate = ligand_xt.new_tensor(0.0)
                    gates.append(gate)
                return torch.stack(gates, dim=0).detach()

            def physical_ratio_gate(
                mode='none',
                strength=0.0,
                clash_weight=0.0,
                severe_weight=0.0,
                overburied_weight=0.0,
                drift_weight=0.0,
                clash_radius=1.75,
                severe_radius=1.45,
                overburied_radius=2.20,
                drift_radius=8.0,
            ):
                if str(mode) in {'none', ''} or float(strength) <= 0:
                    return None, {}
                if str(mode) != 'inverse':
                    raise ValueError(f'Unknown HJB physical ratio gate mode: {mode}')

                risk_values = []
                clash_values = []
                severe_values = []
                over_values = []
                drift_values = []
                for graph_idx in range(num_graphs):
                    lm = batch_ligand == graph_idx
                    pm = batch_protein == graph_idx
                    zero = ligand_xt.new_tensor(0.0)
                    if bool(lm.any()) and bool(pm.any()):
                        lig_pos_g = ligand_xt[lm]
                        prot_pos_g = protein_pos[pm]
                        d = torch.cdist(lig_pos_g, prot_pos_g).clamp_min(1e-4)
                        n_lig = max(int(lig_pos_g.size(0)), 1)
                        clash = torch.relu(ligand_xt.new_tensor(clash_radius) - d).pow(2).sum() / float(n_lig)
                        severe = torch.relu(ligand_xt.new_tensor(severe_radius) - d).pow(2).sum() / float(n_lig)
                        nearest = d.min(dim=1).values
                        overburied = torch.relu(ligand_xt.new_tensor(overburied_radius) - nearest).pow(2).mean()
                        center_dist = (lig_pos_g.mean(dim=0) - prot_pos_g.mean(dim=0)).norm()
                        drift = torch.relu(center_dist - ligand_xt.new_tensor(drift_radius)).pow(2)
                    else:
                        clash = severe = overburied = drift = zero
                    risk = (
                        float(clash_weight) * clash
                        + float(severe_weight) * severe
                        + float(overburied_weight) * overburied
                        + float(drift_weight) * drift
                    )
                    risk_values.append(risk)
                    clash_values.append(clash)
                    severe_values.append(severe)
                    over_values.append(overburied)
                    drift_values.append(drift)

                risk = torch.stack(risk_values, dim=0).detach().clamp_min(0.0)
                gate_scale = (1.0 / (1.0 + float(strength) * risk)).clamp(0.0, 1.0)
                stats = {
                    'hjb_phys_gate_scale': gate_scale.detach().cpu(),
                    'hjb_phys_gate_risk': risk.detach().cpu(),
                    'hjb_phys_gate_clash': torch.stack(clash_values, dim=0).detach().cpu(),
                    'hjb_phys_gate_severe': torch.stack(severe_values, dim=0).detach().cpu(),
                    'hjb_phys_gate_overburied': torch.stack(over_values, dim=0).detach().cpu(),
                    'hjb_phys_gate_drift': torch.stack(drift_values, dim=0).detach().cpu(),
                }
                return gate_scale, stats

            if local_affinity_guidance is not None and float(local_affinity_guidance.get('target_base_ratio', 0.0)) > 0:
                local_projection = local_affinity_guidance.get('projection_mode', 'none')
                local_ratio = float(local_affinity_guidance.get('target_base_ratio', 0.0))
                local_max_scale = float(local_affinity_guidance.get('max_scale', 100.0))
                local_t0 = float(local_affinity_guidance.get('t0', 0.50))
                local_sigmoid_k = float(local_affinity_guidance.get('sigmoid_k', 14.0))
                local_time_fraction = (t.float() / 1000.0).clamp(0.0, 1.0)
                local_reverse_progress = (1.0 - local_time_fraction).clamp(0.0, 1.0)
                if bool(local_affinity_guidance.get('constant_schedule', False)):
                    local_schedule = torch.ones_like(local_reverse_progress)
                else:
                    local_schedule = torch.sigmoid((local_reverse_progress - local_t0) * local_sigmoid_k)
                local_active_steps = int(local_affinity_guidance.get('active_steps', 0))
                if local_active_steps > 0 and guidance_step_index >= local_active_steps:
                    local_schedule = torch.zeros_like(local_schedule)
                local_raw_dx = torch.nan_to_num(
                    pos_guidance,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                local_raw_norm = graph_norm(local_raw_dx).clamp_min(1e-8)
                local_projected_dx, local_raw_dot_base = project_residual(local_raw_dx, local_projection)
                local_projected_norm = graph_norm(local_projected_dx).clamp_min(1e-8)
                local_base_norm = graph_norm(base_dx).clamp_min(1e-8)
                local_raw_scale = local_ratio * local_base_norm / local_projected_norm
                local_scale = local_raw_scale.clamp(max=local_max_scale)
                local_dx = (
                    local_projected_dx
                    * local_scale[batch_ligand].unsqueeze(-1)
                    * local_schedule[batch_ligand].unsqueeze(-1)
                )
                dx = dx + local_dx
                if trace_velocity:
                    local_norm = graph_norm(local_dx)
                    local_affinity_stats = {
                        'local_affinity_schedule': local_schedule.detach().cpu(),
                        'local_affinity_raw_norm': local_raw_norm.detach().cpu(),
                        'local_affinity_projected_norm': local_projected_norm.detach().cpu(),
                        'local_affinity_raw_scale': local_raw_scale.detach().cpu(),
                        'local_affinity_scale': local_scale.detach().cpu(),
                        'local_affinity_scale_clamped': (local_raw_scale > local_max_scale).detach().cpu().float(),
                        'local_affinity_norm': local_norm.detach().cpu(),
                        'local_affinity_to_base_ratio': local_norm.detach().cpu() / local_base_norm.detach().cpu().clamp_min(1e-8),
                        'local_affinity_effective_target_fraction': (
                            local_norm / (local_ratio * local_base_norm * local_schedule).clamp_min(1e-8)
                        ).detach().cpu(),
                        'local_affinity_raw_base_cos': (
                            local_raw_dot_base / (local_raw_norm * local_base_norm).clamp_min(1e-8)
                        ).detach().cpu(),
                    }
                    local_dot_base = graph_dot(local_dx, base_dx)
                    local_affinity_stats['local_affinity_base_cos'] = (
                        local_dot_base / (local_norm * local_base_norm).clamp_min(1e-8)
                    ).detach().cpu()

            if hjb_guidance is not None and float(hjb_guidance.get('target_base_ratio', 0.0)) > 0:
                value_model = hjb_guidance['model']
                value_component = hjb_guidance.get('component', 'total')
                projection_mode = hjb_guidance.get('projection_mode', 'remove_negative_parallel')
                target_base_ratio = float(hjb_guidance.get('target_base_ratio', 0.0))
                max_scale = float(hjb_guidance.get('max_scale', 100.0))
                feas_model = hjb_guidance.get('feas_model', None)
                feas_component = hjb_guidance.get('feas_component', 'total')
                feas_target_base_ratio = float(hjb_guidance.get('feas_target_base_ratio', 0.0))
                feas_max_scale = float(hjb_guidance.get('feas_max_scale', 100.0))
                feas_projection_mode = hjb_guidance.get('feas_projection_mode', 'remove_negative_parallel')
                feas_gate_mode = hjb_guidance.get('feas_gate_mode', 'none')
                feas_gate_threshold = float(hjb_guidance.get('feas_gate_threshold', 0.0))
                feas_gate_temperature = max(float(hjb_guidance.get('feas_gate_temperature', 0.50)), 1e-6)
                feas_gate_strength = float(hjb_guidance.get('feas_gate_strength', 0.0))
                feas_geom_clash_threshold = float(hjb_guidance.get('feas_geom_clash_threshold', 0.02))
                feas_geom_overburied_threshold = float(hjb_guidance.get('feas_geom_overburied_threshold', 0.03))
                feas_geom_temperature = float(hjb_guidance.get('feas_geom_temperature', 0.02))
                disable_normalization = bool(hjb_guidance.get('disable_normalization', False))
                raw_gradient_scale = float(hjb_guidance.get('raw_gradient_scale', 1.0))
                control_mode = str(hjb_guidance.get('control_mode', 'normalized') or 'normalized')
                control_cost_weight = float(hjb_guidance.get('control_cost_weight', 0.0))
                control_cap_ratio = float(hjb_guidance.get('control_cap_ratio', target_base_ratio))
                hjb_t0 = float(hjb_guidance.get('t0', 0.65))
                hjb_sigmoid_k = float(hjb_guidance.get('sigmoid_k', 14.0))
                hjb_late_taper_start = float(hjb_guidance.get('late_taper_start', 1.0))
                hjb_late_taper_k = float(hjb_guidance.get('late_taper_k', 30.0))
                mobility_mode = hjb_guidance.get('mobility_mode', 'none')
                mobility_strength = float(hjb_guidance.get('mobility_strength', 0.0))
                mobility_radius = float(hjb_guidance.get('mobility_radius', 2.40))
                mobility_softness = float(hjb_guidance.get('mobility_softness', 0.35))
                mobility_inward_only = bool(hjb_guidance.get('mobility_inward_only', True))
                barrier_filter = bool(hjb_guidance.get('barrier_filter', False))
                barrier_min_dist = float(hjb_guidance.get('barrier_min_dist', 1.60))
                barrier_active_dist = float(hjb_guidance.get('barrier_active_dist', 2.10))
                barrier_kappa = float(hjb_guidance.get('barrier_kappa', 0.50))
                barrier_strength = float(hjb_guidance.get('barrier_strength', 1.00))
                feasible_corridor_filter = bool(hjb_guidance.get('feasible_corridor_filter', False))
                fc_mode = hjb_guidance.get('fc_mode', 'clash')
                fc_eta = float(hjb_guidance.get('fc_eta', 0.25))
                fc_iters = int(hjb_guidance.get('fc_iters', 2))
                fc_clash_radius = float(hjb_guidance.get('fc_clash_radius', 1.75))
                fc_severe_radius = float(hjb_guidance.get('fc_severe_radius', 1.45))
                fc_overburied_radius = float(hjb_guidance.get('fc_overburied_radius', 2.20))
                fc_drift_radius = float(hjb_guidance.get('fc_drift_radius', 8.0))
                phys_ratio_gate_mode = str(hjb_guidance.get('phys_ratio_gate_mode', 'none') or 'none')
                phys_ratio_gate_strength = float(hjb_guidance.get('phys_ratio_gate_strength', 0.0))
                phys_ratio_gate_clash_weight = float(hjb_guidance.get('phys_ratio_gate_clash_weight', 0.0))
                phys_ratio_gate_severe_weight = float(hjb_guidance.get('phys_ratio_gate_severe_weight', 0.0))
                phys_ratio_gate_overburied_weight = float(hjb_guidance.get('phys_ratio_gate_overburied_weight', 0.0))
                phys_ratio_gate_drift_weight = float(hjb_guidance.get('phys_ratio_gate_drift_weight', 0.0))
                phys_ratio_gate_clash_radius = float(hjb_guidance.get('phys_ratio_gate_clash_radius', 1.75))
                phys_ratio_gate_severe_radius = float(hjb_guidance.get('phys_ratio_gate_severe_radius', 1.45))
                phys_ratio_gate_overburied_radius = float(hjb_guidance.get('phys_ratio_gate_overburied_radius', 2.20))
                phys_ratio_gate_drift_radius = float(hjb_guidance.get('phys_ratio_gate_drift_radius', 8.0))
                hjb_mobility_delta_norm = None
                hjb_barrier_delta_norm = None
                hjb_barrier_to_hjb_ratio = None
                hjb_fc_stats = {}
                hjb_phys_ratio_gate_stats = {}
                feas_score = None
                feas_gate = None
                grad_feas = None
                feas_raw_norm = None
                feas_projected_norm = None
                feas_raw_scale = None
                feas_scale = None
                feas_dx = None
                feas_raw_dot_base = None
                atom_model = hjb_guidance.get('atom_model')
                atom_component = hjb_guidance.get('atom_component', value_component)
                atom_target_base_ratio = float(hjb_guidance.get('atom_target_base_ratio', 0.0))
                atom_max_scale = float(hjb_guidance.get('atom_max_scale', 10.0))
                atom_score = None
                atom_raw_dv = None
                atom_schedule = None
                actor_model = hjb_guidance.get('actor_model')
                actor_raw_norm = None
                actor_neggrad_cos = None
                actor_base_cos = None
                multi_head_mode = str(hjb_guidance.get('multi_head_mode', 'none') or 'none')
                value_time_mode = str(hjb_guidance.get('value_time_mode', 'vp_time') or 'vp_time')
                if value_time_mode not in {'vp_time', 'generation_progress'}:
                    raise ValueError(f'Unknown HJB value-time mode: {value_time_mode}')
                multi_head_components = [
                    x.strip()
                    for x in str(hjb_guidance.get('multi_head_components', '') or '').split(',')
                    if x.strip()
                ]
                multi_head_weight_values = [
                    float(x.strip())
                    for x in str(hjb_guidance.get('multi_head_weights', '') or '').split(',')
                    if x.strip()
                ]
                if multi_head_components and not multi_head_weight_values:
                    multi_head_weight_values = [1.0 for _ in multi_head_components]
                if multi_head_components and len(multi_head_weight_values) < len(multi_head_components):
                    multi_head_weight_values = multi_head_weight_values + [
                        multi_head_weight_values[-1]
                        for _ in range(len(multi_head_components) - len(multi_head_weight_values))
                    ]

                def graph_normalize(vec):
                    vec_norm = graph_norm(vec).clamp_min(1e-8)
                    return vec / vec_norm[batch_ligand].unsqueeze(-1)

                def remove_pairwise_conflict(vec, other):
                    other_sq = graph_dot(other, other).clamp_min(1e-8)
                    coeff = (graph_dot(vec, other) / other_sq).clamp(max=0.0)
                    return vec - coeff[batch_ligand].unsqueeze(-1) * other

                with torch.enable_grad():
                    hjb_time_fraction = (t.float() / 1000.0).clamp(0.0, 1.0)
                    reverse_progress = (1.0 - hjb_time_fraction).clamp(0.0, 1.0)
                    value_time_fraction = (
                        reverse_progress
                        if value_time_mode == 'generation_progress'
                        else hjb_time_fraction
                    )
                    schedule, hjb_late_taper = compute_hjb_schedule(
                        reverse_progress,
                        t0=hjb_t0,
                        sigmoid_k=hjb_sigmoid_k,
                        late_taper_start=hjb_late_taper_start,
                        late_taper_k=hjb_late_taper_k,
                    )
                    atom_schedule = schedule
                    value_out = value_model(
                        ligand_pos=ligand_xt,
                        ligand_v=ligand_vt,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_ligand=batch_ligand,
                        batch_protein=batch_protein,
                        time_fraction=value_time_fraction,
                    )
                    value_score = select_hjb_value(
                        value_out,
                        component=value_component,
                        head_names=getattr(value_model, 'head_names', None),
                    )
                    if multi_head_mode != 'none':
                        if not multi_head_components:
                            multi_head_components = list(getattr(value_model, 'head_names', None) or [])
                        if not multi_head_components:
                            multi_head_components = [value_component]
                        if not multi_head_weight_values:
                            multi_head_weight_values = [1.0 for _ in multi_head_components]
                        if len(multi_head_weight_values) < len(multi_head_components):
                            multi_head_weight_values = multi_head_weight_values + [
                                multi_head_weight_values[-1]
                                for _ in range(len(multi_head_components) - len(multi_head_weight_values))
                            ]
                        head_dirs = []
                        for head_name, head_weight in zip(multi_head_components, multi_head_weight_values):
                            if float(head_weight) == 0.0:
                                continue
                            head_score = select_hjb_value(
                                value_out,
                                component=head_name,
                                head_names=getattr(value_model, 'head_names', None),
                            )
                            grad_head = torch.autograd.grad(
                                head_score.sum(),
                                ligand_xt,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                            if grad_head is None:
                                grad_head = torch.zeros_like(ligand_xt)
                            head_dirs.append(float(head_weight) * graph_normalize(-torch.nan_to_num(
                                grad_head,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )))
                        if head_dirs:
                            if multi_head_mode == 'pcgrad':
                                adjusted = []
                                for idx, vec in enumerate(head_dirs):
                                    out_vec = vec
                                    for jdx, other in enumerate(head_dirs):
                                        if idx == jdx:
                                            continue
                                        out_vec = remove_pairwise_conflict(out_vec, other.detach())
                                    adjusted.append(out_vec)
                                hjb_raw_dx_from_heads = torch.stack(adjusted, dim=0).sum(dim=0)
                            elif multi_head_mode == 'weighted':
                                hjb_raw_dx_from_heads = torch.stack(head_dirs, dim=0).sum(dim=0)
                            else:
                                raise ValueError(f'Unknown HJB multi-head mode: {multi_head_mode}')
                            grad_s = -hjb_raw_dx_from_heads
                        else:
                            grad_s = torch.zeros_like(ligand_xt)
                    else:
                        grad_s = torch.autograd.grad(
                            value_score.sum(),
                            ligand_xt,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                    if feas_model is not None and (
                        str(feas_gate_mode) != 'none'
                        or float(feas_target_base_ratio) > 0
                        or float(feas_gate_strength) > 0
                    ):
                        feas_out = feas_model(
                            ligand_pos=ligand_xt,
                            ligand_v=ligand_vt,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_ligand=batch_ligand,
                            batch_protein=batch_protein,
                            time_fraction=value_time_fraction,
                        )
                        feas_score = select_hjb_value(
                            feas_out,
                            component=feas_component,
                            head_names=getattr(feas_model, 'head_names', None),
                        )
                        if float(feas_target_base_ratio) > 0:
                            grad_feas = torch.autograd.grad(
                                feas_score.sum(),
                                ligand_xt,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                    if atom_model is not None and atom_target_base_ratio > 0:
                        if atom_model is value_model and str(atom_component) == str(value_component):
                            atom_score = value_score
                        else:
                            atom_out = atom_model(
                                ligand_pos=ligand_xt,
                                ligand_v=ligand_vt,
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_ligand=batch_ligand,
                                batch_protein=batch_protein,
                                time_fraction=value_time_fraction,
                            )
                            atom_score = select_hjb_value(
                                atom_out,
                                component=atom_component,
                                head_names=getattr(atom_model, 'head_names', None),
                            )
                        grad_atom_v = torch.autograd.grad(
                            atom_score.sum(),
                            ligand_vt,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad_atom_v is not None:
                            atom_raw_dv = -torch.nan_to_num(
                                grad_atom_v,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            # Logit shifts that add the same constant to every atom type
                            # do not affect the categorical distribution.
                            atom_raw_dv = atom_raw_dv - atom_raw_dv.mean(dim=-1, keepdim=True)
                if grad_s is None:
                    grad_s = torch.zeros_like(ligand_xt)
                hjb_raw_dx = -grad_s
                if actor_model is not None:
                    with torch.no_grad():
                        neg_grad_s = torch.nan_to_num(
                            hjb_raw_dx.detach(),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        actor_base_norm = graph_norm(base_dx.detach()).clamp_min(1e-8)
                        actor_neg_norm = graph_norm(neg_grad_s).clamp_min(1e-8)
                        actor_neggrad_cos = (
                            graph_dot(base_dx.detach(), neg_grad_s) / (actor_base_norm * actor_neg_norm).clamp_min(1e-8)
                        )
                        actor_ratio = actor_neg_norm / actor_base_norm
                        actor_scalars = torch.stack(
                            [
                                value_time_fraction.detach(),
                                value_score.detach(),
                                actor_neggrad_cos.detach(),
                                actor_ratio.detach(),
                            ],
                            dim=-1,
                        )
                        if bool(getattr(actor_model, 'uses_protein_context', False)):
                            actor_raw_dx = actor_model(
                                ligand_xt.detach(),
                                ligand_vt.detach(),
                                batch_ligand,
                                neg_grad_s,
                                base_dx.detach(),
                                actor_scalars,
                                protein_pos=protein_pos.detach(),
                                protein_v=protein_v.detach(),
                                batch_protein=batch_protein,
                            )
                        else:
                            actor_raw_dx = actor_model(
                                ligand_xt.detach(),
                                ligand_vt.detach(),
                                batch_ligand,
                                neg_grad_s,
                                base_dx.detach(),
                                actor_scalars,
                            )
                        actor_raw_dx = torch.nan_to_num(
                            actor_raw_dx,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        actor_raw_norm = graph_norm(actor_raw_dx).clamp_min(1e-8)
                        actor_base_cos = (
                            graph_dot(actor_raw_dx, base_dx.detach()) / (actor_raw_norm * actor_base_norm).clamp_min(1e-8)
                        )
                    hjb_raw_dx = actor_raw_dx
                if mobility_mode == 'protein_normal' and mobility_strength > 0:
                    hjb_raw_dx_before_mobility = hjb_raw_dx
                    hjb_raw_dx = apply_protein_normal_mobility(
                        hjb_raw_dx,
                        strength=mobility_strength,
                        radius=mobility_radius,
                        softness=mobility_softness,
                        inward_only=mobility_inward_only,
                    )
                    hjb_mobility_delta_norm = graph_norm(hjb_raw_dx - hjb_raw_dx_before_mobility)
                elif mobility_mode not in {'none', None}:
                    raise ValueError(f'Unknown HJB mobility mode: {mobility_mode}')

                base_norm = graph_norm(base_dx).clamp_min(1e-8)
                raw_norm = graph_norm(hjb_raw_dx).clamp_min(1e-8)

                if hjb_gate_enabled:
                    d_projection = hjb_guidance.get('gate_d_projection', 'remove_negative_parallel')
                    e_projection = hjb_guidance.get('gate_e_projection', 'tangent_remove_negative_parallel')
                    d_ratio = float(hjb_guidance.get('gate_d_ratio', target_base_ratio))
                    e_ratio = float(hjb_guidance.get('gate_e_ratio', target_base_ratio))
                    gate_warmup_steps = int(hjb_guidance.get('gate_warmup_steps', 15))
                    gate_threshold = float(hjb_guidance.get('gate_threshold', 0.22946724712673572))
                    step_index = int(len(pos_traj))

                    d_projected_dx, raw_dot_base = project_residual(hjb_raw_dx, d_projection)
                    e_projected_dx, _ = project_residual(hjb_raw_dx, e_projection)
                    d_projected_norm = graph_norm(d_projected_dx).clamp_min(1e-8)
                    e_projected_norm = graph_norm(e_projected_dx).clamp_min(1e-8)
                    d_raw_scale = d_ratio * base_norm / d_projected_norm
                    e_raw_scale = e_ratio * base_norm / e_projected_norm
                    d_scale = d_raw_scale.clamp(max=max_scale)
                    e_scale = e_raw_scale.clamp(max=max_scale)
                    d_dx = d_projected_dx * d_scale[batch_ligand].unsqueeze(-1) * schedule[batch_ligand].unsqueeze(-1)
                    e_dx = e_projected_dx * e_scale[batch_ligand].unsqueeze(-1) * schedule[batch_ligand].unsqueeze(-1)
                    d_norm = graph_norm(d_dx).clamp_min(1e-8)
                    e_norm = graph_norm(e_dx).clamp_min(1e-8)
                    d_base_cos = graph_dot(d_dx, base_dx) / (d_norm * base_norm).clamp_min(1e-8)
                    e_base_cos = graph_dot(e_dx, base_dx) / (e_norm * base_norm).clamp_min(1e-8)

                    if step_index < gate_warmup_steps:
                        hjb_gate_cos_sum = hjb_gate_cos_sum + d_base_cos.detach()
                        hjb_gate_cos_count = hjb_gate_cos_count + torch.ones_like(hjb_gate_cos_count)
                        hjb_gate_active_choice = torch.zeros_like(hjb_gate_choice)
                    else:
                        unset = hjb_gate_choice < 0
                        if bool(unset.any()):
                            mean_cos = hjb_gate_cos_sum / hjb_gate_cos_count.clamp_min(1.0)
                            hjb_gate_choice = hjb_gate_choice.clone()
                            hjb_gate_choice[unset] = torch.where(
                                mean_cos[unset] <= gate_threshold,
                                torch.ones_like(hjb_gate_choice[unset]),
                                torch.zeros_like(hjb_gate_choice[unset]),
                            )
                        hjb_gate_active_choice = hjb_gate_choice

                    choose_e = hjb_gate_active_choice == 1
                    hjb_dx = torch.where(choose_e[batch_ligand].unsqueeze(-1), e_dx, d_dx)
                    hjb_projected_dx = torch.where(
                        choose_e[batch_ligand].unsqueeze(-1),
                        e_projected_dx,
                        d_projected_dx,
                    )
                    projected_norm = torch.where(choose_e, e_projected_norm, d_projected_norm)
                    raw_scale = torch.where(choose_e, e_raw_scale, d_raw_scale)
                    scale = torch.where(choose_e, e_scale, d_scale)
                    target_base_ratio_graph = torch.where(
                        choose_e,
                        torch.full_like(base_norm, e_ratio),
                        torch.full_like(base_norm, d_ratio),
                    )
                else:
                    hjb_projected_dx, raw_dot_base = project_residual(hjb_raw_dx, projection_mode)
                    projected_norm = graph_norm(hjb_projected_dx).clamp_min(1e-8)
                    if control_mode == 'control_cost':
                        if control_cost_weight <= 0:
                            raise ValueError('HJB control_cost mode requires --hjb_control_cost_weight > 0.')
                        raw_scale = torch.full_like(base_norm, 1.0 / (2.0 * control_cost_weight))
                        cap_scale = control_cap_ratio * base_norm / projected_norm
                        scale = torch.minimum(raw_scale, cap_scale)
                        target_base_ratio_graph = scale * projected_norm / base_norm
                    elif control_mode != 'normalized':
                        raise ValueError(f'Unknown HJB control mode: {control_mode}')
                    elif disable_normalization:
                        raw_scale = torch.full_like(base_norm, raw_gradient_scale)
                        scale = raw_scale
                        target_base_ratio_graph = raw_gradient_scale * projected_norm / base_norm
                    else:
                        target_base_ratio_graph = torch.full_like(base_norm, target_base_ratio)
                        score_ratio_mode = str(hjb_guidance.get('score_ratio_mode', 'none'))
                        if score_ratio_mode != 'none':
                            if score_ratio_mode != 'sigmoid':
                                raise ValueError(f'Unknown HJB score-ratio mode: {score_ratio_mode}')
                            ratio_min = float(hjb_guidance.get('score_ratio_min', target_base_ratio))
                            ratio_max = float(hjb_guidance.get('score_ratio_max', target_base_ratio))
                            ratio_center = float(hjb_guidance.get('score_ratio_center', 0.0))
                            ratio_temperature = max(float(hjb_guidance.get('score_ratio_temperature', 0.5)), 1e-6)
                            ratio_gate = torch.sigmoid((value_score.detach() - ratio_center) / ratio_temperature)
                            target_base_ratio_graph = ratio_min + (ratio_max - ratio_min) * ratio_gate
                        raw_scale = target_base_ratio_graph * base_norm / projected_norm
                        scale = raw_scale.clamp(max=max_scale)
                    hjb_dx = hjb_projected_dx * scale[batch_ligand].unsqueeze(-1) * schedule[batch_ligand].unsqueeze(-1)

                if feas_score is not None and str(feas_gate_mode) != 'none':
                    feas_gate = torch.sigmoid((feas_score.detach() - feas_gate_threshold) / feas_gate_temperature)
                    if str(feas_gate_mode) == 'soft_geom':
                        geom_gate = feasibility_geometry_gate(
                            clash_threshold=feas_geom_clash_threshold,
                            overburied_threshold=feas_geom_overburied_threshold,
                            temperature=feas_geom_temperature,
                        )
                        feas_gate = torch.maximum(feas_gate, geom_gate)
                    elif str(feas_gate_mode) != 'soft':
                        raise ValueError(f'Unknown feasibility gate mode: {feas_gate_mode}')
                    feas_gate = feas_gate.clamp(0.0, 1.0)
                    if float(feas_gate_strength) > 0:
                        gate_scale = (1.0 - float(feas_gate_strength) * feas_gate).clamp(0.0, 1.0)
                        hjb_dx = hjb_dx * gate_scale[batch_ligand].unsqueeze(-1)
                        target_base_ratio_graph = target_base_ratio_graph * gate_scale

                if feas_model is not None and float(feas_target_base_ratio) > 0:
                    if grad_feas is None:
                        grad_feas = torch.zeros_like(ligand_xt)
                    feas_raw_dx = -grad_feas
                    feas_raw_norm = graph_norm(feas_raw_dx).clamp_min(1e-8)
                    feas_projected_dx, feas_raw_dot_base = project_residual(feas_raw_dx, feas_projection_mode)
                    feas_projected_norm = graph_norm(feas_projected_dx).clamp_min(1e-8)
                    feas_raw_scale = feas_target_base_ratio * base_norm / feas_projected_norm
                    feas_scale = feas_raw_scale.clamp(max=feas_max_scale)
                    if feas_gate is None:
                        feas_gate = torch.ones_like(base_norm)
                    feas_dx = (
                        feas_projected_dx
                        * feas_scale[batch_ligand].unsqueeze(-1)
                        * schedule[batch_ligand].unsqueeze(-1)
                        * feas_gate[batch_ligand].unsqueeze(-1)
                    )
                    hjb_dx = hjb_dx + feas_dx
                    target_base_ratio_graph = target_base_ratio_graph + feas_target_base_ratio * feas_gate

                phys_gate_scale, hjb_phys_ratio_gate_stats = physical_ratio_gate(
                    mode=phys_ratio_gate_mode,
                    strength=phys_ratio_gate_strength,
                    clash_weight=phys_ratio_gate_clash_weight,
                    severe_weight=phys_ratio_gate_severe_weight,
                    overburied_weight=phys_ratio_gate_overburied_weight,
                    drift_weight=phys_ratio_gate_drift_weight,
                    clash_radius=phys_ratio_gate_clash_radius,
                    severe_radius=phys_ratio_gate_severe_radius,
                    overburied_radius=phys_ratio_gate_overburied_radius,
                    drift_radius=phys_ratio_gate_drift_radius,
                )
                if phys_gate_scale is not None:
                    hjb_dx = hjb_dx * phys_gate_scale[batch_ligand].unsqueeze(-1)
                    target_base_ratio_graph = target_base_ratio_graph * phys_gate_scale

                if feasible_corridor_filter:
                    hjb_dx, hjb_fc_stats = apply_feasible_corridor_filter(
                        hjb_dx,
                        base_dx,
                        target_base_ratio_graph,
                        schedule,
                        mode=fc_mode,
                        eta=fc_eta,
                        iters=fc_iters,
                        clash_radius=fc_clash_radius,
                        severe_radius=fc_severe_radius,
                        overburied_radius=fc_overburied_radius,
                        drift_radius=fc_drift_radius,
                    )
                if barrier_filter:
                    hjb_dx_before_barrier = hjb_dx
                    hjb_dx = apply_contact_barrier_filter(
                        hjb_dx,
                        base_dx,
                        min_dist=barrier_min_dist,
                        active_dist=barrier_active_dist,
                        kappa=barrier_kappa,
                        strength=barrier_strength,
                    )
                    hjb_barrier_delta_norm = graph_norm(hjb_dx - hjb_dx_before_barrier)
                    hjb_barrier_to_hjb_ratio = hjb_barrier_delta_norm / graph_norm(hjb_dx_before_barrier).clamp_min(1e-8)
                dx = base_dx + hjb_dx
                if trace_velocity:
                    hjb_norm = graph_norm(hjb_dx)
                    hjb_stats = {
                        'hjb_score': value_score.detach().cpu(),
                        'hjb_schedule': schedule.detach().cpu(),
                        'hjb_late_taper': hjb_late_taper.detach().cpu(),
                        'base_norm': base_norm.detach().cpu(),
                        'hjb_raw_norm': raw_norm.detach().cpu(),
                        'hjb_projected_norm': projected_norm.detach().cpu(),
                        'hjb_projected_to_raw_ratio': (projected_norm / raw_norm.clamp_min(1e-8)).detach().cpu(),
                        'hjb_raw_scale': raw_scale.detach().cpu(),
                        'hjb_scale': scale.detach().cpu(),
                        'hjb_scale_clamped': (raw_scale > scale).detach().cpu().float(),
                        'hjb_target_base_ratio': target_base_ratio_graph.detach().cpu(),
                        'hjb_norm': hjb_norm.detach().cpu(),
                        'hjb_to_base_ratio': hjb_norm.detach().cpu() / base_norm.detach().cpu().clamp_min(1e-8),
                        'hjb_effective_target_fraction': (
                            hjb_norm / (target_base_ratio_graph * base_norm * schedule).clamp_min(1e-8)
                        ).detach().cpu(),
                        'hjb_raw_base_cos': (
                            raw_dot_base / (raw_norm * base_norm).clamp_min(1e-8)
                        ).detach().cpu(),
                    }
                    if actor_raw_norm is not None:
                        hjb_stats['hjb_actor_raw_norm'] = actor_raw_norm.detach().cpu()
                        hjb_stats['hjb_actor_neggrad_cos'] = actor_neggrad_cos.detach().cpu()
                        hjb_stats['hjb_actor_base_cos'] = actor_base_cos.detach().cpu()
                    if control_mode == 'control_cost':
                        hjb_stats['hjb_control_cost_weight'] = torch.full_like(base_norm, control_cost_weight).detach().cpu()
                        hjb_stats['hjb_control_cap_ratio'] = torch.full_like(base_norm, control_cap_ratio).detach().cpu()
                        hjb_stats['hjb_control_uncapped_to_base_ratio'] = (
                            raw_scale * projected_norm / base_norm
                        ).detach().cpu()
                        hjb_stats['hjb_control_cap_active'] = (raw_scale > scale).detach().cpu().float()
                    hjb_stats.update(hjb_phys_ratio_gate_stats)
                    if feas_score is not None:
                        hjb_stats['hjb_feas_score'] = feas_score.detach().cpu()
                    if feas_gate is not None:
                        hjb_stats['hjb_feas_gate'] = feas_gate.detach().cpu()
                    if feas_dx is not None:
                        feas_norm = graph_norm(feas_dx)
                        hjb_stats.update({
                            'hjb_feas_raw_norm': feas_raw_norm.detach().cpu(),
                            'hjb_feas_projected_norm': feas_projected_norm.detach().cpu(),
                            'hjb_feas_raw_scale': feas_raw_scale.detach().cpu(),
                            'hjb_feas_scale': feas_scale.detach().cpu(),
                            'hjb_feas_scale_clamped': (feas_raw_scale > feas_max_scale).detach().cpu().float(),
                            'hjb_feas_norm': feas_norm.detach().cpu(),
                            'hjb_feas_to_base_ratio': feas_norm.detach().cpu() / base_norm.detach().cpu().clamp_min(1e-8),
                        })
                        if feas_raw_dot_base is not None:
                            hjb_stats['hjb_feas_raw_base_cos'] = (
                                feas_raw_dot_base / (feas_raw_norm * base_norm).clamp_min(1e-8)
                            ).detach().cpu()
                    if hjb_mobility_delta_norm is not None:
                        hjb_stats['hjb_mobility_delta_norm'] = hjb_mobility_delta_norm.detach().cpu()
                        hjb_stats['hjb_mobility_to_raw_ratio'] = (
                            hjb_mobility_delta_norm / raw_norm.clamp_min(1e-8)
                        ).detach().cpu()
                    if hjb_barrier_delta_norm is not None:
                        hjb_stats['hjb_barrier_delta_norm'] = hjb_barrier_delta_norm.detach().cpu()
                        hjb_stats['hjb_barrier_to_hjb_ratio'] = hjb_barrier_to_hjb_ratio.detach().cpu()
                    hjb_stats.update(hjb_fc_stats)
                    hjb_dot_base = graph_dot(hjb_dx, base_dx)
                    hjb_stats['hjb_base_cos'] = (
                        hjb_dot_base / (hjb_norm * base_norm).clamp_min(1e-8)
                    ).detach().cpu()
                    if hjb_gate_enabled:
                        gate_mean_cos = hjb_gate_cos_sum / hjb_gate_cos_count.clamp_min(1.0)
                        hjb_stats.update({
                            'hjb_gate_choice': hjb_gate_active_choice.detach().cpu().float(),
                            'hjb_gate_locked': (hjb_gate_choice >= 0).detach().cpu().float(),
                            'hjb_gate_mean_d_hjb_base_cos': gate_mean_cos.detach().cpu(),
                            'hjb_gate_d_hjb_base_cos': d_base_cos.detach().cpu(),
                            'hjb_gate_e_hjb_base_cos': e_base_cos.detach().cpu(),
                            'hjb_gate_d_to_base_ratio': d_norm.detach().cpu() / base_norm.detach().cpu().clamp_min(1e-8),
                            'hjb_gate_e_to_base_ratio': e_norm.detach().cpu() / base_norm.detach().cpu().clamp_min(1e-8),
                        })
            if physics_guidance is not None and float(physics_guidance.get('target_base_ratio', 0.0)) > 0:
                physics_projection = physics_guidance.get('projection_mode', 'remove_negative_parallel')
                physics_ratio = float(physics_guidance.get('target_base_ratio', 0.0))
                physics_max_scale = float(physics_guidance.get('max_scale', 100.0))
                physics_t0 = float(physics_guidance.get('t0', 0.50))
                physics_sigmoid_k = float(physics_guidance.get('sigmoid_k', 14.0))
                clash_radius = float(physics_guidance.get('clash_radius', 1.75))
                severe_radius = float(physics_guidance.get('severe_radius', 1.45))
                overburied_radius = float(physics_guidance.get('overburied_radius', 2.20))
                contact_dist = float(physics_guidance.get('contact_dist', 3.50))
                contact_sigma = max(float(physics_guidance.get('contact_sigma', 0.75)), 1e-6)
                steric_weight = float(physics_guidance.get('steric_weight', 1.0))
                severe_weight = float(physics_guidance.get('severe_weight', 2.0))
                overburied_weight = float(physics_guidance.get('overburied_weight', 0.5))
                contact_weight = float(physics_guidance.get('contact_weight', 0.25))
                with torch.enable_grad():
                    physics_time_fraction = (t.float() / 1000.0).clamp(0.0, 1.0)
                    physics_progress = (1.0 - physics_time_fraction).clamp(0.0, 1.0)
                    if bool(physics_guidance.get('constant_schedule', False)):
                        physics_schedule = torch.ones_like(physics_progress)
                    else:
                        physics_schedule = torch.sigmoid((physics_progress - physics_t0) * physics_sigmoid_k)
                    physics_active_steps = int(physics_guidance.get('active_steps', 0))
                    if physics_active_steps > 0 and guidance_step_index >= physics_active_steps:
                        physics_schedule = torch.zeros_like(physics_schedule)
                    physics_costs = []
                    num_graphs_physics = int(max(batch_ligand.max().item(), batch_protein.max().item()) + 1)
                    for graph_idx in range(num_graphs_physics):
                        lm = batch_ligand == graph_idx
                        pm = batch_protein == graph_idx
                        if bool(lm.any()) and bool(pm.any()):
                            lig_pos_g = ligand_xt[lm]
                            prot_pos_g = protein_pos[pm]
                            d = torch.cdist(lig_pos_g, prot_pos_g).clamp_min(1e-4)
                            nearest_lig = d.min(dim=1).values
                            steric = torch.relu(ligand_xt.new_tensor(clash_radius) - d).pow(2).mean()
                            severe = torch.relu(ligand_xt.new_tensor(severe_radius) - d).pow(2).mean()
                            overburied = torch.relu(ligand_xt.new_tensor(overburied_radius) - nearest_lig).pow(2).mean()
                            contact = torch.exp(-((d - contact_dist) / contact_sigma).pow(2)).mean()
                            cost = (
                                steric_weight * steric
                                + severe_weight * severe
                                + overburied_weight * overburied
                                - contact_weight * contact
                            )
                        else:
                            cost = ligand_xt.new_tensor(0.0)
                        physics_costs.append(cost)
                    physics_score = torch.stack(physics_costs, dim=0)
                    grad_physics = torch.autograd.grad(
                        physics_score.sum(),
                        ligand_xt,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                if grad_physics is None:
                    grad_physics = torch.zeros_like(ligand_xt)
                physics_raw_dx = -grad_physics
                base_norm = graph_norm(base_dx).clamp_min(1e-8)
                physics_raw_norm = graph_norm(physics_raw_dx).clamp_min(1e-8)
                physics_projected_dx, physics_raw_dot_base = project_residual(physics_raw_dx, physics_projection)
                physics_projected_norm = graph_norm(physics_projected_dx).clamp_min(1e-8)
                if bool(physics_guidance.get('adaptive_de_gate', False)) and hjb_gate_active_choice is not None:
                    d_physics_ratio = float(physics_guidance.get('gate_d_ratio', physics_ratio))
                    e_physics_ratio = float(physics_guidance.get('gate_e_ratio', physics_ratio))
                    choose_e = hjb_gate_active_choice == 1
                    physics_ratio_graph = torch.where(
                        choose_e,
                        torch.full_like(base_norm, e_physics_ratio),
                        torch.full_like(base_norm, d_physics_ratio),
                    )
                else:
                    physics_ratio_graph = torch.full_like(base_norm, physics_ratio)
                physics_raw_scale = physics_ratio_graph * base_norm / physics_projected_norm
                physics_scale = physics_raw_scale.clamp(max=physics_max_scale)
                physics_dx = physics_projected_dx * physics_scale[batch_ligand].unsqueeze(-1) * physics_schedule[batch_ligand].unsqueeze(-1)
                dx = dx + physics_dx
                if trace_velocity:
                    physics_norm = graph_norm(physics_dx)
                    physics_stats = {
                        'physics_score': physics_score.detach().cpu(),
                        'physics_schedule': physics_schedule.detach().cpu(),
                        'physics_raw_norm': physics_raw_norm.detach().cpu(),
                        'physics_projected_norm': physics_projected_norm.detach().cpu(),
                        'physics_projected_to_raw_ratio': (
                            physics_projected_norm / physics_raw_norm.clamp_min(1e-8)
                        ).detach().cpu(),
                        'physics_raw_scale': physics_raw_scale.detach().cpu(),
                        'physics_scale': physics_scale.detach().cpu(),
                        'physics_scale_clamped': (physics_raw_scale > physics_max_scale).detach().cpu().float(),
                        'physics_norm': physics_norm.detach().cpu(),
                        'physics_to_base_ratio': physics_norm.detach().cpu() / base_norm.detach().cpu().clamp_min(1e-8),
                        'physics_effective_target_fraction': (
                            physics_norm / (physics_ratio_graph * base_norm * physics_schedule).clamp_min(1e-8)
                        ).detach().cpu(),
                        'physics_target_base_ratio': physics_ratio_graph.detach().cpu(),
                        'physics_raw_base_cos': (
                            physics_raw_dot_base / (physics_raw_norm * base_norm).clamp_min(1e-8)
                        ).detach().cpu(),
                    }
                    physics_dot_base = graph_dot(physics_dx, base_dx)
                    physics_stats['physics_base_cos'] = (
                        physics_dot_base / (physics_norm * base_norm).clamp_min(1e-8)
                    ).detach().cpu()
                    if hjb_dx is not None:
                        physics_hjb_dot = graph_dot(physics_dx, hjb_dx)
                        physics_stats['physics_hjb_cos'] = (
                            physics_hjb_dot / (physics_norm * graph_norm(hjb_dx)).clamp_min(1e-8)
                        ).detach().cpu()
            if branch_response_guidance is not None and float(branch_response_guidance.get('target_base_ratio', 0.0)) > 0:
                from models.intended_action_response import (
                    intended_action_features,
                    orthonormalize_fields,
                    predict_ensemble,
                )
                from models.calibrated_transverse_branch import (
                    COEFFICIENTS as transverse_coefficient_bank,
                    predict_probability_ensemble,
                    state_transverse_orthonormal_basis,
                    transverse_candidates,
                    transverse_features,
                )
                response_checkpoint = branch_response_guidance['checkpoint']
                response_calibrated_transverse = (
                    response_checkpoint.get('experiment') == 'calibrated_transverse_branch_v1'
                )
                response_ratio = float(branch_response_guidance.get('target_base_ratio', 0.10))
                response_t0 = float(branch_response_guidance.get('t0', 0.50))
                response_k = float(branch_response_guidance.get('uncertainty_k', 0.50))
                response_min_improvement = float(branch_response_guidance.get('min_improvement', 0.05))
                response_max_clash = float(branch_response_guidance.get('max_clash_increase', 0.25))
                response_affinity_only = bool(branch_response_guidance.get('affinity_only', False))
                response_contact_barrier = bool(branch_response_guidance.get('contact_barrier', False))
                response_bond_projection = bool(branch_response_guidance.get('bond_projection', False))
                response_progress = (1.0 - (t.float() / 1000.0)).clamp(0.0, 1.0)
                response_active_steps = branch_response_guidance.get('active_steps') or response_checkpoint.get(
                    'active_step_indices', []
                )
                response_step_active = (
                    not response_calibrated_transverse
                    or not response_active_steps
                    or int(guidance_step_index) in {int(value) for value in response_active_steps}
                )
                if response_calibrated_transverse:
                    response_schedule = torch.full_like(
                        response_progress, 1.0 if response_step_active else 0.0
                    )
                else:
                    response_schedule = torch.sigmoid((response_progress - response_t0) * 14.0)
                response_raw = torch.zeros_like(ligand_xt)
                response_gate = torch.zeros(num_graphs, dtype=ligand_xt.dtype, device=ligand_xt.device)
                response_affinity = torch.zeros(num_graphs, dtype=ligand_xt.dtype, device=ligand_xt.device)
                response_clash = torch.zeros_like(response_affinity)
                response_affinity_std = torch.zeros_like(response_affinity)
                response_clash_std = torch.zeros_like(response_affinity)
                response_coefficients = torch.zeros(num_graphs, 3, dtype=ligand_xt.dtype, device=ligand_xt.device)
                with torch.enable_grad():
                    response_steric_cost = []
                    for graph_idx in range(num_graphs):
                        lm = batch_ligand == graph_idx; pm = batch_protein == graph_idx
                        if bool(lm.any()) and bool(pm.any()):
                            distances = torch.cdist(ligand_xt[lm], protein_pos[pm]).clamp_min(1e-4)
                            cost = (
                                torch.relu(ligand_xt.new_tensor(2.0) - distances).pow(2).sum()
                                + 2.0 * torch.relu(ligand_xt.new_tensor(1.55) - distances).pow(2).sum()
                            ) / max(int(lm.sum()), 1)
                            if response_calibrated_transverse:
                                nearest = distances.min(dim=1).values
                                cost = cost + 0.5 * torch.relu(
                                    ligand_xt.new_tensor(2.4) - nearest
                                ).pow(2).mean()
                        else:
                            cost = ligand_xt.sum() * 0.0
                        response_steric_cost.append(cost)
                    response_steric_grad = torch.autograd.grad(
                        torch.stack(response_steric_cost).sum(), ligand_xt,
                        retain_graph=True, allow_unused=True,
                    )[0]
                if response_steric_grad is None:
                    response_steric_grad = torch.zeros_like(ligand_xt)
                for graph_idx in range(num_graphs):
                    lm = batch_ligand == graph_idx; pm = batch_protein == graph_idx
                    if not (bool(lm.any()) and bool(pm.any())):
                        continue
                    if response_calibrated_transverse:
                        if not response_step_active:
                            continue
                        basis, _, _ = state_transverse_orthonormal_basis(
                            pos_guidance[lm].detach(),
                            (-response_steric_grad[lm]).detach(),
                            base_dx[lm].detach(),
                            ligand_xt[lm].detach(),
                            protein_pos[pm].detach(),
                        )
                        candidates = transverse_candidates(basis)
                        if candidates.numel() == 0:
                            continue
                        oriented_actions = []
                        oriented_coefficients = []
                        for action, coefficients_pair in zip(candidates, transverse_coefficient_bank):
                            for orientation in (-1.0, 1.0):
                                oriented_actions.append(float(orientation) * action)
                                oriented_coefficients.append([
                                    float(orientation) * float(coefficients_pair[0]),
                                    float(orientation) * float(coefficients_pair[1]),
                                ])
                        probabilities, probability_stds, max_abs_z_values = [], [], []
                        for action in oriented_actions:
                            features = transverse_features(
                                ligand_xt[lm].detach(), protein_pos[pm].detach(), action, basis,
                                float(guidance_step_index) / 49.0, 2.0,
                            )
                            probability, probability_std, max_abs_z = predict_probability_ensemble(
                                response_checkpoint, features
                            )
                            probabilities.append(probability)
                            probability_stds.append(probability_std)
                            max_abs_z_values.append(max_abs_z)
                        probabilities = torch.stack(probabilities)
                        probability_stds = torch.stack(probability_stds)
                        max_abs_z_values = torch.stack(max_abs_z_values)
                        minimum_probability = float(branch_response_guidance.get(
                            'min_probability', response_checkpoint.get('minimum_probability', 0.60)
                        ))
                        maximum_probability_std = float(branch_response_guidance.get(
                            'max_probability_std', response_checkpoint.get('maximum_probability_std', 0.15)
                        ))
                        maximum_abs_z = float(branch_response_guidance.get(
                            'max_abs_z', response_checkpoint.get('maximum_abs_z', 5.0)
                        ))
                        eligible = (
                            (probabilities >= minimum_probability)
                            & (probability_stds <= maximum_probability_std)
                            & (max_abs_z_values <= maximum_abs_z)
                        )
                        if bool(eligible.any()):
                            threshold = ligand_xt.new_tensor(minimum_probability).clamp(1e-5, 1.0 - 1e-5)
                            threshold_logit = torch.logit(threshold)
                            scores = torch.logit(probabilities.clamp(1e-5, 1.0 - 1e-5)) - threshold_logit
                            scores = scores.masked_fill(~eligible, float('-inf'))
                            mixture_temperature = float(branch_response_guidance.get(
                                'softmax_temperature', response_checkpoint.get('softmax_temperature', 0.15)
                            ))
                            weights = torch.softmax(scores / max(mixture_temperature, 1e-6), dim=0)
                            action_tensor = torch.stack(oriented_actions)
                            coefficient_tensor = ligand_xt.new_tensor(oriented_coefficients)
                            response_raw[lm] = torch.einsum('m,mnd->nd', weights, action_tensor)
                            selected_coefficients = torch.einsum('m,md->d', weights, coefficient_tensor)
                            response_coefficients[graph_idx, :2] = selected_coefficients
                            response_gate[graph_idx] = 1.0
                            response_affinity[graph_idx] = torch.sum(weights * probabilities)
                            response_affinity_std[graph_idx] = torch.sum(weights * probability_stds)
                        continue
                    basis = orthonormalize_fields([
                        pos_guidance[lm].detach(),
                        (-response_steric_grad[lm]).detach(),
                        base_dx[lm].detach(),
                    ])
                    if basis.numel() == 0:
                        continue
                    affinity_means, affinity_stds, clash_means, clash_stds = [], [], [], []
                    for axis in basis:
                        features = intended_action_features(
                            ligand_xt[lm].detach(), protein_pos[pm].detach(), axis, basis,
                            response_progress[graph_idx].detach(), 1.0,
                        )
                        mean, std = predict_ensemble(response_checkpoint, 'gnina_affinity', features)
                        affinity_means.append(mean); affinity_stds.append(std)
                        if not response_affinity_only:
                            mean, std = predict_ensemble(response_checkpoint, 'posecheck_clashes', features)
                            clash_means.append(mean); clash_stds.append(std)
                    affinity_gradient = torch.stack(affinity_means)
                    affinity_sigma = torch.stack(affinity_stds)
                    coefficients = -affinity_gradient / affinity_gradient.norm().clamp_min(1e-8)
                    if response_affinity_only:
                        clash_gradient = torch.zeros_like(affinity_gradient)
                        clash_sigma = torch.zeros_like(affinity_sigma)
                    else:
                        clash_gradient = torch.stack(clash_means)
                        clash_sigma = torch.stack(clash_stds)
                        unsafe = torch.dot(clash_gradient, coefficients)
                        if float(unsafe.detach().cpu()) > 0.0:
                            coefficients = coefficients - unsafe * clash_gradient / clash_gradient.square().sum().clamp_min(1e-8)
                    coefficients = coefficients / coefficients.norm().clamp_min(1.0)
                    predicted_affinity = torch.dot(affinity_gradient, coefficients)
                    predicted_clash = torch.dot(clash_gradient, coefficients)
                    affinity_uncertainty = torch.sqrt(torch.sum((affinity_sigma * coefficients).square()))
                    clash_uncertainty = torch.sqrt(torch.sum((clash_sigma * coefficients).square()))
                    confident = predicted_affinity + response_k * affinity_uncertainty <= -response_min_improvement
                    if not response_affinity_only:
                        confident = confident and predicted_clash + response_k * clash_uncertainty <= response_max_clash
                    if bool(confident):
                        response_raw[lm] = torch.einsum('m,mnd->nd', coefficients, basis)
                        response_gate[graph_idx] = 1.0
                    response_coefficients[graph_idx] = coefficients
                    response_affinity[graph_idx] = predicted_affinity
                    response_clash[graph_idx] = predicted_clash
                    response_affinity_std[graph_idx] = affinity_uncertainty
                    response_clash_std[graph_idx] = clash_uncertainty
                response_base_norm = graph_norm(base_dx).clamp_min(1e-8)
                response_raw_norm = graph_norm(response_raw).clamp_min(1e-8)
                response_scale = response_ratio * response_base_norm / response_raw_norm
                response_dx = response_raw * response_scale[batch_ligand].unsqueeze(-1)
                response_dx = response_dx * response_schedule[batch_ligand].unsqueeze(-1)
                if response_bond_projection:
                    response_dx = remove_ligand_bond_stretch(response_dx)
                if bool(branch_response_guidance.get('pure_safety_projection', False)):
                    response_dx = remove_ligand_bond_stretch(response_dx)
                    response_dx = apply_protein_normal_mobility(
                        response_dx, strength=1.0, radius=2.10, softness=0.25, inward_only=True,
                    )
                if response_contact_barrier:
                    response_dx = apply_contact_barrier_filter(
                        response_dx, dx,
                        min_dist=float(branch_response_guidance.get('barrier_min_dist', 1.60)),
                        active_dist=float(branch_response_guidance.get('barrier_active_dist', 2.10)),
                        kappa=float(branch_response_guidance.get('barrier_kappa', 0.50)),
                        strength=float(branch_response_guidance.get('barrier_strength', 1.00)),
                    )
                dx = dx + response_dx
                if trace_velocity:
                    branch_response_stats = {
                        'branch_response_gate': response_gate.detach().cpu(),
                        'branch_response_schedule': response_schedule.detach().cpu(),
                        'branch_response_affinity': response_affinity.detach().cpu(),
                        'branch_response_clash': response_clash.detach().cpu(),
                        'branch_response_affinity_std': response_affinity_std.detach().cpu(),
                        'branch_response_clash_std': response_clash_std.detach().cpu(),
                        'branch_response_norm': graph_norm(response_dx).detach().cpu(),
                        'branch_response_coefficients': response_coefficients.detach().cpu(),
                    }

            if branch_control is not None and float(branch_control.get('target_base_ratio', 0.0)) > 0:
                branch_active_steps = int(branch_control.get('active_steps', 0))
                branch_active = branch_active_steps <= 0 or guidance_step_index < branch_active_steps
                if branch_active:
                    branch_mode = str(branch_control.get('mode', 'outward'))
                    branch_sign = float(branch_control.get('sign', 1.0))
                    branch_raw_dx = torch.zeros_like(ligand_xt)
                    if branch_mode == 'fixed_action':
                        fixed_action = branch_control.get('action')
                        if fixed_action is None:
                            raise ValueError('branch_control.action is required for fixed_action mode')
                        fixed_action = torch.as_tensor(
                            fixed_action,
                            dtype=ligand_xt.dtype,
                            device=ligand_xt.device,
                        )
                        if fixed_action.shape != ligand_xt.shape:
                            raise ValueError(
                                'branch_control.action shape must match ligand positions: '
                                f'{tuple(fixed_action.shape)} != {tuple(ligand_xt.shape)}'
                            )
                        branch_raw_dx = branch_sign * fixed_action
                    elif branch_mode in {'outward', 'tangent'}:
                        for graph_idx in range(num_graphs):
                            lm = batch_ligand == graph_idx
                            pm = batch_protein == graph_idx
                            if not (bool(lm.any()) and bool(pm.any())):
                                continue
                            radial = ligand_xt[lm].mean(dim=0) - protein_pos[pm].mean(dim=0)
                            radial = radial / radial.norm().clamp_min(1e-8)
                            if branch_mode == 'outward':
                                direction = radial
                            else:
                                axis = radial.new_tensor([0.0, 0.0, 1.0])
                                if float(torch.abs((radial * axis).sum()).detach().cpu()) > 0.90:
                                    axis = radial.new_tensor([0.0, 1.0, 0.0])
                                direction = torch.linalg.cross(radial, axis)
                                direction = direction / direction.norm().clamp_min(1e-8)
                            branch_raw_dx[lm] = branch_sign * direction.unsqueeze(0)
                    elif branch_mode == 'geometry_preserve':
                        with torch.enable_grad():
                            geometry_costs = []
                            for graph_idx in range(num_graphs):
                                lm = batch_ligand == graph_idx
                                if not bool(lm.any()):
                                    geometry_costs.append(ligand_xt.sum() * 0.0)
                                    continue
                                current = ligand_xt[lm]
                                reference = branch_reference_pos[lm]
                                current_dist = torch.cdist(current, current)
                                reference_dist = torch.cdist(reference, reference)
                                mask = (reference_dist > 0.1) & (
                                    reference_dist < float(branch_control.get('pair_cutoff', 2.2))
                                )
                                if bool(mask.any()):
                                    geometry_costs.append((current_dist[mask] - reference_dist[mask]).pow(2).mean())
                                else:
                                    geometry_costs.append(current.sum() * 0.0)
                            geometry_score = torch.stack(geometry_costs).sum()
                            geometry_grad = torch.autograd.grad(
                                geometry_score,
                                ligand_xt,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                        if geometry_grad is not None:
                            branch_raw_dx = -geometry_grad
                    else:
                        raise ValueError(f'Unknown branch control mode: {branch_mode}')

                    branch_base_norm = graph_norm(base_dx).clamp_min(1e-8)
                    branch_raw_norm = graph_norm(branch_raw_dx).clamp_min(1e-8)
                    branch_ratio = float(branch_control.get('target_base_ratio', 0.0))
                    branch_scale = (branch_ratio * branch_base_norm / branch_raw_norm).clamp(
                        max=float(branch_control.get('max_scale', 100.0))
                    )
                    branch_dx = branch_raw_dx * branch_scale[batch_ligand].unsqueeze(-1)
                    dx = dx + branch_dx
                    if trace_velocity:
                        branch_norm = graph_norm(branch_dx)
                        physics_stats.update({
                            'branch_control_norm': branch_norm.detach().cpu(),
                            'branch_control_to_base_ratio': (
                                branch_norm / branch_base_norm.clamp_min(1e-8)
                            ).detach().cpu(),
                        })
            if trace_velocity:
                def graph_norm(vec):
                    return scatter_sum(vec.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                def graph_cos(a, b):
                    dot = scatter_sum((a * b).sum(dim=-1), batch_ligand, dim=0)
                    return dot / (graph_norm(a) * graph_norm(b)).clamp_min(1e-8)

                fm_norm = graph_norm(fm_dx).detach().cpu()
                binding_norm = graph_norm(binding_dx).detach().cpu()
                total_norm = graph_norm(dx).detach().cpu()
                trace_item = {
                    'step_index': int(len(velocity_trace)),
                    'time_index': float(t[0].detach().cpu().item()),
                    'pos_grad_w': float(pos_grad_w),
                    'pred_affinity': pred_affinity.detach().cpu(),
                    'fm_norm': fm_norm,
                    'binding_norm': binding_norm,
                    'total_norm': total_norm,
                    'binding_to_fm_ratio': binding_norm / fm_norm.clamp_min(1e-8),
                    'fm_binding_cos': graph_cos(fm_dx, binding_dx).detach().cpu(),
                    'fm_total_cos': graph_cos(fm_dx, dx).detach().cpu(),
                }
                if flow_sde_D is not None:
                    trace_item.update({
                        'flow_sde_D_mean': scatter_mean(flow_sde_D.squeeze(-1), batch_ligand, dim=0).detach().cpu(),
                        'flow_sde_score_norm': graph_norm(flow_sde_score).detach().cpu(),
                        'flow_sde_drift_norm': graph_norm(flow_sde_drift).detach().cpu(),
                        'flow_sde_sigma_mean': scatter_mean(
                            (2.0 * flow_sde_D * delta_t).sqrt().squeeze(-1), batch_ligand, dim=0
                        ).detach().cpu(),
                    })
                trace_item.update(hjb_stats)
                trace_item.update(physics_stats)
                trace_item.update(local_affinity_stats)
                trace_item.update(branch_response_stats)
                velocity_trace.append(trace_item)

            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            if flow_sde_mode == 'equivalent':
                sde_noise = torch.randn(
                    ligand_xt.shape,
                    dtype=ligand_xt.dtype,
                    device=ligand_xt.device,
                    generator=flow_sde_generator,
                )
                sde_sigma = (2.0 * flow_sde_D * delta_t).sqrt()
                ligand_pos_next = (
                    ligand_xt
                    + (dx * delta_t) * nonzero_mask
                    + nonzero_mask * sde_sigma * sde_noise
                )
            elif noise:
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_xt)
            else:
                ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask
            ligand_xt = ligand_pos_next

            if (t-delta_t*1000)[0] < 0:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v_prev.index_select(0, t.int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            else:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            type_branch_active = (
                branch_type_action is not None
                and abs(branch_type_coefficient) > 0.0
                and branch_type_target_base_ratio > 0.0
                and (branch_type_active_steps <= 0 or guidance_step_index < branch_type_active_steps)
            )
            v1_from_e_prob = F.softmax(v1_from_e, dim=-1)
            if categorical_velocity_mode == 'rao_blackwell':
                # E[one_hot(V_1) | state] is exactly v1_from_e_prob.  Using the
                # conditional expectation removes auxiliary multinomial noise
                # without changing the mean categorical vector field.
                v1_from_e_state = v1_from_e_prob
            elif categorical_velocity_mode == 'stateless_gumbel':
                v1_from_e_index = torch.empty(
                    v1_from_e_prob.size(0), dtype=torch.long,
                    device=v1_from_e_prob.device,
                )
                global_solver_step = 49 - int(i)
                for graph_index in range(int(num_graphs)):
                    atom_mask = batch_ligand == graph_index
                    atom_count = int(atom_mask.sum())
                    generator = torch.Generator(device=v1_from_e_prob.device)
                    stream_id = int(categorical_stream_ids[graph_index])
                    stream_seed = (
                        categorical_velocity_seed
                        + stream_id * 10000019
                        + global_solver_step * 104729
                    ) % (2**63 - 1)
                    generator.manual_seed(stream_seed)
                    uniform = torch.rand(
                        (atom_count, int(self.num_classes)),
                        generator=generator, device=v1_from_e_prob.device,
                        dtype=v1_from_e_prob.dtype,
                    ).clamp_(1e-12, 1.0 - 1e-12)
                    gumbel = -torch.log(-torch.log(uniform))
                    v1_from_e_index[atom_mask] = (
                        v1_from_e_prob[atom_mask].clamp_min(1e-20).log() + gumbel
                    ).argmax(dim=-1)
                v1_from_e_state = F.one_hot(
                    v1_from_e_index, num_classes=self.num_classes
                )
            else:
                v1_from_e_index = torch.multinomial(
                    v1_from_e_prob, num_samples=1
                ).squeeze(-1)
                v1_from_e_state = F.one_hot(
                    v1_from_e_index, num_classes=self.num_classes
                )
            para_v = alpha_v_cumprod_hat * ((1-self.alphas_cumprod_v.index_select(0, t.int()))/self.alphas_cumprod_v.index_select(0, t.int()))

            dv = (v1_from_e_state - init_ligand_v_onehot) * alpha_v_cumprod_hat[batch_ligand][:, None] + para_v[batch_ligand].unsqueeze(1) * v_guidance * v_grad_w
            if atom_raw_dv is not None and atom_target_base_ratio > 0:
                def graph_v_norm(mat):
                    return scatter_sum(mat.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                base_dv_norm = graph_v_norm(dv).clamp_min(1e-8)
                atom_raw_norm = graph_v_norm(atom_raw_dv).clamp_min(1e-8)
                atom_raw_scale = atom_target_base_ratio * base_dv_norm / atom_raw_norm
                atom_scale = atom_raw_scale.clamp(max=atom_max_scale)
                atom_dv = (
                    atom_raw_dv
                    * atom_scale[batch_ligand].unsqueeze(-1)
                    * atom_schedule[batch_ligand].unsqueeze(-1)
                )
                dv = dv + atom_dv
            chemistry_active = chemistry_active_steps <= 0 or guidance_step_index < chemistry_active_steps
            if chemistry_active and chemistry_type_anchor_ratio > 0:
                def graph_type_norm(mat):
                    return scatter_sum(mat.pow(2).sum(dim=-1), batch_ligand, dim=0).sqrt()

                # The sampler initializes categorical logits from an integer
                # one-hot tensor and promotes them after the first Euler update.
                # Chemistry control is active before that update, so cast only
                # this probability view without changing the baseline path.
                current_type_prob = F.softmax(ligand_vt_logits.float(), dim=-1)
                chemistry_type_raw = init_ligand_v_onehot.to(current_type_prob.dtype) - current_type_prob
                # Only atoms participating in the anchor pseudo-bond graph are
                # type-constrained. Coordinate and categorical trust regions
                # therefore share the same local chemistry support.
                if chemistry_type_anchor_mask is not None:
                    chemistry_type_raw = chemistry_type_raw * chemistry_type_anchor_mask.unsqueeze(-1)
                chemistry_type_raw = chemistry_type_raw - chemistry_type_raw.mean(dim=-1, keepdim=True)
                base_type_norm = graph_type_norm(dv).clamp_min(1e-8)
                chemistry_type_norm = graph_type_norm(chemistry_type_raw).clamp_min(1e-8)
                chemistry_type_scale = (
                    chemistry_type_anchor_ratio * base_type_norm / chemistry_type_norm
                ).clamp(max=chemistry_type_max_scale)
                dv = dv + chemistry_type_raw * chemistry_type_scale[batch_ligand].unsqueeze(-1)


            ligand_logits_v_next = ligand_vt_logits + (dv * delta_t) * nonzero_mask
            if type_branch_active:
                # Use the Fisher metric to put coordinate and categorical
                # controls on the same velocity-relative scale.  The Euler
                # factor is applied after normalization, exactly as for dx.
                current_type_prob = F.softmax(ligand_vt_logits.float(), dim=-1)

                def fisher_norm(logit_velocity):
                    centered = logit_velocity - (
                        current_type_prob * logit_velocity
                    ).sum(dim=-1, keepdim=True)
                    return (current_type_prob * centered.square()).sum().sqrt()

                direction = (
                    1.0 if branch_type_coefficient > 0.0 else -1.0
                ) * branch_type_action
                base_fisher_norm = fisher_norm(dv).clamp_min(1e-10)
                action_fisher_norm = fisher_norm(direction).clamp_min(1e-10)
                type_scale = (
                    branch_type_target_base_ratio
                    * abs(branch_type_coefficient)
                    * base_fisher_norm
                    / action_fisher_norm
                ).clamp(max=branch_type_max_scale)
                ligand_logits_v_next = (
                    ligand_logits_v_next
                    + type_scale * direction * delta_t * nonzero_mask
                )

            ligand_vt_prob = F.softmax(ligand_logits_v_next, dim=-1)
            if categorical_transition_mode == 'coupled_gumbel':
                # Separate antithetic sampler calls are seeded identically, so
                # they receive the same uniforms at every transition.  The
                # resulting common-random-number coupling exposes changes in
                # the categorical law without adding independent branch noise.
                uniform = torch.rand_like(ligand_vt_prob).clamp_(1e-12, 1.0 - 1e-12)
                gumbel = -torch.log(-torch.log(uniform))
                ligand_v_next = (
                    ligand_vt_prob.clamp_min(1e-20).log() + gumbel
                ).argmax(dim=-1)
            else:
                ligand_v_next = ligand_vt_prob.argmax(dim=-1)
            if (
                categorical_replay_states is not None
                and guidance_step_index < categorical_replay_active_steps
            ):
                # Replay inside the transition loop: the next coordinate step
                # sees the matched baseline categorical state.  Once the
                # requested prefix ends, categorical evolution is released.
                ligand_v_next = categorical_replay_states[guidance_step_index]
                ligand_logits_v_next = F.one_hot(
                    ligand_v_next, num_classes=self.num_classes
                ).to(dtype=ligand_logits_v_next.dtype)
                ligand_vt_prob = ligand_logits_v_next
            ligand_vt = (
                ligand_vt_prob if categorical_state_mode == 'simplex'
                else ligand_v_next
            )
            ligand_vt_logits = ligand_logits_v_next

            ori_ligand_pos = ligand_xt + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v_next.clone().cpu())
            # Atom classes alone are not a sufficient Markov state for a
            # resumed hybrid flow: the categorical ODE evolves continuous
            # logits.  Preserve them so counterfactual branches can start from
            # exactly the same state as the uninterrupted PAFlow trajectory.
            v_logits_traj.append(ligand_vt_logits.clone().cpu())

        ligand_x1 = ligand_xt
        ligand_v1 = (
            ligand_vt.argmax(dim=-1)
            if categorical_state_mode == 'simplex' else ligand_vt
        )
        ligand_x1 = ligand_x1 + offset[batch_ligand]
        return {
            'pos': ligand_x1,
            'v': ligand_v1,
            'initial_v': initial_ligand_v,
            'categorical_reference_v': reference_v.detach().clone(),
            'v_logits': ligand_vt_logits,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v_logits_traj': v_logits_traj,
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

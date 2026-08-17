import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils import reconstruct, transforms

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
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
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

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
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

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
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

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
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

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
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
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos_all = loss_pos
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
        }

    def get_diffusion_loss_normalized_alpha(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        min_val = torch.min(self.alphas_cumprod)
        max_val = torch.max(self.alphas_cumprod)
        normalized_alphas_cumprod = (self.alphas_cumprod - min_val) / (max_val - min_val)
        a = normalized_alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
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
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos_all = loss_pos
        loss_pos = torch.mean(loss_pos)

        # atom type loss
        # log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        # log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        # log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        # kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
        #                          log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        # loss_v = torch.mean(kl_v)
        loss_v = F.cross_entropy(pred_ligand_v, ligand_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
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
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # time sequence
        # time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        time_seq = list(reversed(range(0, num_steps)))
        delta_t = 1/num_steps
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            # t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            t = torch.full(size=(num_graphs,), fill_value=delta_t*i*1000, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            # ligand_pos_next = pos_model_mean
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }



class ScorePosNet3D_flow(nn.Module):

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

    def forward(self, protein_pos, protein_v, batch_protein, ligand_xt, ligand_vt, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1
        ligand_vt = F.one_hot(ligand_vt, self.num_classes).float()
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

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
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

    def sample_time_ot(self, num_graphs, device, method):
        if method == 'symmetric':
            time_step = torch.rand(num_graphs // 2 + 1, device=device)
            time_step = torch.cat(
                [time_step, 1-time_step], dim=0)[:num_graphs]
            delta_t = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, delta_t[0]

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

    def OT_path_v(self, v_1, t, batch):
        v_1_onehot = F.one_hot(v_1, self.num_classes)
        v0 = torch.randint(0, self.num_classes, (len(v_1),)).to(v_1)
        v_0_onehot = F.one_hot(v0, self.num_classes)
        # v_perturbed = v_1_onehot * t[batch] + (1-t[batch]) * (1. / self.num_classes)
        v_perturbed = v_1_onehot * t[batch] + (1-t[batch]) * v_0_onehot
        uniform = torch.rand_like(v_perturbed.float())
        gumbel_noise = -torch.log(uniform + 1e-30) + 1e-30
        v_t = (v_perturbed/gumbel_noise).argmax(dim=-1)
        log_vt = index_to_log_onehot(v_t, self.num_classes)
        return v_t, log_vt, v_0_onehot

    def VP_field(self, x1, xt, t, delta_t, batch_ligand):
        a_bar_t = self.alphas_cumprod.index_select(0, t.int())
        # a_bar_t_sub_1 = self.alphas_cumprod_prev.index_select(0, t.int())
        if (t-delta_t*1000)[0] < 0:
            a_bar_t_sub_1 = self.alphas_cumprod_prev.index_select(0, t.int())
        else:
            a_bar_t_sub_1 = self.alphas_cumprod.index_select(0, (t-delta_t*1000).int())
        sqrt_a_bar_hat = -(torch.sqrt(a_bar_t_sub_1[batch_ligand])-torch.sqrt(a_bar_t[batch_ligand]))/delta_t # delta_t = 0.001
        M_para = sqrt_a_bar_hat / (1 - a_bar_t[batch_ligand] + 1e-5)
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

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
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
            ligand_vt=ligand_vt,
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
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x1': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            # 'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
        }

    def get_otflow_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None, init_pos='std_gaussian'
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, delta_t = self.sample_time_ot(num_graphs, protein_pos.device, self.sample_time_method)
            t = time_step.view(num_graphs, 1)
        else:
            t = time_step.view(num_graphs, 1)
            delta_t = torch.ones_like(time_step).float() / self.num_timesteps
            delta_t = delta_t[0]

        ligand_x1 = ligand_pos
        ligand_v1 = ligand_v

        # 2. perturb pos and v
        ligand_x0 = torch.zeros_like(ligand_x1)
        if init_pos == 'std_gaussian':
            ligand_x0.normal_()
        elif init_pos == 'prior':
            mean = 0
            std = torch.sqrt(torch.tensor([8.7587, 8.5384, 9.3056])).to(ligand_x0)
            ligand_x0.normal_()
            ligand_x0 = ligand_x0 * std + mean
        # Xt = t * X1 + (1-t) * eps
        ligand_xt = t[batch_ligand] * ligand_x1 + (1.0 - t[batch_ligand]) * ligand_x0
        # Vt = t * V1 + (1-t) / K
        log_ligand_v1 = index_to_log_onehot(ligand_v1, self.num_classes)
        ligand_vt, log_ligand_vt, v_0_onehot = self.OT_path_v(v_1 = ligand_v1, t = t, batch = batch_ligand)
        ligand_vt_one_hot = F.one_hot(ligand_vt, num_classes=self.num_classes)
        # ligand_v_perturbed, log_ligand_vt_diff = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, uniform)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            ligand_xt=ligand_xt,
            ligand_vt=ligand_vt,
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
        # diffusion v loss
        # log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        # log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        # log_v_true_prob = self.q_v_posterior(log_ligand_v1, log_ligand_vt, time_step, batch_ligand)
        # kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v1,
                                #  log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)

        # v loss version_1
        # ligand_v_prob_recon = F.softmax(pred_ligand_v, dim=-1)
        # ligand_v_index_recon = ligand_v_prob_recon.argmax(dim=-1)
        # ligand_v_model_prob = self.q_v_posterior_ot(ligand_v_index_recon, v_0_onehot, ligand_vt_one_hot, delta_t, time_step, batch_ligand)
        # ligand_v_true_prob = self.q_v_posterior_ot(ligand_v1, v_0_onehot, ligand_vt_one_hot, delta_t, time_step, batch_ligand)
        # kl_v = self.compute_v_Lt(log_v_model_prob=torch.log(ligand_v_model_prob), log_v0=log_ligand_v1,
        #                          log_v_true_prob=torch.log(ligand_v_true_prob), t=time_step, batch=batch_ligand)
        # loss_v = torch.mean(kl_v)

        # v loss version_2
        loss_v = F.cross_entropy(pred_ligand_v, ligand_v)

        loss = loss_pos + loss_v * self.loss_v_weight


        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x1': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            # 'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
        }

# modified ot
    def get_motflow_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None, init_pos='std_gaussian'
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        t = time_step.view(num_graphs, 1) / self.num_timesteps

        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)

        ligand_x1 = ligand_pos
        ligand_v1 = ligand_v

        # 2. perturb pos and v
        ligand_x0 = torch.zeros_like(ligand_x1)
        if init_pos == 'std_gaussian':
            ligand_x0.normal_()
        elif init_pos == 'prior':
            mean = 0
            std = torch.sqrt(torch.tensor([8.7587, 8.5384, 9.3056])).to(ligand_x0)
            ligand_x0.normal_()
            ligand_x0 = ligand_x0 * std + mean

        # Xt = alpha_t * X1 + (1-alpha_t) * eps
        ligand_xt = a_pos.sqrt() * ligand_x1 + (1.0 - a_pos.sqrt()) * ligand_x0
        # Vt = alpha_t * V1 + (1-alpha_t) / K
        log_ligand_v1 = index_to_log_onehot(ligand_v1, self.num_classes)
        ligand_vt, log_ligand_vt, uniform = self.VP_path_v(v_1 = ligand_v1, t = time_step, batch = batch_ligand)
        # ligand_v_perturbed, log_ligand_vt_diff = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, uniform)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            ligand_xt=ligand_xt,
            ligand_vt=ligand_vt,
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

        loss = loss_pos + loss_v * self.loss_v_weight


        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x1': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            # 'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'loss_pos_all': loss_pos_all,
            't': time_step,
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
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                ligand_xt=ligand_pos,
                ligand_vt=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            ligand_pos = ligand_pos_next

            if not pos_only:
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }

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
                para_v = alpha_v_cumprod_hat
            else:
                alpha_v_cumprod_hat = (self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
                para_v = alpha_v_cumprod_hat

            # if (t-delta_t*1000)[0] < 0:
            #     alpha_v_cumprod_hat = (self.alphas_cumprod_v_prev.index_select(0, t.int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            #     para_v = alpha_v_cumprod_hat / (1 - self.alphas_cumprod_v.index_select(0, t.int()))
            # else:
            #     alpha_v_cumprod_hat = (self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()) - self.alphas_cumprod_v.index_select(0, t.int()))/delta_t
            #     para_v = alpha_v_cumprod_hat / (1 - self.alphas_cumprod_v.index_select(0, (t-delta_t*1000).int()))

            v1_from_e_prob = F.softmax(v1_from_e, dim=-1)
            v1_from_e_index = torch.multinomial(v1_from_e_prob, num_samples=1).squeeze(-1)
            # 将索引转换为 one-hot 编码
            v1_from_e_one_hot = F.one_hot(v1_from_e_index, num_classes=self.num_classes)
            # dv = (v1_from_e_one_hot - (torch.ones(len(batch_ligand), self.num_classes)
            #     * (1 / self.num_classes)).to(protein_pos.device)) * alpha_v_cumprod_hat[batch_ligand][:, None]
            dv = (v1_from_e_one_hot - init_ligand_v_onehot) * para_v[batch_ligand][:, None]
            # dv = (v1_from_e_one_hot - init_ligand_v_onehot) * alpha_v_cumprod_hat[batch_ligand][:, None]

            # dv = (F.softmax(v1_from_e, dim=-1) - (torch.ones(len(batch_ligand), self.num_classes) * (1 / self.num_classes)).to(protein_pos.device)) * alpha_v_cumprod_hat[batch_ligand][:, None]
            # dv = (F.softmax(v1_from_e, dim=-1) - (torch.ones(len(batch_ligand), self.num_classes) * (1 / self.num_classes)).to(protein_pos.device))
            # uniform = torch.rand_like(dv)
            # gumbel_noise = -torch.log(uniform + 1e-30) + 1e-30
            # dv = dv / gumbel_noise
            # if t[0] == 999:
            #     ligand_vt_onehot = (torch.ones(len(batch_ligand), self.num_classes) * (1 / self.num_classes)).to(protein_pos.device)
            # else:
            #     ligand_vt_onehot = F.one_hot(ligand_vt, self.num_classes)

            ligand_logits_v_next = ligand_vt_logits + (dv * delta_t) * nonzero_mask
            # log_ligand_prob_v_next = torch.log(ligand_prob_v_next)
            # log_ligand_prob_v_next[torch.isnan(log_ligand_prob_v_next)] = -1000
            # ligand_v_next = log_sample_categorical(log_ligand_prob_v_next)
            ligand_vt_prob = F.softmax(ligand_logits_v_next, dim=-1)
            # ligand_v_next = torch.multinomial(ligand_vt_prob, num_samples=1).squeeze(-1)
            ligand_v_next = ligand_vt_prob.argmax(dim=-1)
            ligand_vt = ligand_v_next
            ligand_vt_logits = ligand_logits_v_next


            # if not pos_only:
            #     log_ligand_v_recon = F.log_softmax(v1_from_e, dim=-1)
            #     log_ligand_v = index_to_log_onehot(ligand_vt, self.num_classes)
            #     log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                # ligand_v_next = log_sample_categorical(log_model_prob)

            #     v1_pred_traj.append(log_ligand_v_recon.clone().cpu())
            #     vt_pred_traj.append(log_model_prob.clone().cpu())
            #     ligand_vt = ligand_v_next

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
            # 'v1_traj': v1_pred_traj,
            # 'vt_traj': vt_pred_traj
        }

    @torch.no_grad()
    def sample_flow_OT(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        x1_pred_traj, v1_pred_traj = [], []
        ligand_xt, ligand_vt = init_ligand_pos, init_ligand_v

        init_ligand_v_onehot = F.one_hot(ligand_vt, self.num_classes)
        ligand_vt_logits = init_ligand_v_onehot
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        # time_seq = list(reversed(range(0, 20)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
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

            delta_t = 0.001
            # delta_t = 0.05
            dx = x1_from_e - init_ligand_pos
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = ligand_xt + (dx * delta_t) * nonzero_mask
            ligand_xt = ligand_pos_next

            v1_from_e_prob = F.softmax(v1_from_e, dim=-1)
            v1_from_e_index = torch.multinomial(v1_from_e_prob, num_samples=1).squeeze(-1)
            # 将索引转换为 one-hot 编码
            v1_from_e_one_hot = F.one_hot(v1_from_e_index, num_classes=self.num_classes)
            dv = v1_from_e_one_hot - init_ligand_v_onehot

            ligand_logits_v_next = ligand_vt_logits + (dv * delta_t) * nonzero_mask
            ligand_vt_prob = F.softmax(ligand_logits_v_next, dim=-1)
            # ligand_v_next = torch.multinomial(ligand_vt_prob, num_samples=1).squeeze(-1)
            ligand_v_next = ligand_vt_prob.argmax(dim=-1)
            ligand_vt = ligand_v_next
            ligand_vt_logits = ligand_logits_v_next

            ori_ligand_pos = ligand_xt + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_vt.clone().cpu())
            ori_ligand_x1 = x1_from_e + offset[batch_ligand]
            x1_pred_traj.append(ori_ligand_x1.clone().cpu())
            v1_pred_traj.append(v1_from_e.clone().cpu())

        ligand_x1 = ligand_xt
        ligand_v1 = ligand_vt
        ligand_x1 = ligand_x1 + offset[batch_ligand]
        return {
            'pos': ligand_x1,
            'v': ligand_v1,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'x1_pred_traj': x1_pred_traj,
            'v1_pred_traj': v1_pred_traj
        }

    # modified ot
    @torch.no_grad()
    def sample_flow_MOT(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        x1_pred_traj, v1_pred_traj = [], []
        ligand_xt, ligand_vt = init_ligand_pos, init_ligand_v

        init_ligand_v_onehot = F.one_hot(ligand_vt, self.num_classes)
        ligand_vt_logits = init_ligand_v_onehot
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        # time_seq = list(reversed(range(0, 20)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
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
            x1_from_e = preds['pred_ligand_pos']
            v1_from_e = preds['pred_ligand_v']

            delta_t = 0.001
            # delta_t = 0.05

            a = self.alphas_cumprod.index_select(0, t)  # (num_graphs, )
            a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
            a_prev = self.alphas_cumprod_prev.index_select(0, t)  # (num_graphs, )
            a_pos_prev = a_prev[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
            alpha_t_hat = (a_pos_prev.sqrt() - a_pos.sqrt()) / delta_t

            dx = alpha_t_hat * (x1_from_e - init_ligand_pos)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
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
            ori_ligand_x1 = x1_from_e + offset[batch_ligand]
            x1_pred_traj.append(ori_ligand_x1.clone().cpu())
            v1_pred_traj.append(v1_from_e.clone().cpu())

        ligand_x1 = ligand_xt
        ligand_v1 = ligand_vt
        ligand_x1 = ligand_x1 + offset[batch_ligand]
        return {
            'pos': ligand_x1,
            'v': ligand_v1,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'x1_pred_traj': x1_pred_traj,
            'v1_pred_traj': v1_pred_traj
        }



def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

# if __name__ == "__main__":
#     pred_aromatic = transforms.is_aromatic_from_index()
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes,
        code_dim,
        commitment_weight=0.25,
        codebook_weight=1.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight

        self.codebook = nn.Parameter(torch.empty(num_codes, code_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.codebook, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(self, z_e):
        if z_e.ndim != 2:
            raise ValueError(f'Expected z_e with shape [N, D], got {tuple(z_e.shape)}.')

        z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
        code_sq = torch.sum(self.codebook ** 2, dim=-1)
        distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, self.codebook.t())

        code_indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook.index_select(0, code_indices)
        z_st = z_e + (z_q - z_e).detach()

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = self.commitment_weight * commitment_loss + self.codebook_weight * codebook_loss

        usage = torch.bincount(code_indices, minlength=self.num_codes).float()
        probs = usage / usage.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())

        return {
            'z_e': z_e,
            'z_q': z_q,
            'z_st': z_st,
            'code_indices': code_indices,
            'distances': distances,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'usage': usage,
            'perplexity': perplexity,
        }

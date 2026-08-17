from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1).float()
    else:
        x = x.float()
    if x.size(-1) == dim:
        return x
    if x.size(-1) > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.size(-1), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


class HJBValueGradientModel(nn.Module):
    """Atom-wise value-gradient field used as a velocity teacher.

    The model has the same input contract as ``HJBActorModel`` so a trained
    value-gradient field can be injected into the existing sampler without
    changing the PAFlow state representation.  It predicts a coordinate-space
    descent direction, i.e. a velocity-like approximation of ``-grad_x S``.
    """

    def __init__(
        self,
        ligand_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        scalar_dim: int = 4,
    ):
        super().__init__()
        self.ligand_feature_dim = int(ligand_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.scalar_dim = int(scalar_dim)
        input_dim = 3 + self.ligand_feature_dim + 3 + 3 + self.scalar_dim
        layers = []
        dim = input_dim
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 3))
        self.net = nn.Sequential(*layers)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(last.bias)

    def _ligand_features(self, ligand_v: torch.Tensor) -> torch.Tensor:
        if ligand_v.dim() == 1 or (ligand_v.dim() == 2 and ligand_v.size(-1) == 1):
            ids = ligand_v.long().view(-1).clamp(min=0, max=self.ligand_feature_dim - 1)
            return F.one_hot(ids, num_classes=self.ligand_feature_dim).float()
        return _pad_or_trim(ligand_v, self.ligand_feature_dim)

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        neg_grad_s: torch.Tensor,
        fm_dx: torch.Tensor,
        graph_scalars: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(batch_ligand.max().item()) + 1
        lig_center = scatter_mean(ligand_pos, batch_ligand, dim=0, dim_size=num_graphs)
        lig_rel = ligand_pos - lig_center.index_select(0, batch_ligand)
        lig_feat = self._ligand_features(ligand_v).to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        scalars = graph_scalars.to(dtype=ligand_pos.dtype, device=ligand_pos.device)
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(-1)
        if scalars.size(-1) != self.scalar_dim:
            scalars = _pad_or_trim(scalars, self.scalar_dim)
        node_scalars = scalars.index_select(0, batch_ligand)
        x = torch.cat([lig_rel, lig_feat, neg_grad_s, fm_dx, node_scalars], dim=-1)
        return self.net(x)


def build_hjb_value_gradient_from_checkpoint(
    ckpt: dict,
    device: str | torch.device = "cpu",
) -> HJBValueGradientModel:
    args = ckpt.get("args", {})
    model = HJBValueGradientModel(
        ligand_feature_dim=int(args.get("ligand_feature_dim", ckpt.get("ligand_feature_dim", 16))),
        hidden_dim=int(args.get("hidden_dim", ckpt.get("hidden_dim", 128))),
        num_layers=int(args.get("num_layers", ckpt.get("num_layers", 3))),
        dropout=float(args.get("dropout", ckpt.get("dropout", 0.0))),
        scalar_dim=int(args.get("scalar_dim", ckpt.get("scalar_dim", 4))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

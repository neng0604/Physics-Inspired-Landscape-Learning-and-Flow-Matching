from __future__ import annotations

import torch
import torch.nn as nn

from models.landscape_encoder import PAFlowLandscapeEncoder


def sampler_fraction_to_paflow_time(
    time_fraction: torch.Tensor,
    sampler_steps: int = 50,
    paflow_timesteps: int = 1000,
) -> torch.Tensor:
    """Map a completed 50-step sampler index to PAFlow's diffusion-time index.

    The branch bank stores ``step / (sampler_steps - 1)``.  PAFlow sampling uses
    ``i = sampler_steps - 1 - step`` and passes
    ``t = i * paflow_timesteps / sampler_steps`` to the network.  The current
    checkpoint has no time embedding, but retaining the exact mapping prevents
    an accidental interface change if another checkpoint is used later.
    """

    if int(sampler_steps) <= 1:
        raise ValueError("sampler_steps must be greater than one")
    generation_step = torch.round(time_fraction * float(int(sampler_steps) - 1)).long()
    sampler_i = (int(sampler_steps) - 1 - generation_step).clamp(
        min=0, max=int(sampler_steps) - 1
    )
    scale = float(int(paflow_timesteps)) / float(int(sampler_steps))
    return torch.round(sampler_i.float() * scale).long().clamp(
        min=0, max=int(paflow_timesteps) - 1
    )


class ScalarFeatureHead(nn.Module):
    """Small scalar decoder for a frozen graph representation."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class FrozenPAFlowScalarPotential(nn.Module):
    """Frozen PAFlow encoder followed by a trainable scalar feature head.

    This wrapper is used only for coordinate-gradient and invariance evaluation.
    Training uses cached graph embeddings from the identical frozen encoder.
    """

    def __init__(
        self,
        encoder: PAFlowLandscapeEncoder,
        head: ScalarFeatureHead,
        sampler_steps: int = 50,
        paflow_timesteps: int = 1000,
        fix_x: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.sampler_steps = int(sampler_steps)
        self.paflow_timesteps = int(paflow_timesteps)
        self.fix_x = bool(fix_x)
        self.encoder.freeze_backbone()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        device,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        sampler_steps: int = 50,
        paflow_timesteps: int = 1000,
        fix_x: bool = False,
    ) -> "FrozenPAFlowScalarPotential":
        encoder = PAFlowLandscapeEncoder.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            pooling="mean",
            graph_context="joint",
            graph_emb_dim=None,
            freeze_backbone=True,
            center_pos_mode="protein",
            strict=False,
        ).to(device)
        head = ScalarFeatureHead(
            input_dim=int(encoder.readout_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        ).to(device)
        return cls(
            encoder=encoder,
            head=head,
            sampler_steps=sampler_steps,
            paflow_timesteps=paflow_timesteps,
            fix_x=fix_x,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen pretrained representation must remain deterministic.  Only
        # the scalar head follows the requested train/eval mode.
        self.encoder.eval()
        return self

    def encode(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        time_step = sampler_fraction_to_paflow_time(
            time_fraction=time_fraction,
            sampler_steps=self.sampler_steps,
            paflow_timesteps=self.paflow_timesteps,
        )
        result = self.encoder.encode_nodes(
            protein_pos=protein_pos,
            protein_atom_feature=protein_v,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            time_step=time_step,
            center_pos_mode="protein",
            return_all=False,
            fix_x=self.fix_x,
        )
        return result["graph_emb"]

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(
            ligand_pos,
            ligand_v,
            protein_pos,
            protein_v,
            batch_ligand,
            batch_protein,
            time_fraction,
        )
        value = self.head(features)
        return value, torch.zeros_like(value)


class FrozenPAFlowHJBValue(FrozenPAFlowScalarPotential):
    """Sampling adapter exposing the cached-feature potential as one scalar value."""

    head_names = ["total"]

    def forward(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        batch_protein: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode(
            ligand_pos,
            ligand_v,
            protein_pos,
            protein_v,
            batch_ligand,
            batch_protein,
            time_fraction,
        )
        return self.head(features)

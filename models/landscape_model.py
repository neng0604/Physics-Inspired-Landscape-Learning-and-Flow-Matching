import torch
import torch.nn as nn
import torch.nn.functional as F

from models.landscape_vq import VectorQuantizer


def build_mlp(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
    if num_layers < 1:
        raise ValueError('num_layers must be >= 1')

    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)

    layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class PESLALiteLandscapeModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        code_dim,
        num_codes,
        projector_layers=2,
        transition_hidden_dim=None,
        recon_hidden_dim=None,
        decoder_layers=2,
        projector_dropout=0.0,
        commitment_weight=0.25,
        codebook_weight=1.0,
        use_reconstruction=True,
        use_quality_head=False,
        quality_output_dim=4,
        use_basin_classifier=False,
        basin_num_classes=3,
        use_quality_aware_assignment=False,
        quality_assign_dim=None,
        quality_assign_hidden_dim=None,
        quality_assign_weight=0.0,
    ):
        super().__init__()
        transition_hidden_dim = transition_hidden_dim or hidden_dim
        recon_hidden_dim = recon_hidden_dim or hidden_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.use_reconstruction = use_reconstruction
        self.use_quality_head = bool(use_quality_head)
        self.use_basin_classifier = bool(use_basin_classifier)
        self.use_quality_aware_assignment = bool(use_quality_aware_assignment)
        self.quality_assign_weight = float(quality_assign_weight)

        self.projector = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=code_dim,
            num_layers=projector_layers,
            dropout=projector_dropout,
        )
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
        )
        self.transition_head = build_mlp(
            input_dim=code_dim,
            hidden_dim=transition_hidden_dim,
            output_dim=num_codes,
            num_layers=2,
            dropout=projector_dropout,
        )
        self.quality_head = None
        if self.use_quality_head:
            self.quality_head = build_mlp(
                input_dim=code_dim,
                hidden_dim=transition_hidden_dim,
                output_dim=int(quality_output_dim),
                num_layers=2,
                dropout=projector_dropout,
            )
        self.basin_classifier = None
        if self.use_basin_classifier:
            self.basin_classifier = build_mlp(
                input_dim=code_dim,
                hidden_dim=transition_hidden_dim,
                output_dim=int(basin_num_classes),
                num_layers=2,
                dropout=projector_dropout,
            )
        self.assignment_quality_head = None
        self.code_quality_prototypes = None
        if self.use_quality_aware_assignment:
            quality_assign_dim = int(quality_assign_dim or quality_output_dim)
            quality_assign_hidden_dim = int(quality_assign_hidden_dim or transition_hidden_dim)
            self.assignment_quality_head = build_mlp(
                input_dim=code_dim,
                hidden_dim=quality_assign_hidden_dim,
                output_dim=quality_assign_dim,
                num_layers=2,
                dropout=projector_dropout,
            )
            self.code_quality_prototypes = nn.Parameter(torch.zeros(num_codes, quality_assign_dim))

        if use_reconstruction:
            self.decoder = build_mlp(
                input_dim=code_dim,
                hidden_dim=recon_hidden_dim,
                output_dim=input_dim,
                num_layers=decoder_layers,
                dropout=projector_dropout,
            )
        else:
            self.decoder = None

        self.code_energies = nn.Parameter(torch.zeros(num_codes))
        self.register_buffer('code_values', torch.zeros(num_codes), persistent=False)
        self.register_buffer('code_policy_next', torch.arange(num_codes, dtype=torch.long), persistent=False)
        self.has_code_values = False
        self.has_code_policy_next = False

    def set_code_values(self, values):
        values = values.detach().to(device=self.code_energies.device, dtype=self.code_energies.dtype)
        if values.numel() != self.num_codes:
            raise ValueError(f'Expected {self.num_codes} code values, got {values.numel()}')
        self.code_values = values.view(self.num_codes)
        self.has_code_values = True

    def set_code_policy_next(self, next_codes):
        next_codes = next_codes.detach().to(device=self.code_energies.device, dtype=torch.long)
        if next_codes.numel() != self.num_codes:
            raise ValueError(f'Expected {self.num_codes} next-code entries, got {next_codes.numel()}')
        if next_codes.min().item() < 0 or next_codes.max().item() >= self.num_codes:
            raise ValueError('next-code policy contains indices outside the codebook range')
        self.code_policy_next = next_codes.view(self.num_codes)
        self.has_code_policy_next = True

    def compute_assignment_quality(self, z_e):
        if self.assignment_quality_head is None:
            return None
        return self.assignment_quality_head(z_e)

    def compute_code_distances(self, z_e, assignment_quality=None):
        codebook = self.quantizer.codebook
        z_sq = torch.sum(z_e ** 2, dim=-1, keepdim=True)
        code_sq = torch.sum(codebook ** 2, dim=-1)
        base_distances = z_sq + code_sq.unsqueeze(0) - 2.0 * torch.matmul(z_e, codebook.t())

        quality_distances = None
        if self.use_quality_aware_assignment:
            if assignment_quality is None:
                assignment_quality = self.compute_assignment_quality(z_e)
            proto = self.code_quality_prototypes
            q_sq = torch.sum(assignment_quality ** 2, dim=-1, keepdim=True)
            proto_sq = torch.sum(proto ** 2, dim=-1)
            quality_distances = q_sq + proto_sq.unsqueeze(0) - 2.0 * torch.matmul(assignment_quality, proto.t())
            distances = base_distances + self.quality_assign_weight * quality_distances
        else:
            distances = base_distances
        return distances, base_distances, quality_distances, assignment_quality

    def compute_soft_assign_from_latent(self, z_e, tau=1.0, assignment_quality=None):
        distances, _, _, _ = self.compute_code_distances(z_e, assignment_quality=assignment_quality)
        soft_assign = F.softmax(-distances / max(float(tau), 1e-6), dim=-1)
        return soft_assign

    def compute_value_energy(self, soft_assign):
        if not self.has_code_values:
            raise ValueError('value guidance requires code_values to be attached first.')
        return torch.sum(soft_assign * self.code_values.unsqueeze(0), dim=-1)

    def compute_prototype_attraction_energy(self, z_e, soft_assign):
        if not self.has_code_policy_next:
            raise ValueError('prototype guidance requires a next-code policy to be attached first.')
        next_codes = self.code_policy_next.to(device=self.quantizer.codebook.device)
        next_codebook = self.quantizer.codebook.index_select(0, next_codes)
        target_proto = torch.matmul(soft_assign, next_codebook).detach()
        return torch.mean((z_e - target_proto) ** 2, dim=-1)

    def encode_latent(self, x):
        z_e = self.projector(x)
        assignment_quality_pred = self.compute_assignment_quality(z_e)
        if self.use_quality_aware_assignment:
            distances, base_distances, quality_distances, assignment_quality_pred = self.compute_code_distances(
                z_e,
                assignment_quality=assignment_quality_pred,
            )
            code_indices = torch.argmin(distances, dim=-1)
            z_q = self.quantizer.codebook.index_select(0, code_indices)
            z_st = z_e + (z_q - z_e).detach()

            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            vq_loss = self.quantizer.commitment_weight * commitment_loss + self.quantizer.codebook_weight * codebook_loss

            usage = torch.bincount(code_indices, minlength=self.num_codes).float()
            probs = usage / usage.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            soft_assign = F.softmax(-distances, dim=-1)
            vq_outputs = {
                'z_e': z_e,
                'z_q': z_q,
                'z_st': z_st,
                'code_indices': code_indices,
                'distances': distances,
                'base_distances': base_distances,
                'quality_distances': quality_distances,
                'assignment_quality_pred': assignment_quality_pred,
                'vq_loss': vq_loss,
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'usage': usage,
                'perplexity': perplexity,
            }
        else:
            vq_outputs = self.quantizer(z_e)
            soft_assign = F.softmax(-vq_outputs['distances'], dim=-1)
        return z_e, vq_outputs, soft_assign

    def guidance_energy(self, x, tau=1.0, mode='code'):
        z_e = self.projector(x)
        assignment_quality = self.compute_assignment_quality(z_e)
        soft_assign = self.compute_soft_assign_from_latent(z_e, tau=tau, assignment_quality=assignment_quality)
        if mode == 'value':
            return self.compute_value_energy(soft_assign)
        if mode == 'prototype':
            return self.compute_prototype_attraction_energy(z_e, soft_assign)
        if mode in {'value_proto', 'value_proto_local'}:
            return (
                self.compute_value_energy(soft_assign)
                + self.compute_prototype_attraction_energy(z_e, soft_assign)
            )
        return torch.sum(soft_assign * self.code_energies.unsqueeze(0), dim=-1)

    def forward(self, x):
        z_e, vq_outputs, soft_assign = self.encode_latent(x)
        z_st = vq_outputs['z_st']
        code_indices = vq_outputs['code_indices']
        assignment_quality_pred = vq_outputs.get('assignment_quality_pred')
        state_energy = self.code_energies.index_select(0, code_indices)
        soft_state_energy = torch.sum(soft_assign * self.code_energies.unsqueeze(0), dim=-1)
        transition_logits = self.transition_head(z_st)

        outputs = {
            **vq_outputs,
            'input_x': x,
            'soft_assign': soft_assign,
            'transition_logits': transition_logits,
            'state_energy': state_energy,
            'soft_state_energy': soft_state_energy,
            'code_energies': self.code_energies,
        }

        if self.decoder is not None:
            recon_x = self.decoder(z_st)
            recon_loss = F.mse_loss(recon_x, x)
            outputs.update({
                'recon_x': recon_x,
                'recon_loss': recon_loss,
            })
        else:
            outputs['recon_loss'] = x.new_zeros(())

        if self.quality_head is not None:
            outputs['quality_pred'] = self.quality_head(z_st)
        if self.basin_classifier is not None:
            outputs['basin_logits'] = self.basin_classifier(z_st)
        if assignment_quality_pred is not None:
            outputs['assignment_quality_pred'] = assignment_quality_pred

        return outputs

    def occupancy_energy_loss(self, code_indices, tau=1.0):
        counts = torch.bincount(code_indices, minlength=self.num_codes).float()
        pi_hat = counts / counts.sum().clamp_min(1.0)
        log_pi_energy = F.log_softmax(-self.code_energies / tau, dim=0)
        loss = torch.sum(pi_hat * (torch.log(pi_hat.clamp_min(1e-12)) - log_pi_energy))
        return loss, pi_hat

    def laplacian_smoothness_loss(self, src_code_indices, dst_code_indices, edge_weight=None):
        energy_src = self.code_energies.index_select(0, src_code_indices)
        energy_dst = self.code_energies.index_select(0, dst_code_indices)
        diff_sq = (energy_src - energy_dst) ** 2
        if edge_weight is not None:
            diff_sq = diff_sq * edge_weight
        return diff_sq.mean()


class DualCodebookPESLALandscapeModel(PESLALiteLandscapeModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        code_dim,
        num_codes,
        projector_layers=2,
        transition_hidden_dim=None,
        recon_hidden_dim=None,
        decoder_layers=2,
        projector_dropout=0.0,
        commitment_weight=0.25,
        codebook_weight=1.0,
        use_reconstruction=True,
        use_quality_head=False,
        quality_output_dim=4,
        use_basin_classifier=False,
        basin_num_classes=3,
        use_quality_aware_assignment=False,
        quality_assign_dim=None,
        quality_assign_hidden_dim=None,
        quality_assign_weight=0.0,
        quality_code_dim=None,
        quality_num_codes=None,
        quality_projector_layers=None,
        quality_hidden_dim=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            projector_layers=projector_layers,
            transition_hidden_dim=transition_hidden_dim,
            recon_hidden_dim=recon_hidden_dim,
            decoder_layers=decoder_layers,
            projector_dropout=projector_dropout,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_reconstruction=use_reconstruction,
            use_quality_head=use_quality_head,
            quality_output_dim=quality_output_dim,
            use_basin_classifier=use_basin_classifier,
            basin_num_classes=basin_num_classes,
            use_quality_aware_assignment=use_quality_aware_assignment,
            quality_assign_dim=quality_assign_dim,
            quality_assign_hidden_dim=quality_assign_hidden_dim,
            quality_assign_weight=quality_assign_weight,
        )
        self.quality_code_dim = int(quality_code_dim or code_dim)
        self.quality_num_codes = int(quality_num_codes or num_codes)
        quality_hidden_dim = int(quality_hidden_dim or hidden_dim)
        quality_projector_layers = int(quality_projector_layers or projector_layers)

        self.quality_projector = build_mlp(
            input_dim=input_dim,
            hidden_dim=quality_hidden_dim,
            output_dim=self.quality_code_dim,
            num_layers=quality_projector_layers,
            dropout=projector_dropout,
        )
        self.quality_quantizer = VectorQuantizer(
            num_codes=self.quality_num_codes,
            code_dim=self.quality_code_dim,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
        )
        self.quality_code_energies = nn.Parameter(torch.zeros(self.quality_num_codes))

    def encode_quality_latent(self, x):
        quality_z_e = self.quality_projector(x)
        quality_outputs = self.quality_quantizer(quality_z_e)
        quality_soft_assign = F.softmax(-quality_outputs['distances'], dim=-1)
        return quality_z_e, quality_outputs, quality_soft_assign

    def forward(self, x):
        outputs = super().forward(x)
        quality_z_e, quality_outputs, quality_soft_assign = self.encode_quality_latent(x)
        quality_code_indices = quality_outputs['code_indices']
        quality_state_energy = self.quality_code_energies.index_select(0, quality_code_indices)
        quality_soft_state_energy = torch.sum(
            quality_soft_assign * self.quality_code_energies.unsqueeze(0),
            dim=-1,
        )
        outputs.update({
            'quality_z_e': quality_z_e,
            'quality_z_q': quality_outputs['z_q'],
            'quality_z_st': quality_outputs['z_st'],
            'quality_code_indices': quality_code_indices,
            'quality_distances': quality_outputs['distances'],
            'quality_soft_assign': quality_soft_assign,
            'quality_vq_loss': quality_outputs['vq_loss'],
            'quality_commitment_loss': quality_outputs['commitment_loss'],
            'quality_codebook_loss': quality_outputs['codebook_loss'],
            'quality_usage': quality_outputs['usage'],
            'quality_perplexity': quality_outputs['perplexity'],
            'quality_state_energy': quality_state_energy,
            'quality_soft_state_energy': quality_soft_state_energy,
            'quality_code_energies': self.quality_code_energies,
            'combined_soft_state_energy': outputs['soft_state_energy'].detach() + quality_soft_state_energy,
        })
        return outputs


class LocalResidualPESLALandscapeModel(PESLALiteLandscapeModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        code_dim,
        num_codes,
        projector_layers=2,
        transition_hidden_dim=None,
        recon_hidden_dim=None,
        decoder_layers=2,
        projector_dropout=0.0,
        commitment_weight=0.25,
        codebook_weight=1.0,
        use_reconstruction=True,
        local_hidden_dim=None,
        local_layers=2,
        local_scale=1.0,
        use_quality_head=False,
        quality_output_dim=4,
        use_basin_classifier=False,
        basin_num_classes=3,
        use_quality_aware_assignment=False,
        quality_assign_dim=None,
        quality_assign_hidden_dim=None,
        quality_assign_weight=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            projector_layers=projector_layers,
            transition_hidden_dim=transition_hidden_dim,
            recon_hidden_dim=recon_hidden_dim,
            decoder_layers=decoder_layers,
            projector_dropout=projector_dropout,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_reconstruction=use_reconstruction,
            use_quality_head=use_quality_head,
            quality_output_dim=quality_output_dim,
            use_basin_classifier=use_basin_classifier,
            basin_num_classes=basin_num_classes,
            use_quality_aware_assignment=use_quality_aware_assignment,
            quality_assign_dim=quality_assign_dim,
            quality_assign_hidden_dim=quality_assign_hidden_dim,
            quality_assign_weight=quality_assign_weight,
        )
        local_hidden_dim = local_hidden_dim or hidden_dim
        self.local_scale = float(local_scale)
        self.local_energy_head = build_mlp(
            input_dim=code_dim * 2 + 1,
            hidden_dim=local_hidden_dim,
            output_dim=1,
            num_layers=local_layers,
            dropout=projector_dropout,
        )
        self.reset_local_energy_head()

    def reset_local_energy_head(self):
        last_layer = None
        if isinstance(self.local_energy_head, nn.Sequential):
            for module in reversed(self.local_energy_head):
                if isinstance(module, nn.Linear):
                    last_layer = module
                    break
        elif isinstance(self.local_energy_head, nn.Linear):
            last_layer = self.local_energy_head
        if last_layer is not None:
            nn.init.normal_(last_layer.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(last_layer.bias)

    def compute_local_energy(self, z_e, soft_assign):
        code_context = torch.matmul(soft_assign, self.quantizer.codebook)
        code_energy_context = torch.matmul(soft_assign, self.code_energies.unsqueeze(-1))
        local_input = torch.cat([z_e, code_context, code_energy_context], dim=-1)
        return self.local_scale * self.local_energy_head(local_input).squeeze(-1)

    def guidance_energy(self, x, tau=1.0, mode='local'):
        z_e = self.projector(x)
        soft_assign = self.compute_soft_assign_from_latent(z_e, tau=tau)
        code_energy = torch.sum(soft_assign * self.code_energies.unsqueeze(0), dim=-1)
        value_energy = None
        if mode in {'value', 'value_local', 'value_proto', 'value_proto_local'}:
            value_energy = self.compute_value_energy(soft_assign)
        prototype_energy = None
        if mode in {'prototype', 'value_proto', 'value_proto_local'}:
            prototype_energy = self.compute_prototype_attraction_energy(z_e, soft_assign)
        local_energy = self.compute_local_energy(z_e, soft_assign)
        if mode == 'code':
            return code_energy
        if mode == 'value':
            return value_energy
        if mode == 'prototype':
            return prototype_energy
        if mode == 'value_proto':
            return value_energy + prototype_energy
        if mode == 'value_proto_local':
            return value_energy + prototype_energy + local_energy
        if mode == 'value_local':
            return value_energy + local_energy
        if mode == 'total':
            return code_energy.detach() + local_energy
        if mode == 'local':
            return local_energy
        raise ValueError(f'Unsupported guidance energy mode: {mode}')

    def forward(self, x):
        outputs = super().forward(x)
        local_energy = self.compute_local_energy(outputs['z_e'], outputs['soft_assign'])
        total_soft_state_energy = outputs['soft_state_energy'].detach() + local_energy
        outputs.update({
            'local_energy': local_energy,
            'total_soft_state_energy': total_soft_state_energy,
            'force_energy': local_energy,
        })
        return outputs


class DynamicPESLALandscapeModel(PESLALiteLandscapeModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        code_dim,
        num_codes,
        projector_layers=2,
        transition_hidden_dim=None,
        recon_hidden_dim=None,
        decoder_layers=2,
        projector_dropout=0.0,
        commitment_weight=0.25,
        codebook_weight=1.0,
        use_reconstruction=True,
        dynamic_hidden_dim=128,
        dynamic_num_steps=2,
        dynamic_temperature=1.0,
        dynamic_self_loop=0.1,
        dynamic_residual_weight=1.0,
        dynamic_energy_bias=1.0,
        use_quality_head=False,
        quality_output_dim=4,
        use_basin_classifier=False,
        basin_num_classes=3,
        use_quality_aware_assignment=False,
        quality_assign_dim=None,
        quality_assign_hidden_dim=None,
        quality_assign_weight=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            projector_layers=projector_layers,
            transition_hidden_dim=transition_hidden_dim,
            recon_hidden_dim=recon_hidden_dim,
            decoder_layers=decoder_layers,
            projector_dropout=projector_dropout,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_reconstruction=use_reconstruction,
            use_quality_head=use_quality_head,
            quality_output_dim=quality_output_dim,
            use_basin_classifier=use_basin_classifier,
            basin_num_classes=basin_num_classes,
            use_quality_aware_assignment=use_quality_aware_assignment,
            quality_assign_dim=quality_assign_dim,
            quality_assign_hidden_dim=quality_assign_hidden_dim,
            quality_assign_weight=quality_assign_weight,
        )
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.dynamic_num_steps = dynamic_num_steps
        self.dynamic_temperature = dynamic_temperature
        self.dynamic_self_loop = dynamic_self_loop
        self.dynamic_residual_weight = dynamic_residual_weight
        self.dynamic_energy_bias = dynamic_energy_bias

        self.dynamic_lift = build_mlp(
            input_dim=3,
            hidden_dim=dynamic_hidden_dim,
            output_dim=dynamic_hidden_dim,
            num_layers=2,
            dropout=projector_dropout,
        )
        self.dynamic_update = build_mlp(
            input_dim=dynamic_hidden_dim * 3,
            hidden_dim=dynamic_hidden_dim,
            output_dim=dynamic_hidden_dim,
            num_layers=2,
            dropout=projector_dropout,
        )
        self.dynamic_decode = build_mlp(
            input_dim=dynamic_hidden_dim,
            hidden_dim=dynamic_hidden_dim,
            output_dim=1,
            num_layers=2,
            dropout=projector_dropout,
        )

    def build_soft_code_adjacency(self, src_assign, dst_assign, symmetric=True):
        if src_assign.numel() == 0 or dst_assign.numel() == 0:
            adjacency = self.code_energies.new_zeros((self.num_codes, self.num_codes))
        else:
            adjacency = torch.einsum('bi,bj->ij', src_assign, dst_assign)
        if symmetric:
            adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        if self.dynamic_self_loop > 0:
            adjacency = adjacency + torch.eye(self.num_codes, device=adjacency.device) * float(self.dynamic_self_loop)
        return adjacency

    def build_energy_biased_transition_kernel(self, adjacency):
        energy_src = self.code_energies.view(-1, 1)
        energy_dst = self.code_energies.view(1, -1)
        uphill_penalty = torch.relu(energy_dst - energy_src)
        rates = adjacency * torch.exp(-uphill_penalty / max(float(self.dynamic_temperature), 1e-6))
        row_sum = rates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return rates / row_sum

    def aggregate_soft_occupancy(self, soft_assign, row_groups):
        if len(row_groups) == 0:
            return soft_assign.new_zeros((0, self.num_codes))
        occupancies = []
        for rows in row_groups:
            if rows.numel() == 0:
                occupancies.append(soft_assign.new_zeros((self.num_codes,)))
            else:
                occupancies.append(soft_assign.index_select(0, rows).mean(dim=0))
        return torch.stack(occupancies, dim=0)

    def predict_next_occupancy(self, curr_occupancy, adjacency):
        if curr_occupancy.numel() == 0:
            empty = curr_occupancy.new_zeros((0, self.num_codes))
            return empty, {
                'adjacency': adjacency,
                'transition_kernel': adjacency,
                'pushforward_occupancy': empty,
            }

        transition_kernel = self.build_energy_biased_transition_kernel(adjacency)
        pushforward = torch.matmul(curr_occupancy, transition_kernel)
        energy_feature = (-self.code_energies).unsqueeze(0).expand_as(curr_occupancy)
        lift_input = torch.stack([curr_occupancy, pushforward, energy_feature], dim=-1)
        hidden = self.dynamic_lift(lift_input)

        incoming_kernel = transition_kernel.transpose(0, 1).unsqueeze(0)
        for _ in range(max(int(self.dynamic_num_steps), 1)):
            incoming = torch.matmul(incoming_kernel, hidden)
            update = self.dynamic_update(torch.cat([hidden, incoming, incoming - hidden], dim=-1))
            hidden = hidden + update

        logits = self.dynamic_decode(hidden).squeeze(-1)
        logits = logits + float(self.dynamic_residual_weight) * torch.log(pushforward.clamp_min(1e-8))
        logits = logits + float(self.dynamic_energy_bias) * energy_feature
        pred_next = F.softmax(logits, dim=-1)
        return pred_next, {
            'adjacency': adjacency,
            'transition_kernel': transition_kernel,
            'pushforward_occupancy': pushforward,
        }


class DynamicLocalResidualPESLALandscapeModel(DynamicPESLALandscapeModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        code_dim,
        num_codes,
        projector_layers=2,
        transition_hidden_dim=None,
        recon_hidden_dim=None,
        decoder_layers=2,
        projector_dropout=0.0,
        commitment_weight=0.25,
        codebook_weight=1.0,
        use_reconstruction=True,
        dynamic_hidden_dim=128,
        dynamic_num_steps=2,
        dynamic_temperature=1.0,
        dynamic_self_loop=0.1,
        dynamic_residual_weight=1.0,
        dynamic_energy_bias=1.0,
        local_hidden_dim=None,
        local_layers=2,
        local_scale=1.0,
        use_quality_head=False,
        quality_output_dim=4,
        use_basin_classifier=False,
        basin_num_classes=3,
        use_quality_aware_assignment=False,
        quality_assign_dim=None,
        quality_assign_hidden_dim=None,
        quality_assign_weight=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            projector_layers=projector_layers,
            transition_hidden_dim=transition_hidden_dim,
            recon_hidden_dim=recon_hidden_dim,
            decoder_layers=decoder_layers,
            projector_dropout=projector_dropout,
            commitment_weight=commitment_weight,
            codebook_weight=codebook_weight,
            use_reconstruction=use_reconstruction,
            dynamic_hidden_dim=dynamic_hidden_dim,
            dynamic_num_steps=dynamic_num_steps,
            dynamic_temperature=dynamic_temperature,
            dynamic_self_loop=dynamic_self_loop,
            dynamic_residual_weight=dynamic_residual_weight,
            dynamic_energy_bias=dynamic_energy_bias,
            use_quality_head=use_quality_head,
            quality_output_dim=quality_output_dim,
            use_basin_classifier=use_basin_classifier,
            basin_num_classes=basin_num_classes,
            use_quality_aware_assignment=use_quality_aware_assignment,
            quality_assign_dim=quality_assign_dim,
            quality_assign_hidden_dim=quality_assign_hidden_dim,
            quality_assign_weight=quality_assign_weight,
        )
        local_hidden_dim = local_hidden_dim or hidden_dim
        self.local_scale = float(local_scale)
        self.local_energy_head = build_mlp(
            input_dim=code_dim * 2 + 1,
            hidden_dim=local_hidden_dim,
            output_dim=1,
            num_layers=local_layers,
            dropout=projector_dropout,
        )
        self.reset_local_energy_head()

    def reset_local_energy_head(self):
        last_layer = None
        if isinstance(self.local_energy_head, nn.Sequential):
            for module in reversed(self.local_energy_head):
                if isinstance(module, nn.Linear):
                    last_layer = module
                    break
        elif isinstance(self.local_energy_head, nn.Linear):
            last_layer = self.local_energy_head
        if last_layer is not None:
            nn.init.normal_(last_layer.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(last_layer.bias)

    def compute_local_energy(self, z_e, soft_assign):
        code_context = torch.matmul(soft_assign, self.quantizer.codebook)
        code_energy_context = torch.matmul(soft_assign, self.code_energies.unsqueeze(-1))
        local_input = torch.cat([z_e, code_context, code_energy_context], dim=-1)
        return self.local_scale * self.local_energy_head(local_input).squeeze(-1)

    def guidance_energy(self, x, tau=1.0, mode='local'):
        z_e = self.projector(x)
        soft_assign = self.compute_soft_assign_from_latent(z_e, tau=tau)
        code_energy = torch.sum(soft_assign * self.code_energies.unsqueeze(0), dim=-1)
        value_energy = None
        if mode in {'value', 'value_local', 'value_proto', 'value_proto_local'}:
            value_energy = self.compute_value_energy(soft_assign)
        prototype_energy = None
        if mode in {'prototype', 'value_proto', 'value_proto_local'}:
            prototype_energy = self.compute_prototype_attraction_energy(z_e, soft_assign)
        local_energy = self.compute_local_energy(z_e, soft_assign)
        if mode == 'code':
            return code_energy
        if mode == 'value':
            return value_energy
        if mode == 'prototype':
            return prototype_energy
        if mode == 'value_proto':
            return value_energy + prototype_energy
        if mode == 'value_proto_local':
            return value_energy + prototype_energy + local_energy
        if mode == 'value_local':
            return value_energy + local_energy
        if mode == 'total':
            return code_energy.detach() + local_energy
        if mode == 'local':
            return local_energy
        raise ValueError(f'Unsupported guidance energy mode: {mode}')

    def forward(self, x):
        outputs = super().forward(x)
        local_energy = self.compute_local_energy(outputs['z_e'], outputs['soft_assign'])
        total_soft_state_energy = outputs['soft_state_energy'].detach() + local_energy
        transition_probs = F.softmax(outputs['transition_logits'], dim=-1)
        transition_entropy = -torch.sum(
            transition_probs * torch.log(transition_probs.clamp_min(1e-8)),
            dim=-1,
        )
        outputs.update({
            'local_energy': local_energy,
            'total_soft_state_energy': total_soft_state_energy,
            'force_energy': local_energy,
            'transition_probs': transition_probs,
            'transition_entropy': transition_entropy,
        })
        return outputs


def get_landscape_model_type(config):
    model_cfg = getattr(config, 'model', config)
    return str(getattr(model_cfg, 'type', 'pesla_lite')).lower()


def build_landscape_model(config, input_dim):
    model_cfg = getattr(config, 'model', config)
    common_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=model_cfg.hidden_dim,
        code_dim=model_cfg.code_dim,
        num_codes=model_cfg.num_codes,
        projector_layers=getattr(model_cfg, 'projector_layers', 2),
        transition_hidden_dim=getattr(model_cfg, 'transition_hidden_dim', None),
        recon_hidden_dim=getattr(model_cfg, 'recon_hidden_dim', None),
        decoder_layers=getattr(model_cfg, 'decoder_layers', 2),
        projector_dropout=getattr(model_cfg, 'projector_dropout', 0.0),
        commitment_weight=getattr(model_cfg, 'commitment_weight', 0.25),
        codebook_weight=getattr(model_cfg, 'codebook_weight', 1.0),
        use_reconstruction=getattr(model_cfg, 'use_reconstruction', True),
        use_quality_head=getattr(model_cfg, 'use_quality_head', False),
        quality_output_dim=getattr(model_cfg, 'quality_output_dim', 4),
        use_basin_classifier=getattr(model_cfg, 'use_basin_classifier', False),
        basin_num_classes=getattr(model_cfg, 'basin_num_classes', 3),
        use_quality_aware_assignment=getattr(model_cfg, 'use_quality_aware_assignment', False),
        quality_assign_dim=getattr(model_cfg, 'quality_assign_dim', None),
        quality_assign_hidden_dim=getattr(model_cfg, 'quality_assign_hidden_dim', None),
        quality_assign_weight=getattr(model_cfg, 'quality_assign_weight', 0.0),
    )
    model_type = get_landscape_model_type(config)
    if model_type in {'pesla_lite', 'pesla-lite', 'base'}:
        return PESLALiteLandscapeModel(**common_kwargs)
    if model_type in {'dual_codebook_pesla_lite', 'dual-codebook-pesla-lite', 'dual_codebook'}:
        return DualCodebookPESLALandscapeModel(
            **common_kwargs,
            quality_code_dim=getattr(model_cfg, 'quality_code_dim', None),
            quality_num_codes=getattr(model_cfg, 'quality_num_codes', None),
            quality_projector_layers=getattr(model_cfg, 'quality_projector_layers', None),
            quality_hidden_dim=getattr(model_cfg, 'quality_hidden_dim', None),
        )
    if model_type in {'local_residual_pesla_lite', 'local-residual-pesla-lite', 'local_residual'}:
        return LocalResidualPESLALandscapeModel(
            **common_kwargs,
            local_hidden_dim=getattr(model_cfg, 'local_hidden_dim', None),
            local_layers=getattr(model_cfg, 'local_layers', 2),
            local_scale=getattr(model_cfg, 'local_scale', 1.0),
        )
    if model_type in {'dynamic_pesla_lite', 'dynamic-pesla-lite', 'dynamic'}:
        return DynamicPESLALandscapeModel(
            **common_kwargs,
            dynamic_hidden_dim=getattr(model_cfg, 'dynamic_hidden_dim', 128),
            dynamic_num_steps=getattr(model_cfg, 'dynamic_num_steps', 2),
            dynamic_temperature=getattr(model_cfg, 'dynamic_temperature', 1.0),
            dynamic_self_loop=getattr(model_cfg, 'dynamic_self_loop', 0.1),
            dynamic_residual_weight=getattr(model_cfg, 'dynamic_residual_weight', 1.0),
            dynamic_energy_bias=getattr(model_cfg, 'dynamic_energy_bias', 1.0),
        )
    if model_type in {
        'dynamic_local_residual_pesla_lite',
        'dynamic-local-residual-pesla-lite',
        'dynamic_local_residual',
        'transition_local_residual',
    }:
        return DynamicLocalResidualPESLALandscapeModel(
            **common_kwargs,
            dynamic_hidden_dim=getattr(model_cfg, 'dynamic_hidden_dim', 128),
            dynamic_num_steps=getattr(model_cfg, 'dynamic_num_steps', 2),
            dynamic_temperature=getattr(model_cfg, 'dynamic_temperature', 1.0),
            dynamic_self_loop=getattr(model_cfg, 'dynamic_self_loop', 0.1),
            dynamic_residual_weight=getattr(model_cfg, 'dynamic_residual_weight', 1.0),
            dynamic_energy_bias=getattr(model_cfg, 'dynamic_energy_bias', 1.0),
            local_hidden_dim=getattr(model_cfg, 'local_hidden_dim', None),
            local_layers=getattr(model_cfg, 'local_layers', 2),
            local_scale=getattr(model_cfg, 'local_scale', 1.0),
        )
    raise ValueError(f'Unsupported landscape model type: {model_type}')

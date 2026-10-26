import math

import torch
from torch import nn
from torch.nn import functional as F


class SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, value):
        if self.training and value.numel() // value.shape[1] == 1:
            return F.batch_norm(value, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(value)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, value):
        if self.training and value.shape[0] == 1:
            return F.batch_norm(value, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(value)


def timestep_embedding(timesteps, dimension):
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(result, (0, dimension - result.shape[-1]))


class VariationalInformationBottleneck(nn.Module):
    def __init__(self, latent_dim, token_grid, pred_len, sigma_min, sigma_max,
                 spatial_channels, spatial_grid=4):
        super().__init__()
        self.token_grid, self.pred_len = token_grid, pred_len
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.spatial_channels, self.spatial_grid = spatial_channels, spatial_grid
        if min(spatial_channels, spatial_grid) < 1:
            raise ValueError('VIB spatial channels and grid must be positive')
        self.encoder = nn.Sequential(nn.Linear(token_grid ** 2, latent_dim),
                                     SafeBatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(0.1))
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.scale = nn.Linear(latent_dim, latent_dim)
        self.aux_semantics = nn.Linear(latent_dim, 7)
        self.aux_waypoints = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                          nn.ReLU(), nn.Linear(latent_dim, pred_len * 2))
        self.spatial_projection = nn.Linear(latent_dim, spatial_channels * spatial_grid ** 2)
        self.spatial_refinement = nn.Conv2d(spatial_channels, spatial_channels, 3, padding=1)

    def forward(self, bev, compute_auxiliary=None, spatial_size=None):
        if compute_auxiliary is None:
            compute_auxiliary = self.training
        pooled = F.adaptive_avg_pool2d(bev.mean(1, keepdim=True), (self.token_grid, self.token_grid))
        encoded = self.encoder(pooled.flatten(1))
        mean = self.mean(encoded)
        sigma = F.softplus(self.scale(encoded).float()).clamp(self.sigma_min, self.sigma_max)
        latent = mean.float()
        if self.training:
            latent = latent + sigma * torch.randn_like(sigma)
        loss_kl = 0.5 * (mean.float().square() + sigma.square() - 1.0
                         - 2.0 * sigma.log()).sum(-1).mean()
        latent = latent.to(mean.dtype)
        spatial = self.spatial_projection(latent).reshape(
            bev.shape[0], self.spatial_channels, self.spatial_grid, self.spatial_grid)
        spatial = F.interpolate(spatial, spatial_size or bev.shape[-2:],
                                mode='bilinear', align_corners=False)
        spatial = self.spatial_refinement(F.silu(spatial))
        result = {'semantic_latent': latent, 'spatial_bev': spatial,
                  'latent': latent, 'loss_kl': loss_kl}
        if compute_auxiliary:
            result['aux_semantics'] = self.aux_semantics(latent)
            result['aux_waypoints'] = self.aux_waypoints(latent).reshape(bev.shape[0], self.pred_len, 2)
        return result


class SpatioTemporalGatedModulation(nn.Module):
    def __init__(self, channels, bev_channels, condition_dim, heads):
        super().__init__()
        self.norm = SafeBatchNorm2d(channels)
        self.modulation = nn.Sequential(nn.Linear(condition_dim, channels * 2), nn.SiLU(),
                                        nn.Linear(channels * 2, channels * 3))
        self.spatial_attention = nn.Conv2d(2, 1, 1)
        self.history_projection = nn.Conv2d(bev_channels, channels, 1)
        self.temporal_position = nn.Linear(1, channels)
        self.temporal_attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, feature, condition, frames):
        batch, channels, height, width = feature.shape
        gamma, beta, gate = self.modulation(condition).chunk(3, dim=1)
        modulated = (1.0 + gamma[:, :, None, None]) * self.norm(feature)
        modulated = (modulated + beta[:, :, None, None]) * gate.sigmoid()[:, :, None, None]
        pooled = torch.cat((modulated.mean(1, keepdim=True), modulated.amax(1, keepdim=True)), 1)
        spatial = modulated * self.spatial_attention(pooled).sigmoid()
        time = frames.shape[1]
        context = F.interpolate(frames.flatten(0, 1), (height, width), mode='bilinear', align_corners=False)
        context = self.history_projection(context).reshape(batch, time, channels, height, width)
        context = context.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, channels)
        ages = torch.arange(1 - time, 1, device=feature.device, dtype=feature.dtype).reshape(time, 1)
        context = context + self.temporal_position(ages)[None]
        query = spatial.flatten(2).transpose(1, 2).reshape(-1, 1, channels)
        attended, _ = self.temporal_attention(query, context, context, need_weights=False)
        attended = attended.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return feature + F.silu(spatial + attended)


class ObjectLayoutEncoder(nn.Module):
    def __init__(self, config, dimension):
        super().__init__()
        self.height = float(config.lidar_resolution_height)
        self.width = float(config.lidar_resolution_width)
        self.box_embedding = nn.Sequential(nn.Linear(8, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.class_embedding = nn.Embedding(2, dimension)
        self.null_token = nn.Parameter(torch.zeros(1, 1, dimension))

    def forward(self, labels):
        valid = torch.isfinite(labels).all(-1) & (labels[..., 2] > 0) & (labels[..., 3] > 0)
        boxes = torch.where(valid[..., None], labels, torch.zeros_like(labels))
        geometry = torch.stack((boxes[..., 0] / self.width, boxes[..., 1] / self.height,
                                boxes[..., 2] / self.width, boxes[..., 3] / self.height,
                                boxes[..., 4].sin(), boxes[..., 4].cos(),
                                boxes[..., 5] / 30.0, boxes[..., 6]), -1)
        tokens = self.box_embedding(geometry) + self.class_embedding(valid.long())
        tokens = torch.cat((self.null_token.expand(labels.shape[0], -1, -1), tokens), 1)
        padding = torch.cat((torch.zeros_like(valid[:, :1]), ~valid), 1)
        return tokens, padding


class LayoutCrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads, token_grid):
        super().__init__()
        self.token_grid = token_grid
        self.layout_projection = nn.Linear(context_dim, channels)
        self.query_position = nn.Linear(2, channels)
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, feature, layout):
        tokens, padding = layout
        context = self.context_norm(self.layout_projection(tokens))
        grid = min(self.token_grid, *feature.shape[-2:])
        query = F.adaptive_avg_pool2d(feature, (grid, grid)).flatten(2).transpose(1, 2)
        axis = torch.linspace(-1.0, 1.0, grid, device=feature.device, dtype=feature.dtype)
        rows, columns = torch.meshgrid(axis, axis, indexing='ij')
        positions = torch.stack((rows, columns), -1).reshape(grid * grid, 2)
        query = query + self.query_position(positions)[None]
        attended, _ = self.attention(self.query_norm(query), context, context,
                                     key_padding_mask=padding, need_weights=False)
        attended = attended.transpose(1, 2).reshape(feature.shape[0], -1, grid, grid)
        return feature + F.interpolate(attended, feature.shape[-2:], mode='bilinear', align_corners=False)


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    SafeBatchNorm2d(channels), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1), SafeBatchNorm2d(channels))

    def forward(self, value):
        return F.silu(value + self.layers(value))


class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, bev_channels, width, heads, grid, stgsm):
        super().__init__()
        widths = [width * 2 ** level for level in range(4)]
        self.input = nn.Conv2d(bev_channels, width, 3, padding=1)
        self.encoders = nn.ModuleList([_ResidualBlock(c) for c in widths])
        self.downsamples = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                         for a, b in zip(widths[:-1], widths[1:])])
        self.time_biases = nn.ModuleList([nn.Linear(width, c) for c in widths])
        self.modulations = nn.ModuleList([SpatioTemporalGatedModulation(c, bev_channels, width, heads)
                                          for c in widths]) if stgsm else None
        self.alignments = nn.ModuleList([LayoutCrossAttention(c, width, heads, grid) for c in widths])
        self.upsamples = nn.ModuleList([nn.Conv2d(widths[i + 1] + widths[i], widths[i], 3, padding=1)
                                       for i in range(3)])
        self.decoders = nn.ModuleList([_ResidualBlock(c) for c in widths[:-1]])
        self.decoder_modulations = nn.ModuleList([
            SpatioTemporalGatedModulation(c, bev_channels, width, heads)
            for c in widths[:-1]]) if stgsm else None
        self.output = nn.Conv2d(width, bev_channels, 3, padding=1)

    def forward(self, noisy, condition, frames, layout):
        value = self.input(noisy)
        skips = []
        for level, encoder in enumerate(self.encoders):
            if level:
                value = self.downsamples[level - 1](value)
            value = encoder(value + self.time_biases[level](condition)[:, :, None, None])
            if self.modulations is not None:
                value = self.modulations[level](value, condition, frames)
            value = self.alignments[level](value, layout)
            skips.append(value)
        for level in (2, 1, 0):
            value = F.interpolate(value, skips[level].shape[-2:], mode='bilinear', align_corners=False)
            value = self.decoders[level](self.upsamples[level](torch.cat((value, skips[level]), 1)))
            if self.decoder_modulations is not None:
                value = self.decoder_modulations[level](value, condition, frames)
        return self.output(value)


class BEVEnhancement(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.channels, self.stage = channels, config.bev_stage
        if self.stage not in ('representation', 'planning', 'joint'):
            raise ValueError('Invalid BEV stage')
        self.inference_mode = config.bev_diffusion_inference
        if self.inference_mode not in ('single_step', 'bypass'):
            raise ValueError('bev_diffusion_inference must be single_step or bypass')
        self.working_resolution = int(config.bev_working_resolution)
        self.diffusion_steps = int(config.bev_diffusion_steps)
        self.inference_t = getattr(config, 'bev_diffusion_inference_t', 0)
        self.input_state = getattr(config, 'bev_diffusion_input_state', 'observed_as_xt')
        if isinstance(self.inference_t, bool) or not isinstance(self.inference_t, int) or not 0 <= self.inference_t < self.diffusion_steps:
            raise ValueError('bev_diffusion_inference_t must index the diffusion schedule')
        if self.input_state != 'observed_as_xt':
            raise ValueError('Only the explicit observed_as_xt input assumption is implemented')
        self.semantic_conditioning = config.bev_semantic_conditioning
        self.layout_conditioning = config.bev_layout_conditioning
        self.clean_residual = config.bev_clean_residual
        if self.inference_mode == 'single_step' and (not config.bev_diffusion_enabled or not self.clean_residual):
            raise ValueError('single_step requires diffusion and feature enhancement; select bypass explicitly for the legacy ablation')
        width, heads = int(config.bev_unet_channels), int(config.bev_attention_heads)
        latent_dim, grid = int(config.bev_latent_dim), int(config.bev_token_grid)
        self.latent_dim = latent_dim
        if min(channels, width, heads, latent_dim, grid, self.diffusion_steps, self.working_resolution) < 1 or width % heads:
            raise ValueError('Invalid BEV dimensions or attention heads')
        if not 0 < config.bev_diffusion_beta_start <= config.bev_diffusion_beta_end < 1:
            raise ValueError('Invalid diffusion noise schedule')
        if not 0 < config.bev_sigma_min < config.bev_sigma_max:
            raise ValueError('Invalid VIB scale limits')
        self.vib = VariationalInformationBottleneck(latent_dim, grid, config.pred_len,
                   config.bev_sigma_min, config.bev_sigma_max, channels,
                   getattr(config, 'bev_vib_spatial_grid', 4)) if config.bev_vib_enabled else None
        self.deterministic_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                                   nn.Linear(channels, latent_dim), nn.SiLU()) if self.vib is None else None
        self.time_dimension = width
        self.condition_projection = nn.Sequential(nn.Linear(latent_dim + width, width * 2),
                                                   nn.SiLU(), nn.Linear(width * 2, width))
        self.layout_encoder = ObjectLayoutEncoder(config, width)
        self.clean_modulation = SpatioTemporalGatedModulation(width, channels, width, heads) if config.bev_stgsm_enabled else None
        self.clean_alignment = LayoutCrossAttention(width, width, heads, grid)
        self.clean_projection = nn.Conv2d(channels, width, 1)
        self.clean_output = nn.Conv2d(width, channels, 1)
        self.planning_projection = nn.Linear(channels, 512)
        self.fusion_gain = nn.Parameter(torch.tensor(0.1))
        self.denoiser = ConditionalDiffusionUNet(channels, width, heads, grid,
                         config.bev_stgsm_enabled) if config.bev_diffusion_enabled else None
        betas = torch.linspace(config.bev_diffusion_beta_start, config.bev_diffusion_beta_end, self.diffusion_steps)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', torch.cumprod(1.0 - betas, dim=0))
        self._representation_frozen = False
        if self.stage == 'representation':
            self.planning_projection.requires_grad_(False)
        if not self.layout_conditioning:
            self.layout_encoder.box_embedding.requires_grad_(False)
            self.layout_encoder.class_embedding.requires_grad_(False)
        if not self.clean_residual:
            for module in (self.clean_modulation, self.clean_alignment, self.clean_projection,
                           self.clean_output, self.planning_projection):
                if module is not None:
                    module.requires_grad_(False)
            self.fusion_gain.requires_grad_(False)
        if self.stage == 'planning':
            self.freeze_representation()
            if not self.clean_residual:
                self.requires_grad_(False)

    def freeze_representation(self):
        self._representation_frozen = True
        for module in (self.vib, self.deterministic_encoder, self.denoiser):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._representation_frozen:
            for module in (self.vib, self.deterministic_encoder, self.denoiser):
                if module is not None:
                    module.eval()
        return self

    def _condition(self, latent, timesteps):
        if not self.semantic_conditioning:
            latent = torch.zeros_like(latent)
        time = timestep_embedding(timesteps, self.time_dimension).to(latent.dtype)
        return self.condition_projection(torch.cat((latent, time), 1))

    def q_sample(self, clean, timesteps, noise):
        alpha_bar = self.alpha_bars[timesteps].to(clean.dtype)[:, None, None, None]
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def estimate_clean_features(self, observed, latent, past, layout, timesteps=None):
        """One epsilon prediction, interpreted as an x0 estimate at an explicit t.

        Treating the observed VIB map as x_t is an implementation assumption;
        the method does not specify the input noise level or its timestep. Even
        t=0 has alpha_bar=1-beta_start, not alpha_bar=1. No noise is added here.
        """
        if self.denoiser is None:
            raise ValueError('Feature estimation requires a denoising network')
        if self.alpha_bars.dtype not in (torch.float32, torch.float64):
            raise ValueError('Single-step diffusion requires a float32 or float64 schedule; '
                             'keep the module in float32 and use autocast for low-precision execution')
        if timesteps is None:
            timesteps = torch.full((observed.shape[0],), self.inference_t,
                                   device=observed.device, dtype=torch.long)
        if timesteps.shape != (observed.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError('timesteps must be a B-vector of integer schedule indices')
        if bool(((timesteps < 0) | (timesteps >= self.diffusion_steps)).any()):
            raise ValueError('timesteps must index the diffusion schedule')
        alpha_bar = self.alpha_bars[timesteps].float()[:, None, None, None]
        if not bool((torch.isfinite(alpha_bar) & (alpha_bar > 0) & (alpha_bar <= 1)).all()):
            raise ValueError('Feature estimation requires finite alpha_bar in (0,1]')
        frames = torch.cat((past, observed[:, None]), 1)
        epsilon = self.denoiser(observed, self._condition(latent, timesteps), frames, layout)
        if epsilon.shape != observed.shape:
            raise ValueError('Denoiser epsilon prediction must match the spatial BEV shape')
        estimate = (observed.float() - (1.0 - alpha_bar).sqrt() * epsilon.float()) / alpha_bar.sqrt()
        estimate = estimate.to(observed.dtype)
        if not bool(torch.isfinite(estimate).all()):
            raise FloatingPointError('Single-step diffusion produced nonfinite enhanced features')
        return estimate

    def _noise_loss(self, working, latent, past, layout, timesteps, noise):
        noisy = self.q_sample(working, timesteps, noise)
        frames = torch.cat((past, noisy[:, None]), 1)
        prediction = self.denoiser(noisy, self._condition(latent, timesteps), frames, layout)
        return F.mse_loss(prediction.float(), noise.float())

    def forward(self, bev, history=None, layout=None, validate_diffusion=False):
        if bev.ndim != 4 or bev.shape[1] != self.channels:
            raise ValueError('bev must have shape B,C,H,W with configured channels')
        if layout is None or layout.ndim != 3 or layout.shape[0] != bev.shape[0] or layout.shape[-1] != 7 or layout.shape[1] < 1:
            raise ValueError('Object layout must be padded B,M,7 detection labels')
        size = tuple(min(self.working_resolution, extent) for extent in bev.shape[-2:])
        past = bev.new_empty((bev.shape[0], 0, self.channels, *size))
        if history is not None:
            if history.ndim != 5 or history.shape[0] != bev.shape[0] or history.shape[2:] != bev.shape[1:]:
                raise ValueError('history must have shape B,T,C,H,W matching BEV')
        if history is not None and history.shape[1]:
            past = F.adaptive_avg_pool2d(history.flatten(0, 1), size).reshape(
                bev.shape[0], history.shape[1], self.channels, *size)
        compute_auxiliary = (self.training and self.stage in ('representation', 'joint')) or validate_diffusion
        if self.vib is not None:
            result = self.vib(bev, compute_auxiliary=compute_auxiliary, spatial_size=size)
        else:
            latent = self.deterministic_encoder(bev)
            result = {'semantic_latent': latent, 'latent': latent,
                      'spatial_bev': F.adaptive_avg_pool2d(bev, size), 'loss_kl': bev.new_zeros(())}
        working = result['spatial_bev']
        if self.layout_conditioning:
            tokens = self.layout_encoder(layout.detach())
        else:
            tokens = (self.layout_encoder.null_token.expand(bev.shape[0], -1, -1),
                      torch.zeros((bev.shape[0], 1), device=bev.device, dtype=torch.bool))
        clean_time = torch.zeros(bev.shape[0], device=bev.device, dtype=torch.long)
        result['enhancement_mode'] = self.inference_mode
        result['feature_source'] = 'single_step_epsilon_to_x0' if self.inference_mode == 'single_step' else 'legacy_clean_bypass'
        result['spatial_source'] = 'vib_latent_decoder' if self.vib is not None else 'no_vib_ablation'
        result['inference_t'] = self.inference_t if self.inference_mode == 'single_step' else None
        result['inference_input_state'] = self.input_state if self.inference_mode == 'single_step' else None
        if self.inference_mode == 'single_step':
            enhanced = self.estimate_clean_features(working, result['semantic_latent'], past, tokens)
            result['features'] = F.interpolate(enhanced, bev.shape[-2:], mode='bilinear', align_corners=False)
            result['fused_delta'] = self.planning_projection(result['features'].mean(dim=(-1, -2)))
        elif self.clean_residual:
            clean_working = F.adaptive_avg_pool2d(bev, size)
            condition = self._condition(result['latent'], clean_time)
            fused = self.clean_projection(clean_working) + condition[:, :, None, None]
            if self.clean_modulation is not None:
                fused = self.clean_modulation(fused, condition, torch.cat((past, clean_working[:, None]), 1))
            residual = self.fusion_gain * self.clean_output(self.clean_alignment(fused, tokens))
            result['features'] = bev + F.interpolate(residual, bev.shape[-2:], mode='bilinear', align_corners=False)
            result['fused_delta'] = self.planning_projection(result['features'].mean(dim=(-1, -2)))
        else:
            result['features'] = bev
            result['fused_delta'] = bev.new_zeros((bev.shape[0], 512))
        result['loss_reconstruction'] = bev.new_zeros(())
        if self.denoiser is not None and self.training and self.stage in ('representation', 'joint'):
            timesteps = torch.randint(self.diffusion_steps, (bev.shape[0],), device=bev.device)
            result['loss_reconstruction'] = self._noise_loss(
                working, result['latent'], past, tokens, timesteps, torch.randn_like(working))
        if self.denoiser is not None and validate_diffusion:
            if self.training:
                raise ValueError('Fixed diffusion validation requires eval mode')
            generator = torch.Generator(device=bev.device).manual_seed(1729)
            noise = torch.randn((1, *working.shape[1:]), device=bev.device, dtype=working.dtype,
                                generator=generator).expand_as(working)
            terms = []
            for step in sorted({0, self.diffusion_steps // 2, self.diffusion_steps - 1}):
                timesteps = clean_time + step
                terms.append(self._noise_loss(working, result['latent'], past, tokens, timesteps, noise))
            result['metric_diffusion_mse'] = torch.stack(terms).mean()
            if self.stage in ('representation', 'joint'):
                result['loss_reconstruction'] = result['metric_diffusion_mse']
        return result

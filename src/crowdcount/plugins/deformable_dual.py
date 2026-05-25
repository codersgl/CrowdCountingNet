"""Dual-stream deformable attention fusion for DSGCNet."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _normalise_density(density: torch.Tensor) -> torch.Tensor:
    batch_size = density.shape[0]
    flat = density.reshape(batch_size, -1)
    d_min = flat.min(dim=1, keepdim=True).values.view(batch_size, 1, 1, 1)
    d_max = flat.max(dim=1, keepdim=True).values.view(batch_size, 1, 1, 1)
    return (density - d_min) / (d_max - d_min + 1e-6)


def _make_base_offsets(num_points: int) -> torch.Tensor:
    if num_points == 4:
        offsets = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
    elif num_points == 8:
        offsets = [
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
        ]
    else:
        angles = torch.linspace(0.0, 2.0 * math.pi, num_points + 1)[:-1]
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return torch.tensor(offsets, dtype=torch.float32)


def _sample_map(
    feature_map: torch.Tensor,
    sample_coords: torch.Tensor,
) -> torch.Tensor:
    batch_size, channels, _height, _width = feature_map.shape
    num_nodes = sample_coords.shape[1]
    num_points = sample_coords.shape[2]
    flat_coords = sample_coords.reshape(batch_size, num_nodes * num_points, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        flat_coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).reshape(
        batch_size, num_nodes, num_points, channels
    )


class GuidedDeformableAttention(nn.Module):
    """Residual deformable attention over a fixed local base pattern.

    The learned offsets start at zero, so the initial sampling pattern is the
    fixed local base neighbourhood. A small residual gate keeps the branch close
    to identity at the beginning of training.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_points: int = 4,
        num_heads: int = 4,
        max_offset: float = 4.0,
        use_density_guidance: bool = False,
        density_offset_rho: float = 0.5,
        density_gamma_init: float = 0.5,
        distance_lambda_init: float = 1.0,
        dropout: float = 0.1,
        residual_gate_init: float = 0.001,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
            )
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if max_offset < 0.0:
            raise ValueError(f"max_offset must be non-negative, got {max_offset}")

        self.in_channels = int(in_channels)
        self.num_points = int(num_points)
        self.num_heads = int(num_heads)
        self.head_dim = self.in_channels // self.num_heads
        self.max_offset = float(max_offset)
        self.use_density_guidance = bool(use_density_guidance)
        self.density_offset_rho = float(density_offset_rho)
        self.debug = bool(debug)

        offset_in_channels = in_channels + (1 if use_density_guidance else 0)
        self.offset_pred = nn.Sequential(
            nn.Conv2d(offset_in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 2 * num_points, 1),
        )
        nn.init.zeros_(self.offset_pred[-1].weight)
        nn.init.zeros_(self.offset_pred[-1].bias)

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.density_gamma = nn.Parameter(torch.tensor(float(density_gamma_init)))
        self.distance_lambda = nn.Parameter(torch.tensor(float(distance_lambda_init)))
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self.attn_drop = nn.Dropout(dropout)
        self.register_buffer("base_offsets", _make_base_offsets(num_points))

    def _prepare_density(
        self,
        density: torch.Tensor | None,
        feature_maps: torch.Tensor,
    ) -> torch.Tensor | None:
        if density is None:
            return None
        density = density.to(device=feature_maps.device, dtype=feature_maps.dtype)
        if density.shape[-2:] != feature_maps.shape[-2:]:
            density = F.interpolate(
                density,
                size=feature_maps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return _normalise_density(density)

    def _build_sample_coords(
        self,
        feature_maps: torch.Tensor,
        density_norm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _channels, height, width = feature_maps.shape
        if self.use_density_guidance:
            if density_norm is None:
                density_norm = feature_maps.new_zeros(batch_size, 1, height, width)
            offset_input = torch.cat([feature_maps, density_norm], dim=1)
        else:
            offset_input = feature_maps

        raw_offsets = self.offset_pred(offset_input)
        residual_offsets = raw_offsets.reshape(
            batch_size, self.num_points, 2, height, width
        ).permute(0, 3, 4, 1, 2)
        residual_offsets = residual_offsets.tanh() * self.max_offset
        if self.use_density_guidance and density_norm is not None:
            scale = 1.0 + self.density_offset_rho * density_norm.permute(0, 2, 3, 1)
            residual_offsets = residual_offsets * scale.unsqueeze(3)

        base_offsets = self.base_offsets.to(
            device=feature_maps.device, dtype=feature_maps.dtype
        ).view(1, 1, 1, self.num_points, 2)
        total_offsets = base_offsets + residual_offsets

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=feature_maps.device),
            torch.linspace(-1.0, 1.0, width, device=feature_maps.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).to(feature_maps.dtype)
        base_grid = base_grid.view(1, height, width, 1, 2)
        normaliser = feature_maps.new_tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
            ]
        ).view(1, 1, 1, 1, 2)
        sample_coords = (base_grid + total_offsets * normaliser).clamp(-1.0, 1.0)
        return (
            sample_coords.reshape(batch_size, height * width, self.num_points, 2),
            residual_offsets,
            total_offsets,
        )

    def forward(
        self,
        feature_maps: torch.Tensor,
        density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if feature_maps.dim() != 4:
            raise ValueError(
                f"feature_maps must be 4D, got {tuple(feature_maps.shape)}"
            )
        batch_size, channels, height, width = feature_maps.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}"
            )

        density_norm = self._prepare_density(density, feature_maps)
        sample_coords, residual_offsets, total_offsets = self._build_sample_coords(
            feature_maps, density_norm
        )
        neighbor_feats = _sample_map(feature_maps, sample_coords)
        num_nodes = height * width

        x_flat = feature_maps.permute(0, 2, 3, 1).reshape(batch_size, num_nodes, channels)
        query = self.q_proj(x_flat)
        key = self.k_proj(neighbor_feats.reshape(batch_size * num_nodes, self.num_points, channels))
        value = self.v_proj(neighbor_feats.reshape(batch_size * num_nodes, self.num_points, channels))
        key = key.reshape(batch_size, num_nodes, self.num_points, channels)
        value = value.reshape(batch_size, num_nodes, self.num_points, channels)

        query = query.reshape(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch_size, num_nodes, self.num_points, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        value = value.reshape(
            batch_size, num_nodes, self.num_points, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)

        attn = torch.einsum("bhnd,bhnkd->bhnk", query, key) / (self.head_dim**0.5)

        distance = torch.linalg.vector_norm(total_offsets, dim=-1).reshape(
            batch_size, num_nodes, self.num_points
        )
        distance_lambda = self.distance_lambda.clamp_min(0.0)
        attn = attn - distance_lambda * distance.unsqueeze(1)

        if self.use_density_guidance and density_norm is not None:
            density_neighbors = _sample_map(density_norm, sample_coords).squeeze(-1)
            density_query = density_norm.flatten(2).transpose(1, 2)
            density_delta = (density_query - density_neighbors).abs()
            density_gamma = self.density_gamma.clamp_min(0.0)
            attn = attn - density_gamma * density_delta.unsqueeze(1)
        else:
            density_delta = feature_maps.new_zeros(batch_size, num_nodes, self.num_points)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, value)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, channels)
        out = self.out_proj(out)
        out = x_flat + self.residual_gate.tanh() * out
        out = out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        out = out.contiguous()

        entropy = -(attn.clamp_min(1e-6) * attn.clamp_min(1e-6).log()).sum(dim=-1)
        aux: dict[str, torch.Tensor] = {
            "residual_offset_abs_mean": residual_offsets.detach().abs().mean(),
            "residual_offset_abs_max": residual_offsets.detach().abs().amax(),
            "total_offset_abs_mean": total_offsets.detach().abs().mean(),
            "total_offset_abs_max": total_offsets.detach().abs().amax(),
            "attention_entropy": entropy.detach().mean(),
            "distance_lambda": distance_lambda.detach(),
            "density_gamma": self.density_gamma.clamp_min(0.0).detach(),
            "density_delta_mean": density_delta.detach().mean(),
            "residual_gate": self.residual_gate.tanh().detach(),
        }
        if self.debug:
            aux["sample_coords"] = sample_coords.detach()
        return out, aux


class SpatialTriFusionGate(nn.Module):
    """Density-conditioned three-way fusion gate."""

    def __init__(
        self,
        in_channels: int = 256,
        density_embed_dim: int = 32,
        hidden_channels: int = 128,
        init_weights: tuple[float, float, float] = (0.8, 0.1, 0.1),
        spatial: bool = True,
    ) -> None:
        super().__init__()
        if len(init_weights) != 3:
            raise ValueError("init_weights must contain three values")
        init = torch.tensor(init_weights, dtype=torch.float32)
        if (init <= 0).any():
            raise ValueError("init_weights must be positive")
        init = init / init.sum()
        self.spatial = bool(spatial)
        self.density_embed = nn.Sequential(
            nn.Conv2d(1, density_embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(density_embed_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 3 + density_embed_dim, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        final_conv = self.net[-1]
        assert isinstance(final_conv, nn.Conv2d)
        nn.init.zeros_(final_conv.weight)
        with torch.no_grad():
            final_conv.bias.copy_(init.log())

    def forward(
        self,
        features_pa: torch.Tensor,
        density_feature: torch.Tensor,
        feature_feature: torch.Tensor,
        density: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if density.shape[-2:] != features_pa.shape[-2:]:
            density = F.interpolate(
                density,
                size=features_pa.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        density_embed = self.density_embed(_normalise_density(density))
        gate_input = torch.cat(
            [features_pa, density_feature, feature_feature, density_embed], dim=1
        )
        if not self.spatial:
            gate_input = F.adaptive_avg_pool2d(gate_input, 1)
        weights = self.net(gate_input).softmax(dim=1)
        if not self.spatial:
            weights = weights.expand(-1, -1, features_pa.shape[-2], features_pa.shape[-1])
        fused = (
            weights[:, 0:1] * features_pa
            + weights[:, 1:2] * density_feature
            + weights[:, 2:3] * feature_feature
        )
        return fused, weights


class DeformableDualFusion(nn.Module):
    """Dual deformable-attention replacement for DSGCNet's dual GCN."""

    def __init__(
        self,
        in_channels: int = 256,
        num_points: int = 4,
        num_heads: int = 4,
        max_offset: float = 4.0,
        density_offset_rho: float = 0.5,
        density_gamma_init: float = 0.5,
        distance_lambda_init: float = 1.0,
        dropout: float = 0.1,
        density_embed_dim: int = 32,
        fusion_hidden_channels: int = 128,
        fusion_init_weights: tuple[float, float, float] = (0.8, 0.1, 0.1),
        fusion_spatial: bool = True,
        residual_gate_init: float = 0.001,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.density_branch = GuidedDeformableAttention(
            in_channels=in_channels,
            num_points=num_points,
            num_heads=num_heads,
            max_offset=max_offset,
            use_density_guidance=True,
            density_offset_rho=density_offset_rho,
            density_gamma_init=density_gamma_init,
            distance_lambda_init=distance_lambda_init,
            dropout=dropout,
            residual_gate_init=residual_gate_init,
            debug=debug,
        )
        self.feature_branch = GuidedDeformableAttention(
            in_channels=in_channels,
            num_points=num_points,
            num_heads=num_heads,
            max_offset=max_offset,
            use_density_guidance=False,
            density_offset_rho=0.0,
            density_gamma_init=0.0,
            distance_lambda_init=distance_lambda_init,
            dropout=dropout,
            residual_gate_init=residual_gate_init,
            debug=debug,
        )
        self.fusion_gate = SpatialTriFusionGate(
            in_channels=in_channels,
            density_embed_dim=density_embed_dim,
            hidden_channels=fusion_hidden_channels,
            init_weights=fusion_init_weights,
            spatial=fusion_spatial,
        )

    def forward(
        self,
        features_pa: torch.Tensor,
        density: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        density_feature, density_aux = self.density_branch(features_pa, density=density)
        feature_feature, feature_aux = self.feature_branch(features_pa, density=None)
        fused, fusion_weights = self.fusion_gate(
            features_pa, density_feature, feature_feature, density
        )
        aux: dict[str, torch.Tensor] = {
            "density_residual_offset_abs_mean": density_aux[
                "residual_offset_abs_mean"
            ],
            "density_residual_offset_abs_max": density_aux["residual_offset_abs_max"],
            "density_total_offset_abs_mean": density_aux["total_offset_abs_mean"],
            "density_attention_entropy": density_aux["attention_entropy"],
            "density_gamma": density_aux["density_gamma"],
            "density_delta_mean": density_aux["density_delta_mean"],
            "feature_residual_offset_abs_mean": feature_aux[
                "residual_offset_abs_mean"
            ],
            "feature_residual_offset_abs_max": feature_aux["residual_offset_abs_max"],
            "feature_total_offset_abs_mean": feature_aux["total_offset_abs_mean"],
            "feature_attention_entropy": feature_aux["attention_entropy"],
            "distance_lambda_density": density_aux["distance_lambda"],
            "distance_lambda_feature": feature_aux["distance_lambda"],
            "density_residual_gate": density_aux["residual_gate"],
            "feature_residual_gate": feature_aux["residual_gate"],
            "fusion_weights": fusion_weights.detach(),
        }
        if "sample_coords" in density_aux:
            aux["density_sample_coords"] = density_aux["sample_coords"]
        if "sample_coords" in feature_aux:
            aux["feature_sample_coords"] = feature_aux["sample_coords"]
        return fused, aux
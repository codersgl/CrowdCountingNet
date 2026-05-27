"""Density regression head and point prediction head for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class PointPredHead(nn.Module):
    """Per-pixel point prediction head for auxiliary supervision (P2PNet-style).

    Produces per-pixel classification logits and (dx, dy) offset predictions
    that are compatible with HungarianMatcher_Crowd.

    Reference: Song et al., "Rethinking Counting and Localization in Crowds:
    A Purely Point-Based Framework", ICCV 2021.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        stride: int = 8,
    ) -> None:
        super().__init__()
        self.stride = float(stride)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.reg_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = features.shape
        x = self.trunk(features)
        pred_logits = self.cls_conv(x)  # [B, 2, H, W]
        pred_offsets = self.reg_conv(x).tanh() * self.stride  # [B, 2, H, W]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=features.device, dtype=features.dtype),
            torch.arange(w, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        grid_centers = torch.stack([
            (grid_x + 0.5) * self.stride,
            (grid_y + 0.5) * self.stride,
        ], dim=0)  # [2, H, W]

        pred_logits_flat = pred_logits.reshape(b, 2, h * w).transpose(1, 2)  # [B, N, 2]
        pred_offsets_flat = pred_offsets.reshape(b, 2, h * w).transpose(1, 2)  # [B, N, 2]
        grid_flat = grid_centers.reshape(2, h * w).t().unsqueeze(0)  # [1, N, 2]

        return {
            "pred_logits": pred_logits_flat,
            "pred_points": grid_flat + pred_offsets_flat,
            "pred_offsets": pred_offsets_flat,
        }


class DensityHead(nn.Module):
    """Deeper density head with three Conv3x3 layers for richer feature processing."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        final_activation: str = "softplus",
        initial_density: float = 0.05,
        final_weight_std: float = 1e-4,
        output_kernel_size: int = 1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        if final_activation not in {"relu", "softplus", "none"}:
            raise ValueError("final_activation must be one of: relu, softplus, none")
        if initial_density <= 0:
            raise ValueError("initial_density must be > 0")
        if final_weight_std < 0:
            raise ValueError("final_weight_std must be >= 0")
        if output_kernel_size not in {1, 3, 5}:
            raise ValueError("output_kernel_size must be 1, 3, or 5")
        self.final_activation = final_activation
        self.use_residual = use_residual
        gn1 = min(32, hidden_channels)
        s2_channels = hidden_channels // 2
        gn2 = min(32, s2_channels)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn1, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_channels, s2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn2, s2_channels),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(
            hidden_channels // 2, 1,
            kernel_size=output_kernel_size,
            padding=output_kernel_size // 2,
        )
        self._init_final_layer(initial_density, final_weight_std)
        if use_residual:
            self.residual_proj = nn.Conv2d(in_channels, 1, kernel_size=1)
            self._init_residual_proj(initial_density, final_weight_std)

    def _init_final_layer(self, initial_density: float, final_weight_std: float) -> None:
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=final_weight_std)
        if self.final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, bias_value)

    def _init_residual_proj(self, initial_density: float, final_weight_std: float) -> None:
        nn.init.normal_(self.residual_proj.weight, mean=0.0, std=final_weight_std)
        if self.final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        if self.residual_proj.bias is not None:
            nn.init.constant_(self.residual_proj.bias, bias_value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stage1(features)
        x = self.stage2(x)
        density = self.output_conv(x)
        if self.use_residual:
            density = density + self.residual_proj(features)
        if self.final_activation == "relu":
            return F.relu(density)
        if self.final_activation == "softplus":
            return F.softplus(density, beta=1, threshold=20)
        return density

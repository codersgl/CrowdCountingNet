"""Density regression head and anchor-based point prediction head for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.head import (
    ClassificationModel,
    RegressionModel,
    SharedPredictionTrunk,
)


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class DSGCAnchorPointHead(nn.Module):
    """Anchor-based point prediction head matching the DSGCNet architecture.

    Shared prediction trunk (2 Conv3x3) followed by independent
    ClassificationModel and RegressionModel projection layers, with
    4 anchor points per stride-8 cell (row=2, line=2).

    Replaces the P2PNet-style anchor-free PointPredHead which had:
      - 10,000x too-large regression loss weight
      - No shared trunk between cls/reg branches
      - Only 1 query per pixel (vs 4 anchors per cell)
    """

    def __init__(
        self,
        in_channels: int = 256,
        feature_size: int = 256,
        row: int = 2,
        line: int = 2,
    ) -> None:
        super().__init__()
        num_anchor_points = row * line
        self.pred_trunk = SharedPredictionTrunk(
            in_channels=in_channels,
            feature_size=feature_size,
        )
        self.regression = RegressionModel(
            num_features_in=feature_size,
            num_anchor_points=num_anchor_points,
        )
        self.classification = ClassificationModel(
            num_features_in=feature_size,
            num_anchor_points=num_anchor_points,
            num_classes=2,
            prior=0.01,
        )
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_feat = self.pred_trunk(features)
        regression = self.regression(shared_feat) * 100.0
        classification = self.classification(shared_feat)
        batch_size = features.shape[0]
        anchors = self.anchor_points(image).repeat(batch_size, 1, 1)
        return {
            "pred_logits": classification,
            "pred_points": regression + anchors,
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

"""Density regression head for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class DensityHead(nn.Module):
    """Minimal density head producing a single-channel count-preserving map."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        final_activation: str = "softplus",
        initial_density: float = 0.05,
        final_weight_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if final_activation not in {"relu", "softplus", "none"}:
            raise ValueError("final_activation must be one of: relu, softplus, none")
        if initial_density <= 0:
            raise ValueError("initial_density must be > 0")
        if final_weight_std < 0:
            raise ValueError("final_weight_std must be >= 0")
        self.final_activation = final_activation
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self._init_final_layer(initial_density, final_weight_std)

    def _init_final_layer(self, initial_density: float, final_weight_std: float) -> None:
        final_conv = self.proj[-1]
        if not isinstance(final_conv, nn.Conv2d):
            raise TypeError("DensityHead final layer must be Conv2d")
        nn.init.normal_(final_conv.weight, mean=0.0, std=final_weight_std)
        if self.final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        nn.init.constant_(final_conv.bias, bias_value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        density = self.proj(features)
        if self.final_activation == "relu":
            return F.relu(density)
        if self.final_activation == "softplus":
            return F.softplus(density, beta=1, threshold=20)
        return density

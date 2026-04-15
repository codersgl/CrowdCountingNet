"""Concat-Gate Fusion Module for RGBD dual-branch feature fusion.

Fuses RGB and Depth features at a given scale via:
    concat → 1×1 compress → sigmoid gate → residual output.

This is intentionally lightweight and self-contained — no dependency on
existing ISFM / Geo-Prior modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatGateFusion(nn.Module):
    """Channel-concat + 1×1 gated fusion.

    Given RGB feature ``f_rgb`` and Depth feature ``f_depth`` (same spatial
    size & channel count ``C``), the module:

    1. Concatenates along channels → ``[B, 2C, H, W]``
    2. Projects back to ``C`` via 1×1 conv → ``f_cat``
    3. Produces a spatial gate ``g = σ(conv_gate(f_cat))`` in ``[0, 1]``
    4. Returns ``f_rgb + g * f_cat`` (residual fusion)

    Args:
        in_channels: Number of channels for both RGB and Depth features.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        f_cat = self.compress(torch.cat([f_rgb, f_depth], dim=1))
        g = self.gate(f_cat)
        return f_rgb + g * f_cat

"""Depth Residual Gating: lightweight depth-conditioned feature modulation.

Design C from the depth-guided crowd counting reference architecture.
The module adds a learnable residual path conditioned on depth:

    feat_out = feat + gate * depth_encoder(resize(depth_map))

where ``gate`` is a scalar :class:`nn.Parameter` initialised to **0**.
This ensures the module acts as a pure identity at the start of training,
gradually learning to incorporate depth information as needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthResidualGating(nn.Module):
    """Depth-conditioned residual gating for a single feature scale.

    Args:
        feat_channels: Number of channels in the input feature map (e.g. 256, 512).
        mid_ratio: Reduction ratio for the intermediate channel count.
            ``mid = max(feat_channels // mid_ratio, 32)``.
    """

    def __init__(self, feat_channels: int, mid_ratio: int = 4) -> None:
        super().__init__()
        mid = max(feat_channels // mid_ratio, 32)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
        )

        # Learnable gate initialised to 0 → identity at start of training.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Apply depth-conditioned residual gating.

        Args:
            feat: Feature map ``[B, C, H_f, W_f]``.
            depth_map: Depth / scale map ``[B, 1, H, W]`` (any spatial size).

        Returns:
            Modulated feature ``[B, C, H_f, W_f]``.
        """
        _, _, h_f, w_f = feat.shape
        depth_resized = F.interpolate(
            depth_map, size=(h_f, w_f), mode="bilinear", align_corners=False
        )
        depth_feat = self.depth_encoder(depth_resized)  # [B, C, H_f, W_f]
        return feat + self.gate * depth_feat

"""Sub-pixel density regression head.

Uses PixelShuffle-based upsampling with refinement convolutions to produce
higher-resolution density maps that preserve fine-grained details in dense
crowd regions, preventing nearby heads from merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubPixelDensityHead(nn.Module):
    """PixelShuffle-based sub-pixel density map predictor.

    Upsamples feature maps by 2× via PixelShuffle (learnable sub-pixel
    convolution), then refines with lightweight conv layers. Produces
    density maps at twice the input resolution (e.g., H/8 → H/4).

    Args:
        in_channels: Input feature channels (default 256).
        hidden_channels: Intermediate channels (default 64).
        upscale_factor: PixelShuffle upscale factor (default 2).
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        upscale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        out_ch = hidden_channels * (upscale_factor**2)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(in_channels, out_ch, 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            # After PixelShuffle: [B, hidden_channels, H*2, W*2]
        )

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.density_out = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.ReLU(inplace=True),  # Density must be non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Density map [B, 1, 2H, 2W] at 2× input resolution.
        """
        x = self.pre_conv(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        return self.density_out(x)

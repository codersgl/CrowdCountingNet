"""Cross-Scale Density Refinement for multi-scale density supervision.

Implements coarse-to-fine density prediction where coarser scales guide
finer scales via attention gating and residual refinement.

Also provides a lightweight fusion module that merges multi-scale density
maps into a single density map for GCN graph construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DensityRefinementStage(nn.Module):
    """Single refinement stage: fuse coarse density with fine-scale features.

    Takes an upsampled coarse density map and fine-scale backbone features,
    applies attention gating, and predicts a residual density correction.
    """

    def __init__(self, feat_channels: int) -> None:
        super().__init__()
        # Attention gate: coarse density → spatial mask on fine features
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.Sigmoid(),
        )
        # Residual density predictor from gated features
        self.residual_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, kernel_size=1),
        )

    def forward(
        self, coarse_density: torch.Tensor, fine_feat: torch.Tensor
    ) -> torch.Tensor:
        """Refine density using coarse prediction and fine features.

        Args:
            coarse_density: [B, 1, H_c, W_c] density from coarser scale
            fine_feat: [B, C, H_f, W_f] backbone features at finer scale

        Returns:
            Refined density map [B, 1, H_f, W_f]
        """
        # Upsample coarse density to fine resolution
        coarse_up = F.interpolate(
            coarse_density,
            size=fine_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Attention gating: highlight high-density regions in fine features
        gate = self.gate_conv(coarse_up)  # [B, C, H_f, W_f]
        gated_feat = fine_feat * gate  # element-wise gating

        # Predict residual correction
        residual = self.residual_head(gated_feat)  # [B, 1, H_f, W_f]

        # Coarse-to-fine: refined = coarse_upsampled + residual
        return F.relu(coarse_up + residual)


class CrossScaleDensityRefinement(nn.Module):
    """Coarse-to-fine multi-scale density prediction with cross-scale interaction.

    Flow: c5 → density_block5 → upsample + gate c4 → density_block4 →
          upsample + gate c3 → density_block3

    Each finer scale receives attention guidance from the coarser prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        # Coarsest scale head (block5: 512ch → density)
        self.head_block5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Refinement stages: coarser → finer
        self.refine_5to4 = _DensityRefinementStage(feat_channels=512)  # block4: 512ch
        self.refine_4to3 = _DensityRefinementStage(feat_channels=256)  # block3: 256ch

    def forward(
        self,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with coarse-to-fine refinement.

        Args:
            c3: [B, 256, H/4, W/4] block3 features
            c4: [B, 512, H/8, W/8] block4 features
            c5: [B, 512, H/16, W/16] block5 features

        Returns:
            Dict with density_block3, density_block4, density_block5
        """
        # Stage 1: coarsest prediction from block5
        density_block5 = self.head_block5(c5)  # [B, 1, H/16, W/16]

        # Stage 2: refine to block4 resolution
        density_block4 = self.refine_5to4(density_block5, c4)  # [B, 1, H/8, W/8]

        # Stage 3: refine to block3 resolution
        density_block3 = self.refine_4to3(density_block4, c3)  # [B, 1, H/4, W/4]

        return {
            "density_block3": density_block3,
            "density_block4": density_block4,
            "density_block5": density_block5,
        }


class MultiScaleDensityFusion(nn.Module):
    """Fuse multi-scale density maps into a single density map for GCN input.

    Resizes all scales to the target resolution (H/16) and applies
    learned channel-wise weighting via 1×1 convolution.
    """

    def __init__(self, num_scales: int = 4) -> None:
        super().__init__()
        # num_scales density maps (block3 + block4 + block5 + original) → 1
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(num_scales, num_scales * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_scales * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_scales * 2, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        density_main: torch.Tensor,
        density_block3: torch.Tensor,
        density_block4: torch.Tensor,
        density_block5: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse density maps to target resolution (same as density_main).

        Args:
            density_main: [B, 1, H, W] main PA-FPN density
            density_block3: [B, 1, H3, W3]
            density_block4: [B, 1, H4, W4]
            density_block5: [B, 1, H5, W5]

        Returns:
            Fused density map [B, 1, H, W]
        """
        target_size = density_main.shape[-2:]

        d3 = F.interpolate(
            density_block3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        d4 = F.interpolate(
            density_block4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        d5 = F.interpolate(
            density_block5,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Concat along channel dim → [B, 4, H, W]
        stacked = torch.cat([density_main, d3, d4, d5], dim=1)
        return self.fuse_conv(stacked)

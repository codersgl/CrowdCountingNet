"""Cross-Scale Graph Aggregation.

Constructs graphs at multiple FPN scales (local dense + global sparse)
and aggregates them via cross-scale semantic injection. Local graph
handles fine-grained person boundary refinement while global graph
provides context for distinguishing crowd from background texture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleGraphLayer(nn.Module):
    """Multi-scale convolution aggregation at a single FPN scale.

    Uses multi-branch dilated convolutions to capture neighbours at different
    hop distances, plus a squeeze-excitation channel attention for adaptive
    branch weighting.  Completely resolution-invariant — the same code path
    executes during both training (small patches) and inference (full images).

    Args:
        in_channels: Feature dimension.
        dilations: Dilation rates for the parallel conv branches.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_branches = len(dilations)

        # Local feature extraction via depthwise conv (residual path)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

        # Multi-scale dilated conv branches
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        3,
                        1,
                        padding=d,
                        dilation=d,
                        groups=in_channels,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels, in_channels, 1),
                    nn.BatchNorm2d(in_channels),
                )
            )

        # 1×1 fusion of concatenated branches
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * self.num_branches, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # SE channel attention for adaptive branch weighting
        r = max(1, in_channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, in_channels),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, C, H, W] feature map at this scale.

        Returns:
            Updated features [B, C, H, W].
        """
        # Local residual path
        local_feat = self.local_conv(x)

        # Multi-scale dilated branches
        branch_outs = [branch(x) for branch in self.branches]
        cat = torch.cat(branch_outs, dim=1)  # [B, C*num_branches, H, W]
        fused = self.fuse(cat)  # [B, C, H, W]

        # SE channel attention
        se_w = self.se(fused).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        fused = fused * se_w

        fused = self.drop(fused)
        return fused + local_feat


class CrossScaleGraphAggregation(nn.Module):
    """Cross-scale graph aggregation over FPN features.

    Builds:
        - Local dense aggregation on high-res features (P3, H/4 scale)
        - Global aggregation on low-res features (P5, H/16 scale)
    Then injects global semantics into local features via gated cross-scale edges.

    Final output is at the mid-resolution (H/8) to match the pipeline.

    Args:
        in_channels: Feature dimension at each scale (default 256).
        local_dilations: Dilation rates for local (high-res) aggregation.
        global_dilations: Dilation rates for global (low-res) aggregation.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        local_dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        global_dilations: tuple[int, ...] | list[int] = (1, 3, 6),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Local dense aggregation on high-res features
        self.local_graph = _ScaleGraphLayer(
            in_channels, dilations=local_dilations, dropout=dropout
        )

        # Global aggregation on low-res features
        self.global_graph = _ScaleGraphLayer(
            in_channels, dilations=global_dilations, dropout=dropout
        )

        # Cross-scale injection: global → local via gated fusion
        self.cross_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.cross_proj = nn.Conv2d(in_channels, in_channels, 1)

        # Final fusion to mid-resolution output
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(
        self,
        f_local: torch.Tensor,
        f_mid: torch.Tensor,
        f_global: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            f_local: High-res features [B, C, H1, W1] (from P3, ~H/4).
            f_mid: Mid-res features [B, C, H2, W2] (from P4, ~H/8).
            f_global: Low-res features [B, C, H3, W3] (from P5, ~H/16).

        Returns:
            Aggregated features [B, C, H2, W2] at mid-resolution.
        """
        target_size = f_mid.shape[-2:]

        # Process local graph (high-res)
        local_out = self.local_graph(f_local)

        # Process global graph (low-res)
        global_out = self.global_graph(f_global)

        # Cross-scale injection: upsample global → local resolution, gate
        global_up = F.interpolate(
            global_out, size=f_local.shape[-2:], mode="bilinear", align_corners=False
        )
        gate = self.cross_gate(torch.cat([local_out, global_up], dim=1))
        cross_proj = self.cross_proj(global_up)
        local_enhanced = local_out + gate * cross_proj

        # Resize all to mid-resolution
        local_mid = F.interpolate(
            local_enhanced, size=target_size, mode="bilinear", align_corners=False
        )
        global_mid = F.interpolate(
            global_out, size=target_size, mode="bilinear", align_corners=False
        )

        # Concatenate and fuse
        fused = torch.cat([local_mid, f_mid, global_mid], dim=1)
        out = self.fusion(fused) + f_mid  # residual from mid-scale
        return out

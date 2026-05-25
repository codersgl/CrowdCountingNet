"""FPN necks for MoECountNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.neck import (
    _FastNormalizedFusion,
    _bifpn_refine,
    _resize_like,
    SPDBiFPNBlock,
)


def _conv_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.ReLU(inplace=True),
    )


class EnhancedFPNNeck(nn.Module):
    """Fuse stride-8 and stride-16 features, then add dilated context."""

    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        out_channels: int = 256,
        branch_channels: tuple[int, int, int] = (128, 64, 64),
        dilations: tuple[int, int, int] = (1, 2, 5),
    ) -> None:
        super().__init__()
        if len(branch_channels) != 3 or len(dilations) != 3:
            raise ValueError("branch_channels and dilations must both have length 3")
        if sum(branch_channels) != out_channels:
            raise ValueError(
                "sum(branch_channels) must equal out_channels; "
                f"got {sum(branch_channels)} vs {out_channels}"
            )
        self.c2_proj = nn.Conv2d(c2_channels, out_channels, kernel_size=1)
        self.c3_proj = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.context_branches = nn.ModuleList(
            [
                _conv_relu(out_channels, branch_channels[index], 3, dilations[index])
                for index in range(3)
            ]
        )
        self.context_norm = nn.GroupNorm(32, out_channels)
        self.context_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(32, out_channels)
        self.output_channels = out_channels

    def forward(self, c2_feature: torch.Tensor, c3_feature: torch.Tensor) -> torch.Tensor:
        c2_projected = self.c2_proj(c2_feature)
        c3_projected = self.c3_proj(c3_feature)
        c3_up = F.interpolate(
            c3_projected,
            size=c2_projected.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        base_feature = c2_projected + c3_up
        context_feature = torch.cat(
            [branch(base_feature) for branch in self.context_branches],
            dim=1,
        )
        context_feature = self.context_norm(context_feature)
        context_feature = self.context_fuse(context_feature)
        return self.output_norm(base_feature + context_feature)


class DeepBiFPNNeck(nn.Module):
    """3-level SPD-BiFPN neck with dilated context refinement.

    Upgrades the original EnhancedFPNNeck from 2-level simple fusion to a
    bidirectional feature pyramid with learnable fusion weights, SPD-based
    downsampling, and multi-scale dilated context branches.
    """

    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        out_channels: int = 256,
        num_bifpn_blocks: int = 1,
        branch_channels: tuple[int, int, int] = (128, 64, 64),
        dilations: tuple[int, int, int] = (1, 2, 5),
        use_spd_downsample: bool = True,
        use_depthwise_refine: bool = True,
    ) -> None:
        super().__init__()
        if len(branch_channels) != 3 or len(dilations) != 3:
            raise ValueError("branch_channels and dilations must both have length 3")
        if sum(branch_channels) != out_channels:
            raise ValueError(
                f"sum(branch_channels) must equal out_channels; "
                f"got {sum(branch_channels)} vs {out_channels}"
            )
        if num_bifpn_blocks < 1:
            raise ValueError("num_bifpn_blocks must be >= 1")

        self.out_channels = out_channels

        self.p3_proj = nn.Sequential(
            nn.Conv2d(c2_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.p4_proj = nn.Sequential(
            nn.Conv2d(c3_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.p5_proj = nn.Sequential(
            nn.Conv2d(c4_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.bifpn_blocks = nn.ModuleList([
            SPDBiFPNBlock(
                feature_size=out_channels,
                use_spd_downsample=use_spd_downsample,
                use_depthwise_refine=use_depthwise_refine,
            )
            for _ in range(num_bifpn_blocks)
        ])

        self.stride8_fusion = _FastNormalizedFusion(3)
        self.stride8_refine = _bifpn_refine(out_channels, use_depthwise_refine)

        self.context_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, branch_channels[i], kernel_size=3,
                          padding=dilations[i], dilation=dilations[i], bias=True),
                nn.ReLU(inplace=True),
            )
            for i in range(3)
        ])
        self.context_norm = nn.GroupNorm(32, out_channels)
        self.context_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(32, out_channels)

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        p3 = self.p3_proj(c2)
        p4 = self.p4_proj(c3)
        p5 = self.p5_proj(c4)

        for block in self.bifpn_blocks:
            p3, p4, p5 = block(p3, p4, p5)

        p4_to_s8 = _resize_like(p4, p3)
        p5_to_s8 = _resize_like(p5, p3)

        base_feature = self.stride8_refine(
            self.stride8_fusion([p3, p4_to_s8, p5_to_s8])
        )

        context_feature = torch.cat(
            [branch(base_feature) for branch in self.context_branches],
            dim=1,
        )
        context_feature = self.context_norm(context_feature)
        context_feature = self.context_fuse(context_feature)
        return self.output_norm(base_feature + context_feature)

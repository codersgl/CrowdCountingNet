"""CrowdFPN: multi-scale feature fusion neck for crowd counting.

Re-implementation of CrowdFPN (zf990312/CrowdFPN) adapted for 3 input scales
from Swin backbone (Stage 1-3).  Core components from the original paper:

  1. **CAM** (Cross-level Attention Module): ASPP-based multi-scale feature
     aggregation with per-level sigmoid attention maps.
  2. **BiFPN-style bidirectional path**: SeparableConv + Swish, top-down then
     bottom-up with concatenation skip connections.
  3. **CoordinateAttention**: joint channel-spatial attention on each output
     level via horizontal/vertical pooling.
  4. **Multi-scale fusion**: upsample all levels to the finest resolution,
     concatenate, and project to final feature map.

Adapted from 4 scales to 3 scales (Stage 4 excluded per architecture design).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Swish activation (memory-efficient autograd)
# ---------------------------------------------------------------------------


class _SwishImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SwishImpl.apply(x)


# ---------------------------------------------------------------------------
# SeparableConvBlock (depthwise + pointwise, optional BN + Swish)
# ---------------------------------------------------------------------------


class SeparableConvBlock(nn.Module):
    """Depthwise separable conv (3×3 DW + 1×1 PW) with optional BN + Swish."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        norm: bool = True,
        activation: bool = False,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.bn = (
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3) if norm else None
        )
        self.act = Swish() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# ASPP (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------


class ASPP(nn.Module):
    """ASPP with dilated convolutions + global average pooling branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: tuple[int, ...] = (1, 2, 5, 1),
    ) -> None:
        super().__init__()
        self.aspp = nn.ModuleList()
        for d in dilations:
            ks = 3 if d > 1 else 1
            pad = d if d > 1 else 0
            self.aspp.append(
                nn.Conv2d(
                    in_channels, out_channels, ks, padding=pad, dilation=d, bias=True
                )
            )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_x = self.gap(x)
        out = []
        for i, layer in enumerate(self.aspp):
            inp = avg_x if (i == len(self.aspp) - 1) else x
            out.append(F.relu(layer(inp), inplace=True))
        # Expand global branch to spatial size
        out[-1] = out[-1].expand_as(out[-2])
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# CAM (Cross-level Attention Module)
# ---------------------------------------------------------------------------


class CAM(nn.Module):
    """Cross-level Attention Module: aggregates multi-scale features via ASPP
    and generates per-level sigmoid attention maps.

    Uses parallel multi-scale downsampling instead of cascade to avoid error
    accumulation.  Each attention level has an independent path from the
    aggregated feature.
    """

    def __init__(self, inplanes: int, num_levels: int = 3) -> None:
        super().__init__()
        self.num_levels = num_levels

        # Aggregate + ASPP + refine
        self.dila_conv = nn.Sequential(
            nn.Conv2d(inplanes * num_levels, inplanes, 3, padding=1),
            ASPP(inplanes, inplanes // 4),
            nn.Conv2d(inplanes, inplanes, 3, padding=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )

        # Parallel per-level downsampling + attention conv
        # Each path is independent (no cascading)
        self.down_paths = nn.ModuleList()
        for i in range(num_levels):
            num_down = i  # level 0: no downsampling, level 1: 2×, level 2: 4×
            layers: list[nn.Module] = []
            for _ in range(num_down):
                layers.append(nn.Conv2d(inplanes, inplanes, 3, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(inplanes))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(inplanes, 1, 3, padding=1))
            self.down_paths.append(nn.Sequential(*layers))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """features: list of [B, C, Hi, Wi] at different scales.

        Returns per-level sigmoid attention maps.
        """
        target_size = features[0].shape[2:]
        # Upsample all to finest resolution and concatenate
        multi_feats = [features[0]]
        for i in range(1, len(features)):
            multi_feats.append(
                F.interpolate(features[i], size=target_size, mode="nearest")
            )
        agg = self.dila_conv(torch.cat(multi_feats, dim=1))

        # Generate per-level attention in parallel (no cascading)
        atts = []
        for i in range(self.num_levels):
            att_map = torch.sigmoid(self.down_paths[i](agg))
            # Ensure attention matches target spatial size
            target_sz = features[i].shape[2:]
            if att_map.shape[-2:] != target_sz:
                att_map = F.interpolate(
                    att_map, size=target_sz, mode="bilinear", align_corners=False
                )
            atts.append(att_map)
        return atts


# ---------------------------------------------------------------------------
# Coordinate Attention (Hou et al., CVPR 2021)
# ---------------------------------------------------------------------------


class CoordinateAttention(nn.Module):
    """Coordinate Attention: encodes channel + spatial (H/W) dependencies."""

    def __init__(
        self, in_channels: int, out_channels: int, reduction: int = 32
    ) -> None:
        super().__init__()
        mid = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mid, out_channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Horizontal and vertical pooling
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1] → [B, C, W, 1]

        # Concat along spatial dim, shared bottleneck
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        return x * a_h * a_w


# ---------------------------------------------------------------------------
# CrowdFPN main module
# ---------------------------------------------------------------------------


class CrowdFPN(nn.Module):
    """CrowdFPN: BiFPN-style neck with CAM + CoordinateAttention.

    Adapted from the original CrowdFPN paper for 3 input scales, with
    enhancements: parallel CAM, scale-specific coordinate attention, and
    learnable BiFPN weight fusion.

    Input (from Swin backbone after projection):
        C2: [B, C2_ch, H/4,  W/4 ]  (stride 4)
        C3: [B, C3_ch, H/8,  W/8 ]  (stride 8)
        C4: [B, C4_ch, H/16, W/16]  (stride 16)

    Output:
        fused: [B, feature_size, H/8, W/8]  (stride 8)
    """

    def __init__(
        self,
        C2_channels: int = 256,
        C3_channels: int = 256,
        C4_channels: int = 512,
        feature_size: int = 128,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        fs = feature_size  # internal unified channel dim (128 in original)

        # --- Channel unification (1×1 conv) ---
        self.one_conv1 = nn.Conv2d(C2_channels, fs, 1)
        self.one_conv2 = nn.Conv2d(C3_channels, fs, 1)
        self.one_conv3 = nn.Conv2d(C4_channels, fs, 1)

        # --- CAM (Cross-level Attention Module) ---
        self.cam = CAM(fs, num_levels=3)

        # --- BiFPN top-down pathway with learnable fusion weights ---
        # Weighted fuse(x2, upsample(x3)) → Swish → SeparableConv  [at stride 8]
        self.conv_td_2 = SeparableConvBlock(fs, fs)
        # Weighted fuse(x1, upsample(x2_up)) → Swish → SeparableConv  [at stride 4]
        self.conv_td_1 = SeparableConvBlock(fs, fs)

        # Learnable BiFPN fusion weights (EfficientDet-style)
        # td_2: merge x2 + up(x3) → 2 inputs
        self.td2_weights = nn.Parameter(torch.ones(2))
        # td_1: merge x1 + up(x2_up) → 2 inputs
        self.td1_weights = nn.Parameter(torch.ones(2))

        # --- BiFPN bottom-up pathway with learnable fusion weights ---
        # Weighted fuse(x2, x2_up, downsample(x1_out)) → Swish → SeparableConv  [at stride 8]
        self.conv_bu_2 = SeparableConvBlock(fs, fs)
        # Weighted fuse(x3, downsample(x2_out)) → Swish → SeparableConv  [at stride 16]
        self.conv_bu_3 = SeparableConvBlock(fs, fs)

        # bu_2: merge x2 + x2_up + down(x1_out) → 3 inputs
        self.bu2_weights = nn.Parameter(torch.ones(3))
        # bu_3: merge x3 + down(x2_out) → 2 inputs
        self.bu3_weights = nn.Parameter(torch.ones(2))

        self.swish = Swish()
        self.downsample = nn.Conv2d(fs, fs, 3, stride=2, padding=1)

        # --- Scale-specific CoordinateAttention ---
        self.coord_att_1 = CoordinateAttention(fs, fs)  # stride 4
        self.coord_att_2 = CoordinateAttention(fs, fs)  # stride 8
        self.coord_att_3 = CoordinateAttention(fs, fs)  # stride 16

        # --- Final fusion: upsample all to stride-8, concat → project ---
        self.final_conv = nn.Sequential(
            nn.Conv2d(3 * fs, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _weighted_fuse(
        tensors: list[torch.Tensor], weights: nn.Parameter
    ) -> torch.Tensor:
        """Fuse tensors with learnable softmax-normalised weights."""
        w = F.softmax(weights, dim=0)
        out = sum(w[i] * t for i, t in enumerate(tensors))
        return out

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """inputs: [C2, C3, C4] from backbone.

        Returns fused feature [B, out_channels, H/8, W/8].
        """
        C2, C3, C4 = inputs

        # 1) Channel unification
        x1 = self.one_conv1(C2)  # [B, fs, H/4,  W/4 ]
        x2 = self.one_conv2(C3)  # [B, fs, H/8,  W/8 ]
        x3 = self.one_conv3(C4)  # [B, fs, H/16, W/16]

        # 2) CAM: cross-level attention
        att_list = self.cam([x1, x2, x3])
        # Ensure attention maps match feature spatial sizes exactly
        feats = [x1, x2, x3]
        for i in range(3):
            if att_list[i].shape[-2:] != feats[i].shape[-2:]:
                att_list[i] = F.interpolate(
                    att_list[i],
                    size=feats[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        x1 = (1 + att_list[0]) * x1
        x2 = (1 + att_list[1]) * x2
        x3 = (1 + att_list[2]) * x3

        # 3) BiFPN top-down with learnable weighted fusion
        # x2_up: weighted merge x2 (stride 8) + upsampled x3 (stride 16→8)
        x3_up = F.interpolate(x3, size=x2.shape[-2:], mode="nearest")
        x2_fused_td = self._weighted_fuse([x2, x3_up], self.td2_weights)
        x2_up = self.conv_td_2(self.swish(x2_fused_td))

        # x1_out: weighted merge x1 (stride 4) + upsampled x2_up (stride 8→4)
        x2_up_up = F.interpolate(x2_up, size=x1.shape[-2:], mode="nearest")
        x1_fused_td = self._weighted_fuse([x1, x2_up_up], self.td1_weights)
        x1_out = self.conv_td_1(self.swish(x1_fused_td))

        # 4) BiFPN bottom-up with learnable weighted fusion
        # x2_out: weighted merge x2 + x2_up + downsampled x1_out
        x1_down = self.downsample(x1_out)
        if x1_down.shape[-2:] != x2.shape[-2:]:
            x1_down = F.interpolate(x1_down, size=x2.shape[-2:], mode="nearest")
        x2_fused_bu = self._weighted_fuse([x2, x2_up, x1_down], self.bu2_weights)
        x2_out = self.conv_bu_2(self.swish(x2_fused_bu))

        # x3_out: weighted merge x3 + downsampled x2_out
        x2_down = self.downsample(x2_out)
        if x2_down.shape[-2:] != x3.shape[-2:]:
            x2_down = F.interpolate(x2_down, size=x3.shape[-2:], mode="nearest")
        x3_fused_bu = self._weighted_fuse([x3, x2_down], self.bu3_weights)
        x3_out = self.conv_bu_3(self.swish(x3_fused_bu))

        # 5) Scale-specific Coordinate Attention on each output level
        x1_out = self.coord_att_1(x1_out)  # stride 4
        x2_out = self.coord_att_2(x2_out)  # stride 8
        x3_out = self.coord_att_3(x3_out)  # stride 16

        # 6) Upsample all to stride-8 (x2's resolution) and fuse
        target_size = x2.shape[-2:]
        x1_fused = F.interpolate(
            x1_out, size=target_size, mode="bilinear", align_corners=False
        )
        x2_fused = x2_out  # already stride 8
        x3_fused = F.interpolate(
            x3_out, size=target_size, mode="bilinear", align_corners=False
        )

        fused = torch.cat([x1_fused, x2_fused, x3_fused], dim=1)
        return self.final_conv(fused)  # [B, out_channels, H/8, W/8]

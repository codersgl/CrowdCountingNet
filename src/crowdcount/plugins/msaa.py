from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ====================================================================
# Original MSAA components (kept for backward compatibility)
# ====================================================================


class SpatialAggregation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim, H, W)

        Returns:
            (batch_size, hidden_dim, H, W)
        """
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        pooled = torch.cat([avg_pool_out, max_pool_out], dim=1)  # [batch_size, 2, H, W]
        pooled = self.conv1(pooled)  # [batch_size, 1 H, W]
        spatial_attn = self.sigmoid(pooled)  # [batch_size, 1, H, W]
        return x * spatial_attn  # [batch_size, hidden_dim, H, W]


class ChannelAggregation(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (batch_size, in_channels, H, W)

        Returns:
            (batch_size, in_channels, 1, 1)
        """
        return self.fc(self.avg_pool(features))


class MultiScaleFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)

    def forward(self, features: torch.Tensor):
        """

        Args:
            features: (batch_size, dim, H, W)

        Returns:
            Fused features: (batch_size, dim, H, W)
        """
        x_1 = self.conv_3x3(features)  # [batch_size, dim, H, W]
        x_2 = self.conv_5x5(features)  # [batch_size, dim, H, W]
        x_3 = self.conv_7x7(features)  # [batch_size, dim, H, W]
        fused = x_1 + x_2 + x_3  # [batch_size, dim, H, W]
        return fused


class MSAA(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = in_channels // reduction

        self.down = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=1,
        )

        self.multi_scale_fusion = MultiScaleFusion(hidden_dim)

        self.spatial_aggregation = SpatialAggregation()
        self.channel_aggregation = ChannelAggregation(in_channels, hidden_dim)

        self.up = nn.Conv2d(
            hidden_dim,
            in_channels,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor):
        """

        Args:
            features: (batch_size, in_channels, H, W)
        """
        fused = self.multi_scale_fusion(
            self.down(features)
        )  # [batch_size, hidden_dim, H, W]
        spatial_aggregated_features = self.spatial_aggregation(
            fused
        )  # [batch_size, hidden_dim, H, W]
        spatial_aggregated_features = self.up(
            spatial_aggregated_features
        )  # [batch_size, input_dim, H, W]
        channel_aggregated_features = self.channel_aggregation(
            features
        )  # [batch_size, input_dim, 1, 1]
        return spatial_aggregated_features * channel_aggregated_features + features


class MsaaAdaptiveLayer(nn.Module):
    def __init__(self, in_channels: int = 1280, reduction: int = 4) -> None:
        super().__init__()
        self.msaa1 = MSAA(in_channels=in_channels, reduction=reduction)
        self.msaa2 = MSAA(in_channels=in_channels, reduction=reduction)
        self.msaa3 = MSAA(in_channels=in_channels, reduction=reduction)

    def forward(self, feature_list: List[torch.Tensor]) -> List[torch.Tensor]:

        feature1 = feature_list[1]  # [batch_size, 256, 32, 32]
        feature2 = feature_list[2]  # [batch_size, 512, 16, 16]
        feature3 = feature_list[3]  # [batch_size, 512, 8, 8]
        f_1 = torch.cat(
            [
                feature1,
                F.interpolate(feature2, feature1.size()[-2:], mode="bilinear"),
                F.interpolate(feature3, feature1.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 32, 32]

        f_2 = torch.cat(
            [
                feature2,
                F.interpolate(feature1, feature2.size()[-2:], mode="bilinear"),
                F.interpolate(feature3, feature2.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 16, 16]

        f_3 = torch.cat(
            [
                feature3,
                F.interpolate(feature1, feature3.size()[-2:], mode="bilinear"),
                F.interpolate(feature2, feature3.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 8, 8]
        return [feature_list[0], self.msaa1(f_1), self.msaa2(f_2), self.msaa3(f_3)]


# ====================================================================
# New lightweight components for MSAA v2
# ====================================================================


class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA) via adaptive 1-D convolution.

    Near-zero parameter overhead compared to SE-style FC bottleneck.
    Kernel size is auto-computed from channel count following the ECA-Net paper.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) — channel-reweighted features.
        """
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        y = self.conv(y.unsqueeze(1))  # [B, 1, C]
        y = self.sigmoid(y).unsqueeze(-1)  # [B, 1, C, 1]
        y = y.permute(0, 2, 1, 3)  # [B, C, 1, 1]
        return x * y


class DilatedMultiScaleFusion(nn.Module):
    """Multi-receptive-field feature extraction using depthwise dilated convolutions.

    Equivalent receptive fields to 3×3 / 5×5 / 7×7 kernels but with much
    fewer parameters: depthwise 3×3 (dilation 1/2/3) + pointwise 1×1 mixing.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Depthwise dilated convolutions (groups=dim → 1 filter per channel)
        self.dw_d1 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=False
        )
        self.dw_d2 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False
        )
        self.dw_d3 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False
        )
        # Pointwise mixing
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim, H, W)
        Returns:
            (B, dim, H, W) — fused multi-scale features.
        """
        fused = self.dw_d1(x) + self.dw_d2(x) + self.dw_d3(x)
        return self.relu(self.bn(self.pw(fused)))


# ====================================================================
# Phase 1: MSAALite — Post-PA-FPN lightweight attention
# ====================================================================


class MSAALite(nn.Module):
    """Lightweight multi-scale adaptive aggregation for post-PA-FPN refinement.

    Operates directly on 256-channel fused features — no down/up bottleneck,
    no cross-scale concatenation.  Uses dilated convolutions for multi-scale
    receptive fields and ECA for near-zero-cost channel attention.

    Typical parameter count: ~0.6M (vs ~15M for original MsaaAdaptiveLayer).
    """

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.multi_scale = DilatedMultiScaleFusion(in_channels)
        self.spatial = SpatialAggregation()
        self.channel = ECAAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — PA-FPN output.
        Returns:
            (B, C, H, W) — refined features with residual.
        """
        ms = self.multi_scale(x)  # multi-scale receptive field
        s = self.spatial(ms)  # spatial attention on multi-scale output
        out = self.channel(s)  # channel attention (serial, not parallel)
        return out + x  # residual


# ====================================================================
# Phase 2: FPNAttentionGate — injection inside PA-FPN
# ====================================================================


class FPNAttentionGate(nn.Module):
    """Adaptive gate for cross-scale lateral connections in PA-FPN.

    Applied before each top-down / bottom-up addition to learn per-channel
    blending weights between the lateral and the upsampled/downsampled path.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, lateral: torch.Tensor, transferred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lateral: (B, C, H, W) — feature from current scale's 1×1 conv.
            transferred: (B, C, H, W) — upsampled/downsampled feature from adjacent scale.
        Returns:
            (B, C, H, W) — adaptively blended feature.
        """
        combined = torch.cat(
            [
                F.adaptive_avg_pool2d(lateral, 1),
                F.adaptive_avg_pool2d(transferred, 1),
            ],
            dim=1,
        )  # [B, 2C, 1, 1]
        g = self.gate(combined).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return lateral * g + transferred * (1 - g)


class FPNSpatialAttention(nn.Module):
    """Spatial attention applied after the final multi-scale concat in PA-FPN."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialAggregation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x)


# ====================================================================
# Phase 3: MSAAGate — replace GateMechanism for GCN stream fusion
# ====================================================================


class MSAAGate(nn.Module):
    """Multi-scale attention-based gating for three-stream GCN fusion.

    Replaces SpatialGateMechanism with richer per-pixel fusion weights
    derived from multi-receptive-field features and dual attention.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_streams: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        concat_dim = in_channels * num_streams  # 768 for 3 streams

        # Project concatenated streams to hidden dim
        self.proj = nn.Sequential(
            nn.Conv2d(concat_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Multi-receptive-field extraction
        self.multi_scale = DilatedMultiScaleFusion(hidden_dim)

        # Per-stream spatial weight maps
        self.stream_conv = nn.Conv2d(hidden_dim, num_streams, kernel_size=1, bias=True)

        # Per-stream channel modulation via ECA on each input stream
        self.eca = ECAAttention(in_channels)

        self.num_streams = num_streams
        self.in_channels = in_channels

    def forward(
        self,
        features_pa: torch.Tensor,
        density_gcn: torch.Tensor,
        feature_gcn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features_pa:  (B, C, H, W) — original PA-FPN features.
            density_gcn:  (B, C, H, W) — density-GCN enhanced features.
            feature_gcn:  (B, C, H, W) — feature-GCN enhanced features.
        Returns:
            (B, C, H, W) — fused features.
        """
        concat = torch.cat([features_pa, density_gcn, feature_gcn], dim=1)
        h = self.proj(concat)  # [B, hidden, H, W]
        h = self.multi_scale(h)  # [B, hidden, H, W]

        # Spatial fusion weights: [B, 3, H, W]
        weights = F.softmax(self.stream_conv(h), dim=1)

        # Channel-refined streams
        s0 = self.eca(features_pa)
        s1 = self.eca(density_gcn)
        s2 = self.eca(feature_gcn)

        # Weighted sum
        return s0 * weights[:, 0:1] + s1 * weights[:, 1:2] + s2 * weights[:, 2:3]

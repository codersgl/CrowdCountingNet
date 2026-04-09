"""MSCADecoder: Multi-axis Strip Cross-Attention decoder.

Replaces PA-FPN (neck) + Density_pred + GCN fusion in a single module.
Adapted from MCANet (https://github.com/haoshao-nku/medical_seg) with all
mmseg/mmcv dependencies removed.

Forward: [c3, c4, c5] → (feature_fl, density)
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] → [B, H*W, C]."""
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B, H*W, C] → [B, C, H, W]."""
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# ---------------------------------------------------------------------------
# Layer normalisation variants (operating on channel dim of 4-D tensors)
# ---------------------------------------------------------------------------


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm4d(nn.Module):
    """LayerNorm that accepts 4-D [B, C, H, W] tensors."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.body = _WithBiasLayerNorm(dim) if bias else _BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Multi-axis Strip Cross-Attention (MSCAttention)
# ---------------------------------------------------------------------------


class MSCAttention(nn.Module):
    """Multi-axis strip cross-attention block.

    Uses depthwise strip convolutions at multiple scales (1×7, 7×1, 1×11,
    11×1, 1×21, 21×1) to capture long-range row/column dependencies, then
    performs cross-axis multi-head attention to mine feature correlations.
    This replaces the dual-stream GCN (DensityGCN + FeatureGCN) in the
    original DSGCNet pipeline.
    """

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm4d(dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        # Multi-scale depthwise strip convolutions (horizontal & vertical)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape
        x1 = self.norm1(x)

        # Multi-scale strip convolutions → two directional aggregations
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)

        out1 = self.project_out(attn_00 + attn_10 + attn_20)  # horizontal
        out2 = self.project_out(attn_01 + attn_11 + attn_21)  # vertical

        # Cross-axis multi-head attention
        # Axis-1 (row-wise): keys/values from horizontal, queries from vertical
        k1 = rearrange(out1, "b (head c) h w -> b head h (w c)", head=self.num_heads)
        v1 = rearrange(out1, "b (head c) h w -> b head h (w c)", head=self.num_heads)
        q1 = rearrange(out2, "b (head c) h w -> b head h (w c)", head=self.num_heads)

        # Axis-2 (column-wise): keys/values from vertical, queries from horizontal
        k2 = rearrange(out2, "b (head c) h w -> b head w (h c)", head=self.num_heads)
        v2 = rearrange(out2, "b (head c) h w -> b head w (h c)", head=self.num_heads)
        q2 = rearrange(out1, "b (head c) h w -> b head w (h c)", head=self.num_heads)

        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)).softmax(dim=-1)
        out3 = (attn1 @ v1) + q1

        attn2 = (q2 @ k2.transpose(-2, -1)).softmax(dim=-1)
        out4 = (attn2 @ v2) + q2

        out3 = rearrange(
            out3, "b head h (w c) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out4 = rearrange(
            out4, "b head w (h c) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        return self.project_out(out3) + self.project_out(out4) + x


# ---------------------------------------------------------------------------
# Depthwise separable convolution (pure PyTorch, replaces mmcv module)
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise 3×3 + pointwise 1×1, each with BN+ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.dw_bn(self.depthwise(x)))
        x = self.relu(self.pw_bn(self.pointwise(x)))
        return x


# ---------------------------------------------------------------------------
# Conv + BN + ReLU helper
# ---------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# MSCADecoder — replaces PA-FPN + Density_pred + GCN
# ---------------------------------------------------------------------------


class MSCADecoder(nn.Module):
    """Multi-axis Strip Cross-Attention Decoder.

    Replaces the entire PA-FPN neck → Density_pred → dual-stream GCN fusion
    pipeline.  Takes multi-scale backbone features ``[c3, c4, c5]`` and
    produces the fused feature map (``feature_fl``) and an auxiliary density
    prediction in one shot.

    Interface contract (drop-in for PA-FPN + Density_pred + GCN)::

        decoder = MSCADecoder(C3_size=256, C4_size=512, C5_size=512)
        feature_fl, density = decoder([c3, c4, c5])
        # feature_fl: [B, feature_size, H_c3, W_c3]
        # density:    [B, 1, H_c3, W_c3]

    Parameters
    ----------
    C3_size : int
        Channel count of the c3 backbone feature (default 256 for VGG-16).
    C4_size : int
        Channel count of the c4 backbone feature (default 512).
    C5_size : int
        Channel count of the c5 backbone feature (default 512).
    feature_size : int
        Unified channel dimension throughout the decoder (default 256).
    num_heads : int
        Number of attention heads in :class:`MSCAttention` (default 8).
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        feature_size: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feature_size = feature_size

        # --- Lateral 1×1 convolutions to unify channel count ---------------
        self.lateral_c3 = ConvBNReLU(C3_size, feature_size, 1)
        self.lateral_c4 = ConvBNReLU(C4_size, feature_size, 1)
        self.lateral_c5 = ConvBNReLU(C5_size, feature_size, 1)

        # --- Squeeze: concat c3+c4+c5 (all at c3 resolution) → feature_size
        self.squeeze = ConvBNReLU(feature_size * 3, feature_size, 1)

        # --- Multi-axis strip cross-attention (replaces GCN) ---------------
        self.decoder_level = MSCAttention(feature_size, num_heads, bias=True)

        # --- Bottleneck: cat(attention_out, c3_lateral) → refine -----------
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConv(
                feature_size + feature_size, feature_size, 3, padding=1
            ),
            DepthwiseSeparableConv(feature_size, feature_size, 3, padding=1),
        )

        # --- Align: 1×1 conv to final feature_size output -----------------
        self.align = ConvBNReLU(feature_size, feature_size, 1)

        # --- Density prediction head (replaces Density_pred) ---------------
        self.density_head = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        inputs : list[Tensor]
            ``[c3, c4, c5]`` multi-scale backbone features.
            - c3: [B, C3_size, H, W]
            - c4: [B, C4_size, H/2, W/2]
            - c5: [B, C5_size, H/4, W/4]

        Returns
        -------
        feature_fl : Tensor
            Fused feature map ``[B, feature_size, H/2, W/2]`` (same spatial
            size as c4, matching PA-FPN output convention).  Feeds directly
            into the shared prediction trunk → regression / classification
            heads.
        density : Tensor
            Auxiliary density map ``[B, 1, H/2, W/2]``.
        """
        c3, c4, c5 = inputs
        # Use c4 spatial resolution as the target (matches PA-FPN output)
        target_size = c4.shape[-2:]

        # Lateral projections
        c3_lat = self.lateral_c3(c3)
        c4_lat = self.lateral_c4(c4)
        c5_lat = self.lateral_c5(c5)

        # Resize all to c4 resolution and concatenate
        c3_down = F.interpolate(
            c3_lat, size=target_size, mode="bilinear", align_corners=False
        )
        c5_up = F.interpolate(
            c5_lat, size=target_size, mode="bilinear", align_corners=False
        )

        fused = torch.cat(
            [c3_down, c4_lat, c5_up], dim=1
        )  # [B, 3*feature_size, H/2, W/2]
        x = self.squeeze(fused)  # [B, feature_size, H/2, W/2]

        # Multi-axis strip cross-attention (replaces dual-stream GCN)
        x = self.decoder_level(x)

        # Concatenate with c3 low-level features (downsampled) and refine
        x = torch.cat([x, c3_down], dim=1)  # [B, 2*feature_size, H/2, W/2]
        x = self.sep_bottleneck(x)

        # Final alignment
        feature_fl = self.align(x)  # [B, feature_size, H, W]

        # Density prediction
        density = self.density_head(feature_fl)  # [B, 1, H, W]

        return feature_fl, density

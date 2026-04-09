"""MSCADecoder: Multi-axis Strip Cross-Attention decoder.

Replaces PA-FPN (neck) + Density_pred + GCN fusion in a single module.
Architecture follows the diagram: backbone multi-scale features → channel
attention weighted fusion → density branch → N × MSCA blocks → prediction.

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
# Channel Attention Module (upper gray box in the diagram)
# ---------------------------------------------------------------------------


class ChannelAttentionFusion(nn.Module):
    """Conv → MaxPool+AvgPool → MLP+ReLU+MLP → Softmax → 3-way weighting.

    Learns per-channel attention weights over three branches (top/mid/bot)
    and produces a weighted fusion.
    """

    def __init__(self, feature_size: int, reduction: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(feature_size * 3, feature_size * 3, 1, bias=False)
        mid = max(feature_size * 3 // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size * 3, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, feature_size * 3, bias=False),
        )

    def forward(
        self,
        f_top: torch.Tensor,
        f_mid: torch.Tensor,
        f_bot: torch.Tensor,
    ) -> torch.Tensor:
        """Produce attention-weighted fusion of 3 branches."""
        f_cat = torch.cat([f_top, f_mid, f_bot], dim=1)  # [B, 3C, H, W]
        c = self.conv(f_cat)

        # Dual pooling → channel descriptor
        pool = F.adaptive_max_pool2d(c, 1) + F.adaptive_avg_pool2d(c, 1)
        pool = pool.flatten(1)  # [B, 3C]
        attn_vec = self.mlp(pool)  # [B, 3C]

        # Reshape to [B, 3, C] and softmax over the 3 branches
        b, three_c = attn_vec.shape
        c_dim = three_c // 3
        attn_vec = attn_vec.view(b, 3, c_dim)
        attn_vec = attn_vec.softmax(dim=1)  # [B, 3, C]

        w_top = attn_vec[:, 0].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        w_mid = attn_vec[:, 1].unsqueeze(-1).unsqueeze(-1)
        w_bot = attn_vec[:, 2].unsqueeze(-1).unsqueeze(-1)

        return w_top * f_top + w_mid * f_mid + w_bot * f_bot


# ---------------------------------------------------------------------------
# MSCA Block (bottom box in diagram, repeated ×N)
# ---------------------------------------------------------------------------


class MSCABlock(nn.Module):
    """Multi-axis Strip Cross-Attention block.

    Two parallel branches (horizontal & vertical strip convolutions), each
    with its own LayerNorm.  Parallel conv outputs are **concatenated** then
    projected via 1×1 conv.  A learnable scalar W balances the two branches.
    Cross-attention is applied between branches, and density info is injected
    via element-wise addition.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # ---- Horizontal branch ----
        self.norm_h = LayerNorm4d(dim, bias=True)
        self.conv_h_7 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv_h_11 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv_h_21 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.proj_h = nn.Conv2d(dim * 3, dim, 1)

        # ---- Vertical branch ----
        self.norm_v = LayerNorm4d(dim, bias=True)
        self.conv_v_7 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv_v_11 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv_v_21 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.proj_v = nn.Conv2d(dim * 3, dim, 1)

        # ---- Learnable weight W for branch balancing ----
        # W applied to horizontal, (1-W) applied to vertical
        self.W = nn.Parameter(torch.tensor(0.5))

        # ---- QKV projections for cross-attention ----
        # Horizontal branch → V_h, K_h (used by vertical query)
        # Vertical branch → V_v, K_v (used by horizontal query)
        self.to_qkv_h = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_qkv_v = nn.Conv2d(dim, dim * 3, 1, bias=False)

        # Output projections
        self.proj_out_1 = nn.Conv2d(dim, dim, 1)
        self.proj_out_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, density_prob: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # ---- Horizontal strip convolutions ----
        x_h = self.norm_h(x)
        x_h = torch.cat(
            [self.conv_h_7(x_h), self.conv_h_11(x_h), self.conv_h_21(x_h)], dim=1
        )  # [B, 3C, H, W]
        x_h = self.proj_h(x_h)  # [B, C, H, W]

        # ---- Vertical strip convolutions ----
        x_v = self.norm_v(x)
        x_v = torch.cat(
            [self.conv_v_7(x_v), self.conv_v_11(x_v), self.conv_v_21(x_v)], dim=1
        )  # [B, 3C, H, W]
        x_v = self.proj_v(x_v)  # [B, C, H, W]

        # ---- Learnable weight balance ----
        w_val = self.W.sigmoid()  # keep in [0, 1]
        feat_h = w_val * x_h
        feat_v = (1.0 - w_val) * x_v

        # ---- Cross-attention: Q from one branch queries KV from other ----
        # feat_h → Q_h, K_h, V_h
        qkv_h = self.to_qkv_h(feat_h)
        q_h, k_h, v_h = qkv_h.chunk(3, dim=1)
        # feat_v → Q_v, K_v, V_v
        qkv_v = self.to_qkv_v(feat_v)
        q_v, k_v, v_v = qkv_v.chunk(3, dim=1)

        # Reshape to multi-head: [B, heads, N, head_dim]
        head_dim = c // self.num_heads
        n = h * w

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b (head d) h w -> b head (h w) d", head=self.num_heads)

        q_h_, k_h_, v_h_ = _to_heads(q_h), _to_heads(k_h), _to_heads(v_h)
        q_v_, k_v_, v_v_ = _to_heads(q_v), _to_heads(k_v), _to_heads(v_v)

        scale = head_dim**-0.5

        # Branch 1: Q_h queries K_v, V_v (horizontal queries vertical)
        attn_1 = (q_h_ @ k_v_.transpose(-2, -1)) * scale
        attn_1 = attn_1.softmax(dim=-1)
        out_1 = attn_1 @ v_v_
        out_1 = rearrange(out_1, "b head (h w) d -> b (head d) h w", h=h, w=w)
        out_1 = self.proj_out_1(out_1)

        # Branch 2: Q_v queries K_h, V_h (vertical queries horizontal)
        attn_2 = (q_v_ @ k_h_.transpose(-2, -1)) * scale
        attn_2 = attn_2.softmax(dim=-1)
        out_2 = attn_2 @ v_h_
        out_2 = rearrange(out_2, "b head (h w) d -> b (head d) h w", h=h, w=w)
        out_2 = self.proj_out_2(out_2)

        # ---- Fusion: out_1 + out_2 + density injection ----
        out = out_1 + out_2 + density_prob

        return out


# ---------------------------------------------------------------------------
# MSCADecoder — replaces PA-FPN + Density_pred + GCN
# ---------------------------------------------------------------------------


class MSCADecoder(nn.Module):
    """Multi-axis Strip Cross-Attention Decoder.

    Architecture (matching the diagram)::

        c3 (Downsample) ─┐
        c4 (Upsample)   ─┤→ ChannelAttentionFusion → Conv+BN+ReLU → density
        c5 (direct)      ─┘                                ↓ sigmoid
                                                    N × MSCABlock(feat, density_prob)
                                                            ↓
                                                    Prediction Head (external)

    Interface contract::

        decoder = MSCADecoder(C3_size=256, C4_size=512, C5_size=512)
        feature_fl, density = decoder([c3, c4, c5])

    Parameters
    ----------
    C3_size, C4_size, C5_size : int
        Channel counts of backbone features.
    feature_size : int
        Unified channel dimension (default 256).
    num_heads : int
        Attention heads in each MSCABlock (default 8).
    num_blocks : int
        Number of stacked MSCABlocks (×N in diagram, default 2).
    attn_reduction : int
        Reduction ratio in ChannelAttentionFusion MLP (default 16).
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        feature_size: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        attn_reduction: int = 16,
    ):
        super().__init__()
        self.feature_size = feature_size

        # --- Lateral 1×1 convolutions to unify channel count ---------------
        self.lateral_c3 = ConvBNReLU(C3_size, feature_size, 1)
        self.lateral_c4 = ConvBNReLU(C4_size, feature_size, 1)
        self.lateral_c5 = ConvBNReLU(C5_size, feature_size, 1)

        # --- Downsample for c3, Upsample for c4 (c5 stays) ----------------
        # c3 is at H/4 → needs downsample to H/8 (c4 resolution = target)
        self.downsample_c3 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        # c4 is already at target resolution H/8 → upsample (×2 from c5 res)
        # Per diagram: mid = Upsample(c4).  But c4 is at H/8, c5 at H/16.
        # Target resolution = c4 (H/8).  So c4 stays, c5 needs upsample.
        # Reinterpreting diagram mapping to our backbone:
        #   f_top = c5 → upsample to c4 resolution
        #   f_mid = c4 (already at target)
        #   f_bot = c3 → downsample to c4 resolution

        # --- Channel attention fusion (gray box in diagram) ----------------
        self.channel_attn = ChannelAttentionFusion(
            feature_size, reduction=attn_reduction
        )

        # --- Density branch: Conv+BN+ReLU → density map -------------------
        self.density_conv = ConvBNReLU(feature_size, feature_size, 3, padding=1)
        self.density_head = nn.Sequential(
            nn.Conv2d(feature_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # --- N × MSCA blocks (bottom box in diagram) ----------------------
        self.msca_blocks = nn.ModuleList(
            [MSCABlock(feature_size, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        inputs : list[Tensor]
            ``[c3, c4, c5]`` multi-scale backbone features.
            - c3: [B, C3_size, H, W]       (stride 4)
            - c4: [B, C4_size, H/2, W/2]   (stride 8)
            - c5: [B, C5_size, H/4, W/4]   (stride 16)

        Returns
        -------
        feature_fl : Tensor
            ``[B, feature_size, H/2, W/2]`` (c4 resolution, matching PA-FPN).
        density : Tensor
            ``[B, 1, H/2, W/2]``.
        """
        c3, c4, c5 = inputs
        target_size = c4.shape[-2:]  # H/2, W/2 (c4 resolution)

        # Lateral projections to unified channels
        c3_lat = self.lateral_c3(c3)  # [B, C, H, W]
        c4_lat = self.lateral_c4(c4)  # [B, C, H/2, W/2]
        c5_lat = self.lateral_c5(c5)  # [B, C, H/4, W/4]

        # Multi-scale alignment to c4 resolution:
        #   f_top = c5 upsampled
        #   f_mid = c4 (already at target)
        #   f_bot = c3 downsampled
        f_top = F.interpolate(
            c5_lat, size=target_size, mode="bilinear", align_corners=False
        )
        f_mid = c4_lat
        f_bot = self.downsample_c3(c3_lat)
        # Ensure exact size match after strided conv
        if f_bot.shape[-2:] != target_size:
            f_bot = F.interpolate(
                f_bot, size=target_size, mode="bilinear", align_corners=False
            )

        # Channel attention weighted fusion (gray box)
        f_fused = self.channel_attn(f_top, f_mid, f_bot)  # [B, C, H/2, W/2]

        # Density branch
        f_main = self.density_conv(f_fused)
        density = self.density_head(f_main)  # [B, 1, H/2, W/2]
        density_prob = density.sigmoid()  # condition signal for MSCA blocks

        # N × MSCA blocks with density injection
        f_out = f_main
        for block in self.msca_blocks:
            f_out = block(f_out, density_prob)

        return f_out, density

"""Scale-Decoupled CNN/GCN/Transformer Fusion for DSGCNet.

Replaces the Neck + Dual-Stream GCN pipeline with three parallel streams
at native backbone resolutions (s8/s16/s32), fused via Cross-Attention
and modulated by SE-style density channel attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# 2D Sinusoidal Position Encoding
# ---------------------------------------------------------------------------


def sinusoidal_2d_pe(h: int, w: int, dim: int) -> torch.Tensor:
    """Sinusoidal 2D positional encoding.

    Produces a [1, h*w, dim] tensor where each spatial position is encoded
    with sine/cosine functions at frequencies spanning the channel dimension.
    Half the channels encode y-positions, half encode x-positions.

    Args:
        h, w: spatial height and width of the feature grid.
        dim: total channel dimension (must be divisible by 4).

    Returns:
        [1, h*w, dim] positional encoding, values in [-1, 1].
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    half_dim = dim // 2
    div_term = torch.exp(
        torch.arange(0, half_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / half_dim)
    )

    # y-positions: [h, 1] × [half_dim//2] → [h, half_dim//2] → [h, 1, half_dim//2]
    pos_y = torch.arange(h, dtype=torch.float32).view(-1, 1)
    pe_y_even = torch.sin(pos_y * div_term).unsqueeze(1)  # [h, 1, half_dim//2]
    pe_y_odd = torch.cos(pos_y * div_term).unsqueeze(1)   # [h, 1, half_dim//2]

    # x-positions: [w, 1] × [half_dim//2] → [w, half_dim//2] → [1, w, half_dim//2]
    pos_x = torch.arange(w, dtype=torch.float32).view(-1, 1)
    pe_x_even = torch.sin(pos_x * div_term).unsqueeze(0)  # [1, w, half_dim//2]
    pe_x_odd = torch.cos(pos_x * div_term).unsqueeze(0)   # [1, w, half_dim//2]

    pe_y = torch.zeros(h, w, half_dim)
    pe_x = torch.zeros(h, w, half_dim)
    pe_y[:, :, 0::2] = pe_y_even.expand(h, w, -1)
    pe_y[:, :, 1::2] = pe_y_odd.expand(h, w, -1)
    pe_x[:, :, 0::2] = pe_x_even.expand(h, w, -1)
    pe_x[:, :, 1::2] = pe_x_odd.expand(h, w, -1)

    pe = torch.cat([pe_y, pe_x], dim=-1)  # [h, w, dim]
    return pe.reshape(1, h * w, dim)


# ---------------------------------------------------------------------------
# CNN Stream (stride-8, high-resolution local features)
# ---------------------------------------------------------------------------


class CNNStream(nn.Module):
    """Stride-8 local feature processor: multi-scale dilated convs + FFN + attention.

    Adapted from ``DensityAdaptiveLocalExpert`` without density modulation or
    point auxiliary head.  Internal standard residuals provide gradient
    stability; the output is a pure transform (no output residual).
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        dilations: tuple[int, ...] = (1, 2, 3),
        groups: int = 16,
        ffn_expansion: int = 2,
        use_multi_spectral_se: bool = True,
        ms_num_freqs: int = 4,
    ) -> None:
        super().__init__()
        from crowdcount.models.moecount.experts import MultiSpectralChannelAttention, SE

        self.out_channels = out_channels

        # ---- Stage 1: Multi-scale dilated conv block ----
        self.ms_norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d,
                          groups=min(groups, in_channels), bias=False),
                nn.GELU(),
            )
            for d in dilations
        ])
        self.branch_scales = nn.Parameter(torch.ones(len(dilations)))
        self.fuse_branches = nn.Sequential(
            nn.Conv2d(in_channels * len(dilations), in_channels, 1, bias=False),
            nn.GroupNorm(min(32, in_channels), in_channels),
        )

        # ---- Stage 2: FFN channel expansion ----
        ffn_hidden = in_channels * ffn_expansion
        self.ffn_norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, ffn_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, in_channels, 1, bias=False),
        )

        # ---- Stage 3: Channel attention ----
        if use_multi_spectral_se:
            self.channel_attn: nn.Module = MultiSpectralChannelAttention(
                in_channels, reduction=4, num_freqs=ms_num_freqs,
            )
        else:
            self.channel_attn = SE(in_channels, reduction=4)

        # ---- Output projection to unified channels ----
        self.output_norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Stage 1: Multi-scale dilated convs (internal standard residual)
        normed = self.ms_norm(features)
        branch_outs = [
            branch(normed) * scale
            for branch, scale in zip(self.dilated_branches, self.branch_scales)
        ]
        multi_scale = self.fuse_branches(torch.cat(branch_outs, dim=1))
        x = features + multi_scale

        # Stage 2: FFN (pre-norm, internal standard residual)
        x = x + self.ffn(self.ffn_norm(x))

        # Stage 3: Channel attention
        x = self.channel_attn(x)

        # Output: pure transform
        return self.output_proj(self.output_norm(x))


# ---------------------------------------------------------------------------
# GCN Stream (stride-16, mid-resolution relational reasoning)
# ---------------------------------------------------------------------------


class GCNStream(nn.Module):
    """Stride-16 relational processor: density k-NN graph + GATv2Conv.

    Uses ``SpatialPriorDensityGraphBuilder`` when density is provided,
    falls back to ``FeatureGraphBuilder`` (cosine similarity) when density
    is ``None``.  Each pixel is a graph node → reshape is lossless.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        k: int = 4,
        spatial_alpha: float = 1.0,
        spatial_beta: float = 1.0,
        hidden_channels: int = 512,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from crowdcount.models.gcn import (
            FeatureGraphBuilder,
            GATv2Model,
            SpatialPriorDensityGraphBuilder,
        )

        self.out_channels = out_channels

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

        self.density_builder = SpatialPriorDensityGraphBuilder(
            k=k, alpha=spatial_alpha, beta=spatial_beta,
        )
        self.feature_builder = FeatureGraphBuilder(k=k)

        self.gcn = GATv2Model(
            in_channels=out_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, _, H, W = features.shape
        x = self.input_proj(features)

        # Build graph: density-based if available, else feature-similarity fallback
        if density is not None:
            edge_index, _, _, _, _ = self.density_builder.build_batch_graph(density)
        else:
            edge_index, _, _, _, _ = self.feature_builder.build_batch_graph(x)

        # GCN propagation (node ↔ grid 1:1 mapping, reshape is lossless)
        node_features = x.permute(0, 2, 3, 1).reshape(-1, self.out_channels)
        out = self.gcn(node_features, edge_index)
        return out.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Transformer Stream (stride-32, low-resolution global context)
# ---------------------------------------------------------------------------


class TransformerStream(nn.Module):
    """Stride-32 global context processor: global self-attention Transformer.

    Uses ``FeatureTransformerBlock`` in "global" mode because s32 typically
    has only 49 tokens (7×7) — global self-attention costs only 2401 pairs.
    A learnable 2D position embedding (sinusoidal init) is added before the
    transformer blocks.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        embed_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from crowdcount.models.gcn import FeatureTransformerBlock

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

        self._pe_dim = out_channels
        self.pos_embed: nn.Parameter | None = None

        self.blocks = nn.ModuleList([
            FeatureTransformerBlock(
                in_channels=out_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=8,  # unused in global mode
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                gate_init=0.0,
                mode="global",
            )
            for _ in range(num_blocks)
        ])

    def _get_pos_embed(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create interpolated position embedding for the given spatial size."""
        if self.pos_embed is None or self.pos_embed.shape[-2:] != (h, w):
            pe = sinusoidal_2d_pe(h, w, self._pe_dim)  # [1, h*w, C]
            pe = pe.view(1, h, w, self._pe_dim).permute(0, 3, 1, 2)  # [1, C, h, w]
            self.pos_embed = nn.Parameter(pe.to(device=device, dtype=dtype))
        return self.pos_embed

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        h, w = x.shape[-2:]

        pos = self._get_pos_embed(h, w, x.device, x.dtype)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        return x


# ---------------------------------------------------------------------------
# Cross-Attention Fusion (Core Novel Module)
# ---------------------------------------------------------------------------


class ScaleDecoupledCrossAttention(nn.Module):
    """Cross-Attention: Q ← s8 features, K/V ← [s16, s32] features.

    Key design:
    - N_q ≠ N_kv naturally supported (no interpolate)
    - 2D sinusoidal PE on Q and K/V + learnable scale-level embeddings on K
    - Zero-init residual gates: at training step 0, f = Q_proj (identity)
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_expansion: int = 2,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q projection (from s8 features)
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_norm = nn.LayerNorm(dim)

        # K/V projections (separate for s16 and s32 for clarity)
        self.kv_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)

        # Scale-level embeddings (distinguish s16 vs s32 tokens in K/V)
        self.scale_embed = nn.Embedding(2, dim)
        nn.init.normal_(self.scale_embed.weight, std=0.02)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.attn_gate = nn.Parameter(torch.zeros(1))

        # FFN
        ffn_hidden = dim * ff_expansion
        self.ffn_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout),
        )
        self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        f_s8: torch.Tensor,
        f_s16: torch.Tensor,
        f_s32: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H8, W8 = f_s8.shape
        H16, W16 = f_s16.shape[-2:]
        H32, W32 = f_s32.shape[-2:]
        device = f_s8.device
        dtype = f_s8.dtype

        # --- Q from s8 features ---
        q = self.q_proj(f_s8)
        q = q.flatten(2).transpose(1, 2)  # [B, N_s8, C]
        q = self.q_norm(q) + sinusoidal_2d_pe(H8, W8, C).to(device=device, dtype=dtype)

        # --- K/V from s16 and s32 features ---
        k_s16 = self.kv_proj(f_s16).flatten(2).transpose(1, 2)  # [B, N_s16, C]
        k_s32 = self.kv_proj(f_s32).flatten(2).transpose(1, 2)  # [B, N_s32, C]
        v_s16 = self.kv_proj(f_s16).flatten(2).transpose(1, 2)
        v_s32 = self.kv_proj(f_s32).flatten(2).transpose(1, 2)

        k = torch.cat([k_s16, k_s32], dim=1)  # [B, N_s16+N_s32, C]
        v = torch.cat([v_s16, v_s32], dim=1)

        # 2D spatial PE + scale-level embeddings on K
        pos_kv = torch.cat([
            sinusoidal_2d_pe(H16, W16, C).to(device=device, dtype=dtype),
            sinusoidal_2d_pe(H32, W32, C).to(device=device, dtype=dtype),
        ], dim=1)
        scale_ids = torch.cat([
            torch.zeros(1, k_s16.shape[1], dtype=torch.long, device=device),
            torch.ones(1, k_s32.shape[1], dtype=torch.long, device=device),
        ], dim=1)
        k = k + pos_kv + self.scale_embed(scale_ids)

        k = self.k_norm(k)
        v = self.v_norm(v)

        # --- Multi-Head Cross-Attention ---
        N_q, N_kv = q.shape[1], k.shape[1]

        q_mh = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_mh = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v_mh = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v_mh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, C)

        attn_out = self.out_proj(attn_out)

        # Residual around Q (gate=0 → identity)
        f_attn = q + self.attn_gate.tanh() * attn_out

        # FFN with residual around f_attn
        f = f_attn + self.mlp_gate.tanh() * self.mlp(self.ffn_norm(f_attn))

        # Reshape to spatial
        return f.transpose(1, 2).view(B, C, H8, W8)


# ---------------------------------------------------------------------------
# Density Modulation (SE-style Channel Attention)
# ---------------------------------------------------------------------------


class DensitySEModulation(nn.Module):
    """SE-style density → channel attention modulation.

    Density map (detached) is encoded, globally pooled, and used to produce
    per-channel scaling factors.  Zero-init gain ensures identity at
    training step 0: ``f₁ = f * (1 + 0) = f``.
    """

    def __init__(
        self,
        channels: int = 256,
        density_hidden: int = 64,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, density_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(density_hidden),
            nn.GELU(),
        )

        mid_channels = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(density_hidden, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid(),
        )
        # Zero-init last FC → Sigmoid(0) = 0.5 → identity through residual form
        last_linear = self.se[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        # Safe detach (no-op during inference)
        d = density.detach() if density.requires_grad else density

        if d.shape[-2:] != f.shape[-2:]:
            d = F.interpolate(d, size=f.shape[-2:], mode="bilinear", align_corners=False)

        d_feat = self.density_encoder(d)
        channel_scale = self.se(d_feat).view(-1, f.shape[1], 1, 1)

        # gain=0 → f₁ = f * (1 + 0) = f  (identity at training start)
        return f * (1.0 + self.gain.tanh() * (channel_scale - 0.5))


# ---------------------------------------------------------------------------
# Top-Level Composite Module
# ---------------------------------------------------------------------------


class ScaleDecoupledFusion(nn.Module):
    """Scale-Decoupled CNN/GCN/Transformer → Cross-Attention → SE Modulation.

    Replaces DSGCNet's Neck + DGCN.  All streams process backbone features
    at their native resolutions; cross-attention fuses them at stride-8.
    """

    def __init__(
        self,
        c2_channels: int = 256,
        c3_channels: int = 512,
        c4_channels: int = 512,
        unified_dim: int = 256,
        # CNN stream
        cnn_dilations: tuple[int, ...] = (1, 2, 3),
        cnn_groups: int = 16,
        cnn_ffn_expansion: int = 2,
        cnn_use_multi_spectral_se: bool = True,
        # GCN stream
        gcn_k: int = 4,
        gcn_spatial_alpha: float = 1.0,
        gcn_spatial_beta: float = 1.0,
        gcn_hidden_channels: int = 512,
        gcn_heads: int = 4,
        gcn_dropout: float = 0.1,
        # Transformer stream
        trans_num_blocks: int = 2,
        trans_num_heads: int = 4,
        trans_embed_dim: int = 128,
        trans_mlp_ratio: float = 4.0,
        # Cross-attention
        ca_num_heads: int = 4,
        ca_dropout: float = 0.1,
        ca_ff_expansion: int = 2,
        # Density modulation
        dm_density_hidden: int = 64,
        dm_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.unified_dim = unified_dim

        self.cnn_stream = CNNStream(
            in_channels=c2_channels,
            out_channels=unified_dim,
            dilations=cnn_dilations,
            groups=cnn_groups,
            ffn_expansion=cnn_ffn_expansion,
            use_multi_spectral_se=cnn_use_multi_spectral_se,
        )
        self.gcn_stream = GCNStream(
            in_channels=c3_channels,
            out_channels=unified_dim,
            k=gcn_k,
            spatial_alpha=gcn_spatial_alpha,
            spatial_beta=gcn_spatial_beta,
            hidden_channels=gcn_hidden_channels,
            heads=gcn_heads,
            dropout=gcn_dropout,
        )
        self.transformer_stream = TransformerStream(
            in_channels=c4_channels,
            out_channels=unified_dim,
            num_blocks=trans_num_blocks,
            num_heads=trans_num_heads,
            embed_dim=trans_embed_dim,
            mlp_ratio=trans_mlp_ratio,
        )

        self.cross_attention = ScaleDecoupledCrossAttention(
            dim=unified_dim,
            num_heads=ca_num_heads,
            dropout=ca_dropout,
            ff_expansion=ca_ff_expansion,
        )

        self.density_modulation = DensitySEModulation(
            channels=unified_dim,
            density_hidden=dm_density_hidden,
            reduction=dm_reduction,
        )

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            c2: backbone stride-8 features  [B, C2, H/8, W/8]
            c3: backbone stride-16 features [B, C3, H/16, W/16]
            c4: backbone stride-32 features [B, C4, H/32, W/32]
            density: optional density map for GCN graph building.
                     If None, GCN uses feature-similarity fallback.

        Returns:
            f: fused features [B, unified_dim, H/8, W/8]
            aux: auxiliary dict (reserved for future use)
        """
        f_s8 = self.cnn_stream(c2)
        f_s16 = self.gcn_stream(c3, density=density)
        f_s32 = self.transformer_stream(c4)

        f = self.cross_attention(f_s8, f_s16, f_s32)

        return f, {}

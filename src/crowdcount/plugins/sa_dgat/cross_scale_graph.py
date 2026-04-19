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
    """Lightweight graph attention at a single scale.

    Uses depthwise-separable convolution + window-based self-attention to
    model local/global feature relationships efficiently without computing
    a full N×N attention matrix.

    Args:
        in_channels: Feature dimension.
        num_heads: Attention heads.
        k_neighbors: Number of local neighbours (controls receptive field).
        dropout: Dropout rate.
        window_size: Spatial window size for windowed attention. If 0,
            the window size is auto-selected based on k_neighbors.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.1,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.k = k_neighbors
        # Auto window size: sqrt(k) rounded up, minimum 4, ensures each
        # window has at least k tokens for meaningful sparse attention
        import math

        self.window_size = (
            window_size
            if window_size > 0
            else max(4, int(math.ceil(math.sqrt(k_neighbors))) + 1)
        )

        # Local feature extraction via depthwise conv
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

        # Self-attention for graph message passing
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.norm = nn.LayerNorm(in_channels)
        self.drop = nn.Dropout(dropout)

    def _window_attention(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Window-partitioned self-attention with top-k sparsity.

        Partitions the spatial grid into non-overlapping windows, computes
        attention within each window (much smaller than N×N), and
        reassembles. Handles non-divisible spatial dims via padding.

        Args:
            x: [B, N, C] flattened feature tokens.
            H: Spatial height.
            W: Spatial width.

        Returns:
            [B, N, C] attention output.
        """
        B, N, C = x.shape
        ws = self.window_size

        # If the feature map is small enough, use full attention directly
        if H <= ws and W <= ws:
            return self._full_attention(x)

        # Reshape to spatial: [B, H, W, C]
        x_2d = x.reshape(B, H, W, C)

        # Pad to multiples of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x_2d.shape[1], x_2d.shape[2]
        nH, nW = Hp // ws, Wp // ws

        # Partition into windows: [B*nH*nW, ws*ws, C]
        x_win = (
            x_2d.reshape(B, nH, ws, nW, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B * nH * nW, ws * ws, C)
        )

        # Attention within each window
        out_win = self._full_attention(x_win)

        # Reverse partition: [B, Hp, Wp, C]
        out_2d = (
            out_win.reshape(B, nH, nW, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, Hp, Wp, C)
        )

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out_2d = out_2d[:, :H, :W, :].contiguous()

        return out_2d.reshape(B, N, C)

    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Full self-attention with optional top-k sparsity.

        Args:
            x: [B_win, N_win, C] tokens within a window (or full if small).

        Returns:
            [B_win, N_win, C] attention output.
        """
        Bw, Nw, C = x.shape
        Q = (
            self.W_q(x)
            .reshape(Bw, Nw, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.W_k(x)
            .reshape(Bw, Nw, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.W_v(x)
            .reshape(Bw, Nw, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)

        if Nw > self.k:
            topk_vals, topk_idx = attn_scores.topk(self.k, dim=-1)
            attn_mask = torch.full_like(attn_scores, float("-inf"))
            attn_mask.scatter_(-1, topk_idx, topk_vals)
            attn = F.softmax(attn_mask, dim=-1)
        else:
            attn = F.softmax(attn_scores, dim=-1)

        attn = self.drop(attn)
        out = torch.matmul(attn, V)
        return out.permute(0, 2, 1, 3).reshape(Bw, Nw, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, C, H, W] feature map at this scale.

        Returns:
            Updated features [B, C, H, W].
        """
        B, C, H, W = x.shape
        N = H * W

        # Local convolution path
        local_feat = self.local_conv(x)

        # Window-based self-attention (memory-efficient for large spatial dims)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        out = self._window_attention(x_flat, H, W)
        out = self.out_proj(out)

        # Residual with local conv path
        x_flat = self.norm(x_flat + out)
        out_spatial = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out_spatial + local_feat


class CrossScaleGraphAggregation(nn.Module):
    """Cross-scale graph aggregation over FPN features.

    Builds:
        - Local dense graph on high-res features (F1, H/4 scale)
        - Global sparse graph on low-res features (F3, H/16 scale)
    Then injects global semantics into local features via gated cross-scale edges.

    Final output is at the mid-resolution (H/8) to match the pipeline.

    Args:
        in_channels: Feature dimension at each scale (default 256).
        k_local: Number of neighbours for local graph.
        k_global: Number of neighbours for global graph.
        num_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        k_local: int = 12,
        k_global: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Local dense graph on high-res features
        self.local_graph = _ScaleGraphLayer(in_channels, num_heads, k_local, dropout)

        # Global sparse graph on low-res features
        self.global_graph = _ScaleGraphLayer(in_channels, num_heads, k_global, dropout)

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

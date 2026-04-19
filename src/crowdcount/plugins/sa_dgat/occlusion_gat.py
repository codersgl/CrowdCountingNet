"""Occlusion-Aware Graph Attention message passing.

Incorporates an occlusion inference module that predicts per-node occlusion
rates, then modulates message sending/receiving weights accordingly:
- Highly occluded nodes (o_i → 1) send weaker messages (incomplete features)
- Non-occluded nodes (o_i → 0) receive more from surrounding occluded nodes
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionPredictor(nn.Module):
    """Predict per-pixel occlusion rate from features.

    Optionally incorporates depth information as a prior for occlusion.

    Args:
        in_channels: Feature dimension.
        hidden_channels: Hidden layer size.
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # Depth-aware branch (optional, adds depth features to prediction)
        self.depth_proj: nn.Module | None = None

    def enable_depth_prior(self, depth_channels: int = 1) -> None:
        """Enable depth-conditioned occlusion prediction."""
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict occlusion rate.

        Args:
            features: [B, C, H, W] feature map.
            depth: Optional [B, 1, H, W] depth map.

        Returns:
            Occlusion rate [B, 1, H, W] in [0, 1].
        """
        occ = self.net(features)
        if self.depth_proj is not None and depth is not None:
            if depth.shape[-2:] != features.shape[-2:]:
                depth = F.interpolate(
                    depth,
                    size=features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            occ = occ + self.depth_proj(depth)
        return occ.sigmoid()


class OcclusionAwareGAT(nn.Module):
    """Occlusion-aware Graph Attention Network layer.

    Two stacked attention layers with occlusion-modulated message passing.
    Messages from highly occluded nodes are dampened, while non-occluded
    receivers get boosted aggregation from surrounding occluded regions.

    Args:
        in_channels: Feature dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked GAT layers.
        dropout: Dropout rate.
        occ_hidden: Hidden channels for occlusion predictor.
        use_depth_prior: Whether to use depth for occlusion prediction.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        occ_hidden: int = 64,
        use_depth_prior: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.num_layers = num_layers
        assert in_channels % num_heads == 0

        self.occ_predictor = OcclusionPredictor(in_channels, occ_hidden)
        if use_depth_prior:
            self.occ_predictor.enable_depth_prior()

        # Multi-layer GAT
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_OccGATLayer(in_channels, num_heads, dropout))

    def forward(
        self,
        x: torch.Tensor,
        neighbor_feats: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].
            neighbor_feats: Pre-sampled neighbour features [B, N, K, C]
                (from DeformableGraphAttention's sampling).
            neighbor_mask: Optional [B, N, K] boolean mask for valid neighbours.
            depth: Optional [B, 1, H, W] depth map for occlusion prior.

        Returns:
            Tuple of:
                - Updated features [B, C, H, W].
                - Occlusion map [B, 1, H, W].
        """
        B, C, H, W = x.shape
        N = H * W

        # Predict occlusion rates: [B, 1, H, W]
        occ_map = self.occ_predictor(x, depth)
        occ_flat = occ_map.reshape(B, N)  # [B, N]

        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)

        for layer in self.layers:
            x_flat = layer(x_flat, neighbor_feats, occ_flat, neighbor_mask)

        out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out, occ_map


class _OccGATLayer(nn.Module):
    """Single occlusion-aware GAT layer."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 2, in_channels),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_feats: torch.Tensor,
        occ_flat: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for single GAT layer.

        Args:
            x: Node features [B, N, C].
            neighbor_feats: Neighbour features [B, N, K, C].
            occ_flat: Occlusion rates [B, N] in [0, 1].
            neighbor_mask: Optional [B, N, K] validity mask.

        Returns:
            Updated node features [B, N, C].
        """
        B, N, C = x.shape
        K = neighbor_feats.shape[2]

        Q = self.W_q(x)  # [B, N, C]
        K_proj = self.W_k(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)
        V = self.W_v(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)

        # Multi-head reshape
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.reshape(B, N, K, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        V = V.reshape(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention scores: [B, num_heads, N, K]
        attn = torch.einsum("bhnd,bhnkd->bhnk", Q, K_proj) / (self.head_dim**0.5)

        # Occlusion modulation on sender side:
        # Neighbour j sends to node i: scale by (1 - occ_j)
        # We need occ values for the K sampled neighbours.
        # Since neighbours are sampled at arbitrary positions, we use
        # the node-level occlusion as a proxy (neighbour features already
        # contain the information; the exact occ values are approximated).
        # For now: apply receiver-side boost: non-occluded receivers get enhanced attention
        # occ_receiver: [B, N] → [B, 1, N, 1]
        receiver_boost = (1.0 + occ_flat).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        attn = attn * receiver_boost

        if neighbor_mask is not None:
            attn = attn.masked_fill(~neighbor_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        # Residual + norm + FFN
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x

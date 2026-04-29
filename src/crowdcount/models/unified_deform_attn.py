"""Unified Deformable Spatial Attention — replaces dual-stream GCN with a single
deformable attention module.

Density map guides *where* to sample (offset prediction); feature similarity
determines *how much* to attend (attention weights).  No explicit graph
construction (k-NN + edge_index) needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowdcount.plugins.sa_dgat.deformable_graph import _grid_offset_bias


class UnifiedDeformableSpatialAttention(nn.Module):
    """Deformable self-attention on a 2D feature grid with density guidance.

    Each spatial position attends to *K* learned sampling points whose offsets
    are predicted from [features + density].  A learnable spatial-distance
    penalty and an optional density-difference bias regularise the attention
    pattern, making it a natural unification of the original dual-stream GCN.

    Args:
        d_model: Feature channels (default 256).
        num_heads: Multi-head attention heads.
        num_points: Sampling points per query (*K*, default 4, matching GCN k=4).
        ffn_expansion: Hidden-dim multiplier for the feed-forward network.
        dropout: Dropout rate (attention + FFN).
        lambda_dist_init: Initial distance penalty (learnable).
        gamma_density_init: Initial density-diff bias (learnable, 0 = disabled).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        lambda_dist_init: float = 1.0,
        gamma_density_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.K = num_points

        # ---- Offset prediction (density-conditioned) ----
        self.density_proj = nn.Sequential(
            nn.Conv2d(d_model + 1, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.offset_pred = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, 2 * num_points, 1),
        )
        # Small random init so gradients flow from step 1.
        # Bias is set to the spatial grid prior, so initial offsets ≈ grid bias.
        nn.init.normal_(self.offset_pred[-1].weight, std=1e-4)
        self.offset_pred[-1].bias.data = _grid_offset_bias(num_points)

        # ---- Q / K / V projections ----
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ---- Attention biases (learnable) ----
        self.lambda_dist = nn.Parameter(torch.tensor(lambda_dist_init))
        self.gamma_density = nn.Parameter(torch.tensor(gamma_density_init))

        self.attn_drop = nn.Dropout(dropout)

        # ---- Pre-LN transformer block ----
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

    def _sample_features(
        self, x: torch.Tensor, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear-sample *K* neighbours per spatial position.

        Returns:
            sampled: [B, N, K, C] neighbour features.
            coords:  [B, N, K, 2] sampling coordinates in [-1, 1].
        """
        B, C, H, W = x.shape
        K = self.K
        N = H * W

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        offsets = offsets.reshape(B, K, 2, H, W).permute(0, 3, 4, 1, 2)  # [B,H,W,K,2]
        offsets = offsets * torch.tensor([2.0 / W, 2.0 / H], device=x.device)

        sample_coords = base_grid.unsqueeze(3) + offsets  # [B, H, W, K, 2]
        sample_coords = sample_coords.clamp(-1, 1)

        flat_coords = sample_coords.reshape(B, N * K, 1, 2)
        sampled = F.grid_sample(
            x, flat_coords, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B, C, N*K, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(B, N, K, C)

        coords = sample_coords.reshape(B, N, K, 2)
        return sampled, coords

    def forward(
        self,
        feature_maps: torch.Tensor,
        density_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            feature_maps: [B, C, H, W] — PA-FPN features.
            density_maps: [B, 1, H, W] — density prediction (not detached).

        Returns:
            Enhanced features [B, C, H, W].
        """
        B, C, H, W = feature_maps.shape
        N = H * W
        K = self.K

        # 1. Predict offsets conditioned on [features + density]
        density_resized = density_maps
        if density_maps.shape[-2:] != (H, W):
            density_resized = F.interpolate(
                density_maps, size=(H, W), mode="bilinear", align_corners=False
            )
        offset_input = torch.cat([feature_maps, density_resized], dim=1)  # [B,C+1,H,W]
        offset_feat = self.density_proj(offset_input)  # [B, C, H, W]
        offsets = self.offset_pred(offset_feat)  # [B, 2K, H, W]

        # 2. Bilinear-sample neighbour features
        neighbor_feats, sample_coords = self._sample_features(feature_maps, offsets)
        # neighbor_feats: [B, N, K, C]   sample_coords: [B, N, K, 2]

        # 3. Flatten spatial dims
        x_flat = feature_maps.permute(0, 2, 3, 1).reshape(B, N, C)  # [B, N, C]

        # 4. K / V from sampled neighbour features
        K_proj = self.W_k(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)
        V = self.W_v(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)

        # 5. Pre-LN → Q from normed query features
        normed = self.norm1(x_flat)
        Q = self.W_q(normed)  # [B, N, C]

        # 6. Multi-head reshape
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,heads,N,d]
        K_proj = K_proj.reshape(B, N, K, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )  # [B,heads,N,K,d]
        V = V.reshape(B, N, K, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )  # [B,heads,N,K,d]

        # 7. Attention scores
        attn = torch.einsum("bhnd,bhnkd->bhnk", Q, K_proj) / (self.head_dim**0.5)

        # Spatial distance penalty
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feature_maps.device),
            torch.linspace(-1, 1, W, device=feature_maps.device),
            indexing="ij",
        )
        base_pos = (
            torch.stack([grid_x, grid_y], dim=-1).reshape(1, N, 2).expand(B, -1, -1)
        )
        dist = torch.norm(sample_coords - base_pos.unsqueeze(2), dim=-1)  # [B,N,K]
        attn = attn - self.lambda_dist * dist.unsqueeze(1)

        # Optional density-difference bias
        if self.gamma_density.abs() > 1e-8:
            flat_coords = sample_coords.reshape(B, N * K, 1, 2)
            sampled_density = F.grid_sample(
                density_resized,
                flat_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # [B, 1, N*K, 1]
            sampled_density = sampled_density.squeeze(-1).squeeze(1).reshape(B, N, K)
            query_density = density_resized.reshape(B, N)
            density_diff = torch.abs(
                query_density.unsqueeze(-1) - sampled_density
            )  # [B,N,K]
            attn = attn - self.gamma_density * density_diff.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 8. Aggregate + residual (Pre-LN attention block)
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, V)  # [B, heads, N, d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]
        out = x_flat + self.out_proj(out)

        # 9. Pre-LN FFN block
        out = out + self.ffn(self.norm2(out))

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

"""Deformable Graph Attention for dynamic graph construction.

Instead of fixed KNN or grid-based graphs, each node predicts K relative
offsets to dynamically locate its semantic neighbours. Edge weights combine
feature similarity, spatial distance penalty, and scale matching reward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableGraphAttention(nn.Module):
    """Deformable graph attention with dynamic neighbour sampling.

    For each spatial node, predicts K relative offsets and samples neighbour
    features via bilinear interpolation (fully differentiable). Computes
    attention-weighted message aggregation with distance penalty and
    scale-matching bonus.

    Args:
        in_channels: Input feature dimension.
        num_neighbors: Number of deformable neighbours per node (K).
        num_heads: Number of attention heads.
        lambda_init: Initial distance penalty coefficient.
        mu_init: Initial scale matching reward coefficient.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_neighbors: int = 8,
        num_heads: int = 4,
        lambda_init: float = 1.0,
        mu_init: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.K = num_neighbors
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0

        # Offset predictor: for each node, predict K 2D offsets
        self.offset_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 2 * num_neighbors, 1),
        )
        # Initialize offsets to small values (near-identity)
        nn.init.zeros_(self.offset_pred[-1].weight)
        nn.init.zeros_(self.offset_pred[-1].bias)

        # Query/Key/Value projections
        self.W_q = nn.Linear(in_channels, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        # Learnable distance penalty and scale matching reward
        self.lambda_dist = nn.Parameter(torch.tensor(lambda_init))
        self.mu_scale = nn.Parameter(torch.tensor(mu_init))

        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)

    def _sample_neighbors(
        self, x: torch.Tensor, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample K neighbours per node using predicted offsets.

        Args:
            x: Feature map [B, C, H, W].
            offsets: Predicted offsets [B, 2K, H, W].

        Returns:
            Tuple of:
                - Sampled features [B, N, K, C]
                - Sampling coordinates [B, N, K, 2] in [-1, 1] range
        """
        B, C, H, W = x.shape
        K = self.K

        # Create base grid: [1, H, W, 2] in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Reshape offsets: [B, K, 2, H, W] → [B, H, W, K, 2]
        offsets = offsets.reshape(B, K, 2, H, W).permute(0, 3, 4, 1, 2)

        # Scale offsets to grid range (normalize by spatial dims)
        offsets = offsets * torch.tensor([2.0 / W, 2.0 / H], device=x.device)

        # Compute sampling positions: [B, H, W, K, 2]
        sample_coords = base_grid.unsqueeze(3) + offsets  # [B, H, W, K, 2]
        sample_coords = sample_coords.clamp(-1, 1)

        # Flatten for grid_sample: [B, H*W*K, 1, 2] → sample → reshape
        flat_coords = sample_coords.reshape(B, H * W * K, 1, 2)
        sampled = F.grid_sample(
            x, flat_coords, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B, C, H*W*K, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, H*W*K, C]
        sampled = sampled.reshape(B, H * W, K, C)

        # Flatten coordinates for distance computation
        flat_sample_coords = sample_coords.reshape(B, H * W, K, 2)
        return sampled, flat_sample_coords

    def forward(
        self,
        x: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].
            scale_weights: Optional scale attention weights [B, N, num_prompts]
                from ScalePromptEmbedding, used for scale-matching bonus.

        Returns:
            Updated feature map [B, C, H, W].
        """
        B, C, H, W = x.shape
        N = H * W
        K = self.K

        # Predict offsets
        offsets = self.offset_pred(x)  # [B, 2K, H, W]

        # Sample neighbour features
        neighbor_feats, sample_coords = self._sample_neighbors(x, offsets)
        # neighbor_feats: [B, N, K, C], sample_coords: [B, N, K, 2]

        # Flatten node features
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)  # [B, N, C]

        # Multi-head attention
        Q = self.W_q(x_flat)  # [B, N, C]
        K_proj = self.W_k(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)
        V = self.W_v(neighbor_feats.reshape(B * N, K, C)).reshape(B, N, K, C)

        # Reshape for multi-head: [B, num_heads, N, head_dim] and [B, num_heads, N, K, head_dim]
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.reshape(B, N, K, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        V = V.reshape(B, N, K, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention scores: [B, num_heads, N, K]
        attn = torch.einsum("bhnd,bhnkd->bhnk", Q, K_proj) / (self.head_dim**0.5)

        # Distance penalty: compute L2 distance between base positions and sample positions
        # Base positions in [-1,1]: [B, N, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        base_pos = (
            torch.stack([grid_x, grid_y], dim=-1).reshape(1, N, 2).expand(B, -1, -1)
        )

        # Distance: [B, N, K]
        dist = torch.norm(sample_coords - base_pos.unsqueeze(2), dim=-1)  # [B, N, K]
        attn = attn - self.lambda_dist * dist.unsqueeze(1)  # broadcast over heads

        # Scale matching bonus (if scale_weights available)
        if scale_weights is not None:
            # scale_weights: [B, N, num_prompts]
            # Compute cosine similarity of scale weights between node and its neighbours
            sw_norm = F.normalize(scale_weights, dim=-1)  # [B, N, P]
            # Sample neighbour scale weights using same coordinates
            # Reshape scale_weights to [B, P, H, W] for grid_sample
            P = scale_weights.shape[-1]
            sw_map = scale_weights.reshape(B, H, W, P).permute(
                0, 3, 1, 2
            )  # [B, P, H, W]
            flat_coords = sample_coords.reshape(B, N * K, 1, 2)
            sampled_sw = F.grid_sample(
                sw_map,
                flat_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # [B, P, N*K, 1]
            sampled_sw = sampled_sw.squeeze(-1).permute(0, 2, 1).reshape(B, N, K, P)
            sampled_sw_norm = F.normalize(sampled_sw, dim=-1)

            # Scale match: cosine similarity [B, N, K]
            scale_match = torch.einsum("bnp,bnkp->bnk", sw_norm, sampled_sw_norm)
            attn = attn + self.mu_scale * scale_match.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate: [B, num_heads, N, head_dim]
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]

        out = self.out_proj(out)

        # Residual connection with norm
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        out = self.norm(x_flat + out)

        # Reshape back to spatial
        out_spatial = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Cache sampling coordinates for downstream modules to re-sample
        # from updated features while preserving graph topology
        self._cached_sample_coords = sample_coords  # [B, N, K, 2]

        return out_spatial

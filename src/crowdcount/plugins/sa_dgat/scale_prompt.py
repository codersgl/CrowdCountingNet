"""Scale-Aware Node Embedding via learnable scale prompts.

Each spatial feature is conditioned on a set of learnable scale prompts
through cross-attention, producing scale-aware node representations that
explicitly encode whether a region contains tiny/small/medium/large/crowd
targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalePromptEmbedding(nn.Module):
    """Cross-attention between spatial features and learnable scale prompts.

    Args:
        embed_dim: Feature channel dimension (default 256).
        num_prompts: Number of scale prompts (default 5: tiny/small/medium/large/crowd).
        num_heads: Number of attention heads for cross-attention.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_prompts: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts

        # Learnable scale prompts: [num_prompts, embed_dim]
        self.scale_prompts = nn.Parameter(torch.randn(num_prompts, embed_dim) * 0.02)

        # Cross-attention: query=spatial features, key/value=scale prompts
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].

        Returns:
            Tuple of:
                - Scale-conditioned features [B, C, H, W].
                - Scale attention weights [B, N, num_prompts] for downstream use.
        """
        B, C, H, W = x.shape
        # Flatten spatial dims: [B, N, C] where N = H*W
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Expand prompts for batch: [B, num_prompts, C]
        prompts = self.scale_prompts.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: features attend to scale prompts
        attn_out, attn_weights = self.cross_attn(
            query=x_flat,
            key=prompts,
            value=prompts,
            need_weights=True,
        )  # attn_out: [B, N, C], attn_weights: [B, N, num_prompts]

        # Residual + norm
        x_flat = self.norm(x_flat + attn_out)

        # FFN with residual
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        # Reshape back to spatial: [B, C, H, W]
        out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out, attn_weights

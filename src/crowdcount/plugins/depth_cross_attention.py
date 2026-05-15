"""Depth cross-attention fusion for post-neck RGB features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthCrossAttentionFusion(nn.Module):
    """Fuse RGB features with depth context using residual cross-attention.

    The module treats RGB features as queries and depth features as keys/values.
    A learnable scalar gate is initialised to zero so the block is an identity
    mapping at the start of training.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        depth_mid_channels: int = 64,
        mode: str = "window",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mode not in {"window", "global"}:
            raise ValueError(f"mode must be 'window' or 'global', got {mode!r}")
        if mode == "window" and window_size <= 0:
            raise ValueError(
                f"window_size must be positive for window mode, got {window_size}"
            )

        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.mode = mode

        depth_mid_channels = max(int(depth_mid_channels), 16)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, depth_mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(depth_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_mid_channels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.q_proj = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, rgb_feat: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Return depth-enhanced RGB features with the same shape as input."""
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        if depth_map.dim() != 4 or depth_map.shape[1] != 1:
            raise ValueError(
                "depth_map must have shape [B, 1, H, W] or [B, H, W], "
                f"got {tuple(depth_map.shape)}"
            )
        if rgb_feat.dim() != 4:
            raise ValueError(f"rgb_feat must be 4D, got {tuple(rgb_feat.shape)}")

        depth_map = depth_map.to(device=rgb_feat.device, dtype=rgb_feat.dtype)
        if depth_map.shape[-2:] != rgb_feat.shape[-2:]:
            depth_map = F.interpolate(
                depth_map,
                size=rgb_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        depth_feat = self.depth_encoder(depth_map)
        q = self.q_proj(rgb_feat)
        k = self.k_proj(depth_feat)
        v = self.v_proj(depth_feat)

        if self.mode == "global":
            attn_out = self._global_attention(q, k, v)
        else:
            attn_out = self._window_attention(q, k, v)

        residual = self.out_proj(attn_out)
        return rgb_feat + self.gate.tanh().to(dtype=rgb_feat.dtype) * residual

    def _global_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        bsz, channels, height, width = q.shape
        q_seq = q.flatten(2).transpose(1, 2)
        k_seq = k.flatten(2).transpose(1, 2)
        v_seq = v.flatten(2).transpose(1, 2)
        out = self._scaled_dot_product(q_seq, k_seq, v_seq)
        return out.transpose(1, 2).reshape(bsz, channels, height, width)

    def _window_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        bsz, channels, height, width = q.shape
        window_size = self.window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        padded_h, padded_w = q.shape[-2:]
        q_windows = self._partition_windows(q, window_size)
        k_windows = self._partition_windows(k, window_size)
        v_windows = self._partition_windows(v, window_size)
        out_windows = self._scaled_dot_product(q_windows, k_windows, v_windows)
        out = self._reverse_windows(
            out_windows,
            bsz=bsz,
            channels=channels,
            height=padded_h,
            width=padded_w,
            window_size=window_size,
        )
        return out[:, :, :height, :width]

    @staticmethod
    def _partition_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        x = x.view(
            bsz,
            channels,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, window_size * window_size, channels)

    @staticmethod
    def _reverse_windows(
        x: torch.Tensor,
        bsz: int,
        channels: int,
        height: int,
        width: int,
        window_size: int,
    ) -> torch.Tensor:
        x = x.view(
            bsz,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(bsz, channels, height, width)

    def _scaled_dot_product(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        batch_windows, tokens, _ = q.shape
        q = q.view(batch_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_windows, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(
            batch_windows, tokens, self.embed_dim
        )
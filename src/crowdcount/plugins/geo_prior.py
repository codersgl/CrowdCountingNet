"""Geometry Prior Generation and Attention for Depth-RGB Fusion.

This module implements the Geometry Prior Generation and Decomposed Geometry Self-Attention (GSA)
inspired by DFormer v2 (https://github.com/VCIP-RGBD/DFormer).
It calculates depth disparity among spatial tokens and uses it as an attention bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def angle_transform(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE rotation to the input tensor.

    Args:
        x: [B, num_heads, H, W, head_dim] or flattened equivalents.
        sin: [H, W, head_dim]
        cos: [H, W, head_dim]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


class GeoPriorGen(nn.Module):
    """Generates the geometry prior (RoPE and decay masks) for Decomposed GSA.

    Args:
        embed_dim: Channel dimension (must be divisible by num_heads * 2)
        num_heads: Number of attention heads
        initial_value: Base value for log-decay
        heads_range: Range multiplier for varying decay across heads
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
    ) -> None:
        super().__init__()
        assert embed_dim % (num_heads * 2) == 0, (
            "embed_dim must be divisible by num_heads * 2 for RoPE"
        )

        # RoPE angle freqs calculation
        head_dim_half = embed_dim // num_heads // 2
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim_half))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle)

        # Decay parameters
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1
            - 2
            ** (
                -initial_value
                - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads
            )
        )
        self.register_buffer("decay", decay)

    def generate_1d_decay(self, l: int) -> torch.Tensor:
        """Generate static 1d relative position decay. Output: [num_heads, l, l]"""
        index = torch.arange(l, device=self.decay.device)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(
        self, l: int, s: int, depth_grid: torch.Tensor
    ) -> torch.Tensor:
        """Generate dynamic 1d depth difference decay.
        Args:
            depth_grid: Shape [B, 1, s, l].
            l is the dimension along which we compute attention.
            s is the other spatial dimension.
        Output: [B, num_heads, s, l, l]
        """
        # [B, 1, s, l, 1] - [B, 1, s, 1, l] -> [B, 1, s, l, l]
        mask = depth_grid.unsqueeze(-1) - depth_grid.unsqueeze(-2)
        mask = mask.abs()
        # Multiply by [1, num_heads, 1, 1, 1] -> broadcasts to [B, num_heads, s, l, l]
        mask = mask * self.decay[None, :, None, None, None]
        return mask

    def forward(
        self, HW_tuple: Tuple[int, int], depth_map: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            HW_tuple: Target spatial size (H, W)
            depth_map: Original depth map [B, 1, H0, W0] or sized
        Returns:
            (sin, cos): Each is [H, W, head_dim]
            (mask_h, mask_w):
                mask_h: [B, num_heads, W, H, H]
                mask_w: [B, num_heads, H, W, W]
        """
        H, W = HW_tuple

        # Interpolate depth map to target grid size
        if depth_map.shape[-2:] != (H, W):
            depth_map = F.interpolate(
                depth_map, size=(H, W), mode="bilinear", align_corners=False
            )

        # Normalize depth map for stable decay calculations (assuming [0, max] range)
        max_val = depth_map.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        depth_map = depth_map / max_val

        # Generates RoPE embeddings
        index = torch.arange(H * W, device=self.angle.device)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)

        # Depth decay masks
        mask_d_h = self.generate_1d_depth_decay(H, W, depth_map.transpose(-2, -1))
        mask_d_w = self.generate_1d_depth_decay(W, H, depth_map)

        # Positional decay masks
        mask_h_pos = self.generate_1d_decay(H)
        mask_w_pos = self.generate_1d_decay(W)

        # Combine pos and depth weighting
        mask_h_pos = mask_h_pos.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, H, H]
        mask_w_pos = mask_w_pos.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, W, W]

        mask_h = self.weight[0] * mask_h_pos + self.weight[1] * mask_d_h
        mask_w = self.weight[0] * mask_w_pos + self.weight[1] * mask_d_w

        return (sin, cos), (mask_h, mask_w)


class DWConv2d(nn.Module):
    def __init__(self, dim: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/Output shape: [B, H, W, C] -> Permuted for Conv2d"""
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class DepthGeoPriorAttention(nn.Module):
    """Decomposed Geometry Self-Attention (row/col decomposed) using Depth Prior.

    Acts as a plug-in replacement module to fuse RGB with raw Depth Maps.
    Includes gating initialized to 0 for a stable identity residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = in_channels
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # QKV Projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.lepe = DWConv2d(self.embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.geo = GeoPriorGen(self.embed_dim, num_heads, initial_value, heads_range)

        # Gating parameter starting at 0 for identity residual mapping
        self.gate = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, rgb_feat: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat: [B, C, H, W]
            depth_map: [B, 1, H0, W0] or other raw spatial size
        Returns:
            rgb_fused: [B, C, H, W] enhanced features.
        """
        B, C, H, W = rgb_feat.shape

        # [B, C, H, W] -> [B, H, W, C]
        x_in = rgb_feat.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x_in)

        geo_prior = self.geo((H, W), depth_map)
        (sin, cos), (mask_h, mask_w) = geo_prior

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling

        # Prepare for attention -> [B, num_heads, H, W, head_dim]
        q = q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # RoPE Transformation
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        # -----------------------------
        # Width (Row) Attention
        # -----------------------------
        qr_w = qr.transpose(1, 2)  # [B, H, num_heads, W, head_dim]
        kr_w = kr.transpose(1, 2)
        v_w = v.view(B, H, W, self.num_heads, self.head_dim).permute(
            0, 1, 3, 2, 4
        )  # [B, H, num_heads, W, head_dim]

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        # mask_w is [B, num_heads, H, W, W] -> transpose allows broadcast over batch correctly
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v_fused = torch.matmul(qk_mat_w, v_w)  # [B, H, num_heads, W, head_dim]

        # -----------------------------
        # Height (Column) Attention
        # -----------------------------
        # Re-permute for column focus
        qr_h = qr.permute(0, 3, 1, 2, 4)  # [B, W, num_heads, H, head_dim]
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v_h = v_fused.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v_h)

        # Assemble back to [B, H, W, C]
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)

        # Apply gate and add to original permutation
        out_fused = x_in + self.gate.tanh() * output

        # Return [B, C, H, W]
        return out_fused.permute(0, 3, 1, 2).contiguous()

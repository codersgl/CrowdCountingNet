"""Deformable Cross-Scale Expert for MoECountNet.

DAT-style multi-scale deformable attention replacing fixed window attention.
Reference: DAT (Xia et al., CVPR 2022) and Deformable DETR (Zhu et al., ICLR 2021).

The expert keeps stride-8 resolution throughout (unlike W-MSA which downsamples
to stride-16). For each query position it attends to K deformable sampling points
across L feature scales, learning WHERE and at WHICH resolution to sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.neck import SPD


def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
    """Pad spatial dims to even for SPD compatibility."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class DeformableCrossScaleExpert(nn.Module):
    """Multi-scale deformable attention expert operating at stride-8 resolution.

    Replaces SpatialRelationExpert's fixed window attention with learned sparse
    sampling. Multi-scale features are built internally via SPD, following the
    same pattern as GlobalDensityExpert.

    Architecture
    ------------
    1. Internal multi-scale pyramid: P3(stride-8) / P4(stride-16) / P5(stride-32)
    2. Offset prediction from P3 → per-query, per-scale, per-point sampling offsets
    3. K/V projection per scale → sample at offset positions
    4. Multi-head dot-product attention with scale-level embeddings, distance penalty
    5. FFN + zero-init residual gate
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        num_sampling_points: int = 8,
        num_scale_levels: int = 3,
        max_offset: float = 8.0,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.num_points = num_sampling_points
        self.num_levels = num_scale_levels
        self.max_offset = max_offset
        self.scale = self.head_dim ** -0.5

        # ---- Stage 1: Multi-scale feature pyramid ----
        # Build modules for levels beyond P3 (stride-8 identity).
        # Each level halves spatial resolution via SPD.
        self._down_modules = nn.ModuleList()
        self._norms = nn.ModuleList([nn.LayerNorm(channels)])  # P3 norm
        for _ in range(num_scale_levels - 1):
            self._down_modules.append(
                nn.Sequential(
                    SPD(),
                    nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32, channels),
                    nn.ReLU(inplace=True),
                )
            )
            self._norms.append(nn.LayerNorm(channels))

        # ---- Stage 2: Offset prediction network ----
        # Lightweight: depthwise conv → GELU → pointwise conv
        # Predicts per-query sampling offsets shared across attention heads
        total_offsets = 2 * num_scale_levels * num_sampling_points
        self.offset_pred = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(channels, total_offsets, kernel_size=1),
        )
        # Zero-initialised → initial offsets = 0 (regular grid sampling)
        nn.init.zeros_(self.offset_pred[-1].weight)
        nn.init.zeros_(self.offset_pred[-1].bias)

        # ---- Stage 3 & 4: K/V projections + attention ----
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(num_scale_levels)
        ])
        self.v_proj = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(num_scale_levels)
        ])
        self.out_proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(dropout)

        # ---- Stage 5: FFN ----
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_expansion, channels),
            nn.Dropout(dropout),
        )

        # ---- Learnable parameters ----
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.distance_lambda = nn.Parameter(torch.tensor(1.0))
        self.level_embeds = nn.Parameter(
            torch.zeros(num_scale_levels, num_sampling_points)
        )

        # Optional SE channel attention after FFN
        if use_se:
            from crowdcount.models.moecount.experts import SE
            self.se = SE(channels, reduction=4)
        else:
            self.se = nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: [B, C, H, W] stride-8 fused neck output.

        Returns:
            [B, C, H, W] refined features at the same resolution.
        """
        B, C, H, W = features.shape
        N = H * W

        # ---- Stage 1: Build multi-scale feature pyramid ----
        pyramid: list[torch.Tensor] = [features]  # P3 = identity (stride-8)
        for down in self._down_modules:
            prev = pyramid[-1]
            prev = _pad_to_even(prev)
            pyramid.append(down(prev))

        # Per-scale LayerNorm
        normed_maps: list[torch.Tensor] = []
        for feat, ln in zip(pyramid, self._norms):
            normed_maps.append(
                ln(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )

        # ---- Stage 2: Predict sampling offsets ----
        raw_offsets = self.offset_pred(features)  # [B, 2·L·K, H, W]
        offsets = raw_offsets.reshape(
            B, self.num_levels, self.num_points, 2, H, W
        ).permute(0, 4, 5, 1, 2, 3)  # [B, H, W, L, K, 2]
        offsets = offsets.tanh() * self.max_offset

        # ---- Stage 3: Sample K/V at deformed positions ----
        # Base reference grid at stride-8, normalised to [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, W, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1)  # [H, W, 2]
        base_grid = base_grid.view(1, H, W, 1, 1, 2)  # [1, H, W, 1, 1, 2]

        # Convert pixel-space offsets (stride-8) to [-1, 1] normalised space
        normaliser = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
            device=features.device,
            dtype=features.dtype,
        ).view(1, 1, 1, 1, 1, 2)

        all_sampled_k: list[torch.Tensor] = []
        all_sampled_v: list[torch.Tensor] = []

        for lvl in range(self.num_levels):
            feat = normed_maps[lvl]

            # Keep the level dimension for proper broadcasting with base_grid.
            # offsets[:,:,:,lvl:lvl+1,:,:]  → [B, H, W, 1, K, 2]
            coords = base_grid + offsets[:, :, :, lvl:lvl + 1, :, :] * normaliser
            coords = coords.clamp(-1.0, 1.0)
            # Squeeze the level dim: [B, H, W, 1, K, 2] → [B, H, W, K, 2]
            coords = coords.squeeze(3)
            coords_flat = coords.reshape(B, N * self.num_points, 1, 2)

            # Project then sample (DAT-style)
            k_map = self.k_proj[lvl](feat)  # [B, C, H_l, W_l]
            v_map = self.v_proj[lvl](feat)  # [B, C, H_l, W_l]

            k_s = F.grid_sample(
                k_map, coords_flat, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            k_s = k_s.squeeze(-1).transpose(1, 2).reshape(B, N, self.num_points, C)

            v_s = F.grid_sample(
                v_map, coords_flat, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            v_s = v_s.squeeze(-1).transpose(1, 2).reshape(B, N, self.num_points, C)

            all_sampled_k.append(k_s)
            all_sampled_v.append(v_s)

        k_all = torch.cat(all_sampled_k, dim=2)  # [B, N, L·K, C]
        v_all = torch.cat(all_sampled_v, dim=2)  # [B, N, L·K, C]

        # ---- Stage 4: Multi-head dot-product attention ----
        p3_flat = features.permute(0, 2, 3, 1).reshape(B, N, C)

        q = self.q_proj(p3_flat).reshape(
            B, N, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B, heads, N, head_dim]

        total_points = self.num_levels * self.num_points
        k = k_all.reshape(
            B, N, total_points, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)  # [B, heads, N, L·K, head_dim]

        v = v_all.reshape(
            B, N, total_points, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)  # [B, heads, N, L·K, head_dim]

        attn = torch.einsum("bhnd,bhnkd->bhnk", q, k) * self.scale

        # Scale-level embedding bias
        level_bias = self.level_embeds.reshape(1, 1, 1, -1)  # [1, 1, 1, L·K]
        attn = attn + level_bias

        # Distance penalty (prefers nearby sampling points)
        distance = torch.linalg.vector_norm(offsets, dim=-1)  # [B, H, W, L, K]
        distance = distance.reshape(B, N, -1)  # [B, N, L·K]
        distance_lambda = self.distance_lambda.clamp_min(0.0)
        attn = attn - distance_lambda * distance.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhnk,bhnkd->bhnd", attn, v)  # [B, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        # ---- Stage 5: FFN + residual ----
        out = p3_flat + self.residual_gate.tanh() * out
        out_norm = self.ffn_norm(out)
        out = out + self.ffn(out_norm)

        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        out = self.se(out)

        return out

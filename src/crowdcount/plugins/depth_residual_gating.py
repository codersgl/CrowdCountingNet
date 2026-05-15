"""Depth Residual Gating: lightweight depth-conditioned feature modulation.

Design C from the depth-guided crowd counting reference architecture.
The module adds a learnable residual path conditioned on depth:

    feat_out = feat + gate * depth_encoder(resize(depth_map))

where ``gate`` is a scalar :class:`nn.Parameter` initialised to **0**.
This ensures the module acts as a pure identity at the start of training,
gradually learning to incorporate depth information as needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthResidualGating(nn.Module):
    """Depth-conditioned residual gating for a single feature scale.

    Args:
        feat_channels: Number of channels in the input feature map (e.g. 256, 512).
        mid_ratio: Reduction ratio for the intermediate channel count.
            ``mid = max(feat_channels // mid_ratio, 32)``.
    """

    def __init__(self, feat_channels: int, mid_ratio: int = 4) -> None:
        super().__init__()
        mid = max(feat_channels // mid_ratio, 32)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
        )

        # Learnable gate initialised to 0 → identity at start of training.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Apply depth-conditioned residual gating.

        Args:
            feat: Feature map ``[B, C, H_f, W_f]``.
            depth_map: Depth / scale map ``[B, 1, H, W]`` (any spatial size).

        Returns:
            Modulated feature ``[B, C, H_f, W_f]``.
        """
        _, _, h_f, w_f = feat.shape
        depth_resized = F.interpolate(
            depth_map, size=(h_f, w_f), mode="bilinear", align_corners=False
        )
        depth_feat = self.depth_encoder(depth_resized)  # [B, C, H_f, W_f]
        return feat + self.gate * depth_feat


class DepthResidualGatingV2(nn.Module):
    """Adaptive depth-conditioned residual gating for a single feature scale.

    Compared with :class:`DepthResidualGating`, this version keeps the same
    identity-at-init behaviour but makes the residual path safer and more
    expressive with depth normalisation, bounded global gating, and optional
    spatial/channel gates.

    Args:
        feat_channels: Number of channels in the input feature map.
        mid_ratio: Reduction ratio for hidden channels.
        gate_init: Initial value for the global residual gate.
        use_tanh_gate: Apply ``tanh`` to the global gate before fusion.
        spatial_gate: Enable a spatial sigmoid gate ``[B, 1, H, W]``.
        channel_gate: Enable a channel sigmoid gate ``[B, C, 1, 1]``.
        normalize_depth: Per-sample depth normalisation after resizing.
        eps: Numerical stability term for depth normalisation.
    """

    def __init__(
        self,
        feat_channels: int,
        mid_ratio: int = 4,
        gate_init: float = 0.0,
        use_tanh_gate: bool = True,
        spatial_gate: bool = True,
        channel_gate: bool = True,
        normalize_depth: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feat_channels <= 0:
            raise ValueError(f"feat_channels must be positive, got {feat_channels}")
        if mid_ratio <= 0:
            raise ValueError(f"mid_ratio must be positive, got {mid_ratio}")

        mid = max(feat_channels // mid_ratio, 32)
        self.feat_channels = int(feat_channels)
        self.use_tanh_gate = bool(use_tanh_gate)
        self.normalize_depth = bool(normalize_depth)
        self.eps = float(eps)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
        )
        self.residual_proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
        )

        self.spatial_gate = (
            nn.Sequential(
                nn.Conv2d(feat_channels * 2, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            if spatial_gate
            else None
        )
        self.channel_gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feat_channels * 2, mid, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, feat_channels, 1, bias=True),
                nn.Sigmoid(),
            )
            if channel_gate
            else None
        )

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, feat: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Apply adaptive depth-conditioned residual gating.

        Args:
            feat: Feature map ``[B, C, H_f, W_f]``.
            depth_map: Depth map ``[B, 1, H, W]`` or ``[B, H, W]``.

        Returns:
            Modulated feature ``[B, C, H_f, W_f]``.
        """
        if feat.dim() != 4:
            raise ValueError(f"feat must be 4D [B, C, H, W], got {tuple(feat.shape)}")
        if feat.shape[1] != self.feat_channels:
            raise ValueError(
                f"feat channel mismatch: expected {self.feat_channels}, got {feat.shape[1]}"
            )

        depth_resized = self._prepare_depth(depth_map, feat)
        depth_feat = self.depth_encoder(depth_resized)
        residual = self.residual_proj(depth_feat)

        fusion_context = torch.cat([feat, depth_feat], dim=1)
        if self.channel_gate is not None:
            residual = residual * self.channel_gate(fusion_context)
        if self.spatial_gate is not None:
            residual = residual * self.spatial_gate(fusion_context)

        gate = self.gate.tanh() if self.use_tanh_gate else self.gate
        return feat + gate.to(dtype=feat.dtype) * residual

    def _prepare_depth(self, depth_map: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        if depth_map.dim() != 4 or depth_map.shape[1] != 1:
            raise ValueError(
                "depth_map must have shape [B, 1, H, W] or [B, H, W], "
                f"got {tuple(depth_map.shape)}"
            )
        if depth_map.shape[0] != feat.shape[0]:
            raise ValueError(
                f"depth batch size ({depth_map.shape[0]}) must match feat batch size ({feat.shape[0]})"
            )

        depth_map = depth_map.to(device=feat.device, dtype=feat.dtype)
        if depth_map.shape[-2:] != feat.shape[-2:]:
            depth_map = F.interpolate(
                depth_map,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.normalize_depth:
            mean = depth_map.mean(dim=(1, 2, 3), keepdim=True)
            var = depth_map.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
            depth_map = (depth_map - mean) / (var + self.eps).sqrt()
        return depth_map

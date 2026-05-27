"""Density regression head and point prediction head for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


class Stride4RefineHead(nn.Module):
    """Coarse-to-fine refinement head: upsamples stride-8 density to stride-4,
    concatenates with backbone stride-4 features, and refines for finer detail.

    Reference: SANet (Cao et al., ECCV 2018), AMSNet (Hu et al., TAAI 2021).
    """

    def __init__(
        self,
        s4_channels: int = 128,
        hidden_channels: int = 64,
        final_activation: str = "softplus",
        initial_density: float = 0.0125,
        final_weight_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if final_activation not in {"relu", "softplus", "none"}:
            raise ValueError("final_activation must be one of: relu, softplus, none")
        self.final_activation = final_activation
        gn1 = min(32, hidden_channels)
        s2 = hidden_channels // 2
        gn2 = min(32, s2)
        self.s4_proj = nn.Conv2d(s4_channels, hidden_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn1, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, s2, kernel_size=3, padding=1),
            nn.GroupNorm(gn2, s2),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(s2, 1, kernel_size=1)
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=final_weight_std)
        if final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, bias_value)

    def forward(
        self, feat_s4: torch.Tensor, density_s8: torch.Tensor,
    ) -> torch.Tensor:
        s4 = self.s4_proj(feat_s4)  # [B, H_hid, H/4, W/4]
        d_up = F.interpolate(
            density_s8,
            size=s4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, H/4, W/4]
        fused = torch.cat([s4, d_up], dim=1)
        refined = self.refine(fused)
        density = self.output_conv(refined)
        if self.final_activation == "relu":
            return F.relu(density)
        if self.final_activation == "softplus":
            return F.softplus(density, beta=1, threshold=20)
        return density


class PointPredHead(nn.Module):
    """Per-pixel point prediction head for auxiliary supervision (P2PNet-style).

    Produces per-pixel classification logits and (dx, dy) offset predictions
    that are compatible with HungarianMatcher_Crowd.

    Reference: Song et al., "Rethinking Counting and Localization in Crowds:
    A Purely Point-Based Framework", ICCV 2021.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        stride: int = 8,
    ) -> None:
        super().__init__()
        self.stride = float(stride)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.reg_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = features.shape
        x = self.trunk(features)
        pred_logits = self.cls_conv(x)  # [B, 2, H, W]
        pred_offsets = self.reg_conv(x).tanh() * self.stride  # [B, 2, H, W]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=features.device, dtype=features.dtype),
            torch.arange(w, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        grid_centers = torch.stack([
            (grid_x + 0.5) * self.stride,
            (grid_y + 0.5) * self.stride,
        ], dim=0)  # [2, H, W]

        pred_logits_flat = pred_logits.reshape(b, 2, h * w).transpose(1, 2)  # [B, N, 2]
        pred_offsets_flat = pred_offsets.reshape(b, 2, h * w).transpose(1, 2)  # [B, N, 2]
        grid_flat = grid_centers.reshape(2, h * w).t().unsqueeze(0)  # [1, N, 2]

        return {
            "pred_logits": pred_logits_flat,
            "pred_points": grid_flat + pred_offsets_flat,
            "pred_offsets": pred_offsets_flat,
        }


class DenseASPPScaleHead(nn.Module):
    """DenseASPP density head with dilations [1, 3, 6, 9] for multi-scale context.

    Dense connectivity (Yang et al., CVPR 2018) lets each dilation branch see
    all previous branches' outputs, creating an exponentially growing receptive
    field without large kernels.  At stride-8 the effective kernel covers
    ~152×152 pixels — enough for large heads in sparse crowds.

    Reference: CCTrans (Tian et al., 2021) shows that multi-scale dilated
    convolutions in the regression head are critical for crowd counting.
    """

    def __init__(
        self,
        in_channels: int = 256,
        branch_channels: int = 64,
        dilations: tuple[int, ...] = (1, 3, 6, 9),
        final_activation: str = "softplus",
        initial_density: float = 0.05,
        final_weight_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if final_activation not in {"relu", "softplus", "none"}:
            raise ValueError("final_activation must be one of: relu, softplus, none")
        self.final_activation = final_activation
        self.dilations = tuple(dilations)
        self.branch_channels = int(branch_channels)
        num_branches = len(self.dilations)

        # Dense ASPP branches — each receives input + all previous branch outputs
        self.branches = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            in_ch = in_channels + i * branch_channels
            gn_groups = min(32, branch_channels)
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, branch_channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.GroupNorm(gn_groups, branch_channels),
                nn.ReLU(inplace=True),
            ))

        # Fusion: compress all branch outputs + input
        total_branch_out = in_channels + num_branches * branch_channels
        fuse_hidden = min(branch_channels * num_branches, 256)
        gn_fuse = min(32, fuse_hidden)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_branch_out, fuse_hidden, kernel_size=1),
            nn.GroupNorm(gn_fuse, fuse_hidden),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(fuse_hidden, 1, kernel_size=1)
        self._init_final_layer(initial_density, final_weight_std)

    def _init_final_layer(self, initial_density: float, final_weight_std: float) -> None:
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=final_weight_std)
        if self.final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, bias_value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outs = [features]
        for branch in self.branches:
            inp = torch.cat(outs, dim=1)
            outs.append(branch(inp))
        fused = self.fuse(torch.cat(outs, dim=1))
        density = self.output_conv(fused)
        if self.final_activation == "relu":
            return F.relu(density)
        if self.final_activation == "softplus":
            return F.softplus(density, beta=1, threshold=20)
        return density


class DensityHead(nn.Module):
    """Deeper density head with three Conv3x3 layers for richer feature processing."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        final_activation: str = "softplus",
        initial_density: float = 0.05,
        final_weight_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if final_activation not in {"relu", "softplus", "none"}:
            raise ValueError("final_activation must be one of: relu, softplus, none")
        if initial_density <= 0:
            raise ValueError("initial_density must be > 0")
        if final_weight_std < 0:
            raise ValueError("final_weight_std must be >= 0")
        self.final_activation = final_activation
        gn1 = min(32, hidden_channels)
        s2_channels = hidden_channels // 2
        gn2 = min(32, s2_channels)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn1, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_channels, s2_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn2, s2_channels),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        self._init_final_layer(initial_density, final_weight_std)

    def _init_final_layer(self, initial_density: float, final_weight_std: float) -> None:
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=final_weight_std)
        if self.final_activation == "softplus":
            bias_value = _softplus_inverse(initial_density)
        else:
            bias_value = initial_density
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, bias_value)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stage1(features)
        x = self.stage2(x)
        density = self.output_conv(x)
        if self.final_activation == "relu":
            return F.relu(density)
        if self.final_activation == "softplus":
            return F.softplus(density, beta=1, threshold=20)
        return density

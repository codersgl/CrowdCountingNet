"""Heterogeneous experts for MoECountNet — Scale × Paradigm dual-axis specialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.deformable_expert import DeformableCrossScaleExpert
from crowdcount.models.moecount.gate import SparseTop2Gate
from crowdcount.models.moecount.losses import ExpertImportanceLoss
from crowdcount.models.neck import SPD


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SharedExpert(nn.Module):
    """Minimal shared expert — single conv, forcing routed experts to specialize."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LocalDetailExpert(nn.Module):
    """Stride-8 expert: depthwise conv + channel attention for fine local patterns."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.se = SE(channels, reduction=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.dwconv(features)
        out = self.se(out)
        out = self.fuse(out)
        return features + self.residual_gate.tanh() * out


class SpatialRelationExpert(nn.Module):
    """Stride-16 expert: window self-attention for spatial relation modeling.

    SPD downsampling to stride-16 → Window-MSA (8×8 windows) → FFN → bilinear up.
    """

    def __init__(self, channels: int = 256, num_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

        self.spd_down = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
        )

    def _window_partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, nH, nW, ws, ws, C]
        x = x.reshape(B, (Hp // ws) * (Wp // ws), ws * ws, C)
        return x, H, W, pad_h, pad_w

    def _window_unpartition(self, x: torch.Tensor, H: int, W: int, pad_h: int, pad_w: int) -> torch.Tensor:
        ws = self.window_size
        B, nw, _, C = x.shape
        nH = (H + pad_h) // ws
        nW = (W + pad_w) // ws
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, nH, ws, nW, ws]
        x = x.reshape(B, C, H + pad_h, W + pad_w)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = features
        x = self.spd_down(features)  # stride-8 → stride-16
        B, C, H, W = x.shape

        # Window MSA
        x_windowed, H_orig, W_orig, pad_h, pad_w = self._window_partition(x)
        B, nw, N, C_ = x_windowed.shape
        Bnw = B * nw
        x_flat = x_windowed.reshape(Bnw * N, C_)
        x_ln = self.norm1(x_flat).reshape(Bnw, N, C_)

        qkv = self.qkv(x_ln).reshape(Bnw, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(Bnw, N, C)
        attn_out = self.proj(attn_out)

        x_windowed = x_windowed + attn_out.reshape(B, nw, N, C)
        x_unflat = self._window_unpartition(x_windowed, H_orig, W_orig, pad_h, pad_w)

        # FFN
        x_perm = x_unflat.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_ffn = self.norm2(x_perm)
        x_ffn = self.ffn(x_ffn) + x_perm
        x_out = x_ffn.permute(0, 3, 1, 2)

        # Upsample back to stride-8
        return F.interpolate(x_out, size=identity.shape[-2:], mode="bilinear", align_corners=False)


class GlobalDensityExpert(nn.Module):
    """Stride-32 expert: large-kernel conv + channel attention for global density context.

    SPD×2 downsampling to stride-32 → Conv7×7 DW + SE + Conv1×1 → bilinear up.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.spd_down = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.spd_down2 = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.large_kernel = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.se = SE(channels, reduction=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity_size = features.shape[-2:]
        x = self.spd_down(features)   # s8 → s16
        x = self.spd_down2(x)         # s16 → s32
        x = self.large_kernel(x)
        x = self.se(x)
        x = self.fuse(x)
        out = F.interpolate(x, size=identity_size, mode="bilinear", align_corners=False)
        return features + self.residual_gate.tanh() * out


class HeterogeneousSparseMoE(nn.Module):
    """Three scale×paradigm heterogeneous experts with pixel-wise soft gating.

    Uses HMoDE-style per-pixel softmax routing (Du et al., IEEE TIP 2023)
    instead of hard Top-K selection. All experts always contribute with
    learned spatial weights, preventing expert collapse.
    """

    def __init__(
        self,
        channels: int = 256,
        gate_hidden_channels: int = 128,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.98,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        lambda_importance: float = 0.01,
        lambda_load: float = 0.01,
        shared_scale: float = 0.3,
        use_deformable_expert: bool = False,
        deformable_num_heads: int = 4,
        deformable_num_sampling_points: int = 8,
        deformable_num_scale_levels: int = 3,
        deformable_max_offset: float = 8.0,
        deformable_dropout: float = 0.1,
        deformable_use_se: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = 3
        self.shared_scale = float(shared_scale)
        self.shared_expert = SharedExpert(channels)
        spatial_expert: nn.Module
        if use_deformable_expert:
            spatial_expert = DeformableCrossScaleExpert(
                channels=channels,
                num_heads=deformable_num_heads,
                num_sampling_points=deformable_num_sampling_points,
                num_scale_levels=deformable_num_scale_levels,
                max_offset=deformable_max_offset,
                dropout=deformable_dropout,
                use_se=deformable_use_se,
            )
        else:
            spatial_expert = SpatialRelationExpert(channels)
        self.experts = nn.ModuleList([
            LocalDetailExpert(channels),
            spatial_expert,
            GlobalDensityExpert(channels),
        ])
        self.gate = SparseTop2Gate(
            in_channels=channels,
            num_experts=self.num_experts,
            hidden_channels=gate_hidden_channels,
            top_k=top_k,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            warmup_fraction=warmup_fraction,
            warmup_epochs=warmup_epochs,
        )
        self.eim_loss = ExpertImportanceLoss(
            lambda_importance=lambda_importance,
        )
        self.output_norm = nn.GroupNorm(32, channels)

    @property
    def temperature(self) -> float:
        return self.gate.temperature

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.gate.set_epoch(epoch, total_epochs)

    def update_temperature(self, decay_rate: float | None = None) -> None:
        self.gate.update_temperature(decay_rate)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        shared_out = self.shared_expert(features) * self.shared_scale
        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )  # [B, 3, C, H/8, W/8]

        with torch.no_grad():
            eo = expert_outputs.detach()
            eo_flat = eo.reshape(eo.shape[0], 3, -1)
            eo_norm = F.normalize(eo_flat, dim=-1)
            cos_matrix = torch.bmm(eo_norm, eo_norm.transpose(1, 2))
            avg_cos = cos_matrix.mean(0)
            expert_similarity = {
                "cos_01": avg_cos[0, 1].clone(),
                "cos_02": avg_cos[0, 2].clone(),
                "cos_12": avg_cos[1, 2].clone(),
            }

        route = self.gate(features)
        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        routed = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        fused = self.output_norm(shared_out + routed)

        soft_probs = route["soft_probs"]
        if not isinstance(soft_probs, torch.Tensor):
            raise TypeError("gate soft probs must be a tensor")
        aux_losses = self.eim_loss(soft_probs) if self.training else {}
        route["expert_similarity"] = expert_similarity
        return fused, aux_losses, route

"""Heterogeneous experts for MoECountNet — Scale × Paradigm dual-axis specialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.gate import MultiScaleSparseTop2Gate
from crowdcount.models.moecount.losses import LoadBalanceLoss
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
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.dwconv(features)
        out = self.se(out)
        return self.fuse(out)


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity_size = features.shape[-2:]
        x = self.spd_down(features)   # s8 → s16
        x = self.spd_down2(x)         # s16 → s32
        x = self.large_kernel(x)
        x = self.se(x)
        x = self.fuse(x)
        return F.interpolate(x, size=identity_size, mode="bilinear", align_corners=False)


class HeterogeneousSparseMoE(nn.Module):
    """Three scale×paradigm heterogeneous experts + multi-scale gate routing."""

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
    ) -> None:
        super().__init__()
        self.num_experts = 3
        self.shared_scale = float(shared_scale)
        self.shared_expert = SharedExpert(channels)
        self.experts = nn.ModuleList([
            LocalDetailExpert(channels),
            SpatialRelationExpert(channels),
            GlobalDensityExpert(channels),
        ])
        self.gate = MultiScaleSparseTop2Gate(
            in_channels=channels,
            hidden_channels=gate_hidden_channels,
            num_experts=self.num_experts,
            top_k=top_k,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            warmup_fraction=warmup_fraction,
            warmup_epochs=warmup_epochs,
        )
        self.balance_loss = LoadBalanceLoss(
            lambda_importance=lambda_importance,
            lambda_load=lambda_load,
        )
        self.output_norm = nn.GroupNorm(32, channels)

    @property
    def temperature(self) -> float:
        return float(self.gate.temperature)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.gate.set_epoch(epoch, total_epochs=total_epochs)

    def update_temperature(self, decay_rate: float | None = None) -> None:
        self.gate.update_temperature(decay_rate=decay_rate)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        shared_out = self.shared_expert(features) * self.shared_scale
        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )  # [B, 3, C, H/8, W/8]

        route = self.gate(features)
        if self.training:
            load_fraction = route["load_fraction"]
            if isinstance(load_fraction, torch.Tensor):
                self.gate.update_expert_bias(load_fraction)

        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        routed = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        fused = self.output_norm(shared_out + routed)

        soft_probs = route["soft_probs"]
        hard_mask = route["hard_mask"]
        if not isinstance(soft_probs, torch.Tensor) or not isinstance(hard_mask, torch.Tensor):
            raise TypeError("gate probabilities and hard mask must be tensors")
        aux_losses = self.balance_loss(soft_probs, hard_mask) if self.training else {}
        return fused, aux_losses, route

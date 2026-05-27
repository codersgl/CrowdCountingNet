"""Heterogeneous experts for MoECountNet — Scale × Paradigm dual-axis specialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.gate import MultiScaleSparseTop2Gate, PixelSoftGate
from crowdcount.models.moecount.losses import ExpertImportanceLoss


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

    Input is already stride-16 (from neck p4). Window-MSA (8×8 windows) → FFN → bilinear up to stride-8.
    """

    def __init__(self, channels: int = 256, num_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

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

    def forward(self, features: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        x = features  # input is already stride-16
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

        # Upsample to target stride-8 resolution
        if target_size is None:
            target_size = (x_out.shape[-2] * 2, x_out.shape[-1] * 2)
        return F.interpolate(x_out, size=target_size, mode="bilinear", align_corners=False)


class GlobalDensityExpert(nn.Module):
    """Stride-32 expert with true global context via GAP + channel modulation.

    Input is already stride-32 (from neck p5). Conv7×7 DW + SE + Conv1×1, modulated
    by global-average-pooled context (SENet-style channel gate).

    On a full image the global pooling looks at the *entire* scene, providing
    true global density awareness that the patch-trained gate cannot get from
    local features alone.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.large_kernel = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.se = SE(channels, reduction=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(8, channels // 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, channels // 16), channels),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        x = features  # input is already stride-32

        gate = self.global_mlp(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        x = self.large_kernel(x)
        x = self.se(x)
        x = self.fuse(x)
        x = x * gate  # modulate by global context
        if target_size is None:
            target_size = (x.shape[-2] * 4, x.shape[-1] * 4)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class HeterogeneousSparseMoE(nn.Module):
    """Three scale×paradigm heterogeneous experts with per-scale input routing.

    Each expert receives features at its native scale:
      - LocalDetailExpert  ← stride-8 (neck p3 / fused)
      - SpatialRelationExpert ← stride-16 (neck p4)
      - GlobalDensityExpert   ← stride-32 (neck p5)

    Uses HMoDE-style per-pixel softmax routing (Du et al., IEEE TIP 2023)
    on the stride-8 fused feature. Per-expert density heads provide deep
    supervision that forces each expert to learn independent predictions.
    """

    def __init__(
        self,
        channels: int = 256,
        gate_type: str = "soft",
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
        gate_kwargs = dict(
            in_channels=channels,
            hidden_channels=gate_hidden_channels,
            num_experts=self.num_experts,
        )
        if gate_type == "sparse_top2":
            self.gate = MultiScaleSparseTop2Gate(
                **gate_kwargs,
                top_k=int(top_k),
                temperature_init=float(temperature_init),
                temperature_min=float(temperature_min),
                temperature_decay=float(temperature_decay),
                warmup_fraction=float(warmup_fraction),
                warmup_epochs=warmup_epochs,
            )
        else:
            self.gate = PixelSoftGate(**gate_kwargs)
        self.eim_loss = ExpertImportanceLoss(
            lambda_importance=lambda_importance,
        )
        self.output_norm = nn.GroupNorm(32, channels)

        # Per-expert density heads for deep supervision
        expert_head0 = nn.Conv2d(channels, 1, kernel_size=1)
        expert_head1 = nn.Conv2d(channels, 1, kernel_size=1)
        expert_head2 = nn.Conv2d(channels, 1, kernel_size=1)
        self.expert_density_heads = nn.ModuleList([expert_head0, expert_head1, expert_head2])
        # Initialize per-expert heads with small weights for stability
        for head in [expert_head0, expert_head1, expert_head2]:
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)
            if head.bias is not None:
                nn.init.constant_(head.bias, -2.0)

    @property
    def temperature(self) -> float:
        return float(getattr(self.gate, "temperature", 1.0))

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        if hasattr(self.gate, "set_epoch"):
            self.gate.set_epoch(epoch, total_epochs)

    def update_temperature(self, decay_rate: float | None = None) -> None:
        if hasattr(self.gate, "update_temperature"):
            self.gate.update_temperature(decay_rate)

    def forward(
        self, feat_s8: torch.Tensor, feat_s16: torch.Tensor, feat_s32: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        target_size = (feat_s8.shape[-2], feat_s8.shape[-1])  # stride-8 resolution

        shared_out = self.shared_expert(feat_s8) * self.shared_scale
        expert_outputs = torch.stack([
            self.experts[0](feat_s8),                   # LocalDetail: stride-8 → stride-8
            self.experts[1](feat_s16, target_size),     # SpatialRelation: stride-16 → stride-8
            self.experts[2](feat_s32, target_size),     # GlobalDensity: stride-32 → stride-8
        ], dim=1)  # [B, 3, C, H/8, W/8]

        # Compute cosine similarity in activation space (trainable during training,
        # detached during eval for logging only)
        eo = expert_outputs if self.training else expert_outputs.detach()
        eo_flat = eo.reshape(eo.shape[0], 3, -1)
        eo_norm = F.normalize(eo_flat, dim=-1)
        cos_matrix = torch.bmm(eo_norm, eo_norm.transpose(1, 2))
        avg_cos = cos_matrix.mean(0)
        expert_similarity = {
            "cos_01": avg_cos[0, 1],
            "cos_02": avg_cos[0, 2],
            "cos_12": avg_cos[1, 2],
        }

        route = self.gate(feat_s8, feat_s16, feat_s32) if isinstance(self.gate, MultiScaleSparseTop2Gate) else self.gate(feat_s8)
        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        routed = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        fused = self.output_norm(shared_out + routed)

        # Per-expert density maps for deep supervision
        expert_densities = []
        for i, head in enumerate(self.expert_density_heads):
            eo_i = eo[:, i]  # [B, C, H/8, W/8]
            head_out: torch.Tensor = head(eo_i)  # type: ignore[assignment]
            expert_densities.append(F.softplus(head_out, beta=1, threshold=20))
        expert_densities_out = torch.stack(expert_densities, dim=1)  # [B, 3, 1, H/8, W/8]

        soft_probs = route["soft_probs"]
        if not isinstance(soft_probs, torch.Tensor):
            raise TypeError("gate soft probs must be a tensor")
        aux_losses = self.eim_loss(soft_probs) if self.training else {}
        route["expert_similarity"] = expert_similarity  # with grad during training for diversity loss
        route["expert_densities"] = expert_densities_out
        return fused, aux_losses, route

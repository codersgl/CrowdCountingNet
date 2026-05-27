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


class MultiSpectralChannelAttention(nn.Module):
    """FcaNet-style multi-spectral channel attention via 2D DCT bases.

    Replaces the single GAP scalar (DC component only) with K different
    2D DCT frequency components, enriching channel descriptors with
    multi-frequency texture information at near-zero parameter cost.

    Ref: Qin et al., "FcaNet: Frequency Channel Attention Networks", ICCV 2021.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        num_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        dct_basis = self._build_dct_basis(num_freqs, channels)
        self.register_buffer("dct_basis", dct_basis)  # [C, K, 1, 1]
        self.fc = nn.Sequential(
            nn.Conv2d(channels * num_freqs, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_dct_basis(
        num_freqs: int, channels: int, base_h: int = 8, base_w: int = 8
    ) -> torch.Tensor:
        """Pre-compute K different 2D DCT basis vectors as constant buffers.

        Uses the standard DCT-II formulation where the (u,v) basis image is:

            B_{u,v}(i,j) = cos(pi*u*(i+0.5)/H) * cos(pi*v*(j+0.5)/W)

        The first basis (u=0, v=0) recovers GAP — all coefficients equal.
        Subsequent bases capture progressively higher spatial frequencies.
        """
        basis_list = []
        # Select K DCT frequency pairs distributed across the spectrum.
        # Strategy: grid-scan low-frequency region first, then extend.
        freq_pairs: list[tuple[int, int]] = [(0, 0)]  # DC = GAP
        for d in range(1, num_freqs):
            # Zigzag-like: alternate between adding horizontal and vertical freqs
            if d % 2 == 1:
                freq_pairs.append((d // 2 + 1, 0))
            else:
                freq_pairs.append((0, d // 2))
        freq_pairs = freq_pairs[:num_freqs]

        i = torch.arange(base_h, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        j = torch.arange(base_w, dtype=torch.float32).unsqueeze(0)  # [1, W]

        for u, v in freq_pairs:
            basis_u = torch.cos(torch.pi * u * (i + 0.5) / base_h)  # [H, 1]
            basis_v = torch.cos(torch.pi * v * (j + 0.5) / base_w)  # [1, W]
            basis_2d = basis_u @ basis_v  # [H, W]
            basis_2d = basis_2d / basis_2d.abs().sum().clamp_min(1e-8)  # L1-normalize
            basis_list.append(basis_2d)

        # Stack → [K, H, W], then expand channel dimension: treat equally per channel
        basis_stack = torch.stack(basis_list)  # [K, H, W]
        basis_stack = basis_stack.unsqueeze(0).expand(channels, -1, -1, -1)  # [C, K, H, W]
        return basis_stack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Resize pre-computed bases to match input spatial size
        dct_basis = self.dct_basis  # [C, K, base_h, base_w]
        if H != dct_basis.shape[-2] or W != dct_basis.shape[-1]:
            dct_basis = (
                F.interpolate(
                    dct_basis.flatten(0, 1).unsqueeze(0),  # [1, C*K, base_h, base_w]
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .view(C, self.num_freqs, H, W)
            )
        # Compress each channel with K different DCT bases: [B, C, K]
        freq_components = (x.unsqueeze(2) * dct_basis.unsqueeze(0)).sum(dim=[-2, -1])  # [B, C, K]
        # Flatten to [B, C*K, 1, 1] for FC processing
        freq_flat = freq_components.view(B, C * self.num_freqs, 1, 1)
        attn = self.fc(freq_flat)  # [B, C, 1, 1]
        return x * attn


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
    """Stride-8 expert: multi-branch strip convs + multi-spectral channel attention.

    Three parallel depthwise branches capture local patterns at different
    aspect ratios (3x3 square, 1xK horizontal strip, Kx1 vertical strip),
    followed by FcaNet-style multi-spectral channel attention that enriches
    the channel descriptor with K DCT frequency components beyond GAP's DC.

    Ref:
      - SPCANet (Yuan, PeerJ CS 2024): Strip Pooling for crowd counting
      - FcaNet (Qin et al., ICCV 2021): Multi-spectral channel attention
    """

    def __init__(
        self,
        channels: int = 256,
        use_residual: bool = True,
        use_strip_convs: bool = True,
        strip_kernel: int = 7,
        use_multi_spectral_se: bool = True,
        ms_num_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_strip_convs = use_strip_convs

        # --- Depthwise spatial branches ---
        self.dwconv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        if use_strip_convs:
            self.dwconv_1xK = nn.Conv2d(
                channels, channels,
                kernel_size=(1, strip_kernel),
                padding=(0, strip_kernel // 2),
                groups=channels,
            )
            self.dwconv_Kx1 = nn.Conv2d(
                channels, channels,
                kernel_size=(strip_kernel, 1),
                padding=(strip_kernel // 2, 0),
                groups=channels,
            )
            fuse_in = channels * 3
            # Per-branch gate scalars (initialized to soft-contribute)
            self.branch_gate_3x3 = nn.Parameter(torch.ones(1) * 0.5)
            self.branch_gate_1xK = nn.Parameter(torch.ones(1) * 0.25)
            self.branch_gate_Kx1 = nn.Parameter(torch.ones(1) * 0.25)
        else:
            fuse_in = channels
        self.fuse_strips = nn.Sequential(
            nn.Conv2d(fuse_in, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

        # --- Channel attention ---
        if use_multi_spectral_se:
            self.channel_attn = MultiSpectralChannelAttention(
                channels, reduction=4, num_freqs=ms_num_freqs,
            )
        else:
            self.channel_attn = SE(channels, reduction=4)

        # --- Output projection ---
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        if use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out_3x3 = self.dwconv_3x3(features)
        if self.use_strip_convs:
            g3 = self.branch_gate_3x3.tanh()
            gh = self.branch_gate_1xK.tanh()
            gv = self.branch_gate_Kx1.tanh()
            out_1xK = self.dwconv_1xK(features)
            out_Kx1 = self.dwconv_Kx1(features)
            out = torch.cat([g3 * out_3x3, gh * out_1xK, gv * out_Kx1], dim=1)
        else:
            out = out_3x3
        out = self.fuse_strips(out)
        out = self.channel_attn(out)
        out = self.fuse(out)
        if self.use_residual:
            return features + self.residual_gate.tanh() * out
        return out


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

    def __init__(self, channels: int = 256, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
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
        if use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity_size = features.shape[-2:]
        x = self.spd_down(features)   # s8 → s16
        x = self.spd_down2(x)         # s16 → s32
        x = self.large_kernel(x)
        x = self.se(x)
        x = self.fuse(x)
        out = F.interpolate(x, size=identity_size, mode="bilinear", align_corners=False)
        if self.use_residual:
            return features + self.residual_gate.tanh() * out
        return out


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
        use_input_residual: bool = True,
        expert_local_detail_use_residual: bool = True,
        expert_global_density_use_residual: bool = True,
        expert_local_detail_use_strip_convs: bool = True,
        expert_local_detail_strip_kernel: int = 7,
        expert_local_detail_use_multi_spectral_se: bool = True,
        expert_local_detail_ms_num_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.num_experts = 3
        self.shared_scale = float(shared_scale)
        self.use_input_residual = use_input_residual
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
            LocalDetailExpert(
                channels,
                use_residual=expert_local_detail_use_residual,
                use_strip_convs=expert_local_detail_use_strip_convs,
                strip_kernel=expert_local_detail_strip_kernel,
                use_multi_spectral_se=expert_local_detail_use_multi_spectral_se,
                ms_num_freqs=expert_local_detail_ms_num_freqs,
            ),
            spatial_expert,
            GlobalDensityExpert(channels, use_residual=expert_global_density_use_residual),
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
        if self.use_input_residual:
            fused = self.output_norm(features + shared_out + routed)
        else:
            fused = self.output_norm(shared_out + routed)

        soft_probs = route["soft_probs"]
        if not isinstance(soft_probs, torch.Tensor):
            raise TypeError("gate soft probs must be a tensor")
        aux_losses = self.eim_loss(soft_probs) if self.training else {}
        route["expert_similarity"] = expert_similarity
        return fused, aux_losses, route

"""DAP-Neck: Density-Aware Phase-guided Neck for crowd counting.

Three core modules:
- PEEM: Phase-aware Edge Enhancement Module (frequency-domain decomposition)
- DPGA: Density Prior Guided Attention (Gaussian-prior cross-scale attention)
- ACDR: Adaptive Crowdedness Dynamic Router (density-adaptive dual-path routing)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowdcount.models.neck import DeformConv2dBNReLU


# ---------------------------------------------------------------------------
# PEEM: Phase-aware Edge Enhancement Module
# ---------------------------------------------------------------------------


class PEEM(nn.Module):
    """Frequency-domain decomposition with deformable high-freq alignment.

    Splits input features via FFT into low-freq (semantic) and high-freq (edge)
    components, processes them independently, then fuses via residual addition.

    Args:
        channels: Number of input/output channels.
        freq_cutoff: Low-pass Gaussian cutoff ratio (0, 1]. Lower = more aggressive
            low-pass filtering, keeping less in the low-freq branch.
        use_dcn: Use deformable conv for the high-freq branch.
    """

    def __init__(
        self,
        channels: int,
        freq_cutoff: float = 0.25,
        use_dcn: bool = True,
    ) -> None:
        super().__init__()
        self.freq_cutoff = freq_cutoff

        # Low-freq branch: 1×1 conv for semantic compression
        self.low_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # High-freq branch: deformable conv for dynamic edge alignment
        if use_dcn:
            self.high_conv: nn.Module = DeformConv2dBNReLU(
                channels, channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.high_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        # Learnable fusion weight (initialised to equal blend)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def _build_gaussian_mask(
        self, h: int, w_rfft: int, device: torch.device
    ) -> torch.Tensor:
        """Build a Gaussian low-pass mask in rfft2 frequency space.

        Returns shape [1, 1, h, w_rfft] with values in [0, 1].
        """
        # Frequency coordinates normalised to [-0.5, 0.5]
        freq_y = torch.fft.fftfreq(h, device=device)  # [h]
        freq_x = torch.linspace(0, 0.5, w_rfft, device=device)  # [w_rfft]

        grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
        dist_sq = grid_y**2 + grid_x**2

        sigma = self.freq_cutoff / 2.0  # sigma controls the Gaussian width
        mask = torch.exp(-dist_sq / (2 * sigma**2 + 1e-8))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w_rfft]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FFT decomposition
        x_freq = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]
        w_rfft = x_freq.shape[-1]

        low_mask = self._build_gaussian_mask(H, w_rfft, x.device)
        high_mask = 1.0 - low_mask

        x_low = torch.fft.irfft2(x_freq * low_mask, s=(H, W), norm="ortho")
        x_high = torch.fft.irfft2(x_freq * high_mask, s=(H, W), norm="ortho")

        # Process branches
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        # Weighted residual fusion
        alpha = self.fusion_alpha.sigmoid()
        return x + alpha * x_low + (1 - alpha) * x_high


# ---------------------------------------------------------------------------
# DPGA: Density Prior Guided Attention
# ---------------------------------------------------------------------------


class DPGA(nn.Module):
    """Cross-scale attention with Gaussian prior position bias.

    Deep features (Query) attend to shallow features (Key/Value) with
    multi-sigma Gaussian templates as additive attention bias.

    To remain memory-safe at arbitrary input resolutions (evaluation uses
    full-size images, not 128×128 patches), both Q and KV are adaptively
    pooled to at most ``max_pool_size`` spatial dimensions before computing
    attention.  The resulting modulation map is then bilinearly upsampled
    back to the original query resolution.

    Args:
        dim: Feature channel dimension.
        num_heads: Number of attention heads.
        sigma_list: List of Gaussian sigma values for position bias templates.
        max_pool_size: Maximum spatial dimension (H or W) for the attention
            computation.  Memory cost is bounded by O(max_pool_size⁴).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        sigma_list: list[float] | None = None,
        max_pool_size: int = 32,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim={dim} must be divisible by num_heads={num_heads}"
            raise ValueError(msg)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_pool_size = max_pool_size
        self.sigma_list = sigma_list or [1.0, 2.0, 4.0]

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # Learnable weights for blending multiple Gaussian bias templates
        self.sigma_weights = nn.Parameter(
            torch.ones(len(self.sigma_list)) / len(self.sigma_list)
        )

        # Residual gate (initialised near identity)
        self.gate = nn.Parameter(torch.zeros(1))

    def _build_gaussian_bias(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """Build weighted sum of Gaussian relative position bias maps.

        Returns shape [1, 1, h*w, h*w] to broadcast over batch and heads.
        """
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # [hw, 2]

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=-1)  # [hw, hw]

        weights = self.sigma_weights.softmax(dim=0)
        bias = torch.zeros_like(dist_sq)
        for i, sigma in enumerate(self.sigma_list):
            bias = bias + weights[i] * torch.exp(-dist_sq / (2 * sigma**2 + 1e-8))

        return bias.unsqueeze(0).unsqueeze(0)  # [1, 1, hw, hw]

    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        """Cross-scale attention: query_feat attends to kv_feat.

        Both inputs must have the same spatial dimensions (caller handles
        upsampling of the deeper feature to match the shallower one).

        Args:
            query_feat: Deep features [B, C, H, W] (upsampled to match kv_feat).
            kv_feat: Shallow features [B, C, H, W].

        Returns:
            Attended features [B, C, H, W] with residual connection to query_feat.
        """
        B, C, H, W = query_feat.shape

        # Determine pooled spatial size — bounded by max_pool_size
        pool_h = min(H, self.max_pool_size)
        pool_w = min(W, self.max_pool_size)
        need_pool = (H > pool_h) or (W > pool_w)

        # Pool Q and KV to bounded spatial size for memory-safe attention
        if need_pool:
            q_input = F.adaptive_avg_pool2d(query_feat, (pool_h, pool_w))
            kv_input = F.adaptive_avg_pool2d(kv_feat, (pool_h, pool_w))
        else:
            q_input = query_feat
            kv_input = kv_feat

        q = self.q_proj(q_input)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        Nq = pool_h * pool_w
        Nk = pool_h * pool_w

        # Reshape to multi-head: [B, heads, head_dim, N]
        q = q.reshape(B, self.num_heads, self.head_dim, Nq)
        k = k.reshape(B, self.num_heads, self.head_dim, Nk)
        v = v.reshape(B, self.num_heads, self.head_dim, Nk)

        # Attention: [B, heads, Nq, Nk]
        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale

        # Add Gaussian position bias (always feasible since pool size is bounded)
        gauss_bias = self._build_gaussian_bias(pool_h, pool_w, query_feat.device)
        attn = attn + gauss_bias

        attn = attn.softmax(dim=-1)

        # Aggregate: [B, heads, Nq, head_dim]
        out = torch.matmul(attn, v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(B, C, pool_h, pool_w)

        out = self.out_proj(out)

        # Upsample modulation map back to original resolution
        if need_pool:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        # Gated residual (gate starts at 0 → identity at init)
        return query_feat + self.gate.tanh() * out


# ---------------------------------------------------------------------------
# ACDR: Adaptive Crowdedness Dynamic Router
# ---------------------------------------------------------------------------


class ACDR(nn.Module):
    """Density-adaptive dual-path feature routing.

    Estimates a per-image crowdedness scalar via GAP + MLP, then blends
    a local sharpening path (small kernel) with a contextual path (large
    dilated kernel) based on the estimated crowdedness.

    Args:
        channels: Number of input/output channels.
        large_kernel: Kernel size for the context path (Path B).
        dilation: Dilation rate for the context path.
        hidden_ratio: Hidden dim ratio for the crowdedness estimator MLP.
    """

    def __init__(
        self,
        channels: int = 256,
        large_kernel: int = 7,
        dilation: int = 2,
        hidden_ratio: int = 4,
    ) -> None:
        super().__init__()

        # Crowdedness estimator: GAP → MLP → sigmoid
        hidden = channels // hidden_ratio
        self.crowd_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # Path A: Local sharpening (3×3 depthwise separable conv)
        self.path_a = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Path B: Context texture (large dilated depthwise conv)
        pad = (large_kernel + (large_kernel - 1) * (dilation - 1) - 1) // 2
        self.path_b = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=large_kernel,
                padding=pad,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.crowd_est(x)  # [B, 1]
        c = c.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        fa = self.path_a(x)
        fb = self.path_b(x)

        return (1 - c) * fa + c * fb


# ---------------------------------------------------------------------------
# PixelShuffleUpsample: content-aware upsampling (2×)
# ---------------------------------------------------------------------------


class PixelShuffleUpsample(nn.Module):
    """1×1 conv to expand channels by 4× then PixelShuffle(2) for 2× spatial upsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(channels)  # BN after shuffle for correct statistics
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.ps(self.conv(x))))


# ---------------------------------------------------------------------------
# DAPNeck: Full Density-Aware Phase-guided Neck
# ---------------------------------------------------------------------------


class DAPNeck(nn.Module):
    """Density-Aware Phase-guided Neck.

    End-to-end neck module that replaces PA-FPN. Accepts multi-scale backbone
    features [C3, C4, C5] and outputs a single fused feature map at C4's
    spatial resolution (H/8).

    Pipeline:
        1. PEEM on C3/C4 (skip C5 by default — too small for meaningful FFT)
        2. Channel alignment: 1×1 conv to project all scales to ``feature_size``
        3. Top-down fusion with DPGA: P5↑ + P4 → E4; E4↑ + P3 → E3
        4. Optional bottom-up: E3↓ + E4 → E4'
        5. Downsample E3 to E4 spatial size, fuse with E4
        6. ACDR: adaptive density-aware routing on the fused output

    Args:
        C3_size: Input channels for C3 (typically 256).
        C4_size: Input channels for C4 (typically 512).
        C5_size: Input channels for C5 (typically 512).
        feature_size: Internal / output channel dimension.
        freq_cutoff: PEEM low-pass Gaussian cutoff ratio.
        peem_on_c5: Whether to apply PEEM on C5 (small spatial size).
        num_heads: Number of attention heads for DPGA.
        sigma_list: Gaussian sigma values for DPGA position bias.
        dpga_max_pool_size: Max spatial dim for DPGA attention (bounds memory).
        acdr_large_kernel: Large kernel size for ACDR Path B.
        acdr_dilation: Dilation rate for ACDR Path B.
        use_bottom_up: Enable bottom-up enhancement path.
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        feature_size: int = 256,
        freq_cutoff: float = 0.25,
        peem_on_c5: bool = False,
        num_heads: int = 4,
        sigma_list: list[float] | None = None,
        dpga_max_pool_size: int = 32,
        acdr_large_kernel: int = 7,
        acdr_dilation: int = 2,
        use_bottom_up: bool = True,
    ) -> None:
        super().__init__()
        self.peem_on_c5 = peem_on_c5
        self.use_bottom_up = use_bottom_up

        # PEEM modules (frequency-domain edge preservation)
        self.peem_c3 = PEEM(C3_size, freq_cutoff=freq_cutoff)
        self.peem_c4 = PEEM(C4_size, freq_cutoff=freq_cutoff)
        self.peem_c5 = PEEM(C5_size, freq_cutoff=freq_cutoff) if peem_on_c5 else None

        # Channel alignment: project to feature_size
        self.align_c3 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.align_c4 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.align_c5 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        # Top-down: PixelShuffle upsampling (2×)
        self.up_p5 = PixelShuffleUpsample(feature_size)  # P5 → P4 size
        self.up_e4 = PixelShuffleUpsample(feature_size)  # E4 → P3 size

        # DPGA cross-scale attention
        sigma_list = sigma_list or [1.0, 2.0, 4.0]
        self.dpga_p5_p4 = DPGA(
            dim=feature_size,
            num_heads=num_heads,
            sigma_list=sigma_list,
            max_pool_size=dpga_max_pool_size,
        )
        self.dpga_e4_p3 = DPGA(
            dim=feature_size,
            num_heads=num_heads,
            sigma_list=sigma_list,
            max_pool_size=dpga_max_pool_size,
        )

        # Post-fusion refinement convs
        self.refine_e4 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.refine_e3 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        # Bottom-up enhancement (optional): E3 ↓ + E4 → E4'
        if use_bottom_up:
            self.bu_down = nn.Sequential(
                nn.Conv2d(
                    feature_size,
                    feature_size,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
            )
            self.bu_refine = nn.Sequential(
                nn.Conv2d(
                    feature_size, feature_size, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
            )

        # Final fusion: downsample E3 to E4 size, concat with E4, fuse
        self.e3_down = nn.Sequential(
            nn.Conv2d(
                feature_size,
                feature_size,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.final_fuse = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        # ACDR: adaptive crowdedness dynamic routing
        self.acdr = ACDR(
            channels=feature_size,
            large_kernel=acdr_large_kernel,
            dilation=acdr_dilation,
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: [C3, C4, C5] backbone feature maps.
                C3: [B, C3_size, H/4,  W/4 ]
                C4: [B, C4_size, H/8,  W/8 ]
                C5: [B, C5_size, H/16, W/16]

        Returns:
            Fused feature map [B, feature_size, H/8, W/8].
        """
        c3, c4, c5 = inputs

        # 1. PEEM: frequency-domain edge enhancement
        c3 = self.peem_c3(c3)
        c4 = self.peem_c4(c4)
        if self.peem_c5 is not None:
            c5 = self.peem_c5(c5)

        # 2. Channel alignment → all feature_size
        p3 = self.align_c3(c3)  # [B, D, H/4,  W/4 ]
        p4 = self.align_c4(c4)  # [B, D, H/8,  W/8 ]
        p5 = self.align_c5(c5)  # [B, D, H/16, W/16]

        # 3. Top-down fusion with DPGA
        # P5 ↑ → P4 size, then cross-attention with P4
        p5_up = self.up_p5(p5)  # [B, D, H/8,  W/8]
        # Handle potential size mismatch from PixelShuffle
        if p5_up.shape[-2:] != p4.shape[-2:]:
            p5_up = F.interpolate(p5_up, size=p4.shape[-2:], mode="nearest")
        e4 = self.dpga_p5_p4(query_feat=p5_up, kv_feat=p4)
        e4 = self.refine_e4(e4)  # [B, D, H/8,  W/8]

        # E4 ↑ → P3 size, then cross-attention with P3
        e4_up = self.up_e4(e4)  # [B, D, H/4,  W/4]
        if e4_up.shape[-2:] != p3.shape[-2:]:
            e4_up = F.interpolate(e4_up, size=p3.shape[-2:], mode="nearest")
        e3 = self.dpga_e4_p3(query_feat=e4_up, kv_feat=p3)
        e3 = self.refine_e3(e3)  # [B, D, H/4,  W/4]

        # 4. Optional bottom-up enhancement: E3 ↓ + E4 → E4'
        if self.use_bottom_up:
            e3_down = self.bu_down(e3)  # [B, D, H/8, W/8]
            if e3_down.shape[-2:] != e4.shape[-2:]:
                e3_down = F.interpolate(e3_down, size=e4.shape[-2:], mode="nearest")
            e4 = self.bu_refine(e4 + e3_down)

        # 5. Fuse E3 (downsampled) + E4 at P4 resolution (H/8)
        e3_at_p4 = self.e3_down(e3)  # [B, D, H/8, W/8]
        if e3_at_p4.shape[-2:] != e4.shape[-2:]:
            e3_at_p4 = F.interpolate(e3_at_p4, size=e4.shape[-2:], mode="nearest")
        fused = self.final_fuse(torch.cat([e3_at_p4, e4], dim=1))

        # 6. ACDR: adaptive density-aware routing
        out = self.acdr(fused)

        return out

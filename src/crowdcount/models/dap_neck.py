"""DAP-Neck v2: Density-Aware Phase-guided Neck for crowd counting.

Built on the proven SPD-PAFPN backbone with density-adaptive routing (ACDR)
appended after fusion.  Optional PEEM for frequency-domain edge enhancement.

Key design principle: all operations are resolution-invariant so that the
train (128×128 patches) → eval (full-resolution images) transition is seamless.

Modules:
- PEEM: Phase-aware Edge Enhancement Module (optional, disabled by default)
- ACDR: Adaptive Crowdedness Dynamic Router (density-adaptive dual-path routing)
- DAPNeck: SPD-PAFPN + ACDR end-to-end neck
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdcount.models.neck import SPD, DeformConv2dBNReLU, _conv3x3_block


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
        freq_y = torch.fft.fftfreq(h, device=device)  # [h]
        freq_x = torch.linspace(0, 0.5, w_rfft, device=device)  # [w_rfft]

        grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
        dist_sq = grid_y**2 + grid_x**2

        sigma = self.freq_cutoff / 2.0
        mask = torch.exp(-dist_sq / (2 * sigma**2 + 1e-8))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w_rfft]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]
        w_rfft = x_freq.shape[-1]

        low_mask = self._build_gaussian_mask(H, w_rfft, x.device)
        high_mask = 1.0 - low_mask

        x_low = torch.fft.irfft2(x_freq * low_mask, s=(H, W), norm="ortho")
        x_high = torch.fft.irfft2(x_freq * high_mask, s=(H, W), norm="ortho")

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        alpha = self.fusion_alpha.sigmoid()
        return x + alpha * x_low + (1 - alpha) * x_high


# ---------------------------------------------------------------------------
# ACDR: Adaptive Crowdedness Dynamic Router
# ---------------------------------------------------------------------------


class ACDR(nn.Module):
    """Density-adaptive dual-path feature routing.

    Estimates a per-image crowdedness scalar via GAP + MLP, then blends
    a local sharpening path (small kernel) with a contextual path (large
    dilated kernel) based on the estimated crowdedness.

    All operations are spatially resolution-invariant (GAP + depthwise conv).

    Args:
        channels: Number of input/output channels.
        large_kernel: Kernel size for the context path (Path B).
        dilation: Dilation rate for the context path.
        hidden_ratio: Hidden dim ratio for the crowdedness estimator MLP.
        gate_init: Initial value for the residual gate (0 = identity start).
    """

    def __init__(
        self,
        channels: int = 256,
        large_kernel: int = 7,
        dilation: int = 2,
        hidden_ratio: int = 4,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()

        hidden = channels // hidden_ratio
        # Residual gate: starts at gate_init (default 0 → identity at init)
        self.gate = nn.Parameter(torch.tensor(gate_init))

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

        # Residual: x + gate * (routed_output)
        # gate starts at 0 → identity at init, ACDR gradually learns to contribute
        routed = (1 - c) * fa + c * fb
        return x + self.gate.tanh() * routed


# ---------------------------------------------------------------------------
# DAPNeck: SPD-PAFPN + ACDR
# ---------------------------------------------------------------------------


class DAPNeck(nn.Module):
    """Density-Aware Phase-guided Neck (v2).

    Uses the proven SPD-PAFPN architecture as the multi-scale fusion backbone,
    with ACDR appended for density-adaptive feature routing.  Optional PEEM on
    C3 for frequency-domain edge enhancement.

    All operations are resolution-invariant: element-wise add for cross-scale
    fusion, nearest-neighbour upsampling, SPD for lossless downsampling.

    Pipeline:
        1. Optional PEEM on C3
        2. Channel alignment: 1×1 conv to project all scales to ``feature_size``
        3. Top-down FPN: P5 ↑ + P4 → P4;  P4 ↑ + P3 → P3
        4. Bottom-up PAN with SPD: P3 ↓ + P4 → P4;  P4 ↓ + P5 → P5
        5. All three scales to P4 resolution → concat → 1×1 fusion
        6. ACDR: density-adaptive routing

    Args:
        C3_size: Input channels for C3 (typically 256).
        C4_size: Input channels for C4 (typically 512).
        C5_size: Input channels for C5 (typically 512).
        feature_size: Internal / output channel dimension.
        use_peem: Enable PEEM on C3 (disabled by default).
        freq_cutoff: PEEM low-pass Gaussian cutoff ratio.
        use_dcn: Use deformable conv in PEEM and FPN 3×3 convs.
        acdr_large_kernel: Large kernel size for ACDR Path B.
        acdr_dilation: Dilation rate for ACDR Path B.
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        feature_size: int = 256,
        use_peem: bool = False,
        freq_cutoff: float = 0.25,
        use_dcn: bool = False,
        acdr_large_kernel: int = 7,
        acdr_dilation: int = 2,
    ) -> None:
        super().__init__()
        self.use_peem = use_peem

        # Optional PEEM on C3 (only C3 has enough spatial resolution)
        self.peem_c3 = PEEM(C3_size, freq_cutoff=freq_cutoff) if use_peem else None

        # --- Top-down pathway (FPN) ---
        self.P5_1 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_1 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P3_1 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")

        self.P5_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        self.P4_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        self.P3_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)

        # --- Bottom-up pathway (PAN with SPD) ---
        self.P3_downsampled = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * feature_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_downsampled = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * feature_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_2_bu = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        self.P5_2_bu = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)

        # --- Final 3-scale concat fusion ---
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * feature_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        # --- ACDR: density-adaptive routing ---
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
                C5: [B, C5_size, H/8,  W/8 ] (VGG) or [B, C5_size, H/16, W/16]

        Returns:
            Fused feature map [B, feature_size, H/8, W/8].
        """
        c3, c4, c5 = inputs

        # 1. Optional PEEM on C3
        if self.peem_c3 is not None:
            c3 = self.peem_c3(c3)

        # 2. Top-down pathway
        P5_x = self.P5_1(c5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_lateral = self.P4_1(c4)
        P4_x = P4_lateral + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_lateral = self.P3_1(c3)
        P3_x = P3_lateral + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # 3. Bottom-up pathway with SPD
        P3_down = self.P3_downsampled(P3_x)
        P4_x = P4_x + P3_down
        P4_x = self.P4_2_bu(P4_x)

        P4_down = self.P4_downsampled(P4_x)
        P5_x = P5_x + P4_down
        P5_x = self.P5_2_bu(P5_x)
        P5_x = self.P5_upsampled(P5_x)

        # 4. Fuse all three scales at P4 resolution
        fused = torch.cat([P3_down, P4_x, P5_x], dim=1)
        out = self.fusion(fused)

        # 5. ACDR: density-adaptive routing
        out = self.acdr(out)

        return out

"""SEMC-style feature enhancement blocks for DSGCNet.

Provides lightweight local multi-scale refinement that can be inserted
after the GCN fusion step as a post-processing enhancer.  Only standard
PyTorch primitives are used; no additional dependencies are required.

Components
----------
CAB  : Channel Attention Block (squeeze-and-excitation, avg + max).
SAB  : Spatial Attention Block (CBAM-style, 7×7 conv on avg+max maps).
MSCB : Multi-Scale Convolution Block (expand → parallel DW → add-fuse →
       project, with skip connection).
SEMCEnhancer : Top-level module that chains CAB → SAB → MSCB and wraps
               the result in a residual connection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CAB(nn.Module):
    """Channel Attention Block.

    Computes a [B, C, 1, 1] channel-wise attention map from both average
    and max pooling branches and multiplies it back onto the input.
    """

    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        reduced = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return channel attention weights of shape [B, C, 1, 1]."""
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    """Spatial Attention Block (CBAM-style).

    Returns a [B, 1, H, W] spatial attention map derived from the
    channel-wise average and maximum of the input feature map.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial attention weights of shape [B, 1, H, W]."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class MSCB(nn.Module):
    """Multi-Scale Convolution Block.

    Pipeline: expand (1×1 PW) → parallel DW convs → add-fuse → project
    (1×1 PW) → + skip.  All operations keep spatial resolution intact.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        expansion_factor: Channel multiplier for the intermediate representation.
        kernel_sizes: Depthwise kernel sizes for the parallel branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 4,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        use_skip_connection: bool = False,
    ) -> None:
        super().__init__()
        ex_ch = in_channels * expansion_factor
        self.use_skip_connection = use_skip_connection

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, ex_ch, 1, bias=False),
            nn.BatchNorm2d(ex_ch),
            nn.ReLU6(inplace=True),
        )
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        ex_ch, ex_ch, k, padding=k // 2, groups=ex_ch, bias=False
                    ),
                    nn.BatchNorm2d(ex_ch),
                    nn.ReLU6(inplace=True),
                )
                for k in kernel_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(ex_ch, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip: nn.Module | None = None
        if use_skip_connection:
            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
                if in_channels != out_channels
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        # Parallel multi-scale depthwise convolution, add-fused
        fused = self.dwconvs[0](out)
        for dw in self.dwconvs[1:]:
            fused = fused + dw(out)
        out = self.project(fused)
        if self.skip is not None:
            return out + self.skip(x)
        return out


class SEMCEnhancer(nn.Module):
    """Post-GCN multi-scale feature enhancer inspired by SEMC.

    Designed to be inserted *after* the GCN fusion step in DSGCNet
    (i.e., after ``feature_fl`` is produced) as a local refinement stage.
    Input and output tensors have identical shapes: [B, in_channels, H, W].

    Internal structure::

        x ──► CAB(x) * x ──► SAB(·) * · ──► MSCB(·)
                                                  │
        (optional) density_hint gate ─────────────►· * gate
                                                  │
        x ─────────────────────────────────────── + (if use_residual)
                                                  │
                                                 out

    Args:
        in_channels: Feature channels (256 in DSGCNet).
        expansion_factor: MSCB internal channel expansion ratio.
        kernel_sizes: Depthwise kernel sizes for parallel MSCB branches.
        use_residual: When ``True`` the output is ``x + enhanced``.
            Strongly recommended; prevents gradient disruption during
            early training when the enhancer has not yet converged.
        use_density_hint: When ``True`` an extra ``Conv2d(1→C)`` branch
            computes a density-guided gate applied to the enhanced branch
            before the residual add.
    """

    def __init__(
        self,
        in_channels: int = 256,
        expansion_factor: int = 4,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        use_residual: bool = True,
        use_density_hint: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_density_hint = use_density_hint

        self.cab = CAB(in_channels)
        self.sab = SAB()
        self.mscb = MSCB(
            in_channels,
            in_channels,
            expansion_factor,
            kernel_sizes,
            use_skip_connection=False,
        )

        self.density_proj: nn.Module | None
        if use_density_hint:
            self.density_proj = nn.Sequential(
                nn.Conv2d(1, in_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.density_proj = None

    def forward(
        self, x: torch.Tensor, density_hint: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Feature tensor ``[B, in_channels, H, W]``.
            density_hint: Optional density map ``[B, 1, H, W]``.  Only used
                when ``use_density_hint=True`` was set at construction time.

        Returns:
            Enhanced feature tensor with the same shape as ``x``.
        """
        enhanced = self.cab(x) * x
        enhanced = self.sab(enhanced) * enhanced
        enhanced = self.mscb(enhanced)

        if self.density_proj is not None and density_hint is not None:
            gate = self.density_proj(density_hint)  # [B, C, H, W], values in (0, 1)
            enhanced = enhanced * gate

        if self.use_residual:
            return x + enhanced
        return enhanced

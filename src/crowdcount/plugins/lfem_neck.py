"""LFEM-based multi-scale neck for DSGCNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.neck import SPD
from crowdcount.plugins.LFEM import LFEM


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SPDAutoDownsample(nn.Module):
    """Downsample with SPD, padding odd spatial sizes before pixel shuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == target_size:
            return x
        if x.shape[-2] < target_size[0] or x.shape[-1] < target_size[1]:
            return F.interpolate(x, size=target_size, mode="nearest")

        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        out = self.block(x)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="nearest")
        return out


class _FastNormalizedFusion(nn.Module):
    """Small fast normalized weighted fusion for same-shape tensors."""

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        if num_inputs < 1:
            raise ValueError("num_inputs must be >= 1")
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    @property
    def normalized_weights(self) -> torch.Tensor:
        weights = torch.relu(self.weights)
        return weights / (weights.sum() + self.eps)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} fusion inputs, got {len(inputs)}"
            )
        weights = self.normalized_weights
        out = torch.zeros_like(inputs[0])
        for weight, x in zip(weights, inputs):
            out = out + weight.to(device=x.device, dtype=x.dtype) * x
        return out


def _upsample_to(
    x: torch.Tensor,
    target_size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if x.shape[-2:] == target_size:
        return x
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(x, size=target_size, mode=mode, align_corners=False)
    return F.interpolate(x, size=target_size, mode=mode)


class LFEMMultiScaleNeck(nn.Module):
    """Three-branch LFEM neck that replaces PA-FPN without changing outputs.

    The module consumes DSGCNet backbone features ``[C3, C4, C5]`` and returns
    a 256-channel stride-8 feature map.  Each scale is first projected to a
    shared channel width and enhanced by its own LFEM branch; the branches are
    then aligned to the C4 resolution and fused with normalized learnable
    weights.
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        feature_size: int = 256,
        use_spd_downsample: bool = True,
        fusion_eps: float = 1e-4,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.upsample_mode = upsample_mode

        self.lateral_c3 = _ConvBNAct(C3_size, feature_size, kernel_size=1, padding=0)
        self.lateral_c4 = _ConvBNAct(C4_size, feature_size, kernel_size=1, padding=0)
        self.lateral_c5 = _ConvBNAct(C5_size, feature_size, kernel_size=1, padding=0)

        self.lfem_c3 = LFEM(feature_size)
        self.lfem_c4 = LFEM(feature_size)
        self.lfem_c5 = LFEM(feature_size)

        self.downsample_c3: nn.Module
        if use_spd_downsample:
            self.downsample_c3 = _SPDAutoDownsample(feature_size)
        else:
            self.downsample_c3 = _ConvBNAct(
                feature_size, feature_size, kernel_size=3, stride=2
            )

        self.fusion = _FastNormalizedFusion(3, eps=fusion_eps)
        self.refine = nn.Sequential(
            _ConvBNAct(feature_size, feature_size, kernel_size=3),
            _ConvBNAct(feature_size, feature_size, kernel_size=1, padding=0),
        )

    def _downsample_c3(
        self, p3: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        if isinstance(self.downsample_c3, _SPDAutoDownsample):
            return self.downsample_c3(p3, target_size)
        out = self.downsample_c3(p3)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="nearest")
        return out

    def forward(
        self,
        inputs: list[torch.Tensor],
        return_intermediates: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        c3, c4, c5 = inputs
        p3 = self.lfem_c3(self.lateral_c3(c3))
        p4 = self.lfem_c4(self.lateral_c4(c4))
        p5 = self.lfem_c5(self.lateral_c5(c5))

        target_size = p4.shape[-2:]
        p3_to_p4 = self._downsample_c3(p3, target_size)
        p5_to_p4 = _upsample_to(p5, target_size, self.upsample_mode)
        out = self.refine(self.fusion([p3_to_p4, p4, p5_to_p4]))

        if return_intermediates:
            return out, (p3, p4, p5)
        return out
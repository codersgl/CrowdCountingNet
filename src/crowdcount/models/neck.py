"""Feature fusion neck for DSGCNet: SPD + PA-FPN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class SPD(nn.Module):
    """Space-to-Depth downsampler (2×) with zero parameters."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )


class DeformConv2dBNReLU(nn.Module):
    """DCNv2 + BatchNorm + ReLU with learnable offsets and modulation masks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # 2D offsets per kernel position
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,  # modulation mask per kernel position
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize offsets to zero → acts like standard conv initially
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = self.mask_conv(x).sigmoid()
        out = self.dcn(x, offset, mask)
        return self.relu(self.bn(out))


def _conv3x3_block(in_ch: int, out_ch: int, use_dcn: bool = False) -> nn.Module:
    """Build a 3×3 conv + BN + ReLU block, optionally using deformable conv."""
    if use_dcn:
        return DeformConv2dBNReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DepthwiseSeparableConvBNReLU(nn.Module):
    """Depthwise-separable 3×3 refinement used inside BiFPN nodes."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _bifpn_refine(channels: int, use_depthwise: bool) -> nn.Module:
    if use_depthwise:
        return _DepthwiseSeparableConvBNReLU(channels)
    return _conv3x3_block(channels, channels)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def _resize_bilinear_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(
        x, size=ref.shape[-2:], mode="bilinear", align_corners=False
    )


class _FastNormalizedFusion(nn.Module):
    """BiFPN fast normalized weighted fusion for same-shape tensors."""

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
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


class _BiFPNDownsample(nn.Module):
    """Downsample to a target size, optionally preserving detail with SPD."""

    def __init__(self, channels: int, use_spd: bool = True) -> None:
        super().__init__()
        self.use_spd = use_spd
        if use_spd:
            self.op = nn.Sequential(
                SPD(),
                nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == target_size:
            return x
        if x.shape[-2] < target_size[0] or x.shape[-1] < target_size[1]:
            return F.interpolate(x, size=target_size, mode="nearest")

        if self.use_spd:
            pad_h = x.shape[-2] % 2
            pad_w = x.shape[-1] % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        out = self.op(x)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="nearest")
        return out


class SPDBiFPNBlock(nn.Module):
    """Single SPD-BiFPN block over P3/P4/P5 feature levels."""

    def __init__(
        self,
        feature_size: int = 256,
        use_spd_downsample: bool = True,
        use_depthwise_refine: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.p4_td_fusion = _FastNormalizedFusion(2, eps=eps)
        self.p3_out_fusion = _FastNormalizedFusion(2, eps=eps)
        self.p4_out_fusion = _FastNormalizedFusion(3, eps=eps)
        self.p5_out_fusion = _FastNormalizedFusion(2, eps=eps)

        self.p4_td_refine = _bifpn_refine(feature_size, use_depthwise_refine)
        self.p3_out_refine = _bifpn_refine(feature_size, use_depthwise_refine)
        self.p4_out_refine = _bifpn_refine(feature_size, use_depthwise_refine)
        self.p5_out_refine = _bifpn_refine(feature_size, use_depthwise_refine)

        self.p3_downsample = _BiFPNDownsample(feature_size, use_spd_downsample)
        self.p4_downsample = _BiFPNDownsample(feature_size, use_spd_downsample)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_td = p5
        p4_td = self.p4_td_refine(
            self.p4_td_fusion([p4, _resize_like(p5_td, p4)])
        )
        p3_out = self.p3_out_refine(
            self.p3_out_fusion([p3, _resize_like(p4_td, p3)])
        )

        p3_down = self.p3_downsample(p3_out, p4.shape[-2:])
        p4_out = self.p4_out_refine(
            self.p4_out_fusion([p4, p4_td, p3_down])
        )

        p4_down = self.p4_downsample(p4_out, p5.shape[-2:])
        p5_out = self.p5_out_refine(self.p5_out_fusion([p5, p4_down]))
        return p3_out, p4_out, p5_out


class SPDBiFPNNeck(nn.Module):
    """SPD-BiFPN drop-in replacement for the PA-FPN neck.

    The neck keeps DSGCNet's downstream contract unchanged: it consumes
    ``[C3, C4, C5]`` backbone features and returns a 256-channel stride-8
    fused map.  C3 downsampling uses SPD by default to preserve local point
    layout information for the regression branch.
    """

    def __init__(
        self,
        C3_size: int,
        C4_size: int,
        C5_size: int,
        feature_size: int = 256,
        num_blocks: int = 1,
        use_spd_downsample: bool = True,
        use_depthwise_refine: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("SPDBiFPNNeck requires num_blocks >= 1")
        self.feature_size = feature_size

        self.P3_1 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_1 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P5_1 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                SPDBiFPNBlock(
                    feature_size=feature_size,
                    use_spd_downsample=use_spd_downsample,
                    use_depthwise_refine=use_depthwise_refine,
                    eps=eps,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_p3_downsample = _BiFPNDownsample(feature_size, use_spd_downsample)
        self.final_fusion = _FastNormalizedFusion(3, eps=eps)
        self.final_refine = _bifpn_refine(feature_size, use_depthwise_refine)

    def forward(
        self,
        inputs: list[torch.Tensor],
        return_intermediates: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        c3, c4, c5 = inputs
        p3 = self.P3_1(c3)
        p4 = self.P4_1(c4)
        p5 = self.P5_1(c5)

        for block in self.blocks:
            p3, p4, p5 = block(p3, p4, p5)

        p3_to_p4 = self.final_p3_downsample(p3, p4.shape[-2:])
        p5_to_p4 = _resize_like(p5, p4)
        out = self.final_refine(self.final_fusion([p3_to_p4, p4, p5_to_p4]))

        if return_intermediates:
            return out, (p3, p4, p5)
        return out


class P2PNeXtDecoder(nn.Module):
    """P2PNeXt-style FPN decoder with bilinear upsampling and Conv-BN-ReLU."""

    _OUTPUT_LEVELS = {"p3", "p4", "p5", "fused"}

    def __init__(
        self,
        C3_size: int,
        C4_size: int,
        C5_size: int,
        feature_size: int = 256,
        output_level: str = "p3",
    ) -> None:
        super().__init__()
        if output_level not in self._OUTPUT_LEVELS:
            raise ValueError(
                f"Unsupported output_level={output_level}, "
                f"expected one of {sorted(self._OUTPUT_LEVELS)}"
            )
        self.output_level = output_level

        self.P5_1 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_1 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P3_1 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

        self.P4_2 = _conv3x3_block(feature_size, feature_size)
        self.P3_2 = _conv3x3_block(feature_size, feature_size)
        self.P5_2 = _conv3x3_block(feature_size, feature_size)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * feature_size, feature_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        return_intermediates: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        c3, c4, c5 = inputs
        p5 = self.P5_2(self.P5_1(c5))
        p4_lateral = self.P4_1(c4)
        p4 = self.P4_2(p4_lateral + _resize_bilinear_like(p5, p4_lateral))
        p3_lateral = self.P3_1(c3)
        p3 = self.P3_2(p3_lateral + _resize_bilinear_like(p4, p3_lateral))

        if self.output_level == "fused":
            out = self.fusion(
                torch.cat(
                    [
                        _resize_bilinear_like(p3, p4),
                        p4,
                        _resize_bilinear_like(p5, p4),
                    ],
                    dim=1,
                )
            )
        else:
            selected = {"p3": p3, "p4": p4, "p5": p5}[self.output_level]
            out = _resize_bilinear_like(selected, p4)

        if return_intermediates:
            return out, (p3, p4, p5)
        return out


class Decoder_SPD_PAFPN(nn.Module):
    """SPD-enhanced Path Aggregation FPN decoder."""

    def __init__(
        self,
        C3_size: int,
        C4_size: int,
        C5_size: int,
        feature_size: int = 256,
        use_dcn: bool = False,
        fpn_attention: bool = False,
    ):
        super().__init__()
        self.fpn_attention = fpn_attention

        # Lazy import to avoid circular deps
        if fpn_attention:
            from crowdcount.plugins.msaa import FPNAttentionGate, FPNSpatialAttention

            self.td_gate_p5 = FPNAttentionGate(feature_size)
            self.td_gate_p4 = FPNAttentionGate(feature_size)
            self.bu_gate_p4 = FPNAttentionGate(feature_size)
            self.bu_gate_p5 = FPNAttentionGate(feature_size)
            self.final_spatial = FPNSpatialAttention()
        # Top-down pathway: C5 → P5
        self.P5_1 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        # C4 → P4
        self.P4_1 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        # C3 → P3
        self.P3_1 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P3_2 = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        # Bottom-up pathway with SPD
        self.P3_downsampled = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * feature_size, feature_size, kernel_size=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_downsampled = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * feature_size, feature_size, kernel_size=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        # Independent conv layers for bottom-up pathway
        self.P4_2_bu = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        self.P5_2_bu = _conv3x3_block(feature_size, feature_size, use_dcn=use_dcn)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * feature_size, feature_size, kernel_size=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs, return_intermediates: bool = False):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_lateral = self.P4_1(C4)
        if self.fpn_attention:
            P4_x = self.td_gate_p5(P4_lateral, P5_upsampled_x)
        else:
            P4_x = P4_lateral + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_lateral = self.P3_1(C3)
        if self.fpn_attention:
            P3_x = self.td_gate_p4(P3_lateral, P4_upsampled_x)
        else:
            P3_x = P3_lateral + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # Bottom-up
        P3_down = self.P3_downsampled(P3_x)
        if self.fpn_attention:
            P4_x = self.bu_gate_p4(P4_x, P3_down)
        else:
            P4_x = P4_x + P3_down
        P4_x = self.P4_2_bu(P4_x)
        P4_down = self.P4_downsampled(P4_x)
        if self.fpn_attention:
            P5_x = self.bu_gate_p5(P5_x, P4_down)
        else:
            P5_x = P5_x + P4_down
        P5_x = self.P5_2_bu(P5_x)
        P5_x_out = P5_x  # Save at original low-res for cross-scale use
        P5_x = self.P5_upsampled(P5_x)

        fuse = torch.cat([P3_down, P4_x, P5_x], 1)
        out = self.fusion(fuse)
        if self.fpn_attention:
            out = self.final_spatial(out)
        if return_intermediates:
            return out, (P3_x, P4_x, P5_x_out)
        return out

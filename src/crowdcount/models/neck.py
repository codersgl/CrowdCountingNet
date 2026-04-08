"""Feature fusion neck for DSGCNet: SPD + PA-FPN."""

from __future__ import annotations

import torch
import torch.nn as nn
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

    def forward(self, inputs):
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
        P5_x = self.P5_upsampled(P5_x)

        fuse = torch.cat([P3_down, P4_x, P5_x], 1)
        out = self.fusion(fuse)
        if self.fpn_attention:
            out = self.final_spatial(out)
        return out

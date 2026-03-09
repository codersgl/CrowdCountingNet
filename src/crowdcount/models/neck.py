"""Feature fusion neck for DSGCNet: SPD + PA-FPN."""

import torch
import torch.nn as nn


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


class Decoder_SPD_PAFPN(nn.Module):
    """SPD-enhanced Path Aggregation FPN decoder."""

    def __init__(
        self, C3_size: int, C4_size: int, C5_size: int, feature_size: int = 256
    ):
        super().__init__()
        # Top-down pathway: C5 → P5
        self.P5_1 = nn.Sequential(
            nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        # C4 → P4
        self.P4_1 = nn.Sequential(
            nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        # C3 → P3
        self.P3_1 = nn.Sequential(
            nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.P3_2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
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

        P4_x = self.P4_1(C4) + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3) + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # Bottom-up
        P3_x = self.P3_downsampled(P3_x)
        P4_x = P4_x + P3_x
        P4_x = self.P4_2(P4_x)
        P5_x = P5_x + self.P4_downsampled(P4_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.P5_upsampled(P5_x)

        fuse = torch.cat([P3_x, P4_x, P5_x], 1)
        return self.fusion(fuse)

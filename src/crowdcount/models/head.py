"""Prediction head modules for DSGCNet.

Contains:
  - Density_pred:   density map regression head
  - RegressionModel: point regression head
  - ClassificationModel: point classification head
"""

import torch
import torch.nn as nn


class Density_pred(nn.Module):
    """Density map prediction head."""

    def __init__(self):
        super().__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.v1(x)
        x = self.v2(x)
        x = self.v3(x)
        return self.conv_layers(x)


class RegressionModel(nn.Module):
    """Point coordinate regression head."""

    def __init__(
        self, num_features_in: int, num_anchor_points: int = 4, feature_size: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(
            feature_size, num_anchor_points * 2, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    """Point classification head."""

    def __init__(
        self,
        num_features_in: int,
        num_anchor_points: int = 4,
        num_classes: int = 80,
        prior: float = 0.01,
        feature_size: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(
            feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(
            batch_size, width, height, self.num_anchor_points, self.num_classes
        )
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class DensityPred_Backbone(nn.Module):
    """Parametric density map prediction head for backbone features.

    Supports different input channel sizes (256, 512, etc.) with adaptive head design.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # Three conv blocks with same channel as input
        self.v1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Projection to 1 channel
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, in_channels, H, W]

        Returns:
            Density map of shape [B, 1, H, W]
        """
        x = self.v1(x)
        x = self.v2(x)
        x = self.v3(x)
        return self.conv_layers(x)


class DensityPred_Block3(nn.Module):
    """Density prediction head for VGG block3 features (256 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DensityPred_Block4(nn.Module):
    """Density prediction head for VGG block4 features (512 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DensityPred_Block5(nn.Module):
    """Density prediction head for VGG block5 features (512 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

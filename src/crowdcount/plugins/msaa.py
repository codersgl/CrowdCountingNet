import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpatialAggregation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim, H, W)

        Returns:
            (batch_size, hidden_dim, H, W)
        """
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        pooled = torch.cat([avg_pool_out, max_pool_out], dim=1)  # [batch_size, 2, H, W]
        pooled = self.conv1(pooled)  # [batch_size, 1 H, W]
        spatial_attn = self.sigmoid(pooled)  # [batch_size, 1, H, W]
        return x * spatial_attn  # [batch_size, hidden_dim, H, W]


class ChannelAggregation(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (batch_size, in_channels, H, W)

        Returns:
            (batch_size, in_channels, 1, 1)
        """
        return self.fc(self.avg_pool(features))


class MultiScaleFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3)

    def forward(self, features: torch.Tensor):
        """

        Args:
            features: (batch_size, dim, H, W)

        Returns:
            Fused features: (batch_size, dim, H, W)
        """
        x_1 = self.conv_3x3(features)  # [batch_size, dim, H, W]
        x_2 = self.conv_5x5(features)  # [batch_size, dim, H, W]
        x_3 = self.conv_7x7(features)  # [batch_size, dim, H, W]
        fused = x_1 + x_2 + x_3  # [batch_size, dim, H, W]
        return fused


class MSAA(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = in_channels // reduction

        self.down = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=1,
        )

        self.multi_scale_fusion = MultiScaleFusion(hidden_dim)

        self.spatial_aggregation = SpatialAggregation()
        self.channel_aggregation = ChannelAggregation(in_channels, hidden_dim)

        self.up = nn.Conv2d(
            hidden_dim,
            in_channels,
            kernel_size=1,
        )

    def forward(self, features: torch.Tensor):
        """

        Args:
            features: (batch_size, in_channels, H, W)
        """
        fused = self.multi_scale_fusion(
            self.down(features)
        )  # [batch_size, hidden_dim, H, W]
        spatial_aggregated_features = self.spatial_aggregation(
            fused
        )  # [batch_size, hidden_dim, H, W]
        spatial_aggregated_features = self.up(
            spatial_aggregated_features
        )  # [batch_size, input_dim, H, W]
        channel_aggregated_features = self.channel_aggregation(
            features
        )  # [batch_size, input_dim, 1, 1]
        return spatial_aggregated_features * channel_aggregated_features + features


class MsaaAdaptiveLayer(nn.Module):
    def __init__(self, in_channels: int = 1280, reduction: int = 4) -> None:
        super().__init__()
        self.msaa1 = MSAA(in_channels=in_channels, reduction=reduction)
        self.msaa2 = MSAA(in_channels=in_channels, reduction=reduction)
        self.msaa3 = MSAA(in_channels=in_channels, reduction=reduction)

    def forward(self, feature_list: List[torch.Tensor]) -> List[torch.Tensor]:

        feature1 = feature_list[1]  # [batch_size, 256, 32, 32]
        feature2 = feature_list[2]  # [batch_size, 512, 16, 16]
        feature3 = feature_list[3]  # [batch_size, 512, 8, 8]
        f_1 = torch.cat(
            [
                feature1,
                F.interpolate(feature2, feature1.size()[-2:], mode="bilinear"),
                F.interpolate(feature3, feature1.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 32, 32]

        f_2 = torch.cat(
            [
                feature2,
                F.interpolate(feature1, feature2.size()[-2:], mode="bilinear"),
                F.interpolate(feature3, feature2.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 16, 16]

        f_3 = torch.cat(
            [
                feature3,
                F.interpolate(feature1, feature3.size()[-2:], mode="bilinear"),
                F.interpolate(feature2, feature3.size()[-2:], mode="bilinear"),
            ],
            dim=1,
        )  # [batch_size, 1280, 8, 8]
        return [feature_list[0], self.msaa1(f_1), self.msaa2(f_2), self.msaa3(f_3)]

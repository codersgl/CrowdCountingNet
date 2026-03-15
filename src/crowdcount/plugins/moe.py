import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ContentDrivenSpatialAttention(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim, h, w)

        Returns: (batch_size, 1, h, w)

        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, h, w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch_size, 1, h, w]
        attn = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, h, w]
        attn = self.conv(attn)  # [batch_size, 1, h, w]
        attn = self.sigmoid(attn)  # [batch_size, 1, h, w]
        return attn


class PositionDrivenSpatialAttention(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim + 2, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, 1, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (batch_size, input_dim, h, w)

        Returns: (batch_size, 1, h, w)
        """
        B, _, H, W = x.size()
        device = x.device
        y_coor = (
            torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, 1, H, W).to(device)
        )  # [batch_size, 1, H, W]
        x_coor = (
            torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, 1, H, W).to(device)
        )  # [batch_size, 1, H, W]

        pos = torch.cat([x_coor, y_coor], dim=1)  # [batch_size, 2, h, w]
        x = torch.cat([x, pos], dim=1)  # [batch_size, input_dim + 2, h, w]
        attn = self.conv(x)  # [batch_size, 1, h, w]
        attn = self.sigmoid(attn)
        return attn


class DynamicGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, input_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim // 4),
            nn.GELU(),
            nn.Conv2d(input_dim // 4, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(
        self, content_attn: torch.Tensor, position_attn: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            content_attn: [B, 1, H, W]
            position_attn: [B, 1, H, W]

        Returns:
            fused_attn: [B, 1, H, W]
        """
        cat_attn = torch.cat([content_attn, position_attn], dim=1)  # [B, 2, H, W]

        weights = self.gate_conv(cat_attn)  # [B, 2, H, W]

        fused_attn = (
            weights[:, 0:1] * content_attn + weights[:, 1:2] * position_attn
        )  # [B, 1, H, W]
        return fused_attn


class SpatialAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.content_driven_attention = ContentDrivenSpatialAttention()
        self.position_driven_attention = PositionDrivenSpatialAttention(input_dim)
        self.dynamic_gate = DynamicGate(input_dim)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (batch_size, input_dim, h, w)

        Returns: [batch_size, input_dim, h, w]
        """
        content_attn = self.content_driven_attention(x)  # [batch_size, 1, h, w]
        position_atten = self.position_driven_attention(x)  # [batch_size, 1, h, w]
        attn = self.dynamic_gate(content_attn, position_atten)  # [batch_size, 1, h, w]
        output = x * attn  # [batch_size, input_dim, h, w]
        return self.feature_fusion(output)


class ChannelAttention(nn.Module):
    def __init__(self, input_dim: int, reduction=4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.share_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: [batch_size, input_dim, h, w]

        Returns:
            [TODO:return]
        """
        B, C, _, _ = x.size()
        avg_out = self.avg_pool(x).view(B, C)  # [batch_size, input_dim]
        avg_out = self.share_mlp(avg_out)  # [batch_size, input_dim]
        max_out = self.max_pool(x).view(B, C)  # [batch_size, input_dim]
        max_out = self.share_mlp(max_out)  # [batch_size, input_dim]
        attn = self.sigmoid(avg_out + max_out).view(
            B, C, 1, 1
        )  # [batch_size, input_dim, 1, 1]
        output = x * attn  # [batch_size, input_dim, h, w]
        return output


class ESCA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.spatial_attention = SpatialAttention(input_dim)
        self.channel_attention = ChannelAttention(input_dim, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (batch_size, input_dim, h, w)

        Returns: (batch_size, input_dim, h, w)
        """
        x = self.channel_attention(self.spatial_attention(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        assert input_dim % num_heads == 0, (
            "Embedding dim must be divisible by num_heads"
        )
        self.d_k = input_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, input_dim, h, w]
        Returns: [batch_size, N, input_dim]
        """
        batch_size, input_dim, h, w = x.size()
        N = h * w
        x = x.reshape(batch_size, N, input_dim)
        Q: torch.Tensor = self.W_q(x)  # [batch_size, N, input_dim]
        K: torch.Tensor = self.W_k(x)  # [batch_size, N, input_dim]
        V: torch.Tensor = self.W_v(x)  # [batch_size, N, input_dim]

        Q = Q.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]
        K = K.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]
        V = V.view(
            batch_size, N, self.num_heads, self.d_k
        )  # [batch_size, N, self.num_heads, self.d_k]

        Q = Q.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]
        K = K.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]
        V = V.transpose(1, 2)  # [batch_size, self.num_heads, N, d_k]

        multi_attention_score = (
            Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)
        )  # [batch_size, self.num_heads, N, N]

        multi_attention_weight = F.softmax(
            multi_attention_score, dim=-1
        )  # [batch_size, self.num_heads, N, N]

        multi_attention_weight = self.dropout(multi_attention_weight)

        multi_attention = (
            multi_attention_weight @ V
        )  # [batch_size, self.num_heads, N, self.d_k]
        multi_attention = multi_attention.transpose(
            1, 2
        ).contiguous()  # [batch_size, N, self.num_heads, self.d_k]

        attention = multi_attention.view(
            batch_size, N, self.input_dim
        )  # [batch_size, N, self.input_dim]
        attention = self.W_o(attention)  # [batch_size, N, self.input_dim]

        return attention


class SENet(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        y = self.avg_pool(x).view(b, c)
        # Excitation: (b, c) -> (b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GlobalExpert(nn.Module):
    def __init__(
        self, input_dim: int, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            input_dim, num_heads=num_heads, dropout=dropout
        )
        self.feature_enhancement = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.LayerNorm(2 * input_dim),
            nn.GELU(),
            nn.Linear(2 * input_dim, input_dim),
        )
        self.channel_attention = SENet(input_dim)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: [batch_size, input_dim, h, w]

        Returns:
            [TODO:return]
        """
        b, c, h, w = x.size()
        f_attention = self.multi_attention(x)  # [batch_size, h * w, input_dim]
        f_enhanced = self.feature_enhancement(
            f_attention
        )  # [batch_size, h * w, input_dim]
        f_enhanced = f_enhanced.reshape(b, c, h, w)  # [batch_size, input_dim, h, w]
        f_global = self.channel_attention(f_enhanced)  # [batch_size, input_dim, h, w]
        output = torch.cat([x, f_global], dim=1)  # [batch_size, 2 * input_dim, h, w]
        output = self.feature_fusion(output)
        return output


class SpatialAttentionForRegions(nn.Module):
    def __init__(self, input_dim: int, N: int = 4) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, N, kernel_size=1)
        self.N = N

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            List of N region features, each [B, C, H, W]
        """
        f_mask_logits = self.conv_1x1(x)  # [B, N, H, W]
        masks = torch.softmax(f_mask_logits, dim=1)  # [B, N, H, W]
        regions = []

        for i in range(self.N):
            mask = masks[:, i : i + 1, :, :]  # [B, 1, H, W]
            f = x * mask  # [B, C, H, W]
            regions.append(f)

        return regions


class RegionContrast(nn.Module):
    def __init__(self, input_dim: int, N: int = 4) -> None:
        super().__init__()
        self.N = N
        self.contrast_fusion = nn.Conv2d(
            input_dim * (N * (N - 1) // 2), input_dim, kernel_size=1
        )

    def forward(self, regions: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            regions: List of N region features, each [B, C, H, W]
        Returns:
            aggregated: [B, C, H, W]
        """
        contrast_features = []

        for i in range(self.N):
            for j in range(i + 1, self.N):
                contrast = regions[i] - regions[j]  # [B, C, H, W]
                contrast_features.append(contrast)

        # Stack + Mean
        stacked = torch.stack(contrast_features, dim=1)  # [B, 6, C, H, W]
        aggregated = stacked.mean(dim=1)  # [B, C, H, W]

        return aggregated


class RegionExpert(nn.Module):
    def __init__(self, input_dim: int, N: int = 4) -> None:
        super().__init__()
        self.region_extractor = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )
        self.spatial_attention_for_regions = SpatialAttentionForRegions(input_dim, N)
        self.region_contrast = RegionContrast(input_dim, N)
        self.region_attention = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = SpatialAttention(input_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        f_extractor = self.region_extractor(x)  # [B, C, H, W]
        regions = self.spatial_attention_for_regions(f_extractor)  # N x [B, C, H, W]
        aggregated = self.region_contrast(regions)  # [B, C, H, W]

        # Region Attention
        region_weight = self.region_attention(aggregated)  # [B, 1, H, W]
        f_region = aggregated * region_weight

        # Spatial Attention
        spatial_weight = self.spatial_attention(f_region)  # [B, 1, H, W]
        f_region = f_region * spatial_weight

        return f_region


class SobelKernel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.sobel_x: torch.Tensor
        self.sobel_y: torch.Tensor
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        )

        # Depthwise卷积，对每个通道独立应用
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3).repeat(input_dim, 1, 1, 1)
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3).repeat(input_dim, 1, 1, 1)
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.input_dim)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.input_dim)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)  # [B, C, H, W]
        return edge


class LocalExpert(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        # 1. Local Branch Extractor (dilation=1,2,3)
        self.group_conv1 = nn.Conv2d(
            input_dim, input_dim, kernel_size=3, padding=1, dilation=1, groups=input_dim
        )
        self.group_conv2 = nn.Conv2d(
            input_dim, input_dim, kernel_size=3, padding=2, dilation=2, groups=input_dim
        )
        self.group_conv3 = nn.Conv2d(
            input_dim, input_dim, kernel_size=3, padding=3, dilation=3, groups=input_dim
        )

        # 2. Details Enhancement
        self.details_enhancement = nn.Sequential(
            nn.Conv2d(input_dim * 3, input_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim * 2),
            nn.GELU(),
            nn.Conv2d(input_dim * 2, input_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim * 2),
            nn.GELU(),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

        # 3. Edge Detector
        self.edge_detector = SobelKernel(input_dim)

        # 4. Local Attention
        self.local_attention = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 5. Feature Fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        # Local Branch Extractor
        group1_output = self.group_conv1(x)
        group2_output = self.group_conv2(x)
        group3_output = self.group_conv3(x)
        cat_output = torch.cat(
            [group1_output, group2_output, group3_output], dim=1
        )  # [B, 3C, H, W]

        # Details Enhancement
        f_enhanced = self.details_enhancement(cat_output)  # [B, C, H, W]

        # Local Attention加权
        local_weight = self.local_attention(f_enhanced)  # [B, 1, H, W]
        f_local = f_enhanced * local_weight  # [B, C, H, W]

        # Edge Detection
        f_edge = self.edge_detector(x)  # [B, C, H, W] ✓

        # Feature Fusion
        f_local = torch.cat([f_local, f_edge], dim=1)  # [B, 2C, H, W]
        f_local = self.feature_fusion(f_local)  # [B, C, H, W]

        return f_local


class LaplaceKernel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):

        device = x.device

        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device
        ).reshape((1, 1, 3, 3))

        laplace_out = F.conv2d(x, laplace, padding=1)
        return laplace_out


if __name__ == "__main__":
    x = torch.rand(1, 256, 16, 16)
    esca = ESCA(256)
    global_expert = GlobalExpert(256)
    region_expert = RegionExpert(256)
    local_expert = LocalExpert(256)

    y_esca = esca(x)
    y_global = global_expert(x)
    y_region = region_expert(x)
    y_local = local_expert(x)
    print(y_esca.shape)
    print(y_global.shape)
    print(y_region.shape)
    print(y_local.shape)

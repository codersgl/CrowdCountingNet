import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.flatten(2).transpose(1, 2).contiguous()
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
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_attn_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.max_attn_tokens = max_attn_tokens
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

        attn_input = x
        attn_h, attn_w = h, w
        if h * w > self.max_attn_tokens:
            ratio = math.sqrt(self.max_attn_tokens / float(h * w))
            attn_h = max(1, int(h * ratio))
            attn_w = max(1, int(w * ratio))
            attn_input = F.adaptive_avg_pool2d(x, output_size=(attn_h, attn_w))

        f_attention = self.multi_attention(
            attn_input
        )  # [batch_size, attn_h * attn_w, input_dim]
        f_enhanced = self.feature_enhancement(
            f_attention
        )  # [batch_size, attn_h * attn_w, input_dim]
        f_enhanced = f_enhanced.reshape(
            b, c, attn_h, attn_w
        )  # [batch_size, input_dim, attn_h, attn_w]

        if attn_h != h or attn_w != w:
            f_enhanced = F.interpolate(
                f_enhanced,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

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

        stacked = torch.cat(contrast_features, dim=1)  # [B, 6*C, H, W]
        aggregated = self.contrast_fusion(stacked)  # [B, C, H, W]

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
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).reshape((1, 1, 3, 3))
        self.register_buffer(
            "laplace", laplace.view(1, 1, 3, 3).repeat(input_dim, 1, 1, 1)
        )
        self.laplace: torch.Tensor
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):

        laplace_out = F.conv2d(x, self.laplace, padding=1, groups=self.input_dim)
        return laplace_out


class GaborFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        n_orientations: int = 8,
        lambd_: float = 1.0,
        sigma_ratio: float = 0.5,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations
        self.gamma = gamma
        self.sigma_ratio = sigma_ratio

        thetas = torch.linspace(0, 360 - 360 / n_orientations, n_orientations)

        gabor_kernels = []
        for theta in thetas:
            kernel = self._create_single_gabor_kernel(kernel_size, theta, lambd_)
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
            kernel_dw = kernel_4d.repeat(in_channels, 1, 1, 1)  # [C, 1, kH, kW]
            gabor_kernels.append(kernel_dw)

        gabor_kernel = torch.cat(gabor_kernels, dim=0)  # [C*8, 1, kH, kW]
        self.register_buffer("gabor_kernel", gabor_kernel)
        self.gabor_kernel: torch.Tensor

    def _create_single_gabor_kernel(
        self, kernel_size: int, theta_deg: torch.Tensor, lambd: float
    ):
        """生成单个方向的2D Gabor核"""
        theta_rad = torch.pi * theta_deg / 180.0
        sigma = self.sigma_ratio * lambd

        # 生成坐标网格
        x = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        y = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        x, y = torch.meshgrid(x, y, indexing="ij")

        # 坐标旋转
        x_prime = x * torch.cos(theta_rad) + y * torch.sin(theta_rad)
        y_prime = -x * torch.sin(theta_rad) + y * torch.cos(theta_rad)

        # Gabor核公式
        gabor = torch.exp(
            -(x_prime**2 + self.gamma**2 * y_prime**2) / (2 * sigma**2)
        ) * torch.cos(2 * torch.pi * x_prime / lambd)

        # 归一化
        gabor = gabor - torch.mean(gabor)
        gabor = gabor / (torch.norm(gabor) + 1e-8)

        return gabor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, C, H, W]
        输出: [B, C * n_orientations, H, W]
        """
        _, C, _, _ = x.shape
        outputs = []

        # 对每个方向分别应用 depthwise 卷积
        for i in range(self.n_orientations):
            # 提取第 i 个方向的 kernel: [C, 1, kH, kW]
            kernel_i = self.gabor_kernel[i * C : (i + 1) * C]  # [C, 1, kH, kW]

            # Depthwise 卷积：groups=C，每个输入通道独立卷积
            out_i = F.conv2d(
                x,
                kernel_i,
                padding=(self.kernel_size - 1) // 2,
                groups=C,
            )  # [B, C, H, W]
            outputs.append(out_i)

        # 拼接所有方向的输出: [B, C*8, H, W]
        return torch.cat(outputs, dim=1)


class TextureExpert(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gabor_filter_3x3 = GaborFilter(input_dim, kernel_size=3, lambd_=7)
        self.gabor_filter_5x5 = GaborFilter(input_dim, kernel_size=5, lambd_=15)
        self.highfreq_enhance = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sobel_kernel = SobelKernel(input_dim)
        self.laplace_kernel = LaplaceKernel(input_dim)

        gabor_channels = input_dim * 8  # 每个Gabor输出 C*8 通道

        total_channels = (
            gabor_channels * 2 + input_dim * 2
        )  # 2个Gabor + Sobel + Laplace

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.GELU(),
            nn.Conv2d(total_channels // 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

        self.channel_attention = ChannelAttention(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: [batch_size, input_dim, h, w]

        Returns:
            [TODO:return]
        """
        gabor_filter_3x3_output = self.gabor_filter_3x3(
            x
        )  # [batch_size, 8 * input_dim, h, w]
        gabor_filter_5x5_output = self.gabor_filter_5x5(
            x
        )  # [batch_size, 8 * input_dim, h, w]
        f_enhanced = x * self.highfreq_enhance(x)  # [batch_size, input_dim, h, w]
        f_sobel_filted = self.sobel_kernel(f_enhanced)  # [batch_size, input_dim, h, w]
        f_laplace_filted = self.laplace_kernel(
            f_enhanced
        )  # [batch_size, input_dim, h, w]
        f_texture = torch.cat(
            [
                gabor_filter_3x3_output,
                gabor_filter_5x5_output,
                f_sobel_filted,
                f_laplace_filted,
            ],
            dim=1,
        )  # [batch_size, 18 * input_dim, h, w]
        f_texture = self.feature_fusion(f_texture)  # [batch_size, input_dim, h, w]
        f_texture = self.channel_attention(f_texture)  # [batch_size, input_dim, h, w]
        return f_texture


class RelationalFeature(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Non-Local Block: Q/K 压缩到 C/2
        self.W_q = nn.Linear(input_dim, input_dim // 2)
        self.W_k = nn.Linear(input_dim, input_dim // 2)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

        # GAP → FC(2C) → LN → GELU → FC(C) → LN
        self.relation_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),  # [B, C]
            nn.Linear(input_dim, 2 * input_dim),
            nn.LayerNorm(2 * input_dim),
            nn.GELU(),
            nn.Linear(2 * input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        x: 原始输入特征 [B, C, H, W]
        part_features: Part Extractor 输出的部件特征 [B, C, H, W]
        """
        b, c, h, w = x.size()

        # Self-Attention (Non-Local Block)
        f_flat = x.reshape(b, c, -1).transpose(-1, -2)  # [B, H*W, C]
        Q = self.W_q(f_flat)  # [B, H*W, C/2]
        K = self.W_k(f_flat)  # [B, H*W, C/2]
        V = self.W_v(f_flat)  # [B, H*W, C]

        attn_score = Q @ K.transpose(-1, -2) / math.sqrt(c // 2)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_output = attn_weight @ V  # [B, H*W, C]
        attn_output = self.W_o(attn_output)  # [B, H*W, C]
        attn_output = attn_output.transpose(-1, -2).reshape(b, c, h, w)

        # 关系推理：从全局特征生成门控信号
        gate = self.relation_net(attn_output)  # [B, C]
        gate = gate.view(b, c, 1, 1)  # [B, C, 1, 1] 广播

        f_relational = attn_output * gate

        return f_relational


class PartExpert(nn.Module):
    def __init__(self, input_dim: int, N: int = 8) -> None:
        super().__init__()
        self.part_detector = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, N, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.part_extractor = nn.Sequential(
            nn.Conv2d(input_dim + N, (input_dim + N) // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d((input_dim + N) // 2),
            nn.GELU(),
            nn.Conv2d((input_dim + N) // 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )
        self.relational_features = RelationalFeature(input_dim=input_dim)

        self.spatial_features = nn.Sequential(
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=input_dim,
            ),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=3,
                padding=4,
                dilation=4,
                groups=input_dim,
            ),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(input_dim * 3, input_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim * 2),
            nn.GELU(),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_detected = self.part_detector(x)  # [batch_size, N, h, w]
        f_detected = torch.cat(
            [x, f_detected], dim=1
        )  # [batch_size, input_dim + N, h, w]
        f_part = self.part_extractor(f_detected)  # [batch_size, input_dim, h, w]
        f_relational = self.relational_features(f_part)  # [batch_size, input_dim, h, w]
        f_spatial = self.spatial_features(f_part)  # [batch_size, input_dim, h, w]
        f_part = torch.cat(
            [f_relational, f_part, f_spatial], dim=1
        )  # [batch_size, 3 * input_dim, h, w]
        f_part = self.feature_fuse(f_part)  # [batch_size, input_dim, h, w]
        return f_part


class SpatialContextEncoder(nn.Module):
    """
    空间上下文编码器：提取全局+局部空间特征
    论文3.4节: "extracts both global and local information"
    """

    def __init__(self, input_dim: int, spatial_grid: int = 4):
        super().__init__()
        self.spatial_grid = spatial_grid

        # 空间网格特征处理: [B, C, 4, 4] → [B, C//2, 4, 4] → [B, C//2 * 16]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim // 2),
            nn.GELU(),
            nn.Conv2d(input_dim // 2, input_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim // 2),
            nn.GELU(),
        )

        # 融合层: global(C) + spatial(C//2 * 16) → expert_scores(5)
        self.fusion = nn.Sequential(
            nn.Linear(
                input_dim + input_dim // 2 * spatial_grid * spatial_grid, input_dim
            ),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 5),  # 5个专家
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 来自MSP模块的特征
        Returns:
            scores: [B, 5] 专家重要性分数
        """
        B, C, _, _ = x.shape

        # 1. 全局特征: Global Average Pooling → [B, C]
        f_global = F.adaptive_avg_pool2d(x, 1).view(B, C)

        # 2. 空间网格特征: Adaptive Pool → [B, C, 4, 4]
        f_spatial = F.adaptive_avg_pool2d(x, self.spatial_grid)  # [B, C, 4, 4]
        f_spatial = self.spatial_conv(f_spatial)  # [B, C//2, 4, 4]
        f_spatial = f_spatial.view(B, -1)  # [B, C//2 * 16]

        # 3. 特征融合 → 专家分数
        f_fused = torch.cat([f_global, f_spatial], dim=1)  # [B, C + C//2*16]
        scores = self.fusion(f_fused)  # [B, 5]

        return scores


class DynamicRouter(nn.Module):
    """
    动态路由策略：训练/测试阶段使用不同策略
    论文3.5节: Hard routing (training) / Soft routing (testing)
    """

    def __init__(self, num_experts: int = 5, top_k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

    def _gumbel_softmax(
        self, logits: torch.Tensor, tau: float = 1.0, hard: bool = False
    ):
        """Gumbel-Softmax 实现（训练时硬路由）"""
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            # Straight-through estimator
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def forward(
        self, scores: torch.Tensor, training: bool = True, noise_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            scores: [B, 5] 专家重要性分数
            training: 是否训练阶段
            noise_scale: Gumbel噪声缩放系数，越大探索越充分
        Returns:
            weights: [B, 5] 专家权重（hard: 0/1, soft: 连续值）
        """
        if training:
            # Hard Top-K routing with straight-through gradient.
            # Forward uses hard discrete routing, backward follows soft selected probs.
            tau = max(self.temperature, 1e-6)
            gumbels = -torch.empty_like(scores).exponential_().log()
            noisy_scores = (scores + noise_scale * gumbels) / tau
            probs = F.softmax(noisy_scores, dim=-1)

            k = min(self.top_k, scores.size(-1))
            _, top_idx = torch.topk(noisy_scores, k, dim=-1)
            hard_mask = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)

            selected_soft = probs * hard_mask
            weights = hard_mask - selected_soft.detach() + selected_soft
        else:
            # Soft routing: Softmax 连续权重
            weights = F.softmax(scores / self.temperature, dim=-1)

        return weights


class MoELoss(nn.Module):
    """MoE辅助损失：balance（均匀使用）+ feature decorrelation（去相关）。"""

    def __init__(
        self,
        lambda_balance: float = 0.835,
        lambda_decorr: float = 1.0,
        usage_threshold: float = 0.1,
    ):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_decorr = lambda_decorr
        self.usage_threshold = usage_threshold

    def _balance_loss(self, expert_usage: torch.Tensor) -> torch.Tensor:
        """
        平衡损失: 鼓励专家均匀使用
        expert_usage: [5] 每个专家的使用频率（0~1）
        """
        num_experts = expert_usage.size(0)

        # Normalize usage to a valid probability distribution for entropy metrics.
        p = torch.clamp(expert_usage, min=0.0)
        p = p / (p.sum() + 1e-8)

        # 1. 熵 deficit: 最大化使用分布的熵
        current_entropy = -(p * torch.log(p + 1e-8)).sum()
        max_entropy = torch.log(
            torch.tensor(
                float(num_experts),
                device=expert_usage.device,
                dtype=expert_usage.dtype,
            )
        )
        l_entropy = max_entropy - current_entropy

        # 2. 低使用率惩罚
        l_low = torch.sum(torch.relu(self.usage_threshold - p) ** 2)

        return l_entropy + l_low

    def _feature_decorrelation_loss(self, expert_outputs: list) -> torch.Tensor:
        """特征去相关损失：惩罚专家对的 cos²(θ)，鼓励特征空间正交。
        使用 L2 归一化消除幅度影响，合并了原 diversity_loss 和 orthogonality_loss（两者实质等价）。
        """
        features = [
            F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in expert_outputs
        ]
        features = [F.normalize(f, p=2, dim=-1) for f in features]  # L2 归一化
        loss = torch.tensor(0.0, device=features[0].device, dtype=features[0].dtype)
        count = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dot = (features[i] * features[j]).sum(dim=-1)  # [B]
                loss += dot.pow(2).mean()
                count += 1
        return loss / count if count > 0 else loss

    def forward(self, expert_weights: torch.Tensor, expert_outputs: list) -> dict:
        """
        Args:
            expert_weights: [B, 5] 专家权重
            expert_outputs: List[Tensor] 5个专家的输出
        Returns:
            losses: dict 包含各辅助损失
        """
        expert_usage = expert_weights.mean(dim=0)  # [5]

        losses = {
            "l_balance": self._balance_loss(expert_usage),
            "l_decorr": self._feature_decorrelation_loss(expert_outputs),
        }

        total_aux = (
            self.lambda_balance * losses["l_balance"]
            + self.lambda_decorr * losses["l_decorr"]
        )
        losses["total_aux"] = total_aux

        return losses


class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        lambda_balance: float = 0.835,
        lambda_decorr: float = 1.0,
        ema_momentum: float = 0.99,
    ):
        super().__init__()

        self.top_k = top_k
        self.temperature = temperature_init
        self.temperature_min = temperature_min

        # ========== 1. 五个专家模块 ==========
        self.global_expert = GlobalExpert(input_dim)
        self.region_expert = RegionExpert(input_dim)
        self.local_expert = LocalExpert(input_dim)
        self.texture_expert = TextureExpert(input_dim)
        self.part_expert = PartExpert(input_dim)
        self.experts = nn.ModuleList(
            [
                self.global_expert,
                self.region_expert,
                self.local_expert,
                self.texture_expert,
                self.part_expert,
            ]
        )

        # ========== 2. 门控网络 ==========
        self.context_encoder = SpatialContextEncoder(input_dim)
        self.router = DynamicRouter(num_experts=5, top_k=top_k)

        # ========== 3. 辅助损失 ==========
        self.ema_momentum = ema_momentum
        self.aux_loss = MoELoss(
            lambda_balance=lambda_balance,
            lambda_decorr=lambda_decorr,
        )

        # 训练阶段标记
        self.register_buffer("step", torch.tensor(0))
        self.register_buffer(
            "ema_usage", torch.ones(5) / 5
        )  # 跨 batch EMA 专家使用率（用于监控）
        self.training_stage = "specialization"  # 'specialization' or 'coordination'

    def update_temperature(self, decay_rate: float = 0.9999):
        """逐步降低温度，使训练后期路由更确定"""
        self.temperature = max(self.temperature * decay_rate, self.temperature_min)
        # Sync temperature to router so DynamicRouter.forward() uses the decayed value.
        self.router.temperature = self.temperature
        self.step += 1

    def set_training_stage(self, stage: str):
        """设置训练阶段: 'specialization' 或 'coordination'"""
        self.training_stage = stage

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple:
        """
        Args:
            x: [B, C, H, W] 输入特征（来自MSP模块）
            training: 是否训练阶段
        Returns:
            fused_output: [B, C, H, W] 融合后的特征
            aux_losses: dict 辅助损失（训练时）
            expert_weights: [B, 5] 专家权重（可选，用于可视化）
        """
        # ========== 1. 前向传播通过5个专家 ==========
        expert_outputs = [expert(x) for expert in self.experts]  # List of [B, C, H, W]

        # ========== 2. 门控网络获取专家权重 ==========
        scores = self.context_encoder(x)  # [B, 5]

        if training and self.training_stage == "coordination":
            # 协调学习阶段：使用动态路由
            weights = self.router(scores, training=True)  # [B, 5], hard routing
        elif training and self.training_stage == "specialization":
            # 专家专精阶段：高噪声门控路由，使门控网络从第0轮获得梯度
            # noise_scale=2.0 使Gumbel噪声主导选择，确保多样性的同时训练门控网络。
            weights = self.router(scores, training=True, noise_scale=2.0)
        else:
            # 测试阶段：软路由
            weights = self.router(scores, training=False)  # [B, 5], soft routing

        # ========== 3. 特征融合: F_MoE = Σ w_i * F_Ei ==========
        fused = torch.zeros_like(expert_outputs[0])
        for i, (w, f) in enumerate(zip(weights.unbind(dim=1), expert_outputs)):
            # w: [B], f: [B, C, H, W]
            fused += w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * f

        # ========== 4. 计算辅助损失（仅训练时） ==========
        aux_losses = {}
        if training:
            # 以 EMA 跟踪跨 batch 专家使用率（用于 TensorBoard 监控，不参与 loss 梯度）
            with torch.no_grad():
                batch_usage = weights.detach().float().mean(dim=0)
                self.ema_usage = (
                    self.ema_momentum * self.ema_usage
                    + (1.0 - self.ema_momentum) * batch_usage
                )
            aux_losses = self.aux_loss(weights, expert_outputs)

        return fused, aux_losses, weights

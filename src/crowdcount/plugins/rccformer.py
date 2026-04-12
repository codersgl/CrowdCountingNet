from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CrossAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        Q = self.W_q(q)  # [B, N, C]
        K = self.W_k(k)  # [B, N, C]
        V = self.W_v(v)  # [B, N, C]

        attn = (Q @ K.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ V  # [B, N, C]
        return self.W_o(out)


class MFFM(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

        self.proj_f2 = nn.Conv2d(128, dim, kernel_size=1)
        self.proj_f3 = nn.Conv2d(320, dim, kernel_size=1)
        self.proj_f4 = nn.Conv2d(512, dim, kernel_size=1)

        self.up_f3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_f4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.proj_concat = nn.Conv2d(3 * dim, dim, kernel_size=1)

        self.cross_attn = CrossAttention(dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f2, f3, f4 = features
        b, _, h, w = f2.shape

        f2 = self.proj_f2(f2)
        f3 = self.proj_f3(f3)
        f4 = self.proj_f4(f4)

        # Step 2: 空间分辨率对齐
        f3 = self.up_f3(f3)
        f4 = self.up_f4(f4)

        q = self.proj_concat(torch.cat([f2, f3, f4], dim=1))  # Fc: Concat分支 -> Q
        k = f2 + f3 + f4  # Fa: Add分支    -> K
        v = k  # Fa: Add分支    -> V

        # Step 4: NCHW → [B, HW, C] (permute 确保每个 token 对应一个空间位置)
        q = q.permute(0, 2, 3, 1).reshape(b, -1, self.dim)
        k = k.permute(0, 2, 3, 1).reshape(b, -1, self.dim)
        v = v.permute(0, 2, 3, 1).reshape(b, -1, self.dim)

        # Step 5: 交叉注意力细粒度融合
        out = self.cross_attn(q, k, v)

        out = out.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)

        return out


class IDConv(nn.Module):
    """
    Input-dependent Deformable Convolution (IDConv)
    严格对齐论文公式 (9)-(10) 与 Figure 5
    支持可配置膨胀率，修复奇数通道/权重归一化/边界处理等问题
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_points = kernel_size**2  # 3x3 -> 9
        self.padding = (kernel_size // 2) * dilation

        # ========== Offset 分支 (绿色部分) ==========
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.num_points,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        # 初始化为0，训练初期退化为标准卷积，保障稳定性
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # ========== Weight 分支 (紫色部分) ==========
        # 修复: 使用 (C+1)//2 兼容奇数通道
        mid_channels = (in_channels + 1) // 2
        self.weight_net = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, 1, kernel_size=1),
        )

        # 输出投影
        self.out_proj = (
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            if in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # ===== 1. 生成偏移量 (Offset field) =====
        offset = self.offset_conv(x)  # [B, 2*K, H, W]
        offset = offset.view(B, self.num_points, 2, H, W)

        # ===== 2. 构建带膨胀率的采样网格 =====
        # 生成相对位置: dx, dy ∈ {-1, 0, 1} (kernel=3)
        ky, kx = torch.meshgrid(
            torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1),
            torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1),
            indexing="ij",
        )
        kx = kx.contiguous().view(-1).to(device) * self.dilation  # 应用膨胀率
        ky = ky.contiguous().view(-1).to(device) * self.dilation

        # 归一化至 grid_sample 的 [-1, 1] 坐标系
        norm_kx = kx / max(1, W - 1) * 2.0
        norm_ky = ky / max(1, H - 1) * 2.0

        # 像素自身的归一化坐标网格 [H, W]
        pixel_y, pixel_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        # 基础网格 [1, K, H, W, 2] = 像素绝对坐标 + kernel 相对偏移
        base_grid = torch.zeros(1, self.num_points, H, W, 2, device=device)
        for i in range(self.num_points):
            base_grid[0, i, :, :, 0] = pixel_x + norm_kx[i]
            base_grid[0, i, :, :, 1] = pixel_y + norm_ky[i]

        # 偏移量归一化 (添加 1e-6 避免除零)
        offset_norm = offset.clone()
        offset_norm[:, :, 0, :, :] = offset[:, :, 0, :, :] / (W - 1 + 1e-6) * 2.0
        offset_norm[:, :, 1, :, :] = offset[:, :, 1, :, :] / (H - 1 + 1e-6) * 2.0

        # 叠加偏移: [B, K, H, W, 2]
        grid = base_grid + offset_norm.permute(0, 1, 3, 4, 2)

        # ===== 3. 批量双线性采样 (优化: 避免循环) =====
        # 重塑 grid: [B*K, H, W, 2], 重复输入: [B*K, C, H, W]
        grid_flat = grid.view(B * self.num_points, H, W, 2)
        x_repeat = x.unsqueeze(1).expand(-1, self.num_points, -1, -1, -1)
        x_repeat = x_repeat.reshape(B * self.num_points, C, H, W)

        # 批量采样: [B*K, C, H, W]
        sampled_flat = F.grid_sample(
            x_repeat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # 恢复形状: [B, K, C, H, W]
        sampled = sampled_flat.view(B, self.num_points, C, H, W)

        # ===== 4. 生成动态权重 (Weight branch) =====
        # GAP over spatial dims: [B, K, C] → 转置为 [B, C, K] 以匹配 Conv1d(C, ...)
        gap_feat = sampled.mean(dim=[-1, -2]).permute(0, 2, 1)
        # MLP → [B, 1, K]
        weights = self.weight_net(gap_feat)
        # 🔑 添加 Softmax 归一化，提升数值稳定性
        weights = F.softmax(weights, dim=-1)
        # 广播维度: [B, 1, K, 1, 1] → [B, K, 1, 1, 1] 以匹配 sampled 的 dim=1
        weights = weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        # ===== 5. 加权求和输出 =====
        # y(p0) = Σ w(pn) * x(p0 + pn + Δpn)
        out = (sampled * weights).sum(dim=1)  # [B, C, H, W]

        return self.out_proj(out)


class ASAM(nn.Module):
    """
    Adaptive Scale-Aware Module (ASAM)
    严格对齐论文公式 (11) 与 Figure 3
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        # 阶段1: 基础 IDConv + 通道降维 (C → C/2)
        self.net1 = nn.Sequential(
            IDConv(dim, dim, kernel_size=3, dilation=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim // 2, kernel_size=1),
        )

        # 阶段2: 并行多膨胀率 IDConv (d={1,2,3})
        self.idconv1 = IDConv(dim // 2, dim // 2, kernel_size=3, dilation=1)
        self.idconv2 = IDConv(dim // 2, dim // 2, kernel_size=3, dilation=2)
        self.idconv3 = IDConv(dim // 2, dim // 2, kernel_size=3, dilation=3)

        # 阶段3: 融合 (3*C/2 → C)
        self.net2 = nn.Sequential(
            nn.BatchNorm2d(3 * dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * dim // 2, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 阶段1: 基础特征提取 + 降维
        x = self.net1(x)  # [B, C/2, H, W]

        # 阶段2: 三路并行多尺度感知
        f1 = self.idconv1(x)  # d=1: 局部细节
        f2 = self.idconv2(x)  # d=2: 中等上下文
        f3 = self.idconv3(x)  # d=3: 大范围语义

        # 阶段3: 拼接 + 融合
        f_cat = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C/2, H, W]
        out = self.net2(f_cat)  # [B, C, H, W]

        return out


class LocalAttention(nn.Module):
    """
    局部注意力分支 (对应论文公式 5)
    结构: Conv1×1 → Conv5×5^dw → Conv1×1 → ⊙ V
    """

    def __init__(self, dim: int, num_heads: int, kernel_size: int = 5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        padding = kernel_size // 2

        # 1×1 → 5×5 Depthwise → 1×1
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x_nchw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        x_nchw: [B, C, H, W]  输入特征
        v:      [B, N, HW, head_dim]  全局分支的 Value (已分头)
        return: [B, N, HW, head_dim]  局部注意力调制后的输出
        """
        B, _, H, W = x_nchw.shape
        N = self.num_heads
        head_dim = self.head_dim

        # 1. 提取局部细粒度特征 [B, C, H, W]
        local_feat = self.local_conv(x_nchw)

        # 2. 重塑为多头格式以对齐 V: [B, N, HW, head_dim]
        # 先拆分为 [B, N, head_dim, H, W]，再调整维度顺序并展平空间维
        local_feat = local_feat.reshape(B, N, head_dim, H, W)
        local_feat = local_feat.permute(0, 1, 3, 4, 2).reshape(B, N, -1, head_dim)

        # 3. 与全局 Value 逐元素相乘 (动态权重调制)
        return local_feat * v


class DEA(nn.Module):
    """
    Detail-Embedded Attention Core (对应论文公式 3-7)
    """

    def __init__(self, dim: int, num_heads: int = 8, alpha_init: float = 0.6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # 缩放因子 1/√d_h

        # 全局分支 QKV 投影 (公式 3)
        self.qkv = nn.Linear(dim, dim * 3)

        # 局部分支
        self.local_attn = LocalAttention(dim, num_heads, kernel_size=5)

        # 可学习融合参数 α (公式 6，论文 Fig.9 证实 0.6 最优)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 多头拼接与输出投影 (公式 7)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        """
        x_nhwc: [B, H, W, C]  输入特征 (NHWC 格式)
        """
        B, H, W, C = x_nhwc.shape
        HW = H * W
        N = self.num_heads
        head_dim = self.head_dim

        # ===== 1. QKV 投影与分头 (公式 3) =====
        # [B, HW, C] -> [B, HW, 3, N, head_dim] -> [3, B, N, HW, head_dim]
        qkv = self.qkv(x_nhwc).reshape(B, HW, 3, N, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, N, HW, head_dim]

        # ===== 2. 全局自注意力 (公式 4) =====
        attn_ga = (q @ k.transpose(-2, -1)) * self.scale
        attn_ga = F.softmax(attn_ga, dim=-1)
        out_ga = attn_ga @ v  # [B, N, HW, head_dim]

        # ===== 3. 局部注意力 (公式 5) =====
        # 转回 NCHW 供卷积使用
        x_nchw = x_nhwc.permute(0, 3, 1, 2)
        out_la = self.local_attn(x_nchw, v)  # [B, N, HW, head_dim]

        # ===== 4. 可学习融合 (公式 6) =====
        out = out_ga + self.alpha * out_la  # [B, N, HW, head_dim]

        # ===== 5. 多头拼接 + 线性投影 (公式 7) =====
        out = out.transpose(1, 2).reshape(B, HW, C)
        out = self.proj(out)

        return out.view(B, H, W, C)  # 返回 NHWC


class CFFN(nn.Module):
    """
    卷积前馈网络 (PVTv2 风格，对应论文公式 8 中的 CFFN)
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
            ),  # Depthwise
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return self.net(x_nchw)


class DEAB(nn.Module):
    """
    完整的 Detail-Embedded Attention Block (对应论文公式 8)
    结构: Pre-LN -> DEA -> Residual -> Pre-LN -> CFFN -> Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        alpha_init: float = 0.6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dea = DEA(dim, num_heads, alpha_init)
        self.norm2 = nn.LayerNorm(dim)
        self.cffn = CFFN(dim, mlp_ratio)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """
        x_nchw: [B, C, H, W]  输入特征
        return: [B, C, H, W]  输出特征
        """
        # 公式 8 第一步: X̂ = DEA(LN(X_in)) + X_in
        x_nhwc = x_nchw.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x_hat = x_nhwc + self.dea(self.norm1(x_nhwc))

        # 公式 8 第二步: X_out = CFFN(LN(X̂)) + X̂
        x_hat_nchw = x_hat.permute(0, 3, 1, 2)  # NHWC -> NCHW
        # LayerNorm 默认对最后一维归一化，故需转回 NHWC 输入
        x_norm = self.norm2(x_hat_nchw.permute(0, 2, 3, 1))
        x_out = x_hat_nchw + self.cffn(x_norm.permute(0, 3, 1, 2))

        return x_out  # [B, C, H, W]


class MFFMNeck(nn.Module):
    """MFFM-based neck adapted for VGG backbone channels.

    Replaces PA-FPN as a drop-in alternative.  Accepts the same
    ``[c3, c4, c5]`` list that ``Decoder_SPD_PAFPN`` uses and produces
    ``[B, dim, H/8, W/8]`` output (stride-8 from original image).

    All three feature maps are aligned to H/8 (c4's resolution) before
    cross-attention fusion:
        c3: 256 ch, H/4   → proj → dim, downsample ×2  → H/8
        c4: 512 ch, H/8   → proj → dim                 → H/8
        c5: 512 ch, H/16  → proj → dim, upsample ×2    → H/8
    """

    def __init__(
        self,
        C3_size: int = 256,
        C4_size: int = 512,
        C5_size: int = 512,
        dim: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.proj_f2 = nn.Conv2d(C3_size, dim, kernel_size=1)
        self.proj_f3 = nn.Conv2d(C4_size, dim, kernel_size=1)
        self.proj_f4 = nn.Conv2d(C5_size, dim, kernel_size=1)

        # c3 (H/4) → H/8: learnable stride-2 downsample
        self.down_f2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        # c5 (H/16) → H/8
        self.up_f4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.proj_concat = nn.Conv2d(3 * dim, dim, kernel_size=1)
        self.cross_attn = CrossAttention(dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [c3, c4, c5] from backbone
                c3: [B, C3, H/4,  W/4 ]
                c4: [B, C4, H/8,  W/8 ]
                c5: [B, C5, H/16, W/16]
        Returns:
            [B, dim, H/8, W/8]
        """
        f2, f3, f4 = features

        f2 = self.proj_f2(f2)
        f3 = self.proj_f3(f3)
        f4 = self.proj_f4(f4)

        # Spatial alignment to H/8 (c4's resolution)
        f2 = self.down_f2(f2)  # H/4  → H/8
        f4 = self.up_f4(f4)  # H/16 → H/8

        b, _, h, w = f3.shape  # H/8, W/8

        # Concat branch → Q,  Add branch → K, V
        q = self.proj_concat(torch.cat([f2, f3, f4], dim=1))
        k = f2 + f3 + f4
        v = k

        # NCHW → [B, HW, C] (permute 确保每个 token 对应一个空间位置)
        q = q.permute(0, 2, 3, 1).reshape(b, -1, self.dim)
        k = k.permute(0, 2, 3, 1).reshape(b, -1, self.dim)
        v = v.permute(0, 2, 3, 1).reshape(b, -1, self.dim)

        out = self.cross_attn(q, k, v)
        return out.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)  # [B, dim, H/8, W/8]


class DensityPredDEAB(nn.Module):
    """Density map prediction head using DEAB + ASAM blocks.

    Replaces the plain-conv ``Density_pred`` with attention-enhanced
    feature processing before the final 1-channel regression.

    Structure: N × DEAB → 1 × ASAM → Conv head (dim → 128 → 64 → 1)
    """

    def __init__(
        self,
        dim: int = 256,
        num_deab: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        alpha_init: float = 0.6,
    ) -> None:
        super().__init__()
        self.deab_blocks = nn.ModuleList(
            [
                DEAB(
                    dim, num_heads=num_heads, mlp_ratio=mlp_ratio, alpha_init=alpha_init
                )
                for _ in range(num_deab)
            ]
        )
        self.asam = ASAM(dim)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim, H, W]
        Returns:
            density_map: [B, 1, H, W]
        """
        for blk in self.deab_blocks:
            x = blk(x)
        x = self.asam(x)
        return self.conv_layers(x)

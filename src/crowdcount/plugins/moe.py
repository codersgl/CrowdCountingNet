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


# ---------------------------------------------------------------------------
# LightMoE: micro-expert conditional refinement (designed for gcn_moe mode)
# ---------------------------------------------------------------------------


class MicroBiasCorrector(nn.Module):
    """Channel-wise bias correction via 1×1 bottleneck.  ~0.04M at dim=256."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class MicroEdgeRefiner(nn.Module):
    """Depthwise local refinement for boundary precision.  ~0.05M at dim=256."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class MicroDensityAdapter(nn.Module):
    """Density-conditioned channel modulation.  ~0.04M at dim=256."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.density_proj = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.feat_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        if density is not None:
            if density.shape[-2:] != x.shape[-2:]:
                density = F.interpolate(
                    density, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            scale = self.density_proj(density)
        else:
            scale = x.detach().mean(dim=1, keepdim=True).sigmoid().expand_as(x)
        return self.feat_proj(x * scale) + x


class LightMoERouter(nn.Module):
    """Grid-level soft router for LightMoE (3 micro-experts)."""

    NUM_EXPERTS = 3

    def __init__(
        self, input_dim: int, grid_stride: int = 4, use_density_hint: bool = False
    ) -> None:
        super().__init__()
        self.grid_stride = grid_stride
        self.use_density_hint = (
            False  # Router should NOT see density_hint to avoid shortcut routing
        )
        in_ch = input_dim
        self.score_net = nn.Sequential(
            nn.Conv2d(in_ch, input_dim // 4, kernel_size=1),
            nn.BatchNorm2d(input_dim // 4),
            nn.GELU(),
            nn.Conv2d(input_dim // 4, self.NUM_EXPERTS, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, density_hint: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_density_hint and density_hint is not None:
            if density_hint.shape[-2:] != x.shape[-2:]:
                density_hint = F.interpolate(
                    density_hint,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inp = torch.cat([x, density_hint], dim=1)
        elif self.use_density_hint:
            # density_hint expected but not provided — use zeros
            inp = torch.cat(
                [
                    x,
                    torch.zeros(
                        x.shape[0],
                        1,
                        x.shape[2],
                        x.shape[3],
                        device=x.device,
                        dtype=x.dtype,
                    ),
                ],
                dim=1,
            )
        else:
            inp = x

        r = self.grid_stride
        H, W = inp.shape[-2:]
        if H >= r and W >= r:
            coarse = F.avg_pool2d(inp, kernel_size=r, stride=r)
            scores = self.score_net(coarse)
            scores = F.interpolate(
                scores, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            scores = self.score_net(inp)

        return F.softmax(scores, dim=1)


class LightMoE(nn.Module):
    """Lightweight 3-micro-expert MoE for post-GCN conditional refinement.

    Designed to be placed *after* GCN dual-stream fusion (gcn_moe mode).
    GCN handles relational reasoning; LightMoE handles conditional fine-tuning.
    Total parameter overhead: ~0.2M at input_dim=256.
    """

    NUM_EXPERTS = 3

    def __init__(
        self,
        input_dim: int = 256,
        grid_stride: int = 4,
        use_density_hint: bool = True,
        lambda_balance: float = 0.01,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MicroBiasCorrector(input_dim),
                MicroEdgeRefiner(input_dim),
                MicroDensityAdapter(input_dim),
            ]
        )
        self.router = LightMoERouter(
            input_dim, grid_stride=grid_stride, use_density_hint=False
        )
        self.balance_lambda = lambda_balance

        self.register_buffer(
            "ema_usage", torch.ones(self.NUM_EXPERTS) / self.NUM_EXPERTS
        )
        self.ema_momentum = 0.99

        # Learnable residual gate: initialized to 0 so LightMoE starts as identity
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Returns:
            fused: [B, C, H, W]
            aux_losses: dict
            weights: [B, 3, H, W]
        """
        expert_outputs: list[torch.Tensor] = []
        for expert in self.experts:
            if isinstance(expert, MicroDensityAdapter):
                expert_outputs.append(expert(x, density_hint))
            else:
                expert_outputs.append(expert(x))

        weights = self.router(
            x
        )  # [B, 3, H, W] — router sees only features, not density

        fused = torch.zeros_like(x)
        for k in range(self.NUM_EXPERTS):
            fused = fused + weights[:, k : k + 1] * expert_outputs[k]

        # Learnable residual gate: y = x + beta * (fused - x)
        # beta starts at 0 → identity at init, avoiding random perturbation through *100 regression
        out = x + self.beta * (fused - x)

        aux_losses: dict = {}
        if training:
            with torch.no_grad():
                batch_usage = weights.detach().float().mean(dim=(0, 2, 3))
                self.ema_usage = (
                    self.ema_momentum * self.ema_usage
                    + (1.0 - self.ema_momentum) * batch_usage
                )
            aux_losses = self._compute_balance_loss(weights)
        return out, aux_losses, weights

    def _compute_balance_loss(self, expert_weights: torch.Tensor) -> dict:
        """Entropy-based balance loss for 3 micro-experts."""

        if expert_weights.dim() == 4:
            usage = expert_weights.mean(dim=(0, 2, 3))  # [3]
        else:
            usage = expert_weights.mean(dim=0)

        p = torch.clamp(usage, min=0.0)
        p = p / (p.sum() + 1e-8)
        max_entropy = math.log(float(p.size(0)))
        current_entropy = -(p * torch.log(p + 1e-8)).sum()
        l_balance = max_entropy - current_entropy

        return {"l_balance": l_balance, "total_aux": self.balance_lambda * l_balance}

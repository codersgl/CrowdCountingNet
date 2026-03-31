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
# Compact MoE: 3 lightweight experts organised by error mode
# ---------------------------------------------------------------------------


class CountCalibrationExpert(nn.Module):
    """Addresses whole-image count bias via pooled cross-attention + SE.

    Captures global context to calibrate systematic over/under-counting.
    ~0.5 M params at input_dim=256.
    """

    def __init__(self, input_dim: int, global_tokens: int = 4) -> None:
        super().__init__()
        self.global_tokens = global_tokens
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.o_proj = nn.Linear(input_dim, input_dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(input_dim, input_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, input_dim, bias=False),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        N = H * W
        G = self.global_tokens

        q = self.q_proj(x.flatten(2).transpose(1, 2))  # [B, N, C]
        pooled = F.adaptive_avg_pool2d(x, G).flatten(2).transpose(1, 2)  # [B, G², C]
        k = self.k_proj(pooled)
        v = self.v_proj(pooled)

        attn = F.softmax(q @ k.transpose(-1, -2) / (C**0.5), dim=-1)
        out = self.o_proj(attn @ v).transpose(1, 2).view(B, C, H, W)
        out = self.norm(out)

        se_w = self.se(out).view(B, C, 1, 1)
        return out * se_w + x  # residual


class LocalizationExpert(nn.Module):
    """Addresses spatial localization error via ASPP multi-scale + edge cue.

    Combines dilated convolutions (receptive field diversity) with a learned
    spatial attention gate for boundary-aware refinement. ~0.4 M params.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        bd = max(1, input_dim // 4)
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(input_dim, bd, 3, padding=1, dilation=1),
            nn.BatchNorm2d(bd),
            nn.GELU(),
        )
        self.branch_d3 = nn.Sequential(
            nn.Conv2d(input_dim, bd, 3, padding=3, dilation=3),
            nn.BatchNorm2d(bd),
            nn.GELU(),
        )
        self.branch_d6 = nn.Sequential(
            nn.Conv2d(input_dim, bd, 3, padding=6, dilation=6),
            nn.BatchNorm2d(bd),
            nn.GELU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(bd * 3, bd, 3, padding=1),
            nn.BatchNorm2d(bd),
            nn.GELU(),
            nn.Conv2d(bd, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(bd * 3, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_d1(x)
        f3 = self.branch_d3(x)
        f6 = self.branch_d6(x)
        cat = torch.cat([f1, f3, f6], dim=1)
        gate = self.spatial_gate(cat)
        return self.fuse(cat * gate) + x  # residual


class DensityAdaptiveExpert(nn.Module):
    """Addresses density-dependent FP/FN via dual-branch density conditioning.

    High-density branch (small receptive field) for dense regions; low-density
    branch (large receptive field) for sparse regions.  No internal adaptive
    gate — density routing is handled solely by the outer MoE router.  ~0.3 M.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.density_attention = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
            nn.Conv2d(input_dim, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=4, dilation=4, groups=input_dim),
            nn.Conv2d(input_dim, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
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
            d_w = self.density_attention(density)
        else:
            d_w = x.detach().mean(dim=1, keepdim=True).sigmoid()

        f_high = self.high_branch(x * d_w)
        f_low = self.low_branch(x * (1.0 - d_w))
        return self.fuse(0.5 * f_high + 0.5 * f_low) + x  # residual, fixed 0.5 mix


# ---------------------------------------------------------------------------
# Grid-level soft router
# ---------------------------------------------------------------------------


class GridSoftRouter(nn.Module):
    """Patch-level soft routing: AvgPool(r) → score_net → Upsample.

    Produces spatially smooth routing weights that are consistent within
    r×r patches, reducing routing noise on small datasets.  All experts
    are always activated (no top-k hard mask), and weights sum to 1 in
    both training and evaluation — eliminating the train/test scale mismatch.
    """

    NUM_EXPERTS = 3

    def __init__(
        self,
        input_dim: int,
        grid_stride: int = 4,
        use_density_hint: bool = False,
    ) -> None:
        super().__init__()
        self.grid_stride = grid_stride
        self.use_density_hint = use_density_hint
        in_ch = input_dim + (1 if use_density_hint else 0)
        self.score_net = nn.Sequential(
            nn.Conv2d(in_ch, input_dim // 4, kernel_size=1),
            nn.BatchNorm2d(input_dim // 4),
            nn.GELU(),
            nn.Conv2d(input_dim // 4, self.NUM_EXPERTS, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns [B, 3, H, W] soft routing weights (sum=1 along dim=1)."""
        if self.use_density_hint and density_hint is not None:
            if density_hint.shape[-2:] != x.shape[-2:]:
                density_hint = F.interpolate(
                    density_hint,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            inp = torch.cat([x, density_hint], dim=1)
        else:
            inp = x

        # Coarse grid routing
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

        return F.softmax(scores, dim=1)  # [B, 3, H, W], sum=1


# ---------------------------------------------------------------------------
# Simplified auxiliary loss (balance only, no decorrelation)
# ---------------------------------------------------------------------------


class CompactMoELoss(nn.Module):
    """Entropy-based balance loss only.  No feature decorrelation — the three
    experts have distinct inductive biases by construction."""

    def __init__(self, lambda_balance: float = 0.01) -> None:
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(self, expert_weights: torch.Tensor) -> dict:
        """
        Args:
            expert_weights: [B, 3, H, W]
        Returns:
            dict with l_balance and total_aux.
        """
        if expert_weights.dim() == 4:
            usage = expert_weights.mean(dim=(0, 2, 3))  # [3]
        else:
            usage = expert_weights.mean(dim=0)

        p = torch.clamp(usage, min=0.0)
        p = p / (p.sum() + 1e-8)
        num_experts = p.size(0)
        max_entropy = math.log(float(num_experts))
        current_entropy = -(p * torch.log(p + 1e-8)).sum()
        l_balance = max_entropy - current_entropy

        total = self.lambda_balance * l_balance
        return {"l_balance": l_balance, "total_aux": total}


# ---------------------------------------------------------------------------
# Compact MoE module
# ---------------------------------------------------------------------------


class MoE(nn.Module):
    """Compact 3-expert Mixture-of-Experts with grid-level soft routing.

    Experts are organised by *error mode* (count bias / localisation /
    density-dependent FP-FN) rather than by feature type, ensuring minimal
    functional overlap.  All experts are always activated (soft routing),
    and routing weights sum to 1 identically in training and evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        top_k: int = 3,  # kept for API compat; ignored (all experts active)
        temperature_init: float = 1.0,
        temperature_min: float = 0.4,
        lambda_balance: float = 0.01,
        lambda_decorr: float = 0.0,  # kept for API compat; ignored
        ema_momentum: float = 0.99,
        use_density_hint: bool = False,
        grid_stride: int = 4,
    ) -> None:
        super().__init__()
        self.num_experts = 3

        # Experts
        self.count_expert = CountCalibrationExpert(input_dim)
        self.localization_expert = LocalizationExpert(input_dim)
        self.density_expert = DensityAdaptiveExpert(input_dim)
        self.experts = nn.ModuleList(
            [
                self.count_expert,
                self.localization_expert,
                self.density_expert,
            ]
        )

        # Router
        self.router = GridSoftRouter(
            input_dim,
            grid_stride=grid_stride,
            use_density_hint=use_density_hint,
        )
        # Kept for API compat with DSGCNet.get_moe_gating_parameters()
        self.context_encoder = self.router

        # Loss
        self.aux_loss = CompactMoELoss(lambda_balance=lambda_balance)

        # Monitoring buffers
        self.register_buffer("step", torch.tensor(0))
        self.register_buffer("ema_usage", torch.ones(3) / 3)
        self.ema_momentum = ema_momentum
        # Kept for API compat
        self.temperature = temperature_init
        self.temperature_min = temperature_min
        self._current_noise_scale: float = 0.0  # no noise in soft routing

    def update_temperature(self, decay_rate: float = 0.9999) -> None:
        self.temperature = max(self.temperature * decay_rate, self.temperature_min)
        self.router.temperature = self.temperature
        self.step += 1

    def update_noise_scale(self, progress: float) -> None:
        pass  # no noise needed for soft routing

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple:
        """
        Returns:
            fused_output: [B, C, H, W]
            aux_losses: dict (non-empty only when training)
            expert_weights: [B, 3, H, W]
        """
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            if isinstance(expert, DensityAdaptiveExpert):
                expert_outputs.append(expert(x, density_hint))
            else:
                expert_outputs.append(expert(x))

        # Soft routing weights — identical path for train & eval
        weights = self.router(x, density_hint=density_hint)  # [B, 3, H, W]

        # Weighted sum
        fused = torch.zeros_like(x)
        for k in range(self.num_experts):
            fused = fused + weights[:, k : k + 1] * expert_outputs[k]

        # Aux losses
        aux_losses: dict = {}
        if training:
            with torch.no_grad():
                batch_usage = weights.detach().float().mean(dim=(0, 2, 3))
                self.ema_usage = (
                    self.ema_momentum * self.ema_usage
                    + (1.0 - self.ema_momentum) * batch_usage
                )
            aux_losses = self.aux_loss(weights)

        return fused, aux_losses, weights


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
        self.aux_loss = CompactMoELoss(lambda_balance=lambda_balance)

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
            aux_losses = self.aux_loss(weights)

        return out, aux_losses, weights

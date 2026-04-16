"""MoE-Lite: lightweight Mixture-of-Experts with density-guided soft routing.

Three specialised experts handle different crowd density regimes:
  - DenseRegionExpert:  small receptive field for tightly-packed crowds
  - SparseRegionExpert: large receptive field (dilated convs) for sparse scenes
  - BoundaryExpert:     edge-aware convolutions for crowd boundaries

All experts are always active (soft routing) to avoid train/test mismatch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SE (Squeeze-and-Excitation) block
# ---------------------------------------------------------------------------


class _SE(nn.Module):
    """Channel squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ---------------------------------------------------------------------------
# Expert implementations
# ---------------------------------------------------------------------------


class DenseRegionExpert(nn.Module):
    """Small receptive field expert for high-density regions.

    Architecture: 1×1 bottleneck → 3×3 depthwise → 1×1 expand → SE
    """

    def __init__(self, dim: int = 256, expansion: int = 2) -> None:
        super().__init__()
        mid = dim * expansion
        self.net = nn.Sequential(
            # Bottleneck
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Depthwise 3×3
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Project back
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.se = _SE(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.net(x)) + x)


class SparseRegionExpert(nn.Module):
    """Large receptive field expert for sparse / low-density regions.

    Architecture: dilated 3×3 (d=2) → dilated 3×3 (d=4) → 1×1 → SE
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.se = _SE(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.net(x)) + x)


class BoundaryExpert(nn.Module):
    """Boundary-aware expert for crowd edges.

    Uses a Sobel-initialised 3×3 conv to detect edges, followed by a
    refinement convolution and channel recalibration.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        # Sobel-initialised edge conv (learnable)
        self.edge_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self._init_sobel(self.edge_conv)
        self.edge_bn = nn.BatchNorm2d(dim)
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.se = _SE(dim)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _init_sobel(conv: nn.Conv2d) -> None:
        """Initialise depthwise conv with horizontal Sobel kernel."""
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        with torch.no_grad():
            for i in range(conv.weight.shape[0]):
                # Alternate horizontal / vertical Sobel for diversity
                if i % 2 == 0:
                    conv.weight[i, 0] = sobel
                else:
                    conv.weight[i, 0] = sobel.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = F.relu(self.edge_bn(self.edge_conv(x)), inplace=True)
        out = self.refine(edge)
        return self.act(self.se(out) + x)


# ---------------------------------------------------------------------------
# Density-guided soft router
# ---------------------------------------------------------------------------


class DensityGuidedRouter(nn.Module):
    """Grid-level soft router conditioned on the density map.

    Pools features + density to a coarse grid, computes per-expert weights
    via a lightweight MLP, then upsamples to full spatial resolution.
    All experts receive non-zero weight (soft routing).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_experts: int = 3,
        grid_stride: int = 4,
        temperature_init: float = 1.0,
        temperature_min: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.grid_stride = grid_stride
        self.temperature_min = temperature_min

        # Input: feature (dim) + density (1) → coarse grid
        in_dim = feature_dim + 1
        hidden = max(in_dim // 4, 32)
        self.score_net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )
        self.temperature = nn.Parameter(
            torch.tensor(temperature_init), requires_grad=False
        )

    @torch.no_grad()
    def update_temperature(self, decay_rate: float = 0.9999) -> None:
        self.temperature.clamp_(min=self.temperature_min)
        self.temperature.mul_(decay_rate)

    def forward(self, features: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Return expert weights [B, num_experts, H, W]."""
        B, C, H, W = features.shape

        # Pool to coarse grid
        if self.grid_stride > 1:
            feat_coarse = F.adaptive_avg_pool2d(
                features, (max(1, H // self.grid_stride), max(1, W // self.grid_stride))
            )
            den_coarse = F.adaptive_avg_pool2d(density, feat_coarse.shape[-2:])
        else:
            feat_coarse = features
            den_coarse = density

        x = torch.cat([feat_coarse, den_coarse], dim=1)
        logits = self.score_net(x)  # [B, E, h, w]
        weights = F.softmax(logits / self.temperature, dim=1)  # soft routing

        # Upsample back to full resolution
        if weights.shape[-2:] != (H, W):
            weights = F.interpolate(
                weights, size=(H, W), mode="bilinear", align_corners=False
            )

        return weights


# ---------------------------------------------------------------------------
# Balance loss
# ---------------------------------------------------------------------------


class _BalanceLoss(nn.Module):
    """Entropy-based expert balance loss (encourages uniform expert usage)."""

    def __init__(self, lambda_balance: float = 0.01) -> None:
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: [B, num_experts, H, W]."""
        # Average expert usage across spatial dims and batch
        avg_usage = weights.mean(dim=(0, 2, 3))  # [E]
        # Maximum entropy for uniform distribution
        E = weights.shape[1]
        max_entropy = math.log(E)
        # Current entropy
        entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum()
        # Loss: gap from maximum entropy
        return self.lambda_balance * (max_entropy - entropy)


# ---------------------------------------------------------------------------
# MoE-Lite container
# ---------------------------------------------------------------------------


class MoELite(nn.Module):
    """Lightweight 3-expert MoE with density-guided soft routing.

    All three experts (dense, sparse, boundary) are always active.  The router
    produces per-pixel soft weights conditioned on the predicted density map.
    A learnable residual gate ``beta`` controls how much the MoE output
    deviates from the identity (initialised to 0 → starts as pass-through).
    """

    def __init__(
        self,
        dim: int = 256,
        grid_stride: int = 4,
        temperature_init: float = 1.0,
        temperature_min: float = 0.3,
        lambda_balance: float = 0.01,
        dense_expansion: int = 2,
    ) -> None:
        super().__init__()

        self.experts = nn.ModuleList(
            [
                DenseRegionExpert(dim, expansion=dense_expansion),
                SparseRegionExpert(dim),
                BoundaryExpert(dim),
            ]
        )
        self.router = DensityGuidedRouter(
            feature_dim=dim,
            num_experts=3,
            grid_stride=grid_stride,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
        )
        self.balance_loss = _BalanceLoss(lambda_balance=lambda_balance)

        # Learnable residual gate (starts at 0 → identity at init)
        self.beta = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        """Expose router temperature for engine logging."""
        return float(self.router.temperature.item())

    def update_temperature(self, decay_rate: float = 0.9999) -> None:
        self.router.update_temperature(decay_rate)

    def update_noise_scale(self, progress: float) -> None:
        """No-op: MoELite uses temperature annealing, not noise scaling."""

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            fused:      [B, dim, H, W] — MoE output features
            aux_losses: dict with 'balance_loss' and 'total_aux'
            weights:    [B, 3, H, W] — per-expert routing weights
        """
        weights = self.router(x, density.detach())  # [B, 3, H, W]

        # Run all experts
        expert_outs = [expert(x) for expert in self.experts]

        # Weighted sum
        mixed = torch.zeros_like(x)
        for i, out in enumerate(expert_outs):
            mixed = mixed + weights[:, i : i + 1] * out

        # Residual gate: out = x + beta * (mixed - x)
        fused = x + self.beta.sigmoid() * (mixed - x)

        # Aux losses
        aux: dict[str, torch.Tensor] = {}
        if training:
            bal = self.balance_loss(weights)
            aux["l_balance"] = bal
            aux["total_aux"] = bal
        else:
            zero = torch.tensor(0.0, device=x.device)
            aux["l_balance"] = zero
            aux["total_aux"] = zero

        return fused, aux, weights

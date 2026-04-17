"""MoE-Lite: lightweight Mixture-of-Experts with density-guided soft routing.

Three specialised experts handle different crowd density regimes:
  - DenseRegionExpert:  small receptive field for tightly-packed crowds
  - SparseRegionExpert: large receptive field (dilated convs) for sparse scenes
  - BoundaryExpert:     edge-aware convolutions for crowd boundaries

All experts are always active (soft routing) to avoid train/test mismatch.
Each expert is additionally gated by the density map (Density-Gated Expert),
so its output is modulated by local crowd density before being mixed.

The router uses multi-scale density encoding for richer routing signals,
and the balance loss includes a decorrelation term that encourages
different experts to specialise on different spatial regions.
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

    Architecture: 1×1 bottleneck → 3×3 depthwise → 1×1 expand → SE.
    Optionally density-gated: the expert output is modulated by the density
    map so that it is more active in high-density regions.
    """

    def __init__(
        self, dim: int = 256, expansion: int = 2, use_density_gate: bool = True
    ) -> None:
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

        # Density-gated residual: gate = sigmoid(conv(cat(expert_out, density)))
        self.use_density_gate = use_density_gate
        if use_density_gate:
            self.density_gate = nn.Sequential(
                nn.Conv2d(dim + 1, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.Sigmoid(),
            )

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.act(self.se(self.net(x)) + x)
        if self.use_density_gate and density is not None:
            gate = self.density_gate(torch.cat([out, density], dim=1))
            out = out * gate + x * (1 - gate)
        return out


class SparseRegionExpert(nn.Module):
    """Large receptive field expert for sparse / low-density regions.

    Architecture: dilated 3×3 (d=2) → dilated 3×3 (d=4) → 1×1 → SE.
    Optionally density-gated: modulated by the density map so that it is
    more active in low-density regions.
    """

    def __init__(self, dim: int = 256, use_density_gate: bool = True) -> None:
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

        self.use_density_gate = use_density_gate
        if use_density_gate:
            self.density_gate = nn.Sequential(
                nn.Conv2d(dim + 1, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.Sigmoid(),
            )

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.act(self.se(self.net(x)) + x)
        if self.use_density_gate and density is not None:
            gate = self.density_gate(torch.cat([out, density], dim=1))
            out = out * gate + x * (1 - gate)
        return out


class BoundaryExpert(nn.Module):
    """Boundary-aware expert for crowd edges.

    Uses a Sobel-initialised 3×3 conv to detect edges, followed by a
    refinement convolution and channel recalibration.  Optionally
    density-gated so that it focuses on boundary regions.
    """

    def __init__(self, dim: int = 256, use_density_gate: bool = True) -> None:
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

        self.use_density_gate = use_density_gate
        if use_density_gate:
            self.density_gate = nn.Sequential(
                nn.Conv2d(dim + 1, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.Sigmoid(),
            )

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

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        edge = F.relu(self.edge_bn(self.edge_conv(x)), inplace=True)
        out = self.refine(edge)
        out = self.act(self.se(out) + x)
        if self.use_density_gate and density is not None:
            gate = self.density_gate(torch.cat([out, density], dim=1))
            out = out * gate + x * (1 - gate)
        return out


# ---------------------------------------------------------------------------
# Density-guided soft router
# ---------------------------------------------------------------------------


class DensityGuidedRouter(nn.Module):
    """Grid-level soft router conditioned on multi-scale density encoding.

    Pools features + multi-scale density to a coarse grid, computes per-expert
    weights via a lightweight MLP, then upsamples to full spatial resolution.
    All experts receive non-zero weight (soft routing).

    The density encoding uses three scales (full, 1/2, 1/4) to capture both
    local density and surrounding context, providing richer routing signals.
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

        # Multi-scale density: full (1ch) + 1/2 pool (1ch) + 1/4 pool (1ch) = 3ch
        # Input: feature (dim) + multi-scale density (3) → coarse grid
        in_dim = feature_dim + 3
        hidden = max(in_dim // 2, 64)  # increased capacity from in_dim // 4
        self.score_net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
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

    def _multi_scale_density(
        self, density: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Encode density at three scales: full, 1/2, 1/4.

        All outputs are resized to (target_h, target_w) for concatenation.
        """
        d1 = density  # full resolution
        d2 = F.adaptive_avg_pool2d(
            density, (max(1, density.shape[2] // 2), max(1, density.shape[3] // 2))
        )
        d3 = F.adaptive_avg_pool2d(
            density, (max(1, density.shape[2] // 4), max(1, density.shape[3] // 4))
        )

        # Resize all to target size
        if d1.shape[-2:] != (target_h, target_w):
            d1 = F.interpolate(
                d1, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        if d2.shape[-2:] != (target_h, target_w):
            d2 = F.interpolate(
                d2, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        if d3.shape[-2:] != (target_h, target_w):
            d3 = F.interpolate(
                d3, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return torch.cat([d1, d2, d3], dim=1)  # [B, 3, h, w]

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

        # Multi-scale density encoding
        ms_density = self._multi_scale_density(
            den_coarse, feat_coarse.shape[2], feat_coarse.shape[3]
        )

        x = torch.cat([feat_coarse, ms_density], dim=1)
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
    """Expert balance loss with entropy + decorrelation terms.

    The entropy term encourages uniform average expert usage, while the
    decorrelation term encourages different experts to specialise on different
    spatial regions (low cross-correlation of expert weight maps).
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_decorr: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_decorr = lambda_decorr

    def forward(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """weights: [B, num_experts, H, W].

        Returns (total_balance_loss, decorrelation_loss).
        Both terms are guaranteed non-negative.
        """
        E = weights.shape[1]

        # --- Entropy balance loss ---
        avg_usage = weights.mean(dim=(0, 2, 3))  # [E]
        max_entropy = math.log(E)
        entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum()
        l_balance = self.lambda_balance * (max_entropy - entropy)

        # --- Decorrelation loss ---
        # Penalise POSITIVE correlation between expert weight maps.
        # Anti-correlation (desired specialisation) yields zero loss.
        # ReLU ensures the loss is non-negative — it should never act as
        # a reward that subsidises the main training loss.
        l_decorr = torch.tensor(0.0, device=weights.device)
        if E > 1:
            # Flatten spatial dims: [B, E, H*W]
            w_flat = weights.flatten(2)
            # Center per expert
            w_centered = w_flat - w_flat.mean(dim=2, keepdim=True)
            # Pairwise dot products (proxy for correlation)
            for i in range(E):
                for j in range(i + 1, E):
                    corr_ij = (w_centered[:, i] * w_centered[:, j]).sum(dim=1).mean()
                    l_decorr = l_decorr + F.relu(corr_ij)  # only penalise positive corr
            l_decorr = self.lambda_decorr * l_decorr / (E * (E - 1) / 2)

        return l_balance + l_decorr, l_decorr


# ---------------------------------------------------------------------------
# MoE-Lite container
# ---------------------------------------------------------------------------


class MoELite(nn.Module):
    """Lightweight 3-expert MoE with density-guided soft routing.

    All three experts (dense, sparse, boundary) are always active.  Each expert
    is density-gated so its output is modulated by the local density value.
    The router produces per-pixel soft weights conditioned on multi-scale
    density encoding.  A learnable residual gate ``beta`` controls how much
    the MoE output deviates from the identity (initialised to 0 → pass-through).
    """

    def __init__(
        self,
        dim: int = 256,
        grid_stride: int = 4,
        temperature_init: float = 1.0,
        temperature_min: float = 0.3,
        lambda_balance: float = 0.05,
        dense_expansion: int = 2,
        use_density_gate: bool = True,
        lambda_decorr: float = 0.1,
        lambda_diversity: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_diversity = lambda_diversity

        self.experts = nn.ModuleList(
            [
                DenseRegionExpert(
                    dim, expansion=dense_expansion, use_density_gate=use_density_gate
                ),
                SparseRegionExpert(dim, use_density_gate=use_density_gate),
                BoundaryExpert(dim, use_density_gate=use_density_gate),
            ]
        )
        self.router = DensityGuidedRouter(
            feature_dim=dim,
            num_experts=3,
            grid_stride=grid_stride,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
        )
        self.balance_loss = _BalanceLoss(
            lambda_balance=lambda_balance,
            lambda_decorr=lambda_decorr,
        )

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
            aux_losses: dict with 'l_balance', 'l_decorr' and 'total_aux'
            weights:    [B, 3, H, W] — per-expert routing weights
        """
        weights = self.router(x, density.detach())  # [B, 3, H, W]

        # Run all experts with density gating
        expert_outs = [expert(x, density.detach()) for expert in self.experts]

        # Weighted sum
        mixed = torch.zeros_like(x)
        for i, out in enumerate(expert_outs):
            mixed = mixed + weights[:, i : i + 1] * out

        # Residual gate: out = x + beta * (mixed - x)
        fused = x + self.beta.sigmoid() * (mixed - x)

        # Aux losses
        aux: dict[str, torch.Tensor] = {}
        if training:
            bal_total, l_decorr = self.balance_loss(weights)
            # Expert output diversity: penalise high cosine similarity
            l_diversity = self._diversity_loss(expert_outs)
            aux["l_balance"] = bal_total
            aux["l_decorr"] = l_decorr
            aux["l_diversity"] = l_diversity
            aux["total_aux"] = bal_total + l_diversity
        else:
            zero = torch.tensor(0.0, device=x.device)
            aux["l_balance"] = zero
            aux["l_decorr"] = zero
            aux["l_diversity"] = zero
            aux["total_aux"] = zero

        return fused, aux, weights

    def _diversity_loss(self, expert_outs: list[torch.Tensor]) -> torch.Tensor:
        """Penalise high cosine similarity between expert outputs.

        Encourages experts to produce diverse feature representations rather
        than collapsing to identical outputs.
        """
        E = len(expert_outs)
        if E < 2 or self.lambda_diversity <= 0:
            return torch.tensor(0.0, device=expert_outs[0].device)

        # Flatten spatial dims: [B, C, H*W]
        flats = [o.flatten(2) for o in expert_outs]
        loss = torch.tensor(0.0, device=expert_outs[0].device)
        n_pairs = 0
        for i in range(E):
            for j in range(i + 1, E):
                # Cosine similarity per sample, averaged over batch
                cos_sim = F.cosine_similarity(flats[i], flats[j], dim=1)  # [B, H*W]
                loss = loss + cos_sim.mean()
                n_pairs += 1
        return self.lambda_diversity * loss / n_pairs

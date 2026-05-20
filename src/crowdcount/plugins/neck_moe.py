"""Scale-aware Mixture-of-Experts adapter for DSGCNet neck features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    dilation: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    padding = ((kernel_size - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class LocalDetailExpert(nn.Module):
    """Compact local refinement expert for high-resolution neck details."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class MidScaleExpert(nn.Module):
    """Medium receptive-field expert with lightweight channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out * self.se(out) + x


class ContextExpert(nn.Module):
    """Large-context expert using dilated depthwise convolutions."""

    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        branch_channels = max(channels // len(rates), 16)
        self.branches = nn.ModuleList(
            [
                _conv_bn_act(
                    channels,
                    branch_channels,
                    kernel_size=3,
                    dilation=rate,
                    groups=1,
                )
                for rate in rates
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        fused_channels = branch_channels * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        pooled = self.pool_branch(x)
        pooled = F.interpolate(
            pooled, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1)) + x


class CrossScaleExpert(nn.Module):
    """Expert that reuses P3/P4/P5 neck intermediates when available."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fallback = LocalDetailExpert(channels)

    def forward(
        self,
        x: torch.Tensor,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if pyramid is None:
            return self.fallback(x)

        resized = [
            F.interpolate(level, size=x.shape[-2:], mode="nearest")
            if level.shape[-2:] != x.shape[-2:]
            else level
            for level in pyramid
        ]
        cat = torch.cat(resized, dim=1)
        weights = F.softmax(self.scale_gate(cat), dim=1)
        weighted = torch.cat(
            [weights[:, idx : idx + 1] * level for idx, level in enumerate(resized)],
            dim=1,
        )
        return self.project(weighted) + x


class ScaleAwareGridRouter(nn.Module):
    """Grid-level router that predicts spatial expert weights."""

    def __init__(
        self,
        channels: int,
        num_experts: int,
        grid_stride: int = 4,
        use_pyramid_context: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if grid_stride < 1:
            raise ValueError("grid_stride must be >= 1")
        self.grid_stride = grid_stride
        self.use_pyramid_context = use_pyramid_context
        in_channels = channels * (2 if use_pyramid_context else 1)
        hidden = max(channels // 4, 32)
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_experts, kernel_size=1),
        )

    def _context_from_pyramid(
        self,
        x: torch.Tensor,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.use_pyramid_context:
            return x
        if pyramid is None:
            return torch.cat([x, torch.zeros_like(x)], dim=1)
        levels = [
            F.interpolate(level, size=x.shape[-2:], mode="nearest")
            if level.shape[-2:] != x.shape[-2:]
            else level
            for level in pyramid
        ]
        return torch.cat([x, torch.stack(levels, dim=0).mean(dim=0)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        router_input = self._context_from_pyramid(x, pyramid)
        stride = self.grid_stride
        height, width = router_input.shape[-2:]
        if height >= stride and width >= stride:
            coarse = F.avg_pool2d(router_input, kernel_size=stride, stride=stride)
            scores = self.score_net(coarse)
            scores = F.interpolate(
                scores, size=(height, width), mode="bilinear", align_corners=False
            )
        else:
            scores = self.score_net(router_input)
        return F.softmax(scores, dim=1)


class NeckMoELoss(nn.Module):
    """Entropy balance loss for spatial expert routing."""

    def __init__(self, lambda_balance: float = 0.01) -> None:
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        usage = weights.mean(dim=(0, 2, 3))
        usage = usage / usage.sum().clamp_min(1e-8)
        max_entropy = math.log(float(usage.numel()))
        entropy = -(usage * torch.log(usage + 1e-8)).sum()
        l_balance = max_entropy - entropy
        return {
            "neck_l_balance": l_balance,
            "neck_entropy": entropy,
            "total_aux": self.lambda_balance * l_balance,
        }


class NeckScaleMoE(nn.Module):
    """Scale-aware MoE adapter for post-neck feature refinement."""

    _VALID_ROUTING = {"soft", "topk", "sparse_topk"}

    def __init__(
        self,
        in_channels: int = 256,
        num_experts: int = 4,
        grid_stride: int = 4,
        routing: str = "soft",
        top_k: int = 0,
        use_pyramid_context: bool = True,
        lambda_balance: float = 0.01,
        gate_init: float = 0.0,
        context_rates: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        routing = routing.lower()
        if routing not in self._VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(self._VALID_ROUTING)}, got {routing!r}"
            )
        if num_experts not in {3, 4}:
            raise ValueError("NeckScaleMoE v1 supports num_experts=3 or 4")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if routing in {"topk", "sparse_topk"} and top_k <= 0:
            raise ValueError("top_k must be > 0 when routing uses top-k")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")

        self.num_experts = num_experts
        self.routing = routing
        self.top_k = top_k
        self.use_pyramid_context = use_pyramid_context
        self.router = ScaleAwareGridRouter(
            in_channels,
            num_experts=num_experts,
            grid_stride=grid_stride,
            use_pyramid_context=use_pyramid_context,
        )
        experts: list[nn.Module] = [
            LocalDetailExpert(in_channels),
            MidScaleExpert(in_channels),
            ContextExpert(in_channels, rates=context_rates),
        ]
        if num_experts == 4:
            experts.append(CrossScaleExpert(in_channels))
        self.experts = nn.ModuleList(experts)
        self.aux_loss = NeckMoELoss(lambda_balance=lambda_balance)
        self.beta = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer(
            "ema_usage", torch.ones(num_experts, dtype=torch.float32) / num_experts
        )
        self.ema_momentum = 0.99

    def _apply_topk(self, weights: torch.Tensor) -> torch.Tensor:
        if self.routing == "soft":
            return weights
        values, indices = torch.topk(weights, k=self.top_k, dim=1)
        mask = torch.zeros_like(weights).scatter_(1, indices, 1.0)
        masked = weights * mask
        return masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        x: torch.Tensor,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if training is None:
            training = self.training
        pyramid_for_router = pyramid if self.use_pyramid_context else None
        weights = self.router(x, pyramid_for_router)
        weights = self._apply_topk(weights)

        expert_outputs: list[torch.Tensor] = []
        for expert in self.experts:
            if isinstance(expert, CrossScaleExpert):
                expert_outputs.append(expert(x, pyramid))
            else:
                expert_outputs.append(expert(x))

        fused = torch.zeros_like(x)
        for idx, expert_out in enumerate(expert_outputs):
            fused = fused + weights[:, idx : idx + 1] * expert_out
        out = x + self.beta.tanh() * (fused - x)

        aux_losses: dict[str, torch.Tensor] = {}
        if training:
            with torch.no_grad():
                batch_usage = weights.detach().float().mean(dim=(0, 2, 3))
                self.ema_usage = (
                    self.ema_momentum * self.ema_usage
                    + (1.0 - self.ema_momentum) * batch_usage
                )
            aux_losses = self.aux_loss(weights)
        return out, aux_losses, weights


__all__ = [
    "ContextExpert",
    "CrossScaleExpert",
    "LocalDetailExpert",
    "MidScaleExpert",
    "NeckMoELoss",
    "NeckScaleMoE",
    "ScaleAwareGridRouter",
]

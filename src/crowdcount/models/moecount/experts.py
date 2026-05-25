"""Heterogeneous experts for MoECountNet."""

from __future__ import annotations

import torch
from torch import nn

from crowdcount.models.moecount.gate import SparseTop2Gate
from crowdcount.models.moecount.losses import LoadBalanceLoss


class SharedExpert(nn.Module):
    """Shared expert always active for all spatial positions (DeepSeekMoE/ViMoE pattern)."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LocalDenseExpert(nn.Module):
    """Local detail expert with stacked 3x3 convolutions."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class DilatedSparseExpert(nn.Module):
    """Medium/large-scale sparse expert with wider receptive field."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LargeContextExpert(nn.Module):
    """Large-receptive-field expert for global scene context and occlusion disambiguation."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=6, dilation=6),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=12, dilation=12),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class HeterogeneousSparseMoE(nn.Module):
    """Three heterogeneous experts combined by a sparse Top-2 spatial gate."""

    def __init__(
        self,
        channels: int = 256,
        gate_hidden_channels: int = 128,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.98,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        lambda_importance: float = 0.01,
        lambda_load: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = 3
        self.stem = SharedExpert(channels)
        self.experts = nn.ModuleList(
            [
                LocalDenseExpert(channels),
                DilatedSparseExpert(channels),
                LargeContextExpert(channels),
            ]
        )
        self.gate = SparseTop2Gate(
            in_channels=channels,
            hidden_channels=gate_hidden_channels,
            num_experts=self.num_experts,
            top_k=top_k,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            warmup_fraction=warmup_fraction,
            warmup_epochs=warmup_epochs,
        )
        self.balance_loss = LoadBalanceLoss(
            lambda_importance=lambda_importance,
            lambda_load=lambda_load,
        )
        self.output_norm = nn.GroupNorm(32, channels)

    @property
    def temperature(self) -> float:
        return float(self.gate.temperature)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.gate.set_epoch(epoch, total_epochs=total_epochs)

    def update_temperature(self, decay_rate: float | None = None) -> None:
        self.gate.update_temperature(decay_rate=decay_rate)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        shared_out = self.stem(features)
        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )
        route = self.gate(features)
        if self.training:
            load_fraction = route["load_fraction"]
            if isinstance(load_fraction, torch.Tensor):
                self.gate.update_expert_bias(load_fraction)
        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        routed = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        fused = self.output_norm(shared_out + routed)
        soft_probs = route["soft_probs"]
        hard_mask = route["hard_mask"]
        if not isinstance(soft_probs, torch.Tensor) or not isinstance(hard_mask, torch.Tensor):
            raise TypeError("gate probabilities and hard mask must be tensors")
        aux_losses = self.balance_loss(soft_probs, hard_mask) if self.training else {}
        return fused, aux_losses, route

"""Heterogeneous experts for MoECountNet."""

from __future__ import annotations

import torch
from torch import nn

from crowdcount.models.moecount.gate import SparseTop2Gate
from crowdcount.models.moecount.losses import LoadBalanceLoss


class SharedExpertStem(nn.Module):
    """Two shared 3x3 convolution layers before expert-specific branches."""

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
    """Small-scale dense expert preserving local detail."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
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


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(features))
        max_out = self.shared_mlp(self.max_pool(features))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        avg_out = features.mean(dim=1, keepdim=True)
        max_out, _ = features.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attended = self.channel_attention(features) * features
        return self.spatial_attention(attended) * attended


class OcclusionAwareExpert(nn.Module):
    """Occlusion-aware expert based on channel and spatial attention."""

    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.cbam = CBAM(channels, reduction=reduction)
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.cbam(features))


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
        cbam_reduction: int = 16,
        lambda_importance: float = 0.01,
        lambda_load: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = 3
        self.stem = SharedExpertStem(channels)
        self.experts = nn.ModuleList(
            [
                LocalDenseExpert(channels),
                DilatedSparseExpert(channels),
                OcclusionAwareExpert(channels, reduction=cbam_reduction),
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
        shared_features = self.stem(features)
        expert_outputs = torch.stack(
            [expert(shared_features) for expert in self.experts],
            dim=1,
        )
        route = self.gate(features)
        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        fused = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        fused = self.output_norm(fused)
        soft_probs = route["soft_probs"]
        hard_mask = route["hard_mask"]
        if not isinstance(soft_probs, torch.Tensor) or not isinstance(hard_mask, torch.Tensor):
            raise TypeError("gate probabilities and hard mask must be tensors")
        aux_losses = self.balance_loss(soft_probs, hard_mask) if self.training else {}
        return fused, aux_losses, route

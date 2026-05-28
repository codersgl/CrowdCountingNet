"""Sparse Top-2 gate for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SparseTop2Gate(nn.Module):
    """Full-resolution dilated router with warmup soft routing and ST Top-2."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_experts: int = 3,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.98,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if temperature_init <= 0 or temperature_min <= 0:
            raise ValueError("temperatures must be positive")
        if temperature_decay <= 0:
            raise ValueError("temperature_decay must be positive")
        if warmup_fraction < 0 or warmup_fraction > 1:
            raise ValueError("warmup_fraction must be in [0, 1]")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.temperature_decay = float(temperature_decay)
        self.warmup_fraction = float(warmup_fraction)
        self.warmup_epochs = warmup_epochs if warmup_epochs is None else int(warmup_epochs)
        self.eps = float(eps)
        self.current_epoch = 0
        self.total_epochs: int | None = None
        self.temperature = float(temperature_init)

        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1),
        )

    def _warmup_end(self) -> int:
        if self.warmup_epochs is not None:
            return int(self.warmup_epochs)
        if self.total_epochs is not None:
            return int(math.ceil(self.total_epochs * self.warmup_fraction))
        return 0

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)

        warmup_end = self._warmup_end()
        if self.current_epoch < warmup_end:
            self.temperature = self.temperature_init
        else:
            decay_epochs = self.current_epoch - warmup_end
            self.temperature = max(
                self.temperature_init * (self.temperature_decay ** decay_epochs),
                self.temperature_min,
            )

    def update_temperature(self, decay_rate: float | None = None) -> None:
        rate = self.temperature_decay if decay_rate is None else float(decay_rate)
        self.temperature = max(self.temperature * rate, self.temperature_min)

    def _in_warmup(self) -> bool:
        if not self.training:
            return False
        if self.warmup_epochs is not None:
            return self.current_epoch < self.warmup_epochs
        if self.total_epochs is None:
            return False
        return self.current_epoch < int(math.ceil(self.total_epochs * self.warmup_fraction))

    def update_expert_bias(self, load_fraction: torch.Tensor, bias_lr: float = 0.01) -> None:
        """DeepSeek-V2 style bias adjustment: boost underloaded, penalize overloaded experts."""
        target_load = self.top_k / self.num_experts
        error = target_load - load_fraction.detach()
        if not torch.isfinite(error).all():
            return
        self.expert_bias = self.expert_bias + bias_lr * error.to(device=self.expert_bias.device)

    def _sample_gumbel(self, logits: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(logits).clamp_(self.eps, 1.0 - self.eps)
        return (-torch.log(-torch.log(uniform))).clamp(-10, 10)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | bool]:
        if not torch.isfinite(self.expert_bias).all():
            self.expert_bias.zero_()
        logits = self.router(features) * self.logit_scale.clamp(0.1, 10.0) + self.expert_bias.view(1, -1, 1, 1)
        warmup_active = self._in_warmup()
        temperature = max(float(self.temperature), self.temperature_min)

        if self.training and not warmup_active:
            routed_logits = (logits + self._sample_gumbel(logits)) / temperature
        else:
            routed_logits = logits / temperature
        soft_probs = F.softmax(routed_logits, dim=1)

        top_values, top_indices = soft_probs.topk(self.top_k, dim=1)
        hard_mask = torch.zeros_like(soft_probs).scatter_(1, top_indices, 1.0)
        top1 = top_indices[:, 0]

        if warmup_active or self.top_k == self.num_experts:
            route_weights = soft_probs
        else:
            masked_probs = soft_probs * hard_mask
            hard_weights = masked_probs / masked_probs.sum(dim=1, keepdim=True).clamp_min(self.eps)
            route_weights = soft_probs + (hard_weights - soft_probs).detach()

        load_counts = hard_mask.detach().sum(dim=(0, 2, 3))
        load_distribution = load_counts / load_counts.sum().clamp_min(self.eps)
        entropy = -(load_distribution * torch.log(load_distribution.clamp_min(self.eps))).sum()
        importance = soft_probs.detach().mean(dim=(0, 2, 3))
        load_fraction = load_counts / float(soft_probs.shape[0] * soft_probs.shape[2] * soft_probs.shape[3])

        return {
            "logits": logits,
            "soft_probs": soft_probs,
            "weights": route_weights,
            "hard_mask": hard_mask,
            "top1": top1,
            "top_indices": top_indices,
            "top_values": top_values,
            "load_fraction": load_fraction,
            "importance": importance,
            "entropy": entropy.detach(),
            "temperature": logits.new_tensor(temperature),
            "warmup_active": warmup_active,
        }


class PixelSoftGate(nn.Module):
    """HMoDE-style per-pixel soft gating for expert feature combination.

    Replaces hard Top-K routing with per-pixel softmax weights.
    All experts always contribute, weighted by learned spatial gating maps.

    Reference: Du et al., "Redesigning Multi-Scale Neural Network for
    Crowd Counting", IEEE TIP 2023 (Section 3.2, HMoDE).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_experts: int = 3,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        self.num_experts = int(num_experts)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1),
        )

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        del epoch, total_epochs

    def update_temperature(self, decay_rate: float | None = None) -> None:
        del decay_rate

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | bool]:
        raw_weights = self.gate_net(features)  # [B, K, H, W]
        weights = F.softmax(raw_weights, dim=1)  # per-pixel softmax
        importance = weights.detach().mean(dim=(0, 2, 3))  # [K]
        return {
            "weights": weights,
            "soft_probs": weights,
            "hard_mask": weights,
            "top_indices": weights.argmax(dim=1, keepdim=True),
            "top1": weights.argmax(dim=1).squeeze(1),
            "importance": importance,
            "load_fraction": importance / importance.sum().clamp_min(1e-8),
            "entropy": weights.new_zeros(()),
            "temperature": weights.new_tensor(1.0),
            "warmup_active": False,
            "logits": raw_weights,
            "top_values": weights.max(dim=1).values,
        }


class MultiScaleSparseTop2Gate(SparseTop2Gate):
    """Multi-scale variant: router receives stride-8, 16, and 32 pooled features.

    Inherits temperature annealing, Gumbel-ST, and expert_bias from SparseTop2Gate.
    Only the router input is changed to multi-scale concatenation + compression.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_experts: int = 3,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.98,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_experts=num_experts,
            top_k=top_k,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            warmup_fraction=warmup_fraction,
            warmup_epochs=warmup_epochs,
            eps=eps,
        )
        self.scale_compress = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _build_multi_scale(self, features: torch.Tensor) -> torch.Tensor:
        h, w = features.shape[-2:]
        s16 = F.adaptive_avg_pool2d(features, (max(1, h // 2), max(1, w // 2)))
        s32 = F.adaptive_avg_pool2d(features, (max(1, h // 4), max(1, w // 4)))
        s16_up = F.interpolate(s16, size=(h, w), mode="bilinear", align_corners=False)
        s32_up = F.interpolate(s32, size=(h, w), mode="bilinear", align_corners=False)
        multi_scale = torch.cat([features, s16_up, s32_up], dim=1)
        return self.scale_compress(multi_scale)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | bool]:
        compressed = self._build_multi_scale(features)
        if not torch.isfinite(self.expert_bias).all():
            self.expert_bias.zero_()
        logits = self.router(compressed) * self.logit_scale.clamp(0.1, 10.0) + self.expert_bias.view(1, -1, 1, 1)
        warmup_active = self._in_warmup()
        temperature = max(float(self.temperature), self.temperature_min)

        if self.training and not warmup_active:
            routed_logits = (logits + self._sample_gumbel(logits)) / temperature
        else:
            routed_logits = logits / temperature
        soft_probs = F.softmax(routed_logits, dim=1)

        top_values, top_indices = soft_probs.topk(self.top_k, dim=1)
        hard_mask = torch.zeros_like(soft_probs).scatter_(1, top_indices, 1.0)
        top1 = top_indices[:, 0]

        if warmup_active or self.top_k == self.num_experts:
            route_weights = soft_probs
        else:
            masked_probs = soft_probs * hard_mask
            hard_weights = masked_probs / masked_probs.sum(dim=1, keepdim=True).clamp_min(self.eps)
            route_weights = soft_probs + (hard_weights - soft_probs).detach()

        load_counts = hard_mask.detach().sum(dim=(0, 2, 3))
        load_distribution = load_counts / load_counts.sum().clamp_min(self.eps)
        entropy = -(load_distribution * torch.log(load_distribution.clamp_min(self.eps))).sum()
        importance = soft_probs.detach().mean(dim=(0, 2, 3))
        load_fraction = load_counts / float(soft_probs.shape[0] * soft_probs.shape[2] * soft_probs.shape[3])

        return {
            "logits": logits,
            "soft_probs": soft_probs,
            "weights": route_weights,
            "hard_mask": hard_mask,
            "top1": top1,
            "top_indices": top_indices,
            "top_values": top_values,
            "load_fraction": load_fraction,
            "importance": importance,
            "entropy": entropy.detach(),
            "temperature": logits.new_tensor(temperature),
            "warmup_active": warmup_active,
        }

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
        self.expert_bias = self.expert_bias + bias_lr * error.to(device=self.expert_bias.device)

    def _sample_gumbel(self, logits: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(logits).clamp_(self.eps, 1.0 - self.eps)
        return -torch.log(-torch.log(uniform))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | bool]:
        logits = self.router(features) + self.expert_bias.view(1, -1, 1, 1)
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
            route_weights = hard_weights + soft_probs - soft_probs.detach()

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

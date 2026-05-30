"""Sparse Top-2 gate for MoECountNet."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.gcn import DensityGraphBuilder
from torch_geometric.nn import GCNConv


class SparseTop2Gate(nn.Module):
    """Full-resolution dilated router with warmup soft routing and ST Top-2.

    Optionally accepts a density hint (e.g. a density map) that is projected and
    concatenated with features before the router, giving the gate a direct
    signal about where people are located.

    When ``use_density_bias=True``, an additional density→per-expert bias
    (zero-init, small Conv3×3) is added directly to the routing logits, providing
    the gate with an explicit density→routing shortcut that does not compete
    with the feature-driven router path.
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
        use_density_hint: bool = False,
        density_hidden: int = 8,
        use_density_bias: bool = False,
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
        self.use_density_hint = bool(use_density_hint)
        self.density_hidden = int(density_hidden)
        self.use_density_bias = bool(use_density_bias)

        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        router_in = in_channels + (density_hidden if self.use_density_hint else 0)
        if self.use_density_hint:
            self.density_proj = nn.Sequential(
                nn.Conv2d(1, density_hidden, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        if self.use_density_bias:
            self.density_bias_proj = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, num_experts, kernel_size=1),
            )
            self.density_bias_gain = nn.Parameter(torch.zeros(1))
        self.router = nn.Sequential(
            nn.Conv2d(router_in, hidden_channels, kernel_size=3, padding=1),
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

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | bool]:
        if self.use_density_hint and density is not None:
            if density.shape[-2:] != features.shape[-2:]:
                density = F.interpolate(
                    density, size=features.shape[-2:], mode="bilinear", align_corners=False
                )
            density_feat = self.density_proj(density)
            features = torch.cat([features, density_feat], dim=1)
        elif self.use_density_hint:
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        features.shape[0],
                        self.density_hidden,
                        features.shape[2],
                        features.shape[3],
                        device=features.device,
                        dtype=features.dtype,
                    ),
                ],
                dim=1,
            )
        if not torch.isfinite(self.expert_bias).all():
            self.expert_bias.zero_()
        logits = self.router(features) * self.logit_scale.clamp(0.1, 10.0) + self.expert_bias.view(1, -1, 1, 1)

        # Density→expert direct bias: gives the gate an explicit density→routing
        # shortcut without competing with the feature-driven router.
        if self.use_density_bias and density is not None:
            _db = density
            if _db.shape[-2:] != features.shape[-2:]:
                _db = F.interpolate(_db, size=features.shape[-2:], mode="bilinear", align_corners=False)
            density_bias = self.density_bias_proj(_db)  # [B, K, H, W]
            logits = logits + density_bias * self.density_bias_gain.tanh()

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


class GraphAwareSparseTop2Gate(SparseTop2Gate):
    """SparseTop2Gate with GNN-based neighbourhood context for routing.

    Before the router computes per-pixel logits, a single GCNConv layer
    aggregates features from k-NN neighbours (graph built via density
    similarity), giving each pixel visibility into what similar regions
    are doing.  A zero-init residual gate keeps the module identical to a
    standard SparseTop2Gate at training start, so graph context is
    gradually adopted without destabilising early optimisation.

    Parameters
    ----------
    graph_k : int
        Number of nearest neighbours for the density-based k-NN graph.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_experts: int = 3,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.99998,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        eps: float = 1e-8,
        use_density_hint: bool = False,
        density_hidden: int = 8,
        use_density_bias: bool = False,
        graph_k: int = 4,
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
            use_density_hint=use_density_hint,
            density_hidden=density_hidden,
            use_density_bias=use_density_bias,
        )
        self.graph_k = int(graph_k)
        self.graph_builder = DensityGraphBuilder(k=self.graph_k)
        self.gcn_conv = GCNConv(in_channels, in_channels)
        self.graph_gate = nn.Parameter(torch.zeros(1))

    def _apply_graph_context(
        self, features: torch.Tensor, density: torch.Tensor | None
    ) -> torch.Tensor:
        """Build density k-NN graph, run 1-layer GCN, return zero-init residual."""
        B, C, H, W = features.shape
        graph_input = (
            density if density is not None
            else features.mean(dim=1, keepdim=True)
        )
        if graph_input.shape[-2:] != (H, W):
            graph_input = F.interpolate(
                graph_input, size=(H, W), mode="bilinear", align_corners=False
            )
        edge_index, _, N_total, _, _ = self.graph_builder.build_batch_graph(
            graph_input
        )
        nodes = features.permute(0, 2, 3, 1).reshape(N_total, C)
        graph_feat = self.gcn_conv(nodes, edge_index)
        graph_feat = F.relu(graph_feat)
        graph_feat = graph_feat.view(B, H, W, C).permute(0, 3, 1, 2)
        return features + self.graph_gate.tanh() * graph_feat

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | bool]:
        # GCN neighbourhood aggregation before routing (zero-init → identity at start)
        enhanced = self._apply_graph_context(features, density)
        return super().forward(enhanced, density=density)


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

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | bool]:
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
        use_density_hint: bool = False,
        density_hidden: int = 8,
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
            use_density_hint=use_density_hint,
            density_hidden=density_hidden,
        )
        router_in = in_channels + (density_hidden if use_density_hint else 0)
        self.scale_compress = nn.Conv2d(3 * router_in, router_in, kernel_size=1)
        self.router = nn.Sequential(
            nn.Conv2d(router_in, hidden_channels, kernel_size=3, padding=1),
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

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | bool]:
        if self.use_density_hint and density is not None:
            if density.shape[-2:] != features.shape[-2:]:
                density = F.interpolate(
                    density, size=features.shape[-2:], mode="bilinear", align_corners=False
                )
            density_feat = self.density_proj(density)
            features = torch.cat([features, density_feat], dim=1)
        elif self.use_density_hint:
            features = torch.cat(
                [
                    features,
                    torch.zeros(
                        features.shape[0],
                        self.density_hidden,
                        features.shape[2],
                        features.shape[3],
                        device=features.device,
                        dtype=features.dtype,
                    ),
                ],
                dim=1,
            )
        compressed = self._build_multi_scale(features)
        if not torch.isfinite(self.expert_bias).all():
            self.expert_bias.zero_()
        logits = self.router(compressed) * self.logit_scale.clamp(0.1, 10.0) + self.expert_bias.view(1, -1, 1, 1)

        # Density→expert direct bias (inherited from SparseTop2Gate)
        if self.use_density_bias and density is not None:
            _db = density
            if _db.shape[-2:] != features.shape[-2:]:
                _db = F.interpolate(_db, size=features.shape[-2:], mode="bilinear", align_corners=False)
            density_bias = self.density_bias_proj(_db)
            logits = logits + density_bias * self.density_bias_gain.tanh()

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

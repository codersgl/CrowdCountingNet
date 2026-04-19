from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.ssim_loss import SSIMLoss


def _conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    dilation: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    if padding is None:
        padding = ((kernel_size - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class BackgroundAwareGate(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_act(in_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        soft_mask = self.net(x)
        hard_mask = (soft_mask > threshold).to(dtype=soft_mask.dtype)
        return hard_mask - soft_mask.detach() + soft_mask


class ScaleDecoupledRouter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temperature_init: float = 1.0,
        temperature_min: float = 0.3,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.regression_head = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.routing_head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )

    def update_temperature(self, decay_rate: float = 0.9999) -> None:
        self.temperature = max(self.temperature * decay_rate, self.temperature_min)

    def forward(
        self,
        x: torch.Tensor,
        fg_mask: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sd_map = self.regression_head(x)
        logits = self.routing_head(sd_map)
        if training:
            route_map = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=False,
                dim=1,
            )
        else:
            hard_idx = logits.argmax(dim=1)
            route_map = F.one_hot(hard_idx, num_classes=4).permute(0, 3, 1, 2)
            route_map = route_map.to(dtype=x.dtype)
        route_map = route_map * fg_mask
        scale_map = sd_map[:, 0:1]
        density_map = sd_map[:, 1:2]
        return route_map, scale_map, density_map


class LargeScaleExpert(nn.Module):
    def __init__(self, in_channels: int, rates: tuple[int, ...]) -> None:
        super().__init__()
        branch_channels = max(in_channels // 4, 32)
        self.branches = nn.ModuleList(
            [
                _conv_bn_act(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    dilation=rate,
                )
                for rate in rates
            ]
        )
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        fused_channels = branch_channels * (len(rates) + 1)
        self.project = _conv_bn_act(
            fused_channels, in_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        pooled = self.pool_proj(x)
        pooled = F.interpolate(
            pooled, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class TinyScaleExpert(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.prelu = nn.PReLU(in_channels)
        self.depthwise = _conv_bn_act(
            in_channels,
            in_channels,
            kernel_size=3,
            groups=in_channels,
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.prelu(self.expand(x))
        x = self.depthwise(x)
        x = self.project(x)
        return F.relu(x + residual, inplace=True)


class MidScaleExpert(nn.Module):
    def __init__(self, in_channels: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.block = nn.Sequential(
            _conv_bn_act(in_channels, hidden, kernel_size=1, padding=0),
            _conv_bn_act(hidden, hidden, kernel_size=3, groups=hidden),
            SqueezeExcite(hidden, reduction=16),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)


class OcclusionReasoningExpert(nn.Module):
    def __init__(
        self, in_channels: int, attn_dim: int | None = None, num_heads: int = 2
    ) -> None:
        super().__init__()
        attn_dim = attn_dim or in_channels
        self.proj_in = (
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False)
            if attn_dim != in_channels
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        projected = self.proj_in(x)
        attn_dim = projected.shape[1]
        tokens = projected.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, attn_dim, height, width)
        return F.relu(self.proj(attn_out) + x, inplace=True)


class AdaptiveFeatureAggregator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.sd_modulation = nn.Sequential(
            nn.Conv2d(2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.smooth = _conv_bn_act(in_channels, in_channels, kernel_size=3)

    def forward(
        self,
        expert_outputs: list[torch.Tensor],
        route_map: torch.Tensor,
        scale_map: torch.Tensor,
        density_map: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.zeros_like(expert_outputs[0])
        for idx, expert_out in enumerate(expert_outputs):
            fused = fused + route_map[:, idx : idx + 1] * expert_out
        modulation = self.sd_modulation(torch.cat([scale_map, density_map], dim=1))
        fused = fused * modulation
        return self.smooth(fused)


class AuxDensityHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(x))


@dataclass
class SDDMoEConfig:
    fg_threshold: float = 0.3
    gumbel_temperature: float = 1.0
    gumbel_temp_min: float = 0.3
    aspp_rates: tuple[int, ...] = (1, 6, 12, 18)
    self_attn_heads: int = 2
    self_attn_dim: int | None = None
    aux_loss_weight: float = 0.2
    lambda_balance: float = 0.01
    lambda_scale: float = 0.1
    lambda_ssim: float = 0.1
    ssim_window_size: int = 7


class SDDMoE(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        cfg: object | None = None,
    ) -> None:
        super().__init__()
        self.cfg = SDDMoEConfig(
            fg_threshold=float(getattr(cfg, "fg_threshold", 0.3))
            if cfg is not None
            else 0.3,
            gumbel_temperature=float(getattr(cfg, "gumbel_temperature", 1.0))
            if cfg is not None
            else 1.0,
            gumbel_temp_min=float(getattr(cfg, "gumbel_temp_min", 0.3))
            if cfg is not None
            else 0.3,
            aspp_rates=tuple(int(v) for v in getattr(cfg, "aspp_rates", [1, 6, 12, 18]))
            if cfg is not None
            else (1, 6, 12, 18),
            self_attn_heads=int(getattr(cfg, "self_attn_heads", 2))
            if cfg is not None
            else 2,
            self_attn_dim=(int(getattr(cfg, "self_attn_dim", 0)) or None)
            if cfg is not None
            else None,
            aux_loss_weight=float(getattr(cfg, "aux_loss_weight", 0.2))
            if cfg is not None
            else 0.2,
            lambda_balance=float(getattr(cfg, "lambda_balance", 0.01))
            if cfg is not None
            else 0.01,
            lambda_scale=float(getattr(cfg, "lambda_scale", 0.1))
            if cfg is not None
            else 0.1,
            lambda_ssim=float(getattr(cfg, "lambda_ssim", 0.1))
            if cfg is not None
            else 0.1,
            ssim_window_size=int(getattr(cfg, "ssim_window_size", 7))
            if cfg is not None
            else 7,
        )
        self.gate = BackgroundAwareGate(in_channels)
        self.router = ScaleDecoupledRouter(
            in_channels,
            temperature_init=self.cfg.gumbel_temperature,
            temperature_min=self.cfg.gumbel_temp_min,
        )
        self.large_expert = LargeScaleExpert(in_channels, self.cfg.aspp_rates)
        self.tiny_expert = TinyScaleExpert(in_channels)
        self.mid_expert = MidScaleExpert(in_channels)
        self.occlusion_expert = OcclusionReasoningExpert(
            in_channels,
            attn_dim=self.cfg.self_attn_dim,
            num_heads=self.cfg.self_attn_heads,
        )
        self.aggregator = AdaptiveFeatureAggregator(in_channels)
        self.aux_density_head = AuxDensityHead(in_channels)
        self.ssim = SSIMLoss(window_size=self.cfg.ssim_window_size)

    def update_temperature(self, decay_rate: float = 0.9999) -> None:
        self.router.update_temperature(decay_rate=decay_rate)

    def _density_aware_balance_loss(
        self,
        route_map: torch.Tensor,
        density_hint: torch.Tensor | None,
    ) -> torch.Tensor:
        route_mass = route_map.sum(dim=(2, 3))
        total_mass = route_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fractions = route_mass / total_mass
        target = torch.full_like(fractions, 0.25)
        weights = torch.ones_like(fractions)
        if density_hint is not None:
            global_density = density_hint.mean(dim=(1, 2, 3), keepdim=False)
            dense_factor = torch.sigmoid(global_density)
            sparse_factor = 1.0 - dense_factor
            weights[:, 0] = 1.0 - 0.5 * sparse_factor
            weights[:, 1] = 1.0 - 0.5 * dense_factor
        return (weights * (fractions - target).abs()).mean()

    def _scale_prediction_loss(
        self,
        scale_map: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None,
        image_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if not targets or image_size is None:
            return scale_map.new_zeros(())
        _, _, feat_h, feat_w = scale_map.shape
        img_h, img_w = image_size
        total_loss = scale_map.new_zeros(())
        valid_count = 0
        for batch_idx, target in enumerate(targets):
            points = target.get("point")
            if points is None or points.numel() == 0:
                continue
            points = points.to(device=scale_map.device, dtype=scale_map.dtype)
            if points.shape[0] > 1:
                distances = torch.cdist(points, points)
                diag = torch.eye(
                    points.shape[0], device=points.device, dtype=torch.bool
                )
                distances.masked_fill_(diag, float("inf"))
                nearest = distances.min(dim=1).values
                nearest = torch.where(
                    torch.isfinite(nearest), nearest, torch.zeros_like(nearest)
                )
            else:
                nearest = torch.zeros(
                    points.shape[0], device=points.device, dtype=points.dtype
                )
            x_idx = torch.clamp(
                (points[:, 0] * feat_w / max(img_w, 1)).round().long(), 0, feat_w - 1
            )
            y_idx = torch.clamp(
                (points[:, 1] * feat_h / max(img_h, 1)).round().long(), 0, feat_h - 1
            )
            spatial_scale = 0.5 * (feat_w / max(img_w, 1) + feat_h / max(img_h, 1))
            # Log-compress to stabilise magnitude: crowd distances are log-normal
            target_scale = torch.log1p(nearest * spatial_scale)
            pred_scale = scale_map[batch_idx, 0, y_idx, x_idx]
            total_loss = total_loss + F.smooth_l1_loss(
                pred_scale, target_scale, reduction="mean"
            )
            valid_count += 1
        if valid_count == 0:
            return scale_map.new_zeros(())
        return total_loss / valid_count

    def _ssim_loss(
        self,
        fused_features: torch.Tensor,
        gt_density: torch.Tensor | None,
    ) -> torch.Tensor:
        if gt_density is None:
            return fused_features.new_zeros(())
        pred_density = self.aux_density_head(fused_features)
        if gt_density.dim() == 3:
            gt_density = gt_density.unsqueeze(1)
        gt_density = gt_density.to(
            device=fused_features.device, dtype=fused_features.dtype
        )
        if gt_density.shape[-2:] != pred_density.shape[-2:]:
            gt_density = F.interpolate(
                gt_density,
                size=pred_density.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.ssim(pred_density, gt_density)

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        gt_density: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if training is None:
            training = self.training
        fg_mask = self.gate(x, threshold=self.cfg.fg_threshold)
        route_map, scale_map, density_map = self.router(x, fg_mask, training=training)

        expert_outputs = [
            self.large_expert(x),
            self.tiny_expert(x),
            self.mid_expert(x),
            self.occlusion_expert(x),
        ]
        fused = self.aggregator(expert_outputs, route_map, scale_map, density_map)

        if not training:
            return fused, {}, route_map

        l_balance = self._density_aware_balance_loss(route_map, density_hint)
        l_scale = self._scale_prediction_loss(scale_map, targets, image_size)
        l_ssim = self._ssim_loss(fused, gt_density)
        total_aux = (
            self.cfg.lambda_balance * l_balance
            + self.cfg.lambda_scale * l_scale
            + self.cfg.lambda_ssim * l_ssim
        )
        aux_losses = {
            "l_balance": l_balance,
            "l_scale": l_scale,
            "l_ssim": l_ssim,
            "total_aux": total_aux,
        }
        return fused, aux_losses, route_map


__all__ = [
    "AdaptiveFeatureAggregator",
    "BackgroundAwareGate",
    "LargeScaleExpert",
    "MidScaleExpert",
    "OcclusionReasoningExpert",
    "SDDMoE",
    "ScaleDecoupledRouter",
    "TinyScaleExpert",
]

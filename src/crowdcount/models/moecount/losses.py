"""Losses for MoECountNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _project_onto_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {p : p_i >= 0, sum(p) = 1}. Duchi et al. 2008."""
    if v.numel() == 0:
        return v
    if v.numel() == 1:
        return torch.ones_like(v)
    u, _ = v.sort(descending=True)
    cumsum = u.cumsum(dim=0)
    arange = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
    condition = u > (cumsum - 1) / arange
    rho = int(condition.nonzero(as_tuple=True)[0][-1].item())
    nu = (cumsum[rho] - 1) / (rho + 1)
    return (v - nu).clamp(min=0)


class ProximalMappingLoss(nn.Module):
    """Point-supervised Proximal Mapping Loss (Lin et al., ICLR 2025).

    Eliminates the intersection hypothesis by dividing pixels into non-overlapping
    Voronoi regions (nearest-neighbor assignment) and applying simplex projection
    per GT point.
    """

    requires_points: bool = True

    def __init__(
        self,
        sigma: float = 8.0,
        use_background: bool = False,
        bg_threshold: float = 3.0,
        max_pixels_per_chunk: int = 16384,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self.use_background = bool(use_background)
        self.bg_threshold = float(bg_threshold)
        self.max_pixels_per_chunk = int(max_pixels_per_chunk)

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        image_sizes: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if targets is None:
            raise ValueError("ProximalMappingLoss.forward requires targets")
        del gt_density

        device = pred_density.device
        dtype = pred_density.dtype
        batch_size, _, density_height, density_width = pred_density.shape
        if image_sizes is None:
            image_height, image_width = density_height, density_width
        else:
            image_height, image_width = int(image_sizes[0]), int(image_sizes[1])

        stride_y = image_height / max(density_height, 1)
        stride_x = image_width / max(density_width, 1)
        grid_y_values = (torch.arange(density_height, device=device, dtype=dtype) + 0.5) * stride_y
        grid_x_values = (torch.arange(density_width, device=device, dtype=dtype) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(grid_y_values, grid_x_values, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        bg_distance_sq = (self.bg_threshold * self.sigma) ** 2
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for batch_index in range(batch_size):
            density_flat = pred_density[batch_index, 0].reshape(-1)
            points = targets[batch_index].get("point") if isinstance(targets[batch_index], dict) else None
            if points is None or points.numel() == 0:
                total_loss = total_loss + density_flat.pow(2).sum()
                continue

            points = points.to(device=device, dtype=dtype)
            if points.dim() != 2 or points.shape[-1] != 2:
                raise ValueError(f"target['point'] must be [N, 2], got {tuple(points.shape)}")

            num_points = points.shape[0]
            # Pre-allocate accumulated loss per point (proximal) and per pixel (background)
            point_acc = torch.zeros(num_points, device=device, dtype=dtype)
            point_counts = torch.zeros(num_points, device=device, dtype=dtype)
            bg_acc = torch.zeros((), device=device, dtype=dtype)

            for start in range(0, coords.shape[0], self.max_pixels_per_chunk):
                end = min(start + self.max_pixels_per_chunk, coords.shape[0])
                coord_chunk = coords[start:end]  # [chunk, 2]
                density_chunk = density_flat[start:end]  # [chunk]
                dist_sq = (coord_chunk.unsqueeze(1) - points.unsqueeze(0)).pow(2).sum(dim=-1)  # [chunk, N]
                min_dist_sq, assignment = dist_sq.min(dim=-1)  # [chunk]

                if self.use_background:
                    bg_mask = min_dist_sq > bg_distance_sq
                    if bg_mask.any():
                        bg_acc = bg_acc + density_chunk[bg_mask].pow(2).sum()
                    fg_mask = ~bg_mask
                    if not fg_mask.any():
                        continue
                    fg_density = density_chunk[fg_mask]
                    fg_assignment = assignment[fg_mask]
                else:
                    fg_density = density_chunk
                    fg_assignment = assignment

                for p_idx in range(num_points):
                    point_mask = fg_assignment == p_idx
                    if not point_mask.any():
                        continue
                    p_density = fg_density[point_mask]
                    target = _project_onto_simplex(p_density)
                    point_acc[p_idx] = point_acc[p_idx] + (p_density - target).pow(2).sum()
                    point_counts[p_idx] = point_counts[p_idx] + point_mask.sum().to(dtype)

            point_loss = point_acc.sum()
            # Penalize points with no assigned pixels via log-count residual
            orphan_mask = point_counts == 0
            if orphan_mask.any():
                orphan_count = pred_density.new_tensor(orphan_mask.sum().to(dtype))
                point_loss = point_loss + orphan_count * 0.1

            if self.use_background:
                total_loss = total_loss + point_loss + bg_acc
            else:
                total_loss = total_loss + point_loss

        return total_loss


class BayesianLoss(nn.Module):
    """Point-supervised Bayesian Loss with optional chunked pixel processing."""

    requires_points: bool = True

    def __init__(
        self,
        sigma: float = 8.0,
        use_background: bool = True,
        bg_ratio: float = 0.15,
        count_loss_type: str = "l1",
        max_pixels_per_chunk: int = 16384,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if bg_ratio <= 0:
            raise ValueError("bg_ratio must be > 0")
        if count_loss_type not in {"l1", "mse"}:
            raise ValueError("count_loss_type must be 'l1' or 'mse'")
        if max_pixels_per_chunk <= 0:
            raise ValueError("max_pixels_per_chunk must be > 0")
        self.sigma = float(sigma)
        self.use_background = bool(use_background)
        self.bg_ratio = float(bg_ratio)
        self.count_loss_type = str(count_loss_type)
        self.max_pixels_per_chunk = int(max_pixels_per_chunk)

    def _residual_loss(self, expected_count: torch.Tensor, target: float) -> torch.Tensor:
        diff = expected_count - target
        if self.count_loss_type == "l1":
            return diff.abs().sum()
        return diff.pow(2).sum()

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        image_sizes: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if targets is None:
            raise ValueError("BayesianLoss.forward requires targets")
        del gt_density

        device = pred_density.device
        dtype = pred_density.dtype
        batch_size, _, density_height, density_width = pred_density.shape
        if image_sizes is None:
            image_height, image_width = density_height, density_width
        else:
            image_height, image_width = int(image_sizes[0]), int(image_sizes[1])

        stride_y = image_height / max(density_height, 1)
        stride_x = image_width / max(density_width, 1)
        grid_y_values = (torch.arange(density_height, device=device, dtype=dtype) + 0.5) * stride_y
        grid_x_values = (torch.arange(density_width, device=device, dtype=dtype) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(grid_y_values, grid_x_values, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        background_distance_sq = (self.bg_ratio * float(max(image_height, image_width))) ** 2
        two_sigma_sq = 2.0 * self.sigma * self.sigma
        total_loss = torch.zeros((), device=device, dtype=dtype)

        for batch_index in range(batch_size):
            density_flat = pred_density[batch_index, 0].reshape(-1)
            points = targets[batch_index].get("point") if isinstance(targets[batch_index], dict) else None
            if points is None or points.numel() == 0:
                total_loss = total_loss + self._residual_loss(density_flat.sum().unsqueeze(0), 0.0)
                continue

            points = points.to(device=device, dtype=dtype)
            if points.dim() != 2 or points.shape[-1] != 2:
                raise ValueError(f"target['point'] must be [N, 2], got {tuple(points.shape)}")

            expected_size = points.shape[0] + (1 if self.use_background else 0)
            expected = torch.zeros(expected_size, device=device, dtype=dtype)
            for start in range(0, coords.shape[0], self.max_pixels_per_chunk):
                end = min(start + self.max_pixels_per_chunk, coords.shape[0])
                coord_chunk = coords[start:end]
                density_chunk = density_flat[start:end]
                dist_sq = (coord_chunk.unsqueeze(1) - points.unsqueeze(0)).pow(2).sum(dim=-1)
                if self.use_background:
                    min_dist_sq = dist_sq.min(dim=1, keepdim=True).values.clamp(min=0.0)
                    bg_dist_sq = background_distance_sq / (min_dist_sq + 1e-5)
                    dist_sq = torch.cat([dist_sq, bg_dist_sq], dim=1)
                log_likelihood = -dist_sq / two_sigma_sq
                log_likelihood = log_likelihood - log_likelihood.max(dim=1, keepdim=True).values
                likelihood = log_likelihood.exp()
                posterior = likelihood / likelihood.sum(dim=1, keepdim=True).clamp_min(1e-12)
                expected = expected + (posterior * density_chunk.unsqueeze(1)).sum(dim=0)

            if self.use_background:
                total_loss = total_loss + self._residual_loss(expected[:-1], 1.0)
                total_loss = total_loss + self._residual_loss(expected[-1:], 0.0)
            else:
                total_loss = total_loss + self._residual_loss(expected, 1.0)
        return total_loss


class CountLoss(nn.Module):
    """Count supervision combining SmoothL1 and log-count loss."""

    def forward(
        self,
        pred_density: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_counts = pred_density.sum(dim=(1, 2, 3)).clamp_min(0.0)
        gt_counts = pred_counts.new_tensor([float(target["point"].shape[0]) for target in targets])
        l_smooth = F.smooth_l1_loss(pred_counts, gt_counts, beta=0.5)
        l_log = (torch.log1p(pred_counts) - torch.log1p(gt_counts)).abs().mean()
        return l_smooth + 0.1 * l_log


class LogCountLoss(nn.Module):
    """Log-space count loss for density map integrals (kept for backward compat)."""

    def forward(
        self,
        pred_density: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_counts = pred_density.sum(dim=(1, 2, 3)).clamp_min(0.0)
        gt_counts = pred_counts.new_tensor([float(target["point"].shape[0]) for target in targets])
        return (torch.log1p(pred_counts) - torch.log1p(gt_counts)).abs().mean()


class LogCountWeightSchedule:
    """Piecewise multiplicative schedule for the log-count loss weight."""

    def __init__(
        self,
        initial_weight: float = 0.1,
        decay_epochs: int = 50,
        decay_rate: float = 0.5,
        min_weight: float = 0.05,
    ) -> None:
        if decay_epochs <= 0:
            raise ValueError("decay_epochs must be > 0")
        self.initial_weight = float(initial_weight)
        self.decay_epochs = int(decay_epochs)
        self.decay_rate = float(decay_rate)
        self.min_weight = float(min_weight)

    def weight_at(self, epoch: int) -> float:
        steps = max(int(epoch), 0) // self.decay_epochs
        value = self.initial_weight * (self.decay_rate ** steps)
        return max(value, self.min_weight)


class LoadBalanceLoss(nn.Module):
    """CV-squared balance loss for soft importance and ST hard load."""

    def __init__(
        self,
        lambda_importance: float = 0.01,
        lambda_load: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_importance = float(lambda_importance)
        self.lambda_load = float(lambda_load)
        self.eps = float(eps)

    def _cv_squared(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean().clamp_min(self.eps)
        variance = (values - mean).pow(2).mean()
        return variance / mean.pow(2)

    def forward(
        self,
        soft_probs: torch.Tensor,
        hard_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        importance = soft_probs.sum(dim=(0, 2, 3))
        importance_cv = self._cv_squared(importance)
        if hard_mask is None:
            load_signal = soft_probs
        else:
            load_signal = hard_mask + soft_probs - soft_probs.detach()
        load = load_signal.sum(dim=(0, 2, 3))
        load_cv = self._cv_squared(load)
        total = self.lambda_importance * importance_cv + self.lambda_load * load_cv
        return {
            "l_importance": importance_cv,
            "l_load": load_cv,
            "l_balance": total,
            "total_aux": total,
        }


class MoECountLoss(nn.Module):
    """Aggregate density, count, and MoE balance losses.

    Supports both ProximalMappingLoss (default) and BayesianLoss as the density term.
    Balance loss decays to zero after warmup so expert_bias takes over.
    """

    def __init__(
        self,
        pml_loss: ProximalMappingLoss | None = None,
        bayesian_loss: BayesianLoss | None = None,
        count_loss: CountLoss | LogCountLoss | None = None,
        count_weight: float = 1.0,
        balance_loss: LoadBalanceLoss | None = None,
        warmup_end: int = 0,
        balance_decay_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.pml_loss = pml_loss
        self.bayesian_loss = bayesian_loss
        self.count_loss = count_loss or CountLoss()
        self.count_weight = float(count_weight)
        self.balance_loss = balance_loss or LoadBalanceLoss()
        self.warmup_end = int(warmup_end)
        self.balance_decay_epochs = int(balance_decay_epochs)

    def _balance_scale(self, epoch: int) -> float:
        if epoch <= self.warmup_end:
            return 1.0
        progress = (epoch - self.warmup_end) / max(self.balance_decay_epochs, 1)
        return max(0.0, 1.0 - progress)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        gt_density: torch.Tensor | None,
        image_sizes: tuple[int, int] | torch.Size,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        pred_density = outputs["density_out"]
        if not isinstance(pred_density, torch.Tensor):
            raise TypeError("outputs['density_out'] must be a tensor")
        batch_size = max(pred_density.shape[0], 1)

        # Density loss: PML (preferred) or Bayesian
        if self.pml_loss is not None:
            density_loss = self.pml_loss(
                pred_density,
                gt_density,
                targets=targets,
                image_sizes=image_sizes,
            ) / batch_size
            loss_label = "loss_pml"
        elif self.bayesian_loss is not None:
            density_loss = self.bayesian_loss(
                pred_density,
                gt_density,
                targets=targets,
                image_sizes=image_sizes,
            ) / batch_size
            loss_label = "loss_bayesian"
        else:
            raise RuntimeError("MoECountLoss requires either pml_loss or bayesian_loss")

        # Count loss with fixed weight
        count_raw = self.count_loss(pred_density, targets)
        count = count_raw * self.count_weight

        # Balance loss with warmup-decay schedule
        balance = outputs.get("moe_aux_total")
        if not isinstance(balance, torch.Tensor):
            soft_probs = outputs.get("moe_soft_probs")
            hard_mask = outputs.get("moe_hard_mask")
            if isinstance(soft_probs, torch.Tensor):
                hard_tensor = hard_mask if isinstance(hard_mask, torch.Tensor) else None
                balance = self.balance_loss(soft_probs, hard_tensor)["total_aux"]
            else:
                balance = pred_density.new_zeros(())
        balance_scale = self._balance_scale(epoch)
        scaled_balance = balance * balance_scale

        total = density_loss + count + scaled_balance
        return {
            "loss_total": total,
            loss_label: density_loss,
            "loss_count": count,
            "loss_balance": scaled_balance,
            "loss_balance_raw": balance,
            "balance_scale": pred_density.new_tensor(balance_scale),
            "lambda_count": pred_density.new_tensor(self.count_weight),
        }

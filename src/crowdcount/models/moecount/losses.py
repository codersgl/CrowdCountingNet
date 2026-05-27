"""Losses for MoECountNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.matcher import HungarianMatcher_Crowd
from crowdcount.models.criterion import softmax_focal_loss


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

    Supports sigma annealing: large sigma at start of training stabilises Voronoi
    assignments; small sigma later sharpens density peaks and improves localisation.
    """

    requires_points: bool = True

    def __init__(
        self,
        sigma: float = 8.0,
        use_background: bool = False,
        bg_threshold: float = 3.0,
        max_pixels_per_chunk: int = 16384,
        sigma_schedule: dict | None = None,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self.sigma_schedule: dict | None = None
        if sigma_schedule is not None:
            self.sigma_schedule = {
                "start_epoch": int(sigma_schedule.get("start_epoch", 0)),
                "end_epoch": int(sigma_schedule.get("end_epoch", 300)),
                "sigma_start": float(sigma_schedule.get("sigma_start", sigma)),
                "sigma_end": float(sigma_schedule.get("sigma_end", sigma * 0.75)),
            }
        self.use_background = bool(use_background)
        self.bg_threshold = float(bg_threshold)
        self.max_pixels_per_chunk = int(max_pixels_per_chunk)

    def _current_sigma(self, epoch: int) -> float:
        """Linear interpolation of sigma according to schedule."""
        if self.sigma_schedule is None:
            return self.sigma
        sched = self.sigma_schedule
        if epoch <= sched["start_epoch"]:
            return sched["sigma_start"]
        if epoch >= sched["end_epoch"]:
            return sched["sigma_end"]
        progress = (epoch - sched["start_epoch"]) / max(sched["end_epoch"] - sched["start_epoch"], 1)
        return sched["sigma_start"] + (sched["sigma_end"] - sched["sigma_start"]) * progress

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        image_sizes: tuple[int, int] | torch.Size | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        if targets is None:
            raise ValueError("ProximalMappingLoss.forward requires targets")
        del gt_density

        sigma = self._current_sigma(epoch)
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

        bg_distance_sq = (self.bg_threshold * sigma) ** 2
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
            orphan_mask = point_counts == 0
            if orphan_mask.any():
                orphan_count = torch.as_tensor(orphan_mask.sum().to(dtype), device=device)
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


class PatchSSIMLoss(nn.Module):
    """Patch-wise SSIM loss for local density structure.

    PML only constrains total density per Voronoi cell; this loss penalises
    mismatches in local spatial patterns (peaks, edges, texture) between the
    predicted and GT density maps.

    Reference: SPANet (Cheng et al., PR 2021) — SSIM auxiliary loss prevents
    over-smoothed density predictions in crowd counting.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 2.0,
        weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.weight = float(weight)
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss = (-coords.pow(2) / (2 * sigma * sigma)).exp()
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        self.register_buffer("kernel", kernel_2d.view(1, 1, kernel_size, kernel_size))
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(pred.dtype)
        pad = self.kernel_size // 2
        # Local statistics via Gaussian-filtered convolution
        mu_p = F.conv2d(pred, kernel, padding=pad)
        mu_g = F.conv2d(gt, kernel, padding=pad)
        mu_p_sq, mu_g_sq = mu_p.pow(2), mu_g.pow(2)
        mu_pg = mu_p * mu_g
        sigma_p_sq = F.conv2d(pred.pow(2), kernel, padding=pad) - mu_p_sq
        sigma_g_sq = F.conv2d(gt.pow(2), kernel, padding=pad) - mu_g_sq
        sigma_pg = F.conv2d(pred * gt, kernel, padding=pad) - mu_pg

        ssim = (
            (2 * mu_pg + self.c1) * (2 * sigma_pg + self.c2)
        ) / (
            (mu_p_sq + mu_g_sq + self.c1)
            * (sigma_p_sq + sigma_g_sq + self.c2).clamp_min(1e-8)
        )
        return self.weight * (1 - ssim.mean())


class GradientAwareLoss(nn.Module):
    """L1 loss between density gradient magnitudes (Sobel edge maps).

    Penalises mismatches in the spatial structure of density transitions —
    sharp peaks, crowd edges, and empty→occupied boundaries.  Encourages
    the model to learn correct local density shapes, not just total counts.

    Reference: ADSCNet (Bai et al., CVPR 2020) uses gradient differences
    as a self-correction signal in density map refinement.
    """

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = float(weight)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(pred.dtype)
        sy = self.sobel_y.to(pred.dtype)
        gx_p = F.conv2d(pred, sx, padding=1)
        gy_p = F.conv2d(pred, sy, padding=1)
        gx_g = F.conv2d(gt, sx, padding=1)
        gy_g = F.conv2d(gt, sy, padding=1)
        mag_p = (gx_p.pow(2) + gy_p.pow(2) + 1e-8).sqrt()
        mag_g = (gx_g.pow(2) + gy_g.pow(2) + 1e-8).sqrt()
        return self.weight * (mag_p - mag_g).abs().mean()


class SinkhornOTLoss(nn.Module):
    """Auxiliary OT loss: Sinkhorn divergence between pred and GT point distribution.

    Down-samples to a fixed grid for efficiency, then runs Sinkhorn iterations
    to measure distribution-level mismatch. Based on DM-Count (NeurIPS 2020).

    Works directly on point annotations — no Gaussian smoothing needed.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_iters: int = 50,
        max_grid: int = 32,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.max_grid = int(max_grid)
        self.weight = float(weight)

    def _build_cost_matrix(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
        # Normalise coordinates to [0, 1] so cost entries stay in [0, 2] regardless
        # of grid size.  This keeps exp(-C/epsilon) numerically stable for any grid.
        coords = coords / coords.abs().max().clamp_min(1.0)
        return (coords.unsqueeze(1) - coords.unsqueeze(0)).pow(2).sum(dim=-1)

    def _sinkhorn(
        self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        K = (-C / self.epsilon).exp()
        v = torch.ones_like(b)
        for _ in range(self.num_iters):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        return (P * C).sum()

    def forward(
        self,
        pred_density: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_sizes: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        device = pred_density.device
        # Sinkhorn exponentials underflow in fp16 — work in fp32 for stability.
        work_dtype = torch.float32
        batch_size = pred_density.shape[0]

        # Downsample to max_grid for efficiency
        _, _, H, W = pred_density.shape
        grid_h = min(H, self.max_grid)
        grid_w = min(W, self.max_grid)
        if H != grid_h or W != grid_w:
            pred = F.adaptive_avg_pool2d(pred_density, (grid_h, grid_w))
        else:
            pred = pred_density
        pred = pred.float()

        C = self._build_cost_matrix(grid_h, grid_w, device, work_dtype)
        total = torch.zeros((), device=device, dtype=work_dtype)

        if image_sizes is not None:
            img_h, img_w = int(image_sizes[0]), int(image_sizes[1])
        else:
            img_h, img_w = H, W

        for b in range(batch_size):
            p = pred[b, 0].reshape(-1)
            p_sum = p.sum().clamp_min(1e-8)
            p_norm = p / p_sum

            pts = targets[b]["point"].to(device=device, dtype=work_dtype)
            if pts.numel() == 0:
                total = total + p.pow(2).mean()
                continue

            # Build GT binary map at grid resolution.
            # pts are in image coordinates; scale to grid coordinates.
            gt = torch.zeros(grid_h, grid_w, device=device, dtype=work_dtype)
            scale_y = float(grid_h) / float(img_h)
            scale_x = float(grid_w) / float(img_w)
            pt_y = (pts[:, 1] * scale_y).long().clamp(0, grid_h - 1)
            pt_x = (pts[:, 0] * scale_x).long().clamp(0, grid_w - 1)
            gt[pt_y, pt_x] = 1.0
            g = gt.reshape(-1)
            g = g / g.sum().clamp_min(1e-8)

            total = total + self._sinkhorn(p_norm, g, C)

        return self.weight * total


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


class ExpertImportanceLoss(nn.Module):
    """HMoDE-style expert importance balancing loss (Du et al., IEEE TIP 2023).

    Penalises CV^2 of per-expert importance to prevent any expert from being
    suppressed during pixel-wise soft routing.
    """

    def __init__(self, lambda_importance: float = 0.01, eps: float = 1e-8) -> None:
        super().__init__()
        self.lambda_importance = float(lambda_importance)
        self.eps = float(eps)

    def forward(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        importance = weights.sum(dim=(0, 2, 3))  # [K]
        mean = importance.mean().clamp_min(self.eps)
        variance = (importance - mean).pow(2).mean()
        cv_sq = variance / mean.pow(2)
        total = self.lambda_importance * cv_sq
        return {"l_importance": cv_sq, "total_aux": total}


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
        point_loss_weight: float = 0.0,
        point_cost_class: float = 1.0,
        point_cost_l1: float = 1.0,
        point_focal_alpha: float = 0.75,
        point_focal_gamma: float = 2.0,
        point_eos_coef: float = 0.1,
        ot_loss: SinkhornOTLoss | None = None,
        ot_weight: float = 0.1,
        ssim_loss: PatchSSIMLoss | None = None,
        ssim_weight: float = 0.05,
        grad_loss: GradientAwareLoss | None = None,
        grad_weight: float = 0.01,
        diversity_weight: float = 0.05,
        expert_supervision_weight: float = 0.2,
        density_s4_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.pml_loss = pml_loss
        self.bayesian_loss = bayesian_loss
        self.count_loss = count_loss or CountLoss()
        self.count_weight = float(count_weight)
        self.balance_loss = balance_loss or LoadBalanceLoss()
        self.warmup_end = int(warmup_end)
        self.balance_decay_epochs = int(balance_decay_epochs)
        self.ot_loss = ot_loss
        self.ot_weight = float(ot_weight)
        self.ssim_loss = ssim_loss
        self.ssim_weight = float(ssim_weight)
        self.grad_loss = grad_loss
        self.grad_weight = float(grad_weight)
        self.point_loss_weight = float(point_loss_weight)
        self.diversity_weight = float(diversity_weight)
        self.expert_supervision_weight = float(expert_supervision_weight)
        self.density_s4_weight = float(density_s4_weight)
        if self.point_loss_weight > 0:
            self.matcher = HungarianMatcher_Crowd(
                cost_class=float(point_cost_class),
                cost_point=float(point_cost_l1),
            )
            self.point_focal_alpha = float(point_focal_alpha)
            self.point_focal_gamma = float(point_focal_gamma)
            empty_weight = torch.ones(2)
            empty_weight[0] = float(point_eos_coef)
            self.register_buffer("point_empty_weight", empty_weight)

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
                epoch=epoch,
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

        # OT auxiliary loss (distribution-level, before count loss)
        ot = pred_density.new_zeros(())
        if self.ot_loss is not None:
            ot = self.ot_loss(pred_density, targets, image_sizes)

        # SSIM local structure loss (patch-wise structural similarity)
        ssim = pred_density.new_zeros(())
        if self.ssim_loss is not None and gt_density is not None:
            ssim = self.ssim_loss(pred_density, gt_density)

        # Gradient-aware loss (edge/transition matching)
        grad = pred_density.new_zeros(())
        if self.grad_loss is not None and gt_density is not None:
            grad = self.grad_loss(pred_density, gt_density)

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

        total = density_loss + ot + ssim + grad + count + scaled_balance

        # Expert activation-space diversity loss: penalise high cosine similarity
        # between expert outputs.  During training the cosine matrix is connected
        # to the computation graph (experts.py - eo not detached).  Inspired by
        # Kim et al. (arXiv:2601.00457): regularise in activation space, not weight space.
        diversity = pred_density.new_zeros(())
        expert_similarity = outputs.get("expert_similarity", {})
        if self.diversity_weight > 0 and isinstance(expert_similarity, dict):
            sim_values = [
                v for v in expert_similarity.values()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ]
            if sim_values:
                diversity = torch.stack(sim_values).mean() * self.diversity_weight
                total = total + diversity

        # Expert-wise deep supervision: each expert must independently produce a
        # meaningful density map.  This forces the experts to specialise by design:
        # LocalDetail learns fine textures, SpatialRelation learns mid-range
        # patterns, GlobalDensity learns scene-level density levels.
        expert_sup = pred_density.new_zeros(())
        expert_densities = outputs.get("expert_densities")
        if (
            self.expert_supervision_weight > 0
            and isinstance(expert_densities, torch.Tensor)
            and self.pml_loss is not None
        ):
            num_experts = int(expert_densities.shape[1])
            for e_idx in range(num_experts):
                ed = expert_densities[:, e_idx]  # [B, 1, H, W]
                aligned_ed = ed[:, :, :pred_density.shape[2], :pred_density.shape[3]]
                expert_sup = expert_sup + self.pml_loss(
                    aligned_ed, gt_density, targets=targets, image_sizes=image_sizes, epoch=epoch,
                )
            expert_sup = (expert_sup / (batch_size * num_experts)) * self.expert_supervision_weight
            total = total + expert_sup

        # Stride-4 density supervision (P3-2): coarse-to-fine PML at 4x resolution
        loss_density_s4 = pred_density.new_zeros(())
        density_s4 = outputs.get("density_s4")
        if (
            self.density_s4_weight > 0
            and isinstance(density_s4, torch.Tensor)
            and self.pml_loss is not None
        ):
            loss_density_s4 = self.pml_loss(
                density_s4, gt_density, targets=targets, image_sizes=image_sizes, epoch=epoch,
            ) / batch_size * self.density_s4_weight
            total = total + loss_density_s4

        result = {
            "loss_total": total,
            loss_label: density_loss,
            "loss_ot": ot,
            "loss_ssim": ssim,
            "loss_grad": grad,
            "loss_count": count,
            "loss_balance": scaled_balance,
            "loss_balance_raw": balance,
            "balance_scale": pred_density.new_tensor(balance_scale),
            "lambda_count": pred_density.new_tensor(self.count_weight),
            "loss_diversity": diversity,
            "loss_expert_sup": expert_sup,
            "loss_density_s4": loss_density_s4,
        }

        # Point auxiliary loss (Hungarian matching + focal cls + SmoothL1 reg)
        if (
            self.point_loss_weight > 0
            and "pred_logits" in outputs
            and "pred_points" in outputs
        ):
            point_losses = self._compute_point_loss(outputs, targets)
            result.update(point_losses)
            result["loss_total"] = result["loss_total"] + point_losses["loss_point_total"]

        return result

    def _compute_point_loss(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_points = outputs["pred_points"]
        if not isinstance(pred_logits, torch.Tensor) or not isinstance(pred_points, torch.Tensor):
            return {"loss_point_total": outputs["density_out"].new_zeros(())}  # type: ignore[union-attr]
        device = pred_logits.device

        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_points = pred_points.flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points, tgt_points, p=2)
        C = self.matcher.cost_point * cost_point + self.matcher.cost_class * cost_class
        bs, num_queries = pred_logits.shape[:2]
        C = C.view(bs, num_queries, -1).cpu().detach()

        sizes = [len(v["point"]) for v in targets]
        from scipy.optimize import linear_sum_assignment
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

        # Classification loss (focal)
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            pred_logits.shape[:2], 0, dtype=torch.int64, device=device
        )
        target_classes[batch_idx, src_idx] = target_classes_o

        loss_cls = softmax_focal_loss(
            pred_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            alpha=self.point_focal_alpha,
            gamma=self.point_focal_gamma,
            class_weight=self.point_empty_weight,  # type: ignore[arg-type]
        )

        # Regression loss (SmoothL1)
        src_points = pred_points[batch_idx, src_idx]
        target_pts = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        num_points = max(sum(sizes), 1)
        loss_reg = F.smooth_l1_loss(src_points, target_pts, beta=1.0, reduction="sum") / num_points

        total = loss_cls + loss_reg
        return {
            "loss_point_cls": loss_cls,
            "loss_point_reg": loss_reg,
            "loss_point_total": self.point_loss_weight * total,
            "point_loss_weight": pred_logits.new_tensor(self.point_loss_weight),
        }

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
            # Collect per-point density values across chunks for global simplex projection
            point_densities: list[list[torch.Tensor]] = [[] for _ in range(num_points)]
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
                    point_densities[p_idx].append(fg_density[point_mask])

            # Global simplex projection: process all pixels assigned to each point
            # together so the "sum-to-1" constraint is exact, not per-chunk approximate.
            point_loss = torch.zeros((), device=device, dtype=dtype)
            for p_idx in range(num_points):
                if not point_densities[p_idx]:
                    point_loss = point_loss + 0.1  # orphan penalty
                    continue
                p_density = torch.cat(point_densities[p_idx])
                target = _project_onto_simplex(p_density)
                point_loss = point_loss + (p_density - target).pow(2).sum()

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
        output_stride: int = 8,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        if output_stride < 1:
            raise ValueError("output_stride must be >= 1")
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.max_grid = int(max_grid)
        self.weight = float(weight)
        self.output_stride = int(output_stride)

    def _build_cost_matrix(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
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
    ) -> torch.Tensor:
        device = pred_density.device
        dtype = pred_density.dtype
        batch_size = pred_density.shape[0]

        # Downsample to max_grid for efficiency
        _, _, H, W = pred_density.shape
        grid_h = min(H, self.max_grid)
        grid_w = min(W, self.max_grid)
        if H != grid_h or W != grid_w:
            pred = F.adaptive_avg_pool2d(pred_density, (grid_h, grid_w))
        else:
            pred = pred_density

        C = self._build_cost_matrix(grid_h, grid_w, device, dtype)
        total = torch.zeros((), device=device, dtype=dtype)

        for b in range(batch_size):
            p = pred[b, 0].reshape(-1)
            p_sum = p.sum().clamp_min(1e-8)
            p_norm = p / p_sum

            pts = targets[b]["point"].to(device=device, dtype=dtype)
            if pts.numel() == 0:
                total = total + p.pow(2).mean()
                continue

            # Build GT binary map at grid resolution.
            # Targets are in image pixel coords; first convert to stride-K coords
            # (dividing by output_stride), then scale to grid coords.
            gt = torch.zeros(grid_h, grid_w, device=device, dtype=dtype)
            pt_y = (pts[:, 1] / self.output_stride * float(grid_h) / float(H)).long().clamp(0, grid_h - 1)
            pt_x = (pts[:, 0] / self.output_stride * float(grid_w) / float(W)).long().clamp(0, grid_w - 1)
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


class TotalVariationLoss(nn.Module):
    """Total variation regularizer to encourage spatially smooth density maps."""

    def __init__(self, weight: float = 0.001) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, pred_density: torch.Tensor) -> torch.Tensor:
        diff_x = (pred_density[:, :, :, 1:] - pred_density[:, :, :, :-1]).abs().mean()
        diff_y = (pred_density[:, :, 1:, :] - pred_density[:, :, :-1, :]).abs().mean()
        return self.weight * (diff_x + diff_y)


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
        tv_loss: TotalVariationLoss | None = None,
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
        self.tv_loss = tv_loss
        self.point_loss_weight = float(point_loss_weight)
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
            ot = self.ot_loss(pred_density, targets) * self.ot_weight

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

        tv = self.tv_loss(pred_density) if self.tv_loss is not None else pred_density.new_zeros(())
        total = density_loss + ot + tv + count + scaled_balance
        result = {
            "loss_total": total,
            loss_label: density_loss,
            "loss_ot": ot,
            "loss_tv": tv,
            "loss_count": count,
            "loss_balance": scaled_balance,
            "loss_balance_raw": balance,
            "balance_scale": pred_density.new_tensor(balance_scale),
            "lambda_count": pred_density.new_tensor(self.count_weight),
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

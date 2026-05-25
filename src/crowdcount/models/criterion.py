"""SetCriterion for crowd counting (classification + point regression losses)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.utils.misc import get_world_size, is_dist_avail_and_initialized


def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax focal loss for mutually exclusive classification.

    Uses softmax (not sigmoid) since crowd counting has mutually exclusive
    foreground/background classes.  Normalisation matches
    ``F.cross_entropy(..., weight=..., reduction='mean')`` when ``gamma=0``
    so that switching between focal and CE does not change loss scale.

    Args:
        inputs: [N, C] raw logits (before softmax).
        targets: [N] integer class labels in [0, C).
        alpha: Weighting factor for the **foreground** class (background
               gets ``1 - alpha``).  Only used when *class_weight* is None.
        gamma: Focusing parameter (higher ⇒ more focus on hard examples).
        class_weight: Optional per-class weight tensor of shape [C].
               When provided, this is used **instead of** alpha to avoid
               double-weighting.

    Returns:
        Scalar focal loss (weighted-mean normalised).
    """
    # Per-sample unweighted CE
    ce = F.cross_entropy(inputs, targets, reduction="none")  # [N]

    # p_t: softmax probability assigned to the true class
    p = F.softmax(inputs, dim=-1)
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    focal_weight = (1 - p_t) ** gamma  # [N]

    # Class weighting: use class_weight if provided, otherwise use alpha
    if class_weight is not None:
        w_t = class_weight[targets]  # [N]
    else:
        # Foreground (label>0) gets alpha, background gets 1-alpha
        w_t = torch.where(
            targets > 0,
            torch.tensor(alpha, device=inputs.device),
            torch.tensor(1 - alpha, device=inputs.device),
        )

    loss = w_t * focal_weight * ce  # [N]

    # Weighted-mean: divide by sum(w_t) to match F.cross_entropy(weight=...) scale
    return loss.sum() / w_t.sum().clamp(min=1.0)


def quality_focal_loss(
    inputs: torch.Tensor,
    targets_soft: torch.Tensor,
    beta: float = 2.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quality Focal Loss (QFL) for soft classification targets.

    Extends focal loss to support continuous quality targets in [0, 1] instead
    of hard 0/1 labels.  For matched predictions the target encodes localisation
    quality (e.g. ``exp(-dist / sigma)``); for unmatched predictions it is 0.

    The loss for a single sample with 2-class softmax probability ``p`` and
    soft target ``y ∈ [0, 1]`` (foreground class) is::

        QFL = -|y - p|^β · [y · log(p) + (1 - y) · log(1 - p)]

    where ``p = softmax(z)[fg_class]`` and β controls focus on hard examples.

    Args:
        inputs:       [N, 2] raw logits (background, foreground).
        targets_soft: [N] soft quality target for the foreground class, in [0, 1].
        beta:         Focusing parameter (default 2.0).
        class_weight: Optional [2] tensor for background/foreground weighting.

    Returns:
        Scalar QFL loss (mean-normalised).
    """
    probs = F.softmax(inputs, dim=-1)  # [N, 2]
    p_fg = probs[:, 1]  # foreground probability

    # Binary cross-entropy with soft target (per-sample, no reduction)
    bce = -(
        targets_soft * torch.log(p_fg.clamp(min=1e-7))
        + (1 - targets_soft) * torch.log((1 - p_fg).clamp(min=1e-7))
    )  # [N]

    # Focusing weight: |y - p|^β
    focal_weight = (targets_soft - p_fg).abs().pow(beta)  # [N]

    loss = focal_weight * bce  # [N]

    # Optional class weighting: use fg weight for matched, bg weight for unmatched
    if class_weight is not None:
        w_t = torch.where(
            targets_soft > 0,
            class_weight[1],
            class_weight[0],
        )
        loss = loss * w_t
        return loss.sum() / w_t.sum().clamp(min=1.0)

    return loss.mean()


class SetCriterion_Crowd(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: dict,
        eos_coef: float,
        losses,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_uncertainty_weighting: bool = False,
        uncertainty_boost: float = 2.0,
        use_qfl: bool = False,
        qfl_beta: float = 2.0,
        qfl_sigma: float = 10.0,
        label_smoothing: float = 0.0,
        point_loss_type: str = "smooth_l1",
        point_smooth_l1_beta: float = 1.0,
        point_density_feedback_margin: float = 1.0,
        point_density_feedback_count_weight: float = 0.1,
        point_density_feedback_detach_points: bool = True,
        point_density_feedback_detach_scores: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty_boost = uncertainty_boost
        if use_qfl and use_focal_loss:
            raise ValueError(
                "use_qfl and use_focal_loss are mutually exclusive; enable only one."
            )
        self.label_smoothing = float(label_smoothing)
        if self.label_smoothing < 0.0 or self.label_smoothing >= 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.label_smoothing > 0.0 and (use_focal_loss or use_qfl):
            raise ValueError(
                "label_smoothing is only supported with standard cross entropy; "
                "disable focal/QFL or set label_smoothing=0.0"
            )
        self.use_qfl = use_qfl
        self.qfl_beta = qfl_beta
        self.qfl_sigma = qfl_sigma
        point_loss_type = point_loss_type.lower()
        if point_loss_type == "l2":
            point_loss_type = "mse"
        if point_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(
                "point_loss_type must be one of {'smooth_l1', 'mse', 'l2'}"
            )
        self.point_loss_type = point_loss_type
        self.point_smooth_l1_beta = point_smooth_l1_beta
        if point_density_feedback_margin <= 0:
            raise ValueError("point_density_feedback_margin must be positive")
        if point_density_feedback_count_weight < 0:
            raise ValueError("point_density_feedback_count_weight must be non-negative")
        self.point_density_feedback_margin = float(point_density_feedback_margin)
        self.point_density_feedback_count_weight = float(
            point_density_feedback_count_weight
        )
        self.point_density_feedback_detach_points = bool(
            point_density_feedback_detach_points
        )
        self.point_density_feedback_detach_scores = bool(
            point_density_feedback_detach_scores
        )
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if self.use_qfl:
            # Quality Focal Loss: soft targets encoding localisation quality
            assert "pred_points" in outputs
            src_points = outputs["pred_points"]  # [B, Q, 2]

            # Build soft quality target: 0 for unmatched, exp(-dist/sigma) for matched
            soft_targets = torch.zeros(
                src_logits.shape[:2], device=src_logits.device
            )  # [B, Q]
            # Compute quality scores for matched predictions
            matched_src_pts = src_points[idx]  # [M, 2]
            matched_gt_pts = torch.cat(
                [t["point"][J] for t, (_, J) in zip(targets, indices)], dim=0
            )  # [M, 2]
            dists = (matched_src_pts.detach() - matched_gt_pts).pow(2).sum(-1).sqrt()
            quality = torch.exp(-dists / self.qfl_sigma)
            soft_targets[idx] = quality

            loss_ce = quality_focal_loss(
                src_logits.flatten(0, 1),  # [B*Q, 2]
                soft_targets.flatten(0, 1),  # [B*Q]
                beta=self.qfl_beta,
                class_weight=self.empty_weight,
            )
        elif self.use_focal_loss:
            loss_ce = softmax_focal_loss(
                src_logits.flatten(0, 1),  # [B*Q, C]
                target_classes.flatten(0, 1),  # [B*Q]
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                class_weight=self.empty_weight,
            )
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight,
                label_smoothing=self.label_smoothing,
            )
        return {"loss_ce": loss_ce}

    def loss_points(self, outputs, targets, indices, num_points):
        assert "pred_points" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        if self.point_loss_type == "mse":
            loss_bbox = F.mse_loss(src_points, target_points, reduction="none")
        else:
            loss_bbox = F.smooth_l1_loss(
                src_points,
                target_points,
                reduction="none",
                beta=self.point_smooth_l1_beta,
            )

        # Uncertainty weighting: boost loss for points in high-uncertainty regions
        uncertainty_map = outputs.get("uncertainty_map")
        if self.use_uncertainty_weighting and uncertainty_map is not None:
            # Sample uncertainty at each matched GT point location
            # target_points are in pixel coords; uncertainty_map is [B,1,H,W]
            # Normalise by image spatial dims so grid_sample maps correctly
            # to the downsampled feature map.
            img_size = outputs.get("img_size")  # (img_h, img_w)
            if img_size is not None:
                H_norm, W_norm = float(img_size[0]), float(img_size[1])
            else:
                _, _, H_norm, W_norm = (
                    1,
                    1,
                    float(uncertainty_map.shape[2]),
                    float(uncertainty_map.shape[3]),
                )
            per_point_unc = []
            for b_val, t_pts, (_, J) in zip(range(len(targets)), targets, indices):
                pts = t_pts["point"][J]  # [n, 2] in pixel coords
                if pts.numel() == 0:
                    continue
                # Normalise coords to [-1, 1] for grid_sample
                grid_x = (pts[:, 0] / max(W_norm - 1, 1)) * 2.0 - 1.0
                grid_y = (pts[:, 1] / max(H_norm - 1, 1)) * 2.0 - 1.0
                grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
                sampled = F.grid_sample(
                    uncertainty_map[b_val : b_val + 1],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )  # [1, 1, 1, n]
                per_point_unc.append(sampled.view(-1))
            if per_point_unc:
                unc_weights = torch.cat(per_point_unc, dim=0)  # [total_matched]
                weights = 1.0 + self.uncertainty_boost * unc_weights  # [1, 1+boost]
                loss_bbox = loss_bbox * weights.unsqueeze(-1)

        return {"loss_points": loss_bbox.sum() / num_points}

    def loss_count(self, outputs, targets, indices, num_points):
        """Global counting loss: L1(predicted_count, gt_count)."""
        pred_scores = outputs["pred_logits"].softmax(-1)[:, :, 1]  # [B, Q]
        pred_counts = pred_scores.sum(dim=1)  # [B]
        gt_counts = torch.tensor(
            [t["point"].shape[0] for t in targets],
            dtype=torch.float,
            device=pred_scores.device,
        )
        return {"loss_count": F.l1_loss(pred_counts, gt_counts)}

    def loss_refine(self, outputs, targets, indices, num_points):
        """Intermediate refinement loss: weighted Smooth L1 across steps.

        Uses the same Hungarian matching indices as the main point loss so
        that each refinement step is supervised towards the same GT target.
        """
        intermediates = outputs.get("refine_intermediates")
        if intermediates is None or len(intermediates) <= 1:
            device = outputs["pred_logits"].device
            return {"loss_refine": torch.tensor(0.0, device=device)}

        idx = self._get_src_permutation_idx(indices)
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        T = len(intermediates) - 1  # number of refinement steps
        total = torch.tensor(0.0, device=target_points.device)
        for t, pts in enumerate(intermediates):
            # Linearly increasing weight: 0.5 for step 0 → 1.0 for step T
            w_t = 0.5 + 0.5 * t / max(T, 1)
            src = pts[idx]
            step_loss = F.smooth_l1_loss(
                src,
                target_points,
                reduction="none",
                beta=1.0,
            )
            total = total + w_t * step_loss.sum() / max(num_points, 1)

        return {"loss_refine": total / (T + 1)}

    def loss_consistency(self, outputs, targets, indices, num_points):
        """Density-point consistency loss (mPrompt-inspired bidirectional constraint).

        Two terms:
          1) Point→Density: density map should be high at matched GT point locations.
             Uses a hinge loss: mean(max(0, margin - sampled_density)).
          2) Count consistency: density integral should match foreground score sum.
             Uses L1 between density_sum and predicted_fg_count.

        Requires ``density_out`` in *outputs*.
        """
        density_out = outputs.get("density_out")
        if density_out is None:
            device = outputs["pred_logits"].device
            return {"loss_consistency": torch.tensor(0.0, device=device)}

        device = density_out.device
        B = density_out.shape[0]

        # Use image spatial dims for coordinate normalisation so that
        # grid_sample maps pixel coords correctly onto the downsampled
        # density map.  Falls back to density map dims when img_size is
        # unavailable (e.g. in unit tests with minimal outputs).
        img_size = outputs.get("img_size")  # (img_h, img_w)
        if img_size is not None:
            H_norm, W_norm = float(img_size[0]), float(img_size[1])
        else:
            H_norm = float(density_out.shape[2])
            W_norm = float(density_out.shape[3])

        # --- Term 1: Point→Density agreement ---
        # Sample density at each matched GT point via bilinear interpolation.
        per_point_density: list[torch.Tensor] = []
        for b_val, t, (_, J) in zip(range(B), targets, indices):
            pts = t["point"][J]  # [n, 2] in pixel coords (x, y)
            if pts.numel() == 0:
                continue
            # Normalise coords to [-1, 1] for grid_sample
            grid_x = (pts[:, 0] / max(W_norm - 1, 1)) * 2.0 - 1.0
            grid_y = (pts[:, 1] / max(H_norm - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
            sampled = F.grid_sample(
                density_out[b_val : b_val + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # [1, 1, 1, n]
            per_point_density.append(sampled.view(-1))

        if per_point_density:
            all_density = torch.cat(per_point_density, dim=0)  # [total_matched]
            # Hinge loss: density at GT points should be >= 1.0
            loss_point_density = torch.clamp(1.0 - all_density, min=0.0).mean()
        else:
            loss_point_density = torch.tensor(0.0, device=device)

        # --- Term 2: Count consistency ---
        # density integral vs foreground prediction score sum
        density_count = density_out.sum(dim=[1, 2, 3])  # [B]
        pred_fg_scores = outputs["pred_logits"].softmax(-1)[:, :, 1]  # [B, Q]
        pred_count = pred_fg_scores.sum(dim=1)  # [B]
        loss_count_consistency = F.l1_loss(density_count, pred_count)

        return {"loss_consistency": loss_point_density + loss_count_consistency}

    def loss_point_density_feedback(self, outputs, targets, indices, num_points):
        """Use matched predicted points as a detached localisation prior for density."""
        density_out = outputs.get("density_out")
        if density_out is None:
            device = outputs["pred_logits"].device
            return {"loss_point_density_feedback": torch.tensor(0.0, device=device)}

        device = density_out.device
        pred_points = outputs["pred_points"]
        pred_logits = outputs["pred_logits"]
        img_size = outputs.get("img_size")
        if img_size is not None:
            H_norm, W_norm = float(img_size[0]), float(img_size[1])
        else:
            H_norm = float(density_out.shape[2])
            W_norm = float(density_out.shape[3])

        sampled_density: list[torch.Tensor] = []
        sampled_weight: list[torch.Tensor] = []
        fg_scores = pred_logits.softmax(-1)[:, :, 1]
        for b_val, (src_idx, _) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device)
            pts = pred_points[b_val, src_idx]
            scores = fg_scores[b_val, src_idx].clamp_min(0.0)
            if self.point_density_feedback_detach_points:
                pts = pts.detach()
            if self.point_density_feedback_detach_scores:
                scores = scores.detach()

            grid_x = (pts[:, 0] / max(W_norm - 1, 1)) * 2.0 - 1.0
            grid_y = (pts[:, 1] / max(H_norm - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
            sampled = F.grid_sample(
                density_out[b_val : b_val + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_density.append(sampled.view(-1))
            sampled_weight.append(scores)

        if sampled_density:
            all_density = torch.cat(sampled_density, dim=0)
            all_weight = torch.cat(sampled_weight, dim=0)
            point_loss = (
                torch.clamp(
                    self.point_density_feedback_margin - all_density,
                    min=0.0,
                )
                * all_weight
            ).sum() / all_weight.sum().clamp_min(1.0)
        else:
            point_loss = torch.tensor(0.0, device=device)

        gt_count = density_out.new_tensor([len(t["labels"]) for t in targets])
        density_count = density_out.sum(dim=[1, 2, 3])
        count_loss = (
            (density_count - gt_count).abs() / gt_count.clamp_min(1.0)
        ).mean()
        total = point_loss + self.point_density_feedback_count_weight * count_loss
        return {"loss_point_density_feedback": total}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "points": self.loss_points,
            "count": self.loss_count,
            "refine": self.loss_refine,
            "consistency": self.loss_consistency,
            "point_density_feedback": self.loss_point_density_feedback,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        output1 = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
            "refine_intermediates": outputs.get("refine_intermediates"),
            "uncertainty_map": outputs.get("uncertainty_map"),
            "density_out": outputs.get("density_out"),
            "density_base": outputs.get("density_base"),
            "point_feedback_heatmap": outputs.get("point_feedback_heatmap"),
            "img_size": outputs.get("img_size"),
        }
        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=next(iter(output1.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))
        return losses

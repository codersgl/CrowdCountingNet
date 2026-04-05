"""SetCriterion for crowd counting (classification + point regression losses)."""

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.utils.misc import get_world_size, is_dist_avail_and_initialized


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sigmoid focal loss for dense classification (RetinaNet-style).

    Args:
        inputs: [N, C] raw logits (before sigmoid).
        targets: [N] integer class labels in [0, C).
        alpha: Weighting factor for the foreground class.
        gamma: Focusing parameter (higher ⇒ more focus on hard examples).
        class_weight: Optional per-class weight tensor of shape [C].

    Returns:
        Scalar mean focal loss.
    """
    num_classes = inputs.shape[-1]
    # One-hot encode targets → [N, C]
    target_onehot = F.one_hot(targets, num_classes=num_classes).float()

    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction="none")
    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    focal_weight = (1 - p_t) ** gamma

    # Per-class alpha weighting: foreground gets alpha, background gets (1 - alpha)
    alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)

    loss = alpha_t * focal_weight * ce  # [N, C]

    # Apply optional class weight (e.g. eos_coef for background)
    if class_weight is not None:
        loss = loss * class_weight.unsqueeze(0)

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

        if self.use_focal_loss:
            loss_ce = sigmoid_focal_loss(
                src_logits.flatten(0, 1),  # [B*Q, C]
                target_classes.flatten(0, 1),  # [B*Q]
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                class_weight=self.empty_weight,
            )
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        return {"loss_ce": loss_ce}

    def loss_points(self, outputs, targets, indices, num_points):
        assert "pred_points" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_bbox = F.smooth_l1_loss(
            src_points, target_points, reduction="none", beta=1.0
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

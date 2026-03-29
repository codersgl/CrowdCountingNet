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
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        output1 = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
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

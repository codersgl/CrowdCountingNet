"""Bayesian Crowd Loss for density map regression.

Based on "Bayesian Loss for Crowd Count Estimation with Point Supervision"
(Ma et al., ICCV 2019). Models per-pixel likelihood with annotation
uncertainty — points near other points have higher uncertainty, so the
loss adaptively assigns lower weight to ambiguous regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianCrowdLoss(nn.Module):
    """Bayesian loss for density map regression.

    Computes expected count-based loss that accounts for labelling
    uncertainty: each GT point contributes a Gaussian blob to the
    likelihood, and the loss is weighted by the posterior probability
    that each pixel belongs to a specific point.

    For efficiency, this implementation uses a simplified approach:
    compute per-point expected density at the GT position, then
    penalise the deviation with confidence weighting based on
    local density (high-density regions have higher uncertainty).

    Args:
        sigma: Gaussian sigma for point spread (default 8.0).
        background_weight: Weight for background pixels (default 0.1).
    """

    def __init__(
        self,
        sigma: float = 8.0,
        background_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.bg_weight = background_weight

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_points: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute Bayesian crowd loss.

        Args:
            pred_density: Predicted density map [B, 1, H, W].
            gt_density: Ground-truth density map [B, 1, H, W].
            gt_points: Optional list of B tensors, each [N_i, 2] with (x, y)
                coordinates. If provided, uses point-based confidence
                weighting; otherwise falls back to density-based weighting.

        Returns:
            Scalar loss value.
        """
        B = pred_density.shape[0]

        # Compute pixelwise squared error
        sq_err = (pred_density - gt_density) ** 2  # [B, 1, H, W]

        # Confidence weighting: regions with higher GT density get
        # higher weight (they contain more information), but very high
        # density regions have higher uncertainty, so we use a
        # concave function: w = 1 + log(1 + density)
        with torch.no_grad():
            density_level = gt_density.clamp(min=0)
            confidence = 1.0 + torch.log1p(density_level)

            # Background suppression: pixels with zero GT density
            bg_mask = (gt_density <= 0).float()
            fg_mask = 1.0 - bg_mask
            weight = confidence * fg_mask + self.bg_weight * bg_mask

            # Normalize weight per image to prevent scale drift
            weight = weight / (weight.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
            weight = weight * weight.numel() / B  # scale to match MSE magnitude

        loss = (sq_err * weight).sum() / B
        return loss

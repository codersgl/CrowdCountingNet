"""Local Count Ranking Loss.

Splits the density map into patches and enforces that the relative ordering
of predicted patch counts matches the ground-truth ordering. This provides
a strong supervisory signal that is robust to absolute annotation noise
and prevents local over-/under-counting in occluded regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCountRankingLoss(nn.Module):
    """Local count ranking loss for density map regression.

    Divides the density map into a grid of patches, computes per-patch
    counts, then applies a margin ranking loss on random pairs to enforce
    correct relative ordering.

    Args:
        grid_size: Number of patches per dimension (grid_size × grid_size).
        margin: Margin for ranking loss.
        num_pairs: Number of random patch pairs to sample per image.
    """

    def __init__(
        self,
        grid_size: int = 4,
        margin: float = 0.0,
        num_pairs: int = 16,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin
        self.num_pairs = num_pairs
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local count ranking loss.

        Args:
            pred_density: Predicted density [B, 1, H, W].
            gt_density: Ground-truth density [B, 1, H, W].

        Returns:
            Scalar loss value.
        """
        B = pred_density.shape[0]
        G = self.grid_size

        # Compute patch counts via adaptive average pooling + area scaling
        pred_patch = F.adaptive_avg_pool2d(pred_density, G)  # [B, 1, G, G]
        gt_patch = F.adaptive_avg_pool2d(gt_density, G)  # [B, 1, G, G]

        # Convert avg to sum (approximate)
        H, W = pred_density.shape[-2:]
        patch_area = (H / G) * (W / G)
        pred_counts = pred_patch.reshape(B, -1) * patch_area  # [B, G*G]
        gt_counts = gt_patch.reshape(B, -1) * patch_area  # [B, G*G]

        num_patches = G * G
        if num_patches < 2:
            return torch.tensor(0.0, device=pred_density.device)

        # Sample random pairs
        actual_pairs = min(self.num_pairs, num_patches * (num_patches - 1) // 2)
        idx_a = torch.randint(
            0, num_patches, (B, actual_pairs), device=pred_density.device
        )
        idx_b = torch.randint(
            0, num_patches, (B, actual_pairs), device=pred_density.device
        )

        # Ensure different patches
        same_mask = idx_a == idx_b
        idx_b[same_mask] = (idx_b[same_mask] + 1) % num_patches

        # Gather counts
        pred_a = torch.gather(pred_counts, 1, idx_a)  # [B, P]
        pred_b = torch.gather(pred_counts, 1, idx_b)  # [B, P]
        gt_a = torch.gather(gt_counts, 1, idx_a)  # [B, P]
        gt_b = torch.gather(gt_counts, 1, idx_b)  # [B, P]

        # Target: +1 if gt_a > gt_b, -1 if gt_a < gt_b, 0 if equal
        target = torch.sign(gt_a - gt_b)  # [B, P]

        # Filter out ties (target == 0)
        valid = target != 0
        if not valid.any():
            return torch.tensor(0.0, device=pred_density.device)

        pred_a_valid = pred_a[valid]
        pred_b_valid = pred_b[valid]
        target_valid = target[valid]

        return self.ranking_loss(pred_a_valid, pred_b_valid, target_valid)

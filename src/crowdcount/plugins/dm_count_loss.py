"""DM-Count style density loss for RCCFormer.

Implements the three-component loss from RCCFormer (arXiv:2504.04935) §3.3,
which follows the DM-Count (Distribution Matching for Crowd Counting) paradigm:

    L_count = L_C(C', C) + λ1 · L_OT(D', D) + λ2 · L_TV(D')

Components:
    - L_C:  Counting loss — L1 or MSE between predicted and GT total counts.
    - L_OT: Optimal Transport loss — 1D Wasserstein distance computed on
            H-marginal and W-marginal distributions (closed-form, efficient).
    - L_TV: Total Variation loss — spatial smoothness regularisation on the
            predicted density map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CountingLoss(nn.Module):
    """Global counting loss: |sum(D') - sum(D)|."""

    def __init__(self, mode: str = "l1") -> None:
        super().__init__()
        if mode not in ("l1", "mse"):
            raise ValueError(f"count_loss_type must be 'l1' or 'mse', got '{mode}'")
        self.mode = mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 1, H, W] predicted density map.
            target: [B, 1, H, W] ground-truth density map.
        Returns:
            Scalar loss averaged over the batch.
        """
        pred_count = pred.sum(dim=(1, 2, 3))  # [B]
        gt_count = target.sum(dim=(1, 2, 3))  # [B]
        if self.mode == "l1":
            return (pred_count - gt_count).abs().mean()
        return ((pred_count - gt_count) ** 2).mean()


class OTLoss(nn.Module):
    """1-D Wasserstein distance on H-marginal + W-marginal distributions.

    Following DM-Count, the 2-D OT is decomposed into two independent 1-D
    problems for efficiency.  The closed-form 1-D Wasserstein-1 distance is
    simply the L1 distance between cumulative distribution functions (CDFs):

        W_1(p, q) = Σ_i |CDF_p(i) − CDF_q(i)|
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 1, H, W] predicted density map (non-negative).
            target: [B, 1, H, W] ground-truth density map (non-negative).
        Returns:
            Scalar OT loss averaged over the batch.
        """
        # Squeeze channel dim → [B, H, W]
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        # H-marginal: sum over W → [B, H]
        pred_h = pred.sum(dim=-1)
        gt_h = target.sum(dim=-1)

        # W-marginal: sum over H → [B, W]
        pred_w = pred.sum(dim=-2)
        gt_w = target.sum(dim=-2)

        w1_h = self._wasserstein_1d(pred_h, gt_h)
        w1_w = self._wasserstein_1d(pred_w, gt_w)

        return (w1_h + w1_w).mean()

    def _wasserstein_1d(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute 1-D Wasserstein-1 distance per sample.

        Args:
            p, q: [B, N] non-negative distributions (need not be normalised).
        Returns:
            [B] per-sample W1 distances.
        """
        # Normalise to probability distributions
        p_sum = p.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        q_sum = q.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p_norm = p / p_sum
        q_norm = q / q_sum

        # CDF = cumulative sum
        cdf_p = p_norm.cumsum(dim=-1)
        cdf_q = q_norm.cumsum(dim=-1)

        return (cdf_p - cdf_q).abs().sum(dim=-1)


class TVLoss(nn.Module):
    """Anisotropic Total Variation loss on the predicted density map."""

    def forward(
        self, pred: torch.Tensor, _target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            pred:    [B, 1, H, W] predicted density map.
            _target: Ignored (accepted for uniform call signature).
        Returns:
            Scalar TV loss averaged over the batch.
        """
        # Horizontal and vertical differences
        diff_h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
        diff_w = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
        return (diff_h.sum(dim=(1, 2, 3)) + diff_w.sum(dim=(1, 2, 3))).mean()


class DMCountLoss(nn.Module):
    """Combined DM-Count density loss: L_C + λ1·L_OT + λ2·L_TV.

    Drop-in replacement for ``nn.MSELoss(reduction="sum")`` used as the
    density criterion.  Returns a scalar loss compatible with the existing
    training loop.

    Args:
        lambda_count: Weight for counting loss (default 1.0).
        lambda_ot:    Weight for OT loss (default 0.1).
        lambda_tv:    Weight for TV loss (default 0.01).
        count_loss_type: ``"l1"`` or ``"mse"`` for counting loss.
    """

    def __init__(
        self,
        lambda_count: float = 1.0,
        lambda_ot: float = 0.1,
        lambda_tv: float = 0.01,
        count_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        self.lambda_count = lambda_count
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv

        self.count_loss = CountingLoss(mode=count_loss_type)
        self.ot_loss = OTLoss()
        self.tv_loss = TVLoss()

        # Stash last-computed components for logging (detached scalars)
        self._last_components: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 1, H, W] predicted density map.
            target: [B, 1, H, W] ground-truth density map.
        Returns:
            Scalar combined loss.
        """
        l_c = self.count_loss(pred, target)
        l_ot = self.ot_loss(pred, target)
        l_tv = self.tv_loss(pred)

        total = self.lambda_count * l_c + self.lambda_ot * l_ot + self.lambda_tv * l_tv

        # Cache for metric logging (no grad overhead)
        self._last_components = {
            "den_count_loss": l_c.detach().item(),
            "den_ot_loss": l_ot.detach().item(),
            "den_tv_loss": l_tv.detach().item(),
        }

        # Multiply by batch size to match the existing MSELoss(reduction="sum")
        # convention used in engine.py (which divides by batch size afterwards).
        batch_size = pred.shape[0]
        return total * batch_size

    @property
    def last_components(self) -> dict[str, float]:
        """Return per-component losses from the most recent forward pass."""
        return self._last_components

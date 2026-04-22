"""DM-Count style density loss for RCCFormer.

Implements the three-component loss from DM-Count (Wang et al., NeurIPS 2020,
arXiv:2009.13077) as adopted by RCCFormer (arXiv:2504.04935) §3.3:

    L_count = L_C(C', C) + λ1 · L_OT(D', D) + λ2 · L_TV(D', D)

Components:
    - L_C:  Counting loss — L1 or MSE between predicted and GT total counts.
    - L_OT: Optimal Transport loss — 1D Wasserstein-1 distance on the
            H-marginal and W-marginal distributions (closed-form via CDFs;
            this is the RCCFormer simplification of the original 2D Sinkhorn).
    - L_TV: Total Variation distance between *normalised* predicted and GT
            density distributions, weighted by the predicted total mass:
                ‖μ‖₁ · ½ · ‖μ̄ − ẑ‖₁
            (DM-Count Eq. 5).  Together with L_C and L_OT this provides an
            upper bound on the true counting error.

Output convention: the combined loss is multiplied by the batch size so that
the existing engine's ``/B`` step yields the per-sample mean of the three
components (matching the ``reduction="sum"``-then-``/B`` pattern used for
MSE).  **Magnitude warning:** DMCount loss is naturally on the scale of the
crowd count (e.g. O(100)–1000 per sample on ShanghaiTech) whereas
``MSELoss(reduction="sum")/B`` is on the scale of
``H·W · (per-pixel density)²`` (e.g. O(1)–10 on the same data).  These
units are fundamentally different and cannot be reconciled by a constant
factor; users **must** retune ``cfg.density_loss_weight`` (typically
lowering it by ~100×) when switching from MSE to DMCount.
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
            [B] per-sample W1 distances.  Returns 0 for samples where both
            distributions are near-zero (empty crops).
        """
        p_sum = p.sum(dim=-1, keepdim=True)
        q_sum = q.sum(dim=-1, keepdim=True)

        # Mask out samples where both pred and GT are near-zero (empty crop):
        # OT is undefined for zero-mass distributions and would produce
        # misleading gradients.
        valid = (p_sum.squeeze(-1) > self.eps) | (q_sum.squeeze(-1) > self.eps)

        # Normalise to probability distributions
        p_norm = p / p_sum.clamp(min=self.eps)
        q_norm = q / q_sum.clamp(min=self.eps)

        # CDF = cumulative sum
        cdf_p = p_norm.cumsum(dim=-1)
        cdf_q = q_norm.cumsum(dim=-1)

        w1 = (cdf_p - cdf_q).abs().sum(dim=-1)
        # Zero out invalid samples so they contribute no gradient
        return w1 * valid.float()


class TVLoss(nn.Module):
    """DM-Count Total Variation distribution distance (Eq. 5 of the paper).

        L_TV = ‖z‖₁ · ‖ μ̄ − ẑ ‖₁

    where ``μ`` is the predicted density, ``z`` is the GT density, and
    ``μ̄``, ``ẑ`` are their L1-normalised counterparts.  This is a *distribution*
    TV (between two probability simplices), not the spatial smoothness TV used
    in image denoising.  Combined with L_C and L_OT it upper-bounds the true
    counting error (DM-Count Theorem 1).

    Implementation notes:
    * Weighting uses the **GT** mass ``‖z‖₁`` (constant w.r.t. the model), as
      in the official `cvlab-stonybrook/DM-Count` implementation.  Weighting
      by predicted mass would let the model trivially reduce the loss by
      collapsing predictions toward zero, fighting L_C.
    * The standard TV distance includes a ½ factor; following the official
      reference implementation we drop it (it is absorbed into ``lambda_tv``
      so that ``λ_tv = 0.01`` matches the published recipe).
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
            Scalar TV loss averaged over the batch.
        """
        # Flatten spatial dims → [B, N]
        p = pred.flatten(1)
        q = target.flatten(1)

        p_sum = p.sum(dim=-1, keepdim=True)
        q_sum = q.sum(dim=-1, keepdim=True)

        # Mask out samples where both pred and GT are near-zero (empty crop):
        # the normalised distributions are undefined and would inject noise.
        valid = (p_sum.squeeze(-1) > self.eps) | (q_sum.squeeze(-1) > self.eps)

        p_norm = p / p_sum.clamp(min=self.eps)
        q_norm = q / q_sum.clamp(min=self.eps)

        # ‖p̄ − q̄‖₁ (no ½ factor; absorbed into λ_tv per official DM-Count code)
        tv_dist = (p_norm - q_norm).abs().sum(dim=-1)

        # Weight by GT total mass ‖z‖₁ (constant w.r.t. the model)
        per_sample = q_sum.squeeze(-1).detach() * tv_dist
        per_sample = per_sample * valid.float()
        return per_sample.mean()


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
        l_tv = self.tv_loss(pred, target)

        total = self.lambda_count * l_c + self.lambda_ot * l_ot + self.lambda_tv * l_tv

        # Cache for metric logging (no grad overhead)
        self._last_components = {
            "den_count_loss": l_c.detach().item(),
            "den_ot_loss": l_ot.detach().item(),
            "den_tv_loss": l_tv.detach().item(),
        }

        # Multiply by batch size to undo engine.py's subsequent ``/B`` step,
        # leaving the per-sample mean of the three components.  Note that
        # this is NOT magnitude-equivalent to ``MSELoss(reduction="sum")``
        # -- see the module docstring for the magnitude caveat.
        batch_size = pred.shape[0]
        return total * batch_size

    @property
    def last_components(self) -> dict[str, float]:
        """Return per-component losses from the most recent forward pass."""
        return self._last_components

"""MDS-Loss: Multi-Domain Structural Loss for crowd density maps.

A drop-in density criterion combining three complementary objectives:

    L_MDS = λ1 · L_ACPL  +  r(epoch) · ( λ2 · L_GSF + λ3 · L_OT )

Components:
    - **L_ACPL** (Adaptive Count-Pixel Loss): per-sample count error +
      density-aware spatially weighted Smooth-L1.  The adaptive weight is
      computed against the *per-sample* GT max so it is well-behaved for
      Gaussian-blurred density maps (which typically peak at <0.1).  The
      Huber transition β is set per batch from the GT median absolute
      deviation (detached) so the loss adapts to dataset density scale.
    - **L_GSF** (Gradient-Structure Fidelity): (1 − SSIM) plus a normalised
      Sobel-magnitude L1.  SSIM uses the project's :class:`SSIMLoss`
      (auto data_range, Gaussian window).  Sobel magnitudes are normalised
      by per-sample GT std so the term is dimensionless and stable across
      crowd densities.
    - **L_OT** (Optimal Transport, marginal): 1-D Wasserstein-1 distance on
      the H- and W-marginals of the density maps, computed via the proven
      CDF-L1 implementation in :class:`~crowdcount.plugins.dm_count_loss.OTLoss`
      (handles empty crops via valid mask).

Engineering contract: the forward returns ``L_MDS * batch_size`` so that
``engine.py``'s subsequent ``/B`` and ``* density_loss_weight`` give the
per-sample mean, matching the convention used by ASACL / DM-Count.

Compared to plain MSE the magnitude of MDS-Loss is on the order of the
crowd count, so users **must** retune ``cfg.density_loss_weight`` (typical
range 0.1–1.0 instead of 0.01).

References:
    - Wang et al., "Image Quality Assessment: From Error Visibility to SSIM",
      IEEE TIP 2004.
    - Wang et al., "Distribution Matching for Crowd Counting", NeurIPS 2020.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.ssim_loss import SSIMLoss
from crowdcount.plugins.dm_count_loss import OTLoss


# ---------------------------------------------------------------------------
# Component 1: Adaptive Count-Pixel Loss
# ---------------------------------------------------------------------------


class ACPLComponent(nn.Module):
    """Adaptive count + density-weighted Smooth-L1 loss.

    Args:
        alpha: Strength of the spatial weighting; effective weight is
            ``1 + alpha * gt / gt.amax(per-sample)``.  Higher → stronger
            emphasis on dense regions.
        weight_cap: Hard upper bound on the spatial weight.
        huber_delta_floor: Lower bound for the auto-derived Smooth-L1 β,
            preventing the loss from collapsing to L2 on near-empty batches.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        weight_cap: float = 10.0,
        huber_delta_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.weight_cap = float(weight_cap)
        self.huber_delta_floor = float(huber_delta_floor)

    @staticmethod
    def _adaptive_delta(target: torch.Tensor, floor: float) -> float:
        """MAD of **non-background** GT pixels, detached and clamped.

        Using the full tensor is wrong for Gaussian-blurred density maps:
        >90 % of pixels are at or near 0, so the full-tensor MAD is always
        ~0 and the clamp returns the floor value every time.
        Computing on non-zero pixels gives a meaningful adaptive δ.
        """
        with torch.no_grad():
            flat = target.detach().flatten()
            nz = flat[flat > 1e-7]
            if nz.numel() < 2:
                return floor
            med = nz.median()
            mad = (nz - med).abs().median()
            return float(max(mad.item(), floor))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Per-sample count error, normalised to per-pixel density units.
        # count = integral(density) = sum of all pixel values.  Dividing by
        # n_pixels converts the integral constraint to a mean per-pixel error,
        # making it commensurable with pixel_err (also a mean over pixels).
        # Without this, count_err ≈ O(100–1000) while pixel_err ≈ O(0.001–1)
        # and the spatial term contributes essentially nothing.
        pred_count = pred.sum(dim=(1, 2, 3))  # [B]
        gt_count = target.sum(dim=(1, 2, 3))  # [B]
        n_pixels = float(pred.shape[-1] * pred.shape[-2])
        count_err = (pred_count - gt_count).abs().mean() / n_pixels

        # Per-sample max for normalised spatial weight ---------------------
        b = target.shape[0]
        gt_max = target.detach().reshape(b, -1).amax(dim=1).clamp(min=1e-6)  # [B]
        gt_max = gt_max.view(b, 1, 1, 1)
        weight = 1.0 + self.alpha * (target.detach() / gt_max)
        weight = weight.clamp(max=self.weight_cap)

        # Smooth-L1 with adaptive β ----------------------------------------
        delta = self._adaptive_delta(target, self.huber_delta_floor)
        per_pix = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
        # Weighted mean (not arithmetic mean): the spatial weight only has
        # interpretable effect when the normalisation matches Σw.  With plain
        # ``.mean()`` the background dominates the denominator (256 px vs ~5
        # foreground px on 16×16), diluting the per-pixel weighting boost.
        pixel_err = (weight * per_pix).sum() / weight.sum().clamp(min=1.0)

        return count_err + pixel_err


# ---------------------------------------------------------------------------
# Component 2: Gradient-Structure Fidelity
# ---------------------------------------------------------------------------


class GradientFidelityComponent(nn.Module):
    """SSIM + per-sample-normalised Sobel-magnitude L1.

    SSIM is delegated to :class:`SSIMLoss` so behaviour is consistent with
    the ``density_ssim`` configuration block.  The Sobel term provides a
    first-order edge / cluster-boundary penalty that is complementary to
    SSIM (which is dominated by second-order statistics) and avoids the
    quantitative degeneracy of a raw Laplacian on density maps with very
    small dynamic range.
    """

    def __init__(
        self,
        ssim_window_size: int = 7,
        ssim_sigma: float = 1.5,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.ssim = SSIMLoss(
            window_size=int(ssim_window_size),
            sigma=float(ssim_sigma),
            data_range=None,
        )

        # 3x3 Sobel kernels (registered as buffers → moved with .to(device))
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude with reflect padding to suppress edges."""
        # Pad reflectively to avoid zero-padding artefacts at the borders
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ---- SSIM with per-sample JOINT-max normalisation --------------
        # Project density maps to [0, ~1] using the per-sample max of
        # ``max(pred, target)`` before SSIM.  Two requirements drive this:
        # (a) ``SSIMLoss`` clamps its auto data-range to ``≥ 1.0`` while
        #     density values are O(0.01–0.05), so ``C1 = (0.01·1)² = 1e-4``
        #     would dwarf the local variance and saturate the map.
        # (b) Normalising only by the **target** max is unsafe early in
        #     training when ``pred`` can be 100× larger than ``target``:
        #     after dividing pred by 0.05 it reaches values of 60+, sending
        #     SSIM into its noisy regime and producing destructive gradients
        #     that push pred *away* from target (verified empirically).
        # Using the joint max keeps both inputs in [0, 1] regardless of how
        # far off pred is, so SSIM behaves as a well-defined structural
        # similarity measure.  ``dr`` is detached → only structural error
        # contributes to gradients, not the normaliser.
        b = pred.shape[0]
        joint = torch.maximum(pred.detach().abs(), target.detach()).reshape(b, -1)
        dr = joint.amax(dim=1).clamp(min=1e-6).view(b, 1, 1, 1)
        l_ssim = self.ssim(pred / dr, target / dr)

        # ---- Sobel L1, plain (no std normalisation) --------------------
        # Earlier revisions divided the Sobel magnitude by the GT std, but
        # for sparse density maps ``std ≈ 5e-3``; dividing inflates gradient
        # magnitudes by ~200×, completely overwhelming the count term and
        # causing pred to *diverge* from target during optimisation
        # (verified: GSF-only training pushes count from 100 → 130 instead
        # of toward the GT value of 0.25).  Density values are already in
        # a known small dynamic range, so the raw Sobel L1 is well-scaled
        # and stable.  An empty-crop mask is still applied so all-zero GT
        # samples don't pull pred toward arbitrary edges.
        s_gt = target.detach().reshape(b, -1).std(dim=1, unbiased=False)
        valid = (s_gt > 1e-4).view(b, 1, 1, 1).float()  # [B,1,1,1]
        grad_p = self._grad_mag(pred)
        grad_g = self._grad_mag(target)
        diff = (grad_p - grad_g).abs() * valid
        # Per-pixel mean over valid samples only (avoids dilution by empties)
        denom = valid.sum() * grad_p.shape[-1] * grad_p.shape[-2]
        l_grad = diff.sum() / denom.clamp(min=1.0)

        return l_ssim + self.beta * l_grad


# ---------------------------------------------------------------------------
# Composite MDS-Loss
# ---------------------------------------------------------------------------


class MDSLoss(nn.Module):
    """Multi-Domain Structural Loss for density maps.

    Args:
        lambda_acpl: Weight for the count-pixel term.
        lambda_gsf:  Weight for the SSIM + gradient term (ramped via warmup).
        lambda_ot:   Weight for the marginal OT term (ramped via warmup).
        alpha:       ACPL spatial-weight strength.
        beta:        GSF Sobel-term weight.
        huber_delta_floor: Lower bound for the auto-derived Smooth-L1 β.
        ssim_window_size: SSIM Gaussian window; must be odd; ≤7 for 16×16
            density maps used in this project.
        ssim_sigma:  SSIM Gaussian sigma.
        warmup_epochs: Number of epochs over which ``λ_gsf`` and ``λ_ot``
            are linearly ramped from 0 to their target value.  ``0``
            disables warmup.
    """

    def __init__(
        self,
        lambda_acpl: float = 1.0,
        lambda_gsf: float = 0.5,
        lambda_ot: float = 0.3,
        alpha: float = 4.0,
        beta: float = 1.0,
        huber_delta_floor: float = 1e-3,
        ssim_window_size: int = 7,
        ssim_sigma: float = 1.5,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.lambda_acpl = float(lambda_acpl)
        self.lambda_gsf = float(lambda_gsf)
        self.lambda_ot = float(lambda_ot)
        self.warmup_epochs = int(max(warmup_epochs, 0))

        self.acpl = ACPLComponent(
            alpha=alpha,
            huber_delta_floor=huber_delta_floor,
        )
        self.gsf = GradientFidelityComponent(
            ssim_window_size=ssim_window_size,
            ssim_sigma=ssim_sigma,
            beta=beta,
        )
        self.ot = OTLoss()

        # Current epoch index (set by Trainer via set_epoch())
        self._current_epoch: int = 0
        self._last_components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Inject the current epoch so the warmup ramp can advance."""
        self._current_epoch = int(epoch)

    def warmup_factor(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return float(min(1.0, max(self._current_epoch, 0) / self.warmup_epochs))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shapes, got {pred.shape} and {target.shape}"
            )
        if pred.dim() != 4:
            raise ValueError(f"expected 4D tensors [B,1,H,W], got {pred.dim()}D")

        l_acpl = self.acpl(pred, target)
        l_gsf = self.gsf(pred, target)
        # ---- OT-M on per-sample COUNT-NORMALISED maps ------------------
        # The marginal 1-D Wasserstein in ``OTLoss`` is responsible for the
        # *spatial distribution* of mass, not its magnitude.  If we feed it
        # the raw density it conflates count error and spatial error, and
        # since its gradient on row/col CDFs is rank-deficient (translation
        # invariant), it can move pred mass in ways that *increase* count
        # error (verified: OT-only optimisation diverges).  Normalising
        # each sample to a probability distribution before the call lets
        # ACPL handle count and OT-M handle layout, which composes well.
        # Negative pred is also clamped (CDF requires non-negative mass).
        eps = 1e-6
        pred_pos = pred.clamp(min=0.0)
        b = pred.shape[0]
        p_sum = (
            pred_pos.detach().reshape(b, -1).sum(dim=1).clamp(min=eps).view(b, 1, 1, 1)
        )
        t_sum = (
            target.detach().reshape(b, -1).sum(dim=1).clamp(min=eps).view(b, 1, 1, 1)
        )
        # Mask samples with empty GT (would otherwise produce undefined OT)
        ot_valid = (target.detach().reshape(b, -1).sum(dim=1) > eps).float()
        if ot_valid.sum() > 0:
            l_ot = self.ot(pred_pos / p_sum, target / t_sum)
        else:
            l_ot = pred.sum() * 0.0  # zero with grad path

        ramp = self.warmup_factor()
        total = (
            self.lambda_acpl * l_acpl
            + ramp * self.lambda_gsf * l_gsf
            + ramp * self.lambda_ot * l_ot
        )

        # Cache components for metric logging (detached scalars)
        self._last_components = {
            "den_mds_acpl": float(l_acpl.detach().item()),
            "den_mds_gsf": float(l_gsf.detach().item()),
            "den_mds_ot": float(l_ot.detach().item()),
            "den_mds_warmup": ramp,
        }

        # Multiply by batch size so engine.py's subsequent /B yields the
        # per-sample mean (matches ASACL / DM-Count convention).
        return total * pred.shape[0]

    @property
    def last_components(self) -> dict[str, float]:
        """Return per-component losses from the most recent forward pass."""
        return self._last_components

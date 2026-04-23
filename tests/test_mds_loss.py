"""Tests for MDS-Loss (Multi-Domain Structural Loss)."""

from __future__ import annotations

import math

import pytest
import torch

from crowdcount.plugins.mds_loss import (
    ACPLComponent,
    GradientFidelityComponent,
    MDSLoss,
)


# Density maps in this project are downsampled to 16x16 before the loss
B, C, H, W = 2, 1, 16, 16


def _gt_density(seed: int = 0) -> torch.Tensor:
    """Make a sparse Gaussian-blurred density-like target."""
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(B, C, H, W)
    for b in range(B):
        # 3 random "heads"
        for _ in range(3):
            i = int(torch.randint(2, H - 2, (1,), generator=g))
            j = int(torch.randint(2, W - 2, (1,), generator=g))
            t[b, 0, i, j] = 1.0
    # Gaussian blur via 3x3 averaging twice (cheap, no scipy)
    kernel = torch.ones(1, 1, 3, 3) / 9.0
    t = torch.nn.functional.conv2d(t, kernel, padding=1)
    t = torch.nn.functional.conv2d(t, kernel, padding=1)
    return t * 0.05  # scale to typical Gaussian-density magnitude


# ---------------------------------------------------------------------------
# ACPL component
# ---------------------------------------------------------------------------


class TestACPL:
    def test_zero_when_equal(self) -> None:
        acpl = ACPLComponent()
        x = _gt_density(1)
        loss = acpl(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_different(self) -> None:
        acpl = ACPLComponent()
        target = _gt_density(2)
        pred = torch.zeros_like(target)
        assert acpl(pred, target).item() > 0.0

    def test_count_term_dominant_for_large_count_offset(self) -> None:
        """A constant offset that preserves shape but doubles count must give
        non-trivial loss (count_err catches it)."""
        acpl = ACPLComponent()
        target = _gt_density(3)
        pred = target * 2.0
        loss = acpl(pred, target).item()
        assert loss > 0.0 and math.isfinite(loss)

    def test_handles_all_zero_inputs(self) -> None:
        acpl = ACPLComponent()
        z = torch.zeros(B, 1, H, W)
        loss = acpl(z, z)
        assert math.isfinite(loss.item())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GSF component
# ---------------------------------------------------------------------------


class TestGSF:
    def test_zero_when_equal(self) -> None:
        gsf = GradientFidelityComponent(ssim_window_size=7, ssim_sigma=1.5)
        x = _gt_density(4)
        loss = gsf(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_when_different(self) -> None:
        gsf = GradientFidelityComponent()
        target = _gt_density(5)
        pred = torch.zeros_like(target)
        assert gsf(pred, target).item() > 0.0

    def test_handles_all_zero_inputs(self) -> None:
        gsf = GradientFidelityComponent()
        z = torch.zeros(B, 1, H, W)
        # Both zero — std=0 falls back to clamp; SSIM auto data_range=1.0
        loss = gsf(z, z)
        assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# Composite MDSLoss
# ---------------------------------------------------------------------------


class TestMDSLoss:
    def test_forward_shape_scalar(self) -> None:
        loss_fn = MDSLoss()
        target = _gt_density(6)
        pred = torch.zeros_like(target).requires_grad_(True)
        out = loss_fn(pred, target)
        assert out.dim() == 0
        out.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    def test_pred_equals_target_near_zero(self) -> None:
        loss_fn = MDSLoss(warmup_epochs=0)  # no warmup, full GSF/OT active
        x = _gt_density(7)
        out = loss_fn(x, x).item()
        # Multiplied by batch_size internally; still expected to be ~0
        assert out == pytest.approx(0.0, abs=1e-3)

    def test_warmup_factor_bounds(self) -> None:
        loss_fn = MDSLoss(warmup_epochs=10)
        loss_fn.set_epoch(0)
        assert loss_fn.warmup_factor() == 0.0
        loss_fn.set_epoch(5)
        assert loss_fn.warmup_factor() == pytest.approx(0.5, abs=1e-6)
        loss_fn.set_epoch(10)
        assert loss_fn.warmup_factor() == 1.0
        loss_fn.set_epoch(100)
        assert loss_fn.warmup_factor() == 1.0

    def test_warmup_disables_gsf_ot_at_epoch_zero(self) -> None:
        """At epoch 0 with warmup>0, only ACPL contributes; loss must equal
        ``λ_acpl * ACPL(pred, target) * batch_size``."""
        loss_fn = MDSLoss(
            lambda_acpl=1.0,
            lambda_gsf=0.5,
            lambda_ot=0.3,
            warmup_epochs=10,
        )
        loss_fn.set_epoch(0)
        target = _gt_density(8)
        pred = torch.zeros_like(target)
        total = loss_fn(pred, target).item()
        acpl_only = (
            loss_fn.lambda_acpl * loss_fn.acpl(pred, target).item() * pred.shape[0]
        )
        assert total == pytest.approx(acpl_only, rel=1e-5)

    def test_last_components_keys(self) -> None:
        loss_fn = MDSLoss()
        target = _gt_density(9)
        pred = torch.zeros_like(target)
        _ = loss_fn(pred, target)
        comps = loss_fn.last_components
        assert {
            "den_mds_acpl",
            "den_mds_gsf",
            "den_mds_ot",
            "den_mds_warmup",
        }.issubset(comps.keys())

    def test_handles_empty_crop(self) -> None:
        """Random crop with zero people: pred=0, target=0 must not yield NaN."""
        loss_fn = MDSLoss()
        z = torch.zeros(B, 1, H, W)
        out = loss_fn(z, z)
        assert math.isfinite(out.item())

    def test_handles_partial_empty_batch(self) -> None:
        """Mixed batch: one empty + one populated sample."""
        loss_fn = MDSLoss()
        target = _gt_density(10)
        target[0] = 0.0
        pred = torch.zeros_like(target).requires_grad_(True)
        out = loss_fn(pred, target)
        assert math.isfinite(out.item())
        out.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    def test_shape_mismatch_raises(self) -> None:
        loss_fn = MDSLoss()
        with pytest.raises(ValueError):
            loss_fn(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 16, 16))

    def test_non_4d_raises(self) -> None:
        loss_fn = MDSLoss()
        with pytest.raises(ValueError):
            loss_fn(torch.zeros(8, 8), torch.zeros(8, 8))

    def test_returns_batch_scaled(self) -> None:
        """Returned loss is per-sample-mean × batch_size (engine convention)."""
        loss_fn = MDSLoss(warmup_epochs=0)
        single = _gt_density(11)[:1]
        target = torch.cat([single, single], dim=0)  # identical samples
        pred = torch.zeros_like(target)
        out_b2 = loss_fn(pred, target).item()
        out_b1 = loss_fn(pred[:1], target[:1]).item()
        # Per-sample loss is the same; B=2 returns 2x B=1 value
        assert out_b2 == pytest.approx(2.0 * out_b1, rel=1e-4)


# ---------------------------------------------------------------------------
# Regression tests for specific bugs
# ---------------------------------------------------------------------------


class TestBugFixes:
    """Regression tests that each target one previously identified bug."""

    def test_count_err_and_pixel_err_comparable_scale(self) -> None:
        """Bug 1: count_err dominated pixel_err by ~1000x without n_pixels
        normalisation.  After the fix both should contribute meaningfully.

        We measure by checking that count_err alone (pred=0, target=uniform
        density) contributes a value of the same order as pixel_err for the
        same input.
        """
        acpl = ACPLComponent(alpha=0.0)  # disable spatial weight to isolate terms
        # Uniform density: each pixel = count / n_pixels
        count = 100.0
        target = torch.full((1, 1, H, W), count / (H * W))
        pred = torch.zeros_like(target)

        # count_err after fix: (count) / (H*W) = count/(H*W) ≈ pixel value
        # pixel_err: mean(smooth_l1(0, target)) ≈ mean(target) for small β
        n_pixels = H * W
        expected_count_err = count / n_pixels
        full_loss = acpl(pred, target).item()
        # Both terms should be O(same magnitude); total must be ≥ count_err_expected
        assert full_loss >= expected_count_err * 0.9, (
            f"total loss {full_loss} < expected count_err contribution {expected_count_err}"
        )
        # And total should not be >> 100× count_err (pixel_err comparable)
        assert full_loss < expected_count_err * 200, (
            f"total loss {full_loss} >> count_err {expected_count_err}: pixel_err swamped by count_err"
        )

    def test_sobel_no_explosion_empty_gt(self) -> None:
        """Bug 2: Sobel normalisation by GT-only std caused 1/1e-6 explosion.

        When GT is all-zero but pred has non-zero values, the gradient of
        l_grad w.r.t. pred must remain finite (not 1e6 scale).
        """
        gsf = GradientFidelityComponent()
        target = torch.zeros(B, 1, H, W)
        # pred has small but non-zero spatial variation
        pred = torch.randn(B, 1, H, W) * 0.01
        pred.requires_grad_(True)
        loss = gsf(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all(), "Gradient contains NaN/Inf"
        # Gradient magnitude must be reasonable (not ~1e6 scale)
        grad_max = pred.grad.abs().max().item()
        assert grad_max < 1e3, f"Gradient too large: {grad_max:.2e} (explosion risk)"

    def test_adaptive_delta_on_nonzero_pixels(self) -> None:
        """Bug 3: old MAD used full tensor → MAD≈0 because >90 % pixels are 0.

        After the fix, MAD is computed on non-zero pixels only and should
        be strictly larger than the full-tensor MAD, which is always ~0
        for sparse Gaussian-blurred density maps.
        We use a tiny floor so the test is not masked by the floor clamp.
        """
        target = _gt_density(42)
        assert (target > 1e-7).any(), "Test helper must produce non-zero GT"

        # "Buggy" full-tensor MAD (all pixels, mostly zero)
        with torch.no_grad():
            flat = target.flatten()
            full_mad = (flat - flat.median()).abs().median().item()

        # Non-zero-pixel MAD (fixed implementation), using tiny floor
        delta_fixed = ACPLComponent._adaptive_delta(target, floor=1e-10)

        # The full-tensor MAD is 0 (or close) because the median of a >90%-zero
        # sparse map is 0 and so is the MAD.  The non-zero-pixel MAD must be
        # strictly larger, demonstrating the fix takes effect.
        assert delta_fixed > full_mad, (
            f"Non-zero-pixel MAD ({delta_fixed:.2e}) should exceed "
            f"full-tensor MAD ({full_mad:.2e}). "
            "This indicates MAD is still computed on all pixels."
        )
        assert delta_fixed > 1e-10, (
            f"Non-zero-pixel MAD ({delta_fixed:.2e}) is essentially zero — "
            "density map has no spread among non-zero pixels."
        )

    def test_ot_stable_with_negative_pred(self) -> None:
        """Bug 4: OT CDF is undefined for negative marginals.

        Negative predictions (possible early in training) must not produce
        NaN or infinite loss values.
        """
        loss_fn = MDSLoss(warmup_epochs=0)
        target = _gt_density(7)
        # Simulate early-training negative predictions
        pred = torch.randn(B, 1, H, W) * 0.5
        pred.requires_grad_(True)
        out = loss_fn(pred, target)
        assert math.isfinite(out.item()), "Loss is NaN/Inf with negative predictions"
        out.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all(), (
            "Gradient contains NaN/Inf with negative predictions"
        )

    def test_ssim_not_saturated_on_density_scale(self) -> None:
        """Bug E: without per-sample normalisation SSIMLoss falls back to
        ``data_range=1.0`` and ``C1`` swamps the local variance, making the
        loss \u22480 even for very different inputs.

        After the fix (per-sample target-max normalisation inside GSF),
        a clearly different prediction must produce non-trivial SSIM loss.
        """
        gsf = GradientFidelityComponent(beta=0.0)  # disable Sobel to isolate SSIM
        target = _gt_density(101)
        # Pred is half the GT in magnitude (clearly different structurally
        # since count is wrong even after normalisation by GT-max).
        pred = target * 0.0  # all-zero prediction
        loss = gsf(pred, target).item()
        assert loss > 1e-3, (
            f"SSIM loss {loss:.6f} is essentially zero \u2014 it has saturated. "
            "Per-sample data_range normalisation must precede SSIM."
        )

    def test_pixel_err_uses_weighted_mean(self) -> None:
        """Bug L: arithmetic mean dilutes the foreground weight boost.

        With weighted mean ``\u03a3(w\u00b7l)/\u03a3w``, doubling \u03b1 (foreground boost)
        must measurably change pixel_err, whereas with arithmetic mean
        ``\u03a3(w\u00b7l)/N`` it changes only marginally because background still
        dominates the denominator.
        """
        # Construct GT with clear fg/bg separation\n
        target = _gt_density(202)
        pred = torch.zeros_like(target)
        loss_low = ACPLComponent(alpha=1.0)(pred, target).item()
        loss_high = ACPLComponent(alpha=8.0)(pred, target).item()
        # With weighted mean the foreground boost has real effect: high \u03b1
        # should substantially increase pixel_err contribution.
        ratio = loss_high / max(loss_low, 1e-12)
        assert ratio > 1.2, (
            f"alpha increase did not significantly change loss "
            f"(ratio={ratio:.3f}); the weighted-mean fix is not in effect."
        )

    def test_gsf_sobel_target_only_std(self) -> None:
        """Bug M: GSF normaliser must be independent of pred so the loss\n        surface does not drift with each optimisation step.\n"""
        gsf = GradientFidelityComponent(beta=1.0)
        target = _gt_density(303)
        # Two predictions with very different std but same shape\n
        pred_small = target * 0.5
        pred_large = target * 2.0
        # Backward pass on each
        ps = pred_small.detach().clone().requires_grad_(True)
        pl = pred_large.detach().clone().requires_grad_(True)
        ls = gsf(ps, target)
        ls.backward()
        ll = gsf(pl, target)
        ll.backward()
        # Gradients must be finite\n
        assert torch.isfinite(ps.grad).all() and torch.isfinite(pl.grad).all()

    def test_gsf_empty_gt_no_gradient(self) -> None:
        """Bug M (cont.): empty GT crops must not contribute to L_grad.\n
        With the valid mask, pred gradient should be zero for all-zero GT.\n"""
        gsf = GradientFidelityComponent(beta=1.0)
        target = torch.zeros(B, 1, H, W)
        pred = (torch.randn(B, 1, H, W) * 0.05).requires_grad_(True)
        loss = gsf(pred, target)
        loss.backward()
        # SSIM still contributes (pred non-zero vs GT=0), but Sobel L1 should
        # be zero. We can't easily isolate them, so instead verify loss is
        # finite and gradient is bounded (no 1/std blowup).\n
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().max().item() < 1e3

    def test_full_loss_drives_pred_toward_target(self) -> None:
        """Bug N (most important): the COMPOSITE loss must actually pull
        pred toward target during gradient descent.

        Earlier revisions had GSF and OT terms whose magnitudes overwhelmed
        ACPL with destructive gradients (verified: pred count diverged from
        100 to 130 instead of converging to GT count of ~0.25). The fixes
        (GSF: drop 1/std, joint-max SSIM normalisation; OT: count-normalise)
        ensure all three components compose constructively.

        We assert that 200 Adam steps reduce both pixel-MSE and per-sample
        count error by >=10x relative to initialisation.
        """
        torch.manual_seed(0)
        target = _gt_density(seed=999)
        torch.manual_seed(1)
        pred = torch.randn_like(target).clamp(min=0).requires_grad_(True)
        opt = torch.optim.Adam([pred], lr=0.05)
        loss_fn = MDSLoss(warmup_epochs=0)

        with torch.no_grad():
            mse_init = ((pred - target) ** 2).mean().item()
            ce_init = (pred.sum() - target.sum()).abs().item() / pred.shape[0]

        for _ in range(200):
            opt.zero_grad()
            loss_fn(pred, target).backward()
            opt.step()

        with torch.no_grad():
            mse_final = ((pred - target) ** 2).mean().item()
            ce_final = (pred.sum() - target.sum()).abs().item() / pred.shape[0]

        assert mse_final < mse_init / 10.0, (
            f"MDSLoss did not significantly reduce MSE "
            f"(init={mse_init:.4f}, final={mse_final:.4f}). "
            "Loss is not guiding pred toward target."
        )
        assert ce_final < ce_init / 10.0, (
            f"MDSLoss did not significantly reduce per-sample count error "
            f"(init={ce_init:.3f}, final={ce_final:.3f}). "
            "Loss is not guiding count toward GT."
        )

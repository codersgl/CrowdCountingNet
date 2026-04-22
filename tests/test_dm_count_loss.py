"""Tests for DM-Count style density loss."""

from __future__ import annotations

import torch
import pytest

from crowdcount.plugins.dm_count_loss import (
    CountingLoss,
    DMCountLoss,
    OTLoss,
    TVLoss,
)


# ---------------------------------------------------------------------------
# CountingLoss
# ---------------------------------------------------------------------------


class TestCountingLoss:
    def test_l1_zero_when_equal(self) -> None:
        loss_fn = CountingLoss(mode="l1")
        x = torch.rand(2, 1, 8, 8)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_zero_when_equal(self) -> None:
        loss_fn = CountingLoss(mode="mse")
        x = torch.rand(2, 1, 8, 8)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_l1_value(self) -> None:
        loss_fn = CountingLoss(mode="l1")
        pred = torch.ones(1, 1, 4, 4)  # sum = 16
        gt = torch.zeros(1, 1, 4, 4)  # sum = 0
        assert loss_fn(pred, gt).item() == pytest.approx(16.0, abs=1e-5)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="count_loss_type"):
            CountingLoss(mode="huber")


# ---------------------------------------------------------------------------
# OTLoss
# ---------------------------------------------------------------------------


class TestOTLoss:
    def test_zero_for_identical_distributions(self) -> None:
        loss_fn = OTLoss()
        x = torch.rand(2, 1, 8, 8).clamp(min=0.01)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different_distributions(self) -> None:
        loss_fn = OTLoss()
        pred = torch.zeros(1, 1, 8, 8)
        pred[0, 0, 0, 0] = 1.0  # all mass top-left
        gt = torch.zeros(1, 1, 8, 8)
        gt[0, 0, 7, 7] = 1.0  # all mass bottom-right
        assert loss_fn(pred, gt).item() > 0.0

    def test_symmetric(self) -> None:
        loss_fn = OTLoss()
        a = torch.rand(2, 1, 8, 8)
        b = torch.rand(2, 1, 8, 8)
        assert loss_fn(a, b).item() == pytest.approx(loss_fn(b, a).item(), abs=1e-6)

    def test_handles_zero_maps(self) -> None:
        """All-zero maps should not crash (eps protects division)."""
        loss_fn = OTLoss()
        z = torch.zeros(1, 1, 4, 4)
        loss_fn(z, z)  # should not raise


# ---------------------------------------------------------------------------
# TVLoss
# ---------------------------------------------------------------------------


class TestTVLoss:
    def test_zero_for_identical_distributions(self) -> None:
        loss_fn = TVLoss()
        x = torch.rand(2, 1, 8, 8).clamp(min=0.01)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different_distributions(self) -> None:
        loss_fn = TVLoss()
        pred = torch.zeros(1, 1, 8, 8)
        pred[0, 0, 0, 0] = 1.0  # all mass top-left
        gt = torch.zeros(1, 1, 8, 8)
        gt[0, 0, 7, 7] = 1.0  # all mass bottom-right
        # Two disjoint normalised diracs → ‖p̄−q̄‖₁ = 2; ‖z‖₁ = 1 → 2.0
        assert loss_fn(pred, gt).item() == pytest.approx(2.0, abs=1e-5)

    def test_handles_zero_maps(self) -> None:
        """All-zero maps must not crash and should yield zero loss."""
        loss_fn = TVLoss()
        z = torch.zeros(2, 1, 4, 4)
        assert loss_fn(z, z).item() == pytest.approx(0.0, abs=1e-6)

    def test_weighted_by_gt_mass(self) -> None:
        """L_TV scales linearly with GT total mass ‖z‖₁."""
        loss_fn = TVLoss()
        pred = torch.zeros(1, 1, 4, 4)
        pred[0, 0, 0, 0] = 1.0
        gt_1x = torch.zeros(1, 1, 4, 4)
        gt_1x[0, 0, 3, 3] = 1.0
        gt_3x = gt_1x * 3.0  # same shape, 3x mass
        loss_1x = loss_fn(pred, gt_1x).item()
        loss_3x = loss_fn(pred, gt_3x).item()
        assert loss_3x == pytest.approx(3.0 * loss_1x, abs=1e-5)

    def test_invariant_to_pred_scale(self) -> None:
        """Loss must NOT change when only the predicted mass is rescaled.

        This guards against the trivial-collapse failure mode: weighting by
        predicted mass would let the model reduce L_TV by shrinking
        predictions to zero.
        """
        loss_fn = TVLoss()
        pred = torch.zeros(1, 1, 4, 4)
        pred[0, 0, 0, 0] = 1.0
        gt = torch.zeros(1, 1, 4, 4)
        gt[0, 0, 3, 3] = 1.0
        loss_a = loss_fn(pred, gt).item()
        loss_b = loss_fn(pred * 0.01, gt).item()  # predict 100x weaker
        assert loss_a == pytest.approx(loss_b, abs=1e-5)


# ---------------------------------------------------------------------------
# DMCountLoss (combined)
# ---------------------------------------------------------------------------


class TestDMCountLoss:
    def test_output_is_scalar(self) -> None:
        loss_fn = DMCountLoss()
        pred = torch.rand(2, 1, 8, 8)
        gt = torch.rand(2, 1, 8, 8)
        loss = loss_fn(pred, gt)
        assert loss.dim() == 0

    def test_gradient_flows(self) -> None:
        loss_fn = DMCountLoss()
        pred = torch.rand(2, 1, 8, 8, requires_grad=True)
        gt = torch.rand(2, 1, 8, 8)
        loss = loss_fn(pred, gt)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_last_components_populated(self) -> None:
        loss_fn = DMCountLoss()
        pred = torch.rand(2, 1, 8, 8)
        gt = torch.rand(2, 1, 8, 8)
        loss_fn(pred, gt)
        comp = loss_fn.last_components
        assert "den_count_loss" in comp
        assert "den_ot_loss" in comp
        assert "den_tv_loss" in comp

    def test_batch_size_scaling(self) -> None:
        """Loss should scale with batch size (matching MSELoss sum convention)."""
        loss_fn = DMCountLoss(lambda_count=1.0, lambda_ot=0.0, lambda_tv=0.0)
        pred = torch.ones(1, 1, 4, 4)
        gt = torch.zeros(1, 1, 4, 4)
        loss_b1 = loss_fn(pred, gt).item()

        pred2 = torch.ones(2, 1, 4, 4)
        gt2 = torch.zeros(2, 1, 4, 4)
        loss_b2 = loss_fn(pred2, gt2).item()
        # After engine.py divides by batch_size, per-sample cost should match
        assert loss_b1 / 1 == pytest.approx(loss_b2 / 2, abs=1e-5)

    def test_magnitude_is_count_scale(self) -> None:
        """DMCount loss magnitude is on the count scale (NOT MSE-sum scale).

        This documents the units of the loss: with λ_count=1 and pred=0, the
        per-sample loss equals the GT count.  Users must retune
        ``density_loss_weight`` accordingly when switching from MSE.
        """
        loss_fn = DMCountLoss(lambda_count=1.0, lambda_ot=0.0, lambda_tv=0.0)
        b, h, w = 2, 16, 16
        pred = torch.zeros(b, 1, h, w)
        gt = torch.full((b, 1, h, w), 2.0)  # per-sample count = 2 * 16 * 16 = 512
        # forward returns total * B; engine then /B → total. Per-sample count
        # error = 512, batch mean = 512, * B = 1024.
        per_sample_count = 2.0 * h * w
        assert loss_fn(pred, gt).item() == pytest.approx(per_sample_count * b, abs=1e-3)

    def test_zero_weights_disable_components(self) -> None:
        loss_fn = DMCountLoss(lambda_count=0.0, lambda_ot=0.0, lambda_tv=0.0)
        pred = torch.rand(2, 1, 8, 8)
        gt = torch.rand(2, 1, 8, 8)
        assert loss_fn(pred, gt).item() == pytest.approx(0.0, abs=1e-6)

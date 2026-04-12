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
    def test_zero_for_constant_map(self) -> None:
        loss_fn = TVLoss()
        x = torch.ones(2, 1, 8, 8) * 5.0
        assert loss_fn(x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_varying_map(self) -> None:
        loss_fn = TVLoss()
        x = torch.randn(2, 1, 8, 8)
        assert loss_fn(x).item() > 0.0

    def test_known_value(self) -> None:
        """Simple 1x1x2x2 checkerboard."""
        loss_fn = TVLoss()
        x = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])  # [1,1,2,2]
        # h-diffs: |1-0|+|0-1| = 2; w-diffs: |1-0|+|0-1| = 2; total = 4
        assert loss_fn(x).item() == pytest.approx(4.0, abs=1e-5)


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

    def test_zero_weights_disable_components(self) -> None:
        loss_fn = DMCountLoss(lambda_count=0.0, lambda_ot=0.0, lambda_tv=0.0)
        pred = torch.rand(2, 1, 8, 8)
        gt = torch.rand(2, 1, 8, 8)
        assert loss_fn(pred, gt).item() == pytest.approx(0.0, abs=1e-6)

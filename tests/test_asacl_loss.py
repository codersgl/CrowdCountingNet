"""Tests for ASACL (Adaptive Structural-Perceptual Composite Loss)."""

from __future__ import annotations

import torch
import pytest

from crowdcount.plugins.asacl_loss import AdaptiveStructuralPerceptualLoss


# Use small spatial size that can pass through VGG conv layers
H, W = 32, 32


@pytest.fixture()
def loss_fn() -> AdaptiveStructuralPerceptualLoss:
    return AdaptiveStructuralPerceptualLoss(
        beta=1.0, lambda_adapt=1.0, lambda_struct=0.5, lambda_percept=0.1
    )


# ---------------------------------------------------------------------------
# AdaptiveWeightedL1
# ---------------------------------------------------------------------------


class TestAdaptiveWeightedL1:
    def test_zero_when_equal(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        x = torch.rand(2, 1, H, W)
        assert loss_fn.adaptive_weighted_l1(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_different(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.ones(1, 1, H, W)
        gt = torch.zeros(1, 1, H, W)
        assert loss_fn.adaptive_weighted_l1(pred, gt).item() > 0.0

    def test_weight_suppresses_high_density(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        """High-density GT pixels receive lower weight, reducing their
        contribution *relative to* unweighted L1."""
        pred = torch.zeros(1, 1, 4, 4)
        gt_high = torch.full((1, 1, 4, 4), 100.0)
        weighted = loss_fn.adaptive_weighted_l1(pred, gt_high).item()
        unweighted = (pred - gt_high).abs().mean().item()  # plain L1
        # Adaptive weight should substantially reduce the loss vs plain L1
        assert weighted < unweighted * 0.5


# ---------------------------------------------------------------------------
# Structural (SSIM) loss
# ---------------------------------------------------------------------------


class TestStructuralLoss:
    def test_zero_when_equal(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        x = torch.rand(2, 1, H, W)
        loss = loss_fn.structural_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_when_different(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.rand(1, 1, H, W)
        gt = torch.rand(1, 1, H, W)
        assert loss_fn.structural_loss(pred, gt).item() > 0.0

    def test_bounded(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        """SSIM-based loss should be in [0, ~2]."""
        pred = torch.rand(2, 1, H, W)
        gt = torch.rand(2, 1, H, W)
        loss = loss_fn.structural_loss(pred, gt)
        assert 0.0 <= loss.item() <= 2.0


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------


class TestPerceptualLoss:
    def test_zero_when_equal(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        x = torch.rand(1, 1, H, W)
        loss = loss_fn.perceptual_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_different(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.rand(1, 1, H, W) * 10
        gt = torch.rand(1, 1, H, W) * 10
        assert loss_fn.perceptual_loss(pred, gt).item() > 0.0

    def test_vgg_params_frozen(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        """VGG parameters must not require gradients."""
        for param in loss_fn.vgg_slices.parameters():
            assert not param.requires_grad

    def test_vgg_stays_eval_after_train(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        """VGG BN must stay in eval mode even when .train() is called."""
        loss_fn.train()
        for module in loss_fn.vgg_slices.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training, "VGG BN should stay in eval mode"


# ---------------------------------------------------------------------------
# Combined forward
# ---------------------------------------------------------------------------


class TestASACLCombined:
    def test_output_is_scalar(self, loss_fn: AdaptiveStructuralPerceptualLoss) -> None:
        pred = torch.rand(2, 1, H, W)
        gt = torch.rand(2, 1, H, W)
        loss = loss_fn(pred, gt)
        assert loss.dim() == 0

    def test_gradient_flows_through_pred(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.rand(2, 1, H, W, requires_grad=True)
        gt = torch.rand(2, 1, H, W)
        loss = loss_fn(pred, gt)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_vgg_no_grad_after_backward(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.rand(1, 1, H, W, requires_grad=True)
        gt = torch.rand(1, 1, H, W)
        loss = loss_fn(pred, gt)
        loss.backward()
        for param in loss_fn.vgg_slices.parameters():
            assert param.grad is None

    def test_last_components_populated(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        pred = torch.rand(2, 1, H, W)
        gt = torch.rand(2, 1, H, W)
        loss_fn(pred, gt)
        comp = loss_fn.last_components
        assert "den_adapt_loss" in comp
        assert "den_struct_loss" in comp
        assert "den_percept_loss" in comp

    def test_batch_size_scaling(
        self, loss_fn: AdaptiveStructuralPerceptualLoss
    ) -> None:
        """Loss pre-multiplied by batch size (MSELoss sum convention)."""
        # Disable percept for determinism; use only adapt
        fn = AdaptiveStructuralPerceptualLoss(
            lambda_adapt=1.0, lambda_struct=0.0, lambda_percept=0.0
        )
        torch.manual_seed(0)
        pred = torch.rand(1, 1, H, W)
        gt = torch.rand(1, 1, H, W)
        loss_b1 = fn(pred, gt).item()

        pred2 = pred.repeat(2, 1, 1, 1)
        gt2 = gt.repeat(2, 1, 1, 1)
        loss_b2 = fn(pred2, gt2).item()
        # After engine divides by B, per-sample cost should match
        assert loss_b1 / 1 == pytest.approx(loss_b2 / 2, rel=1e-5)

    def test_zero_weights_disable_components(self) -> None:
        fn = AdaptiveStructuralPerceptualLoss(
            lambda_adapt=0.0, lambda_struct=0.0, lambda_percept=0.0
        )
        pred = torch.rand(2, 1, H, W)
        gt = torch.rand(2, 1, H, W)
        assert fn(pred, gt).item() == pytest.approx(0.0, abs=1e-6)

    def test_disable_percept_only(self) -> None:
        fn = AdaptiveStructuralPerceptualLoss(
            lambda_adapt=1.0, lambda_struct=0.5, lambda_percept=0.0
        )
        pred = torch.rand(1, 1, H, W)
        gt = torch.rand(1, 1, H, W)
        loss = fn(pred, gt)
        assert loss.item() > 0.0
        assert fn.last_components["den_percept_loss"] > 0.0  # still computed

    def test_disable_struct_only(self) -> None:
        fn = AdaptiveStructuralPerceptualLoss(
            lambda_adapt=1.0, lambda_struct=0.0, lambda_percept=0.1
        )
        pred = torch.rand(1, 1, H, W)
        gt = torch.rand(1, 1, H, W)
        loss = fn(pred, gt)
        assert loss.item() > 0.0

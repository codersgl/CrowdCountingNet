"""Tests for ForegroundSuppressionBranch."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from crowdcount.models.head import ForegroundSuppressionBranch


@pytest.fixture
def branch() -> ForegroundSuppressionBranch:
    return ForegroundSuppressionBranch(in_channels=256, hidden_channels=64)


class TestForegroundSuppressionBranch:
    def test_output_shapes(self, branch: ForegroundSuppressionBranch) -> None:
        x = torch.randn(2, 256, 8, 8)
        gated, fg_logits, fg_prob = branch(x)
        assert gated.shape == (2, 256, 8, 8)
        assert fg_logits.shape == (2, 1, 8, 8)
        assert fg_prob.shape == (2, 1, 8, 8)

    def test_fg_prob_range(self, branch: ForegroundSuppressionBranch) -> None:
        x = torch.randn(2, 256, 8, 8)
        _, _, fg_prob = branch(x)
        assert fg_prob.min() >= 0.0
        assert fg_prob.max() <= 1.0

    def test_residual_gating_preserves_signal(self) -> None:
        """With base=0.5, even if fg_prob=0 the output is 0.5*x (not zero)."""
        branch = ForegroundSuppressionBranch(base=0.5, scale=0.5)
        branch.eval()
        x = torch.ones(1, 256, 4, 4)
        gated, _, fg_prob = branch(x)
        # gated = x * (0.5 + 0.5 * fg_prob); always >= 0.5 * x
        assert gated.min() >= 0.5 * x.min() - 1e-5

    def test_fg_loss_computation(self, branch: ForegroundSuppressionBranch) -> None:
        """Verify BCE loss with pos_weight computes without error."""
        x = torch.randn(2, 256, 8, 8)
        _, fg_logits, _ = branch(x)
        # Simulate GT: sparse foreground
        fg_gt = torch.zeros(2, 1, 8, 8)
        fg_gt[:, :, 3:5, 3:5] = 1.0
        loss = F.binary_cross_entropy_with_logits(
            fg_logits, fg_gt, pos_weight=torch.tensor(5.0)
        )
        assert loss.isfinite()
        loss.backward()

    def test_fg_loss_all_background(self, branch: ForegroundSuppressionBranch) -> None:
        """Loss should be finite even when GT has no foreground pixels."""
        x = torch.randn(1, 256, 4, 4)
        _, fg_logits, _ = branch(x)
        fg_gt = torch.zeros(1, 1, 4, 4)
        loss = F.binary_cross_entropy_with_logits(
            fg_logits, fg_gt, pos_weight=torch.tensor(5.0)
        )
        assert loss.isfinite()

    def test_custom_base_scale(self) -> None:
        branch = ForegroundSuppressionBranch(base=0.3, scale=0.7)
        assert branch.base == 0.3
        assert branch.scale == 0.7

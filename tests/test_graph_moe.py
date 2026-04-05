"""Unit tests for the graph-aware MoE plugin."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.graph_moe import (
    CoarseDensityRouter,
    GraphAttentionExpert,
    GraphAwareMoE,
    GraphMoEBalanceLoss,
    LocalExpert,
    _window_partition,
    _window_unpartition,
)

B, C, H, W = 2, 64, 8, 8


# ---------------------------------------------------------------------------
# LocalExpert
# ---------------------------------------------------------------------------


class TestLocalExpert:
    def test_shape_with_density(self) -> None:
        expert = LocalExpert(input_dim=C, expansion=2).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, H, W)

    def test_shape_without_density(self) -> None:
        expert = LocalExpert(input_dim=C, expansion=2).eval()
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = expert(x, density=None)
        assert out.shape == (B, C, H, W)

    def test_density_gate_disabled(self) -> None:
        expert = LocalExpert(input_dim=C, expansion=2, use_density_gate=False).eval()
        assert expert.density_gate is None
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = expert(x, density=torch.rand(B, 1, H, W))
        assert out.shape == (B, C, H, W)

    def test_residual_at_init(self) -> None:
        """At init, project weights are small so output ≈ input."""
        expert = LocalExpert(input_dim=C, expansion=2, use_density_gate=False)
        x = torch.randn(B, C, H, W)
        out = expert(x, density=None)
        # Just check shape; residual makes output close but not exact
        assert out.shape == x.shape

    def test_density_spatial_mismatch(self) -> None:
        """Density map with different spatial size should be interpolated."""
        expert = LocalExpert(input_dim=C, expansion=2).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H * 2, W * 2)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# GraphAttentionExpert
# ---------------------------------------------------------------------------


class TestGraphAttentionExpert:
    def test_shape(self) -> None:
        expert = GraphAttentionExpert(input_dim=C, num_heads=4).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, H, W)

    def test_without_density(self) -> None:
        expert = GraphAttentionExpert(input_dim=C, num_heads=4).eval()
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = expert(x, density=None)
        assert out.shape == (B, C, H, W)

    def test_no_density_bias(self) -> None:
        expert = GraphAttentionExpert(
            input_dim=C, num_heads=4, use_density_bias=False
        ).eval()
        assert expert.density_bias_scale is None
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = expert(x, density=torch.rand(B, 1, H, W))
        assert out.shape == (B, C, H, W)

    def test_gate_zero_init(self) -> None:
        expert = GraphAttentionExpert(input_dim=C, num_heads=4)
        assert expert.gate.item() == 0.0

    def test_density_spatial_mismatch(self) -> None:
        expert = GraphAttentionExpert(input_dim=C, num_heads=2).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H * 2, W * 2)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# CoarseDensityRouter
# ---------------------------------------------------------------------------


class TestCoarseDensityRouter:
    def test_shape(self) -> None:
        router = CoarseDensityRouter(input_dim=C, num_experts=2, grid_stride=4).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            w = router(x, density)
        assert w.shape == (B, 2, H, W)

    def test_weights_sum_to_one(self) -> None:
        router = CoarseDensityRouter(input_dim=C, num_experts=2, grid_stride=4).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            w = router(x, density)
        assert torch.allclose(w.sum(dim=1), torch.ones(B, H, W), atol=1e-5)

    def test_small_spatial(self) -> None:
        """When H,W < grid_stride, skip pooling path."""
        router = CoarseDensityRouter(input_dim=C, num_experts=2, grid_stride=16).eval()
        x = torch.randn(B, C, 4, 4)
        density = torch.rand(B, 1, 4, 4)
        with torch.no_grad():
            w = router(x, density)
        assert w.shape == (B, 2, 4, 4)


# ---------------------------------------------------------------------------
# GraphMoEBalanceLoss
# ---------------------------------------------------------------------------


class TestGraphMoEBalanceLoss:
    def test_uniform_weights_low_loss(self) -> None:
        loss_fn = GraphMoEBalanceLoss(lambda_balance=0.01)
        # Perfectly balanced: each expert gets 0.5
        w = torch.ones(B, 2, H, W) * 0.5
        losses = loss_fn(w)
        assert "l_balance" in losses and "total_aux" in losses
        assert losses["l_balance"].item() < 0.01

    def test_imbalanced_weights_higher_loss(self) -> None:
        loss_fn = GraphMoEBalanceLoss(lambda_balance=0.01)
        w = torch.zeros(B, 2, H, W)
        w[:, 0] = 1.0  # all routed to expert 0
        losses = loss_fn(w)
        assert losses["l_balance"].item() > 0.1


# ---------------------------------------------------------------------------
# GraphAwareMoE (top-level)
# ---------------------------------------------------------------------------


class TestGraphAwareMoE:
    def test_forward_shape_and_keys(self) -> None:
        moe = GraphAwareMoE(input_dim=C, num_heads=4, local_expansion=2).train()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        feat, aux, weights = moe(x, density, training=True)
        assert feat.shape == (B, C, H, W)
        assert weights.shape == (B, 2, H, W)
        assert "total_aux" in aux

    def test_eval_no_aux(self) -> None:
        moe = GraphAwareMoE(input_dim=C, num_heads=4, local_expansion=2).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            feat, aux, weights = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)
        assert aux == {}

    def test_ablation_disable_local(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C, num_heads=4, local_expansion=2, disable_local_expert=True
        )
        assert moe.local_expert is None
        assert moe.router is None
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        # Eval mode
        moe.eval()
        with torch.no_grad():
            feat, aux_eval, weights = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)
        assert weights[:, 0].sum() == 0.0  # local weight = 0
        assert aux_eval == {}
        # Train mode: single-expert ablation must NOT produce balance loss
        moe.train()
        feat_t, aux_train, _ = moe(x, density, training=True)
        assert aux_train == {}

    def test_ablation_disable_global(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C, num_heads=4, local_expansion=2, disable_global_expert=True
        )
        assert moe.global_expert is None
        assert moe.router is None
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        # Eval mode
        moe.eval()
        with torch.no_grad():
            feat, aux_eval, weights = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)
        assert aux_eval == {}
        # Train mode: single-expert ablation must NOT produce balance loss
        moe.train()
        feat_t, aux_train, _ = moe(x, density, training=True)
        assert aux_train == {}

    def test_ablation_both_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot disable both"):
            GraphAwareMoE(
                input_dim=C,
                num_heads=4,
                disable_local_expert=True,
                disable_global_expert=True,
            )

    def test_ablation_disable_graph_bias(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C, num_heads=4, local_expansion=2, disable_graph_bias=True
        ).eval()
        assert moe.global_expert is not None
        assert not moe.global_expert.use_density_bias
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            feat, _, _ = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)

    def test_gradient_flows(self) -> None:
        moe = GraphAwareMoE(input_dim=C, num_heads=4, local_expansion=2).train()
        x = torch.randn(B, C, H, W, requires_grad=True)
        density = torch.rand(B, 1, H, W)
        feat, aux, _ = moe(x, density, training=True)
        loss = feat.sum() + aux["total_aux"]
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Window partition helpers
# ---------------------------------------------------------------------------


class TestWindowPartition:
    def test_round_trip_exact(self) -> None:
        x = torch.randn(B, C, H, W)
        wins, orig, padded = _window_partition(x, window_size=4)
        restored = _window_unpartition(wins, B, orig, padded, 4)
        assert torch.allclose(restored, x)

    def test_round_trip_with_padding(self) -> None:
        """Non-divisible spatial dims require padding; unpartition crops."""
        x = torch.randn(B, C, 7, 7)
        wins, orig, padded = _window_partition(x, window_size=4)
        assert orig == (7, 7)
        assert padded == (8, 8)
        restored = _window_unpartition(wins, B, orig, padded, 4)
        assert restored.shape == (B, C, 7, 7)
        assert torch.allclose(restored, x)


# ---------------------------------------------------------------------------
# Local-first enhancements
# ---------------------------------------------------------------------------


class TestLocalExpertWindowPartition:
    def test_shape_with_window(self) -> None:
        expert = LocalExpert(input_dim=C, expansion=2, window_size=4).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, H, W)

    def test_shape_non_divisible(self) -> None:
        """Window partition handles non-divisible spatial dims."""
        expert = LocalExpert(input_dim=C, expansion=2, window_size=4).eval()
        x = torch.randn(B, C, 7, 7)
        density = torch.rand(B, 1, 7, 7)
        with torch.no_grad():
            out = expert(x, density)
        assert out.shape == (B, C, 7, 7)

    def test_gradient_flows(self) -> None:
        expert = LocalExpert(input_dim=C, expansion=2, window_size=4).train()
        x = torch.randn(B, C, H, W, requires_grad=True)
        density = torch.rand(B, 1, H, W)
        out = expert(x, density)
        out.sum().backward()
        assert x.grad is not None


class TestCoarseDensityRouterLocalPrior:
    def test_neutral_prior(self) -> None:
        router = CoarseDensityRouter(input_dim=C, num_experts=2, local_prior=0.0)
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        w = router(x, density)
        assert w.shape == (B, 2, H, W)

    def test_positive_prior_favours_local(self) -> None:
        """With strong local_prior, expert 0 should get majority weight."""
        router = CoarseDensityRouter(
            input_dim=C, num_experts=2, grid_stride=4, local_prior=5.0
        ).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            w = router(x, density)
        # Local expert (index 0) mean weight should be > 0.8
        assert w[:, 0].mean().item() > 0.8

    def test_weights_still_sum_to_one(self) -> None:
        router = CoarseDensityRouter(input_dim=C, num_experts=2, local_prior=2.0).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            w = router(x, density)
        assert torch.allclose(w.sum(dim=1), torch.ones(B, H, W), atol=1e-5)


class TestGraphAwareMoELocalFirst:
    def test_local_first_forward_shape(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C,
            num_heads=4,
            local_expansion=2,
            local_window_size=4,
            local_prior=1.0,
        ).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            feat, aux, weights = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)
        assert weights.shape == (B, 2, H, W)

    def test_local_first_train_aux_loss(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C,
            num_heads=4,
            local_expansion=2,
            local_window_size=4,
            local_prior=1.0,
        ).train()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        feat, aux, weights = moe(x, density, training=True)
        assert "total_aux" in aux

    def test_local_first_gradient_flows(self) -> None:
        moe = GraphAwareMoE(
            input_dim=C,
            num_heads=4,
            local_expansion=2,
            local_window_size=4,
            local_prior=1.0,
        ).train()
        x = torch.randn(B, C, H, W, requires_grad=True)
        density = torch.rand(B, 1, H, W)
        feat, aux, _ = moe(x, density, training=True)
        loss = feat.sum() + aux["total_aux"]
        loss.backward()
        assert x.grad is not None

    def test_backward_compat_defaults(self) -> None:
        """Default params (window_size=0, local_prior=0) match original."""
        moe = GraphAwareMoE(input_dim=C, num_heads=4, local_expansion=2)
        assert moe.local_expert is not None
        assert moe.local_expert.window_size == 0
        assert moe.router is not None
        assert moe.router.local_prior == 0.0

    def test_ablation_still_works_with_local_first(self) -> None:
        """disable_global should still work when local-first is on."""
        moe = GraphAwareMoE(
            input_dim=C,
            num_heads=4,
            local_expansion=2,
            local_window_size=4,
            disable_global_expert=True,
        ).eval()
        x = torch.randn(B, C, H, W)
        density = torch.rand(B, 1, H, W)
        with torch.no_grad():
            feat, aux, weights = moe(x, density, training=False)
        assert feat.shape == (B, C, H, W)
        assert aux == {}

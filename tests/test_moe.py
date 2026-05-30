"""Unit tests for ESCA and LightMoE plugins."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.moe import (
    ESCA,
    LightMoE,
    MicroBiasCorrector,
    MicroEdgeRefiner,
    MicroDensityAdapter,
)


def test_esca_shape_preserved() -> None:
    module = ESCA(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# LightMoE (micro-expert) unit tests
# ---------------------------------------------------------------------------


def test_micro_bias_corrector_residual() -> None:
    m = MicroBiasCorrector(dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_micro_edge_refiner_residual() -> None:
    m = MicroEdgeRefiner(dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_micro_density_adapter_with_density() -> None:
    m = MicroDensityAdapter(dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    d = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        y = m(x, d)
    assert y.shape == x.shape


def test_micro_density_adapter_without_density() -> None:
    m = MicroDensityAdapter(dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = m(x, None)
    assert y.shape == x.shape


def test_light_moe_forward_shape() -> None:
    moe = LightMoE(input_dim=32, grid_stride=2).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        fused, aux, weights = moe(x, training=False)
    assert fused.shape == x.shape
    assert weights.shape == (2, 3, 8, 8)
    assert len(aux) == 0  # no aux in eval


def test_light_moe_weights_sum_to_one() -> None:
    moe = LightMoE(input_dim=32, grid_stride=2).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        _, _, weights = moe(x, training=False)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_light_moe_training_aux_loss() -> None:
    moe = LightMoE(input_dim=32, grid_stride=2).train()
    x = torch.randn(2, 32, 8, 8)
    _, aux, _ = moe(x, training=True)
    assert "total_aux" in aux
    assert "l_balance" in aux


def test_light_moe_param_count_small() -> None:
    moe = LightMoE(input_dim=256, grid_stride=4)
    total = sum(p.numel() for p in moe.parameters())
    assert total < 500_000, f"LightMoE params={total}, expected < 500k"


def test_light_moe_with_density_hint() -> None:
    moe = LightMoE(input_dim=32, grid_stride=2, use_density_hint=True).eval()
    x = torch.randn(2, 32, 8, 8)
    d = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        fused, _, weights = moe(x, density_hint=d, training=False)
    assert fused.shape == x.shape


def test_light_moe_beta_gate_starts_at_zero() -> None:
    """Beta gate should be initialized to 0, so output equals input initially."""
    moe = LightMoE(input_dim=32, grid_stride=2).eval()
    assert moe.beta.item() == 0.0
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        fused, _, _ = moe(x, training=False)
    # With beta=0, fused should be identical to input
    assert torch.allclose(fused, x, atol=1e-6)


def test_light_moe_beta_gate_grad_flows() -> None:
    """Beta should receive gradients from the main loss."""
    moe = LightMoE(input_dim=32, grid_stride=2).train()
    x = torch.randn(2, 32, 8, 8)
    fused, _, _ = moe(x, training=True)
    loss = fused.sum()
    loss.backward()
    assert moe.beta.grad is not None


def test_light_moe_router_ignores_density_hint() -> None:
    """Router should NOT use density_hint, even when use_density_hint=True."""
    moe = LightMoE(input_dim=32, grid_stride=2, use_density_hint=True).eval()
    x = torch.randn(2, 32, 8, 8)
    d1 = torch.rand(2, 1, 8, 8)
    d2 = torch.rand(2, 1, 8, 8) * 10  # very different density
    with torch.no_grad():
        _, _, w1 = moe(x, density_hint=d1, training=False)
        _, _, w2 = moe(x, density_hint=d2, training=False)
    # Router weights must be identical since router doesn't see density
    assert torch.allclose(w1, w2, atol=1e-6)

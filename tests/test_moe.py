"""Unit tests for ESCA and compact MoE plugins."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.moe import (
    ESCA,
    MoE,
    CountCalibrationExpert,
    LocalizationExpert,
    DensityAdaptiveExpert,
    GridSoftRouter,
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


def test_count_calibration_expert_shape() -> None:
    expert = CountCalibrationExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_localization_expert_shape() -> None:
    expert = LocalizationExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_density_adaptive_expert_with_density() -> None:
    expert = DensityAdaptiveExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        y = expert(x, density)
    assert y.shape == x.shape


def test_density_adaptive_expert_without_density() -> None:
    expert = DensityAdaptiveExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x, density=None)
    assert y.shape == x.shape


def test_grid_soft_router_output_shape() -> None:
    router = GridSoftRouter(input_dim=32, grid_stride=4).eval()
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        weights = router(x)
    assert weights.shape == (2, 3, 16, 16)


def test_grid_soft_router_weights_sum_to_one() -> None:
    router = GridSoftRouter(input_dim=32, grid_stride=4).eval()
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        weights = router(x)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2, 16, 16),
        atol=1e-6,
    )


def test_grid_soft_router_small_input() -> None:
    router = GridSoftRouter(input_dim=32, grid_stride=4).eval()
    x = torch.randn(2, 32, 2, 2)
    with torch.no_grad():
        weights = router(x)
    assert weights.shape == (2, 3, 2, 2)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2, 2, 2),
        atol=1e-6,
    )


def test_grid_soft_router_with_density_hint() -> None:
    router = GridSoftRouter(input_dim=32, grid_stride=4, use_density_hint=True).eval()
    x = torch.randn(2, 32, 16, 16)
    density = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        weights = router(x, density_hint=density)
    assert weights.shape == (2, 3, 16, 16)


def test_moe_soft_routing_weights_sum_to_one_train() -> None:
    moe = MoE(input_dim=32).train()
    x = torch.randn(4, 32, 8, 8)
    with torch.no_grad():
        _, _, weights = moe(x, training=True)
    assert weights.shape == (4, 3, 8, 8)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(4, 8, 8),
        atol=1e-6,
    )


def test_moe_soft_routing_weights_sum_to_one_eval() -> None:
    moe = MoE(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        _, _, weights = moe(x, training=False)
    assert weights.shape == (2, 3, 8, 8)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2, 8, 8),
        atol=1e-6,
    )


def test_moe_train_test_scale_consistency() -> None:
    moe = MoE(input_dim=32)
    x = torch.randn(2, 32, 8, 8)
    moe.train()
    with torch.no_grad():
        _, _, w_train = moe(x, training=True)
    moe.eval()
    with torch.no_grad():
        _, _, w_eval = moe(x, training=False)
    assert torch.allclose(w_train.sum(dim=1), torch.ones(2, 8, 8), atol=1e-6)
    assert torch.allclose(w_eval.sum(dim=1), torch.ones(2, 8, 8), atol=1e-6)


def test_moe_output_shape() -> None:
    moe = MoE(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        fused, _, _ = moe(x, training=False)
    assert fused.shape == x.shape


def test_moe_with_density_hint() -> None:
    moe = MoE(input_dim=32, use_density_hint=True).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        fused, _, weights = moe(x, density_hint=density, training=False)
    assert fused.shape == x.shape
    assert weights.shape == (2, 3, 8, 8)


def test_moe_aux_losses_in_train_mode() -> None:
    moe = MoE(input_dim=32).train()
    x = torch.randn(2, 32, 8, 8)
    _, aux_losses, _ = moe(x, training=True)
    assert "total_aux" in aux_losses
    assert "l_balance" in aux_losses


def test_moe_no_aux_losses_in_eval_mode() -> None:
    moe = MoE(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        _, aux_losses, _ = moe(x, training=False)
    assert aux_losses == {}


def test_temperature_decay_reduces_temperature() -> None:
    moe = MoE(input_dim=32, temperature_init=1.0, temperature_min=0.4)
    initial_temp = moe.temperature
    for _ in range(100):
        moe.update_temperature(decay_rate=0.99)
    assert moe.temperature < initial_temp
    assert moe.temperature >= 0.4


def test_temperature_clamps_at_minimum() -> None:
    moe = MoE(input_dim=32, temperature_init=1.0, temperature_min=0.4)
    for _ in range(100_000):
        moe.update_temperature(decay_rate=0.99)
    assert moe.temperature == 0.4


def test_noise_scale_is_noop() -> None:
    moe = MoE(input_dim=32)
    assert moe._current_noise_scale == 0.0
    moe.update_noise_scale(0.5)
    assert moe._current_noise_scale == 0.0


def test_aux_loss_grad_flows_to_router_parameters() -> None:
    moe = MoE(input_dim=32).train()
    x = torch.randn(2, 32, 8, 8)
    _, aux_losses, _ = moe(x, training=True)
    total_aux = aux_losses["total_aux"]
    total_aux.backward()
    router_params = list(moe.router.parameters())
    grads = [p.grad for p in router_params if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_no_decorrelation_loss() -> None:
    moe = MoE(input_dim=32).train()
    x = torch.randn(2, 32, 8, 8)
    _, aux_losses, _ = moe(x, training=True)
    assert "l_decorr" not in aux_losses


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

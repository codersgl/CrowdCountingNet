"""Unit tests for ESCA and MoE plugins."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.moe import (
    ESCA,
    MoE,
    GlobalExpert,
    ScaleAdaptiveExpert,
    DensityAwareExpert,
)


def test_esca_shape_preserved() -> None:
    module = ESCA(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape


def test_moe_specialization_routing_is_one_hot() -> None:
    moe = MoE(input_dim=32, top_k=2)
    moe.train()

    x = torch.randn(4, 32, 8, 8)
    with torch.no_grad():
        _, _, weights = moe(x, training=True)

    assert weights.shape == (4, 5, 8, 8)
    # 每个像素位置选择 top_k=2 个专家，展展求和应为 2
    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((4, 8, 8), 2.0, device=weights.device),
    )


def test_moe_coordination_hard_routing_topk() -> None:
    moe = MoE(input_dim=32, top_k=2)
    moe.train()

    x = torch.randn(3, 32, 8, 8)
    with torch.no_grad():
        _, aux_losses, weights = moe(x, training=True)

    assert weights.shape == (3, 5, 8, 8)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((3, 8, 8), 2.0, device=weights.device),
    )
    assert "total_aux" in aux_losses


def test_moe_eval_soft_routing_probabilities() -> None:
    moe = MoE(input_dim=32, top_k=2)
    moe.eval()

    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        _, aux_losses, weights = moe(x, training=False)

    assert weights.shape == (2, 5, 8, 8)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, 8, 8, device=weights.device), atol=1e-6
    )


def test_scale_adaptive_expert_shape() -> None:
    expert = ScaleAdaptiveExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_density_aware_expert_with_density() -> None:
    expert = DensityAwareExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        y = expert(x, density)
    assert y.shape == x.shape


def test_density_aware_expert_without_density() -> None:
    expert = DensityAwareExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x, density=None)
    assert y.shape == x.shape


def test_moe_with_density_hint() -> None:
    moe = MoE(input_dim=32, top_k=2, use_density_hint=True).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        fused, aux_losses, weights = moe(x, density_hint=density, training=False)
    assert fused.shape == x.shape
    assert weights.shape == (2, 5, 8, 8)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, 8, 8, device=weights.device), atol=1e-6
    )


def test_moe_default_hyperparameters_match_repo_config() -> None:
    moe = MoE(input_dim=32)

    assert moe.temperature_min == 0.4
    assert moe.aux_loss.lambda_balance == 0.05
    assert moe.aux_loss.lambda_decorr == 10.0


def test_global_expert_output_shape() -> None:
    """GlobalExpert (PCA) 应保持 spatial 尺寸不变。"""
    expert = GlobalExpert(input_dim=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_global_expert_large_input_no_truncation() -> None:
    """GlobalExpert 处理超过原 max_attn_tokens=1024 的大尺寸输入时不截断不上采样。"""
    expert = GlobalExpert(input_dim=32).eval()
    # N = 32*32 = 1024, 恰好等于原截断阈值
    x_exact = torch.randn(1, 32, 32, 32)
    with torch.no_grad():
        y_exact = expert(x_exact)
    assert y_exact.shape == x_exact.shape

    # N = 48*64 = 3072, 远超原阈值，验证无信息损失路径
    x_large = torch.randn(1, 32, 48, 64)
    with torch.no_grad():
        y_large = expert(x_large)
    assert y_large.shape == x_large.shape


def test_global_expert_train_mode_batch1() -> None:
    """batch_size=1 训练模式下 GlobalExpert BN 应正常运行（无 '> 1 value' 报错）。"""
    expert = GlobalExpert(input_dim=32).train()
    x = torch.randn(1, 32, 8, 8)
    y = expert(x)  # 不应抛出 ValueError
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Decay schedule and gradient-flow tests
# ---------------------------------------------------------------------------


def test_temperature_decay_reduces_temperature() -> None:
    """update_temperature 应按 decay_rate 衰减 temperature，同时同步 router.temperature。"""
    moe = MoE(input_dim=32, top_k=2, temperature_init=1.0, temperature_min=0.4)
    initial_temp = moe.temperature

    for _ in range(100):
        moe.update_temperature(decay_rate=0.99)

    assert moe.temperature < initial_temp
    assert moe.temperature >= 0.4  # 不低于 temperature_min
    # router 必须与 moe 保持同步
    assert moe.router.temperature == moe.temperature


def test_temperature_clamps_at_minimum() -> None:
    """经过大量 decay 后，temperature 应保持在 temperature_min，不再继续降低。"""
    moe = MoE(input_dim=32, top_k=2, temperature_init=1.0, temperature_min=0.4)
    for _ in range(100_000):
        moe.update_temperature(decay_rate=0.99)

    assert moe.temperature == 0.4
    assert moe.router.temperature == 0.4


def test_noise_scale_reaches_zero_after_twenty_percent() -> None:
    """update_noise_scale 应在 progress >= 0.2 时将噪声衰减为 0。"""
    moe = MoE(input_dim=32, top_k=2)
    assert moe._current_noise_scale == 0.5  # 初始值

    moe.update_noise_scale(0.0)
    assert moe._current_noise_scale == pytest.approx(0.5, abs=1e-6)

    moe.update_noise_scale(0.1)  # 50% 进度（在 20% 区间内）
    assert moe._current_noise_scale == pytest.approx(0.25, abs=1e-5)

    moe.update_noise_scale(0.2)
    assert moe._current_noise_scale == pytest.approx(0.0, abs=1e-6)

    moe.update_noise_scale(1.0)
    assert moe._current_noise_scale == pytest.approx(0.0, abs=1e-6)


def test_aux_loss_grad_flows_to_router_parameters() -> None:
    """total_aux 必须对 router 和 context_encoder 的参数有非零梯度。"""
    moe = MoE(input_dim=32, top_k=2).train()
    x = torch.randn(2, 32, 8, 8, requires_grad=False)

    _, aux_losses, _ = moe(x, training=True)
    total_aux = aux_losses["total_aux"]
    total_aux.backward()

    router_params = list(moe.router.parameters()) + list(
        moe.context_encoder.parameters()
    )
    grads = [p.grad for p in router_params if p.grad is not None]
    assert len(grads) > 0, "router/context_encoder 参数应收到梯度"
    assert any(g.abs().sum().item() > 0 for g in grads), "至少一个参数的梯度应非零"

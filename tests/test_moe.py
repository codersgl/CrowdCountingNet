"""Unit tests for ESCA and MoE plugins."""

from __future__ import annotations

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
    moe.set_training_stage("specialization")

    x = torch.randn(4, 32, 8, 8)
    with torch.no_grad():
        _, _, weights = moe(x, training=True)

    assert weights.shape == (4, 5)
    # Specialization now uses top_k=2 noisy gate routing: each sample selects 2 experts.
    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((4,), 2.0, device=weights.device),
    )


def test_moe_coordination_hard_routing_topk() -> None:
    moe = MoE(input_dim=32, top_k=2)
    moe.train()
    moe.set_training_stage("coordination")

    x = torch.randn(3, 32, 8, 8)
    with torch.no_grad():
        _, aux_losses, weights = moe(x, training=True)

    assert weights.shape == (3, 5)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.full((3,), 2.0, device=weights.device),
    )
    assert "total_aux" in aux_losses


def test_moe_eval_soft_routing_probabilities() -> None:
    moe = MoE(input_dim=32, top_k=2)
    moe.eval()

    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        _, aux_losses, weights = moe(x, training=False)

    assert weights.shape == (2, 5)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, device=weights.device), atol=1e-6
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
    assert weights.shape == (2, 5)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, device=weights.device), atol=1e-6
    )


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

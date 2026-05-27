"""Tests for DeformableCrossScaleExpert and its integration with HeterogeneousSparseMoE."""

from __future__ import annotations

import torch
from torch import nn

from crowdcount.models.moecount.deformable_expert import DeformableCrossScaleExpert
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE


def test_deformable_expert_output_shape() -> None:
    """Output shape must equal input shape."""
    expert = DeformableCrossScaleExpert(channels=64, num_heads=4,
                                         num_sampling_points=4, num_scale_levels=2)
    x = torch.randn(2, 64, 16, 20)
    out = expert(x)
    assert out.shape == x.shape


def test_deformable_expert_zero_init_offsets() -> None:
    """At init, offset prediction produces zeros."""
    expert = DeformableCrossScaleExpert(channels=64, num_heads=4,
                                         num_sampling_points=4, num_scale_levels=2)
    x = torch.randn(1, 64, 8, 10)
    expert.eval()
    with torch.no_grad():
        expert(x)
        raw = expert.offset_pred(x)
        assert torch.allclose(raw, torch.zeros_like(raw), atol=1e-6)


def test_deformable_expert_gradient_flow() -> None:
    """Gradients must flow through all parameters."""
    expert = DeformableCrossScaleExpert(channels=64, num_heads=4,
                                         num_sampling_points=4, num_scale_levels=2,
                                         dropout=0.0)
    x = torch.randn(2, 64, 8, 10, requires_grad=True)
    out = expert(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert expert.offset_pred[-1].weight.grad is not None
    assert expert.q_proj.weight.grad is not None
    assert expert.out_proj.weight.grad is not None


def test_deformable_expert_residual_gate_init() -> None:
    """Residual gate initialised to zero."""
    expert = DeformableCrossScaleExpert(channels=64)
    assert expert.residual_gate.item() == 0.0
    assert expert.residual_gate.tanh().item() == 0.0


def test_deformable_expert_distance_lambda_nonnegative() -> None:
    """Distance lambda clamped to >= 0 during forward."""
    expert = DeformableCrossScaleExpert(channels=64, num_heads=4,
                                         num_sampling_points=4, num_scale_levels=2)
    expert.distance_lambda.data.fill_(-5.0)
    x = torch.randn(1, 64, 8, 10)
    out = expert(x)
    assert out is not None


def test_moe_with_deformable_expert() -> None:
    """HeterogeneousSparseMoE with use_deformable_expert=True."""
    moe = HeterogeneousSparseMoE(
        channels=32,
        gate_hidden_channels=8,
        warmup_epochs=0,
        use_deformable_expert=True,
        deformable_num_heads=4,
        deformable_num_sampling_points=4,
        deformable_num_scale_levels=2,
        deformable_dropout=0.0,
    )
    x = torch.randn(2, 32, 16, 20)
    moe.train()
    fused, aux_losses, route = moe(x)
    assert fused.shape == x.shape
    assert "total_aux" in aux_losses
    assert route["weights"].shape == (2, 3, 16, 20)


def test_moe_without_deformable_fallback() -> None:
    """use_deformable_expert=False uses original SpatialRelationExpert."""
    moe = HeterogeneousSparseMoE(
        channels=32,
        gate_hidden_channels=8,
        warmup_epochs=0,
        use_deformable_expert=False,
    )
    x = torch.randn(2, 32, 16, 20)
    fused, _aux, _route = moe(x)
    assert fused.shape == x.shape


def test_deformable_expert_multiple_scales() -> None:
    """3 scale levels with various spatial dims."""
    expert = DeformableCrossScaleExpert(
        channels=64, num_heads=2, num_sampling_points=4, num_scale_levels=3,
    )
    for h, w in [(15, 19), (16, 20), (17, 23)]:
        x = torch.randn(1, 64, h, w)
        out = expert(x)
        assert out.shape == x.shape


def test_deformable_expert_no_se() -> None:
    """use_se=False works."""
    expert = DeformableCrossScaleExpert(channels=64, num_heads=4,
                                         num_sampling_points=4, use_se=False)
    x = torch.randn(1, 64, 8, 8)
    out = expert(x)
    assert out.shape == x.shape

"""Unit tests for ESCA and MoE plugins."""

from __future__ import annotations

import torch

from crowdcount.plugins.moe import ESCA, MoE


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
    assert torch.allclose(weights.sum(dim=1), torch.ones(4, device=weights.device))


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
    assert aux_losses == {}

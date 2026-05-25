from __future__ import annotations

import torch

from crowdcount.models.moecount.gate import SparseTop2Gate
from crowdcount.models.moecount.losses import LoadBalanceLoss


def test_sparse_gate_warmup_uses_soft_all_routing() -> None:
    gate = SparseTop2Gate(
        in_channels=16,
        hidden_channels=8,
        warmup_epochs=5,
        temperature_init=1.0,
    ).train()
    gate.set_epoch(0, total_epochs=20)
    output = gate(torch.randn(2, 16, 8, 8))
    weights = output["weights"]
    assert output["warmup_active"] is True
    assert isinstance(weights, torch.Tensor)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 8, 8), atol=1e-6)
    assert torch.all(weights > 0)


def test_sparse_gate_eval_uses_hard_top2_forward_weights() -> None:
    gate = SparseTop2Gate(in_channels=16, hidden_channels=8).eval()
    output = gate(torch.randn(2, 16, 8, 8))
    weights = output["weights"]
    assert isinstance(weights, torch.Tensor)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 8, 8), atol=1e-6)
    assert torch.all((weights > 0).sum(dim=1) == 2)


def test_sparse_gate_temperature_clamps() -> None:
    gate = SparseTop2Gate(
        in_channels=16,
        hidden_channels=8,
        temperature_init=1.0,
        temperature_min=0.1,
        temperature_decay=0.5,
    )
    gate.set_epoch(20, total_epochs=100)
    assert gate.temperature == 0.1


def test_balance_loss_grad_flows_to_gate_parameters() -> None:
    gate = SparseTop2Gate(in_channels=16, hidden_channels=8, warmup_epochs=0).train()
    gate.set_epoch(10, total_epochs=20)
    route = gate(torch.randn(2, 16, 8, 8))
    balance = LoadBalanceLoss(lambda_importance=0.01, lambda_load=0.01)
    loss = balance(route["soft_probs"], route["hard_mask"])["total_aux"]
    loss.backward()
    gradients = [parameter.grad for parameter in gate.parameters() if parameter.grad is not None]
    assert gradients
    assert any(gradient.abs().sum().item() > 0 for gradient in gradients)

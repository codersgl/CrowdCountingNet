from __future__ import annotations

import torch

from crowdcount.models.moecount.losses import (
    BayesianLoss,
    LoadBalanceLoss,
    LogCountLoss,
    LogCountWeightSchedule,
)


def test_bayesian_loss_handles_empty_points() -> None:
    loss_fn = BayesianLoss(max_pixels_per_chunk=16)
    pred_density = torch.ones(1, 1, 4, 4)
    targets = [{"point": torch.zeros(0, 2)}]
    loss = loss_fn(pred_density, targets=targets, image_sizes=(32, 32))
    assert torch.isclose(loss, pred_density.sum())


def test_bayesian_loss_multi_point_is_finite() -> None:
    loss_fn = BayesianLoss(max_pixels_per_chunk=8)
    pred_density = torch.full((1, 1, 4, 4), 0.25)
    targets = [{"point": torch.tensor([[8.0, 8.0], [24.0, 24.0]])}]
    loss = loss_fn(pred_density, targets=targets, image_sizes=(32, 32))
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_log_count_loss_matches_manual_value() -> None:
    loss_fn = LogCountLoss()
    pred_density = torch.ones(1, 1, 2, 2)
    targets = [{"point": torch.zeros(3, 2)}]
    loss = loss_fn(pred_density, targets)
    expected = (torch.log1p(torch.tensor(4.0)) - torch.log1p(torch.tensor(3.0))).abs()
    assert torch.allclose(loss, expected)


def test_log_count_weight_schedule_decays_to_floor() -> None:
    schedule = LogCountWeightSchedule(
        initial_weight=0.1,
        decay_epochs=50,
        decay_rate=0.5,
        min_weight=0.05,
    )
    assert schedule.weight_at(0) == 0.1
    assert schedule.weight_at(50) == 0.05
    assert schedule.weight_at(500) == 0.05


def test_load_balance_loss_penalizes_biased_routing() -> None:
    balance = LoadBalanceLoss(lambda_importance=1.0, lambda_load=1.0)
    uniform = torch.full((2, 3, 4, 4), 1.0 / 3.0)
    biased = torch.zeros(2, 3, 4, 4)
    biased[:, 0] = 1.0
    uniform_loss = balance(uniform)["total_aux"]
    biased_loss = balance(biased)["total_aux"]
    assert uniform_loss.item() < 1e-6
    assert biased_loss.item() > uniform_loss.item()

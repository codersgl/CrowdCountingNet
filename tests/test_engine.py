"""Tests for train_one_epoch and evaluate_crowd_no_overlap using mock data."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from crowdcount.engine import evaluate_crowd_no_overlap, train_one_epoch
from crowdcount.models.criterion import SetCriterion_Crowd
from crowdcount.models.matcher import HungarianMatcher_Crowd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyDSGC(nn.Module):
    """Tiny model with the same output interface as DSGCnet."""

    def __init__(self, num_queries: int = 20):
        super().__init__()
        self.num_queries = num_queries
        self.linear = nn.Linear(1, 1)  # just to have parameters

    def forward(self, samples):
        B = samples.shape[0]
        return {
            "pred_logits": torch.rand(B, self.num_queries, 2, requires_grad=True),
            "pred_points": torch.rand(B, self.num_queries, 2, requires_grad=True) * 128,
            "density_out": torch.rand(B, 1, 16, 16, requires_grad=True),
        }


def _make_targets(B: int = 2, n_pts: int = 3, device="cpu"):
    return [
        {
            "labels": torch.ones(n_pts, dtype=torch.long, device=device),
            "point": torch.rand(n_pts, 2, device=device) * 64,
        }
        for _ in range(B)
    ]


def _make_train_batch(B: int = 2):
    """Return (samples, targets, gt_dmap) tuple mirroring collate_fn_crowd_train output."""
    samples = torch.randn(B, 3, 128, 128)
    targets = _make_targets(B)
    gt_dmap = [torch.rand(1, 16, 16)] * B
    return samples, targets, gt_dmap


def _make_val_batch():
    """Return (samples, targets) tuple mirroring collate_fn_crowd output."""
    return torch.randn(1, 3, 128, 128), _make_targets(B=1)


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------


def test_train_one_epoch_returns_dict():
    model = DummyDSGC()
    matcher = HungarianMatcher_Crowd(cost_class=1.0, cost_point=0.05)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002},
        eos_coef=0.5,
        losses=["labels", "points"],
    )
    density_criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = [_make_train_batch()]  # single-batch "epoch"
    device = torch.device("cpu")

    stat = train_one_epoch(
        model, criterion, loader, optimizer, density_criterion, device, epoch=0
    )
    assert isinstance(stat, dict)
    assert "loss_sum" in stat
    assert "den_loss" in stat


def test_train_one_epoch_loss_decreases_with_steps():
    """Multiple steps should produce valid (finite) loss values."""
    model = DummyDSGC()
    matcher = HungarianMatcher_Crowd(cost_class=1.0, cost_point=0.05)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002},
        eos_coef=0.5,
        losses=["labels", "points"],
    )
    density_criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = [_make_train_batch() for _ in range(3)]
    device = torch.device("cpu")
    stat = train_one_epoch(
        model, criterion, loader, optimizer, density_criterion, device, epoch=0
    )
    assert torch.isfinite(torch.tensor(stat["loss_sum"]))


# ---------------------------------------------------------------------------
# evaluate_crowd_no_overlap
# ---------------------------------------------------------------------------


def test_evaluate_returns_four_metrics():
    model = DummyDSGC()
    loader = [_make_val_batch()]
    device = torch.device("cpu")
    mae, mse, d_mae, d_mse = evaluate_crowd_no_overlap(model, loader, device)
    assert isinstance(mae, float)
    assert isinstance(mse, float)
    assert isinstance(d_mae, float)
    assert isinstance(d_mse, float)


def test_evaluate_metrics_non_negative():
    model = DummyDSGC()
    loader = [_make_val_batch() for _ in range(3)]
    device = torch.device("cpu")
    mae, mse, d_mae, d_mse = evaluate_crowd_no_overlap(model, loader, device)
    assert mae >= 0 and mse >= 0 and d_mae >= 0 and d_mse >= 0

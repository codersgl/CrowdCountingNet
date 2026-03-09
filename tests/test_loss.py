"""Tests for HungarianMatcher_Crowd and SetCriterion_Crowd."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crowdcount.models.matcher import HungarianMatcher_Crowd, build_matcher_crowd
from crowdcount.models.criterion import SetCriterion_Crowd


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {
                "set_cost_class": 1.0,
                "set_cost_point": 0.05,
                "eos_coef": 0.5,
                "point_loss_coef": 0.0002,
            }
        }
    )


@pytest.fixture
def dummy_outputs():
    B, Q = 2, 20
    return {
        "pred_logits": torch.rand(B, Q, 2),
        "pred_points": torch.rand(B, Q, 2) * 128,
        "density_out": torch.rand(B, 1, 16, 16),
    }


@pytest.fixture
def dummy_targets():
    n = 5
    return [
        {"labels": torch.ones(n, dtype=torch.long), "point": torch.rand(n, 2) * 64},
        {"labels": torch.ones(n, dtype=torch.long), "point": torch.rand(n, 2) * 64},
    ]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def test_matcher_returns_pairs(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    indices = matcher(dummy_outputs, dummy_targets)
    assert len(indices) == 2  # one per batch item
    for src_idx, tgt_idx in indices:
        assert src_idx.shape == tgt_idx.shape
        assert len(src_idx) == 5  # 5 GT points per item


def test_matcher_valid_src_indices(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    indices = matcher(dummy_outputs, dummy_targets)
    Q = dummy_outputs["pred_logits"].shape[1]
    for src_idx, _ in indices:
        assert (src_idx < Q).all()


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


def test_criterion_loss_keys(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": cfg.model.point_loss_coef},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_ce" in losses
    assert "loss_points" in losses


def test_criterion_losses_are_scalar(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": cfg.model.point_loss_coef},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        assert v.dim() == 0, f"Loss {k} should be scalar"


def test_criterion_loss_values_finite(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": cfg.model.point_loss_coef},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite"

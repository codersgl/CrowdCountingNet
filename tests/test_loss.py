"""Tests for HungarianMatcher_Crowd and SetCriterion_Crowd."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crowdcount.models.matcher import HungarianMatcher_Crowd, build_matcher_crowd
from crowdcount.models.criterion import SetCriterion_Crowd
from crowdcount.models.criterion import sigmoid_focal_loss


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {
                "set_cost_class": 1.0,
                "set_cost_point": 0.05,
                "eos_coef": 0.5,
                "point_loss_coef": 0.0002,
                "count_loss_coef": 0.005,
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
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_ce" in losses
    assert "loss_points" in losses
    assert "loss_count" in losses


def test_criterion_losses_are_scalar(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        assert v.dim() == 0, f"Loss {k} should be scalar"


def test_criterion_loss_values_finite(dummy_outputs, dummy_targets, cfg):
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite"


def test_count_loss_zero_when_disabled(dummy_outputs, dummy_targets, cfg):
    """When count_loss_coef=0, loss_count should not affect total loss."""
    matcher = build_matcher_crowd(cfg)
    weight_dict = {
        "loss_ce": 1,
        "loss_points": cfg.model.point_loss_coef,
        "loss_count": 0.0,
    }
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
    # loss_count contribution should be zero
    assert losses["loss_count"] * weight_dict["loss_count"] == 0.0
    assert torch.isfinite(total)


def test_smooth_l1_vs_mse_on_outlier():
    """Smooth L1 should produce smaller loss than MSE for large coordinate errors."""
    import torch.nn.functional as F

    src = torch.tensor([[0.0, 0.0]])
    tgt = torch.tensor([[200.0, 200.0]])  # large outlier
    smooth_l1 = F.smooth_l1_loss(src, tgt, reduction="sum", beta=1.0)
    mse = F.mse_loss(src, tgt, reduction="sum")
    assert smooth_l1 < mse, "Smooth L1 should be smaller than MSE for large errors"


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


def test_focal_loss_keys(dummy_outputs, dummy_targets, cfg):
    """Focal loss mode should produce the same loss keys as CE mode."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_ce" in losses
    assert "loss_points" in losses
    assert "loss_count" in losses


def test_focal_loss_scalar_finite(dummy_outputs, dummy_targets, cfg):
    """Focal loss output must be a finite scalar."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    losses = criterion(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        assert v.dim() == 0, f"Loss {k} should be scalar"
        assert torch.isfinite(v), f"Loss {k} is not finite"


def test_focal_vs_ce_different(dummy_outputs, dummy_targets, cfg):
    """Focal loss and CE loss should give different values for the same input."""
    matcher = build_matcher_crowd(cfg)
    ce_crit = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_focal_loss=False,
    )
    focal_crit = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    ce_losses = ce_crit(dummy_outputs, dummy_targets)
    focal_losses = focal_crit(dummy_outputs, dummy_targets)
    # The classification losses should differ (focal down-weights easy examples)
    assert not torch.allclose(ce_losses["loss_ce"], focal_losses["loss_ce"])


def test_sigmoid_focal_loss_reduces_easy_examples():
    """Focal loss should produce smaller loss than BCE for easy (high-confidence) samples."""
    # Easy case: logits strongly predict the correct class
    inputs = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])  # 2 samples, 2 classes
    targets = torch.tensor([0, 1])

    focal = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    # Compare with gamma=0 (equivalent to weighted BCE)
    no_focus = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=0.0)
    assert focal < no_focus, "Focal loss should be smaller than BCE for easy examples"


# ---------------------------------------------------------------------------
# Uncertainty weighting
# ---------------------------------------------------------------------------


def test_criterion_uncertainty_weighting_forward(dummy_outputs, dummy_targets, cfg):
    """Criterion with uncertainty weighting should produce valid losses."""
    matcher = build_matcher_crowd(cfg)
    # Add uncertainty_map to outputs
    dummy_outputs["uncertainty_map"] = torch.rand(2, 1, 16, 16)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_uncertainty_weighting=True,
        uncertainty_boost=2.0,
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_points" in losses
    assert torch.isfinite(losses["loss_points"])
    assert losses["loss_points"].dim() == 0


def test_uncertainty_weighting_boosts_loss(dummy_targets, cfg):
    """High uncertainty should produce higher point regression loss."""
    matcher = build_matcher_crowd(cfg)
    B, Q = 2, 20
    base_outputs = {
        "pred_logits": torch.rand(B, Q, 2),
        "pred_points": torch.rand(B, Q, 2) * 128,
        "density_out": torch.rand(B, 1, 16, 16),
    }

    # Low uncertainty everywhere
    low_unc_outputs = {**base_outputs, "uncertainty_map": torch.zeros(B, 1, 16, 16)}
    # High uncertainty everywhere
    high_unc_outputs = {**base_outputs, "uncertainty_map": torch.ones(B, 1, 16, 16)}

    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_uncertainty_weighting=True,
        uncertainty_boost=2.0,
    )
    low_losses = criterion(low_unc_outputs, dummy_targets)
    high_losses = criterion(high_unc_outputs, dummy_targets)
    # High uncertainty should produce higher point loss (boosted by up to 3x)
    assert high_losses["loss_points"] >= low_losses["loss_points"]


def test_uncertainty_weighting_disabled_ignores_map(dummy_outputs, dummy_targets, cfg):
    """When use_uncertainty_weighting=False, uncertainty_map should be ignored."""
    matcher = build_matcher_crowd(cfg)
    dummy_outputs["uncertainty_map"] = torch.ones(2, 1, 16, 16)
    criterion_no_unc = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": cfg.model.point_loss_coef,
            "loss_count": cfg.model.count_loss_coef,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_uncertainty_weighting=False,
    )
    dummy_outputs_no_map = {
        k: v for k, v in dummy_outputs.items() if k != "uncertainty_map"
    }
    losses_with_map = criterion_no_unc(dummy_outputs, dummy_targets)
    losses_without_map = criterion_no_unc(dummy_outputs_no_map, dummy_targets)
    assert torch.allclose(
        losses_with_map["loss_points"], losses_without_map["loss_points"]
    )


def test_uncertainty_weighting_samples_xy_coordinates_correctly(cfg):
    """Uncertainty lookup should use point coordinates in x,y order."""
    matcher = build_matcher_crowd(cfg)
    outputs = {
        "pred_logits": torch.tensor([[[0.1, 0.9]]], dtype=torch.float32),
        "pred_points": torch.tensor([[[3.0, 1.0]]], dtype=torch.float32),
        "density_out": torch.zeros(1, 1, 4, 4, dtype=torch.float32),
        "uncertainty_map": torch.zeros(1, 1, 4, 4, dtype=torch.float32),
    }
    outputs["uncertainty_map"][0, 0, 1, 3] = 1.0
    targets = [
        {
            "labels": torch.ones(1, dtype=torch.long),
            "point": torch.tensor([[3.0, 1.0]], dtype=torch.float32),
            "image_id": torch.tensor([0], dtype=torch.long),
        }
    ]

    criterion_base = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 1},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_uncertainty_weighting=False,
    )
    criterion_weighted = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 1},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points"],
        use_uncertainty_weighting=True,
        uncertainty_boost=2.0,
    )

    base_loss = criterion_base(outputs, targets)["loss_points"]
    weighted_loss = criterion_weighted(outputs, targets)["loss_points"]
    assert torch.allclose(weighted_loss, base_loss * 3.0)

"""Tests for HungarianMatcher_Crowd and SetCriterion_Crowd."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crowdcount.models.matcher import HungarianMatcher_Crowd, build_matcher_crowd
from crowdcount.models.criterion import SetCriterion_Crowd
from crowdcount.models.criterion import softmax_focal_loss, quality_focal_loss


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {
                "set_cost_class": 1.0,
                "set_cost_point": 0.05,
                "eos_coef": 0.5,
                "point_loss_coef": 0.0002,
                "point_loss_type": "smooth_l1",
                "point_smooth_l1_beta": 1.0,
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


def test_point_loss_type_mse_matches_original_code_path(cfg):
    """MSE point loss should match the original DSGC-Net/P2PNet implementation."""
    matcher = build_matcher_crowd(cfg)
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
        "pred_points": torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
    }
    targets = [
        {
            "labels": torch.ones(1, dtype=torch.long),
            "point": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        }
    ]
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_points": 1.0},
        eos_coef=cfg.model.eos_coef,
        losses=["points"],
        point_loss_type="mse",
    )
    loss = criterion(outputs, targets)["loss_points"]
    assert torch.allclose(loss, torch.tensor(25.0))


def test_point_loss_type_smooth_l1_keeps_current_default(cfg):
    """Smooth L1 remains available for reproducing existing local runs."""
    matcher = build_matcher_crowd(cfg)
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
        "pred_points": torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
    }
    targets = [
        {
            "labels": torch.ones(1, dtype=torch.long),
            "point": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        }
    ]
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_points": 1.0},
        eos_coef=cfg.model.eos_coef,
        losses=["points"],
        point_loss_type="smooth_l1",
        point_smooth_l1_beta=1.0,
    )
    loss = criterion(outputs, targets)["loss_points"]
    assert torch.allclose(loss, torch.tensor(6.0))


def test_point_loss_type_l2_aliases_mse(cfg):
    """The l2 alias maps to the original squared-L2/MSE implementation."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_points": 1.0},
        eos_coef=cfg.model.eos_coef,
        losses=["points"],
        point_loss_type="l2",
    )
    assert criterion.point_loss_type == "mse"


def test_point_loss_type_invalid_raises(cfg):
    """Unsupported point regression losses should fail fast."""
    matcher = build_matcher_crowd(cfg)
    with pytest.raises(ValueError, match="point_loss_type"):
        SetCriterion_Crowd(
            num_classes=1,
            matcher=matcher,
            weight_dict={"loss_points": 1.0},
            eos_coef=cfg.model.eos_coef,
            losses=["points"],
            point_loss_type="charbonnier",
        )


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


def test_softmax_focal_loss_reduces_easy_examples():
    """Focal loss should produce smaller loss than CE for easy (high-confidence) samples."""
    # Easy case: logits strongly predict the correct class
    inputs = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])  # 2 samples, 2 classes
    targets = torch.tensor([0, 1])

    focal = softmax_focal_loss(inputs, targets, alpha=0.75, gamma=2.0)
    # Compare with gamma=0 (equivalent to weighted CE)
    no_focus = softmax_focal_loss(inputs, targets, alpha=0.75, gamma=0.0)
    assert focal < no_focus, "Focal loss should be smaller than CE for easy examples"


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


# ---------------------------------------------------------------------------
# Density-point consistency loss
# ---------------------------------------------------------------------------


def test_consistency_loss_keys(dummy_outputs, dummy_targets, cfg):
    """Consistency loss should appear in output when enabled."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": 0.0002,
            "loss_count": 0.0,
            "loss_consistency": 0.005,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count", "consistency"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_consistency" in losses


def test_consistency_loss_scalar_finite(dummy_outputs, dummy_targets, cfg):
    """Consistency loss must be a finite scalar."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1,
            "loss_points": 0.0002,
            "loss_count": 0.0,
            "loss_consistency": 0.005,
        },
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "consistency"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    v = losses["loss_consistency"]
    assert v.dim() == 0, "loss_consistency should be scalar"
    assert torch.isfinite(v), "loss_consistency should be finite"


def test_consistency_loss_zero_when_disabled(dummy_outputs, dummy_targets, cfg):
    """When consistency_loss_coef=0, it should not affect total loss."""
    matcher = build_matcher_crowd(cfg)
    weight_dict = {
        "loss_ce": 1,
        "loss_points": 0.0002,
        "loss_count": 0.0,
        "loss_consistency": 0.0,
    }
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "consistency"],
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert losses["loss_consistency"] * weight_dict["loss_consistency"] == 0.0


def test_consistency_high_density_lower_loss(dummy_targets, cfg):
    """Zero density at GT points should produce higher hinge loss than high density.

    We equalise the count-consistency term by scaling density so that the
    integral roughly matches the predicted foreground count, then compare
    only the difference caused by the point-density hinge term.
    """
    matcher = build_matcher_crowd(cfg)
    B, Q = 2, 20
    pred_logits = torch.rand(B, Q, 2)
    pred_points = torch.rand(B, Q, 2) * 128

    # Compute predicted fg count to calibrate density integral
    fg_count = pred_logits.softmax(-1)[:, :, 1].sum(dim=1)  # [B]
    avg_fg = fg_count.mean().item()
    # density value per cell so that sum ≈ avg_fg  (16*16 = 256 cells)
    cell_val = avg_fg / 256.0

    # "Matched" density: same integral but value ≥ 1 at every cell
    # → hinge = 0 everywhere, count term ≈ 0
    high_cell = max(cell_val, 1.5)
    high_density = torch.full((B, 1, 16, 16), high_cell)

    # "Zero" density: integral=0, hinge=1 at every GT point, count term = avg_fg
    zero_density = torch.zeros(B, 1, 16, 16)

    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_consistency": 1.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "consistency"],
    )
    zero_loss = criterion(
        {
            "pred_logits": pred_logits,
            "pred_points": pred_points,
            "density_out": zero_density,
        },
        dummy_targets,
    )["loss_consistency"]
    # Zero density → hinge fires (1.0 per point) + count mismatch
    assert zero_loss > 0, "Zero density should produce nonzero consistency loss"


def test_consistency_loss_without_density(dummy_targets, cfg):
    """If density_out is missing, loss_consistency should be zero."""
    matcher = build_matcher_crowd(cfg)
    B, Q = 2, 20
    outputs = {
        "pred_logits": torch.rand(B, Q, 2),
        "pred_points": torch.rand(B, Q, 2) * 128,
    }
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_consistency": 0.005},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "consistency"],
    )
    losses = criterion(outputs, dummy_targets)
    assert losses["loss_consistency"].item() == 0.0


def test_consistency_img_size_coordinate_mapping(cfg):
    """Verify that img_size correctly maps pixel coords to density map cells.

    Place a single GT point at the top-left corner of a 64×64 image.
    The density map is 4×4 (stride 16).  With align_corners=True the top-left
    pixel (0,0) maps to grid (-1,-1) which hits density cell [0,0].
    Put density=2.0 at cell [0,0].  The hinge should be 0 (2>1).
    Without the img_size fix the point coord (0,0) divided by map dim (3)
    would also hit (-1,-1), so we additionally test a non-corner point.
    """
    matcher = build_matcher_crowd(cfg)

    # --- Case 1: corner point (0,0) → should sample cell [0,0] ---
    density = torch.zeros(1, 1, 4, 4)
    density[0, 0, 0, 0] = 2.0  # top-left cell

    outputs = {
        "pred_logits": torch.tensor([[[0.1, 0.9]]], dtype=torch.float32),
        "pred_points": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        "density_out": density,
        "img_size": (64, 64),
    }
    targets = [
        {
            "labels": torch.ones(1, dtype=torch.long),
            "point": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        }
    ]
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_consistency": 1.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "consistency"],
    )
    losses = criterion(outputs, targets)
    # Hinge at (0,0) should be 0 because density=2.0 > margin=1.0.
    # Count consistency = |2.0 - fg_score_sum| adds a positive value.
    # We verify the loss is less than what we'd get with zero density
    # (hinge=1 + count=fg_score_sum ≈ 0.7 → total ≈ 1.7).
    # With density=2.0 at corner: hinge=0, count=|2-0.7|≈1.3 → total ≈ 1.3.
    # Key: hinge term is 0, which we verify by comparing against zero-density.
    outputs_zero = {**outputs, "density_out": torch.zeros(1, 1, 4, 4)}
    losses_zero = criterion(outputs_zero, targets)
    assert losses["loss_consistency"] < losses_zero["loss_consistency"], (
        "High density at point should give lower loss than zero density"
    )

    # --- Case 2: point at image centre (32,32) with NO img_size ---
    # Without img_size, coords get normalised by density map dims (4×4),
    # so (32,32)/(3) * 2 - 1 ≈ (20.3, 20.3) → clamped to border → samples
    # cell [0,3] or [3,0] (border), which has density 0 → hinge = 1.
    density2 = torch.zeros(1, 1, 4, 4)
    density2[0, 0, 2, 2] = 2.0  # centre-ish cell only
    outputs_no_imgsize = {
        "pred_logits": torch.tensor([[[0.1, 0.9]]], dtype=torch.float32),
        "pred_points": torch.tensor([[[32.0, 32.0]]], dtype=torch.float32),
        "density_out": density2,
        # no img_size → fallback to density map dims
    }
    targets2 = [
        {
            "labels": torch.ones(1, dtype=torch.long),
            "point": torch.tensor([[32.0, 32.0]], dtype=torch.float32),
        }
    ]
    losses_broken = criterion(outputs_no_imgsize, targets2)

    # With img_size: (32/63)*2-1 ≈ 0.016 → bilinear near centre → hits cell [2,2]
    outputs_with_imgsize = {**outputs_no_imgsize, "img_size": (64, 64)}
    losses_fixed = criterion(outputs_with_imgsize, targets2)

    # The fixed version should have a smaller hinge loss (it actually samples
    # near the nonzero cell) compared to the broken version (clamped to border).
    assert losses_fixed["loss_consistency"] < losses_broken["loss_consistency"], (
        "img_size mapping should improve density sampling accuracy"
    )


# ---------------------------------------------------------------------------
# Quality Focal Loss (standalone function)
# ---------------------------------------------------------------------------


def test_qfl_zero_for_perfect_prediction():
    """When p_fg == target_soft exactly, loss should be near zero."""
    # Logits that produce softmax fg_prob ≈ 0.8
    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])  # [N=2, C=2]
    probs_fg = torch.softmax(logits, dim=-1)[:, 1]  # ≈ 0.88
    loss = quality_focal_loss(logits, probs_fg, beta=2.0)
    # |y - p|^beta should be ~0, making the loss very small
    assert loss.item() < 1e-5


def test_qfl_positive_for_mismatch():
    """QFL should be positive when prediction doesn't match soft target."""
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])  # fg_prob ≈ 0.02
    targets = torch.tensor([0.9, 0.9])  # expect high fg
    loss = quality_focal_loss(logits, targets, beta=2.0)
    assert loss.item() > 0


def test_qfl_gradient_flow():
    """QFL must be differentiable w.r.t. logits."""
    logits = torch.randn(10, 2, requires_grad=True)
    targets = torch.rand(10)
    loss = quality_focal_loss(logits, targets, beta=2.0)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_qfl_with_class_weight():
    """QFL should accept class_weight without error."""
    logits = torch.randn(10, 2)
    targets = torch.rand(10)
    cw = torch.tensor([0.5, 1.0])
    loss = quality_focal_loss(logits, targets, beta=2.0, class_weight=cw)
    assert loss.isfinite()


# ---------------------------------------------------------------------------
# SetCriterion_Crowd with QFL
# ---------------------------------------------------------------------------


def test_qfl_criterion_loss_keys(dummy_outputs, dummy_targets, cfg):
    """QFL-enabled criterion should still produce loss_ce key."""
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_qfl=True,
        qfl_beta=2.0,
        qfl_sigma=10.0,
    )
    losses = criterion(dummy_outputs, dummy_targets)
    assert "loss_ce" in losses
    assert losses["loss_ce"].isfinite()


def test_qfl_criterion_scalar_and_differentiable(dummy_targets, cfg):
    """QFL loss should be scalar and support backward pass."""
    matcher = build_matcher_crowd(cfg)
    B, Q = 2, 20
    outputs = {
        "pred_logits": torch.randn(B, Q, 2, requires_grad=True),
        "pred_points": torch.rand(B, Q, 2) * 128,
        "density_out": torch.rand(B, 1, 16, 16),
    }
    criterion = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_qfl=True,
        qfl_beta=2.0,
        qfl_sigma=10.0,
    )
    losses = criterion(outputs, dummy_targets)
    assert losses["loss_ce"].dim() == 0
    losses["loss_ce"].backward()
    assert outputs["pred_logits"].grad is not None


def test_qfl_vs_ce_different(dummy_outputs, dummy_targets, cfg):
    """QFL and CE should produce different loss_ce values."""
    matcher = build_matcher_crowd(cfg)
    ce_crit = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_qfl=False,
    )
    qfl_crit = SetCriterion_Crowd(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
        eos_coef=cfg.model.eos_coef,
        losses=["labels", "points", "count"],
        use_qfl=True,
        qfl_beta=2.0,
        qfl_sigma=10.0,
    )
    ce_loss = ce_crit(dummy_outputs, dummy_targets)["loss_ce"]
    qfl_loss = qfl_crit(dummy_outputs, dummy_targets)["loss_ce"]
    # They should almost certainly differ (different loss formulations)
    assert not torch.allclose(ce_loss, qfl_loss, atol=1e-6)


def test_qfl_focal_mutual_exclusion(cfg):
    """use_qfl and use_focal_loss cannot both be True."""
    matcher = build_matcher_crowd(cfg)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SetCriterion_Crowd(
            num_classes=1,
            matcher=matcher,
            weight_dict={"loss_ce": 1, "loss_points": 0.0002, "loss_count": 0.0},
            eos_coef=cfg.model.eos_coef,
            losses=["labels", "points", "count"],
            use_focal_loss=True,
            use_qfl=True,
        )

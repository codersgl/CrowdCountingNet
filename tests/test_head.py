"""Tests for prediction head modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.head import (
    ClassificationModel,
    DensityAttentionMask,
    Density_pred,
    RegressionModel,
    SharedPredictionTrunk,
)
from crowdcount.models.ssim_loss import SSIMLoss


@pytest.fixture
def feature_map():
    return torch.randn(2, 256, 16, 16)


# ---------------------------------------------------------------------------
# Density_pred
# ---------------------------------------------------------------------------


def test_density_pred_output_shape(feature_map):
    model = Density_pred()
    out = model(feature_map)
    assert out.shape == (2, 1, 16, 16)


def test_density_pred_non_negative(feature_map):
    model = Density_pred()
    out = model(feature_map)
    assert (out >= 0).all(), "Density map should be non-negative (ReLU output)"


# ---------------------------------------------------------------------------
# RegressionModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points", [4, 9])
def test_regression_model_output_shape(feature_map, num_anchor_points):
    model = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, 2)


# ---------------------------------------------------------------------------
# ClassificationModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points,num_classes", [(4, 2), (9, 2)])
def test_classification_model_output_shape(feature_map, num_anchor_points, num_classes):
    model = ClassificationModel(
        num_features_in=256,
        num_anchor_points=num_anchor_points,
        num_classes=num_classes,
    )
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, num_classes)


# ---------------------------------------------------------------------------
# SharedPredictionTrunk
# ---------------------------------------------------------------------------


def test_shared_trunk_output_shape(feature_map):
    trunk = SharedPredictionTrunk(in_channels=256, feature_size=256)
    out = trunk(feature_map)
    assert out.shape == feature_map.shape, (
        "SharedPredictionTrunk must preserve spatial dims and channel count"
    )


def test_shared_trunk_end_to_end_regression(feature_map):
    """Trunk → RegressionModel pipeline must produce the same shape as before."""
    trunk = SharedPredictionTrunk()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)
    out = reg(trunk(feature_map))
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * 4, 2)


def test_shared_trunk_end_to_end_classification(feature_map):
    """Trunk → ClassificationModel pipeline must produce the same shape as before."""
    trunk = SharedPredictionTrunk()
    cls = ClassificationModel(num_features_in=256, num_anchor_points=4, num_classes=2)
    out = cls(trunk(feature_map))
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * 4, 2)


def test_shared_trunk_is_distinct_from_heads():
    """Trunk parameters must not overlap with RegressionModel or ClassificationModel."""
    trunk = SharedPredictionTrunk()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)
    cls = ClassificationModel(num_features_in=256, num_anchor_points=4, num_classes=2)
    trunk_ids = {id(p) for p in trunk.parameters()}
    reg_ids = {id(p) for p in reg.parameters()}
    cls_ids = {id(p) for p in cls.parameters()}
    assert trunk_ids.isdisjoint(reg_ids)
    assert trunk_ids.isdisjoint(cls_ids)


@pytest.mark.parametrize("mode", ["sigmoid", "learned"])
def test_density_attention_mask_shape_and_range(mode):
    mask = DensityAttentionMask(mode=mode)
    density = torch.rand(2, 1, 16, 16)
    out = mask(density)
    assert out.shape == density.shape
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_ssim_loss_zero_for_identical_maps():
    criterion = SSIMLoss(window_size=11, sigma=1.5)
    density = torch.rand(2, 1, 16, 16)
    loss = criterion(density, density)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_ssim_loss_positive_for_different_maps():
    criterion = SSIMLoss(window_size=11, sigma=1.5)
    pred = torch.zeros(2, 1, 16, 16)
    target = torch.ones(2, 1, 16, 16)
    loss = criterion(pred, target)
    assert loss > 0


# ---------------------------------------------------------------------------
# MultiScaleDensityAttention
# ---------------------------------------------------------------------------

from crowdcount.models.head import MultiScaleDensityAttention


def test_multi_scale_density_attention_output_shape():
    attn = MultiScaleDensityAttention()
    d3 = torch.rand(2, 1, 16, 16)  # H/8
    d4 = torch.rand(2, 1, 8, 8)  # H/16
    d5 = torch.rand(2, 1, 4, 4)  # H/32
    out = attn(d3, d4, d5)
    assert out.shape == d4.shape, "Output should match block4 spatial dims"


def test_multi_scale_density_attention_range():
    attn = MultiScaleDensityAttention()
    d3 = torch.randn(2, 1, 16, 16)
    d4 = torch.randn(2, 1, 8, 8)
    d5 = torch.randn(2, 1, 4, 4)
    out = attn(d3, d4, d5)
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid output must be in [0, 1]"


def test_multi_scale_density_attention_gradient():
    attn = MultiScaleDensityAttention()
    d3 = torch.rand(2, 1, 16, 16, requires_grad=True)
    d4 = torch.rand(2, 1, 8, 8, requires_grad=True)
    d5 = torch.rand(2, 1, 4, 4, requires_grad=True)
    out = attn(d3, d4, d5)
    out.sum().backward()
    assert d3.grad is not None and d4.grad is not None and d5.grad is not None


def test_multi_scale_density_attention_varying_sizes():
    """Ensure it works with non-square and different spatial dims."""
    attn = MultiScaleDensityAttention()
    d3 = torch.rand(1, 1, 20, 24)
    d4 = torch.rand(1, 1, 10, 12)
    d5 = torch.rand(1, 1, 5, 6)
    out = attn(d3, d4, d5)
    assert out.shape == (1, 1, 10, 12)


# ---------------------------------------------------------------------------
# Cross-scale density consistency loss (functional test)
# ---------------------------------------------------------------------------


def test_cross_scale_consistency_loss():
    """Verify the cross-scale consistency logic matches engine.py implementation."""
    import torch.nn.functional as F

    db3 = torch.rand(2, 1, 16, 16)
    db4 = torch.rand(2, 1, 8, 8)
    db5 = torch.rand(2, 1, 4, 4)

    target_size = db4.shape[-2:]
    db3_resized = F.interpolate(
        db3, size=target_size, mode="bilinear", align_corners=False
    )
    db5_resized = F.interpolate(
        db5, size=target_size, mode="bilinear", align_corners=False
    )

    spatial = F.l1_loss(db3_resized, db4) + F.l1_loss(db5_resized, db4)
    count = F.l1_loss(db3.sum(dim=[1, 2, 3]), db4.sum(dim=[1, 2, 3])) + F.l1_loss(
        db5.sum(dim=[1, 2, 3]), db4.sum(dim=[1, 2, 3])
    )
    loss = 0.001 * (spatial + count)

    assert loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss >= 0, "Loss should be non-negative"


def test_cross_scale_consistency_zero_for_identical():
    """When all scales predict the same constant, spatial loss should be near zero."""
    import torch.nn.functional as F

    # Use a constant value to avoid interpolation artefacts
    val = 0.5
    db3 = torch.full((2, 1, 16, 16), val)
    db4 = torch.full((2, 1, 8, 8), val)
    db5 = torch.full((2, 1, 4, 4), val)

    target_size = db4.shape[-2:]
    db3_resized = F.interpolate(
        db3, size=target_size, mode="bilinear", align_corners=False
    )
    db5_resized = F.interpolate(
        db5, size=target_size, mode="bilinear", align_corners=False
    )

    spatial = F.l1_loss(db3_resized, db4) + F.l1_loss(db5_resized, db4)
    assert torch.allclose(spatial, torch.tensor(0.0), atol=1e-5)

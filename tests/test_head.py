"""Tests for prediction head modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.head import (
    ClassificationModel,
    Density_pred,
    RegressionModel,
    SharedPredictionTrunk,
)


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

"""Tests for prediction head modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.head import ClassificationModel, Density_pred, RegressionModel


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

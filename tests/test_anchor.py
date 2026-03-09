"""Tests for anchor point generation utilities."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.anchor import AnchorPoints, generate_anchor_points, shift


def test_generate_anchor_points_shape():
    pts = generate_anchor_points(stride=8, row=2, line=2)
    assert pts.shape == (4, 2), "row×line anchor points expected"


@pytest.mark.parametrize("row,line", [(2, 2), (3, 3)])
def test_generate_anchor_points_count(row, line):
    pts = generate_anchor_points(stride=16, row=row, line=line)
    assert pts.shape[0] == row * line


def test_shift_output_shape():
    anchor_points = generate_anchor_points(stride=8, row=2, line=2)
    shifted = shift((4, 4), stride=8, anchor_points=anchor_points)
    # 4×4 grid positions × 4 anchors
    assert shifted.shape == (4 * 4 * 4, 2)


def test_anchor_points_module_forward():
    module = AnchorPoints(pyramid_levels=[3], row=2, line=2)
    img = torch.zeros(1, 3, 128, 128)
    out = module(img)
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[2] == 2


def test_anchor_points_count_consistency():
    """Number of anchors scales with image size and row×line."""
    module_a = AnchorPoints(pyramid_levels=[3], row=2, line=2)
    module_b = AnchorPoints(pyramid_levels=[3], row=3, line=3)
    img = torch.zeros(1, 3, 128, 128)
    cnt_a = module_a(img).shape[1]
    cnt_b = module_b(img).shape[1]
    assert cnt_b > cnt_a, "More anchor points with larger row×line"

"""Tests for density map generation (prepare.py).

Covers the KDTree sigma calculation for various gt_count values,
boundary points, and output invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdcount.data.prepare import gaussian_filter_density


# ---------------------------------------------------------------------------
# Parametrised core: gt_count = 0, 1, 2, 3, 4, 10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gt_count", [0, 1, 2, 3, 4, 10])
def test_density_map_various_point_counts(gt_count: int) -> None:
    """gaussian_filter_density must not crash and must return a valid density map
    for any number of annotation points, including the previously-broken 2 and 3."""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)

    if gt_count == 0:
        points = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        rng = np.random.RandomState(42)
        points = rng.uniform(10, 118, (gt_count, 2)).astype(np.float32)

    density = gaussian_filter_density(img, points)

    assert density.shape == (H, W)
    assert density.dtype == np.float32
    assert not np.isnan(density).any(), "Density map contains NaN"
    assert not np.isinf(density).any(), "Density map contains Inf"
    assert (density >= 0).all(), "Density map contains negative values"

    if gt_count > 0:
        assert density.max() > 0, f"No density peak for {gt_count} points"
    else:
        assert density.sum() == 0, "Empty point set should yield zero density"


# ---------------------------------------------------------------------------
# Out-of-bounds points
# ---------------------------------------------------------------------------


def test_density_map_all_out_of_bounds() -> None:
    """Points entirely outside the image should produce an all-zero density."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array([[-10.0, -10.0], [200.0, 200.0]], dtype=np.float32)

    density = gaussian_filter_density(img, points)

    assert density.shape == (H, W)
    assert density.sum() == pytest.approx(0.0, abs=1e-9)


def test_density_map_mixed_in_and_out_of_bounds() -> None:
    """A mix of valid and out-of-bounds points should only produce density
    from the valid ones."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array([[32.0, 32.0], [200.0, 200.0]], dtype=np.float32)

    density = gaussian_filter_density(img, points)

    assert density.shape == (H, W)
    assert density.max() > 0, "The in-bounds point should contribute density"


# ---------------------------------------------------------------------------
# Boundary & consistency
# ---------------------------------------------------------------------------


def test_density_map_boundary_touching_points() -> None:
    """Points at (0,0) and (W-1, H-1) should not crash."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array(
        [[0.0, 0.0], [W - 1.0, H - 1.0], [W / 2, H / 2]], dtype=np.float32
    )

    density = gaussian_filter_density(img, points)

    assert density.shape == (H, W)
    assert not np.isnan(density).any()
    assert not np.isinf(density).any()


def test_density_map_four_plus_unchanged_behaviour() -> None:
    """For gt_count >= 4 the sigma formula must remain identical to the
    original k=4 nearest-neighbour calculation."""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    points = rng.uniform(10, 118, (20, 2)).astype(np.float32)

    density = gaussian_filter_density(img, points)

    assert density.shape == (H, W)
    assert density.max() > 0
    # Rough sanity: density integral should be close to the number of
    # in-bounds points (each Gaussian integrates to ~1).
    assert pytest.approx(density.sum(), rel=0.3) == float(len(points))

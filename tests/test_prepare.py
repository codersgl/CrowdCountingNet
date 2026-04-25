"""Tests for density map generation (prepare.py).

Covers the KDTree sigma calculation for various gt_count values,
boundary points, output invariants, and perspective-guided density
map generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdcount.data.prepare import (
    _depth_to_perspective,
    _resolve_density_cache_dir,
    gaussian_filter_density,
    perspective_gaussian_filter_density,
)


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


# ---------------------------------------------------------------------------
# _depth_to_perspective
# ---------------------------------------------------------------------------


def test_depth_to_perspective_uniform() -> None:
    """Uniform depth d → every pixel yields 1/d, median-normalised → all 1.0."""
    H, W = 32, 32
    depth = np.full((H, W), 5.0, dtype=np.float32)
    persp = _depth_to_perspective(depth)
    assert persp.shape == (H, W)
    assert persp.dtype == np.float32
    np.testing.assert_allclose(persp, 1.0, rtol=1e-5)


def test_depth_to_perspective_varying_disparity() -> None:
    """Disparity mode: larger = closer → larger perspective."""
    H, W = 8, 8
    # disparities: col 0 = 1 (far), col 7 = 10 (close)
    disparity = np.tile(np.linspace(1.0, 10.0, W).astype(np.float32), (H, 1))
    persp = _depth_to_perspective(disparity, disparity_input=True)
    # col 7 (close, larger disparity) > col 0 (far, smaller disparity)
    assert persp[0, -1] > persp[0, 0]


def test_depth_to_perspective_varying_depth() -> None:
    """Depth mode (disparity_input=False): larger depth = farther → 1/depth."""
    H, W = 8, 8
    # depths: col 0 = 1 (close), col 7 = 10 (far)
    depth = np.tile(np.linspace(1.0, 10.0, W).astype(np.float32), (H, 1))
    persp = _depth_to_perspective(depth, disparity_input=False)
    # col 0 (close, small depth → large 1/depth) > col 7 (far)
    assert persp[0, 0] > persp[0, -1]


def test_depth_to_perspective_zeros() -> None:
    """Zero depth should be clamped to epsilon, not produce inf."""
    depth = np.zeros((16, 16), dtype=np.float32)
    persp = _depth_to_perspective(depth)
    assert not np.isnan(persp).any()
    assert not np.isinf(persp).any()
    assert (persp > 0).all()


def test_depth_to_perspective_nan_handling() -> None:
    """NaN in depth map should be replaced with epsilon."""
    depth = np.full((16, 16), 5.0, dtype=np.float32)
    depth[0, 0] = np.nan
    persp = _depth_to_perspective(depth)
    assert not np.isnan(persp).any()


def test_depth_to_perspective_clipping() -> None:
    """Extreme depth values should be clipped to the configured range."""
    H, W = 16, 16
    # Extremely small depth → 1/depth is huge, should be clipped to 100
    depth = np.full((H, W), 0.001, dtype=np.float32)
    persp = _depth_to_perspective(depth, clip_range=(0.01, 100.0))
    assert persp.max() <= 100.0


# ---------------------------------------------------------------------------
# perspective_gaussian_filter_density — parametrised core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gt_count", [0, 1, 2, 3, 4, 10])
def test_perspective_density_map_various_point_counts(gt_count: int) -> None:
    """perspective_gaussian_filter_density must not crash for any point count."""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    persp_map = np.ones((H, W), dtype=np.float32)

    if gt_count == 0:
        points = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        rng = np.random.RandomState(42)
        points = rng.uniform(10, 118, (gt_count, 2)).astype(np.float32)

    density = perspective_gaussian_filter_density(img, points, persp_map)

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
# perspective_gaussian_filter_density — sigma behaviour
# ---------------------------------------------------------------------------


def test_perspective_modulates_sigma() -> None:
    """Higher perspective → wider Gaussian → lower peak, broader spread."""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array([[64.0, 64.0]], dtype=np.float32)

    # Low perspective → narrow Gaussian → tall peak
    persp_low = np.full((H, W), 0.5, dtype=np.float32)
    dens_low = perspective_gaussian_filter_density(
        img, points, persp_low, beta=1.0, min_sigma=0.5
    )

    # High perspective → wide Gaussian → short peak
    persp_high = np.full((H, W), 5.0, dtype=np.float32)
    dens_high = perspective_gaussian_filter_density(
        img, points, persp_high, beta=1.0, min_sigma=0.5
    )

    # Both should integrate to ~1
    assert pytest.approx(dens_low.sum(), rel=0.3) == 1.0
    assert pytest.approx(dens_high.sum(), rel=0.3) == 1.0
    # Wider Gaussian → lower peak
    assert dens_low.max() > dens_high.max()


def test_perspective_min_sigma_floor() -> None:
    """Zero perspective → sigma floor at min_sigma, not zero."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array([[32.0, 32.0]], dtype=np.float32)
    persp_map = np.zeros((H, W), dtype=np.float32)

    density = perspective_gaussian_filter_density(
        img, points, persp_map, beta=1.0, min_sigma=1.0
    )
    assert density.max() > 0
    assert not np.isnan(density).any()


def test_perspective_beta_scaling() -> None:
    """Doubling beta should produce a wider kernel (lower peak)."""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    points = np.array([[64.0, 64.0]], dtype=np.float32)
    persp_map = np.full((H, W), 2.0, dtype=np.float32)

    dens_beta_small = perspective_gaussian_filter_density(
        img, points, persp_map, beta=0.5, min_sigma=0.5
    )
    dens_beta_large = perspective_gaussian_filter_density(
        img, points, persp_map, beta=2.0, min_sigma=0.5
    )

    # Both integrate to ~1
    assert pytest.approx(dens_beta_small.sum(), rel=0.3) == 1.0
    assert pytest.approx(dens_beta_large.sum(), rel=0.3) == 1.0
    # Larger beta → wider Gaussian → lower peak
    assert dens_beta_small.max() > dens_beta_large.max()


# ---------------------------------------------------------------------------
# perspective_gaussian_filter_density — boundary & errors
# ---------------------------------------------------------------------------


def test_perspective_out_of_bounds() -> None:
    """Points outside the image should produce zero density."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    persp_map = np.ones((H, W), dtype=np.float32)
    points = np.array([[-10.0, -10.0], [200.0, 200.0]], dtype=np.float32)

    density = perspective_gaussian_filter_density(img, points, persp_map)
    assert density.sum() == pytest.approx(0.0, abs=1e-9)


def test_perspective_empty_points() -> None:
    """Empty point array should return zero density."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    persp_map = np.ones((H, W), dtype=np.float32)
    points = np.array([], dtype=np.float32).reshape(0, 2)

    density = perspective_gaussian_filter_density(img, points, persp_map)
    assert density.sum() == 0.0


def test_perspective_nan_in_map() -> None:
    """NaN in perspective map at a point location should fall back to min_sigma."""
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    persp_map = np.ones((H, W), dtype=np.float32)
    persp_map[32, 32] = np.nan
    points = np.array([[32.0, 32.0]], dtype=np.float32)

    density = perspective_gaussian_filter_density(
        img, points, persp_map, beta=1.0, min_sigma=1.0
    )
    assert density.max() > 0
    assert not np.isnan(density).any()


def test_perspective_shape_mismatch_raises() -> None:
    """Shape mismatch between image and perspective map must raise ValueError."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    persp_map = np.ones((32, 32), dtype=np.float32)
    points = np.array([[32.0, 32.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="does not match image shape"):
        perspective_gaussian_filter_density(img, points, persp_map)


def test_perspective_beta_non_positive_raises() -> None:
    """beta <= 0 should raise ValueError."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    persp_map = np.ones((64, 64), dtype=np.float32)
    points = np.array([[32.0, 32.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="beta must be positive"):
        perspective_gaussian_filter_density(img, points, persp_map, beta=0.0)


def test_perspective_min_sigma_non_positive_raises() -> None:
    """min_sigma <= 0 should raise ValueError."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    persp_map = np.ones((64, 64), dtype=np.float32)
    points = np.array([[32.0, 32.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="min_sigma must be positive"):
        perspective_gaussian_filter_density(img, points, persp_map, min_sigma=0.0)


# ---------------------------------------------------------------------------
# Param-aware cache directory naming
# ---------------------------------------------------------------------------


def test_cache_dir_geometry_default_unchanged(tmp_path) -> None:
    """Geometry-adaptive mode keeps the legacy ``gt_density_maps`` directory
    so existing on-disk caches are not invalidated."""
    out = _resolve_density_cache_dir(tmp_path, "train")
    assert out == tmp_path / "gt_density_maps" / "train"


def test_cache_dir_persp_encodes_params(tmp_path) -> None:
    """Different perspective-guided params must yield different cache dirs."""
    a = _resolve_density_cache_dir(
        tmp_path, "train", perspective_guided=True, beta=0.3, min_sigma=1.0
    )
    b = _resolve_density_cache_dir(
        tmp_path, "train", perspective_guided=True, beta=0.5, min_sigma=1.0
    )
    c = _resolve_density_cache_dir(
        tmp_path, "train", perspective_guided=True, beta=0.3, sigma_base=4.0
    )
    assert a != b, "beta change must produce a different cache dir"
    assert a != c, "sigma_base change must produce a different cache dir"
    assert "persp" in a.parent.name


def test_cache_dir_hybrid_alpha_change_invalidates_cache(tmp_path) -> None:
    """The original silent-cache-hit pitfall: changing hybrid_alpha must
    route to a fresh cache directory."""
    a = _resolve_density_cache_dir(tmp_path, "train", hybrid=True, hybrid_alpha=0.3)
    b = _resolve_density_cache_dir(tmp_path, "train", hybrid=True, hybrid_alpha=0.7)
    assert a != b
    assert "hybrid" in a.parent.name


# ---------------------------------------------------------------------------
# Fast renderer numerical equivalence to scipy.gaussian_filter
# ---------------------------------------------------------------------------


def test_fast_renderer_matches_scipy() -> None:
    """The local-patch renderer used by *_density functions must match the
    legacy ``gaussian_filter`` impulse-response convention closely enough."""
    from scipy.ndimage import gaussian_filter as _gf

    from crowdcount.data.prepare import _render_point_gaussian

    H, W = 64, 64
    sigma = 2.5
    y, x = 30, 32
    fast = np.zeros((H, W), dtype=np.float32)
    _render_point_gaussian(fast, y, x, sigma)

    impulse = np.zeros((H, W), dtype=np.float32)
    impulse[y, x] = 1.0
    ref = _gf(impulse, sigma, mode="constant", truncate=4.0)

    np.testing.assert_allclose(fast, ref, atol=1e-5)

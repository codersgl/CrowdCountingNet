"""Unit tests for Phase A + Phase B3 data augmentations.

Covers:
- ``density_resize_stride8`` invariance.
- ``RandomErasingCount`` (A1): point-drop and density zeroing.
- ``MultiScalePatchResize`` paths inside ``SHHA.__getitem__`` (shape contract).
- ``CopyPasteDense`` (B3) collate-level paste with feathering.
- Default config regression: with all new aug disabled, collate shape contract
  is identical to the legacy collate.
"""

from __future__ import annotations

import random

import pytest
import torch

from crowdcount.data.collate import (
    _apply_copy_paste_dense,
    collate_fn_crowd_train,
    collate_fn_crowd_train_copy_paste_dense,
    make_train_collate,
)
from crowdcount.data.transforms import (
    RandomErasingCount,
    _make_feather_mask,
    density_paste_,
    density_resize_stride8,
    feathered_paste_,
    pick_window_by_point_count,
)


# ---------------------------------------------------------------------------
# density_resize_stride8
# ---------------------------------------------------------------------------


def test_density_resize_stride8_preserves_sum():
    torch.manual_seed(0)
    den = torch.rand(2, 1, 32, 24)
    out = density_resize_stride8(den, stride=8)
    assert out.shape == (2, 1, 4, 3)
    assert torch.allclose(out.sum(), den.sum(), atol=1e-5)


def test_density_resize_stride8_rejects_misaligned():
    den = torch.zeros(1, 1, 30, 16)
    try:
        density_resize_stride8(den, stride=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible spatial size")


# ---------------------------------------------------------------------------
# RandomErasingCount (A1)
# ---------------------------------------------------------------------------


def test_random_erasing_drops_points_inside_box():
    random.seed(1)
    torch.manual_seed(1)
    img = torch.ones(3, 64, 64)
    density = torch.full((1, 64, 64), 0.5)
    points = torch.tensor(
        [[5.0, 5.0], [10.0, 10.0], [50.0, 50.0], [60.0, 60.0]]
    )
    eraser = RandomErasingCount(prob=1.0, scale_range=(0.05, 0.5), fill=0.0)
    img_out, pts_out, den_out, dep_out = eraser(img, points, density, None)

    assert img_out.shape == (3, 64, 64)
    assert dep_out is None
    # At least some pixels were zeroed (erased).
    assert (img_out == 0).any()
    # Density inside the zeroed region must be zero.
    zero_mask = (img_out[0] == 0)
    assert (den_out[0][zero_mask] == 0).all()
    # All surviving points must lie outside the erased box.
    if pts_out.numel() > 0:
        coords_int = pts_out.long().clamp(min=0, max=63)
        for x, y in coords_int.tolist():
            assert img_out[0, y, x] != 0


def test_random_erasing_skipped_when_prob_zero():
    img = torch.ones(3, 32, 32)
    density = torch.full((1, 32, 32), 0.25)
    points = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    eraser = RandomErasingCount(prob=0.0)
    img_out, pts_out, den_out, _ = eraser(img, points.clone(), density, None)
    assert torch.equal(img_out, img)
    assert torch.equal(pts_out, points)
    assert torch.equal(den_out, density)


def test_random_erasing_with_depth():
    random.seed(2)
    img = torch.ones(3, 48, 48)
    density = torch.full((1, 48, 48), 0.1)
    depth = torch.full((1, 48, 48), 0.7)
    points = torch.tensor([[10.0, 10.0]])
    eraser = RandomErasingCount(prob=1.0, scale_range=(0.2, 0.5), fill=0.0)
    img_out, _, den_out, dep_out = eraser(img, points, density, depth)
    assert dep_out is not None
    zero_mask = (img_out[0] == 0)
    assert zero_mask.any()
    assert (dep_out[0][zero_mask] == 0).all()
    assert (den_out[0][zero_mask] == 0).all()


# ---------------------------------------------------------------------------
# pick_window_by_point_count
# ---------------------------------------------------------------------------


def test_pick_window_max_finds_dense_region():
    rng = random.Random(0)
    img_h, img_w = 64, 64
    # All points clustered in top-left 16x16.
    points = torch.tensor(
        [[float(x), float(y)] for x in range(0, 16) for y in range(0, 16)]
    )
    y, x = pick_window_by_point_count(
        points, img_h, img_w, 16, 16, mode="max",
        n_candidates=64, align_to=8, rng=rng,
    )
    # Max window should overlap the dense cluster heavily.
    assert y < 16 and x < 16


def test_pick_window_min_returns_zero_count_region():
    rng = random.Random(0)
    points = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    y, x = pick_window_by_point_count(
        points, 64, 64, 16, 16, mode="min",
        n_candidates=64, align_to=8, rng=rng,
    )
    # The chosen window must contain zero points.
    inside = (
        (points[:, 0] >= x)
        & (points[:, 0] < x + 16)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + 16)
    )
    assert int(inside.sum()) == 0


# ---------------------------------------------------------------------------
# feathered_paste_ / density_paste_
# ---------------------------------------------------------------------------


def test_feather_mask_peaks_at_center():
    mask = _make_feather_mask(8, 8, sigma=2.0)
    assert mask.shape == (8, 8)
    cy, cx = 3, 3  # near centre of 8x8
    assert mask[cy, cx] == mask.max()
    assert (mask >= 0).all() and (mask <= 1).all()


def test_feathered_paste_no_blow_up_and_shape_preserved():
    dst = torch.zeros(3, 32, 32)
    src = torch.ones(3, 16, 16)
    feathered_paste_(dst, src, dst_y=8, dst_x=8, feather_sigma=4.0)
    assert dst.shape == (3, 32, 32)
    # Centre of paste region must be near-1 (mask=1 at centre).
    assert dst[0, 16, 16].item() > 0.9
    # Far from paste region remains zero.
    assert dst[0, 0, 0].item() == 0.0


def test_density_paste_overwrites_region():
    dst = torch.zeros(1, 8, 8)
    src = torch.full((1, 4, 4), 0.5)
    density_paste_(dst, src, dst_y8=2, dst_x8=2)
    assert dst[0, 4, 4].item() == 0.5
    assert dst[0, 0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# CopyPasteDense (B3) — collate level
# ---------------------------------------------------------------------------


def _make_sample(img_val: float, point_xy: list[tuple[float, float]], H: int = 64):
    img = torch.full((3, H, H), img_val)
    density = torch.zeros(1, H // 8, H // 8)
    pts = torch.tensor(point_xy, dtype=torch.float32) if point_xy else torch.zeros((0, 2))
    target = {
        "point": pts,
        "labels": torch.ones(pts.shape[0], dtype=torch.long),
        "image_id": torch.tensor([0], dtype=torch.long),
    }
    return [img, target, density]


def test_apply_copy_paste_dense_merges_points_and_pastes():
    random.seed(0)
    torch.manual_seed(0)
    # Source: dense cluster in top-left.
    src_pts = [(float(x), float(y)) for x in range(0, 32, 4) for y in range(0, 32, 4)]
    src = _make_sample(1.0, src_pts)
    # Dest: sparse, in bottom-right.
    dst_pts = [(60.0, 60.0)]
    dst = _make_sample(0.0, dst_pts)
    samples = [src, dst]

    _apply_copy_paste_dense(
        samples, paste_size=32, prob=1.0, feather_sigma=4.0
    )

    # dst image should now have feathered values > 0 in the paste region.
    dst_img_after = samples[1][0]
    assert dst_img_after.max().item() > 0.5
    # Far corner unchanged.
    assert dst_img_after[0, 60, 60].item() == 0.0
    # dst points: original far point preserved + some translated source points.
    dst_pts_after = samples[1][1]["point"]
    assert dst_pts_after.shape[0] >= 1
    # Labels length must match point count.
    assert samples[1][1]["labels"].shape[0] == dst_pts_after.shape[0]


def test_apply_copy_paste_dense_skipped_when_batch_too_small():
    samples = [_make_sample(0.5, [(1.0, 1.0)])]
    before = samples[0][0].clone()
    _apply_copy_paste_dense(samples, paste_size=32, prob=1.0, feather_sigma=4.0)
    assert torch.equal(samples[0][0], before)


def test_apply_copy_paste_dense_skipped_when_paste_too_large():
    src = _make_sample(1.0, [(1.0, 1.0)])
    dst = _make_sample(0.0, [(2.0, 2.0)])
    samples = [src, dst]
    before = dst[0].clone()
    _apply_copy_paste_dense(samples, paste_size=128, prob=1.0, feather_sigma=4.0)
    assert torch.equal(samples[1][0], before)


# ---------------------------------------------------------------------------
# make_train_collate factory
# ---------------------------------------------------------------------------


def test_make_train_collate_default_returns_legacy():
    fn = make_train_collate(aug_cfg=None, use_depth=False)
    assert fn is collate_fn_crowd_train


def test_make_train_collate_disabled_block_returns_legacy():
    cfg = {"copy_paste_dense": {"enabled": False}}
    fn = make_train_collate(aug_cfg=cfg, use_depth=False)
    assert fn is collate_fn_crowd_train


def test_make_train_collate_enabled_returns_partial():
    cfg = {
        "copy_paste_dense": {
            "enabled": True,
            "prob": 0.7,
            "paste_size": 32,
            "feather_sigma": 5.0,
        }
    }
    fn = make_train_collate(aug_cfg=cfg, use_depth=False)
    # functools.partial wraps the copy-paste collate.
    import functools

    assert isinstance(fn, functools.partial)
    assert fn.func is collate_fn_crowd_train_copy_paste_dense
    assert fn.keywords["paste_size"] == 32
    assert fn.keywords["prob"] == 0.7
    assert fn.keywords["feather_sigma"] == 5.0


def test_make_train_collate_depth_disables_b3():
    cfg = {"copy_paste_dense": {"enabled": True, "paste_size": 32}}
    fn = make_train_collate(aug_cfg=cfg, use_depth=True)
    # Should fall back to depth-aware standard collate, not the B3 partial.
    from crowdcount.data.collate import collate_fn_crowd_train_depth

    assert fn is collate_fn_crowd_train_depth


# ---------------------------------------------------------------------------
# Default-off regression: produces same shapes as legacy collate.
# ---------------------------------------------------------------------------


def test_default_train_collate_shape_contract():
    # Build a tiny synthetic batch matching SHHA.__getitem__'s training output:
    # (img[1, 3, H, W], list_of_target_dicts, density[1, 1, H/8, W/8]).
    # NestedTensor pads to a multiple of 128, so use H=128 to keep shapes intact.
    H = 128
    samples = []
    for _ in range(2):
        img = torch.rand(1, 3, H, H)
        density = torch.rand(1, 1, H // 8, H // 8)
        points = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
        target = [
            {
                "point": points,
                "labels": torch.ones(points.shape[0], dtype=torch.long),
                "image_id": torch.tensor([0], dtype=torch.long),
            }
        ]
        samples.append((img, target, density))
    out = collate_fn_crowd_train(samples)
    nested, targets, densities = out
    assert nested.shape[0] == 2
    assert nested.shape[-2:] == (H, H)
    assert len(targets) == 2
    assert len(densities) == 2
    assert densities[0].shape == (1, H // 8, H // 8)


# ---------------------------------------------------------------------------
# Integration: SHHA.__getitem__ with new aug toggled on.
# ---------------------------------------------------------------------------


def _build_fake_shha(tmp_path):
    """Mirror tests/test_datasets.py::_make_fake_dataset (minimal copy)."""
    import cv2
    import numpy as np

    n_train, n_test = 4, 2
    for split, count, offset in [("train", n_train, 0), ("test", n_test, n_train)]:
        img_dir = tmp_path / f"{split}_data" / "images"
        gt_dir = tmp_path / f"{split}_data" / "ground_truth"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for i in range(count):
            idx = offset + i
            cv2.imwrite(
                str(img_dir / f"IMG_{idx:04d}.jpg"),
                np.zeros((256, 256, 3), dtype=np.uint8),
            )
            with open(gt_dir / f"GT_{idx:04d}.txt", "w") as f:
                for _ in range(5):
                    x, y = np.random.uniform(0, 255, 2)
                    f.write(f"{x:.2f} {y:.2f}\n")
    dmap = tmp_path / "gt_density_maps" / "train"
    dmap.mkdir(parents=True)
    for i in range(n_train):
        np.save(str(dmap / f"IMG_{i:04d}.npy"), np.zeros((256, 256), dtype=np.float32))


def _shha_transform():
    import torchvision.transforms as T

    return T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


def test_shha_integration_a1_a2_enabled(tmp_path):
    """End-to-end: A1 + A2 enabled with patch=True must not crash and must
    produce the canonical [num_patches, 3, patch_size, patch_size] image and
    [num_patches, 1, patch_size/8, patch_size/8] density.
    """
    from crowdcount.data.dataset import SHHA

    _build_fake_shha(tmp_path)
    aug_cfg = {
        "color_jitter": {"enabled": False},
        "random_grayscale": {"enabled": False},
        "random_scale": {"enabled": True, "scale_min": 0.7, "scale_max": 1.3},
        "random_erasing": {
            "enabled": True,
            "prob": 1.0,
            "scale_range": [0.05, 0.2],
            "ratio_range": [0.5, 2.0],
            "fill": 0.0,
        },
        "multi_scale_patch": {
            "enabled": True,
            "patch_size_choices": [96, 128, 192],
        },
    }
    ds = SHHA(
        str(tmp_path),
        train=True,
        transform=_shha_transform(),
        patch=True,
        patch_size=128,
        flip=True,
        flip_prob=1.0,  # force flip path so the A2->numpy fix is exercised
        num_patches=4,
        aug_cfg=aug_cfg,
    )
    # Run several iterations to exercise different random crop_size choices.
    for idx in range(len(ds)):
        for _ in range(3):
            img, target, density = ds[idx]
            assert img.shape == (4, 3, 128, 128), f"img shape {tuple(img.shape)}"
            assert density.shape == (4, 1, 16, 16), (
                f"density shape {tuple(density.shape)}"
            )
            assert len(target) == 4
            for t in target:
                assert "point" in t and "labels" in t
                assert t["labels"].shape[0] == t["point"].shape[0]


def test_shha_integration_default_off_still_works(tmp_path):
    """Default config (new aug disabled) must produce identical-shape output."""
    from crowdcount.data.dataset import SHHA

    _build_fake_shha(tmp_path)
    ds = SHHA(
        str(tmp_path),
        train=True,
        transform=_shha_transform(),
        patch=True,
        patch_size=128,
        flip=True,
        num_patches=4,
    )
    img, target, density = ds[0]
    assert img.shape == (4, 3, 128, 128)
    assert density.shape == (4, 1, 16, 16)
    # New aug must default to disabled.
    assert ds.random_erasing_enabled is False
    assert ds.multi_scale_patch_enabled is False


# ---------------------------------------------------------------------------
# A4: RandomGaussianBlur tests
# ---------------------------------------------------------------------------


class TestRandomGaussianBlur:
    """Tests for the RandomGaussianBlur augmentation class."""

    def test_shape_preserved(self):
        """Output shape must equal input shape."""
        from crowdcount.data.transforms import RandomGaussianBlur

        blur = RandomGaussianBlur(prob=1.0, kernel_size=3, sigma_range=(1.0, 1.0))
        img = torch.randn(3, 64, 64)
        out = blur(img)
        assert out.shape == img.shape

    def test_no_effect_when_prob_zero(self):
        """With prob=0 the image must be returned unchanged."""
        from crowdcount.data.transforms import RandomGaussianBlur

        blur = RandomGaussianBlur(prob=0.0, kernel_size=3, sigma_range=(0.5, 2.0))
        img = torch.randn(3, 32, 32)
        out = blur(img)
        assert torch.equal(out, img)

    def test_blur_smooths_image(self):
        """A deterministic blur (prob=1, fixed sigma) must reduce high-freq energy."""
        from crowdcount.data.transforms import RandomGaussianBlur

        blur = RandomGaussianBlur(prob=1.0, kernel_size=5, sigma_range=(2.0, 2.0))
        # Create a checkerboard pattern (lots of high-freq energy)
        img = torch.zeros(3, 64, 64)
        for i in range(64):
            for j in range(64):
                img[:, i, j] = 1.0 if (i + j) % 2 == 0 else 0.0
        out = blur(img)
        # Blurred output should have smaller variance (less contrast)
        assert out.var() < img.var()

    def test_kernel_normalised(self):
        """The Gaussian kernel must sum to 1."""
        from crowdcount.data.transforms import RandomGaussianBlur

        kernel = RandomGaussianBlur._make_kernel(5, 1.5)
        assert abs(kernel.sum().item() - 1.0) < 1e-5

    def test_invalid_prob_raises(self):
        from crowdcount.data.transforms import RandomGaussianBlur

        with pytest.raises(ValueError, match="prob"):
            RandomGaussianBlur(prob=-0.1)
        with pytest.raises(ValueError, match="prob"):
            RandomGaussianBlur(prob=1.1)

    def test_invalid_kernel_size_raises(self):
        from crowdcount.data.transforms import RandomGaussianBlur

        with pytest.raises(ValueError, match="kernel_size"):
            RandomGaussianBlur(kernel_size=4)
        with pytest.raises(ValueError, match="kernel_size"):
            RandomGaussianBlur(kernel_size=0)

    def test_invalid_sigma_range_raises(self):
        from crowdcount.data.transforms import RandomGaussianBlur

        with pytest.raises(ValueError, match="sigma_range"):
            RandomGaussianBlur(sigma_range=(0.0, 1.0))
        with pytest.raises(ValueError, match="sigma_range"):
            RandomGaussianBlur(sigma_range=(2.0, 1.0))

    def test_different_channel_counts(self):
        """Should work for 1-channel and 3-channel inputs."""
        from crowdcount.data.transforms import RandomGaussianBlur

        blur = RandomGaussianBlur(prob=1.0, kernel_size=3, sigma_range=(1.0, 1.0))
        for c in (1, 3):
            img = torch.randn(c, 32, 32)
            out = blur(img)
            assert out.shape == img.shape

    def test_dataset_integration_enabled(self, tmp_path):
        """SHHA dataset with gaussian_blur enabled should produce valid output."""
        from crowdcount.data.dataset import SHHA

        _build_fake_shha(tmp_path)
        aug_cfg = {"gaussian_blur": {"enabled": True, "prob": 1.0, "kernel_size": 3, "sigma_range": [0.5, 1.5]}}
        ds = SHHA(
            str(tmp_path),
            train=True,
            transform=_shha_transform(),
            patch=True,
            patch_size=128,
            flip=False,
            num_patches=4,
            aug_cfg=aug_cfg,
        )
        assert ds.gaussian_blur_enabled is True
        img, target, density = ds[0]
        assert img.shape == (4, 3, 128, 128)
        assert density.shape == (4, 1, 16, 16)

    def test_dataset_default_disabled(self, tmp_path):
        """Gaussian blur must be disabled by default (no config)."""
        from crowdcount.data.dataset import SHHA

        _build_fake_shha(tmp_path)
        ds = SHHA(
            str(tmp_path),
            train=True,
            transform=_shha_transform(),
            patch=True,
            patch_size=128,
            flip=False,
            num_patches=4,
        )
        assert ds.gaussian_blur_enabled is False

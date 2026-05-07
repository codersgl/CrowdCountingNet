"""Tests for SHHA dataset (mock filesystem, no real images required)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import scipy.io
import torch
import torchvision.transforms as transforms

from crowdcount.data.dataset import SHHA, _load_data, _random_crop
from crowdcount.data.prepare import _find_image_gt_pairs, _load_points


# ---------------------------------------------------------------------------
# Helpers that build a minimal fake dataset directory
# ---------------------------------------------------------------------------


def _make_fake_dataset(root: Path, n_train: int = 4, n_test: int = 2):
    """Create a minimal fake ShanghaiTech-style directory structure.

    Layout mirrors the standard ShanghaiTech Part-A layout::

        root/
          train_data/
            images/         ← IMG_xxxx.jpg
            ground_truth/   ← GT_xxxx.txt  (plain text, no .mat needed in tests)
          test_data/
            images/
            ground_truth/
          gt_density_maps/train/   ← pre-populated so generation is skipped
    """
    for split, count, offset in [("train", n_train, 0), ("test", n_test, n_train)]:
        img_dir = root / f"{split}_data" / "images"
        gt_dir = root / f"{split}_data" / "ground_truth"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        for i in range(count):
            idx = offset + i
            img_name = f"IMG_{idx:04d}.jpg"
            gt_name = f"GT_{idx:04d}.txt"

            img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / img_name), img)

            with open(gt_dir / gt_name, "w") as f:
                for _ in range(3):
                    x, y = np.random.uniform(0, 127, 2)
                    f.write(f"{x:.2f} {y:.2f}\n")

    # Pre-create density maps so the dataset does not try to generate them
    dmap_dir = root / "gt_density_maps" / "train"
    dmap_dir.mkdir(parents=True)
    for i in range(n_train):
        np.save(
            str(dmap_dir / f"IMG_{i:04d}.npy"), np.zeros((128, 128), dtype=np.float32)
        )


def _make_fake_ucf_qnrf(root: Path):
    """Create a minimal UCF-QNRF-style directory with annPoints .mat files."""
    train_points = np.array([[800.0, 400.0], [400.0, 200.0]], dtype=np.float32)
    test_points = np.array([[100.0, 50.0]], dtype=np.float32)
    for split, points in [("Train", train_points), ("Test", test_points)]:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        img = np.zeros((800, 1600, 3), dtype=np.uint8)
        cv2.imwrite(str(split_dir / "img_0001.jpg"), img)
        scipy.io.savemat(str(split_dir / "img_0001_ann.mat"), {"annPoints": points})

    dmap = np.zeros((800, 1600), dtype=np.float32)
    for x, y in train_points:
        dmap[int(y), int(x)] = 1.0
    dmap_dir = root / "gt_density_maps" / "train"
    dmap_dir.mkdir(parents=True)
    np.save(str(dmap_dir / "img_0001.npy"), dmap)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dataset_root(tmp_path):
    _make_fake_dataset(tmp_path)
    return tmp_path


def test_shha_len_train(fake_dataset_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(
        str(fake_dataset_root), train=True, transform=transform, patch=False, flip=False
    )
    assert len(ds) == 4


def test_shha_len_val(fake_dataset_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(str(fake_dataset_root), train=False, transform=transform)
    assert len(ds) == 2


def test_shha_getitem_val(fake_dataset_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(str(fake_dataset_root), train=False, transform=transform)
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3  # C, H, W
    assert isinstance(target, list)
    assert "point" in target[0]


def test_shha_target_labels_all_ones(fake_dataset_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(str(fake_dataset_root), train=False, transform=transform)
    _, target = ds[0]
    labels = target[0]["labels"]
    assert (labels == 1).all()


def test_random_crop_output_shape():
    img = torch.randn(4, 128, 128)  # 4 channels, 128×128
    den = np.array([[32.0, 32.0], [64.0, 64.0]])
    result_img, result_den = _random_crop(img, den, num_patch=2)
    assert result_img.shape == (2, 4, 128, 128)
    assert len(result_den) == 2


def test_shha_augmentation_config_default(fake_dataset_root):
    """Test that default augmentation config values match original hardcoded values."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(
        str(fake_dataset_root),
        train=True,
        transform=transform,
        patch=False,
        flip=False,
    )
    # Verify default values match original hardcoded values
    assert ds.color_jitter_enabled is True
    assert ds.color_jitter_apply_prob == 0.5
    assert ds.color_jitter_brightness == 0.5
    assert ds.color_jitter_contrast == 0.5
    assert ds.color_jitter_saturation == 0.5
    assert ds.color_jitter_hue == 0.5
    assert ds.grayscale_enabled is True
    assert ds.grayscale_prob == 0.5
    assert ds.scale_enabled is True
    assert ds.scale_min == 0.7
    assert ds.scale_max == 1.3
    assert ds.flip_prob == 0.5
    assert ds.num_patches == 4
    assert ds.depth_blur_kernel == 15
    assert ds.depth_blur_sigma == 5.0


def test_shha_augmentation_config_custom(fake_dataset_root):
    """Test that custom augmentation config is correctly applied."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Custom augmentation config
    aug_cfg = {
        "color_jitter": {
            "enabled": False,
            "apply_prob": 0.3,
            "brightness": 0.2,
            "contrast": 0.3,
            "saturation": 0.4,
            "hue": 0.1,
        },
        "random_grayscale": {
            "enabled": False,
            "prob": 0.2,
        },
        "random_scale": {
            "enabled": True,
            "scale_min": 0.8,
            "scale_max": 1.2,
        },
    }

    depth_blur_cfg = {
        "kernel_size": 11,
        "sigma": 3.0,
    }

    ds = SHHA(
        str(fake_dataset_root),
        train=True,
        transform=transform,
        patch=False,
        flip=True,
        aug_cfg=aug_cfg,
        flip_prob=0.7,
        num_patches=8,
        depth_blur_cfg=depth_blur_cfg,
    )

    # Verify custom values are applied
    assert ds.color_jitter_enabled is False
    assert ds.color_jitter_apply_prob == 0.3
    assert ds.color_jitter_brightness == 0.2
    assert ds.color_jitter_contrast == 0.3
    assert ds.color_jitter_saturation == 0.4
    assert ds.color_jitter_hue == 0.1
    assert ds.grayscale_enabled is False
    assert ds.grayscale_prob == 0.2
    assert ds.scale_enabled is True
    assert ds.scale_min == 0.8
    assert ds.scale_max == 1.2
    assert ds.flip_prob == 0.7
    assert ds.num_patches == 8
    assert ds.depth_blur_kernel == 11
    assert ds.depth_blur_sigma == 3.0


def test_random_crop_custom_num_patches():
    """Test that _random_crop respects custom num_patch parameter."""
    img = torch.randn(4, 128, 128)
    den = np.array([[32.0, 32.0], [64.0, 64.0], [96.0, 96.0]])

    # Test with num_patch=3
    result_img, result_den = _random_crop(img, den, num_patch=3, crop_size=64)
    assert result_img.shape == (3, 4, 64, 64)
    assert len(result_den) == 3

    # Test with num_patch=5
    result_img, result_den = _random_crop(img, den, num_patch=5, crop_size=64)
    assert result_img.shape == (5, 4, 64, 64)
    assert len(result_den) == 5


def test_shha_depth_blur_kernel_validation(fake_dataset_root):
    """Test that invalid depth blur kernel sizes raise ValueError."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Test even kernel size
    with pytest.raises(ValueError, match="must be a positive odd number"):
        SHHA(
            str(fake_dataset_root),
            train=True,
            transform=transform,
            depth_blur_cfg={"kernel_size": 14},
        )

    # Test negative kernel size
    with pytest.raises(ValueError, match="must be a positive odd number"):
        SHHA(
            str(fake_dataset_root),
            train=True,
            transform=transform,
            depth_blur_cfg={"kernel_size": -15},
        )

    # Test zero kernel size
    with pytest.raises(ValueError, match="must be a positive odd number"):
        SHHA(
            str(fake_dataset_root),
            train=True,
            transform=transform,
            depth_blur_cfg={"kernel_size": 0},
        )


def test_shha_scale_range_validation(fake_dataset_root):
    """Test that invalid scale ranges raise ValueError."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Test scale_min > scale_max
    with pytest.raises(ValueError, match="scale_min .* must be <= scale_max"):
        SHHA(
            str(fake_dataset_root),
            train=True,
            transform=transform,
            aug_cfg={
                "random_scale": {
                    "scale_min": 1.5,
                    "scale_max": 0.8,
                }
            },
        )

    # Test negative scale_min
    with pytest.raises(ValueError, match="scale_min must be positive"):
        SHHA(
            str(fake_dataset_root),
            train=True,
            transform=transform,
            aug_cfg={
                "random_scale": {
                    "scale_min": -0.5,
                    "scale_max": 1.3,
                }
            },
        )


def test_shha_augmentation_disabled_creates_no_transforms(fake_dataset_root):
    """Test that disabling all augmentations doesn't break the pipeline."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    aug_cfg = {
        "color_jitter": {"enabled": False},
        "random_grayscale": {"enabled": False},
        "random_scale": {"enabled": False},
    }

    ds = SHHA(
        str(fake_dataset_root),
        train=True,
        transform=transform,
        patch=False,
        flip=False,
        aug_cfg=aug_cfg,
    )

    # Should not raise any errors
    result = ds[0]
    # Train mode returns (img, target, density_images) or (img, target, density_images, depth)
    assert len(result) >= 3
    img, target, density_images = result[:3]
    assert isinstance(img, torch.Tensor)
    assert isinstance(target, list)
    assert isinstance(density_images, torch.Tensor)


def test_ucf_qnrf_layout_and_annpoints(tmp_path):
    _make_fake_ucf_qnrf(tmp_path)

    train_pairs = _find_image_gt_pairs(tmp_path, "train")
    test_pairs = _find_image_gt_pairs(tmp_path, "test")

    assert len(train_pairs) == 1
    assert len(test_pairs) == 1
    assert train_pairs[0][0].name == "img_0001.jpg"
    assert train_pairs[0][1].name == "img_0001_ann.mat"

    points = _load_points(train_pairs[0][1])
    assert points.shape == (2, 2)
    np.testing.assert_allclose(points[0], [800.0, 400.0])


def test_ucf_qnrf_long_side_resize_train(tmp_path):
    _make_fake_ucf_qnrf(tmp_path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(
        str(tmp_path),
        train=True,
        transform=transform,
        patch=False,
        flip=False,
        aug_cfg={"random_scale": {"enabled": False}},
        resize_cfg={
            "enabled": True,
            "max_long_side": 1408,
            "keep_aspect_ratio": True,
        },
    )

    img, target, density_images = ds[0]

    assert img.shape[-2:] == (704, 1408)
    assert density_images.shape[-2:] == (88, 176)
    np.testing.assert_allclose(target[0]["point"][0].numpy(), [704.0, 352.0])
    assert float(density_images.sum()) == pytest.approx(2.0, rel=1e-5)


def test_ucf_qnrf_long_side_resize_val(tmp_path):
    _make_fake_ucf_qnrf(tmp_path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = SHHA(
        str(tmp_path),
        train=False,
        transform=transform,
        resize_cfg={
            "enabled": True,
            "max_long_side": 1408,
            "keep_aspect_ratio": True,
        },
    )

    img, target = ds[0]

    assert img.shape[-2:] == (704, 1408)
    np.testing.assert_allclose(target[0]["point"][0].numpy(), [88.0, 44.0])

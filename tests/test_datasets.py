"""Tests for SHHA dataset (mock filesystem, no real images required)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

from crowdcount.data.dataset import SHHA, _load_data, _random_crop


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

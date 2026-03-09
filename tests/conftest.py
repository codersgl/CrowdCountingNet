"""Shared pytest fixtures for crowd counting tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_cfg():
    """Minimal OmegaConf config mirroring configs/config.yaml structure."""
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "gpu_id": 0,
            "num_workers": 0,
            "eval_freq": 1,
            "epochs": 2,
            "clip_max_norm": 0.1,
            "frozen_weights": None,
            "resume": "",
            "start_epoch": 0,
            "checkpoints_dir": "/tmp/test_ckpts",
            "tensorboard_dir": "/tmp/test_runs",
            "data": {
                "dataset": "SHHA",
                "data_root": "",
                "batch_size": 2,
                "patch": True,
                "flip": True,
            },
            "model": {
                "backbone": "vgg16_bn",
                "backbone_type": "vgg",
                "row": 2,
                "line": 2,
                "set_cost_class": 1.0,
                "set_cost_point": 0.05,
                "point_loss_coef": 0.0002,
                "eos_coef": 0.5,
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-4,
                "lr_backbone": 1e-5,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "name": "step_lr",
                "lr_drop": 800,
            },
        }
    )
    return cfg


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_batch():
    """A batch of 2 images, 3 channels, 128×128."""
    return torch.randn(2, 3, 128, 128)


@pytest.fixture
def dummy_targets():
    """Two fake annotation dicts (no real points)."""
    n_pts = 5
    return [
        {
            "point": torch.rand(n_pts, 2) * 64,
            "labels": torch.ones(n_pts, dtype=torch.long),
            "image_id": torch.tensor([1]),
        },
        {
            "point": torch.rand(n_pts, 2) * 64,
            "labels": torch.ones(n_pts, dtype=torch.long),
            "image_id": torch.tensor([2]),
        },
    ]

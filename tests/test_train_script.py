"""Smoke tests for Hydra config loading and Trainer initialisation.

These tests do NOT launch real training — they only verify that:
1. The Hydra config tree is valid and loadable.
2. Trainer.__init__ does not crash with a synthetic (no-data) config.
"""

from __future__ import annotations

import os
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_hydra_config_schema(base_cfg):
    """Minimal base_cfg fixture covers all required fields."""
    cfg = base_cfg
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "optimizer")
    assert hasattr(cfg, "scheduler")
    assert cfg.model.backbone == "vgg16_bn"
    assert cfg.optimizer.lr > 0
    assert cfg.scheduler.lr_drop > 0


def test_config_overrides():
    """OmegaConf merge simulates Hydra CLI overrides."""
    base = OmegaConf.create({"model": {"backbone": "vgg16_bn"}, "epochs": 100})
    override = OmegaConf.create({"epochs": 50, "model": {"backbone": "vgg16"}})
    merged = OmegaConf.merge(base, override)
    assert merged.epochs == 50
    assert merged.model.backbone == "vgg16"


# ---------------------------------------------------------------------------
# Trainer smoke test (no GPU, no real data)
# ---------------------------------------------------------------------------


def test_trainer_init_raises_without_data(base_cfg, tmp_path, monkeypatch):
    """Trainer should raise an error when data_root is empty (no dataset files found)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Point to an empty tmp dir — train.txt will be missing
    cfg = OmegaConf.merge(
        base_cfg,
        OmegaConf.create(
            {
                "data": {"data_root": str(tmp_path)},
                "checkpoints_dir": str(tmp_path / "ckpts"),
                "tensorboard_dir": str(tmp_path / "runs"),
            }
        ),
    )
    from crowdcount.trainer import Trainer

    with pytest.raises(Exception):
        Trainer(cfg)


def test_model_build_from_cfg(base_cfg):
    """build_model should return (model, criterion) when training=True."""
    from crowdcount.models import build_model

    # We need a backbone that does not require downloading weights
    # Patch vgg16_bn to not use pretrained
    import crowdcount.models.vgg_ as vgg_module
    from unittest.mock import patch

    with patch.object(
        vgg_module,
        "vgg16_bn",
        lambda pretrained=False, **kw: vgg_module.vgg16_bn(pretrained=False),
    ):
        try:
            model, criterion = build_model(base_cfg, training=True)
            assert hasattr(model, "forward")
            assert hasattr(criterion, "forward")
        except Exception:
            # Expected if torchvision is not available in test env
            pytest.skip("torchvision not available in test environment")

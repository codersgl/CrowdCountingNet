from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_moecount_model_config_loads() -> None:
    cfg = OmegaConf.load(Path("configs/model/moecount.yaml"))
    assert cfg.name == "moecount"
    assert cfg.output_stride == 8
    assert cfg.backbone.arch == "convnext_tiny"
    assert cfg.backbone.pretrained_path is None
    assert cfg.moe.top_k == 2
    assert cfg.head.final_activation == "softplus"
    assert cfg.head.initial_density == 0.01


def test_moecount_root_config_loads_with_interpolation() -> None:
    cfg = OmegaConf.load(Path("configs/moecount_config.yaml"))
    assert cfg.epochs == 600
    assert int(cfg.scheduler.T_max) == 600
    assert cfg.optimizer.weight_decay == 0.01
    assert cfg.mixed_precision.enabled is True

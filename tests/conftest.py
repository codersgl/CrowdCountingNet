"""Shared pytest fixtures for crowd counting tests."""

from __future__ import annotations

import pytest
import torch
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
            "density_ssim": {
                "enabled": False,
                "weight": 0.005,
                "window_size": 7,
                "sigma": 1.5,
            },
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
                "point_loss_type": "smooth_l1",
                "point_smooth_l1_beta": 1.0,
                "eos_coef": 0.5,
                "use_depth": False,
                "depth": {
                    "encoder": "vitb",
                    "weight_path": "checkpoints/depth_anything_v2_vitb.pth",
                    "mix_init": 1.5,
                    "embed_dim": 128,
                    "num_isf_layers": 1,
                },
                "use_depth_geo": False,
                "use_depth_geo_post": False,
                "depth_geo": {
                    "num_heads": 8,
                    "initial_value": 2.0,
                    "heads_range": 4.0,
                },
                "use_depth_dual_vgg": False,
                "depth_dual_vgg": {
                    "variant": "vgg16_bn",
                    "pretrained": True,
                    "frozen_stages": 0,
                },
                "use_depth_attn": False,
                "depth_attn": {
                    "version": "v1",
                    "mid_ratio": 4,
                    "gate_init": 0.0,
                    "use_tanh_gate": True,
                    "spatial_gate": True,
                    "channel_gate": True,
                    "normalize_depth": True,
                    "require_depth": True,
                },
                "use_depth_cross_attn": False,
                "depth_cross_attn": {
                    "embed_dim": 128,
                    "num_heads": 4,
                    "window_size": 8,
                    "dropout": 0.0,
                    "gate_init": 0.0,
                    "depth_mid_channels": 64,
                    "mode": "window",
                },
                "use_depth_aux": False,
                "depth_aux": {
                    "in_channels": 256,
                    "hidden_channels": 64,
                    "num_layers": 3,
                    "dropout": 0.0,
                    "detach_features": False,
                    "loss_type": "smooth_l1",
                    "smooth_l1_beta": 0.1,
                    "loss_weight": 0.05,
                    "gradient_weight": 0.02,
                    "density_focus_weight": 0.5,
                    "warmup_epochs": 10,
                },
                "use_refine": False,
                "refine": {
                    "num_steps": 2,
                    "hidden_dim": 256,
                    "share_weights": True,
                },
                "gcn_mode": "fixed",
                "gcn_num_supernodes": 8,
                "gcn_supernode_heads": 4,
                "use_freq_head": False,
                "freq_head_kernel": 3,
                "use_density_attention": False,
                "density_attention_mode": "sigmoid",
                "use_subpix_refine": False,
                "subpix_refine": {
                    "top_k": 512,
                    "hidden_dim": 128,
                },
                "use_uncertainty": False,
                "uncertainty_scale": 6.0,
                "uncertainty_boost": 2.0,
                "consistency_loss_coef": 0.0,
                "use_msca_decoder": False,
                "msca_num_heads": 8,
                "msca_num_blocks": 2,
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
def depth_sample():
    """A batch of 2 depth maps, 1 channel, 128×128."""
    return torch.randn(2, 1, 128, 128)


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

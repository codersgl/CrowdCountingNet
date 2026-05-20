"""Tests for the depth auxiliary supervision path."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

import crowdcount.data.loader as data_loader_module
from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.head import DepthAuxHead


class TinyVGGBackbone(nn.Module):
    """Minimal VGG-like backbone with the channel contract DSGCnet expects."""

    def forward(self, samples: torch.Tensor):
        batch_size, _, height, width = samples.shape
        return [
            samples.new_zeros(batch_size, 128, height // 2, width // 2),
            samples.new_zeros(batch_size, 256, height // 4, width // 4),
            samples.new_zeros(batch_size, 512, height // 8, width // 8),
            samples.new_zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_depth_aux_head_shape_and_range():
    head = DepthAuxHead(in_channels=256, hidden_channels=32, num_layers=2)
    features = torch.randn(2, 256, 16, 16)

    prediction = head(features)

    assert prediction.shape == (2, 1, 16, 16)
    assert torch.all(prediction >= 0.0)
    assert torch.all(prediction <= 1.0)


def test_dsgcnet_depth_aux_forward_rgb_only():
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_depth_aux=True,
        depth_aux_cfg=OmegaConf.create(
            {"hidden_channels": 32, "num_layers": 2, "dropout": 0.0}
        ),
    )
    model.eval()
    samples = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        outputs = model(samples)

    assert "depth_aux_out" in outputs
    assert outputs["depth_aux_out"].shape == outputs["density_out"].shape
    assert outputs["pred_logits"].shape[0] == samples.shape[0]


def test_build_dataset_depth_aux_train_only(tmp_path, monkeypatch):
    calls: list[tuple[bool, bool]] = []

    class DummyDataset:
        def __init__(self, *args, train: bool, use_depth: bool, **kwargs) -> None:
            calls.append((train, use_depth))

    monkeypatch.setattr(data_loader_module, "SHHA", DummyDataset)
    cfg = OmegaConf.create(
        {
            "data": {
                "data_root": str(tmp_path),
                "patch": True,
                "flip": True,
            },
            "model": {
                "backbone_type": "vgg",
                "use_depth": False,
                "use_depth_geo": False,
                "use_depth_geo_post": False,
                "use_depth_dual_vgg": False,
                "use_depth_attn": False,
                "use_depth_cross_attn": False,
                "use_depth_aux": True,
                "depth_graph_prior": {"enabled": False},
                "depth": {},
            },
        }
    )

    data_loader_module.build_dataset(cfg)

    assert calls == [(True, True), (False, False)]
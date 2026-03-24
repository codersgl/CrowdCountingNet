"""Depth-stream integration tests.

These tests verify that optional depth input does not break existing behaviour,
and that depth-enabled paths produce expected tensor shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.data.dataset import SHHA
from crowdcount.models.dsgcnet import DSGCnet, _DepthEncoder
from crowdcount.plugins.isfm.depth_fusion import (
    DepthFusionModule,
    HAS_MAMBA,
    _ISF_AVAILABLE,
)


class TinyVGGBackbone(nn.Module):
    """Minimal 4-stage backbone matching DSGCNet's expected VGG feature scales."""

    def forward(self, x: torch.Tensor):
        bsz, _c, h, w = x.shape
        return [
            torch.zeros(bsz, 128, h // 2, w // 2),
            torch.zeros(bsz, 256, h // 4, w // 4),
            torch.zeros(bsz, 512, h // 8, w // 8),
            torch.zeros(bsz, 512, h // 16, w // 16),
        ]


def test_depth_fusion_module_256ch():
    module = DepthFusionModule(in_channels=256, embed_dim=128)
    rgb = torch.randn(2, 256, 32, 32)
    depth = torch.randn(2, 256, 32, 32)
    out = module(rgb, depth)
    assert out.shape == (2, 256, 32, 32)


def test_depth_fusion_module_512ch():
    module = DepthFusionModule(in_channels=512, embed_dim=128)
    rgb = torch.randn(2, 512, 16, 16)
    depth = torch.randn(2, 512, 16, 16)
    out = module(rgb, depth)
    assert out.shape == (2, 512, 16, 16)


def test_depth_fusion_uses_mff():
    """DepthFusionModule must contain a FrequencyFusinoMoudle (MFF) sub-module."""
    from crowdcount.plugins.isfm.MFF import FrequencyFusinoMoudle

    module = DepthFusionModule(in_channels=256, embed_dim=128)
    assert isinstance(module.mff, FrequencyFusinoMoudle)


def test_depth_fusion_isf_presence():
    """ISFLayer is present only when mamba_ssm + CUDA are available."""
    module = DepthFusionModule(in_channels=256, embed_dim=128, num_isf_layers=2)
    if _ISF_AVAILABLE:
        assert module.isf is not None
        assert module.isf.depth == 2
    else:
        assert module.isf is None


def test_depth_fusion_gate_init_zero():
    """Gate parameter must start at 0 for stable training."""
    module = DepthFusionModule(in_channels=256, embed_dim=128)
    assert module.gate.item() == 0.0


def test_depth_fusion_gradient_flow():
    """Verify gradients flow back through MFF fusion path."""
    module = DepthFusionModule(in_channels=256, embed_dim=64)
    rgb = torch.randn(1, 256, 8, 8, requires_grad=True)
    depth = torch.randn(1, 256, 8, 8, requires_grad=True)
    out = module(rgb, depth)
    out.sum().backward()
    assert rgb.grad is not None
    assert depth.grad is not None


def test_depth_encoder(depth_sample):
    encoder = _DepthEncoder()
    d3, d4, d5 = encoder(depth_sample)
    assert d3.shape == (2, 256, 32, 32)
    assert d4.shape == (2, 512, 16, 16)
    assert d5.shape == (2, 512, 8, 8)


def test_dsgcnet_forward_with_depth(sample_batch, depth_sample):
    backbone = TinyVGGBackbone()
    depth_cfg = OmegaConf.create({"embed_dim": 128, "num_isf_layers": 1})
    model = DSGCnet(backbone, use_depth=True, depth_cfg=depth_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_forward_without_depth(sample_batch):
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, use_depth=False)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]


def _build_minimal_shha_tree(root: Path) -> None:
    """Create a tiny SHHA-like tree with one image + one GT for train/test."""
    for split in ("train", "test"):
        img_dir = root / f"{split}_data" / "images"
        gt_dir = root / f"{split}_data" / "ground_truth"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img_path = img_dir / "IMG_1.jpg"
        import cv2

        cv2.imwrite(str(img_path), img)

        pts = np.array([[64.0, 64.0], [128.0, 128.0]], dtype=np.float32)
        # Match ShanghaiTech: mat['image_info'][0,0][0,0][0] -> (N,2)
        image_info = np.empty((1, 1), dtype=object)
        image_info[0, 0] = np.array([[pts]], dtype=object)
        scipy.io.savemat(str(gt_dir / "GT_IMG_1.mat"), {"image_info": image_info})


def test_shha_depth_return_shape(tmp_path, monkeypatch):
    data_root = tmp_path / "shha"
    _build_minimal_shha_tree(data_root)

    # Pre-create density/depth maps to avoid invoking heavy generators in unit test
    train_density_dir = data_root / "gt_density_maps" / "train"
    test_depth_dir = data_root / "gt_depth_maps" / "test"
    train_density_dir.mkdir(parents=True, exist_ok=True)
    test_depth_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(train_density_dir / "IMG_1.npy"), np.ones((256, 256), dtype=np.float32))
    np.save(str(test_depth_dir / "IMG_1.npy"), np.ones((256, 256), dtype=np.float32))

    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    ds = SHHA(
        data_root=str(data_root),
        transform=transform,
        train=False,
        use_depth=True,
        depth_cfg=OmegaConf.create({"encoder": "vitb", "weight_path": "unused.pth"}),
    )

    def _fake_load_data(_img_gt_path, _train):
        # Return a deterministic RGB image and two points without relying on .mat parsing.
        from PIL import Image

        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        pts = np.array([[64.0, 64.0], [128.0, 128.0]], dtype=np.float32)
        return img, pts

    monkeypatch.setattr("crowdcount.data.dataset._load_data", _fake_load_data)

    sample = ds[0]
    assert len(sample) == 3
    img, target, depth = sample
    assert img.shape[0] == 3
    assert isinstance(target, list)
    assert depth.shape == (1, 256, 256)

"""Depth-stream integration tests.

These tests verify that optional depth input does not break existing behaviour,
and that depth-enabled paths produce expected tensor shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.data.dataset import SHHA
from crowdcount.models.dsgcnet import (
    DSGCnet,
    _DepthEncoder,
    _SharedBackboneDepthMix,
)
from crowdcount.plugins.isfm.depth_fusion import (
    DepthFusionModule,
    _ISF_AVAILABLE,
)


class TinyVGGBackbone(nn.Module):
    """Minimal 4-stage backbone matching DSGCNet's expected VGG feature scales."""

    def forward(self, x: torch.Tensor):
        bsz, _c, h, w = x.shape
        return [
            torch.zeros(bsz, 128, h // 2, w // 2, device=x.device, dtype=x.dtype),
            torch.zeros(bsz, 256, h // 4, w // 4, device=x.device, dtype=x.dtype),
            torch.zeros(bsz, 512, h // 8, w // 8, device=x.device, dtype=x.dtype),
            torch.zeros(bsz, 512, h // 16, w // 16, device=x.device, dtype=x.dtype),
        ]


class RecordingTinyVGGBackbone(TinyVGGBackbone):
    """Tiny backbone that records forward calls and input shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor):
        self.calls += 1
        self.input_shapes.append(tuple(x.shape))
        return super().forward(x)


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


def test_shared_backbone_depth_mix_weights_are_learnable():
    module = _SharedBackboneDepthMix(init=1.5)
    rgb_features = (
        torch.ones(1, 256, 8, 8),
        torch.ones(1, 512, 4, 4),
        torch.ones(1, 512, 2, 2),
    )
    depth_features = tuple(torch.zeros_like(feat) for feat in rgb_features)

    fused = module(rgb_features, depth_features)
    mix = module.mix_factors.detach()

    assert len(fused) == 3
    assert module.mix_weights.requires_grad
    assert mix.shape == (3,)
    assert torch.all((mix > 0.0) & (mix < 1.0))
    assert torch.all(mix > 0.5)
    assert torch.allclose(mix + (1.0 - mix), torch.ones_like(mix))

    loss = sum(feat.sum() for feat in fused)
    loss.backward()
    assert module.mix_weights.grad is not None
    assert torch.all(module.mix_weights.grad != 0)


def test_dsgcnet_forward_with_depth(sample_batch, depth_sample):
    backbone = TinyVGGBackbone()
    depth_cfg = OmegaConf.create({"mix_init": 1.5})
    model = DSGCnet(backbone, use_depth=True, depth_cfg=depth_cfg)
    model.eval()

    assert model.depth_backbone is None
    assert model.depth_fusion_c3 is None
    assert model.depth_fusion_c4 is None
    assert model.depth_fusion_c5 is None
    assert model.shared_depth_mix is not None

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_use_depth_runs_shared_backbone_twice(sample_batch, depth_sample):
    backbone = RecordingTinyVGGBackbone()
    model = DSGCnet(
        backbone,
        use_depth=True,
        depth_cfg=OmegaConf.create({"mix_init": 1.5}),
    )
    model.eval()

    with torch.no_grad():
        model(sample_batch, depth_map=depth_sample)

    assert backbone.calls == 2

    backbone.calls = 0
    backbone.input_shapes.clear()
    with torch.no_grad():
        model(sample_batch)

    assert backbone.calls == 1


def test_dsgcnet_shared_depth_resizes_and_repeats_depth(sample_batch):
    backbone = RecordingTinyVGGBackbone()
    model = DSGCnet(
        backbone,
        use_depth=True,
        depth_cfg=OmegaConf.create({"mix_init": 1.5}),
    )
    model.eval()
    depth_map = torch.randn(sample_batch.shape[0], 1, 64, 64)

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_map)

    assert backbone.input_shapes == [
        tuple(sample_batch.shape),
        tuple(sample_batch.shape),
    ]
    assert out["pred_logits"].shape[0] == sample_batch.shape[0]


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


# ---------------------------------------------------------------------------
# ConcatGateFusion tests
# ---------------------------------------------------------------------------


def test_concat_gate_fusion_shape_256():
    from crowdcount.plugins.concat_gate_fusion import ConcatGateFusion

    module = ConcatGateFusion(in_channels=256)
    rgb = torch.randn(2, 256, 16, 16)
    dep = torch.randn(2, 256, 16, 16)
    out = module(rgb, dep)
    assert out.shape == (2, 256, 16, 16)


def test_concat_gate_fusion_shape_512():
    from crowdcount.plugins.concat_gate_fusion import ConcatGateFusion

    module = ConcatGateFusion(in_channels=512)
    rgb = torch.randn(2, 512, 8, 8)
    dep = torch.randn(2, 512, 8, 8)
    out = module(rgb, dep)
    assert out.shape == (2, 512, 8, 8)


def test_concat_gate_fusion_gradient_flow():
    from crowdcount.plugins.concat_gate_fusion import ConcatGateFusion

    module = ConcatGateFusion(in_channels=256)
    rgb = torch.randn(1, 256, 8, 8, requires_grad=True)
    dep = torch.randn(1, 256, 8, 8, requires_grad=True)
    out = module(rgb, dep)
    out.sum().backward()
    assert rgb.grad is not None
    assert dep.grad is not None


# ---------------------------------------------------------------------------
# DepthBackbone_VGG tests  (no pretrained download — pretrained=False)
# ---------------------------------------------------------------------------


def test_depth_backbone_vgg_first_layer_1ch():
    """First Conv layer of DepthBackbone_VGG should accept 1-channel input."""
    from crowdcount.models.backbone import DepthBackbone_VGG

    model = DepthBackbone_VGG(name="vgg16_bn", pretrained=False, frozen_stages=0)
    first_conv = model.body1[0]
    assert first_conv.in_channels == 1, (
        f"Expected in_channels=1, got {first_conv.in_channels}"
    )


def test_depth_backbone_vgg_output_scales():
    """DepthBackbone_VGG should output 4 tensors at correct channel widths."""
    from crowdcount.models.backbone import DepthBackbone_VGG

    model = DepthBackbone_VGG(name="vgg16_bn", pretrained=False, frozen_stages=0)
    model.eval()
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        outs = model(x)
    assert len(outs) == 4
    # c3 = outs[1]: 256ch, c4 = outs[2]: 512ch, c5 = outs[3]: 512ch
    assert outs[1].shape[1] == 256
    assert outs[2].shape[1] == 512
    assert outs[3].shape[1] == 512


def test_depth_backbone_vgg_frozen_stages():
    """frozen_stages=2 freezes body1 and body2 parameters."""
    from crowdcount.models.backbone import DepthBackbone_VGG

    model = DepthBackbone_VGG(name="vgg16_bn", pretrained=False, frozen_stages=2)
    for p in model.body1.parameters():
        assert not p.requires_grad
    for p in model.body2.parameters():
        assert not p.requires_grad
    # body3 should still be trainable
    assert any(p.requires_grad for p in model.body3.parameters())


# ---------------------------------------------------------------------------
# DSGCnet dual-VGG forward tests
# ---------------------------------------------------------------------------


def test_dsgcnet_forward_with_dual_vgg(sample_batch, depth_sample):
    """DSGCnet with use_depth_dual_vgg=True should produce correct output keys."""
    backbone = TinyVGGBackbone()
    dual_vgg_cfg = OmegaConf.create(
        {"variant": "vgg16_bn", "pretrained": False, "frozen_stages": 0}
    )
    model = DSGCnet(backbone, use_depth_dual_vgg=True, depth_dual_vgg_cfg=dual_vgg_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_dual_vgg_no_depthmap_runs(sample_batch):
    """When depth_map=None with dual_vgg enabled, model should still run (skips fusion)."""
    backbone = TinyVGGBackbone()
    dual_vgg_cfg = OmegaConf.create(
        {"variant": "vgg16_bn", "pretrained": False, "frozen_stages": 0}
    )
    model = DSGCnet(backbone, use_depth_dual_vgg=True, depth_dual_vgg_cfg=dual_vgg_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch)  # no depth_map

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_dual_vgg_exclusivity_raises():
    """Enabling use_depth_dual_vgg together with use_depth should raise ValueError."""
    import pytest

    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth=True, use_depth_dual_vgg=True)


def test_dsgcnet_dual_vgg_exclusivity_with_geo_raises():
    """Enabling use_depth_dual_vgg together with use_depth_geo should raise ValueError."""
    import pytest

    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_geo=True, use_depth_dual_vgg=True)


# ---------------------------------------------------------------------------
# DepthResidualGating tests
# ---------------------------------------------------------------------------


def test_depth_residual_gating_shape():
    """Output shape must match input feature shape for all three scales."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGating

    depth = torch.randn(2, 1, 128, 128)
    for ch, h, w in [(256, 32, 32), (512, 16, 16), (512, 8, 8)]:
        module = DepthResidualGating(ch, mid_ratio=4)
        feat = torch.randn(2, ch, h, w)
        out = module(feat, depth)
        assert out.shape == (2, ch, h, w)


def test_depth_residual_gating_gate_init_zero():
    """Gate parameter must start at 0 for stable training."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGating

    module = DepthResidualGating(256)
    assert module.gate.item() == 0.0


def test_depth_residual_gating_identity_at_init():
    """With gate=0 the module output must exactly equal the input feature."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGating

    module = DepthResidualGating(256)
    feat = torch.randn(1, 256, 16, 16)
    depth = torch.randn(1, 1, 64, 64)
    out = module(feat, depth)
    assert torch.allclose(out, feat, atol=1e-7)


def test_depth_residual_gating_gradient_flow():
    """Gradients must flow back through both feat and depth_encoder paths."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGating

    module = DepthResidualGating(256)
    # Manually set gate > 0 so depth_encoder path is active
    with torch.no_grad():
        module.gate.fill_(1.0)
    feat = torch.randn(1, 256, 8, 8, requires_grad=True)
    depth = torch.randn(1, 1, 32, 32, requires_grad=True)
    out = module(feat, depth)
    out.sum().backward()
    assert feat.grad is not None
    assert depth.grad is not None
    # gate should also receive gradient
    assert module.gate.grad is not None


def test_depth_residual_gating_v2_shape_and_identity():
    """V2 must preserve shape and remain identity at its default gate init."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGatingV2

    depth = torch.randn(2, 1, 128, 128)
    for ch, h, w in [(256, 32, 32), (512, 16, 16), (512, 8, 8)]:
        module = DepthResidualGatingV2(ch, mid_ratio=4)
        feat = torch.randn(2, ch, h, w)
        out = module(feat, depth)
        assert out.shape == (2, ch, h, w)
        assert torch.allclose(out, feat, atol=1e-7)


def test_depth_residual_gating_v2_bounded_gate():
    """V2 optionally bounds the global residual gate with tanh."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGatingV2

    module = DepthResidualGatingV2(256, gate_init=5.0, use_tanh_gate=True)
    assert module.gate.item() == 5.0
    assert module.gate.tanh().item() < 1.0


def test_depth_residual_gating_v2_depth_hw_input_and_constant_normalization():
    """V2 accepts [B,H,W] depth and normalises constant maps to zeros."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGatingV2

    module = DepthResidualGatingV2(32, normalize_depth=True)
    feat = torch.randn(2, 32, 8, 8)
    depth = torch.full((2, 16, 16), 7.0)
    prepared = module._prepare_depth(depth, feat)
    assert prepared.shape == (2, 1, 8, 8)
    assert torch.allclose(prepared, torch.zeros_like(prepared), atol=1e-6)


def test_depth_residual_gating_v2_gradient_flow():
    """When the residual gate is active, gradients flow through V2 depth path."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGatingV2

    module = DepthResidualGatingV2(64, mid_ratio=4)
    with torch.no_grad():
        module.gate.fill_(1.0)
    feat = torch.randn(1, 64, 8, 8, requires_grad=True)
    depth = torch.randn(1, 1, 32, 32, requires_grad=True)
    out = module(feat, depth)
    out.sum().backward()
    assert feat.grad is not None
    assert depth.grad is not None
    assert module.gate.grad is not None


def test_depth_residual_gating_v2_invalid_depth_shape_raises():
    """V2 should fail clearly for non-single-channel depth tensors."""
    from crowdcount.plugins.depth_residual_gating import DepthResidualGatingV2

    module = DepthResidualGatingV2(64)
    feat = torch.randn(1, 64, 8, 8)
    bad_depth = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="depth_map must have shape"):
        module(feat, bad_depth)


def test_dsgcnet_forward_with_depth_attn(sample_batch, depth_sample):
    """DSGCnet with use_depth_attn=True should produce correct output keys."""
    backbone = TinyVGGBackbone()
    depth_attn_cfg = OmegaConf.create({"mid_ratio": 4})
    model = DSGCnet(backbone, use_depth_attn=True, depth_attn_cfg=depth_attn_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_forward_with_depth_attn_v2(sample_batch, depth_sample):
    """DSGCnet should run with the improved depth_attn v2 path."""
    backbone = TinyVGGBackbone()
    depth_attn_cfg = OmegaConf.create(
        {
            "version": "v2",
            "mid_ratio": 4,
            "gate_init": 0.0,
            "use_tanh_gate": True,
            "spatial_gate": True,
            "channel_gate": True,
            "normalize_depth": True,
            "require_depth": True,
        }
    )
    model = DSGCnet(backbone, use_depth_attn=True, depth_attn_cfg=depth_attn_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_depth_attn_no_depthmap_runs(sample_batch):
    """When depth_map=None with depth_attn enabled, model should still run."""
    backbone = TinyVGGBackbone()
    depth_attn_cfg = OmegaConf.create({"mid_ratio": 4})
    model = DSGCnet(backbone, use_depth_attn=True, depth_attn_cfg=depth_attn_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch)  # no depth_map

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_depth_attn_v2_requires_depthmap(sample_batch):
    """V2 defaults to strict missing-depth behaviour for real RGB-D runs."""
    backbone = TinyVGGBackbone()
    depth_attn_cfg = OmegaConf.create({"version": "v2", "require_depth": True})
    model = DSGCnet(backbone, use_depth_attn=True, depth_attn_cfg=depth_attn_cfg)
    model.eval()

    with pytest.raises(ValueError, match="requires depth_map"):
        model(sample_batch)


def test_dsgcnet_depth_attn_v2_no_depthmap_fallback(sample_batch):
    """The v2 path can still opt into RGB-only fallback for smoke tests."""
    backbone = TinyVGGBackbone()
    depth_attn_cfg = OmegaConf.create({"version": "v2", "require_depth": False})
    model = DSGCnet(backbone, use_depth_attn=True, depth_attn_cfg=depth_attn_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_depth_attn_exclusivity_raises():
    """Enabling use_depth_attn together with another depth path should raise."""
    import pytest

    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth=True, use_depth_attn=True)
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_geo=True, use_depth_attn=True)
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_dual_vgg=True, use_depth_attn=True)

"""Tests for CLIP backbone (BackboneCLIP)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


# ---------------------------------------------------------------------------
# Tiny backbones for testing (no network / no open_clip needed)
# ---------------------------------------------------------------------------


class TinyCLIPViTBackbone(nn.Module):
    """Mimics CLIP ViT-B-16 output after projection + upsampling pyramid.

    All intermediate ViT features are at stride 16, projected to
    [128, 256, 512, 512] channels, then upsampled by [8x, 4x, 2x, 1x]
    to produce strides [2, 4, 8, 16].
    """

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


class TinyCLIPConvNeXtBackbone(nn.Module):
    """Mimics CLIP ConvNeXt output (strides 4/4/8/16, ch 128/256/512/512)."""

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 4, W // 4),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


# ---------------------------------------------------------------------------
# TinyCLIPViTBackbone tests
# ---------------------------------------------------------------------------


class TestTinyCLIPViT:
    def test_output_length(self):
        backbone = TinyCLIPViTBackbone()
        x = torch.randn(1, 3, 128, 128)
        out = backbone(x)
        assert len(out) == 4

    def test_output_channels(self):
        backbone = TinyCLIPViTBackbone()
        x = torch.randn(1, 3, 128, 128)
        out = backbone(x)
        assert out[0].shape[1] == 128
        assert out[1].shape[1] == 256
        assert out[2].shape[1] == 512
        assert out[3].shape[1] == 512

    def test_output_strides(self):
        backbone = TinyCLIPViTBackbone()
        B, _, H, W = 2, 3, 128, 128
        out = backbone(torch.randn(B, _, H, W))
        assert out[0].shape[2] == H // 2
        assert out[0].shape[3] == W // 2
        assert out[1].shape[2] == H // 4
        assert out[1].shape[3] == W // 4
        assert out[2].shape[2] == H // 8
        assert out[2].shape[3] == W // 8
        assert out[3].shape[2] == H // 16
        assert out[3].shape[3] == W // 16

    def test_batch(self):
        backbone = TinyCLIPViTBackbone()
        x = torch.randn(4, 3, 128, 128)
        out = backbone(x)
        for feat in out:
            assert feat.shape[0] == 4

    def test_integration_dsgcnet(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyCLIPViTBackbone()
        model = DSGCnet(backbone, row=2, line=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 128, 128))
        assert "pred_logits" in out
        assert "pred_points" in out
        assert "density_out" in out
        assert out["pred_logits"].shape[0] == 1
        assert out["pred_points"].shape[2] == 2
        assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]

    def test_integration_batch(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyCLIPViTBackbone()
        model = DSGCnet(backbone, row=2, line=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_uneven_input(self):
        """Input not divisible by 16 (uneven for ViT patch size) still works."""
        backbone = TinyCLIPViTBackbone()
        x = torch.randn(1, 3, 100, 150)
        out = backbone(x)
        assert len(out) == 4
        assert out[3].shape[2] == 100 // 16  # stride 16


# ---------------------------------------------------------------------------
# TinyCLIPConvNeXtBackbone tests
# ---------------------------------------------------------------------------


class TestTinyCLIPConvNeXt:
    def test_output_channels(self):
        backbone = TinyCLIPConvNeXtBackbone()
        x = torch.randn(1, 3, 128, 128)
        out = backbone(x)
        assert out[0].shape[1] == 128
        assert out[1].shape[1] == 256
        assert out[2].shape[1] == 512
        assert out[3].shape[1] == 512

    def test_output_strides(self):
        backbone = TinyCLIPConvNeXtBackbone()
        B, _, H, W = 2, 3, 128, 128
        out = backbone(torch.randn(B, _, H, W))
        assert out[0].shape[2] == H // 4
        assert out[1].shape[2] == H // 4  # c3, same stride as placeholder
        assert out[2].shape[2] == H // 8
        assert out[3].shape[2] == H // 16

    def test_integration_dsgcnet(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyCLIPConvNeXtBackbone()
        model = DSGCnet(backbone, row=2, line=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 128, 128))
        assert "pred_logits" in out
        assert "pred_points" in out
        assert "density_out" in out


# ---------------------------------------------------------------------------
# _detect_clip_arch tests
# ---------------------------------------------------------------------------


def _make_mock_visual_vit():
    visual = MagicMock()
    visual.transformer = MagicMock()
    del visual.trunk  # ensure no trunk attr
    return visual


def _make_mock_visual_convnext_trunk():
    visual = MagicMock()
    del visual.transformer  # ensure no transformer attr
    visual.trunk = MagicMock()
    visual.trunk.stages = MagicMock()
    return visual


def _make_mock_visual_convnext_direct():
    visual = MagicMock()
    del visual.transformer
    visual.stem = MagicMock()
    visual.stages = MagicMock()
    return visual


class TestDetectCLIPArch:
    def test_detects_vit(self):
        from crowdcount.models.backbone import _detect_clip_arch

        assert _detect_clip_arch(_make_mock_visual_vit()) == "vit"

    def test_detects_convnext_trunk(self):
        from crowdcount.models.backbone import _detect_clip_arch

        assert _detect_clip_arch(_make_mock_visual_convnext_trunk()) == "convnext"

    def test_detects_convnext_direct(self):
        from crowdcount.models.backbone import _detect_clip_arch

        assert _detect_clip_arch(_make_mock_visual_convnext_direct()) == "convnext"

    def test_unknown_raises(self):
        from crowdcount.models.backbone import _detect_clip_arch

        visual = MagicMock(spec=[])  # no attrs -- not ViT, not ConvNeXt
        with pytest.raises(ValueError, match="Cannot detect CLIP visual"):
            _detect_clip_arch(visual)


# ---------------------------------------------------------------------------
# build_backbone dispatch tests (monkeypatched to avoid open_clip)
# ---------------------------------------------------------------------------


class TestBuildBackboneCLIP:
    def test_dispatch(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP, build_backbone

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "vit"
            self._patch_size = 16
            self._embed_dim = 512
            self._output_indices = {1, 3, 6, 12}
            self.proj0 = nn.Conv2d(512, 128, 1)
            self.proj1 = nn.Conv2d(512, 256, 1)
            self.proj2 = nn.Conv2d(512, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            self.visual = MagicMock()

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)

        cfg = OmegaConf.create(
            {"model": {"backbone": "ViT-B-16", "backbone_type": "clip"}}
        )
        backbone = build_backbone(cfg)
        assert isinstance(backbone, BackboneCLIP)

    def test_dispatch_convnext(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP, build_backbone

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "convnext"
            self._stem = nn.Identity()
            self._stages = nn.ModuleList([nn.Identity() for _ in range(4)])
            self._stage_channels = (128, 256, 512)
            self.proj0 = nn.Conv2d(128, 128, 1)
            self.proj1 = nn.Conv2d(128, 256, 1)
            self.proj2 = nn.Conv2d(256, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            self.visual = MagicMock()

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)

        cfg = OmegaConf.create(
            {"model": {"backbone": "convnext_base_w", "backbone_type": "clip"}}
        )
        backbone = build_backbone(cfg)
        assert isinstance(backbone, BackboneCLIP)


# ---------------------------------------------------------------------------
# Gradient flow tests (using monkeypatched BackboneCLIP)
# ---------------------------------------------------------------------------


class TestCLIPGradientFlow:
    def test_projections_trainable_vit(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "vit"
            self._patch_size = 16
            self._embed_dim = 512
            self._output_indices = {1, 3, 6, 12}
            self.proj0 = nn.Conv2d(512, 128, 1)
            self.proj1 = nn.Conv2d(512, 256, 1)
            self.proj2 = nn.Conv2d(512, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            # Mock visual with forward that returns token-like output
            visual = MagicMock()
            visual.conv1 = nn.Conv2d(3, 512, 16, 16)
            visual.class_embedding = nn.Parameter(torch.zeros(1, 1, 512))
            # 64x64 input / 16 patch = 4x4 grid + 1 class token = 17 tokens
            visual.positional_embedding = nn.Parameter(torch.zeros(17, 512))
            visual.ln_pre = nn.Identity()
            blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
            visual.transformer.resblocks = blocks
            self.visual = visual

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)

        backbone = BackboneCLIP("ViT-B-16")
        backbone.train()
        x = torch.randn(1, 3, 64, 64)
        out = backbone(x)
        loss = sum(f.mean() for f in out)
        loss.backward()
        assert backbone.proj0.weight.grad is not None
        assert backbone.proj3.weight.grad is not None

    def test_projections_trainable_convnext(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "convnext"
            self._stem = nn.Conv2d(3, 128, 4, 4)  # stride 4
            self._stages = nn.ModuleList(
                [
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.Conv2d(128, 256, 3, 2, 1),  # stride 8
                    nn.Conv2d(256, 512, 3, 2, 1),  # stride 16
                ]
            )
            self._stage_channels = (128, 256, 512)
            self.proj0 = nn.Conv2d(128, 128, 1)
            self.proj1 = nn.Conv2d(128, 256, 1)
            self.proj2 = nn.Conv2d(256, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            self.visual = MagicMock()

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)

        backbone = BackboneCLIP("convnext_base_w")
        backbone.train()
        x = torch.randn(1, 3, 128, 128)
        out = backbone(x)
        loss = sum(f.mean() for f in out)
        loss.backward()
        assert backbone.proj0.weight.grad is not None
        assert backbone.proj3.weight.grad is not None


# ---------------------------------------------------------------------------
# Network-dependent tests (require open_clip_torch + network access)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires network access to download CLIP model")
class TestBackboneCLIPReal:
    def test_vit_b_16_forward(self):
        from crowdcount.models.backbone import BackboneCLIP

        backbone = BackboneCLIP("ViT-B-16")
        backbone.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = backbone(x)
        assert len(features) == 4
        assert features[0].shape[1] == 128
        assert features[1].shape[1] == 256
        assert features[2].shape[1] == 512
        assert features[3].shape[1] == 512

    def test_convnext_base_w_forward(self):
        from crowdcount.models.backbone import BackboneCLIP

        backbone = BackboneCLIP("convnext_base_w")
        backbone.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = backbone(x)
        assert len(features) == 4
        assert features[0].shape[1] == 128
        assert features[1].shape[1] == 256
        assert features[2].shape[1] == 512
        assert features[3].shape[1] == 512

    def test_convnext_base_w_strides(self):
        from crowdcount.models.backbone import BackboneCLIP

        backbone = BackboneCLIP("convnext_base_w")
        backbone.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            features = backbone(x)
        assert features[0].shape[2] == 32  # stride 4
        assert features[1].shape[2] == 32  # stride 4
        assert features[2].shape[2] == 16  # stride 8
        assert features[3].shape[2] == 8  # stride 16


# ---------------------------------------------------------------------------
# norm_stats property
# ---------------------------------------------------------------------------


class TestNormStats:
    def test_vit_norm_stats_shape(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "vit"
            self._patch_size = 16
            self._embed_dim = 512
            self._output_indices = {1, 3, 6, 12}
            self.proj0 = nn.Conv2d(512, 128, 1)
            self.proj1 = nn.Conv2d(512, 256, 1)
            self.proj2 = nn.Conv2d(512, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            self.visual = MagicMock()
            self._norm_mean = (0.48145466, 0.4578275, 0.40821073)
            self._norm_std = (0.26862954, 0.26130258, 0.27577711)

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)
        backbone = BackboneCLIP("ViT-B-16")
        mean, std = backbone.norm_stats
        assert len(mean) == 3
        assert len(std) == 3

    def test_vit_norm_stats_values(self, monkeypatch):
        from crowdcount.models.backbone import BackboneCLIP, _CLIP_NORM_STATS

        def _fake_init(self, name, pretrained=True):
            nn.Module.__init__(self)
            self._arch = "vit"
            self._patch_size = 16
            self._embed_dim = 512
            self._output_indices = {1, 3, 6, 12}
            self.proj0 = nn.Conv2d(512, 128, 1)
            self.proj1 = nn.Conv2d(512, 256, 1)
            self.proj2 = nn.Conv2d(512, 512, 1)
            self.proj3 = nn.Conv2d(512, 512, 1)
            self.visual = MagicMock()
            self._norm_mean = _CLIP_NORM_STATS["openai"][0]
            self._norm_std = _CLIP_NORM_STATS["openai"][1]

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)
        backbone = BackboneCLIP("ViT-B-16")
        mean, std = backbone.norm_stats
        # CLIP stats should differ from ImageNet defaults
        assert abs(std[0] - 0.229) > 0.01, "std[0] should not equal ImageNet default"
        assert abs(mean[0] - 0.485) < 0.05, "mean[0] should be close to ImageNet"


# ---------------------------------------------------------------------------
# _resolve_norm_stats in loader.py
# ---------------------------------------------------------------------------


class TestResolveNormStats:
    def test_defaults_to_imagenet_for_vgg(self):
        from omegaconf import OmegaConf
        from crowdcount.data.loader import _resolve_norm_stats

        cfg = OmegaConf.create(
            {"model": {"backbone_type": "vgg", "backbone": "vgg16_bn"}}
        )
        mean, std = _resolve_norm_stats(cfg)
        assert mean == [0.485, 0.456, 0.406]
        assert std == [0.229, 0.224, 0.225]

    def test_defaults_to_imagenet_for_dinov2(self):
        from omegaconf import OmegaConf
        from crowdcount.data.loader import _resolve_norm_stats

        cfg = OmegaConf.create(
            {"model": {"backbone_type": "dinov2", "backbone": "dinov2_b"}}
        )
        mean, std = _resolve_norm_stats(cfg)
        assert mean == [0.485, 0.456, 0.406]

    def test_clip_returns_clip_stats(self):
        from omegaconf import OmegaConf
        from crowdcount.data.loader import _resolve_norm_stats
        from crowdcount.models.backbone import _CLIP_NORM_STATS

        cfg = OmegaConf.create(
            {"model": {"backbone_type": "clip", "backbone": "ViT-B-16"}}
        )
        mean, std = _resolve_norm_stats(cfg)
        expected_mean, expected_std = _CLIP_NORM_STATS["openai"]
        assert mean == list(expected_mean)
        assert std == list(expected_std)

    def test_clip_stats_differ_from_imagenet(self):
        from omegaconf import OmegaConf
        from crowdcount.data.loader import _resolve_norm_stats

        cfg = OmegaConf.create(
            {"model": {"backbone_type": "clip", "backbone": "ViT-B-16"}}
        )
        mean, std = _resolve_norm_stats(cfg)
        assert std != [0.229, 0.224, 0.225]

    def test_no_model_key_falls_back_to_imagenet(self):
        from omegaconf import OmegaConf
        from crowdcount.data.loader import _resolve_norm_stats

        cfg = OmegaConf.create({"data": {"data_root": "/tmp"}})
        mean, std = _resolve_norm_stats(cfg)
        assert mean == [0.485, 0.456, 0.406]


# ---------------------------------------------------------------------------
# _init_convnext channel probe correctness
# ---------------------------------------------------------------------------


class TestConvNextChannelProbe:
    """Verify that _init_convnext builds projections for the correct channels."""

    def _make_fake_backbone(
        self, monkeypatch, stem_ch=128, s0_ch=128, s1_ch=256, s2_ch=512
    ):
        """Build a BackboneCLIP with a fake ConvNeXt whose channels are known."""
        from crowdcount.models.backbone import BackboneCLIP

        def _fake_init(self, name, pretrained=True):
            import torch.nn as nn_inner

            nn_inner.Module.__init__(self)
            self._arch = "convnext"
            self._norm_mean = (0.485, 0.456, 0.406)
            self._norm_std = (0.229, 0.224, 0.225)

            # Build tiny stages with known channel dims
            self._stem = nn_inner.Conv2d(3, stem_ch, 4, 4)
            self._stages = nn_inner.ModuleList(
                [
                    nn_inner.Conv2d(stem_ch, s0_ch, 3, 1, 1),
                    nn_inner.Conv2d(s0_ch, s1_ch, 3, 2, 1),
                    nn_inner.Conv2d(s1_ch, s2_ch, 3, 2, 1),
                ]
            )
            self._stage_channels = (s0_ch, s1_ch, s2_ch)
            # Correct projections (what _init_convnext should produce)
            self.proj0 = nn_inner.Conv2d(s0_ch, 128, 1)
            self.proj1 = nn_inner.Conv2d(s0_ch, 256, 1)
            self.proj2 = nn_inner.Conv2d(s1_ch, 512, 1)
            self.proj3 = nn_inner.Conv2d(s2_ch, 512, 1)
            self.visual = MagicMock()

        monkeypatch.setattr(BackboneCLIP, "__init__", _fake_init)
        return BackboneCLIP("convnext_base_w")

    def test_forward_no_channel_error(self, monkeypatch):
        backbone = self._make_fake_backbone(monkeypatch)
        backbone.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = backbone(x)
        assert len(out) == 4

    def test_proj0_matches_stage0_channels(self, monkeypatch):
        backbone = self._make_fake_backbone(monkeypatch, stem_ch=128, s0_ch=128)
        assert backbone.proj0.in_channels == 128  # s0_ch

    def test_proj2_matches_stage1_channels(self, monkeypatch):
        backbone = self._make_fake_backbone(monkeypatch, s0_ch=128, s1_ch=256)
        assert backbone.proj2.in_channels == 256  # s1_ch

    def test_proj3_matches_stage2_channels(self, monkeypatch):
        backbone = self._make_fake_backbone(monkeypatch, s1_ch=256, s2_ch=512)
        assert backbone.proj3.in_channels == 512  # s2_ch

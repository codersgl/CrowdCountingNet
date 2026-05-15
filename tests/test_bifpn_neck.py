"""Tests for SPD-BiFPN neck integration."""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.models.neck import SPDBiFPNNeck


@pytest.fixture
def feat_c3() -> torch.Tensor:
    return torch.randn(2, 256, 32, 32)


@pytest.fixture
def feat_c4() -> torch.Tensor:
    return torch.randn(2, 512, 16, 16)


@pytest.fixture
def feat_c5() -> torch.Tensor:
    return torch.randn(2, 512, 8, 8)


class TestSPDBiFPNNeck:
    def test_output_shape(self, feat_c3, feat_c4, feat_c5):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        out = neck([feat_c3, feat_c4, feat_c5])
        assert out.shape == (2, 256, 16, 16)

    def test_output_shape_multiple_blocks(self, feat_c3, feat_c4, feat_c5):
        neck = SPDBiFPNNeck(
            C3_size=256,
            C4_size=512,
            C5_size=512,
            num_blocks=2,
        )
        out = neck([feat_c3, feat_c4, feat_c5])
        assert out.shape == (2, 256, 16, 16)

    def test_return_intermediates(self, feat_c3, feat_c4, feat_c5):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        out, (p3, p4, p5) = neck(
            [feat_c3, feat_c4, feat_c5], return_intermediates=True
        )
        assert out.shape == (2, 256, 16, 16)
        assert p3.shape == (2, 256, 32, 32)
        assert p4.shape == (2, 256, 16, 16)
        assert p5.shape == (2, 256, 8, 8)

    def test_vgg_like_equal_c4_c5_resolution(self, feat_c3, feat_c4):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        c5 = torch.randn(2, 512, 16, 16)
        out = neck([feat_c3, feat_c4, c5])
        assert out.shape == (2, 256, 16, 16)

    def test_eval_resolution(self):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        neck.eval()
        c3 = torch.randn(1, 256, 128, 192)
        c4 = torch.randn(1, 512, 64, 96)
        c5 = torch.randn(1, 512, 32, 48)
        with torch.no_grad():
            out = neck([c3, c4, c5])
        assert out.shape == (1, 256, 64, 96)
        assert torch.isfinite(out).all()

    def test_handles_zero_input(self):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        inputs = [
            torch.zeros(1, 256, 32, 32),
            torch.zeros(1, 512, 16, 16),
            torch.zeros(1, 512, 8, 8),
        ]
        out = neck(inputs)
        assert out.shape == (1, 256, 16, 16)
        assert torch.isfinite(out).all()

    def test_backward_pass(self, feat_c3, feat_c4, feat_c5):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        feat_c3.requires_grad_(True)
        out = neck([feat_c3, feat_c4, feat_c5])
        out.sum().backward()
        assert feat_c3.grad is not None

    def test_normalized_fusion_weights_are_finite(self):
        neck = SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512)
        fusion = dict(neck.named_modules())["blocks.0.p4_td_fusion"]
        weights = cast(torch.Tensor, getattr(fusion, "normalized_weights"))
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-4)

    def test_rejects_empty_stack(self):
        with pytest.raises(ValueError, match="num_blocks"):
            SPDBiFPNNeck(C3_size=256, C4_size=512, C5_size=512, num_blocks=0)


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2),
            torch.zeros(batch_size, 256, height // 4, width // 4),
            torch.zeros(batch_size, 512, height // 8, width // 8),
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ]


class TestSPDBiFPNIntegration:
    def test_dsgcnet_with_bifpn_neck(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyVGGBackbone()
        model = DSGCnet(backbone, row=2, line=2, use_bifpn_neck=True)
        model.eval()
        x = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert "pred_logits" in out
        assert "pred_points" in out
        assert "density_out" in out
        assert out["pred_logits"].shape[0] == 1
        assert out["pred_points"].shape[2] == 2
        assert out["density_out"].shape == (1, 1, 16, 16)

    def test_dsgcnet_bifpn_mutual_exclusion(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyVGGBackbone()
        with pytest.raises(ValueError, match="mutually exclusive"):
            DSGCnet(backbone, row=2, line=2, use_bifpn_neck=True, use_dap_neck=True)

    def test_dsgcnet_bifpn_custom_config(self):
        from crowdcount.models.dsgcnet import DSGCnet

        cfg = OmegaConf.create(
            {
                "num_blocks": 2,
                "use_spd_downsample": True,
                "use_depthwise_refine": False,
                "eps": 1e-3,
            }
        )
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone,
            row=2,
            line=2,
            use_bifpn_neck=True,
            bifpn_neck_cfg=cfg,
        )
        model.eval()
        x = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["density_out"].shape == (1, 1, 16, 16)

    def test_dsgcnet_bifpn_with_post_acdr(self):
        from crowdcount.models.dsgcnet import DSGCnet

        cfg = OmegaConf.create(
            {
                "enabled": True,
                "large_kernel": 5,
                "dilation": 1,
                "hidden_ratio": 4,
                "gate_init": 0.0,
            }
        )
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone,
            row=2,
            line=2,
            use_bifpn_neck=True,
            neck_acdr_cfg=cfg,
        )
        assert model.neck_acdr is not None
        model.eval()
        x = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["density_out"].shape == (1, 1, 16, 16)
"""Tests for the P2PNeXt-style decoder neck."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.models.neck import P2PNeXtDecoder


@pytest.fixture
def feat_c3() -> torch.Tensor:
    return torch.randn(2, 256, 32, 32)


@pytest.fixture
def feat_c4() -> torch.Tensor:
    return torch.randn(2, 512, 16, 16)


@pytest.fixture
def feat_c5() -> torch.Tensor:
    return torch.randn(2, 512, 8, 8)


@pytest.mark.parametrize("output_level", ["p3", "p4", "p5", "fused"])
def test_p2pnext_decoder_output_shape(output_level, feat_c3, feat_c4, feat_c5):
    neck = P2PNeXtDecoder(
        C3_size=256,
        C4_size=512,
        C5_size=512,
        output_level=output_level,
    )
    out = neck([feat_c3, feat_c4, feat_c5])
    assert out.shape == (2, 256, 16, 16)
    assert torch.isfinite(out).all()


def test_p2pnext_decoder_return_intermediates(feat_c3, feat_c4, feat_c5):
    neck = P2PNeXtDecoder(C3_size=256, C4_size=512, C5_size=512)
    out, (p3, p4, p5) = neck(
        [feat_c3, feat_c4, feat_c5], return_intermediates=True
    )
    assert out.shape == (2, 256, 16, 16)
    assert p3.shape == (2, 256, 32, 32)
    assert p4.shape == (2, 256, 16, 16)
    assert p5.shape == (2, 256, 8, 8)


def test_p2pnext_decoder_rejects_invalid_output_level():
    with pytest.raises(ValueError, match="output_level"):
        P2PNeXtDecoder(
            C3_size=256,
            C4_size=512,
            C5_size=512,
            output_level="p6",
        )


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2),
            torch.zeros(batch_size, 256, height // 4, width // 4),
            torch.zeros(batch_size, 512, height // 8, width // 8),
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_dsgcnet_with_p2pnext_neck():
    from crowdcount.models.dsgcnet import DSGCnet

    cfg = OmegaConf.create({"output_level": "fused"})
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_p2pnext_neck=True,
        p2pnext_neck_cfg=cfg,
    )
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


def test_dsgcnet_p2pnext_neck_mutual_exclusion():
    from crowdcount.models.dsgcnet import DSGCnet

    with pytest.raises(ValueError, match="mutually exclusive"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            use_p2pnext_neck=True,
            use_bifpn_neck=True,
        )
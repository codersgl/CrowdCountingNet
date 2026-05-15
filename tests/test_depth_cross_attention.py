"""Tests for depth cross-attention fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.plugins.depth_cross_attention import DepthCrossAttentionFusion


class TinyVGGBackbone(nn.Module):
    """Minimal 4-stage backbone matching DSGCNet's expected VGG scales."""

    def forward(self, x: torch.Tensor):
        bsz, _channels, height, width = x.shape
        return [
            torch.zeros(bsz, 128, height // 2, width // 2, device=x.device),
            torch.zeros(bsz, 256, height // 4, width // 4, device=x.device),
            torch.zeros(bsz, 512, height // 8, width // 8, device=x.device),
            torch.zeros(bsz, 512, height // 16, width // 16, device=x.device),
        ]


def _module(**kwargs) -> DepthCrossAttentionFusion:
    defaults = {
        "in_channels": 256,
        "embed_dim": 64,
        "num_heads": 4,
        "window_size": 4,
        "depth_mid_channels": 16,
    }
    defaults.update(kwargs)
    return DepthCrossAttentionFusion(**defaults)


def test_depth_cross_attention_shape_with_window_padding() -> None:
    module = _module(window_size=7)
    rgb = torch.randn(2, 256, 15, 17)
    depth = torch.randn(2, 1, 61, 67)

    out = module(rgb, depth)

    assert out.shape == rgb.shape


def test_depth_cross_attention_identity_at_init() -> None:
    module = _module(gate_init=0.0)
    rgb = torch.randn(1, 256, 8, 8)
    depth = torch.randn(1, 1, 32, 32)

    out = module(rgb, depth)

    assert torch.allclose(out, rgb, atol=1e-7)


def test_depth_cross_attention_gradient_flow_when_gate_active() -> None:
    module = _module()
    with torch.no_grad():
        module.gate.fill_(1.0)

    rgb = torch.randn(1, 256, 8, 8, requires_grad=True)
    depth = torch.randn(1, 1, 32, 32, requires_grad=True)
    out = module(rgb, depth)
    out.sum().backward()

    assert rgb.grad is not None
    assert depth.grad is not None
    assert module.gate.grad is not None


def test_depth_cross_attention_global_mode_shape() -> None:
    module = _module(mode="global")
    rgb = torch.randn(1, 256, 6, 5)
    depth = torch.randn(1, 1, 24, 20)

    out = module(rgb, depth)

    assert out.shape == rgb.shape


def test_depth_cross_attention_invalid_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DepthCrossAttentionFusion(embed_dim=62, num_heads=4)


def test_dsgcnet_forward_with_depth_cross_attention() -> None:
    backbone = TinyVGGBackbone()
    depth_cross_attn_cfg = OmegaConf.create(
        {
            "embed_dim": 64,
            "num_heads": 4,
            "window_size": 4,
            "depth_mid_channels": 16,
            "mode": "window",
        }
    )
    model = DSGCnet(
        backbone,
        use_depth_cross_attn=True,
        depth_cross_attn_cfg=depth_cross_attn_cfg,
    )
    model.eval()
    sample = torch.zeros(1, 3, 128, 128)
    depth = torch.zeros(1, 1, 128, 128)

    with torch.no_grad():
        out = model(sample, depth_map=depth)

    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["pred_logits"].shape[0] == sample.shape[0]
    assert out["pred_points"].shape[0] == sample.shape[0]
    assert out["density_out"].shape[0] == sample.shape[0]


def test_dsgcnet_depth_cross_attention_no_depthmap_runs() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create(
        {
            "embed_dim": 64,
            "num_heads": 4,
            "window_size": 4,
            "depth_mid_channels": 16,
        }
    )
    model = DSGCnet(backbone, use_depth_cross_attn=True, depth_cross_attn_cfg=cfg)
    model.eval()

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 1


def test_dsgcnet_depth_cross_attention_exclusivity_raises() -> None:
    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth=True, use_depth_cross_attn=True)
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_geo=True, use_depth_cross_attn=True)
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_dual_vgg=True, use_depth_cross_attn=True)
    with pytest.raises(ValueError, match="At most one depth fusion path"):
        DSGCnet(backbone, use_depth_attn=True, use_depth_cross_attn=True)


def test_dsgcnet_depth_cross_attention_msca_decoder_raises() -> None:
    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="use_msca_decoder"):
        DSGCnet(backbone, use_depth_cross_attn=True, use_msca_decoder=True)
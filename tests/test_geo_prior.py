from __future__ import annotations

import pytest
import torch
from crowdcount.plugins.geo_prior import GeoPriorGen, DepthGeoPriorAttention
from crowdcount.models.dsgcnet import DSGCnet
from omegaconf import OmegaConf
import torch.nn as nn


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


def test_geo_prior_gen_output_shapes():
    B, num_heads = 2, 8
    H, W = 16, 16
    embed_dim = 256
    geo = GeoPriorGen(embed_dim=embed_dim, num_heads=num_heads)
    depth_map = torch.rand(
        B, 1, 32, 32
    )  # different size, should be interpolated inside

    (sin, cos), (mask_h, mask_w) = geo((H, W), depth_map)

    head_dim_half = embed_dim // num_heads // 2
    head_dim = head_dim_half * 2

    # sin, cos should be [H, W, head_dim] -> [16, 16, 32]
    assert sin.shape == (H, W, head_dim)
    assert cos.shape == (H, W, head_dim)

    # mask_h corresponds to height dimension computation => W pieces, each HxH
    # So expected shape: [B, num_heads, W, H, H]
    assert mask_h.shape == (B, num_heads, W, H, H)

    # mask_w corresponds to width dimension => H pieces, each WxW
    # So expected shape: [B, num_heads, H, W, W]
    assert mask_w.shape == (B, num_heads, H, W, W)


def test_depth_geo_attn_256ch():
    attn = DepthGeoPriorAttention(in_channels=256, num_heads=8)
    rgb = torch.randn(2, 256, 32, 32)
    depth = torch.randn(2, 1, 32, 32)
    out = attn(rgb, depth)
    assert out.shape == (2, 256, 32, 32)


def test_depth_geo_attn_512ch():
    attn = DepthGeoPriorAttention(in_channels=512, num_heads=8)
    rgb = torch.randn(2, 512, 16, 16)
    depth = torch.randn(2, 1, 16, 16)
    out = attn(rgb, depth)
    assert out.shape == (2, 512, 16, 16)


def test_depth_geo_attn_gate_zero_init():
    attn = DepthGeoPriorAttention(in_channels=256, num_heads=8)
    assert attn.gate.item() == 0.0


def test_depth_geo_attn_gradient_flow():
    attn = DepthGeoPriorAttention(in_channels=256, num_heads=8)
    rgb = torch.randn(1, 256, 16, 16, requires_grad=True)
    depth = torch.randn(1, 1, 16, 16, requires_grad=True)
    out = attn(rgb, depth)
    out.sum().backward()
    assert rgb.grad is not None
    # depth parameter affects mask which modifies attention weights
    assert depth.grad is not None


def test_dsgcnet_use_depth_geo_forward(sample_batch, depth_sample):
    backbone = TinyVGGBackbone()
    depth_geo_cfg = OmegaConf.create(
        {"num_heads": 8, "initial_value": 2.0, "heads_range": 4.0}
    )
    model = DSGCnet(backbone, use_depth_geo=True, depth_geo_cfg=depth_geo_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=depth_sample)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]
    assert out["pred_points"].shape[0] == sample_batch.shape[0]
    assert out["density_out"].shape[0] == sample_batch.shape[0]


def test_dsgcnet_depth_geo_no_depth_input(sample_batch):
    backbone = TinyVGGBackbone()
    depth_geo_cfg = OmegaConf.create(
        {"num_heads": 8, "initial_value": 2.0, "heads_range": 4.0}
    )
    model = DSGCnet(backbone, use_depth_geo=True, depth_geo_cfg=depth_geo_cfg)
    model.eval()

    with torch.no_grad():
        out = model(sample_batch, depth_map=None)

    assert out["pred_logits"].shape[0] == sample_batch.shape[0]


# ---------------------------------------------------------------------------
# Depth normalisation semantics — geo_prior must NOT re-normalise
# ---------------------------------------------------------------------------


def test_geo_prior_preserves_depth_scale():
    """GeoPriorGen must not alter the depth_map values it receives.

    The dataset already normalises depth to [0, 1] per-image.  If the module
    did a second per-batch max-normalisation the output masks would lose
    cross-image relative depth information.

    We craft two images with the same *relative* depth gradient but different
    absolute scales.  Without re-normalisation the decay masks must differ
    because the pairwise depth differences are larger for the high-scale image.
    """
    B, num_heads = 2, 8
    H, W = 16, 16
    embed_dim = 256
    geo = GeoPriorGen(embed_dim=embed_dim, num_heads=num_heads)

    # Image A: gentle gradient  [0.0, 0.1]
    # Image B: steep gradient   [0.0, 0.5]
    # Both live in [0, 1] but have different absolute pairwise differences.
    ramp = torch.linspace(0.0, 1.0, H).view(1, 1, H, 1).expand(1, 1, H, W)
    depth_a = ramp * 0.1  # max diff ≈ 0.1
    depth_b = ramp * 0.5  # max diff ≈ 0.5

    (_, _), (mask_h_a, _) = geo((H, W), depth_a)
    (_, _), (mask_h_b, _) = geo((H, W), depth_b)

    # The depth-dependent decay masks must differ for different depth scales
    assert not torch.allclose(mask_h_a, mask_h_b, atol=1e-6), (
        "Masks for different depth scales should differ — "
        "a second normalisation would erase this difference"
    )


def test_geo_prior_no_nan_inf_with_normalized_depth():
    """Passing pre-normalised [0, 1] depth should produce neither NaN nor Inf."""
    B, num_heads = 2, 8
    H, W = 8, 8
    embed_dim = 256
    geo = GeoPriorGen(embed_dim=embed_dim, num_heads=num_heads)
    depth = torch.rand(B, 1, H, W)  # already in [0, 1]

    (sin, cos), (mask_h, mask_w) = geo((H, W), depth)

    for name, t in [("sin", sin), ("cos", cos), ("mask_h", mask_h), ("mask_w", mask_w)]:
        assert not torch.isnan(t).any(), f"{name} contains NaN"
        assert not torch.isinf(t).any(), f"{name} contains Inf"

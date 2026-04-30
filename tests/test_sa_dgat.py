"""Tests for SA-DGAT: Scale-Aware Deformable Graph Attention Network."""

from __future__ import annotations

import pytest
import torch

from crowdcount.plugins.sa_dgat import (
    CrossScaleGraphAggregation,
    DeformableGraphAttention,
    LocalCountRankingLoss,
    OcclusionAwareGAT,
    SADGATFusion,
    ScalePromptEmbedding,
    SubPixelDensityHead,
)


B, C, H, W = 2, 64, 16, 16  # Smaller dims for fast CPU tests


@pytest.fixture
def feat():
    """Random feature map [B, C, H, W]."""
    return torch.randn(B, C, H, W)


@pytest.fixture
def density():
    """Random density map [B, 1, H, W]."""
    return torch.rand(B, 1, H, W)


# ── ScalePromptEmbedding ──────────────────────────────────────────


class TestScalePromptEmbedding:
    def test_output_shape(self, feat):
        mod = ScalePromptEmbedding(embed_dim=C, num_prompts=5, num_heads=4)
        out, attn = mod(feat)
        assert out.shape == feat.shape
        assert attn.shape[0] == B
        assert attn.shape[2] == 5  # num_prompts

    def test_gradient_flow(self, feat):
        mod = ScalePromptEmbedding(embed_dim=C, num_prompts=3, num_heads=4)
        out, _ = mod(feat)
        loss = out.sum()
        loss.backward()
        assert mod.scale_prompts.grad is not None


# ── DeformableGraphAttention ──────────────────────────────────────


class TestDeformableGraphAttention:
    def test_output_shape(self, feat, density):
        mod = DeformableGraphAttention(in_channels=C, num_neighbors=4, num_heads=4)
        out = mod(feat, density=density)
        assert out.shape == feat.shape

    def test_with_scale_weights(self, feat, density):
        mod = DeformableGraphAttention(in_channels=C, num_neighbors=4, num_heads=4)
        scale_w = torch.randn(B, H * W, 3)
        out = mod(feat, scale_weights=scale_w, density=density)
        assert out.shape == feat.shape

    def test_gradient_flow(self, feat, density):
        mod = DeformableGraphAttention(in_channels=C, num_neighbors=4, num_heads=4)
        out = mod(feat, density=density)
        out.sum().backward()
        assert mod.offset_pred[-1].bias.grad is not None


# ── OcclusionAwareGAT ─────────────────────────────────────────────


class TestOcclusionAwareGAT:
    def test_output_shape(self, feat):
        mod = OcclusionAwareGAT(in_channels=C, num_heads=4, num_layers=1, occ_hidden=32)
        # Need to prepare neighbor_feats: [B, N, K, C]
        N = H * W
        K = 4
        neighbor_feats = torch.randn(B, N, K, C)
        out, occ_map = mod(feat, neighbor_feats=neighbor_feats)
        assert out.shape == feat.shape
        assert occ_map.shape == (B, 1, H, W)

    def test_occlusion_range(self, feat):
        mod = OcclusionAwareGAT(in_channels=C, num_heads=4, num_layers=1, occ_hidden=32)
        N = H * W
        neighbor_feats = torch.randn(B, N, 4, C)
        _, occ_map = mod(feat, neighbor_feats=neighbor_feats)
        # Occlusion should be in [0, 1] due to sigmoid
        assert occ_map.min() >= 0.0
        assert occ_map.max() <= 1.0


# ── CrossScaleGraphAggregation ────────────────────────────────────


class TestCrossScaleGraphAggregation:
    def test_output_shape(self):
        mod = CrossScaleGraphAggregation(
            in_channels=C, local_dilations=(1, 2), global_dilations=(1, 3)
        )
        f_local = torch.randn(B, C, H * 2, W * 2)  # High-res
        f_mid = torch.randn(B, C, H, W)  # Mid-res
        f_global = torch.randn(B, C, H // 2, W // 2)  # Low-res
        out = mod(f_local, f_mid, f_global)
        assert out.shape == f_mid.shape  # Output at mid resolution

    def test_gradient_flow(self):
        mod = CrossScaleGraphAggregation(
            in_channels=C, local_dilations=(1, 2), global_dilations=(1, 3)
        )
        f_local = torch.randn(B, C, H * 2, W * 2, requires_grad=True)
        f_mid = torch.randn(B, C, H, W, requires_grad=True)
        f_global = torch.randn(B, C, H // 2, W // 2, requires_grad=True)
        out = mod(f_local, f_mid, f_global)
        out.sum().backward()
        assert f_local.grad is not None
        assert f_global.grad is not None


# ── SubPixelDensityHead ───────────────────────────────────────────


class TestSubPixelDensityHead:
    def test_output_shape(self, feat):
        mod = SubPixelDensityHead(in_channels=C, hidden_channels=32, upscale_factor=2)
        out = mod(feat)
        assert out.shape == (B, 1, H * 2, W * 2)

    def test_non_negative(self, feat):
        mod = SubPixelDensityHead(in_channels=C, hidden_channels=32)
        out = mod(feat)
        assert out.min() >= 0.0  # ReLU at the end


# ── LocalCountRankingLoss ─────────────────────────────────────────


class TestLocalCountRankingLoss:
    def test_scalar_output(self, density):
        loss_fn = LocalCountRankingLoss(grid_size=4, num_pairs=8)
        pred = torch.rand_like(density)
        loss = loss_fn(pred, density)
        assert loss.dim() == 0

    def test_zero_on_identical(self):
        loss_fn = LocalCountRankingLoss(grid_size=2, num_pairs=4)
        d = torch.rand(1, 1, 8, 8)
        loss = loss_fn(d, d)
        assert loss.item() < 1e-4


# ── SADGATFusion (end-to-end) ─────────────────────────────────────


class TestSADGATFusion:
    def test_output_shape_no_cross_scale(self, feat, density):
        mod = SADGATFusion(
            in_channels=C,
            num_scale_prompts=3,
            deformable_k=4,
            num_heads=4,
            use_cross_scale=False,
        )
        out, aux = mod(feat, density)
        assert out.shape == feat.shape
        assert "scale_weights" in aux
        assert "occlusion_map" in aux

    def test_output_shape_with_cross_scale(self, feat, density):
        mod = SADGATFusion(
            in_channels=C,
            num_scale_prompts=3,
            deformable_k=4,
            num_heads=4,
            local_dilations=(1, 2),
            global_dilations=(1, 3),
            use_cross_scale=True,
        )
        p3 = torch.randn(B, C, H * 2, W * 2)
        p4 = torch.randn(B, C, H, W)
        p5 = torch.randn(B, C, H // 2, W // 2)
        out, aux = mod(feat, density, fpn_intermediates=(p3, p4, p5))
        assert out.shape == feat.shape

    def test_gradient_flow(self, feat, density):
        mod = SADGATFusion(
            in_channels=C,
            num_scale_prompts=3,
            deformable_k=4,
            num_heads=4,
            use_cross_scale=False,
        )
        feat_req = feat.clone().requires_grad_(True)
        out, _ = mod(feat_req, density)
        out.sum().backward()
        assert feat_req.grad is not None

    def test_depth_prior(self, feat, density):
        mod = SADGATFusion(
            in_channels=C,
            num_scale_prompts=3,
            deformable_k=4,
            num_heads=4,
            use_cross_scale=False,
            use_depth_prior=True,
        )
        depth = torch.rand(B, 1, H * 4, W * 4)  # Full-res depth
        out, aux = mod(feat, density, depth_map=depth)
        assert out.shape == feat.shape
        assert aux["occlusion_map"].shape == (B, 1, H, W)

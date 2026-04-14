"""Tests for cross-scale density refinement, multi-scale density fusion,
and cross-scale consistency loss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from crowdcount.plugins.cross_scale_density import (
    CrossScaleDensityRefinement,
    MultiScaleDensityFusion,
    _DensityRefinementStage,
)
from crowdcount.models.dsgcnet import DSGCnet


# ---------------------------------------------------------------------------
# Tiny VGG backbone (same as test_dsgc.py)
# ---------------------------------------------------------------------------
class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        f0 = torch.zeros(B, 128, H // 2, W // 2)
        f1 = torch.zeros(B, 256, H // 4, W // 4)
        f2 = torch.zeros(B, 512, H // 8, W // 8)
        f3 = torch.zeros(B, 512, H // 16, W // 16)
        return [f0, f1, f2, f3]


# ============================================================================
# Unit tests for CrossScaleDensityRefinement
# ============================================================================


class TestCrossScaleDensityRefinement:
    def test_forward_shapes(self) -> None:
        module = CrossScaleDensityRefinement()
        c3 = torch.randn(2, 256, 32, 32)  # H/4
        c4 = torch.randn(2, 512, 16, 16)  # H/8
        c5 = torch.randn(2, 512, 8, 8)  # H/16

        out = module(c3, c4, c5)

        assert out["density_block5"].shape == (2, 1, 8, 8)
        assert out["density_block4"].shape == (2, 1, 16, 16)
        assert out["density_block3"].shape == (2, 1, 32, 32)

    def test_output_non_negative(self) -> None:
        """All density outputs should be >= 0 (ReLU)."""
        module = CrossScaleDensityRefinement()
        c3 = torch.randn(2, 256, 32, 32)
        c4 = torch.randn(2, 512, 16, 16)
        c5 = torch.randn(2, 512, 8, 8)

        out = module(c3, c4, c5)
        for key in ["density_block3", "density_block4", "density_block5"]:
            assert (out[key] >= 0).all(), f"{key} has negative values"

    def test_backward(self) -> None:
        """Gradients flow through all three stages."""
        module = CrossScaleDensityRefinement()
        c3 = torch.randn(2, 256, 32, 32, requires_grad=True)
        c4 = torch.randn(2, 512, 16, 16, requires_grad=True)
        c5 = torch.randn(2, 512, 8, 8, requires_grad=True)

        out = module(c3, c4, c5)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        assert c3.grad is not None
        assert c4.grad is not None
        assert c5.grad is not None


# ============================================================================
# Unit tests for MultiScaleDensityFusion
# ============================================================================


class TestMultiScaleDensityFusion:
    def test_forward_shape(self) -> None:
        fuse = MultiScaleDensityFusion(num_scales=4)
        d_main = torch.randn(2, 1, 8, 8)
        d3 = torch.randn(2, 1, 32, 32)
        d4 = torch.randn(2, 1, 16, 16)
        d5 = torch.randn(2, 1, 8, 8)

        out = fuse(d_main, d3, d4, d5)
        assert out.shape == (2, 1, 8, 8)

    def test_output_non_negative(self) -> None:
        fuse = MultiScaleDensityFusion(num_scales=4)
        d_main = torch.randn(2, 1, 8, 8)
        d3 = torch.randn(2, 1, 32, 32)
        d4 = torch.randn(2, 1, 16, 16)
        d5 = torch.randn(2, 1, 8, 8)

        out = fuse(d_main, d3, d4, d5)
        assert (out >= 0).all()

    def test_backward(self) -> None:
        fuse = MultiScaleDensityFusion(num_scales=4)
        d_main = torch.randn(2, 1, 8, 8, requires_grad=True)
        d3 = torch.randn(2, 1, 32, 32, requires_grad=True)
        d4 = torch.randn(2, 1, 16, 16, requires_grad=True)
        d5 = torch.randn(2, 1, 8, 8, requires_grad=True)

        out = fuse(d_main, d3, d4, d5)
        out.sum().backward()

        for t in [d_main, d3, d4, d5]:
            assert t.grad is not None


# ============================================================================
# Unit tests for _DensityRefinementStage
# ============================================================================


class TestDensityRefinementStage:
    def test_forward_shape(self) -> None:
        stage = _DensityRefinementStage(feat_channels=512)
        coarse = torch.randn(2, 1, 8, 8)
        fine_feat = torch.randn(2, 512, 16, 16)

        out = stage(coarse, fine_feat)
        assert out.shape == (2, 1, 16, 16)


# ============================================================================
# Integration tests with DSGCnet
# ============================================================================


class TestDSGCnetCrossScaleRefine:
    def test_cross_scale_refine_outputs(self) -> None:
        """When cross_scale_refine is enabled, model outputs refined densities."""
        backbone = TinyVGGBackbone()
        cfg = OmegaConf.create(
            {
                "density_multi_scale": {
                    "enabled": True,
                    "cross_scale_refine": True,
                    "fuse_to_gcn": False,
                }
            }
        )
        model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))

        assert "density_block3" in out
        assert "density_block4" in out
        assert "density_block5" in out
        assert out["density_block3"].shape[0] == 2
        assert out["density_block3"].shape[1] == 1

    def test_fuse_to_gcn_outputs(self) -> None:
        """When fuse_to_gcn is enabled, density_fused key should exist."""
        backbone = TinyVGGBackbone()
        cfg = OmegaConf.create(
            {
                "density_multi_scale": {
                    "enabled": True,
                    "cross_scale_refine": True,
                    "fuse_to_gcn": True,
                }
            }
        )
        model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))

        assert "density_fused" in out
        # Fused density has same shape as main density
        assert out["density_fused"].shape == out["density_out"].shape

    def test_independent_heads_still_work(self) -> None:
        """When cross_scale_refine=False, original independent heads are used."""
        backbone = TinyVGGBackbone()
        cfg = OmegaConf.create(
            {
                "density_multi_scale": {
                    "enabled": True,
                    "cross_scale_refine": False,
                    "fuse_to_gcn": False,
                }
            }
        )
        model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))

        assert "density_block3" in out
        assert "density_block4" in out
        assert "density_block5" in out

    def test_disabled_no_extra_outputs(self) -> None:
        """When disabled, no multi-scale keys in output."""
        backbone = TinyVGGBackbone()
        cfg = OmegaConf.create({"density_multi_scale": {"enabled": False}})
        model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

        with torch.no_grad():
            out = model(torch.zeros(1, 3, 128, 128))

        assert "density_block3" not in out
        assert "density_fused" not in out

    def test_fuse_to_gcn_without_cross_scale(self) -> None:
        """fuse_to_gcn works with independent heads (no cross_scale_refine)."""
        backbone = TinyVGGBackbone()
        cfg = OmegaConf.create(
            {
                "density_multi_scale": {
                    "enabled": True,
                    "cross_scale_refine": False,
                    "fuse_to_gcn": True,
                }
            }
        )
        model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))

        assert "density_fused" in out
        assert "density_block3" in out

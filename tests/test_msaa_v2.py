"""Tests for MSAA v2 modules: MSAALite, FPNAttentionGate, MSAAGate.

Validates shapes, gradients, parameter counts, and integration with DSGCnet.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.plugins.msaa import (
    DilatedMultiScaleFusion,
    ECAAttention,
    FPNAttentionGate,
    FPNSpatialAttention,
    MSAAGate,
    MSAALite,
)
from crowdcount.models.neck import Decoder_SPD_PAFPN
from crowdcount.models.dsgcnet import DSGCnet


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


# ====================================================================
# Unit tests for lightweight components
# ====================================================================


class TestECAAttention:
    def test_output_shape(self) -> None:
        m = ECAAttention(256)
        x = torch.randn(2, 256, 16, 16)
        assert m(x).shape == (2, 256, 16, 16)

    def test_gradient_flow(self) -> None:
        m = ECAAttention(256)
        x = torch.randn(2, 256, 8, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count_tiny(self) -> None:
        m = ECAAttention(256)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params < 100  # ECA has near-zero params


class TestDilatedMultiScaleFusion:
    def test_output_shape(self) -> None:
        m = DilatedMultiScaleFusion(256)
        x = torch.randn(2, 256, 16, 16)
        assert m(x).shape == (2, 256, 16, 16)

    def test_gradient_flow(self) -> None:
        m = DilatedMultiScaleFusion(256)
        x = torch.randn(2, 256, 8, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ====================================================================
# Phase 1: MSAALite
# ====================================================================


class TestMSAALite:
    def test_output_shape(self) -> None:
        m = MSAALite(256)
        x = torch.randn(2, 256, 16, 16)
        assert m(x).shape == (2, 256, 16, 16)

    def test_residual_connection(self) -> None:
        """Output should differ from input (residual + attention)."""
        m = MSAALite(256).eval()
        x = torch.randn(2, 256, 16, 16)
        with torch.no_grad():
            out = m(x)
        assert not torch.allclose(out, x)

    def test_parameter_count(self) -> None:
        m = MSAALite(256)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params < 1_000_000  # < 1M params

    def test_gradient_to_all_params(self) -> None:
        m = MSAALite(256)
        x = torch.randn(1, 256, 8, 8)
        m(x).sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ====================================================================
# Phase 2: FPNAttentionGate
# ====================================================================


class TestFPNAttentionGate:
    def test_output_shape(self) -> None:
        m = FPNAttentionGate(256)
        lat = torch.randn(2, 256, 16, 16)
        trans = torch.randn(2, 256, 16, 16)
        assert m(lat, trans).shape == (2, 256, 16, 16)

    def test_gate_starts_near_balanced(self) -> None:
        """Fresh gate should give near-uniform blending."""
        m = FPNAttentionGate(256).eval()
        lat = torch.randn(2, 256, 8, 8)
        trans = torch.randn(2, 256, 8, 8)
        with torch.no_grad():
            out = m(lat, trans)
        # Should be somewhere between lateral and transferred, not all-one-side
        assert not torch.allclose(out, lat, atol=0.01)
        assert not torch.allclose(out, trans, atol=0.01)

    def test_fpn_with_attention(self) -> None:
        """PA-FPN with fpn_attention=True should produce same shape."""
        fpn = Decoder_SPD_PAFPN(256, 512, 512, fpn_attention=True)
        c3 = torch.randn(2, 256, 32, 32)
        c4 = torch.randn(2, 512, 16, 16)
        c5 = torch.randn(2, 512, 8, 8)
        out = fpn([c3, c4, c5])
        assert out.shape == (2, 256, 16, 16)

    def test_fpn_without_attention_unchanged(self) -> None:
        """PA-FPN with fpn_attention=False should work as before."""
        fpn = Decoder_SPD_PAFPN(256, 512, 512, fpn_attention=False)
        c3 = torch.randn(2, 256, 32, 32)
        c4 = torch.randn(2, 512, 16, 16)
        c5 = torch.randn(2, 512, 8, 8)
        out = fpn([c3, c4, c5])
        assert out.shape == (2, 256, 16, 16)


# ====================================================================
# Phase 3: MSAAGate
# ====================================================================


class TestMSAAGate:
    def test_output_shape(self) -> None:
        m = MSAAGate(256, num_streams=3)
        s0 = torch.randn(2, 256, 16, 16)
        s1 = torch.randn(2, 256, 16, 16)
        s2 = torch.randn(2, 256, 16, 16)
        assert m(s0, s1, s2).shape == (2, 256, 16, 16)

    def test_parameter_count(self) -> None:
        m = MSAAGate(256, num_streams=3)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params < 1_500_000  # < 1.5M

    def test_gradient_to_all_params(self) -> None:
        m = MSAAGate(256, num_streams=3)
        s0 = torch.randn(1, 256, 8, 8)
        s1 = torch.randn(1, 256, 8, 8)
        s2 = torch.randn(1, 256, 8, 8)
        m(s0, s1, s2).sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ====================================================================
# Integration tests: DSGCnet with new variants
# ====================================================================


class TestDSGCnetMSAAVariants:
    def test_msaa_lite_variant(self) -> None:
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone, row=2, line=2, use_msaa=True, msaa_variant="lite"
        ).eval()
        assert model.msaa is None  # legacy MSAA disabled
        assert model.msaa_lite is not None
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2
        assert out["density_out"].shape[0] == 2

    def test_msaa_gate_variant(self) -> None:
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone, row=2, line=2, use_msaa=True, msaa_variant="msaa_gate"
        ).eval()
        assert model.msaa is None
        assert model.msaa_gate is not None
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_legacy_variant_still_works(self) -> None:
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone, row=2, line=2, use_msaa=True, msaa_variant="legacy"
        ).eval()
        assert model.msaa is not None
        assert model.msaa_lite is None
        assert model.msaa_gate is None
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_fpn_attention_integration(self) -> None:
        backbone = TinyVGGBackbone()
        model = DSGCnet(backbone, row=2, line=2, fpn_attention=True).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_lite_plus_fpn_attention(self) -> None:
        """Phase 1 + Phase 2 combined."""
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone,
            row=2,
            line=2,
            use_msaa=True,
            msaa_variant="lite",
            fpn_attention=True,
        ).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_msaa_gate_plus_fpn_attention(self) -> None:
        """Phase 2 + Phase 3 combined."""
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone,
            row=2,
            line=2,
            use_msaa=True,
            msaa_variant="msaa_gate",
            fpn_attention=True,
        ).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_all_three_phases(self) -> None:
        """All phases combined: lite + fpn_attention + msaa_gate is NOT valid
        (lite and msaa_gate are mutually exclusive variants).
        Check that lite + fpn_attention works fine."""
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone,
            row=2,
            line=2,
            use_msaa=True,
            msaa_variant="lite",
            fpn_attention=True,
        ).eval()
        assert model.msaa_lite is not None
        assert model.msaa_gate is None  # can't have both
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out["pred_logits"].shape[0] == 2

    def test_pa_channels_unchanged_for_lite(self) -> None:
        """MSAALite should NOT change PA-FPN input channels (stays 256/512/512)."""
        backbone = TinyVGGBackbone()
        model = DSGCnet(backbone, row=2, line=2, use_msaa=True, msaa_variant="lite")
        assert model.pa.P5_1[0].in_channels == 512
        assert model.pa.P3_1[0].in_channels == 256

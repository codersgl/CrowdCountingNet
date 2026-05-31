"""Tests for scale_decoupled_fusion — synthetic tensors, no GPU/data required."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.scale_decoupled_fusion import (
    CNNStream,
    DensityGCNRefine,
    DensitySEModulation,
    GCNStream,
    ScaleDecoupledCrossAttention,
    ScaleDecoupledFusion,
    TransformerStream,
    sinusoidal_2d_pe,
)


class TestSinusoidal2DPE:
    def test_output_shape_square(self):
        pe = sinusoidal_2d_pe(7, 7, 256)
        assert pe.shape == (1, 49, 256)

    def test_output_shape_rectangular(self):
        pe = sinusoidal_2d_pe(14, 14, 256)
        assert pe.shape == (1, 196, 256)

    def test_value_range(self):
        pe = sinusoidal_2d_pe(3, 3, 64)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_deterministic(self):
        pe1 = sinusoidal_2d_pe(5, 5, 128)
        pe2 = sinusoidal_2d_pe(5, 5, 128)
        assert torch.equal(pe1, pe2)

    def test_dim_not_divisible_by_4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            sinusoidal_2d_pe(7, 7, 10)


class TestCNNStream:
    @pytest.fixture
    def stream(self):
        return CNNStream(in_channels=256)

    def test_output_shape(self, stream):
        x = torch.randn(2, 256, 28, 28)
        out = stream(x)
        assert out.shape == (2, 256, 28, 28)

    def test_no_nan(self, stream):
        x = torch.randn(4, 256, 16, 16)
        stream.eval()
        with torch.no_grad():
            out = stream(x)
        assert not torch.isnan(out).any()

    def test_training_mode(self, stream):
        stream.train()
        x = torch.randn(2, 256, 20, 20)
        out = stream(x)
        assert out.shape == (2, 256, 20, 20)


class TestGCNStream:
    @pytest.fixture
    def stream(self):
        return GCNStream(in_channels=512, out_channels=256, k=4)

    def test_output_shape_no_density(self, stream):
        x = torch.randn(2, 512, 14, 14)
        out = stream(x)
        assert out.shape == (2, 256, 14, 14)

    def test_with_density(self, stream):
        x = torch.randn(2, 512, 14, 14)
        density = torch.rand(2, 1, 14, 14)
        out = stream(x, density=density)
        assert out.shape == (2, 256, 14, 14)

    def test_no_nan_fallback(self, stream):
        x = torch.randn(1, 512, 10, 10)
        stream.eval()
        with torch.no_grad():
            out = stream(x, density=None)
        assert out.shape == (1, 256, 10, 10)
        assert not torch.isnan(out).any()


class TestTransformerStream:
    @pytest.fixture
    def stream(self):
        return TransformerStream(in_channels=512, out_channels=256, num_blocks=2)

    def test_output_shape(self, stream):
        x = torch.randn(2, 512, 7, 7)
        out = stream(x)
        assert out.shape == (2, 256, 7, 7)

    def test_small_input(self, stream):
        x = torch.randn(1, 512, 4, 4)
        out = stream(x)
        assert out.shape == (1, 256, 4, 4)

    def test_different_input_sizes(self):
        stream = TransformerStream(in_channels=512, out_channels=256, num_blocks=1)
        x1 = torch.randn(1, 512, 7, 7)
        x2 = torch.randn(1, 512, 5, 5)
        out1 = stream(x1)
        out2 = stream(x2)
        assert out1.shape == (1, 256, 7, 7)
        assert out2.shape == (1, 256, 5, 5)

    def test_no_nan(self, stream):
        x = torch.randn(2, 512, 7, 7)
        stream.eval()
        with torch.no_grad():
            out = stream(x)
        assert not torch.isnan(out).any()


class TestScaleDecoupledCrossAttention:
    @pytest.fixture
    def ca(self):
        return ScaleDecoupledCrossAttention(dim=256, num_heads=4)

    def test_output_shape(self, ca):
        f8 = torch.randn(2, 256, 28, 28)
        f16 = torch.randn(2, 256, 14, 14)
        f32 = torch.randn(2, 256, 7, 7)
        out = ca(f8, f16, f32)
        assert out.shape == (2, 256, 28, 28)

    def test_nq_not_equal_nkv(self, ca):
        f8 = torch.randn(1, 256, 28, 28)
        f16 = torch.randn(1, 256, 14, 14)
        f32 = torch.randn(1, 256, 7, 7)
        out = ca(f8, f16, f32)
        assert out.shape == (1, 256, 28, 28)

    def test_no_nan(self, ca):
        f8 = torch.randn(2, 256, 16, 16)
        f16 = torch.randn(2, 256, 8, 8)
        f32 = torch.randn(2, 256, 4, 4)
        ca.eval()
        with torch.no_grad():
            out = ca(f8, f16, f32)
        assert not torch.isnan(out).any()

    def test_training_mode(self, ca):
        ca.train()
        f8 = torch.randn(2, 256, 16, 16)
        f16 = torch.randn(2, 256, 8, 8)
        f32 = torch.randn(2, 256, 4, 4)
        out = ca(f8, f16, f32)
        assert out.shape == (2, 256, 16, 16)

    def test_identity_behavior(self, ca):
        f8 = torch.randn(2, 256, 16, 16)
        f16 = torch.randn(2, 256, 8, 8)
        f32 = torch.randn(2, 256, 4, 4)
        ca.eval()
        with torch.no_grad():
            out = ca(f8, f16, f32)
        assert out.std() > 0


class TestDensitySEModulation:
    @pytest.fixture
    def mod(self):
        return DensitySEModulation(channels=256, density_hidden=64, reduction=4)

    def test_output_shape(self, mod):
        f = torch.randn(2, 256, 28, 28)
        density = torch.rand(2, 1, 28, 28)
        out = mod(f, density)
        assert out.shape == (2, 256, 28, 28)

    def test_modulation_has_effect(self, mod):
        """Standard SE residual → output ≠ input at init."""
        f = torch.randn(2, 256, 16, 16)
        density = torch.rand(2, 1, 16, 16)
        mod.eval()
        with torch.no_grad():
            out = mod(f, density)
        # Direct SE modulation — must differ from input
        assert not torch.allclose(out, f)

    def test_density_interpolation(self, mod):
        f = torch.randn(1, 256, 28, 28)
        density = torch.rand(1, 1, 14, 14)
        out = mod(f, density)
        assert out.shape == (1, 256, 28, 28)

    def test_detach_inference(self, mod):
        f = torch.randn(2, 256, 28, 28)
        density = torch.rand(2, 1, 28, 28)
        mod.eval()
        with torch.no_grad():
            out = mod(f, density)
        assert not torch.isnan(out).any()


class TestDensityGCNRefine:
    @pytest.fixture
    def refine(self):
        return DensityGCNRefine(channels=256, k=4)

    def test_output_shape(self, refine):
        f = torch.randn(2, 256, 14, 14)
        density = torch.rand(2, 1, 14, 14)
        out = refine(f, density)
        assert out.shape == (2, 256, 14, 14)

    def test_residual_has_effect(self, refine):
        """Direct residual (no gate) → output ≠ input at init.

        The GATv2 uses standard Xavier init and contributes a non-trivial
        residual from step 0, just like a standard ResNet block.
        """
        f = torch.randn(2, 256, 14, 14)
        density = torch.rand(2, 1, 14, 14)
        refine.eval()
        with torch.no_grad():
            out = refine(f, density)
        # Direct residual — output should differ from input
        assert not torch.allclose(out, f)

    def test_no_nan(self, refine):
        f = torch.randn(1, 256, 10, 10)
        density = torch.rand(1, 1, 10, 10)
        refine.eval()
        with torch.no_grad():
            out = refine(f, density)
        assert not torch.isnan(out).any()

    def test_training_mode(self, refine):
        refine.train()
        f = torch.randn(2, 256, 14, 14)
        density = torch.rand(2, 1, 14, 14)
        out = refine(f, density)
        assert out.shape == (2, 256, 14, 14)


class TestScaleDecoupledFusion:
    @pytest.fixture
    def fusion(self):
        return ScaleDecoupledFusion(
            c2_channels=256, c3_channels=512, c4_channels=512, unified_dim=256,
        )

    def test_full_forward(self, fusion):
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        f, aux = fusion(c2, c3, c4)
        # Q←GCN(c3) → output follows Q resolution (14×14)
        assert f.shape == (2, 256, 14, 14)
        assert isinstance(aux, dict)

    def test_with_modulation(self, fusion):
        c2 = torch.randn(1, 256, 28, 28)
        c3 = torch.randn(1, 512, 14, 14)
        c4 = torch.randn(1, 512, 7, 7)
        f, _ = fusion(c2, c3, c4)
        density = torch.rand(1, 1, 28, 28)  # density interpolated to 14×14 internally
        f_mod = fusion.density_modulation(f, density)
        assert f_mod.shape == (1, 256, 14, 14)

    def test_varying_sizes(self, fusion):
        c2 = torch.randn(1, 256, 32, 32)
        c3 = torch.randn(1, 512, 16, 16)
        c4 = torch.randn(1, 512, 8, 8)
        f, _ = fusion(c2, c3, c4)
        # Q←GCN(c3) → output at 16×16
        assert f.shape == (1, 256, 16, 16)

    def test_training_mode(self, fusion):
        fusion.train()
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        f, aux = fusion(c2, c3, c4)
        # Q←GCN(c3) → output at 14×14
        assert f.shape == (2, 256, 14, 14)

    def test_no_nan(self, fusion):
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        fusion.eval()
        with torch.no_grad():
            f, _ = fusion(c2, c3, c4)
        assert not torch.isnan(f).any()

    def test_refine_with_density(self, fusion):
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        f, _ = fusion(c2, c3, c4)
        density = torch.rand(2, 1, 14, 14)
        f_refined = fusion.refine_with_density(f, density)
        assert f_refined.shape == f.shape

    def test_refine_with_density_has_effect(self, fusion):
        """Direct residual (no gate) → refined ≠ input at init.

        The GATv2 uses standard Xavier init — its residual contribution
        is non-trivial from step 0, like any ResNet block.
        """
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        fusion.eval()
        with torch.no_grad():
            f, _ = fusion(c2, c3, c4)
            density = torch.rand(2, 1, 14, 14)
            f_refined = fusion.refine_with_density(f, density)
        # Direct residual — output must differ from input
        assert not torch.allclose(f_refined, f)

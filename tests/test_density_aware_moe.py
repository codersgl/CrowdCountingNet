"""Tests for density-conditioned GlobalDensityExpert and HeterogeneousSparseMoE."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from crowdcount.models.moecount.experts import (
    GlobalDensityExpert,
    HeterogeneousSparseMoE,
)


class TestGlobalDensityExpertDensity:
    """Tests for density-conditioned GlobalDensityExpert."""

    @pytest.fixture
    def features(self) -> torch.Tensor:
        return torch.randn(2, 256, 64, 64)

    @pytest.fixture
    def density(self) -> torch.Tensor:
        return torch.randn(2, 1, 64, 64)

    def test_forward_with_density_shape(self, features, density):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        out = expert(features, density=density)
        assert out.shape == features.shape

    def test_forward_without_density_shape(self, features):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        out = expert(features, density=None)
        assert out.shape == features.shape

    def test_forward_without_density_kwarg(self, features):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        out = expert(features)
        assert out.shape == features.shape

    def test_use_density_false_rejects_density(self, features, density):
        expert = GlobalDensityExpert(channels=256, use_density=False)
        out = expert(features, density=density)
        assert out.shape == features.shape
        assert not hasattr(expert, "density_fuse")

    def test_gradient_flows_through_density_path(self, features, density):
        expert = GlobalDensityExpert(channels=256, use_density=True, use_residual=False)
        features.requires_grad_(True)
        density.requires_grad_(True)
        out = expert(features, density=density)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert density.grad is not None
        assert not torch.allclose(density.grad, torch.zeros_like(density.grad))

    def test_density_fuse_parameter_count(self):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        density_params = sum(
            p.numel() for p in expert.density_fuse.parameters()
        )
        assert density_params > 0
        assert density_params < 70000  # ~66K: 257*256 + 256*2 (GN)

    def test_density_fuse_module_exists(self):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        assert isinstance(expert.density_fuse, nn.Sequential)
        assert isinstance(expert.density_fuse[0], nn.Conv2d)
        assert expert.density_fuse[0].in_channels == 257
        assert expert.density_fuse[0].out_channels == 256

    def test_no_nan_with_density(self, features, density):
        expert = GlobalDensityExpert(channels=256, use_density=True)
        out = expert(features, density=density)
        assert not torch.isnan(out).any()

    def test_different_output_with_vs_without_density(self, features, density):
        expert = GlobalDensityExpert(channels=256, use_density=True, use_residual=False)
        out_with = expert(features, density=density)
        out_without = expert(features, density=None)
        assert not torch.allclose(out_with, out_without)


class TestHeterogeneousSparseMoEDensity:
    """Tests for HeterogeneousSparseMoE with density-conditioned global expert."""

    @pytest.fixture
    def features(self) -> torch.Tensor:
        return torch.randn(2, 256, 32, 32)

    @pytest.fixture
    def density(self) -> torch.Tensor:
        return torch.randn(2, 1, 32, 32)

    def test_forward_with_density_shape(self, features, density):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        fused, aux, route = moe(features, density=density)
        assert fused.shape == features.shape
        assert "weights" in route

    def test_forward_without_density_no_crash(self, features):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        fused, aux, route = moe(features)
        assert fused.shape == features.shape

    def test_forward_without_density_none(self, features):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        fused, aux, route = moe(features, density=None)
        assert fused.shape == features.shape

    def test_gradient_through_density_path(self, features, density):
        """Density gradient flows through GlobalDensityExpert.density_fuse path.

        When ``expert_global_density_use_density=True``, GlobalDensityExpert
        concatenates density as an extra input channel and fuses it via a 1×1
        conv.  This creates a legitimate gradient highway from the MoE output
        back to the density prediction head.

        Note: at initialisation the gradient magnitude is near-zero because
        ``residual_gate.tanh() ≈ 0`` gates the expert's non-identity output.
        The test only asserts that density is *in the graph* (grad is not
        None); the gradient magnitude grows as the residual gate opens.
        """
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        features.requires_grad_(True)
        density.requires_grad_(True)
        fused, aux, route = moe(features, density=density)
        loss = fused.sum()
        loss.backward()
        assert features.grad is not None
        assert density.grad is not None

    def test_density_disabled_moe_no_density_fuse(self, features, density):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=False
        )
        global_expert = moe.experts[2]
        assert isinstance(global_expert, GlobalDensityExpert)
        assert not global_expert.use_density
        assert not hasattr(global_expert, "density_fuse")
        out = moe(features, density=density)
        assert out[0].shape == features.shape

    def test_no_nan_with_density(self, features, density):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        fused, aux, route = moe(features, density=density)
        assert not torch.isnan(fused).any()

    def test_route_weights_sum_to_one_per_pixel(self, features, density):
        moe = HeterogeneousSparseMoE(
            channels=256, expert_global_density_use_density=True
        )
        fused, aux, route = moe(features, density=density)
        weights = route["weights"]
        w_sum = weights.sum(dim=1)
        assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)

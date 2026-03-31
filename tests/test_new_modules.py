"""Tests for the three new architecture modules:
- SuperNodeGCNProcessor (gcn.py)
- FreqDecoupledRouter (head.py)
- SubPixelRefineModule (head.py)

Also tests end-to-end DSGCnet forward with each module enabled.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.models.gcn import SuperNodeGCNProcessor
from crowdcount.models.head import FreqDecoupledRouter, SubPixelRefineModule
from crowdcount.models.dsgcnet import DSGCnet


# ---------------------------------------------------------------------------
# Reusable tiny backbone (same as test_dsgc.py)
# ---------------------------------------------------------------------------


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


@pytest.fixture
def sample_tensor():
    return torch.zeros(2, 3, 128, 128)


# ===========================================================================
# SuperNodeGCNProcessor
# ===========================================================================


class TestSuperNodeGCNProcessor:
    @pytest.fixture
    def features(self):
        return torch.randn(2, 256, 8, 8)

    @pytest.fixture
    def density(self):
        return torch.rand(2, 1, 8, 8)

    def test_output_shape(self, features, density):
        proc = SuperNodeGCNProcessor(in_channels=256, num_supernodes=8, num_heads=4)
        out = proc(features, density)
        assert out.shape == features.shape

    def test_gated_residual_init_identity(self, features, density):
        """At init, gate=0 → output ≈ input (gated residual = identity)."""
        proc = SuperNodeGCNProcessor(in_channels=256, num_supernodes=4, num_heads=2)
        proc.eval()
        with torch.no_grad():
            out = proc(features, density)
        assert torch.allclose(out, features, atol=1e-5)

    def test_different_supernode_counts(self, features, density):
        for M in [2, 4, 8, 16]:
            proc = SuperNodeGCNProcessor(in_channels=256, num_supernodes=M, num_heads=2)
            out = proc(features, density)
            assert out.shape == features.shape

    def test_gradient_flows(self, features, density):
        proc = SuperNodeGCNProcessor(in_channels=256, num_supernodes=8, num_heads=4)
        out = proc(features, density)
        loss = out.sum()
        loss.backward()
        assert proc.prototypes.grad is not None


# ===========================================================================
# FreqDecoupledRouter
# ===========================================================================


class TestFreqDecoupledRouter:
    @pytest.fixture
    def shared_feat(self):
        return torch.randn(2, 256, 16, 16)

    def test_output_shapes(self, shared_feat):
        router = FreqDecoupledRouter(kernel_size=3)
        f_low, f_high, f_full = router(shared_feat)
        assert f_low.shape == shared_feat.shape
        assert f_high.shape == shared_feat.shape
        assert f_full.shape == shared_feat.shape

    def test_frequency_decomposition_identity(self, shared_feat):
        """f_low + f_high should reconstruct the original feature exactly."""
        router = FreqDecoupledRouter(kernel_size=3)
        f_low, f_high, f_full = router(shared_feat)
        reconstructed = f_low + f_high
        assert torch.allclose(reconstructed, shared_feat, atol=1e-5)

    def test_f_full_is_input(self, shared_feat):
        """f_full should be the original shared_feat."""
        router = FreqDecoupledRouter(kernel_size=3)
        _, _, f_full = router(shared_feat)
        assert torch.equal(f_full, shared_feat)

    def test_zero_learnable_params(self):
        router = FreqDecoupledRouter(kernel_size=5)
        num_params = sum(p.numel() for p in router.parameters())
        assert num_params == 0

    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_various_kernel_sizes(self, shared_feat, k):
        router = FreqDecoupledRouter(kernel_size=k)
        f_low, f_high, f_full = router(shared_feat)
        assert torch.allclose(f_low + f_high, shared_feat, atol=1e-5)


# ===========================================================================
# SubPixelRefineModule
# ===========================================================================


class TestSubPixelRefineModule:
    @pytest.fixture
    def module(self):
        return SubPixelRefineModule(
            hr_channels=256, lr_channels=256, hidden_dim=64, top_k=8
        )

    @pytest.fixture
    def hr_feat(self):
        return torch.randn(2, 256, 16, 16)  # C3 at stride 8

    @pytest.fixture
    def lr_feat(self):
        return torch.randn(2, 256, 8, 8)  # features_pa at stride 16

    @pytest.fixture
    def pred_points(self):
        return torch.rand(2, 32, 2) * 128  # [B, Q, 2] in pixel coords

    @pytest.fixture
    def pred_scores(self):
        return torch.rand(2, 32)  # [B, Q]

    def test_output_shape(self, module, hr_feat, lr_feat, pred_points, pred_scores):
        refined = module(hr_feat, lr_feat, pred_points, pred_scores, 128, 128)
        assert refined.shape == pred_points.shape

    def test_init_zero_offset(self, module, hr_feat, lr_feat, pred_points, pred_scores):
        """At init, MLP weights are zero → refined ≈ original."""
        module.eval()
        with torch.no_grad():
            refined = module(hr_feat, lr_feat, pred_points, pred_scores, 128, 128)
        assert torch.allclose(refined, pred_points, atol=1e-5)

    def test_top_k_clipping(self, hr_feat, lr_feat):
        """top_k > Q should not crash."""
        mod = SubPixelRefineModule(hr_channels=256, lr_channels=256, top_k=999)
        points = torch.rand(2, 10, 2) * 128
        scores = torch.rand(2, 10)
        refined = mod(hr_feat, lr_feat, points, scores, 128, 128)
        assert refined.shape == points.shape

    def test_gradient_flows(self, module, hr_feat, lr_feat, pred_points, pred_scores):
        refined = module(hr_feat, lr_feat, pred_points, pred_scores, 128, 128)
        loss = refined.sum()
        loss.backward()
        # Check MLP has gradients
        assert module.mlp[-1].weight.grad is not None


# ===========================================================================
# DSGCnet integration: SuperNode GCN
# ===========================================================================


class TestDSGCnetSuperNodeGCN:
    @pytest.fixture
    def model(self):
        backbone = TinyVGGBackbone()
        return DSGCnet(
            backbone,
            row=2,
            line=2,
            gcn_mode="supernode",
            gcn_num_supernodes=4,
            gcn_supernode_heads=2,
        ).eval()

    def test_forward_runs(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert "pred_logits" in out
        assert "pred_points" in out
        assert out["pred_logits"].shape[0] == 2
        assert out["pred_points"].shape[0] == 2

    def test_alpha_absent_in_supernode_mode(self, model):
        assert model.alpha is None
        assert model.supernode_gcn is not None

    def test_dual_gcn_absent_in_supernode_mode(self, model):
        assert model.density_gcn is None
        assert model.feature_gcn is None


# ===========================================================================
# DSGCnet integration: FreqDecoupledRouter
# ===========================================================================


class TestDSGCnetFreqRouter:
    @pytest.fixture
    def model(self):
        backbone = TinyVGGBackbone()
        return DSGCnet(
            backbone,
            row=2,
            line=2,
            use_freq_head=True,
            freq_head_kernel=3,
        ).eval()

    def test_forward_runs(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out["pred_logits"].shape[0] == 2
        assert out["pred_points"].shape[0] == 2

    def test_freq_router_present(self, model):
        assert model.freq_router is not None


# ===========================================================================
# DSGCnet integration: SubPixelRefine
# ===========================================================================


class TestDSGCnetSubPixRefine:
    @pytest.fixture
    def model(self):
        backbone = TinyVGGBackbone()
        from omegaconf import OmegaConf

        sp_cfg = OmegaConf.create({"top_k": 8, "hidden_dim": 64})
        return DSGCnet(
            backbone,
            row=2,
            line=2,
            use_subpix_refine=True,
            subpix_refine_cfg=sp_cfg,
        ).eval()

    def test_forward_runs(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out["pred_logits"].shape[0] == 2
        assert out["pred_points"].shape[0] == 2

    def test_subpix_refine_present(self, model):
        assert model.subpix_refine is not None


# ===========================================================================
# DSGCnet integration: all three enabled together
# ===========================================================================


class TestDSGCnetAllThreeEnabled:
    @pytest.fixture
    def model(self):
        backbone = TinyVGGBackbone()
        from omegaconf import OmegaConf

        sp_cfg = OmegaConf.create({"top_k": 8, "hidden_dim": 64})
        return DSGCnet(
            backbone,
            row=2,
            line=2,
            gcn_mode="supernode",
            gcn_num_supernodes=4,
            gcn_supernode_heads=2,
            use_freq_head=True,
            freq_head_kernel=3,
            use_subpix_refine=True,
            subpix_refine_cfg=sp_cfg,
        ).eval()

    def test_forward_runs(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out["pred_logits"].shape[0] == 2
        assert out["pred_points"].shape[0] == 2
        assert out["density_out"].shape[0] == 2

    def test_output_shapes_consistent(self, model, sample_tensor):
        with torch.no_grad():
            out = model(sample_tensor)
        assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]
        assert out["pred_logits"].shape[2] == 2
        assert out["pred_points"].shape[2] == 2

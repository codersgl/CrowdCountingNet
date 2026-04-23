"""Tests for DensityAdaptiveFusion module."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.gcn import DensityAdaptiveFusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_feature_map():
    return torch.randn(2, 256, 8, 8)  # B=2, C=256, H=8, W=8


@pytest.fixture
def small_density_map():
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# Unit tests for DensityAdaptiveFusion
# ---------------------------------------------------------------------------


class TestDensityAdaptiveFusionSpatial:
    """Tests for spatial (per-pixel) mode."""

    def test_output_shape(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=True)
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        assert out.shape == small_feature_map.shape

    def test_weights_start_uniform(self, small_feature_map, small_density_map):
        """At init, weights should be ~[1/3, 1/3, 1/3] (zero-init projection)."""
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=True)
        daf.eval()
        # Check that the final conv has zero weights and bias
        final_conv = daf.weight_proj[-1]
        assert final_conv.weight.abs().max().item() == 0.0
        assert final_conv.bias.abs().max().item() == 0.0

    def test_gradient_flow(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=True)
        daf.train()
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        loss = out.sum()
        loss.backward()
        # Check that density_encoder gets gradients
        assert daf.density_encoder[0].weight.grad is not None
        assert daf.weight_proj[0].weight.grad is not None

    def test_different_density_different_weights(self):
        """Dense vs sparse density maps should produce different fusion weights
        after a few training steps (zero-init means uniform at start)."""
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=True)
        daf.train()

        feat = torch.randn(1, 256, 4, 4)
        density_dense = torch.ones(1, 1, 4, 4) * 10.0  # high density
        density_sparse = torch.ones(1, 1, 4, 4) * 0.1  # low density

        # Run a few training steps to break symmetry
        optimizer = torch.optim.SGD(daf.parameters(), lr=0.01)
        for _ in range(5):
            out = daf(feat, feat, feat, density_dense)
            out.sum().backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        # Now check that density encoder produces different embeddings
        d_emb_dense = daf.density_encoder(density_dense)
        d_emb_sparse = daf.density_encoder(density_sparse)
        assert not torch.allclose(d_emb_dense, d_emb_sparse, atol=1e-4)

    def test_custom_embed_dim(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=32, spatial=True)
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        assert out.shape == small_feature_map.shape
        assert daf.density_embed_dim == 32


class TestDensityAdaptiveFusionGlobal:
    """Tests for per-image (global) mode."""

    def test_output_shape(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=False)
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        assert out.shape == small_feature_map.shape

    def test_weights_start_uniform(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=False)
        # Final linear layer should be zero-initialised
        final_linear = daf.weight_mlp[-1]
        assert final_linear.weight.abs().max().item() == 0.0
        assert final_linear.bias.abs().max().item() == 0.0

    def test_global_weights_sum_to_one(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=False)
        daf.eval()
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        # Weights are softmax → should sum to 1 per sample
        d_emb = daf.density_encoder(small_density_map)
        combined = torch.cat([
            daf.gap(small_feature_map),
            daf.gap(small_feature_map),
            daf.gap(small_feature_map),
            daf.gap(d_emb),
        ], dim=1).flatten(1)
        weights = torch.softmax(daf.weight_mlp(combined), dim=1)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_gradient_flow(self, small_feature_map, small_density_map):
        daf = DensityAdaptiveFusion(in_channels=256, density_embed_dim=64, spatial=False)
        daf.train()
        out = daf(small_feature_map, small_feature_map, small_feature_map, small_density_map)
        loss = out.sum()
        loss.backward()
        assert daf.density_encoder[0].weight.grad is not None
        assert daf.weight_mlp[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Integration test: DSGCnet with DensityAdaptiveFusion
# ---------------------------------------------------------------------------


def _make_tiny_vgg():
    """Create a minimal VGG16-BN backbone for testing."""
    from crowdcount.models.backbone import Backbone_VGG

    return Backbone_VGG("vgg16_bn", return_interm_layers=True)


def test_dsgcnet_with_density_adaptive_fusion():
    """Smoke test: DSGCnet with use_density_adaptive_fusion=True."""
    from crowdcount.models.dsgcnet import DSGCnet
    from omegaconf import OmegaConf

    backbone = _make_tiny_vgg()
    daf_cfg = OmegaConf.create({"density_embed_dim": 64, "spatial": True})
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="gcn",
        use_density_adaptive_fusion=True,
        density_adaptive_fusion_cfg=daf_cfg,
    )
    model.eval()

    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "density_out" in out
    assert "pred_logits" in out
    assert "pred_points" in out

    # Verify alpha is None (disabled by DAF)
    assert model.alpha is None
    assert model.gm is None
    assert model.density_adaptive_fusion is not None


def test_dsgcnet_with_density_adaptive_fusion_global():
    """Smoke test: DSGCnet with spatial=False (global mode)."""
    from crowdcount.models.dsgcnet import DSGCnet
    from omegaconf import OmegaConf

    backbone = _make_tiny_vgg()
    daf_cfg = OmegaConf.create({"density_embed_dim": 32, "spatial": False})
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="gcn",
        use_density_adaptive_fusion=True,
        density_adaptive_fusion_cfg=daf_cfg,
    )
    model.eval()

    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "density_out" in out
    assert model.density_adaptive_fusion is not None
    assert model.density_adaptive_fusion.spatial is False


def test_dsgcnet_without_density_adaptive_fusion():
    """Baseline: DSGCnet without DAF should still work (alpha-based fusion)."""
    from crowdcount.models.dsgcnet import DSGCnet

    backbone = _make_tiny_vgg()
    model = DSGCnet(backbone, row=2, line=2, fusion_mode="gcn")
    model.eval()

    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "density_out" in out
    assert model.alpha is not None
    assert model.density_adaptive_fusion is None

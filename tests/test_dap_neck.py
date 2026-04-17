"""Tests for DAP-Neck: PEEM, DPGA, ACDR, and full DAPNeck integration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.models.dap_neck import ACDR, DPGA, PEEM, DAPNeck, PixelShuffleUpsample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def feat_c3():
    """C3-scale feature: [B=2, 256, 32, 32] (H/4 for 128 input)."""
    return torch.randn(2, 256, 32, 32)


@pytest.fixture
def feat_c4():
    """C4-scale feature: [B=2, 512, 16, 16] (H/8 for 128 input)."""
    return torch.randn(2, 512, 16, 16)


@pytest.fixture
def feat_c5():
    """C5-scale feature: [B=2, 512, 8, 8] (H/16 for 128 input)."""
    return torch.randn(2, 512, 8, 8)


# ---------------------------------------------------------------------------
# PEEM tests
# ---------------------------------------------------------------------------


class TestPEEM:
    def test_output_shape_preserves_input(self):
        peem = PEEM(channels=256, freq_cutoff=0.25)
        x = torch.randn(2, 256, 32, 32)
        out = peem(x)
        assert out.shape == x.shape

    def test_output_shape_different_channels(self):
        peem = PEEM(channels=512, freq_cutoff=0.3)
        x = torch.randn(1, 512, 16, 16)
        out = peem(x)
        assert out.shape == x.shape

    def test_handles_zero_input(self):
        peem = PEEM(channels=256)
        x = torch.zeros(1, 256, 8, 8)
        out = peem(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_without_dcn(self):
        peem = PEEM(channels=256, use_dcn=False)
        x = torch.randn(1, 256, 16, 16)
        out = peem(x)
        assert out.shape == x.shape

    def test_small_spatial(self):
        """PEEM should work even on very small spatial dims."""
        peem = PEEM(channels=512, freq_cutoff=0.25)
        x = torch.randn(1, 512, 4, 4)
        out = peem(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# DPGA tests
# ---------------------------------------------------------------------------


class TestDPGA:
    def test_output_shape(self):
        dpga = DPGA(dim=256, num_heads=4, sigma_list=[1.0, 2.0, 4.0])
        q = torch.randn(2, 256, 16, 16)
        kv = torch.randn(2, 256, 16, 16)
        out = dpga(q, kv)
        assert out.shape == q.shape

    def test_different_sigma_counts(self):
        dpga = DPGA(dim=256, num_heads=4, sigma_list=[0.5, 1.0])
        q = torch.randn(1, 256, 8, 8)
        kv = torch.randn(1, 256, 8, 8)
        out = dpga(q, kv)
        assert out.shape == q.shape

    def test_with_pooling(self):
        dpga = DPGA(dim=256, num_heads=4, max_pool_size=8)
        q = torch.randn(1, 256, 16, 16)
        kv = torch.randn(1, 256, 16, 16)
        out = dpga(q, kv)
        assert out.shape == q.shape

    def test_residual_at_init(self):
        """At initialisation, gate=0 so output should be close to query_feat."""
        dpga = DPGA(dim=64, num_heads=4)
        q = torch.randn(1, 64, 8, 8)
        kv = torch.randn(1, 64, 8, 8)
        out = dpga(q, kv)
        # gate starts at 0 → tanh(0) = 0 → output ≈ query_feat
        assert torch.allclose(out, q, atol=1e-6)

    def test_single_head(self):
        dpga = DPGA(dim=256, num_heads=1)
        q = torch.randn(1, 256, 8, 8)
        kv = torch.randn(1, 256, 8, 8)
        out = dpga(q, kv)
        assert out.shape == q.shape

    def test_pools_large_spatial_dims(self):
        """DPGA should handle large spatial dims via adaptive pooling."""
        dpga = DPGA(dim=64, num_heads=4, max_pool_size=8)
        q = torch.randn(1, 64, 64, 48)  # much larger than max_pool_size
        kv = torch.randn(1, 64, 64, 48)
        out = dpga(q, kv)
        assert out.shape == q.shape

    def test_rejects_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            DPGA(dim=256, num_heads=3)


# ---------------------------------------------------------------------------
# ACDR tests
# ---------------------------------------------------------------------------


class TestACDR:
    def test_output_shape(self):
        acdr = ACDR(channels=256, large_kernel=7, dilation=2)
        x = torch.randn(2, 256, 16, 16)
        out = acdr(x)
        assert out.shape == x.shape

    def test_crowdedness_range(self):
        """Crowdedness estimator should output values in [0, 1]."""
        acdr = ACDR(channels=256)
        x = torch.randn(4, 256, 16, 16)
        c = acdr.crowd_est(x)
        assert c.shape == (4, 1)
        assert (c >= 0).all() and (c <= 1).all()

    def test_different_kernel_sizes(self):
        for k in [3, 5, 7, 11]:
            acdr = ACDR(channels=128, large_kernel=k, dilation=2)
            x = torch.randn(1, 128, 8, 8)
            out = acdr(x)
            assert out.shape == x.shape

    def test_handles_small_spatial(self):
        acdr = ACDR(channels=256, large_kernel=7, dilation=2)
        x = torch.randn(1, 256, 4, 4)
        out = acdr(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# PixelShuffleUpsample tests
# ---------------------------------------------------------------------------


class TestPixelShuffleUpsample:
    def test_doubles_spatial(self):
        up = PixelShuffleUpsample(channels=256)
        x = torch.randn(1, 256, 8, 8)
        out = up(x)
        assert out.shape == (1, 256, 16, 16)


# ---------------------------------------------------------------------------
# DAPNeck end-to-end tests
# ---------------------------------------------------------------------------


class TestDAPNeck:
    def test_output_shape(self, feat_c3, feat_c4, feat_c5):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512, feature_size=256)
        out = neck([feat_c3, feat_c4, feat_c5])
        # Output should be at C4 resolution (H/8)
        assert out.shape == (2, 256, 16, 16)

    def test_output_shape_without_bottom_up(self, feat_c3, feat_c4, feat_c5):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512, use_bottom_up=False)
        out = neck([feat_c3, feat_c4, feat_c5])
        assert out.shape == (2, 256, 16, 16)

    def test_output_shape_with_peem_on_c5(self, feat_c3, feat_c4, feat_c5):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512, peem_on_c5=True)
        out = neck([feat_c3, feat_c4, feat_c5])
        assert out.shape == (2, 256, 16, 16)

    def test_output_shape_with_small_pool(self, feat_c3, feat_c4, feat_c5):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512, dpga_max_pool_size=8)
        out = neck([feat_c3, feat_c4, feat_c5])
        assert out.shape == (2, 256, 16, 16)

    def test_handles_zero_input(self):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512)
        inputs = [
            torch.zeros(1, 256, 32, 32),
            torch.zeros(1, 512, 16, 16),
            torch.zeros(1, 512, 8, 8),
        ]
        out = neck(inputs)
        assert out.shape == (1, 256, 16, 16)
        assert torch.isfinite(out).all()

    def test_backward_pass(self, feat_c3, feat_c4, feat_c5):
        neck = DAPNeck(C3_size=256, C4_size=512, C5_size=512)
        feat_c3.requires_grad_(True)
        out = neck([feat_c3, feat_c4, feat_c5])
        loss = out.sum()
        loss.backward()
        assert feat_c3.grad is not None


# ---------------------------------------------------------------------------
# DSGCnet integration tests
# ---------------------------------------------------------------------------


class TinyVGGBackbone(nn.Module):
    """Minimal 4-stage backbone mirroring VGG16-BN channels."""

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


class TestDAPNeckIntegration:
    def test_dsgcnet_with_dap_neck(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyVGGBackbone()
        model = DSGCnet(backbone, row=2, line=2, use_dap_neck=True)
        model.eval()
        x = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert "pred_logits" in out
        assert "pred_points" in out
        assert "density_out" in out
        assert out["pred_logits"].shape[0] == 1
        assert out["pred_points"].shape[2] == 2

    def test_dsgcnet_dap_neck_mutual_exclusion(self):
        from crowdcount.models.dsgcnet import DSGCnet

        backbone = TinyVGGBackbone()
        with pytest.raises(ValueError, match="mutually exclusive"):
            DSGCnet(backbone, row=2, line=2, use_dap_neck=True, use_msca_neck=True)

    def test_dsgcnet_dap_neck_with_custom_config(self):
        from omegaconf import OmegaConf
        from crowdcount.models.dsgcnet import DSGCnet

        dap_cfg = OmegaConf.create(
            {
                "freq_cutoff": 0.3,
                "peem_on_c5": True,
                "num_heads": 2,
                "sigma_list": [1.0, 3.0],
                "dpga_max_pool_size": 16,
                "acdr_large_kernel": 5,
                "acdr_dilation": 1,
                "use_bottom_up": False,
            }
        )
        backbone = TinyVGGBackbone()
        model = DSGCnet(
            backbone, row=2, line=2, use_dap_neck=True, dap_neck_cfg=dap_cfg
        )
        model.eval()
        x = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out["density_out"].shape == (1, 1, 16, 16)

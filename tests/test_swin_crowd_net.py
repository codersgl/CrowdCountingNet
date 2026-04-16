"""Tests for SwinCrowdNet: Swin-B + CrowdFPN + MoE-Lite."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from crowdcount.models.crowd_fpn import CrowdFPN
from crowdcount.models.moe_lite import (
    BoundaryExpert,
    DenseRegionExpert,
    MoELite,
    SparseRegionExpert,
)
from crowdcount.models.swin_crowd_net import SwinCrowdNet


# ---------------------------------------------------------------------------
# Tiny backbone mock (avoids downloading Swin weights in CI)
# ---------------------------------------------------------------------------


class TinySwinBackbone(nn.Module):
    """Minimal mock that mimics BackboneSwin output contract."""

    def __init__(self) -> None:
        super().__init__()
        # Tiny conv layers just to have parameters
        self.s1 = nn.Conv2d(3, 256, 3, stride=4, padding=1)
        self.s2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.s3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c2 = self.s1(x)  # [B, 256, H/4, W/4]
        c3 = self.s2(c2)  # [B, 256, H/8, W/8]
        c4 = self.s3(c3)  # [B, 512, H/16, W/16]
        return [c2, c3, c4]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def swin_model(device):
    """Build SwinCrowdNet with tiny mock backbone."""
    backbone = TinySwinBackbone()
    model = SwinCrowdNet(
        backbone=backbone,
        row=2,
        line=2,
        feature_dim=256,
        moe_grid_stride=2,
        moe_temperature_init=1.0,
        moe_lambda_balance=0.01,
    )
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_input(device):
    """[B=2, 3, 128, 128] input tensor."""
    return torch.randn(2, 3, 128, 128, device=device)


# ---------------------------------------------------------------------------
# CrowdFPN tests
# ---------------------------------------------------------------------------


class TestCrowdFPN:
    def test_output_shape(self, device):
        fpn = CrowdFPN(C2_channels=256, C3_channels=256, C4_channels=512).to(device)
        c2 = torch.randn(2, 256, 32, 32, device=device)  # stride 4
        c3 = torch.randn(2, 256, 16, 16, device=device)  # stride 8
        c4 = torch.randn(2, 512, 8, 8, device=device)  # stride 16
        out = fpn([c2, c3, c4])
        assert out.shape == (2, 256, 16, 16), (
            f"Expected stride-8 output, got {out.shape}"
        )

    def test_single_batch(self, device):
        fpn = CrowdFPN().to(device)
        c2 = torch.randn(1, 256, 16, 16, device=device)
        c3 = torch.randn(1, 256, 8, 8, device=device)
        c4 = torch.randn(1, 512, 4, 4, device=device)
        out = fpn([c2, c3, c4])
        assert out.shape == (1, 256, 8, 8)


# ---------------------------------------------------------------------------
# Expert tests
# ---------------------------------------------------------------------------


class TestExperts:
    def test_dense_expert_residual(self, device):
        expert = DenseRegionExpert(dim=64).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = expert(x)
        assert out.shape == x.shape

    def test_sparse_expert_residual(self, device):
        expert = SparseRegionExpert(dim=64).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = expert(x)
        assert out.shape == x.shape

    def test_boundary_expert_residual(self, device):
        expert = BoundaryExpert(dim=64).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = expert(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# MoE-Lite tests
# ---------------------------------------------------------------------------


class TestMoELite:
    def test_forward_shape(self, device):
        moe = MoELite(dim=64, grid_stride=2).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        density = torch.randn(2, 1, 8, 8, device=device).abs()
        fused, aux, weights = moe(x, density, training=True)
        assert fused.shape == x.shape
        assert weights.shape == (2, 3, 8, 8)
        assert "l_balance" in aux
        assert "total_aux" in aux

    def test_weights_sum_to_one(self, device):
        moe = MoELite(dim=64, grid_stride=2).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        density = torch.randn(2, 1, 8, 8, device=device).abs()
        _, _, weights = moe(x, density)
        weight_sum = weights.sum(dim=1)
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)

    def test_temperature_decay(self, device):
        moe = MoELite(dim=64, temperature_init=1.0, temperature_min=0.3).to(device)
        t_before = moe.router.temperature.item()
        for _ in range(100):
            moe.update_temperature(decay_rate=0.99)
        t_after = moe.router.temperature.item()
        assert t_after < t_before
        assert t_after >= 0.3

    def test_residual_gate_init(self, device):
        """Beta starts at 0 → sigmoid(0) ≈ 0.5, so MoE starts at midpoint."""
        moe = MoELite(dim=64).to(device)
        assert moe.beta.item() == 0.0

    def test_eval_mode_no_aux(self, device):
        moe = MoELite(dim=64, grid_stride=2).to(device)
        moe.eval()
        x = torch.randn(1, 64, 4, 4, device=device)
        density = torch.randn(1, 1, 4, 4, device=device).abs()
        _, aux, _ = moe(x, density, training=False)
        assert aux["total_aux"].item() == 0.0


# ---------------------------------------------------------------------------
# SwinCrowdNet end-to-end tests
# ---------------------------------------------------------------------------


class TestSwinCrowdNet:
    def test_output_keys(self, swin_model, sample_input):
        out = swin_model(sample_input)
        required_keys = {"pred_logits", "pred_points", "density_out"}
        assert required_keys.issubset(out.keys())

    def test_output_shapes(self, swin_model, sample_input):
        out = swin_model(sample_input)
        B = sample_input.shape[0]
        # pred_logits: [B, Q, 2]
        assert out["pred_logits"].dim() == 3
        assert out["pred_logits"].shape[0] == B
        assert out["pred_logits"].shape[2] == 2
        # pred_points: [B, Q, 2]
        assert out["pred_points"].dim() == 3
        assert out["pred_points"].shape[0] == B
        assert out["pred_points"].shape[2] == 2
        # density_out: [B, 1, H', W']
        assert out["density_out"].dim() == 4
        assert out["density_out"].shape[0] == B
        assert out["density_out"].shape[1] == 1

    def test_moe_interface(self, swin_model):
        assert swin_model.supports_moe() is True
        params = swin_model.get_moe_gating_parameters()
        assert len(params) > 0
        # Temperature update should not error
        swin_model.update_moe_temperature(0.999)

    def test_moe_weights_in_output(self, swin_model, sample_input):
        swin_model.train()
        out = swin_model(sample_input)
        assert out["moe_weights"] is not None
        assert out["moe_aux_losses"] is not None
        assert "total_aux" in out["moe_aux_losses"]

    def test_backward_pass(self, swin_model, sample_input):
        """Ensure gradients flow through the entire model."""
        swin_model.train()
        out = swin_model(sample_input)
        loss = (
            out["pred_logits"].sum()
            + out["pred_points"].sum()
            + out["density_out"].sum()
        )
        loss.backward()
        # Check backbone has gradients
        for p in swin_model.backbone.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_single_batch(self, device):
        """Evaluation mode with batch_size=1."""
        backbone = TinySwinBackbone()
        model = SwinCrowdNet(backbone=backbone, row=2, line=2).to(device)
        model.eval()
        x = torch.randn(1, 3, 128, 128, device=device)
        out = model(x)
        assert out["pred_logits"].shape[0] == 1


# ---------------------------------------------------------------------------
# build_model integration test
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_build_swin_crowd_net(self):
        """Test that build_model dispatches to SwinCrowdNet correctly."""
        cfg = OmegaConf.create(
            {
                "model": {
                    "architecture": "swin_crowd_net",
                    "backbone": "swin_base",
                    "backbone_pretrained": False,  # no download in tests
                    "row": 2,
                    "line": 2,
                    "feature_dim": 256,
                    "fpn_c2_channels": 256,
                    "fpn_c3_channels": 256,
                    "fpn_c4_channels": 512,
                    "set_cost_class": 1.0,
                    "set_cost_point": 0.05,
                    "point_loss_coef": 0.0002,
                    "eos_coef": 0.5,
                    "count_loss_coef": 0.0,
                    "consistency_loss_coef": 0.0,
                    "use_focal_loss": False,
                    "moe": {
                        "grid_stride": 4,
                        "temperature_init": 1.0,
                        "temperature_min": 0.3,
                        "lambda_balance": 0.01,
                        "dense_expansion": 2,
                    },
                },
                "uncertainty_weighting": {"enabled": False},
            }
        )
        from crowdcount.models import build_model

        model, criterion, uw = build_model(cfg, training=True)
        assert isinstance(model, SwinCrowdNet)
        assert uw is None

    def test_build_default_dsgcnet(self, base_cfg):
        """Default architecture still builds DSGCNet."""
        from crowdcount.models import build_model
        from crowdcount.models.dsgcnet import DSGCnet

        model = build_model(base_cfg, training=False)
        assert isinstance(model, DSGCnet)

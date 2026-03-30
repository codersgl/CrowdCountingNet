"""Tests for the Iterative Point Refinement Decoder (PointRefineModule)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crowdcount.models.head import PointRefineModule


# ---------------------------------------------------------------------------
# PointRefineModule unit tests
# ---------------------------------------------------------------------------


class TestPointRefineModule:
    def test_output_shapes(self):
        mod = PointRefineModule(feature_dim=256, hidden_dim=256, num_steps=2)
        feat = torch.randn(2, 256, 8, 8)
        pts = torch.rand(2, 64, 2) * 128
        refined, intermediates = mod(feat, pts, img_h=128, img_w=128)
        assert refined.shape == (2, 64, 2)
        assert len(intermediates) == 3  # init + 2 steps

    def test_zero_steps_returns_input(self):
        mod = PointRefineModule(feature_dim=256, hidden_dim=256, num_steps=0)
        feat = torch.randn(1, 256, 8, 8)
        pts = torch.rand(1, 16, 2) * 128
        refined, intermediates = mod(feat, pts, img_h=128, img_w=128)
        assert torch.allclose(refined, pts)
        assert len(intermediates) == 1

    def test_gradients_flow(self):
        mod = PointRefineModule(feature_dim=256, hidden_dim=128, num_steps=2)
        feat = torch.randn(2, 256, 8, 8, requires_grad=True)
        pts = torch.rand(2, 16, 2) * 128
        refined, _ = mod(feat, pts, img_h=128, img_w=128)
        loss = refined.sum()
        loss.backward()
        assert feat.grad is not None
        assert feat.grad.abs().sum() > 0

    def test_unshared_weights(self):
        mod = PointRefineModule(
            feature_dim=256,
            hidden_dim=128,
            num_steps=2,
            share_weights=False,
        )
        feat = torch.randn(1, 256, 8, 8)
        pts = torch.rand(1, 16, 2) * 128
        refined, intermediates = mod(feat, pts, img_h=128, img_w=128)
        assert refined.shape == (1, 16, 2)
        assert len(intermediates) == 3

    def test_refinement_changes_coords(self):
        """After training-like forward, refined coords should differ from init."""
        mod = PointRefineModule(feature_dim=256, hidden_dim=128, num_steps=2)
        feat = torch.randn(1, 256, 8, 8)
        pts = torch.rand(1, 16, 2) * 128
        refined, _ = mod(feat, pts, img_h=128, img_w=128)
        # At least some coordinates should change (MLP isn't zero-initialized)
        assert not torch.allclose(refined, pts, atol=1e-6)

    def test_intermediate_shapes(self):
        mod = PointRefineModule(feature_dim=256, hidden_dim=128, num_steps=3)
        feat = torch.randn(1, 256, 8, 8)
        pts = torch.rand(1, 32, 2) * 128
        _, intermediates = mod(feat, pts, img_h=128, img_w=128)
        assert len(intermediates) == 4  # init + 3 steps
        for step_pts in intermediates:
            assert step_pts.shape == (1, 32, 2)


# ---------------------------------------------------------------------------
# Integration: DSGCnet with refinement
# ---------------------------------------------------------------------------


class TestRefineIntegration:
    def test_dsgcnet_forward_with_refine(self, base_cfg, sample_batch, device):
        """DSGCnet with use_refine=True should include intermediates in output."""
        cfg = OmegaConf.merge(base_cfg, {"model": {"use_refine": True}})
        from crowdcount.models import build_model

        model = build_model(cfg, training=False)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch.to(device))

        assert "refine_intermediates" in out
        assert out["refine_intermediates"] is not None
        assert len(out["refine_intermediates"]) == 3  # init + 2 steps
        assert out["pred_points"].shape[0] == sample_batch.shape[0]
        assert out["pred_points"].shape[2] == 2

    def test_dsgcnet_forward_without_refine(self, base_cfg, sample_batch, device):
        """DSGCnet with use_refine=False should have refine_intermediates=None."""
        from crowdcount.models import build_model

        model = build_model(base_cfg, training=False)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch.to(device))

        assert out["refine_intermediates"] is None

    def test_refine_loss_in_criterion(
        self, base_cfg, sample_batch, dummy_targets, device
    ):
        """Criterion should compute loss_refine when intermediates are present."""
        cfg = OmegaConf.merge(
            base_cfg,
            {
                "model": {"use_refine": True},
                "refine_loss_weight": 0.001,
            },
        )
        from crowdcount.models import build_model

        model, criterion = build_model(cfg, training=True)
        model = model.to(device)
        criterion = criterion.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_batch.to(device))

        targets = [{k: v.to(device) for k, v in t.items()} for t in dummy_targets]
        loss_dict = criterion(outputs, targets)

        assert "loss_refine" in loss_dict
        assert loss_dict["loss_refine"].dim() == 0
        # Weight should be in weight_dict
        assert "loss_refine" in criterion.weight_dict

    def test_refine_loss_zero_when_disabled(
        self, base_cfg, sample_batch, dummy_targets, device
    ):
        """When use_refine=False, loss_refine should not be in losses list."""
        from crowdcount.models import build_model

        model, criterion = build_model(base_cfg, training=True)
        model = model.to(device)
        criterion = criterion.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_batch.to(device))

        targets = [{k: v.to(device) for k, v in t.items()} for t in dummy_targets]
        loss_dict = criterion(outputs, targets)

        # "refine" not in losses list → loss_refine not computed
        assert "loss_refine" not in loss_dict

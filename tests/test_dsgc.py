"""End-to-end forward pass tests for DSGCnet.

Uses a tiny synthetic backbone to avoid downloading pretrained weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.plugins.gm import GateMechanism


class TinyVGGBackbone(nn.Module):
    """Minimal 4-stage backbone that mirrors the VGG16-BN strides/channels.

    Real VGG16-BN splits (128×128 input):
      features[:13]  → 128ch, H/2  (body1)
      features[13:23]→ 256ch, H/4  (body2)
      features[23:33]→ 512ch, H/8  (body3)
      features[33:43]→ 512ch, H/16 (body4)
    DSGCNet uses features[1..3] for the PA-FPN, so the spatial sizes here
    must match those three stages.
    """

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),  # body1: stride 2, 128ch
            torch.zeros(B, 256, H // 4, W // 4),  # body2: stride 4, 256ch
            torch.zeros(B, 512, H // 8, W // 8),  # body3: stride 8, 512ch
            torch.zeros(B, 512, H // 16, W // 16),  # body4: stride 16, 512ch
        ]


@pytest.fixture
def model():
    backbone = TinyVGGBackbone()
    return DSGCnet(backbone, row=2, line=2)


@pytest.fixture
def sample_tensor():
    return torch.zeros(1, 3, 128, 128)


# ---------------------------------------------------------------------------
# Output keys
# ---------------------------------------------------------------------------


def test_forward_output_keys(model, sample_tensor):
    model.eval()
    with torch.no_grad():
        out = model(sample_tensor)
    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out


def test_pred_logits_shape(model, sample_tensor):
    model.eval()
    with torch.no_grad():
        out = model(sample_tensor)
    # B=1, num_queries, num_classes=2
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_logits"].shape[2] == 2


def test_pred_points_shape(model, sample_tensor):
    model.eval()
    with torch.no_grad():
        out = model(sample_tensor)
    # B=1, num_queries, 2 (x, y)
    assert out["pred_points"].shape[0] == 1
    assert out["pred_points"].shape[2] == 2
    assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]


def test_density_out_non_negative(model, sample_tensor):
    model.eval()
    with torch.no_grad():
        out = model(sample_tensor)
    assert (out["density_out"] >= 0).all()


def test_batch_consistency(model):
    """Outputs for different batch sizes should have consistent query counts."""
    model.eval()
    with torch.no_grad():
        out1 = model(torch.zeros(1, 3, 128, 128))
        out2 = model(torch.zeros(2, 3, 128, 128))
    assert out1["pred_logits"].shape[1] == out2["pred_logits"].shape[1]


def test_alpha_learnable(model):
    """alpha parameters should be learnable."""
    assert model.alpha.requires_grad


def test_gate_mechanism_initialized_when_enabled() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_gm=True)
    assert model.gm is not None
    assert isinstance(model.gm, GateMechanism)


def test_gate_mechanism_disabled_when_false() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_gm=False)
    assert model.gm is None


def test_gate_mechanism_forward_shapes_match() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_gm=True).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape[0] == 2


def test_gate_weight_is_valid_probability_distribution() -> None:
    gm = GateMechanism(input_dim=256, hidden_dim=128).eval()
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        gate_weight = gm(x)

    assert gate_weight.shape == (2, 3)
    row_sums = gate_weight.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)

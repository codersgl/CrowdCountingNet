"""Tests for dual-stream deformable attention fusion."""

from __future__ import annotations

import torch
from torch import nn

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.plugins.deformable_dual import (
    DeformableDualFusion,
    GuidedDeformableAttention,
)


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _channels, height, width = x.shape
        return [
            x.new_zeros(batch_size, 128, height // 2, width // 2),
            x.new_zeros(batch_size, 256, height // 4, width // 4),
            x.new_zeros(batch_size, 512, height // 8, width // 8),
            x.new_zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_guided_deformable_attention_shape_and_gradient() -> None:
    feature = torch.randn(2, 64, 8, 8, requires_grad=True)
    density = torch.rand(2, 1, 16, 16)
    module = GuidedDeformableAttention(
        in_channels=64,
        num_points=4,
        num_heads=4,
        use_density_guidance=True,
        dropout=0.0,
    )

    out, aux = module(feature, density=density)
    assert out.shape == feature.shape
    assert aux["residual_offset_abs_max"].item() == 0.0
    out.mean().backward()
    assert feature.grad is not None
    assert module.offset_pred[-1].bias.grad is not None


def test_guided_deformable_attention_initial_base_offsets() -> None:
    feature = torch.randn(1, 32, 6, 6)
    density = torch.rand(1, 1, 6, 6)
    module = GuidedDeformableAttention(
        in_channels=32,
        num_points=4,
        num_heads=4,
        use_density_guidance=True,
        dropout=0.0,
    )

    _out, aux = module(feature, density=density)
    assert torch.allclose(aux["residual_offset_abs_max"], torch.tensor(0.0))
    assert torch.allclose(aux["total_offset_abs_max"], torch.tensor(1.0))
    assert aux["density_gamma"].item() == 0.5


def test_deformable_dual_fusion_initial_gate_weights() -> None:
    feature = torch.randn(2, 64, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    module = DeformableDualFusion(
        in_channels=64,
        num_points=4,
        num_heads=4,
        dropout=0.0,
        fusion_init_weights=(0.8, 0.1, 0.1),
    )

    out, aux = module(feature, density)
    assert out.shape == feature.shape
    weights = aux["fusion_weights"]
    assert weights.shape == (2, 3, 8, 8)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 8, 8), atol=1e-6)
    mean_weights = weights.mean(dim=(0, 2, 3))
    assert torch.allclose(mean_weights, torch.tensor([0.8, 0.1, 0.1]), atol=1e-6)


def test_dsgcnet_deformable_dual_forward() -> None:
    model = DSGCnet(TinyVGGBackbone(), row=2, line=2, fusion_mode="deformable_dual")
    model.eval()
    assert model.deformable_dual_fusion is not None
    assert model.alpha is None
    assert model.density_gcn is None
    assert model.feature_gcn is None

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[0] == 1
    assert out["density_out"].shape[0] == 1
    assert "deformable_dual_aux" in out
    fusion_weights = out["deformable_dual_aux"]["fusion_weights"]
    assert fusion_weights.shape[1] == 3
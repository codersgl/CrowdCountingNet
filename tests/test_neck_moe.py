"""Tests for scale-aware Neck MoE adapter."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.plugins.neck_moe import (
    CrossScaleExpert,
    NeckScaleMoE,
    ScaleAwareGridRouter,
)


@pytest.fixture
def neck_feature() -> torch.Tensor:
    return torch.randn(2, 32, 16, 16)


@pytest.fixture
def pyramid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 32, 8, 8),
    )


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2),
            torch.zeros(batch_size, 256, height // 4, width // 4),
            torch.zeros(batch_size, 512, height // 8, width // 8),
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_router_weights_sum_to_one(neck_feature, pyramid) -> None:
    router = ScaleAwareGridRouter(
        channels=32,
        num_experts=4,
        grid_stride=4,
        use_pyramid_context=True,
    ).eval()
    with torch.no_grad():
        weights = router(neck_feature, pyramid)
    assert weights.shape == (2, 4, 16, 16)
    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(2, 16, 16),
        atol=1e-6,
    )


def test_router_handles_missing_pyramid_context(neck_feature) -> None:
    router = ScaleAwareGridRouter(
        channels=32,
        num_experts=4,
        grid_stride=4,
        use_pyramid_context=True,
    ).eval()
    with torch.no_grad():
        weights = router(neck_feature, pyramid=None)
    assert weights.shape == (2, 4, 16, 16)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 16, 16), atol=1e-6)


def test_cross_scale_expert_shape(neck_feature, pyramid) -> None:
    expert = CrossScaleExpert(channels=32).eval()
    with torch.no_grad():
        out = expert(neck_feature, pyramid)
    assert out.shape == neck_feature.shape


def test_neck_scale_moe_output_shape(neck_feature, pyramid) -> None:
    moe = NeckScaleMoE(in_channels=32, num_experts=4, grid_stride=4).eval()
    with torch.no_grad():
        out, aux, weights = moe(neck_feature, pyramid=pyramid, training=False)
    assert out.shape == neck_feature.shape
    assert aux == {}
    assert weights.shape == (2, 4, 16, 16)


def test_neck_scale_moe_small_input() -> None:
    moe = NeckScaleMoE(in_channels=32, num_experts=4, grid_stride=4).eval()
    x = torch.randn(2, 32, 2, 2)
    with torch.no_grad():
        out, _, weights = moe(x, training=False)
    assert out.shape == x.shape
    assert weights.shape == (2, 4, 2, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 2, 2), atol=1e-6)


def test_neck_scale_moe_beta_gate_starts_identity(neck_feature, pyramid) -> None:
    moe = NeckScaleMoE(in_channels=32, num_experts=4, gate_init=0.0).eval()
    assert moe.beta.item() == 0.0
    with torch.no_grad():
        out, _, _ = moe(neck_feature, pyramid=pyramid, training=False)
    assert torch.allclose(out, neck_feature, atol=1e-6)


def test_neck_scale_moe_aux_grad_flows(neck_feature, pyramid) -> None:
    moe = NeckScaleMoE(in_channels=32, num_experts=4, grid_stride=4).train()
    out, aux, _ = moe(neck_feature, pyramid=pyramid, training=True)
    loss = out.mean() + aux["total_aux"]
    loss.backward()
    router_grads = [p.grad for p in moe.router.parameters() if p.grad is not None]
    assert len(router_grads) > 0
    assert any(grad.abs().sum().item() > 0 for grad in router_grads)
    assert moe.beta.grad is not None


def test_neck_scale_moe_topk_routing_renormalizes(neck_feature, pyramid) -> None:
    moe = NeckScaleMoE(
        in_channels=32,
        num_experts=4,
        routing="topk",
        top_k=2,
        grid_stride=4,
    ).eval()
    with torch.no_grad():
        _, _, weights = moe(neck_feature, pyramid=pyramid, training=False)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 16, 16), atol=1e-6)
    active = (weights > 0).sum(dim=1)
    assert active.max().item() <= 2


def test_dsgcnet_default_neck_moe_smoke() -> None:
    model = DSGCnet(TinyVGGBackbone(), row=2, line=2, use_neck_moe=True).eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["density_out"].shape == (1, 1, 16, 16)
    assert out["neck_moe_weights"].shape == (1, 4, 16, 16)
    assert out["moe_weights"].shape == (1, 4, 16, 16)


def test_dsgcnet_dap_neck_moe_smoke() -> None:
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_dap_neck=True,
        use_neck_moe=True,
    ).eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["density_out"].shape == (1, 1, 16, 16)
    assert out["neck_moe_weights"].shape == (1, 4, 16, 16)


def test_dsgcnet_bifpn_neck_moe_smoke() -> None:
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_bifpn_neck=True,
        use_neck_moe=True,
    ).eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["density_out"].shape == (1, 1, 16, 16)
    assert out["neck_moe_weights"].shape == (1, 4, 16, 16)


def test_dsgcnet_neck_moe_post_acdr_position() -> None:
    cfg = OmegaConf.create({"position": "post_acdr"})
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_dap_neck=True,
        use_neck_moe=True,
        neck_moe_cfg=cfg,
    ).eval()
    assert model.neck_moe_position == "post_acdr"
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["neck_moe_weights"].shape == (1, 4, 16, 16)


def test_dsgcnet_neck_moe_rejects_msca_decoder() -> None:
    with pytest.raises(ValueError, match="use_msca_decoder"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            use_msca_decoder=True,
            use_neck_moe=True,
        )


def test_dsgcnet_neck_moe_rejects_fusion_moe_by_default() -> None:
    with pytest.raises(ValueError, match="fusion_mode='gcn'"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            fusion_mode="sdd_moe",
            use_neck_moe=True,
        )


def test_dsgcnet_neck_moe_requires_supported_pyramid_context() -> None:
    with pytest.raises(ValueError, match="use_pyramid_context"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            use_rccformer_neck=True,
            use_neck_moe=True,
        )

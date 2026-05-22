"""Tests for LFEM and the LFEM multi-scale neck."""

from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.plugins.LFEM import CALayer, GatedWeightGenerator, LFEM
from crowdcount.plugins.lfem_neck import LFEMMultiScaleNeck


class ConstantBranch(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


class FixedGate(nn.Module):
    def __init__(self, weights: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32).view(1, -1, 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights.to(device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1, -1, -1
        )


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2, device=x.device),
            torch.zeros(batch_size, 256, height // 4, width // 4, device=x.device),
            torch.zeros(batch_size, 512, height // 8, width // 8, device=x.device),
            torch.zeros(batch_size, 512, height // 16, width // 16, device=x.device),
        ]


def test_gated_weight_generator_handles_tiny_channels() -> None:
    gate = GatedWeightGenerator(in_channels=1, num_experts=4)
    weights = gate(torch.randn(2, 1, 4, 4))
    assert weights.shape == (2, 4, 1, 1, 1)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1, 1, 1), atol=1e-6)


def test_ca_layer_handles_reduction_larger_than_channels() -> None:
    ca = CALayer(channel=4, reduction=16)
    x = torch.randn(2, 4, 8, 8)
    out = ca(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_lfem_uses_each_expert_gate_weight() -> None:
    lfem = LFEM(4)
    lfem.CBSk1 = nn.Identity()
    lfem.CBSk3 = ConstantBranch(1.0)
    lfem.dconv = ConstantBranch(2.0)
    lfem.dfconv = ConstantBranch(3.0)
    lfem.bn = nn.Identity()
    lfem.silu = nn.Identity()
    lfem.dwconv = ConstantBranch(4.0)
    lfem.gate = FixedGate([0.1, 0.2, 0.3, 0.4])
    lfem.ca = nn.Identity()
    lfem.last = nn.Identity()

    x = torch.zeros(2, 4, 3, 3)
    out = lfem(x)
    assert torch.allclose(out, torch.full_like(x, 3.0), atol=1e-6)


def test_lfem_forward_shape_and_backward() -> None:
    lfem = LFEM(8)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    out = lfem(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert x.grad is not None


@pytest.fixture
def feat_c3() -> torch.Tensor:
    return torch.randn(2, 16, 32, 32)


@pytest.fixture
def feat_c4() -> torch.Tensor:
    return torch.randn(2, 32, 16, 16)


@pytest.fixture
def feat_c5() -> torch.Tensor:
    return torch.randn(2, 64, 8, 8)


def _small_neck(**kwargs) -> LFEMMultiScaleNeck:
    return LFEMMultiScaleNeck(
        C3_size=16,
        C4_size=32,
        C5_size=64,
        feature_size=32,
        **kwargs,
    )


def test_lfem_neck_output_shape(feat_c3, feat_c4, feat_c5) -> None:
    neck = _small_neck()
    out = neck([feat_c3, feat_c4, feat_c5])
    assert out.shape == (2, 32, 16, 16)
    assert torch.isfinite(out).all()


def test_lfem_neck_return_intermediates(feat_c3, feat_c4, feat_c5) -> None:
    neck = _small_neck()
    out, (p3, p4, p5) = neck(
        [feat_c3, feat_c4, feat_c5], return_intermediates=True
    )
    assert out.shape == (2, 32, 16, 16)
    assert p3.shape == (2, 32, 32, 32)
    assert p4.shape == (2, 32, 16, 16)
    assert p5.shape == (2, 32, 8, 8)


def test_lfem_neck_eval_resolution() -> None:
    neck = _small_neck(upsample_mode="bilinear")
    neck.eval()
    c3 = torch.randn(1, 16, 48, 64)
    c4 = torch.randn(1, 32, 24, 32)
    c5 = torch.randn(1, 64, 12, 16)
    with torch.no_grad():
        out = neck([c3, c4, c5])
    assert out.shape == (1, 32, 24, 32)
    assert torch.isfinite(out).all()


def test_lfem_neck_handles_zero_input() -> None:
    neck = _small_neck()
    neck.eval()
    inputs = [
        torch.zeros(1, 16, 32, 32),
        torch.zeros(1, 32, 16, 16),
        torch.zeros(1, 64, 8, 8),
    ]
    with torch.no_grad():
        out = neck(inputs)
    assert out.shape == (1, 32, 16, 16)
    assert torch.isfinite(out).all()


def test_lfem_neck_backward_pass(feat_c3, feat_c4, feat_c5) -> None:
    neck = _small_neck(use_spd_downsample=False)
    feat_c3.requires_grad_(True)
    out = neck([feat_c3, feat_c4, feat_c5])
    out.mean().backward()
    assert feat_c3.grad is not None


def test_lfem_neck_normalized_fusion_weights_are_finite() -> None:
    neck = _small_neck()
    fusion = dict(neck.named_modules())["fusion"]
    weights = cast(torch.Tensor, getattr(fusion, "normalized_weights"))
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-4)


def test_dsgcnet_with_lfem_neck() -> None:
    from crowdcount.models.dsgcnet import DSGCnet

    model = DSGCnet(TinyVGGBackbone(), row=2, line=2, use_lfem_neck=True)
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape == (1, 1, 16, 16)


def test_dsgcnet_lfem_neck_custom_config() -> None:
    from crowdcount.models.dsgcnet import DSGCnet

    cfg = OmegaConf.create(
        {
            "feature_size": 256,
            "use_spd_downsample": False,
            "fusion_eps": 1e-3,
            "upsample_mode": "bilinear",
        }
    )
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_lfem_neck=True,
        lfem_neck_cfg=cfg,
    )
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["density_out"].shape == (1, 1, 16, 16)


def test_dsgcnet_lfem_neck_with_neck_moe_pyramid_context() -> None:
    from crowdcount.models.dsgcnet import DSGCnet

    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        use_lfem_neck=True,
        use_neck_moe=True,
    )
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["density_out"].shape == (1, 1, 16, 16)


def test_dsgcnet_lfem_neck_mutual_exclusion() -> None:
    from crowdcount.models.dsgcnet import DSGCnet

    with pytest.raises(ValueError, match="mutually exclusive"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            use_lfem_neck=True,
            use_bifpn_neck=True,
        )


def test_dsgcnet_lfem_neck_rejects_legacy_msaa() -> None:
    from crowdcount.models.dsgcnet import DSGCnet

    with pytest.raises(ValueError, match="legacy MSAA"):
        DSGCnet(
            TinyVGGBackbone(),
            row=2,
            line=2,
            use_lfem_neck=True,
            use_msaa=True,
            msaa_variant="legacy",
        )
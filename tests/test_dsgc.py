"""End-to-end forward pass tests for DSGCnet.

Uses a tiny synthetic backbone to avoid downloading pretrained weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.semc_blocks import SEMCEnhancer
from crowdcount.plugins.gm import GateMechanism, SpatialGateMechanism
from crowdcount.plugins.moe import ESCA, MoE, LightMoE
from crowdcount.plugins.msaa import MsaaAdaptiveLayer


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
    assert model.alpha is not None
    assert model.alpha.requires_grad


def test_alpha_absent_in_moe_mode() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"top_k": 2})
    model = DSGCnet(backbone, row=2, line=2, fusion_mode="esca_moe", moe_cfg=cfg)
    assert model.alpha is None


def test_gate_mechanism_initialized_when_enabled() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_gm=True)
    assert model.gm is not None
    # Default gm_spatial=True → SpatialGateMechanism
    assert isinstance(model.gm, SpatialGateMechanism)


def test_legacy_gate_mechanism_when_gm_spatial_false() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_gm=True, gm_spatial=False)
    assert model.gm is not None
    assert isinstance(model.gm, GateMechanism)


def test_adaptive_gcn_forward() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        gcn_adaptive=True,
        gcn_k=4,
        gcn_k_min=2,
        gcn_k_max=6,
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2


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


# ---------------------------------------------------------------------------
# MassAdaptiveLayer (MSAA) tests
# ---------------------------------------------------------------------------


def test_msaa_initialized_when_enabled() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msaa=True)
    assert model.msaa is not None
    assert isinstance(model.msaa, MsaaAdaptiveLayer)


def test_msaa_disabled_when_false() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msaa=False)
    assert model.msaa is None


def test_msaa_pa_channels_when_enabled() -> None:
    """When use_msaa=True, PA-FPN should be initialised with 1280-channel inputs."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msaa=True)
    # P5_1 is the first conv that ingests C5; its in_channels should be 1280
    assert model.pa.P5_1[0].in_channels == 1280
    assert model.pa.P4_1[0].in_channels == 1280
    assert model.pa.P3_1[0].in_channels == 1280


def test_msaa_forward_shapes_match() -> None:
    """With use_msaa=True the output shapes must be identical to the default path."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msaa=True).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape[0] == 2


def test_multiscale_density_outputs_present_and_shaped() -> None:
    """When enabled, multi-scale density outputs should exist and align with c3/c4/c5."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"density_multi_scale": {"enabled": True}})
    model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

    x = torch.zeros(2, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(x)
        out = model(x)

    assert "density_block3" in out
    assert "density_block4" in out
    assert "density_block5" in out

    assert out["density_block3"].shape == (2, 1, feats[1].shape[2], feats[1].shape[3])
    assert out["density_block4"].shape == (2, 1, feats[2].shape[2], feats[2].shape[3])
    assert out["density_block5"].shape == (2, 1, feats[3].shape[2], feats[3].shape[3])


def test_multiscale_density_outputs_absent_when_disabled() -> None:
    """When disabled, model should keep legacy output keys only."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"density_multi_scale": {"enabled": False}})
    model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert "density_block3" not in out
    assert "density_block4" not in out
    assert "density_block5" not in out


def test_moe_initialized_when_enabled() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"top_k": 2})
    model = DSGCnet(backbone, row=2, line=2, fusion_mode="esca_moe", moe_cfg=cfg)
    assert model.esca is not None
    assert model.moe is not None
    assert isinstance(model.esca, ESCA)
    assert isinstance(model.moe, MoE)


def test_moe_forward_shapes_match() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"top_k": 2})
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="esca_moe",
        moe_cfg=cfg,
    ).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape[0] == 2
    assert out["moe_weights"] is not None


def test_moe_outputs_aux_losses_in_train_mode() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"top_k": 2})
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="esca_moe",
        moe_cfg=cfg,
    ).train()

    out = model(torch.zeros(1, 3, 128, 128))
    assert out["moe_aux_losses"] is not None
    assert out["moe_aux_total"] is not None


# ---------------------------------------------------------------------------
# SEMCEnhancer tests
# ---------------------------------------------------------------------------


def test_semc_enhancer_initialized_when_enabled() -> None:
    """SEMCEnhancer should be created when use_semc_enhancer=True."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"use_semc_enhancer": True})
    model = DSGCnet(backbone, row=2, line=2, cfg=cfg)
    assert model.semc_enhancer is not None
    assert isinstance(model.semc_enhancer, SEMCEnhancer)


def test_semc_enhancer_absent_when_disabled() -> None:
    """SEMCEnhancer should be None when use_semc_enhancer is not set."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2)
    assert model.semc_enhancer is None


def test_semc_enhancer_forward_output_shapes() -> None:
    """Enabling SEMCEnhancer must not change any output key or shape."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"use_semc_enhancer": True})
    model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape[0] == 2
    # query count must be consistent with baseline (row*line anchors * spatial cells)
    assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]


def test_semc_enhancer_density_hint_forward() -> None:
    """use_density_hint=True path must remain stable and produce non-negative density."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create(
        {
            "use_semc_enhancer": True,
            "semc": {"use_density_hint": True, "expansion_factor": 2},
        }
    )
    model = DSGCnet(backbone, row=2, line=2, cfg=cfg).eval()

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert out["density_out"].shape[0] == 1
    assert (out["density_out"] >= 0).all()


def test_semc_enhancer_invalid_position_raises() -> None:
    """Only post_gcn is supported in the current implementation."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create(
        {
            "use_semc_enhancer": True,
            "semc": {"position": "pre_gcn"},
        }
    )

    with pytest.raises(ValueError, match="semc.position"):
        DSGCnet(backbone, row=2, line=2, cfg=cfg)


def test_semc_enhancer_with_moe_mode_raises() -> None:
    """SEMCEnhancer should not silently alter the MoE path."""
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create({"top_k": 2, "use_semc_enhancer": True})

    with pytest.raises(ValueError, match="fusion_mode='gcn'"):
        DSGCnet(
            backbone,
            row=2,
            line=2,
            fusion_mode="esca_moe",
            moe_cfg=cfg,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# MoE wiring and checkpoint restore tests
# ---------------------------------------------------------------------------


def test_density_hint_wired_through_to_moe() -> None:
    """density_hint 应传入 MoE，并进入路由网络的输入通道。"""
    backbone = TinyVGGBackbone()
    moe_cfg = OmegaConf.create({"top_k": 2, "use_density_hint": True})
    model = DSGCnet(
        backbone, row=2, line=2, fusion_mode="esca_moe", moe_cfg=moe_cfg
    ).eval()

    x = torch.randn(1, 3, 128, 128)
    captured: dict = {}
    _orig_forward = model.moe.forward  # type: ignore[union-attr]

    def _spy_forward(feat, density_hint=None, training=True):
        captured["density_hint"] = density_hint
        return _orig_forward(feat, density_hint=density_hint, training=training)

    model.moe.forward = _spy_forward  # type: ignore[union-attr]
    with torch.no_grad():
        model(x)

    assert captured.get("density_hint") is not None, (
        "DSGCnet 未将 density map 传递给 MoE.forward()"
    )

    first_conv_inputs: list[torch.Tensor] = []

    def _capture_first_conv_input(module, args):
        first_conv_inputs.append(args[0].detach().clone())

    hook = model.moe.context_encoder.score_net[0].register_forward_pre_hook(  # type: ignore[union-attr]
        _capture_first_conv_input
    )
    with torch.no_grad():
        features = model.backbone(x)
        features_list = [features[0], features[1], features[2], features[3]]
        c3, c4, c5 = features_list[1], features_list[2], features_list[3]
        features_pa = model.pa([c3, c4, c5])
        density = model.density_pred(features_pa)
        esca_feature = model.esca(features_pa)  # type: ignore[union-attr]
        model.moe.context_encoder(esca_feature, density_hint=density)  # type: ignore[union-attr]

    hook.remove()
    assert len(first_conv_inputs) == 1, "未捕获到 context_encoder 的第一层输入"
    routed_input = first_conv_inputs[0]
    assert routed_input.shape[1] == esca_feature.shape[1] + 1
    # GridSoftRouter uses AvgPool (grid_stride) before score_net, so the
    # captured input is at coarse resolution.  Verify that the density
    # channel is present and comes from avg-pooling the original density.
    coarse_density = routed_input[:, -1:]
    grid_stride = model.moe.router.grid_stride  # type: ignore[union-attr]
    expected_density = F.avg_pool2d(
        density, kernel_size=grid_stride, stride=grid_stride
    )
    assert coarse_density.shape == expected_density.shape, (
        f"coarse density shape {coarse_density.shape} != expected {expected_density.shape}"
    )
    assert torch.allclose(coarse_density, expected_density, atol=1e-5), (
        "density_hint 没有作为额外通道送入路由网络"
    )
    model.moe.forward = _orig_forward  # type: ignore[union-attr]


def test_checkpoint_temperature_restore_syncs_router() -> None:
    """模拟 checkpoint 恢复：必须同时将 temperature 写入 moe 和 moe.router。"""
    backbone = TinyVGGBackbone()
    moe_cfg = OmegaConf.create({"top_k": 2})
    model = DSGCnet(backbone, row=2, line=2, fusion_mode="esca_moe", moe_cfg=moe_cfg)

    saved_temp = 0.6
    # Simulate the checkpoint restore pattern used in trainer.py
    model.moe.temperature = saved_temp  # type: ignore[union-attr]
    model.moe.router.temperature = saved_temp  # type: ignore[union-attr]

    assert model.moe.temperature == saved_temp  # type: ignore[union-attr]
    assert model.moe.router.temperature == saved_temp  # type: ignore[union-attr]
    # Subsequent decay must start from the restored value
    model.update_moe_temperature(decay_rate=0.9)
    assert model.moe.temperature < saved_temp  # type: ignore[union-attr]
    assert model.moe.router.temperature == model.moe.temperature  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# GCN + LightMoE (gcn_moe mode) tests
# ---------------------------------------------------------------------------


@pytest.fixture
def gcn_moe_model():
    backbone = TinyVGGBackbone()
    moe_cfg = OmegaConf.create(
        {"grid_stride": 4, "use_density_hint": True, "lambda_balance": 0.01}
    )
    return DSGCnet(backbone, row=2, line=2, fusion_mode="gcn_moe", moe_cfg=moe_cfg)


def test_gcn_moe_init_has_both_gcn_and_light_moe(gcn_moe_model) -> None:
    """gcn_moe mode should have GCN processors AND LightMoE."""
    assert gcn_moe_model.density_gcn is not None
    assert gcn_moe_model.feature_gcn is not None
    assert gcn_moe_model.alpha is not None
    assert gcn_moe_model.light_moe is not None
    assert isinstance(gcn_moe_model.light_moe, LightMoE)
    # esca_moe-specific modules should be absent
    assert gcn_moe_model.esca is None
    assert gcn_moe_model.moe is None


def test_gcn_moe_forward_output_keys(gcn_moe_model) -> None:
    gcn_moe_model.eval()
    with torch.no_grad():
        out = gcn_moe_model(torch.zeros(1, 3, 128, 128))
    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["moe_weights"] is not None
    assert out["moe_aux_losses"] is not None


def test_gcn_moe_forward_shapes(gcn_moe_model) -> None:
    gcn_moe_model.eval()
    with torch.no_grad():
        out = gcn_moe_model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["moe_weights"].shape[0] == 2
    assert out["moe_weights"].shape[1] == 3  # 3 micro-experts


def test_gcn_moe_weights_sum_to_one(gcn_moe_model) -> None:
    gcn_moe_model.eval()
    with torch.no_grad():
        out = gcn_moe_model(torch.zeros(2, 3, 128, 128))
    w = out["moe_weights"]
    sums = w.sum(dim=1)  # [B, H, W]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gcn_moe_supports_moe(gcn_moe_model) -> None:
    assert gcn_moe_model.supports_moe() is True


def test_gcn_moe_gating_parameters_from_light_moe(gcn_moe_model) -> None:
    params = gcn_moe_model.get_moe_gating_parameters()
    assert len(params) > 0
    # All should come from light_moe.router
    router_params = set(id(p) for p in gcn_moe_model.light_moe.router.parameters())
    for p in params:
        assert id(p) in router_params


def test_gcn_moe_training_produces_aux_loss() -> None:
    backbone = TinyVGGBackbone()
    moe_cfg = OmegaConf.create({"grid_stride": 4, "use_density_hint": True})
    model = DSGCnet(backbone, row=2, line=2, fusion_mode="gcn_moe", moe_cfg=moe_cfg)
    model.train()
    out = model(torch.randn(2, 3, 128, 128))
    assert out["moe_aux_total"] is not None
    assert out["moe_aux_total"].requires_grad


def test_gcn_moe_light_moe_param_count_small(gcn_moe_model) -> None:
    """LightMoE should add < 0.5M parameters."""
    light_params = sum(p.numel() for p in gcn_moe_model.light_moe.parameters())
    assert light_params < 500_000, (
        f"LightMoE has {light_params} params, expected < 500k"
    )


def test_gcn_moe_with_gm() -> None:
    """gcn_moe should work together with gate mechanism."""
    backbone = TinyVGGBackbone()
    moe_cfg = OmegaConf.create({"grid_stride": 4})
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="gcn_moe",
        use_gm=True,
        moe_cfg=moe_cfg,
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))
    assert out["pred_logits"] is not None
    assert out["moe_weights"] is not None

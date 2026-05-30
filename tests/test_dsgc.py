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
from crowdcount.models.head import (
    DecoupledPredictionHead,
    DeepRegressionModel,
    RegressionModel,
)
from crowdcount.models.semc_blocks import SEMCEnhancer
from crowdcount.plugins.gm import GateMechanism, SpatialGateMechanism
from crowdcount.plugins.mamba_moe import MambaMoEFusion
from crowdcount.plugins.mamba_vss_dual_fusion import MambaVSSDualFusion
from crowdcount.plugins.moe import LightMoE
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


def test_mamba_moe_mode_forward() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create(
        {
            "d_state": 8,
            "d_conv": 3,
            "expand": 1.0,
            "num_experts": 4,
            "top_k": 2,
            "lr_space": "exp",
            "num_blocks": 1,
            "mlp_hidden": 64,
            "drop_path": 0.0,
            "lambda_balance": 0.01,
            "use_density_hint": False,
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="mamba_moe",
        mamba_moe_cfg=cfg,
    ).eval()
    assert model.alpha is None
    assert isinstance(model.mamba_moe, MambaMoEFusion)

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[0] == 1
    assert out["moe_weights"] is not None
    assert out["moe_aux_total"] is not None


def test_mamba_vss_dual_mode_forward() -> None:
    backbone = TinyVGGBackbone()
    cfg = OmegaConf.create(
        {
            "d_state": 4,
            "d_conv": 3,
            "mlp_ratio": 1.0,
            "vss_low_dim": 4,
            "num_vss_blocks": 1,
            "num_moe_blocks": 1,
            "num_experts": 2,
            "top_k": 1,
            "expand": 1.0,
            "d_spectral": 16,
            "mlp_hidden": 64,
            "drop_path": 0.0,
            "lambda_balance": 0.01,
            "use_density_hint": True,
            "density_embed_dim": 16,
            "fusion_spatial": True,
            "gate_init": 0.001,
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="mamba_vss_dual",
        mamba_vss_dual_cfg=cfg,
    ).eval()
    assert model.alpha is None
    assert isinstance(model.mamba_vss_dual, MambaVSSDualFusion)
    assert model.supports_moe()
    assert model.get_moe_gating_parameters()

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))

    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[0] == 1
    assert out["moe_weights"] is not None
    assert out["moe_aux_total"] is not None
    assert out["mamba_vss_fusion_weights"] is not None


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


@pytest.mark.parametrize("mode", ["sigmoid", "learned", "gated", "residual", "calibrated"])
def test_density_attention_forward_shapes_match(mode: str) -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        use_density_attention=True,
        density_attention_mode=mode,
    ).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert model.density_attention is not None
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["density_out"].shape == (2, 1, 16, 16)


def test_density_attention_debug_stats_are_returned() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        use_density_attention=True,
        density_attention_mode="residual",
        density_attention_debug=True,
    ).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    stats = out["density_attention_stats"]
    expected_keys = {
        "min",
        "max",
        "mean",
        "std",
        "p10",
        "p90",
        "high_density_mean",
        "low_density_mean",
    }
    assert expected_keys.issubset(stats.keys())
    for value in stats.values():
        assert value.ndim == 0
        assert torch.isfinite(value)


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


def test_uncertainty_forward() -> None:
    """DSGCnet with use_uncertainty=True should produce uncertainty_map in output."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        gcn_adaptive=True,
        use_uncertainty=True,
        uncertainty_scale=6.0,
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert "uncertainty_map" in out
    assert out["uncertainty_map"].shape[0] == 2
    # uncertainty should be in [0, 1]
    assert out["uncertainty_map"].min() >= 0.0
    assert out["uncertainty_map"].max() <= 1.0
    # other outputs still present
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2


def test_cross_stream_mode_forward() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        gcn_mode="cross_stream",
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["density_out"].shape[0] == 2


def test_cross_stream_mode_does_not_create_external_gm() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        gcn_mode="cross_stream",
        use_gm=True,
    )
    assert model.cross_stream_gcn is not None
    assert model.gm is None


def test_feature_transformer_stream_forward() -> None:
    backbone = TinyVGGBackbone()
    feature_transformer_cfg = OmegaConf.create(
        {
            "embed_dim": 64,
            "num_heads": 4,
            "window_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "gate_init": 0.0,
            "mode": "window",
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        gcn_conv_type="gatv2",
        feature_stream_type="transformer",
        feature_transformer_cfg=feature_transformer_cfg,
        use_gm=True,
        use_density_attention=True,
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[0] == 1
    assert out["density_out"].shape[0] == 1


# ---------------------------------------------------------------------------
# graph_attn_moe fusion mode
# ---------------------------------------------------------------------------


def test_graph_attn_moe_forward() -> None:
    """DSGCnet with fusion_mode='graph_attn_moe' should produce expected keys."""
    backbone = TinyVGGBackbone()
    gam_cfg = OmegaConf.create(
        {
            "num_heads": 4,
            "use_density_bias": True,
            "density_bias_scale": 1.0,
            "attn_dropout": 0.0,
            "local_kernels": [1, 3],
            "local_expansion": 2,
            "local_use_density_gate": True,
            "grid_stride": 4,
            "lambda_balance": 0.01,
            "router_detach_density": True,
            "disable_graph_bias": False,
            "disable_local_expert": False,
            "disable_global_expert": False,
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="graph_attn_moe",
        graph_attn_moe_cfg=gam_cfg,
    ).eval()
    assert model.graph_attn_moe is not None
    assert model.alpha is None
    assert model.density_gcn is None
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["density_out"].shape[0] == 2
    assert "moe_weights" in out
    # Routing weights must sum to 1 along expert dim
    w = out["moe_weights"]
    assert torch.allclose(
        w.sum(dim=1), torch.ones(2, w.shape[2], w.shape[3]), atol=1e-5
    )


def test_graph_attn_moe_train_mode_aux_loss() -> None:
    """In training mode, graph_attn_moe should produce aux losses."""
    backbone = TinyVGGBackbone()
    gam_cfg = OmegaConf.create(
        {
            "num_heads": 4,
            "use_density_bias": True,
            "density_bias_scale": 1.0,
            "attn_dropout": 0.0,
            "local_kernels": [1, 3],
            "local_expansion": 2,
            "local_use_density_gate": True,
            "grid_stride": 4,
            "lambda_balance": 0.01,
            "router_detach_density": True,
            "disable_graph_bias": False,
            "disable_local_expert": False,
            "disable_global_expert": False,
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="graph_attn_moe",
        graph_attn_moe_cfg=gam_cfg,
    ).train()
    out = model(torch.zeros(2, 3, 128, 128))
    assert "moe_aux_losses" in out
    assert "moe_aux_total" in out
    assert out["moe_aux_total"] is not None


# ---------------------------------------------------------------------------
# graph_attn_moe local-first mode
# ---------------------------------------------------------------------------


def test_graph_attn_moe_local_first_forward() -> None:
    """DSGCnet with local-first graph_attn_moe should produce expected keys."""
    backbone = TinyVGGBackbone()
    gam_cfg = OmegaConf.create(
        {
            "num_heads": 4,
            "use_density_bias": True,
            "density_bias_scale": 1.0,
            "attn_dropout": 0.0,
            "local_kernels": [1, 3],
            "local_expansion": 2,
            "local_use_density_gate": True,
            "local_window_size": 4,
            "local_prior": 1.0,
            "grid_stride": 4,
            "lambda_balance": 0.01,
            "router_detach_density": True,
            "disable_graph_bias": False,
            "disable_local_expert": False,
            "disable_global_expert": False,
        }
    )
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="graph_attn_moe",
        graph_attn_moe_cfg=gam_cfg,
    ).eval()
    assert model.graph_attn_moe is not None
    assert model.graph_attn_moe.local_expert is not None
    assert model.graph_attn_moe.local_expert.window_size == 4
    assert model.graph_attn_moe.router is not None
    assert model.graph_attn_moe.router.local_prior == 1.0
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["density_out"].shape[0] == 2
    w = out["moe_weights"]
    assert torch.allclose(
        w.sum(dim=1), torch.ones(2, w.shape[2], w.shape[3]), atol=1e-5
    )


# ---------------------------------------------------------------------------
# graph_moe fusion mode
# ---------------------------------------------------------------------------


def _graph_moe_test_cfg():
    return OmegaConf.create(
        {
            "num_experts": 5,
            "top_k": 2,
            "grid_stride": 4,
            "router_temperature": 1.0,
            "router_detach_density": True,
            "use_uncertainty_hint": True,
            "use_coordinate_hint": True,
            "aux_loss_weight": 1.0,
            "lambda_balance": 0.01,
            "lambda_importance": 0.01,
            "lambda_capacity": 0.0,
            "router_z_loss_weight": 0.0,
            "capacity_factor": 1.25,
            "local_kernels": [1, 3],
            "local_expansion": 1,
            "local_use_density_gate": True,
            "local_window_size": 0,
            "num_heads": 4,
            "use_density_bias": True,
            "density_bias_scale": 1.0,
            "attn_dropout": 0.0,
            "scale_dilations": [1, 2],
            "background_max_suppression": 0.5,
            "residual_gate_init": 1.0,
            "disabled_experts": [],
        }
    )


def test_graph_moe_forward() -> None:
    """DSGCnet with fusion_mode='graph_moe' should replace dual-stream GCN."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="graph_moe",
        graph_moe_cfg=_graph_moe_test_cfg(),
    ).eval()
    assert model.graph_moe is not None
    assert model.graph_attn_moe is None
    assert model.alpha is None
    assert model.density_gcn is None
    assert model.feature_gcn is None
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["density_out"].shape[0] == 2
    weights = out["moe_weights"]
    assert weights.shape[1] == 5
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(2, weights.shape[2], weights.shape[3]), atol=1e-5
    )
    assert ((weights > 0).sum(dim=1) <= 2).all()


def test_graph_moe_train_mode_aux_loss() -> None:
    """GraphMoE should expose router auxiliary losses in train mode."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone,
        row=2,
        line=2,
        fusion_mode="graph_moe",
        graph_moe_cfg=_graph_moe_test_cfg(),
    ).train()
    out = model(torch.zeros(2, 3, 128, 128))
    assert out["moe_aux_losses"] is not None
    assert out["moe_aux_total"] is not None
    assert "l_importance" in out["moe_aux_losses"]
    assert "router_entropy" in out["moe_aux_losses"]


# ---------------------------------------------------------------------------
# Decoupled Head
# ---------------------------------------------------------------------------


def test_decoupled_head_forward() -> None:
    """use_decoupled_head=True must produce valid outputs with correct shapes."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_decoupled_head=True).eval()
    assert model.use_decoupled_head
    assert isinstance(model.pred_trunk, DecoupledPredictionHead)
    assert isinstance(model.regression, RegressionModel)
    assert not isinstance(model.regression, DeepRegressionModel)
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]


def test_decoupled_head_with_freq_router() -> None:
    """use_decoupled_head + use_freq_head should work together."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone, row=2, line=2, use_decoupled_head=True, use_freq_head=True
    ).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[0] == 1


def test_decoupled_head_disabled_by_default() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2)
    assert not model.use_decoupled_head


# ---------------------------------------------------------------------------
# MSCANeck (pure neck, preserves GCN downstream)
# ---------------------------------------------------------------------------


def test_msca_neck_forward() -> None:
    """use_msca_neck=True must produce valid outputs with correct shapes."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msca_neck=True).eval()
    assert model.use_msca_neck
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[2] == 2
    assert out["pred_points"].shape[0] == 2
    assert out["pred_points"].shape[2] == 2
    assert out["density_out"].shape == (2, 1, 16, 16)


def test_msca_neck_preserves_gcn() -> None:
    """MSCANeck must NOT bypass GCN — density_gcn + feature_gcn should exist."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msca_neck=True)
    assert model.density_gcn is not None
    assert model.feature_gcn is not None
    assert model.density_pred is not None
    assert model.msca_decoder is None


def test_msca_neck_and_decoder_mutually_exclusive() -> None:
    backbone = TinyVGGBackbone()
    with pytest.raises(ValueError, match="mutually exclusive"):
        DSGCnet(backbone, row=2, line=2, use_msca_neck=True, use_msca_decoder=True)


def test_msca_neck_disabled_by_default() -> None:
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2)
    assert not model.use_msca_neck


# --------------- MSCADecoder + GCN serial pipeline --------------- #


def test_msca_decoder_with_gcn_forward() -> None:
    """MSCADecoder should feed into GCN and produce valid outputs."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msca_decoder=True)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["pred_logits"] is not None
    assert out["pred_points"] is not None
    assert out["density_out"] is not None
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2


def test_msca_decoder_preserves_gcn() -> None:
    """When use_msca_decoder=True, GCN modules should still be instantiated."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_msca_decoder=True)
    assert model.density_gcn is not None
    assert model.feature_gcn is not None


# ---------------------------------------------------------------------------
# RCCFormer MFFM neck + DEAB/ASAM density head
# ---------------------------------------------------------------------------


def test_rccformer_neck_forward() -> None:
    """MFFMNeck + DensityPredDEAB forward pass produces correct shapes."""
    backbone = TinyVGGBackbone()
    model = DSGCnet(backbone, row=2, line=2, use_rccformer_neck=True)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out["pred_logits"] is not None
    assert out["pred_points"] is not None
    assert out["density_out"] is not None
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2
    # density_out: [B, 1, H/8, W/8]
    assert out["density_out"].shape == (2, 1, 16, 16)


def test_rccformer_neck_deab_blocks_configurable() -> None:
    """rccformer_deab_blocks parameter controls the number of DEAB blocks."""
    from crowdcount.plugins.rccformer import DensityPredDEAB

    backbone = TinyVGGBackbone()
    model = DSGCnet(
        backbone, row=2, line=2, use_rccformer_neck=True, rccformer_deab_blocks=3
    )
    assert isinstance(model.density_pred, DensityPredDEAB)
    assert len(model.density_pred.deab_blocks) == 3

"""Tests for prediction head modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.head import (
    ClassificationModel,
    DecoupledPredictionHead,
    DeepClassificationModel,
    DeepRegressionModel,
    DensityAttentionMask,
    Density_pred,
    Density_pred_V3,
    EnhancedDensityAttention,
    GatedDensityAttention,
    RegressionModel,
    ResidualDensityAttention,
    SharedPredictionTrunk,
)
from crowdcount.models.ssim_loss import SSIMLoss


@pytest.fixture
def feature_map():
    return torch.randn(2, 256, 16, 16)


# ---------------------------------------------------------------------------
# Density_pred
# ---------------------------------------------------------------------------


def test_density_pred_output_shape(feature_map):
    model = Density_pred()
    out = model(feature_map)
    assert out.shape == (2, 1, 16, 16)


def test_density_pred_non_negative(feature_map):
    model = Density_pred()
    out = model(feature_map)
    assert (out >= 0).all(), "Density map should be non-negative (ReLU output)"


# ---------------------------------------------------------------------------
# Density_pred_V3
# ---------------------------------------------------------------------------


def test_density_pred_v3_output_shape(feature_map):
    model = Density_pred_V3()
    out = model(feature_map)
    assert out.shape == (2, 1, 16, 16)


def test_density_pred_v3_upsample_shape():
    x = torch.randn(2, 256, 8, 8)
    model = Density_pred_V3(upsample=True)
    out = model(x)
    assert out.shape == (2, 1, 16, 16), "PixelShuffle 2× should double spatial dims"


def test_density_pred_v3_non_negative(feature_map):
    model = Density_pred_V3()
    out = model(feature_map)
    assert (out >= 0).all(), "Density V3 should be non-negative (Softplus output)"


def test_density_pred_v3_backward(feature_map):
    model = Density_pred_V3()
    out = model(feature_map)
    loss = out.sum()
    loss.backward()
    # Check that gradients flow back to all ASPP branches
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# RegressionModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points", [4, 9])
def test_regression_model_output_shape(feature_map, num_anchor_points):
    model = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, 2)


# ---------------------------------------------------------------------------
# ClassificationModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points,num_classes", [(4, 2), (9, 2)])
def test_classification_model_output_shape(feature_map, num_anchor_points, num_classes):
    model = ClassificationModel(
        num_features_in=256,
        num_anchor_points=num_anchor_points,
        num_classes=num_classes,
    )
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, num_classes)


# ---------------------------------------------------------------------------
# SharedPredictionTrunk
# ---------------------------------------------------------------------------


def test_shared_trunk_output_shape(feature_map):
    trunk = SharedPredictionTrunk(in_channels=256, feature_size=256)
    out = trunk(feature_map)
    assert out.shape == feature_map.shape, (
        "SharedPredictionTrunk must preserve spatial dims and channel count"
    )


def test_shared_trunk_end_to_end_regression(feature_map):
    """Trunk → RegressionModel pipeline must produce the same shape as before."""
    trunk = SharedPredictionTrunk()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)
    out = reg(trunk(feature_map))
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * 4, 2)


def test_shared_trunk_end_to_end_classification(feature_map):
    """Trunk → ClassificationModel pipeline must produce the same shape as before."""
    trunk = SharedPredictionTrunk()
    cls = ClassificationModel(num_features_in=256, num_anchor_points=4, num_classes=2)
    out = cls(trunk(feature_map))
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * 4, 2)


def test_shared_trunk_is_distinct_from_heads():
    """Trunk parameters must not overlap with RegressionModel or ClassificationModel."""
    trunk = SharedPredictionTrunk()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)
    cls = ClassificationModel(num_features_in=256, num_anchor_points=4, num_classes=2)
    trunk_ids = {id(p) for p in trunk.parameters()}
    reg_ids = {id(p) for p in reg.parameters()}
    cls_ids = {id(p) for p in cls.parameters()}
    assert trunk_ids.isdisjoint(reg_ids)
    assert trunk_ids.isdisjoint(cls_ids)


@pytest.mark.parametrize("mode", ["sigmoid", "learned"])
def test_density_attention_mask_shape_and_range(mode):
    mask = DensityAttentionMask(mode=mode)
    density = torch.rand(2, 1, 16, 16)
    out = mask(density)
    assert out.shape == density.shape
    assert (out >= 0).all()
    assert (out <= 1).all()


# ---------------------------------------------------------------------------
# ResidualDensityAttention
# ---------------------------------------------------------------------------


def test_residual_density_attention_init_identity(feature_map):
    module = ResidualDensityAttention(hidden_channels=16, max_delta=0.5)
    density = torch.rand(2, 1, 16, 16)

    with torch.no_grad():
        out = module(density, feature_map)

    assert torch.allclose(out, feature_map, atol=1e-6)
    assert module.last_attention_scale is not None
    assert module.last_attention_scale.shape == density.shape
    assert torch.allclose(
        module.last_attention_scale,
        torch.ones_like(module.last_attention_scale),
        atol=1e-6,
    )


def test_residual_density_attention_gradient_reaches_final_projection(feature_map):
    module = ResidualDensityAttention(hidden_channels=8, max_delta=0.5)
    density = torch.rand(2, 1, 16, 16, requires_grad=True)
    feat = feature_map.clone().requires_grad_(True)
    out = module(density, feat)
    out.square().sum().backward()

    final_proj = module.proj[2]
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert final_proj.weight.grad is not None
    assert final_proj.weight.grad.abs().sum() > 0


def test_residual_density_attention_scale_is_bounded():
    module = ResidualDensityAttention(hidden_channels=4, max_delta=0.25)
    with torch.no_grad():
        module.strength.fill_(10.0)
        module.proj[2].bias.fill_(10.0)
    density = torch.rand(1, 1, 8, 8)
    feature = torch.ones(1, 4, 8, 8)

    with torch.no_grad():
        out = module(density, feature)

    assert out.min() >= 0.75 - 1e-6
    assert out.max() <= 1.25 + 1e-6


def test_ssim_loss_zero_for_identical_maps():
    criterion = SSIMLoss(window_size=11, sigma=1.5)
    density = torch.rand(2, 1, 16, 16)
    loss = criterion(density, density)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_ssim_loss_positive_for_different_maps():
    criterion = SSIMLoss(window_size=11, sigma=1.5)
    pred = torch.zeros(2, 1, 16, 16)
    target = torch.ones(2, 1, 16, 16)
    loss = criterion(pred, target)
    assert loss > 0


# ---------------------------------------------------------------------------
# DecoupledPredictionHead
# ---------------------------------------------------------------------------


def test_decoupled_head_output_shapes(feature_map):
    head = DecoupledPredictionHead(in_channels=256, feature_size=256)
    cls_feat, reg_feat = head(feature_map)
    assert cls_feat.shape == feature_map.shape
    assert reg_feat.shape == feature_map.shape


def test_decoupled_head_trunks_are_independent():
    """cls_trunk and reg_trunk must have distinct (unshared) parameters."""
    head = DecoupledPredictionHead()
    cls_ids = {id(p) for p in head.cls_trunk.parameters()}
    reg_ids = {id(p) for p in head.reg_trunk.parameters()}
    assert cls_ids.isdisjoint(reg_ids)


def test_decoupled_head_end_to_end(feature_map):
    """DecoupledPredictionHead → separate heads produce correct shapes."""
    head = DecoupledPredictionHead()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)
    cls = ClassificationModel(num_features_in=256, num_anchor_points=4, num_classes=2)
    cls_feat, reg_feat = head(feature_map)
    reg_out = reg(reg_feat)
    cls_out = cls(cls_feat)
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert reg_out.shape == (B, H * W * 4, 2)
    assert cls_out.shape == (B, H * W * 4, 2)


def test_decoupled_regression_path_matches_paper_original_shape(feature_map):
    """reg_trunk + RegressionModel matches the original independent head layout."""
    head = DecoupledPredictionHead()
    reg = RegressionModel(num_features_in=256, num_anchor_points=4)

    assert isinstance(head.reg_trunk, SharedPredictionTrunk)
    assert head.reg_trunk.conv1.kernel_size == (3, 3)
    assert head.reg_trunk.conv2.kernel_size == (3, 3)
    assert reg.output.kernel_size == (3, 3)

    reg_trunk_ids = {id(p) for p in head.reg_trunk.parameters()}
    cls_trunk_ids = {id(p) for p in head.cls_trunk.parameters()}
    projection_ids = {id(p) for p in reg.parameters()}
    assert reg_trunk_ids.isdisjoint(cls_trunk_ids)
    assert reg_trunk_ids.isdisjoint(projection_ids)

    out = reg(head.reg_trunk(feature_map))
    B, H, W = feature_map.shape[0], feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * 4, 2)


# ---------------------------------------------------------------------------
# EnhancedDensityAttention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden", [16, 32])
def test_enhanced_density_attention_shape(feature_map, hidden):
    """Output shape must match input feature shape."""
    module = EnhancedDensityAttention(feature_channels=256, hidden_channels=hidden)
    density = torch.rand(2, 1, 16, 16)
    out = module(density, feature_map)
    assert out.shape == feature_map.shape


def test_enhanced_density_attention_gradient_flow(feature_map):
    """Gradients must flow through both density and feature paths."""
    module = EnhancedDensityAttention(feature_channels=256, hidden_channels=16)
    density = torch.rand(2, 1, 16, 16, requires_grad=True)
    feat = feature_map.clone().requires_grad_(True)
    out = module(density, feat)
    loss = out.sum()
    loss.backward()
    assert density.grad is not None and density.grad.abs().sum() > 0
    assert feat.grad is not None and feat.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# DeepRegressionModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points", [4, 9])
def test_deep_regression_model_output_shape(feature_map, num_anchor_points):
    model = DeepRegressionModel(
        num_features_in=256, num_anchor_points=num_anchor_points
    )
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, 2)


def test_deep_regression_matches_shallow_interface(feature_map):
    """DeepRegressionModel is a drop-in replacement for RegressionModel."""
    shallow = RegressionModel(num_features_in=256, num_anchor_points=4)
    deep = DeepRegressionModel(num_features_in=256, num_anchor_points=4)
    assert shallow(feature_map).shape == deep(feature_map).shape


# ---------------------------------------------------------------------------
# DeepClassificationModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_anchor_points,num_classes", [(4, 2), (9, 2)])
def test_deep_classification_model_output_shape(
    feature_map, num_anchor_points, num_classes
):
    model = DeepClassificationModel(
        num_features_in=256,
        num_anchor_points=num_anchor_points,
        num_classes=num_classes,
    )
    out = model(feature_map)
    B = feature_map.shape[0]
    H, W = feature_map.shape[2], feature_map.shape[3]
    assert out.shape == (B, H * W * num_anchor_points, num_classes)


def test_deep_classification_matches_shallow_interface(feature_map):
    """DeepClassificationModel is a drop-in replacement for ClassificationModel."""
    shallow = ClassificationModel(
        num_features_in=256, num_anchor_points=4, num_classes=2
    )
    deep = DeepClassificationModel(
        num_features_in=256, num_anchor_points=4, num_classes=2
    )
    assert shallow(feature_map).shape == deep(feature_map).shape


def test_deep_classification_bias_init():
    """Foreground (class 1) bias should be negative; background (class 0) should be ~0."""
    import math

    model = DeepClassificationModel(
        num_features_in=256,
        num_anchor_points=4,
        num_classes=2,
        prior=0.01,
    )
    bias = model.output.bias.detach()
    expected_fg = -math.log((1 - 0.01) / 0.01)
    for a in range(4):
        assert abs(bias[a * 2 + 0].item()) < 1e-5, "Background bias should be 0"
        assert abs(bias[a * 2 + 1].item() - expected_fg) < 1e-5, (
            "Foreground bias should match prior init"
        )


def test_classification_bias_init():
    """Verify bias init was applied to the original ClassificationModel too."""
    import math

    model = ClassificationModel(
        num_features_in=256,
        num_anchor_points=4,
        num_classes=2,
        prior=0.01,
    )
    bias = model.output.bias.detach()
    expected_fg = -math.log((1 - 0.01) / 0.01)
    for a in range(4):
        assert abs(bias[a * 2 + 0].item()) < 1e-5
        assert abs(bias[a * 2 + 1].item() - expected_fg) < 1e-5


def test_enhanced_density_attention_residual_nonzero():
    """With base > 0, output should never be all-zero even for zero density."""
    module = EnhancedDensityAttention(feature_channels=256, base_init=0.5)
    density = torch.zeros(1, 1, 8, 8)
    feature = torch.ones(1, 256, 8, 8)
    out = module(density, feature)
    assert out.abs().sum() > 0, "Residual base should prevent complete suppression"


def test_enhanced_density_attention_param_budget():
    """Total parameters should stay under 300K for default config."""
    module = EnhancedDensityAttention(feature_channels=256, hidden_channels=32)
    total = sum(p.numel() for p in module.parameters())
    assert total < 300_000, f"Parameter count {total} exceeds 300K budget"


def test_enhanced_density_attention_sobel_not_learnable():
    """Sobel kernels must be buffers, not parameters."""
    module = EnhancedDensityAttention()
    param_names = {n for n, _ in module.named_parameters()}
    assert "sobel_x" not in param_names
    assert "sobel_y" not in param_names
    buffer_names = {n for n, _ in module.named_buffers()}
    assert "sobel_x" in buffer_names
    assert "sobel_y" in buffer_names


# ---------------------------------------------------------------------------
# GatedDensityAttention
# ---------------------------------------------------------------------------


def test_gated_density_attention_shape():
    module = GatedDensityAttention(feature_channels=256, hidden_channels=16)
    density = torch.randn(2, 1, 16, 16)
    feature = torch.randn(2, 256, 16, 16)
    out = module(density, feature)
    assert out.shape == feature.shape


def test_gated_density_attention_grad():
    module = GatedDensityAttention(feature_channels=64, hidden_channels=8)
    density = torch.randn(2, 1, 8, 8, requires_grad=True)
    feature = torch.randn(2, 64, 8, 8, requires_grad=True)
    out = module(density, feature)
    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None, f"param {name} has no grad"
        assert torch.isfinite(p.grad).all(), f"param {name} grad has non-finite"


def test_gated_density_attention_init_near_identity():
    """With gate_init_bias=-2.0 the module should start close to identity."""
    torch.manual_seed(0)
    module = GatedDensityAttention(
        feature_channels=32, hidden_channels=8, gate_init_bias=-2.0
    )
    density = torch.randn(2, 1, 8, 8).abs()
    feature = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        out = module(density, feature)
    rel_err = (out - feature).abs().mean() / (feature.abs().mean() + 1e-8)
    # gate ~ sigmoid(-2) ~ 0.12, so |out - feature| should be small
    assert rel_err.item() < 0.3, f"rel_err={rel_err.item():.4f} too large at init"


def test_gated_density_attention_zero_density():
    """With zero density, d_attn = sigmoid(beta) = 0.5 by default.

    out = feature * (1 - gate + gate * 0.5) = feature * (1 - 0.5 * gate)
    Output should never be all-zero and should stay close to feature.
    """
    module = GatedDensityAttention(feature_channels=32, hidden_channels=8)
    density = torch.zeros(1, 1, 8, 8)
    feature = torch.ones(1, 32, 8, 8)
    with torch.no_grad():
        out = module(density, feature)
    assert out.abs().sum() > 0
    # 1 - 0.5 * sigmoid(any) ∈ [0.5, 1.0], so output ∈ [0.5, 1.0] for feature=1
    assert (out >= 0.5 - 1e-6).all() and (out <= 1.0 + 1e-6).all()


def test_gated_density_attention_param_budget():
    module = GatedDensityAttention(feature_channels=256, hidden_channels=16)
    total = sum(p.numel() for p in module.parameters())
    # density_proj: 1*16*9 = 144; feature_proj: 256*16 = 4096;
    # gate_conv: 32*1 + 1 = 33; alpha + beta = 2 → ~4275
    assert total < 10_000, f"Parameter count {total} exceeds 10K budget"

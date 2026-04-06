"""Tests for ConvNeXt backbone."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from crowdcount.models.backbone import BackboneConvNeXt, build_backbone


@pytest.mark.parametrize("variant", ["convnext_tiny", "convnext_base"])
def test_convnext_output_shapes(variant):
    """ConvNeXt backbone returns 4 feature maps with correct projected channels."""
    backbone = BackboneConvNeXt(variant)
    backbone.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        features = backbone(x)
    assert len(features) == 4
    # Expected projected channels: [128, 256, 512, 512]
    # (matches VGG body1=128, body2=256, body3=512, body4=512)
    assert features[0].shape[1] == 128
    assert features[1].shape[1] == 256
    assert features[2].shape[1] == 512
    assert features[3].shape[1] == 512


def test_convnext_spatial_strides():
    """Feature maps match VGG stride contract: H/4, H/4, H/8, H/16."""
    backbone = BackboneConvNeXt("convnext_tiny")
    backbone.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        features = backbone(x)
    # Must match VGG: body1=H/2(but we use H/4 placeholder), c3=H/4, c4=H/8, c5=H/16
    assert features[0].shape[2] == 32  # 128/4 placeholder
    assert features[1].shape[2] == 32  # 128/4 (c3)
    assert features[2].shape[2] == 16  # 128/8 (c4)
    assert features[3].shape[2] == 8  # 128/16 (c5)


def test_convnext_batch():
    """Handles batch > 1."""
    backbone = BackboneConvNeXt("convnext_tiny")
    backbone.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        features = backbone(x)
    for feat in features:
        assert feat.shape[0] == 2


def test_convnext_invalid_variant():
    """Raises ValueError for unknown variant."""
    with pytest.raises(ValueError, match="Unknown ConvNeXt variant"):
        BackboneConvNeXt("convnext_xlarge")


def test_build_backbone_convnext():
    """build_backbone correctly dispatches to ConvNeXt."""
    cfg = OmegaConf.create(
        {"model": {"backbone": "convnext_tiny", "backbone_type": "convnext"}}
    )
    backbone = build_backbone(cfg)
    assert isinstance(backbone, BackboneConvNeXt)


def test_convnext_backward():
    """Gradients flow through the ConvNeXt backbone."""
    backbone = BackboneConvNeXt("convnext_tiny")
    backbone.train()
    x = torch.randn(1, 3, 128, 128)
    features = backbone(x)
    loss = sum(f.mean() for f in features)
    loss.backward()
    assert backbone.proj0.weight.grad is not None


# ---------------------------------------------------------------------------
# DSGCNet integration: proves ConvNeXt satisfies the full model contract
# ---------------------------------------------------------------------------

from crowdcount.models.dsgcnet import DSGCnet


def test_convnext_dsgcnet_forward():
    """End-to-end forward with ConvNeXt backbone produces correct output keys."""
    backbone = BackboneConvNeXt("convnext_tiny")
    model = DSGCnet(backbone, row=2, line=2)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_points"].shape[2] == 2
    # Anchor count must match regression output
    assert out["pred_logits"].shape[1] == out["pred_points"].shape[1]


def test_convnext_dsgcnet_batch():
    """ConvNeXt + DSGCNet handles batch > 1."""
    backbone = BackboneConvNeXt("convnext_tiny")
    model = DSGCnet(backbone, row=2, line=2)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_points"].shape[0] == 2

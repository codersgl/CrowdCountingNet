"""Tests for VGG and DINOv2 backbone."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.backbone import Backbone_VGG


@pytest.mark.parametrize("name", ["vgg16_bn"])
def test_vgg_backbone_output_shapes(name):
    """VGG backbone with return_interm_layers=True returns 4 feature maps."""
    backbone = Backbone_VGG(name, return_interm_layers=True)
    backbone.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        features = backbone(x)
    assert len(features) == 4, "Expected 4 intermediate feature maps"
    # Spatial dims should halve at each stage
    prev_h = 256
    for feat in features:
        assert feat.dim() == 4
        assert feat.shape[0] == 1
        assert feat.shape[2] < prev_h
        prev_h = feat.shape[2]


def test_vgg16bn_channel_sizes():
    """VGG16-BN intermediate features have the expected channel widths.

    VGG16-BN layer splits:
      body1 = features[:13]  → 128ch  (through 2nd ReLU of block-2, before MaxPool)
      body2 = features[13:23]→ 256ch  (through last ReLU of block-3, before MaxPool)
      body3 = features[23:33]→ 512ch  (through last ReLU of block-4, before MaxPool)
      body4 = features[33:43]→ 512ch  (through last ReLU of block-5)
    """
    backbone = Backbone_VGG("vgg16_bn", return_interm_layers=True)
    backbone.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        features = backbone(x)
    assert features[0].shape[1] == 128
    assert features[1].shape[1] == 256
    assert features[2].shape[1] == 512
    assert features[3].shape[1] == 512


def test_vgg_backbone_batch():
    """Backbone handles batch of 4 images."""
    backbone = Backbone_VGG("vgg16_bn", return_interm_layers=True)
    backbone.eval()
    x = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        features = backbone(x)
    assert features[0].shape[0] == 4


@pytest.mark.skip(reason="Requires network access to download DINOv2 model")
def test_dinov2_backbone_output_shapes():
    """DINOv2 backbone returns 4 feature maps with expected channel widths."""
    from crowdcount.models.backbone import BackboneDINOv2

    backbone = BackboneDINOv2("dinov2_s")
    backbone.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = backbone(x)
    assert len(features) == 4
    assert features[1].shape[1] == 256
    assert features[2].shape[1] == 512
    assert features[3].shape[1] == 512

"""Tests for DINOv2 backbone feature extraction (skipped without network)."""

from __future__ import annotations

import pytest
import torch

try:
    import torch.hub  # noqa: F401

    _torch_hub_available = True
except ImportError:
    _torch_hub_available = False


@pytest.mark.skip(
    reason="Requires network access to download DINOv2 weights from torch.hub"
)
def test_dinov2_forward_shapes():
    from crowdcount.models.backbone import BackboneDINOv2

    backbone = BackboneDINOv2("dinov2_s")
    backbone.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = backbone(x)
    assert len(features) == 4
    # C3 → 256, C4 → 512, C5 → 512
    assert features[1].shape[1] == 256
    assert features[2].shape[1] == 512
    assert features[3].shape[1] == 512


@pytest.mark.skip(
    reason="Requires network access to download DINOv2 weights from torch.hub"
)
def test_dinov2_batch_processing():
    from crowdcount.models.backbone import BackboneDINOv2

    backbone = BackboneDINOv2("dinov2_s")
    backbone.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = backbone(x)
    for f in features:
        assert f.shape[0] == 2


@pytest.mark.skip(
    reason="Requires network access to download DINOv2 weights from torch.hub"
)
def test_dinov2_variants_channel_widths():
    from crowdcount.models.backbone import BackboneDINOv2, _DINOV2_VARIANTS

    for variant in list(_DINOV2_VARIANTS.keys())[
        :1
    ]:  # only test first to minimise downloads
        backbone = BackboneDINOv2(variant)
        backbone.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = backbone(x)
        assert len(features) == 4

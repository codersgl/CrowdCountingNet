"""Tests for DINOv2 semantic feature extraction (placeholder tests)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skip(
    reason="Requires DINOv2 download — semantic feature tests are placeholders"
)
def test_dinov2_semantic_clustering():
    """Placeholder: verify DINOv2 produces semantically meaningful features."""
    from crowdcount.models.backbone import BackboneDINOv2

    backbone = BackboneDINOv2("dinov2_s")
    backbone.eval()

    # Two views of the same random image should be more similar than two unrelated images
    img_a = torch.randn(1, 3, 224, 224)
    img_b = img_a + 0.01 * torch.randn_like(img_a)  # near duplicate
    img_c = torch.randn(1, 3, 224, 224)  # unrelated

    with torch.no_grad():
        feat_a = backbone(img_a)[1].flatten()
        feat_b = backbone(img_b)[1].flatten()
        feat_c = backbone(img_c)[1].flatten()

    import torch.nn.functional as F

    sim_ab = F.cosine_similarity(feat_a.unsqueeze(0), feat_b.unsqueeze(0)).item()
    sim_ac = F.cosine_similarity(feat_a.unsqueeze(0), feat_c.unsqueeze(0)).item()
    assert sim_ab > sim_ac, "Near-duplicate images should be more similar"


@pytest.mark.skip(reason="Requires DINOv2 download")
def test_dinov2_feature_dims_with_dsgcnet():
    """Placeholder: verify DINOv2 backbone integrates with DSGCnet end-to-end."""
    from crowdcount.models.backbone import BackboneDINOv2
    from crowdcount.models.dsgcnet import DSGCnet

    backbone = BackboneDINOv2("dinov2_s")
    model = DSGCnet(backbone, row=2, line=2)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert "pred_logits" in out
    assert "pred_points" in out
    assert "density_out" in out

"""Tests for point-to-density feedback modules."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.head import (
    PointGuidedDensityRefiner,
    point_predictions_to_density_map,
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


def test_point_predictions_to_density_map_preserves_score_sum() -> None:
    points = torch.tensor([[[16.0, 16.0], [48.0, 48.0]]])
    scores = torch.tensor([[0.75, 0.25]])

    heatmap = point_predictions_to_density_map(
        points,
        scores,
        density_size=(8, 8),
        image_size=(64, 64),
        gaussian_sigma=1.0,
    )

    assert heatmap.shape == (1, 1, 8, 8)
    assert torch.isfinite(heatmap).all()
    assert (heatmap >= 0).all()
    assert torch.allclose(heatmap.sum(), scores.sum(), atol=1e-5)


def test_point_predictions_to_density_map_threshold_can_empty_map() -> None:
    points = torch.tensor([[[16.0, 16.0], [48.0, 48.0]]])
    scores = torch.tensor([[0.2, 0.3]])

    heatmap = point_predictions_to_density_map(
        points,
        scores,
        density_size=(8, 8),
        image_size=(64, 64),
        gaussian_sigma=0.0,
        score_threshold=0.5,
    )

    assert heatmap.shape == (1, 1, 8, 8)
    assert heatmap.sum().item() == 0.0


def test_point_guided_density_refiner_is_identity_initialised() -> None:
    torch.manual_seed(0)
    refiner = PointGuidedDensityRefiner(
        feature_channels=4,
        hidden_channels=8,
        max_delta=0.5,
        strength_init=1e-3,
    ).eval()
    feature = torch.randn(2, 4, 8, 8)
    density = torch.rand(2, 1, 8, 8) + 0.1
    heatmap = torch.rand(2, 1, 8, 8)

    with torch.no_grad():
        refined = refiner(feature, density, heatmap)

    assert torch.allclose(refined, density, atol=1e-6)
    assert refiner.last_delta is not None
    assert refiner.last_delta.abs().max().item() == 0.0


def test_dsgcnet_point_density_feedback_forward_outputs_debug_tensors() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "point_density_feedback": {
                    "enabled": True,
                    "hidden_channels": 8,
                    "gaussian_sigma": 1.0,
                    "score_threshold": 0.0,
                    "detach_points": True,
                    "detach_scores": True,
                    "max_delta": 0.5,
                    "strength_init": 1e-3,
                    "debug": True,
                },
                "density_head_version": "v1",
                "use_ms_density_head": False,
            }
        }
    )
    model = DSGCnet(TinyVGGBackbone(), row=2, line=2, cfg=cfg).eval()

    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 128, 128))

    assert outputs["density_out"].shape == outputs["density_base"].shape
    assert outputs["point_feedback_heatmap"].shape == outputs["density_out"].shape
    assert "point_feedback_stats" in outputs
    assert torch.isfinite(outputs["density_out"]).all()

"""Tests for evaluation TTA module and integration with evaluate_crowd_no_overlap."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.engine import evaluate_crowd_no_overlap
from crowdcount.eval.tta import (
    _aggregate,
    _resize_to_divisor,
    count_from_outputs,
    tta_predict,
)


# ---------------------------------------------------------------------------
# Dummy model whose output is a deterministic function of input shape.
# ---------------------------------------------------------------------------


class ShapeAwareDummy(nn.Module):
    """Model where predict_cnt depends on input H*W and is symmetric to flip."""

    def __init__(self, num_queries: int = 32):
        super().__init__()
        self.num_queries = num_queries
        self.linear = nn.Linear(1, 1)

    def forward(self, samples, **kwargs):
        B, _, H, W = samples.shape
        # Logits: first k queries get strong "foreground" score, rest "background".
        # k scales with H*W so larger inputs predict more points.
        k = max(1, (H * W) // (128 * 128))
        k = min(k, self.num_queries)
        logits = torch.full((B, self.num_queries, 2), -10.0)
        logits[:, :, 0] = -10.0  # background
        logits[:, :k, 1] = 10.0  # foreground
        logits[:, k:, 1] = -10.0
        density = torch.full((B, 1, max(1, H // 8), max(1, W // 8)), 0.01)
        return {
            "pred_logits": logits,
            "pred_points": torch.zeros(B, self.num_queries, 2),
            "density_out": density,
        }


def _val_batch(H: int = 128, W: int = 128, n_pts: int = 5):
    samples = torch.randn(1, 3, H, W)
    targets = [
        {
            "labels": torch.ones(n_pts, dtype=torch.long),
            "point": torch.rand(n_pts, 2) * H,
        }
    ]
    return samples, targets


# ---------------------------------------------------------------------------
# _aggregate / _resize helpers
# ---------------------------------------------------------------------------


def test_aggregate_mean():
    assert _aggregate([1.0, 2.0, 3.0], "mean") == pytest.approx(2.0)


def test_aggregate_median():
    assert _aggregate([1.0, 100.0, 3.0], "median") == pytest.approx(3.0)


def test_aggregate_empty():
    assert _aggregate([], "mean") == 0.0


def test_resize_to_divisor_snaps_to_multiple():
    img = torch.randn(1, 3, 100, 200)
    out = _resize_to_divisor(img, scale=1.0, divisor=128)
    assert out.shape[-2] % 128 == 0
    assert out.shape[-1] % 128 == 0


def test_resize_to_divisor_identity():
    img = torch.randn(1, 3, 128, 256)
    out = _resize_to_divisor(img, scale=1.0, divisor=128)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# count_from_outputs (extracted logic — must match engine semantics)
# ---------------------------------------------------------------------------


def test_count_from_outputs_threshold_branch():
    model = ShapeAwareDummy()
    samples = torch.randn(1, 3, 128, 128)
    outputs = model(samples)
    eval_cfg = OmegaConf.create({"method": "threshold", "threshold": 0.5})
    cnt, dsum = count_from_outputs(outputs, eval_cfg)
    # k = 1 for 128x128
    assert cnt == 1.0
    assert dsum > 0


def test_count_from_outputs_default_threshold_no_cfg():
    model = ShapeAwareDummy()
    samples = torch.randn(1, 3, 128, 128)
    outputs = model(samples)
    cnt, _ = count_from_outputs(outputs, None)
    assert cnt == 1.0


def test_count_from_outputs_density_guided():
    model = ShapeAwareDummy()
    samples = torch.randn(1, 3, 128, 128)
    outputs = model(samples)
    eval_cfg = OmegaConf.create(
        {"method": "density_guided", "threshold": 0.5, "min_score": 0.3}
    )
    cnt, _ = count_from_outputs(outputs, eval_cfg)
    assert cnt >= 0.0


# ---------------------------------------------------------------------------
# tta_predict
# ---------------------------------------------------------------------------


def test_tta_predict_single_view_matches_baseline():
    """scales=[1.0], flip=False → identical to a plain forward."""
    model = ShapeAwareDummy()
    samples = torch.randn(1, 3, 128, 128)

    def fwd(s, d):
        return model(s)

    tta_cfg = OmegaConf.create(
        {
            "enabled": True,
            "flip": False,
            "scales": [1.0],
            "aggregate": "mean",
            "size_divisor": 128,
        }
    )
    eval_cfg = OmegaConf.create({"method": "threshold", "threshold": 0.5})
    cnt_tta, _ = tta_predict(samples, None, fwd, tta_cfg, eval_cfg)
    cnt_base, _ = count_from_outputs(model(samples), eval_cfg)
    assert cnt_tta == pytest.approx(cnt_base)


def test_tta_predict_flip_symmetric_count_unchanged():
    """For flip-symmetric model, flip TTA should not change aggregated count."""
    model = ShapeAwareDummy()
    samples = torch.randn(1, 3, 128, 128)

    def fwd(s, d):
        return model(s)

    tta_cfg = OmegaConf.create(
        {
            "enabled": True,
            "flip": True,
            "scales": [1.0],
            "aggregate": "mean",
            "size_divisor": 128,
        }
    )
    eval_cfg = OmegaConf.create({"method": "threshold", "threshold": 0.5})
    cnt_tta, _ = tta_predict(samples, None, fwd, tta_cfg, eval_cfg)
    cnt_base, _ = count_from_outputs(model(samples), eval_cfg)
    assert cnt_tta == pytest.approx(cnt_base)


def test_tta_predict_multi_scale_mean():
    """Multi-scale aggregation = mean of per-view counts."""
    model = ShapeAwareDummy(num_queries=64)
    samples = torch.randn(1, 3, 128, 128)

    def fwd(s, d):
        return model(s)

    tta_cfg = OmegaConf.create(
        {
            "enabled": True,
            "flip": False,
            "scales": [1.0, 2.0],
            "aggregate": "mean",
            "size_divisor": 128,
        }
    )
    eval_cfg = OmegaConf.create({"method": "threshold", "threshold": 0.5})
    cnt_tta, _ = tta_predict(samples, None, fwd, tta_cfg, eval_cfg)
    # scale 1.0 → 128x128 → k=1; scale 2.0 → 256x256 → k=4. mean = 2.5
    assert cnt_tta == pytest.approx(2.5)


def test_tta_predict_passes_depth():
    """Depth tensor should be resized & flipped in lock-step with samples."""
    captured: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class M(nn.Module):
        def forward(self, s, **kw):
            B, _, H, W = s.shape
            return {
                "pred_logits": torch.zeros(B, 4, 2),
                "pred_points": torch.zeros(B, 4, 2),
                "density_out": torch.zeros(B, 1, max(1, H // 8), max(1, W // 8)),
            }

    model = M()
    samples = torch.randn(1, 3, 128, 128)
    depth = torch.randn(1, 1, 128, 128)

    def fwd(s, d):
        captured.append((s.shape, None if d is None else d.shape))
        return model(s)

    tta_cfg = OmegaConf.create(
        {
            "enabled": True,
            "flip": True,
            "scales": [1.0],
            "aggregate": "mean",
            "size_divisor": 128,
        }
    )
    tta_predict(samples, depth, fwd, tta_cfg, None)
    # 1 scale × 2 flip variants = 2 calls; depth shape must mirror sample shape
    assert len(captured) == 2
    for s_shape, d_shape in captured:
        assert d_shape is not None
        assert s_shape[-2:] == d_shape[-2:]


# ---------------------------------------------------------------------------
# Integration: evaluate_crowd_no_overlap with eval_tta cfg
# ---------------------------------------------------------------------------


def test_evaluate_with_tta_disabled_matches_no_cfg():
    """eval_tta.enabled=false should produce identical metrics to no cfg."""
    torch.manual_seed(0)
    model = ShapeAwareDummy()
    loader = [_val_batch(n_pts=1) for _ in range(3)]
    device = torch.device("cpu")

    cfg = OmegaConf.create(
        {
            "eval_counting": {"method": "threshold", "threshold": 0.5},
            "eval_tta": {
                "enabled": False,
                "flip": True,
                "scales": [1.0, 1.2],
                "aggregate": "mean",
                "size_divisor": 128,
            },
        }
    )
    a = evaluate_crowd_no_overlap(model, loader, device, cfg=cfg)
    b = evaluate_crowd_no_overlap(model, loader, device)  # no cfg
    assert a == b


def test_evaluate_with_tta_enabled_runs():
    """Smoke test: TTA path executes end-to-end and returns finite metrics."""
    model = ShapeAwareDummy(num_queries=64)
    loader = [_val_batch(n_pts=2) for _ in range(2)]
    device = torch.device("cpu")
    cfg = OmegaConf.create(
        {
            "eval_counting": {"method": "threshold", "threshold": 0.5},
            "eval_tta": {
                "enabled": True,
                "flip": True,
                "scales": [1.0],
                "aggregate": "mean",
                "size_divisor": 128,
            },
        }
    )
    mae, mse, d_mae, d_mse = evaluate_crowd_no_overlap(model, loader, device, cfg=cfg)
    for v in (mae, mse, d_mae, d_mse):
        assert isinstance(v, float)
        assert v >= 0
        assert torch.isfinite(torch.tensor(v))

"""Test-time augmentation (TTA) for crowd-counting evaluation.

Aggregation is performed at the *count* level (per-image scalar `predict_cnt`
and `density_sum`) rather than at the point level, because point-level NMS
across scale-augmented views is fragile (anchor positions shift with input
size). Mean / median aggregation of counts is the standard practice in
crowd-counting TTA.

Public API
----------
- ``count_from_outputs(outputs, eval_cfg)`` → ``(predict_cnt, density_sum)``
  Replicates the counting logic used inside ``evaluate_crowd_no_overlap``.
- ``tta_predict(model, samples, depth_map, tta_cfg, eval_cfg, forward_fn)``
  Returns the aggregated ``(predict_cnt, density_sum)`` over all configured
  TTA views (flip × scales).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Counting logic shared with evaluate_crowd_no_overlap
# ---------------------------------------------------------------------------


def count_from_outputs(outputs: dict, eval_cfg=None) -> tuple[float, float]:
    """Extract ``(predict_cnt, density_sum)`` from a single forward pass.

    Mirrors the threshold / density_guided branches in
    ``crowdcount.engine.evaluate_crowd_no_overlap``.

    Assumes ``outputs['pred_logits']`` and ``outputs['density_out']`` have
    batch size 1.
    """
    pred_logits = outputs["pred_logits"]
    et_dmap = outputs["density_out"]
    assert pred_logits.shape[0] == 1, (
        f"count_from_outputs expects batch_size=1 (got {pred_logits.shape[0]})"
    )

    outputs_scores = torch.softmax(pred_logits, dim=-1)[0, :, 1]
    et_dmap_sum = float(torch.sum(et_dmap).item())

    counting_method = (
        str(getattr(eval_cfg, "method", "threshold")) if eval_cfg else "threshold"
    )

    if counting_method == "density_guided":
        min_score = float(getattr(eval_cfg, "min_score", 0.3))
        density_cnt = max(1, round(et_dmap_sum))
        valid_mask = outputs_scores > min_score
        num_valid = int(valid_mask.sum().item())
        if num_valid > 0 and density_cnt <= num_valid:
            _, topk_indices = outputs_scores[valid_mask].topk(
                min(density_cnt, num_valid)
            )
            predict_cnt = float(len(topk_indices))
        else:
            predict_cnt = float(num_valid)
    else:
        threshold = float(getattr(eval_cfg, "threshold", 0.5)) if eval_cfg else 0.5
        predict_cnt = float(int((outputs_scores > threshold).sum()))

    return predict_cnt, et_dmap_sum


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------


def _round_to_divisor(x: int, divisor: int) -> int:
    """Round ``x`` to the nearest multiple of ``divisor``, minimum ``divisor``."""
    if divisor <= 1:
        return max(1, x)
    return max(divisor, int(round(x / divisor)) * divisor)


def _resize_to_divisor(
    img: torch.Tensor,
    scale: float,
    divisor: int,
) -> torch.Tensor:
    """Scale image and snap H/W to multiples of ``divisor``.

    Args:
        img: tensor [B, C, H, W]
        scale: multiplicative scale factor (e.g. 0.8 / 1.0 / 1.2)
        divisor: required spatial divisor (e.g. 128 for VGG backbone)
    """
    _, _, H, W = img.shape
    new_h = _round_to_divisor(int(round(H * scale)), divisor)
    new_w = _round_to_divisor(int(round(W * scale)), divisor)
    if new_h == H and new_w == W:
        return img
    return F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _aggregate(values: list[float], method: str) -> float:
    """Aggregate per-view scalar predictions."""
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float64)
    method = (method or "mean").lower()
    if method == "median":
        return float(t.median().item())
    # default: mean
    return float(t.mean().item())


def _iter_tta_views(
    samples: torch.Tensor,
    depth_map: torch.Tensor | None,
    scales: Iterable[float],
    flip: bool,
    size_divisor: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor | None]]:
    """Yield ``(samples_view, depth_view)`` for every configured TTA view."""
    for scale in scales:
        s = _resize_to_divisor(samples, scale, size_divisor)
        d = (
            _resize_to_divisor(depth_map, scale, size_divisor)
            if depth_map is not None
            else None
        )
        yield s, d
        if flip:
            s_f = torch.flip(s, dims=[-1])
            d_f = torch.flip(d, dims=[-1]) if d is not None else None
            yield s_f, d_f


# ---------------------------------------------------------------------------
# Public TTA entry point
# ---------------------------------------------------------------------------


def tta_predict(
    samples: torch.Tensor,
    depth_map: torch.Tensor | None,
    forward_fn: Callable[[torch.Tensor, torch.Tensor | None], dict],
    tta_cfg=None,
    eval_cfg=None,
) -> tuple[float, float]:
    """Run TTA forward passes and return aggregated ``(predict_cnt, density_sum)``.

    Args:
        samples: input image tensor [1, 3, H, W] (already on device, normalised)
        depth_map: optional depth tensor [1, 1, H, W] (already on device) or None
        forward_fn: callable ``(samples, depth_map) -> outputs_dict``; should
            invoke the model and return the same dict shape as
            ``DSGCnet.forward``.
        tta_cfg: OmegaConf / dict with keys: enabled, flip, scales,
            aggregate, size_divisor
        eval_cfg: passed through to ``count_from_outputs``

    Returns:
        (predict_cnt, density_sum) aggregated across views.
    """
    flip = bool(getattr(tta_cfg, "flip", False)) if tta_cfg is not None else False
    raw_scales = (
        list(getattr(tta_cfg, "scales", [1.0])) if tta_cfg is not None else [1.0]
    )
    scales = [float(s) for s in raw_scales] or [1.0]
    aggregate = (
        str(getattr(tta_cfg, "aggregate", "mean")) if tta_cfg is not None else "mean"
    )
    size_divisor = (
        int(getattr(tta_cfg, "size_divisor", 128)) if tta_cfg is not None else 128
    )

    cnts: list[float] = []
    dsums: list[float] = []
    for samples_v, depth_v in _iter_tta_views(
        samples, depth_map, scales, flip, size_divisor
    ):
        outputs = forward_fn(samples_v, depth_v)
        cnt, dsum = count_from_outputs(outputs, eval_cfg)
        cnts.append(cnt)
        dsums.append(dsum)

    return _aggregate(cnts, aggregate), _aggregate(dsums, aggregate)

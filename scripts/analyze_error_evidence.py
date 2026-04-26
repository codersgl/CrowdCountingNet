"""Run evidence-strengthening diagnostics for DSGCNet counting errors.

This script collects validation predictions once, then checks whether conclusions
about high-score false positives are robust to matching rules, matching radius,
bootstrap resampling, and visual inspection.

Example:

    uv run python scripts/analyze_error_evidence.py \
        data.data_root=data/shanghaitech/part_A_final \
        +predict.weight_path=outputs/2026-04-22/19-23-32/checkpoints/best_mae.pth \
        model.use_gm=true \
        model.use_dap_neck=true \
        model.use_density_attention=true \
        model.density_head_version=v3 \
        model.gcn_conv_type=gatv2 \
        scheduler=step_lr \
        scheduler.lr_drop=800 \
        data.density_generation.hybrid=true
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.engine import _forward_model
from crowdcount.models import build_model
from crowdcount.utils.logging import logger, setup_logger

MatcherName = Literal["greedy", "hungarian"]


@dataclass
class EvidenceRecord:
    image_id: int
    image_path: str
    gt_count: int
    scores: np.ndarray
    points: np.ndarray
    gt_points: np.ndarray


@dataclass
class MatchResult:
    matched_pred: np.ndarray
    matched_gt: np.ndarray
    pred_to_gt: np.ndarray
    gt_to_pred: np.ndarray
    nearest_gt_dist: np.ndarray
    nearest_gt_idx: np.ndarray
    nearest_pred_dist: np.ndarray
    nearest_pred_idx: np.ndarray


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _parse_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _bias(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred - gt))


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    sample_means = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, len(values), size=len(values))
        sample_means[idx] = float(np.mean(values[sample_idx]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(sample_means, alpha / 2)),
        "ci_high": float(np.quantile(sample_means, 1.0 - alpha / 2)),
    }


def _nearest_arrays(
    pred_points: np.ndarray, gt_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if len(pred_points) == 0:
        nearest_pred_dist = np.full(len(gt_points), np.nan, dtype=np.float64)
        nearest_pred_idx = np.full(len(gt_points), -1, dtype=np.int64)
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            nearest_pred_dist,
            nearest_pred_idx,
            None,
        )
    if len(gt_points) == 0:
        nearest_gt_dist = np.full(len(pred_points), np.nan, dtype=np.float64)
        nearest_gt_idx = np.full(len(pred_points), -1, dtype=np.int64)
        return (
            nearest_gt_dist,
            nearest_gt_idx,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            None,
        )
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    nearest_gt_idx = np.argmin(distances, axis=1).astype(np.int64)
    nearest_gt_dist = distances[np.arange(len(pred_points)), nearest_gt_idx]
    nearest_pred_idx = np.argmin(distances, axis=0).astype(np.int64)
    nearest_pred_dist = distances[nearest_pred_idx, np.arange(len(gt_points))]
    return (
        nearest_gt_dist,
        nearest_gt_idx,
        nearest_pred_dist,
        nearest_pred_idx,
        distances,
    )


def _match_predictions(
    scores: np.ndarray,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_distance: float,
    matcher: MatcherName,
) -> MatchResult:
    matched_pred = np.zeros(len(scores), dtype=bool)
    matched_gt = np.zeros(len(gt_points), dtype=bool)
    pred_to_gt = np.full(len(scores), -1, dtype=np.int64)
    gt_to_pred = np.full(len(gt_points), -1, dtype=np.int64)
    (
        nearest_gt_dist,
        nearest_gt_idx,
        nearest_pred_dist,
        nearest_pred_idx,
        distances,
    ) = _nearest_arrays(pred_points, gt_points)
    if distances is None or len(scores) == 0 or len(gt_points) == 0:
        return MatchResult(
            matched_pred,
            matched_gt,
            pred_to_gt,
            gt_to_pred,
            nearest_gt_dist,
            nearest_gt_idx,
            nearest_pred_dist,
            nearest_pred_idx,
        )

    if matcher == "greedy":
        order = np.argsort(scores)[::-1]
        used_gt: set[int] = set()
        for pred_idx in order:
            pred_idx_int = int(pred_idx)
            for gt_idx in np.argsort(distances[pred_idx_int]):
                gt_idx_int = int(gt_idx)
                if gt_idx_int in used_gt:
                    continue
                if distances[pred_idx_int, gt_idx_int] <= max_distance:
                    matched_pred[pred_idx_int] = True
                    matched_gt[gt_idx_int] = True
                    pred_to_gt[pred_idx_int] = gt_idx_int
                    gt_to_pred[gt_idx_int] = pred_idx_int
                    used_gt.add(gt_idx_int)
                break
    elif matcher == "hungarian":
        cost = distances.copy()
        cost[cost > max_distance] = max_distance + 1e6
        pred_idx, gt_idx = linear_sum_assignment(cost)
        valid = distances[pred_idx, gt_idx] <= max_distance
        for pred, gt in zip(pred_idx[valid], gt_idx[valid]):
            pred_int = int(pred)
            gt_int = int(gt)
            matched_pred[pred_int] = True
            matched_gt[gt_int] = True
            pred_to_gt[pred_int] = gt_int
            gt_to_pred[gt_int] = pred_int
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    return MatchResult(
        matched_pred,
        matched_gt,
        pred_to_gt,
        gt_to_pred,
        nearest_gt_dist,
        nearest_gt_idx,
        nearest_pred_dist,
        nearest_pred_idx,
    )


def _greedy_nms(scores: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    keep = np.zeros(len(scores), dtype=bool)
    suppressed = np.zeros(len(scores), dtype=bool)
    if len(scores) == 0:
        return keep
    order = np.argsort(scores)[::-1]
    for pred_idx in order:
        pred_idx_int = int(pred_idx)
        if suppressed[pred_idx_int]:
            continue
        keep[pred_idx_int] = True
        if radius <= 0:
            continue
        distances = np.linalg.norm(points - points[pred_idx_int], axis=1)
        suppress_mask = (distances <= radius) & (~keep) & (~suppressed)
        suppress_mask[pred_idx_int] = False
        suppressed[suppress_mask] = True
    return keep


def _counts_at_threshold(records: list[EvidenceRecord], threshold: float) -> np.ndarray:
    return np.asarray(
        [np.sum(record.scores > threshold) for record in records], dtype=np.float64
    )


def _counts_after_nms(
    records: list[EvidenceRecord], threshold: float, radius: float
) -> np.ndarray:
    counts = np.zeros(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        mask = record.scores > threshold
        counts[index] = int(_greedy_nms(record.scores[mask], record.points[mask], radius).sum())
    return counts


def _threshold_sweep(
    records: list[EvidenceRecord], gt: np.ndarray, thresholds: np.ndarray
) -> list[dict[str, float]]:
    rows = []
    for threshold in thresholds:
        pred = _counts_at_threshold(records, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "mae": _mae(pred, gt),
                "rmse": _rmse(pred, gt),
                "bias": _bias(pred, gt),
            }
        )
    return rows


def _nms_rows(
    records: list[EvidenceRecord],
    gt: np.ndarray,
    threshold: float,
    radii: list[float],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    raw_counts = _counts_at_threshold(records, threshold)
    raw_abs = np.abs(raw_counts - gt)
    rows: list[dict[str, Any]] = []
    for radius in radii:
        nms_counts = _counts_after_nms(records, threshold, radius)
        nms_abs = np.abs(nms_counts - gt)
        delta = raw_abs - nms_abs
        delta_ci = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "radius": float(radius),
                "raw_mae": _mae(raw_counts, gt),
                "nms_mae": _mae(nms_counts, gt),
                "mae_gain": _mae(raw_counts, gt) - _mae(nms_counts, gt),
                "gain_ci_low": delta_ci["ci_low"],
                "gain_ci_high": delta_ci["ci_high"],
                "removed": int(np.sum(raw_counts - nms_counts)),
                "nms_bias": _bias(nms_counts, gt),
            }
        )
    return rows


def _matching_sensitivity_rows(
    records: list[EvidenceRecord],
    threshold: float,
    high_score: float,
    match_distances: list[float],
    matcher_specs: list[tuple[MatcherName, list[float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    del match_distances
    for matcher, matcher_distances in matcher_specs:
        for match_distance in matcher_distances:
            totals = {
                "pred": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "high_pred": 0,
                "high_fp": 0,
                "high_fp_near_gt": 0,
                "high_fp_near_matched_gt": 0,
                "fn_near_score05": 0,
                "fn_near_any_score": 0,
            }
            nearest_high_fp: list[float] = []
            for record in records:
                candidate_mask = record.scores > threshold
                scores = record.scores[candidate_mask]
                points = record.points[candidate_mask]
                result = _match_predictions(
                    scores,
                    points,
                    record.gt_points,
                    max_distance=match_distance,
                    matcher=matcher,
                )
                matched_gt_set = set(np.flatnonzero(result.matched_gt).tolist())
                fp_mask = ~result.matched_pred
                high_mask = scores >= high_score
                high_fp_mask = high_mask & fp_mask
                nearest_gt_valid = result.nearest_gt_idx >= 0
                high_fp_near_gt_mask = (
                    high_fp_mask
                    & nearest_gt_valid
                    & (result.nearest_gt_dist <= match_distance)
                )
                high_fp_near_matched_gt = 0
                for pred_idx in np.flatnonzero(high_fp_near_gt_mask):
                    if int(result.nearest_gt_idx[pred_idx]) in matched_gt_set:
                        high_fp_near_matched_gt += 1
                if np.any(high_fp_mask):
                    nearest_high_fp.extend(result.nearest_gt_dist[high_fp_mask].tolist())

                fn_near_score05 = 0
                fn_near_any_score = 0
                full_nearest_pred = _nearest_unmatched_gt_candidates(
                    record, result, match_distance, score_threshold=threshold
                )
                fn_near_score05 += int(full_nearest_pred["near_threshold"])
                fn_near_any_score += int(full_nearest_pred["near_any"])

                totals["pred"] += int(len(scores))
                totals["tp"] += int(result.matched_pred.sum())
                totals["fp"] += int(fp_mask.sum())
                totals["fn"] += int((~result.matched_gt).sum())
                totals["high_pred"] += int(high_mask.sum())
                totals["high_fp"] += int(high_fp_mask.sum())
                totals["high_fp_near_gt"] += int(high_fp_near_gt_mask.sum())
                totals["high_fp_near_matched_gt"] += int(high_fp_near_matched_gt)
                totals["fn_near_score05"] += fn_near_score05
                totals["fn_near_any_score"] += fn_near_any_score

                if matcher == "greedy" and abs(match_distance - 8.0) < 1e-6:
                    per_image_rows.append(
                        {
                            "image_id": record.image_id,
                            "image_path": record.image_path,
                            "gt_count": record.gt_count,
                            "pred_count": int(len(scores)),
                            "tp": int(result.matched_pred.sum()),
                            "fp": int(fp_mask.sum()),
                            "fn": int((~result.matched_gt).sum()),
                            "high_fp": int(high_fp_mask.sum()),
                        }
                    )
            rows.append(
                {
                    "matcher": matcher,
                    "match_distance": float(match_distance),
                    **totals,
                    "precision": float(totals["tp"] / max(totals["pred"], 1)),
                    "recall": float(totals["tp"] / max(totals["tp"] + totals["fn"], 1)),
                    "high_fp_near_gt_rate": float(
                        totals["high_fp_near_gt"] / max(totals["high_fp"], 1)
                    ),
                    "high_fp_near_matched_gt_rate": float(
                        totals["high_fp_near_matched_gt"] / max(totals["high_fp"], 1)
                    ),
                    "fn_near_score05_rate": float(
                        totals["fn_near_score05"] / max(totals["fn"], 1)
                    ),
                    "fn_near_any_score_rate": float(
                        totals["fn_near_any_score"] / max(totals["fn"], 1)
                    ),
                    "high_fp_nearest_dist_mean": float(np.mean(nearest_high_fp))
                    if nearest_high_fp
                    else float("nan"),
                    "high_fp_nearest_dist_median": float(np.median(nearest_high_fp))
                    if nearest_high_fp
                    else float("nan"),
                }
            )
    return rows, per_image_rows


def _nearest_unmatched_gt_candidates(
    record: EvidenceRecord,
    result: MatchResult,
    match_distance: float,
    score_threshold: float,
) -> dict[str, int]:
    near_threshold = 0
    near_any = 0
    if len(record.gt_points) == 0 or len(record.points) == 0:
        return {"near_threshold": near_threshold, "near_any": near_any}
    distances = np.linalg.norm(record.gt_points[:, None, :] - record.points[None, :, :], axis=2)
    for gt_idx in np.flatnonzero(~result.matched_gt):
        gt_dist = distances[int(gt_idx)]
        near_mask = gt_dist <= match_distance
        if np.any(near_mask):
            near_any += 1
            if np.any(record.scores[near_mask] > score_threshold):
                near_threshold += 1
    return {"near_threshold": near_threshold, "near_any": near_any}


def _score_band_rows(
    records: list[EvidenceRecord],
    threshold: float,
    match_distance: float,
    matcher: MatcherName,
    bands: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    totals = [
        {"pred": 0, "tp": 0, "fp": 0, "near_gt_fp": 0, "near_matched_gt_fp": 0}
        for _ in range(len(bands) - 1)
    ]
    for record in records:
        candidate_mask = record.scores > threshold
        scores = record.scores[candidate_mask]
        points = record.points[candidate_mask]
        result = _match_predictions(scores, points, record.gt_points, match_distance, matcher)
        matched_gt_set = set(np.flatnonzero(result.matched_gt).tolist())
        for band_idx in range(len(bands) - 1):
            low = float(bands[band_idx])
            high = float(bands[band_idx + 1])
            band_mask = (scores >= low) & (scores <= high) if band_idx == len(bands) - 2 else (scores >= low) & (scores < high)
            fp_mask = band_mask & (~result.matched_pred)
            near_gt_mask = fp_mask & (result.nearest_gt_idx >= 0) & (result.nearest_gt_dist <= match_distance)
            near_matched = 0
            for pred_idx in np.flatnonzero(near_gt_mask):
                if int(result.nearest_gt_idx[pred_idx]) in matched_gt_set:
                    near_matched += 1
            totals[band_idx]["pred"] += int(band_mask.sum())
            totals[band_idx]["tp"] += int((band_mask & result.matched_pred).sum())
            totals[band_idx]["fp"] += int(fp_mask.sum())
            totals[band_idx]["near_gt_fp"] += int(near_gt_mask.sum())
            totals[band_idx]["near_matched_gt_fp"] += near_matched
    for band_idx, total in enumerate(totals):
        rows.append(
            {
                "bin_low": float(bands[band_idx]),
                "bin_high": float(bands[band_idx + 1]),
                **total,
                "precision": float(total["tp"] / max(total["pred"], 1)),
                "near_gt_fp_rate": float(total["near_gt_fp"] / max(total["fp"], 1)),
                "near_matched_gt_fp_rate": float(
                    total["near_matched_gt_fp"] / max(total["fp"], 1)
                ),
            }
        )
    return rows


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
    data_root: str,
) -> list[EvidenceRecord]:
    model.eval()
    records: list[EvidenceRecord] = []
    for index, batch in enumerate(data_loader):
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None
        samples = samples.to(device)
        outputs = _forward_model(model, samples, depth_map=depth_map)
        logits = outputs["pred_logits"][0].detach().cpu()
        scores = torch.softmax(logits, dim=-1)[:, 1].numpy().astype(np.float64)
        target = targets[0]
        image_id = int(target["image_id"].item()) if "image_id" in target else index
        image_path = str(Path(data_root) / "test_data" / "images" / f"IMG_{image_id}.jpg")
        records.append(
            EvidenceRecord(
                image_id=image_id,
                image_path=image_path,
                gt_count=int(target["point"].shape[0]),
                scores=scores,
                points=outputs["pred_points"][0].detach().cpu().numpy().astype(np.float64),
                gt_points=target["point"].detach().cpu().numpy().astype(np.float64),
            )
        )
        if (index + 1) % 25 == 0:
            logger.info(f"Collected predictions for {index + 1}/{len(data_loader.dataset)} images")
    return records


def _visualize_top_images(
    records: list[EvidenceRecord],
    per_image_rows: list[dict[str, Any]],
    output_dir: Path,
    threshold: float,
    high_score: float,
    match_distance: float,
    max_images: int,
) -> list[dict[str, Any]]:
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    record_by_id = {record.image_id: record for record in records}
    selected = sorted(per_image_rows, key=lambda row: int(row["high_fp"]), reverse=True)[:max_images]
    saved: list[dict[str, Any]] = []
    for row in selected:
        record = record_by_id[int(row["image_id"])]
        image = cv2.imread(record.image_path)
        if image is None:
            logger.warning(f"Could not read image for visualization: {record.image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        candidate_mask = record.scores > threshold
        scores = record.scores[candidate_mask]
        points = record.points[candidate_mask]
        result = _match_predictions(scores, points, record.gt_points, match_distance, "greedy")
        fp_mask = ~result.matched_pred
        high_fp_mask = fp_mask & (scores >= high_score)
        fn_gt = record.gt_points[~result.matched_gt]

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        if len(record.gt_points):
            plt.scatter(record.gt_points[:, 0], record.gt_points[:, 1], s=18, c="lime", marker="+", label="GT")
        if np.any(result.matched_pred):
            plt.scatter(points[result.matched_pred, 0], points[result.matched_pred, 1], s=8, c="cyan", label="TP")
        if np.any(fp_mask):
            plt.scatter(points[fp_mask, 0], points[fp_mask, 1], s=10, c="red", alpha=0.55, label="FP")
        if np.any(high_fp_mask):
            plt.scatter(points[high_fp_mask, 0], points[high_fp_mask, 1], s=32, facecolors="none", edgecolors="magenta", linewidths=1.2, label="FP score>=0.9")
        if len(fn_gt):
            plt.scatter(fn_gt[:, 0], fn_gt[:, 1], s=34, c="yellow", marker="x", label="FN GT")
        plt.title(
            f"IMG_{record.image_id}: GT={record.gt_count}, pred={len(scores)}, FP={int(fp_mask.sum())}, highFP={int(high_fp_mask.sum())}"
        )
        plt.axis("off")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        out_path = vis_dir / f"IMG_{record.image_id}_high_fp.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append({"image_id": record.image_id, "path": str(out_path), **row})
    return saved


def _print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    logger.info(title)
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    logger.info(fmt.format(*headers))
    logger.info(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        logger.info(fmt.format(*row))


def _save_plots(
    output_dir: Path,
    threshold_rows: list[dict[str, float]],
    nms_rows: list[dict[str, Any]],
    band_rows: list[dict[str, float | int]],
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot([row["threshold"] for row in threshold_rows], [row["mae"] for row in threshold_rows])
    best = min(threshold_rows, key=lambda row: row["mae"])
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5")
    plt.axvline(best["threshold"], color="red", linestyle="--", linewidth=1, label="best")
    plt.xlabel("Score threshold")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sensitivity.png", dpi=160)
    plt.close()

    radii = [float(row["radius"]) for row in nms_rows]
    gains = [float(row["mae_gain"]) for row in nms_rows]
    ci_low = [float(row["gain_ci_low"]) for row in nms_rows]
    ci_high = [float(row["gain_ci_high"]) for row in nms_rows]
    lower_err = np.asarray(gains) - np.asarray(ci_low)
    upper_err = np.asarray(ci_high) - np.asarray(gains)
    plt.figure(figsize=(7, 5))
    plt.errorbar(radii, gains, yerr=[lower_err, upper_err], marker="o", capsize=3)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("NMS radius (px)")
    plt.ylabel("MAE gain vs raw")
    plt.tight_layout()
    plt.savefig(output_dir / "nms_gain_bootstrap_ci.png", dpi=160)
    plt.close()

    labels = [f"{row['bin_low']:.1f}-{row['bin_high']:.1f}" for row in band_rows]
    x = np.arange(len(labels))
    fp = np.asarray([row["fp"] for row in band_rows], dtype=np.float64)
    near_matched = np.asarray([row["near_matched_gt_fp"] for row in band_rows], dtype=np.float64)
    plt.figure(figsize=(9, 5))
    plt.bar(x, fp, label="FP")
    plt.bar(x, near_matched, label="FP near matched GT")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fp_attribution_by_score_band.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    evidence_cfg = getattr(cfg, "evidence_diag", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(
        _cfg_get(evidence_cfg, "output_dir", f"outputs/error_evidence/{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="error_evidence.log")

    threshold = float(_cfg_get(evidence_cfg, "threshold", 0.5))
    high_score = float(_cfg_get(evidence_cfg, "high_score", 0.9))
    match_distances = _parse_float_list(
        _cfg_get(evidence_cfg, "match_distances", None), [4.0, 6.0, 8.0, 10.0, 12.0]
    )
    nms_radii = _parse_float_list(
        _cfg_get(evidence_cfg, "nms_radii", None), [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    )
    threshold_min = float(_cfg_get(evidence_cfg, "threshold_min", 0.1))
    threshold_max = float(_cfg_get(evidence_cfg, "threshold_max", 0.95))
    threshold_step = float(_cfg_get(evidence_cfg, "threshold_step", 0.01))
    n_bootstrap = int(_cfg_get(evidence_cfg, "n_bootstrap", 500))
    seed = int(_cfg_get(evidence_cfg, "seed", 42))
    max_visualizations = int(_cfg_get(evidence_cfg, "max_visualizations", 8))
    bands = np.arange(0.0, 1.0 + 0.05, 0.1)
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)

    predict_cfg = OmegaConf.to_container(cfg, resolve=True).get("predict", {})
    weight_path = str(predict_cfg.get("weight_path", "weights/SHTechA.pth"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}; output_dir={output_dir}")

    model = build_model(cfg, training=False).to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {weight_path}")

    use_depth = bool(getattr(cfg.model, "use_depth", False))
    _, val_set = build_dataset(cfg)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(val_set),
        drop_last=False,
        collate_fn=collate_fn_crowd_depth if use_depth else collate_fn_crowd,
        num_workers=cfg.num_workers,
    )
    records = _collect_records(
        model, val_loader, device, use_depth=use_depth, data_root=str(cfg.data.data_root)
    )
    logger.info("Finished prediction collection; running statistical diagnostics")
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    raw_counts = _counts_at_threshold(records, threshold)
    raw_abs = np.abs(raw_counts - gt)

    threshold_rows = _threshold_sweep(records, gt, thresholds)
    best_threshold = min(threshold_rows, key=lambda row: row["mae"])
    nms_summary = _nms_rows(
        records,
        gt,
        threshold=threshold,
        radii=nms_radii,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    matching_rows, per_image_rows = _matching_sensitivity_rows(
        records,
        threshold=threshold,
        high_score=high_score,
        match_distances=match_distances,
        matcher_specs=[("greedy", match_distances)],
    )
    band_rows = _score_band_rows(records, threshold, 8.0, "greedy", bands)
    visualizations = _visualize_top_images(
        records,
        per_image_rows,
        output_dir,
        threshold=threshold,
        high_score=high_score,
        match_distance=8.0,
        max_images=max_visualizations,
    )
    raw_ci = _bootstrap_ci(raw_abs, n_bootstrap=n_bootstrap, seed=seed)

    summary = {
        "config": {
            "weight_path": weight_path,
            "num_images": len(records),
            "threshold": threshold,
            "high_score": high_score,
            "match_distances": match_distances,
            "nms_radii": nms_radii,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "raw": {
            "mae": _mae(raw_counts, gt),
            "rmse": _rmse(raw_counts, gt),
            "bias": _bias(raw_counts, gt),
            "mae_ci_low": raw_ci["ci_low"],
            "mae_ci_high": raw_ci["ci_high"],
        },
        "best_threshold_full": best_threshold,
        "threshold_sweep": threshold_rows,
        "nms_bootstrap": nms_summary,
        "matching_sensitivity": matching_rows,
        "score_bands_greedy_8px": band_rows,
        "top_high_fp_images": visualizations,
    }
    with (output_dir / "error_evidence_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    _save_plots(output_dir, threshold_rows, nms_summary, band_rows)

    _print_table(
        "--- Raw Count ---",
        ["metric", "value"],
        [
            ["mae", f"{summary['raw']['mae']:.3f}"],
            ["mae_95ci", f"[{raw_ci['ci_low']:.3f}, {raw_ci['ci_high']:.3f}]"],
            ["rmse", f"{summary['raw']['rmse']:.3f}"],
            ["bias", f"{summary['raw']['bias']:+.3f}"],
            ["best_threshold_full", f"{best_threshold['threshold']:.2f}"],
            ["best_threshold_mae", f"{best_threshold['mae']:.3f}"],
        ],
    )
    _print_table(
        "--- NMS Bootstrap Gain ---",
        ["radius", "nms_mae", "gain", "gain_95ci", "removed", "bias"],
        [
            [
                f"{row['radius']:g}",
                f"{row['nms_mae']:.3f}",
                f"{row['mae_gain']:+.3f}",
                f"[{row['gain_ci_low']:+.3f}, {row['gain_ci_high']:+.3f}]",
                row["removed"],
                f"{row['nms_bias']:+.3f}",
            ]
            for row in nms_summary
        ],
    )
    _print_table(
        "--- Matching Sensitivity ---",
        [
            "matcher",
            "dist",
            "tp",
            "fp",
            "fn",
            "hi_fp",
            "hi_fp_near_gt",
            "hi_fp_near_matched",
            "fn_near_s05",
        ],
        [
            [
                row["matcher"],
                f"{row['match_distance']:g}",
                row["tp"],
                row["fp"],
                row["fn"],
                row["high_fp"],
                f"{row['high_fp_near_gt_rate']:.3f}",
                f"{row['high_fp_near_matched_gt_rate']:.3f}",
                f"{row['fn_near_score05_rate']:.3f}",
            ]
            for row in matching_rows
        ],
    )
    _print_table(
        "--- Score Bands, Greedy 8px ---",
        ["band", "pred", "tp", "fp", "prec", "fp_near_gt", "fp_near_matched"],
        [
            [
                f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
                row["pred"],
                row["tp"],
                row["fp"],
                f"{row['precision']:.3f}",
                f"{row['near_gt_fp_rate']:.3f}",
                f"{row['near_matched_gt_fp_rate']:.3f}",
            ]
            for row in band_rows
        ],
    )
    logger.info(f"Saved evidence diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
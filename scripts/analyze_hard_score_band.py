"""Diagnose hard query score bands for a trained DSGCNet checkpoint.

This script looks inside the ambiguous score range (typically 0.4-0.8) and
reports whether errors are mostly high-score false positives, missed ground
truth points, or localization misses.

Example:

    uv run python scripts/analyze_hard_score_band.py \
        data.data_root=data/shanghaitech/part_A_final \
        +predict.weight_path=outputs/2026-04-22/19-23-32/checkpoints/best_mae.pth \
        model.use_gm=true \
        model.use_dap_neck=true \
        model.use_density_attention=true \
        model.density_head_version=v3 \
        model.gcn_conv_type=gatv2 \
        data.density_generation.hybrid=true
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.engine import _forward_model
from crowdcount.models import build_model
from crowdcount.utils.logging import logger, setup_logger


@dataclass
class ImageBandStats:
    image_id: int
    gt_count: int
    pred_count_05: int
    tp_total: int
    fp_total: int
    fn_total: int
    band_rows: list[dict[str, float | int]]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _finite_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _match_predictions(
    scores: np.ndarray,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Greedy score-ordered one-to-one matching.

    Returns:
        matched_pred: bool per prediction
        nearest_dist: nearest GT distance per prediction
        matched_dist: matched GT distance per prediction (nan if unmatched)
        unmatched_gt_count: number of GT points not matched by any prediction
    """
    matched_pred = np.zeros(len(scores), dtype=bool)
    matched_dist = np.full(len(scores), np.nan, dtype=np.float64)
    if len(scores) == 0:
        return (
            matched_pred,
            np.zeros(0, dtype=np.float64),
            matched_dist,
            int(len(gt_points)),
        )
    if len(gt_points) == 0:
        return (
            matched_pred,
            np.full(len(scores), np.nan, dtype=np.float64),
            matched_dist,
            0,
        )

    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    nearest_dist = np.min(distances, axis=1)
    order = np.argsort(scores)[::-1]
    matched_gt: set[int] = set()
    for pred_idx in order:
        nearest_gt_order = np.argsort(distances[pred_idx])
        for gt_idx in nearest_gt_order:
            gt_idx_int = int(gt_idx)
            if gt_idx_int in matched_gt:
                continue
            if distances[pred_idx, gt_idx_int] <= max_distance:
                matched_pred[pred_idx] = True
                matched_dist[pred_idx] = float(distances[pred_idx, gt_idx_int])
                matched_gt.add(gt_idx_int)
            break
    return (
        matched_pred,
        nearest_dist,
        matched_dist,
        int(len(gt_points) - len(matched_gt)),
    )


def _band_stats_for_image(
    image_id: int,
    gt_count: int,
    scores: np.ndarray,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    bands: np.ndarray,
    max_distance: float,
) -> ImageBandStats:
    matched_pred, nearest_dist, matched_dist, unmatched_gt_count = _match_predictions(
        scores,
        pred_points,
        gt_points,
        max_distance=max_distance,
    )
    band_rows: list[dict[str, float | int]] = []
    for idx in range(len(bands) - 1):
        low = float(bands[idx])
        high = float(bands[idx + 1])
        if idx == len(bands) - 2:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        band_pred = int(mask.sum())
        tp = int((matched_pred & mask).sum())
        fp = int(band_pred - tp)
        band_rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "pred": band_pred,
                "tp": tp,
                "fp": fp,
                "precision": float(tp / max(band_pred, 1)),
                "nearest_dist_mean": _finite_mean(nearest_dist[mask]),
                "nearest_dist_median": _finite_median(nearest_dist[mask]),
                "matched_dist_mean": _finite_mean(matched_dist[matched_pred & mask]),
                "matched_dist_median": _finite_median(
                    matched_dist[matched_pred & mask]
                ),
            }
        )
    return ImageBandStats(
        image_id=image_id,
        gt_count=gt_count,
        pred_count_05=int(np.sum(scores > 0.5)),
        tp_total=int(matched_pred.sum()),
        fp_total=int((scores > 0.5).sum() - (matched_pred & (scores > 0.5)).sum()),
        fn_total=unmatched_gt_count,
        band_rows=band_rows,
    )


def _aggregate_band_rows(
    image_stats: list[ImageBandStats], bands: np.ndarray
) -> list[dict[str, float | int]]:
    rows = []
    for idx in range(len(bands) - 1):
        pred = sum(int(stats.band_rows[idx]["pred"]) for stats in image_stats)
        tp = sum(int(stats.band_rows[idx]["tp"]) for stats in image_stats)
        fp = sum(int(stats.band_rows[idx]["fp"]) for stats in image_stats)
        nearest_values = []
        matched_values = []
        for stats in image_stats:
            row = stats.band_rows[idx]
            nearest_weight = int(row["pred"])
            matched_weight = int(row["tp"])
            if nearest_weight > 0 and not np.isnan(float(row["nearest_dist_mean"])):
                nearest_values.extend(
                    [float(row["nearest_dist_mean"])] * nearest_weight
                )
            if matched_weight > 0 and not np.isnan(float(row["matched_dist_mean"])):
                matched_values.extend(
                    [float(row["matched_dist_mean"])] * matched_weight
                )
        rows.append(
            {
                "bin_low": float(bands[idx]),
                "bin_high": float(bands[idx + 1]),
                "pred": pred,
                "tp": tp,
                "fp": fp,
                "precision": float(tp / max(pred, 1)),
                "fp_share": float(
                    fp / max(sum(int(s.pred_count_05) for s in image_stats), 1)
                ),
                "nearest_dist_mean": _finite_mean(
                    np.asarray(nearest_values, dtype=np.float64)
                ),
                "matched_dist_mean": _finite_mean(
                    np.asarray(matched_values, dtype=np.float64)
                ),
            }
        )
    return rows


@torch.no_grad()
def _run_analysis(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
    bands: np.ndarray,
    max_distance: float,
) -> list[ImageBandStats]:
    model.eval()
    all_stats: list[ImageBandStats] = []
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
        all_stats.append(
            _band_stats_for_image(
                image_id=int(target["image_id"].item())
                if "image_id" in target
                else index,
                gt_count=int(target["point"].shape[0]),
                scores=scores,
                pred_points=outputs["pred_points"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                gt_points=target["point"].detach().cpu().numpy().astype(np.float64),
                bands=bands,
                max_distance=max_distance,
            )
        )
        if (index + 1) % 25 == 0:
            logger.info(f"Processed {index + 1}/{len(data_loader.dataset)} images")
    return all_stats


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


def _save_plots(output_dir: Path, band_rows: list[dict[str, float | int]]) -> None:
    labels = [f"{row['bin_low']:.1f}-{row['bin_high']:.1f}" for row in band_rows]
    x = np.arange(len(labels))
    tp = np.asarray([row["tp"] for row in band_rows], dtype=np.float64)
    fp = np.asarray([row["fp"] for row in band_rows], dtype=np.float64)
    precision = np.asarray([row["precision"] for row in band_rows], dtype=np.float64)
    nearest = np.asarray(
        [row["nearest_dist_mean"] for row in band_rows], dtype=np.float64
    )
    matched = np.asarray(
        [row["matched_dist_mean"] for row in band_rows], dtype=np.float64
    )

    plt.figure(figsize=(9, 5))
    plt.bar(x, tp, label="TP")
    plt.bar(x, fp, bottom=tp, label="FP")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_band_fp_fn.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, precision, marker="o")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_by_score_band.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, nearest, marker="o", label="nearest GT distance")
    plt.plot(x, matched, marker="o", label="matched distance")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "distance_by_score_band.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    band_cfg = getattr(cfg, "hard_band", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(
        _cfg_get(band_cfg, "output_dir", f"outputs/hard_score_band/{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="hard_score_band.log")

    band_step = float(_cfg_get(band_cfg, "band_step", 0.1))
    max_distance = float(_cfg_get(band_cfg, "match_distance", 8.0))
    bands = np.arange(0.0, 1.0 + band_step / 2, band_step)

    predict_cfg = OmegaConf.to_container(cfg, resolve=True).get("predict", {})
    weight_path = str(predict_cfg.get("weight_path", "weights/SHTechA.pth"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}; output_dir={output_dir}")

    model = build_model(cfg, training=False).to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
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
    image_stats = _run_analysis(
        model,
        val_loader,
        device,
        use_depth=use_depth,
        bands=bands,
        max_distance=max_distance,
    )
    band_rows = _aggregate_band_rows(image_stats, bands)

    total_gt = sum(stats.gt_count for stats in image_stats)
    total_pred_05 = sum(stats.pred_count_05 for stats in image_stats)
    total_tp_05 = sum(
        sum(int(row["tp"]) for row in stats.band_rows if float(row["bin_low"]) >= 0.5)
        for stats in image_stats
    )
    total_fp_05 = total_pred_05 - total_tp_05
    total_fn_any_score = sum(stats.fn_total for stats in image_stats)
    high_score_fp = sum(
        int(row["fp"]) for row in band_rows if float(row["bin_low"]) >= 0.5
    )
    mid_score_fp = sum(
        int(row["fp"]) for row in band_rows if 0.4 <= float(row["bin_low"]) < 0.8
    )

    summary = {
        "config": {
            "weight_path": weight_path,
            "match_distance": max_distance,
            "band_step": band_step,
            "num_images": len(image_stats),
        },
        "overall": {
            "total_gt": total_gt,
            "total_pred_at_0_5": total_pred_05,
            "tp_at_0_5": total_tp_05,
            "fp_at_0_5": total_fp_05,
            "fn_unmatched_any_score": total_fn_any_score,
            "high_score_fp_ge_0_5": high_score_fp,
            "mid_score_fp_0_4_to_0_8": mid_score_fp,
        },
        "bands": band_rows,
        "per_image": [stats.__dict__ for stats in image_stats],
    }
    with (output_dir / "hard_score_band_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    _save_plots(output_dir, band_rows)

    _print_table(
        "--- Overall ---",
        ["metric", "value"],
        [[key, value] for key, value in summary["overall"].items()],
    )
    _print_table(
        "--- Score Band Diagnostics ---",
        ["band", "pred", "tp", "fp", "precision", "nearest_dist", "matched_dist"],
        [
            [
                f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
                row["pred"],
                row["tp"],
                row["fp"],
                f"{float(row['precision']):.3f}",
                "nan"
                if np.isnan(float(row["nearest_dist_mean"]))
                else f"{float(row['nearest_dist_mean']):.2f}",
                "nan"
                if np.isnan(float(row["matched_dist_mean"]))
                else f"{float(row['matched_dist_mean']):.2f}",
            ]
            for row in band_rows
        ],
    )
    logger.info(f"Saved hard score-band analysis to {output_dir}")


if __name__ == "__main__":
    main()

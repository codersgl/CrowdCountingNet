"""Analyze query-score counting rules for a trained DSGCNet checkpoint.

The script validates whether MAE can be reduced by changing how per-query
classification scores are converted into counts, without retraining:

- global threshold sweep
- K-fold global threshold selection
- K-fold bucket-specific thresholds by raw count / density count
- soft count and temperature-scaled soft count
- score reliability table using greedy point matching

Example:

    uv run python scripts/analyze_score_counting.py \
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
class ScoreRecord:
    image_id: int
    gt_count: int
    density_count: float
    scores: np.ndarray
    margins: np.ndarray
    points: np.ndarray
    gt_points: np.ndarray


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _bias(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred - gt))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _kfold_indices(n_samples: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, folds) if len(fold) > 0]


def _counts_at_threshold(records: list[ScoreRecord], threshold: float) -> np.ndarray:
    return np.asarray(
        [np.sum(record.scores > threshold) for record in records], dtype=np.float64
    )


def _soft_counts(records: list[ScoreRecord]) -> np.ndarray:
    return np.asarray([np.sum(record.scores) for record in records], dtype=np.float64)


def _temperature_counts(records: list[ScoreRecord], temperature: float) -> np.ndarray:
    return np.asarray(
        [np.sum(_sigmoid(record.margins / temperature)) for record in records],
        dtype=np.float64,
    )


def _metric_row(
    method: str, pred: np.ndarray, gt: np.ndarray
) -> dict[str, float | str]:
    return {
        "method": method,
        "mae": _mae(pred, gt),
        "rmse": _rmse(pred, gt),
        "bias": _bias(pred, gt),
    }


def _threshold_sweep(
    records: list[ScoreRecord],
    gt: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
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
    best = min(rows, key=lambda row: row["mae"])
    return rows, best


def _temperature_sweep(
    records: list[ScoreRecord],
    gt: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    for temperature in temperatures:
        raw = _temperature_counts(records, float(temperature))
        bias = float(np.mean(gt - raw))
        pred = raw + bias
        rows.append(
            {
                "temperature": float(temperature),
                "bias_correction": bias,
                "mae": _mae(pred, gt),
                "rmse": _rmse(pred, gt),
                "bias": _bias(pred, gt),
            }
        )
    best = min(rows, key=lambda row: row["mae"])
    return rows, best


def _fit_threshold(
    train_records: list[ScoreRecord], train_gt: np.ndarray, thresholds: np.ndarray
) -> float:
    best_threshold = float(thresholds[0])
    best_mae = float("inf")
    for threshold in thresholds:
        pred = _counts_at_threshold(train_records, float(threshold))
        mae = _mae(pred, train_gt)
        if mae < best_mae:
            best_mae = mae
            best_threshold = float(threshold)
    return best_threshold


def _cv_global_threshold(
    records: list[ScoreRecord],
    gt: np.ndarray,
    thresholds: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    fold_indices = _kfold_indices(len(records), folds=folds, seed=seed)
    all_indices = np.arange(len(records))
    pred = np.zeros_like(gt, dtype=np.float64)
    chosen: list[float] = []
    for val_idx in fold_indices:
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        train_records = [records[int(i)] for i in train_idx]
        val_records = [records[int(i)] for i in val_idx]
        threshold = _fit_threshold(train_records, gt[train_idx], thresholds)
        chosen.append(threshold)
        pred[val_idx] = _counts_at_threshold(val_records, threshold)
    row = _metric_row("cv_global_threshold", pred, gt)
    row["chosen_thresholds"] = chosen
    return row


def _fit_bucket_thresholds(
    train_records: list[ScoreRecord],
    train_gt: np.ndarray,
    signal: np.ndarray,
    thresholds: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(signal, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    selected: list[float] = []
    for bin_idx in range(bins):
        mask = (signal >= edges[bin_idx]) & (signal < edges[bin_idx + 1])
        if not np.any(mask):
            selected.append(_fit_threshold(train_records, train_gt, thresholds))
            continue
        bucket_records = [record for record, use in zip(train_records, mask) if use]
        selected.append(_fit_threshold(bucket_records, train_gt[mask], thresholds))
    return edges, selected


def _apply_bucket_thresholds(
    records: list[ScoreRecord],
    signal: np.ndarray,
    edges: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    bucket_ids = np.clip(
        np.searchsorted(edges[1:-1], signal, side="right"), 0, len(thresholds) - 1
    )
    pred = np.zeros(len(records), dtype=np.float64)
    for idx, record in enumerate(records):
        pred[idx] = np.sum(record.scores > thresholds[int(bucket_ids[idx])])
    return pred


def _cv_bucket_threshold(
    records: list[ScoreRecord],
    gt: np.ndarray,
    signal_name: str,
    thresholds: np.ndarray,
    folds: int,
    seed: int,
    bins: int,
) -> dict[str, Any]:
    signal_all = np.asarray(
        [
            np.sum(record.scores > 0.5)
            if signal_name == "cls"
            else record.density_count
            for record in records
        ],
        dtype=np.float64,
    )
    fold_indices = _kfold_indices(len(records), folds=folds, seed=seed)
    all_indices = np.arange(len(records))
    pred = np.zeros_like(gt, dtype=np.float64)
    fold_params: list[dict[str, Any]] = []
    for val_idx in fold_indices:
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        train_records = [records[int(i)] for i in train_idx]
        val_records = [records[int(i)] for i in val_idx]
        edges, selected = _fit_bucket_thresholds(
            train_records,
            gt[train_idx],
            signal_all[train_idx],
            thresholds,
            bins=bins,
        )
        pred[val_idx] = _apply_bucket_thresholds(
            val_records, signal_all[val_idx], edges, selected
        )
        fold_params.append(
            {
                "edges": [float(v) if np.isfinite(v) else str(v) for v in edges],
                "thresholds": selected,
            }
        )
    row = _metric_row(f"cv_bucket_threshold_by_{signal_name}", pred, gt)
    row["fold_params"] = fold_params
    return row


def _cv_temperature(
    records: list[ScoreRecord],
    gt: np.ndarray,
    temperatures: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    fold_indices = _kfold_indices(len(records), folds=folds, seed=seed)
    all_indices = np.arange(len(records))
    pred = np.zeros_like(gt, dtype=np.float64)
    params: list[dict[str, float]] = []
    for val_idx in fold_indices:
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        train_records = [records[int(i)] for i in train_idx]
        val_records = [records[int(i)] for i in val_idx]
        best_temp = float(temperatures[0])
        best_bias = 0.0
        best_mae = float("inf")
        for temperature in temperatures:
            raw = _temperature_counts(train_records, float(temperature))
            bias = float(np.mean(gt[train_idx] - raw))
            mae = _mae(raw + bias, gt[train_idx])
            if mae < best_mae:
                best_mae = mae
                best_temp = float(temperature)
                best_bias = bias
        pred[val_idx] = _temperature_counts(val_records, best_temp) + best_bias
        params.append({"temperature": best_temp, "bias": best_bias})
    row = _metric_row("cv_temperature_soft_count", pred, gt)
    row["fold_params"] = params
    return row


def _greedy_tp_flags(
    scores: np.ndarray,
    points: np.ndarray,
    gt_points: np.ndarray,
    max_distance: float,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(scores) == 0:
        return scores, np.zeros(0, dtype=bool)
    if len(scores) > max_candidates:
        candidate_idx = np.argsort(scores)[-max_candidates:]
    else:
        candidate_idx = np.arange(len(scores))
    candidate_scores = scores[candidate_idx]
    candidate_points = points[candidate_idx]
    flags = np.zeros(len(candidate_scores), dtype=bool)
    if len(gt_points) == 0 or len(candidate_points) == 0:
        return candidate_scores, flags

    dist = np.linalg.norm(candidate_points[:, None, :] - gt_points[None, :, :], axis=2)
    order = np.argsort(candidate_scores)[::-1]
    matched_gt: set[int] = set()
    for pred_idx in order:
        nearest_order = np.argsort(dist[pred_idx])
        for gt_idx in nearest_order:
            if int(gt_idx) in matched_gt:
                continue
            if dist[pred_idx, gt_idx] <= max_distance:
                flags[pred_idx] = True
                matched_gt.add(int(gt_idx))
            break
    return candidate_scores, flags


def _reliability_table(
    records: list[ScoreRecord],
    bins: int,
    max_distance: float,
    max_candidates: int,
) -> list[dict[str, float]]:
    all_scores: list[np.ndarray] = []
    all_flags: list[np.ndarray] = []
    for record in records:
        scores, flags = _greedy_tp_flags(
            record.scores,
            record.points,
            record.gt_points,
            max_distance=max_distance,
            max_candidates=max_candidates,
        )
        all_scores.append(scores)
        all_flags.append(flags)
    scores_arr = np.concatenate(all_scores) if all_scores else np.zeros(0)
    flags_arr = np.concatenate(all_flags) if all_flags else np.zeros(0, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx in range(bins):
        low, high = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (scores_arr >= low) & (scores_arr <= high)
        else:
            mask = (scores_arr >= low) & (scores_arr < high)
        count = int(mask.sum())
        avg_score = float(np.mean(scores_arr[mask])) if count else float("nan")
        precision = float(np.mean(flags_arr[mask])) if count else float("nan")
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "avg_score": avg_score,
                "empirical_precision": precision,
            }
        )
    return rows


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
) -> list[ScoreRecord]:
    model.eval()
    records: list[ScoreRecord] = []
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
        probs = torch.softmax(logits, dim=-1)[:, 1]
        margins = logits[:, 1] - logits[:, 0]
        target = targets[0]
        records.append(
            ScoreRecord(
                image_id=int(target["image_id"].item())
                if "image_id" in target
                else index,
                gt_count=int(target["point"].shape[0]),
                density_count=float(outputs["density_out"].sum().detach().cpu().item()),
                scores=probs.numpy().astype(np.float64),
                margins=margins.numpy().astype(np.float64),
                points=outputs["pred_points"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                gt_points=target["point"].detach().cpu().numpy().astype(np.float64),
            )
        )
        if (index + 1) % 25 == 0:
            logger.info(
                f"Collected scores for {index + 1}/{len(data_loader.dataset)} images"
            )
    return records


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
    temperature_rows: list[dict[str, float]],
    reliability_rows: list[dict[str, float]],
    method_rows: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        [r["threshold"] for r in threshold_rows], [r["mae"] for r in threshold_rows]
    )
    best = min(threshold_rows, key=lambda row: row["mae"])
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="default 0.5")
    plt.axvline(
        best["threshold"], color="red", linestyle="--", linewidth=1, label="best"
    )
    plt.xlabel("Threshold")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        [r["temperature"] for r in temperature_rows],
        [r["mae"] for r in temperature_rows],
    )
    plt.xlabel("Temperature")
    plt.ylabel("MAE after bias correction")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_soft_count.png", dpi=160)
    plt.close()

    valid = [r for r in reliability_rows if r["count"] > 0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.plot(
        [r["avg_score"] for r in valid],
        [r["empirical_precision"] for r in valid],
        marker="o",
    )
    plt.xlabel("Average score")
    plt.ylabel("Empirical precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram.png", dpi=160)
    plt.close()

    sorted_methods = sorted(method_rows, key=lambda row: float(row["mae"]))
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(sorted_methods)), [row["mae"] for row in sorted_methods])
    plt.xticks(
        np.arange(len(sorted_methods)),
        [str(row["method"]) for row in sorted_methods],
        rotation=30,
        ha="right",
    )
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "method_mae_bar.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    score_cfg = getattr(cfg, "score_diag", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(
        _cfg_get(score_cfg, "output_dir", f"outputs/score_counting/{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="score_counting.log")

    threshold_min = float(_cfg_get(score_cfg, "threshold_min", 0.1))
    threshold_max = float(_cfg_get(score_cfg, "threshold_max", 0.95))
    threshold_step = float(_cfg_get(score_cfg, "threshold_step", 0.01))
    temp_min = float(_cfg_get(score_cfg, "temp_min", 0.5))
    temp_max = float(_cfg_get(score_cfg, "temp_max", 4.0))
    temp_step = float(_cfg_get(score_cfg, "temp_step", 0.05))
    folds = int(_cfg_get(score_cfg, "folds", 5))
    seed = int(_cfg_get(score_cfg, "seed", 42))
    bins = int(_cfg_get(score_cfg, "bins", 4))
    reliability_bins = int(_cfg_get(score_cfg, "reliability_bins", 10))
    match_distance = float(_cfg_get(score_cfg, "match_distance", 8.0))
    max_reliability_candidates = int(
        _cfg_get(score_cfg, "max_reliability_candidates", 4096)
    )

    thresholds = np.arange(
        threshold_min, threshold_max + threshold_step / 2, threshold_step
    )
    temperatures = np.arange(temp_min, temp_max + temp_step / 2, temp_step)

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
    records = _collect_records(model, val_loader, device, use_depth=use_depth)
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)

    threshold_rows, best_threshold = _threshold_sweep(records, gt, thresholds)
    temperature_rows, best_temperature = _temperature_sweep(records, gt, temperatures)
    raw_cls = _counts_at_threshold(records, 0.5)
    soft = _soft_counts(records)
    soft_bias = float(np.mean(gt - soft))
    method_rows: list[dict[str, Any]] = [
        _metric_row("hard_threshold_0.5", raw_cls, gt),
        _metric_row("soft_sum_scores", soft, gt),
        _metric_row("soft_sum_scores_bias", soft + soft_bias, gt),
        _metric_row(
            f"best_threshold_full_{best_threshold['threshold']:.2f}",
            _counts_at_threshold(records, float(best_threshold["threshold"])),
            gt,
        ),
        _metric_row(
            f"best_temperature_full_{best_temperature['temperature']:.2f}",
            _temperature_counts(records, float(best_temperature["temperature"]))
            + float(best_temperature["bias_correction"]),
            gt,
        ),
        _cv_global_threshold(records, gt, thresholds, folds=folds, seed=seed),
        _cv_bucket_threshold(
            records, gt, "cls", thresholds, folds=folds, seed=seed, bins=bins
        ),
        _cv_bucket_threshold(
            records, gt, "density", thresholds, folds=folds, seed=seed, bins=bins
        ),
        _cv_temperature(records, gt, temperatures, folds=folds, seed=seed),
    ]
    reliability_rows = _reliability_table(
        records,
        bins=reliability_bins,
        max_distance=match_distance,
        max_candidates=max_reliability_candidates,
    )

    sorted_methods = sorted(method_rows, key=lambda row: float(row["mae"]))
    _print_table(
        "--- Score Counting Methods ---",
        ["method", "mae", "rmse", "bias", "gain_vs_0.5"],
        [
            [
                row["method"],
                f"{float(row['mae']):.3f}",
                f"{float(row['rmse']):.3f}",
                f"{float(row['bias']):+.3f}",
                f"{float(method_rows[0]['mae']) - float(row['mae']):+.3f}",
            ]
            for row in sorted_methods
        ],
    )
    _print_table(
        "--- Reliability Table ---",
        ["score_bin", "count", "avg_score", "emp_precision"],
        [
            [
                f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
                row["count"],
                "nan" if np.isnan(row["avg_score"]) else f"{row['avg_score']:.3f}",
                "nan"
                if np.isnan(row["empirical_precision"])
                else f"{row['empirical_precision']:.3f}",
            ]
            for row in reliability_rows
        ],
    )

    summary = {
        "config": {
            "weight_path": weight_path,
            "num_images": len(records),
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "folds": folds,
            "seed": seed,
            "bins": bins,
            "match_distance": match_distance,
        },
        "methods": method_rows,
        "threshold_sweep": threshold_rows,
        "best_threshold_full": best_threshold,
        "temperature_sweep": temperature_rows,
        "best_temperature_full": best_temperature,
        "reliability": reliability_rows,
    }
    with (output_dir / "score_counting_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)
    _save_plots(
        output_dir, threshold_rows, temperature_rows, reliability_rows, method_rows
    )
    logger.info(f"Saved score-counting analysis to {output_dir}")


if __name__ == "__main__":
    main()

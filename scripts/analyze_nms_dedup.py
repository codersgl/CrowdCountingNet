"""Analyze duplicate predictions and inference-time NMS for DSGCNet.

The script tests whether high-score false positives are mostly local duplicates.
It runs the validation set once, then sweeps distance-based greedy NMS radii and
reports count MAE before and after suppression.

Example:

    uv run python scripts/analyze_nms_dedup.py \
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
from crowdcount.models.checkpoint import load_model_state_dict
from crowdcount.utils.logging import logger, setup_logger


@dataclass
class NMSRecord:
    image_id: int
    gt_count: int
    scores: np.ndarray
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


def _parse_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _greedy_nms(
    scores: np.ndarray,
    points: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.zeros(len(scores), dtype=bool)
    suppressed = np.zeros(len(scores), dtype=bool)
    suppressor = np.full(len(scores), -1, dtype=np.int64)
    if len(scores) == 0:
        return keep, suppressed, suppressor

    order = np.argsort(scores)[::-1]
    for pred_idx in order:
        pred_idx_int = int(pred_idx)
        if suppressed[pred_idx_int]:
            continue
        keep[pred_idx_int] = True
        if radius <= 0.0:
            continue
        distances = np.linalg.norm(points - points[pred_idx_int], axis=1)
        duplicate_mask = (distances <= radius) & (~keep) & (~suppressed)
        duplicate_mask[pred_idx_int] = False
        suppressed[duplicate_mask] = True
        suppressor[duplicate_mask] = pred_idx_int
    return keep, suppressed, suppressor


def _match_predictions(
    scores: np.ndarray,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, int]:
    matched_pred = np.zeros(len(scores), dtype=bool)
    if len(scores) == 0:
        return matched_pred, int(len(gt_points))
    if len(gt_points) == 0:
        return matched_pred, 0

    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    order = np.argsort(scores)[::-1]
    matched_gt: set[int] = set()
    for pred_idx in order:
        pred_idx_int = int(pred_idx)
        for gt_idx in np.argsort(distances[pred_idx_int]):
            gt_idx_int = int(gt_idx)
            if gt_idx_int in matched_gt:
                continue
            if distances[pred_idx_int, gt_idx_int] <= max_distance:
                matched_pred[pred_idx_int] = True
                matched_gt.add(gt_idx_int)
            break
    return matched_pred, int(len(gt_points) - len(matched_gt))


def _score_band_rows(
    scores: np.ndarray,
    matched_before: np.ndarray,
    suppressed: np.ndarray,
    bands: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for band_idx in range(len(bands) - 1):
        low = float(bands[band_idx])
        high = float(bands[band_idx + 1])
        if band_idx == len(bands) - 2:
            band_mask = (scores >= low) & (scores <= high)
        else:
            band_mask = (scores >= low) & (scores < high)
        pred = int(band_mask.sum())
        dup = int((band_mask & suppressed).sum())
        dup_tp = int((band_mask & suppressed & matched_before).sum())
        dup_fp = int((band_mask & suppressed & (~matched_before)).sum())
        fp = int((band_mask & (~matched_before)).sum())
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "pred": pred,
                "fp": fp,
                "suppressed": dup,
                "suppressed_tp": dup_tp,
                "suppressed_fp": dup_fp,
                "suppressed_rate": float(dup / max(pred, 1)),
                "fp_suppressed_rate": float(dup_fp / max(fp, 1)),
            }
        )
    return rows


def _merge_band_rows(
    per_image_rows: list[list[dict[str, float | int]]], bands: np.ndarray
) -> list[dict[str, float | int]]:
    merged: list[dict[str, float | int]] = []
    for band_idx in range(len(bands) - 1):
        pred = sum(int(rows[band_idx]["pred"]) for rows in per_image_rows)
        fp = sum(int(rows[band_idx]["fp"]) for rows in per_image_rows)
        suppressed = sum(int(rows[band_idx]["suppressed"]) for rows in per_image_rows)
        suppressed_tp = sum(
            int(rows[band_idx]["suppressed_tp"]) for rows in per_image_rows
        )
        suppressed_fp = sum(
            int(rows[band_idx]["suppressed_fp"]) for rows in per_image_rows
        )
        merged.append(
            {
                "bin_low": float(bands[band_idx]),
                "bin_high": float(bands[band_idx + 1]),
                "pred": pred,
                "fp": fp,
                "suppressed": suppressed,
                "suppressed_tp": suppressed_tp,
                "suppressed_fp": suppressed_fp,
                "suppressed_rate": float(suppressed / max(pred, 1)),
                "fp_suppressed_rate": float(suppressed_fp / max(fp, 1)),
            }
        )
    return merged


def _analyze_radius(
    records: list[NMSRecord],
    threshold: float,
    radius: float,
    match_distance: float,
    bands: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, float | int]]]:
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    raw_counts = np.zeros(len(records), dtype=np.float64)
    nms_counts = np.zeros(len(records), dtype=np.float64)
    per_image: list[dict[str, Any]] = []
    per_image_bands: list[list[dict[str, float | int]]] = []

    totals = {
        "tp_before": 0,
        "fp_before": 0,
        "fn_before": 0,
        "tp_after": 0,
        "fp_after": 0,
        "fn_after": 0,
        "suppressed": 0,
        "suppressed_tp": 0,
        "suppressed_fp": 0,
    }
    for index, record in enumerate(records):
        candidate_mask = record.scores > threshold
        scores = record.scores[candidate_mask]
        points = record.points[candidate_mask]
        raw_counts[index] = len(scores)

        matched_before, fn_before = _match_predictions(
            scores, points, record.gt_points, max_distance=match_distance
        )
        keep, suppressed, _ = _greedy_nms(scores, points, radius=radius)
        matched_after, fn_after = _match_predictions(
            scores[keep], points[keep], record.gt_points, max_distance=match_distance
        )

        tp_before = int(matched_before.sum())
        fp_before = int(len(scores) - tp_before)
        tp_after = int(matched_after.sum())
        fp_after = int(keep.sum() - tp_after)
        suppressed_tp = int((suppressed & matched_before).sum())
        suppressed_fp = int((suppressed & (~matched_before)).sum())
        nms_counts[index] = int(keep.sum())

        totals["tp_before"] += tp_before
        totals["fp_before"] += fp_before
        totals["fn_before"] += fn_before
        totals["tp_after"] += tp_after
        totals["fp_after"] += fp_after
        totals["fn_after"] += fn_after
        totals["suppressed"] += int(suppressed.sum())
        totals["suppressed_tp"] += suppressed_tp
        totals["suppressed_fp"] += suppressed_fp

        band_rows = _score_band_rows(scores, matched_before, suppressed, bands)
        per_image_bands.append(band_rows)
        per_image.append(
            {
                "image_id": record.image_id,
                "gt_count": record.gt_count,
                "raw_count": int(raw_counts[index]),
                "nms_count": int(nms_counts[index]),
                "tp_before": tp_before,
                "fp_before": fp_before,
                "fn_before": fn_before,
                "tp_after": tp_after,
                "fp_after": fp_after,
                "fn_after": fn_after,
                "suppressed": int(suppressed.sum()),
                "suppressed_tp": suppressed_tp,
                "suppressed_fp": suppressed_fp,
            }
        )

    row: dict[str, Any] = {
        "radius": radius,
        "threshold": threshold,
        "raw_mae": _mae(raw_counts, gt),
        "raw_rmse": _rmse(raw_counts, gt),
        "raw_bias": _bias(raw_counts, gt),
        "nms_mae": _mae(nms_counts, gt),
        "nms_rmse": _rmse(nms_counts, gt),
        "nms_bias": _bias(nms_counts, gt),
        "mae_gain": _mae(raw_counts, gt) - _mae(nms_counts, gt),
        "count_removed": int(np.sum(raw_counts - nms_counts)),
        "avg_removed_per_image": float(np.mean(raw_counts - nms_counts)),
        "tp_before": totals["tp_before"],
        "fp_before": totals["fp_before"],
        "fn_before": totals["fn_before"],
        "tp_after": totals["tp_after"],
        "fp_after": totals["fp_after"],
        "fn_after": totals["fn_after"],
        "suppressed": totals["suppressed"],
        "suppressed_tp": totals["suppressed_tp"],
        "suppressed_fp": totals["suppressed_fp"],
        "suppressed_fp_share_of_fp": float(
            totals["suppressed_fp"] / max(totals["fp_before"], 1)
        ),
        "suppressed_tp_share_of_tp": float(
            totals["suppressed_tp"] / max(totals["tp_before"], 1)
        ),
    }
    return row, per_image, _merge_band_rows(per_image_bands, bands)


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
) -> list[NMSRecord]:
    model.eval()
    records: list[NMSRecord] = []
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
        scores = torch.softmax(logits, dim=-1)[:, 1]
        target = targets[0]
        records.append(
            NMSRecord(
                image_id=int(target["image_id"].item())
                if "image_id" in target
                else index,
                gt_count=int(target["point"].shape[0]),
                scores=scores.numpy().astype(np.float64),
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
                f"Collected predictions for {index + 1}/{len(data_loader.dataset)} images"
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
    radius_rows: list[dict[str, Any]],
    best_band_rows: list[dict[str, float | int]],
) -> None:
    radii = [float(row["radius"]) for row in radius_rows]
    raw_mae = [float(row["raw_mae"]) for row in radius_rows]
    nms_mae = [float(row["nms_mae"]) for row in radius_rows]
    removed = [int(row["count_removed"]) for row in radius_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(radii, raw_mae, marker="o", label="raw")
    plt.plot(radii, nms_mae, marker="o", label="after NMS")
    plt.xlabel("NMS radius (px)")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "nms_mae_by_radius.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(len(radii)), removed)
    plt.xticks(np.arange(len(radii)), [f"{radius:g}" for radius in radii])
    plt.xlabel("NMS radius (px)")
    plt.ylabel("Suppressed predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "suppressed_by_radius.png", dpi=160)
    plt.close()

    labels = [f"{row['bin_low']:.1f}-{row['bin_high']:.1f}" for row in best_band_rows]
    x = np.arange(len(labels))
    suppressed_tp = np.asarray(
        [row["suppressed_tp"] for row in best_band_rows], dtype=np.float64
    )
    suppressed_fp = np.asarray(
        [row["suppressed_fp"] for row in best_band_rows], dtype=np.float64
    )
    plt.figure(figsize=(9, 5))
    plt.bar(x, suppressed_tp, label="suppressed TP")
    plt.bar(x, suppressed_fp, bottom=suppressed_tp, label="suppressed FP")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Suppressed predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "suppressed_by_score_band.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    nms_cfg = getattr(cfg, "nms_diag", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(_cfg_get(nms_cfg, "output_dir", f"outputs/nms_dedup/{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="nms_dedup.log")

    threshold = float(_cfg_get(nms_cfg, "threshold", 0.5))
    match_distance = float(_cfg_get(nms_cfg, "match_distance", 8.0))
    radii = _parse_float_list(_cfg_get(nms_cfg, "radii", None), [4.0, 6.0, 8.0, 10.0])
    band_step = float(_cfg_get(nms_cfg, "band_step", 0.1))
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
    load_model_state_dict(model, checkpoint, logger=logger)
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

    radius_rows: list[dict[str, Any]] = []
    per_radius: dict[str, Any] = {}
    for radius in radii:
        row, per_image, band_rows = _analyze_radius(
            records,
            threshold=threshold,
            radius=float(radius),
            match_distance=match_distance,
            bands=bands,
        )
        radius_rows.append(row)
        per_radius[f"{float(radius):g}"] = {
            "per_image": per_image,
            "bands": band_rows,
        }

    best_row = min(radius_rows, key=lambda row: float(row["nms_mae"]))
    best_radius_key = f"{float(best_row['radius']):g}"
    best_band_rows = per_radius[best_radius_key]["bands"]

    summary = {
        "config": {
            "weight_path": weight_path,
            "num_images": len(records),
            "threshold": threshold,
            "match_distance": match_distance,
            "radii": radii,
            "band_step": band_step,
        },
        "radii": radius_rows,
        "best_radius": best_row,
        "per_radius": per_radius,
    }
    with (output_dir / "nms_dedup_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    _save_plots(output_dir, radius_rows, best_band_rows)

    _print_table(
        "--- NMS Radius Sweep ---",
        [
            "radius",
            "raw_mae",
            "nms_mae",
            "gain",
            "removed",
            "supp_fp/fp",
            "supp_tp/tp",
            "bias",
        ],
        [
            [
                f"{float(row['radius']):g}",
                f"{float(row['raw_mae']):.3f}",
                f"{float(row['nms_mae']):.3f}",
                f"{float(row['mae_gain']):+.3f}",
                row["count_removed"],
                f"{float(row['suppressed_fp_share_of_fp']):.3f}",
                f"{float(row['suppressed_tp_share_of_tp']):.3f}",
                f"{float(row['nms_bias']):+.3f}",
            ]
            for row in radius_rows
        ],
    )
    _print_table(
        f"--- Score Band Suppression at Best Radius {best_radius_key}px ---",
        ["band", "pred", "fp", "supp", "supp_fp", "supp_tp", "fp_supp_rate"],
        [
            [
                f"{row['bin_low']:.1f}-{row['bin_high']:.1f}",
                row["pred"],
                row["fp"],
                row["suppressed"],
                row["suppressed_fp"],
                row["suppressed_tp"],
                f"{float(row['fp_suppressed_rate']):.3f}",
            ]
            for row in best_band_rows
        ],
    )
    logger.info(f"Saved NMS duplicate analysis to {output_dir}")


if __name__ == "__main__":
    main()

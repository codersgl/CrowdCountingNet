"""Diagnose which part of a trained DSGCNet baseline has room to improve.

The script runs one validation forward pass and reports:

- classification-head count MAE vs density-head integral MAE
- MAE by ground-truth count buckets
- TP/FP/FN style point-detection diagnostics
- density-map quality metrics and their correlation with count errors

Example:

    uv run python scripts/diag_baseline.py \
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
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.data.prepare import gaussian_filter_density
from crowdcount.engine import _forward_model
from crowdcount.models import build_model
from crowdcount.models.ssim_loss import SSIMLoss
from crowdcount.utils.logging import logger, setup_logger


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    unique_values, inverse, counts = np.unique(
        arr, return_inverse=True, return_counts=True
    )
    del unique_values
    for group_idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == group_idx
            ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return _pearson(_rankdata(x).tolist(), _rankdata(y).tolist())


def _resize_count_preserving(
    density: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize [1, 1, H, W] density while preserving its integral."""
    src_h, src_w = density.shape[-2:]
    dst_h, dst_w = size
    resized = F.interpolate(density, size=size, mode="bilinear", align_corners=False)
    return resized * ((src_h * src_w) / max(dst_h * dst_w, 1))


def _density_quality(
    pred_density: torch.Tensor,
    points: torch.Tensor,
    image_size: tuple[int, int],
    ssim_loss: SSIMLoss,
) -> dict[str, float]:
    """Compare predicted density with an on-the-fly Gaussian GT map."""
    image_h, image_w = image_size
    pred = pred_density.detach().float().cpu()
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)

    gt_points = points.detach().cpu().numpy().astype(np.float32)
    gt_full = gaussian_filter_density(
        np.zeros((image_h, image_w), dtype=np.float32), gt_points
    )
    gt = torch.from_numpy(gt_full).unsqueeze(0).unsqueeze(0).float()
    gt = _resize_count_preserving(gt, pred.shape[-2:])

    pred_sum = pred.sum().clamp(min=1e-12)
    pred_norm = pred / pred_sum * max(float(len(gt_points)), 1.0)

    l1 = float(F.l1_loss(pred_norm, gt).item())
    mse = float(F.mse_loss(pred_norm, gt).item())
    peak = float(torch.max(gt.max(), pred_norm.max()).clamp(min=1e-6).item())
    psnr = float(20.0 * math.log10(peak) - 10.0 * math.log10(max(mse, 1e-12)))
    ssim = 1.0 - float(ssim_loss(pred_norm, gt).item())
    return {
        "density_l1": l1,
        "density_mse": mse,
        "density_psnr": psnr,
        "density_ssim": ssim,
    }


def _nearest_match_counts(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    max_dist: float,
) -> tuple[int, int, int]:
    if len(gt_points) == 0:
        return 0, int(len(pred_points)), 0
    if len(pred_points) == 0:
        return 0, 0, int(len(gt_points))

    distances = torch.cdist(
        pred_points.float().cpu(), gt_points.float().cpu(), p=2
    ).numpy()
    pred_idx, gt_idx = linear_sum_assignment(distances)
    matched = distances[pred_idx, gt_idx] <= max_dist
    tp = int(np.sum(matched))
    fp = int(len(pred_points) - tp)
    fn = int(len(gt_points) - tp)
    return tp, fp, fn


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
    threshold: float,
    match_distance: float,
    thresholds: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, int]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    pr_acc = {float(t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    ssim_loss = SSIMLoss(window_size=7, data_range=None).cpu()

    for index, batch in enumerate(data_loader):
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None

        samples = samples.to(device)
        outputs = _forward_model(model, samples, depth_map=depth_map)

        scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1].detach().cpu()
        pred_points = outputs["pred_points"][0].detach().cpu()
        pred_density = outputs["density_out"][0].detach().cpu()
        gt_points = targets[0]["point"].detach().cpu()
        gt_count = int(gt_points.shape[0])
        cls_count = int((scores > threshold).sum().item())
        density_count = float(outputs["density_out"].sum().detach().cpu().item())

        image_h, image_w = int(samples.shape[-2]), int(samples.shape[-1])
        quality = _density_quality(
            pred_density, gt_points, (image_h, image_w), ssim_loss
        )

        tp, fp, fn = _nearest_match_counts(
            pred_points[scores > threshold], gt_points, max_dist=match_distance
        )
        for candidate in thresholds:
            cand_tp, cand_fp, cand_fn = _nearest_match_counts(
                pred_points[scores > float(candidate)],
                gt_points,
                max_dist=match_distance,
            )
            pr_acc[float(candidate)]["tp"] += cand_tp
            pr_acc[float(candidate)]["fp"] += cand_fp
            pr_acc[float(candidate)]["fn"] += cand_fn

        records.append(
            {
                "index": index,
                "image_id": int(targets[0]["image_id"].item())
                if "image_id" in targets[0]
                else index,
                "gt_count": gt_count,
                "cls_count": cls_count,
                "density_count": density_count,
                "cls_abs_error": abs(cls_count - gt_count),
                "density_abs_error": abs(density_count - gt_count),
                "oracle_abs_error": min(
                    abs(cls_count - gt_count), abs(density_count - gt_count)
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **quality,
            }
        )

        if (index + 1) % 25 == 0:
            logger.info(
                f"Processed {index + 1}/{len(data_loader.dataset)} validation images"
            )

    return records, pr_acc


def _bucket_summary(records: list[dict[str, Any]], n_bins: int) -> list[dict[str, Any]]:
    if not records:
        return []
    gt_counts = np.asarray([r["gt_count"] for r in records], dtype=np.float64)
    quantiles = np.quantile(gt_counts, np.linspace(0, 1, n_bins + 1))
    summaries: list[dict[str, Any]] = []
    for bin_idx in range(n_bins):
        low = quantiles[bin_idx]
        high = quantiles[bin_idx + 1]
        if bin_idx == n_bins - 1:
            subset = [r for r in records if low <= r["gt_count"] <= high]
        else:
            subset = [r for r in records if low <= r["gt_count"] < high]
        if not subset:
            continue
        mean_gt = _mean([float(r["gt_count"]) for r in subset])
        summaries.append(
            {
                "bucket": bin_idx + 1,
                "count_range": [float(low), float(high)],
                "n": len(subset),
                "mean_gt": mean_gt,
                "cls_mae": _mean([float(r["cls_abs_error"]) for r in subset]),
                "density_mae": _mean([float(r["density_abs_error"]) for r in subset]),
                "cls_relative_error": _mean(
                    [
                        float(r["cls_abs_error"]) / max(float(r["gt_count"]), 1.0)
                        for r in subset
                    ]
                ),
                "density_relative_error": _mean(
                    [
                        float(r["density_abs_error"]) / max(float(r["gt_count"]), 1.0)
                        for r in subset
                    ]
                ),
            }
        )
    return summaries


def _pr_summary(pr_acc: dict[float, dict[str, int]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold, counts in sorted(pr_acc.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": float(threshold),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def _log_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    logger.info(title)
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    logger.info(fmt.format(*headers))
    logger.info(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        logger.info(fmt.format(*row))


def _save_plots(
    records: list[dict[str, Any]],
    buckets: list[dict[str, Any]],
    pr_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    gt = [r["gt_count"] for r in records]
    cls = [r["cls_count"] for r in records]
    den = [r["density_count"] for r in records]

    plt.figure(figsize=(7, 6))
    plt.scatter(gt, cls, s=18, alpha=0.75, label="classification head")
    plt.scatter(gt, den, s=18, alpha=0.75, label="density head")
    max_count = max(gt + cls + [int(round(v)) for v in den] + [1])
    plt.plot([0, max_count], [0, max_count], "k--", linewidth=1, label="ideal")
    plt.xlabel("GT count")
    plt.ylabel("Predicted count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "head_scatter.png", dpi=160)
    plt.close()

    labels = [
        f"B{b['bucket']}\n{b['count_range'][0]:.0f}-{b['count_range'][1]:.0f}"
        for b in buckets
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [b["cls_mae"] for b in buckets], width, label="cls MAE")
    plt.bar(
        x + width / 2, [b["density_mae"] for b in buckets], width, label="density MAE"
    )
    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bin_mae_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(
        [r["recall"] for r in pr_rows], [r["precision"] for r in pr_rows], marker="."
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()

    density_errors = [r["density_abs_error"] for r in records]
    ssim_values = [r["density_ssim"] for r in records]
    plt.figure(figsize=(6, 5))
    plt.scatter(ssim_values, density_errors, s=18, alpha=0.75)
    if len(records) > 1:
        coeff = np.polyfit(ssim_values, density_errors, deg=1)
        x_line = np.linspace(min(ssim_values), max(ssim_values), 100)
        plt.plot(x_line, coeff[0] * x_line + coeff[1], "r--", linewidth=1)
    plt.xlabel("Density SSIM")
    plt.ylabel("Density count absolute error")
    plt.tight_layout()
    plt.savefig(output_dir / "density_quality_corr.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    diag_cfg = getattr(cfg, "diag", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(
        _cfg_get(diag_cfg, "output_dir", f"outputs/diag_baseline/{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="diag_baseline.log")

    predict_cfg = OmegaConf.to_container(cfg, resolve=True).get("predict", {})
    weight_path = str(predict_cfg.get("weight_path", "weights/SHTechA.pth"))
    threshold = float(
        _cfg_get(diag_cfg, "threshold", getattr(cfg.eval_counting, "threshold", 0.5))
    )
    match_distance = float(_cfg_get(diag_cfg, "match_distance", 8.0))
    n_bins = int(_cfg_get(diag_cfg, "bins", 4))
    t_min = float(_cfg_get(diag_cfg, "t_min", 0.1))
    t_max = float(_cfg_get(diag_cfg, "t_max", 0.95))
    t_step = float(_cfg_get(diag_cfg, "t_step", 0.05))
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)

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

    logger.info(f"Running diagnostics on {len(val_set)} validation images")
    records, pr_acc = _collect_records(
        model,
        val_loader,
        device,
        use_depth=use_depth,
        threshold=threshold,
        match_distance=match_distance,
        thresholds=thresholds,
    )

    cls_mae = _mean([float(r["cls_abs_error"]) for r in records])
    density_mae = _mean([float(r["density_abs_error"]) for r in records])
    oracle_mae = _mean([float(r["oracle_abs_error"]) for r in records])
    cls_bias = _mean([float(r["cls_count"] - r["gt_count"]) for r in records])
    density_bias = _mean([float(r["density_count"] - r["gt_count"]) for r in records])
    head_corr = _pearson(
        [float(r["cls_count"]) for r in records],
        [float(r["density_count"]) for r in records],
    )

    tp = sum(int(r["tp"]) for r in records)
    fp = sum(int(r["fp"]) for r in records)
    fn = sum(int(r["fn"]) for r in records)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    match_lower_bound = _mean([float(abs(r["fp"] - r["fn"])) for r in records])

    count_error = [float(r["density_abs_error"]) for r in records]
    quality_summary = {
        "density_l1_mean": _mean([float(r["density_l1"]) for r in records]),
        "density_psnr_mean": _mean([float(r["density_psnr"]) for r in records]),
        "density_ssim_mean": _mean([float(r["density_ssim"]) for r in records]),
        "ssim_vs_density_count_error_pearson": _pearson(
            [float(r["density_ssim"]) for r in records], count_error
        ),
        "l1_vs_density_count_error_pearson": _pearson(
            [float(r["density_l1"]) for r in records], count_error
        ),
        "ssim_vs_density_count_error_spearman": _spearman(
            [float(r["density_ssim"]) for r in records], count_error
        ),
    }
    buckets = _bucket_summary(records, n_bins=n_bins)
    pr_rows = _pr_summary(pr_acc)

    _log_table(
        "--- Head Decomposition ---",
        ["metric", "value"],
        [
            ["classification_mae", f"{cls_mae:.3f}"],
            ["density_integral_mae", f"{density_mae:.3f}"],
            ["per_image_oracle_mae", f"{oracle_mae:.3f}"],
            ["cls_minus_oracle_gap", f"{cls_mae - oracle_mae:.3f}"],
            ["classification_bias", f"{cls_bias:.3f}"],
            ["density_bias", f"{density_bias:.3f}"],
            ["cls_density_count_corr", f"{head_corr:.3f}"],
        ],
    )
    _log_table(
        "--- GT Count Buckets ---",
        ["bucket", "range", "n", "mean_gt", "cls_mae", "den_mae", "cls_rel", "den_rel"],
        [
            [
                b["bucket"],
                f"{b['count_range'][0]:.0f}-{b['count_range'][1]:.0f}",
                b["n"],
                f"{b['mean_gt']:.1f}",
                f"{b['cls_mae']:.2f}",
                f"{b['density_mae']:.2f}",
                f"{b['cls_relative_error']:.3f}",
                f"{b['density_relative_error']:.3f}",
            ]
            for b in buckets
        ],
    )
    _log_table(
        "--- Point Detection Diagnostics ---",
        [
            "threshold",
            "match_px",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mae_lb",
        ],
        [
            [
                f"{threshold:.2f}",
                f"{match_distance:.1f}",
                tp,
                fp,
                fn,
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{f1:.3f}",
                f"{match_lower_bound:.3f}",
            ]
        ],
    )
    _log_table(
        "--- Density Quality ---",
        ["metric", "value"],
        [[k, f"{v:.4f}"] for k, v in quality_summary.items()],
    )

    summary = {
        "config": {
            "weight_path": weight_path,
            "threshold": threshold,
            "match_distance": match_distance,
            "num_images": len(records),
        },
        "head_decomposition": {
            "classification_mae": cls_mae,
            "density_integral_mae": density_mae,
            "per_image_oracle_mae": oracle_mae,
            "cls_minus_oracle_gap": cls_mae - oracle_mae,
            "classification_bias": cls_bias,
            "density_bias": density_bias,
            "cls_density_count_corr": head_corr,
        },
        "buckets": buckets,
        "point_detection": {
            "threshold": threshold,
            "match_distance": match_distance,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mae_lower_bound_abs_fp_minus_fn": match_lower_bound,
            "pr_curve": pr_rows,
        },
        "density_quality": quality_summary,
        "per_image": records,
    }
    with open(output_dir / "diag_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    _save_plots(records, buckets, pr_rows, output_dir)
    logger.info(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

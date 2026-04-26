"""Offline proxy sweep for matcher point-distance cost.

This script does not estimate retrained MAE. Instead, it answers a narrower
question: if the current checkpoint's predictions were matched with different
``set_cost_point`` values, how would the training supervision assignment change?

Example:

    uv run python scripts/analyze_matcher_cost_proxy.py \
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
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.engine import _forward_model
from crowdcount.models import build_model
from crowdcount.utils.logging import logger, setup_logger


@dataclass
class PredictionRecord:
    image_id: int
    gt_count: int
    scores: np.ndarray
    points: np.ndarray
    gt_points: np.ndarray


@dataclass
class AssignmentRecord:
    gt_to_query: np.ndarray
    matched_scores: np.ndarray
    matched_distances: np.ndarray
    matched_costs: np.ndarray


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


def _finite_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def _finite_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def _select_candidate_indices(
    scores: np.ndarray,
    gt_count: int,
    candidate_factor: float,
    candidate_extra: int,
    min_candidates: int,
    max_candidates: int,
) -> np.ndarray:
    if len(scores) == 0:
        return np.zeros(0, dtype=np.int64)
    target = int(max(min_candidates, gt_count * candidate_factor + candidate_extra))
    target = min(max_candidates, target, len(scores))
    if target >= len(scores):
        return np.arange(len(scores), dtype=np.int64)
    return np.argsort(scores)[-target:].astype(np.int64)


def _assign_for_cost(
    record: PredictionRecord,
    cost_class: float,
    cost_point: float,
    candidate_factor: float,
    candidate_extra: int,
    min_candidates: int,
    max_candidates: int,
) -> AssignmentRecord:
    gt_count = len(record.gt_points)
    gt_to_query = np.full(gt_count, -1, dtype=np.int64)
    matched_scores = np.full(gt_count, np.nan, dtype=np.float64)
    matched_distances = np.full(gt_count, np.nan, dtype=np.float64)
    matched_costs = np.full(gt_count, np.nan, dtype=np.float64)
    if gt_count == 0 or len(record.scores) == 0:
        return AssignmentRecord(gt_to_query, matched_scores, matched_distances, matched_costs)

    candidate_idx = _select_candidate_indices(
        record.scores,
        gt_count=gt_count,
        candidate_factor=candidate_factor,
        candidate_extra=candidate_extra,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
    )
    candidate_scores = record.scores[candidate_idx]
    candidate_points = record.points[candidate_idx]
    distances = np.linalg.norm(
        candidate_points[:, None, :] - record.gt_points[None, :, :], axis=2
    )
    costs = cost_point * distances - cost_class * candidate_scores[:, None]
    pred_idx, gt_idx = linear_sum_assignment(costs)
    full_query_idx = candidate_idx[pred_idx]
    gt_to_query[gt_idx] = full_query_idx
    matched_scores[gt_idx] = record.scores[full_query_idx]
    matched_distances[gt_idx] = distances[pred_idx, gt_idx]
    matched_costs[gt_idx] = costs[pred_idx, gt_idx]
    return AssignmentRecord(gt_to_query, matched_scores, matched_distances, matched_costs)


def _summarize_assignment(
    cost_point: float,
    assignments: list[AssignmentRecord],
    baseline_assignments: list[AssignmentRecord] | None,
) -> dict[str, float | int]:
    scores = np.concatenate([item.matched_scores for item in assignments])
    distances = np.concatenate([item.matched_distances for item in assignments])
    costs = np.concatenate([item.matched_costs for item in assignments])
    valid = np.isfinite(scores) & np.isfinite(distances)
    scores = scores[valid]
    distances = distances[valid]
    costs = costs[valid]
    row: dict[str, float | int] = {
        "cost_point": float(cost_point),
        "matches": int(len(scores)),
        "score_mean": _finite_mean(scores),
        "score_median": _finite_median(scores),
        "score_p10": _finite_quantile(scores, 0.10),
        "dist_mean": _finite_mean(distances),
        "dist_median": _finite_median(distances),
        "dist_p90": _finite_quantile(distances, 0.90),
        "cost_mean": _finite_mean(costs),
        "matched_score_gt_05": float(np.mean(scores > 0.5)) if scores.size else float("nan"),
        "matched_score_gt_09": float(np.mean(scores >= 0.9)) if scores.size else float("nan"),
        "matched_dist_gt_8": float(np.mean(distances > 8.0)) if distances.size else float("nan"),
        "matched_dist_gt_12": float(np.mean(distances > 12.0)) if distances.size else float("nan"),
    }
    if baseline_assignments is None:
        row.update(
            {
                "same_gt_query_rate": 1.0,
                "score_delta_vs_base": 0.0,
                "dist_delta_vs_base": 0.0,
                "changed_score_delta": 0.0,
                "changed_dist_delta": 0.0,
                "changed_gt_count": 0,
            }
        )
        return row

    same_flags: list[np.ndarray] = []
    score_deltas: list[np.ndarray] = []
    dist_deltas: list[np.ndarray] = []
    changed_score_deltas: list[np.ndarray] = []
    changed_dist_deltas: list[np.ndarray] = []
    for assignment, baseline in zip(assignments, baseline_assignments):
        valid_pair = (assignment.gt_to_query >= 0) & (baseline.gt_to_query >= 0)
        if not np.any(valid_pair):
            continue
        same = assignment.gt_to_query[valid_pair] == baseline.gt_to_query[valid_pair]
        score_delta = assignment.matched_scores[valid_pair] - baseline.matched_scores[valid_pair]
        dist_delta = assignment.matched_distances[valid_pair] - baseline.matched_distances[valid_pair]
        same_flags.append(same)
        score_deltas.append(score_delta)
        dist_deltas.append(dist_delta)
        if np.any(~same):
            changed_score_deltas.append(score_delta[~same])
            changed_dist_deltas.append(dist_delta[~same])

    same_all = np.concatenate(same_flags) if same_flags else np.zeros(0, dtype=bool)
    score_delta_all = np.concatenate(score_deltas) if score_deltas else np.zeros(0)
    dist_delta_all = np.concatenate(dist_deltas) if dist_deltas else np.zeros(0)
    changed_score_all = (
        np.concatenate(changed_score_deltas) if changed_score_deltas else np.zeros(0)
    )
    changed_dist_all = (
        np.concatenate(changed_dist_deltas) if changed_dist_deltas else np.zeros(0)
    )
    row.update(
        {
            "same_gt_query_rate": float(np.mean(same_all)) if same_all.size else float("nan"),
            "score_delta_vs_base": _finite_mean(score_delta_all),
            "dist_delta_vs_base": _finite_mean(dist_delta_all),
            "changed_score_delta": _finite_mean(changed_score_all),
            "changed_dist_delta": _finite_mean(changed_dist_all),
            "changed_gt_count": int(changed_score_all.size),
        }
    )
    return row


@torch.no_grad()
def _collect_records(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_depth: bool,
) -> list[PredictionRecord]:
    model.eval()
    records: list[PredictionRecord] = []
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
            PredictionRecord(
                image_id=int(target["image_id"].item()) if "image_id" in target else index,
                gt_count=int(target["point"].shape[0]),
                scores=scores.numpy().astype(np.float64),
                points=outputs["pred_points"][0].detach().cpu().numpy().astype(np.float64),
                gt_points=target["point"].detach().cpu().numpy().astype(np.float64),
            )
        )
        if (index + 1) % 25 == 0:
            logger.info(f"Collected predictions for {index + 1}/{len(data_loader.dataset)} images")
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


def _save_plots(output_dir: Path, rows: list[dict[str, float | int]]) -> None:
    costs = [float(row["cost_point"]) for row in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(costs, [float(row["dist_mean"]) for row in rows], marker="o", label="mean")
    plt.plot(costs, [float(row["dist_p90"]) for row in rows], marker="o", label="p90")
    plt.xscale("log")
    plt.xlabel("Matcher cost_point")
    plt.ylabel("Matched distance (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "matched_distance_by_cost.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(costs, [float(row["score_mean"]) for row in rows], marker="o", label="mean")
    plt.plot(costs, [float(row["score_p10"]) for row in rows], marker="o", label="p10")
    plt.xscale("log")
    plt.xlabel("Matcher cost_point")
    plt.ylabel("Matched foreground score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "matched_score_by_cost.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(costs, [float(row["same_gt_query_rate"]) for row in rows], marker="o")
    plt.xscale("log")
    plt.xlabel("Matcher cost_point")
    plt.ylabel("Same GT-query assignment vs baseline")
    plt.tight_layout()
    plt.savefig(output_dir / "assignment_churn_by_cost.png", dpi=160)
    plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    proxy_cfg = getattr(cfg, "matcher_proxy", None)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path(
        _cfg_get(proxy_cfg, "output_dir", f"outputs/matcher_proxy/{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), log_file="matcher_proxy.log")

    cost_points = _parse_float_list(
        _cfg_get(proxy_cfg, "cost_points", None), [0.025, 0.05, 0.1, 0.2, 0.4]
    )
    baseline_cost_point = float(_cfg_get(proxy_cfg, "baseline_cost_point", cfg.model.set_cost_point))
    cost_class = float(_cfg_get(proxy_cfg, "cost_class", cfg.model.set_cost_class))
    candidate_factor = float(_cfg_get(proxy_cfg, "candidate_factor", 3.0))
    candidate_extra = int(_cfg_get(proxy_cfg, "candidate_extra", 512))
    min_candidates = int(_cfg_get(proxy_cfg, "min_candidates", 1024))
    max_candidates = int(_cfg_get(proxy_cfg, "max_candidates", 4096))
    if baseline_cost_point not in cost_points:
        cost_points = sorted([*cost_points, baseline_cost_point])

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
    records = _collect_records(model, val_loader, device, use_depth=use_depth)
    logger.info("Finished prediction collection; running matcher proxy sweep")

    assignments_by_cost: dict[float, list[AssignmentRecord]] = {}
    for cost_point in cost_points:
        assignments_by_cost[float(cost_point)] = [
            _assign_for_cost(
                record,
                cost_class=cost_class,
                cost_point=float(cost_point),
                candidate_factor=candidate_factor,
                candidate_extra=candidate_extra,
                min_candidates=min_candidates,
                max_candidates=max_candidates,
            )
            for record in records
        ]
        logger.info(f"Matched proxy assignments for cost_point={float(cost_point):g}")

    baseline_assignments = assignments_by_cost[baseline_cost_point]
    rows = [
        _summarize_assignment(
            float(cost_point),
            assignments_by_cost[float(cost_point)],
            None if float(cost_point) == baseline_cost_point else baseline_assignments,
        )
        for cost_point in cost_points
    ]

    summary = {
        "config": {
            "weight_path": weight_path,
            "num_images": len(records),
            "cost_class": cost_class,
            "baseline_cost_point": baseline_cost_point,
            "cost_points": cost_points,
            "candidate_factor": candidate_factor,
            "candidate_extra": candidate_extra,
            "min_candidates": min_candidates,
            "max_candidates": max_candidates,
            "note": "Offline proxy only: current predictions are fixed; this does not estimate retrained MAE.",
        },
        "rows": rows,
    }
    with (output_dir / "matcher_proxy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    _save_plots(output_dir, rows)

    _print_table(
        "--- Matcher Cost Proxy Sweep ---",
        [
            "cost_pt",
            "dist_mean",
            "dist_p90",
            "score_mean",
            "score_p10",
            "score>0.5",
            "dist>8",
            "same_vs_base",
            "d_dist",
            "d_score",
            "chg",
        ],
        [
            [
                f"{float(row['cost_point']):g}",
                f"{float(row['dist_mean']):.2f}",
                f"{float(row['dist_p90']):.2f}",
                f"{float(row['score_mean']):.3f}",
                f"{float(row['score_p10']):.3f}",
                f"{float(row['matched_score_gt_05']):.3f}",
                f"{float(row['matched_dist_gt_8']):.3f}",
                f"{float(row['same_gt_query_rate']):.3f}",
                f"{float(row['dist_delta_vs_base']):+.2f}",
                f"{float(row['score_delta_vs_base']):+.3f}",
                row["changed_gt_count"],
            ]
            for row in rows
        ],
    )
    _print_table(
        "--- Changed Assignment Deltas vs Baseline ---",
        ["cost_pt", "changed", "changed_d_dist", "changed_d_score"],
        [
            [
                f"{float(row['cost_point']):g}",
                row["changed_gt_count"],
                f"{float(row['changed_dist_delta']):+.2f}",
                f"{float(row['changed_score_delta']):+.3f}",
            ]
            for row in rows
        ],
    )
    logger.info(f"Saved matcher proxy sweep to {output_dir}")


if __name__ == "__main__":
    main()
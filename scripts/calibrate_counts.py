"""Offline count calibration experiments from a diag_baseline JSON file.

The script estimates whether existing classification and density counts can be
post-calibrated into a lower-MAE count without retraining the model. It reports
out-of-fold metrics for simple calibrators and saves a JSON plus two plots.

Example:

    uv run python scripts/calibrate_counts.py \
        --diag outputs/diag_baseline/2026-04-25/22-59-37/diag_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class Dataset:
    gt: np.ndarray
    cls: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class Result:
    name: str
    pred: np.ndarray
    params: dict[str, object]


Predictor = Callable[[Dataset, Dataset], tuple[np.ndarray, dict[str, object]]]


def load_dataset(path: Path) -> Dataset:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload["per_image"]
    return Dataset(
        gt=np.asarray([r["gt_count"] for r in rows], dtype=np.float64),
        cls=np.asarray([r["cls_count"] for r in rows], dtype=np.float64),
        density=np.asarray([r["density_count"] for r in rows], dtype=np.float64),
    )


def subset(data: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(data.gt[indices], data.cls[indices], data.density[indices])


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = pred - gt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
    }


def least_squares(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return test_x @ weights, weights


def pred_baseline_cls(
    _: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    return test.cls.copy(), {}


def pred_baseline_density(
    _: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    return test.density.copy(), {}


def pred_cls_bias(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    bias = float(np.mean(train.gt - train.cls))
    return test.cls + bias, {"bias": bias}


def pred_density_bias(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    bias = float(np.mean(train.gt - train.density))
    return test.density + bias, {"bias": bias}


def pred_cls_linear(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    train_x = np.column_stack([train.cls, np.ones_like(train.cls)])
    test_x = np.column_stack([test.cls, np.ones_like(test.cls)])
    pred, weights = least_squares(train_x, train.gt, test_x)
    return pred, {"a_cls": float(weights[0]), "c": float(weights[1])}


def pred_density_linear(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    train_x = np.column_stack([train.density, np.ones_like(train.density)])
    test_x = np.column_stack([test.density, np.ones_like(test.density)])
    pred, weights = least_squares(train_x, train.gt, test_x)
    return pred, {"a_density": float(weights[0]), "c": float(weights[1])}


def pred_two_head_linear(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    train_x = np.column_stack([train.cls, train.density, np.ones_like(train.cls)])
    test_x = np.column_stack([test.cls, test.density, np.ones_like(test.cls)])
    pred, weights = least_squares(train_x, train.gt, test_x)
    return pred, {
        "a_cls": float(weights[0]),
        "b_density": float(weights[1]),
        "c": float(weights[2]),
    }


def pred_grid_blend(
    train: Dataset, test: Dataset
) -> tuple[np.ndarray, dict[str, object]]:
    best_mae = float("inf")
    best_alpha = 1.0
    best_bias = 0.0
    for alpha in np.linspace(-0.5, 1.5, 401):
        base = alpha * train.cls + (1.0 - alpha) * train.density
        bias = float(np.mean(train.gt - base))
        mae = float(np.mean(np.abs(base + bias - train.gt)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
            best_bias = bias
    pred = best_alpha * test.cls + (1.0 - best_alpha) * test.density + best_bias
    return pred, {
        "alpha_cls": best_alpha,
        "density_weight": 1.0 - best_alpha,
        "bias": best_bias,
    }


def bucket_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bucket_residual(
    train_feature: np.ndarray,
    train_residual: np.ndarray,
    test_feature: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, dict[str, object]]:
    edges = bucket_edges(train_feature, n_bins)
    global_residual = float(np.mean(train_residual))
    residuals: list[float] = []
    counts: list[int] = []
    for idx in range(n_bins):
        mask = (train_feature >= edges[idx]) & (train_feature < edges[idx + 1])
        counts.append(int(mask.sum()))
        residuals.append(
            float(np.mean(train_residual[mask])) if np.any(mask) else global_residual
        )
    bucket_ids = np.clip(
        np.searchsorted(edges[1:-1], test_feature, side="right"), 0, n_bins - 1
    )
    correction = np.asarray([residuals[i] for i in bucket_ids], dtype=np.float64)
    return correction, {
        "edges": [float(v) if np.isfinite(v) else str(v) for v in edges],
        "residuals": residuals,
        "counts": counts,
    }


def make_pred_cls_bucket(n_bins: int) -> Predictor:
    def predict(train: Dataset, test: Dataset) -> tuple[np.ndarray, dict[str, object]]:
        correction, params = apply_bucket_residual(
            train.cls, train.gt - train.cls, test.cls, n_bins
        )
        return test.cls + correction, params

    return predict


def make_pred_density_bucket(n_bins: int) -> Predictor:
    def predict(train: Dataset, test: Dataset) -> tuple[np.ndarray, dict[str, object]]:
        correction, params = apply_bucket_residual(
            train.density, train.gt - train.cls, test.density, n_bins
        )
        return test.cls + correction, params

    return predict


def kfold_indices(n_samples: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, folds) if len(fold) > 0]


def cross_validate(
    data: Dataset, methods: dict[str, Predictor], folds: int, seed: int
) -> list[Result]:
    fold_indices = kfold_indices(len(data.gt), folds=folds, seed=seed)
    all_indices = np.arange(len(data.gt))
    results: list[Result] = []
    for name, method in methods.items():
        pred = np.empty_like(data.gt)
        params_by_fold: list[dict[str, object]] = []
        for val_idx in fold_indices:
            train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
            fold_pred, params = method(subset(data, train_idx), subset(data, val_idx))
            pred[val_idx] = fold_pred
            params_by_fold.append(params)
        results.append(
            Result(name=name, pred=pred, params={"fold_params": params_by_fold})
        )
    return results


def fit_full(
    data: Dataset, methods: dict[str, Predictor]
) -> dict[str, dict[str, object]]:
    full_params: dict[str, dict[str, object]] = {}
    for name, method in methods.items():
        _, params = method(data, data)
        full_params[name] = params
    return full_params


def save_plots(data: Dataset, results: list[Result], output_dir: Path) -> None:
    ranked = sorted(results, key=lambda result: metrics(result.pred, data.gt)["mae"])
    best = ranked[0]
    plt.figure(figsize=(7, 6))
    plt.scatter(data.gt, data.cls, s=18, alpha=0.65, label="baseline cls")
    plt.scatter(data.gt, best.pred, s=18, alpha=0.65, label=f"best: {best.name}")
    max_count = float(max(np.max(data.gt), np.max(data.cls), np.max(best.pred), 1.0))
    plt.plot([0, max_count], [0, max_count], "k--", linewidth=1)
    plt.xlabel("GT count")
    plt.ylabel("Predicted count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibrated_scatter.png", dpi=160)
    plt.close()

    names = [result.name for result in ranked]
    maes = [metrics(result.pred, data.gt)["mae"] for result in ranked]
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(names)), maes)
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right")
    plt.ylabel("Out-of-fold MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_mae_bar.png", dpi=160)
    plt.close()


def print_table(rows: list[list[object]], headers: list[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    data = load_dataset(args.diag)
    output_dir = args.out or args.diag.parent / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Predictor] = {
        "baseline_cls": pred_baseline_cls,
        "baseline_density": pred_baseline_density,
        "cls_bias": pred_cls_bias,
        "density_bias": pred_density_bias,
        "cls_linear": pred_cls_linear,
        "density_linear": pred_density_linear,
        "two_head_linear": pred_two_head_linear,
        "grid_blend": pred_grid_blend,
        f"cls_bucket_{args.bins}": make_pred_cls_bucket(args.bins),
        f"density_bucket_{args.bins}": make_pred_density_bucket(args.bins),
    }
    results = cross_validate(data, methods, folds=args.folds, seed=args.seed)
    full_params = fit_full(data, methods)
    ranked = sorted(results, key=lambda result: metrics(result.pred, data.gt)["mae"])
    baseline_mae = metrics(data.cls, data.gt)["mae"]

    table_rows = []
    for result in ranked:
        result_metrics = metrics(result.pred, data.gt)
        table_rows.append(
            [
                result.name,
                f"{result_metrics['mae']:.3f}",
                f"{result_metrics['rmse']:.3f}",
                f"{result_metrics['bias']:.3f}",
                f"{result_metrics['median_abs_error']:.3f}",
                f"{baseline_mae - result_metrics['mae']:+.3f}",
            ]
        )
    print_table(
        table_rows, ["method", "mae", "rmse", "bias", "med_ae", "mae_gain_vs_cls"]
    )

    payload = {
        "diag": str(args.diag),
        "folds": args.folds,
        "seed": args.seed,
        "bins": args.bins,
        "num_samples": int(len(data.gt)),
        "best_method": ranked[0].name,
        "results": {
            result.name: {
                **metrics(result.pred, data.gt),
                "mae_gain_vs_baseline_cls": baseline_mae
                - metrics(result.pred, data.gt)["mae"],
                "full_fit_params": full_params[result.name],
                "out_of_fold_pred": result.pred.tolist(),
            }
            for result in ranked
        },
        "gt_count": data.gt.tolist(),
        "cls_count": data.cls.tolist(),
        "density_count": data.density.tolist(),
    }
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    save_plots(data, ranked, output_dir)
    print(f"\nBest method: {ranked[0].name}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""Compare density-map generation quality by density integral vs GT count.

The script renders or reuses cached density maps for multiple datasets and
methods, then reports how close each full-image density-map sum is to the
number of annotated points.

Example:

    uv run python scripts/analyze_density_generation_quality.py \
        +analysis.datasets=[shha,shhb,ucf_qnrf] \
        +analysis.methods=[fixed,geometry_adaptive,hybrid] \
        +analysis.output_dir=outputs/density_quality_2026-05-08
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crowdcount.data.prepare import (
    _find_image_gt_pairs,
    _load_points,
    _resolve_density_cache_dir,
    generate_density_maps,
)
from crowdcount.utils.logging import logger, setup_logger


METHOD_ALIASES = {
    "geo": "geometry_adaptive",
    "geometry": "geometry_adaptive",
    "geometry-adaptive": "geometry_adaptive",
    "geometry_adaptive": "geometry_adaptive",
    "fixed": "fixed",
    "hybrid": "hybrid",
}

DEFAULT_DATASETS = ["shha", "shhb", "ucf_qnrf"]
DEFAULT_METHODS = ["fixed", "geometry_adaptive", "hybrid"]


@dataclass(frozen=True)
class MethodParams:
    method: str
    mode: str
    perspective_guided: bool
    hybrid: bool
    fixed_sigma: float
    beta: float
    min_sigma: float
    sigma_base: float
    persp_max_sigma: float | None
    hybrid_min_sigma: float
    hybrid_max_sigma: float | None
    hybrid_alpha: float
    disparity_input: bool


@dataclass(frozen=True)
class DensityRecord:
    dataset: str
    method: str
    split: str
    image: str
    gt_count: int
    in_bounds_gt_count: int
    out_of_bounds_count: int
    image_height: int
    image_width: int
    density_sum: float
    error: float
    abs_error: float
    rel_abs_error: float
    in_bounds_error: float
    in_bounds_abs_error: float
    in_bounds_rel_abs_error: float
    cache_dir: str


@dataclass(frozen=True)
class SummaryRecord:
    dataset: str
    method: str
    split: str
    n_images: int
    gt_total: int
    density_total: float
    mae: float
    rmse: float
    bias: float
    mean_rel_abs_error: float
    max_abs_error: float
    under_ratio: float
    over_ratio: float


@dataclass(frozen=True)
class InBoundsSummaryRecord:
    dataset: str
    method: str
    split: str
    n_images: int
    gt_total: int
    in_bounds_gt_total: int
    out_of_bounds_total: int
    out_of_bounds_rate: float
    density_total: float
    in_bounds_mae: float
    in_bounds_rmse: float
    in_bounds_bias: float
    in_bounds_mean_rel_abs_error: float
    in_bounds_max_abs_error: float
    in_bounds_under_ratio: float
    in_bounds_over_ratio: float


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalise_method(method: str) -> str:
    key = method.strip().lower()
    if key not in METHOD_ALIASES:
        valid = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown density method {method!r}; expected one of: {valid}")
    return METHOD_ALIASES[key]


def _method_params(method: str, density_cfg: Any) -> MethodParams:
    method = _normalise_method(method)
    fixed_sigma = float(_cfg_get(density_cfg, "fixed_sigma", 8.0))
    beta = float(_cfg_get(density_cfg, "beta", 0.3))
    min_sigma = float(_cfg_get(density_cfg, "min_sigma", 1.0))
    sigma_base = float(_cfg_get(density_cfg, "sigma_base", 1.0))
    persp_max_sigma = _as_optional_float(_cfg_get(density_cfg, "persp_max_sigma", None))
    hybrid_min_sigma = float(_cfg_get(density_cfg, "hybrid_min_sigma", 1.5))
    hybrid_max_sigma = _as_optional_float(_cfg_get(density_cfg, "hybrid_max_sigma", None))
    hybrid_alpha = float(_cfg_get(density_cfg, "hybrid_alpha", 0.5))
    disparity_input = bool(_cfg_get(density_cfg, "disparity_input", True))

    if method == "fixed":
        return MethodParams(
            method=method,
            mode="fixed",
            perspective_guided=False,
            hybrid=False,
            fixed_sigma=fixed_sigma,
            beta=beta,
            min_sigma=min_sigma,
            sigma_base=sigma_base,
            persp_max_sigma=persp_max_sigma,
            hybrid_min_sigma=hybrid_min_sigma,
            hybrid_max_sigma=hybrid_max_sigma,
            hybrid_alpha=hybrid_alpha,
            disparity_input=disparity_input,
        )
    if method == "hybrid":
        return MethodParams(
            method=method,
            mode="hybrid",
            perspective_guided=False,
            hybrid=True,
            fixed_sigma=fixed_sigma,
            beta=beta,
            min_sigma=min_sigma,
            sigma_base=sigma_base,
            persp_max_sigma=persp_max_sigma,
            hybrid_min_sigma=hybrid_min_sigma,
            hybrid_max_sigma=hybrid_max_sigma,
            hybrid_alpha=hybrid_alpha,
            disparity_input=disparity_input,
        )
    return MethodParams(
        method=method,
        mode="geometry_adaptive",
        perspective_guided=False,
        hybrid=False,
        fixed_sigma=fixed_sigma,
        beta=beta,
        min_sigma=min_sigma,
        sigma_base=sigma_base,
        persp_max_sigma=persp_max_sigma,
        hybrid_min_sigma=hybrid_min_sigma,
        hybrid_max_sigma=hybrid_max_sigma,
        hybrid_alpha=hybrid_alpha,
        disparity_input=disparity_input,
    )


def _summary(records: list[DensityRecord]) -> SummaryRecord:
    if not records:
        raise ValueError("Cannot summarise an empty record list")
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    density = np.asarray([record.density_sum for record in records], dtype=np.float64)
    errors = density - gt
    abs_errors = np.abs(errors)
    return SummaryRecord(
        dataset=records[0].dataset,
        method=records[0].method,
        split=records[0].split,
        n_images=len(records),
        gt_total=int(np.sum(gt)),
        density_total=float(np.sum(density)),
        mae=float(np.mean(abs_errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        bias=float(np.mean(errors)),
        mean_rel_abs_error=float(np.mean([record.rel_abs_error for record in records])),
        max_abs_error=float(np.max(abs_errors)),
        under_ratio=float(np.mean(errors < -1e-6)),
        over_ratio=float(np.mean(errors > 1e-6)),
    )


def _in_bounds_summary(records: list[DensityRecord]) -> InBoundsSummaryRecord:
    if not records:
        raise ValueError("Cannot summarise an empty record list")
    raw_gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    in_bounds_gt = np.asarray(
        [record.in_bounds_gt_count for record in records], dtype=np.float64
    )
    density = np.asarray([record.density_sum for record in records], dtype=np.float64)
    errors = density - in_bounds_gt
    abs_errors = np.abs(errors)
    gt_total = int(np.sum(raw_gt))
    out_of_bounds_total = int(
        np.sum([record.out_of_bounds_count for record in records])
    )
    return InBoundsSummaryRecord(
        dataset=records[0].dataset,
        method=records[0].method,
        split=records[0].split,
        n_images=len(records),
        gt_total=gt_total,
        in_bounds_gt_total=int(np.sum(in_bounds_gt)),
        out_of_bounds_total=out_of_bounds_total,
        out_of_bounds_rate=(out_of_bounds_total / gt_total if gt_total > 0 else 0.0),
        density_total=float(np.sum(density)),
        in_bounds_mae=float(np.mean(abs_errors)),
        in_bounds_rmse=float(np.sqrt(np.mean(errors**2))),
        in_bounds_bias=float(np.mean(errors)),
        in_bounds_mean_rel_abs_error=float(
            np.mean([record.in_bounds_rel_abs_error for record in records])
        ),
        in_bounds_max_abs_error=float(np.max(abs_errors)),
        in_bounds_under_ratio=float(np.mean(errors < -1e-6)),
        in_bounds_over_ratio=float(np.mean(errors > 1e-6)),
    )


def _count_in_bounds(points: np.ndarray, height: int, width: int) -> int:
    if len(points) == 0:
        return 0
    rounded = np.round(points).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    return int(np.sum(in_bounds))


def _density_kwargs(params: MethodParams) -> dict[str, Any]:
    return {
        "mode": params.mode,
        "perspective_guided": params.perspective_guided,
        "hybrid": params.hybrid,
        "fixed_sigma": params.fixed_sigma,
        "beta": params.beta,
        "min_sigma": params.min_sigma,
        "sigma_base": params.sigma_base,
        "persp_max_sigma": params.persp_max_sigma,
        "hybrid_min_sigma": params.hybrid_min_sigma,
        "hybrid_max_sigma": params.hybrid_max_sigma,
        "hybrid_alpha": params.hybrid_alpha,
    }


def collect_density_records(
    *,
    dataset_name: str,
    data_root: str | Path,
    split: str,
    params: MethodParams,
    ensure_cache: bool = True,
) -> list[DensityRecord]:
    """Collect per-image density-sum records for one dataset and method."""
    data_root = Path(data_root)
    kwargs = _density_kwargs(params)
    if ensure_cache:
        generate_density_maps(
            data_root,
            split=split,
            disparity_input=params.disparity_input,
            **kwargs,
        )

    cache_dir = _resolve_density_cache_dir(data_root, split, **kwargs)
    pairs = _find_image_gt_pairs(data_root, split)
    records: list[DensityRecord] = []
    missing: list[Path] = []
    for img_path, gt_path in pairs:
        density_path = cache_dir / f"{img_path.stem}.npy"
        if not density_path.exists():
            missing.append(density_path)
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            raise OSError(f"Cannot read image: {img_path}")
        height, width = img.shape[:2]
        points = _load_points(gt_path)
        gt_count = int(points.shape[0])
        in_bounds_gt_count = _count_in_bounds(points, height, width)
        out_of_bounds_count = gt_count - in_bounds_gt_count
        density_sum = float(np.load(str(density_path)).sum())
        error = density_sum - float(gt_count)
        abs_error = abs(error)
        rel_abs_error = abs_error / float(gt_count) if gt_count > 0 else abs_error
        in_bounds_error = density_sum - float(in_bounds_gt_count)
        in_bounds_abs_error = abs(in_bounds_error)
        in_bounds_rel_abs_error = (
            in_bounds_abs_error / float(in_bounds_gt_count)
            if in_bounds_gt_count > 0
            else in_bounds_abs_error
        )
        records.append(
            DensityRecord(
                dataset=dataset_name,
                method=params.method,
                split=split,
                image=img_path.name,
                gt_count=gt_count,
                in_bounds_gt_count=in_bounds_gt_count,
                out_of_bounds_count=out_of_bounds_count,
                image_height=height,
                image_width=width,
                density_sum=density_sum,
                error=error,
                abs_error=abs_error,
                rel_abs_error=rel_abs_error,
                in_bounds_error=in_bounds_error,
                in_bounds_abs_error=in_bounds_abs_error,
                in_bounds_rel_abs_error=in_bounds_rel_abs_error,
                cache_dir=str(cache_dir),
            )
        )
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} density maps for {dataset_name}/{params.method}. "
            f"First missing paths: {preview}"
        )
    return records


def _write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_summary_table(rows: list[SummaryRecord]) -> None:
    if not rows:
        logger.warning("No summary rows to print")
        return
    headers = [
        "dataset",
        "method",
        "n",
        "gt_total",
        "density_total",
        "mae",
        "rmse",
        "bias",
        "rel_abs",
        "under",
        "over",
    ]
    table_rows = [
        [
            row.dataset,
            row.method,
            str(row.n_images),
            str(row.gt_total),
            f"{row.density_total:.2f}",
            f"{row.mae:.4f}",
            f"{row.rmse:.4f}",
            f"{row.bias:.4f}",
            f"{row.mean_rel_abs_error:.6f}",
            f"{row.under_ratio:.3f}",
            f"{row.over_ratio:.3f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]
    header_line = "  ".join(item.ljust(widths[idx]) for idx, item in enumerate(headers))
    sep_line = "  ".join("-" * width for width in widths)
    logger.info("\n" + header_line + "\n" + sep_line)
    for row in table_rows:
        logger.info("  ".join(item.ljust(widths[idx]) for idx, item in enumerate(row)))


def _print_in_bounds_summary_table(rows: list[InBoundsSummaryRecord]) -> None:
    if not rows:
        logger.warning("No in-bound summary rows to print")
        return
    headers = [
        "dataset",
        "method",
        "n",
        "raw_gt",
        "in_gt",
        "out_gt",
        "out_rate",
        "density_total",
        "mae",
        "rmse",
        "bias",
        "rel_abs",
    ]
    table_rows = [
        [
            row.dataset,
            row.method,
            str(row.n_images),
            str(row.gt_total),
            str(row.in_bounds_gt_total),
            str(row.out_of_bounds_total),
            f"{row.out_of_bounds_rate:.4f}",
            f"{row.density_total:.2f}",
            f"{row.in_bounds_mae:.4f}",
            f"{row.in_bounds_rmse:.4f}",
            f"{row.in_bounds_bias:.4f}",
            f"{row.in_bounds_mean_rel_abs_error:.6f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]
    header_line = "  ".join(item.ljust(widths[idx]) for idx, item in enumerate(headers))
    sep_line = "  ".join("-" * width for width in widths)
    logger.info("\nIn-bound GT summary\n" + header_line + "\n" + sep_line)
    for row in table_rows:
        logger.info("  ".join(item.ljust(widths[idx]) for idx, item in enumerate(row)))


def _plot_scatter(records: list[DensityRecord], output_dir: Path) -> None:
    _plot_scatter_by_reference(
        records,
        output_dir,
        x_attr="gt_count",
        x_label="Raw GT count",
        title_suffix="density sum vs raw GT",
        file_suffix="density_sum_vs_gt",
    )


def _plot_in_bounds_scatter(records: list[DensityRecord], output_dir: Path) -> None:
    _plot_scatter_by_reference(
        records,
        output_dir,
        x_attr="in_bounds_gt_count",
        x_label="In-bound GT count",
        title_suffix="density sum vs in-bound GT",
        file_suffix="density_sum_vs_in_bounds_gt",
    )


def _plot_scatter_by_reference(
    records: list[DensityRecord],
    output_dir: Path,
    *,
    x_attr: str,
    x_label: str,
    title_suffix: str,
    file_suffix: str,
) -> None:
    by_dataset: dict[str, list[DensityRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)

    for dataset, dataset_records in by_dataset.items():
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        max_value = 1.0
        for method in DEFAULT_METHODS:
            method_records = [r for r in dataset_records if r.method == method]
            if not method_records:
                continue
            gt = np.asarray(
                [getattr(r, x_attr) for r in method_records], dtype=np.float64
            )
            density = np.asarray([r.density_sum for r in method_records], dtype=np.float64)
            max_value = max(max_value, float(gt.max(initial=0.0)), float(density.max(initial=0.0)))
            ax.scatter(gt, density, s=14, alpha=0.55, label=method)
        ax.plot([0, max_value], [0, max_value], color="black", linewidth=1.0, linestyle="--", label="y=x")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density map sum")
        ax.set_title(f"{dataset}: {title_suffix}")
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_{file_suffix}.png", dpi=160)
        plt.close(fig)


def _plot_summary(rows: list[SummaryRecord], output_dir: Path) -> None:
    if not rows:
        return
    datasets = list(dict.fromkeys(row.dataset for row in rows))
    methods = list(dict.fromkeys(row.method for row in rows))
    metrics = ["mae", "bias", "mean_rel_abs_error"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.8 / max(len(methods), 1)
    row_map = {(row.dataset, row.method): row for row in rows}
    for axis, metric in zip(axes, metrics):
        for idx, method in enumerate(methods):
            values = [float(getattr(row_map[(dataset, method)], metric)) for dataset in datasets]
            offset = (idx - (len(methods) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=method)
        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=20, ha="right")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "density_generation_summary.png", dpi=160)
    plt.close(fig)


def _plot_in_bounds_summary(rows: list[InBoundsSummaryRecord], output_dir: Path) -> None:
    if not rows:
        return
    datasets = list(dict.fromkeys(row.dataset for row in rows))
    methods = list(dict.fromkeys(row.method for row in rows))
    metrics = [
        "in_bounds_mae",
        "in_bounds_bias",
        "in_bounds_mean_rel_abs_error",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.8 / max(len(methods), 1)
    row_map = {(row.dataset, row.method): row for row in rows}
    for axis, metric in zip(axes, metrics):
        for idx, method in enumerate(methods):
            values = [float(getattr(row_map[(dataset, method)], metric)) for dataset in datasets]
            offset = (idx - (len(methods) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=method)
        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels(datasets, rotation=20, ha="right")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "density_generation_in_bounds_summary.png", dpi=160)
    plt.close(fig)


def _load_dataset_cfg(config_dir: Path, dataset_name: str) -> DictConfig:
    path = config_dir / "data" / f"{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    dataset_cfg = OmegaConf.load(path)
    if not isinstance(dataset_cfg, DictConfig):
        raise TypeError(f"Expected mapping config in {path}, got {type(dataset_cfg)}")
    return dataset_cfg


def _analysis_cfg(cfg: DictConfig) -> Any:
    return getattr(cfg, "analysis", None)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    analysis = _analysis_cfg(cfg)
    datasets = _as_list(_cfg_get(analysis, "datasets", None), DEFAULT_DATASETS)
    methods = [
        _normalise_method(method)
        for method in _as_list(_cfg_get(analysis, "methods", None), DEFAULT_METHODS)
    ]
    split = str(_cfg_get(analysis, "split", "train"))
    ensure_cache = bool(_cfg_get(analysis, "ensure_cache", True))
    output_dir_value = _cfg_get(analysis, "output_dir", None)
    output_dir = (
        Path(output_dir_value)
        if output_dir_value
        else Path("outputs") / f"density_quality_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir, log_file="density_generation_quality.log")

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    all_records: list[DensityRecord] = []
    summaries: list[SummaryRecord] = []
    in_bounds_summaries: list[InBoundsSummaryRecord] = []

    for dataset_name in datasets:
        dataset_cfg = _load_dataset_cfg(config_dir, dataset_name)
        if dataset_name == str(getattr(cfg.data, "dataset", "")).lower():
            dataset_cfg = OmegaConf.merge(dataset_cfg, cfg.data)
        data_root = Path(str(dataset_cfg.data_root))
        density_cfg = getattr(dataset_cfg, "density_generation", None)
        logger.info(f"Analyzing dataset={dataset_name}, split={split}, root={data_root}")
        for method in methods:
            params = _method_params(method, density_cfg)
            logger.info(f"Collecting method={method} with params={params}")
            records = collect_density_records(
                dataset_name=dataset_name,
                data_root=data_root,
                split=split,
                params=params,
                ensure_cache=ensure_cache,
            )
            all_records.extend(records)
            summaries.append(_summary(records))
            in_bounds_summaries.append(_in_bounds_summary(records))

    _write_csv(output_dir / "density_generation_per_image.csv", all_records)
    _write_csv(output_dir / "density_generation_summary.csv", summaries)
    _write_csv(
        output_dir / "density_generation_in_bounds_summary.csv",
        in_bounds_summaries,
    )
    _plot_scatter(all_records, output_dir)
    _plot_in_bounds_scatter(all_records, output_dir)
    _plot_summary(summaries, output_dir)
    _plot_in_bounds_summary(in_bounds_summaries, output_dir)
    OmegaConf.save(cfg, output_dir / "run_config.yaml")
    _print_summary_table(summaries)
    _print_in_bounds_summary_table(in_bounds_summaries)
    logger.info(f"Density generation quality outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

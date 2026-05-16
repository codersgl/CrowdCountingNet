"""Generate paper-quality density generation strategy comparison figure.

Creates a combined figure for the paper showing:
- Left: Density sum vs GT count scatter for all methods on SHA
- Right: MAE/RMSE bar chart comparing methods across datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 200,
    }
)

METHOD_COLORS = {
    "fixed": "#e74c3c",
    "geometry_adaptive": "#3498db",
    "hybrid": "#2ecc71",
}
METHOD_LABELS = {
    "fixed": "Fixed σ=8.0",
    "geometry_adaptive": "Geometry-Adaptive",
    "hybrid": "Depth-Aware (α=0.7, Ours)",
}
DATASET_LABELS = {
    "shha": "SHTech Part A",
    "shhb": "SHTech Part B",
}


def plot_density_comparison(
    per_image_csv: Path,
    summary_csv: Path,
    output_path: Path,
) -> None:
    """Generate a combined 2-panel density comparison figure."""
    df = pd.read_csv(per_image_csv)
    summary = pd.read_csv(summary_csv)

    fig = plt.figure(figsize=(14, 5.8))

    # --- Left panel: SHA scatter ---
    ax1 = fig.add_subplot(1, 2, 1)
    sha_df = df[df["dataset"] == "shha"]
    max_val = 1.0
    for method in ["fixed", "geometry_adaptive", "hybrid"]:
        method_df = sha_df[sha_df["method"] == method]
        if method_df.empty:
            continue
        gt = method_df["gt_count"].values
        density = method_df["density_sum"].values
        max_val = max(max_val, gt.max(), density.max())
        ax1.scatter(
            gt,
            density,
            s=10,
            alpha=0.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolors="none",
        )
    ax1.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Ground-Truth Count")
    ax1.set_ylabel("Density Map Sum")
    ax1.set_title("SHTech Part A: Density Sum vs GT Count")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, linewidth=0.3, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # --- Right panel: MAE/RMSE bar chart ---
    ax2 = fig.add_subplot(1, 2, 2)
    datasets_ordered = ["shha", "shhb"]
    methods_ordered = ["fixed", "geometry_adaptive", "hybrid"]
    metrics = ["mae", "rmse"]
    metric_labels = {"mae": "MAE", "rmse": "RMSE"}

    x = np.arange(len(datasets_ordered), dtype=np.float64)
    n_methods = len(methods_ordered)
    n_metrics = len(metrics)
    total_bars = n_methods * n_metrics
    width = 0.8 / total_bars

    row_map = {}
    for _, row in summary.iterrows():
        row_map[(row["dataset"], row["method"])] = row

    hatch_styles = ["", "//"]
    for mi, method in enumerate(methods_ordered):
        for ei, metric in enumerate(metrics):
            values = []
            for dataset in datasets_ordered:
                key = (dataset, method)
                if key in row_map:
                    values.append(row_map[key][metric])
                else:
                    values.append(0)
            idx_offset = mi * n_metrics + ei
            offset = (idx_offset - (total_bars - 1) / 2.0) * width
            bar_label = f"{METHOD_LABELS[method]} {metric_labels[metric]}"
            ax2.bar(
                x + offset,
                values,
                width=width * 0.9,
                color=METHOD_COLORS[method],
                alpha=0.55 if ei == 1 else 0.85,
                hatch=hatch_styles[ei],
                label=bar_label,
                edgecolor="white" if ei == 0 else METHOD_COLORS[method],
                linewidth=0.5,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[d] for d in datasets_ordered])
    ax2.set_ylabel("Error (Density Integral vs GT)")
    ax2.set_title("Density Integral Error by Method and Dataset")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax2.grid(True, axis="y", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_counting_error_comparison(
    per_image_csv: Path,
    output_path: Path,
) -> None:
    """Generate counting relative/absolute error comparison across methods."""
    df = pd.read_csv(per_image_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    for ax_idx, (metric, ylabel, title) in enumerate(
        [
            ("abs_error", "Absolute Error |D_sum - GT|", "Absolute Counting Error"),
            (
                "rel_abs_error",
                "Relative Error |D_sum - GT| / GT",
                "Relative Counting Error",
            ),
        ]
    ):
        ax = axes[ax_idx]
        for dataset in ["shha", "shhb"]:
            ds_df = df[df["dataset"] == dataset]
            positions = []
            values = []
            for mi, method in enumerate(["fixed", "geometry_adaptive", "hybrid"]):
                method_df = ds_df[ds_df["method"] == method]
                if method_df.empty:
                    continue
                vals = method_df[metric].values
                positions.append(mi + (0 if dataset == "shha" else 0))
                values.append(np.mean(vals))

            # Simplified: just show SHA data points for clarity
            if dataset == "shha":
                for mi, method in enumerate(["fixed", "geometry_adaptive", "hybrid"]):
                    method_df = ds_df[ds_df["method"] == method]
                    if method_df.empty:
                        continue
                    vals = method_df[metric].values
                    ax.bar(
                        mi,
                        np.mean(vals),
                        color=METHOD_COLORS[method],
                        alpha=0.8,
                        label=METHOD_LABELS[method],
                        width=0.6,
                    )
                    ax.errorbar(
                        mi,
                        np.mean(vals),
                        yerr=np.std(vals),
                        fmt="none",
                        ecolor="black",
                        capsize=4,
                        linewidth=1,
                    )

        ax.set_xticks(range(3))
        ax.set_xticklabels(["Fixed σ=8.0", "Geo-Adaptive", "Depth-Aware (Ours)"], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (SHTech Part A)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=Path("outputs/density_quality_comparison/density_generation_per_image.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/density_quality_comparison/density_generation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/density_quality_comparison/"),
    )
    args = parser.parse_args()

    plot_density_comparison(
        args.per_image_csv,
        args.summary_csv,
        args.output_dir / "fig5_density_comparison.png",
    )
    plot_counting_error_comparison(
        args.per_image_csv,
        args.output_dir / "fig_counting_error_comparison.png",
    )


if __name__ == "__main__":
    main()

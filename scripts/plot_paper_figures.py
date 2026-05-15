"""Generate lightweight paper figures from existing experiment artifacts.

This script is intentionally read-only with respect to training artifacts. It
parses logs and summary CSV files, then writes static figures under docs/figures.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


EVAL_RE = re.compile(
    r"\[Eval\]\s+mae=(?P<mae>\d+(?:\.\d+)?)\s+"
    r"mse=(?P<mse>\d+(?:\.\d+)?)\s+.*?"
    r"best_mae=(?P<best_mae>\d+(?:\.\d+)?)\s+"
    r"best_mse=(?P<best_mse>\d+(?:\.\d+)?)"
)
EPOCH_RE = re.compile(r"\[ep (?P<epoch>\d+)\]")


def parse_eval_log(log_path: Path) -> pd.DataFrame:
    """Parse evaluation metrics from a DSGCNet train.log file."""
    rows: list[dict[str, float | int]] = []
    current_epoch: int | None = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match is not None:
            current_epoch = int(epoch_match.group("epoch"))

        eval_match = EVAL_RE.search(line)
        if eval_match is None:
            continue

        rows.append(
            {
                "eval_index": len(rows) + 1,
                "epoch": current_epoch if current_epoch is not None else len(rows),
                "mae": float(eval_match.group("mae")),
                "mse": float(eval_match.group("mse")),
                "best_mae": float(eval_match.group("best_mae")),
                "best_mse": float(eval_match.group("best_mse")),
            }
        )

    if not rows:
        raise ValueError(f"No evaluation rows found in {log_path}")
    return pd.DataFrame(rows)


def save_training_curve(eval_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Save the best-run MAE/MSE convergence curve."""
    output_path = output_dir / "fig_training_curve_shha_best.png"
    csv_path = output_dir / "fig_training_curve_shha_best.csv"
    eval_frame.to_csv(csv_path, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    axis.plot(
        eval_frame["eval_index"],
        eval_frame["mae"],
        color="#2563eb",
        linewidth=1.3,
        alpha=0.55,
        label="Eval MAE",
    )
    axis.plot(
        eval_frame["eval_index"],
        eval_frame["best_mae"],
        color="#0f766e",
        linewidth=2.0,
        label="Best MAE",
    )
    axis.plot(
        eval_frame["eval_index"],
        eval_frame["best_mse"],
        color="#dc2626",
        linewidth=1.7,
        label="Best MSE",
    )
    best_row = eval_frame.loc[eval_frame["best_mae"].idxmin()]
    axis.scatter(
        [best_row["eval_index"]],
        [best_row["best_mae"]],
        color="#111827",
        s=34,
        zorder=4,
    )
    axis.annotate(
        f"MAE {best_row['best_mae']:.2f}\nMSE {best_row['best_mse']:.2f}",
        xy=(best_row["eval_index"], best_row["best_mae"]),
        xytext=(18, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.8},
        fontsize=8.5,
    )
    axis.set_title("SHHA best configuration convergence", fontsize=12, pad=10)
    axis.set_xlabel("Evaluation index")
    axis.set_ylabel("Error")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_component_pass_rates(output_dir: Path) -> Path:
    """Save grouped pass-rate evidence from the historical ablation report."""
    output_path = output_dir / "fig_component_pass_rates.png"
    component_frame = pd.DataFrame(
        [
            {"group": "GM", "setting": "enabled", "pass_rate": 68},
            {"group": "GM", "setting": "disabled", "pass_rate": 0},
            {"group": "Density attention", "setting": "enabled", "pass_rate": 67},
            {"group": "Density attention", "setting": "disabled", "pass_rate": 20},
            {"group": "DAP neck", "setting": "enabled", "pass_rate": 76},
            {"group": "DAP neck", "setting": "disabled", "pass_rate": 0},
            {"group": "GATv2", "setting": "enabled", "pass_rate": 83},
            {"group": "GATv2", "setting": "other/legacy", "pass_rate": 44},
            {"group": "Batch size", "setting": "8", "pass_rate": 62},
            {"group": "Batch size", "setting": "4", "pass_rate": 0},
        ]
    )
    component_frame.to_csv(output_dir / "fig_component_pass_rates.csv", index=False)

    groups = component_frame["group"].drop_duplicates().tolist()
    enabled_values: list[float] = []
    disabled_values: list[float] = []
    disabled_labels: list[str] = []
    for group_name in groups:
        group_rows = component_frame[component_frame["group"] == group_name]
        enabled_values.append(float(group_rows.iloc[0]["pass_rate"]))
        disabled_values.append(float(group_rows.iloc[1]["pass_rate"]))
        disabled_labels.append(str(group_rows.iloc[1]["setting"]))

    x_positions = range(len(groups))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        enabled_values,
        width=bar_width,
        color="#0f766e",
        label="recommended setting",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        disabled_values,
        width=bar_width,
        color="#f97316",
        label="comparison setting",
    )
    axis.set_title("Historical component evidence on SHHA", fontsize=12, pad=10)
    axis.set_ylabel("Pass rate under MAE < 55 (%)")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(
        [f"{group}\nvs {label}" for group, label in zip(groups, disabled_labels)],
        fontsize=8,
    )
    axis.set_ylim(0, 100)
    axis.legend(frameon=True, fontsize=8.5)
    for position, enabled_value, disabled_value in zip(
        x_positions, enabled_values, disabled_values
    ):
        axis.text(
            position - bar_width / 2,
            enabled_value + 2,
            f"{enabled_value:.0f}%",
            ha="center",
            fontsize=8,
        )
        axis.text(
            position + bar_width / 2,
            disabled_value + 2,
            f"{disabled_value:.0f}%",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_shha_density_quality(summary_csv: Path, output_dir: Path) -> Path:
    """Save SHHA in-bound density-generation quality bars."""
    output_path = output_dir / "fig_density_generation_shha_inbound.png"
    summary_frame = pd.read_csv(summary_csv)
    shha_frame = summary_frame[summary_frame["dataset"] == "shha"].copy()
    method_order = ["fixed", "geometry_adaptive", "hybrid"]
    shha_frame["method"] = pd.Categorical(
        shha_frame["method"], categories=method_order, ordered=True
    )
    shha_frame = shha_frame.sort_values("method")
    shha_frame.to_csv(output_dir / "fig_density_generation_shha_inbound.csv", index=False)

    x_positions = range(len(shha_frame))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        shha_frame["in_bounds_mae"],
        width=bar_width,
        color="#2563eb",
        label="MAE",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        shha_frame["in_bounds_rmse"],
        width=bar_width,
        color="#dc2626",
        label="RMSE",
    )
    axis.set_title("SHHA density-generation integral error", fontsize=12, pad=10)
    axis.set_ylabel("Error against in-bound GT")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels([str(method) for method in shha_frame["method"]], fontsize=8.5)
    axis.legend(frameon=True, fontsize=8.5)
    axis.set_ylim(0, max(shha_frame["in_bounds_rmse"]) * 1.25)
    for position, mae_value, rmse_value in zip(
        x_positions, shha_frame["in_bounds_mae"], shha_frame["in_bounds_rmse"]
    ):
        axis.text(position - bar_width / 2, mae_value + 0.15, f"{mae_value:.2f}", ha="center", fontsize=8)
        axis.text(position + bar_width / 2, rmse_value + 0.15, f"{rmse_value:.2f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("outputs/2026-04-25/22-51-51/train.log"),
        help="Best-run train.log to parse.",
    )
    parser.add_argument(
        "--density-summary",
        type=Path,
        default=Path(
            "outputs/density_quality_2026-05-08/density_generation_in_bounds_summary.csv"
        ),
        help="Density-generation in-bound summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/paper"),
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_frame = parse_eval_log(args.log)
    written_paths = [
        save_training_curve(eval_frame, output_dir),
        save_component_pass_rates(output_dir),
        save_shha_density_quality(args.density_summary, output_dir),
    ]
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
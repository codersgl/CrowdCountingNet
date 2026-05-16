"""Visualize three density generation strategies side-by-side.

Generates an N×4 panel figure: each row shows one test image with its
density maps from fixed-sigma, geometry-adaptive, and hybrid strategies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 180,
    }
)

# Cache directories for SHA
SHA_CACHE = {
    "fixed": "gt_density_maps_fixed_s8p00",
    "geo": "gt_density_maps",
    "hybrid": "gt_density_maps_hybrid_a0p70_sb4p00_min1p50_maxinf",
}

SHB_CACHE = {
    "fixed": "gt_density_maps_fixed_s8p00",
    "geo": "gt_density_maps",
    "hybrid": "gt_density_maps_hybrid_a0p70_sb4p00_min1p50_maxinf",
}

METHOD_LABELS = {
    "fixed": "Fixed σ=8.0",
    "geo": "Geometry-Adaptive",
    "hybrid": "Depth-Aware (α=0.7)",
}


def load_gt_points(gt_path: Path) -> np.ndarray:
    """Load point annotations from .mat or txt file."""
    if gt_path.suffix == ".mat":
        import scipy.io

        mat = scipy.io.loadmat(str(gt_path))
        if "annPoints" in mat:
            pts = np.asarray(mat["annPoints"], dtype=np.float32)
        elif "image_info" in mat:
            pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        else:
            raise KeyError(f"Cannot find point array in {gt_path}")
        return pts
    points = []
    for line in gt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            points.append([float(parts[0]), float(parts[1])])
    return np.array(points, dtype=np.float32)


def make_row(
    img_path: Path,
    data_root: Path,
    cache_dirs: dict[str, str],
    split: str = "test",
    target_width: int = 900,
) -> tuple[plt.Figure, list[float]]:
    """Create a 1×4 row: input image | fixed | geo | hybrid."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = target_width / w
    target_h = int(h * scale)

    img_resized = cv2.resize(img, (target_width, target_h))

    gt_path = (
        data_root
        / "test_data"
        / "ground_truth"
        / f"GT_{img_path.stem}.mat"
    )
    if not gt_path.exists():
        gt_path = data_root / "test_data" / "ground_truth" / f"{img_path.stem}.txt"
    if not gt_path.exists():
        gt_path = (
            data_root
            / "test_data"
            / "ground-truth"
            / f"GT_{img_path.stem}.csv"
        )

    points = load_gt_points(gt_path) if gt_path.exists() else np.zeros((0, 2), dtype=np.float32)
    scaled_points = points * scale if len(points) > 0 else points

    density_sums = {}
    density_maps = {}

    for method, cache_sub in cache_dirs.items():
        density_path = data_root / cache_sub / split / f"{img_path.stem}.npy"
        if density_path.exists():
            dmap = np.load(str(density_path))
            density_sums[method] = float(dmap.sum())
            dmap_resized = cv2.resize(dmap, (target_width, target_h))
            density_maps[method] = dmap_resized
        else:
            density_sums[method] = 0.0
            density_maps[method] = np.zeros((target_h, target_width), dtype=np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # Input image
    axes[0].imshow(img_resized)
    if len(points) > 0:
        axes[0].scatter(
            scaled_points[:, 0],
            scaled_points[:, 1],
            c="lime",
            s=4,
            marker=".",
            alpha=0.9,
        )
    axes[0].set_title(f"Input ({len(points)} GT pts)", fontsize=10)
    axes[0].axis("off")

    # Three density maps
    vmax = max(dm.max() for dm in density_maps.values()) if density_maps else 1.0
    for ax, (method, label) in zip(axes[1:], METHOD_LABELS.items()):
        dmap = density_maps[method]
        im = ax.imshow(dmap, cmap="hot", vmin=0, vmax=vmax)
        if len(points) > 0:
            ax.scatter(
                scaled_points[:, 0],
                scaled_points[:, 1],
                c="lime",
                s=3,
                marker=".",
                alpha=0.7,
            )
        dsum = density_sums[method]
        ax.set_title(f"{label}\nΣ = {dsum:.1f} (err={dsum - len(points):+.1f})", fontsize=9)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    return fig, list(density_sums.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="shha",
        choices=["shha", "shhb"],
        help="Dataset to visualize.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Image filenames (e.g., IMG_113.jpg IMG_47.jpg IMG_8.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/density_quality_comparison"),
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=900,
        help="Target image width in pixels.",
    )
    args = parser.parse_args()

    if args.dataset == "shha":
        data_root = Path("data/shanghaitech/part_A_final")
        cache_dirs = SHA_CACHE
    else:
        data_root = Path("data/shanghaitech/part_B_final")
        cache_dirs = SHB_CACHE

    # Default images: sparse, medium, dense
    if args.images is None:
        if args.dataset == "shha":
            args.images = ["IMG_113.jpg", "IMG_47.jpg", "IMG_8.jpg"]
        else:
            args.images = ["IMG_1.jpg", "IMG_80.jpg", "IMG_200.jpg"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating density map visualization for {args.dataset}...")
    for img_name in args.images:
        img_path = data_root / "test_data" / "images" / img_name
        if not img_path.exists():
            # Try test split
            img_path = data_root / "test" / "images" / img_name
        if not img_path.exists():
            print(f"  SKIP: image not found: {img_path}")
            continue

        print(f"  Processing {img_name}...")
        fig, sums = make_row(
            img_path,
            data_root,
            cache_dirs,
            target_width=args.target_width,
        )
        out_name = f"density_maps_visual_{args.dataset}_{img_path.stem}.png"
        out_path = args.output_dir / out_name
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path}")
        print(f"    Density sums: fixed={sums[0]:.1f}, geo={sums[1]:.1f}, hybrid={sums[2]:.1f}")


if __name__ == "__main__":
    main()

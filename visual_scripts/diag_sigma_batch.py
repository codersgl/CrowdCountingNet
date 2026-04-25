"""Dataset-level statistics for σ assignment strategies.

Aggregates per-image diagnostics from ``diag_sigma_strategies.py`` over many
images and reports mean ± std of:

  - Spearman ρ(σ, k-NN distance)         -> umbrella-principle alignment
  - Spearman ρ(σ, perspective)           -> depth dependence
  - σ summary stats (median, IQR)        -> sanity check on dynamic range

This is the dataset-level evidence needed before committing to a σ strategy
for full training: a single image's correlations could be coincidental.

Usage
-----
    python visual_scripts/diag_sigma_batch.py DATA_ROOT
    python visual_scripts/diag_sigma_batch.py DATA_ROOT --limit 100
    python visual_scripts/diag_sigma_batch.py DATA_ROOT \\
        --hybrid-alpha 0.5 --sigma-base 4.0 --output_csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import scipy.spatial
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Dataset-level σ-strategy statistics across many images"
)
parser.add_argument("data_root", type=str, help="Dataset root")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument(
    "--limit", type=int, default=0, help="Max number of images (0 = all)"
)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--sigma-base", type=float, default=4.0)
parser.add_argument("--min-sigma", type=float, default=0.5)
parser.add_argument("--max-sigma", type=float, default=30.0)
parser.add_argument(
    "--hybrid-alpha",
    type=float,
    default=0.5,
    help="α for the hybrid strategy column",
)
parser.add_argument("--output_csv", type=str, default=None, help="Per-image CSV dump")
parser.add_argument(
    "--min-points", type=int, default=8, help="Skip images with fewer in-bounds points"
)
args = parser.parse_args()

from crowdcount.data.prepare import (  # noqa: E402
    _depth_to_perspective,
    _find_image_gt_pairs,
    _load_points,
)

data_root = Path(args.data_root)
pairs = _find_image_gt_pairs(data_root, args.split)
if args.limit > 0:
    pairs = pairs[: args.limit]
if not pairs:
    sys.exit("No image/GT pairs found.")

persp_dir = data_root / "gt_perspective" / args.split
depth_dir = data_root / "gt_depth_maps" / args.split

STRATEGIES = [
    "geo",
    "persp",
    "persp_inv",
    f"hybrid(α={args.hybrid_alpha})",
    "max",
    "rss",
]


def _per_image_metrics(
    img_h: int, img_w: int, points: np.ndarray, persp_map: np.ndarray
) -> dict:
    """Return {strategy_name: {rho_knn, rho_persp, sigma_median, sigma_iqr}}.

    Returns an empty dict if the image is unusable (too few points, etc.).
    """
    rounded = np.round(points).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < img_w)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < img_h)
    )
    points = points[in_bounds]
    rounded = rounded[in_bounds]
    n = len(points)
    if n < args.min_points:
        return {}

    persp_pt = persp_map[rounded[:, 1], rounded[:, 0]].astype(np.float64)
    persp_pt = np.nan_to_num(persp_pt, nan=1.0, posinf=1.0, neginf=1.0)
    persp_pt = np.clip(persp_pt, 1e-3, None)

    tree = scipy.spatial.KDTree(points)
    dists, _ = tree.query(points, k=4)
    knn_mean = dists[:, 1:4].mean(axis=1)
    geo = (dists[:, 1] + dists[:, 2] + dists[:, 3]) * 0.1

    head = args.beta * args.sigma_base * persp_pt
    geo_safe = np.maximum(geo, 1e-6)
    head_safe = np.maximum(head, 1e-6)
    a = float(args.hybrid_alpha)

    sigmas = {
        "geo": geo,
        "persp": head,
        "persp_inv": args.beta * args.sigma_base / persp_pt,
        f"hybrid(α={a})": (head_safe ** (1.0 - a)) * (geo_safe**a),
        "max": np.maximum(head, geo),
        "rss": np.sqrt(head**2 + geo**2),
    }

    out = {}
    for name, sig in sigmas.items():
        sig = np.clip(sig, args.min_sigma, args.max_sigma)
        # spearmanr returns nan for constant inputs; guard it.
        if np.std(sig) < 1e-9 or np.std(knn_mean) < 1e-9:
            rho_knn = 0.0
        else:
            rho_knn, _ = spearmanr(sig, knn_mean)
        if np.std(sig) < 1e-9 or np.std(persp_pt) < 1e-9:
            rho_persp = 0.0
        else:
            rho_persp, _ = spearmanr(sig, persp_pt)
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        out[name] = {
            "rho_knn": float(rho_knn) if np.isfinite(rho_knn) else 0.0,
            "rho_persp": float(rho_persp) if np.isfinite(rho_persp) else 0.0,
            "sigma_median": float(q50),
            "sigma_iqr": float(q75 - q25),
            "n_points": n,
        }
    return out


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
import cv2

records: list[tuple[str, str, dict]] = []  # (img_stem, strategy, metrics)
n_used = 0
n_skipped = 0

for img_path, gt_path in tqdm(pairs, desc="images", unit="img"):
    img = cv2.imread(str(img_path))
    if img is None:
        n_skipped += 1
        continue
    H, W = img.shape[:2]

    persp_path = persp_dir / f"{img_path.stem}.npy"
    depth_path = depth_dir / f"{img_path.stem}.npy"
    if persp_path.exists():
        persp_map = np.load(str(persp_path)).astype(np.float32)
    elif depth_path.exists():
        persp_map = _depth_to_perspective(np.load(str(depth_path)))
    else:
        n_skipped += 1
        continue

    if persp_map.shape[:2] != (H, W):
        n_skipped += 1
        continue

    try:
        points = _load_points(gt_path).astype(np.float32)
    except Exception as e:
        logger.warning(f"{img_path.name}: failed to load GT ({e}); skipping")
        n_skipped += 1
        continue

    per_img = _per_image_metrics(H, W, points, persp_map)
    if not per_img:
        n_skipped += 1
        continue
    n_used += 1
    for name, m in per_img.items():
        records.append((img_path.stem, name, m))

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
print(f"\nProcessed {n_used} images (skipped {n_skipped}).")
if n_used == 0:
    sys.exit("No usable images.")

agg = {
    name: {"rho_knn": [], "rho_persp": [], "sigma_median": [], "sigma_iqr": []}
    for name in STRATEGIES
}
for _stem, name, m in records:
    if name not in agg:
        agg[name] = {
            "rho_knn": [],
            "rho_persp": [],
            "sigma_median": [],
            "sigma_iqr": [],
        }
    for k in ("rho_knn", "rho_persp", "sigma_median", "sigma_iqr"):
        agg[name][k].append(m[k])

print(f"\n=== Dataset-level σ-strategy statistics  (N = {n_used} images) ===")
print(
    f"\n{'strategy':<20}  {'ρ(σ, knn)':>16}  {'ρ(σ, persp)':>16}  "
    f"{'σ_median(px)':>14}  {'σ_IQR(px)':>12}"
)
print("-" * 86)
for name in STRATEGIES:
    if not agg[name]["rho_knn"]:
        continue
    rk = np.array(agg[name]["rho_knn"])
    rp = np.array(agg[name]["rho_persp"])
    sm = np.array(agg[name]["sigma_median"])
    si = np.array(agg[name]["sigma_iqr"])
    print(
        f"  {name:<18}  {rk.mean():+.3f} ± {rk.std():.3f}  "
        f"{rp.mean():+.3f} ± {rp.std():.3f}  "
        f"{sm.mean():>6.2f} ± {sm.std():.2f}  "
        f"{si.mean():>6.2f} ± {si.std():.2f}"
    )

print("\nInterpretation:")
print("  • ρ(σ, knn) close to +1  → strategy follows the umbrella principle.")
print("  • ρ(σ, persp) measures pure depth dependence; high values mean σ is")
print("    largely determined by depth alone (head-radius prior).")
print("  • σ_median + IQR sanity-check the actual pixel scale.")

# ---------------------------------------------------------------------------
# Optional CSV
# ---------------------------------------------------------------------------
if args.output_csv:
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "strategy",
                "n_points",
                "rho_knn",
                "rho_persp",
                "sigma_median",
                "sigma_iqr",
            ]
        )
        for stem, name, m in records:
            w.writerow(
                [
                    stem,
                    name,
                    m["n_points"],
                    f"{m['rho_knn']:.4f}",
                    f"{m['rho_persp']:.4f}",
                    f"{m['sigma_median']:.4f}",
                    f"{m['sigma_iqr']:.4f}",
                ]
            )
    print(f"\nPer-image CSV written to {args.output_csv}")

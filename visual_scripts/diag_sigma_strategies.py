"""Diagnose σ assignment strategies for adaptive density-map generation.

Question this script answers
----------------------------
For each annotated head, four candidate strategies assign a Gaussian σ:

  - geo       : σ = (d1 + d2 + d3) / 10   (k-NN with k=3, current default)
  - persp     : σ = β · sigma_base · persp     (current "perspective-guided")
  - persp_inv : σ = β · sigma_base / persp     (theory-corrected: far → big σ)
  - hybrid    : σ = persp^(1-α) · geo^α            (current "hybrid")
  - max       : σ = max(head_radius, geo)      (perspective as physical lower bound)
  - rss       : σ = sqrt(head_radius² + geo²)  (smooth max)

where ``head_radius = sigma_base · persp`` (median-normalised perspective ≈
relative head size). The "umbrella" principle from adaptive KDE says: σ should
correlate **positively** with local sparsity (i.e. with k-NN distance), and
should be **independent of depth alone** — depth only determines the *physical
floor* (head radius in pixels).

Outputs
-------
1. Per-point scatter: σ vs k-NN distance (sparsity) for each strategy.
2. Per-point scatter: σ vs perspective (depth proxy) for each strategy.
3. Image overlay: each point coloured by its σ, side-by-side per strategy.
4. Spearman correlations printed to stdout — the closer to +1 between σ and
   k-NN distance, the better the strategy follows the umbrella principle.

Usage
-----
    python visual_scripts/diag_sigma_strategies.py DATA_ROOT
    python visual_scripts/diag_sigma_strategies.py DATA_ROOT --image IMG_3
    python visual_scripts/diag_sigma_strategies.py DATA_ROOT \\
        --beta 1.0 --sigma-base 4.0 --output diag.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Diagnose σ assignment for adaptive density-map strategies"
)
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("--image", type=str, default=None, help="Image stem, e.g. IMG_3")
parser.add_argument(
    "--index", type=int, default=0, help="0-based image index if --image not given"
)
parser.add_argument(
    "--beta",
    type=float,
    default=1.0,
    help="perspective β (default 1.0 for diagnostics)",
)
parser.add_argument(
    "--sigma-base", type=float, default=4.0, help="σ anchor at median depth, in pixels"
)
parser.add_argument(
    "--min-sigma", type=float, default=0.5, help="lower clip applied to ALL strategies"
)
parser.add_argument(
    "--max-sigma", type=float, default=30.0, help="upper clip applied to ALL strategies"
)
parser.add_argument(
    "--hybrid-alpha",
    type=float,
    default=0.5,
    help="hybrid α (geometric blend weight); 0 = persp, 1 = geo",
)
parser.add_argument("--output", type=str, default=None, help="Save figure to path")
args = parser.parse_args()

# Imports after CLI so missing data deps surface clean errors.
from crowdcount.data.prepare import (  # noqa: E402
    _depth_to_perspective,
    _find_image_gt_pairs,
    _load_points,
)

# ---------------------------------------------------------------------------
# Locate image + GT
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
pairs = _find_image_gt_pairs(data_root, args.split)
if not pairs:
    sys.exit(f"No image/GT pairs found for split='{args.split}' in {data_root}")

if args.image:
    selected = [(ip, gp) for ip, gp in pairs if ip.stem == args.image]
    if not selected:
        sys.exit(f"Image '{args.image}' not found.")
    img_path, gt_path = selected[0]
else:
    idx = min(max(args.index, 0), len(pairs) - 1)
    img_path, gt_path = pairs[idx]

img = cv2.imread(str(img_path))
if img is None:
    sys.exit(f"Failed to read image: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

points = _load_points(gt_path).astype(np.float32)
in_bounds = (
    (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)
)
points = points[in_bounds]
N = len(points)
if N < 4:
    sys.exit(f"Need ≥4 in-bounds points for k-NN diagnosis; got {N}")

print(f"Image: {img_path.name}  ({W}×{H}), {N} in-bounds points")

# ---------------------------------------------------------------------------
# Perspective map (load if cached, else derive from depth)
# ---------------------------------------------------------------------------
persp_path = data_root / "gt_perspective" / args.split / f"{img_path.stem}.npy"
depth_path = data_root / "gt_depth_maps" / args.split / f"{img_path.stem}.npy"
if persp_path.exists():
    persp_map = np.load(str(persp_path)).astype(np.float32)
    print(f"loaded persp:  {persp_path}")
elif depth_path.exists():
    depth_map = np.load(str(depth_path))
    persp_map = _depth_to_perspective(depth_map)
    print(f"derived persp from depth: {depth_path}")
else:
    sys.exit(
        "No pre-generated depth or perspective maps found.\n"
        'Run: python -c "from crowdcount.data.prepare import generate_depth_maps; '
        f"generate_depth_maps('{args.data_root}', '{args.split}')\""
    )

if persp_map.shape[:2] != (H, W):
    sys.exit(f"persp shape {persp_map.shape[:2]} != image shape ({H}, {W})")

# ---------------------------------------------------------------------------
# Per-point quantities
# ---------------------------------------------------------------------------
rounded = np.round(points).astype(np.int64)
xs, ys = rounded[:, 0], rounded[:, 1]
persp_pt = persp_map[ys, xs].astype(np.float64)
persp_pt = np.nan_to_num(persp_pt, nan=1.0, posinf=1.0, neginf=1.0)
persp_pt = np.clip(persp_pt, 1e-3, None)

# k-NN distances (k=3 nearest neighbours, exclude self)
tree = scipy.spatial.KDTree(points)
dists, _ = tree.query(points, k=4)  # col 0 = self → 0
knn_mean = dists[:, 1:4].mean(axis=1)  # mean of nearest 3 neighbours
geo_sigma = (dists[:, 1] + dists[:, 2] + dists[:, 3]) * 0.1

# Strategies
sb = args.sigma_base
beta = args.beta
head_radius = beta * sb * persp_pt  # physical floor: head radius in px
sigma_persp = head_radius  # current "perspective-guided"
sigma_persp_inv = beta * sb / persp_pt  # direction-flipped
alpha = float(args.hybrid_alpha)
# Current hybrid_density formula: σ = persp^(1-α) · geo^α (weighted geometric mean).
# Use the median-normalised perspective directly (as prepare.hybrid_density does)
# — NOT the head_radius scaled by sigma_base — to faithfully mirror it.
sigma_hybrid = (persp_pt ** (1.0 - alpha)) * (geo_sigma**alpha)
sigma_max = np.maximum(head_radius, geo_sigma)
sigma_rss = np.sqrt(head_radius**2 + geo_sigma**2)

strategies: dict[str, np.ndarray] = {
    "geo": geo_sigma,
    "persp": sigma_persp,
    "persp_inv": sigma_persp_inv,
    f"hybrid (α={alpha:.2f})": sigma_hybrid,
    "max(head, geo)": sigma_max,
    "sqrt(head²+geo²)": sigma_rss,
}

# Apply common clip for fair comparison
clipped = {k: np.clip(v, args.min_sigma, args.max_sigma) for k, v in strategies.items()}

# ---------------------------------------------------------------------------
# Quantitative correlations
# ---------------------------------------------------------------------------
print("\n=== Spearman ρ (higher = better aligned with 'sparse → big σ') ===")
print(f"{'strategy':<22}  ρ(σ, knn_dist)   ρ(σ, persp)")
for name, sig in clipped.items():
    rho_knn, _ = spearmanr(sig, knn_mean)
    rho_persp, _ = spearmanr(sig, persp_pt)
    print(f"  {name:<20}  {rho_knn:+.3f}            {rho_persp:+.3f}")
print(
    "\nIdeal: ρ(σ, knn_dist) close to +1 (umbrella principle), "
    "ρ(σ, persp) close to 0 or weakly negative (depth alone shouldn't drive σ)."
)

# ---------------------------------------------------------------------------
# Figure: 3 rows × 5 cols
#  row 0: σ vs knn_mean (sparsity)
#  row 1: σ vs persp_pt (depth)
#  row 2: image overlay coloured by σ
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 6, figsize=(28, 13))
fig.suptitle(
    f"{img_path.stem}  N={N}  β={beta}  σ_base={sb}  α={alpha}  "
    f"clip=[{args.min_sigma}, {args.max_sigma}]",
    fontsize=12,
)

# Determine global σ range for colour scale
sig_vmin = min(s.min() for s in clipped.values())
sig_vmax = max(s.max() for s in clipped.values())

for col, (name, sig) in enumerate(clipped.items()):
    rho_knn, _ = spearmanr(sig, knn_mean)
    rho_persp, _ = spearmanr(sig, persp_pt)

    # Row 0: σ vs knn distance
    ax0 = axes[0, col]
    ax0.scatter(knn_mean, sig, s=6, alpha=0.55, c="tab:blue")
    ax0.set_xlabel("k-NN mean dist (sparsity)")
    ax0.set_ylabel("σ (px)")
    ax0.set_title(f"{name}\nρ(σ, knn) = {rho_knn:+.3f}")
    ax0.grid(alpha=0.3)

    # Row 1: σ vs perspective
    ax1 = axes[1, col]
    ax1.scatter(persp_pt, sig, s=6, alpha=0.55, c="tab:orange")
    ax1.set_xlabel("perspective (∝ 1/depth)")
    ax1.set_ylabel("σ (px)")
    ax1.set_title(f"ρ(σ, persp) = {rho_persp:+.3f}")
    ax1.grid(alpha=0.3)

    # Row 2: image overlay
    ax2 = axes[2, col]
    ax2.imshow(img_rgb)
    sc = ax2.scatter(
        points[:, 0],
        points[:, 1],
        c=sig,
        cmap="viridis",
        s=12,
        vmin=sig_vmin,
        vmax=sig_vmax,
        edgecolors="k",
        linewidths=0.2,
    )
    ax2.set_title(f"σ overlay  [{sig.min():.1f}, {sig.max():.1f}]")
    ax2.axis("off")
    plt.colorbar(sc, ax=ax2, fraction=0.046)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
if args.output:
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"\nFigure saved to {args.output}")
else:
    plt.show()

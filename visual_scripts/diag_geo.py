"""Diagnose geometry-adaptive density map: per-point sigma sanity check.

Usage:
    python visual_scripts/diag_geo.py data/shanghaitech/part_A_final
    python visual_scripts/diag_geo.py DATA_ROOT --image IMG_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial

parser = argparse.ArgumentParser(description="Diagnose geometry-adaptive sigma distribution")
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num-images", type=int, default=10, help="Images to sample for stats")
args = parser.parse_args()

from tqdm import tqdm

from crowdcount.data.prepare import _find_image_gt_pairs, _load_points, gaussian_filter_density

# ---------------------------------------------------------------------------
# Select images
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
pairs = _find_image_gt_pairs(data_root, args.split)
if not pairs:
    sys.exit("No image/GT pairs found")

if args.image:
    selected = [(ip, gp) for ip, gp in pairs if ip.stem == args.image]
    if not selected:
        sys.exit(f"Not found: {args.image}")
    pairs = selected
else:
    rng = np.random.RandomState(42)
    n = min(args.num_images, len(pairs))
    pairs = [pairs[i] for i in rng.choice(len(pairs), n, replace=False)]

# ---------------------------------------------------------------------------
# Collect per-point sigma values
# ---------------------------------------------------------------------------
all_sigmas: list[float] = []
all_densities: list[float] = []  # peak value at each point
all_nn_dists: list[list[float]] = []  # 3 nearest neighbor distances per point
image_stats: list[dict] = []

for img_path, gt_path in tqdm(pairs, desc="Analysing", unit="img"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    points_orig = _load_points(gt_path)
    H, W = img.shape[:2]

    rounded = np.round(points_orig).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < W)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < H)
    )
    points = points_orig[in_bounds]
    rounded = rounded[in_bounds]
    gt_count = len(points)
    if gt_count < 4:
        continue

    # Per-point sigma (same logic as gaussian_filter_density)
    tree = scipy.spatial.KDTree(points.copy(), leafsize=2048)
    distances, _ = tree.query(points, k=4)
    for i in range(gt_count):
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        all_sigmas.append(sigma)
        all_nn_dists.append([distances[i][1], distances[i][2], distances[i][3]])

    # Density peaks at each point (single-point gaussian)
    from scipy.ndimage import gaussian_filter

    for i in range(gt_count):
        pt2d = np.zeros([H, W], dtype=np.float32)
        pt2d[rounded[i, 1], rounded[i, 0]] = 1.0
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        g = gaussian_filter(pt2d, sigma, mode="constant")
        all_densities.append(g.max())

    image_stats.append({
        "name": img_path.stem,
        "gt": gt_count,
        "sigma_mean": np.mean([(distances[j][1]+distances[j][2]+distances[j][3])*0.1 for j in range(gt_count)]),
        "sigma_min": min([(distances[j][1]+distances[j][2]+distances[j][3])*0.1 for j in range(gt_count)]),
    })

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
sigmas = np.array(all_sigmas)
nn_dists = np.array(all_nn_dists)
peaks = np.array(all_densities)

print()
print("=" * 60)
print("  Geometry-Adaptive Density Map — Sigma Analysis")
print("=" * 60)
print(f"  Images analysed: {len(image_stats)}")
print(f"  Total points:    {len(sigmas)}")
print()
print(f"  Sigma stats:")
print(f"    mean   = {sigmas.mean():.2f}")
print(f"    median = {np.median(sigmas):.2f}")
print(f"    std    = {sigmas.std():.2f}")
print(f"    min    = {sigmas.min():.4f}")
print(f"    max    = {sigmas.max():.2f}")
print(f"    < 0.5   : {(sigmas < 0.5).sum()} pts ({(sigmas < 0.5).mean()*100:.1f}%)  ← too narrow!")
print(f"    < 1.0   : {(sigmas < 1.0).sum()} pts ({(sigmas < 1.0).mean()*100:.1f}%)")
print(f"    < 2.0   : {(sigmas < 2.0).sum()} pts ({(sigmas < 2.0).mean()*100:.1f}%)")
print(f"    > 10.0  : {(sigmas > 10.0).sum()} pts ({(sigmas > 10.0).mean()*100:.1f}%)")

# Nearest neighbour stats
d1 = nn_dists[:, 0]
d2 = nn_dists[:, 1]
d3 = nn_dists[:, 2]
print(f"\n  Nearest neighbour distances (pixels):")
print(f"    1-NN: mean={d1.mean():.1f}, median={np.median(d1):.1f}, min={d1.min():.1f}, max={d1.max():.1f}")
print(f"    2-NN: mean={d2.mean():.1f}, median={np.median(d2):.1f}, min={d2.min():.1f}")
print(f"    3-NN: mean={d3.mean():.1f}, median={np.median(d3):.1f}, min={d3.min():.1f}")

# Peak value at point center
print(f"\n  Peak density at point centers:")
print(f"    mean   = {peaks.mean():.6f}")
print(f"    median = {np.median(peaks):.6f}")
print(f"    max    = {peaks.max():.6f}")
# For a Gaussian centered at pixel, peak = 1/(2*pi*sigma^2)
# sigma=0.5 → peak≈0.637, sigma=1.0 → peak≈0.159, sigma=5.0 → peak≈0.006
sigma_at_max_peak = 1.0 / np.sqrt(2 * np.pi * peaks.max()) if peaks.max() > 0 else float("inf")
print(f"    sigma at max peak ≈ {sigma_at_max_peak:.3f}  (estimated from peak value)")

# Per-image summary
print(f"\n  Per-image sigma mean:")
for s in sorted(image_stats, key=lambda x: x["sigma_mean"]):
    flag = " *** SPIKE" if s["sigma_min"] < 0.5 else ""
    print(f"    {s['name']:<16}  gt={s['gt']:>5}  σ_mean={s['sigma_mean']:6.2f}  σ_min={s['sigma_min']:6.3f}{flag}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print()
print("-" * 60)
issues = []
if (sigmas < 0.5).mean() > 0.05:
    issues.append(f"{(sigmas < 0.5).mean()*100:.0f}% of points have sigma < 0.5 → density looks like point map")
if (sigmas < 1.0).mean() > 0.3:
    issues.append(f"{(sigmas < 1.0).mean()*100:.0f}% of points have sigma < 1.0 → narrow Gaussians")
if sigmas.std() > 5:
    issues.append(f"Sigma std={sigmas.std():.1f} is large → uneven smoothing across image")

if issues:
    print("  ISSUES FOUND:")
    for iss in issues:
        print(f"    ✗ {iss}")
    print()
    print("  Root cause: sigma = avg(3-NN distances) × 0.1, no minimum floor.")
    print("  Dense crowds → tiny NN distances → tiny sigma → 'point map' effect.")
    print("  Missing annotations → large NN distances → large sigma → over-smoothed.")
else:
    print("  No major issues found.")

print("-" * 60)

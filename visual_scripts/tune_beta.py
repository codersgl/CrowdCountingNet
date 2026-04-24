"""Tune beta × min_sigma by matching geometry-adaptive sigma distribution.

The key insight: gaussian_filter ALWAYS preserves sum (each point = 1.0),
so per-point sigma determines the shape, not the sum. The right metric is
how well perspective-guided sigmas match the geometry-adaptive distribution
that's proven effective for training.

Usage:
    python visual_scripts/tune_beta.py data/shanghaitech/part_A_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Tune beta × min_sigma by matching geo-adaptive sigma distribution"
)
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--num-images", type=int, default=20, help="Images to sample")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--beta-min", type=float, default=2.0)
parser.add_argument("--beta-max", type=float, default=12.0)
parser.add_argument("--beta-step", type=float, default=0.5)
parser.add_argument("--ms-min", type=float, default=1.0)
parser.add_argument("--ms-max", type=float, default=6.0)
parser.add_argument("--ms-step", type=float, default=0.5)
args = parser.parse_args()

from crowdcount.data.prepare import _depth_to_perspective, _find_image_gt_pairs, _load_points, gaussian_filter_density

# ---------------------------------------------------------------------------
# Load samples + collect per-point geo sigmas & perspective values
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
pairs = _find_image_gt_pairs(data_root, args.split)
if not pairs:
    sys.exit("No image/GT pairs found")

rng = np.random.RandomState(42)
n_sample = min(args.num_images, len(pairs))
sampled = [pairs[rng.choice(len(pairs))] for _ in range(n_sample)]

persp_dir = data_root / "gt_perspective" / args.split
depth_dir = data_root / "gt_depth_maps" / args.split
use_persp = persp_dir.is_dir() and any(persp_dir.iterdir())
if not use_persp and not (depth_dir.is_dir() and any(depth_dir.iterdir())):
    sys.exit("No perspective/depth maps. Run depth_map.py first.")

all_geo: list[float] = []  # geometry-adaptive sigmas
all_pv: list[float] = []   # perspective values at each point
all_sums: list[tuple[float, float, int]] = []  # (geo_sum, persp_sum, gt) per image for best params

for img_path, gt_path in tqdm(sampled, desc="Loading & computing geo sigmas"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    points = _load_points(gt_path)
    H, W = img.shape[:2]

    rounded = np.round(points).astype(np.int64)
    in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < W) & (rounded[:, 1] >= 0) & (rounded[:, 1] < H)
    points = points[in_bounds]
    rounded = rounded[in_bounds]
    gt_count = len(points)
    if gt_count < 4:
        continue

    # Geo sigmas
    tree = scipy.spatial.KDTree(points.copy(), leafsize=2048)
    distances, _ = tree.query(points, k=4)
    for i in range(gt_count):
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        all_geo.append(sigma)

    # Perspective values
    if use_persp:
        pmap = np.load(str(persp_dir / f"{img_path.stem}.npy"))
    else:
        dmap = np.load(str(depth_dir / f"{img_path.stem}.npy"))
        pmap = _depth_to_perspective(dmap)

    for i in range(gt_count):
        pv = float(pmap[rounded[i, 1], rounded[i, 0]])
        if np.isnan(pv) or np.isinf(pv):
            pv = 0.0
        all_pv.append(pv)

geo_arr = np.array(all_geo, dtype=np.float32)
pv_arr = np.array(all_pv, dtype=np.float32)
geo_median = float(np.median(geo_arr))

print(f"\n  Points: {len(geo_arr)}")
print(f"  Geo sigma:   median={geo_median:.1f}, mean={geo_arr.mean():.1f}, [{geo_arr.min():.1f}, {geo_arr.max():.1f}]")
print(f"  Persp value: median={np.median(pv_arr):.1f}, mean={pv_arr.mean():.1f}, [{pv_arr.min():.3f}, {pv_arr.max():.1f}]")
print()

# ---------------------------------------------------------------------------
# Coarse grid
# ---------------------------------------------------------------------------
beta_list = [round(b, 2) for b in np.arange(args.beta_min, args.beta_max + args.beta_step / 2, args.beta_step)]
ms_list = [round(m, 2) for m in np.arange(args.ms_min, args.ms_max + args.ms_step / 2, args.ms_step)]
total = len(beta_list) * len(ms_list)
print(f"Grid: beta={beta_list[0]}..{beta_list[-1]} × min_sigma={ms_list[0]}..{ms_list[-1]} = {total} combos\n")

results = []  # (score, beta, ms, avg_sigma, pct_within_factor2)
for beta in tqdm(beta_list, desc="Grid search"):
    for ms in ms_list:
        persp_sigmas = np.maximum(beta * pv_arr, ms)
        # Log-space MAE — penalises large multiplicative differences
        log_ratio = np.abs(np.log(persp_sigmas + 1e-6) - np.log(geo_arr + 1e-6))
        score = float(np.mean(log_ratio))
        avg_s = float(np.mean(persp_sigmas))
        within2 = float(np.mean((persp_sigmas >= geo_arr / 2.0) & (persp_sigmas <= geo_arr * 2.0)))
        results.append((score, beta, ms, avg_s, within2))

results.sort(key=lambda x: x[0])

# ---------------------------------------------------------------------------
# Top-15
# ---------------------------------------------------------------------------
print(f"\n{'Rank':<5} {'Beta':<8} {'min_s':<10} {'LogMAE':>8} {'Avg σp':>8} {'±2x geo':>10} {'Sum vs GT':>10}")
print("-" * 65)
# For top candidates, also compute sum conservation
from crowdcount.data.prepare import perspective_gaussian_filter_density

for rank, (score, beta, ms, avg_s, within2) in enumerate(results[:15], 1):
    # Sum check on a subset (3 images)
    sum_errs = []
    for img_path, _ in sampled[:3]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_path = [gp for ip, gp in pairs if ip.stem == img_path.stem][0] if any(True for _ in [1]) else None
        pts = _load_points(gt_path)
        if use_persp:
            pmap = np.load(str(persp_dir / f"{img_path.stem}.npy"))
        else:
            dmap = np.load(str(depth_dir / f"{img_path.stem}.npy"))
            pmap = _depth_to_perspective(dmap)
        dens = perspective_gaussian_filter_density(img, pts, pmap, beta=beta, min_sigma=ms)
        H2, W2 = img.shape[:2]
        rounded2 = np.round(pts).astype(np.int64)
        in_b2 = (rounded2[:, 0] >= 0) & (rounded2[:, 0] < W2) & (rounded2[:, 1] >= 0) & (rounded2[:, 1] < H2)
        sum_errs.append(abs(dens.sum() - int(in_b2.sum())))

    sum_mae = np.mean(sum_errs) if sum_errs else 0
    flag = " <<<" if rank == 1 else ""
    print(f"{rank:<5} {beta:<8.2f} {ms:<10.2f} {score:>8.3f} {avg_s:>8.1f} {within2:>9.1%} {sum_mae:>10.2f}{flag}")

# ---------------------------------------------------------------------------
# Best
# ---------------------------------------------------------------------------
best_score, best_beta, best_ms, best_avg, best_w2 = results[0]
persp_best = np.maximum(best_beta * pv_arr, best_ms)

print()
print("=" * 60)
print(f"  Recommended (sigma distribution matching):")
print(f"    beta      = {best_beta}")
print(f"    min_sigma = {best_ms}")
print(f"    Log-MAE   = {best_score:.3f}")
print(f"    Persp σ:    median={np.median(persp_best):.1f}, mean={persp_best.mean():.1f}")
print(f"    Geo σ:      median={geo_median:.1f}, mean={geo_arr.mean():.1f}")
print(f"    {best_w2*100:.0f}% of points within 2× of geo sigma")
print("=" * 60)
print()
print("# configs/data/shha.yaml →")
print("density_generation:")
print("  perspective_guided: true")
print(f"  beta: {best_beta}")
print(f"  min_sigma: {best_ms}")

"""Verify persisted density maps match on-the-fly generation.

Checks: gt_perspective/ vs depth→persp on-the-fly,
        gt_density_maps_persp/ vs perspective_gaussian_filter_density on-the-fly.

Usage:
    python visual_scripts/verify_density_match.py data/shanghaitech/part_A_final --image IMG_10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Verify density map consistency")
parser.add_argument("data_root", type=str)
parser.add_argument("--image", type=str, default="IMG_10")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--beta", type=float, default=5.0)
parser.add_argument("--min-sigma", type=float, default=1.5)
args = parser.parse_args()

from crowdcount.data.prepare import (
    _depth_to_perspective,
    _find_image_gt_pairs,
    _load_points,
    gaussian_filter_density,
    perspective_gaussian_filter_density,
)

data_root = Path(args.data_root)

# ---------------------------------------------------------------------------
# Step 1: compare perspective maps (persisted vs on-the-fly)
# ---------------------------------------------------------------------------
persp_path = data_root / "gt_perspective" / args.split / f"{args.image}.npy"
depth_path = data_root / "gt_depth_maps" / args.split / f"{args.image}.npy"

if depth_path.exists():
    depth_map = np.load(str(depth_path))
    persp_on_the_fly = _depth_to_perspective(depth_map, disparity_input=True)

if persp_path.exists():
    persp_persisted = np.load(str(persp_path))
    diffs = np.abs(persp_persisted - persp_on_the_fly)
    print(f"[Perspective Map] {args.image}")
    print(f"  Persisted:     {persp_path}")
    print(f"  On-the-fly:    _depth_to_perspective(depth, disparity_input=True)")
    print(f"  Max diff:      {diffs.max():.6f}")
    print(f"  Mean diff:     {diffs.mean():.6f}")
    if diffs.max() > 1e-5:
        print(f"  ✗ MISMATCH — delete {persp_path} and regenerate")
    else:
        print(f"  ✓ Match")
else:
    print(f"[Perspective Map] Not found at {persp_path}")

# ---------------------------------------------------------------------------
# Step 2: compare density maps (persisted vs on-the-fly)
# ---------------------------------------------------------------------------
dens_path = data_root / "gt_density_maps_persp" / args.split / f"{args.image}.npy"

pairs = _find_image_gt_pairs(data_root, args.split)
img_path = None
for ip, _ in pairs:
    if ip.stem == args.image:
        img_path = ip
        break

if img_path is None:
    sys.exit(f"Image {args.image} not found")

img = cv2.imread(str(img_path))
if img is None:
    sys.exit(f"Failed to read image")

points = _load_points(gt_path := [gp for ip, gp in pairs if ip.stem == args.image][0])

# Use the perspective map we verified above
if persp_path.exists() and persp_on_the_fly is not None:
    pmap = persp_on_the_fly
elif depth_path.exists():
    pmap = _depth_to_perspective(np.load(str(depth_path)))
else:
    sys.exit("No depth or perspective map found")

dens_on_the_fly = perspective_gaussian_filter_density(
    img, points, pmap, beta=args.beta, min_sigma=args.min_sigma
)

print()
if dens_path.exists():
    dens_persisted = np.load(str(dens_path))
    diffs = np.abs(dens_persisted - dens_on_the_fly)
    print(f"[Density Map] {args.image}")
    print(f"  Persisted:     {dens_path}")
    print(f"  On-the-fly:    perspective_gaussian_filter_density(img, pts, persp, beta={args.beta}, min_sigma={args.min_sigma})")
    print(f"  Persisted sum: {dens_persisted.sum():.2f}")
    print(f"  On-the-fly sum:{dens_on_the_fly.sum():.2f}")
    print(f"  Max diff:      {diffs.max():.6f}")
    print(f"  Mean diff:     {diffs.mean():.6f}")
    if diffs.max() > 1e-5:
        print(f"  ✗ MISMATCH — persisted density map differs from on-the-fly generation!")
        print(f"  Possible causes:")
        if diffs.mean() > 1:
            print(f"    - Generated with wrong beta/min_sigma (not {args.beta}/{args.min_sigma})")
        if (diffs > 100).any():
            print(f"    - Generated with old (inverted) perspective formula")
        print(f"  Fix: delete {dens_path.parent} and regenerate")
    else:
        print(f"  ✓ Match")
else:
    print(f"[Density Map] Not found at {dens_path}")

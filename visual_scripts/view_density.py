"""Quick viewer for pre-generated density maps — saves result as PNG.

Usage:
    python visual_scripts/view_density.py data/shanghaitech/part_A_final
    python visual_scripts/view_density.py DATA_ROOT --image IMG_1 --persp
    python visual_scripts/view_density.py DATA_ROOT --split test --index 5 -o out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize pre-generated density maps")
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default=None, help="Image stem, e.g. IMG_1")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("--index", type=int, default=0, help="0-based image index")
parser.add_argument("--persp", action="store_true", help="Show perspective-guided density maps")
parser.add_argument("--no-points", action="store_true", help="Hide GT point overlay")
parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG path (auto-generated if omitted)")
args = parser.parse_args()

from crowdcount.data.prepare import _find_image_gt_pairs, _load_points

# ---------------------------------------------------------------------------
# Determine density map directory
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
dens_key = "gt_density_maps_persp" if args.persp else "gt_density_maps"
dens_dir = data_root / dens_key / args.split
if not dens_dir.is_dir() or not any(dens_dir.iterdir()):
    mode = "perspective-guided" if args.persp else "geometry-adaptive"
    sys.exit(f"No {mode} density maps found at {dens_dir}")

# ---------------------------------------------------------------------------
# Pick image
# ---------------------------------------------------------------------------
pairs = _find_image_gt_pairs(data_root, args.split)
if not pairs:
    sys.exit(f"No image/GT pairs found for split='{args.split}' in {data_root}")

if args.image:
    selected = [(ip, gp) for ip, gp in pairs if ip.stem == args.image]
    if not selected:
        sys.exit(f"Image '{args.image}' not found. Available: {[p[0].stem for p in pairs[:10]]}...")
    img_path, gt_path = selected[0]
else:
    idx = min(args.index, len(pairs) - 1)
    img_path, gt_path = pairs[idx]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
npy_path = dens_dir / f"{img_path.stem}.npy"
if not npy_path.exists():
    sys.exit(f"Density map not found: {npy_path}")

img = cv2.imread(str(img_path))
if img is None:
    sys.exit(f"Failed to read: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

points = _load_points(gt_path)
in_bounds = (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)
valid_pts = points[in_bounds]
n_total, n_valid = len(points), int(in_bounds.sum())

density = np.load(str(npy_path))

# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------
print(f"Image:    {img_path.name}  ({W}x{H})")
print(f"GT:       {n_total} points ({n_valid} in-bounds)")
print(f"Density:  sum={density.sum():.1f}, max={density.max():.4f}, nonzero={(density > 0).sum():,}")
print(f"Source:   {npy_path}")

# ---------------------------------------------------------------------------
# Build canvas
# ---------------------------------------------------------------------------
d_min, d_max = density.min(), density.max()
if d_max - d_min > 1e-9:
    d_vis = ((density - d_min) / (d_max - d_min) * 255).astype(np.uint8)
else:
    d_vis = np.zeros_like(density, dtype=np.uint8)
d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)

alpha = 0.55
blend = cv2.addWeighted(img_rgb, 1.0 - alpha, d_color, alpha, 0)

if not args.no_points and n_valid > 0:
    for x, y in valid_pts:
        cv2.circle(blend, (int(x), int(y)), 2, (0, 255, 0), -1)

img_annot = img_rgb.copy()
if not args.no_points and n_valid > 0:
    for x, y in valid_pts:
        cv2.circle(img_annot, (int(x), int(y)), 2, (0, 255, 0), -1)

label_img = np.zeros((30, W, 3), dtype=np.uint8)
cv2.putText(label_img, f"Image + {n_valid} GT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

label_dens = np.zeros((30, W, 3), dtype=np.uint8)
cv2.putText(label_dens, f"Density (sum={density.sum():.1f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

label_blend = np.zeros((30, W, 3), dtype=np.uint8)
cv2.putText(label_blend, "Blended", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

top = np.vstack([label_img, img_annot])
mid = np.vstack([label_dens, d_color])
bot = np.vstack([label_blend, blend])
canvas = np.hstack([top, mid, bot])

max_display_w = 1920
if canvas.shape[1] > max_display_w:
    scale = max_display_w / canvas.shape[1]
    new_w = int(canvas.shape[1] * scale)
    new_h = int(canvas.shape[0] * scale)
    canvas = cv2.resize(canvas, (new_w, new_h))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = args.output or f"density_{img_path.stem}_{'persp' if args.persp else 'geo'}_{args.split}.png"
cv2.imwrite(out_path, canvas)
print(f"Saved to {out_path}")

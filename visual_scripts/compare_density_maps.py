"""Side-by-side comparison of geometry-adaptive vs perspective-guided density maps.

Usage:
    python visual_scripts/compare_density_maps.py data/shanghaitech/part_A_final
    python visual_scripts/compare_density_maps.py DATA_ROOT --image IMG_1 --beta 0.3 --min-sigma 1.0
    python visual_scripts/compare_density_maps.py DATA_ROOT --split test --index 5 --output compare.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compare geometry-adaptive vs perspective-guided density maps"
)
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default=None, help="Image stem, e.g. IMG_1")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("--index", type=int, default=0, help="Which image to show (0-based)")
parser.add_argument("--beta", type=float, default=0.3, help="Perspective sigma scaling")
parser.add_argument("--min-sigma", type=float, default=1.0, help="Minimum sigma floor")
parser.add_argument("--output", type=str, default=None, help="Save figure to path")
parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"])
parser.add_argument("--no-depth-model", action="store_true", help="Skip DepthAnythingV2; require pre-generated depth maps")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports (after args, so missing deps are reported early)
# ---------------------------------------------------------------------------
from crowdcount.data.prepare import (
    _depth_to_perspective,
    _find_image_gt_pairs,
    _load_points,
    gaussian_filter_density,
    perspective_gaussian_filter_density,
)

# ---------------------------------------------------------------------------
# Load image & GT points
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
pairs = _find_image_gt_pairs(data_root, args.split)

if not pairs:
    sys.exit(f"No image/GT pairs found for split='{args.split}' in {data_root}")

if args.image:
    # Find by stem
    selected = [(ip, gp) for ip, gp in pairs if ip.stem == args.image]
    if not selected:
        sys.exit(f"Image '{args.image}' not found. Available: {[p[0].stem for p in pairs[:10]]}...")
    img_path, gt_path = selected[0]
else:
    idx = min(args.index, len(pairs) - 1)
    img_path, gt_path = pairs[idx]

img = cv2.imread(str(img_path))
if img is None:
    sys.exit(f"Failed to read image: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
points = _load_points(gt_path)
in_bounds = (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)
n_total = len(points)
n_valid = int(in_bounds.sum())
print(f"Image: {img_path.name}  ({W}x{H})")
print(f"GT points: {n_total} total, {n_valid} in-bounds")

# ---------------------------------------------------------------------------
# Geometry-adaptive density map
# ---------------------------------------------------------------------------
print("Generating geometry-adaptive density map...")
dens_geo = gaussian_filter_density(img, points)

# ---------------------------------------------------------------------------
# Perspective map
# ---------------------------------------------------------------------------
# Look for pre-generated perspective map first, then depth map
persp_dir = data_root / "gt_perspective" / args.split
persp_path = persp_dir / f"{img_path.stem}.npy"

depth_dir = data_root / "gt_depth_maps" / args.split
depth_path = depth_dir / f"{img_path.stem}.npy"

if persp_path.exists():
    print(f"Loading perspective map: {persp_path}")
    persp_map = np.load(str(persp_path))
elif depth_path.exists():
    print(f"Converting depth map to perspective: {depth_path}")
    depth_map = np.load(str(depth_path))
    persp_map = _depth_to_perspective(depth_map)
elif not args.no_depth_model:
    print("Generating depth map with DepthAnythingV2 (may require GPU)...")
    import torch

    from crowdcount.plugins.depth_anything_v2.dpt import DepthAnythingV2

    encoder_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(**encoder_configs[args.encoder])
    weight_path = f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    if not os.path.exists(weight_path):
        sys.exit(f"DepthAnythingV2 checkpoint not found: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    depth_map = model.infer_image(img)
    persp_map = _depth_to_perspective(depth_map)
else:
    sys.exit(
        "No pre-generated depth or perspective maps found. "
        "Run generate_depth_maps() first or remove --no-depth-model."
    )

# ---------------------------------------------------------------------------
# Perspective-guided density map
# ---------------------------------------------------------------------------
print("Generating perspective-guided density map...")
dens_persp = perspective_gaussian_filter_density(
    img, points, persp_map, beta=args.beta, min_sigma=args.min_sigma
)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"{img_path.stem} — beta={args.beta}, min_sigma={args.min_sigma}  "
    f"({n_total} GT points, {n_valid} in-bounds)",
    fontsize=13,
)

# Row 0: image + density maps
ax_img = axes[0, 0]
ax_img.imshow(img_rgb)
if n_valid > 0:
    valid_pts = points[in_bounds.numpy() if hasattr(in_bounds, "numpy") else in_bounds]
    ax_img.scatter(valid_pts[:, 0], valid_pts[:, 1], c="lime", s=3, alpha=0.7)
ax_img.set_title(f"Image + {n_valid} GT points")
ax_img.axis("off")

ax_geo = axes[0, 1]
im_geo = ax_geo.imshow(dens_geo, cmap="jet")
ax_geo.set_title(f"Geometry-Adaptive\nsum={dens_geo.sum():.1f}, max={dens_geo.max():.4f}")
ax_geo.axis("off")
plt.colorbar(im_geo, ax=ax_geo, fraction=0.046)

ax_persp = axes[0, 2]
im_persp = ax_persp.imshow(dens_persp, cmap="jet")
ax_persp.set_title(f"Perspective-Guided\nsum={dens_persp.sum():.1f}, max={dens_persp.max():.4f}")
ax_persp.axis("off")
plt.colorbar(im_persp, ax=ax_persp, fraction=0.046)

# Row 1: perspective map, difference, zoomed diff
ax_pmap = axes[1, 0]
im_pmap = ax_pmap.imshow(persp_map, cmap="inferno")
ax_pmap.set_title(f"Perspective Map\nmedian={float(np.median(persp_map)):.2f}")
ax_pmap.axis("off")
plt.colorbar(im_pmap, ax=ax_pmap, fraction=0.046)

diff = dens_persp - dens_geo
vmax = max(abs(diff.min()), abs(diff.max()))
ax_diff = axes[1, 1]
im_diff = ax_diff.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
ax_diff.set_title(f"Difference (Persp − Geo)\nsum={diff.sum():.3f}, max|diff|={vmax:.4f}")
ax_diff.axis("off")
plt.colorbar(im_diff, ax=ax_diff, fraction=0.046)

# Central crop of difference for detail
cy, cx = H // 2, W // 2
crop_size = min(128, H, W)
y0, y1 = max(0, cy - crop_size // 2), min(H, cy + crop_size // 2)
x0, x1 = max(0, cx - crop_size // 2), min(W, cx + crop_size // 2)
diff_crop = diff[y0:y1, x0:x1]
vmax_c = max(abs(diff_crop.min()), abs(diff_crop.max()))
ax_zoom = axes[1, 2]
im_zoom = ax_zoom.imshow(diff_crop, cmap="coolwarm", vmin=-vmax_c, vmax=vmax_c)
ax_zoom.set_title(f"Diff zoom (center {crop_size}px)")
ax_zoom.axis("off")
plt.colorbar(im_zoom, ax=ax_zoom, fraction=0.046)

plt.tight_layout()

if args.output:
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
else:
    out_default = f"compare_density_{img_path.stem}.png"
    plt.savefig(out_default, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_default}")

plt.close()

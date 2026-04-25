"""Side-by-side comparison of geometry-adaptive, perspective-guided, and hybrid density maps.

Usage:
    python visual_scripts/compare_density_maps.py data/shanghaitech/part_A_final
    python visual_scripts/compare_density_maps.py DATA_ROOT --image IMG_1 --beta 0.3 --min-sigma 1.0
    python visual_scripts/compare_density_maps.py DATA_ROOT --split test --index 5 --output compare.png

The hybrid mode (persp × (geo/persp)^alpha) requires no beta tuning.
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
    description="Compare geometry-adaptive vs perspective-guided vs hybrid density maps"
)
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default=None, help="Image stem, e.g. IMG_1")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("--index", type=int, default=0, help="Which image to show (0-based)")
parser.add_argument("--beta", type=float, default=0.3, help="Perspective sigma scaling")
parser.add_argument("--min-sigma", type=float, default=1.0, help="Minimum sigma floor")
parser.add_argument("--hybrid-min-sigma", type=float, default=1.5, help="Hybrid min sigma floor")
parser.add_argument("--hybrid-max-sigma", type=float, default=None, help="Hybrid max sigma ceiling")
parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Density-modulation weight in [0,1]; 0=persp-only, 1=geo-only")
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
    hybrid_density,
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
# Perspective map
# ---------------------------------------------------------------------------
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
# Generate all three density maps
# ---------------------------------------------------------------------------
print("Generating geometry-adaptive density map...")
dens_geo = gaussian_filter_density(img, points)

print(f"Generating perspective-guided density map (beta={args.beta}, min_sigma={args.min_sigma})...")
dens_persp = perspective_gaussian_filter_density(
    img, points, persp_map, beta=args.beta, min_sigma=args.min_sigma
)

hms_str = f"alpha={args.hybrid_alpha}, min_sigma={args.hybrid_min_sigma}"
if args.hybrid_max_sigma is not None:
    hms_str += f", max_sigma={args.hybrid_max_sigma}"
print(f"Generating hybrid density map ({hms_str})...")
dens_hybrid = hybrid_density(
    img, points, persp_map,
    min_sigma=args.hybrid_min_sigma,
    max_sigma=args.hybrid_max_sigma,
    alpha=args.hybrid_alpha,
)

# ---------------------------------------------------------------------------
# Figure (3×3 layout)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle(
    f"{img_path.stem} — {n_total} GT points ({n_valid} in-bounds)\n"
    f"persp-guided: beta={args.beta}, min_sigma={args.min_sigma}  |  "
    f"hybrid: alpha={args.hybrid_alpha}, min_sigma={args.hybrid_min_sigma}, max_sigma={args.hybrid_max_sigma}",
    fontsize=11,
)

# Row 0: Image + 3 density maps
# [0,0] Image + GT points
ax_img = axes[0, 0]
ax_img.imshow(img_rgb)
if n_valid > 0:
    valid_pts = points[in_bounds.numpy() if hasattr(in_bounds, "numpy") else in_bounds]
    ax_img.scatter(valid_pts[:, 0], valid_pts[:, 1], c="lime", s=2, alpha=0.6)
ax_img.set_title(f"Image + GT points")
ax_img.axis("off")

# [0,1] Geometry-Adaptive
ax_geo = axes[0, 1]
im_geo = ax_geo.imshow(dens_geo, cmap="jet")
ax_geo.set_title(f"Geometry-Adaptive\nsum={dens_geo.sum():.1f}, max={dens_geo.max():.4f}")
ax_geo.axis("off")
plt.colorbar(im_geo, ax=ax_geo, fraction=0.046)

# [0,2] Perspective-Guided
ax_persp = axes[0, 2]
im_persp = ax_persp.imshow(dens_persp, cmap="jet")
ax_persp.set_title(f"Perspective-Guided\nsum={dens_persp.sum():.1f}, max={dens_persp.max():.4f}")
ax_persp.axis("off")
plt.colorbar(im_persp, ax=ax_persp, fraction=0.046)

# Row 1: Perspective Map + Hybrid + Diff(Geo vs Hybrid)
# [1,0] Perspective Map
ax_pmap = axes[1, 0]
im_pmap = ax_pmap.imshow(persp_map, cmap="inferno")
ax_pmap.set_title(f"Perspective Map\nmedian={float(np.median(persp_map)):.2f}")
ax_pmap.axis("off")
plt.colorbar(im_pmap, ax=ax_pmap, fraction=0.046)

# [1,1] Hybrid
ax_hybrid = axes[1, 1]
im_hybrid = ax_hybrid.imshow(dens_hybrid, cmap="jet")
ax_hybrid.set_title(f"Hybrid (persp·(geo/persp)^α)\nsum={dens_hybrid.sum():.1f}, max={dens_hybrid.max():.4f}")
ax_hybrid.axis("off")
plt.colorbar(im_hybrid, ax=ax_hybrid, fraction=0.046)

# [1,2] Diff: Geo vs Hybrid
diff_gh = dens_geo - dens_hybrid
vmax_gh = max(abs(diff_gh.min()), abs(diff_gh.max()), 1e-9)
ax_diff_gh = axes[1, 2]
im_diff_gh = ax_diff_gh.imshow(diff_gh, cmap="coolwarm", vmin=-vmax_gh, vmax=vmax_gh)
ax_diff_gh.set_title(f"Diff (Geo − Hybrid)\nsum={diff_gh.sum():.3f}, max|diff|={vmax_gh:.4f}")
ax_diff_gh.axis("off")
plt.colorbar(im_diff_gh, ax=ax_diff_gh, fraction=0.046)

# Row 2: Diff(Persp vs Geo) + Diff(Hybrid vs Persp) + Zoom Diff
# [2,0] Diff: Persp vs Geo
diff_pg = dens_persp - dens_geo
vmax_pg = max(abs(diff_pg.min()), abs(diff_pg.max()), 1e-9)
ax_diff_pg = axes[2, 0]
im_diff_pg = ax_diff_pg.imshow(diff_pg, cmap="coolwarm", vmin=-vmax_pg, vmax=vmax_pg)
ax_diff_pg.set_title(f"Diff (Persp − Geo)\nsum={diff_pg.sum():.3f}, max|diff|={vmax_pg:.4f}")
ax_diff_pg.axis("off")
plt.colorbar(im_diff_pg, ax=ax_diff_pg, fraction=0.046)

# [2,1] Diff: Hybrid vs Persp
diff_hp = dens_hybrid - dens_persp
vmax_hp = max(abs(diff_hp.min()), abs(diff_hp.max()), 1e-9)
ax_diff_hp = axes[2, 1]
im_diff_hp = ax_diff_hp.imshow(diff_hp, cmap="coolwarm", vmin=-vmax_hp, vmax=vmax_hp)
ax_diff_hp.set_title(f"Diff (Hybrid − Persp)\nsum={diff_hp.sum():.3f}, max|diff|={vmax_hp:.4f}")
ax_diff_hp.axis("off")
plt.colorbar(im_diff_hp, ax=ax_diff_hp, fraction=0.046)

# [2,2] Zoom: center crop diff (Geo vs Hybrid) for detail
cy, cx = H // 2, W // 2
crop_size = min(128, H, W)
y0, y1 = max(0, cy - crop_size // 2), min(H, cy + crop_size // 2)
x0, x1 = max(0, cx - crop_size // 2), min(W, cx + crop_size // 2)
diff_crop = diff_gh[y0:y1, x0:x1]
vmax_c = max(abs(diff_crop.min()), abs(diff_crop.max()), 1e-9)
ax_zoom = axes[2, 2]
im_zoom = ax_zoom.imshow(diff_crop, cmap="coolwarm", vmin=-vmax_c, vmax=vmax_c)
ax_zoom.set_title(f"Geo−Hybrid zoom (center {crop_size}px)")
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

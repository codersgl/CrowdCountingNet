"""Diagnose depth map orientation: are closer objects brighter or darker?

Save a PNG comparing depth, perspective, and the original image side-by-side,
with automatic sampling of top/bottom/center regions.

Usage:
    python visual_scripts/depth_map.py DATA_ROOT --image IMG_1  # generate depth first
    python visual_scripts/diag_depth.py DATA_ROOT --image IMG_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Check depth map close/far orientation")
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default="IMG_1")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()

data_root = Path(args.data_root)
depth_dir = data_root / "gt_depth_maps" / args.split
depth_path = depth_dir / f"{args.image}.npy"

if not depth_path.exists():
    sys.exit(f"Depth map not found: {depth_path}\n"
             f"Run: python visual_scripts/depth_map.py {args.data_root} --image {args.image} --split {args.split}")

from crowdcount.data.prepare import _depth_to_perspective

depth = np.load(str(depth_path))
persp = _depth_to_perspective(depth)

# Load image
pairs_loader = __import__("crowdcount.data.prepare", fromlist=["_find_image_gt_pairs"])
pair_func = getattr(pairs_loader, "_find_image_gt_pairs")
pairs = pair_func(data_root, args.split)
img_paths = {ip.stem: ip for ip, _ in pairs}
if args.image not in img_paths:
    sys.exit(f"Image not found: {args.image}")

img = cv2.imread(str(img_paths[args.image]))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
print(f"Image: {args.image} ({W}x{H})")
print(f"Depth:      min={depth.min():.4f}  max={depth.max():.4f}  median={np.median(depth):.4f}")
print(f"Persp:      min={persp.min():.4f}  max={persp.max():.4f}  median={np.median(persp):.4f}")
print()

# Automatic sampling: top 1/3 (sky/far) vs bottom 1/3 (ground/close)
top_d = depth[:H//3, :]
bot_d = depth[2*H//3:, :]
print(f"Top  1/3 depth mean: {top_d.mean():.4f}  (expect LARGER if depth=farther)")
print(f"Bot  1/3 depth mean: {bot_d.mean():.4f}  (expect SMALLER if depth=closer)")

top_p = persp[:H//3, :]
bot_p = persp[2*H//3:, :]
print(f"Top  1/3 persp mean: {top_p.mean():.4f}  (expect SMALLER = far)")
print(f"Bot  1/3 persp mean: {bot_p.mean():.4f}  (expect LARGER = close)")
print()

if bot_d.mean() < top_d.mean():
    print("✓ Depth orientation OK: bottom closer < top farther")
else:
    print("✗ WARNING: Depth appears INVERTED!")
    print("   Bottom (expected close) has larger depth than top (expected far).")
    print("   This means DepthAnythingV2 is outputting disparity (1/depth) instead of depth.")
    print("   → Fix: skip 1/x in _depth_to_perspective, use depth directly as perspective")
print()

# ---------------------------------------------------------------------------
# Visualisation — save to file
# ---------------------------------------------------------------------------
def norm_u8(x):
    xmin, xmax = x.min(), x.max()
    if xmax - xmin > 1e-9:
        return ((x - xmin) / (xmax - xmin) * 255).astype(np.uint8)
    return np.zeros_like(x, dtype=np.uint8)

# Build 3-column layout: IMG | DEPTH | PERSPECTIVE
depth_color = cv2.applyColorMap(norm_u8(depth), cv2.COLORMAP_INFERNO)
persp_color = cv2.applyColorMap(norm_u8(persp), cv2.COLORMAP_INFERNO)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
lbl_h = 30
label_bar = np.zeros((lbl_h, W * 3, 3), dtype=np.uint8)
cv2.putText(label_bar, "Original Image", (10, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(label_bar, "Depth (bright=far, dark=near)", (W + 10, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(label_bar, "Perspective (bright=close, dark=far)", (W * 2 + 10, 20), font, 0.5, (255, 255, 255), 1)

canvas = np.vstack([label_bar, np.hstack([img_rgb, depth_color, persp_color])])

# Draw horizontal lines at 1/3 and 2/3 on depth/perspective views for reference
y1, y2 = H // 3, 2 * H // 3
cv2.line(canvas, (W, lbl_h + y1), (3 * W, lbl_h + y1), (0, 255, 0), 2)
cv2.line(canvas, (W, lbl_h + y2), (3 * W, lbl_h + y2), (0, 255, 0), 2)

# Label the regions on the image column
cv2.putText(canvas, "TOP (far?)", (10, lbl_h + y1 - 10), font, 0.6, (0, 255, 0), 1)
cv2.putText(canvas, "BOTTOM (close?)", (10, lbl_h + y2 + 20), font, 0.6, (0, 255, 0), 1)

# ---------------------------------------------------------------------------
# Add a diagnostic strip at the bottom: sampled grid of values
# ---------------------------------------------------------------------------
grid_h = 120
grid = np.zeros((grid_h, W * 3, 3), dtype=np.uint8)
grid_samples = 8
sample_y = np.linspace(0, H - 1, grid_samples).astype(int)
sample_x = W // 2
for i, sy in enumerate(sample_y):
    d_val = depth[sy, sample_x]
    p_val = persp[sy, sample_x]
    text_y = 15 + i * (grid_h // grid_samples)
    cv2.putText(grid, f"y={sy:3d}: depth={d_val:.4f}, persp={p_val:.4f}",
                (10, text_y), font, 0.4, (255, 255, 255), 1)

canvas = np.vstack([canvas, grid])

# Save
out_path = args.output or f"diag_depth_{args.image}.png"
cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"Saved to {out_path}")
print()
print("Check the image:")
print("  - If close (bottom) people appear BRIGHT in the Perspective column → ✓ CORRECT")
print("  - If close (bottom) people appear DARK in the Perspective column → ✗ INVERTED")

"""Run DepthAnythingV2 on a single image or a whole split and save results.

Usage:
    # Single image (auto-locate via stem)
    python visual_scripts/depth_map.py data/shanghaitech/part_A_final --image IMG_1

    # Whole split
    python visual_scripts/depth_map.py data/shanghaitech/part_A_final --split train

    # Custom encoder / output dir
    python visual_scripts/depth_map.py DATA_ROOT --encoder vitl -o output_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from crowdcount.data.prepare import _find_image_gt_pairs
from crowdcount.plugins.depth_anything_v2.dpt import DepthAnythingV2

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate depth maps with DepthAnythingV2")
parser.add_argument("data_root", type=str, help="ShanghaiTech dataset root")
parser.add_argument("--image", type=str, default=None, help="Single image stem, e.g. IMG_1 (batch mode if omitted)")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"])
parser.add_argument("-o", "--output", type=str, default=None, help="Output dir (default: data_root/gt_depth_maps/<split>/)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_encoder_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

model = DepthAnythingV2(**_encoder_configs[args.encoder])
weight_path = f"checkpoints/depth_anything_v2_{args.encoder}.pth"
if not Path(weight_path).exists():
    sys.exit(f"Checkpoint not found: {weight_path}")
model.load_state_dict(torch.load(weight_path, map_location="cpu"))
model = model.to(_device).eval()
print(f"Loaded DepthAnythingV2-{args.encoder} on {_device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
out_dir = Path(args.output) if args.output else data_root / "gt_depth_maps" / args.split
out_dir.mkdir(parents=True, exist_ok=True)

pairs = _find_image_gt_pairs(data_root, args.split)
if not pairs:
    sys.exit(f"No image/GT pairs found for split='{args.split}' in {data_root}")

if args.image:
    selected = [(ip, gp) for ip, gp in pairs if ip.stem == args.image]
    if not selected:
        sys.exit(f"Image '{args.image}' not found. Available: {[p[0].stem for p in pairs[:10]]}...")
    pairs = selected

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
print(f"Processing {len(pairs)} image(s) — saving to {out_dir}")

n_generated = 0
for img_path, _gt_path in tqdm(pairs, desc="depth maps", unit="img"):
    npy_path = out_dir / f"{img_path.stem}.npy"
    preview_path = out_dir / f"{img_path.stem}.png"

    # --- Generate .npy if missing
    if not npy_path.exists():
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print(f"  Failed to read: {img_path.name}")
            continue

        depth = model.infer_image(raw_img)
        np.save(str(npy_path), depth.astype(np.float32))
        n_generated += 1

    # --- Generate preview PNG if missing
    if not preview_path.exists():
        depth = np.load(str(npy_path))
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-9:
            depth_vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(preview_path), depth_color)

print(f"Done — {n_generated} new .npy, rest previews synced → {out_dir}")

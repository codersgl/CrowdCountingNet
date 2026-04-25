"""Dump (features_pa, density, gt_points) cache for SHA test images.

Run once before the diag_*.py scripts:

    uv run python -m visual_scripts.diag_gcn.dump_features \
        --num-samples 50 --device cuda

Cache size: ~7 MB / image at 1024x768 (FP16 features + FP32 density).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from visual_scripts.diag_gcn.common import (
    CACHE_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_WEIGHTS,
    iter_test_samples,
    load_extractor,
    save_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples", type=int, default=50, help="number of test images"
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--clear", action="store_true", help="wipe cache before dumping"
    )
    args = parser.parse_args()

    if args.clear and CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.pt"):
            p.unlink()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[dump] device={device}")
    model = load_extractor(args.weights, device=device)

    t0 = time.time()
    n_done = 0
    for sample in tqdm(
        iter_test_samples(args.test_dir, num_samples=args.num_samples, seed=args.seed),
        total=args.num_samples,
    ):
        img = sample.image.to(device)
        with torch.no_grad():
            features_pa, density = model(img)
        save_cache(sample, features_pa, density)
        n_done += 1

    print(f"[dump] cached {n_done} samples → {CACHE_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

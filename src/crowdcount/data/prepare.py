"""Density map generation for ShanghaiTech dataset.

Generates ground-truth density maps using k-nearest-neighbor Gaussian kernels.
This logic is adapted from density_data_preparation/k_nearest_gaussian_kernel.py.

Supported dataset layouts
--------------------------
ShanghaiTech (original)::

    data_root/
      train_data/
        images/          ← IMG_xxx.jpg
        ground_truth/    ← GT_xxx.mat  (scipy.io, field 'image_info')
      test_data/
        images/
        ground_truth/

Flat layout (alternative)::

    data_root/
      images/            ← IMG_xxx.jpg
      ground_truth/      ← GT_xxx.mat  (or .txt: "x y" per line)

Generated maps are cached to::

    data_root/gt_density_maps/<split>/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.spatial
from loguru import logger
from scipy.ndimage import gaussian_filter


def gaussian_filter_density(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Generate a density map for a single image given crowd point annotations.

    Args:
        img: H×W (or H×W×C) image array — only shape is used.
        points: N×2 array of (x, y) ground-truth point annotations.

    Returns:
        density: H×W float32 density map.
    """
    img_shape = [img.shape[0], img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, _ = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.0
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(img_shape)) / 2.0 / 2.0
        density += gaussian_filter(pt2d, sigma, mode="constant")
    return density


def _load_points(gt_path: Path) -> np.ndarray:
    """Load point annotations from a .mat or .txt ground-truth file.

    .mat: ShanghaiTech format — ``mat['image_info'][0,0][0,0][0]`` gives an
          N×2 array of (x, y) coordinates.
    .txt: plain text, one "x y" pair per line.

    Returns:
        float32 ndarray of shape (N, 2).
    """
    if gt_path.suffix == ".mat":
        import scipy.io

        mat = scipy.io.loadmat(str(gt_path))
        # Standard ShanghaiTech field layout
        pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        return pts  # shape (N, 2)
    else:  # .txt fallback
        points = []
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
        return np.array(points, dtype=np.float32)


def _find_image_gt_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Discover (image_path, gt_path) pairs without any list file.

    Tries the following candidate image directories in order:
      1. data_root/<split>_data/images/
      2. data_root/images/

    GT files are located in the sibling ``ground_truth/`` directory and must
    match the image stem via ShanghaiTech's naming convention
    (``IMG_xxx.jpg`` ↔ ``GT_xxx.mat``) or share the same stem with a
    ``.mat`` / ``.txt`` extension.
    """
    # Candidate image directories
    candidates = [
        data_root / f"{split}_data" / "images",
        data_root / "images",
    ]
    img_dir: Path | None = next((p for p in candidates if p.is_dir()), None)
    if img_dir is None:
        raise FileNotFoundError(
            f"Cannot find images directory for split='{split}' under {data_root}. "
            f"Tried: {candidates}"
        )

    gt_dir = img_dir.parent / "ground_truth"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected ground_truth directory at {gt_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. "IMG_1"
        # ShanghaiTech naming possibilities (tried in order):
        #   IMG_xxx.jpg  <->  GT_IMG_xxx.mat   (Part-A / Part-B official)
        #   IMG_xxx.jpg  <->  GT_xxx.mat        (some re-packs)
        #   IMG_xxx.jpg  <->  GT_xxx.txt        (plain-text alternative)
        candidate_stems = [f"GT_{stem}", stem.replace("IMG_", "GT_", 1)]
        # de-duplicate while preserving order
        seen: set[str] = set()
        candidate_stems = [s for s in candidate_stems if not (s in seen or seen.add(s))]

        gt_path: Path | None = None
        for gt_stem in candidate_stems:
            for ext in (".mat", ".txt"):
                candidate = gt_dir / f"{gt_stem}{ext}"
                if candidate.exists():
                    gt_path = candidate
                    break
            if gt_path is not None:
                break

        if gt_path is None:
            logger.warning(f"No GT file found for {img_path.name}, skipping.")
            continue
        pairs.append((img_path, gt_path))
    return pairs


def generate_density_maps(data_root: str | Path, split: str = "train") -> None:
    """Generate density maps (.npy) for all images in a dataset split.

    No list file required — images and GT files are discovered automatically
    from the directory structure (see module docstring).

    Generated maps are saved to ``data_root/gt_density_maps/<split>/``.
    """
    import cv2
    from tqdm import tqdm

    data_root = Path(data_root)
    out_dir = data_root / "gt_density_maps" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_gt_pairs(data_root, split)
    logger.info(f"Generating density maps for split='{split}' ({len(pairs)} images)...")

    for img_path, gt_path in tqdm(pairs, desc=f"density maps [{split}]", unit="img"):
        out_path = out_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue

        points = _load_points(gt_path)
        density = gaussian_filter_density(img, points)
        np.save(str(out_path), density)

    logger.info(f"Density maps saved to {out_dir}")

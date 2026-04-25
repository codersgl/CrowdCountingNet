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


def _depth_to_perspective(
    depth_map: np.ndarray,
    epsilon: float = 1e-6,
    clip_range: tuple[float, float] = (0.01, 100.0),
    disparity_input: bool = True,
) -> np.ndarray:
    """Convert a depth/disparity map to a median-normalised perspective map.

    Perspective values are normalised so the median equals 1.0.
    Larger perspective → closer to camera → wider Gaussian kernel.

    DepthAnythingV2 outputs *disparity* (larger = closer), not metric depth.
    Use ``disparity_input=True`` for DepthAnythingV2, ``False`` for true depth.

    Args:
        depth_map: H×W float32 depth or disparity values.
        epsilon: Small positive floor to avoid division by zero.
        clip_range: (min, max) for clipping extreme perspective values.
        disparity_input: True if input is disparity (larger = closer, e.g.
            DepthAnythingV2). False if input is metric depth (larger = farther).

    Returns:
        H×W float32 perspective map.
    """
    safe = np.maximum(depth_map, epsilon)
    safe = np.nan_to_num(safe, nan=epsilon, posinf=1e6, neginf=epsilon)
    if disparity_input:
        # Disparity: larger = closer → use directly as perspective
        persp = safe.astype(np.float64)
    else:
        # True depth: larger = farther → perspective = 1 / depth
        persp = 1.0 / safe.astype(np.float64)
    median_val = float(np.median(persp))
    if median_val > 1e-9:
        persp = persp / median_val
    return np.clip(persp, clip_range[0], clip_range[1]).astype(np.float32)


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
    if len(points) == 0:
        return density

    # Pre-filter out-of-bound points (using rounded coordinates) BEFORE building
    # the KDTree, otherwise their distances would still corrupt other points'
    # sigma estimates via k-NN.
    rounded = np.round(points).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < img_shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < img_shape[0])
    )
    points = points[in_bounds]
    rounded = rounded[in_bounds]
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # Query min(4, gt_count) neighbours to avoid inf distances when gt_count < 4
    k_query = min(4, gt_count)
    distances, _ = tree.query(points, k=k_query)
    # Ensure distances is always 2-D even when k_query == 1
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]

    for i in range(gt_count):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[rounded[i, 1], rounded[i, 0]] = 1.0
        if gt_count >= 4:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        elif gt_count >= 2:
            # Only 1..2 valid neighbours available; use their mean × 0.3
            valid_dists = distances[i][1:]
            sigma = float(np.mean(valid_dists)) * 0.3
        else:
            # Single point: no neighbours available, fall back to image-size heuristic
            sigma = np.average(np.array(img_shape)) / 2.0 / 2.0
        density += gaussian_filter(pt2d, sigma, mode="constant")
    return density


def perspective_gaussian_filter_density(
    img: np.ndarray,
    points: np.ndarray,
    perspective_map: np.ndarray,
    beta: float = 0.3,
    min_sigma: float = 1.0,
) -> np.ndarray:
    """Generate a density map using a depth-derived perspective map for sigma.

    For each annotated point at (x, y), the Gaussian kernel sigma is::

        sigma_i = max(beta * perspective_map[y, x], min_sigma)

    This replaces k-NN geometry-adaptive sigmas with perspective-aware widths
    where closer pedestrians (larger perspective values) receive wider kernels.

    Args:
        img: H×W (or H×W×C) image array — only shape is used.
        points: N×2 array of (x, y) ground-truth point annotations.
        perspective_map: H×W float32, median-normalised perspective values
            (typically 1/depth per image).
        beta: Scaling factor applied to perspective value for sigma.
        min_sigma: Minimum sigma floor (pixels).

    Returns:
        density: H×W float32 density map.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive, got {min_sigma}")

    img_shape = [img.shape[0], img.shape[1]]
    if perspective_map.shape[:2] != tuple(img_shape):
        raise ValueError(
            f"perspective_map shape {perspective_map.shape[:2]} "
            f"does not match image shape {tuple(img_shape)}"
        )

    density = np.zeros(img_shape, dtype=np.float32)
    if len(points) == 0:
        return density

    # Filter out-of-bound points (same logic as gaussian_filter_density)
    rounded = np.round(points).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < img_shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < img_shape[0])
    )
    points = points[in_bounds]
    rounded = rounded[in_bounds]
    gt_count = len(points)
    if gt_count == 0:
        return density

    for i in range(gt_count):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        yi, xi = rounded[i, 1], rounded[i, 0]
        pt2d[yi, xi] = 1.0
        pval = float(perspective_map[yi, xi])
        if np.isnan(pval) or np.isinf(pval):
            pval = 0.0
        sigma = max(beta * pval, min_sigma)
        density += gaussian_filter(pt2d, sigma, mode="constant")
    return density


def hybrid_density(
    img: np.ndarray,
    points: np.ndarray,
    perspective_map: np.ndarray,
    min_sigma: float = 1.5,
    max_sigma: float | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Generate a density map combining geometry-adaptive k-NN with perspective.

    The formula treats *perspective as the foundation* (head-size prior) and
    uses the density component of *geo_sigma* as a modulation::

        sigma_i = clip(persp_i × (geo_sigma_i / persp_i) ^ alpha,
                       min_sigma, max_sigma)

    - ``persp_i``: median-normalised perspective — the "brush size"
      (head size in pixels, ∝ 1/depth).
    - ``geo_sigma_i``: k-NN geometry-adaptive sigma, which encodes both
      local crowd density AND implicit perspective (∝ 1/depth).
    - ``geo_sigma_i / persp_i``: cancels the shared 1/depth factor from
      *geo_sigma*, leaving an approximate "pure density" signal.

    The parameter ``alpha ∈ [0, 1]`` controls the density-modulation weight:

    ========  ==============================================
    alpha=0   ``sigma = persp`` — pure head-size prior
    alpha=0.5 ``sigma = sqrt(persp × geo_sigma)`` — balanced
    alpha=1   ``sigma = geo_sigma`` — pure geometry-adaptive
    ========  ==============================================

    At median depth (persp = 1.0), ``sigma = geo_sigma ^ alpha``.

    Args:
        img: H×W (or H×W×C) image array — only shape is used.
        points: N×2 array of (x, y) ground-truth point annotations.
        perspective_map: H×W float32, median-normalised perspective values.
        min_sigma: Minimum sigma floor in pixels (prevents degenerate
            spikes for very distant points).
        max_sigma: Optional upper bound; ``None`` for no limit.
        alpha: Density-modulation weight in [0, 1] (default 0.5).

    Returns:
        density: H×W float32 density map.
    """
    if min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive, got {min_sigma}")
    if max_sigma is not None and max_sigma <= 0:
        raise ValueError(f"max_sigma must be positive, got {max_sigma}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    img_shape = [img.shape[0], img.shape[1]]
    if perspective_map.shape[:2] != tuple(img_shape):
        raise ValueError(
            f"perspective_map shape {perspective_map.shape[:2]} "
            f"does not match image shape {tuple(img_shape)}"
        )

    density = np.zeros(img_shape, dtype=np.float32)
    if len(points) == 0:
        return density

    # Filter out-of-bound points
    rounded = np.round(points).astype(np.int64)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < img_shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < img_shape[0])
    )
    points = points[in_bounds]
    rounded = rounded[in_bounds]
    gt_count = len(points)
    if gt_count == 0:
        return density

    # --- Geometry-adaptive sigmas (same logic as gaussian_filter_density) ---
    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    k_query = min(4, gt_count)
    distances, _ = tree.query(points, k=k_query)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]

    geo_sigmas = np.empty(gt_count, dtype=np.float64)
    if gt_count >= 4:
        geo_sigmas = (distances[:, 1] + distances[:, 2] + distances[:, 3]) * 0.1
    elif gt_count >= 2:
        valid_dists = distances[:, 1:]
        geo_sigmas = np.mean(valid_dists, axis=1) * 0.3
    else:
        geo_sigmas[:] = float(np.average(np.array(img_shape))) / 2.0 / 2.0

    # --- Density modulation on perspective foundation ---
    # persp = head-size prior (brush size); geo_sigma/persp ≈ pure density
    persp_values = perspective_map[rounded[:, 1], rounded[:, 0]]
    persp_values = np.nan_to_num(persp_values, nan=1.0, posinf=1.0, neginf=1.0)
    persp = persp_values.astype(np.float64)

    density_factor = np.ones_like(persp)
    if alpha > 0:
        ratio = np.maximum(geo_sigmas, 1e-6) / np.maximum(persp, 1e-6)
        density_factor = ratio ** alpha

    sigmas = persp * density_factor
    sigmas = np.clip(sigmas, min_sigma, max_sigma or np.inf)

    # --- Render ---
    for i in range(gt_count):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[rounded[i, 1], rounded[i, 0]] = 1.0
        density += gaussian_filter(pt2d, sigmas[i], mode="constant")
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


def generate_density_maps(
    data_root: str | Path,
    split: str = "train",
    perspective_guided: bool = False,
    hybrid: bool = False,
    beta: float = 0.3,
    min_sigma: float = 1.0,
    hybrid_min_sigma: float = 1.5,
    hybrid_max_sigma: float | None = None,
    hybrid_alpha: float = 0.5,
    disparity_input: bool = True,
) -> None:
    """Generate density maps (.npy) for all images in a dataset split.

    No list file required — images and GT files are discovered automatically
    from the directory structure (see module docstring).

    Three generation modes (mutually exclusive; *hybrid* takes priority)::

        *geometry-adaptive* (default) — k-NN based sigma.
        *perspective-guided* — ``sigma = max(beta × perspective, min_sigma)``.
        *hybrid* — ``sigma = clip(persp × (geo_sigma/persp)^alpha, min, max)``.

    Args:
        data_root: Dataset root directory.
        split: Dataset split ('train' or 'test').
        perspective_guided: If True, use perspective-guided sigma.
        hybrid: If True, use geometry × perspective hybrid (takes priority).
        beta: Sigma scaling factor (only used when perspective_guided=True).
        min_sigma: Minimum sigma floor (only used when perspective_guided=True).
        hybrid_min_sigma: Floor for hybrid mode (default 1.5).
        hybrid_max_sigma: Optional ceiling for hybrid mode.
        hybrid_alpha: Density-modulation weight for hybrid mode (default 0.5).
        disparity_input: True if depth maps are disparity (DepthAnythingV2).
    """
    import cv2
    from tqdm import tqdm

    data_root = Path(data_root)

    # Resolve perspective directory (needed for perspective_guided and hybrid)
    _need_persp = perspective_guided or hybrid
    if _need_persp:
        persp_dir = data_root / "gt_perspective" / split
        if not persp_dir.is_dir() or not any(persp_dir.iterdir()):
            logger.info("Perspective maps not found, generating from depth maps...")
            generate_perspective_maps(data_root, split, disparity_input=disparity_input)

    if hybrid:
        out_dir = data_root / "gt_density_maps_hybrid" / split
        mode_label = "hybrid (geo × persp)"
    elif perspective_guided:
        out_dir = data_root / "gt_density_maps_persp" / split
        mode_label = "perspective-guided"
    else:
        out_dir = data_root / "gt_density_maps" / split
        mode_label = "geometry-adaptive"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_gt_pairs(data_root, split)
    logger.info(
        f"Generating {mode_label} density maps for split='{split}' "
        f"({len(pairs)} images)..."
    )

    for img_path, gt_path in tqdm(pairs, desc=f"density maps [{split}]", unit="img"):
        out_path = out_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue

        points = _load_points(gt_path)

        if hybrid:
            persp_path = persp_dir / f"{img_path.stem}.npy"
            if not persp_path.exists():
                logger.warning(f"Perspective map missing: {persp_path}, skipping")
                continue
            persp_map = np.load(str(persp_path))
            density = hybrid_density(
                img, points, persp_map,
                min_sigma=hybrid_min_sigma,
                max_sigma=hybrid_max_sigma,
                alpha=hybrid_alpha,
            )
        elif perspective_guided:
            persp_path = persp_dir / f"{img_path.stem}.npy"
            if not persp_path.exists():
                logger.warning(f"Perspective map missing: {persp_path}, skipping")
                continue
            persp_map = np.load(str(persp_path))
            density = perspective_gaussian_filter_density(
                img, points, persp_map, beta=beta, min_sigma=min_sigma
            )
        else:
            density = gaussian_filter_density(img, points)

        np.save(str(out_path), density)

    logger.info(f"Density maps saved to {out_dir}")


def generate_depth_maps(
    data_root: str | Path,
    split: str = "train",
    encoder: str = "vitb",
    weight_path: str | None = None,
) -> None:
    """Generate depth maps (.npy) for all images in a dataset split using DepthAnythingV2.

    Maps are saved at original image resolution as float32 (H×W) to::

        data_root/gt_depth_maps/<split>/<stem>.npy

    Args:
        data_root: Dataset root directory.
        split: Dataset split ('train' or 'test').
        encoder: DepthAnythingV2 encoder variant ('vits', 'vitb', 'vitl').
        weight_path: Path to the .pth checkpoint. Defaults to
            checkpoints/depth_anything_v2_{encoder}.pth relative to CWD.
    """
    import cv2
    import torch
    from tqdm import tqdm

    from crowdcount.plugins.depth_anything_v2.dpt import DepthAnythingV2

    data_root = Path(data_root)
    out_dir = data_root / "gt_depth_maps" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if weight_path is None:
        weight_path = f"checkpoints/depth_anything_v2_{encoder}.pth"

    _encoder_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }
    if encoder not in _encoder_configs:
        raise ValueError(
            f"Unknown encoder '{encoder}', choose from {list(_encoder_configs)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(**_encoder_configs[encoder])
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    logger.info(f"Loaded DepthAnythingV2-{encoder} from {weight_path}")

    pairs = _find_image_gt_pairs(data_root, split)
    logger.info(f"Generating depth maps for split='{split}' ({len(pairs)} images)...")

    for img_path, _ in tqdm(pairs, desc=f"depth maps [{split}]", unit="img"):
        out_path = out_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            continue

        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue

        depth = model.infer_image(raw_img)  # H×W float32 numpy array
        np.save(str(out_path), depth.astype(np.float32))

    logger.info(f"Depth maps saved to {out_dir}")


def generate_perspective_maps(
    data_root: str | Path,
    split: str = "train",
    disparity_input: bool = True,
) -> None:
    """Convert depth maps to perspective maps for perspective-guided density generation.

    Perspective maps are saved to ``data_root/gt_perspective/<split>/<stem>.npy``
    as H×W float32 arrays.

    Requires pre-generated depth maps in ``data_root/gt_depth_maps/<split>/``.
    Run ``generate_depth_maps()`` first if they do not exist.
    """
    from tqdm import tqdm

    data_root = Path(data_root)
    depth_dir = data_root / "gt_depth_maps" / split
    if not depth_dir.is_dir() or not any(depth_dir.iterdir()):
        raise FileNotFoundError(
            f"Depth maps not found at {depth_dir}. "
            f"Run generate_depth_maps(data_root, split='{split}') first."
        )

    out_dir = data_root / "gt_perspective" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.npy"))
    logger.info(
        f"Generating perspective maps for split='{split}' "
        f"({len(depth_files)} depth files)..."
    )

    for depth_path in tqdm(depth_files, desc=f"perspective maps [{split}]", unit="img"):
        out_path = out_dir / depth_path.name
        if out_path.exists():
            continue

        depth_map = np.load(str(depth_path))
        persp_map = _depth_to_perspective(depth_map, disparity_input=disparity_input)
        np.save(str(out_path), persp_map)

    logger.info(f"Perspective maps saved to {out_dir}")

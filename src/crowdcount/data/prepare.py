"""Density map generation for crowd-counting datasets.

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

UCF-QNRF (ECCV 2018)::

        data_root/
            Train/             ← img_0001.jpg + img_0001_ann.mat
            Test/              ← img_0001.jpg + img_0001_ann.mat

Generated maps are cached to::

    data_root/gt_density_maps/<split>/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.spatial
from loguru import logger

# scipy's default truncate radius for gaussian_filter; matches the implicit
# truncation used by the legacy full-image rendering path.
_GAUSS_TRUNCATE = 4.0


def _render_point_gaussian(
    density: np.ndarray,
    y: int,
    x: int,
    sigma: float,
    truncate: float = _GAUSS_TRUNCATE,
) -> None:
    """Add an isotropic Gaussian centred at (y, x) into ``density`` in-place.

    Numerically equivalent (within float32 rounding) to::

        impulse = np.zeros_like(density)
        impulse[y, x] = 1.0
        density += gaussian_filter(impulse, sigma, mode='constant',
                                   truncate=truncate)

    but only writes to a ``(2r+1) x (2r+1)`` patch with ``r = ceil(truncate*sigma)``,
    cutting cost from O(H*W) to O(sigma**2) per point.

    Boundary handling matches ``mode='constant'`` (kernel mass that falls off
    the image is implicitly discarded — i.e. NOT renormalised).
    """
    H, W = density.shape
    if sigma <= 0:
        if 0 <= y < H and 0 <= x < W:
            density[y, x] += 1.0
        return
    r = int(math.ceil(truncate * sigma))
    y0, y1 = max(0, y - r), min(H - 1, y + r)
    x0, x1 = max(0, x - r), min(W - 1, x + r)
    if y0 > y1 or x0 > x1:
        return
    ys = np.arange(y0, y1 + 1, dtype=np.float64) - y
    xs = np.arange(x0, x1 + 1, dtype=np.float64) - x
    inv_two_sig2 = 1.0 / (2.0 * sigma * sigma)
    ky = np.exp(-(ys * ys) * inv_two_sig2)
    kx = np.exp(-(xs * xs) * inv_two_sig2)
    # Normalise the *full* (untruncated) 1-D kernels so the resulting 2-D
    # kernel integrates to ~1 before boundary clipping (same convention as
    # scipy.ndimage.gaussian_filter on an impulse).
    full_ys = np.arange(-r, r + 1, dtype=np.float64)
    norm = np.exp(-(full_ys * full_ys) * inv_two_sig2).sum()
    ky = ky / norm
    kx = kx / norm
    patch = np.outer(ky, kx).astype(density.dtype, copy=False)
    density[y0 : y1 + 1, x0 : x1 + 1] += patch


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
        if gt_count >= 4:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        elif gt_count >= 2:
            # Only 1..2 valid neighbours available; use their mean × 0.3
            valid_dists = distances[i][1:]
            sigma = float(np.mean(valid_dists)) * 0.3
        else:
            # Single point: no neighbours available, fall back to image-size heuristic
            sigma = np.average(np.array(img_shape)) / 2.0 / 2.0
        _render_point_gaussian(
            density, int(rounded[i, 1]), int(rounded[i, 0]), float(sigma)
        )
    return density


def perspective_gaussian_filter_density(
    img: np.ndarray,
    points: np.ndarray,
    perspective_map: np.ndarray,
    beta: float = 0.3,
    min_sigma: float = 1.0,
    max_sigma: float | None = None,
    sigma_base: float = 1.0,
) -> np.ndarray:
    """Generate a density map using a depth-derived perspective map for sigma.

    For each annotated point at (x, y), the Gaussian kernel sigma is::

        sigma_i = clip(beta * sigma_base * perspective_map[y, x],
                       min_sigma, max_sigma)

    ``perspective_map`` is typically median-normalised (median = 1.0), so it
    is dimensionless. ``sigma_base`` lets the caller anchor the sigma at the
    *median depth* in pixels (e.g. set to the expected head-radius at the
    image's median depth). With ``sigma_base=1.0`` (default) the legacy
    behaviour ``sigma = beta * persp`` is preserved.

    Closer pedestrians (larger perspective values) receive wider kernels.

    Args:
        img: H×W (or H×W×C) image array — only shape is used.
        points: N×2 array of (x, y) ground-truth point annotations.
        perspective_map: H×W float32, median-normalised perspective values.
        beta: Multiplicative scaling factor on perspective.
        min_sigma: Minimum sigma floor (pixels).
        max_sigma: Optional sigma ceiling (pixels); ``None`` for no limit.
        sigma_base: Pixel-scale anchor at median depth (default 1.0 for
            backward compatibility).

    Returns:
        density: H×W float32 density map.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive, got {min_sigma}")
    if max_sigma is not None and max_sigma <= 0:
        raise ValueError(f"max_sigma must be positive, got {max_sigma}")
    if sigma_base <= 0:
        raise ValueError(f"sigma_base must be positive, got {sigma_base}")

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

    ceil = max_sigma if max_sigma is not None else np.inf
    for i in range(gt_count):
        yi, xi = int(rounded[i, 1]), int(rounded[i, 0])
        pval = float(perspective_map[yi, xi])
        if np.isnan(pval) or np.isinf(pval):
            pval = 0.0
        sigma = float(np.clip(beta * sigma_base * pval, min_sigma, ceil))
        _render_point_gaussian(density, yi, xi, sigma)
    return density


def hybrid_density(
    img: np.ndarray,
    points: np.ndarray,
    perspective_map: np.ndarray,
    min_sigma: float = 1.5,
    max_sigma: float | None = None,
    alpha: float = 0.5,
    sigma_base: float = 1.0,
) -> np.ndarray:
    """Generate a density map combining geometry-adaptive k-NN with perspective.

    Algebraically the formula is the *weighted geometric mean* of the two
    sigma sources, with a ``sigma_base`` (px) anchor restoring dimensional
    consistency::

        head_radius_i = sigma_base * persp_i           # pixels (head size at point i)
        sigma_i       = clip(head_radius_i ** (1 - alpha) * geo_sigma_i ** alpha,
                             min_sigma, max_sigma)

    Two equivalent intuitions:

    - **Perspective as foundation, density as modulation** ::

          sigma_i = persp_i * (geo_sigma_i / persp_i) ** alpha

      where ``persp_i / median(persp) ≈ 1/depth`` (dimensionless) acts as a
      relative head-size prior, and ``geo_sigma_i / persp_i`` removes the
      shared 1/depth factor from k-NN, leaving an approximate "pure density"
      signal.

    - **Geometric blend on a log scale** ::

          log(sigma_i) = (1 - alpha) * log(persp_i) + alpha * log(geo_sigma_i)

    The parameter ``alpha ∈ [0, 1]`` controls the geometry-adaptive weight:

    ========  ==============================================
    alpha=0   ``sigma = sigma_base * persp`` — pure head-radius prior
    alpha=0.5 ``sigma = sqrt(sigma_base * persp × geo_sigma)`` — balanced
    alpha=1   ``sigma = geo_sigma`` — pure geometry-adaptive
    ========  ==============================================

    NOTE on units: ``perspective_map`` is median-normalised (≈1.0 at median
    depth) and dimensionless. Multiplying by ``sigma_base`` (pixels) turns it
    into an estimated head radius, so the geometric mean with ``geo_sigma``
    (also pixels) is dimensionally consistent. Pick ``sigma_base`` close to
    the typical head radius at the dataset's median depth (e.g. 4 px on SHHA).

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
    if sigma_base <= 0:
        raise ValueError(f"sigma_base must be positive, got {sigma_base}")

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
    # head_radius_i = sigma_base * persp_i (pixels) is the head-size prior.
    # geo_sigma is in pixels too, so the weighted geometric mean is
    # dimensionally consistent for any alpha in [0, 1].
    persp_values = perspective_map[rounded[:, 1], rounded[:, 0]]
    persp_values = np.nan_to_num(persp_values, nan=1.0, posinf=1.0, neginf=1.0)
    head_radius = sigma_base * persp_values.astype(np.float64)
    head_radius = np.maximum(head_radius, 1e-6)
    geo_safe = np.maximum(geo_sigmas, 1e-6)

    if alpha <= 0:
        sigmas = head_radius
    elif alpha >= 1:
        sigmas = geo_safe
    else:
        sigmas = (head_radius ** (1.0 - alpha)) * (geo_safe**alpha)
    sigmas = np.clip(sigmas, min_sigma, max_sigma or np.inf)

    # --- Render ---
    for i in range(gt_count):
        _render_point_gaussian(
            density, int(rounded[i, 1]), int(rounded[i, 0]), float(sigmas[i])
        )
    return density


def _load_points(gt_path: Path) -> np.ndarray:
    """Load point annotations from a .mat or .txt ground-truth file.

    .mat: ShanghaiTech format — ``mat['image_info'][0,0][0,0][0]`` gives an
          N×2 array of (x, y) coordinates.
    .mat: UCF-QNRF format — ``mat['annPoints']`` gives an N×2 array of
        (x, y) coordinates.
    .txt: plain text, one "x y" pair per line.

    Returns:
        float32 ndarray of shape (N, 2).
    """
    if gt_path.suffix == ".mat":
        import scipy.io

        mat = scipy.io.loadmat(str(gt_path))
        if "annPoints" in mat:
            pts = np.asarray(mat["annPoints"], dtype=np.float32)
        elif "image_info" in mat:
            pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        else:
            keys = sorted(k for k in mat.keys() if not k.startswith("__"))
            raise KeyError(
                f"Unsupported .mat annotation format in {gt_path}; "
                f"expected 'image_info' or 'annPoints', got keys={keys}"
            )
        return pts.reshape(-1, 2)
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
      3. data_root/Train or data_root/Test (UCF-QNRF)

    GT files are located either in the sibling ``ground_truth/`` directory or
    next to the image (UCF-QNRF). They must match the image stem via supported
    naming conventions such as ``IMG_xxx.jpg`` ↔ ``GT_IMG_xxx.mat`` or
    ``img_xxxx.jpg`` ↔ ``img_xxxx_ann.mat``.
    """
    split_lut = {"train": "Train", "test": "Test", "val": "Test"}
    ucf_split = split_lut.get(split.lower(), split)
    candidates: list[tuple[Path, Path]] = [
        (
            data_root / f"{split}_data" / "images",
            data_root / f"{split}_data" / "ground_truth",
        ),
        (data_root / "images", data_root / "ground_truth"),
        (data_root / ucf_split, data_root / ucf_split),
    ]
    found: tuple[Path, Path] | None = next(
        ((img_p, gt_p) for img_p, gt_p in candidates if img_p.is_dir()), None
    )
    img_dir: Path | None = found[0] if found is not None else None
    if img_dir is None:
        raise FileNotFoundError(
            f"Cannot find images directory for split='{split}' under {data_root}. "
            f"Tried: {[p for p, _ in candidates]}"
        )

    gt_dir = found[1]
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected ground_truth directory at {gt_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. "IMG_1"
        # ShanghaiTech naming possibilities (tried in order):
        #   IMG_xxx.jpg  <->  GT_IMG_xxx.mat   (Part-A / Part-B official)
        #   IMG_xxx.jpg  <->  GT_xxx.mat        (some re-packs)
        #   IMG_xxx.jpg  <->  GT_xxx.txt        (plain-text alternative)
        #   img_xxxx.jpg <->  img_xxxx_ann.mat  (UCF-QNRF official)
        candidate_stems = [
            f"{stem}_ann",
            f"GT_{stem}",
            stem.replace("IMG_", "GT_", 1),
            stem,
        ]
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


def _format_param_tag(value: float | None) -> str:
    """Compact, filename-safe tag for a numeric param (e.g. 0.5 -> '0p50', None -> 'inf')."""
    if value is None:
        return "inf"
    return f"{value:.2f}".replace(".", "p")


def _density_cache_dir(
    data_root: Path,
    split: str,
    *,
    perspective_guided: bool,
    hybrid: bool,
    beta: float,
    min_sigma: float,
    sigma_base: float,
    persp_max_sigma: float | None,
    hybrid_min_sigma: float,
    hybrid_max_sigma: float | None,
    hybrid_alpha: float,
) -> tuple[Path, str, dict]:
    """Build a parameter-aware cache directory for density maps.

    Encoding the relevant hyperparameters in the directory name avoids the
    silent-cache-hit pitfall where switching e.g. ``hybrid_alpha`` would
    otherwise reuse stale ``.npy`` files. A ``meta.json`` is also written
    inside the directory for human-readable provenance.

    Returns:
        (out_dir, mode_label, meta_dict)
    """
    if hybrid:
        tag = (
            f"a{_format_param_tag(hybrid_alpha)}"
            f"_sb{_format_param_tag(sigma_base)}"
            f"_min{_format_param_tag(hybrid_min_sigma)}"
            f"_max{_format_param_tag(hybrid_max_sigma)}"
        )
        out_dir = data_root / f"gt_density_maps_hybrid_{tag}" / split
        mode_label = "hybrid (geo \u00d7 persp)"
        meta = {
            "mode": "hybrid",
            "alpha": hybrid_alpha,
            "sigma_base": sigma_base,
            "min_sigma": hybrid_min_sigma,
            "max_sigma": hybrid_max_sigma,
        }
    elif perspective_guided:
        tag = (
            f"b{_format_param_tag(beta)}"
            f"_sb{_format_param_tag(sigma_base)}"
            f"_min{_format_param_tag(min_sigma)}"
            f"_max{_format_param_tag(persp_max_sigma)}"
        )
        out_dir = data_root / f"gt_density_maps_persp_{tag}" / split
        mode_label = "perspective-guided"
        meta = {
            "mode": "perspective_guided",
            "beta": beta,
            "sigma_base": sigma_base,
            "min_sigma": min_sigma,
            "max_sigma": persp_max_sigma,
        }
    else:
        out_dir = data_root / "gt_density_maps" / split
        mode_label = "geometry-adaptive"
        meta = {"mode": "geometry_adaptive"}
    return out_dir, mode_label, meta


def _resolve_density_cache_dir(
    data_root: str | Path,
    split: str,
    *,
    perspective_guided: bool = False,
    hybrid: bool = False,
    beta: float = 0.3,
    min_sigma: float = 1.0,
    sigma_base: float = 1.0,
    persp_max_sigma: float | None = None,
    hybrid_min_sigma: float = 1.5,
    hybrid_max_sigma: float | None = None,
    hybrid_alpha: float = 0.5,
) -> Path:
    """Public helper: returns the cache directory consumers should read from.

    Mirrors the directory-naming logic of :func:`generate_density_maps` so
    callers (e.g. the dataset class) can locate the correct ``.npy`` files
    for a given hyperparameter set without re-implementing the convention.
    """
    out_dir, _, _ = _density_cache_dir(
        Path(data_root),
        split,
        perspective_guided=perspective_guided,
        hybrid=hybrid,
        beta=beta,
        min_sigma=min_sigma,
        sigma_base=sigma_base,
        persp_max_sigma=persp_max_sigma,
        hybrid_min_sigma=hybrid_min_sigma,
        hybrid_max_sigma=hybrid_max_sigma,
        hybrid_alpha=hybrid_alpha,
    )
    return out_dir


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
    sigma_base: float = 1.0,
    persp_max_sigma: float | None = None,
) -> None:
    """Generate density maps (.npy) for all images in a dataset split.

    No list file required — images and GT files are discovered automatically
    from the directory structure (see module docstring).

    Three generation modes (mutually exclusive; *hybrid* takes priority)::

        *geometry-adaptive* (default) — k-NN based sigma.
        *perspective-guided* — ``sigma = clip(beta·sigma_base·persp, min, max)``.
        *hybrid* — ``sigma = clip(persp^(1-α) · geo_sigma^α, min, max)``.

    Hyperparameters that affect the rendered density are encoded in the
    output directory name (e.g. ``gt_density_maps_hybrid_a0p50_min1p50_maxinf``)
    so that switching parameters does not silently reuse a stale cache.

    Args:
        data_root: Dataset root directory.
        split: Dataset split ('train' or 'test').
        perspective_guided: If True, use perspective-guided sigma.
        hybrid: If True, use geometry × perspective hybrid (takes priority).
        beta: Sigma scaling factor (perspective_guided mode).
        min_sigma: Minimum sigma floor (perspective_guided mode).
        sigma_base: Pixel-scale anchor at median depth (perspective_guided).
        persp_max_sigma: Optional sigma ceiling (perspective_guided).
        hybrid_min_sigma: Floor for hybrid mode (default 1.5).
        hybrid_max_sigma: Optional ceiling for hybrid mode.
        hybrid_alpha: Geometry-adaptive weight for hybrid mode (default 0.5).
        disparity_input: True if depth maps are disparity (DepthAnythingV2).
    """
    import cv2
    from tqdm import tqdm

    data_root = Path(data_root)

    # Resolve perspective directory (needed for perspective_guided and hybrid)
    _need_persp = perspective_guided or hybrid
    persp_dir: Path | None = None
    if _need_persp:
        persp_dir = data_root / "gt_perspective" / split
        if not persp_dir.is_dir() or not any(persp_dir.iterdir()):
            logger.info("Perspective maps not found, generating from depth maps...")
            generate_perspective_maps(data_root, split, disparity_input=disparity_input)

    out_dir, mode_label, meta = _density_cache_dir(
        data_root,
        split,
        perspective_guided=perspective_guided,
        hybrid=hybrid,
        beta=beta,
        min_sigma=min_sigma,
        sigma_base=sigma_base,
        persp_max_sigma=persp_max_sigma,
        hybrid_min_sigma=hybrid_min_sigma,
        hybrid_max_sigma=hybrid_max_sigma,
        hybrid_alpha=hybrid_alpha,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop a meta.json with the exact params used for this cache (overwrites
    # are fine \u2014 cache dirs are pinned to params via their name).
    meta.update({"split": split, "disparity_input": disparity_input})
    try:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    except OSError as e:
        logger.warning(f"Could not write {out_dir / 'meta.json'}: {e}")

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
            assert persp_dir is not None
            persp_path = persp_dir / f"{img_path.stem}.npy"
            if not persp_path.exists():
                logger.warning(f"Perspective map missing: {persp_path}, skipping")
                continue
            persp_map = np.load(str(persp_path))
            density = hybrid_density(
                img,
                points,
                persp_map,
                min_sigma=hybrid_min_sigma,
                max_sigma=hybrid_max_sigma,
                alpha=hybrid_alpha,
                sigma_base=sigma_base,
            )
        elif perspective_guided:
            assert persp_dir is not None
            persp_path = persp_dir / f"{img_path.stem}.npy"
            if not persp_path.exists():
                logger.warning(f"Perspective map missing: {persp_path}, skipping")
                continue
            persp_map = np.load(str(persp_path))
            density = perspective_gaussian_filter_density(
                img,
                points,
                persp_map,
                beta=beta,
                min_sigma=min_sigma,
                sigma_base=sigma_base,
                max_sigma=persp_max_sigma,
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

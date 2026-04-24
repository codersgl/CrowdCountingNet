"""Image transforms for crowd counting."""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch


class DeNormalize:
    """Reverse ImageNet normalization."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def density_resize_stride8(density: torch.Tensor, stride: int = 8) -> torch.Tensor:
    """Reduce a full-resolution density map by summing within stride x stride blocks.

    Args:
        density: tensor of shape ``[B, 1, H, W]`` where H, W are divisible by ``stride``.
        stride: block size to sum over (default 8, matching PA-FPN output stride).

    Returns:
        Tensor of shape ``[B, 1, H // stride, W // stride]``.
    """
    if density.ndim != 4:
        raise ValueError(f"density must be 4D, got shape {tuple(density.shape)}")
    B, C, H, W = density.shape
    if H % stride != 0 or W % stride != 0:
        raise ValueError(
            f"density spatial size ({H},{W}) must be divisible by stride={stride}"
        )
    Ht, Wt = H // stride, W // stride
    return density.reshape(B, C, Ht, stride, Wt, stride).sum(dim=(3, 5))


def _make_feather_mask(
    h: int, w: int, sigma: float, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Build an h x w 2D Gaussian-feathered alpha mask peaking at 1 in the centre."""
    if sigma <= 0:
        return torch.ones((h, w), device=device, dtype=dtype)
    yy = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2.0
    xx = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
    gy = torch.exp(-(yy**2) / (2.0 * sigma**2))
    gx = torch.exp(-(xx**2) / (2.0 * sigma**2))
    mask = gy[:, None] * gx[None, :]
    mx = mask.max()
    if mx > 0:
        mask = mask / mx
    return mask


def pick_window_by_point_count(
    points: torch.Tensor,
    img_h: int,
    img_w: int,
    win_h: int,
    win_w: int,
    mode: str = "max",
    n_candidates: int = 16,
    align_to: int = 1,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Pick a (y, x) top-left window position whose contained point count is
    maximised (``mode='max'``) or minimised (``mode='min'``).

    Strategy: sample ``n_candidates`` random aligned positions and pick the
    extremum. Cheap and sufficient for our augmentation use-case.

    Args:
        points: tensor of shape ``[N, 2]`` with (x, y) coordinates in image space.
        img_h, img_w: image height / width.
        win_h, win_w: window height / width.
        mode: ``'max'`` or ``'min'``.
        n_candidates: number of random positions to sample.
        align_to: snap top-left corner to a multiple of ``align_to`` (e.g. stride).
        rng: optional ``random.Random`` instance for deterministic tests.

    Returns:
        ``(y, x)`` top-left corner of the chosen window.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
    if win_h > img_h or win_w > img_w:
        raise ValueError(
            f"window ({win_h},{win_w}) larger than image ({img_h},{img_w})"
        )
    rng = rng if rng is not None else random
    max_y = img_h - win_h
    max_x = img_w - win_w

    def _snap(v: int, hi: int) -> int:
        v = (v // align_to) * align_to
        return min(v, (hi // align_to) * align_to)

    best_y, best_x = 0, 0
    best_count = -1 if mode == "max" else math.inf
    n_pts = points.shape[0]
    if n_pts == 0:
        # Any position is equally good; pick a random aligned one.
        y = _snap(rng.randint(0, max_y), max_y)
        x = _snap(rng.randint(0, max_x), max_x)
        return y, x

    px = points[:, 0]
    py = points[:, 1]
    for _ in range(n_candidates):
        y = _snap(rng.randint(0, max_y), max_y)
        x = _snap(rng.randint(0, max_x), max_x)
        inside = (px >= x) & (px < x + win_w) & (py >= y) & (py < y + win_h)
        cnt = int(inside.sum().item())
        if (mode == "max" and cnt > best_count) or (mode == "min" and cnt < best_count):
            best_count = cnt
            best_y, best_x = y, x
            if mode == "min" and cnt == 0:
                break
    return best_y, best_x


# ---------------------------------------------------------------------------
# A1: RandomErasingCount
# ---------------------------------------------------------------------------


class RandomErasingCount:
    """Erase a random rectangle from image / density / depth and drop any points
    falling inside the erased region. Preserves the invariant that density.sum()
    of the erased patch corresponds to the remaining points (within the
    augmentation's own approximation -- the source density was already smoothed).
    """

    def __init__(
        self,
        prob: float = 0.5,
        scale_range: Sequence[float] = (0.02, 0.2),
        ratio_range: Sequence[float] = (0.3, 3.3),
        fill: float = 0.0,
        max_attempts: int = 10,
    ):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0,1], got {prob}")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"invalid scale_range {scale_range}")
        if ratio_range[0] <= 0 or ratio_range[0] > ratio_range[1]:
            raise ValueError(f"invalid ratio_range {ratio_range}")
        self.prob = float(prob)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.ratio_range = (float(ratio_range[0]), float(ratio_range[1]))
        self.fill = float(fill)
        self.max_attempts = int(max_attempts)

    def __call__(
        self,
        img: torch.Tensor,
        points: torch.Tensor,
        density: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply erasing in-place on the provided tensors.

        Args:
            img: ``[C, H, W]`` (will be modified in place).
            points: ``[N, 2]`` (x, y) image-space coordinates.
            density: ``[1, H, W]`` full-resolution density map (modified in place).
            depth: optional ``[1, H, W]`` depth map (modified in place).

        Returns:
            ``(img, points_filtered, density, depth)``. Image / density / depth
            are the same tensors mutated in place; points may be a new tensor
            with rows removed.
        """
        if random.random() > self.prob:
            return img, points, density, depth
        H, W = img.shape[-2:]
        area = H * W
        for _ in range(self.max_attempts):
            target_area = area * random.uniform(*self.scale_range)
            ratio = random.uniform(*self.ratio_range)
            h = int(round(math.sqrt(target_area * ratio)))
            w = int(round(math.sqrt(target_area / ratio)))
            if 0 < h < H and 0 < w < W:
                y0 = random.randint(0, H - h)
                x0 = random.randint(0, W - w)
                img[..., y0 : y0 + h, x0 : x0 + w] = self.fill
                density[..., y0 : y0 + h, x0 : x0 + w] = 0
                if depth is not None:
                    depth[..., y0 : y0 + h, x0 : x0 + w] = 0
                if points.numel() > 0:
                    inside = (
                        (points[:, 0] >= x0)
                        & (points[:, 0] < x0 + w)
                        & (points[:, 1] >= y0)
                        & (points[:, 1] < y0 + h)
                    )
                    if inside.any():
                        points = points[~inside]
                return img, points, density, depth
        return img, points, density, depth


# ---------------------------------------------------------------------------
# B3: CopyPasteDense (helpers)
# ---------------------------------------------------------------------------


def feathered_paste_(
    dst_img: torch.Tensor,
    src_patch: torch.Tensor,
    dst_y: int,
    dst_x: int,
    feather_sigma: float,
) -> None:
    """In-place feathered paste of ``src_patch`` into ``dst_img`` at ``(dst_y, dst_x)``.

    Args:
        dst_img: ``[C, H, W]`` destination tensor (modified in place).
        src_patch: ``[C, h, w]`` source patch.
        dst_y, dst_x: top-left corner in destination coordinates.
        feather_sigma: Gaussian sigma controlling edge blending; ``0`` => hard paste.
    """
    C, h, w = src_patch.shape
    H, W = dst_img.shape[-2:]
    if dst_y + h > H or dst_x + w > W:
        raise ValueError(
            f"paste region ({dst_y}+{h},{dst_x}+{w}) exceeds image ({H},{W})"
        )
    mask = _make_feather_mask(
        h, w, feather_sigma, device=dst_img.device, dtype=dst_img.dtype
    )  # [h, w]
    region = dst_img[:, dst_y : dst_y + h, dst_x : dst_x + w]
    region.mul_(1.0 - mask).add_(src_patch * mask)


def density_paste_(
    dst_density: torch.Tensor,
    src_density: torch.Tensor,
    dst_y8: int,
    dst_x8: int,
) -> None:
    """In-place hard paste of source stride-8 density into destination at the
    given stride-8 coordinates. Both tensors are ``[1, h8, w8]`` shaped.
    """
    _, h8, w8 = src_density.shape
    dst_density[:, dst_y8 : dst_y8 + h8, dst_x8 : dst_x8 + w8] = src_density


# ---------------------------------------------------------------------------
# A4: RandomGaussianBlur
# ---------------------------------------------------------------------------


class RandomGaussianBlur:
    """Apply random Gaussian blur to the image tensor.

    Only blurs the RGB image channels; density and depth maps are left untouched
    so that count annotations remain exact.

    This augmentation simulates out-of-focus images and improves robustness
    to varying image quality in real-world crowd scenes.
    """

    def __init__(
        self,
        prob: float = 0.1,
        kernel_size: int = 5,
        sigma_range: Sequence[float] = (0.3, 1.5),
    ):
        """Args:
            prob: probability of applying the blur.
            kernel_size: size of the Gaussian kernel (must be odd).
            sigma_range: ``(min, max)`` range for uniform sigma sampling.
        """
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0,1], got {prob}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd number, got {kernel_size}"
            )
        if len(sigma_range) != 2 or sigma_range[0] <= 0 or sigma_range[0] > sigma_range[1]:
            raise ValueError(f"sigma_range must be (min, max) with 0 < min <= max, got {sigma_range}")
        self.prob = float(prob)
        self.kernel_size = int(kernel_size)
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_kernel(size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
        """Build a 2D Gaussian kernel of shape ``[1, 1, size, size]``."""
        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2.0 * sigma**2))
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

    # ------------------------------------------------------------------
    # Callable
    # ------------------------------------------------------------------

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to ``[C, H, W]`` image tensor.

        Args:
            img: normalised image tensor ``[C, H, W]``.

        Returns:
            Blurred image tensor (same shape). When the augmentation is
            skipped (random draw), the original tensor is returned unchanged.
        """
        if random.random() > self.prob:
            return img
        sigma = random.uniform(*self.sigma_range)
        kernel = self._make_kernel(self.kernel_size, sigma, device=img.device, dtype=img.dtype)
        C = img.shape[0]
        # ``groups=C`` applies a separate (identical) kernel per channel,
        # which is equivalent to conv2d with a [C, 1, k, k] weight.
        pad = self.kernel_size // 2
        blurred = torch.nn.functional.conv2d(
            img.unsqueeze(0), kernel.expand(C, -1, -1, -1),
            padding=pad, groups=C,
        ).squeeze(0)
        return blurred

"""Image transforms for crowd counting."""

from __future__ import annotations

import random

import torch
import numpy as np


class DeNormalize:
    """Reverse ImageNet normalization."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class CopyPasteCrowdAugmentation:
    """Copy-Paste augmentation for crowd counting.

    Randomly crops dense crowd patches from the image and pastes them
    onto background regions, forcing the model to handle complex
    crowd-background boundaries. Point annotations are updated accordingly.

    Args:
        paste_prob: Probability of applying paste per image.
        num_pastes: Maximum number of paste operations.
        min_patch_ratio: Minimum patch size as fraction of image dim.
        max_patch_ratio: Maximum patch size as fraction of image dim.
        density_threshold: Minimum mean density to consider a patch "dense".
    """

    def __init__(
        self,
        paste_prob: float = 0.5,
        num_pastes: int = 2,
        min_patch_ratio: float = 0.1,
        max_patch_ratio: float = 0.3,
        density_threshold: float = 0.001,
    ) -> None:
        self.paste_prob = paste_prob
        self.num_pastes = num_pastes
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio
        self.density_threshold = density_threshold

    def __call__(
        self,
        image: np.ndarray,
        points: np.ndarray,
        density_map: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply CopyPaste augmentation.

        Args:
            image: [H, W, 3] uint8 image.
            points: [N, 2] float array of (x, y) point annotations.
            density_map: Optional [H, W] density map for selecting dense regions.

        Returns:
            Tuple of augmented (image, points).
        """
        if random.random() > self.paste_prob:
            return image, points

        H, W = image.shape[:2]
        image = image.copy()
        points = points.copy()
        all_new_points: list[np.ndarray] = []

        for _ in range(self.num_pastes):
            # Random source patch size
            ph = int(H * random.uniform(self.min_patch_ratio, self.max_patch_ratio))
            pw = int(W * random.uniform(self.min_patch_ratio, self.max_patch_ratio))
            ph = max(ph, 16)
            pw = max(pw, 16)

            # Find a source patch (prefer dense regions if density available)
            src_y, src_x = self._sample_source(H, W, ph, pw, density_map)

            # Extract patch
            patch = image[src_y : src_y + ph, src_x : src_x + pw].copy()

            # Find points inside the source patch
            mask = (
                (points[:, 0] >= src_x)
                & (points[:, 0] < src_x + pw)
                & (points[:, 1] >= src_y)
                & (points[:, 1] < src_y + ph)
            )
            patch_points = points[mask].copy()

            if len(patch_points) < 2:
                continue  # Skip if source patch has too few points

            # Random target location (prefer low-density background)
            tgt_y, tgt_x = self._sample_target(H, W, ph, pw, density_map)

            # Paste patch onto image
            # Clip to image bounds
            paste_h = min(ph, H - tgt_y)
            paste_w = min(pw, W - tgt_x)
            image[tgt_y : tgt_y + paste_h, tgt_x : tgt_x + paste_w] = patch[
                :paste_h, :paste_w
            ]

            # Translate points to target location
            new_pts = patch_points.copy()
            new_pts[:, 0] = new_pts[:, 0] - src_x + tgt_x
            new_pts[:, 1] = new_pts[:, 1] - src_y + tgt_y

            # Filter points within image bounds
            valid = (
                (new_pts[:, 0] >= 0)
                & (new_pts[:, 0] < W)
                & (new_pts[:, 1] >= 0)
                & (new_pts[:, 1] < H)
            )
            all_new_points.append(new_pts[valid])

        if all_new_points:
            points = np.concatenate([points] + all_new_points, axis=0)

        return image, points

    def _sample_source(
        self,
        H: int,
        W: int,
        ph: int,
        pw: int,
        density_map: np.ndarray | None,
    ) -> tuple[int, int]:
        """Sample source patch location, preferring dense regions."""
        if density_map is not None:
            # Pool density to coarse grid and sample proportional to density
            stride = 16
            h_pool = max(1, (H - ph) // stride)
            w_pool = max(1, (W - pw) // stride)
            scores = []
            coords = []
            for yi in range(h_pool):
                for xi in range(w_pool):
                    y, x = yi * stride, xi * stride
                    region = density_map[y : y + ph, x : x + pw]
                    scores.append(region.mean())
                    coords.append((y, x))
            if coords:
                scores_arr = np.array(scores, dtype=np.float64)
                scores_arr = scores_arr - scores_arr.min()
                total = scores_arr.sum()
                if total > 0:
                    probs = scores_arr / total
                    idx = np.random.choice(len(coords), p=probs)
                    return coords[idx]
        # Fallback: uniform random
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        return y, x

    def _sample_target(
        self,
        H: int,
        W: int,
        ph: int,
        pw: int,
        density_map: np.ndarray | None,
    ) -> tuple[int, int]:
        """Sample target paste location, preferring low-density regions."""
        if density_map is not None:
            stride = 16
            h_pool = max(1, (H - ph) // stride)
            w_pool = max(1, (W - pw) // stride)
            scores = []
            coords = []
            for yi in range(h_pool):
                for xi in range(w_pool):
                    y, x = yi * stride, xi * stride
                    region = density_map[y : y + ph, x : x + pw]
                    scores.append(1.0 / (region.mean() + 1e-6))
                    coords.append((y, x))
            if coords:
                scores_arr = np.array(scores, dtype=np.float64)
                scores_arr = scores_arr - scores_arr.min()
                total = scores_arr.sum()
                if total > 0:
                    probs = scores_arr / total
                    idx = np.random.choice(len(coords), p=probs)
                    return coords[idx]
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        return y, x

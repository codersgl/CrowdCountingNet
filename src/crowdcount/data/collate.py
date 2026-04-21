"""Collate functions for crowd-counting dataloaders.

Adapted from util/misc.py collate_fn_crowd / collate_fn_crowd_train.
"""

from __future__ import annotations

import functools
import random
from typing import Any, List, Tuple

import torch

from crowdcount.utils.misc import nested_tensor_from_tensor_list


def collate_fn_crowd(batch):
    """Collate for evaluation (img, targets)."""
    batch_new = []
    for b in batch:
        imgs, points = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i], points[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def collate_fn_crowd_train(batch):
    """Collate for training (img, targets, density)."""
    batch_new = []
    for b in batch:
        imgs, points, density = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if density.ndim == 3:
            density = density.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i], points[i], density[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def collate_fn_crowd_depth(batch):
    """Collate for evaluation with depth (img, targets, depth_map)."""
    batch_new = []
    for b in batch:
        imgs, points, depth = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i], points[i], depth[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def collate_fn_crowd_train_depth(batch):
    """Collate for training with depth (img, targets, density, depth_map)."""
    batch_new = []
    for b in batch:
        imgs, points, density, depth = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if density.ndim == 3:
            density = density.unsqueeze(0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i], points[i], density[i], depth[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


# ---------------------------------------------------------------------------
# B3: CopyPasteDense (collate-level) -----------------------------------------
# ---------------------------------------------------------------------------


def _apply_copy_paste_dense(
    samples: List[Tuple[torch.Tensor, dict, torch.Tensor]],
    paste_size: int,
    prob: float,
    feather_sigma: float,
) -> None:
    """Pair up samples in the flattened batch and apply copy-paste densification.

    Samples are mutated in place. Each ``samples[i]`` is a tuple of
    ``(img[C,H,W], target_dict, density[1,H/8,W/8])``. The dest sample's image,
    density tensor and ``target['point']`` / ``target['labels']`` are updated.
    """
    from crowdcount.data.transforms import (
        density_paste_,
        feathered_paste_,
        pick_window_by_point_count,
    )

    n = len(samples)
    if n < 2 or paste_size <= 0 or paste_size % 8 != 0:
        return

    indices = list(range(n))
    random.shuffle(indices)
    for k in range(0, n - 1, 2):
        if random.random() > prob:
            continue
        src_idx = indices[k]
        dst_idx = indices[k + 1]
        src_img, src_tgt, src_den = samples[src_idx]
        dst_img, dst_tgt, dst_den = samples[dst_idx]

        # Shape sanity (skip mismatched pairs rather than failing the batch).
        if src_img.shape != dst_img.shape or src_den.shape != dst_den.shape:
            continue
        H, W = src_img.shape[-2:]
        if paste_size > H or paste_size > W:
            continue
        stride = 8
        h8 = paste_size // stride
        w8 = paste_size // stride
        if dst_den.shape[-2] != H // stride or dst_den.shape[-1] != W // stride:
            continue

        src_pts = src_tgt["point"]
        dst_pts = dst_tgt["point"]
        src_y, src_x = pick_window_by_point_count(
            src_pts, H, W, paste_size, paste_size, mode="max", align_to=stride
        )
        dst_y, dst_x = pick_window_by_point_count(
            dst_pts, H, W, paste_size, paste_size, mode="min", align_to=stride
        )

        # Build source patches (clone to avoid mutating the source sample).
        src_img_patch = src_img[
            :, src_y : src_y + paste_size, src_x : src_x + paste_size
        ].clone()
        src_den_patch = src_den[
            :,
            src_y // stride : src_y // stride + h8,
            src_x // stride : src_x // stride + w8,
        ].clone()

        # Translate kept source points to dest coords.
        if src_pts.numel() > 0:
            in_src = (
                (src_pts[:, 0] >= src_x)
                & (src_pts[:, 0] < src_x + paste_size)
                & (src_pts[:, 1] >= src_y)
                & (src_pts[:, 1] < src_y + paste_size)
            )
            translated = src_pts[in_src].clone()
            if translated.numel() > 0:
                translated[:, 0] += dst_x - src_x
                translated[:, 1] += dst_y - src_y
        else:
            translated = src_pts.new_zeros((0, 2))

        # Drop dest points inside the paste window.
        if dst_pts.numel() > 0:
            in_dst = (
                (dst_pts[:, 0] >= dst_x)
                & (dst_pts[:, 0] < dst_x + paste_size)
                & (dst_pts[:, 1] >= dst_y)
                & (dst_pts[:, 1] < dst_y + paste_size)
            )
            kept_dst = dst_pts[~in_dst]
        else:
            kept_dst = dst_pts

        merged = torch.cat([kept_dst, translated], dim=0)
        feathered_paste_(dst_img, src_img_patch, dst_y, dst_x, feather_sigma)
        density_paste_(dst_den, src_den_patch, dst_y // stride, dst_x // stride)
        dst_tgt["point"] = merged
        dst_tgt["labels"] = torch.ones(merged.shape[0], dtype=torch.long)


def collate_fn_crowd_train_copy_paste_dense(
    batch: Any,
    paste_size: int,
    prob: float,
    feather_sigma: float,
):
    """Train collate that applies CopyPasteDense over flattened patches."""
    batch_new: List[Tuple[torch.Tensor, dict, torch.Tensor]] = []
    for b in batch:
        imgs, points, density = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if density.ndim == 3:
            density = density.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i], points[i], density[i]))
    _apply_copy_paste_dense(batch_new, paste_size, prob, feather_sigma)
    zipped = list(zip(*batch_new))
    zipped[0] = nested_tensor_from_tensor_list(list(zipped[0]))
    return tuple(zipped)


def make_train_collate(aug_cfg: Any, use_depth: bool):
    """Return the appropriate train collate for the given augmentation config.

    Falls back to the standard collate when CopyPasteDense is disabled, or when
    ``use_depth=True`` (depth maps would be semantically inconsistent under
    cross-image patch pasting).
    """
    cpd = {}
    if aug_cfg is not None:
        try:
            cpd = aug_cfg.get("copy_paste_dense", {}) or {}
        except AttributeError:
            cpd = {}
    enabled = bool(cpd.get("enabled", False)) if cpd else False
    if not enabled:
        return collate_fn_crowd_train_depth if use_depth else collate_fn_crowd_train
    if use_depth:
        from loguru import logger

        logger.warning(
            "CopyPasteDense disabled because use_depth=True; depth maps would "
            "be semantically inconsistent across pasted regions."
        )
        return collate_fn_crowd_train_depth
    return functools.partial(
        collate_fn_crowd_train_copy_paste_dense,
        paste_size=int(cpd.get("paste_size", 64)),
        prob=float(cpd.get("prob", 0.5)),
        feather_sigma=float(cpd.get("feather_sigma", 8.0)),
    )

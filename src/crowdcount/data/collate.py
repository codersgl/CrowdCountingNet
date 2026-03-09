"""Collate functions for crowd-counting dataloaders.

Adapted from util/misc.py collate_fn_crowd / collate_fn_crowd_train.
"""

from __future__ import annotations

from typing import List, Tuple

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

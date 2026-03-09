"""Dataset loader factory.

Returns (train_set, val_set) for the configured dataset.
"""

from __future__ import annotations

import os

import torchvision.transforms as standard_transforms
from omegaconf import DictConfig

from crowdcount.data.dataset import SHHA
from crowdcount.data.transforms import DeNormalize


def build_dataset(cfg: DictConfig):
    """Return (train_set, val_set).

    Args:
        cfg: top-level hydra DictConfig; uses cfg.data.data_root,
             cfg.data.patch, cfg.data.flip.
    """
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    data_root = cfg.data.data_root
    if not data_root or not os.path.isdir(data_root):
        raise ValueError(
            f"data.data_root '{data_root}' does not exist or is not set. "
            "Pass it on the command line: data.data_root=/path/to/dataset"
        )
    train_set = SHHA(
        data_root,
        train=True,
        transform=transform,
        patch=cfg.data.patch,
        flip=cfg.data.flip,
    )
    val_set = SHHA(data_root, train=False, transform=transform)
    return train_set, val_set

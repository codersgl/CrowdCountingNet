"""Dataset loader factory.

Returns (train_set, val_set) for the configured dataset.
"""

from __future__ import annotations

import os

import torchvision.transforms as standard_transforms
from omegaconf import DictConfig

from crowdcount.data.dataset import SHHA


def build_dataset(cfg: DictConfig):
    """Return (train_set, val_set).

    Args:
        cfg: top-level hydra DictConfig; uses cfg.data.data_root,
             cfg.data.patch, cfg.data.flip, cfg.model.use_depth.
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

    use_depth = bool(getattr(getattr(cfg, "model", None), "use_depth", False))
    use_depth_geo = bool(getattr(getattr(cfg, "model", None), "use_depth_geo", False))
    use_depth_dual_vgg = bool(
        getattr(getattr(cfg, "model", None), "use_depth_dual_vgg", False)
    )
    use_depth_attn = bool(getattr(getattr(cfg, "model", None), "use_depth_attn", False))
    needs_depth = use_depth or use_depth_geo or use_depth_dual_vgg or use_depth_attn
    depth_cfg = (
        getattr(getattr(cfg, "model", None), "depth", None) if needs_depth else None
    )

    # Extract augmentation configuration
    aug_cfg = cfg.data.get("augmentation", None)
    flip_prob = float(cfg.data.get("flip_prob", 0.5))
    num_patches = int(cfg.data.get("num_patches", 4))
    depth_blur_cfg = cfg.data.get("depth_blur", None)
    density_gen_cfg = cfg.data.get("density_generation", None)

    train_set = SHHA(
        data_root,
        train=True,
        transform=transform,
        patch=cfg.data.patch,
        patch_size=int(cfg.data.get("patch_size", 128)),
        flip=cfg.data.flip,
        use_depth=needs_depth,
        depth_cfg=depth_cfg,
        aug_cfg=aug_cfg,
        flip_prob=flip_prob,
        num_patches=num_patches,
        depth_blur_cfg=depth_blur_cfg,
        density_gen_cfg=density_gen_cfg,
    )
    val_set = SHHA(
        data_root,
        train=False,
        transform=transform,
        use_depth=needs_depth,
        depth_cfg=depth_cfg,
        aug_cfg=aug_cfg,
        flip_prob=flip_prob,
        num_patches=num_patches,
        depth_blur_cfg=depth_blur_cfg,
        density_gen_cfg=density_gen_cfg,
    )
    return train_set, val_set

"""Dataset loader factory.

Returns (train_set, val_set) for the configured dataset.
"""

from __future__ import annotations

import os

import torchvision.transforms as standard_transforms
from omegaconf import DictConfig

from crowdcount.data.dataset import SHHA


def _resolve_norm_stats(
    cfg: DictConfig,
) -> tuple[list[float], list[float]]:
    """Return ``(mean, std)`` for image normalization.

    For CLIP backbones the stats differ from ImageNet defaults; the backbone
    class exposes them via ``BackboneCLIP.norm_stats``.  We resolve them here
    from config alone (without instantiating the backbone) so that the data
    pipeline is independent of the model graph.
    """
    backbone_type = getattr(getattr(cfg, "model", None), "backbone_type", "vgg")
    if backbone_type == "clip":
        from crowdcount.models.backbone import (
            _CLIP_DEFAULT_PRETRAINED,
            _CLIP_NORM_STATS,
        )

        backbone_name = getattr(getattr(cfg, "model", None), "backbone", "ViT-B-16")
        pretrained_tag = _CLIP_DEFAULT_PRETRAINED.get(backbone_name, "openai")
        mean, std = _CLIP_NORM_STATS.get(pretrained_tag, _CLIP_NORM_STATS["openai"])
        return list(mean), list(std)
    # Default: ImageNet stats (VGG, ConvNeXt, DINOv2)
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_dataset(cfg: DictConfig):
    """Return (train_set, val_set).

    Args:
           cfg: top-level hydra DictConfig; uses cfg.data.data_root,
               cfg.data.patch, cfg.data.flip, and model depth flags.
    """
    norm_mean, norm_std = _resolve_norm_stats(cfg)
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=norm_mean, std=norm_std),
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
    use_depth_geo_post = bool(
        getattr(getattr(cfg, "model", None), "use_depth_geo_post", False)
    )
    use_depth_dual_vgg = bool(
        getattr(getattr(cfg, "model", None), "use_depth_dual_vgg", False)
    )
    use_depth_attn = bool(getattr(getattr(cfg, "model", None), "use_depth_attn", False))
    use_depth_cross_attn = bool(
        getattr(getattr(cfg, "model", None), "use_depth_cross_attn", False)
    )
    depth_graph_prior = getattr(getattr(cfg, "model", None), "depth_graph_prior", None)
    use_depth_graph_prior = bool(getattr(depth_graph_prior, "enabled", False))
    use_depth_aux = bool(getattr(getattr(cfg, "model", None), "use_depth_aux", False))
    needs_depth_input = (
        use_depth
        or use_depth_geo
        or use_depth_geo_post
        or use_depth_dual_vgg
        or use_depth_attn
        or use_depth_cross_attn
        or use_depth_graph_prior
    )
    needs_depth_train = needs_depth_input or use_depth_aux
    needs_depth_eval = needs_depth_input
    depth_cfg = (
        getattr(getattr(cfg, "model", None), "depth", None)
        if (needs_depth_train or needs_depth_eval)
        else None
    )

    # Extract augmentation configuration
    aug_cfg = cfg.data.get("augmentation", None)
    flip_prob = float(cfg.data.get("flip_prob", 0.5))
    num_patches = int(cfg.data.get("num_patches", 4))
    depth_blur_cfg = cfg.data.get("depth_blur", None)
    density_gen_cfg = cfg.data.get("density_generation", None)
    resize_cfg = cfg.data.get("resize", None)

    train_set = SHHA(
        data_root,
        train=True,
        transform=transform,
        patch=cfg.data.patch,
        patch_size=int(cfg.data.get("patch_size", 128)),
        flip=cfg.data.flip,
        use_depth=needs_depth_train,
        depth_cfg=depth_cfg,
        aug_cfg=aug_cfg,
        flip_prob=flip_prob,
        num_patches=num_patches,
        depth_blur_cfg=depth_blur_cfg,
        density_gen_cfg=density_gen_cfg,
        resize_cfg=resize_cfg,
    )
    val_set = SHHA(
        data_root,
        train=False,
        transform=transform,
        use_depth=needs_depth_eval,
        depth_cfg=depth_cfg,
        aug_cfg=aug_cfg,
        flip_prob=flip_prob,
        num_patches=num_patches,
        depth_blur_cfg=depth_blur_cfg,
        density_gen_cfg=density_gen_cfg,
        resize_cfg=resize_cfg,
    )
    return train_set, val_set

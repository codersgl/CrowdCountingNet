"""ShanghaiTech Part-A (SHHA) dataset.

Logic is unchanged from crowd_datasets/SHHA/SHHA.py.
Extension: on first construction, auto-generates gt_density_maps if missing.
"""

from __future__ import annotations

import os
import random
import re
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from crowdcount.data.prepare import _resolve_density_cache_dir, generate_density_maps


class SHHA(Dataset):
    def __init__(
        self,
        data_root: str,
        transform=None,
        train: bool = False,
        patch: bool = False,
        patch_size: int = 128,
        flip: bool = False,
        use_depth: bool = False,
        depth_cfg=None,
        aug_cfg=None,
        flip_prob: float = 0.5,
        num_patches: int = 4,
        depth_blur_cfg=None,
        density_gen_cfg=None,
        resize_cfg=None,
    ):
        self.root_path = data_root
        self.use_depth = use_depth

        if resize_cfg is None:
            resize_cfg = {}
        self.resize_enabled = bool(resize_cfg.get("enabled", False))
        _max_long_side = resize_cfg.get("max_long_side", None)
        self.max_long_side = int(_max_long_side) if _max_long_side is not None else None
        self.keep_aspect_ratio = bool(resize_cfg.get("keep_aspect_ratio", True))
        if self.resize_enabled:
            if self.max_long_side is None or self.max_long_side <= 0:
                raise ValueError(
                    "resize.max_long_side must be a positive integer when resize is enabled"
                )
            if not self.keep_aspect_ratio:
                raise ValueError(
                    "Only keep_aspect_ratio=true is supported for dataset resize"
                )

        # Parse density generation config
        if density_gen_cfg is None:
            density_gen_cfg = {}
        self.perspective_guided = bool(density_gen_cfg.get("perspective_guided", False))
        self.persp_beta = float(density_gen_cfg.get("beta", 0.3))
        self.persp_min_sigma = float(density_gen_cfg.get("min_sigma", 1.0))
        self.persp_sigma_base = float(density_gen_cfg.get("sigma_base", 1.0))
        _pms = density_gen_cfg.get("persp_max_sigma", None)
        self.persp_max_sigma = float(_pms) if _pms is not None else None
        self.persp_disparity_input = bool(density_gen_cfg.get("disparity_input", True))
        self.hybrid = bool(density_gen_cfg.get("hybrid", False))
        self.hybrid_min_sigma = float(density_gen_cfg.get("hybrid_min_sigma", 1.5))
        _hms = density_gen_cfg.get("hybrid_max_sigma", None)
        self.hybrid_max_sigma = float(_hms) if _hms is not None else None
        self.hybrid_alpha = float(density_gen_cfg.get("hybrid_alpha", 0.5))
        if patch_size <= 0 or patch_size % 8 != 0:
            raise ValueError(
                f"patch_size must be a positive multiple of 8, got {patch_size}"
            )
        self.patch_size = patch_size
        self.num_patches = num_patches
        split = "train" if train else "test"

        # Parse augmentation config
        if aug_cfg is None:
            aug_cfg = {}

        # Color augmentation config
        color_jitter_cfg = aug_cfg.get("color_jitter", {})
        self.color_jitter_enabled = color_jitter_cfg.get("enabled", True)
        self.color_jitter_apply_prob = float(color_jitter_cfg.get("apply_prob", 0.5))
        self.color_jitter_brightness = float(color_jitter_cfg.get("brightness", 0.5))
        self.color_jitter_contrast = float(color_jitter_cfg.get("contrast", 0.5))
        self.color_jitter_saturation = float(color_jitter_cfg.get("saturation", 0.5))
        self.color_jitter_hue = float(color_jitter_cfg.get("hue", 0.5))

        # Grayscale config
        grayscale_cfg = aug_cfg.get("random_grayscale", {})
        self.grayscale_enabled = grayscale_cfg.get("enabled", True)
        self.grayscale_prob = float(grayscale_cfg.get("prob", 0.5))

        # Random scaling config
        scale_cfg = aug_cfg.get("random_scale", {})
        self.scale_enabled = scale_cfg.get("enabled", True)
        self.scale_min = float(scale_cfg.get("scale_min", 0.7))
        self.scale_max = float(scale_cfg.get("scale_max", 1.3))

        # A1: Random erasing config
        erasing_cfg = aug_cfg.get("random_erasing", {})
        self.random_erasing_enabled = bool(erasing_cfg.get("enabled", False))
        self.random_erasing_prob = float(erasing_cfg.get("prob", 0.5))
        _scale_range = erasing_cfg.get("scale_range", [0.02, 0.2])
        _ratio_range = erasing_cfg.get("ratio_range", [0.3, 3.3])
        self.random_erasing_scale_range = (
            float(_scale_range[0]),
            float(_scale_range[1]),
        )
        self.random_erasing_ratio_range = (
            float(_ratio_range[0]),
            float(_ratio_range[1]),
        )
        self.random_erasing_fill = float(erasing_cfg.get("fill", 0.0))

        # A2: Multi-scale patch config
        msp_cfg = aug_cfg.get("multi_scale_patch", {})
        self.multi_scale_patch_enabled = bool(msp_cfg.get("enabled", False))
        _choices = list(msp_cfg.get("patch_size_choices", [96, 128, 192]))
        if self.multi_scale_patch_enabled:
            if len(_choices) == 0:
                raise ValueError(
                    "multi_scale_patch.patch_size_choices must be non-empty"
                )
            for c in _choices:
                if int(c) <= 0 or int(c) % 8 != 0:
                    raise ValueError(
                        f"patch_size_choices entries must be positive multiples of 8, got {c}"
                    )
        self.multi_scale_patch_choices = [int(c) for c in _choices]

        # A4: Random Gaussian Blur config
        blur_cfg = aug_cfg.get("gaussian_blur", {})
        self.gaussian_blur_enabled = bool(blur_cfg.get("enabled", False))
        self.gaussian_blur_prob = float(blur_cfg.get("prob", 0.1))
        self.gaussian_blur_kernel_size = int(blur_cfg.get("kernel_size", 5))
        _blur_sigma = blur_cfg.get("sigma_range", [0.3, 1.5])
        self.gaussian_blur_sigma_range = (
            float(_blur_sigma[0]),
            float(_blur_sigma[1]),
        )

        # Flip probability
        self.flip_prob = flip_prob

        # Depth blur config
        if depth_blur_cfg is None:
            depth_blur_cfg = {}
        self.depth_blur_kernel = int(depth_blur_cfg.get("kernel_size", 15))
        self.depth_blur_sigma = float(depth_blur_cfg.get("sigma", 5.0))

        # Validate depth blur kernel size (must be positive odd number)
        if self.depth_blur_kernel <= 0 or self.depth_blur_kernel % 2 == 0:
            raise ValueError(
                f"depth_blur kernel_size must be a positive odd number, got {self.depth_blur_kernel}"
            )

        # Validate scale range
        if self.scale_min > self.scale_max:
            raise ValueError(
                f"scale_min ({self.scale_min}) must be <= scale_max ({self.scale_max})"
            )
        if self.scale_min <= 0:
            raise ValueError(f"scale_min must be positive, got {self.scale_min}")

        if train:
            cache_dir = _resolve_density_cache_dir(
                self.root_path,
                "train",
                perspective_guided=self.perspective_guided,
                hybrid=self.hybrid,
                beta=self.persp_beta,
                min_sigma=self.persp_min_sigma,
                sigma_base=self.persp_sigma_base,
                persp_max_sigma=self.persp_max_sigma,
                hybrid_min_sigma=self.hybrid_min_sigma,
                hybrid_max_sigma=self.hybrid_max_sigma,
                hybrid_alpha=self.hybrid_alpha,
            )
            self.gt_dmap_root = str(cache_dir)
            # Auto-generate density maps on first run (or when params changed,
            # which produces a fresh cache directory).
            existing = (
                [p for p in os.listdir(self.gt_dmap_root) if p.endswith(".npy")]
                if os.path.isdir(self.gt_dmap_root)
                else []
            )
            if not existing:
                generate_density_maps(
                    data_root,
                    split="train",
                    perspective_guided=self.perspective_guided,
                    hybrid=self.hybrid,
                    beta=self.persp_beta,
                    min_sigma=self.persp_min_sigma,
                    hybrid_min_sigma=self.hybrid_min_sigma,
                    hybrid_max_sigma=self.hybrid_max_sigma,
                    hybrid_alpha=self.hybrid_alpha,
                    disparity_input=self.persp_disparity_input,
                    sigma_base=self.persp_sigma_base,
                    persp_max_sigma=self.persp_max_sigma,
                )

        if use_depth:
            depth_split = "train" if train else "test"
            self.gt_depth_root = os.path.join(data_root, "gt_depth_maps", depth_split)
            if not os.path.isdir(self.gt_depth_root) or not os.listdir(
                self.gt_depth_root
            ):
                from crowdcount.data.prepare import generate_depth_maps

                encoder = "vitb"
                weight_path = None
                if depth_cfg is not None:
                    encoder = str(getattr(depth_cfg, "encoder", "vitb"))
                    wp = getattr(depth_cfg, "weight_path", None)
                    weight_path = str(wp) if wp is not None else None
                generate_depth_maps(
                    data_root,
                    split=depth_split,
                    encoder=encoder,
                    weight_path=weight_path,
                )

        # Discover image/GT pairs without any list file
        from crowdcount.data.prepare import _find_image_gt_pairs

        pairs = _find_image_gt_pairs(Path(data_root), split)
        self.img_map: dict = {str(img_p): str(gt_p) for img_p, gt_p in pairs}
        self.img_list = sorted(self.img_map.keys())
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self) -> int:
        return self.nSamples

    def __getitem__(self, index: int):
        assert index <= len(self), "index range error"
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        imgname = os.path.basename(img_path)

        if self.train:
            gt_dmap = np.load(
                os.path.join(self.gt_dmap_root, imgname.replace(".jpg", ".npy"))
            )
            gt_dmap = torch.from_numpy(gt_dmap)
            gt_dmap1 = gt_dmap.unsqueeze(0)

            if self.use_depth:
                depth_npy = np.load(
                    os.path.join(self.gt_depth_root, imgname.replace(".jpg", ".npy"))
                ).astype(np.float32)
                # Gaussian blur to smooth depth edge discontinuities
                depth_npy = cv2.GaussianBlur(
                    depth_npy,
                    (self.depth_blur_kernel, self.depth_blur_kernel),
                    sigmaX=self.depth_blur_sigma,
                )
                # Min-max normalise depth to [0, 1]
                d_min, d_max = depth_npy.min(), depth_npy.max()
                if d_max - d_min > 1e-6:
                    depth_npy = (depth_npy - d_min) / (d_max - d_min)
                gt_depth1 = torch.from_numpy(depth_npy).unsqueeze(0)  # [1, H, W]

        img, point = _load_data((img_path, gt_path), self.train)

        if self.resize_enabled:
            img, point, resize_scale = _resize_image_and_points_long_side(
                img, point, self.max_long_side
            )
            if self.train and resize_scale != 1.0:
                gt_dmap1 = _resize_single_channel_preserve_sum(
                    gt_dmap1,
                    size=(img.height, img.width),
                    target_sum=float(torch.sum(gt_dmap).item()),
                )
                if self.use_depth:
                    gt_depth1 = torch.nn.functional.interpolate(
                        gt_depth1.unsqueeze(0),
                        size=(img.height, img.width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

        if self.train:
            aug_list = []
            # Color jitter augmentation
            if self.color_jitter_enabled:
                aug_list.append(
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=self.color_jitter_brightness,
                                contrast=self.color_jitter_contrast,
                                saturation=self.color_jitter_saturation,
                                hue=self.color_jitter_hue,
                            )
                        ],
                        p=self.color_jitter_apply_prob,
                    )
                )
            # Random grayscale augmentation
            if self.grayscale_enabled:
                aug_list.append(transforms.RandomGrayscale(p=self.grayscale_prob))

            if aug_list:
                augmentation = transforms.Compose(aug_list)
                img = augmentation(img)

        if self.transform is not None:
            img = self.transform(img)

        # A4: Random Gaussian Blur (applied on normalised tensor, before scaling)
        if self.train and self.gaussian_blur_enabled:
            from crowdcount.data.transforms import RandomGaussianBlur

            _blurrer = RandomGaussianBlur(
                prob=self.gaussian_blur_prob,
                kernel_size=self.gaussian_blur_kernel_size,
                sigma_range=self.gaussian_blur_sigma_range,
            )
            img = _blurrer(img)

        if self.train and self.scale_enabled:
            min_size = min(img.shape[1:])
            min_crop = self.patch_size if self.patch else 128
            # Dynamic lower bound: ensure scale * min_size > min_crop so the
            # downstream crop is always feasible. This eliminates the silent
            # "skip when scale<1 on small images" branch that biased the
            # effective scale distribution toward zoom-in.
            effective_min = max(self.scale_min, (min_crop + 1) / float(min_size))
            effective_max = max(effective_min, self.scale_max)
            scale = random.uniform(effective_min, effective_max)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_dmap1 = torch.nn.functional.interpolate(
                gt_dmap1.unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            dmap_sum = torch.sum(gt_dmap1)
            if dmap_sum > 0:
                gt_dmap1 = gt_dmap1 / dmap_sum * torch.sum(gt_dmap)
            if self.use_depth:
                gt_depth1 = torch.nn.functional.interpolate(
                    gt_depth1.unsqueeze(0),
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            point *= scale

        if self.train:
            # Build joint augmentation stack: [img(3) + density(1) + depth(1)]
            # Depth channel appended last so it can be sliced off cleanly.
            if self.use_depth:
                img_with_density = torch.cat((img, gt_dmap1, gt_depth1), dim=0)
            else:
                img_with_density = torch.cat((img, gt_dmap1), dim=0)

        if self.train and self.patch:
            # A2: choose per-call crop size (same for all patches in this call).
            if self.multi_scale_patch_enabled:
                hw_min = min(img_with_density.shape[-2:])
                valid = [c for c in self.multi_scale_patch_choices if c <= hw_min]
                # Always include canonical patch_size as a safety floor.
                if not valid:
                    valid = [self.patch_size]
                crop_size = random.choice(valid)
            else:
                crop_size = self.patch_size

            # Ensure image is large enough for cropping
            h, w = img_with_density.shape[-2:]
            if h < crop_size or w < crop_size:
                scale_up = max(crop_size / h, crop_size / w)
                img_with_density = torch.nn.functional.interpolate(
                    img_with_density.unsqueeze(0),
                    scale_factor=scale_up,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                point *= scale_up
            img_with_density, point = _random_crop(
                img_with_density,
                point,
                num_patch=self.num_patches,
                crop_size=crop_size,
            )
            for i in range(len(point)):
                point[i] = torch.Tensor(point[i])

            # A2: resize each cropped patch back to canonical patch_size and
            # rescale points; renormalise the density channel to preserve sum.
            if crop_size != self.patch_size:
                img_with_density = torch.from_numpy(
                    np.ascontiguousarray(img_with_density)
                ).float()
                density_ch_start = img_with_density.shape[1] - (
                    2 if self.use_depth else 1
                )
                density_ch_end = density_ch_start + 1
                orig_sums = img_with_density[:, density_ch_start:density_ch_end].sum(
                    dim=(2, 3), keepdim=True
                )
                img_with_density = torch.nn.functional.interpolate(
                    img_with_density,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                new_sums = img_with_density[:, density_ch_start:density_ch_end].sum(
                    dim=(2, 3), keepdim=True
                )
                factor = orig_sums / new_sums.clamp(min=1e-9)
                img_with_density[:, density_ch_start:density_ch_end] *= factor
                pt_scale = float(self.patch_size) / float(crop_size)
                for i in range(len(point)):
                    if point[i].numel() > 0:
                        point[i] = point[i] * pt_scale
                # Restore numpy dtype so downstream flip's [::-1] slicing works
                # (PyTorch tensors do not support negative-step slicing).
                img_with_density = img_with_density.numpy()

        if self.train and not self.patch and not isinstance(point, list):
            point = [torch.Tensor(point)]

        if random.random() < self.flip_prob and self.train and self.flip:
            if img_with_density.ndim == 4:
                img_with_density = torch.Tensor(img_with_density[:, :, :, ::-1].copy())
                flip_w = img_with_density.shape[3]
            else:
                img_with_density = torch.Tensor(img_with_density[:, :, ::-1].copy())
                flip_w = img_with_density.shape[2]
            for i in range(len(point)):
                point[i][:, 0] = flip_w - point[i][:, 0]

        if self.train:
            if img_with_density.ndim == 4:
                if self.use_depth:
                    img = img_with_density[:, :-2, :, :]
                    density = img_with_density[:, -2:-1, :, :]
                    depth = img_with_density[:, -1:, :, :]
                else:
                    img = img_with_density[:, :-1, :, :]
                    density = img_with_density[:, -1:, :, :]
            else:
                if self.use_depth:
                    img = img_with_density[:-2, :, :]
                    density = img_with_density[-2:-1, :, :]
                    depth = img_with_density[-1:, :, :]
                else:
                    img = img_with_density[:-1, :, :]
                    density = img_with_density[-1:, :, :]
            density = torch.Tensor(density)
            if self.use_depth:
                depth = torch.Tensor(depth)

            # Ensure density has batch dimension for consistent processing
            if density.ndim == 3:  # [C, H, W] -> [1, C, H, W]
                density = density.unsqueeze(0)
            if self.use_depth and depth.ndim == 3:  # [C, H, W] -> [1, C, H, W]
                depth = depth.unsqueeze(0)

            # A1: Random erasing applied independently per patch.
            # Guarded to the patched 4D path; non-patch path uses 3D img where
            # img.shape[0] is the channel count, not patch count.
            if self.random_erasing_enabled and img.ndim == 4:
                from crowdcount.data.transforms import RandomErasingCount

                eraser = RandomErasingCount(
                    prob=self.random_erasing_prob,
                    scale_range=self.random_erasing_scale_range,
                    ratio_range=self.random_erasing_ratio_range,
                    fill=self.random_erasing_fill,
                )
                num_patches = img.shape[0]
                for i in range(num_patches):
                    img_i = img[i]
                    den_i = density[i]  # [1, H, W]
                    dep_i = depth[i] if self.use_depth else None
                    pts_i = point[i]
                    img_i, pts_i, den_i, dep_i = eraser(img_i, pts_i, den_i, dep_i)
                    img[i] = img_i
                    density[i] = den_i
                    if self.use_depth:
                        depth[i] = dep_i
                    point[i] = pts_i

        if not self.train:
            point = [point]
            if self.use_depth:
                depth_npy = np.load(
                    os.path.join(self.gt_depth_root, imgname.replace(".jpg", ".npy"))
                ).astype(np.float32)
                # Gaussian blur to smooth depth edge discontinuities
                depth_npy = cv2.GaussianBlur(
                    depth_npy,
                    (self.depth_blur_kernel, self.depth_blur_kernel),
                    sigmaX=self.depth_blur_sigma,
                )
                d_min, d_max = depth_npy.min(), depth_npy.max()
                if d_max - d_min > 1e-6:
                    depth_npy = (depth_npy - d_min) / (d_max - d_min)
                depth = torch.from_numpy(depth_npy).unsqueeze(0)  # [1, H, W]
                if self.resize_enabled:
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),
                        size=(img.shape[-2], img.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

        img = torch.Tensor(img)
        target = [{} for _ in range(len(point))]
        for i in range(len(point)):
            target[i]["point"] = torch.Tensor(point[i])
            image_id = _image_id_from_path(img_path)
            target[i]["image_id"] = torch.Tensor([image_id]).long()
            target[i]["labels"] = torch.ones([point[i].shape[0]]).long()

        if self.train:
            from crowdcount.data.transforms import density_resize_stride8

            density_images = density_resize_stride8(density, stride=8)
            if self.use_depth:
                return img, target, density_images, depth
            return img, target, density_images
        else:
            if self.use_depth:
                return img, target, depth
            return img, target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(img_gt_path, train: bool):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    from crowdcount.data.prepare import _load_points

    points = _load_points(Path(gt_path))
    return img, points


def _image_id_from_path(img_path: str) -> int:
    stem = Path(img_path).stem
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return zlib.crc32(stem.encode("utf-8")) & 0x7FFFFFFF
    return int(match.group(1))


def _resize_image_and_points_long_side(
    img: Image.Image, points: np.ndarray, max_long_side: int | None
) -> tuple[Image.Image, np.ndarray, float]:
    if max_long_side is None:
        return img, points, 1.0
    width, height = img.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return img, points, 1.0
    scale = float(max_long_side) / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    img = img.resize((new_width, new_height), Image.BICUBIC)
    points = points.astype(np.float32, copy=True) * scale
    return img, points, scale


def _resize_single_channel_preserve_sum(
    tensor: torch.Tensor, size: tuple[int, int], target_sum: float
) -> torch.Tensor:
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized_sum = torch.sum(resized)
    if resized_sum > 0:
        resized = resized / resized_sum * target_sum
    return resized


def _random_crop(img, den, num_patch: int = 4, crop_size: int = 128):
    half_h = crop_size
    half_w = crop_size
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        idx = (
            (den[:, 0] >= start_w)
            & (den[:, 0] < end_w)
            & (den[:, 1] >= start_h)
            & (den[:, 1] < end_h)
        )
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)
    return result_img, result_den

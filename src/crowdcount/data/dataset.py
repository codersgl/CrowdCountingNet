"""ShanghaiTech Part-A (SHHA) dataset.

Logic is unchanged from crowd_datasets/SHHA/SHHA.py.
Extension: on first construction, auto-generates gt_density_maps if missing.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from crowdcount.data.prepare import generate_density_maps


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
    ):
        self.root_path = data_root
        self.gt_density = "gt_density_maps"
        self.use_depth = use_depth
        if patch_size <= 0 or patch_size % 8 != 0:
            raise ValueError(
                f"patch_size must be a positive multiple of 8, got {patch_size}"
            )
        self.patch_size = patch_size
        split = "train" if train else "test"

        if train:
            self.gt_dmap_root = os.path.join(self.root_path, self.gt_density, "train")
            # Auto-generate density maps on first run
            if not os.path.isdir(self.gt_dmap_root) or not os.listdir(
                self.gt_dmap_root
            ):
                generate_density_maps(data_root, split="train")

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
                # Min-max normalise depth to [0, 1]
                d_min, d_max = depth_npy.min(), depth_npy.max()
                if d_max - d_min > 1e-6:
                    depth_npy = (depth_npy - d_min) / (d_max - d_min)
                gt_depth1 = torch.from_numpy(depth_npy).unsqueeze(0)  # [1, H, W]

        img, point = _load_data((img_path, gt_path), self.train)

        if self.train:
            augmentation = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                            )
                        ],
                        p=0.5,
                    ),
                    transforms.RandomGrayscale(p=0.5),
                ]
            )
            img = augmentation(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            min_crop = self.patch_size if self.patch else 128
            if scale * min_size > min_crop:
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
                gt_dmap1 = gt_dmap1 / torch.sum(gt_dmap1) * torch.sum(gt_dmap)
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
            # Ensure image is large enough for cropping
            h, w = img_with_density.shape[-2:]
            if h < self.patch_size or w < self.patch_size:
                scale_up = max(self.patch_size / h, self.patch_size / w)
                img_with_density = torch.nn.functional.interpolate(
                    img_with_density.unsqueeze(0),
                    scale_factor=scale_up,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                point *= scale_up
            img_with_density, point = _random_crop(
                img_with_density, point, crop_size=self.patch_size
            )
            for i in range(len(point)):
                point[i] = torch.Tensor(point[i])

        if random.random() > 0.5 and self.train and self.flip:
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

        if not self.train:
            point = [point]
            if self.use_depth:
                depth_npy = np.load(
                    os.path.join(self.gt_depth_root, imgname.replace(".jpg", ".npy"))
                ).astype(np.float32)
                d_min, d_max = depth_npy.min(), depth_npy.max()
                if d_max - d_min > 1e-6:
                    depth_npy = (depth_npy - d_min) / (d_max - d_min)
                depth = torch.from_numpy(depth_npy).unsqueeze(0)  # [1, H, W]

        img = torch.Tensor(img)
        target = [{} for _ in range(len(point))]
        for i in range(len(point)):
            target[i]["point"] = torch.Tensor(point[i])
            image_id = int(img_path.split("/")[-1].split(".")[0].split("_")[-1])
            target[i]["image_id"] = torch.Tensor([image_id]).long()
            target[i]["labels"] = torch.ones([point[i].shape[0]]).long()

        if self.train:
            stride = 8  # PA-FPN output stride
            density_target_h = density.shape[-2] // stride
            density_target_w = density.shape[-1] // stride
            density_images = torch.zeros(
                (density.shape[0], 1, density_target_h, density_target_w),
                dtype=density.dtype,
            )
            for i in range(density.shape[0]):
                density_img = density[i, 0, :, :]
                resized_img = density_img.reshape(
                    [density_target_h, stride, density_target_w, stride]
                ).sum(axis=(1, 3))
                density_images[i, 0, :, :] = resized_img
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

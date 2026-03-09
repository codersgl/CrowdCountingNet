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
        flip: bool = False,
    ):
        self.root_path = data_root
        self.gt_density = "gt_density_maps"
        split = "train" if train else "test"

        if train:
            self.gt_dmap_root = os.path.join(self.root_path, self.gt_density, "train")
            # Auto-generate density maps on first run
            if not os.path.isdir(self.gt_dmap_root) or not os.listdir(
                self.gt_dmap_root
            ):
                generate_density_maps(data_root, split="train")

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
            if scale * min_size > 128:
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
                point *= scale

        if self.train:
            img_with_density = torch.cat((img, gt_dmap1), dim=0)

        if self.train and self.patch:
            img_with_density, point = _random_crop(img_with_density, point)
            for i in range(len(point)):
                point[i] = torch.Tensor(point[i])

        if random.random() > 0.5 and self.train and self.flip:
            img_with_density = torch.Tensor(img_with_density[:, :, :, ::-1].copy())
            for i in range(len(point)):
                point[i][:, 0] = 128 - point[i][:, 0]

        if self.train:
            img = img_with_density[:, :-1, :, :]
            density = img_with_density[:, -1:, :, :]
            density = torch.Tensor(density)

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        target = [{} for _ in range(len(point))]
        for i in range(len(point)):
            target[i]["point"] = torch.Tensor(point[i])
            image_id = int(img_path.split("/")[-1].split(".")[0].split("_")[-1])
            target[i]["image_id"] = torch.Tensor([image_id]).long()
            target[i]["labels"] = torch.ones([point[i].shape[0]]).long()

        if self.train:
            density_images = torch.zeros(
                (density.shape[0], 1, 16, 16), dtype=density.dtype
            )
            for i in range(density.shape[0]):
                density_img = density[i, 0, :, :]
                resized_img = density_img.reshape([16, 8, 16, 8]).sum(axis=(1, 3))
                density_images[i, 0, :, :] = resized_img
            return img, target, density_images
        else:
            return img, target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(img_gt_path, train: bool):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            parts = line.strip().split(" ")
            x = float(parts[0])
            y = float(parts[1])
            points.append([x, y])
    return img, np.array(points)


def _random_crop(img, den, num_patch: int = 4):
    half_h = 128
    half_w = 128
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
            & (den[:, 0] <= end_w)
            & (den[:, 1] >= start_h)
            & (den[:, 1] <= end_h)
        )
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)
    return result_img, result_den

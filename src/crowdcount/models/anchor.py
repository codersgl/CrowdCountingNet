"""Anchor point generation utilities for DSGCNet."""

import numpy as np
import torch
import torch.nn as nn


def generate_anchor_points(stride: int = 16, row: int = 3, line: int = 3) -> np.ndarray:
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    return anchor_points


def shift(shape, stride: int, anchor_points: np.ndarray) -> np.ndarray:
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = anchor_points.reshape((1, A, 2)) + shifts.reshape(
        (1, K, 2)
    ).transpose((1, 0, 2))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))
    return all_anchor_points


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row: int = 3, line: int = 3):
        super().__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(
                image_shapes[idx], self.strides[idx], anchor_points
            )
            all_anchor_points = np.append(
                all_anchor_points, shifted_anchor_points, axis=0
            )

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(image.device)

"""Image transforms for crowd counting."""

import torch


class DeNormalize:
    """Reverse ImageNet normalization."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

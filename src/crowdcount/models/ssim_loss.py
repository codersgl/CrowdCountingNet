"""SSIM loss for density map supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


class SSIMLoss(nn.Module):
    """Structural similarity loss for dense prediction maps.

    Notes on window_size selection:
        DSGCNet density maps are downsampled to [B, 1, 16, 16] before loss
        computation.  With zero-padding the fraction of pixels unaffected by
        boundary artifacts is (H-ws+1)^2 / H^2.  Typical values:
          ws=11 → 14 %  (too small for 16×16)
          ws= 7 → 39 %  (recommended for 16×16, default in configs/)
          ws= 5 → 56 %  (safe minimum)
        Use ``window_size=11`` only when the density map resolution ≥ ~32×32.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average

        window = _gaussian_kernel(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shapes, got {pred.shape} and {target.shape}"
            )
        if pred.dim() != 4:
            raise ValueError(f"expected 4D tensors [B, C, H, W], got {pred.dim()}D")

        channels = pred.shape[1]
        window = self.window.to(device=pred.device, dtype=pred.dtype).expand(
            channels, 1, self.window_size, self.window_size
        )
        padding = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_target = F.conv2d(target, window, padding=padding, groups=channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = (
            F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
        )
        sigma_target_sq = (
            F.conv2d(target * target, window, padding=padding, groups=channels)
            - mu_target_sq
        )
        sigma_pred_target = (
            F.conv2d(pred * target, window, padding=padding, groups=channels)
            - mu_pred_target
        )

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
            + 1e-12
        )
        if self.size_average:
            return 1.0 - ssim_map.mean()
        return 1.0 - ssim_map.mean(dim=(1, 2, 3))

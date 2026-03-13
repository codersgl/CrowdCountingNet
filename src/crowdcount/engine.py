"""Training and evaluation loops for DSGCNet.

Adapted from engine.py — functional logic unchanged,
print() statements replaced with loguru logger.
"""

from __future__ import annotations

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from crowdcount.utils.misc import MetricLogger, SmoothedValue, reduce_dict
from loguru import logger


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    density_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    cfg=None,
) -> dict:
    """Train for one epoch.

    Args:
        model: DSGCNet model
        criterion: Main criterion for point matching
        data_loader: Training data loader
        optimizer: Optimizer
        density_criterion: MSELoss for density maps
        device: Device to train on
        epoch: Current epoch number
        max_norm: Max norm for gradient clipping
        cfg: Optional config for multi-scale density prediction
    """
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # Check if multi-scale density prediction is enabled
    density_cfg = getattr(cfg, "density_multi_scale", None) if cfg is not None else None
    use_multi_scale_density = bool(
        getattr(density_cfg, "enabled", False) if density_cfg is not None else False
    )

    for samples, targets, gt_dmap in data_loader:
        samples = samples.to(device)
        gt_dmap = torch.stack(gt_dmap)
        gt_dmap = gt_dmap.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        et_dmap = outputs["density_out"]

        # Compute density loss (single or multi-scale)
        if use_multi_scale_density and all(
            k in outputs for k in ["density_block3", "density_block4", "density_block5"]
        ):
            # Multi-scale density prediction
            # Resize GT to each prediction shape for robust supervision.
            gt_density_block3 = F.interpolate(
                gt_dmap,
                size=outputs["density_block3"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gt_density_block4 = F.interpolate(
                gt_dmap,
                size=outputs["density_block4"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gt_density_block5 = F.interpolate(
                gt_dmap,
                size=outputs["density_block5"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Compute individual losses
            loss_block3 = density_criterion(
                outputs["density_block3"], gt_density_block3
            )
            loss_block4 = density_criterion(
                outputs["density_block4"], gt_density_block4
            )
            loss_block5 = density_criterion(
                outputs["density_block5"], gt_density_block5
            )
            loss_orig = density_criterion(et_dmap, gt_dmap)

            # Get weights from config
            weights_cfg = getattr(density_cfg, "weights", None)
            w3 = float(getattr(weights_cfg, "block3", 1.0))
            w4 = float(getattr(weights_cfg, "block4", 1.0))
            w5 = float(getattr(weights_cfg, "block5", 1.0))
            w_orig = float(getattr(weights_cfg, "original", 1.0))

            # Weighted sum
            density_loss = (
                (
                    w3 * loss_block3
                    + w4 * loss_block4
                    + w5 * loss_block5
                    + w_orig * loss_orig
                )
                / gt_dmap.shape[0]
                * 0.01
            )

            # Log individual losses for monitoring
            metric_logger.update(
                den_loss_block3=(loss_block3 / gt_dmap.shape[0] * 0.01).item(),
                den_loss_block4=(loss_block4 / gt_dmap.shape[0] * 0.01).item(),
                den_loss_block5=(loss_block5 / gt_dmap.shape[0] * 0.01).item(),
                den_loss_orig=(loss_orig / gt_dmap.shape[0] * 0.01).item(),
            )
        else:
            # Single-scale density prediction (original behavior)
            density_loss = density_criterion(et_dmap, gt_dmap) / gt_dmap.shape[0] * 0.01

        loss_sum = losses + density_loss

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_sum.item()}, stopping training")
            logger.error(str(loss_dict_reduced))
            sys.exit(1)

        optimizer.zero_grad()
        loss_sum.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss_sum=loss_sum.item(),
            losses=loss_value,
            den_loss=density_loss.item(),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_crowd_no_overlap(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    vis_dir: Optional[str] = None,
) -> tuple[float, float, float, float]:
    """Evaluate on validation set (no overlap).

    Returns:
        (mae, mse, density_mae, density_mse)
    """
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    maes, mses, density_maes, density_mses = [], [], [], []

    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)

        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ]
        assert outputs_scores.shape[0] == 1, (
            "evaluate_crowd_no_overlap expects batch_size=1"
        )
        outputs_scores = outputs_scores[0]
        outputs_points = outputs["pred_points"][0]
        gt_cnt = targets[0]["point"].shape[0]
        threshold = 0.5

        points = (
            outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        )
        predict_cnt = int((outputs_scores > threshold).sum())

        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) ** 2
        maes.append(float(mae))
        mses.append(float(mse))

        et_dmap = outputs["density_out"]
        et_dmap_sum = int(torch.sum(et_dmap))
        density_mae = abs(et_dmap_sum - gt_cnt)
        density_mse = (et_dmap_sum - gt_cnt) ** 2
        density_maes.append(float(density_mae))
        density_mses.append(float(density_mse))

    mae = float(np.mean(maes))
    mse = float(np.sqrt(np.mean(mses)))
    density_mae = float(np.mean(density_maes))
    density_mse = float(np.sqrt(np.mean(density_mses)))
    return mae, mse, density_mae, density_mse

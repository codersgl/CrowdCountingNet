"""Training and evaluation loops for DSGCNet.

Adapted from engine.py — functional logic unchanged,
print() statements replaced with loguru logger.
"""

from __future__ import annotations

import math
import sys
from typing import Iterable, Optional, cast

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
    ssim_criterion: nn.Module | None = None,
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
    density_loss_weight = (
        float(getattr(cfg, "density_loss_weight", 0.01)) if cfg is not None else 0.01
    )
    density_ssim_cfg = getattr(cfg, "density_ssim", None) if cfg is not None else None
    use_density_ssim = (
        bool(
            getattr(density_ssim_cfg, "enabled", False)
            if density_ssim_cfg is not None
            else False
        )
        and ssim_criterion is not None
    )
    density_ssim_weight = (
        float(getattr(density_ssim_cfg, "weight", 0.005))
        if density_ssim_cfg is not None
        else 0.005
    )
    model_moe_cfg = getattr(getattr(cfg, "model", None), "moe", None)
    moe_aux_weight = (
        float(getattr(model_moe_cfg, "aux_loss_weight", 1.0))
        if model_moe_cfg is not None
        else 1.0
    )
    moe_temperature_decay = (
        float(getattr(model_moe_cfg, "temperature_decay", 0.9999))
        if model_moe_cfg is not None
        else 0.9999
    )
    fg_loss_weight = (
        float(getattr(cfg, "fg_loss_weight", 0.1)) if cfg is not None else 0.1
    )
    fg_pos_weight = (
        float(getattr(cfg, "fg_pos_weight", 5.0)) if cfg is not None else 5.0
    )
    use_depth = bool(
        getattr(getattr(cfg, "model", None), "use_depth", False)
        if cfg is not None
        else False
    )
    use_depth_geo = bool(
        getattr(getattr(cfg, "model", None), "use_depth_geo", False)
        if cfg is not None
        else False
    )
    use_depth = use_depth or use_depth_geo  # either flag requires depth data in batch

    for batch in data_loader:
        if use_depth:
            samples, targets, gt_dmap, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets, gt_dmap = batch
            depth_map = None
        samples = samples.to(device)
        gt_dmap = torch.stack(gt_dmap)
        gt_dmap = gt_dmap.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if depth_map is not None:
            outputs = model(samples, depth_map=depth_map)
        else:
            outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = cast(dict[str, torch.Tensor | float], criterion.weight_dict)
        losses = sum(
            (
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            ),
            torch.tensor(0.0, device=samples.device),
        )

        et_dmap = outputs["density_out"]

        # Crop predicted density map to GT size (padding from collation may
        # make the prediction spatially larger than the un-padded GT).
        if et_dmap.shape[-2:] != gt_dmap.shape[-2:]:
            gt_h, gt_w = gt_dmap.shape[-2:]
            et_dmap = et_dmap[:, :, :gt_h, :gt_w]

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
                * density_loss_weight
            )

            # Log individual losses for monitoring
            metric_logger.update(
                den_loss_block3=(
                    loss_block3 / gt_dmap.shape[0] * density_loss_weight
                ).item(),
                den_loss_block4=(
                    loss_block4 / gt_dmap.shape[0] * density_loss_weight
                ).item(),
                den_loss_block5=(
                    loss_block5 / gt_dmap.shape[0] * density_loss_weight
                ).item(),
                den_loss_orig=(
                    loss_orig / gt_dmap.shape[0] * density_loss_weight
                ).item(),
            )
        else:
            # Single-scale density prediction (original behavior)
            density_loss = (
                density_criterion(et_dmap, gt_dmap)
                / gt_dmap.shape[0]
                * density_loss_weight
            )

        density_ssim_loss = torch.tensor(0.0, device=samples.device)
        if use_density_ssim:
            assert ssim_criterion is not None
            density_ssim_loss = density_ssim_weight * ssim_criterion(et_dmap, gt_dmap)
            density_loss = density_loss + density_ssim_loss

        moe_aux_total = outputs.get("moe_aux_total")
        moe_aux_component = torch.tensor(0.0, device=samples.device)
        if moe_aux_total is not None:
            moe_aux_component = moe_aux_weight * moe_aux_total

        # Foreground suppression branch loss
        fg_loss = torch.tensor(0.0, device=samples.device)
        fg_logits = outputs.get("fg_logits")
        if fg_logits is not None:
            fg_gt = F.adaptive_max_pool2d(
                (gt_dmap > 0).float(), output_size=fg_logits.shape[-2:]
            )
            fg_loss = (
                F.binary_cross_entropy_with_logits(
                    fg_logits,
                    fg_gt,
                    pos_weight=torch.tensor(fg_pos_weight, device=samples.device),
                )
                * fg_loss_weight
            )

        loss_sum = losses + density_loss + moe_aux_component + fg_loss

        loss_dict_reduced = reduce_dict(loss_dict)
        # Only log losses whose weight is non-zero (or have no weight entry)
        active_keys = {
            k for k in loss_dict_reduced if k not in weight_dict or weight_dict[k]
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items() if k in active_keys
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict and k in active_keys
        }
        losses_reduced_scaled = sum(
            loss_dict_reduced_scaled.values(),
            torch.tensor(0.0, device=samples.device),
        )
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_sum.item()):
            logger.error(
                f"loss_sum is {loss_sum.item()} ("
                f"task={loss_value:.4f}, "
                f"density={density_loss.item():.4f}, "
                f"moe_aux={moe_aux_component.item():.4f}), stopping training"
            )
            logger.error(str(loss_dict_reduced))
            sys.exit(1)

        optimizer.zero_grad()
        loss_sum.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if fg_logits is not None:
            metric_logger.update(fg_loss=fg_loss.item())

        update_temperature = getattr(model, "update_moe_temperature", None)
        if callable(update_temperature):
            update_temperature(decay_rate=moe_temperature_decay)

        metric_logger.update(
            loss_sum=loss_sum.item(),
            losses=loss_value,
            den_loss=density_loss.item(),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        # Log DM-Count sub-components when available
        _dm_components = getattr(density_criterion, "last_components", None)
        if _dm_components:
            metric_logger.update(**_dm_components)
        if use_density_ssim:
            metric_logger.update(den_ssim=density_ssim_loss.item())

        if moe_aux_total is not None:
            metric_logger.update(
                moe_aux_total=moe_aux_component.item(),
                moe_aux_raw=float(moe_aux_total.item()),
            )
            moe_aux_losses = outputs.get("moe_aux_losses") or {}
            for key in ("l_balance", "l_decorr"):
                if key in moe_aux_losses:
                    metric_logger.update(**{key: float(moe_aux_losses[key].item())})

            moe_module = getattr(model, "moe", None) or getattr(
                model, "light_moe", None
            )
            if moe_module is not None:
                if hasattr(moe_module, "temperature"):
                    metric_logger.update(moe_temperature=float(moe_module.temperature))
                # Log EMA expert usage spread for monitoring load balance
                if hasattr(moe_module, "ema_usage"):
                    ema_u = moe_module.ema_usage
                    metric_logger.update(
                        ema_usage_min=float(ema_u.min().item()),
                        ema_usage_max=float(ema_u.max().item()),
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
    use_depth: bool = False,
    cfg=None,
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

    for batch in data_loader:
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None
        samples = samples.to(device)
        if depth_map is not None:
            outputs = model(samples, depth_map=depth_map)
        else:
            outputs = model(samples)

        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ]
        assert outputs_scores.shape[0] == 1, (
            "evaluate_crowd_no_overlap expects batch_size=1"
        )
        outputs_scores = outputs_scores[0]
        gt_cnt = targets[0]["point"].shape[0]

        # Compute density map integral first (needed by density_guided)
        et_dmap = outputs["density_out"]
        et_dmap_sum = float(torch.sum(et_dmap).item())

        # Parse counting config
        eval_cfg = getattr(cfg, "eval_counting", None) if cfg is not None else None
        counting_method = (
            str(getattr(eval_cfg, "method", "threshold")) if eval_cfg else "threshold"
        )

        if counting_method == "density_guided":
            min_score = float(getattr(eval_cfg, "min_score", 0.3))
            density_cnt = max(1, round(et_dmap_sum))
            # Filter by minimum confidence
            valid_mask = outputs_scores > min_score
            num_valid = int(valid_mask.sum().item())
            if num_valid > 0 and density_cnt <= num_valid:
                # Take top-k scores where k = density_cnt
                _, topk_indices = outputs_scores[valid_mask].topk(
                    min(density_cnt, num_valid)
                )
                predict_cnt = len(topk_indices)
            else:
                # All valid points count (density overestimates or equals)
                predict_cnt = num_valid
        else:
            threshold = float(getattr(eval_cfg, "threshold", 0.5)) if eval_cfg else 0.5
            predict_cnt = int((outputs_scores > threshold).sum())

        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) ** 2
        maes.append(float(mae))
        mses.append(float(mse))

        density_mae = abs(et_dmap_sum - gt_cnt)
        density_mse = (et_dmap_sum - gt_cnt) ** 2
        density_maes.append(float(density_mae))
        density_mses.append(float(density_mse))

    mae = float(np.mean(maes))
    mse = float(np.sqrt(np.mean(mses)))
    density_mae = float(np.mean(density_maes))
    density_mse = float(np.sqrt(np.mean(density_mses)))
    return mae, mse, density_mae, density_mse

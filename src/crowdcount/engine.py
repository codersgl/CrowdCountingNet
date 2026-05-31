"""Training and evaluation loops for DSGCNet.

Adapted from engine.py — functional logic unchanged,
print() statements replaced with loguru logger.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

from crowdcount.data.transforms import DeNormalize
from crowdcount.eval.tta import count_from_outputs, tta_predict
from crowdcount.utils.misc import MetricLogger, SmoothedValue, reduce_dict
from loguru import logger

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _density_heatmap(density: torch.Tensor) -> torch.Tensor:
    """Convert a 2D density map to a JET RGB heatmap [3, H, W] in [0, 1]."""
    import matplotlib

    d = density.detach().cpu().float()
    vmax = torch.quantile(d, 0.995).clamp_min(1e-8)
    d = (d / vmax).clamp(0, 1)
    jet = matplotlib.colormaps["jet"]
    rgb = jet(d.numpy())[..., :3].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _save_expert_route_image(top1: torch.Tensor, output_path: Path) -> torch.Tensor:
    """Save per-pixel expert routing map as RGB image. Returns [3, H, W] tensor."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = top1.new_tensor(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=torch.uint8,
    )
    top1_cpu = top1[0].detach().to(torch.long).cpu().clamp(0, 2)
    rgb = palette.cpu()[top1_cpu]
    Image.fromarray(rgb.numpy()).save(output_path)
    return rgb.permute(2, 0, 1).float() / 255.0


def _save_density_overlay(
    sample: torch.Tensor,
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    output_path: Path,
    pred_points: torch.Tensor | None = None,
    gt_points: torch.Tensor | None = None,
    point_radius: int = 2,
) -> torch.Tensor:
    """Save original | pred-overlay | gt-overlay side-by-side.

    Optionally draws predicted points (red) and GT points (green) as circles.
    Returns [3, H, 3*W] tensor for TensorBoard.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denorm = DeNormalize(_IMAGENET_MEAN, _IMAGENET_STD)
    img = denorm(sample[0].detach().cpu().clone()).clamp(0, 1)
    pred = pred_density[0].detach().cpu()
    gt = gt_density[0].detach().cpu()
    pred_count = float(pred.sum().item())
    gt_count = float(gt.sum().item())
    img_h, img_w = img.shape[-2:]

    pred_full = F.interpolate(
        pred.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False,
    ).squeeze(0)
    gt_full = F.interpolate(
        gt.unsqueeze(0).float(), size=(img_h, img_w), mode="bilinear", align_corners=False,
    ).squeeze(0)

    def _overlay(image: torch.Tensor, density: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
        hm = _density_heatmap(density.squeeze(0))
        return image * (1 - alpha) + hm * alpha

    overlay_pred = _overlay(img, pred_full)
    overlay_gt = _overlay(img, gt_full)
    combined = torch.cat([img, overlay_pred, overlay_gt], dim=-1)
    combined_uint8 = (combined.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()

    pil_img = Image.fromarray(combined_uint8)
    draw = ImageDraw.Draw(pil_img)
    w = img_w
    draw.text((w + 8, 8), f"Pred: {pred_count:.1f}", fill=(255, 255, 0))
    draw.text((2 * w + 8, 8), f"GT:   {gt_count:.1f}", fill=(255, 255, 0))

    # Draw predicted points on the pred-overlay panel (cyan fill, white outline, offset by w)
    if pred_points is not None and pred_points.numel() > 0:
        pts = pred_points.detach().cpu()
        for x, y in pts:
            _x = float(x) + w
            _y = float(y)
            draw.ellipse(
                (_x - point_radius, _y - point_radius, _x + point_radius, _y + point_radius),
                fill=(0, 255, 255), outline=(255, 255, 255),
            )

    # Draw GT points on the GT-overlay panel (magenta fill, white outline, offset by 2*w)
    if gt_points is not None and gt_points.numel() > 0:
        pts = gt_points.detach().cpu()
        for x, y in pts:
            _x = float(x) + 2 * w
            _y = float(y)
            draw.ellipse(
                (_x - point_radius, _y - point_radius, _x + point_radius, _y + point_radius),
                fill=(255, 0, 255), outline=(255, 255, 255),
            )

    pil_img.save(output_path)
    return combined


def _featmap_to_image(feat: torch.Tensor, output_path: Path) -> torch.Tensor:
    """Convert a [B, C, H, W] feature map to a 3-channel RGB image for TensorBoard.

    Uses mean over channels → pseudo-color (red=high, blue=low) for visualisation.
    Returns [3, H, W] float tensor in [0, 1].
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = feat[0].detach().cpu().float()  # [C, H, W]
    # Mean activation across channels as a proxy for "where the stream responds"
    mean_act = f.mean(dim=0)  # [H, W]
    # Normalize to [0, 1] per-sample for consistent rendering
    a_min, a_max = mean_act.min(), mean_act.max()
    if a_max - a_min > 1e-8:
        mean_act = (mean_act - a_min) / (a_max - a_min)
    else:
        mean_act = mean_act.zero_()
    # Simple heatmap: red=high, blue=low
    h, w = mean_act.shape
    rgb = torch.zeros(3, h, w)
    rgb[0] = mean_act  # R channel = activation
    rgb[2] = 1.0 - mean_act  # B channel = inverse
    rgb = rgb.clamp(0, 1)
    # Save
    rgb_uint8 = (rgb.permute(1, 2, 0) * 255).byte().numpy()
    Image.fromarray(rgb_uint8).save(output_path)
    return rgb


def _featmap_diff_image(
    feat_a: torch.Tensor, feat_b: torch.Tensor, output_path: Path,
) -> torch.Tensor:
    """Render the absolute difference between two feature maps.

    Uses channel-mean of both, computes |mean(a) - mean(b)|, and renders
    with a shared normalisation based on feat_a's dynamic range so the
    difference is comparable across epochs.
    Returns [3, H, W] float tensor in [0, 1].
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fa = feat_a[0].detach().cpu().float()  # [C, H, W]
    fb = feat_b[0].detach().cpu().float()  # [C, H, W]
    ma = fa.mean(dim=0)  # [H, W]
    mb = fb.mean(dim=0)  # [H, W]
    diff = (ma - mb).abs()  # [H, W]

    # Shared scale: % of feat_a's dynamic range
    a_range = ma.max() - ma.min()
    a_range = max(a_range.item(), 1e-8)
    diff_clipped = (diff / a_range).clamp(0, 1)

    h, w = diff.shape
    rgb = torch.zeros(3, h, w)
    rgb[0] = diff_clipped          # R = larger diff
    rgb[1] = diff_clipped * 0.3   # G = slight green tint
    rgb[2] = 1.0 - diff_clipped   # B = inverse
    rgb = rgb.clamp(0, 1)

    rgb_uint8 = (rgb.permute(1, 2, 0) * 255).byte().numpy()
    Image.fromarray(rgb_uint8).save(output_path)
    return rgb


def _eval_point_head(
    pred_logits: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    match_threshold: float = 8.0,
    cls_threshold: float = 0.5,
) -> dict[str, float] | None:
    """Compute point head metrics (MAE, precision, recall) via Hungarian matching."""
    gt_count = gt_points.shape[0]
    if gt_count == 0:
        return None
    gt_points = gt_points.to(device=pred_points.device, dtype=pred_points.dtype)
    probs = pred_logits[0].softmax(dim=-1)
    fg_mask = probs[:, 1] > cls_threshold
    if not fg_mask.any():
        return {"mae": float(gt_count), "precision": 0.0, "recall": 0.0}
    fg_pts = pred_points[0][fg_mask]
    pred_count = fg_pts.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    cost = torch.cdist(fg_pts.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)
    cost_np = cost.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    matched = sum(1 for r, c in zip(row_ind, col_ind) if cost_np[r, c] <= match_threshold)
    return {
        "mae": float(abs(pred_count - gt_count)),
        "precision": matched / max(pred_count, 1),
        "recall": matched / max(gt_count, 1),
    }


def _forward_model(
    model: nn.Module,
    samples: torch.Tensor,
    depth_map: torch.Tensor | None = None,
    targets: list[dict[str, torch.Tensor]] | None = None,
    gt_density: torch.Tensor | None = None,
) -> dict:
    signature = inspect.signature(model.forward)
    kwargs: dict[str, object] = {}
    if depth_map is not None and "depth_map" in signature.parameters:
        kwargs["depth_map"] = depth_map
    if targets is not None and "targets" in signature.parameters:
        kwargs["targets"] = targets
    if gt_density is not None and "gt_density" in signature.parameters:
        kwargs["gt_density"] = gt_density
    return model(samples, **kwargs)


def _resize_depth_target(
    depth_map: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)
    if depth_map.shape[1] != 1:
        raise ValueError(f"depth_map must have one channel, got {depth_map.shape[1]}")
    depth_target = depth_map.to(device=prediction.device, dtype=prediction.dtype)
    if depth_target.shape[-2:] != prediction.shape[-2:]:
        depth_target = F.interpolate(
            depth_target,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return depth_target.clamp(0.0, 1.0)


def _sobel_gradients(depth_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = depth_tensor.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = depth_tensor.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(depth_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(depth_tensor, sobel_y, padding=1)
    return grad_x, grad_y


def _density_focus_weights(
    gt_density: torch.Tensor,
    prediction: torch.Tensor,
    focus_weight: float,
) -> torch.Tensor | None:
    if focus_weight <= 0:
        return None
    density_focus = gt_density.detach().to(
        device=prediction.device, dtype=prediction.dtype
    )
    if density_focus.shape[-2:] != prediction.shape[-2:]:
        density_focus = F.interpolate(
            density_focus,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    density_focus = torch.log1p(density_focus.clamp_min(0.0))
    focus_max = density_focus.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return 1.0 + float(focus_weight) * (density_focus / focus_max)


def _weighted_mean(
    loss_map: torch.Tensor,
    weight_map: torch.Tensor | None,
) -> torch.Tensor:
    if weight_map is None:
        return loss_map.mean()
    return (loss_map * weight_map).sum() / weight_map.sum().clamp_min(1e-6)


def _compute_depth_aux_loss(
    prediction: torch.Tensor,
    depth_map: torch.Tensor,
    gt_density: torch.Tensor,
    depth_aux_cfg: object | None,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    loss_type = str(getattr(depth_aux_cfg, "loss_type", "smooth_l1")).lower()
    if loss_type not in {"smooth_l1", "l1", "mse"}:
        raise ValueError("depth_aux.loss_type must be one of smooth_l1, l1, mse")
    loss_weight = float(getattr(depth_aux_cfg, "loss_weight", 0.05))
    gradient_weight = float(getattr(depth_aux_cfg, "gradient_weight", 0.0))
    density_focus_weight = float(getattr(depth_aux_cfg, "density_focus_weight", 0.0))
    smooth_l1_beta = float(getattr(depth_aux_cfg, "smooth_l1_beta", 0.1))
    warmup_epochs = int(getattr(depth_aux_cfg, "warmup_epochs", 0))

    depth_target = _resize_depth_target(depth_map, prediction)
    focus_weights = _density_focus_weights(
        gt_density,
        prediction,
        density_focus_weight,
    )
    if loss_type == "smooth_l1":
        pixel_loss_map = F.smooth_l1_loss(
            prediction,
            depth_target,
            beta=smooth_l1_beta,
            reduction="none",
        )
    elif loss_type == "l1":
        pixel_loss_map = F.l1_loss(prediction, depth_target, reduction="none")
    else:
        pixel_loss_map = F.mse_loss(prediction, depth_target, reduction="none")
    pixel_loss = _weighted_mean(pixel_loss_map, focus_weights)

    grad_loss = prediction.new_tensor(0.0)
    if gradient_weight > 0:
        pred_grad_x, pred_grad_y = _sobel_gradients(prediction)
        target_grad_x, target_grad_y = _sobel_gradients(depth_target)
        grad_loss = 0.5 * (
            F.l1_loss(pred_grad_x, target_grad_x)
            + F.l1_loss(pred_grad_y, target_grad_y)
        )

    if warmup_epochs > 0:
        warmup_factor = min(1.0, float(epoch + 1) / float(warmup_epochs))
    else:
        warmup_factor = 1.0
    total_loss = (pixel_loss + gradient_weight * grad_loss) * loss_weight
    return total_loss * warmup_factor, pixel_loss, grad_loss, warmup_factor


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
    uncertainty_weighter: nn.Module | None = None,
    writer=None,
    vis_dir: str | None = None,
    vis_interval: int = 500,
    density_mse_aux: nn.Module | None = None,
    density_mse_aux_weight: float = 0.0,
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
    consistency_weight = float(
        getattr(density_cfg, "consistency_weight", 0.0)
        if density_cfg is not None
        else 0.0
    )
    count_consistency_weight = float(
        getattr(density_cfg, "count_consistency_weight", 0.0)
        if density_cfg is not None
        else 0.0
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
    model_graph_attn_moe_cfg = getattr(
        getattr(cfg, "model", None), "graph_attn_moe", None
    )
    model_graph_moe_cfg = getattr(getattr(cfg, "model", None), "graph_moe", None)
    model_neck_moe_cfg = getattr(getattr(cfg, "model", None), "neck_moe", None)
    fusion_mode = str(getattr(getattr(cfg, "model", None), "fusion_mode", "gcn"))
    use_neck_moe = bool(
        getattr(getattr(cfg, "model", None), "use_neck_moe", False)
        if cfg is not None
        else False
    )
    moe_aux_weight = (
        float(getattr(model_moe_cfg, "aux_loss_weight", 1.0))
        if model_moe_cfg is not None
        else 1.0
    )
    if use_neck_moe and fusion_mode == "gcn" and model_neck_moe_cfg is not None:
        moe_aux_weight = float(
            getattr(model_neck_moe_cfg, "aux_loss_weight", moe_aux_weight)
        )
    if fusion_mode == "graph_attn_moe" and model_graph_attn_moe_cfg is not None:
        moe_aux_weight = float(
            getattr(model_graph_attn_moe_cfg, "aux_loss_weight", moe_aux_weight)
        )
    if fusion_mode == "graph_moe" and model_graph_moe_cfg is not None:
        moe_aux_weight = float(
            getattr(model_graph_moe_cfg, "aux_loss_weight", moe_aux_weight)
        )
    model_sdd_moe_cfg = getattr(getattr(cfg, "model", None), "sdd_moe", None)
    if fusion_mode == "sdd_moe" and model_sdd_moe_cfg is not None:
        moe_aux_weight = float(
            getattr(model_sdd_moe_cfg, "aux_loss_weight", moe_aux_weight)
        )
    model_moecount_moe_cfg = getattr(getattr(cfg, "model", None), "moecount_moe", None)
    if fusion_mode == "moe" and model_moecount_moe_cfg is not None:
        moe_aux_weight = float(
            getattr(model_moecount_moe_cfg, "aux_loss_weight", moe_aux_weight)
        )
    moe_temperature_decay = (
        float(getattr(model_moe_cfg, "temperature_decay", 0.9999))
        if model_moe_cfg is not None
        else 0.9999
    )
    if fusion_mode == "sdd_moe" and model_sdd_moe_cfg is not None:
        moe_temperature_decay = float(
            getattr(model_sdd_moe_cfg, "gumbel_temp_decay", moe_temperature_decay)
        )
    fg_loss_weight = (
        float(getattr(cfg, "fg_loss_weight", 0.1)) if cfg is not None else 0.1
    )
    fg_pos_weight = (
        float(getattr(cfg, "fg_pos_weight", 5.0)) if cfg is not None else 5.0
    )
    prompt_cfg = getattr(getattr(cfg, "model", None), "clip_prompt_density", None)
    prompt_align_weight = (
        float(getattr(prompt_cfg, "align_loss_weight", 0.0))
        if prompt_cfg is not None
        else 0.0
    )
    prompt_align_pos_weight = (
        float(getattr(prompt_cfg, "align_pos_weight", 1.0))
        if prompt_cfg is not None
        else 1.0
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
    use_depth_geo_post = bool(
        getattr(getattr(cfg, "model", None), "use_depth_geo_post", False)
        if cfg is not None
        else False
    )
    use_depth_dual_vgg = bool(
        getattr(getattr(cfg, "model", None), "use_depth_dual_vgg", False)
        if cfg is not None
        else False
    )
    use_depth_attn = bool(
        getattr(getattr(cfg, "model", None), "use_depth_attn", False)
        if cfg is not None
        else False
    )
    use_depth_cross_attn = bool(
        getattr(getattr(cfg, "model", None), "use_depth_cross_attn", False)
        if cfg is not None
        else False
    )
    use_depth_aux = bool(
        getattr(getattr(cfg, "model", None), "use_depth_aux", False)
        if cfg is not None
        else False
    )
    depth_aux_cfg = (
        getattr(getattr(cfg, "model", None), "depth_aux", None)
        if cfg is not None
        else None
    )
    depth_graph_prior = (
        getattr(getattr(cfg, "model", None), "depth_graph_prior", None)
        if cfg is not None
        else None
    )
    use_depth_graph_prior = bool(
        getattr(depth_graph_prior, "enabled", False)
        if depth_graph_prior is not None
        else False
    )
    use_depth_gcn = bool(
        getattr(getattr(cfg, "model", None), "use_depth_gcn", False)
        if cfg is not None
        else False
    )
    use_depth_input = (
        use_depth
        or use_depth_geo
        or use_depth_geo_post
        or use_depth_dual_vgg
        or use_depth_attn
        or use_depth_cross_attn
        or use_depth_graph_prior
        or use_depth_gcn
    )
    needs_depth_batch = use_depth_input or use_depth_aux
    point_feedback_cfg = (
        getattr(getattr(cfg, "model", None), "point_density_feedback", None)
        if cfg is not None
        else None
    )
    point_feedback_warmup_epochs = int(
        getattr(point_feedback_cfg, "warmup_epochs", 0)
        if point_feedback_cfg is not None
        else 0
    )
    if point_feedback_warmup_epochs > 0:
        point_feedback_warmup_factor = min(
            1.0,
            float(epoch + 1) / float(point_feedback_warmup_epochs),
        )
    else:
        point_feedback_warmup_factor = 1.0

    batch_idx = 0

    for batch in data_loader:
        if needs_depth_batch:
            samples, targets, gt_dmap, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets, gt_dmap = batch
            depth_map = None
        samples = samples.to(device)
        gt_dmap = torch.stack(gt_dmap)
        gt_dmap = gt_dmap.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        depth_input_map = depth_map if use_depth_input else None

        outputs = _forward_model(
            model,
            samples,
            depth_map=depth_input_map,
            targets=targets,
            gt_density=gt_dmap,
        )
        loss_dict = criterion(outputs, targets)
        weight_dict = dict(cast(dict[str, torch.Tensor | float], criterion.weight_dict))
        if "loss_point_density_feedback" in weight_dict:
            weight_dict["loss_point_density_feedback"] = (
                weight_dict["loss_point_density_feedback"]
                * point_feedback_warmup_factor
            )

        # When uncertainty weighter is active, CE and regression losses are
        # weighted by learned σ parameters instead of fixed weight_dict values.
        if uncertainty_weighter is not None:
            # Sum only auxiliary losses (count, consistency, refine) with fixed weights
            _uw_keys = {"loss_ce", "loss_points"}
            losses = sum(
                (
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict and k not in _uw_keys
                ),
                torch.tensor(0.0, device=samples.device),
            )
        else:
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
            # Multi-scale density prediction.
            # Each block predicts a density map at a lower spatial
            # resolution. We upsample predictions to GT resolution before
            # computing the loss so that:
            #   (a) the GT keeps its original count-conserving values, and
            #   (b) all per-scale losses are directly comparable in magnitude.
            # Bilinear upsampling preserves pixel values (not the integral),
            # so we rescale by the area ratio so that
            # ``sum(pred_upsampled) == sum(pred_original)``. This way the
            # network is encouraged to produce count-consistent
            # low-resolution maps rather than count/(scale^2)-scaled maps.
            gt_size = gt_dmap.shape[-2:]
            gt_area = float(gt_size[0] * gt_size[1])

            def _upsample_count_preserving(d: torch.Tensor) -> torch.Tensor:
                area_ratio = (d.shape[-2] * d.shape[-1]) / gt_area
                up = F.interpolate(
                    d, size=gt_size, mode="bilinear", align_corners=False
                )
                return up * area_ratio

            pred_block3 = _upsample_count_preserving(outputs["density_block3"])
            pred_block4 = _upsample_count_preserving(outputs["density_block4"])
            pred_block5 = _upsample_count_preserving(outputs["density_block5"])

            loss_block3 = density_criterion(pred_block3, gt_dmap)
            loss_block4 = density_criterion(pred_block4, gt_dmap)
            loss_block5 = density_criterion(pred_block5, gt_dmap)
            loss_orig = density_criterion(et_dmap, gt_dmap)

            # Get weights from config and normalise so that they sum to 1.
            # This keeps ``density_loss_weight`` semantically equivalent
            # between single-scale and multi-scale modes.
            weights_cfg = getattr(density_cfg, "weights", None)
            w3 = float(getattr(weights_cfg, "block3", 1.0))
            w4 = float(getattr(weights_cfg, "block4", 1.0))
            w5 = float(getattr(weights_cfg, "block5", 1.0))
            w_orig = float(getattr(weights_cfg, "original", 1.0))
            w_total = w3 + w4 + w5 + w_orig
            if w_total <= 0:
                w_total = 1.0
            w3 /= w_total
            w4 /= w_total
            w5 /= w_total
            w_orig /= w_total

            density_loss_raw = (
                w3 * loss_block3
                + w4 * loss_block4
                + w5 * loss_block5
                + w_orig * loss_orig
            ) / gt_dmap.shape[0]
            if uncertainty_weighter is not None:
                # UW will apply its own learned weight; skip fixed scale.
                density_loss = density_loss_raw
            else:
                density_loss = density_loss_raw * density_loss_weight

            # Log individual losses for monitoring (in their final units).
            _log_scale = (
                1.0 if uncertainty_weighter is not None else density_loss_weight
            )
            metric_logger.update(
                den_loss_block3=(loss_block3 / gt_dmap.shape[0] * _log_scale).item(),
                den_loss_block4=(loss_block4 / gt_dmap.shape[0] * _log_scale).item(),
                den_loss_block5=(loss_block5 / gt_dmap.shape[0] * _log_scale).item(),
                den_loss_orig=(loss_orig / gt_dmap.shape[0] * _log_scale).item(),
            )

            # Cross-scale consistency loss: enforce aligned predictions
            if consistency_weight > 0:
                target_size = et_dmap.shape[-2:]
                target_area = float(target_size[0] * target_size[1])

                def _to_target_count_preserving(d: torch.Tensor) -> torch.Tensor:
                    # Mirror the supervision-time rescaling so that the
                    # low-resolution prediction is interpreted at the same
                    # count scale as ``et_dmap`` (otherwise count-preserving
                    # low-res maps -- which intentionally have much larger
                    # per-pixel values -- would be dragged back down by this
                    # loss, fighting the main density loss).
                    area_ratio = (d.shape[-2] * d.shape[-1]) / target_area
                    up = F.interpolate(
                        d, size=target_size, mode="bilinear", align_corners=False
                    )
                    return up * area_ratio

                d3_aligned = _to_target_count_preserving(outputs["density_block3"])
                d4_aligned = _to_target_count_preserving(outputs["density_block4"])
                d5_aligned = _to_target_count_preserving(outputs["density_block5"])
                # All three predictions are pulled toward the canonical
                # PA-FPN density (detached). Bidirectional L1 between
                # sibling outputs would encourage mutual collapse instead
                # of consistency.
                teacher = et_dmap.detach()
                consist_loss = (
                    (
                        F.l1_loss(d3_aligned, teacher)
                        + F.l1_loss(d4_aligned, teacher)
                        + F.l1_loss(d5_aligned, teacher)
                    )
                    / 3.0
                    * consistency_weight
                )
                density_loss = density_loss + consist_loss
                metric_logger.update(den_consist=consist_loss.item())

            # Count consistency loss: each scale's density integral should
            # match the GT count. Comparing means across scales is wrong
            # for count-preserving density maps (mean = count / area, which
            # depends on resolution). We compare per-sample sums instead.
            if count_consistency_weight > 0:
                gt_count = gt_dmap.sum(dim=(1, 2, 3)).detach()  # [B]
                sum_block3 = outputs["density_block3"].sum(dim=(1, 2, 3))
                sum_block4 = outputs["density_block4"].sum(dim=(1, 2, 3))
                sum_block5 = outputs["density_block5"].sum(dim=(1, 2, 3))
                sum_orig = et_dmap.sum(dim=(1, 2, 3))
                count_loss = (
                    (
                        F.l1_loss(sum_block3, gt_count)
                        + F.l1_loss(sum_block4, gt_count)
                        + F.l1_loss(sum_block5, gt_count)
                        + F.l1_loss(sum_orig, gt_count)
                    )
                    / 4.0
                    * count_consistency_weight
                )
                density_loss = density_loss + count_loss
                metric_logger.update(den_count_consist=count_loss.item())
        else:
            # Single-scale density prediction (original behavior).
            # Some density criteria (e.g. Bayesian Loss) are point-supervised
            # rather than density-map supervised, and consume `targets` and
            # `image_sizes` directly.  The trainer guards against pairing
            # such criteria with multi-scale density supervision, so the
            # branch above never dispatches them.
            if getattr(density_criterion, "requires_points", False):
                density_loss_raw = (
                    density_criterion(
                        et_dmap,
                        gt_dmap,
                        targets=targets,
                        image_sizes=samples.shape[-2:],
                    )
                    / gt_dmap.shape[0]
                )
            else:
                density_loss_raw = (
                    density_criterion(et_dmap, gt_dmap) / gt_dmap.shape[0]
                )
            if uncertainty_weighter is not None:
                # UW will apply its own learned weight; skip fixed scale
                density_loss = density_loss_raw
            else:
                density_loss = density_loss_raw * density_loss_weight

        density_ssim_loss = torch.tensor(0.0, device=samples.device)
        if use_density_ssim:
            assert ssim_criterion is not None
            density_ssim_loss = density_ssim_weight * ssim_criterion(et_dmap, gt_dmap)
            density_loss = density_loss + density_ssim_loss

        density_mse_aux_loss = torch.tensor(0.0, device=samples.device)
        if density_mse_aux is not None and density_mse_aux_weight > 0:
            density_mse_aux_loss = (
                density_mse_aux(et_dmap, gt_dmap) / gt_dmap.shape[0] * density_mse_aux_weight
            )
            density_loss = density_loss + density_mse_aux_loss

        depth_aux_loss = torch.tensor(0.0, device=samples.device)
        depth_aux_pixel_loss = torch.tensor(0.0, device=samples.device)
        depth_aux_grad_loss = torch.tensor(0.0, device=samples.device)
        depth_aux_warmup = 0.0
        if use_depth_aux:
            depth_aux_out = outputs.get("depth_aux_out")
            if depth_aux_out is None:
                raise ValueError(
                    "model.use_depth_aux=True but model output has no depth_aux_out"
                )
            if depth_map is None:
                raise ValueError(
                    "model.use_depth_aux=True requires depth maps in the training batch"
                )
            (
                depth_aux_loss,
                depth_aux_pixel_loss,
                depth_aux_grad_loss,
                depth_aux_warmup,
            ) = _compute_depth_aux_loss(
                depth_aux_out,
                depth_map,
                gt_dmap,
                depth_aux_cfg,
                epoch,
            )

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

        prompt_align_loss = torch.tensor(0.0, device=samples.device)
        prompt_logits = outputs.get("clip_prompt_foreground_logits")
        if prompt_logits is not None and prompt_align_weight > 0:
            prompt_target = gt_dmap.float().clamp_min(0.0)
            if prompt_target.shape[-2:] != prompt_logits.shape[-2:]:
                prompt_target = F.interpolate(
                    prompt_target,
                    size=prompt_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prompt_target = torch.log1p(prompt_target)
            target_max = prompt_target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            prompt_target = (prompt_target / target_max).to(dtype=prompt_logits.dtype)
            prompt_align_loss = (
                F.binary_cross_entropy_with_logits(
                    prompt_logits,
                    prompt_target,
                    pos_weight=torch.tensor(
                        prompt_align_pos_weight,
                        device=samples.device,
                        dtype=prompt_logits.dtype,
                    ),
                )
                * prompt_align_weight
            )

        if uncertainty_weighter is not None:
            # Learned weighting for the three main branches
            loss_ce_raw = loss_dict.get(
                "loss_ce", torch.tensor(0.0, device=samples.device)
            )
            loss_reg_raw = loss_dict.get(
                "loss_points", torch.tensor(0.0, device=samples.device)
            )
            uw_loss = uncertainty_weighter(density_loss, loss_ce_raw, loss_reg_raw)
            loss_sum = (
                uw_loss
                + losses
                + moe_aux_component
                + fg_loss
                + prompt_align_loss
                + depth_aux_loss
            )
        else:
            loss_sum = (
                losses
                + density_loss
                + moe_aux_component
                + fg_loss
                + prompt_align_loss
                + depth_aux_loss
            )

        # SA-DGAT auxiliary losses: local count ranking loss
        sa_dgat_ranking_loss = torch.tensor(0.0, device=samples.device)
        if fusion_mode == "sa_dgat" and cfg is not None:
            _sa_dgat_cfg = getattr(getattr(cfg, "model", None), "sa_dgat", None)
            ranking_weight = float(
                getattr(_sa_dgat_cfg, "ranking_loss_weight", 0.0)
                if _sa_dgat_cfg is not None
                else 0.0
            )
            if ranking_weight > 0:
                from crowdcount.plugins.sa_dgat.ranking_loss import (
                    LocalCountRankingLoss,
                )

                if not hasattr(train_one_epoch, "_ranking_loss"):
                    train_one_epoch._ranking_loss = LocalCountRankingLoss(
                        grid_size=int(
                            getattr(_sa_dgat_cfg, "ranking_grid_size", 4)
                            if _sa_dgat_cfg
                            else 4
                        ),
                        margin=float(
                            getattr(_sa_dgat_cfg, "ranking_margin", 1.0)
                            if _sa_dgat_cfg
                            else 1.0
                        ),
                        num_pairs=int(
                            getattr(_sa_dgat_cfg, "ranking_num_pairs", 16)
                            if _sa_dgat_cfg
                            else 16
                        ),
                    ).to(samples.device)
                sa_dgat_ranking_loss = (
                    train_one_epoch._ranking_loss(et_dmap, gt_dmap) * ranking_weight
                )
                loss_sum = loss_sum + sa_dgat_ranking_loss

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
                f"moe_aux={moe_aux_component.item():.4f}, "
                f"depth_aux={depth_aux_loss.item():.4f}), stopping training"
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
        if prompt_align_loss.item() > 0:
            metric_logger.update(prompt_align_loss=prompt_align_loss.item())
        if use_depth_aux:
            metric_logger.update(
                depth_aux_loss=depth_aux_loss.item(),
                depth_aux_pixel_loss=depth_aux_pixel_loss.item(),
                depth_aux_grad_loss=depth_aux_grad_loss.item(),
                depth_aux_warmup=depth_aux_warmup,
            )
        if (
            point_feedback_warmup_epochs > 0
            and "loss_point_density_feedback" in loss_dict
        ):
            metric_logger.update(
                point_density_feedback_warmup=point_feedback_warmup_factor
            )

        if fusion_mode == "sa_dgat":
            if sa_dgat_ranking_loss.item() > 0:
                metric_logger.update(ranking_loss=sa_dgat_ranking_loss.item())

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
        if density_mse_aux is not None and density_mse_aux_weight > 0:
            metric_logger.update(density_mse_aux=density_mse_aux_loss.item())

        if moe_aux_total is not None:
            metric_logger.update(
                moe_aux_total=moe_aux_component.item(),
                moe_aux_raw=float(moe_aux_total.item()),
            )
            moe_aux_losses = outputs.get("moe_aux_losses") or {}
            for key in (
                "l_balance",
                "l_importance",
                "l_capacity",
                "l_router_z",
                "router_entropy",
                "l_decorr",
                "l_scale",
                "l_ssim",
                "neck_l_balance",
                "neck_entropy",
                # --- New expert-specific aux losses ---
                "l_pl_point",
                "l_pl_point_cls",
                "l_pl_point_reg",
                "l_occ_consistency",
                "l_dp_class",
            ):
                if key in moe_aux_losses:
                    metric_logger.update(**{key: float(moe_aux_losses[key].item())})

            moe_module = (
                getattr(model, "moe", None)
                or getattr(model, "light_moe", None)
                or getattr(model, "graph_moe", None)
                or getattr(model, "graph_attn_moe", None)
                or getattr(model, "sdd_moe", None)
                or getattr(model, "neck_moe", None)
                or getattr(model, "moecount_moe", None)
            )
            if moe_module is not None:
                # Temperature: direct attr or nested in router (SDDMoE)
                _temp = getattr(moe_module, "temperature", None)
                if _temp is None:
                    _router = getattr(moe_module, "router", None)
                    _temp = (
                        getattr(_router, "temperature", None)
                        if _router is not None
                        else None
                    )
                if _temp is not None:
                    metric_logger.update(moe_temperature=float(_temp))
                # Log EMA expert usage spread for monitoring load balance
                if hasattr(moe_module, "ema_usage"):
                    ema_u = moe_module.ema_usage
                    metric_logger.update(
                        ema_usage_min=float(ema_u.min().item()),
                        ema_usage_max=float(ema_u.max().item()),
                    )

            # --- MoECountNet MoE metrics ---
            moe_load_fraction = outputs.get("moe_load_fraction")
            if moe_load_fraction is not None and isinstance(moe_load_fraction, torch.Tensor):
                lf = moe_load_fraction.detach().cpu()
                for i, val in enumerate(lf.tolist(), start=1):
                    metric_logger.update(**{f"moe_e{i}_load": float(val)})

            moe_entropy = outputs.get("moe_entropy")
            if moe_entropy is not None:
                if isinstance(moe_entropy, torch.Tensor):
                    metric_logger.update(moe_entropy=float(moe_entropy.item()))
                elif isinstance(moe_entropy, bool):
                    pass

            expert_similarity = outputs.get("expert_similarity") or {}
            if isinstance(expert_similarity, dict):
                for sim_key, sim_val in expert_similarity.items():
                    if isinstance(sim_val, torch.Tensor):
                        metric_logger.update(**{f"expert_{sim_key}": float(sim_val.item())})

            moe_warmup = outputs.get("moe_warmup_active")
            if isinstance(moe_warmup, bool):
                metric_logger.update(moe_warmup=1.0 if moe_warmup else 0.0)

            # --- Visualization (expert route + density overlay) ---
            if vis_dir is not None and vis_interval > 0:
                if batch_idx % vis_interval == 0:
                    moe_top1 = outputs.get("moe_top1")
                    if moe_top1 is not None and isinstance(moe_top1, torch.Tensor):
                        route_path = Path(vis_dir) / "expert_route_latest.png"
                        route_img = _save_expert_route_image(moe_top1, route_path)
                        if writer is not None:
                            writer.add_image("moe/top1_expert", route_img, epoch)

                        pd_density = outputs.get("pred_density")
                        if pd_density is None:
                            pd_density = outputs.get("density_out")
                        if pd_density is not None and isinstance(pd_density, torch.Tensor):
                            # Extract predicted points (confidence > 0.5)
                            _pred_pts = None
                            _pred_logits = outputs.get("pred_logits")
                            _pred_points = outputs.get("pred_points")
                            if _pred_logits is not None and _pred_points is not None:
                                _probs = _pred_logits[0].softmax(dim=-1)
                                _fg = _probs[:, 1] > 0.5
                                if _fg.any():
                                    _pred_pts = _pred_points[0][_fg]

                            _gt_pts = targets[0].get("point") if targets else None

                            overlay_path = Path(vis_dir) / "density_overlay_latest.png"
                            overlay_img = _save_density_overlay(
                                samples, pd_density, gt_dmap, overlay_path,
                                pred_points=_pred_pts,
                                gt_points=_gt_pts,
                            )
                            if writer is not None:
                                writer.add_image("moe/density_overlay", overlay_img, epoch)

        # --- Scale-Decoupled visualization ---
        sd_intermediates = outputs.get("sd_intermediates")
        if (
            vis_dir is not None
            and sd_intermediates is not None
            and batch_idx % vis_interval == 0
        ):
            # Feature maps from CNN, GCN, Transformer streams + CrossAttn output
            for key in ("cnn_feat", "gcn_feat", "trans_feat", "cross_attn_feat"):
                fmap = sd_intermediates.get(key)
                if fmap is not None and isinstance(fmap, torch.Tensor):
                    feat_path = Path(vis_dir) / f"sd_{key}_latest.png"
                    feat_img = _featmap_to_image(fmap, feat_path)
                    if writer is not None:
                        writer.add_image(f"scale_decoupled/{key}", feat_img, epoch)

            # DensityGCN refined features (post-CrossAttn)
            sd_dgcn = outputs.get("sd_dgcn_feat")
            if sd_dgcn is not None and isinstance(sd_dgcn, torch.Tensor):
                dgcn_path = Path(vis_dir) / "sd_dgcn_feat_latest.png"
                dgcn_img = _featmap_to_image(sd_dgcn, dgcn_path)
                if writer is not None:
                    writer.add_image("scale_decoupled/dgcn_feat", dgcn_img, epoch)

            # CrossAttn → DGCN difference map (uses shared scale, catches dead gates)
            sd_cross = sd_intermediates.get("cross_attn_feat")
            if sd_cross is not None and sd_dgcn is not None:
                diff_path = Path(vis_dir) / "sd_cross_to_dgcn_diff_latest.png"
                diff_img = _featmap_diff_image(sd_cross, sd_dgcn, diff_path)
                if writer is not None:
                    writer.add_image("scale_decoupled/cross_to_dgcn_diff", diff_img, epoch)

            # Gate scalars — the ground truth of whether learning is happening
            sd_gates = outputs.get("sd_gate_values")
            if sd_gates is not None and writer is not None:
                for gate_name, gate_val in sd_gates.items():
                    writer.add_scalar(
                        f"scale_decoupled/gates/{gate_name}", gate_val, epoch,
                    )

            # Density overlay (predicted density + GT)
            pd_density_sd = outputs.get("pred_density")
            if pd_density_sd is None:
                pd_density_sd = outputs.get("density_out")
            if pd_density_sd is not None and isinstance(pd_density_sd, torch.Tensor):
                _pred_pts_sd = None
                _pred_logits_sd = outputs.get("pred_logits")
                _pred_points_sd = outputs.get("pred_points")
                if _pred_logits_sd is not None and _pred_points_sd is not None:
                    _probs_sd = _pred_logits_sd[0].softmax(dim=-1)
                    _fg_sd = _probs_sd[:, 1] > 0.5
                    if _fg_sd.any():
                        _pred_pts_sd = _pred_points_sd[0][_fg_sd]
                _gt_pts_sd = targets[0].get("point") if targets else None

                overlay_path_sd = Path(vis_dir) / "sd_density_overlay_latest.png"
                overlay_img_sd = _save_density_overlay(
                    samples, pd_density_sd, gt_dmap, overlay_path_sd,
                    pred_points=_pred_pts_sd,
                    gt_points=_gt_pts_sd,
                )
                if writer is not None:
                    writer.add_image("scale_decoupled/density_overlay", overlay_img_sd, epoch)

        batch_idx += 1

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
    eval_point_head: bool = False,
    point_match_threshold: float = 8.0,
    point_cls_threshold: float = 0.5,
) -> tuple[float, float, float, float, dict[str, float]]:
    """Evaluate on validation set (no overlap).

    Returns:
        (mae, mse, density_mae, density_mse, pt_metrics)
        pt_metrics is a dict with optional keys: point_mae, point_precision, point_recall
    """
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    maes, mses, density_maes, density_mses = [], [], [], []
    point_maes: list[float] = []
    point_precisions: list[float] = []
    point_recalls: list[float] = []

    eval_cfg = getattr(cfg, "eval_counting", None) if cfg is not None else None
    tta_cfg = getattr(cfg, "eval_tta", None) if cfg is not None else None
    tta_enabled = (
        bool(getattr(tta_cfg, "enabled", False)) if tta_cfg is not None else False
    )

    def _forward(samples_in: torch.Tensor, depth_in: torch.Tensor | None) -> dict:
        return _forward_model(model, samples_in, depth_map=depth_in)

    for batch in data_loader:
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None
        samples = samples.to(device)
        gt_cnt = targets[0]["point"].shape[0]

        if tta_enabled:
            predict_cnt, et_dmap_sum = tta_predict(
                samples,
                depth_map,
                _forward,
                tta_cfg=tta_cfg,
                eval_cfg=eval_cfg,
            )
        else:
            outputs = _forward(samples, depth_map)
            assert outputs["pred_logits"].shape[0] == 1, (
                "evaluate_crowd_no_overlap expects batch_size=1"
            )
            predict_cnt, et_dmap_sum = count_from_outputs(outputs, eval_cfg)

        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) ** 2
        maes.append(float(mae))
        mses.append(float(mse))

        density_mae = abs(et_dmap_sum - gt_cnt)
        density_mse = (et_dmap_sum - gt_cnt) ** 2
        density_maes.append(float(density_mae))
        density_mses.append(float(density_mse))

        # Point head precision/recall via Hungarian matching
        if eval_point_head and not tta_enabled and "pred_logits" in outputs and "pred_points" in outputs:
            pt = _eval_point_head(
                outputs["pred_logits"],
                outputs["pred_points"],
                targets[0]["point"],
                match_threshold=point_match_threshold,
                cls_threshold=point_cls_threshold,
            )
            if pt is not None:
                point_maes.append(pt["mae"])
                point_precisions.append(pt["precision"])
                point_recalls.append(pt["recall"])

    mae = float(np.mean(maes))
    mse = float(np.sqrt(np.mean(mses)))
    density_mae = float(np.mean(density_maes))
    density_mse = float(np.sqrt(np.mean(density_mses)))

    pt_metrics: dict[str, float] = {}
    if point_maes:
        pt_metrics["point_mae"] = float(np.mean(point_maes))
        pt_metrics["point_precision"] = float(np.mean(point_precisions))
        pt_metrics["point_recall"] = float(np.mean(point_recalls))

    return mae, mse, density_mae, density_mse, pt_metrics


# ---------------------------------------------------------------------------
# Optimal threshold search
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_scores_and_counts(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    use_depth: bool = False,
) -> tuple[list[torch.Tensor], list[int], list[float]]:
    """Run a single forward pass over the val set and collect per-image scores.

    Returns:
        all_scores: list of 1-D CPU tensors (softmax class-1 probabilities)
        gt_counts:  list of ground-truth point counts
        density_sums: list of density-map integrals
    """
    model.eval()
    all_scores: list[torch.Tensor] = []
    gt_counts: list[int] = []
    density_sums: list[float] = []

    for batch in data_loader:
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None

        samples = samples.to(device)
        outputs = _forward_model(model, samples, depth_map=depth_map)

        scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
        assert scores.shape[0] == 1, "collect_scores_and_counts expects batch_size=1"

        all_scores.append(scores[0].cpu())
        gt_counts.append(int(targets[0]["point"].shape[0]))
        density_sums.append(float(torch.sum(outputs["density_out"]).item()))

    return all_scores, gt_counts, density_sums


def search_optimal_threshold(
    all_scores: list[torch.Tensor],
    gt_counts: list[int],
    t_min: float = 0.1,
    t_max: float = 0.95,
    t_step: float = 0.01,
) -> tuple[float, float, dict[float, float]]:
    """Sweep thresholds on cached scores and return the one with lowest MAE.

    Args:
        all_scores: Per-image score tensors from :func:`collect_scores_and_counts`.
        gt_counts:  Corresponding ground-truth counts.
        t_min:  Lower bound of search range (inclusive).
        t_max:  Upper bound of search range (inclusive).
        t_step: Step size.

    Returns:
        best_threshold: Threshold that minimises MAE.
        best_mae: MAE at the best threshold.
        results: Dict mapping each candidate threshold to its MAE.
    """
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    results: dict[float, float] = {}

    for t in thresholds:
        t_val = float(t)
        maes = [
            abs(int((s > t_val).sum().item()) - gt)
            for s, gt in zip(all_scores, gt_counts)
        ]
        results[round(t_val, 4)] = float(np.mean(maes))

    best_threshold = min(results, key=results.get)  # type: ignore[arg-type]
    best_mae = results[best_threshold]

    logger.info(
        f"Threshold search complete: best={best_threshold:.2f}, MAE={best_mae:.2f} "
        f"(searched {len(thresholds)} candidates in [{t_min}, {t_max}])"
    )
    return best_threshold, best_mae, results

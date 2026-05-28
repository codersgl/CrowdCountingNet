"""Training and evaluation loops for MoECountNet."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

from crowdcount.data.transforms import DeNormalize
from crowdcount.models.moecount.losses import MoECountLoss
from crowdcount.utils.logging import logger
from crowdcount.utils.misc import MetricLogger

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_EVAL_REFLECT_BORDER = 32  # reflect-pad input by 32px → 4px at stride-8


def _move_targets(
    targets: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def _stack_density_maps(
    density_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(list(density_maps)).to(device)


def _align_density_to_target(pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
    if pred_density.shape[-2:] == gt_density.shape[-2:]:
        return pred_density
    target_height, target_width = gt_density.shape[-2:]
    pred_height, pred_width = pred_density.shape[-2:]
    if pred_height >= target_height and pred_width >= target_width:
        return pred_density[:, :, :target_height, :target_width]
    area_ratio = float(pred_height * pred_width) / float(target_height * target_width)
    return F.interpolate(
        pred_density,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ) * area_ratio


def _target_density_size(target: dict[str, torch.Tensor], output_stride: int) -> tuple[int, int] | None:
    orig_size = target.get("orig_size")
    if orig_size is None:
        return None
    image_height = int(orig_size[0].item())
    image_width = int(orig_size[1].item())
    return (
        max(1, math.ceil(image_height / float(output_stride))),
        max(1, math.ceil(image_width / float(output_stride))),
    )


def _crop_density_for_eval(
    pred_density: torch.Tensor,
    target: dict[str, torch.Tensor],
    output_stride: int,
) -> torch.Tensor:
    density_size = _target_density_size(target, output_stride)
    if density_size is None:
        return pred_density
    density_height, density_width = density_size
    return pred_density[:, :, :density_height, :density_width]


def _save_expert_route_image(top1: torch.Tensor, output_path: Path) -> torch.Tensor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = top1.new_tensor(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=torch.uint8,
    )
    top1_cpu = top1[0].detach().to(torch.long).cpu().clamp(0, 2)
    rgb = palette.cpu()[top1_cpu]
    Image.fromarray(rgb.numpy()).save(output_path)
    return rgb.permute(2, 0, 1).float() / 255.0


def _density_heatmap(density: torch.Tensor) -> torch.Tensor:
    """Convert a 2D density map to a JET RGB heatmap [3, H, W] in [0, 1].

    Uses matplotlib's JET colormap with 99.5th-percentile normalisation to
    avoid a single outlier pixel washing out the whole map.
    """
    import matplotlib

    d = density.detach().cpu().float()
    vmax = torch.quantile(d, 0.995).clamp_min(1e-8)
    d = (d / vmax).clamp(0, 1)
    jet = matplotlib.colormaps["jet"]
    rgb = jet(d.numpy())[..., :3].copy()  # [H, W, 3] in [0, 1]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _save_density_overlay(
    sample: torch.Tensor,
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    output_path: Path,
) -> torch.Tensor:
    """Save original image with predicted and GT density overlaid side-by-side.

    Returns a [3, H, 3*W] tensor for TensorBoard logging.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    denorm = DeNormalize(_IMAGENET_MEAN, _IMAGENET_STD)
    img = denorm(sample[0].detach().cpu().clone())  # [3, H_img, W_img]
    img = img.clamp(0, 1)

    pred = pred_density[0].detach().cpu()  # [1, H8, W8]
    gt = gt_density[0].detach().cpu()      # [1, H_img, W_img]

    pred_count = float(pred.sum().item())
    gt_count = float(gt.sum().item())

    # Upsample prediction and GT density to image resolution
    pred_full = F.interpolate(
        pred.unsqueeze(0),
        size=img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [1, H_img, W_img]
    gt_full = F.interpolate(
        gt.unsqueeze(0).float(),
        size=img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [1, H_img, W_img]

    def _overlay(image: torch.Tensor, density: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
        hm = _density_heatmap(density.squeeze(0))  # [3, H, W]
        return image * (1 - alpha) + hm * alpha

    overlay_pred = _overlay(img, pred_full)
    overlay_gt = _overlay(img, gt_full)

    # Concatenate: [Original | Pred Overlay | GT Overlay]
    combined = torch.cat([img, overlay_pred, overlay_gt], dim=-1)  # [3, H, 3*W]
    combined_uint8 = (combined.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()

    # Annotate counts on the overlay panels
    pil_img = Image.fromarray(combined_uint8)
    draw = ImageDraw.Draw(pil_img)
    w = img.shape[-1]
    draw.text((w + 8, 8), f"Pred: {pred_count:.1f}", fill=(255, 255, 0))
    draw.text((2 * w + 8, 8), f"GT:   {gt_count:.1f}", fill=(255, 255, 0))
    pil_img.save(output_path)

    return combined


def train_moecount_one_epoch(
    model: nn.Module,
    loss_fn: MoECountLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_norm: float = 0.0,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    writer=None,
    global_step: int = 0,
    log_interval: int = 100,
    vis_interval: int = 500,
    vis_dir: str | Path | None = None,
    output_stride: int = 8,
) -> tuple[dict[str, float], int]:
    model.train()
    set_epoch = getattr(model, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch, total_epochs)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_logger.meters["lr"])
    header = f"Epoch: [{epoch}]"
    amp_enabled = bool(use_amp and device.type == "cuda" and scaler is not None)
    route_image_path = Path(vis_dir) / "expert_route_top1.png" if vis_dir is not None else None

    for batch in metric_logger.log_every(data_loader, max(1, log_interval), header):
        samples, targets, gt_density_maps = batch[:3]
        samples = samples.to(device)
        gt_density = _stack_density_maps(gt_density_maps, device)
        targets = _move_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(samples)
            pred_density = outputs["density_out"]
            if not isinstance(pred_density, torch.Tensor):
                raise TypeError("model output density_out must be a tensor")
            aligned_pred = _align_density_to_target(pred_density, gt_density)
            outputs = dict(outputs)
            outputs["density_out"] = aligned_pred
            outputs["pred_density"] = aligned_pred

            # DeepSeek-V2 style expert bias update (post-warmup load balancing).
            # Only applies to SparseTop2Gate; PixelSoftGate has no concept of
            # hard load balancing or expert bias.
            if hasattr(model.moe.gate, "update_expert_bias"):
                load_frac = outputs.get("moe_load_fraction")
                warmup_flag = outputs.get("moe_warmup_active")
                if (
                    isinstance(load_frac, torch.Tensor)
                    and not (isinstance(warmup_flag, bool) and warmup_flag)
                ):
                    model.moe.gate.update_expert_bias(load_frac)

            image_sizes = (
                int(gt_density.shape[-2] * output_stride),
                int(gt_density.shape[-1] * output_stride),
            )
            loss_dict = loss_fn(outputs, targets, gt_density, image_sizes, epoch)
            loss_total = loss_dict["loss_total"]

        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"MoECount loss is not finite: {float(loss_total.item())}")

        if amp_enabled:
            assert scaler is not None
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_total.backward()

        # Pre-clip gradient norm for monitoring
        total_norm: float = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if amp_enabled:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        density_loss_key = "loss_pml" if "loss_pml" in loss_dict else "loss_bayesian"
        metrics: dict[str, float] = {
            "loss_total": float(loss_dict["loss_total"].detach().item()),
            density_loss_key: float(loss_dict[density_loss_key].detach().item()),
            "loss_count": float(loss_dict["loss_count"].detach().item()),
            "loss_balance": float(loss_dict["loss_balance"].detach().item()),
            "lambda_count": float(loss_dict["lambda_count"].detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
        }
        for loss_key in ("loss_point_cls", "loss_point_reg", "loss_ot"):
            value = loss_dict.get(loss_key)
            if isinstance(value, torch.Tensor):
                metrics[loss_key] = float(value.detach().item())
        for output_key, metric_key in (
            ("moe_entropy", "moe_entropy"),
            ("moe_temperature", "moe_temperature"),
        ):
            value = outputs.get(output_key)
            if isinstance(value, torch.Tensor):
                metrics[metric_key] = float(value.detach().item())
        load_fraction = outputs.get("moe_load_fraction")
        if isinstance(load_fraction, torch.Tensor):
            load_fraction_cpu = load_fraction.detach().cpu()
            for expert_index, value in enumerate(load_fraction_cpu.tolist(), start=1):
                metrics[f"moe_e{expert_index}_load"] = float(value)
        expert_similarity = outputs.get("expert_similarity", {})
        if isinstance(expert_similarity, dict):
            for sim_key, sim_value in expert_similarity.items():
                if isinstance(sim_value, torch.Tensor):
                    metrics[f"expert_{sim_key}"] = float(sim_value.detach().item())
        metric_logger.update(**metrics)

        if writer is not None:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f"train/{metric_name}", metric_value, global_step)
        if (
            route_image_path is not None
            and vis_interval > 0
            and global_step % vis_interval == 0
            and isinstance(outputs.get("moe_top1"), torch.Tensor)
        ):
            image_tensor = _save_expert_route_image(outputs["moe_top1"], route_image_path)
            if writer is not None:
                writer.add_image("moe/top1_expert", image_tensor, global_step)

        if (
            vis_dir is not None
            and vis_interval > 0
            and global_step % vis_interval == 0
            and isinstance(outputs.get("pred_density"), torch.Tensor)
        ):
            density_overlay_path = Path(vis_dir) / "density_overlay.png"
            overlay_tensor = _save_density_overlay(
                samples, outputs["pred_density"], gt_density,
                density_overlay_path,
            )
            if writer is not None:
                writer.add_image("moe/density_overlay", overlay_tensor, global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    stats = {name: meter.global_avg for name, meter in metric_logger.meters.items()}
    return stats, global_step


@torch.no_grad()
def evaluate_moecount(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    output_stride: int = 8,
    eval_point_head: bool = True,
    point_match_threshold: float = 8.0,
    point_cls_threshold: float = 0.3,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    maes: list[float] = []
    mses: list[float] = []
    point_maes: list[float] = []
    point_precisions: list[float] = []
    point_recalls: list[float] = []
    border = _EVAL_REFLECT_BORDER
    crop = border // output_stride  # px at stride-8 density resolution

    for batch in data_loader:
        samples, targets = batch[:2]
        samples = samples.to(device)
        # Reflect-pad input to avoid zero-padding edge artifacts at inference.
        samples_padded = F.pad(samples, (border, border, border, border), mode="reflect")
        outputs = model(samples_padded)
        pred_density = outputs["density_out"]
        if not isinstance(pred_density, torch.Tensor):
            raise TypeError("model output density_out must be a tensor")
        pred_density = pred_density[:, :, crop:-crop or None, crop:-crop or None]
        pred_density = _crop_density_for_eval(pred_density, targets[0], output_stride)
        pred_count = float(pred_density.sum().item())
        gt_count = float(targets[0]["point"].shape[0])
        error = pred_count - gt_count
        maes.append(abs(error))
        mses.append(error * error)

        # Point head eval
        if eval_point_head and "pred_logits" in outputs and "pred_points" in outputs:
            pt_metrics = _eval_point_head(
                outputs["pred_logits"],
                outputs["pred_points"],
                targets[0]["point"],
                match_threshold=point_match_threshold,
                cls_threshold=point_cls_threshold,
            )
            if pt_metrics is not None:
                point_maes.append(pt_metrics["mae"])
                point_precisions.append(pt_metrics["precision"])
                point_recalls.append(pt_metrics["recall"])

    mae = float(np.mean(maes)) if maes else 0.0
    mse = float(np.sqrt(np.mean(mses))) if mses else 0.0
    pt_metrics: dict[str, float] = {}
    if point_maes:
        pt_metrics = {
            "point_mae": float(np.mean(point_maes)),
            "point_precision": float(np.mean(point_precisions)),
            "point_recall": float(np.mean(point_recalls)),
        }
        logger.info(
            f"[MoECount Eval] mae={mae:.2f} mse={mse:.2f} | "
            f"point_mae={pt_metrics['point_mae']:.2f} "
            f"point_prec={pt_metrics['point_precision']:.3f} "
            f"point_recall={pt_metrics['point_recall']:.3f}"
        )
    else:
        logger.info(f"[MoECount Eval] mae={mae:.2f} mse={mse:.2f}")
    return mae, mse, pt_metrics


def _eval_point_head(
    pred_logits: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    match_threshold: float = 8.0,
    cls_threshold: float = 0.3,
) -> dict[str, float] | None:
    """Compute point head metrics via Hungarian matching."""
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

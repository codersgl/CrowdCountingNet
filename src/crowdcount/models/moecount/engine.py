"""Training and evaluation loops for MoECountNet."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from crowdcount.models.moecount.losses import MoECountLoss
from crowdcount.utils.logging import logger
from crowdcount.utils.misc import MetricLogger


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
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metrics: dict[str, float] = {
            "loss_total": float(loss_dict["loss_total"].detach().item()),
            "loss_bayesian": float(loss_dict["loss_bayesian"].detach().item()),
            "loss_count": float(loss_dict["loss_count"].detach().item()),
            "loss_balance": float(loss_dict["loss_balance"].detach().item()),
            "lambda_count": float(loss_dict["lambda_count"].detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
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
) -> tuple[float, float]:
    model.eval()
    maes: list[float] = []
    mses: list[float] = []

    for batch in data_loader:
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pred_density = outputs["density_out"]
        if not isinstance(pred_density, torch.Tensor):
            raise TypeError("model output density_out must be a tensor")
        pred_density = _crop_density_for_eval(pred_density, targets[0], output_stride)
        pred_count = float(pred_density.sum().item())
        gt_count = float(targets[0]["point"].shape[0])
        error = pred_count - gt_count
        maes.append(abs(error))
        mses.append(error * error)

    mae = float(np.mean(maes)) if maes else 0.0
    mse = float(np.sqrt(np.mean(mses))) if mses else 0.0
    logger.info(f"[MoECount Eval] mae={mae:.2f} mse={mse:.2f}")
    return mae, mse

"""Trainer for MoECountNet."""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crowdcount.data import build_dataset
from crowdcount.data.collate import make_train_collate
from crowdcount.models.moecount import build_moecount
from crowdcount.models.moecount.engine import evaluate_moecount, train_moecount_one_epoch
from crowdcount.models.moecount.losses import (
    BayesianLoss,
    CountLoss,
    GradientAwareLoss,
    LoadBalanceLoss,
    MoECountLoss,
    PatchSSIMLoss,
    ProximalMappingLoss,
    SinkhornOTLoss,
)
from crowdcount.utils.logging import logger, setup_logger
from crowdcount.utils.misc import get_rank, nested_tensor_from_tensor_list


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn_moecount_eval(batch: list[tuple[torch.Tensor, list[dict[str, torch.Tensor]]]]):
    """Eval collate that keeps original image size before padding."""
    flattened: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for images, targets in batch:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        for image_index in range(len(images)):
            target = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in targets[image_index].items()
            }
            target["orig_size"] = torch.tensor(
                [images[image_index].shape[-2], images[image_index].shape[-1]],
                dtype=torch.long,
            )
            flattened.append((images[image_index], target))
    images_tuple, targets_tuple = list(zip(*flattened))
    return nested_tensor_from_tensor_list(list(images_tuple)), targets_tuple


class MoECountTrainer:
    """Standalone pure-density trainer for MoECountNet."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = int(cfg.seed) + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(seed)

        try:
            hydra_output = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            hydra_output = Path(".")
        self.output_dir = hydra_output
        self.checkpoints_dir = hydra_output / cfg.checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = hydra_output / str(getattr(getattr(cfg, "monitor", None), "vis_dir", "visuals/moecount"))
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        setup_logger(log_dir=str(hydra_output), log_file="train.log")

        self.model = build_moecount(cfg).to(self.device)
        self.loss_fn = self._build_loss().to(self.device)
        n_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        logger.info(f"MoECount trainable parameters: {n_params:,}")

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()

        train_set, val_set = build_dataset(cfg)
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train,
            int(cfg.data.batch_size),
            drop_last=True,
        )
        train_collate = make_train_collate(getattr(cfg.data, "augmentation", None), use_depth=False)
        self.data_loader_train = DataLoader(
            train_set,
            batch_sampler=batch_sampler_train,
            collate_fn=train_collate,
            num_workers=int(cfg.num_workers),
            worker_init_fn=_seed_worker,
            generator=self._loader_generator,
        )
        self.data_loader_val = DataLoader(
            val_set,
            batch_size=1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=collate_fn_moecount_eval,
            num_workers=int(cfg.num_workers),
            worker_init_fn=_seed_worker,
            generator=self._loader_generator,
        )

        self.amp_enabled = bool(getattr(getattr(cfg, "mixed_precision", None), "enabled", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self._resume_mae_history: list[float] = []
        self._resume_mse_history: list[float] = []
        self.best_mae = float("inf")
        self._load_initial_state()

        tb_dir = hydra_output / cfg.tensorboard_dir
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    def _build_loss(self) -> MoECountLoss:
        loss_cfg = getattr(self.cfg, "moecount_loss", None)
        if loss_cfg is None:
            return MoECountLoss()

        use_pml = bool(getattr(loss_cfg, "use_pml", True))
        pml_cfg = getattr(loss_cfg, "pml", None)
        bayes_cfg = getattr(loss_cfg, "bayesian", None)
        count_cfg = getattr(loss_cfg, "count", None)
        balance_cfg = getattr(loss_cfg, "balance", None)

        pml_loss = None
        bayesian_loss = None
        if use_pml:
            sigma_schedule = getattr(pml_cfg, "sigma_schedule", None)
            if sigma_schedule is not None and not isinstance(sigma_schedule, dict):
                sigma_schedule = dict(sigma_schedule)
            pml_loss = ProximalMappingLoss(
                sigma=float(getattr(pml_cfg, "sigma", 8.0)),
                use_background=bool(getattr(pml_cfg, "use_background", False)),
                bg_threshold=float(getattr(pml_cfg, "bg_threshold", 3.0)),
                max_pixels_per_chunk=int(getattr(pml_cfg, "max_pixels_per_chunk", 16384)),
                sigma_schedule=sigma_schedule,
            )
        else:
            bayesian_loss = BayesianLoss(
                sigma=float(getattr(bayes_cfg, "sigma", 8.0)),
                use_background=bool(getattr(bayes_cfg, "use_background", True)),
                bg_ratio=float(getattr(bayes_cfg, "bg_ratio", 0.15)),
                count_loss_type=str(getattr(bayes_cfg, "count_loss_type", "l1")),
                max_pixels_per_chunk=int(getattr(bayes_cfg, "max_pixels_per_chunk", 16384)),
            )

        count_weight = float(getattr(count_cfg, "weight", 1.0))

        # Point loss config
        point_cfg = getattr(loss_cfg, "point", None)
        point_loss_weight = float(getattr(point_cfg, "weight", 0.0)) if point_cfg is not None else 0.0
        point_cost_class = float(getattr(point_cfg, "cost_class", 1.0)) if point_cfg is not None else 1.0
        point_cost_l1 = float(getattr(point_cfg, "cost_l1", 1.0)) if point_cfg is not None else 1.0
        point_focal_alpha = float(getattr(point_cfg, "focal_alpha", 0.75)) if point_cfg is not None else 0.75
        point_focal_gamma = float(getattr(point_cfg, "focal_gamma", 2.0)) if point_cfg is not None else 2.0
        point_eos_coef = float(getattr(point_cfg, "eos_coef", 0.1)) if point_cfg is not None else 0.1

        # OT loss config
        ot_cfg = getattr(loss_cfg, "ot", None)
        if ot_cfg is not None and bool(getattr(ot_cfg, "enabled", False)):
            ot_loss = SinkhornOTLoss(
                epsilon=float(getattr(ot_cfg, "epsilon", 0.1)),
                num_iters=int(getattr(ot_cfg, "num_iters", 50)),
                max_grid=int(getattr(ot_cfg, "max_grid", 32)),
                weight=1.0,  # weight applied in MoECountLoss
            )
            ot_weight = float(getattr(ot_cfg, "weight", 0.05))
        else:
            ot_loss = None
            ot_weight = 0.0

        # Compute warmup end for balance loss decay
        moe_cfg = getattr(self.cfg.model, "moe", None)
        warmup_epochs = getattr(moe_cfg, "warmup_epochs", None) if moe_cfg is not None else None
        if warmup_epochs is not None:
            warmup_end = int(warmup_epochs)
        else:
            warmup_fraction = float(getattr(moe_cfg, "warmup_fraction", 0.2)) if moe_cfg is not None else 0.2
            warmup_end = int(math.ceil(float(self.cfg.epochs) * warmup_fraction))

        diversity_weight = float(getattr(loss_cfg, "diversity_weight", 0.05))
        expert_sup_weight = float(getattr(loss_cfg, "expert_supervision_weight", 0.2))
        density_s4_weight = float(getattr(loss_cfg, "density_s4_weight", 0.3))

        # SSIM local structure loss
        ssim_weight = float(getattr(loss_cfg, "ssim_weight", 0.05))
        if ssim_weight > 0:
            ssim_loss = PatchSSIMLoss(kernel_size=5, sigma=2.0, weight=1.0)
        else:
            ssim_loss = None

        # Gradient-aware loss
        grad_weight = float(getattr(loss_cfg, "grad_weight", 0.01))
        if grad_weight > 0:
            grad_loss = GradientAwareLoss(weight=1.0)
        else:
            grad_loss = None

        return MoECountLoss(
            pml_loss=pml_loss,
            bayesian_loss=bayesian_loss,
            count_loss=CountLoss(),
            count_weight=count_weight,
            balance_loss=LoadBalanceLoss(
                lambda_importance=float(getattr(balance_cfg, "lambda_importance", 0.01)),
                lambda_load=float(getattr(balance_cfg, "lambda_load", 0.01)),
            ),
            warmup_end=warmup_end,
            balance_decay_epochs=int(getattr(balance_cfg, "decay_epochs", 50)),
            point_loss_weight=point_loss_weight,
            point_cost_class=point_cost_class,
            point_cost_l1=point_cost_l1,
            point_focal_alpha=point_focal_alpha,
            point_focal_gamma=point_focal_gamma,
            point_eos_coef=point_eos_coef,
            ot_loss=ot_loss,
            ot_weight=ot_weight,
            ssim_loss=ssim_loss,
            ssim_weight=ssim_weight,
            grad_loss=grad_loss,
            grad_weight=grad_weight,
            diversity_weight=diversity_weight,
            expert_supervision_weight=expert_sup_weight,
            density_s4_weight=density_s4_weight,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        gate_params = [parameter for parameter in self.model.moe.gate.parameters() if parameter.requires_grad]
        backbone_params = [parameter for parameter in self.model.backbone.parameters() if parameter.requires_grad]
        gate_param_ids = {id(parameter) for parameter in gate_params}
        backbone_param_ids = {id(parameter) for parameter in backbone_params}
        other_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in gate_param_ids
            and id(parameter) not in backbone_param_ids
        ]
        param_groups: list[dict[str, Any]] = []
        if other_params:
            param_groups.append({"params": other_params, "name": "head"})
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": float(self.cfg.optimizer.lr_backbone),
                    "name": "backbone",
                }
            )
        if gate_params:
            param_groups.append(
                {
                    "params": gate_params,
                    "lr": float(getattr(self.cfg.optimizer, "lr_gate", self.cfg.optimizer.lr)),
                    "name": "gate",
                }
            )

        optimizer_name = str(self.cfg.optimizer.get("name", "adamw")).lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=float(self.cfg.optimizer.lr),
                weight_decay=float(self.cfg.optimizer.weight_decay),
                amsgrad=bool(self.cfg.optimizer.get("amsgrad", False)),
            )
        else:
            optimizer = torch.optim.Adam(
                param_groups,
                lr=float(self.cfg.optimizer.lr),
                weight_decay=float(self.cfg.optimizer.weight_decay),
            )
        logger.info(f"Optimizer: {type(optimizer).__name__}")
        for group in optimizer.param_groups:
            logger.info(f"Param group {group.get('name', 'unnamed')}: lr={group['lr']}")
        return optimizer

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        sched = self.cfg.scheduler
        name = str(sched.get("name", "cosine_annealing"))
        warmup_epochs = int(sched.get("warmup_epochs", 0))
        warmup_start_factor = float(sched.get("warmup_start_factor", 0.001))
        if name == "cosine_annealing":
            t_max_effective = max(int(sched.T_max) - warmup_epochs, 1)
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max_effective,
                eta_min=float(sched.eta_min),
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sched.lr_drop),
            )
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            logger.info(f"Scheduler: {name} with linear warmup ({warmup_epochs} epochs)")
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs],
            )
        logger.info(f"Scheduler: {name} (no warmup)")
        return main_sched

    def _load_initial_state(self) -> None:
        cfg = self.cfg
        if cfg.frozen_weights is not None:
            checkpoint = torch.load(cfg.frozen_weights, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            logger.info(f"Loaded MoECount weights from {cfg.frozen_weights}")
        if not cfg.resume:
            return
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        reset_optimizer = bool(getattr(cfg, "reset_optimizer", False))
        if not reset_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not reset_optimizer and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if not reset_optimizer and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if "epoch" in checkpoint and not reset_optimizer:
            cfg.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_mae = float(checkpoint.get("best_mae", self.best_mae))
        self._resume_mae_history = list(checkpoint.get("mae_history", []))
        self._resume_mse_history = list(checkpoint.get("mse_history", []))
        moe_temperature = checkpoint.get("moe_temperature")
        if moe_temperature is not None:
            self.model.moe.gate.temperature = float(moe_temperature)
        logger.info(f"Resumed MoECount from {cfg.resume} (epoch {cfg.start_epoch})")

    def _checkpoint_payload(
        self,
        epoch: int,
        mae_history: list[float],
        mse_history: list[float],
    ) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch,
            "best_mae": self.best_mae,
            "mae_history": mae_history,
            "mse_history": mse_history,
            "moe_temperature": self.model.moe.temperature,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

    def train(self) -> None:
        cfg = self.cfg
        logger.info("Start MoECount training")
        start_time = time.time()
        mae_history = list(self._resume_mae_history)
        mse_history = list(self._resume_mse_history)
        global_step = 0
        monitor_cfg = getattr(cfg, "monitor", None)
        log_interval = int(getattr(monitor_cfg, "log_interval", 100))
        vis_interval = int(getattr(monitor_cfg, "vis_interval", 500))
        output_stride = int(getattr(cfg.model, "output_stride", 8))

        for epoch in range(int(cfg.start_epoch), int(cfg.epochs)):
            epoch_start = time.time()
            train_stats, global_step = train_moecount_one_epoch(
                self.model,
                self.loss_fn,
                self.data_loader_train,
                self.optimizer,
                self.device,
                epoch,
                int(cfg.epochs),
                max_norm=float(cfg.clip_max_norm),
                scaler=self.scaler,
                use_amp=self.amp_enabled,
                writer=self.writer,
                global_step=global_step,
                log_interval=log_interval,
                vis_interval=vis_interval,
                vis_dir=self.vis_dir,
                output_stride=output_stride,
            )
            logger.info(
                f"[MoECount ep {epoch}][lr {self.optimizer.param_groups[0]['lr']:.7f}]"
                f"[{time.time() - epoch_start:.2f}s]"
            )
            for stat_name, stat_value in train_stats.items():
                self.writer.add_scalar(f"epoch/{stat_name}", stat_value, epoch)
            for group in self.optimizer.param_groups:
                group_name = str(group.get("name", "group"))
                self.writer.add_scalar(f"lr/{group_name}", float(group["lr"]), epoch)

            do_eval = (epoch + 1) % int(cfg.eval_freq) == 0
            if do_eval:
                mae, mse = evaluate_moecount(
                    self.model,
                    self.data_loader_val,
                    self.device,
                    output_stride=output_stride,
                )
                mae_history.append(mae)
                mse_history.append(mse)
                self.writer.add_scalar("metric/mae", mae, epoch)
                self.writer.add_scalar("metric/mse", mse, epoch)
                if mae <= self.best_mae:
                    self.best_mae = mae
                    torch.save(
                        self._checkpoint_payload(epoch, mae_history, mse_history),
                        self.checkpoints_dir / "best_mae.pth",
                    )
                    logger.info(f"Saved new MoECount best checkpoint: MAE={mae:.2f}")

            torch.save(
                self._checkpoint_payload(epoch, mae_history, mse_history),
                self.checkpoints_dir / "latest.pth",
            )
            self.lr_scheduler.step()

        self.writer.close()
        total_time = time.time() - start_time
        logger.info(f"MoECount training finished in {total_time / 60:.1f} min")

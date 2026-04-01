"""Trainer: encapsulates the full training loop for DSGCNet.

Adapted from train.py main() — all logic unchanged.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crowdcount.data import build_dataset, collate_fn_crowd, collate_fn_crowd_train
from crowdcount.data.collate import collate_fn_crowd_depth, collate_fn_crowd_train_depth
from crowdcount.engine import evaluate_crowd_no_overlap, train_one_epoch
from crowdcount.models import build_model
from crowdcount.models.ssim_loss import SSIMLoss
from crowdcount.utils.logging import logger, setup_logger
from crowdcount.utils.misc import get_rank


class Trainer:
    """Encapsulates the complete DSGCNet training pipeline."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducibility
        seed = cfg.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Directories — use Hydra output dir so relative paths aren't broken
        # by the project root working directory.
        try:
            hydra_output = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            hydra_output = Path(".")
        self.checkpoints_dir = hydra_output / cfg.checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        setup_logger(log_dir=str(hydra_output), log_file="train.log")

        # Model
        model, criterion = cast(
            tuple[nn.Module, nn.Module], build_model(cfg, training=True)
        )
        model.to(self.device)
        criterion.to(self.device)
        self.model = model
        self.criterion = criterion
        self.density_criterion = nn.MSELoss(reduction="sum").to(self.device)
        density_ssim_cfg = getattr(cfg, "density_ssim", None)
        if bool(getattr(density_ssim_cfg, "enabled", False)):
            self.ssim_criterion: nn.Module | None = SSIMLoss(
                window_size=int(getattr(density_ssim_cfg, "window_size", 11)),
                sigma=float(getattr(density_ssim_cfg, "sigma", 1.5)),
            ).to(self.device)
        else:
            self.ssim_criterion = None

        self.use_moe = bool(getattr(self.model, "supports_moe", lambda: False)())
        self.use_depth = bool(getattr(cfg.model, "use_depth", False))
        self.use_depth_geo = bool(getattr(cfg.model, "use_depth_geo", False))
        self._needs_depth = self.use_depth or self.use_depth_geo

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {n_params:,}")

        # Optimizer
        non_backbone_params = [
            p
            for n, p in model.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]
        backbone_params = [
            p
            for n, p in model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]

        param_dicts = [
            {"params": non_backbone_params},
            {"params": backbone_params, "lr": cfg.optimizer.lr_backbone},
        ]
        _opt_name = cfg.optimizer.get("name", "adam").lower()
        if _opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_dicts,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                amsgrad=cfg.optimizer.get("amsgrad", False),
            )
        else:
            self.optimizer = torch.optim.Adam(
                param_dicts,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        self.lr_scheduler = self._build_scheduler()

        # Data
        train_set, val_set = build_dataset(cfg)
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.data.batch_size, drop_last=True
        )
        self.data_loader_train = DataLoader(
            train_set,
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn_crowd_train_depth
            if self._needs_depth
            else collate_fn_crowd_train,
            num_workers=cfg.num_workers,
        )
        self.data_loader_val = DataLoader(
            val_set,
            batch_size=1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=collate_fn_crowd_depth
            if self._needs_depth
            else collate_fn_crowd,
            num_workers=cfg.num_workers,
        )

        # Optional: resume from checkpoint
        if cfg.frozen_weights is not None:
            ckpt = torch.load(cfg.frozen_weights, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        if cfg.resume:
            ckpt = torch.load(cfg.resume, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            reset_opt = bool(getattr(cfg, "reset_optimizer", False))
            if not reset_opt and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if not reset_opt and "lr_scheduler" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if "moe_temperature" in ckpt and ckpt["moe_temperature"] is not None:
                moe_module = getattr(model, "moe", None)
                if moe_module is not None:
                    moe_module.temperature = ckpt["moe_temperature"]
                    moe_module.router.temperature = ckpt["moe_temperature"]
            if "epoch" in ckpt and not reset_opt:
                cfg.start_epoch = ckpt["epoch"] + 1
            if "mae_history" in ckpt:
                self._resume_mae_history = ckpt["mae_history"]
            if "density_mae_history" in ckpt:
                self._resume_density_mae_history = ckpt["density_mae_history"]
            if reset_opt:
                logger.info(
                    f"Resumed model weights from {cfg.resume} (optimizer reset, training from epoch 0)"
                )
            else:
                logger.info(f"Resumed from {cfg.resume} (epoch {cfg.start_epoch})")

        # TensorBoard
        tb_dir = hydra_output / cfg.tensorboard_dir
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))

        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        sched = self.cfg.scheduler
        name = sched.get("name", "step_lr")
        warmup_epochs = int(sched.get("warmup_epochs", 0))
        warmup_start_factor = float(sched.get("warmup_start_factor", 0.001))

        if name == "cosine_annealing":
            # Subtract warmup from the cosine period so the full T_max budget is
            # correctly spent on the cosine phase after warmup completes.
            t_max_effective = max(int(sched.T_max) - warmup_epochs, 1)
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max_effective,
                eta_min=sched.eta_min,
            )
        else:
            # default: step_lr
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched.lr_drop
            )

        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            logger.info(
                f"Scheduler: {name} with linear warmup "
                f"({warmup_epochs} epochs, start_factor={warmup_start_factor})"
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs],
            )

        logger.info(f"Scheduler: {name} (no warmup)")
        return main_sched

    def train(self) -> None:
        cfg = self.cfg
        logger.info("Start training")
        start_time = time.time()

        mae_history = list(getattr(self, "_resume_mae_history", []))
        density_mae_history = list(getattr(self, "_resume_density_mae_history", []))
        mse_history, density_mse_history = [], []
        step = 0

        for epoch in range(cfg.start_epoch, cfg.epochs):
            moe_module = getattr(self.model, "moe", None)
            if self.use_moe and moe_module is not None:
                moe_module.update_noise_scale(epoch / cfg.epochs)

            t1 = time.time()
            stat = train_one_epoch(
                self.model,
                self.criterion,
                self.data_loader_train,
                self.optimizer,
                self.density_criterion,
                self.device,
                epoch,
                cfg.clip_max_norm,
                cfg=cfg,  # Pass config for multi-scale density prediction
                ssim_criterion=self.ssim_criterion,
            )
            t2 = time.time()

            logger.info(
                f"[ep {epoch}][lr {self.optimizer.param_groups[0]['lr']:.7f}][{t2 - t1:.2f}s]"
            )

            # TensorBoard
            for key in ("loss_sum", "losses", "den_loss", "loss_ce"):
                if key in stat:
                    self.writer.add_scalar(f"loss/{key}", stat[key], epoch)
            for i, pg in enumerate(self.optimizer.param_groups):
                tag = "lr/backbone" if i == 1 else "lr/base"
                self.writer.add_scalar(tag, pg["lr"], epoch)

            self.lr_scheduler.step()

            # Save latest checkpoint
            ckpt_path = self.checkpoints_dir / "latest.pth"
            moe_temperature = (
                getattr(moe_module, "temperature", None)
                if moe_module is not None
                else None
            )
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "mae_history": mae_history,
                    "density_mae_history": density_mae_history,
                    "moe_temperature": moe_temperature,
                },
                ckpt_path,
            )

            # Evaluation
            if epoch % cfg.eval_freq == 0 and epoch != 0:
                t1 = time.time()
                result = evaluate_crowd_no_overlap(
                    self.model,
                    self.data_loader_val,
                    self.device,
                    use_depth=self._needs_depth,
                )
                t2 = time.time()

                mae_history.append(result[0])
                mse_history.append(result[1])
                density_mae_history.append(result[2])
                density_mse_history.append(result[3])

                logger.info(
                    f"[Eval] mae={result[0]:.2f}  mse={result[1]:.2f}  "
                    f"time={t2 - t1:.1f}s  best_mae={np.min(mae_history):.2f}  best_mse={np.min(mse_history):.2f}"
                )
                logger.info(
                    f"[Eval] density_mae={result[2]:.2f}  density_mse={result[3]:.2f}  "
                    f"best_density_mae={np.min(density_mae_history):.2f}"
                )

                self.writer.add_scalar("metric/mae", result[0], step)
                self.writer.add_scalar("metric/mse", result[1], step)
                self.writer.add_scalar("metric/density_mae", result[2], step)
                self.writer.add_scalar("metric/density_mse", result[3], step)
                step += 1

                # Save best MAE checkpoint (strict: only when current epoch is the new minimum)
                if result[0] <= np.min(mae_history):
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "mae_history": mae_history,
                            "density_mae_history": density_mae_history,
                            "moe_temperature": moe_temperature,
                        },
                        self.checkpoints_dir / "best_mae.pth",
                    )

        self.writer.close()
        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time / 60:.1f} min")

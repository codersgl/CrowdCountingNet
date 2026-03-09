"""Trainer: encapsulates the full training loop for DSGCNet.

Adapted from train.py main() — all logic unchanged.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crowdcount.data import build_dataset, collate_fn_crowd, collate_fn_crowd_train
from crowdcount.engine import evaluate_crowd_no_overlap, train_one_epoch
from crowdcount.models import build_model
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
        model, criterion = build_model(cfg, training=True)
        model.to(self.device)
        criterion.to(self.device)
        self.model = model
        self.criterion = criterion
        self.density_criterion = nn.MSELoss(reduction="sum").to(self.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {n_params:,}")

        # Optimizer
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": cfg.optimizer.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.Adam(
            param_dicts, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.scheduler.lr_drop
        )

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
            collate_fn=collate_fn_crowd_train,
            num_workers=cfg.num_workers,
        )
        self.data_loader_val = DataLoader(
            val_set,
            batch_size=1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=collate_fn_crowd,
            num_workers=cfg.num_workers,
        )

        # Optional: resume from checkpoint
        if cfg.frozen_weights is not None:
            ckpt = torch.load(cfg.frozen_weights, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        if cfg.resume:
            ckpt = torch.load(cfg.resume, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            logger.info(f"Resumed from {cfg.resume}")

        # TensorBoard
        tb_dir = hydra_output / cfg.tensorboard_dir
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))

        logger.info(f"Config:\n{cfg}")

    def train(self) -> None:
        cfg = self.cfg
        logger.info("Start training")
        start_time = time.time()

        mae_history, mse_history = [], []
        density_mae_history, density_mse_history = [], []
        step = 0

        for epoch in range(cfg.start_epoch, cfg.epochs):
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
            )
            t2 = time.time()

            logger.info(
                f"[ep {epoch}][lr {self.optimizer.param_groups[0]['lr']:.7f}][{t2 - t1:.2f}s]"
            )

            # TensorBoard
            for key in ("loss_sum", "losses", "den_loss", "loss_ce"):
                if key in stat:
                    self.writer.add_scalar(f"loss/{key}", stat[key], epoch)

            self.lr_scheduler.step()

            # Save latest checkpoint
            ckpt_path = self.checkpoints_dir / "latest.pth"
            torch.save({"model": self.model.state_dict()}, ckpt_path)

            # Evaluation
            if epoch % cfg.eval_freq == 0 and epoch != 0:
                t1 = time.time()
                result = evaluate_crowd_no_overlap(
                    self.model, self.data_loader_val, self.device
                )
                t2 = time.time()

                mae_history.append(result[0])
                mse_history.append(result[1])
                density_mae_history.append(result[2])
                density_mse_history.append(result[3])

                logger.info(
                    f"[Eval] mae={result[0]:.2f}  mse={result[1]:.2f}  "
                    f"time={t2 - t1:.1f}s  best_mae={np.min(mae_history):.2f}"
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

                # Save best MAE checkpoint
                if abs(np.min(mae_history) - result[0]) < 0.01:
                    torch.save(
                        {"model": self.model.state_dict()},
                        self.checkpoints_dir / "best_mae.pth",
                    )

        self.writer.close()
        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time / 60:.1f} min")

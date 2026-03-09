"""Training entry point using Hydra for configuration management.

Usage::

    # Basic training with default config
    python scripts/train.py data.data_root=/path/to/shanghaitech

    # Override hyperparameters
    python scripts/train.py data.data_root=/path/to/sha optimizer.lr=5e-5 epochs=1000

    # Resume from a checkpoint
    python scripts/train.py data.data_root=/path/to/sha resume=checkpoints/latest.pth
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from crowdcount.trainer import Trainer

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

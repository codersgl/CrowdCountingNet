"""MoECountNet training entry point using Hydra."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="moecount_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from crowdcount.trainers.moecount_trainer import MoECountTrainer

    trainer = MoECountTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

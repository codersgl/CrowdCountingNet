"""Search for the optimal classification threshold on the validation set.

Usage::

    python scripts/search_threshold.py \\
        data.data_root=/path/to/shanghaitech \\
        +predict.weight_path=weights/SHTechA.pth

The script runs a single forward pass over the val set, then sweeps
thresholds in [0.1, 0.95] (step 0.01) and prints the MAE at each
candidate, highlighting the best.
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.engine import collect_scores_and_counts, search_optimal_threshold
from crowdcount.models import build_model
from crowdcount.models.checkpoint import load_model_state_dict
from crowdcount.utils.logging import logger, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(log_dir=".", log_file="search_threshold.log")

    predict_cfg = OmegaConf.to_container(cfg, resolve=True)
    weight_path = predict_cfg.get("predict", {}).get(
        "weight_path", "weights/SHTechA.pth"
    )
    gpu_id = cfg.gpu_id

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(cfg, training=False)
    model.to(device)

    if weight_path and os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location="cpu")
        load_model_state_dict(model, checkpoint, logger=logger)
        logger.info(f"Loaded weights from {weight_path}")
    else:
        logger.warning(
            f"Weight file not found: {weight_path}. Using random initialisation."
        )

    # Build val dataloader
    use_depth = bool(getattr(cfg.model, "use_depth", False))
    _, val_set = build_dataset(cfg)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(val_set),
        drop_last=False,
        collate_fn=collate_fn_crowd_depth if use_depth else collate_fn_crowd,
        num_workers=cfg.num_workers,
    )

    # Collect scores in a single forward pass
    logger.info(f"Running forward pass on {len(val_set)} val images...")
    all_scores, gt_counts, density_sums = collect_scores_and_counts(
        model, val_loader, device, use_depth=use_depth
    )

    # Search
    best_t, best_mae, results = search_optimal_threshold(all_scores, gt_counts)

    # Print results table
    logger.info("--- Threshold Search Results ---")
    logger.info(f"{'Threshold':>10s}  {'MAE':>8s}")
    for t in sorted(results.keys()):
        marker = " <-- best" if t == best_t else ""
        logger.info(f"{t:10.2f}  {results[t]:8.2f}{marker}")

    logger.info(f"\nBest threshold: {best_t:.2f}  (MAE = {best_mae:.2f})")
    logger.info(
        f"To use: python scripts/predict.py eval_counting.threshold={best_t:.2f} ..."
    )


if __name__ == "__main__":
    main()

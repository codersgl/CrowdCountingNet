"""Strict checkpoint evaluation for DSGCNet.

This script is intended for final, reproducible model testing rather than
ad-hoc visualization. It reconstructs the model from the checkpoint run's
Hydra config, forces test-time data settings, strictly loads weights, and
writes both aggregate metrics and per-image predictions.

Usage::

    uv run python scripts/test_model.py \
        +test.weight_path=outputs/2026-04-22/19-23-32/checkpoints/best_mae.pth

Optional examples::

    uv run python scripts/test_model.py \
        +test.weight_path=... \
        data.data_root=data/shanghaitech/part_A_final \
        +test.tta=true \
        +test.seeds='[42,43,44]'
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler

from crowdcount.data import build_dataset, collate_fn_crowd
from crowdcount.data.collate import collate_fn_crowd_depth
from crowdcount.engine import evaluate_crowd_no_overlap, search_optimal_threshold
from crowdcount.models import build_model
from crowdcount.utils.logging import logger, setup_logger


@dataclass(frozen=True)
class ImagePrediction:
    image_id: int
    gt_count: int
    pred_fixed: int
    pred_best: int
    density_pred: float

    @property
    def abs_err_fixed(self) -> float:
        return float(abs(self.pred_fixed - self.gt_count))

    @property
    def abs_err_best(self) -> float:
        return float(abs(self.pred_best - self.gt_count))

    @property
    def sq_err_fixed(self) -> float:
        return float((self.pred_fixed - self.gt_count) ** 2)

    @property
    def sq_err_best(self) -> float:
        return float((self.pred_best - self.gt_count) ** 2)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _find_checkpoint_config(weight_path: Path) -> Path:
    candidates = [
        weight_path.parent.parent / ".hydra" / "config.yaml",
        weight_path.parent / ".hydra" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find checkpoint Hydra config. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _cli_config_overrides() -> DictConfig:
    """Return non-test CLI key overrides that should patch the checkpoint config.

    The checkpoint config is authoritative for model structure. We still allow
    concrete dotted-value overrides such as ``data.data_root=...`` or
    ``gpu_id=1``. Hydra group overrides are intentionally ignored here because
    they could silently create a model architecture different from the saved
    weights.
    """
    overrides: list[str] = []
    try:
        task_overrides = HydraConfig.get().overrides.task
    except Exception:
        task_overrides = []

    safe_top_level = {"gpu_id", "num_workers", "seed"}
    for raw in task_overrides:
        item = str(raw)
        normalized = item[1:] if item.startswith("+") else item
        if normalized.startswith("test."):
            continue
        if "=" not in normalized:
            logger.warning(
                f"Ignoring malformed override for strict test config: {item}"
            )
            continue
        key = normalized.split("=", 1)[0]
        if "." not in key and key not in safe_top_level:
            logger.warning(
                f"Ignoring non-dotted override for strict test config: {item}"
            )
            continue
        overrides.append(normalized)
    return OmegaConf.from_dotlist(overrides)


def _load_checkpoint_config(hydra_cfg: DictConfig, weight_path: Path) -> DictConfig:
    config_path = _find_checkpoint_config(weight_path)
    run_cfg = OmegaConf.load(config_path)
    cli_patch = _cli_config_overrides()
    cfg = OmegaConf.merge(run_cfg, cli_patch)

    # Keep test options from the live Hydra config separate from the checkpoint
    # training config. They control the script, not the model architecture.
    if "test" in hydra_cfg:
        cfg.test = deepcopy(hydra_cfg.test)
    return cfg


def _disable_augmentation(cfg: DictConfig) -> None:
    if "data" not in cfg:
        return
    cfg.data.batch_size = 1
    cfg.data.patch = False
    cfg.data.flip = False
    cfg.data.num_patches = 1

    aug_cfg = cfg.data.get("augmentation", None)
    if aug_cfg is None:
        return
    for _, section in aug_cfg.items():
        if isinstance(section, DictConfig) and "enabled" in section:
            section.enabled = False


def _needs_depth(cfg: DictConfig) -> bool:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return False
    return any(
        bool(getattr(model_cfg, key, False))
        for key in (
            "use_depth",
            "use_depth_geo",
            "use_depth_dual_vgg",
            "use_depth_attn",
        )
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loader(cfg: DictConfig) -> tuple[DataLoader, int]:
    _, test_set = build_dataset(cfg)
    loader = DataLoader(
        test_set,
        batch_size=1,
        sampler=SequentialSampler(test_set),
        drop_last=False,
        collate_fn=collate_fn_crowd_depth if _needs_depth(cfg) else collate_fn_crowd,
        num_workers=int(getattr(cfg, "num_workers", 0)),
    )
    return loader, len(test_set)


def _load_model(
    cfg: DictConfig, weight_path: Path, device: torch.device
) -> torch.nn.Module:
    model = build_model(cfg, training=False)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logger.error(
            "Strict checkpoint loading failed. The config and weights do not match."
        )
        logger.error(str(exc))
        raise SystemExit(1) from exc
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _collect_scores(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    use_depth: bool,
) -> tuple[list[torch.Tensor], list[int], list[int], list[float]]:
    model.eval()
    all_scores: list[torch.Tensor] = []
    gt_counts: list[int] = []
    image_ids: list[int] = []
    density_sums: list[float] = []

    for batch in loader:
        if use_depth:
            samples, targets, depth_map = batch
            depth_map = torch.stack(depth_map).to(device)
        else:
            samples, targets = batch
            depth_map = None

        samples = samples.to(device)
        outputs = (
            model(samples, depth_map=depth_map)
            if depth_map is not None
            else model(samples)
        )
        scores = torch.softmax(outputs["pred_logits"], dim=-1)[:, :, 1]
        assert scores.shape[0] == 1, "Strict test expects batch_size=1"

        target = targets[0]
        all_scores.append(scores[0].cpu())
        gt_counts.append(int(target["point"].shape[0]))
        image_ids.append(int(target["image_id"].item()))
        density_sums.append(float(torch.sum(outputs["density_out"]).item()))

    return all_scores, gt_counts, image_ids, density_sums


def _counts_at_threshold(all_scores: list[torch.Tensor], threshold: float) -> list[int]:
    return [int((scores > threshold).sum().item()) for scores in all_scores]


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _bootstrap_ci(
    values: np.ndarray,
    reducer: str = "mean",
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    if values.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[indices]
    if reducer == "rmse":
        stats = np.sqrt(np.mean(np.square(samples), axis=1))
    else:
        stats = np.mean(samples, axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def _summarize_counts(gt_counts: list[int], pred_counts: list[int]) -> dict[str, Any]:
    errors = np.asarray(pred_counts, dtype=np.float64) - np.asarray(
        gt_counts, dtype=np.float64
    )
    abs_errors = np.abs(errors)
    return {
        "mae": float(np.mean(abs_errors)),
        "mse": _rmse(errors),
        "mae_ci95": _bootstrap_ci(abs_errors, reducer="mean"),
        "mse_ci95": _bootstrap_ci(errors, reducer="rmse"),
    }


def _density_metrics(
    gt_counts: list[int], density_sums: list[float]
) -> dict[str, float]:
    errors = np.asarray(density_sums, dtype=np.float64) - np.asarray(
        gt_counts, dtype=np.float64
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": _rmse(errors),
    }


def _write_per_image(path: Path, rows: list[ImagePrediction]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "gt",
                "pred_fixed_thr",
                "pred_best_thr",
                "density_pred",
                "abs_err_fixed",
                "abs_err_best",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "gt": row.gt_count,
                    "pred_fixed_thr": row.pred_fixed,
                    "pred_best_thr": row.pred_best,
                    "density_pred": f"{row.density_pred:.6f}",
                    "abs_err_fixed": f"{row.abs_err_fixed:.6f}",
                    "abs_err_best": f"{row.abs_err_best:.6f}",
                }
            )


def _strict_eval_once(
    cfg: DictConfig,
    weight_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(cfg, weight_path, device)
    loader, dataset_size = _build_loader(cfg)
    use_depth = _needs_depth(cfg)

    logger.info(f"Running strict test on {dataset_size} test images (seed={seed})...")
    all_scores, gt_counts, image_ids, density_sums = _collect_scores(
        model, loader, device, use_depth=use_depth
    )

    eval_cfg = getattr(cfg, "eval_counting", None)
    fixed_threshold = float(getattr(eval_cfg, "threshold", 0.5)) if eval_cfg else 0.5
    pred_fixed = _counts_at_threshold(all_scores, fixed_threshold)
    best_threshold, best_mae, threshold_results = search_optimal_threshold(
        all_scores, gt_counts
    )
    pred_best = _counts_at_threshold(all_scores, float(best_threshold))

    rows = [
        ImagePrediction(
            image_id=image_id,
            gt_count=gt,
            pred_fixed=fixed,
            pred_best=best,
            density_pred=density,
        )
        for image_id, gt, fixed, best, density in zip(
            image_ids, gt_counts, pred_fixed, pred_best, density_sums
        )
    ]
    _write_per_image(output_dir / f"per_image_seed_{seed}.csv", rows)

    fixed_metrics = _summarize_counts(gt_counts, pred_fixed)
    best_metrics = _summarize_counts(gt_counts, pred_best)
    density = _density_metrics(gt_counts, density_sums)

    metrics: dict[str, Any] = {
        "seed": seed,
        "dataset_size": dataset_size,
        "fixed_threshold": fixed_threshold,
        "fixed_threshold_metrics": fixed_metrics,
        "best_threshold": float(best_threshold),
        "best_threshold_metrics": {**best_metrics, "mae_from_search": float(best_mae)},
        "density_metrics": density,
        "threshold_search": {str(k): float(v) for k, v in threshold_results.items()},
    }

    logger.info(
        "Seed {seed}: fixed@{thr:.2f} MAE={mae:.3f} MSE={mse:.3f}; "
        "best@{best_t:.2f} MAE={best_mae:.3f}; density MAE={den_mae:.3f}".format(
            seed=seed,
            thr=fixed_threshold,
            mae=fixed_metrics["mae"],
            mse=fixed_metrics["mse"],
            best_t=float(best_threshold),
            best_mae=best_metrics["mae"],
            den_mae=density["mae"],
        )
    )
    return metrics


def _run_tta_eval(
    cfg: DictConfig,
    weight_path: Path,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_tta = deepcopy(cfg)
    if "eval_tta" not in cfg_tta:
        cfg_tta.eval_tta = {}
    cfg_tta.eval_tta.enabled = True

    model = _load_model(cfg_tta, weight_path, device)
    loader, dataset_size = _build_loader(cfg_tta)
    mae, mse, density_mae, density_mse = evaluate_crowd_no_overlap(
        model,
        loader,
        device,
        use_depth=_needs_depth(cfg_tta),
        cfg=cfg_tta,
    )
    logger.info(
        f"TTA test on {dataset_size} images: MAE={mae:.3f} MSE={mse:.3f} "
        f"density_MAE={density_mae:.3f} density_MSE={density_mse:.3f}"
    )
    return {
        "seed": seed,
        "dataset_size": dataset_size,
        "mae": float(mae),
        "mse": float(mse),
        "density_mae": float(density_mae),
        "density_mse": float(density_mse),
        "eval_tta": _to_builtin(OmegaConf.to_container(cfg_tta.eval_tta, resolve=True)),
    }


def _metadata(cfg: DictConfig, weight_path: Path) -> dict[str, Any]:
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "git_commit": _git_commit(),
        "config_sha256": _sha256_text(resolved_yaml),
        "weight_sha256": _sha256_file(weight_path),
        "weight_path": str(weight_path),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device_name),
        "argv": sys.argv,
    }


def _test_options(cfg: DictConfig) -> dict[str, Any]:
    test_cfg = getattr(cfg, "test", {})
    return _to_builtin(OmegaConf.to_container(test_cfg, resolve=True))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(hydra_cfg: DictConfig) -> None:
    hydra_output = Path(HydraConfig.get().runtime.output_dir)
    hydra_output.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(hydra_output), log_file="test.log")

    test_cfg = getattr(hydra_cfg, "test", None)
    if test_cfg is None or not getattr(test_cfg, "weight_path", None):
        raise ValueError(
            "Pass a checkpoint with +test.weight_path=/path/to/checkpoint.pth"
        )

    weight_path = Path(str(test_cfg.weight_path)).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    cfg = _load_checkpoint_config(hydra_cfg, weight_path)
    _disable_augmentation(cfg)
    cfg.eval_tta.enabled = False

    seeds = list(getattr(test_cfg, "seeds", [int(getattr(cfg, "seed", 42))]))
    seeds = [int(seed) for seed in seeds]
    run_tta = bool(getattr(test_cfg, "tta", False))

    logger.info(f"Loaded checkpoint config from {_find_checkpoint_config(weight_path)}")
    logger.info(f"Strict test output dir: {hydra_output}")
    logger.info(f"Test options: {_test_options(cfg)}")

    seed_metrics = [
        _strict_eval_once(cfg, weight_path, hydra_output, seed=seed) for seed in seeds
    ]

    deterministic = True
    if len(seed_metrics) > 1:
        first_mae = seed_metrics[0]["fixed_threshold_metrics"]["mae"]
        deterministic = all(
            abs(metric["fixed_threshold_metrics"]["mae"] - first_mae) <= 1e-6
            for metric in seed_metrics[1:]
        )
        if not deterministic:
            logger.warning(
                "Evaluation changed across seeds; inspect deterministic settings."
            )

    tta_metrics = None
    if run_tta:
        tta_metrics = _run_tta_eval(cfg, weight_path, seed=seeds[0])

    metrics: dict[str, Any] = {
        "metadata": _metadata(cfg, weight_path),
        "test_options": _test_options(cfg),
        "deterministic_across_seeds": deterministic,
        "seeds": seed_metrics,
        "tta": tta_metrics,
    }

    metrics_path = hydra_output / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(metrics), handle, indent=2, sort_keys=True)

    logger.info(f"Wrote metrics to {metrics_path}")
    logger.info(f"Wrote per-image CSV files to {hydra_output}")


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSGCNet (Dual-Stream Graph Convolutional Network) is a crowd counting model published at PRCV 2025. It achieves state-of-the-art performance via dual-branch GCN-based feature correlation mining with density prediction auxiliary task.

- **Paper**: [arXiv 2509.02261](https://arxiv.org/abs/2509.02261)
- **Datasets**: ShanghaiTech Part A/B, UCF-QNRF
- **Package manager**: `uv` (not pip directly)
- **Config system**: Hydra + OmegaConf

## Commands

```bash
# Install dependencies
uv sync
uv sync --extra dev  # includes pytest

# Run tests (no GPU or real data required)
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/crowdcount --cov-report=term-missing

# Training
python scripts/train.py data.data_root=DATA_ROOT
python scripts/train.py data.data_root=DATA_ROOT epochs=3500 optimizer.lr=0.0001 gpu_id=0

# Resume training
python scripts/train.py data.data_root=DATA_ROOT resume=checkpoints/latest.pth

# Prediction / inference
python scripts/predict.py \
    +predict.weight_path=checkpoints/SHTechA.pth \
    +predict.root_dir=./sha_a/test \
    +predict.output_dir=./pred_result \
    +predict.threshold=0.5

# TensorBoard
tensorboard --logdir runs/
```

Outputs are written to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`.

## Architecture

```
scripts/train.py          # Hydra entry point → instantiates Trainer
src/crowdcount/
  trainer.py              # High-level training orchestration
  engine.py               # train_one_epoch(), evaluate_crowd_no_overlap()
  models/
    dsgcnet.py           # DSGCnet: top-level model
    backbone.py          # VGG16/VGG16-BN, DINOv2 wrappers
    neck.py              # Decoder_SPD_PAFPN: Space-to-Depth + PA-FPN
    head.py              # Density_pred, RegressionModel, ClassificationModel
    gcn.py               # DensityGCNProcessor, FeatureGCNProcessor (k=4)
    anchor.py            # AnchorPoints: spatial anchor grid
    criterion.py         # SetCriterion_Crowd: multi-task loss
    matcher.py           # HungarianMatcher_Crowd
  data/
    dataset.py           # SHHA dataset class
    prepare.py           # Auto-generates density maps
    transforms.py        # Data augmentation
    collate.py           # NestedTensor collation
    loader.py            # DataLoader factory
  plugins/
    gm.py               # GateMechanism (disabled by default)
    msaa.py             # MultiScaleAdaptiveAggregation (disabled by default)
```

## Configuration

All hyperparameters live in `configs/`; override via Hydra CLI dot-notation:
- `configs/config.yaml` - Root config (epochs=2500, seed=42, clip_max_norm=0.1)
- `configs/data/shha.yaml` - Data settings
- `configs/model/dsgcnet.yaml` - Model architecture
- `configs/optimizer/adamw.yaml` - Optimizer settings (AdamW)
- `configs/scheduler/cosine_annealing.yaml` - LR scheduler

## Key Conventions

- **Python 3.10+**: Uses union syntax `X | Y`, `from __future__ import annotations`
- **Logging**: Use `loguru` (`from loguru import logger`), never `print()`
- **Plugins**: `use_gm` and `use_msaa` are disabled by default; enable via CLI
- **Density maps**: Auto-generated on first run, cached to `{data_root}/gt_density_maps/train/`
- **Evaluation**: Requires `batch_size=1` (enforced in code)
- **Graph k-factor**: Hardcoded to `k=4` in `gcn.py`

## Testing Requirements

Tests must not require GPU or real dataset files. Run with `uv run pytest`.

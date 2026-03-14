# CrowdCounting-DSGCNet Project Guidelines

## Project Overview

**DSGCNet** (Dual-Stream Graph Convolutional Network) is a crowd counting model published at PRCV 2025.
It achieves state-of-the-art performance (MAE 48.9 / 5.9 on ShanghaiTech A/B) via dual-branch GCN-based
feature correlation mining combined with a density prediction auxiliary task.

- **Paper**: [arXiv 2509.02261](https://arxiv.org/abs/2509.02261)
- **Backbone**: VGG-16/VGG-16BN (default) or DINOv2
- **Config system**: Hydra + OmegaConf (`configs/`)
- **Package manager**: `uv` (not pip directly)

---

## Build and Test

```bash
# Install dependencies
uv sync
uv sync --extra dev          # includes pytest, pytest-cov

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

Outputs are written to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/` (checkpoints + TensorBoard runs).

---

## Architecture

```
scripts/train.py          # Hydra entry point → instantiates Trainer
src/crowdcount/
  trainer.py              # High-level training orchestration, checkpoint management
  engine.py               # train_one_epoch(), evaluate_crowd_no_overlap()
  models/
    dsgcnet.py            # DSGCnet: top-level model integrating all components
    backbone.py           # Backbone_VGG (VGG16/VGG16-BN), DINOv2 wrappers
    neck.py               # Decoder_SPD_PAFPN: Space-to-Depth + PA-FPN
    head.py               # Density_pred, RegressionModel, ClassificationModel
    gcn.py                # DensityGCNProcessor, FeatureGCNProcessor (k-NN, k=4)
    anchor.py             # AnchorPoints: spatial anchor grid generation
    criterion.py          # SetCriterion_Crowd: classification + regression + density loss
    matcher.py            # HungarianMatcher_Crowd: bipartite matching
  data/
    dataset.py            # SHHA dataset class (ShanghaiTech A/B)
    prepare.py            # Auto-generates & caches density maps (.npy) on first run
    transforms.py         # ColorJitter, RandomGrayscale, random scaling [0.7, 1.3]
    collate.py            # collate_fn_crowd_train / collate_fn_crowd (NestedTensor)
    loader.py             # DataLoader factory
  plugins/
    gm.py                 # GateMechanism: learnable 3-stream weighted fusion (use_gm: true)
    msaa.py               # MultiScaleAdaptiveAggregation: channel+spatial attention (use_msaa: true)
  utils/
    logging.py            # loguru-based setup_logger
    misc.py               # nested_tensor_from_tensor_list, distributed utils
configs/
  config.yaml             # Root config; defaults: epochs=2500, seed=42, clip_max_norm=0.1
  data/shha.yaml          # batch_size=8, patch=True, flip=True
  model/dsgcnet.yaml      # backbone, row/line anchors, plugin flags
  optimizer/adam.yaml     # lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4
  scheduler/step_lr.yaml  # StepLR, lr_drop=800
```

**Forward pass summary:**
`Image → Backbone → PA-FPN → density_map (aux) → Density GCN + Feature GCN → [Gate/Alpha fusion] → Regression + Classification heads → point coords + logits`

---

## Conventions

### Code Style

- **Python 3.10+**: uses `X | Y` union syntax, `from __future__ import annotations`
- **Type hints**: always annotate function signatures and key variables
- **Logging**: use `loguru` (`from loguru import logger`); never use `print()` for runtime info
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables, `_underscore` for private methods

### Configuration

- All hyperparameters live in `configs/`; override via Hydra CLI dot-notation
- Plugins (`use_gm`, `use_msaa`) are **disabled by default**; enable explicitly in config or CLI
- Never hardcode paths; always use `data.data_root` config key

### Testing

- Tests are in `tests/`; key fixtures in `tests/conftest.py`:
  - `device` (CPU), `base_cfg` (minimal OmegaConf config), `sample_batch` ([2,3,128,128]), `dummy_targets`
- Tests must not require GPU or real dataset files
- Run with `uv run pytest`, not plain `pytest`

### Data

- Density maps are auto-generated on first training run and cached to `{data_root}/gt_density_maps/train/`
- **Do not delete** `gt_density_maps/` unless you intend to regenerate
- Evaluation dataloader requires `batch_size=1` (enforced in code)

---

## Common Pitfalls

| Issue                            | Fix                                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `RuntimeError` on training start | Must provide `data.data_root=/path` on CLI                                                   |
| Backbone learning rate too high  | Backbone uses `lr_backbone` (1e-5), not base `lr` (1e-4); this is intentional                |
| Graph k-factor not configurable  | `DensityGCNProcessor`/`FeatureGCNProcessor` use hardcoded `k=4`; change in `gcn.py` directly |
| Hydra changes working dir        | `hydra.job.chdir: false` is set; relative paths resolve from project root                    |
| Missing `torch_geometric`        | Included in `pyproject.toml`; install via `uv sync` not pip                                  |

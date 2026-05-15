# CrowdCountingNet Agent Instructions

This repository implements DSGCNet, a Hydra/OmegaConf crowd-counting research codebase. Keep agent changes small, config-driven, and testable on CPU unless the user explicitly asks for a training run.

## Start Here

- Read [README.md](../README.md) for project overview, dataset layouts, and user-facing commands.
- Read [CLAUDE.md](../CLAUDE.md) for the compact architecture map and command cheatsheet.
- Use the experiment reports as references instead of duplicating their findings: [docs/ablation_full_report_2026-04-26.md](../docs/ablation_full_report_2026-04-26.md) and [docs/density_generation_quality_report_2026-05-08.md](../docs/density_generation_quality_report_2026-05-08.md).

## Commands Agents Should Use

```bash
uv sync
uv sync --extra dev
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/crowdcount --cov-report=term-missing
```

- Use `uv run` for Python tooling and tests. Do not assume plain `pytest` or direct `pip` installs.
- Training and analysis entry points are scripts under [scripts/](../scripts/). For training, pass `data.data_root=...`; Hydra leaves `hydra.job.chdir: false`, so relative paths resolve from the repo root.
- Tests must not require GPU, external downloads, or the real datasets. Prefer existing fixtures in [tests/conftest.py](../tests/conftest.py).

## Current Defaults To Respect

- Root defaults are in [configs/config.yaml](../configs/config.yaml): `optimizer: adamw`, `scheduler: cosine_annealing`, `epochs: 3500`, `seed: 42`, `clip_max_norm: 0.1`.
- Optimizer defaults are [configs/optimizer/adamw.yaml](../configs/optimizer/adamw.yaml): `lr=1e-4`, `lr_backbone=1e-5`, `weight_decay=1e-4`.
- `step_lr` is available but not the root default; [configs/scheduler/step_lr.yaml](../configs/scheduler/step_lr.yaml) uses `lr_drop: 3500`, so with `epochs: 3500` it effectively does not decay during a standard run.
- Most experimental modules are disabled by default in [configs/model/dsgcnet.yaml](../configs/model/dsgcnet.yaml), including GM, MSAA, depth branches, MoE variants, uncertainty, focal/QFL, refinement, and multi-scale density supervision.

## Best Known Run

- Current best known experiment: [outputs/2026-04-25/22-51-51](../outputs/2026-04-25/22-51-51), with best `MAE=48.51` / `MSE=79.87` in [train.log](../outputs/2026-04-25/22-51-51/train.log).
- Use its Hydra files for reproducibility: [config.yaml](../outputs/2026-04-25/22-51-51/.hydra/config.yaml) and [overrides.yaml](../outputs/2026-04-25/22-51-51/.hydra/overrides.yaml).
- Key overrides: `model.use_gm=true`, `model.use_dap_neck=true`, `model.use_density_attention=true`, `model.density_head_version=v3`, `model.gcn_conv_type=gatv2`, `scheduler=step_lr`, `scheduler.lr_drop=800`, `data.density_generation.hybrid=true`.
- Best checkpoint is [outputs/2026-04-25/22-51-51/checkpoints/best_mae.pth](../outputs/2026-04-25/22-51-51/checkpoints/best_mae.pth). Treat this run as the performance baseline before claiming a new improvement.

## Architecture Boundaries

- Keep top-level model wiring in [src/crowdcount/models/dsgcnet.py](../src/crowdcount/models/dsgcnet.py); use model/config flags instead of hardcoded experiment switches.
- Core model pieces live in [src/crowdcount/models/](../src/crowdcount/models/): backbone, neck/DAP neck, density/classification/regression heads, GCN processors, matcher, and criterion.
- Optional/experimental modules live in [src/crowdcount/plugins/](../src/crowdcount/plugins/): density losses, GM/MSAA, MoE variants, depth/geometric priors, RCCFormer, and SA-DGAT.
- Data loading, density generation, transforms, and collation live in [src/crowdcount/data/](../src/crowdcount/data/). Keep dataset paths configurable through `configs/data/*.yaml` and CLI overrides.

## Code Style

- Python 3.10+ style is expected: `from __future__ import annotations`, `X | Y` unions, typed public function signatures, and `snake_case` names.
- Use `loguru` (`from loguru import logger`) for runtime logging; avoid `print()` in library/training code.
- Prefer structured Hydra config fields over ad hoc constants. Do not hardcode dataset roots, checkpoint paths, or output paths.
- Keep new comments sparse and useful; preserve existing research-script pragmatism where the surrounding file is procedural.

## Project Pitfalls

- Density maps are generated and cached under dataset roots. Do not delete `gt_density_maps*` directories unless regeneration is intended.
- Density generation modes and parameters affect cache paths; when changing mode, sigma, perspective, or hybrid settings, add or update tests that prove cache invalidation.
- Evaluation dataloaders require `batch_size=1`.
- Density loss magnitude is easy to misread: global `density_loss_weight` combines with additional scaling in the training engine, so retune carefully when swapping MSE, Bayesian, ASACL, DM-Count, MDS, or SSIM variants.
- `use_msaa=true` together with `density_multi_scale.enabled=true` can create channel mismatches between `MsaaAdaptiveLayer` outputs and density prediction blocks unless the channel contracts are updated together.
- When diagnosing experiments, trust Hydra runtime outputs (`outputs/.../.hydra/config.yaml` and overrides) over defaults. This repo has had typo-like config values in logs, so verify the actual run config.

## Testing Guidance

- For model and data changes, run the narrowest relevant tests first, then broader `uv run pytest tests/ -v` when risk is shared.
- Add tests under [tests/](../tests/) using synthetic tensors or temporary fake datasets. Real ShanghaiTech/UCF-QNRF data and GPU availability must not be required.
- For density-generation changes, cover numerical integral behavior, invalid parameters, and cache directory naming.

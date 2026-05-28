# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSGCNet (Dual-Stream Graph Convolutional Network) is a crowd counting codebase published at PRCV 2025. It contains two model families:

- **DSGCNet** — Dual-branch GCN-based feature correlation mining (the published paper model). Best known result: MAE=48.5 / MSE=79.9 on SHA.
- **MoECountNet** — Separable density-map counter with Mixture-of-Experts pixel-wise routing (newer, active development). Config-driven, currently on `exp1` branch.

- **Paper**: [arXiv 2509.02261](https://arxiv.org/abs/2509.02261)
- **Datasets**: ShanghaiTech Part A/B, UCF-QNRF
- **Package manager**: `uv` (not pip directly)
- **Config system**: Hydra + OmegaConf

## Commands

**Always use `uv run`** to invoke Python scripts so the virtual environment is active. Do not assume `pip` or bare `python`.

```bash
# Install
uv sync
uv sync --extra dev  # includes pytest

# Tests (no GPU or real data required)
uv run pytest tests/ -v
uv run pytest tests/test_deformable_expert.py -v  # single test file
uv run pytest tests/ --cov=src/crowdcount --cov-report=term-missing  # with coverage

# === DSGCNet (paper model) ===
uv run python scripts/train.py data.data_root=DATA_ROOT
uv run python scripts/train.py data.data_root=DATA_ROOT epochs=3500 optimizer.lr=0.0001 gpu_id=0
uv run python scripts/train.py data.data_root=DATA_ROOT resume=checkpoints/latest.pth

# Reproduce the best known SHA result (MAE=48.51 / MSE=79.87)
uv run python scripts/train.py data.data_root=DATA_ROOT \
  model.use_gm=true model.use_dap_neck=true model.use_density_attention=true \
  model.density_head_version=v3 model.gcn_conv_type=gatv2 \
  scheduler=step_lr scheduler.lr_drop=800 \
  data.density_generation.hybrid=true

# === MoECountNet (newer architecture) ===
uv run python scripts/train_moecount.py data.data_root=DATA_ROOT
# Smoke test (no pretrained weights, 1 epoch)
uv run python scripts/train_moecount.py data.data_root=DATA_ROOT \
  model.backbone.pretrained=false epochs=1 data.batch_size=1 \
  data.num_patches=1 num_workers=0

# Prediction / inference
uv run python scripts/predict.py \
    +predict.weight_path=checkpoints/SHTechA.pth \
    +predict.root_dir=./sha_a/test \
    +predict.output_dir=./pred_result \
    +predict.threshold=0.5

# TensorBoard
tensorboard --logdir runs/
```

Alternative entry points are registered in pyproject.toml:
```bash
crowdcount-train data.data_root=DATA_ROOT     # ≡ scripts/train.py
crowdcount-predict +predict.weight_path=...    # ≡ scripts/predict.py
```

Outputs are written to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`.

**`hydra.job.chdir: false`** is set in configs, so the working directory stays at the repo root during runs. All relative paths (e.g., `resume=checkpoints/latest.pth`) resolve from the project root, not from the output directory.

**uv index mirror**: `pyproject.toml` defaults to `pypi.tuna.tsinghua.edu.cn`. Users outside China may need to remove or override the `[[tool.uv.index]]` entry.

## Architecture

### DSGCNet (paper model)

```
scripts/train.py          → instantiates Trainer (trainer.py)
src/crowdcount/
  trainer.py              → training orchestration
  engine.py               → train_one_epoch(), evaluate_crowd_no_overlap()
  models/
    dsgcnet.py            → DSGCnet: top-level model with fusion_mode dispatch
    backbone.py           → VGG16/VGG16-BN, DINOv2 wrappers
    neck.py               → Decoder_SPD_PAFPN (SPD=Space-to-Depth), SPDBiFPNNeck
    head.py               → Density_pred, RegressionModel, ClassificationModel
    gcn.py                → DensityGCNProcessor, FeatureGCNProcessor (k=4)
    anchor.py             → AnchorPoints spatial grid
    criterion.py          → SetCriterion_Crowd: multi-task loss
    matcher.py            → HungarianMatcher_Crowd
  data/                   → dataset, transforms, collation, density-map prep
  plugins/                → optional/experimental modules (see below)
```

### MoECountNet (standalone model)

```
scripts/train_moecount.py → entry point → MoECountTrainer
src/crowdcount/
  trainers/
    moecount_trainer.py   → MoECountTrainer: build, train loop, checkpoint
  models/moecount/
    moecount.py           → MoECountNet: top-level model + build_moecount()
    backbone.py           → MoEConvNeXtBackbone (ConvNeXt, features_only)
    neck.py               → DeepBiFPNNeck (3-level SPD-BiFPN), EnhancedFPNNeck (2-level)
    experts.py            → HeterogeneousSparseMoE: 3 scale×paradigm experts + shared expert
    deformable_expert.py  → DeformableCrossScaleExpert: DAT-style multi-scale deformable attention
    gate.py               → PixelSoftGate, SparseTop2Gate, MultiScaleSparseTop2Gate
    head.py               → DensityHead (softplus), PointPredHead (P2PNet-style)
    losses.py             → MoECountLoss (PML/Bayesian + Count + Balance + Point + OT)
    gcn_refine.py         → DensityGCNRefine (optional post-MoE GCN)
    engine.py             → train_moecount_one_epoch(), evaluate_moecount()
```

**MoECountNet forward flow**: Backbone(C2/C3/C4) → BiFPN Neck → HeterogeneousSparseMoE (shared expert + 3 routed experts with SparseTop2Gate Gumbel-Softmax routing) → optional GCN refine → DensityHead + PointPredHead.

The three experts are: `LocalDetailExpert` (stride-8, DWConv + SE), `DeformableCrossScaleExpert` (stride-8, DAT-style multi-scale deformable attention; replaces `SpatialRelationExpert`'s W-MSA when `use_deformable: true`), `GlobalDensityExpert` (stride-32, large-kernel DWConv + SE). All experts internally handle their own downsampling via SPD. `SparseTop2Gate` provides Gumbel-Softmax Top-2 sparse routing with temperature annealing (warmup → hard routing with straight-through gradients).

**MoECountNet loss**: `MoECountLoss` composites: primary density loss (BayesianLoss or ProximalMappingLoss) + CountLoss (L1) + LoadBalanceLoss (CV² importance+batch load) + optional PointPredHead auxiliary loss (Hungarian matching + focal) + optional SinkhornOT loss. Balance loss decays linearly to 0 after warmup.

### Plugins (experimental modules for DSGCNet)

Key plugins in `src/crowdcount/plugins/`:
- `moe.py` — ESCA attention + 3-expert MoE (count calibration / localization / density)
- `deformable_dual.py` — GuidedDeformableAttention + DeformableDualFusion (dual deformable-attention branches)
- `graph_moe.py` — GraphAttentionExpert (MHSA with density-similarity bias) + 5-expert GraphMoE
- `sdd_moe.py` — Scale-Decoupled MoE with OcclusionReasoningExpert
- `mamba_moe.py` — Mamba SSM blocks with spatial MoE routing
- `sa_dgat/` — Scale-Aware Deformable Graph Attention (DeformableGraphAttention + OcclusionAwareGAT + ScalePromptEmbedding)
- `neck_moe.py` — NeckScaleMoE (pre/post neck insertion)
- `depth_cross_attention.py` — Window-based RGB-query / depth-key-value cross-attention
- `rccformer.py` — IDConv (input-dependent deformable conv)
- `clip_prompt_density.py` — CLIP-prompted density head
- Multiple density losses: `bayesian_loss.py`, `dm_count_loss.py`, `asacl_loss.py`, `mds_loss.py`

## Configuration

### DSGCNet config
- `configs/config.yaml` — Root (epochs=3500, seed=42, clip_max_norm=0.1, optimizer=adamw, scheduler=cosine_annealing)
- `configs/model/dsgcnet.yaml` — Model architecture + fusion_mode (gcn, esca_moe, deformable_dual, etc.)
- `configs/data/shha.yaml` — Data settings
- `configs/optimizer/adamw.yaml` — Default: lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4
- Note: `step_lr` is available but not the root default; it was used as an override in the best known run (`scheduler=step_lr scheduler.lr_drop=800`)

### MoECountNet config
- `configs/moecount_config.yaml` — Root (epochs=1500, clip_max_norm=5.0, AMP enabled)
- `configs/model/moecount.yaml` — Model arch + deformable_expert settings
- `configs/data/shha.yaml` — Shared data config

Override via Hydra CLI dot-notation: `data.batch_size=4`, `model.moe.deformable_expert.use_deformable=false`, etc.

### MoECountNet deformable expert config

```yaml
model:
  moe:
    deformable_expert:
      use_deformable: true    # false → original W-MSA SpatialRelationExpert
      num_heads: 4
      num_sampling_points: 8  # K: per-scale sampling points
      num_scale_levels: 3     # L: P3(stride-8) + P4 + P5
      max_offset: 8.0         # tanh clamp in stride-8 pixels
      dropout: 0.1
      use_se: true
```

## Key Conventions

- **Python 3.10+**: `from __future__ import annotations`, `X | Y` unions
- **Logging**: Use `loguru` (`from loguru import logger`), never `print()`
- **Density maps**: Auto-generated on first training run, cached to `{data_root}/gt_density_maps/train/`
- **Evaluation**: `batch_size=1` is enforced for evaluation dataloaders
- **Graph k-factor**: Hardcoded to `k=4` in `gcn.py`
- **SPD (Space-to-Depth)**: Used throughout for lossless downsampling. Requires even spatial dimensions — pad before use.
- **Zero-initialization pattern**: Learnable offsets and residual gates in deformable modules start at zero so training begins with identity behavior
- **Tests**: Must not require GPU or real dataset files. Use synthetic tensors and fake configs.

## Pitfalls

- **Density map cache staleness**: Changing sigma, perspective, or hybrid settings silently reuses stale cached maps. Delete `gt_density_maps/` directories when changing density generation parameters.
- **MSAA + multi_scale channel mismatch**: `use_msaa=true` combined with `density_multi_scale.enabled=true` can break unless channel contracts are updated together.
- **Trust Hydra output configs over defaults**: When diagnosing experiments, always read `outputs/.../.hydra/config.yaml` and `overrides.yaml`. This repo has had typo-like config values in logs.
- **Density loss magnitude**: `density_loss_weight` is a global scale applied in the training engine. Retune carefully when swapping loss types (MSE, Bayesian, ASACL, DM-Count, MDS) since each has different magnitude.
- **Evaluation dataloaders require batch_size=1**, enforced automatically. Do not override.

## Reference Documents

- Experiment reports (consult instead of duplicating findings):
  - `docs/ablation_full_report_2026-04-26.md`
  - `docs/density_generation_quality_report_2026-05-08.md`
- Best known run: `outputs/2026-04-25/22-51-51/` (MAE=48.51, MSE=79.87 on SHA). Checkpoint at `checkpoints/best_mae.pth`. Use this as the performance baseline before claiming improvement.
- Diagnostic scripts in `scripts/` are useful for debugging: `analyze_density_generation_quality.py`, `analyze_hard_score_band.py`, `calibrate_counts.py`, `search_threshold.py`, plus `diag_*.py` and `probe_*.py` tools.

## Training Considerations

- MoECountNet has 3 parameter groups: head (lr=1e-4), backbone (lr=1e-5), gate (lr=1e-4)
- MoECountNet uses SparseTop2Gate with Gumbel-Softmax temperature annealing. The gate starts in soft-routing warmup (all experts contribute with soft weights), then transitions to hard Top-2 routing with straight-through gradients. `model.moe.temperature_init`, `temperature_decay`, `warmup_epochs` control the schedule.
- Balance loss (CV² of expert load/importance) decays linearly to 0 over `decay_epochs` after warmup
- The deformable expert's residual gate is initialized to 0 (identity pass-through), so it needs many epochs to "warm up" — expect e2 gate load to start low and gradually increase
- DSGCNet default `clip_max_norm=0.1`; MoECountNet uses `clip_max_norm=5.0`
- MoECountNet uses `mixed_precision.enabled: true` by default; DSGCNet does not
- MoECountNet active development is on branch `exp1`

## MoECountNet Config Defaults

- `configs/moecount_config.yaml`: `epochs: 1500`, `weight_decay: 0.0001`, `use_pml: false` (BayesianLoss)
- `configs/model/moecount.yaml`: `output_stride: 8`, `backbone.arch: convnext_tiny`, `moe.top_k: 2`, `head.final_activation: softplus`, `head.final_weight_std: 0.01`
- The primary density loss is BayesianLoss (ICCV 2019) with `use_background: true`, `bg_ratio: 0.15`
- Point prediction auxiliary head is enabled by default (`head.use_point_head: true`)
- Sinkhorn OT loss is disabled by default (`moecount_loss.ot.enabled: false`)

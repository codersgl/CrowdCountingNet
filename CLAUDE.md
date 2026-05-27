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

```bash
# Install
uv sync
uv sync --extra dev  # includes pytest

# Tests (no GPU or real data required)
uv run pytest tests/ -v
uv run pytest tests/test_deformable_expert.py -v  # single test file

# === DSGCNet (paper model) ===
python scripts/train.py data.data_root=DATA_ROOT
python scripts/train.py data.data_root=DATA_ROOT epochs=3500 optimizer.lr=0.0001 gpu_id=0
python scripts/train.py data.data_root=DATA_ROOT resume=checkpoints/latest.pth

# === MoECountNet (newer architecture) ===
python scripts/train_moecount.py data.data_root=DATA_ROOT
# Smoke test (no pretrained weights, 1 epoch)
python scripts/train_moecount.py data.data_root=DATA_ROOT \
  model.backbone.pretrained=false epochs=1 data.batch_size=1 \
  data.num_patches=1 num_workers=0

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
    experts.py            → HeterogeneousSparseMoE: 3 scale×paradigm experts + PixelSoftGate
    deformable_expert.py  → DeformableCrossScaleExpert: DAT-style multi-scale deformable attention
    gate.py               → PixelSoftGate, SparseTop2Gate, MultiScaleSparseTop2Gate
    head.py               → DensityHead (softplus), PointPredHead (P2PNet-style)
    losses.py             → MoECountLoss (PML/Bayesian + Count + Balance + Point + OT)
    gcn_refine.py         → DensityGCNRefine (optional post-MoE GCN)
    engine.py             → train_moecount_one_epoch(), evaluate_moecount()
```

**MoECountNet forward flow**: Backbone(C2/C3/C4) → BiFPN Neck → HeterogeneousSparseMoE (shared expert + 3 routed experts with pixel-wise softmax gate) → optional GCN refine → DensityHead + PointPredHead.

The three experts are: `LocalDetailExpert` (stride-8, DWConv), `DeformableCrossScaleExpert` (stride-8, deformable attention; replaces `SpatialRelationExpert`'s W-MSA when `use_deformable: true`), `GlobalDensityExpert` (stride-32, large-kernel conv). All experts internally handle their own downsampling via SPD.

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
- `configs/config.yaml` — Root (epochs=2500, seed=42, clip_max_norm=0.1)
- `configs/model/dsgcnet.yaml` — Model architecture + fusion_mode (gcn, esca_moe, deformable_dual, etc.)
- `configs/data/shha.yaml` — Data settings

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
- **AMP**: MoECountNet uses `mixed_precision.enabled: true` by default; DSGCNet does not

## Training Considerations

- MoECountNet has 3 parameter groups: head (lr=1e-4), backbone (lr=1e-5), gate (lr=1e-4)
- Balance loss (CV² of expert load/importance) decays linearly to 0 over `decay_epochs`
- The deformable expert's residual gate is initialized to 0 (identity pass-through), so it needs many epochs to "warm up" — expect e2 gate load to start low and gradually increase
- MoECountNet optimizer config differs from DSGCNet: `weight_decay=0.01` (vs 1e-4), `amsgrad=false`
- DSGCNet default `clip_max_norm=0.1`; MoECountNet uses `clip_max_norm=5.0`

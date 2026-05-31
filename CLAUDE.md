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
uv run pytest tests/test_scale_decoupled_fusion.py -v  # scale_decoupled module
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

# === DSGCNet with ScaleDecoupledFusion (newer architecture) ===
uv run python scripts/train.py data.data_root=DATA_ROOT model.fusion_mode=scale_decoupled

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
    gcn.py                → DensityGCNProcessor, FeatureGCNProcessor (k=4), FeatureTransformerBlock
    scale_decoupled_fusion.py → ScaleDecoupledFusion: CNN/GCN/Transformer + CrossAttn (fusion_mode=scale_decoupled)
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

The three default routed experts are now **task-oriented** (controlled by `use_point_localization_expert`, `use_occlusion_reasoning_expert`, `use_density_pattern_expert`):

| Slot | Flag | Expert | Scale | Description |
|------|------|--------|-------|-------------|
| e0 | `use_point_localization_expert` (default: true in dsgcnet.yaml) | **DensityAdaptiveLocalExpert** (or PointLocalizationExpert mode) | stride-8 | Multi-scale dilated convs (d=1,2,3) + FFN + density-adaptive modulation. When `expert_pl_use_point_aux=true`, adds an internal point head (Hungarian matching + focal loss) for per-pixel point localization. |
| e1 | `use_occlusion_reasoning_expert` (default: true) | **OcclusionReasoningExpert** (or DeformableCrossScaleExpert) | stride-8 | Visibility assessment + 3×3 neighborhood context completion for occluded heads. Falls back to `DeformableCrossScaleExpert` (DAT-style multi-scale deformable attention) when disabled, or `SpatialRelationExpert` (W-MSA) when `use_deformable_expert=false`. |
| e2 | `use_density_pattern_expert` (default: true) | **DensityPatternExpert** (or GlobalDensityExpert) | stride-32 | PSPNet PPM + density bin classifier (8 bins). Falls back to `GlobalDensityExpert` (large-kernel DWConv + SE) when disabled. |

All three flags false → legacy experts: `LocalDetailExpert` / `SpatialRelationExpert` / `GlobalDensityExpert`.

All experts internally handle downsampling via SPD. **SharedExpert** (always active, × `shared_scale`) provides a common gradient highway. Recent enhancement: deepened to 3 residual blocks (`shared_num_blocks: 3`) with learnable `shared_scale` (default 0.5, `shared_scale_learnable: true`).

**Expert density dispatch** (`needs_density`): Each expert declares whether it receives the predicted density map as a kwarg. DensityAdaptiveLocalExpert uses it via a **zero-init modulation gate** (`density.detach()` — feed-forward only, no gradient feedback). GlobalDensityExpert concat-fuses it via Conv1×1 (gradient flows). DensityPatternExpert receives it for the density-bin classifier.

**Expert auxiliary supervision** (`compute_aux_loss`): Each expert can contribute auxiliary losses during training — e0 point localization loss, e1 occlusion consistency loss, e2 density bin classification loss. These are collected in `moe_aux_losses` and summed into `total_aux`.

**MoECountNet loss**: `MoECountLoss` composites: primary density loss (BayesianLoss or ProximalMappingLoss) + CountLoss (L1) + LoadBalanceLoss (CV² importance+batch load) + optional PointPredHead auxiliary loss (Hungarian matching + focal) + optional SinkhornOT loss + Total Variation smoothness regularizer (`tv.weight: 0.0005`) + density map MSE supervision against pre-computed GT density maps (`density_map.weight: 0.1`). Balance loss decays linearly to 0 after warmup.

**Gate variants**: Two router types in `gate.py`:
- **SparseTop2Gate** (default in DSGCNet `fusion_mode=moe`): Gumbel-Softmax Top-2 sparse routing with temperature annealing. Uses conv3×3 + dilated conv3×3 router. Supports `use_density_hint` (concat density features) and `use_density_bias` (per-expert logit bias from density). During warmup, all experts contribute with soft weights; after warmup, hard Top-2 with straight-through gradients.
- **PixelSoftGate**: HMoDE-style per-pixel softmax — all experts always contribute, no sparsity. Simpler, no temperature schedule. Default in standalone MoECountNet (`configs/model/moecount.yaml`).
- **MultiScaleSparseTop2Gate**: Extends SparseTop2Gate with stride-8/16/32 pooled features for the router.

**DSGCNet `fusion_mode=moe` integration**: When `model.fusion_mode=moe`, DSGCNet replaces the dual-stream GCN with `HeterogeneousSparseMoE` as a drop-in module:
```
features_pa [B,256,H/8,W/8] → HeterogeneousSparseMoE → feature_fl [B,256,H/8,W/8]
                                ↑ density as conditioning
```
The MoE output feeds directly into the standard DSGCNet prediction trunk (SharedPredictionTrunk → Regression + Classification heads). Density map is produced by DSGCNet's own `Density_pred` head (not MoECountNet's), and passed to the MoE as a conditioning signal. MoE auxiliary losses (`moe_aux_losses`, `total_aux`) are merged into the DSGCNet training loop.

### ScaleDecoupledFusion (`fusion_mode=scale_decoupled`)

A newer DSGCNet fusion mode that replaces Neck + DGCN entirely with scale-decoupled parallel streams:

```
VGG body2(s4,256ch) → CNN(dilated d=1,2,3 + FFN + SE) → pool→s8 ─┐
VGG body3(s8,512ch) → GCN(GATv2, spatial-prior k-NN)            ─┤
VGG body4(s16,512ch)→ Transformer(global MHA×2 + 2D PE)         ─┘
                                      ↓
    Cross-Attention: Q←CNN(pooled s8), K/V←GCN(s8)+Transformer(s16)
    + 2D sinusoidal PE + learnable scale-level embeddings
    + zero-init residual gates (gate=0 → identity at training start)
                                      ↓
                              f [B,256,s8,s8]
                                      ↓
    Density_pred → density_out ──→ DensitySEModulation(detach) → f₁
                                      ↓
                         SharedPredictionTrunk
                         ├─ Regression → pred_points
                         └─ Classification → pred_logits
```

**Design rationale**: CNN excels at local texture, GCN at relational reasoning (head proximity), Transformer at global context. Cross-Attention lets local features query relational+global context. CNN is pooled to s8 before K/V to keep memory manageable — full-image SHA evaluation (~12K Q × ~15K KV tokens) uses `F.scaled_dot_product_attention` (flash attention) to avoid materializing the attention matrix.

**Key files**:
- `src/crowdcount/models/scale_decoupled_fusion.py` — `CNNStream`, `GCNStream`, `TransformerStream`, `ScaleDecoupledCrossAttention`, `DensitySEModulation`, `ScaleDecoupledFusion`
- Config: `configs/model/dsgcnet.yaml` → `scale_decoupled_fusion.*`
- Tests: `tests/test_scale_decoupled_fusion.py` (29 synthetic tests)

**Usage**: `uv run python scripts/train.py data.data_root=DATA_ROOT model.fusion_mode=scale_decoupled`

**Memory**: Requires ~6GB VRAM for full-image evaluation with flash attention. On smaller GPUs, reduce `ca_num_heads` or downsample evaluation inputs.

### Plugins (experimental modules for DSGCNet)

Key plugins in `src/crowdcount/plugins/`:
- `moe.py` — ESCA spatial/channel attention + LightMoE (post-GCN micro-expert refinement, used in `gcn_moe` mode)
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
- `configs/model/dsgcnet.yaml` — Model architecture + fusion_mode (gcn, gcn_moe, graph_attn_moe, graph_moe, mamba_moe, mamba_vss_dual, sdd_moe, sa_dgat, deformable_dual, moe, scale_decoupled)
- `configs/data/shha.yaml` — Data settings
- `configs/optimizer/adamw.yaml` — Default: lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4. `adam` optimizer also available.
- Note: `step_lr` is available but not the root default; it was used as an override in the best known run (`scheduler=step_lr scheduler.lr_drop=800`)
- Schedulers (`configs/scheduler/`): `cosine_annealing` (DSGCNet warmup=100 epochs, MoECountNet warmup=5 epochs) and `step_lr` (default lr_drop=3500). Both have `warmup_epochs` and `warmup_start_factor` params.

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
- ScaleDecoupledFusion design spec: `docs/superpowers/specs/2026-05-31-scale-decoupled-fusion-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-31-scale-decoupled-fusion-plan.md`
- Best known run: `outputs/2026-04-25/22-51-51/` (MAE=48.51, MSE=79.87 on SHA). Checkpoint at `checkpoints/best_mae.pth`. Use this as the performance baseline before claiming improvement.
- Diagnostic scripts in `scripts/` are useful for debugging: `analyze_density_generation_quality.py`, `analyze_hard_score_band.py`, `calibrate_counts.py`, `search_threshold.py`, plus `diag_*.py` and `probe_*.py` tools.

## Training Considerations

- MoECountNet has 3 parameter groups: head (lr=1e-4), backbone (lr=1e-5), gate (lr=1e-4)
- MoECountNet uses SparseTop2Gate with Gumbel-Softmax temperature annealing. The gate starts in soft-routing warmup (all experts contribute with soft weights), then transitions to hard Top-2 routing with straight-through gradients. `model.moe.temperature_init`, `temperature_decay`, `warmup_epochs` control the schedule.
- Balance loss (CV² of expert load/importance) decays linearly to 0 over `decay_epochs` after warmup
- The deformable expert's residual gate is initialized to 0 (identity pass-through), so it needs many epochs to "warm up" — expect e2 gate load to start low and gradually increase
- **SharedExpert** now uses `shared_num_blocks` (default 3) residual blocks instead of a single conv. This provides a stronger gradient highway; `shared_scale` is learnable by default (`shared_scale_learnable: true`) so the model can balance shared vs routed contributions
- **Expert `needs_density` dispatch**: Experts declare `needs_density = True` (class attribute) to receive the predicted density map. DensityAdaptiveLocalExpert uses `.detach()` for feed-forward modulation (no gradient feedback to the density head). GlobalDensityExpert concat-fuses density with features (gradient flows through the fusion conv)
- DSGCNet default `clip_max_norm=0.1`; MoECountNet uses `clip_max_norm=5.0`
- MoECountNet uses `mixed_precision.enabled: true` by default; DSGCNet does not
- MoECountNet active development is on branch `exp1`

## MoECountNet Config Defaults

- `configs/moecount_config.yaml`: `epochs: 1500`, `weight_decay: 0.0001`, `use_pml: true` (ProximalMappingLoss; set `false` for BayesianLoss)
- `configs/model/moecount.yaml`: `output_stride: 8`, `backbone.arch: convnext_tiny`, `moe.top_k: 2`, `head.final_activation: softplus`, `head.final_weight_std: 0.01`
- `configs/model/dsgcnet.yaml` (DSGCNet `fusion_mode=moe` path): uses SparseTop2Gate with `moecount_moe.*` config; `shared_scale: 0.5`, `shared_num_blocks: 3`, `shared_scale_learnable: true`
- The default primary density loss is ProximalMappingLoss (`use_pml: true`). When using BayesianLoss (`use_pml: false`), it uses `use_background: true`, `bg_ratio: 0.15`
- Point prediction auxiliary head is enabled by default (`head.use_point_head: true`)
- Sinkhorn OT loss is disabled by default (`moecount_loss.ot.enabled: false`)
- Expert replacement flags (dsgcnet.yaml): all three default to `true` (PointLocalization, OcclusionReasoning, DensityPattern). Set all to `false` for legacy LocalDetail/SpatialRelation/GlobalDensity experts.

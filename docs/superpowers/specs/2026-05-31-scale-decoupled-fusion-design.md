# Scale-Decoupled CNN/GCN/Transformer Fusion Module for DSGCNet

**Date**: 2026-05-31
**Status**: Design Approved
**Target**: DSGCNet — replace Neck + Dual-Stream GCN with scale-decoupled multi-stream fusion

## Motivation

Replace DSGCNet's existing Neck (PA-FPN / SPD-BiFPN) + Dual-Stream GCN pipeline with a
**scale-decoupled multi-stream architecture** where backbone features at different resolutions
are processed by different operator paradigms — CNN at high-res, GCN at mid-res, Transformer at low-res —
then fused via Cross-Attention and modulated by density for point prediction.

**Key insight**: Scale-operator decoupling matches theoretical intuitions:
- **CNN @ s8**: excels at local texture/edge patterns
- **GCN @ s16**: excels at relational reasoning (head proximity, density similarity)
- **Transformer @ s32**: excels at global scene context

The fused feature is modulated by the predicted density map (detached) via SE-style channel attention,
and used as the primary representation for **point prediction** (the main task).

## Design Constraints

- **No up/down sampling** in the three processing streams — native resolutions preserved
- **Cross-Attention Q/K/V lengths may differ** — Q←F_s8, K/V←[F_s16, F_s32], no interpolate needed
- **Zero-init gates** on all residual paths for training stability (existing codebase convention)
- **Parameter budget**: ~4M, roughly neutral vs. Neck(~2M) + DGCN(~3M) = ~5M replaced
- **Detach density** before modulation — feed-forward conditioning, no gradient feedback

## Architecture Overview

```
Backbone (VGG16-BN / ConvNeXt)
├─ C2: [B, 256, s8, s8]     →  CNN Stream        → F_s8  [B, 256, s8, s8]
├─ C3: [B, 512, s16, s16]   →  GCN Stream        → F_s16 [B, 256, s16, s16]
└─ C4: [B, 512, s32, s32]   →  Transformer Stream → F_s32 [B, 256, s32, s32]
                                   ↓
    Cross-Attention: Q←F_s8, K/V←[F_s16, F_s32]   →  f [B, 256, s8, s8]
                                   ↓
    Density Map (detach) → SE Modulation           →  f₁ [B, 256, s8, s8]
                                   ↓
            ├─ Density Head → Density Map
            └─ Point Pred Head → Point Predictions
```

## Component Specification

### 1. CNN Stream (s8, High-Resolution Local Features)

Adapted from `DensityAdaptiveLocalExpert` (remove density modulation, point aux):

```
C2: [B, 256, s8, s8]
  → Multi-scale Dilated Convs (d=1,2,3, groups=16, GELU)
    → per-branch learnable scales → 1×1 fuse → internal residual
  → FFN: GroupNorm → Conv1×1(256→512) → GELU → Conv1×1(512→256) → residual
  → MultiSpectralChannelAttention (num_freqs=4)
  → Conv1×1 output → F_s8: [B, 256, s8, s8]
```

- Reuse code: `DensityAdaptiveLocalExpert` from `experts.py`
- Remove: `density_gate`, `density_gain`, `point_head`, `point_gain`
- Remove: `use_density_modulation` flag — density modulation moved to final SE module
- Params: ~0.4M

### 2. GCN Stream (s16, Mid-Resolution Relational Reasoning)

Reuse pattern from `DensityGCNProcessor` with GATv2Conv, operating at stride-16:

```
C3: [B, 512, s16, s16]
  → Conv1×1: 512→256 = F_s16_raw [B, 256, s16, s16]

Density map (detach) → interpolate to s16 → SpatialPriorDensityGraphBuilder(k=4)
  └─ alpha=1.0 (density similarity), beta=1.0 (spatial distance penalty)
  └─ Edge attribute: exp(-|Δd|)

GATv2Conv ×2:
  └─ Layer1: 256→512 (heads=4, concat=True) + LayerNorm + GELU + dropout=0.1
  └─ Layer2: 512→256 (heads=4, concat=False) + LayerNorm + GELU + dropout=0.1
  └─ residual projection (Linear if dim mismatch) + output

Reshape: [B*H16*W16, 256] → [B, 256, s16, s16] = F_s16
```

- Node count: 196 (14×14 @ s16 for typical input) — 16x fewer than s8 GCN
- Graph construction: O(196²) = ~38K pair distances, negligible
- `spatial_prior=True` is required — prevents 67.8% long-range false edges (lesson from ablation)
- Reuse code: `DensityGCNProcessor`, `SpatialPriorDensityGraphBuilder`, `GATv2Model` from `gcn.py`
- Params: ~0.3M

### 3. Transformer Stream (s32, Low-Resolution Global Context)

Reuse `FeatureTransformerBlock` with global attention:

```
C4: [B, 512, s32, s32]
  → Conv1×1: 512→256 = F_s32_raw [B, 256, s32, s32]
  → Learnable 2D Positional Embedding (sinusoidal init): [1, 256, s32, s32]

FeatureTransformerBlock × 2:
  └─ mode: "global" (7×7=49 tokens, global attn feasible)
  └─ QKV projection: Conv1×1(256→384), 3-way chunk
  └─ num_heads=4, embed_dim=128, head_dim=32
  └─ mlp_ratio=4.0, dropout=0.0
  └─ zero-init attn_gate + mlp_gate

→ F_s32: [B, 256, s32, s32]
```

- Token count: 49 (7×7 @ s32 for typical input) — global self-attention costs only 2401 pairs
- Reuse code: `FeatureTransformerBlock` from `gcn.py`
- Params: ~2M (2 blocks)

### 4. Cross-Attention Fusion (Core Novel Module)

**New module** — `ScaleDecoupledCrossAttention`:

```
输入:
  F_s8:  [B, 256, s8, s8]      → Q,  N_q  ≈ 784
  F_s16: [B, 256, s16, s16]    → K/V, N_kv ≈ 196+49=245
  F_s32: [B, 256, s32, s32]    → K/V

1. Project to unified dimension D=256:
   Q_proj = Conv1×1(256→256), flatten → [B, N_s8, 256]
   K_proj = Conv1×1(256→256), flatten → [B, N_s16+N_s32, 256]
   V_proj = Conv1×1(256→256), flatten → [B, N_s16+N_s32, 256]
   Q = LayerNorm(Q_proj)
   K = LayerNorm(K_proj)
   V = LayerNorm(V_proj)

2. Position Encoding (2D spatial PE + scale-level embedding):
   a) 2D Spatial PE (sinusoidal):
      pos_s8  = sinusoidal_2D(s8, s8,  D)  →  [1, N_s8,  256]
      pos_s16 = sinusoidal_2D(s16, s16, D)  →  [1, N_s16, 256]
      pos_s32 = sinusoidal_2D(s32, s32, D)  →  [1, N_s32, 256]
      Q += pos_s8
      K += cat([pos_s16, pos_s32], dim=1)
   b) Scale-Level Embeddings (learnable):
      K += self.scale_embed[0]  ← broadcast over N_s16 tokens
      K += self.scale_embed[1]  ← broadcast over N_s32 tokens
      self.scale_embed = nn.Embedding(2, D)

   Without 2D spatial PE, the attention degenerates to channel-only mixing
   — Q cannot establish geometric correspondence with K/V across scales.

3. Multi-Head Cross-Attention (h=4, head_dim=64, batch_first=True):
   attn_out, _ = nn.MultiheadAttention(
       embed_dim=256, num_heads=4, batch_first=True
   )(Q, K, V)   ← is_causal=False (default, correct for cross-attn)
   attn_out = self.out_proj(attn_out)  →  [B, N_q, 256]

   # Residual around Q (not around attn_out):
   # gate=0 → f_attn = Q (identity), gate>0 → gradually adds attention
   f_attn = Q + self.attn_gate.tanh() * attn_out

4. FFN + Residual (same identity-preserving pattern):
   normed = LayerNorm(f_attn)
   f = f_attn + self.mlp_gate.tanh() * MLP(normed)
     └─ MLP: Linear(256→512) → GELU → Linear(512→256)

5. Reshape:
   f → [B, 256, s8, s8]

All gates zero-initialized:
  attn_gate = 0  → f_attn = Q (identity: attention bypassed)
  mlp_gate  = 0  → f = f_attn (identity: FFN bypassed)
  At training step 0: f = Q_proj (input pass-through, stable start)
```

- Attention pairs: 784 × 245 ≈ 192K — moderate, well within limits
- **No interpolate needed** — Cross-Attention naturally supports N_q ≠ N_kv
- Reuse pattern from: `depth_cross_attention.py`: `DepthCrossAttentionFusion`
- Params: ~1M

### 5. Density Modulation (SE-style Channel Attention)

```
输入:
  f:       [B, 256, s8, s8]
  d_pred:  [B, 1, H_d, W_d]  ← from Density Head (detached)

1. Density encoding:
   d_pred_safe = d_pred.detach() if d_pred.requires_grad else d_pred
   d = interpolate(d_pred_safe, size=s8)
   d_feat = Conv(1→64) → BN → GELU  →  [B, 64, s8, s8]

2. SE Channel Attention:
   gap = AdaptiveAvgPool2d(1)(d_feat)     →  [B, 64, 1, 1]
   fc  = FC(64→64) → ReLU → FC(64→256) → Sigmoid
   channel_scale = fc(gap)                →  [B, 256, 1, 1]

3. Zero-init residual modulation:
   f₁ = f * (1.0 + gain.tanh() * (channel_scale - 0.5))
   └─ gain initialized to 0 → f₁ = f at training start
```

- Reuse code: `SE` class from `experts.py`, density encoding from `DensityAdaptiveLocalExpert`
- Params: ~0.01M (negligible)

### 6. Prediction Heads (Reuse Existing)

- **Density Head**: Reuse `DensityHead` from `moecount/head.py` (Softplus activation)
- **Point Prediction Head**: Reuse `DSGCAnchorPointHead` from `moecount/head.py` (anchor-based)
- Or alternatively: reuse DSGCNet's existing `Density_pred` + point head from `head.py`

## Integration into DSGCNet

### Modified `DSGCNet.forward()` (pseudocode, using Option B fallback strategy)

```python
def forward(self, images, targets=None):
    # Backbone
    features = self.backbone(images)  # C2, C3, C4

    # Scale-decoupled streams (replace Neck)
    F_s8  = self.cnn_stream(features["c2"])                    # CNN @ s8
    F_s16 = self.gcn_stream(features["c3"], density=None)       # GCN @ s16 (feature-graph fallback)
    F_s32 = self.transformer_stream(features["c4"])             # Transformer @ s32

    # Cross-Attention fusion (replace DGCN)
    f = self.cross_attention(F_s8, F_s16, F_s32)

    # Density + point prediction
    density = self.density_head(f)
    f_mod = self.density_modulation(f, density)
    point_preds = self.point_head(f_mod)

    return {"density": density, "point": point_preds}
```

The GCN stream uses `FeatureGraphBuilder` (cosine similarity) when density is `None`,
and `DensityGraphBuilder` / `SpatialPriorDensityGraphBuilder` when density is provided.

### GCN Stream's Density Dependency

The GCN stream needs a density map to build the k-NN graph. In the first forward pass,
density is not yet available. Two strategies:

**A. Two-pass forward (recommended)**:
```python
# Pass 1: dummy density for GCN, get rough density
density_init = torch.zeros(...)
F_s16 = self.gcn_stream(c3, density=density_init)
f = self.cross_attention(F_s8, F_s16, F_s32)
density = self.density_head(f).detach()

# Pass 2: re-run GCN with real density
F_s16 = self.gcn_stream(c3, density=density)
f = self.cross_attention(F_s8, F_s16, F_s32)
density = self.density_head(f)
f_mod = self.density_modulation(f, density)
point_preds = self.point_head(f_mod)
```

**B. Use feature-similarity graph as fallback** (when density is None):
```python
# GCN stream
if density is not None:
    graph = DensityGraphBuilder(k=4).build(density)
else:
    graph = FeatureGraphBuilder(k=4).build(features)
```

Option B is simpler and matches the dual-graph pattern in existing `CrossStreamGCNProcessor`.

**Decision**: Use option B for simplicity. The GCN stream uses `FeatureGraphBuilder` (cosine similarity) when no density is available, and `DensityGraphBuilder` when density is provided. This also makes the module usable without a density head at all.

### Config Structure (Hydra)

```yaml
model:
  scale_decoupled_fusion:
    enabled: true
    unified_dim: 256

    cnn_stream:
      dilations: [1, 2, 3]
      groups: 16
      ffn_expansion: 2
      use_multi_spectral_se: true
      ms_num_freqs: 4

    gcn_stream:
      k: 4
      graph_type: "spatial_prior"  # spatial_prior | density | feature
      spatial_alpha: 1.0
      spatial_beta: 1.0
      conv_type: "gatv2"           # gatv2 | gcn | eca
      hidden_channels: 512
      heads: 4
      dropout: 0.1

    transformer_stream:
      num_blocks: 2
      num_heads: 4
      embed_dim: 128
      mlp_ratio: 4.0
      mode: "global"               # global (49 tokens, feasible)
      dropout: 0.0
      pe_type: "learnable_2d"      # learnable_2d | sinusoidal_2d

    cross_attention:
      num_heads: 4
      head_dim: 64
      dropout: 0.1
      ff_expansion: 2

    density_modulation:
      density_hidden: 64
      se_reduction: 4
```

## Computational Cost Estimate

| Component | Nodes/Tokens | Attention Pairs | Estimated Params |
|-----------|-------------|-----------------|-----------------|
| CNN @ s8 | 784 (spatial) | N/A (conv) | ~0.4M |
| GCN @ s16 | 196 | ~38K (graph edges) | ~0.3M |
| Transformer @ s32 | 49 | 2.4K (self-attn) | ~2M |
| Cross-Attention | Q:784, KV:245 | 192K | ~1M |
| SE Modulation | N/A | N/A | ~0.01M |
| **Total** | | | **~3.7M** |

Comparison: replaced Neck(~2M) + DGCN(~3M) = ~5M → **net parameter reduction of ~1.3M**.

GCN and Transformer compute is drastically cheaper than existing DGCN at stride-8:
- GCN: 196² vs 784² → **16× fewer graph edges**
- Transformer: 49² = 2.4K self-attn pairs → **negligible**

Main compute: CNN @ s8 + Cross-Attention (192K pairs) — both moderate.

## Implementation Plan (File-Level)

1. **New file**: `src/crowdcount/models/scale_decoupled_fusion.py`
   - `CNNStream`: adapted `DensityAdaptiveLocalExpert` (no density mod, no point aux)
   - `GCNStream`: wraps `DensityGCNProcessor` / `GATv2Model` at s16
   - `TransformerStream`: wraps `FeatureTransformerBlock` × N at s32
   - `ScaleDecoupledCrossAttention`: new Cross-Attention with scale-level embeddings
   - `DensitySEModulation`: SE-style density→channel modulation
   - `ScaleDecoupledFusion`: top-level module composing all 5 components

2. **Modified file**: `src/crowdcount/models/dsgcnet.py`
   - Add `fusion_mode: "scale_decoupled"` branch
   - Replace Neck + DGCN with `ScaleDecoupledFusion`
   - Conditional two-pass or fallback strategy for GCN

3. **Modified file**: `configs/model/dsgcnet.yaml`
   - Add `scale_decoupled_fusion` config section

4. **Tests**: `tests/test_scale_decoupled_fusion.py`
   - Synthetic tensor tests, no GPU/data required
   - Test Cross-Attention with N_q ≠ N_kv
   - Test GCN with/without density fallback
   - Test SE modulation zero-init → identity at step 0

## Edge Cases & Pitfalls

- **GCN zero-density input**: When all density values are zero (background image), k-NN graph has uniform distances → GCN degenerates to uniform message passing. Use feature graph fallback.
- **s32 resolution too small**: If input image is small (e.g., 128×128), s32 = 4×4 = 16 tokens. Self-attention on 16 tokens is still valid but may lose meaningful structure. Consider switching to s16 Transformer for very small inputs.
- **Density head bootstrap**: First epoch density predictions are garbage. The SE modulation starts at zero-gain (identity), so it naturally ignores bad density early in training.
- **Cross-Attention memory**: 192K attention pairs is per-image. With batch_size=8, that's ~1.5M pairs — well within GPU memory limits.

## References

- Existing `DensityAdaptiveLocalExpert` — [experts.py:122](src/crowdcount/models/moecount/experts.py#L122)
- Existing `DensityGCNProcessor` / `GATv2Model` — [gcn.py:942](src/crowdcount/models/gcn.py#L942) / [gcn.py:666](src/crowdcount/models/gcn.py#L666)
- Existing `FeatureTransformerBlock` — [gcn.py:729](src/crowdcount/models/gcn.py#L729)
- Existing `SE` class — [experts.py:15](src/crowdcount/models/moecount/experts.py#L15)
- Existing `DepthCrossAttentionFusion` (cross-attn template) — [depth_cross_attention.py](src/crowdcount/plugins/depth_cross_attention.py)
- Existing `SpatialPriorDensityGraphBuilder` — [gcn.py:96](src/crowdcount/models/gcn.py#L96)
- Density-modulation pattern (zero-init gate + detach) — [experts.py:282-284](src/crowdcount/models/moecount/experts.py#L282-L284)

# Scale-Decoupled CNN/GCN/Transformer Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DSGCNet's Neck + DGCN with scale-decoupled CNN/GCN/Transformer streams fused by Cross-Attention, density-modulated for point prediction.

**Architecture:** Backbone C2/C3/C4 → CNN@s8 / GCN@s16 / Transformer@s32 → Cross-Attn (Q←s8, K/V←s16+s32) → SE density modulation → Density Head + Point Head. All gates zero-initialized for identity startup.

**Tech Stack:** PyTorch, torch_geometric (GATv2Conv), Hydra/OmegaConf, loguru logging

**Spec:** `docs/superpowers/specs/2026-05-31-scale-decoupled-fusion-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/crowdcount/models/scale_decoupled_fusion.py` | All 6 sub-modules + top-level composite |
| Modify | `src/crowdcount/models/dsgcnet.py` | Add `fusion_mode="scale_decoupled"` branch |
| Modify | `configs/model/dsgcnet.yaml` | Add `scale_decoupled_fusion` config section |
| Create | `tests/test_scale_decoupled_fusion.py` | Synthetic tensor tests |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `src/crowdcount/models/scale_decoupled_fusion.py`
- Create: `tests/test_scale_decoupled_fusion.py`

- [ ] **Step 1: Create empty module file with imports**

Write to `src/crowdcount/models/scale_decoupled_fusion.py`:
```python
"""Scale-Decoupled CNN/GCN/Transformer Fusion for DSGCNet.

Replaces the Neck + Dual-Stream GCN pipeline with three parallel streams
at native backbone resolutions (s8/s16/s32), fused via Cross-Attention
and modulated by SE-style density channel attention.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn
```

- [ ] **Step 2: Create empty test file**

Write to `tests/test_scale_decoupled_fusion.py`:
```python
"""Tests for scale_decoupled_fusion — synthetic tensors, no GPU/data."""
from __future__ import annotations
import pytest
import torch
from crowdcount.models.scale_decoupled_fusion import (
    CNNStream, GCNStream, TransformerStream,
    ScaleDecoupledCrossAttention, DensitySEModulation,
    ScaleDecoupledFusion, sinusoidal_2d_pe,
)
```

- [ ] **Step 3: Verify import errors (TDD start)**

```bash
uv run python -c "from crowdcount.models.scale_decoupled_fusion import CNNStream; print('ok')"
```
Expected: ImportError (classes not defined yet).

- [ ] **Step 4: Commit**

```bash
git add tests/test_scale_decoupled_fusion.py src/crowdcount/models/scale_decoupled_fusion.py
git commit -m "feat: scaffold scale_decoupled_fusion module and test file"
```

---

### Task 2: Sinusoidal 2D Position Encoding

**Files:**
- Modify: `src/crowdcount/models/scale_decoupled_fusion.py` (add fn)
- Modify: `tests/test_scale_decoupled_fusion.py` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestSinusoidal2DPE:
    def test_output_shape(self):
        pe = sinusoidal_2d_pe(7, 7, 256)
        assert pe.shape == (1, 49, 256)

    def test_rectangular(self):
        pe = sinusoidal_2d_pe(14, 14, 256)
        assert pe.shape == (1, 196, 256)

    def test_value_range(self):
        pe = sinusoidal_2d_pe(3, 3, 64)
        assert pe.min() >= -1.0 and pe.max() <= 1.0

    def test_deterministic(self):
        pe1 = sinusoidal_2d_pe(5, 5, 128)
        pe2 = sinusoidal_2d_pe(5, 5, 128)
        assert torch.equal(pe1, pe2)
```

- [ ] **Step 2: Verify test fails**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestSinusoidal2DPE -v
```
Expected: FAIL.

- [ ] **Step 3: Implement sinusoidal_2d_pe**

```python
def sinusoidal_2d_pe(h: int, w: int, dim: int) -> torch.Tensor:
    """Sinusoidal 2D positional encoding. Returns [1, h*w, dim]."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    half_dim = dim // 2
    div_term = torch.exp(
        torch.arange(0, half_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / half_dim)
    )
    pos_y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    pos_x = torch.arange(w, dtype=torch.float32).unsqueeze(0)
    pe_y = torch.zeros(h, w, half_dim)
    pe_x = torch.zeros(h, w, half_dim)
    pe_y[:, :, 0::2] = torch.sin(pos_y * div_term)
    pe_y[:, :, 1::2] = torch.cos(pos_y * div_term)
    pe_x[:, :, 0::2] = torch.sin(pos_x * div_term)
    pe_x[:, :, 1::2] = torch.cos(pos_x * div_term)
    pe = torch.cat([pe_y, pe_x], dim=-1)  # [h, w, dim]
    return pe.reshape(1, h * w, dim)
```

- [ ] **Step 4: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestSinusoidal2DPE -v
git add -A && git commit -m "feat: add sinusoidal_2d_pe for cross-attention PE"
```

---

### Task 3: CNNStream (s8, High-Resolution CNN)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestCNNStream:
    @pytest.fixture
    def stream(self):
        return CNNStream(in_channels=256)

    def test_output_shape(self, stream):
        x = torch.randn(2, 256, 28, 28)
        out = stream(x)
        assert out.shape == (2, 256, 28, 28)

    def test_no_nan(self, stream):
        x = torch.randn(4, 256, 16, 16)
        stream.eval()
        with torch.no_grad():
            out = stream(x)
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: Verify FAIL, then implement**

CNNStream adapts `DensityAdaptiveLocalExpert` without density modulation or point aux:
- Multi-scale dilated convs (d=1,2,3, groups=16) → 1×1 fuse → internal residual
- FFN: GN → Conv1×1(256→512) → GELU → Conv1×1(512→256) → residual
- MultiSpectralChannelAttention or SE
- Conv1×1 output

Reuse: `MultiSpectralChannelAttention` / `SE` from `crowdcount.models.moecount.experts`.

- [ ] **Step 3: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestCNNStream -v
git add -A && git commit -m "feat: add CNNStream — multi-scale dilated conv @ s8"
```

---

### Task 4: GCNStream (s16, Mid-Resolution Graph Reasoning)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestGCNStream:
    @pytest.fixture
    def stream(self):
        return GCNStream(in_channels=512, out_channels=256, k=4)

    def test_output_shape_no_density(self, stream):
        x = torch.randn(2, 512, 14, 14)
        out = stream(x)  # density=None → feature graph fallback
        assert out.shape == (2, 256, 14, 14)

    def test_with_density(self, stream):
        x = torch.randn(2, 512, 14, 14)
        density = torch.rand(2, 1, 14, 14)
        out = stream(x, density=density)
        assert out.shape == (2, 256, 14, 14)
```

- [ ] **Step 2: Verify FAIL, then implement**

GCNStream wraps:
- Input: Conv1×1(512→256) + GN + ReLU
- Graph: `SpatialPriorDensityGraphBuilder(k=4)` when density given, `FeatureGraphBuilder(k=4)` fallback
- GCN: `GATv2Model` from `gcn.py` (256→512→256, heads=4, dropout=0.1)
- Output: reshape from nodes back to [B, C, H, W]

Reuse: `SpatialPriorDensityGraphBuilder`, `FeatureGraphBuilder`, `GATv2Model` from `crowdcount.models.gcn`.

- [ ] **Step 3: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestGCNStream -v
git add -A && git commit -m "feat: add GCNStream — density/feature graph + GATv2 @ s16"
```

---

### Task 5: TransformerStream (s32, Global Context)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestTransformerStream:
    @pytest.fixture
    def stream(self):
        return TransformerStream(in_channels=512, out_channels=256, num_blocks=2)

    def test_output_shape(self, stream):
        x = torch.randn(2, 512, 7, 7)
        out = stream(x)
        assert out.shape == (2, 256, 7, 7)

    def test_small_input(self, stream):
        x = torch.randn(1, 512, 4, 4)
        out = stream(x)
        assert out.shape == (1, 256, 4, 4)
```

- [ ] **Step 2: Verify FAIL, then implement**

TransformerStream wraps:
- Input: Conv1×1(512→256) + GN + ReLU
- PE: Learnable 2D PE (sinusoidal init, interpolates for different sizes)
- Blocks: `FeatureTransformerBlock` × 2 from `gcn.py`, mode="global"
- Params: embed_dim=128, num_heads=4, mlp_ratio=4.0, gate_init=0.0

Reuse: `FeatureTransformerBlock` from `crowdcount.models.gcn`.

- [ ] **Step 3: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestTransformerStream -v
git add -A && git commit -m "feat: add TransformerStream — global self-attn Transformer @ s32"
```

---

### Task 6: ScaleDecoupledCrossAttention (Core Fusion)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestScaleDecoupledCrossAttention:
    @pytest.fixture
    def ca(self):
        return ScaleDecoupledCrossAttention(dim=256, num_heads=4)

    def test_output_shape(self, ca):
        f8 = torch.randn(2, 256, 28, 28)
        f16 = torch.randn(2, 256, 14, 14)
        f32 = torch.randn(2, 256, 7, 7)
        out = ca(f8, f16, f32)
        assert out.shape == (2, 256, 28, 28)

    def test_nq_not_equal_nkv(self, ca):
        f8 = torch.randn(1, 256, 28, 28)    # N_q = 784
        f16 = torch.randn(1, 256, 14, 14)   # N_kv part
        f32 = torch.randn(1, 256, 7, 7)     # N_kv part
        out = ca(f8, f16, f32)
        assert out.shape == (1, 256, 28, 28)  # N_kv=245 ≠ N_q=784

    def test_identity_at_init(self, ca):
        """gate=0 → output ≈ Q (pass-through)."""
        f8 = torch.randn(2, 256, 16, 16)
        f16 = torch.randn(2, 256, 8, 8)
        f32 = torch.randn(2, 256, 4, 4)
        ca.eval()
        with torch.no_grad():
            out = ca(f8, f16, f32)
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: Verify FAIL, then implement**

ScaleDecoupledCrossAttention design:
1. Q ← Conv1×1(F_s8)→flatten→LayerNorm + sinusoidal_2d_pe(s8)
2. K ← Conv1×1(F_s16|F_s32)→flatten→LayerNorm + sinusoidal_2d_pe(s16|s32) + scale_embed(2, dim)
3. V ← same as K projection, separate weight
4. Multi-head dot-product attention (h=4, head_dim=64)
5. Residual around Q: `f_attn = Q + attn_gate.tanh() * out_proj(attn_out)`
6. FFN: `f = f_attn + mlp_gate.tanh() * MLP(LayerNorm(f_attn))`
7. Reshape to [B, 256, s8, s8]
8. All gates zero-initialized

- [ ] **Step 3: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestScaleDecoupledCrossAttention -v
git add -A && git commit -m "feat: add ScaleDecoupledCrossAttention — Q←s8, K/V←s16+s32"
```

---

### Task 7: DensitySEModulation (SE-style Modulation)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing test**

```python
class TestDensitySEModulation:
    @pytest.fixture
    def mod(self):
        return DensitySEModulation(channels=256, density_hidden=64, reduction=4)

    def test_output_shape(self, mod):
        f = torch.randn(2, 256, 28, 28)
        density = torch.rand(2, 1, 28, 28)
        out = mod(f, density)
        assert out.shape == (2, 256, 28, 28)

    def test_identity_at_init(self, mod):
        """Zero-init gain → output = input."""
        f = torch.randn(2, 256, 16, 16)
        density = torch.rand(2, 1, 16, 16)
        mod.eval()
        with torch.no_grad():
            out = mod(f, density)
        torch.testing.assert_close(out, f)

    def test_density_interpolation(self, mod):
        f = torch.randn(1, 256, 28, 28)
        density = torch.rand(1, 1, 14, 14)  # different res
        out = mod(f, density)
        assert out.shape == (1, 256, 28, 28)
```

- [ ] **Step 2: Verify FAIL, then implement**

DensitySEModulation design:
1. Safe detach: `d = density.detach() if density.requires_grad else density`
2. Interpolate density to feature resolution
3. Encode: Conv(1→64)→BN→GELU
4. SE: GAP → FC(64→C/4)→ReLU→FC(C/4→C)→Sigmoid (last layer zero-init)
5. `f * (1 + gain.tanh() * (channel_scale - 0.5))`, gain=0→identity

- [ ] **Step 3: Run tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py::TestDensitySEModulation -v
git add -A && git commit -m "feat: add DensitySEModulation — SE-style density channel attention"
```

---

### Task 8: ScaleDecoupledFusion (Top-Level Composite)

**Files:** Modify `scale_decoupled_fusion.py` (add class), `tests/` (add tests)

- [ ] **Step 1: Write failing end-to-end test**

```python
class TestScaleDecoupledFusion:
    @pytest.fixture
    def fusion(self):
        return ScaleDecoupledFusion(
            c2_channels=256, c3_channels=512, c4_channels=512,
            unified_dim=256,
        )

    def test_full_forward(self, fusion):
        c2 = torch.randn(2, 256, 28, 28)
        c3 = torch.randn(2, 512, 14, 14)
        c4 = torch.randn(2, 512, 7, 7)
        f, aux = fusion(c2, c3, c4)
        assert f.shape == (2, 256, 28, 28)
        assert isinstance(aux, dict)

    def test_with_modulation(self, fusion):
        c2 = torch.randn(1, 256, 28, 28)
        c3 = torch.randn(1, 512, 14, 14)
        c4 = torch.randn(1, 512, 7, 7)
        f, _ = fusion(c2, c3, c4)
        density = torch.rand(1, 1, 28, 28)
        f_mod = fusion.density_modulation(f, density)
        assert f_mod.shape == (1, 256, 28, 28)

    def test_varying_sizes(self, fusion):
        c2 = torch.randn(1, 256, 32, 32)
        c3 = torch.randn(1, 512, 16, 16)
        c4 = torch.randn(1, 512, 8, 8)
        f, _ = fusion(c2, c3, c4)
        assert f.shape == (1, 256, 32, 32)
```

- [ ] **Step 2: Verify FAIL, then implement**

Compose CNNStream + GCNStream + TransformerStream + ScaleDecoupledCrossAttention + DensitySEModulation. Forward: C2/C3/C4 → three streams → cross-attention → fused features. Density modulation applied separately via `fusion.density_modulation(f, density)`.

- [ ] **Step 3: Run all tests → PASS, then commit**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py -v
git add -A && git commit -m "feat: add ScaleDecoupledFusion — top-level composite module"
```

---

### Task 9: Integrate into DSGCNet

**Files:** Modify `src/crowdcount/models/dsgcnet.py`

Steps:
1. Add `from crowdcount.models.scale_decoupled_fusion import ScaleDecoupledFusion` to imports
2. Add `"scale_decoupled"` to valid `fusion_mode` set (line ~501)
3. Add `self.use_scale_decoupled = fusion_mode == "scale_decoupled"` and `self.scale_decoupled_fusion = None` attributes
4. After existing MoE branch, add `elif self.use_scale_decoupled:` that constructs `ScaleDecoupledFusion` from config, nullifies other fusion components
5. Modify `forward()`: add branch where backbone C2/C3/C4 are routed through `self.scale_decoupled_fusion` instead of neck+DGCN
6. Wire density modulation: `features_for_points = self.scale_decoupled_fusion.density_modulation(f, density_out)`

Key forward integration pseudocode:
```python
if self.use_scale_decoupled:
    c2 = features_list[1]  # stride-8
    c3 = features_list[2]  # stride-16
    c4 = features_list[3]  # stride-32
    features_fl, _ = self.scale_decoupled_fusion(c2, c3, c4)
    density_out = self.density_pred(features_fl)
    features_for_points = self.scale_decoupled_fusion.density_modulation(
        features_fl, density_out
    )
```

- [ ] **Smoke test:**

```bash
uv run python -c "
from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.backbone import BackboneBase_VGG
import crowdcount.models.vgg_ as vgg_models
import torch
bb = BackboneBase_VGG(vgg_models.vgg16_bn(pretrained=False), 256, 'vgg16_bn', return_interm_layers=True)
m = DSGCnet(bb, fusion_mode='scale_decoupled')
m.eval()
with torch.no_grad():
    out = m(torch.randn(1,3,224,224))
print('Forward OK, keys:', list(out.keys()))
"
```
Expected: Forward completes without error.

- [ ] **Commit**

```bash
git add src/crowdcount/models/dsgcnet.py
git commit -m "feat: integrate ScaleDecoupledFusion into DSGCNet (fusion_mode=scale_decoupled)"
```

---

### Task 10: Config

**Files:** Modify `configs/model/dsgcnet.yaml`

Add `scale_decoupled_fusion` config block (after `moecount_moe` section):

```yaml
scale_decoupled_fusion:
  enabled: true
  unified_dim: 256
  cnn_dilations: [1, 2, 3]
  cnn_groups: 16
  cnn_ffn_expansion: 2
  cnn_use_multi_spectral_se: true
  gcn_k: 4
  gcn_spatial_alpha: 1.0
  gcn_spatial_beta: 1.0
  gcn_hidden_channels: 512
  gcn_heads: 4
  gcn_dropout: 0.1
  trans_num_blocks: 2
  trans_num_heads: 4
  trans_embed_dim: 128
  trans_mlp_ratio: 4.0
  ca_num_heads: 4
  ca_dropout: 0.1
  ca_ff_expansion: 2
  dm_density_hidden: 64
  dm_reduction: 4
```

Verify: `uv run python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/model/dsgcnet.yaml'); print(list(cfg.scale_decoupled_fusion.keys()))"`

- [ ] **Commit**

```bash
git add configs/model/dsgcnet.yaml
git commit -m "feat: add scale_decoupled_fusion config to dsgcnet.yaml"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run all new tests**

```bash
uv run pytest tests/test_scale_decoupled_fusion.py -v
```

- [ ] **Step 2: Run existing tests for regressions**

```bash
uv run pytest tests/ -v --timeout=60
```

- [ ] **Step 3: Import sanity**

```bash
uv run python -c "
from crowdcount.models.scale_decoupled_fusion import (
    CNNStream, GCNStream, TransformerStream,
    ScaleDecoupledCrossAttention, DensitySEModulation,
    ScaleDecoupledFusion, sinusoidal_2d_pe,
)
from crowdcount.models.dsgcnet import DSGCnet
print('All imports OK')
"
```

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore: final verification for scale_decoupled_fusion"
```

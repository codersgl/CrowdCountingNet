---
name: dl-training-debug
description: >
  Systematic debugging of frozen or collapsed metrics in deep learning training
  (especially crowd counting / density prediction models). Use this skill when
  evaluation metrics are completely frozen across epochs, when all model
  predictions are zero or nearly zero, when training loss varies but eval never
  changes, or when switching from training to evaluation causes all predictions
  to collapse. Covers BatchNorm train/eval discrepancy, feature activation
  drift, Bayesian loss background dominance, gradient clipping issues, and
  expert/routing module collapse in MoE architectures.
---

# Deep Learning Training Debugging

## Overview

This skill provides a systematic, layered diagnostic approach for debugging
frozen or collapsed training metrics in deep learning models—particularly
density prediction and crowd counting architectures. The workflow is organized
by depth: start from the output and trace backward through the model to find
the exact layer where things break.

## Diagnostic Workflow

### Layer 0: Verify the symptom

Before investigating the model, confirm the symptom is real:

1. **Run the model on multiple different inputs** and check whether predictions
   vary at all. If every input gives the exact same output, the model produces
   constant predictions. If predictions are all exactly zero (or exactly the
   same value), the issue is likely a normalization layer collapse.

2. **Load checkpoints from different epochs** and eval them independently. If
   epoch 0 and epoch N give the exact same MAE to 15 decimal places, the model
   produces identical output regardless of weight changes—pointing to a
   train/eval discrepancy in normalization layers.

3. **Check if training loss changes but eval doesn't.** A changing training
   loss with frozen eval strongly suggests a mismatch between training-mode and
   eval-mode computations (e.g., BatchNorm running stats).

### Layer 1: Check normalization layers

BatchNorm is the single most common cause of train/eval discrepancy:

1. **Audit every `BatchNorm2d`** in the model. In architectures that include
   feature pyramids (FPN), MoE routing, or dilated context branches, BatchNorm
   running stats can drift to extreme values during training (e.g., running
   mean 0→60, running var 1→339 in 48 epochs). In eval mode, these extreme
   stats normalize all features to near-zero.

2. **Replace `BatchNorm2d` with `GroupNorm`** for any layer where the
   batch-independent normalization is acceptable. The pattern:
   ```python
   # Before
   nn.BatchNorm2d(channels)
   # After
   nn.GroupNorm(32, channels)  # 32 groups works well for 256 channels
   ```

3. **Check LayerNorm-based backbones** (ConvNeXt, ViT, etc.). These use
   LayerNorm which computes per-sample statistics and does not have train/eval
   discrepancy. If the backbone uses LayerNorm, the problem is downstream.

4. **Add output normalization** at each major stage boundary (neck output, MoE
   output, etc.) to prevent feature statistics from drifting:
   ```python
   self.output_norm = nn.GroupNorm(32, channels)
   # In forward: return self.output_norm(result)
   ```

### Layer 2: Trace intermediate activations

Break the model forward pass into steps and inspect each one:

1. **Check raw output values** (density maps, logits) rather than just the
   final metric. A model might have changing MAE but still produce near-zero
   density values (e.g., `softplus(-61.9) ≈ 0`).

2. **Trace feature statistics layer by layer:**
   ```python
   # Backbone → Neck → MoE Stem → Each Expert → Fused MoE → Pre-activation → Output
   for name, tensor in [("neck", neck_out), ("stem", stem_out),
                        ("expert0", e0), ("fused", fused),
                        ("pre_act", pre_softplus)]:
       print(f"{name}: mean={tensor.mean():.4f} std={tensor.std():.4f} "
             f"min={tensor.min():.4f} max={tensor.max():.4f}")
   ```

3. **Compare trained vs. untrained model** at every intermediate layer. This
   reveals which stage caused the collapse. In our case, the neck output was
   fine (mean≈0, std≈1) but the MoE fused features grew 6.5x during training
   (mean 0.11→0.71), overwhelming the density head.

4. **Check activation functions' operating points.** For softplus with bias
   `b`: `softplus(b) ≈ 0` when `b ≪ 0`, and `softplus(b) ≈ b` when `b ≫ 0`.
   If the pre-activation is extremely negative (−60 instead of −4), the output
   collapses to zero. Check the bias value and the pre-activation distribution.

### Layer 3: Expert and routing modules (MoE-specific)

For Mixture-of-Experts architectures:

1. **Experts without normalization will drift.** Conv→ReLU experts with no
   normalization can amplify or suppress their outputs arbitrarily during
   training. Each expert needs GroupNorm after every convolution:
   ```python
   nn.Sequential(
       nn.Conv2d(c, c, 3, padding=1),
       nn.GroupNorm(32, c),  # ← critical
       nn.ReLU(inplace=True),
   )
   ```

2. **Add output normalization after MoE fusion** so the weighted sum of expert
   outputs stays at a controlled scale regardless of expert weight changes:
   ```python
   fused = (expert_outputs * route_weights).sum(dim=1)
   fused = self.output_norm(fused)  # prevent drift into density head
   ```

3. **Check gate load balance.** If one expert has 100% load and others 0%,
   the gate has collapsed. This is often a consequence of upstream gradient
   issues, not a gate bug. Fix the gradient flow first.

### Layer 4: Loss function analysis

1. **Check for background/auxiliary term dominance.** In Bayesian loss for
   crowd counting, a background "virtual point" (at `bg_ratio × max(H,W)`)
   dominates the posterior for pixels far from any person. With 3 random points
   in a 16×16 density grid, 96% of pixels have background posterior > 0.5.
   This pushes ALL predictions toward zero.

2. **Quantify background dominance:**
   ```python
   bg_posterior_mean = posterior[:, -1].mean()  # should be ≪ 0.5 for healthy training
   point_posterior_mean = posterior[:, :N].sum(dim=1).mean()
   ```
   If background posterior mean > 0.8, the background term will dominate
   gradients and push predictions to zero.

3. **Compute per-pixel gradient components.** A positive mean gradient on
   density pushes predictions DOWN. If background posterior dominates, the
   gradient is always positive regardless of whether predictions are too low:
   ```
   net_gradient = -point_posterior_mean + bg_posterior_mean
   # If > 0: density pushed DOWN (toward zero)
   # If < 0: density pushed UP (toward correct values)
   ```

4. **Fixes for background dominance:**
   - Set `use_background=False` during early training epochs
   - Reduce `bg_ratio` (e.g., 0.15→0.05 or 0.02)
   - Increase `initial_density` so expected[p] is closer to 1.0 from the start

### Layer 5: Gradient flow

1. **Check gradient clipping.** `clip_max_norm=0.1` is extremely aggressive for
   a 28M-param model. The unclipped norm is typically 10–50, so gradients are
   scaled down 100–500x. Effective LR becomes ~1e-6—barely above zero. Use
   `clip_max_norm=5.0` as a starting point.

2. **Measure gradient norms per parameter group** to find bottlenecks:
   ```python
   total_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
   head_norm = nn.utils.clip_grad_norm_(head_params, float("inf"))
   backbone_norm = nn.utils.clip_grad_norm_(backbone_params, float("inf"))
   ```

3. **Check final layer initialization scale.** `final_weight_std=1e-4` (nearly
   zero weights) creates a gradient bottleneck—the softplus gradient at
   `softplus_inverse(0.05) ≈ -2.97` is `sigmoid(-2.97) ≈ 0.049`, attenuating
   gradients 20x. Use `final_weight_std=1.0` to let gradients flow.

## Common Root Cause Patterns

| Symptom | Most Likely Cause | Fix |
|---------|------------------|-----|
| Eval MAE frozen at exact same value (15 decimal places) | BatchNorm running stats explosion | Replace BN with GroupNorm |
| Predictions all zero after training but non-zero initially | Expert activation drift without normalization | Add GroupNorm to every expert + MoE output |
| Loss changes but predictions always zero | Background term in Bayesian loss dominates | Disable background or reduce bg_ratio |
| Loss values always multiples of 0.0625 (with batch_size=8) | Model produces constant predictions for all inputs | Fix normalization layers first |
| One expert has 100% load, others 0% | Gate collapse due to upstream gradient issues | Fix gradient flow before debugging gate |
| Parameters barely change after training | clip_max_norm too aggressive | Use clip_max_norm=5.0 for 28M-param models |

## Verification Checklist

After applying fixes, verify with a short 5-epoch run:

- [ ] Eval MAE changes between epochs (not frozen at a constant)
- [ ] Loss values are proper floats, not multiples of 1/batch_size
- [ ] Predictions are non-zero and vary across different input images
- [ ] MoE gate load is distributed across experts (no 100/0/0 split)
- [ ] Training loss decreases over time (not stuck or monotonically increasing)
- [ ] Pre-activation values at the output layer have reasonable range (e.g., −5 to +5)

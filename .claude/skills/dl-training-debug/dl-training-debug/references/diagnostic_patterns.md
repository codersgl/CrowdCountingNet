# Reusable Diagnostic Script Patterns

These are canonical patterns for diagnosing frozen training metrics. Adapt the model construction and data loading to the project's API.

## Pattern A: Layer-by-layer activation trace

Compare trained vs. untrained model at every intermediate stage:

```python
"""Trace activations through a trained model to find collapse point."""
import torch
from hydra import compose, initialize_config_dir

# --- Build model and load checkpoint ---
config_dir = "<project_config_dir>"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="<config_name>", overrides=[...])

model = build_model(cfg).to(device)
ckpt = torch.load("<checkpoint_path>", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Also build untrained model for comparison
model_untrained = build_model(cfg).to(device)
model_untrained.eval()

# --- Trace both models on the same input ---
batch = next(iter(val_loader))
samples = batch[0].to(device)

with torch.no_grad():
    for label, m in [("trained", model), ("untrained", model_untrained)]:
        # Backbone
        fm = m.backbone(samples)
        print(f"\n=== {label} ===")
        c2_mean = fm['c2'].mean().item()
        print(f"  backbone c2: mean={c2_mean:.4f} std={fm['c2'].std():.4f}")

        # Neck
        neck_out = m.neck(fm["c2"], fm["c3"])
        print(f"  neck_out: mean={neck_out.mean():.4f} std={neck_out.std():.4f} "
              f"min={neck_out.min():.4f} max={neck_out.max():.4f}")

        # MoE stem
        stem = m.moe.stem(neck_out)
        print(f"  stem: mean={stem.mean():.4f} std={stem.std():.4f}")

        # Each expert
        for i, expert in enumerate(m.moe.experts):
            e_out = expert(stem)
            print(f"  expert{i}: mean={e_out.mean():.6f} min={e_out.min():.6f} max={e_out.max():.6f}")

        # Fused MoE output
        route = m.moe.gate(neck_out)
        rw = route["weights"]
        expert_outputs = torch.stack([e(stem) for e in m.moe.experts], dim=1)
        fused = (expert_outputs * rw.unsqueeze(2)).sum(dim=1)
        print(f"  fused: mean={fused.mean():.6f} min={fused.min():.6f} max={fused.max():.6f}")

        # Pre-activation and output
        pre_act = m.density_head.proj(fused)
        output = torch.nn.functional.softplus(pre_act)
        print(f"  pre_softplus: min={pre_act.min():.4f} max={pre_act.max():.4f} mean={pre_act.mean():.4f}")
        print(f"  output: min={output.min():.10f} max={output.max():.10f} mean={output.mean():.10f}")
```

## Pattern B: Bayesian loss background dominance check

Quantify whether the background term overwhelms point gradients:

```python
"""Check background posterior dominance in Bayesian loss."""
import torch

H, W, stride = 16, 16, 8
image_size = (H * stride, W * stride)
sigma = 8.0
two_sigma_sq = 2.0 * sigma * sigma
bg_ratio = 0.15
bg_dist_sq = (bg_ratio * max(image_size[0], image_size[1])) ** 2

# Build grid and compute posterior
gy = (torch.arange(H, dtype=torch.float32) + 0.5) * stride
gx = (torch.arange(W, dtype=torch.float32) + 0.5) * stride
grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

# Use actual point coordinates from a sample
pts = targets[0]["point"].to(device, dtype=torch.float32)
dist_sq = (coords.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(dim=-1)
min_dist_sq = dist_sq.min(dim=1, keepdim=True).values.clamp(min=0.0)
bg_dist_sq_vec = bg_dist_sq / (min_dist_sq + 1e-5)
dist_sq_ext = torch.cat([dist_sq, bg_dist_sq_vec], dim=1)

log_like = -dist_sq_ext / two_sigma_sq
log_like = log_like - log_like.max(dim=1, keepdim=True).values
likelihood = log_like.exp()
posterior = likelihood / likelihood.sum(dim=1, keepdim=True).clamp_min(1e-12)

bg_posterior = posterior[:, -1]
pts_posterior = posterior[:, :-1].sum(dim=1)

print(f"Fraction bg_posterior > 0.5: {(bg_posterior > 0.5).float().mean():.3f}")
print(f"Fraction bg_posterior > 0.8: {(bg_posterior > 0.8).float().mean():.3f}")
print(f"Mean bg posterior: {bg_posterior.mean():.4f}")
print(f"Mean point posterior: {pts_posterior.mean():.4f}")
print(f"Net gradient: {(-pts_posterior + bg_posterior).mean():+.6f}")
# If net gradient > 0: density is pushed DOWN (toward zero)
# Background is problematic if bg_posterior.mean() > 0.5
```

## Pattern C: Gradient norm audit

Measure gradient norms per parameter group without AMP to find bottlenecks:

```python
"""Measure gradient norms per parameter group."""
model.train()
optimizer.zero_grad(set_to_none=True)

loss = loss_fn(outputs, targets)
loss.backward()

# Measure BEFORE clipping
total_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

# Per-group norms
head_norm = nn.utils.clip_grad_norm_(head_params, float("inf"))
backbone_norm = nn.utils.clip_grad_norm_(backbone_params, float("inf"))
gate_norm = nn.utils.clip_grad_norm_(gate_params, float("inf"))

effective_lr = base_lr * min(1.0, clip_max_norm / max(total_norm, 1e-8))
print(f"Total norm: {total_norm:.1f} (head={head_norm:.1f} "
      f"backbone={backbone_norm:.1f} gate={gate_norm:.1f})")
print(f"Clip ratio: {total_norm / clip_max_norm:.1f}x")
print(f"Effective LR: {effective_lr:.2e}")
```

## Pattern D: Checkpoint comparison

Confirm model parameters are actually changing during training:

```python
"""Compare checkpoints to verify parameters are changing."""
ckpt_best = torch.load("best.pth", map_location="cpu", weights_only=False)
ckpt_latest = torch.load("latest.pth", map_location="cpu", weights_only=False)

changed = 0
max_change = 0.0
for key in ckpt_latest["model"]:
    if key in ckpt_best["model"]:
        diff = (ckpt_latest["model"][key] - ckpt_best["model"][key]).abs().max().item()
        if diff > 1e-8:
            changed += 1
            max_change = max(max_change, diff)

print(f"Parameters changed: {changed}")
print(f"Max change: {max_change:.8f}")
# If changed == 0: clip_max_norm is too aggressive or LR is too low
```

## Pattern E: Expert gate load balance

Check if MoE routing has collapsed:

```python
"""Check expert load balance."""
route = model.moe.gate(features)
hard_mask = route["hard_mask"]  # [B, num_experts, H, W]
for ei in range(num_experts):
    load = hard_mask[:, ei].float().mean().item()
    print(f"Expert {ei} load: {load:.4f}")
# Healthy: each expert has non-zero load, ideally balanced
# Collapsed: one expert = 1.0, all others = 0.0
```

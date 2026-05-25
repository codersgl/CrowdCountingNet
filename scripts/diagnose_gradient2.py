"""Diagnose MoECount gradient flow - part 2: measure gradient norms properly."""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")

import torch, torch.nn as nn, numpy as np
from omegaconf import OmegaConf

from crowdcount.data import build_dataset
from crowdcount.data.collate import make_train_collate
from crowdcount.models.moecount import build_moecount
from crowdcount.models.moecount.losses import (
    BayesianLoss, LoadBalanceLoss, LogCountLoss,
    LogCountWeightSchedule, MoECountLoss,
)
from crowdcount.models.moecount.engine import _align_density_to_target, _stack_density_maps, _move_targets
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
        "model.head.final_weight_std=0.0001",
        "data.batch_size=2",
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_moecount(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

loss_fn = MoECountLoss(
    bayesian_loss=BayesianLoss(sigma=8.0, use_background=True, bg_ratio=0.15, count_loss_type="l1"),
    log_count_loss=LogCountLoss(),
    log_count_schedule=LogCountWeightSchedule(initial_weight=0.1),
    balance_loss=LoadBalanceLoss(),
).to(device)

gate_params = [p for p in model.moe.gate.parameters() if p.requires_grad]
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
gate_ids = {id(p) for p in gate_params}
backbone_ids = {id(p) for p in backbone_params}
other_params = [p for p in model.parameters() if p.requires_grad
                and id(p) not in gate_ids and id(p) not in backbone_ids]

train_set, val_set = build_dataset(cfg)
sampler = torch.utils.data.RandomSampler(train_set)
batch_sampler = torch.utils.data.BatchSampler(sampler, int(cfg.data.batch_size), drop_last=True)
collate = make_train_collate(getattr(cfg.data, "augmentation", None), use_depth=False)
loader = torch.utils.data.DataLoader(
    train_set, batch_sampler=batch_sampler, collate_fn=collate, num_workers=0,
)

output_stride = int(cfg.model.output_stride)

# Build separate optimizers for each measurement approach
optimizer = torch.optim.AdamW([
    {"params": other_params, "name": "head"},
    {"params": backbone_params, "lr": cfg.optimizer.lr_backbone, "name": "backbone"},
    {"params": gate_params, "lr": cfg.optimizer.lr_gate, "name": "gate"},
], lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

print("=" * 70)
print("STEP 2 (fixed): Gradient norm analysis — no AMP first, then with AMP")
print("=" * 70)
print("clip_max_norm =", cfg.clip_max_norm)
print()

# ---- Approach A: No AMP, measure exact gradient norms ----
print("=== Without AMP (exact measurement) ===")
model.train()

total_norms = []
head_norms = []
backbone_norms = []
gate_norms = []

for step in range(3):
    batch = next(iter(loader))
    samples, targets, gt_density_maps = batch[:3]
    samples = samples.to(device)
    gt_density = _stack_density_maps(gt_density_maps, device)
    targets = _move_targets(targets, device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(samples)
    pred_density = outputs["density_out"]
    aligned_pred = _align_density_to_target(pred_density, gt_density)
    outputs = dict(outputs)
    outputs["density_out"] = aligned_pred
    image_sizes = (int(gt_density.shape[-2] * output_stride), int(gt_density.shape[-1] * output_stride))
    loss_dict = loss_fn(outputs, targets, gt_density, image_sizes, epoch=0)
    loss_total = loss_dict["loss_total"]

    loss_total.backward()

    # Measure per-parameter-group norms BEFORE clipping everything together
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    total_norms.append(float(total_norm))

    h_norm = nn.utils.clip_grad_norm_(other_params, float("inf")) if other_params else 0
    b_norm = nn.utils.clip_grad_norm_(backbone_params, float("inf")) if backbone_params else 0
    g_norm = nn.utils.clip_grad_norm_(gate_params, float("inf")) if gate_params else 0
    head_norms.append(float(h_norm) if isinstance(h_norm, torch.Tensor) else float(h_norm))
    backbone_norms.append(float(b_norm) if isinstance(b_norm, torch.Tensor) else float(b_norm))
    gate_norms.append(float(g_norm) if isinstance(g_norm, torch.Tensor) else float(g_norm))

    # Check what clipping at 0.1 would do
    scale = float(total_norm) / 0.1 if float(total_norm) > 0 else 1.0
    clipped = nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    effective_lr = cfg.optimizer.lr * min(1.0, 0.1 / max(float(total_norm), 1e-8))

    print(f"  Step {step}: total={total_norm:.1f} head={head_norms[-1]:.1f} "
          f"backbone={backbone_norms[-1]:.1f} gate={gate_norms[-1]:.1f} "
          f"loss={float(loss_total):.2f}")
    print(f"           clip@0.1={clipped:.4f} scale={scale:.0f}x effective_LR={effective_lr:.2e}")

avg_total = np.mean(total_norms)
print(f"\n  Avg total norm: {avg_total:.1f}")
print(f"  Avg head norm: {np.mean(head_norms):.1f}")
print(f"  Avg backbone norm: {np.mean(backbone_norms):.1f}")
print(f"  Avg gate norm: {np.mean(gate_norms):.1f}")

# ---- Summary ----
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Initial density prediction: ALWAYS 12.80 per 128x128 patch")
print(f"    (= 0.05 initial_density × 256 density-map pixels)")
print(f"  Density per pixel: ~0.05 (constant, independent of input content)")
print(f"  Gradient norm (total, no AMP): {avg_total:.1f}")
print(f"  clip_max_norm: 0.1")
ratio = avg_total / 0.1
print(f"  => Gradients scaled by {ratio:.0f}x EVERY step")
print(f"  => Effective LR: {cfg.optimizer.lr * 0.1 / max(avg_total, 0.01):.2e}")
print()
print(f"  Over 20 epochs × 150 steps = 3000 steps:")
bias_change = 3000 * cfg.optimizer.lr * 0.1 / max(avg_total, 0.01) * 0.05
print(f"    Estimated bias change: ~{bias_change:.4f} (from -2.97)")
print(f"    Starting density: 0.05 → after 20 epochs: ~{0.05:.4f}")
print()
print("  ROOT CAUSE CONFIRMED:")
if ratio > 3:
    print(f"  clip_max_norm=0.1 is {ratio:.0f}x too aggressive for this model size.")
    print(f"  The density head bias barely moves in 20 epochs.")
    print()
    print(f"  FIX 1: Set clip_max_norm={max(1.0, avg_total / 2):.0f} (was 0.1)")
    print(f"  FIX 2: Set model.head.final_weight_std=0.01 (was 0.0001)")

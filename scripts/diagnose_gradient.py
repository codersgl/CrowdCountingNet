"""Diagnose MoECount gradient flow: measure gradient norms before clipping."""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from crowdcount.data import build_dataset
from crowdcount.data.collate import make_train_collate
from crowdcount.models.moecount import build_moecount
from crowdcount.models.moecount.losses import (
    BayesianLoss, LoadBalanceLoss, LogCountLoss,
    LogCountWeightSchedule, MoECountLoss,
)
from crowdcount.models.moecount.engine import _align_density_to_target, _stack_density_maps, _move_targets

# Use Hydra's compose API to properly resolve defaults
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
overrides = [
    "data.data_root=data/shanghaitech/part_A_final",
    "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
    "model.head.final_activation=softplus",
    "model.head.initial_density=0.05",
    "model.head.final_weight_std=0.0001",
    "data.batch_size=2",
]

with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=overrides)

print("=" * 70)
print("DIAGNOSTIC: MoECount gradient flow analysis")
print("=" * 70)
print("clip_max_norm =", cfg.clip_max_norm)
print("head.final_weight_std =", cfg.model.head.final_weight_std)
print("head.initial_density =", cfg.model.head.initial_density)
print("batch_size =", cfg.data.batch_size)
print("patch =", cfg.data.patch, ", num_patches =", cfg.data.num_patches)
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = build_moecount(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", f"{n_params:,}")

loss_fn = MoECountLoss(
    bayesian_loss=BayesianLoss(sigma=8.0, use_background=True, bg_ratio=0.15, count_loss_type="l1"),
    log_count_loss=LogCountLoss(),
    log_count_schedule=LogCountWeightSchedule(initial_weight=0.1),
    balance_loss=LoadBalanceLoss(lambda_importance=0.01, lambda_load=0.01),
).to(device)

gate_params = [p for p in model.moe.gate.parameters() if p.requires_grad]
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
gate_ids = {id(p) for p in gate_params}
backbone_ids = {id(p) for p in backbone_params}
other_params = [p for p in model.parameters() if p.requires_grad
                and id(p) not in gate_ids and id(p) not in backbone_ids]
optimizer = torch.optim.AdamW(
    [
        {"params": other_params, "name": "head"},
        {"params": backbone_params, "lr": cfg.optimizer.lr_backbone, "name": "backbone"},
        {"params": gate_params, "lr": cfg.optimizer.lr_gate, "name": "gate"},
    ],
    lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
)

train_set, val_set = build_dataset(cfg)
sampler = torch.utils.data.RandomSampler(train_set)
batch_sampler = torch.utils.data.BatchSampler(sampler, int(cfg.data.batch_size), drop_last=True)
collate = make_train_collate(getattr(cfg.data, "augmentation", None), use_depth=False)
loader = torch.utils.data.DataLoader(
    train_set, batch_sampler=batch_sampler, collate_fn=collate, num_workers=0,
)

output_stride = int(cfg.model.output_stride)

# --- Step 1: Check initial density head output ---
print("\n" + "=" * 70)
print("STEP 1: Density head initial output analysis")
print("=" * 70)

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:
            break
        samples, targets, gt_density_maps = batch[:3]
        samples = samples.to(device)
        targets = _move_targets(targets, device)
        outputs = model(samples)
        pred_density = outputs["density_out"]
        n_show = min(4, pred_density.shape[0])
        for i in range(n_show):
            psum = float(pred_density[i].sum().item())
            pmean = float(pred_density[i].mean().item())
            pmin = float(pred_density[i].min().item())
            pmax = float(pred_density[i].max().item())
            gt_count = int(targets[i]["point"].shape[0])
            print(f"  Patch {i}: sum={psum:.2f} mean={pmean:.6f} min={pmin:.6f} max={pmax:.6f} gt={gt_count}")

# --- Step 2: Measure gradient norms ---
print("\n" + "=" * 70)
print("STEP 2: Gradient norm analysis (5 training steps)")
print("=" * 70)

model.train()
amp_enabled = bool(getattr(cfg.mixed_precision, "enabled", True)) and device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

total_norms_unclipped = []
head_norms = []
backbone_norms = []
gate_norms = []

for step in range(5):
    batch = next(iter(loader))
    samples, targets, gt_density_maps = batch[:3]
    samples = samples.to(device)
    gt_density = _stack_density_maps(gt_density_maps, device)
    targets = _move_targets(targets, device)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        outputs = model(samples)
        pred_density = outputs["density_out"]
        aligned_pred = _align_density_to_target(pred_density, gt_density)
        outputs = dict(outputs)
        outputs["density_out"] = aligned_pred
        image_sizes = (int(gt_density.shape[-2] * output_stride), int(gt_density.shape[-1] * output_stride))
        loss_dict = loss_fn(outputs, targets, gt_density, image_sizes, epoch=0)
        loss_total = loss_dict["loss_total"]

    if amp_enabled:
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
    else:
        loss_total.backward()

    # Measure unclipped gradient norm
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    total_norms_unclipped.append(float(total_norm))

    for param_list in [other_params, backbone_params, gate_params]:
        if param_list:
            gnorm = nn.utils.clip_grad_norm_(param_list, float("inf"))
            [head_norms, backbone_norms, gate_norms][[other_params, backbone_params, gate_params].index(param_list)].append(float(gnorm))

    effective_scale = float(total_norm) / 0.1 if float(total_norm) > 0 else 0
    clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    print(f"  Step {step}: unclipped={total_norm:.2f}, clipped@0.1={clipped_norm:.4f}, "
          f"scale={effective_scale:.1f}x, loss={float(loss_total):.4f}")

avg = np.mean(total_norms_unclipped) if total_norms_unclipped else 0
print()
print(f"  Avg unclipped norm: {avg:.2f}")
print(f"  Avg head norm:      {np.mean(head_norms) if head_norms else 0:.2f}")
print(f"  Avg backbone norm:  {np.mean(backbone_norms) if backbone_norms else 0:.2f}")
print(f"  Avg gate norm:      {np.mean(gate_norms) if gate_norms else 0:.2f}")
if avg > 0:
    print(f"  Clip@0.1 scale:     {avg / 0.1:.1f}x reduction")
    print(f"  Effective LR:       {cfg.optimizer.lr * 0.1 / max(avg, 0.01):.2e}")

# --- Step 3: Prediction diversity ---
print("\n" + "=" * 70)
print("STEP 3: Prediction diversity (different images)")
print("=" * 70)

model.eval()
all_counts = []
all_gts = []
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:
            break
        samples, targets, gt_density_maps = batch[:3]
        samples = samples.to(device)
        targets = _move_targets(targets, device)
        outputs = model(samples)
        pred_density = outputs["density_out"]
        for i in range(pred_density.shape[0]):
            all_counts.append(float(pred_density[i].sum().item()))
            all_gts.append(int(targets[i]["point"].shape[0]))

print(f"  Pred counts: {[f'{c:.1f}' for c in all_counts[:10]]}")
print(f"  GT counts:   {all_gts[:10]}")
print(f"  Std of preds: {np.std(all_counts):.2f}")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
ratio = avg / float(cfg.clip_max_norm) if avg > 0 else 0
print(f"  Model: {n_params:,} params, batch_size={cfg.data.batch_size}")
print(f"  Gradient norm (avg across 5 steps): {avg:.1f}")
print(f"  clip_max_norm setting: {cfg.clip_max_norm}")
print(f"  Gradient scale factor: {ratio:.0f}x reduction")
print(f"  Head final_weight_std: {cfg.model.head.final_weight_std}")

if ratio > 3:
    print()
    print("  *** PROBLEM CONFIRMED: clip_max_norm is TOO AGGRESSIVE ***")
    print(f"  Gradients are reduced by {ratio:.0f}x every step.")
    print(f"  This is why metrics are frozen for 20 epochs.")
    print()
    print(f"  Fix 1: clip_max_norm >= {max(1.0, avg / 2):.1f} (currently 0.1)")
    print(f"  Fix 2: model.head.final_weight_std >= 0.01 (currently {cfg.model.head.final_weight_std})")
elif ratio > 1.5:
    print()
    print("  => clipping is somewhat aggressive but may not fully explain frozen metrics")
else:
    print("  => clip_max_norm is reasonable for this model scale.")

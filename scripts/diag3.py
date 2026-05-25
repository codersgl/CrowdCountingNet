"""Deep dive: why is eval frozen even with fixes?"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, torch.nn as nn, numpy as np
from crowdcount.data import build_dataset
from crowdcount.data.collate import make_train_collate
from crowdcount.models.moecount import build_moecount
from crowdcount.models.moecount.engine import (_align_density_to_target, _stack_density_maps, _move_targets, _crop_density_for_eval)
from crowdcount.models.moecount.losses import BayesianLoss, LogCountLoss, LogCountWeightSchedule, LoadBalanceLoss, MoECountLoss
from crowdcount.trainers.moecount_trainer import collate_fn_moecount_eval
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
        "data.batch_size=2",
    ])

device = torch.device("cuda")
model = build_moecount(cfg).to(device)
print("clip_max_norm:", cfg.clip_max_norm)
print("final_weight_std:", cfg.model.head.final_weight_std)

initial_bias = model.density_head.proj[-1].bias.data.clone()
sp_init = float(torch.nn.functional.softplus(initial_bias).item())
print(f"\nInitial density_head final bias: {initial_bias.item():.6f}")
print(f"  softplus(bias) = {sp_init:.6f} (= initial_density)")

# Check eval predictions
val_set = build_dataset(cfg)[1]
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn_moecount_eval, num_workers=0)

model.eval()
print("\n--- Eval BEFORE training (3 images) ---")
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 3: break
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_cropped = _crop_density_for_eval(pd, targets[0], 8)
        psum = float(pd_cropped.sum().item())
        pmean = float(pd_cropped.mean().item())
        pstd = float(pd_cropped.std().item())
        gt = int(targets[0]["point"].shape[0])
        print(f"  Img {idx}: count={psum:.2f} mean={pmean:.6f} std={pstd:.6f} gt={gt}")

# Build training stuff
train_set = build_dataset(cfg)[0]
sampler = torch.utils.data.RandomSampler(train_set)
batch_sampler = torch.utils.data.BatchSampler(sampler, 2, drop_last=True)
collate = make_train_collate(getattr(cfg.data, "augmentation", None), use_depth=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=batch_sampler, collate_fn=collate, num_workers=0)

loss_fn = MoECountLoss(
    bayesian_loss=BayesianLoss(sigma=8.0, use_background=True, bg_ratio=0.15, count_loss_type="l1"),
    log_count_loss=LogCountLoss(), log_count_schedule=LogCountWeightSchedule(initial_weight=0.1),
    balance_loss=LoadBalanceLoss(),
).to(device)

gate_params = [p for p in model.moe.gate.parameters() if p.requires_grad]
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
gate_ids = {id(p) for p in gate_params}
backbone_ids = {id(p) for p in backbone_params}
other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids and id(p) not in backbone_ids]

optimizer = torch.optim.AdamW([
    {"params": other_params, "name": "head"},
    {"params": backbone_params, "lr": cfg.optimizer.lr_backbone, "name": "backbone"},
    {"params": gate_params, "lr": cfg.optimizer.lr_gate, "name": "gate"},
], lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

# Save initial params
initial_params = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        initial_params[name] = param.data.clone()

scaler = torch.cuda.amp.GradScaler(enabled=True)
output_stride = 8
clip_val = float(cfg.clip_max_norm)

print(f"\n--- Training 10 steps (clip_max_norm={clip_val}, final_weight_std={cfg.model.head.final_weight_std}) ---")
model.train()
for step in range(10):
    batch = next(iter(train_loader))
    samples, targets, gt_density_maps = batch[:3]
    samples = samples.to(device)
    gt_density = _stack_density_maps(gt_density_maps, device)
    targets_d = _move_targets(targets, device)

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(samples)
        pd_out = outputs["density_out"]
        aligned = _align_density_to_target(pd_out, gt_density)
        outputs_d = dict(outputs); outputs_d["density_out"] = aligned
        img_sz = (int(gt_density.shape[-2]*8), int(gt_density.shape[-1]*8))
        loss_dict = loss_fn(outputs_d, targets_d, gt_density, img_sz, epoch=0)
        loss = loss_dict["loss_total"]

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    if clip_val > 0:
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    scaler.step(optimizer)
    scaler.update()

    if step < 3 or step == 9:
        print(f"  Step {step}: unclipped_norm={total_norm:.1f} loss={float(loss.detach()):.4f}")

# Check param changes
print("\n--- Parameter changes after 10 steps ---")
changed_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        change = (param.data - initial_params[name]).abs().max().item()
        if change > 1e-8:
            changed_params.append((name, change))

final_bias_val = model.density_head.proj[-1].bias.data.clone()
sp_final = float(torch.nn.functional.softplus(final_bias_val).item())
print(f"  density_head bias: {initial_bias.item():.6f} -> {final_bias_val.item():.6f}")
print(f"    delta = {final_bias_val.item() - initial_bias.item():.8f}")
print(f"    softplus: {sp_init:.6f} -> {sp_final:.6f}")
print(f"  # params that changed (>1e-8): {len(changed_params)}")

if changed_params:
    print("  Top 10 changes:")
    for name, change in sorted(changed_params, key=lambda x: -x[1])[:10]:
        print(f"    {name}: max_change={change:.8f}")
else:
    print("  *** NO PARAMETERS CHANGED AT ALL ***")

# Check eval after training
print("\n--- Eval AFTER 10 training steps ---")
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 3: break
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_cropped = _crop_density_for_eval(pd, targets[0], 8)
        psum = float(pd_cropped.sum().item())
        pmean = float(pd_cropped.mean().item())
        pstd = float(pd_cropped.std().item())
        gt = int(targets[0]["point"].shape[0])
        print(f"  Img {idx}: count={psum:.2f} mean={pmean:.6f} std={pstd:.6f} gt={gt}")

"""Compare fresh model vs checkpoint model"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, numpy as np
from crowdcount.data import build_dataset
from crowdcount.models.moecount import build_moecount
from crowdcount.models.moecount.engine import _crop_density_for_eval
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

# Fresh model
model_fresh = build_moecount(cfg).to(device)
print("clip_max_norm:", cfg.clip_max_norm)
print("final_weight_std:", cfg.model.head.final_weight_std)

# Load checkpoint model
ckpt_path = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/outputs/2026-05-25/17-55-48/checkpoints/latest.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
model_ckpt = build_moecount(cfg).to(device)
model_ckpt.load_state_dict(ckpt["model"])
print(f"\nCheckpoint epoch: {ckpt['epoch']}, best_mae: {ckpt['best_mae']:.2f}")

# Compare key parameters
print("\n=== Key parameter comparison ===")
for name, p_fresh in model_fresh.named_parameters():
    if p_fresh.requires_grad:
        p_ckpt = dict(model_ckpt.named_parameters())[name]
        diff = (p_ckpt - p_fresh).abs().max().item()
        if diff > 1e-6:
            print(f"  {name}: max_diff={diff:.6f} (fresh_norm={p_fresh.data.norm().item():.4f}, ckpt_norm={p_ckpt.data.norm().item():.4f})")

# Bias check
fresh_bias = model_fresh.density_head.proj[-1].bias.item()
ckpt_bias = model_ckpt.density_head.proj[-1].bias.item()
print(f"\n  density_head bias: fresh={fresh_bias:.6f}, ckpt={ckpt_bias:.6f}, delta={ckpt_bias-fresh_bias:.8f}")
print(f"    softplus(fresh)={float(torch.nn.functional.softplus(torch.tensor(fresh_bias)).item()):.6f}")
print(f"    softplus(ckpt)={float(torch.nn.functional.softplus(torch.tensor(ckpt_bias)).item()):.6f}")

# Compare BN running stats
print("\n=== BatchNorm running stats comparison ===")
for name, m_fresh in model_fresh.named_modules():
    if isinstance(m_fresh, torch.nn.BatchNorm2d):
        m_ckpt = dict(model_ckpt.named_modules())[name]
        rm_diff = (m_ckpt.running_mean - m_fresh.running_mean).abs().max().item()
        rv_diff = (m_ckpt.running_var - m_fresh.running_var).abs().max().item()
        print(f"  {name}: running_mean_diff={rm_diff:.6f}, running_var_diff={rv_diff:.6f}")

# Eval comparison
val_set = build_dataset(cfg)[1]
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn_moecount_eval, num_workers=0)

print("\n=== Eval on first 3 images ===")
for label, model in [("fresh", model_fresh), ("ckpt", model_ckpt)]:
    model.eval()
    counts = []
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
            gt = int(targets[0]["point"].shape[0])
            counts.append((psum, gt))
    print(f"  {label}:")
    for i, (c, g) in enumerate(counts):
        print(f"    Img {i}: pred={c:.2f} gt={g} err={abs(c-g):.1f}")

"""Eval best vs latest checkpoint from 19-20-31 run."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, numpy as np
from crowdcount.models.moecount import build_moecount
from crowdcount.data import build_dataset
from crowdcount.trainers.moecount_trainer import collate_fn_moecount_eval
from crowdcount.models.moecount.engine import evaluate_moecount, _crop_density_for_eval
from torch.utils.data import DataLoader, SequentialSampler
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
    ])

device = torch.device("cuda")

_, val_set = build_dataset(cfg)
sampler_val = SequentialSampler(val_set)
val_loader = DataLoader(val_set, batch_size=1, sampler=sampler_val, drop_last=False,
                        collate_fn=collate_fn_moecount_eval, num_workers=0)

# Load best (epoch 0) checkpoint
best_ckpt = torch.load("outputs/2026-05-25/19-20-31/checkpoints/best_mae.pth",
                       map_location=device, weights_only=False)
latest_ckpt = torch.load("outputs/2026-05-25/19-20-31/checkpoints/latest.pth",
                         map_location=device, weights_only=False)

print(f"Best epoch: {best_ckpt['epoch']}")
print(f"Latest epoch: {latest_ckpt['epoch']}")

model_best = build_moecount(cfg).to(device)
model_latest = build_moecount(cfg).to(device)

model_best.load_state_dict(best_ckpt["model"])
model_latest.load_state_dict(latest_ckpt["model"])

mae_best, mse_best = evaluate_moecount(model_best, val_loader, device, output_stride=int(cfg.model.output_stride))
print(f"\nBest (epoch 0): mae={mae_best:.10f} mse={mse_best:.10f}")

mae_latest, mse_latest = evaluate_moecount(model_latest, val_loader, device, output_stride=int(cfg.model.output_stride))
print(f"Latest (epoch 7): mae={mae_latest:.10f} mse={mse_latest:.10f}")

print(f"\nMAE changed? {abs(mae_best - mae_latest) > 0.001}")
print(f"MAE diff: {abs(mae_best - mae_latest):.6f}")

# Also check per-image predictions
model_latest.eval()
preds_latest = []
preds_best = []
model_best.eval()
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 10:
            break
        samples, targets = batch[:2]
        samples = samples.to(device)
        out_b = model_best(samples)["density_out"]
        out_l = model_latest(samples)["density_out"]
        preds_best.append(float(_crop_density_for_eval(out_b, targets[0], 8).sum().item()))
        preds_latest.append(float(_crop_density_for_eval(out_l, targets[0], 8).sum().item()))

print("\nFirst 10 predictions comparison:")
for i in range(10):
    print(f"  Img {i}: best={preds_best[i]:.2f} latest={preds_latest[i]:.2f} diff={preds_latest[i]-preds_best[i]:+.4f}")

"""Compute exact MAE on full test set with untrained model."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, numpy as np
from crowdcount.models.moecount import build_moecount
from crowdcount.data import build_dataset
from crowdcount.trainers.moecount_trainer import collate_fn_moecount_eval
from crowdcount.models.moecount.engine import _crop_density_for_eval
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
        "data.batch_size=1",
    ])

device = torch.device("cuda")
model = build_moecount(cfg).to(device)
model.eval()

val_set = build_dataset(cfg)[1]
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                         collate_fn=collate_fn_moecount_eval, num_workers=0)

maes, mses = [], []
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_crop = _crop_density_for_eval(pd, targets[0], 8)
        pred_count = float(pd_crop.sum().item())
        gt_count = float(targets[0]["point"].shape[0])
        error = pred_count - gt_count
        maes.append(abs(error))
        mses.append(error * error)

mae = float(np.mean(maes))
mse = float(np.sqrt(np.mean(mses)))
print(f"Full eval (untrained): n={len(maes)} mae={mae:.10f} mse={mse:.10f}")

# Print full precision
print(f"MAE raw = {mae}")
print(f"MSE raw = {mse}")

# Check if this matches the training output
print(f"\nTraining output was: mae=433.90 mse=559.81")
print(f"Match: {abs(mae - 433.9010989010981) < 0.001}")

# Now load the trained checkpoint and compare
ckpt = torch.load("outputs/2026-05-25/18-18-25/checkpoints/latest.pth", map_location=device,
                  weights_only=False)
# Check if architectures match
ckpt_keys = set(ckpt["model"].keys())
model_keys = set(model.state_dict().keys())
missing = model_keys - ckpt_keys
extra = ckpt_keys - model_keys
if missing:
    print(f"\nArchitecture mismatch: model missing keys from checkpoint: {missing}")
if extra:
    print(f"Architecture mismatch: checkpoint has extra keys not in model: {extra}")

# Try loading matching keys only
if missing or extra:
    matching = {k: v for k, v in ckpt["model"].items() if k in model_keys}
    model.load_state_dict(matching, strict=False)
    print(f"Loaded {len(matching)} / {len(model_keys)} matching keys")
else:
    model.load_state_dict(ckpt["model"])
    print(f"Loaded all {len(model_keys)} keys")

# Re-eval
model.eval()
maes2, mses2 = [], []
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_crop = _crop_density_for_eval(pd, targets[0], 8)
        pred_count = float(pd_crop.sum().item())
        gt_count = float(targets[0]["point"].shape[0])
        error = pred_count - gt_count
        maes2.append(abs(error))
        mses2.append(error * error)

mae2 = float(np.mean(maes2))
mse2 = float(np.sqrt(np.mean(mses2)))
print(f"\nFull eval (trained ckpt): mae={mae2:.10f} mse={mse2:.10f}")

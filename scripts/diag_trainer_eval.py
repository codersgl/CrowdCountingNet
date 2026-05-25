"""Reproduce eval exactly as trainer does it."""
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
model = build_moecount(cfg).to(device)

# Reproduce trainer's val loader construction exactly
_, val_set = build_dataset(cfg)
sampler_val = SequentialSampler(val_set)
val_loader = DataLoader(
    val_set,
    batch_size=1,
    sampler=sampler_val,
    drop_last=False,
    collate_fn=collate_fn_moecount_eval,
    num_workers=0,
)

# Use EXACT same eval function
mae, mse = evaluate_moecount(model, val_loader, device, output_stride=int(cfg.model.output_stride))
print(f"evaluate_moecount: mae={mae:.10f} mse={mse:.10f}")

# Also compute manually
model.eval()
maes, mses = [], []
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_crop = _crop_density_for_eval(pd, targets[0], int(cfg.model.output_stride))
        pred_count = float(pd_crop.sum().item())
        gt_count = float(targets[0]["point"].shape[0])
        error = pred_count - gt_count
        maes.append(abs(error))
        mses.append(error * error)

mae2 = float(np.mean(maes))
mse2 = float(np.sqrt(np.mean(mses)))
print(f"Manual: mae={mae2:.10f} mse={mse2:.10f}")
print(f"Match: {abs(mae - mae2) < 1e-6}")

# Print a few pred counts
print(f"\nFirst 5 pred counts: {[f'{float(c):.2f}' for c in maes[:5]]}")
print(f"GT counts used: N/A")

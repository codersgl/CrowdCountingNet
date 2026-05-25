"""Check if model produces non-constant predictions for different images."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch
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

print("=== Predictions for 10 eval images (untrained model) ===")
counts = []
densities_mean = []
densities_std = []
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 10:
            break
        samples, targets = batch[:2]
        samples = samples.to(device)
        outputs = model(samples)
        pd = outputs["density_out"]
        pd_crop = _crop_density_for_eval(pd, targets[0], 8)
        c = float(pd_crop.sum().item())
        pmean = float(pd_crop.mean().item())
        pstd = float(pd_crop.std().item())
        gt = int(targets[0]["point"].shape[0])
        counts.append(c)
        densities_mean.append(pmean)
        densities_std.append(pstd)
        print(f"  Img {idx}: count={c:.2f} mean={pmean:.6f} std={pstd:.6f} gt={gt}")

print(f"\nCounts: {[f'{c:.2f}' for c in counts]}")
print(f"Range: {min(counts):.2f} - {max(counts):.2f} (diff={max(counts)-min(counts):.4f})")
print(f"Density means: {[f'{m:.6f}' for m in densities_mean]}")
print(f"Density stds: {[f'{s:.6f}' for s in densities_std]}")

# Check final layer bias
bias = model.density_head.proj[-1].bias.data.item()
sp = float(torch.nn.functional.softplus(torch.tensor(bias)).item())
print(f"\nDensity head final bias: {bias:.6f}")
print(f"softplus(bias) = {sp:.6f}")

# Check neck output
print("\n=== Neck output analysis ===")
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 3:
            break
        samples, targets = batch[:2]
        samples = samples.to(device)
        c3, c2_list = model.backbone(samples)
        neck_out = model.neck(c2_list[1], c3)
        print(f"  Img {idx}: neck mean={neck_out.mean().item():.4f} std={neck_out.std().item():.4f} min={neck_out.min().item():.4f} max={neck_out.max().item():.4f}")

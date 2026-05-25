"""Compare initial model weights vs trained checkpoint from latest run."""
from __future__ import annotations
import sys, os
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")

# Find the most recent training output
outputs = sorted(os.listdir("outputs/2026-05-25/"))
latest = [d for d in outputs if os.path.isdir(f"outputs/2026-05-25/{d}/checkpoints")]
latest_run = latest[-1] if latest else None
print(f"Latest run: {latest_run}")

ckpt_path = f"outputs/2026-05-25/{latest_run}/checkpoints/latest.pth"
best_path = f"outputs/2026-05-25/{latest_run}/checkpoints/best_mae.pth"

import torch
ckpt_latest = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ckpt_best = torch.load(best_path, map_location='cpu', weights_only=False)

print(f"Latest epoch: {ckpt_latest['epoch']}")
print(f"Best epoch: {ckpt_best['epoch']}")

# Compare model weights
latest_model = ckpt_latest['model']
best_model = ckpt_best['model']

changed = 0
unchanged = 0
total = 0
max_change = 0.0
max_change_name = ""
for key in latest_model:
    if key in best_model:
        diff = (latest_model[key] - best_model[key]).abs().max().item()
        total += 1
        if diff > 1e-8:
            changed += 1
            if diff > max_change:
                max_change = diff
                max_change_name = key
        else:
            unchanged += 1

print(f"\nModel parameter comparison (latest vs best):")
print(f"  Total keys: {total}")
print(f"  Changed (>1e-8): {changed}")
print(f"  Unchanged: {unchanged}")
print(f"  Max change: {max_change:.8f} ({max_change_name})")

# Show top 10 most changed parameters
changes = []
for key in latest_model:
    if key in best_model:
        diff = (latest_model[key] - best_model[key]).abs().max().item()
        changes.append((key, diff, latest_model[key].numel()))
changes.sort(key=lambda x: -x[1])
print("\nTop 10 parameter changes:")
for name, diff, n in changes[:10]:
    print(f"  {name}: max_diff={diff:.8f} params={n}")

# Also compare with a freshly built model
from crowdcount.models.moecount import build_moecount
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
    ])
torch.manual_seed(42)
model = build_moecount(cfg)
fresh_state = model.state_dict()

# Compare latest checkpoint to fresh build
same = 0
for key in fresh_state:
    if key in latest_model:
        diff = (fresh_state[key] - latest_model[key]).abs().max().item()
        if diff < 1e-8:
            same += 1

print(f"\nFresh model vs latest checkpoint: {same}/{len(fresh_state)} keys unchanged")
print("(backbone params should match since lr_backbone=0.00001 and clip_max_norm=5.0)")

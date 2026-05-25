"""Test if eval predictions vary at all after training."""
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

# Load the latest checkpoint
ckpt = torch.load("outputs/2026-05-25/18-18-25/checkpoints/latest.pth", map_location=device)
model.load_state_dict(ckpt["model"])
print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

val_set = build_dataset(cfg)[1]
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn_moecount_eval, num_workers=0)

# Also test with an untrained model for comparison
model_untrained = build_moecount(cfg).to(device)

print("\n=== Trained model predictions (first 5 images) ===")
model.eval()
counts_trained = []
counts_untrained = []
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        if idx >= 5:
            break
        samples, targets = batch[:2]
        samples = samples.to(device)

        out_trained = model(samples)
        out_untrained = model_untrained(samples)

        pd_t = out_trained["density_out"]
        pd_u = out_untrained["density_out"]

        pd_t_crop = _crop_density_for_eval(pd_t, targets[0], 8)
        pd_u_crop = _crop_density_for_eval(pd_u, targets[0], 8)

        ct = float(pd_t_crop.sum().item())
        cu = float(pd_u_crop.sum().item())
        gt = int(targets[0]["point"].shape[0])

        # Check if predictions vary spatially
        t_std = float(pd_t_crop.std().item())
        u_std = float(pd_u_crop.std().item())

        counts_trained.append(ct)
        counts_untrained.append(cu)

        print(f"  Img {idx}: trained_count={ct:.2f} untrained_count={cu:.2f} gt={gt} "
              f"trained_std={t_std:.6f} untrained_std={u_std:.6f}")

print(f"\nTrained counts:   {counts_trained}")
print(f"Untrained counts: {counts_untrained}")
print(f"Variation in trained: {max(counts_trained) - min(counts_trained):.2f}")
print(f"Variation in untrained: {max(counts_untrained) - min(counts_untrained):.2f}")

# Check if density_head bias changed from init
trained_bias = model.density_head.proj[-1].bias.data.item()
untrained_bias = model_untrained.density_head.proj[-1].bias.data.item()
print(f"\nDensity head final bias: untrained={untrained_bias:.6f} trained={trained_bias:.6f}")
print(f"  softplus(untrained)={torch.nn.functional.softplus(torch.tensor(untrained_bias)).item():.6f}")
print(f"  softplus(trained)={torch.nn.functional.softplus(torch.tensor(trained_bias)).item():.6f}")

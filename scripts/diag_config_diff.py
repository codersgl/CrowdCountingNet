"""Check if config differs between with and without batch_size override."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, numpy as np
from crowdcount.models.moecount import build_moecount
from crowdcount.data import build_dataset
from crowdcount.trainers.moecount_trainer import collate_fn_moecount_eval
from crowdcount.models.moecount.engine import _crop_density_for_eval, evaluate_moecount
from torch.utils.data import DataLoader, SequentialSampler
from hydra import compose, initialize_config_dir

config_dir = "/home/codersgl/sci-research/CrowdCounting-DSGCNet/configs"

# Config A: no overrides (like training)
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfgA = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
    ])

# Config B: batch_size=1 override
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfgB = compose(config_name="moecount_config", overrides=[
        "data.data_root=data/shanghaitech/part_A_final",
        "model.backbone.pretrained_path=convnext_tiny_fb22k_ft_in1k.tar.gz",
        "data.batch_size=1",
    ])

# Compare model configs
from omegaconf import OmegaConf
print("=== Model config comparison ===")
diff = OmegaConf.to_yaml(cfgA.model) == OmegaConf.to_yaml(cfgB.model)
print(f"Model configs identical: {diff}")

print(f"\ncfgA.data.batch_size: {cfgA.data.batch_size}")
print(f"cfgB.data.batch_size: {cfgB.data.batch_size}")

print(f"\ncfgA.model.head.initial_density: {cfgA.model.head.initial_density}")
print(f"cfgB.model.head.initial_density: {cfgB.model.head.initial_density}")
print(f"cfgA.model.head.final_weight_std: {cfgA.model.head.final_weight_std}")
print(f"cfgB.model.head.final_weight_std: {cfgB.model.head.final_weight_std}")

device = torch.device("cuda")
torch.manual_seed(42)
np.random.seed(42)

modelA = build_moecount(cfgA).to(device)
modelB = build_moecount(cfgB).to(device)

# Check if model weights differ
for (na, pa), (nb, pb) in zip(modelA.named_parameters(), modelB.named_parameters()):
    if na != nb:
        print(f"Param name mismatch: {na} vs {nb}")
        break
    if not torch.equal(pa.data, pb.data):
        max_diff = (pa.data - pb.data).abs().max().item()
        print(f"Param {na} differs: max_diff={max_diff:.10f}")
        
# Check single prediction on sample image
_, val_set = build_dataset(cfgA)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                        collate_fn=collate_fn_moecount_eval, num_workers=0)

modelA.eval()
modelB.eval()
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        samples, targets = batch[:2]
        samples = samples.to(device)
        outA = modelA(samples)["density_out"]
        outB = modelB(samples)["density_out"]
        diff = (outA - outB).abs().max().item()
        print(f"Img {idx}: pred diff = {diff:.10f}")
        if idx >= 2:
            break

print("\nDone. Model diff check complete.")

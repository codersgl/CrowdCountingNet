"""Check density map values from trained checkpoint."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch
from crowdcount.models.moecount import build_moecount
from crowdcount.data import build_dataset
from crowdcount.trainers.moecount_trainer import collate_fn_moecount_eval
from crowdcount.models.moecount.engine import _crop_density_for_eval
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

ckpt = torch.load("outputs/2026-05-25/19-20-31/checkpoints/latest.pth",
                  map_location=device, weights_only=False)
model = build_moecount(cfg).to(device)
model.load_state_dict(ckpt["model"])

# Check GroupNorm layers
print("=== Trained GroupNorm params ===")
for name, m in model.named_modules():
    if isinstance(m, torch.nn.GroupNorm):
        w = m.weight.data
        b = m.bias.data
        print(f"  {name}: weight mean={w.mean().item():.4f} bias mean={b.mean().item():.4f}")

# Check density head
fc = model.density_head.proj[-1]
print(f"\nFinal conv: weight mean={fc.weight.mean().item():.6f} std={fc.weight.std().item():.6f} bias={fc.bias.item():.6f}")

# Full model forward with feature tracing
model.eval()
with torch.no_grad():
    batch = next(iter(val_loader))
    samples, targets = batch[:2]
    samples = samples.to(device)

    feature_maps = model.backbone(samples)
    c2, c3 = feature_maps["c2"], feature_maps["c3"]
    print(f"\nbackbone c2: mean={c2.mean().item():.4f} std={c2.std().item():.4f} min={c2.min().item():.4f} max={c2.max().item():.4f}")
    print(f"backbone c3: mean={c3.mean().item():.4f} std={c3.std().item():.4f} min={c3.min().item():.4f} max={c3.max().item():.4f}")

    # Neck step-by-step
    c2_proj = model.neck.c2_proj(c2)
    c3_proj = model.neck.c3_proj(c3)
    c3_up = torch.nn.functional.interpolate(c3_proj, size=c2_proj.shape[-2:], mode="bilinear", align_corners=False)
    base = c2_proj + c3_up
    print(f"\nneck base: mean={base.mean().item():.4f} std={base.std().item():.4f} min={base.min().item():.4f} max={base.max().item():.4f}")

    context = torch.cat([branch(base) for branch in model.neck.context_branches], dim=1)
    context = model.neck.context_norm(context)
    context = model.neck.context_fuse(context)
    print(f"neck context: mean={context.mean().item():.4f} std={context.std().item():.4f} min={context.min().item():.4f} max={context.max().item():.4f}")

    neck_out = model.neck.output_norm(base + context)
    print(f"neck out: mean={neck_out.mean().item():.4f} std={neck_out.std().item():.4f} min={neck_out.min().item():.4f} max={neck_out.max().item():.4f}")

    # MoE step-by-step
    stem_out = model.moe.stem(neck_out)
    print(f"\nmoe stem: mean={stem_out.mean().item():.4f} std={stem_out.std().item():.4f} min={stem_out.min().item():.4f} max={stem_out.max().item():.4f}")

    e0 = model.moe.experts[0](stem_out)
    e1 = model.moe.experts[1](stem_out)
    e2 = model.moe.experts[2](stem_out)
    print(f"expert0 (local): mean={e0.mean().item():.6f} std={e0.std().item():.6f} min={e0.min().item():.6f} max={e0.max().item():.6f}")
    print(f"expert1 (dilated): mean={e1.mean().item():.6f} std={e1.std().item():.6f} min={e1.min().item():.6f} max={e1.max().item():.6f}")
    print(f"expert2 (cbam): mean={e2.mean().item():.6f} std={e2.std().item():.6f} min={e2.min().item():.6f} max={e2.max().item():.6f}")

    # MoE routing
    route = model.moe.gate(neck_out)
    rw = route["weights"]
    print(f"\nroute weights: shape={rw.shape} min={rw.min().item():.4f} max={rw.max().item():.4f}")
    print(f"  e0 weight: mean={rw[:,0].mean().item():.4f}")
    print(f"  e1 weight: mean={rw[:,1].mean().item():.4f}")
    print(f"  e2 weight: mean={rw[:,2].mean().item():.4f}")

    expert_outputs = torch.stack([e0, e1, e2], dim=1)
    fused = (expert_outputs * rw.unsqueeze(2)).sum(dim=1)
    print(f"\nfused moe: mean={fused.mean().item():.6f} std={fused.std().item():.6f} min={fused.min().item():.6f} max={fused.max().item():.6f}")

    pre_softplus = model.density_head.proj(fused)
    print(f"\npre-softplus: min={pre_softplus.min().item():.6f} max={pre_softplus.max().item():.6f} mean={pre_softplus.mean().item():.6f}")

    density = torch.nn.functional.softplus(pre_softplus)
    density_crop = _crop_density_for_eval(density, targets[0], 8)
    print(f"density: min={density_crop.min().item():.10f} max={density_crop.max().item():.10f} mean={density_crop.mean().item():.10f} sum={density_crop.sum().item():.6f}")

# Compare with untrained
model2 = build_moecount(cfg).to(device)
model2.eval()
print("\n=== UNTRAINED MODEL ===")
with torch.no_grad():
    batch = next(iter(val_loader))
    samples, targets = batch[:2]
    samples = samples.to(device)

    feature_maps = model2.backbone(samples)
    c2, c3 = feature_maps["c2"], feature_maps["c3"]
    neck_out = model2.neck(c2, c3)
    print(f"neck out: mean={neck_out.mean().item():.4f} std={neck_out.std().item():.4f} min={neck_out.min().item():.4f} max={neck_out.max().item():.4f}")

    stem_out = model2.moe.stem(neck_out)
    print(f"moe stem: mean={stem_out.mean().item():.4f} std={stem_out.std().item():.4f} min={stem_out.min().item():.4f} max={stem_out.max().item():.4f}")

    e0 = model2.moe.experts[0](stem_out)
    e1 = model2.moe.experts[1](stem_out)
    e2 = model2.moe.experts[2](stem_out)
    print(f"expert0: mean={e0.mean().item():.6f} min={e0.min().item():.6f} max={e0.max().item():.6f}")
    print(f"expert1: mean={e1.mean().item():.6f} min={e1.min().item():.6f} max={e1.max().item():.6f}")
    print(f"expert2: mean={e2.mean().item():.6f} min={e2.min().item():.6f} max={e2.max().item():.6f}")

    route = model2.moe.gate(neck_out)
    rw = route["weights"]
    expert_outputs = torch.stack([e0, e1, e2], dim=1)
    fused2 = (expert_outputs * rw.unsqueeze(2)).sum(dim=1)
    print(f"fused moe: mean={fused2.mean().item():.6f} min={fused2.min().item():.6f} max={fused2.max().item():.6f}")

    pre_softplus2 = model2.density_head.proj(fused2)
    print(f"pre-softplus: min={pre_softplus2.min().item():.6f} max={pre_softplus2.max().item():.6f} mean={pre_softplus2.mean().item():.6f}")

    density2 = torch.nn.functional.softplus(pre_softplus2)
    density2_crop = _crop_density_for_eval(density2, targets[0], 8)
    print(f"density: min={density2_crop.min().item():.6f} max={density2_crop.max().item():.6f} mean={density2_crop.mean().item():.6f} sum={density2_crop.sum().item():.2f}")

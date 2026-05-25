"""Quantify background dominance in Bayesian loss at different bg_ratio values."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch
from crowdcount.models.moecount.losses import BayesianLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

H, W, stride = 16, 16, 8
image_size = (H * stride, W * stride)
n_patches, n_pts = 8, 3

# Create targets with random points (same for all tests)
targets = []
for i in range(n_patches):
    pts_x = torch.rand(n_pts, device=device) * image_size[1]
    pts_y = torch.rand(n_pts, device=device) * image_size[0]
    pts = torch.stack([pts_x, pts_y], dim=-1)
    targets.append({"point": pts})

# Test different bg_ratio and initial_density values
print("=== Effect of bg_ratio on gradient at initialization ===")
print(f"  sigma=8.0, {n_patches} patches, {n_pts} pts each, d=constant")
print()

for bg_ratio in [0.02, 0.05, 0.10, 0.15]:
    for density_val in [0.01, 0.05]:
        density = torch.full((n_patches, 1, H, W), density_val, device=device, requires_grad=True)

        loss_fn = BayesianLoss(sigma=8.0, use_background=True, bg_ratio=bg_ratio, count_loss_type="l1")
        loss = loss_fn(density, targets=targets, image_sizes=image_size)
        loss.backward()
        grad = density.grad.clone()
        density.grad = None

        density2 = torch.full((n_patches, 1, H, W), density_val, device=device, requires_grad=True)
        loss_fn_nobg = BayesianLoss(sigma=8.0, use_background=False, count_loss_type="l1")
        loss_nobg = loss_fn_nobg(density2, targets=targets, image_sizes=image_size)
        loss_nobg.backward()
        grad_nobg = density2.grad.clone()

        grad_mean = grad.mean().item()
        grad_nobg_mean = grad_nobg.mean().item()
        predicted_total = density_val * H * W

        print(f"  bg_ratio={bg_ratio:.2f}  d={density_val:.2f} (total/patch={predicted_total:.1f}): "
              f"loss={loss.item():.2f} grad_mean={grad_mean:+.6f}  "
              f"no_bg: loss={loss_nobg.item():.2f} grad={grad_nobg_mean:+.6f}")

print()
print("=== Per-pixel posterior breakdown (bg_ratio=0.15, d=0.01) ===")
# Show posterior distribution for one patch
bi = 0
pts = targets[bi]["point"].to(device=device, dtype=torch.float32)
gy = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
gx = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

sigma = 8.0
two_sigma_sq = 2.0 * sigma * sigma
bg_ratio = 0.15
bg_dist_sq = (bg_ratio * max(image_size[0], image_size[1])) ** 2

dist_sq = (coords.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(dim=-1)
min_dist_sq = dist_sq.min(dim=1, keepdim=True).values.clamp(min=0.0)
bg_dist_sq_vec = bg_dist_sq / (min_dist_sq + 1e-5)
dist_sq_ext = torch.cat([dist_sq, bg_dist_sq_vec], dim=1)

log_like = -dist_sq_ext / two_sigma_sq
log_like = log_like - log_like.max(dim=1, keepdim=True).values
likelihood = log_like.exp()
posterior = likelihood / likelihood.sum(dim=1, keepdim=True).clamp_min(1e-12)

# posterior[:, :3] = point posteriors, posterior[:, 3] = background
bg_posterior = posterior[:, -1]
pts_posterior = posterior[:, :3].sum(dim=1)

print(f"  Fraction of pixels with bg_posterior > 0.5: {(bg_posterior > 0.5).float().mean().item():.3f}")
print(f"  Fraction of pixels with bg_posterior > 0.8: {(bg_posterior > 0.8).float().mean().item():.3f}")
print(f"  Mean bg posterior: {bg_posterior.mean().item():.4f}")
print(f"  Mean point posterior (all 3): {pts_posterior.mean().item():.4f}")

# Gradient components
density = torch.full((1, 1, H, W), 0.01, device=device)
d = density[0, 0].reshape(-1)
K = 4  # 3 points + 1 bg
expected = torch.zeros(K, device=device)
for p_idx in range(coords.shape[0]):
    expected += posterior[p_idx] * d[p_idx]

print(f"\n  Expected counts: pts={expected[:3].tolist()} bg={expected[-1].item():.4f}")
print(f"  Point contribution to grad (sign=-1 for L1, expected<1): {-pts_posterior.mean().item():.4f}")
print(f"  BG contribution to grad (sign=+1 for L1): {bg_posterior.mean().item():.4f}")
print(f"  Net per-pixel gradient: {(-pts_posterior + bg_posterior).mean().item():+.6f}")
direction = "DOWN (toward zero)" if (-pts_posterior + bg_posterior).mean().item() > 0 else "UP (away from zero)"
print(f"  => Model is pushed {direction}")

print()
print("=== Summary ===")
print("  Background term dominates because at initialization, most pixels are far from any point.")
print("  The background 'virtual point' (at bg_ratio * max(H,W)) has higher posterior for far pixels.")
print("  Fix: reduce bg_ratio OR disable background during early training epochs.")

"""Isolate BayesianLoss gradient bug by scaling from 1 to 8 patches."""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/codersgl/sci-research/CrowdCounting-DSGCNet")
import torch, torch.nn as nn
from crowdcount.models.moecount.losses import BayesianLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

sigma = 8.0
use_bg = True
bg_ratio = 0.15

# Simulate density maps: each patch is 16x16 = 256 pixels
# Use constant density d=0.01 to match current init
H, W = 16, 16
n_pixels = H * W

# Realistic points: 3 per patch at random positions
n_pts_per_patch = 3

for n_patches in [1, 2, 4, 8]:
    print(f"\n{'='*60}")
    print(f"Testing {n_patches} patches, {n_pts_per_patch} points each")

    # Create density maps
    density = torch.full((n_patches, 1, H, W), 0.01, device=device)

    # Create targets with random points
    targets = []
    for i in range(n_patches):
        pts_x = torch.rand(n_pts_per_patch, device=device) * (W * 8)  # stride-8 coords
        pts_y = torch.rand(n_pts_per_patch, device=device) * (H * 8)
        pts = torch.stack([pts_x, pts_y], dim=-1)
        targets.append({"point": pts})

    image_size = (H * 8, W * 8)

    # --- Manual gradient computation ---
    density_manual = density.clone().detach().requires_grad_(True)
    loss_fn = BayesianLoss(sigma=sigma, use_background=use_bg, bg_ratio=bg_ratio, count_loss_type="l1")
    loss_module = loss_fn(density_manual, targets=targets, image_sizes=image_size)
    loss_module.backward()
    grad_module = density_manual.grad.clone()

    # --- Manual computation (per-sample) ---
    two_sigma_sq = 2.0 * sigma * sigma
    bg_dist_sq_val = (bg_ratio * max(image_size[0], image_size[1])) ** 2

    total_manual_grad = torch.zeros_like(density)
    for bi in range(n_patches):
        pts = targets[bi]["point"].to(device=device, dtype=torch.float32)
        d = density[bi, 0]  # [H, W]

        # Build coordinate grid
        gy = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * 8.0
        gx = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * 8.0
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        # Compute posterior
        dist_sq = (coords.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(dim=-1)
        if use_bg:
            min_dist_sq = dist_sq.min(dim=1, keepdim=True).values.clamp(min=0.0)
            bg_dist_sq = bg_dist_sq_val / (min_dist_sq + 1e-5)
            dist_sq_ext = torch.cat([dist_sq, bg_dist_sq], dim=1)
        else:
            dist_sq_ext = dist_sq

        log_like = -dist_sq_ext / two_sigma_sq
        log_like = log_like - log_like.max(dim=1, keepdim=True).values
        likelihood = log_like.exp()
        posterior = likelihood / likelihood.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # For L1 loss, gradient of |expected[p] - 1.0| w.r.t. each d[i] is:
        # sign(expected[p] - 1.0) * posterior[i, p]
        # Similarly for background: sign(expected[bg] - 0.0) * posterior[i, bg]

        N = pts.shape[0]
        K = N + 1 if use_bg else N  # +1 for background

        # Compute expected counts
        expected = torch.zeros(K, device=device, dtype=torch.float32)
        for p_idx in range(coords.shape[0]):
            expected += posterior[p_idx] * d.reshape(-1)[p_idx]

        # Compute gradient manually
        grad_flat = torch.zeros(n_pixels, device=device, dtype=torch.float32)
        for p_idx in range(N):
            diff = expected[p_idx] - 1.0
            sign_p = torch.sign(diff) if diff != 0 else torch.tensor(0.0, device=device)
            grad_flat += sign_p * posterior[:, p_idx]

        if use_bg:
            diff_bg = expected[-1] - 0.0
            sign_bg = torch.sign(diff_bg) if diff_bg != 0 else torch.tensor(0.0, device=device)
            grad_flat += sign_bg * posterior[:, -1]

        total_manual_grad[bi, 0] = grad_flat.reshape(H, W)

    # Compare
    diff = (grad_module - total_manual_grad).abs()
    max_diff = diff.max().item()
    rel_diff = diff.mean().item() / (grad_module.abs().mean().item() + 1e-8)

    # Also compute what the final gradient direction is
    module_mean_grad = grad_module.mean().item()
    manual_mean_grad = total_manual_grad.mean().item()

    print(f"  Module loss: {loss_module.item():.6f}")
    print(f"  Module grad mean: {module_mean_grad:.8f}  (negative = push DOWN)")
    print(f"  Manual grad mean: {manual_mean_grad:.8f}  (negative = push DOWN)")
    print(f"  Max |diff|: {max_diff:.8f},  Rel diff: {rel_diff:.8f}")

    if max_diff > 1e-6:
        print(f"  *** BUG DETECTED at {n_patches} patches! ***")
        # Print per-patch gradient means
        for bi in range(n_patches):
            mg = grad_module[bi, 0].mean().item()
            mg_man = total_manual_grad[bi, 0].mean().item()
            print(f"    Patch {bi}: module={mg:.8f} manual={mg_man:.8f}")
    else:
        print(f"  ✓ Module and manual match")

print("\nDone.")

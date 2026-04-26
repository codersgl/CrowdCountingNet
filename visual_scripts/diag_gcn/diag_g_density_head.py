"""Diag G: Density Head improvement candidates — offline data check.

Runs five questions on the cached 50-sample SHA test set:

  Q1  v1 ReLU truncation rate (TRAINED baseline)
      → How many pixels does the trained baseline head zero out?
      → If high, switching ReLU → Softplus is a free win.

  Q2  v1 vs v2 vs v3 head structural comparison (RANDOM-init for all three)
      → Sanity: dynamic range, saturation rate, output sparsity.
      → Even untrained, multi-scale receptive fields should yield richer
        spatial structure (measured by output spatial std + Sobel energy).

  Q3  Density-Feature alignment
      → Pearson(‖features_pa‖_per_pixel, density_pred_per_pixel)
      → If median is far below 1.0 (say <0.3), there's room for an
        alignment auxiliary loss.

  Q4  GT density sanity (sum vs head count)
      → Generate GT density on-the-fly via the project's own
        gaussian_filter_density.  sum(GT_density) should ≈ GT_count.
        Anything > 5 % off means the GT supervision is broken before we
        even train.

  Q5  Output resolution effect
      → Compare bilinear-upsampled (H/8 → H/4) baseline density with
        a sum-preserving GT density downsampled to H/8 and to H/4.
        Reports PSNR & SSIM at both scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from crowdcount.data.prepare import gaussian_filter_density
from crowdcount.models.head import Density_pred, Density_pred_MS, Density_pred_V3
from visual_scripts.diag_gcn.common import list_cache, load_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = a.norm() * b.norm() + 1e-8
    return float((a * b).sum().item() / denom.item())


def _sobel_energy(x: torch.Tensor) -> float:
    """Mean |grad| of x over its spatial dims; proxy for structural richness."""
    gx = x[..., 1:, :] - x[..., :-1, :]
    gy = x[..., :, 1:] - x[..., :, :-1]
    return float(gx.abs().mean().item() + gy.abs().mean().item())


def _ssim(a: torch.Tensor, b: torch.Tensor, win: int = 7) -> float:
    """Rough single-channel SSIM on FP32 tensors of identical shape [1,1,H,W]."""
    C1, C2 = 0.01**2, 0.03**2
    pad = win // 2
    mu_a = F.avg_pool2d(a, win, 1, pad)
    mu_b = F.avg_pool2d(b, win, 1, pad)
    sigma_a = F.avg_pool2d(a * a, win, 1, pad) - mu_a * mu_a
    sigma_b = F.avg_pool2d(b * b, win, 1, pad) - mu_b * mu_b
    sigma_ab = F.avg_pool2d(a * b, win, 1, pad) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
        (mu_a * mu_a + mu_b * mu_b + C1) * (sigma_a + sigma_b + C2)
    )
    return float(s.mean().item())


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse < 1e-12:
        return 100.0
    peak = max(float(a.max().item()), float(b.max().item()), 1e-6)
    return float(10.0 * np.log10(peak * peak / mse))


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_one(
    c: dict,
    head_v1: Density_pred,
    head_v2: Density_pred_MS,
    head_v3: Density_pred_V3,
    device: str,
) -> dict:
    feat = c["features_pa"].float().to(device)  # [1, 256, H/8, W/8]
    base_d = c["density"].to(device)            # [1, 1, H/8, W/8]  (trained v1)
    H, W = c["H"], c["W"]

    # ----- Q1 : ReLU truncation rate (trained baseline) -----
    trunc_rate = float((base_d <= 1e-6).float().mean().item())
    base_min = float(base_d.min().item())
    base_max = float(base_d.max().item())
    base_mean = float(base_d.mean().item())
    base_std = float(base_d.std().item())

    # ----- Q2 : structural comparison v1 / v2 / v3 (all random init) -----
    d1 = head_v1(feat)
    d2 = head_v2(feat)
    d3 = head_v3(feat)
    metrics = {
        "sob_v1_rand": _sobel_energy(d1),
        "sob_v2_rand": _sobel_energy(d2),
        "sob_v3_rand": _sobel_energy(d3),
        "std_v1_rand": float(d1.std().item()),
        "std_v2_rand": float(d2.std().item()),
        "std_v3_rand": float(d3.std().item()),
        "zero_v1_rand": float((d1 <= 1e-6).float().mean().item()),
        "zero_v2_rand": float((d2 <= 1e-6).float().mean().item()),
        "zero_v3_rand": float((d3 <= 1e-6).float().mean().item()),
    }

    # ----- Q3 : density-feature alignment (uses TRAINED baseline density) -----
    feat_norm = feat.norm(dim=1, keepdim=True)  # [1,1,H8,W8]
    align_corr = _pearson(feat_norm.flatten(), base_d.flatten())

    # ----- Q4 : GT density sum sanity -----
    gt = np.asarray(c["gt_points"], dtype=np.float32)
    gt_count = int(len(gt))
    img_dummy = np.zeros((H, W), dtype=np.float32)
    gt_density_full = gaussian_filter_density(img_dummy, gt)  # [H, W]
    gt_sum = float(gt_density_full.sum())
    gt_sum_err = (gt_sum - gt_count) / max(gt_count, 1)

    # ----- Q5 : output resolution effect (compare *shapes*, not magnitude) -----
    gt_t = torch.from_numpy(gt_density_full).to(device).view(1, 1, H, W)
    # Sum-preserving downsample: avg-pool * area
    gt_h8 = F.avg_pool2d(gt_t, kernel_size=8, stride=8) * 64.0
    gt_h4 = F.avg_pool2d(gt_t, kernel_size=4, stride=4) * 16.0

    target_cnt = max(gt_sum, 1.0)

    def rescale_to_count(d: torch.Tensor) -> torch.Tensor:
        s = d.sum().clamp_min(1e-6)
        return d * (target_cnt / s)

    base_h8_norm = rescale_to_count(base_d)
    base_h4_pred = F.interpolate(
        base_d, scale_factor=2, mode="bilinear", align_corners=False
    )
    base_h4_norm = rescale_to_count(base_h4_pred)

    return {
        "name": c["name"],
        "gt_count": gt_count,
        "trunc_rate_v1_trained": trunc_rate,
        "base_min": base_min,
        "base_max": base_max,
        "base_mean": base_mean,
        "base_std": base_std,
        **metrics,
        "align_corr": align_corr,
        "gt_sum": gt_sum,
        "gt_sum_err": gt_sum_err,
        "psnr_h8": _psnr(base_h8_norm, gt_h8),
        "psnr_h4": _psnr(base_h4_norm, gt_h4),
        "ssim_h8": _ssim(base_h8_norm, gt_h8),
        "ssim_h4": _ssim(base_h4_norm, gt_h4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diag_cache/diag_g.csv"),
    )
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[diag_g] {len(names)} samples on {device}")

    torch.manual_seed(0)
    head_v1 = Density_pred().to(device).eval()
    head_v2 = Density_pred_MS().to(device).eval()
    head_v3 = Density_pred_V3().to(device).eval()

    rows = []
    for n in tqdm(names, desc="diag_g"):
        c = load_cache(n)
        rows.append(run_one(c, head_v1, head_v2, head_v3, device))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with args.out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                    for k in keys
                )
                + "\n"
            )

    def avg(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    def med(k: str) -> float:
        return float(np.median([r[k] for r in rows]))

    print()
    print("=== Q1  Trained-v1 baseline density properties ===")
    print(
        f"  ReLU-truncated pixels (==0)  mean={avg('trunc_rate_v1_trained'):.4f}  "
        f"median={med('trunc_rate_v1_trained'):.4f}"
    )
    print(f"  output min   mean={avg('base_min'):.4e}")
    print(f"  output max   mean={avg('base_max'):.4e}")
    print(f"  output mean  mean={avg('base_mean'):.4e}")
    print(f"  output std   mean={avg('base_std'):.4e}")
    if avg("trunc_rate_v1_trained") > 0.10:
        print("  [verdict] >10% pixels truncated → Softplus replacement likely helps.")
    else:
        print("  [verdict] truncation rate small; Softplus expected ≈ neutral.")

    print()
    print("=== Q2  Random-init v1 / v2 / v3 structural richness ===")
    print(f"  Sobel energy (higher = more structure)")
    print(
        f"    v1_rand={avg('sob_v1_rand'):.4e}   "
        f"v2_rand={avg('sob_v2_rand'):.4e}   "
        f"v3_rand={avg('sob_v3_rand'):.4e}"
    )
    print(f"  Spatial std")
    print(
        f"    v1_rand={avg('std_v1_rand'):.4e}   "
        f"v2_rand={avg('std_v2_rand'):.4e}   "
        f"v3_rand={avg('std_v3_rand'):.4e}"
    )
    print(f"  Zero-output fraction (Softplus → ~0 expected)")
    print(
        f"    v1_rand={avg('zero_v1_rand'):.4f}   "
        f"v2_rand={avg('zero_v2_rand'):.4f}   "
        f"v3_rand={avg('zero_v3_rand'):.4f}"
    )
    if avg("zero_v1_rand") > 0.30:
        print(
            "  [verdict] v1 ReLU drops a large fraction of activations even "
            "at init → Softplus removes this dead zone."
        )

    print()
    print("=== Q3  Density-feature alignment (Pearson ‖feat‖ vs density) ===")
    print(
        f"  align_corr   mean={avg('align_corr'):+.4f}   "
        f"median={med('align_corr'):+.4f}"
    )
    if med("align_corr") < 0.30:
        print(
            "  [verdict] median<0.30 → adding L_align = 1 - corr "
            "(weight ~1e-3) is well-motivated."
        )

    print()
    print("=== Q4  GT density sum vs ground-truth count ===")
    abs_err = float(np.mean([abs(r["gt_sum_err"]) for r in rows]))
    print(
        f"  sum/count error   mean={avg('gt_sum_err'):+.4f}   "
        f"median={med('gt_sum_err'):+.4f}   abs-mean={abs_err:.4f}"
    )
    if abs(avg("gt_sum_err")) > 0.05:
        print(
            "  [verdict] |error|>5% → density-map normalisation is biased "
            "(check sigma / kernel cropping in prepare.py)."
        )
    else:
        print("  [verdict] GT density sum ≈ count (within 5%), no fix needed.")

    print()
    print("=== Q5  Output-resolution effect (baseline rescaled to GT count) ===")
    print(f"  PSNR  H/8 = {avg('psnr_h8'):.2f} dB     H/4 = {avg('psnr_h4'):.2f} dB")
    print(f"  SSIM  H/8 = {avg('ssim_h8'):.4f}        H/4 = {avg('ssim_h4'):.4f}")
    if avg("ssim_h4") > avg("ssim_h8") + 0.02:
        print("  [verdict] H/4 SSIM ≥ H/8 + 0.02 → double-resolution head worth a try.")
    else:
        print(
            "  [verdict] H/4 not meaningfully better than H/8 — bilinear-upsample "
            "alone won't help; would need a learned PixelShuffle path."
        )

    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()

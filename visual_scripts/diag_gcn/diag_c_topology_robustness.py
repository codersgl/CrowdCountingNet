"""Diag C — Topology robustness to density noise.

Tests whether the current k-NN graph is fragile when the density head's
output is noisy. If yes, a learnable / differentiable topology (DGM) can help.

Procedure for each sample:
  1. Build current density-graph from clean density.
  2. Build N noisy variants with Gaussian noise σ ∈ {0.05, 0.1, 0.2}.
  3. Measure:
     - edge_iou : |E_clean ∩ E_noisy| / |E_clean ∪ E_noisy|
     - feat_drift : mean cosine distance between
                     mean-aggregated features (clean vs noisy graph topology)
                     using IDENTICAL input features (so drift comes purely
                     from topology change, not feature change).

C is value-worth-doing if at σ=0.1:
  edge_iou < 60%  AND  feat_drift > 0.10

Usage:
    uv run python -m visual_scripts.diag_gcn.diag_c_topology_robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visual_scripts.diag_gcn.diag_a_spatial_prior import density_topk_graph
from visual_scripts.diag_gcn.common import list_cache, load_cache


def edge_iou(e1: torch.Tensor, e2: torch.Tensor, N: int) -> float:
    """Treat edges as ordered pairs, IoU = |intersection| / |union|."""
    s1 = set((int(a), int(b)) for a, b in zip(e1[0].tolist(), e1[1].tolist()))
    s2 = set((int(a), int(b)) for a, b in zip(e2[0].tolist(), e2[1].tolist()))
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / max(1, len(s1 | s2))


def mean_aggregate(edge_index: torch.Tensor, x: torch.Tensor, N: int) -> torch.Tensor:
    """Parameter-free GCN proxy: h_i' = mean of x_j over neighbours j of i."""
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    deg = torch.zeros(N, device=x.device)
    out.index_add_(0, src, x[dst])
    deg.index_add_(0, src, torch.ones(src.shape[0], device=x.device))
    deg = deg.clamp(min=1.0).unsqueeze(-1)
    return out / deg


def cosine_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
    return float((1 - cos.mean()).item())


def run_one(
    c: dict,
    k: int = 4,
    sigmas=(0.05, 0.1, 0.2),
    n_trials: int = 3,
    device: str = "cuda",
) -> dict:
    feat = c["features_pa"].float().to(device)
    dens = c["density"].to(device)
    _, C, H, W = feat.shape
    N = H * W
    feat_flat = feat.permute(0, 2, 3, 1).reshape(N, C)
    dens_flat = dens.reshape(N)
    dens_std = float(dens_flat.std().item()) + 1e-8

    e_clean, _ = density_topk_graph(dens_flat, k=k)
    h_clean = mean_aggregate(e_clean, feat_flat, N)

    out = {"name": c["name"], "N": N}
    for sigma in sigmas:
        ious = []
        drifts = []
        for t in range(n_trials):
            noise = torch.randn_like(dens_flat) * sigma * dens_std
            e_noisy, _ = density_topk_graph(dens_flat + noise, k=k)
            ious.append(edge_iou(e_clean, e_noisy, N))
            h_noisy = mean_aggregate(e_noisy, feat_flat, N)
            drifts.append(cosine_drift(h_clean, h_noisy))
        out[f"iou_s{sigma}"] = float(np.mean(ious))
        out[f"drift_s{sigma}"] = float(np.mean(drifts))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/diag_cache/diag_c.csv")
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[diag_c] {len(names)} samples on {device}")

    rows = []
    for n in tqdm(names, desc="diag_c"):
        c = load_cache(n)
        rows.append(run_one(c, k=args.k, device=device))

    keys = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def agg(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in rows], dtype=np.float32)
        return float(v.mean()), float(v.std())

    print("\n=== Diag C: Topology Robustness ===")
    print("    σ      edge_IoU       feat_drift     verdict")
    fragile_at_01 = False
    for s in (0.05, 0.1, 0.2):
        iou_m, iou_s = agg(f"iou_s{s}")
        drift_m, drift_s = agg(f"drift_s{s}")
        flag = ""
        if s == 0.1:
            fragile = iou_m < 0.6 and drift_m > 0.10
            fragile_at_01 = fragile
            flag = "  ← KEY" + ("  ✓ fragile" if fragile else "  ✗ stable")
        print(
            f"  {s:5.2f}  {iou_m:.3f}±{iou_s:.3f}   {drift_m:.3f}±{drift_s:.3f}{flag}"
        )

    print()
    print(
        f"  Verdict: DGM (learnable topology) is {'WORTH IT' if fragile_at_01 else 'overkill — current k-NN is robust enough'}"
    )


if __name__ == "__main__":
    main()

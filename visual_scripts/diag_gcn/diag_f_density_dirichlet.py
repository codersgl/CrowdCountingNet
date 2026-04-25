"""Diag F — Density Dirichlet energy magnitude (regulariser sizing).

Computes per-image Dirichlet energy of the predicted density map and
of the GCN feature output (mean-aggregated) so we can size a Dirichlet
regulariser loss term: λ · Σ ‖h_i - h_j‖² along k-NN edges.

Reports magnitude of: density Dirichlet, feature Dirichlet — these tell
us the natural scale and thus the right loss coefficient λ such that
λ·E ≈ 0.01–0.1 × main_loss (typical density loss ≈ 1–10).

Usage:
    uv run python -m visual_scripts.diag_gcn.diag_f_density_dirichlet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from visual_scripts.diag_gcn.diag_a_spatial_prior import density_topk_graph
from visual_scripts.diag_gcn.diag_c_topology_robustness import mean_aggregate
from visual_scripts.diag_gcn.common import list_cache, load_cache


def four_neighbour_edge_index(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return 4-neighbour edge_index [2, E] for an H×W grid (no self-loops)."""
    src, dst = [], []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for y in range(H):
            for x in range(W):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    src.append(y * W + x)
                    dst.append(yy * W + xx)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/diag_cache/diag_f.csv")
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[diag_f] {len(names)} samples on {device}")

    rows = []
    for n in tqdm(names, desc="diag_f"):
        c = load_cache(n)
        feat = c["features_pa"].float().to(device)
        dens = c["density"].to(device)
        _, C, H, W = feat.shape
        N = H * W
        feat_flat = feat.permute(0, 2, 3, 1).reshape(N, C)
        dens_flat = dens.reshape(N)

        # 4-neighbour edges for density (continuity prior — image-adjacent pixels)
        e4 = four_neighbour_edge_index(H, W, device=feat.device)
        # k-NN density graph for feature (over current GCN topology)
        ek, _ = density_topk_graph(dens_flat, k=args.k)

        # Density Dirichlet (per-edge mean)
        d_e4 = (dens_flat[e4[0]] - dens_flat[e4[1]]).pow(2).mean().item()
        # Feature mean-aggregated, then Dirichlet
        h = mean_aggregate(ek, feat_flat, N)
        f_ek = (h[ek[0]] - h[ek[1]]).pow(2).sum(-1).mean().item()
        f_e4 = (feat_flat[e4[0]] - feat_flat[e4[1]]).pow(2).sum(-1).mean().item()

        rows.append(
            {
                "name": c["name"],
                "N": N,
                "density_sum": float(dens_flat.sum().item()),
                "gt_count": int(len(c["gt_points"])),
                "dirichlet_density_4nbr": float(d_e4),
                "dirichlet_feat_knn_mean_agg": float(f_ek),
                "dirichlet_feat_4nbr": float(f_e4),
            }
        )

    keys = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def agg(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return float(v.mean()), float(np.median(v))

    print("\n=== Diag F: Dirichlet Magnitudes ===")
    print(f"  samples = {len(rows)}")
    print()
    for k in [
        "dirichlet_density_4nbr",
        "dirichlet_feat_4nbr",
        "dirichlet_feat_knn_mean_agg",
    ]:
        m, med = agg(k)
        print(f"  {k:34s} mean={m:.5g}  median={med:.5g}")

    print()
    print("  Recommended Dirichlet regulariser λ ≈ 0.01 / median(dirichlet_feat_4nbr)")
    _, med = agg("dirichlet_feat_4nbr")
    if med > 0:
        print(f"   → λ ≈ {0.01 / med:.4g}  (loss term contributes ~0.01)")


if __name__ == "__main__":
    main()

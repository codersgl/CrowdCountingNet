"""Diag E — Over-smoothing under increasing GCN depth.

Tests whether stacking more GCN layers collapses node representations
(over-smoothing), and how strongly dropout p ∈ {0.5, 0.2, 0.1} interacts.

Metrics per sample (lower = more smoothing):
  - dirichlet_ratio : Dir(h_L) / Dir(h_0) measured on the same edge set,
        proper over-smoothing indicator (insensitive to dropout's variance).
  - density_corr : Pearson corr( ρ(node), ‖h_i‖ ) — preserved density signal.

Layers tested: L ∈ {1, 2, 4, 8} on the current density k-NN graph.

E is value-worth-doing if:
  (a) at L=2, dropout=0.5 reduces MAD by > 30 % vs dropout=0.1
  (b) at L=4 or L=8, MAD with current dropout=0.5 collapses (< 0.05);
      switching to GCNII-style residual or lower dropout keeps MAD high.

Usage:
    uv run python -m visual_scripts.diag_gcn.diag_e_oversmoothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visual_scripts.diag_gcn.diag_a_spatial_prior import density_topk_graph
from visual_scripts.diag_gcn.diag_c_topology_robustness import mean_aggregate
from visual_scripts.diag_gcn.common import list_cache, load_cache


def mad(x: torch.Tensor, sample: int = 2000) -> float:
    """Mean pairwise cosine distance, sub-sampled for tractability."""
    n = x.shape[0]
    if n > sample:
        idx = torch.randperm(n, device=x.device)[:sample]
        x = x[idx]
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.t()
    sim.fill_diagonal_(0.0)
    return float((1 - sim).mean().item())


def dirichlet(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    src, dst = edge_index[0], edge_index[1]
    diff = (x[src] - x[dst]).pow(2).sum(-1).mean()
    return float(diff.item())


def gcnii_propagate(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    L: int,
    alpha: float = 0.1,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Approximate GCNII: h_l = (1-α)·Â·h_(l-1) + α·h_0, with dropout per layer."""
    h0 = x
    h = x
    N = x.shape[0]
    for _ in range(L):
        h_agg = mean_aggregate(edge_index, h, N)
        h = (1 - alpha) * h_agg + alpha * h0
        if dropout > 0:
            h = F.dropout(h, p=dropout, training=True)
    return h


def vanilla_propagate(
    edge_index: torch.Tensor, x: torch.Tensor, L: int, dropout: float = 0.0
) -> torch.Tensor:
    """Vanilla GCN: h_l = Â · h_(l-1), no residual."""
    h = x
    N = x.shape[0]
    for _ in range(L):
        h = mean_aggregate(edge_index, h, N)
        if dropout > 0:
            h = F.dropout(h, p=dropout, training=True)
    return h


def density_corr(h: torch.Tensor, density: torch.Tensor) -> float:
    """Pearson correlation between ‖h_i‖ and density at node i.

    Dropout-robust because we measure node norms, not raw values.
    """
    norm = h.norm(dim=-1)
    a = norm - norm.mean()
    b = density - density.mean()
    den = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).clamp(min=1e-8)
    return float((a * b).sum().div(den).item())


def run_one(c: dict, k: int = 4, device: str = "cuda") -> dict:
    feat = c["features_pa"].float().to(device)
    dens = c["density"].to(device)
    _, C, H, W = feat.shape
    N = H * W
    feat_flat = feat.permute(0, 2, 3, 1).reshape(N, C)
    dens_flat = dens.reshape(N)
    e, _ = density_topk_graph(dens_flat, k=k)

    dir0 = dirichlet(feat_flat, e)
    corr0 = density_corr(feat_flat, dens_flat)
    out = {"name": c["name"], "N": N, "dir_input": dir0, "corr_input": corr0}

    for L in (1, 2, 4, 8):
        for dr in (0.0, 0.1, 0.2, 0.5):
            h = vanilla_propagate(e, feat_flat, L=L, dropout=dr)
            out[f"vanilla_L{L}_d{dr}_dirratio"] = dirichlet(h, e) / max(dir0, 1e-8)
            out[f"vanilla_L{L}_d{dr}_corr"] = density_corr(h, dens_flat)
        # GCNII at default dropout 0.1 only
        h = gcnii_propagate(e, feat_flat, L=L, dropout=0.1)
        out[f"gcnii_L{L}_dirratio"] = dirichlet(h, e) / max(dir0, 1e-8)
        out[f"gcnii_L{L}_corr"] = density_corr(h, dens_flat)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/diag_cache/diag_e.csv")
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[diag_e] {len(names)} samples on {device}")

    rows = []
    for n in tqdm(names, desc="diag_e"):
        c = load_cache(n)
        rows.append(run_one(c, k=args.k, device=device))

    keys = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def agg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    print("\n=== Diag E: Over-smoothing & Dropout ===")
    print(f"  baseline density-corr (input features) = {agg('corr_input'):.4f}\n")

    print("  Vanilla GCN — Dirichlet RATIO Dir(h_L)/Dir(h_0)")
    print("  (lower = smoother;  smoothing is bad once < ~0.05)")
    print("       drop=0.0   drop=0.1   drop=0.2   drop=0.5")
    for L in (1, 2, 4, 8):
        cells = [agg(f"vanilla_L{L}_d{dr}_dirratio") for dr in (0.0, 0.1, 0.2, 0.5)]
        print(f"  L={L}  " + "  ".join(f"{v:8.4f}" for v in cells))

    print("\n  Vanilla GCN — density correlation (preserved counting signal)")
    print("       drop=0.0   drop=0.1   drop=0.2   drop=0.5")
    for L in (1, 2, 4, 8):
        cells = [agg(f"vanilla_L{L}_d{dr}_corr") for dr in (0.0, 0.1, 0.2, 0.5)]
        print(f"  L={L}  " + "  ".join(f"{v:+8.4f}" for v in cells))

    print("\n  GCNII (initial-residual α=0.1, dropout=0.1)")
    print("       L=1       L=2       L=4       L=8")
    cells_dr = [agg(f"gcnii_L{L}_dirratio") for L in (1, 2, 4, 8)]
    cells_co = [agg(f"gcnii_L{L}_corr") for L in (1, 2, 4, 8)]
    print("  Dir   " + "  ".join(f"{v:7.4f}" for v in cells_dr))
    print("  Corr  " + "  ".join(f"{v:+7.4f}" for v in cells_co))

    print()
    # Verdicts
    drop_corr_gap = agg("vanilla_L2_d0.1_corr") - agg("vanilla_L2_d0.5_corr")
    deep_corr_van = agg("vanilla_L8_d0.5_corr")
    deep_corr_gcnii = agg("gcnii_L8_corr")
    base_corr = agg("corr_input")

    print(
        f"  [verdict] dropout 0.1 vs 0.5 (L=2) corr-gap = {drop_corr_gap:+.4f} "
        f"({'lower dropout WINS' if drop_corr_gap > 0.02 else 'no clear winner'})"
    )
    print(
        f"  [verdict] L=8 vanilla d=0.5 corr = {deep_corr_van:+.4f} (input {base_corr:+.4f})"
    )
    print(
        f"  [verdict] L=8 GCNII   corr = {deep_corr_gcnii:+.4f}  "
        f"({'GCNII preserves signal' if deep_corr_gcnii > deep_corr_van + 0.02 else 'no clear gain'})"
    )


if __name__ == "__main__":
    main()

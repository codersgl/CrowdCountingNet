"""Diag A — Spatial geometric prior in graph construction.

Tests whether the current density-distance / feature-similarity k-NN graphs
contain too many *long-range* edges that hurt homophily.

Metrics (per cached sample, then aggregated):
  - long_range_frac : edges with spatial dist > P75
  - homophily_density : fraction of edges connecting nodes in same density bin
  - homophily_density_spatial : same, but graph rebuilt with mixed cost
                                  (density + spatial)
  - homophily_feature : feature-graph homophily under density bin
  - homophily_feature_spatial : feature graph rebuilt with mixed cost

A is "value worth doing" if:
  (a) long_range_frac >= 15%
  (b) homophily improvement >= 3 pp under spatial-aware graph

Usage:
    uv run python -m visual_scripts.diag_gcn.diag_a_spatial_prior
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visual_scripts.diag_gcn.common import (
    density_bins,
    homophily_ratio,
    list_cache,
    load_cache,
)


def _coords(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return [H*W, 2] (y, x) coords."""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # [N, 2]


def density_topk_graph(
    density_flat: torch.Tensor, k: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph by abs density distance (matches DensityGraphBuilder).

    Args:
        density_flat: [N] density values.
        k: neighbours per node.

    Returns:
        edge_index [2, N*k], edge_dist (spatial-agnostic) [N*k].
    """
    N = density_flat.shape[0]
    dist = (density_flat.unsqueeze(0) - density_flat.unsqueeze(1)).abs()
    nb = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]  # [N, k]
    src = (
        torch.arange(N, device=density_flat.device)
        .unsqueeze(1)
        .expand(N, k)
        .reshape(-1)
    )
    dst = nb.reshape(-1)
    return torch.stack([src, dst], dim=0), dist.gather(1, nb).reshape(-1)


def feature_topk_graph(
    feature_flat: torch.Tensor, k: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph by cosine similarity (matches FeatureGraphBuilder)."""
    norm = F.normalize(feature_flat, p=2, dim=-1)
    sim = norm @ norm.t()  # [N, N]
    sim.fill_diagonal_(-2.0)
    nb = torch.topk(sim, k=k, largest=True).indices  # [N, k]
    N = feature_flat.shape[0]
    src = (
        torch.arange(N, device=feature_flat.device)
        .unsqueeze(1)
        .expand(N, k)
        .reshape(-1)
    )
    dst = nb.reshape(-1)
    return torch.stack([src, dst], dim=0), sim.gather(1, nb).reshape(-1)


def density_spatial_topk_graph(
    density_flat: torch.Tensor,
    coords: torch.Tensor,
    k: int = 4,
    spatial_sigma: float | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Mixed-cost graph: argmin( alpha*|Δd|/scale_d + beta*‖Δp‖/sigma )."""
    N = density_flat.shape[0]
    d_dist = (density_flat.unsqueeze(0) - density_flat.unsqueeze(1)).abs()
    p_dist = torch.cdist(coords, coords, p=2.0)
    if spatial_sigma is None:
        spatial_sigma = float(p_dist.median().item())
    d_scale = float(d_dist.median().item()) + 1e-6
    cost = alpha * d_dist / d_scale + beta * p_dist / spatial_sigma
    nb = torch.topk(cost, k=k + 1, largest=False).indices[:, 1:]
    src = (
        torch.arange(N, device=density_flat.device)
        .unsqueeze(1)
        .expand(N, k)
        .reshape(-1)
    )
    dst = nb.reshape(-1)
    return torch.stack([src, dst], dim=0)


def feature_spatial_topk_graph(
    feature_flat: torch.Tensor,
    coords: torch.Tensor,
    k: int = 4,
    spatial_sigma: float | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Mixed-cost graph: argmax( alpha*sim - beta*‖Δp‖/sigma )."""
    norm = F.normalize(feature_flat, p=2, dim=-1)
    sim = norm @ norm.t()
    p_dist = torch.cdist(coords, coords, p=2.0)
    if spatial_sigma is None:
        spatial_sigma = float(p_dist.median().item())
    score = alpha * sim - beta * p_dist / spatial_sigma
    score.fill_diagonal_(-1e9)
    nb = torch.topk(score, k=k, largest=True).indices
    N = feature_flat.shape[0]
    src = (
        torch.arange(N, device=feature_flat.device)
        .unsqueeze(1)
        .expand(N, k)
        .reshape(-1)
    )
    dst = nb.reshape(-1)
    return torch.stack([src, dst], dim=0)


def edge_spatial_dist(edge_index: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    return (coords[edge_index[0]] - coords[edge_index[1]]).norm(dim=-1)


def run_one(c: dict, k: int = 4, beta: float = 1.0, device: str = "cuda") -> dict:
    feat = c["features_pa"].float().to(device)  # [1, 256, H, W]
    dens = c["density"].to(device)  # [1, 1, H, W]
    _, C, H, W = feat.shape
    N = H * W
    feat_flat = feat.permute(0, 2, 3, 1).reshape(N, C)
    dens_flat = dens.reshape(N)

    coords = _coords(H, W, feat.device)
    img_diag = float(np.sqrt(H * H + W * W))

    # GT-cell label: 1 if any GT point falls in feature cell, else 0
    gt = c["gt_points"]
    fy = np.clip(gt[:, 1] / 8, 0, H - 1).astype(np.int64)
    fx = np.clip(gt[:, 0] / 8, 0, W - 1).astype(np.int64)
    gt_mask = np.zeros(N, dtype=np.int64)
    gt_mask[fy * W + fx] = 1
    gt_label = torch.from_numpy(gt_mask).to(feat.device)

    bins = density_bins(dens_flat.cpu().numpy(), n_bins=3)
    bin_label = torch.from_numpy(bins).to(feat.device)

    def edge_stats(e: torch.Tensor) -> tuple[float, float, float]:
        """Return (mean_dist_frac, p95_dist_frac, frac_long).

        - mean_dist_frac : mean spatial dist / image diagonal
        - p95_dist_frac  : 95th percentile, same units
        - frac_long      : fraction with dist > 0.25 × diagonal
        """
        s = edge_spatial_dist(e, coords)
        return (
            float(s.mean().item()) / img_diag,
            float(s.quantile(0.95).item()) / img_diag,
            float((s > 0.25 * img_diag).float().mean().item()),
        )

    # ---- current density graph ----
    e_d, _ = density_topk_graph(dens_flat, k=k)
    d_mean, d_p95, d_long = edge_stats(e_d)
    h_d_gt = homophily_ratio(e_d, gt_label)
    h_d_bin = homophily_ratio(
        e_d, bin_label
    )  # near 1.0 trivially; kept as sanity check

    # ---- current feature graph ----
    e_f, _ = feature_topk_graph(feat_flat, k=k)
    f_mean, f_p95, f_long = edge_stats(e_f)
    h_f_gt = homophily_ratio(e_f, gt_label)
    h_f_bin = homophily_ratio(e_f, bin_label)

    # ---- spatial-aware density graph ----
    e_d_sp = density_spatial_topk_graph(dens_flat, coords, k=k, beta=beta)
    d_mean_sp, _, d_long_sp = edge_stats(e_d_sp)
    h_d_gt_sp = homophily_ratio(e_d_sp, gt_label)
    h_d_bin_sp = homophily_ratio(e_d_sp, bin_label)

    # ---- spatial-aware feature graph ----
    e_f_sp = feature_spatial_topk_graph(feat_flat, coords, k=k, beta=beta)
    f_mean_sp, _, f_long_sp = edge_stats(e_f_sp)
    h_f_gt_sp = homophily_ratio(e_f_sp, gt_label)
    h_f_bin_sp = homophily_ratio(e_f_sp, bin_label)

    return {
        "name": c["name"],
        "N": N,
        "img_diag": img_diag,
        # current graphs
        "edge_dist_d_mean_frac": d_mean,
        "edge_dist_d_p95_frac": d_p95,
        "long_range_d": d_long,
        "edge_dist_f_mean_frac": f_mean,
        "edge_dist_f_p95_frac": f_p95,
        "long_range_f": f_long,
        # spatial-aware graphs
        "edge_dist_d_sp_mean_frac": d_mean_sp,
        "long_range_d_sp": d_long_sp,
        "edge_dist_f_sp_mean_frac": f_mean_sp,
        "long_range_f_sp": f_long_sp,
        # homophily
        "h_d_bin": h_d_bin,
        "h_d_bin_sp": h_d_bin_sp,
        "h_d_gt": h_d_gt,
        "h_d_gt_sp": h_d_gt_sp,
        "h_f_bin": h_f_bin,
        "h_f_bin_sp": h_f_bin_sp,
        "h_f_gt": h_f_gt,
        "h_f_gt_sp": h_f_gt_sp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="weight of spatial-distance term in mixed cost",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/diag_cache/diag_a.csv")
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[diag_a] {len(names)} samples on {device}, beta={args.beta}")

    rows = []
    for n in tqdm(names, desc="diag_a"):
        c = load_cache(n)
        rows.append(run_one(c, k=args.k, beta=args.beta, device=device))

    keys = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---- Aggregate ----
    def agg(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in rows], dtype=np.float32)
        return float(v.mean()), float(v.std())

    print("\n=== Diag A: Spatial Prior ===")
    print(f"  k={args.k}  beta={args.beta}  samples={len(rows)}")
    print(f"  feat-grid diagonal (avg) : {agg('img_diag')[0]:.1f} px\n")

    print("  Edge spatial-distance / image-diagonal")
    print(
        "  graph         current_mean   current_p95   long_range(>0.25D)   spatial_mean   spatial_long"
    )
    for tag in ("d", "f"):
        cm, _ = agg(f"edge_dist_{tag}_mean_frac")
        cp, _ = agg(f"edge_dist_{tag}_p95_frac")
        cl, _ = agg(f"long_range_{tag}")
        sm, _ = agg(f"edge_dist_{tag}_sp_mean_frac")
        sl, _ = agg(f"long_range_{tag}_sp")
        name = "density-kNN " if tag == "d" else "feature-kNN "
        print(
            f"  {name}      {cm:6.3f}        {cp:6.3f}       {cl * 100:5.2f}%             "
            f"{sm:6.3f}         {sl * 100:5.2f}%"
        )

    print(
        "\n  Homophily under (a) density-bin label  (b) GT-cell label  — current → spatial-aware"
    )
    for tag in ["h_d_bin", "h_d_gt", "h_f_bin", "h_f_gt"]:
        m_cur, _ = agg(tag)
        m_sp, _ = agg(tag + "_sp")
        delta = (m_sp - m_cur) * 100  # pp
        flag = "+" if delta >= 3.0 else ("-" if delta <= -3.0 else " ")
        print(f"  [{flag}] {tag:10s}: {m_cur:.3f} → {m_sp:.3f}   (Δ = {delta:+.2f} pp)")

    print("\n  Interpretation:")
    print(
        "   - long_range fraction > 15%  → current graph spans large image area (geometric prior helps)."
    )
    print(
        "   - h_*_gt rising under spatial aware  → connections to GT-cells stay informative."
    )
    print("   - h_d_bin staying ~1.0  → density k-NN trivially homophilous (sanity).")


if __name__ == "__main__":
    main()

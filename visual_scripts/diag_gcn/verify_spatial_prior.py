"""Verify SpatialPriorDensityGraphBuilder on cached SHA features.

Compares the *production* graph builder (`crowdcount.models.gcn`) against
the baseline `DensityGraphBuilder` on the same 50-sample cache used by Diag A.

For each sample we measure on the produced graph (excluding self-loops):
  - long_range  : fraction of edges with spatial dist > 0.25 * image diag
  - mean_dist   : mean edge spatial dist / image diag
  - h_bin       : edge homophily under 3-bin density labels (sanity)
  - h_gt        : edge homophily under "GT cell vs background" labels
  - corr_drop   : density-correlation drop after one parameter-free
                  mean-aggregation step  (lower is better — means the
                  per-node norm still tracks density, i.e. the graph is
                  not pulling each node toward a far-away dissimilar
                  region).  computed as
                      corr(in) - corr(agg)
                  where corr(.) = Pearson(‖h_i‖, density_i).

A useful prior should reduce long_range, increase h_bin / h_gt
(or keep them flat), and reduce corr_drop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from crowdcount.models.gcn import (
    DensityGraphBuilder,
    SpatialPriorDensityGraphBuilder,
)
from visual_scripts.diag_gcn.common import (
    density_bins,
    homophily_ratio,
    list_cache,
    load_cache,
)


def _coords(H: int, W: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def _strip_self_loops(edge_index: torch.Tensor, num_nodes_total: int) -> torch.Tensor:
    """Remove the trailing self-loops appended by the builders."""
    return edge_index[:, : edge_index.shape[1] - num_nodes_total]


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm() + 1e-8
    return float((a * b).sum().item() / denom.item())


@torch.no_grad()
def _mean_aggregate(feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Parameter-free symmetric mean aggregation for one hop.

    out_i = (h_i + mean_{j∈N(i)} h_j) / 2
    """
    N = feat.shape[0]
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(N, device=feat.device, dtype=feat.dtype)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=feat.dtype))
    deg = deg.clamp_min(1.0)
    sum_nb = torch.zeros_like(feat)
    sum_nb.index_add_(0, src, feat[dst])
    return 0.5 * (feat + sum_nb / deg.unsqueeze(1))


def run_one(c: dict, k: int, device: str) -> dict:
    feat = c["features_pa"].float().to(device)  # [1, 256, H, W]
    dens = c["density"].to(device)  # [1, 1, H, W]
    _, C, H, W = feat.shape
    N = H * W
    img_diag = float(np.sqrt(H * H + W * W))

    feat_flat = feat.permute(0, 2, 3, 1).reshape(N, C)
    dens_flat = dens.reshape(N)
    coords = _coords(H, W, feat.device)

    # GT-cell label: 1 if any GT point falls in 1/8-resolution feature cell
    gt = c["gt_points"]
    fy = np.clip(gt[:, 1] / 8, 0, H - 1).astype(np.int64)
    fx = np.clip(gt[:, 0] / 8, 0, W - 1).astype(np.int64)
    gt_mask = np.zeros(N, dtype=np.int64)
    gt_mask[fy * W + fx] = 1
    gt_label = torch.from_numpy(gt_mask).to(feat.device)

    bin_label = torch.from_numpy(density_bins(dens_flat.cpu().numpy(), n_bins=3)).to(
        feat.device
    )

    in_corr = _pearson(feat_flat.norm(dim=-1), dens_flat)

    def measure(builder) -> dict:
        edge_index, _, num_nodes_total, _, _ = builder.build_batch_graph(dens)
        ei = _strip_self_loops(edge_index, num_nodes_total)
        s = (coords[ei[0]] - coords[ei[1]]).norm(dim=-1)
        out_feat = _mean_aggregate(feat_flat, ei)
        agg_corr = _pearson(out_feat.norm(dim=-1), dens_flat)
        return {
            "long_range": float((s > 0.25 * img_diag).float().mean().item()),
            "mean_dist": float(s.mean().item()) / img_diag,
            "h_bin": homophily_ratio(ei, bin_label),
            "h_gt": homophily_ratio(ei, gt_label),
            "corr_drop": in_corr - agg_corr,
        }

    base = measure(DensityGraphBuilder(k=k))
    sp = measure(SpatialPriorDensityGraphBuilder(k=k, alpha=1.0, beta=1.0))

    return {
        "name": c["name"],
        "in_corr": in_corr,
        **{f"base_{m}": v for m, v in base.items()},
        **{f"sp_{m}": v for m, v in sp.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diag_cache/verify_spatial_prior.csv"),
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    names = list_cache()
    if not names:
        raise RuntimeError("no cached samples; run dump_features.py first")
    print(f"[verify_spatial_prior] {len(names)} samples on {device}, k={args.k}")

    rows = []
    for n in tqdm(names, desc="verify"):
        c = load_cache(n)
        rows.append(run_one(c, k=args.k, device=device))

    keys = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
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

    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    print()
    print(
        "=== Verification: SpatialPriorDensityGraphBuilder vs DensityGraphBuilder ==="
    )
    print(f"  k = {args.k}    samples = {len(rows)}")
    print(f"  baseline density-corr (input)         = {avg('in_corr'):+.4f}")
    print()
    print(f"  metric                baseline   spatial-prior   delta")
    rows_to_print = [
        ("long_range  (>0.25 D)", "base_long_range", "sp_long_range", "pp"),
        ("mean_dist   (/ D)    ", "base_mean_dist", "sp_mean_dist", "frac"),
        ("h_bin (3-bin density)", "base_h_bin", "sp_h_bin", "pp"),
        ("h_gt  (GT cell)      ", "base_h_gt", "sp_h_gt", "pp"),
        ("corr_drop (1-hop agg)", "base_corr_drop", "sp_corr_drop", "abs"),
    ]
    for label, kb, ks, unit in rows_to_print:
        b, s = avg(kb), avg(ks)
        if unit == "pp":
            d = f"{(s - b) * 100:+.2f} pp"
        elif unit == "frac":
            d = f"{(s - b):+.4f}"
        else:
            d = f"{(s - b):+.4f}"
        print(f"  {label}  {b:8.4f}    {s:8.4f}     {d}")
    print()
    # verdicts
    long_drop = avg("base_long_range") - avg("sp_long_range")
    h_bin_gain = (avg("sp_h_bin") - avg("base_h_bin")) * 100
    h_gt_loss = (avg("base_h_gt") - avg("sp_h_gt")) * 100
    corr_drop_gain = avg("base_corr_drop") - avg("sp_corr_drop")
    print(f"  [verdict] long-range edges suppressed by   {long_drop * 100:+.2f} pp")
    print(f"  [verdict] density-bin homophily gain        {h_bin_gain:+.2f} pp")
    print(
        f"  [verdict] GT-cell homophily change          {-h_gt_loss:+.2f} pp  "
        "(positive = better)"
    )
    print(
        f"  [verdict] 1-hop corr_drop reduction         {corr_drop_gain:+.4f}  "
        "(positive = signal preserved better)"
    )

    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()

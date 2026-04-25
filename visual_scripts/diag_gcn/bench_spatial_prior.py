"""Quick benchmark: SpatialPriorDensityGraphBuilder vs DensityGraphBuilder.

Times graph construction on the cached SHA samples and re-checks the key
edge-distance metric to make sure the optimisation didn't regress.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from crowdcount.models.gcn import (
    DensityGraphBuilder,
    SpatialPriorDensityGraphBuilder,
)
from visual_scripts.diag_gcn.common import list_cache, load_cache


def bench(builder, dens: torch.Tensor, n_warmup: int = 1, n_repeat: int = 3) -> float:
    for _ in range(n_warmup):
        builder.build_batch_graph(dens)
    if dens.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        builder.build_batch_graph(dens)
    if dens.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeat


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list_cache()[:10]  # 10 samples is enough for timing
    base = DensityGraphBuilder(k=4)
    sp = SpatialPriorDensityGraphBuilder(k=4, alpha=1.0, beta=1.0)

    print(f"benchmark on {len(names)} samples, device={device}")
    print(
        f"  {'name':<10}  {'H*W':>7}  {'baseline_ms':>11}  {'spatial_ms':>10}  {'overhead':>9}  {'long_range_sp':>13}"
    )
    base_total = sp_total = 0.0
    long_range_sp_all = []
    for n in names:
        c = load_cache(n)
        dens = c["density"].to(device)
        H, W = dens.shape[-2:]
        N = H * W
        tb = bench(base, dens)
        ts = bench(sp, dens)
        base_total += tb
        sp_total += ts

        # quick sanity: long_range fraction on spatial-prior must remain low
        ei, _, num_total, _, _ = sp.build_batch_graph(dens)
        ei = ei[:, : ei.shape[1] - num_total]
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        s = (coords[ei[0]] - coords[ei[1]]).norm(dim=-1)
        D = float((H * H + W * W) ** 0.5)
        lr = float((s > 0.25 * D).float().mean().item())
        long_range_sp_all.append(lr)

        print(
            f"  {n:<10}  {N:>7d}  {tb * 1000:>10.1f}   {ts * 1000:>9.1f}    {ts / tb:>7.2f}x  {lr:>12.4f}"
        )

    print()
    print(
        f"  TOTAL baseline = {base_total * 1000:>7.1f} ms   "
        f"spatial-prior = {sp_total * 1000:>7.1f} ms   "
        f"avg overhead = {sp_total / base_total:.2f}x"
    )
    print(
        f"  spatial long-range mean = {np.mean(long_range_sp_all):.4f} "
        f"(should be <0.01)"
    )


if __name__ == "__main__":
    main()

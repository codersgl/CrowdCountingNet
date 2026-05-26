"""Lightweight density-guided GCN feature refinement for MoECountNet.

A single-stream GCN that builds a k-NN graph from a lightweight internal
density preview and refines MoE features via graph convolution with a gated
residual connection. Simpler than DSGCNet's dual-stream CrossStreamGCN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.gcn import DensityGraphBuilder
from crowdcount.models.gcn import GCNConv as _GCNConv


class DensityGCNRefine(nn.Module):
    """Single-layer density-guided GCN with gated residual.

    A minimal 1x1 Conv produces an internal density preview used only for
    k-NN graph construction.  One GCNConv layer refines features; a
    learnable gate (init 0) lets the network gradually incorporate GCN
    context without destabilising early training.
    """

    def __init__(
        self,
        channels: int = 256,
        k: int = 4,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_channels if hidden_channels is not None else channels
        self.preview_conv = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.normal_(self.preview_conv.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.preview_conv.bias, -2.0)  # softplus(-2) ≈ 0.13
        self.graph_builder = DensityGraphBuilder(k=k)
        self.conv = _GCNConv(channels, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, channels) if hidden != channels else nn.Identity()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        preview = F.softplus(self.preview_conv(features)).detach()  # [B, 1, H, W]
        edge_index, _, _, _, _ = self.graph_builder.build_batch_graph(preview)
        nodes = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        refined = self.conv(nodes, edge_index)
        refined = self.norm(refined)
        refined = F.relu(refined)
        refined = self.proj(refined)
        refined = refined.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return features + self.gate.tanh() * refined

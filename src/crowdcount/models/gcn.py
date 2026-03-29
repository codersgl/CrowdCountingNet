"""GCN modules for DSGCNet.

Density-guided and feature-guided graph convolutional processors.
Supports both fixed-k and adaptive graph construction strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


class DensityGraphBuilder:
    def __init__(self, k: int = 4):
        self.k = k

    def build_batch_graph(self, density_maps: torch.Tensor):
        B, C, H, W = density_maps.shape
        num_nodes = H * W
        flat_density = density_maps.view(B, -1)

        dist = torch.abs(flat_density.unsqueeze(2) - flat_density.unsqueeze(1))
        sorted_indices = torch.argsort(dist, dim=2)[:, :, 1 : self.k + 1]

        device = density_maps.device
        src_nodes = (
            torch.arange(num_nodes, device=device)
            .view(1, num_nodes, 1)
            .expand(B, num_nodes, self.k)
        )
        tgt_nodes = sorted_indices

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src_nodes = src_nodes + batch_offset
        tgt_nodes = tgt_nodes + batch_offset

        src_nodes = src_nodes.reshape(-1)
        tgt_nodes = tgt_nodes.reshape(-1)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)

        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        return edge_index, num_nodes_total, H, W


class AdaptiveDensityGraphBuilder:
    """Density-guided adaptive graph builder.

    Nodes in high-density regions connect to more neighbours (larger k),
    while nodes in low-density regions use fewer (smaller k).
    """

    def __init__(
        self,
        k_base: int = 4,
        k_min: int = 2,
        k_max: int = 8,
        density_scale: float = 4.0,
    ):
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.density_scale = density_scale

    def build_batch_graph(self, density_maps: torch.Tensor):
        B, C, H, W = density_maps.shape
        num_nodes = H * W
        device = density_maps.device
        flat_density = density_maps.view(B, -1)  # [B, N]

        # Per-image min-max normalisation → [0, 1]
        d_min = flat_density.min(dim=1, keepdim=True).values
        d_max = flat_density.max(dim=1, keepdim=True).values
        d_norm = (flat_density - d_min) / (d_max - d_min + 1e-8)

        # k per node: higher density → more neighbours
        k_per_node = torch.clamp(
            torch.round(
                torch.tensor(self.k_base, dtype=torch.float32, device=device)
                + self.density_scale * d_norm
            ).long(),
            min=self.k_min,
            max=self.k_max,
        )  # [B, N]

        # Pairwise density distance → top-k_max candidates
        dist = torch.abs(flat_density.unsqueeze(2) - flat_density.unsqueeze(1))
        sorted_indices = torch.argsort(dist, dim=2)[
            :, :, 1 : self.k_max + 1
        ]  # [B, N, k_max]

        # Boolean mask: keep only the first k_per_node[b, i] neighbours
        range_idx = torch.arange(self.k_max, device=device).view(1, 1, -1)
        mask = range_idx < k_per_node.unsqueeze(2)  # [B, N, k_max]

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src = (
            torch.arange(num_nodes, device=device)
            .view(1, -1, 1)
            .expand(B, num_nodes, self.k_max)
            + batch_offset
        )
        tgt = sorted_indices + batch_offset

        edge_index = torch.stack([src[mask], tgt[mask]], dim=0)
        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        return edge_index, num_nodes_total, H, W


class FeatureGraphBuilder:
    def __init__(self, k: int = 4):
        self.k = k

    def build_batch_graph(self, feature_maps: torch.Tensor):
        B, C, H, W = feature_maps.shape
        num_nodes = H * W
        device = feature_maps.device

        flat_features = (
            feature_maps.permute(0, 2, 3, 1).contiguous().view(B, num_nodes, C)
        )
        norm_features = F.normalize(flat_features, p=2, dim=-1)
        sim = torch.matmul(norm_features, norm_features.transpose(-1, -2))

        _, sorted_indices = torch.topk(sim, k=self.k + 1, dim=2, largest=True)
        sorted_indices = sorted_indices[:, :, 1:]

        src_nodes = (
            torch.arange(num_nodes, device=device)
            .view(1, num_nodes, 1)
            .expand(B, num_nodes, self.k)
        )
        tgt_nodes = sorted_indices

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src_nodes = src_nodes + batch_offset
        tgt_nodes = tgt_nodes + batch_offset

        src_nodes = src_nodes.reshape(-1)
        tgt_nodes = tgt_nodes.reshape(-1)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)

        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        return edge_index, num_nodes_total, H, W


class AdaptiveFeatureGraphBuilder:
    """Similarity-threshold adaptive graph builder.

    Instead of a fixed k, connects all node pairs whose cosine similarity
    exceeds *sim_threshold*, while guaranteeing at least *k_min* neighbours
    and capping at *k_max* for memory safety.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 8,
        sim_threshold: float = 0.5,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.sim_threshold = sim_threshold

    def build_batch_graph(self, feature_maps: torch.Tensor):
        B, C, H, W = feature_maps.shape
        num_nodes = H * W
        device = feature_maps.device

        flat_features = (
            feature_maps.permute(0, 2, 3, 1).contiguous().view(B, num_nodes, C)
        )
        norm_features = F.normalize(flat_features, p=2, dim=-1)
        sim = torch.matmul(norm_features, norm_features.transpose(-1, -2))

        # Top-k_max candidates (excluding self)
        top_values, sorted_indices = torch.topk(
            sim, k=self.k_max + 1, dim=2, largest=True
        )
        top_values = top_values[:, :, 1:]  # [B, N, k_max]
        sorted_indices = sorted_indices[:, :, 1:]

        # Keep edges above threshold, but guarantee at least k_min
        threshold_mask = top_values > self.sim_threshold  # [B, N, k_max]
        range_idx = torch.arange(self.k_max, device=device).view(1, 1, -1)
        min_mask = range_idx < self.k_min  # always keep first k_min
        mask = threshold_mask | min_mask  # [B, N, k_max]

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src = (
            torch.arange(num_nodes, device=device)
            .view(1, -1, 1)
            .expand(B, num_nodes, self.k_max)
            + batch_offset
        )
        tgt = sorted_indices + batch_offset

        edge_index = torch.stack([src[mask], tgt[mask]], dim=0)
        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        return edge_index, num_nodes_total, H, W


class GCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, tensor: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(tensor, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class DensityGCNProcessor(nn.Module):
    def __init__(
        self,
        k: int = 4,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
        adaptive: bool = False,
        k_min: int = 2,
        k_max: int = 8,
        density_scale: float = 4.0,
    ):
        super().__init__()
        if adaptive:
            self.graph_builder = AdaptiveDensityGraphBuilder(
                k_base=k, k_min=k_min, k_max=k_max, density_scale=density_scale
            )
        else:
            self.graph_builder = DensityGraphBuilder(k)
        self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(
        self, density_maps: torch.Tensor, feature_maps: torch.Tensor
    ) -> torch.Tensor:
        B, in_channels, H, W = feature_maps.shape
        edge_index, _, H, W = self.graph_builder.build_batch_graph(density_maps)
        node_features = (
            feature_maps.permute(0, 2, 3, 1).contiguous().view(-1, in_channels)
        )
        out = self.gcn(node_features, edge_index)
        return out.view(B, H, W, in_channels).permute(0, 3, 1, 2).contiguous()


class FeatureGCNProcessor(nn.Module):
    def __init__(
        self,
        k: int = 4,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
        adaptive: bool = False,
        k_min: int = 2,
        k_max: int = 8,
        sim_threshold: float = 0.5,
    ):
        super().__init__()
        if adaptive:
            self.graph_builder = AdaptiveFeatureGraphBuilder(
                k_min=k_min, k_max=k_max, sim_threshold=sim_threshold
            )
        else:
            self.graph_builder = FeatureGraphBuilder(k)
        self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature_maps.shape
        edge_index, _, H, W = self.graph_builder.build_batch_graph(feature_maps)
        node_features = feature_maps.permute(0, 2, 3, 1).contiguous().view(-1, C)
        out = self.gcn(node_features, edge_index)
        return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

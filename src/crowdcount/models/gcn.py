"""GCN modules for DSGCNet.

Density-guided and feature-guided graph convolutional processors.
Supports fixed-k, adaptive, and super-node graph construction strategies.
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


class SuperNodeGCNProcessor(nn.Module):
    """Super-Node GCN: global scene reasoning via learnable prototypes.

    Instead of building an O(N²) pixel-to-pixel graph, introduces M learnable
    "super-node" prototypes that act as global scene codebook entries.

    Three-step message passing:
      1. **Gather**: Each super-node attends to all pixel nodes → absorbs
         region-level statistics (O(NM) complexity).
      2. **Process**: Super-nodes exchange information via a fully-connected
         GCN layer (O(M²) complexity, M is tiny).
      3. **Scatter**: Pixel nodes attend to refined super-nodes → receive
         global context (O(NM) complexity).

    Total complexity: O(NM + M²) << O(N²) when M << N.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_supernodes: int = 8,
        num_heads: int = 4,
        hidden_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_supernodes = num_supernodes
        self.num_heads = num_heads
        head_dim = in_channels // num_heads

        # Learnable super-node embeddings (scene prototypes)
        self.prototypes = nn.Parameter(
            torch.randn(1, num_supernodes, in_channels) * 0.02
        )

        # Gather: pixel → super-node cross-attention
        self.gather_q = nn.Linear(in_channels, in_channels, bias=False)
        self.gather_k = nn.Linear(in_channels, in_channels, bias=False)
        self.gather_v = nn.Linear(in_channels, in_channels, bias=False)

        # Process: super-node ↔ super-node (small fully-connected GCN)
        self.sn_conv = GCNConv(in_channels, hidden_channels)
        self.sn_conv2 = GCNConv(hidden_channels, in_channels)

        # Scatter: super-node → pixel cross-attention
        self.scatter_q = nn.Linear(in_channels, in_channels, bias=False)
        self.scatter_k = nn.Linear(in_channels, in_channels, bias=False)
        self.scatter_v = nn.Linear(in_channels, in_channels, bias=False)

        # Output projection with residual gate (initialised to 0 → identity)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.gate = nn.Parameter(torch.zeros(1))

        self._head_dim = head_dim
        # Pre-build fully-connected edge_index for M super-nodes
        src = []
        tgt = []
        for i in range(num_supernodes):
            for j in range(num_supernodes):
                src.append(i)
                tgt.append(j)
        self.register_buffer(
            "_fc_edges",
            torch.tensor([src, tgt], dtype=torch.long),
        )

    def _multihead_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head scaled dot-product attention.

        Args:
            q: [B, Nq, C]
            k: [B, Nk, C]
            v: [B, Nk, C]

        Returns:
            [B, Nq, C]
        """
        B, Nq, C = q.shape
        Nk = k.shape[1]
        H = self.num_heads
        d = self._head_dim

        q = q.view(B, Nq, H, d).transpose(1, 2)  # [B, H, Nq, d]
        k = k.view(B, Nk, H, d).transpose(1, 2)  # [B, H, Nk, d]
        v = v.view(B, Nk, H, d).transpose(1, 2)  # [B, H, Nk, d]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)  # [B, H, Nq, Nk]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, H, Nq, d]
        return out.transpose(1, 2).contiguous().view(B, Nq, C)

    def forward(
        self,
        feature_maps: torch.Tensor,
        density_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            feature_maps: [B, C, H, W] PA-FPN fused features.
            density_maps: [B, 1, H, W] optional density (unused in current
                          version; reserved for density-conditioned gather).

        Returns:
            [B, C, H, W] features enhanced with global super-node context.
        """
        B, C, H, W = feature_maps.shape
        N = H * W
        M = self.num_supernodes

        # Flatten spatial dims: [B, N, C]
        pixels = feature_maps.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        protos = self.prototypes.expand(B, -1, -1)  # [B, M, C]

        # --- Step 1: Gather (pixel → super-node) ---
        q_g = self.gather_q(protos)  # [B, M, C]
        k_g = self.gather_k(pixels)  # [B, N, C]
        v_g = self.gather_v(pixels)  # [B, N, C]
        supernodes = self._multihead_attn(q_g, k_g, v_g)  # [B, M, C]

        # --- Step 2: Process (super-node ↔ super-node GCN) ---
        # Flatten batch for torch_geometric
        sn_flat = supernodes.view(B * M, C)
        batch_offset = torch.arange(B, device=feature_maps.device).view(B, 1, 1) * M
        edges = self._fc_edges.unsqueeze(0).expand(B, -1, -1) + batch_offset
        edges = edges.view(2, -1)  # [2, B*M*M]
        sn_flat = F.relu(self.sn_conv(sn_flat, edges))
        sn_flat = F.dropout(sn_flat, p=0.3, training=self.training)
        sn_flat = F.relu(self.sn_conv2(sn_flat, edges))
        supernodes_refined = sn_flat.view(B, M, C)

        # --- Step 3: Scatter (super-node → pixel) ---
        q_s = self.scatter_q(pixels)  # [B, N, C]
        k_s = self.scatter_k(supernodes_refined)  # [B, M, C]
        v_s = self.scatter_v(supernodes_refined)  # [B, M, C]
        pixel_update = self._multihead_attn(q_s, k_s, v_s)  # [B, N, C]

        # Gated residual (gate=0 at init → identity)
        pixel_update = self.out_proj(pixel_update)
        enhanced = pixels + self.gate.tanh() * pixel_update

        return enhanced.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

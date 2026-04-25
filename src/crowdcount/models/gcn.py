"""GCN modules for DSGCNet.

Density-guided and feature-guided graph convolutional processors.
Supports fixed-k, adaptive, uncertainty-guided, and super-node graph construction strategies.
Also contains DensityAdaptiveFusion for density-conditioned multi-stream fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops


def compute_uncertainty(density: torch.Tensor) -> torch.Tensor:
    """Compute pixel-wise uncertainty from a density prediction.

    Uses binary entropy of the sigmoid-normalised density as the
    uncertainty measure: u = -p*log(p) - (1-p)*log(1-p), then
    min-max normalised per image to [0, 1].

    Args:
        density: [B, 1, H, W] raw density prediction (detached recommended).

    Returns:
        [B, 1, H, W] uncertainty map in [0, 1].
    """
    p = density.sigmoid()  # [B, 1, H, W] in (0, 1)
    eps = 1e-6
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))  # [B,1,H,W]
    # Per-image min-max normalisation
    B = entropy.shape[0]
    flat = entropy.view(B, -1)  # [B, N]
    e_min = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    e_max = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    return (entropy - e_min) / (e_max - e_min + eps)


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

        # Extract edge distances for selected neighbours
        edge_dist = torch.gather(dist, 2, sorted_indices)  # [B, N, k]
        edge_attr = torch.exp(-edge_dist).reshape(-1, 1)  # [B*N*k, 1]

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src_nodes = src_nodes + batch_offset
        tgt_nodes = tgt_nodes + batch_offset

        src_nodes = src_nodes.reshape(-1)
        tgt_nodes = tgt_nodes.reshape(-1)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)

        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        # Append weight=1.0 for self-loop edges
        num_self_loops = num_nodes_total
        self_loop_attr = torch.ones(num_self_loops, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


class SpatialPriorDensityGraphBuilder:
    """Density k-NN graph with spatial-distance regularisation.

    Edge cost: ``alpha * |Δd| / scale_d  +  beta * ‖Δp‖ / sigma_p``,
    where ``scale_d`` is the per-image median pairwise density distance and
    ``sigma_p`` is the median pairwise spatial distance on the feature grid.
    The spatial term suppresses semantically-spurious long-range edges that
    plain density k-NN tends to produce when many cells share similar
    densities (Diag A: long-range edges 67.8% → 0.1% on SHA test).
    The output schema (``edge_index, edge_attr, num_nodes_total, H, W``)
    matches ``DensityGraphBuilder`` so the rest of the pipeline is
    unchanged.
    """

    def __init__(self, k: int = 4, alpha: float = 1.0, beta: float = 1.0):
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def build_batch_graph(self, density_maps: torch.Tensor):
        B, _C, H, W = density_maps.shape
        num_nodes = H * W
        device = density_maps.device
        dtype = density_maps.dtype
        flat_density = density_maps.view(B, -1)  # [B, N]

        # Spatial coordinates on the feature grid (shared across the batch)
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [N, 2]
        p_dist = torch.cdist(coords, coords, p=2.0)  # [N, N]
        sigma_p = p_dist.median().clamp_min(1e-6)

        # Per-image density distance and its scale
        d_dist = (
            flat_density.unsqueeze(2) - flat_density.unsqueeze(1)
        ).abs()  # [B, N, N]
        # Robust per-image scale: median across all pairs, broadcast back.
        d_scale = d_dist.reshape(B, -1).median(dim=1).values.view(B, 1, 1) + 1e-6

        cost = self.alpha * d_dist / d_scale + self.beta * (
            p_dist.unsqueeze(0) / sigma_p
        )  # [B, N, N]
        sorted_indices = torch.argsort(cost, dim=2)[:, :, 1 : self.k + 1]  # [B, N, k]

        # Edge attribute uses density distance (consistent with the baseline
        # builder), so downstream GCN sees the same edge-weight semantics.
        edge_d = torch.gather(d_dist, 2, sorted_indices)  # [B, N, k]
        edge_attr = torch.exp(-edge_d).reshape(-1, 1)  # [B*N*k, 1]

        src_nodes = (
            torch.arange(num_nodes, device=device)
            .view(1, num_nodes, 1)
            .expand(B, num_nodes, self.k)
        )
        tgt_nodes = sorted_indices

        batch_offset = torch.arange(B, device=device).view(B, 1, 1) * num_nodes
        src_nodes = (src_nodes + batch_offset).reshape(-1)
        tgt_nodes = (tgt_nodes + batch_offset).reshape(-1)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)

        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)
        self_loop_attr = torch.ones(num_nodes_total, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


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

        # Extract edge distances for selected neighbours
        sorted_dist = torch.gather(dist, 2, sorted_indices)  # [B, N, k_max]
        edge_attr = torch.exp(-sorted_dist)[mask].unsqueeze(1)  # [E, 1]

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
        # Append weight=1.0 for self-loop edges
        self_loop_attr = torch.ones(num_nodes_total, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


class UncertaintyAdaptiveDensityGraphBuilder:
    """Uncertainty-guided adaptive graph builder.

    Like AdaptiveDensityGraphBuilder but uses an external uncertainty map
    (instead of density) to modulate per-node k: higher uncertainty → more
    neighbours, giving GCN more context to disambiguate hard regions.
    Falls back to density-based modulation when no uncertainty is provided.
    """

    def __init__(
        self,
        k_base: int = 4,
        k_min: int = 2,
        k_max: int = 8,
        density_scale: float = 4.0,
        uncertainty_scale: float = 6.0,
    ):
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.density_scale = density_scale
        self.uncertainty_scale = uncertainty_scale

    def build_batch_graph(
        self,
        density_maps: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ):
        B, C, H, W = density_maps.shape
        num_nodes = H * W
        device = density_maps.device
        flat_density = density_maps.view(B, -1)  # [B, N]

        # Decide modulation source: uncertainty if available, else density
        if uncertainty is not None:
            mod_map = uncertainty.view(B, -1)  # already in [0, 1]
            scale = self.uncertainty_scale
        else:
            d_min = flat_density.min(dim=1, keepdim=True).values
            d_max = flat_density.max(dim=1, keepdim=True).values
            mod_map = (flat_density - d_min) / (d_max - d_min + 1e-8)
            scale = self.density_scale

        k_per_node = torch.clamp(
            torch.round(
                torch.tensor(self.k_base, dtype=torch.float32, device=device)
                + scale * mod_map
            ).long(),
            min=self.k_min,
            max=self.k_max,
        )  # [B, N]

        # Pairwise density distance → top-k_max candidates
        dist = torch.abs(flat_density.unsqueeze(2) - flat_density.unsqueeze(1))
        sorted_indices = torch.argsort(dist, dim=2)[
            :, :, 1 : self.k_max + 1
        ]  # [B, N, k_max]

        range_idx = torch.arange(self.k_max, device=device).view(1, 1, -1)
        mask = range_idx < k_per_node.unsqueeze(2)  # [B, N, k_max]

        # Extract edge distances for selected neighbours
        sorted_dist = torch.gather(dist, 2, sorted_indices)  # [B, N, k_max]
        edge_attr = torch.exp(-sorted_dist)[mask].unsqueeze(1)  # [E, 1]

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
        # Append weight=1.0 for self-loop edges
        self_loop_attr = torch.ones(num_nodes_total, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


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

        top_values, sorted_indices = torch.topk(sim, k=self.k + 1, dim=2, largest=True)
        top_values = top_values[:, :, 1:]  # [B, N, k]
        sorted_indices = sorted_indices[:, :, 1:]

        # Edge attr: cosine similarity of selected neighbours
        edge_attr = top_values.reshape(-1, 1)  # [B*N*k, 1]

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
        # Append weight=1.0 for self-loop edges
        self_loop_attr = torch.ones(num_nodes_total, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


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

        # Edge attr: cosine similarity of selected neighbours
        edge_attr = top_values[mask].unsqueeze(1)  # [E, 1]

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
        # Append weight=1.0 for self-loop edges
        self_loop_attr = torch.ones(num_nodes_total, 1, device=device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return edge_index, edge_attr, num_nodes_total, H, W


class ECAConv(MessagePassing):
    """Edge-Conditioned Anisotropic Graph Convolution.

    Extends standard GCN by making message weights edge-dependent:
      m_{j→i} = (edge_gate_{ij}) * (W_v · x_j)
    where edge_gate is produced by an MLP over the edge attribute (similarity/distance).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # x: [N, in_channels], edge_attr: [E, 1]
        x = self.lin(x)
        edge_gate = self.edge_mlp(edge_attr)  # [E, out_channels]
        out = self.propagate(edge_index, x=x, edge_gate=edge_gate)
        return out + self.bias

    def message(self, x_j: torch.Tensor, edge_gate: torch.Tensor) -> torch.Tensor:
        return edge_gate * x_j


class ECAGCNModel(nn.Module):
    """Two-layer ECA-GCN with LayerNorm, residual connections, and lower dropout."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = ECAConv(in_channels, hidden_channels)
        self.conv2 = ECAConv(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = dropout
        # Residual projection if dims mismatch
        self.res_proj = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h + self.res_proj(residual)


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


class GATv2Model(nn.Module):
    """Two-layer GATv2 with LayerNorm, residual connections, and multi-head attention.

    GATv2 (Brody et al., 2022) computes *dynamic* attention coefficients,
    making it strictly more expressive than GATv1 for distinguishing
    neighbour importance — beneficial for density-varying crowd scenes.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        # Layer 1: multi-head with concat → output dim = hidden_channels
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=False,
        )
        # Layer 2: multi-head without concat → output dim = out_channels
        self.conv2 = GATv2Conv(
            hidden_channels,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = dropout
        self.res_proj = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h + self.res_proj(residual)


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
        use_uncertainty: bool = False,
        uncertainty_scale: float = 6.0,
        anisotropic: bool = False,
        conv_type: str = "gcn",
        spatial_prior: bool = False,
        spatial_alpha: float = 1.0,
        spatial_beta: float = 1.0,
    ):
        super().__init__()
        self._use_uncertainty = use_uncertainty
        self._anisotropic = anisotropic
        self._conv_type = conv_type
        if use_uncertainty and adaptive:
            self.graph_builder = UncertaintyAdaptiveDensityGraphBuilder(
                k_base=k,
                k_min=k_min,
                k_max=k_max,
                density_scale=density_scale,
                uncertainty_scale=uncertainty_scale,
            )
        elif adaptive:
            self.graph_builder = AdaptiveDensityGraphBuilder(
                k_base=k, k_min=k_min, k_max=k_max, density_scale=density_scale
            )
        elif spatial_prior:
            self.graph_builder = SpatialPriorDensityGraphBuilder(
                k=k, alpha=spatial_alpha, beta=spatial_beta
            )
        else:
            self.graph_builder = DensityGraphBuilder(k)
        if conv_type == "gatv2":
            self.gcn = GATv2Model(in_channels, hidden_channels, out_channels)
        elif anisotropic:
            self.gcn = ECAGCNModel(in_channels, hidden_channels, out_channels)
        else:
            self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(
        self,
        density_maps: torch.Tensor,
        feature_maps: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, in_channels, H, W = feature_maps.shape
        if self._use_uncertainty and isinstance(
            self.graph_builder, UncertaintyAdaptiveDensityGraphBuilder
        ):
            edge_index, edge_attr, _, H, W = self.graph_builder.build_batch_graph(
                density_maps, uncertainty=uncertainty
            )
        else:
            edge_index, edge_attr, _, H, W = self.graph_builder.build_batch_graph(
                density_maps
            )
        node_features = (
            feature_maps.permute(0, 2, 3, 1).contiguous().view(-1, in_channels)
        )
        if self._anisotropic and self._conv_type != "gatv2":
            out = self.gcn(node_features, edge_index, edge_attr)
        else:
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
        anisotropic: bool = False,
        conv_type: str = "gcn",
    ):
        super().__init__()
        self._anisotropic = anisotropic
        self._conv_type = conv_type
        if adaptive:
            self.graph_builder = AdaptiveFeatureGraphBuilder(
                k_min=k_min, k_max=k_max, sim_threshold=sim_threshold
            )
        else:
            self.graph_builder = FeatureGraphBuilder(k)
        if conv_type == "gatv2":
            self.gcn = GATv2Model(in_channels, hidden_channels, out_channels)
        elif anisotropic:
            self.gcn = ECAGCNModel(in_channels, hidden_channels, out_channels)
        else:
            self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature_maps.shape
        edge_index, edge_attr, _, H, W = self.graph_builder.build_batch_graph(
            feature_maps
        )
        node_features = feature_maps.permute(0, 2, 3, 1).contiguous().view(-1, C)
        if self._anisotropic and self._conv_type != "gatv2":
            out = self.gcn(node_features, edge_index, edge_attr)
        else:
            out = self.gcn(node_features, edge_index)
        return out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class CrossStreamGate(nn.Module):
    """Learnable gate controlling cross-stream information injection.

    Given hidden features from two streams, produces a per-node gate in [0, 1]
    that scales the injected signal.  Bias is initialised to a negative value
    so the gate starts near zero — the model begins as independent dual-stream
    and gradually learns to open cross-stream pathways.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        # Initialise last layer bias to -2 so sigmoid(-2) ≈ 0.12
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, h_self: torch.Tensor, h_other: torch.Tensor) -> torch.Tensor:
        """Return gate values in [0, 1] with shape [N, C]."""
        return torch.sigmoid(self.net(torch.cat([h_self, h_other], dim=-1)))


class CrossStreamGCNModel(nn.Module):
    """Two-layer interleaved dual-stream GCN with cross-stream gating.

    Each stream has its own graph topology (density-guided vs feature-guided)
    and its own GCNConv weights.  After each layer, a CrossStreamGate injects
    information from the other stream, enabling cross-stream interaction
    *during* message passing rather than only after.

    Forward pass:
        Layer 1: independent GCN propagation → cross-stream gate injection
        Layer 2: independent GCN propagation → cross-stream gate injection
        Output:  mean of two streams + input residual
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 512,
        out_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Density stream convolutions
        self.d_conv1 = GCNConv(in_channels, hidden_channels)
        self.d_conv2 = GCNConv(hidden_channels, out_channels)
        # Feature stream convolutions
        self.f_conv1 = GCNConv(in_channels, hidden_channels)
        self.f_conv2 = GCNConv(hidden_channels, out_channels)

        # Cross-stream gates (one per layer)
        self.gate_d1 = CrossStreamGate(hidden_channels)  # inject feature→density
        self.gate_f1 = CrossStreamGate(hidden_channels)  # inject density→feature
        self.gate_d2 = CrossStreamGate(out_channels)
        self.gate_f2 = CrossStreamGate(out_channels)

        # Normalisation
        self.norm_d1 = nn.LayerNorm(hidden_channels)
        self.norm_f1 = nn.LayerNorm(hidden_channels)
        self.norm_d2 = nn.LayerNorm(out_channels)
        self.norm_f2 = nn.LayerNorm(out_channels)

        self.dropout = dropout
        self.res_proj = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        density_edge_index: torch.Tensor,
        feature_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        # --- Layer 1: propagate on respective graphs ---
        d_h = self.d_conv1(x, density_edge_index)
        d_h = self.norm_d1(d_h)
        d_h = F.gelu(d_h)
        d_h = F.dropout(d_h, p=self.dropout, training=self.training)

        f_h = self.f_conv1(x, feature_edge_index)
        f_h = self.norm_f1(f_h)
        f_h = F.gelu(f_h)
        f_h = F.dropout(f_h, p=self.dropout, training=self.training)

        # Cross-stream injection after layer 1 — use pre-injection snapshots
        # so both streams see each other's unmodified state (symmetric exchange)
        d_h_pre, f_h_pre = d_h, f_h
        d_h = d_h_pre + self.gate_d1(d_h_pre, f_h_pre) * f_h_pre
        f_h = f_h_pre + self.gate_f1(f_h_pre, d_h_pre) * d_h_pre

        # --- Layer 2: propagate on respective graphs ---
        d_h = self.d_conv2(d_h, density_edge_index)
        d_h = self.norm_d2(d_h)
        d_h = F.gelu(d_h)
        d_h = F.dropout(d_h, p=self.dropout, training=self.training)

        f_h = self.f_conv2(f_h, feature_edge_index)
        f_h = self.norm_f2(f_h)
        f_h = F.gelu(f_h)
        f_h = F.dropout(f_h, p=self.dropout, training=self.training)

        # Cross-stream injection after layer 2 — use pre-injection snapshots
        d_h_pre, f_h_pre = d_h, f_h
        d_h = d_h_pre + self.gate_d2(d_h_pre, f_h_pre) * f_h_pre
        f_h = f_h_pre + self.gate_f2(f_h_pre, d_h_pre) * d_h_pre

        # Merge: mean of two streams + residual
        return (d_h + f_h) * 0.5 + self.res_proj(residual)


class CrossStreamGCNProcessor(nn.Module):
    """Unified dual-stream processor with cross-stream interleaved GCN.

    Replaces independent DensityGCNProcessor + FeatureGCNProcessor + external
    alpha/gate fusion with a single module that builds both graphs and runs
    CrossStreamGCNModel for in-process cross-stream interaction.
    """

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
        sim_threshold: float = 0.5,
        use_uncertainty: bool = False,
        uncertainty_scale: float = 6.0,
    ) -> None:
        super().__init__()

        # Density graph builder
        if use_uncertainty and adaptive:
            self.density_builder = UncertaintyAdaptiveDensityGraphBuilder(
                k_base=k,
                k_min=k_min,
                k_max=k_max,
                density_scale=density_scale,
                uncertainty_scale=uncertainty_scale,
            )
        elif adaptive:
            self.density_builder = AdaptiveDensityGraphBuilder(
                k_base=k, k_min=k_min, k_max=k_max, density_scale=density_scale
            )
        else:
            self.density_builder = DensityGraphBuilder(k)

        # Feature graph builder
        if adaptive:
            self.feature_builder = AdaptiveFeatureGraphBuilder(
                k_min=k_min, k_max=k_max, sim_threshold=sim_threshold
            )
        else:
            self.feature_builder = FeatureGraphBuilder(k)

        self.gcn = CrossStreamGCNModel(in_channels, hidden_channels, out_channels)
        # Only flag uncertainty if the builder actually supports it
        self._use_uncertainty = isinstance(
            self.density_builder, UncertaintyAdaptiveDensityGraphBuilder
        )

    def forward(
        self,
        density_maps: torch.Tensor,
        feature_maps: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = feature_maps.shape

        # Build density graph
        if self._use_uncertainty and isinstance(
            self.density_builder, UncertaintyAdaptiveDensityGraphBuilder
        ):
            d_edge_index, _, _, _, _ = self.density_builder.build_batch_graph(
                density_maps, uncertainty=uncertainty
            )
        else:
            d_edge_index, _, _, _, _ = self.density_builder.build_batch_graph(
                density_maps
            )

        # Build feature graph
        f_edge_index, _, _, _, _ = self.feature_builder.build_batch_graph(feature_maps)

        # Flatten to node features and run interleaved GCN
        node_features = feature_maps.permute(0, 2, 3, 1).contiguous().view(-1, C)
        out = self.gcn(node_features, d_edge_index, f_edge_index)
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


# ---------------------------------------------------------------------------
# Density-Adaptive Fusion
# ---------------------------------------------------------------------------


class DensityAdaptiveFusion(nn.Module):
    """Density-conditioned adaptive fusion for GCN dual-stream outputs.

    Replaces static ``softmax(alpha)`` weights or simple :class:`GateMechanism`
    with density-aware per-pixel (or per-image) fusion weights.  The key
    insight is that different spatial regions benefit from different stream
    emphases:

    - Dense regions → density-GCN stream is more reliable (density similarity
      captures crowd structure)
    - Sparse regions → feature-GCN stream is more stable (feature
      representation is discriminative when density signal is weak)
    - The PA-FPN baseline stream provides a stable anchor

    Architecture (``spatial=True``):

        d_emb    = Conv3x3→BN→ReLU→Conv3x3→BN→ReLU(density)   # density encoding
        combined = cat(feat_pa, density_gcn, feature_gcn, d_emb)
        weights  = softmax(Conv1x1→BN→ReLU→Conv1x1(combined))  # [B, 3, H, W]
        fused    = w0*feat_pa + w1*density_gcn + w2*feature_gcn

    Architecture (``spatial=False``):

        d_emb    = same density encoding
        combined = cat(GAP(feat_pa), GAP(density_gcn), GAP(feature_gcn), GAP(d_emb))
        weights  = softmax(MLP(combined))  # [B, 3]
        fused    = w0*feat_pa + w1*density_gcn + w2*feature_gcn

    The final Conv1x1 / Linear layer is zero-initialised so that the softmax
    output starts at uniform [1/3, 1/3, 1/3], preserving the baseline at
    training start.

    Args:
        in_channels: Feature channel dimension (default 256).
        density_embed_dim: Internal channels for density encoding (default 64).
        spatial: If ``True``, produce per-pixel spatial weights;
            if ``False``, produce per-image global weights.
    """

    def __init__(
        self,
        in_channels: int = 256,
        density_embed_dim: int = 64,
        spatial: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.density_embed_dim = density_embed_dim
        self.spatial = spatial

        # Density encoder: multi-scale pattern extraction
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, density_embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(density_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(density_embed_dim, density_embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(density_embed_dim),
            nn.ReLU(inplace=True),
        )

        if spatial:
            # Per-pixel: concatenate all features + density embedding → 3-way softmax
            combined_ch = 3 * in_channels + density_embed_dim
            self.weight_proj = nn.Sequential(
                nn.Conv2d(combined_ch, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 3, 1),  # 3 streams
            )
        else:
            # Per-image: global average pooling → MLP → 3 weights
            combined_ch = 3 * in_channels + density_embed_dim
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.weight_mlp = nn.Sequential(
                nn.Linear(combined_ch, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, 3),
            )

        # Zero-init final layer → softmax starts at uniform [1/3, 1/3, 1/3]
        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-initialise the final projection so fusion starts uniform."""
        if self.spatial:
            nn.init.zeros_(self.weight_proj[-1].weight)
            nn.init.zeros_(self.weight_proj[-1].bias)
        else:
            nn.init.zeros_(self.weight_mlp[-1].weight)
            nn.init.zeros_(self.weight_mlp[-1].bias)

    def forward(
        self,
        features_pa: torch.Tensor,
        density_gcn_feat: torch.Tensor,
        feature_gcn_feat: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Compute density-adaptive fused features.

        Args:
            features_pa: [B, C, H, W] PA-FPN baseline features.
            density_gcn_feat: [B, C, H, W] Density-GCN stream features.
            feature_gcn_feat: [B, C, H, W] Feature-GCN stream features.
            density: [B, 1, H, W] Density prediction (should be detached).

        Returns:
            [B, C, H, W] adaptively fused features.
        """
        d_emb = self.density_encoder(density)  # [B, D, H, W]

        if self.spatial:
            combined = torch.cat(
                [features_pa, density_gcn_feat, feature_gcn_feat, d_emb], dim=1
            )
            weights = F.softmax(self.weight_proj(combined), dim=1)  # [B, 3, H, W]
            fused = (
                weights[:, 0:1] * features_pa
                + weights[:, 1:2] * density_gcn_feat
                + weights[:, 2:3] * feature_gcn_feat
            )
        else:
            combined = torch.cat(
                [
                    self.gap(features_pa),
                    self.gap(density_gcn_feat),
                    self.gap(feature_gcn_feat),
                    self.gap(d_emb),
                ],
                dim=1,
            ).flatten(1)  # [B, 3C + D]
            weights = F.softmax(self.weight_mlp(combined), dim=1)  # [B, 3]
            fused = (
                weights[:, 0].view(-1, 1, 1, 1) * features_pa
                + weights[:, 1].view(-1, 1, 1, 1) * density_gcn_feat
                + weights[:, 2].view(-1, 1, 1, 1) * feature_gcn_feat
            )

        return fused

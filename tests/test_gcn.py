"""Tests for GCN modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.gcn import (
    AdaptiveDensityGraphBuilder,
    AdaptiveFeatureGraphBuilder,
    CrossStreamGate,
    CrossStreamGCNModel,
    CrossStreamGCNProcessor,
    DensityGCNProcessor,
    DensityGraphBuilder,
    ECAConv,
    ECAGCNModel,
    FeatureGCNProcessor,
    FeatureGraphBuilder,
    GATv2Model,
    GCNModel,
    SpatialPriorDensityGraphBuilder,
    UncertaintyAdaptiveDensityGraphBuilder,
    compute_uncertainty,
)


@pytest.fixture
def small_feature_map():
    return torch.randn(2, 256, 8, 8)  # B=2, C=256, H=8, W=8


@pytest.fixture
def small_density_map():
    return torch.rand(2, 1, 8, 8)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def test_density_graph_builder(small_density_map):
    builder = DensityGraphBuilder(k=2)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_density_map
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8
    assert H == 8 and W == 8


def test_feature_graph_builder(small_feature_map):
    builder = FeatureGraphBuilder(k=2)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_feature_map
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8


def test_adaptive_density_graph_builder(small_density_map):
    builder = AdaptiveDensityGraphBuilder(k_base=4, k_min=2, k_max=6, density_scale=2.0)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_density_map
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8
    # Adaptive: edge count should be between k_min and k_max per node (+ self loops)
    num_edges_no_self = edge_index.shape[1] - num_nodes_total
    num_nodes_per_batch = 8 * 8
    total_nodes = 2 * num_nodes_per_batch
    assert num_edges_no_self >= total_nodes * 2  # at least k_min per node
    assert num_edges_no_self <= total_nodes * 6  # at most k_max per node


def test_adaptive_feature_graph_builder(small_feature_map):
    builder = AdaptiveFeatureGraphBuilder(k_min=2, k_max=6, sim_threshold=0.5)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_feature_map
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8
    # At least k_min edges per node guaranteed
    num_edges_no_self = edge_index.shape[1] - num_nodes_total
    total_nodes = 2 * 8 * 8
    assert num_edges_no_self >= total_nodes * 2


def test_spatial_prior_density_graph_builder(small_density_map):
    builder = SpatialPriorDensityGraphBuilder(k=2, alpha=1.0, beta=1.0)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_density_map
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8
    # k * num_nodes neighbour edges + num_nodes self-loop edges
    assert edge_index.shape[1] == 2 * 8 * 8 * (2 + 1)


def test_spatial_prior_density_graph_builder_shorter_edges():
    """The spatial prior should reduce mean edge length vs. plain density k-NN."""
    torch.manual_seed(0)
    # Random density map so density k-NN tends to wander.
    density = torch.rand(1, 1, 16, 16)
    plain = DensityGraphBuilder(k=4)
    spatial = SpatialPriorDensityGraphBuilder(k=4, alpha=1.0, beta=1.0)

    H, W = 16, 16
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    def mean_edge_dist(builder):
        ei, _, num_nodes_total, _, _ = builder.build_batch_graph(density)
        # Drop the self-loops (last num_nodes_total entries)
        ei = ei[:, : ei.shape[1] - num_nodes_total]
        return (coords[ei[0]] - coords[ei[1]]).norm(dim=-1).mean().item()

    assert mean_edge_dist(spatial) < mean_edge_dist(plain) * 0.5


def test_density_gcn_processor_with_spatial_prior(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=2,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        spatial_prior=True,
        spatial_alpha=1.0,
        spatial_beta=1.0,
    )
    assert isinstance(proc.graph_builder, SpatialPriorDensityGraphBuilder)
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


def test_adaptive_density_graph_builder_uniform_density():
    """Uniform density should yield ~k_base neighbours per node."""
    density = torch.ones(1, 1, 4, 4) * 0.5
    builder = AdaptiveDensityGraphBuilder(k_base=3, k_min=2, k_max=6, density_scale=2.0)
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(density)
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 16


# ---------------------------------------------------------------------------
# GCN Model
# ---------------------------------------------------------------------------


def test_gcn_model_forward():
    model = GCNModel(in_channels=16, hidden_channels=32, out_channels=16)
    # minimal graph: 10 nodes, a few edges
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 16)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def test_density_gcn_processor_output_shape(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape, (
        "DensityGCN output should match feature map shape"
    )


def test_feature_gcn_processor_output_shape(small_feature_map):
    proc = FeatureGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256
    )
    out = proc(small_feature_map)
    assert out.shape == small_feature_map.shape, (
        "FeatureGCN output should match feature map shape"
    )


def test_adaptive_density_gcn_processor_output_shape(
    small_feature_map, small_density_map
):
    proc = DensityGCNProcessor(
        k=4,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        adaptive=True,
        k_min=2,
        k_max=6,
        density_scale=2.0,
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


def test_adaptive_feature_gcn_processor_output_shape(small_feature_map):
    proc = FeatureGCNProcessor(
        k=4,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        adaptive=True,
        k_min=2,
        k_max=6,
        sim_threshold=0.3,
    )
    out = proc(small_feature_map)
    assert out.shape == small_feature_map.shape


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_compute_uncertainty_output_range():
    density = torch.randn(2, 1, 8, 8)
    unc = compute_uncertainty(density)
    assert unc.shape == (2, 1, 8, 8)
    assert unc.min() >= 0.0 - 1e-5
    assert unc.max() <= 1.0 + 1e-5


def test_compute_uncertainty_uniform_density():
    """Uniform density should produce near-zero uncertainty range."""
    density = torch.ones(1, 1, 4, 4) * 5.0
    unc = compute_uncertainty(density)
    assert unc.max() - unc.min() < 0.01


def test_uncertainty_adaptive_graph_builder(small_density_map):
    unc = compute_uncertainty(small_density_map)
    builder = UncertaintyAdaptiveDensityGraphBuilder(
        k_base=3, k_min=2, k_max=8, uncertainty_scale=6.0
    )
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_density_map, uncertainty=unc
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8


def test_uncertainty_builder_more_edges_for_high_uncertainty():
    """High-uncertainty regions should produce more graph edges."""
    density = torch.rand(1, 1, 4, 4)
    unc = torch.zeros(1, 1, 4, 4)
    unc[:, :, :2, :] = 1.0  # high uncertainty top rows
    builder = UncertaintyAdaptiveDensityGraphBuilder(
        k_base=2, k_min=2, k_max=6, uncertainty_scale=4.0
    )
    edge_index, _, _, _, _ = builder.build_batch_graph(density, uncertainty=unc)
    num_nodes = 4 * 4
    num_edges_no_self = edge_index.shape[1] - num_nodes
    assert num_edges_no_self > num_nodes * 2  # more than k_min=2 per node on average


def test_uncertainty_builder_fallback_without_uncertainty(small_density_map):
    """Without uncertainty, should fall back to density-based modulation."""
    builder = UncertaintyAdaptiveDensityGraphBuilder(
        k_base=3, k_min=2, k_max=6, density_scale=2.0
    )
    edge_index, edge_attr, num_nodes_total, H, W = builder.build_batch_graph(
        small_density_map, uncertainty=None
    )
    assert edge_index.shape[0] == 2
    assert edge_attr.shape == (edge_index.shape[1], 1)
    assert num_nodes_total == 2 * 8 * 8


def test_density_gcn_processor_with_uncertainty(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=3,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        adaptive=True,
        k_min=2,
        k_max=6,
        use_uncertainty=True,
        uncertainty_scale=4.0,
    )
    unc = compute_uncertainty(small_density_map)
    out = proc(small_density_map, small_feature_map, uncertainty=unc)
    assert out.shape == small_feature_map.shape


def test_density_gcn_processor_uncertainty_none_fallback(
    small_feature_map, small_density_map
):
    """use_uncertainty=True but uncertainty=None should still work (fallback)."""
    proc = DensityGCNProcessor(
        k=3,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        adaptive=True,
        k_min=2,
        k_max=6,
        use_uncertainty=True,
    )
    out = proc(small_density_map, small_feature_map, uncertainty=None)
    assert out.shape == small_feature_map.shape


# ---------------------------------------------------------------------------
# ECA-GCN (Edge-Conditioned Anisotropic)
# ---------------------------------------------------------------------------


def test_eca_conv_forward():
    conv = ECAConv(in_channels=16, out_channels=32)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.rand(4, 1)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (10, 32)


def test_eca_gcn_model_forward():
    model = ECAGCNModel(in_channels=16, hidden_channels=32, out_channels=16)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.rand(4, 1)
    out = model(x, edge_index, edge_attr)
    assert out.shape == (10, 16)


def test_eca_gcn_model_residual():
    """Output should differ from input but maintain shape (residual connection)."""
    model = ECAGCNModel(in_channels=16, hidden_channels=32, out_channels=16).eval()
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.ones(3, 1)
    out = model(x, edge_index, edge_attr)
    assert out.shape == x.shape
    assert not torch.allclose(out, x, atol=1e-4)


def test_anisotropic_density_gcn_processor(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256, anisotropic=True
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


def test_anisotropic_feature_gcn_processor(small_feature_map):
    proc = FeatureGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256, anisotropic=True
    )
    out = proc(small_feature_map)
    assert out.shape == small_feature_map.shape


def test_anisotropic_adaptive_density_gcn_processor(
    small_feature_map, small_density_map
):
    proc = DensityGCNProcessor(
        k=4,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
        adaptive=True,
        k_min=2,
        k_max=6,
        anisotropic=True,
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


def test_edge_attr_values_density_builder():
    """Edge attr for density builder should be in (0, 1] range (exp(-dist))."""
    density = torch.rand(1, 1, 4, 4)
    builder = DensityGraphBuilder(k=2)
    _, edge_attr, _, _, _ = builder.build_batch_graph(density)
    assert (edge_attr > 0).all()
    assert (edge_attr <= 1.0 + 1e-6).all()


def test_edge_attr_values_feature_builder():
    """Edge attr for feature builder should be cosine similarities."""
    features = torch.randn(1, 16, 4, 4)
    builder = FeatureGraphBuilder(k=2)
    _, edge_attr, _, _, _ = builder.build_batch_graph(features)
    # Self-loop attrs are 1.0; neighbour attrs are cosine sims in [-1, 1]
    assert (edge_attr >= -1.0 - 1e-6).all()
    assert (edge_attr <= 1.0 + 1e-6).all()


def test_cross_stream_gate_forward():
    gate = CrossStreamGate(channels=16)
    h_self = torch.randn(10, 16)
    h_other = torch.randn(10, 16)
    out = gate(h_self, h_other)
    assert out.shape == (10, 16)
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()
    assert out.mean() < 0.25


def test_cross_stream_gcn_model_forward():
    model = CrossStreamGCNModel(in_channels=16, hidden_channels=32, out_channels=16)
    x = torch.randn(10, 16)
    density_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    feature_edge_index = torch.tensor([[0, 2, 4, 6], [2, 4, 6, 8]], dtype=torch.long)
    out = model(x, density_edge_index, feature_edge_index)
    assert out.shape == (10, 16)


def test_cross_stream_gcn_processor_output_shape(small_feature_map, small_density_map):
    proc = CrossStreamGCNProcessor(
        k=2,
        in_channels=256,
        hidden_channels=128,
        out_channels=256,
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


# ---------------------------------------------------------------------------
# GATv2 Model & Processors
# ---------------------------------------------------------------------------


def test_gatv2_model_forward():
    model = GATv2Model(in_channels=16, hidden_channels=32, out_channels=16, heads=4)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 16)


def test_density_gcn_processor_gatv2(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256, conv_type="gatv2"
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape


def test_feature_gcn_processor_gatv2(small_feature_map):
    proc = FeatureGCNProcessor(
        k=2, in_channels=256, hidden_channels=128, out_channels=256, conv_type="gatv2"
    )
    out = proc(small_feature_map)
    assert out.shape == small_feature_map.shape


# ---------------------------------------------------------------------------
# Deformable Graph Attention (conv_type="deformable")
# ---------------------------------------------------------------------------


def test_density_gcn_processor_deformable(small_feature_map, small_density_map):
    proc = DensityGCNProcessor(
        k=4, in_channels=256, hidden_channels=128, out_channels=256, conv_type="deformable"
    )
    out = proc(small_density_map, small_feature_map)
    assert out.shape == small_feature_map.shape
    assert proc.graph_builder is None


def test_feature_gcn_processor_deformable(small_feature_map):
    proc = FeatureGCNProcessor(
        k=4, in_channels=256, hidden_channels=128, out_channels=256, conv_type="deformable"
    )
    out = proc(small_feature_map)
    assert out.shape == small_feature_map.shape
    assert proc.graph_builder is None


def test_density_gcn_processor_deformable_gradient_flow(small_feature_map, small_density_map):
    """Verify the offset predictor in DeformableGraphAttention gets gradients."""
    proc = DensityGCNProcessor(
        k=4, in_channels=256, hidden_channels=128, out_channels=256, conv_type="deformable"
    )
    feature_maps = small_feature_map.clone().requires_grad_(True)
    density_maps = small_density_map.clone()
    out = proc(density_maps, feature_maps)
    loss = out.sum()
    loss.backward()
    # The offset_pred[-1].weight should have non-zero grads (it was zero-initialised
    # but the density_proj path provides a non-zero input)
    offset_weight_grad = proc.gcn.offset_pred[-1].weight.grad
    assert offset_weight_grad is not None
    assert offset_weight_grad.abs().sum() > 0


def test_feature_gcn_processor_deformable_gradient_flow(small_feature_map):
    """Verify the offset predictor in FeatureGCN (no density) gets gradients."""
    proc = FeatureGCNProcessor(
        k=4, in_channels=256, hidden_channels=128, out_channels=256, conv_type="deformable"
    )
    feature_maps = small_feature_map.clone().requires_grad_(True)
    out = proc(feature_maps)
    loss = out.sum()
    loss.backward()
    offset_weight_grad = proc.gcn.offset_pred[-1].weight.grad
    assert offset_weight_grad is not None
    assert offset_weight_grad.abs().sum() > 0

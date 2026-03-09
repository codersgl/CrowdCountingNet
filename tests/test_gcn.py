"""Tests for GCN modules."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.gcn import (
    DensityGCNProcessor,
    DensityGraphBuilder,
    FeatureGCNProcessor,
    FeatureGraphBuilder,
    GCNModel,
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
    edge_index, num_nodes_total, H, W = builder.build_batch_graph(small_density_map)
    assert edge_index.shape[0] == 2
    assert num_nodes_total == 2 * 8 * 8
    assert H == 8 and W == 8


def test_feature_graph_builder(small_feature_map):
    builder = FeatureGraphBuilder(k=2)
    edge_index, num_nodes_total, H, W = builder.build_batch_graph(small_feature_map)
    assert edge_index.shape[0] == 2
    assert num_nodes_total == 2 * 8 * 8


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

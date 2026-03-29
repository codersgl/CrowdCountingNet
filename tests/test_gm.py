"""Unit tests for GateMechanism and SpatialGateMechanism plugins."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.plugins.gm import GateMechanism, SpatialGateMechanism


@pytest.fixture
def default_gm() -> GateMechanism:
    return GateMechanism()


@pytest.fixture
def custom_gm() -> GateMechanism:
    return GateMechanism(input_dim=512, hidden_dim=256)


def test_gm_import() -> None:
    assert GateMechanism is not None
    assert issubclass(GateMechanism, nn.Module)


def test_gm_initialization_default(default_gm: GateMechanism) -> None:
    assert default_gm.fc1.in_features == 256
    assert default_gm.fc1.out_features == 128
    assert default_gm.fc2.in_features == 128
    assert default_gm.fc2.out_features == 3


def test_gm_initialization_custom(custom_gm: GateMechanism) -> None:
    assert custom_gm.fc1.in_features == 512
    assert custom_gm.fc1.out_features == 256
    assert custom_gm.fc2.in_features == 256
    assert custom_gm.fc2.out_features == 3


@pytest.mark.parametrize(
    "batch,channels,height,width",
    [(1, 256, 16, 16), (2, 256, 32, 32), (4, 256, 64, 64)],
)
def test_forward_shape_default(
    batch: int,
    channels: int,
    height: int,
    width: int,
    default_gm: GateMechanism,
) -> None:
    x = torch.randn(batch, channels, height, width)
    with torch.no_grad():
        output = default_gm(x)
    assert output.shape == (batch, 3)
    assert isinstance(output, torch.Tensor)


def test_forward_shape_custom(custom_gm: GateMechanism) -> None:
    x = torch.randn(2, 512, 32, 32)
    with torch.no_grad():
        output = custom_gm(x)
    assert output.shape == (2, 3)


def test_output_is_softmax_probability(default_gm: GateMechanism) -> None:
    x = torch.randn(3, 256, 20, 20)
    with torch.no_grad():
        output = default_gm(x)

    row_sums = output.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


# ---------------------------------------------------------------------------
# SpatialGateMechanism tests
# ---------------------------------------------------------------------------


@pytest.fixture
def default_spatial_gm() -> SpatialGateMechanism:
    return SpatialGateMechanism()


def test_spatial_gm_import() -> None:
    assert SpatialGateMechanism is not None
    assert issubclass(SpatialGateMechanism, nn.Module)


@pytest.mark.parametrize(
    "batch,channels,height,width",
    [(1, 256, 16, 16), (2, 256, 32, 32), (4, 256, 64, 64)],
)
def test_spatial_gm_forward_shape(
    batch: int,
    channels: int,
    height: int,
    width: int,
    default_spatial_gm: SpatialGateMechanism,
) -> None:
    x = torch.randn(batch, channels, height, width)
    with torch.no_grad():
        output = default_spatial_gm(x)
    assert output.shape == (batch, 3, height, width)


def test_spatial_gm_softmax_over_streams(
    default_spatial_gm: SpatialGateMechanism,
) -> None:
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        output = default_spatial_gm(x)
    # Sum over stream dim (dim=1) should be 1.0 at every spatial location
    stream_sums = output.sum(dim=1)  # [B, H, W]
    assert torch.allclose(
        stream_sums, torch.ones_like(stream_sums), rtol=1e-5, atol=1e-6
    )
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_spatial_gm_custom_dims() -> None:
    gm = SpatialGateMechanism(input_dim=512, hidden_dim=128, num_streams=4)
    x = torch.randn(2, 512, 8, 8)
    with torch.no_grad():
        output = gm(x)
    assert output.shape == (2, 4, 8, 8)

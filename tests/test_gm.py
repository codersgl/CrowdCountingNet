"""Unit tests for GateMechanism plugin."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from crowdcount.plugins.gm import GateMechanism


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

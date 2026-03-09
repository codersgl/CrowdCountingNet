"""Tests for feature fusion neck (SPD + PA-FPN)."""

from __future__ import annotations

import pytest
import torch

from crowdcount.models.neck import Decoder_SPD_PAFPN, SPD


# ---------------------------------------------------------------------------
# SPD
# ---------------------------------------------------------------------------


def test_spd_output_channels():
    spd = SPD()
    x = torch.randn(2, 64, 16, 16)
    out = spd(x)
    assert out.shape == (2, 256, 8, 8), (
        "SPD should quadruple channels and halve spatial dims"
    )


def test_spd_no_learnable_params():
    spd = SPD()
    assert sum(p.numel() for p in spd.parameters()) == 0, "SPD has no parameters"


# ---------------------------------------------------------------------------
# Decoder_SPD_PAFPN
# ---------------------------------------------------------------------------


@pytest.fixture
def fpn_inputs():
    """(C3, C4, C5) feature maps matching VGG16-BN outputs."""
    C3 = torch.randn(2, 256, 32, 32)
    C4 = torch.randn(2, 512, 16, 16)
    C5 = torch.randn(2, 512, 8, 8)
    return C3, C4, C5


def test_pafpn_output_shape(fpn_inputs):
    neck = Decoder_SPD_PAFPN(C3_size=256, C4_size=512, C5_size=512, feature_size=256)
    out = neck(fpn_inputs)
    # Output should have feature_size channels and C4 spatial size (16×16)
    assert out.shape[1] == 256
    assert out.shape[0] == 2


def test_pafpn_spatial_resolution(fpn_inputs):
    neck = Decoder_SPD_PAFPN(C3_size=256, C4_size=512, C5_size=512, feature_size=256)
    out = neck(fpn_inputs)
    # P4-level output after fusion
    assert out.shape[2] == 16 and out.shape[3] == 16

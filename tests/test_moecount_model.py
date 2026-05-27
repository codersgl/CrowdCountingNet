from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.backbone import (
    MoEVGGBackbone,
    convert_convnext_state_dict_for_features_only,
)
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.moecount.head import DensityHead
from crowdcount.models.moecount.moecount import MoECountNet
from crowdcount.models.moecount.neck import DeepBiFPNNeck, EnhancedFPNNeck


class TinyMoEBackbone(nn.Module):
    out_channels = (8, 16)

    def __init__(self) -> None:
        super().__init__()
        self.c2_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "c2": self.c2_conv(F.avg_pool2d(images, kernel_size=8, stride=8)),
            "c3": self.c3_conv(F.avg_pool2d(images, kernel_size=16, stride=16)),
        }


def build_tiny_moecount(final_activation: str = "softplus") -> MoECountNet:
    return MoECountNet(
        TinyMoEBackbone(),
        EnhancedFPNNeck(8, 16, out_channels=32, branch_channels=(16, 8, 8)),
        HeterogeneousSparseMoE(
            channels=32,
            gate_hidden_channels=8,
            warmup_epochs=0,
        ),
        DensityHead(in_channels=32, hidden_channels=8, final_activation=final_activation),
    )


def test_moecount_forward_stride8_and_keys() -> None:
    model = build_tiny_moecount().eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 128, 160))
    assert output["density_out"].shape == (2, 1, 16, 20)
    assert torch.all(output["density_out"] >= 0)
    for key in ("moe_weights", "moe_soft_probs", "moe_hard_mask", "moe_top1"):
        assert key in output


def test_moecount_train_mode_returns_balance_loss() -> None:
    model = build_tiny_moecount(final_activation="softplus").train()
    model.set_epoch(10, total_epochs=20)
    output = model(torch.randn(1, 3, 128, 128))
    assert output["moe_aux_total"] is not None
    assert output["density_out"].shape[-2:] == (16, 16)


def test_density_head_softplus_initial_density() -> None:
    head = DensityHead(
        in_channels=4,
        hidden_channels=8,
        final_activation="softplus",
        initial_density=0.05,
        final_weight_std=0.0,
    )
    with torch.no_grad():
        density = head(torch.randn(2, 4, 5, 7))
    assert torch.allclose(density, torch.full_like(density, 0.05), atol=1e-6)


def test_convnext_local_state_dict_key_conversion() -> None:
    state_dict = {
        "stem.0.weight": torch.ones(2, 3, 3, 3),
        "stem.1.bias": torch.ones(2),
        "stages.1.blocks.0.gamma": torch.ones(4),
        "head.fc.weight": torch.ones(10, 4),
    }
    model_state = {
        "stem_0.weight": torch.zeros(2, 3, 3, 3),
        "stem_1.bias": torch.zeros(2),
        "stages_1.blocks.0.gamma": torch.zeros(4),
    }
    converted, skipped = convert_convnext_state_dict_for_features_only(
        state_dict,
        model_state,
    )
    assert set(converted) == {
        "stem_0.weight",
        "stem_1.bias",
        "stages_1.blocks.0.gamma",
    }
    assert skipped == 1


def test_vgg_backbone_forward() -> None:
    backbone = MoEVGGBackbone(vgg_name="vgg16_bn", pretrained=False, out_levels=3)
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.randn(2, 3, 256, 256))
    assert set(out) == {"c2", "c3", "c4"}
    assert out["c2"].shape == (2, 256, 64, 64)  # stride 4
    assert out["c3"].shape == (2, 512, 32, 32)  # stride 8
    assert out["c4"].shape == (2, 512, 16, 16)  # stride 16
    assert backbone.out_channels == (256, 512, 512)


def test_vgg_backbone_out_levels_2() -> None:
    backbone = MoEVGGBackbone(vgg_name="vgg16_bn", pretrained=False, out_levels=2)
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.randn(1, 3, 128, 128))
    assert set(out) == {"c2", "c3"}
    assert out["c2"].shape == (1, 256, 32, 32)  # stride 4
    assert out["c3"].shape == (1, 512, 16, 16)  # stride 8
    assert backbone.out_channels == (256, 512)


def test_moecount_with_vgg_backbone() -> None:
    backbone = MoEVGGBackbone(vgg_name="vgg16_bn", pretrained=False, out_levels=3)
    c2_ch, c3_ch, c4_ch = backbone.out_channels
    neck = DeepBiFPNNeck(c2_ch, c3_ch, c4_ch, out_channels=64, branch_channels=(32, 16, 16))
    moe = HeterogeneousSparseMoE(channels=64, gate_hidden_channels=16, warmup_epochs=0)
    head = DensityHead(in_channels=64, hidden_channels=16, final_activation="softplus")
    model = MoECountNet(backbone, neck, moe, head)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 256, 256))
    assert output["density_out"].shape == (1, 1, 64, 64)  # c2 stride = 4
    assert "moe_weights" in output

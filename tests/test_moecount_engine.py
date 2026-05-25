from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.engine import evaluate_moecount, train_moecount_one_epoch
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.moecount.head import DensityHead
from crowdcount.models.moecount.losses import BayesianLoss, MoECountLoss
from crowdcount.models.moecount.moecount import MoECountNet
from crowdcount.models.moecount.neck import EnhancedFPNNeck


class TinyMoEBackbone(nn.Module):
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


def test_train_moecount_one_epoch_updates_parameters() -> None:
    model = build_tiny_moecount(final_activation="softplus").train()
    loss_fn = MoECountLoss(bayesian_loss=BayesianLoss(max_pixels_per_chunk=64))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = torch.randn(1, 3, 128, 128)
    targets = ({"point": torch.tensor([[32.0, 32.0]])},)
    gt_density = (torch.ones(1, 16, 16) / 256.0,)
    before = model.density_head.proj[0].weight.detach().clone()
    stats, global_step = train_moecount_one_epoch(
        model,
        loss_fn,
        [(samples, targets, gt_density)],
        optimizer,
        torch.device("cpu"),
        epoch=1,
        total_epochs=10,
        max_norm=0.1,
        use_amp=False,
        log_interval=1,
        vis_interval=0,
    )
    after = model.density_head.proj[0].weight.detach()
    assert global_step == 1
    assert "loss_total" in stats
    assert not torch.allclose(before, after)


def test_evaluate_moecount_returns_density_metrics() -> None:
    model = build_tiny_moecount(final_activation="softplus").eval()
    samples = torch.randn(1, 3, 128, 128)
    targets = ({"point": torch.zeros(0, 2), "orig_size": torch.tensor([128, 128])},)
    mae, mse = evaluate_moecount(
        model,
        [(samples, targets)],
        torch.device("cpu"),
        output_stride=8,
    )
    assert isinstance(mae, float)
    assert isinstance(mse, float)
    assert mae >= 0
    assert mse >= 0

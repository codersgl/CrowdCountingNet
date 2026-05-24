"""Tests for configurable regularization knobs."""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.head import SharedPredictionTrunk


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2),
            torch.zeros(batch_size, 256, height // 4, width // 4),
            torch.zeros(batch_size, 512, height // 8, width // 8),
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_regularization_config_reaches_default_model_components() -> None:
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        neck_dropout=0.05,
        head_dropout=0.15,
        density_dropout=0.25,
        gcn_dropout=0.35,
    )

    assert model.neck_dropout == 0.05
    assert isinstance(model.pred_trunk, SharedPredictionTrunk)
    assert model.pred_trunk.dropout == 0.15
    assert model.density_pred.dropout == 0.25
    assert model.density_gcn is not None
    assert model.feature_gcn is not None
    assert model.density_gcn.gcn.dropout == 0.35
    assert model.feature_gcn.gcn.dropout == 0.35


def test_head_dropout_is_stochastic_only_in_train_mode() -> None:
    torch.manual_seed(0)
    trunk = SharedPredictionTrunk(in_channels=4, feature_size=4, dropout=0.5)
    for module in trunk.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.weight, 0.1)
            nn.init.constant_(module.bias, 0.1)
    x = torch.ones(1, 4, 8, 8)

    trunk.train()
    torch.manual_seed(1)
    train_out_a = trunk(x)
    torch.manual_seed(2)
    train_out_b = trunk(x)
    assert not torch.allclose(train_out_a, train_out_b)

    trunk.eval()
    torch.manual_seed(1)
    eval_out_a = trunk(x)
    torch.manual_seed(2)
    eval_out_b = trunk(x)
    assert torch.allclose(eval_out_a, eval_out_b)


def test_model_forward_shapes_with_regularization_enabled() -> None:
    model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        neck_dropout=0.1,
        head_dropout=0.1,
        density_dropout=0.1,
        gcn_dropout=0.1,
    ).eval()

    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 128, 128))

    assert outputs["pred_logits"].shape[0] == 1
    assert outputs["pred_logits"].shape[2] == 2
    assert outputs["pred_points"].shape[0] == 1
    assert outputs["pred_points"].shape[2] == 2
    assert outputs["density_out"].shape[0] == 1
"""Tests for checkpoint compatibility migrations."""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdcount.models.checkpoint import load_model_state_dict
from crowdcount.models.dsgcnet import DSGCnet


class TinyVGGBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        return [
            torch.zeros(batch_size, 128, height // 2, width // 2),
            torch.zeros(batch_size, 256, height // 4, width // 4),
            torch.zeros(batch_size, 512, height // 8, width // 8),
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ]


def test_legacy_dap_acdr_keys_load_into_post_neck_acdr() -> None:
    source_model = DSGCnet(TinyVGGBackbone(), row=2, line=2, use_dap_neck=True)
    legacy_state_dict = {}
    for key, value in source_model.state_dict().items():
        if key.startswith("neck_acdr."):
            key = "pa.acdr." + key[len("neck_acdr.") :]
        legacy_state_dict[key] = value.clone()

    target_model = DSGCnet(TinyVGGBackbone(), row=2, line=2, use_dap_neck=True)
    load_model_state_dict(target_model, {"model": legacy_state_dict})


def test_regularization_dropout_does_not_change_state_dict_keys() -> None:
    source_model = DSGCnet(TinyVGGBackbone(), row=2, line=2)
    target_model = DSGCnet(
        TinyVGGBackbone(),
        row=2,
        line=2,
        neck_dropout=0.1,
        head_dropout=0.1,
        density_dropout=0.1,
        gcn_dropout=0.1,
    )

    assert set(source_model.state_dict()) == set(target_model.state_dict())
    load_model_state_dict(target_model, {"model": source_model.state_dict()})
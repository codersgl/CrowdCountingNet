"""Top-level MoECountNet model."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from crowdcount.models.moecount.backbone import MoEConvNeXtBackbone
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.moecount.head import DensityHead
from crowdcount.models.moecount.neck import EnhancedFPNNeck


class MoECountNet(nn.Module):
    """Pure density-map crowd counter with heterogeneous sparse MoE routing."""

    def __init__(
        self,
        backbone: MoEConvNeXtBackbone,
        neck: EnhancedFPNNeck,
        moe: HeterogeneousSparseMoE,
        density_head: DensityHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.moe = moe
        self.density_head = density_head

    def supports_moe(self) -> bool:
        return True

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.moe.set_epoch(epoch, total_epochs=total_epochs)

    def update_moe_temperature(self, decay_rate: float | None = None) -> None:
        self.moe.update_temperature(decay_rate=decay_rate)

    def get_gate_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.moe.gate.parameters() if parameter.requires_grad]

    def forward(self, samples: torch.Tensor, epoch: int | None = None) -> dict[str, Any]:
        if hasattr(samples, "tensors"):
            samples = samples.tensors
        if epoch is not None:
            self.moe.set_epoch(epoch)
        feature_maps = self.backbone(samples)
        fused_neck = self.neck(feature_maps["c2"], feature_maps["c3"])
        moe_features, moe_aux_losses, route = self.moe(fused_neck)
        density = self.density_head(moe_features)
        moe_aux_total = moe_aux_losses.get("total_aux") if moe_aux_losses else None
        return {
            "density_out": density,
            "pred_density": density,
            "moe_aux_total": moe_aux_total,
            "moe_aux_losses": moe_aux_losses,
            "moe_weights": route["weights"],
            "moe_soft_probs": route["soft_probs"],
            "moe_hard_mask": route["hard_mask"],
            "moe_top1": route["top1"],
            "moe_top_indices": route["top_indices"],
            "moe_load_fraction": route["load_fraction"],
            "moe_importance": route["importance"],
            "moe_entropy": route["entropy"],
            "moe_temperature": route["temperature"],
            "moe_warmup_active": route["warmup_active"],
        }


def build_moecount(cfg: DictConfig) -> MoECountNet:
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg
    backbone_cfg = getattr(model_cfg, "backbone", None)
    neck_cfg = getattr(model_cfg, "neck", None)
    moe_cfg = getattr(model_cfg, "moe", None)
    head_cfg = getattr(model_cfg, "head", None)

    backbone = MoEConvNeXtBackbone(
        arch=str(getattr(backbone_cfg, "arch", "convnext_tiny")),
        model_name=getattr(backbone_cfg, "model_name", None),
        pretrained=bool(getattr(backbone_cfg, "pretrained", True)),
        pretrained_path=getattr(backbone_cfg, "pretrained_path", None),
        out_indices=tuple(getattr(backbone_cfg, "out_indices", (1, 2))),
    )
    neck = EnhancedFPNNeck(
        c2_channels=backbone.out_channels[0],
        c3_channels=backbone.out_channels[1],
        out_channels=int(getattr(neck_cfg, "out_channels", 256)),
        branch_channels=tuple(getattr(neck_cfg, "branch_channels", (128, 64, 64))),
        dilations=tuple(getattr(neck_cfg, "dilations", (1, 2, 5))),
    )
    moe = HeterogeneousSparseMoE(
        channels=int(getattr(neck_cfg, "out_channels", 256)),
        gate_hidden_channels=int(getattr(moe_cfg, "gate_hidden_channels", 128)),
        top_k=int(getattr(moe_cfg, "top_k", 2)),
        temperature_init=float(getattr(moe_cfg, "temperature_init", 1.0)),
        temperature_min=float(getattr(moe_cfg, "temperature_min", 0.1)),
        temperature_decay=float(getattr(moe_cfg, "temperature_decay", 0.98)),
        warmup_fraction=float(getattr(moe_cfg, "warmup_fraction", 0.2)),
        warmup_epochs=getattr(moe_cfg, "warmup_epochs", None),
        cbam_reduction=int(getattr(moe_cfg, "cbam_reduction", 16)),
        lambda_importance=float(getattr(moe_cfg, "lambda_importance", 0.01)),
        lambda_load=float(getattr(moe_cfg, "lambda_load", 0.01)),
    )
    density_head = DensityHead(
        in_channels=int(getattr(neck_cfg, "out_channels", 256)),
        hidden_channels=int(getattr(head_cfg, "hidden_channels", 64)),
        final_activation=str(getattr(head_cfg, "final_activation", "softplus")),
        initial_density=float(getattr(head_cfg, "initial_density", 0.05)),
        final_weight_std=float(getattr(head_cfg, "final_weight_std", 1e-4)),
    )
    return MoECountNet(backbone, neck, moe, density_head)

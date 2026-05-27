"""Top-level MoECountNet model."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from crowdcount.models.moecount.backbone import MoEConvNeXtBackbone
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.moecount.gcn_refine import DensityGCNRefine
from crowdcount.models.moecount.head import DenseASPPScaleHead, DensityHead, PointPredHead, Stride4RefineHead
from crowdcount.models.moecount.neck import DeepBiFPNNeck, EnhancedFPNNeck


class MoECountNet(nn.Module):
    """Density-map crowd counter with pixel-wise soft-gated MoE and point aux head."""

    def __init__(
        self,
        backbone: MoEConvNeXtBackbone,
        neck: EnhancedFPNNeck | DeepBiFPNNeck,
        moe: HeterogeneousSparseMoE,
        density_head: DensityHead | DenseASPPScaleHead,
        point_head: PointPredHead | None = None,
        gcn_refine: DensityGCNRefine | None = None,
        stride4_refine: Stride4RefineHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.moe = moe
        self.density_head = density_head
        self.point_head = point_head
        self.gcn_refine = gcn_refine
        self.stride4_refine = stride4_refine

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

        neck_out = (
            self.neck(feature_maps["c2"], feature_maps["c3"], feature_maps["c4"])
            if "c4" in feature_maps
            else self.neck(feature_maps["c2"], feature_maps["c3"])
        )
        fused_s8, feat_s16, feat_s32 = neck_out  # all necks now return 3-tuple

        moe_features, moe_aux_losses, route = self.moe(fused_s8, feat_s16, feat_s32)
        if self.gcn_refine is not None:
            moe_features = self.gcn_refine(moe_features)
        density = self.density_head(moe_features)

        # Stride-4 coarse-to-fine refinement (P3-2)
        density_s4 = None
        if self.stride4_refine is not None and "c1" in feature_maps:
            density_s4 = self.stride4_refine(feature_maps["c1"], density)

        moe_aux_total = moe_aux_losses.get("total_aux") if moe_aux_losses else None
        result: dict[str, Any] = {
            "density_out": density,
            "pred_density": density,
            "density_s4": density_s4,
            "moe_aux_losses": moe_aux_losses,
            "moe_aux_total": moe_aux_total,
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
            "expert_similarity": route.get("expert_similarity", {}),
            "expert_densities": route.get("expert_densities"),
        }
        if self.point_head is not None:
            result.update(self.point_head(moe_features))
        return result


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
    num_backbone_levels = len(backbone.out_channels)
    # Neck always receives stride-8/16/32 (c2, c3, c4) regardless of backbone size.
    # With a 4-level backbone, we skip c1 (stride-4) for the neck.
    if num_backbone_levels >= 3:
        neck_ch_offset = num_backbone_levels - 3
        neck = DeepBiFPNNeck(
            c2_channels=backbone.out_channels[neck_ch_offset],
            c3_channels=backbone.out_channels[neck_ch_offset + 1],
            c4_channels=backbone.out_channels[neck_ch_offset + 2],
            out_channels=int(getattr(neck_cfg, "out_channels", 256)),
            num_bifpn_blocks=int(getattr(neck_cfg, "num_bifpn_blocks", 1)),
            branch_channels=tuple(getattr(neck_cfg, "branch_channels", (128, 64, 64))),
            dilations=tuple(getattr(neck_cfg, "dilations", (1, 2, 5))),
            use_spd_downsample=bool(getattr(neck_cfg, "use_spd_downsample", True)),
            use_depthwise_refine=bool(getattr(neck_cfg, "use_depthwise_refine", True)),
        )
    else:
        neck = EnhancedFPNNeck(
            c2_channels=backbone.out_channels[0],
            c3_channels=backbone.out_channels[1],
            out_channels=int(getattr(neck_cfg, "out_channels", 256)),
            branch_channels=tuple(getattr(neck_cfg, "branch_channels", (128, 64, 64))),
            dilations=tuple(getattr(neck_cfg, "dilations", (1, 2, 5))),
        )
    moe = HeterogeneousSparseMoE(
        channels=int(getattr(neck_cfg, "out_channels", 256)),
        gate_type=str(getattr(moe_cfg, "gate_type", "soft")),
        gate_hidden_channels=int(getattr(moe_cfg, "gate_hidden_channels", 128)),
        top_k=int(getattr(moe_cfg, "top_k", 2)),
        temperature_init=float(getattr(moe_cfg, "temperature_init", 1.0)),
        temperature_min=float(getattr(moe_cfg, "temperature_min", 0.1)),
        temperature_decay=float(getattr(moe_cfg, "temperature_decay", 0.98)),
        warmup_fraction=float(getattr(moe_cfg, "warmup_fraction", 0.2)),
        warmup_epochs=getattr(moe_cfg, "warmup_epochs", None),
        lambda_importance=float(getattr(moe_cfg, "lambda_importance", 0.01)),
        lambda_load=float(getattr(moe_cfg, "lambda_load", 0.01)),
        shared_scale=float(getattr(moe_cfg, "shared_scale", 0.3)),
    )
    head_type = str(getattr(head_cfg, "head_type", "standard"))
    if head_type == "denseaspp":
        density_head = DenseASPPScaleHead(
            in_channels=int(getattr(neck_cfg, "out_channels", 256)),
            branch_channels=int(getattr(head_cfg, "branch_channels", 64)),
            dilations=tuple(getattr(head_cfg, "dilations", (1, 3, 6, 9))),
            final_activation=str(getattr(head_cfg, "final_activation", "softplus")),
            initial_density=float(getattr(head_cfg, "initial_density", 0.05)),
            final_weight_std=float(getattr(head_cfg, "final_weight_std", 1e-4)),
        )
    else:
        density_head = DensityHead(
            in_channels=int(getattr(neck_cfg, "out_channels", 256)),
            hidden_channels=int(getattr(head_cfg, "hidden_channels", 128)),
            final_activation=str(getattr(head_cfg, "final_activation", "softplus")),
            initial_density=float(getattr(head_cfg, "initial_density", 0.05)),
            final_weight_std=float(getattr(head_cfg, "final_weight_std", 1e-4)),
        )
    use_point_head = bool(getattr(head_cfg, "use_point_head", True))
    if use_point_head:
        point_head = PointPredHead(
            in_channels=int(getattr(neck_cfg, "out_channels", 256)),
            hidden_channels=int(getattr(head_cfg, "point_hidden_channels", 128)),
            stride=int(getattr(model_cfg, "output_stride", 8)),
        )
    else:
        point_head = None
    use_gcn = bool(getattr(model_cfg, "use_gcn", False))
    gcn_refine = None
    if use_gcn:
        gcn_refine = DensityGCNRefine(
            channels=int(getattr(neck_cfg, "out_channels", 256)),
            k=int(getattr(model_cfg, "gcn_k", 4)),
        )

    use_s4_refine = bool(getattr(model_cfg, "use_stride4_refine", True))
    stride4_refine = None
    if use_s4_refine and num_backbone_levels >= 4:
        s4_channels = backbone.out_channels[0]  # c1 channels (stride-4, e.g., 96 for ConvNeXt-T)
        stride4_refine = Stride4RefineHead(
            s4_channels=s4_channels,
            hidden_channels=int(getattr(head_cfg, "s4_hidden_channels", 64)),
            final_activation=str(getattr(head_cfg, "final_activation", "softplus")),
            initial_density=float(getattr(head_cfg, "s4_initial_density", 0.0125)),
            final_weight_std=float(getattr(head_cfg, "final_weight_std", 1e-4)),
        )

    return MoECountNet(backbone, neck, moe, density_head, point_head, gcn_refine, stride4_refine)

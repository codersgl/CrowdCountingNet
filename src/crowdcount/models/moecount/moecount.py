"""Top-level MoECountNet model."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from crowdcount.models.moecount.backbone import MoEConvNeXtBackbone, MoEVGGBackbone
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.moecount.gcn_refine import DensityGCNRefine
from crowdcount.models.moecount.head import DensityHead, DSGCAnchorPointHead
from crowdcount.models.moecount.neck import DeepBiFPNNeck, EnhancedFPNNeck


class MoECountNet(nn.Module):
    """Density-map crowd counter with pixel-wise soft-gated MoE and point aux head."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: EnhancedFPNNeck | DeepBiFPNNeck,
        moe: HeterogeneousSparseMoE,
        density_head: DensityHead,
        point_head: DSGCAnchorPointHead | None = None,
        gcn_refine: DensityGCNRefine | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.moe = moe
        self.density_head = density_head
        self.point_head = point_head
        self.gcn_refine = gcn_refine

    def supports_moe(self) -> bool:
        return True

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.moe.set_epoch(epoch, total_epochs=total_epochs)

    def update_moe_temperature(self, decay_rate: float | None = None) -> None:
        self.moe.update_temperature(decay_rate=decay_rate)

    def get_gate_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.moe.gate.parameters() if parameter.requires_grad]

    def forward(
        self,
        samples: torch.Tensor,
        epoch: int | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        gt_density: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if hasattr(samples, "tensors"):
            samples = samples.tensors
        if epoch is not None:
            self.moe.set_epoch(epoch)
        feature_maps = self.backbone(samples)
        if "c4" in feature_maps:
            fused_neck = self.neck(feature_maps["c2"], feature_maps["c3"], feature_maps["c4"])
        else:
            fused_neck = self.neck(feature_maps["c2"], feature_maps["c3"])
        moe_features, moe_aux_losses, route = self.moe(
            fused_neck, targets=targets, gt_density=gt_density
        )
        if self.gcn_refine is not None:
            moe_features = self.gcn_refine(moe_features)
        density = self.density_head(moe_features)
        moe_aux_total = moe_aux_losses.get("total_aux") if moe_aux_losses else None
        result: dict[str, Any] = {
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
            "expert_similarity": route.get("expert_similarity", {}),
        }
        if self.point_head is not None:
            result.update(self.point_head(moe_features, samples))
        return result


def build_moecount(cfg: DictConfig) -> MoECountNet:
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg
    backbone_cfg = getattr(model_cfg, "backbone", None)
    neck_cfg = getattr(model_cfg, "neck", None)
    moe_cfg = getattr(model_cfg, "moe", None)
    head_cfg = getattr(model_cfg, "head", None)

    backbone_type = str(getattr(backbone_cfg, "type", "convnext")).lower()

    if backbone_type == "vgg":
        backbone = MoEVGGBackbone(
            vgg_name=str(getattr(backbone_cfg, "vgg_name", "vgg16_bn")),
            pretrained=bool(getattr(backbone_cfg, "pretrained", True)),
            out_levels=int(getattr(backbone_cfg, "num_output_levels", 3)),
        )
    else:
        backbone = MoEConvNeXtBackbone(
            arch=str(getattr(backbone_cfg, "arch", "convnext_tiny")),
            model_name=getattr(backbone_cfg, "model_name", None),
            pretrained=bool(getattr(backbone_cfg, "pretrained", True)),
            pretrained_path=getattr(backbone_cfg, "pretrained_path", None),
            out_indices=tuple(getattr(backbone_cfg, "out_indices", (1, 2))),
        )
    num_backbone_levels = len(backbone.out_channels)
    if num_backbone_levels == 3:
        neck = DeepBiFPNNeck(
            c2_channels=backbone.out_channels[0],
            c3_channels=backbone.out_channels[1],
            c4_channels=backbone.out_channels[2],
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
    deformable_cfg = getattr(moe_cfg, "deformable_expert", None)
    use_deformable = bool(getattr(deformable_cfg, "use_deformable", False))
    expert_config = getattr(moe_cfg, "expert_config", None)
    local_detail_cfg = getattr(expert_config, "local_detail", None) if expert_config is not None else None
    global_density_cfg = getattr(expert_config, "global_density", None) if expert_config is not None else None
    occ_cfg = getattr(expert_config, "occlusion_reasoning", None) if expert_config is not None else None
    dp_cfg = getattr(expert_config, "density_pattern", None) if expert_config is not None else None
    moe = HeterogeneousSparseMoE(
        channels=int(getattr(neck_cfg, "out_channels", 256)),
        gate_hidden_channels=int(getattr(moe_cfg, "gate_hidden_channels", 128)),
        top_k=int(getattr(moe_cfg, "top_k", 2)),
        temperature_init=float(getattr(moe_cfg, "temperature_init", 1.0)),
        temperature_min=float(getattr(moe_cfg, "temperature_min", 0.1)),
        temperature_decay=float(getattr(moe_cfg, "temperature_decay", 0.98)),
        warmup_fraction=float(getattr(moe_cfg, "warmup_fraction", 0.2)),
        warmup_epochs=getattr(moe_cfg, "warmup_epochs", None),
        lambda_importance=float(getattr(moe_cfg, "lambda_importance", 0.01)),
        lambda_load=float(getattr(moe_cfg, "lambda_load", 0.01)),
        shared_scale=float(getattr(moe_cfg, "shared_scale", 0.5)),
        shared_num_blocks=int(getattr(moe_cfg, "shared_num_blocks", 3)),
        shared_scale_learnable=bool(getattr(moe_cfg, "shared_scale_learnable", True)),
        use_deformable_expert=use_deformable,
        deformable_num_heads=int(getattr(deformable_cfg, "num_heads", 4)),
        deformable_num_sampling_points=int(getattr(deformable_cfg, "num_sampling_points", 8)),
        deformable_num_scale_levels=int(getattr(deformable_cfg, "num_scale_levels", 3)),
        deformable_max_offset=float(getattr(deformable_cfg, "max_offset", 8.0)),
        deformable_dropout=float(getattr(deformable_cfg, "dropout", 0.1)),
        deformable_use_se=bool(getattr(deformable_cfg, "use_se", True)),
        deformable_use_density_bias=bool(getattr(deformable_cfg, "use_density_bias", False)),
        use_input_residual=bool(getattr(moe_cfg, "use_input_residual", True)),
        expert_local_detail_use_residual=bool(getattr(local_detail_cfg, "use_residual", True) if local_detail_cfg is not None else True),
        expert_global_density_use_residual=bool(getattr(global_density_cfg, "use_residual", True) if global_density_cfg is not None else True),
        expert_local_detail_use_strip_convs=bool(getattr(local_detail_cfg, "use_strip_convs", True) if local_detail_cfg is not None else True),
        expert_local_detail_strip_kernel=int(getattr(local_detail_cfg, "strip_kernel", 7) if local_detail_cfg is not None else 7),
        expert_local_detail_use_multi_spectral_se=bool(getattr(local_detail_cfg, "use_multi_spectral_se", True) if local_detail_cfg is not None else True),
        expert_local_detail_ms_num_freqs=int(getattr(local_detail_cfg, "ms_num_freqs", 4) if local_detail_cfg is not None else 4),
        expert_local_detail_use_density_adaptive=bool(getattr(local_detail_cfg, "use_density_adaptive", True) if local_detail_cfg is not None else True),
        expert_local_detail_dilations=tuple(int(b) for b in getattr(local_detail_cfg, "dilations", [1, 2, 3])),
        expert_local_detail_groups=int(getattr(local_detail_cfg, "groups", 16) if local_detail_cfg is not None else 16),
        expert_local_detail_ffn_expansion=int(getattr(local_detail_cfg, "ffn_expansion", 2) if local_detail_cfg is not None else 2),
        expert_local_detail_use_density_modulation=bool(getattr(local_detail_cfg, "use_density_modulation", True) if local_detail_cfg is not None else True),
        expert_global_density_use_density=bool(getattr(global_density_cfg, "use_density", True) if global_density_cfg is not None else True),
        gate_type=str(getattr(moe_cfg, "gate_type", "sparse_top2")),
        gate_use_density_hint=bool(getattr(moe_cfg, "gate_use_density_hint", True)),
        gate_use_density_bias=bool(getattr(moe_cfg, "gate_use_density_bias", False)),
        # --- Expert replacement flags ---
        use_point_localization_expert=bool(getattr(moe_cfg, "use_point_localization_expert", True)),
        use_occlusion_reasoning_expert=bool(getattr(moe_cfg, "use_occlusion_reasoning_expert", True)),
        use_density_pattern_expert=bool(getattr(moe_cfg, "use_density_pattern_expert", True)),
        # --- PointLocalizationExpert (e0) config ---
        expert_pl_use_point_aux=bool(getattr(local_detail_cfg, "use_point_aux", True) if local_detail_cfg is not None else False),
        expert_pl_point_hidden=int(getattr(local_detail_cfg, "point_hidden", 64) if local_detail_cfg is not None else 64),
        expert_pl_point_loss_weight=float(getattr(local_detail_cfg, "point_loss_weight", 1.0) if local_detail_cfg is not None else 1.0),
        expert_pl_point_cls_weight=float(getattr(local_detail_cfg, "point_cls_weight", 1.0) if local_detail_cfg is not None else 1.0),
        expert_pl_point_reg_weight=float(getattr(local_detail_cfg, "point_reg_weight", 0.0002) if local_detail_cfg is not None else 0.0002),
        expert_pl_point_cost_class=float(getattr(local_detail_cfg, "point_cost_class", 1.0) if local_detail_cfg is not None else 1.0),
        expert_pl_point_cost_point=float(getattr(local_detail_cfg, "point_cost_point", 0.05) if local_detail_cfg is not None else 0.05),
        expert_pl_point_eos_coef=float(getattr(local_detail_cfg, "point_eos_coef", 0.5) if local_detail_cfg is not None else 0.5),
        expert_pl_point_max_candidates=int(getattr(local_detail_cfg, "point_max_candidates", 512) if local_detail_cfg is not None else 512),
        # --- OcclusionReasoningExpert (e1) config ---
        expert_occ_use_aux=bool(getattr(occ_cfg, "use_aux", True) if occ_cfg is not None else False),
        expert_occ_emb_hidden=int(getattr(occ_cfg, "emb_hidden", 16) if occ_cfg is not None else 16),
        expert_occ_consistency_weight=float(getattr(occ_cfg, "consistency_weight", 1.0) if occ_cfg is not None else 1.0),
        expert_occ_density_threshold=float(getattr(occ_cfg, "density_threshold", 5.0) if occ_cfg is not None else 5.0),
        expert_occ_head_hidden=int(getattr(occ_cfg, "head_hidden", 128) if occ_cfg is not None else 128),
        expert_occ_use_residual=bool(getattr(occ_cfg, "use_residual", True) if occ_cfg is not None else True),
        # --- DensityPatternExpert (e2) config ---
        expert_dp_use_aux=bool(getattr(dp_cfg, "use_aux", True) if dp_cfg is not None else False),
        expert_dp_ppm_bins=tuple(int(b) for b in getattr(dp_cfg, "ppm_bins", [1, 2, 3, 6])),
        expert_dp_ppm_reduction=int(getattr(dp_cfg, "ppm_reduction", 4) if dp_cfg is not None else 4),
        expert_dp_pattern_num_bins=int(getattr(dp_cfg, "pattern_num_bins", 8) if dp_cfg is not None else 8),
        expert_dp_pattern_class_weight=float(getattr(dp_cfg, "pattern_class_weight", 1.0) if dp_cfg is not None else 1.0),
        expert_dp_use_residual=bool(getattr(dp_cfg, "use_residual", True) if dp_cfg is not None else True),
    )
    density_head = DensityHead(
        in_channels=int(getattr(neck_cfg, "out_channels", 256)),
        hidden_channels=int(getattr(head_cfg, "hidden_channels", 128)),
        final_activation=str(getattr(head_cfg, "final_activation", "softplus")),
        initial_density=float(getattr(head_cfg, "initial_density", 0.05)),
        final_weight_std=float(getattr(head_cfg, "final_weight_std", 1e-4)),
        output_kernel_size=int(getattr(head_cfg, "output_kernel_size", 1)),
        use_residual=bool(getattr(head_cfg, "use_residual", False)),
    )
    use_point_head = bool(getattr(head_cfg, "use_point_head", True))
    if use_point_head:
        point_head = DSGCAnchorPointHead(
            in_channels=int(getattr(neck_cfg, "out_channels", 256)),
            feature_size=int(getattr(head_cfg, "point_feature_size", 256)),
            row=int(getattr(head_cfg, "point_anchor_row", 2)),
            line=int(getattr(head_cfg, "point_anchor_line", 2)),
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
    return MoECountNet(backbone, neck, moe, density_head, point_head, gcn_refine)

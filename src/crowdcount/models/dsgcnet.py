"""DSGCNet main model definition."""

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.gcn import (
    CrossStreamGCNProcessor,
    DensityAdaptiveFusion,
    DensityGCNProcessor,
    FeatureGCNProcessor,
    FeatureTransformerProcessor,
    SuperNodeGCNProcessor,
    compute_uncertainty,
)
from crowdcount.models.head import (
    ClassificationModel,
    DecoupledPredictionHead,
    DeepClassificationModel,
    DeepRegressionModel,
    DepthAuxHead,
    DensityAttentionMask,
    Density_pred,
    Density_pred_MS,
    Density_pred_V3,
    EnhancedDensityAttention,
    ForegroundSuppressionBranch,
    GatedDensityAttention,
    FreqDecoupledRouter,
    PointGuidedDensityRefiner,
    PointRefineModule,
    RegressionModel,
    ResidualDensityAttention,
    SharedPredictionTrunk,
    SubPixelRefineModule,
    point_predictions_to_density_map,
    DensityPred_Block3,
    DensityPred_Block4,
    DensityPred_Block5,
)
from crowdcount.models.dap_neck import ACDR, DAPNeck
from crowdcount.models.neck import Decoder_SPD_PAFPN, P2PNeXtDecoder, SPDBiFPNNeck
from crowdcount.models.semc_blocks import SEMCEnhancer
from crowdcount.plugins.gm import GateMechanism, SpatialGateMechanism
from crowdcount.plugins.concat_gate_fusion import ConcatGateFusion
from crowdcount.plugins.depth_cross_attention import DepthCrossAttentionFusion
from crowdcount.plugins.deformable_dual import DeformableDualFusion
from crowdcount.plugins.depth_residual_gating import (
    DepthResidualGating,
    DepthResidualGatingV2,
)
from crowdcount.plugins.geo_prior import DepthGeoPriorAttention
from crowdcount.models.backbone import DepthBackbone_VGG
from crowdcount.plugins.mamba_moe import MambaMoEFusion
from crowdcount.plugins.mamba_vss_dual_fusion import MambaVSSDualFusion
from crowdcount.plugins.moe import LightMoE
from crowdcount.plugins.graph_moe import GraphAwareMoE, GraphMoE
from crowdcount.plugins.sdd_moe import SDDMoE
from crowdcount.plugins.sa_dgat import SADGATFusion
from crowdcount.plugins.msaa import MSAAGate, MSAALite, MsaaAdaptiveLayer
from crowdcount.plugins.MSCA import MSCADecoder, MSCANeck
from crowdcount.plugins.lfem_neck import LFEMMultiScaleNeck
from crowdcount.plugins.rccformer import DensityPredDEAB, MFFMNeck
from crowdcount.plugins.cross_scale_density import (
    CrossScaleDensityRefinement,
    MultiScaleDensityFusion,
)
from crowdcount.plugins.clip_prompt_density import CLIPPromptDensityGuide
from crowdcount.plugins.neck_moe import NeckScaleMoE
from crowdcount.models.moecount.experts import HeterogeneousSparseMoE
from crowdcount.models.scale_decoupled_fusion import ScaleDecoupledFusion


class _DepthEncoder(nn.Module):
    """Lightweight depth feature extractor: 1ch → (d3, d4, d5).

    Produces three feature maps that match the spatial scales of the VGG
    backbone's c3 / c4 / c5 outputs (features_list[1..3]):
        d3: [B, 256, H/4,  W/4 ]
        d4: [B, 512, H/8,  W/8 ]
        d5: [B, 512, H/16, W/16]
    """

    def __init__(self) -> None:
        super().__init__()

        def _block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stage1 = _block(1, 64)  # H/2
        self.stage2 = _block(64, 256)  # H/4  → d3
        self.stage3 = _block(256, 512)  # H/8  → d4
        self.stage4 = _block(512, 512)  # H/16 → d5

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        d3 = self.stage2(x)  # [B, 256, H/4,  W/4 ]
        d4 = self.stage3(d3)  # [B, 512, H/8,  W/8 ]
        d5 = self.stage4(d4)  # [B, 512, H/16, W/16]
        return d3, d4, d5


class _SharedBackboneDepthMix(nn.Module):
    """Layer-wise Mix fusion for shared-backbone RGB/depth features."""

    def __init__(self, init: float = 1.5, num_scales: int = 3) -> None:
        super().__init__()
        self.mix_weights = nn.Parameter(
            torch.full((num_scales,), float(init), dtype=torch.float32)
        )
        self.mix_block = nn.Sigmoid()

    @property
    def mix_factors(self) -> torch.Tensor:
        return self.mix_block(self.mix_weights)

    def forward(
        self,
        rgb_features: tuple[torch.Tensor, ...],
        depth_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(rgb_features) != len(depth_features):
            raise ValueError(
                "RGB and depth feature tuples must have the same number of scales"
            )
        if len(rgb_features) != self.mix_weights.numel():
            raise ValueError(
                f"Expected {self.mix_weights.numel()} feature scales, "
                f"got {len(rgb_features)}"
            )

        fused_features: list[torch.Tensor] = []
        for mix_factor, rgb_feat, depth_feat in zip(
            self.mix_factors, rgb_features, depth_features
        ):
            if rgb_feat.shape != depth_feat.shape:
                raise ValueError(
                    "RGB and depth features must have matching shapes for Mix fusion; "
                    f"got {tuple(rgb_feat.shape)} and {tuple(depth_feat.shape)}"
                )
            mix = mix_factor.to(device=rgb_feat.device, dtype=rgb_feat.dtype).view(
                1, 1, 1, 1
            )
            fused_features.append(rgb_feat * mix + depth_feat * (1.0 - mix))
        return tuple(fused_features)


def _validate_dropout(dropout: float | None, name: str) -> float | None:
    if dropout is None:
        return None
    dropout = float(dropout)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {dropout}")
    return dropout


class DSGCnet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        row: int = 2,
        line: int = 2,
        fusion_mode: str = "gcn",
        use_gm: bool = False,
        gm_input_dim: int = 256,
        gm_hidden_dim: int = 128,
        gm_spatial: bool = True,
        use_msaa: bool = False,
        msaa_in_channels: int = 1280,
        msaa_reduction: int = 4,
        msaa_variant: str = "legacy",
        moe_cfg: DictConfig | None = None,
        graph_attn_moe_cfg: DictConfig | None = None,
        graph_moe_cfg: DictConfig | None = None,
        mamba_moe_cfg: DictConfig | None = None,
        mamba_vss_dual_cfg: DictConfig | None = None,
        sdd_moe_cfg: DictConfig | None = None,
        moecount_moe_cfg: DictConfig | None = None,
        use_depth: bool = False,
        depth_cfg: DictConfig | None = None,
        use_depth_geo: bool = False,
        use_depth_geo_post: bool = False,
        depth_geo_cfg: DictConfig | None = None,
        use_depth_dual_vgg: bool = False,
        depth_dual_vgg_cfg: DictConfig | None = None,
        use_depth_attn: bool = False,
        depth_attn_cfg: DictConfig | None = None,
        use_depth_cross_attn: bool = False,
        depth_cross_attn_cfg: DictConfig | None = None,
        use_depth_aux: bool = False,
        depth_aux_cfg: DictConfig | None = None,
        gcn_adaptive: bool = False,
        gcn_k: int = 4,
        gcn_k_min: int = 2,
        gcn_k_max: int = 8,
        gcn_density_scale: float = 4.0,
        gcn_sim_threshold: float = 0.5,
        gcn_spatial_prior: bool = False,
        gcn_spatial_alpha: float = 1.0,
        gcn_spatial_beta: float = 1.0,
        gcn_mode: str = "fixed",
        gcn_num_supernodes: int = 8,
        gcn_supernode_heads: int = 4,
        cfg: DictConfig | None = None,
        use_dcn: bool = False,
        use_refine: bool = False,
        refine_cfg: DictConfig | None = None,
        use_freq_head: bool = False,
        freq_head_kernel: int = 3,
        use_density_attention: bool = False,
        density_attention_mode: str = "sigmoid",
        density_attention_pre_gcn: bool = False,
        density_attention_hidden: int = 32,
        density_attention_base: float = 0.5,
        density_attention_max_delta: float = 0.5,
        density_attention_strength_init: float = 1e-3,
        density_attention_debug: bool = False,
        use_clip_prompt_density: bool = False,
        clip_prompt_density_cfg: DictConfig | None = None,
        use_subpix_refine: bool = False,
        subpix_refine_cfg: DictConfig | None = None,
        use_uncertainty: bool = False,
        uncertainty_scale: float = 6.0,
        gcn_aniso: bool = False,
        gcn_conv_type: str = "gcn",
        feature_stream_type: str = "gcn",
        feature_transformer_cfg: DictConfig | None = None,
        use_fg_branch: bool = False,
        fg_branch_base: float = 0.5,
        fg_branch_scale: float = 0.5,
        fpn_attention: bool = False,
        use_msca_decoder: bool = False,
        msca_num_heads: int = 8,
        msca_num_blocks: int = 2,
        use_decoupled_head: bool = False,
        use_msca_neck: bool = False,
        use_rccformer_neck: bool = False,
        rccformer_deab_blocks: int = 2,
        use_dap_neck: bool = False,
        dap_neck_cfg: DictConfig | None = None,
        use_bifpn_neck: bool = False,
        bifpn_neck_cfg: DictConfig | None = None,
        use_p2pnext_neck: bool = False,
        p2pnext_neck_cfg: DictConfig | None = None,
        use_lfem_neck: bool = False,
        lfem_neck_cfg: DictConfig | None = None,
        neck_acdr_cfg: DictConfig | None = None,
        use_neck_moe: bool = False,
        neck_moe_cfg: DictConfig | None = None,
        use_deep_head: bool = False,
        use_density_adaptive_fusion: bool = False,
        density_adaptive_fusion_cfg: DictConfig | None = None,
        neck_dropout: float = 0.0,
        head_dropout: float = 0.0,
        density_dropout: float | None = None,
        gcn_dropout: float | None = None,
        regularization_drop_path: float | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg
        self.neck_dropout = float(
            _validate_dropout(neck_dropout, "neck_dropout") or 0.0
        )
        head_dropout = float(_validate_dropout(head_dropout, "head_dropout") or 0.0)
        density_dropout = _validate_dropout(density_dropout, "density_dropout")
        gcn_dropout = _validate_dropout(gcn_dropout, "gcn_dropout")
        regularization_drop_path = _validate_dropout(
            regularization_drop_path, "drop_path"
        )
        self.fusion_mode = fusion_mode
        self.use_gcn_moe = fusion_mode == "gcn_moe"
        self.use_graph_attn_moe = fusion_mode == "graph_attn_moe"
        self.use_graph_moe = fusion_mode == "graph_moe"
        self.use_mamba_moe = fusion_mode == "mamba_moe"
        self.use_mamba_vss_dual = fusion_mode == "mamba_vss_dual"
        self.use_sdd_moe = fusion_mode == "sdd_moe"
        self.use_sa_dgat = fusion_mode == "sa_dgat"
        self.use_deformable_dual = fusion_mode == "deformable_dual"
        self.use_moecount_moe = fusion_mode == "moe"
        self.use_scale_decoupled = fusion_mode == "scale_decoupled"
        self.scale_decoupled_fusion: ScaleDecoupledFusion | None = None
        self.mamba_vss_dual: MambaVSSDualFusion | None = None
        self.sdd_moe: SDDMoE | None = None
        self.sa_dgat_fusion: SADGATFusion | None = None
        self.deformable_dual_fusion: DeformableDualFusion | None = None
        self.moecount_moe: HeterogeneousSparseMoE | None = None
        self.graph_moe: GraphMoE | None = None
        self.use_depth = use_depth
        self.use_depth_geo = use_depth_geo
        self.use_depth_geo_post = use_depth_geo_post
        self.use_depth_dual_vgg = use_depth_dual_vgg
        self.use_depth_attn = use_depth_attn
        self.use_depth_cross_attn = use_depth_cross_attn
        self.use_depth_aux = use_depth_aux

        # Mutual exclusion: only one depth fusion path at a time
        _depth_flags = sum(
            [
                use_depth,
                use_depth_geo,
                use_depth_geo_post,
                use_depth_dual_vgg,
                use_depth_attn,
                use_depth_cross_attn,
            ]
        )
        if _depth_flags > 1:
            raise ValueError(
                "At most one depth fusion path may be enabled. Got: "
                f"use_depth={use_depth}, use_depth_geo={use_depth_geo}, "
                f"use_depth_geo_post={use_depth_geo_post}, "
                f"use_depth_dual_vgg={use_depth_dual_vgg}, "
                f"use_depth_attn={use_depth_attn}, "
                f"use_depth_cross_attn={use_depth_cross_attn}"
            )
        self.use_freq_head = use_freq_head
        self.use_density_attention = use_density_attention
        self.density_attention_debug = density_attention_debug
        self.use_subpix_refine = use_subpix_refine
        self._gcn_mode = gcn_mode
        self.feature_stream_type = str(feature_stream_type).lower()
        self.use_uncertainty = use_uncertainty
        self.use_msca_decoder = use_msca_decoder
        self.use_decoupled_head = use_decoupled_head
        self.use_msca_neck = use_msca_neck
        self.use_rccformer_neck = use_rccformer_neck
        self.use_dap_neck = use_dap_neck
        self.use_bifpn_neck = use_bifpn_neck
        self.use_p2pnext_neck = use_p2pnext_neck
        self.use_lfem_neck = use_lfem_neck
        self.use_neck_moe = use_neck_moe
        self.use_density_adaptive_fusion = use_density_adaptive_fusion

        model_cfg = getattr(cfg, "model", cfg) if cfg is not None else None
        point_feedback_cfg = (
            getattr(model_cfg, "point_density_feedback", None)
            if model_cfg is not None
            else None
        )
        self.point_density_feedback_enabled = bool(
            getattr(point_feedback_cfg, "enabled", False)
            if point_feedback_cfg is not None
            else False
        )
        self.point_density_feedback_detach_points = bool(
            getattr(point_feedback_cfg, "detach_points", True)
            if point_feedback_cfg is not None
            else True
        )
        self.point_density_feedback_detach_scores = bool(
            getattr(point_feedback_cfg, "detach_scores", True)
            if point_feedback_cfg is not None
            else True
        )
        self.point_density_feedback_score_threshold = float(
            getattr(point_feedback_cfg, "score_threshold", 0.0)
            if point_feedback_cfg is not None
            else 0.0
        )
        self.point_density_feedback_gaussian_sigma = float(
            getattr(point_feedback_cfg, "gaussian_sigma", 1.0)
            if point_feedback_cfg is not None
            else 1.0
        )
        self.point_density_feedback_debug = bool(
            getattr(point_feedback_cfg, "debug", False)
            if point_feedback_cfg is not None
            else False
        )
        if self.point_density_feedback_gaussian_sigma < 0.0:
            raise ValueError("point_density_feedback.gaussian_sigma must be non-negative")
        if self.point_density_feedback_score_threshold < 0.0:
            raise ValueError("point_density_feedback.score_threshold must be non-negative")
        self.point_density_refiner: PointGuidedDensityRefiner | None = (
            PointGuidedDensityRefiner(
                feature_channels=256,
                hidden_channels=int(
                    getattr(point_feedback_cfg, "hidden_channels", 32)
                    if point_feedback_cfg is not None
                    else 32
                ),
                max_delta=float(
                    getattr(point_feedback_cfg, "max_delta", 0.5)
                    if point_feedback_cfg is not None
                    else 0.5
                ),
                strength_init=float(
                    getattr(point_feedback_cfg, "strength_init", 1e-3)
                    if point_feedback_cfg is not None
                    else 1e-3
                ),
            )
            if self.point_density_feedback_enabled
            else None
        )

        _neck_flags = sum(
            [
                use_msca_neck,
                use_msca_decoder,
                use_rccformer_neck,
                use_dap_neck,
                use_bifpn_neck,
                use_p2pnext_neck,
                use_lfem_neck,
            ]
        )
        if _neck_flags > 1:
            raise ValueError(
                "Neck options are mutually exclusive; enable at most one. Got: "
                f"use_msca_neck={use_msca_neck}, use_msca_decoder={use_msca_decoder}, "
                f"use_rccformer_neck={use_rccformer_neck}, use_dap_neck={use_dap_neck}, "
                f"use_bifpn_neck={use_bifpn_neck}, "
                f"use_p2pnext_neck={use_p2pnext_neck}, "
                f"use_lfem_neck={use_lfem_neck}"
            )

        if use_lfem_neck and use_msaa and msaa_variant == "legacy":
            raise ValueError(
                "use_lfem_neck does not support legacy MSAA because legacy MSAA "
                "changes C3/C4/C5 to 1280 channels. Use msaa_variant='lite' or "
                "disable use_msaa for LFEM neck ablations."
            )

        if use_depth_cross_attn and use_msca_decoder:
            raise ValueError(
                "use_depth_cross_attn is not supported with use_msca_decoder because "
                "MSCADecoder produces density before post-neck fusion."
            )

        if use_depth_geo_post and use_msca_decoder:
            raise ValueError(
                "use_depth_geo_post is not supported with use_msca_decoder because "
                "MSCADecoder produces density before post-neck fusion."
            )

        if use_msca_neck and use_msca_decoder:
            raise ValueError(
                "use_msca_neck and use_msca_decoder are mutually exclusive; "
                "enable only one."
            )

        if use_clip_prompt_density and use_msca_decoder:
            raise ValueError(
                "use_clip_prompt_density is not supported with use_msca_decoder because "
                "MSCADecoder owns density prediction internally."
            )

        neck_moe_position = str(
            getattr(neck_moe_cfg, "position", "pre_acdr")
            if neck_moe_cfg is not None
            else "pre_acdr"
        ).lower()
        if neck_moe_position not in {"pre_acdr", "post_acdr"}:
            raise ValueError(
                "neck_moe.position must be 'pre_acdr' or 'post_acdr', "
                f"got {neck_moe_position!r}"
            )
        neck_moe_use_pyramid_context = bool(
            getattr(neck_moe_cfg, "use_pyramid_context", True)
            if neck_moe_cfg is not None
            else True
        )
        neck_moe_allow_with_fusion_moe = bool(
            getattr(neck_moe_cfg, "allow_with_fusion_moe", False)
            if neck_moe_cfg is not None
            else False
        )
        if use_neck_moe and use_msca_decoder:
            raise ValueError(
                "use_neck_moe is not supported with use_msca_decoder because "
                "MSCADecoder owns the neck feature contract."
            )
        if (
            use_neck_moe
            and self.fusion_mode != "gcn"
            and not neck_moe_allow_with_fusion_moe
        ):
            raise ValueError(
                "use_neck_moe v1 is supported with fusion_mode='gcn'. Set "
                "neck_moe.allow_with_fusion_moe=true only for explicit ablations."
            )
        if use_neck_moe and neck_moe_use_pyramid_context and (
            use_msca_neck or use_rccformer_neck
        ):
            raise ValueError(
                "neck_moe.use_pyramid_context=true requires a neck that returns "
                "P3/P4/P5 intermediates. Set use_pyramid_context=false for "
                "MSCA/RCCFormer neck ablations."
            )

        if self.fusion_mode not in {
            "gcn",
            "gcn_moe",
            "graph_attn_moe",
            "graph_moe",
            "mamba_moe",
            "mamba_vss_dual",
            "sdd_moe",
            "sa_dgat",
            "deformable_dual",
            "moe",
            "scale_decoupled",
        }:
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode}, expected 'gcn', 'gcn_moe', 'graph_attn_moe', 'graph_moe', 'mamba_moe', 'mamba_vss_dual', 'sdd_moe', 'sa_dgat', 'deformable_dual', 'moe', or 'scale_decoupled'"
            )
        if self.feature_stream_type not in {"gcn", "transformer", "window_transformer"}:
            raise ValueError(
                "feature_stream_type must be 'gcn', 'transformer', or "
                f"'window_transformer', got {feature_stream_type!r}"
            )

        density_cfg = (
            getattr(cfg, "density_multi_scale", None) if cfg is not None else None
        )
        self.use_multi_scale_density = bool(
            getattr(density_cfg, "enabled", False) if density_cfg is not None else False
        )
        num_anchor_points = row * line

        self.pred_trunk: SharedPredictionTrunk | DecoupledPredictionHead = (
            DecoupledPredictionHead(
                in_channels=256,
                feature_size=256,
                dropout=head_dropout,
            )
            if use_decoupled_head
            else SharedPredictionTrunk(
                in_channels=256,
                feature_size=256,
                dropout=head_dropout,
            )
        )
        if use_deep_head:
            self.regression = DeepRegressionModel(
                num_features_in=256,
                num_anchor_points=num_anchor_points,
                dropout=head_dropout,
            )
            self.classification = DeepClassificationModel(
                num_features_in=256,
                num_classes=self.num_classes,
                num_anchor_points=num_anchor_points,
                dropout=head_dropout,
            )
        else:
            self.regression = RegressionModel(
                num_features_in=256, num_anchor_points=num_anchor_points
            )
            self.classification = ClassificationModel(
                num_features_in=256,
                num_classes=self.num_classes,
                num_anchor_points=num_anchor_points,
            )

        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)

        def _build_standard_density_head() -> nn.Module:
            model_cfg = getattr(cfg, "model", None) if cfg is not None else None
            density_version = str(getattr(model_cfg, "density_head_version", "v1"))
            density_dropout_v1 = 0.0 if density_dropout is None else density_dropout
            density_dropout_v3 = 0.1 if density_dropout is None else density_dropout
            if (
                bool(getattr(model_cfg, "use_ms_density_head", False))
                and density_version == "v1"
            ):
                density_version = "ms"
            if density_version == "v3":
                upsample = bool(getattr(model_cfg, "density_head_upsample", False))
                return Density_pred_V3(upsample=upsample, dropout=density_dropout_v3)
            if density_version == "ms":
                return Density_pred_MS(dropout=density_dropout_v1)
            return Density_pred(dropout=density_dropout_v1)

        def _neck_acdr_enabled() -> bool:
            enabled = getattr(neck_acdr_cfg, "enabled", "auto")
            if isinstance(enabled, str):
                enabled = enabled.lower()
                if enabled == "auto":
                    return use_dap_neck
                if enabled in {"true", "yes", "1"}:
                    return True
                if enabled in {"false", "no", "0"}:
                    return False
            return bool(enabled)

        use_neck_acdr = _neck_acdr_enabled()
        if use_msca_decoder and use_neck_acdr:
            raise ValueError(
                "neck_acdr is not supported with use_msca_decoder because "
                "MSCADecoder owns both neck features and density prediction."
            )

        acdr_enabled_value = str(getattr(neck_acdr_cfg, "enabled", "auto")).lower()
        if use_dap_neck and acdr_enabled_value == "auto":
            acdr_large_kernel = int(
                getattr(
                    dap_neck_cfg,
                    "acdr_large_kernel",
                    getattr(neck_acdr_cfg, "large_kernel", 7),
                )
            )
            acdr_dilation = int(
                getattr(
                    dap_neck_cfg,
                    "acdr_dilation",
                    getattr(neck_acdr_cfg, "dilation", 2),
                )
            )
        else:
            acdr_large_kernel = int(getattr(neck_acdr_cfg, "large_kernel", 7))
            acdr_dilation = int(getattr(neck_acdr_cfg, "dilation", 2))
        self.neck_acdr = (
            ACDR(
                channels=256,
                large_kernel=acdr_large_kernel,
                dilation=acdr_dilation,
                hidden_ratio=int(getattr(neck_acdr_cfg, "hidden_ratio", 4)),
                gate_init=float(getattr(neck_acdr_cfg, "gate_init", 0.0)),
            )
            if use_neck_acdr
            else None
        )

        if use_dap_neck:
            # DAP-Neck v2 fusion backbone; ACDR is wired as post-neck module.
            self.msca_decoder = None
            _dn = dap_neck_cfg
            self.pa = DAPNeck(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=256,
                use_peem=bool(getattr(_dn, "use_peem", False)) if _dn else False,
                freq_cutoff=float(getattr(_dn, "freq_cutoff", 0.25)) if _dn else 0.25,
                use_dcn=bool(getattr(_dn, "use_dcn", False)) if _dn else False,
                use_acdr=False,
            )
            self.density_pred = _build_standard_density_head()
        elif use_bifpn_neck:
            # SPD-BiFPN: weighted bidirectional fusion with detail-preserving SPD.
            self.msca_decoder = None
            _bn = bifpn_neck_cfg
            self.pa = SPDBiFPNNeck(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=256,
                num_blocks=int(getattr(_bn, "num_blocks", 1)) if _bn else 1,
                use_spd_downsample=bool(getattr(_bn, "use_spd_downsample", True))
                if _bn
                else True,
                use_depthwise_refine=bool(
                    getattr(_bn, "use_depthwise_refine", True)
                )
                if _bn
                else True,
                eps=float(getattr(_bn, "eps", 1e-4)) if _bn else 1e-4,
            )
            self.density_pred = _build_standard_density_head()
        elif use_p2pnext_neck:
            self.msca_decoder = None
            _pn = p2pnext_neck_cfg
            self.pa = P2PNeXtDecoder(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=int(getattr(_pn, "feature_size", 256)) if _pn else 256,
                output_level=str(getattr(_pn, "output_level", "p3"))
                if _pn
                else "p3",
            )
            self.density_pred = _build_standard_density_head()
        elif use_lfem_neck:
            # LFEM neck: three parallel LFEM branches over C3/C4/C5.
            self.msca_decoder = None
            _ln = lfem_neck_cfg
            lfem_feature_size = int(getattr(_ln, "feature_size", 256)) if _ln else 256
            if lfem_feature_size != 256:
                raise ValueError(
                    "lfem_neck.feature_size must be 256 because DSGCNet's "
                    "density, GCN, and point heads are wired for 256 channels."
                )
            self.pa = LFEMMultiScaleNeck(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=lfem_feature_size,
                use_spd_downsample=bool(
                    getattr(_ln, "use_spd_downsample", True)
                )
                if _ln
                else True,
                fusion_eps=float(getattr(_ln, "fusion_eps", 1e-4))
                if _ln
                else 1e-4,
                upsample_mode=str(getattr(_ln, "upsample_mode", "nearest"))
                if _ln
                else "nearest",
            )
            self.density_pred = _build_standard_density_head()
        elif use_rccformer_neck:
            # RCCFormer MFFM neck + DEAB/ASAM density head
            self.msca_decoder = None
            self.pa = MFFMNeck(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                dim=256,
            )
            self.density_pred = DensityPredDEAB(
                dim=256,
                num_deab=rccformer_deab_blocks,
            )
        elif use_msca_decoder:
            # MSCADecoder replaces PA-FPN + Density_pred + GCN in one module
            self.msca_decoder: MSCADecoder | None = MSCADecoder(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=256,
                num_heads=msca_num_heads,
                num_blocks=msca_num_blocks,
            )
            self.pa = None  # type: ignore[assignment]
            self.density_pred = None  # type: ignore[assignment]
        elif use_msca_neck:
            # MSCANeck replaces PA-FPN only; density_pred + GCN still active
            self.msca_decoder = None
            self.pa = MSCANeck(
                C3_size=256,
                C4_size=512,
                C5_size=512,
                feature_size=256,
                num_heads=msca_num_heads,
                num_blocks=msca_num_blocks,
            )
            self.density_pred = Density_pred(
                dropout=0.0 if density_dropout is None else density_dropout
            )
        else:
            self.msca_decoder = None
            if use_msaa and msaa_variant == "legacy":
                self.pa = Decoder_SPD_PAFPN(
                    1280, 1280, 1280, use_dcn=use_dcn, fpn_attention=fpn_attention
                )
            else:
                self.pa = Decoder_SPD_PAFPN(
                    256, 512, 512, use_dcn=use_dcn, fpn_attention=fpn_attention
                )
            self.density_pred = Density_pred(
                dropout=0.0 if density_dropout is None else density_dropout
            )

        if use_neck_moe:
            _nm = neck_moe_cfg
            _rates_raw = getattr(_nm, "context_rates", [1, 3, 5]) if _nm else [1, 3, 5]
            self.neck_moe: NeckScaleMoE | None = NeckScaleMoE(
                in_channels=int(getattr(_nm, "in_channels", 256)) if _nm else 256,
                num_experts=int(getattr(_nm, "num_experts", 4)) if _nm else 4,
                grid_stride=int(getattr(_nm, "grid_stride", 4)) if _nm else 4,
                routing=str(getattr(_nm, "routing", "soft")) if _nm else "soft",
                top_k=int(getattr(_nm, "top_k", 0)) if _nm else 0,
                use_pyramid_context=neck_moe_use_pyramid_context,
                lambda_balance=float(getattr(_nm, "lambda_balance", 0.01))
                if _nm
                else 0.01,
                gate_init=float(getattr(_nm, "gate_init", 0.0)) if _nm else 0.0,
                context_rates=tuple(int(rate) for rate in _rates_raw),
            )
        else:
            self.neck_moe = None
        self.neck_moe_position = neck_moe_position
        self.neck_moe_use_pyramid_context = neck_moe_use_pyramid_context
        self.depth_aux_head: DepthAuxHead | None = (
            DepthAuxHead(
                in_channels=int(getattr(depth_aux_cfg, "in_channels", 256)),
                hidden_channels=int(getattr(depth_aux_cfg, "hidden_channels", 64)),
                num_layers=int(getattr(depth_aux_cfg, "num_layers", 3)),
                dropout=float(getattr(depth_aux_cfg, "dropout", 0.0)),
                detach_features=bool(getattr(depth_aux_cfg, "detach_features", False)),
            )
            if use_depth_aux
            else None
        )

        self.clip_prompt_density: CLIPPromptDensityGuide | None
        self.clip_prompt_density_apply_to = "density_only"
        self.clip_prompt_density_debug = False
        if use_clip_prompt_density:
            _cpd = clip_prompt_density_cfg
            _model_cfg = getattr(cfg, "model", cfg) if cfg is not None else None
            _clip_model = str(
                getattr(
                    _cpd,
                    "clip_model",
                    getattr(_model_cfg, "backbone", "ViT-B-16")
                    if _model_cfg is not None
                    else "ViT-B-16",
                )
            )
            self.clip_prompt_density_apply_to = str(
                getattr(_cpd, "apply_to", "density_only")
                if _cpd is not None
                else "density_only"
            )
            if self.clip_prompt_density_apply_to not in {"density_only", "shared"}:
                raise ValueError(
                    "clip_prompt_density.apply_to must be 'density_only' or 'shared', "
                    f"got {self.clip_prompt_density_apply_to!r}"
                )
            self.clip_prompt_density_debug = bool(
                getattr(_cpd, "debug", False) if _cpd is not None else False
            )
            self.clip_prompt_density = CLIPPromptDensityGuide(
                feature_channels=256,
                clip_model=_clip_model,
                pretrained=getattr(_cpd, "pretrained", True)
                if _cpd is not None
                else True,
                positive_prompts=getattr(_cpd, "positive_prompts", None)
                if _cpd is not None
                else None,
                negative_prompts=getattr(_cpd, "negative_prompts", None)
                if _cpd is not None
                else None,
                temperature=float(getattr(_cpd, "temperature", 0.07))
                if _cpd is not None
                else 0.07,
                hidden_dim=int(getattr(_cpd, "hidden_dim", 128))
                if _cpd is not None
                else 128,
                max_delta=float(getattr(_cpd, "max_delta", 0.5))
                if _cpd is not None
                else 0.5,
                strength_init=float(getattr(_cpd, "strength_init", 1e-3))
                if _cpd is not None
                else 1e-3,
            )
        else:
            self.clip_prompt_density = None
        self.density_attention: (
            DensityAttentionMask
            | EnhancedDensityAttention
            | GatedDensityAttention
            | ResidualDensityAttention
            | None
        )
        if use_density_attention:
            if density_attention_mode == "enhanced":
                self.density_attention = EnhancedDensityAttention(
                    feature_channels=256,
                    hidden_channels=density_attention_hidden,
                    base_init=density_attention_base,
                )
            elif density_attention_mode == "gated":
                self.density_attention = GatedDensityAttention(
                    feature_channels=256,
                    hidden_channels=density_attention_hidden,
                )
            elif density_attention_mode in {"residual", "calibrated"}:
                self.density_attention = ResidualDensityAttention(
                    hidden_channels=density_attention_hidden,
                    max_delta=density_attention_max_delta,
                    strength_init=density_attention_strength_init,
                )
            else:
                self.density_attention = DensityAttentionMask(
                    mode=density_attention_mode
                )
        else:
            self.density_attention = None

        # Pre-GCN density attention (lightweight sigmoid gate, independent weights)
        self.density_attention_pre_gcn: DensityAttentionMask | None = (
            DensityAttentionMask(mode="sigmoid") if density_attention_pre_gcn else None
        )

        # Multi-scale density prediction (optional)
        self.use_cross_scale_refine = bool(
            getattr(density_cfg, "cross_scale_refine", False)
            if density_cfg is not None
            else False
        )
        self.use_fuse_to_gcn = bool(
            getattr(density_cfg, "fuse_to_gcn", False)
            if density_cfg is not None
            else False
        )
        if self.use_multi_scale_density:
            if self.use_cross_scale_refine:
                # Coarse-to-fine refinement replaces independent heads
                self.cross_scale_refine = CrossScaleDensityRefinement()
            else:
                self.density_pred_block3 = DensityPred_Block3()
                self.density_pred_block4 = DensityPred_Block4()
                self.density_pred_block5 = DensityPred_Block5()
            if self.use_fuse_to_gcn:
                self.density_fuse = MultiScaleDensityFusion(num_scales=4)

        if self.use_graph_moe:
            _gm_cfg = graph_moe_cfg
            self.graph_moe = GraphMoE(
                input_dim=256,
                num_experts=int(getattr(_gm_cfg, "num_experts", 5))
                if _gm_cfg
                else 5,
                top_k=int(getattr(_gm_cfg, "top_k", 2)) if _gm_cfg else 2,
                router_temperature=float(
                    getattr(_gm_cfg, "router_temperature", 1.0)
                )
                if _gm_cfg
                else 1.0,
                noisy_routing_std=float(getattr(_gm_cfg, "noisy_routing_std", 0.0))
                if _gm_cfg
                else 0.0,
                grid_stride=int(getattr(_gm_cfg, "grid_stride", 4))
                if _gm_cfg
                else 4,
                router_detach_density=bool(
                    getattr(_gm_cfg, "router_detach_density", True)
                )
                if _gm_cfg
                else True,
                use_uncertainty_hint=bool(
                    getattr(_gm_cfg, "use_uncertainty_hint", True)
                )
                if _gm_cfg
                else True,
                use_coordinate_hint=bool(
                    getattr(_gm_cfg, "use_coordinate_hint", True)
                )
                if _gm_cfg
                else True,
                expert_prior=tuple(getattr(_gm_cfg, "expert_prior", []))
                if _gm_cfg and getattr(_gm_cfg, "expert_prior", None) is not None
                else None,
                aux_loss_weight=float(getattr(_gm_cfg, "aux_loss_weight", 1.0))
                if _gm_cfg
                else 1.0,
                lambda_balance=float(getattr(_gm_cfg, "lambda_balance", 0.01))
                if _gm_cfg
                else 0.01,
                lambda_importance=float(getattr(_gm_cfg, "lambda_importance", 0.01))
                if _gm_cfg
                else 0.01,
                lambda_capacity=float(getattr(_gm_cfg, "lambda_capacity", 0.0))
                if _gm_cfg
                else 0.0,
                router_z_loss_weight=float(
                    getattr(_gm_cfg, "router_z_loss_weight", 0.0)
                )
                if _gm_cfg
                else 0.0,
                capacity_factor=float(getattr(_gm_cfg, "capacity_factor", 1.25))
                if _gm_cfg
                else 1.25,
                local_kernels=tuple(getattr(_gm_cfg, "local_kernels", [1, 3, 5]))
                if _gm_cfg
                else (1, 3, 5),
                local_expansion=int(getattr(_gm_cfg, "local_expansion", 2))
                if _gm_cfg
                else 2,
                local_use_density_gate=bool(
                    getattr(_gm_cfg, "local_use_density_gate", True)
                )
                if _gm_cfg
                else True,
                local_window_size=int(getattr(_gm_cfg, "local_window_size", 0))
                if _gm_cfg
                else 0,
                num_heads=int(getattr(_gm_cfg, "num_heads", 4)) if _gm_cfg else 4,
                use_density_bias=bool(getattr(_gm_cfg, "use_density_bias", True))
                if _gm_cfg
                else True,
                density_bias_scale=float(
                    getattr(_gm_cfg, "density_bias_scale", 1.0)
                )
                if _gm_cfg
                else 1.0,
                attn_dropout=float(getattr(_gm_cfg, "attn_dropout", 0.1))
                if _gm_cfg
                else 0.1,
                scale_dilations=tuple(getattr(_gm_cfg, "scale_dilations", [1, 2, 4]))
                if _gm_cfg
                else (1, 2, 4),
                background_max_suppression=float(
                    getattr(_gm_cfg, "background_max_suppression", 0.5)
                )
                if _gm_cfg
                else 0.5,
                residual_gate_init=float(
                    getattr(_gm_cfg, "residual_gate_init", 1.0)
                )
                if _gm_cfg
                else 1.0,
                disabled_experts=tuple(getattr(_gm_cfg, "disabled_experts", []))
                if _gm_cfg
                else (),
                disable_local_occlusion=bool(
                    getattr(_gm_cfg, "disable_local_occlusion", False)
                )
                if _gm_cfg
                else False,
                disable_nonlocal_context=bool(
                    getattr(_gm_cfg, "disable_nonlocal_context", False)
                )
                if _gm_cfg
                else False,
                disable_tiny_perspective=bool(
                    getattr(_gm_cfg, "disable_tiny_perspective", False)
                )
                if _gm_cfg
                else False,
                disable_scale_specialist=bool(
                    getattr(_gm_cfg, "disable_scale_specialist", False)
                )
                if _gm_cfg
                else False,
                disable_background_suppress=bool(
                    getattr(_gm_cfg, "disable_background_suppress", False)
                )
                if _gm_cfg
                else False,
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.mamba_moe = None
            self.sdd_moe = None
        elif self.use_graph_attn_moe:
            _gam = graph_attn_moe_cfg
            self.graph_attn_moe: GraphAwareMoE | None = GraphAwareMoE(
                input_dim=256,
                num_heads=int(getattr(_gam, "num_heads", 4)) if _gam else 4,
                use_density_bias=bool(getattr(_gam, "use_density_bias", True))
                if _gam
                else True,
                density_bias_scale=float(getattr(_gam, "density_bias_scale", 1.0))
                if _gam
                else 1.0,
                attn_dropout=float(getattr(_gam, "attn_dropout", 0.1)) if _gam else 0.1,
                local_kernels=tuple(getattr(_gam, "local_kernels", [1, 3, 5]))
                if _gam
                else (1, 3, 5),
                local_expansion=int(getattr(_gam, "local_expansion", 4)) if _gam else 4,
                local_use_density_gate=bool(
                    getattr(_gam, "local_use_density_gate", True)
                )
                if _gam
                else True,
                local_window_size=int(getattr(_gam, "local_window_size", 0))
                if _gam
                else 0,
                grid_stride=int(getattr(_gam, "grid_stride", 4)) if _gam else 4,
                local_prior=float(getattr(_gam, "local_prior", 0.0)) if _gam else 0.0,
                lambda_balance=float(getattr(_gam, "lambda_balance", 0.01))
                if _gam
                else 0.01,
                router_detach_density=bool(getattr(_gam, "router_detach_density", True))
                if _gam
                else True,
                disable_graph_bias=bool(getattr(_gam, "disable_graph_bias", False))
                if _gam
                else False,
                disable_local_expert=bool(getattr(_gam, "disable_local_expert", False))
                if _gam
                else False,
                disable_global_expert=bool(
                    getattr(_gam, "disable_global_expert", False)
                )
                if _gam
                else False,
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.mamba_moe = None
        elif self.use_mamba_moe:
            _mmm = mamba_moe_cfg
            self.mamba_moe: MambaMoEFusion | None = MambaMoEFusion(
                input_dim=256,
                d_state=int(getattr(_mmm, "d_state", 16)) if _mmm else 16,
                d_conv=int(getattr(_mmm, "d_conv", 3)) if _mmm else 3,
                expand=float(getattr(_mmm, "expand", 2.0)) if _mmm else 2.0,
                num_experts=int(getattr(_mmm, "num_experts", 4)) if _mmm else 4,
                top_k=int(getattr(_mmm, "top_k", 2)) if _mmm else 2,
                lr_space=str(getattr(_mmm, "lr_space", "exp")) if _mmm else "exp",
                num_blocks=int(getattr(_mmm, "num_blocks", 1)) if _mmm else 1,
                mlp_hidden=int(getattr(_mmm, "mlp_hidden", 256)) if _mmm else 256,
                drop_path=(
                    regularization_drop_path
                    if regularization_drop_path is not None
                    else float(getattr(_mmm, "drop_path", 0.1))
                    if _mmm
                    else 0.1
                ),
                lambda_balance=float(getattr(_mmm, "lambda_balance", 0.01))
                if _mmm
                else 0.01,
                use_density_hint=bool(getattr(_mmm, "use_density_hint", False))
                if _mmm
                else False,
                d_spectral=int(getattr(_mmm, "d_spectral", 256)) if _mmm else 256,
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.sdd_moe = None
        elif self.use_mamba_vss_dual:
            _mvd = mamba_vss_dual_cfg
            self.mamba_vss_dual = MambaVSSDualFusion(
                in_channels=256,
                density_embed_dim=int(getattr(_mvd, "density_embed_dim", 64))
                if _mvd
                else 64,
                d_state=int(getattr(_mvd, "d_state", 16)) if _mvd else 16,
                d_conv=int(getattr(_mvd, "d_conv", 3)) if _mvd else 3,
                mlp_ratio=float(getattr(_mvd, "mlp_ratio", 2.0)) if _mvd else 2.0,
                vss_low_dim=(
                    int(getattr(_mvd, "vss_low_dim"))
                    if _mvd is not None and getattr(_mvd, "vss_low_dim", None) is not None
                    else None
                ),
                num_vss_blocks=int(getattr(_mvd, "num_vss_blocks", 1))
                if _mvd
                else 1,
                num_moe_blocks=int(getattr(_mvd, "num_moe_blocks", 1))
                if _mvd
                else 1,
                num_experts=int(getattr(_mvd, "num_experts", 4)) if _mvd else 4,
                top_k=int(getattr(_mvd, "top_k", 2)) if _mvd else 2,
                lr_space=str(getattr(_mvd, "lr_space", "exp")) if _mvd else "exp",
                expand=float(getattr(_mvd, "expand", 2.0)) if _mvd else 2.0,
                d_spectral=int(getattr(_mvd, "d_spectral", 256)) if _mvd else 256,
                mlp_hidden=int(getattr(_mvd, "mlp_hidden", 256)) if _mvd else 256,
                drop_path=(
                    regularization_drop_path
                    if regularization_drop_path is not None
                    else float(getattr(_mvd, "drop_path", 0.1))
                    if _mvd
                    else 0.1
                ),
                lambda_balance=float(getattr(_mvd, "lambda_balance", 0.01))
                if _mvd
                else 0.01,
                use_density_hint=bool(getattr(_mvd, "use_density_hint", True))
                if _mvd
                else True,
                fusion_spatial=bool(getattr(_mvd, "fusion_spatial", True))
                if _mvd
                else True,
                gate_init=float(getattr(_mvd, "gate_init", 1e-3))
                if _mvd
                else 1e-3,
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.sdd_moe = None
        elif self.use_sdd_moe:
            self.sdd_moe: SDDMoE | None = SDDMoE(
                in_channels=256,
                cfg=sdd_moe_cfg,
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
        elif self.use_deformable_dual:
            _dd_cfg_root = getattr(cfg, "model", cfg) if cfg is not None else None
            _dd_cfg = (
                getattr(_dd_cfg_root, "deformable_dual", None)
                if _dd_cfg_root is not None
                else None
            )

            def _get_dd(key: str, default):
                return getattr(_dd_cfg, key, default) if _dd_cfg else default

            _fusion_init = _get_dd("fusion_init_weights", [0.8, 0.1, 0.1])
            self.deformable_dual_fusion = DeformableDualFusion(
                in_channels=256,
                num_points=int(_get_dd("num_points", 4)),
                num_heads=int(_get_dd("num_heads", 4)),
                max_offset=float(_get_dd("max_offset", 4.0)),
                density_offset_rho=float(_get_dd("density_offset_rho", 0.5)),
                density_gamma_init=float(_get_dd("density_gamma_init", 0.5)),
                distance_lambda_init=float(_get_dd("distance_lambda_init", 1.0)),
                dropout=float(_get_dd("dropout", 0.1)),
                density_embed_dim=int(_get_dd("density_embed_dim", 32)),
                fusion_hidden_channels=int(_get_dd("fusion_hidden_channels", 128)),
                fusion_init_weights=tuple(float(w) for w in _fusion_init),
                fusion_spatial=bool(_get_dd("fusion_spatial", True)),
                residual_gate_init=float(_get_dd("residual_gate_init", 0.001)),
                debug=bool(_get_dd("debug", False)),
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
        elif self.use_sa_dgat:
            _sa_cfg = getattr(cfg, "model", cfg) if cfg is not None else None
            _sa_dgat_cfg = (
                getattr(_sa_cfg, "sa_dgat", None) if _sa_cfg is not None else None
            )

            def _get(key: str, default):
                return getattr(_sa_dgat_cfg, key, default) if _sa_dgat_cfg else default

            _local_dil = _get("local_dilations", [1, 2, 4])
            _global_dil = _get("global_dilations", [1, 3, 6])
            self.sa_dgat_fusion = SADGATFusion(
                in_channels=256,
                num_scale_prompts=int(_get("num_scale_prompts", 5)),
                deformable_k=int(_get("deformable_k", 8)),
                num_heads=int(_get("num_heads", 4)),
                lambda_init=float(_get("lambda_init", 1.0)),
                mu_init=float(_get("mu_init", 1.0)),
                local_dilations=tuple(int(d) for d in _local_dil),
                global_dilations=tuple(int(d) for d in _global_dil),
                num_gat_layers=int(_get("num_gat_layers", 2)),
                occ_hidden=int(_get("occ_hidden", 64)),
                use_depth_prior=bool(_get("use_depth_prior", False)),
                use_cross_scale=bool(_get("use_cross_scale", True)),
                dropout=float(_get("dropout", 0.1)),
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
        elif self.use_moecount_moe:
            _mm_cfg = moecount_moe_cfg
            _def = lambda k, d: getattr(_mm_cfg, k, d) if _mm_cfg is not None else d
            self.moecount_moe = HeterogeneousSparseMoE(
                channels=256,
                gate_type=str(_def("gate_type", "sparse_top2")),
                top_k=int(_def("top_k", 2)),
                temperature_init=float(_def("temperature_init", 1.0)),
                temperature_min=float(_def("temperature_min", 0.1)),
                temperature_decay=float(_def("temperature_decay", 0.99998)),
                warmup_fraction=float(_def("warmup_fraction", 0.2)),
                warmup_epochs=_def("warmup_epochs", None),
                lambda_importance=float(_def("lambda_importance", 0.01)),
                lambda_load=float(_def("lambda_load", 0.01)),
                shared_scale=float(_def("shared_scale", 0.5)),
                shared_num_blocks=int(_def("shared_num_blocks", 3)),
                shared_scale_learnable=bool(_def("shared_scale_learnable", True)),
                use_deformable_expert=bool(_def("use_deformable_expert", False)),
                use_input_residual=bool(_def("use_input_residual", True)),
                gate_use_density_hint=bool(_def("gate_use_density_hint", False)),
                gate_density_hidden=int(_def("gate_density_hidden", 8)),
                gate_use_density_bias=bool(_def("gate_use_density_bias", False)),
                gate_graph_k=int(_def("gate_graph_k", 4)),
                expert_use_density=bool(_def("expert_use_density", True)),
                expert_local_detail_use_residual=bool(
                    _def("expert_local_detail_use_residual", False)
                ),
                expert_global_density_use_residual=bool(
                    _def("expert_global_density_use_residual", False)
                ),
                expert_global_density_use_density=bool(
                    _def("expert_global_density_use_density", True)
                ),
                expert_local_detail_use_density_adaptive=bool(
                    _def("expert_local_detail_use_density_adaptive", True)
                ),
                expert_local_detail_dilations=tuple(
                    int(b) for b in _def("expert_local_detail_dilations", [1, 2, 3])
                ),
                expert_local_detail_groups=int(
                    _def("expert_local_detail_groups", 16)
                ),
                expert_local_detail_ffn_expansion=int(
                    _def("expert_local_detail_ffn_expansion", 2)
                ),
                expert_local_detail_use_density_modulation=bool(
                    _def("expert_local_detail_use_density_modulation", True)
                ),
                deformable_use_density_bias=bool(
                    _def("deformable_use_density_bias", False)
                ),
                # --- Expert replacement flags ---
                use_point_localization_expert=bool(
                    _def("use_point_localization_expert", False)
                ),
                use_occlusion_reasoning_expert=bool(
                    _def("use_occlusion_reasoning_expert", False)
                ),
                use_density_pattern_expert=bool(
                    _def("use_density_pattern_expert", False)
                ),
                # --- PointLocalizationExpert (e0) config ---
                expert_pl_use_point_aux=bool(
                    _def("expert_pl_use_point_aux", False)
                ),
                expert_pl_point_hidden=int(_def("expert_pl_point_hidden", 64)),
                expert_pl_point_loss_weight=float(_def("expert_pl_point_loss_weight", 1.0)),
                expert_pl_point_cls_weight=float(_def("expert_pl_point_cls_weight", 1.0)),
                expert_pl_point_reg_weight=float(_def("expert_pl_point_reg_weight", 0.0002)),
                expert_pl_point_cost_class=float(_def("expert_pl_point_cost_class", 1.0)),
                expert_pl_point_cost_point=float(_def("expert_pl_point_cost_point", 0.05)),
                expert_pl_point_eos_coef=float(_def("expert_pl_point_eos_coef", 0.5)),
                expert_pl_point_max_candidates=int(_def("expert_pl_point_max_candidates", 512)),
                # --- OcclusionReasoningExpert (e1) config ---
                expert_occ_use_aux=bool(_def("expert_occ_use_aux", False)),
                expert_occ_emb_hidden=int(_def("expert_occ_emb_hidden", 16)),
                expert_occ_consistency_weight=float(_def("expert_occ_consistency_weight", 1.0)),
                expert_occ_density_threshold=float(_def("expert_occ_density_threshold", 5.0)),
                expert_occ_head_hidden=int(_def("expert_occ_head_hidden", 128)),
                expert_occ_use_residual=bool(_def("expert_occ_use_residual", True)),
                # --- DensityPatternExpert (e2) config ---
                expert_dp_use_aux=bool(_def("expert_dp_use_aux", False)),
                expert_dp_ppm_bins=tuple(
                    int(b) for b in _def("expert_dp_ppm_bins", [1, 2, 3, 6])
                ),
                expert_dp_ppm_reduction=int(_def("expert_dp_ppm_reduction", 4)),
                expert_dp_pattern_num_bins=int(_def("expert_dp_pattern_num_bins", 8)),
                expert_dp_pattern_class_weight=float(_def("expert_dp_pattern_class_weight", 1.0)),
                expert_dp_use_residual=bool(_def("expert_dp_use_residual", True)),
            )

            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.sdd_moe = None
        elif self.use_scale_decoupled:
            _sdf_cfg = getattr(
                getattr(cfg, "model", cfg), "scale_decoupled_fusion", None
            ) if cfg is not None else None

            def _sc(key: str, default):
                return getattr(_sdf_cfg, key, default) if _sdf_cfg is not None else default

            self.scale_decoupled_fusion = ScaleDecoupledFusion(
                c2_channels=256,  # VGG body2: stride-4, 256ch
                c3_channels=512,  # VGG body3: stride-8, 512ch
                c4_channels=512,  # VGG body4: stride-16, 512ch
                unified_dim=int(_sc("unified_dim", 256)),
                cnn_dilations=tuple(int(d) for d in _sc("cnn_dilations", [1, 2, 3])),
                cnn_groups=int(_sc("cnn_groups", 16)),
                cnn_ffn_expansion=int(_sc("cnn_ffn_expansion", 2)),
                cnn_use_multi_spectral_se=bool(_sc("cnn_use_multi_spectral_se", True)),
                gcn_k=int(_sc("gcn_k", 4)),
                gcn_spatial_alpha=float(_sc("gcn_spatial_alpha", 1.0)),
                gcn_spatial_beta=float(_sc("gcn_spatial_beta", 1.0)),
                gcn_hidden_channels=int(_sc("gcn_hidden_channels", 512)),
                gcn_heads=int(_sc("gcn_heads", 4)),
                gcn_dropout=float(_sc("gcn_dropout", 0.1)),
                trans_num_blocks=int(_sc("trans_num_blocks", 2)),
                trans_num_heads=int(_sc("trans_num_heads", 4)),
                trans_embed_dim=int(_sc("trans_embed_dim", 128)),
                trans_mlp_ratio=float(_sc("trans_mlp_ratio", 4.0)),
                ca_num_heads=int(_sc("ca_num_heads", 4)),
                ca_dropout=float(_sc("ca_dropout", 0.1)),
                ca_ff_expansion=int(_sc("ca_ff_expansion", 2)),
                dm_density_hidden=int(_sc("dm_density_hidden", 64)),
                dm_reduction=int(_sc("dm_reduction", 4)),
            )

            # Nullify other fusion components
            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
            self.sdd_moe = None
            self.light_moe = None
            # Nullify neck (not used in scale_decoupled mode)
            self.pa = None  # type: ignore[assignment]
        else:
            if gcn_mode == "supernode":
                self.supernode_gcn: SuperNodeGCNProcessor | None = (
                    SuperNodeGCNProcessor(
                        in_channels=256,
                        num_supernodes=gcn_num_supernodes,
                        num_heads=gcn_supernode_heads,
                        dropout=gcn_dropout,
                    )
                )
                self.cross_stream_gcn = None
                self.density_gcn = None
                self.feature_gcn = None
                self.alpha = None
            elif gcn_mode == "cross_stream":
                self.cross_stream_gcn: CrossStreamGCNProcessor | None = (
                    CrossStreamGCNProcessor(
                        k=gcn_k,
                        adaptive=gcn_adaptive,
                        k_min=gcn_k_min,
                        k_max=gcn_k_max,
                        density_scale=gcn_density_scale,
                        sim_threshold=gcn_sim_threshold,
                        use_uncertainty=use_uncertainty,
                        uncertainty_scale=uncertainty_scale,
                        dropout=gcn_dropout,
                    )
                )
                self.supernode_gcn = None
                self.density_gcn = None
                self.feature_gcn = None
                self.alpha = None
            else:
                self.supernode_gcn = None
                self.cross_stream_gcn = None
                self.density_gcn: DensityGCNProcessor | None = DensityGCNProcessor(
                    k=gcn_k,
                    adaptive=gcn_adaptive,
                    k_min=gcn_k_min,
                    k_max=gcn_k_max,
                    density_scale=gcn_density_scale,
                    use_uncertainty=use_uncertainty,
                    uncertainty_scale=uncertainty_scale,
                    anisotropic=gcn_aniso,
                    conv_type=gcn_conv_type,
                    spatial_prior=gcn_spatial_prior,
                    spatial_alpha=gcn_spatial_alpha,
                    spatial_beta=gcn_spatial_beta,
                    depth_prior_cfg=(
                        getattr(getattr(cfg, "model", cfg), "depth_graph_prior", None)
                        if cfg is not None
                        else None
                    ),
                    dropout=gcn_dropout,
                )
                if self.feature_stream_type in {"transformer", "window_transformer"}:
                    _ft_cfg = feature_transformer_cfg
                    self.feature_gcn: FeatureGCNProcessor | FeatureTransformerProcessor | None = FeatureTransformerProcessor(
                        in_channels=256,
                        embed_dim=int(getattr(_ft_cfg, "embed_dim", 128))
                        if _ft_cfg is not None
                        else 128,
                        num_heads=int(getattr(_ft_cfg, "num_heads", 4))
                        if _ft_cfg is not None
                        else 4,
                        window_size=int(getattr(_ft_cfg, "window_size", 8))
                        if _ft_cfg is not None
                        else 8,
                        num_layers=int(getattr(_ft_cfg, "num_layers", 1))
                        if _ft_cfg is not None
                        else 1,
                        mlp_ratio=float(getattr(_ft_cfg, "mlp_ratio", 4.0))
                        if _ft_cfg is not None
                        else 4.0,
                        dropout=float(getattr(_ft_cfg, "dropout", 0.0))
                        if _ft_cfg is not None
                        else 0.0,
                        gate_init=float(getattr(_ft_cfg, "gate_init", 0.0))
                        if _ft_cfg is not None
                        else 0.0,
                        mode=str(getattr(_ft_cfg, "mode", "window"))
                        if _ft_cfg is not None
                        else "window",
                    )
                else:
                    self.feature_gcn = FeatureGCNProcessor(
                        k=gcn_k,
                        adaptive=gcn_adaptive,
                        k_min=gcn_k_min,
                        k_max=gcn_k_max,
                        sim_threshold=gcn_sim_threshold,
                        anisotropic=gcn_aniso,
                        conv_type=gcn_conv_type,
                        dropout=gcn_dropout,
                    )
                self.alpha: nn.Parameter | None = nn.Parameter(
                    torch.ones(3, dtype=torch.float32)
                )
            if use_gm and gcn_mode not in {"supernode", "cross_stream"}:
                if gm_spatial:
                    self.gm: SpatialGateMechanism | GateMechanism | None = (
                        SpatialGateMechanism(input_dim=gm_input_dim)
                    )
                else:
                    self.gm = GateMechanism(
                        input_dim=gm_input_dim, hidden_dim=gm_hidden_dim
                    )
            else:
                self.gm = None

            # Density-Adaptive Fusion: replaces alpha / gm when enabled
            if use_density_adaptive_fusion and gcn_mode == "fixed":
                _daf_cfg = density_adaptive_fusion_cfg
                _daf_embed = (
                    int(getattr(_daf_cfg, "density_embed_dim", 64))
                    if _daf_cfg is not None
                    else 64
                )
                _daf_spatial = (
                    bool(getattr(_daf_cfg, "spatial", True))
                    if _daf_cfg is not None
                    else True
                )
                self.density_adaptive_fusion: DensityAdaptiveFusion | None = (
                    DensityAdaptiveFusion(
                        in_channels=256,
                        density_embed_dim=_daf_embed,
                        spatial=_daf_spatial,
                    )
                )
                # When density-adaptive fusion is active, disable alpha and gm
                self.alpha = None
                self.gm = None
            else:
                self.density_adaptive_fusion = None

            self.graph_attn_moe = None
            self.mamba_moe = None

        # LightMoE post-GCN refinement (only for gcn_moe mode)
        if self.use_gcn_moe:
            _light_grid = (
                int(getattr(moe_cfg, "grid_stride", 4)) if moe_cfg is not None else 4
            )
            _light_density = (
                bool(getattr(moe_cfg, "use_density_hint", True))
                if moe_cfg is not None
                else True
            )
            _light_balance = (
                float(getattr(moe_cfg, "lambda_balance", 0.01))
                if moe_cfg is not None
                else 0.01
            )
            self.light_moe: LightMoE | None = LightMoE(
                input_dim=256,
                grid_stride=_light_grid,
                use_density_hint=_light_density,
                lambda_balance=_light_balance,
            )
        else:
            self.light_moe = None

        self.msaa: MsaaAdaptiveLayer | None = (
            MsaaAdaptiveLayer(in_channels=msaa_in_channels, reduction=msaa_reduction)
            if use_msaa and msaa_variant == "legacy"
            else None
        )

        # MSAALite: lightweight post-PA-FPN attention (Phase 1)
        self.msaa_lite: MSAALite | None = (
            MSAALite(in_channels=256) if use_msaa and msaa_variant == "lite" else None
        )

        # MSAAGate: attention-based GCN stream fusion (Phase 3)
        # Replaces GateMechanism when msaa_variant == "msaa_gate"
        self.msaa_gate: MSAAGate | None = (
            MSAAGate(in_channels=256, num_streams=3)
            if use_msaa and msaa_variant == "msaa_gate"
            else None
        )

        # Depth shared-backbone Mix fusion (optional, disabled by default)
        if use_depth:
            mix_init = (
                float(getattr(depth_cfg, "mix_init", 1.5))
                if depth_cfg is not None
                else 1.5
            )
            self.shared_depth_mix: _SharedBackboneDepthMix | None = (
                _SharedBackboneDepthMix(init=mix_init)
            )
            self.depth_backbone: _DepthEncoder | None = None
            self.depth_fusion_c3 = None
            self.depth_fusion_c4 = None
            self.depth_fusion_c5 = None
        else:
            self.shared_depth_mix = None
            self.depth_backbone = None
            self.depth_fusion_c3 = None
            self.depth_fusion_c4 = None
            self.depth_fusion_c5 = None

        # Depth geo-prior dual-stream fusion (optional, alternative to use_depth)
        if use_depth_geo:
            geo_num_heads = (
                int(getattr(depth_geo_cfg, "num_heads", 8))
                if depth_geo_cfg is not None
                else 8
            )
            geo_init_val = (
                float(getattr(depth_geo_cfg, "initial_value", 2.0))
                if depth_geo_cfg is not None
                else 2.0
            )
            geo_hr = (
                float(getattr(depth_geo_cfg, "heads_range", 4.0))
                if depth_geo_cfg is not None
                else 4.0
            )
            self.geo_attn_c3 = DepthGeoPriorAttention(
                256,
                num_heads=geo_num_heads,
                initial_value=geo_init_val,
                heads_range=geo_hr,
            )
            self.geo_attn_c4 = DepthGeoPriorAttention(
                512,
                num_heads=geo_num_heads,
                initial_value=geo_init_val,
                heads_range=geo_hr,
            )
            self.geo_attn_c5 = DepthGeoPriorAttention(
                512,
                num_heads=geo_num_heads,
                initial_value=geo_init_val,
                heads_range=geo_hr,
            )
        else:
            self.geo_attn_c3 = None
            self.geo_attn_c4 = None
            self.geo_attn_c5 = None

        # Post-neck DFormerv2-style geometry self-attention. This keeps the
        # backbone/neck unchanged and injects depth as a geometry prior once on
        # the shared 256-channel feature before density prediction and GCN.
        if use_depth_geo_post:
            geo_num_heads = (
                int(getattr(depth_geo_cfg, "num_heads", 8))
                if depth_geo_cfg is not None
                else 8
            )
            geo_init_val = (
                float(getattr(depth_geo_cfg, "initial_value", 2.0))
                if depth_geo_cfg is not None
                else 2.0
            )
            geo_hr = (
                float(getattr(depth_geo_cfg, "heads_range", 4.0))
                if depth_geo_cfg is not None
                else 4.0
            )
            self.geo_attn_post: DepthGeoPriorAttention | None = DepthGeoPriorAttention(
                256,
                num_heads=geo_num_heads,
                initial_value=geo_init_val,
                heads_range=geo_hr,
            )
        else:
            self.geo_attn_post = None

        # Dual-VGG RGBD fusion (optional, alternative to use_depth / use_depth_geo / use_depth_geo_post)
        if use_depth_dual_vgg:
            _dvgg_variant = (
                str(getattr(depth_dual_vgg_cfg, "variant", "vgg16_bn"))
                if depth_dual_vgg_cfg is not None
                else "vgg16_bn"
            )
            _dvgg_pretrained = (
                bool(getattr(depth_dual_vgg_cfg, "pretrained", True))
                if depth_dual_vgg_cfg is not None
                else True
            )
            _dvgg_frozen = (
                int(getattr(depth_dual_vgg_cfg, "frozen_stages", 0))
                if depth_dual_vgg_cfg is not None
                else 0
            )
            self.depth_vgg_backbone: DepthBackbone_VGG | None = DepthBackbone_VGG(
                name=_dvgg_variant,
                pretrained=_dvgg_pretrained,
                frozen_stages=_dvgg_frozen,
            )
            self.dvgg_fusion_c3: ConcatGateFusion | None = ConcatGateFusion(256)
            self.dvgg_fusion_c4: ConcatGateFusion | None = ConcatGateFusion(512)
            self.dvgg_fusion_c5: ConcatGateFusion | None = ConcatGateFusion(512)
        else:
            self.depth_vgg_backbone = None
            self.dvgg_fusion_c3 = None
            self.dvgg_fusion_c4 = None
            self.dvgg_fusion_c5 = None

        # Path 4: use_depth_attn — DepthResidualGating (lightweight residual gate)
        self.depth_attn_require_depth = False
        if use_depth_attn:
            _da_version = (
                str(getattr(depth_attn_cfg, "version", "v1"))
                if depth_attn_cfg is not None
                else "v1"
            ).lower()
            if _da_version not in {"v1", "v2"}:
                raise ValueError(
                    f"depth_attn.version must be 'v1' or 'v2', got {_da_version!r}"
                )
            _da_mid_ratio = (
                int(getattr(depth_attn_cfg, "mid_ratio", 4))
                if depth_attn_cfg is not None
                else 4
            )
            if _da_version == "v1":
                self.depth_attn_c3: DepthResidualGating | DepthResidualGatingV2 | None = DepthResidualGating(
                    256, mid_ratio=_da_mid_ratio
                )
                self.depth_attn_c4: DepthResidualGating | DepthResidualGatingV2 | None = DepthResidualGating(
                    512, mid_ratio=_da_mid_ratio
                )
                self.depth_attn_c5: DepthResidualGating | DepthResidualGatingV2 | None = DepthResidualGating(
                    512, mid_ratio=_da_mid_ratio
                )
            else:
                self.depth_attn_require_depth = (
                    bool(getattr(depth_attn_cfg, "require_depth", True))
                    if depth_attn_cfg is not None
                    else True
                )
                _da_gate_init = (
                    float(getattr(depth_attn_cfg, "gate_init", 0.0))
                    if depth_attn_cfg is not None
                    else 0.0
                )
                _da_use_tanh = (
                    bool(getattr(depth_attn_cfg, "use_tanh_gate", True))
                    if depth_attn_cfg is not None
                    else True
                )
                _da_spatial_gate = (
                    bool(getattr(depth_attn_cfg, "spatial_gate", True))
                    if depth_attn_cfg is not None
                    else True
                )
                _da_channel_gate = (
                    bool(getattr(depth_attn_cfg, "channel_gate", True))
                    if depth_attn_cfg is not None
                    else True
                )
                _da_normalize = (
                    bool(getattr(depth_attn_cfg, "normalize_depth", True))
                    if depth_attn_cfg is not None
                    else True
                )
                self.depth_attn_c3 = DepthResidualGatingV2(
                    256,
                    mid_ratio=_da_mid_ratio,
                    gate_init=_da_gate_init,
                    use_tanh_gate=_da_use_tanh,
                    spatial_gate=_da_spatial_gate,
                    channel_gate=_da_channel_gate,
                    normalize_depth=_da_normalize,
                )
                self.depth_attn_c4 = DepthResidualGatingV2(
                    512,
                    mid_ratio=_da_mid_ratio,
                    gate_init=_da_gate_init,
                    use_tanh_gate=_da_use_tanh,
                    spatial_gate=_da_spatial_gate,
                    channel_gate=_da_channel_gate,
                    normalize_depth=_da_normalize,
                )
                self.depth_attn_c5 = DepthResidualGatingV2(
                    512,
                    mid_ratio=_da_mid_ratio,
                    gate_init=_da_gate_init,
                    use_tanh_gate=_da_use_tanh,
                    spatial_gate=_da_spatial_gate,
                    channel_gate=_da_channel_gate,
                    normalize_depth=_da_normalize,
                )
        else:
            self.depth_attn_c3 = None
            self.depth_attn_c4 = None
            self.depth_attn_c5 = None

        # Path 5: post-neck RGB-depth cross-attention (optional)
        if use_depth_cross_attn:
            _dca_embed_dim = (
                int(getattr(depth_cross_attn_cfg, "embed_dim", 128))
                if depth_cross_attn_cfg is not None
                else 128
            )
            _dca_heads = (
                int(getattr(depth_cross_attn_cfg, "num_heads", 4))
                if depth_cross_attn_cfg is not None
                else 4
            )
            _dca_window = (
                int(getattr(depth_cross_attn_cfg, "window_size", 8))
                if depth_cross_attn_cfg is not None
                else 8
            )
            _dca_dropout = (
                float(getattr(depth_cross_attn_cfg, "dropout", 0.0))
                if depth_cross_attn_cfg is not None
                else 0.0
            )
            _dca_gate = (
                float(getattr(depth_cross_attn_cfg, "gate_init", 0.0))
                if depth_cross_attn_cfg is not None
                else 0.0
            )
            _dca_mid = (
                int(getattr(depth_cross_attn_cfg, "depth_mid_channels", 64))
                if depth_cross_attn_cfg is not None
                else 64
            )
            _dca_mode = (
                str(getattr(depth_cross_attn_cfg, "mode", "window"))
                if depth_cross_attn_cfg is not None
                else "window"
            )
            self.depth_cross_attn: DepthCrossAttentionFusion | None = (
                DepthCrossAttentionFusion(
                    in_channels=256,
                    embed_dim=_dca_embed_dim,
                    num_heads=_dca_heads,
                    window_size=_dca_window,
                    dropout=_dca_dropout,
                    gate_init=_dca_gate,
                    depth_mid_channels=_dca_mid,
                    mode=_dca_mode,
                )
            )
        else:
            self.depth_cross_attn = None

        # DINOv2 semantic injection (optional, disabled by default)
        # cfg may be full hydra config (has .model) or flat model-level config from tests
        _model_cfg = getattr(cfg, "model", cfg) if cfg is not None else None
        use_dino_inject = (
            bool(getattr(_model_cfg, "use_dino_inject", False))
            if _model_cfg is not None
            else False
        )
        if use_dino_inject:
            from crowdcount.models.backbone import DINOv2SemanticInjector

            dino_variant = (
                getattr(_model_cfg, "dino_inject_variant", "dinov2_b")
                if _model_cfg is not None
                else "dinov2_b"
            )
            self.dino_injector: DINOv2SemanticInjector | None = DINOv2SemanticInjector(
                dino_variant
            )
            self.dino_gate: nn.Parameter | None = nn.Parameter(torch.zeros(1))
        else:
            self.dino_injector = None
            self.dino_gate = None

        # SEMC post-GCN enhancer (optional, disabled by default)
        # Reads use_semc_enhancer and semc.* from the same _model_cfg resolved above.
        _use_semc = (
            bool(getattr(_model_cfg, "use_semc_enhancer", False))
            if _model_cfg is not None
            else False
        )
        if _use_semc:
            _semc_cfg = (
                getattr(_model_cfg, "semc", None) if _model_cfg is not None else None
            )
            _position = (
                str(getattr(_semc_cfg, "position", "post_gcn"))
                if _semc_cfg is not None
                else "post_gcn"
            )
            if _position != "post_gcn":
                raise ValueError(
                    f"Unsupported semc.position={_position}, expected 'post_gcn'"
                )
            _exp_f = (
                int(getattr(_semc_cfg, "expansion_factor", 4))
                if _semc_cfg is not None
                else 4
            )
            _ks_raw = (
                getattr(_semc_cfg, "kernel_sizes", [1, 3, 5])
                if _semc_cfg is not None
                else [1, 3, 5]
            )
            _ks = tuple(int(k) for k in _ks_raw)
            _resid = (
                bool(getattr(_semc_cfg, "use_residual", True))
                if _semc_cfg is not None
                else True
            )
            _dh = (
                bool(getattr(_semc_cfg, "use_density_hint", False))
                if _semc_cfg is not None
                else False
            )
            self.semc_enhancer: SEMCEnhancer | None = SEMCEnhancer(
                in_channels=256,
                expansion_factor=_exp_f,
                kernel_sizes=_ks,
                use_residual=_resid,
                use_density_hint=_dh,
            )
            self._semc_position = _position
            self._semc_use_density_hint: bool = _dh
        else:
            self.semc_enhancer = None
            self._semc_position = None
            self._semc_use_density_hint = False

        # Iterative Point Refinement (optional, disabled by default)
        if use_refine:
            _refine_hidden = (
                int(getattr(refine_cfg, "hidden_dim", 256))
                if refine_cfg is not None
                else 256
            )
            _refine_steps = (
                int(getattr(refine_cfg, "num_steps", 2))
                if refine_cfg is not None
                else 2
            )
            _refine_share = (
                bool(getattr(refine_cfg, "share_weights", True))
                if refine_cfg is not None
                else True
            )
            self.point_refine: PointRefineModule | None = PointRefineModule(
                feature_dim=256,
                hidden_dim=_refine_hidden,
                num_steps=_refine_steps,
                share_weights=_refine_share,
            )
        else:
            self.point_refine = None

        # Frequency-Decoupled Head routing (optional, disabled by default)
        self.freq_router: FreqDecoupledRouter | None = (
            FreqDecoupledRouter(kernel_size=freq_head_kernel) if use_freq_head else None
        )

        # Sub-Pixel Refinement (optional, disabled by default)
        if use_subpix_refine:
            _sp_top_k = (
                int(getattr(subpix_refine_cfg, "top_k", 512))
                if subpix_refine_cfg is not None
                else 512
            )
            _sp_hidden = (
                int(getattr(subpix_refine_cfg, "hidden_dim", 128))
                if subpix_refine_cfg is not None
                else 128
            )
            self.subpix_refine: SubPixelRefineModule | None = SubPixelRefineModule(
                hr_channels=256,
                lr_channels=256,
                hidden_dim=_sp_hidden,
                top_k=_sp_top_k,
            )
        else:
            self.subpix_refine = None

        # Foreground Suppression Branch (optional, disabled by default)
        self.fg_branch: ForegroundSuppressionBranch | None = (
            ForegroundSuppressionBranch(
                in_channels=256,
                base=fg_branch_base,
                scale=fg_branch_scale,
            )
            if use_fg_branch
            else None
        )

    def supports_moe(self) -> bool:
        return (
            (self.use_mamba_moe and self.mamba_moe is not None)
            or (self.use_mamba_vss_dual and self.mamba_vss_dual is not None)
            or (self.use_sdd_moe and self.sdd_moe is not None)
            or self.light_moe is not None
            or self.neck_moe is not None
            or (self.use_moecount_moe and self.moecount_moe is not None)
        )

    def get_moe_gating_parameters(self) -> list[nn.Parameter]:
        if self.neck_moe is not None:
            return list(self.neck_moe.router.parameters())
        if self.light_moe is not None:
            return list(self.light_moe.router.parameters())
        if self.sdd_moe is not None:
            return list(self.sdd_moe.router.parameters())
        if self.mamba_moe is not None:
            params: list[nn.Parameter] = []
            for momeb in self.mamba_moe.blocks:
                params.extend(momeb.block.spatial_moe.router.parameters())  # type: ignore[union-attr]
            return params
        if self.mamba_vss_dual is not None:
            return self.mamba_vss_dual.get_router_parameters()
        if self.moecount_moe is not None:
            return list(self.moecount_moe.gate.parameters())
        return []

    def update_moe_temperature(self, decay_rate: float = 0.9999) -> None:
        if self.sdd_moe is not None:
            self.sdd_moe.update_temperature(decay_rate=decay_rate)
        if self.moecount_moe is not None:
            self.moecount_moe.update_temperature(decay_rate=decay_rate)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        """Propagate epoch to MoE gate for temperature warmup scheduling."""
        if self.moecount_moe is not None:
            self.moecount_moe.set_epoch(epoch, total_epochs)

    @staticmethod
    def _density_attention_stats(
        attention_scale: torch.Tensor, density: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        scale = attention_scale.detach().float()
        if scale.shape[1] != 1:
            scale = scale.mean(dim=1, keepdim=True)

        density_ref = density.detach().float()
        if density_ref.shape[-2:] != scale.shape[-2:]:
            density_ref = F.interpolate(
                density_ref, size=scale.shape[-2:], mode="bilinear", align_corners=False
            )

        flat = scale.reshape(-1)
        overall_mean = scale.mean()
        density_mean = density_ref.mean(dim=(-2, -1), keepdim=True)
        high_mask = density_ref >= density_mean
        low_mask = ~high_mask
        high_mean = scale[high_mask].mean() if high_mask.any().item() else overall_mean
        low_mean = scale[low_mask].mean() if low_mask.any().item() else overall_mean

        return {
            "min": scale.amin(),
            "max": scale.amax(),
            "mean": overall_mean,
            "std": scale.std(unbiased=False),
            "p10": torch.quantile(flat, 0.1),
            "p90": torch.quantile(flat, 0.9),
            "high_density_mean": high_mean,
            "low_density_mean": low_mean,
        }

    @staticmethod
    def _clip_prompt_density_stats(
        prompt_info: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        foreground_prob = prompt_info["foreground_prob"].detach().float()
        positive_weight = prompt_info["positive_weight"].detach().float()
        negative_weight = prompt_info["negative_weight"].detach().float()
        strength = prompt_info["strength"].detach().float()
        return {
            "foreground_mean": foreground_prob.mean(),
            "foreground_std": foreground_prob.std(unbiased=False),
            "positive_weight_mean": positive_weight.mean(),
            "negative_weight_mean": negative_weight.mean(),
            "strength": strength.reshape(()),
        }

    def forward(
        self,
        samples: torch.Tensor,
        depth_map: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        gt_density: torch.Tensor | None = None,
    ) -> dict:
        features = self.backbone(samples)
        clip_prompt_info: dict[str, torch.Tensor] | None = None
        neck_moe_aux_losses: dict[str, torch.Tensor] | None = None
        neck_moe_aux_total: torch.Tensor | None = None
        neck_moe_weights: torch.Tensor | None = None

        # Convert dict to list format for compatibility with MSAA and PA-FPN
        features_list = [features[0], features[1], features[2], features[3]]

        # Shared-backbone depth fusion: run depth through the same backbone and
        # Mix c3/c4/c5 before downstream neck modules.
        if self.use_depth and depth_map is not None:
            assert self.shared_depth_mix is not None
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(1)
            depth_map = depth_map.to(device=samples.device, dtype=samples.dtype)
            if depth_map.shape[-2:] != samples.shape[-2:]:
                depth_map = F.interpolate(
                    depth_map,
                    size=samples.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if depth_map.shape[1] == 1:
                depth_samples = depth_map.repeat(1, samples.shape[1], 1, 1)
            elif depth_map.shape[1] == samples.shape[1]:
                depth_samples = depth_map
            else:
                raise ValueError(
                    "depth_map must have either one channel or match sample channels; "
                    f"got depth_map.shape={tuple(depth_map.shape)} and "
                    f"samples.shape={tuple(samples.shape)}"
                )
            depth_features = self.backbone(depth_samples)
            depth_features_list = [
                depth_features[0],
                depth_features[1],
                depth_features[2],
                depth_features[3],
            ]
            c3, c4, c5 = self.shared_depth_mix(
                (features_list[1], features_list[2], features_list[3]),
                (depth_features_list[1], depth_features_list[2], depth_features_list[3]),
            )
            features_list[1], features_list[2], features_list[3] = c3, c4, c5

        if self.msaa is not None:
            features_list = self.msaa(features_list)

        # Use stable list indices across VGG and DINO backbones:
        # c3: 256ch, c4: 512ch, c5: 512ch
        c3, c4, c5 = features_list[1], features_list[2], features_list[3]
        c3_hr = c3  # Cache high-res features for optional sub-pixel refinement

        if self.use_depth_geo and depth_map is not None:
            c3 = self.geo_attn_c3(c3, depth_map)  # type: ignore[misc]
            c4 = self.geo_attn_c4(c4, depth_map)  # type: ignore[misc]
            c5 = self.geo_attn_c5(c5, depth_map)  # type: ignore[misc]

        # Dual-VGG RGBD fusion: depth through a separate VGG, then concat-gate
        if self.use_depth_dual_vgg and depth_map is not None:
            assert self.depth_vgg_backbone is not None
            if depth_map.shape[-2:] != samples.shape[-2:]:
                depth_map = F.interpolate(
                    depth_map,
                    size=samples.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            depth_feats = self.depth_vgg_backbone(depth_map)
            # depth_feats: [d_c1, d_c3, d_c4, d_c5] matching VGG body1..body4
            d_c3, d_c4, d_c5 = depth_feats[1], depth_feats[2], depth_feats[3]
            c3 = self.dvgg_fusion_c3(c3, d_c3)  # type: ignore[misc]
            c4 = self.dvgg_fusion_c4(c4, d_c4)  # type: ignore[misc]
            c5 = self.dvgg_fusion_c5(c5, d_c5)  # type: ignore[misc]

        # Path 4: DepthResidualGating — lightweight residual gate per scale
        if self.use_depth_attn:
            if depth_map is None:
                if self.depth_attn_require_depth:
                    raise ValueError(
                        "use_depth_attn=True with depth_attn.version='v2' requires depth_map. "
                        "Set model.depth_attn.require_depth=false to keep RGB-only fallback."
                    )
            else:
                c3 = self.depth_attn_c3(c3, depth_map)  # type: ignore[misc]
                c4 = self.depth_attn_c4(c4, depth_map)  # type: ignore[misc]
                c5 = self.depth_attn_c5(c5, depth_map)  # type: ignore[misc]

        # --- Scale-Decoupled Fusion path: replaces Neck + DGCN entirely ---
        fpn_intermediates = None
        if self.use_scale_decoupled:
            assert self.scale_decoupled_fusion is not None
            assert self.density_pred is not None
            # VGG backbone features: [body1(s2), body2(s4,256), body3(s8,512), body4(s16,512)]
            # Scale-decoupled mapping: s4→CNN, s8→GCN, s16→Transformer
            feature_fl, _ = self.scale_decoupled_fusion(
                features_list[1],  # body2: stride-4, 256ch
                features_list[2],  # body3: stride-8, 512ch
                features_list[3],  # body4: stride-16, 512ch
            )
            # Downsample to stride-8 for downstream heads (anchor points are fixed at s8)
            target_size = features_list[2].shape[-2:]
            if feature_fl.shape[-2:] != target_size:
                feature_fl = F.adaptive_avg_pool2d(feature_fl, target_size)
            features_pa = feature_fl
            features_pa = F.dropout2d(
                features_pa, p=self.neck_dropout, training=self.training
            )
            density = self.density_pred(features_pa)
        # --- MSCADecoder path: replaces PA-FPN + Density_pred, GCN runs downstream ---
        elif self.use_msca_decoder:
            assert self.msca_decoder is not None
            feature_fl, density = self.msca_decoder([c3, c4, c5])
            features_pa = feature_fl  # alias for downstream consumers
        else:
            assert self.pa is not None
            assert self.density_pred is not None
            need_neck_intermediates = self.use_sa_dgat or (
                self.neck_moe is not None and self.neck_moe_use_pyramid_context
            )
            if need_neck_intermediates:
                _pa_result = self.pa([c3, c4, c5], return_intermediates=True)
                features_pa, fpn_intermediates = _pa_result
            else:
                features_pa = self.pa([c3, c4, c5])  # [batch_size, 256, 16, 16]

            if self.neck_moe is not None and self.neck_moe_position == "pre_acdr":
                features_pa, neck_moe_aux_losses, neck_moe_weights = self.neck_moe(
                    features_pa,
                    pyramid=fpn_intermediates,
                    training=self.training,
                )
                neck_moe_aux_total = neck_moe_aux_losses.get("total_aux")

            if self.neck_acdr is not None:
                features_pa = self.neck_acdr(features_pa)

            if self.neck_moe is not None and self.neck_moe_position == "post_acdr":
                features_pa, neck_moe_aux_losses, neck_moe_weights = self.neck_moe(
                    features_pa,
                    pyramid=fpn_intermediates,
                    training=self.training,
                )
                neck_moe_aux_total = neck_moe_aux_losses.get("total_aux")

            # Phase 1: MSAALite post-PA-FPN attention refinement
            if self.msaa_lite is not None:
                features_pa = self.msaa_lite(features_pa)

            # DINOv2 semantic injection: bounded gate (tanh) starts at 0
            if self.dino_injector is not None and self.dino_gate is not None:
                dino_feat = self.dino_injector(
                    samples, target_size=features_pa.shape[-2:]
                )
                features_pa = features_pa + self.dino_gate.tanh() * dino_feat

            if self.depth_cross_attn is not None and depth_map is not None:
                features_pa = self.depth_cross_attn(features_pa, depth_map)

            if self.geo_attn_post is not None and depth_map is not None:
                features_pa = self.geo_attn_post(features_pa, depth_map)

            features_pa = F.dropout2d(
                features_pa, p=self.neck_dropout, training=self.training
            )
            density_features = features_pa
            if self.clip_prompt_density is not None:
                density_features, clip_prompt_info = self.clip_prompt_density(
                    features_pa
                )
                if self.clip_prompt_density_apply_to == "shared":
                    features_pa = density_features

            density = self.density_pred(density_features)

        if self.use_msca_decoder:
            features_pa = F.dropout2d(
                features_pa, p=self.neck_dropout, training=self.training
            )

        depth_aux_out = (
            self.depth_aux_head(features_pa) if self.depth_aux_head is not None else None
        )

        batch_size = features_list[0].shape[0]

        # Uncertainty map from density prediction (detach to avoid gradient leak)
        uncertainty = (
            compute_uncertainty(density.detach()) if self.use_uncertainty else None
        )

        # Multi-scale density prediction (if enabled)
        output_dict = {
            "pred_logits": None,
            "pred_points": None,
            "density_out": density,
            "uncertainty_map": uncertainty,
            "img_size": (samples.shape[-2], samples.shape[-1]),
            "moe_aux_losses": None,
            "moe_aux_total": None,
            "moe_weights": None,
            "neck_moe_aux_losses": neck_moe_aux_losses,
            "neck_moe_aux_total": neck_moe_aux_total,
            "neck_moe_weights": neck_moe_weights,
        }
        if depth_aux_out is not None:
            output_dict["depth_aux_out"] = depth_aux_out
        if self.neck_moe is not None:
            output_dict["moe_aux_losses"] = neck_moe_aux_losses
            output_dict["moe_aux_total"] = neck_moe_aux_total
            output_dict["moe_weights"] = neck_moe_weights

        if clip_prompt_info is not None:
            output_dict["clip_prompt_foreground_logits"] = clip_prompt_info[
                "foreground_logits"
            ]
            if self.clip_prompt_density_debug:
                output_dict["clip_prompt_density_stats"] = (
                    self._clip_prompt_density_stats(clip_prompt_info)
                )

        if self.use_multi_scale_density:
            if self.use_cross_scale_refine:
                ms_densities = self.cross_scale_refine(c3, c4, c5)
            else:
                ms_densities = {
                    "density_block3": self.density_pred_block3(c3),
                    "density_block4": self.density_pred_block4(c4),
                    "density_block5": self.density_pred_block5(c5),
                }
            output_dict.update(ms_densities)

            # Fuse multi-scale densities into GCN input (optional)
            if self.use_fuse_to_gcn:
                density = self.density_fuse(
                    density,
                    ms_densities["density_block3"],
                    ms_densities["density_block4"],
                    ms_densities["density_block5"],
                )
                output_dict["density_fused"] = density

        # MSCADecoder produced features_pa + density; GCN still runs below
        # Pre-GCN density attention: lightweight spatial gating before graph construction
        if self.density_attention_pre_gcn is not None:
            pre_mask = self.density_attention_pre_gcn(density.detach()).to(
                features_pa.dtype
            )
            features_pa = features_pa * pre_mask

        if self.use_scale_decoupled:
            pass  # fusion already done by ScaleDecoupledFusion above
        elif self.use_graph_moe:
            assert self.graph_moe is not None
            feature_fl, graph_aux_losses, graph_weights = self.graph_moe(
                features_pa,
                density,
                uncertainty=uncertainty,
                training=self.training,
            )
            output_dict["moe_aux_losses"] = graph_aux_losses
            output_dict["moe_aux_total"] = graph_aux_losses.get("total_aux")
            output_dict["moe_weights"] = graph_weights
        elif self.use_graph_attn_moe:
            assert self.graph_attn_moe is not None
            feature_fl, gam_aux_losses, gam_weights = self.graph_attn_moe(
                features_pa, density, training=self.training
            )
            output_dict["moe_aux_losses"] = gam_aux_losses
            output_dict["moe_aux_total"] = gam_aux_losses.get("total_aux")
            output_dict["moe_weights"] = gam_weights
        elif self.use_mamba_moe:
            assert self.mamba_moe is not None
            feature_fl, mamba_aux_losses, mamba_weights = self.mamba_moe(
                features_pa, density.detach(), training=self.training
            )
            output_dict["moe_aux_losses"] = mamba_aux_losses
            output_dict["moe_aux_total"] = mamba_aux_losses.get("total_aux")
            output_dict["moe_weights"] = mamba_weights
        elif self.use_mamba_vss_dual:
            assert self.mamba_vss_dual is not None
            feature_fl, mvd_aux_losses, mvd_weights = self.mamba_vss_dual(
                features_pa, density.detach(), training=self.training
            )
            output_dict["moe_aux_losses"] = mvd_aux_losses
            output_dict["moe_aux_total"] = mvd_aux_losses.get("total_aux")
            output_dict["moe_weights"] = mvd_weights
            output_dict["mamba_vss_fusion_weights"] = (
                self.mamba_vss_dual.last_fusion_weights
            )
        elif self.use_sdd_moe:
            assert self.sdd_moe is not None
            feature_fl, sdd_aux_losses, sdd_weights = self.sdd_moe(
                features_pa,
                density_hint=density.detach(),
                targets=targets if self.training else None,
                gt_density=gt_density if self.training else None,
                image_size=(samples.shape[-2], samples.shape[-1]),
                training=self.training,
            )
            output_dict["moe_aux_losses"] = sdd_aux_losses
            output_dict["moe_aux_total"] = sdd_aux_losses.get("total_aux")
            output_dict["moe_weights"] = sdd_weights
        elif self.use_deformable_dual:
            assert self.deformable_dual_fusion is not None
            feature_fl, deformable_dual_aux = self.deformable_dual_fusion(
                features_pa,
                density,
            )
            output_dict["deformable_dual_aux"] = deformable_dual_aux
        elif self.use_sa_dgat:
            assert self.sa_dgat_fusion is not None
            feature_fl, sa_dgat_aux = self.sa_dgat_fusion(
                features_pa,
                density,
                depth_map=depth_map,
                fpn_intermediates=fpn_intermediates,
            )
            output_dict["sa_dgat_aux"] = sa_dgat_aux
        elif self.use_moecount_moe:
            assert self.moecount_moe is not None
            feature_fl, moe_aux_losses, route = self.moecount_moe(
                features_pa,
                density=density,
                targets=targets if self.training else None,
                gt_density=gt_density if self.training else None,
            )
            output_dict["moe_aux_losses"] = moe_aux_losses
            output_dict["moe_aux_total"] = moe_aux_losses.get("total_aux")
            output_dict["moe_weights"] = route.get("weights")
            output_dict["moe_top1"] = route.get("top1")
            output_dict["moe_load_fraction"] = route.get("load_fraction")
            output_dict["moe_importance"] = route.get("importance")
            output_dict["moe_entropy"] = route.get("entropy")
            output_dict["moe_temperature"] = route.get("temperature")
            output_dict["moe_warmup_active"] = route.get("warmup_active")
            output_dict["expert_similarity"] = route.get("expert_similarity")
        else:
            if self._gcn_mode == "supernode":
                assert self.supernode_gcn is not None
                feature_fl = self.supernode_gcn(features_pa, density)
            elif self._gcn_mode == "cross_stream":
                assert self.cross_stream_gcn is not None
                feature_fl = self.cross_stream_gcn(
                    density, features_pa, uncertainty=uncertainty
                )
            else:
                assert self.density_gcn is not None
                assert self.feature_gcn is not None
                density_gcn_feature = self.density_gcn(
                    density,
                    features_pa,
                    uncertainty=uncertainty,
                    depth_map=depth_map,
                )
                feature_gcn_feature = self.feature_gcn(features_pa)
                if self.msaa_gate is not None:
                    # Phase 3: MSAAGate multi-scale attention fusion
                    feature_fl = self.msaa_gate(
                        features_pa, density_gcn_feature, feature_gcn_feature
                    )
                elif self.density_adaptive_fusion is not None:
                    # Density-Adaptive Fusion: density-conditioned per-pixel weights
                    feature_fl = self.density_adaptive_fusion(
                        features_pa,
                        density_gcn_feature,
                        feature_gcn_feature,
                        density.detach(),
                    )
                elif self.gm is not None:
                    gate_weight = self.gm(features_pa)
                    if gate_weight.dim() == 4:
                        # SpatialGateMechanism: [B, 3, H, W]
                        feature_fl = (
                            features_pa * gate_weight[:, 0:1]
                            + density_gcn_feature * gate_weight[:, 1:2]
                            + feature_gcn_feature * gate_weight[:, 2:3]
                        )
                    else:
                        # Legacy GateMechanism: [B, 3]
                        w_1 = gate_weight[:, 0].view(-1, 1, 1, 1)
                        w_2 = gate_weight[:, 1].view(-1, 1, 1, 1)
                        w_3 = gate_weight[:, 2].view(-1, 1, 1, 1)
                        feature_fl = (
                            features_pa * w_1
                            + density_gcn_feature * w_2
                            + feature_gcn_feature * w_3
                        )
                else:
                    assert self.alpha is not None
                    w = F.softmax(self.alpha, dim=0)
                    feature_fl = (
                        w[0] * features_pa
                        + w[1] * density_gcn_feature
                        + w[2] * feature_gcn_feature
                    )

        # LightMoE post-GCN conditional refinement (gcn_moe mode)
        if self.light_moe is not None and not self.use_moecount_moe:
            feature_fl, light_aux, light_weights = self.light_moe(
                feature_fl, density_hint=density, training=self.training
            )
            output_dict["moe_aux_losses"] = light_aux
            output_dict["moe_aux_total"] = light_aux.get("total_aux")
            output_dict["moe_weights"] = light_weights

        # Post-GCN density attention: spatial+channel modulation of fused features
        attention_scale = None
        if self.density_attention is not None:
            if isinstance(
                self.density_attention,
                (
                    EnhancedDensityAttention,
                    GatedDensityAttention,
                    ResidualDensityAttention,
                ),
            ):
                feature_fl = self.density_attention(density.detach(), feature_fl)
                attention_scale = self.density_attention.last_attention_scale
            else:
                attention_mask = self.density_attention(density.detach()).to(
                    feature_fl.dtype
                )
                feature_fl = feature_fl * attention_mask
                attention_scale = attention_mask

        if self.density_attention_debug and attention_scale is not None:
            output_dict["density_attention_stats"] = self._density_attention_stats(
                attention_scale, density
            )

        # SEMC post-GCN enhancement (optional, disabled by default)
        if self.semc_enhancer is not None and not self.use_moecount_moe:
            feature_fl = self.semc_enhancer(
                feature_fl,
                density if self._semc_use_density_hint else None,
            )

        # Foreground suppression: residual-gated pixel-level FG prior
        if self.fg_branch is not None:
            feature_fl, fg_logits, fg_prob = self.fg_branch(feature_fl)
            output_dict["fg_logits"] = fg_logits
            output_dict["fg_prob"] = fg_prob

        if self.use_decoupled_head:
            cls_feat, reg_feat = self.pred_trunk(feature_fl)
            if self.freq_router is not None:
                _f_low, reg_feat, _ = self.freq_router(reg_feat)
            regression = self.regression(reg_feat) * 100
            classification = self.classification(cls_feat)
        else:
            shared_feat = self.pred_trunk(feature_fl)
            # Frequency-decoupled head routing (optional)
            if self.freq_router is not None:
                _f_low, f_high, f_full = self.freq_router(shared_feat)
                regression = self.regression(f_high) * 100
                classification = self.classification(f_full)
            else:
                regression = self.regression(shared_feat) * 100
                classification = self.classification(shared_feat)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification

        # Iterative point refinement (optional)
        if self.point_refine is not None:
            img_h, img_w = samples.shape[-2], samples.shape[-1]
            refined_coord, refine_intermediates = self.point_refine(
                feature_fl,
                output_coord,
                img_h,
                img_w,
            )
            output_dict["pred_points"] = refined_coord
            output_dict["refine_intermediates"] = refine_intermediates
        else:
            output_dict["pred_points"] = output_coord
            output_dict["refine_intermediates"] = None

        output_dict["pred_logits"] = output_class

        # Sub-pixel refinement for dense-region points (optional)
        if self.subpix_refine is not None:
            img_h, img_w = samples.shape[-2], samples.shape[-1]
            fg_scores = F.softmax(output_class, dim=-1)[:, :, 1]
            output_dict["pred_points"] = self.subpix_refine(
                hr_feat=c3_hr,
                lr_feat=features_pa,
                pred_points=output_dict["pred_points"],
                pred_scores=fg_scores,
                img_h=img_h,
                img_w=img_w,
            )

        if self.point_density_refiner is not None:
            density_base = output_dict["density_out"]
            fg_scores = F.softmax(output_dict["pred_logits"], dim=-1)[:, :, 1]
            point_heatmap = point_predictions_to_density_map(
                output_dict["pred_points"],
                fg_scores,
                density_size=density_base.shape[-2:],
                image_size=(samples.shape[-2], samples.shape[-1]),
                gaussian_sigma=self.point_density_feedback_gaussian_sigma,
                score_threshold=self.point_density_feedback_score_threshold,
                detach_points=self.point_density_feedback_detach_points,
                detach_scores=self.point_density_feedback_detach_scores,
            ).to(dtype=density_base.dtype)
            output_dict["density_base"] = density_base
            output_dict["point_feedback_heatmap"] = point_heatmap
            output_dict["density_out"] = self.point_density_refiner(
                features_pa,
                density_base,
                point_heatmap,
            )
            if self.point_density_feedback_debug:
                density_delta = self.point_density_refiner.last_delta
                output_dict["point_feedback_stats"] = {
                    "heatmap_sum": point_heatmap.sum(dim=(-2, -1)).mean().detach(),
                    "heatmap_max": point_heatmap.amax(dim=(-2, -1)).mean().detach(),
                    "delta_abs_mean": density_delta.abs().mean().detach()
                    if density_delta is not None
                    else density_base.new_tensor(0.0),
                    "strength": self.point_density_refiner.last_strength
                    if self.point_density_refiner.last_strength is not None
                    else density_base.new_tensor(0.0),
                }

        return output_dict

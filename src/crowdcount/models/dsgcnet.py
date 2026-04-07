"""DSGCNet main model definition."""

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.gcn import (
    CrossStreamGCNProcessor,
    DensityGCNProcessor,
    FeatureGCNProcessor,
    SuperNodeGCNProcessor,
    compute_uncertainty,
)
from crowdcount.models.head import (
    ClassificationModel,
    DensityAttentionMask,
    Density_pred,
    MultiScaleDensityAttention,
    ForegroundSuppressionBranch,
    FreqDecoupledRouter,
    PointRefineModule,
    RegressionModel,
    SharedPredictionTrunk,
    SubPixelRefineModule,
    DensityPred_Block3,
    DensityPred_Block4,
    DensityPred_Block5,
)
from crowdcount.models.neck import Decoder_SPD_PAFPN
from crowdcount.models.semc_blocks import SEMCEnhancer
from crowdcount.plugins.gm import GateMechanism, SpatialGateMechanism
from crowdcount.plugins.isfm.depth_fusion import DepthFusionModule
from crowdcount.plugins.geo_prior import DepthGeoPriorAttention
from crowdcount.plugins.mamba_moe import MambaMoEFusion
from crowdcount.plugins.moe import ESCA, MoE, LightMoE
from crowdcount.plugins.graph_moe import GraphAwareMoE
from crowdcount.plugins.msaa import MsaaAdaptiveLayer


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
        moe_cfg: DictConfig | None = None,
        graph_attn_moe_cfg: DictConfig | None = None,
        mamba_moe_cfg: DictConfig | None = None,
        use_depth: bool = False,
        depth_cfg: DictConfig | None = None,
        use_depth_geo: bool = False,
        depth_geo_cfg: DictConfig | None = None,
        gcn_adaptive: bool = False,
        gcn_k: int = 4,
        gcn_k_min: int = 2,
        gcn_k_max: int = 8,
        gcn_density_scale: float = 4.0,
        gcn_sim_threshold: float = 0.5,
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
        use_subpix_refine: bool = False,
        subpix_refine_cfg: DictConfig | None = None,
        use_uncertainty: bool = False,
        uncertainty_scale: float = 6.0,
        gcn_aniso: bool = False,
        use_fg_branch: bool = False,
        fg_branch_base: float = 0.5,
        fg_branch_scale: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg
        self.fusion_mode = fusion_mode
        self.use_moe = fusion_mode == "esca_moe"
        self.use_gcn_moe = fusion_mode == "gcn_moe"
        self.use_graph_attn_moe = fusion_mode == "graph_attn_moe"
        self.use_mamba_moe = fusion_mode == "mamba_moe"
        self.use_depth = use_depth
        self.use_depth_geo = use_depth_geo
        self.use_freq_head = use_freq_head
        self.use_density_attention = use_density_attention
        self.use_subpix_refine = use_subpix_refine
        self._gcn_mode = gcn_mode
        self.use_uncertainty = use_uncertainty

        if self.fusion_mode not in {
            "gcn",
            "esca_moe",
            "gcn_moe",
            "graph_attn_moe",
            "mamba_moe",
        }:
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode}, expected 'gcn', 'esca_moe', 'gcn_moe', 'graph_attn_moe', or 'mamba_moe'"
            )

        density_cfg = (
            getattr(cfg, "density_multi_scale", None) if cfg is not None else None
        )
        self.use_multi_scale_density = bool(
            getattr(density_cfg, "enabled", False) if density_cfg is not None else False
        )
        num_anchor_points = row * line

        self.pred_trunk = SharedPredictionTrunk(in_channels=256, feature_size=256)
        self.regression = RegressionModel(
            num_features_in=256, num_anchor_points=num_anchor_points
        )
        self.classification = ClassificationModel(
            num_features_in=256,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points,
        )

        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        if use_msaa:
            self.pa = Decoder_SPD_PAFPN(1280, 1280, 1280, use_dcn=use_dcn)
        else:
            self.pa = Decoder_SPD_PAFPN(256, 512, 512, use_dcn=use_dcn)
        self.density_pred = Density_pred()
        # Density attention: use multi-scale variant when both density_attention
        # and multi-scale density are enabled; otherwise fall back to single-scale.
        self.density_attention: MultiScaleDensityAttention | DensityAttentionMask | None
        if use_density_attention and self.use_multi_scale_density:
            self.density_attention = MultiScaleDensityAttention()
        elif use_density_attention:
            self.density_attention = DensityAttentionMask(mode=density_attention_mode)
        else:
            self.density_attention = None

        # Multi-scale density prediction (optional)
        if self.use_multi_scale_density:
            self.density_pred_block3 = DensityPred_Block3()
            self.density_pred_block4 = DensityPred_Block4()
            self.density_pred_block5 = DensityPred_Block5()

        if self.use_moe:
            top_k = int(getattr(moe_cfg, "top_k", 2)) if moe_cfg is not None else 2
            temperature_init = (
                float(getattr(moe_cfg, "temperature_init", 1.0))
                if moe_cfg is not None
                else 1.0
            )
            temperature_min = (
                float(getattr(moe_cfg, "temperature_min", 0.4))
                if moe_cfg is not None
                else 0.4
            )
            lambda_balance = (
                float(getattr(moe_cfg, "lambda_balance", 0.01))
                if moe_cfg is not None
                else 0.01
            )
            use_density_hint = (
                bool(getattr(moe_cfg, "use_density_hint", True))
                if moe_cfg is not None
                else True
            )
            grid_stride = (
                int(getattr(moe_cfg, "grid_stride", 4)) if moe_cfg is not None else 4
            )

            self.esca: ESCA | None = ESCA(256)
            self.moe: MoE | None = MoE(
                input_dim=256,
                top_k=top_k,
                temperature_init=temperature_init,
                temperature_min=temperature_min,
                lambda_balance=lambda_balance,
                use_density_hint=use_density_hint,
                grid_stride=grid_stride,
            )
            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.mamba_moe = None
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
            self.esca = None
            self.moe = None
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
                drop_path=float(getattr(_mmm, "drop_path", 0.1)) if _mmm else 0.1,
                lambda_balance=float(getattr(_mmm, "lambda_balance", 0.01))
                if _mmm
                else 0.01,
                use_density_hint=bool(getattr(_mmm, "use_density_hint", False))
                if _mmm
                else False,
                d_spectral=int(getattr(_mmm, "d_spectral", 256)) if _mmm else 256,
            )
            self.esca = None
            self.moe = None
            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
            self.graph_attn_moe = None
            self.supernode_gcn = None
            self.cross_stream_gcn = None
        else:
            if gcn_mode == "supernode":
                self.supernode_gcn: SuperNodeGCNProcessor | None = (
                    SuperNodeGCNProcessor(
                        in_channels=256,
                        num_supernodes=gcn_num_supernodes,
                        num_heads=gcn_supernode_heads,
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
                )
                self.feature_gcn: FeatureGCNProcessor | None = FeatureGCNProcessor(
                    k=gcn_k,
                    adaptive=gcn_adaptive,
                    k_min=gcn_k_min,
                    k_max=gcn_k_max,
                    sim_threshold=gcn_sim_threshold,
                    anisotropic=gcn_aniso,
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
            self.esca = None
            self.moe = None
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
            if use_msaa
            else None
        )

        # Depth dual-stream fusion (optional, disabled by default)
        if use_depth:
            embed_dim = (
                int(getattr(depth_cfg, "embed_dim", 128))
                if depth_cfg is not None
                else 128
            )
            num_isf_layers = (
                int(getattr(depth_cfg, "num_isf_layers", 1))
                if depth_cfg is not None
                else 1
            )
            self.depth_backbone: _DepthEncoder | None = _DepthEncoder()
            self.depth_fusion_c3: DepthFusionModule | None = DepthFusionModule(
                in_channels=256, embed_dim=embed_dim, num_isf_layers=num_isf_layers
            )
            self.depth_fusion_c4: DepthFusionModule | None = DepthFusionModule(
                in_channels=512, embed_dim=embed_dim, num_isf_layers=num_isf_layers
            )
            self.depth_fusion_c5: DepthFusionModule | None = DepthFusionModule(
                in_channels=512, embed_dim=embed_dim, num_isf_layers=num_isf_layers
            )
        else:
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
            if self.use_moe:
                raise ValueError(
                    "SEMCEnhancer is currently supported only for fusion_mode='gcn'"
                )
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
            (self.use_moe and self.moe is not None)
            or (self.use_mamba_moe and self.mamba_moe is not None)
            or self.light_moe is not None
        )

    def get_moe_gating_parameters(self) -> list[nn.Parameter]:
        if self.light_moe is not None:
            return list(self.light_moe.router.parameters())
        if self.mamba_moe is not None:
            params: list[nn.Parameter] = []
            for momeb in self.mamba_moe.blocks:
                params.extend(momeb.block.spatial_moe.router.parameters())  # type: ignore[union-attr]
            return params
        if self.moe is None:
            return []
        return list(self.moe.context_encoder.parameters()) + list(
            self.moe.router.parameters()
        )

    def update_moe_temperature(self, decay_rate: float = 0.9999) -> None:
        if self.moe is None:
            return
        self.moe.update_temperature(decay_rate=decay_rate)

    def forward(
        self, samples: torch.Tensor, depth_map: torch.Tensor | None = None
    ) -> dict:
        features = self.backbone(samples)

        # Convert dict to list format for compatibility with MSAA and PA-FPN
        features_list = [features[0], features[1], features[2], features[3]]

        if self.msaa is not None:
            features_list = self.msaa(features_list)

        # Use stable list indices across VGG and DINO backbones:
        # c3: 256ch, c4: 512ch, c5: 512ch
        c3, c4, c5 = features_list[1], features_list[2], features_list[3]
        c3_hr = c3  # Cache high-res features for optional sub-pixel refinement

        # Depth dual-stream fusion: fuse each scale before PA-FPN
        if self.use_depth and depth_map is not None:
            assert self.depth_backbone is not None
            # Resize depth_map to match input spatial dims (may differ due to padding)
            if depth_map.shape[-2:] != samples.shape[-2:]:
                depth_map = F.interpolate(
                    depth_map,
                    size=samples.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            d3, d4, d5 = self.depth_backbone(depth_map)
            c3 = self.depth_fusion_c3(c3, d3)  # type: ignore[misc]
            c4 = self.depth_fusion_c4(c4, d4)  # type: ignore[misc]
            c5 = self.depth_fusion_c5(c5, d5)  # type: ignore[misc]

        if self.use_depth_geo and depth_map is not None:
            c3 = self.geo_attn_c3(c3, depth_map)  # type: ignore[misc]
            c4 = self.geo_attn_c4(c4, depth_map)  # type: ignore[misc]
            c5 = self.geo_attn_c5(c5, depth_map)  # type: ignore[misc]

        features_pa = self.pa([c3, c4, c5])  # [batch_size, 256, 16, 16]

        # DINOv2 semantic injection: bounded gate (tanh) starts at 0
        if self.dino_injector is not None and self.dino_gate is not None:
            dino_feat = self.dino_injector(samples, target_size=features_pa.shape[-2:])
            features_pa = features_pa + self.dino_gate.tanh() * dino_feat

        batch_size = features_list[0].shape[0]
        density = self.density_pred(features_pa)

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
        }

        if self.use_multi_scale_density:
            density_block3 = self.density_pred_block3(c3)
            density_block4 = self.density_pred_block4(c4)
            density_block5 = self.density_pred_block5(c5)

            output_dict.update(
                {
                    "density_block3": density_block3,
                    "density_block4": density_block4,
                    "density_block5": density_block5,
                }
            )

        if self.use_moe:
            assert self.esca is not None and self.moe is not None
            esca_feature = self.esca(features_pa)
            feature_fl, moe_aux_losses, moe_weights = self.moe(
                esca_feature, density_hint=density, training=self.training
            )
            output_dict["moe_aux_losses"] = moe_aux_losses
            output_dict["moe_aux_total"] = moe_aux_losses.get("total_aux")
            output_dict["moe_weights"] = moe_weights
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
                    density, features_pa, uncertainty=uncertainty
                )
                feature_gcn_feature = self.feature_gcn(features_pa)
                if self.gm is not None:
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
        if self.light_moe is not None and not self.use_moe:
            feature_fl, light_aux, light_weights = self.light_moe(
                feature_fl, density_hint=density, training=self.training
            )
            output_dict["moe_aux_losses"] = light_aux
            output_dict["moe_aux_total"] = light_aux.get("total_aux")
            output_dict["moe_weights"] = light_weights

        # SEMC post-GCN enhancement (optional, disabled by default)
        if self.density_attention is not None:
            if self.use_multi_scale_density and isinstance(
                self.density_attention, MultiScaleDensityAttention
            ):
                attention_mask = self.density_attention(
                    output_dict["density_block3"].detach(),
                    output_dict["density_block4"].detach(),
                    output_dict["density_block5"].detach(),
                ).to(feature_fl.dtype)
            else:
                attention_mask = self.density_attention(density.detach()).to(
                    feature_fl.dtype
                )
            feature_fl = feature_fl * attention_mask

        if self.semc_enhancer is not None and not self.use_moe:
            feature_fl = self.semc_enhancer(
                feature_fl,
                density if self._semc_use_density_hint else None,
            )

        # Foreground suppression: residual-gated pixel-level FG prior
        if self.fg_branch is not None:
            feature_fl, fg_logits, fg_prob = self.fg_branch(feature_fl)
            output_dict["fg_logits"] = fg_logits
            output_dict["fg_prob"] = fg_prob

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
            fg_scores = output_class[:, :, 1].sigmoid()
            output_dict["pred_points"] = self.subpix_refine(
                hr_feat=c3_hr,
                lr_feat=features_pa,
                pred_points=output_dict["pred_points"],
                pred_scores=fg_scores,
                img_h=img_h,
                img_w=img_w,
            )

        return output_dict

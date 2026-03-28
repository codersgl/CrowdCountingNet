"""DSGCNet main model definition."""

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.gcn import DensityGCNProcessor, FeatureGCNProcessor
from crowdcount.models.head import (
    ClassificationModel,
    Density_pred,
    RegressionModel,
    DensityPred_Block3,
    DensityPred_Block4,
    DensityPred_Block5,
)
from crowdcount.models.neck import Decoder_SPD_PAFPN
from crowdcount.models.semc_blocks import SEMCEnhancer
from crowdcount.plugins.gm import GateMechanism
from crowdcount.plugins.isfm.depth_fusion import DepthFusionModule
from crowdcount.plugins.geo_prior import DepthGeoPriorAttention
from crowdcount.plugins.moe import ESCA, MoE
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
        use_msaa: bool = False,
        msaa_in_channels: int = 1280,
        msaa_reduction: int = 4,
        moe_cfg: DictConfig | None = None,
        use_depth: bool = False,
        depth_cfg: DictConfig | None = None,
        use_depth_geo: bool = False,
        depth_geo_cfg: DictConfig | None = None,
        cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg
        self.fusion_mode = fusion_mode
        self.use_moe = fusion_mode == "esca_moe"
        self.use_depth = use_depth
        self.use_depth_geo = use_depth_geo

        if self.fusion_mode not in {"gcn", "esca_moe"}:
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode}, expected 'gcn' or 'esca_moe'"
            )

        density_cfg = (
            getattr(cfg, "density_multi_scale", None) if cfg is not None else None
        )
        self.use_multi_scale_density = bool(
            getattr(density_cfg, "enabled", False) if density_cfg is not None else False
        )
        num_anchor_points = row * line

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
            self.pa = Decoder_SPD_PAFPN(1280, 1280, 1280)
        else:
            self.pa = Decoder_SPD_PAFPN(256, 512, 512)
        self.density_pred = Density_pred()

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
                float(getattr(moe_cfg, "temperature_min", 0.1))
                if moe_cfg is not None
                else 0.1
            )
            lambda_balance = (
                float(getattr(moe_cfg, "lambda_balance", 0.835))
                if moe_cfg is not None
                else 0.835
            )
            lambda_decorr = (
                float(getattr(moe_cfg, "lambda_decorr", 1.0))
                if moe_cfg is not None
                else 1.0
            )
            use_density_hint = (
                bool(getattr(moe_cfg, "use_density_hint", True))
                if moe_cfg is not None
                else True
            )

            self.esca: ESCA | None = ESCA(256)
            self.moe: MoE | None = MoE(
                input_dim=256,
                top_k=top_k,
                temperature_init=temperature_init,
                temperature_min=temperature_min,
                lambda_balance=lambda_balance,
                lambda_decorr=lambda_decorr,
                use_density_hint=use_density_hint,
            )
            self.density_gcn = None
            self.feature_gcn = None
            self.alpha = None
            self.gm = None
        else:
            self.density_gcn: DensityGCNProcessor | None = DensityGCNProcessor(k=4)
            self.feature_gcn: FeatureGCNProcessor | None = FeatureGCNProcessor(k=4)
            self.alpha: nn.Parameter | None = nn.Parameter(
                torch.ones(3, dtype=torch.float32)
            )
            self.gm: GateMechanism | None = (
                GateMechanism(input_dim=gm_input_dim, hidden_dim=gm_hidden_dim)
                if use_gm
                else None
            )
            self.esca = None
            self.moe = None

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

    def supports_moe(self) -> bool:
        return self.use_moe and self.moe is not None

    def get_moe_gating_parameters(self) -> list[nn.Parameter]:
        if self.moe is None:
            return []
        return list(self.moe.context_encoder.parameters()) + list(
            self.moe.router.parameters()
        )

    def set_moe_gating_trainable(self, trainable: bool) -> None:
        if self.moe is None:
            return
        for parameter in self.get_moe_gating_parameters():
            parameter.requires_grad = trainable

    def set_moe_training_stage(self, stage: str) -> None:
        if self.moe is None:
            return
        self.moe.set_training_stage(stage)
        # Gate is always trainable: in specialization it learns with high Gumbel noise
        # (exploration), in coordination it learns with normal noise (exploitation).
        # Freezing the gate in specialization would nullify noisy-gate routing.
        self.set_moe_gating_trainable(True)

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

        # Multi-scale density prediction (if enabled)
        output_dict = {
            "pred_logits": None,
            "pred_points": None,
            "density_out": density,
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
        else:
            assert self.density_gcn is not None
            assert self.feature_gcn is not None
            density_gcn_feature = self.density_gcn(density, features_pa)
            feature_gcn_feature = self.feature_gcn(features_pa)
            if self.gm is not None:
                gate_weight = self.gm(features_pa)
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

        # SEMC post-GCN enhancement (optional, disabled by default)
        if self.semc_enhancer is not None and not self.use_moe:
            feature_fl = self.semc_enhancer(
                feature_fl,
                density if self._semc_use_density_hint else None,
            )

        regression = self.regression(feature_fl) * 100
        classification = self.classification(feature_fl)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification

        output_dict["pred_logits"] = output_class
        output_dict["pred_points"] = output_coord

        return output_dict

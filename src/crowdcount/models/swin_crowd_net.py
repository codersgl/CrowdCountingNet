"""SwinCrowdNet: Swin-B + CrowdFPN + MoE-Lite for crowd counting.

Independent model class that reuses the existing prediction heads, anchor
generation, criterion, and training pipeline.  Output dict is compatible with
``engine.train_one_epoch`` and ``engine.evaluate_crowd_no_overlap``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.crowd_fpn import CrowdFPN
from crowdcount.models.head import (
    ClassificationModel,
    DecoupledPredictionHead,
    Density_pred,
    FreqDecoupledRouter,
    RegressionModel,
    SharedPredictionTrunk,
)
from crowdcount.models.moe_lite import MoELite
from crowdcount.models.swin_backbone import BackboneSwin


# ---------------------------------------------------------------------------
# Density-Feature Refinement module
# ---------------------------------------------------------------------------


class DensityFeatureRefine(nn.Module):
    """Density-guided feature refinement: bidirectional density-feature feedback.

    Uses the predicted density map to produce both spatial and channel
    attention on the feature map, creating a density-aware feature
    representation before the MoE-Lite module.

    Architecture:
        - Spatial attention: density → 3×3 DW conv → 1×1 conv → sigmoid
          → feature * (1 + mask)  (residual gating)
        - Channel attention:  density → GAP → FC → sigmoid
          → feature * weight  (SE-style recalibration)
    """

    def __init__(self, feature_dim: int = 256, reduction: int = 16) -> None:
        super().__init__()
        # Spatial attention path
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 1),
        )
        # Channel attention path
        mid = max(feature_dim // reduction, 16)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Refine feature using density information.

        Args:
            feature: [B, C, H, W] feature map from CrowdFPN.
            density: [B, 1, H, W] predicted density map (detached externally).

        Returns:
            Refined feature [B, C, H, W].
        """
        # Spatial attention: residual gating
        spatial_mask = self.spatial_conv(density).sigmoid()  # [B, 1, H, W]
        feat_spatial = feature * (1 + spatial_mask)

        # Channel attention: SE-style
        ch_weight = self.channel_fc(density)  # [B, C]
        ch_weight = ch_weight.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        feat_refined = feat_spatial * ch_weight

        return feat_refined


class SwinCrowdNet(nn.Module):
    """Swin Transformer + CrowdFPN + MoE-Lite crowd counting model.

    Forward pass:
        Image → Swin Stage 1-3 → CrowdFPN (stride 8) → Density_pred →
        DensityFeatureRefine → MoE-Lite (density-guided 3-expert fusion) →
        Prediction heads → output dict

    The output dict is fully compatible with the existing DSGCNet training
    pipeline (SetCriterion_Crowd, train_one_epoch, evaluate_crowd_no_overlap).
    """

    def __init__(
        self,
        backbone: nn.Module,
        row: int = 2,
        line: int = 2,
        feature_dim: int = 256,
        # CrowdFPN
        fpn_c2_channels: int = 256,
        fpn_c3_channels: int = 256,
        fpn_c4_channels: int = 512,
        # MoE-Lite
        moe_grid_stride: int = 4,
        moe_temperature_init: float = 1.0,
        moe_temperature_min: float = 0.3,
        moe_lambda_balance: float = 0.05,
        moe_dense_expansion: int = 2,
        moe_use_density_gate: bool = True,
        moe_lambda_decorr: float = 0.1,
        moe_lambda_diversity: float = 0.1,
        # Density-Feature Refine
        use_dfr: bool = True,
        # Prediction head
        use_decoupled_head: bool = True,
        use_freq_router: bool = True,
        # Misc
        cfg: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg
        self.use_dfr = use_dfr
        self.use_decoupled_head = use_decoupled_head
        self.use_freq_router = use_freq_router

        # --- Neck ---
        self.crowd_fpn = CrowdFPN(
            C2_channels=fpn_c2_channels,
            C3_channels=fpn_c3_channels,
            C4_channels=fpn_c4_channels,
            feature_size=128,
            out_channels=feature_dim,
        )

        # --- Density prediction (auxiliary task) ---
        self.density_pred = Density_pred()

        # --- Density-Feature Refinement ---
        self.dfr: DensityFeatureRefine | None = (
            DensityFeatureRefine(feature_dim=feature_dim) if use_dfr else None
        )

        # --- MoE-Lite ---
        self.moe = MoELite(
            dim=feature_dim,
            grid_stride=moe_grid_stride,
            temperature_init=moe_temperature_init,
            temperature_min=moe_temperature_min,
            lambda_balance=moe_lambda_balance,
            dense_expansion=moe_dense_expansion,
            use_density_gate=moe_use_density_gate,
            lambda_decorr=moe_lambda_decorr,
            lambda_diversity=moe_lambda_diversity,
        )

        # --- Prediction heads ---
        num_anchor_points = row * line
        if use_decoupled_head:
            self.pred_trunk: SharedPredictionTrunk | DecoupledPredictionHead = (
                DecoupledPredictionHead(
                    in_channels=feature_dim, feature_size=feature_dim
                )
            )
        else:
            self.pred_trunk = SharedPredictionTrunk(
                in_channels=feature_dim, feature_size=feature_dim
            )
        self.regression = RegressionModel(
            num_features_in=feature_dim, num_anchor_points=num_anchor_points
        )
        self.classification = ClassificationModel(
            num_features_in=feature_dim,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points,
        )

        # --- Frequency-Decoupled Router ---
        self.freq_router: FreqDecoupledRouter | None = (
            FreqDecoupledRouter(kernel_size=3) if use_freq_router else None
        )

        # --- Anchor generation ---
        # CrowdFPN output is at stride 8 → pyramid_level 3
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)

    # ------------------------------------------------------------------
    # MoE interface (consumed by Trainer / engine)
    # ------------------------------------------------------------------

    def supports_moe(self) -> bool:
        return True

    def get_moe_gating_parameters(self) -> list[nn.Parameter]:
        return list(self.moe.router.parameters())

    def update_moe_temperature(self, decay_rate: float = 0.9999) -> None:
        self.moe.update_temperature(decay_rate)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, samples: torch.Tensor, **kwargs) -> dict:
        # 1) Backbone: Swin Stage 1-3
        features = self.backbone(samples)  # [C2, C3, C4]

        # 2) CrowdFPN → fused features at stride 8
        feat = self.crowd_fpn(features)  # [B, 256, H/8, W/8]

        # 3) Density prediction
        density = self.density_pred(feat)  # [B, 1, H/8, W/8]

        # 4) Density-Feature Refinement: density → spatial + channel attention
        if self.dfr is not None:
            feat = self.dfr(feat, density.detach())

        # 5) MoE-Lite (density-guided expert fusion)
        feat, moe_aux, moe_weights = self.moe(feat, density, training=self.training)

        # 6) Prediction heads
        if self.use_decoupled_head:
            cls_feat, reg_feat = self.pred_trunk(feat)
            if self.freq_router is not None:
                _f_low, reg_feat, _ = self.freq_router(reg_feat)
            regression = self.regression(reg_feat) * 100
            classification = self.classification(cls_feat)
        else:
            shared_feat = self.pred_trunk(feat)
            if self.freq_router is not None:
                _f_low, f_high, f_full = self.freq_router(shared_feat)
                regression = self.regression(f_high) * 100
                classification = self.classification(f_full)
            else:
                regression = self.regression(shared_feat) * 100
                classification = self.classification(shared_feat)

        # 7) Anchor points
        batch_size = samples.shape[0]
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points

        return {
            "pred_logits": classification,
            "pred_points": output_coord,
            "density_out": density,
            "uncertainty_map": None,
            "img_size": (samples.shape[-2], samples.shape[-1]),
            "moe_aux_losses": moe_aux,
            "moe_aux_total": moe_aux.get("total_aux"),
            "moe_weights": moe_weights,
        }

"""SwinCrowdNet: Swin-B + CrowdFPN + MoE-Lite for crowd counting.

Independent model class that reuses the existing prediction heads, anchor
generation, criterion, and training pipeline.  Output dict is compatible with
``engine.train_one_epoch`` and ``engine.evaluate_crowd_no_overlap``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.crowd_fpn import CrowdFPN
from crowdcount.models.head import (
    ClassificationModel,
    Density_pred,
    RegressionModel,
    SharedPredictionTrunk,
)
from crowdcount.models.moe_lite import MoELite
from crowdcount.models.swin_backbone import BackboneSwin


class SwinCrowdNet(nn.Module):
    """Swin Transformer + CrowdFPN + MoE-Lite crowd counting model.

    Forward pass:
        Image → Swin Stage 1-3 → CrowdFPN (stride 8) → Density_pred →
        MoE-Lite (density-guided 3-expert fusion) → SharedPredictionTrunk →
        RegressionModel + ClassificationModel → output dict

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
        moe_lambda_balance: float = 0.01,
        moe_dense_expansion: int = 2,
        # Misc
        cfg: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg

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

        # --- MoE-Lite ---
        self.moe = MoELite(
            dim=feature_dim,
            grid_stride=moe_grid_stride,
            temperature_init=moe_temperature_init,
            temperature_min=moe_temperature_min,
            lambda_balance=moe_lambda_balance,
            dense_expansion=moe_dense_expansion,
        )

        # --- Prediction heads (reused from DSGCNet) ---
        num_anchor_points = row * line
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

        # 4) MoE-Lite (density-guided expert fusion)
        feat, moe_aux, moe_weights = self.moe(feat, density, training=self.training)

        # 5) Shared prediction trunk → regression + classification
        shared_feat = self.pred_trunk(feat)
        regression = self.regression(shared_feat) * 100
        classification = self.classification(shared_feat)

        # 6) Anchor points
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

"""DSGCNet main model definition."""

import torch
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
from crowdcount.plugins.gm import GateMechanism
from crowdcount.plugins.msaa import MsaaAdaptiveLayer


class DSGCnet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        row: int = 2,
        line: int = 2,
        use_gm: bool = False,
        gm_input_dim: int = 256,
        gm_hidden_dim: int = 128,
        use_msaa: bool = False,
        msaa_in_channels: int = 1280,
        msaa_reduction: int = 4,
        cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.cfg = cfg
        density_cfg = (
            getattr(cfg, "density_multi_scale", None) if cfg is not None else None
        )
        self.use_multi_scale_density = bool(
            getattr(density_cfg, "enabled", False) if density_cfg is not None else False
        )
        num_anchor_points = row * line

        self.fusion_total = nn.Sequential(
            nn.Conv2d(3 * 256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
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

        self.density_gcn = DensityGCNProcessor(k=4)
        self.feature_gcn = FeatureGCNProcessor(k=4)
        self.alpha = nn.Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        )
        self.gm: GateMechanism | None = (
            GateMechanism(input_dim=gm_input_dim, hidden_dim=gm_hidden_dim)
            if use_gm
            else None
        )
        self.msaa: MsaaAdaptiveLayer | None = (
            MsaaAdaptiveLayer(in_channels=msaa_in_channels, reduction=msaa_reduction)
            if use_msaa
            else None
        )

    def forward(self, samples: torch.Tensor) -> dict:
        features = self.backbone(samples)

        # Convert dict to list format for compatibility with MSAA and PA-FPN
        features_list = [features[0], features[1], features[2], features[3]]

        if self.msaa is not None:
            features_list = self.msaa(features_list)

        # Use stable list indices across VGG and DINO backbones:
        # c3: 256ch, c4: 512ch, c5: 512ch
        c3, c4, c5 = features_list[1], features_list[2], features_list[3]

        features_pa = self.pa([c3, c4, c5])  # [batch_size, 256, 16, 16]

        batch_size = features_list[0].shape[0]
        density = self.density_pred(features_pa)

        # Multi-scale density prediction (if enabled)
        output_dict = {
            "pred_logits": None,
            "pred_points": None,
            "density_out": density,
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
            feature_fl = (
                features_pa
                + self.alpha[0] * density_gcn_feature
                + self.alpha[1] * feature_gcn_feature
            )

        regression = self.regression(feature_fl) * 100
        classification = self.classification(feature_fl)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification

        output_dict["pred_logits"] = output_class
        output_dict["pred_points"] = output_coord

        return output_dict

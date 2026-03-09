"""DSGCNet main model definition."""

import torch
from torch import nn

from crowdcount.utils.misc import nested_tensor_from_tensor_list
from crowdcount.models.anchor import AnchorPoints
from crowdcount.models.gcn import DensityGCNProcessor, FeatureGCNProcessor
from crowdcount.models.head import ClassificationModel, Density_pred, RegressionModel
from crowdcount.models.neck import Decoder_SPD_PAFPN


class DSGCnet(nn.Module):
    def __init__(self, backbone: nn.Module, row: int = 2, line: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
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
        self.pa = Decoder_SPD_PAFPN(256, 512, 512)
        self.density_pred = Density_pred()
        self.density_gcn = DensityGCNProcessor(k=4)
        self.feature_gcn = FeatureGCNProcessor(k=4)
        self.alpha = nn.Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, samples: torch.Tensor) -> dict:
        features = self.backbone(samples)
        features_pa = self.pa([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        density = self.density_pred(features_pa)
        density_gcn_feature = self.density_gcn(density, features_pa)
        feature_gcn_feature = self.feature_gcn(features_pa)
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

        return {
            "pred_logits": output_class,
            "pred_points": output_coord,
            "density_out": density,
        }

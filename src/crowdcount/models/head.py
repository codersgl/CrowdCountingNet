"""Prediction head modules for DSGCNet.

Contains:
  - Density_pred:         density map regression head
  - SharedPredictionTrunk: shared 2-layer conv trunk for regression & classification
  - RegressionModel:      point regression projection (used after SharedPredictionTrunk)
  - ClassificationModel:  point classification projection (used after SharedPredictionTrunk)
  - PointRefineModule:    iterative coordinate refinement via feature re-sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPredictionTrunk(nn.Module):
    """Shared 2-layer conv feature extractor for regression and classification heads.

    Replaces the duplicate conv1/conv2 pairs that previously existed independently
    in both RegressionModel and ClassificationModel.

    Input:  [B, in_channels, H, W]
    Output: [B, feature_size, H, W]
    """

    def __init__(self, in_channels: int = 256, feature_size: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        return self.act2(self.conv2(out))


class Density_pred(nn.Module):
    """Density map prediction head."""

    def __init__(self):
        super().__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.v1(x)
        x = self.v2(x)
        x = self.v3(x)
        return self.conv_layers(x)


class RegressionModel(nn.Module):
    """Point coordinate regression projection head.

    Applies a single output convolution on features that have already been
    processed by SharedPredictionTrunk.  The ``num_features_in`` argument is
    retained for API compatibility; it must equal the trunk's ``feature_size``
    (default 256).
    """

    def __init__(
        self, num_features_in: int, num_anchor_points: int = 4, feature_size: int = 256
    ):
        super().__init__()
        self.output = nn.Conv2d(
            num_features_in, num_anchor_points * 2, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.output(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    """Point classification projection head.

    Applies a single output convolution on features that have already been
    processed by SharedPredictionTrunk.  The ``num_features_in`` and
    ``prior`` arguments are retained for API compatibility.
    """

    def __init__(
        self,
        num_features_in: int,
        num_anchor_points: int = 4,
        num_classes: int = 80,
        prior: float = 0.01,
        feature_size: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.output = nn.Conv2d(
            num_features_in, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.output(x)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(
            batch_size, width, height, self.num_anchor_points, self.num_classes
        )
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class DensityPred_Backbone(nn.Module):
    """Parametric density map prediction head for backbone features.

    Supports different input channel sizes (256, 512, etc.) with adaptive head design.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # Three conv blocks with same channel as input
        self.v1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Projection to 1 channel
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, in_channels, H, W]

        Returns:
            Density map of shape [B, 1, H, W]
        """
        x = self.v1(x)
        x = self.v2(x)
        x = self.v3(x)
        return self.conv_layers(x)


class DensityPred_Block3(nn.Module):
    """Density prediction head for VGG block3 features (256 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DensityPred_Block4(nn.Module):
    """Density prediction head for VGG block4 features (512 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DensityPred_Block5(nn.Module):
    """Density prediction head for VGG block5 features (512 channels)."""

    def __init__(self):
        super().__init__()
        self.head = DensityPred_Backbone(in_channels=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PointRefineModule(nn.Module):
    """Iterative point coordinate refinement via feature re-sampling.

    At each step, bilinear-samples from the feature map at the current
    predicted coordinates, then predicts a small residual offset via a
    shared MLP.  After *T* steps the accumulated refinement significantly
    reduces localisation error compared to single-shot regression.

    Args:
        feature_dim: Channel dimension of the feature map to sample from.
        hidden_dim:  MLP hidden dimension.
        num_steps:   Number of refinement iterations *T*.
        share_weights: If True, all steps share the same MLP parameters.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_steps: int = 2,
        share_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.share_weights = share_weights

        if share_weights:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, 2),
                    )
                    for _ in range(num_steps)
                ]
            )

    def _sample_features(
        self,
        feature_map: torch.Tensor,
        points: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Bilinear-sample features at *points* (pixel coords).

        Args:
            feature_map: [B, C, Hf, Wf]
            points:      [B, Q, 2]  (x, y) in pixel space
            img_h:       Original image height
            img_w:       Original image width

        Returns:
            [B, Q, C] sampled feature vectors.
        """
        B, Q, _ = points.shape
        # Normalise to [-1, 1] for grid_sample
        grid_x = 2.0 * points[:, :, 0] / max(img_w - 1, 1) - 1.0
        grid_y = 2.0 * points[:, :, 1] / max(img_h - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, Q, 2]
        grid = grid.unsqueeze(2)  # [B, Q, 1, 2]

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, Q, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, Q, C]

    def forward(
        self,
        feature_map: torch.Tensor,
        init_points: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run iterative refinement.

        Args:
            feature_map: [B, C, Hf, Wf] — the feature map to sample from.
            init_points: [B, Q, 2] — initial point predictions (pixel coords).
            img_h: Image height (pixels).
            img_w: Image width (pixels).

        Returns:
            refined_points: [B, Q, 2] — final refined coordinates.
            intermediates:  List of [B, Q, 2] tensors, one per step
                            (including the initial prediction at index 0).
        """
        points = init_points
        intermediates = [points]

        for t in range(self.num_steps):
            feat = self._sample_features(feature_map, points.detach(), img_h, img_w)
            mlp = self.mlp if self.share_weights else self.mlps[t]
            delta = mlp(feat)  # [B, Q, 2]
            points = points + delta
            intermediates.append(points)

        return points, intermediates

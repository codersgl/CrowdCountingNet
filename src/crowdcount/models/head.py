"""Prediction head modules for DSGCNet.

Contains:
  - Density_pred:           density map regression head
  - SharedPredictionTrunk:  shared 2-layer conv trunk for regression & classification
  - DecoupledPredictionHead: independent trunks for classification & regression (decoupled)
  - RegressionModel:        point regression projection (used after SharedPredictionTrunk)
  - ClassificationModel:    point classification projection (used after SharedPredictionTrunk)
  - PointRefineModule:      iterative coordinate refinement via feature re-sampling
  - FreqDecoupledRouter:    frequency-domain Laplacian decomposition for head routing
  - SubPixelRefineModule:   dense-region sub-pixel refinement via high-res feature sampling
  - ForegroundSuppressionBranch: pixel-level foreground gating with residual pass-through
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityAttentionMask(nn.Module):
    """Convert density maps into a broadcastable spatial attention mask."""

    def __init__(self, mode: str = "sigmoid", hidden_channels: int = 16) -> None:
        super().__init__()
        if mode not in {"sigmoid", "learned"}:
            raise ValueError(
                f"Unsupported density attention mode={mode}, expected 'sigmoid' or 'learned'"
            )
        self.mode = mode

        if mode == "sigmoid":
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
            self.proj = None
        else:
            self.scale = None
            self.bias = None
            self.proj = nn.Sequential(
                nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        if self.mode == "sigmoid":
            assert self.scale is not None and self.bias is not None
            return torch.sigmoid(density * self.scale + self.bias)
        assert self.proj is not None
        return torch.sigmoid(self.proj(density))


class EnhancedDensityAttention(nn.Module):
    """Multi-scale, channel+spatial density attention with gradient-aware boundary enhancement.

    Combines four complementary mechanisms to maximise the utility of density maps:

    1. **Multi-scale density encoder** – depthwise dilated convolutions (d=1,2,3)
       capture density patterns at multiple receptive-field sizes.
    2. **Density gradient (boundary) branch** – fixed Sobel filters extract horizontal
       and vertical gradients; a 1×1 conv compresses them into a boundary-aware feature.
    3. **Channel attention** – SE-style squeeze-excite produces per-channel weights
       conditioned on the density encoding, so each feature channel is density-aware.
    4. **Spatial attention** – a lightweight projection produces a per-pixel mask
       that highlights foreground / suppresses background.
    5. **Residual dual-path output** – ``feature * (base + spatial_mask) * channel_weight``
       where *base* is a learnable scalar (init 0.5) that prevents complete suppression.

    Args:
        feature_channels: Number of channels in the feature tensor (default 256).
        hidden_channels: Internal width of density encoder / gradient branch (default 32).
        base_init: Initial value for the residual base scalar (default 0.5).
    """

    def __init__(
        self,
        feature_channels: int = 256,
        hidden_channels: int = 32,
        base_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels

        # --- Multi-scale density encoder (dilation 1/2/3) ---
        self.ms_dw_d1 = nn.Conv2d(
            1, hidden_channels, kernel_size=3, padding=1, dilation=1, bias=False
        )
        self.ms_dw_d2 = nn.Conv2d(
            1, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=False
        )
        self.ms_dw_d3 = nn.Conv2d(
            1, hidden_channels, kernel_size=3, padding=3, dilation=3, bias=False
        )
        self.ms_bn = nn.BatchNorm2d(hidden_channels)

        # --- Density gradient (Sobel) branch ---
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.grad_proj = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # --- Fusion of multi-scale + gradient features ---
        fused_channels = hidden_channels * 2  # ms + grad
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # --- Channel attention (SE-style) ---
        reduction = max(feature_channels // 16, 4)
        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(hidden_channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, feature_channels, bias=False),
            nn.Sigmoid(),
        )

        # --- Spatial attention ---
        self.sa_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # --- Residual base ---
        self.base = nn.Parameter(torch.tensor(base_init))

    def forward(self, density: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """Apply density-guided channel + spatial attention with residual path.

        Args:
            density: Density map ``[B, 1, H, W]`` (detached recommended).
            feature: Feature tensor ``[B, C, H, W]`` to be modulated.

        Returns:
            Modulated features ``[B, C, H, W]``.
        """
        # Multi-scale density encoding
        ms = self.ms_dw_d1(density) + self.ms_dw_d2(density) + self.ms_dw_d3(density)
        ms = F.relu(self.ms_bn(ms), inplace=True)  # [B, hidden, H, W]

        # Gradient (boundary) encoding
        gx = F.conv2d(density, self.sobel_x, padding=1)
        gy = F.conv2d(density, self.sobel_y, padding=1)
        grad = self.grad_proj(torch.cat([gx, gy], dim=1))  # [B, hidden, H, W]

        # Fuse multi-scale + gradient
        fused = self.fuse_conv(torch.cat([ms, grad], dim=1))  # [B, hidden, H, W]

        # Channel attention
        ca = self.ca_pool(fused).squeeze(-1).squeeze(-1)  # [B, hidden]
        ca = self.ca_fc(ca).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Spatial attention
        sa = self.sa_conv(fused)  # [B, 1, H, W]

        # Residual dual-path: feature * (base + spatial_mask) * channel_weight
        return feature * (self.base + sa) * ca


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


class DecoupledPredictionHead(nn.Module):
    """Fully decoupled prediction head with independent trunks for each task.

    Instead of sharing a single :class:`SharedPredictionTrunk` between the
    classification and regression branches, this module maintains two
    independent trunks so that each task learns its own feature representation.

    Input:  [B, in_channels, H, W]
    Output: (cls_feat, reg_feat) — each [B, feature_size, H, W]
    """

    def __init__(self, in_channels: int = 256, feature_size: int = 256) -> None:
        super().__init__()
        self.cls_trunk = SharedPredictionTrunk(in_channels, feature_size)
        self.reg_trunk = SharedPredictionTrunk(in_channels, feature_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_trunk(x), self.reg_trunk(x)


class Density_pred(nn.Module):
    """Density map prediction head (baseline)."""

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


class Density_pred_MS(nn.Module):
    """Improved density head: multi-scale dilated convolutions + residual + Softplus."""

    def __init__(self):
        super().__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=3, dilation=3),
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
            nn.Softplus(beta=1, threshold=20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.v1(x) + self.v2(x) + self.v3(x)
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


class FreqDecoupledRouter(nn.Module):
    """Frequency-domain Laplacian decomposition for task-specific head routing.

    Splits the shared trunk features into low-frequency and high-frequency
    components using a simple AvgPool-based Laplacian decomposition:
        F_low  = AvgPool(F)           — smooth / low-freq component
        F_high = F - F_low            — edge / high-freq component

    Routes:
        - Density head    ← F_low   (density maps are Gaussian-smoothed → low-freq)
        - Regression head ← F_high  (point coords are Dirac-like → high-freq)
        - Classification  ← F       (original, no frequency bias)

    Zero learnable parameters, zero information loss (F_low + F_high = F).
    """

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def forward(
        self, shared_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (feat_for_density, feat_for_regression, feat_for_classification)."""
        f_low = self.pool(shared_feat)
        f_high = shared_feat - f_low
        return f_low, f_high, shared_feat


class SubPixelRefineModule(nn.Module):
    """Dense-region sub-pixel refinement via high-resolution feature sampling.

    For the top-K highest-confidence predicted points, bilinear-samples from
    the high-resolution backbone feature map (C3, stride 8) and predicts a
    residual coordinate offset, breaking the Nyquist resolution limit of the
    H/16 prediction grid.

    Args:
        hr_channels: Channel dim of the high-res feature map (default 256 for C3).
        lr_channels: Channel dim of the low-res feature map (default 256 for features_pa).
        hidden_dim:  MLP hidden dimension.
        top_k:       Number of high-confidence points to refine.
    """

    def __init__(
        self,
        hr_channels: int = 256,
        lr_channels: int = 256,
        hidden_dim: int = 128,
        top_k: int = 512,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.mlp = nn.Sequential(
            nn.Linear(hr_channels + lr_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        # Initialize to zero offset (identity residual)
        output_layer = self.mlp[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    @staticmethod
    def _sample_at(
        feat_map: torch.Tensor,
        points: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Bilinear-sample *feat_map* at pixel coordinates *points*.

        Args:
            feat_map: [B, C, Hf, Wf]
            points:   [B, K, 2]  (x, y) in original image pixel space
            img_h, img_w: original image dimensions for normalisation

        Returns:
            [B, K, C] sampled feature vectors.
        """
        grid_x = 2.0 * points[:, :, 0] / max(img_w - 1, 1) - 1.0
        grid_y = 2.0 * points[:, :, 1] / max(img_h - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [B, K, 1, 2]
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, K, 1]
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, K, C]

    def forward(
        self,
        hr_feat: torch.Tensor,
        lr_feat: torch.Tensor,
        pred_points: torch.Tensor,
        pred_scores: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Refine top-K points using high-resolution features.

        Args:
            hr_feat:      [B, C_hr, H/8, W/8] high-res backbone feature (C3).
            lr_feat:      [B, C_lr, H/16, W/16] low-res fused feature.
            pred_points:  [B, Q, 2] predicted point coordinates.
            pred_scores:  [B, Q] foreground confidence scores.
            img_h, img_w: original image dimensions.

        Returns:
            refined_points: [B, Q, 2] with top-K points refined in-place.
        """
        B, Q, _ = pred_points.shape
        K = min(self.top_k, Q)

        # Select top-K confident points per image
        _, topk_idx = torch.topk(pred_scores, K, dim=1)  # [B, K]

        # Gather coordinates for selected points
        topk_points = torch.gather(
            pred_points, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, K, 2]

        # Sample features from both resolutions
        hr_sampled = self._sample_at(hr_feat, topk_points.detach(), img_h, img_w)
        lr_sampled = self._sample_at(lr_feat, topk_points.detach(), img_h, img_w)

        # Predict residual offset
        delta = self.mlp(torch.cat([hr_sampled, lr_sampled], dim=-1))  # [B, K, 2]

        # Scatter refined offsets back
        refined = pred_points.clone()
        scatter_idx = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        refined = refined.scatter(1, scatter_idx, topk_points + delta)

        return refined


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


class ForegroundSuppressionBranch(nn.Module):
    """Lightweight pixel-level foreground probability branch.

    Produces a spatial foreground mask supervised by binarised GT density maps.
    Used with residual gating so that even when fg_prob is low, a base fraction
    of the feature is preserved (avoids recall collapse in early training).

    Args:
        in_channels: Input feature channels (default 256).
        hidden_channels: Intermediate conv channels (default 64).
        base: Minimum feature pass-through ratio (default 0.5).
        scale: Maximum additional boost from fg_prob (default 0.5).
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        base: float = 0.5,
        scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = base
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (gated_feature, fg_logits, fg_prob).

        - gated_feature: x * (base + scale * fg_prob)  [B, C, H, W]
        - fg_logits:     raw logits before sigmoid       [B, 1, H, W]
        - fg_prob:       sigmoid(fg_logits)              [B, 1, H, W]
        """
        fg_logits = self.conv2(self.act(self.bn(self.conv1(x))))
        fg_prob = fg_logits.sigmoid()
        gated = x * (self.base + self.scale * fg_prob)
        return gated, fg_logits, fg_prob

"""ASACL: Adaptive Structural-Perceptual Composite Loss for crowd density maps.

A multi-component density map loss that addresses three core challenges:
    1. Density imbalance — adaptive spatial weighting suppresses gradient
       domination by high-density pixels while enforcing precision on
       low-density / background regions.
    2. Shape distortion — local SSIM (window=5) preserves per-head Gaussian
       peak structure.
    3. Semantic gap — multi-scale VGG perceptual features tolerate minor
       spatial misalignment in GT annotations.

Total loss:
    L_total = λ_adapt · L_adapt + λ_struct · L_struct + λ_percept · L_percept

References:
    - SANet (AAAI 2018), CLTR (BMVC 2022), DHCNet (TIP 2024)
    - SSIM: Wang et al., IEEE TIP 2004
    - Perceptual loss: Johnson et al., ECCV 2016
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class AdaptiveStructuralPerceptualLoss(nn.Module):
    """ASACL composite density map loss.

    Drop-in replacement for ``nn.MSELoss(reduction="sum")`` used as the
    density criterion in *engine.py*.  Returns a scalar loss compatible
    with the existing training loop (pre-multiplied by batch size).

    Args:
        beta:          Adaptive weight control (higher → less suppression of
                       high-density pixels).  Default 1.0.
        lambda_adapt:  Weight for adaptive spatial L1.  Default 1.0.
        lambda_struct: Weight for SSIM structure loss.  Default 0.5.
        lambda_percept: Weight for VGG perceptual loss.  Default 0.1.
    """

    def __init__(
        self,
        beta: float = 1.0,
        lambda_adapt: float = 1.0,
        lambda_struct: float = 0.5,
        lambda_percept: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.lambda_adapt = lambda_adapt
        self.lambda_struct = lambda_struct
        self.lambda_percept = lambda_percept

        # --- Perceptual feature extractor (frozen VGG16-BN) ---
        # Split into 3 segments to extract intermediate features:
        #   slice0: input → conv1_2 relu  (features[0:6])
        #   slice1: → conv2_2 relu        (features[6:13])
        #   slice2: → conv3_3 relu        (features[13:23])
        vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        feats = list(vgg.features.children())
        self.vgg_slices = nn.ModuleList(
            [
                nn.Sequential(*feats[:6]),
                nn.Sequential(*feats[6:13]),
                nn.Sequential(*feats[13:23]),
            ]
        )
        for param in self.vgg_slices.parameters():
            param.requires_grad = False
        # Lock BN to eval mode so it uses pretrained running stats
        self.vgg_slices.eval()

        # ImageNet normalisation buffers
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Stash last-computed components for logging (detached scalars)
        self._last_components: dict[str, float] = {}

    def train(self, mode: bool = True) -> "AdaptiveStructuralPerceptualLoss":
        """Override to keep VGG slices permanently in eval mode.

        Without this, ``model.train()`` in the training loop would switch
        VGG's BatchNorm layers to batch-statistics mode, corrupting the
        pretrained running mean/var and making perceptual loss unstable.
        """
        super().train(mode)
        # Always revert VGG to eval — frozen BN must use running stats
        self.vgg_slices.eval()
        return self

    # ------------------------------------------------------------------
    # Component 1: Adaptive spatial-weighted L1
    # ------------------------------------------------------------------
    def adaptive_weighted_l1(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """W_i = β / (|Y_i| + β);  L = mean(W · |pred - target|)."""
        weight = self.beta / (target.abs() + self.beta + 1e-6)
        return (weight * (pred - target).abs()).mean()

    # ------------------------------------------------------------------
    # Component 2: Local SSIM structure loss (window=5)
    # ------------------------------------------------------------------
    def structural_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 − SSIM with 5×5 average-pooling window and ε-shift."""
        eps = 1e-6
        C1 = 0.01**2
        C2 = 0.03**2

        pred_s = pred + eps
        target_s = target + eps

        mu_p = F.avg_pool2d(pred_s, kernel_size=5, stride=1, padding=2)
        mu_t = F.avg_pool2d(target_s, kernel_size=5, stride=1, padding=2)

        sigma_p = (
            F.avg_pool2d(pred_s * pred_s, kernel_size=5, stride=1, padding=2)
            - mu_p * mu_p
        )
        sigma_t = (
            F.avg_pool2d(target_s * target_s, kernel_size=5, stride=1, padding=2)
            - mu_t * mu_t
        )
        sigma_pt = (
            F.avg_pool2d(pred_s * target_s, kernel_size=5, stride=1, padding=2)
            - mu_p * mu_t
        )

        # Clamp variances to avoid negative values from numerical errors
        sigma_p = sigma_p.clamp(min=0.0)
        sigma_t = sigma_t.clamp(min=0.0)

        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
            (mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2)
        )

        return 1.0 - ssim_map.mean()

    # ------------------------------------------------------------------
    # Component 3: Multi-scale VGG perceptual loss
    # ------------------------------------------------------------------
    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 distance between VGG16-BN intermediate features."""
        pred_3c = self._prepare_for_vgg(pred)
        # Target has no grad; skip graph construction to save GPU memory
        with torch.no_grad():
            target_3c = self._prepare_for_vgg(target)

        loss = torch.tensor(0.0, device=pred.device)
        p_feat = pred_3c
        with torch.no_grad():
            t_feat = target_3c
        for vgg_slice in self.vgg_slices:
            p_feat = vgg_slice(p_feat)
            with torch.no_grad():
                t_feat = vgg_slice(t_feat)
            loss = loss + F.l1_loss(p_feat, t_feat)

        return loss / len(self.vgg_slices)

    def _prepare_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise density map to [0,1], replicate to 3 ch, apply ImageNet stats.

        Uses per-sample min-max normalisation when the dynamic range is
        meaningful (> ``_MIN_RANGE``).  Falls back to simple clamping for
        near-constant maps (e.g. all-zero predictions early in training)
        to avoid extreme gradient magnification from dividing by ≈0.
        """
        _MIN_RANGE = 1e-3  # below this, treat as near-constant
        b = x.shape[0]
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=1).values.view(b, 1, 1, 1)
        x_max = x_flat.max(dim=1).values.view(b, 1, 1, 1)
        drange = x_max - x_min  # [B, 1, 1, 1]

        # Per-sample: use min-max when range is large enough, else clamp to [0,1]
        safe_drange = drange.clamp(min=_MIN_RANGE)
        x_minmax = (x - x_min) / safe_drange
        x_clamp = x.clamp(0.0, 1.0)
        # Smoothly select: when drange >= _MIN_RANGE use minmax, else clamp
        use_minmax = (drange >= _MIN_RANGE).float()
        x_norm = use_minmax * x_minmax + (1.0 - use_minmax) * x_clamp

        # Replicate to 3 channels (contiguous copy) and apply ImageNet normalisation
        x_3c = x_norm.repeat(1, 3, 1, 1)
        return (x_3c - self.vgg_mean) / self.vgg_std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ASACL loss.

        Args:
            pred:   [B, 1, H, W] predicted density map.
            target: [B, 1, H, W] ground-truth density map.

        Returns:
            Scalar combined loss (pre-multiplied by B × H × W to match
            ``MSELoss(reduction="sum")`` convention in *engine.py*).
        """
        l_adapt = self.adaptive_weighted_l1(pred, target)
        l_struct = self.structural_loss(pred, target)
        l_percept = self.perceptual_loss(pred, target)

        total = (
            self.lambda_adapt * l_adapt
            + self.lambda_struct * l_struct
            + self.lambda_percept * l_percept
        )

        # Cache for metric logging (no grad overhead)
        self._last_components = {
            "den_adapt_loss": l_adapt.detach().item(),
            "den_struct_loss": l_struct.detach().item(),
            "den_percept_loss": l_percept.detach().item(),
        }

        # Multiply by B × H × W to match MSELoss(reduction="sum") convention.
        # MSE(sum) aggregates over all spatial elements; without the H×W factor
        # the density gradient is negligible compared to the detection losses.
        return total * pred.shape[0] * pred.shape[2] * pred.shape[3]

    @property
    def last_components(self) -> dict[str, float]:
        """Return per-component losses from the most recent forward pass."""
        return self._last_components

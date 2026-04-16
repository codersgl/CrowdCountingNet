"""Swin Transformer backbone wrapper using timm."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class BackboneSwin(nn.Module):
    """Swin Transformer backbone extracting Stage 1-3 multi-scale features.

    Uses ``timm.create_model`` with ``features_only=True`` to extract
    intermediate feature maps.  Stage 4 is excluded by design (not connected
    to CrowdFPN in the architecture).

    Output contract (after projection):
        - C2: [B, 256, H/4,  W/4 ]   (Stage 1)
        - C3: [B, 256, H/8,  W/8 ]   (Stage 2)
        - C4: [B, 512, H/16, W/16]   (Stage 3)
    """

    # Map user-facing names to timm model identifiers
    VARIANT_MAP: dict[str, tuple[str, list[int]]] = {
        "swin_tiny": ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", [96, 192, 384]),
        "swin_small": (
            "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
            [96, 192, 384],
        ),
        "swin_base": ("swin_base_patch4_window7_224.ms_in22k_ft_in1k", [128, 256, 512]),
    }

    def __init__(self, variant: str = "swin_base", pretrained: bool = True) -> None:
        super().__init__()

        if variant not in self.VARIANT_MAP:
            raise ValueError(
                f"Unknown Swin variant '{variant}'. "
                f"Choose from {list(self.VARIANT_MAP.keys())}"
            )

        timm_name, raw_channels = self.VARIANT_MAP[variant]

        # Extract Stage 1-3 only (out_indices 0,1,2)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2),
        )

        # Enable dynamic input resolution:
        # 1) Disable strict img_size check in PatchEmbed
        # 2) Clear precomputed shifted-window attention masks so they are
        #    recomputed on-the-fly for any spatial size.
        self.backbone.patch_embed.strict_img_size = False
        for _name, module in self.backbone.named_modules():
            if hasattr(module, "attn_mask") and module.attn_mask is not None:
                module.attn_mask = None

        # Projection layers to normalise channel dimensions:
        # Stage 1 → 256, Stage 2 → 256, Stage 3 → 512
        target_channels = [256, 256, 512]
        self.projections = nn.ModuleList()
        for raw_ch, tgt_ch in zip(raw_channels, target_channels):
            if raw_ch == tgt_ch:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(
                    nn.Sequential(
                        nn.Conv2d(raw_ch, tgt_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(tgt_ch),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return projected multi-scale features [C2, C3, C4]."""
        features = self.backbone(x)  # list of 3 tensors, NHWC format
        # timm Swin outputs NHWC; convert to NCHW for conv projections
        return [
            proj(feat.permute(0, 3, 1, 2).contiguous())
            for proj, feat in zip(self.projections, features)
        ]

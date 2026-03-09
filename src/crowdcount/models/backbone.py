"""VGG and DINOv2 backbone wrappers for DSGCNet.

Supports:
  - vgg16_bn / vgg16 (default)
  - dinov2_s / dinov2_b / dinov2_l / dinov2_g (optional, loaded via torch.hub)
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

import crowdcount.models.vgg_ as vgg_models


# ---------------------------------------------------------------------------
# VGG backbone
# ---------------------------------------------------------------------------


class BackboneBase_VGG(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_channels: int,
        name: str,
        return_interm_layers: bool,
    ):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == "vgg16_bn":
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:  # vgg16
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == "vgg16_bn":
                self.body = nn.Sequential(*features[:44])
            elif name == "vgg16":
                self.body = nn.Sequential(*features[:30])
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list) -> List[torch.Tensor]:
        out = []
        if self.return_interm_layers:
            xs = tensor_list
            for layer in [self.body1, self.body2, self.body3, self.body4]:
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    def __init__(self, name: str, return_interm_layers: bool):
        if name == "vgg16_bn":
            backbone = vgg_models.vgg16_bn(pretrained=True)
        elif name == "vgg16":
            backbone = vgg_models.vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG variant: {name}")
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


# ---------------------------------------------------------------------------
# DINOv2 backbone
# ---------------------------------------------------------------------------

_DINOV2_VARIANTS = {
    "dinov2_s": ("facebookresearch/dinov2", "dinov2_vits14", 384),
    "dinov2_b": ("facebookresearch/dinov2", "dinov2_vitb14", 768),
    "dinov2_l": ("facebookresearch/dinov2", "dinov2_vitl14", 1024),
    "dinov2_g": ("facebookresearch/dinov2", "dinov2_vitg14", 1536),
}


class BackboneDINOv2(nn.Module):
    """Thin wrapper around a DINOv2 ViT that exposes the same 4-scale interface
    expected by DSGCnet's neck (Decoder_SPD_PAFPN).

    DINOv2 outputs a single [B, C, H/14, W/14] patch grid.  We project it to
    the three channel widths that PA-FPN expects (256 / 512 / 512) via simple
    1×1 convolutions so that the rest of the network is unchanged.
    """

    def __init__(self, variant: str = "dinov2_s"):
        super().__init__()
        if variant not in _DINOV2_VARIANTS:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. Choose from {list(_DINOV2_VARIANTS)}"
            )
        repo, model_name, embed_dim = _DINOV2_VARIANTS[variant]
        self.dino = torch.hub.load(repo, model_name, pretrained=True)
        self.embed_dim = embed_dim
        self.num_channels = 256

        # Project DINOv2 features to the three scales PA-FPN expects
        self.proj3 = nn.Conv2d(embed_dim, 256, 1)  # C3 → 256
        self.proj4 = nn.Conv2d(embed_dim, 512, 1)  # C4 → 512
        self.proj5 = nn.Conv2d(embed_dim, 512, 1)  # C5 → 512
        # C1 placeholder (same as C3) so indexing [0..3] is consistent
        self.proj1 = nn.Conv2d(embed_dim, 256, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        # DINOv2 uses 14-pixel patches; adjust to multiple of 14
        H14 = (H // 14) * 14
        W14 = (W // 14) * 14
        if H14 != H or W14 != W:
            x = torch.nn.functional.interpolate(
                x, size=(H14, W14), mode="bilinear", align_corners=False
            )

        patch_tokens = self.dino.get_intermediate_layers(
            x, n=4, return_class_token=False
        )
        # Each element: [B, num_patches, embed_dim]
        h, w = H14 // 14, W14 // 14
        feats = [
            t.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2) for t in patch_tokens
        ]
        # Map to: [C1(256), C2(256), C3(256), C4(512), C5(512)]
        # We expose indices 0,1,2,3 matching VGG body1-body4 ordering
        out = [
            self.proj1(feats[0]),
            self.proj3(feats[1]),
            self.proj4(feats[2]),
            self.proj5(feats[3]),
        ]
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(cfg) -> nn.Module:
    """cfg: OmegaConf DictConfig with fields model.backbone and model.backbone_type."""
    backbone_type = getattr(cfg.model, "backbone_type", "vgg")
    backbone_name = cfg.model.backbone

    if backbone_type == "dinov2":
        return BackboneDINOv2(backbone_name)
    else:
        return Backbone_VGG(backbone_name, True)

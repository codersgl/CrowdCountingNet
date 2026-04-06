"""VGG, DINOv2 and ConvNeXt backbone wrappers for DSGCNet.

Supports:
  - vgg16_bn / vgg16 (default)
  - dinov2_s / dinov2_b / dinov2_l / dinov2_g (optional, loaded via torch.hub)
  - convnext_tiny / convnext_small / convnext_base / convnext_large (torchvision)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models as tv_models

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


class DINOv2SemanticInjector(nn.Module):
    """Takes only DINOv2's last-layer output, projects to 256ch for semantic injection.

    The DINOv2 weights are frozen; only the projection Conv and the gate
    (in DSGCnet) are trained.
    """

    def __init__(self, variant: str = "dinov2_b"):
        super().__init__()
        if variant not in _DINOV2_VARIANTS:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. Choose from {list(_DINOV2_VARIANTS)}"
            )
        repo, model_name, embed_dim = _DINOV2_VARIANTS[variant]
        self.dino = torch.hub.load(repo, model_name, pretrained=True)
        # Freeze DINOv2 — only proj will be trained
        for p in self.dino.parameters():
            p.requires_grad = False
        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        B, C, H, W = x.shape
        H14 = (H // 14) * 14
        W14 = (W // 14) * 14
        if H14 != H or W14 != W:
            x = F.interpolate(x, size=(H14, W14), mode="bilinear", align_corners=False)
        tokens = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
        h, w = H14 // 14, W14 // 14
        feat = tokens.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2)
        feat = self.proj(feat)
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(
                feat, size=target_size, mode="bilinear", align_corners=False
            )
        return feat


# ---------------------------------------------------------------------------
# ConvNeXt backbone
# ---------------------------------------------------------------------------

_CONVNEXT_VARIANTS = {
    "convnext_tiny": (
        tv_models.convnext_tiny,
        tv_models.ConvNeXt_Tiny_Weights.DEFAULT,
        [96, 192, 384, 768],
    ),
    "convnext_small": (
        tv_models.convnext_small,
        tv_models.ConvNeXt_Small_Weights.DEFAULT,
        [96, 192, 384, 768],
    ),
    "convnext_base": (
        tv_models.convnext_base,
        tv_models.ConvNeXt_Base_Weights.DEFAULT,
        [128, 256, 512, 1024],
    ),
    "convnext_large": (
        tv_models.convnext_large,
        tv_models.ConvNeXt_Large_Weights.DEFAULT,
        [192, 384, 768, 1536],
    ),
}


class BackboneConvNeXt(nn.Module):
    """ConvNeXt backbone wrapper exposing the same 4-scale interface as VGG.

    Uses ConvNeXt stages 1–3 (strides 4/8/16) to match the VGG backbone
    contract that DSGCNet expects.  Stage 4 (stride 32) is intentionally
    unused so that PA-FPN output and anchor grids stay spatially aligned:

        out[0]: 128ch  (stride 4)  — placeholder, same role as VGG body1
        out[1]: 256ch  (stride 4)  — c3
        out[2]: 512ch  (stride 8)  — c4
        out[3]: 512ch  (stride 16) — c5

    Note: ``use_msaa=True`` is not supported with ConvNeXt because MSAA
    expects four *distinct* spatial scales, while out[0] and out[1] share
    the same stride here.
    """

    def __init__(self, variant: str = "convnext_base"):
        super().__init__()
        if variant not in _CONVNEXT_VARIANTS:
            raise ValueError(
                f"Unknown ConvNeXt variant '{variant}'. "
                f"Choose from {list(_CONVNEXT_VARIANTS)}"
            )
        factory_fn, weights, channels = _CONVNEXT_VARIANTS[variant]
        backbone = factory_fn(weights=weights)

        # ConvNeXt features: 8 sub-blocks [stem, stage1, ds2, stage2, ds3, stage3, ds4, stage4].
        # We use stages 1–3 (strides 4/8/16), skipping stage 4 (stride 32).
        feats = list(backbone.features.children())
        self.stage1 = nn.Sequential(*feats[:2])  # stem + stage1 → stride 4
        self.stage2 = nn.Sequential(*feats[2:4])  # → stride 8
        self.stage3 = nn.Sequential(*feats[4:6])  # → stride 16

        self.num_channels = 256

        # 1×1 projections to match VGG / PA-FPN expected channels
        self.proj0 = nn.Conv2d(channels[0], 128, 1)  # placeholder (VGG body1 = 128ch)
        self.proj1 = nn.Conv2d(channels[0], 256, 1)  # c3 → 256
        self.proj2 = nn.Conv2d(channels[1], 512, 1)  # c4 → 512
        self.proj3 = nn.Conv2d(channels[2], 512, 1)  # c5 → 512

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s1 = self.stage1(x)  # [B, C0, H/4,  W/4]
        s2 = self.stage2(s1)  # [B, C1, H/8,  W/8]
        s3 = self.stage3(s2)  # [B, C2, H/16, W/16]
        return [
            self.proj0(s1),  # placeholder: [B, 128, H/4,  W/4]
            self.proj1(s1),  # c3:          [B, 256, H/4,  W/4]
            self.proj2(s2),  # c4:          [B, 512, H/8,  W/8]
            self.proj3(s3),  # c5:          [B, 512, H/16, W/16]
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_backbone(cfg) -> nn.Module:
    """cfg: OmegaConf DictConfig with fields model.backbone and model.backbone_type."""
    backbone_type = getattr(cfg.model, "backbone_type", "vgg")
    backbone_name = cfg.model.backbone

    if backbone_type == "dinov2":
        return BackboneDINOv2(backbone_name)
    elif backbone_type == "convnext":
        return BackboneConvNeXt(backbone_name)
    else:
        return Backbone_VGG(backbone_name, True)

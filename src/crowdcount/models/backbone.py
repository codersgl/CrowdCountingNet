"""VGG, DINOv2, ConvNeXt and CLIP backbone wrappers for DSGCNet.

Supports:
  - vgg16_bn / vgg16 (default)
  - dinov2_s / dinov2_b / dinov2_l / dinov2_g (optional, loaded via torch.hub)
  - convnext_tiny / convnext_small / convnext_base / convnext_large (torchvision)
  - ViT-B-16 / convnext_base_w (OpenCLIP, loaded via open_clip)
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


class DepthBackbone_VGG(BackboneBase_VGG):
    """VGG backbone adapted for single-channel depth input.

    The first Conv2d layer is replaced with a 1-channel version.  When
    ``pretrained=True`` the original 3-channel kernel weights are averaged
    across the input-channel dimension so that the pretrained statistics
    transfer meaningfully to the depth domain.
    """

    def __init__(
        self,
        name: str = "vgg16_bn",
        pretrained: bool = True,
        frozen_stages: int = 0,
    ):
        if name == "vgg16_bn":
            backbone = vgg_models.vgg16_bn(pretrained=pretrained)
        elif name == "vgg16":
            backbone = vgg_models.vgg16(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported VGG variant for depth branch: {name}")

        # --- Adapt first Conv layer: 3ch → 1ch ---
        old_conv: nn.Conv2d = backbone.features[0]  # type: ignore[assignment]
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,  # type: ignore[arg-type]
            stride=old_conv.stride,  # type: ignore[arg-type]
            padding=old_conv.padding,  # type: ignore[arg-type]
            bias=old_conv.bias is not None,
        )
        if pretrained:
            # Average RGB kernel weights → 1-channel kernel
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        backbone.features[0] = new_conv

        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers=True)

        # Optionally freeze early stages for transfer learning
        if frozen_stages >= 1:
            for p in self.body1.parameters():
                p.requires_grad = False
        if frozen_stages >= 2:
            for p in self.body2.parameters():
                p.requires_grad = False
        if frozen_stages >= 3:
            for p in self.body3.parameters():
                p.requires_grad = False
        if frozen_stages >= 4:
            for p in self.body4.parameters():
                p.requires_grad = False


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
# CLIP backbone (OpenCLIP)
# ---------------------------------------------------------------------------


def _detect_clip_arch(visual) -> str:
    """Detect whether an OpenCLIP visual encoder is ViT- or ConvNeXt-based."""
    if hasattr(visual, "transformer"):
        return "vit"
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "stages"):
        return "convnext"
    if hasattr(visual, "stem") and hasattr(visual, "stages"):
        return "convnext"
    raise ValueError(
        f"Cannot detect CLIP visual architecture. "
        f"Expected ViT (with .transformer) or ConvNeXt (with .stem + .stages). "
        f"Got type: {type(visual).__name__}"
    )


# Default pretrained tags for CLIP models. When ``pretrained=True`` the
# corresponding tag is passed to ``open_clip.create_model`` so users don't
# need to know model-specific tag strings.
_CLIP_DEFAULT_PRETRAINED = {
    "ViT-B-16": "openai",
    "convnext_base_w": "laion2b_s13b_b82k",
}

# Per-model normalization stats (mean, std) used during CLIP pretraining.
# These should be used by the data loader instead of ImageNet defaults.
_CLIP_NORM_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # OpenAI CLIP: mean/std from the original paper / open_clip source
    "openai": (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    # LAION-2B default (same as openai stats in open_clip)
    "laion2b_s13b_b82k": (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    "laion2b_s34b_b88k": (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
}


class BackboneCLIP(nn.Module):
    """CLIP backbone wrapper exposing the 4-scale interface expected by DSGCNet.

    Uses OpenCLIP (``open_clip``) to load pretrained CLIP vision encoders.
    All CLIP weights are frozen; only the 1x1 projection convolutions are
    trainable.

    ViT path (e.g. ViT-B-16)
    ------------------------
    Transformer blocks are split into incremental groups.  Intermediate
    features are extracted at group boundaries, projected via 1x1 convolutions
    to the expected channel widths, then upsampled to form a multi-scale
    feature pyramid:

        out[0]: 128ch, stride patch_size//8
        out[1]: 256ch, stride patch_size//4
        out[2]: 512ch, stride patch_size//2
        out[3]: 512ch, stride patch_size

    ConvNeXt path (e.g. convnext_base_w)
    ------------------------------------
    Stage features are extracted directly (strides 4/8/16) and projected to
    the expected channel widths.  Stage 3 (stride 32) is unused, matching the
    existing ``BackboneConvNeXt`` convention.
    """

    def __init__(self, name: str, pretrained: bool | str = True):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for CLIP backbone. "
                "Install it with: uv sync --extra dev  or  pip install open_clip_torch"
            )

        self._name = name

        # Resolve pretrained tag: True → default tag, False/None → no weights
        if pretrained is True:
            pretrained_tag = _CLIP_DEFAULT_PRETRAINED.get(name)
        elif pretrained is False or pretrained is None:
            pretrained_tag = None
        else:
            pretrained_tag = pretrained

        model = open_clip.create_model(name, pretrained=pretrained_tag)
        self.visual = model.visual
        self._arch = _detect_clip_arch(self.visual)

        # Freeze all CLIP weights -- only projection layers are trained
        for p in self.visual.parameters():
            p.requires_grad = False

        # Expose the correct normalization stats for the data loader
        tag = pretrained_tag or "openai"
        if tag in _CLIP_NORM_STATS:
            _mean, _std = _CLIP_NORM_STATS[tag]
        else:
            # Fall back to the most common CLIP stats
            _mean, _std = _CLIP_NORM_STATS["openai"]
        self._norm_mean: tuple[float, ...] = _mean
        self._norm_std: tuple[float, ...] = _std

        if self._arch == "vit":
            self._init_vit()
        else:
            self._init_convnext()

    # ------------------------------------------------------------------
    # ViT initialisation
    # ------------------------------------------------------------------

    def _init_vit(self) -> None:
        embed_dim = self.visual.conv1.out_channels
        patch_size = self.visual.conv1.kernel_size[0]
        depth = len(self.visual.transformer.resblocks)

        self._embed_dim = embed_dim
        self._patch_size = patch_size

        # Incremental block grouping (matching DINOv2-style extraction)
        d = depth
        self._output_indices = {max(1, d // 12), max(1, d // 4), max(1, d // 2), d}

        # 1x1 projections to match the expected channel contract
        self.proj0 = nn.Conv2d(embed_dim, 128, 1)  # placeholder
        self.proj1 = nn.Conv2d(embed_dim, 256, 1)  # c3
        self.proj2 = nn.Conv2d(embed_dim, 512, 1)  # c4
        self.proj3 = nn.Conv2d(embed_dim, 512, 1)  # c5

    # ------------------------------------------------------------------
    # ConvNeXt initialisation
    # ------------------------------------------------------------------

    def _init_convnext(self) -> None:
        # open_clip may wrap the ConvNeXt in a .trunk or expose it directly
        trunk = getattr(self.visual, "trunk", self.visual)

        stem = trunk.stem
        stages = trunk.stages

        # Probe channel dimensions by running a dummy forward.
        # s1 = stage0 output (stride 4), s2 = stage1 output (stride 8),
        # s3 = stage2 output (stride 16).  We need each stage's actual
        # output channels because stages[i] may change the channel count.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 128)
            s0 = stem(dummy)
            s1 = stages[0](s0)
            c1 = s1.shape[1]  # stage0 output channels (stride 4)
            s2 = stages[1](s1)
            c2 = s2.shape[1]  # stage1 output channels (stride 8)
            s3 = stages[2](s2)
            c3 = s3.shape[1]  # stage2 output channels (stride 16)

        self._stem = stem
        self._stages = stages
        self._stage_channels = (c1, c2, c3)

        # Projections matching the VGG / PA-FPN channel contract.
        # proj0/proj1 both take s1 (stride 4); proj2 takes s2 (stride 8);
        # proj3 takes s3 (stride 16).
        self.proj0 = nn.Conv2d(c1, 128, 1)  # placeholder (stride 4)
        self.proj1 = nn.Conv2d(c1, 256, 1)  # c3 (stride 4)
        self.proj2 = nn.Conv2d(c2, 512, 1)  # c4 (stride 8)
        self.proj3 = nn.Conv2d(c3, 512, 1)  # c5 (stride 16)

    # ------------------------------------------------------------------
    # Normalization stats (for the data loader)
    # ------------------------------------------------------------------

    @property
    def norm_stats(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(mean, std)`` that matches the CLIP pretraining normalization."""
        return self._norm_mean, self._norm_std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._arch == "vit":
            return self._forward_vit(x)
        return self._forward_convnext(x)

    def _forward_vit(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, C, H, W = x.shape
        ps = self._patch_size

        # Adjust spatial dims to a multiple of patch_size
        H_adj = (H // ps) * ps
        W_adj = (W // ps) * ps
        if H_adj != H or W_adj != W:
            x = F.interpolate(
                x, size=(H_adj, W_adj), mode="bilinear", align_corners=False
            )

        # Patch embed → [B, D, h, w]
        tokens = self.visual.conv1(x)
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = tokens.reshape(B, self._embed_dim, -1).permute(0, 2, 1)  # [B, N, D]

        # Class token + position embedding
        cls_embed = getattr(self.visual, "class_embedding", None)
        if cls_embed is None:
            cls_embed = getattr(self.visual, "cls_token")
        if cls_embed.ndim == 1:
            cls_embed = cls_embed.view(1, 1, -1)
        tokens = torch.cat(
            [cls_embed.expand(B, -1, -1).to(tokens.dtype), tokens], dim=1
        )

        pos_embed = getattr(self.visual, "positional_embedding", None)
        if pos_embed is None:
            pos_embed = getattr(self.visual, "pos_embed")
        # Interpolate position embedding when grid size differs from pretrained size
        num_patches = h * w
        pos_embed_num_patches = pos_embed.shape[0] - 1  # subtract class token
        if num_patches != pos_embed_num_patches:
            # Separate class token (row 0) from patch positions (rows 1:)
            pos_cls = pos_embed[:1, :]  # [1, D]
            pos_patch = pos_embed[1:, :]  # [N_pre, D]
            h_pre = int(pos_embed_num_patches**0.5)
            # Reshape to 2D, interpolate, reshape back
            pos_patch = pos_patch.reshape(1, h_pre, -1, self._embed_dim).permute(
                0, 3, 1, 2
            )
            pos_patch = F.interpolate(
                pos_patch, size=(h, w), mode="bicubic", align_corners=False
            )
            pos_patch = pos_patch.permute(0, 2, 3, 1).reshape(-1, self._embed_dim)
            pos_embed = torch.cat([pos_cls, pos_patch], dim=0)

        tokens = tokens + pos_embed.to(tokens.dtype)

        ln_pre = getattr(self.visual, "ln_pre", None)
        if ln_pre is not None:
            tokens = ln_pre(tokens)

        # Step through transformer blocks, collecting at output indices
        blocks = self.visual.transformer.resblocks
        feats = []
        for i, block in enumerate(blocks):
            tokens = block(tokens)
            if (i + 1) in self._output_indices:
                patch_feat = tokens[:, 1:, :]  # strip class token
                patch_feat = patch_feat.permute(0, 2, 1).reshape(
                    B, self._embed_dim, h, w
                )
                feats.append(patch_feat)

        # Project and upsample to build multi-scale pyramid
        projs = [self.proj0, self.proj1, self.proj2, self.proj3]
        up_factors = [8, 4, 2, 1]  # stride: ps/8, ps/4, ps/2, ps
        out = []
        for feat, proj, scale in zip(feats, projs, up_factors):
            feat = proj(feat)
            if scale > 1:
                feat = F.interpolate(
                    feat,
                    scale_factor=float(scale),
                    mode="bilinear",
                    align_corners=False,
                )
            out.append(feat)
        return out

    def _forward_convnext(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self._stem(x)
        s1 = self._stages[0](s0)
        s2 = self._stages[1](s1)
        s3 = self._stages[2](s2)
        return [
            self.proj0(s1),  # stride 4, placeholder
            self.proj1(s1),  # stride 4, c3
            self.proj2(s2),  # stride 8, c4
            self.proj3(s3),  # stride 16, c5
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
    elif backbone_type == "clip":
        return BackboneCLIP(backbone_name)
    else:
        return Backbone_VGG(backbone_name, True)

"""ConvNeXt and VGG backbone wrappers for MoECountNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import tempfile
from typing import Any, cast

import torch
from torch import nn
from loguru import logger



_CONVNEXT_MODEL_NAMES = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
    "convnext_large": "convnext_large.fb_in22k_ft_in1k",
}


@dataclass(frozen=True)
class BackboneOutputInfo:
    channels: tuple[int, ...]
    reductions: tuple[int, ...]
    model_name: str


class MoEConvNeXtBackbone(nn.Module):
    """Timm ConvNeXt wrapper exposing stride-8 and stride-16 features."""

    def __init__(
        self,
        arch: str = "convnext_tiny",
        model_name: str | None = None,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        out_indices: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        num_levels = len(out_indices)
        if num_levels not in (2, 3):
            raise ValueError(f"out_indices must contain 2 or 3 feature levels, got {num_levels}")
        resolved_name = model_name or _CONVNEXT_MODEL_NAMES.get(arch)
        if resolved_name is None:
            choices = ", ".join(sorted(_CONVNEXT_MODEL_NAMES))
            raise ValueError(f"Unknown ConvNeXt arch '{arch}'. Choose from: {choices}")

        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for MoECountNet ConvNeXt backbones") from exc

        self.body = timm.create_model(
            resolved_name,
            pretrained=pretrained and not pretrained_path,
            features_only=True,
            out_indices=out_indices,
        )
        if pretrained_path:
            loaded, skipped = load_local_convnext_weights(self.body, pretrained_path)
            logger.info(
                "Loaded local ConvNeXt weights from {} "
                "(matched={}, skipped={})",
                pretrained_path,
                loaded,
                skipped,
            )
        feature_info = cast(Any, self.body.feature_info)
        channels = tuple(int(value) for value in feature_info.channels())
        reductions = tuple(int(value) for value in feature_info.reduction())
        if len(channels) != num_levels or len(reductions) != num_levels:
            raise RuntimeError(
                f"MoEConvNeXtBackbone expected {num_levels} feature maps, "
                f"got channels={channels}, reductions={reductions}"
            )
        self.output_info = BackboneOutputInfo(
            channels=channels,
            reductions=reductions,
            model_name=resolved_name,
        )

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self.output_info.channels

    @property
    def out_reductions(self) -> tuple[int, ...]:
        return self.output_info.reductions

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.body(images)
        num_levels = len(self.output_info.channels)
        if len(features) != num_levels:
            raise RuntimeError(f"Expected {num_levels} ConvNeXt features, got {len(features)}")
        result: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(["c2", "c3", "c4"][:num_levels]):
            result[key] = features[idx]
        return result


def _load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return cast(dict[str, torch.Tensor], checkpoint)


def _load_state_dict_from_local_path(pretrained_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(pretrained_path)
    if not path.exists():
        raise FileNotFoundError(f"Local pretrained_path does not exist: {path}")

    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            candidates = [
                member
                for ext in (".safetensors", ".pth", ".pt", ".bin")
                for member in members
                if Path(member.name).suffix == ext
            ]
            if not candidates:
                raise ValueError(f"No supported weight file found inside {path}")
            member = candidates[0]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read {member.name} from {path}")
            with tempfile.NamedTemporaryFile(suffix=Path(member.name).suffix) as tmp:
                tmp.write(extracted.read())
                tmp.flush()
                return _load_tensor_file(Path(tmp.name))

    return _load_tensor_file(path)


def convert_convnext_state_dict_for_features_only(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Map raw timm ConvNeXt keys to FeatureListNet keys and filter heads."""
    converted: dict[str, torch.Tensor] = {}
    skipped = 0
    for raw_key, value in state_dict.items():
        key = raw_key
        for prefix in ("module.", "model.", "body."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        key = key.replace("stem.0.", "stem_0.").replace("stem.1.", "stem_1.")
        for stage_index in range(4):
            key = key.replace(f"stages.{stage_index}.", f"stages_{stage_index}.")
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            converted[key] = value
        else:
            skipped += 1
    return converted, skipped


class MoEVGGBackbone(nn.Module):
    """VGG16-BN backbone wrapper for MoECountNet, exposing stride-8/16/32 features.

    Maps VGG's body2/body3/body4 to ``c2``/``c3``/``c4`` dict keys so the
    MoECountNet neck can consume them identically to ConvNeXt features.
    """

    def __init__(
        self,
        vgg_name: str = "vgg16_bn",
        pretrained: bool = True,
        out_levels: int = 3,
    ) -> None:
        super().__init__()
        if out_levels not in (2, 3):
            raise ValueError(f"out_levels must be 2 or 3, got {out_levels}")

        from crowdcount.models import vgg_ as vgg_models
        from crowdcount.models.backbone import BackboneBase_VGG

        backbone = vgg_models.vgg16_bn(pretrained=pretrained)
        self.body = BackboneBase_VGG(backbone, 256, vgg_name, return_interm_layers=True)

        if vgg_name in ("vgg16_bn", "vgg16"):
            self._out_channels = (256, 512, 512)[:out_levels]
        else:
            self._out_channels = (128, 256, 512)[:out_levels]
        self._out_levels = out_levels

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._out_channels

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.body(images)  # [body1, body2, body3, body4]
        result: dict[str, torch.Tensor] = {}
        keys = ["c2", "c3", "c4"][: self._out_levels]
        for idx, key in enumerate(keys):
            result[key] = features[idx + 1]  # skip body1 (stride-4)
        return result


def load_local_convnext_weights(model: nn.Module, pretrained_path: str | Path) -> tuple[int, int]:
    state_dict = _load_state_dict_from_local_path(pretrained_path)
    model_state = model.state_dict()
    converted, skipped = convert_convnext_state_dict_for_features_only(
        state_dict,
        model_state,
    )
    if not converted:
        raise ValueError(
            f"No matching ConvNeXt weights were found in local checkpoint {pretrained_path}"
        )
    model.load_state_dict(converted, strict=False)
    return len(converted), skipped

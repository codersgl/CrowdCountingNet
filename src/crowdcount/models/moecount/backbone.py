"""ConvNeXt backbone wrapper for MoECountNet."""

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
    channels: tuple[int, int]
    reductions: tuple[int, int]
    model_name: str


class MoEConvNeXtBackbone(nn.Module):
    """Timm ConvNeXt wrapper exposing stride-8 and stride-16 features."""

    def __init__(
        self,
        arch: str = "convnext_tiny",
        model_name: str | None = None,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        out_indices: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__()
        if len(out_indices) != 2:
            raise ValueError("out_indices must contain exactly two feature levels")
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
        if len(channels) != 2 or len(reductions) != 2:
            raise RuntimeError(
                "MoEConvNeXtBackbone expected two feature maps, "
                f"got channels={channels}, reductions={reductions}"
            )
        self.output_info = BackboneOutputInfo(
            channels=channels,
            reductions=reductions,
            model_name=resolved_name,
        )

    @property
    def out_channels(self) -> tuple[int, int]:
        return self.output_info.channels

    @property
    def out_reductions(self) -> tuple[int, int]:
        return self.output_info.reductions

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.body(images)
        if len(features) != 2:
            raise RuntimeError(f"Expected two ConvNeXt features, got {len(features)}")
        return {"c2": features[0], "c3": features[1]}


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

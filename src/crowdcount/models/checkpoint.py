"""Checkpoint compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


StateDict = Mapping[str, torch.Tensor]


def extract_model_state_dict(checkpoint: object) -> StateDict:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a model state_dict mapping")
    return state_dict  # type: ignore[return-value]


def migrate_legacy_state_dict_for_model(
    state_dict: StateDict,
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Rename legacy checkpoint keys to match the current model when safe."""
    target_keys = set(model.state_dict().keys())
    migrated = dict(state_dict)
    legacy_prefix = "pa.acdr."
    current_prefix = "neck_acdr."
    remapped_acdr = 0

    for old_key, value in list(state_dict.items()):
        if not old_key.startswith(legacy_prefix):
            continue
        new_key = current_prefix + old_key[len(legacy_prefix) :]
        if new_key not in target_keys:
            continue
        if new_key not in migrated:
            migrated[new_key] = value
        migrated.pop(old_key, None)
        remapped_acdr += 1

    return migrated, {"pa_acdr_to_neck_acdr": remapped_acdr}


def load_model_state_dict(
    model: nn.Module,
    checkpoint: object,
    *,
    strict: bool = True,
    logger: Any | None = None,
) -> Any:
    state_dict = extract_model_state_dict(checkpoint)
    state_dict, migration_counts = migrate_legacy_state_dict_for_model(
        state_dict, model
    )
    remapped_acdr = migration_counts["pa_acdr_to_neck_acdr"]
    if logger is not None and remapped_acdr:
        logger.info(
            f"Remapped {remapped_acdr} legacy DAP ACDR checkpoint keys "
            "from pa.acdr.* to neck_acdr.*"
        )
    return model.load_state_dict(state_dict, strict=strict)
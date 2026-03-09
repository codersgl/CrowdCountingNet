"""Model factory: build_model(cfg) returns (model, criterion) or model."""

from __future__ import annotations

from omegaconf import DictConfig

from crowdcount.models.backbone import build_backbone
from crowdcount.models.criterion import SetCriterion_Crowd
from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.matcher import build_matcher_crowd


def build_model(cfg: DictConfig, training: bool = False):
    """
    Args:
        cfg: OmegaConf DictConfig (hydra config).
        training: if True returns (model, criterion); else model only.
    """
    num_classes = 1
    backbone = build_backbone(cfg)
    model = DSGCnet(backbone, row=cfg.model.row, line=cfg.model.line)

    if not training:
        return model

    weight_dict = {"loss_ce": 1, "loss_points": cfg.model.point_loss_coef}
    losses = ["labels", "points"]
    matcher = build_matcher_crowd(cfg)
    criterion = SetCriterion_Crowd(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.model.eos_coef,
        losses=losses,
    )
    return model, criterion


__all__ = ["build_model", "DSGCnet", "SetCriterion_Crowd"]

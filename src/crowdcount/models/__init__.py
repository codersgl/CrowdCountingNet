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
    model = DSGCnet(
        backbone,
        row=cfg.model.row,
        line=cfg.model.line,
        fusion_mode=getattr(cfg.model, "fusion_mode", "gcn"),
        use_gm=getattr(cfg.model, "use_gm", False),
        gm_input_dim=getattr(cfg.model, "gm_input_dim", 256),
        gm_hidden_dim=getattr(cfg.model, "gm_hidden_dim", 128),
        gm_spatial=getattr(cfg.model, "gm_spatial", True),
        use_msaa=getattr(cfg.model, "use_msaa", False),
        msaa_in_channels=getattr(cfg.model, "msaa_in_channels", 1280),
        msaa_reduction=getattr(cfg.model, "msaa_reduction", 4),
        moe_cfg=getattr(cfg.model, "moe", None),
        use_depth=getattr(cfg.model, "use_depth", False),
        depth_cfg=getattr(cfg.model, "depth", None),
        use_depth_geo=getattr(cfg.model, "use_depth_geo", False),
        depth_geo_cfg=getattr(cfg.model, "depth_geo", None),
        gcn_adaptive=getattr(cfg.model, "gcn_adaptive", False),
        gcn_k=getattr(cfg.model, "gcn_k", 4),
        gcn_k_min=getattr(cfg.model, "gcn_k_min", 2),
        gcn_k_max=getattr(cfg.model, "gcn_k_max", 8),
        gcn_density_scale=getattr(cfg.model, "gcn_density_scale", 4.0),
        gcn_sim_threshold=getattr(cfg.model, "gcn_sim_threshold", 0.5),
        cfg=cfg,  # Pass config for multi-scale density prediction
        use_dcn=getattr(cfg.model, "use_dcn", False),
    )

    if not training:
        return model

    weight_dict = {
        "loss_ce": 1,
        "loss_points": cfg.model.point_loss_coef,
        "loss_count": getattr(cfg.model, "count_loss_coef", 0.0),
    }
    losses = ["labels", "points", "count"]
    matcher = build_matcher_crowd(cfg)

    # Focal loss config
    use_focal = getattr(cfg.model, "use_focal_loss", False)
    focal_cfg = getattr(cfg.model, "focal_loss", None)
    focal_alpha = float(getattr(focal_cfg, "alpha", 0.25)) if focal_cfg else 0.25
    focal_gamma = float(getattr(focal_cfg, "gamma", 2.0)) if focal_cfg else 2.0

    criterion = SetCriterion_Crowd(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.model.eos_coef,
        losses=losses,
        use_focal_loss=use_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    return model, criterion


__all__ = ["build_model", "DSGCnet", "SetCriterion_Crowd"]

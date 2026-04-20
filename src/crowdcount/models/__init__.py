"""Model factory: build_model(cfg) returns (model, criterion) or model."""

from __future__ import annotations

from omegaconf import DictConfig

from crowdcount.models.backbone import build_backbone
from crowdcount.models.criterion import SetCriterion_Crowd
from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.models.matcher import build_matcher_crowd
from crowdcount.models.uncertainty_loss import UncertaintyWeighter


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
        msaa_variant=getattr(cfg.model, "msaa_variant", "legacy"),
        moe_cfg=getattr(cfg.model, "moe", None),
        graph_attn_moe_cfg=getattr(cfg.model, "graph_attn_moe", None),
        mamba_moe_cfg=getattr(cfg.model, "mamba_moe", None),
        sdd_moe_cfg=getattr(cfg.model, "sdd_moe", None),
        use_depth=getattr(cfg.model, "use_depth", False),
        depth_cfg=getattr(cfg.model, "depth", None),
        use_depth_geo=getattr(cfg.model, "use_depth_geo", False),
        depth_geo_cfg=getattr(cfg.model, "depth_geo", None),
        use_depth_dual_vgg=getattr(cfg.model, "use_depth_dual_vgg", False),
        depth_dual_vgg_cfg=getattr(cfg.model, "depth_dual_vgg", None),
        use_depth_attn=getattr(cfg.model, "use_depth_attn", False),
        depth_attn_cfg=getattr(cfg.model, "depth_attn", None),
        gcn_adaptive=getattr(cfg.model, "gcn_adaptive", False),
        gcn_k=getattr(cfg.model, "gcn_k", 4),
        gcn_k_min=getattr(cfg.model, "gcn_k_min", 2),
        gcn_k_max=getattr(cfg.model, "gcn_k_max", 8),
        gcn_density_scale=getattr(cfg.model, "gcn_density_scale", 4.0),
        gcn_sim_threshold=getattr(cfg.model, "gcn_sim_threshold", 0.5),
        cfg=cfg,  # Pass config for multi-scale density prediction
        use_dcn=getattr(cfg.model, "use_dcn", False),
        use_refine=getattr(cfg.model, "use_refine", False),
        refine_cfg=getattr(cfg.model, "refine", None),
        gcn_mode=getattr(cfg.model, "gcn_mode", "fixed"),
        gcn_num_supernodes=getattr(cfg.model, "gcn_num_supernodes", 8),
        gcn_supernode_heads=getattr(cfg.model, "gcn_supernode_heads", 4),
        use_freq_head=getattr(cfg.model, "use_freq_head", False),
        freq_head_kernel=getattr(cfg.model, "freq_head_kernel", 3),
        use_density_attention=getattr(cfg.model, "use_density_attention", False),
        density_attention_mode=getattr(cfg.model, "density_attention_mode", "sigmoid"),
        density_attention_pre_gcn=getattr(
            cfg.model, "density_attention_pre_gcn", False
        ),
        density_attention_hidden=int(
            getattr(cfg.model, "density_attention_hidden", 32)
        ),
        density_attention_base=float(getattr(cfg.model, "density_attention_base", 0.5)),
        use_subpix_refine=getattr(cfg.model, "use_subpix_refine", False),
        subpix_refine_cfg=getattr(cfg.model, "subpix_refine", None),
        use_uncertainty=getattr(cfg.model, "use_uncertainty", False),
        uncertainty_scale=float(getattr(cfg.model, "uncertainty_scale", 6.0)),
        gcn_aniso=getattr(cfg.model, "gcn_aniso", False),
        gcn_conv_type=getattr(cfg.model, "gcn_conv_type", "gcn"),
        use_fg_branch=getattr(cfg.model, "use_fg_branch", False),
        fg_branch_base=float(getattr(cfg.model, "fg_branch_base", 0.5)),
        fg_branch_scale=float(getattr(cfg.model, "fg_branch_scale", 0.5)),
        fpn_attention=getattr(cfg.model, "fpn_attention", False),
        use_msca_decoder=getattr(cfg.model, "use_msca_decoder", False),
        msca_num_heads=int(getattr(cfg.model, "msca_num_heads", 8)),
        msca_num_blocks=int(getattr(cfg.model, "msca_num_blocks", 2)),
        use_decoupled_head=getattr(cfg.model, "use_decoupled_head", False),
        use_msca_neck=getattr(cfg.model, "use_msca_neck", False),
        use_rccformer_neck=getattr(cfg.model, "use_rccformer_neck", False),
        rccformer_deab_blocks=int(getattr(cfg.model, "rccformer_deab_blocks", 2)),
        use_dap_neck=getattr(cfg.model, "use_dap_neck", False),
        dap_neck_cfg=getattr(cfg.model, "dap_neck", None),
        use_deep_head=getattr(cfg.model, "use_deep_head", False),
    )

    if not training:
        return model

    weight_dict = {
        "loss_ce": 1,
        "loss_points": cfg.model.point_loss_coef,
        "loss_count": getattr(cfg.model, "count_loss_coef", 0.0),
        "loss_refine": float(getattr(cfg, "refine_loss_weight", 0.0))
        if getattr(cfg.model, "use_refine", False)
        else 0.0,
        "loss_consistency": float(getattr(cfg.model, "consistency_loss_coef", 0.0)),
    }
    losses = ["labels", "points", "count", "consistency"]
    if getattr(cfg.model, "use_refine", False):
        losses.append("refine")
    matcher = build_matcher_crowd(cfg)

    # Focal loss config
    use_focal = getattr(cfg.model, "use_focal_loss", False)
    focal_cfg = getattr(cfg.model, "focal_loss", None)
    focal_alpha = float(getattr(focal_cfg, "alpha", 0.25)) if focal_cfg else 0.25
    focal_gamma = float(getattr(focal_cfg, "gamma", 2.0)) if focal_cfg else 2.0

    # Quality Focal Loss config
    use_qfl = getattr(cfg.model, "use_qfl", False)
    qfl_cfg = getattr(cfg.model, "qfl", None)
    qfl_beta = float(getattr(qfl_cfg, "beta", 2.0)) if qfl_cfg else 2.0
    qfl_sigma = float(getattr(qfl_cfg, "sigma", 10.0)) if qfl_cfg else 10.0

    criterion = SetCriterion_Crowd(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.model.eos_coef,
        losses=losses,
        use_focal_loss=use_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        use_uncertainty_weighting=getattr(cfg.model, "use_uncertainty", False),
        uncertainty_boost=float(getattr(cfg.model, "uncertainty_boost", 2.0)),
        use_qfl=use_qfl,
        qfl_beta=qfl_beta,
        qfl_sigma=qfl_sigma,
    )

    # Uncertainty weighting (Kendall et al. 2018)
    uw_cfg = getattr(cfg, "uncertainty_weighting", None)
    uncertainty_weighter: UncertaintyWeighter | None = None
    if uw_cfg is not None and bool(getattr(uw_cfg, "enabled", False)):
        uncertainty_weighter = UncertaintyWeighter(
            init_log_var_den=float(getattr(uw_cfg, "init_log_var_den", 3.91)),
            init_log_var_ce=float(getattr(uw_cfg, "init_log_var_ce", -0.693)),
            init_log_var_reg=float(getattr(uw_cfg, "init_log_var_reg", 8.52)),
        )

    return model, criterion, uncertainty_weighter


__all__ = ["build_model", "DSGCnet", "SetCriterion_Crowd", "UncertaintyWeighter"]

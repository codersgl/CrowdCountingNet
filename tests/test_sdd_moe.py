from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn

from crowdcount.models.dsgcnet import DSGCnet
from crowdcount.plugins.sdd_moe import (
    BackgroundAwareGate,
    LargeScaleExpert,
    MidScaleExpert,
    OcclusionReasoningExpert,
    SDDMoE,
    ScaleDecoupledRouter,
    TinyScaleExpert,
)


class DummyBackbone(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch_size, _, height, width = x.shape
        device = x.device
        return [
            torch.randn(batch_size, 128, height // 4, width // 4, device=device),
            torch.randn(batch_size, 256, height // 4, width // 4, device=device),
            torch.randn(batch_size, 512, height // 8, width // 8, device=device),
            torch.randn(batch_size, 512, height // 16, width // 16, device=device),
        ]


def _make_targets(
    batch_size: int = 2, n_points: int = 5
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "point": torch.rand(n_points, 2) * 128,
            "labels": torch.ones(n_points, dtype=torch.long),
        }
        for _ in range(batch_size)
    ]


def test_background_aware_gate_shape_and_grad() -> None:
    gate = BackgroundAwareGate(in_channels=32)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    mask = gate(x, threshold=0.3)
    assert mask.shape == (2, 1, 8, 8)
    assert torch.all((mask == 0) | (mask == 1))
    mask.mean().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_scale_decoupled_router_respects_fg_mask() -> None:
    router = ScaleDecoupledRouter(in_channels=32)
    x = torch.randn(2, 32, 8, 8)
    fg_mask = torch.ones(2, 1, 8, 8)
    fg_mask[:, :, :4] = 0
    route_map, scale_map, density_map = router(x, fg_mask, training=True)
    assert route_map.shape == (2, 4, 8, 8)
    assert scale_map.shape == (2, 1, 8, 8)
    assert density_map.shape == (2, 1, 8, 8)
    assert torch.allclose(route_map.sum(dim=1), fg_mask.squeeze(1), atol=1e-5)


def test_large_scale_expert_shape() -> None:
    expert = LargeScaleExpert(in_channels=32, rates=(1, 2, 3, 4)).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_tiny_scale_expert_shape() -> None:
    expert = TinyScaleExpert(in_channels=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_mid_scale_expert_shape() -> None:
    expert = MidScaleExpert(in_channels=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_occlusion_reasoning_expert_shape() -> None:
    expert = OcclusionReasoningExpert(in_channels=32, num_heads=2).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_occlusion_reasoning_expert_reduced_dim() -> None:
    expert = OcclusionReasoningExpert(in_channels=32, attn_dim=16, num_heads=2).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y = expert(x)
    assert y.shape == x.shape


def test_sdd_moe_forward_train_mode() -> None:
    cfg = OmegaConf.create(
        {
            "fg_threshold": 0.3,
            "gumbel_temperature": 1.0,
            "gumbel_temp_min": 0.3,
            "aspp_rates": [1, 2, 3, 4],
            "self_attn_heads": 2,
            "lambda_balance": 0.01,
            "lambda_scale": 0.1,
            "lambda_ssim": 0.1,
            "ssim_window_size": 7,
        }
    )
    moe = SDDMoE(in_channels=32, cfg=cfg).train()
    x = torch.randn(2, 32, 16, 16)
    density_hint = torch.rand(2, 1, 16, 16)
    gt_density = torch.rand(2, 1, 16, 16)
    fused, aux_losses, weights = moe(
        x,
        density_hint=density_hint,
        targets=_make_targets(batch_size=2),
        gt_density=gt_density,
        image_size=(128, 128),
        training=True,
    )
    assert fused.shape == x.shape
    assert weights.shape == (2, 4, 16, 16)
    assert "total_aux" in aux_losses
    assert "l_balance" in aux_losses
    assert "l_scale" in aux_losses
    assert "l_ssim" in aux_losses


def test_sdd_moe_backward() -> None:
    moe = SDDMoE(in_channels=32).train()
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    density_hint = torch.rand(2, 1, 16, 16)
    gt_density = torch.rand(2, 1, 16, 16)
    fused, aux_losses, _ = moe(
        x,
        density_hint=density_hint,
        targets=_make_targets(batch_size=2),
        gt_density=gt_density,
        image_size=(128, 128),
        training=True,
    )
    loss = fused.mean() + aux_losses["total_aux"]
    loss.backward()
    router_grads = [p.grad for p in moe.router.parameters() if p.grad is not None]
    assert x.grad is not None
    assert len(router_grads) > 0
    assert any(g.abs().sum().item() > 0 for g in router_grads)


def test_sdd_moe_eval_mode_returns_hard_routes() -> None:
    moe = SDDMoE(in_channels=32).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        fused, aux_losses, weights = moe(x, training=False)
    assert fused.shape == x.shape
    assert aux_losses == {}
    assert torch.all((weights == 0) | (weights == 1))


def test_dsgcnet_sdd_moe_smoke() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "vgg16_bn",
                "backbone_type": "vgg",
                "row": 2,
                "line": 2,
                "fusion_mode": "sdd_moe",
                "use_gm": False,
                "use_msaa": False,
                "use_density_attention": False,
                "density_attention_mode": "sigmoid",
                "density_attention_pre_gcn": False,
                "density_attention_hidden": 32,
                "density_attention_base": 0.5,
                "use_refine": False,
                "use_subpix_refine": False,
                "use_uncertainty": False,
                "gcn_aniso": False,
                "use_fg_branch": False,
                "fpn_attention": False,
                "use_msca_decoder": False,
                "use_decoupled_head": False,
                "use_msca_neck": False,
                "use_rccformer_neck": False,
                "use_dap_neck": False,
                "use_deep_head": False,
                "sdd_moe": {
                    "fg_threshold": 0.3,
                    "gumbel_temperature": 1.0,
                    "gumbel_temp_min": 0.3,
                    "aspp_rates": [1, 2, 3, 4],
                    "self_attn_heads": 2,
                    "self_attn_dim": 128,
                    "lambda_balance": 0.01,
                    "lambda_scale": 0.1,
                    "lambda_ssim": 0.1,
                    "ssim_window_size": 7,
                },
            }
        }
    )
    model = DSGCnet(
        DummyBackbone(), fusion_mode="sdd_moe", cfg=cfg, sdd_moe_cfg=cfg.model.sdd_moe
    ).train()
    samples = torch.randn(2, 3, 128, 128)
    gt_density = torch.rand(2, 1, 16, 16)
    outputs = model(
        samples,
        targets=_make_targets(batch_size=2),
        gt_density=gt_density,
    )
    assert "pred_logits" in outputs
    assert "pred_points" in outputs
    assert "moe_aux_total" in outputs
    assert outputs["moe_weights"].shape[1] == 4

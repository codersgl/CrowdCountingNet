"""Unit tests for the MambaMoE fusion plugin."""

from __future__ import annotations

import torch

from crowdcount.plugins.mamba_moe import (
    BiDirectionalChannelSSM,
    MambaMoEBalanceLoss,
    MambaMoEFusion,
    MoEBlock,
    MoMEB,
    SingleScanSSM,
    SpatialMoELayer,
    SpatialMoERouter,
)


def test_single_scan_ssm_shape_preserved() -> None:
    module = SingleScanSSM(d_model=32, low_dim=8).eval()
    x = torch.randn(2, 8, 8, 32)
    with torch.no_grad():
        out = module(x, direction=0)
    assert out.shape == x.shape


def test_router_output_shape() -> None:
    router = SpatialMoERouter(input_dim=32, num_experts=4, use_density_hint=True).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        logits = router(x, density_hint=density)
    assert logits.shape == (2, 4)


def test_moe_layer_directions() -> None:
    moe = SpatialMoELayer(input_dim=32, num_experts=4)
    assert moe._generate_directions() == (0, 1, 2, 3)


def test_moe_layer_topk_inference() -> None:
    moe = SpatialMoELayer(input_dim=32, num_experts=4, top_k=2).eval()
    x = torch.randn(2, 8, 8, 32)
    with torch.no_grad():
        out, weights = moe(x, training=False)
    assert out.shape == x.shape
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all((weights > 0).sum(dim=-1) <= 2)


def test_bidirectional_channel_ssm_shape() -> None:
    module = BiDirectionalChannelSSM(d_model=32, expand=1.0, d_spectral=64).eval()
    x = torch.randn(2, 8, 8, 32)
    with torch.no_grad():
        out = module(x)
    assert out.shape == x.shape


def test_bidirectional_channel_ssm_adaptive_pool() -> None:
    """d_spectral != H*W triggers adaptive pooling/interpolation."""
    module = BiDirectionalChannelSSM(d_model=32, expand=1.0, d_spectral=16).eval()
    x = torch.randn(1, 6, 6, 32)  # H*W=36 != d_spectral=16
    with torch.no_grad():
        out = module(x)
    assert out.shape == x.shape


def test_mamba_moe_fusion_training_mode() -> None:
    """training=None should fall back to module.training state."""
    module = MambaMoEFusion(input_dim=16, num_blocks=1, num_experts=4, mlp_hidden=32)
    x = torch.randn(1, 16, 4, 4)
    # In train mode (default), training=None should use soft routing
    module.train()
    out_train, _, w_train = module(x, training=None)
    assert out_train.shape == x.shape
    # All experts should have non-zero weight (soft routing)
    assert torch.all(w_train > 0)


def test_mamba_moe_gradient_flow() -> None:
    module = MambaMoEFusion(
        input_dim=16, num_blocks=1, num_experts=4, mlp_hidden=32, d_spectral=16
    )
    module.train()
    x = torch.randn(1, 16, 4, 4, requires_grad=True)
    out, aux, _ = module(x, training=True)
    loss = out.sum() + aux["total_aux"]
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_num_experts_validation() -> None:
    """num_experts > 4 should raise ValueError."""
    try:
        SpatialMoELayer(input_dim=32, num_experts=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_moe_block_split_merge() -> None:
    block = MoEBlock(input_dim=32, num_experts=4, top_k=2).eval()
    x = torch.randn(2, 8, 8, 32)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        out, weights = block(x, density_hint=density, training=False)
    assert out.shape == x.shape
    assert weights.shape == (2, 4)


def test_momeb_residual_shape() -> None:
    block = MoMEB(input_dim=32, num_experts=4, mlp_hidden=64, drop_path=0.0).eval()
    x = torch.randn(2, 8, 8, 32)
    with torch.no_grad():
        out, weights = block(x, training=False)
    assert out.shape == x.shape
    assert weights.shape == (2, 4)


def test_mamba_moe_fusion_interface() -> None:
    module = MambaMoEFusion(
        input_dim=32,
        num_experts=4,
        num_blocks=1,
        mlp_hidden=64,
        use_density_hint=True,
    ).eval()
    x = torch.randn(2, 32, 8, 8)
    density = torch.rand(2, 1, 8, 8)
    with torch.no_grad():
        out, aux, weights = module(x, density_hint=density, training=False)
    assert out.shape == x.shape
    assert weights.shape == (2, 4)
    assert "l_balance" in aux
    assert "total_aux" in aux


def test_balance_loss_outputs() -> None:
    loss_fn = MambaMoEBalanceLoss(lambda_balance=0.05)
    weights = torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1]])
    aux = loss_fn(weights)
    assert aux["l_balance"].ndim == 0
    assert aux["total_aux"].ndim == 0
    assert aux["total_aux"].item() >= 0.0


def test_cpu_fallback_runs() -> None:
    module = MambaMoEFusion(
        input_dim=16, num_blocks=1, num_experts=4, mlp_hidden=32, d_spectral=16
    ).eval()
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        out, aux, weights = module(x, density_hint=None, training=False)
    assert out.shape == x.shape
    assert weights.shape == (1, 4)
    assert aux["total_aux"].item() >= 0.0

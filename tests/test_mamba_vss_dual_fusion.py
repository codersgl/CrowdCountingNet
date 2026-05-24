from __future__ import annotations

import torch
import pytest

from crowdcount.plugins.mamba_vss_dual_fusion import MambaVSSDualFusion


def _build_module(
    *,
    fusion_spatial: bool = True,
    use_density_hint: bool = True,
) -> MambaVSSDualFusion:
    return MambaVSSDualFusion(
        in_channels=16,
        density_embed_dim=8,
        d_state=4,
        d_conv=3,
        mlp_ratio=1.0,
        vss_low_dim=4,
        num_vss_blocks=1,
        num_moe_blocks=1,
        num_experts=2,
        top_k=1,
        expand=1.0,
        d_spectral=8,
        mlp_hidden=32,
        drop_path=0.0,
        lambda_balance=0.01,
        use_density_hint=use_density_hint,
        fusion_spatial=fusion_spatial,
        gate_init=1e-3,
    )


def test_mamba_vss_dual_output_shape_and_aux() -> None:
    module = _build_module().eval()
    features = torch.randn(2, 16, 4, 4)
    density = torch.rand(2, 1, 4, 4)

    with torch.no_grad():
        out, aux, weights = module(features, density, training=False)

    assert out.shape == features.shape
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all((weights > 0).sum(dim=-1) <= 1)
    assert "l_balance" in aux
    assert "total_aux" in aux
    assert "fusion_entropy" in aux
    assert aux["fusion_entropy"].ndim == 0


def test_spatial_fusion_weights_sum_to_one() -> None:
    module = _build_module(fusion_spatial=True).eval()
    features = torch.randn(1, 16, 4, 4)
    density = torch.rand(1, 1, 4, 4)

    with torch.no_grad():
        module(features, density, training=False)

    fusion_weights = module.last_fusion_weights
    assert fusion_weights is not None
    assert fusion_weights.shape == (1, 3, 4, 4)
    assert torch.allclose(
        fusion_weights.sum(dim=1), torch.ones(1, 4, 4), atol=1e-6
    )


def test_global_fusion_weights_sum_to_one() -> None:
    module = _build_module(fusion_spatial=False).eval()
    features = torch.randn(2, 16, 4, 4)
    density = torch.rand(2, 1, 4, 4)

    with torch.no_grad():
        module(features, density, training=False)

    fusion_weights = module.last_fusion_weights
    assert fusion_weights is not None
    assert fusion_weights.shape == (2, 3)
    assert torch.allclose(fusion_weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_mamba_vss_dual_gradient_flow() -> None:
    module = _build_module()
    module.train()
    features = torch.randn(1, 16, 4, 4, requires_grad=True)
    density = torch.rand(1, 1, 4, 4)

    out, aux, _ = module(features, density, training=True)
    loss = out.mean() + aux["total_aux"]
    loss.backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0


def test_mamba_vss_dual_resizes_density_hint() -> None:
    module = _build_module().eval()
    features = torch.randn(1, 16, 4, 4)
    density = torch.rand(1, 1, 2, 2)

    with torch.no_grad():
        out, _, _ = module(features, density, training=False)

    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_mamba_vss_dual_rejects_bad_density_shape() -> None:
    module = _build_module().eval()
    features = torch.randn(1, 16, 4, 4)
    density = torch.rand(1, 2, 4, 4)

    with pytest.raises(ValueError, match="density_hint"):
        module(features, density, training=False)
"""Tests for CLIP text-prompt density guidance."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from crowdcount.plugins.clip_prompt_density import CLIPPromptDensityGuide


def _prompt_bank(num_prompts: int = 4, dim: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.randn(num_prompts, dim)
    is_positive = torch.tensor([True, True, False, False])
    return embeddings, is_positive


class TinyBackbone(nn.Module):
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        return [
            torch.zeros(B, 128, H // 2, W // 2),
            torch.zeros(B, 256, H // 4, W // 4),
            torch.zeros(B, 512, H // 8, W // 8),
            torch.zeros(B, 512, H // 16, W // 16),
        ]


class FakePromptGuide(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        prob = torch.sigmoid(logits)
        return x, {
            "foreground_logits": logits,
            "foreground_prob": prob,
            "positive_weight": prob,
            "negative_weight": 1.0 - prob,
            "strength": torch.tensor(0.0, device=x.device),
        }


def test_clip_prompt_density_shape_and_identity_init() -> None:
    embeddings, is_positive = _prompt_bank()
    module = CLIPPromptDensityGuide(
        feature_channels=256,
        text_embeddings=embeddings,
        prompt_is_positive=is_positive,
    )
    x = torch.randn(2, 256, 16, 16)

    with torch.no_grad():
        out, info = module(x)

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)
    assert info["foreground_logits"].shape == (2, 1, 16, 16)
    weights_sum = info["positive_weight"] + info["negative_weight"]
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6)


def test_clip_prompt_density_grad_reaches_trainable_adapter() -> None:
    embeddings, is_positive = _prompt_bank()
    module = CLIPPromptDensityGuide(
        feature_channels=256,
        text_embeddings=embeddings,
        prompt_is_positive=is_positive,
    )
    x = torch.randn(2, 256, 8, 8, requires_grad=True)

    out, _ = module(x)
    out.square().mean().backward()

    final = module.film[2]
    assert isinstance(final, nn.Conv2d)
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0
    assert module.prompt_embeddings.grad is None


def test_clip_prompt_density_requires_positive_and_negative_prompts() -> None:
    embeddings = torch.randn(3, 16)
    is_positive = torch.tensor([True, True, True])

    with pytest.raises(ValueError, match="positive and negative"):
        CLIPPromptDensityGuide(
            text_embeddings=embeddings,
            prompt_is_positive=is_positive,
        )


def test_dsgcnet_clip_prompt_density_forward_uses_density_head(monkeypatch) -> None:
    import crowdcount.models.dsgcnet as dsgcnet_mod

    monkeypatch.setattr(dsgcnet_mod, "CLIPPromptDensityGuide", FakePromptGuide)
    cfg = OmegaConf.create({"apply_to": "density_only", "debug": True})
    model = dsgcnet_mod.DSGCnet(
        TinyBackbone(),
        row=2,
        line=2,
        use_clip_prompt_density=True,
        clip_prompt_density_cfg=cfg,
    ).eval()

    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))

    assert out["density_out"].shape == (2, 1, 16, 16)
    assert out["clip_prompt_foreground_logits"].shape == (2, 1, 16, 16)
    assert "clip_prompt_density_stats" in out


def test_dsgcnet_clip_prompt_density_rejects_msca_decoder() -> None:
    import crowdcount.models.dsgcnet as dsgcnet_mod

    with pytest.raises(ValueError, match="use_clip_prompt_density"):
        dsgcnet_mod.DSGCnet(
            TinyBackbone(),
            use_clip_prompt_density=True,
            use_msca_decoder=True,
        )

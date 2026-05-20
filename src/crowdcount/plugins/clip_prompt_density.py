"""CLIP text-prompt guidance for density regression features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_POSITIVE_PROMPTS = (
    "dense crowd of people",
    "sparse pedestrians in a public scene",
    "human heads in a crowd",
    "many people gathered together",
)
_DEFAULT_NEGATIVE_PROMPTS = (
    "empty background without people",
    "road building tree background",
    "scene with no pedestrians",
    "non human background regions",
)


def _as_prompt_tuple(
    prompts: Sequence[str] | None,
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    if prompts is None:
        return defaults
    values = tuple(str(prompt) for prompt in prompts if str(prompt).strip())
    return values or defaults


def _resolve_pretrained_tag(
    clip_model: str, pretrained: bool | str | None
) -> str | None:
    if pretrained is True:
        from crowdcount.models.backbone import _CLIP_DEFAULT_PRETRAINED

        return _CLIP_DEFAULT_PRETRAINED.get(clip_model, "openai")
    if pretrained is False or pretrained is None:
        return None
    return str(pretrained)


def _encode_prompts_with_open_clip(
    clip_model: str,
    pretrained: bool | str | None,
    prompts: tuple[str, ...],
) -> torch.Tensor:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for CLIP prompt density guidance. "
            "Install dependencies with: uv sync --extra dev"
        ) from exc

    pretrained_tag = _resolve_pretrained_tag(clip_model, pretrained)
    model = open_clip.create_model(clip_model, pretrained=pretrained_tag)
    tokenizer = open_clip.get_tokenizer(clip_model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokens = tokenizer(list(prompts))
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1)


class CLIPPromptDensityGuide(nn.Module):
    """Condition density-head features on a frozen CLIP text prompt bank.

    The module keeps CLIP text embeddings frozen as buffers. It projects each
    spatial feature into the text embedding space, computes prompt affinities,
    builds a local text context, then applies identity-safe residual FiLM.
    """

    def __init__(
        self,
        feature_channels: int = 256,
        clip_model: str = "ViT-B-16",
        pretrained: bool | str | None = True,
        positive_prompts: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        temperature: float = 0.07,
        hidden_dim: int = 128,
        max_delta: float = 0.5,
        strength_init: float = 1e-3,
        text_embeddings: torch.Tensor | None = None,
        prompt_is_positive: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")

        positive = _as_prompt_tuple(positive_prompts, _DEFAULT_POSITIVE_PROMPTS)
        negative = _as_prompt_tuple(negative_prompts, _DEFAULT_NEGATIVE_PROMPTS)
        prompts = positive + negative
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError(
                "At least one positive and one negative prompt are required"
            )

        if text_embeddings is None:
            embeddings = _encode_prompts_with_open_clip(clip_model, pretrained, prompts)
            is_positive = torch.tensor(
                [True] * len(positive) + [False] * len(negative), dtype=torch.bool
            )
        else:
            if text_embeddings.ndim != 2:
                raise ValueError("text_embeddings must have shape [num_prompts, dim]")
            embeddings = F.normalize(text_embeddings.detach().float(), dim=-1)
            if prompt_is_positive is None:
                is_positive = torch.tensor(
                    [True] * len(positive) + [False] * len(negative), dtype=torch.bool
                )
            else:
                is_positive = prompt_is_positive.detach().bool().flatten()
            if is_positive.numel() != embeddings.shape[0]:
                raise ValueError(
                    "prompt_is_positive length must match text_embeddings first dimension"
                )

        if embeddings.shape[0] < 2:
            raise ValueError("At least two prompts are required")
        if not is_positive.any().item() or is_positive.all().item():
            raise ValueError("Prompt bank must contain positive and negative prompts")

        prompt_dim = int(embeddings.shape[1])
        self.feature_channels = int(feature_channels)
        self.prompt_dim = prompt_dim
        self.temperature = float(temperature)
        self.max_delta = float(max_delta)

        self.register_buffer("prompt_embeddings", embeddings, persistent=True)
        self.register_buffer("prompt_is_positive", is_positive, persistent=True)

        self.feature_proj = nn.Conv2d(
            feature_channels, prompt_dim, kernel_size=1, bias=False
        )
        self.film = nn.Sequential(
            nn.Conv2d(prompt_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_channels * 2, kernel_size=1),
        )
        final = self.film[2]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.strength = nn.Parameter(torch.tensor(float(strength_init)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected feature map [B,C,H,W], got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} channels, got {x.shape[1]}"
            )

        prompts = F.normalize(self.prompt_embeddings.float(), dim=-1)
        projected = self.feature_proj(x)
        projected_norm = F.normalize(projected.float(), dim=1)
        logits = torch.einsum("bdhw,pd->bphw", projected_norm, prompts)
        logits = logits / self.temperature
        prompt_weights = torch.softmax(logits, dim=1)
        context = torch.einsum("bphw,pd->bdhw", prompt_weights, prompts).to(
            dtype=x.dtype
        )

        film = self.film(context)
        gamma, beta = film.chunk(2, dim=1)
        pos_mask = self.prompt_is_positive
        neg_mask = ~pos_mask
        pos_logits = logits[:, pos_mask].logsumexp(dim=1, keepdim=True)
        neg_logits = logits[:, neg_mask].logsumexp(dim=1, keepdim=True)
        foreground_logits = (pos_logits - neg_logits).to(dtype=x.dtype)
        foreground_prob = torch.sigmoid(foreground_logits)

        spatial_weight = 0.5 + foreground_prob
        delta = (x * torch.tanh(gamma) + torch.tanh(beta)) * spatial_weight
        guided = (
            x + torch.tanh(self.strength).to(dtype=x.dtype) * self.max_delta * delta
        )

        info = {
            "foreground_logits": foreground_logits,
            "foreground_prob": foreground_prob,
            "positive_weight": prompt_weights[:, pos_mask]
            .sum(dim=1, keepdim=True)
            .to(dtype=x.dtype),
            "negative_weight": prompt_weights[:, neg_mask]
            .sum(dim=1, keepdim=True)
            .to(dtype=x.dtype),
            "strength": torch.tanh(self.strength).detach(),
        }
        return guided, info

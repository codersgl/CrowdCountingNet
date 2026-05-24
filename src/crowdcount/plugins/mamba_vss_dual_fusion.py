from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.plugins.mamba_moe import MambaMoEFusion, SingleScanSSM


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _ConvFFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = max(channels, int(channels * mlp_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensityGuidedVSSBlock(nn.Module):
    """VSSBlock_Spa-style spatial SSM block for the density-guided stream."""

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 3,
        mlp_ratio: float = 2.0,
        vss_low_dim: int | None = None,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        low_dim = int(vss_low_dim) if vss_low_dim is not None else int(8 * mlp_ratio)
        low_dim = max(2, min(channels, low_dim))

        self.norm_scan = nn.LayerNorm(channels)
        self.scans = nn.ModuleList(
            [
                SingleScanSSM(
                    d_model=channels,
                    low_dim=low_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for _ in range(4)
            ]
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(channels), channels),
            nn.GELU(),
            _ChannelAttention(channels),
        )
        channel_hidden = max(1, channels // 8)
        spatial_hidden = max(1, channels // 16)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channel_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channel_hidden, channels, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(channels, spatial_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spatial_hidden, 1, kernel_size=1),
        )

        self.norm_conv = nn.LayerNorm(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = _ConvFFN(channels, mlp_ratio=mlp_ratio)

        init = torch.tensor(float(gate_init), dtype=torch.float32)
        self.mix_scale = nn.Parameter(init.clone())
        self.conv_scale = nn.Parameter(init.clone())
        self.ffn_scale = nn.Parameter(init.clone())

    @staticmethod
    def _to_channels_last(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")

        scan_input = self.norm_scan(self._to_channels_last(x))
        scan_out = torch.stack(
            [scan(scan_input, direction=direction) for direction, scan in enumerate(self.scans)],
            dim=0,
        ).mean(dim=0)
        scan_out = self._to_channels_first(scan_out)

        local_out = self.local_branch(x)
        channel_gate = torch.sigmoid(self.channel_interaction(local_out))
        spatial_gate = torch.sigmoid(self.spatial_interaction(scan_out))
        mixed = scan_out * channel_gate + local_out * spatial_gate
        x = x + self.mix_scale.tanh() * mixed

        conv_in = self._to_channels_first(self.norm_conv(self._to_channels_last(x)))
        x = x + self.conv_scale.tanh() * self.conv(conv_in)

        ffn_in = self._to_channels_first(self.norm_ffn(self._to_channels_last(x)))
        x = x + self.ffn_scale.tanh() * self.ffn(ffn_in)
        return x


class DensityGuidedVSSStream(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        density_embed_dim: int = 64,
        num_blocks: int = 1,
        d_state: int = 16,
        d_conv: int = 3,
        mlp_ratio: float = 2.0,
        vss_low_dim: int | None = None,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        groups = _valid_group_count(density_embed_dim)
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, density_embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, density_embed_dim),
            nn.SiLU(),
            nn.Conv2d(
                density_embed_dim,
                density_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, density_embed_dim),
            nn.SiLU(),
        )
        self.density_proj = nn.Conv2d(density_embed_dim, in_channels, kernel_size=1)
        self.density_scale = nn.Parameter(torch.tensor(float(gate_init)))
        self.blocks = nn.ModuleList(
            [
                DensityGuidedVSSBlock(
                    channels=in_channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    mlp_ratio=mlp_ratio,
                    vss_low_dim=vss_low_dim,
                    gate_init=gate_init,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        density_hint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if density_hint.shape[-2:] != features.shape[-2:]:
            density_hint = F.interpolate(
                density_hint,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        density_hint = density_hint.to(device=features.device, dtype=features.dtype)
        density_embed = self.density_encoder(density_hint)
        hidden = features + self.density_scale.tanh() * self.density_proj(density_embed)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden, density_embed


class DensityConditionedTriFusion(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        density_embed_dim: int = 64,
        spatial: bool = True,
    ) -> None:
        super().__init__()
        self.spatial = bool(spatial)
        combined_channels = 3 * in_channels + density_embed_dim
        hidden_channels = max(16, in_channels // 2)
        if self.spatial:
            self.weight_proj = nn.Sequential(
                nn.Conv2d(combined_channels, hidden_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 3, kernel_size=1),
            )
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.weight_mlp = nn.Sequential(
                nn.Linear(combined_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 3),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.spatial:
            last_proj = self.weight_proj[-1]
            if isinstance(last_proj, nn.Conv2d):
                nn.init.zeros_(last_proj.weight)
                if last_proj.bias is not None:
                    nn.init.zeros_(last_proj.bias)
        else:
            last_linear = self.weight_mlp[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        features: torch.Tensor,
        density_stream: torch.Tensor,
        feature_stream: torch.Tensor,
        density_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if density_embed.shape[-2:] != features.shape[-2:]:
            density_embed = F.interpolate(
                density_embed,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.spatial:
            combined = torch.cat(
                [features, density_stream, feature_stream, density_embed], dim=1
            )
            weights = torch.softmax(self.weight_proj(combined), dim=1)
            fused = (
                weights[:, 0:1] * features
                + weights[:, 1:2] * density_stream
                + weights[:, 2:3] * feature_stream
            )
            return fused, weights

        pooled = torch.cat(
            [
                self.gap(features),
                self.gap(density_stream),
                self.gap(feature_stream),
                self.gap(density_embed),
            ],
            dim=1,
        ).flatten(1)
        weights = torch.softmax(self.weight_mlp(pooled), dim=1)
        fused = (
            weights[:, 0].view(-1, 1, 1, 1) * features
            + weights[:, 1].view(-1, 1, 1, 1) * density_stream
            + weights[:, 2].view(-1, 1, 1, 1) * feature_stream
        )
        return fused, weights


class MambaVSSDualFusion(nn.Module):
    """Mamba/VSS replacement for DSGCNet's dual-stream GCN/GAT fusion."""

    def __init__(
        self,
        in_channels: int = 256,
        density_embed_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 3,
        mlp_ratio: float = 2.0,
        vss_low_dim: int | None = None,
        num_vss_blocks: int = 1,
        num_moe_blocks: int = 1,
        num_experts: int = 4,
        top_k: int = 2,
        lr_space: str = "exp",
        expand: float = 2.0,
        d_spectral: int = 256,
        mlp_hidden: int = 256,
        drop_path: float = 0.1,
        lambda_balance: float = 0.01,
        use_density_hint: bool = True,
        fusion_spatial: bool = True,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.use_density_hint = bool(use_density_hint)
        self.density_stream = DensityGuidedVSSStream(
            in_channels=in_channels,
            density_embed_dim=density_embed_dim,
            num_blocks=num_vss_blocks,
            d_state=d_state,
            d_conv=d_conv,
            mlp_ratio=mlp_ratio,
            vss_low_dim=vss_low_dim,
            gate_init=gate_init,
        )
        self.feature_stream = MambaMoEFusion(
            input_dim=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_experts=num_experts,
            top_k=top_k,
            lr_space=lr_space,
            num_blocks=num_moe_blocks,
            mlp_hidden=mlp_hidden,
            drop_path=drop_path,
            lambda_balance=lambda_balance,
            use_density_hint=use_density_hint,
            d_spectral=d_spectral,
        )
        self.fusion = DensityConditionedTriFusion(
            in_channels=in_channels,
            density_embed_dim=density_embed_dim,
            spatial=fusion_spatial,
        )
        self.last_fusion_weights: torch.Tensor | None = None

    def get_router_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for block in self.feature_stream.blocks:
            moe_block = getattr(block, "block", None)
            spatial_moe = getattr(moe_block, "spatial_moe", None)
            router = getattr(spatial_moe, "router", None)
            if isinstance(router, nn.Module):
                params.extend(router.parameters())
        return params

    @staticmethod
    def _fusion_entropy(weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 4:
            entropy = -(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8))).sum(
                dim=1
            )
            return entropy.mean()
        return -(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8))).sum(
            dim=1
        ).mean()

    def forward(
        self,
        features: torch.Tensor,
        density_hint: torch.Tensor | None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if features.dim() != 4:
            raise ValueError(
                f"features must have shape [B, C, H, W], got {tuple(features.shape)}"
            )
        if density_hint is None:
            density_hint = features.new_zeros(features.shape[0], 1, *features.shape[-2:])
        if density_hint.dim() != 4 or density_hint.shape[1] != 1:
            raise ValueError(
                "density_hint must have shape [B, 1, H, W], "
                f"got {tuple(density_hint.shape)}"
            )

        density_hint = density_hint.to(device=features.device, dtype=features.dtype)
        density_stream, density_embed = self.density_stream(features, density_hint)
        moe_density_hint = density_hint if self.use_density_hint else None
        feature_stream, aux_losses, moe_weights = self.feature_stream(
            features,
            density_hint=moe_density_hint,
            training=training,
        )
        fused, fusion_weights = self.fusion(
            features,
            density_stream,
            feature_stream,
            density_embed,
        )
        self.last_fusion_weights = fusion_weights.detach()

        aux = dict(aux_losses)
        aux["fusion_entropy"] = self._fusion_entropy(fusion_weights)
        return fused, aux, moe_weights


__all__ = [
    "DensityConditionedTriFusion",
    "DensityGuidedVSSBlock",
    "DensityGuidedVSSStream",
    "MambaVSSDualFusion",
]
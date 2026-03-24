"""Depth-RGB feature fusion via ISFM components.

Fuses depth encoder features into the RGB backbone feature maps using the
ISFM native components:
  * **MFF** (FrequencyFusinoMoudle): DWT-based frequency-domain fusion — always
    available, pure PyTorch, CPU-compatible.
  * **ISF** (ISFLayer): Mamba SSM-based spatial fusion — only available when
    ``mamba_ssm`` is installed (requires CUDA).

When ``mamba_ssm`` is not importable the module gracefully degrades to
MFF-only fusion so the model can still be initialised and tested on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from crowdcount.plugins.isfm.MFF import FrequencyFusinoMoudle

# Conditional import: ISFLayer requires mamba_ssm (CUDA)
try:
    from crowdcount.plugins.isfm.ISF import ISFLayer

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    ISFLayer = None  # type: ignore[assignment, misc]

# ISFLayer needs both mamba_ssm AND a CUDA device at runtime
_ISF_AVAILABLE = HAS_MAMBA and torch.cuda.is_available()
if not _ISF_AVAILABLE:
    logger.info(
        "ISFLayer unavailable (mamba_ssm={}, cuda={}) — using MFF-only fusion",
        HAS_MAMBA,
        torch.cuda.is_available(),
    )


class DepthFusionModule(nn.Module):
    """Fuse an RGB feature map with a depth feature map of the same spatial size.

    Uses ISFM native components: MFF (frequency-domain, always active) and
    ISFLayer (spatial Mamba SSM, active only when ``mamba_ssm`` is installed).

    Args:
        in_channels:    Number of channels of both input feature maps.
        embed_dim:      Internal ISFM embedding dimension (default 128).
        num_isf_layers: Depth of the ISFLayer block (number of FGM sub-blocks).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        num_isf_layers: int = 1,
    ) -> None:
        super().__init__()

        # Project both streams into the ISFM embedding space
        self.proj_rgb = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.proj_depth = nn.Conv2d(in_channels, embed_dim, 1, bias=False)

        # MFF: DWT frequency-domain fusion (always available, CPU-compatible)
        self.mff = FrequencyFusinoMoudle(dim=embed_dim)

        # ISF: Mamba spatial fusion (only when mamba_ssm + CUDA available)
        if _ISF_AVAILABLE:
            self.isf: ISFLayer | None = ISFLayer(
                dim=embed_dim,
                input_resolution=(16, 16),  # placeholder; actual size passed in forward
                depth=num_isf_layers,
            )
        else:
            self.isf = None

        # Back-project to original channel dim
        self.out_proj = nn.Conv2d(embed_dim, in_channels, 1, bias=False)

        # Learnable gate initialised to 0 → residual starts at identity
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """Fuse *rgb_feat* with *depth_feat* and return enhanced RGB features.

        Both inputs must be ``[B, C, H, W]`` with identical shapes.
        """
        B, _C, H, W = rgb_feat.shape

        # 1. Project to ISFM embedding space
        rgb_e = self.proj_rgb(rgb_feat)  # [B, E, H, W]
        dep_e = self.proj_depth(depth_feat)  # [B, E, H, W]

        # 2. Flatten to sequences for MFF / ISF
        rgb_seq = rgb_e.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # [B, L, E]
        dep_seq = dep_e.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # [B, L, E]

        # 3. MFF: frequency-domain fusion (always runs)
        fre_fused, lf_fuse, hf_fuse = self.mff(
            rgb_seq, dep_seq, rgb_e, dep_e, (H, W)
        )  # fre_fused: [B, E, H, W]

        # 4. ISF: spatial Mamba fusion (only on CUDA tensors)
        if self.isf is not None and rgb_feat.is_cuda:
            spa_fused_seq = self.isf(
                rgb_seq, dep_seq, lf_fuse, hf_fuse, (H, W)
            )  # [B, L, E]
            spa_fused = spa_fused_seq.permute(0, 2, 1).reshape(B, -1, H, W)
            fused = fre_fused + spa_fused
        else:
            fused = fre_fused

        # 5. Project back and apply gated residual
        out = self.out_proj(fused)  # [B, in_channels, H, W]
        return rgb_feat + self.gate.tanh() * out

"""SA-DGAT Fusion: Scale-Aware Deformable Graph Attention Network.

Orchestrates all SA-DGAT components into a single fusion module that
replaces the standard dual-stream GCN in DSGCNet. Pipeline:
    Scale Prompt → Deformable Graph Attention → Occlusion-Aware GAT
    → Cross-Scale Graph Aggregation → Output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crowdcount.plugins.sa_dgat.scale_prompt import ScalePromptEmbedding
from crowdcount.plugins.sa_dgat.deformable_graph import DeformableGraphAttention
from crowdcount.plugins.sa_dgat.occlusion_gat import OcclusionAwareGAT
from crowdcount.plugins.sa_dgat.cross_scale_graph import CrossScaleGraphAggregation


class SADGATFusion(nn.Module):
    """SA-DGAT fusion module for crowd counting.

    Integrates scale-aware node embedding, deformable graph attention,
    occlusion-aware message passing, and cross-scale graph aggregation.

    Args:
        in_channels: Feature dimension (default 256).
        num_scale_prompts: Number of learnable scale prompts.
        deformable_k: Number of deformable neighbours.
        num_heads: Attention heads for all sub-modules.
        lambda_init: Distance penalty init for deformable graph.
        mu_init: Scale matching reward init.
        local_dilations: Dilation rates for local cross-scale aggregation.
        global_dilations: Dilation rates for global cross-scale aggregation.
        num_gat_layers: Stacked GAT layers in occlusion module.
        occ_hidden: Hidden channels for occlusion predictor.
        use_depth_prior: Whether to use depth for occlusion.
        use_cross_scale: Whether to use cross-scale aggregation
            (requires multi-scale neck outputs).
        dropout: Dropout rate for attention layers.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_scale_prompts: int = 5,
        deformable_k: int = 8,
        num_heads: int = 4,
        lambda_init: float = 1.0,
        mu_init: float = 1.0,
        local_dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        global_dilations: tuple[int, ...] | list[int] = (1, 3, 6),
        num_gat_layers: int = 2,
        occ_hidden: int = 64,
        use_depth_prior: bool = False,
        use_cross_scale: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_cross_scale = use_cross_scale

        # 1. Scale-Aware Node Embedding
        self.scale_prompt = ScalePromptEmbedding(
            embed_dim=in_channels,
            num_prompts=num_scale_prompts,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 2. Deformable Graph Attention
        self.deformable_graph = DeformableGraphAttention(
            in_channels=in_channels,
            num_neighbors=deformable_k,
            num_heads=num_heads,
            lambda_init=lambda_init,
            mu_init=mu_init,
            dropout=dropout,
        )

        # 3. Occlusion-Aware GAT
        self.occlusion_gat = OcclusionAwareGAT(
            in_channels=in_channels,
            num_heads=num_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
            occ_hidden=occ_hidden,
            use_depth_prior=use_depth_prior,
        )

        # 4. Cross-Scale Graph Aggregation (optional)
        if use_cross_scale:
            self.cross_scale = CrossScaleGraphAggregation(
                in_channels=in_channels,
                local_dilations=local_dilations,
                global_dilations=global_dilations,
                dropout=dropout,
            )

        # Final projection to ensure channel consistency
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(
        self,
        features_pa: torch.Tensor,
        density: torch.Tensor,
        depth_map: torch.Tensor | None = None,
        fpn_intermediates: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            features_pa: Main fused features from PA-FPN [B, C, H, W].
            density: Density prediction [B, 1, H, W] (used as context).
            depth_map: Optional depth map [B, 1, H_img, W_img].
            fpn_intermediates: Optional (P3, P4, P5) from neck for cross-scale.
                P3: [B, C, H*2, W*2], P4: [B, C, H, W], P5: [B, C, H/2, W/2]

        Returns:
            Tuple of:
                - Fused features [B, C, H, W].
                - Auxiliary outputs dict (occlusion_map, scale_weights).
        """
        aux = {}

        # Step 1: Scale-aware conditioning
        x, scale_weights = self.scale_prompt(features_pa)
        aux["scale_weights"] = scale_weights

        # Step 2: Deformable graph attention (dynamic neighbour discovery)
        # Density prior guides offset prediction (high-density → search further)
        x_graph = self.deformable_graph(x, scale_weights=scale_weights, density=density)

        # Step 3: Occlusion-aware message passing
        # Re-sample neighbors from x_graph using cached coordinates from step 2.
        # This ensures neighbor features are in the same representation space as
        # the node features (both post-deformable-attention), which is critical
        # for the Q-K attention alignment in OcclusionAwareGAT.
        B, C, H, W = x_graph.shape
        sample_coords = self.deformable_graph._cached_sample_coords  # [B, N, K, 2]
        N, K = sample_coords.shape[1], sample_coords.shape[2]
        flat_coords = sample_coords.reshape(B, N * K, 1, 2)
        neighbor_feats = F.grid_sample(
            x_graph,
            flat_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # [B, C, N*K, 1]
        neighbor_feats = neighbor_feats.squeeze(-1).permute(0, 2, 1).reshape(B, N, K, C)

        x_occ, occ_map = self.occlusion_gat(
            x_graph,
            neighbor_feats=neighbor_feats,
            depth=depth_map,
            sample_coords=sample_coords,
        )
        aux["occlusion_map"] = occ_map

        # Step 4: Cross-scale aggregation (if intermediates available)
        if self.use_cross_scale and fpn_intermediates is not None:
            p3, p4, p5 = fpn_intermediates
            # Replace P4 with our processed features
            x_cross = self.cross_scale(p3, x_occ, p5)
        else:
            x_cross = x_occ

        # Final projection + residual
        out = self.out_proj(x_cross) + features_pa

        return out, aux

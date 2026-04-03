"""Graph-Aware MoE: two-expert mixture with density-graph attention bias.

Two experts with complementary inductive biases:
  - **LocalExpert**: multi-scale depthwise convolution with density-gated
    receptive field selection.  Excels at fine-grained discrimination in
    high-density regions.
  - **GraphAttentionExpert**: multi-head self-attention whose QK^T scores
    receive an additive bias derived from the density similarity graph.
    Captures long-range context guided by crowd density topology.

A lightweight **CoarseDensityRouter** produces per-pixel soft weights
conditioned on the predicted density map, deciding how much each spatial
location should rely on local vs. global processing.

Integration
-----------
Drop-in replacement for the GCN fusion stage in DSGCNet.  Activated by
setting ``fusion_mode: graph_attn_moe`` in the model config.

Input / output contract::

    features_pa  [B, C, H, W]  (C=256, from PA-FPN)
    density_out  [B, 1, H, W]  (from Density_pred, typically detached)
    ──►  feature_fl  [B, C, H, W]
         aux_losses  dict
         weights     [B, 2, H, W]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Local Expert
# ---------------------------------------------------------------------------


class LocalExpert(nn.Module):
    """Multi-scale depthwise convolution with density-gated receptive field.

    Three parallel branches with increasing dilation rates provide compact,
    medium, and wide receptive fields.  A lightweight gate driven by the
    density map steers each spatial location toward the branch best suited
    to its local crowd density (small RF for dense, large for sparse).

    Args:
        input_dim: Feature channel count (256 in DSGCNet).
        kernel_sizes: Depthwise kernel sizes for the parallel branches.
        expansion: Channel expansion factor for the internal representation.
        use_density_gate: If True, density map modulates branch mixing.
    """

    def __init__(
        self,
        input_dim: int = 256,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        expansion: int = 4,
        use_density_gate: bool = True,
    ) -> None:
        super().__init__()
        ex_ch = input_dim * expansion
        self.use_density_gate = use_density_gate

        self.expand = nn.Sequential(
            nn.Conv2d(input_dim, ex_ch, 1, bias=False),
            nn.BatchNorm2d(ex_ch),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        ex_ch, ex_ch, k, padding=k // 2, groups=ex_ch, bias=False
                    ),
                    nn.BatchNorm2d(ex_ch),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(ex_ch, input_dim, 1, bias=False),
            nn.BatchNorm2d(input_dim),
        )

        # Density-conditioned gate: [B, 1, H, W] → [B, num_branches, H, W]
        num_branches = len(kernel_sizes)
        if use_density_gate:
            self.density_gate = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, num_branches, 1),
            )
        else:
            self.density_gate = None

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features.
            density: [B, 1, H, W] density map (detached).

        Returns:
            [B, C, H, W] locally-enhanced features with residual.
        """
        expanded = self.expand(x)

        # Compute all branch outputs
        branch_outputs = torch.stack(
            [branch(expanded) for branch in self.branches], dim=1
        )  # [B, num_branches, ex_ch, H, W]

        if self.density_gate is not None and density is not None:
            if density.shape[-2:] != x.shape[-2:]:
                density = F.interpolate(
                    density, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            gate = self.density_gate(density)  # [B, num_branches, H, W]
            gate = F.softmax(gate, dim=1).unsqueeze(2)  # [B, num_branches, 1, H, W]
            fused = (branch_outputs * gate).sum(dim=1)  # [B, ex_ch, H, W]
        else:
            # Equal weighting fallback
            fused = branch_outputs.mean(dim=1)

        return self.project(fused) + x  # residual


# ---------------------------------------------------------------------------
# Graph Attention Expert
# ---------------------------------------------------------------------------


class GraphAttentionExpert(nn.Module):
    """Multi-head self-attention with density-graph additive bias.

    Standard MHSA on the H×W spatial tokens, augmented with a soft density
    similarity matrix as attention bias.  This injects graph-structural
    inductive bias (which pixels are in similar density contexts) without
    requiring explicit graph construction, k-NN, or PyG dependencies.

    For 16×16 feature maps (N=256 tokens), the O(N²) attention cost is
    negligible (~4M FLOPs per head).

    Args:
        input_dim: Feature channel count.
        num_heads: Number of attention heads.
        use_density_bias: Whether to add density similarity as attention bias.
        density_bias_scale: Initial scale for the density bias (learnable).
        attn_dropout: Dropout rate on attention weights.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_heads: int = 4,
        use_density_bias: bool = True,
        density_bias_scale: float = 1.0,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.use_density_bias = use_density_bias

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
        )

        # Zero-initialised output gate → starts as identity (residual only)
        self.gate = nn.Parameter(torch.zeros(1))

        # Learnable scale for density bias
        if use_density_bias:
            self.density_bias_scale = nn.Parameter(
                torch.tensor(density_bias_scale, dtype=torch.float32)
            )
        else:
            self.density_bias_scale = None

    def _compute_density_bias(
        self, density: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Compute [B, 1, N, N] additive attention bias from density map.

        bias[i,j] = -|d_i - d_j|  (negative → similar densities get higher attention)
        Scaled by a learnable parameter.

        Args:
            density: [B, 1, H_d, W_d] density map.
            H, W: Target spatial dims of feature map.

        Returns:
            [B, 1, N, N] attention bias (N = H*W).
        """
        if density.shape[-2:] != (H, W):
            density = F.interpolate(
                density, size=(H, W), mode="bilinear", align_corners=False
            )
        # Flatten to [B, N]
        d_flat = density.flatten(2).squeeze(1)  # [B, N]

        # Pairwise absolute difference: [B, N, N]
        diff = torch.abs(d_flat.unsqueeze(2) - d_flat.unsqueeze(1))

        # Negative distance → similar pixels get positive bias
        assert self.density_bias_scale is not None
        bias = -diff * self.density_bias_scale  # [B, N, N]
        return bias.unsqueeze(1)  # [B, 1, N, N] for head broadcast

    def forward(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features.
            density: [B, 1, H, W] density map (typically detached).

        Returns:
            [B, C, H, W] globally-enhanced features with gated residual.
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims to token sequence
        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)  # [B, N, C]

        # Pre-norm attention
        normed = self.norm1(tokens)

        # QKV projection
        qkv = self.qkv(normed).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)  # each [B, heads, N, d]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, heads, N, N]

        # Add density-graph bias
        if self.use_density_bias and density is not None:
            bias = self._compute_density_bias(density, H, W)  # [B, 1, N, N]
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        out = self.out_proj(out)

        # Gated residual (gate=0 at init → identity)
        g = self.gate.tanh()
        tokens = tokens + g * out

        # FFN with pre-norm and gated residual (same gate keeps identity-init)
        tokens = tokens + g * self.ffn(self.norm2(tokens))

        return tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Coarse Density Router
# ---------------------------------------------------------------------------


class CoarseDensityRouter(nn.Module):
    """Patch-level soft router conditioned on density map.

    Produces [B, 2, H, W] routing weights via coarse-grid pooling,
    lightweight scoring, and bilinear upsampling.  All experts are always
    activated (soft routing); weights sum to 1 along expert dim.

    Args:
        input_dim: Feature channel count.
        num_experts: Number of experts to route (default 2).
        grid_stride: Spatial stride for coarse routing patches.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 2,
        grid_stride: int = 4,
    ) -> None:
        super().__init__()
        self.grid_stride = grid_stride
        self.num_experts = num_experts
        # +1 for density channel
        in_ch = input_dim + 1
        self.score_net = nn.Sequential(
            nn.Conv2d(in_ch, input_dim // 4, kernel_size=1),
            nn.BatchNorm2d(input_dim // 4),
            nn.GELU(),
            nn.Conv2d(input_dim // 4, num_experts, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] features.
            density: [B, 1, H, W] density map (typically detached).

        Returns:
            [B, num_experts, H, W] soft routing weights (sum=1 along dim=1).
        """
        if density.shape[-2:] != x.shape[-2:]:
            density = F.interpolate(
                density, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        inp = torch.cat([x, density], dim=1)  # [B, C+1, H, W]

        r = self.grid_stride
        H, W = inp.shape[-2:]
        if H >= r and W >= r:
            coarse = F.avg_pool2d(inp, kernel_size=r, stride=r)
            scores = self.score_net(coarse)
            scores = F.interpolate(
                scores, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            scores = self.score_net(inp)

        return F.softmax(scores, dim=1)  # [B, num_experts, H, W]


# ---------------------------------------------------------------------------
# Balance Loss (reusable from compact MoE, kept self-contained here)
# ---------------------------------------------------------------------------


class GraphMoEBalanceLoss(nn.Module):
    """Entropy-based expert balance loss for two-expert routing."""

    def __init__(self, lambda_balance: float = 0.01) -> None:
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(self, expert_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            expert_weights: [B, num_experts, H, W]

        Returns:
            dict with ``l_balance`` and ``total_aux``.
        """
        if expert_weights.dim() == 4:
            usage = expert_weights.mean(dim=(0, 2, 3))  # [num_experts]
        else:
            usage = expert_weights.mean(dim=0)

        p = torch.clamp(usage, min=0.0)
        p = p / (p.sum() + 1e-8)
        num_experts = p.size(0)
        max_entropy = math.log(float(num_experts))
        current_entropy = -(p * torch.log(p + 1e-8)).sum()
        l_balance = max_entropy - current_entropy

        total = self.lambda_balance * l_balance
        return {"l_balance": l_balance, "total_aux": total}


# ---------------------------------------------------------------------------
# Top-level Graph-Aware MoE
# ---------------------------------------------------------------------------


class GraphAwareMoE(nn.Module):
    """Two-expert Mixture-of-Experts with density-graph attention bias.

    Orchestrates a LocalExpert (multi-scale conv) and a GraphAttentionExpert
    (MHSA + density bias) with a CoarseDensityRouter.

    Args:
        input_dim: Feature channel count (256).
        num_heads: MHSA heads for the global expert.
        use_density_bias: Add density similarity as attention bias.
        density_bias_scale: Initial learnable scale for density bias.
        attn_dropout: Attention dropout in global expert.
        local_kernels: Kernel sizes for LocalExpert branches.
        local_expansion: Channel expansion factor for LocalExpert.
        local_use_density_gate: Density-gated RF selection in LocalExpert.
        grid_stride: Coarse routing patch stride.
        lambda_balance: Expert balance loss weight.
        router_detach_density: Whether to detach density for router input.
        disable_graph_bias: Ablation: disable graph bias in attention.
        disable_local_expert: Ablation: disable local expert entirely.
        disable_global_expert: Ablation: disable global expert entirely.
    """

    def __init__(
        self,
        input_dim: int = 256,
        # GraphAttentionExpert
        num_heads: int = 4,
        use_density_bias: bool = True,
        density_bias_scale: float = 1.0,
        attn_dropout: float = 0.1,
        # LocalExpert
        local_kernels: tuple[int, ...] = (1, 3, 5),
        local_expansion: int = 4,
        local_use_density_gate: bool = True,
        # Router
        grid_stride: int = 4,
        lambda_balance: float = 0.01,
        router_detach_density: bool = True,
        # Ablation switches
        disable_graph_bias: bool = False,
        disable_local_expert: bool = False,
        disable_global_expert: bool = False,
    ) -> None:
        super().__init__()
        if disable_local_expert and disable_global_expert:
            raise ValueError(
                "Cannot disable both local and global experts simultaneously"
            )

        self.router_detach_density = router_detach_density
        self.disable_local = disable_local_expert
        self.disable_global = disable_global_expert

        # Experts
        self.local_expert: LocalExpert | None = (
            None
            if disable_local_expert
            else LocalExpert(
                input_dim=input_dim,
                kernel_sizes=local_kernels,
                expansion=local_expansion,
                use_density_gate=local_use_density_gate,
            )
        )

        effective_density_bias = use_density_bias and not disable_graph_bias
        self.global_expert: GraphAttentionExpert | None = (
            None
            if disable_global_expert
            else GraphAttentionExpert(
                input_dim=input_dim,
                num_heads=num_heads,
                use_density_bias=effective_density_bias,
                density_bias_scale=density_bias_scale,
                attn_dropout=attn_dropout,
            )
        )

        # Router (only needed when both experts are active)
        num_active = int(not disable_local_expert) + int(not disable_global_expert)
        if num_active == 2:
            self.router: CoarseDensityRouter | None = CoarseDensityRouter(
                input_dim=input_dim,
                num_experts=2,
                grid_stride=grid_stride,
            )
        else:
            self.router = None

        # Aux loss
        self.aux_loss = GraphMoEBalanceLoss(lambda_balance=lambda_balance)

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] features from PA-FPN.
            density: [B, 1, H, W] density prediction.
            training: Whether in training mode.

        Returns:
            feature_fl: [B, C, H, W] fused expert output.
            aux_losses: dict (non-empty only when training).
            weights: [B, 2, H, W] routing weights for logging.
        """
        density_for_experts = density.detach()
        density_for_router = density.detach() if self.router_detach_density else density

        # --- Single-expert ablation paths ---
        if self.disable_global:
            assert self.local_expert is not None
            out = self.local_expert(x, density_for_experts)
            B, _, H, W = x.shape
            dummy_weights = torch.ones(B, 2, H, W, device=x.device)
            dummy_weights[:, 1] = 0.0
            return out, {}, dummy_weights

        if self.disable_local:
            assert self.global_expert is not None
            out = self.global_expert(x, density_for_experts)
            B, _, H, W = x.shape
            dummy_weights = torch.ones(B, 2, H, W, device=x.device)
            dummy_weights[:, 0] = 0.0
            return out, {}, dummy_weights

        # --- Normal two-expert path ---
        assert self.local_expert is not None
        assert self.global_expert is not None
        assert self.router is not None

        local_out = self.local_expert(x, density_for_experts)
        global_out = self.global_expert(x, density_for_experts)

        weights = self.router(x, density_for_router)  # [B, 2, H, W]

        feature_fl = (
            local_out * weights[:, 0:1] + global_out * weights[:, 1:2]
        )  # [B, C, H, W]

        aux_losses: dict[str, torch.Tensor] = {}
        if training:
            aux_losses = self.aux_loss(weights)

        return feature_fl, aux_losses, weights

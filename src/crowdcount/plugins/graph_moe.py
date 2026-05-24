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
# Window partition helpers (local-first enhancement)
# ---------------------------------------------------------------------------


def _window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Partition [B, C, H, W] into non-overlapping windows.

    Returns:
        windows: [B * nH * nW, C, ws, ws]
        orig_hw: (H, W) before padding.
        padded_hw: (Hp, Wp) after padding.
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    nH, nW = Hp // window_size, Wp // window_size
    x = x.view(B, C, nH, window_size, nW, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B * nH * nW, C, window_size, window_size), (H, W), (Hp, Wp)


def _window_unpartition(
    windows: torch.Tensor,
    batch_size: int,
    orig_hw: tuple[int, int],
    padded_hw: tuple[int, int],
    window_size: int,
) -> torch.Tensor:
    """Reverse of :func:`_window_partition`."""
    Hp, Wp = padded_hw
    H, W = orig_hw
    C = windows.shape[1]
    nH, nW = Hp // window_size, Wp // window_size
    x = windows.view(batch_size, nH, nW, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch_size, C, Hp, Wp)
    if Hp > H or Wp > W:
        x = x[:, :, :H, :W]
    return x


def _align_density(
    density: torch.Tensor | None,
    size: tuple[int, int] | torch.Size,
) -> torch.Tensor | None:
    if density is None:
        return None
    if density.shape[-2:] != size:
        density = F.interpolate(
            density, size=size, mode="bilinear", align_corners=False
        )
    return density


def _density_entropy(density: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = density.sigmoid()
    entropy = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
    flat = entropy.flatten(1)
    e_min = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    e_max = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    return (entropy - e_min) / (e_max - e_min + eps)


def _coordinate_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=0).unsqueeze(0)
    return coords.expand(batch_size, 2, height, width)


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
        window_size: If >0, partition features into non-overlapping windows
            of this size before processing.  Forces strictly local receptive
            fields and local-only density gating.  0 disables (default).
    """

    def __init__(
        self,
        input_dim: int = 256,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
        expansion: int = 4,
        use_density_gate: bool = True,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        ex_ch = input_dim * expansion
        self.use_density_gate = use_density_gate
        self.window_size = window_size

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

    def _forward_core(
        self, x: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Core multi-scale conv + density gate (no residual)."""
        expanded = self.expand(x)
        branch_outputs = torch.stack(
            [branch(expanded) for branch in self.branches], dim=1
        )  # [B, num_branches, ex_ch, H, W]

        if self.density_gate is not None and density is not None:
            gate = self.density_gate(density)  # [B, num_branches, H, W]
            gate = F.softmax(gate, dim=1).unsqueeze(2)  # [B, num_branches, 1, H, W]
            fused = (branch_outputs * gate).sum(dim=1)  # [B, ex_ch, H, W]
        else:
            fused = branch_outputs.mean(dim=1)

        return self.project(fused)

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
        # Align density spatial size once, before any partitioning
        if density is not None and density.shape[-2:] != x.shape[-2:]:
            density = F.interpolate(
                density, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.window_size > 0:
            B = x.shape[0]
            x_win, orig_hw, padded_hw = _window_partition(x, self.window_size)
            d_win = (
                _window_partition(density, self.window_size)[0]
                if density is not None
                else None
            )
            enhanced = self._forward_core(x_win, d_win)
            enhanced = _window_unpartition(
                enhanced, B, orig_hw, padded_hw, self.window_size
            )
            return enhanced + x

        return self._forward_core(x, density) + x  # residual


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


class NonLocalContextExpert(GraphAttentionExpert):
    """Density-biased attention expert for occlusion-heavy non-local cues."""


class TinyPerspectiveExpert(nn.Module):
    """Detail-preserving expert for weak far-field and tiny-head responses."""

    def __init__(self, input_dim: int = 256, hidden_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = max(input_dim // hidden_ratio, 16)
        self.detail_filter = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 1, bias=False),
            nn.BatchNorm2d(input_dim),
        )
        self.hint_gate = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim, 1),
            nn.Sigmoid(),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density = _align_density(density, x.shape[-2:])
        if uncertainty is None and density is not None:
            uncertainty = _density_entropy(density)
        uncertainty = _align_density(uncertainty, x.shape[-2:])
        if density is None:
            density = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        if uncertainty is None:
            uncertainty = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])

        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = x - smooth
        detail = self.detail_filter(detail)
        hint = torch.cat([1.0 - density.sigmoid(), uncertainty], dim=1)
        gate = self.hint_gate(hint)
        return x + self.gate.tanh() * detail * gate


class ScaleSpecialistExpert(nn.Module):
    """ASPP-lite expert that separates small/medium/large scale processing."""

    def __init__(
        self,
        input_dim: int = 256,
        dilations: tuple[int, ...] = (1, 2, 4),
        hidden_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        input_dim,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=input_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(input_dim),
                    nn.GELU(),
                    nn.Conv2d(input_dim, input_dim, 1, bias=False),
                    nn.BatchNorm2d(input_dim),
                )
                for dilation in dilations
            ]
        )
        hidden_dim = max(input_dim // hidden_ratio, 16)
        self.scale_gate = nn.Sequential(
            nn.Conv2d(input_dim + 1, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, len(dilations), 1),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density = _align_density(density, x.shape[-2:])
        if density is None:
            density = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=1)
        gate_input = torch.cat([x, density.sigmoid()], dim=1)
        scale_weights = F.softmax(self.scale_gate(gate_input), dim=1).unsqueeze(2)
        fused = (branch_outputs * scale_weights).sum(dim=1)
        return x + self.gate.tanh() * fused


class BackgroundSuppressExpert(nn.Module):
    """Residual suppressor for repetitive non-human background patterns."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_ratio: int = 4,
        max_suppression: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_dim = max(input_dim // hidden_ratio, 16)
        self.max_suppression = float(max_suppression)
        self.bg_gate = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 1, bias=False),
            nn.BatchNorm2d(input_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density = _align_density(density, x.shape[-2:])
        if uncertainty is None and density is not None:
            uncertainty = _density_entropy(density)
        uncertainty = _align_density(uncertainty, x.shape[-2:])
        if density is None:
            density = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        if uncertainty is None:
            uncertainty = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])

        avg_feat = x.mean(dim=1, keepdim=True)
        max_feat = x.amax(dim=1, keepdim=True)
        hint = torch.cat([avg_feat, max_feat, 1.0 - density.sigmoid(), uncertainty], dim=1)
        suppress = self.bg_gate(hint)
        residual = self.residual(x)
        scale = self.max_suppression * self.gate.tanh().abs()
        return x - scale * suppress * residual


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
        local_prior: Additive logit bias for the local expert (index 0).
            Positive values make the router favour local processing by default.
            0.0 means neutral (original behaviour).
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 2,
        grid_stride: int = 4,
        local_prior: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_stride = grid_stride
        self.num_experts = num_experts
        self.local_prior = local_prior
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

        # Local-first bias: add prior to local expert (index 0) before softmax
        if self.local_prior != 0.0:
            bias = scores.new_zeros(1, self.num_experts, 1, 1)
            bias[0, 0, 0, 0] = self.local_prior
            scores = scores + bias

        return F.softmax(scores, dim=1)  # [B, num_experts, H, W]


# ---------------------------------------------------------------------------
# Balance Loss (reusable from compact MoE, kept self-contained here)
# ---------------------------------------------------------------------------


class GraphMoEBalanceLoss(nn.Module):
    """Expert-balance regulariser for dense or top-k GraphMoE routing."""

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_importance: float = 0.0,
        lambda_capacity: float = 0.0,
        router_z_loss_weight: float = 0.0,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.lambda_balance = float(lambda_balance)
        self.lambda_importance = float(lambda_importance)
        self.lambda_capacity = float(lambda_capacity)
        self.router_z_loss_weight = float(router_z_loss_weight)
        self.capacity_factor = float(capacity_factor)

    @staticmethod
    def _cv_squared(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return values.new_zeros(())
        mean = values.mean()
        return values.var(unbiased=False) / (mean * mean + 1e-8)

    def forward(
        self,
        expert_weights: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            expert_weights: [B, num_experts, H, W]
            router_logits: Optional pre-softmax router scores.

        Returns:
            dict with balance components and ``total_aux``.
        """
        if expert_weights.dim() == 4:
            usage = expert_weights.mean(dim=(0, 2, 3))  # [num_experts]
            load = (expert_weights > 1e-6).float().mean(dim=(0, 2, 3))
        else:
            usage = expert_weights.mean(dim=0)
            load = (expert_weights > 1e-6).float().mean(dim=0)

        p = torch.clamp(usage, min=0.0)
        p = p / (p.sum() + 1e-8)
        num_experts = p.size(0)
        max_entropy = math.log(float(num_experts))
        current_entropy = -(p * torch.log(p + 1e-8)).sum()
        l_balance = max_entropy - current_entropy

        l_importance = self._cv_squared(usage) + self._cv_squared(load)
        capacity = self.capacity_factor / max(float(num_experts), 1.0)
        l_capacity = torch.clamp(usage - capacity, min=0.0).pow(2).sum()
        if router_logits is None:
            l_router_z = usage.new_zeros(())
        else:
            l_router_z = torch.logsumexp(router_logits, dim=1).pow(2).mean()

        total = (
            self.lambda_balance * l_balance
            + self.lambda_importance * l_importance
            + self.lambda_capacity * l_capacity
            + self.router_z_loss_weight * l_router_z
        )
        return {
            "l_balance": l_balance,
            "l_importance": l_importance,
            "l_capacity": l_capacity,
            "l_router_z": l_router_z,
            "total_aux": total,
        }


class GraphMoERouter(nn.Module):
    """Coarse-to-fine router that emits per-token top-k expert weights."""

    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 5,
        grid_stride: int = 4,
        top_k: int = 2,
        temperature: float = 1.0,
        noisy_routing_std: float = 0.0,
        use_uncertainty_hint: bool = True,
        use_coordinate_hint: bool = True,
        expert_prior: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.num_experts = int(num_experts)
        self.grid_stride = int(grid_stride)
        self.top_k = min(int(top_k), self.num_experts)
        self.temperature = float(temperature)
        self.noisy_routing_std = float(noisy_routing_std)
        self.use_uncertainty_hint = bool(use_uncertainty_hint)
        self.use_coordinate_hint = bool(use_coordinate_hint)

        hint_channels = 1
        if self.use_uncertainty_hint:
            hint_channels += 1
        if self.use_coordinate_hint:
            hint_channels += 2
        in_ch = input_dim + hint_channels
        hidden_dim = max(input_dim // 4, 32)
        self.score_net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.num_experts, 1),
        )

        if expert_prior is None:
            prior = torch.zeros(self.num_experts, dtype=torch.float32)
        else:
            prior = torch.tensor(tuple(expert_prior), dtype=torch.float32)
            if prior.numel() != self.num_experts:
                raise ValueError(
                    f"expert_prior must have {self.num_experts} values, "
                    f"got {prior.numel()}"
                )
        self.register_buffer("expert_prior", prior.view(1, self.num_experts, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        density_aligned = _align_density(density, x.shape[-2:])
        assert density_aligned is not None
        density = density_aligned
        hints = [density.sigmoid()]
        if self.use_uncertainty_hint:
            if uncertainty is None:
                uncertainty = _density_entropy(density)
            uncertainty_aligned = _align_density(uncertainty, x.shape[-2:])
            assert uncertainty_aligned is not None
            hints.append(uncertainty_aligned)
        if self.use_coordinate_hint:
            hints.append(
                _coordinate_grid(
                    x.shape[0], x.shape[2], x.shape[3], x.device, x.dtype
                )
            )
        inp = torch.cat([x, *hints], dim=1)

        height, width = inp.shape[-2:]
        stride = self.grid_stride
        if height >= stride and width >= stride:
            coarse = F.avg_pool2d(inp, kernel_size=stride, stride=stride)
            logits = self.score_net(coarse)
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        else:
            logits = self.score_net(inp)
        expert_prior = self.expert_prior
        assert isinstance(expert_prior, torch.Tensor)
        logits = logits + expert_prior.to(device=logits.device, dtype=logits.dtype)

        route_logits = logits / self.temperature
        if training and self.noisy_routing_std > 0.0:
            route_logits = route_logits + torch.randn_like(route_logits) * self.noisy_routing_std

        if self.top_k < self.num_experts:
            top_idx = torch.topk(route_logits, k=self.top_k, dim=1).indices
            mask = torch.zeros_like(route_logits, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            route_logits = route_logits.masked_fill(~mask, torch.finfo(route_logits.dtype).min)

        weights = F.softmax(route_logits, dim=1)
        return weights, logits


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
        local_window_size: Window partition size for LocalExpert.  0=disabled
            (default).  Positive values confine convolutions and density
            gating to non-overlapping windows, enforcing local-first bias.
        grid_stride: Coarse routing patch stride.
        local_prior: Additive logit bias toward the local expert in the
            router.  0.0=neutral (default).  Positive values (e.g. 1.0) make
            the router favour local processing unless density evidence
            overrides.
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
        local_window_size: int = 0,
        # Router
        grid_stride: int = 4,
        local_prior: float = 0.0,
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
                window_size=local_window_size,
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
                local_prior=local_prior,
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


class GraphMoE(nn.Module):
    """Unified Graph Mixture-of-Experts replacement for DSGCNet dual GCN.

    The module keeps the fusion-stage contract unchanged while routing each
    spatial token to complementary graph/scale/background experts:

    0. local_occlusion: local density-gated multi-scale convolution
    1. nonlocal_context: density-biased attention for long-range cues
    2. tiny_perspective: far-field detail enhancement
    3. scale_specialist: ASPP-lite scale-separated processing
    4. background_suppress: residual background suppression
    """

    EXPERT_NAMES = (
        "local_occlusion",
        "nonlocal_context",
        "tiny_perspective",
        "scale_specialist",
        "background_suppress",
    )

    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 5,
        top_k: int = 2,
        router_temperature: float = 1.0,
        noisy_routing_std: float = 0.0,
        grid_stride: int = 4,
        router_detach_density: bool = True,
        use_uncertainty_hint: bool = True,
        use_coordinate_hint: bool = True,
        expert_prior: tuple[float, ...] | None = None,
        aux_loss_weight: float = 1.0,
        lambda_balance: float = 0.01,
        lambda_importance: float = 0.01,
        lambda_capacity: float = 0.0,
        router_z_loss_weight: float = 0.0,
        capacity_factor: float = 1.25,
        # Shared/local expert knobs
        local_kernels: tuple[int, ...] = (1, 3, 5),
        local_expansion: int = 2,
        local_use_density_gate: bool = True,
        local_window_size: int = 0,
        # Non-local expert knobs
        num_heads: int = 4,
        use_density_bias: bool = True,
        density_bias_scale: float = 1.0,
        attn_dropout: float = 0.1,
        # Scale/background knobs
        scale_dilations: tuple[int, ...] = (1, 2, 4),
        background_max_suppression: float = 0.5,
        residual_gate_init: float = 1.0,
        disabled_experts: tuple[str | int, ...] = (),
        disable_local_occlusion: bool = False,
        disable_nonlocal_context: bool = False,
        disable_tiny_perspective: bool = False,
        disable_scale_specialist: bool = False,
        disable_background_suppress: bool = False,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or num_experts > len(self.EXPERT_NAMES):
            raise ValueError(
                f"num_experts must be in [1, {len(self.EXPERT_NAMES)}], "
                f"got {num_experts}"
            )
        self.input_dim = input_dim
        self.expert_names = self.EXPERT_NAMES[: int(num_experts)]
        self.router_detach_density = router_detach_density
        self.aux_loss_weight = float(aux_loss_weight)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

        disabled = self._normalise_disabled(disabled_experts)
        flag_disabled = {
            "local_occlusion": disable_local_occlusion,
            "nonlocal_context": disable_nonlocal_context,
            "tiny_perspective": disable_tiny_perspective,
            "scale_specialist": disable_scale_specialist,
            "background_suppress": disable_background_suppress,
        }
        disabled.update(name for name, flag in flag_disabled.items() if flag)
        self.active_names = tuple(
            name for name in self.expert_names if name not in disabled
        )
        if not self.active_names:
            raise ValueError("GraphMoE must keep at least one expert active")

        modules: dict[str, nn.Module] = {}
        if "local_occlusion" in self.active_names:
            modules["local_occlusion"] = LocalExpert(
                input_dim=input_dim,
                kernel_sizes=local_kernels,
                expansion=local_expansion,
                use_density_gate=local_use_density_gate,
                window_size=local_window_size,
            )
        if "nonlocal_context" in self.active_names:
            modules["nonlocal_context"] = NonLocalContextExpert(
                input_dim=input_dim,
                num_heads=num_heads,
                use_density_bias=use_density_bias,
                density_bias_scale=density_bias_scale,
                attn_dropout=attn_dropout,
            )
        if "tiny_perspective" in self.active_names:
            modules["tiny_perspective"] = TinyPerspectiveExpert(input_dim=input_dim)
        if "scale_specialist" in self.active_names:
            modules["scale_specialist"] = ScaleSpecialistExpert(
                input_dim=input_dim,
                dilations=scale_dilations,
            )
        if "background_suppress" in self.active_names:
            modules["background_suppress"] = BackgroundSuppressExpert(
                input_dim=input_dim,
                max_suppression=background_max_suppression,
            )
        self.experts = nn.ModuleDict(modules)

        active_prior = None
        if expert_prior is not None:
            if len(expert_prior) != len(self.expert_names):
                raise ValueError(
                    f"expert_prior must have {len(self.expert_names)} values, "
                    f"got {len(expert_prior)}"
                )
            active_prior = tuple(
                float(expert_prior[self.expert_names.index(name)])
                for name in self.active_names
            )

        self.router: GraphMoERouter | None = None
        if len(self.active_names) > 1:
            self.router = GraphMoERouter(
                input_dim=input_dim,
                num_experts=len(self.active_names),
                grid_stride=grid_stride,
                top_k=top_k,
                temperature=router_temperature,
                noisy_routing_std=noisy_routing_std,
                use_uncertainty_hint=use_uncertainty_hint,
                use_coordinate_hint=use_coordinate_hint,
                expert_prior=active_prior,
            )
        self.aux_loss = GraphMoEBalanceLoss(
            lambda_balance=lambda_balance,
            lambda_importance=lambda_importance,
            lambda_capacity=lambda_capacity,
            router_z_loss_weight=router_z_loss_weight,
            capacity_factor=capacity_factor,
        )
        self.last_usage: torch.Tensor | None = None

    def _normalise_disabled(
        self, disabled_experts: tuple[str | int, ...]
    ) -> set[str]:
        disabled: set[str] = set()
        for value in disabled_experts:
            if isinstance(value, int):
                if value < 0 or value >= len(self.expert_names):
                    raise ValueError(f"disabled expert index out of range: {value}")
                disabled.add(self.expert_names[value])
            else:
                name = str(value)
                if name not in self.expert_names:
                    raise ValueError(f"Unknown GraphMoE expert {name!r}")
                disabled.add(name)
        return disabled

    def _run_expert(
        self,
        name: str,
        x: torch.Tensor,
        density: torch.Tensor,
        uncertainty: torch.Tensor | None,
    ) -> torch.Tensor:
        expert = self.experts[name]
        if name in {"tiny_perspective", "background_suppress"}:
            return expert(x, density, uncertainty)  # type: ignore[misc]
        return expert(x, density)  # type: ignore[misc]

    def _scatter_active_weights(
        self,
        active_weights: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        B, _, H, W = active_weights.shape
        weights = active_weights.new_zeros(B, num_experts, H, W)
        for active_idx, name in enumerate(self.active_names):
            full_idx = self.expert_names.index(name)
            weights[:, full_idx] = active_weights[:, active_idx]
        return weights

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        density_for_experts = density.detach()
        density_for_router = density.detach() if self.router_detach_density else density
        uncertainty_for_router = uncertainty.detach() if uncertainty is not None else None

        expert_outputs = torch.stack(
            [
                self._run_expert(
                    name,
                    x,
                    density_for_experts,
                    uncertainty_for_router,
                )
                for name in self.active_names
            ],
            dim=1,
        )

        if len(self.active_names) == 1:
            active_weights = x.new_ones(x.shape[0], 1, x.shape[2], x.shape[3])
            active_logits = None
        else:
            assert self.router is not None
            active_weights, active_logits = self.router(
                x,
                density_for_router,
                uncertainty=uncertainty_for_router,
                training=training,
            )

        mixed = (expert_outputs * active_weights.unsqueeze(2)).sum(dim=1)
        feature_fl = x + self.residual_gate.tanh() * (mixed - x)
        weights = self._scatter_active_weights(active_weights, len(self.expert_names))
        self.last_usage = weights.mean(dim=(0, 2, 3)).detach()

        aux_losses: dict[str, torch.Tensor] = {}
        if training and len(self.active_names) > 1:
            aux_losses = self.aux_loss(active_weights, active_logits)
            router_entropy = -(
                active_weights.clamp_min(1e-8) * torch.log(active_weights.clamp_min(1e-8))
            ).sum(dim=1).mean()
            aux_losses["router_entropy"] = router_entropy

        return feature_fl, aux_losses, weights

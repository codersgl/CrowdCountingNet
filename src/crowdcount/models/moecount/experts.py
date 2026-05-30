"""Heterogeneous experts for MoECountNet — Scale × Paradigm dual-axis specialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from crowdcount.models.moecount.deformable_expert import DeformableCrossScaleExpert
from crowdcount.models.moecount.gate import GraphAwareSparseTop2Gate, PixelSoftGate, SparseTop2Gate
from crowdcount.models.moecount.losses import ExpertImportanceLoss
from crowdcount.models.neck import SPD


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class MultiSpectralChannelAttention(nn.Module):
    """FcaNet-style multi-spectral channel attention via 2D DCT bases.

    Replaces the single GAP scalar (DC component only) with K different
    2D DCT frequency components, enriching channel descriptors with
    multi-frequency texture information at near-zero parameter cost.

    Ref: Qin et al., "FcaNet: Frequency Channel Attention Networks", ICCV 2021.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        num_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        dct_basis = self._build_dct_basis(num_freqs, channels)
        self.register_buffer("dct_basis", dct_basis)  # [C, K, 1, 1]
        self.fc = nn.Sequential(
            nn.Conv2d(channels * num_freqs, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_dct_basis(
        num_freqs: int, channels: int, base_h: int = 8, base_w: int = 8
    ) -> torch.Tensor:
        """Pre-compute K different 2D DCT basis vectors as constant buffers.

        Uses the standard DCT-II formulation where the (u,v) basis image is:

            B_{u,v}(i,j) = cos(pi*u*(i+0.5)/H) * cos(pi*v*(j+0.5)/W)

        The first basis (u=0, v=0) recovers GAP — all coefficients equal.
        Subsequent bases capture progressively higher spatial frequencies.
        """
        basis_list = []
        # Select K DCT frequency pairs distributed across the spectrum.
        # Strategy: grid-scan low-frequency region first, then extend.
        freq_pairs: list[tuple[int, int]] = [(0, 0)]  # DC = GAP
        for d in range(1, num_freqs):
            # Zigzag-like: alternate between adding horizontal and vertical freqs
            if d % 2 == 1:
                freq_pairs.append((d // 2 + 1, 0))
            else:
                freq_pairs.append((0, d // 2))
        freq_pairs = freq_pairs[:num_freqs]

        i = torch.arange(base_h, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        j = torch.arange(base_w, dtype=torch.float32).unsqueeze(0)  # [1, W]

        for u, v in freq_pairs:
            basis_u = torch.cos(torch.pi * u * (i + 0.5) / base_h)  # [H, 1]
            basis_v = torch.cos(torch.pi * v * (j + 0.5) / base_w)  # [1, W]
            basis_2d = basis_u @ basis_v  # [H, W]
            basis_2d = basis_2d / basis_2d.abs().sum().clamp_min(1e-8)  # L1-normalize
            basis_list.append(basis_2d)

        # Stack → [K, H, W], then expand channel dimension: treat equally per channel
        basis_stack = torch.stack(basis_list)  # [K, H, W]
        basis_stack = basis_stack.unsqueeze(0).expand(channels, -1, -1, -1)  # [C, K, H, W]
        return basis_stack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Resize pre-computed bases to match input spatial size
        dct_basis = self.dct_basis  # [C, K, base_h, base_w]
        if H != dct_basis.shape[-2] or W != dct_basis.shape[-1]:
            dct_basis = (
                F.interpolate(
                    dct_basis.flatten(0, 1).unsqueeze(0),  # [1, C*K, base_h, base_w]
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .view(C, self.num_freqs, H, W)
            )
        # Compress each channel with K different DCT bases: [B, C, K]
        freq_components = (x.unsqueeze(2) * dct_basis.unsqueeze(0)).sum(dim=[-2, -1])  # [B, C, K]
        # Flatten to [B, C*K, 1, 1] for FC processing
        freq_flat = freq_components.view(B, C * self.num_freqs, 1, 1)
        attn = self.fc(freq_flat)  # [B, C, 1, 1]
        return x * attn


class DensityAdaptiveLocalExpert(nn.Module):
    """Stride-8 expert: density-adaptive multi-scale dilated conv + FFN + MSCA.

    Replaces LocalDetailExpert with three design improvements:

    1. **Multi-scale dilated convs** (d=1,2,3) replace strip convs — circular
       (non-strip) receptive fields at progressive scales matching head sizes.
       Grouped convs (default groups=16) provide partial cross-channel
       interaction during spatial processing, unlike pure depthwise.

    2. **FFN channel expansion** (default 256→512→256) for deeper non-linear
       processing, matching the depth of SpatialRelationExpert's FFN.

    3. **Density-adaptive modulation** — processing intensity scales with local
       crowd density via a zero-init per-pixel gate. This gives e1 a unique
       capability that e2 (stride-16 MSA) cannot provide: stride-8 resolution
       density-aware local refinement.

    When `use_point_aux=True`, adds a point response head for precise head
    localization plus point-guided feature gating, transforming this expert
    into a **PointLocalizationExpert**. The point head predicts per-pixel
    (offset_x, offset_y, confidence, radius), supervised via Hungarian matching
    against ground-truth head annotations.

    Internal **standard** residuals (random-init, non-zero from step 1) on
    both the multi-scale block and the FFN ensure stable gradient flow
    while producing differentiated features (no identity collapse).
    Output uses a zero-init gated residual for training stability.
    """

    needs_density = True  # gate dispatch: this expert receives the density kwarg

    def __init__(
        self,
        channels: int = 256,
        dilations: tuple[int, ...] = (1, 2, 3),
        groups: int = 16,
        ffn_expansion: int = 2,
        use_density_modulation: bool = True,
        use_multi_spectral_se: bool = True,
        ms_num_freqs: int = 4,
        # --- PointLocalizationExpert params ---
        use_point_aux: bool = False,
        point_hidden: int = 64,
        point_loss_weight: float = 1.0,
        point_cls_weight: float = 1.0,
        point_reg_weight: float = 0.0002,
        point_cost_class: float = 1.0,
        point_cost_point: float = 0.05,
        point_eos_coef: float = 0.5,
        point_max_candidates: int = 512,
    ) -> None:
        super().__init__()
        self.use_density_modulation = use_density_modulation
        self.use_point_aux = use_point_aux
        self.point_loss_weight = point_loss_weight
        self.point_cls_weight = point_cls_weight
        self.point_reg_weight = point_reg_weight
        self.point_cost_class = point_cost_class
        self.point_cost_point = point_cost_point
        self.point_eos_coef = point_eos_coef
        self.point_max_candidates = point_max_candidates

        # ---- Stage 1: Multi-scale dilated conv block ----
        self.ms_norm = nn.GroupNorm(min(32, channels), channels)
        self.dilated_branches = nn.ModuleList()
        for d in dilations:
            self.dilated_branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=d,         # padding = dilation to preserve spatial size
                        dilation=d,
                        groups=groups,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
        # Per-dilation learnable scales so the network can weight branches
        self.branch_scales = nn.Parameter(torch.ones(len(dilations)))
        # Fuse multi-scale branches back to channels
        branch_in = channels * len(dilations)
        self.fuse_branches = nn.Sequential(
            nn.Conv2d(branch_in, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
        )

        # ---- Stage 2: FFN channel expansion ----
        ffn_hidden = channels * ffn_expansion
        self.ffn_norm = nn.GroupNorm(min(32, channels), channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, channels, kernel_size=1, bias=False),
        )

        # ---- Stage 3: Channel attention ----
        if use_multi_spectral_se:
            self.channel_attn: nn.Module = MultiSpectralChannelAttention(
                channels, reduction=4, num_freqs=ms_num_freqs,
            )
        else:
            self.channel_attn = SE(channels, reduction=4)

        # ---- Stage 4: Density-adaptive modulation ----
        if use_density_modulation:
            self.density_gate = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.density_gain = nn.Parameter(torch.zeros(1))

        # ---- Output projection (standard residual, like e2's internal blocks) ----
        self.output_norm = nn.GroupNorm(min(32, channels), channels)
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # ---- Stage 5 (optional): Point response head (PointLocalizationExpert) ----
        if use_point_aux:
            self.point_head = nn.Sequential(
                nn.Conv2d(channels, point_hidden, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, point_hidden), point_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(point_hidden, 4, kernel_size=1),
            )
            # zero-init → point gating is disabled at training start
            self.point_gain = nn.Parameter(torch.zeros(1))
            # internal Hungarian matcher (lazily constructed)
            self._matcher: nn.Module | None = None
            # storage for aux loss computation
            self.last_point_preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Stage 1: Multi-scale dilated conv block (internal standard residual)
        normed = self.ms_norm(features)
        branch_outputs: list[torch.Tensor] = []
        for i, branch in enumerate(self.dilated_branches):
            out = branch(normed)
            branch_outputs.append(out * self.branch_scales[i])
        multi_scale = self.fuse_branches(torch.cat(branch_outputs, dim=1))
        x = features + multi_scale  # standard residual (random init, non-zero)

        # Stage 2: FFN (pre-norm, internal standard residual)
        x = x + self.ffn(self.ffn_norm(x))

        # Stage 3: Channel attention
        x = self.channel_attn(x)

        # Stage 4: Density-adaptive modulation (zero-init → disabled at start).
        # Density is detached here — the modulation gate is a feed-forward
        # conditioning signal, NOT a gradient path.  Allowing gradient feedback
        # through the gate risks the density head optimising for "expert-pleasing"
        # density maps rather than GT-faithful ones.
        if self.use_density_modulation and density is not None:
            gate = self.density_gate(density.detach())
            x = x * (1.0 + self.density_gain.tanh() * gate)

        # Stage 5 (optional): Point response head (PointLocalizationExpert)
        if self.use_point_aux:
            point_out: torch.Tensor = self.point_head(x)  # [B, 4, H, W]
            offset_xy = torch.tanh(point_out[:, :2])  # [-1, 1] in stride-8 pixels
            confidence = point_out[:, 2:3].sigmoid()  # [B, 1, H, W]
            radius = F.softplus(point_out[:, 3:4]) + 0.5  # positive, min=0.5
            self.last_point_preds = (offset_xy, confidence, radius)
            # Point-guided feature gating (zero-init → disabled at start)
            x = x * (1.0 + self.point_gain.tanh() * confidence)

        # Output: pure transform (no residual) — differentiated from e2's residual path.
        # Internal residuals (dilated block + FFN) provide gradient stability.
        return self.output_proj(self.output_norm(x))

    def compute_aux_loss(
        self, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict[str, torch.Tensor]:
        """Hungarian-matched point prediction loss for PointLocalizationExpert.

        Only called when ``use_point_aux=True`` and ``self.training``.
        Matches top-K predicted points to ground-truth head annotations
        using the Hungarian algorithm, then computes focal classification
        loss + SmoothL1 offset regression loss.
        """
        if not self.use_point_aux or self.last_point_preds is None or targets is None:
            return {}
        offset_xy, confidence, radius = self.last_point_preds
        B, _, H, W = confidence.shape
        device = confidence.device
        dtype = confidence.dtype

        # Lazy-construct Hungarian matcher (can't deepcopy scipy, so we build on first use)
        if self._matcher is None:
            from crowdcount.models.matcher import HungarianMatcher_Crowd
            self._matcher = HungarianMatcher_Crowd(
                cost_class=self.point_cost_class,
                cost_point=self.point_cost_point,
            )

        total_cls_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
        matched_count = 0

        for b in range(B):
            tgt_points = targets[b].get("point")  # [N_gt, 2] in image coordinates
            if tgt_points is None or tgt_points.numel() == 0:
                continue
            tgt_points_f = tgt_points.to(device=device, dtype=dtype)
            N_gt = tgt_points_f.shape[0]

            # --- Top-K candidate selection ---
            conf_b = confidence[b, 0]  # [H, W]
            K = min(self.point_max_candidates, conf_b.numel())
            top_vals, top_flat_idx = conf_b.flatten().topk(K)
            top_rows = top_flat_idx // W
            top_cols = top_flat_idx % W

            # Candidate predictions in stride-8 coordinates
            pred_offsets_xy = offset_xy[b, :, top_rows, top_cols].T  # [K, 2]
            pred_points_s8 = torch.stack([
                top_cols.to(dtype=dtype) + pred_offsets_xy[:, 0],
                top_rows.to(dtype=dtype) + pred_offsets_xy[:, 1],
            ], dim=1)  # [K, 2]

            # GT points in stride-8 coordinates
            gt_points_s8 = tgt_points_f / 8.0  # image coords → stride-8

            # --- Hungarian matching ---
            # Cls cost: negative log-confidence
            cls_cost = -(top_vals + 1e-8).log().unsqueeze(1).expand(K, N_gt)  # [K, N_gt]
            # Point cost: L1 distance
            point_cost = torch.cdist(pred_points_s8.unsqueeze(0), gt_points_s8.unsqueeze(0)).squeeze(0)  # [K, N_gt]
            cost_matrix = self.point_cost_class * cls_cost + self.point_cost_point * point_cost

            from scipy.optimize import linear_sum_assignment
            cost_cpu = cost_matrix.detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_cpu)
            matched_k = len(row_idx)

            if matched_k == 0:
                continue

            row_idx_t = torch.as_tensor(row_idx, device=device, dtype=torch.int64)
            col_idx_t = torch.as_tensor(col_idx, device=device, dtype=torch.int64)

            # Classification loss (focal-style): matched→positive, unmatched→background
            pos_mask = torch.zeros(K, device=device)
            pos_mask[row_idx_t] = 1.0
            # Binary focal loss: -α*(1-p)^γ*log(p) for positives, -α*p^γ*log(1-p) for negatives
            alpha = 0.25
            gamma = 2.0
            p_t = torch.where(pos_mask > 0.5, top_vals, 1.0 - top_vals)
            alpha_t = torch.where(pos_mask > 0.5, alpha, 1.0 - alpha)
            focal = -alpha_t * (1.0 - p_t).pow(gamma) * (p_t + 1e-8).log()
            total_cls_loss = total_cls_loss + focal.mean()

            # Regression loss (SmoothL1 on matched offset)
            matched_pred = pred_points_s8[row_idx_t]  # [M, 2]
            matched_gt = gt_points_s8[col_idx_t]  # [M, 2]
            reg_loss = F.smooth_l1_loss(matched_pred, matched_gt, beta=1.0)
            total_reg_loss = total_reg_loss + reg_loss
            matched_count += 1

        if matched_count == 0:
            return {}

        cls_loss = self.point_cls_weight * total_cls_loss / max(matched_count, 1)
        reg_loss = self.point_reg_weight * total_reg_loss / max(matched_count, 1)
        return {
            "l_pl_point_cls": cls_loss,
            "l_pl_point_reg": reg_loss,
            "l_pl_point": self.point_loss_weight * (cls_loss + reg_loss),
        }


class _SharedResBlock(nn.Module):
    """Conv3×3 → GN → ReLU → Conv3×3 → GN + residual for SharedExpert."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)


class SharedExpert(nn.Module):
    """Deepened shared expert with residual blocks for stronger baseline features.

    Provides a reliable gradient highway that is always active, allowing
    routed experts to focus on learning specialised residuals rather than
    basic feature extraction.  The deeper architecture (default 3 residual
    blocks) gives the shared path enough capacity to handle common-case
    feature refinement, which stabilises training and accelerates convergence.
    """

    def __init__(self, channels: int = 256, num_blocks: int = 3) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.blocks = nn.Sequential(*[
            _SharedResBlock(channels) for _ in range(num_blocks)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.blocks(features)


class OcclusionReasoningExpert(nn.Module):
    """Stride-8 expert: visibility assessment + local context completion for occlusion.

    Designed for dense crowd regions where heavy inter-person occlusion causes
    the backbone to see only partial head features (→ missed detections).

    Instead of expensive cross-attention, uses lightweight neighborhood feature
    pooling: each position borrows statistics from its 3×3 locality to "complete"
    partially visible heads, conditioned on an occlusion probability map derived
    from the current density estimate.

    Stages:
      1. Visibility assessment — tiny density→occlusion-probability embed
      2. Context completion — F.unfold 3×3 neighborhood → 1×1 fusion
      3. Occlusion-conditioned gating — zero-init, disabled at start
      4. FFN (channel expansion) + SE + output projection
      5. Internal density head (for aux consistency supervision)
    """

    needs_density = True  # gate dispatch: this expert receives the density kwarg

    def __init__(
        self,
        channels: int = 256,
        ffn_expansion: int = 2,
        use_se: bool = True,
        # --- Occlusion aux supervision ---
        use_occlusion_aux: bool = False,
        occ_emb_hidden: int = 16,
        occ_consistency_weight: float = 1.0,
        occ_density_threshold: float = 5.0,
        occ_head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.use_occlusion_aux = use_occlusion_aux
        self.occ_consistency_weight = occ_consistency_weight
        self.occ_density_threshold = occ_density_threshold

        # ---- Stage 1: Visibility assessment ----
        self.occ_embed = nn.Sequential(
            nn.Conv2d(1, occ_emb_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(occ_emb_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # zero-init — occlusion gating disabled at training start
        self.occ_gain = nn.Parameter(torch.zeros(1))

        # ---- Stage 2: Context completion via neighborhood pooling ----
        half_c = channels // 2
        self.proj_local = nn.Conv2d(channels, half_c, kernel_size=1, bias=False)
        self.proj_ctx = nn.Conv2d(channels, half_c, kernel_size=1, bias=False)
        self.fuse_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
        )
        self.context_norm = nn.GroupNorm(min(32, channels), channels)

        # ---- Stage 3: FFN ----
        ffn_hidden = channels * ffn_expansion
        self.ffn_norm = nn.GroupNorm(min(32, channels), channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, channels, kernel_size=1, bias=False),
        )

        # ---- Stage 4: Channel attention ----
        self.se: nn.Module | None = SE(channels, reduction=4) if use_se else None

        # ---- Output projection ----
        self.output_norm = nn.GroupNorm(min(32, channels), channels)
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # ---- Stage 5 (optional): Internal density head for aux supervision ----
        if use_occlusion_aux:
            self.occ_density_head = nn.Sequential(
                nn.Conv2d(channels, occ_head_hidden, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, occ_head_hidden), occ_head_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(occ_head_hidden, 1, kernel_size=1),
            )
            self.last_occ_density_out: torch.Tensor | None = None
        else:
            self.occ_density_head = None

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = features
        B, C, H, W = x.shape

        # ---- Stage 1: Visibility assessment ----
        if density is not None:
            if density.shape[-2:] != (H, W):
                density = F.interpolate(
                    density.detach(), size=(H, W), mode="bilinear", align_corners=False
                )
            occ_prob = self.occ_embed(density.detach())
        else:
            occ_prob = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # ---- Stage 2: Context completion via neighborhood pooling ----
        # F.unfold extracts all 3×3 patches → average over neighbors
        unfolded = F.unfold(x, kernel_size=3, padding=1)  # [B, C*9, HW]
        unfolded = unfolded.view(B, C, 9, H * W).mean(dim=2).view(B, C, H, W)  # neighborhood avg
        # Lightweight fusion: project local + context to half-C, concat, fuse to C
        local_half = self.proj_local(x)
        ctx_half = self.proj_ctx(unfolded)
        fused = torch.cat([local_half, ctx_half], dim=1)  # [B, C, H, W]
        x = self.fuse_context(fused)
        x = self.context_norm(x)

        # ---- Stage 3: Occlusion-conditioned gating (zero-init → disabled at start) ----
        x = x * (1.0 + self.occ_gain.tanh() * occ_prob)

        # ---- Stage 4: FFN (pre-norm, internal residual) ----
        x = x + self.ffn(self.ffn_norm(x))

        # ---- Stage 5: Channel attention ----
        if self.se is not None:
            x = self.se(x)

        # ---- Stage 6: Internal density head (for aux supervision) ----
        if self.occ_density_head is not None:
            self.last_occ_density_out = self.occ_density_head(x)

        # ---- Output projection (pure transform, same as e0) ----
        # Internal FFN residual already provides gradient stability.
        return self.output_proj(self.output_norm(x))

    def compute_aux_loss(
        self, gt_density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Consistency loss: |internal_density_head(e1_out) - GT| in high-density regions."""
        if (
            not self.use_occlusion_aux
            or self.occ_density_head is None
            or self.last_occ_density_out is None
            or gt_density is None
        ):
            return {}
        pred = self.last_occ_density_out  # [B, 1, H, W]
        if pred.shape[-2:] != gt_density.shape[-2:]:
            pred = F.interpolate(
                pred, size=gt_density.shape[-2:], mode="bilinear", align_corners=False
            )
        mask = (gt_density > self.occ_density_threshold).float().detach()
        denom = mask.sum().clamp_min(1)
        loss = (mask * (pred - gt_density).abs()).sum() / denom
        return {"l_occ_consistency": self.occ_consistency_weight * loss}


class LocalDetailExpert(nn.Module):
    """Stride-8 expert: multi-branch strip convs + multi-spectral channel attention.

    Three parallel depthwise branches capture local patterns at different
    aspect ratios (3x3 square, 1xK horizontal strip, Kx1 vertical strip),
    followed by FcaNet-style multi-spectral channel attention that enriches
    the channel descriptor with K DCT frequency components beyond GAP's DC.

    Ref:
      - SPCANet (Yuan, PeerJ CS 2024): Strip Pooling for crowd counting
      - FcaNet (Qin et al., ICCV 2021): Multi-spectral channel attention
    """

    def __init__(
        self,
        channels: int = 256,
        use_residual: bool = True,
        use_strip_convs: bool = True,
        strip_kernel: int = 7,
        use_multi_spectral_se: bool = True,
        ms_num_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_strip_convs = use_strip_convs

        # --- Depthwise spatial branches ---
        self.dwconv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        if use_strip_convs:
            self.dwconv_1xK = nn.Conv2d(
                channels, channels,
                kernel_size=(1, strip_kernel),
                padding=(0, strip_kernel // 2),
                groups=channels,
            )
            self.dwconv_Kx1 = nn.Conv2d(
                channels, channels,
                kernel_size=(strip_kernel, 1),
                padding=(strip_kernel // 2, 0),
                groups=channels,
            )
            fuse_in = channels * 3
        else:
            fuse_in = channels
        self.fuse_strips = nn.Sequential(
            nn.Conv2d(fuse_in, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

        # --- Channel attention ---
        if use_multi_spectral_se:
            self.channel_attn = MultiSpectralChannelAttention(
                channels, reduction=4, num_freqs=ms_num_freqs,
            )
        else:
            self.channel_attn = SE(channels, reduction=4)

        # --- Output projection ---
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        if use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out_3x3 = self.dwconv_3x3(features)
        if self.use_strip_convs:
            out_1xK = self.dwconv_1xK(features)
            out_Kx1 = self.dwconv_Kx1(features)
            out = torch.cat([out_3x3, out_1xK, out_Kx1], dim=1)
        else:
            out = out_3x3
        out = self.fuse_strips(out)
        out = self.channel_attn(out)
        out = self.fuse(out)
        if self.use_residual:
            return features + self.residual_gate.tanh() * out
        return out


class SpatialRelationExpert(nn.Module):
    """Stride-16 expert: window self-attention for spatial relation modeling.

    SPD downsampling to stride-16 → Window-MSA (8×8 windows) → FFN → bilinear up.
    """

    def __init__(self, channels: int = 256, num_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

        self.spd_down = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
        )

    def _window_partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, nH, nW, ws, ws, C]
        x = x.reshape(B, (Hp // ws) * (Wp // ws), ws * ws, C)
        return x, H, W, pad_h, pad_w

    def _window_unpartition(self, x: torch.Tensor, H: int, W: int, pad_h: int, pad_w: int) -> torch.Tensor:
        ws = self.window_size
        B, nw, _, C = x.shape
        nH = (H + pad_h) // ws
        nW = (W + pad_w) // ws
        x = x.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, nH, ws, nW, ws]
        x = x.reshape(B, C, H + pad_h, W + pad_w)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = features
        x = self.spd_down(features)  # stride-8 → stride-16
        B, C, H, W = x.shape

        # Window MSA
        x_windowed, H_orig, W_orig, pad_h, pad_w = self._window_partition(x)
        B, nw, N, C_ = x_windowed.shape
        Bnw = B * nw
        x_flat = x_windowed.reshape(Bnw * N, C_)
        x_ln = self.norm1(x_flat).reshape(Bnw, N, C_)

        qkv = self.qkv(x_ln).reshape(Bnw, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.clamp(-50, 50), dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(Bnw, N, C)
        attn_out = self.proj(attn_out)

        x_windowed = x_windowed + attn_out.reshape(B, nw, N, C)
        x_unflat = self._window_unpartition(x_windowed, H_orig, W_orig, pad_h, pad_w)

        # FFN
        x_perm = x_unflat.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_ffn = self.norm2(x_perm)
        x_ffn = self.ffn(x_ffn) + x_perm
        x_out = x_ffn.permute(0, 3, 1, 2)

        # Upsample back to stride-8
        return F.interpolate(x_out, size=identity.shape[-2:], mode="bilinear", align_corners=False)


class DensityPatternExpert(nn.Module):
    """Stride-32 expert: multi-scale pyramid pooling + density bin classification.

    Designed for very dense crowd regions (>5 people/stride-8 cell) where
    individuals are unresolvable — counting reduces to a pattern→count mapping.
    Replaces the original GlobalDensityExpert's single DWConv7×7 with:

    1. **PSPNet-style PPM** — AdaptiveAvgPool2d at bins [1,2,3,6] captures
       density context from global (1×1) to neighborhood (6×6) scales.
    2. **Density bin classifier** — quantized density level prediction at
       stride-32, supervised via cross-entropy.
    3. **Pattern prior re-injection** — softmax(detached classifier output)
       → weighted Embedding → zero-init gated addition to features.
    """

    needs_density = True  # gate dispatch: this expert receives the density kwarg

    def __init__(
        self,
        channels: int = 256,
        use_density: bool = True,
        # --- Pattern expert params ---
        use_pattern_aux: bool = False,
        ppm_bins: tuple[int, ...] = (1, 2, 3, 6),
        ppm_reduction: int = 4,
        pattern_num_bins: int = 8,
        pattern_class_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_density = use_density
        self.use_pattern_aux = use_pattern_aux
        self.pattern_num_bins = pattern_num_bins
        self.pattern_class_weight = pattern_class_weight

        # ---- Stage 1: Density-aware feature fusion ----
        if use_density:
            self.density_fuse = nn.Sequential(
                nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, channels), channels),
                nn.ReLU(inplace=True),
            )

        # ---- Stage 2: SPD ×2 → stride-32 ----
        self.spd1 = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
        )
        self.spd2 = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
        )

        # ---- Stage 3: PSPNet Pyramid Pooling Module ----
        ppm_in = channels
        ppm_per_bin = channels // ppm_reduction
        self.ppm_bins = ppm_bins
        self.ppm_pools = nn.ModuleList()
        self.ppm_convs = nn.ModuleList()
        for _bin in ppm_bins:
            self.ppm_convs.append(
                nn.Sequential(
                    nn.Conv2d(ppm_in, ppm_per_bin, kernel_size=1, bias=False),
                    nn.GroupNorm(min(32, ppm_per_bin), ppm_per_bin),
                    nn.ReLU(inplace=True),
                )
            )
        ppm_total_in = ppm_in + len(ppm_bins) * ppm_per_bin
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(ppm_total_in, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
        )

        # ---- Stage 4: Density bin classifier ----
        if use_pattern_aux:
            self.pattern_classifier = nn.Conv2d(channels, pattern_num_bins, kernel_size=1)
            self.pattern_embed = nn.Embedding(pattern_num_bins, channels)
            nn.init.normal_(self.pattern_embed.weight, std=0.01)
            # zero-init → pattern prior disabled at start
            self.pattern_gain = nn.Parameter(torch.zeros(1))
            self.last_density_bin_logits: torch.Tensor | None = None

        # ---- Output projection ----
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        identity_size = features.shape[-2:]

        # ---- Stage 1: Density-aware feature fusion ----
        if self.use_density and density is not None:
            x = torch.cat([features, density], dim=1)
            x = self.density_fuse(x)
        else:
            x = features

        # ---- Stage 2: SPD ×2 → stride-32 ----
        x = self.spd1(x)   # s8 → s16
        x = self.spd2(x)   # s16 → s32
        s32_h, s32_w = x.shape[-2:]

        # ---- Stage 3: PSPNet Pyramid Pooling Module ----
        ppm_outs = [x]
        for i, _bin in enumerate(self.ppm_bins):
            pooled = F.adaptive_avg_pool2d(x, (_bin, _bin))
            pooled = self.ppm_convs[i](pooled)
            pooled = F.interpolate(
                pooled, size=(s32_h, s32_w), mode="bilinear", align_corners=False
            )
            ppm_outs.append(pooled)
        x = self.ppm_fuse(torch.cat(ppm_outs, dim=1))

        # ---- Stage 4: Density bin classifier + pattern prior ----
        if self.use_pattern_aux:
            bin_logits = self.pattern_classifier(x)  # [B, N, H/32, W/32]
            self.last_density_bin_logits = bin_logits
            # Pattern prior re-injection (detached → no gradient through softmax pathway)
            pattern_soft = F.softmax(bin_logits.detach(), dim=1)  # [B, N, H, W]
            pattern_feat = (
                pattern_soft.transpose(1, -1) @ self.pattern_embed.weight
            ).transpose(1, -1)  # [B, C, H, W]
            x = x + self.pattern_gain.tanh() * pattern_feat

        # ---- Stage 5: Output (pure transform) ----
        # Internal PPM + pattern prior already provide gradient paths.
        x = self.fuse(x)
        return F.interpolate(
            x, size=identity_size, mode="bilinear", align_corners=False
        )

    def compute_aux_loss(
        self, gt_density: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Cross-entropy on density bin classification at stride-32."""
        if (
            not self.use_pattern_aux
            or self.last_density_bin_logits is None
            or gt_density is None
        ):
            return {}
        logits = self.last_density_bin_logits  # [B, N, Hs, Ws]
        B, N, Hs, Ws = logits.shape
        # Downsample GT density to stride-32
        gt_s32 = F.adaptive_avg_pool2d(gt_density, (Hs, Ws))  # [B, 1, Hs, Ws]
        # Each stride-32 pixel covers 16 stride-8 cells → scale up count values
        gt_count_per_s32 = gt_s32 * 16.0  # stride-32 / stride-8 = 4, 4*4=16
        # Quantize to N bins (0, 1, 2, ..., 7+ people per stride-8 cell)
        # At stride-32, max expected density per bin = N-1
        bin_width = float(N - 1)
        bin_label = (
            (gt_count_per_s32 / bin_width).clamp(0, N - 1).long().squeeze(1)
        )  # [B, Hs, Ws]
        loss = F.cross_entropy(logits, bin_label, reduction="mean")
        return {"l_dp_class": self.pattern_class_weight * loss}


class GlobalDensityExpert(nn.Module):
    """Stride-32 expert: large-kernel conv + channel attention for global density context.

    SPD×2 downsampling to stride-32 → Conv7×7 DW + SE + Conv1×1 → bilinear up.

    When use_density=True, the predicted density map is concatenated as an
    additional input channel and fused via a 1×1 projection before the SPD
    downsamples, making the expert truly density-aware.
    """

    needs_density = True  # gate dispatch: receives density for concat-fusion path

    def __init__(
        self, channels: int = 256, use_residual: bool = True, use_density: bool = True
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_density = use_density
        if use_density:
            self.density_fuse = nn.Sequential(
                nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
            )
        self.spd_down = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.spd_down2 = nn.Sequential(
            SPD(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.large_kernel = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.se = SE(channels, reduction=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        if use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, features: torch.Tensor, density: torch.Tensor | None = None
    ) -> torch.Tensor:
        identity_size = features.shape[-2:]
        if self.use_density and density is not None:
            x = torch.cat([features, density], dim=1)
            x = self.density_fuse(x)
        else:
            x = features
        x = self.spd_down(x)   # s8 → s16
        x = self.spd_down2(x)  # s16 → s32
        x = self.large_kernel(x)
        x = self.se(x)
        x = self.fuse(x)
        out = F.interpolate(x, size=identity_size, mode="bilinear", align_corners=False)
        if self.use_residual:
            return features + self.residual_gate.tanh() * out
        return out


class HeterogeneousSparseMoE(nn.Module):
    """Three scale×paradigm heterogeneous experts with pixel-wise soft gating.

    Uses HMoDE-style per-pixel softmax routing (Du et al., IEEE TIP 2023)
    instead of hard Top-K selection. All experts always contribute with
    learned spatial weights, preventing expert collapse.
    """

    def __init__(
        self,
        channels: int = 256,
        gate_hidden_channels: int = 128,
        top_k: int = 2,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.98,
        warmup_fraction: float = 0.2,
        warmup_epochs: int | None = None,
        lambda_importance: float = 0.01,
        lambda_load: float = 0.01,
        shared_scale: float = 0.5,
        shared_num_blocks: int = 3,
        shared_scale_learnable: bool = True,
        use_deformable_expert: bool = False,
        deformable_num_heads: int = 4,
        deformable_num_sampling_points: int = 8,
        deformable_num_scale_levels: int = 3,
        deformable_max_offset: float = 8.0,
        deformable_dropout: float = 0.1,
        deformable_use_se: bool = True,
        deformable_use_density_bias: bool = False,
        use_input_residual: bool = True,
        expert_local_detail_use_residual: bool = True,
        expert_global_density_use_residual: bool = True,
        expert_local_detail_use_strip_convs: bool = True,
        expert_local_detail_strip_kernel: int = 7,
        expert_local_detail_use_multi_spectral_se: bool = True,
        expert_local_detail_ms_num_freqs: int = 4,
        gate_type: str = "sparse_top2",
        gate_use_density_hint: bool = False,
        gate_density_hidden: int = 8,
        gate_use_density_bias: bool = False,
        gate_graph_k: int = 4,
        expert_use_density: bool = True,
        expert_global_density_use_density: bool = True,
        # --- DensityAdaptiveLocalExpert config ---
        expert_local_detail_use_density_adaptive: bool = True,
        expert_local_detail_dilations: tuple[int, ...] = (1, 2, 3),
        expert_local_detail_groups: int = 16,
        expert_local_detail_ffn_expansion: int = 2,
        expert_local_detail_use_density_modulation: bool = True,
        # --- Expert replacement flags ---
        use_point_localization_expert: bool = False,
        use_occlusion_reasoning_expert: bool = False,
        use_density_pattern_expert: bool = False,
        # --- PointLocalizationExpert (e0) config ---
        expert_pl_use_point_aux: bool = False,
        expert_pl_point_hidden: int = 64,
        expert_pl_point_loss_weight: float = 1.0,
        expert_pl_point_cls_weight: float = 1.0,
        expert_pl_point_reg_weight: float = 0.0002,
        expert_pl_point_cost_class: float = 1.0,
        expert_pl_point_cost_point: float = 0.05,
        expert_pl_point_eos_coef: float = 0.5,
        expert_pl_point_max_candidates: int = 512,
        # --- OcclusionReasoningExpert (e1) config ---
        expert_occ_use_aux: bool = False,
        expert_occ_emb_hidden: int = 16,
        expert_occ_consistency_weight: float = 1.0,
        expert_occ_density_threshold: float = 5.0,
        expert_occ_head_hidden: int = 128,
        expert_occ_use_residual: bool = True,
        # --- DensityPatternExpert (e2) config ---
        expert_dp_use_aux: bool = False,
        expert_dp_ppm_bins: tuple[int, ...] = (1, 2, 3, 6),
        expert_dp_ppm_reduction: int = 4,
        expert_dp_pattern_num_bins: int = 8,
        expert_dp_pattern_class_weight: float = 1.0,
        expert_dp_use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = 3
        if shared_scale_learnable:
            self.shared_scale = nn.Parameter(torch.tensor(float(shared_scale)))
        else:
            self.register_buffer(
                "shared_scale", torch.tensor(float(shared_scale))
            )
        self.shared_scale_learnable = shared_scale_learnable
        self.use_input_residual = use_input_residual
        self.gate_type = gate_type
        self.shared_expert = SharedExpert(channels, num_blocks=shared_num_blocks)
        spatial_expert: nn.Module
        if use_occlusion_reasoning_expert:
            spatial_expert = OcclusionReasoningExpert(
                channels=channels,
                ffn_expansion=2,
                use_se=True,
                use_occlusion_aux=expert_occ_use_aux,
                occ_emb_hidden=expert_occ_emb_hidden,
                occ_consistency_weight=expert_occ_consistency_weight,
                occ_density_threshold=expert_occ_density_threshold,
                occ_head_hidden=expert_occ_head_hidden,
            )
        elif use_deformable_expert:
            spatial_expert = DeformableCrossScaleExpert(
                channels=channels,
                num_heads=deformable_num_heads,
                num_sampling_points=deformable_num_sampling_points,
                num_scale_levels=deformable_num_scale_levels,
                max_offset=deformable_max_offset,
                dropout=deformable_dropout,
                use_se=deformable_use_se,
                use_density_bias=deformable_use_density_bias,
            )
        else:
            spatial_expert = SpatialRelationExpert(channels)
        local_expert: nn.Module
        if use_point_localization_expert or expert_pl_use_point_aux:
            local_expert = DensityAdaptiveLocalExpert(
                channels,
                dilations=expert_local_detail_dilations,
                groups=expert_local_detail_groups,
                ffn_expansion=expert_local_detail_ffn_expansion,
                use_density_modulation=expert_local_detail_use_density_modulation,
                use_multi_spectral_se=expert_local_detail_use_multi_spectral_se,
                ms_num_freqs=expert_local_detail_ms_num_freqs,
                use_point_aux=expert_pl_use_point_aux,
                point_hidden=expert_pl_point_hidden,
                point_loss_weight=expert_pl_point_loss_weight,
                point_cls_weight=expert_pl_point_cls_weight,
                point_reg_weight=expert_pl_point_reg_weight,
                point_cost_class=expert_pl_point_cost_class,
                point_cost_point=expert_pl_point_cost_point,
                point_eos_coef=expert_pl_point_eos_coef,
                point_max_candidates=expert_pl_point_max_candidates,
            )
        elif expert_local_detail_use_density_adaptive:
            local_expert = DensityAdaptiveLocalExpert(
                channels,
                dilations=expert_local_detail_dilations,
                groups=expert_local_detail_groups,
                ffn_expansion=expert_local_detail_ffn_expansion,
                use_density_modulation=expert_local_detail_use_density_modulation,
                use_multi_spectral_se=expert_local_detail_use_multi_spectral_se,
                ms_num_freqs=expert_local_detail_ms_num_freqs,
            )
        else:
            local_expert = LocalDetailExpert(
                channels,
                use_residual=expert_local_detail_use_residual,
                use_strip_convs=expert_local_detail_use_strip_convs,
                strip_kernel=expert_local_detail_strip_kernel,
                use_multi_spectral_se=expert_local_detail_use_multi_spectral_se,
                ms_num_freqs=expert_local_detail_ms_num_freqs,
            )
        global_expert: nn.Module
        if use_density_pattern_expert:
            global_expert = DensityPatternExpert(
                channels=channels,
                use_density=expert_global_density_use_density,
                use_pattern_aux=expert_dp_use_aux,
                ppm_bins=expert_dp_ppm_bins,
                ppm_reduction=expert_dp_ppm_reduction,
                pattern_num_bins=expert_dp_pattern_num_bins,
                pattern_class_weight=expert_dp_pattern_class_weight,
            )
        else:
            global_expert = GlobalDensityExpert(
                channels,
                use_residual=expert_global_density_use_residual,
                use_density=expert_global_density_use_density,
            )
        self.experts = nn.ModuleList([
            local_expert,
            spatial_expert,
            global_expert,
        ])
        if not expert_use_density:
            for expert in self.experts:
                expert.needs_density = False
        if gate_type == "soft":
            self.gate = PixelSoftGate(
                in_channels=channels,
                num_experts=self.num_experts,
                hidden_channels=gate_hidden_channels,
            )
        elif gate_type == "graph_aware":
            self.gate = GraphAwareSparseTop2Gate(
                in_channels=channels,
                num_experts=self.num_experts,
                hidden_channels=gate_hidden_channels,
                top_k=top_k,
                temperature_init=temperature_init,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay,
                warmup_fraction=warmup_fraction,
                warmup_epochs=warmup_epochs,
                use_density_hint=gate_use_density_hint,
                density_hidden=gate_density_hidden,
                use_density_bias=gate_use_density_bias,
                graph_k=gate_graph_k,
            )
        else:
            self.gate = SparseTop2Gate(
                in_channels=channels,
                num_experts=self.num_experts,
                hidden_channels=gate_hidden_channels,
                top_k=top_k,
                temperature_init=temperature_init,
                temperature_min=temperature_min,
                temperature_decay=temperature_decay,
                warmup_fraction=warmup_fraction,
                warmup_epochs=warmup_epochs,
                use_density_hint=gate_use_density_hint,
                density_hidden=gate_density_hidden,
                use_density_bias=gate_use_density_bias,
            )
        self.eim_loss = ExpertImportanceLoss(
            lambda_importance=lambda_importance,
        )
        self.output_norm = nn.GroupNorm(32, channels)

    @property
    def temperature(self) -> float:
        return getattr(self.gate, "temperature", 1.0)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.gate.set_epoch(epoch, total_epochs)

    def update_temperature(self, decay_rate: float | None = None) -> None:
        self.gate.update_temperature(decay_rate)

    def forward(
        self,
        features: torch.Tensor,
        density: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        gt_density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        shared_out = self.shared_expert(features) * self.shared_scale

        # Dispatch: experts with `needs_density = True` get the density kwarg
        def _call_expert(expert: nn.Module, feat: torch.Tensor) -> torch.Tensor:
            if getattr(expert, "needs_density", False):
                return expert(feat, density=density)
            return expert(feat)

        expert_outputs = torch.stack(
            [_call_expert(expert, features) for expert in self.experts],
            dim=1,
        )  # [B, 3, C, H/8, W/8]

        with torch.no_grad():
            eo = expert_outputs.detach()
            eo_flat = eo.reshape(eo.shape[0], 3, -1)
            eo_norm = F.normalize(eo_flat, dim=-1)
            cos_matrix = torch.bmm(eo_norm, eo_norm.transpose(1, 2))
            avg_cos = cos_matrix.mean(0)
            expert_similarity = {
                "cos_01": avg_cos[0, 1].clone(),
                "cos_02": avg_cos[0, 2].clone(),
                "cos_12": avg_cos[1, 2].clone(),
            }

        route = self.gate(features, density=density.detach() if density is not None else None)
        route_weights = route["weights"]
        if not isinstance(route_weights, torch.Tensor):
            raise TypeError("gate route weights must be a tensor")
        routed = (expert_outputs * route_weights.unsqueeze(2)).sum(dim=1)
        if self.use_input_residual:
            fused = self.output_norm(features + shared_out + routed)
        else:
            fused = self.output_norm(shared_out + routed)

        soft_probs = route["soft_probs"]
        if not isinstance(soft_probs, torch.Tensor):
            raise TypeError("gate soft probs must be a tensor")
        aux_losses: dict[str, torch.Tensor] = self.eim_loss(soft_probs) if self.training else {}

        # --- Expert-specific auxiliary losses ---
        if self.training:
            for i, expert in enumerate(self.experts):
                compute_fn = getattr(expert, "compute_aux_loss", None)
                if compute_fn is not None:
                    # Dispatch correct kwargs per expert type
                    if isinstance(expert, DensityAdaptiveLocalExpert):
                        expert_losses = compute_fn(targets=targets)
                    elif isinstance(expert, OcclusionReasoningExpert):
                        expert_losses = compute_fn(gt_density=gt_density)
                    elif isinstance(expert, DensityPatternExpert):
                        expert_losses = compute_fn(gt_density=gt_density)
                    else:
                        expert_losses = compute_fn()
                    for k, v in expert_losses.items():
                        key = f"e{i}_{k}" if k in aux_losses else k
                        aux_losses[key] = v

            # Recompute total_aux as sum of all sub-losses
            _total = torch.zeros((), device=features.device, dtype=features.dtype)
            for k, v in aux_losses.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    _total = _total + v
            aux_losses["total_aux"] = _total

        route["expert_similarity"] = expert_similarity
        return fused, aux_losses, route

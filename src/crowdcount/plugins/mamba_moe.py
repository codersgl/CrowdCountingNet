from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from timm.layers.drop import DropPath
except ImportError:
    try:
        from timm.models.layers import DropPath
    except ImportError:

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob <= 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor


def _selective_scan_fallback(
    xs: torch.Tensor,
    dts: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, steps = xs.shape
    state_dim = As.shape[-1]

    delta = dts
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    state = xs.new_zeros(batch_size, channels, state_dim)
    outputs: list[torch.Tensor] = []

    As_expanded = As.unsqueeze(0)
    Ds_expanded = Ds.view(1, channels)

    for step in range(steps):
        x_t = xs[:, :, step]
        dt_t = delta[:, :, step]
        b_t = Bs[:, :, step].unsqueeze(1)
        c_t = Cs[:, :, step].unsqueeze(1)
        decay = torch.exp(dt_t.unsqueeze(-1) * As_expanded)
        state = decay * state + dt_t.unsqueeze(-1) * b_t * x_t.unsqueeze(-1)
        y_t = (state * c_t).sum(dim=-1) + Ds_expanded * x_t
        if z is not None:
            y_t = y_t * z[:, :, step]
        outputs.append(y_t.unsqueeze(-1))

    output = torch.cat(outputs, dim=-1)
    if return_last_state:
        return output, state
    return output


def _run_selective_scan(
    xs: torch.Tensor,
    dts: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if selective_scan_fn is not None and xs.is_cuda:
        return selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )
    return _selective_scan_fallback(
        xs,
        dts,
        As,
        Bs,
        Cs,
        Ds,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=False,
    )


def _flatten_direction(
    x: torch.Tensor, direction: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    batch_size, height, width, channels = x.shape
    if direction == 0:
        return x.reshape(batch_size, height * width, channels), (height, width)
    if direction == 1:
        transposed = x.transpose(1, 2).contiguous()
        return transposed.reshape(batch_size, height * width, channels), (width, height)
    if direction == 2:
        seq = torch.flip(x.reshape(batch_size, height * width, channels), dims=[1])
        return seq, (height, width)
    if direction == 3:
        transposed = (
            x.transpose(1, 2).contiguous().reshape(batch_size, height * width, channels)
        )
        return torch.flip(transposed, dims=[1]), (width, height)
    raise ValueError(f"Unsupported direction index: {direction}")


def _restore_direction(
    x: torch.Tensor,
    direction: int,
    height: int,
    width: int,
    seq_hw: tuple[int, int],
) -> torch.Tensor:
    batch_size, _, channels = x.shape
    if direction == 0:
        return x.reshape(batch_size, height, width, channels)
    if direction == 1:
        restored = x.reshape(batch_size, seq_hw[0], seq_hw[1], channels)
        return restored.transpose(1, 2).contiguous()
    if direction == 2:
        restored = torch.flip(x, dims=[1])
        return restored.reshape(batch_size, height, width, channels)
    if direction == 3:
        restored = torch.flip(x, dims=[1]).reshape(
            batch_size, seq_hw[0], seq_hw[1], channels
        )
        return restored.transpose(1, 2).contiguous()
    raise ValueError(f"Unsupported direction index: {direction}")


class MambaMoEBalanceLoss(nn.Module):
    def __init__(self, lambda_balance: float = 0.01) -> None:
        super().__init__()
        self.lambda_balance = float(lambda_balance)

    def forward(self, expert_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        usage = expert_weights.mean(dim=0)
        probs = usage.clamp_min(0.0)
        probs = probs / (probs.sum() + 1e-8)
        max_entropy = math.log(float(probs.numel()))
        current_entropy = -(probs * torch.log(probs + 1e-8)).sum()
        l_balance = max_entropy - current_entropy
        total_aux = self.lambda_balance * l_balance
        return {"l_balance": l_balance, "total_aux": total_aux}


class SingleScanSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        low_dim: int,
        d_state: int = 16,
        d_conv: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.low_dim = int(low_dim)
        self.d_state = int(d_state)
        self.dt_rank = max(1, math.ceil(self.low_dim / 16))

        self.in_proj = nn.Linear(self.d_model, self.low_dim * 2, bias=False)
        self.conv2d = nn.Conv2d(
            self.low_dim,
            self.low_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.low_dim,
            bias=True,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.low_dim, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.low_dim, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.low_dim, 1
        )
        self.A_logs = nn.Parameter(torch.log(A))
        self.Ds = nn.Parameter(torch.ones(self.low_dim, dtype=torch.float32))
        self.out_norm = nn.LayerNorm(self.low_dim)
        self.out_proj = nn.Linear(self.low_dim, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, direction: int) -> torch.Tensor:
        batch_size, height, width, _ = x.shape
        xz = self.in_proj(x)
        x_part, z_part = xz.chunk(2, dim=-1)

        conv_input = x_part.permute(0, 3, 1, 2).contiguous()
        conv_output = self.act(self.conv2d(conv_input)).permute(0, 2, 3, 1).contiguous()

        seq, seq_hw = _flatten_direction(conv_output, direction)
        proj = self.x_proj(seq)
        dts, Bs, Cs = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dts = self.dt_proj(dts).transpose(1, 2).contiguous().float()
        xs = seq.transpose(1, 2).contiguous().float()
        Bs = Bs.transpose(1, 2).contiguous().float()
        Cs = Cs.transpose(1, 2).contiguous().float()
        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        delta_bias = self.dt_proj.bias.float()

        scanned = _run_selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=delta_bias
        )
        scanned = scanned.transpose(1, 2).contiguous()
        scanned = _restore_direction(scanned, direction, height, width, seq_hw)

        scanned = self.out_norm(scanned)
        scanned = scanned * F.silu(z_part)
        return self.out_proj(scanned)


class BiDirectionalChannelSSM(nn.Module):
    """Bidirectional Spectral/Channel SSM aligned with MambaMoE paper.

    The SSM scans along the *channel* (d_inner) dimension, treating each spatial
    position as an independent SSM channel.  SSM channels = d_spectral (fixed at
    init, with adaptive pooling when actual H*W differs).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        d_spectral: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(self.d_model * float(expand))
        self.d_spectral = int(d_spectral)
        self.dt_rank = max(1, math.ceil(self.d_model / 16))

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_spectral,
            self.d_spectral,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_spectral,
            bias=True,
        )
        self.act = nn.SiLU()

        # Forward direction SSM params
        self.x_proj_fwd = nn.Linear(
            self.d_spectral, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj_fwd = nn.Linear(self.dt_rank, self.d_spectral, bias=True)
        A_fwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_spectral, 1
        )
        self.A_logs_fwd = nn.Parameter(torch.log(A_fwd))
        self.Ds_fwd = nn.Parameter(torch.ones(self.d_spectral, dtype=torch.float32))

        # Backward direction SSM params
        self.x_proj_bwd = nn.Linear(
            self.d_spectral, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_spectral, bias=True)
        A_bwd = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_spectral, 1
        )
        self.A_logs_bwd = nn.Parameter(torch.log(A_bwd))
        self.Ds_bwd = nn.Parameter(torch.ones(self.d_spectral, dtype=torch.float32))

        self.out_norm = nn.LayerNorm(self.d_spectral)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _scan_direction(
        self,
        seq: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_logs: nn.Parameter,
        Ds: nn.Parameter,
    ) -> torch.Tensor:
        """Run SSM scan. seq: [B*d_inner, d_spectral, 1] → same shape."""
        # Remove trailing dim for projection: [B*d_inner, d_spectral]
        seq_2d = seq.squeeze(-1)
        proj = x_proj(seq_2d).unsqueeze(-1)  # [B*d_inner, dt_rank+2*d_state, 1]
        dts, Bs, Cs = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=1
        )
        # dts: [B*d_inner, dt_rank, 1]
        dts = dt_proj(dts.squeeze(-1)).unsqueeze(-1)  # [B*d_inner, d_spectral, 1]

        xs = seq.float()  # [B*d_inner, d_spectral, 1]
        dts = dts.float()
        Bs = Bs.float()
        Cs = Cs.float()
        As = -torch.exp(A_logs.float())
        Ds_f = Ds.float()
        delta_bias = dt_proj.bias.float()

        out = _run_selective_scan(
            xs, dts, As, Bs, Cs, Ds_f, z=None, delta_bias=delta_bias
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        hw = height * width
        xz = self.in_proj(x)  # [B, H, W, d_inner*2]
        x_part, z_part = xz.chunk(2, dim=-1)  # each [B, H, W, d_inner]

        # Reshape to [B, H*W, d_inner] then to [B*d_inner, H*W] for spectral view
        x_flat = x_part.reshape(batch_size, hw, self.d_inner)

        # Adaptive pooling if spatial size differs from d_spectral
        if hw != self.d_spectral:
            # [B, d_inner, H*W] → adaptive pool to [B, d_inner, d_spectral]
            x_pool = x_flat.transpose(1, 2).contiguous()
            x_pool = F.adaptive_avg_pool1d(x_pool, self.d_spectral)
            x_pool = x_pool.transpose(1, 2).contiguous()  # [B, d_spectral, d_inner]
        else:
            x_pool = x_flat  # [B, d_spectral, d_inner]

        # Conv1d along d_spectral: [B*d_inner, d_spectral, 1]
        # Reshape: [B, d_spectral, d_inner] → [B*d_inner, d_spectral, 1]
        conv_in = x_pool.permute(0, 2, 1).contiguous()  # [B, d_inner, d_spectral]
        conv_in = conv_in.reshape(batch_size * self.d_inner, self.d_spectral, 1)
        conv_in = self.act(self.conv1d(conv_in))  # [B*d_inner, d_spectral, 1]

        # Forward scan
        fwd_out = self._scan_direction(
            conv_in, self.x_proj_fwd, self.dt_proj_fwd, self.A_logs_fwd, self.Ds_fwd
        )
        # Backward scan
        bwd_out = self._scan_direction(
            torch.flip(conv_in, dims=[2]),
            self.x_proj_bwd,
            self.dt_proj_bwd,
            self.A_logs_bwd,
            self.Ds_bwd,
        )
        bwd_out = torch.flip(bwd_out, dims=[2])

        merged = fwd_out + bwd_out  # [B*d_inner, d_spectral, 1]
        # Normalize over d_spectral dim
        merged = merged.squeeze(-1)  # [B*d_inner, d_spectral]
        merged = self.out_norm(merged)
        # Reshape back: [B*d_inner, d_spectral] → [B, d_inner, d_spectral] → [B, d_spectral, d_inner]
        merged = merged.reshape(batch_size, self.d_inner, self.d_spectral)
        merged = merged.transpose(1, 2).contiguous()  # [B, d_spectral, d_inner]

        # Interpolate back if adaptive pooling was used
        if hw != self.d_spectral:
            merged = merged.transpose(1, 2).contiguous()  # [B, d_inner, d_spectral]
            merged = F.interpolate(merged, size=hw, mode="linear", align_corners=False)
            merged = merged.transpose(1, 2).contiguous()  # [B, hw, d_inner]

        merged = merged.reshape(batch_size, height, width, self.d_inner)
        merged = merged * F.silu(z_part)
        return self.out_proj(merged)


class SpatialMoERouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        use_density_hint: bool = False,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_dim, num_experts, bias=False)
        self.density_proj = (
            nn.Linear(1, num_experts, bias=False) if use_density_hint else None
        )

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        logits = self.fc(pooled)
        if self.density_proj is not None and density_hint is not None:
            pooled_density = self.pool(density_hint).flatten(1)
            logits = logits + self.density_proj(pooled_density)
        return logits


class SpatialMoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        lr_space: str = "exp",
        use_density_hint: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        if self.num_experts > 4:
            raise ValueError(
                f"num_experts must be <= 4 (one per scan direction), got {self.num_experts}"
            )
        self.top_k = int(top_k)
        self.router = SpatialMoERouter(
            input_dim=input_dim,
            num_experts=self.num_experts,
            use_density_hint=use_density_hint,
        )

        if lr_space == "exp":
            low_dims = [2 ** (idx + 1) for idx in range(self.num_experts)]
        else:
            low_dims = [2 * (idx + 1) for idx in range(self.num_experts)]

        self.low_dims = low_dims
        self.experts = nn.ModuleList(
            [
                SingleScanSSM(
                    d_model=input_dim,
                    low_dim=low_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for low_dim in self.low_dims
            ]
        )

    def _generate_directions(self) -> tuple[int, ...]:
        return tuple(range(self.num_experts))

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training is None:
            training = self.training
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        router_logits = self.router(x_bchw, density_hint=density_hint)
        weights = torch.softmax(router_logits, dim=-1)

        if not training:
            k = min(self.top_k, self.num_experts)
            topk = torch.topk(weights, k=k, dim=-1)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(1, topk.indices, topk.values)
            weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        outputs = []
        for direction, expert in zip(self._generate_directions(), self.experts):
            outputs.append(expert(x, direction=direction))
        stacked = torch.stack(outputs, dim=1)
        mixed = torch.sum(stacked * weights[:, :, None, None, None], dim=1)
        return mixed, weights


class MoEBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        lr_space: str = "exp",
        use_density_hint: bool = False,
        d_spectral: int = 256,
    ) -> None:
        super().__init__()
        self.pre_proj = nn.Conv2d(input_dim, input_dim * 2, kernel_size=1, bias=False)
        self.post_proj = nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, bias=False)
        self.spatial_moe = SpatialMoELayer(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            d_state=d_state,
            d_conv=d_conv,
            lr_space=lr_space,
            use_density_hint=use_density_hint,
        )
        self.channel_ssm = BiDirectionalChannelSSM(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            d_spectral=d_spectral,
        )

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training is None:
            training = self.training
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        split = self.pre_proj(x_bchw)
        spatial_branch, channel_branch = split.chunk(2, dim=1)

        spatial_branch_bhwc = spatial_branch.permute(0, 2, 3, 1).contiguous()
        channel_branch_bhwc = channel_branch.permute(0, 2, 3, 1).contiguous()

        spatial_out, weights = self.spatial_moe(
            spatial_branch_bhwc,
            density_hint=density_hint,
            training=training,
        )
        channel_out = self.channel_ssm(channel_branch_bhwc)
        fused = torch.cat([channel_out, spatial_out], dim=-1)
        fused = fused.permute(0, 3, 1, 2).contiguous()
        fused = self.post_proj(fused)
        return fused.permute(0, 2, 3, 1).contiguous(), weights


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoMEB(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        lr_space: str = "exp",
        mlp_hidden: int = 256,
        drop_path: float = 0.0,
        use_density_hint: bool = False,
        d_spectral: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.block = MoEBlock(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            lr_space=lr_space,
            use_density_hint=use_density_hint,
            d_spectral=d_spectral,
        )
        self.mlp = _Mlp(input_dim, mlp_hidden)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.skip_scale1 = nn.Parameter(torch.ones(input_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(input_dim))

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training is None:
            training = self.training
        moe_out, weights = self.block(
            self.norm1(x),
            density_hint=density_hint,
            training=training,
        )
        x = x + self.drop_path1(self.skip_scale1.view(1, 1, 1, -1) * moe_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path2(self.skip_scale2.view(1, 1, 1, -1) * mlp_out)
        return x, weights


class MambaMoEFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        num_experts: int = 4,
        top_k: int = 2,
        lr_space: str = "exp",
        num_blocks: int = 1,
        mlp_hidden: int = 256,
        drop_path: float = 0.1,
        lambda_balance: float = 0.01,
        use_density_hint: bool = False,
        d_spectral: int = 256,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MoMEB(
                    input_dim=input_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    lr_space=lr_space,
                    mlp_hidden=mlp_hidden,
                    drop_path=drop_path,
                    use_density_hint=use_density_hint,
                    d_spectral=d_spectral,
                )
                for _ in range(num_blocks)
            ]
        )
        self.balance_loss = MambaMoEBalanceLoss(lambda_balance=lambda_balance)

    def forward(
        self,
        x: torch.Tensor,
        density_hint: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if training is None:
            training = self.training
        hidden = x.permute(0, 2, 3, 1).contiguous()
        collected_weights = []
        for block in self.blocks:
            hidden, weights = block(
                hidden, density_hint=density_hint, training=training
            )
            collected_weights.append(weights)

        mean_weights = torch.stack(collected_weights, dim=0).mean(dim=0)
        aux_losses = self.balance_loss(mean_weights)
        output = hidden.permute(0, 3, 1, 2).contiguous()
        return output, aux_losses, mean_weights


__all__ = [
    "BiDirectionalChannelSSM",
    "MambaMoEBalanceLoss",
    "MambaMoEFusion",
    "MoEBlock",
    "MoMEB",
    "SingleScanSSM",
    "SpatialMoELayer",
    "SpatialMoERouter",
]

import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange, repeat
import collections.abc
from thop import profile
import os
import math
from timm.models.layers import DropPath
from functools import partial
from typing import Callable
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch import Tensor


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif hasattr(x, '__iter__'):
        return tuple(x)
    else:
        raise ValueError("Unsupported input type for to_2tuple")

to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

# FFN with single kernel
class FFN(nn.Module):
    def __init__(self, dim, bias,kernel_size):
        super(FFN, self).__init__()
        if kernel_size not in [3, 5, 7]:
            raise ValueError("Invalid kernel_size. Must be 3, 5, or 7.")

        self.kernel_size = kernel_size
        hidden_features = 180

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,groups=hidden_features, bias=bias) #dwconv
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if self.kernel_size == 3:
            x = self.relu3(self.dwconv3x3(x))
        elif self.kernel_size == 5:
            x = self.relu3(self.dwconv5x5(x))
        elif self.kernel_size == 7:
            x = self.relu3(self.dwconv7x7(x))
        x = self.project_out(x)

        return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=4,squeeze_factor=8):
        super(CAB, self).__init__()
        if is_light_sr: # we use depth-wise conv for light-SR to achieve more efficient
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else: # for classic SR
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Selective_Scan_Spa(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False, device=None, dtype=None, mode='Spa', **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.mode = mode

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.mode == 'Spe':
            self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                  groups=1, bias=conv_bias, **factory_kwargs)
        elif self.mode == 'Spa':
            self.conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2,
                                  groups=self.d_inner, bias=conv_bias, **factory_kwargs)
        else:
            raise ValueError("mode must be one of ['Spe', 'Spa']")

        self.act = nn.SiLU()

        # projection weights
        self.x_proj = nn.ModuleList([nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs) for _ in range(4)])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = nn.ModuleList([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn  # You should define or import this externally

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # mode-aware enhancement
        if self.mode == 'Spe':
            x_spectral = x.view(B, C, L)
            x_spectral = self.act(self.conv(x_spectral))
            x = x_spectral.view(B, C, H, W)
        elif self.mode == 'Spa':
            x = self.act(self.conv(x))

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1)
        x_hwwh = x_hwwh.view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  # to (B, C, H, W)
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class VSSBlock_Spa(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            mode= 'Spa',
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Selective_Scan_Spa(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,mode=mode, **kwargs)  # 2D-SSM 'Spa' or 'Spe'

        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

        self.conv = nn.Conv2d(hidden_dim,hidden_dim,3,1,1)

        self.proj = nn.Sequential(
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 16, kernel_size=1),
            # nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 16, 1, kernel_size=1)
        )
        self.ca = CAB(num_feat=hidden_dim)

        self.ln_3 = norm_layer(hidden_dim)
        self.ffn = FeedForward(hidden_dim)

    def forward(self, input):
        B, C, H, W = input.shape # B,C,H,W
        L = H * W
        x = input
        # print('!!!!!!', x.shape)
        # part one
        conv_x = self.ca(self.dwconv(x)) # local channel branch # B,C,H,W

        # Spa-VSSM
        x1 = x.permute(0,2,3,1).contiguous()  # B,H,W,C
        attened_x = self.self_attention(self.ln_1(x1))  # B,H,W,C spatial branch
        attened_x = attened_x.permute(0,3,1,2).contiguous() # B,C,H,W

        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(attened_x)  # 1,1,64,64  H*W

        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)  # 1,1,180  1*C

        # C-I
        attened_x = attened_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # B,L,C
        attened_x = attened_x * torch.sigmoid(channel_map)  # [1,4096,180] * [1,1,180] -> [1,4096,180]

        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x  # [1,1,64,64]  * [1,180,64,64] ->  [1,180,64,64]
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # 1,4096,180

        x = (x1 * self.skip_scale).view(B,L,C).contiguous() + attened_x + conv_x   # B,L,C
        x = x.view(B,H,W,C).contiguous()   #B,H,W,C

        # part two
        x = x * self.skip_scale2 + self.conv(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                    1).contiguous()  # B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()  # B,C,H,W

        x = x + self.ffn(self.ln_3(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous())

        return x


# # class PHDMamba(nn.Module):
# #     def __init__(self,
# #                  num_features=3,
# #                  embed_dim=64,
# #                  img_size=128,
# #                  patch_size=1,
# #                  norm_layer=nn.LayerNorm,
# #                  depth=4,
# #                  drop_rate=0.,
# #                  num_classes=9,
# #                  group_num=4,
# #                  patch_norm = True
# #                  ):
# #         super(PHDMamba, self).__init__()

# #         self.patch_norm = patch_norm
# #         self.conv0 = nn.Conv2d(num_features, embed_dim, 1)
# #         self.bn = nn.BatchNorm2d(embed_dim)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.patch_embed = PatchEmbed(
# #             img_size=img_size,
# #             patch_size=patch_size,
# #             in_chans=embed_dim,
# #             embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #         self.pos_drop = nn.Dropout(p=drop_rate)

# #         self.patch_embedding = nn.Sequential(
# #             nn.Conv2d(in_channels=num_features, out_channels=embed_dim, kernel_size=1, stride=1, padding=0),
# #             nn.GroupNorm(group_num, embed_dim),
# #             nn.SiLU())

# #         self.phdmambablock_1 = nn.Sequential(
# #             VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa'),
# #         )
# #         self.conv_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
# #         self.phdmambablock_2 = nn.Sequential(
# #             VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16,
# #                          mode='Spa'),
# #         )
# #         self.conv_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
# #         self.phdmambablock_3 = nn.Sequential(
# #             VSSBlock_Spa(hidden_dim=embed_dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16,
# #                          mode='Spa'),
# #         )

# #         self.cls_head = nn.Sequential(nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
# #                                       nn.GroupNorm(group_num,128),
# #                                       nn.SiLU(),
# #                                       nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

# #     def forward(self, x):
# #         interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4]) #b,c,h,w
# #         x = self.patch_embedding(x) # b,c,h,w
# #         _,_,H,W = x.size()
# #         layer1 = self.phdmambablock_1(x)  # 256,256
# #         layer2 = self.phdmambablock_2(self.conv_1(layer1)) # 128，128
# #         layer3 = self.phdmambablock_3(self.conv_2(layer2))  # 64，64

# #         x = interpolation(layer3)
# #         x = self.cls_head(x)

# #         return x

# class RestorationBranch(nn.Module):
#     """
#     (b) Restoration Branch (RB) 
#     结构：3次下采样 -> 1个中间层 -> 3次上采样
#     特征提取器：VSSBlock_Spa (原 TFB)
#     """
#     def __init__(self, dim):
#         super().__init__()
#         # --- Encoder (下采样阶段) ---
#         self.enc1 = VSSBlock_Spa(dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
#         self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1) # 1/2
        
#         self.enc2 = VSSBlock_Spa(dim*2, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
#         self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1) # 1/4
        
#         self.enc3 = VSSBlock_Spa(dim*4, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
#         self.down3 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1) # 1/8

#         # --- Bottleneck (中间层) ---
#         self.mid_block = nn.Sequential(VSSBlock_Spa(dim * 8),VSSBlock_Spa(dim * 8),VSSBlock_Spa(dim * 8))

#         # --- Decoder (上采样阶段) ---
#         # 使用 PixelShuffle 或 ConvTranspose2d，这里以 ConvTranspose 为例
#         self.up3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
#         self.dec3 = VSSBlock_Spa(dim*4, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
        
#         self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
#         self.dec2 = VSSBlock_Spa(dim*2, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
        
#         self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
#         self.dec1 = VSSBlock_Spa(dim, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')

#     def forward(self, x):
#         # Encoder
#         s1 = self.enc1(x)
#         x = self.down1(s1)
        
#         s2 = self.enc2(x)
#         x = self.down2(s2)
        
#         s3 = self.enc3(x)
#         x = self.down3(s3)

#         # Middle
#         x = self.mid_block(x)

#         # Decoder + Skip Connections (跳跃连接以保持细节)
#         x = self.up3(x)
#         x = self.dec3(x + s3) # 与 Encoder 对应的层相加
        
#         x = self.up2(x)
#         x = self.dec2(x + s2)
        
#         x = self.up1(x)
#         x = self.dec1(x + s1)
        
#         return x

# ##########################################################################
# ##---------- Prompt Gen Module -----------------------
# class PromptGenBlock(nn.Module):
#     def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
#         super(PromptGenBlock,self).__init__()
#         self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
#         self.linear_layer = nn.Linear(lin_dim,prompt_len)
#         self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

#     def forward(self,x):
#         B,C,H,W = x.shape
#         emb = x.mean(dim=(-2,-1))
#         prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
#         prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
#         prompt = torch.sum(prompt,dim=1)
#         prompt = F.interpolate(prompt,(H,W),mode="bilinear")
#         prompt = self.conv3x3(prompt)

#         return prompt

# class ActiveBlock_Prompt_Basis(nn.Module):
#     def __init__(self, dim=128, num_task=6, num_basis=8, embed_dim=64):
#         super().__init__()
#         self.basis_mlp = Mlp(dim, hidden_features=num_basis, out_features=num_basis)
#         # self.basis_mlp = nn.Linear(dim, num_basis)
#         self.task_mlp = Mlp(dim, hidden_features=num_task, out_features=num_task)
#         # self.task_mlp = nn.Linear(dim, num_task)
#         self.num_basis = num_basis
#         self.num_task = num_task
#         self.dim = dim

#         self.prompt = nn.Parameter(torch.rand(1, num_task, num_basis, embed_dim, 1, 1))
#         self.conv3x3 = nn.Conv2d(embed_dim,embed_dim,kernel_size=3,stride=1,padding=1,bias=False)
    
#     def forward(self, x):
#         '''
#         x: image features [B, C, H, W]
#         prompt: [1, L, B, C, 1, 1]
#         '''
#         B,C,H,W = x.shape
#         x = x.permute(0, 2, 3, 1)
#         x_basis = F.softmax(self.basis_mlp(x).permute(0, 3, 1, 2).contiguous(), dim=1)
#         # print('!123321!', x_basis.shape)
#         x_task = F.softmax(self.task_mlp(x).permute(0, 3, 1, 2).contiguous(), dim=1)
#         prompts = x_basis.unsqueeze(1).unsqueeze(-3) * self.prompt.unsqueeze(0).repeat(B,1,1,1,1,1,1).squeeze(1)  # B, C, H, W -> B, 1, C, 1, H, W
#         # prompts = self.prompt.unsqueeze(0).repeat((b, 1, 1, 1, 1))  # B, nb, 1, 1, C
#         prompts = torch.sum(prompts,dim=2) # B, 1, 1, H, W
#         prompts = x_task.unsqueeze(-3) * prompts 
#         prompts = torch.sum(prompts,dim=1)
#         prompts = self.conv3x3(prompts)
#         return prompts

# class PIM(nn.Module):
#     def __init__(self,dim):
#         super(PIM,self).__init__()
#         self.prompt = ActiveBlock_Prompt_Basis(dim=dim, num_task=7, num_basis=8, embed_dim=dim)
#         self.vb = VSSBlock_Spa(dim*2, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16, mode='Spa')
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0),
#             nn.SiLU(),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1))

#     def forward(self,x):
#         x1 = self.prompt(x)
#         x2 = torch.cat([x, x1], dim=1)
#         # print('!!!!!!', x.shape, x1.shape, x2.shape)
#         x3 = self.vb(x2)
#         x4 = self.conv_last(x3)
#         return x4

# class DualRouting(nn.Module):
#     """
#     Dual Routing expert selection.
#     - Group 0: num_experts / 2 experts
#     - Group 1: num_experts / 2 experts
#     - group_idx is encoded and added to the input for expert selection.
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_experts: int,
#         capacity_factor: float = 1.0,
#         epsilon: float = 1e-6,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor
#         self.epsilon = epsilon

#         # Group sizes
#         self.group_size = num_experts // 2

#         # First-stage gate: selects a group (2 options)
#         self.w_group_gate = nn.Linear(dim, 2)

#         # Second-stage gates: selects an expert within the chosen group
#         self.w_expert_gates = nn.ModuleList([
#             nn.Linear(dim, self.group_size),  # Group 0
#             nn.Linear(dim, self.group_size),  # Group 1
#         ])

#         # Embedding for group_idx (2 groups)
#         self.group_embedding = nn.Embedding(2, dim)  # 2 groups, each with a dim-dimensional embedding

#     def forward(self, x: Tensor, use_aux_loss=False):
#         """
#         Forward pass of the modified DualRouting module.

#         Args:
#             x (Tensor): Input tensor of shape (B, C, H, W).

#         Returns:
#             Tensor: Gate scores of shape (B, H, W, num_experts).
#             None: Placeholder for auxiliary loss.
#         """
#         B, C, H, W = x.shape
#         x = rearrange(x, 'b c h w -> (b h w) c', b=B, h=H, w=W, c=C)

#         # First-stage: select a group
#         group_scores = F.softmax(self.w_group_gate(x), dim=-1)  # (B*H*W, 2)
#         _, top_group_indices = group_scores.topk(1, dim=-1)  # (B*H*W, 1)

#         # Initialize one-hot encoding
#         one_hot = torch.zeros(x.size(0), self.num_experts, device=x.device)  # (B*H*W, num_experts)

#         # Handle all groups
#         for group_idx in range(2):  # Groups 0-1
#             group_mask = (top_group_indices == group_idx).squeeze(-1)  # Mask for current group
#             if group_mask.any():
#                 # Get group_idx embedding and add it to x[group_mask]
#                 group_idx_tensor = torch.tensor(group_idx, device=x.device).long()  # Convert to tensor
#                 group_embed = self.group_embedding(group_idx_tensor)  # (1, dim)
#                 group_embed = group_embed.expand(x[group_mask].size(0), -1)  # (N, dim)

#                 # Add group embedding to the input
#                 x_group = x[group_mask] + group_embed  # (N, dim)

#                 # Get expert scores for the current group
#                 expert_scores = F.softmax(self.w_expert_gates[group_idx](x_group), dim=-1)  # (N, group_size)
#                 _, expert_indices = expert_scores.topk(1, dim=-1)  # (N, 1)

#                 # Map to global expert indices
#                 global_expert_indices = group_idx * self.group_size + expert_indices.squeeze(-1)
#                 one_hot[group_mask, global_expert_indices] = 1  # Set the selected experts

#         # Reshape back to (B, H, W, num_experts)
#         one_hot = rearrange(one_hot, '(b h w) n -> b h w n', b=B, h=H, w=W, n=self.num_experts)

#         return one_hot, None

# #############################################################
# class ResidualAmplitudePhaseBlock(nn.Module):
#     """
#     残差幅度/相位模块 (Residual Amplitude/Phase Block)
#     结构：空域分支 + 频域分支（幅度/相位分离处理）+ 拼接融合
#     输入:
#         x: 输入特征图 (B, C, H, W)
#     输出:
#         out: 输出特征图 (B, C, H, W)，与输入尺寸一致
#     """
#     def __init__(self, in_channels, out_channels=None):
#         super().__init__()
#         if out_channels is None:
#             out_channels = in_channels
        
#         # -------------------------- 空域分支 (Spatial Domain Branch) --------------------------
#         # 主路径: Conv3x3 -> ReLU -> Conv3x3 -> ReLU
#         self.spatial_main = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
#             nn.ReLU(inplace=True)
#         )
#         # 捷径路径: Conv1x1 (匹配维度，实现残差连接)
#         self.spatial_shortcut = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        
#         # -------------------------- 频域分支 (Frequency Domain Branch) --------------------------
#         # 幅度处理路径: Conv1x1 -> ReLU -> Conv1x1
#         self.amp_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#         )
#         # 相位处理路径: Conv1x1 -> ReLU -> Conv1x1
#         self.phase_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
#         )
        
#         # -------------------------- 融合模块 (Fusion Module) --------------------------
#         # 拼接后 1x1 卷积融合空域与频域特征
#         # self.fusion_conv = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=True)
#         self.reduction = 16  # 通道降维系数，减少参数量
#         self.weight_generator = nn.Sequential(
#             # 全局平均池化：(B, C, H, W) → (B, C, 1, 1)，提取全局空间信息
#             nn.AdaptiveAvgPool2d(1),
#             # 降维：减少计算量
#             nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1, bias=True),
#             nn.ReLU(inplace=True),
#             # 升维：生成最终权重（2个权重对应两个模块）
#             nn.Conv2d(in_channels // self.reduction, 2, kernel_size=1, bias=True),
#             # 展平：(B, 2, 1, 1) → (B, 2)
#             nn.Flatten()
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # -------------------------- 1. 空域分支计算 --------------------------
#         spatial_main_out = self.spatial_main(x)   # 主路径特征
#         spatial_shortcut_out = self.spatial_shortcut(x)  # 捷径特征
#         f_spa = spatial_main_out + spatial_shortcut_out  # 残差融合得到空域特征 F_spa
        
#         # -------------------------- 2. 频域分支计算 --------------------------
#         # 2.1 FFT: 空域 -> 频域
#         fft_out = torch.fft.fft2(f_spa, dim=(-2, -1), norm='ortho')  # 正交归一化FFT
#         amp = torch.abs(fft_out)          # 幅度谱 (Amplitude)
#         phase = torch.angle(fft_out)     # 相位谱 (Phase)
        
#         # 2.2 分别处理幅度和相位
#         amp_out = self.amp_conv(amp)
#         phase_out = self.phase_conv(phase)
        
#         # 2.3 重构复数频域特征 + IFFT: 频域 -> 空域
#         fft_recon = amp_out * torch.exp(1j * phase_out)  # 重构复数谱
#         f_fre = torch.fft.ifft2(fft_recon, dim=(-2, -1), norm='ortho').real  # IFFT取实部
        
#         # -------------------------- 3. 拼接融合 --------------------------
#         # concat_feat = torch.cat([f_spa, f_fre], dim=1)  # 通道维度拼接空域+频域特征
#         # out = self.fusion_conv(concat_feat)            # 1x1卷积融合得到最终输出
#         weights = self.weight_generator(x)  # (B, 2)
#         weights = torch.softmax(weights, dim=1)  # 归一化：α + β = 1，(B, 2)
#         alpha = weights[:, 0].view(B, 1, 1, 1)  # 左模块权重 (B, 1, 1, 1)
#         beta = weights[:, 1].view(B, 1, 1, 1)   # 右模块权重 (B, 1, 1, 1)
#         out = alpha * f_spa + beta * f_fre
        
#         return out

# class HMoE(nn.Module):
#     """
#     A module that implements the Switched Mixture of Experts (MoE) architecture.

#     Args:
#         dim (int): The input dimension.
#         hidden_dim (int): The hidden dimension of the feedforward network.
#         output_dim (int): The output dimension.
#         num_experts (int): The number of experts in the MoE.
#         capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
#         mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Attributes:
#         dim (int): The input dimension.
#         hidden_dim (int): The hidden dimension of the feedforward network.
#         output_dim (int): The output dimension.
#         num_experts (int): The number of experts in the MoE.
#         capacity_factor (float): The capacity factor that controls the capacity of the MoE.
#         mult (int): The multiplier for the hidden dimension of the feedforward network.
#         experts (nn.ModuleList): The list of feedforward networks representing the experts.
#         gate (DualRouting): The switch gate module.

#     """

#     def __init__(
#         self,
#         dim: int,
#         output_dim: int,
#         num_experts: int,
#         capacity_factor: float = 1.0,
#         use_aux_loss: bool = False,
#         use_freq: bool = False,
#         freq_method: str = 'fft',      # 新增：选择频域方法 'fft' 或 'dct'
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.output_dim = output_dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor
#         self.use_aux_loss = use_aux_loss
#         self.use_freq = use_freq
#         self.freq_method = freq_method

#         self.experts = nn.ModuleList()

#         for _ in range(int(num_experts/2)):
#             self.experts.append(nn.Conv2d(dim, output_dim, kernel_size=1, stride=1, padding=0))

#         for _ in range(int(num_experts/2)):
#             self.experts.append(nn.Conv2d(dim, output_dim, kernel_size=3, stride=1, padding=1))

#         self.gate = DualRouting(
#             dim,
#             num_experts,
#             capacity_factor,
#         )

#         # 频域融合模块（若启用）
#         if self.use_freq:
#             self.freq_fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)
        
#         self.fre = ResidualAmplitudePhaseBlock(dim)

#     def _extract_freq_features_fft(self, x):
#         # """
#         # 使用 FFT 提取低频和高频分量
#         # x: (B, C, H, W)
#         # return: (low_freq, high_freq) 均为 (B, C, H, W)
#         # """
#         # B, C, H, W = x.shape
#         # # 转换为浮点类型并执行 FFT
#         # x_fft = torch.fft.rfft2(x.float())  # (B, C, H, W//2+1) 复数

#         # # 构造低通掩码：保留中心区域（低频），其余置零
#         # # 中心频率坐标：H//2, 0（因为 rfft 的最后一维是半平面）
#         # # 定义截止频率 radius（可设置为 min(H,W)//4 或可学习参数，这里固定为 H//4）
#         # radius = min(H, W) // 4
#         # # 生成网格坐标
#         # fy = torch.arange(H, device=x.device).view(-1, 1).float()
#         # fx = torch.arange(W // 2 + 1, device=x.device).view(1, -1).float()
#         # # 计算距离中心的距离（注意频率中心需要根据 FFT 的排列调整）
#         # center_y = H // 2
#         # center_x = 0  # rfft 的低频分量位于 (0,0) 附近？需要确认：rfft2 输出中，低频分量位于 (0,0) 和 (0, W//2) 周围，中心实际上是 (0,0) 和 (0, W//2) 的连线。简单起见，我们使用一个圆形的低通掩码以 (0,0) 为中心可能不合适。
#         # # 更准确的方法：直接使用一个矩形低通掩码，保留最低的几个频率分量。
#         # # 简化：保留前 k 个频率系数（按行优先），但会导致各向异性。
#         # # 为简化实现，我们采用高斯低通滤波（在频域乘高斯核），避免坐标问题。
#         # # 但为了清晰演示，这里采用简单的理想低通：保留中心矩形区域。
#         # # 构造掩码：保留低频范围 [ -radius:radius, -radius:radius ]，考虑到 rfft 布局，需要适当处理。
#         # # 由于实现较复杂，我们使用 torch.fft.fftshift 处理全平面 FFT？但 rfft 没有直接 shift。
#         # # 建议使用 torch.fft.fft2 得到全平面 FFT，然后使用 shift 和掩码。
#         # # 为降低复杂度，我们直接使用 torch.fft.fft2 和 torch.fft.ifft2。

#         # # 使用全平面 FFT（复数）
#         # x_fft_full = torch.fft.fft2(x.float())  # (B, C, H, W)
#         # # 中心化
#         # x_fft_shift = torch.fft.fftshift(x_fft_full, dim=(-2, -1))

#         # # 构建低通掩码
#         # mask = torch.zeros((H, W), device=x.device, dtype=torch.float32)
#         # center_h, center_w = H // 2, W // 2
#         # mask[center_h - radius:center_h + radius + 1, center_w - radius:center_w + radius + 1] = 1.0
#         # mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

#         # # 低通 FFT = 原 FFT * 掩码
#         # low_fft_shift = x_fft_shift * mask
#         # # 高通 FFT = 原 FFT * (1 - 掩码)
#         # high_fft_shift = x_fft_shift * (1 - mask)

#         # # 逆 shift 并逆 FFT
#         # low_fft = torch.fft.ifftshift(low_fft_shift, dim=(-2, -1))
#         # high_fft = torch.fft.ifftshift(high_fft_shift, dim=(-2, -1))

#         # low_freq = torch.fft.ifft2(low_fft).real
#         # high_freq = torch.fft.ifft2(high_fft).real

#         # # 确保输出类型与输入一致
#         # low_freq = low_freq.to(x.dtype)
#         # high_freq = high_freq.to(x.dtype)

#         # return low_freq, high_freq

#         #####################################################################
#         x_fft = self.fre(x)

#         return x_fft
#         #####################################################################

        

#     def _extract_freq_features_dct(self, x):
#         """
#         使用 DCT 提取低频和高频分量
#         需要安装或实现 DCT，这里给出伪代码
#         """
#         # 可以使用 scipy.fftpack.dct 或自定义，但 PyTorch 无内置
#         # 为保持可运行，我们暂用 FFT 替代，或提示需要第三方库
#         raise NotImplementedError("DCT not implemented, use FFT instead or install 'dct' package.")

#     def forward(self, x: Tensor, x_t: Tensor):
#         """
#         Forward pass of the HMoE module.

#         Args:
#             x (Tensor): The input tensor.

#         Returns:
#             Tensor: The output tensor of the MoE.

#         """
#         # (batch_size, seq_len, num_experts)
#         B, N, H, W = x.shape

#         # 频域特征提取与融合
#         if self.use_freq:
#             if self.freq_method == 'fft':
#                 x_fft = self._extract_freq_features_fft(x_t)
#             elif self.freq_method == 'dct':
#                 x_fft = self._extract_freq_features_dct(x_t)
#             else:
#                 raise ValueError(f"Unknown freq_method: {self.freq_method}")

#             # # 拼接空间特征、低频、高频
#             # fused = torch.cat([x_t, x_fft], dim=1)  # (B, 3*C, H, W)
#             # gate_input = self.freq_fusion(fused)  # (B, C, H, W)
#             gate_input = x_t + x_fft
#         else:
#             gate_input = x_t

#         gate_scores, loss = self.gate(
#             gate_input, use_aux_loss=self.use_aux_loss
#         )

#         # Dispatch to experts
#         expert_outputs = [rearrange(expert(x), 'b c h w -> (b h w) c', b=B, h=H, w=W, c=self.output_dim) for expert in self.experts]

#         # Check if any gate scores are nan and handle
#         if torch.isnan(gate_scores).any():
#             print("NaN in gate scores")
#             gate_scores[torch.isnan(gate_scores)] = 0

#         # Stack and weight outputs
#         stacked_expert_outputs = torch.stack(
#             expert_outputs, dim=-1
#         )  # (batch_size, seq_len, output_dim, num_experts)
#         if torch.isnan(stacked_expert_outputs).any():
#             stacked_expert_outputs[
#                 torch.isnan(stacked_expert_outputs)
#             ] = 0

#         # Combine expert outputs and gating scores
#         moe_output = torch.sum(
#             rearrange(gate_scores, 'b h w n -> (b h w) n', b=B, h=H, w=W, n=self.num_experts).unsqueeze(-2) * stacked_expert_outputs, dim=-1
#         )
#         moe_output = rearrange(moe_output, '(b h w) c -> b c h w ', b=B, h=H, w=W, c=self.output_dim)
#         return moe_output


# class MoEUpsample(nn.Sequential):
#     """MoEUpsample module.

#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#         num_experts (int): Number of experts in the HMoE layer.
#     """

#     def __init__(self, scale, num_feat, num_experts):
#         m = []
#         m.append(HMoE(num_feat, scale * scale * num_feat, num_experts, use_freq=False, freq_method='fft'))
#         m.append(nn.PixelShuffle(scale))
#         super(MoEUpsample, self).__init__(*m)

#     def forward(self, x, x_t):
#         for module in self:
#             if isinstance(module, HMoE):
#                 x = module(x, x_t)
#             else:
#                 x = module(x)
#         return x


# # 整体网络结构 (Strictly following the diagram)
# class PHMNet(nn.Module):
#     def __init__(self, dim=64, num_rbs=3, upscale=4, num_experts=8):
#         super().__init__()

#         # 初始特征提取
#         self.extraction = nn.Conv2d(3, dim, 3, padding=1)
        
#         # 级联的 RB 和 PM 模块
#         self.pms = nn.ModuleList([PIM(dim) for _ in range(num_rbs)])
#         self.rbs = nn.ModuleList([RestorationBranch(dim) for _ in range(num_rbs)])
        
#         # 频率分量输出层 (conv + LReLU)
#         self.imf_heads = nn.ModuleList([
#             nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.LeakyReLU(0.2))
#             for _ in range(num_rbs)
#         ])
        
#         # # 最后的重建层
#         # self.pixel_shuffle = nn.Sequential(
#         #     nn.Conv2d(dim, 3 * (upscale ** 2), 3, padding=1), 
#         #     nn.PixelShuffle(upscale)
#         # )
#         self.upsample = MoEUpsample(upscale, dim, num_experts)
#         self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)

#         self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.conv_before_upsample = nn.Sequential(
#                 nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(inplace=True))

#     def forward(self, lr):
        
#         feat = self.extraction(lr)
#         feat0 = feat
        
#         imf_accumulator = 0
#         # 2. 级联 RB 与 PM 交互逻辑
#         for i in range(len(self.rbs)):
#             # PM 动态调制 
#             # print('!!!!!!!!!!!!!!!', feat.shape)
#             mod_signal = self.pms[i](feat)
            
#             # RB 恢复处理，注入 PM 信号
#             feat = self.rbs[i](feat + mod_signal)
            
#             # 分支预测 IMF 频率分量并累加
#             imf = self.imf_heads[i](feat)
#             imf_accumulator = imf_accumulator + imf
        
#         feat = self.conv_after_body(feat) + feat0
#         feat = self.conv_before_upsample(feat)

#         fea = self.upsample(feat, imf)
#         final_out = self.conv_last(fea)
        
#         # # 3. 汇总特征并进行 Pixel Shuffle 重建
#         # final_out = self.pixel_shuffle(feat)

#         return final_out
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input = torch.rand(4,20,64,64).cuda()
    model = VSSBlock_Spa(hidden_dim=20, drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=2., d_state=16,
                         mode='Spa').cuda()
    output= model(input)
    print(output.size())

    flops, params = profile(model, inputs=(input,))
    print('Param:{} K' .format(params/1e3))
    print('Flops:{} G' .format(flops/1e9))  ## 打印计算量

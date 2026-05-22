import torch
import torch.nn as nn
import math

##### FreDFT: Frequency Domain Fusion Transformer for Visible-Infrared Object Detection
#  local feature enhancement module (LFEM)

# -------------------------- 对应结构图【Df-Conv】可变形卷积V2模块 --------------------------
class DeformConv2d(nn.Module):
    """
    带调制的可变形卷积V2（Deformable ConvNets v2）
    对应结构图绿色Df-Conv模块，核心功能：自适应学习卷积核的采样偏移，精准捕捉形变目标、不规则边缘的局部特征
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 主卷积层，对偏移后的特征做卷积
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # 偏移量预测卷积：输出每个采样点的x/y偏移
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        # 调制门控（V2核心）：学习每个采样点的权重
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        """偏移量学习率缩放，保证训练稳定性"""
        grad_input = tuple(grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = tuple(grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # 预测采样偏移量
        offset = self.p_conv(x)
        # 预测调制权重（V2）
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # 计算最终采样坐标
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        # 双线性插值的四个邻域坐标
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 裁剪坐标避免越界
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # 双线性插值核计算
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 获取四个邻域的特征值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 双线性插值聚合特征
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 调制权重加权（V2核心）
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 特征维度重整，适配主卷积
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    # 生成卷积核的基准采样坐标
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    # 生成每个像素的基准中心坐标
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 生成最终采样坐标 = 基准坐标 + 核偏移 + 学习到的偏移量
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    # 根据采样坐标获取对应特征
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    # 重整偏移特征，适配卷积输入格式
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset

# -------------------------- 对应结构图【Channel Shuffle】通道混洗模块 --------------------------
def channel_shuffle(x, groups):
    """
    通道混洗操作，实现跨分支的特征信息交互，打破通道冗余
    对应结构图Channel Shuffle模块，核心功能：打乱通道顺序，让不同分支的特征充分交互
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # 通道分组-转置-展平，实现混洗
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

# -------------------------- 基础卷积工具函数 --------------------------
def autopad(k, p=None):
    """自动计算padding，保证输出尺寸与输入一致"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """
    标准CBS卷积块：Conv + BatchNorm + SiLU
    对应结构图中所有黄色Conv模块，是LFEM的基础卷积单元
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -------------------------- LFEM完整模块 对应结构图全流程 --------------------------
class LFEM(nn.Module):
    """
    LFEM: Local Feature Enhancement Module 局部特征增强模块
    完整对应结构图全流程，核心功能：通过多分支多感受野卷积，捕捉多尺度、形变、细节特征，实现局部特征的全面增强
    """
    def __init__(self, in_channels):
        super(LFEM, self).__init__()
        # 输入1×1卷积，对应结构图最左侧Conv，调整通道、降低计算量
        self.CBSk1 = Conv(in_channels, in_channels, 1, 1)
        # 分支1：标准3×3卷积，对应结构图黄色Conv，捕捉常规局部特征
        self.CBSk3 = Conv(in_channels, in_channels, 3, 1)
        # 分支2：空洞卷积（D-Conv），对应结构图蓝色D-Conv，扩大感受野，捕捉大尺度/上下文特征
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        # 分支3：可变形卷积（Df-Conv），对应结构图绿色Df-Conv，捕捉形变目标、不规则边缘特征
        self.dfconv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, modulation=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU()
        # 分支4：深度可分离卷积（Dw-Conv），对应结构图橙色Dw-Conv，轻量化捕捉细节特征，降低参数量
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        # 特征融合1×1卷积，对应结构图最右侧Conv，将多分支特征融合为原始通道数
        self.CBS4C = Conv(4 * in_channels, in_channels, 1, 1)

        self.gate = GatedWeightGenerator(in_channels=in_channels, num_experts=4)

        self.ca = CALayer(in_channels)

        self.last = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x):
        rgb_fea = x  # 输入特征 [B, C, H, W]
        # 步骤1：1×1卷积预处理，统一通道维度
        rgb_fea0 = self.CBSk1(rgb_fea)
        gate_weights = self.gate(rgb_fea0)
        # 步骤2：四分支并行特征提取，对应结构图四个并行Conv分支
        rgb_fea1 = self.CBSk3(rgb_fea0)    # 标准3×3卷积分支
        rgb_fea2 = self.dconv(rgb_fea0)     # 空洞卷积分支
        rgb_fea3 = self.silu(self.bn(self.dfconv(rgb_fea0)))  # 可变形卷积分支
        rgb_fea4 = self.dwconv(rgb_fea0)    # 深度可分离卷积分支
        # 步骤3：多分支特征拼接，对应结构图C拼接环节
        # rgb_fea_cat = torch.cat([rgb_fea1, rgb_fea2, rgb_fea3, rgb_fea4], dim=1)
        rgb_fea_cat = rgb_fea1 * gate_weights[:,0,:,:,:] + rgb_fea2 * gate_weights[:,0,:,:,:] + rgb_fea3 * gate_weights[:,0,:,:,:] + rgb_fea4 * gate_weights[:,0,:,:,:]
        rgb_fea_cat = self.ca(rgb_fea_cat)
        new_rgb_fea = self.last(rgb_fea_cat)
        # # 步骤4：通道混洗，实现跨分支特征交互
        # rgb_fea_cat = channel_shuffle(rgb_fea_cat, 32)
        # # 步骤5：1×1卷积融合多分支特征
        # new_rgb_fea = self.CBS4C(rgb_fea_cat)
        # 步骤6：残差连接
        new_rgb = new_rgb_fea + rgb_fea
        return new_rgb

# -------------------------- 门控权重生成网络 --------------------------
class GatedWeightGenerator(nn.Module):
    """门控权重生成网络（对应图左侧模块）：生成各专家的权重"""
    def __init__(self, in_channels, num_experts=3):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # AAP: 自适应平均池化
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.mlp1 = nn.Linear(in_channels, in_channels // 2)  # 第一个MLP
        self.relu = nn.ReLU(inplace=True)
        self.mlp2 = nn.Linear(in_channels // 2, num_experts)  # 第二个MLP（输出专家数）
        self.softmax = nn.Softmax(dim=1)  # 归一化权重

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        out = self.aap(x) + self.amp(x)  # (bs, in_channels, 1, 1)
        out = out.flatten(1)  # 展平为 (bs, in_channels)
        out = self.mlp1(out)  # 压缩通道: (bs, in_channels//2)
        out = self.relu(out)
        out = self.mlp2(out)  # 输出专家权重: (bs, num_experts)
        gate_weights = self.softmax(out)  # 归一化权重
        # 扩展维度以匹配专家输出（后续逐元素乘）: (bs, num_experts, 1, 1, 1)
        return gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# 模块测试代码
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = LFEM(64).to(device)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
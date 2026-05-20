from crowdcount.plugins.gm import GateMechanism
from crowdcount.plugins.concat_gate_fusion import ConcatGateFusion
from crowdcount.plugins.clip_prompt_density import CLIPPromptDensityGuide
from crowdcount.plugins.depth_cross_attention import DepthCrossAttentionFusion
from crowdcount.plugins.depth_residual_gating import (
    DepthResidualGating,
    DepthResidualGatingV2,
)
from crowdcount.plugins.isfm.depth_fusion import DepthFusionModule
from crowdcount.plugins.mamba_moe import MambaMoEFusion
from crowdcount.plugins.moe import ESCA, MoE
from crowdcount.plugins.sdd_moe import SDDMoE
from crowdcount.plugins.msaa import (
    FPNAttentionGate,
    FPNSpatialAttention,
    MSAAGate,
    MSAALite,
    MsaaAdaptiveLayer,
    MSAA,
)

__all__ = [
    "ConcatGateFusion",
    "CLIPPromptDensityGuide",
    "DepthCrossAttentionFusion",
    "DepthResidualGating",
    "DepthResidualGatingV2",
    "GateMechanism",
    "DepthFusionModule",
    "MambaMoEFusion",
    "MsaaAdaptiveLayer",
    "MSAA",
    "MSAALite",
    "MSAAGate",
    "FPNAttentionGate",
    "FPNSpatialAttention",
    "ESCA",
    "MoE",
    "SDDMoE",
]

"""SA-DGAT: Scale-Aware Deformable Graph Attention Network for crowd counting."""

from crowdcount.plugins.sa_dgat.sa_dgat_fusion import SADGATFusion
from crowdcount.plugins.sa_dgat.scale_prompt import ScalePromptEmbedding
from crowdcount.plugins.sa_dgat.deformable_graph import DeformableGraphAttention
from crowdcount.plugins.sa_dgat.occlusion_gat import OcclusionAwareGAT
from crowdcount.plugins.sa_dgat.cross_scale_graph import CrossScaleGraphAggregation
from crowdcount.plugins.sa_dgat.subpixel_head import SubPixelDensityHead
from crowdcount.plugins.sa_dgat.bayesian_loss import BayesianCrowdLoss
from crowdcount.plugins.sa_dgat.ranking_loss import LocalCountRankingLoss

__all__ = [
    "SADGATFusion",
    "ScalePromptEmbedding",
    "DeformableGraphAttention",
    "OcclusionAwareGAT",
    "CrossScaleGraphAggregation",
    "SubPixelDensityHead",
    "BayesianCrowdLoss",
    "LocalCountRankingLoss",
]

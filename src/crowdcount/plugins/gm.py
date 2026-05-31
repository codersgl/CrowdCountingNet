import torch
import torch.nn as nn
import torch.nn.functional as F


class GateMechanism(nn.Module):
    """Gate Mechanism (global, legacy).

    Produces per-image fusion weights ``[B, num_streams]`` via global
    average pooling + MLP.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_streams: int = 3,
    ) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim, h, w)
        """
        x = self.aap(x).flatten(1)  # [batch_size, input_dim]
        x = self.activation(self.fc1(x))  # [batch_size, hidden_dim]
        x = self.fc2(x)  # [batch_size, num_streams]
        x = F.softmax(x, dim=-1)
        return x


class SpatialGateMechanism(nn.Module):
    """Spatial-aware gate mechanism.

    Produces per-pixel fusion weights ``[B, num_streams, H, W]`` using a
    lightweight convolutional head (3×3 for spatial coherence + 1×1 for
    channel projection).  Each spatial location independently decides
    how to blend the three feature streams (original, density-GCN,
    feature-GCN).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        num_streams: int = 3,
    ) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_streams, kernel_size=1, bias=True),
        )
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, H, W)

        Returns:
            gate weights: (B, num_streams, H, W), softmax-normalised over
            the *num_streams* dimension.
        """
        gate = self.gate_conv(x)  # [B, num_streams, H, W]
        return F.softmax(gate, dim=1)

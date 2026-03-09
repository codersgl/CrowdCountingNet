import torch
import torch.nn as nn
import torch.nn.functional as F


class GateMechanism(nn.Module):
    """Gate Mechanism"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim, h, w)
        """
        x = self.aap(x).flatten(1)  # [batch_size, input_dim]
        x = self.activation(self.fc1(x))  # [batch_size, hidden_dim]
        x = self.fc2(x)  # [batch_size, 3]
        x = F.softmax(x, dim=-1)
        return x

"""Homoscedastic uncertainty weighting for multi-task loss balancing.

Implements Kendall et al. (CVPR 2018) "Multi-Task Learning Using Uncertainty
to Weigh Losses for Scene Geometry and Semantics".

Each task *i* has a learnable log-variance ``s_i = log(σ_i²)``.  The weighted
loss becomes:

    L = Σ_i [ (1/2) exp(-s_i) L_i + (1/2) s_i ]

Using ``s_i`` (unconstrained) avoids explicit positivity constraints and is
numerically more stable than parameterising σ directly.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class UncertaintyWeighter(nn.Module):
    """Learnable multi-task loss weighter with three branches.

    Branches:
        - **den**: density map MSE loss
        - **ce**: classification (cross-entropy / focal) loss
        - **reg**: point regression (smooth-L1) loss

    Args:
        init_log_var_den: initial ``log(σ²)`` for density branch.
        init_log_var_ce: initial ``log(σ²)`` for classification branch.
        init_log_var_reg: initial ``log(σ²)`` for regression branch.
    """

    def __init__(
        self,
        init_log_var_den: float = 3.91,
        init_log_var_ce: float = -0.693,
        init_log_var_reg: float = 8.52,
    ) -> None:
        super().__init__()
        self.log_var_den = nn.Parameter(torch.tensor(init_log_var_den))
        self.log_var_ce = nn.Parameter(torch.tensor(init_log_var_ce))
        self.log_var_reg = nn.Parameter(torch.tensor(init_log_var_reg))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def forward(
        self,
        loss_den: torch.Tensor,
        loss_ce: torch.Tensor,
        loss_reg: torch.Tensor,
    ) -> torch.Tensor:
        """Return the uncertainty-weighted total loss.

        .. math::
            L = \\frac{1}{2} e^{-s_i} L_i + \\frac{1}{2} s_i
        """
        total = (
            0.5 * torch.exp(-self.log_var_den) * loss_den
            + 0.5 * self.log_var_den
            + 0.5 * torch.exp(-self.log_var_ce) * loss_ce
            + 0.5 * self.log_var_ce
            + 0.5 * torch.exp(-self.log_var_reg) * loss_reg
            + 0.5 * self.log_var_reg
        )
        return total

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """Return the *effective* weight ``1/(2σ²) = exp(-s)/2`` per branch."""
        with torch.no_grad():
            return {
                "w_den": (0.5 * torch.exp(-self.log_var_den)).item(),
                "w_ce": (0.5 * torch.exp(-self.log_var_ce)).item(),
                "w_reg": (0.5 * torch.exp(-self.log_var_reg)).item(),
            }

    def get_log_vars(self) -> dict[str, float]:
        """Return current ``log(σ²)`` values for logging."""
        with torch.no_grad():
            return {
                "log_var_den": self.log_var_den.item(),
                "log_var_ce": self.log_var_ce.item(),
                "log_var_reg": self.log_var_reg.item(),
            }

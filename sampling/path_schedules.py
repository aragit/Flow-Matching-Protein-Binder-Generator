"""
Probability Path definitions for Flow Matching.

Defines p_t(x|x1) and conditional vector field u_t(x|x1).
Supports linear, cosine, and polynomial schedules.
"""

import torch
import math


class ProbabilityPath:
    """
    Defines p_t(x|x1) and conditional vector field u_t(x|x1).
    Supports linear, cosine, and polynomial schedules.
    """

    def __init__(self, schedule: str = "linear"):
        self.schedule = schedule

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from conditional path p_t(x|x1).
        Args:
            x0: [batch, N, 3] noise
            x1: [batch, N, 3] data
            t: [batch, 1] time
        Returns:
            x_t: [batch, N, 3]
        """
        if self.schedule == "linear":
            # Standard OT path: x_t = t*x1 + (1-t)*x0
            return t.unsqueeze(1) * x1 + (1 - t).unsqueeze(1) * x0
        elif self.schedule == "cosine":
            # Cosine schedule from Ho et al. adapted to flows
            alpha_t = torch.cos(t * math.pi / 2).unsqueeze(1)
            return alpha_t * x1 + (1 - alpha_t) * x0
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def conditional_vector_field(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        True conditional vector field u_t(x_t|x1) = dx_t/dt.
        For linear path: u_t = x1 - x0 (constant in time!)
        """
        return x1 - x0  # Straight-line velocity

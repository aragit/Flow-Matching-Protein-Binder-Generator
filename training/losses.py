"""
Flow Matching Loss computation.
"""

import torch
import torch.nn as nn
from sampling.path_schedules import ProbabilityPath


class FlowMatchingLoss(nn.Module):
    """
    Computes the conditional Flow Matching objective.
    L = E[||v_theta(x_t, t) - u_t(x_t|x1)||^2]
    """

    def __init__(self, path: ProbabilityPath):
        super().__init__()
        self.path = path

    def forward(self, model, x1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model: FlowMatcherNetwork
            x1: [batch, N, 3] real data samples
        Returns:
            loss: scalar
        """
        batch_size, n_res, _ = x1.shape
        device = x1.device

        # Sample noise and time
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)

        # Sample from conditional path
        x_t = self.path.sample_xt(x0, x1, t)

        # True conditional vector field
        u_t = self.path.conditional_vector_field(x0, x1, t)

        # Model prediction
        v_theta = model(x_t, t)

        # MSE loss
        return torch.mean((v_theta - u_t) ** 2)

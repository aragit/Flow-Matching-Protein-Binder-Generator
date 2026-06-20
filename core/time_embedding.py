"""
Sinusoidal time embedding for Flow Matching.
Fourier features enable the network to resolve fine-grained temporal structure.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Fourier features for time conditioning. Critical for Flow Matching
    to resolve fine-grained temporal structure in the vector field.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch, 1] in [0, 1]
        Returns:
            [batch, dim] sinusoidal embedding
        """
        half_dim = self.dim // 2
        freqs = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -freqs)
        args = t * freqs.unsqueeze(0) * 2 * math.pi
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        return embedding

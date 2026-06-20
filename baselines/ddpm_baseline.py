"""
DDPM baseline for ablation comparison.

Demonstrates why Flow Matching is preferred:
requires many more steps, no ODE guarantees.
"""

import torch
import torch.nn as nn


class DDPMSampler:
    """
    DDPM baseline for ablation comparison. Demonstrates why Flow Matching
    is preferred: requires many more steps, no ODE guarantees.
    """

    def __init__(self, model: nn.Module, num_steps: int = 1000):
        self.model = model
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, shape: tuple, device: str = "cpu") -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0], 1), t, device=device)
            noise_pred = self.model(x, t_batch / self.num_steps)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # DDPM reverse step
            x = (
                x - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_t)
            if t > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        return x

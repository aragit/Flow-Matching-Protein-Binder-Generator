"""
Modular ODE Solver framework for Flow Matching generation.

Provides Euler, RK4, and Heun solvers with a common interface.
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable


class ODESolver(ABC):
    """Base class for ODE solvers in Flow Matching."""

    def __init__(self, steps: int = 50):
        self.steps = steps

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: float,
        dt: float,
        vector_field: Callable,
    ) -> torch.Tensor:
        pass

    def integrate(
        self,
        x0: torch.Tensor,
        vector_field: Callable,
        device: str = "cpu",
    ) -> torch.Tensor:
        x = x0
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = i * dt
            x = self.step(x, t, dt, vector_field)
        return x


class EulerSolver(ODESolver):
    """First-order explicit Euler. Fast but less accurate."""

    def step(self, x, t, dt, vf):
        return x + vf(x, torch.tensor([[t]], device=x.device)) * dt


class RK4Solver(ODESolver):
    """Fourth-order Runge-Kutta. Better accuracy for complex flows."""

    def step(self, x, t, dt, vf):
        t_tensor = lambda s: torch.tensor([[s]], device=x.device)
        k1 = vf(x, t_tensor(t))
        k2 = vf(x + 0.5 * dt * k1, t_tensor(t + 0.5 * dt))
        k3 = vf(x + 0.5 * dt * k2, t_tensor(t + 0.5 * dt))
        k4 = vf(x + dt * k3, t_tensor(t + dt))
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class HeunSolver(ODESolver):
    """Second-order Heun (improved Euler). Adaptive-ready stub."""

    def step(self, x, t, dt, vf):
        t_tensor = torch.tensor([[t]], device=x.device)
        k1 = vf(x, t_tensor)
        x_euler = x + dt * k1
        k2 = vf(x_euler, torch.tensor([[t + dt]], device=x.device))
        return x + 0.5 * dt * (k1 + k2)

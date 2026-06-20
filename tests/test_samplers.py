"""
Tests for ODE solver convergence and shape preservation.
"""

import torch
from sampling.ode_solvers import EulerSolver, RK4Solver, HeunSolver
from core.vector_field import FlowMatcherNetwork


def dummy_vector_field(x, t):
    """Simple linear vector field for testing."""
    return -x  # Drives toward origin


def test_euler_solver_shape():
    solver = EulerSolver(steps=10)
    x0 = torch.randn(1, 10, 3)
    x1 = solver.integrate(x0, dummy_vector_field)
    assert x1.shape == x0.shape


def test_rk4_solver_shape():
    solver = RK4Solver(steps=10)
    x0 = torch.randn(1, 10, 3)
    x1 = solver.integrate(x0, dummy_vector_field)
    assert x1.shape == x0.shape


def test_heun_solver_shape():
    solver = HeunSolver(steps=10)
    x0 = torch.randn(1, 10, 3)
    x1 = solver.integrate(x0, dummy_vector_field)
    assert x1.shape == x0.shape


def test_solver_convergence():
    """All solvers should drive x toward origin with v = -x."""
    x0 = torch.randn(1, 5, 3)
    for SolverClass in [EulerSolver, RK4Solver, HeunSolver]:
        solver = SolverClass(steps=100)
        x_final = solver.integrate(x0, dummy_vector_field)
        # Should be closer to origin than start
        assert torch.norm(x_final) < torch.norm(x0)

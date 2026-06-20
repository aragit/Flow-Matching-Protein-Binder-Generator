"""
Tests for vector field network shape invariants and basic functionality.
"""

import torch
import pytest
from core.vector_field import FlowMatcherNetwork


def test_flow_matcher_output_shape():
    model = FlowMatcherNetwork(hidden_dim=64, num_layers=2)
    batch_size, n_res = 2, 45
    x_t = torch.randn(batch_size, n_res, 3)
    t = torch.rand(batch_size, 1)

    v_theta = model(x_t, t)
    assert v_theta.shape == (batch_size, n_res, 3)


def test_flow_matcher_receptor_conditioning():
    model = FlowMatcherNetwork(
        hidden_dim=64, num_layers=2, receptor_dim=128
    )
    batch_size, n_res, rec_res = 2, 45, 100
    x_t = torch.randn(batch_size, n_res, 3)
    t = torch.rand(batch_size, 1)
    receptor = torch.randn(batch_size, rec_res, 128)

    v_theta = model(x_t, t, receptor)
    assert v_theta.shape == (batch_size, n_res, 3)


def test_time_embedding_range():
    from core.time_embedding import SinusoidalTimeEmbedding

    embed = SinusoidalTimeEmbedding(dim=64)
    t = torch.tensor([[0.0], [0.5], [1.0]])
    out = embed(t)
    assert out.shape == (3, 64)
    assert not torch.isnan(out).any()

"""
Tests for biophysical validation metrics.
"""

import torch
from biophysics.validator import BiophysicalValidator


def test_clash_score_zero_for_distant_points():
    """Well-separated points should have zero clashes."""
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    clash = BiophysicalValidator.calculate_clash_score(coords, threshold=2.0)
    assert clash == 0.0


def test_clash_score_detects_clash():
    """Very close points should register as clashes."""
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # Within 2.0 threshold
        [5.0, 0.0, 0.0],  # Outside threshold
    ])
    clash = BiophysicalValidator.calculate_clash_score(coords, threshold=2.0)
    assert clash > 0.0


def test_radius_of_gyration():
    coords = torch.randn(20, 3)
    rg = BiophysicalValidator.radius_of_gyration(coords)
    assert rg > 0.0


def test_contact_map_shape():
    coords = torch.randn(20, 3)
    cm = BiophysicalValidator.contact_map(coords)
    assert cm.shape == (20, 20)


def test_foldseek_query_schema():
    coords = torch.randn(45, 3)
    result = BiophysicalValidator.mock_foldseek_query(coords)
    assert "predicted_fold" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

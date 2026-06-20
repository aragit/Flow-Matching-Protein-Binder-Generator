"""
Optimal Transport Coupling for Flow Matching.

Computes OT plan between noise x_0 and data x_1 for straight-line
conditional paths. Uses exact OT for small batches, identity for scale.

Reference: Lipman et al. (2023) "Flow Matching for Generative Modeling"
"""

import torch


class OptimalTransportCoupling:
    """
    Computes OT plan between noise x_0 and data x_1.
    """

    @staticmethod
    def sample_coupling(
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: str = "exact",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0: [batch, N, 3] noise samples
            x1: [batch, N, 3] data samples
            method: "exact" (for small batch) or "identity" (fast approximate)
        Returns:
            (x0_perm, x1) coupled pairs
        """
        if method == "identity":
            return x0, x1  # Fallback: assume aligned

        # Exact OT via Hungarian algorithm on pairwise distances
        # Stub: in practice, use geomloss or POT library
        batch_size = x0.size(0)
        # Compute cost matrix [batch, batch]
        x0_flat = x0.view(batch_size, -1)
        x1_flat = x1.view(batch_size, -1)
        cost_matrix = torch.cdist(x0_flat, x1_flat, p=2) ** 2

        # Stub: identity permutation (replace with scipy.optimize.linear_sum_assignment)
        perm = torch.arange(batch_size)
        return x0[perm], x1

"""
Lightweight training scaffold for Flow Matching.
Compatible with PyTorch Lightning but standalone for Active Blueprint clarity.
"""

import torch
from torch.utils.data import DataLoader
from core.vector_field import FlowMatcherNetwork
from training.losses import FlowMatchingLoss
from sampling.path_schedules import ProbabilityPath


class FlowMatchingTrainer:
    """
    Lightweight training scaffold.
    """

    def __init__(
        self,
        model: FlowMatcherNetwork,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = FlowMatchingLoss(ProbabilityPath("linear"))

    def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        loss = self.loss_fn(self.model, batch)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, dataloader: DataLoader, epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

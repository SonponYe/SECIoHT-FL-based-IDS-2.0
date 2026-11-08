"""S2a - the 2025 paper's DNN, exactly as built in the notebook (Stage 2)."""
import torch.nn as nn


class DNN(nn.Module):
    """Linear(36->64->32->2) with ReLU; two output logits (softmax / cross-entropy)."""

    def __init__(self, input_dim=36, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, output_dim))

    def forward(self, x):
        return self.net(x)

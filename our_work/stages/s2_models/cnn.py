"""S2b - the notebook's CNN (a reconstruction; the paper gives no kernel / stride / pooling)."""
import torch.nn as nn


class CNN(nn.Module):
    """Four Conv1d layers (128/64/32/16, kernel 3, padding 1) -> AdaptiveAvgPool1d(1) -> Linear(16->2)."""

    def __init__(self, input_dim=36, output_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        return self.fc(self.pool(x).squeeze(-1))

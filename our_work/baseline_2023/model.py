"""baseline_2023 / S2 - the 2023 DNN-FL (Section III-B1): one input layer, hidden layers of
64, 64 and 32 neurons (ReLU), one output neuron with a sigmoid (applied inside the loss / predict)."""
import torch.nn as nn


class DNN2023(nn.Module):
    def __init__(self, input_dim=36):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logits; sigmoid is applied by BCEWithLogits / predict_proba_2023

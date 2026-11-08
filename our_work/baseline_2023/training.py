"""baseline_2023 / S3 - training with binary cross-entropy on a sigmoid output (Table II)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(X, y, batch, shuffle=True):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
                      batch_size=batch, shuffle=shuffle)


def train_bce(model, X, y, epochs, opt, batch):
    """`epochs` passes over (X, y) with an already-built optimizer (so state can persist or not)."""
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in _loader(X, y, batch):
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()


def predict_proba_2023(model, X, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs], dtype=torch.float32))).numpy())
    return np.concatenate(out)


def fit_central_2023(make_model, X, y, epochs, lr, batch, optimizer, seed):
    """The paper's centralised DNN baseline (lr 0.1, batch 20; optimizer and epochs are not stated)."""
    torch.manual_seed(seed)
    model = make_model()
    opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    train_bce(model, X, y, epochs, opt, batch)
    return model

"""S3c - data loaders, the local training loop and prediction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict_proba(model, X, bs=4096):
    """P(attack) for a 2-logit classifier."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(F.softmax(model(torch.tensor(X[i:i + bs], dtype=torch.float32)), dim=-1)[:, 1].numpy())
    return np.concatenate(out)


def train_epochs(model, loader, epochs, opt, weights=None, aux_w=0.0, net=None):
    """Local training. `net` is the Opacus-wrapped module when DP is on (else `model`)."""
    net = net if net is not None else model
    crit = nn.CrossEntropyLoss(weight=weights)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(net(xb), yb)
            if aux_w and hasattr(model, "aux"):
                loss = loss + aux_w * model.aux
            loss.backward()
            opt.step()

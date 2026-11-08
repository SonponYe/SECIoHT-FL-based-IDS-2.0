"""S4b - one persistent federated client (a simulated hospital)."""
import numpy as np
import torch

from stages.s1_preprocessing.balancing import smote_resample
from stages.s3_training.loops import make_loader, train_epochs
from stages.s3_training.optimizers import make_optimizer
from stages.s5_privacy.engine import wrap_private


class Client:
    """The model/optimizer/PrivacyEngine are built ONCE and live for the whole run; only the weights
    are overwritten each round, so the privacy accountant keeps accumulating."""

    def __init__(self, make_model, X, y, recipe, client_bal="weights", aux_w=0.0, dp=None, total_epochs=None, seed=0):
        self.base = make_model()
        if client_bal == "smote":
            try:
                X, y = smote_resample(X, y, seed)
            except ValueError:
                client_bal = "weights"
        self.X, self.y, self.n = X, y, len(y)
        counts = np.bincount(y, minlength=2).astype(float)
        self.w = (torch.tensor(len(y) / (2 * np.maximum(counts, 1)), dtype=torch.float32)
                  if client_bal == "weights" else None)
        self.aux_w, self.dp, self.pe, self.net = aux_w, dp, None, None
        opt = make_optimizer(self.base.parameters(), recipe["opt"], recipe["lr"])
        loader = make_loader(X, y, recipe["batch"])
        if dp:
            self.pe, self.net, opt, loader = wrap_private(self.base, opt, loader, dp, total_epochs)
        self.opt, self.loader = opt, loader

    def train(self, epochs):
        train_epochs(self.base, self.loader, epochs, self.opt, self.w, self.aux_w, self.net)

    def eps(self):
        return self.pe.get_epsilon(self.dp["delta"]) if self.pe else None

    def noise(self):
        return self.opt.noise_multiplier if self.pe else None

"""S3d - centralised training with validation-based checkpointing."""
import copy

import numpy as np
import torch

from stages.s3_training.loops import make_loader, predict_proba, train_epochs
from stages.s3_training.optimizers import make_optimizer
from stages.s6_evaluation.metrics import best_threshold


def fit_central(make_model, split, recipe, seed, epochs=None, aux_w=0.0, n_checks=12):
    """Train in one place; keep the checkpoint with the best VALIDATION F1 (threshold-tuned)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = epochs or recipe["epochs"]
    model = make_model()
    opt = make_optimizer(model.parameters(), recipe["opt"], recipe["lr"])
    loader = make_loader(split.Xtr, split.ytr, recipe["batch"])
    w = torch.tensor(split.class_w, dtype=torch.float32) if recipe["balance"] == "weights" else None
    every = max(1, epochs // n_checks)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        train_epochs(model, loader, 1, opt, w, aux_w)
        if ep % every == 0 or ep == epochs:
            f = best_threshold(split.yva, predict_proba(model, split.Xva))[1]
            if f > best:
                best, best_state = f, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best

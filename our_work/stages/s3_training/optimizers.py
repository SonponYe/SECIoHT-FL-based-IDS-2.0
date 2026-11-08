"""S3b - optimizer factory."""
import torch

# default learning rate per optimizer, as used in the optimizer comparison
DEFAULT_LR = {"sgd": 0.01, "momentum": 0.01, "adam": 1e-3}


def make_optimizer(params, opt, lr):
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if opt == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(opt)

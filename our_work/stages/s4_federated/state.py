"""S4a - model-weight helpers shared by every aggregation rule."""


def get_state(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def set_state(model, state, keys=None):
    import torch
    with torch.no_grad():
        for k, p in model.named_parameters():
            if keys is None or k in keys:
                p.copy_(state[k])


def wavg(states, weights, keys):
    """Weighted average of the named parameters `keys`."""
    tot = float(sum(weights))
    return {k: sum(s[k] * (w / tot) for s, w in zip(states, weights)) for k in keys}

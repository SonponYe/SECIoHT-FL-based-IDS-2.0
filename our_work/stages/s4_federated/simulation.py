"""S4c - the federated round loop and FedAvg.

Other aggregation rules (domain-aware, local-only, ...) are passed in as `aggregator`; they live
in their own approach folder (e.g. moe/aggregation.py)."""
import copy

import numpy as np
import torch

from stages.s3_training.loops import predict_proba
from stages.s4_federated.client import Client
from stages.s4_federated.state import get_state, set_state, wavg


def fedavg_agg(rnd, clients, states, ns, ctx):
    g = wavg(states, ns, list(states[0].keys()))
    return [copy.deepcopy(g) for _ in clients], {}


def run_fl(make_model, parts, rounds, local_epochs, recipe, aggregator=fedavg_agg, dp=None, aux_w=0.0,
           seed=0, client_bal="weights", trace_eps=False, ctx=None):
    """Generic federated loop.  aggregator(rnd, clients, states, ns, ctx) -> (new_client_states, info).
    Returns dict(clients, cstates, eps, noise, eps_trace, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    total = rounds * local_epochs
    clients = [Client(make_model, X, y, recipe, client_bal, aux_w, dp, total, seed + i) for i, (X, y) in enumerate(parts)]
    init = get_state(make_model())
    cstates = [copy.deepcopy(init) for _ in clients]
    ctx = dict(ctx or {}, make_model=make_model)
    eps_trace, info = [], {}
    for rnd in range(1, rounds + 1):
        for c, st in zip(clients, cstates):
            set_state(c.base, st)
            c.train(local_epochs)
        states = [get_state(c.base) for c in clients]
        cstates, info = aggregator(rnd, clients, states, [c.n for c in clients], ctx)
        if dp and trace_eps:
            e = max(c.eps() for c in clients)
            eps_trace.append(round(float(e), 4))
            print(f"    round {rnd}/{rounds}: epsilon so far = {e:.4f}", flush=True)
    if trace_eps and rounds >= 2 and dp:
        assert eps_trace[-1] > eps_trace[0], f"epsilon did not grow across rounds: {eps_trace}"
    eps = max(c.eps() for c in clients) if dp else None
    noise = max(c.noise() for c in clients) if dp else None
    return dict(clients=clients, cstates=cstates, eps=eps, noise=noise, eps_trace=eps_trace, info=info)


def eval_state(make_model, state, X):
    m = make_model()
    set_state(m, state)
    return predict_proba(m, X)


def eval_personalised(make_model, cstates, tests):
    """Score each client's own model on its own test slice, pooled."""
    ps, ys = [], []
    for st, (X, y) in zip(cstates, tests):
        ps.append(eval_state(make_model, st, X))
        ys.append(y)
    return np.concatenate(ys), np.concatenate(ps)

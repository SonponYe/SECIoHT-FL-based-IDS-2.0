"""Training loops, persistent federated clients (Opacus wrapped ONCE), FedAvg and the DP grid.

A *recipe* is a dict: {"opt": "sgd"|"momentum"|"adam", "lr": float, "balance": "smote"|"weights",
"batch": int, "epochs": int}.  Track Paper selects the best recipe per protocol on the validation
slice and writes shared/best_recipe.json; Track MoE reads it so the two tracks cannot drift apart.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import best_threshold, metrics

RECIPE_PATH = Path(__file__).resolve().parent / "best_recipe.json"
DEFAULT_RECIPE = {"opt": "adam", "lr": 1e-3, "balance": "weights", "batch": 64, "epochs": 60}
DP_DELTA = 1e-4          # the paper's delta
PAPER_EPS = 0.44


def load_recipe(protocol):
    if RECIPE_PATH.exists():
        r = json.loads(RECIPE_PATH.read_text()).get(protocol)
        if r:
            return dict(r)
    return dict(DEFAULT_RECIPE)


def save_recipe(protocol, recipe, note=""):
    cur = json.loads(RECIPE_PATH.read_text()) if RECIPE_PATH.exists() else {}
    cur[protocol] = recipe
    cur["note"] = note or "selected by mean validation F1 (threshold-tuned) in track_paper stage P2"
    RECIPE_PATH.write_text(json.dumps(cur, indent=2))


def make_optimizer(params, opt, lr):
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if opt == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(opt)


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict_proba(model, X, bs=4096):
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


def fit_central(make_model, split, recipe, seed, epochs=None, aux_w=0.0, n_checks=12):
    """Centralised training; keeps the checkpoint with the best VALIDATION F1 (threshold-tuned)."""
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


# ----------------------------------------------------------------------------- federated
def get_state(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def set_state(model, state, keys=None):
    with torch.no_grad():
        for k, p in model.named_parameters():
            if keys is None or k in keys:
                p.copy_(state[k])


def wavg(states, weights, keys):
    tot = float(sum(weights))
    return {k: sum(s[k] * (w / tot) for s, w in zip(states, weights)) for k in keys}


class Client:
    """Persistent federated client.  The model/optimizer/PrivacyEngine are built ONCE and live for
    the whole run; only the weights are overwritten each round, so the privacy accountant keeps
    accumulating (re-wrapping every round would silently reset epsilon)."""

    def __init__(self, make_model, X, y, recipe, client_bal="weights", aux_w=0.0, dp=None, total_epochs=None, seed=0):
        self.base = make_model()
        if client_bal == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                X, y = SMOTE(random_state=seed).fit_resample(X, y)
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
            from opacus import PrivacyEngine
            self.pe = PrivacyEngine(secure_mode=False)
            if dp.get("target_eps"):
                self.net, opt, loader = self.pe.make_private_with_epsilon(
                    module=self.base, optimizer=opt, data_loader=loader, target_epsilon=dp["target_eps"],
                    target_delta=dp["delta"], epochs=total_epochs, max_grad_norm=dp["clip"])
            else:
                self.net, opt, loader = self.pe.make_private(
                    module=self.base, optimizer=opt, data_loader=loader,
                    noise_multiplier=dp["noise"], max_grad_norm=dp["clip"])
        self.opt, self.loader = opt, loader

    def train(self, epochs):
        train_epochs(self.base, self.loader, epochs, self.opt, self.w, self.aux_w, self.net)

    def eps(self):
        return self.pe.get_epsilon(self.dp["delta"]) if self.pe else None

    def noise(self):
        return self.opt.noise_multiplier if self.pe else None


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
    ps, ys = [], []
    for st, (X, y) in zip(cstates, tests):
        ps.append(eval_state(make_model, st, X))
        ys.append(y)
    return np.concatenate(ys), np.concatenate(ps)


def dp_grid(grid, recipe):
    """The DP settings run by BOTH tracks (identical so the comparison is fair).

    Clip values are crossed with the paper's plain SGD, because Adam rescales gradients and makes
    the clip norm almost irrelevant; the recipe optimizer is added at clip 1.0.  The epsilon-matched
    runs let Opacus choose the noise to hit the paper's epsilon 0.44 at delta 1e-4."""
    sgd = dict(opt="sgd", lr=0.01)
    rec = dict(opt=recipe["opt"], lr=recipe["lr"])
    noises, clips = ((1.5, 0.5), (1e-4, 0.1, 1.0)) if grid == "full" else ((1.5,), (1e-4, 1.0))
    out = []

    def add(name, o, noise, clip, target=None):
        out.append(dict(setting=name, opt=o["opt"], lr=o["lr"], noise=noise, clip=clip, target_eps=target,
                        delta=DP_DELTA))

    for n in noises:
        for c in clips:
            add(f"DP noise={n} clip={c:g} opt=sgd", sgd, n, c)
    if rec["opt"] != "sgd":
        for n in noises:
            add(f"DP noise={n} clip=1 opt={rec['opt']}", rec, n, 1.0)
    add(f"DP eps-matched {PAPER_EPS} clip=1 opt=sgd", sgd, None, 1.0, PAPER_EPS)
    if rec["opt"] != "sgd":
        add(f"DP eps-matched {PAPER_EPS} clip=1 opt={rec['opt']}", rec, None, 1.0, PAPER_EPS)
    return out

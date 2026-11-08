"""baseline_2023 / S4 - the 2023 federated loop (TensorFlow Federated, FedAvg with a server optimizer).

Each round: every client starts from the global weights, trains `local_epochs` epochs with a fresh
Adam(lr=client_lr) and BCE; the server averages the client weights (weighted by sample count),
treats (global - average) as a pseudo-gradient and applies ONE Adam step with lr=server_lr
(TFF's `server_optimizer_fn`).  No differential privacy in this paper.
"""
import copy

import numpy as np
import torch

from baseline_2023.training import train_bce
from stages.s4_federated.state import get_state, set_state, wavg


def run_fl_2023(make_model, clients, rounds, local_epochs, client_lr, server_lr, batch, seed, log_every=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gmodel = make_model()
    server_opt = torch.optim.Adam(gmodel.parameters(), lr=server_lr)
    local = make_model()
    ns = [len(y) for _, y in clients]
    for rnd in range(1, rounds + 1):
        gstate = get_state(gmodel)
        states = []
        for X, y in clients:
            set_state(local, gstate)
            train_bce(local, X, y, local_epochs, torch.optim.Adam(local.parameters(), lr=client_lr), batch)
            states.append(get_state(local))
        avg = wavg(states, ns, list(gstate.keys()))
        server_opt.zero_grad()
        for k, p in gmodel.named_parameters():
            p.grad = (gstate[k] - avg[k]).clone()          # pseudo-gradient
        server_opt.step()
        if log_every and rnd % log_every == 0:
            print(f"    round {rnd}/{rounds}", flush=True)
    return gmodel

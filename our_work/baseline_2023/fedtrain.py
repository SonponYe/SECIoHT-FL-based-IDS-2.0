"""baseline_2023 / step 8 - FEDERATED training of the 2023 DNN in PyTorch (FedAvg + server-side Adam).

It mirrors what TensorFlow Federated's `build_federated_averaging_process` does in the authors' notebook:

  every round   1. broadcast the global weights to every client
                2. each client trains `LOCAL_EPOCHS` (=1) epoch on its own balanced rows, batch 1024, with a FRESH
                   Adam(lr 0.01, eps 1e-7) (TFF builds the client optimizer anew each round)
                3. clients return weight DELTAS (client - global), averaged WEIGHTED BY NUMBER OF EXAMPLES
                4. the server treats -(average delta) as the gradient and applies ONE step of its own persistent
                   Adam(lr 0.05, eps 1e-7)
  100 rounds, Keras-style initialisation (Glorot-uniform weights, zero biases), BCE-with-logits loss.
  (The paper's "epochs 100" is these 100 ROUNDS; local epochs are never set in the authors' code, so 1.)

Data: build(mode, n_clients, seed) from pipeline.py.  mode='as_published' is the authors' pipeline (35 inputs, the column
bug, pooled per-client validation slices as THE evaluation set); 'as_published_fixed' removes the bug; 'honest' is our
corrected pipeline (36 inputs, held-out test set).

THE TEST SET IS NEVER PASSED TO THIS MODULE.  Per round it records validation metrics at threshold 0.5 (for as_published
this validation set IS the authors' reported evaluation).  Choosing a threshold and scoring the honest test set is step 9.
No differential privacy.  Training is deterministic for a given (mode, n_clients, seed); weights are not stored, they are
re-created by re-running (seconds).

Run (from our_work/):   python -m baseline_2023.fedtrain
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from baseline_2023.fl2023 import make_model
from baseline_2023.pipeline import build
from baseline_2023.training import predict_proba_2023
from stages.s6_evaluation.metrics import metrics
from stages.s6_evaluation.reporting import log_row

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS = RESULTS_DIR / "federated.jsonl"
HISTORY = RESULTS_DIR / "federated_history.json"
REPORT = RESULTS_DIR / "federated_training.md"
CLIENT_LR, SERVER_LR, ADAM_EPS, BATCH, ROUNDS, LOCAL_EPOCHS = 0.01, 0.05, 1e-7, 1024, 100, 1
MODES = ("as_published", "as_published_fixed", "honest")
TRACK, STAGE = "baseline-2023", "B3"
TABLE_III = {15: (90.64, 70.39, 49.76), 10: (91.40, 65.05, 61.42), 5: (90.70, 56.40, 76.63)}   # acc / precision / recall (%)


def _bce(y, p) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def client_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, gen: torch.Generator) -> tuple[float, int]:
    """One local epoch with a FRESH Adam, batches of 1024 (the last batch may be smaller; nothing is dropped)."""
    opt = torch.optim.Adam(model.parameters(), lr=CLIENT_LR, eps=ADAM_EPS)
    crit, n = nn.BCEWithLogitsLoss(), len(y)
    model.train()
    perm, total, batches = torch.randperm(n, generator=gen), 0.0, 0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = crit(model(X[idx]), y[idx])
        loss.backward()
        opt.step()
        total, batches = total + loss.item() * len(idx), batches + 1
    return total, batches


def federated_train(clients: list[dict], X_val: np.ndarray, y_val: np.ndarray, *, seed: int, rounds: int = ROUNDS,
                    local_epochs: int = LOCAL_EPOCHS) -> dict:
    """FedAvg + server Adam.  `clients` are the build() client dicts (only X_train / y_train are used).
    Returns final-round and best-validation-AUC weights, the per-round history and timing.

    Runs on ONE torch thread: with several threads the floating-point summation order changes, and 100 rounds of training
    amplify that, so the same seed gave visibly different results (val AUC 0.851 vs 0.830 for 1 vs 2 threads).  The
    caller's thread setting is restored afterwards."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _federated_train(clients, X_val, y_val, seed=seed, rounds=rounds, local_epochs=local_epochs)
    finally:
        torch.set_num_threads(prev_threads)


def _federated_train(clients, X_val, y_val, *, seed, rounds, local_epochs) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_in = clients[0]["X_train"].shape[1]
    gmodel, local = make_model(n_in), make_model(n_in)
    server_opt = torch.optim.Adam(gmodel.parameters(), lr=SERVER_LR, eps=ADAM_EPS)     # persistent across rounds
    data = [(torch.tensor(c["X_train"], dtype=torch.float32), torch.tensor(c["y_train"], dtype=torch.float32)) for c in clients]
    sizes = [len(y) for _, y in data]
    total_n = float(sum(sizes))
    hist = {k: [] for k in ("train_loss", "examples", "batches", "val_loss", "val_auc", "val_acc", "val_prec", "val_rec")}
    best_auc, best_round, best_state, first = -1.0, 0, None, {}
    client_opts = 0
    t0 = time.perf_counter()
    for rnd in range(1, rounds + 1):
        gstate = {k: v.detach().clone() for k, v in gmodel.state_dict().items()}
        avg_delta = {k: torch.zeros_like(v) for k, v in gstate.items()}
        loss_sum, batches = 0.0, 0
        for ci, (X, y) in enumerate(data):
            local.load_state_dict(gstate)
            gen = torch.Generator().manual_seed(seed * 1_000_003 + rnd * 1_009 + ci)
            for _ in range(local_epochs):
                ls, nb = client_epoch(local, X, y, gen)
                loss_sum, batches, client_opts = loss_sum + ls, batches + nb, client_opts + 1
            w = sizes[ci] / total_n                                                      # weight by number of examples
            for k, v in local.state_dict().items():
                avg_delta[k] += w * (v.detach() - gstate[k])
        server_opt.zero_grad()
        for k, p in gmodel.named_parameters():
            p.grad = -avg_delta[k]                                                       # server "gradient" = -(mean delta)
        server_opt.step()
        if rnd == 1:                                                                     # Adam's first step is ~lr * sign(g)
            first = {"abs_update": np.concatenate([(gmodel.state_dict()[k] - gstate[k]).abs().flatten().numpy() for k in gstate]),
                     "abs_grad": np.concatenate([avg_delta[k].abs().flatten().numpy() for k in gstate])}
        pv = predict_proba_2023(gmodel, X_val)
        m = metrics(y_val, pv, 0.5)
        auc = float(roc_auc_score(y_val, pv))
        for k, v in (("train_loss", loss_sum / total_n), ("examples", int(total_n) * local_epochs), ("batches", batches),
                     ("val_loss", _bce(y_val, pv)), ("val_auc", auc), ("val_acc", m["acc"]), ("val_prec", m["prec"]), ("val_rec", m["rec"])):
            hist[k].append(v)
        if auc > best_auc:
            best_auc, best_round = auc, rnd
            best_state = {k: v.detach().clone() for k, v in gmodel.state_dict().items()}
    secs = time.perf_counter() - t0
    return {"final_state": {k: v.detach().clone() for k, v in gmodel.state_dict().items()}, "best_state": best_state,
            "best_round": best_round, "best_val_auc": best_auc, "history": hist, "train_seconds": secs, "rounds": rounds,
            "local_epochs": local_epochs, "n_clients": len(clients), "client_sizes": sizes, "client_optimizer_builds": client_opts,
            "first_round": first, "threads": 1}


# ------------------------------------------------------------------------------------------------ the grid
def run_one(mode: str, n_clients: int, seed: int) -> dict:
    d = build(mode, n_clients, seed=seed)
    fit = federated_train(d["clients"], d["X_val"], d["y_val"], seed=seed)
    h, y_val = fit["history"], d["y_val"]
    m_final = {k: h[k][-1] for k in ("val_loss", "val_auc", "val_acc", "val_prec", "val_rec")}
    row = {"track": TRACK, "stage": STAGE, "group": "Federated (no DP)", "model": "DNN 2023 FedAvg + server Adam", "protocol": mode,
           "mode": mode, "n_clients": n_clients, "seed": seed, "rounds": fit["rounds"], "local_epochs": fit["local_epochs"],
           "n_inputs": len(d["feature_names"]), "temp_is_input": mode != "as_published", "client_sizes": fit["client_sizes"],
           "examples_per_round": h["examples"][0], "batches_per_round": h["batches"][0],
           "train_loss_first": h["train_loss"][0], "train_loss_last": h["train_loss"][-1],
           "val_rows": int(len(y_val)), "val_attacks": int(y_val.sum()), "val_majority_acc": float(max(y_val.mean(), 1 - y_val.mean())),
           "final_val_acc": m_final["val_acc"], "final_val_prec": m_final["val_prec"], "final_val_rec": m_final["val_rec"],
           "final_val_auc": m_final["val_auc"], "final_val_loss": m_final["val_loss"],
           "best_round": fit["best_round"], "best_val_auc": fit["best_val_auc"], "train_seconds": round(fit["train_seconds"], 3),
           "client_optimizer_builds": fit["client_optimizer_builds"], "threads": fit["threads"], "test_used": False}
    return {"row": row, "fit": fit, "data": d}


def run_federated(modes=MODES, clients=(15, 10, 5), seeds=(0, 1, 2), write: bool = True) -> list[dict]:
    """Train every (mode, n_clients, seed).  write=True appends one summary row per run to federated.jsonl (append-only,
    one run_id per call), rewrites the history file and the markdown table."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = []
    for mode in modes:
        for k in clients:
            for s in seeds:
                r = run_one(mode, k, s)
                runs.append(r)
                w = r["row"]
                print(f"[B3 {mode:18s} {k:2d} clients seed {s}] final-round validation @0.5: acc {w['final_val_acc']:.4f} P {w['final_val_prec']:.3f} "
                      f"R {w['final_val_rec']:.3f} AUC {w['final_val_auc']:.3f} | best round {w['best_round']} (AUC {w['best_val_auc']:.3f}) | "
                      f"loss {w['train_loss_first']:.3f} -> {w['train_loss_last']:.3f} | {w['train_seconds']:.1f}s", flush=True)
    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        for r in runs:
            log_row(RUNS, {**r["row"], "run_id": run_id})
        HISTORY.write_text(json.dumps({"run_id": run_id, "runs": {f"{r['row']['mode']}|{r['row']['n_clients']}|{r['row']['seed']}": r["fit"]["history"] | {"best_round": r["fit"]["best_round"]} for r in runs}}), encoding="utf8")
        REPORT.write_text(report_markdown([r["row"] for r in runs], run_id), encoding="utf8")
    return runs


def load_federated() -> tuple[list[dict], dict]:
    """Rows of the latest run from federated.jsonl and its per-round history."""
    if not RUNS.exists():
        raise FileNotFoundError("no results yet: run `python -m baseline_2023.fedtrain` from our_work/")
    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf8").splitlines() if l.strip()]
    latest = max(r["run_id"] for r in rows)
    return [r for r in rows if r["run_id"] == latest], json.loads(HISTORY.read_text(encoding="utf8"))


def _ms(v, pct=False) -> str:
    v = np.asarray(v, dtype=float) * (100 if pct else 1)
    f = "{:.2f}" if pct else "{:.3f}"
    return (f + " ± " + f).format(v.mean(), v.std(ddof=1)) if len(v) > 1 else f.format(v.mean())


def report_markdown(rows: list[dict], run_id: str = "") -> str:
    out = ["# 2023 DNN trained federated (step 8): FedAvg + server Adam, no DP", "",
           f"Run `{run_id}`. {ROUNDS} rounds x {LOCAL_EPOCHS} local epoch, client Adam lr {CLIENT_LR}, server Adam lr {SERVER_LR}, eps {ADAM_EPS}, "
           f"batch {BATCH}, Keras-style init. Mean ± sample std over seeds {sorted({r['seed'] for r in rows})}. **The honest test set is not used here "
           "(step 9).** Numbers are the FINAL-round model on the VALIDATION set at threshold 0.5; for `as_published` that validation set is the "
           "authors' reported evaluation set (no test set exists in their pipeline).", ""]
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        out += [f"## {mode}", "", "| clients | inputs | val rows / attacks | always-normal acc | accuracy % | precision % | recall % | AUC | best round | train loss r1 -> r100 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted({r["n_clients"] for r in sub}, reverse=True):
            g = [r for r in sub if r["n_clients"] == k]
            out.append(f"| {k} | {g[0]['n_inputs']} | {g[0]['val_rows']} / {g[0]['val_attacks']} | {g[0]['val_majority_acc']:.4f} | "
                       f"{_ms([r['final_val_acc'] for r in g], True)} | {_ms([r['final_val_prec'] for r in g], True)} | "
                       f"{_ms([r['final_val_rec'] for r in g], True)} | {_ms([r['final_val_auc'] for r in g])} | "
                       f"{int(np.median([r['best_round'] for r in g]))} | {np.mean([r['train_loss_first'] for r in g]):.3f} -> {np.mean([r['train_loss_last'] for r in g]):.3f} |")
        if mode == "as_published":
            out += ["", "Paper Table III (the authors' notebook output), for comparison:", "", "| clients | accuracy % | precision % | recall % |", "|---|---|---|---|"]
            out += [f"| {k} | {a:.2f} | {p:.2f} | {r:.2f} |" for k, (a, p, r) in TABLE_III.items()]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    run_federated()

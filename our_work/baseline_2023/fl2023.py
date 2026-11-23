"""baseline_2023 / step 7 - the 2023 DNN, trained CENTRALLY on the honest pipeline (no federation, no DP).

Model     reused from model.py (64-64-32-1 logit, the architecture the paper and the authors' code agree on) with
          Keras-style initialisation added here: Glorot-uniform weights, zero biases.  model.py is NOT edited, so run.py
          is unaffected.
Data      build("honest", n_clients=1, seed): 1:1 balanced training rows, global validation set, held-out test set.
Training  Adam(lr 0.01, eps 1e-7), batch 1024, 100 epochs.  After every epoch the validation AUC is computed and the
          best checkpoint is kept.  The decision threshold maximising F1 is then chosen on the VALIDATION set only.
Test      scored exactly ONCE per seed, through OneShotTest, and never seen by the training function.

Run (from our_work/):   python -m baseline_2023.fl2023            -> trains seeds 0 1 2, writes results/
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from baseline_2023.model import DNN2023
from baseline_2023.pipeline import build
from baseline_2023.training import predict_proba_2023
from stages.s6_evaluation.metrics import best_threshold, metrics
from stages.s6_evaluation.reporting import log_row

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS = RESULTS_DIR / "results.jsonl"
HISTORY = RESULTS_DIR / "central_dnn_history.json"
REPORT = RESULTS_DIR / "central_dnn.md"
LR, ADAM_EPS, BATCH, MAX_EPOCHS = 0.01, 1e-7, 1024, 100      # Table II client optimizer; Keras epsilon default
TRACK, STAGE, N_FEATURES = "baseline-2023", "B2", 36


# ------------------------------------------------------------------------------------------------ model
def glorot_zero_init(model: nn.Module) -> nn.Module:
    """Keras Dense defaults: Glorot-uniform weights, zero biases."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


class KerasDNN2023(DNN2023):
    """DNN2023 (64-64-32-1, ReLU, logit output) with Keras-style initialisation."""

    def __init__(self, input_dim: int = N_FEATURES):
        super().__init__(input_dim)
        glorot_zero_init(self)


def make_model(input_dim: int = N_FEATURES) -> KerasDNN2023:
    return KerasDNN2023(input_dim)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """The optimizer the federated clients will use too: Adam, lr 0.01, epsilon 1e-7."""
    return torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)


def model_summary(model: nn.Module) -> str:
    rows, total = [], 0
    for i, m in enumerate(l for l in model.net if isinstance(l, nn.Linear)):
        n = sum(p.numel() for p in m.parameters())
        total += n
        act = "linear (logit; sigmoid is inside BCEWithLogitsLoss)" if m.out_features == 1 else "ReLU"
        rows.append(f"Dense {m.in_features:>2} -> {m.out_features:<2}  activation: {act:<52} params: {n:>5}")
    return "\n".join(rows + [f"{'total trainable parameters':<84}  {total:>5}"])


# ------------------------------------------------------------------------------------------------ data guards
def _fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update("\n".join(map(str, a)).encode() if a.dtype == object else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


class OneShotTest:
    """The held-out test inputs are reachable ONLY through score(), which may be called once.  It is never passed to the
    training function, and calls are counted so the checks can prove the test set was touched exactly once."""

    def __init__(self, X, y, cat):
        self._X, self.y, self.cat, self.calls = X, y, cat, 0
        self.fingerprint = _fingerprint(X, y, cat)

    def score(self, model: nn.Module) -> np.ndarray:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("the test set may be scored only once")
        return predict_proba_2023(model, self._X)


# ------------------------------------------------------------------------------------------------ training
def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_central(X, y, X_val, y_val, seed: int) -> dict:
    """Adam(0.01, eps 1e-7), batch 1024, 100 epochs; keeps the checkpoint with the best VALIDATION AUC.
    Sees only training and validation data.  The timer starts before the loop and stops after it (it includes the
    per-epoch validation pass)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(X.shape[1])
    opt, crit = make_optimizer(model), nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    n = len(yt)
    hist = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_epoch, best_state = -1.0, 0, None
    t0 = time.perf_counter()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        pv = predict_proba_2023(model, X_val)
        auc = float(roc_auc_score(y_val, pv))
        hist["train_loss"].append(total / n)
        hist["val_loss"].append(_bce(y_val, pv))
        hist["val_auc"].append(auc)
        if auc > best_auc:
            best_auc, best_epoch, best_state = auc, ep, copy.deepcopy(model.state_dict())
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return {"model": model, "history": hist, "best_epoch": best_epoch, "best_val_auc": best_auc,
            "epochs_run": MAX_EPOCHS, "train_seconds": secs}


# ------------------------------------------------------------------------------------------------ one seed
def run_seed(seed: int) -> dict:
    d = build("honest", 1, seed=seed)
    names = d["feature_names"]
    assert len(names) == N_FEATURES and "Temp" in names and "Dport" in names, f"expected 36 features incl. Temp and Dport, got {len(names)}"
    c = d["clients"][0]
    test = OneShotTest(d["X_test"], d["y_test"], d["cat_test"])
    fit = train_central(c["X_train"], c["y_train"], d["X_val"], d["y_val"], seed)
    model = fit["model"]
    p_val = predict_proba_2023(model, d["X_val"])
    thr, val_f1 = best_threshold(d["y_val"], p_val)            # VALIDATION only
    p_test = test.score(model)                                   # the one and only look at the test set
    info = d["info"]
    base = {"track": TRACK, "protocol": "honest", "stage": STAGE, "group": "Centralized",
            "model": "DNN 2023 (centralised, Keras init)", "dp_guarantee": "no (no DP)",
            "setting": f"B2 central, Adam lr {LR} eps {ADAM_EPS}, batch {BATCH}, {MAX_EPOCHS} epochs, 1:1 balanced",
            "majority_acc": info["eval_sets"]["test"]["majority_acc"], "test_attack_rate": info["eval_sets"]["test"]["attack_share"]}
    rows = []
    for mode, t in (("@0.5", 0.5), ("+thr", thr)):
        m = metrics(d["y_test"], p_test, t)
        pred = p_test >= t
        extra = {"rec_spoofing": float(pred[d["cat_test"] == "Spoofing"].mean()),
                 "rec_data_alteration": float(pred[d["cat_test"] == "Data Alteration"].mean()),
                 "n_spoofing": int((d["cat_test"] == "Spoofing").sum()), "n_data_alteration": int((d["cat_test"] == "Data Alteration").sum())}
        rows.append({**base, "seed": seed, "thr_mode": mode, "val_f1": round(float(val_f1), 4), **m, **extra,
                     "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"], "best_val_auc": fit["best_val_auc"],
                     "train_seconds": round(fit["train_seconds"], 3), "n_train_rows": int(len(c["y_train"])),
                     "n_features": len(names), "test_calls": test.calls, "test_hash": test.fingerprint})
    return {"seed": seed, "feature_names": names, "history": fit["history"], "rows": rows, "test_calls": test.calls,
            "test_hash": test.fingerprint, "best_epoch": fit["best_epoch"], "best_val_auc": fit["best_val_auc"],
            "epochs_run": fit["epochs_run"], "train_seconds": fit["train_seconds"], "threshold": thr, "val_f1": val_f1,
            "n_features": len(names), "val_probs": p_val, "y_val": d["y_val"]}


# ------------------------------------------------------------------------------------------------ all seeds, reporting
def run_central(seeds=(0, 1, 2), write: bool = True) -> list[dict]:
    """Train every seed.  write=True appends to results.jsonl (append-only, one run_id per call), rewrites the history
    file and the markdown table."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        for row in r["rows"]:
            m = row
            print(f"[B2 seed {s}] {row['thr_mode']:5s} thr={row['thr']:.2f} acc={m['acc']:.4f} P={m['prec']:.3f} R={m['rec']:.3f} "
                  f"F1={m['f1']:.3f} AUC={m['auc']:.3f} | best epoch {r['best_epoch']} (val AUC {r['best_val_auc']:.4f}) "
                  f"{r['epochs_run']} epochs in {r['train_seconds']:.1f}s", flush=True)
    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        for r in runs:
            for row in r["rows"]:
                log_row(RESULTS, {**row, "run_id": run_id})
        HISTORY.write_text(json.dumps({"run_id": run_id, "seeds": {str(r["seed"]): {**r["history"], "best_epoch": r["best_epoch"]}
                                                                    for r in runs}}), encoding="utf8")
        REPORT.write_text(report_markdown([row for r in runs for row in r["rows"]], run_id), encoding="utf8")
    return runs


def load_central() -> tuple[list[dict], dict]:
    """Rows of the latest run (all seeds) from results.jsonl, and its per-epoch history."""
    if not RESULTS.exists():
        raise FileNotFoundError("no results yet: run `python -m baseline_2023.fl2023` from our_work/")
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("stage") == STAGE and "run_id" in r]
    latest = max(r["run_id"] for r in rows)
    return [r for r in rows if r["run_id"] == latest], json.loads(HISTORY.read_text(encoding="utf8"))


METRIC_COLS = [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "F1"), ("macro_f1", "macro-F1"),
               ("rec_normal", "normal-class recall"), ("auc", "AUC"), ("rec_spoofing", "recall Spoofing"),
               ("rec_data_alteration", "recall Data Alteration")]


def _ms(vals) -> str:
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def report_markdown(rows: list[dict], run_id: str = "") -> str:
    seeds = sorted({r["seed"] for r in rows})
    r0 = rows[0]
    out = [f"# 2023 DNN trained centrally on the honest pipeline (step 7)", "",
           f"Run `{run_id}`, seeds {seeds}. Adam lr {LR}, eps {ADAM_EPS}, batch {BATCH}, {MAX_EPOCHS} epochs, Keras-style init, "
           f"1:1 balanced training rows ({r0['n_train_rows']}), {r0['n_features']} features. Checkpoint = best validation AUC; "
           "threshold = F1-maximising on the VALIDATION set; test set scored once per seed. Mean ± sample std over seeds.", "",
           f"Test set: {r0['tp'] + r0['tn'] + r0['fp'] + r0['fn']} rows, {r0['tp'] + r0['fn']} attacks "
           f"({r0['n_spoofing']} Spoofing + {r0['n_data_alteration']} Data Alteration). "
           f"**Always-predict-normal accuracy: {r0['majority_acc']:.4f}.**", "",
           "| metric | threshold 0.5 | validation-tuned threshold |", "|---|---|---|"]
    by = {m: [r for r in rows if r["thr_mode"] == m] for m in ("@0.5", "+thr")}
    for key, label in METRIC_COLS:
        out.append(f"| {label} | {_ms([r[key] for r in by['@0.5']])} | {_ms([r[key] for r in by['+thr']])} |")
    out.append(f"| threshold | 0.50 | {_ms([r['thr'] for r in by['+thr']])} |")
    out += ["", "Confusion matrices per seed (rows = actual, columns = predicted; TN FP / FN TP):", "",
            "| seed | threshold | TN | FP | FN | TP | best epoch | best val AUC | train seconds |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['seed']} | {r['thr']:.2f} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} | {r['best_epoch']} | "
                   f"{r['best_val_auc']:.4f} | {r['train_seconds']:.1f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    run_central()

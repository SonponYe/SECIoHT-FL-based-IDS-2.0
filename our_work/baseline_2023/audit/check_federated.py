"""Checks for step 8 (federated training of the 2023 DNN: FedAvg + server Adam, 100 rounds x 1 local epoch).

Run from our_work/:   python -m baseline_2023.audit.check_federated
Re-trains the whole grid (3 modes x 15/10/5 clients x seeds 0-2) WITHOUT writing results, then checks it.
Checks and thresholds were fixed before the first full run and are never adjusted to pass.  Lines starting [INFO] are
report-only and never fail.  Exit code 1 if any check fails.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np
import torch

from baseline_2023 import fedtrain as T
from baseline_2023.pipeline import build

RESULTS: list[bool] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def info(name: str, detail: str) -> None:
    print(f"[INFO] {name}: {detail}")


def main() -> int:
    runs = T.run_federated(write=False)
    rows = [r["row"] for r in runs]
    by = {(w["mode"], w["n_clients"], w["seed"]): r for r, w in zip(runs, rows)}

    # ---- configuration
    rec("training function takes no test data (signature has no test argument) and no run used the test set",
        not any("test" in p for p in inspect.signature(T.federated_train).parameters) and not any(w["test_used"] for w in rows),
        f"parameters: {list(inspect.signature(T.federated_train).parameters)}")
    rec("settings: 100 rounds x 1 local epoch, client Adam lr 0.01, server Adam lr 0.05, eps 1e-7, batch 1024",
        (T.ROUNDS, T.LOCAL_EPOCHS, T.CLIENT_LR, T.SERVER_LR, T.ADAM_EPS, T.BATCH) == (100, 1, 0.01, 0.05, 1e-7, 1024) and
        all(w["rounds"] == 100 and w["local_epochs"] == 1 for w in rows), f"{len(rows)} runs")
    rec("27 runs: 3 modes x clients 15/10/5 x seeds 0/1/2", len(rows) == 27 and len(by) == 27, f"{len(rows)} runs")

    # ---- inputs per mode
    ok_in = all(w["n_inputs"] == (35 if w["mode"] != "honest" else 36) for w in rows)
    rec("inputs: 35 for as_published and as_published_fixed, 36 for honest", ok_in, "")
    rec("as_published has no Temp input (column bug); as_published_fixed and honest have it",
        all(w["temp_is_input"] == (w["mode"] != "as_published") for w in rows), "")
    d_h = build("honest", 15, seed=0)
    rec("honest: 36 inputs with 'Temp' and 'Dport' in feature_names", len(d_h["feature_names"]) == 36 and "Temp" in d_h["feature_names"] and "Dport" in d_h["feature_names"], "")

    # ---- federation mechanics
    rec("a fresh client optimizer is built for every client in every round (clients x rounds x local epochs)",
        all(w["client_optimizer_builds"] == w["n_clients"] * w["rounds"] * w["local_epochs"] for w in rows),
        f"e.g. 15 clients -> {by[('honest', 15, 0)]['row']['client_optimizer_builds']}")
    rec("every round sees every client's rows once: examples per round = balanced training rows",
        all(w["examples_per_round"] == sum(w["client_sizes"]) for w in rows), "")
    nb = all(w["batches_per_round"] == sum(-(-n // 1024) for n in w["client_sizes"]) for w in rows)
    rec("batches per round = sum over clients of ceil(rows / 1024) (last partial batch kept)", nb, "")
    ap15 = by[("as_published", 15, 0)]["row"]
    rec("as_published, 15 clients: 5,493 examples in 15 batches per round (the authors' notebook output)",
        (ap15["examples_per_round"], ap15["batches_per_round"]) == (5493, 15), f"{ap15['examples_per_round']} in {ap15['batches_per_round']}")
    for k, (ex, ba) in ((10, (5547, 10)), (5, (5586, 9))):
        w = by[("as_published", k, 0)]["row"]
        info(f"REPORT ONLY as_published {k} clients per-round examples / batches",
             f"ours {w['examples_per_round']} / {w['batches_per_round']}   authors' notebook {ex} / {ba}  (their {k}-client files were made outside the notebook)")

    # ---- FedAvg + server Adam correctness (independent of the training loop)
    # Adam's very first step is exactly  lr * |g| / (|g| + eps)  (bias-corrected m = g, v = g^2): about lr = 0.05 when |g| >> eps,
    # slightly less when |g| is small.  Checked for EVERY parameter against that formula, not against a rounded constant.
    f = by[("as_published", 15, 0)]["fit"]["first_round"]
    expected = T.SERVER_LR * f["abs_grad"] / (f["abs_grad"] + T.ADAM_EPS)
    err = float(np.abs(f["abs_update"] - expected).max())
    big = f["abs_grad"] > 1e-3
    rec("server Adam: the round-1 update of EVERY parameter equals lr*|g|/(|g|+eps) (persistent server optimizer, lr 0.05, eps 1e-7)",
        len(expected) == 8577 and err < 2e-6 and f["abs_update"].max() <= 0.05 + 1e-6,       # 8,577 = parameters with 35 inputs
        f"{len(expected)} parameters, max |update - formula| = {err:.1e}; {int(big.sum())} with |g|>1e-3 move by {f['abs_update'][big].min():.6f}..{f['abs_update'][big].max():.6f}")
    # FedAvg weighting: weights are proportional to client rows and sum to 1 (recomputed from the logged sizes)
    sz = np.array(by[("as_published", 15, 0)]["row"]["client_sizes"], dtype=float)
    rec("FedAvg weights = rows / total rows and sum to 1 (clients are NOT equally weighted)", abs((sz / sz.sum()).sum() - 1) < 1e-12 and len(set(sz)) > 1,
        f"client rows {int(sz.min())}..{int(sz.max())}")

    # ---- learning
    for mode in T.MODES:
        for k in (15, 10, 5):
            g = [by[(mode, k, s)]["fit"]["history"] for s in (0, 1, 2)]
            ok = all(np.mean(h["train_loss"][-10:]) < np.mean(h["train_loss"][:10]) for h in g)
            rec(f"{mode} {k} clients: mean training loss of the last 10 rounds < first 10 rounds (all 3 seeds)", ok,
                "; ".join(f"{np.mean(h['train_loss'][:10]):.3f}->{np.mean(h['train_loss'][-10:]):.3f}" for h in g))
    for k in (15, 10, 5):
        aucs = [by[("honest", k, s)]["row"]["best_val_auc"] for s in (0, 1, 2)]
        rec(f"honest {k} clients: best validation AUC > 0.9 on every seed", min(aucs) > 0.9, f"{[round(a, 4) for a in aucs]}")

    # ---- reproducibility
    a = T.run_one("as_published", 5, 1)
    b = T.run_one("as_published", 5, 1)
    same = all(torch.equal(a["fit"]["final_state"][k], b["fit"]["final_state"][k]) for k in a["fit"]["final_state"])
    rec("same (mode, clients, seed) gives bit-identical final weights", same, "as_published, 5 clients, seed 1, run twice")
    prev = torch.get_num_threads()
    torch.set_num_threads(3)
    c = T.run_one("as_published", 5, 1)
    torch.set_num_threads(prev)
    rec("result does not depend on the caller's torch thread count", all(torch.equal(a["fit"]["final_state"][k], c["fit"]["final_state"][k]) for k in a["fit"]["final_state"]),
        "caller threads 3 vs default")
    try:
        logged, _ = T.load_federated()
        diffs = []
        for r in logged:
            mine = by[(r["mode"], r["n_clients"], r["seed"])]["row"]
            diffs.append(abs(mine["final_val_auc"] - r["final_val_auc"]) + abs(mine["final_val_acc"] - r["final_val_acc"]) + abs(mine["train_loss_last"] - r["train_loss_last"]))
        rec("re-running reproduces every logged run (27 of 27)", len(logged) == 27 and max(diffs) < 1e-9, f"max difference {max(diffs):.1e}")
    except FileNotFoundError:
        info("reproduction of logged results", "no logged results yet; skipped")

    # ---- report only: closeness to the paper's Table III (final round, validation @ 0.5)
    for k, (pa, pp, pr) in T.TABLE_III.items():
        g = [by[("as_published", k, s)]["row"] for s in (0, 1, 2)]
        info(f"REPORT ONLY as_published {k} clients vs Table III",
             f"ours acc {100 * np.mean([w['final_val_acc'] for w in g]):.2f} / prec {100 * np.mean([w['final_val_prec'] for w in g]):.2f} / rec {100 * np.mean([w['final_val_rec'] for w in g]):.2f}"
             f"   paper {pa:.2f} / {pp:.2f} / {pr:.2f}")

    bad = RESULTS.count(False)
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} checks passed" + (f"  ({bad} FAILED)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

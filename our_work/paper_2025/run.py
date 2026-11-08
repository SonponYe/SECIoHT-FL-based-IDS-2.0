"""paper_2025: improve the 2025 paper's OWN models/pipeline (DNN, CNN, FedAvg, DP) - no MoE anywhere.

Run from our_work/ :   python -m paper_2025.run --stage optim --protocol honest --seeds 0 1 2
Stages: reproduce(P0) optim(P1) threshold(P2) cnn(P3) fedavg(P4) dp(P5) gbt(P6) fedgbt(P7) | all

This file only ORCHESTRATES: every step is a call into stages/ (s1 preprocessing -> s2 models ->
s3 training -> s4 federated -> s5 privacy -> s6 evaluation).  Change a block there, not here.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from stages.s1_preprocessing.loading import DEFAULT_CSV
from stages.s1_preprocessing.partitioning import iid_parts
from stages.s1_preprocessing.protocols import client_balance, make_split
from stages.s2_models.cnn import CNN
from stages.s2_models.dnn import DNN
from stages.s3_training.centralized import fit_central
from stages.s3_training.loops import make_loader, predict_proba, train_epochs
from stages.s3_training.optimizers import DEFAULT_LR, make_optimizer
from stages.s3_training.recipes import load_recipe, save_recipe
from stages.s4_federated.simulation import eval_state, run_fl
from stages.s5_privacy.dp_settings import dp_grid
from stages.s6_evaluation.reporting import log_row, report, report_fixed
from stages.s6_evaluation.summary import load_rows, summarize

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TRACK = "paper-track"          # label stored in results; kept so earlier results stay valid
STAGES = ["reproduce", "optim", "threshold", "cnn", "fedavg", "dp", "gbt", "fedgbt"]


def base(split, stage, group, model, setting, dp_guarantee="no (no DP)"):
    return dict(track=TRACK, protocol=split.info["protocol"], stage=stage, group=group, model=model,
                setting=setting, dp_guarantee=dp_guarantee, majority_acc=split.info["majority_acc"],
                test_attack_rate=split.info["test_attack_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--protocol", default="both", choices=["paper", "honest", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--quick", action="store_true", help="tiny rounds/epochs for smoke tests")
    ap.add_argument("--epochs", type=int, default=60, help="centralised epochs for P1/P3")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--grid", default="full", choices=["full", "small"], help="DP grid size (P5)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.quick:
        a.epochs, a.rounds, a.local_epochs = 3, 2, 1
    out = HERE / "results" / ("quick" if a.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "results.jsonl"
    protocols = ["paper", "honest"] if a.protocol == "both" else [a.protocol]
    stages = STAGES if a.stage == "all" else [a.stage]
    gbt_iter = 30 if a.quick else 400
    print(f"== paper_2025 | stages={stages} protocols={protocols} seeds={a.seeds} quick={a.quick} -> {jl}", flush=True)

    for stage in stages:
        for protocol in protocols:
            if stage == "threshold":
                stage_threshold(jl, protocol, save=not a.quick)
                continue
            for seed in a.seeds:
                t0 = time.time()
                globals()["stage_" + stage](a, jl, protocol, seed, gbt_iter)
                secs = round(time.time() - t0, 1)
                log_row(out / "timing.jsonl", dict(stage=stage, protocol=protocol, seed=seed, secs=secs, quick=a.quick))
                print(f"  -- {stage}/{protocol}/seed{seed} took {secs}s", flush=True)
    if jl.exists():
        summarize(jl, out / "summary.csv")
        print("summary ->", out / "summary.csv")


# ------------------------------------------------------------------ P0
def stage_reproduce(a, jl, protocol, seed, _):
    """The notebook's exact DNN recipe (plain SGD lr .01, batch 64, 100 epochs, SMOTE, last epoch)
    under BOTH protocols - shows how much the evaluation design alone moves the scores."""
    epochs = 3 if a.quick else 100
    sp = make_split(protocol, seed, balance="smote", csv=a.csv, val_frac=0)
    print(f"   {protocol}: test attack rate {sp.info['test_attack_rate']}  all-normal accuracy {sp.info['majority_acc']}")
    torch.manual_seed(seed)
    m = DNN()
    opt = make_optimizer(m.parameters(), "sgd", 0.01)
    train_epochs(m, make_loader(sp.Xtr, sp.ytr, 64), epochs, opt)
    p = predict_proba(m, sp.Xte)
    report_fixed(jl, base(sp, "P0", "Centralized", "DNN", "P0 notebook recipe (SGD, SMOTE, 100 ep)"), sp.yte, p, seed,
                 extra=dict(recipe=dict(opt="sgd", lr=0.01, batch=64, epochs=epochs, balance="smote")))


# ------------------------------------------------------------------ P1
def stage_optim(a, jl, protocol, seed, _):
    """Optimizer x balancing comparison; keeps the best-validation-F1 checkpoint."""
    balances = ["smote"] if protocol == "paper" else ["smote", "weights"]   # `paper` data is pre-balanced
    for opt, lr in DEFAULT_LR.items():
        for bal in balances:
            recipe = dict(opt=opt, lr=lr, balance=bal, batch=64, epochs=a.epochs)
            sp = make_split(protocol, seed, bal, a.csv)
            model, _ = fit_central(DNN, sp, recipe, seed)
            report(jl, base(sp, "P1", "Centralized", "DNN", f"P1 {opt}+{bal}"), sp.yva,
                   predict_proba(model, sp.Xva), sp.yte, predict_proba(model, sp.Xte), seed,
                   extra=dict(recipe=recipe))


# ------------------------------------------------------------------ P2
def stage_threshold(jl, protocol, save=True):
    """Pick the best P1 setting by mean VALIDATION F1 (threshold-tuned), save it as the shared
    recipe, and show the fixed-0.5 vs tuned-threshold scores for it."""
    df = load_rows(jl)
    df = df[(df.stage == "P1") & (df.protocol == protocol)]
    if df.empty:
        print(f"P2/{protocol}: no P1 rows yet -- run stage optim first")
        return
    tuned = df[df.thr_mode == "+thr"].groupby("setting").val_f1.mean().sort_values(ascending=False)
    best = tuned.index[0]
    rec = df[df.setting == best].iloc[0]["recipe"]
    if save:
        save_recipe(protocol, rec)
    else:
        print("  (quick mode: recipe NOT saved)")
    print(f"P2/{protocol}: validation F1 by setting:\n{tuned.round(4).to_string()}\n  -> best = {best}  recipe={rec}")
    sub = df[df.setting == best].groupby("thr_mode")[["acc", "prec", "rec", "f1", "auc"]].mean().round(4)
    print("  test scores of the chosen setting (mean over seeds):\n" + sub.to_string())


# ------------------------------------------------------------------ P3
def stage_cnn(a, jl, protocol, seed, _):
    r = load_recipe(protocol)
    sp = make_split(protocol, seed, r["balance"], a.csv)
    model, _ = fit_central(CNN, sp, dict(r, epochs=a.epochs if a.quick else r["epochs"]), seed)
    report(jl, base(sp, "P3", "Centralized", "CNN", f"P3 best recipe ({r['opt']}+{r['balance']})"), sp.yva,
           predict_proba(model, sp.Xva), sp.yte, predict_proba(model, sp.Xte), seed, extra=dict(recipe=r))


# ------------------------------------------------------------------ P4
def stage_fedavg(a, jl, protocol, seed, _):
    r = load_recipe(protocol)
    sp = make_split(protocol, seed, "none", a.csv)
    parts = iid_parts(sp.Xtr, sp.ytr, seed, 3)
    res = run_fl(DNN, parts, a.rounds, a.local_epochs, r, seed=seed, client_bal=client_balance(protocol, r["balance"]))
    g = res["cstates"][0]
    report(jl, base(sp, "P4", "Federated (no DP)", "DNN", f"P4 FedAvg 3 IID clients ({a.rounds}r x {a.local_epochs}e)"),
           sp.yva, eval_state(DNN, g, sp.Xva), sp.yte, eval_state(DNN, g, sp.Xte), seed,
           extra=dict(recipe=r, rounds=a.rounds, local_epochs=a.local_epochs))


# ------------------------------------------------------------------ P5
def stage_dp(a, jl, protocol, seed, _):
    r = load_recipe(protocol)
    sp = make_split(protocol, seed, "none", a.csv)
    parts = iid_parts(sp.Xtr, sp.ytr, seed, 3)
    for cfg in dp_grid(a.grid, r):
        print(f"   {cfg['setting']}", flush=True)
        rr = dict(r, opt=cfg["opt"], lr=cfg["lr"])
        dp = dict(noise=cfg["noise"], clip=cfg["clip"], delta=cfg["delta"], target_eps=cfg["target_eps"])
        res = run_fl(DNN, parts, a.rounds, a.local_epochs, rr, dp=dp, seed=seed,
                     client_bal=client_balance(protocol, r["balance"]), trace_eps=a.quick)
        g = res["cstates"][0]
        report(jl, base(sp, "P5", "Federated + DP", "DNN", cfg["setting"], "yes (epsilon, delta)"), sp.yva,
               eval_state(DNN, g, sp.Xva), sp.yte, eval_state(DNN, g, sp.Xte), seed,
               extra=dict(eps=round(float(res["eps"]), 4), delta=cfg["delta"], noise=round(float(res["noise"]), 4),
                          clip=cfg["clip"], eps_trace=res["eps_trace"], recipe=rr, rounds=a.rounds,
                          local_epochs=a.local_epochs))


# ------------------------------------------------------------------ P6 / P7
def _gbt(it, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, class_weight="balanced",
                                          early_stopping=True, validation_fraction=0.15, random_state=seed)


def stage_gbt(a, jl, protocol, seed, it):
    sp = make_split(protocol, seed, "none", a.csv)
    m = _gbt(it, seed).fit(sp.Xtr, sp.ytr)
    report(jl, base(sp, "P6", "Centralized", "GBT (centralized: no FL, no DP; upper bound)", "P6 HistGradientBoosting",
                    "no (no FL, no DP)"), sp.yva, m.predict_proba(sp.Xva)[:, 1], sp.yte, m.predict_proba(sp.Xte)[:, 1], seed)


def stage_fedgbt(a, jl, protocol, seed, it):
    sp = make_split(protocol, seed, "none", a.csv)
    parts = iid_parts(sp.Xtr, sp.ytr, seed, 3)
    ms = [_gbt(it, seed).fit(X, y) for X, y in parts]
    w = [len(y) for _, y in parts]
    ens = lambda X: np.average([m.predict_proba(X)[:, 1] for m in ms], axis=0, weights=w)
    report(jl, base(sp, "P7", "Federated (no DP)", "FedGBT (prediction averaging)", "P7 3 IID clients",
                    "no DP guarantee"), sp.yva, ens(sp.Xva), sp.yte, ens(sp.Xte), seed)


if __name__ == "__main__":
    main()

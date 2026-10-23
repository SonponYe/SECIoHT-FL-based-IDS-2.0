"""Track MoE: the design document's MoE + domain-aware aggregation applied to WUSTL.

Run from our_work/ :   python -m track_moe.run_moe --stage central --protocol honest --seeds 0 1 2
Stages: central(M1) fedavg(M2) dp(M3) noniid(M4) noniid_dp(M5) | all
The training recipe is the SAME one Track Paper selected (shared/best_recipe.json), so MoE and DNN
get identical optimizer / epochs / batch / balancing / rounds / DP settings.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from shared import fl
from shared.data import DEFAULT_CSV, client_balance, iid_parts, make_split, partition_noniid
from shared.metrics import log_row, report, report_fixed, summarize
from shared.models_base import DNN
from track_moe.aggregation import domain_aggregator, local_aggregator
from track_moe.models_moe import MoE

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TRACK = "moe"
STAGES = ["central", "fedavg", "dp", "noniid", "noniid_dp"]
AUX_W = 0.01


def base(split, stage, group, model, setting, dp_guarantee="no (no DP)"):
    return dict(track=TRACK, protocol=split.info["protocol"], stage=stage, group=group, model=model,
                setting=setting, dp_guarantee=dp_guarantee, majority_acc=split.info["majority_acc"],
                test_attack_rate=split.info["test_attack_rate"])


def moe_extra(states, Xs, K=4):
    """Expert usage share and routing entropy, averaged over the given (state, data) pairs."""
    shares, ents = [], []
    for st, X in zip(states, Xs):
        m = MoE()
        fl.set_state(m, st)
        t = torch.tensor(X, dtype=torch.float32)
        shares.append(m.routing_stats(t)[1])
        ents.append(m.routing_entropy(t))
    return dict(expert_share=[round(float(v), 4) for v in np.mean(shares, axis=0)],
                routing_entropy=round(float(np.mean(ents)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--protocol", default="both", choices=["paper", "honest", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=0, help="override recipe epochs (centralised)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--grid", default="full", choices=["full", "small"])
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
    print(f"== track_moe | stages={stages} protocols={protocols} seeds={a.seeds} quick={a.quick} -> {jl}", flush=True)
    for stage in stages:
        for protocol in protocols:
            for seed in a.seeds:
                t0 = time.time()
                globals()["stage_" + stage](a, jl, protocol, seed)
                secs = round(time.time() - t0, 1)
                log_row(out / "timing.jsonl", dict(stage=stage, protocol=protocol, seed=seed, secs=secs, quick=a.quick))
                print(f"  -- {stage}/{protocol}/seed{seed} took {secs}s", flush=True)
    summarize(jl, out / "summary.csv")
    print("summary ->", out / "summary.csv")


# ------------------------------------------------------------------ M1
def stage_central(a, jl, protocol, seed):
    r = fl.load_recipe(protocol)
    sp = make_split(protocol, seed, r["balance"], a.csv)
    model, _ = fl.fit_central(MoE, sp, r, seed, epochs=a.epochs or r["epochs"], aux_w=AUX_W)
    t = torch.tensor(sp.Xte, dtype=torch.float32)
    extra = dict(recipe=r, expert_share=[round(float(v), 4) for v in model.routing_stats(t)[1]],
                 routing_entropy=round(model.routing_entropy(t), 4))
    print("   expert share", extra["expert_share"], "entropy", extra["routing_entropy"])
    report(jl, base(sp, "M1", "Centralized", "MoE", f"M1 MoE ({r['opt']}+{r['balance']})"), sp.yva,
           fl.predict_proba(model, sp.Xva), sp.yte, fl.predict_proba(model, sp.Xte), seed, extra=extra)


# ------------------------------------------------------------------ M2
def stage_fedavg(a, jl, protocol, seed):
    r = fl.load_recipe(protocol)
    sp = make_split(protocol, seed, "none", a.csv)
    parts = iid_parts(sp.Xtr, sp.ytr, seed, 3)
    res = fl.run_fl(MoE, parts, a.rounds, a.local_epochs, r, aux_w=AUX_W, seed=seed,
                    client_bal=client_balance(protocol, r["balance"]))
    g = res["cstates"][0]
    extra = dict(recipe=r, rounds=a.rounds, local_epochs=a.local_epochs, **moe_extra([g], [sp.Xte]))
    report(jl, base(sp, "M2", "Federated (no DP)", "MoE", f"M2 FedAvg 3 IID clients ({a.rounds}r x {a.local_epochs}e)"),
           sp.yva, fl.eval_state(MoE, g, sp.Xva), sp.yte, fl.eval_state(MoE, g, sp.Xte), seed, extra=extra)


# ------------------------------------------------------------------ M3
def stage_dp(a, jl, protocol, seed):
    """Identical grid to Track Paper P5. Aux loss weight is 0 under DP: the balancing term averages
    over the batch and would couple samples, breaking the per-sample sensitivity analysis."""
    r = fl.load_recipe(protocol)
    sp = make_split(protocol, seed, "none", a.csv)
    parts = iid_parts(sp.Xtr, sp.ytr, seed, 3)
    for cfg in fl.dp_grid(a.grid, r):
        print(f"   {cfg['setting']}", flush=True)
        rr = dict(r, opt=cfg["opt"], lr=cfg["lr"])
        dp = dict(noise=cfg["noise"], clip=cfg["clip"], delta=cfg["delta"], target_eps=cfg["target_eps"])
        res = fl.run_fl(MoE, parts, a.rounds, a.local_epochs, rr, dp=dp, aux_w=0.0, seed=seed,
                        client_bal=client_balance(protocol, r["balance"]), trace_eps=a.quick)
        g = res["cstates"][0]
        extra = dict(eps=round(float(res["eps"]), 4), delta=cfg["delta"], noise=round(float(res["noise"]), 4),
                     clip=cfg["clip"], eps_trace=res["eps_trace"], recipe=rr, rounds=a.rounds,
                     local_epochs=a.local_epochs, **moe_extra([g], [sp.Xte]))
        report(jl, base(sp, "M3", "Federated + DP", "MoE", cfg["setting"], "yes (epsilon, delta)"), sp.yva,
               fl.eval_state(MoE, g, sp.Xva), sp.yte, fl.eval_state(MoE, g, sp.Xte), seed, extra=extra)


# ------------------------------------------------------------------ M4 / M5
def _noniid(a, jl, protocol, seed, stage, dp_cfg):
    r = fl.load_recipe(protocol)
    sp = make_split(protocol, seed, "none", a.csv)
    tr, te = partition_noniid(sp.Xtr, sp.ytr, sp.Xte, sp.yte, seed=seed)
    parts = [(sp.Xtr[i], sp.ytr[i]) for i in tr]
    tests = [(sp.Xte[i], sp.yte[i]) for i in te]
    print("   client sizes", [len(p[1]) for p in parts], "attack rates", [round(float(p[1].mean()), 2) for p in parts])
    cb = client_balance(protocol, r["balance"])
    dp = dp_cfg
    dpg = "yes (epsilon, delta)" if dp else "no (no DP)"
    group = "Non-IID + DP" if dp else "Non-IID"
    tag = f"6 clients, Dirichlet 0.3, k-means domains, {a.rounds}r x {a.local_epochs}e" + (", DP noise 1.5 clip 1 (delta 1e-4)" if dp else "")

    def run(make_model, aggregator, aux):
        return fl.run_fl(make_model, parts, a.rounds, a.local_epochs, r, aggregator=aggregator, dp=dp,
                         aux_w=0.0 if dp else aux, seed=seed, client_bal=cb, trace_eps=a.quick)

    def dp_extra(res):
        return dict(eps=round(float(res["eps"]), 4), delta=dp["delta"], noise=round(float(res["noise"]), 4),
                    clip=dp["clip"]) if dp else {}

    # FedAvg DNN / MoE: one global model scored on the pooled test set
    for name, mk, aux in (("DNN", DNN, 0.0), ("MoE", MoE, AUX_W)):
        res = run(mk, fl.fedavg_agg, aux)
        g = res["cstates"][0]
        extra = dict(recipe=r, **dp_extra(res), **(moe_extra([g], [sp.Xte]) if name == "MoE" else {}))
        report_fixed(jl, base(sp, stage, group, name, f"{stage.upper()} FedAvg global ({tag})", dpg), sp.yte,
                     fl.eval_state(mk, g, sp.Xte), seed, extra=extra)
    # Domain-aware MoE and local-only MoE: personalised models scored on each client's own test slice
    for name, aggr, note in (("MoE domain-aware", domain_aggregator(), dpg if not dp else "yes (weights only; descriptors un-noised)"),
                             ("MoE local-only", local_aggregator, dpg if not dp else "yes (local DP, no communication)")):
        res = run(MoE, aggr, AUX_W)
        y, p = fl.eval_personalised(MoE, res["cstates"], tests)
        extra = dict(recipe=r, **res["info"], **dp_extra(res), **moe_extra(res["cstates"], [t[0] for t in tests]))
        report_fixed(jl, base(sp, stage, group, name, f"{stage.upper()} personalised ({tag})", note), y, p, seed, extra=extra)


def stage_noniid(a, jl, protocol, seed):
    _noniid(a, jl, protocol, seed, "M4", None)


def stage_noniid_dp(a, jl, protocol, seed):
    _noniid(a, jl, protocol, seed, "M5", dict(noise=1.5, clip=1.0, delta=fl.DP_DELTA, target_eps=None))


if __name__ == "__main__":
    main()

"""baseline_2023: reconstruct the 2023 DNN-FL (INFOCOM WKSHPS) as a transparent reference.

Run from our_work/ :   python -m baseline_2023.run --stage federated --clients 5 10 15 --seeds 0 1 2
Stages: central(B0)  federated(B1)  | all
Defaults follow the paper's Table II: client Adam lr 0.01, server Adam lr 0.05, batch 1024, 100 rounds,
100 local epochs.  Everything the paper does not state is documented in README.md.
"""
import argparse
import time
import warnings
from pathlib import Path

import torch

from baseline_2023.federated import run_fl_2023
from baseline_2023.model import DNN2023
from baseline_2023.preprocessing import make_data_2023
from baseline_2023.training import fit_central_2023, predict_proba_2023
from stages.s1_preprocessing.loading import DEFAULT_CSV
from stages.s6_evaluation.reporting import log_row, report
from stages.s6_evaluation.summary import summarize

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TRACK = "baseline-2023"


def base(d, stage, group, model, setting):
    return dict(track=TRACK, protocol="honest", stage=stage, group=group, model=model, setting=setting,
                dp_guarantee="no (no DP)", majority_acc=d["info"]["majority_acc"],
                test_attack_rate=d["info"]["test_attack_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["central", "federated", "all"])
    ap.add_argument("--clients", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local-epochs", type=int, default=100)
    ap.add_argument("--central-epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.quick:
        a.rounds, a.local_epochs, a.central_epochs, a.clients = 2, 1, 2, [5]
    out = HERE / "results" / ("quick" if a.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "results.jsonl"
    stages = ["central", "federated"] if a.stage == "all" else [a.stage]
    print(f"== baseline_2023 | stages={stages} clients={a.clients} seeds={a.seeds} quick={a.quick} -> {jl}", flush=True)
    for stage in stages:
        for seed in a.seeds:
            for n in ([a.clients[0]] if stage == "central" else a.clients):
                t0 = time.time()
                d = make_data_2023(seed, n, a.csv)
                if stage == "central":
                    # paper's centralised DNN: lr 0.1, batch 20 (optimizer/epochs unstated -> SGD, 100 epochs)
                    m = fit_central_2023(DNN2023, *d["central"], a.central_epochs, 0.1, 20, "sgd", seed)
                    b = base(d, "B0", "Centralized", "DNN (2023 centralised baseline)", "B0 undersampled, lr 0.1, batch 20")
                else:
                    m = run_fl_2023(DNN2023, d["clients"], a.rounds, a.local_epochs, 0.01, 0.05, 1024, seed)
                    b = base(d, "B1", "Federated (no DP)", "DNN-FL 2023", f"B1 DNN-FL {n} clients ({a.rounds}r x {a.local_epochs}e)")
                report(jl, b, d["yva"], predict_proba_2023(m, d["Xva"]), d["yte"], predict_proba_2023(m, d["Xte"]), seed,
                       extra=dict(n_clients=n, rounds=a.rounds, local_epochs=a.local_epochs))
                secs = round(time.time() - t0, 1)
                log_row(out / "timing.jsonl", dict(stage=stage, seed=seed, clients=n, secs=secs, quick=a.quick))
                print(f"  -- {stage}/clients{n}/seed{seed} took {secs}s", flush=True)
    summarize(jl, out / "summary.csv")


if __name__ == "__main__":
    main()

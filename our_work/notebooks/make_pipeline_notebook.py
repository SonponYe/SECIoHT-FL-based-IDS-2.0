"""Generates 01_pipeline.ipynb. The notebook only CALLS the stage blocks in stages/; change behaviour there."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# 01 - The pipeline, block by block

This notebook **only calls** the building blocks in `our_work/stages/`, in pipeline order:
S1 preprocessing -> S2 model -> S3 training -> S4 federated -> S5 privacy -> S6 evaluation.
**To change how anything works, edit the block in `stages/`, not this notebook.**
Edit only the settings in the next cell.
"""),
    code("""
import sys, pathlib
ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent                      # our_work/
sys.path.insert(0, str(ROOT))               # lets us import stages, moe, baseline_2023
print("project root:", ROOT)
"""),
    md("## Settings (the only cell you edit)"),
    code("""
PROTOCOL     = "honest"     # "honest" (trusted) or "paper" (balanced-leak imitation)
MODEL        = "dnn"        # "dnn" | "cnn" | "moe"
SEED         = 0
QUICK        = True         # True = tiny run (2 rounds, 3 epochs) to check the plumbing
ROUNDS       = 2 if QUICK else 10
LOCAL_EPOCHS = 1 if QUICK else 3
CENTRAL_EPOCHS = 3 if QUICK else None       # None = use the recipe's epochs
DP_ON        = False        # True = add differential privacy to the federated run (S5)
DP_NOISE, DP_CLIP = 1.5, 1.0                # delta is fixed at 1e-4 (the paper's)
OUT = ROOT / "notebooks" / "_runs" / "results.jsonl"      # git-ignored
OUT.parent.mkdir(exist_ok=True)
"""),
    md("## S1 - Preprocessing\nClean, split (per protocol), balance, scale. Test set is fixed; thresholds are tuned on a validation slice."),
    code("""
from stages.s3_training.recipes import load_recipe
from stages.s1_preprocessing.protocols import make_split, client_balance
from stages.s1_preprocessing.partitioning import iid_parts

recipe = load_recipe(PROTOCOL)              # best training settings found by paper_2025 stage `threshold`
split = make_split(PROTOCOL, SEED, recipe["balance"])
print("recipe:", recipe)
print(split.info)
print("all-normal baseline accuracy on this test set:", split.info["majority_acc"])
"""),
    md("## S2 - Build the model"),
    code("""
from stages.s2_models.registry import get_model_factory
make_model = get_model_factory(MODEL)
aux_w = 0.01 if MODEL == "moe" else 0.0     # MoE load-balancing term (switched off under DP)
print(make_model())
"""),
    md("## S3 - Train centrally (one place, no federation)"),
    code("""
from stages.s3_training.centralized import fit_central
from stages.s3_training.loops import predict_proba
from stages.s6_evaluation.reporting import report

base = dict(track="notebook", protocol=PROTOCOL, stage="NB", group="Centralized", model=MODEL.upper(),
            setting="notebook central", dp_guarantee="no (no DP)",
            majority_acc=split.info["majority_acc"], test_attack_rate=split.info["test_attack_rate"])
model, val_f1 = fit_central(make_model, split, recipe, SEED, epochs=CENTRAL_EPOCHS, aux_w=aux_w)
_ = report(OUT, base, split.yva, predict_proba(model, split.Xva), split.yte, predict_proba(model, split.Xte), SEED)
"""),
    md("## S4 (+ S5) - Federated training across 3 simulated hospitals"),
    code("""
from stages.s4_federated.simulation import run_fl, eval_state

dp = dict(noise=DP_NOISE, clip=DP_CLIP, delta=1e-4, target_eps=None) if DP_ON else None   # S5
sp_fl = make_split(PROTOCOL, SEED, "none")                      # unbalanced; clients balance themselves
parts = iid_parts(sp_fl.Xtr, sp_fl.ytr, SEED, 3)
res = run_fl(make_model, parts, ROUNDS, LOCAL_EPOCHS, recipe, dp=dp, aux_w=0.0 if DP_ON else aux_w, seed=SEED,
             client_bal=client_balance(PROTOCOL, recipe["balance"]), trace_eps=DP_ON)
g = res["cstates"][0]
if DP_ON:
    print("epsilon spent:", round(res["eps"], 3), " delta: 1e-4  noise multiplier:", round(res["noise"], 3))
"""),
    md("## S6 - Evaluate and compare"),
    code("""
import pandas as pd
base_fl = dict(base, group="Federated + DP" if DP_ON else "Federated (no DP)", setting="notebook FedAvg x3",
               dp_guarantee="yes (epsilon, delta)" if DP_ON else "no (no DP)")
_ = report(OUT, base_fl, sp_fl.yva, eval_state(make_model, g, sp_fl.Xva), sp_fl.yte, eval_state(make_model, g, sp_fl.Xte), SEED,
           extra=dict(eps=res["eps"], delta=1e-4) if DP_ON else {})

from stages.s6_evaluation.summary import load_rows
df = load_rows(OUT)
df[["group", "model", "thr_mode", "acc", "prec", "rec", "f1", "macro_f1", "rec_normal", "auc"]].tail(4).round(3)
"""),
    md("""
## The 2023 baseline (its own blocks)

The 2023 paper uses a different preprocessing (client split, undersampling), model (64-64-32, sigmoid), loss (BCE) and server (Adam).
Those blocks live in `baseline_2023/`, in the same stage order.
"""),
    code("""
from baseline_2023.preprocessing import make_data_2023     # S1
from baseline_2023.model import DNN2023                    # S2
from baseline_2023.federated import run_fl_2023            # S3/S4
from baseline_2023.training import predict_proba_2023      # S6

d = make_data_2023(SEED, n_clients=5)
m = run_fl_2023(DNN2023, d["clients"], rounds=2 if QUICK else 100, local_epochs=1,
                client_lr=0.01, server_lr=0.05, batch=1024, seed=SEED)
base23 = dict(track="notebook", protocol="honest", stage="NB23", group="Federated (no DP)", model="DNN-FL 2023",
              setting="notebook 5 clients", dp_guarantee="no (no DP)", majority_acc=d["info"]["majority_acc"],
              test_attack_rate=d["info"]["test_attack_rate"])
_ = report(OUT, base23, d["yva"], predict_proba_2023(m, d["Xva"]), d["yte"], predict_proba_2023(m, d["Xte"]), SEED)
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "01_pipeline.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
print("wrote 01_pipeline.ipynb")

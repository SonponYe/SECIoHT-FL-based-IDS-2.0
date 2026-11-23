"""Checks for step 7 (the 2023 DNN trained centrally on the honest pipeline).

Run from our_work/:   python -m baseline_2023.audit.check_central_dnn
Trains seeds 0 1 2 again WITHOUT writing results (so the logged results are not duplicated), then checks them.
Thresholds below are fixed by the brief and are never adjusted to make a check pass.  Exit code 1 if any check fails.
"""
from __future__ import annotations

import sys

import numpy as np
import torch.nn as nn

from baseline_2023 import fl2023 as F
from stages.s6_evaluation.metrics import best_threshold

RESULTS: list[bool] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main() -> int:
    runs = F.run_central(seeds=(0, 1, 2), write=False)

    m = F.make_model()
    lin = [l for l in m.net if isinstance(l, nn.Linear)]
    limits = [float(np.sqrt(6.0 / (l.in_features + l.out_features))) for l in lin]
    rec("model is 36 -> 64 -> 64 -> 32 -> 1 with Keras-style init (zero biases, Glorot-uniform weights)",
        [(l.in_features, l.out_features) for l in lin] == [(36, 64), (64, 64), (64, 32), (32, 1)] and
        all(float(l.bias.detach().abs().max()) == 0.0 and float(l.weight.detach().abs().max()) <= lim + 1e-9 and
            float(l.weight.detach().abs().max()) > 0.5 * lim for l, lim in zip(lin, limits)),
        f"{sum(p.numel() for p in m.parameters())} parameters")
    opt = F.make_optimizer(m)
    g = opt.param_groups[0]
    rec("optimizer is Adam with lr 0.01, epsilon 1e-7; batch 1024; up to 100 epochs",
        type(opt).__name__ == "Adam" and g["lr"] == 0.01 and g["eps"] == 1e-7 and F.BATCH == 1024 and F.MAX_EPOCHS == 100,
        f"lr {g['lr']}, eps {g['eps']}, batch {F.BATCH}, {F.MAX_EPOCHS} epochs")

    for r in runs:
        s, h = r["seed"], r["history"]
        n = r["feature_names"]
        rec(f"seed {s}: input size 36 with Temp and Dport in feature_names", len(n) == 36 and r["n_features"] == 36 and "Temp" in n and "Dport" in n,
            f"{len(n)} features")
        rec(f"seed {s}: training loss at the last epoch < training loss at epoch 1", h["train_loss"][-1] < h["train_loss"][0],
            f"{h['train_loss'][0]:.4f} -> {h['train_loss'][-1]:.4f}")
        rec(f"seed {s}: validation AUC of the chosen checkpoint > 0.9",
            r["best_val_auc"] > 0.9 and r["best_val_auc"] == max(h["val_auc"]) and h["val_auc"][r["best_epoch"] - 1] == r["best_val_auc"],
            f"epoch {r['best_epoch']} of {r['epochs_run']}, val AUC {r['best_val_auc']:.4f}")
        rec(f"seed {s}: the test set is scored exactly once", r["test_calls"] == 1, f"{r['test_calls']} call(s)")
        rec(f"seed {s}: 1:1 balanced training rows, trained for the full {F.MAX_EPOCHS} epochs, timed",
            r["rows"][0]["n_train_rows"] == 2782 and r["epochs_run"] == F.MAX_EPOCHS and len(h["train_loss"]) == F.MAX_EPOCHS and r["train_seconds"] > 0,
            f"{r['rows'][0]['n_train_rows']} rows, {r['train_seconds']:.1f}s")
        thr_again, f1_again = best_threshold(r["y_val"], r["val_probs"])
        rec(f"seed {s}: the tuned threshold is the F1-maximiser on the VALIDATION set (recomputed from validation probabilities)",
            thr_again == r["threshold"] and f1_again == r["val_f1"] and r["rows"][1]["thr"] == r["threshold"],
            f"threshold {r['threshold']:.2f}, validation F1 {r['val_f1']:.3f}")

    hashes = {r["test_hash"] for r in runs}
    rec("the test set is identical across seeds (hash of inputs, labels and categories)", len(hashes) == 1, f"{sorted(hashes)}")
    rec("majority-class accuracy of the test set is about 0.875", all(abs(r["rows"][0]["majority_acc"] - 0.875) < 0.005 for r in runs),
        f"{runs[0]['rows'][0]['majority_acc']:.4f}")

    try:
        logged, _ = F.load_central()
        diffs = []
        for r in runs:
            for row in r["rows"]:
                old = next(x for x in logged if x["seed"] == row["seed"] and x["thr_mode"] == row["thr_mode"])
                diffs.append(abs(old["f1"] - row["f1"]) + abs(old["auc"] - row["auc"]))
        rec("re-running reproduces the logged results (deterministic)", max(diffs) < 1e-9, f"max difference {max(diffs):.1e} over {len(diffs)} rows")
    except FileNotFoundError:
        print("[INFO] no logged results yet; skipping the reproducibility check")

    bad = RESULTS.count(False)
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} checks passed" + (f"  ({bad} FAILED)" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

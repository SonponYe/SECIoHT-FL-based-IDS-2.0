"""Metrics, validation threshold selection, JSONL logging and mean±std summaries."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

KEYS = ["track", "protocol", "stage", "group", "model", "setting", "thr_mode"]
VALS = ["acc", "prec", "rec", "f1", "auc"]


def metrics(y, p, thr=0.5):
    """p = P(attack)."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=f1, auc=auc,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), thr=float(thr))


def best_threshold(y, p):
    """Threshold that maximises F1 on a VALIDATION set. Returns (thr, val_f1)."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        f = metrics(y, p, t)["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def log_row(path, row):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def _print(base, mode, seed, m):
    print(f"[{base['protocol']}/{base['stage']}] {base['model']} | {base['setting']} {mode} s{seed}: "
          f"acc={m['acc']:.4f} P={m['prec']:.3f} R={m['rec']:.3f} F1={m['f1']:.3f} AUC={m['auc']:.3f}",
          flush=True)


def report(path, base, yva, pva, yte, pte, seed, extra=None):
    """Log the @0.5 row and the validation-tuned-threshold row for one model."""
    extra = extra or {}
    thr, val_f1 = best_threshold(yva, pva)
    rows = []
    for mode, t in (("@0.5", 0.5), ("+thr", thr)):
        m = metrics(yte, pte, t)
        row = dict(base, seed=seed, thr_mode=mode, val_f1=round(val_f1, 4), **m, **extra)
        log_row(path, row)
        rows.append(row)
        _print(base, mode, seed, m)
    return rows


def report_fixed(path, base, y, p, seed, extra=None):
    """Single row at thr 0.5 (used where no validation slice applies, e.g. personalised models)."""
    m = metrics(y, p, 0.5)
    row = dict(base, seed=seed, thr_mode="@0.5", val_f1=None, **m, **(extra or {}))
    log_row(path, row)
    _print(base, "@0.5", seed, m)
    return row


def load_rows(path):
    with open(path, encoding="utf8") as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def summarize(jsonl_path, csv_path=None, keys=KEYS):
    """mean ± std over seeds per experiment key."""
    df = load_rows(jsonl_path)
    keys = [k for k in keys if k in df.columns]
    g = df.groupby(keys, dropna=False)
    out = g[VALS].mean().add_suffix("_mean").join(g[VALS].std().add_suffix("_std"))
    out["n_seeds"] = g.size()
    for c in ("tp", "tn", "fp", "fn", "thr", "eps", "delta", "noise", "clip", "routing_entropy"):
        if c in df.columns:
            out[c] = g[c].mean()
    for c in ("dp_guarantee", "majority_acc", "test_attack_rate"):
        if c in df.columns:
            out[c] = g[c].first()
    out = out.reset_index()
    if csv_path:
        out.to_csv(csv_path, index=False)
    return out


def fmt(mean, std):
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}"

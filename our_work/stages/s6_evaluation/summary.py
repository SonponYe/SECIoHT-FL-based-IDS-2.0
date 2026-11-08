"""S6c - mean ± std over seeds."""
import json

import pandas as pd

KEYS = ["track", "protocol", "stage", "group", "model", "setting", "thr_mode"]
VALS = ["acc", "prec", "rec", "f1", "auc", "rec_normal", "macro_f1"]


def load_rows(path):
    """Read results.jsonl; rows written before macro-F1 existed are back-filled from their confusion counts."""
    with open(path, encoding="utf8") as f:
        df = pd.DataFrame([json.loads(l) for l in f if l.strip()])
    if "macro_f1" not in df.columns:
        df["macro_f1"], df["rec_normal"] = float("nan"), float("nan")
    need = df["macro_f1"].isna() & df[["tp", "tn", "fp", "fn"]].notna().all(axis=1)
    if need.any():
        tp, tn, fp, fn = (df.loc[need, c].astype(float) for c in ("tp", "tn", "fp", "fn"))
        prec0 = (tn / (tn + fn)).fillna(0.0)
        rec0 = (tn / (tn + fp)).fillna(0.0)
        f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)).fillna(0.0)
        df.loc[need, "rec_normal"] = rec0
        df.loc[need, "macro_f1"] = (df.loc[need, "f1"] + f1_0) / 2
    return df


def summarize(jsonl_path, csv_path=None, keys=KEYS):
    """One row per experiment key: mean and std of each score over seeds."""
    df = load_rows(jsonl_path)
    keys = [k for k in keys if k in df.columns]
    vals = [v for v in VALS if v in df.columns]      # older results have no macro_f1 / rec_normal
    g = df.groupby(keys, dropna=False)
    out = g[vals].mean().add_suffix("_mean").join(g[vals].std().add_suffix("_std"))
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

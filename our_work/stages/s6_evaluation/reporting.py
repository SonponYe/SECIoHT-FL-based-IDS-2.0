"""S6b - write one JSON line per run (the single source of truth for results)."""
import json

from stages.s6_evaluation.metrics import best_threshold, metrics


def log_row(path, row):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def _print(base, mode, seed, m):
    print(f"[{base['protocol']}/{base['stage']}] {base['model']} | {base['setting']} {mode} s{seed}: "
          f"acc={m['acc']:.4f} P={m['prec']:.3f} R={m['rec']:.3f} F1={m['f1']:.3f} AUC={m['auc']:.3f} "
          f"macroF1={m['macro_f1']:.3f}", flush=True)


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
    """Single row at threshold 0.5 (used where no validation slice applies, e.g. personalised models)."""
    m = metrics(y, p, 0.5)
    row = dict(base, seed=seed, thr_mode="@0.5", val_f1=None, **m, **(extra or {}))
    log_row(path, row)
    _print(base, "@0.5", seed, m)
    return row

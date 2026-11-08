"""S6a - scores and validation-tuned decision thresholds."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def metrics(y, p, thr=0.5):
    """p = P(attack).  Reports the attack class (precision/recall/F1) plus accuracy, AUC and the
    confusion counts.  Also reports normal-class recall and macro-F1, because accuracy alone is
    misleading on imbalanced data (a model that always says "normal" scores ~87.5%)."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=f1, auc=auc,
                rec_normal=rec0, macro_f1=(f1 + f1_0) / 2,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), thr=float(thr))


def best_threshold(y, p):
    """Threshold that maximises F1 on a VALIDATION set. Returns (thr, val_f1)."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        f = metrics(y, p, t)["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f

"""S1c - the two evaluation protocols, validation split and scaling.

TWO PROTOCOLS (every experiment can run under both)
---------------------------------------------------
honest  Our leak-free pipeline: stratified 80/20 split on REAL data first -> a 15% validation
        slice is carved from the real training part -> the training part is balanced (SMOTE,
        class weights or nothing) -> scaler fit on the training part only.  The test set stays
        100% real (~12.5% attack).  For federated runs SMOTE is applied per client on that
        client's own shard only (as in notebook Stage 3).  THIS IS THE PROTOCOL WE TRUST.

paper   A *guess* at how the 2025 paper's test set could have become easy ("balanced-leak"):
        SMOTE on the WHOLE cleaned dataset -> StandardScaler on everything -> stratified 80/20
        split (seed 42).  Synthetic rows land in the TEST set, so it is ~50% attack.
        CAUTION: the replication audit (references/SECIoHT_FL_Replication_Audit.pptx) shows the
        authors' notebook actually used a larger, ~89%-attack test set and undersampling, built
        from a private file.  So this protocol does NOT reproduce the paper's data; it only
        shows how much an easy test set can inflate scores.  The key "paper" is kept so that
        stored results stay valid.

Checkpoints, thresholds and hyper-parameters are chosen on the validation slice ONLY; the test
set is touched once for the final score.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from stages.s1_preprocessing.balancing import class_weights, smote_resample
from stages.s1_preprocessing.loading import load_clean

SPLIT_SEED = 42
VAL_FRAC = 0.15


@dataclass
class Split:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    class_w: np.ndarray
    info: dict = field(default_factory=dict)


def make_split(protocol, seed, balance="smote", csv=None, val_frac=VAL_FRAC):
    """balance: 'smote' | 'weights' | 'none' (honest only; `paper` is always pre-SMOTEd).
    `seed` only affects the validation carve-out and SMOTE randomness; the test set is fixed."""
    assert protocol in ("paper", "honest"), protocol
    X, y = load_clean(csv)
    if protocol == "paper":
        Xs, ys = smote_resample(X, y, SPLIT_SEED)
        Xs = StandardScaler().fit_transform(Xs)
        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=SPLIT_SEED)
        if val_frac:
            Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=val_frac, stratify=ytr, random_state=100 + seed)
        else:
            Xva, yva = Xtr[:0], ytr[:0]
        balance_used = "pre-SMOTE (whole dataset)"
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
        if val_frac:
            Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=val_frac, stratify=ytr, random_state=100 + seed)
        else:
            Xva, yva = Xtr[:0], ytr[:0]
        if balance == "smote":
            Xtr, ytr = smote_resample(Xtr, ytr, seed)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        Xva = sc.transform(Xva) if len(Xva) else Xva
        balance_used = balance
    rate = float(yte.mean())
    if protocol == "paper":
        assert abs(rate - 0.5) < 0.03, f"paper protocol test attack rate {rate:.3f}, expected ~0.50"
    else:
        assert abs(rate - 0.125) < 0.02, f"honest protocol test attack rate {rate:.3f}, expected ~0.125"
    info = dict(protocol=protocol, balance=balance_used, n_train=len(ytr), n_val=len(yva), n_test=len(yte),
                test_attack_rate=round(rate, 4), train_attack_rate=round(float(ytr.mean()), 4),
                majority_acc=round(max(rate, 1 - rate), 4))
    return Split(Xtr, ytr, Xva, yva, Xte, yte, class_weights(ytr), info)


def client_balance(protocol, balance):
    """What each federated client should do locally: `paper` data is already balanced."""
    return "none" if protocol == "paper" else balance

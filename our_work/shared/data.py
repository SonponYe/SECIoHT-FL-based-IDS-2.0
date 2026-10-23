"""Data loading, the two evaluation protocols, validation split and client partitions.

TWO PROTOCOLS (both are supported by every experiment)
------------------------------------------------------
paper   Replicates what the published paper most likely did (HYPOTHESIS, tested by stage
        P0): SMOTE on the WHOLE cleaned dataset -> StandardScaler on everything ->
        stratified 80/20 split (seed 42).  Synthetic rows end up in the TEST set, so the
        test set is ~50% attack (synthetic + real) instead of the real ~12.5%.  Use this
        only to compare directly against the paper's 93.2%.  For federated runs the
        resulting training set is partitioned across clients as-is.

honest  Our leak-free pipeline: stratified 80/20 split on REAL data first -> a 15%
        validation slice is carved from the real training part -> the training part is
        balanced (SMOTE, class weights or nothing) -> scaler fit on the training part
        only.  The test set stays 100% real (~12.5% attack).  For federated runs SMOTE is
        applied per client on that client's own shard only (as in notebook Stage 3).

Checkpoints, thresholds and hyper-parameters are chosen on the validation slice ONLY; the
test set is touched once for the final score.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "wustl-ehms-2020_with_attacks_categories (1).csv"
DROP_COLS = ["Dir", "Flgs", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num", "Attack Category"]
SPLIT_SEED = 42
VAL_FRAC = 0.15
D = 36


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


def load_clean(csv=None):
    """Identical cleaning to the notebook (Stage 1/2): 36 features + Label."""
    df = pd.read_csv(csv or DEFAULT_CSV)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.drop(columns=DROP_COLS)
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X["Sport"] = pd.to_numeric(X["Sport"], errors="coerce")
    X.dropna(inplace=True)
    y = y[X.index]
    assert X.shape[1] == D, X.shape
    return X.values.astype(np.float64), y.values.astype(np.int64)


def _class_w(y):
    counts = np.bincount(y, minlength=2).astype(float)
    return len(y) / (2 * np.maximum(counts, 1))


def make_split(protocol, seed, balance="smote", csv=None, val_frac=VAL_FRAC):
    """balance: 'smote' | 'weights' | 'none' (honest only; paper is always pre-SMOTEd).
    seed only affects the validation carve-out and SMOTE randomness; the test set is fixed."""
    assert protocol in ("paper", "honest"), protocol
    from imblearn.over_sampling import SMOTE
    X, y = load_clean(csv)
    if protocol == "paper":
        Xs, ys = SMOTE(random_state=SPLIT_SEED).fit_resample(X, y)
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
            Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
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
    return Split(Xtr, ytr, Xva, yva, Xte, yte, _class_w(ytr), info)


def client_balance(protocol, balance):
    """What each federated client should do locally: paper data is already balanced."""
    return "none" if protocol == "paper" else balance


def iid_parts(X, y, seed, n=3):
    idx = np.random.RandomState(seed).permutation(len(X))
    return [(X[i], y[i]) for i in np.array_split(idx, n)]


def partition_noniid(Xtr, ytr, Xte, yte, n_clients=6, alpha=0.3, n_domains=3, seed=0):
    """Domain-skewed split: k-means traffic clusters distributed with Dirichlet(alpha).
    The same per-domain proportions are applied to the test set so each client is scored on
    data from its own distribution.  Returns (train_idx_lists, test_idx_lists)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_domains, n_init=5, random_state=seed).fit(Xtr)
    dtr, dte = km.labels_, km.predict(Xte)
    rng = np.random.RandomState(seed)
    for _ in range(200):
        tr = [[] for _ in range(n_clients)]
        te = [[] for _ in range(n_clients)]
        for d in range(n_domains):
            q = rng.dirichlet(alpha * np.ones(n_clients))
            for arr, out in ((dtr, tr), (dte, te)):
                idx = rng.permutation(np.where(arr == d)[0])
                cuts = (np.cumsum(q) * len(idx)).astype(int)[:-1]
                for c, part in enumerate(np.split(idx, cuts)):
                    out[c].extend(part.tolist())
        if all(len(tr[c]) >= 300 and len(te[c]) >= 50 and len(set(ytr[tr[c]])) == 2 and len(set(yte[te[c]])) == 2
               for c in range(n_clients)):
            return [np.array(t) for t in tr], [np.array(t) for t in te]
    raise RuntimeError("could not build a valid non-IID partition")

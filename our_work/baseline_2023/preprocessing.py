"""baseline_2023 / S1 - preprocessing as described in the 2023 paper (Section IV-C), with every
unstated detail resolved explicitly (see README.md "What the paper does NOT say").

Paper order: raw CSV -> divide the data among N clients (5, 10 or 15) -> remove NaN/duplicates ->
undersample to balance -> StandardScaler.
Our reconstruction: clean -> stratified 80/20 split on REAL data (same test set as our `honest`
protocol, ~12.5% attack; consistent with the ~12% attack share implied by the paper's federated
rows) -> a 15% validation slice from train -> ONE StandardScaler fit on the real training part ->
IID split of the training part across N clients -> each client undersamples its own shard 50/50.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from stages.s1_preprocessing.balancing import undersample
from stages.s1_preprocessing.loading import load_clean
from stages.s1_preprocessing.partitioning import iid_parts
from stages.s1_preprocessing.protocols import SPLIT_SEED, VAL_FRAC


def make_data_2023(seed, n_clients, csv=None):
    X, y = load_clean(csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=VAL_FRAC, stratify=ytr, random_state=100 + seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    clients = [undersample(Xc, yc, seed + i) for i, (Xc, yc) in enumerate(iid_parts(Xtr, ytr, seed, n_clients))]
    central = undersample(Xtr, ytr, seed)
    rate = float(yte.mean())
    assert abs(rate - 0.125) < 0.02, rate
    return dict(clients=clients, central=central, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
                info=dict(protocol="honest", test_attack_rate=round(rate, 4), majority_acc=round(max(rate, 1 - rate), 4),
                          n_clients=n_clients, client_sizes=[len(c[1]) for c in clients]))

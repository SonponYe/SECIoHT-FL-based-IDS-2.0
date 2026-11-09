"""Does the leaked row index change the 2023 notebook's results? Replays its per-client preprocessing.
Run: python baseline_2023/audit/index_leak_test.py   (from our_work/)"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CSV = Path(__file__).resolve().parents[2] / "data" / "wustl-ehms-2020_with_attacks_categories (1).csv"
raw = pd.read_csv(CSV)
drop = ["Dir", "Flgs", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num", "Attack Category", "Dport"]
df = raw.drop(columns=drop).copy()
df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
df.insert(0, "Unnamed: 0", np.arange(len(df)))
feat_all = [c for c in df.columns if c not in ("Label",)]          # index + 35 real
real = [c for c in feat_all if c != "Unnamed: 0"]


def build(n_clients, cols, seed=0):
    """Replay the notebook per client: shuffle(13) -> 90/10 split(1337) -> per-client StandardScaler ->
    undersample train to 2 normal : 1 attack. Returns pooled train and (unbalanced) validation."""
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xva, yva = [], [], [], []
    for idx in np.array_split(np.arange(len(df)), n_clients):
        c = df.iloc[idx].sample(frac=1, random_state=13).reset_index(drop=True)
        tr, va = train_test_split(c, test_size=0.1, random_state=1337)
        sc = StandardScaler().fit(tr[cols])
        pos, neg = tr[tr.Label == 1], tr[tr.Label == 0]
        neg = neg.iloc[rng.choice(len(neg), min(len(neg), 2 * len(pos)), replace=False)]
        tb = pd.concat([pos, neg])
        Xtr.append(sc.transform(tb[cols])); ytr.append(tb.Label.values)
        Xva.append(sc.transform(va[cols])); yva.append(va.Label.values)
    return np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xva), np.concatenate(yva)


def score(name, cols, n_clients=15, seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        Xt, yt, Xv, yv = build(n_clients, cols, s)
        m = MLPClassifier((64, 64, 32), solver="adam", learning_rate_init=0.01, batch_size=1024, max_iter=150,
                          random_state=s).fit(Xt, yt)
        p = m.predict_proba(Xv)[:, 1]
        pred = (p >= .5).astype(int)
        rows.append([accuracy_score(yv, pred), precision_score(yv, pred, zero_division=0),
                     recall_score(yv, pred), f1_score(yv, pred), roc_auc_score(yv, p)])
    r = np.mean(rows, 0)
    print(f"{name:52s} acc {r[0]:.3f}  P {r[1]:.3f}  R {r[2]:.3f}  F1 {r[3]:.3f}  AUC {r[4]:.3f}   (val n={len(yv)}, attack share {yv.mean():.3f})")


print("15 clients, pooled validation (the notebook's evaluation set), MLP 64-64-32, mean of 3 seeds\n")
score("A  as in notebook: 34 real (Temp lost) + row index", [c for c in feat_all if c != "Temp"])
score("B  34 real features (Temp lost), NO row index", [c for c in real if c != "Temp"])
score("C  all 35 real features, NO row index (intended)", real)
score("D  all 35 real + row index", feat_all)
score("E  row index ONLY", ["Unnamed: 0"])

# index-only signal with a tree model, per the same split
Xt, yt, Xv, yv = build(15, ["Unnamed: 0"], 0)
g = HistGradientBoostingClassifier(max_depth=3, max_iter=100).fit(Xt, yt)
print(f"\nrow-index-only boosted trees, AUC on validation: {roc_auc_score(yv, g.predict_proba(Xv)[:, 1]):.3f}  (0.5 = no signal)")

"""Rebuild the authors' clean_wustl.csv from the public CSV and replay the pandas lines that reassign columns.
Run: python baseline_2023/audit/replay_column_bug.py   (from our_work/)"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CSV = Path(__file__).resolve().parents[2] / "data" / "wustl-ehms-2020_with_attacks_categories (1).csv"
raw = pd.read_csv(CSV)
print("our raw columns:", raw.shape[1], " rows:", len(raw))

# ---- 1. rebuild the authors' clean_wustl.csv from the public file: drop 8 columns + Dport
drop = ["Dir", "Flgs", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num", "Attack Category", "Dport"]
df = raw.drop(columns=drop).copy()
df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
df.insert(0, "Unnamed: 0", np.arange(len(df)))
print("rebuilt columns:", df.shape[1], "(notebook shows 37)")
print(list(df.columns))

# ---- 2. does it match what the notebook printed?
print("\nfirst 5 rows vs notebook head():")
print(df.head(5)[["Unnamed: 0", "Sport", "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "Rate", "Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST", "Label"]].to_string())
print("notebook: Sport 58059,58062,58065,58067,58069 | SrcLoad 276914,230984,218470,203376,235723 | Rate 558.295,465.694,440.464,410.032,475.247")

def split(d, n):
    return [d.iloc[i] for i in np.array_split(np.arange(len(d)), n)]

chunks15 = split(df, 15)
c1 = chunks15[0]
print("\nclient-1 label counts:", c1.Label.value_counts().to_dict(), "(notebook: 0:935, 1:153)")
c1s = c1.sample(frac=1, random_state=13).reset_index(drop=True)
print("client-1 shuffled first 'Unnamed: 0' values:", c1s["Unnamed: 0"].head(5).tolist(), "(notebook: 248,204,142,472,370)")

# ---- 3. how uneven are the contiguous clients?
for n in (5, 10, 15):
    ch = split(df, n)
    print(f"\n{n} clients: attacks per client:", [int(c.Label.sum()) for c in ch], " sizes:", [len(c) for c in ch][:3], "...")

# ---- 4. replay the column-reassignment line
c = c1s.copy()
train_df, val_df = train_test_split(c, test_size=0.1, random_state=1337)
train_df, val_df = train_df.copy(), val_df.copy()
orig = [x for x in train_df.columns if x != "Label"]                       # order fed to the scaler
tf = StandardScaler().fit_transform(train_df.drop(["Label"], axis=1))
ordered_names = list(train_df.columns.difference(["Label"]))                  # what the notebook assigns INTO (sorted)
print("\noriginal order  :", orig[:6], "...")
print("sorted (assigned):", ordered_names[:6], "...")
train_df[train_df.columns.difference(["Label"])] = tf
slot = ordered_names.index("Unnamed: 0")
print(f"\n'Unnamed: 0' sits at sorted position {slot}; that slot receives the standardised values of -> '{orig[slot]}'")
print(f"standardised row-INDEX (orig col 0) is written into the column named -> '{ordered_names[0]}'")
# prove it with the data
same = np.allclose(train_df["Unnamed: 0"].values, tf[:, slot])
print("check: column 'Unnamed: 0' now holds standardised", orig[slot], "->", same)
print("column named", ordered_names[0], "now equals standardised row index ->", np.allclose(train_df[ordered_names[0]].values, tf[:, 0]))
print("make_tf_dataset then DROPS the column named 'Unnamed: 0'  => the real feature", repr(orig[slot]), "is LOST,")
print("and the standardised row index stays in as an input feature.")
mapping = {name: orig[ordered_names.index(name)] for name in ordered_names if name != "Unnamed: 0"}
print("\nmodel input (35 columns, in dataframe order) actually contains:")
cols_in_df = [x for x in train_df.columns if x not in ("Label", "Unnamed: 0")]
print([mapping[x] for x in cols_in_df])

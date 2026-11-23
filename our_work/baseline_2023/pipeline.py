"""baseline_2023 / data pipeline, steps 1-6 (data -> cleaning -> split -> partition -> scaling -> balancing).

Rebuilds the data side of the 2023 DNN-FL paper (Mosaiyebzadeh et al., INFOCOM WKSHPS 2023) and of the authors'
notebook (`references/authors_2023_DNN_FL_based_IDS.ipynb`), in three modes:

  as_published        the authors' notebook exactly, INCLUDING the column bug (audit finding F4)
  as_published_fixed  identical, but the column bug is fixed
  honest              our corrected pipeline (the main benchmark)

Every audit finding is either reproduced on purpose (as_published*) or corrected (honest); nothing is
silently fixed.  Findings: F1 Dport removed | F2 three text `Sport` values | F3 `Unnamed: 0` index column |
F4 column-assignment bug | F5 no test set | F6 contiguous client chunks | F7 per-client scaler fitted on the
unbalanced split (reproduced in as_published*; honest uses ONE global scaler, see below) | F8 2 normals per attack |
F9 unseeded sampling | F10 no NaN/duplicate step | F11 array_split returns arrays on a DataFrame (we split row
positions instead).

Scaling: as_published / as_published_fixed use one StandardScaler PER CLIENT (fitted on that client's unbalanced
training rows), exactly like the authors.  honest uses ONE global scaler (aggregate mean/variance shared across
clients), fitted on the union of all clients' unbalanced training rows and applied to every client's training rows
AND to the global validation and test sets, so training and evaluation use the same scale.

Public entry point:  build(mode, n_clients, seed=0, *, keep_dport=None, balance_ratio=None, partition=None, csv=None)
  keep_dport / balance_ratio / partition default (None) to True / 1 / "iid" in honest.  as_published* force
  False / 2 / "contiguous" (what the authors did); explicitly passing a different value prints a warning.

NOT built here (steps 7+), recorded so they are not forgotten:
  * "epochs 100" in the paper = 100 federated ROUNDS with 1 local epoch; batch 1,024; client Adam 0.01,
    server Adam 0.05; model 64-64-32-1, sigmoid, BCE.
  * as_published: evaluate at threshold 0.5 after round 100 on the pooled validation rows (Table III).
  * honest: validation-tuned threshold, ONE score on the held-out test set, per-attack-type recall
    (`cat_*` is carried for that), majority-class baseline in every table.
`Attack Category` is never an input feature; it is only carried alongside as `cat_*`.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "wustl-ehms-2020_with_attacks_categories (1).csv"
MODES = ("as_published", "as_published_fixed", "honest")

PUBLIC_COLUMNS = [
    "Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "Dport", "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "SrcGap",
    "DstGap", "SIntPkt", "DIntPkt", "SIntPktAct", "DIntPktAct", "SrcJitter", "DstJitter", "sMaxPktSz", "dMaxPktSz",
    "sMinPktSz", "dMinPktSz", "Dur", "Trans", "TotPkts", "TotBytes", "Load", "Loss", "pLoss", "pSrcLoss", "pDstLoss",
    "Rate", "SrcMac", "DstMac", "Packet_num", "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate",
    "ST", "Attack Category", "Label",
]
DROP_ALWAYS = ["Dir", "Flgs", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num", "Attack Category"]
EXPECTED_ROWS, EXPECTED_LABELS = 16318, {0: 14272, 1: 2046}
EXPECTED_CATEGORIES = {"normal": 14272, "Spoofing": 1124, "Data Alteration": 922}


@dataclass
class Cfg:
    mode: str
    n_clients: int
    seed: int
    keep_dport: bool
    balance_ratio: int
    partition: str
    csv: Path
    passed: dict            # what the caller gave (None = not given)
    overridden: list        # as_published* only: explicit values that were replaced by the authors' ones


AS_PUBLISHED_FORCED = dict(keep_dport=False, balance_ratio=2, partition="contiguous")   # what the authors did
HONEST_DEFAULTS = dict(keep_dport=True, balance_ratio=1, partition="iid")


def _make_cfg(mode, n_clients, seed, keep_dport, balance_ratio, partition, csv):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    passed = dict(keep_dport=keep_dport, balance_ratio=balance_ratio, partition=partition)
    overridden = []
    if mode != "honest":                       # what the authors did; not configurable
        for arg, forced in AS_PUBLISHED_FORCED.items():
            if passed[arg] is not None and passed[arg] != forced:
                overridden.append({"arg": arg, "passed": passed[arg], "used": forced})
        if overridden:
            warnings.warn(f"mode {mode!r} reproduces the authors' pipeline, so these arguments were OVERRIDDEN: " +
                          ", ".join(f"{o['arg']}={o['passed']!r} -> {o['used']!r}" for o in overridden) +
                          ". Use mode='honest' to choose them freely.", UserWarning, stacklevel=3)
        keep_dport, balance_ratio, partition = AS_PUBLISHED_FORCED.values()
    else:
        keep_dport = HONEST_DEFAULTS["keep_dport"] if keep_dport is None else keep_dport
        balance_ratio = HONEST_DEFAULTS["balance_ratio"] if balance_ratio is None else balance_ratio
        partition = HONEST_DEFAULTS["partition"] if partition is None else partition
    if partition not in ("iid", "contiguous"):
        raise ValueError(f"partition must be 'iid' or 'contiguous', got {partition!r}")
    return Cfg(mode, int(n_clients), int(seed), bool(keep_dport), int(balance_ratio), partition,
               Path(csv) if csv else DEFAULT_CSV, passed, overridden)


def _new_info(cfg: Cfg) -> dict:
    return {
        "mode": cfg.mode,
        "args": dict(n_clients=cfg.n_clients, seed=cfg.seed, **cfg.passed, csv=str(cfg.csv)),
        "effective_args": dict(n_clients=cfg.n_clients, seed=cfg.seed, keep_dport=cfg.keep_dport,
                               balance_ratio=cfg.balance_ratio, partition=cfg.partition),
        "forced_by_mode": cfg.mode != "honest",
        "overridden_args": cfg.overridden,
        "rows": {},
        "assumptions": [],
    }


def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray) and p.dtype == object:
            h.update("\n".join(map(str, p)).encode())
        else:
            h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------------- step 1
def step1_load(S: dict) -> None:
    """Step 1 - load the public CSV, assert it is exactly the file the paper describes, pull out `Attack Category`."""
    cfg, info = S["cfg"], S["info"]
    raw = pd.read_csv(cfg.csv, low_memory=False)
    assert raw.shape == (EXPECTED_ROWS, 45), f"expected (16318, 45), got {raw.shape}"
    assert list(raw.columns) == PUBLIC_COLUMNS, "column order differs from the public WUSTL file"
    labels = {int(k): int(v) for k, v in raw["Label"].value_counts().items()}
    assert labels == EXPECTED_LABELS, labels
    cats = {str(k): int(v) for k, v in raw["Attack Category"].value_counts().items()}
    assert cats == EXPECTED_CATEGORIES, cats

    sport_num = pd.to_numeric(raw["Sport"], errors="coerce")
    bad = np.where(sport_num.isna().to_numpy())[0]
    info["rows"]["raw"] = len(raw)
    info["raw"] = {"shape": tuple(raw.shape), "labels": labels, "categories": cats,
                   "sport_text_rows": [{"row": int(i), "value": str(raw["Sport"].iloc[i]),
                                        "label": int(raw["Label"].iloc[i]),
                                        "category": str(raw["Attack Category"].iloc[i])} for i in bad]}
    S["raw"] = raw
    S["cat_all"] = raw["Attack Category"].to_numpy(dtype=object)     # aligned with the public row numbers


# --------------------------------------------------------------------------------------------- step 2
def step2_clean(S: dict) -> None:
    """Step 2 - cleaning.

    Drops the identifier / label-leak columns (+ `Dport` in the authors' file, F1), converts `Sport` to numeric (F2),
    adds the `Unnamed: 0` index column the authors' CSV carried (F3, as_published* only) and, for honest, runs the
    dropna / drop_duplicates step the paper describes but the notebook lacks (F10)."""
    cfg, info, raw = S["cfg"], S["info"], S["raw"]
    honest = cfg.mode == "honest"
    clean = {}
    if honest:                                               # F10: paper says NaN + duplicates removed
        n_nan, n_dup = int(raw.isna().any(axis=1).sum()), int(raw.duplicated().sum())
        assert n_nan == 0, f"dropna would remove {n_nan} rows"
        assert n_dup == 0, f"drop_duplicates would remove {n_dup} rows"
        clean = {"dropna_removed": n_nan, "drop_duplicates_removed": n_dup}
    info["clean"] = clean or None

    dropped = DROP_ALWAYS + ([] if cfg.keep_dport else ["Dport"])
    feats = raw.drop(columns=DROP_ALWAYS + ["Label"]).copy()      # Dport is dropped below, AFTER the constants are listed
    sport = pd.to_numeric(feats["Sport"], errors="coerce")
    bad = sport.isna().to_numpy()
    n_bad = int(bad.sum())
    if honest:                                               # F2: coerce, then DROP the 3 text rows
        keep = ~bad
        feats = feats.loc[keep]
        feats["Sport"] = sport.loc[keep]
        y = raw["Label"].loc[keep]
        info["sport"] = {"rows_converted": n_bad, "handling": "coerced to numeric, rows dropped", "rows_dropped": n_bad}
    else:                                                    # F2: authors kept all 16,318 rows -> fill with 0
        feats["Sport"] = sport.fillna(0)
        y = raw["Label"]
        info["sport"] = {"rows_converted": n_bad, "handling": "coerced to numeric, 3 text values filled with 0, rows kept",
                         "rows_dropped": 0}
        info["assumptions"].append(
            "Sport text values (dircproxy, fido, tfido; rows 7923, 8412, 10633) are filled with 0 in as_published*: the "
            "authors' file keeps all 16,318 rows but the conversion step is not in their notebook, so 0 is our assumption.")
    feats = feats.astype("float64")                          # the authors' file stores every feature as float
    # Columns with a single value in every row carry NO information.  Reported only; NEVER dropped here (Dport is the
    # only one the authors removed, F1).  Constant = zero range (np.std of identical floats can be ~1e-16).
    const_all = [c for c in feats.columns if np.ptp(feats[c].to_numpy()) == 0]
    feats = feats.drop(columns=[] if cfg.keep_dport else ["Dport"])
    info["constant_columns"] = {
        "columns": [c for c in const_all if c in feats.columns],
        "removed_before_modelling": [c for c in const_all if c not in feats.columns],
        "values": {c: float(raw[c].iloc[0]) for c in const_all},
        "carry_information": False,
        "action": "reported only, NOT dropped (except Dport in as_published*, as the authors did)"}
    df = pd.concat([feats, y.astype("int64").rename("Label")], axis=1)   # keeps the public row numbers as its index
    if not honest:                                           # F3: the CSV round-trip created this column
        df.insert(0, "Unnamed: 0", np.arange(len(df)))

    assert not df.isna().any().any(), "NaN left after cleaning"
    assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes), "non-numeric column left after cleaning"
    if honest:
        assert not {"Unnamed: 0", "index"} & set(df.columns), "an index column leaked into the honest features"

    S["df"] = df
    S["cat"] = S["cat_all"][df.index.to_numpy()]
    info["rows"]["after_clean"] = len(df)
    info["features_dropped"] = dropped
    info["features_kept"] = [c for c in df.columns if c not in ("Label", "Unnamed: 0")]
    info["columns_after_cleaning"] = list(df.columns)
    info["n_features"] = len(info["features_kept"])


# --------------------------------------------------------------------------------------------- step 3
def _set_summary(y: np.ndarray, cat: np.ndarray) -> dict:
    n, a = len(y), int(y.sum())
    maj = int(a * 2 > n)
    return {"rows": n, "attacks": a, "attack_share": a / n if n else float("nan"),
            "categories": {c: int((cat == c).sum()) for c in ("normal", "Spoofing", "Data Alteration")},
            "majority_class": maj, "majority_acc": max(a, n - a) / n if n else float("nan")}


def step3_split(S: dict) -> None:
    """Step 3 - splitting.

    as_published*: nothing here; the authors have no test set, each client is split 90/10 in step 4 (F5).
    honest: a stratified 80/20 TEST set is held out first (seed 42), then 15% of the rest is the validation set.
    Splitting is done on row positions so `cat` stays aligned through every split."""
    cfg, info, df, cat = S["cfg"], S["info"], S["df"], S["cat"]
    if cfg.mode != "honest":
        S["split"] = None
        info["split"] = {"note": "no global split: per-client 90/10 split in step 4 (the authors have no test set, F5)"}
        return
    y = df["Label"].to_numpy()
    pos = np.arange(len(df))
    pos_trval, pos_test = train_test_split(pos, test_size=0.2, stratify=y, random_state=42)
    pos_train, pos_val = train_test_split(pos_trval, test_size=0.15, stratify=y[pos_trval], random_state=42)
    S["split"] = {"train": np.sort(pos_train), "val": np.sort(pos_val), "test": np.sort(pos_test)}
    info["split"] = {k: _set_summary(y[v], cat[v]) for k, v in S["split"].items()}
    info["split"]["test_positions_hash"] = _hash(np.sort(df.index.to_numpy()[pos_test]))
    info["rows"].update(train_part=len(pos_train), val=len(pos_val), test=len(pos_test))


# --------------------------------------------------------------------------------------------- step 4
def _counts(y: np.ndarray) -> dict:
    n, a = len(y), int(np.sum(y))
    return {"rows": n, "attacks": a, "normals": n - a, "attack_share": a / n if n else float("nan")}


def step4_partition(S: dict) -> None:
    """Step 4 - federated partition.

    as_published*: contiguous chunks of the file (`np.array_split` on row POSITIONS, F6/F11), each chunk shuffled with
    random_state=13, then a plain (not stratified) 90/10 split inside every client with random_state=1337 (F5).
    The pooled 10% slices are the authors' evaluation set.
    honest: only the TRAINING part is shared out (`iid` = seeded shuffle, or `contiguous`); validation stays global
    and the test set is never touched by any client."""
    cfg, info, df = S["cfg"], S["info"], S["df"]
    k, n = cfg.n_clients, len(df)
    clients, cinfo = [], []
    if cfg.mode != "honest":
        for i, idx in enumerate(np.array_split(np.arange(n), k)):
            chunk = df.iloc[idx]
            shuffled = chunk.sample(frac=1, random_state=13).reset_index(drop=True)
            tr, va = train_test_split(shuffled, test_size=0.1, random_state=1337)
            tr, va = tr.copy(), va.copy()
            # `Unnamed: 0` is the public row number; read it NOW, before the column bug can scramble it (step 5)
            clients.append({"train_df": tr, "val_df": va,
                            "pos_train": tr["Unnamed: 0"].to_numpy().astype(int),
                            "pos_val": va["Unnamed: 0"].to_numpy().astype(int)})
            cinfo.append({"client": i + 1, "chunk": _counts(chunk["Label"].to_numpy()),
                          "shuffled_head_row_numbers": shuffled["Unnamed: 0"].head(5).astype(int).tolist(),
                          "train": _counts(tr["Label"].to_numpy()), "val": _counts(va["Label"].to_numpy())})
        pooled = np.concatenate([c["val_df"]["Label"].to_numpy() for c in clients])
        pooled_pos = np.concatenate([c["pos_val"] for c in clients])
        info["eval_sets"] = {"val": _set_summary(pooled, S["cat_all"][pooled_pos])}
        info["eval_sets"]["val"]["note"] = "pooled per-client 10% slices: the authors' reported evaluation set (no test set)"
        info["assumptions"].append("90/10 split uses the notebook's plain train_test_split (not stratified), random_state=1337.")
        info["assumptions"].append("Authors' undersampling used the unseeded random.sample, so their run cannot be repeated exactly; "
                                   "we use a seeded RandomState(seed + client_index) (F9).")
    else:
        tr_pos = S["split"]["train"]
        if cfg.partition == "iid":
            chunks = np.array_split(np.random.RandomState(cfg.seed).permutation(tr_pos), k)
        else:
            chunks = np.array_split(np.sort(tr_pos), k)
        public_rows = df.index.to_numpy()
        for i, ch in enumerate(chunks):
            tr = df.iloc[ch]
            clients.append({"train_df": tr, "val_df": None, "pos_train": public_rows[ch], "pos_val": None})
            cinfo.append({"client": i + 1, "train": _counts(tr["Label"].to_numpy())})
        assert all(c["train"]["attacks"] >= 1 for c in cinfo), "a client has no attack rows; use fewer clients"
        S["val_df"], S["test_df"] = df.iloc[S["split"]["val"]], df.iloc[S["split"]["test"]]
        info["eval_sets"] = {"val": info["split"]["val"], "test": info["split"]["test"]}
    S["clients"] = clients
    info["clients"] = cinfo
    info["rows"]["clients_train_total"] = sum(c["train"]["rows"] for c in cinfo)


# --------------------------------------------------------------------------------------------- step 5
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; nan if either input is constant.  Constant = zero RANGE, not zero std: np.std of identical
    floats can be ~1e-16 because of rounding."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _same_by_value(out: np.ndarray, names: list, raw: pd.DataFrame) -> dict:
    """Each output column must be a positive linear image of the SAME-named raw column (correlation > 0.999).
    A column that is constant in the raw rows must come out constant too.  A NaN correlation (output constant while the
    raw column varies) counts as a FAILURE (0.0), never as a skip.  Returns the worst case."""
    worst, worst_name, n_const = 1.0, None, 0
    for j, name in enumerate(names):
        r, o = raw[name].to_numpy(), out[:, j]
        if np.ptp(r) == 0:
            n_const += 1
            c = 1.0 if np.ptp(o) < 1e-9 else 0.0
        else:
            c = _corr(o, r)
            if np.isnan(c):
                c = 0.0
        if c < worst:
            worst, worst_name = c, name
    return {"min_corr": worst, "worst_column": worst_name, "constant_columns": n_const}


def step5_scale(S: dict) -> None:
    """Step 5 - scaling (F4, F7).

    as_published / as_published_fixed: one StandardScaler PER CLIENT, fitted on that client's UNBALANCED training rows
    (before step 6), exactly like the authors (F7).
    honest: ONE global StandardScaler (aggregate mean/variance shared across clients), fitted on the union of all
    clients' unbalanced training rows and applied to every client's training rows AND to the global validation and test
    sets, so training and evaluation use the same scale.  The same scaler OBJECT is used for all of them (recorded in
    S["scaler_used"]) and there are no per-client scalers in honest mode.
    as_published reproduces the authors' line exactly, with its column bug:
        train_df[train_df.columns.difference(['Label'])] = scaler.fit_transform(train_df.drop(['Label'], axis=1))
    `.difference()` returns the names SORTED, the scaler returns the values in file order, so each scaled column is
    written under another column's name: scaled `Temp` lands in `Unnamed: 0` (which is then dropped -> temperature is
    lost) and the scaled row index lands in `DIA` (-> the row's position in the file becomes an input).
    as_published_fixed / honest assign by one explicit column list.  Everything is then verified by VALUE."""
    cfg, info, df = S["cfg"], S["info"], S["df"]
    mode = cfg.mode
    ver = {"clients": [], "note": "correlation of each scaled column with raw columns, by value not by name"}
    if mode != "honest":
        for i, c in enumerate(S["clients"]):
            tr, va = c["train_df"], c["val_df"]
            raw_tr, raw_va = tr.drop(columns=["Label"]).copy(), va.drop(columns=["Label"]).copy()
            scaler = StandardScaler()
            if mode == "as_published":                                   # F4: reproduced on purpose
                tr_f = scaler.fit_transform(tr.drop(["Label"], axis=1))
                va_f = scaler.transform(va.drop(["Label"], axis=1))
                tr[tr.columns.difference(["Label"])] = tr_f
                va[va.columns.difference(["Label"])] = va_f
            else:                                                        # fixed: assign by the same explicit list
                cols = [col for col in tr.columns if col != "Label"]
                tr[cols] = scaler.fit_transform(tr[cols])
                va[cols] = scaler.transform(va[cols])
            Xtr, Xva = tr.drop(columns=["Unnamed: 0", "Label"]), va.drop(columns=["Unnamed: 0", "Label"])
            names = list(Xtr.columns)
            c.update(Xs_train=Xtr.to_numpy(), ys_train=tr["Label"].to_numpy(), Xs_val=Xva.to_numpy(),
                     ys_val=va["Label"].to_numpy(), names=names)
            entry = {"client": i + 1}
            if mode == "as_published":
                j = names.index("DIA")
                entry["DIA_vs_row_index"] = {"train": _corr(c["Xs_train"][:, j], raw_tr["Unnamed: 0"].to_numpy()),
                                             "val": _corr(c["Xs_val"][:, j], raw_va["Unnamed: 0"].to_numpy())}
                temp_tr = raw_tr["Temp"].to_numpy()
                cs = [_corr(c["Xs_train"][:, k], temp_tr) for k in range(len(names))]
                entry["max_corr_with_raw_Temp"] = float(np.nanmax(cs)) if not np.all(np.isnan(cs)) else float("nan")
                entry["Temp_constant_in_client"] = bool(np.std(temp_tr) == 0)
            else:
                entry["train"] = _same_by_value(c["Xs_train"], names, raw_tr)
                entry["val"] = _same_by_value(c["Xs_val"], names, raw_va)
            ver["clients"].append(entry)
        S["feature_names"] = S["clients"][0]["names"]
        S["val"] = {"X": np.vstack([c["Xs_val"] for c in S["clients"]]),
                    "y": np.concatenate([c["ys_val"] for c in S["clients"]]),
                    "pos": np.concatenate([c["pos_val"] for c in S["clients"]])}
        S["test"] = None
        info["n_inputs"] = len(S["feature_names"])
        if mode == "as_published":
            tr0 = S["clients"][0]["train_df"]
            srt, orig = list(tr0.columns.difference(["Label"])), [x for x in tr0.columns if x != "Label"]
            cmap = dict(zip(srt, orig))
            info["column_bug"] = {
                "column_content_map": cmap, "lost_feature": cmap["Unnamed: 0"], "row_index_in_column": [k for k, v in cmap.items() if v == "Unnamed: 0"][0],
                "meaning": "content of the column NAMED k is the scaled version of the column cmap[k]"}
        info["assumptions"].append("Evaluation rows of each client are scaled with that client's own training scaler (the authors' code).")
    else:
        cols = info["features_kept"]
        # ONE global scaler fitted on the union of all clients' unbalanced training rows, used for EVERYTHING below.
        # (sorted by public row number: summing the same rows in a different order, e.g. another seed or client count,
        #  would change the mean/variance in the last bits and make the scaled test set differ between runs)
        scaler = StandardScaler().fit(pd.concat([c["train_df"][cols] for c in S["clients"]]).sort_index())
        S["scaler"], S["scaler_used"] = scaler, []

        def transform(tag: str, frame: pd.DataFrame) -> np.ndarray:
            S["scaler_used"].append((tag, scaler))             # lets the checks prove it is the same object every time
            return scaler.transform(frame[cols])

        for i, c in enumerate(S["clients"]):
            tr = c["train_df"]
            c.update(Xs_train=transform(f"client{i + 1}_train", tr), ys_train=tr["Label"].to_numpy(), Xs_val=None,
                     ys_val=None, names=list(cols))
            ver["clients"].append({"client": i + 1, "train": _same_by_value(c["Xs_train"], cols, tr)})
        vdf, tdf = S["val_df"], S["test_df"]
        S["val"] = {"X": transform("val", vdf), "y": vdf["Label"].to_numpy(), "pos": vdf.index.to_numpy()}
        S["test"] = {"X": transform("test", tdf), "y": tdf["Label"].to_numpy(), "pos": tdf.index.to_numpy()}
        ver["val"], ver["test"] = _same_by_value(S["val"]["X"], cols, vdf), _same_by_value(S["test"]["X"], cols, tdf)
        S["feature_names"] = list(cols)
        info["n_inputs"] = len(cols)
        info["split"]["test_hash_scaled"] = _hash(S["test"]["X"], S["test"]["y"], S["cat_all"][S["test"]["pos"]])
        info["assumptions"].append(
            "honest uses one global scaler (aggregate mean/variance shared across clients), so training and evaluation use "
            "the same scale. It is fitted on the union of all clients' unbalanced training rows; there are no per-client "
            "scalers in honest mode.")
    info["scaling"] = {
        "scaler": ("ONE global StandardScaler fitted on the union of all clients' unbalanced training rows; applied to every "
                   "client's training rows and to the global validation and test sets" if mode == "honest" else
                   "StandardScaler, one per client, fitted on that client's unbalanced training rows (as the authors)"),
        "n_scalers": 1 if mode == "honest" else cfg.n_clients,
        "column_bug_reproduced": mode == "as_published", "inputs": info["n_inputs"], "verification": ver}


# --------------------------------------------------------------------------------------------- step 6
def step6_balance(S: dict) -> None:
    """Step 6 - balancing, TRAINING rows only (F8, F9).

    Keep every attack, sample `balance_ratio x n_attacks` normals WITHOUT replacement with a seeded
    RandomState(seed + client_index) (the authors used the unseeded random.sample), then shuffle (seeded).  If a client has
    fewer normals than needed it keeps all of them and the shortfall is logged.  Validation and test are never balanced."""
    cfg, info = S["cfg"], S["info"]
    before = {k: (len(v["y"]), int(v["y"].sum())) for k, v in (("val", S["val"]), ("test", S["test"])) if v is not None}
    for i, c in enumerate(S["clients"]):
        X, y, pos = c["Xs_train"], c["ys_train"], c["pos_train"]
        rng = np.random.RandomState(cfg.seed + i)
        pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
        want = cfg.balance_ratio * len(pos_idx)
        short = len(neg_idx) < want
        keep_neg = neg_idx if short else rng.choice(neg_idx, size=want, replace=False)
        keep = rng.permutation(np.concatenate([pos_idx, keep_neg]))
        c["X_train"], c["y_train"], c["pos_train_balanced"] = X[keep], y[keep], pos[keep]
        c["n_attack_train"], c["n_normal_train"] = int(len(pos_idx)), int(len(keep_neg))
        info["clients"][i]["balanced"] = {**_counts(c["y_train"]), "normals_available": int(len(neg_idx)),
                                          "normals_wanted": int(want), "short_of_normals": bool(short)}
        if short:
            info["assumptions"].append(f"client {i + 1}: only {len(neg_idx)} normals for {want} wanted; all kept.")
    after = {k: (len(v["y"]), int(v["y"].sum())) for k, v in (("val", S["val"]), ("test", S["test"])) if v is not None}
    assert before == after, f"evaluation sets changed during balancing: {before} -> {after}"
    info["rows"]["balanced_train_total"] = sum(c["y_train"].shape[0] for c in S["clients"])
    info["balancing"] = {"ratio_normals_per_attack": cfg.balance_ratio, "seeded": True,
                         "authors": "unseeded random.sample, 2 normals per attack",
                         "evaluation_sets_untouched": before == after}
    info["assumptions"].append("Balancing is seeded (RandomState(seed + client_index)); the authors' run was unseeded, so their exact "
                               "subsets of normals cannot be recovered (F9).")


# --------------------------------------------------------------------------------------------- driver
_STEPS = [step1_load, step2_clean, step3_split, step4_partition, step5_scale, step6_balance]


def _package(S: dict) -> dict:
    """The public return value of build()."""
    cat_all, info = S["cat_all"], S["info"]
    clients = []
    for c in S["clients"]:
        has_val = c["Xs_val"] is not None
        clients.append({"X_train": c["X_train"], "y_train": c["y_train"],
                        "X_val": c["Xs_val"] if has_val else None, "y_val": c["ys_val"] if has_val else None,
                        "cat_val": cat_all[c["pos_val"]] if has_val else None,
                        "n_attack_train": c["n_attack_train"], "n_normal_train": c["n_normal_train"],
                        "pos_train": c["pos_train_balanced"], "pos_val": c["pos_val"] if has_val else None})
    t = S["test"]
    return {"clients": clients,
            "X_val": S["val"]["X"], "y_val": S["val"]["y"], "cat_val": cat_all[S["val"]["pos"]], "pos_val": S["val"]["pos"],
            "X_test": t["X"] if t else None, "y_test": t["y"] if t else None,
            "cat_test": cat_all[t["pos"]] if t else None, "pos_test": t["pos"] if t else None,
            "feature_names": S["feature_names"], "info": info}


def build(mode: str, n_clients: int, seed: int = 0, *, keep_dport: bool | None = None, balance_ratio: int | None = None,
          partition: str | None = None, csv=None) -> dict:
    """Run steps 1-6 and return the pipeline output.

    mode: 'as_published' | 'as_published_fixed' | 'honest'.  For the two as_published modes keep_dport=False,
    balance_ratio=2 and partition='contiguous' are forced (what the authors did).
    Returns {"clients": [{X_train, y_train, X_val, y_val, cat_val, n_attack_train, n_normal_train, pos_train, pos_val}],
             "X_val", "y_val", "cat_val", "X_test", "y_test", "cat_test" (honest only; None otherwise),
             "feature_names", "info", plus pos_* = public CSV row numbers}.
    For honest, per-client X_val/y_val/cat_val are None: validation is global (top-level X_val ...).
    For as_published*, the top-level X_val ... is the pooled per-client 10% slices (the authors' evaluation set)."""
    return _package(run_until(mode, n_clients, seed, keep_dport=keep_dport, balance_ratio=balance_ratio,
                              partition=partition, csv=csv, upto=6))


def run_until(mode: str, n_clients: int, seed: int = 0, *, keep_dport: bool | None = None, balance_ratio: int | None = None,
              partition: str | None = None, csv=None, upto: int = 6) -> dict:
    """Run steps 1..upto and return the internal state (used by the check script and by build())."""
    cfg = _make_cfg(mode, n_clients, seed, keep_dport, balance_ratio, partition, csv)
    S = {"cfg": cfg, "info": _new_info(cfg)}
    for fn in _STEPS[:upto]:
        fn(S)
    return S

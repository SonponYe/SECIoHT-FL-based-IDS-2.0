"""Acceptance checks for baseline_2023/pipeline.py (steps 1-6).

Run from our_work/:   python -m baseline_2023.audit.check_pipeline            (all steps)
                      python -m baseline_2023.audit.check_pipeline --upto 3   (steps 1-3 only)
Prints PASS/FAIL per check; exit code 1 if any check fails.  INFO lines are report-only (never fail).
Expected numbers come from the authors' notebook output and from the brief; they are never adjusted to pass.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from baseline_2023 import pipeline as P

RESULTS: list[tuple[int, str, bool]] = []
UPTO = 6
_CACHE: dict = {}


def S(mode, n, seed=0, **kw):
    key = (mode, n, seed, tuple(sorted(kw.items())), UPTO)
    if key not in _CACHE:
        _CACHE[key] = P.run_until(mode, n, seed, upto=UPTO, **kw)
    return _CACHE[key]


def rec(step: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((step, name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] S{step} {name}" + (f": {detail}" if detail else ""))


def info_line(step: int, name: str, detail: str) -> None:
    print(f"[INFO] S{step} {name}: {detail}")


# ----------------------------------------------------------------------------------------- step 1
def checks_step1() -> None:
    st = S("as_published", 15)
    raw, inf = st["raw"], st["info"]["raw"]
    rec(1, "public file shape", raw.shape == (16318, 45), f"{raw.shape}")
    rec(1, "column order is the public order", list(raw.columns) == P.PUBLIC_COLUMNS, f"{raw.shape[1]} columns")
    rec(1, "Label counts", inf["labels"] == {0: 14272, 1: 2046}, f"{inf['labels']}")
    rec(1, "Attack Category counts",
        inf["categories"] == {"normal": 14272, "Spoofing": 1124, "Data Alteration": 922}, f"{inf['categories']}")
    rows = inf["sport_text_rows"]
    rec(1, "3 text values in Sport, rows 7923/8412/10633, all normal",
        [(r["row"], r["value"], r["label"]) for r in rows] == [(7923, "dircproxy", 0), (8412, "fido", 0), (10633, "tfido", 0)],
        f"{[(r['row'], r['value'], r['label']) for r in rows]}")
    rec(1, "Attack Category carried separately, aligned with rows",
        len(st["cat_all"]) == 16318 and st["cat_all"][0] == raw["Attack Category"].iloc[0], f"{len(st['cat_all'])} values")


# ----------------------------------------------------------------------------------------- step 2
AS_PUBLISHED_COLUMNS = (
    "Unnamed: 0, Sport, SrcBytes, DstBytes, SrcLoad, DstLoad, SrcGap, DstGap, SIntPkt, DIntPkt, SIntPktAct, DIntPktAct, "
    "SrcJitter, DstJitter, sMaxPktSz, dMaxPktSz, sMinPktSz, dMinPktSz, Dur, Trans, TotPkts, TotBytes, Load, Loss, pLoss, "
    "pSrcLoss, pDstLoss, Rate, Temp, SpO2, Pulse_Rate, SYS, DIA, Heart_rate, Resp_Rate, ST, Label").split(", ")


def checks_step2() -> None:
    a = S("as_published", 15)
    df = a["df"]
    rec(2, "as_published: 37 columns in the authors' order", list(df.columns) == AS_PUBLISHED_COLUMNS,
        f"{df.shape[1]} columns; Dport present={'Dport' in df.columns}")
    rec(2, "as_published: 16,318 rows kept", len(df) == 16318, f"{len(df)} rows")
    rec(2, "as_published: Unnamed: 0 is the 0-based row position",
        np.array_equal(df["Unnamed: 0"].to_numpy(), np.arange(16318)), "0..16317")
    rec(2, "as_published: first five Sport values", df["Sport"].head(5).tolist() == [58059, 58062, 58065, 58067, 58069],
        f"{df['Sport'].head(5).astype(int).tolist()}")
    rec(2, "as_published: first five SrcLoad values", df["SrcLoad"].head(5).tolist() == [276914, 230984, 218470, 203376, 235723],
        f"{df['SrcLoad'].head(5).astype(int).tolist()}")
    rec(2, "as_published: the 3 text Sport rows are filled with 0",
        df.loc[[7923, 8412, 10633], "Sport"].tolist() == [0, 0, 0], f"{df.loc[[7923, 8412, 10633], 'Sport'].tolist()}")
    rec(2, "as_published: no NaN, all numeric", not df.isna().any().any(), "")

    h = S("honest", 15)["df"]
    hi = S("honest", 15)["info"]
    rec(2, "honest: 16,315 rows (3 text-Sport rows dropped)", len(h) == 16315, f"{len(h)} rows")
    rec(2, "honest: the dropped rows are 7923, 8412, 10633",
        sorted(set(range(16318)) - set(h.index)) == [7923, 8412, 10633], f"{sorted(set(range(16318)) - set(h.index))}")
    rec(2, "honest: 36 features + Label, Dport kept", h.shape[1] == 37 and "Dport" in h.columns,
        f"{hi['n_features']} features")
    rec(2, "honest: no Unnamed: 0 / index column", not ({"Unnamed: 0", "index"} & set(h.columns)), "")
    rec(2, "honest: dropna and drop_duplicates remove 0 rows", hi["clean"] == {"dropna_removed": 0, "drop_duplicates_removed": 0},
        f"{hi['clean']}")
    rec(2, "honest: no NaN, all numeric", not h.isna().any().any() and all(pd.api.types.is_numeric_dtype(t) for t in h.dtypes), "")
    h35 = S("honest", 15, keep_dport=False)["df"]
    rec(2, "honest keep_dport=False: 35 features", h35.shape[1] == 36 and "Dport" not in h35.columns, f"{h35.shape[1] - 1} features")
    f = S("as_published_fixed", 15)["df"]
    rec(2, "as_published_fixed: same cleaned frame as as_published", f.equals(df), "")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fst = S("as_published", 15, keep_dport=True, balance_ratio=1, partition="iid")
    forced = fst["info"]["effective_args"]
    msgs = [str(x.message) for x in w if "OVERRIDDEN" in str(x.message)]
    rec(2, "as_published forces keep_dport=False, ratio 2, contiguous",
        (forced["keep_dport"], forced["balance_ratio"], forced["partition"]) == (False, 2, "contiguous"), f"{forced}")
    rec(2, "overriding the three arguments prints ONE warning naming each overridden value",
        len(msgs) == 1 and all(t in msgs[0] for t in ("keep_dport=True -> False", "balance_ratio=1 -> 2", "partition='iid' -> 'contiguous'")),
        (msgs[0][:140] + "...") if msgs else "NO WARNING")
    rec(2, "overrides are also recorded in info['overridden_args']",
        [o["arg"] for o in fst["info"]["overridden_args"]] == ["keep_dport", "balance_ratio", "partition"], "")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P.run_until("as_published", 7, upto=1)                                                           # nothing passed
        P.run_until("as_published", 7, keep_dport=False, balance_ratio=2, partition="contiguous", upto=1)  # same as the authors
        P.run_until("honest", 7, upto=1)
    rec(2, "no warning when nothing is overridden (defaults, or values equal to the authors')",
        not [x for x in w if "OVERRIDDEN" in str(x.message)], "")
    cc, ca = S("honest", 15)["info"]["constant_columns"], a["info"]["constant_columns"]
    six = {"Dport", "SrcGap", "DstGap", "DIntPktAct", "Trans", "dMinPktSz"}
    rec(2, "honest: the 6 constant columns are reported, flagged as carrying no information, and NOT dropped",
        set(cc["columns"]) == six and cc["carry_information"] is False and six <= set(h.columns) and not cc["removed_before_modelling"],
        f"{cc['columns']} values {cc['values']}")
    rec(2, "as_published: same list; Dport shown as removed by the authors, the other 5 kept",
        set(ca["columns"]) == six - {"Dport"} and ca["removed_before_modelling"] == ["Dport"],
        f"kept {ca['columns']}, removed {ca['removed_before_modelling']}")


# ----------------------------------------------------------------------------------------- step 3
def checks_step3() -> None:
    h = S("honest", 15)
    sp = h["info"]["split"]
    rec(3, "honest: test set 3,263 rows / 409 attacks", (sp["test"]["rows"], sp["test"]["attacks"]) == (3263, 409),
        f"{sp['test']['rows']} rows / {sp['test']['attacks']} attacks")
    rec(3, "honest: test attacks = 199 Spoofing + 210 Data Alteration",
        (sp["test"]["categories"]["Spoofing"], sp["test"]["categories"]["Data Alteration"]) == (199, 210),
        f"{sp['test']['categories']}")
    rec(3, "honest: training part 11,094 rows / 1,391 attacks", (sp["train"]["rows"], sp["train"]["attacks"]) == (11094, 1391),
        f"{sp['train']['rows']} rows / {sp['train']['attacks']} attacks")
    rec(3, "honest: validation 1,958 rows / 246 attacks", (sp["val"]["rows"], sp["val"]["attacks"]) == (1958, 246),
        f"{sp['val']['rows']} rows / {sp['val']['attacks']} attacks")
    s = h["split"]
    rec(3, "honest: train / val / test are disjoint and cover all 16,315 rows",
        len(set(s["train"]) | set(s["val"]) | set(s["test"])) == 16315 and
        len(s["train"]) + len(s["val"]) + len(s["test"]) == 16315, "")
    other = S("honest", 5, seed=1)["info"]["split"]["test_positions_hash"]
    rec(3, "honest: test rows identical across seeds and client counts (hash)",
        other == sp["test_positions_hash"] == S("honest", 10, seed=2)["info"]["split"]["test_positions_hash"],
        f"{sp['test_positions_hash']}")
    rec(3, "as_published: no global split (per-client 90/10 happens in step 4)", S("as_published", 15)["split"] is None, "")
    info_line(3, "honest majority-class accuracy", f"test {sp['test']['majority_acc']:.4f}, val {sp['val']['majority_acc']:.4f}")


# ----------------------------------------------------------------------------------------- step 4
ATTACKS_PER_CLIENT_15 = [153, 139, 119, 136, 174, 145, 174, 138, 118, 124, 128, 93, 156, 137, 112]


def checks_step4() -> None:
    a = S("as_published", 15)
    ci = a["info"]["clients"]
    rec(4, "as_published: 15 contiguous clients, 16,318 rows in total",
        len(ci) == 15 and sum(c["chunk"]["rows"] for c in ci) == 16318, f"{len(ci)} clients")
    rec(4, "client 1 before splitting: 935 normal / 153 attack",
        (ci[0]["chunk"]["normals"], ci[0]["chunk"]["attacks"]) == (935, 153), f"{ci[0]['chunk']['normals']} / {ci[0]['chunk']['attacks']}")
    rec(4, "client 1 first five Unnamed: 0 after the shuffle = 248, 204, 142, 472, 370",
        ci[0]["shuffled_head_row_numbers"] == [248, 204, 142, 472, 370], f"{ci[0]['shuffled_head_row_numbers']}")
    got = [c["chunk"]["attacks"] for c in ci]
    rec(4, "attacks per client (15 clients) match the notebook", got == ATTACKS_PER_CLIENT_15, f"{got}")
    rec(4, "attacks per client sum to 2,046", sum(got) == 2046, f"{sum(got)}")
    v = a["info"]["eval_sets"]["val"]
    rec(4, "pooled validation: 1,635 rows", v["rows"] == 1635, f"{v['rows']} rows")
    rec(4, "pooled validation: 215 attacks (13.1%)", v["attacks"] == 215, f"{v['attacks']} attacks = {v['attack_share']:.4f}")
    rec(4, "per-client train+val = chunk rows",
        all(c["train"]["rows"] + c["val"]["rows"] == c["chunk"]["rows"] for c in ci), "")
    pv = np.concatenate([c["pos_val"] for c in a["clients"]]); pt = np.concatenate([c["pos_train"] for c in a["clients"]])
    rec(4, "train and validation row numbers are disjoint and cover every row once",
        len(set(pv) & set(pt)) == 0 and len(set(pv) | set(pt)) == 16318, f"{len(pt)} + {len(pv)}")
    f = S("as_published_fixed", 15)
    rec(4, "as_published_fixed: identical partition (same validation rows)",
        all(np.array_equal(x["pos_val"], y["pos_val"]) for x, y in zip(a["clients"], f["clients"])), "")
    for k, exp_rows, exp_att in ((10, 1640, 197), (5, 1635, 184)):
        e = S("as_published", k)["info"]["eval_sets"]["val"]
        info_line(4, f"REPORT ONLY as_published {k} clients pooled validation",
                  f"ours {e['rows']} rows / {e['attacks']} attacks   expected {exp_rows} / {exp_att}  "
                  f"(authors' {k}-client files were made outside the notebook)")

    h = S("honest", 15)
    hc = h["info"]["clients"]
    rec(4, "honest: 15 clients share exactly the training part (11,094 rows)",
        h["info"]["rows"]["clients_train_total"] == 11094, f"{h['info']['rows']['clients_train_total']}")
    rec(4, "honest: every client has at least one attack", all(c["train"]["attacks"] >= 1 for c in hc),
        f"min {min(c['train']['attacks'] for c in hc)}, max {max(c['train']['attacks'] for c in hc)} attacks")
    allpos = np.concatenate([c["pos_train"] for c in h["clients"]])
    rec(4, "honest: client rows are disjoint from validation and test",
        len(set(allpos) & set(h["val_df"].index)) == 0 and len(set(allpos) & set(h["test_df"].index)) == 0 and
        len(allpos) == len(set(allpos)), "")
    rec(4, "honest: validation stays global (1,958 rows), test untouched (3,263)",
        (len(h["val_df"]), len(h["test_df"])) == (1958, 3263), f"{len(h['val_df'])} / {len(h['test_df'])}")
    cont = S("honest", 15, partition="contiguous")["info"]["clients"]
    rec(4, "honest partition='contiguous' also gives >=1 attack per client", all(c["train"]["attacks"] >= 1 for c in cont),
        f"{[c['train']['attacks'] for c in cont]}")
    info_line(4, "honest iid attacks per client", f"{[c['train']['attacks'] for c in hc]}")


# ----------------------------------------------------------------------------------------- step 5
def _corr(a, b):
    """Constant means zero range (np.std of identical floats can be ~1e-16, which once fooled this check)."""
    return float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")


def _independent_by_value(out, names, raw_df):
    """Recompute, independently of the pipeline, that every output column is the scaled SAME-named raw column."""
    bad = []
    for j, nm in enumerate(names):
        r = raw_df[nm].to_numpy()
        ok = (np.ptp(out[:, j]) < 1e-9) if np.ptp(r) == 0 else (_corr(out[:, j], r) > 0.999)
        if not ok:
            bad.append(nm)
    return bad


def checks_step5() -> None:
    a = S("as_published", 15)
    df = a["df"]
    cl = a["clients"]
    names = a["feature_names"]
    rec(5, "as_published: 35 inputs per client, no Unnamed: 0 column",
        all(c["Xs_train"].shape[1] == 35 for c in cl) and "Unnamed: 0" not in names, f"{len(names)} inputs")
    dia = names.index("DIA")
    c_tr = [_corr(c["Xs_train"][:, dia], df.loc[c["pos_train"], "Unnamed: 0"].to_numpy()) for c in cl]
    c_va = [_corr(c["Xs_val"][:, dia], df.loc[c["pos_val"], "Unnamed: 0"].to_numpy()) for c in cl]
    rec(5, "as_published: column NAMED 'DIA' holds the scaled ROW INDEX (corr > 0.999, all 15 clients, train and val)",
        min(c_tr + c_va) > 0.999, f"min corr {min(c_tr + c_va):.6f}")
    cand = []
    for ci, c in enumerate(cl):
        t = df.loc[c["pos_train"], "Temp"].to_numpy()
        cand += [(abs(_corr(c["Xs_train"][:, k], t)), ci + 1, names[k]) for k in range(35)]
    cand = [x for x in cand if not np.isnan(x[0])]
    top = max(cand)
    holds = a["info"]["column_bug"]["column_content_map"][top[2]]
    rec(5, "as_published: temperature is gone (no output column correlates > 0.999 with raw Temp)",
        top[0] < 0.999, f"max |corr| with raw Temp = {top[0]:.4f} (client {top[1]}, column NAMED '{top[2]}', which holds the scaled '{holds}')")
    bug = a["info"]["column_bug"]
    rec(5, "as_published: bug map says Temp is lost and the row index sits in DIA",
        (bug["lost_feature"], bug["row_index_in_column"]) == ("Temp", "DIA"), f"lost={bug['lost_feature']}, row index in column {bug['row_index_in_column']}")
    mu = np.vstack([c["Xs_train"].mean(0) for c in cl]); sd = np.vstack([c["Xs_train"].std(0) for c in cl])
    rec(5, "as_published: each client's training inputs are standardised (mean 0, std 1 or constant 0)",
        np.abs(mu).max() < 1e-8 and np.all((np.abs(sd - 1) < 1e-8) | (sd < 1e-12)), f"max |mean| {np.abs(mu).max():.1e}")

    f = S("as_published_fixed", 15)
    fn = f["feature_names"]
    bad = [x for i, c in enumerate(f["clients"]) for x in _independent_by_value(c["Xs_train"], fn, f["df"].loc[c["pos_train"]]) +
           _independent_by_value(c["Xs_val"], fn, f["df"].loc[c["pos_val"]])]
    rec(5, "as_published_fixed: every column equals the scaled same-named raw column (all clients, train and val)",
        not bad, f"{len(fn)} inputs, {len(bad)} mismatches")
    rec(5, "as_published_fixed: 35 real inputs, Temp present, no Unnamed: 0", len(fn) == 35 and "Temp" in fn and "Unnamed: 0" not in fn, f"{len(fn)} inputs")
    iv = f["info"]["scaling"]["verification"]["clients"]
    rec(5, "as_published_fixed: pipeline's own verification agrees (min corr > 0.999)",
        min(min(e["train"]["min_corr"], e["val"]["min_corr"]) for e in iv) > 0.999,
        f"min corr {min(min(e['train']['min_corr'], e['val']['min_corr']) for e in iv):.6f}")

    h = S("honest", 15)
    hn = h["feature_names"]
    bad = [x for c in h["clients"] for x in _independent_by_value(c["Xs_train"], hn, h["df"].loc[c["pos_train"]])]
    bad += _independent_by_value(h["val"]["X"], hn, h["df"].loc[h["val"]["pos"]])
    bad += _independent_by_value(h["test"]["X"], hn, h["df"].loc[h["test"]["pos"]])
    rec(5, "honest: every column equals the scaled same-named raw column (15 clients, validation, test)", not bad,
        f"{len(hn)} inputs, {len(bad)} mismatches")
    rec(5, "honest: 36 inputs, Dport and Temp present, no Unnamed: 0 / index",
        len(hn) == 36 and "Dport" in hn and "Temp" in hn and not ({"Unnamed: 0", "index"} & set(hn)), f"{len(hn)} inputs")
    sc = h["info"]["split"]["test_hash_scaled"]
    others = [S("honest", 5, seed=1)["info"]["split"]["test_hash_scaled"], S("honest", 10, seed=2)["info"]["split"]["test_hash_scaled"]]
    rec(5, "honest: scaled test set identical across seeds and client counts (hash)", all(o == sc for o in others), f"{sc}")
    hv = h["info"]["scaling"]["verification"]
    rec(5, "honest: pipeline's own verification agrees (min corr > 0.999)",
        min([e["train"]["min_corr"] for e in hv["clients"]] + [hv["val"]["min_corr"], hv["test"]["min_corr"]]) > 0.999, "")
    # Temp is a real input and its scaled column is the scaled raw Temp
    jt = fn.index("Temp")
    tf = [_corr(c["Xs_train"][:, jt], f["df"].loc[c["pos_train"], "Temp"].to_numpy()) for c in f["clients"]] + \
         [_corr(c["Xs_val"][:, jt], f["df"].loc[c["pos_val"], "Temp"].to_numpy()) for c in f["clients"]]
    rec(5, "as_published_fixed: 'Temp' in feature_names and its scaled column correlates > 0.999 with raw Temp (15 clients, train and val)",
        "Temp" in fn and not np.isnan(tf).any() and min(tf) > 0.999, f"min corr {min(tf):.6f} over {len(tf)} slices")
    jh = hn.index("Temp")
    th = [_corr(c["Xs_train"][:, jh], h["df"].loc[c["pos_train"], "Temp"].to_numpy()) for c in h["clients"]] + \
         [_corr(h["val"]["X"][:, jh], h["df"].loc[h["val"]["pos"], "Temp"].to_numpy()),
          _corr(h["test"]["X"][:, jh], h["df"].loc[h["test"]["pos"], "Temp"].to_numpy())]
    rec(5, "honest: 'Temp' in feature_names and its scaled column correlates > 0.999 with raw Temp (15 clients, validation, test)",
        "Temp" in hn and not np.isnan(th).any() and min(th) > 0.999, f"min corr {min(th):.6f} over {len(th)} sets")
    rec(5, "honest: len(feature_names) == 36 with both 'Dport' and 'Temp' present",
        len(hn) == 36 and "Dport" in hn and "Temp" in hn, f"{len(hn)} names")

    # honest: ONE global scaler for everything
    used = h["scaler_used"]
    tags = [t for t, _ in used]
    n_tr = sum(t.endswith("_train") for t in tags)
    rec(5, "honest: validation and test are transformed by the SAME scaler object as every client's training rows",
        all(o is h["scaler"] for _, o in used) and "val" in tags and "test" in tags and n_tr == 15,
        f"{len(used)} transform calls ({n_tr} client train + val + test), 1 scaler object")
    ns = (h["info"]["scaling"]["n_scalers"], a["info"]["scaling"]["n_scalers"], f["info"]["scaling"]["n_scalers"])
    rec(5, "honest: no per-client scalers (n_scalers == 1); as_published* keep one per client (n_scalers == 15)",
        ns == (1, 15, 15), f"honest {ns[0]}, as_published {ns[1]}, as_published_fixed {ns[2]}")
    pos_all = np.concatenate([c["pos_train"] for c in h["clients"]])
    ref = h["df"].loc[pos_all, hn]
    mu, sd = ref.mean().to_numpy(), ref.std(ddof=0).to_numpy()
    sd = np.where(sd < 1e-12, 1.0, sd)

    def _err(X, pos):
        return float(np.abs((h["df"].loc[pos, hn].to_numpy() - mu) / sd - X).max())

    errs = [_err(c["Xs_train"], c["pos_train"]) for c in h["clients"]] + \
           [_err(h["val"]["X"], h["val"]["pos"]), _err(h["test"]["X"], h["test"]["pos"])]
    rec(5, "honest: every set equals (raw - mean) / std of the pooled unbalanced training rows (independent recomputation)",
        max(errs) < 1e-8, f"max abs difference {max(errs):.1e} over 15 client train sets + val + test")
    rec(5, "honest: info states the single global scaler",
        any(x.startswith("honest uses one global scaler (aggregate mean/variance shared across clients), so training and "
                         "evaluation use the same scale") for x in h["info"]["assumptions"]), "")

    pooled = a["val"]
    rec(5, "as_published: pooled validation still 1,635 rows / 215 attacks after scaling",
        (len(pooled["y"]), int(pooled["y"].sum())) == (1635, 215), f"{len(pooled['y'])} / {int(pooled['y'].sum())}")


# ----------------------------------------------------------------------------------------- step 6
def checks_step6() -> None:
    a = P.build("as_published", 15)
    cl = a["clients"]
    tot = sum(len(c["y_train"]) for c in cl)
    rec(6, "as_published 15 clients: balanced training rows sum to 5,493", tot == 5493, f"{tot}")
    rec(6, "as_published: 2 normals per attack in every client (33.3% attacks)",
        all(c["n_normal_train"] == 2 * c["n_attack_train"] and int(c["y_train"].sum()) == c["n_attack_train"] and
            len(c["y_train"]) == 3 * c["n_attack_train"] for c in cl), f"attacks {sum(c['n_attack_train'] for c in cl)}")
    rec(6, "as_published: training attacks = 2,046 - 215 validation = 1,831", sum(c["n_attack_train"] for c in cl) == 1831,
        f"{sum(c['n_attack_train'] for c in cl)}")
    rec(6, "as_published: pooled validation unbalanced and unchanged (1,635 rows / 215 attacks)",
        (len(a["y_val"]), int(a["y_val"].sum())) == (1635, 215), f"{len(a['y_val'])} / {int(a['y_val'].sum())}  share {a['y_val'].mean():.4f}")
    rec(6, "as_published: 35 inputs, X_test is None, no Attack Category input",
        a["clients"][0]["X_train"].shape[1] == 35 and a["X_test"] is None and
        "Attack Category" not in a["feature_names"], f"{len(a['feature_names'])} inputs")
    f = P.build("as_published_fixed", 15)
    rec(6, "as_published_fixed: same sizes as as_published (5,493 / 1,635 / 215)",
        sum(len(c["y_train"]) for c in f["clients"]) == 5493 and (len(f["y_val"]), int(f["y_val"].sum())) == (1635, 215), "")

    h = P.build("honest", 15)
    hc = h["clients"]
    rec(6, "honest ratio 1: every client is 50% attacks", all(c["n_normal_train"] == c["n_attack_train"] for c in hc),
        f"balanced rows {sum(len(c['y_train']) for c in hc)} (= 2 x 1,391)")
    rec(6, "honest: 1,391 training attacks kept, 2,782 balanced rows",
        sum(c["n_attack_train"] for c in hc) == 1391 and sum(len(c["y_train"]) for c in hc) == 2782, "")
    rec(6, "honest: validation 1,958 / 246 and test 3,263 / 409 are NOT balanced",
        (len(h["y_val"]), int(h["y_val"].sum()), len(h["y_test"]), int(h["y_test"].sum())) == (1958, 246, 3263, 409),
        f"val share {h['y_val'].mean():.4f}, test share {h['y_test'].mean():.4f}")
    rec(6, "honest: per-client X_val is None (validation is global), 36 inputs, no Unnamed: 0",
        all(c["X_val"] is None for c in hc) and hc[0]["X_train"].shape[1] == 36 and h["X_test"].shape[1] == 36 and
        not ({"Unnamed: 0", "index", "Attack Category"} & set(h["feature_names"])), "")
    rec(6, "honest: majority-class accuracy logged for val and test",
        abs(h["info"]["eval_sets"]["test"]["majority_acc"] - (1 - 409 / 3263)) < 1e-12,
        f"test {h['info']['eval_sets']['test']['majority_acc']:.4f}, val {h['info']['eval_sets']['val']['majority_acc']:.4f}")
    h2 = P.build("honest", 15, balance_ratio=2)
    rec(6, "honest balance_ratio=2: 3 x 1,391 = 4,173 balanced rows", sum(len(c["y_train"]) for c in h2["clients"]) == 4173, "")
    h20 = P.build("honest", 15, balance_ratio=20)
    short = [c for c in h20["info"]["clients"] if c["balanced"]["short_of_normals"]]
    rec(6, "client short of normals keeps all of them and logs it (balance_ratio=20)",
        len(short) == 15 and all(c["balanced"]["rows"] == c["train"]["rows"] for c in short), f"{len(short)} clients short")

    # alignment of the carried Attack Category with the labels, and absence of NaN / non-finite values
    ok_cat = all(((c["cat_val"] != "normal") == (c["y_val"] == 1)).all() for c in cl) and \
        ((h["cat_test"] != "normal") == (h["y_test"] == 1)).all() and ((h["cat_val"] != "normal") == (h["y_val"] == 1)).all()
    rec(6, "cat_* is aligned with the labels (category != normal <=> attack) in every evaluation set", ok_cat,
        f"test {dict(zip(*np.unique(h['cat_test'], return_counts=True)))}")
    allx = [c["X_train"] for c in cl + hc] + [a["X_val"], h["X_val"], h["X_test"]]
    rec(6, "all inputs finite (no NaN / inf)", all(np.isfinite(x).all() for x in allx), "")

    # reproducibility (F9): same seed -> identical; different seed -> different normals sampled
    h_again, h_s1 = P.build("honest", 15), P.build("honest", 15, seed=1)
    rec(6, "seeded: same seed gives identical training sets", all(np.array_equal(x["X_train"], y["X_train"]) for x, y in zip(hc, h_again["clients"])), "")
    rec(6, "seeded: a different seed gives a different partition/sample, same test set",
        not all(np.array_equal(x["pos_train"], y["pos_train"]) for x, y in zip(hc, h_s1["clients"])) and
        np.array_equal(h["X_test"], h_s1["X_test"]), "")
    a_again = P.build("as_published", 15)
    rec(6, "as_published rebuild is deterministic (seeded balancing)", all(np.array_equal(x["X_train"], y["X_train"]) for x, y in zip(cl, a_again["clients"])), "")
    assert h["info"]["balancing"]["evaluation_sets_untouched"] and a["info"]["balancing"]["evaluation_sets_untouched"]


CHECKS = {1: checks_step1, 2: checks_step2, 3: checks_step3, 4: checks_step4, 5: checks_step5, 6: checks_step6}


def main() -> int:
    global UPTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=6, help="run checks for steps 1..N")
    UPTO = ap.parse_args().upto
    for step in range(1, UPTO + 1):
        if step in CHECKS:
            CHECKS[step]()
    n_fail = sum(not ok for _, _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed" + ("" if not n_fail else f"  ({n_fail} FAILED)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""Merge both tracks into one final table.   Run from our_work/ :  python -m compare.compare [--quick]

Writes compare/final_comparison.csv and compare/final_comparison.md
"""
import argparse
from pathlib import Path

import pandas as pd

from shared.metrics import fmt, load_rows, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = dict(acc=0.932, prec=0.915, rec=0.942, f1=0.928, eps=0.44, delta=1e-4)
GROUPS = ["Centralized", "Federated (no DP)", "Federated + DP", "Non-IID", "Non-IID + DP"]
PROTOCOL_NOTE = {
    "paper": "paper protocol: SMOTE on the whole dataset BEFORE the split -> test set is ~50% attack. "
             "Use only for a direct comparison with the paper's numbers.",
    "honest": "honest protocol: split first, balance training data only -> real test set (~12.5% attack). "
              "This is the protocol we trust.",
}


def build(quick):
    sub = "quick" if quick else ""
    frames = []
    for t in ("track_paper", "track_moe"):
        p = ROOT / t / "results" / sub / "results.jsonl"
        if p.exists():
            frames.append(load_rows(p))
        else:
            print(f"warning: {p} missing")
    if not frames:
        raise SystemExit("no results found")
    raw = pd.concat(frames, ignore_index=True)
    tmp = HERE / "_all.jsonl"
    raw.to_json(tmp, orient="records", lines=True)
    s = summarize(tmp)
    tmp.unlink()
    s["eps"] = s["eps"] if "eps" in s else float("nan")
    return s


def table(df, with_dp=True):
    cols = ["track", "model", "setting", "thr", "Acc", "Prec", "Rec", "F1", "AUC"] + (["eps", "delta", "DP guarantee"] if with_dp else [])
    rows = []
    for _, r in df.iterrows():
        rows.append([r.track, r.model, r.setting, r.thr_mode, fmt(r.acc_mean, r.acc_std), fmt(r.prec_mean, r.prec_std),
                     fmt(r.rec_mean, r.rec_std), fmt(r.f1_mean, r.f1_std), fmt(r.auc_mean, r.auc_std)]
                    + ([("-" if pd.isna(r.eps) else f"{r.eps:.2f}"), ("-" if pd.isna(r.get("delta")) else f"{r.delta:g}"),
                        r.dp_guarantee] if with_dp else []))
    return cols, rows


def md(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def winners(s, protocol):
    d = s[(s.protocol == protocol) & (s.thr_mode == "+thr") & (s.stage != "P0")]
    lines = []
    dp = d[d.group.isin(["Federated + DP"]) & d.dp_guarantee.astype(str).str.startswith("yes")]
    if len(dp):
        b = dp.sort_values("f1_mean", ascending=False).iloc[0]
        lines.append(f"- Best federated + DP result ({protocol}): **{b.track} / {b.model} / {b.setting}** - "
                     f"acc {b.acc_mean:.3f}, F1 {b.f1_mean:.3f}, epsilon {b.eps:.2f}.")
        e = dp[dp.eps <= PAPER["eps"] + 1e-6]
        if len(e):
            b2 = e.sort_values("f1_mean", ascending=False).iloc[0]
            lines.append(f"- Best result at epsilon <= 0.44 ({protocol}): **{b2.track} / {b2.model} / {b2.setting}** - "
                         f"acc {b2.acc_mean:.3f}, F1 {b2.f1_mean:.3f}, epsilon {b2.eps:.2f}.")
            lines.append(f"  - Beats the paper on accuracy AND F1 (0.932 / 0.928)? "
                         f"**{'YES' if b2.acc_mean > PAPER['acc'] and b2.f1_mean > PAPER['f1'] else 'NO'}**"
                         + ("  (paper protocol only: comparison with the paper's own, likely leaky, pipeline)"
                            if protocol == "paper" else "  (note: the paper's numbers come from a different, likely leaky, test set)"))
        else:
            lines.append(f"- No DP run reached epsilon <= 0.44 in the {protocol} protocol.")
        b3 = dp.sort_values("f1_mean", ascending=False).iloc[0]
        lines.append(f"  - Best DP run beats the paper on accuracy AND F1? "
                     f"**{'YES' if b3.acc_mean > PAPER['acc'] and b3.f1_mean > PAPER['f1'] else 'NO'}** (at epsilon {b3.eps:.2f}, not matched to 0.44)")
    for g in GROUPS:
        gg = d[d.group == g]
        if len(gg):
            b = gg.sort_values("f1_mean", ascending=False).iloc[0]
            lines.append(f"- Best in *{g}*: {b.track} / {b.model} / {b.setting} - F1 {b.f1_mean:.3f}, acc {b.acc_mean:.3f}.")
    # like-for-like DP pairs (identical setting string) -> MoE minus DNN
    p = d[(d.stage == "P5")].set_index("setting")
    m = d[(d.stage == "M3")].set_index("setting")
    common = p.index.intersection(m.index)
    if len(common):
        lines.append("- MoE minus DNN, F1, same DP setting: " + "; ".join(
            f"{k}: {m.loc[k].f1_mean - p.loc[k].f1_mean:+.3f}" for k in common))
    return "\n".join(lines) if lines else "- (not enough rows yet)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="read results/quick (smoke-test) files")
    a = ap.parse_args()
    s = build(a.quick)
    s.to_csv(HERE / ("final_comparison_quick.csv" if a.quick else "final_comparison.csv"), index=False)
    doc = ["# Final comparison: Track Paper vs Track MoE", "",
           "Mean ± std over seeds. `thr` = decision threshold: `@0.5` fixed, `+thr` tuned on a validation slice "
           "(never on the test set). Rows marked *no DP guarantee* are not private.", ""]
    for protocol in ("honest", "paper"):
        d = s[s.protocol == protocol]
        if d.empty:
            continue
        maj = d.majority_acc.dropna().iloc[0] if "majority_acc" in d and d.majority_acc.notna().any() else float("nan")
        doc += [f"## Protocol: {protocol}", "", f"_{PROTOCOL_NOTE[protocol]}_", ""]
        ref = [["-", "Paper (reported)", "WUSTL DNN, noise 1.5", "-", f"{PAPER['acc']:.3f}", f"{PAPER['prec']:.3f}",
                f"{PAPER['rec']:.3f}", f"{PAPER['f1']:.3f}", "-", "0.44", "0.0001", "yes (reported)"],
               ["-", "Baseline", "predict everything normal", "-", f"{maj:.3f}", "-", "0.000", "0.000", "0.500", "-", "-", "-"]]
        for g in GROUPS:
            gd = d[d.group == g].sort_values(["stage", "setting", "model", "thr_mode"])
            if gd.empty:
                continue
            cols, rows = table(gd)
            doc += [f"### {g}", "", md(cols, (ref if g == GROUPS[0] else []) + rows), ""]
        doc += ["### Winners", "", winners(s, protocol), ""]
    (HERE / ("final_comparison_quick.md" if a.quick else "final_comparison.md")).write_text("\n".join(doc), encoding="utf8")
    print("\n".join(doc))


if __name__ == "__main__":
    main()

import os
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = r"D:\微信\xwechat_files\wxid_4m43mtilpnek22_4b5b\msg\file\2026-07\张瑞-实验环境\实验环境\event_extraction"
OUT_DIR = r"C:\Users\50317\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a7e861ad62998b716fef0f0"

DATASETS = ["AmazonHelp", "AppleSupport", "SpotifyCares"]

METHODS = [
    ("LLMs-EEM DS-R", "WorkFlow/DeepSeekReasoner-DeepSeekReasoner",
     "twcs-{ds}-300-predicted-DeepSeekReasoner-DeepSeekReasoner.xlsx"),
    ("LLMs-EEM Gemini", "WorkFlow/Gemini3Pro-Gemini3Pro",
     "twcs-{ds}-300-predicted-Gemini3Pro-Gemini3Pro.xlsx"),
    ("Zero-shot DS-R", "Single_LLM/DeepSeekReasoner",
     "twcs-{ds}-300-predicted-DeepSeekReasoner.xlsx"),
    ("Zero-shot Gemini", "Single_LLM/Gemini3Pro",
     "twcs-{ds}-300-predicted-Gemini3Pro.xlsx"),
]


def load_pred(rel_folder, fname, ds):
    p = os.path.join(BASE, rel_folder, fname.format(ds=ds))
    df = pd.read_excel(p)
    acts = [c for c in df.columns if not c.startswith("pred_")
            and c in {cc[len("pred_"):] for cc in df.columns if cc.startswith("pred_")}]
    return df, acts


def method_error_stats(df, acts):
    stats = {}
    conf = defaultdict(int)
    for a in acts:
        gold = df[a].fillna(0).astype(int).values
        pred = df[f"pred_{a}"].fillna(0).astype(int).values
        tp = int(((gold == 1) & (pred == 1)).sum())
        fp = int(((gold == 0) & (pred == 1)).sum())
        fn = int(((gold == 1) & (pred == 0)).sum())
        stats[a] = {"TP": tp, "FP": fp, "FN": fn, "support": int(gold.sum())}
    # substitution pairs: missed gold activity a, spurious pred activity b on same message
    for _, row in df.iterrows():
        missed = [a for a in acts if pd.notna(row[a]) and row[a] == 1 and row[f"pred_{a}"] == 0]
        spurious = [b for b in acts if pd.isna(row[b]) and row[f"pred_{b}"] == 1]
        for a in missed:
            for b in spurious:
                conf[(a, b)] += 1
    return stats, conf


def main():
    all_rows = []
    conf_summary = defaultdict(int)
    for ds in DATASETS:
        for method, folder, pattern in METHODS:
            df, acts = load_pred(folder, pattern, ds)
            stats, conf = method_error_stats(df, acts)
            for a, s in stats.items():
                all_rows.append({
                    "Dataset": ds, "Method": method, "Activity": a,
                    "Support": s["support"], "TP": s["TP"], "FP": s["FP"], "FN": s["FN"],
                })
            if method.startswith("LLMs-EEM"):
                for (a, b), c in conf.items():
                    conf_summary[(ds, method, a, b)] += c

    err = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "error_analysis_per_activity.csv")
    err.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out_csv)
    print()

    for method in [m[0] for m in METHODS]:
        sub = err[err["Method"] == method]
        tot_fp, tot_fn = sub["FP"].sum(), sub["FN"].sum()
        print(f"{method:<18} total FP={tot_fp:<4} total FN={tot_fn:<4} "
              f"(prec-loss {tot_fp/(tot_fp+sub['TP'].sum()):.1%}, "
              f"rec-loss {tot_fn/(tot_fn+sub['TP'].sum()):.1%})")

    print()
    print("Top error-prone activities by (FP+FN), LLMs-EEM aggregated over backbones:")
    sub = err[err["Method"].str.startswith("LLMs-EEM")]
    agg = sub.groupby(["Dataset", "Activity"]).agg(
        Support=("Support", "mean"), FP=("FP", "sum"), FN=("FN", "sum")).reset_index()
    agg["Err"] = agg["FP"] + agg["FN"]
    for ds in DATASETS:
        top = agg[agg["Dataset"] == ds].nlargest(5, "Err")
        print(f"  {ds}:")
        for _, r in top.iterrows():
            print(f"    {r['Activity']:<35} n={int(r['Support']):<3} FP={int(r['FP']):<3} FN={int(r['FN']):<3}")

    print()
    print("Top substitution pairs (gold missed -> spurious predicted), LLMs-EEM:")
    pairs = sorted(conf_summary.items(), key=lambda kv: -kv[1])[:15]
    for (ds, method, a, b), c in pairs:
        print(f"  {c:>3}x  [{ds} {method}]  {a}  ->  {b}")

    # Error-type attribution: rare vs frequent activities
    print()
    print("Error rate by activity frequency quartile (LLMs-EEM aggregated):")
    sub2 = err[err["Method"].str.startswith("LLMs-EEM")].copy()
    g = sub2.groupby("Activity").agg(Support=("Support", "mean"),
                                     TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum")).reset_index()
    g["Recall"] = g["TP"] / (g["TP"] + g["FN"])
    g["Precision"] = g["TP"] / (g["TP"] + g["FP"])
    g["freq_bin"] = pd.qcut(g["Support"], 4, labels=["Q1 rarest", "Q2", "Q3", "Q4 most frequent"])
    print(g.groupby("freq_bin", observed=True).agg(
        n_activities=("Activity", "count"),
        median_support=("Support", "median"),
        mean_recall=("Recall", "mean"),
        mean_precision=("Precision", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()

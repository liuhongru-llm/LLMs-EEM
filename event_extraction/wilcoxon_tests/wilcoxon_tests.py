import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"D:\微信\xwechat_files\wxid_4m43mtilpnek22_4b5b\msg\file\2026-07\张瑞-实验环境\实验环境\event_extraction"

DATASETS = ["AmazonHelp", "AppleSupport", "SpotifyCares"]

BACKBONES = [
    ("DS-R", "WorkFlow_Evaluation_Results/DeepSeekReasoner-DeepSeekReasoner",
             "Single_LLM_Evaluation_Results/DeepSeekReasoner"),
    ("Gemini", "WorkFlow_Evaluation_Results/Gemini3Pro-Gemini3Pro",
               "Single_LLM_Evaluation_Results/Gemini3Pro"),
]

METRICS = ["F1", "MCC"]


def load_per_activity(rel_path, dataset, metric):
    folder = os.path.join(BASE, rel_path)
    for fname in sorted(os.listdir(folder)):
        if not (fname.startswith("eval_twcs-") and fname.endswith(".xlsx")):
            continue
        if not fname.startswith(f"eval_twcs-{dataset}-"):
            continue
        df = pd.read_excel(os.path.join(folder, fname))
        if "Header" in df.columns and "Macro Average" in df["Header"].values:
            df = df[df["Header"] != "Macro Average"]
        series = pd.to_numeric(df[metric], errors="coerce")
        return dict(zip(df["Header"], series))
    return {}


def wilcoxon_effect(diffs):
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return None
    ranks = stats.rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    r_rb = (r_plus - r_minus) / (r_plus + r_minus)
    return r_rb


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        corrected[idx] = running
    return corrected


def main():
    results = {m: [] for m in METRICS}
    for metric in METRICS:
        print("=" * 100)
        print(f"Paired Wilcoxon signed-rank tests — metric: {metric}  (LLMs-EEM vs Zero-shot)")
        print("=" * 100)
        for bk_name, wf_rel, zs_rel in BACKBONES:
            for ds in DATASETS:
                wf_map = load_per_activity(wf_rel, ds, metric)
                zs_map = load_per_activity(zs_rel, ds, metric)
                common = [h for h in wf_map if h in zs_map
                          and not np.isnan(wf_map[h]) and not np.isnan(zs_map[h])]
                wf_vals = np.array([wf_map[h] for h in common])
                zs_vals = np.array([zs_map[h] for h in common])
                diffs = wf_vals - zs_vals
                n_pairs = len(common)
                n_zero = int((diffs == 0).sum())
                nz = diffs[diffs != 0]
                if len(nz) >= 5:
                    stat, p = stats.wilcoxon(wf_vals, zs_vals, zero_method="wilcox",
                                             alternative="two-sided")
                    r_rb = wilcoxon_effect(diffs)
                else:
                    stat, p, r_rb = np.nan, np.nan, np.nan
                results[metric].append({
                    "backbone": bk_name, "dataset": ds, "n": n_pairs,
                    "n_zero": n_zero, "W": stat, "p": p, "r_rb": r_rb,
                    "median_gain": float(np.median(diffs)) if len(diffs) else np.nan,
                    "n_positive": int((diffs > 0).sum()),
                    "n_negative": int((diffs < 0).sum()),
                })
                print(f"  {bk_name:<7} {ds:<14} n={n_pairs:<3} ties={n_zero:<2} "
                      f"W={stat:>8.1f}  p={p:.3e}  r_rb={r_rb:+.3f}  "
                      f"median_gain={np.median(diffs):+.4f}  (+{int((diffs>0).sum())}/-{int((diffs<0).sum())})")

        # Holm correction within the 6 comparisons of this metric
        pvals = np.array([r["p"] for r in results[metric]], dtype=float)
        adj = holm(pvals)
        print(f"\n  Holm-corrected p-values ({metric}, m=6):")
        for r, pa in zip(results[metric], adj):
            flag = "significant" if pa < 0.001 else ("significant" if pa < 0.05 else "NOT significant")
            print(f"    {r['backbone']:<7} {r['dataset']:<14} p_holm={pa:.3e}   [{flag}]")
        print()

    # Pooled per backbone (all three datasets combined, for narrative)
    print("=" * 100)
    print("Pooled per backbone (all datasets combined, paired by activity)")
    print("=" * 100)
    for metric in METRICS:
        for bk_name, wf_rel, zs_rel in BACKBONES:
            all_wf, all_zs = [], []
            for ds in DATASETS:
                wf_map = load_per_activity(wf_rel, ds, metric)
                zs_map = load_per_activity(zs_rel, ds, metric)
                common = [h for h in wf_map if h in zs_map
                          and not np.isnan(wf_map[h]) and not np.isnan(zs_map[h])]
                all_wf += [wf_map[h] for h in common]
                all_zs += [zs_map[h] for h in common]
            wf_vals = np.array(all_wf)
            zs_vals = np.array(all_zs)
            diffs = wf_vals - zs_vals
            stat, p = stats.wilcoxon(wf_vals, zs_vals, zero_method="wilcox",
                                     alternative="two-sided")
            r_rb = wilcoxon_effect(diffs)
            print(f"  {metric:<4} {bk_name:<7} n={len(wf_vals):<3} W={stat:>8.1f}  "
                  f"p={p:.3e}  r_rb={r_rb:+.3f}  median_gain={np.median(diffs):+.4f}")

    # Export summary table for the paper
    rows = []
    for metric in METRICS:
        pvals = np.array([r["p"] for r in results[metric]], dtype=float)
        adj = holm(pvals)
        for r, pa in zip(results[metric], adj):
            rows.append({
                "Metric": metric, "Backbone": r["backbone"], "Dataset": r["dataset"],
                "n_pairs": r["n"], "n_ties": r["n_zero"],
                "W": round(float(r["W"]), 1),
                "p_raw": f"{r['p']:.2e}",
                "p_holm": f"{pa:.2e}",
                "rank_biserial_r": round(float(r["r_rb"]), 3),
                "median_gain": round(r["median_gain"], 4),
                "wins": f"+{r['n_positive']}/-{r['n_negative']}",
            })
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "wilcoxon_results.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

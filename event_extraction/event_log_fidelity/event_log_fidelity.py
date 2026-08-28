import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"D:\微信\xwechat_files\wxid_4m43mtilpnek22_4b5b\msg\file\2026-07\张瑞-实验环境\实验环境\event_extraction"
OUT = r"C:\Users\50317\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a7e861ad62998b716fef0f0"
DATASETS = ["AmazonHelp", "AppleSupport", "SpotifyCares"]
META = {"main_tweet_id", "tweet_id", "in_response_to_tweet_id", "response_tweet_id",
        "created_at", "author_id", "inbound", "text", "company"}

METHODS = {
    "LLMs-EEM (DS-R)": os.path.join(BASE, "WorkFlow", "DeepSeekReasoner-DeepSeekReasoner", "twcs-{ds}-300-predicted-DeepSeekReasoner-DeepSeekReasoner.xlsx"),
    "LLMs-EEM (Gemini)": os.path.join(BASE, "WorkFlow", "Gemini3Pro-Gemini3Pro", "twcs-{ds}-300-predicted-Gemini3Pro-Gemini3Pro.xlsx"),
    "Zero-shot (DS-R)": os.path.join(BASE, "Single_LLM", "DeepSeekReasoner", "twcs-{ds}-300-predicted-DeepSeekReasoner.xlsx"),
    "Zero-shot (Gemini)": os.path.join(BASE, "Single_LLM", "Gemini3Pro", "twcs-{ds}-300-predicted-Gemini3Pro.xlsx"),
}


def load_log(path):
    df = pd.read_excel(path)
    acts = sorted(c for c in df.columns
                  if c not in META and not c.startswith("pred_") and not str(c).startswith("Unnamed"))
    df["_ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values(["main_tweet_id", "_ts"], kind="stable").reset_index(drop=True)
    gold = df[acts].fillna(0).astype(int).values
    pred = df[["pred_" + a for a in acts]].fillna(0).astype(int).values
    return df, acts, gold, pred


def build_traces(df, mat, acts):
    traces = {}
    for cid, idx in df.groupby("main_tweet_id", sort=False).indices.items():
        seq = []
        for i in idx:
            for j in np.nonzero(mat[i])[0]:
                seq.append(acts[j])
        traces[cid] = seq
    return traces


def dfg_edges(traces):
    edges = Counter()
    for seq in traces.values():
        for a, b in zip(seq[:-1], seq[1:]):
            edges[(a, b)] += 1
    return edges


def jaccard(s1, s2):
    u = s1 | s2
    if not u:
        return 1.0
    return len(s1 & s2) / len(u)


def evaluate(gold_traces, pred_traces):
    cids = sorted(gold_traces.keys())
    n = len(cids)
    trace_exact = sum(gold_traces[c] == pred_traces[c] for c in cids)
    set_exact = sum(set(gold_traces[c]) == set(pred_traces[c]) for c in cids)
    mean_jaccard = np.mean([jaccard(set(gold_traces[c]), set(pred_traces[c])) for c in cids])
    ge, pe = dfg_edges(gold_traces), dfg_edges(pred_traces)
    gs, ps = set(ge), set(pe)
    inter = len(gs & ps)
    p_set = inter / len(ps) if ps else 1.0
    r_set = inter / len(gs) if gs else 1.0
    f_set = 2 * p_set * r_set / (p_set + r_set) if (p_set + r_set) else 0.0
    overlap = sum(min(ge[e], pe[e]) for e in gs & ps)
    p_w = overlap / sum(pe.values()) if pe else 1.0
    r_w = overlap / sum(ge.values()) if ge else 1.0
    f_w = 2 * p_w * r_w / (p_w + r_w) if (p_w + r_w) else 0.0
    return {
        "n_traces": n,
        "trace_exact": trace_exact / n,
        "set_exact": set_exact / n,
        "jaccard": mean_jaccard,
        "dfg_P": p_set, "dfg_R": r_set, "dfg_F1": f_set,
        "dfg_P_w": p_w, "dfg_R_w": r_w, "dfg_F1_w": f_w,
        "n_gold_edges": len(gs), "n_pred_edges": len(ps),
    }


rows = []
log_lines = []

for method, tmpl in METHODS.items():
    all_gold, all_pred = {}, {}
    for ds in DATASETS:
        path = tmpl.format(ds=ds)
        df, acts, gold, pred = load_log(path)
        gt = build_traces(df, gold, acts)
        pt = build_traces(df, pred, acts)
        res = evaluate(gt, pt)
        res.update({"method": method, "dataset": ds, "n_msgs": len(df)})
        rows.append(res)
        for k in all_gold:
            pass
        for cid, seq in gt.items():
            all_gold[(ds, cid)] = seq
        for cid, seq in pt.items():
            all_pred[(ds, cid)] = seq
        log_lines.append(f"{method:<20} {ds:<14} msgs={len(df):>3} traces={res['n_traces']:>3} "
                         f"traceEM={res['trace_exact']:.3f} setEM={res['set_exact']:.3f} "
                         f"J={res['jaccard']:.3f} DFG-F1={res['dfg_F1']:.3f} (P={res['dfg_P']:.3f} R={res['dfg_R']:.3f}) "
                         f"wDFG-F1={res['dfg_F1_w']:.3f}")
    res_all = evaluate(all_gold, all_pred)
    res_all.update({"method": method, "dataset": "OVERALL", "n_msgs": sum(r["n_msgs"] for r in rows if r["method"] == method)})
    rows.append(res_all)
    log_lines.append(f"{method:<20} {'OVERALL':<14} msgs={res_all['n_msgs']:>3} traces={res_all['n_traces']:>3} "
                     f"traceEM={res_all['trace_exact']:.3f} setEM={res_all['set_exact']:.3f} "
                     f"J={res_all['jaccard']:.3f} DFG-F1={res_all['dfg_F1']:.3f} (P={res_all['dfg_P']:.3f} R={res_all['dfg_R']:.3f}) "
                     f"wDFG-F1={res_all['dfg_F1_w']:.3f}")
    log_lines.append("")

# gold sanity: number of traces per dataset from gold data
log_lines.insert(0, "gold trace counts per dataset (from LLMs-EEM DS-R file, gold columns):")
tmpl = METHODS["LLMs-EEM (DS-R)"]
for ds in DATASETS:
    df, acts, gold, _ = load_log(tmpl.format(ds=ds))
    gt = build_traces(df, gold, acts)
    n_events = sum(len(v) for v in gt.values())
    log_lines.insert(1, f"  {ds}: {len(gt)} traces, {len(df)} messages, {n_events} gold events")

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(OUT, "event_log_fidelity.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(OUT, "event_log_fidelity.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(log_lines))
    fh.write("\n\n=== full table ===\n")
    fh.write(res_df[["method", "dataset", "n_msgs", "n_traces", "trace_exact", "set_exact",
                     "jaccard", "dfg_P", "dfg_R", "dfg_F1", "dfg_P_w", "dfg_R_w", "dfg_F1_w"]].to_string(index=False))
print("done")

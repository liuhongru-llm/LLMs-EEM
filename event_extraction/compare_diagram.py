import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.lines import Line2D

# matplotlib.use('Agg')  # 服务器无屏幕时取消注释
matplotlib.use('TkAgg')

# --- 论文通用样式 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 21

# ------------------------------------------------------------------ #
#  路径配置                                                            #
# ------------------------------------------------------------------ #
FINA_DIR   = "WorkFlow_Evaluation_Results"          # WorkflowA / WorkflowB
SINGLE_DIR = "Single_LLM_Evaluation_Results"        # BaselineA / BaselineB
NLI_DIR    = "compare_Evaluation_Results"         # NLI+BART

# ------------------------------------------------------------------ #
#  指标配置：切换 "Balanced_Accuracy" 或 "MCC"或者“F1”                         #
# ------------------------------------------------------------------ #
METRIC = "MCC"   # ← 改这一行即可切换指标

BA_COL_EVAL = METRIC
BA_COL_NLI  = METRIC

METRIC_YLIM = (-0.05, 1.05)  # Y轴跟随数据范围，MCC不强制显示负轴

# 方法名 → 显示名
METHOD_NAMES = {
    "WorkflowA":  "LLMs-EEM(DeepSeekReasoner)",
    "WorkflowB":  "LLMs-EEM(Gemini3Pro)",
    "BaselineA":  "Zero-shot‑LLM(DeepSeekReasoner)",
    "BaselineB":  "Zero-shot‑LLM(Gemini3Pro)",
    "NLI":        "NLI-BART",
}

# 方法名 → 颜色
METHOD_COLORS = {
    "WorkflowA": "#27AE60",   # 深绿
    "WorkflowB": "#F2994A",   # 深橙
    "BaselineA": "#82D9A0",   # 浅绿
    "BaselineB": "#FFCA8A",   # 浅橙
    "NLI":       "#A0A0A0",   # 灰
}

# ------------------------------------------------------------------ #
#  辅助：从文件名中提取公司名                                           #
# ------------------------------------------------------------------ #
def extract_company(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


# ------------------------------------------------------------------ #
#  1. 读取 Workflow 数据                                               #
# ------------------------------------------------------------------ #
WORKFLOW_MAP = {
    "DeepSeekReasoner-DeepSeekReasoner": "WorkflowA",
    "Gemini3Pro-Gemini3Pro":             "WorkflowB",
}

records = []

for subdir, method_key in WORKFLOW_MAP.items():
    folder = os.path.join(FINA_DIR, subdir)
    if not os.path.exists(folder):
        print(f"[WARN] Not found: {folder}")
        continue
    for fname in os.listdir(folder):
        if not (fname.startswith("eval_") and fname.endswith(".xlsx")):
            continue
        company = extract_company(fname.replace("eval_twcs-", "twcs-"))
        fpath = os.path.join(folder, fname)
        try:
            df = pd.read_excel(fpath)
            df = df[df["Header"] != "Macro Average"]
            if BA_COL_EVAL in df.columns:
                for val in df[BA_COL_EVAL].dropna():
                    records.append({"Method": method_key,
                                    "Company": company,
                                    METRIC: val})
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")

# ------------------------------------------------------------------ #
#  2. 读取 Baseline 数据                                               #
# ------------------------------------------------------------------ #
BASELINE_MAP = {
    "DeepSeekReasoner": "BaselineA",
    "Gemini3Pro":       "BaselineB",
}

for subdir, method_key in BASELINE_MAP.items():
    folder = os.path.join(SINGLE_DIR, subdir)
    if not os.path.exists(folder):
        print(f"[WARN] Not found: {folder}")
        continue
    for fname in os.listdir(folder):
        if not (fname.startswith("eval_") and fname.endswith(".xlsx")):
            continue
        company = extract_company(fname.replace("eval_twcs-", "twcs-"))
        fpath = os.path.join(folder, fname)
        try:
            df = pd.read_excel(fpath)
            df = df[df["Header"] != "Macro Average"]
            if BA_COL_EVAL in df.columns:
                for val in df[BA_COL_EVAL].dropna():
                    records.append({"Method": method_key,
                                    "Company": company,
                                    METRIC: val})
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")

# ------------------------------------------------------------------ #
#  3. 读取 NLI+BART 数据                                               #
# ------------------------------------------------------------------ #
if os.path.exists(NLI_DIR):
    for fname in os.listdir(NLI_DIR):
        if not fname.endswith("-metrics.xlsx"):
            continue
        company = fname.replace("-metrics.xlsx", "")
        fpath = os.path.join(NLI_DIR, fname)
        try:
            df = pd.read_excel(fpath)
            df = df[df["Header"] != "Macro Average"]
            if BA_COL_NLI in df.columns:
                for val in df[BA_COL_NLI].dropna():
                    records.append({"Method": "NLI",
                                    "Company": company,
                                    METRIC: val})
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")
else:
    print(f"[WARN] NLI dir not found: {NLI_DIR}")

# ------------------------------------------------------------------ #
#  4. 整理 DataFrame                                                   #
# ------------------------------------------------------------------ #
df_all = pd.DataFrame(records)

if df_all.empty:
    print("[ERROR] No data loaded. Check your directory paths.")
    exit()

df_all["Method_Label"] = df_all["Method"].map(METHOD_NAMES)

hue_order = [METHOD_NAMES[k] for k in ["WorkflowA", "WorkflowB", "BaselineA", "BaselineB", "NLI"]]
palette    = {METHOD_NAMES[k]: METHOD_COLORS[k] for k in METHOD_COLORS}

companies = ["SpotifyCares", "AppleSupport", "AmazonHelp"]

print(f"Companies found: {companies}")
print(f"Methods found:   {df_all['Method'].unique()}")
print(f"Total rows:      {len(df_all)}")

# ------------------------------------------------------------------ #
#  5. 绘图                                                             #
# ------------------------------------------------------------------ #
print("\n" + "="*80)
print(f"  Quartile Statistics ({METRIC})")
print("="*80)

for company in companies:
    print(f"\n  Company: {company}")
    print(f"  {'Method':<35} {'Q1':>8} {'Q2 (Med)':>10} {'Q3':>8} {'IQR':>8} {'N':>5}")
    print("  " + "-"*74)
    for method_key in ["WorkflowA", "WorkflowB", "BaselineA", "BaselineB", "NLI"]:
        label = METHOD_NAMES[method_key]
        subset = df_all[
            (df_all["Company"] == company) &
            (df_all["Method"] == method_key)
        ][METRIC].dropna()
        if subset.empty:
            print(f"  {label:<35} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'0':>5}")
            continue
        q1  = subset.quantile(0.25)
        q2  = subset.quantile(0.50)
        q3  = subset.quantile(0.75)
        iqr = q3 - q1
        print(f"  {label:<35} {q1:>8.4f} {q2:>10.4f} {q3:>8.4f} {iqr:>8.4f} {len(subset):>5}")

print("\n" + "="*80 + "\n")
fig, ax = plt.subplots(figsize=(22, 12))

sns.violinplot(
    x="Company",
    y=METRIC,
    hue="Method_Label",
    data=df_all,
    palette=palette,
    hue_order=hue_order,
    order=companies,
    dodge=True,
    gap=0.2,
    linewidth=1.5,
    inner="quartile",
    cut=0,
    scale="width",
    ax=ax,
    width=0.8,
    saturation=1.0,
    inner_kws=dict(color='black', linewidth=4)
)

ax.set_xlabel("")
ax.set_ylabel(METRIC.replace("_", " "), fontsize=22, fontweight='bold')
ax.set_ylim(*METRIC_YLIM)

plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontweight='bold', fontsize=22)
ax.tick_params(axis='x', which='major', length=8, width=2, direction='out')
ax.tick_params(axis='y', which='major', length=6, width=2, direction='out')
ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

existing_labels = df_all["Method_Label"].unique()
legend_elements = [
    Line2D([0], [0], color=palette[m], lw=6, label=m)
    for m in hue_order if m in existing_labels
]

# 图例放在图内，x 轴上方偏低位置
legend = ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    fontsize=20,
    frameon=True,
    framealpha=1
)
legend.get_frame().set_linewidth(2.5)
legend.get_frame().set_edgecolor('black')

plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.99)

plt.savefig("Comparison_Merged_ColorPaired.png", dpi=600)
plt.show()
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# matplotlib.use('Agg')  # 服务器无屏幕时取消注释
matplotlib.use('TkAgg')

# --- 论文通用样式 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 21

# ------------------------------------------------------------------ #
#  指标配置：切换 "Balanced_Accuracy" 或 "MCC" 等                      #
# ------------------------------------------------------------------ #
METRIC = "F1"   # ← 改这一行即可切换指标
METRIC_YLIM = (-0.05, 1.05)

# ------------------------------------------------------------------ #
#  路径配置                                                            #
# ------------------------------------------------------------------ #

FINA_DIR          = "WorkFLow_Evaluation_Results"           # WorkflowA / WorkflowB
SINGLE_DIR        = "WorkFLow_no_self_Evaluation_Results"         # BaselineA / BaselineB
NO_VALIDATION_DIR = "WorkFLow_no_val_Evaluation_Results"   # NoValidation
NO_RAG_DIR        = "WorkFLow_no_RAG_Evaluation_Results"    # NoRAG
# ------------------------------------------------------------------ #
#  方法配置                                                            #
# ------------------------------------------------------------------ #
METHOD_NAMES = {
    "WorkflowA":       "LLMs-EEM(DeepSeekReasoner)",
    "WorkflowB":       "LLMs-EEM(Gemini3Pro)",
    "NoValidationA":   "LLMs-EEM w/o Val(DeepSeekReasoner)",
    "NoValidationB":   "LLMs-EEM w/o Val(Gemini3Pro)",
    "NoRAGA":          "LLMs-EEM w/o RAG(DeepSeekReasoner)",
    "NoRAGB":          "LLMs-EEM w/o RAG(Gemini3Pro)",
    "NoSCA":           "LLMs-EEM w/o SC(DeepSeekReasoner)",
    "NoSCB":           "LLMs-EEM w/o SC(Gemini3Pro)",
}

METHOD_COLORS = {
    "WorkflowA":     "#2ECC71",
    "WorkflowB":     "#E8873A",
    "NoValidationA": "#58D68D",
    "NoValidationB": "#F5B07A",
    "NoRAGA":        "#A9DFBF",
    "NoRAGB":        "#FAD7A0",
    "NoSCA":         "#D5F5E3",
    "NoSCB":         "#FDEBD0",
}

HUE_ORDER_KEYS = [
    "WorkflowA", "WorkflowB",
    "NoValidationA", "NoValidationB",
    "NoRAGA", "NoRAGB",
    "NoSCA", "NoSCB",
]

# ------------------------------------------------------------------ #
#  辅助函数                                                            #
# ------------------------------------------------------------------ #
def extract_company(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def load_from_dir(base_dir, subdir_map, records):
    for subdir, method_key in subdir_map.items():
        folder = os.path.join(base_dir, subdir)
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
                if METRIC in df.columns:
                    for val in df[METRIC].dropna():
                        records.append({"Method": method_key,
                                        "Company": company,
                                        METRIC: val})
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")


records = []

# 1. Workflow（完整流程）
load_from_dir(FINA_DIR, {
    "DeepSeekReasoner-DeepSeekReasoner": "WorkflowA",
    "Gemini3Pro-Gemini3Pro":             "WorkflowB",
}, records)

# 2. NoValidation（去掉验证步骤）
load_from_dir(NO_VALIDATION_DIR, {
    "DeepSeekReasoner": "NoValidationA",
    "Gemini3Pro":             "NoValidationB",
}, records)

# 3. NoRAG（去掉知识库）
load_from_dir(NO_RAG_DIR, {
    "DeepSeekReasoner": "NoRAGA",
    "Gemini3Pro":             "NoRAGB",
}, records)

# 4. Baseline（单模型）
load_from_dir(SINGLE_DIR, {
    "DeepSeekReasoner": "NoSCA",
    "Gemini3Pro":       "NoSCB",
}, records)
# ------------------------------------------------------------------ #
#  整理 DataFrame                                                      #
# ------------------------------------------------------------------ #
df_all = pd.DataFrame(records)

if df_all.empty:
    print("[ERROR] No data loaded. Check your directory paths.")
    exit()

df_all["Method_Label"] = df_all["Method"].map(METHOD_NAMES)

hue_order = [METHOD_NAMES[k] for k in HUE_ORDER_KEYS]
palette   = {METHOD_NAMES[k]: METHOD_COLORS[k] for k in METHOD_COLORS}
companies = ["SpotifyCares", "AppleSupport", "AmazonHelp"]

print(f"Companies found: {companies}")
print(f"Methods found:   {df_all['Method'].unique()}")
print(f"Total rows:      {len(df_all)}")

# ------------------------------------------------------------------ #
#  四分位数统计输出                                                     #
# ------------------------------------------------------------------ #
print("\n" + "=" * 80)
print(f"  Quartile Statistics ({METRIC})")
print("=" * 80)

for company in companies:
    print(f"\n  Company: {company}")
    print(f"  {'Method':<45} {'Q1':>8} {'Q2 (Med)':>10} {'Q3':>8} {'IQR':>8} {'N':>5}")
    print("  " + "-" * 84)
    df_c = df_all[df_all["Company"] == company]
    for key in HUE_ORDER_KEYS:
        label = METHOD_NAMES[key]
        vals = df_c[df_c["Method"] == key][METRIC].dropna()
        if len(vals) == 0:
            continue
        q1  = np.percentile(vals, 25)
        q2  = np.percentile(vals, 50)
        q3  = np.percentile(vals, 75)
        iqr = q3 - q1
        n   = len(vals)
        print(f"  {label:<45} {q1:>8.4f} {q2:>10.4f} {q3:>8.4f} {iqr:>8.4f} {n:>5}")

print("\n" + "=" * 80 + "\n")

# ------------------------------------------------------------------ #
#  绘图                                                                #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(26, 12))

sns.violinplot(
    x="Company",
    y=METRIC,
    hue="Method_Label",
    data=df_all,
    palette=palette,
    hue_order=hue_order,
    order=companies,
    dodge=True,
    gap=0.15,
    linewidth=1.5,
    inner="quartile",
    cut=0,
    scale="width",
    ax=ax,
    width=0.85,
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

# ------------------------------------------------------------------ #
#  图例：移至图内下方居中（与图片位置一致）                              #
# ------------------------------------------------------------------ #
existing_labels = df_all["Method_Label"].unique()
legend_elements = [
    Line2D([0], [0], color=palette[m], lw=6, label=m)
    for m in hue_order if m in existing_labels
]

legend = ax.legend(
    handles=legend_elements,
    loc='lower center',          # 图内下方居中
    bbox_to_anchor=(0.5, 0.01),  # 微调垂直位置，紧贴底部但在轴内
    ncol=2,
    fontsize=20,
    frameon=True,
    framealpha=1
)
legend.get_frame().set_linewidth(2.5)
legend.get_frame().set_edgecolor('black')

# 图例移入图内后，底部无需留额外空间
plt.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.99)

plt.savefig("Comparison_Merged_ColorPaired.png", dpi=600, bbox_inches='tight')
plt.show()
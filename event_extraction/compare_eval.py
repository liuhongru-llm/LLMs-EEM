import os
import pandas as pd

PREDICTED_DIR = "compare"
OUTPUT_DIR = "compare_Evaluation_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_COLS = {
    "main_tweet_id", "tweet_id", "text", "company",
    "created_at", "author_id", "inbound",
    "response_tweet_id", "in_response_to_tweet_id"
}


def save_formatted_excel(df_results, out_path):
    try:
        writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        num_rows = len(df_results) + 1

        format_options = {
            'type': '3_color_scale',
            'min_color': '#FF0000',
            'mid_color': '#FFFF00',
            'max_color': '#00FF00',
            'min_type': 'num',
            'mid_type': 'percentile',
            'max_type': 'num',
            'min_value': 0,
            'max_value': 1,
            'mid_value': 50
        }

        metrics_cols = ['B', 'C', 'D', 'E']  # F1 Precision Accuracy Recall

        for column in metrics_cols:
            range_str = f'{column}2:{column}{num_rows}'
            worksheet.conditional_format(range_str, format_options)

        worksheet.set_column('A:B', 28)
        for col in metrics_cols:
            worksheet.set_column(f'{col}:{col}', 14)
        worksheet.set_column('G:G', 12)

        bold_format = workbook.add_format({'bold': True, 'bg_color': '#E0E0E0'})
        worksheet.set_row(num_rows - 1, None, bold_format)

        writer.close()
        print(f"✅ Saved formatted: {out_path}")

    except Exception as e:
        print(f"⚠️ Format failed, fallback save: {e}")
        df_results.to_excel(out_path, index=False)


def parse_brand(filename):
    # twcs-AmazonHelp-146-outbound-predicted.xlsx → AmazonHelp
    return filename.split("-")[1]


brand_stats = {}
brand_order = {}   # ✅ 新增：记录每个 brand 的 header 顺序（按文件列出现顺序）

# ========== 1. 扫描 predicted 文件 ==========
import os
import pandas as pd

PREDICTED_DIR = "compare"
OUTPUT_DIR = "compare_Evaluation_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_COLS = {
    "main_tweet_id", "tweet_id", "text", "company",
    "created_at", "author_id", "inbound",
    "response_tweet_id", "in_response_to_tweet_id"
}


def save_formatted_excel(df_results, out_path):
    try:
        writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        num_rows = len(df_results) + 1

        format_options = {
            'type': '3_color_scale',
            'min_color': '#FF0000',
            'mid_color': '#FFFF00',
            'max_color': '#00FF00',
            'min_type': 'num',
            'mid_type': 'percentile',
            'max_type': 'num',
            'min_value': 0,
            'max_value': 1,
            'mid_value': 50
        }

        metrics_cols = ['B', 'C', 'D', 'E']  # F1 Precision Accuracy Recall

        for column in metrics_cols:
            range_str = f'{column}2:{column}{num_rows}'
            worksheet.conditional_format(range_str, format_options)

        worksheet.set_column('A:B', 28)
        for col in metrics_cols:
            worksheet.set_column(f'{col}:{col}', 14)
        worksheet.set_column('F:F', 12)

        bold_format = workbook.add_format({'bold': True, 'bg_color': '#E0E0E0'})
        worksheet.set_row(num_rows - 1, None, bold_format)

        writer.close()
        print(f"✅ Saved formatted: {out_path}")

    except Exception as e:
        print(f"⚠️ Format failed, fallback save: {e}")
        df_results.to_excel(out_path, index=False)


def parse_brand(filename):
    # twcs-AmazonHelp-146-outbound-predicted.xlsx → AmazonHelp
    return filename.split("-")[1]


brand_stats = {}
brand_order = {}   # ✅ 新增：记录每个 brand 的 header 顺序（按文件列出现顺序）

# ========== 1. 扫描 predicted 文件 ==========
for file in os.listdir(PREDICTED_DIR):
    if not file.endswith(".xlsx"):
        continue

    brand = parse_brand(file)
    path = os.path.join(PREDICTED_DIR, file)

    df = pd.read_excel(path)

    label_cols = [c for c in df.columns if not c.startswith("pred_") and c not in IGNORE_COLS]
    # ✅ 只在该 brand 第一次出现时记录顺序（用该文件的列顺序）
    brand_order.setdefault(brand, label_cols)

    for header in label_cols:
        pred_candidates = [c for c in df.columns if c.startswith(f"pred_{header}_")]
        if not pred_candidates:
            continue

        pred_col = pred_candidates[0]

        y_true = df[header].astype(int)
        y_pred = df[pred_col].astype(int)

        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()

        brand_stats.setdefault(brand, {})
        brand_stats[brand].setdefault(header, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

        brand_stats[brand][header]["TP"] += TP
        brand_stats[brand][header]["FP"] += FP
        brand_stats[brand][header]["FN"] += FN
        brand_stats[brand][header]["TN"] += TN

# ========== 2. 计算指标并导出 ==========
for brand, headers in brand_stats.items():
    rows = []
    total_items = 0

    for header, cm in headers.items():
        TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
        items = TP + FN
        total_items += items

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        rows.append({
            "Header": header,
            "F1": f1,
            "Precision": precision,
            "Accuracy": accuracy,
            "Recall": recall,
            "Items": items
        })

    # Macro Average
    if rows:
        macro = {
            "Header": "Macro Average",
            "F1": sum(r["F1"] for r in rows) / len(rows),
            "Precision": sum(r["Precision"] for r in rows) / len(rows),
            "Accuracy": sum(r["Accuracy"] for r in rows) / len(rows),
            "Recall": sum(r["Recall"] for r in rows) / len(rows),
            "Items": total_items
        }
        rows.append(macro)

    df_out = pd.DataFrame(rows)

    # 取出宏平均（copy() 避免 SettingWithCopyWarning）
    macro_row = df_out.loc[df_out["Header"] == "Macro Average"].copy()
    df_main = df_out.loc[df_out["Header"] != "Macro Average"].copy()

    # 按文件中 label_cols 的顺序排序
    order = brand_order.get(brand, [])
    rank = {h: i for i, h in enumerate(order)}

    df_main["_rank"] = df_main["Header"].map(rank).fillna(10**9)
    df_main = df_main.sort_values("_rank").drop(columns=["_rank"])

    # 宏平均放最后
    df_out = pd.concat([df_main, macro_row], ignore_index=True)

    # ✅ 生成并保存文件（你之前漏掉的部分）
    out_path = os.path.join(OUTPUT_DIR, f"{brand}.xlsx")
    save_formatted_excel(df_out, out_path)

print("\n🎉 Finished: 3 brand-level metric files generated!")

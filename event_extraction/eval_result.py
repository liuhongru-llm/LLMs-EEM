import os
import math
import numpy as np
import pandas as pd

np.random.seed(1868)


def calc_mcc(TP, FP, FN, TN):
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return (TP * TN - FP * FN) / denom if denom != 0 else 0.0


def calc_balanced_accuracy(TP, FP, FN, TN):
    tpr = TP / (TP + FN) if (TP + FN) else 0.0
    tnr = TN / (TN + FP) if (TN + FP) else 0.0
    return (tpr + tnr) / 2


def evaluate_predictions_simplified(y_true, y_pred):
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    mcc = calc_mcc(TP, FP, FN, TN)
    balanced_acc = calc_balanced_accuracy(TP, FP, FN, TN)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    items = int(y_true.sum())

    return mcc, balanced_acc, f1, precision, recall, items


def process_single_file(df):
    results = []

    pred_columns = [col for col in df.columns if col.startswith('pred_')]

    for pred_col in pred_columns:
        label_col = pred_col.replace('pred_', '')

        if label_col not in df.columns:
            continue

        val_true = pd.to_numeric(df[label_col], errors='coerce').fillna(0)
        val_pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0)

        y_true = (val_true > 0).astype(int)
        y_pred = (val_pred > 0).astype(int)

        if len(y_true) > 0:
            mcc, balanced_acc, f1, precision, recall, items = evaluate_predictions_simplified(y_true, y_pred)

            results.append({
                'Header': label_col,
                'Label': label_col,
                'MCC': round(mcc, 4),
                'Balanced_Accuracy': round(balanced_acc, 4),
                'F1': round(f1, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'Items': items
            })

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        metric_cols = ['MCC', 'Balanced_Accuracy', 'F1', 'Precision', 'Recall']
        mean_values = df_results[metric_cols].mean()

        macro_row = {
            'Header': 'Macro Average',
            'Label': 'ALL',
            'MCC': round(mean_values['MCC'], 4),
            'Balanced_Accuracy': round(mean_values['Balanced_Accuracy'], 4),
            'F1': round(mean_values['F1'], 4),
            'Precision': round(mean_values['Precision'], 4),
            'Recall': round(mean_values['Recall'], 4),
            'Items': int(df_results['Items'].sum())
        }

        df_results = pd.concat([df_results, pd.DataFrame([macro_row])], ignore_index=True)

    return df_results


def save_formatted_excel(df_results, out_path):
    try:
        writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        num_rows = len(df_results) + 1

        # MCC: [-1, 1], mid at 0
        mcc_format = {
            'type': '3_color_scale',
            'min_color': '#FF0000',
            'mid_color': '#FFFF00',
            'max_color': '#00FF00',
            'min_type': 'num',
            'mid_type': 'num',
            'max_type': 'num',
            'min_value': -1,
            'mid_value': 0,
            'max_value': 1,
        }

        # Balanced Accuracy / F1 / Precision / Recall: [0, 1]
        zero_one_format = {
            'type': '3_color_scale',
            'min_color': '#FF0000',
            'mid_color': '#FFFF00',
            'max_color': '#00FF00',
            'min_type': 'num',
            'mid_type': 'percentile',
            'max_type': 'num',
            'min_value': 0,
            'max_value': 1,
            'mid_value': 50,
        }

        # Columns: A=Header, B=Label, C=MCC, D=Balanced_Accuracy, E=F1, F=Precision, G=Recall, H=Items
        worksheet.conditional_format(f'C2:C{num_rows}', mcc_format)
        worksheet.conditional_format(f'D2:G{num_rows}', zero_one_format)

        worksheet.set_column('A:B', 25)
        worksheet.set_column('C:G', 18)
        worksheet.set_column('H:H', 10)

        bold_format = workbook.add_format({'bold': True, 'bg_color': '#E0E0E0'})
        worksheet.set_row(num_rows - 1, None, bold_format)

        writer.close()
        print(f"  成功保存: {out_path}")

    except Exception as e:
        print(f"  保存格式化Excel失败，尝试普通保存: {e}")
        df_results.to_excel(out_path, index=False)


if __name__ == '__main__':
    input_root = 'WorkFlow_no_RAG'
    output_root = 'WorkFlow_no_RAG_Evaluation_Results'

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    if not os.path.exists(input_root):
        print(f"错误: 输入目录 {input_root} 不存在")
        exit()

    for item in os.listdir(input_root):
        model_dir_path = os.path.join(input_root, item)

        if not os.path.isdir(model_dir_path):
            continue

        print(f"正在处理目录: {item}")

        output_dir_path = os.path.join(output_root, item)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        for filename in os.listdir(model_dir_path):
            if not filename.endswith('.xlsx') or filename.startswith('~$'):
                continue

            file_path = os.path.join(model_dir_path, filename)

            try:
                df = pd.read_excel(file_path)
                df_results = process_single_file(df)

                if df_results.empty:
                    print(f"  [警告] 未找到有效的预测列对或结果为空: {filename}")
                    continue

                out_filename = f"eval_{filename}"
                out_path = os.path.join(output_dir_path, out_filename)

                save_formatted_excel(df_results, out_path)

            except Exception as e:
                print(f"  [错误] 处理文件 {filename} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
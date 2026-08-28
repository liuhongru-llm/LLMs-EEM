import requests
import json
import os
import pandas as pd
from collections import OrderedDict
import time
import sys

# ================= 配置参数 =================
# API 密钥通过环境变量提供，不要硬编码在代码中：
#   Windows PowerShell:  $env:DIFY_API_KEY = "app-你的密钥"
#   Linux/macOS:        export DIFY_API_KEY="app-你的密钥"
API_KEY = os.environ.get("DIFY_API_KEY", "")  # 从环境变量读取 Dify 应用密钥
ENDPOINT = "http://localhost/v1/workflows/run"
USER_ID = "user-abc-123"

# 定义各公司的 Events 分界点
SPLIT_POINTS = {
    "AmazonHelp": "Investigate Issue",
    "AppleSupport": "Recommend backing up",
    "SpotifyCares": "Request reinstall"
}


# 自定义异常类
class APIRequestError(Exception):
    """API请求异常"""
    pass


# ===========================================

def parse_filename(filename):
    """解析文件名: twcs-AmazonHelp-300.xlsx -> AmazonHelp, 300"""
    parts = filename.split('.')[0].split('-')
    if len(parts) >= 3:
        company = parts[1]
        count = parts[2]
        return company, count
    return "Unknown", "0"


def get_filtered_events(all_events, company, inbound_status):
    """
    根据公司和 inbound 状态筛选 events
    inbound=True: 取分界点之后的
    inbound=False: 取分界点之前的（含分界点）
    """
    split_point = SPLIT_POINTS.get(company)

    if not split_point or split_point not in all_events:
        return all_events

    idx = all_events.index(split_point)

    if inbound_status:
        return all_events[idx + 1:]
    else:
        return all_events[:idx + 1]


def predict(df, df_nli_template, df_keywords_template, nlp_cache, company_name, output_dir, count, model):
    df = df.fillna("")

    # 1. 预计算 Context (上下文)
    df['prev_text'] = df['text'].shift(1)
    df['prev_main_id'] = df['main_tweet_id'].shift(1)

    df['context'] = "无"
    mask = df['main_tweet_id'] == df['prev_main_id']
    df.loc[mask, 'context'] = df['prev_text']

    df.drop(columns=['prev_text', 'prev_main_id'], inplace=True)

    all_hypotheses = df_nli_template.columns.tolist()

    print(f"开始处理 {company_name} 的数据，共 {len(df)} 条...")

    def process_row(row):
        text = str(row['text'])
        context = str(row['context'])

        inbound_raw = row['inbound']
        if isinstance(inbound_raw, str):
            is_inbound = inbound_raw.strip().upper() == 'TRUE'
        else:
            is_inbound = bool(inbound_raw)

        doc_suffix = "-inbound-knowledge" if is_inbound else "-outbound-knowledge"
        document_name = f"{company_name}{doc_suffix}"

        current_events = get_filtered_events(all_hypotheses, company_name, is_inbound)

        if not current_events:
            return pd.Series({f'pred_{k}': 0 for k in all_hypotheses})

        scores = nlp(text, context, document_name, current_events, df_keywords_template, nlp_cache, company_name)

        full_result = OrderedDict()
        for h in all_hypotheses:
            key = 'pred_' + h
            full_result[key] = scores.get(key, 0)

        return pd.Series(full_result)

    prediction_results = df.apply(process_row, axis=1)
    df = pd.concat([df, prediction_results], axis=1)

    return df


def get_keywords_for_events(events, df_keywords_template):
    """
    根据 events 列表，从 keywords 模板中查找对应的关键词
    返回格式: "Event1:keyword1/keyword2,Event2:keyword3"
    """
    keywords_parts = []

    for event in events:
        if event in df_keywords_template.columns:
            keywords_in_column = []
            for idx in range(len(df_keywords_template)):
                keyword = df_keywords_template[event].iloc[idx]
                if pd.notna(keyword) and str(keyword).strip():
                    keywords_in_column.append(str(keyword).strip())

            if keywords_in_column:
                keywords_str = '/'.join(keywords_in_column)
                keywords_parts.append(f"{event}:{keywords_str}")

    return ','.join(keywords_parts) if keywords_parts else "无"


def nlp(text, context, document_name, candidate_labels, df_keywords_template, nlp_cache, company_name):
    result_dict = OrderedDict()

    unique_content_key = f"{text}__{document_name}"

    # 1. 检查缓存（缓存只存储分数值）
    uncached_labels = []
    for label in candidate_labels:
        cache_key = unique_content_key + '__' + label
        if cache_key in nlp_cache:
            result_dict[label] = nlp_cache[cache_key]
        else:
            uncached_labels.append(label)
            result_dict[label] = None

    # 2. 调用 API 处理未缓存的标签
    if uncached_labels:
        keywords = get_keywords_for_events(uncached_labels, df_keywords_template)

        try:
            classified = execute_work_flow(text, context, document_name, uncached_labels, keywords)
        except requests.exceptions.RequestException as e:
            print(f"❌ API请求失败: {e}")
            raise APIRequestError(e)
        except Exception as e:
            print(f"❌ 处理API响应时出错: {e}")
            raise APIRequestError(e)

        # 更新结果和缓存（只存分数）
        for key, value in zip(classified['labels'], classified['scores']):
            result_dict[key] = value
            cache_key = unique_content_key + '__' + key
            nlp_cache[cache_key] = value   # 直接存分数

        # 打印简要信息（不再输出 tokens/time）
        keywords_display = keywords if len(keywords) <= 100 else keywords[:97] + "..."
        print(f"API调用 | Doc: {document_name} | Events: {len(uncached_labels)} 个 | "
              f"Keywords: {keywords_display}")

    # 3. 格式化输出（不再附加 api_tokens / api_elapsed_time）
    final_output = OrderedDict(('pred_' + k, v if v is not None else 0) for k, v in result_dict.items())
    return final_output


def execute_work_flow(text, context, document_name, events, keywords):
    str_events = ','.join(events)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept-Charset": "UTF-8"
    }

    payload = {
        "inputs": {
            "text": text,
            "context": context,
            "document_name": document_name,
            "activities": str_events,
            "activity_triggers": keywords,
        },
        "response_mode": "streaming",
        "user": USER_ID
    }

    result_data = {
        "labels": events,
        "scores": [0] * len(events),
        "tokens": 0,
        "elapsed_time": 0
    }

    start_time = time.time()

    try:
        response = requests.post(
            url=ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith('data:'):
                    decoded_line = decoded_line[5:].strip()

                try:
                    if decoded_line:
                        data = json.loads(decoded_line)

                        if data.get("event") == "workflow_finished":
                            workflow_data = data.get("data", {})
                            outputs = workflow_data.get("outputs", {})

                            # 不再使用 tokens / elapsed_time
                            if outputs:
                                final_result = outputs.get("finalResult", "")
                                score_dict = {}

                                for result_line in final_result.strip().split('\n'):
                                    result_line = result_line.strip()
                                    if not result_line or ':' not in result_line:
                                        continue
                                    try:
                                        key, val = result_line.split(':', 1)
                                        key = key.strip()
                                        val = val.strip()
                                        if key in events:
                                            try:
                                                score_dict[key] = int(val)
                                            except ValueError:
                                                score_dict[key] = 0
                                    except Exception:
                                        pass

                                result_data["scores"] = [score_dict.get(label, 0) for label in events]

                            return result_data

                except json.JSONDecodeError:
                    print(f"JSON解析错误: {decoded_line}")
    except requests.exceptions.SSLError as e:
        print(f"❌ SSL连接错误: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求发生错误: {e}")
        raise

    return result_data


def save_partial_results(df_partial, output_dir, company, count, model, reason="中断"):
    """保存部分处理结果"""
    if 'context' in df_partial.columns:
        df_partial = df_partial.drop(columns=['context'])

    partial_filename = f'twcs-{company}-{count}-partial-{model}-{int(time.time())}.xlsx'
    partial_path = os.path.join(output_dir, partial_filename)

    df_partial.to_excel(partial_path, index=False)
    print(f"📝 已保存部分处理结果: {partial_path} ({reason})")
    return partial_path


if __name__ == "__main__":
    model = 'DeepSeekReasoner'
    output_base_dir = os.path.join('WorkFlow_no_RAG', model)

    input_dir = os.path.join('data', 'labeled')
    if not os.path.exists(input_dir):
        print(f"请确保 {input_dir} 目录存在并包含 Excel 文件")
        exit()

    os.makedirs(output_base_dir, exist_ok=True)

    # 遍历文件
    for filename in os.listdir(input_dir):
        if not filename.endswith('.xlsx'):
            continue

        try:
            company, count = parse_filename(filename)
            file_path = os.path.join(input_dir, filename)

            # 使用 JSON 缓存文件
            cache_path = os.path.join(output_base_dir, f'nlp_cache_{company}_{count}.json')

            # 加载缓存
            nlp_cache = {}
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        nlp_cache = json.load(f)
                    print(f"已加载缓存: {cache_path} (大小: {len(nlp_cache)}条记录)")
                except Exception as e:
                    print(f"缓存加载失败: {e}，重置缓存")
                    nlp_cache = {}

            # 读取数据
            df = pd.read_excel(file_path)

            # 查找模板文件
            nli_template_name = f'twcs-{company}-keywords.xlsx'
            template_path = os.path.join('data', 'keywords-template', nli_template_name)

            if not os.path.exists(template_path):
                print(f"⚠️ 模板文件未找到: {template_path}，跳过 {filename}")
                continue

            df_nli_template = pd.read_excel(template_path)
            df_keywords_template = pd.read_excel(template_path)

            file_start_time = time.time()

            # 执行预测
            try:
                df = predict(df, df_nli_template, df_keywords_template, nlp_cache, company, output_base_dir, count, model)
            except APIRequestError as e:
                print(f"🚫 检测到API请求错误，停止程序运行")
                print(f"💾 正在保存已处理的数据和缓存...")

                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(nlp_cache, f, ensure_ascii=False, indent=2)
                print(f"✅ 缓存已保存: {cache_path}")

                if 'pred_' in ''.join(df.columns.tolist()):
                    save_partial_results(df, output_base_dir, company, count, model, "API错误")

                sys.exit(1)

            file_elapsed_time = time.time() - file_start_time

            if 'context' in df.columns:
                df = df.drop(columns=['context'])

            outfile = f'twcs-{company}-{count}-predicted-{model}.xlsx'
            out_path = os.path.join(output_base_dir, outfile)
            df.to_excel(out_path, index=False)
            print(f"✅ 结果已保存: {out_path}")
            print(f"📊 文件处理耗时: {file_elapsed_time:.2f}s")

            # 保存 JSON 缓存
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(nlp_cache, f, ensure_ascii=False, indent=2)
            print(f"🔄 更新缓存完成\n")

        except KeyboardInterrupt:
            print(f"\n⏹️ 检测到用户中断，正在保存当前状态...")

            if 'nlp_cache' in locals() and len(nlp_cache) > 0:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(nlp_cache, f, ensure_ascii=False, indent=2)
                print(f"✅ 缓存已保存: {cache_path}")

            if 'df' in locals() and 'pred_' in ''.join(df.columns.tolist()):
                save_partial_results(df, output_base_dir, company, count, model, "用户中断")

            print(f"👋 程序已安全退出")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 处理文件 {filename} 失败: {str(e)}")
            import traceback
            traceback.print_exc()

            if 'nlp_cache' in locals() and len(nlp_cache) > 0:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(nlp_cache, f, ensure_ascii=False, indent=2)
                print(f"✅ 缓存已保存: {cache_path}")

            if 'df' in locals() and 'pred_' in ''.join(df.columns.tolist()):
                save_partial_results(df, output_base_dir, company, count, model, "程序异常")

    print("\n处理完成！")
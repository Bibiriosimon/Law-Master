import os
import json
import re
import time
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random

# ========== 0. 配置 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DeepSeek API 配置 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-4ba5df9144f14d5e95c86caf2fe5240d")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- 数据集生成配置 ---
LAW_LIST = [
    "中华人民共和国刑法", "中华人民共和国民法典", "公司法", "证券法", "海商法",
    "民事诉讼法", "刑事诉讼法", "民用航空法", "公安机关办理行政案件程序规定",
    "公安机关办理刑事案件程序规定", "人民检察院刑事诉訟规则",
    "最高人民法院关于适用《民事诉讼法》的解释", "最高人民法院关于适用《刑事诉讼法》的解释"
]
TOPICS_PER_LAW = 15
QUESTIONS_PER_TOPIC = 5

# --- 性能配置 ---
OUTPUT_FILE = "synthetic_query_rewriter_dataset_fast_1k.jsonl"
MAX_WORKERS = 8      # 并行处理的线程数
TOPIC_BATCH_SIZE = 5   # 每次API调用生成问题的批处理大小
QUESTION_BATCH_SIZE = 10 # 每次API调用转换问题的批处理大小

# 创建一个会话以复用连接
session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
})

# ========== 1. API 调用与解析函数 ==========

def call_api(system_message, user_prompt, temperature=0.5, use_json_mode=False):
    """通用的API调用函数"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": 4096, # 为批量任务提供充足空间
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    try:
        response = session.post(API_URL, json=payload, timeout=180) # 延长超时
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"API调用失败: {e}")
        return None

def safe_json_loads(json_string, expected_keys):
    """更安全地解析JSON，并验证其结构"""
    try:
        data = json.loads(json_string)
        if all(key in data for key in expected_keys):
            return data
        else:
            logging.warning(f"解析的JSON缺少键。需要: {expected_keys}, 得到: {data.keys()}")
            return None
    except json.JSONDecodeError:
        logging.warning(f"无法解析JSON字符串: {json_string[:200]}...")
        return None

# ========== 2. 并行&批量的数据生成工作流 ==========

def generate_topics_for_law(law_name):
    """阶段1: 为一部法律生成多个核心主题"""
    prompt = f"请针对《{law_name}》，构思出 {TOPICS_PER_LAW} 个普通民众最关心的核心法律主题或场景。每个主题一行，不要有任何多余的解释。"
    response = call_api("你是一位资深的法律专家和教育家。", prompt, temperature=0.7)
    if response:
        return [re.sub(r'^\d+\.\s*', '', line).strip() for line in response.split('\n') if line.strip()]
    return []

def batch_generate_questions(topic_batch_with_laws):
    """阶段2: 为一批主题批量生成口语化问题"""
    formatted_topics = "\n".join([f"{i+1}. {law}: {topic}" for i, (law, topic) in enumerate(topic_batch_with_laws)])
    prompt = f"""
请为以下 {len(topic_batch_with_laws)} 个法律主题，各自生成 {QUESTIONS_PER_TOPIC} 个普通人会问的口语化问题。
严格按JSON格式返回，键为 "results"，值为一个数组，数组每个元素是对应主题的问题列表（字符串数组）。

待处理主题：
{formatted_topics}
"""
    response = call_api("你是一位内容创作者，擅长模仿普通网民的口吻提问。", prompt, temperature=0.8, use_json_mode=True)
    if response:
        data = safe_json_loads(response, ["results"])
        if data and isinstance(data['results'], list) and len(data['results']) == len(topic_batch_with_laws):
            return [q for sublist in data['results'] for q in sublist if q and '?' in q]
    return []

def batch_create_training_samples(question_batch):
    """阶段3: 将一批问题批量转换为结构化的JSON输出"""
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(question_batch)])
    prompt = f"""
请分析以下 {len(question_batch)} 个用户的法律问题，并为每一个问题生成一个结构化的JSON对象。
严格按JSON格式返回，最外层键为 "results"，值为一个JSON数组，数组的每个元素是对应问题的结构化输出。
每个结构化输出对象必须包含两个键:
1. `keywords_for_search`: 包含3-5个核心法律术语的数组。
2. `query_for_vector_search`: 一个书面化的、概括性的查询字符串。

待处理的问题:
{formatted_questions}
"""
    response = call_api("你是一个顶级的法律查询分析引擎。", prompt, temperature=0.1, use_json_mode=True)
    samples = []
    if response:
        data = safe_json_loads(response, ["results"])
        if data and isinstance(data['results'], list) and len(data['results']) == len(question_batch):
            for i, res_obj in enumerate(data['results']):
                if isinstance(res_obj, dict) and 'keywords_for_search' in res_obj and 'query_for_vector_search' in res_obj:
                    samples.append({
                        "instruction": "你是一个法律查询助手。请分析用户的法律问题，并将其转换为结构化的JSON对象...",
                        "input": question_batch[i],
                        "output": json.dumps(res_obj, ensure_ascii=False)
                    })
    return samples

# ========== 3. 主执行流程 ==========

if __name__ == "__main__":
    logging.info("--- 开始高速生成高质量的查询重写微调数据集 ---")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # --- 阶段1: 并行生成所有主题 ---
        law_to_topics_futures = {executor.submit(generate_topics_for_law, law): law for law in LAW_LIST}
        topics_with_laws = []
        for future in tqdm(as_completed(law_to_topics_futures), total=len(LAW_LIST), desc="阶段1: 生成主题"):
            law = law_to_topics_futures[future]
            try:
                topics = future.result()
                topics_with_laws.extend([(law, topic) for topic in topics])
            except Exception as e:
                logging.error(f"为《{law}》生成主题失败: {e}")
        
        random.shuffle(topics_with_laws)
        logging.info(f"成功生成 {len(topics_with_laws)} 个主题，准备生成问题...")

        # --- 阶段2: 并行&批量生成所有问题 ---
        topic_batches = [topics_with_laws[i:i + TOPIC_BATCH_SIZE] for i in range(0, len(topics_with_laws), TOPIC_BATCH_SIZE)]
        batch_to_questions_futures = {executor.submit(batch_generate_questions, batch): batch for batch in topic_batches}
        all_questions = []
        for future in tqdm(as_completed(batch_to_questions_futures), total=len(topic_batches), desc="阶段2: 生成问题"):
            try:
                all_questions.extend(future.result())
            except Exception as e:
                logging.error(f"一个问题生成批次失败: {e}")
        
        random.shuffle(all_questions)
        logging.info(f"成功生成 {len(all_questions)} 个问题，准备转换为训练样本...")

        # --- 阶段3: 并行&批量转换问题为训练样本 ---
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            question_batches = [all_questions[i:i + QUESTION_BATCH_SIZE] for i in range(0, len(all_questions), QUESTION_BATCH_SIZE)]
            batch_to_samples_futures = {executor.submit(batch_create_training_samples, batch): batch for batch in question_batches}
            
            for future in tqdm(as_completed(batch_to_samples_futures), total=len(question_batches), desc="阶段3: 转换样本"):
                try:
                    samples = future.result()
                    for sample in samples:
                        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
                except Exception as e:
                    logging.error(f"一个样本转换批次失败: {e}")

    logging.info(f"\n🎉🎉🎉 高性能数据集生成完毕！🎉🎉🎉")
    logging.info(f"文件已保存至: {OUTPUT_FILE}")

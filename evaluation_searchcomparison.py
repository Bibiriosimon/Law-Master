import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import time
from datetime import datetime

# --- 导入我们之前编写的检索器 ---
# 确保 retriever.py 和此脚本在同一个文件夹下
from retriever import HybridRetriever

# ========== 1. 配置 ==========
console = Console()

# --- 模型和文件路径 (‼️请务必根据您的实际路径修改‼️) ---
BASE_MODEL_PATH = '/root/autodl-tmp/legal_finetune/deepseek'
SLM_ADAPTER_PATH = './output_query_rewriter_lora/final_model'
EMBEDDING_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese'
FAISS_INDEX_PATH = 'law_enhanced_vector_db.faiss'
CHUNK_MAP_PATH = 'index_to_chunk_map.json'

# --- DeepSeek API 配置 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-4ba5df9144f14d5e95c86caf2fe5240d") # 请替换
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- 报告输出配置 ---
REPORT_FILE = "final_large_scale_test_report.md"

# ========== 2. 核心功能函数 ==========

# --- 用于稳定SLM输出的停止规则 ---
class StopOnBrace(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids('}')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False

def generate_test_queries(num_queries=100): # ✅ 增加到100组数据
    """使用DeepSeek API生成测试问题"""
    console.print(f"[bold cyan]正在调用DeepSeek API生成{num_queries}个测试问题...[/bold cyan]")
    prompt = f"""
请你扮演一名普通中国民众，生成 {num_queries} 个你会在现实生活中遇到的、多样化的法律问题。
要求：
- 问题必须口语化、自然，就像在搜索引擎里提问一样。
- 覆盖不同领域，如民事纠纷、刑事问题、劳动争议、交通意外、公司法务等。
- 有些问题可以写得模糊一些，包含日常用语，例如“动手了”、“老板不给钱”、“车给蹭了”。
- 每个问题一行，以数字加点号开头 (e.g., 1. ...)。
"""
    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}, timeout=180) # 延长超时
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return [re.sub(r'^\d+\.\s*', '', line).strip() for line in content.split('\n') if line.strip()]
    except Exception as e:
        console.print(f"[bold red]生成测试问题失败: {e}[/bold red]")
        return ["我邻居半夜唱歌，报警有用吗？", "老板拖欠工资不给，我该怎么办？", "开车不小心蹭了别人的车，私了还是报警好？"]

def load_query_rewriter_slm(base_model_path, adapter_path):
    """加载微调后的查询重写SLM"""
    console.print(f"[cyan]正在从 {base_model_path} 加载基础模型...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", trust_remote_code=True, device_map="auto")
    console.print(f"[cyan]正在从 {adapter_path} 加载LoRA适配器...[/cyan]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    console.print("[bold green]✅ 查询重写SLM加载成功！[/bold green]")
    return model, tokenizer

def rewrite_query_with_slm(query, model, tokenizer, retries=3):
    """使用SLM重写查询，并集成“打破循环”的稳定策略"""
    prompt_template = ("你是一个法律查询助手。请分析用户的法律问题，并将其转换为结构化的JSON对象，包含用于关键词搜索的'keywords_for_search'和用于向量搜索的'query_for_vector_search'。\n\n### 用户问题:\n{input}\n\n### JSON输出:\n")
    prompt = prompt_template.format(input=query)
    stopping_criteria = StoppingCriteriaList([StopOnBrace(tokenizer)])
    
    for attempt in range(retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    repetition_penalty=1.2,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95
                )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_match = re.search(r'\{[\s\S]*?\}', response_text)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                if "keywords_for_search" in parsed_json and "query_for_vector_search" in parsed_json:
                    return parsed_json
        except Exception as e:
            console.print(f"[yellow]警告: 第 {attempt + 1} 次查询重写尝试失败: {e}[/yellow]")
            time.sleep(1)
    console.print(f"[bold red]查询重写在 {retries} 次尝试后仍失败。[/bold red]")
    return None

def extract_keywords_from_raw_query(query):
    """使用API从原始问题中提取关键词"""
    prompt = f"""请从以下用户的法律问题中，提取出3到5个最核心的关键词。要求：以JSON格式返回，键为 "keywords"，值为一个字符串列表。\n\n用户问题：\n"{query}" """
    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}, timeout=30)
        response.raise_for_status()
        data = response.json()['choices'][0]['message']['content']
        return json.loads(data).get("keywords", [])
    except Exception as e:
        console.print(f"[yellow]警告: 从原始问题提取关键词失败: {e}[/yellow]")
        return []

def generate_ground_truth(query, retriever):
    """✅ 使用LLM为问题生成扩大版的标准答案"""
    console.print(f"   [cyan]正在为问题 '{query[:30]}...' 生成标准答案...[/cyan]")
    # 1. ✅ 初步召回扩大一倍的候选集
    candidate_ids = retriever.search(query, extract_keywords_from_raw_query(query), top_k=40)
    context = retriever.assemble_context(candidate_ids)
    
    # 2. ✅ 调用API请求专家判断，要求挑选10个
    prompt = f"""
你是一名资深的中国法官。你的任务是根据用户的问题和下面提供的一系列候选法律条文，选出与问题最直接相关的10个法条。

**用户问题:**
{query}

**候选法律条文:**
{context}

**任务要求:**
请严格按照JSON格式返回，最外层键为 "top_10_articles"，值为一个JSON数组，包含你选出的10个最相关法条的完整名称。如果相关内容不足10个，返回所有相关的即可。
"""
    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "messages": [{"role": "system", "content": "你是一名资深的中国法官。"}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}, timeout=180) # 延长超时
        response.raise_for_status()
        data = response.json()['choices'][0]['message']['content']
        return json.loads(data).get("top_10_articles", [])
    except Exception as e:
        console.print(f"[yellow]警告: 生成标准答案失败: {e}[/yellow]")
        return []

def calculate_hits(retrieved_ids, ground_truth_ids, chunk_map):
    """计算命中数"""
    retrieved_articles = {chunk_map[id]['article_number'] for id in retrieved_ids if id in chunk_map}
    ground_truth_set = set(ground_truth_ids)
    return len(retrieved_articles.intersection(ground_truth_set))

def generate_report(results_log):
    """生成Markdown格式的实验报告"""
    console.print("[bold blue]正在生成实验报告...[/bold blue]")
    
    total_hits = {"strategy_1": 0, "strategy_2": 0, "strategy_3": 0, "strategy_4": 0, "strategy_5": 0}
    total_retrieved = {"strategy_1": 0, "strategy_2": 0, "strategy_3": 0, "strategy_4": 0, "strategy_5": 0}
    
    report_content = f"# RAG检索策略对比实验报告 (大规模测试)\n\n**测试时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_content += "## 详细案例分析\n\n"
    for result in results_log:
        report_content += f"### 问题: {result['query']}\n\n**标准答案 (LLM生成):**\n"
        for article in result['ground_truth']: report_content += f"- {article}\n"
        report_content += "\n| 策略 | 命中数 | 召回结果 |\n|:---|:---:|:---|\n"
        
        strategies = ["strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"]
        strategy_names = ["SLM+混合", "SLM+关键词", "SLM+向量", "原文+向量", "原文+关键词"]

        for s_key, s_name in zip(strategies, strategy_names):
            hits = result[s_key]['hits']
            retrieved_count = len(result[s_key]['retrieved_articles'])
            total_hits[s_key] += hits
            total_retrieved[s_key] += retrieved_count
            recalled_articles = "<br>".join([f"- {name}" for name in result[s_key]['retrieved_articles']]) if retrieved_count > 0 else "无"
            report_content += f"| **{s_name}** | {hits}/{retrieved_count} | {recalled_articles} |\n"
        report_content += "\n---\n\n"

    num_queries = len(results_log)
    total_possible_hits = sum(len(r['ground_truth']) for r in results_log)
    
    summary_content = "## 总体性能摘要\n\n"
    summary_content += f"本次测试共包含 **{num_queries}** 个问题，由LLM生成的标准答案总计 **{total_possible_hits}** 个法条。\n\n"
    summary_content += "| 策略 | 总命中数 | 总召回数 | **命中率 (命中数/召回数)** |\n|:---|:---:|:---:|:---:|\n"
    
    for s_key, s_name in zip(strategies, strategy_names):
        hit_rate = (total_hits[s_key] / total_retrieved[s_key]) * 100 if total_retrieved[s_key] > 0 else 0
        summary_content += f"| **{s_name}** | {total_hits[s_key]} | {total_retrieved[s_key]} | **{hit_rate:.2f}%** |\n"
        
    report_content = summary_content + report_content
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f: f.write(report_content)
    console.print(f"[bold green]✅ 实验报告已生成: {REPORT_FILE}[/bold green]")

# ========== 3. 主执行流程 ==========

if __name__ == "__main__":
    console.rule("[bold blue]初始化系统组件[/bold blue]")
    rewriter_model, rewriter_tokenizer = load_query_rewriter_slm(BASE_MODEL_PATH, SLM_ADAPTER_PATH)
    retriever = HybridRetriever(EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH, CHUNK_MAP_PATH)
    
    console.rule("[bold blue]生成测试数据[/bold blue]")
    test_queries = generate_test_queries()
    
    console.print(f"\n[bold magenta]生成的{len(test_queries)}个测试问题如下：[/bold magenta]")
    for i, q in enumerate(test_queries, 1): console.print(f"{i}. {q}")
    
    results_log = []
    
    console.rule("[bold blue]开始大规模对比测试[/bold blue]")
    for i, query in enumerate(test_queries, 1):
        console.print(Panel(f"[bold]正在测试问题 {i}/{len(test_queries)}: [italic]{query}[/italic][/bold]", border_style="bold green"))
        
        ground_truth_articles = generate_ground_truth(query, retriever)
        console.print(f"   [green]标准答案已生成: {ground_truth_articles}[/green]")

        rewritten_data = rewrite_query_with_slm(query, rewriter_model, rewriter_tokenizer)
        slm_keywords, slm_vector_query = (rewritten_data.get("keywords_for_search", []), rewritten_data.get("query_for_vector_search", query)) if rewritten_data else ([], query)
        
        strategies_results = {}
        # ... (执行五种策略的代码保持不变)
        ids = retriever.search(slm_vector_query, slm_keywords, top_k=3)
        strategies_results["strategy_1"] = {"ids": ids, "hits": calculate_hits(ids, ground_truth_articles, retriever.chunk_map)}
        ids = [doc_id for doc_id, score in retriever._keyword_search(slm_keywords, k=3)]
        strategies_results["strategy_2"] = {"ids": ids, "hits": calculate_hits(ids, ground_truth_articles, retriever.chunk_map)}
        ids = [doc_id for doc_id, score in retriever._vector_search(slm_vector_query, k=3)]
        strategies_results["strategy_3"] = {"ids": ids, "hits": calculate_hits(ids, ground_truth_articles, retriever.chunk_map)}
        ids = [doc_id for doc_id, score in retriever._vector_search(query, k=3)]
        strategies_results["strategy_4"] = {"ids": ids, "hits": calculate_hits(ids, ground_truth_articles, retriever.chunk_map)}
        raw_keywords = extract_keywords_from_raw_query(query)
        ids = [doc_id for doc_id, score in retriever._keyword_search(raw_keywords, k=3)]
        strategies_results["strategy_5"] = {"ids": ids, "hits": calculate_hits(ids, ground_truth_articles, retriever.chunk_map)}

        query_log = {"query": query, "ground_truth": ground_truth_articles}
        for name, data in strategies_results.items():
             data['retrieved_articles'] = [retriever.chunk_map[id]['article_number'] for id in data['ids'] if id in retriever.chunk_map]
             query_log[name] = data
        results_log.append(query_log)
        
        console.print(f"   [bold]命中数 (SLM+混合/SLM+关键词/SLM+向量/原文+向量/原文+关键词):[/bold] "
                      f"{strategies_results['strategy_1']['hits']}/{strategies_results['strategy_2']['hits']}/"
                      f"{strategies_results['strategy_3']['hits']}/{strategies_results['strategy_4']['hits']}/"
                      f"{strategies_results['strategy_5']['hits']}")
        console.print("\n" + "="*180 + "\n")
        
    console.rule("[bold blue]生成实验报告[/bold blue]")
    generate_report(results_log)
    console.rule("[bold green]所有测试完成！[/bold green]")


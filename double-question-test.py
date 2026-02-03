import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import subprocess
import uuid
import glob
import time
import numpy as np
import sys

# --- 分别导入两个检索器类 ---
from retriever import HybridRetriever
from news_retriever import NewsRetriever

# ========== 1. 配置 ==========
console = Console()
BASE_MODEL_PATH = '/root/autodl-tmp/legal_finetune/deepseek'
QUERY_REWRITER_SLM_PATH = './output_query_rewriter_lora/final_model'
GENERATION_SLM_ADAPTER_PATH = './output_deepseek_legal_lora_v2/final_model' 
EMBEDDING_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese'
FAISS_INDEX_PATH = 'law_enhanced_vector_db.faiss'
CHUNK_MAP_PATH = 'index_to_chunk_map.json'
SUMMARIZER_CWD = "./authoritative_summarizer" 
SUMMARIZER_TEMP_DIR = os.path.join(SUMMARIZER_CWD, "temp")
REPORT_FILE = "rag_comparison_report.md"

# ========== 2. 核心功能函数 ==========

class StopOnBrace(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids('}')
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False

def load_slm(base_model_path, adapter_path):
    """加载带LoRA适配器的小语言模型"""
    console.print(f"[cyan]正在加载 {os.path.basename(adapter_path)} ...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.float16, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print(f"[bold green]✅ 模型 {os.path.basename(adapter_path)} 加载成功！[/bold green]")
    return model.eval(), tokenizer

def rewrite_query_with_slm(prompt, model, tokenizer):
    """使用SML模型对用户问题进行重写，输出结构化JSON"""
    stopping_criteria = StoppingCriteriaList([StopOnBrace(tokenizer)])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.2,
            top_p=0.95
        )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_match = re.search(r'\{[\s\S]*?\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            console.print("[red]⚠️ JSON解析失败，模型输出非标准结构。[/red]")
            console.print(response_text)
            return None
    console.print("[yellow]⚠️ 模型未生成JSON结构，返回None。[/yellow]")
    console.print(response_text)
    return None

def generate_final_answer(query, context, model, tokenizer):
    """基于法条和新闻背景生成最终回答"""
    system_prompt = "你是一名专业的AI助手，任务是综合'参考法条'和'相关新闻/事件背景'，为'用户问题'生成一个全面、客观、条理清晰的回答。禁止在回答中使用任何英文单词。"
    user_prompt = f"{context}\n\n用户问题：\n{query}"
    final_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.1, do_sample=True, temperature=0.7, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text.split("<|im_start|>assistant")[-1].strip()

def combine_contexts(legal_context, news_context):
    """将新闻和法条内容整合为统一上下文"""
    combined = ""
    if news_context:
        combined += news_context + "\n\n"
    if legal_context and legal_context != "未找到相关法律条文。":
        combined += legal_context
    return combined.strip()

def run_summarizer_local(query):
    """调用本地新闻摘要器"""
    console.print(f"[cyan]调用摘要器，检索相关新闻: '{query}'[/cyan]")
    try:
        start_time = time.time()
        process = subprocess.run(
            [sys.executable, "-m", "app.main", query, "--mode", "local"],
            cwd=SUMMARIZER_CWD, capture_output=True, text=True, check=True, timeout=120
        )
        time.sleep(1)
        subdirs = [os.path.join(SUMMARIZER_TEMP_DIR, d) for d in os.listdir(SUMMARIZER_TEMP_DIR) if os.path.isdir(os.path.join(SUMMARIZER_TEMP_DIR, d))]
        new_dirs = [d for d in subdirs if os.path.getctime(d) > start_time]
        if not new_dirs:
            console.print("[yellow]⚠️ 未找到摘要输出。[/yellow]")
            return None
        latest = max(new_dirs, key=os.path.getctime)
        console.print(f"[green]✅ 摘要结果目录: {latest}[/green]")
        return latest
    except Exception as e:
        console.print(f"[red]摘要器调用错误: {e}[/red]")
        return None

def parse_summarizer_output(output_dir):
    """解析摘要结果目录"""
    data_list = []
    if not output_dir or not os.path.isdir(output_dir):
        return data_list
    for file in glob.glob(os.path.join(output_dir, "*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("url", "未知来源")
            events = data.get("event_paragraphs", [])
            disps = data.get("disposition_paragraphs", [])
            text = "\n\n".join(events + disps)
            if text.strip():
                data_list.append({"source": src, "content": text})
        except Exception as e:
            console.print(f"[yellow]解析失败 {file}: {e}[/yellow]")
    return data_list

def simulate_manual_review(summaries):
    """简易人工审核模拟"""
    if not summaries:
        console.print("[yellow]未检测到任何可用新闻片段。[/yellow]")
        return []
    for i, s in enumerate(summaries):
        console.print(f"[blue]来源 {i+1}: {s.get('source')}[/blue]")
        console.print(f"[italic]{s.get('content')[:200]}...[/italic]")
    return summaries  # 默认直接导入

# ========== 3. 主流程 ==========
if __name__ == "__main__":
    console.rule("[bold blue]启动法律双知识库RAG系统[/bold blue]")

    try:
        rewriter_model, rewriter_tokenizer = load_slm(BASE_MODEL_PATH, QUERY_REWRITER_SLM_PATH)
        generator_model, generator_tokenizer = load_slm(BASE_MODEL_PATH, GENERATION_SLM_ADAPTER_PATH)
        legal_retriever = HybridRetriever(EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH, CHUNK_MAP_PATH)
        news_retriever = NewsRetriever(
            embedding_model=legal_retriever.embedding_model,
            tokenizer=legal_retriever.tokenizer,
            device=legal_retriever.device,
            dimension=legal_retriever.embedding_model.config.hidden_size
        )
    except Exception as e:
        console.print(f"[bold red]初始化失败: {e}[/bold red]")
        sys.exit(1)

    test_cases = [
        {"user_query": "你对开封夜骑事件怎么看待？", "summarizer_query": "开封夜骑事件"},
        {"user_query": "你对武汉大学图书馆事件怎么看待？", "summarizer_query": "武汉大学图书馆事件"},
    ]

    for i, case in enumerate(test_cases, 1):
        console.rule(f"[bold yellow]案例 #{i}[/bold yellow]")
        query = case["user_query"]
        summarizer_query = case["summarizer_query"]

        # ---- Step 1. 基线流程 ----
        console.print(Panel("执行基线流程 (仅法律知识库)", title="[cyan]基线RAG[/cyan]"))
        baseline_data = rewrite_query_with_slm(query, rewriter_model, rewriter_tokenizer)
        if not baseline_data:
            console.print("[yellow]⚠️ 查询重写失败，使用原始问题。[/yellow]")
            keywords, vector_query = [], query
        else:
            keywords = baseline_data.get("keywords_for_search", [])
            vector_query = baseline_data.get("query_for_vector_search", query)

        law_ids = legal_retriever.search(vector_query, keywords, top_k=3)
        law_context = legal_retriever.assemble_context(law_ids)
        base_answer = generate_final_answer(query, law_context, generator_model, generator_tokenizer)
        console.print(Panel(Markdown(base_answer), title="基线回答", border_style="green"))

        # ---- Step 2. 联网流程 ----
        console.print(Panel("执行增强流程 (加入新闻知识库)", title="[cyan]增强RAG[/cyan]"))
        output_dir = run_summarizer_local(summarizer_query)
        if output_dir:
            summaries = parse_summarizer_output(output_dir)
            chunks = simulate_manual_review(summaries)
            if chunks:
                news_retriever.add_documents(chunks)

        news_docs = news_retriever.search(query, top_k=2)
        news_context = news_retriever.assemble_context(news_docs)

        contextual_prompt = f"请结合以下新闻背景与用户问题，生成一个JSON结构用于法条检索：\n\n新闻背景：{news_context}\n\n用户问题：{query}\n\nJSON输出："
        final_data = rewrite_query_with_slm(contextual_prompt, rewriter_model, rewriter_tokenizer)
        if not final_data:
            console.print("[yellow]⚠️ 上下文重写失败，使用原始问题。[/yellow]")
            keywords, vector_query = [], query
        else:
            keywords = final_data.get("keywords_for_search", [])
            vector_query = final_data.get("query_for_vector_search", query)

        law_ids = legal_retriever.search(vector_query, keywords, top_k=3)
        law_context = legal_retriever.assemble_context(law_ids)
        full_context = combine_contexts(law_context, news_context)
        final_answer = generate_final_answer(query, full_context, generator_model, generator_tokenizer)

        console.print(Panel(Markdown(final_answer), title="[magenta]最终增强回答[/magenta]", border_style="magenta"))
        console.print(Panel(news_context if news_context else "无新闻背景", title="新闻背景", border_style="yellow"))
        console.print(Panel(law_context if law_context else "无法条内容", title="参考法条", border_style="yellow"))

    console.rule("[bold green]✅ 所有案例执行完毕！[/bold green]")

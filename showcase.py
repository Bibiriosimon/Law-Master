import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import requests
from datetime import datetime

# --- 导入我们之前编写的检索器 ---
# 确保 retriever.py 和此脚本在同一个文件夹下
from retriever import HybridRetriever

# ========== 1. 配置 ==========
console = Console()

# --- 模型和文件路径 (‼️请务必根据您的实际路径修改‼️) ---
BASE_MODEL_PATH = '/root/autodl-tmp/legal_finetune/deepseek'
# 路径1: 查询重写模型
QUERY_REWRITER_SLM_PATH = './output_query_rewriter_lora/final_model'
# 路径2: 最终答案生成模型 (您第一个训练的模型)
GENERATION_SLM_ADAPTER_PATH = './output_deepseek_legal_lora_v2/final_model' 

# --- RAG组件路径 ---
EMBEDDING_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese'
FAISS_INDEX_PATH = 'law_enhanced_vector_db.faiss'
CHUNK_MAP_PATH = 'index_to_chunk_map.json'

# --- API 和报告配置 ---
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-4ba5df9144f14d5e95c86caf2fe5240d") 
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
REPORT_FILE = "upgrated_prompt_outcome.md"


# ========== 2. 核心功能函数 ==========

# --- 用于稳定SLM输出的停止规则 ---
class StopOnBrace(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids('}')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 在generate循环中，input_ids总是一个有效的Tensor，所以直接检查最后一个token即可。
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False

def load_slm(base_model_path, adapter_path):
    """通用SLM加载函数"""
    console.print(f"[cyan]正在从 {adapter_path} 加载LoRA适配器...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    console.print(f"[bold green]✅ 模型 {os.path.basename(adapter_path)} 加载成功！[/bold green]")
    return model.eval(), tokenizer

def generate_test_queries(num_queries=20):
    """使用DeepSeek API生成测试问题"""
    console.print(f"[bold cyan]正在调用DeepSeek API生成{num_queries}个测试问题...[/bold cyan]")
    prompt = f"请你扮演一名普通中国民众，生成 {num_queries} 个你会在现实生活中遇到的、多样化的法律问题。要求：问题必须口语化、自然；覆盖不同领域；有些问题可以写得模糊一些；每个问题一行，以数字加点号开头。"
    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}, timeout=90)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        queries = [re.sub(r'^\d+\.\s*', '', line).strip() for line in content.split('\n') if line.strip()]
        console.print(f"[bold green]✅ 成功生成 {len(queries)} 个测试问题。[/bold green]")
        return queries
    except Exception as e:
        console.print(f"[bold red]生成测试问题失败: {e}[/bold red]")
        return ["我邻居半夜唱歌，报警有用吗？", "老板拖欠工资不给，我该怎么办？"]

def rewrite_query_with_slm(query, model, tokenizer):
    """使用查询重写SLM将用户问题转换为结构化JSON"""
    # ✅ --- 优化后的Prompt 1 ---
    prompt_template = (
        "你是一个顶级的法律查询分析引擎。你的任务是接收用户的口语化问题，并将其严格转换为一个结构化的JSON对象，用于后续的混合检索系统。\n\n"
        "### 任务要求:\n"
        "1. **分析用户问题**: 深入理解用户的核心法律诉求。\n"
        "2. **提取关键词**: 在`keywords_for_search`字段中，提炼出3-5个最核心的法律术语或实体名词，例如“劳动合同”、“交通事故责任”、“继承权”等。关键词应简明扼要。\n"
        "3. **改写向量查询**: 在`query_for_vector_search`字段中，将原问题改写成一个更加书面化、概括性的查询语句，用于向量语义搜索。该语句应捕捉问题的本质，但去除口语化表达。\n"
        "4. **严格遵循格式**: 最终输出必须是一个纯粹的、不含任何解释性文字的JSON对象。\n\n"

    )
    prompt = prompt_template.format(input=query)
    stopping_criteria = StoppingCriteriaList([StopOnBrace(tokenizer)])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            repetition_penalty=1.2, do_sample=True, temperature=0.1, top_p=0.95
        )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_match = re.search(r'\{[\s\S]*?\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def generate_final_answer(query, context, model, tokenizer):
    """使用主生成模型，根据上下文和问题生成最终答案"""
    # ✅ --- 优化后的Prompt 2 ---
    system_prompt = (
        "你是一名顶级的中国法律AI专家。你的核心任务是严格、仅根据下面提供的'参考法条'，为'用户问题'生成一个专业、严谨,详细并且通俗易懂的回答。\n\n"
        "### 回答规则:\n"
        "1. **结构清晰**: 你的回答应该条理清晰并且详细，建议分点阐述，例如：首先直接回答核心问题，然后解释法律依据，最后给出建议。\n"
        "2. **语言要求**: 使用清晰、准确的简体中文。禁止在回答中使用任何英文单词或向用户反问问题,禁止出现英文单词。\n"
        "3. **保持客观**: 作为一个AI助手，保持中立和客观，不要提供投机性建议。"
    )
    user_prompt = f"参考法条：\n{context}\n\n用户问题：\n{query}"
    
    final_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response_text.split("<|im_start|>assistant")[-1].strip()
    return assistant_response, final_prompt

# ========== 3. 主执行流程 ==========

def run_rag_qa_pipeline(query, rewriter_model, rewriter_tokenizer, generator_model, generator_tokenizer, retriever, report_handle):
    """执行完整的RAG问答流程，并将结果写入报告"""
    
    # --- 步骤1: 查询重写 ---
    console.print(Panel(f"[bold]自然语言输入:[/bold]\n[italic]{query}[/italic]", title="[cyan]步骤1: 查询重写[/cyan]", border_style="cyan"))
    report_handle.write(f"### 1. 自然语言输入\n\n> {query}\n\n")
    
    rewritten_data = rewrite_query_with_slm(query, rewriter_model, rewriter_tokenizer)
    
    if not rewritten_data:
        console.print("[bold red]查询重写失败，无法继续流程。[/bold red]")
        report_handle.write("**查询重写失败，流程中止。**\n\n---\n\n")
        return
        
    console.print("[bold]结构化查询输出:[/bold]")
    console.print_json(data=rewritten_data)
    report_handle.write(f"### 2. 结构化查询输出\n\n```json\n{json.dumps(rewritten_data, indent=2, ensure_ascii=False)}\n```\n\n")
    
    # --- 步骤2: 混合检索 ---
    console.print(Panel("[bold]正在使用混合搜索从知识库中检索相关法条...[/bold]", title="[cyan]步骤2: 混合检索[/cyan]", border_style="cyan"))
    slm_keywords = rewritten_data.get("keywords_for_search", [])
    slm_vector_query = rewritten_data.get("query_for_vector_search", query)
    
    retrieved_ids = retriever.search(slm_vector_query, slm_keywords, top_k=5)
    context = retriever.assemble_context(retrieved_ids)
    report_handle.write(f"### 3. 参考法条\n\n```\n{context}\n```\n\n")

    # --- 步骤3: 答案生成 ---
    console.print(Panel("[bold]正在构建Prompt并调用主模型生成最终答案...[/bold]", title="[cyan]步骤3: 答案生成[/cyan]", border_style="cyan"))
    final_answer, final_prompt = generate_final_answer(query, context, generator_model, generator_tokenizer)
    report_handle.write(f"### 4. 模型输出\n\n{final_answer}\n\n---\n\n")

    # --- 步骤4: 结果展示 ---
    console.rule("[bold green]最终结果[/bold green]", style="bold green")
    console.print(Panel(Markdown(final_answer), title="[magenta]模型输出[/magenta]", border_style="magenta"))
    console.print(Panel(context, title="[yellow]参考法条[/yellow]", border_style="yellow"))

if __name__ == "__main__":
    console.rule("[bold blue]初始化RAG系统所有组件[/bold blue]")
    rewriter_model, rewriter_tokenizer = load_slm(BASE_MODEL_PATH, QUERY_REWRITER_SLM_PATH)
    generator_model, generator_tokenizer = load_slm(BASE_MODEL_PATH, GENERATION_SLM_ADAPTER_PATH)
    retriever = HybridRetriever(EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH, CHUNK_MAP_PATH)
    
    console.rule("[bold blue]开始处理样例输入并生成报告[/bold blue]")

    sample_queries = generate_test_queries(num_queries=20)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as report_f:
        report_f.write(f"# 端到端RAG问答流程演示报告\n\n**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, user_query in enumerate(sample_queries, 1):
            console.rule(f"[bold yellow]样例 #{i}/{len(sample_queries)}[/bold yellow]", style="bold yellow")
            report_f.write(f"## 样例 #{i}\n\n")
            run_rag_qa_pipeline(
                query=user_query,
                rewriter_model=rewriter_model,
                rewriter_tokenizer=rewriter_tokenizer,
                generator_model=generator_model,
                generator_tokenizer=generator_tokenizer,
                retriever=retriever,
                report_handle=report_f
            )
            console.print("\n\n")

    console.rule(f"[bold green]所有样例处理完毕！报告已生成于 {REPORT_FILE}[/bold green]")


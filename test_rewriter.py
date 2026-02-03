import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import re
import json
from rich.console import Console
from rich.panel import Panel

# ========== 1. 配置 ==========
console = Console()

# --- 模型和文件路径 (‼️请务必根据您的实际路径修改‼️) ---
BASE_MODEL_PATH = '/root/autodl-tmp/legal_finetune/deepseek'
SLM_ADAPTER_PATH = './output_query_rewriter_lora/final_model'

# --- 要测试的特定问题 ---
TEST_QUERY = "网上买东西被骗了，钱付了但卖家一直不发货，人也联系不上，该怎么办？"

# --- 停止规则 ---
class StopOnBrace(StoppingCriteria):
    """
    当模型生成右花括号 '}' 时停止生成的规则。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 获取右花括号 '}' 对应的token ID
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids('}')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最新生成的token是否是我们的停止token
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False

# ========== 2. 加载模型和分词器 ==========

def load_model_and_tokenizer(base_model_path, adapter_path):
    """加载微调后的查询重写SLM"""
    console.print(f"[cyan]正在从 {base_model_path} 加载基础模型...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto"
    )
    console.print(f"[cyan]正在从 {adapter_path} 加载LoRA适配器...[/cyan]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print("[bold green]✅ 模型和分词器加载成功！[/bold green]")
    return model.eval(), tokenizer

# ========== 3. 主执行流程 ==========

if __name__ == "__main__":
    console.rule("[bold blue]SLM输出诊断[/bold blue]")
    
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, SLM_ADAPTER_PATH)

    prompt_template = (
        "你是一个法律查询助手。请分析用户的法律问题，并将其转换为结构化的JSON对象，"
        "包含用于关键词搜索的'keywords_for_search'和用于向量搜索的'query_for_vector_search'。\n\n"
        "### 用户问题:\n{input}\n\n### JSON输出:\n"
    )
    prompt = prompt_template.format(input=TEST_QUERY)
    console.print(Panel(prompt, title="[yellow]发送给模型的完整Prompt[/yellow]", border_style="yellow"))

    stopping_criteria = StoppingCriteriaList([StopOnBrace(tokenizer)])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # ✅ --- 核心修改：加入repetition_penalty和采样参数来解决重复问题 ---
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            # --- 新增参数 ---
            repetition_penalty=1.2, # 对重复内容施加惩罚
            do_sample=True,         # 启用采样
            temperature=0.1,        # 使用极低的温度，确保结果的确定性
            top_p=0.95              # 配合采样使用
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    clean_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    console.print(Panel(raw_output, title="[red]原始模型输出 (应用新生成策略后)[/red]", border_style="red", highlight=True))
    
    # --- 步骤4: 尝试解析输出 ---
    console.rule("[bold blue]输出解析诊断[/bold blue]")
    
    response_part = clean_output.split("### JSON输出:")[-1].strip()
    console.print(f"[cyan]提取出的响应部分:[/cyan]\n[italic]{response_part}[/italic]\n")

    json_match = re.search(r'\{[\s\S]*?\}', response_part)
    
    if json_match:
        extracted_json_str = json_match.group(0)
        console.print(Panel(extracted_json_str, title="[green]成功提取到的JSON字符串[/green]", border_style="green"))
        try:
            parsed_json = json.loads(extracted_json_str)
            console.print("[bold green]✅ JSON解析成功！模型已学会任务格式。[/bold green]")
            console.print("解析后的Python字典内容:")
            console.print(parsed_json)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]❌ JSON解析失败！虽然找到了JSON结构，但格式无效。[/bold red]")
            console.print(f"错误信息: {e}")
    else:
        console.print("[bold red]❌ 未能在模型输出中找到有效的JSON结构！模型可能未完全学会任务格式。[/bold red]")


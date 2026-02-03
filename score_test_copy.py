import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from retriever import HybridRetriever

console = Console()

BASE_MODEL_PATH = '/root/autodl-tmp/legal_finetune/deepseek'
QUERY_REWRITER_SLM_PATH = './output_query_rewriter_lora/final_model'
GENERATION_SLM_ADAPTER_PATH = './output_deepseek_legal_lora_v2/final_model'
EMBEDDING_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese'
FAISS_INDEX_PATH = 'law_enhanced_vector_db.faiss'
CHUNK_MAP_PATH = 'index_to_chunk_map.json'
DATA_SPLITS = ('one_shot', 'zero_shot')
SELECTED_FILES = {
    'one_shot': ['3-8.json'],
    'zero_shot': ['3-8.json'],
}
ENTRY_LIMIT = -1



class StopOnBrace(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token_id = tokenizer.convert_tokens_to_ids('}')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return bool(len(input_ids) > 0 and input_ids[0][-1] == self.stop_token_id)


def load_slm(base_model_path: str, adapter_path: str):
    """Load a base causal language model together with its LoRA adapter."""
    console.print(f"[cyan]Loading adapter from {adapter_path}...[/cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print(f"[green]Adapter {os.path.basename(adapter_path)} loaded successfully.[/green]")
    return model.eval(), tokenizer


def rewrite_query_with_slm(query: str, model, tokenizer):
    """Use the query-rewrite SLM to convert the question into structured JSON."""
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

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response_text.split('<|im_start|>assistant')[-1].strip()
    parsed_json = {}

    if assistant_response:
        candidates = [assistant_response]
        json_match = re.search(r'\{[\s\S]*?\}', assistant_response)
        if json_match:
            candidates.append(json_match.group(0))
        for candidate in candidates:
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    parsed_json = loaded
                    break
            except json.JSONDecodeError:
                continue

    if not parsed_json:
        console.print('[bold yellow]Failed to parse query rewrite result as JSON; using the raw question directly.[/bold yellow]')

    return parsed_json, assistant_response


def generate_final_answer(query: str, context: str, model, tokenizer):
    """Generate the final answer using the main model."""
    system_prompt = (
        "你是一名顶级的中国法律AI专家。你的核心任务是根据“用户问题”，生成一个【结构与语言都完全符合标准范例】的专业、简明、准确、符合现行中国法律体系的回答。\n\n"
        "### 回答格式规范:\n"
        "请严格按照以下固定格式作答：\n"
        "回答:（直接说明问题的结论与处理建议，用简洁明白的语言解释）\n"
        "法律依据:（引用具体的法律条文或法规条款，并说明其与本案的关联）\n\n"
        "### 具体要求:\n"
        "1. **格式固定**：务必以“回答:”和“法律依据:”两个独立段落作答，中间不得插入其他标题或编号。\n"
        "2. **内容要求**：\n"
        "   - “回答”部分：应当直接切入结论，语气权威但通俗，避免使用推测性词汇（如“可能”“大概”）。\n"
        "   - “法律依据”部分：引用《中华人民共和国民法典》《刑法》《合同法》《道路交通安全法》等相关现行法律条款，并解释其适用逻辑。\n"
        "3. **语言要求**：\n"
        "   - 全文使用清晰、准确的简体中文。\n"
        "   - 禁止输出英文单词、符号或拼音。\n"
        "   - 不得反问用户问题。\n"
        "4. **逻辑要求**：\n"
        "   - 优先结合法律条款给出实质性结论。\n"
        "   - 保持中立、客观，不渲染情绪，不提供投机性建议。\n"
        "5. **风格要求**：\n"
        "   - 语气应正式、平实，句式参考以下标准示例：\n"
        "     例如：\n"
        "     “回答: 租赁合同甲方可以两个人签字，应符合法律规定。”\n"
        "     “法律依据: 《合同法》第四十四条规定，依法成立的合同，自成立时生效。”\n\n"
        "### 输出示例:\n"
        "回答: 房屋买卖合同在房产过户后生效，若卖方已履行交付义务，房屋所有权即合法转移。\n"
        "法律依据: 《民法典》第二百零九条规定，不动产物权的设立、变更、转让和消灭，经依法登记，发生效力。\n"

    )
    user_prompt = (
        "参考法条:\n"
        f"{context}\n\n"
        "用户问题:\n"
        f"{query}"
    )
    final_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(final_prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response_text.split('<|im_start|>assistant')[-1].strip()
    return assistant_response, final_prompt


def run_rag_qa_pipeline(
    query: str,
    rewriter_model,
    rewriter_tokenizer,
    generator_model,
    generator_tokenizer,
    retriever: HybridRetriever,
    verbose: bool = True,
):
    """Execute the full RAG pipeline for a single query and return the generated answer."""
    if verbose:
        message = f"[bold]User Input:[/bold]\n[italic]{query}[/italic]"
        console.print(
            Panel(
                message,
                title='[cyan]Step 1: Query Rewrite[/cyan]',
                border_style='cyan',
            )
        )

    rewritten_data, _ = rewrite_query_with_slm(query, rewriter_model, rewriter_tokenizer)

    if rewritten_data:
        keywords = rewritten_data.get('keywords_for_search', [])
        vector_query = rewritten_data.get('query_for_vector_search', query)
        if verbose:
            console.print('[bold]Structured query output[/bold]')
            console.print_json(data=rewritten_data)
    else:
        keywords = []
        vector_query = query
        if verbose:
            console.print('[bold yellow]Query rewrite did not return JSON output; falling back to the original question.[/bold yellow]')

    if verbose:
        console.print(
            Panel('[bold]Running hybrid retrieval...[/bold]', title='[cyan]Step 2: Retrieval[/cyan]', border_style='cyan')
        )
    retrieved_ids = retriever.search(vector_query, keywords, top_k=5)
    context = retriever.assemble_context(retrieved_ids)

    if verbose:
        console.print(
            Panel('[bold]Generating final answer...[/bold]', title='[cyan]Step 3: Answer Generation[/cyan]', border_style='cyan')
        )
    final_answer, _ = generate_final_answer(query, context, generator_model, generator_tokenizer)

    if verbose:
        console.rule('[bold green]Answer[/bold green]', style='bold green')
        console.print(Panel(Markdown(final_answer), title='[magenta]Model Output[/magenta]', border_style='magenta'))
        console.print(Panel(context, title='[yellow]Retrieved Context[/yellow]', border_style='yellow'))

    return final_answer


def build_query_from_entry(entry: dict) -> str:
    """Combine instruction and question into a single prompt."""
    parts = []
    for field in (entry.get('instruction'), entry.get('question')):
        if isinstance(field, str):
            cleaned = field.strip()
            if cleaned:
                parts.append(cleaned)
    return '\n\n'.join(parts)


def process_dataset_split(
    split_name: str,
    rewriter_model,
    rewriter_tokenizer,
    generator_model,
    generator_tokenizer,
    retriever: HybridRetriever,
):
    """Process only selected JSON files and limited entries."""
    target_files = SELECTED_FILES.get(split_name)
    if not target_files:
        console.print(f"[bold yellow]Skipping split '{split_name}': no files selected for testing.[/bold yellow]")
        return

    input_dir = os.path.join('data', split_name)
    output_dir = os.path.join('results', split_name)

    if not os.path.isdir(input_dir):
        console.print(f"[bold red]Input directory missing for split '{split_name}': {input_dir}[/bold red]")
        return

    os.makedirs(output_dir, exist_ok=True)

    for file_name in target_files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        if not os.path.isfile(input_path):
            console.print(f"[bold red]File {input_path} not found; skipping.[/bold red]")
            continue

        console.rule(f"[bold blue]Processing {split_name}/{file_name} (first {ENTRY_LIMIT} entries)[/bold blue]")

        try:
            with open(input_path, 'r', encoding='utf-8') as handle:
                dataset = json.load(handle)
        except Exception as exc:
            console.print(f"[bold red]Failed to read {input_path}: {exc}[/bold red]")
            continue

        if not isinstance(dataset, list):
            console.print(f"[bold red]{input_path} is not a JSON list. Skipping.[/bold red]")
            continue

        subset = dataset[:ENTRY_LIMIT]
        results = []

        for index, item in enumerate(subset, 1):
            prompt = build_query_from_entry(item)
            if not prompt:
                console.print(
                    f"[bold yellow]{input_path} entry #{index} lacks instruction/question text; writing empty answer.[/bold yellow]"
                )
                answer_text = ''
            else:
                answer_text = run_rag_qa_pipeline(
                    prompt,
                    rewriter_model,
                    rewriter_tokenizer,
                    generator_model,
                    generator_tokenizer,
                    retriever,
                    verbose=False,
                ) or ''

            updated_item = dict(item)
            updated_item['answer'] = answer_text
            results.append(updated_item)

        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        console.print(f"[bold green]Saved answers to {output_path}[/bold green]")



if __name__ == '__main__':
    console.rule('[bold blue]Initialising RAG components[/bold blue]')
    rewriter_model, rewriter_tokenizer = load_slm(BASE_MODEL_PATH, QUERY_REWRITER_SLM_PATH)
    generator_model, generator_tokenizer = load_slm(BASE_MODEL_PATH, GENERATION_SLM_ADAPTER_PATH)
    retriever = HybridRetriever(EMBEDDING_MODEL_PATH, FAISS_INDEX_PATH, CHUNK_MAP_PATH)

    console.rule('[bold blue]Starting dataset processing[/bold blue]')
    for split in DATA_SPLITS:
        process_dataset_split(split, rewriter_model, rewriter_tokenizer, generator_model, generator_tokenizer, retriever)

    console.rule('[bold green]Dataset processing completed[/bold green]')

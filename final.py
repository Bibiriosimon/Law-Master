import json
import time
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.nn import functional as F
# --- 1. 全局配置 (请根据您的环境修改这里的路径) ---

# --- 检索器组件路径 ---
INDEX_PATH = 'final_law_db.index'
KNOWLEDGE_BASE_PATH = 'merged_knowledge_base.json'
EMBEDDING_MODEL_PATH = './text2vec-base-chinese' 

# --- 大语言模型组件路径 ---
# !!! 核心：已根据您的信息更新为本地基础模型路径 !!!
BASE_MODEL_PATH = './deepseek' 
LORA_ADAPTER_PATH = './output_deepseek_legal_lora/checkpoint-1250'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 检索器 (Retriever) ---
class LawRetriever:
    def __init__(self, index_path, docs_path, embedding_model_path):
        print("开始加载检索器...")
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"✅ 知识库原文加载完成，共 {len(self.documents)} 条。")

        self.index = faiss.read_index(index_path)
        print("✅ FAISS索引加载完成。")

        from transformers import AutoTokenizer as EmbeddingTokenizer, AutoModel as EmbeddingModel
        from torch.nn import functional as F

        self.embedding_tokenizer = EmbeddingTokenizer.from_pretrained(embedding_model_path)
        self.embedding_model = EmbeddingModel.from_pretrained(embedding_model_path).to(DEVICE)
        self.embedding_model.eval()
        print("✅ 嵌入模型加载完成。")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        with torch.no_grad():
            encoded_input = self.embedding_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
            model_output = self.embedding_model(**encoded_input)
            query_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            query_embedding = F.normalize(query_embedding, p=2, dim=1).cpu().numpy()

        distances, indices = self.index.search(query_embedding, k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs

# --- 3. 加载应用了LoRA的语言模型 ---
def load_model_with_lora(base_model_path, lora_path):
    print(f"开始加载基础模型: {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ 基础模型加载完成。")
    
    print(f"开始加载并合并LoRA适配器: {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"✅ LoRA适配器加载并合并完成。")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    return model, tokenizer

# --- 4. 核心推理逻辑 ---
prompt_template_finding = """
# 角色
你是一名资深的中国法律专家，任务是起草一份专业的法律案件初步分析报告。

# 任务
严格根据下文提供的[相关法条]，结合[用户问题]，撰写一份结构化、逻辑清晰的“文书判决”。

# 要求
1.  **严格循证**: 你的每一项分析和结论，都必须明确引用[相关法条]中的具体原文作为依据，例如：“根据《工伤保险条例》第十四条的规定...”。
2.  **结构清晰**: 请按照“事实梳理”、“法律适用分析”和“初步结论”三个部分进行撰写。
3.  **语言专业**: 使用严谨、客观、正式的法律专业术语。
4.  **格式规范**: 你的输出应是完整的文书内容，不要包含任何思考过程、XML标签（如</think>）或其他无关字符。

# 输入信息
[相关法条]
{context}

[用户问题]
{query}

# 输出报告
[你的分析和文书判决]
"""
prompt_template_action = """
# 角色
你是一位充满人情味且经验丰富的法律援助顾问。你的沟通对象是一位可能正处于焦虑和困惑中的普通人。

# 任务
基于已有的[初步的文书判决]和[相关法条]，为用户提供一份温暖、清晰、充满鼓励的行动指南。

# 要求
1.  **语气和风格**: 你的语气必须是**平易近人、充满鼓励且有同理心的**。请像和朋友聊天一样，用大白话解释复杂的法律问题。**请将所有建议整合成一段或几段连贯的文字，绝对不要使用生硬的数字列表（如1、2、3...）。**
2.  **内容结构 (Chain of Thought)**:
    * **首先，安抚和共情**：用温暖的话语肯定用户维权的勇气，并用最通俗的语言解释核心法条的含义，让他/她知道“法律是站在你这边的”。
    * **其次，解读判决**：简单说明一下“文书判决”的结论对当事人意味着什么，给予其信心。
    * **最后，给出清晰路径**：将维权的步骤融合成一个流畅的行动路线图。要非常具体，例如：“第一步，你需要带着这些材料，去这个地方...”、“不用担心，法律规定了他们必须在多长时间内给你答复...”、“如果遇到这种情况，你可以接着这样做...”。
3.  **结尾鼓励**: 在最后，请再次给予用户力量和支持。

# 输入信息
[相关法条]
{context}

[用户问题]
{query}

[初步的文书判决]
{finding}

# 输出指南
[你的思考和行动措施]
"""

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    # 注意：1.5B的模型可能不需要特别长的max_new_tokens
    outputs = model.generate(**inputs, max_new_tokens=768, do_sample=True, top_p=0.9, temperature=0.6)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

def run_rag_chain(query, retriever, llm, llm_tokenizer):
    print("\n" + "="*50)
    print(f"收到用户问题: {query}")
    print("="*50)

    print("\n🔍 步骤一：检索相关法条...")
    retrieved_context = retriever.retrieve(query, k=5)
    context_str = "\n\n".join(retrieved_context)
    print("✅ 检索完成。")

    print("\n🧠 步骤二：生成“文书判决”...")
    prompt1 = prompt_template_finding.format(context=context_str, query=query)
    preliminary_finding = generate_response(llm, llm_tokenizer, prompt1)
    print("✅ “文书判决”已生成。")

    print("\n🚀 步骤三：生成“行动措施”...")
    prompt2 = prompt_template_action.format(context=context_str, query=query, finding=preliminary_finding)
    actionable_advice = generate_response(llm, llm_tokenizer, prompt2)
    print("✅ “行动措施”已生成。")
    
    return preliminary_finding, actionable_advice

# --- 5. 主程序入口 ---
if __name__ == "__main__":
    retriever = LawRetriever(INDEX_PATH, KNOWLEDGE_BASE_PATH, EMBEDDING_MODEL_PATH)
    llm, llm_tokenizer = load_model_with_lora(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    user_query = "我买了一个电脑，商家谎骗我发了虚假的显卡，但是我因为缺乏专业知识一直没有发现，7天后过了退货时间商家就拒绝退货了，我应该怎么要回我的钱？"
    
    finding, actions = run_rag_chain(user_query, retriever, llm, llm_tokenizer)

    print("\n\n" + "#"*80)
    print("                 最终法律咨询结果")
    print("#"*80)
    print("\n【第一部分：初步文书判决】")
    print("-------------------------")
    print(finding)
    print("\n【第二部分：具体行动措施】")
    print("-------------------------")
    print(actions)
    print("\n" + "#"*80)

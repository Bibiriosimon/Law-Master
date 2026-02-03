import json
import time
import faiss
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========== 1. 配置参数 ==========
# --- 输入文件 ---
# 使用我们上一步高速处理后生成的增强版语料库
ENRICHED_CORPUS_PATH = 'corpus_enriched_fast.jsonl' 

# --- 本地模型路径 ---
# 请确保这个路径是正确的
LOCAL_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese' 

# --- 输出文件 ---
INDEX_SAVE_PATH = 'law_enhanced_vector_db.faiss'
MAPPING_SAVE_PATH = 'index_to_chunk_map.json' # 这个映射文件对于检索至关重要

# --- 计算设备与批处理大小 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 可以根据您的GPU显存适当调整

# ========== 2. 辅助函数与数据加载 ==========

def mean_pooling(model_output, attention_mask):
    """
    平均池化 - 从Token Embeddings计算句子Embedding的标准方法。
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_enriched_corpus(file_path):
    """
    加载增强后的法律语料库 (.jsonl格式)。
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# ========== 3. 主执行流程 ==========

if __name__ == "__main__":
    print("--- 向量数据库构建流程启动 ---")

    # --- 步骤 1: 加载知识库 ---
    print(f"\n--- 步骤 1: 从 '{ENRICHED_CORPUS_PATH}' 加载增强后的知识库...")
    try:
        chunks = load_enriched_corpus(ENRICHED_CORPUS_PATH)
        print(f"✅ 成功加载了 {len(chunks)} 条文档。")
        if not chunks:
            raise ValueError("错误：未能加载任何文档。")
    except Exception as e:
        print(f"❌ 错误: 加载知识库失败: {e}"); exit()

    # --- 步骤 2: 准备用于向量化的文本 ---
    print("\n--- 步骤 2: 准备用于向量化的文本（合并内容与问题）...")
    texts_to_embed = []
    for chunk in chunks:
        questions_str = "\n".join(chunk.get("hypothetical_questions", []))
        combined_text = f"相关问题：\n{questions_str}\n\n法律条文：\n{chunk['content']}"
        texts_to_embed.append(combined_text)
    print("✅ 文本准备完成。示例如下:")
    print("="*25 + "\n" + texts_to_embed[0] + "\n" + "="*25)
    
    # --- 步骤 3: 从本地路径加载模型和分词器 ---
    print(f"\n--- 步骤 3: 从 '{LOCAL_MODEL_PATH}' 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
        model.eval()
        print(f"✅ 模型和分词器加载成功，将运行在: {DEVICE}")
    except Exception as e:
        print(f"❌ 错误: 加载模型失败，请检查路径。错误: {e}"); exit()

    # --- 步骤 4: 进行文本向量化 ---
    print(f"\n--- 步骤 4: 开始进行文本向量化 (共 {len(texts_to_embed)} 条)...")
    start_time = time.time()
    all_embeddings = []
    
    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="向量化进度"):
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]
        
        # 使用您的分词和编码逻辑
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # L2 归一化
        normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        all_embeddings.append(normalized_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype('float32')
    end_time = time.time()
    print(f"✅ 向量化完成，耗时 {end_time - start_time:.2f} 秒。")
    print(f"向量矩阵形状: {embeddings.shape}")

    # --- 步骤 5: 构建并保存FAISS索引 ---
    print("\n--- 步骤 5: 构建并保存FAISS索引...")
    try:
        d = embeddings.shape[1]
        # 使用 IndexFlatL2 进行精确的L2距离搜索
        index = faiss.IndexFlatL2(d)
        
        # 使用 IndexIDMap 将向量的顺序索引 (0, 1, 2, ...) 保存下来
        # 这使得我们可以通过向量的ID直接映射回原始数据
        ids = np.arange(len(chunks))
        index = faiss.IndexIDMap(index)
        index.add_with_ids(embeddings, ids)

        faiss.write_index(index, INDEX_SAVE_PATH)
        print(f"✅ FAISS索引构建完成，共包含 {index.ntotal} 个向量。")
        print(f"索引文件已保存至: '{INDEX_SAVE_PATH}'")
    except Exception as e:
        print(f"❌ 错误: 构建或保存FAISS索引失败: {e}")

    # --- 步骤 6: 创建并保存ID到数据块的映射文件 ---
    print(f"\n--- 步骤 6: 创建并保存索引ID到原始数据的映射文件...")
    try:
        # 这个映射关系是RAG检索召回后获取原文的关键
        index_to_chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        with open(MAPPING_SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(index_to_chunk_map, f, ensure_ascii=False, indent=4)
        print(f"✅ 映射文件已成功保存到: '{MAPPING_SAVE_PATH}'")
    except Exception as e:
        print(f"❌ 错误: 创建或保存映射文件失败: {e}")

    print(f"\n🎉🎉🎉 恭喜！向量数据库及映射文件全部创建成功！🎉🎉🎉")
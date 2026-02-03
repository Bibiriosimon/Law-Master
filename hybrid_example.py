import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import logging

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRetriever:
    """
    一个集成了向量搜索和关键词搜索的混合检索器。
    """
    def __init__(self, embedding_model_path, faiss_index_path, chunk_map_path):
        # --- 1. 初始化配置和设备 ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"检索器将在设备: {self.device} 上运行")

        # --- 2. 加载嵌入模型和分词器 ---
        logging.info(f"正在从 '{embedding_model_path}' 加载嵌入模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_path).to(self.device)
        self.embedding_model.eval()
        logging.info("✅ 嵌入模型加载成功。")

        # --- 3. 加载FAISS索引和数据映射 ---
        logging.info(f"正在加载FAISS索引: {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        logging.info(f"✅ FAISS索引加载成功，包含 {self.faiss_index.ntotal} 个向量。")
        
        logging.info(f"正在加载数据映射文件: {chunk_map_path}")
        with open(chunk_map_path, 'r', encoding='utf-8') as f:
            self.chunk_map = {int(k): v for k, v in json.load(f).items()}
        logging.info("✅ 数据映射文件加载成功。")

        # --- 4. 构建用于关键词搜索的倒排索引 ---
        logging.info("正在构建关键词倒排索引...")
        self.inverted_index = self._build_inverted_index()
        logging.info("✅ 倒排索引构建完成。")

    def _build_inverted_index(self):
        """
        为关键词搜索构建一个倒排索引，提高搜索效率。
        格式: {'关键词': [文档ID1, 文档ID2, ...]}
        """
        inverted_index = {}
        for doc_id, chunk_data in self.chunk_map.items():
            for keyword in chunk_data.get('keywords', []):
                if keyword not in inverted_index:
                    inverted_index[keyword] = []
                inverted_index[keyword].append(doc_id)
        return inverted_index

    def _mean_pooling(self, model_output, attention_mask):
        """平均池化函数"""
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _encode(self, texts):
        """将文本编码为向量"""
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _vector_search(self, query_text, k=5):
        """执行向量搜索"""
        query_vector = self._encode([query_text]).cpu().numpy().astype('float32')
        distances, ids = self.faiss_index.search(query_vector, k)
        
        # 返回 (id, score) 列表，分数是相似度（1 - 距离）
        results = []
        for i in range(len(ids[0])):
            if ids[0][i] != -1: # FAISS在结果不足k个时会返回-1
                results.append((ids[0][i], 1 - distances[0][i]))
        return results

    def _keyword_search(self, keywords, k=10):
        """执行关键词搜索"""
        if not keywords:
            return []
        
        doc_ids = []
        for kw in keywords:
            if kw in self.inverted_index:
                doc_ids.extend(self.inverted_index[kw])
        
        # 使用Counter计算每个文档ID的命中次数作为分数
        id_counts = Counter(doc_ids)
        
        # 返回 (id, score) 列表
        return [(doc_id, score) for doc_id, score in id_counts.most_common(k)]

    def _fuse_and_rerank(self, vector_results, keyword_results, keyword_weight=0.7, vector_weight=0.3):
        """
        融合向量和关键词搜索结果并重排序。
        这里使用一个简单的加权融合策略。
        """
        fused_scores = {}
        
        # 添加关键词搜索结果
        for doc_id, score in keyword_results:
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score * keyword_weight
            
        # 添加向量搜索结果
        for doc_id, score in vector_results:
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score * vector_weight

        # 按总分降序排序
        sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, score in sorted_results]

    def search(self, query_for_vector, keywords_for_search, top_k=5):
        """
        执行混合搜索的公共接口。
        """
        logging.info(f"执行向量搜索，查询: '{query_for_vector}'")
        vector_results = self._vector_search(query_for_vector, k=top_k * 2) # 多召回一些用于重排
        
        logging.info(f"执行关键词搜索，关键词: {keywords_for_search}")
        keyword_results = self._keyword_search(keywords_for_search, k=top_k * 2)
        
        logging.info("融合并重排序结果...")
        final_ids = self._fuse_and_rerank(vector_results, keyword_results)
        
        return final_ids[:top_k]

    def assemble_context(self, doc_ids):
        """根据文档ID列表，组装用于最终生成的上下文"""
        context_parts = []
        for i, doc_id in enumerate(doc_ids):
            chunk = self.chunk_map.get(doc_id)
            if chunk:
                context_parts.append(
                    f"[{i+1}] {chunk['article_number']}\n{chunk['content']}"
                )
        
        if not context_parts:
            return "未找到相关法律条文。"
            
        return "参考法律条文如下：\n\n" + "\n\n".join(context_parts)

if __name__ == '__main__':
    # --- 使用示例 ---
    # 1. 定义文件和模型路径
    EMBEDDING_MODEL_PATH = '/root/autodl-tmp/legal_finetune/text2vec-base-chinese'
    FAISS_INDEX_PATH = 'law_enhanced_vector_db.faiss'
    CHUNK_MAP_PATH = 'index_to_chunk_map.json'

    # 2. 初始化检索器
    retriever = HybridRetriever(
        embedding_model_path=EMBEDDING_MODEL_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        chunk_map_path=CHUNK_MAP_PATH
    )

    # 3. 模拟“查询重写模型”的输出
    # 假设用户问：“他们两个动手了，会怎么处理？”
    rewritten_query = {
      "keywords_for_search": ["打架斗殴", "故意伤害", "寻衅滋事", "人身伤害"],
      "query_for_vector_search": "两人发生肢体冲突并造成人身伤害的法律后果"
    }

    # 4. 执行搜索
    top_k_ids = retriever.search(
        query_for_vector=rewritten_query["query_for_vector_search"],
        keywords_for_search=rewritten_query["keywords_for_search"],
        top_k=3
    )
    print("\n--- 检索到的Top-K文档ID ---")
    print(top_k_ids)

    # 5. 组装上下文
    final_context = retriever.assemble_context(top_k_ids)
    print("\n--- 组装好的上下文 ---")
    print(final_context)

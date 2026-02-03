import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import logging
from rich.console import Console

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

class HybridRetriever:
    # ... (您现有的 HybridRetriever 代码保持不变) ...
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
        try:
            self.faiss_index = faiss.read_index(faiss_index_path)
            logging.info(f"✅ FAISS索引加载成功，包含 {self.faiss_index.ntotal} 个向量。")
        except Exception as e:
            logging.error(f"❌ 加载FAISS索引失败: {e}")
            raise 

        logging.info(f"正在加载数据映射文件: {chunk_map_path}")
        try:
            with open(chunk_map_path, 'r', encoding='utf-8') as f:
                self.chunk_map = {int(k): v for k, v in json.load(f).items()}
            logging.info("✅ 数据映射文件加载成功。")
        except Exception as e:
            logging.error(f"❌ 加载数据映射文件失败: {e}")
            raise

        # --- 4. 构建用于关键词搜索的倒排索引 ---
        logging.info("正在构建关键词倒排索引...")
        self.inverted_index = self._build_inverted_index()
        logging.info("✅ 倒排索引构建完成。")

    def _build_inverted_index(self):
        """构建倒排索引"""
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
        query_vector_torch = self._encode([query_text])
        query_vector_np = query_vector_torch.cpu().numpy()
        
        if query_vector_np.dtype != np.float32:
            query_vector_np = query_vector_np.astype(np.float32)

        if query_vector_np.ndim == 1:
            query_vector_np = np.expand_dims(query_vector_np, axis=0)

        if np.isnan(query_vector_np).any() or np.isinf(query_vector_np).any():
            return []

        try:
            distances, ids = self.faiss_index.search(query_vector_np, k)
        except Exception as e:
            console.print(f"[bold red]FAISS 搜索失败: {e}[/bold red]")
            return []

        results = []
        if ids.size > 0:
            for i in range(len(ids[0])):
                if ids[0][i] != -1:
                    dist = float(distances[0][i])
                    results.append((ids[0][i], 1.0 - dist if dist <= 1.0 else 0.0))
        return results

    def _keyword_search(self, keywords, k=10):
        """执行关键词搜索"""
        if not keywords: return []
        doc_ids = []
        for kw in keywords:
            if kw in self.inverted_index: doc_ids.extend(self.inverted_index[kw])
        id_counts = Counter(doc_ids)
        return [(int(doc_id), score) for doc_id, score in id_counts.most_common(k)]

    def _fuse_and_rerank(self, vector_results, keyword_results, keyword_weight=0.7, vector_weight=0.3):
        """融合结果并重排序"""
        fused_scores = {}
        for doc_id, score in keyword_results: fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score * keyword_weight
        for doc_id, score in vector_results: fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score * vector_weight
        sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [int(doc_id) for doc_id, score in sorted_results]

    def search(self, query_for_vector, keywords_for_search, top_k=5):
        """执行混合搜索的公共接口"""
        logging.info(f"执行向量搜索，查询: '{query_for_vector}'")
        vector_results = self._vector_search(query_for_vector, k=top_k * 2)

        logging.info(f"执行关键词搜索，关键词: {keywords_for_search}")
        keyword_results = self._keyword_search(keywords_for_search, k=top_k * 2)

        logging.info("融合并重排序结果...")
        final_ids = self._fuse_and_rerank(vector_results, keyword_results)

        return final_ids[:top_k]

    def assemble_context(self, doc_ids):
        """根据文档ID列表，组装用于最终生成的上下文"""
        context_parts = []
        for i, doc_id in enumerate(doc_ids):
            chunk = self.chunk_map.get(int(doc_id))
            if chunk:
                context_parts.append(
                    f"[{i+1}] {chunk.get('article_number', f'ID_{doc_id}')}\n{chunk.get('content', '')}"
                )
        if not context_parts: return "未找到相关法律条文。"
        return "参考法律条文如下：\n\n" + "\n\n".join(context_parts)

# ✅ --- 新增 NewsRetriever 类 ---
class NewsRetriever:
    """
    一个专门用于管理和搜索动态新闻知识库的检索器。
    它只使用向量搜索，并且支持在运行时动态添加新知识。
    """
    def __init__(self, embedding_model, tokenizer, device, dimension=768):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_map = {}  # 从 ID 映射到新闻块内容
        self.next_id = 0

        # 初始化一个空的FAISS索引
        logging.info(f"正在为新闻知识库初始化一个空的FAISS索引 (维度: {dimension})...")
        index = faiss.IndexFlatL2(dimension)
        self.faiss_index = faiss.IndexIDMap(index)
        logging.info("✅ 新闻知识库检索器初始化成功。")

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

    def add_documents(self, chunks: list[dict]):
        """将审核通过的新闻块添加到知识库中"""
        if not chunks:
            return
            
        console.print(f"[cyan]正在向新闻知识库添加 {len(chunks)} 个新段落...[/cyan]")
        texts_to_embed = [chunk['content'] for chunk in chunks]
        
        # 1. 向量化
        new_embeddings = self._encode(texts_to_embed).cpu().numpy().astype(np.float32)
        
        # 2. 生成新ID
        new_ids = list(range(self.next_id, self.next_id + len(chunks)))
        self.next_id += len(chunks)
        
        # 3. 更新FAISS索引
        try:
            self.faiss_index.add_with_ids(new_embeddings, np.array(new_ids))
            console.print(f"[green]✅ {len(chunks)} 个新向量已添加到新闻索引。当前总数: {self.faiss_index.ntotal}[/green]")
        except Exception as e:
            console.print(f"[bold red]❌ 添加向量到新闻索引失败: {e}[/bold red]")
            return

        # 4. 更新映射文件
        for i, chunk_data in enumerate(chunks):
            new_id = new_ids[i]
            self.chunk_map[new_id] = {
                "id": new_id,
                "source": chunk_data.get('source', '未知在线来源'),
                "content": chunk_data['content'],
                "type": "news_event" # 标记为新闻事件
            }
        console.print("[green]✅ 新闻数据映射已更新。[/green]")

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """对新闻知识库执行向量搜索"""
        if self.faiss_index.ntotal == 0:
            logging.info("新闻知识库为空，跳过搜索。")
            return []
            
        logging.info(f"在新闻知识库中搜索: '{query_text[:50]}...'")
        
        query_vector = self._encode([query_text]).cpu().numpy().astype(np.float32)
        
        try:
            distances, ids = self.faiss_index.search(query_vector, min(top_k, self.faiss_index.ntotal))
            
            results = []
            if ids.size > 0:
                for i in range(len(ids[0])):
                    doc_id = int(ids[0][i])
                    if doc_id != -1 and doc_id in self.chunk_map:
                        results.append(self.chunk_map[doc_id])
            return results
        except Exception as e:
            console.print(f"[bold red]❌ 新闻知识库搜索失败: {e}[/bold red]")
            return []

    def assemble_context(self, docs: list[dict]):
        """为新闻文档组装上下文"""
        if not docs:
            return ""
        context_parts = [f"[{i+1}] 来源: {doc['source']}\n内容: {doc['content']}" for i, doc in enumerate(docs)]
        return "相关新闻/事件背景如下：\n\n" + "\n\n".join(context_parts)

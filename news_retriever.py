import faiss
import numpy as np
import torch
import logging
from rich.console import Console

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

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
        
        # 1. 向量化 (✅ 使用 np.float32)
        new_embeddings = self._encode(texts_to_embed).cpu().numpy().astype(np.float32)
        
        # 2. 生成新ID
        new_ids = list(range(self.next_id, self.next_id + len(chunks)))
        
        # 3. 更新FAISS索引
        try:
            self.faiss_index.add_with_ids(new_embeddings, np.array(new_ids))
            self.next_id += len(chunks) 
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
                "type": "news_event"
            }
        console.print("[green]✅ 新闻数据映射已更新。[/green]")

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """对新闻知识库执行向量搜索"""
        if self.faiss_index.ntotal == 0:
            logging.info("新闻知识库为空，跳过搜索。")
            return []
            
        logging.info(f"在新闻知识库中搜索: '{query_text[:50]}...'")
        
        # ✅ 使用 np.float32
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


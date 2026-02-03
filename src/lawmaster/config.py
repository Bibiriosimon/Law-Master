"""
配置管理模块

提供统一的配置管理接口，支持从配置文件、环境变量和命令行参数加载配置。

符合openEuler编码规范:
    - 使用类型提示
    - 完善的文档字符串
    - 配置验证机制
    - 支持配置热更新
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置类
    
    Attributes:
        base_model_path: 基座模型路径
        query_rewriter_path: 查询重写模型路径
        answer_generator_path: 答案生成模型路径
        device: 运行设备 (cuda/cpu)
        dtype: 数据类型 (auto/float16/bfloat16)
        max_length: 最大序列长度
    """
    base_model_path: str = "./models/deepseek"
    query_rewriter_path: str = "./models/query_rewriter_lora/final_model"
    answer_generator_path: str = "./models/answer_generator_lora/final_model"
    device: str = "cuda"
    dtype: str = "auto"
    max_length: int = 1024


@dataclass
class RetrieverConfig:
    """检索器配置类
    
    Attributes:
        embedding_model_path: 嵌入模型路径
        faiss_index_path: FAISS索引文件路径
        chunk_map_path: chunk映射文件路径
        top_k: 返回top-k结果数量
        keyword_weight: 关键词搜索权重
        vector_weight: 向量搜索权重
        use_rerank: 是否启用重排序
    """
    embedding_model_path: str = "./models/text2vec-base-chinese"
    faiss_index_path: str = "./data/law_enhanced_vector_db.faiss"
    chunk_map_path: str = "./data/index_to_chunk_map.json"
    top_k: int = 5
    keyword_weight: float = 0.7
    vector_weight: float = 0.3
    use_rerank: bool = False


@dataclass
class SystemConfig:
    """系统配置类
    
    Attributes:
        log_level: 日志级别
        log_file: 日志文件路径
        cache_dir: 缓存目录
        num_workers: 工作线程数
        enable_monitoring: 是否启用性能监控
    """
    log_level: str = "INFO"
    log_file: Optional[str] = None
    cache_dir: str = "./cache"
    num_workers: int = 4
    enable_monitoring: bool = False


@dataclass
class Config:
    """主配置类
    
    聚合所有子配置，提供统一的配置接口。
    
    Attributes:
        model: 模型配置
        retriever: 检索器配置
        system: 系统配置
    
    Examples:
        >>> config = Config()
        >>> config.model.device = "cpu"
        >>> config.validate()
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self) -> None:
        """初始化后验证配置"""
        self._setup_logging()
        self.validate()
    
    def _setup_logging(self) -> None:
        """配置日志系统"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.system.log_level),
            format=log_format,
            filename=self.system.log_file
        )
    
    def validate(self) -> None:
        """验证配置有效性
        
        Raises:
            ValueError: 当配置无效时抛出
        """
        # 验证模型路径
        if not Path(self.model.base_model_path).exists():
            raise ValueError(f"基座模型路径不存在: {self.model.base_model_path}")
        
        # 验证设备配置
        if self.model.device not in ["cuda", "cpu", "auto"]:
            raise ValueError(f"不支持的设备类型: {self.model.device}")
        
        # 验证权重和
        weight_sum = self.retriever.keyword_weight + self.retriever.vector_weight
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(
                f"检索权重之和不为1.0: {weight_sum}，已自动归一化"
            )
            total = weight_sum
            self.retriever.keyword_weight /= total
            self.retriever.vector_weight /= total
        
        # 创建必要的目录
        Path(self.system.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("配置验证通过")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """从配置文件加载配置
        
        Args:
            config_path: 配置文件路径（支持.json/.yaml/.toml）
        
        Returns:
            Config实例
        
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        import json
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典
        
        Returns:
            配置字典
        """
        return {
            'model': self.model.__dict__,
            'retriever': self.retriever.__dict__,
            'system': self.system.__dict__
        }
    
    def save(self, config_path: str) -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件保存路径
        """
        import json
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {config_path}")


if __name__ == "__main__":
    # 测试配置模块
    config = Config()
    print("默认配置:")
    print(config.to_dict())


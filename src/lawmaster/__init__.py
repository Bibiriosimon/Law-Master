"""
LAW MASTER - 智能法律助手系统

本模块提供基于大语言模型与检索增强生成（RAG）架构的智能法律咨询服务。

主要功能:
    - 法律问题查询重写
    - 混合检索（向量+关键词）
    - 专业法律答案生成
    - 多轮对话支持

作者: LAW MASTER Team
许可证: MIT
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "LAW MASTER Team"
__license__ = "MIT"

from .config import Config
from .retriever import HybridRetriever
from .generator import LegalAnswerGenerator
from .pipeline import LegalQAPipeline

__all__ = [
    "Config",
    "HybridRetriever",
    "LegalAnswerGenerator",
    "LegalQAPipeline",
]


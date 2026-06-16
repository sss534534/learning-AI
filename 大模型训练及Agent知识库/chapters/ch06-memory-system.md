# 第六章：Agent记忆系统

> 记忆系统是Agent的核心组件，让智能体能够存储和检索信息，维持对话连贯性并积累知识。本章将深入讲解短期记忆、长期记忆、记忆检索策略以及记忆压缩与优化技术。

## 目录

1. [短期记忆](#1-短期记忆)
2. [长期记忆](#2-长期记忆)
3. [记忆检索策略](#3-记忆检索策略)
4. [记忆压缩与优化](#4-记忆压缩与优化)

---

## 元数据
- **难度**: ⭐⭐
- **前置知识**: ../chapters/ch05-tool-calling.md
- **关联文件**: ../chapters/ch07-multi-agent.md
- **最后更新**: 2026-06-12
---

## 1. 短期记忆

### 1.1 对话历史与上下文管理

**短期记忆** 主要用于存储当前对话的上下文信息，通常保存在内存中，具有以下特点：
- 访问速度快
- 容量有限
- 会话结束后可能丢失

```
┌─────────────────────────────────────────────────────────┐
│                    短期记忆系统                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              对话历史缓冲区                        │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  [用户消息] 你好，帮我查一下北京的天气         │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │  [助手回复] 好的，让我查询一下...             │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │  [工具结果] 北京天气：晴，25°C              │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │  [助手回复] 北京今天天气晴朗，温度25°C        │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  │                   ↑                              │   │
│  │              滑动窗口机制                          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 滑动窗口实现

```python
from typing import List, Dict
from collections import deque

class ShortTermMemory:
    """短期记忆管理器"""
    
    def __init__(self, max_tokens: int = 4096, max_messages: int = 50):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.messages: List[Dict] = []
        self.token_count = 0
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（简化版）"""
        # 实际应用中应该使用tiktoken或模型特定的tokenizer
        return len(text) // 4
    
    def add(self, role: str, content: str):
        """添加消息到短期记忆"""
        message = {"role": role, "content": content}
        message_tokens = self._estimate_tokens(content)
        
        # 添加新消息
        self.messages.append(message)
        self.token_count += message_tokens
        
        # 如果超过最大消息数，移除最早的消息
        while len(self.messages) &gt; self.max_messages:
            removed = self.messages.pop(0)
            self.token_count -= self._estimate_tokens(removed["content"])
        
        # 如果超过token限制，移除最早的消息直到符合限制
        while self.token_count &gt; self.max_tokens and len(self.messages) &gt; 1:
            removed = self.messages.pop(0)
            self.token_count -= self._estimate_tokens(removed["content"])
    
    def get_context(self) -> List[Dict]:
        """获取当前上下文"""
        return self.messages.copy()
    
    def clear(self):
        """清空短期记忆"""
        self.messages = []
        self.token_count = 0

# 使用示例
memory = ShortTermMemory(max_tokens=2048, max_messages=20)
memory.add("user", "你好，帮我查一下北京的天气")
memory.add("assistant", "好的，让我查询一下...")
memory.add("user", "谢谢！")

context = memory.get_context()
print(context)
```

### 1.3 上下文压缩

```python
class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_old_messages(self, messages: List[Dict], keep_latest: int = 5) -> List[Dict]:
        """压缩旧消息，保留最新的几条"""
        if len(messages) &lt;= keep_latest:
            return messages
        
        old_messages = messages[:-keep_latest]
        latest_messages = messages[-keep_latest:]
        
        # 压缩旧消息
        summary = self._summarize_conversation(old_messages)
        
        # 返回压缩后的上下文
        compressed_context = [
            {"role": "system", "content": f"Previous conversation summary:\n{summary}"}
        ] + latest_messages
        
        return compressed_context
    
    def _summarize_conversation(self, messages: List[Dict]) -> str:
        """使用LLM压缩对话历史"""
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        summary_prompt = f"""Summarize the following conversation concisely, 
capturing the key points and information that might be needed later.

Conversation:
{conversation_text}

Summary:"""
        
        return self.llm.complete(summary_prompt)

# 使用示例
# compressor = ContextCompressor(llm)
# compressed = compressor.summarize_old_messages(memory.get_context(), keep_latest=5)
```

---

## 2. 长期记忆

### 2.1 向量数据库存储

**长期记忆** 用于持久化存储重要信息，通常使用向量数据库进行语义检索。

```
┌─────────────────────────────────────────────────────────┐
│                    长期记忆系统                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              向量嵌入 (Embedding)                 │   │
│  │  文本 → [0.12, -0.45, 0.78, ..., 0.32]           │   │
│  └─────────────────────────────────────────────────┘   │
│                           ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              向量数据库 (Vector DB)               │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │ ID  |  向量  |  元数据  |  内容          │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │ 001 | [向量] | {...}   | "北京是中国首都" │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │ 002 | [向量] | {...}   | "机器学习是AI分支" │    │   │
│  │  ├─────────────────────────────────────────┤    │   │
│  │  │ 003 | [向量] | {...}   | "Python是编程语言" │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 向量数据库实现

```python
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: float

class VectorDatabase:
    """简单的向量数据库实现"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.items: List[MemoryItem] = []
        self.id_to_index: Dict[str, int] = {}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def add(self, item: MemoryItem):
        """添加记忆项"""
        if item.id in self.id_to_index:
            # 更新现有项
            idx = self.id_to_index[item.id]
            self.items[idx] = item
        else:
            # 添加新项
            self.id_to_index[item.id] = len(self.items)
            self.items.append(item)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        """语义搜索"""
        if not self.items:
            return []
        
        # 计算相似度
        similarities = []
        for item in self.items:
            sim = self._cosine_similarity(query_embedding, item.embedding)
            similarities.append((sim, item))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:top_k]]
    
    def delete(self, item_id: str):
        """删除记忆项"""
        if item_id in self.id_to_index:
            idx = self.id_to_index[item_id]
            self.items.pop(idx)
            # 更新索引
            self.id_to_index = {item.id: i for i, item in enumerate(self.items)}
    
    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆项"""
        return self.items.copy()

# 使用OpenAI Embeddings的完整实现
import openai
import time

class EmbeddingModel:
    """嵌入模型封装"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def embed(self, text: str) -> List[float]:
        """获取文本嵌入"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

class LongTermMemory:
    """长期记忆管理器"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.vector_db = VectorDatabase()
        self.next_id = 1
    
    def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        """存储记忆"""
        # 生成嵌入
        embedding = self.embedding_model.embed(content)
        
        # 创建记忆项
        item_id = f"mem_{self.next_id}"
        self.next_id += 1
        
        item = MemoryItem(
            id=item_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # 存储到向量数据库
        self.vector_db.add(item)
        return item_id
    
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """检索记忆"""
        # 获取查询的嵌入
        query_embedding = self.embedding_model.embed(query)
        
        # 搜索
        return self.vector_db.search(query_embedding, top_k)

# 使用示例
# embedding_model = EmbeddingModel(api_key="your-api-key")
# long_term_memory = LongTermMemory(embedding_model)
# 
# # 存储记忆
# long_term_memory.store(
#     "北京是中华人民共和国的首都，位于中国北部",
#     {"category": "geography", "source": "knowledge_base"}
# )
# long_term_memory.store(
#     "Python是一种高级编程语言，以简洁著称",
#     {"category": "programming", "source": "documentation"}
# )
# 
# # 检索记忆
# results = long_term_memory.retrieve("中国的首都是哪里？", top_k=3)
# for result in results:
#     print(f"Content: {result.content}, Similarity: ...")
```

### 2.3 知识图谱存储

```python
from typing import Dict, Set, Tuple

class KnowledgeGraph:
    """简单的知识图谱实现"""
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}  # 实体
        self.relations: Set[Tuple[str, str, str]] = set()  # (主体, 关系, 客体)
    
    def add_entity(self, entity_id: str, properties: Dict):
        """添加实体"""
        if entity_id not in self.entities:
            self.entities[entity_id] = {}
        self.entities[entity_id].update(properties)
    
    def add_relation(self, subject: str, predicate: str, object: str):
        """添加关系"""
        self.relations.add((subject, predicate, object))
    
    def query(self, subject: Optional[str] = None, 
              predicate: Optional[str] = None, 
              object: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """查询知识图谱"""
        results = []
        for s, p, o in self.relations:
            if (subject is None or s == subject) and \
               (predicate is None or p == predicate) and \
               (object is None or o == object):
                results.append((s, p, o))
        return results
    
    def get_entity_properties(self, entity_id: str) -> Dict:
        """获取实体属性"""
        return self.entities.get(entity_id, {})

# 使用示例
# kg = KnowledgeGraph()
# 
# # 添加实体
# kg.add_entity("北京", {"type": "城市", "人口": "2189万"})
# kg.add_entity("中国", {"type": "国家", "面积": "960万平方公里"})
# 
# # 添加关系
# kg.add_relation("北京", "是首都", "中国")
# kg.add_relation("北京", "位于", "中国北部")
# 
# # 查询
# results = kg.query(subject="北京")
# print(results)  # [('北京', '是首都', '中国'), ('北京', '位于', '中国北部')]
```

---

## 3. 记忆检索策略

### 3.1 语义检索

**语义检索** 基于向量相似度查找相关记忆，是最常用的检索方式。

```python
class SemanticRetriever:
    """语义检索器"""
    
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory
    
    def retrieve(self, query: str, top_k: int = 5, 
                 min_similarity: float = 0.7) -> List[Dict]:
        """检索相关记忆"""
        # 从长期记忆中检索
        items = self.long_term_memory.retrieve(query, top_k)
        
        # 过滤低相似度结果
        # 注意：这里简化了，实际需要重新计算相似度
        results = []
        for item in items:
            results.append({
                "content": item.content,
                "metadata": item.metadata,
                "timestamp": item.timestamp,
                "id": item.id
            })
        
        return results
    
    def retrieve_with_reranking(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索并重排序"""
        # 初始检索
        candidates = self.retrieve(query, top_k * 2)
        
        # 使用LLM重排序
        reranked = self._rerank(query, candidates)
        
        return reranked[:top_k]
    
    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用LLM重排序"""
        # 实际应用中应该使用交叉编码器或LLM进行重排序
        # 这里简化处理
        return candidates
```

### 3.2 混合检索

**混合检索** 结合语义检索和关键词检索，提高检索准确率。

```python
import re
from typing import List, Dict

class KeywordRetriever:
    """关键词检索器"""
    
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """关键词检索"""
        # 提取查询关键词
        keywords = self._extract_keywords(query)
        
        # 获取所有记忆项
        all_items = self.long_term_memory.vector_db.get_all()
        
        # 计算匹配分数
        scored_items = []
        for item in all_items:
            score = self._calculate_keyword_score(item.content, keywords)
            if score &gt; 0:
                scored_items.append((score, item))
        
        # 排序并返回top_k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored_items[:top_k]:
            results.append({
                "content": item.content,
                "metadata": item.metadata,
                "timestamp": item.timestamp,
                "id": item.id,
                "keyword_score": score
            })
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 实际应用中应该使用分词和停用词过滤
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) &gt; 2]
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        content_lower = content.lower()
        score = 0.0
        for keyword in keywords:
            if keyword in content_lower:
                score += 1.0
        return score

class HybridRetriever:
    """混合检索器（语义+关键词）"""
    
    def __init__(self, semantic_retriever: SemanticRetriever, 
                 keyword_retriever: KeywordRetriever,
                 semantic_weight: float = 0.6,
                 keyword_weight: float = 0.4):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索"""
        # 分别获取两种检索结果
        semantic_results = self.semantic_retriever.retrieve(query, top_k * 2)
        keyword_results = self.keyword_retriever.retrieve(query, top_k * 2)
        
        # 合并结果
        combined = self._merge_results(semantic_results, keyword_results)
        
        return combined[:top_k]
    
    def _merge_results(self, semantic: List[Dict], keyword: List[Dict]) -> List[Dict]:
        """合并两种检索结果"""
        # 创建分数字典
        scores = {}
        
        # 处理语义检索结果
        for i, result in enumerate(semantic):
            doc_id = result["id"]
            if doc_id not in scores:
                scores[doc_id] = {"result": result, "semantic_score": 0, "keyword_score": 0}
            scores[doc_id]["semantic_score"] = 1.0 - (i / len(semantic))
        
        # 处理关键词检索结果
        for i, result in enumerate(keyword):
            doc_id = result["id"]
            if doc_id not in scores:
                scores[doc_id] = {"result": result, "semantic_score": 0, "keyword_score": 0}
            scores[doc_id]["keyword_score"] = 1.0 - (i / len(keyword))
        
        # 计算综合分数
        merged = []
        for doc_id, data in scores.items():
            total_score = (self.semantic_weight * data["semantic_score"] + 
                          self.keyword_weight * data["keyword_score"])
            merged.append((total_score, data["result"]))
        
        # 排序
        merged.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in merged]
```

### 3.3 时间加权检索

```python
import time

class TimeWeightedRetriever:
    """时间加权检索器"""
    
    def __init__(self, semantic_retriever: SemanticRetriever, 
                 half_life_days: float = 30.0):
        self.semantic_retriever = semantic_retriever
        self.half_life_seconds = half_life_days * 24 * 3600
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """时间加权检索"""
        # 获取基础检索结果
        results = self.semantic_retriever.retrieve(query, top_k * 2)
        
        # 计算时间衰减因子
        now = time.time()
        scored = []
        for result in results:
            age = now - result["timestamp"]
            time_decay = 0.5 ** (age / self.half_life_seconds)
            # 这里简化了，实际应该结合相似度分数
            scored.append((time_decay, result))
        
        # 按时间加权排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored[:top_k]]
```

---

## 4. 记忆压缩与优化

### 4.1 记忆摘要

```python
class MemorySummarizer:
    """记忆摘要器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_memories(self, memories: List[Dict], target_length: int = 500) -> str:
        """摘要多个记忆"""
        memories_text = "\n\n".join([
            f"- {mem['content']}"
            for mem in memories
        ])
        
        summary_prompt = f"""Summarize the following information into a concise summary 
(max {target_length} characters). Focus on key facts and knowledge that would be 
useful for future reference.

Information to summarize:
{memories_text}

Concise summary:"""
        
        return self.llm.complete(summary_prompt)
    
    def compress_memory_group(self, memories: List[Dict]) -> Dict:
        """将一组记忆压缩为单个摘要记忆"""
        summary = self.summarize_memories(memories)
        
        return {
            "content": summary,
            "metadata": {
                "type": "compressed",
                "original_count": len(memories),
                "original_ids": [m["id"] for m in memories]
            },
            "timestamp": time.time()
        }
```

### 4.2 记忆重要性评分

```python
class MemoryScorer:
    """记忆重要性评分器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def score_importance(self, memory_content: str) -> float:
        """评分记忆重要性 (0.0 - 1.0)"""
        scoring_prompt = f"""On a scale of 0.0 to 1.0, rate the importance of the 
following information for long-term retention. Consider:
- How useful is this for future tasks?
- Is this factual knowledge or transient information?
- Does this reveal user preferences or characteristics?

Information:
{memory_content}

Return only a number between 0.0 and 1.0:"""
        
        response = self.llm.complete(scoring_prompt)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # 默认中等重要性
    
    def filter_low_importance(self, memories: List[Dict], 
                            threshold: float = 0.3) -> List[Dict]:
        """过滤低重要性记忆"""
        filtered = []
        for mem in memories:
            score = self.score_importance(mem["content"])
            if score &gt;= threshold:
                mem["importance_score"] = score
                filtered.append(mem)
        return filtered
```

### 4.3 完整的记忆系统集成

```python
class UnifiedMemorySystem:
    """统一记忆系统"""
    
    def __init__(self, 
                 short_term: ShortTermMemory,
                 long_term: LongTermMemory,
                 retriever: HybridRetriever,
                 summarizer: MemorySummarizer,
                 scorer: MemoryScorer):
        self.short_term = short_term
        self.long_term = long_term
        self.retriever = retriever
        self.summarizer = summarizer
        self.scorer = scorer
    
    def add_message(self, role: str, content: str):
        """添加消息到记忆系统"""
        # 添加到短期记忆
        self.short_term.add(role, content)
        
        # 如果是用户消息，考虑存储到长期记忆
        if role == "user":
            # 简单启发式：存储较长的消息
            if len(content) &gt; 50:
                self.long_term.store(
                    content,
                    {"type": "conversation", "role": role}
                )
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> Dict:
        """获取相关上下文"""
        # 获取短期记忆
        short_term_context = self.short_term.get_context()
        
        # 从长期记忆中检索相关内容
        long_term_results = self.retriever.retrieve(query, top_k)
        
        return {
            "short_term": short_term_context,
            "long_term": long_term_results
        }
    
    def format_context_for_llm(self, context: Dict) -> str:
        """格式化上下文供LLM使用"""
        parts = []
        
        # 添加长期记忆
        if context["long_term"]:
            parts.append("Relevant information from memory:")
            for mem in context["long_term"]:
                parts.append(f"- {mem['content']}")
            parts.append("")
        
        # 添加对话历史
        parts.append("Conversation history:")
        for msg in context["short_term"]:
            parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(parts)

# 使用示例
# 初始化各个组件
# embedding_model = EmbeddingModel(api_key="your-api-key")
# short_term = ShortTermMemory()
# long_term = LongTermMemory(embedding_model)
# semantic_retriever = SemanticRetriever(long_term)
# keyword_retriever = KeywordRetriever(long_term)
# hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
# summarizer = MemorySummarizer(llm)
# scorer = MemoryScorer(llm)
# 
# # 创建统一记忆系统
# memory_system = UnifiedMemorySystem(
#     short_term=short_term,
#     long_term=long_term,
#     retriever=hybrid_retriever,
#     summarizer=summarizer,
#     scorer=scorer
# )
# 
# # 使用记忆系统
# memory_system.add_message("user", "你好，我叫张三，我喜欢编程")
# memory_system.add_message("assistant", "你好张三！很高兴认识你，你喜欢什么编程语言？")
# memory_system.add_message("user", "我最喜欢Python，它很简洁")
# 
# # 检索相关上下文
# context = memory_system.get_relevant_context("用户喜欢什么编程语言？")
# formatted = memory_system.format_context_for_llm(context)
# print(formatted)
```

---

## 深度分析

记忆系统是Agent实现持续性交互和知识积累的核心基础设施。从认知科学的角度看，Agent的记忆架构与人类记忆有着深层的对应关系：短期记忆（STM）对应工作记忆，负责维护当前对话上下文的连贯性；长期记忆（LTM）对应情景记忆和语义记忆，通过向量嵌入将信息转化为可检索的语义表示。这种分层记忆架构的关键设计决策在于如何在有限的上下文窗口内，平衡信息保留的完整性与token消耗的经济性。滑动窗口和上下文压缩是对抗上下文窗口限制的两种基本策略，前者简单有效但可能丢失窗口之外的关键信息，后者通过摘要保留语义但可能引入信息失真。

在检索层面，混合检索策略（语义检索+关键词检索）已经成为工业级记忆系统的标配。纯语义检索依赖于嵌入质量，对同义词和近义词效果良好，但对精确匹配和稀有实体表现不稳定；关键词检索恰好形成互补。时间加权检索则引入了记忆的时效性维度，特别适用于需要区分近期和远期信息的场景。更高级的记忆系统还引入了记忆重要性评分和自动压缩机制，使得系统能够自主决定哪些信息值得长期保留、哪些可以合并或遗忘——这与人类记忆的巩固和遗忘机制高度相似。

从工程实现角度看，向量数据库的选择和嵌入模型的质量直接影响记忆系统的检索效果。实际部署中需要关注嵌入维度对存储和检索效率的影响、索引结构的检索速度与准确率权衡、以及多租户场景下的隔离策略。短期记忆和长期记忆的协同工作是一个容易被低估的挑战——如何判断何时将短期记忆中的信息转移到长期记忆（记忆巩固），以及如何在检索结果中融合两类记忆的上下文，需要精心设计策略。未来，具有分层遗忘机制和主动记忆回忆能力的Agent记忆系统，将是构建真正"有生命感"的智能体的关键方向。

---

## Checklist

- [ ] 理解短期记忆的实现机制，包括滑动窗口和token预算管理
- [ ] 掌握上下文压缩策略，使用LLM对旧消息进行摘要
- [ ] 实现基于向量数据库的长期记忆存储和检索
- [ ] 掌握嵌入模型的使用（如text-embedding-ada-002）
- [ ] 实现语义检索器，包括相似度计算和阈值过滤
- [ ] 实现混合检索器（语义+关键词），了解权重调优方法
- [ ] 实现时间加权检索，掌握半衰期参数的设计
- [ ] 实现记忆重要性评分和自动压缩机制
- [ ] 集成短期记忆和长期记忆为统一的记忆系统
- [ ] 在实际Agent中接入记忆系统，测试记忆回召效果

---

## 延伸阅读

- [第五章：工具调用与Function Calling](../chapters/ch05-tool-calling.md) - 工具调用与记忆系统的协同
- [第七章：多Agent协作系统](../chapters/ch07-multi-agent.md) - 多Agent场景中的记忆共享策略
- Pinecone文档 - https://docs.pinecone.io - 向量数据库最佳实践
- LangChain Memory模块 - https://python.langchain.com/docs/modules/memory/
- Memory Systems in AI Agents综述 - Agent记忆系统技术对比

---

## 本章小结

记忆系统是Agent的关键组件：

1. **短期记忆** 通过滑动窗口管理对话上下文
2. **长期记忆** 使用向量数据库实现语义检索
3. **混合检索** 结合语义和关键词检索提高准确率
4. **记忆压缩** 通过摘要和重要性评分优化存储

**下一章：** 我们将学习多Agent协作系统。

---

*最后更新: 2026-06-12*

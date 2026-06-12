# LLM 语义缓存架构

## 元数据
- **难度**: ⭐⭐
- **前置知识**: RAG基础概念
- **关联文件**: 02-模型服务与推理优化.md, 05-向量数据库架构深度.md, 07-流式AI与实时推理架构.md
- **最后更新**: 2026-06-12
---

> 语义缓存是 LLM 应用 ROI 最高的优化手段——30-50% 成本节省，几乎零质量损失。
> 但做得不好，轻则缓存污染，重则返回过期数据导致生产事故。

## 目录

1. [为什么要语义缓存](#1-为什么要语义缓存)
2. [GPTCache 实战](#2-gptcache-实战)
3. [自建语义缓存架构](#3-自建语义缓存架构)
4. [多级缓存架构](#4-多级缓存架构)
5. [前缀缓存与 KV Cache](#5-前缀缓存与-kv-cache)
6. [缓存策略与失效管理](#6-缓存策略与失效管理)
7. [生产级部署](#7-生产级部署)

---

## 1. 为什么要语义缓存

### 1.1 缓存类型对比

```
精确匹配缓存:
  请求A: "查一下设备 ABC-001 的告警"
  请求B: "查一下设备 ABC-001 的告警"      ← 完全相同，命中
  请求C: "设备 ABC-001 有什么告警"        ← 不同文字，不命中 ✗

语义缓存:
  请求A: "查一下设备 ABC-001 的告警"
  请求B: "设备 ABC-001 有什么告警"        ← 语义相似 95%，命中 ✓
  请求C: "ABC-001 告警情况"              ← 语义相似 92%，命中 ✓
```

### 1.2 收益量化

| 场景 | 重复查询比例 | 语义缓存命中率 | 成本节省 |
|------|------------|-------------|---------|
| 客服 | 60-80% | 40-60% | $3000→$1500 |
| RAG问答 | 30-50% | 25-40% | $2000→$1400 |
| 网管运维 | 50-70% | 35-55% | $800→$440 |
| 代码辅助 | 20-30% | 15-25% | $1500→$1200 |

---

## 2. GPTCache 实战

### 2.1 快速集成

```python
# pip install gptcache

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation import OnnxModelEvaluation
from gptcache.processor.post import temperature_softmax

# ============ 初始化 GPTCache ============

# 1. Embedding 模型 (将文本转为向量)
onnx_embedding = EmbeddingOnnx()

# 2. 向量存储 (Faiss)
vector_store = manager_factory(
    "sqlite,faiss",  # 元数据存 SQLite, 向量存 Faiss
    data_dir="./gptcache_data",
    vector_params={"dimension": onnx_embedding.dimension}
)

# 3. 相似度评估 (多少算"命中")
similarity_eval = OnnxModelEvaluation()

# 4. 创建缓存配置
cache.init(
    embedding_func=onnx_embedding.to_embeddings,
    data_manager=vector_store,
    similarity_evaluation=similarity_eval,
    config=Config(similarity_threshold=0.8)  # 相似度 > 0.8 才算命中
)

# ============ 使用 ============
from gptcache.adapter import openai

# 正常使用 OpenAI API，GPTCache 自动拦截
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "OSPF协议的工作原理是什么？"}]
)
# 首次调用 → LLM → 缓存结果

response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "请解释OSPF是如何工作的"}]
)
# 语义相似 → 命中缓存 → 直接返回 (不调用LLM)
```

### 2.2 高级配置

```python
from gptcache import Cache
from gptcache.adapter.api import get, put

class GPTCacheAdvanced:
    """GPTCache 高级用法"""
    
    def __init__(self):
        # 自定义缓存键提取 (只用 user message，忽略 system prompt)
        def custom_key_func(**kwargs):
            messages = kwargs.get("messages", [])
            # 只取最后一条用户消息
            user_messages = [m["content"] for m in messages if m["role"] == "user"]
            return user_messages[-1] if user_messages else ""
        
        cache.init(
            embedding_func=onnx_embedding.to_embeddings,
            data_manager=vector_store,
            similarity_evaluation=similarity_eval,
            pre_func=custom_key_func,  # 自定义缓存键
            config=Config(
                similarity_threshold=0.8,
                max_size=10000,  # 最多缓存 10000 条
                eviction="LRU",  # 淘汰策略: LRU
            )
        )
    
    def query_with_cache(self, prompt: str, max_age_seconds: int = 3600):
        """带 TTL 的缓存查询"""
        cached = get(prompt)
        
        if cached:
            age = time.time() - cached["timestamp"]
            if age < max_age_seconds:
                return cached["response"], True  # 缓存命中
        
        # 调用 LLM
        response = self.call_llm(prompt)
        
        # 存入缓存
        put(prompt, {
            "response": response,
            "timestamp": time.time()
        })
        
        return response, False
```

---

## 3. 自建语义缓存架构

### 3.1 核心架构

```
                  请求进入
                      │
                      ▼
              ┌───────────────┐
              │  缓存键提取    │  ← 提取用户查询文本
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Embedding 层  │  ← 文本 → 向量 (768/1536 dim)
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Faiss 检索    │  ← ANN (近似最近邻) 搜索 Top-K
              └───────┬───────┘
                      │
             ┌────────┴────────┐
             │ 相似度 > 阈值?   │
             └────────┬────────┘
              │               │
             是              否
              │               │
              ▼               ▼
        ┌──────────┐   ┌──────────┐
        │ 返回缓存  │   │ 调用 LLM │
        │ 结果     │   │ + 写入缓存│
        └──────────┘   └──────────┘
```

### 3.2 完整实现

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
import time
import json

class SemanticCache:
    """生产级语义缓存"""
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 100000,
                 redis_host: str = "localhost"):
        
        # Embedding 模型
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Faiss 索引 (IVF + PQ 加速)
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dimension, 
            nlist=100,  # 聚类中心数
            m=8,        # PQ 子向量数
            nbits=8     # 每个子向量编码位数
        )
        
        self.threshold = similarity_threshold
        self.max_size = max_cache_size
        
        # Redis 存储完整的缓存数据
        self.redis = redis.Redis(
            host=redis_host, decode_responses=True
        )
        
        # ID 映射
        self.faiss_to_key = {}  # Faiss索引ID → Redis key
        self.current_size = 0
    
    def _encode(self, text: str) -> np.ndarray:
        """文本转向量"""
        embedding = self.encoder.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32).reshape(1, -1)
    
    def search(self, query: str, k: int = 5) -> tuple:
        """搜索相似缓存"""
        if self.index.ntotal == 0:
            return None, 0.0
        
        query_vec = self._encode(query)
        scores, indices = self.index.search(query_vec, k)
        
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        
        if best_score >= self.threshold and best_idx >= 0:
            redis_key = self.faiss_to_key.get(best_idx)
            if redis_key:
                cached = self.redis.get(redis_key)
                if cached:
                    return json.loads(cached), best_score
        
        return None, best_score
    
    def store(self, query: str, response: dict, ttl: int = 3600):
        """存储缓存"""
        # 生成唯一键
        cache_key = f"semcache:{hash(query)}"
        
        # 存入 Redis
        cache_data = {
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl
        }
        self.redis.setex(cache_key, ttl, json.dumps(cache_data))
        
        # 存入 Faiss
        query_vec = self._encode(query)
        faiss_id = self.current_size
        
        if self.index.is_trained:
            self.index.add(query_vec)
        else:
            # 首次需要训练
            self.index.train(query_vec)
            self.index.add(query_vec)
        
        self.faiss_to_key[faiss_id] = cache_key
        self.current_size += 1
        
        # LRU 淘汰
        if self.current_size > self.max_size:
            self._evict_lru()
    
    def _evict_lru(self):
        """淘汰最旧的条目"""
        # 简化版: 移除索引中最早的一项
        oldest_id = min(self.faiss_to_key.keys())
        oldest_key = self.faiss_to_key.pop(oldest_id)
        self.redis.delete(oldest_key)
        self.current_size -= 1
```

### 3.3 相似度阈值调优

| 阈值 | 命中率 | 返回错误概率 | 建议场景 |
|------|--------|------------|---------|
| 0.95+ | 10-20% | 极低 | 严格场景 (金融/医疗) |
| 0.85-0.95 | 25-40% | 很低 | 推荐场景 (生产默认) |
| 0.75-0.85 | 35-55% | 低 | 探索场景 (研发/测试) |
| 0.65-0.75 | 45-65% | 中等 | 不推荐生产 |
| < 0.65 | 60%+ | 高 | 不建议使用 |

---

## 4. 多级缓存架构

### 4.1 三级缓存

```
请求
  │
  ▼
┌──────────────┐
│ L1: 内存缓存   │  ← 精确匹配 + LFU, 容量 1000 条, < 1ms
│ (LRU Cache)  │
└──────┬───────┘
       │ 未命中
       ▼
┌──────────────┐
│ L2: 语义缓存   │  ← Faiss 向量检索, 容量 10万条, < 10ms
│ (Redis+Faiss)│
└──────┬───────┘
       │ 未命中
       ▼
┌──────────────┐
│ L3: 持久缓存   │  ← MySQL/PostgreSQL, 全量, < 50ms
│ (DB)         │
└──────┬───────┘
       │ 未命中
       ▼
调用 LLM
```

### 4.2 实现

```python
class TieredCache:
    """三级缓存"""
    
    def __init__(self):
        # L1: 进程内存 (Python LRU)
        from functools import lru_cache
        self.l1_cache = {}
        self.l1_max_size = 1000
        
        # L2: Redis + Faiss 语义缓存
        self.l2_cache = SemanticCache(
            similarity_threshold=0.85,
            max_cache_size=100000
        )
        
        # L3: 数据库持久化
        self.l3_db = None  # 使用 ORM 连接
    
    def get(self, query: str) -> tuple:
        """逐级查询缓存"""
        
        # L1: 精确匹配 (哈希)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.l1_cache:
            cached, timestamp = self.l1_cache[query_hash]
            if time.time() - timestamp < 300:  # 5分钟TTL
                return cached, True, "L1_exact"
        
        # L2: 语义匹配
        result, score = self.l2_cache.search(query)
        if result:
            # 回填 L1
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.l1_cache[query_hash] = (result, time.time())
            return result, True, f"L2_semantic({score:.2f})"
        
        # L3: 数据库查询
        # ... (用于冷数据)
        
        return None, False, "miss"
    
    def put(self, query: str, response: dict):
        """写入所有层级"""
        # L1
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.l1_cache[query_hash] = (response, time.time())
        
        # L2
        self.l2_cache.store(query, response)
        
        # L3 (异步写入)
        # self.l3_db.insert_async(...)
```

---

## 5. 前缀缓存与 KV Cache

### 5.1 前缀缓存原理

```
请求A: System Prompt (2000 tokens) + "查询设备A的告警"
请求B: System Prompt (2000 tokens) + "查询设备B的告警"
                                      ↑
                        相同的 System Prompt (2000 tokens)
                        可以复用 KV Cache!
```

前缀缓存保存 LLM 对相同前缀的计算结果（KV Cache），避免重复计算：

```python
class PrefixCache:
    """前缀缓存 - 复用 System Prompt 的 KV Cache"""
    
    def __init__(self, max_prefix_length: int = 4096):
        self.cache = {}  # prefix_hash → KV Cache
        self.max_length = max_prefix_length
    
    def get_prefix_kv(self, messages: list) -> tuple:
        """提取 System Prompt 的 KV Cache"""
        # 计算 System Prompt 的哈希
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"]
        
        if not system_content:
            return None, None
        
        prefix_hash = hashlib.sha256(system_content.encode()).hexdigest()
        
        if prefix_hash in self.cache:
            return self.cache[prefix_hash], prefix_hash
        
        return None, prefix_hash
    
    def store_prefix_kv(self, prefix_hash: str, kv_cache, ttl: int = 3600):
        """存储 KV Cache"""
        self.cache[prefix_hash] = {
            "kv_cache": kv_cache,
            "timestamp": time.time(),
            "ttl": ttl
        }
```

> **限制**: KV Cache 有显存开销。以 Llama-3-70B 为例，每层 KV Cache 约 2.6MB，80层约为 208MB/请求。需要在缓存效益和显存占用间平衡。

### 5.2 多层 Token 缓存对比

| 方法 | 粒度 | 命中率 | 延迟节省 | 复杂度 |
|------|------|--------|---------|--------|
| 精确匹配缓存 | 请求级 | 15-30% | 100% | 低 |
| 语义缓存 | 请求级 | 30-50% | 100% | 中 |
| 前缀缓存 (KV) | Token级 | 40-60% | 50-80% | 高 |
| 投机解码 | Token级 | N/A | 30-50% | 很高 |

---

## 6. 缓存策略与失效管理

### 6.1 TTL 策略

```python
class CacheTTLStrategy:
    """缓存 TTL 策略"""
    
    # 不同场景的 TTL
    ttls = {
        "knowledge_qa": 3600,      # 知识问答: 1小时 (知识变化慢)
        "real_time_data": 60,      # 实时数据: 1分钟
        "code_generation": 7200,   # 代码生成: 2小时
        "translation": 86400,      # 翻译: 24小时 (结果不变)
        "alarm_query": 300,        # 告警查询: 5分钟
    }
    
    def get_ttl(self, request_type: str, data_freshness_requirement: str) -> int:
        base_ttl = self.ttls.get(request_type, 3600)
        
        # 根据数据新鲜度要求调整
        freshness_multiplier = {
            "realtime": 0.1,   # 10% of base TTL
            "hourly": 0.5,
            "daily": 1.0,
            "weekly": 7.0,
            "static": 30.0,    # 基本不变
        }
        
        return int(base_ttl * freshness_multiplier.get(data_freshness_requirement, 1.0))
```

### 6.2 缓存失效触发

```yaml
invalidation_triggers:
  # 数据变更驱动
  - trigger: "device_status_changed"
    action: "invalidate_prefix"
    prefix: "semcache:device:{device_id}"
    
  - trigger: "knowledge_base_updated"
    action: "invalidate_all"
    reason: "知识库更新，所有RAG缓存可能过期"
  
  # 时间驱动
  - trigger: "scheduled"
    cron: "0 2 * * *"  # 凌晨 2 点
    action: "invalidate_pattern"
    pattern: "semcache:alarm:*"
    reason: "每日告警数据刷新"
  
  # 质量驱动
  - trigger: "hallucination_detected"
    action: "invalidate_exact"
    key: "{affected_cache_key}"
    reason: "缓存返回了错误结果"
```

---

## 7. 生产级部署

### 7.1 缓存监控

```python
class CacheMonitor:
    """缓存监控"""
    
    def get_metrics(self) -> dict:
        return {
            # 性能指标
            "l1_hit_rate": self.calc_l1_hit_rate(),      # L1命中率
            "l2_hit_rate": self.calc_l2_hit_rate(),      # L2语义命中率
            "overall_hit_rate": self.calc_overall_hit_rate(),
            
            # 成本指标
            "cost_saved_today": self.calc_cost_saved(),   # 今日节省
            "llm_calls_avoided": self.calc_calls_avoided(),
            
            # 质量指标
            "cache_accuracy": self.calc_cache_accuracy(), # 缓存准确率
            "stale_rate": self.calc_stale_rate(),         # 过期率
            "false_positive_rate": self.calc_false_positive(),
            
            # 容量指标
            "cache_size": self.get_cache_size(),
            "memory_usage": self.get_memory_usage(),
            "eviction_rate": self.calc_eviction_rate(),
        }
```

### 7.2 生产 Checklist

- [ ] 相似度阈值通过 A/B 测试确定
- [ ] 缓存命中/未命中 Metrics 接入 Prometheus
- [ ] 缓存准确率持续监控（抽样对比 LLM 直接回答）
- [ ] TTL 策略按场景差异化配置
- [ ] 数据变更事件触发缓存失效
- [ ] 缓存大小有上限，淘汰策略已测试
- [ ] Redis 持久化配置 (RDB + AOF)
- [ ] 缓存预热策略 (冷启动时预加载高频查询)
- [ ] 缓存击穿保护 (热点数据永不过期 + 互斥锁)
- [ ] 缓存穿透保护 (空值缓存，TTL 较短)

---

---

## 深度分析

语义缓存是LLM应用中性价比最高的优化手段，在几乎零质量损失的前提下可实现30-50%的成本节省。其核心原理是将用户查询映射到向量空间，通过近似最近邻搜索找到语义相似的已缓存结果。与精确缓存不同，语义缓存能够处理"查一下设备A的告警"和"设备A有什么告警"这种同义不同文的查询，大幅提升了缓存命中率。实现这一能力的关键在于Embedding模型的精度和相似度阈值的选择——阈值过高则命中率低，过低则可能导致缓存误命中。

多级缓存架构是生产级语义缓存的推荐模式。L1进程内存提供精确匹配的亚毫秒级响应，L2 Redis+Faiss提供语义匹配的毫秒级检索，L3数据库提供持久化全量存储。这种分层设计在延迟、命中率和成本之间取得了平衡。缓存失效策略是容易被忽视但至关重要的环节——数据变更驱动的精确失效、时间驱动的批量失效、质量驱动的异常失效三种模式需要结合实际业务场景组合使用。

前缀缓存和KV Cache优化代表了Token级别的缓存范式。与请求级语义缓存不同，前缀缓存复用的是LLM计算的中间状态（KV Cache），能够节省Prefill阶段50-80%的计算量。vLLM的PagedAttention和SGLang的RadixAttention实现了自动化的前缀检测和共享，在System Prompt长的场景下效果尤为显著。生产部署时需要权衡KV Cache的显存占用和缓存收益，对于Llama-3-70B级别的大模型，每层KV Cache约2.6MB，部署前需精确估算。

## Checklist

- [ ] 缓存ROI评估：分析业务查询的重复比例，量化语义缓存的预期成本节省
- [ ] 相似度阈值调优：通过A/B测试确定最优阈值，建议从0.85开始逐步调整
- [ ] 多级缓存部署：实现L1进程内存+L2 Redis+Faiss+L3数据库三层缓存架构
- [ ] 缓存失效策略：配置数据变更驱动/时间驱动/质量驱动三级失效触发机制
- [ ] TTL策略差异化：按场景（知识问答/实时数据/代码生成/翻译）配置不同的TTL
- [ ] 前缀缓存配置：启用vLLM prefix caching或SGLang RadixAttention，估算显存开销
- [ ] 缓存监控接入：监控命中率/节省成本/缓存准确率/过期率/误命中率
- [ ] 生产安全措施：配置缓存击穿保护（互斥锁+热点永不过期）和穿透保护（空值短TTL）
- [ ] 缓存预热策略：冷启动时从数据库加载高频查询预热语义缓存

## 延伸阅读

- [02-模型服务与推理优化.md](./02-模型服务与推理优化.md) — Prefix Caching与RadixAttention
- [05-向量数据库架构深度.md](./05-向量数据库架构深度.md) — Faiss索引优化与量化压缩
- [07-流式AI与实时推理架构.md](./07-流式AI与实时推理架构.md) — 流式推理中的缓存策略
- [04-Embedding模型选型与实践.md](../03-RAG架构设计/04-Embedding模型选型与实践.md) — 嵌入模型选型与相似度计算
- GPTCache官方文档 — 语义缓存框架使用指南

# 前沿RAG架构

> 2025-2026年最前沿RAG架构模式深度解析

## 目录
1. [GraphRAG — 知识图谱增强检索](#1-graphrag--知识图谱增强检索)
2. [Agentic RAG — Agent驱动的自适应检索](#2-agentic-rag--agent驱动的自适应检索)
3. [多模态RAG — 跨模态检索与理解](#3-多模态rag--跨模态检索与理解)
4. [RAG评估体系 — 量化质量保障](#4-rag评估体系--量化质量保障)
5. [RAG前沿优化 — 检索精度与效率突破](#5-rag前沿优化--检索精度与效率突破)

---

## 1. GraphRAG — 知识图谱增强检索

### 1.1 微软GraphRAG架构原理

微软GraphRAG的核心洞察：标准RAG擅长局部细节检索，但在全局性、跨文档的推理问题上表现薄弱。GraphRAG通过构建知识图谱 + 社区层次摘要，实现从"文档片段检索"到"知识网络推理"的跃迁。

**架构总览：**

```
┌──────────────────────────────────────────────────────────────┐
│                    GraphRAG 索引流水线                         │
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ 原始文档 │───→│ LLM实体/关系  │───→│   知识图谱构建     │   │
│  │ (Chunks) │    │   抽取       │    │ (Entities+Edges)  │   │
│  └─────────┘    └──────────────┘    └────────┬──────────┘   │
│                                              │               │
│                                     ┌────────▼──────────┐   │
│                                     │   社区检测算法      │   │
│                                     │ (Leiden算法)       │   │
│                                     └────────┬──────────┘   │
│                                              │               │
│                              ┌───────────────┼───────────┐  │
│                              ▼               ▼           ▼  │
│                     ┌─────────────┐  ┌────────────┐  ┌────┐│
│                     │社区层次摘要   │  │实体/关系表  │  │文本││
│                     │(多级Community│  │(Parquet)   │  │单元││
│                     │ Summaries)  │  │            │  │映射││
│                     └─────────────┘  └────────────┘  └────┘│
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    GraphRAG 查询流水线                         │
│                                                              │
│  用户Query                                                   │
│      │                                                       │
│      ├──→ 全局查询: 遍历社区摘要 → 聚合多社区答案 → 最终答案  │
│      │                                                       │
│      └──→ 局部查询: 实体识别 → 子图检索 → 局部上下文 → 答案   │
└──────────────────────────────────────────────────────────────┘
```

**两阶段核心流程：**

| 阶段 | 目标 | 输入 | 输出 |
|------|------|------|------|
| **索引阶段** | 构建知识图谱与社区摘要 | 原始文档集 | 知识图谱 + 多级社区摘要 |
| **查询阶段** | 基于图谱结构回答问题 | 用户查询 | 带来源的全局/局部答案 |

### 1.2 知识图谱构建

#### 实体抽取

```python
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import json

class Entity(BaseModel):
    name: str
    type: str
    description: str

class Relationship(BaseModel):
    source: str
    target: str
    relation: str
    description: str
    weight: float = 1.0

class GraphExtractor:
    """基于LLM的实体与关系抽取器"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    EXTRACTION_PROMPT = """从以下文本中抽取实体和关系。

文本:
{text}

请以JSON格式返回:
{{
    "entities": [
        {{"name": "实体名", "type": "人物/组织/地点/概念/事件", "description": "描述"}}
    ],
    "relationships": [
        {{"source": "源实体", "target": "目标实体", "relation": "关系类型", "description": "关系描述"}}
    ]
}}

要求:
1. 实体名统一规范化（如"张三"和"老张"统一为同一实体）
2. 关系类型使用动词短语（如"担任"、"投资"、"位于"）
3. 抽取粒度适中，避免过度碎片化"""

    def extract(self, text: str) -> dict:
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)

    def extract_from_chunks(self, chunks: List[str], batch_size: int = 5) -> dict:
        all_entities = []
        all_relationships = []

        for chunk in chunks:
            result = self.extract(chunk)
            all_entities.extend(result.get("entities", []))
            all_relationships.extend(result.get("relationships", []))

        return {
            "entities": self._deduplicate_entities(all_entities),
            "relationships": self._deduplicate_relationships(all_relationships)
        }

    def _deduplicate_entities(self, entities: List[dict]) -> List[dict]:
        seen = {}
        for e in entities:
            key = e["name"].lower().strip()
            if key not in seen:
                seen[key] = e
            else:
                seen[key]["description"] += f"; {e['description']}"
        return list(seen.values())

    def _deduplicate_relationships(self, relationships: List[dict]) -> List[dict]:
        seen = set()
        unique = []
        for r in relationships:
            key = (r["source"].lower(), r["relation"].lower(), r["target"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
```

#### 关系构建与图谱存储

```python
import networkx as nx
from dataclasses import dataclass, field

@dataclass
class KnowledgeGraph:
    """知识图谱数据结构"""
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    entity_index: dict = field(default_factory=dict)

    def add_entity(self, entity: dict):
        self.graph.add_node(
            entity["name"],
            type=entity.get("type", "unknown"),
            description=entity.get("description", "")
        )
        self.entity_index[entity["name"].lower()] = entity["name"]

    def add_relationship(self, rel: dict):
        self.graph.add_edge(
            rel["source"],
            rel["target"],
            relation=rel.get("relation", "related_to"),
            description=rel.get("description", ""),
            weight=rel.get("weight", 1.0)
        )

    def get_subgraph(self, entity: str, hops: int = 2) -> nx.DiGraph:
        if entity not in self.graph:
            entity = self.entity_index.get(entity.lower(), entity)
        if entity not in self.graph:
            return nx.DiGraph()

        nodes = {entity}
        frontier = {entity}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.successors(node):
                    if neighbor not in nodes:
                        next_frontier.add(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in nodes:
                        next_frontier.add(neighbor)
            nodes.update(next_frontier)
            frontier = next_frontier

        return self.graph.subgraph(nodes).copy()

    def get_entity_context(self, entity: str, max_neighbors: int = 20) -> str:
        subgraph = self.get_subgraph(entity, hops=1)
        context_parts = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            context_parts.append(f"{node}({node_data.get('type', '')}): {node_data.get('description', '')}")
        for u, v, data in subgraph.edges(data=True):
            context_parts.append(f"{u} --[{data.get('relation', '')}]--> {v}: {data.get('description', '')}")
        return "\n".join(context_parts[:max_neighbors])
```

### 1.3 社区检测与摘要

GraphRAG的关键创新：使用Leiden社区检测算法将知识图谱划分为层次化社区，每个社区生成摘要，从而支持全局性问题的回答。

```python
import community as community_louvain

class CommunityDetector:
    """社区检测与层次化摘要"""

    def __init__(self, kg: KnowledgeGraph, llm_client: OpenAI):
        self.kg = kg
        self.client = llm_client
        self.communities = {}
        self.community_summaries = {}

    def detect_communities(self, resolution: float = 1.0) -> dict:
        undirected = self.kg.graph.to_undirected()
        partition = community_louvain.best_partition(
            undirected,
            resolution=resolution
        )
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        self.communities = communities
        return communities

    def generate_community_summary(self, community_id: int, members: List[str]) -> str:
        subgraph = self.kg.graph.subgraph(members)
        context_lines = []
        for node in subgraph.nodes():
            data = subgraph.nodes[node]
            context_lines.append(f"- 实体: {node} | 类型: {data.get('type', '')} | 描述: {data.get('description', '')}")
        for u, v, data in subgraph.edges(data=True):
            context_lines.append(f"- 关系: {u} --[{data.get('relation', '')}]--> {v}")

        prompt = f"""请为以下知识图谱社区生成结构化摘要。

社区成员及关系:
{chr(10).join(context_lines[:200])}

请生成:
1. 社区主题（一句话概括）
2. 核心实体列表
3. 关键关系总结
4. 社区整体描述（2-3句话）"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def build_all_summaries(self) -> dict:
        communities = self.detect_communities()
        for cid, members in communities.items():
            self.community_summaries[cid] = self.generate_community_summary(cid, members)
        return self.community_summaries

    def build_hierarchical_summaries(self, levels: int = 3) -> dict:
        hierarchical = {}
        for level in range(levels):
            resolution = 0.5 * (level + 1)
            communities = self.detect_communities(resolution=resolution)
            level_summaries = {}
            for cid, members in communities.items():
                level_summaries[cid] = self.generate_community_summary(cid, members)
            hierarchical[f"level_{level}"] = {
                "resolution": resolution,
                "community_count": len(communities),
                "summaries": level_summaries
            }
        return hierarchical
```

### 1.4 全局查询 vs 局部查询

```
┌─────────────────────────────────────────────────────────────┐
│                    查询路由决策                               │
│                                                             │
│  用户Query                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────┐    全局性问题        ┌───────────────────┐    │
│  │ 查询分类  │───────────────────→ │   全局查询引擎     │    │
│  │          │  ("整体趋势是什么？")  │                   │    │
│  │          │                     │ 1. 遍历社区摘要     │    │
│  │          │                     │ 2. Map-Reduce聚合   │    │
│  │          │                     │ 3. 中间答案融合     │    │
│  │          │                     └───────────────────┘    │
│  │          │                                              │
│  │          │    局部性问题        ┌───────────────────┐    │
│  │          │───────────────────→ │   局部查询引擎     │    │
│  └──────────┘  ("张三的职位？")    │                   │    │
│                                 │ 1. 实体识别         │    │
│                                 │ 2. 子图检索         │    │
│                                 │ 3. 局部上下文构建    │    │
│                                 └───────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

```python
from enum import Enum
from typing import List, Dict

class QueryType(Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"

class GraphRAGQueryEngine:
    """GraphRAG查询引擎 - 支持全局与局部查询"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        community_summaries: Dict,
        llm_client: OpenAI
    ):
        self.kg = kg
        self.community_summaries = community_summaries
        self.client = llm_client

    def classify_query(self, query: str) -> QueryType:
        prompt = f"""判断以下查询的类型:
- GLOBAL: 需要全局视角、整体趋势、跨文档综合（如"主要主题有哪些？"）
- LOCAL: 针对特定实体或关系的局部查询（如"张三的职位？"）
- HYBRID: 同时需要全局和局部信息

查询: {query}

只返回 GLOBAL/LOCAL/HYBRID 之一。"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().upper()
        return QueryType(result if result in [e.value for e in QueryType] else "LOCAL")

    def global_search(self, query: str) -> str:
        intermediate_answers = []
        for cid, summary in self.community_summaries.items():
            prompt = f"""基于以下社区摘要回答问题。如果社区摘要与问题无关，回复"无法回答"。

社区摘要:
{summary}

问题: {query}

请给出简短答案。"""
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            answer = response.choices[0].message.content
            if "无法回答" not in answer:
                intermediate_answers.append(answer)

        if not intermediate_answers:
            return "未找到相关信息。"

        fusion_prompt = f"""基于以下多个社区的局部答案，综合生成最终答案。

局部答案:
{chr(10).join(f'- {a}' for a in intermediate_answers)}

原始问题: {query}

请综合以上信息，给出完整、一致的答案。"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fusion_prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def local_search(self, query: str, hops: int = 2) -> str:
        entities = self._extract_query_entities(query)
        contexts = []
        for entity in entities:
            context = self.kg.get_entity_context(entity, max_neighbors=30)
            if context:
                contexts.append(context)

        if not contexts:
            return "未找到相关实体。"

        prompt = f"""基于以下知识图谱上下文回答问题。

上下文:
{chr(10).join(contexts)}

问题: {query}

请基于上下文给出准确答案，并引用相关实体和关系。"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def query(self, query: str) -> str:
        query_type = self.classify_query(query)
        if query_type == QueryType.GLOBAL:
            return self.global_search(query)
        elif query_type == QueryType.LOCAL:
            return self.local_search(query)
        else:
            global_answer = self.global_search(query)
            local_answer = self.local_search(query)
            return f"【全局视角】\n{global_answer}\n\n【局部细节】\n{local_answer}"

    def _extract_query_entities(self, query: str) -> List[str]:
        entities = []
        for node in self.kg.graph.nodes():
            if node in query or node.lower() in query.lower():
                entities.append(node)
        return entities
```

### 1.5 GraphRAG vs 标准RAG对比

| 维度 | 标准RAG | GraphRAG |
|------|---------|----------|
| **知识表示** | 扁平文档片段 | 结构化知识图谱 + 社区层次 |
| **全局性问题** | ❌ 难以回答 | ✅ 社区摘要支持全局推理 |
| **多跳推理** | ❌ 依赖片段拼接 | ✅ 图遍历天然支持 |
| **实体关系** | ❌ 隐含在文本中 | ✅ 显式建模 |
| **索引成本** | 低（向量索引） | 高（LLM抽取 + 图构建） |
| **查询延迟** | 低（毫秒级） | 高（多轮LLM调用） |
| **更新维护** | 增量更新容易 | 图更新需重新社区检测 |
| **适用场景** | 事实性问答、文档检索 | 关系推理、全局分析、知识发现 |
| **Token消耗** | 中等 | 高（索引阶段大量LLM调用） |

**选型建议：**

```
是否需要全局性分析？
├── 是 → GraphRAG
│   └── 数据量是否>10万文档？
│       ├── 是 → 分层GraphRAG + 增量社区更新
│       └── 否 → 标准GraphRAG
└── 否 → 是否需要多跳推理？
    ├── 是 → GraphRAG + 局部查询
    └── 否 → 标准RAG / Advanced RAG
```

---

## 2. Agentic RAG — Agent驱动的自适应检索

### 2.1 Agent驱动的自适应检索

Agentic RAG将RAG从"被动检索-生成"管道升级为"主动规划-检索-反思"循环。核心区别：标准RAG是数据流管道，Agentic RAG是具有自主决策能力的智能体。

```
┌─────────────────────────────────────────────────────────────┐
│                  Agentic RAG 架构                            │
│                                                             │
│  ┌─────────┐                                                │
│  │ 用户请求 │                                                │
│  └────┬────┘                                                │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Agent 控制器 (ReAct循环)                  │   │
│  │                                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │ 规划模块  │→│ 执行模块  │→│ 反思模块  │──┐        │   │
│  │  │ Planner  │  │ Executor │  │ Reflector│  │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘  │        │   │
│  │       ↑                                     │        │   │
│  │       └─────────────────────────────────────┘        │   │
│  │              (迭代直到满意)                            │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│          ┌──────────────┼──────────────┐                    │
│          ▼              ▼              ▼                    │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐            │
│   │ 检索工具    │ │ 计算工具    │ │ 外部API     │            │
│   │ (多源RAG)  │ │(Calculator)│ │(Web Search) │            │
│   └────────────┘ └────────────┘ └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 查询规划与分解

```python
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import OpenAI
import json

class SubQuery(BaseModel):
    query: str
    tool: str
    priority: int = 0
    depends_on: List[int] = []

class QueryPlan(BaseModel):
    original_query: str
    sub_queries: List[SubQuery]
    strategy: str

class QueryPlanner:
    """查询规划器 - 将复杂查询分解为可执行的子查询"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    PLAN_PROMPT = """你是一个查询规划专家。将用户的复杂查询分解为可执行的子查询计划。

可用工具:
- vector_search: 向量语义检索，适合语义相似性查询
- keyword_search: 关键词精确检索，适合专有名词、编号查询
- graph_search: 知识图谱检索，适合关系推理、多跳查询
- web_search: 网络搜索，适合实时信息、最新动态
- calculator: 数值计算，适合数学运算

用户查询: {query}

请生成执行计划，以JSON格式返回:
{{
    "strategy": "sequential/parallel/mixed",
    "sub_queries": [
        {{
            "query": "子查询内容",
            "tool": "工具名称",
            "priority": 0,
            "depends_on": []
        }}
    ]
}}

注意:
- priority越小越先执行
- depends_on列出依赖的子查询索引（0-based）
- 无依赖的子查询可并行执行"""

    def plan(self, query: str) -> QueryPlan:
        prompt = self.PLAN_PROMPT.format(query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        return QueryPlan(
            original_query=query,
            sub_queries=[SubQuery(**sq) for sq in result["sub_queries"]],
            strategy=result["strategy"]
        )

    def get_execution_order(self, plan: QueryPlan) -> List[List[SubQuery]]:
        ordered = []
        completed = set()
        remaining = list(plan.sub_queries)

        while remaining:
            ready = [
                sq for sq in remaining
                if all(d in completed for d in sq.depends_on)
            ]
            if not ready:
                ready = [remaining[0]]
            ordered.append(sorted(ready, key=lambda x: x.priority))
            for sq in ready:
                completed.add(plan.sub_queries.index(sq))
                remaining.remove(sq)

        return ordered
```

### 2.3 迭代检索与反思

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievalState:
    query: str
    iteration: int = 0
    max_iterations: int = 3
    retrieved_docs: List[Dict] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 0.0
    is_sufficient: bool = False

class IterativeRetriever:
    """迭代检索器 - 检索-反思-再检索循环"""

    def __init__(self, retriever, llm_client: OpenAI, max_iterations: int = 3):
        self.retriever = retriever
        self.client = llm_client
        self.max_iterations = max_iterations

    def retrieve_with_reflection(self, query: str) -> RetrievalState:
        state = RetrievalState(query=query, max_iterations=self.max_iterations)

        while state.iteration < state.max_iterations and not state.is_sufficient:
            state.iteration += 1

            search_query = self._formulate_query(state)
            new_docs = self.retriever.search(search_query, top_k=5)
            state.retrieved_docs.extend(new_docs)

            state = self._reflect(state)

        return state

    def _formulate_query(self, state: RetrievalState) -> str:
        if state.iteration == 1:
            return state.query

        prompt = f"""原始查询: {state.query}

已检索到的信息:
{self._format_docs(state.retrieved_docs)}

之前的反思:
{chr(10).join(state.reflections)}

基于以上反思，生成一个改进的检索查询，以弥补信息缺口。
只返回查询文本，不要解释。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def _reflect(self, state: RetrievalState) -> RetrievalState:
        prompt = f"""原始查询: {state.query}

已检索到的信息:
{self._format_docs(state.retrieved_docs)}

请评估:
1. 当前信息是否足以回答原始查询？
2. 如果不足，缺少什么信息？
3. 给出置信度(0-1)

以JSON格式返回:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reflection": "反思内容",
    "missing_info": "缺少的信息描述"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)

        state.is_sufficient = result["is_sufficient"]
        state.confidence = result["confidence"]
        state.reflections.append(result["reflection"])

        return state

    def _format_docs(self, docs: List[Dict]) -> str:
        return "\n".join(
            f"[{i+1}] {doc.get('content', doc.get('text', ''))[:300]}"
            for i, doc in enumerate(docs[-10:])
        )
```

### 2.4 多源检索编排

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

class MultiSourceRetriever:
    """多源检索编排器"""

    def __init__(self, max_workers: int = 5):
        self.sources: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def register_source(self, name: str, retriever: Callable, weight: float = 1.0):
        self.sources[name] = retriever
        self.weights[name] = weight

    async def search_parallel(self, query: str, top_k: int = 10) -> List[Dict]:
        loop = asyncio.get_event_loop()
        tasks = {
            name: loop.run_in_executor(self.executor, retriever, query, top_k * 2)
            for name, retriever in self.sources.items()
        }

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await asyncio.wait_for(
                    asyncio.ensure_future(task), timeout=10.0
                )
            except Exception as e:
                results[name] = []

        return self._fuse_results(results, top_k)

    def _fuse_results(self, all_results: Dict[str, List], top_k: int) -> List[Dict]:
        doc_scores = {}
        k = 60

        for source, results in all_results.items():
            weight = self.weights.get(source, 1.0)
            for rank, result in enumerate(results, start=1):
                doc_id = result.get("doc_id", result.get("id", f"{source}_{rank}"))
                rrf_score = weight / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0,
                        "sources": [],
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["sources"].append(source)

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:top_k]

        return [
            {
                "doc_id": doc_id,
                "content": data["content"],
                "score": data["score"],
                "sources": data["sources"],
                "metadata": data["metadata"]
            }
            for doc_id, data in sorted_docs
        ]
```

### 2.5 Self-RAG架构

Self-RAG让LLM在生成过程中自主决定是否需要检索、评估检索结果的相关性、验证生成内容的支撑度。

```
┌─────────────────────────────────────────────────────────────┐
│                   Self-RAG 生成流程                          │
│                                                             │
│  输入: Query                                                │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────┐                                        │
│  │ [Retrieve]判断   │──→ 需要? ──→ 执行检索 → 获取文档D     │
│  │ 是否需要检索     │──→ 不需要 → 直接生成                   │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ [IsRel]判断      │──→ 相关? ──→ 继续生成                 │
│  │ 检索结果是否相关  │──→ 不相关 → 重新检索或忽略            │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ 生成Segment      │                                        │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ [IsSup]判断      │──→ 有支撑? → 保留                     │
│  │ 生成内容是否有    │──→ 无支撑 → 重新生成/补充检索          │
│  │ 检索文档支撑     │                                        │
│  └────────┬────────┘                                        │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │ [IsUse]判断      │──→ 有用? → 输出                       │
│  │ 生成内容是否有用  │──→ 无用 → 重新生成                    │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

```python
from enum import Enum
from typing import Tuple, Optional

class ReflectionToken(Enum):
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    IS_REL = "is_rel"
    IS_SUP = "is_sup"
    IS_USE = "is_use"

class SelfRAG:
    """Self-RAG: 自反思检索增强生成"""

    def __init__(self, llm_client: OpenAI, retriever):
        self.client = llm_client
        self.retriever = retriever

    def generate(self, query: str, max_segments: int = 5) -> Dict:
        segments = []
        all_docs = []

        for _ in range(max_segments):
            need_retrieve = self._judge_retrieve(query, segments)
            docs = []

            if need_retrieve:
                docs = self.retriever.search(query, top_k=5)
                is_rel = self._judge_relevance(query, docs)
                if not is_rel:
                    docs = []

            segment = self._generate_segment(query, segments, docs)
            is_sup = self._judge_support(segment, docs) if docs else True
            is_use = self._judge_usefulness(query, segment)

            if is_sup and is_use:
                segments.append({
                    "text": segment,
                    "retrieved": len(docs) > 0,
                    "supported": is_sup,
                    "useful": is_use
                })
                all_docs.extend(docs)

            if self._is_complete(query, segments):
                break

        return {
            "answer": " ".join(s["text"] for s in segments),
            "segments": segments,
            "source_docs": all_docs,
            "retrieval_count": sum(1 for s in segments if s["retrieved"])
        }

    def _judge_retrieve(self, query: str, existing_segments: List[Dict]) -> bool:
        context = " ".join(s["text"] for s in existing_segments) if existing_segments else ""
        prompt = f"""判断回答以下问题是否需要检索外部信息。

问题: {query}
已生成内容: {context}

如果已有知识足以回答，回复NO_RETRIEVE。
如果需要检索外部信息，回复RETRIEVE。
只回复一个词。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "RETRIEVE" in response.choices[0].message.content.upper()

    def _judge_relevance(self, query: str, docs: List[Dict]) -> bool:
        docs_text = "\n".join(d.get("content", "")[:200] for d in docs[:3])
        prompt = f"""判断以下检索结果是否与问题相关。

问题: {query}
检索结果: {docs_text}

相关回复RELEVANT，不相关回复IRRELEVANT。只回复一个词。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "RELEVANT" in response.choices[0].message.content.upper()

    def _judge_support(self, segment: str, docs: List[Dict]) -> bool:
        docs_text = "\n".join(d.get("content", "")[:200] for d in docs[:3])
        prompt = f"""判断以下生成内容是否有检索文档的支撑。

生成内容: {segment}
检索文档: {docs_text}

有支撑回复SUPPORTED，无支撑回复UNSUPPORTED。只回复一个词。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "SUPPORTED" in response.choices[0].message.content.upper()

    def _judge_usefulness(self, query: str, segment: str) -> bool:
        prompt = f"""判断以下生成内容对回答问题是否有用。

问题: {query}
生成内容: {segment}

有用回复USEFUL，无用回复USELESS。只回复一个词。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "USEFUL" in response.choices[0].message.content.upper()

    def _generate_segment(self, query: str, existing: List[Dict], docs: List[Dict]) -> str:
        context = " ".join(s["text"] for s in existing) if existing else ""
        docs_text = "\n".join(d.get("content", "")[:300] for d in docs[:3]) if docs else "无检索结果"

        prompt = f"""基于以下信息继续回答问题。

问题: {query}
已有内容: {context}
检索文档: {docs_text}

请继续生成下一段回答（不要重复已有内容）。"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def _is_complete(self, query: str, segments: List[Dict]) -> bool:
        if len(segments) == 0:
            return False
        full_answer = " ".join(s["text"] for s in segments)
        prompt = f"""判断以下回答是否已经完整回答了问题。

问题: {query}
回答: {full_answer}

完整回复COMPLETE，不完整回复INCOMPLETE。只回复一个词。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "COMPLETE" in response.choices[0].message.content.upper()
```

### Agentic RAG模式对比

| 模式 | 核心机制 | 适用场景 | 复杂度 | Token消耗 |
|------|----------|----------|--------|-----------|
| **Self-RAG** | 生成中自评估检索需求 | 高可信度问答 | 中 | 中 |
| **Corrective RAG** | 检索后纠错补充 | 开放域问答 | 低 | 低 |
| **Adaptive RAG** | 查询复杂度路由 | 混合复杂度场景 | 低 | 低 |
| **Full Agentic RAG** | 规划-执行-反思循环 | 复杂多步推理 | 高 | 高 |
| **Iterative RAG** | 检索-反思迭代 | 信息缺口大的场景 | 中 | 中 |

---

## 3. 多模态RAG — 跨模态检索与理解

### 3.1 图文混合检索

```
┌─────────────────────────────────────────────────────────────┐
│                  多模态RAG架构                                │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  文本块   │  │  图片块   │  │  表格块   │  │  视频帧   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│       ▼              ▼              ▼              ▼         │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Text    │  │ CLIP/    │  │ Table    │  │ Video    │    │
│  │Embedding│  │ SigLIP   │  │Encoder   │  │Encoder   │    │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │         │
│       └──────────────┴──────┬───────┴──────────────┘        │
│                             ▼                                │
│                  ┌─────────────────────┐                     │
│                  │   统一向量空间        │                     │
│                  │ (多模态Embedding)     │                     │
│                  └──────────┬──────────┘                     │
│                             ▼                                │
│                  ┌─────────────────────┐                     │
│                  │  跨模态检索 + 重排序  │                     │
│                  └──────────┬──────────┘                     │
│                             ▼                                │
│                  ┌─────────────────────┐                     │
│                  │  多模态LLM生成       │                     │
│                  │ (GPT-4o/Gemini)     │                     │
│                  └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from PIL import Image
import base64
import io

@dataclass
class MultiModalChunk:
    chunk_id: str
    modality: str  # text/image/table/video_frame
    content: str
    image: Optional[Image.Image] = None
    embedding: Optional[List[float]] = None
    metadata: Dict = None

class MultiModalIndexer:
    """多模态文档索引器"""

    def __init__(self, text_model, image_model, table_model=None):
        self.text_model = text_model
        self.image_model = image_model
        self.table_model = table_model

    def index_document(self, doc_path: str) -> List[MultiModalChunk]:
        chunks = []
        elements = self._parse_document(doc_path)

        for elem in elements:
            if elem["type"] == "text":
                embedding = self.text_model.encode([elem["content"]])[0]
                chunks.append(MultiModalChunk(
                    chunk_id=elem["id"],
                    modality="text",
                    content=elem["content"],
                    embedding=embedding.tolist()
                ))
            elif elem["type"] == "image":
                image = Image.open(io.BytesIO(elem["bytes"]))
                embedding = self.image_model.encode(image)
                description = self._describe_image(image)
                chunks.append(MultiModalChunk(
                    chunk_id=elem["id"],
                    modality="image",
                    content=description,
                    image=image,
                    embedding=embedding.tolist(),
                    metadata={"original_path": elem.get("path", "")}
                ))
            elif elem["type"] == "table":
                table_text = self._table_to_text(elem)
                embedding = self.text_model.encode([table_text])[0]
                chunks.append(MultiModalChunk(
                    chunk_id=elem["id"],
                    modality="table",
                    content=table_text,
                    embedding=embedding.tolist()
                ))

        return chunks

    def _describe_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容，包括关键信息、数据、文字等。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            temperature=0.0
        )
        return response.choices[0].message.content

    def _table_to_text(self, elem: dict) -> str:
        rows = elem.get("rows", [])
        if not rows:
            return ""
        header = " | ".join(str(c) for c in rows[0])
        body_lines = []
        for row in rows[1:]:
            body_lines.append(" | ".join(str(c) for c in row))
        return f"表格标题: {elem.get('caption', '未知')}\n列: {header}\n" + "\n".join(body_lines)

    def _parse_document(self, doc_path: str) -> List[Dict]:
        return []
```

### 3.2 表格与图表理解

```python
class TableChartRAG:
    """表格与图表理解RAG"""

    def __init__(self, llm_client: OpenAI, text_embedder):
        self.client = llm_client
        self.embedder = text_embedder

    def extract_table(self, image: Image.Image) -> Dict:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = """请从图片中提取表格数据，以JSON格式返回:
{
    "caption": "表格标题",
    "headers": ["列1", "列2", ...],
    "rows": [["值1", "值2", ...], ...],
    "summary": "表格内容摘要",
    "key_insights": ["关键发现1", "关键发现2"]
}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        import json
        return json.loads(response.choices[0].message.content)

    def answer_table_query(self, query: str, tables: List[Dict]) -> str:
        table_contexts = []
        for i, t in enumerate(tables):
            headers = " | ".join(t.get("headers", []))
            rows_str = "\n".join(
                " | ".join(str(c) for c in row)
                for row in t.get("rows", [])
            )
            table_contexts.append(f"表格{i+1}: {t.get('caption', '')}\n{headers}\n{rows_str}\n摘要: {t.get('summary', '')}")

        prompt = f"""基于以下表格数据回答问题。

{chr(10).join(table_contexts)}

问题: {query}

请给出准确答案，必要时引用具体数据。"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
```

### 3.3 视频检索

```python
import cv2
from typing import List, Tuple

class VideoRAG:
    """视频检索RAG"""

    def __init__(self, llm_client: OpenAI, image_embedder, text_embedder):
        self.client = llm_client
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder

    def extract_keyframes(
        self,
        video_path: str,
        sample_fps: float = 1.0,
        scene_threshold: float = 30.0
    ) -> List[Tuple[float, Image.Image]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_fps)

        keyframes = []
        prev_frame = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray).mean()
                    if diff < scene_threshold and frame_count % (frame_interval * 5) != 0:
                        frame_count += 1
                        continue

                timestamp = frame_count / fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                keyframes.append((timestamp, pil_image))
                prev_frame = gray

            frame_count += 1

        cap.release()
        return keyframes

    def index_video(self, video_path: str) -> List[Dict]:
        keyframes = self.extract_keyframes(video_path)
        indexed_frames = []

        for timestamp, frame in keyframes:
            description = self._describe_frame(frame)
            embedding = self.image_embedder.encode(frame)

            indexed_frames.append({
                "timestamp": timestamp,
                "description": description,
                "embedding": embedding.tolist(),
                "video_path": video_path
            })

        return indexed_frames

    def search_video(
        self,
        query: str,
        indexed_frames: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        query_embedding = self.text_embedder.encode([query])[0]

        import numpy as np
        scores = []
        for frame in indexed_frames:
            frame_emb = np.array(frame["embedding"])
            similarity = np.dot(query_embedding, frame_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(frame_emb) + 1e-8
            )
            scores.append((similarity, frame))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "timestamp": frame["timestamp"],
                "description": frame["description"],
                "score": float(score),
                "video_path": frame["video_path"]
            }
            for score, frame in scores[:top_k]
        ]

    def _describe_frame(self, frame: Image.Image) -> str:
        buffered = io.BytesIO()
        frame.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "简洁描述这个视频帧的内容，包括场景、人物、动作、文字等关键信息。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            temperature=0.0
        )
        return response.choices[0].message.content
```

### 3.4 多模态嵌入模型选型

| 模型 | 模态支持 | 向量维度 | 中文支持 | 最佳场景 | 特点 |
|------|----------|----------|----------|----------|------|
| **CLIP (ViT-L/14)** | 文本+图像 | 768 | 一般 | 通用图文检索 | 开源生态成熟 |
| **SigLIP-SO400M** | 文本+图像 | 1152 | 较好 | 精细图文匹配 | Sigmoid替代Softmax，性能更优 |
| **Jina-CLIP-v2** | 文本+图像 | 1024 | ✅ 优秀 | 中英图文检索 | 多语言原生支持 |
| **BGE-visualized** | 文本+图像 | 1024 | ✅ 优秀 | 中文图文检索 | 中文场景最佳 |
| **2B-GE** | 文本+图像+表格 | 2048 | 较好 | 文档级多模态 | 支持表格理解 |
| **LanguageBind** | 文本+图像+视频+音频 | 1024 | 一般 | 全模态检索 | 统一5种模态 |
| **ImageBind** | 6种模态 | 1024 | 一般 | 全模态对齐 | Meta出品，模态最全 |

**选型决策：**

```
模态需求?
├── 仅文本+图像
│   ├── 中文为主 → BGE-visualized / Jina-CLIP-v2
│   └── 英文为主 → SigLIP-SO400M
├── 需要视频检索 → LanguageBind
├── 需要全模态 → ImageBind
└── 需要表格理解 → 2B-GE / GPT-4o表格提取
```

---

## 4. RAG评估体系 — 量化质量保障

### 4.1 RAGAS框架

RAGAS（Retrieval Augmented Generation Assessment）是RAG领域最主流的评估框架，从检索质量和生成质量两个维度量化评估。

```
┌─────────────────────────────────────────────────────────────┐
│                    RAGAS 评估维度                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              上下文质量 (检索侧)                      │   │
│  │                                                     │   │
│  │  Context Precision    Context Recall                │   │
│  │  ┌───────────────┐   ┌───────────────┐             │   │
│  │  │检索结果中相关   │   │必要信息是否被   │             │   │
│  │  │文档的排名位置   │   │完整检索到      │             │   │
│  │  │越高越好        │   │越高越好        │             │   │
│  │  └───────────────┘   └───────────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              生成质量 (生成侧)                        │   │
│  │                                                     │   │
│  │  Faithfulness         Answer Relevancy              │   │
│  │  ┌───────────────┐   ┌───────────────┐             │   │
│  │  │答案是否忠实于   │   │答案是否真正     │             │   │
│  │  │检索到的上下文   │   │回答了用户问题   │             │   │
│  │  │无幻觉          │   │切题            │             │   │
│  │  └───────────────┘   └───────────────┘             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**四大核心指标详解：**

| 指标 | 评估目标 | 输入 | 计算方式 | 理想值 |
|------|----------|------|----------|--------|
| **Faithfulness** | 答案忠实度 | question, contexts, answer | 将答案拆分为claim，验证每个claim是否可被contexts支撑 | 1.0 |
| **Answer Relevancy** | 答案相关性 | question, answer | 从answer反向生成问题，计算与原question的语义相似度 | 1.0 |
| **Context Precision** | 上下文精确率 | question, contexts | 相关文档在检索结果中的排名位置加权 | 1.0 |
| **Context Recall** | 上下文召回率 | question, contexts, ground_truth | ground_truth中的信息是否被contexts覆盖 | 1.0 |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

class RAGASEvaluator:
    """RAGAS评估器"""

    def __init__(self, llm=None, embeddings=None):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def evaluate_rag(
        self,
        test_data: List[Dict]
    ) -> Dict:
        dataset_dict = {
            "question": [d["question"] for d in test_data],
            "answer": [d["answer"] for d in test_data],
            "contexts": [d["contexts"] for d in test_data],
            "ground_truth": [d["ground_truth"] for d in test_data]
        }
        dataset = Dataset.from_dict(dataset_dict)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )

        return result

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict:
        test_data = [{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }]
        return self.evaluate_rag(test_data)

    def generate_test_data(
        self,
        documents: List[str],
        num_questions: int = 20
    ) -> List[Dict]:
        from openai import OpenAI
        client = OpenAI()
        test_data = []

        for doc in documents[:num_questions]:
            prompt = f"""基于以下文档内容，生成一个问答对。

文档: {doc[:2000]}

请以JSON格式返回:
{{
    "question": "基于文档的问题",
    "ground_truth": "基于文档的标准答案"
}}"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            import json
            qa = json.loads(response.choices[0].message.content)
            test_data.append({
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "answer": "",
                "contexts": []
            })

        return test_data
```

### 4.2 自动化评估流水线

```python
import time
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)

class RAGEvalPipeline:
    """RAG自动化评估流水线"""

    def __init__(
        self,
        rag_pipeline: Callable,
        evaluator: RAGASEvaluator,
        test_dataset: List[Dict]
    ):
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator
        self.test_dataset = test_dataset
        self.results: List[EvalResult] = []

    def run_evaluation(self) -> Dict:
        rag_outputs = []

        for sample in self.test_dataset:
            start = time.time()
            rag_result = self.rag_pipeline(sample["question"])
            latency = (time.time() - start) * 1000

            output = {
                "question": sample["question"],
                "answer": rag_result.get("answer", ""),
                "contexts": rag_result.get("contexts", []),
                "ground_truth": sample["ground_truth"],
                "latency_ms": latency
            }
            rag_outputs.append(output)

        eval_results = self.evaluator.evaluate_rag(rag_outputs)

        for i, sample in enumerate(rag_outputs):
            self.results.append(EvalResult(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                ground_truth=sample["ground_truth"],
                latency_ms=sample["latency_ms"]
            ))

        return self._aggregate_results(eval_results)

    def _aggregate_results(self, eval_results) -> Dict:
        latencies = [r.latency_ms for r in self.results]

        return {
            "total_samples": len(self.results),
            "metrics": {
                "faithfulness": eval_results.get("faithfulness", 0),
                "answer_relevancy": eval_results.get("answer_relevancy", 0),
                "context_precision": eval_results.get("context_precision", 0),
                "context_recall": eval_results.get("context_recall", 0)
            },
            "latency": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                "p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
                "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
            },
            "per_sample": [
                {
                    "question": r.question,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ]
        }

    def compare_pipelines(self, other_pipeline: Callable, label: str = "B") -> Dict:
        baseline = self.run_evaluation()

        original_pipeline = self.rag_pipeline
        self.rag_pipeline = other_pipeline
        self.results = []
        comparison = self.run_evaluation()
        self.rag_pipeline = original_pipeline

        return {
            "baseline": baseline["metrics"],
            f"pipeline_{label}": comparison["metrics"],
            "delta": {
                k: comparison["metrics"][k] - baseline["metrics"][k]
                for k in baseline["metrics"]
            }
        }
```

### 4.3 A/B测试与持续优化

```python
import hashlib
import random
from collections import defaultdict

class RAGABTest:
    """RAG系统A/B测试框架"""

    def __init__(
        self,
        pipeline_a: Callable,
        pipeline_b: Callable,
        traffic_split: float = 0.5
    ):
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.traffic_split = traffic_split
        self.results = defaultdict(list)

    def _get_variant(self, user_id: str) -> str:
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "A" if (hash_val % 100) < (self.traffic_split * 100) else "B"

    def query(self, question: str, user_id: str = None) -> Dict:
        if user_id is None:
            user_id = str(random.randint(0, 1000000))

        variant = self._get_variant(user_id)
        pipeline = self.pipeline_a if variant == "A" else self.pipeline_b

        start = time.time()
        result = pipeline(question)
        latency = (time.time() - start) * 1000

        self.results[variant].append({
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
            "latency_ms": latency,
            "timestamp": time.time()
        })

        result["variant"] = variant
        result["latency_ms"] = latency
        return result

    def get_stats(self) -> Dict:
        stats = {}
        for variant in ["A", "B"]:
            data = self.results[variant]
            if not data:
                stats[variant] = {"count": 0}
                continue

            latencies = [d["latency_ms"] for d in data]
            stats[variant] = {
                "count": len(data),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2]
            }

        return stats

    def evaluate_with_ragas(self, evaluator: RAGASEvaluator) -> Dict:
        results = {}
        for variant in ["A", "B"]:
            data = self.results[variant]
            if not data:
                continue

            test_data = [
                {
                    "question": d["question"],
                    "answer": d["answer"],
                    "contexts": d["contexts"],
                    "ground_truth": ""
                }
                for d in data[:50]
            ]
            results[variant] = evaluator.evaluate_rag(test_data)

        return results


class ContinuousOptimizer:
    """持续优化器 - 基于评估反馈自动调参"""

    def __init__(self, rag_pipeline, evaluator: RAGASEvaluator):
        self.pipeline = rag_pipeline
        self.evaluator = evaluator
        self.optimization_history = []

    def optimize_chunk_size(
        self,
        test_data: List[Dict],
        chunk_sizes: List[int] = [128, 256, 512, 1024]
    ) -> Dict:
        results = {}

        for chunk_size in chunk_sizes:
            self.pipeline.set_chunk_size(chunk_size)

            eval_data = []
            for sample in test_data:
                rag_result = self.pipeline.query(sample["question"])
                eval_data.append({
                    "question": sample["question"],
                    "answer": rag_result.get("answer", ""),
                    "contexts": rag_result.get("contexts", []),
                    "ground_truth": sample["ground_truth"]
                })

            eval_result = self.evaluator.evaluate_rag(eval_data)
            results[chunk_size] = eval_result

            self.optimization_history.append({
                "parameter": "chunk_size",
                "value": chunk_size,
                "metrics": eval_result
            })

        best_size = max(
            results.keys(),
            key=lambda k: sum(results[k].get(m, 0) for m in ["faithfulness", "answer_relevancy"])
        )

        return {
            "best_chunk_size": best_size,
            "all_results": results,
            "recommendation": f"推荐chunk_size={best_size}，综合指标最优"
        }

    def optimize_top_k(
        self,
        test_data: List[Dict],
        top_k_values: List[int] = [3, 5, 10, 15, 20]
    ) -> Dict:
        results = {}

        for top_k in top_k_values:
            self.pipeline.set_top_k(top_k)

            eval_data = []
            for sample in test_data:
                rag_result = self.pipeline.query(sample["question"])
                eval_data.append({
                    "question": sample["question"],
                    "answer": rag_result.get("answer", ""),
                    "contexts": rag_result.get("contexts", []),
                    "ground_truth": sample["ground_truth"]
                })

            eval_result = self.evaluator.evaluate_rag(eval_data)
            results[top_k] = eval_result

        best_k = max(
            results.keys(),
            key=lambda k: results[k].get("context_recall", 0) * 0.5 + results[k].get("faithfulness", 0) * 0.5
        )

        return {
            "best_top_k": best_k,
            "all_results": results
        }
```

### 评估指标速查表

| 指标 | 评估侧 | 好的阈值 | 差的信号 | 优化方向 |
|------|--------|----------|----------|----------|
| **Faithfulness < 0.7** | 生成 | ≥0.85 | 幻觉严重 | 加强Prompt约束、减少上下文噪声 |
| **Answer Relevancy < 0.7** | 生成 | ≥0.80 | 答非所问 | 优化Prompt、查询重写 |
| **Context Precision < 0.6** | 检索 | ≥0.75 | 检索噪声大 | 优化Embedding、添加Reranker |
| **Context Recall < 0.6** | 检索 | ≥0.80 | 信息遗漏 | 增大Top-K、查询扩展、混合检索 |

---

## 5. RAG前沿优化 — 检索精度与效率突破

### 5.1 Late Interaction模型（ColBERT）

ColBERT的核心创新：不再将整个文档压缩为单一向量，而是保留Token级别的细粒度交互，通过MaxSim操作实现精确匹配。

```
┌─────────────────────────────────────────────────────────────┐
│            ColBERT Late Interaction 原理                     │
│                                                             │
│  传统Bi-Encoder:                                            │
│  Query → [CLS] → 单一向量 (768维)                           │
│  Doc   → [CLS] → 单一向量 (768维)                           │
│  相似度 = cos(q_vec, d_vec)  ← 丢失细粒度信息               │
│                                                             │
│  ColBERT Late Interaction:                                  │
│  Query → [Q1][Q2]...[Qn] → 多向量 (n × 128维)              │
│  Doc   → [D1][D2]...[Dm] → 多向量 (m × 128维)              │
│                                                             │
│  相似度 = Σ_max(cos(Qi, Dj))  ← 每个Query Token            │
│           i  j                找到最佳匹配Doc Token          │
│                                                             │
│  ┌─────────────────────────────────────────────┐           │
│  │  Q1 ──max──→ D3  (0.92)                     │           │
│  │  Q2 ──max──→ D7  (0.85)                     │           │
│  │  Q3 ──max──→ D1  (0.78)                     │           │
│  │  Q4 ──max──→ D5  (0.91)                     │           │
│  │  ─────────────────────────                   │           │
│  │  ColBERT Score = (0.92+0.85+0.78+0.91)/4    │           │
│  │                 = 0.865                       │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np

class ColBERTRetriever:
    """ColBERT Late Interaction检索器"""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.doc_embeddings = {}
        self.doc_ids = []

    def encode_query(self, query: str, max_length: int = 32) -> np.ndarray:
        inputs = self.tokenizer(
            query,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.squeeze(0).numpy()

    def encode_document(self, doc: str, max_length: int = 180) -> np.ndarray:
        inputs = self.tokenizer(
            doc,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.squeeze(0).numpy()

    def index_documents(self, documents: List[str], doc_ids: List[str]):
        for doc_id, doc in zip(doc_ids, documents):
            self.doc_embeddings[doc_id] = self.encode_document(doc)
        self.doc_ids = list(self.doc_embeddings.keys())

    def maxsim_score(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        similarity_matrix = np.dot(query_emb, doc_emb.T)
        max_sim_per_query = np.max(similarity_matrix, axis=1)
        return float(np.mean(max_sim_per_query))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_emb = self.encode_query(query)

        scores = []
        for doc_id in self.doc_ids:
            doc_emb = self.doc_embeddings[doc_id]
            score = self.maxsim_score(query_emb, doc_emb)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in scores[:top_k]
        ]
```

**ColBERT vs Bi-Encoder vs Cross-Encoder对比：**

| 维度 | Bi-Encoder | ColBERT | Cross-Encoder |
|------|-----------|---------|---------------|
| **交互粒度** | 文档级 | Token级 | Token级(全交叉) |
| **检索方式** | ANN近似检索 | ANN + 精排 | 逐对打分 |
| **精度** | 中 | 高 | 最高 |
| **速度** | 最快 | 中 | 最慢 |
| **存储** | 低(1向量/文档) | 高(N向量/文档) | 无索引 |
| **适用阶段** | 粗排 | 精排 | 终排 |
| **推荐组合** | 粗排候选 | 精排Top-K | 小集终排 |

### 5.2 混合检索新策略

2025-2026年混合检索的关键进化：从简单的"向量+BM25"双路召回，走向多策略、自适应的智能融合。

```
┌─────────────────────────────────────────────────────────────┐
│              新一代混合检索架构                                │
│                                                             │
│  Query                                                      │
│    │                                                        │
│    ▼                                                        │
│  ┌──────────────┐                                           │
│  │ 查询分析器    │──→ 意图/实体/复杂度/模态需求               │
│  └──────┬───────┘                                           │
│         │                                                   │
│    ┌────┼────────────┬──────────────┐                       │
│    ▼    ▼            ▼              ▼                       │
│  ┌───┐ ┌───┐  ┌──────────┐  ┌──────────┐                  │
│  │稠密│ │稀疏│  │ColBERT   │  │知识图谱   │                  │
│  │向量│ │向量│  │Late Int. │  │检索      │                  │
│  └─┬─┘ └─┬─┘  └────┬─────┘  └────┬─────┘                  │
│    │     │          │              │                         │
│    └─────┴──────────┴──────────────┘                         │
│              │                                               │
│              ▼                                               │
│  ┌──────────────────────┐                                    │
│  │ 自适应融合引擎        │                                    │
│  │ (动态权重 + RRF)      │                                    │
│  └──────────┬───────────┘                                    │
│             ▼                                                │
│  ┌──────────────────────┐                                    │
│  │ 级联重排序            │                                    │
│  │ ColBERT → Cross-Enc  │                                    │
│  └──────────┬───────────┘                                    │
│             ▼                                                │
│        最终Top-K结果                                         │
└─────────────────────────────────────────────────────────────┘
```

```python
from typing import Dict, List, Optional
import numpy as np

class AdaptiveHybridRetriever:
    """自适应混合检索器"""

    def __init__(self):
        self.retrievers: Dict[str, object] = {}
        self.weights: Dict[str, float] = {}
        self.default_weights = {
            "dense": 1.0,
            "sparse": 0.8,
            "colbert": 1.2,
            "graph": 0.9
        }

    def register(self, name: str, retriever, weight: float = None):
        self.retrievers[name] = retriever
        self.weights[name] = weight or self.default_weights.get(name, 1.0)

    def _analyze_query(self, query: str) -> Dict:
        features = {
            "has_named_entity": self._has_named_entity(query),
            "is_factual": self._is_factual(query),
            "needs_reasoning": self._needs_reasoning(query),
            "query_length": len(query.split()),
            "has_comparison": any(w in query for w in ["对比", "比较", "区别", "vs", "相比"])
        }

        weights = {}
        if features["has_named_entity"]:
            weights["sparse"] = 1.5
            weights["dense"] = 0.8
        if features["needs_reasoning"]:
            weights["graph"] = 1.5
            weights["colbert"] = 1.3
        if features["is_factual"]:
            weights["dense"] = 1.2
        if features["has_comparison"]:
            weights["colbert"] = 1.5
            weights["graph"] = 1.2

        for name in self.retrievers:
            if name not in weights:
                weights[name] = self.weights.get(name, 1.0)

        return {"features": features, "adaptive_weights": weights}

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        analysis = self._analyze_query(query)
        adaptive_weights = analysis["adaptive_weights"]

        all_results = {}
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.search(query, top_k=top_k * 3)
                all_results[name] = results
            except Exception:
                all_results[name] = []

        fused = self._weighted_rrf(all_results, adaptive_weights, top_k)
        return fused

    def _weighted_rrf(
        self,
        all_results: Dict[str, List],
        weights: Dict[str, float],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        doc_scores = {}

        for source, results in all_results.items():
            weight = weights.get(source, 1.0)
            for rank, result in enumerate(results, start=1):
                doc_id = result.get("doc_id", result.get("id", f"{source}_{rank}"))
                rrf_score = weight / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "score": 0,
                        "sources": [],
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                doc_scores[doc_id]["score"] += rrf_score
                doc_scores[doc_id]["sources"].append(source)

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:top_k]

        return [
            {
                "doc_id": doc_id,
                "content": data["content"],
                "score": data["score"],
                "sources": data["sources"],
                "metadata": data["metadata"]
            }
            for doc_id, data in sorted_docs
        ]

    def _has_named_entity(self, query: str) -> bool:
        import re
        patterns = [
            r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+',
            r'[\u4e00-\u9fff]{2,4}(?:公司|集团|银行|大学|医院)',
            r'\d{4}年',
            r'[A-Z]{2,5}\d+'
        ]
        return any(re.search(p, query) for p in patterns)

    def _is_factual(self, query: str) -> bool:
        factual_keywords = ["什么是", "多少", "何时", "在哪", "是谁", "what is", "how many", "when"]
        return any(kw in query.lower() for kw in factual_keywords)

    def _needs_reasoning(self, query: str) -> bool:
        reasoning_keywords = ["为什么", "如何", "原因", "影响", "关系", "why", "how", "because", "impact"]
        return any(kw in query.lower() for kw in reasoning_keywords)
```

### 5.3 上下文压缩与选择

```python
from openai import OpenAI
import json

class ContextCompressor:
    """上下文压缩器 - 减少噪声、保留关键信息"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def compress(self, query: str, contexts: List[str], max_tokens: int = 2000) -> List[str]:
        compressed = []
        total_tokens = 0

        scored = self._score_relevance(query, contexts)
        scored.sort(key=lambda x: x[1], reverse=True)

        for context, score in scored:
            estimated_tokens = len(context) // 4
            if total_tokens + estimated_tokens > max_tokens:
                extracted = self._extract_relevant_sentences(query, context, max_tokens - total_tokens)
                if extracted:
                    compressed.append(extracted)
                break

            compressed.append(context)
            total_tokens += estimated_tokens

        return compressed

    def _score_relevance(self, query: str, contexts: List[str]) -> List[tuple]:
        scored = []
        for ctx in contexts:
            prompt = f"""评估以下文本与查询的相关性，给出0-1的分数。

查询: {query}
文本: {ctx[:500]}

只返回0到1之间的数字。"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            try:
                score = float(response.choices[0].message.content.strip())
            except ValueError:
                score = 0.5
            scored.append((ctx, score))

        return scored

    def _extract_relevant_sentences(self, query: str, context: str, max_tokens: int) -> str:
        prompt = f"""从以下文本中提取与查询最相关的句子，总长度不超过{max_tokens * 4}个字符。

查询: {query}
文本: {context}

只返回提取的句子，不要添加任何解释。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content


class ContextSelector:
    """上下文选择器 - 基于多样性和相关性选择最优上下文组合"""

    def __init__(self, embedding_model):
        self.embedder = embedding_model

    def select(
        self,
        query: str,
        contexts: List[str],
        top_k: int = 5,
        diversity_weight: float = 0.3
    ) -> List[str]:
        query_emb = self.embedder.encode([query])[0]
        ctx_embs = self.embedder.encode(contexts)

        relevance_scores = np.dot(ctx_embs, query_emb)
        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)

        selected_indices = []
        for _ in range(min(top_k, len(contexts))):
            best_idx = -1
            best_score = -float("inf")

            for i in range(len(contexts)):
                if i in selected_indices:
                    continue

                score = (1 - diversity_weight) * relevance_scores[i]

                if selected_indices:
                    similarities = np.dot(ctx_embs[selected_indices], ctx_embs[i])
                    max_sim = np.max(similarities)
                    diversity_penalty = diversity_weight * max_sim
                    score -= diversity_penalty

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)

        return [contexts[i] for i in selected_indices]
```

### 5.4 长文档RAG策略

```
┌─────────────────────────────────────────────────────────────┐
│               长文档RAG策略对比                               │
│                                                             │
│  策略1: 滑动窗口分块                                        │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                                 │
│  │C1│C2│C3│C4│C5│C6│C7│C8│  固定大小，重叠窗口             │
│  └──┴──┴──┴──┴──┴──┴──┴──┘  简单但可能切断语义             │
│                                                             │
│  策略2: 语义分块                                            │
│  ┌─────┐ ┌───────┐ ┌────┐ ┌────────┐                      │
│  │ Seg1│ │ Seg2  │ │Seg3│ │ Seg4   │  按语义边界分割       │
│  └─────┘ └───────┘ └────┘ └────────┘  保留完整语义         │
│                                                             │
│  策略3: 分层摘要                                            │
│  ┌─────────────────────────────────┐                        │
│  │     文档级摘要 (L0)             │  全局概览               │
│  ├────────┬────────┬───────────────┤                        │
│  │章摘要1 │章摘要2 │章摘要3        │  章节级摘要             │
│  ├────┬───┼────┬───┼────┬──────────┤                        │
│  │段1 │段2│段3 │段4│段5 │段6       │  段落级原文             │
│  └────┴───┴────┴───┴────┴──────────┘                        │
│                                                             │
│  策略4: 父子文档检索                                        │
│  ┌──────────────────────────────┐                           │
│  │  父文档 (大块，用于生成)      │                           │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ │                           │
│  │  │子块1 │ │子块2 │ │子块3 │ │  子块(小块，用于检索)      │
│  │  └──────┘ └──────┘ └──────┘ │                           │
│  └──────────────────────────────┘                           │
│  检索子块 → 返回父文档 → 完整上下文生成                      │
└─────────────────────────────────────────────────────────────┘
```

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    chunk_id: str
    parent_id: str = ""
    level: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict = None

class LongDocumentRAG:
    """长文档RAG - 多策略支持"""

    def __init__(self, text_embedder, llm_client: OpenAI):
        self.embedder = text_embedder
        self.client = llm_client

    def sliding_window_chunk(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[DocumentChunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(DocumentChunk(
                content=chunk_text,
                chunk_id=f"chunk_{idx}",
                start_char=start,
                end_char=min(end, len(text))
            ))
            start += chunk_size - overlap
            idx += 1
        return chunks

    def semantic_chunk(self, text: str, threshold: float = 0.85) -> List[DocumentChunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = self.embedder.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_start = 0

        for i in range(1, len(sentences)):
            similarity = np.dot(embeddings[i], embeddings[i - 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-8
            )

            if similarity < threshold:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"sem_chunk_{len(chunks)}",
                    start_char=current_start,
                    end_char=current_start + len(chunk_text)
                ))
                current_chunk = [sentences[i]]
                current_start += len(" ".join(current_chunk[:-1])) + 1 if len(current_chunk) > 1 else 0
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(DocumentChunk(
                content=" ".join(current_chunk),
                chunk_id=f"sem_chunk_{len(chunks)}",
                start_char=current_start,
                end_char=len(text)
            ))

        return chunks

    def hierarchical_chunk(
        self,
        text: str,
        doc_title: str = ""
    ) -> List[DocumentChunk]:
        chunks = []

        doc_summary = self._summarize(text, max_length=300)
        chunks.append(DocumentChunk(
            content=doc_summary,
            chunk_id="doc_summary",
            level=0,
            metadata={"type": "document_summary", "title": doc_title}
        ))

        sections = self._split_sections(text)
        for i, section in enumerate(sections):
            section_summary = self._summarize(section["content"], max_length=150)
            chunks.append(DocumentChunk(
                content=section_summary,
                chunk_id=f"section_summary_{i}",
                level=1,
                metadata={"type": "section_summary", "heading": section.get("heading", "")}
            ))

            paragraphs = self._split_into_paragraphs(section["content"], max_chars=800)
            for j, para in enumerate(paragraphs):
                chunks.append(DocumentChunk(
                    content=para,
                    chunk_id=f"para_{i}_{j}",
                    parent_id=f"section_summary_{i}",
                    level=2,
                    metadata={"type": "paragraph", "section": section.get("heading", "")}
                ))

        return chunks

    def parent_child_chunk(
        self,
        text: str,
        parent_size: int = 2000,
        child_size: int = 300,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        chunks = []
        parent_idx = 0
        start = 0

        while start < len(text):
            parent_end = min(start + parent_size, len(text))
            parent_text = text[start:parent_end]
            parent_id = f"parent_{parent_idx}"

            chunks.append(DocumentChunk(
                content=parent_text,
                chunk_id=parent_id,
                level=0,
                start_char=start,
                end_char=parent_end,
                metadata={"type": "parent"}
            ))

            child_start = start
            child_idx = 0
            while child_start < parent_end:
                child_end = min(child_start + child_size, parent_end)
                child_text = text[child_start:child_end]

                chunks.append(DocumentChunk(
                    content=child_text,
                    chunk_id=f"child_{parent_idx}_{child_idx}",
                    parent_id=parent_id,
                    level=1,
                    start_char=child_start,
                    end_char=child_end,
                    metadata={"type": "child"}
                ))

                child_start += child_size - overlap
                child_idx += 1

            start += parent_size
            parent_idx += 1

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_sections(self, text: str) -> List[Dict]:
        import re
        pattern = r'(?=^#{1,3}\s+.+)|(?=^第[一二三四五六七八九十]+[章节])'
        parts = re.split(pattern, text, flags=re.MULTILINE)
        sections = []
        for part in parts:
            if part and part.strip():
                lines = part.strip().split("\n")
                heading = lines[0] if lines else ""
                content = "\n".join(lines[1:]) if len(lines) > 1 else part
                sections.append({"heading": heading[:100], "content": content})
        if not sections:
            sections = [{"heading": "全文", "content": text}]
        return sections

    def _split_into_paragraphs(self, text: str, max_chars: int = 800) -> List[str]:
        paragraphs = text.split("\n\n")
        result = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > max_chars and current:
                result.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        if current.strip():
            result.append(current.strip())
        return result

    def _summarize(self, text: str, max_length: int = 200) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"请用{max_length}字以内概括以下内容的关键信息:\n\n{text[:3000]}"
            }],
            temperature=0.0,
            max_tokens=max_length
        )
        return response.choices[0].message.content
```

### 长文档策略选型对比

| 策略 | 分块粒度 | 上下文完整性 | 检索精度 | 存储成本 | 最佳场景 |
|------|----------|-------------|----------|----------|----------|
| **滑动窗口** | 固定大小 | ⭐⭐ | ⭐⭐⭐ | 低 | 通用场景、快速原型 |
| **语义分块** | 语义边界 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | 高质量要求 |
| **分层摘要** | 多级层次 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 高(摘要LLM调用) | 超长文档、全局理解 |
| **父子文档** | 双层大小 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | 兼顾精度与完整性 |

---

## 总结

### 前沿RAG架构演进路线

```
2023                    2024                     2025-2026
  │                       │                         │
  ▼                       ▼                         ▼
Standard RAG          Advanced RAG             Agentic RAG
  │                    │  │  │                    │  │  │
  │                    │  │  └→ Corrective RAG    │  │  └→ Full Agent
  │                    │  │                       │  │
  │                    │  └→ Self-RAG             │  └→ Self-RAG v2
  │                    │                          │
  │                    └→ Modular RAG             └→ GraphRAG + Agent
  │                                               │
  └→ Multi-Modal RAG (基础)                       └→ Multi-Modal RAG (深度)
                                                  │
                                                  └→ ColBERT + 自适应融合
```

### 架构选型速查

| 需求场景 | 推荐架构 | 关键技术 |
|----------|----------|----------|
| 全局性分析、趋势洞察 | GraphRAG | 知识图谱 + 社区摘要 |
| 复杂多步推理 | Agentic RAG | 规划-执行-反思循环 |
| 高可信度问答 | Self-RAG | 生成中自评估 |
| 图文混合检索 | 多模态RAG | CLIP/SigLIP + 多模态LLM |
| 精细语义匹配 | ColBERT精排 | Late Interaction |
| 超长文档理解 | 分层摘要 + 父子文档 | 多级索引 |
| 量化质量保障 | RAGAS评估体系 | 自动化评估流水线 |
| 生产环境持续优化 | A/B测试 + 持续优化 | 自适应调参 |

---

*最后更新：2026-05-12*

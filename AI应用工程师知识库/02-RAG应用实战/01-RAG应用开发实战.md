# RAG应用开发实战

> 从零构建生产级RAG应用

## 1. RAG应用架构总览

### 1.1 完整RAG流水线

```
┌─────────────────────────────────────────────────────────────┐
│                      离线索引阶段                             │
│                                                              │
│  文档源 → 文档加载 → 文本分块 → 向量化 → 向量数据库存储      │
│  (PDF/    (Unstruct-  (Recursive   (Embedding  (Milvus/     │
│   Word/    ured/       Splitter)    Model)     Pinecone)    │
│   Web)     LlamaParse)                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      在线查询阶段                             │
│                                                              │
│  用户Query → 查询处理 → 向量检索 → 重排序 → LLM生成 → 答案   │
│             (重写/扩展)  (Top-K)    (Rerank)  (带引用)       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型矩阵

| 组件 | 推荐方案 | 备选方案 |
|------|----------|----------|
| **应用框架** | LangChain / LlamaIndex | Haystack / Semantic Kernel |
| **文档解析** | Unstructured / LlamaParse | PyMuPDF / pdfplumber |
| **文本分块** | RecursiveCharacterTextSplitter | SemanticChunker |
| **Embedding** | text-embedding-3-large / BGE-M3 | Cohere Embed / Jina |
| **向量数据库** | Milvus / pgvector | Pinecone / Weaviate / Chroma |
| **重排序** | Cohere Rerank / BGE-Reranker | Cross-encoder |
| **LLM** | GPT-4o / Qwen-72B | Claude-3.5 / DeepSeek |
| **编排** | LangGraph / Spring AI | 自研 |

---

## 2. 文档处理

### 2.1 文档加载

**支持格式：**
- PDF（扫描件/文本）
- Word/Excel/PPT
- Markdown/HTML
- 网页（爬取）
- 数据库（SQL）
- API（JSON）

**LangChain文档加载器：**
```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
    TextLoader,
    CSVLoader
)

# PDF加载
pdf_loader = PyPDFLoader("document.pdf")
pages = pdf_loader.load()

# 网页加载
web_loader = WebBaseLoader("https://example.com/docs")
pages = web_loader.load()

# 目录批量加载
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="./docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
```

**高级文档解析（LlamaParse）：**
```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-xxx",
    result_type="markdown",  # markdown / text
    verbose=True
)

# 解析文档，保留表格和图片
documents = parser.load_data("complex_document.pdf")
```

### 2.2 文本分块策略

**策略对比：**

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **固定长度** | 按字符数切分 | 简单可控 | 可能切断语义 |
| **递归分割** | 按分隔符层级切分 | 保留结构 | 需调参 |
| **语义分割** | 按语义相似度切分 | 语义完整 | 计算开销大 |
| **文档结构** | 按标题/段落切分 | 结构清晰 | 依赖文档格式 |

**递归分割（推荐）：**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 每块最大字符数
    chunk_overlap=50,        # 块间重叠字符数
    separators=["\n\n", "\n", "。", ".", " ", ""],  # 分隔符优先级
    length_function=len,
)

chunks = splitter.split_documents(documents)
print(f"原始文档: {len(documents)} 页 → 分块: {len(chunks)} 块")
```

**语义分割：**
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # percentile / standard_deviation / interquartile
    breakpoint_threshold_amount=95,
)

chunks = semantic_splitter.split_documents(documents)
```

**分块参数调优建议：**

| 场景 | chunk_size | overlap | 说明 |
|------|------------|---------|------|
| 问答 | 300-500 | 50 | 精确匹配 |
| 摘要 | 1000-2000 | 200 | 保留完整上下文 |
| 代码 | 500-1000 | 100 | 保留函数完整性 |
| 法律 | 800-1200 | 150 | 保留条款完整性 |

### 2.3 元数据管理

```python
from datetime import datetime

# 为每个chunk添加元数据
for i, chunk in enumerate(chunks):
    chunk.metadata.update({
        "source": chunk.metadata.get("source", "unknown"),
        "chunk_id": f"{chunk.metadata['source']}_chunk_{i}",
        "chunk_index": i,
        "created_at": datetime.now().isoformat(),
        "page": chunk.metadata.get("page", 0),
        "doc_type": "pdf",  # pdf/html/docx
        "language": "zh",
    })
```

---

## 3. 向量化与存储

### 3.1 Embedding模型选型

| 模型 | 维度 | 多语言 | 特点 | 价格 |
|------|------|--------|------|------|
| **OpenAI text-embedding-3-large** | 3072 | ✅ | 效果好 | $0.13/1M tokens |
| **OpenAI text-embedding-3-small** | 1536 | ✅ | 性价比高 | $0.02/1M tokens |
| **BGE-M3** | 1024 | ✅ | 开源、多语言 | 免费 |
| **Cohere Embed v3** | 1024 | ✅ | 多语言优化 | 按用量 |
| **Jina Embeddings v2** | 768 | ✅ | 长文本支持 | 按用量 |

**使用OpenAI Embedding：**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536  # 可降维
)

# 测试
vector = embeddings.embed_query("什么是RAG？")
print(f"向量维度: {len(vector)}")
```

**使用开源BGE-M3：**
```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
```

### 3.2 向量数据库选型

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Milvus** | 开源/云 | 功能全面、性能好 | 企业级生产 |
| **pgvector** | PostgreSQL扩展 | 基于现有PG | 已有PG基础设施 |
| **Pinecone** | 云服务 | 全托管、易用 | 快速启动 |
| **Weaviate** | 开源/云 | 混合搜索 | 语义+关键词 |
| **Chroma** | 开源 | 轻量、嵌入式 | 开发测试 |
| **Qdrant** | 开源 | Rust实现、高性能 | 高性能需求 |

**Milvus集成：**
```python
from langchain_community.vectorstores import Milvus

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_args={
        "host": "localhost",
        "port": "19530"
    },
    collection_name="knowledge_base",
    drop_old=True  # 开发时重建
)
```

**pgvector集成：**
```python
from langchain_community.vectorstores import PGVector

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string="postgresql://user:pass@localhost:5432/vectordb",
    collection_name="knowledge_base",
    pre_collection_name="knowledge_base"  # 使用已有表
)
```

### 3.3 混合检索

**向量+关键词混合：**
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 向量检索
vector_retriever = vectorstore.as_retriever(
    search_type="mmr",       # 最大边际相关性
    search_kwargs={"k": 10, "fetch_k": 20}
)

# BM25关键词检索
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 向量70% + 关键词30%
)
```

---

## 4. 查询处理

### 4.1 查询重写

**问题：** 用户Query可能表述不清

**方案1：LLM重写**
```python
query_rewrite_prompt = """
请将以下用户问题改写为更适合检索的形式。
要求：
1. 补充隐含信息
2. 使用更精确的术语
3. 保持原意不变

原始问题：{query}
改写后：
"""

def rewrite_query(query, llm):
    return llm.invoke(query_rewrite_prompt.format(query=query))
```

**方案2：多查询扩展**
```python
multi_query_prompt = """
请为以下问题生成3个不同角度的搜索查询，用于从知识库中检索相关信息。

原始问题：{query}

请以JSON列表形式输出3个查询：
["query1", "query2", "query3"]
"""

def expand_query(query, llm):
    response = llm.invoke(multi_query_prompt.format(query=query))
    queries = json.loads(response.content)
    return [query] + queries  # 包含原始查询
```

**方案3：HyDE（假设文档嵌入）**
```python
hyde_prompt = """
请为以下问题生成一个可能的答案段落（即使不确定也请生成）：

问题：{query}

答案段落：
"""

def hyde_embedding(query, llm, embeddings):
    # 生成假设答案
    hypothetical_answer = llm.invoke(hyde_prompt.format(query=query))
    # 用假设答案的向量检索
    return embeddings.embed_query(hypothetical_answer.content)
```

### 4.2 重排序（Rerank）

**为什么需要Rerank？**
- 向量检索是粗排（Top-K可能不够精确）
- Cross-encoder更精确但更慢
- 先粗排后精排，平衡效率和质量

**Cohere Rerank：**
```python
from langchain_community.retrievers import CohereRerank

reranker = CohereRerank(
    cohere_api_key="xxx",
    top_n=5  # 最终返回5条
)

# 包装检索器
retriever_with_rerank = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vector_retriever
)
```

**BGE Reranker（开源）：**
```python
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 手动重排序
def rerank(query, documents, top_n=5):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_n]]
```

---

## 5. 答案生成

### 5.1 RAG Prompt设计

**标准RAG Prompt：**
```python
RAG_PROMPT = """
你是一个专业的问答助手。请基于以下参考信息回答用户问题。

参考信息：
{context}

用户问题：{question}

回答要求：
1. 只基于参考信息回答，不要编造
2. 如果参考信息不足以回答，请明确说明
3. 引用信息来源（标注[来源X]）
4. 结构清晰，分点论述
5. 语言简洁专业

回答：
"""
```

**带引用的Prompt：**
```python
RAG_PROMPT_WITH_CITATION = """
基于以下文档片段回答问题。每个论点后用[citation:X]标注来源。

文档片段：
{context}

问题：{question}

回答格式：
基于文档[citation:1]，...
另外[citation:2]指出...
综合以上信息，...
"""
```

### 5.2 RAG Chain实现

**LangChain实现：**
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Question-Answer Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# RAG Chain
rag_chain = create_retrieval_chain(
    retriever_with_rerank,
    question_answer_chain
)

# 使用
result = rag_chain.invoke({"question": "什么是RAG？"})
print(result["answer"])
print(result.get("context", []))  # 检索到的文档
```

**LlamaIndex实现：**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

# 加载文档
documents = SimpleDirectoryReader("./docs").load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 配置查询引擎
query_engine = RetrieverQueryEngine(
    retriever=index.as_retriever(similarity_top_k=10),
    node_postprocessors=[
        SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=5
        )
    ]
)

# 查询
response = query_engine.query("什么是RAG？")
print(response)
```

### 5.3 流式输出

```python
# LangChain流式RAG
for chunk in rag_chain.stream({"question": "什么是RAG？"}):
    print(chunk, end="", flush=True)
```

---

## 6. 生产级优化

### 6.1 索引管理

**增量更新：**
```python
def update_index(new_documents, vectorstore, embeddings):
    """增量添加新文档"""
    # 分块
    chunks = text_splitter.split_documents(new_documents)
    
    # 添加元数据
    for chunk in chunks:
        chunk.metadata["indexed_at"] = datetime.now().isoformat()
    
    # 增量添加
    vectorstore.add_documents(chunks)
```

**文档去重：**
```python
def deduplicate_documents(documents):
    """基于内容哈希去重"""
    seen = set()
    unique = []
    
    for doc in documents:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    
    return unique
```

### 6.2 性能优化

| 优化点 | 方案 | 效果 |
|--------|------|------|
| **检索延迟** | 向量索引（HNSW/IVF） | 10-100ms |
| **生成延迟** | 流式输出 | 用户体验提升 |
| **并发** | 异步处理 | 支持多用户 |
| **缓存** | 语义缓存 | 重复查询加速 |
| **批量** | 批量Embedding | 索引构建加速 |

**语义缓存：**
```python
from langchain.storage import InMemoryStore
from langchain.embeddings import CacheBackedEmbeddings

# 缓存Embedding结果
store = InMemoryStore()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store
)
```

### 6.3 质量监控

```python
class RAGQualityMonitor:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "avg_retrieval_score": 0,
            "avg_response_time": 0,
            "user_feedback": []
        }
    
    def log_query(self, query, retrieval_scores, response_time):
        self.metrics["total_queries"] += 1
        self.metrics["avg_retrieval_score"] = (
            self.metrics["avg_retrieval_score"] * (self.metrics["total_queries"] - 1)
            + sum(retrieval_scores) / len(retrieval_scores)
        ) / self.metrics["total_queries"]
        self.metrics["avg_response_time"] = (
            self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1)
            + response_time
        ) / self.metrics["total_queries"]
    
    def get_report(self):
        return self.metrics.copy()
```

---

## 7. 开发者Checklist

### 7.1 RAG应用开发Checklist

**索引阶段：**
- [ ] 选择合适的文档加载器
- [ ] 设计分块策略（大小、重叠）
- [ ] 添加文档元数据
- [ ] 选择Embedding模型
- [ ] 选择向量数据库
- [ ] 实现增量更新机制

**查询阶段：**
- [ ] 实现查询重写/扩展
- [ ] 配置混合检索
- [ ] 添加重排序
- [ ] 设计RAG Prompt
- [ ] 实现流式输出
- [ ] 添加来源引用

**生产化：**
- [ ] 性能测试和优化
- [ ] 质量监控
- [ ] 错误处理
- [ ] 日志追踪
- [ ] 安全防护

### 7.2 常见陷阱

**陷阱1：分块太大**
- 问题：检索不精确，噪声多
- 解决：chunk_size控制在300-500

**陷阱2：缺少重排序**
- 问题：Top-K结果质量不稳定
- 解决：添加Reranker

**陷阱3：忽视元数据**
- 问题：检索到过期或不相关内容
- 解决：利用元数据过滤

**陷阱4：Prompt不够明确**
- 问题：模型不利用检索内容
- 解决：明确指令"只基于参考信息"

---

*最后更新：2026-05-07*

# RAG完整实战项目

> 从零构建一个企业级RAG问答系统

## 1. 项目概述

### 1.1 项目背景

本项目将构建一个**企业级知识库问答系统**，具备以下特性：
- 支持多种文档格式（PDF、Word、Markdown、网页）
- 混合检索（向量+关键词）
- 多轮对话记忆
- 流式输出
- 质量监控

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         架构总览                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   │   文档源      │───→│  文档处理    │───→│   向量数据库     │    │
│   │ (PDF/Word/   │    │ (分块/嵌入)  │    │   (Milvus)      │    │
│   │  Web/API)    │    └──────────────┘    └────────┬─────────┘    │
│   └──────────────┘                                 │               │
│                                                    ▼               │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   │   用户查询    │───→│  查询处理    │───→│   检索与重排序   │    │
│   │              │    │ (重写/扩展)  │    │ (Hybrid/Rerank) │    │
│   └──────────────┘    └──────────────┘    └────────┬─────────┘    │
│                                                    ▼               │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   │   记忆管理    │←───│  LLM生成     │←───│   答案合成      │    │
│   │ (对话历史)   │    │ (GPT-4o)     │    │                 │    │
│   └──────────────┘    └──────────────┘    └──────────────────┘    │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| **语言** | Python | 3.11+ |
| **框架** | LangChain | 0.1.x |
| **向量数据库** | Milvus | 2.4+ |
| **嵌入模型** | BGE-M3 / OpenAI | - |
| **重排序** | BGE-Reranker | - |
| **LLM** | GPT-4o / Qwen | - |
| **API框架** | FastAPI | 0.100+ |

---

## 2. 环境准备

### 2.1 依赖安装

```bash
# 创建虚拟环境
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac
# 或
rag-env\Scripts\activate    # Windows

# 安装核心依赖
pip install langchain langchain-openai langchain-community
pip install pymilvus sentence-transformers
pip install fastapi uvicorn python-multipart
pip install pypdf python-docx beautifulsoup4
pip install python-dotenv
```

### 2.2 Milvus部署

**方式一：Docker部署**
```bash
# 创建docker-compose.yml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ./volumes/etcd:/etcd
    networks:
      - milvus

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - milvus

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
    networks:
      - milvus

networks:
  milvus:
    driver: bridge

# 启动
docker-compose up -d
```

**方式二：使用托管服务**
- Milvus Cloud: https://cloud.zilliz.com
- Pinecone: https://www.pinecone.io

---

## 3. 核心代码实现

### 3.1 配置文件

```python
# config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM配置
    openai_api_key: str
    openai_model: str = "gpt-4o"
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "enterprise_knowledge_base"
    
    # 嵌入模型配置
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    
    # 检索配置
    top_k: int = 10
    rerank_top_n: int = 5
    
    # 应用配置
    debug: bool = False
    max_context_length: int = 8192
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.2 文档加载模块

```python
# document_loader.py
from typing import List, Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
    CSVLoader,
    DirectoryLoader
)
from langchain_core.documents import Document

class DocumentLoader:
    """文档加载器"""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".md": TextLoader,
    }
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """加载单个文档"""
        ext = file_path.lower().split(".")[-1]
        
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "docx":
            loader = Docx2txtLoader(file_path)
        elif ext in ["txt", "md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == "csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        return loader.load()
    
    @staticmethod
    def load_web_page(url: str) -> List[Document]:
        """加载网页"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    @staticmethod
    def load_directory(dir_path: str, pattern: str = "**/*") -> List[Document]:
        """加载目录下所有文档"""
        loader = DirectoryLoader(
            dir_path,
            glob=pattern,
            use_multithreading=True
        )
        return loader.load()
    
    @staticmethod
    def load_from_text(text: str, source: str = "text") -> List[Document]:
        """从文本创建文档"""
        return [Document(page_content=text, metadata={"source": source})]
```

### 3.3 文本分块模块

```python
# text_splitter.py
from typing import List
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

class TextSplitter:
    """文本分块器"""
    
    @staticmethod
    def recursive_split(
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """递归分割（推荐）"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        return splitter.split_documents(documents)
    
    @staticmethod
    def token_split(
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> List[Document]:
        """基于Token分割"""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)
    
    @staticmethod
    def semantic_split(
        documents: List[Document],
        model_name: str = "BAAI/bge-m3"
    ) -> List[Document]:
        """语义分割"""
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
        splitter = SemanticChunker(embeddings)
        return splitter.split_documents(documents)
    
    @staticmethod
    def optimize_chunk_size(doc_type: str) -> tuple:
        """根据文档类型推荐分块参数"""
        configs = {
            "qa": (300, 50),
            "summary": (1000, 200),
            "code": (500, 100),
            "legal": (800, 150),
            "technical": (500, 100)
        }
        return configs.get(doc_type, (500, 50))
```

### 3.4 向量存储模块

```python
# vector_store.py
from typing import List
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import settings

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self):
        self.embeddings = self._init_embeddings()
        self.vector_store = None
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        if settings.embedding_model.startswith("text-embedding"):
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
        else:
            return HuggingFaceBgeEmbeddings(
                model_name=settings.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
    
    def create_collection(self, documents: List[Document], drop_old: bool = False):
        """创建/更新向量索引"""
        self.vector_store = Milvus.from_documents(
            documents=documents,
            embedding=self.embeddings,
            connection_args={
                "host": settings.milvus_host,
                "port": settings.milvus_port
            },
            collection_name=settings.milvus_collection_name,
            drop_old=drop_old
        )
    
    def add_documents(self, documents: List[Document]):
        """增量添加文档"""
        if not self.vector_store:
            self._connect()
        self.vector_store.add_documents(documents)
    
    def _connect(self):
        """连接到已有集合"""
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={
                "host": settings.milvus_host,
                "port": settings.milvus_port
            },
            collection_name=settings.milvus_collection_name
        )
    
    def as_retriever(self, search_type: str = "mmr", **kwargs):
        """获取检索器"""
        if not self.vector_store:
            self._connect()
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={**kwargs}
        )
    
    def search(self, query: str, k: int = 10):
        """直接搜索"""
        if not self.vector_store:
            self._connect()
        
        return self.vector_store.similarity_search(query, k=k)
```

### 3.5 查询处理模块

```python
# query_processor.py
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json

class QueryRewriter:
    """查询重写器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0
        )
    
    def rewrite_query(self, query: str) -> str:
        """重写查询"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个查询重写专家。请将用户的问题改写为更适合检索的形式。"),
            ("user", "原始问题：{query}\n\n改写后的问题：")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        return response.content
    
    def expand_query(self, query: str, n: int = 3) -> List[str]:
        """扩展查询为多个角度"""
        class QueryList(BaseModel):
            queries: List[str] = Field(description="多个搜索查询")
        
        parser = JsonOutputParser(pydantic_object=QueryList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个查询扩展专家。请为给定问题生成多个不同角度的搜索查询。"),
            ("user", "原始问题：{query}\n\n请生成{n}个不同角度的搜索查询，以JSON格式输出：\n{format_instructions}")
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "query": query,
            "n": n,
            "format_instructions": parser.get_format_instructions()
        })
        
        return [query] + result["queries"]
    
    def hyde_query(self, query: str) -> str:
        """生成假设文档嵌入"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "请为以下问题生成一个详细的假设答案段落。"),
            ("user", "问题：{query}\n\n假设答案：")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        return response.content
```

### 3.6 RAG链实现

```python
# rag_chain.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from vector_store import VectorStoreManager
from query_processor import QueryRewriter
from config import settings

class RAGChain:
    """RAG问答链"""
    
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.query_rewriter = QueryRewriter()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0
        )
        self.reranker = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3"
        )
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """构建RAG链"""
        # 基础检索器
        vector_retriever = self.vector_store_manager.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.top_k, "fetch_k": settings.top_k * 2}
        )
        
        # 混合检索（可选）
        # bm25_retriever = BM25Retriever.from_documents(documents)
        # ensemble_retriever = EnsembleRetriever(
        #     retrievers=[vector_retriever, bm25_retriever],
        #     weights=[0.7, 0.3]
        # )
        
        # 重排序
        compressor = CrossEncoderReranker(
            model=self.reranker,
            top_n=settings.rerank_top_n
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_retriever
        )
        
        # RAG Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "参考文档：\n{context}")
        ])
        
        # 创建链
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
    
    def invoke(self, query: str, chat_history: list = None):
        """执行查询"""
        # 查询重写
        rewritten_query = self.query_rewriter.rewrite_query(query)
        
        # 构建消息历史
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # 执行RAG链
        result = self.chain.invoke({
            "input": rewritten_query,
            "chat_history": messages,
            "context": ""
        })
        
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "") for doc in result.get("context", [])],
            "rewritten_query": rewritten_query
        }
    
    def stream(self, query: str, chat_history: list = None):
        """流式执行"""
        rewritten_query = self.query_rewriter.rewrite_query(query)
        
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        for chunk in self.chain.stream({
            "input": rewritten_query,
            "chat_history": messages,
            "context": ""
        }):
            if "answer" in chunk:
                yield chunk["answer"]

# 系统Prompt
SYSTEM_PROMPT = """
你是一个专业的企业知识库问答助手。请基于提供的参考文档回答用户问题。

回答要求：
1. 只基于参考文档内容回答，不要编造信息
2. 如果文档中没有足够信息，明确说明"信息不足"
3. 引用信息来源（标注来源文件名）
4. 结构清晰，使用列表或分点论述
5. 语言简洁专业，避免冗长

格式示例：
基于文档[document.pdf]，...

另外[guide.md]指出...

综合以上信息，...
"""
```

### 3.7 API服务

```python
# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from rag_chain import RAGChain
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_store import VectorStoreManager
import os

app = FastAPI(title="企业RAG知识库", version="1.0")

# 全局实例
rag_chain = None
vector_store_manager = VectorStoreManager()

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = None
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    rewritten_query: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    document_count: int
    chunk_count: int
    message: str

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    global rag_chain
    try:
        rag_chain = RAGChain()
    except Exception as e:
        print(f"初始化失败，可能需要先构建索引: {e}")

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG链未初始化")
    
    try:
        result = rag_chain.invoke(
            query=request.query,
            chat_history=request.chat_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest):
    """流式查询接口"""
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG链未初始化")
    
    async def generate():
        for chunk in rag_chain.stream(
            query=request.query,
            chat_history=request.chat_history
        ):
            yield chunk
    
    return generate()

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """上传文档并添加到索引"""
    documents = []
    
    for file in files:
        # 保存临时文件
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # 加载文档
        try:
            docs = DocumentLoader.load_document(temp_path)
            documents.extend(docs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法处理文件 {file.filename}: {str(e)}")
        finally:
            os.remove(temp_path)
    
    # 分块
    chunks = TextSplitter.recursive_split(documents)
    
    # 添加到向量存储
    try:
        vector_store_manager.add_documents(chunks)
        return DocumentUploadResponse(
            success=True,
            document_count=len(documents),
            chunk_count=len(chunks),
            message="文档上传成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加到索引失败: {str(e)}")

@app.post("/api/v1/index/build")
async def build_index(directory: str):
    """从目录构建索引"""
    try:
        # 加载文档
        documents = DocumentLoader.load_directory(directory)
        
        # 分块
        chunks = TextSplitter.recursive_split(documents)
        
        # 创建索引
        vector_store_manager.create_collection(chunks, drop_old=True)
        
        # 重新初始化RAG链
        global rag_chain
        rag_chain = RAGChain()
        
        return {
            "success": True,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "message": "索引构建成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 4. 部署与运行

### 4.1 运行服务

```bash
# 启动服务
python main.py

# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.2 API调用示例

**构建索引：**
```bash
curl -X POST "http://localhost:8000/api/v1/index/build" \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/documents"}'
```

**上传文档：**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

**查询：**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "公司的请假政策是什么？",
    "chat_history": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
    ]
  }'
```

---

## 5. 生产级优化

### 5.1 性能优化

```python
# 缓存优化
from langchain.storage import InMemoryStore
from langchain.embeddings import CacheBackedEmbeddings

# 缓存Embedding结果
store = InMemoryStore()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store
)
```

### 5.2 质量监控

```python
# 质量监控示例
class RAGMonitor:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "avg_retrieval_score": 0,
            "avg_response_time": 0
        }
    
    def log_query(self, retrieval_scores, response_time):
        self.metrics["total_queries"] += 1
        self.metrics["avg_retrieval_score"] = (
            self.metrics["avg_retrieval_score"] * (self.metrics["total_queries"] - 1)
            + sum(retrieval_scores) / len(retrieval_scores)
        ) / self.metrics["total_queries"]
        self.metrics["avg_response_time"] = (
            self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1)
            + response_time
        ) / self.metrics["total_queries"]
```

### 5.3 日志与追踪

```python
# 使用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "enterprise-rag"
```

---

## 6. 项目结构

```
enterprise-rag/
├── main.py              # API入口
├── config.py            # 配置文件
├── document_loader.py   # 文档加载
├── text_splitter.py     # 文本分块
├── vector_store.py      # 向量存储
├── query_processor.py   # 查询处理
├── rag_chain.py         # RAG链
├── .env                 # 环境变量
├── requirements.txt     # 依赖
└── docs/                # 知识库文档
```

---

## 7. 部署清单

- [ ] 配置Milvus向量数据库
- [ ] 设置OpenAI API密钥
- [ ] 安装依赖包
- [ ] 构建初始索引
- [ ] 启动API服务
- [ ] 配置Nginx反向代理（可选）
- [ ] 设置HTTPS（可选）
- [ ] 配置监控告警（可选）

---

*最后更新：2026-05-12*

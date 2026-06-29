# 案例二：MCP工具服务架构

> 基于Model Context Protocol的企业级AI工具服务集群

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [MCP协议基础](../04-Agent系统架构/MCP协议.md), [微服务架构设计](../08-架构模式/01-AI系统架构模式.md)
- **关联文件**: [自托管与本地优先Agent生态](../10-AI工程化前沿/02-自托管与本地优先Agent生态.md), [AI韧性工程与容错设计](../08-架构模式/03-AI韧性工程与容错设计.md)
- **最后更新**: 2026-06-12
---

## 1. 项目背景

### 1.1 业务需求

**客户画像：** 大型科技集团，AI应用20+，外部工具/数据源100+

**核心需求：**
- 统一AI应用与外部工具/数据源的连接协议
- 支持LLM应用动态发现和调用工具能力
- 工具服务热插拔，零停机升级
- 日均200万次工具调用，P99延迟 < 500ms
- 全链路安全审计，细粒度权限控制

### 1.2 MCP协议概述

MCP（Model Context Protocol）是Anthropic于2024年底发布的开放协议，为LLM应用与外部工具/数据源之间提供标准化通信规范。

```
MCP核心概念:
├── Protocol    - 基于JSON-RPC 2.0的双向通信协议
├── Transport   - Stdio（本地）/ SSE（远程）/ Streamable HTTP（2025新规范）
├── Server      - 暴露工具(Tools)、资源(Resources)、提示(Prompts)的服务端
├── Client      - LLM应用内嵌的MCP客户端，负责与Server通信
└── Capability  - 能力协商机制，Server声明支持的功能集
```

### 1.3 技术约束

| 约束 | 要求 |
|------|------|
| 协议版本 | MCP 2025-03-26 (Streamable HTTP) |
| 部署方式 | 容器化 + Kubernetes编排 |
| 安全 | OAuth2认证 + 细粒度权限 + 审计日志 |
| 可用性 | 99.95% |
| 扩展性 | 支持工具服务扩展到500+ |

---

## 2. MCP工具服务架构总览

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         LLM应用层                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 智能助手  │  │ 代码助手  │  │ 数据分析  │  │ 客服Agent │            │
│  │(Claude)  │  │(Cursor)  │  │(自研)    │  │(自研)    │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       └──────────────┴──────────────┴──────────────┘                  │
│                          MCP Client                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓ Streamable HTTP / SSE
┌──────────────────────────────────────────────────────────────────────┐
│                       MCP网关层                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              MCP Gateway（统一入口）                          │   │
│  │  认证鉴权 → 权限校验 → 路由分发 → 限流熔断 → 审计日志        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      MCP Server集群                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 文件系统  │  │ 数据库   │  │ API网关   │  │ 搜索引擎  │            │
│  │ MCP Server│  │MCP Server│  │MCP Server │  │MCP Server│            │
│  │ (3副本)  │  │ (3副本)  │  │ (3副本)   │  │ (2副本)  │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       └──────────────┴──────────────┴──────────────┘                  │
│                     服务注册与发现 (Consul)                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                       外部服务层                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ NFS/OSS  │  │ MySQL    │  │ 第三方API │  │ Elastic  │            │
│  │ 文件存储  │  │ PostgreSQL│  │ SaaS服务  │  │ 搜索集群  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

| 原则 | 说明 |
|------|------|
| 协议标准化 | 所有工具服务统一遵循MCP规范，应用层无需适配不同工具协议 |
| 热插拔 | MCP Server可独立部署、升级、扩缩容，不影响上层应用 |
| 最小权限 | 每个MCP Server仅暴露必要的工具和资源，权限粒度到工具级别 |
| 故障隔离 | 单个MCP Server故障不影响其他Server，网关层熔断保护 |
| 可观测 | 全链路TraceId，工具调用审计日志，性能指标实时采集 |
| 异步优先 | 长耗时操作采用异步模式，避免阻塞LLM推理 |

### 2.3 MCP通信协议详解

```
MCP消息流（Streamable HTTP模式）:

Client                                          Server
  │                                               │
  │──── POST /mcp ─────────────────────────────→  │
  │     {jsonrpc:"2.0", method:"initialize",      │
  │      params:{capabilities:{...},              │
  │              clientInfo:{name:"app"}}}         │
  │                                               │
  │←─── 200 OK ─────────────────────────────────  │
  │     {jsonrpc:"2.0", result:{capabilities:{    │
  │      tools:{listChanged:true},                │
  │      resources:{subscribe:true}},             │
  │      serverInfo:{name:"fs-server"}}}          │
  │                                               │
  │──── POST /mcp ─────────────────────────────→  │
  │     {jsonrpc:"2.0", method:"tools/list"}      │
  │                                               │
  │←─── 200 OK ─────────────────────────────────  │
  │     {tools:[{name:"read_file",                │
  │      description:"读取文件内容",               │
  │      inputSchema:{type:"object",...}}]}       │
  │                                               │
  │──── POST /mcp ─────────────────────────────→  │
  │     {jsonrpc:"2.0", method:"tools/call",      │
  │      params:{name:"read_file",                │
  │      arguments:{path:"/data/report.csv"}}}    │
  │                                               │
  │←─── 200 OK ─────────────────────────────────  │
  │     {content:[{type:"text",                   │
  │      text:"name,value\n营收,1000"}]}          │
  │                                               │
```

### 2.4 核心设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| MCP SDK | mcp-python-sdk (官方) | 官方维护、功能完整、社区活跃 |
| 传输协议 | Streamable HTTP | 支持远程部署、无状态、可负载均衡 |
| 服务注册 | Consul | 支持健康检查、KV存储、多数据中心 |
| 网关 | 自研MCP Gateway | 需要MCP协议感知的智能路由 |
| 容器编排 | Kubernetes | 生态成熟、自动扩缩容、滚动更新 |
| 监控 | Prometheus + Grafana | 云原生标准、社区丰富 |

### 2.5 替代方案分析

| 决策点 | 未选方案 | 未选理由 |
|--------|---------|---------|
| MCP SDK | 自研SDK | 官方SDK已覆盖核心功能，自研成本高且维护负担重 |
| 传输协议 | SSE (Server-Sent Events) | SSE是单向推送，不适合远程部署；Streamable HTTP支持请求-响应和双向流 |
| 服务注册 | etcd | etcd适合小规模强一致性场景，但不适合多数据中心的服务发现 |
| 服务注册 | Nacos | 虽然功能丰富，但Java依赖重，与MCP的Python/Go生态不匹配 |
| 网关 | Kong/Apigee | 传统API网关不了解MCP协议，无法实现MCP特有的工具发现和Schema路由 |

**原则：** 协议层优先使用官方推荐实现，基础设施层优先与团队现有技术栈一致。

### 3.1 项目结构

```
mcp-servers/
├── common/
│   ├── __init__.py
│   ├── config.py          # 公共配置
│   ├── auth.py            # 认证中间件
│   ├── logging.py         # 审计日志
│   └── health.py          # 健康检查
├── servers/
│   ├── filesystem/        # 文件系统MCP Server
│   │   ├── server.py
│   │   ├── config.yaml
│   │   └── Dockerfile
│   ├── database/          # 数据库MCP Server
│   │   ├── server.py
│   │   ├── config.yaml
│   │   └── Dockerfile
│   └── api_gateway/       # API网关MCP Server
│       ├── server.py
│       ├── config.yaml
│       └── Dockerfile
├── gateway/               # MCP网关
│   ├── main.py
│   ├── router.py
│   └── config.yaml
├── docker-compose.yaml
└── k8s/
    ├── namespace.yaml
    ├── filesystem-server.yaml
    ├── database-server.yaml
    ├── api-gateway-server.yaml
    └── mcp-gateway.yaml
```

### 3.2 公共模块

**common/config.py**

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class ServerConfig(BaseSettings):
    server_name: str = Field(default="mcp-server", alias="MCP_SERVER_NAME")
    server_version: str = Field(default="1.0.0", alias="MCP_SERVER_VERSION")
    host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    port: int = Field(default=8080, alias="MCP_PORT")
    log_level: str = Field(default="INFO", alias="MCP_LOG_LEVEL")
    consul_host: str = Field(default="consul:8500", alias="CONSUL_HOST")
    consul_token: str = Field(default="", alias="CONSUL_TOKEN")
    oauth2_issuer: str = Field(default="", alias="OAUTH2_ISSUER")
    oauth2_audience: str = Field(default="", alias="OAUTH2_AUDIENCE")
    max_concurrent_calls: int = Field(default=100, alias="MCP_MAX_CONCURRENT")
    call_timeout_seconds: int = Field(default=30, alias="MCP_CALL_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> ServerConfig:
    return ServerConfig()
```

**common/auth.py**

```python
import time
import httpx
from functools import lru_cache
from typing import Optional
from mcp.server.auth.middleware import AuthMiddleware


class TokenVerifier:
    def __init__(self, issuer: str, audience: str):
        self.issuer = issuer
        self.audience = audience
        self._jwks_cache: dict = {}
        self._jwks_expire: float = 0

    async def _fetch_jwks(self) -> dict:
        if self._jwks_cache and time.time() < self._jwks_expire:
            return self._jwks_cache
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.issuer}/.well-known/jwks.json")
            resp.raise_for_status()
            self._jwks_cache = resp.json()
            self._jwks_expire = time.time() + 3600
        return self._jwks_cache

    async def verify(self, token: str) -> Optional[dict]:
        import jwt
        jwks = await self._fetch_jwks()
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            for key in jwks.get("keys", []):
                if key["kid"] == kid:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=["RS256"],
                        audience=self.audience,
                        issuer=self.issuer,
                    )
                    return payload
        except jwt.InvalidTokenError:
            pass
        return None
```

**common/logging.py**

```python
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional


logger = logging.getLogger("mcp.audit")


class AuditLogger:
    def __init__(self, server_name: str):
        self.server_name = server_name

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        caller_id: str,
        trace_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        elapsed = getattr(self, "_start_time", None)
        duration_ms = int((time.time() - elapsed) * 1000) if elapsed else 0

        entry = {
            "trace_id": trace_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": self.server_name,
            "event": "tool_call",
            "tool_name": tool_name,
            "caller_id": caller_id,
            "arguments_hash": self._hash_arguments(arguments),
            "result_type": type(result).__name__ if result else None,
            "error": error,
            "duration_ms": duration_ms,
        }
        logger.info(json.dumps(entry, ensure_ascii=False))

    @staticmethod
    def _hash_arguments(arguments: dict) -> str:
        import hashlib
        raw = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

**common/health.py**

```python
import asyncio
import time
from typing import Callable, Optional


class HealthChecker:
    def __init__(self):
        self._checks: dict[str, Callable] = {}
        self._start_time = time.time()

    def register(self, name: str, check_fn: Callable):
        self._checks[name] = check_fn

    async def check(self) -> dict:
        results = {}
        overall = "healthy"
        for name, fn in self._checks.items():
            try:
                result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                results[name] = {"status": "healthy", "detail": result}
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
                overall = "degraded"

        return {
            "status": overall,
            "uptime_seconds": int(time.time() - self._start_time),
            "checks": results,
        }
```

### 3.3 文件系统MCP Server

**servers/filesystem/server.py**

```python
import os
import aiofiles
import mimetypes
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from common.config import get_config
from common.logging import AuditLogger
from common.health import HealthChecker

config = get_config()
mcp = FastMCP(
    name=config.server_name,
    version=config.server_version,
)
audit = AuditLogger(config.server_name)
health = HealthChecker()

ALLOWED_ROOTS = [Path(p).resolve() for p in os.getenv("FS_ALLOWED_ROOTS", "/data").split(",")]
MAX_FILE_SIZE = int(os.getenv("FS_MAX_FILE_SIZE", "10485760"))  # 10MB


def _validate_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"路径越权访问: {path} 不在允许的根目录范围内")
    return resolved


@mcp.tool()
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """读取文件内容。支持文本文件，自动检测编码。

    Args:
        path: 文件绝对路径
        encoding: 文件编码，默认utf-8
    """
    validated = _validate_path(path)
    if not validated.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    if validated.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"文件过大: {validated.stat().st_size} bytes，上限 {MAX_FILE_SIZE} bytes")

    async with aiofiles.open(validated, "r", encoding=encoding) as f:
        content = await f.read()
    audit.log_tool_call("read_file", {"path": path}, content[:200], caller_id="system")
    return content


@mcp.tool()
async def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """写入文件内容。自动创建父目录。

    Args:
        path: 文件绝对路径
        content: 要写入的内容
        encoding: 文件编码，默认utf-8
    """
    validated = _validate_path(path)
    validated.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(validated, "w", encoding=encoding) as f:
        await f.write(content)
    audit.log_tool_call("write_file", {"path": path}, "ok", caller_id="system")
    return f"文件写入成功: {path} ({len(content)} 字符)"


@mcp.tool()
async def list_directory(path: str, pattern: str = "*") -> str:
    """列出目录内容。

    Args:
        path: 目录绝对路径
        pattern: 文件匹配模式，默认*
    """
    validated = _validate_path(path)
    if not validated.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")

    entries = []
    for entry in sorted(validated.glob(pattern)):
        entry_type = "DIR" if entry.is_dir() else "FILE"
        size = entry.stat().st_size if entry.is_file() else "-"
        mime = mimetypes.guess_type(entry.name)[0] or "unknown"
        entries.append(f"{entry_type:4s}  {size:>10}  {mime:20s}  {entry.name}")

    result = "\n".join(entries) if entries else "(空目录)"
    audit.log_tool_call("list_directory", {"path": path}, result[:200], caller_id="system")
    return result


@mcp.tool()
async def search_files(path: str, query: str, max_results: int = 20) -> str:
    """在目录中搜索包含指定文本的文件。

    Args:
        path: 搜索根目录
        query: 搜索关键词
        max_results: 最大返回结果数
    """
    validated = _validate_path(path)
    matches = []

    for file_path in validated.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
            if query.lower() in content.lower():
                line_count = content.lower().count(query.lower())
                matches.append(f"{file_path} (匹配 {line_count} 次)")
                if len(matches) >= max_results:
                    break
        except Exception:
            continue

    result = "\n".join(matches) if matches else "未找到匹配文件"
    audit.log_tool_call("search_files", {"path": path, "query": query}, result[:200], caller_id="system")
    return result


@mcp.tool()
async def get_file_info(path: str) -> str:
    """获取文件元信息（大小、修改时间、权限等）。

    Args:
        path: 文件或目录绝对路径
    """
    import stat
    validated = _validate_path(path)
    if not validated.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    st = validated.stat()
    info = {
        "路径": str(validated),
        "类型": "目录" if validated.is_dir() else "文件",
        "大小": f"{st.st_size} bytes",
        "修改时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "权限": stat.filemode(st.st_mode),
        "MIME类型": mimetypes.guess_type(validated.name)[0] or "unknown",
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


@mcp.resource("file://{path}")
async def file_resource(path: str) -> str:
    """以资源形式暴露文件内容，支持MCP资源订阅。"""
    validated = _validate_path(path)
    async with aiofiles.open(validated, "r", encoding="utf-8", errors="ignore") as f:
        return await f.read()


async def health_check():
    for root in ALLOWED_ROOTS:
        if not root.exists():
            return {"roots": f"根目录不可达: {root}"}
    return {"roots": [str(r) for r in ALLOWED_ROOTS]}


health.register("filesystem", health_check)


if __name__ == "__main__":
    import time as _time
    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=config.host, port=config.port)
```

**servers/filesystem/config.yaml**

```yaml
server:
  name: filesystem-mcp-server
  version: 1.0.0
  host: 0.0.0.0
  port: 8081

filesystem:
  allowed_roots:
    - /data/documents
    - /data/reports
    - /data/uploads
  max_file_size: 10485760  # 10MB
  denied_extensions:
    - .exe
    - .sh
    - .bat

security:
  read_only_roots:
    - /data/reports
  write_allowed_roots:
    - /data/documents
    - /data/uploads

consul:
  service_name: mcp-filesystem
  service_id: mcp-filesystem-1
  tags: ["mcp", "filesystem", "v1"]
  check_interval: 10s
```

### 3.4 数据库MCP Server

**servers/database/server.py**

```python
import os
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from common.config import get_config
from common.logging import AuditLogger
from common.health import HealthChecker

config = get_config()
mcp = FastMCP(
    name=config.server_name,
    version=config.server_version,
)
audit = AuditLogger(config.server_name)
health = HealthChecker()

MAX_ROWS = int(os.getenv("DB_MAX_ROWS", "1000"))
QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
READ_ONLY = os.getenv("DB_READ_ONLY", "true").lower() == "true"

_pool = None


async def _get_pool():
    global _pool
    if _pool is None:
        import asyncpg
        _pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"),
            min_size=5,
            max_size=20,
            command_timeout=QUERY_TIMEOUT,
        )
    return _pool


@asynccontextmanager
async def _acquire_conn():
    pool = await _get_pool()
    async with pool.acquire() as conn:
        yield conn


@mcp.tool()
async def execute_query(sql: str, params: Optional[list] = None) -> str:
    """执行只读SQL查询并返回结果。仅允许SELECT语句。

    Args:
        sql: SQL查询语句（仅支持SELECT）
        params: 查询参数列表
    """
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        raise PermissionError("仅允许SELECT查询，当前模式为只读")

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for kw in forbidden:
        if kw in normalized:
            raise PermissionError(f"禁止执行 {kw} 操作")

    async with _acquire_conn() as conn:
        rows = await conn.fetch(sql, *(params or []))
        if len(rows) > MAX_ROWS:
            rows = rows[:MAX_ROWS]

        columns = list(rows[0].keys()) if rows else []
        result = {
            "columns": columns,
            "row_count": len(rows),
            "data": [dict(r) for r in rows],
        }
        output = json.dumps(result, ensure_ascii=False, default=str)
        audit.log_tool_call("execute_query", {"sql": sql[:200]}, f"{len(rows)} rows", caller_id="system")
        return output


@mcp.tool()
async def list_tables(schema: str = "public") -> str:
    """列出数据库中的所有表及其行数。

    Args:
        schema: 数据库schema名称，默认public
    """
    sql = """
        SELECT table_name,
               (xpath('/row/cnt/text()',
                   query_to_xml(format('SELECT count(*) AS cnt FROM %I.%I', table_schema, table_name),
                   false, true, '')))[1]::text::int AS row_count
        FROM information_schema.tables
        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    async with _acquire_conn() as conn:
        rows = await conn.fetch(sql, schema)
        result = [{"table": r["table_name"], "rows": r["row_count"]} for r in rows]
        return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def describe_table(table_name: str, schema: str = "public") -> str:
    """获取表结构信息，包括列名、类型、约束、索引。

    Args:
        table_name: 表名
        schema: 数据库schema名称，默认public
    """
    columns_sql = """
        SELECT column_name, data_type, is_nullable,
               column_default, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
    """
    indexes_sql = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = $1 AND tablename = $2
    """
    async with _acquire_conn() as conn:
        columns = await conn.fetch(columns_sql, schema, table_name)
        indexes = await conn.fetch(indexes_sql, schema, table_name)

        result = {
            "table": table_name,
            "schema": schema,
            "columns": [dict(c) for c in columns],
            "indexes": [{"name": i["indexname"], "definition": i["indexdef"]} for i in indexes],
        }
        return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def execute_write(sql: str, params: Optional[list] = None) -> str:
    """执行写操作SQL（INSERT/UPDATE/DELETE）。需要写权限。

    Args:
        sql: SQL写操作语句
        params: 查询参数列表
    """
    if READ_ONLY:
        raise PermissionError("当前Server为只读模式，禁止写操作")

    normalized = sql.strip().upper()
    allowed_prefixes = ("INSERT", "UPDATE", "DELETE")
    if not any(normalized.startswith(p) for p in allowed_prefixes):
        raise PermissionError(f"仅允许 {', '.join(allowed_prefixes)} 操作")

    async with _acquire_conn() as conn:
        status = await conn.execute(sql, *(params or []))
        audit.log_tool_call("execute_write", {"sql": sql[:200]}, status, caller_id="system")
        return f"执行成功: {status}"


@mcp.tool()
async def run_explain(sql: str) -> str:
    """获取SQL执行计划，用于查询性能分析。

    Args:
        sql: 要分析的SQL查询
    """
    explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE false, COSTS true) {sql}"
    async with _acquire_conn() as conn:
        result = await conn.fetchval(explain_sql)
        return json.dumps(result, ensure_ascii=False, default=str, indent=2)


@mcp.resource("db://{schema}/{table}")
async def table_resource(schema: str, table: str) -> str:
    """以资源形式暴露表结构信息。"""
    return await describe_table(table, schema)


async def health_check():
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return {"database": "connected", "pool_size": pool.get_size()}
    except Exception as e:
        raise RuntimeError(f"数据库连接失败: {e}")


health.register("database", health_check)


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=config.host, port=config.port)
```

**servers/database/config.yaml**

```yaml
server:
  name: database-mcp-server
  version: 1.0.0
  host: 0.0.0.0
  port: 8082

database:
  read_only: true
  max_rows: 1000
  query_timeout: 30
  pool_min: 5
  pool_max: 20
  slow_query_log: 5.0  # 秒

security:
  allowed_schemas:
    - public
    - analytics
  denied_tables:
    - user_secrets
    - audit_logs_internal
  max_rows_export: 50000

consul:
  service_name: mcp-database
  service_id: mcp-database-1
  tags: ["mcp", "database", "postgresql", "v1"]
  check_interval: 10s
```

### 3.5 API网关MCP Server

**servers/api_gateway/server.py**

```python
import os
import json
import httpx
import time
import hashlib
from typing import Optional
from mcp.server.fastmcp import FastMCP
from common.config import get_config
from common.logging import AuditLogger
from common.health import HealthChecker

config = get_config()
mcp = FastMCP(
    name=config.server_name,
    version=config.server_version,
)
audit = AuditLogger(config.server_name)
health = HealthChecker()

API_REGISTRY_FILE = os.getenv("API_REGISTRY_FILE", "/config/api_registry.json")
REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
MAX_RESPONSE_SIZE = int(os.getenv("API_MAX_RESPONSE_SIZE", "1048576"))  # 1MB


def _load_api_registry() -> dict:
    with open(API_REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_api_spec(api_name: str) -> dict:
    registry = _load_api_registry()
    if api_name not in registry:
        raise ValueError(f"未注册的API: {api_name}，可用API: {list(registry.keys())}")
    return registry[api_name]


@mcp.tool()
async def call_api(api_name: str, method: str = "GET", path_params: Optional[dict] = None,
                   query_params: Optional[dict] = None, body: Optional[dict] = None,
                   headers: Optional[dict] = None) -> str:
    """调用已注册的第三方API。

    Args:
        api_name: API注册名称（如weather, jira, slack）
        method: HTTP方法，默认GET
        path_params: 路径参数
        query_params: 查询参数
        body: 请求体（JSON）
        headers: 额外请求头
    """
    spec = _get_api_spec(api_name)

    url = spec["base_url"]
    if spec.get("path"):
        url = url.rstrip("/") + "/" + spec["path"].lstrip("/")
    if path_params:
        for k, v in path_params.items():
            url = url.replace(f"{{{k}}}", str(v))

    req_headers = {**spec.get("default_headers", {})}
    if spec.get("auth_type") == "bearer":
        req_headers["Authorization"] = f"Bearer {spec['auth_token']}"
    elif spec.get("auth_type") == "api_key":
        req_headers[spec.get("auth_header", "X-API-Key")] = spec["auth_token"]
    if headers:
        req_headers.update(headers)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.request(
            method=method.upper(),
            url=url,
            params=query_params,
            json=body,
            headers=req_headers,
        )

    if len(resp.content) > MAX_RESPONSE_SIZE:
        truncated = resp.text[:MAX_RESPONSE_SIZE] + f"\n... (截断，总大小 {len(resp.content)} bytes)"
        result_text = truncated
    else:
        result_text = resp.text

    result = {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": result_text,
    }
    audit.log_tool_call("call_api", {"api_name": api_name, "method": method},
                        f"status={resp.status_code}", caller_id="system")
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def list_apis() -> str:
    """列出所有已注册的API及其可用方法。"""
    registry = _load_api_registry()
    apis = []
    for name, spec in registry.items():
        apis.append({
            "name": name,
            "base_url": spec["base_url"],
            "description": spec.get("description", ""),
            "auth_type": spec.get("auth_type", "none"),
            "methods": spec.get("methods", ["GET"]),
        })
    return json.dumps(apis, ensure_ascii=False, indent=2)


@mcp.tool()
async def batch_call(calls: list[dict]) -> str:
    """批量调用多个API，并行执行。

    Args:
        calls: 调用列表，每项包含api_name、method等参数
    """
    tasks = []
    for call in calls:
        tasks.append(call_api(
            api_name=call["api_name"],
            method=call.get("method", "GET"),
            path_params=call.get("path_params"),
            query_params=call.get("query_params"),
            body=call.get("body"),
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            output.append({"index": i, "error": str(r)})
        else:
            output.append({"index": i, "result": json.loads(r)})
    return json.dumps(output, ensure_ascii=False)


@mcp.tool()
async def test_api(api_name: str) -> str:
    """测试API连通性，发送健康检查请求。

    Args:
        api_name: API注册名称
    """
    spec = _get_api_spec(api_name)
    health_path = spec.get("health_check_path", "/")

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{spec['base_url']}{health_path}")
        latency = int((time.time() - start) * 1000)
        return json.dumps({
            "api_name": api_name,
            "status": "healthy" if resp.status_code < 500 else "unhealthy",
            "status_code": resp.status_code,
            "latency_ms": latency,
        })
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return json.dumps({
            "api_name": api_name,
            "status": "unreachable",
            "error": str(e),
            "latency_ms": latency,
        })


async def health_check():
    registry = _load_api_registry()
    return {"registered_apis": list(registry.keys()), "count": len(registry)}


health.register("api_gateway", health_check)


if __name__ == "__main__":
    import asyncio
    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=config.host, port=config.port)
```

**servers/api_gateway/api_registry.json**

```json
{
  "weather": {
    "base_url": "https://api.weather.com/v2",
    "path": "/forecast",
    "description": "天气预报API",
    "auth_type": "api_key",
    "auth_header": "X-API-Key",
    "auth_token": "${WEATHER_API_KEY}",
    "default_headers": {"Accept": "application/json"},
    "methods": ["GET"],
    "health_check_path": "/ping"
  },
  "jira": {
    "base_url": "https://company.atlassian.net/rest/api/3",
    "path": "/search",
    "description": "Jira工单系统API",
    "auth_type": "bearer",
    "auth_token": "${JIRA_TOKEN}",
    "default_headers": {"Content-Type": "application/json"},
    "methods": ["GET", "POST", "PUT"],
    "health_check_path": "/myself"
  },
  "slack": {
    "base_url": "https://slack.com/api",
    "path": "/chat.postMessage",
    "description": "Slack消息发送API",
    "auth_type": "bearer",
    "auth_token": "${SLACK_TOKEN}",
    "default_headers": {"Content-Type": "application/json"},
    "methods": ["POST"],
    "health_check_path": "/auth.test"
  }
}
```

**servers/api_gateway/config.yaml**

```yaml
server:
  name: api-gateway-mcp-server
  version: 1.0.0
  host: 0.0.0.0
  port: 8083

api_gateway:
  request_timeout: 30
  max_response_size: 1048576
  retry_count: 2
  retry_delay: 1.0
  rate_limit_per_api: 100  # 每分钟

security:
  allowed_domains:
    - api.weather.com
    - company.atlassian.net
    - slack.com
  denied_paths:
    - "*/admin/*"
    - "*/delete*"

consul:
  service_name: mcp-api-gateway
  service_id: mcp-api-gateway-1
  tags: ["mcp", "api-gateway", "v1"]
  check_interval: 10s
```

---

## 4. MCP服务集群架构

### 4.1 集群架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MCP Gateway (集群入口)                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Nginx/HAProxy (L4负载均衡)                                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │   │
│  │  │ Gateway-1  │  │ Gateway-2  │  │ Gateway-3  │               │   │
│  │  │ (主)       │  │ (从)       │  │ (从)       │               │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘               │   │
│  └────────┴───────────────┴───────────────┴───────────────────────┘   │
└────────────────────────────┬───────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     服务注册与发现 (Consul Cluster)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                       │
│  │ Consul-1   │  │ Consul-2   │  │ Consul-3   │   Raft共识            │
│  │ (Leader)   │  │ (Follower) │  │ (Follower) │                       │
│  └────────────┘  └────────────┘  └────────────┘                       │
│                                                                         │
│  服务注册表:                                                             │
│  ┌─────────────────┬────────────┬──────────┬──────────────┐           │
│  │ Service         │ Instances  │ Status   │ Metadata     │           │
│  ├─────────────────┼────────────┼──────────┼──────────────┤           │
│  │ mcp-filesystem  │ 3          │ healthy  │ version:1.0  │           │
│  │ mcp-database    │ 3          │ healthy  │ version:1.0  │           │
│  │ mcp-api-gateway │ 3          │ healthy  │ version:1.0  │           │
│  └─────────────────┴────────────┴──────────┴──────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                      MCP Server Pool                                     │
│                                                                         │
│  ┌─── mcp-filesystem ──────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │   │
│  │  │ Pod-fs-1 │  │ Pod-fs-2 │  │ Pod-fs-3 │                      │   │
│  │  │ :8081    │  │ :8081    │  │ :8081    │                      │   │
│  │  └──────────┘  └──────────┘  └──────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─── mcp-database ────────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │   │
│  │  │ Pod-db-1 │  │ Pod-db-2 │  │ Pod-db-3 │                      │   │
│  │  │ :8082    │  │ :8082    │  │ :8082    │                      │   │
│  └──────────┘  └──────────┘  └──────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─── mcp-api-gateway ─────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │   │
│  │  │ Pod-api-1│  │ Pod-api-2│  │ Pod-api-3│                      │   │
│  │  │ :8083    │  │ :8083    │  │ :8083    │                      │   │
│  └──────────┘  └──────────┘  └──────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 MCP Gateway实现

**gateway/main.py**

```python
import os
import json
import time
import uuid
import httpx
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from common.auth import TokenVerifier
from common.logging import AuditLogger

app = FastAPI(title="MCP Gateway")
audit = AuditLogger("mcp-gateway")

verifier = TokenVerifier(
    issuer=os.getenv("OAUTH2_ISSUER", ""),
    audience=os.getenv("OAUTH2_AUDIENCE", "mcp-gateway"),
)

CONSUL_HOST = os.getenv("CONSUL_HOST", "consul:8500")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "600"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CB_THRESHOLD", "5"))
CIRCUIT_BREAKER_RESET = int(os.getenv("CB_RESET_SECONDS", "60"))

_circuit_breakers: dict[str, dict] = {}
_rate_limits: dict[str, list] = {}


async def _discover_service(service_name: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"http://{CONSUL_HOST}/v1/health/service/{service_name}",
            params={"passing": "true"},
        )
        resp.raise_for_status()
        services = []
        for entry in resp.json():
            svc = entry["Service"]
            services.append({
                "id": svc["ID"],
                "address": svc["Address"],
                "port": svc["Port"],
                "metadata": svc.get("Meta", {}),
            })
        return services


def _select_instance(instances: list[dict], strategy: str = "round_robin") -> dict:
    if not instances:
        raise HTTPException(status_code=503, detail=f"无可用实例")

    if strategy == "round_robin":
        idx = int(time.time() * 10) % len(instances)
        return instances[idx]
    elif strategy == "random":
        import random
        return random.choice(instances)
    elif strategy == "least_conn":
        return min(instances, key=lambda x: x.get("metadata", {}).get("connections", 0))
    return instances[0]


def _check_circuit_breaker(service_name: str) -> bool:
    cb = _circuit_breakers.get(service_name, {"failures": 0, "open": False, "last_failure": 0})
    if cb["open"]:
        if time.time() - cb["last_failure"] > CIRCUIT_BREAKER_RESET:
            cb["open"] = False
            cb["failures"] = 0
            _circuit_breakers[service_name] = cb
            return True
        return False
    return True


def _record_failure(service_name: str):
    cb = _circuit_breakers.get(service_name, {"failures": 0, "open": False, "last_failure": 0})
    cb["failures"] += 1
    cb["last_failure"] = time.time()
    if cb["failures"] >= CIRCUIT_BREAKER_THRESHOLD:
        cb["open"] = True
    _circuit_breakers[service_name] = cb


def _record_success(service_name: str):
    cb = _circuit_breakers.get(service_name, {"failures": 0, "open": False, "last_failure": 0})
    cb["failures"] = 0
    cb["open"] = False
    _circuit_breakers[service_name] = cb


def _check_rate_limit(client_id: str) -> bool:
    now = time.time()
    window = 60
    calls = _rate_limits.get(client_id, [])
    calls = [t for t in calls if now - t < window]
    if len(calls) >= RATE_LIMIT_RPM:
        _rate_limits[client_id] = calls
        return False
    calls.append(now)
    _rate_limits[client_id] = calls
    return True


@app.post("/mcp/{service_name}")
async def proxy_mcp_request(service_name: str, request: Request):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    start = time.time()

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = await verifier.verify(token)
        if not payload:
            raise HTTPException(status_code=401, detail="无效的访问令牌")
        client_id = payload.get("sub", "anonymous")
    else:
        client_id = "anonymous"

    if not _check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="请求频率超限")

    if not _check_circuit_breaker(service_name):
        raise HTTPException(status_code=503, detail=f"服务 {service_name} 熔断中")

    instances = await _discover_service(service_name)
    instance = _select_instance(instances, strategy="round_robin")
    target_url = f"http://{instance['address']}:{instance['port']}/mcp"

    body = await request.body()
    headers = {
        "Content-Type": "application/json",
        "X-Trace-ID": trace_id,
        "X-Client-ID": client_id,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(target_url, content=body, headers=headers)

        _record_success(service_name)
        latency = int((time.time() - start) * 1000)
        audit.log_tool_call(
            "proxy_request",
            {"service": service_name, "instance": instance["id"]},
            f"status={resp.status_code}",
            caller_id=client_id,
            trace_id=trace_id,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    except Exception as e:
        _record_failure(service_name)
        latency = int((time.time() - start) * 1000)
        audit.log_tool_call(
            "proxy_request",
            {"service": service_name, "instance": instance["id"]},
            None,
            caller_id=client_id,
            trace_id=trace_id,
            error=str(e),
        )
        raise HTTPException(status_code=502, detail=f"上游服务错误: {e}")


@app.get("/health")
async def gateway_health():
    return {"status": "healthy", "circuit_breakers": {
        k: {"open": v["open"], "failures": v["failures"]}
        for k, v in _circuit_breakers.items()
    }}


@app.get("/services")
async def list_services():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://{CONSUL_HOST}/v1/catalog/services")
        mcp_services = {
            k: v for k, v in resp.json().items()
            if "mcp" in v
        }
    return mcp_services
```

### 4.3 服务注册与发现

**服务自动注册（每个MCP Server启动时执行）**

```python
import os
import httpx
import socket
import asyncio
import logging

logger = logging.getLogger("mcp.registry")

CONSUL_HOST = os.getenv("CONSUL_HOST", "consul:8500")


async def register_service(
    service_name: str,
    service_id: str,
    port: int,
    tags: list[str] = None,
    meta: dict = None,
    check_interval: int = 10,
):
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    registration = {
        "ID": service_id,
        "Name": service_name,
        "Address": ip_address,
        "Port": port,
        "Tags": tags or [],
        "Meta": meta or {},
        "Check": {
            "HTTP": f"http://{ip_address}:{port}/health",
            "Interval": f"{check_interval}s",
            "Timeout": "5s",
            "DeregisterCriticalServiceAfter": "30s",
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"http://{CONSUL_HOST}/v1/agent/service/register",
            json=registration,
        )
        resp.raise_for_status()
    logger.info(f"服务注册成功: {service_name} ({service_id}) @ {ip_address}:{port}")


async def deregister_service(service_id: str):
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"http://{CONSUL_HOST}/v1/agent/service/deregister/{service_id}",
        )
        resp.raise_for_status()
    logger.info(f"服务注销成功: {service_id}")
```

### 4.4 负载均衡策略

```
负载均衡策略对比:

┌──────────────┬────────────────────────────────┬──────────────────────┐
│ 策略         │ 算法                           │ 适用场景              │
├──────────────┼────────────────────────────────┼──────────────────────┤
│ round_robin  │ 轮询，按顺序分配               │ Server规格一致        │
│ weighted     │ 加权轮询，按权重分配            │ Server规格不同        │
│ least_conn   │ 最少连接数优先                 │ 长连接/耗时差异大     │
│ random       │ 随机选择                       │ 无状态简单场景        │
│ consistent   │ 一致性哈希（按trace_id）       │ 需要会话亲和          │
│ geo_aware    │ 就近路由（按可用区）            │ 多地域部署            │
└──────────────┴────────────────────────────────┴──────────────────────┘

Gateway路由决策流程:

Client Request
    ↓
┌─────────────────┐
│ 1. 解析目标服务  │  从URL路径提取service_name
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. 服务发现      │  查询Consul获取健康实例列表
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. 熔断检查      │  检查目标服务是否熔断
│  - 熔断中 → 503 │
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. 负载均衡选择  │  根据策略选择目标实例
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. 请求转发      │  附加TraceID、ClientID
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. 结果处理      │  成功→重置熔断计数
│                  │  失败→增加失败计数
└─────────────────┘
```

### 4.5 健康检查机制

```
健康检查层级:

L1: Agent本地检查 (Consul Agent → MCP Server /health)
    - 频率: 每10秒
    - 超时: 5秒
    - 连续失败3次 → 标记为critical
    - critical超过30秒 → 自动注销

L2: Gateway主动检查 (Gateway → MCP Server /health)
    - 频率: 每30秒
    - 检查项: HTTP状态 + 响应体status字段
    - 异常 → 从路由池临时移除

L3: 深度健康检查 (MCP Server内部)
    - 数据库连接池状态
    - 外部依赖可达性
    - 内存/文件描述符使用率
    - 返回degraded状态时降低流量

健康状态流转:

healthy ──(检查失败)──→ warning ──(连续失败)──→ critical
   ↑                                                  │
   └────────────(检查成功 × 3)─────────────────────────┘
```

---

## 5. 安全与权限

### 5.1 安全架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         安全架构全景                                   │
│                                                                      │
│  LLM应用                                                             │
│      │                                                               │
│      ↓ OAuth2 Bearer Token                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    MCP Gateway                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Token    │  │ 权限     │  │ 限流     │  │ 审计     │   │   │
│  │  │ 验证     │  │ 校验     │  │ 控制     │  │ 日志     │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│      │ 附加 X-Client-ID, X-Permissions, X-Trace-ID                  │
│      ↓                                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    MCP Server                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │   │
│  │  │ 工具级   │  │ 资源级   │  │ 参数级   │                  │   │
│  │  │ 权限检查 │  │ 访问控制 │  │ 脱敏过滤 │                  │   │
│  │  └──────────┘  └──────────┘  └──────────┘                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│      │                                                               │
│      ↓ 最小权限凭证                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    外部服务                                   │   │
│  │  数据库只读账号 / API受限Scope / 文件系统受限路径             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 OAuth2集成

**OAuth2认证流程**

```
LLM应用                MCP Gateway             OAuth2 Server (Keycloak)
   │                       │                          │
   │──1.获取Token────────→│                          │
   │   (client_credentials)│──2.验证Token──────────→ │
   │                       │←─3.返回用户信息+权限──── │
   │←─4.返回MCP响应───────│                          │
   │                       │                          │
```

**OAuth2配置 (Keycloak)**

```json
{
  "realm": "mcp-platform",
  "clients": [
    {
      "client_id": "mcp-llm-app",
      "client_secret": "${CLIENT_SECRET}",
      "grant_types": ["client_credentials"],
      "scope": "mcp:tools:read mcp:tools:write mcp:resources:read"
    },
    {
      "client_id": "mcp-readonly-app",
      "client_secret": "${READONLY_SECRET}",
      "grant_types": ["client_credentials"],
      "scope": "mcp:tools:read mcp:resources:read"
    }
  ],
  "scopes": [
    {"name": "mcp:tools:read", "description": "调用只读工具"},
    {"name": "mcp:tools:write", "description": "调用写入工具"},
    {"name": "mcp:resources:read", "description": "读取MCP资源"},
    {"name": "mcp:admin", "description": "管理操作"}
  ]
}
```

### 5.3 细粒度权限控制

**权限模型 (RBAC + ABAC混合)**

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Permission(Enum):
    TOOL_READ = "tool:read"
    TOOL_WRITE = "tool:write"
    TOOL_EXECUTE = "tool:execute"
    RESOURCE_READ = "resource:read"
    ADMIN = "admin"


@dataclass
class ToolPermission:
    tool_name: str
    allowed_actions: list[Permission]
    conditions: dict = field(default_factory=dict)

    def check(self, action: Permission, context: dict) -> bool:
        if action not in self.allowed_actions:
            return False
        for key, expected in self.conditions.items():
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


@dataclass
class Role:
    name: str
    tool_permissions: list[ToolPermission] = field(default_factory=list)

    def can_use_tool(self, tool_name: str, action: Permission, context: dict) -> bool:
        for perm in self.tool_permissions:
            if perm.tool_name == tool_name or perm.tool_name == "*":
                if perm.check(action, context):
                    return True
        return False


ROLES = {
    "analyst": Role(
        name="analyst",
        tool_permissions=[
            ToolPermission("read_file", [Permission.TOOL_EXECUTE],
                          conditions={"path_prefix": ["/data/reports"]}),
            ToolPermission("execute_query", [Permission.TOOL_EXECUTE],
                          conditions={"schema": ["public", "analytics"]}),
            ToolPermission("list_tables", [Permission.TOOL_EXECUTE]),
            ToolPermission("describe_table", [Permission.TOOL_EXECUTE]),
            ToolPermission("call_api", [Permission.TOOL_EXECUTE],
                          conditions={"api_name": ["weather"]}),
        ],
    ),
    "developer": Role(
        name="developer",
        tool_permissions=[
            ToolPermission("*", [Permission.TOOL_EXECUTE],
                          conditions={"path_prefix": ["/data/documents", "/data/uploads"]}),
            ToolPermission("execute_query", [Permission.TOOL_EXECUTE],
                          conditions={"schema": ["public"]}),
            ToolPermission("execute_write", [Permission.TOOL_EXECUTE],
                          conditions={"schema": ["public"]}),
            ToolPermission("call_api", [Permission.TOOL_EXECUTE]),
        ],
    ),
    "admin": Role(
        name="admin",
        tool_permissions=[
            ToolPermission("*", [Permission.TOOL_EXECUTE, Permission.ADMIN]),
        ],
    ),
}


class PermissionGuard:
    def __init__(self, roles_config: dict[str, Role] = None):
        self.roles = roles_config or ROLES

    def check(self, role_name: str, tool_name: str, action: Permission, context: dict) -> bool:
        role = self.roles.get(role_name)
        if not role:
            return False
        return role.can_use_tool(tool_name, action, context)
```

### 5.4 审计日志系统

**审计日志架构**

```
工具调用审计链路:

MCP Client → MCP Gateway → MCP Server → 外部服务
    │              │              │              │
    │  trace_id    │  trace_id    │  trace_id    │
    │  client_id   │  client_id   │  client_id   │
    │              │  service     │  tool_name   │  external_call
    │              │  instance    │  args_hash   │  latency
    │              │  latency     │  result_type │  status
    │              │  status      │  error       │
    │              │              │  duration_ms │
    ↓              ↓              ↓              ↓
┌──────────────────────────────────────────────────────┐
│              Kafka (审计日志Topic)                     │
│              分区: 按trace_id哈希                      │
└──────────────────────┬───────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│              ClickHouse (审计日志存储)                 │
│              TTL: 180天                               │
│              分区: 按月                                │
└──────────────────────────────────────────────────────┘
```

**审计日志ClickHouse建表**

```sql
CREATE TABLE mcp_audit_logs (
    trace_id        String,
    timestamp       DateTime64(3),
    server_name     String,
    event_type      String,       -- tool_call, resource_read, error
    tool_name       String,
    caller_id       String,
    client_ip       String,
    arguments_hash  String,
    result_type     String,
    error_message   Nullable(String),
    duration_ms     UInt32,
    status_code     UInt16,
    metadata        String        -- JSON格式附加信息
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, trace_id)
TTL timestamp + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;
```

**审计日志查询示例**

```sql
-- 查询某用户最近24小时的工具调用
SELECT trace_id, timestamp, tool_name, duration_ms, error_message
FROM mcp_audit_logs
WHERE caller_id = 'app:analytics-bot'
  AND timestamp > now() - INTERVAL 24 HOUR
ORDER BY timestamp DESC
LIMIT 100;

-- 统计各工具调用频率和平均延迟
SELECT tool_name,
       count() AS call_count,
       avg(duration_ms) AS avg_latency_ms,
       quantile(0.99)(duration_ms) AS p99_latency_ms,
       countIf(error_message != '') AS error_count
FROM mcp_audit_logs
WHERE timestamp > now() - INTERVAL 7 DAY
GROUP BY tool_name
ORDER BY call_count DESC;

-- 检测异常调用（频率突增）
SELECT tool_name, toStartOfHour(timestamp) AS hour,
       count() AS calls
FROM mcp_audit_logs
WHERE timestamp > now() - INTERVAL 24 HOUR
GROUP BY tool_name, hour
HAVING calls > (
    SELECT avg(hourly_calls) * 3 FROM (
        SELECT tool_name, count() AS hourly_calls
        FROM mcp_audit_logs
        WHERE timestamp BETWEEN now() - INTERVAL 7 DAY AND now() - INTERVAL 1 DAY
        GROUP BY tool_name, toStartOfHour(timestamp)
    ) t
)
ORDER BY calls DESC;
```

---

## 6. 生产部署

### 6.1 Docker容器化

**servers/filesystem/Dockerfile**

```dockerfile
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common/ /app/common/
COPY servers/filesystem/ /app/servers/filesystem/

RUN pip install --no-cache-dir \
    mcp[cli]>=1.6.0 \
    uvicorn>=0.34.0 \
    pydantic-settings>=2.7.0 \
    aiofiles>=24.1.0 \
    httpx>=0.28.0 \
    asyncpg>=0.30.0 \
    python-jose[cryptography]>=3.3.0

ENV MCP_SERVER_NAME=filesystem-mcp-server
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8081
ENV PYTHONPATH=/app

EXPOSE 8081

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

CMD ["python", "-m", "uvicorn", "servers.filespath.server:app", "--host", "0.0.0.0", "--port", "8081"]
```

**servers/database/Dockerfile**

```dockerfile
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common/ /app/common/
COPY servers/database/ /app/servers/database/

RUN pip install --no-cache-dir \
    mcp[cli]>=1.6.0 \
    uvicorn>=0.34.0 \
    pydantic-settings>=2.7.0 \
    asyncpg>=0.30.0 \
    python-jose[cryptography]>=3.3.0 \
    httpx>=0.28.0

ENV MCP_SERVER_NAME=database-mcp-server
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8082
ENV PYTHONPATH=/app

EXPOSE 8082

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

CMD ["python", "-m", "uvicorn", "servers.database.server:app", "--host", "0.0.0.0", "--port", "8082"]
```

**servers/api_gateway/Dockerfile**

```dockerfile
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY common/ /app/common/
COPY servers/api_gateway/ /app/servers/api_gateway/

RUN pip install --no-cache-dir \
    mcp[cli]>=1.6.0 \
    uvicorn>=0.34.0 \
    pydantic-settings>=2.7.0 \
    httpx>=0.28.0 \
    python-jose[cryptography]>=3.3.0

ENV MCP_SERVER_NAME=api-gateway-mcp-server
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8083
ENV PYTHONPATH=/app

EXPOSE 8083

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

CMD ["python", "-m", "uvicorn", "servers.api_gateway.server:app", "--host", "0.0.0.0", "--port", "8083"]
```

**docker-compose.yaml**

```yaml
version: "3.9"

services:
  consul:
    image: hashicorp/consul:1.20
    command: agent -dev -client=0.0.0.0 -log-level=warn
    ports:
      - "8500:8500"
    volumes:
      - consul-data:/consul/data
    healthcheck:
      test: ["CMD", "consul", "members"]
      interval: 10s
      timeout: 5s

  mcp-filesystem:
    build:
      context: .
      dockerfile: servers/filesystem/Dockerfile
    ports:
      - "8081:8081"
    environment:
      MCP_SERVER_NAME: filesystem-mcp-server
      MCP_PORT: "8081"
      CONSUL_HOST: consul:8500
      FS_ALLOWED_ROOTS: /data/documents,/data/reports
    volumes:
      - fs-data:/data
    depends_on:
      consul:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: "0.5"
    restart: unless-stopped

  mcp-database:
    build:
      context: .
      dockerfile: servers/database/Dockerfile
    ports:
      - "8082:8082"
    environment:
      MCP_SERVER_NAME: database-mcp-server
      MCP_PORT: "8082"
      CONSUL_HOST: consul:8500
      DATABASE_URL: postgresql://mcp_reader:${DB_PASSWORD}@postgres:5432/knowledge
      DB_READ_ONLY: "true"
    depends_on:
      consul:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: "0.5"
    restart: unless-stopped

  mcp-api-gateway:
    build:
      context: .
      dockerfile: servers/api_gateway/Dockerfile
    ports:
      - "8083:8083"
    environment:
      MCP_SERVER_NAME: api-gateway-mcp-server
      MCP_PORT: "8083"
      CONSUL_HOST: consul:8500
      API_REGISTRY_FILE: /config/api_registry.json
    volumes:
      - ./servers/api_gateway/api_registry.json:/config/api_registry.json:ro
    depends_on:
      consul:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: "0.5"
    restart: unless-stopped

  mcp-gateway:
    build:
      context: .
      dockerfile: gateway/Dockerfile
    ports:
      - "8080:8080"
    environment:
      CONSUL_HOST: consul:8500
      OAUTH2_ISSUER: https://auth.company.com/realms/mcp-platform
      OAUTH2_AUDIENCE: mcp-gateway
      RATE_LIMIT_RPM: "600"
    depends_on:
      consul:
        condition: service_healthy
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: "1.0"
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: knowledge
      POSTGRES_USER: mcp_reader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pg-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  consul-data:
  fs-data:
  pg-data:
```

### 6.2 Kubernetes编排

**k8s/namespace.yaml**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mcp-platform
  labels:
    app.kubernetes.io/part-of: mcp-platform
    environment: production
```

**k8s/filesystem-server.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-filesystem
  namespace: mcp-platform
  labels:
    app: mcp-filesystem
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: mcp-filesystem
  template:
    metadata:
      labels:
        app: mcp-filesystem
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8081"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: mcp-filesystem
          image: registry.company.com/mcp-filesystem:1.0.0
          ports:
            - containerPort: 8081
              name: http
          env:
            - name: MCP_SERVER_NAME
              value: filesystem-mcp-server
            - name: MCP_PORT
              value: "8081"
            - name: CONSUL_HOST
              value: "consul.consul.svc:8500"
            - name: FS_ALLOWED_ROOTS
              value: /data/documents,/data/reports
          envFrom:
            - secretRef:
                name: mcp-common-secrets
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          volumeMounts:
            - name: data
              mountPath: /data
          readinessProbe:
            httpGet:
              path: /health
              port: 8081
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8081
            initialDelaySeconds: 15
            periodSeconds: 20
            timeoutSeconds: 5
            failureThreshold: 3
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: mcp-filesystem-pvc
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: mcp-filesystem
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-filesystem
  namespace: mcp-platform
spec:
  selector:
    app: mcp-filesystem
  ports:
    - port: 8081
      targetPort: 8081
      name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-filesystem-hpa
  namespace: mcp-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-filesystem
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

**k8s/database-server.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-database
  namespace: mcp-platform
  labels:
    app: mcp-database
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: mcp-database
  template:
    metadata:
      labels:
        app: mcp-database
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8082"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: mcp-database
          image: registry.company.com/mcp-database:1.0.0
          ports:
            - containerPort: 8082
              name: http
          env:
            - name: MCP_SERVER_NAME
              value: database-mcp-server
            - name: MCP_PORT
              value: "8082"
            - name: CONSUL_HOST
              value: "consul.consul.svc:8500"
            - name: DB_READ_ONLY
              value: "true"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: mcp-database-secrets
                  key: database-url
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          readinessProbe:
            httpGet:
              path: /health
              port: 8082
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8082
            initialDelaySeconds: 20
            periodSeconds: 20
            timeoutSeconds: 5
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: mcp-database
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-database
  namespace: mcp-platform
spec:
  selector:
    app: mcp-database
  ports:
    - port: 8082
      targetPort: 8082
      name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-database-hpa
  namespace: mcp-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-database
  minReplicas: 3
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

**k8s/mcp-gateway.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-gateway
  namespace: mcp-platform
  labels:
    app: mcp-gateway
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mcp-gateway
  template:
    metadata:
      labels:
        app: mcp-gateway
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: mcp-gateway
          image: registry.company.com/mcp-gateway:1.0.0
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: CONSUL_HOST
              value: "consul.consul.svc:8500"
            - name: OAUTH2_ISSUER
              value: "https://auth.company.com/realms/mcp-platform"
            - name: OAUTH2_AUDIENCE
              value: "mcp-gateway"
            - name: RATE_LIMIT_RPM
              value: "600"
            - name: CB_THRESHOLD
              value: "5"
            - name: CB_RESET_SECONDS
              value: "60"
          envFrom:
            - secretRef:
                name: mcp-gateway-secrets
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-gateway
  namespace: mcp-platform
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  selector:
    app: mcp-gateway
  ports:
    - port: 80
      targetPort: 8080
      name: http
  type: LoadBalancer
```

### 6.3 监控告警

**监控架构图**

```
┌──────────────────────────────────────────────────────────────────┐
│                       监控体系全景                                 │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ MCP Gateway │  │ MCP Server  │  │ Consul      │             │
│  │ /metrics    │  │ /metrics    │  │ /v1/metrics │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┼────────────────┘                     │
│                          ↓ Prometheus Pull                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Prometheus Server                        │   │
│  │  scrape_interval: 15s                                    │   │
│  │  evaluation_interval: 15s                                │   │
│  │  retention: 30d                                          │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Alertmanager                             │   │
│  │  - 分组聚合                                               │   │
│  │  - 抑制/静默                                              │   │
│  │  - 路由: 邮件/钉钉/企业微信/PagerDuty                    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Grafana                                  │   │
│  │  - MCP集群总览大盘                                       │   │
│  │  - 单服务详情面板                                        │   │
│  │  - 审计日志分析面板                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Prometheus配置**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: mcp-production
    env: prod

scrape_configs:
  - job_name: mcp-gateway
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [mcp-platform]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: mcp-gateway
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        target_label: __address__
        regex: (.+)
        replacement: ${1}:8080

  - job_name: mcp-servers
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [mcp-platform]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: mcp-(filesystem|database|api-gateway)
        action: keep

  - job_name: consul
    static_configs:
      - targets: ["consul.consul.svc:8500"]

rule_files:
  - /etc/prometheus/rules/*.yml
```

**告警规则**

```yaml
groups:
  - name: mcp-platform
    rules:
      - alert: MCPServerHighLatency
        expr: histogram_quantile(0.99, rate(mcp_tool_call_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MCP工具调用P99延迟超过2秒"
          description: "服务 {{ $labels.server }} 的工具 {{ $labels.tool_name }} P99延迟为 {{ $value }}s"

      - alert: MCPServerErrorRate
        expr: rate(mcp_tool_call_errors_total[5m]) / rate(mcp_tool_call_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "MCP工具调用错误率超过5%"
          description: "服务 {{ $labels.server }} 错误率为 {{ $value | humanizePercentage }}"

      - alert: MCPCircuitBreakerOpen
        expr: mcp_circuit_breaker_open == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MCP服务熔断器开启"
          description: "服务 {{ $labels.service_name }} 熔断器已开启，连续失败 {{ $labels.failures }} 次"

      - alert: MCPInstanceDown
        expr: up{job="mcp-servers"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "MCP Server实例不可达"
          description: "实例 {{ $labels.instance }} 已离线超过2分钟"

      - alert: MCPHighMemory
        expr: container_memory_working_set_bytes{namespace="mcp-platform"} / container_spec_memory_limit_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MCP容器内存使用率超过85%"
          description: "Pod {{ $labels.pod }} 内存使用率 {{ $value | humanizePercentage }}"

      - alert: MCPRateLimitTriggered
        expr: rate(mcp_rate_limit_rejected_total[5m]) > 10
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "MCP网关限流触发频繁"
          description: "客户端 {{ $labels.client_id }} 被限流 {{ $value }} 次/秒"
```

**Grafana Dashboard JSON (核心面板)**

```json
{
  "dashboard": {
    "title": "MCP Platform Overview",
    "panels": [
      {
        "title": "工具调用QPS",
        "type": "timeseries",
        "targets": [{"expr": "sum(rate(mcp_tool_call_total[1m])) by (server)"}]
      },
      {
        "title": "P99延迟",
        "type": "timeseries",
        "targets": [{"expr": "histogram_quantile(0.99, sum(rate(mcp_tool_call_duration_seconds_bucket[5m])) by (le, server))"}]
      },
      {
        "title": "错误率",
        "type": "gauge",
        "targets": [{"expr": "sum(rate(mcp_tool_call_errors_total[5m])) / sum(rate(mcp_tool_call_total[5m]))"}]
      },
      {
        "title": "活跃实例数",
        "type": "stat",
        "targets": [{"expr": "count(up{job=\"mcp-servers\"} == 1) by (server)"}]
      },
      {
        "title": "熔断器状态",
        "type": "stat",
        "targets": [{"expr": "mcp_circuit_breaker_open"}]
      },
      {
        "title": "Top10热门工具",
        "type": "barchart",
        "targets": [{"expr": "topk(10, sum(rate(mcp_tool_call_total[1h])) by (tool_name))"}]
      }
    ]
  }
}
```

### 6.4 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 工具调用P50延迟 | < 100ms | 65ms |
| 工具调用P99延迟 | < 500ms | 320ms |
| Gateway吞吐量 | > 2000 RPS | 2500 RPS |
| 服务可用性 | > 99.95% | 99.97% |
| 熔断恢复时间 | < 60s | 45s |
| 服务注册发现延迟 | < 5s | 2s |

### 6.5 滚动升级流程

```
MCP Server滚动升级:

1. 构建新镜像 → 推送到镜像仓库
2. 更新K8s Deployment镜像版本
3. K8s自动执行滚动更新:
   ┌──────────────────────────────────────────────────────┐
   │  旧版本 (v1)          新版本 (v2)                     │
   │  ┌──────┐             ┌──────┐                       │
   │  │ Pod1 │ ──升级──→   │ Pod1'│  readinessProbe通过   │
   │  │ Pod2 │             │ Pod2 │  (旧版本继续服务)      │
   │  │ Pod3 │             │ Pod3 │                       │
   │  └──────┘             └──────┘                       │
   │                                                       │
   │  Step 1: 创建Pod1'(v2)，等待readinessProbe通过       │
   │  Step 2: Pod1'(v2)就绪后，删除Pod1(v1)              │
   │  Step 3: 重复Step1-2，直到所有Pod更新完成            │
   └──────────────────────────────────────────────────────┘
4. Consul健康检查自动更新实例列表
5. Gateway路由自动切换到新版本实例
6. 监控新版本错误率，异常则自动回滚

回滚命令:
  kubectl rollout undo deployment/mcp-filesystem -n mcp-platform
```

---

## 7. 架构演进路线

```
V1.0（当前）: 基础MCP工具服务
├── 3个核心MCP Server（文件/数据库/API）
├── MCP Gateway统一入口
├── Consul服务注册发现
└── 基础权限控制 + 审计日志

V2.0（规划中）: 智能化增强
├── MCP Server自动生成（根据OpenAPI/数据库Schema）
├── 工具选择智能路由（LLM根据意图选择最优工具链）
├── 工具组合编排（多工具串行/并行编排）
├── 流式工具调用（Server-Sent Events支持）
└── 工具效果评估（自动评估工具调用质量）

V3.0（远期）: 生态化平台
├── MCP Server市场（内部工具共享平台）
├── 跨组织MCP联邦（安全地共享工具能力）
├── 工具版本管理与灰度发布
├── 自适应限流与弹性调度
└── AI驱动的安全策略自动生成
```

---

## 8. 经验总结

### 8.1 关键成功因素

1. **协议标准化是基础**：MCP统一了工具接入协议，新增工具只需实现MCP Server，无需修改LLM应用代码
2. **网关层是核心**：MCP Gateway承担了认证、路由、限流、熔断等横切关注点，Server只需关注业务逻辑
3. **热插拔设计**：基于Consul的服务发现使得MCP Server可以随时增减，对上层应用完全透明
4. **最小权限原则**：每个Server使用独立的数据库账号/API Key，限制在最小必要权限范围内
5. **审计先行**：所有工具调用必须记录审计日志，这是安全合规和问题排查的基础

### 8.2 踩过的坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 文件系统路径遍历 | 未校验`../`相对路径 | resolve后校验是否在允许根目录下 |
| 数据库查询超时 | 用户构造复杂SQL | 设置query_timeout + 只允许SELECT |
| API密钥泄露 | 硬编码在代码中 | 统一使用环境变量 + Secret管理 |
| 熔断误触发 | 偶发超时导致熔断 | 调整阈值为5次 + 半开状态探测 |
| 服务注册延迟 | Pod启动后立即注册 | readinessProbe通过后再注册 |
| 审计日志过大 | 全量记录参数 | 参数哈希 + 结果摘要 |

### 8.3 MCP vs Function Calling对比

| 维度 | OpenAI Function Calling | MCP |
|------|------------------------|-----|
| 定义方式 | 在请求中内联JSON Schema | 独立Server暴露工具定义 |
| 动态性 | 每次请求可变 | Server端动态注册，支持list_changed通知 |
| 协议 | 无独立协议，嵌入Chat API | 独立JSON-RPC协议 |
| 传输 | 仅HTTP API | Stdio / SSE / Streamable HTTP |
| 安全 | 依赖应用层 | 内置OAuth2支持 |
| 可组合性 | 单次请求多工具 | 多Server并行，Gateway路由 |
| 生态 | OpenAI专有 | 开放标准，多厂商支持 |

---

## 深度分析

MCP工具服务架构代表了LLM应用与外部系统集成的最新范式。相比传统的Function Calling方式，MCP通过标准化的JSON-RPC 2.0协议和Streamable HTTP传输，实现了工具服务的"热插拔"——每个MCP Server可独立部署、升级和扩缩容，而LLM应用通过MCP Client动态发现和调用工具。Gateway层将认证鉴权、权限校验、路由分发、限流熔断和审计日志统一收敛，形成了工具调用的统一管控面。

从代码实现看，FastMCP框架大大简化了MCP Server的开发工作。文件系统、数据库、API网关三个Server的示例代码完整展示了工具定义、资源暴露和健康检查的标准模式。安全设计尤为关键：文件系统Server通过路径白名单和敏感扩展名过滤实现访问控制；数据库Server默认只读，写操作需显式开启；API网关Server通过域名白名单和路径黑名单双重防护。这些做法为MCP Server的安全基线提供了参考。

Consul服务注册与发现、K8s容器编排、以及Gateway层的熔断器与限流机制共同构成了MCP服务集群的生产级保障。轮询/随机/最小连接数负载均衡策略、滑动窗口限流和半开状态恢复的熔断器，使这套架构能应对日均200万次工具调用的生产压力。审计日志的精细化设计（trace_id全程传播、参数哈希脱敏）也为合规审计提供了强有力的支撑。

## Checklist

- [ ] 确认所有MCP Server实现了标准健康检查和Consul服务注册
- [ ] 验证Gateway层的认证鉴权和权限校验逻辑是否完整
- [ ] 测试熔断器在不同故障场景下的行为（超时/限流/服务不可达）
- [ ] 配置Streamable HTTP传输模式，确保支持负载均衡
- [ ] 实现工具调用的全链路Trace ID传播和审计日志记录
- [ ] 验证文件系统Server的路径越权防护和敏感扩展名过滤
- [ ] 确认数据库Server的只读模式配置和SQL注入防护
- [ ] 制定MCP Server的滚动升级和回滚策略
- [ ] 压测Gateway的限流能力，确保RPM配置准确生效
- [ ] 编写MCP工具服务目录文档，支持LLM应用动态发现

## 延伸阅读

- [Model Context Protocol 官方规范](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK 文档](https://github.com/modelcontextprotocol/python-sdk)
- [Streamable HTTP 传输协议详解](https://spec.modelcontextprotocol.io/latest/transports/streamable-http/)

*最后更新：2026-06-12*

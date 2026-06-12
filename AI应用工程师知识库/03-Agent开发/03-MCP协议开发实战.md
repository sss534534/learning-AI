# MCP 协议开发实战

> 从零构建 MCP Server 和 Client，掌握 2026 年标准的 Agent 工具协议

## 元数据

- **难度**: ⭐⭐
- **前置知识**: [Agent协议与通信架构](../../AI架构师知识库/04-Agent系统架构/03-Agent协议与通信架构.md), [MCP协议2026演进](../../AI架构师知识库/04-Agent系统架构/09-MCP协议2026演进与无状态传输.md)
- **关联文件**: [AI Agent开发实战](./01-AI-Agent开发实战.md), [Agent应用架构：MCP+A2A模式](../04-AI应用架构设计/02-Agent应用架构：MCP%2BA2A模式.md), [AI开发框架选型](../06-工具与框架/01-AI开发框架选型.md)
- **最后更新**: 2026-06-12

---

## 目录

- [1. MCP 开发基础](#1-mcp-开发基础)
  - [1.1 核心概念速览](#11-核心概念速览)
  - [1.2 环境准备](#12-环境准备)
- [2. 构建第一个 MCP Server](#2-构建第一个-mcp-server)
  - [2.1 Python Server](#21-python-server)
  - [2.2 TypeScript Server](#22-typescript-server)
  - [2.3 测试 MCP Server](#23-测试-mcp-server)
- [3. 构建 MCP Client](#3-构建-mcp-client)
- [4. MCP Server 部署选项](#4-mcp-server-部署选项)
  - [4.1 Streamable HTTP 部署](#41-streamable-http-部署)
- [5. MCP 开发最佳实践](#5-mcp-开发最佳实践)
  - [5.1 工具设计原则](#51-工具设计原则)
  - [5.2 安全注意事项](#52-安全注意事项)

---

## 1. MCP 开发基础

### 1.1 核心概念速览

| 概念 | 说明 | 开发者视角 |
|------|------|-----------|
| Server | 提供 Tools/Resources/Prompts 的服务 | 你实现的工具端 |
| Client | 与 Server 1:1 连接的客户端 | SDK 自动管理 |
| Host | 管理多个 Client 的宿主应用 | Claude Desktop / 你的 App |
| Transport | 通信协议（stdio / Streamable HTTP） | 根据部署场景选择 |

### 1.2 环境准备

```bash
# 安装 MCP Python SDK
pip install mcp

# 或 TypeScript SDK
npm install @modelcontextprotocol/sdk
```

## 2. 构建第一个 MCP Server

### 2.1 Python Server

```python
"""
MCP Server 示例：文件搜索工具
通过 stdio 传输运行，可被 Claude Desktop 或任意 MCP Client 加载
"""
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import os
import glob
from typing import Any

# 创建 Server 实例
server = Server("file-search")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """声明工具列表（MCP 发现机制）"""
    return [
        types.Tool(
            name="search_files",
            description="按文件名模式搜索文件，支持通配符",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "搜索模式，如 '*.py' 或 '**/*.md'"
                    },
                    "root_dir": {
                        "type": "string",
                        "description": "搜索根目录（可选，默认当前目录）"
                    }
                },
                "required": ["pattern"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """执行工具调用"""
    if name != "search_files":
        raise ValueError(f"Unknown tool: {name}")

    pattern = arguments["pattern"]
    root = arguments.get("root_dir", ".")

    results = glob.glob(os.path.join(root, pattern), recursive=True)

    return [types.TextContent(
        type="text",
        text=f"找到 {len(results)} 个文件:\n" + "\n".join(results[:50])
    )]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="file-search",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 2.2 TypeScript Server

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "weather-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "get_weather",
    description: "获取城市天气信息",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string", description: "城市名称" }
      },
      required: ["city"]
    }
  }]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "get_weather") {
    throw new Error("Unknown tool");
  }
  const city = request.params.arguments?.city;
  // 实际场景调用天气 API
  return {
    content: [{ type: "text", text: `${city} 天气：晴，25°C` }]
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 2.3 测试 MCP Server

```bash
# 方法 1：使用 mcp-cli 工具
mcp-cli dev path/to/server.py

# 方法 2：本地运行（stdio 模式）
python path/to/server.py

# 方法 3：使用 Claude Desktop 加载
# 在 claude_desktop_config.json 中添加：
{
  "mcpServers": {
    "file-search": {
      "command": "python",
      "args": ["path/to/server.py"]
    }
  }
}
```

## 3. 构建 MCP Client

```python
"""
MCP Client 示例：通过 MCP 协议连接远程工具
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPToolClient:
    """通用 MCP 工具客户端"""

    def __init__(self, server_command: str, server_args: list[str]):
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args
        )
        self.session = None

    async def connect(self):
        """连接 MCP Server"""
        read, write = await stdio_client(self.server_params)
        self.session = await ClientSession(read, write).__aenter__()
        await self.session.initialize()

    async def list_tools(self):
        """列出可用工具"""
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """调用工具"""
        result = await self.session.call_tool(tool_name, arguments)
        return result.content

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)

async def main():
    client = MCPToolClient("python", ["path/to/server.py"])
    await client.connect()

    # 发现工具
    tools = await client.list_tools()
    print(f"可用工具: {[t.name for t in tools]}")

    # 调用工具
    result = await client.call_tool("search_files", {"pattern": "*.py"})
    print(result)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. MCP Server 部署选项

| 传输方式 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **stdio** | Claude Desktop、本地开发 | 零配置，自动管理 | 只能本地 |
| **Streamable HTTP** | 远程服务、生产部署 | 可远程访问、可扩展 | 需要 Web 服务器 |
| **SSE（遗留）** | 向后兼容 | - | 已弃用，不推荐新项目 |

### 4.1 Streamable HTTP 部署

```python
"""基于 FastAPI 的 MCP HTTP Server"""
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.transport.http import HttpTransport

app = FastAPI()
server = Server("remote-search")

@server.list_tools()
async def list_tools():
    # ... 同前
    pass

@server.call_tool()
async def call_tool(name, arguments):
    # ... 同前
    pass

@app.post("/mcp")
async def mcp_endpoint(request: dict):
    """MCP HTTP 端点"""
    transport = HttpTransport()
    return await transport.handle_request(server, request)
```

## 5. MCP 开发最佳实践

### 5.1 工具设计原则

| 原则 | 说明 | 反例 |
|------|------|------|
| 单一职责 | 每个工具只做一件事 | `do_everything()` 函数 |
| 清晰描述 | 工具名 + 描述 + 参数说明 | `process(data)` |
| 输入验证 | 参数类型和范围检查 | 不检查直接执行 |
| 错误处理 | 返回友好错误信息 | 抛出未处理异常 |
| 幂等性 | 相同输入产生相同结果 | 每次调用副作用不同 |

### 5.2 安全注意事项

- **权限最小化**：每个工具只访问必要资源
- **输入校验**：防止注入攻击
- **限流**：防止滥用
- **审计日志**：记录每次工具调用

---

## 6. 深度分析

### 6.1 MCP vs 传统 API 集成

传统 REST API 集成是静态的——客户端在编译时就知道端点、请求格式和响应结构。集成的变更通常需要两端同时更新。MCP 通过**动态工具发现**（`list_tools`）彻底改变了这一模式：Server 声明能力，Client 在运行时获取可用工具列表及其 Schema，实现了解耦。

| 维度 | 传统 API | MCP |
|------|----------|-----|
| **发现机制** | 静态文档 / OpenAPI 规范 | 运行时动态发现（`list_tools`） |
| **组合性** | 手动编排多个 API | Host 统一管理多个 Server 的工具组合 |
| **契约绑定** | 编译时强绑定 | 运行时 Schema 协商 |
| **变更影响** | 接口变更需两端同步更新 | Server 更新能力，Client 自动感知 |

### 6.2 Streamable HTTP vs stdio

| 传输方式 | 连接模型 | 适用场景 | 部署架构 | 注意事项 |
|----------|----------|----------|----------|----------|
| **stdio** | 子进程 stdin/stdout | 本地开发、桌面集成 | 进程内 | 生命周期由 Host 管理，无需网络 |
| **Streamable HTTP** | HTTP POST 流式响应 | 远程服务、生产环境 | 独立 Web 服务 | 需要认证、TLS、负载均衡 |

**选择建议**：开发阶段用 stdio 以获得最快的迭代反馈。需要将工具暴露给远程 Agent 或集成到生产系统时，升级到 Streamable HTTP。

### 6.3 安全考量

- **工具能力范围限定**：每个工具应声明明确的能力边界，借助 `inputSchema` 约束参数范围和类型
- **认证传播**：Streamable HTTP 模式下，MCP 请求需要携带上游认证上下文（Bearer Token、mTLS 等）
- **注入风险**：工具参数直接拼接系统命令时存在命令注入风险，必须使用参数化 API 而非字符串拼接
- **权限审计**：Server 应记录每次工具调用的发起方、时间、参数和结果

### 6.4 生产就绪

```python
"""MCP Server 生产级错误处理与超时"""
from contextlib import asynccontextmanager
from mcp.server import Server

server = Server("production-server")

@server.call_tool()
async def safe_call_tool(name: str, arguments: dict) -> list:
    try:
        if not arguments or "input" not in arguments:
            return [types.TextContent(type="text", text="错误：缺少必要参数")]
        await rate_limiter.check(name)
        result = await execute_with_timeout(name, arguments, timeout=30)
        return [types.TextContent(type="text", text=str(result))]
    except TimeoutError:
        return [types.TextContent(type="text", text="错误：工具调用超时")]
    except RateLimitError:
        return [types.TextContent(type="text", text="错误：请求过于频繁，请稍后重试")]
    except Exception as e:
        logger.error(f"Tool call failed: {name} {arguments}", exc_info=e)
        return [types.TextContent(type="text", text=f"错误：{str(e)}")]
```

### 6.5 2026 MCP 生态

- **工具注册中心（Tool Registry）**：公共和私有的 MCP 工具市场，支持版本管理、评分和发现
- **MCP 网关（Gateway）**：企业级网关统一管理多个 MCP Server 的路由、认证、限流和监控
- **认证标准**：MCP 社区正推动标准化的 OAuth 2.0 集成规范，使远程 MCP 工具可以复用现有的身份体系
- **多云工具编排**：通过 MCP Gateway 聚合来自不同云 provider 的工具，实现跨云 Agent 工作流

---

## 7. Checklist

### 开发准备
- [ ] 安装 MCP SDK（Python: `pip install mcp` / TypeScript: `npm install @modelcontextprotocol/sdk`）
- [ ] 确定传输模式（本地开发用 stdio，远程部署用 Streamable HTTP）
- [ ] 安装 mcp-cli 调试工具

### Server 实现
- [ ] 定义清晰的工具名称和描述
- [ ] 为每个工具设计完整的 `inputSchema`（类型、描述、必填项）
- [ ] 实现 `list_tools` 和 `call_tool` 处理器
- [ ] 添加输入验证和错误处理
- [ ] 本地运行并用 mcp-cli 验证工具调用

### Client 集成
- [ ] 初始化 ClientSession
- [ ] 调用 `list_tools` 获取 Server 能力列表
- [ ] 根据业务逻辑选择并调用工具
- [ ] 处理工具返回结果和错误
- [ ] 实现 session 生命周期管理（连接、重连、关闭）

### 安全加固
- [ ] 最小权限原则：每个工具只赋予必要的数据访问权限
- [ ] 参数注入防护：使用参数化查询而非字符串拼接
- [ ] Streamable HTTP 模式配置 TLS 和认证
- [ ] 添加速率限制防止滥用
- [ ] 启用审计日志记录每次调用

### 生产部署
- [ ] 选择 Streamable HTTP 传输（或基于 WSGI/ASGI 的自定义传输）
- [ ] 配置健康检查和优雅关闭
- [ ] 设置合理的超时时间（工具调用建议 30-60s）
- [ ] 集成监控和告警
- [ ] 编写部署文档和运行手册

---

## 8. 延伸阅读

### 内部参考
- [AI Agent开发实战](./01-AI-Agent开发实战.md) — Agent 系统中整合 MCP 工具的完整案例
- [Agent应用架构：MCP+A2A模式](../04-AI应用架构设计/02-Agent应用架构：MCP%2BA2A模式.md) — MCP 与 A2A 协议的协同架构
- [Agent协议与通信架构](../../AI架构师知识库/04-Agent系统架构/03-Agent协议与通信架构.md) — 协议层理论基础
- [MCP协议2026演进](../../AI架构师知识库/04-Agent系统架构/09-MCP协议2026演进与无状态传输.md) — 协议演进方向与无状态传输设计

### 外部资源
1. [MCP 官方规范](https://spec.modelcontextprotocol.io) — 协议完整定义和传输层细节
2. [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python SDK 源码与示例
3. [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript SDK 源码与示例
4. [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) — 社区维护的 MCP Server 列表
5. [Anthropic MCP 文档](https://modelcontextprotocol.io) — 官方入门指南和最佳实践

---

*最后更新：2026-06-12*

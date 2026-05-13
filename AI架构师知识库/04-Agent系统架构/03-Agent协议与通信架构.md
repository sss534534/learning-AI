# Agent协议与通信架构

> 从MCP工具协议到A2A互操作，构建Agent互联网的通信基石

## 目录
1. [MCP协议（Model Context Protocol）](#1-mcp协议model-context-protocol)
2. [A2A协议（Agent2Agent）](#2-a2a协议agent2agent)
3. [ACP/UCP等其他协议](#3-acpucp等其他协议)
4. [Agentic RAG架构](#4-agentic-rag架构)
5. [推理模型Agent架构](#5-推理模型agent架构)

---

## 1. MCP协议（Model Context Protocol）

### 1.1 MCP概述

**定义：** MCP是由Anthropic于2024年11月发布的开放协议，旨在标准化LLM应用与外部数据源、工具之间的连接方式。如同USB-C统一了设备接口，MCP统一了AI模型的上下文接入方式。

**核心价值：**
- **标准化**：一次开发，所有MCP兼容客户端可用
- **解耦**：工具提供方与模型消费方独立演进
- **安全**：内置权限模型与沙箱机制
- **生态**：2025年已积累10000+公开MCP服务器

**协议版本演进：**
```
2024-11  MCP 1.0    初始发布，定义核心原语
2025-03  MCP 2025-03-26  新增Streamable HTTP传输、OAuth 2.1认证
2025-XX  MCP 2.0(草案)   增强采样协议、结构化资源
```

### 1.2 MCP三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Host（宿主）                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  LLM应用（Claude Desktop / VS Code / Cursor / 自定义App） │  │
│  │  - 管理多个MCP Client                                     │  │
│  │  - 协调工具调用与结果聚合                                   │  │
│  │  - 控制安全策略与权限边界                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│         │                    │                    │              │
│    ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐        │
│    │MCP Client│        │MCP Client│        │MCP Client│        │
│    │ (1:1连接) │        │ (1:1连接) │        │ (1:1连接) │        │
│    └────┬─────┘        └────┬─────┘        └────┬─────┘        │
└─────────┼───────────────────┼───────────────────┼──────────────┘
          │                   │                   │
    ┌─────▼──────┐     ┌─────▼──────┐     ┌─────▼──────┐
    │MCP Server  │     │MCP Server  │     │MCP Server  │
    │ (文件系统)  │     │ (数据库)    │     │ (Web搜索)   │
    │ Tools:     │     │ Tools:     │     │ Tools:     │
    │  read_file │     │  query_db  │     │  search    │
    │ Resources: │     │ Resources: │     │ Resources: │
    │  /docs/*   │     │  /schema   │     │  /results  │
    └────────────┘     └────────────┘     └────────────┘
```

**三层职责：**

| 层级 | 角色 | 职责 | 示例 |
|------|------|------|------|
| **Host** | 宿主应用 | 管理Client生命周期、聚合结果、安全策略 | Claude Desktop、VS Code |
| **Client** | 协议客户端 | 与Server建立1:1连接、协议协商、消息路由 | 内置于Host的连接管理器 |
| **Server** | 服务提供者 | 暴露Tools/Resources/Prompts、执行操作 | 文件系统、数据库、API适配器 |

### 1.3 MCP核心原语

MCP定义了三大类原语（Primitives），构成协议的功能骨架：

```
┌─────────────────────────────────────────────────────┐
│                  MCP Primitives                      │
├─────────────┬───────────────┬───────────────────────┤
│   Tools     │  Resources    │      Prompts          │
│  (工具)      │  (资源)       │      (提示模板)        │
├─────────────┼───────────────┼───────────────────────┤
│ 模型调用的   │ 模型读取的     │ 模型使用的             │
│ 可执行函数   │ 结构化数据     │ 预定义提示             │
├─────────────┼───────────────┼───────────────────────┤
│ tools/list  │ resources/list│ prompts/list          │
│ tools/call  │ resources/read│ prompts/get           │
│             │ resources/    │                       │
│             │  subscribe    │                       │
└─────────────┴───────────────┴───────────────────────┘
```

**工具（Tools）：**
```json
{
  "name": "query_database",
  "description": "执行SQL查询并返回结果",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql": {
        "type": "string",
        "description": "SQL查询语句"
      },
      "database": {
        "type": "string",
        "enum": ["production", "staging", "analytics"],
        "description": "目标数据库"
      }
    },
    "required": ["sql"]
  }
}
```

**资源（Resources）：**
```json
{
  "uri": "postgres://db.example.com/users/schema",
  "name": "用户表结构",
  "description": "用户表的完整Schema定义",
  "mimeType": "application/json"
}
```

**提示模板（Prompts）：**
```json
{
  "name": "code_review",
  "description": "代码审查提示模板",
  "arguments": [
    {
      "name": "language",
      "description": "编程语言",
      "required": true
    },
    {
      "name": "focus_area",
      "description": "审查重点",
      "required": false
    }
  ]
}
```

### 1.4 工具发现与调用机制

**工具发现流程：**
```
Host                    Client                  Server
 │                        │                        │
 │  初始化连接              │                        │
 │───────────────────────>│  initialize             │
 │                        │───────────────────────>│
 │                        │  initialize响应          │
 │                        │<───────────────────────│
 │                        │  initialized            │
 │                        │───────────────────────>│
 │                        │                        │
 │  请求工具列表            │                        │
 │───────────────────────>│  tools/list             │
 │                        │───────────────────────>│
 │                        │  返回工具定义列表         │
 │                        │<───────────────────────│
 │  展示可用工具            │                        │
 │<───────────────────────│                        │
```

**工具调用时序：**
```
LLM                     Host                    Client               Server
 │                        │                        │                     │
 │  生成tool_call          │                        │                     │
 │  (name, arguments)     │                        │                     │
 │───────────────────────>│                        │                     │
 │                        │  路由到对应Client       │                     │
 │                        │───────────────────────>│                     │
 │                        │                        │  tools/call          │
 │                        │                        │  {name, arguments}   │
 │                        │                        │────────────────────>│
 │                        │                        │                     │
 │                        │                        │     ┌───────────┐   │
 │                        │                        │     │ 执行工具   │   │
 │                        │                        │     │ 返回结果   │   │
 │                        │                        │     └───────────┘   │
 │                        │                        │  CallResult          │
 │                        │                        │<────────────────────│
 │                        │  返回工具执行结果       │                     │
 │                        │<───────────────────────│                     │
 │  将结果注入上下文        │                        │                     │
 │<───────────────────────│                        │                     │
 │  继续生成...            │                        │                     │
```

**工具调用代码示例：**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server_postgres"],
    env={"DATABASE_URL": "postgresql://localhost/mydb"}
)

async def call_tool_example():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")

            result = await session.call_tool(
                "query_database",
                arguments={"sql": "SELECT COUNT(*) FROM users"}
            )
            for content in result.content:
                print(content.text)
```

### 1.5 资源管理

**资源类型：**

| 类型 | URI格式 | 说明 | 示例 |
|------|---------|------|------|
| **文本资源** | `file:///path/to/file` | 文本内容 | 代码文件、配置 |
| **二进制资源** | `binary:///blob/id` | Base64编码 | 图片、PDF |
| **模板资源** | `db://{table}/schema` | URI模板参数化 | 数据库Schema |

**资源订阅机制：**
```
Client                          Server
  │                               │
  │  resources/subscribe          │
  │  {uri: "file:///logs/app"}    │
  │──────────────────────────────>│
  │                               │
  │     (文件内容变化)              │
  │                               │
  │  notifications/resources      │
  │  /updated                     │
  │<──────────────────────────────│
  │                               │
  │  resources/read               │
  │  {uri: "file:///logs/app"}    │
  │──────────────────────────────>│
  │  返回最新内容                   │
  │<──────────────────────────────│
```

### 1.6 采样协议（Sampling）

采样协议允许MCP Server向Host请求LLM推理能力，实现"反向调用"：

```
┌─────────┐                    ┌─────────┐
│  Host   │                    │  Server │
│ (LLM)   │                    │ (工具)   │
└────┬────┘                    └────┬────┘
     │                              │
     │  tools/call (正向调用)         │
     │─────────────────────────────>│
     │                              │
     │  sampling/createMessage      │
     │  (Server请求LLM推理)          │
     │<─────────────────────────────│
     │                              │
     │  人类审批（可选）              │
     │  ┌─────────────┐             │
     │  │ 用户确认/拒绝 │             │
     │  └─────────────┘             │
     │                              │
     │  返回LLM生成结果              │
     │─────────────────────────────>│
     │                              │
     │  返回工具最终结果              │
     │<─────────────────────────────│
```

**采样请求示例：**
```json
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "分析以下SQL查询的性能问题：SELECT * FROM orders WHERE status = 'pending'"
        }
      }
    ],
    "maxTokens": 1024,
    "systemPrompt": "你是一个数据库性能优化专家",
    "includeContext": "thisServer"
  }
}
```

### 1.7 传输层

MCP支持两种传输机制：

**1. Stdio传输（本地进程间通信）：**
```
┌──────────────┐    stdin/stdout    ┌──────────────┐
│  MCP Client  │<=================>│  MCP Server  │
│  (Host进程)   │                   │  (子进程)     │
└──────────────┘                   └──────────────┘

特点:
- 同机通信，零网络开销
- Server作为Host子进程启动
- 适合桌面应用场景
- 每个Client-Server对独占进程
```

**2. Streamable HTTP传输（远程通信）：**
```
┌──────────────┐     HTTP POST      ┌──────────────┐
│  MCP Client  │===================>│  MCP Server  │
│              │<===================│  (远程服务)   │
│              │   SSE Stream       │              │
└──────────────┘                   └──────────────┘

特点:
- 支持远程部署与分布式架构
- 单HTTP端点处理所有请求
- SSE（Server-Sent Events）推送通知
- 支持OAuth 2.1认证
- 可无状态水平扩展
```

**传输层对比：**

| 特性 | Stdio | Streamable HTTP |
|------|-------|-----------------|
| **部署位置** | 本地 | 本地/远程 |
| **网络要求** | 无 | HTTP/HTTPS |
| **延迟** | 极低 | 取决于网络 |
| **认证** | OS级 | OAuth 2.1 |
| **多Client** | 独占进程 | 共享服务 |
| **适用场景** | 桌面应用 | 云服务/SaaS |
| **会话管理** | 进程生命周期 | 可无状态 |

### 1.8 MCP Server开发实践

**Python MCP Server示例：**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("数据分析服务", version="1.0.0")

@mcp.tool()
async def analyze_csv(file_path: str, query: str) -> str:
    """分析CSV文件并回答数据查询问题

    Args:
        file_path: CSV文件路径
        query: 数据分析查询
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    result = df.describe()
    return result.to_string()

@mcp.tool()
async def generate_chart(
    data_source: str,
    chart_type: str,
    x_axis: str,
    y_axis: str
) -> bytes:
    """生成数据可视化图表

    Args:
        data_source: 数据源路径
        chart_type: 图表类型 (bar/line/scatter/pie)
        x_axis: X轴字段名
        y_axis: Y轴字段名
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(data_source)
    fig, ax = plt.subplots()
    getattr(df.plot, chart_type)(x=x_axis, y=y_axis, ax=ax)
    fig.savefig("chart.png")
    with open("chart.png", "rb") as f:
        return f.read()

@mcp.resource("data://{dataset}/schema")
async def get_dataset_schema(dataset: str) -> str:
    """获取数据集的Schema信息"""
    schemas = {
        "users": "id: INT, name: VARCHAR, email: VARCHAR, created_at: TIMESTAMP",
        "orders": "id: INT, user_id: INT, amount: DECIMAL, status: VARCHAR"
    }
    return schemas.get(dataset, "未知数据集")

@mcp.prompt()
def data_analysis_prompt(dataset: str, question: str) -> str:
    """数据分析提示模板"""
    return f"""你是一个数据分析专家。请分析{dataset}数据集，回答以下问题：

问题：{question}

请遵循以下步骤：
1. 理解数据结构
2. 制定分析方案
3. 执行分析
4. 总结发现
"""

if __name__ == "__main__":
    mcp.run()
```

**TypeScript MCP Server示例：**

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "代码分析服务",
  version: "1.0.0",
});

server.tool(
  "analyze_code",
  { code: z.string().describe("待分析的代码"), language: z.string().describe("编程语言") },
  async ({ code, language }) => {
    const issues = await performStaticAnalysis(code, language);
    return {
      content: [{ type: "text", text: JSON.stringify(issues, null, 2) }],
    };
  }
);

server.resource(
  "repo://structure",
  async () => ({
    contents: [{ uri: "repo://structure", mimeType: "application/json", text: JSON.stringify(await getRepoStructure()) }],
  })
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 1.9 MCP生态现状

**生态规模（2025-2026）：**

| 类别 | 数量 | 代表项目 |
|------|------|----------|
| **公开MCP Server** | 10000+ | mcp-servers-hub、awesome-mcp |
| **官方集成** | 20+ | PostgreSQL、GitHub、Google Drive、Slack |
| **Host应用** | 30+ | Claude Desktop、VS Code、Cursor、Windsurf |
| **开发SDK** | 5+ | Python、TypeScript、Java、Go、Rust |

**热门MCP Server分类：**

```
MCP Server生态
├── 数据库        PostgreSQL、MySQL、SQLite、MongoDB、Redis
├── 开发工具      GitHub、GitLab、Jira、Linear、Sentry
├── 文件系统      本地文件、S3、Google Drive、Dropbox
├── 搜索引擎      Brave Search、Tavily、Exa
├── 通信平台      Slack、Discord、Email、Telegram
├── AI/ML        HuggingFace、Pinecone、ChromaDB
├── 云服务        AWS、GCP、Azure、Vercel
└── 企业应用      Salesforce、Notion、Confluence、Zendesk
```

---

## 2. A2A协议（Agent2Agent）

### 2.1 A2A概述

**定义：** A2A是Google于2025年4月发布的开放协议，专注于解决不同框架/平台Agent之间的互操作问题。如果说MCP解决了"Agent如何使用工具"，A2A则解决了"Agent如何与其他Agent协作"。

**核心定位：**
```
MCP: Agent ↔ 工具/数据源  （纵向连接）
A2A: Agent ↔ Agent        （横向连接）
```

**设计原则：**
- **Agent不可知**：不假设Agent内部实现
- **基于现有标准**：HTTP、JSON-RPC、Server-Sent Events
- **默认安全**：企业级认证与授权
- **长时间任务支持**：异步任务与状态推送

### 2.2 A2A协议架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        A2A 协议层                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    A2A协议     ┌─────────────────┐         │
│  │   Client Agent  │<=============>│   Remote Agent   │         │
│  │                 │               │                   │         │
│  │  ┌───────────┐  │               │  ┌───────────┐   │         │
│  │  │ A2A Client│──┼──HTTP/JSON───┼─>│ A2A Server│   │         │
│  │  └───────────┘  │   RPC/SSE    │  └───────────┘   │         │
│  │                 │               │                   │         │
│  │  内部实现任意    │               │  内部实现任意      │         │
│  │  (LangChain/    │               │  (CrewAI/         │         │
│  │   AutoGen/...)  │               │   MetaGPT/...)    │         │
│  └─────────────────┘               └─────────────────┘         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Agent Card（Agent发现与能力描述）             │     │
│  └─────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Task（任务生命周期管理）                      │     │
│  └─────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Message/Part（消息与多模态部件）              │     │
│  └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 Agent Card

Agent Card是A2A的发现机制，描述Agent的身份与能力：

```json
{
  "name": "财务分析Agent",
  "description": "专业的财务数据分析与报告生成Agent",
  "url": "https://finance-agent.example.com/a2a",
  "provider": {
    "organization": "FinanceAI Corp",
    "url": "https://financeai.example.com"
  },
  "version": "2.1.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "authentication": {
    "schemes": ["oauth2", "api_key"]
  },
  "skills": [
    {
      "id": "financial_analysis",
      "name": "财务报表分析",
      "description": "分析企业财务报表，生成洞察报告",
      "tags": ["finance", "analysis", "report"],
      "examples": [
        "分析阿里巴巴2025年Q3财报",
        "对比腾讯和百度的利润率趋势"
      ],
      "inputModes": ["text", "file"],
      "outputModes": ["text", "file"]
    },
    {
      "id": "risk_assessment",
      "name": "风险评估",
      "description": "基于财务数据进行风险评估",
      "tags": ["risk", "assessment"],
      "examples": ["评估某公司的信用风险等级"],
      "inputModes": ["text"],
      "outputModes": ["text"]
    }
  ]
}
```

**Agent Card发现流程：**
```
Client Agent                        Remote Agent
     │                                   │
     │  GET /.well-known/agent.json      │
     │──────────────────────────────────>│
     │                                   │
     │  返回Agent Card                    │
     │<──────────────────────────────────│
     │                                   │
     │  解析skills与capabilities          │
     │  匹配任务需求                      │
     │                                   │
     │  选择合适的Remote Agent            │
     │                                   │
```

### 2.4 Task生命周期

Task是A2A的核心抽象，代表Client委托给Remote Agent的工作单元：

```
                    ┌──────────┐
                    │ submitted │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌─────│ working  │─────┐
              │     └────┬─────┘     │
              │          │           │
         ┌────▼────┐     │    ┌─────▼─────┐
         │  input  │     │    │ completed  │
         │required │     │    └───────────┘
         └────┬────┘     │
              │          │
              └───> 回到working
                         │
                   ┌─────▼──────┐
                   │  failed    │
                   └────────────┘
                         │
                   ┌─────▼──────┐
                   │  rejected  │
                   └────────────┘
                         │
                   ┌─────▼──────┐
                   │  canceled  │
                   └────────────┘
```

**Task状态说明：**

| 状态 | 说明 | 可转换到 |
|------|------|----------|
| **submitted** | 任务已提交，等待Remote Agent接受 | working, rejected |
| **working** | Agent正在处理任务 | completed, failed, input-required, canceled |
| **input-required** | 需要Client提供额外输入 | working |
| **completed** | 任务完成 | - (终态) |
| **failed** | 任务失败 | - (终态) |
| **rejected** | Agent拒绝任务 | - (终态) |
| **canceled** | 任务被取消 | - (终态) |

**Task交互时序：**

```
Client Agent                              Remote Agent
     │                                         │
     │  tasks/send                             │
     │  {message: {parts: [...]}}              │
     │────────────────────────────────────────>│
     │                                         │
     │  Task {status: "working"}               │
     │<────────────────────────────────────────│
     │                                         │
     │  (异步处理中...)                          │
     │                                         │
     │  SSE: Task {status: "input-required"}   │
     │<────────────────────────────────────────│
     │                                         │
     │  tasks/send (提供额外输入)                │
     │────────────────────────────────────────>│
     │                                         │
     │  Task {status: "working"}               │
     │<────────────────────────────────────────│
     │                                         │
     │  SSE: Task {status: "completed",        │
     │        artifacts: [...]}                │
     │<────────────────────────────────────────│
     │                                         │
```

### 2.5 消息与部件（Part）机制

A2A的消息由Part组成，支持多模态内容：

```
Message
├── role: "user" | "agent"
├── parts: Part[]
│   ├── TextPart      {type: "text", text: "..."}
│   ├── FilePart      {type: "file", file: {uri/mimeType/bytes/name}}
│   └── DataPart      {type: "data", data: {...}}
└── metadata: {}

Artifact (Agent产出)
├── parts: Part[]
├── name: string
├── description: string
└── metadata: {}
```

**多模态消息示例：**
```json
{
  "role": "user",
  "parts": [
    {
      "type": "text",
      "text": "请分析这张财务报表的趋势"
    },
    {
      "type": "file",
      "file": {
        "name": "q3_report.xlsx",
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "bytes": "UEsDBBQABgAIAAAAIQ..."
      }
    },
    {
      "type": "data",
      "data": {
        "quarter": "Q3-2025",
        "currency": "CNY",
        "comparison_period": "year-over-year"
      }
    }
  ]
}
```

### 2.6 A2A核心API

| 方法 | 说明 | 请求 | 响应 |
|------|------|------|------|
| `tasks/send` | 发送消息创建/更新Task | Message | Task |
| `tasks/sendSubscribe` | 发送消息并订阅SSE流 | Message | SSE Stream |
| `tasks/get` | 获取Task状态 | taskId | Task |
| `tasks/cancel` | 取消Task | taskId | Task |
| `tasks/resubscribe` | 重新订阅Task的SSE流 | taskId | SSE Stream |

**A2A Client代码示例：**

```python
import httpx
import json

class A2AClient:
    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.client = httpx.Client(timeout=120)

    def get_agent_card(self) -> dict:
        resp = self.client.get(
            f"{self.agent_url}/.well-known/agent.json"
        )
        return resp.json()

    def send_task(self, message: str, session_id: str = None) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "id": session_id or "task-" + str(hash(message))[:8],
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}]
                }
            }
        }
        resp = self.client.post(
            f"{self.agent_url}/a2a",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return resp.json()

    def get_task(self, task_id: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": {"id": task_id}
        }
        resp = self.client.post(
            f"{self.agent_url}/a2a",
            json=payload
        )
        return resp.json()

async def collaborate_example():
    client = A2AClient("https://research-agent.example.com")

    card = client.get_agent_card()
    print(f"连接到: {card['name']}")
    print(f"技能: {[s['name'] for s in card['skills']]}")

    result = client.send_task("分析2025年全球AI芯片市场趋势")
    task = result.get("result", {})
    print(f"任务状态: {task.get('status', {}).get('state')}")

    if task["status"]["state"] == "completed":
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part["type"] == "text":
                    print(part["text"])
```

### 2.7 A2A vs MCP对比分析

```
┌─────────────────────────────────────────────────────────────┐
│                    协议定位对比                               │
│                                                             │
│    Agent A ──── A2A ──── Agent B ──── MCP ──── Tool/Data   │
│                                                             │
│    A2A: Agent间横向协作    MCP: Agent与工具纵向连接           │
└─────────────────────────────────────────────────────────────┘
```

| 维度 | MCP | A2A |
|------|-----|-----|
| **发布方** | Anthropic | Google |
| **解决什么问题** | Agent如何使用工具和数据 | Agent如何与其他Agent协作 |
| **连接模型** | 1 Client : 1 Server | N Client : N Agent |
| **核心抽象** | Tool / Resource / Prompt | Task / Message / Artifact |
| **发现机制** | tools/list（运行时发现） | Agent Card（.well-known） |
| **传输协议** | Stdio / Streamable HTTP | HTTP + JSON-RPC + SSE |
| **任务管理** | 无（即时调用） | 完整Task生命周期 |
| **多模态** | 部分支持（Resource） | 原生支持（Part机制） |
| **安全模型** | 权限边界 + 沙箱 | OAuth 2.0 + 企业级认证 |
| **反向调用** | Sampling协议 | 无（Agent自主决策） |
| **适用场景** | 工具集成、数据接入 | 跨框架Agent互操作 |

**互补架构：MCP + A2A联合使用**

```
┌──────────────────────────────────────────────────────────────┐
│                     编排Agent (Host)                         │
│                                                              │
│  ┌──────────────┐                    ┌──────────────┐        │
│  │ MCP Client   │                    │ A2A Client   │        │
│  └──────┬───────┘                    └──────┬───────┘        │
└─────────┼──────────────────────────────────┼────────────────┘
          │ MCP                              │ A2A
          ▼                                  ▼
   ┌──────────────┐                   ┌──────────────┐
   │ MCP Server   │                   │ Remote Agent │
   │ (数据库/文件) │                   │ (专业Agent)   │
   └──────────────┘                   └──────┬───────┘
                                             │ MCP
                                             ▼
                                      ┌──────────────┐
                                      │ MCP Server   │
                                      │ (该Agent的工具)│
                                      └──────────────┘
```

### 2.8 跨框架Agent互操作

A2A的核心价值在于打破框架壁垒：

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A互操作层                                │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │LangChain │  │ CrewAI   │  │ AutoGen  │  │自定义Agent│   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │          │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐  │
│  │              A2A Protocol (HTTP/JSON-RPC)              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**互操作场景示例：**

```python
class OrchestratorAgent:
    def __init__(self):
        self.a2a_clients = {}
        self.mcp_sessions = {}

    async def discover_agents(self):
        agent_urls = [
            "https://research-agent.internal/a2a",
            "https://coding-agent.internal/a2a",
            "https://review-agent.internal/a2a",
        ]
        for url in agent_urls:
            client = A2AClient(url)
            card = client.get_agent_card()
            self.a2a_clients[card["name"]] = {
                "client": client,
                "skills": {s["id"]: s for s in card["skills"]},
            }

    async def delegate_task(self, task_description: str):
        best_agent = self._match_skill(task_description)
        result = self.a2a_clients[best_agent]["client"].send_task(
            task_description
        )
        return result

    def _match_skill(self, description: str) -> str:
        for name, info in self.a2a_clients.items():
            for skill_id, skill in info["skills"].items():
                for tag in skill["tags"]:
                    if tag in description.lower():
                        return name
        return list(self.a2a_clients.keys())[0]
```

---

## 3. ACP/UCP等其他协议

### 3.1 协议生态全景

```
Agent通信协议生态 (2025-2026)
│
├── 工具接入层（Agent ↔ 工具/数据）
│   └── MCP (Anthropic) ─── 事实标准，生态最大
│
├── Agent互操作层（Agent ↔ Agent）
│   ├── A2A (Google) ──── 跨框架协作标准
│   ├── ACP (IBM/BEA) ─── 企业级Agent通信
│   └── AGP (AGNTCY) ──── 开源Agent组网协议
│
├── 通用通信层（应用 ↔ AI服务）
│   ├── UCP (UCP.ai) ──── 通用通信协议
│   └── OpenAPI/REST ──── 传统API标准
│
└── 底层传输层
    ├── HTTP/2 + JSON-RPC
    ├── gRPC + Protocol Buffers
    ├── WebSocket
    └── Stdio (本地IPC)
```

### 3.2 ACP（Agent Communication Protocol）

**定义：** ACP是由IBM等企业推动的Agent通信协议，侧重于企业级场景下的Agent间安全、可靠通信。

**ACP架构：**
```
┌──────────────────────────────────────────────────────────┐
│                    ACP 协议栈                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │           应用层 (Application Layer)                │  │
│  │  任务委托 | 结果返回 | 状态查询 | 能力发现          │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │           语义层 (Semantic Layer)                   │  │
│  │  意图理解 | 上下文传递 | 消息路由 | 协议转换        │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │           安全层 (Security Layer)                   │  │
│  │  身份认证 | 授权控制 | 审计日志 | 数据加密          │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │           传输层 (Transport Layer)                  │  │
│  │  gRPC | HTTP/2 | WebSocket | 消息队列              │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**ACP核心特性：**

| 特性 | 说明 | 与A2A差异 |
|------|------|-----------|
| **强类型消息** | Protocol Buffers定义消息格式 | A2A使用JSON |
| **企业级安全** | 集成IAM、RBAC、审计 | A2A使用OAuth 2.0 |
| **可靠传输** | 支持消息队列、事务消息 | A2A基于HTTP请求-响应 |
| **协议转换** | 内置不同Agent框架的适配器 | A2A要求Agent实现A2A接口 |
| **服务治理** | 服务发现、负载均衡、熔断 | A2A依赖Agent Card发现 |

### 3.3 UCP（Universal Communication Protocol）

**定义：** UCP旨在提供应用与AI服务之间的通用通信协议，覆盖比Agent更广的AI服务交互场景。

**UCP核心设计：**
```
┌─────────────────────────────────────────────────────────┐
│                     UCP 架构                             │
│                                                         │
│  ┌───────────┐    UCP协议    ┌───────────────────────┐  │
│  │  应用端    │<============>│    AI服务端             │  │
│  │           │               │                       │  │
│  │ - 请求发起 │               │ - 模型推理             │  │
│  │ - 上下文   │               │ - Agent执行            │  │
│  │   管理    │               │ - 工具调用             │  │
│  │ - 流式    │               │ - 结果聚合             │  │
│  │   消费    │               │                       │  │
│  └───────────┘               └───────────────────────┘  │
│                                                         │
│  统一抽象:                                               │
│  ├── Request/Response (同步)                             │
│  ├── Stream (异步流式)                                   │
│  ├── Subscription (事件订阅)                             │
│  └── Negotiation (能力协商)                              │
└─────────────────────────────────────────────────────────┘
```

### 3.4 协议生态对比

| 维度 | MCP | A2A | ACP | UCP |
|------|-----|-----|-----|-----|
| **发起方** | Anthropic | Google | IBM等 | UCP.ai |
| **定位** | 工具/数据接入 | Agent互操作 | 企业Agent通信 | 通用AI通信 |
| **连接模型** | Client-Server | Peer-to-Peer | Service Mesh | Client-Server |
| **消息格式** | JSON | JSON-RPC | Protobuf/JSON | JSON |
| **传输协议** | Stdio/HTTP | HTTP+SSE | gRPC/HTTP/MQ | HTTP/WebSocket |
| **发现机制** | 运行时list | Agent Card | 服务注册中心 | 能力协商 |
| **安全模型** | 权限边界 | OAuth 2.0 | IAM/RBAC/审计 | API Key/OAuth |
| **任务管理** | 即时调用 | 完整生命周期 | 事务性任务 | 请求-响应 |
| **多模态** | 部分 | 原生Part | 强类型 | 支持 |
| **成熟度** | 高（10000+生态） | 中（快速增长） | 中（企业场景） | 早期 |
| **开源** | 是 | 是 | 是 | 是 |

**协议选型决策树：**
```
需要Agent通信协议？
│
├── 连接对象是什么？
│   ├── 工具/数据源 ──────────> MCP
│   │   └── 需要远程部署？ ──> MCP (Streamable HTTP)
│   │   └── 仅本地使用？ ────> MCP (Stdio)
│   │
│   ├── 其他Agent ─────────────> A2A 或 ACP
│   │   ├── 跨框架/开放生态 ──> A2A
│   │   └── 企业内部/强安全 ──> ACP
│   │
│   └── 通用AI服务 ───────────> UCP
│
└── 混合场景 ──────────────────> MCP + A2A 组合
```

---

## 4. Agentic RAG架构

### 4.1 从标准RAG到Agentic RAG

**标准RAG的局限：**
```
标准RAG流程（被动检索）:
用户Query → 向量检索 → 拼接上下文 → LLM生成 → 输出

问题:
1. 单次检索，无法根据中间结果调整查询
2. 检索质量完全依赖初始Query质量
3. 无法处理需要多步推理的复杂问题
4. 缺乏对检索结果的验证与纠错
5. 无法动态选择检索源
```

**Agentic RAG的核心思想：** 将Agent的自主决策能力注入RAG流程

```
Agentic RAG流程（主动检索）:
用户Query → Agent规划 → 查询生成 → 检索 → 评估 →
  ↑                                          │
  │  (不满足)                                  │
  │←── 查询改写 ←── 结果验证 ←────────────────┘
  │
  │  (满足)
  ▼
整合答案 → 输出
```

### 4.2 架构对比

```
标准RAG架构:
┌────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
│  Query  │──>│ Embedding │──>│ 向量检索  │──>│  LLM   │──> Answer
└────────┘   └──────────┘   └──────────┘   └────────┘
                                ↑
                           ┌──────────┐
                           │ 知识库    │
                           └──────────┘

Agentic RAG架构:
┌────────┐   ┌──────────────────────────────────────────────┐   ┌────────┐
│  Query  │──>│              Agent 控制层                     │──>│ Answer │
└────────┘   │                                              │   └────────┘
             │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
             │  │ 查询规划器 │  │ 检索路由器 │  │ 结果评估器 │   │
             │  └─────┬────┘  └─────┬────┘  └─────┬────┘   │
             │        │             │             │         │
             │        ▼             ▼             │         │
             │  ┌──────────┐  ┌──────────┐       │         │
             │  │ 查询改写  │  │ 源选择器  │       │         │
             │  └──────────┘  └──────────┘       │         │
             │        │             │             │         │
             │        └──────┬──────┘             │         │
             │               ▼                    │         │
             │  ┌──────────────────────────┐      │         │
             │  │     检索执行器             │      │         │
             │  │  ┌─────┐ ┌─────┐ ┌─────┐│      │         │
             │  │  │向量DB│ │SQL  │ │Web  ││      │         │
             │  │  │     │ │DB   │ │Search││      │         │
             │  │  └─────┘ └─────┘ └─────┘│      │         │
             │  └──────────────────────────┘      │         │
             │               │                    │         │
             │               └────────────────────┘         │
             │          (评估检索质量，决定是否继续)          │
             └──────────────────────────────────────────────┘
```

**核心差异对比：**

| 维度 | 标准RAG | Agentic RAG |
|------|---------|-------------|
| **检索策略** | 单次被动检索 | 多轮主动检索 |
| **查询生成** | 直接使用用户Query | Agent规划+改写+分解 |
| **检索源** | 固定向量库 | 动态路由多源 |
| **结果验证** | 无 | 评估器判断质量 |
| **错误恢复** | 无 | 查询改写+重试 |
| **推理深度** | 单步 | 多步迭代 |
| **适应性** | 静态Pipeline | 动态决策树 |
| **成本** | 低 | 较高（多轮LLM调用） |

### 4.3 Agentic RAG工作流设计

**完整工作流时序：**

```
用户                   Agent控制层              检索层              LLM
 │                        │                      │                 │
 │  提交复杂问题           │                      │                 │
 │───────────────────────>│                      │                 │
 │                        │                      │                 │
 │                        │  1.查询规划           │                 │
 │                        │─────────────────────>│                 │
 │                        │                      │                 │
 │                        │  2.生成检索查询        │                 │
 │                        │─────────────────────>│                 │
 │                        │                      │                 │
 │                        │  3.执行检索           │                 │
 │                        │─────────────────────>│                 │
 │                        │  返回检索结果          │                 │
 │                        │<─────────────────────│                 │
 │                        │                      │                 │
 │                        │  4.评估结果质量        │                 │
 │                        │─────────────────────────────────────>│
 │                        │  质量评估反馈          │                 │
 │                        │<─────────────────────────────────────│
 │                        │                      │                 │
 │                        │  5.质量不足？改写查询  │                 │
 │                        │─────────────────────>│                 │
 │                        │  新的检索结果          │                 │
 │                        │<─────────────────────│                 │
 │                        │                      │                 │
 │                        │  6.整合生成最终答案    │                 │
 │                        │─────────────────────────────────────>│
 │                        │  最终答案             │                 │
 │                        │<─────────────────────────────────────│
 │  返回答案               │                      │                 │
 │<───────────────────────│                      │                 │
```

### 4.4 核心模块设计

**1. 查询规划器（Query Planner）：**

```python
from pydantic import BaseModel
from typing import List, Optional

class RetrievalStep(BaseModel):
    query: str
    source: str
    purpose: str
    depends_on: Optional[List[int]] = None

class QueryPlan(BaseModel):
    original_question: str
    sub_queries: List[RetrievalStep]
    reasoning: str

QUERY_PLANNER_PROMPT = """你是一个查询规划专家。将用户的复杂问题分解为多个检索步骤。

规则:
1. 每个步骤有明确的检索目标和数据源
2. 标注步骤间的依赖关系
3. 选择最合适的检索源（vector_db / sql_db / web_search）
4. 每个查询应具体、可检索

输出JSON格式的QueryPlan。"""

async def plan_query(question: str, llm) -> QueryPlan:
    response = await llm.ainvoke(
        QUERY_PLANNER_PROMPT + f"\n\n用户问题: {question}"
    )
    return QueryPlan.model_validate_json(response)
```

**2. 检索路由器（Retrieval Router）：**

```python
from enum import Enum
from dataclasses import dataclass

class RetrievalSource(Enum):
    VECTOR_DB = "vector_db"
    SQL_DB = "sql_db"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_GRAPH = "knowledge_graph"

@dataclass
class RetrievalResult:
    content: str
    source: RetrievalSource
    relevance_score: float
    metadata: dict

class RetrievalRouter:
    def __init__(self):
        self.retrievers = {
            RetrievalSource.VECTOR_DB: VectorDBRetriever(),
            RetrievalSource.SQL_DB: SQLRetriever(),
            RetrievalSource.WEB_SEARCH: WebSearchRetriever(),
            RetrievalSource.KNOWLEDGE_GRAPH: KGRetriever(),
        }

    async def retrieve(self, step: RetrievalStep) -> List[RetrievalResult]:
        source = RetrievalSource(step.source)
        retriever = self.retrievers[source]
        results = await retriever.search(step.query)
        return results

    async def multi_source_retrieve(
        self, query: str, sources: List[RetrievalSource]
    ) -> List[RetrievalResult]:
        tasks = [self.retrievers[s].search(query) for s in sources]
        all_results = await asyncio.gather(*tasks)
        return [r for results in all_results for r in results]
```

**3. 结果评估器（Result Evaluator）：**

```python
EVALUATOR_PROMPT = """评估以下检索结果是否足以回答用户问题。

用户问题: {question}
检索结果: {results}

请评估:
1. 相关性 (1-5): 结果与问题的相关程度
2. 完整性 (1-5): 信息是否足够回答问题
3. 可信度 (1-5): 来源的可靠性

如果总分 < 12，建议:
- 改写查询关键词
- 切换检索源
- 补充额外查询

输出JSON: {{"sufficient": bool, "scores": {...}, "suggestion": "..."}}"""

async def evaluate_results(
    question: str, results: List[RetrievalResult], llm
) -> dict:
    formatted = "\n".join(
        f"[{r.source.value}] (相关度:{r.relevance_score}) {r.content[:200]}"
        for r in results
    )
    response = await llm.ainvoke(
        EVALUATOR_PROMPT.format(question=question, results=formatted)
    )
    return json.loads(response)
```

**4. 完整Agentic RAG编排：**

```python
class AgenticRAG:
    def __init__(self, llm, max_iterations: int = 3):
        self.llm = llm
        self.planner = QueryPlanner(llm)
        self.router = RetrievalRouter()
        self.evaluator = ResultEvaluator(llm)
        self.max_iterations = max_iterations

    async def query(self, question: str) -> str:
        plan = await self.planner.plan_query(question, self.llm)
        all_context = []

        for iteration in range(self.max_iterations):
            for step in plan.sub_queries:
                if step.depends_on:
                    dep_context = [
                        all_context[i] for i in step.depends_on
                        if i < len(all_context)
                    ]
                    enriched_query = f"{step.query}\n参考上下文: {dep_context}"
                else:
                    enriched_query = step.query

                results = await self.router.retrieve(step)
                all_context.extend(results)

            evaluation = await self.evaluator.evaluate(
                question, all_context, self.llm
            )

            if evaluation["sufficient"]:
                break

            plan = await self.planner.refine_plan(
                question, all_context, evaluation["suggestion"], self.llm
            )

        return await self._generate_answer(question, all_context)

    async def _generate_answer(
        self, question: str, context: List[RetrievalResult]
    ) -> str:
        formatted_context = "\n".join(
            f"[{r.source.value}] {r.content}" for r in context
        )
        prompt = f"""基于以下检索结果回答用户问题。请引用来源。

检索结果:
{formatted_context}

用户问题: {question}"""
        return await self.llm.ainvoke(prompt)
```

### 4.5 Agentic RAG高级模式

**1. 自适应检索策略：**
```
                    ┌─────────────┐
                    │  Query分析   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 事实型    │ │ 分析型    │ │ 创意型    │
        │ 精确检索  │ │ 多源检索  │ │ 广泛探索  │
        │ 单轮即可  │ │ 迭代检索  │ │ 发散收敛  │
        └──────────┘ └──────────┘ └──────────┘
```

**2. 混合检索路由：**

| 问题类型 | 检索策略 | 检索源 | 迭代次数 |
|----------|----------|--------|----------|
| 事实查询 | 精确匹配 | 向量DB | 1 |
| 对比分析 | 多源聚合 | 向量DB + SQL | 2-3 |
| 趋势预测 | 时序检索 | SQL + Web | 2-3 |
| 开放讨论 | 广泛探索 | Web + KG | 3+ |
| 代码生成 | 代码检索 | 向量DB + GitHub | 1-2 |

---

## 5. 推理模型Agent架构

### 5.1 推理模型带来的范式转变

**传统Agent vs 推理模型Agent：**

```
传统Agent (GPT-4/Claude 3.5):
┌──────────────────────────────────────────────┐
│  System Prompt + 工具定义 + 用户输入           │
│         │                                     │
│         ▼                                     │
│  LLM直接输出: tool_call 或 text               │
│  (隐式推理，推理过程不可控)                     │
└──────────────────────────────────────────────┘

推理模型Agent (DeepSeek R1 / o3):
┌──────────────────────────────────────────────┐
│  System Prompt + 工具定义 + 用户输入           │
│         │                                     │
│         ▼                                     │
│  ┌─────────────────────────────────┐          │
│  │  推理阶段 (Thinking)             │          │
│  │  - 显式思维链                    │          │
│  │  - 自我验证与纠错                │          │
│  │  - 工具调用规划                  │          │
│  │  - 推理预算控制                  │          │
│  └──────────────┬──────────────────┘          │
│                 ▼                              │
│  ┌─────────────────────────────────┐          │
│  │  执行阶段 (Execution)            │          │
│  │  - 基于推理结果执行工具调用       │          │
│  │  - 结构化输出                    │          │
│  └─────────────────────────────────┘          │
└──────────────────────────────────────────────┘
```

**核心差异：**

| 维度 | 传统LLM Agent | 推理模型Agent |
|------|--------------|--------------|
| **推理方式** | 隐式（嵌入参数） | 显式（思维链） |
| **推理深度** | 固定（单次前向传播） | 可变（推理Token数量） |
| **错误检测** | 事后反思 | 推理中自我纠错 |
| **工具调用决策** | 直接输出 | 先推理再调用 |
| **成本控制** | 固定 | 推理预算可调 |
| **可解释性** | 低 | 高（思维链可见） |

### 5.2 思维链与工具调用的结合

**推理模型Agent的决策循环：**

```
┌─────────────────────────────────────────────────────────────┐
│                    推理模型Agent循环                          │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │  感知输入    │────>│  推理思考     │────>│  决策输出    │  │
│  │  (Observation)│    │  (Thinking)   │    │  (Action)    │  │
│  └─────────────┘     └──────┬───────┘     └──────┬──────┘  │
│         ▲                   │                     │         │
│         │            ┌──────▼───────┐             │         │
│         │            │  内部验证     │             │         │
│         │            │  - 逻辑检查   │             │         │
│         │            │  - 工具选择   │             │         │
│         │            │  - 预算评估   │             │         │
│         │            └──────────────┘             │         │
│         │                                         │         │
│         │            ┌──────────────┐             │         │
│         │            │  工具执行     │<────────────┘         │
│         │            │  结果观察     │                       │
│         │            └──────┬───────┘                       │
│         │                   │                               │
│         └───────────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**推理模型工具调用时序：**

```
用户                     推理模型Agent                   工具
 │                            │                          │
 │  复杂问题                   │                          │
 │───────────────────────────>│                          │
 │                            │                          │
 │                            │  ┌────────────────────┐  │
 │                            │  │ Thinking Phase:    │  │
 │                            │  │ 这个问题需要...     │  │
 │                            │  │ 首先查询数据库获取  │  │
 │                            │  │ 基础数据，然后...   │  │
 │                            │  │ 等等，我需要先确认  │  │
 │                            │  │ 数据库是否有权限... │  │
 │                            │  │ 好的，先调用...     │  │
 │                            │  └────────────────────┘  │
 │                            │                          │
 │                            │  tool_call: query_db     │
 │                            │─────────────────────────>│
 │                            │  结果                    │
 │                            │<─────────────────────────│
 │                            │                          │
 │                            │  ┌────────────────────┐  │
 │                            │  │ Thinking Phase:    │  │
 │                            │  │ 数据显示...但我    │  │
 │                            │  │ 需要补充市场数据   │  │
 │                            │  │ 来验证这个趋势...  │  │
 │                            │  │ 搜索最新报告...    │  │
 │                            │  └────────────────────┘  │
 │                            │                          │
 │                            │  tool_call: web_search   │
 │                            │─────────────────────────>│
 │                            │  结果                    │
 │                            │<─────────────────────────│
 │                            │                          │
 │                            │  ┌────────────────────┐  │
 │                            │  │ Thinking Phase:    │  │
 │                            │  │ 现在我有了足够信息  │  │
 │                            │  │ 可以综合回答了...   │  │
 │                            │  │ 让我验证一下结论   │  │
 │                            │  │ 的准确性...        │  │
 │                            │  └────────────────────┘  │
 │                            │                          │
 │  最终答案（含推理过程）      │                          │
 │<───────────────────────────│                          │
```

### 5.3 推理预算与工具调用的平衡

**推理预算概念：**

```
推理Token预算分配:
┌──────────────────────────────────────────────────────────┐
│                    总推理预算 (reasoning_effort)           │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  思维链Token (thinking_tokens)                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│  │  │ 问题理解  │ │ 规划推理  │ │ 验证反思  │           │  │
│  │  └──────────┘ └──────────┘ └──────────┘           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  工具调用Token (tool_tokens)                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│  │  │ 调用请求  │ │ 结果解析  │ │ 上下文注入│           │  │
│  │  └──────────┘ └──────────┘ └──────────┘           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  输出Token (output_tokens)                          │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**预算策略对比：**

| 策略 | reasoning_effort | 思维链长度 | 工具调用次数 | 适用场景 |
|------|-----------------|-----------|-------------|----------|
| **低预算** | low | 短（1-2步） | 少（0-1次） | 简单问答、格式转换 |
| **中预算** | medium | 中等（3-5步） | 适中（2-3次） | 常规分析、代码编写 |
| **高预算** | high | 长（5+步） | 多（3-5次） | 复杂推理、多步规划 |
| **自适应** | auto | 动态调整 | 按需调用 | 通用场景 |

**自适应预算控制代码：**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ReasoningBudget:
    effort: Literal["low", "medium", "high", "auto"]
    max_thinking_tokens: int = 10000
    max_tool_calls: int = 10
    max_total_tokens: int = 32000

    @classmethod
    def from_task_complexity(cls, complexity: str) -> "ReasoningBudget":
        budgets = {
            "simple": cls(
                effort="low",
                max_thinking_tokens=2000,
                max_tool_calls=2,
                max_total_tokens=8000,
            ),
            "moderate": cls(
                effort="medium",
                max_thinking_tokens=6000,
                max_tool_calls=5,
                max_total_tokens=16000,
            ),
            "complex": cls(
                effort="high",
                max_thinking_tokens=15000,
                max_tool_calls=10,
                max_total_tokens=32000,
            ),
        }
        return budgets.get(complexity, budgets["moderate"])


class ReasoningAgent:
    def __init__(self, model: str = "deepseek-r1", budget: ReasoningBudget = None):
        self.model = model
        self.budget = budget or ReasoningBudget(effort="auto")
        self.tool_call_count = 0
        self.thinking_tokens_used = 0

    async def run(self, question: str, tools: list) -> str:
        complexity = await self._assess_complexity(question)
        if self.budget.effort == "auto":
            self.budget = ReasoningBudget.from_task_complexity(complexity)

        messages = [{"role": "user", "content": question}]
        max_rounds = self.budget.max_tool_calls

        for round_idx in range(max_rounds):
            if self.tool_call_count >= self.budget.max_tool_calls:
                break

            response = await self._call_model(messages, tools)

            if response.thinking_content:
                self.thinking_tokens_used += len(
                    response.thinking_content.split()
                )

            if not response.tool_calls:
                return response.content

            for tool_call in response.tool_calls:
                self.tool_call_count += 1
                result = await self._execute_tool(tool_call)
                messages.append(
                    {"role": "tool", "content": result, "tool_call_id": tool_call.id}
                )

        final_response = await self._call_model(messages, tools=[])
        return final_response.content

    async def _assess_complexity(self, question: str) -> str:
        keywords_complex = [
            "分析", "对比", "评估", "规划", "设计",
            "多步", "综合", "推导", "证明"
        ]
        keywords_simple = ["查询", "翻译", "格式化", "总结", "列出"]

        complex_count = sum(1 for k in keywords_complex if k in question)
        simple_count = sum(1 for k in keywords_simple if k in question)

        if complex_count > simple_count + 1:
            return "complex"
        elif simple_count > complex_count:
            return "simple"
        return "moderate"
```

### 5.4 DeepSeek R1 Agent架构

**DeepSeek R1特点：**
- 原生支持长思维链（最长64K推理Token）
- 思维链中自然融入工具调用决策
- 支持"顿悟"（Aha Moment）——推理中自我纠错

**R1 Agent工作流：**
```
┌──────────────────────────────────────────────────────────────┐
│                  DeepSeek R1 Agent                            │
│                                                              │
│  输入: 用户问题 + 工具定义 + 系统提示                          │
│                     │                                        │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Thinking Block (推理阶段)                  │  │
│  │                                                        │  │
│  │  <think&gt;                                            │  │
│  │  让我分析这个问题...                                    │  │
│  │  首先需要获取X数据...                                   │  │
│  │  等等，我应该先确认Y是否正确...                          │  │
│  │  好的，我的计划是:                                      │  │
│  │  1. 调用工具A获取基础数据                                │  │
│  │  2. 基于结果调用工具B进行验证                            │  │
│  │  3. 综合分析给出答案                                    │  │
│  │  </think&gt;                                           │  │
│  └────────────────────────────────────────────────────────┘  │
│                     │                                        │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Action Block (执行阶段)                    │  │
│  │                                                        │  │
│  │  tool_calls: [                                         │  │
│  │    {"name": "tool_A", "arguments": {...}},             │  │
│  │  ]                                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                     │                                        │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Observation (观察结果)                     │  │
│  │                                                        │  │
│  │  tool_results: [{...}]                                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                     │                                        │
│                     ▼                                        │
│            (循环回到Thinking Block)                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.5 o3/o4-mini Agent架构

**OpenAI o系列特点：**
- reasoning_effort参数控制推理深度
- 原生支持function calling与推理结合
- 推理过程不对外暴露（黑盒思维链）

**o3 Agent代码示例：**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

class O3Agent:
    def __init__(
        self,
        reasoning_effort: str = "medium",
        model: str = "o3",
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.conversation_history = []

    async def run(self, user_message: str, tools: list = None) -> str:
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )

        while True:
            kwargs = {
                "model": self.model,
                "messages": self.conversation_history,
                "reasoning_effort": self.reasoning_effort,
            }
            if tools:
                kwargs["tools"] = tools

            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            if message.tool_calls:
                self.conversation_history.append(message)

                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    self.conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                self.conversation_history.append(message)
                return message.content

    async def _execute_tool(self, tool_call) -> dict:
        tool_map = {
            "search": self._search,
            "calculate": self._calculate,
            "query_db": self._query_db,
        }
        func = tool_map.get(tool_call.function.name)
        if func:
            args = json.loads(tool_call.function.arguments)
            return await func(**args)
        return {"error": f"Unknown tool: {tool_call.function.name}"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索网络信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "查询数据库",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL查询语句"}
                },
                "required": ["sql"],
            },
        },
    },
]

agent = O3Agent(reasoning_effort="high")
answer = await agent.run(
    "分析2025年中国AI芯片市场的竞争格局，包括主要厂商市场份额和技术路线对比",
    tools=tools,
)
```

### 5.6 推理模型Agent vs 传统Agent对比

| 维度 | 传统Agent (GPT-4) | 推理Agent (R1/o3) |
|------|-------------------|-------------------|
| **工具调用决策** | 直接从输入映射到tool_call | 先推理分析再决策调用 |
| **错误处理** | 外部ReAct循环重试 | 内部思维链自我纠错 |
| **规划能力** | 依赖外部Prompt引导 | 内生规划能力 |
| **复杂推理** | 需要Chain-of-Thought Prompt | 原生思维链 |
| **成本结构** | 输入+输出Token | 输入+推理Token+输出Token |
| **延迟** | 较低 | 较高（推理阶段耗时） |
| **可解释性** | 低 | 高（思维链可审计） |
| **适用场景** | 快速响应、简单工具调用 | 复杂推理、多步规划 |

### 5.7 推理模型Agent最佳实践

**1. 推理预算优化策略：**

```
策略1: 分级推理
┌────────────────────────────────────────────┐
│  路由器判断任务复杂度                        │
│  ├── 简单 → 低推理预算 (reasoning_effort=low)│
│  ├── 中等 → 中推理预算 (medium)             │
│  └── 复杂 → 高推理预算 (high)               │
└────────────────────────────────────────────┘

策略2: 渐进推理
┌────────────────────────────────────────────┐
│  先用低预算快速尝试                          │
│  ├── 成功 → 返回结果                        │
│  └── 失败 → 提升预算重试                    │
│      ├── 成功 → 返回结果                    │
│      └── 失败 → 继续提升或人工介入           │
└────────────────────────────────────────────┘

策略3: 混合模型
┌────────────────────────────────────────────┐
│  规划阶段: 推理模型 (R1/o3)                 │
│  执行阶段: 快速模型 (GPT-4o-mini/Flash)     │
│  验证阶段: 推理模型 (R1/o3)                 │
└────────────────────────────────────────────┘
```

**2. 思维链与工具调用协调：**

```python
class HybridReasoningAgent:
    def __init__(self):
        self.reasoning_model = "deepseek-r1"
        self.fast_model = "gpt-4o-mini"
        self.max_reasoning_rounds = 3
        self.max_execution_rounds = 10

    async def run(self, question: str, tools: list) -> str:
        plan = await self._plan_with_reasoning(question, tools)

        results = await self._execute_with_fast_model(plan, tools)

        verified = await self._verify_with_reasoning(question, results)

        if not verified["correct"]:
            refined_plan = await self._plan_with_reasoning(
                f"原计划有误: {verified['issue']}\n原始问题: {question}",
                tools,
            )
            results = await self._execute_with_fast_model(refined_plan, tools)

        return results

    async def _plan_with_reasoning(self, question: str, tools: list) -> list:
        response = await self._call_model(
            self.reasoning_model,
            f"制定执行计划（不要直接执行）:\n{question}",
            tools,
        )
        return self._parse_plan(response)

    async def _execute_with_fast_model(self, plan: list, tools: list) -> str:
        messages = [
            {"role": "system", "content": "按计划执行工具调用，不要额外规划"},
            {"role": "user", "content": json.dumps(plan)},
        ]
        response = await self._call_model(self.fast_model, messages, tools)
        return response

    async def _verify_with_reasoning(
        self, question: str, results: str
    ) -> dict:
        response = await self._call_model(
            self.reasoning_model,
            f"验证以下结果是否正确回答了问题:\n问题: {question}\n结果: {results}",
        )
        return {"correct": "正确" in response, "issue": response}
```

---

## 附录：协议与架构速查表

### Agent通信协议速查

| 协议 | 一句话定位 | 核心抽象 | 传输 | 成熟度 |
|------|-----------|---------|------|--------|
| **MCP** | Agent使用工具的标准接口 | Tool/Resource/Prompt | Stdio/HTTP | ★★★★★ |
| **A2A** | Agent间协作的通用语言 | Task/Message/Artifact | HTTP+JSON-RPC | ★★★☆☆ |
| **ACP** | 企业级Agent安全通信 | 强类型消息/事务 | gRPC/MQ | ★★★☆☆ |
| **UCP** | 通用AI服务通信 | Request/Stream/Subscribe | HTTP/WS | ★★☆☆☆ |

### RAG架构速查

| 架构 | 检索策略 | 推理深度 | 适用场景 | 成本 |
|------|---------|---------|---------|------|
| **标准RAG** | 单次被动 | 浅 | 简单问答 | 低 |
| **Agentic RAG** | 多轮主动 | 深 | 复杂分析 | 中 |
| **Graph RAG** | 图结构遍历 | 中 | 关系推理 | 中 |
| **Hybrid RAG** | 多源混合 | 可变 | 通用 | 中高 |

### 推理模型Agent速查

| 模型 | 推理方式 | 工具调用 | 预算控制 | 适用场景 |
|------|---------|---------|---------|---------|
| **DeepSeek R1** | 显式思维链 | 思维链中决策 | Token限制 | 复杂推理+可解释 |
| **o3** | 隐式推理 | 原生支持 | reasoning_effort | 复杂推理+高精度 |
| **o4-mini** | 隐式推理 | 原生支持 | reasoning_effort | 中等推理+低成本 |
| **GPT-4o** | 无专用推理 | 原生支持 | 无 | 快速响应+工具调用 |

---

*最后更新：2026-05-12*

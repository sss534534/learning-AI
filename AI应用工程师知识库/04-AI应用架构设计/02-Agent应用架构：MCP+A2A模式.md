# Agent 应用架构：MCP + A2A 模式

> 2026 年标准参考架构：MCP 连接工具，A2A 连接 Agent，两者共同构成 Agent 应用的通信骨架

## 元数据
- **难度**: ⭐⭐⭐⭐
- **前置知识**: `../03-Agent开发/01-AI-Agent开发实战.md` | `../04-AI应用架构设计/01-AI应用架构设计.md`
- **关联文件**: `../03-Agent开发/04-OpenAI Agents SDK与Claude Code实战.md` | `../../AI架构师知识库/06-MCP与A2A协议设计.md`
- **最后更新**: 2026-06-12
---

## 1. 参考架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         Agent 应用                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    A2A     ┌─────────────┐    A2A     ┌──────┐ │
│  │ 编排 Agent   │◄──────────►│ 专业 Agent   │◄──────────►│  ...  │ │
│  │ (Orchestrator)│           │ (Specialist) │            │      │ │
│  └──────┬───────┘           └──────┬───────┘            └──────┘ │
│         │                          │                             │
│         │ MCP                      │ MCP                         │
│         ▼                          ▼                             │
│  ┌──────────────┐          ┌──────────────┐                     │
│  │  MCP Server   │          │  MCP Server   │                     │
│  │  · 搜索       │          │  · 数据库     │                     │
│  │  · 文件系统   │          │  · 计算引擎   │                     │
│  │  · 外部 API   │          │  · 缓存服务   │                     │
│  └──────────────┘          └──────────────┘                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

双重协议：
  MCP = Agent ↔ 工具（纵向，Agent 控制工具）
  A2A = Agent ↔ Agent（横向，对等协作）
```

## 2. MCP 集成模式

### 2.1 工具封装模式

将现有 API 封装为 MCP Server：

```python
"""将内部 API 封装为 MCP Server"""
from mcp.server import Server
import requests

server = Server("internal-api")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="create_ticket",
        description="创建 IT 支持工单",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "description": {"type": "string"}
            },
            "required": ["title", "description"]
        }
    )]

@server.call_tool()
async def call_tool(name, arguments):
    # 代理到内部 API
    resp = requests.post(
        "https://api.internal/tickets",
        json=arguments,
        headers={"Authorization": "Bearer " + get_service_token()}
    )
    return [TextContent(type="text", text=resp.json())]
```

### 2.2 MCP Gateway 模式

集中管理多个 MCP Server：

```python
class MCPGateway:
    """MCP 网关：统一管理多个后端 MCP Server"""

    def __init__(self):
        self.servers: dict[str, MCPToolClient] = {}

    async def register_server(self, name: str, command: str, args: list[str]):
        """注册后端 MCP Server"""
        client = MCPToolClient(command, args)
        await client.connect()
        self.servers[name] = client

    async def search_tools(self, query: str) -> list[dict]:
        """跨 Server 搜索工具"""
        results = []
        for name, client in self.servers.items():
            tools = await client.list_tools()
            for tool in tools:
                if query.lower() in tool.name.lower():
                    results.append({
                        "server": name,
                        "tool": tool.name,
                        "description": tool.description
                    })
        return results
```

## 3. A2A 集成模式

### 3.1 Agent Card 暴露

```python
"""通过 Agent Card 暴露 Agent 能力（A2A 发现机制）"""
agent_card = {
    "name": "research-agent",
    "description": "技术研究 Agent，搜索和分析技术资料",
    "version": "1.0.0",
    "capabilities": {
        "skills": [
            {
                "id": "search-papers",
                "name": "论文搜索",
                "description": "搜索学术论文并提取关键信息",
                "input": {"topic": "string"},
                "output": {"papers": "array"}
            },
            {
                "id": "tech-analysis",
                "name": "技术分析",
                "description": "分析技术趋势并生成报告",
                "input": {"technology": "string"},
                "output": {"report": "string"}
            }
        ],
        "protocols": ["a2a", "mcp"],
        "auth": {
            "type": "oauth2",
            "scopes": ["research:read"]
        }
    }
}
```

### 3.2 A2A 客户端

```python
"""通过 A2A 协议与其他 Agent 协作"""
import httpx
from typing import Any

class A2AClient:
    """A2A 协议客户端"""

    def __init__(self, agent_card_url: str):
        self.card_url = agent_card_url
        self.agent_card = None

    async def discover(self):
        """发现 Agent 能力（获取 Agent Card）"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.card_url)
            self.agent_card = resp.json()
            return self.agent_card

    async def send_task(self, skill_id: str, input_data: dict) -> dict:
        """向远程 Agent 发送任务"""
        if not self.agent_card:
            await self.discover()

        # 找到匹配 skill
        skill = next(
            (s for s in self.agent_card["capabilities"]["skills"]
             if s["id"] == skill_id),
            None
        )
        if not skill:
            raise ValueError(f"Agent 不支持 skill: {skill_id}")

        # 发送任务（A2A Task 协议）
        task = {
            "id": generate_uuid(),
            "skill": skill_id,
            "input": input_data,
            "callback": "https://my-agent/task-callback"
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.agent_card["url"] + "/a2a/task",
                json=task
            )
            return resp.json()
```

## 4. 架构模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **单 Agent + MCP** | 简单工具调用 | 架构简单、延迟低 | 能力有限 |
| **多 Agent + A2A** | 跨领域协作 | 可扩展、专业化 | 复杂度高 |
| **编排 Agent + 子 Agent** | 复杂业务流程 | 集中控制、可管理 | 单点瓶颈 |
| **MCP Gateway** | 多工具统一管理 | 工具发现、负载均衡 | 额外延迟 |

## 5. 生产部署 Checklist

- [ ] 每个 MCP Server 声明清晰的 tool schema
- [ ] MCP Server 部署为独立服务（Streamable HTTP）
- [ ] Agent Card 暴露在 `/.well-known/agent-card.json`
- [ ] A2A 通信启用 skill-scoped OAuth
- [ ] 所有 Agent 操作写入审计日志
- [ ] 配置模型级联成本策略
- [ ] 实现 Agent → human 升级路径

## 深度分析

MCP（Model Context Protocol）和 A2A（Agent-to-Agent）构成了 2026 年 Agent 架构的通信骨架。MCP 解决了"Agent 如何调用工具"的问题——通过标准化的 tool schema 声明、Streamable HTTP 传输和 skill-scoped OAuth 认证，使任何兼容 MCP 的 Agent 都能发现和调用任何 MCP Server 提供的工具。A2A 则解决了"Agent 如何与其他 Agent 协作"的问题——通过 Agent Card 暴露能力清单、Skill ID 路由任务、callback 接收结果，实现跨框架、跨组织的 Agent 互操作。

架构模式的选择取决于业务复杂度：单 Agent + MCP 适合简单的工具调用场景，延迟低且架构简单；多 Agent + A2A 适合跨领域协作场景，可扩展性好但复杂度高；编排 Agent + 子 Agent 模式在集中控制和灵活性之间取得平衡，是大多数企业级应用的首选。MCP Gateway 模式通过统一的工具注册和发现中心，解决了多 MCP Server 的管理痛点，适合工具数量超过 10 个的复杂系统。

## Checklist

- [ ] 为每个后端 API 封装 MCP Server，声明清晰的 tool schema
- [ ] MCP Server 部署为独立服务，使用 Streamable HTTP 传输
- [ ] 为每个 Agent 暴露 Agent Card（`/.well-known/agent-card.json`）
- [ ] 实现 A2A 任务的 Skill 路由和 callback 回调机制
- [ ] 配置 skill-scoped OAuth 确保跨 Agent 调用的安全性
- [ ] 部署 MCP Gateway 统一管理多个后端 MCP Server
- [ ] 启用工具按需加载以降低 Context 消耗
- [ ] 实现 Agent → human 升级路径（高风险操作人工审批）
- [ ] 配置模型级联成本策略和 AI Credits 计量
- [ ] 所有 Agent 操作写入审计日志，支持操作回放

## 延伸阅读

- `../03-Agent开发/01-AI-Agent开发实战.md` — Agent 框架选型、状态机设计和工具开发的最佳实践
- `../03-Agent开发/04-OpenAI Agents SDK与Claude Code实战.md` — 两大主流 SDK 对 MCP/A2A 协议的兼容实践
- `../04-AI应用架构设计/01-AI应用架构设计.md` — MCP+A2A 架构在完整 AI 系统架构中的定位
- `../../AI架构师知识库/06-MCP与A2A协议设计.md` — MCP 和 A2A 协议的标准化设计规范和高级用法

---

*最后更新：2026-06-12*

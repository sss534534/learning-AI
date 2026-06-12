# OpenAI Agents SDK 与 Microsoft Agent Framework

> 两大科技巨头的 Agent 框架路线对比：轻量 SDK vs 企业平台

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [[01-Agent基础架构]]
- **关联文件**: [[02-LangGraph与CrewAI实战对比]], [[04-开源框架Hermes与ClaudeCode]]
- **最后更新**: 2026-06-12

## 1. OpenAI Agents SDK

### 1.1 背景

OpenAI 在 2026 年将实验性的 Swarm 项目归档，正式推出生产级 **Agents SDK**。这是 OpenAI 从"研究驱动"到"产品驱动"转变的标志性动作。

Swarm 的实验性质（手写 agent 循环、无内置安全）已无法满足生产需求，Agents SDK 从零开始设计为生产级框架。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| Safety Guardrails | 内置内容安全、输出验证、PII 检测 |
| Managed Hosting | OpenAI 托管运行时，无需自建基础设施 |
| 模型集成 | 原生支持 GPT-5.x、o-series 推理模型 |
| Tool Calling | 自动管理 tool 注册、调用、结果处理 |
| MCP 支持 | 兼容 MCP 协议工具生态 |
| Streaming | 原生支持 SSE 结果流式输出 |

### 1.3 代码示例

```python
from agents import Agent, Runner, guardrail

@guardrail
def check_safety(output: str) -> bool:
    forbidden = ["harmful", "illegal", "dangerous"]
    return not any(w in output.lower() for w in forbidden)

agent = Agent(
    name="ResearchAssistant",
    instructions="你是一个研究助手，基于工具返回的数据回答问题",
    tools=[web_search, arxiv_search, calculator],
    guardrails=[check_safety],
    model="o4-mini"  # 推理模型优化
)

result = Runner.run(
    agent,
    "搜索 2026 年 Agent 框架的最新研究论文并总结",
    max_turns=10
)
print(result.final_output)
```

### 1.4 适用场景

- 深度绑定 OpenAI 模型的产品
- 需要托管运行时的团队（无需运维）
- 快速原型 → 生产级部署的无缝过渡

## 2. Microsoft Agent Framework

### 2.1 背景

**AutoGen** 进入维护模式后，其核心能力被合并到 **Semantic Kernel**，形成统一的 **Microsoft Agent Framework**（2026-04 GA）。这是微软"统一 Agent 开发体验"战略的核心产品。

### 2.2 核心特性

| 特性 | 说明 |
|------|------|
| Entra ID 身份 | Agent 像人一样拥有企业身份和 RBAC 权限 |
| Purview 合规 | 数据丢失保护 (DLP) 覆盖 Agent 的所有操作 |
| A2A 原生支持 | 内置 A2A 客户端和服务器 |
| Copilot 生态 | 与 Microsoft 365 Agent 深度集成 |
| 成本治理 | 内置预算控制和用量审计 |
| 多云部署 | Azure、AWS、Google Cloud 均支持 |

### 2.3 架构

```
┌───────────────────────────────────────────────┐
│          Microsoft Agent Framework            │
├───────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │    Agent Runtime │  │  Control Plane   │   │
│  │  · 执行引擎      │  │  · Entra ID      │   │
│  │  · 工具调度      │  │  · Purview DLP   │   │
│  │  · 状态管理      │  │  · 审计日志      │   │
│  │  · MCP/A2A 网关  │  │  · 预算控制      │   │
│  └──────────────────┘  └──────────────────┘   │
└───────────────────────────────────────────────┘
```

### 2.4 代码示例

```python
from ms.agents import Agent, AgentIdentity
from ms.agents.tools import SharePointTool, OutlookTool

# Agent 拥有企业身份
agent = Agent(
    name="SupportAgent",
    identity=AgentIdentity.from_entra_id(
        "support-bot@company.com",
        department="IT Support"
    ),
    instructions="处理 IT 支持工单",
    tools=[
        SharePointTool("kb/docs"),
        OutlookTool("tickets")
    ],
    compliance_enabled=True,
    budget_limit=500  # 每月 $500 token 预算上限
)

# 通过 A2A 与 HR Agent 协作
hr_agent = await agent.discover(
    protocol="a2a",
    agent_card="https://hr.internal/.well-known/agent-card.json"
)

result = await agent.run(
    "新员工入职流程是什么？",
    collaborate_with=[hr_agent]
)
```

### 2.5 适用场景

- 企业级 Microsoft 365 生态
- 需要严格合规和身份管理的场景
- 跨系统 Agent 编排（SharePoint、Outlook、Teams）

## 3. 对比总结

| 维度 | OpenAI Agents SDK | Microsoft Agent Framework |
|------|-------------------|--------------------------|
| 架构风格 | 轻量 SDK | 重量级平台 |
| 部署方式 | 托管（OpenAI） | 自托管 + Azure |
| 身份系统 | API Key / Token | Entra ID（企业身份） |
| 安全合规 | 内置 guardrails | Purview DLP + 审计 |
| 生态绑定 | OpenAI 模型 | Microsoft 365 |
| 协议支持 | MCP | MCP + A2A |
| 定价模型 | 按 token 计费 | $15/用户/月 |
| 开放度 | 封闭（依赖 OpenAI） | 半开放（多云支持） |
| 适用规模 | 初创到中型 | 中大型企业 |

## 4. 选型决策

```
使用 OpenAI Agents SDK 如果：
  ✅ 已有 OpenAI API 集成
  ✅ 需要快速上线
  ✅ 不想管理基础设施
  ❌ 需要企业身份和合规审计

使用 Microsoft Agent Framework 如果：
  ✅ 已在 Microsoft 365 生态内
  ✅ 需要严格合规和身份管理
  ✅ 需要跨系统协作
  ❌ 不想绑定微软云

## 深度分析

OpenAI Agents SDK 和 Microsoft Agent Framework 代表了 Agent 框架的两条截然不同的路线：轻量 SDK vs 重量级平台。OpenAI 从 Swarm 的实验性质走向生产级 SDK，核心卖点是 Safety Guardrails、Managed Hosting 和 GPT-5.x 深度集成；微软则从 AutoGen 走向 Semantic Kernel 统一框架，核心卖点是 Entra ID 企业身份、Purview 合规和 Microsoft 365 生态。

选型的本质是生态绑定决策。使用 OpenAI Agents SDK 意味着信任 OpenAI 的模型能力和托管服务，适合初创到中型团队追求快速上线；使用 Microsoft Agent Framework 意味着接受微软云生态，适合已在 Microsoft 365 内的中大型企业。两者的定价模型（按 Token 计费 vs $15/用户/月）也反映了不同的商业模式。

协议支持方面，两者都支持 MCP，但微软额外支持 A2A（Agent-to-Agent）协议，这可能成为企业级跨系统协作的关键差异化能力。对于需要自主可控的团队，开源框架可能是更好的选择。

## Checklist

- [ ] 团队是否已绑定 OpenAI 或微软云生态？
- [ ] 需要托管运行时还是自建基础设施？
- [ ] 企业身份管理和合规审计是否是刚需？
- [ ] Agent 是否需要跨系统协作？
- [ ] 预算模型更适合按 Token 还是按用户？
- [ ] 是否需要 A2A 协议的跨系统 Agent 通信？
- [ ] 框架的安全护栏是否满足生产要求？
- [ ] 是否评估过框架切换的迁移成本？
- [ ] 团队的运维能力是否支持自托管方案？
- [ ] 是否考虑过混合使用商业和开源框架？

## 延伸阅读

- [[02-LangGraph与CrewAI实战对比]] — 开源商业框架的补充参考
- [[04-开源框架Hermes与ClaudeCode]] — 自托管替代方案
- [[01-Agent基础架构]] — 框架之上的 Agent 架构设计
- [[07-Agent安全防护体系]] — SDK 安全护栏的理论基础
- [[04-Agent架构评审方法论]] — 框架选型的评估维度
```

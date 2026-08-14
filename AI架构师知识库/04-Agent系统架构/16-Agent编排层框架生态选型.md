# Agent 编排层框架生态选型指南

> 2026 年主流 Agent 编排框架全景对比：从 AgentScope、LangGraph、CrewAI 到 Microsoft Agent Framework、Google ADK、Mastra、Dapr Agents 等 12+ 框架的深度选型分析。

## 元数据
- **难度**: ⭐⭐⭐⭐
- **前置知识**: [[01-Agent架构演进]], [[10-AgentOS架构与运行时]], [[11-生产级Agent四层架构工程落地]]
- **关联文件**: [[02-Multi-Agent协作架构]], [[03-Agent协议与通信架构]], [[09-MCP协议2026演进与无状态传输]], [[12-AgentOS技术组件选型指南]], [[13-十万用户级企业架构方案]], [[14-百万用户级超级架构方案]], [[15-亿级用户终极架构方案]]
- **关联目录**: [[01-框架格局总览2026|../09-Agent框架生态/01-框架格局总览2026]], [[02-LangGraph与CrewAI实战对比|../09-Agent框架生态/02-LangGraph与CrewAI实战对比]], [[06-生产级框架选型批判性决策指南|../09-Agent框架生态/06-生产级框架选型批判性决策指南]]
- **最后更新**: 2026-08-14

---

## 目录

- [1. 框架全景格局](#1-框架全景格局)
- [2. 第一梯队框架深度对比](#2-第一梯队框架深度对比)
- [3. AgentScope 定位与能力边界](#3-agentscope-定位与能力边界)
- [4. 协议支持矩阵](#4-协议支持矩阵)
- [5. 分布式与 Actor 模型](#5-分布式与-actor-模型)
- [6. 性能 Benchmark](#6-性能-benchmark)
- [7. 与规模化架构的映射](#7-与规模化架构的映射)
- [8. 选型决策矩阵](#8-选型决策矩阵)
- [9. 2026 新兴框架与趋势](#9-2026-新兴框架与趋势)
- [10. 深度分析](#10-深度分析)
- [11. Checklist](#11-checklist)
- [12. 延伸阅读](#12-延伸阅读)

---

## 1. 框架全景格局

2026 年的 Agent 编排框架已从"百花齐放"进入清晰的分层格局。以下 HTML 全景图展示了各框架的定位与关系。

<div style="font-family: -apple-system, 'Segoe UI', 'PingFang SC', sans-serif; padding: 24px; background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%); border-radius: 14px; border: 1px solid #334155; overflow-x: auto;">

<div style="text-align: center; margin-bottom: 20px;">
  <div style="font-size: 18px; font-weight: 700; color: #f1f5f9;">2026 Agent 编排框架全景格局</div>
  <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">12+ 主流框架 · 7 大编排范式 · A2A + MCP 双协议标准化</div>
</div>

<!-- 范式分层 -->
<div style="display: flex; flex-direction: column; gap: 10px;">

  <!-- 图式编排 -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #6d5dfc, #4c1d95); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(109,93,252,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">图式编排</div>
      <div style="color: #c4b5fd; font-size: 10px; margin-top: 2px;">有向图 + 显式状态</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(109,93,252,0.15); border: 1px solid rgba(109,93,252,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #a78bfa; font-weight: 600; font-size: 12px;">LangGraph</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">LangChain · 15K★</div>
        <div style="color: #64748b; font-size: 10px;">StateGraph + Checkpoint</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(109,93,252,0.15); border: 1px solid rgba(109,93,252,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #a78bfa; font-weight: 600; font-size: 12px;">AgentScope</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">阿里 · 8K★</div>
        <div style="color: #64748b; font-size: 10px;">Actor + ReAct + msghub</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(109,93,252,0.15); border: 1px solid rgba(109,93,252,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #a78bfa; font-weight: 600; font-size: 12px;">MS Agent Framework</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">微软 · 10K★</div>
        <div style="color: #64748b; font-size: 10px;">v1.0 GA · A2A 一等公民</div>
      </div>
    </div>
  </div>

  <!-- 角色化协作 -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #f59e0b, #b45309); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(245,158,11,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">角色化协作</div>
      <div style="color: #fde68a; font-size: 10px; margin-top: 2px;">角色扮演 + 任务分工</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #fbbf24; font-weight: 600; font-size: 12px;">CrewAI</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">28K★ · 最流行</div>
        <div style="color: #64748b; font-size: 10px;">Crew + 顺序/层级</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #fbbf24; font-weight: 600; font-size: 12px;">Camel-AI</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">6K★ · 学术</div>
        <div style="color: #64748b; font-size: 10px;">角色扮演 + SoM</div>
      </div>
    </div>
  </div>

  <!-- 对话式协作 -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #06b6d4, #0e7490); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(6,182,212,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">对话式协作</div>
      <div style="color: #a5f3fc; font-size: 10px; margin-top: 2px;">消息传递 + 动态协商</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #22d3ee; font-weight: 600; font-size: 12px;">AutoGen / AG2</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">微软 → 社区</div>
        <div style="color: #64748b; font-size: 10px;">ConversableAgent</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #22d3ee; font-weight: 600; font-size: 12px;">MS Agent Framework</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">GroupChat 模式</div>
        <div style="color: #64748b; font-size: 10px;">Sequential/Concurrent</div>
      </div>
    </div>
  </div>

  <!-- 极简 Loop -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #10b981, #047857); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(16,185,129,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">极简 Loop</div>
      <div style="color: #a7f3d0; font-size: 10px; margin-top: 2px;">最小抽象 + Handoff</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #34d399; font-weight: 600; font-size: 12px;">OpenAI Agents SDK</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">18K★ · 原 Swarm</div>
        <div style="color: #64748b; font-size: 10px;">Handoff + Orchestrator</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #34d399; font-weight: 600; font-size: 12px;">PydanticAI</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">8K★</div>
        <div style="color: #64748b; font-size: 10px;">类型安全 + 结构化</div>
      </div>
    </div>
  </div>

  <!-- TypeScript 原生 -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #3b82f6, #1d4ed8); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(59,130,246,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">TypeScript 原生</div>
      <div style="color: #bfdbfe; font-size: 10px; margin-top: 2px;">JS/TS 一等公民</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #60a5fa; font-weight: 600; font-size: 12px;">Mastra</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">26.8K★ · YC W25</div>
        <div style="color: #64748b; font-size: 10px;">v1.0 · 双向 MCP</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #60a5fa; font-weight: 600; font-size: 12px;">Bee Agent</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">IBM · 2K★</div>
        <div style="color: #64748b; font-size: 10px;">企业级 TS</div>
      </div>
    </div>
  </div>

  <!-- 云原生运行时 -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #ec4899, #be185d); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(236,72,153,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">云原生运行时</div>
      <div style="color: #fbcfe8; font-size: 10px; margin-top: 2px;">平台绑定 + K8s</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #f472b6; font-weight: 600; font-size: 12px;">Google ADK</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">8K★</div>
        <div style="color: #64748b; font-size: 10px;">Vertex AI 原生</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #f472b6; font-weight: 600; font-size: 12px;">AWS Strands</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">AWS 2026</div>
        <div style="color: #64748b; font-size: 10px;">Bedrock 集成</div>
      </div>
    </div>
  </div>

  <!-- 分布式 Actor -->
  <div style="display: flex; gap: 10px; align-items: stretch;">
    <div style="min-width: 140px; background: linear-gradient(135deg, #ef4444, #b91c1c); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 12px rgba(239,68,68,0.3);">
      <div style="color: #fff; font-weight: 700; font-size: 13px;">分布式 Actor</div>
      <div style="color: #fecaca; font-size: 10px; margin-top: 2px;">跨机器 + 持久化</div>
    </div>
    <div style="flex: 1; display: flex; gap: 8px; flex-wrap: wrap;">
      <div style="flex: 1; min-width: 120px; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #f87171; font-weight: 600; font-size: 12px;">Dapr Agents</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">v1.0 GA · K8s</div>
        <div style="color: #64748b; font-size: 10px;">Sidecar + 状态存储</div>
      </div>
      <div style="flex: 1; min-width: 120px; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); border-radius: 8px; padding: 10px; text-align: center;">
        <div style="color: #f87171; font-weight: 600; font-size: 12px;">Ray + Matrix</div>
        <div style="color: #94a3b8; font-size: 10px; margin-top: 3px;">万级并发</div>
        <div style="color: #64748b; font-size: 10px;">@ray.remote + Object Store</div>
      </div>
    </div>
  </div>

</div>

<!-- 图例 -->
<div style="display: flex; gap: 14px; justify-content: center; margin-top: 16px; flex-wrap: wrap;">
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #6d5dfc; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">图式编排</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #f59e0b; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">角色化协作</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #06b6d4; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">对话式协作</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #10b981; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">极简 Loop</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #3b82f6; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">TypeScript</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #ec4899; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">云原生</span></div>
  <div style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; border-radius: 3px; background: #ef4444; display: inline-block;"></span><span style="font-size: 10px; color: #94a3b8;">分布式 Actor</span></div>
</div>

</div>

---

## 2. 第一梯队框架深度对比

### 2.1 核心特性对比表

| 框架 | 语言 | 协作模式 | 状态管理 | 分布式能力 | 工具调用 | 2026 版本 |
|------|------|---------|---------|-----------|---------|----------|
| **AgentScope** | Python/Java | ReAct + msghub + Supervisor-Worker | 短/长/持久三层记忆 | Actor 模型 + Nacos 注册 | 原生工具 + MCP | v1.0.7 (Java) |
| **LangGraph** | Python/JS | 有向图 + 并行子图 + 人机协作 | StateGraph + Checkpoint | LangGraph Platform (托管) | MCP adapters + 任意工具 | v1.x 稳定 |
| **CrewAI** | Python | 角色化 Crew + 顺序/层级/共识 | 短期记忆 + 外部长期 | CrewAI Enterprise (有限) | 装饰器工具 + MCP | 持续迭代 |
| **MS Agent Framework** | Python/.NET | Sequential/Concurrent/Handoff/GroupChat | Workflow 持久化 + Session | A2A + MCP 跨运行时 | 多 Provider + MCP | **v1.0 GA** (2026-04) |
| **Google ADK** | Python | Agent + Workflow + Session | Session + Memory Service | Vertex AI 原生 | MCP + Skills | v1.x |
| **OpenAI Agents SDK** | Python | Handoff + Orchestrator | Session + Handoff 上下文 | OpenAI 原生 (有限) | 内置 + Function Tools | 持续 |
| **PydanticAI** | Python | 类型安全 Agent + 工具 | Pydantic 结构化状态 | 无原生分布式 | 类型化工具 + MCP | 持续 |
| **Camel-AI** | Python | 角色扮演 + Task Scheduler | 消息历史 + 记忆模块 | Society of Mind | Tool Hub | 持续 |
| **LlamaIndex Workflows** | Python | 事件驱动 + 多 Agent | Context + Memory | LlamaCloud (有限) | 数据连接器为主 | 持续 |
| **Bee Agent** (IBM) | TypeScript | Agent Loop + Workflow | 可观测 + 持久化 | 企业级集成 | MCP + 企业工具 | v1.x |
| **Mastra** | TypeScript | Agent + Workflow + Supervisor | 4 层记忆 | Mastra Cloud + Edge | 双向 MCP + Zod | **v1.0** (2026-01) |
| **Dapr Agents** | Python | Workflow + PubSub Actor | Dapr 持久化状态 | **K8s 原生 + 跨机器** | Dapr bindings | **v1.0 GA** |

### 2.2 生产成熟度对比

| 框架 | GitHub Stars | 企业采用 | 社区活跃 | 生产就绪 |
|------|-------------|---------|---------|---------|
| LangGraph | 15K+ | LinkedIn, Replit, Klarna | ★★★★★ | 高 |
| CrewAI | 28K+ | 众多中小企业 | ★★★★★ | 中高 |
| MS Agent Framework | 10K+ | 微软生态企业 | ★★★★★ | **高（v1.0 LTS）** |
| Google ADK | 8K+ | GCP 客户 | ★★★★ | 高 |
| OpenAI Agents SDK | 18K+ | OpenAI 客户 | ★★★★ | 中 |
| AgentScope | 8K+ | 阿里云 + 国内企业 | ★★★★ | **高（Java v1.0）** |
| PydanticAI | 8K+ | Pydantic 生态 | ★★★★ | 中 |
| Mastra | **26.8K** | Replit, SoftBank | ★★★★★ | 高 |
| Dapr Agents | 26K+ (Dapr) | 企业 K8s | ★★★★ | **高（v1.0）** |
| Bee Agent (IBM) | 2K+ | IBM 客户 | ★★★ | 中 |
| Camel-AI | 6K+ | 学术研究 | ★★★ | 中 |

### 2.3 适用规模对比

| 框架 | 单机 | 小规模 (<10) | 中规模 (10-100) | 大规模 (>100) |
|------|------|------------|---------------|-------------|
| LangGraph | ✅ 优秀 | ✅ 优秀 | ✅ 良好 (Platform) | ⚠️ 需自建 |
| CrewAI | ✅ 优秀 | ✅ 优秀 | ⚠️ 有限 | ❌ 不擅长 |
| MS Agent Framework | ✅ | ✅ | ✅ (A2A) | ⚠️ Azure 扩展 |
| Google ADK | ✅ | ✅ | ✅ (Vertex) | ⚠️ 云内 |
| AgentScope | ✅ | ✅ | ✅ (Cell 内) | ⚠️ 单 Cell 10-30 万 |
| Mastra | ✅ | ✅ | ✅ (Cloud) | ⚠️ Edge 有限 |
| Dapr Agents | ✅ | ✅ | ✅ | ✅ **K8s 原生** |
| Ray + Matrix | ❌ 过重 | ⚠️ | ✅ | ✅ **万级并发** |

---

## 3. AgentScope 定位与能力边界

### 3.1 AgentScope 的独特优势

| 优势 | 说明 |
|------|------|
| **Actor 模型分布式** | `to_dist()` 一行代码转分布式，Nacos 3.0 作为注册中心 |
| **三层记忆** | 短期（会话）+ 长期（向量）+ 持久（Workspace） |
| **安全沙箱** | 容器级隔离，多租户隔离（SESSION/USER/AGENT/GLOBAL） |
| **实时介入** | `interrupt()` 暂停 + 状态保存 + 自定义中断处理 |
| **A2A 原生 + Nacos** | 分布式 Agent 通信与服务发现 |
| **Java 企业级 SDK** | 原生 Java 支持，适配企业 Spring 生态 |
| **代码透明度** | 高代码透明，便于审计与合规 |

### 3.2 与竞品的差异定位

| 维度 | AgentScope | LangGraph | CrewAI | MS Agent Framework | Mastra |
|------|-----------|-----------|--------|-------------------|--------|
| **核心定位** | 生产就绪、阿里云生态 | 图状态机、精确控制 | 角色化快速原型 | 微软统一继承者 | TS 生产框架 |
| **编排范式** | ReAct + msghub + Actor | StateGraph 有向图 | Crew 角色 + 任务 | Sequential/Handoff | Step Workflow |
| **分布式原生** | ✅ Actor + Nacos | ❌ 需 Platform | ❌ | ⚠️ A2A 跨 Agent | ❌ |
| **A2A 支持** | ✅ 原生 + Nacos | ✅ LangSmith A2A | ✅ CrewAI A2A | ✅ **一等公民** | ⚠️ MCP 间接 |
| **MCP 支持** | ✅ 原生 + Higress | ✅ @langchain/mcp | ✅ | ✅ **一等公民** | ✅ **双向 MCP** |
| **Java 生态** | ✅ **原生 Java SDK** | ❌ | ❌ | ✅ **.NET 原生** | ❌ |
| **观测** | OpenTelemetry | LangSmith 深度集成 | 有限 | Azure Monitor | 内置 Traces |
| **优势场景** | Java 企业、国内云、Cell | 复杂状态流、合规 | 快速多角色协作 | 微软栈、多 Provider | TS 全栈 |

### 3.3 AgentScope 的能力边界

> [!important] AgentScope 管不到什么
> AgentScope 解决的是「Agent 怎么思考、怎么协作、怎么安全执行」，不解决：
> - **LLM 推理调度**（vLLM/SGLang 的职责）
> - **向量库集群**（Qdrant/Milvus 的职责）
> - **跨地域容灾**（K8s + GSLB 的职责）
> - **GPU 产能与 FinOps**（基础设施层的职责）
> - **边缘推理与联邦学习**（亿级架构的职责）

**规模上限**：AgentScope ≈ 单 Cell 编排层（约 10-30 万用户规模的 Agent 运行时）

---

## 4. 协议支持矩阵

### 4.1 A2A 协议支持

> [!info] A2A 现状（2026-04 一周年）
> 150+ 组织、22K+ GitHub Stars、5 种 SDK（Python/JS/Java/Go/.NET）、三大云全面 GA、v1.0 稳定版。

| 框架 | A2A 支持 | 集成深度 |
|------|---------|---------|
| AgentScope | ✅ | **原生 + Nacos 3.0 Registry** |
| MS Agent Framework | ✅ | **一等公民（Foundry 内嵌）** |
| LangGraph | ✅ | LangSmith Server A2A |
| CrewAI | ✅ | A2A Agent Delegation |
| Google ADK | ✅ | Vertex AI 原生（发起方） |
| Mastra | ⚠️ | 通过 MCP 间接 |
| OpenAI Agents SDK | ⚠️ | 有限 |
| PydanticAI / Camel-AI / LlamaIndex | ❌ | 未原生支持 |

### 4.2 MCP 协议集成度

> [!info] MCP 现状（2026-06 v1.0 GA）
> Anthropic + 微软 + OpenAI 联合发布，60% 财富 500 强作为强制标准。

| 框架 | MCP 集成 | 特色 |
|------|---------|------|
| Mastra | ✅ **双向** | 既消费 MCP Server，又能暴露为 MCP Server（最完整） |
| MS Agent Framework | ✅ | Claude SDK + MCP 一等公民 |
| AgentScope | ✅ | 原生 + Higress 工具搜索 |
| LangGraph | ✅ | @langchain/mcp-adapters 多服务器聚合 |
| Google ADK | ✅ | MCP + Skills 渐进式披露 |
| Spring AI | ✅ | Boot Starter（Java 企业级） |

### 4.3 A2A vs MCP 关系

```
MCP = 垂直集成：Agent ↔ 工具/数据源（"给 Agent 一双手"）
A2A = 水平协作：Agent ↔ Agent（"给 Agent 一群同事"）
二者互补，非竞争。完整栈同时需要两者。
```

---

## 5. 分布式与 Actor 模型

### 5.1 分布式 Actor 模型支持对比

| 框架/工具 | Actor 模型 | 分布式机制 | 适用场景 |
|----------|-----------|-----------|---------|
| **AgentScope** | ✅ 自有 Actor | Nacos 注册 + RPC | 单 Cell 10-30 万用户 |
| **Dapr Agents** | ✅ Dapr Actor | K8s + Sidecar + 状态存储 | 企业级持久化 Agent |
| **Ray** (Matrix) | ✅ @ray.remote | 集群 + Object Store | 万级并发合成数据 |
| **Akka** | ✅ JVM Actor | Cluster Sharding + CRDT | 银行/电信级可靠性 |
| LangGraph | ⚠️ 图节点非严格 Actor | Platform 托管 | 中规模图编排 |
| CrewAI / OpenAI SDK | ❌ | 无 | 单机/小规模 |
| MS Agent Framework | ⚠️ Workflow Actor 式 | A2A 跨 Agent | 多 Provider 协作 |

### 5.2 关键洞察

> [!tip] 纯 Actor 模型的边界
> Ray/Akka 为 LLM Agent 提供封装、异步消息、位置透明、监督树、背压等能力，但**不自动提供正确性、成本控制或调试便利**——需配合上层框架使用。

---

## 6. 性能 Benchmark

### 6.1 Arena 基准（ACM CAIS'26，固定 Claude Sonnet 4.5）

| 框架 | LoC | 代码复杂度 | Tokens (In/Out) | 步骤效率 | 延迟(s) | 正确率 | 成本($) |
|------|-----|-----------|----------------|---------|---------|--------|---------|
| LangGraph | 136 | 2.4 | 6,517/760 | 0.63 | 15.92 | 0.76 | 0.031 |
| AWS Strands | 141 | 2.4 | 7,469/758 | 0.67 | 16.85 | **0.84** | 0.034 |
| Claude Agent SDK | **最少** | **最低** | — | — | — | 持平 | **最低** |

> **结论**：模型足够强时，"Prompt 驱动的通用 Loop" 在正确率/一致性/成本上与"代码驱动编排"持平甚至更优，且代码量少 2-4 倍。

### 6.2 任务质量基准

| 框架 | 质量(1-10) | 延迟 | Tokens | 一致性(Std) |
|------|-----------|------|--------|------------|
| CrewAI | **9.66** | 246s | 27,684 | 0.30 |
| AutoGen | 9.63 | 572s | 10,793 | 0.45 |
| LangGraph | 9.42 | 506s | 8,823 | 0.32 |
| OpenAI Agents SDK | 9.31 | 448s | 8,676 | 0.36 |

### 6.3 协调吞吐（Ruflo 自报，8 Agent）

| 框架 | Tasks/sec | P95 延迟 | 备注 |
|------|-----------|---------|------|
| Ruflo | 12.4 | 85ms | 层级 Raft 共识 |
| LangGraph | 3.2 | 320ms | 顺序执行 |
| AutoGen | 2.1 | 480ms | 反思循环 |
| CrewAI | 1.8 | 620ms | 角色化 |

> ⚠️ Ruflo 数据来自项目自报，未第三方验证。

### 6.4 综合洞察

- **简单任务**：所有框架差异可忽略
- **复杂任务**：LangGraph 在有状态管道控制占优；CrewAI 在开发者体验占优；AutoGen 在对话式协作占优
- **成本**：智能路由 + 量化可带来 3-10 倍成本优化
- **编排 vs 模型**：随着模型趋同，编排拓扑选择成为新的主导优化变量

---

## 7. 与规模化架构的映射

以下 HTML 图展示了各框架在 10 万 / 100 万 / 1 亿用户架构中的适用层级。

<div style="font-family: -apple-system, 'Segoe UI', 'PingFang SC', sans-serif; padding: 24px; background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%); border-radius: 14px; border: 1px solid #334155; overflow-x: auto;">

<div style="text-align: center; margin-bottom: 20px;">
  <div style="font-size: 18px; font-weight: 700; color: #f1f5f9;">框架与规模化架构的映射</div>
  <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">各框架在不同用户规模架构中的适用层级</div>
</div>

<table style="width: 100%; border-collapse: collapse; font-size: 12px; color: #e2e8f0;">
<thead>
<tr style="border-bottom: 2px solid #334155;">
<th style="padding: 10px; text-align: left; color: #94a3b8; font-size: 11px; text-transform: uppercase;">框架</th>
<th style="padding: 10px; text-align: center; color: #94a3b8; font-size: 11px; text-transform: uppercase;">10 万用户<br><span style="color: #64748b;">[[13]]</span></th>
<th style="padding: 10px; text-align: center; color: #94a3b8; font-size: 11px; text-transform: uppercase;">100 万用户<br><span style="color: #64748b;">[[14]]</span></th>
<th style="padding: 10px; text-align: center; color: #94a3b8; font-size: 11px; text-transform: uppercase;">1 亿用户<br><span style="color: #64748b;">[[15]]</span></th>
<th style="padding: 10px; text-align: left; color: #94a3b8; font-size: 11px; text-transform: uppercase;">适用层级</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">AgentScope</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">完整编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; color: #94a3b8;">单 Cell 编排层（10-30 万）</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">LangGraph</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">完整编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; color: #94a3b8;">Cell 编排层 + Platform 扩展</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">CrewAI</td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">小规模 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内有限</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #450a0a; color: #f87171; padding: 2px 8px; border-radius: 4px; font-size: 11px;">不适用 ✗</span></td>
<td style="padding: 10px; color: #94a3b8;">小规模角色协作（<10 Agent）</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">MS Agent Framework</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">完整编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内（Azure）</span></td>
<td style="padding: 10px; color: #94a3b8;">微软栈 Cell 编排层</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">Google ADK</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">完整编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内编排 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内（GCP）</span></td>
<td style="padding: 10px; color: #94a3b8;">GCP 栈 Cell 编排层</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">Mastra</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">完整编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内（TS）</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Edge 层（TS）</span></td>
<td style="padding: 10px; color: #94a3b8;">TS 全栈 + Edge 编排</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">Dapr Agents</td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">编排层 ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">跨 Cell ✓</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">跨 Cell ✓</span></td>
<td style="padding: 10px; color: #94a3b8;">K8s 原生分布式（跨 Cell）</td>
</tr>
<tr style="border-bottom: 1px solid #1e293b;">
<td style="padding: 10px; font-weight: 600;">Ray + Matrix</td>
<td style="padding: 10px; text-align: center;"><span style="background: #450a0a; color: #f87171; padding: 2px 8px; border-radius: 4px; font-size: 11px;">过重 ✗</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #422006; color: #facc15; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Cell 内</span></td>
<td style="padding: 10px; text-align: center;"><span style="background: #052e16; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Hub 训练 ✓</span></td>
<td style="padding: 10px; color: #94a3b8;">Hub 层联邦学习 + 万级并发</td>
</tr>
</tbody>
</table>

<div style="margin-top: 16px; padding: 12px; background: rgba(109,93,252,0.1); border: 1px solid rgba(109,93,252,0.3); border-radius: 8px;">
<div style="color: #a78bfa; font-weight: 600; font-size: 13px; margin-bottom: 6px;">关键洞察</div>
<div style="color: #cbd5e1; font-size: 12px; line-height: 1.6;">
<strong>10 万用户</strong>：任一第一梯队框架可作为完整编排层<br>
<strong>100 万用户</strong>：框架在 Cell 内适用，跨 Cell 需 Dapr Agents 或自建控制平面<br>
<strong>1 亿用户</strong>：框架仅在 Cell 内编排层有价值，Edge/Hub/控制平面完全超出范围<br>
<strong>唯一跨 Cell 框架</strong>：Dapr Agents（K8s 原生分布式）可跨 Cell 编排，但仍管不到 Edge 层
</div>
</div>

</div>

---

## 8. 选型决策矩阵

### 8.1 按场景选型

| 场景 | 首选 | 备选 | 理由 |
|------|------|------|------|
| Java 企业级、阿里云生态 | **AgentScope-Java** | Spring AI + MCP | 原生 Java + Actor + Nacos |
| 复杂有状态工作流、合规审计 | **LangGraph** | MS Agent Framework | StateGraph + Checkpoint |
| 快速多角色协作原型 | **CrewAI** | Camel-AI | 角色化 + 低学习曲线 |
| 微软栈、多 Provider、.NET | **MS Agent Framework** | Azure AI Foundry | v1.0 GA + A2A 一等公民 |
| GCP 原生、Gemini 集成 | **Google ADK** | Vertex Agent Engine | Vertex 深度集成 |
| TypeScript 全栈、Next.js | **Mastra** | Bee Agent (IBM) | v1.0 + 双向 MCP |
| 极简 OpenAI 原生 | **OpenAI Agents SDK** | PydanticAI | 最小抽象 |
| K8s 原生分布式、跨语言 | **Dapr Agents** | Ray + Matrix | v1.0 + K8s Sidecar |
| 万级并发合成数据 | **Ray + Matrix** | Dapr Agents | @ray.remote + Object Store |
| 跨组织 Agent 协作 | **A2A + 任意框架** | — | 协议层标准化 |
| 个人持久助手 | **SemaClaw / Hermes** | HALO | Harness 工程 + 行为安全 |

### 8.2 按规模选型

| 规模 | 编排层选型 | 分布式扩展 | 关键约束 |
|------|-----------|-----------|---------|
| **10 万** | AgentScope / LangGraph / CrewAI | 单集群 K8s | 8-32 卡 GPU |
| **100 万** | AgentScope / LangGraph（Cell 内） | Dapr Agents 跨 Cell | 200-600 卡 + 多 Cell |
| **1 亿** | AgentScope / LangGraph（Cell 内） | Dapr Agents 跨 Cell + Ray Hub | 2-6 万卡 + Edge 分流 |

---

## 9. 2026 新兴框架与趋势

### 9.1 新兴框架

| 框架 | 来源 | 核心创新 | 状态 |
|------|------|---------|------|
| **SemaClaw** | 美的 AIRC | Harness 工程、DAG 两阶段编排、PermissionBridge 行为安全 | 开源 (2026-03) |
| **HALO** | 开源社区 | 生物仿生、章鱼式分布式神经、硬件感知 | 开源 (2026-03) |
| **AdaptOrch** | 学术 | 任务自适应拓扑选择、性能收敛缩放律 | 论文 (2026-02) |
| **Ruflo** | 开源 | 层级 Raft 共识、WASM 隔离、Ed25519 Agent 身份 | 开源 (2026-05) |
| **AG2** | AutoGen 社区 | ConversableAgent + AgentOS 升级 | 开源 (2026) |

### 9.2 2026 演进方向

1. **Harness 工程化**：从 Prompt/Context 工程转向"围绕模型的完整基础设施"——审计、可控、生产可靠
2. **性能收敛驱动编排优先**：模型趋同后，编排拓扑选择成为主导优化变量（AdaptOrch 实证 12-23% 提升）
3. **协议标准化**：A2A + MCP 成为跨框架协作的"HTTP + USB-C"
4. **持久记忆与自我演化**：跨会话记忆、经验蒸馏、技能自动生成
5. **行为安全内建**：PermissionBridge、运行时授权检查点取代应用级配置
6. **分布式 Actor 复兴**：Ray、Dapr、Akka 被重新引入 Agent 编排

---

## 10. 深度分析

### 10.1 AgentScope 的独特定位

> [!important] 无直接竞品的三位一体
> AgentScope 在 **Java 企业级 + 分布式 Actor + A2A/Nacos 治理** 三位一体上目前无直接竞品。
> - LangGraph 在图编排上更强但无 Java SDK 且分布式弱
> - MS Agent Framework 在 A2A/MCP 上对等但锁定微软云
> - Dapr Agents 在分布式上更强但无 Java 原生 + 无 LLM 编排抽象

### 10.2 "Prompt vs 代码"之争

Arena 基准显示，强模型下显式编排代码的边际收益递减——这对 AgentScope 这类"高代码透明度"框架是挑战，需在**可控性/审计/合规**场景强调差异化价值。

### 10.3 批判性审视

> [!abstract] 用「The Fool」复盘
> - **Expose My Assumptions**：假设编排框架是生产瓶颈——实际上 LLM 推理成本和延迟才是
> - **Argue the Other Side**：为什么不直接用 Claude Agent SDK / OpenAI Agents SDK？因为企业需要多 Provider + 审计 + 合规
> - **Find the Failure Modes**：最可能死在"框架锁定"——选了 LangGraph 后迁移成本极高
> - **Attack This**：AgentScope 真的比 LangGraph 好吗？在图编排上 LangGraph 更成熟，AgentScope 优势在分布式 Actor
> - **Test the Evidence**：Benchmark 数据多来自自报，需在生产环境实测

---

## 11. Checklist

### 框架选型
- [ ] 是否明确编排层需求（图式/角色化/对话式/极简）？
- [ ] 是否评估分布式需求（单机/Cell 内/跨 Cell）？
- [ ] 是否检查 A2A 协议支持（跨 Agent 协作）？
- [ ] 是否检查 MCP 协议集成度（工具接入）？
- [ ] 是否评估语言生态（Python/Java/TypeScript/.NET）？
- [ ] 是否评估生产成熟度（GitHub stars + 企业采用）？

### 规模匹配
- [ ] 10 万用户：是否选择完整编排层框架？
- [ ] 100 万用户：是否规划 Cell 内编排 + 跨 Cell 扩展？
- [ ] 1 亿用户：是否明确框架仅限 Cell 内 + 自建 Edge/Hub？
- [ ] 是否评估 Dapr Agents 用于跨 Cell 分布式？
- [ ] 是否评估 Ray 用于 Hub 层联邦学习？

### 协议合规
- [ ] 是否同时支持 A2A + MCP？
- [ ] 是否评估跨框架协作需求？
- [ ] 是否检查审计与合规需求（决定选代码驱动 vs Prompt 驱动）？

---

## 12. 延伸阅读

### 本目录关联
- [[01-Agent架构演进]] — Agent 架构演进史
- [[02-Multi-Agent协作架构]] — 多 Agent 协作模式深度
- [[03-Agent协议与通信架构]] — A2A/MCP 协议详解
- [[10-AgentOS架构与运行时]] — AgentOS 运行时设计
- [[11-生产级Agent四层架构工程落地]] — Brain/Memory/Action/Governance 四层
- [[12-AgentOS技术组件选型指南]] — 各层组件选型
- [[13-十万用户级企业架构方案]] — 10 万用户方案
- [[14-百万用户级超级架构方案]] — 100 万用户方案
- [[15-亿级用户终极架构方案]] — 1 亿用户方案

### 跨目录关联
- [[01-框架格局总览2026|../09-Agent框架生态/01-框架格局总览2026]] — 2026 框架版图
- [[02-LangGraph与CrewAI实战对比|../09-Agent框架生态/02-LangGraph与CrewAI实战对比]] — LangGraph vs CrewAI
- [[05-Agent脚手架与快速启动工具生态|../09-Agent框架生态/05-Agent脚手架与快速启动工具生态]] — 脚手架工具
- [[06-生产级框架选型批判性决策指南|../09-Agent框架生态/06-生产级框架选型批判性决策指南]] — 批判性选型方法论

### 外部资源
- [AgentScope GitHub](https://github.com/agentscope-ai/agentscope)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Google ADK](https://google.github.io/adk-docs/)
- [Mastra Documentation](https://mastra.ai/docs)
- [Dapr Agents](https://dapr.dev/)
- [A2A Protocol](https://a2a-protocol.org/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Arena Benchmark (ACM CAIS'26)](https://arxiv.org/abs/2502.02595)

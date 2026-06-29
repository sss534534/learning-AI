# Agent 架构演进

> 从 Function Calling 到 MCP+A2A 协议时代的发展脉络（2023-2026）

## 元数据

- **难度**: ⭐⭐
- **前置知识**: [Transformer架构详解](../01-LLM基础理论/01-Transformer架构详解.md)
- **关联文件**: [Agent协议与通信架构](./03-Agent协议与通信架构.md), [AgentOS架构与运行时](./10-AgentOS架构与运行时.md), [框架格局总览2026](../09-Agent框架生态/01-框架格局总览2026.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [核心概念](#1-核心概念)
2. [架构演进五阶段](#2-架构演进五阶段)
3. [核心组件设计](#3-核心组件设计)
4. [Multi-Agent 系统设计](#4-multi-agent-系统设计)
5. [深度分析](#5-深度分析)
6. [Checklist](#6-checklist)
7. [延伸阅读](#7-延伸阅读)

---

## 1. 核心概念

### 1.1 什么是 AI Agent

**定义：** AI Agent 是能够感知环境、做出决策并执行动作的自主系统。与普通 LLM 调用的本质区别在于**循环**：

```
传统LLM调用: 输入 → 模型 → 输出（一次性，无状态）
Agent系统:   感知 → 推理 → 行动 → 观察 → 循环（有状态，自主）
```

**Agent 的四个核心能力：**

| 能力 | 说明 | 实现方式 |
|------|------|----------|
| **规划（Planning）** | 分解复杂任务、制定执行计划 | CoT、ToT、ReAct |
| **记忆（Memory）** | 保存上下文和历史 | 上下文窗口、向量数据库 |
| **工具使用（Tool Use）** | 调用外部功能和 API | MCP 协议、Function Calling |
| **反思（Reflection）** | 自我评估和改进 | 自检 prompt、多轮验证 |

### 1.2 为什么需要 Agent

**大模型的固有限制：**
- 训练数据截止 → 无法获取实时信息
- 无外部操作能力 → 不能执行动作
- 单轮推理 → 难以处理多步复杂任务
- 上下文窗口有限 → 缺乏长期记忆

**Agent 的破局价值：**
- 通过工具调用扩展 LLM 能力边界
- 通过循环实现自主任务执行
- 通过多 Agent 协作处理复杂工作流
- 通过记忆系统实现持续学习

---

## 2. 架构演进五阶段

### 2.1 阶段一：Function Calling（2023）

**代表：** OpenAI Function Calling API

**核心机制：**

```
用户Query → LLM识别需要调用函数 → 输出结构化JSON → 执行函数 → 结果返回LLM → 回答
```

```json
{
  "name": "get_weather",
  "arguments": {
    "location": "北京",
    "date": "2024-01-01"
  }
}
```

**特点：** 单次调用，无状态，结构化输出
**局限：** 无法处理多步任务，无错误恢复，缺乏规划

### 2.2 阶段二：ReAct 模式（2023）

**论文：** ReAct: Synergizing Reasoning and Acting in Language Models

**核心模式：** 推理（Thought）和行动（Action）交替进行：

```
Thought: 我需要查找2024年诺贝尔物理学奖得主
Action: search("2024 Nobel Prize Physics")
Observation: John Hopfield 和 Geoffrey Hinton 因机器学习贡献获奖
Thought: 信息足够，可以回答了
Final Answer: 2024年诺贝尔物理学奖授予...
```

**优势：** 显式推理过程、可解释性强
**局限：** 单线程执行、无长期规划、复杂任务效率低

**在 2026 年的定位：** ReAct 仍是单 Agent 的默认模式，但 IBM Research 证明结构化 Agent 逻辑（如 I3 Agent）在性能上比裸 ReAct+GPT-5.1 好 4x，token 消耗低 30x。详见 [推理模型与Test-Time Compute](../01-LLM基础理论/06-推理模型与Test-Time%20Compute.md)。

### 2.3 阶段三：Plan-and-Execute（2023-2024）

**代表：** LangChain Plan-and-Execute、BabyAGI

```
┌───────────────────────────────────────┐
│  规划阶段                              │
│  任务 → 分解为子任务 → 依赖分析        │
└──────────┬────────────────────────────┘
           ↓
┌───────────────────────────────────────┐
│  执行阶段                              │
│  并行/串行执行 → 结果整合              │
└───────────────────────────────────────┘
```

**规划策略对比：**

| 策略 | 方式 | 适用 | 局限 |
|------|------|------|------|
| CoT（Chain-of-Thought） | 线性推理链 | 数学、逻辑 | 无分支探索 |
| ToT（Tree of Thoughts） | 树状分支搜索 | 创意写作、规划 | 计算开销大 |
| GoT（Graph of Thoughts） | 图结构推理 | 复杂分析 | 实现复杂 |
| Plan+Execute | 先规划再执行 | 可分解任务 | 计划可能不合理 |

### 2.4 阶段四：Multi-Agent 协作（2024-2025）

**核心思想：** 多个专业 Agent 协作

**三种协作模式：**

```
层级式             平等式              工作流式
  主管            设计 ↔ 开发         需求 → 设计 → 开发 → 部署
 / | \              ↕                   
A1 A2 A3          测试
```

| 要素 | 说明 |
|------|------|
| **角色定义** | 每个 Agent 的专业领域和职责边界 |
| **通信协议** | Agent 间交互方式（消息传递、共享内存） |
| **协作机制** | 如何协调工作（轮询、竞标、协商） |
| **冲突解决** | 意见不一致时处理（投票、仲裁） |

### 2.5 阶段五：协议标准时代（2025-2026）

2025-2026 年 Agent 架构的核心转变：从"框架绑定"到"协议驱动"。

| 维度 | 阶段四（2024） | 阶段五（2026） |
|------|---------------|---------------|
| 工具集成 | 自定义 tool 函数 | MCP 协议（标准化） |
| Agent 通信 | 框架内消息 | A2A 协议（互操作） |
| 框架角色 | 基础设施 | 可替换的实现细节 |
| 治理 | 无 | AAIF、Agent Card |

**2026 参考架构：**

```
                    ┌──────────────────────┐
                    │    Orchestrator       │
                    │     Agent             │
                    └──────┬───────────────┘
                           │ A2A
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 搜索 Agent│ │ 分析 Agent│ │ 报告 Agent│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │ MCP        │ MCP        │ MCP
        ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │ WebSearch │ │ DataAPI  │ │ DocGen   │
        │ MCP Svr   │ │ MCP Svr  │ │ MCP Svr  │
        └──────────┘ └──────────┘ └──────────┘
```

详见 [Agent协议与通信架构](./03-Agent协议与通信架构.md) 和 [Agent应用架构：MCP+A2A模式](../../AI应用工程师知识库/04-AI应用架构设计/02-Agent应用架构：MCP+A2A模式.md)。

---

## 3. 核心组件设计

### 3.1 规划模块

**任务分解策略：**

| 分解方式 | 说明 | 示例 |
|----------|------|------|
| 按步骤分解 | 线性步骤序列 | 写文章 → 1.搜索 2.大纲 3.撰写 4.校对 |
| 按子目标分解 | 独立子目标 | 市场分析 → 竞品分析 / 用户调研 / 趋势预测 |
| 按领域分解 | 按专业领域切分 | 软件开发 → 前端 / 后端 / 数据库 / 部署 |

### 3.2 记忆模块

| 类型 | 说明 | 存储 | 容量 |
|------|------|------|------|
| 短期记忆 | 当前对话上下文 | LLM 上下文窗口 | 4K-1M tokens |
| 工作记忆 | 当前任务中间状态 | Agent State | 变量/缓存 |
| 长期记忆 | 跨会话知识 | 向量数据库 | 无上限 |
| 实体记忆 | 关键实体关系 | 知识图谱 | 结构化 |

### 3.3 工具模块（MCP 视角 2026）

2026 年工具接入的标准方式是通过 [MCP 协议](../04-Agent系统架构/09-MCP协议2026演进与无状态传输.md)：

```python
# MCP 工具 vs 传统 tool 函数

# 传统方式（2024）：自定义函数，每个框架不同
@tool
def search(query: str) -> str:
    return call_search_api(query)

# MCP 方式（2026）：标准化接口，跨框架兼容
# Server 端声明
@server.list_tools()
async def list_tools():
    return [Tool(name="search", description="搜索", inputSchema={...})]

# 任何 MCP 兼容的 Agent 框架都可以调用
```

---

## 4. Multi-Agent 系统设计

### 4.1 角色定义模板

```yaml
Agent 角色:
  name: 研究员
  description: 擅长信息搜集和分析
  system_prompt: |
    你是一位专业研究员，擅长：
    1. 从多个来源搜集信息
    2. 分析信息可靠性
    3. 整理结构化报告
  tools:
    - search (MCP Server)
    - document_reader (MCP Server)
    - calculator (MCP Server)
  protocols:
    - MCP（工具连接）
    - A2A（Agent协作）
```

### 4.2 通信协议

详见 [Agent协议与通信架构](./03-Agent协议与通信架构.md) — MCP（Agent↔工具）和 A2A（Agent↔Agent）的双协议架构。

现代 Agent 通信的统一模型（2026）：

```
统一模型 = MCP（执行） + A2A（编排）

MCP → "Agent 通过什么工具完成工作"
A2A → "Agent 之间如何协作完成任务"
```

---

## 5. 深度分析

### 5.1 架构演进驱动因素

```
2023         2024           2025          2026
单一LLM  →  Chain/Agent   →  Multi-Agent →  协议标准化
                     ↓               ↓               ↓
            LangChain        AutoGen         MCP+A2A
            ReAct            CrewAI          AAIF
                             框架竞争         协议融合
```

**关键洞察：** 每次架构升级都是对前一层能力的补充而非替代。
- 2026 年仍在使用 Function Calling（不是过时了，而是作为 Agent 循环中的原子操作）
- ReAct 仍是单 Agent 的默认模式
- Multi-Agent 不是"更好的"单 Agent，而是针对不同问题的不同工具

### 5.2 常见误区

1. **"Multi-Agent 比单 Agent 更好"**
   - 实际上单 Agent 能解决 80%+ 的问题，Multi-Agent 增加了复杂度
   - 原则：从单 Agent 开始，只有在需要跨领域专业能力时才升级

2. **"框架是选型的关键"**
   - 2026 年的框架是可替换的，协议（MCP/A2A）才是基础设施
   - 先选协议，再选框架

3. **"Agent 越自主越好"**
   - 生产环境需要人机协同（Human-in-the-Loop）
   - 关键决策、高成本操作、安全敏感场景需要人工确认

4. **"2026 年不需要框架内通信"**
   - A2A 主要解决跨组织/跨框架协作
   - 同一个框架内的 Agent 可以直接使用框架的通信机制

### 5.3 2026 前沿方向

| 方向 | 进展 | 生产就绪度 |
|------|------|-----------|
| MCP 无状态传输 | 2026-07 RC | 🟡 预览 |
| A2A v1.0 | 150+ 组织支持 | 🟢 生产 |
| Dynamic Workflows | Claude Opus 4.8 | 🟢 生产 |
| Harness Engineering | OpenAI 方法论 | 🟡 采用中 |
| Agent 安全治理 | MS Agent 365 GA | 🟢 生产 |
| 精通级 Agent | 结构化逻辑 > 裸 ReAct | 🟡 研究→工程 |

---

## 6. Checklist

- [ ] 理解 Agent 的四个核心能力（规划/记忆/工具/反思）
- [ ] 掌握五阶段演进：FC → ReAct → Plan → Multi → Protocol
- [ ] 理解 MCP（工具协议）和 A2A（Agent 协议）的分工
- [ ] 知道 2026 年"协议优于框架"的思想
- [ ] 理解 Multi-Agent 只在真正需要跨领域专业化时使用
- [ ] 记住单 Agent 能解决 80%+ 的问题

---

## 7. 延伸阅读

### 必读论文
1. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
2. [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601)
3. [Graph of Thoughts: Solving Elaborate Problems](https://arxiv.org/abs/2308.09687)

### 实践资源
- [AI-Agent开发实战](../../AI应用工程师知识库/03-Agent开发/01-AI-Agent开发实战.md)
- [MCP协议开发实战](../../AI应用工程师知识库/03-Agent开发/03-MCP协议开发实战.md)

---

*最后更新: 2026-06-12*

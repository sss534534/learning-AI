# 知识库前沿更新实施计划

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 AI/LLM/Agent 学习知识库更新至 2026 年 6 月业界前沿水平

**Architecture:** 以 `AI架构师知识库/` 为主战场，4 个模块平行推进：(1) 协议层更新 (2) Agent 框架生态 (3) 前沿研究 (4) 工程与治理。每个模块含新增文件 + 现有文件更新。

**Tech Stack:** Markdown（知识库文档）

**前置操作：**
- [ ] 确认工作目录为 `E:\workspace\学习知识库`

---

### Task 1: MCP 协议 2026 演进专题（新文件）

**文件：** `AI架构师知识库/04-Agent系统架构/09-MCP协议2026演进与无状态传输.md`

- [ ] **Step 1: 创建文件头 + MCP 2026 无状态传输**

内容要点：
```markdown
# MCP 协议 2026 演进与无状态传输

> 2026 年 MCP 从 1.0 走向生产级基础设施：无状态传输、统一授权、Apps/Tasks 扩展

## 1. MCP 2026-07-28 RC：无状态传输革命

### 1.1 背景：粘性路由的问题
- 传统 MCP 传输：每个 tools/call 依赖连接建立时协商的状态（capabilities、auth、protocol version）
- 问题：负载均衡 shuffle、服务器重启、蓝绿部署会导致会话中断
- 生产环境中 sticky session 绑定成为运维噩梦

### 1.2 Stateless Transport 方案
- PR #2750（2026-05-22）：tools/call 请求自包含所有字段
  - protocol version、declared client capabilities、routing keys、auth context
- 任意服务器可服务任意请求 → 无需 sticky session
- 影响：
  - 负载均衡：从 session-affinity 到 round-robin
  - 部署：蓝绿/滚动升级不再中断 agent 任务
  - 缓存：per-connection 缓存须迁移到共享存储（Redis-equivalent）

### 1.3 迁移路径
- 已有 Server：从连接读取状态 → 从请求读取状态
- 兼容性：向后兼容旧传输（legacy SSE 仍支持但 deprecated）
```

- [ ] **Step 2: 补全 MCP Authorization + Apps/Tasks + AAIF**

```markdown
## 2. MCP Authorization 模型

### 2.1 从 OAuth 到 Per-Request 验证
- 2025-03-26 引入 OAuth 2.1 认证
- 2026-07-28：每个 tools/call 携带 bearer token / signed capability / tenant identifier
- 服务器不再在进程内存中持有会话状态
- Token 内省缓存位于共享存储（Redis）

### 2.2 安全边界
- Skill-scoped OAuth：每个工具声明所需 scope
- Agent Card 签名验证
- mTLS / DPoP sender-constrained tokens

## 3. MCP Apps 与 Tasks 扩展

### 3.1 MCP Apps
- 定义：可组合的 MCP Server 集合，作为一个应用单元发布
- 用例：数据分析 App = 数据库 Server + 可视化 Server + 报告 Server

### 3.2 MCP Tasks
- 定义：跨多个 Server 的有状态工作流
- 与 A2A 的边界：MCP Tasks = 工具编排，A2A = Agent 协作

## 4. MCP 在 Agentic AI Foundation 下的治理

### 4.1 AAIF 成立（2025-12）
- Linux Foundation 旗下定向基金
- 创始成员：OpenAI、Anthropic、Google、Microsoft、AWS、Block
- 入驻项目：MCP（Anthropic）、Goose（Block）、AGENTS.md（OpenAI）
- 截至 2026-04：170+ 成员

### 4.2 MCP + A2A 联合规范
- 预期 2026 下半年发布
- 目标：形式化 tool 层与 agent 层的交接协议
```

- [ ] **Step 3: 更新 `03-Agent协议与通信架构.md`** — 在 MCP 版本表追加 2026 行

读 `03-Agent协议与通信架构.md`，找到 MCP 版本演进表（约 line 27-31），修改为：
```markdown
2024-11  MCP 1.0    初始发布，定义核心原语
2025-03  MCP 2025-03-26  新增Streamable HTTP传输、OAuth 2.1认证
2025-12  MCP → AAIF  Anthropic 将 MCP 捐赠给 Linux Foundation Agentic AI Foundation
2026-07  MCP 2026-07-28 RC  无状态传输、Per-Request Authorization、Apps/Tasks
```

同时在 A2A 章节补充：
```
- 2025-08：ACP 与 A2A 合并（IBM + Google）
- 2026-01：A2A v1.0 发布，150+ 组织支持
- 2026-XX：A2A 在 AAIF 下治理，与 MCP 联合规范制定中
```

---

### Task 2: 协议层收尾检查

- [ ] **Step 1: 验证文件创建和更新正确** — 检查文件是否存在，关键内容完整
- [ ] **Step 2: 提交** — 确认后可 git commit

---

### Task 3: Agent 框架生态目录 + 总览

**文件：** 新建 `AI架构师知识库/09-Agent框架生态/01-框架格局总览2026.md`

- [ ] **Step 1: 创建总览文件**

内容要点：
```markdown
# Agent 框架生态总览 2026

> 2026 年 Agent 框架战争尘埃落定，格局清晰化

## 1. 重大格局变化

### 2026 框架版图重绘

| 事件 | 时间 | 影响 |
|------|------|------|
| AutoGen 进入维护模式，功能合并到 Semantic Kernel | 2026-04 | Microsoft Agent Framework GA |
| OpenAI 归档 Swarm，转向 Agents SDK | 2026 | 生产级替代实验性项目 |
| LangGraph 1.0 GA | 2026 | 最成熟的有向图 Agent 框架 |
| CrewAI 1.0 | 2026 | 角色化多 Agent 协作标杆 |
| Hermes Agent 172k ⭐ | 2026 | 开源 Agent 平台领跑者 |
| OpenClaw 375k+ ⭐ | 2026 | 最大开源 Agent 社区 |

### 2026 框架选择决策矩阵

| 需求 | 推荐框架 | 理由 |
|------|----------|------|
| 企业级 RAG/Agent | LangGraph | 成熟度最高，工具链完善 |
| 多 Agent 角色协作 | CrewAI | 角色化设计开箱即用 |
| Deep OpenAI 集成 | OpenAI Agents SDK | 原生 safety guardrails |
| 微软生态 | Microsoft Agent Framework | Entra ID 集成、Purview 合规 |
| 开源自托管 | Hermes Agent | MCP 兼容、skill-based 架构 |
| CLI 开发 Agent | Claude Code | Dynamic Workflows、最强 coding agent |
```

- [ ] **Step 2: 创建剩余 3 个框架文件**（02, 03, 04）

每个文件约 200-300 行，包含：
- 架构图（ASCII）
- 核心概念
- 实战代码示例
- 优缺点对比
- 最佳实践

---

### Task 4: 前沿研究专题

**文件位置：** `AI架构师知识库/01-LLM基础理论/`

- [ ] **Step 1: 创建 `09-T² Scaling Laws与推理时计算前沿.md`**

内容要点：
```markdown
# T² Scaling Laws 与推理时计算前沿

## 1. T² Scaling Laws（Train-to-Test）

### 1.1 核心发现
- 当 Test-Time Compute（通过重复采样）被纳入训练决策时，最优模型更小更过度训练
- 在固定总预算（训练 + 推理）下，标准 Chinchilla 比例不最优
- 两个互补建模方法：NLL 建模 vs pass@k 直接建模，结论一致

### 1.2 对实践者的建议
- 如果知道 Test-Time Scaling 预算，应该训练更小的模型更长的时间
- T² Scaling 提供了联合优化蓝图

## 2. Test-Time Compute 三阶段

| 阶段 | 预算 | 表现 |
|------|------|------|
| 线性区（Linear） | 低预算，简单任务 | 每加倍计算 ≈ 比例收益 |
| 饱和区（Saturation） | 中等预算 | 接近天花板，额外计算浪费 |
| 突破区（Breakthrough） | 极高预算，困难任务 | 偶尔解锁无法解决的问题 |

### 2.1 Reasoning Floor
- 通用模型即使增加 10 倍推理计算也无法追上推理优化权重的基线
- 通过 RL 内化推理协议比外部搜索更有效

## 3. Parallel Thinking 与 VGS
- Parallel Thinking：并行 rollout + 聚合，降低延迟
- Value-Guided Search：值模型引导树搜索
```

- [ ] **Step 2: 创建 `10-新型模型架构：Mamba-Transformer与MoE演进.md`**

内容要点：
```markdown
# 新型模型架构：Mamba-Transformer 与 MoE 演进

## 1. Nemotron 3 Ultra（NVIDIA, 2026-06）
- 550B 参数 MoE，55B 活跃参数
- 优化目标：长时间运行 Agent 工作流的编排

### 1.1 架构创新
| 技术 | 效果 |
|------|------|
| Hybrid Mamba-Transformer | Mamba 层高效长上下文，Transformer 层精确召回 |
| NVFP4 量化（4-bit） | 跨 Hopper/Blackwell/Ampere 统一部署，5x 吞吐 |
| LatentMoE | 更高效的 expert 路由，横跨推理/代码/工具调用 |
| Multi-token Prediction | 减少生成时间，提高多轮吞吐 |

### 1.2 性能
- SWE-bench / Terminal Bench 2.0：更少 token 完成任务
- Agentic 任务成本降低 30%
- 10M 新 SFT 样本 + 1M RL 任务 + 15 个新 RL 环境

## 2. 架构 > 参数量的新时代
- IBM ASTER：Devstral 24B 结构化 Agent 逻辑 → 行覆盖 +20-45%，token 消耗低 15x
- IBM I3 Agent：比 ReAct+GPT-5.1 好 4x，token 消耗低 30x
- Nemotron 3.5 Content Safety：4B 参数匹配 12B 替代品
```

- [ ] **Step 3: 创建 `11-知识Agent：KARL与OAPL离线策略RL.md`**

内容要点：
```markdown
# 知识 Agent：KARL 与 OAPL 离线策略 RL

## 1. KARL（Knowledge Agent via Reinforcement Learning）
- GLM 4.5 Air 基座 + 多任务 RL
- Pareto 最优：同等质量下成本最低，同等成本下质量优于 GPT 5.2 / Claude 4.6
- 泛化到 OOD 任务（KARLBench 基准）

## 2. OAPL：Off-Policy Agentic RL
- 离线策略 RL，对 trainer/inference engine 差异鲁棒
- 无需 clipped importance weighting / data deletion / router replay
- 多任务训练：BrowseComp-Plus + TREC-Biogen 联合损失，两任务同时提升

## 3. 核心方法论
- 合成数据创建（BrowseComp-Plus）
- 多任务 RL（hard-to-verify tasks）
- Test-Time Compute Scaling
```

- [ ] **Step 4: 更新 `06-推理模型与Test-Time Compute.md`** — 在文末追加章节

追加 "2026 前沿进展" 章节，包含：
```markdown
## 9. 2026 前沿进展

### 9.1 Reasoning Floor
- 非推理模型即使 10 倍推理计算也无法追上推理优化权重
- 内部化推理协议（通过 RL） >> 外部搜索方法

### 9.2 IBM ASTER / I3 Agent
- 结构化 Agent 逻辑 > 裸 ReAct+Frontier Model
- 4x 性能提升，30x token 节省

### 9.3 T² Scaling Laws
参见 `09-T² Scaling Laws与推理时计算前沿.md`

### 9.4 2026 推理模型谱系更新
- Claude Opus 4.8（2026-05）：Dynamic Workflows
- Nemotron 3 Ultra（2026-06）：Agent 编排优化
```

- [ ] **Step 5: 更新 `03-大模型原理与选型.md`** — 在模型对比表追加 2026 行

追加行：
```markdown
| Claude Opus 4.8 | 2026-05 | 未公开 | RL+ | Dynamic Workflows，parallel subagent |
| Nemotron 3 Ultra | 2026-06 | 550B MoE | RL+ | Mamba-Transformer Hybrid，Agent 优化 |
| GPT-5.x | 2026 | 未公开 | - | - |
| DeepSeek V4 | 2026 | 未公开 | - | - |
```

---

### Task 5: 工程与治理

- [ ] **Step 1: 更新 `06-LLM FinOps成本管理体系.md`**

在文末追加章节：
```markdown
## 8. 2026 前沿：AI Credits 与 Agent 计量经济

### 8.1 GitHub AI Credits（2026-06-01）
- GitHub Copilot 所有计划转为使用量计费
- 1 credit = $0.01，按 input/output/cached token 计量
- 套餐：Pro 1500 credits、Pro+ 7000、Max 20000
- 代码补全无限量
- 信号：Agentic Coding 是计量资源，按计算定价而非固定席位

### 8.2 Agent 成本归因框架
- 按 agent/任务/用户 细分
- 每任务 token 消耗审计
- 多模型级联成本优化
```

- [ ] **Step 2: 更新 `07-AI安全测试与红队攻击.md`**

在文末追加章节：
```markdown
## 9. 2026 Agent 安全态势

### 9.1 严峻数据
- 88% 组织报告 Agent 安全事件
- 多 Agent 涌现协调风险（"Bonnie and Clyde"）
- AI 辅助的零日漏洞发现已确认

### 9.2 Microsoft Agent 365（2026-05 GA）
- 身份优先：Entra ID 为 Agent 签发身份
- Purview 数据丢失保护
- 实时审计追踪，注册表同步到 AWS Bedrock / Google Cloud
- $15/用户/月

### 9.3 Agent Card 安全
- 签名 Agent Card（发布者身份验证）
- skill-scoped OAuth
- 发现机制：/.well-known/agent-card.json
```

- [ ] **Step 3: 新建目录 + `01-Harness Engineering.md`**

创建 `AI架构师知识库/10-AI工程化前沿/01-Harness Engineering.md`

内容要点：
```markdown
# Harness Engineering：AI 编码 Agent 的工程方法论

> OpenAI 于 2026 年 2 月正式提出，基于 5 个月实验：用 Codex 构建约 100 万行 beta 产品，零手写代码，~3.5 PRs/人/天

## 1. 核心公式
- **Human steer, Agent execute**
- 工程师描述任务 → Agent 开 PR → Agent 自我审查 → 请求额外 Agent 审查 → 迭代 → Human review optional

## 2. Harness 的组成部分
- AGENTS.md：仓库地图
- Golden Rules：不可协商的架构约束（linter 强制执行）
- 结构化测试：验证生成的代码
- Agent-to-Agent review loop：本地 + 云端 Agent 审查
- 约 1500 PRs 中多数审查由 Agent 完成

## 3. 2026 工具支持
- Claude Code：CLAUDE.md + skills + /code-review agent loops
- OpenAI Codex
- GitHub Copilot Agent tasks REST API（2026-06-04）
```

- [ ] **Step 4: 创建 `02-自托管与本地优先Agent生态.md`**

内容要点：
```markdown
# 自托管与本地优先的 Agent 生态

> 2026 年 Agent 生态的另一面：开源、自托管、数据主权

## 1. 开源 Agent 运行时
- MCP 本地注册表
- 开源 skill registries
- 社区驱动的 Agent 框架（Hermes Agent、OpenClaw）

## 2. 为什么自托管
- 数据主权：Agent 操作不离开本地网络
- 成本控制：按 token 计费的大规模部署成本不可控
- 延迟：本地运行消除网络开销
- 可定制：自由选择模型、工具、安全策略

## 3. 架构参考
- 本地 MCP Server 集群
- 本地向量数据库（Chroma / Qdrant）
- 私有模型推理（vLLM + local LLM）
- A2A 在内部网络中的 Agent 协作
```

---

### 执行顺序建议

```
Task 1 (MCP 2026) ──────────────────────┐
Task 3 (框架生态总览) ───── Task 3.2-3.4 ─┤
Task 4 (前沿研究 x3) ─── Task 4.4-4.5 ───┤── 可并行
Task 5 (工程治理 x4) ──────────────────┘
```

- **并行组**：Task 1 / Task 3 Step 1 / Task 4 Step 1-3 / Task 5 Step 1-4 可同时执行
- **顺序依赖**：Task 3 Step 2-4 依赖 Step 1；Task 4 Step 4-5 依赖 Step 1-3

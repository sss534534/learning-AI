# 知识库前沿更新设计方案

## 概述

将个人 AI/LLM/Agent 学习知识库更新至 2026 年业界前沿水平。策略：广度覆盖优先，以新增文件为主 + 少量 In-place 更新。

## 模块设计

### 模块 1：协议层更新

**位置**: `AI架构师知识库/04-Agent系统架构/`

| 操作 | 文件 | 内容 |
|------|------|------|
| **新增** | `09-MCP协议2026演进与无状态传输.md` | MCP 2026-07-28 RC 无状态传输、Authorization、MCP Apps/Tasks、AAIF 治理、MCP+A2A 联合规范 |
| **更新** | `03-Agent协议与通信架构.md` | MCP 版本演进表追加 2026 行、A2A 章节补充 v1.0 + ACP 合并、Agent Card 标准化 |

### 模块 2：Agent 框架生态 2026

**位置**: 新建 `AI架构师知识库/09-Agent框架生态/`

| 文件 | 内容 |
|------|------|
| `01-框架格局总览2026.md` | 2026 框架版图：AutoGen→MS Agent Framework、Swarm→Agents SDK、LangGraph 1.0、CrewAI 1.0 |
| `02-LangGraph与CrewAI实战对比.md` | 选型决策矩阵、适用场景、架构对比 |
| `03-OpenAI Agents SDK与MS Agent Framework.md` | 两大商业框架深度解析 |
| `04-开源框架：Hermes Agent与Claude Code.md` | Hermes Agent (172k ⭐)、OpenClaw (375k+ ⭐)、Claude Code Dynamic Workflows、GitHub Copilot Agent API |

### 模块 3：前沿研究

**位置**: `AI架构师知识库/01-LLM基础理论/`

| 操作 | 文件 | 内容 |
|------|------|------|
| **新增** | `09-T² Scaling Laws与推理时计算前沿.md` | Train-to-Test 联合优化、T² scaling laws、Three regimes (linear/saturation/breakthrough) |
| **新增** | `10-新型模型架构：Mamba-Transformer与MoE演进.md` | Nemotron 3 Ultra (550B MoE)、Hybrid Mamba-Transformer、NVFP4 量化 (5x 吞吐)、LatentMoE、Multi-token Prediction |
| **新增** | `11-知识Agent：KARL与OAPL离线策略RL.md` | KARL Pareto-optimal vs GPT 5.2/Claude 4.6、OAPL off-policy RL、Parallel Thinking + VGS |
| **更新** | `06-推理模型与Test-Time Compute.md` | 追加 Reasoning Floor、Parallel Thinking、Value-Guided Search、IBM ASTER/I3 Agent |
| **更新** | `03-大模型原理与选型.md` | 补充 2026 模型格局：Nemotron 3、Claude Opus 4.8、GPT-5.x、DeepSeek V4 |

### 模块 4：工程与治理

**位置**: `AI架构师知识库/06-LLMOps体系/` + 新建 `AI架构师知识库/10-AI工程化前沿/`

| 操作 | 文件 | 内容 |
|------|------|------|
| **更新** | `06-LLM FinOps成本管理体系.md` | 追加 AI Credits 计量经济、GitHub Copilot 使用量计费 (2026-06)、Agent 成本归因 |
| **更新** | `07-AI安全测试与红队攻击.md` | 追加 88% Agent 安全事件率、Microsoft Agent 365 治理平面、Agent Card 签名/OAuth |
| **新增** | `10-AI工程化前沿/01-Harness Engineering.md` | OpenAI 2026.02 提出、AGENTS.md 基础设施、agent-to-agent 审查、3.5 PRs/人/天 |
| **新增** | `10-AI工程化前沿/02-自托管与本地优先Agent生态.md` | MCP 本地注册表、开源 Agent 运行时、数据主权、AAIF 开源生态 |

### 汇总

- 新文件：11 个（含 2 个新目录）
- 更新文件：5 个
- 涉及知识库：2 个（AI架构师知识库为主）

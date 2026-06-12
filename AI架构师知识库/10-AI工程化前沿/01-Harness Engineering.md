# Harness Engineering：AI 编码 Agent 的工程方法论

> OpenAI 于 2026 年 2 月正式提出。核心模式：Human steer, Agent execute —— 人类指引方向，Agent 执行实现

## 元数据

- **难度**: ⭐⭐
- **前置知识**: [Agent架构演进](../04-Agent系统架构/01-Agent架构演进.md), [推理模型与Test-Time Compute](../01-LLM基础理论/06-推理模型与Test-Time%20Compute.md)
- **关联文件**: [AI原生软件工程平台](../07-架构案例/08-AI原生软件工程平台.md), [Agent框架生态总览](../09-Agent框架生态/01-框架格局总览2026.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [背景](#1-背景)
2. [核心公式](#2-核心公式)
   - [2.1 AGENTS.md：仓库地图](#21-agentsmd仓库地图)
   - [2.2 Golden Rules](#22-golden-rules)
   - [2.3 Agent-to-Agent 审查循环](#23-agent-to-agent-审查循环)
   - [2.4 结构化测试](#24-结构化测试)
3. [2026 工具支持](#3-2026-工具支持)
   - [3.1 Claude Code 实践](#31-claude-code-实践)
4. [什么时候采用 Harness Engineering](#4-什么时候采用-harness-engineering)
5. [核心原则](#5-核心原则)
6. [参考资料](#参考资料)
7. [深度分析](#6-深度分析)
8. [Checklist](#7-checklist)
9. [延伸阅读](#8-延伸阅读)

---

## 1. 背景

OpenAI 在 2025-2026 年进行了一项为期 5 个月的实验：使用 Codex 构建了一个约 **100 万行**的 beta 产品，**零手写代码**，达到 **~3.5 PRs/人/天** 的产出效率。

关键数据：

| 指标 | 数据 |
|------|------|
| 代码行数 | ~100 万行 |
| 手写代码 | 0 行 |
| PR 产出 | ~3.5 PRs/人/天 |
| 总 PR 数 | ~1,500 |
| 审查方式 | 多数由 Agent 审查，Human review 变为 optional |

## 2. 核心公式

```
成功 = AGENTS.md 地图 + Golden Rules + Agent-to-Agent 审查 + 结构化测试
```

### 2.1 AGENTS.md：仓库地图

AGENTS.md 是 Agent 的"第一份文档"，告诉 Agent：
- 这个仓库是做什么的
- 代码结构和命名规则
- 构建/测试/lint 命令
- 代码风格和约定

没有 AGENTS.md，Agent 就像一个没有地图的导航者。

### 2.2 Golden Rules

不可协商的架构约束，通过 linter 强制执行：

```python
# Golden Rule 示例：不允许直接调用数据库
# 违反模式：
result = db.query("SELECT * FROM users")  # ❌

# 正确模式：
result = UserRepository.find_all()  # ✅
```

Golden Rules 是"代码规范"的 Agent 版本 —— 不是风格建议，而是架构边界。

### 2.3 Agent-to-Agent 审查循环

```
工程师描述任务
  → Agent A 实现并开 PR
  → Agent A 自我审查
  → Agent B（本地）审查
  → Agent C（云端）审查
  → 迭代直到所有 Agent Reviewer 满意
  → Human review（optional）

结果：1,500 PRs 中多数审查由 Agent 完成
```

### 2.4 结构化测试

不仅仅是单元测试，还包括：
- 架构约束测试（Golden Rules 的代码化）
- 生成代码的结构完整性测试
- 回归测试套件

## 3. 2026 工具支持

| 工具 | 能力 | 适用 |
|------|------|------|
| Claude Code | CLAUDE.md + skills + /code-review agent loops | 个人/小团队 |
| OpenAI Codex | 原生 Agents SDK 集成 | OpenAI 生态 |
| GitHub Copilot Agent API | Agent tasks REST API（2026-06-04） | GitHub 生态 |
| 自定义 Harness | linter + 结构化测试 + AGENTS.md | 企业级 |

### 3.1 Claude Code 实践

```
AGENTS.md（仓库地图）
  ↓
skills/（可复用技能）
  ├── review-agent.md（审查 Agent 配置）
  └── deploy-agent.md（部署 Agent 配置）
  ↓
/code-review 命令（Agent 间审查循环）
```

## 4. 什么时候采用 Harness Engineering

```
采用如果：
  ✅ Agent 每天在你的仓库中生成代码
  ✅ 你的团队有 2+ 人协作
  ✅ 代码质量和架构一致性是关注点

跳过如果：
  ❌ 你仍在手写大部分代码
  ❌ 单人项目，Agent 只偶尔使用
  ❌ 项目是探索性质的，不需要架构约束
```

## 5. 核心原则

- **Human steer, Agent execute** — 人做决策，Agent 做实现
- **Code is infrastructure, docs are maps** — 代码是基础设施，文档是 Agent 的地图
- **Review loops scale with agents** — 审查循环应该随 Agent 数量扩展
- **Mechanically enforced > written in docs** — 代码强制执行优于文档说明

## 参考资料

- OpenAI: Harness Engineering Blog Post（2026-02）
- GitHub Copilot Agent Tasks REST API Announcement（2026-06-04）
- Claude Code Dynamic Workflows Documentation

## 6. 深度分析

### 6.1 与传统软件工程的对比

Harness Engineering 代表了软件工程范式的根本性转变：

| 维度 | 传统软件工程 | Harness Engineering |
|------|-------------|-------------------|
| 编码主体 | 开发者手写每一行代码 | Agent 生成代码，Human 做决策 |
| 文档角色 | 团队沟通与知识沉淀媒介 | Agent 的"导航地图"（AGENTS.md） |
| 代码审查 | 纯人工 Code Review | Agent-to-Agent 审查为主，Human 兜底 |
| 测试策略 | 开发者手动编写测试用例 | 结构化测试 + 架构约束自动化测试 |
| 交付效率 | 取决于个人能力，~0.5 PRs/人/天 | ~3.5 PRs/人/天，但需要更多 review |
| 错误成本 | 手写 bug 在编码阶段引入 | Agent 误解上下文导致的架构偏差 |

核心差异在于：传统工程关注"如何写出正确的代码"，而 Harness Engineering 关注"如何设计正确的约束让 Agent 生成正确的代码"。

### 6.2 与其他方法论的关系

- **Harness vs 传统 Prompt Engineering**：传统 Prompt Engineering 针对单次对话优化提示词；Harness 面向仓库级长期协作，通过 AGENTS.md 提供持久化的上下文地图，Golden Rules 提供可执行的架构约束。

- **Harness vs MCP（Model Context Protocol）**：MCP 是 Agent 与外部工具/数据源的通信协议，解决"Agent 如何获取信息"；Harness 是工程管理方法论，解决"Agent 如何规模化协作"。两者互补：MCP 提供工具层能力，Harness 提供流程层管理。

- **Harness vs 传统 Agile**：Agile 关注人与人的交互和迭代节奏；Harness 关注人与 Agent 的分工和 Agent 间的审查循环。在 Harness 模式下，Sprint 的"开发任务"可能变成"设计任务"——工程师定义边界条件，Agent 批量实现。

### 6.3 组织影响

团队结构和角色正在经历深刻变化：

**角色演变**：
- 传统开发者 → **AI Director（AI 导演）**：从"写代码"转为"定义任务、审查输出、设计架构约束"
- QA 工程师 → **AI 质量官**：设计结构化测试套件和 Golden Rules，而非手动测试
- Tech Lead → **AI 架构师**：维护 AGENTS.md 地图，制定 Golden Rules 策略

**技能需求变化**：
- 纯编码能力 → 需求下降
- 架构设计、提示词工程、Agent 审查能力 → 需求上升
- 代码审查焦点：从"审查语法和逻辑"转向"审查架构一致性和约束遵守"

**团队规模效应**：
- 2-3 人小团队：Harness 主要提升个人效率
- 10+ 人团队：Harness 成为协作基础设施，AGENTS.md 统一所有 Agent 的行为

### 6.4 质量保障

AI 原生开发的质量保障体系需要重新设计：

**测试层级**：
1. **架构约束测试**（L1）：Golden Rules 的自动化检查，确保生成代码不违反架构边界
2. **结构完整性测试**（L2）：验证生成代码的模块结构、接口签名、依赖方向
3. **功能测试**（L3）：传统单元测试和集成测试，由 Agent 生成并由 Agent 审查
4. **回归测试套件**（L4）：持续积累的自动化回归

**审查流程**：
- 自审查（Agent A）→ 同级审查（Agent B）→ 独立审查（Agent C）→ Human 抽样
- 审查焦点从"代码是否正确"转向"Agent 是否理解上下文"和"架构约束是否被遵守"

**风险控制**：
- Human review 从 mandatory 变为 optional，但 Human 需要具备"快速识别 Agent 偏差"的能力
- 建立"Agent 行为监控"机制：跟踪 Agent 生成代码的缺陷率、修改频率、架构违规次数

### 6.5 局限性

Harness Engineering 并不适用于所有场景：

1. **探索性项目**：需求不明确、需要频繁试错时，AGENTS.md 和 Golden Rules 的维护成本高于收益
2. **高安全性场景**：金融交易系统、医疗设备软件等需要人类深度参与每一行代码
3. **单人项目**：Agent-to-Agent 审查循环的开销大于单人手动审查
4. **低 Agent 使用率**：Agent 只偶尔使用时，完整 Harness 体系的投入产出比不高
5. **快速原型**：原型阶段不需要架构约束，Harness 会拖慢迭代速度
6. **组织成熟度不足**：团队需要具备"定义清晰架构边界"的能力，否则 AGENTS.md 本身会成为混乱之源

### 6.6 2026 演进方向

截至 2026 年中，Harness Engineering 正在经历以下演进：

**并行子 Agent 工作流**：
- Claude Opus 4.8 支持同时调度多个子 Agent 并行编写不同模块
- 需要 AGENTS.md 提供精确的"模块边界描述"以避免冲突
- 审查循环从串行变为并行 + 依赖图解析

**多 Agent 代码生成**：
- 架构 Agent 负责设计模块结构 → 实现 Agent 负责填充代码 → 审查 Agent 负责验证
- 跨 Agent 上下文共享成为关键挑战
- 趋势：从"单 Agent 辅助开发"向"Agent 团队协作开发"演进

**工具生态成熟**：
- GitHub Copilot Agent API 提供 RESTful Agent 任务调度接口
- Claude Code skills 机制允许封装可复用的 Agent 行为模式
- 企业级 Harness 平台正在出现，整合 linting、测试、审查循环

**组织采纳趋势**：
- 领先科技公司已开始设立"AI 工程效能"团队，负责维护 AGENTS.md 和 Golden Rules
- "AI Director"正在成为一个新兴职位，区别于传统 Engineering Manager
- 社区开始讨论 Harness Engineering 的标准化：KPI、最佳实践、成熟度模型

## 7. Checklist

- [ ] **AGENTS.md 设计 (AGENTS.md Design)**：为仓库编写清晰的 AGENTS.md，涵盖项目描述、代码结构、构建/测试/lint 命令、代码风格约定
- [ ] **Golden Rules 制定 (Golden Rules Definition)**：定义不可协商的架构约束，并通过 linter/结构化测试强制执行
- [ ] **工作流设计 (Workflow Design)**：设计 Agent-to-Agent 审查循环流程，确定审查层级和各 Agent 角色
- [ ] **质量保障 (Quality Assurance)**：建立结构化测试套件（架构约束测试 + 功能测试 + 回归测试）
- [ ] **团队转型 (Team Transformation)**：制定团队角色演变计划，培养"AI Director"能力，建立 Agent 行为监控机制

## 8. 延伸阅读

### 内部关联

- [AI原生软件工程平台](../07-架构案例/08-AI原生软件工程平台.md) — Harness Engineering 的企业级落地案例
- [Agent框架生态总览](../09-Agent框架生态/01-框架格局总览2026.md) — Harness 在 Agent 工具生态中的定位
- [Agent架构演进](../04-Agent系统架构/01-Agent架构演进.md) — Agent 从单轮到多轮、从单 Agent 到多 Agent 的演进背景
- [推理模型与Test-Time Compute](../01-LLM基础理论/06-推理模型与Test-Time%20Compute.md) — Agent 推理能力的基础理论支撑

### 外部资源

- OpenAI: Harness Engineering Blog Post（2026-02）— 原始论文和实验数据
- GitHub Copilot Agent Tasks REST API Announcement（2026-06-04）— Agent 任务调度接口标准化
- "Agent-to-Agent Code Review: Patterns and Pitfalls" — 多 Agent 代码审查模式研究
- "From Developer to AI Director: The Changing Role of Software Engineers" — 角色转型分析
- "Testing in the Age of AI-Generated Code" — AI 生成代码的质量保障策略

# 开源框架：Hermes Agent 与 Claude Code

> 开源 Agent 生态的另一极——社区驱动、自托管、可定制

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [[01-Agent基础架构]]
- **关联文件**: [[02-LangGraph与CrewAI实战对比]], [[03-OpenAI Agents SDK与MS Agent Framework]]
- **最后更新**: 2026-06-12

## 1. Hermes Agent（Nous Research）

**GitHub ⭐：172k+** | 开源 Agent 平台领跑者

### 1.1 定位

通用型开源 Agent 平台，致力于成为"Agent 界的 Linux"——一个厂商中立、可自托管、高度可定制的 Agent 运行时。

### 1.2 核心架构

```
┌───────────────────────────────────────────────┐
│                 Hermes Agent                   │
├───────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐  │
│  │           Agent Runtime                  │  │
│  │  · 状态管理                              │  │
│  │  · 工具执行引擎                           │  │
│  │  · 记忆系统                              │  │
│  └────────────────┬────────────────────────┘  │
│                   │                            │
│  ┌────────────────▼────────────────────────┐  │
│  │           Skill Registry                 │  │
│  │  · 技能发现 / 注册 / 版本管理            │  │
│  │  · 依赖解析                              │  │
│  └────────────────┬────────────────────────┘  │
│                   │                            │
│  ┌────────────────▼────────────────────────┐  │
│  │          Transport Layer                 │  │
│  │  · MCP（默认）                           │  │
│  │  · ACP（兼容）                           │  │
│  │  · 自定义传输                            │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

### 1.3 关键特性

| 特性 | 说明 |
|------|------|
| 多模型兼容 | Claude、GPT、Codex 及任意 OpenAI-compatible API |
| Skill-based | 模块化技能注册，类似"Agent 的 App Store" |
| MCP 原生 | 默认传输层，兼容 10000+ MCP Server |
| 可观测性 | 内置 tracing、token 审计、性能监控 |
| 零配置启动 | 5 分钟部署一个可用 Agent |
| 热加载 | 技能和工具无需重启即可更新 |

### 1.4 快速示例

```python
from hermes import HermesAgent, Skill

agent = HermesAgent(model="claude-opus-4")

@agent.skill
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    return call_search_api(query)

@agent.skill
def calculate(expr: str) -> float:
    """执行数学计算"""
    return eval(expr)

result = agent.run("搜索 2026 年 AI 市场规模并计算增长率")
```

## 2. Claude Code（Anthropic）

CLI-first 开发 Agent，2026 年发布重大升级

### 2.1 Dynamic Workflows（2026-05）

Claude Opus 4.8 的核心 Agent 能力增强：

```
传统模式：
  用户输入 → Claude 思考 → 调用工具 → 返回结果

Dynamic Workflows：
  用户输入 → Orchestrator 拆解任务
    ├── Subagent 1：搜索 API 文档（并行）
    ├── Subagent 2：读取代码库（并行）
    ├── Subagent 3：分析依赖（并行）
    └── Orchestrator 汇总结果 → 生成方案
```

### 2.2 Tool Search + Deferred Loading

```
Before: 所有工具定义一次性加载到 context
  → Context 被工具定义占满
  → Agent 只有少量 token 可用于实际推理

After: Tool Search + Deferred Loading
  → 仅加载匹配当前任务的工具
  → Context 使用减少 85%+
  → Agent 有更多 token 用于推理
```

### 2.3 内置 Agent 循环

```bash
# 使用 /code-review 进行 Agent 间代码审查
claude /code-review --pr 42 --reviewers 3

# 多 Agent 并行执行
claude /task:refactor-auth --agents 3
```

### 2.4 适用场景

- 软件工程（代码生成、重构、审查）
- CLI 自动化脚本
- CI/CD 管道集成

## 3. OpenClaw

**GitHub ⭐：375k+** | 最大开源 Agent 社区

### 3.1 差异化优势

| 能力 | 说明 |
|------|------|
| 插件市场 | 社区贡献的即插即用插件，3000+ 可用 |
| 配置可移植 | Agent 配置文件跨部署迁移 |
| 企业部署工具 | Helm Chart、Terraform、Docker Compose |
| 可视化面板 | 拖拽式 Agent 工作流设计器 |

## 4. 开源框架选型建议

| 框架 | 优势 | 最佳场景 |
|------|------|----------|
| Hermes Agent | 模型兼容性最广，MCP 原生 | 多模型混合架构、自托管 |
| Claude Code | 最强代码理解和生成 | 软件开发 Agent、CLI 工具 |
| OpenClaw | 社区最大、插件最丰富 | 需要现成插件的场景 |

## 5. 开源 vs 商业框架决策

```
选择开源如果：
  ✅ 需要数据主权和控制权
  ✅ 预算敏感（按 token 计费成本高）
  ✅ 需要深度定制
  ✅ 不想绑定特定云厂商

选择商业框架如果：
  ✅ 需要托管服务和 SLA
  ✅ 团队没有运维能力
  ✅ 需要企业级合规和身份管理
  ✅ 已经在目标生态（OpenAI/Microsoft）内

## 深度分析

开源 Agent 框架（Hermes、Claude Code、OpenClaw）构成了 Agent 生态的第三极——社区驱动、自托管、可定制。Hermes 以"Agent 界的 Linux"为愿景，强调模型兼容性（Claude、GPT、Codex 任意切换）和 MCP 原生支持；Claude Code 则聚焦软件开发场景，核心优势在于 Dynamic Workflows 和 Tool Search 的创新；OpenClaw 以 375k+ Star 的社区规模和 3000+ 插件生态吸引用户。

开箱即用 vs 深度定制的权衡是选型的核心。Claude Code 的 CLI-first 设计对软件开发者极其友好，内置 Agent 循环和 /code-review 子命令让代码审查等场景开箱即用；Hermes 的 Skill Registry 设计则提供了更高的可扩展性，适合需要深度定制 Agent 行为的场景。OpenClaw 的配置可移植性和企业部署工具（Helm/Terraform）让它在企业用户中具有独特优势。

开源 vs 商业框架的决策应该基于数据主权、预算和运维能力三个维度。需要数据主权和深度定制的场景优先选择开源；需要托管 SLA 和缺乏运维能力的团队选择商业框架。许多成熟团队的策略是：核心能力自托管开源框架 + 非核心能力使用商业框架的托管服务。

## Checklist

- [ ] 是否需要数据主权和完全的自托管能力？
- [ ] 团队是否有足够运维能力管理开源框架？
- [ ] 是否需要 Hermes 的多模型兼容性？
- [ ] 软件开发场景是否优先考虑 Claude Code？
- [ ] 是否需要 OpenClaw 的 3000+ 插件生态？
- [ ] 是否需要 Hermes 的 Skill Registry 热加载？
- [ ] 预算是否敏感不适合商业框架按 Token 计费？
- [ ] 是否评估过框架的社区活跃度和长期维护？
- [ ] 企业部署是否需要 Helm/Terraform 支持？
- [ ] 是否考虑过开源+商业混合使用策略？

## 延伸阅读

- [[02-LangGraph与CrewAI实战对比]] — 图工作流和角色化协作的开源方案
- [[03-OpenAI Agents SDK与MS Agent Framework]] — 商业框架的对比参考
- [[01-Agent基础架构]] — 框架之上的架构设计原则
- [[02-Multi-Agent协作架构]] — 开源框架的协作模式实现
- [[04-Agent架构评审方法论]] — 开源框架的评估维度
```

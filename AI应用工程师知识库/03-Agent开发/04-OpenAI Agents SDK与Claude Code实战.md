# OpenAI Agents SDK 与 Claude Code 实战

> 2026 年两大主流 Agent SDK 的实战指南

## 元数据
- **难度**: ⭐⭐
- **前置知识**: `../03-Agent开发/01-AI-Agent开发实战.md`
- **关联文件**: `../03-Agent开发/02-Agent实战案例集.md` | `../04-AI应用架构设计/02-Agent应用架构：MCP+A2A模式.md`
- **最后更新**: 2026-06-12
---

## 1. OpenAI Agents SDK

### 1.1 快速开始

```bash
pip install openai-agents
```

```python
"""
OpenAI Agents SDK 基础用法
"""
from agents import Agent, Runner, function_tool
from typing import List

# 定义工具
@function_tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库"""
    # 实际场景调用向量数据库
    return f"关于 '{query}' 的检索结果..."

@function_tool
def calculate_cost(tokens: int, model: str) -> float:
    """计算 API 调用成本"""
    rates = {"gpt-4o": 0.01, "o4-mini": 0.002}
    return tokens * rates.get(model, 0.01) / 1000

# 创建 Agent
agent = Agent(
    name="Assistant",
    instructions="你是一个帮助研发团队的技术助手",
    tools=[search_knowledge_base, calculate_cost],
    model="o4-mini"  # 推理模型优化
)

# 运行 Agent（单轮）
result = Runner.run_sync(
    agent,
    "查询 RAG 系统的最佳实践文档，并估算 10 万 token 的成本"
)
print(result.final_output)
```

### 1.2 Safety Guardrails

```python
from agents import Agent, Runner, GuardrailFunction, GuardrailResult

# 输入护栏：检查用户输入
def input_guardrail(content: str) -> GuardrailResult:
    forbidden = ["删除数据库", "DROP TABLE", "rm -rf"]
    for f in forbidden:
        if f in content:
            return GuardrailResult(
                passed=False,
                message=f"输入包含禁止操作: {f}"
            )
    return GuardrailResult(passed=True)

# 输出护栏：检查模型输出
def output_guardrail(content: str) -> GuardrailResult:
    if content.count("```") % 2 != 0:
        return GuardrailResult(
            passed=False,
            message="输出包含不完整的代码块"
        )
    return GuardrailResult(passed=True)

agent = Agent(
    name="SafeAssistant",
    instructions="帮助用户解决技术问题",
    guardrails=[input_guardrail, output_guardrail],
    tools=[search_knowledge_base]
)

result = Runner.run_sync(agent, "如何优化 RAG 系统的检索精度？")
```

### 1.3 多 Agent 编排

```python
from agents import Agent, Runner, orchestrate

# 专业 Agent
researcher = Agent(
    name="Researcher",
    instructions="搜索和整理技术资料",
    tools=[search_knowledge_base]
)

writer = Agent(
    name="Writer",
    instructions="将技术资料整理为文档"
)

reviewer = Agent(
    name="Reviewer",
    instructions="检查文档的技术准确性"
)

# 编排多 Agent 工作流
async def research_pipeline(topic: str):
    # 阶段 1：研究
    research = await Runner.run(researcher, f"收集 {topic} 的最新资料")

    # 阶段 2：写作
    draft = await Runner.run(
        writer,
        f"基于以下资料撰写文档：\n{research.final_output}"
    )

    # 阶段 3：审查
    review = await Runner.run(
        reviewer,
        f"审查以下文档的技术准确性：\n{draft.final_output}"
    )

    return review.final_output
```

## 2. Claude Code

### 2.1 基础用法

Claude Code 是 Anthropic 的 CLI-first 开发 Agent，2026 年已升级到 Dynamic Workflows。

```bash
# 安装
npm install -g @anthropic/claude-code

# 启动交互式会话
claude

# 直接运行任务
claude "重构 src/auth 模块，添加 OAuth 2.1 支持"
```

### 2.2 Dynamic Workflows（2026-05）

```bash
# 并行执行多 Agent 任务
claude /task:refactor-auth --agents 3

# Agent 间代码审查
claude /code-review --pr 42 --reviewers 2

# 工具按需加载（减少 85% context）
claude --tool-search "Search for tools related to database operations"
```

### 2.3 AGENTS.md 配合使用

在项目根目录创建 `AGENTS.md`，Claude Code 会自动读取：

```markdown
# Project AGENTS.md

## Build Commands
- `npm run build` - 构建项目
- `npm test` - 运行测试
- `npm run lint` - 代码检查

## Architecture
- `src/` - 源代码
- `src/api/` - API 路由
- `src/db/` - 数据库操作

## Code Style
- TypeScript strict mode
- 函数命名：camelCase
- 组件命名：PascalCase
```

### 2.4 自定义 Skills

```bash
# 创建可复用的 Skill
mkdir -p .claude/skills
cat > .claude/skills/database-review.md << 'EOF'
# Database Review Skill

当审查涉及数据库的 PR 时：
1. 检查 SQL 注入风险
2. 确认索引使用
3. 验证事务边界
4. 检查 N+1 查询问题
EOF
```

## 3. 选型决策

| 维度 | OpenAI Agents SDK | Claude Code |
|------|-------------------|-------------|
| 定位 | 通用 Agent SDK | 代码开发 Agent |
| 首要场景 | 构建 Agent 应用 | 辅助软件工程 |
| 运行模式 | Python SDK + API | CLI 交互 |
| 模型绑定 | OpenAI 模型 | Claude 模型 |
| 托管 | 是（OpenAI 运行时） | 本地 CLI |
| 适用团队 | AI 应用开发者 | 全栈/后端开发者 |
| 学习成本 | 低 | 低 |

## 深度分析

OpenAI Agents SDK 和 Claude Code 代表了 Agent 开发的两条不同路径。OpenAI Agents SDK 是通用 Agent SDK，定位是"让 Python 开发者构建 Agent 应用"——通过 @function_tool 装饰器定义工具、通过 Guardrail 实现安全控制、通过 Runner orchestrate 实现多 Agent 编排，所有逻辑在 API 端托管运行。其核心优势是与 OpenAI 生态的深度整合（模型、护栏、托管运行时），适合构建面向外部的 Agent 服务。

Claude Code 则走了一条不同的路——CLI-first 的交互模式使其天然适合软件工程场景。Dynamic Workflows（并行多 Agent 任务）、AGENTS.md（项目级上下文配置）和自定义 Skills（可复用的审查/构建流程）构成了"Agent 辅助开发"的新范式。其 Tool Search 功能通过按需加载工具将 Context 消耗降低 85%，解决了长会话中的 Token 膨胀问题。两条路径并非互斥——可以用 Claude Code 辅助开发，将产出部署为 OpenAI Agents SDK 托管的 Agent 服务。

## Checklist

- [ ] 安装并配置 OpenAI Agents SDK 开发环境
- [ ] 为 Agent 定义 @function_tool 工具，包含清晰的描述和参数 Schema
- [ ] 实现输入护栏和输出护栏的双向安全检查
- [ ] 使用 Runner.run / Runner.run_sync 编排多 Agent 工作流
- [ ] 安装 Claude Code CLI 并配置项目级 AGENTS.md
- [ ] 创建自定义 Skills 目录（.claude/skills/）封装团队最佳实践
- [ ] 使用 Dynamic Workflows 并行执行多 Agent 开发任务
- [ ] 评估 OpenAI Agents SDK vs Claude Code 的适用场景
- [ ] 实现 Agent 执行的日志采集和性能监控
- [ ] 将自定义工具封装为兼容两大 SDK 的标准化接口

## 延伸阅读

- `../03-Agent开发/01-AI-Agent开发实战.md` — Agent 开发的核心概念、框架选型和状态机设计
- `../03-Agent开发/02-Agent实战案例集.md` — 多个真实场景的 Agent 实现，可与 SDK 方式对照学习
- `../04-AI应用架构设计/02-Agent应用架构：MCP+A2A模式.md` — 将 SDK 工具封装为标准化 MCP Server
- `../../AI架构师知识库/06-MCP与A2A协议设计.md` — 协议层标准化，实现跨 SDK 的工具复用

---

*最后更新：2026-06-12*

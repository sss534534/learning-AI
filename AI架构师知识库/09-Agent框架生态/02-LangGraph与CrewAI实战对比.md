# LangGraph 与 CrewAI 实战对比

> 2026 年两大主流 Agent 框架的深度对比——图工作流 vs 角色化协作

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [[01-Agent基础架构]], [[02-Multi-Agent协作架构]]
- **关联文件**: [[03-OpenAI Agents SDK与MS Agent Framework]], [[04-开源框架Hermes与ClaudeCode]]
- **最后更新**: 2026-06-12

## 1. 设计哲学对比

| 维度 | LangGraph | CrewAI |
|------|-----------|--------|
| 核心理念 | 有向图 (DAG) 工作流 | 角色化团队协作 |
| 抽象层级 | 低（Node + Edge） | 高（Agent + Task + Crew） |
| 学习曲线 | 陡峭 | 平缓 |
| 灵活性 | 极高，可控制每个执行细节 | 中等，约定优于配置 |
| 开箱即用 | 需手动搭建状态图 | 快速启动，5 行代码运行 |
| 状态管理 | 内置 State + Checkpoint | 隐式状态管理 |
| 人机协同 | 内置 interrupt 机制 | 需自行实现 |

## 2. 核心概念对比

### LangGraph 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| StateGraph | 状态图的定义入口 | 整个流程的蓝图 |
| Node | 执行单元（函数或可调用对象） | 一个步骤 |
| Edge | 节点间的连接 | 流程线 |
| State | 跨节点的共享状态 | 全局变量 |
| Checkpoint | 状态持久化与恢复 | 快照 |
| Interrupt | 暂停流程等待人工输入 | 闸门 |

### CrewAI 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| Agent | 拥有角色/目标/背景的智能体 | 团队成员 |
| Task | 分配给 Agent 的任务描述 | 工作项 |
| Crew | Agent + Task 的集合 | 项目团队 |
| Process | 执行流程（顺序/层级） | 协作模式 |
| Tool | Agent 可使用的工具 | 工具箱 |

## 3. 实战对比：新闻摘要 Agent

### LangGraph 实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class NewsState(TypedDict):
    urls: List[str]
    articles: List[str]
    summary: str

async def fetch_articles(state: NewsState):
    articles = []
    for url in state["urls"]:
        articles.append(await scrape(url))
    return {"articles": articles}

async def summarize(state: NewsState):
    combined = "\n".join(state["articles"])
    summary = await llm(f"总结以下新闻：\n{combined}")
    return {"summary": summary}

graph = StateGraph(NewsState)
graph.add_node("fetch", fetch_articles)
graph.add_node("summarize", summarize)
graph.add_edge("fetch", "summarize")
graph.add_edge("summarize", END)
graph.set_entry_point("fetch")
app = graph.compile()

result = await app.ainvoke({
    "urls": ["https://news.example.com/ai-2026"]
})
print(result["summary"])
```

### CrewAI 实现

```python
from crewai import Agent, Task, Crew

reporter = Agent(
    role="科技新闻记者",
    goal="收集并总结最新 AI 新闻",
    backstory="你是一名经验丰富的科技记者，擅长提炼关键信息",
    tools=[search_tool, scrape_tool]
)

task = Task(
    description="搜索 {topic} 相关的最新新闻，\
                 提取关键信息，\
                 生成 200 字以内的摘要",
    agent=reporter,
    expected_output="一段简洁的新闻摘要"
)

crew = Crew(
    agents=[reporter],
    tasks=[task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI Agents 2026"})
print(result)
```

## 4. 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 复杂工作流编排 | LangGraph | 图结构灵活控制执行流 |
| 快速 MVP 验证 | CrewAI | 最少代码上线 |
| 生产级 RAG 系统 | LangGraph | 状态管理 + 持久化 + LangSmith 可观测 |
| 多 Agent 协作 | CrewAI | 角色化设计开箱即用 |
| 需人机协同 | LangGraph | 内置 interrupt 机制 |
| 团队新手上手 | CrewAI | 学习曲线更平缓 |
| 需要极致控制 | LangGraph | 每个细节都可定制 |

## 5. 生态对比

| 维度 | LangGraph | CrewAI |
|------|-----------|--------|
| 可观测性 | LangSmith（成熟） | 内置可视化面板 |
| 模板/分享 | LangHub | 插件市场 |
| 第三方集成 | 100+ 集成 | 50+ 集成 |
| 部署选项 | LangServe、自托管 | 自托管、企业版 |
| 社区规模 | 最大 | 快速增长 |
| 文档质量 | 优秀 | 良好 |

## 深度分析

LangGraph 与 CrewAI 代表了 Agent 框架设计的两种截然不同的哲学：底层图工作流引擎 vs 高层角色化协作抽象。LangGraph 以 StateGraph + Node + Edge 为核心，给予开发者对执行流的绝对控制权，适合复杂工作流编排和生产级 RAG 系统；CrewAI 以 Agent + Task + Crew 的抽象，让开发者 5 行代码就能运行一个多 Agent 系统，适合快速 MVP 验证和团队上手。

选型的核心考量是"控制 vs 效率"的权衡。需要人机协同（interrupt 机制）、精细状态管理和 LangSmith 可观测性的场景选 LangGraph；需要快速搭建多 Agent 协作原型、团队学习成本敏感的场景选 CrewAI。两者不是替代关系，一个成熟团队可能同时使用——先用 CrewAI 验证想法，再用 LangGraph 重构生产版本。

值得注意的是生态对比中的可观测性和部署选项差异。LangGraph 背靠 LangSmith 的成熟可观测体系，而 CrewAI 的内置可视化面板更适合开发调试。生产部署方面，LangServe 提供了完善的模型服务能力，而 CrewAI 的企业版仍在完善中。

## Checklist

- [ ] 当前场景需要图工作流还是角色化协作？
- [ ] 是否需要 LangGraph 的 interrupt 人机协同机制？
- [ ] 团队对学习曲线和上手速度的容忍度如何？
- [ ] 是否需要 LangSmith 的 Agent 可观测性？
- [ ] MVP 阶段是否可以先选 CrewAI 再迁移？
- [ ] 状态管理是否需要精细的 Checkpoint 控制？
- [ ] 是否考虑框架的社区规模和长期维护？
- [ ] 生产部署是否需要 LangServe 的模型服务？
- [ ] 是否评估过框架的第三方集成丰富度？
- [ ] 是否考虑过框架切换的迁移成本？

## 延伸阅读

- [[03-OpenAI Agents SDK与MS Agent Framework]] — 商业框架的对比参考
- [[04-开源框架Hermes与ClaudeCode]] — 开源生态的另一极
- [[02-Multi-Agent协作架构]] — 框架底层协作模式的理论基础
- [[01-Agent基础架构]] — 框架之上的架构设计原则
- [[04-Agent架构评审方法论]] — 框架选型的评审维度

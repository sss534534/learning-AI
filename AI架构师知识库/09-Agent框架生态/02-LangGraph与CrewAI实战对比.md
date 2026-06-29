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

LangGraph 与 CrewAI 代表了 Agent 框架设计的两种截然不同的哲学：底层图工作流引擎 vs 高层角色化协作抽象。

### 选型的核心权衡：控制 vs 效率

```
控制导向 (LangGraph)                 效率导向 (CrewAI)
    │                                      │
    │ 可以精确控制每一步                     │ 5行代码跑起来
    │ 适合复杂工作流                        │ 适合快速验证
    │ 适合生产级系统                        │ 适合新手团队
    │ 学习曲线陡峭                         │ 学习曲线平缓
    │ 需要更多代码                         │ 约定优于配置
    │                                      │
```

选型不是非此即彼。一个成熟团队可能同时使用——先用 CrewAI 验证想法，再用 LangGraph 重构生产版本。

### 迁移成本分析

| 从 → 到 | 迁移成本 | 主要变更 |
|---------|---------|---------|
| CrewAI → LangGraph | 高 | 整个执行模型不同（角色→图），需重写所有Agent逻辑 |
| LangGraph → CrewAI | 中 | 有向图可映射为顺序执行序列，但丢失图表达力 |
| 两者 → 自研 | 极高 | 自定义状态管理、持久化、可观测性 |

**迁移成本预测：**
- CrewAI → LangGraph 的生产迁移，一个中等复杂度的Agent系统（10+ Agent）约需2-4周
- 风险：CrewAI的隐式状态在LangGraph中需显式定义，边界情况容易遗漏
- 建议：MVP阶段就确定最终框架，迁移代价高于框架本身的任何特性差异

### 供应商锁定风险评估

| 风险维度 | LangGraph | CrewAI |
|---------|-----------|--------|
| 协议层锁定 | 低（MCP/A2A兼容） | 低（MCP/A2A兼容） |
| 运行时锁定 | 中（LangGraph状态图独有） | 低（标准Python） |
| 可观测性锁定 | 高（LangSmith深度集成） | 低（开源面板） |
| 部署锁定 | 中（LangServe） | 低（标准容器） |
| 社区依赖 | 中（LangChain公司主导） | 中（CrewAI公司主导） |
| 数据可迁移 | 中（Checkpoint格式专有） | 高（标准JSON） |

**缓解策略：**
- 在框架上层封装抽象层（如统一的Agent接口），降低迁移时的代码改动量
- Agent逻辑放在独立的微服务中而非框架内，框架只做编排层
- 评估集和测试用例与框架解耦，切换框架可复用

### 组织准备度评估

```
你的团队适合哪个框架？

问题1：团队对Agent的熟悉程度
  ├─ 新手（<3个月）→ CrewAI（低门槛，快速建立认知）
  └─ 有经验（>6个月）→ LangGraph（控制力换取长期灵活性）

问题2：对可观测性的要求
  ├─ 生产系统需要完整追踪 → LangGraph + LangSmith
  └─ 开发调试级即可 → CrewAI内置面板

问题3：对人机协同的需求
  ├─ 需要（客服审核、人工介入）→ LangGraph interrupt
  └─ 不需要 → 两者均可

问题4：长期维护团队规模
  ├─ >5人 → LangGraph（分工明确，图结构可读性强）
  └─ <3人 → CrewAI（少代码量，减少维护负担）
```

### 2026年的新变化

1. **LangGraph 支持了 Agent 角色定义的简化模式**，降低了上手门槛，但核心仍是图模型
2. **CrewAI 引入了 Flow 模式**，一定程度支持了流程控制，但灵活性仍不及 LangGraph
3. **MCP/A2A 协议的成熟**使得框架层面的通信差异减小，Agent工具和Agent间通信逐渐标准化
4. **两者都开始支持对方的核心模式**——LangGraph 加入高层Agent API，CrewAI 加入底层流程控制

趋势判断：2027年两者的功能差距将持续缩小，选型将更多取决于团队偏好和生态绑定。

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

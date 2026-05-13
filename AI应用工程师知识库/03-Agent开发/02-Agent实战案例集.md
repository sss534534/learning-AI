# Agent实战案例集

> 包含多个真实场景的Agent实现案例

## 目录

1. [智能代码审查Agent](#1-智能代码审查agent)
2. [数据分析Agent](#2-数据分析agent)
3. [智能客服Agent](#3-智能客服agent)
4. [多Agent协作工作流](#4-多agent协作工作流)
5. [知识图谱问答Agent](#5-知识图谱问答agent)

---

## 1. 智能代码审查Agent

### 1.1 场景描述

自动审查代码提交，发现潜在问题并给出优化建议。

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    代码审查Agent                           │
├─────────────────────────────────────────────────────────────┤
│                                                           │
│  GitHub API ──→ 代码获取 ──→ 语法分析 ──→ LLM审查         │
│                                   ↓                       │
│                              问题识别                      │
│                                   ↓                       │
│                         生成改进建议                       │
│                                   ↓                       │
│                          输出审查报告                      │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 实现代码

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import subprocess
import json

# 工具定义
@tool
def fetch_git_diff(repo_path: str) -> str:
    """获取当前git diff"""
    result = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return result.stdout

@tool
def run_linter(file_path: str) -> str:
    """运行代码检查工具"""
    try:
        result = subprocess.run(
            ["flake8", file_path],
            capture_output=True,
            text=True
        )
        return result.stdout if result.returncode == 0 else f"错误: {result.stderr}"
    except Exception as e:
        return f"lint工具未安装: {str(e)}"

@tool
def check_type_hints(file_path: str) -> str:
    """检查类型注解"""
    try:
        result = subprocess.run(
            ["mypy", file_path],
            capture_output=True,
            text=True
        )
        return result.stdout if result.returncode == 0 else f"类型错误: {result.stderr}"
    except Exception as e:
        return f"mypy未安装: {str(e)}"

@tool
def search_code_pattern(pattern: str, repo_path: str) -> str:
    """搜索代码模式"""
    result = subprocess.run(
        ["grep", "-rn", pattern, repo_path],
        capture_output=True,
        text=True
    )
    return result.stdout

# 审查Prompt
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一位资深代码审查专家。请基于以下代码变更进行全面审查：

审查维度：
1. 代码正确性 - 是否有语法错误、逻辑漏洞
2. 代码风格 - 是否符合PEP8规范
3. 类型安全 - 是否有类型注解问题
4. 性能优化 - 是否有性能改进空间
5. 安全性 - 是否有安全隐患
6. 可维护性 - 代码结构是否清晰

输出格式：
## 问题清单
### 严重问题
- [问题描述] (位置: 文件名:行号)
  - 影响: [说明]
  - 建议: [修复方案]

### 改进建议
- [建议内容]

## 代码质量评分
- 正确性: 1-10分
- 可读性: 1-10分
- 安全性: 1-10分
- 整体评分: 1-10分
"""),
    ("human", "代码变更:\n{input}\n\n审查结果:"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [fetch_git_diff, run_linter, check_type_hints, search_code_pattern]

agent = create_tool_calling_agent(llm, tools, review_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用示例
result = agent_executor.invoke({
    "input": "审查当前git暂存区的代码变更"
})
print(result["output"])
```

---

## 2. 数据分析Agent

### 2.1 场景描述

自动化数据分析工作流：加载数据 → 探索分析 → 可视化 → 生成报告

### 2.2 实现代码

```python
import pandas as pd
import matplotlib.pyplot as plt
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

@tool
def load_csv(file_path: str) -> str:
    """加载CSV文件并返回数据摘要"""
    df = pd.read_csv(file_path)
    return f"""
数据摘要:
- 行数: {len(df)}
- 列数: {len(df.columns)}
- 列名: {', '.join(df.columns)}
- 数据类型:
{df.dtypes.to_string()}
- 前5行预览:
{df.head().to_string()}
"""

@tool
def describe_data(file_path: str) -> str:
    """生成数据统计描述"""
    df = pd.read_csv(file_path)
    return df.describe().to_string()

@tool
def plot_data(file_path: str, plot_type: str, x_col: str, y_col: str) -> str:
    """生成数据可视化图表"""
    df = pd.read_csv(file_path)
    
    plt.figure(figsize=(10, 6))
    
    if plot_type == "line":
        df.plot(x=x_col, y=y_col, kind='line')
    elif plot_type == "bar":
        df.plot(x=x_col, y=y_col, kind='bar')
    elif plot_type == "scatter":
        df.plot(x=x_col, y=y_col, kind='scatter')
    elif plot_type == "hist":
        df[y_col].hist()
    
    plot_path = f"/tmp/{x_col}_{y_col}_{plot_type}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return f"图表已保存到: {plot_path}"

@tool
def query_data(file_path: str, query: str) -> str:
    """执行数据查询"""
    df = pd.read_csv(file_path)
    
    try:
        result = df.query(query)
        return result.to_string()
    except Exception as e:
        return f"查询失败: {str(e)}"

# 创建Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [load_csv, describe_data, plot_data, query_data]

prompt = PromptTemplate.from_template("""
你是一位数据分析专家。请根据用户请求完成数据分析任务。

可用工具: {tool_names}

工具描述: {tool_descriptions}

任务: {input}

请按照以下格式输出:
思考: [你的思考过程]
行动: 工具名(参数)
观察: [工具返回结果]
...
最终答案: [总结报告]
""")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用示例
result = executor.invoke({
    "input": "分析sales_data.csv：1) 查看数据摘要 2) 生成销售额统计 3) 创建月度销售趋势图"
})
```

---

## 3. 智能客服Agent

### 3.1 场景描述

企业级智能客服，支持多轮对话、知识检索、工单创建等功能。

### 3.2 实现代码

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 初始化知识库
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory="./support_knowledge", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取相关信息"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建工单"""
    ticket = {
        "id": f"TICKET-{hash(title) % 10000:04d}",
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open"
    }
    return f"工单已创建成功!\n工单ID: {ticket['id']}\n优先级: {ticket['priority']}"

@tool
def check_ticket_status(ticket_id: str) -> str:
    """查询工单状态"""
    # 模拟数据库查询
    status_db = {
        "TICKET-1234": {"status": "处理中", "assignee": "张工"},
        "TICKET-5678": {"status": "已解决", "assignee": "李工"}
    }
    return status_db.get(ticket_id, {"status": "未找到", "assignee": "-"})

@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    user_db = {
        "u1001": {"name": "张三", "level": "VIP", "points": 5000},
        "u1002": {"name": "李四", "level": "普通", "points": 1200}
    }
    return user_db.get(user_id, {"name": "未知", "level": "普通", "points": 0})

# 客服Prompt
support_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一位专业的客服机器人。请遵循以下规则：

1. 优先使用知识库回答问题
2. 如果知识库信息不足，清晰说明并建议创建工单
3. 根据用户等级提供相应服务
4. 语气友好、专业

可用工具: {tool_names}

工具描述: {tool_descriptions}
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
tools = [search_knowledge, create_ticket, check_ticket_status, get_user_info]

agent = create_tool_calling_agent(llm, tools, support_prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用示例
result = executor.invoke({
    "input": "我是VIP用户张三(u1001)，我的订单还没发货，能帮我查一下吗？"
})
```

---

## 4. 多Agent协作工作流

### 4.1 场景描述

内容创作工作流：研究 → 写作 → 审核 → 发布

### 4.2 实现代码（CrewAI）

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 定义工具
@tool
def web_search(query: str) -> str:
    """搜索网络信息"""
    return f"搜索结果关于'{query}'..."

@tool
def save_document(content: str, filename: str) -> str:
    """保存文档"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"文档已保存: {filename}"

@tool
def read_document(filename: str) -> str:
    """读取文档"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# 定义Agent角色
researcher = Agent(
    role="资深研究员",
    goal="深入研究指定主题，收集最新信息和数据",
    backstory="你是一位拥有10年经验的科技研究员，擅长从多渠道获取信息并进行深度分析。",
    tools=[web_search],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o")
)

writer = Agent(
    role="技术作家",
    goal="撰写高质量技术文章",
    backstory="你是一位技术写作专家，擅长将复杂技术概念转化为清晰易懂的文章。",
    tools=[save_document],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o")
)

editor = Agent(
    role="内容审核员",
    goal="审核和优化文章质量",
    backstory="你是一位严格的内容审核专家，关注准确性、完整性和可读性。",
    tools=[read_document],
    verbose=True,
    llm=ChatOpenAI(model="gpt-4o")
)

# 定义任务
research_task = Task(
    description="研究2024年AI Agent技术的最新发展趋势，包括主要框架、典型应用场景和未来展望",
    agent=researcher,
    expected_output="结构化的研究报告，包含关键数据和引用来源"
)

write_task = Task(
    description="基于研究报告撰写一篇技术文章，要求结构清晰、内容详实、语言流畅",
    agent=writer,
    expected_output="完整的技术文章（约2000字）",
    context=[research_task]
)

review_task = Task(
    description="审核文章质量，检查技术准确性、逻辑连贯性和写作风格",
    agent=editor,
    expected_output="审核报告和修改建议",
    context=[write_task]
)

# 组建团队
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, review_task],
    process=Process.sequential,
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

---

## 5. 知识图谱问答Agent

### 5.1 场景描述

基于知识图谱进行复杂关系查询和推理

### 5.2 实现代码

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from pyvis.network import Network
import json

# 模拟知识图谱
knowledge_graph = {
    "nodes": [
        {"id": "张小明", "type": "person", "title": "产品经理"},
        {"id": "李华", "type": "person", "title": "工程师"},
        {"id": "产品A", "type": "product", "category": "SaaS"},
        {"id": "团队X", "type": "team", "department": "研发"},
        {"id": "技术文档", "type": "document", "format": "PDF"}
    ],
    "edges": [
        {"source": "张小明", "target": "产品A", "relation": "负责"},
        {"source": "李华", "target": "产品A", "relation": "开发"},
        {"source": "张小明", "target": "团队X", "relation": "管理"},
        {"source": "李华", "target": "团队X", "relation": "成员"},
        {"source": "产品A", "target": "技术文档", "relation": "关联"}
    ]
}

@tool
def query_graph(query: str) -> str:
    """查询知识图谱"""
    results = []
    
    # 简单的图查询
    for edge in knowledge_graph["edges"]:
        if query in edge["source"] or query in edge["target"]:
            results.append(f"{edge['source']} -{edge['relation']}-> {edge['target']}")
    
    for node in knowledge_graph["nodes"]:
        if query in node["id"]:
            results.append(f"节点信息: {node}")
    
    return "\n".join(results) if results else "未找到相关信息"

@tool
def find_path(start: str, end: str) -> str:
    """查找两点之间的路径"""
    visited = set()
    path = []
    
    def dfs(node, target, current_path):
        if node == target:
            return current_path + [node]
        
        visited.add(node)
        for edge in knowledge_graph["edges"]:
            if edge["source"] == node and edge["target"] not in visited:
                result = dfs(edge["target"], target, current_path + [node])
                if result:
                    return result
            if edge["target"] == node and edge["source"] not in visited:
                result = dfs(edge["source"], target, current_path + [node])
                if result:
                    return result
        return None
    
    result = dfs(start, end, [])
    if result:
        return " -> ".join(result)
    return "未找到路径"

@tool
def visualize_graph(output_path: str = "graph.html") -> str:
    """可视化知识图谱"""
    net = Network(notebook=True, directed=True)
    
    for node in knowledge_graph["nodes"]:
        color = "blue" if node["type"] == "person" else "green"
        net.add_node(node["id"], label=node["id"], color=color)
    
    for edge in knowledge_graph["edges"]:
        net.add_edge(edge["source"], edge["target"], label=edge["relation"])
    
    net.write_html(output_path)
    return f"图谱已保存到: {output_path}"

# 创建Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [query_graph, find_path, visualize_graph]

prompt = PromptTemplate.from_template("""
你是一位知识图谱专家。请根据用户请求查询或分析知识图谱。

可用工具: {tool_names}

工具描述: {tool_descriptions}

问题: {input}

请按照以下格式输出:
思考: [你的思考]
行动: 工具名(参数)
观察: [结果]
最终答案: [总结]
""")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用示例
result = executor.invoke({
    "input": "张小明和技术文档之间有什么联系？"
})
```

---

## 6. Agent开发最佳实践

### 6.1 工具设计原则

| 原则 | 说明 |
|------|------|
| 单一职责 | 每个工具只做一件事 |
| 描述清晰 | 工具描述要明确、具体 |
| 参数校验 | 添加输入验证 |
| 错误处理 | 返回友好的错误信息 |
| 幂等性 | 多次调用结果一致 |

### 6.2 Agent调试技巧

1. **启用verbose模式** - 查看思考过程
2. **日志追踪** - 使用LangSmith记录执行过程
3. **单元测试** - 为每个工具编写测试
4. **边界测试** - 测试极端输入

### 6.3 性能优化

| 优化项 | 方案 |
|--------|------|
| 工具缓存 | 缓存重复查询结果 |
| 异步调用 | 并行执行独立工具 |
| 流式输出 | 渐进式返回结果 |
| 资源限制 | 设置超时和重试策略 |

---

*最后更新：2026-05-12*

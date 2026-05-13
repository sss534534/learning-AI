# AI Agent开发实战

> 构建智能Agent应用的完整指南

## 1. Agent开发框架

### 1.1 框架对比选型

| 框架 | 语言 | 特点 | 学习曲线 | 适用场景 |
|------|------|------|----------|----------|
| **LangChain + LangGraph** | Python/JS | 生态最大、灵活 | 中 | 通用开发 |
| **LlamaIndex** | Python | RAG+Agent一体化 | 低 | 知识密集型 |
| **CrewAI** | Python | 多Agent协作、易用 | 低 | 业务流程 |
| **AutoGen** | Python | 对话驱动、研究导向 | 中 | 研究/探索 |
| **Spring AI** | Java | Java生态集成 | 中 | 企业级Java应用 |
| **LangChain4j** | Java | 声明式AI服务 | 低 | Java后端 |

### 1.2 LangGraph核心概念

**为什么用LangGraph？**
- LangChain适合简单链式调用
- LangGraph适合**有循环、条件分支、状态管理**的复杂Agent

**核心抽象：**
```
StateGraph
├── State: 状态定义（TypedDict）
├── Node: 执行节点（函数）
├── Edge: 普通边（固定流转）
├── ConditionalEdge: 条件边（动态路由）
└── Memory: 持久化状态
```

---

## 2. 单Agent开发

### 2.1 ReAct Agent

**最常用的Agent模式：思考→行动→观察→循环**

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

# 1. 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    # 实际调用搜索API
    return f"搜索结果：关于'{query}'的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算错误，请检查表达式"

@tool
def read_file(filepath: str) -> str:
    """读取本地文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

tools = [search_web, calculator, read_file]

# 2. 创建Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的AI助手，可以使用工具来完成任务。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
)

# 3. 执行
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,        # 打印思考过程
    max_iterations=10,   # 最大循环次数
    handle_parsing_errors=True
)

result = agent_executor.invoke({
    "input": "搜索2024年中国GDP数据，然后计算同比增长率（假设2023年为126万亿）"
})
```

### 2.2 LangGraph状态机Agent

**适合需要精细控制的场景：**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息历史（累加）
    query: str                                 # 用户查询
    tool_calls: List[str]                       # 已调用的工具
    final_answer: str                           # 最终答案

# 2. 定义节点
def router_node(state: AgentState):
    """路由节点：决定下一步"""
    query = state["query"]
    
    # 简单路由逻辑
    if "搜索" in query or "最新" in query:
        return "search"
    elif "计算" in query or "多少" in query:
        return "calculate"
    else:
        return "direct_answer"

def search_node(state: AgentState):
    """搜索节点"""
    result = search_web.invoke(state["query"])
    return {"messages": [{"role": "tool", "content": result}]}

def calculate_node(state: AgentState):
    """计算节点"""
    result = calculator.invoke(state["query"])
    return {"messages": [{"role": "tool", "content": result}]}

def answer_node(state: AgentState):
    """生成最终答案"""
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(state["messages"])
    return {"final_answer": response.content}

# 3. 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("router", router_node)
graph.add_node("search", search_node)
graph.add_node("calculate", calculate_node)
graph.add_node("answer", answer_node)

# 设置入口
graph.set_entry_point("router")

# 添加条件边
graph.add_conditional_edges(
    "router",
    router_node,
    {
        "search": "search",
        "calculate": "calculate",
        "direct_answer": "answer"
    }
)

# 添加普通边
graph.add_edge("search", "answer")
graph.add_edge("calculate", "answer")
graph.add_edge("answer", END)

# 4. 编译和运行
app = graph.compile()

result = app.invoke({
    "messages": [],
    "query": "搜索最新的AI新闻",
    "tool_calls": [],
    "final_answer": ""
})
```

### 2.3 带记忆的Agent

```python
from langgraph.checkpoint.memory import MemorySaver

# 添加记忆
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用thread_id区分会话
config = {"configurable": {"thread_id": "user_session_1"}}

# 第一轮对话
result1 = app.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}], "query": "我叫张三"},
    config=config
)

# 第二轮对话（Agent记得之前的信息）
result2 = app.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}], "query": "我叫什么名字？"},
    config=config
)
```

---

## 3. 工具开发

### 3.1 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个工具做一件事 | search_web vs search_and_analyze |
| **清晰描述** | 工具名和描述要明确 | get_weather(city) |
| **参数校验** | 验证输入参数 | 类型检查、范围检查 |
| **错误处理** | 友好的错误信息 | 返回错误描述而非抛异常 |
| **幂等性** | 多次调用结果一致 | GET类操作 |

### 3.2 常用工具类型

**API调用工具：**
```python
@tool
def query_database(sql: str) -> str:
    """
    执行SQL查询数据库。
    
    Args:
        sql: SQL查询语句（只支持SELECT）
    
    Returns:
        查询结果（JSON格式）
    """
    # 安全校验：只允许SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return "错误：只支持SELECT查询"
    
    try:
        result = db.execute(sql)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"查询错误：{str(e)}"
```

**文件操作工具：**
```python
@tool
def write_file(filepath: str, content: str) -> str:
    """
    将内容写入文件。
    
    Args:
        filepath: 文件路径
        content: 文件内容
    """
    # 安全校验：限制路径范围
    allowed_dir = "/workspace"
    if not filepath.startswith(allowed_dir):
        return "错误：只能在允许的目录下操作"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入 {filepath}"
    except Exception as e:
        return f"写入错误：{str(e)}"
```

**HTTP请求工具：**
```python
@tool
def http_request(url: str, method: str = "GET", body: str = "") -> str:
    """
    发送HTTP请求。
    
    Args:
        url: 请求URL
        method: HTTP方法（GET/POST）
        body: 请求体（POST时使用）
    """
    import requests
    
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=json.loads(body), timeout=10)
        
        return f"状态码: {resp.status_code}\n响应: {resp.text[:500]}"
    except Exception as e:
        return f"请求错误：{str(e)}"
```

### 3.3 MCP工具集成

**MCP（Model Context Protocol）** 是标准化工具协议

```python
from langchain_mcp_adapters.tools import load_mcp_tools

# 加载MCP服务器提供的工具
tools = load_mcp_tools(
    server_url="http://localhost:3000",
    api_key="xxx"
)

# 像普通工具一样使用
agent = create_react_agent(llm, tools, prompt)
```

---

## 4. 多Agent系统

### 4.1 CrewAI多Agent

**适合业务流程自动化：**

```python
from crewai import Agent, Task, Crew, Process

# 1. 定义Agent角色
researcher = Agent(
    role="资深研究员",
    goal="搜集和分析信息",
    backstory="你是一位拥有10年经验的研究员，擅长从多渠道获取信息并进行深度分析。",
    tools=[search_web, read_file],
    verbose=True
)

writer = Agent(
    role="技术写手",
    goal="撰写高质量技术文档",
    backstory="你是一位技术写作专家，擅长将复杂技术概念转化为清晰易懂的文档。",
    tools=[write_file],
    verbose=True
)

reviewer = Agent(
    role="质量审核员",
    goal="确保文档质量",
    backstory="你是一位严格的质量审核员，关注准确性、完整性和可读性。",
    verbose=True
)

# 2. 定义任务
research_task = Task(
    description="研究{topic}的最新技术趋势和最佳实践",
    agent=researcher,
    expected_output="结构化的研究报告"
)

writing_task = Task(
    description="基于研究报告撰写技术文档",
    agent=writer,
    expected_output="完整的技术文档"
)

review_task = Task(
    description="审核文档质量并提出改进建议",
    agent=reviewer,
    expected_output="审核报告和最终版本"
)

# 3. 组建团队
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential  # 顺序执行
)

# 4. 执行
result = crew.kickoff(inputs={"topic": "RAG架构设计"})
```

### 4.2 LangGraph多Agent编排

**适合需要精细控制的协作流程：**

```python
# 多Agent状态
class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    research_result: str
    draft_content: str
    final_content: str

# 研究Agent节点
def research_agent(state):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"基于以下消息进行深入研究：{state['messages']}"
    result = llm.invoke(prompt)
    return {"research_result": result.content, "current_agent": "writer"}

# 写作Agent节点
def writer_agent(state):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"基于研究结果撰写文档：{state['research_result']}"
    result = llm.invoke(prompt)
    return {"draft_content": result.content, "current_agent": "reviewer"}

# 审核Agent节点
def reviewer_agent(state):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"审核以下文档：{state['draft_content']}"
    result = llm.invoke(prompt)
    return {"final_content": result.content, "current_agent": "done"}

# 构建多Agent图
graph = StateGraph(MultiAgentState)
graph.add_node("research", research_agent)
graph.add_node("write", writer_agent)
graph.add_node("review", reviewer_agent)

graph.set_entry_point("research")
graph.add_edge("research", "write")
graph.add_edge("write", "review")
graph.add_edge("review", END)
```

---

## 5. 人机协同（Human-in-the-Loop）

### 5.1 人工审批节点

```python
from langgraph.types import interrupt

def sensitive_action_node(state):
    """需要人工审批的操作"""
    # 暂停执行，等待人工输入
    action = interrupt({
        "question": "是否执行以下操作？",
        "action": state["proposed_action"],
        "options": ["批准", "拒绝", "修改"]
    })
    
    if action == "批准":
        return {"status": "approved"}
    elif action == "拒绝":
        return {"status": "rejected"}
    else:
        return {"status": "modified"}
```

### 5.2 人工介入触发条件

```python
def should_ask_human(state):
    """判断是否需要人工介入"""
    # 高风险操作
    if state.get("risk_level") == "high":
        return "human_review"
    
    # 低置信度
    if state.get("confidence", 1.0) < 0.7:
        return "human_review"
    
    # 涉及敏感信息
    if contains_sensitive_info(state["query"]):
        return "human_review"
    
    return "auto_proceed"
```

---

## 6. Agent安全

### 6.1 工具执行沙箱

```python
import subprocess
import tempfile
import os

def safe_execute(code: str, timeout: int = 30) -> str:
    """在沙箱中安全执行代码"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            # 限制资源
            cwd="/tmp/sandbox"
        )
        return result.stdout if result.returncode == 0 else f"错误: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "错误：执行超时"
    finally:
        os.unlink(temp_file)
```

### 6.2 权限控制

```python
# 工具权限分级
TOOL_PERMISSIONS = {
    "search_web": "all",           # 所有用户可用
    "calculator": "all",
    "query_database": "authenticated",  # 需要认证
    "write_file": "admin",         # 仅管理员
    "delete_file": "admin",
    "http_request": "authenticated",
}

def check_permission(tool_name, user_role):
    """检查工具使用权限"""
    required = TOOL_PERMISSIONS.get(tool_name, "admin")
    
    if required == "all":
        return True
    elif required == "authenticated" and user_role in ["user", "admin"]:
        return True
    elif required == "admin" and user_role == "admin":
        return True
    return False
```

---

## 7. 开发者Checklist

### 7.1 Agent开发Checklist

- [ ] 选择合适的Agent框架
- [ ] 设计Agent角色和职责
- [ ] 开发必要的工具
- [ ] 实现工具参数校验
- [ ] 实现错误处理和重试
- [ ] 添加记忆/状态管理
- [ ] 配置最大循环次数
- [ ] 实现人机协同机制
- [ ] 添加权限控制
- [ ] 实现日志和追踪
- [ ] 编写测试用例
- [ ] 性能测试和优化

### 7.2 常见陷阱

**陷阱1：无限循环**
- 问题：Agent反复调用同一工具
- 解决：设置max_iterations、添加循环检测

**陷阱2：工具描述不清**
- 问题：Agent选错工具或参数错误
- 解决：优化工具描述，添加示例

**陷阱3：状态爆炸**
- 问题：长对话状态过大
- 解决：定期清理、摘要压缩

**陷阱4：缺乏可观测性**
- 问题：Agent行为难以调试
- 解决：添加详细日志、使用LangSmith追踪

---

*最后更新：2026-05-07*

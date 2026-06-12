# 第八章：Agent框架与实践

&gt; 本章将深入介绍主流的Agent开发框架，包括LangChain、AutoGen和Semantic Kernel，并通过实战案例展示如何使用这些框架构建实际应用。

## 目录

1. [LangChain Agents](#1-langchain-agents)
2. [AutoGen多Agent](#2-autogen多agent)
3. [Semantic Kernel](#3-semantic-kernel)
4. [实战案例](#4-实战案例)

---

## 元数据
- **难度**: ⭐⭐
- **前置知识**: ../chapters/ch05-tool-calling.md, ../chapters/ch06-memory-system.md, ../chapters/ch07-multi-agent.md
- **关联文件**: ../chapters/ch05-tool-calling.md, ../chapters/ch07-multi-agent.md
- **最后更新**: 2026-06-12
---

## 1. LangChain Agents

### 1.1 LangChain概述

**LangChain** 是目前最流行的LLM应用开发框架，提供了丰富的组件用于构建Agent系统。

**LangChain的核心组件：

```
┌─────────────────────────────────────────────────────────┐
│                     LangChain                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  LLM    │  │  Prompt │  │  Memory  │  │  Tools  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │              │              │              │        │
│       └──────────────┼──────────────┼──────────────┘        │
│                      │                                     │
│               ┌───────▼───────┐                             │
│               │     Agent      │                             │
│               └───────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

### 1.2 LangChain Agent基础使用

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 定义工具
@tool
def search_web(query: str) -> str:
    """Search the web for information about a query."""
    # 实际实现会调用搜索引擎API
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [search_web, calculate]

# 2. 初始化LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the tools provided to answer the user's question."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. 创建Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. 创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 运行Agent
result = agent_executor.invoke({"input": "What is the population of Tokyo multiplied by 2?"})
print(result["output"])
```

### 1.3 ReAct Agent实现

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 获取预定义的ReAct提示
react_prompt = hub.pull("hwchase17/react")

# 创建ReAct Agent
react_agent = create_react_agent(llm, tools, react_prompt)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)

# 运行
result = react_executor.invoke({"input": "What is the square root of the population of Paris?"})
```

### 1.4 Plan-and-Execute Agent

```python
from langchain.agents import create_openai_functions_agent
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# 加载规划器
planner = load_chat_planner(llm)

# 加载执行器
executor = load_agent_executor(llm, tools, verbose=True)

# 创建Plan-and-Execute Agent
plan_execute_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 运行
result = plan_execute_agent.invoke({"input": "Plan a trip to Beijing, including transportation, accommodation, and activities"})
```

### 1.5 自定义Agent

```python
from langchain.agents import AgentOutputParser, AgentAction, AgentFinish
from langchain.schema import OutputParserException
from typing import Union
import re

class CustomOutputParser(AgentOutputParser):
    """自定义输出解析器"""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(tool=action, tool_input=action_input, log=text)

# 使用自定义解析器
class CustomAgent(Agent):
    """自定义Agent"""
    
    @classmethod
    def create_prompt(cls, tools):
        template = """You are a helpful assistant.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools]),
            },
        )
    
    @classmethod
    def _get_default_output_parser(cls):
        return CustomOutputParser()
    
    @property
    def _agent_type(self):
        return "custom-agent"
```

---

## 2. AutoGen多Agent

### 2.1 AutoGen概述

**AutoGen** 是微软开发的多Agent框架，专注于Agent之间的协作与对话。

**AutoGen核心概念：**
- **AssistantAgent：助手Agent
- **UserProxyAgent：用户代理Agent
- **GroupChat：多Agent群聊
- **ConversableAgent：可对话Agent基类

### 2.2 基础双Agent对话

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# 1. 配置LLM
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}

# 2. 创建助手Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

# 3. 创建用户代理Agent
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
)

# 4. 启动对话
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of stock price of NVIDIA for the last 1 year."
)
```

### 2.3 多Agent群聊

```python
from autogen import GroupChat, GroupChatManager

# 创建多个Agent
researcher = AssistantAgent(
    name="researcher",
    llm_config=llm_config,
    system_message="You are a researcher. You gather and analyze information."
)

writer = AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="You are a writer. You create well-structured reports."
)

critic = AssistantAgent(
    name="critic",
    llm_config=llm_config,
    system_message="You are a critic. You review and provide feedback."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
)

# 创建群聊
groupchat = GroupChat(
    agents=[user_proxy, researcher, writer, critic],
    messages=[],
    max_round=12
)

# 创建群聊管理器
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 启动群聊
user_proxy.initiate_chat(
    manager,
    message="Write a report about the future of AI in healthcare."
)
```

### 2.4 自定义Agent协作系统

```python
from autogen import ConversableAgent

# 创建代码执行Agent
code_executor = ConversableAgent(
    name="code_executor",
    llm_config=False,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    human_input_mode="NEVER",
)

# 创建规划Agent
planner = AssistantAgent(
    name="planner",
    llm_config=llm_config,
    system_message="You are a planner. You break down complex tasks into steps."
)

# 创建执行Agent
executor = AssistantAgent(
    name="executor",
    llm_config=llm_config,
    system_message="You are an executor. You execute code to complete tasks."
)

# 创建验证Agent
validator = AssistantAgent(
    name="validator",
    llm_config=llm_config,
    system_message="You are a validator. You verify the results."
)

# 注册对话流程
def custom_chat(initial_message):
    # 1. 规划阶段
    planner.initiate_chat(
        code_executor,
        message=initial_message,
        clear_history=False
    )
    
    # 2. 执行阶段
    executor.initiate_chat(
        code_executor,
        message="Execute the planned steps",
        clear_history=False
    )
    
    # 3. 验证阶段
    validator.initiate_chat(
        code_executor,
        message="Validate the results",
        clear_history=False
    )

custom_chat("Analyze the sales data from last quarter and create a visualization.")
```

### 2.5 工具注册与使用

```python
from typing import Annotated, Literal

# 定义工具函数
def search_tool(query: Annotated[str, "Search query"]) -> str:
    """Search the web."""
    return f"Searching: {query}"

def math_tool(expression: Annotated[str, "Math expression"]) -> str:
    """Calculate math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

# 注册工具
assistant.register_for_llm(name="search", description="Search the web")(search_tool)
assistant.register_for_llm(name="math", description="Calculate math")(math_tool)

# 注册执行器
user_proxy.register_for_execution(name="search")(search_tool)
user_proxy.register_for_execution(name="math")(math_tool)

# 使用
user_proxy.initiate_chat(
    assistant,
    message="What is 123 * 456?"
)
```

---

## 3. Semantic Kernel

### 3.1 Semantic Kernel概述

**Semantic Kernel** 是微软开发的轻量级AI应用开发框架，支持C#和Python。

**核心概念：**
- **Kernel：** 核心运行时
- **Plugins：** 插件（技能）
- **Functions：** 函数
- **Connectors：** 连接器

### 3.2 Semantic Kernel基础

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.core_plugins import MathPlugin, TimePlugin

# 1. 创建Kernel
kernel = sk.Kernel()

# 2. 添加AI服务
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_service(
    OpenAIChatCompletion("gpt-4", api_key=api_key, org_id=org_id)
)
kernel.add_service(
    OpenAITextEmbedding("text-embedding-ada-002", api_key=api_key, org_id=org_id)
)

# 3. 导入插件
kernel.import_plugin(MathPlugin(), "math")
kernel.import_plugin(TimePlugin(), "time")

# 4. 创建提示函数
prompt = """
Write a joke about the current time: {{time.now}}
"""

joke_function = kernel.create_function_from_prompt(
    function_name="TellJoke",
    plugin_name="JokePlugin",
    prompt=prompt,
)

# 5. 执行
result = await kernel.invoke(joke_function)
print(result)
```

### 3.3 自定义插件

```python
from semantic_kernel.functions import kernel_function
from typing import Annotated

class WeatherPlugin:
    """天气插件"""
    
    @kernel_function(
        name="GetWeather",
        description="Get the current weather for a location"
    )
    def get_weather(
        self,
        location: Annotated[str, "The city and state, e.g. San Francisco, CA"]
    ) -> str:
        """获取天气"""
        # 实际实现会调用天气API
        return f"Weather in {location}: 25°C, sunny"
    
    @kernel_function(
        name="GetForecast",
        description="Get the weather forecast"
    )
    def get_forecast(
        self,
        location: Annotated[str, "Location"],
        days: Annotated[int, "Number of days"] = 3
    ) -> str:
        """获取预报"""
        return f"Forecast for {location} for {days} days: ..."

# 导入插件
weather_plugin = WeatherPlugin()
kernel.import_plugin(weather_plugin, "weather")

# 使用
result = await kernel.invoke(
    kernel.plugins["weather"]["GetWeather"],
    location="Beijing"
)
print(result)
```

### 3.4 链式调用

```python
from semantic_kernel.functions import KernelArguments

# 创建函数链
async def process_query(query: str):
    # 步骤1: 分析问题
    analyze_prompt = """
    Analyze the user's query and determine what information is needed.
    Query: {{$query}}
    What tools are needed? (weather, time, math, etc.)
    """
    
    analyze_func = kernel.create_function_from_prompt(
        function_name="AnalyzeQuery",
        prompt=analyze_prompt
    )
    
    analysis = await kernel.invoke(analyze_func, KernelArguments(query=query))
    print(f"Analysis: {analysis}")
    
    # 步骤2: 获取所需信息
    if "weather" in str(analysis).lower():
        weather_result = await kernel.invoke(
            kernel.plugins["weather"]["GetWeather"],
            location="Beijing"
        )
        context = str(weather_result)
    else:
        context = ""
    
    # 步骤3: 生成回答
    answer_prompt = """
    Answer the user's query using the provided context.
    Query: {{$query}}
    Context: {{$context}}
    """
    
    answer_func = kernel.create_function_from_prompt(
        function_name="GenerateAnswer",
        prompt=answer_prompt
    )
    
    answer = await kernel.invoke(
        answer_func,
        KernelArguments(query=query, context=context)
    )
    
    return answer

result = await process_query("What's the weather like today?")
print(f"Answer: {result}")
```

### 3.5 Planner（规划器）

```python
from semantic_kernel.planners import FunctionCallingStepwisePlanner

# 创建规划器
planner = FunctionCallingStepwisePlanner(service_id="default")

# 使用规划器
ask = "What's the weather in Beijing and what is 25 squared?"

# 执行规划
result = await planner.invoke(kernel, ask)

print(f"Final answer: {result.final_answer}")
print(f"\nPlan used: {result.chat_history}")
```

---

## 4. 实战案例

### 4.1 案例1：智能文档助手（LangChain）

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool

# 1. 加载文档
loader = TextLoader("documents/manual.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 4. 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. 定义工具
def search_document(query: str) -> str:
    """Search the document for information."""
    result = qa_chain.invoke(query)
    return result["result"]

tools = [
    Tool(
        name="DocumentSearch",
        func=search_document,
        description="Search the document for information"
    )
]

# 6. 创建Agent
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document assistant. Use the DocumentSearch tool to find information."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. 使用
result = agent_executor.invoke({"input": "How do I reset the device?"})
```

### 4.2 案例2：多Agent代码审查系统（AutoGen）

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 配置
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list, "temperature": 0}

# 创建Agent
code_writer = AssistantAgent(
    name="code_writer",
    llm_config=llm_config,
    system_message="You are a Python developer. Write clean, well-documented code."
)

code_reviewer = AssistantAgent(
    name="code_reviewer",
    llm_config=llm_config,
    system_message="You are a code reviewer. Check for bugs, security issues, and best practices."
)

tester = AssistantAgent(
    name="tester",
    llm_config=llm_config,
    system_message="You are a tester. Write test cases and verify the code works."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# 群聊
groupchat = GroupChat(
    agents=[user_proxy, code_writer, code_reviewer, tester],
    messages=[],
    max_round=15
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 启动
user_proxy.initiate_chat(
    manager,
    message="Write a Python function to calculate Fibonacci numbers efficiently."
)
```

### 4.3 案例3：智能客服系统（Semantic Kernel）

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from typing import Annotated

# 定义客服插件
class CustomerServicePlugin:
    """客服插件"""
    
    @kernel_function(
        name="GetOrderStatus",
        description="Get order status"
    )
    def get_order_status(
        self,
        order_id: Annotated[str, "Order ID"]
    ) -> str:
        """获取订单状态"""
        return f"Order {order_id}: Shipped, arriving tomorrow"
    
    @kernel_function(
        name="CancelOrder",
        description="Cancel an order"
    )
    def cancel_order(
        self,
        order_id: Annotated[str, "Order ID"]
    ) -> str:
        """取消订单"""
        return f"Order {order_id} has been cancelled."
    
    @kernel_function(
        name="RefundOrder",
        description="Process a refund"
    )
    def refund_order(
        self,
        order_id: Annotated[str, "Order ID"],
        reason: Annotated[str, "Reason for refund"]
    ) -> str:
        """退款"""
        return f"Refund processed for order {order_id}. Reason: {reason}"

# 初始化
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_service(
    OpenAIChatCompletion("gpt-4", api_key=api_key, org_id=org_id)
)

# 导入插件
customer_plugin = CustomerServicePlugin()
kernel.import_plugin(customer_plugin, "customer_service")

# 创建客服提示
customer_service_prompt = """
You are a helpful customer service agent. Use the available tools to help the customer.

Customer: {{$input}}

Available tools:
- GetOrderStatus: Get order status
- CancelOrder: Cancel an order
- RefundOrder: Process a refund

Respond politely and helpfully.
"""

cs_function = kernel.create_function_from_prompt(
    function_name="CustomerService",
    plugin_name="CustomerServicePlugin",
    prompt=customer_service_prompt
)

# 客服聊天循环
async def customer_service_chat():
    print("Customer Service: Hello! How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Customer Service: Goodbye!")
            break
        
        result = await kernel.invoke(
            cs_function,
            input=user_input
        )
        print(f"Customer Service: {result}")

# 运行
import asyncio
asyncio.run(customer_service_chat())
```

### 4.4 综合案例：AI研究助手（多框架整合）

```python
# 使用LangChain进行文档处理
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# 文档处理
def load_and_process_urls(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# 使用Semantic Kernel进行编排
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# 研究助手插件
class ResearchHelperPlugin:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    @kernel_function(
        name="SearchResearch",
        description="Search research materials"
    )
    def search_research(
        self,
        query: Annotated[str, "Search query"]
    ) -> str:
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

# 主函数
async def research_assistant(query):
    # 1. 加载研究材料
    urls = [
        "https://example.com/paper1",
        "https://example.com/paper2"
    ]
    vectorstore = load_and_process_urls(urls)
    
    # 2. 初始化Semantic Kernel
    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_service(
        OpenAIChatCompletion("gpt-4", api_key=api_key, org_id=org_id)
    )
    
    # 3. 导入插件
    research_plugin = ResearchHelperPlugin(vectorstore)
    kernel.import_plugin(research_plugin, "research")
    
    # 4. 创建研究提示
    research_prompt = """
    You are a research assistant. Use the SearchResearch tool to find information and then summarize.
    
    Query: {{$query}}
    
    Please provide:
    1. Summary of key findings
    2. Key points
    3. References
    """
    
    research_func = kernel.create_function_from_prompt(
        function_name="DoResearch",
        prompt=research_prompt
    )
    
    # 5. 执行
    result = await kernel.invoke(
        research_func,
        query=query
    )
    
    return result

# 使用
result = await research_assistant("What are the latest developments in AI?")
print(result)
```

---

## 深度分析

三大主流Agent框架——LangChain、AutoGen和Semantic Kernel——代表了Agent开发的三种不同哲学。LangChain采用"工具链+Agent执行器"的架构，通过丰富的组件生态和链式调用简化了Agent的构建过程，其ReAct和Plan-and-Execute模式直观地体现了"思考-行动-观察"的循环。LangChain的优势在于生态成熟度和社区活跃度，但也存在抽象层过厚、调试困难的问题，在简单场景下可能显得过度工程化。AutoGen则聚焦于多Agent对话协作，通过GroupChat和ConversableAgent等抽象，将Agent间的交互建模为结构化对话，适合需要多个角色协同完成的复杂任务。

Semantic Kernel走的是轻量级路线，以Plugin和Kernel为核心，强调与现有企业系统的无缝集成。其Planner（规划器）能够根据用户需求自动编排插件调用序列，体现了"意图驱动"的Agent设计理念。SK的独特优势在于对多语言（C#/Python/Java）的支持，使其在企业级应用中具有天然的优势。从框架选型角度看，选择哪个框架不应仅看功能完备性，更应考虑与现有技术栈的匹配度、团队的学习成本以及社区的长期维护能力。

实战案例展示了三个框架在不同场景下的应用模式。值得注意的是，框架本身只是工具，Agent系统的质量更多取决于工具定义的质量、提示词的设计以及错误处理机制的完善程度。跨框架整合代表了未来的趋势——大型项目往往需要同时利用不同框架的优势：用LangChain处理文档和检索，用AutoGen管理多Agent协作，用Semantic Kernel做企业级服务的编排。理解每个框架的设计哲学和适用边界，比单纯掌握某个框架的API更为重要。

---

## Checklist

- [ ] 熟悉LangChain Agent的创建流程：工具定义→LLM初始化→提示模板→Agent创建→执行器
- [ ] 理解ReAct Agent的"思考-行动-观察"循环原理
- [ ] 掌握Plan-and-Execute Agent的规划与执行分离架构
- [ ] 熟悉AutoGen的核心概念：AssistantAgent/UserProxyAgent/GroupChat
- [ ] 实现AutoGen多Agent群聊系统，包括角色定义和对话管理
- [ ] 掌握AutoGen中的工具注册与使用（register_for_llm/register_for_execution）
- [ ] 熟悉Semantic Kernel的Kernel/Plugin/Function三层架构
- [ ] 掌握Semantic Kernel的自定义插件开发和链式调用
- [ ] 实现至少一个完整的Agent实战项目（文档助手/代码审查/客服系统）
- [ ] 理解跨框架整合的设计思路，比较各框架的适用场景

---

## 延伸阅读

- [第五章：工具调用与Function Calling](../chapters/ch05-tool-calling.md) - Agent框架中的工具调用基础
- [第六章：Agent记忆系统](../chapters/ch06-memory-system.md) - Agent框架中的记忆集成
- [第七章：多Agent协作系统](../chapters/ch07-multi-agent.md) - 多Agent框架设计原理
- LangChain官方文档 - https://python.langchain.com/docs/
- Semantic Kernel官方文档 - https://learn.microsoft.com/en-us/semantic-kernel/

---

## 本章小结

主流Agent框架各有优势：

1. **LangChain** - 丰富的组件生态，适合快速开发
2. **AutoGen** - 多Agent协作，适合复杂任务
3. **Semantic Kernel** - 轻量级，支持多语言

**选择建议：**
- 快速原型：LangChain
- 多Agent协作：AutoGen
- 企业级应用：Semantic Kernel

**下一章：** 我们将学习Agent评估与优化方法。

---

*最后更新: 2026-06-12*
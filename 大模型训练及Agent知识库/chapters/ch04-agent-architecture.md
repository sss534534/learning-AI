# 第四章：Agent架构设计

> Agent（智能体）是基于大语言模型的自主系统，能够感知环境、做出决策并执行动作。本章将深入讲解Agent的核心概念、架构模式以及主流设计范式。

## 目录

1. [Agent基础概念](#1-agent基础概念)
2. [ReAct架构](#2-react架构)
3. [Plan-and-Execute架构](#3-plan-and-execute架构)
4. [反射与自我改进](#4-反射与自我改进)
5. [多步推理与工具使用](#5-多步推理与工具使用)

---

## 1. Agent基础概念

### 1.1 什么是Agent？

**Agent定义：**
Agent是一个能够自主感知环境、做出决策并执行动作的实体。

**LLM-based Agent的核心组件：**

```
┌─────────────────────────────────────────────────────────┐
│                      Agent系统                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   感知模块   │───→│   思考模块   │───→│   行动模块   │ │
│  │ (Perception)│    │  (Reasoning)│    │   (Action)  │ │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘ │
│                            │                   │        │
│                            ▼                   ▼        │
│                     ┌─────────────┐    ┌─────────────┐  │
│                     │   记忆系统   │    │   工具集合   │  │
│                     │  (Memory)   │    │   (Tools)   │  │
│                     └─────────────┘    └─────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Agent vs 传统LLM

| 特性 | 传统LLM | Agent |
|------|---------|-------|
| 交互方式 | 单次问答 | 多轮交互 |
| 工具使用 | 无 | 可调用外部工具 |
| 记忆能力 | 无状态 | 有状态（短期/长期记忆） |
| 推理能力 | 单次推理 | 多步推理、规划 |
| 自主性 | 被动响应 | 主动决策 |

### 1.3 Agent的核心能力

1. **规划（Planning）**：将复杂任务分解为子任务
2. **记忆（Memory）**：存储和检索信息
3. **工具使用（Tool Use）**：调用外部API或函数
4. **推理（Reasoning）**：逻辑推理和决策
5. **反思（Reflection）**：自我评估和改进

---

## 2. ReAct架构

### 2.1 ReAct概述

**ReAct（Reasoning + Acting）** 将推理和行动交错进行，让Agent能够进行多步推理。

**核心思想：**
- **Thought（思考）**：分析问题，制定计划
- **Action（行动）**：执行工具调用
- **Observation（观察）**：获取工具返回结果

### 2.2 ReAct流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Thought: 我需要搜索相关信息来回答这个问题                  │
│ Action: search[问题关键词]                               │
│ Observation: [搜索结果]                                  │
│                                                         │
│ Thought: 根据搜索结果，我需要进一步分析...                 │
│ Action: calculate[计算公式]                              │
│ Observation: [计算结果]                                  │
│                                                         │
│ Thought: 现在我有足够的信息来回答用户                     │
│ Action: finish[最终答案]                                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
最终输出
```

### 2.3 ReAct实现

```python
class ReActAgent:
    """ReAct Agent实现"""
    
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        
        self.system_prompt = """You are a helpful AI assistant that can use tools to solve problems.

You should follow this format:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]

After receiving the observation:
Thought: [your reasoning based on the observation]
...

When you have the final answer:
Thought: I have enough information to answer
Final Answer: [your answer]

Available tools:
{tools_description}
"""
    
    def get_tools_description(self):
        """获取工具描述"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def parse_action(self, text):
        """解析Action和Action Input"""
        action_match = re.search(r'Action:\s*(\w+)', text)
        action_input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', text, re.DOTALL)
        
        if action_match:
            action = action_match.group(1)
            action_input = action_input_match.group(1).strip() if action_input_match else ""
            return action, action_input
        return None, None
    
    def run(self, query):
        """运行Agent"""
        # 初始化对话历史
        messages = [
            {"role": "system", "content": self.system_prompt.format(
                tools_description=self.get_tools_description()
            )},
            {"role": "user", "content": f"Question: {query}"}
        ]
        
        for i in range(self.max_iterations):
            # 调用LLM
            response = self.llm.chat(messages)
            content = response.choices[0].message.content
            
            print(f"\n=== Step {i+1} ===")
            print(content)
            
            # 检查是否得到最终答案
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[1].strip()
                return final_answer
            
            # 解析Action
            action, action_input = self.parse_action(content)
            
            if action and action in self.tools:
                # 执行工具
                tool = self.tools[action]
                observation = tool.run(action_input)
                
                print(f"\nObservation: {observation}")
                
                # 添加Observation到对话
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # 没有Action，继续思考
                messages.append({"role": "assistant", "content": content})
        
        return "Max iterations reached without final answer."

# 使用示例
class SearchTool:
    name = "search"
    description = "Search for information on the internet"
    
    def run(self, query):
        # 实际实现会调用搜索引擎API
        return f"Search results for '{query}': ..."

class CalculatorTool:
    name = "calculate"
    description = "Perform mathematical calculations"
    
    def run(self, expression):
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Error in calculation"

# 创建Agent
agent = ReActAgent(
    llm=OpenAI(),
    tools=[SearchTool(), CalculatorTool()],
    max_iterations=5
)

# 运行
result = agent.run("What is the population of Tokyo divided by the population of New York?")
print(result)
```

---

## 3. Plan-and-Execute架构

### 3.1 Plan-and-Execute概述

**Plan-and-Execute** 将任务分解为两个阶段：
1. **Planning（规划）**：制定执行计划
2. **Execution（执行）**：按计划逐步执行

### 3.2 与ReAct的区别

| 特性 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| 规划方式 | 边做边规划 | 先规划后执行 |
| 适应性 | 高（可动态调整） | 中（可重新规划） |
| 适用场景 | 探索性任务 | 结构化任务 |
| 效率 | 可能冗余 | 更高效 |

### 3.3 Plan-and-Execute实现

```python
class PlanAndExecuteAgent:
    """Plan-and-Execute Agent实现"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    def create_plan(self, task):
        """创建执行计划"""
        planning_prompt = f"""You are a planning assistant. Create a step-by-step plan to complete the following task.

Task: {task}

Create a plan with clear, actionable steps. Each step should be a single action.

Plan:
1. """
        
        response = self.llm.complete(planning_prompt)
        plan_text = "1. " + response
        
        # 解析计划步骤
        steps = []
        for line in plan_text.strip().split('\n'):
            if line.strip() and line[0].isdigit():
                step = line.split('.', 1)[1].strip()
                steps.append(step)
        
        return steps
    
    def execute_step(self, step, context):
        """执行单个步骤"""
        execution_prompt = f"""Execute the following step using available tools if needed.

Context so far:
{context}

Current step: {step}

Available tools: {', '.join(self.tools.keys())}

If you need to use a tool, respond with:
Tool: [tool_name]
Input: [tool_input]

If you can complete the step without tools, respond with:
Result: [your result]
"""
        
        response = self.llm.complete(execution_prompt)
        
        # 解析是否使用工具
        if "Tool:" in response:
            tool_name = response.split("Tool:")[1].split("\n")[0].strip()
            tool_input = response.split("Input:")[1].strip() if "Input:" in response else ""
            
            if tool_name in self.tools:
                result = self.tools[tool_name].run(tool_input)
                return f"Used {tool_name}: {result}"
        
        if "Result:" in response:
            return response.split("Result:")[1].strip()
        
        return response
    
    def run(self, task):
        """运行Agent"""
        print(f"Task: {task}\n")
        
        # 阶段1: 规划
        print("=== Planning ===")
        plan = self.create_plan(task)
        for i, step in enumerate(plan, 1):
            print(f"{i}. {step}")
        
        # 阶段2: 执行
        print("\n=== Execution ===")
        context = []
        
        for i, step in enumerate(plan, 1):
            print(f"\nStep {i}: {step}")
            result = self.execute_step(step, "\n".join(context))
            print(f"Result: {result}")
            context.append(f"Step {i}: {step}\nResult: {result}")
        
        # 生成最终答案
        final_prompt = f"""Based on the execution results, provide a final answer to the original task.

Task: {task}

Execution results:
{'\n'.join(context)}

Final Answer:"""
        
        final_answer = self.llm.complete(final_prompt)
        return final_answer
```

---

## 4. 反射与自我改进

### 4.1 反射机制

**反射（Reflection）** 让Agent能够评估自己的表现并改进。

```python
class ReflectiveAgent:
    """带反射机制的Agent"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.reflection_history = []
    
    def reflect(self, task, actions, result):
        """反思执行过程"""
        reflection_prompt = f"""Reflect on the following task execution:

Task: {task}

Actions taken:
{actions}

Result: {result}

Please analyze:
1. What went well?
2. What could be improved?
3. What would you do differently next time?

Reflection:"""
        
        reflection = self.llm.complete(reflection_prompt)
        self.reflection_history.append({
            "task": task,
            "reflection": reflection
        })
        
        return reflection
    
    def run_with_reflection(self, task):
        """带反射的执行"""
        # 标准执行
        actions = []
        result = self.execute_task(task, actions)
        
        # 反思
        reflection = self.reflect(task, "\n".join(actions), result)
        print(f"\n=== Reflection ===\n{reflection}")
        
        # 如果反思建议改进，可以重新执行
        if "improve" in reflection.lower() or "better" in reflection.lower():
            print("\n=== Re-executing with improvements ===")
            improved_result = self.execute_task(task, [], reflection)
            return improved_result
        
        return result
```

### 4.2 自我改进循环

```
        ┌─────────────────────────────────────────┐
        │              自我改进循环                │
        └─────────────────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────────┐
        │  1. 执行任务                            │
        │     (Execute Task)                      │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  2. 评估结果                            │
        │     (Evaluate Result)                   │
        └──────────────────┬──────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │ 成功     │              │ 失败/可改进 │
        └────┬─────┘              └────┬─────┘
             │                         │
             ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │ 返回结果  │              │ 分析原因  │
        └──────────┘              └────┬─────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  3. 学习/调整策略 │
                              │  (Learn/Adjust)  │
                              └───────┬────────┘
                                      │
                                      └────────→ 回到步骤1
```

---

## 5. 多步推理与工具使用

### 5.1 链式思考（Chain-of-Thought）

```python
class ChainOfThoughtAgent:
    """链式思考Agent"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def solve(self, problem):
        """使用CoT解决问题"""
        cot_prompt = f"""Solve the following problem step by step:

Problem: {problem}

Let's think through this step by step:
"""
        
        # 生成思考过程
        reasoning = self.llm.complete(cot_prompt)
        
        # 提取最终答案
        answer_prompt = f"""Based on the following reasoning, provide the final answer:

Reasoning:
{reasoning}

Final Answer:"""
        
        answer = self.llm.complete(answer_prompt)
        return answer, reasoning
```

### 5.2 工具选择与组合

```python
class ToolSelectionAgent:
    """智能工具选择Agent"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def select_tools(self, task):
        """选择需要的工具"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        selection_prompt = f"""Given the task and available tools, select which tools are needed.

Task: {task}

Available tools:
{tool_descriptions}

Select tools (comma-separated):"""
        
        selected = self.llm.complete(selection_prompt)
        tool_names = [name.strip() for name in selected.split(",")]
        
        return [tool for tool in self.tools if tool.name in tool_names]
    
    def execute_with_tools(self, task):
        """使用选定的工具执行任务"""
        # 选择工具
        selected_tools = self.select_tools(task)
        print(f"Selected tools: {[t.name for t in selected_tools]}")
        
        # 创建子Agent使用选定的工具
        sub_agent = ReActAgent(self.llm, selected_tools)
        result = sub_agent.run(task)
        
        return result
```

---

## 本章小结

Agent架构设计是构建智能系统的核心：

1. **ReAct** 通过交错推理和行动实现多步问题解决
2. **Plan-and-Execute** 先规划后执行，适合结构化任务
3. **反射机制** 让Agent能够自我评估和改进
4. **工具使用** 扩展了Agent的能力边界

**下一章：** 我们将学习工具调用与Function Calling的具体实现。

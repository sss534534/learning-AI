# 第五章：工具调用与Function Calling

> 工具调用（Tool Calling）是Agent系统的核心能力，让LLM能够与外部世界交互。本章将深入讲解Function Calling的原理、实现、工具定义以及错误处理机制。

## 目录

1. [Function Calling概述](#1-function-calling概述)
2. [工具定义与Schema](#2-工具定义与schema)
3. [调用流程与实现](#3-调用流程与实现)
4. [错误处理与重试](#4-错误处理与重试)
5. [高级工具调用模式](#5-高级工具调用模式)

---

## 元数据
- **难度**: ⭐⭐
- **前置知识**: ../chapters/ch04-agent-architecture.md
- **关联文件**: ../chapters/ch06-memory-system.md, ../chapters/ch08-agent-frameworks.md
- **最后更新**: 2026-06-12
---

## 1. Function Calling概述

### 1.1 什么是Function Calling？

**Function Calling** 允许LLM生成结构化的函数调用请求，而不是直接生成文本回答。

**核心流程：**
```
用户输入
    │
    ▼
┌─────────────────────────────────────────┐
│  LLM分析需求                            │
│  决定是否需要调用工具                    │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
┌─────────────┐         ┌─────────────┐
│ 需要工具     │         │ 直接回答     │
└──────┬──────┘         └──────┬──────┘
       │                       │
       ▼                       ▼
┌─────────────┐         ┌─────────────┐
│ 生成工具调用  │         │ 返回文本答案 │
│ JSON格式     │         │             │
└──────┬──────┘         └─────────────┘
       │
       ▼
┌─────────────┐
│ 执行工具     │
│ 获取结果     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 结果返回LLM  │
│ 生成最终答案 │
└─────────────┘
```

### 1.2 主流模型的Function Calling支持

| 模型 | 支持情况 | 特点 |
|------|----------|------|
| **GPT-4** | 原生支持 | 稳定、准确 |
| **GPT-3.5** | 原生支持 | 速度快 |
| **Claude** | 原生支持 | 长上下文 |
| **LLaMA** | 需微调 | 开源 |
| **Qwen** | 原生支持 | 中文优化 |
| **GLM-4** | 原生支持 | 国产 |

---

## 2. 工具定义与Schema

### 2.1 OpenAI Function Calling格式

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search for information in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 2.2 工具基类设计

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ToolParameters(BaseModel):
    """工具参数基类"""
    pass

class ToolResult(BaseModel):
    """工具结果"""
    success: bool
    data: Any
    error: Optional[str] = None

class BaseTool(ABC):
    """工具基类"""
    
    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {}
    
    @abstractmethod
    def _run(self, **kwargs) -> ToolResult:
        """执行工具的具体逻辑"""
        pass
    
    def run(self, **kwargs) -> ToolResult:
        """运行工具（带错误处理）"""
        try:
            # 参数验证
            self._validate_parameters(kwargs)
            
            # 执行
            result = self._run(**kwargs)
            return result
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _validate_parameters(self, params: Dict[str, Any]):
        """验证参数"""
        required = self.parameters_schema.get("required", [])
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为OpenAI格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }

# 具体工具实现示例
class WeatherTool(BaseTool):
    """天气查询工具"""
    
    name = "get_weather"
    description = "Get current weather information for a location"
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g., 'Beijing' or 'New York'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
    
    def _run(self, location: str, unit: str = "celsius") -> ToolResult:
        """查询天气"""
        # 实际实现会调用天气API
        weather_data = {
            "location": location,
            "temperature": 25,
            "unit": unit,
            "condition": "sunny",
            "humidity": 60
        }
        
        return ToolResult(success=True, data=weather_data)

class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name = "calculate"
    description = "Perform mathematical calculations"
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
            }
        },
        "required": ["expression"]
    }
    
    def _run(self, expression: str) -> ToolResult:
        """计算表达式"""
        try:
            # 安全计算（使用eval的受限版本）
            allowed_names = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(success=True, data={"result": result})
            
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

---

## 3. 调用流程与实现

### 3.1 完整的Function Calling流程

```python
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

class FunctionCaller:
    """Function Calling实现"""
    
    def __init__(self, llm_client: OpenAI, tools: List[BaseTool]):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.to_openai_schema() for tool in tools]
    
    def call(self, messages: List[Dict[str, str]], max_iterations: int = 5) -> str:
        """
        执行Function Calling
        
        Args:
            messages: 对话历史
            max_iterations: 最大工具调用次数
            
        Returns:
            最终回答
        """
        for i in range(max_iterations):
            # 调用LLM
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if not message.tool_calls:
                # 没有工具调用，直接返回内容
                return message.content
            
            # 添加助手消息到历史
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 执行工具调用
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\n[Tool Call] {function_name}({function_args})")
                
                # 执行工具
                if function_name in self.tools:
                    tool = self.tools[function_name]
                    result = tool.run(**function_args)
                    
                    # 添加工具结果到历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.dict())
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Tool {function_name} not found"})
                    })
        
        # 达到最大迭代次数
        return "Max iterations reached."

# 使用示例
client = OpenAI()
tools = [WeatherTool(), CalculatorTool()]
caller = FunctionCaller(client, tools)

messages = [
    {"role": "user", "content": "What's the weather in Beijing and what is 15 * 23?"}
]

result = caller.call(messages)
print(f"\nFinal Answer: {result}")
```

### 3.2 并行工具调用

```python
class ParallelFunctionCaller(FunctionCaller):
    """支持并行工具调用的Function Caller"""
    
    def call(self, messages: List[Dict[str, str]], max_iterations: int = 5) -> str:
        for i in range(max_iterations):
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                parallel_tool_calls=True  # 启用并行调用
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                return message.content
            
            # 添加助手消息
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 并行执行所有工具调用
            import concurrent.futures
            
            def execute_tool(tool_call):
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in self.tools:
                    tool = self.tools[function_name]
                    result = tool.run(**function_args)
                    return {
                        "tool_call_id": tool_call.id,
                        "result": result
                    }
                else:
                    return {
                        "tool_call_id": tool_call.id,
                        "result": ToolResult(success=False, error=f"Tool {function_name} not found")
                    }
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(execute_tool, tc) for tc in message.tool_calls]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # 添加所有工具结果
            for result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps(result["result"].dict())
                })
        
        return "Max iterations reached."
```

---

## 4. 错误处理与重试

### 4.1 错误分类与处理

```python
from enum import Enum
from typing import Callable
import time

class ToolErrorType(Enum):
    """工具错误类型"""
    VALIDATION_ERROR = "validation_error"      # 参数验证错误
    EXECUTION_ERROR = "execution_error"        # 执行错误
    TIMEOUT_ERROR = "timeout_error"           # 超时错误
    NETWORK_ERROR = "network_error"           # 网络错误
    RATE_LIMIT_ERROR = "rate_limit_error"     # 限流错误
    UNKNOWN_ERROR = "unknown_error"           # 未知错误

class ToolError(Exception):
    """工具错误"""
    def __init__(self, error_type: ToolErrorType, message: str, recoverable: bool = True):
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """带退避的重试装饰器"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ToolError as e:
                    if not e.recoverable or attempt == max_retries - 1:
                        raise
                    
                    # 计算延迟
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    print(f"Attempt {attempt + 1} failed: {e.message}. Retrying in {delay}s...")
                    time.sleep(delay)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

class RobustTool(BaseTool):
    """带错误处理的工具基类"""
    
    @retry_with_backoff(max_retries=3)
    def run(self, **kwargs) -> ToolResult:
        """带重试的运行"""
        return super().run(**kwargs)
    
    def _handle_error(self, error: Exception) -> ToolResult:
        """错误处理"""
        if isinstance(error, ToolError):
            return ToolResult(
                success=False,
                error=f"[{error.error_type.value}] {error.message}"
            )
        else:
            return ToolResult(
                success=False,
                error=f"[UNEXPECTED_ERROR] {str(error)}"
            )
```

### 4.2 工具调用失败恢复

```python
class ResilientFunctionCaller(FunctionCaller):
    """弹性Function Caller，支持失败恢复"""
    
    def call_with_recovery(self, messages: List[Dict[str, str]], max_iterations: int = 5) -> str:
        """带恢复的调用"""
        failed_tools = []
        
        for i in range(max_iterations):
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                return message.content
            
            # 添加助手消息
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 执行工具调用
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in self.tools:
                    tool = self.tools[function_name]
                    result = tool.run(**function_args)
                    
                    # 如果工具失败，尝试修复
                    if not result.success:
                        print(f"Tool {function_name} failed: {result.error}")
                        
                        # 尝试修复参数
                        fixed_args = self._attempt_fix_args(function_name, function_args, result.error)
                        if fixed_args:
                            print(f"Retrying with fixed args: {fixed_args}")
                            result = tool.run(**fixed_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.dict())
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Tool {function_name} not found"})
                    })
        
        return "Max iterations reached."
    
    def _attempt_fix_args(self, tool_name: str, args: Dict, error: str) -> Optional[Dict]:
        """尝试修复参数"""
        # 使用LLM尝试修复参数
        fix_prompt = f"""The tool call failed with the following error:

Tool: {tool_name}
Arguments: {json.dumps(args)}
Error: {error}

Please provide corrected arguments in JSON format, or respond with "CANNOT_FIX" if unrecoverable.

Corrected arguments:"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": fix_prompt}]
        )
        
        content = response.choices[0].message.content.strip()
        
        if "CANNOT_FIX" in content:
            return None
        
        try:
            # 尝试解析JSON
            fixed = json.loads(content)
            return fixed
        except:
            return None
```

---

## 5. 高级工具调用模式

### 5.1 工具链（Tool Chaining）

```python
class ToolChain:
    """工具链 - 按顺序执行多个工具"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
    
    def execute(self, initial_input: Dict[str, Any]) -> ToolResult:
        """执行工具链"""
        current_data = initial_input
        
        for i, tool in enumerate(self.tools):
            print(f"[Tool Chain] Step {i+1}/{len(self.tools)}: {tool.name}")
            
            # 将上一步的输出作为下一步的输入
            result = tool.run(**current_data)
            
            if not result.success:
                return ToolResult(
                    success=False,
                    error=f"Tool chain failed at {tool.name}: {result.error}"
                )
            
            # 更新数据供下一步使用
            current_data = result.data if isinstance(result.data, dict) else {"input": result.data}
        
        return ToolResult(success=True, data=current_data)

# 使用示例：数据处理链
# chain = ToolChain([FetchDataTool(), CleanDataTool(), AnalyzeDataTool()])
# result = chain.execute({"url": "https://example.com/data.csv"})
```

### 5.2 条件工具调用

```python
class ConditionalTool(BaseTool):
    """条件工具 - 根据条件选择执行哪个工具"""
    
    name = "conditional_executor"
    description = "Execute different tools based on conditions"
    
    def __init__(self, conditions: List[tuple]):
        """
        conditions: List of (condition_func, tool) tuples
        """
        self.conditions = conditions
    
    def _run(self, **kwargs) -> ToolResult:
        """执行条件判断并调用相应工具"""
        for condition_func, tool in self.conditions:
            if condition_func(kwargs):
                return tool.run(**kwargs)
        
        return ToolResult(success=False, error="No condition matched")

# 使用示例
# def is_weather_query(params):
#     return "weather" in params.get("query", "").lower()
#
# def is_math_query(params):
#     return any(op in params.get("query", "") for op in ["+", "-", "*", "/"])
#
# conditional_tool = ConditionalTool([
#     (is_weather_query, WeatherTool()),
#     (is_math_query, CalculatorTool())
# ])
```

### 5.3 工具组合（Tool Composition）

```python
class ComposedTool(BaseTool):
    """组合工具 - 将多个工具组合成一个"""
    
    def __init__(self, name: str, description: str, tools: List[BaseTool], composer_func: Callable):
        self.name = name
        self.description = description
        self.tools = tools
        self.composer_func = composer_func
    
    def _run(self, **kwargs) -> ToolResult:
        """执行组合"""
        # 并行执行所有工具
        results = []
        for tool in self.tools:
            result = tool.run(**kwargs)
            results.append(result)
        
        # 使用组合函数合并结果
        combined = self.composer_func(results)
        
        return ToolResult(success=True, data=combined)

# 使用示例：聚合多个搜索结果
# def aggregate_search_results(results):
#     all_items = []
#     for r in results:
#         if r.success:
#             all_items.extend(r.data.get("items", []))
#     return {"items": sorted(all_items, key=lambda x: x["relevance"], reverse=True)}
#
# search_aggregator = ComposedTool(
#     name="aggregated_search",
#     description="Search across multiple sources",
#     tools=[GoogleSearchTool(), BingSearchTool(), DuckDuckGoTool()],
#     composer_func=aggregate_search_results
# )
```

---

## 深度分析

工具调用（Function Calling/Tool Calling）是Agent与外部世界交互的核心桥梁。从技术实现角度看，其本质是将LLM的文本生成能力与结构化API调用相结合——模型不是直接输出最终答案，而是输出JSON格式的函数调用请求，由外部系统执行后返回结果。这种方式的关键优势在于：利用LLM的语义理解能力来解析用户意图并选择正确的工具；通过结构化参数传递保证与外部系统的精确对接；支持多次迭代调用实现复杂的工作流编排。Function Calling的标准化（如OpenAI的tool schema格式）极大地降低了不同模型和框架之间的集成成本，成为现代Agent系统的基石能力之一。

在实际工程中，工具调用面临若干关键挑战。首先是工具选择的准确性——当系统中注册的工具数量增长到数十甚至上百个时，LLM可能选择错误的工具或生成不符合Schema的参数。针对这一问题，常见的优化手段包括：提供高质量的description描述、为参数设置严格的enum约束、以及引入工具选择专用的精调模型。其次是错误恢复机制——工具调用失败（如API超时、参数类型错误）需要通过重试、参数修复或降级策略来保证系统的鲁棒性。最后是安全性问题，尤其是在执行工具链或代码生成场景中，需要严格的沙箱隔离与权限控制，防止恶意代码执行或数据泄露。

随着模型能力的提升，工具调用的范式也在持续演进。GPT-4的并行工具调用（Parallel Tool Calling）允许一次生成多个工具调用请求并并发执行，大幅提升了效率。Claude的Computer Use能力将工具调用的边界从API调用扩展到了GUI操作层面。未来，工具调用将朝着更自然的交互方向发展——模型不仅能够调用预定义工具，还能根据任务描述动态生成新的工具组合，实现真正的"工具即服务"（Tool-as-a-Service）范式。

---

## Checklist

- [ ] 理解Function Calling的核心流程：意图识别→参数生成→工具执行→结果注入
- [ ] 掌握OpenAI格式的工具Schema定义，包括name、description、parameters等字段
- [ ] 实现基类BaseTool，包含参数验证、错误处理和Schema导出
- [ ] 实现完整的Function Calling循环，包括消息历史维护和多轮工具调用
- [ ] 实现并行工具调用（Parallel Tool Calling）以提升多工具场景的效率
- [ ] 为工具调用添加重试机制，支持指数退避（Exponential Backoff）
- [ ] 实现工具调用失败后的参数自动修复逻辑
- [ ] 理解和实现工具链（Tool Chaining）、条件调用、工具组合等高级模式
- [ ] 处理tool_choice参数的auto/required/none等不同策略
- [ ] 在实际Agent项目中注册并测试至少3种不同类型的工具

---

## 延伸阅读

- [第六章：Agent记忆系统](../chapters/ch06-memory-system.md) - 记忆系统与工具调用的协同工作
- [第八章：Agent框架与实践](../chapters/ch08-agent-frameworks.md) - 主流Agent框架的工具调用实现对比
- OpenAI Function Calling官方文档 - https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use指南 - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- LangChain Tools文档 - https://python.langchain.com/docs/modules/agents/tools/

---

## 本章小结

工具调用是Agent系统的核心能力：

1. **Function Calling** 让LLM能够生成结构化工具调用
2. **工具Schema** 定义了工具的接口规范
3. **错误处理** 机制确保系统的健壮性
4. **高级模式**（工具链、条件调用、组合）扩展了工具使用能力

**下一章：** 我们将学习记忆系统的设计与实现。

---

*最后更新: 2026-06-12*

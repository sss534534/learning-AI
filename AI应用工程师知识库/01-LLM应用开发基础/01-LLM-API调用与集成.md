# LLM API调用与集成

> 大模型API接入的完整开发指南

## 元数据
- **难度**: ⭐⭐
- **前置知识**: 无
- **关联文件**: `../01-LLM应用开发基础/02-Prompt工程实战指南.md` | `../../AI架构师知识库/01-大模型基础概念.md`
- **最后更新**: 2026-06-12
---

## 1. 主流LLM API概览

### 1.1 API服务商对比

| 服务商 | 模型 | 特点 | 定价 |
|--------|------|------|------|
| **OpenAI** | GPT-4o/4/3.5 | 生态最成熟、多模态 | 按Token计费 |
| **Anthropic** | Claude-3.5/3 | 长上下文、安全性高 | 按Token计费 |
| **Google** | Gemini-1.5 Pro/Flash | 超长上下文(1M) | 按Token计费 |
| **智谱AI** | GLM-4 | 中文优化、Function Call | 按Token计费 |
| **阿里云** | Qwen-Max/Plus/Turbo | 全尺寸覆盖、开源可部署 | 按Token计费 |
| **百度** | ERNIE-4.0/3.5 | 中文理解强 | 按Token计费 |
| **DeepSeek** | DeepSeek-V2/V3 | 性价比极高 | 按Token计费 |
| **月之暗面** | Moonshot/Kimi | 超长上下文 | 按Token计费 |

### 1.2 API通用概念

**核心参数：**

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `model` | 模型标识 | gpt-4o, claude-3-5-sonnet |
| `messages` | 对话消息列表 | role + content |
| `temperature` | 随机性控制 | 0.0-2.0, 推荐0.7 |
| `max_tokens` | 最大生成Token数 | 4096 |
| `top_p` | 核采样 | 0.9 |
| `stream` | 是否流式输出 | true/false |
| `stop` | 停止序列 | ["\n", "用户:"] |

**Token计费模型：**
```
费用 = (输入Token数 × 输入单价) + (输出Token数 × 输出单价)

示例（GPT-4o）：
- 输入：$5.00 / 1M tokens
- 输出：$15.00 / 1M tokens

1000字中文 ≈ 1500-2000 tokens
```

---

## 2. OpenAI API开发

### 2.1 基础调用

**Python SDK：**
```python
from openai import OpenAI

client = OpenAI(api_key="sk-xxx")

# 基础对话
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是RAG？"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

**流式输出（SSE）：**
```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 2.2 Function Calling

**定义工具：**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**处理工具调用：**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 检查是否需要调用工具
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # 执行本地函数
        result = get_weather(**function_args)
        
        # 将结果返回给模型
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
        
        # 继续生成最终回答
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
```

### 2.3 多模态调用

**图片理解：**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "low"  # low/high/auto
                    }
                }
            ]
        }
    ]
)
```

**结构化输出（JSON Mode）：**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "分析这段文本的情感"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"},
                    "reasons": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["sentiment", "confidence"]
            }
        }
    }
)
```

---

## 3. Java生态集成

### 3.1 Spring AI

**核心概念：**
```
Spring AI = AI能力的Spring抽象层
├── ChatClient: 对话客户端
├── EmbeddingClient: 向量化客户端
├── VectorStore: 向量存储
├── ToolCallback: 工具回调
└── Advisor: 请求/响应拦截器
```

**Maven依赖：**
```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

**配置：**
```yaml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4o
          temperature: 0.7
```

**基础使用：**
```java
@RestController
public class ChatController {
    
    private final ChatClient chatClient;
    
    public ChatController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }
    
    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }
    
    // 流式输出
    @GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chatStream(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .stream()
            .content();
    }
}
```

**System Prompt配置：**
```java
ChatClient chatClient = ChatClient.builder(chatModel)
    .defaultSystem("你是一个专业的Java架构师助手")
    .defaultAdvisors(new MessageChatMemoryAdvisor(chatMemory))
    .build();
```

### 3.2 LangChain4j

**Maven依赖：**
```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-open-ai</artifactId>
    <version>0.35.0</version>
</dependency>
```

**声明式AI Service：**
```java
// 定义AI服务接口
interface Assistant {
    
    @SystemMessage("你是一个专业的客服助手")
    String chat(@UserMessage String message);
    
    @SystemMessage("提取以下文本中的实体")
    List<Entity> extract(@UserMessage String text);
}

// 使用
Assistant assistant = AiServices.create(Assistant.class, chatModel);
String response = assistant.chat("你好");
```

**工具集成：**
```java
public class CalculatorTools {
    
    @Tool("计算两个数的和")
    public double add(@P("第一个数") double a, @P("第二个数") double b) {
        return a + b;
    }
}

// 注册工具
Assistant assistant = AiServices.builder(Assistant.class)
    .chatLanguageModel(chatModel)
    .tools(new CalculatorTools())
    .build();
```

**RAG集成：**
```java
// 文档加载
DocumentLoader loader = new FileSystemDocumentLoader();
List<Document> documents = loader.loadDocuments("docs/");

// 文本分割
TextSplitter splitter = DocumentSplitters.recursive(500, 50);
List<TextSegment> segments = splitter.splitAll(documents);

// 向量化存储
EmbeddingModel embeddingModel = OpenAiEmbeddingModel.withApiKey("sk-xxx");
EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
EmbeddingStoreIngestor.ingest(segments, embeddingModel, store);

// RAG查询
ContentRetriever retriever = EmbeddingStoreContentRetriever.from(store, embeddingModel);
Assistant assistant = AiServices.builder(Assistant.class)
    .chatLanguageModel(chatModel)
    .contentRetriever(retriever)
    .build();
```

---

## 4. 对话管理

### 4.1 多轮对话实现

**核心：维护消息历史**

```python
class ConversationManager:
    def __init__(self, client, system_prompt=""):
        self.client = client
        self.system_prompt = system_prompt
        self.messages = []
        
    def chat(self, user_message):
        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 构建完整消息列表
        full_messages = []
        if self.system_prompt:
            full_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        full_messages.extend(self.messages)
        
        # 调用API
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=full_messages
        )
        
        # 保存助手回复
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
```

### 4.2 上下文窗口管理

**问题：** 对话过长超出模型上下文限制

**策略：**

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **滑动窗口** | 保留最近N轮 | 简单对话 |
| **摘要压缩** | 定期摘要历史 | 长对话 |
| **Token计数** | 按Token数裁剪 | 精确控制 |
| **分层记忆** | 短期+长期 | 复杂对话 |

**Token计数实现：**
```python
import tiktoken

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def trim_messages(messages, max_tokens=120000):
    """保留system消息，裁剪历史"""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    
    total = sum(count_tokens(m["content"]) for m in history)
    while total > max_tokens and history:
        removed = history.pop(0)
        total -= count_tokens(removed["content"])
    
    return system + history
```

### 4.3 会话持久化

**存储方案：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Redis** | 快速、支持TTL | 内存成本 |
| **数据库** | 持久化、可查询 | 速度较慢 |
| **文件** | 简单 | 不适合并发 |

**Redis实现：**
```python
import redis
import json

class RedisConversationStore:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
    
    def save(self, session_id, messages, ttl=86400):
        key = f"conversation:{session_id}"
        self.redis.setex(key, ttl, json.dumps(messages))
    
    def load(self, session_id):
        key = f"conversation:{session_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else []
    
    def append(self, session_id, message):
        messages = self.load(session_id)
        messages.append(message)
        self.save(session_id, messages)
```

---

## 5. 错误处理与重试

### 5.1 常见错误类型

| 错误 | HTTP状态码 | 原因 | 处理 |
|------|------------|------|------|
| Rate Limit | 429 | 请求频率过高 | 指数退避重试 |
| Invalid Request | 400 | 参数错误 | 检查请求 |
| Auth Error | 401 | API Key无效 | 检查密钥 |
| Server Error | 500/503 | 服务端故障 | 重试 |
| Timeout | - | 请求超时 | 重试+降级 |
| Context Length | 400 | 超出上下文 | 裁剪消息 |

### 5.2 重试策略

**指数退避重试：**
```python
import time
import random

def call_with_retry(func, max_retries=5, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # 计算退避时间
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            
            # 处理Rate Limit的retry-after
            if hasattr(e, 'response') and e.response.status_code == 429:
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    delay = float(retry_after)
            
            print(f"Retry {attempt + 1}/{max_retries}, wait {delay:.1f}s")
            time.sleep(delay)
```

### 5.3 降级策略

```python
class LLMClientWithFallback:
    def __init__(self):
        self.primary = OpenAI(api_key="sk-primary")
        self.fallback = OpenAI(
            api_key="sk-fallback",
            base_url="https://api.deepseek.com/v1"
        )
        self.local_model = None  # 本地模型备选
    
    def chat(self, messages, **kwargs):
        try:
            return self._call(self.primary, messages, **kwargs)
        except Exception:
            try:
                return self._call(self.fallback, messages, **kwargs)
            except Exception:
                return self._call_local(messages, **kwargs)
    
    def _call(self, client, messages, **kwargs):
        return client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o"),
            messages=messages,
            **kwargs
        )
```

---

## 6. 成本控制

### 6.1 Token用量追踪

```python
class TokenTracker:
    def __init__(self):
        self.usage = {}  # {model: {input: n, output: n}}
    
    def track(self, model, response):
        if model not in self.usage:
            self.usage[model] = {"input": 0, "output": 0}
        
        self.usage[model]["input"] += response.usage.prompt_tokens
        self.usage[model]["output"] += response.usage.completion_tokens
    
    def report(self):
        total_cost = 0
        for model, tokens in self.usage.items():
            cost = self._calculate_cost(model, tokens)
            total_cost += cost
            print(f"{model}: 输入{tokens['input']}, 输出{tokens['output']}, 费用${cost:.4f}")
        print(f"总费用: ${total_cost:.4f}")
```

### 6.2 成本优化策略

| 策略 | 节省幅度 | 实现难度 |
|------|----------|----------|
| **模型路由** | 30-70% | 中 |
| **Prompt精简** | 10-30% | 低 |
| **缓存重复查询** | 10-50% | 中 |
| **本地小模型** | 50-90% | 高 |
| **批处理** | 20-50% | 低 |

**模型路由示例：**
```python
class ModelRouter:
    def __init__(self):
        self.simple_model = "gpt-4o-mini"  # 便宜
        self.complex_model = "gpt-4o"       # 贵
    
    def route(self, messages):
        """根据复杂度选择模型"""
        complexity = self._estimate_complexity(messages)
        
        if complexity == "simple":
            return self.simple_model
        elif complexity == "medium":
            return self.simple_model  # 或中等模型
        else:
            return self.complex_model
    
    def _estimate_complexity(self, messages):
        last_message = messages[-1]["content"]
        
        # 简单判断规则
        if len(last_message) < 50:
            return "simple"
        elif any(kw in last_message for kw in ["分析", "对比", "详细"]):
            return "complex"
        return "medium"
```

---

## 7. 安全最佳实践

### 7.1 API Key管理

```
✅ 正确做法：
- 使用环境变量存储API Key
- 使用密钥管理服务（Vault/AWS Secrets Manager）
- 不同环境使用不同Key
- 定期轮换Key

❌ 错误做法：
- 硬编码在代码中
- 提交到Git仓库
- 前端直接调用（暴露Key）
- 所有服务共用一个Key
```

### 7.2 输入安全

```python
import re

def sanitize_input(text):
    """清洗用户输入"""
    # 移除Prompt注入尝试
    patterns = [
        r"ignore previous instructions",
        r"ignore all above",
        r"system prompt",
        r"you are now",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # 记录可疑行为
            log_suspicious_activity(text)
            # 截断可疑部分
            text = text.split(pattern, 1)[0]
    
    # 限制长度
    max_length = 4000
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()
```

### 7.3 输出安全

```python
def filter_output(text):
    """过滤敏感输出"""
    # PII检测（简单版）
    patterns = {
        "phone": r"\d{11}",
        "id_card": r"\d{17}[\dXx]",
        "email": r"[\w.-]+@[\w.-]+\.\w+",
    }
    
    for name, pattern in patterns.items():
        text = re.sub(pattern, f"[{name}_masked]", text)
    
    return text
```

---

## 8. 开发者Checklist

### 8.1 API集成Checklist

- [ ] 选择合适的API服务商和模型
- [ ] 实现API Key安全管理
- [ ] 实现基础对话功能
- [ ] 实现流式输出
- [ ] 实现多轮对话管理
- [ ] 实现上下文窗口管理
- [ ] 实现错误处理和重试
- [ ] 实现降级策略
- [ ] 实现Token用量追踪
- [ ] 实现输入输出安全过滤
- [ ] 实现成本优化策略
- [ ] 编写单元测试和集成测试

### 8.2 常见陷阱

**陷阱1：忽视流式输出**
- 问题：用户等待时间长，体验差
- 解决：所有生成场景都应支持SSE流式

**陷阱2：上下文无限增长**
- 问题：长对话导致Token爆炸、成本失控
- 解决：实现上下文窗口管理策略

**陷阱3：错误处理不足**
- 问题：API故障导致应用崩溃
- 解决：重试+降级+优雅降级

**陷阱4：API Key泄露**
- 问题：Key提交到代码仓库
- 解决：环境变量+密钥管理服务

## 深度分析

LLM API 调用是构建 AI 应用的基础能力，但生产环境远比简单调用复杂。开发者需要建立完善的错误处理体系——指数退避重试应对限流、多模型降级应对服务故障、熔断机制防止级联失败。实践中最容易被忽视的是流式输出的全链路支持：从 SDK 的 stream 参数到后端的 SSE 响应，再到前端的逐 chunk 渲染，任一环节缺失都会导致用户体验降级。

对话管理是另一个核心挑战。Token 窗口管理策略的选择直接影响体验和成本：滑动窗口适合简单场景，摘要压缩适合长对话，分层记忆则适合需要长期知识沉淀的应用。结合 Redis 实现会话持久化时，须注意 TTL 策略和序列化效率的平衡。此外，安全防护不可缺位——Prompt 注入检测、API Key 的密钥管理服务托管、输出层的 PII 脱敏，是生产部署必须解决的三个基本安全课题。

## Checklist

- [ ] 实现完整的错误处理体系（指数退避重试 + 多模型降级 + 熔断机制）
- [ ] 部署全链路流式输出（SSE 协议，前后端配合）
- [ ] 配置 Token 用量追踪及成本告警阈值
- [ ] 实现对话上下文窗口管理策略（滑动窗口 / 摘要压缩 / 分层记忆）
- [ ] 实现模型路由（简单任务用小模型，复杂任务用大模型）
- [ ] 使用密钥管理服务（Vault / AWS Secrets Manager）管理 API Key
- [ ] 实现输入注入检测和输出 PII 脱敏过滤
- [ ] 添加请求粒度的日志追踪和性能指标采集
- [ ] 配置限流策略（令牌桶 / 漏斗算法）保护后端 LLM 服务
- [ ] 编写 API 集成测试 + 异常场景覆盖测试

## 延伸阅读

- `../01-LLM应用开发基础/02-Prompt工程实战指南.md` — Prompt 质量直接影响 API 输出效果，掌握设计技巧可事半功倍
- `../../AI架构师知识库/01-大模型基础概念.md` — 深入理解 Token、上下文窗口、注意力机制等核心概念
- `../04-AI应用架构设计/01-AI应用架构设计.md` — 了解 LLM API 在生产级 AI 系统架构中的完整位置
- `../../AI架构师知识库/02-模型选型与评估框架.md` — 系统化的模型评估方法指导服务商与模型选型

---

*最后更新：2026-06-12*

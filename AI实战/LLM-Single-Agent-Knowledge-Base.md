# LLM 单 Agent 开发知识库

> 方法论 + 五层架构 + 最佳实践 + 反模式 + 检查清单

---

## 核心认知

**单 Agent 是一切 Multi-Agent 的基础。单 Agent 做不好的，Multi-Agent 只会放大错误。**

一个优秀的单 Agent 系统由五个层次组成：运行时循环 → 工具层 → 提示层 → 记忆层 → 可观测层。这五层都做对了，才谈得上 Multi-Agent。

---

## 第一部分：五层架构

### L1 — 运行时循环 (Agent Loop)

运行时循环是 Agent 的心跳。目前最主流的是 **ReAct 模式**。

#### ReAct 模式

```
用户输入
  -> Thought:  我应该做什么？需要什么信息？
  -> Action:   调用工具获取信息
  -> Observation: 查看工具返回结果
  -> Thought:  信息够了吗？下一步是什么？
  -> Action:   继续调用工具...
  -> ...
  -> Thought:  信息够了，可以回答用户了
  -> Final Answer: 给用户的最终输出
```

**终止条件（必须全设）：**
- `max_steps`：最大步数硬限制（推荐 10-20）
- `finish_reason`：模型主动输出终止信号
- `timeout`：总时间限制（推荐 60-120s）

#### 变体模式

| 模式 | 流程 | 适用场景 |
|------|------|----------|
| **Plan-then-Execute** | 先规划完整步骤，再逐步执行 | 任务可预分解（如代码生成） |
| **Tool-calling Loop** | 没有显式 Thought，工具调用驱动 | API 编排、数据查询 |
| **Chain-of-Thought** | 多步推理但不调用工具 | 数学/逻辑推理 |
| **Reflection** | 生成 → 自我审查 → 修正 → 输出 | 高质量写作、代码审查 |

#### 最小化实现骨架

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentConfig:
    max_steps: int = 15
    timeout: int = 120
    model: str = "gpt-4o"

@dataclass
class AgentState:
    steps: list = field(default_factory=list)
    tools_called: list = field(default_factory=list)
    total_tokens: int = 0

async def agent_loop(user_input: str, config: AgentConfig) -> str:
    state = AgentState()
    messages = build_system_prompt() + [{"role": "user", "content": user_input}]

    for step in range(config.max_steps):
        response = await call_llm(messages, config.model)
        state.steps.append(response)
        state.total_tokens += response.usage.total_tokens

        if response.has_tool_calls():
            for tc in response.tool_calls:
                result = await execute_tool(tc)
                state.tools_called.append(tc.name)
                messages.append(tool_result_message(tc, result))
        else:
            return response.content  # 最终答案

    raise MaxStepsExceeded(f"Agent exceeded {config.max_steps} steps")
```

---

### L2 — 工具设计 (Tool Design)

工具设计决定 Agent 的能力边界。好工具让 Agent 如虎添翼，烂工具让它寸步难行。

#### 工具设计六原则

**1. 单一职责**

每个工具只做一件事，名如其责。

```
✅ search_documents(query: str) -> List[Document]
✅ create_file(path: str, content: str) -> bool
❌ do_everything(action: str, **kwargs) -> Any  # 太大了，LLM 不会用
```

**2. 命名即文档**

函数名和参数名本身就是 prompt。不要用缩写。

```
✅ get_user_by_email(email: str) -> User
❌ get_usr(em: str) -> dict
```

**3. 参数类型严格**

使用 type hints / JSON Schema 约束参数，不要用 `**kwargs` 或 `Any`。

```python
# ✅ 严格类型
def search_alerts(
    severity: Literal["P0", "P1", "P2"],
    zone: str,
    limit: int = 20,
    start_time: datetime | None = None,
) -> list[Alert]:
    ...

# ❌ 模糊类型
def search_alerts(filters: dict) -> list:
    ...
```

**4. 错误信息可诊断**

返回有意义的错误，让 LLM 能据此调整行为。

```
✅ return {"error": "zone 'tokyo' not found. Available zones: us-east, eu-west, ap-south"}
❌ return {"error": "not found"}
❌ raise Exception("something went wrong")  # LLM 看不到 traceback
```

**5. 幂等优先**

能重复调用的工具比只能调一次的工具安全 100 倍。

```
✅ create_if_not_exists(...)  # 幂等
✅ upsert(...)                # 幂等
⚠️ create(...)                # 非幂等，需要额外处理
```

**6. 权限分级**

```python
class ToolPermission(Enum):
    READ_ONLY = "read_only"    # 搜索、查询、读取
    WRITE = "write"            # 创建、更新
    DANGEROUS = "dangerous"    # 删除、发送消息、外部 API

# 根据用户信任级别决定暴露哪些工具
```

#### 工具描述规范

LLM 通过工具的 description 字段理解工具。每个工具描述必须包含：

```python
{
    "name": "search_alerts",
    "description": (
        "Search for active alerts in the monitoring system. "
        "Use this when the user asks about current incidents, "
        "outages, or alerts. Returns up to 50 matching alerts "
        "sorted by severity (P0 first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "severity": {
                "type": "string",
                "enum": ["P0", "P1", "P2"],
                "description": "Filter by alert severity. P0=critical."
            },
            "zone": {
                "type": "string",
                "description": "Infrastructure zone, e.g. 'us-east-1'."
            }
        },
        "required": ["severity"]
    }
}
```

**description 四要素**：做什么 + 什么时候用 + 返回什么 + 注意事项。

---

### L3 — 提示工程 (Prompt Engineering)

System Prompt 是 Agent 的灵魂。四段式结构是最稳定的模式：

#### 四段式 System Prompt

```
# 1. 角色定义（我是谁）
你是一个 [角色名称]，专门负责 [核心职责]。
你的能力范围：[列出你能做的事]。
你不负责：[明确排除的职责]。

# 2. 行为规则（我怎么做）
- [规则 1：强制性约束，用 MUST / 必须]
- [规则 2：推荐行为，用 SHOULD / 应该]
- [规则 3：禁止行为，用 NEVER / 禁止]

# 3. 工具使用指南
你有以下工具可用：
- tool_1: [用途 + 使用时机]
- tool_2: [用途 + 使用时机]

工具调用原则：
- 先思考再行动，不要盲目调用
- 每次调用一个工具，等结果回来再决定下一步
- 如果工具返回错误，分析原因后调整参数重试一次

# 4. 输出格式
你的最终回答必须使用以下格式：
```json
{ "answer": "...", "sources": [...], "confidence": 0.0-1.0 }
```
```

#### Prompt 长度黄金法则

| 部分 | 推荐 token 数 | 说明 |
|------|---------------|------|
| 角色定义 | 100-200 | 越短越好 |
| 行为规则 | 200-500 | 用列表，不用段落 |
| 工具描述 | 100-200 per tool | 只保留必要的 |
| 示例 (few-shot) | 200-400 per example | 1-2 个高质量示例足够 |
| **总计** | **800-2000** | 超过 2000 考虑做摘要 |

#### Prompt 优化策略

**策略一：用规则代替示例**

```
✅ 规则：输出使用 emoji 标记状态：✅ 成功 / ❌ 失败 / ⚠️ 警告
❌ 示例：写 5 个带 emoji 的输出样例
```

**策略二：用约束代替"建议"**

```
✅ 如果你不确定答案，直接说"我不确定"，禁止编造。
❌ 请尽量提供准确的答案。
```

**策略三：正面指令 > 反面指令**

```
✅ 始终返回 JSON 格式。
❌ 不要返回纯文本。
```

**策略四：测试驱动迭代**

```
1. 准备 20 条 golden test case
2. 跑当前 prompt，记录失败 case
3. 只修改导致失败的规则，不动成功的规则
4. 重新跑全部 case，确认没有退化
```

---

### L4 — 记忆系统 (Memory)

Agent 的记忆分为三个层级：

#### 三层记忆架构

```
┌──────────────────────────────────────┐
│  L1: Context Window (短期)           │
│  ├─ System Prompt                    │
│  ├─ 当前对话历史                      │
│  └─ 工具调用结果                      │
│  容量: 4K-128K tokens               │
│  失效: 会话结束                       │
└──────────────────────────────────────┘
          ↓ 摘要/压缩
┌──────────────────────────────────────┐
│  L2: Working Memory (中期)           │
│  ├─ .workbuddy/memory/*.md          │
│  ├─ 本次会话的状态变量               │
│  └─ 用户偏好 / 项目配置              │
│  容量: 文件级，无硬限制               │
│  失效: 会话结束（可持久化）           │
└──────────────────────────────────────┘
          ↓ 向量化/索引
┌──────────────────────────────────────┐
│  L3: Long-term Memory (长期)         │
│  ├─ 向量数据库 (Chroma/Pinecone)     │
│  ├─ 知识图谱                        │
│  └─ 关系数据库                       │
│  容量: 百万级记录                    │
│  失效: 手动清理                      │
└──────────────────────────────────────┘
```

#### 上下文管理策略

**问题**：context window 越满，模型推理质量越差（lost-in-the-middle 效应）。

**解法**：

```python
def manage_context(messages: list, max_tokens: int = 8000) -> list:
    """上下文管理策略"""

    # 1. System prompt 永远保留，不可裁剪
    system = [m for m in messages if m["role"] == "system"]

    # 2. 保留最近 N 轮对话
    recent = messages[-6:]  # 最近 3 组 Q&A

    # 3. 中间的对话做摘要压缩
    middle = messages[len(system):-6]
    if middle:
        summary = llm_summarize(middle, max_tokens=500)
        compressed = [{"role": "assistant", "content": f"[历史摘要] {summary}"}]
    else:
        compressed = []

    # 4. 拼接
    return system + compressed + recent
```

#### 会话持久化规范

```json
{
  "session_id": "sess-2026-05-25-001",
  "created_at": "2026-05-25T08:00:00Z",
  "state": {
    "current_phase": "analysis",
    "completed_steps": ["data_collection", "cleaning"],
    "user_preferences": {
      "language": "zh-CN",
      "output_format": "markdown"
    }
  },
  "summary": "用户正在分析上周告警数据，已完成数据采集和清洗",
  "vector_index_id": "idx-sess-001"  // 关联的长期记忆索引
}
```

---

### L5 — 可观测性 (Observability)

没有可观测性的 Agent 是黑盒。出问题时你只知道"结果不对"，但不知道为什么。

#### 必须追踪的五个维度

```python
@dataclass
class TraceStep:
    step_index: int
    thought: str | None         # Agent 的推理过程
    action: str | None          # 调用了哪个工具
    action_input: dict | None   # 工具调用的参数
    observation: str | None     # 工具返回的结果
    model: str                  # 使用的模型
    input_tokens: int           # 输入 token 数
    output_tokens: int          # 输出 token 数
    latency_ms: int             # 本步耗时
    error: str | None           # 错误信息

@dataclass
class TraceSession:
    session_id: str
    user_input: str
    final_output: str
    steps: list[TraceStep]
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    success: bool
```

#### 评估体系

```yaml
evals:
  tool_calling:
    - name: "工具选择准确率"
      metric: "exact_match"
      threshold: 0.92

  answer_quality:
    - name: "事实准确性"
      metric: "human_judge + auto_grader"
      threshold: 0.85

    - name: "指令遵循度"
      metric: "rule_compliance_score"
      threshold: 0.95

  robustness:
    - name: "幻觉率"
      metric: "hallucination_rate"
      threshold: 0.05  # 低于 5%

    - name: "拒答合理性"
      metric: "should_refuse_accuracy"
      threshold: 0.98
```

---

## 第二部分：六大黄金法则

### 法则一：工具最小化

每个 Agent 只保留**完成任务所必需的最少工具**。工具越多 → 决策空间越大 → 选错工具的几率越高。

```
经验值：
- 简单 Agent（分类/提取）：0-2 个工具
- 中等 Agent（查询/分析）：3-5 个工具
- 复杂 Agent（代码/自动化）：5-8 个工具
- 超过 10 个工具 → 考虑拆分为多个 Agent
```

如果 Agent 有 10+ 个工具但只用其中 3 个，删掉那 7 个。

### 法则二：Prompt 原子化

将 system prompt 拆分为可独立测试的模块，而不是写一个巨大的 prompt。

```
✅ 原子化：
  - role.md:      "你是一个 SRE 运维助手"
  - rules.md:     "工具调用前先确认参数完整性"
  - format.md:    "最终答案必须是 JSON"
  - examples.md:  [1 个 few-shot 示例]

❌ 巨石 prompt：
  system_prompt = "你是一个 SRE 运维助手...（2000 tokens 混杂在一起）"
```

原子化的好处：修改一条规则时不影响其他规则，A/B 测试时能精确定位哪条规则导致了变化。

### 法则三：强制结构化输出

所有 Agent 的最终输出必须使用结构化格式。

```python
from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    answer: str = Field(description="给用户的最终回答")
    sources: list[str] = Field(default_factory=list, description="信息来源")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    tool_calls_summary: str | None = Field(default=None)

# 使用 response_format 锁定
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_schema", "json_schema": {...}}
)
```

结构化输出的价值：解析可靠 + 下游可编程 + 错误可检测。如果 Agent 返回了不匹配 schema 的输出，你能立刻发现并重试。

### 法则四：防幻觉护栏

幻觉是 LLM 的原罪。三层防护：

```
Layer 1: Prompt 层
  - 在 system prompt 中明确："不知道就说不确定，禁止编造"
  - 要求引用信息来源

Layer 2: 输出验证层
  - 检查输出是否匹配 json_schema
  - 检查引用的 source 是否真实存在
  - 检查数值范围是否合理

Layer 3: 行为约束层
  - 限制 Agent 只能调用白名单工具
  - 限制文件操作范围 (chroot / sandbox)
  - 外部 API 调用需要显式确认
```

### 法则五：记忆分层

不要把所有信息都塞进 context window。使用三层记忆：

```
热数据（context window）：当前推理需要的信息，< 8K tokens
温数据（working memory）：本次会话的状态和中间结果，文件存储
冷数据（long-term memory）：历史经验和知识，向量检索
```

每次传给模型的 context 都要问自己："这条信息，如果删掉，模型还能正确回答吗？" 如果答案是"能"，就删掉。

### 法则六：先 eval 再上线

没有 eval 的 Agent = 凭感觉开发。

最小 eval 集：
- 20 条 happy path case（正常输入 → 正常输出）
- 10 条边界条件 case（极端输入、空输入、超长输入）
- 5 条工具调用 case（验证工具选择和参数正确性）
- 5 条安全 case（注入攻击、越权请求、不当内容）

每次修改 prompt 或工具定义后，跑一遍完整 eval。任何一条退化都不能上线。

---

## 第三部分：常见反模式

| 反模式 | 症状 | 正确做法 |
|--------|------|----------|
| **过度 tool-calling** | Agent 对简单问题也调用 5+ 个工具 | 加规则："如果不需要工具就能回答，直接回答" |
| **Prompt 膨胀** | System prompt 超过 3000 tokens | 拆分为模块，延迟加载，无关内容放外部文档 |
| **幻觉信任** | 不验证 Agent 输出就展示给用户 | 结构化输出 + schema 验证 + source 检查 |
| **隐式状态丢失** | 会话切换后 Agent "忘"了之前的决策 | 持久化 working memory，新会话加载状态 |
| **无限循环** | Agent 反复调用同一个工具得不到结果 | 检测连续 3 次相同/相似的工具调用 → 强制终止 |
| **模糊错误处理** | 工具返回 `{"error": "failed"}` | 返回 `{"error": "...", "suggestion": "..."}` |
| **暴力破解** | 遇到解析失败就写正则 workaround | 从 prompt 源头约束输出格式，不走 post-processing hack |
| **缺少人类确认环节** | Agent 直接执行破坏性操作 | 对 dangerous 级别工具加确认流程 |

---

## 第四部分：生产部署检查清单

```
□ 1. Agent Loop 有 max_steps 硬限制吗？（推荐 15）
□ 2. 每个工具都有 timeout 吗？（推荐 30s）
□ 3. System prompt 在 2000 tokens 以内吗？
□ 4. 每条输出都用 json_schema 锁定了吗？
□ 5. 有防幻觉的三层护栏吗？（prompt + 验证 + 约束）
□ 6. 上下文管理策略实现了吗？（滑动窗口 + 摘要压缩）
□ 7. Working memory 能跨轮次持久化吗？
□ 8. 危险工具（删除/发送/外部 API）有确认步骤吗？
□ 9. 每条 LLM 调用都有 trace 记录吗？
□ 10. 工具连续调用 3 次失败有熔断吗？
□ 11. 有 >50 条 golden eval case 吗？
□ 12. 修改 prompt 后跑了完整回归测试吗？
□ 13. Token 成本和延迟有监控 dashboard 吗？
□ 14. 模型降级策略准备好了吗（GPT-4o 挂了切 4o-mini）？
```

---

## 开发流程总结

```
Phase 1: 定义范围
  这个 Agent 要解决什么问题？边界在哪里？

Phase 2: 设计工具
  完成任务最少需要哪些工具？每个工具的输入输出是什么？

Phase 3: 写 System Prompt
  角色 + 规则 + 工具指南 + 输出格式，四段式

Phase 4: 写 Eval
  50+ 条 golden case，覆盖正常/边界/异常/安全

Phase 5: 迭代调优
  跑 eval → 分析失败 case → 只改出问题的那条规则 → 再跑 eval

Phase 6: 成本优化
  能否用弱模型？能否缓存？能否减少工具调用？

Phase 7: 生产部署
  加 trace → 加监控 → 加告警 → 灰度上线
```

---

## 核心思想

> **Agent 不是一个能自主思考的实体，它是一个被 Prompt + Tools + Loop 严格约束的执行单元。你控制得越精确，它表现得越好。**

三条铁律：

1. **能用规则解决的，不要让 LLM "自己判断"。** 规则是确定的，LLM 是不确定的。
2. **能结构化输出的，不要让 LLM 自由发挥。** JSON 永远比自然语言可靠。
3. **能在 prompt 里约束的，不要在代码里 patch。** 源头修正比下游打补丁干净一万倍。

---

*最后更新: 2026-05-25*

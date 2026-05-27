# LLM Multi-Agent 开发知识库

> 方法论 + 最佳实践 + 反模式 + 检查清单，一套完整体系。

---

## 第一部分：方法论

### 一、架构模式选择

这是第一个关键决策。不同模式适用于不同场景：

| 模式 | 结构 | 适用场景 | 核心权衡 |
|------|------|----------|----------|
| **层级式** (Manager-Worker) | 一个 Orchestrator -> N 个 Specialist | 复杂任务分解、多步骤工作流 | 可控性强，但单点瓶颈 |
| **对等式** (Peer-to-Peer) | 各 Agent 平等对话，互相质询 | 代码审查、头脑风暴 | 冗余度高，幻觉风险大 |
| **流水线式** (Pipeline) | A -> B -> C 顺序加工 | 数据 ETL、文档生成流水线 | 延迟高，但每步可独立优化 |
| **辩论/反思式** | 多个 Agent 生成 -> 互相批判 -> 迭代收敛 | 安全性关键场景、高质量写作 | 质量高，但成本成倍增长 |
| **混合编排式** | 上述模式的动态组合 (DAG) | 生产级复杂系统 | 灵活但编排复杂度陡增 |

**选择依据**：任务能否预先分解为独立子任务？
- 能 -> 层级式 / 流水线式
- 不能 -> 对等式 / 辩论式
- 对延迟敏感 -> 对等式并行
- 对质量要求极高 -> 辩论式

---

### 二、Agent 设计原则

单一 Agent 的质量决定系统的天花板。

**1. 角色边界清晰**

每个 Agent 的 system prompt 必须包含：
- 角色定义（我是谁，我擅长什么）
- 能力边界（我能做什么，不能做什么）
- 禁止行为（绝对不要做的事情）
- 输出格式（必须产出的结构）

不要让一个 Agent 做所有事——这和单 Agent 没区别。

**2. 工具集最小化**

只给 Agent 它职责范围内需要的工具。工具越多，决策空间越大，出错概率越高。

**3. 记忆分层设计**

| 层级 | 存储方式 | 生命周期 | 用途 |
|------|----------|----------|------|
| 短期 | 对话上下文窗口 | 当前对话 | 直接推理 |
| 中期 | 工作记忆文件 (读写) | 当前会话 | 跨轮次状态 |
| 长期 | 向量检索 / 知识图谱 | 跨会话 | 历史经验复用 |

**4. 结构化输出强制**

Agent 间的通信必须用结构化格式（JSON Schema / Pydantic），自然语言仅用于人机交互。结构化输出是防止解析错误和幻觉传播的第一道防线。

---

### 三、协调与通信机制

#### 3.1 任务分解与分配

```
Orchestrator 接收用户意图
  -> 分析为子任务 DAG（有向无环图）
  -> 识别并行机会（无依赖的兄弟节点）
  -> 为每个节点匹配最合适的 Agent
  -> 按拓扑序调度，并行节点并发执行
  -> 聚合结果，检查完整性
```

DAG 节点定义规范：
```json
{
  "id": "task-01",
  "agent_type": "alert_analyzer",
  "input_schema": { "alerts": "List[Alert]" },
  "output_schema": { "analysis": "AlertAnalysis" },
  "dependencies": [],
  "timeout": 30,
  "retry_policy": { "max_attempts": 2, "backoff": "exponential" }
}
```

#### 3.2 通信模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **消息队列**（推荐） | 每个 Agent 有独立 inbox，Orchestrator 路由 | 需要可靠投递和追踪 |
| **共享黑板** | 所有 Agent 读写同一个状态对象 | 简单系统，但难调试 |
| **事件总线** | Agent 订阅感兴趣的事件类型 | 松耦合系统 |

推荐数据结构：
```json
{
  "from": "orchestrator",
  "to": "alert_analyzer",
  "type": "task_assignment",
  "payload": { ... },
  "correlation_id": "req-2026-001"
}
```

#### 3.3 冲突解决策略

当多个 Agent 对同一问题产生分歧：

1. **投票**：简单，各 Agent 权重相等
2. **加权投票**：给专家 Agent 更高权重
3. **仲裁器**（推荐）：独立 Evaluator Agent 做最终判决
4. **多数+置信度**：Agent 需要输出置信度分数

推荐策略：加权投票 + 仲裁器组合。先按领域权重聚合，分歧超过阈值时引入独立仲裁。

消息格式示例：
```json
{
  "agent_id": "security_agent",
  "decision": "BLOCK",
  "confidence": 0.92,
  "reasoning": "输入包含潜在 SQL 注入特征",
  "alternatives": ["ALLOW_WITH_SANITIZATION"]
}
```

---

### 四、工程化实践

#### 4.1 评估体系（三层）

| 层次 | 评估内容 | 指标 |
|------|----------|------|
| Agent 级 | 工具调用准确率、任务完成率 | exact_match, fuzzy_match |
| 编排级 | 任务分解合理性、并行利用率 | human_judge_score |
| 端到端 | 用户满意度、延迟、成本 | win_rate vs baseline |

#### 4.2 可观测性

每条 LLM 调用必须记录：
```json
{
  "agent_id": "alert_analyzer",
  "prompt": "...",
  "response": "...",
  "input_tokens": 1234,
  "output_tokens": 567,
  "latency_ms": 2300,
  "tool_calls": ["search_alert_db"],
  "timestamp": "2026-05-25T15:30:00Z",
  "correlation_id": "req-2026-001"
}
```

#### 4.3 成本控制

弱模型先行策略：
```
输入 -> GPT-4o-mini ($0.15/1M tokens)
  -> 能解决？-> 返回结果
  -> 不能？-> 升级到 GPT-4o ($2.50/1M tokens)
  -> 还不能？-> 升级到 Claude Opus ($15/1M tokens)
```

加 prompt 缓存 + 语义缓存，可将成本降低 40-70%。

#### 4.4 开发流程

```
Phase 1: 问题建模
  - 这个任务单 Agent 能解决吗？-> 能 -> 别用 Multi-Agent
  - 不能 -> 画出子任务 DAG

Phase 2: 单 Agent 验证
  - 先让每个 Specialist Agent 独立跑通
  - 确保单个 Agent 的输出质量达标

Phase 3: 编排上线
  - 先跑串行流水线（最可控）
  - 逐步放开并行
  - 引入辩论/反思（如果质量不够）

Phase 4: 观测迭代
  - 看哪些 Agent 是瓶颈
  - 优化 prompt、换模型、调整工具集
  - 持续 A/B 测试
```

---

## 第二部分：最佳实践 - 六大黄金法则

### 法则一：从单 Agent 开始，只拆必要的

**这是最容易犯的错——上来就画五个 Agent 的架构图。**

需要 Multi-Agent 的三种场景：
- ✅ 上下文溢出：单次任务需要的信息 > 128K tokens
- ✅ 能力不可合并：需要同时具备"写代码"和"法律审查"两种正交能力
- ✅ 需要并行加速：多个独立子任务可并发

不需要 Multi-Agent 的场景：
- ❌ "多个人讨论会更严谨" -> 加约束 prompt，别加 Agent
- ❌ "架构很酷" -> 这不是工程理由
- ❌ "流程图看起来更完整" -> 复杂 != 好

**实践准则**：先用一个强模型单 Agent 跑通 full pipeline。测量延迟、成本、准确率。只有某个指标确实不满足业务需求时，才针对那个瓶颈拆 Agent。

---

### 法则二：结构化通信，杜绝 Agent 间用自然语言

**这条踩坑率最高。**

反例（自然语言通信）：
```
Agent A 输出: "找到 30 个左右的告警，大部分在当前园区"
Agent B 解析: 正则匹配到 "30" -> 拿去查 30 条 -> 实际是 27 条 -> 漏了 3 条 P0
```

正例（结构化通信）：
```python
from pydantic import BaseModel
from typing import List, Dict

class AlertItem(BaseModel):
    id: str
    severity: str  # P0, P1, P2
    zone: str
    message: str

class AlertAnalysisOutput(BaseModel):
    alerts: List[AlertItem]
    total: int
    severity_distribution: Dict[str, int]
    summary: str  # 仅用于 logging，不用于下游解析
```

**实现方式**：Agent 间的所有消息使用 Pydantic / JSON Schema 定义。System prompt 里强制要求结构化输出，用 response_format 参数锁定。

---

### 法则三：模型分层路由

不是每个 Agent 都需要最强模型。

```
Level 1: 结构化提取、格式转换、简单分类
  模型: GPT-4o-mini / Claude Haiku / Gemini Flash
  成本: ~$0.15/M input tokens

Level 2: 推理、分析、中等复杂度
  模型: GPT-4o / Claude Sonnet
  成本: ~$2.50/M input tokens

Level 3: 复杂推理、多步规划、创意生成
  模型: Claude Opus / GPT-4.5 / o1
  成本: $10-15/M input tokens

路由规则: 低层模型失败 -> 自动升级到上一层重试
```

**关键实现**：让一个轻量 Router Agent（或规则引擎）根据任务元信息自动路由。模型选择是工程决策，不应让 LLM 自己选。

---

### 法则四：防御式编排

把每个 Agent 当成不可靠的远程服务来设计。

```python
import asyncio
from functools import wraps

def resilient_agent_call(max_attempts=2, timeout=30,
                          circuit_threshold=5, recovery_timeout=60):
    """防御式 Agent 调用装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(agent, task):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    result = await asyncio.wait_for(
                        func(agent, task),
                        timeout=timeout
                    )
                    # 验证输出格式
                    if not validate_output(result, task.output_schema):
                        raise ValidationError(
                            f"Agent output doesn't match schema"
                        )
                    return result
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    continue
                except ValidationError as e:
                    last_error = str(e)
                    continue
                except Exception as e:
                    last_error = str(e)
                    continue

            # 全部重试失败 -> 降级
            return AgentResult(
                success=False,
                fallback=task.fallback_value,
                error=last_error
            )
        return wrapper
    return decorator
```

**关键配置**：
- timeout = 30s（单 Agent 调用）
- retry = 2 次（指数退避）
- fallback = 默认输出 / 部分结果
- circuit_breaker = 5 次失败后熔断 60s

**Orchestrator 降级策略**：如果子 Agent 挂了但其他并行任务完成了，返回部分结果 + 标注缺失项。比整个请求 500 好得多。

---

### 法则五：评估先行

这是 Multi-Agent 系统从"玩具"到"产品"的分水岭。

#### 评估配置规范

```yaml
evals:
  agent_level:
    - name: "告警分析 Agent 准确率"
      cases: 50
      metric: "exact_match + fuzzy_match"
      threshold: 0.90
    - name: "工具调用准确率"
      cases: 30
      metric: "tool_call_correctness"
      threshold: 0.95

  orchestration_level:
    - name: "任务分解合理性"
      cases: 30
      metric: "human_judge_score"
      threshold: 4.2  # out of 5
    - name: "并行利用率"
      metric: "parallelism_ratio"
      threshold: 0.60  # 60%+ 的可并行任务被并行执行

  end_to_end:
    - name: "用户满意度"
      cases: 100
      metric: "win_rate vs baseline"
      threshold: 0.05  # 相比单 Agent 提升 5%+
```

#### Golden Test Case 设计原则

- Happy path: 70%
- 边界条件: 20%
- 故意触发 Agent 故障: 10%
- 每条包含 `输入 + 期望输出 + 可接受的偏差范围`

---

### 法则六：记忆卫生

Context window 是昂贵资源。每多 1K tokens -> 推理延迟增加 50-100ms，成本线性增加，推理质量下降（lost-in-the-middle 效应）。

错误做法 vs 正确做法：

```python
# ❌ 错误：把所有历史对话塞进 context
def bad_context(agent, task):
    history = load_full_history()  # 可能 50K tokens
    return history + task

# ✅ 正确：只传最小必要信息
def good_context(agent, task):
    return {
        "current_task": task,
        # 只检索最相关的 3 条历史
        "relevant_history": vector_search(task, top_k=3),
        # 当前进度，不传完整历史
        "state": {"step": 5, "total_steps": 10},
    }
    # 确保 context < 4K tokens
```

**实践建议**：给每个 Agent 设 `max_context_tokens`（建议 4K-8K）。超过则先裁剪、压缩、摘要，再传入。

---

## 第三部分：常见反模式

| 反模式 | 症状 | 正确做法 |
|--------|------|----------|
| **Agent 膨胀** | 系统有 15+ 个 Agent，大部分互相重复 | 合并相似职责，一个 Agent 可以有多步思维链 |
| **无限对话循环** | Agent A 和 Agent B 来回讨论不停 | 设 max_rounds=3，超时则取最近一次输出 |
| **Prompt 泄露** | Agent 向用户暴露内部编排细节 | 输出分 internal（给其他 Agent）和 external（给用户），严格隔离 |
| **幻觉级联** | A输出小错 -> B放大 -> C彻底跑偏 | 每个输出附带 confidence_score，低置信度请求重试 |
| **调试黑洞** | 系统错了，完全不知道哪个 Agent 出问题 | 每步记录完整追踪链路，构建回放能力 |
| **过早优化** | 还没验证单 Agent 就开始拆多 Agent | 先跑通单 Agent，测量瓶颈，再针对拆解 |
| **Prompt 膨胀** | system prompt 越写越长（> 2K tokens） | 精简到核心指令，参考文档放外部检索 |
| **串行化滥用** | 本来能并行的任务却串行执行 | 画 DAG，识别无依赖的兄弟节点并行执行 |

---

## 第四部分：生产部署检查清单

部署到生产环境前，逐条自查：

```
□ 1. 单 Agent 真做不了吗？（跑过 benchmark 对比）
□ 2. 每个 Agent 之间的接口是 JSON Schema 吗？
□ 3. 每个 Agent 有 timeout + retry + fallback 吗？
□ 4. 模型路由策略是显式的、可配置的吗？
□ 5. 三层 eval 都跑过，且分数达标吗？
□ 6. 每个 Agent 的 context 都控制在 4-8K tokens 以内吗？
□ 7. 有 max_rounds 防止无限循环吗？
□ 8. 每条 Agent 调用有追踪链路吗（可回放）？
□ 9. 输出区分了 internal 和 external 吗？
□ 10. 降级策略测试过吗（模拟某 Agent 挂了，系统还能输出什么）？
□ 11. prompt cache 开启了吗？（相同 system prompt 不重复计费）
□ 12. 敏感信息过滤到位了吗？（Agent 输入输出都过一遍安全校验）
□ 13. 速率限制配好了吗？（防止恶意触发大量 Agent 调用）
□ 14. 回滚方案准备好了吗？（一键切回单 Agent 或旧版本）
```

---

## 核心思想总结

> **把 Multi-Agent 当成分布式系统来设计，而不是当成"让多个 LLM 聊天"来设计。**

分布式系统的一切原则——容错、超时、降级、可观测、幂等——在 Multi-Agent 系统中全部适用。LLM 只是让你的节点变聪明了，但节点的不可靠性并没变。

三条铁律：

1. **可控 > 智能**：一个可预测的简单系统，好过一个不可预测的复杂系统。
2. **可观测 > 可运行**：能跑起来不重要，出问题能定位才重要。
3. **可降级 > 完美**：给用户部分结果，比什么都不给强 100 倍。

---

*最后更新: 2026-05-25*

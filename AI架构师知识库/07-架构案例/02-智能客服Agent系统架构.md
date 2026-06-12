# 案例二：智能客服Agent系统架构

> 支持10万+日活的多轮对话Agent平台

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [Agent系统设计](../04-Agent系统/Agent架构模式.md), [RAG系统设计基础](../01-基础理论/RAG.md)
- **关联文件**: [企业级RAG知识库系统架构](./01-企业级RAG知识库系统架构.md), [多模型路由网关架构](./04-多模型路由网关架构.md)
- **最后更新**: 2026-06-12
---

## 1. 项目背景

### 1.1 业务需求

**客户画像：** 大型电商平台，日均客服咨询20万+

**核心需求：**
- 自动处理70%+常见咨询（退换货、物流、账户等）
- 复杂问题无缝转人工
- 支持多轮对话（平均5-8轮）
- 接入企业内部系统（订单、库存、物流）
- 情绪识别与安抚

### 1.2 关键挑战

| 挑战 | 描述 |
|------|------|
| **多轮理解** | 用户表述模糊，需要追问澄清 |
| **工具调用** | 查询订单、退款、修改地址等10+内部API |
| **情绪管理** | 识别用户情绪，愤怒时优先转人工 |
| **知识更新** | 促销活动频繁，知识需实时更新 |
| **成本控制** | 月Token预算有限制 |

---

## 2. 整体架构

### 2.1 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                          接入层                                   │
│  Web/APP/小程序 → WebSocket网关 → 会话管理 → 路由分发            │
└──────────────┬───────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│                       Agent编排层                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    LangGraph 状态机                       │    │
│  │                                                          │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │    │
│  │  │意图  │→│路由  │→│工具  │→│生成  │→│后处理│     │    │
│  │  │识别  │  │决策  │  │调用  │  │回答  │  │检查  │     │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘     │    │
│  │       ↑                                    │            │    │
│  │       └──────── 循环（多轮对话）────────────┘            │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        能力层                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ LLM服务  │  │ RAG服务   │  │ 工具服务  │  │ 情绪分析  │        │
│  │(GPT-4o)  │  │(知识库)   │  │(10+API)  │  │(BERT)    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 编排框架 | LangGraph | 状态机+条件路由+持久化 |
| 主力LLM | GPT-4o（API） | 推理能力强，工具调用准确 |
| 降级模型 | GPT-4o-mini | 成本敏感场景降级 |
| 知识检索 | Milvus + Rerank | 促销信息实时检索 |
| 情绪识别 | 微调BERT | 轻量、快速、准确 |
| 会话存储 | Redis + PostgreSQL | Redis热数据+PG持久化 |

---

## 3. Agent状态机设计

### 3.1 状态定义

```python
class CustomerServiceState(TypedDict):
    # 会话信息
    session_id: str
    user_id: str
    message_count: int
    
    # 对话内容
    messages: Annotated[list, operator.add]  # 对话历史
    current_query: str                         # 当前用户输入
    
    # 意图与路由
    intent: str              # 意图分类
    sub_intent: str          # 子意图
    confidence: float        # 意图置信度
    need_human: bool         # 是否需要转人工
    
    # 工具调用
    tool_calls: list         # 已调用的工具
    tool_results: dict       # 工具返回结果
    
    # 情绪
    emotion: str             # 情绪：normal/angry/anxious/happy
    emotion_score: float     # 情绪强度
    
    # 业务上下文
    order_id: str            # 关联订单号
    user_profile: dict       # 用户画像
    
    # 控制
    max_turns: int           # 最大轮数
    should_end: bool         # 是否结束对话
```

### 3.2 状态流转图

```
                    ┌──────────┐
                    │  开始    │
                    └────┬─────┘
                         ↓
                ┌────────────────┐
                │  输入预处理     │
                │  (情绪检测)     │
                └───────┬────────┘
                        ↓
              ┌─────────────────────┐
         ┌───│     情绪是否异常？    │───┐
         │   └─────────────────────┘   │
         ↓ 是                          ↓ 否
  ┌──────────────┐          ┌────────────────┐
  │ 转人工+安抚   │          │   意图识别      │
  └──────────────┘          └───────┬────────┘
                                     ↓
                           ┌─────────────────────┐
                      ┌────│   意图置信度>0.8？   │────┐
                      │    └─────────────────────┘    │
                      ↓ 是                             ↓ 否
              ┌──────────────┐              ┌──────────────┐
              │  路由到对应    │              │  追问澄清     │
              │  处理流程      │              │  (多轮理解)   │
              └──────┬───────┘              └──────┬───────┘
                     ↓                             ↓
         ┌───────────────────────┐     ┌──────────────────┐
         │  是否需要调用工具？     │     │  等待用户回复     │
         └──────┬────────┬───────┘     └────────┬─────────┘
           是   ↓        ↓ 否                   ↓
    ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
    │  工具调用     │  │  知识库检索   │   │  回到输入     │
    │  (订单/退款)  │  │  (RAG)       │   │  预处理       │
    └──────┬───────┘  └──────┬───────┘   └──────────────┘
           └────────┬───────┘
                    ↓
           ┌──────────────┐
           │  生成回答     │
           │  (LLM)       │
           └──────┬───────┘
                  ↓
           ┌──────────────┐
           │  后处理检查   │
           │  (合规/质量)  │
           └──────┬───────┘
                  ↓
           ┌──────────────┐     是
           │  是否结束？   │────────→ 结束
           └──────┬───────┘
                  ↓ 否
           ┌──────────────┐
           │  等待用户回复  │──→ 回到输入预处理
           └──────────────┘
```

---

## 4. 核心模块设计

### 4.1 意图识别

**意图分类体系：**

| 一级意图 | 二级意图 | 处理方式 |
|----------|----------|----------|
| **订单相关** | 查询订单状态 | 工具调用：查询订单API |
| | 修改收货地址 | 工具调用：修改地址API |
| | 取消订单 | 工具调用：取消订单API |
| **售后相关** | 申请退款 | 工具调用：退款API |
| | 退换货 | 工具调用：退换货API |
| | 投诉 | 转人工 |
| **物流相关** | 查询物流 | 工具调用：物流查询API |
| | 催发货 | 工具调用+安抚话术 |
| **账户相关** | 修改密码 | 引导自助操作 |
| | 账户异常 | 转人工 |
| **商品相关** | 商品咨询 | RAG知识库检索 |
| | 库存查询 | 工具调用：库存API |
| **活动相关** | 促销规则 | RAG知识库检索 |
| | 优惠券使用 | 工具调用+知识库 |
| **其他** | 闲聊/致谢 | 直接回复 |
| | 无法识别 | 追问澄清 |

**意图识别实现：**
```python
class IntentClassifier:
    """两级意图分类"""
    
    def classify(self, query: str, conversation_history: list):
        # 第一级：LLM分类（高准确率）
        prompt = f"""
        根据用户消息和对话历史，识别用户意图。

        对话历史：{conversation_history[-6:]}
        当前消息：{query}

        意图列表：{self.intent_list}

        输出JSON：
        {{"intent": "一级意图", "sub_intent": "二级意图", "confidence": 0.0-1.0, "entities": {{}}}}
        """
        
        result = self.llm.invoke(prompt)
        intent_data = json.loads(result.content)
        
        # 低置信度 → 追问
        if intent_data["confidence"] < 0.8:
            return {
                "intent": "clarify",
                "action": "ask_clarification",
                "message": self._generate_clarification(query, intent_data)
            }
        
        return intent_data
```

### 4.2 工具调用层

**工具注册：**
```python
# 订单相关工具
@tool("query_order", "查询订单状态和详情")
def query_order(order_id: str, user_id: str) -> dict:
    """调用订单服务API"""
    
@tool("modify_address", "修改订单收货地址")
def modify_address(order_id: str, new_address: str, user_id: str) -> dict:
    """调用订单修改API"""
    
@tool("cancel_order", "取消未发货订单")
def cancel_order(order_id: str, reason: str, user_id: str) -> dict:
    """调用取消订单API"""

@tool("query_logistics", "查询物流信息")
def query_logistics(order_id: str) -> dict:
    """调用物流查询API"""

@tool("apply_refund", "申请退款")
def apply_refund(order_id: str, reason: str, amount: float, user_id: str) -> dict:
    """调用退款API"""

@tool("query_inventory", "查询商品库存")
def query_inventory(sku_id: str) -> dict:
    """调用库存API"""

# 工具权限映射
TOOL_PERMISSIONS = {
    "query_order": ["all"],
    "modify_address": ["self_only", "before_ship"],
    "cancel_order": ["self_only", "before_ship"],
    "query_logistics": ["all"],
    "apply_refund": ["self_only", "after_receive"],
    "query_inventory": ["all"],
}
```

**工具调用安全：**
```python
class SafeToolExecutor:
    """安全工具执行器"""
    
    def execute(self, tool_name: str, args: dict, user_context: dict):
        # 1. 权限检查
        if not self._check_permission(tool_name, args, user_context):
            return {"error": "无权执行此操作", "code": "PERMISSION_DENIED"}
        
        # 2. 参数校验
        if not self._validate_args(tool_name, args):
            return {"error": "参数不合法", "code": "INVALID_ARGS"}
        
        # 3. 限流检查
        if not self._check_rate_limit(user_context["user_id"], tool_name):
            return {"error": "操作过于频繁，请稍后再试", "code": "RATE_LIMITED"}
        
        # 4. 敏感操作确认
        if tool_name in ["cancel_order", "apply_refund"]:
            return {
                "need_confirmation": True,
                "message": f"确认要执行{tool_name}吗？",
                "details": args
            }
        
        # 5. 执行
        try:
            result = self._call_tool(tool_name, args)
            self._audit_log(tool_name, args, result, user_context)
            return result
        except Exception as e:
            return {"error": "系统繁忙，请稍后重试", "code": "SYSTEM_ERROR"}
```

### 4.3 情绪管理

```python
class EmotionManager:
    """情绪识别与管理"""
    
    def detect(self, message: str, history: list) -> dict:
        """检测用户情绪"""
        # 基于规则+模型
        anger_keywords = ["垃圾", "骗子", "投诉", "举报", "差评", "退款"]
        anxiety_keywords = ["急", "赶紧", "什么时候", "等不了"]
        
        # 快速规则检测
        if any(kw in message for kw in anger_keywords):
            return {"emotion": "angry", "score": 0.9, "action": "escalate"}
        
        # BERT模型检测
        emotion_result = self.emotion_model.predict(message)
        
        # 结合历史（连续3次负面 → 升级）
        recent_emotions = [h.get("emotion") for h in history[-3:]]
        if all(e == "angry" for e in recent_emotions):
            return {"emotion": "angry", "score": 1.0, "action": "escalate"}
        
        return emotion_result
    
    def generate_response_strategy(self, emotion: dict):
        """根据情绪生成回复策略"""
        if emotion["emotion"] == "angry":
            return {
                "tone": "empathetic",      # 共情语气
                "priority": "human_transfer",  # 优先转人工
                "template": "非常抱歉给您带来不好的体验..."
            }
        elif emotion["emotion"] == "anxious":
            return {
                "tone": "reassuring",     # 安抚语气
                "priority": "quick_resolve",  # 快速解决
                "template": "理解您的心情，我马上为您处理..."
            }
        else:
            return {
                "tone": "professional",
                "priority": "normal",
                "template": None
            }
```

---

## 5. 知识库管理

### 5.1 知识分类

```
知识库结构:
├── 商品知识（SKU属性、规格、保修）
├── 订单知识（下单、支付、发票）
├── 物流知识（配送范围、时效、费用）
├── 售后知识（退换货、退款、维修）
├── 活动知识（促销规则、优惠券、满减）
├── 账户知识（注册、密码、积分）
└── 通用知识（公司介绍、联系方式）
```

### 5.2 实时知识更新

```
促销活动知识更新流程:
1. 运营在CMS发布活动规则
2. Webhook触发知识更新
3. 解析活动文档 → 分块 → 向量化
4. 写入Milvus（带活动时间范围）
5. 过期活动自动标记失效

确保:
- 新活动上线后5分钟内可被检索到
- 过期活动不再出现在检索结果中
```

---

## 6. 人机协作

### 6.1 转人工策略

```python
class HumanTransferStrategy:
    """转人工决策"""
    
    def should_transfer(self, state: CustomerServiceState) -> dict:
        triggers = []
        
        # 1. 情绪触发
        if state["emotion"] == "angry" and state["emotion_score"] > 0.8:
            triggers.append("emotion")
        
        # 2. 意图触发
        if state["intent"] in ["投诉", "账户异常", "法律纠纷"]:
            triggers.append("intent")
        
        # 3. 轮数触发
        if state["message_count"] > 10:
            triggers.append("max_turns")
        
        # 4. 置信度触发
        if state["confidence"] < 0.5 and state["message_count"] > 3:
            triggers.append("low_confidence")
        
        # 5. 用户主动要求
        if "人工" in state["current_query"] or "客服" in state["current_query"]:
            triggers.append("user_request")
        
        if triggers:
            return {
                "transfer": True,
                "reason": triggers,
                "context_summary": self._summarize_context(state)
            }
        
        return {"transfer": False}
    
    def _summarize_context(self, state):
        """为人工客服生成对话摘要"""
        return self.llm.invoke(f"""
        请将以下对话摘要为3-5句话，供人工客服参考：
        {state['messages']}
        
        包含：用户问题、已处理事项、待解决问题。
        """)
```

### 6.2 人工接管与回退

```
转人工流程:
1. Agent生成转人工消息："正在为您转接人工客服..."
2. 对话上下文摘要发送给人工座席
3. 人工座席接管会话
4. 人工解决问题后，可将会话转回Agent
5. Agent继续后续服务

技术实现:
- WebSocket连接保持
- 会话状态在Redis中持久化
- 人工座席通过WebSocket接入同一会话
- 消息队列异步通知人工座席
```

---

## 7. 性能与成本

### 7.1 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 自动解决率 | > 70% | 73% |
| 平均响应时间 | < 2s | 1.5s |
| 平均对话轮数 | 3-5轮 | 4.2轮 |
| 转人工率 | < 30% | 27% |
| 用户满意度 | > 85% | 88% |

### 7.2 成本优化

| 策略 | 节省 | 实现 |
|------|------|------|
| **模型路由** | 40% | 简单问题用GPT-4o-mini |
| **语义缓存** | 15% | 相似问题缓存 |
| **知识库优先** | 20% | RAG检索优先于LLM推理 |
| **流式输出** | 用户体验 | 减少感知延迟 |

**月成本估算：**
```
日均20万次对话:
- GPT-4o: 30% × 20万 × $0.01 = $600/天
- GPT-4o-mini: 70% × 20万 × $0.001 = $140/天
- Embedding: 20万 × $0.0001 = $20/天
- 月合计: ≈ $23,400/月（约17万人民币/月）
```

---

## 8. 经验总结

### 8.1 关键成功因素

1. **意图识别准确率是核心**：错误路由会导致用户体验极差
2. **工具调用必须安全**：涉及订单/退款等敏感操作，必须有确认机制
3. **情绪管理不可忽视**：愤怒用户处理不当会导致投诉升级
4. **知识库实时性**：促销活动信息必须及时更新
5. **人机切换要无缝**：转人工时上下文必须完整传递

### 8.2 踩过的坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent反复调用同一工具 | 状态管理不当 | 添加调用记录，避免重复 |
| 退款操作误执行 | 缺少确认机制 | 敏感操作强制二次确认 |
| 促销信息过期 | 知识未及时更新 | CMS Webhook实时触发 |
| 多轮对话丢失上下文 | 会话超时 | Redis持久化+自动恢复 |

---

## 深度分析

智能客服Agent系统是多轮对话AI最典型的应用场景之一，其架构复杂度远超单轮问答系统。本文采用LangGraph状态机作为核心编排引擎，将对话流程建模为"意图识别→路由决策→工具调用→生成回答→后处理检查"的循环状态机。这种设计相比传统的链式调用（Pipeline）有本质区别——状态机支持条件分支、循环回溯和状态持久化，能够灵活应对用户意图不明确时需要追问澄清、工具调用失败时需要重试等复杂对话场景。

意图识别与情绪管理的双引擎设计是本系统的关键创新。意图识别采用LLM分类（高准确率）配合置信度阈值，低置信度时触发追问而非盲目决策；情绪管理则使用轻量规则+BERT模型的混合策略，快速检测愤怒/焦虑情绪并触发转人工或安抚话术。从成本优化角度看，模型路由（简单问题用GPT-4o-mini）和语义缓存策略在实际生产中将月成本控制在可控范围内，而紧张用户满意度88%和自动解决率73%的指标说明这套架构在生产环境中表现良好。

转人工策略的精细设计体现了"人机协作"而非"人机替代"的产品理念。五种转人工触发条件（情绪、意图、轮数、置信度、用户请求）构成了完整的升级机制，而对话上下文摘要功能确保人工客服接管时信息不丢失。这种设计思路在《8.2 踩过的坑》中也有所体现——退款操作缺少确认机制导致误执行，促销信息过期导致错误答复，这些问题都指向一个核心原则：Agent的能力边界必须清晰，超出边界时应当优雅地转交人类处理。

## Checklist

- [ ] 验证LangGraph状态机的状态流转是否正确覆盖所有业务场景
- [ ] 测试意图分类的准确率，确保低置信度时触发追问而非错误路由
- [ ] 配置情绪识别模型和规则引擎，验证愤怒情绪的及时转人工
- [ ] 实现敏感操作的二次确认机制（退款/取消订单等）
- [ ] 验证工具调用的权限检查、参数校验和限流逻辑
- [ ] 配置转人工触发条件和上下文摘要生成
- [ ] 部署语义缓存并测试缓存命中率对成本的影响
- [ ] 建立促销活动知识的实时更新机制（CMS Webhook）
- [ ] 压测会话并发能力，验证Redis热数据+PG持久化的可靠性
- [ ] 监控自动解决率和用户满意度，建立持续优化反馈闭环

## 延伸阅读

- [LangGraph 官方文档：会话Agent构建](https://langchain-ai.github.io/langgraph/)
- [客服系统智能化的最佳实践](https://arxiv.org/abs/2401.12345)
- [多轮对话中的情绪识别技术综述](https://aclanthology.org/2023.emnlp-main.1/)

*最后更新：2026-06-12*

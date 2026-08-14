# 生产级 Agent 四层架构工程落地

> Brain / Memory / Action / Governance——2026 年被 LangChain/LangGraph、微软 AutoGen、字节 Coze 企业版、蚂蚁 AgentStack 等踩成的相对标准"生产骨架"。不堆概念，直接说每块用什么、怎么接、坑在哪。

## 元数据
- **难度**: ⭐⭐⭐⭐
- **前置知识**: [[01-Agent架构演进]], [[02-Multi-Agent协作架构]], [[10-AgentOS架构与运行时]]
- **关联文件**: [[01-Agent架构演进]], [[03-Agent协议与通信架构]], [[07-Agent安全防护体系]], [[09-MCP协议2026演进与无状态传输]], [[10-AgentOS架构与运行时]], [[12-AgentOS技术组件选型指南]], [[13-十万用户级企业架构方案]], [[14-百万用户级超级架构方案]], [[15-亿级用户终极架构方案]], [[05-Agent脚手架与快速启动工具生态|../09-Agent框架生态/05-Agent脚手架与快速启动工具生态]], [[06-生产级框架选型批判性决策指南|../09-Agent框架生态/06-生产级框架选型批判性决策指南]]
- **最后更新**: 2026-08-14

---

## 目录

- [1. 架构全景：四层稳态](#1-架构全景四层稳态)
- [2. Brain 推理层：LLM + 递归推理编排](#2-brain-推理层llm--递归推理编排)
  - [2.1 底座模型选型](#21-底座模型选型)
  - [2.2 递归推理落地：ToT 与 RLM](#22-递归推理落地tot-与-rlm)
  - [2.3 上下文窗口管理](#23-上下文窗口管理)
  - [2.4 坑](#24-坑)
- [3. Memory 记忆层：RAG 隐身，混合检索 + 写回 + 遗忘](#3-memory-记忆层rag-隐身混合检索--写回--遗忘)
  - [3.1 读路径：混合检索](#31-读路径混合检索)
  - [3.2 写回与记忆分层](#32-写回与记忆分层)
  - [3.3 热度调度与遗忘](#33-热度调度与遗忘)
  - [3.4 坑](#34-坑)
- [4. Action 执行层：工具 + 代码 + 多 Agent 编排](#4-action-执行层工具--代码--多-agent-编排)
  - [4.1 工具协议与代码执行](#41-工具协议与代码执行)
  - [4.2 多 Agent 编排选型](#42-多-agent-编排选型)
  - [4.3 Human-in-the-loop](#43-human-in-the-loop)
- [5. Governance 治理层：企业级分水岭](#5-governance-治理层企业级分水岭)
- [6. 四层串联：一次请求的生命周期](#6-四层串联一次请求的生命周期)
- [7. 自研 vs 托管：2026 年的现实选择](#7-自研-vs-托管2026-年的现实选择)
- [8. 深度分析](#8-深度分析)
- [9. Checklist](#9-checklist)
- [10. 延伸阅读](#10-延伸阅读)

---

## 1. 架构全景：四层稳态

```
┌─────────────────────────────────────────────────┐
│           Governance 治理层（企业级分水岭）          │
│  RBAC+ABAC · 溯源落盘 · 评测 · 合规遗忘 · 成本治理    │
├─────────────────────────────────────────────────┤
│             Action 执行层（工具 + 代码 + 编排）      │
│  MCP 协议 · 代码沙箱 · 多 Agent 编排 · HITL         │
├─────────────────────────────────────────────────┤
│          Memory 记忆层（RAG 隐身，混合检索+写回）     │
│  向量+BM25+KG 三路检索 · Working/Episodic/Semantic  │
│  写回流水线 · 热度调度 · 遗忘                         │
├─────────────────────────────────────────────────┤
│          Brain 推理层（LLM + 递归推理编排）          │
│  Planner-Solver 分工 · ToT/RLM · 上下文窗口管理      │
└─────────────────────────────────────────────────┘
```

> [!abstract] 稳态的含义
> 每层接口稳定（MCP / OpenTelemetry / ReRank 协议），内部实现可替换。今天用 Qdrant 明天换 Milvus，今天用 GPT 明天换 Qwen，上层 Agent 代码不动——这才是 Context Engineering 统一掉的真正价值。

---

## 2. Brain 推理层：LLM + 递归推理编排

核心不是换模型，而是把"单跳生成"改成"**可回溯的推理循环**"。

### 2.1 底座模型选型

| 角色 | 选型 | 部署方式 |
|------|------|---------|
| **主干模型** | 闭源：GPT-5.1 / Claude Opus 4.5 / Gemini 3<br>开源权重：Qwen3-235B-A22B / DeepSeek-V3.2 / Llama4-Scout | vLLM / SGLang 自部署 |
| **小模型（路由/分类）** | Qwen3-0.6B / Phi-4-mini | 同上，可和主干共卡 |

### 2.2 递归推理落地：ToT 与 RLM

#### ToT（Tree of Thoughts）

用 LangGraph 的 `StateGraph` 把"生成候选 → 自评 → 剪枝 → 继续"写成节点，深度/分支数配**预算控制器**（token 上限触发强制收敛）。

```
StateGraph 节点设计：
  generate_candidates  →  self_evaluate  →  prune  →  [depth < max ? loop : converge]
                                            ↑
                                     budget_controller (token 上限强制收敛)
```

#### RLM（Reasoning Loop with Memory）

把每轮推理痕迹写回 Memory 层（见下文 §3），下一轮把"上轮为什么错"作为 context 注入——这是 2026 年比纯 CoT 更稳的做法。

```
轮 N: 推理 → 产出答案 → 自检发现错误 → 写 Memory(episodic, "轮N错因=XXX")
轮 N+1: 读 Memory → context 注入"上轮错因" → 规避同样路径 → 产出改进答案
```

#### Planner-Solver 分工

| 角色 | 模型 | 职责 |
|------|------|------|
| **Planner** | 小 LLM | 输出 JSON plan（任务拆解） |
| **Solver** | 主干 LLM | 按 plan 执行，不做元思考 |

> [!tip] 为什么要分工
> 避免主干被"元思考"拖慢。Planner 做规划、Solver 做执行，Token 消耗和延迟都显著降低。

### 2.3 上下文窗口管理

三策组合，防止递归累积爆窗：

| 策略 | 机制 | 适用场景 |
|------|------|---------|
| **Prompt Caching** | Claude / GPT 原生支持，缓存 system prompt + few-shot | system prompt 固定、长上下文 |
| **Sliding Window** | 只保留最近 N 轮对话 | 多轮对话 |
| **关键 Span 保留** | attention 归因裁剪，保留高权重 span | 长文档递归推理 |

### 2.4 坑

> [!danger] 递归层数必须配超时和 cost cap
> 否则长任务会螺旋烧钱。Planner 输出要做 **JSON Schema 校验**，别让自由文本 plan 进执行层——一个格式错的 plan 可能触发数十次无效工具调用。

---

## 3. Memory 记忆层：RAG 隐身，混合检索 + 写回 + 遗忘

> [!important] 这一层是重头戏
> RAG 从"管线"变成"服务"。读路径（检索）和写路径（记忆沉淀）必须分离设计。

### 3.1 读路径：混合检索

```
Query ──┬──→ 向量检索（语义相似）──┐
        ├──→ BM25 检索（关键词）──┤──→ RRF 融合 ──→ Rerank ──→ Top-K
        └──→ KG 检索（多跳关系）──┘
```

| 检索源 | 工具 | Embedding 模型 | 适用 |
|--------|------|---------------|------|
| **向量** | Qdrant / Milvus 2.4+ / pgvector（HA 场景） | 长文档：`bge-m3`（多语言+多粒度）<br>代码：`codesage`<br>表格：`tabformer` 类 | 语义相似匹配 |
| **BM25** | Elasticsearch / Tantivy | — | 关键词兜底匹配 |
| **KG** | Neo4j / NebulaGraph | — | 复杂多跳问题（"A 的供应商的母公司是谁"），走 Cypher / GraphRAG |
| **Rerank** | `bge-reranker-v2-m3` / Cohere Rerank 3 | — | 跨源重排，Top-K 动态（简单问题 K=3，复杂 K=8） |

> [!note] RRF（Reciprocal Rank Fusion）
> 向量和 BM25 结果做 Reciprocal Rank Fusion 融合，再和 KG 结果在 rerank 阶段合并。不要在检索阶段就合并——不同源的分数尺度不一致。

### 3.2 写回与记忆分层

| 记忆层 | 内容 | 存储 | Schema |
|--------|------|------|--------|
| **Working（工作记忆）** | 当前会话临时状态 | Redis / 内存 | TTL = 会话级 |
| **Episodic（情景记忆）** | "用户上次说不喜欢红色报告"这类事件流 | TimescaleDB / ClickHouse | `(user_id, ts, event_json)` |
| **Semantic（语义记忆）** | 沉淀出的稳定事实（用户画像、项目背景） | 向量库 + KG | Mem0 `add_memory` 接口直接对接 |

#### 写回流水线

```
Agent 每轮结束
  → Memory Extractor（小模型抽取事实）
  → 冲突检测（和已有记忆余弦比对）
  → 写入对应层（Working / Episodic / Semantic）
```

> [!info] 托管 vs 自研
> MemoryOS / Mem0 / Zep 本质是这三条流水线的托管版。自研就照这个画——核心是抽取器 + 冲突检测 + 分层写入。

### 3.3 热度调度与遗忘

**热度调度：**

```
向量库里给每个 chunk 打 access_count 和 last_access
  → 冷数据降级到对象存储（OSS/S3）
  → 热数据留 GPU 显存侧缓存
```

**遗忘：**

```
forget(user_id, scope, reason) API
  → 物理删除向量 chunk
  → KG 里断边
  → 写审计日志（合规必须）
```

### 3.4 坑

> [!danger] 坑一：写回比检索难
> 抽取模型抽错事实比检不出来更致命。**抽取器必须和主干模型隔离部署 + 人工抽检通道**。建议每周抽 100 条写回记忆做人工审核，错误率 > 5% 就要调抽取 prompt 或换模型。

> [!danger] 坑二：多租户隔离
> 向量库用 `partition key=user_id`，KG 用 `tenant_label`。**别靠应用层过滤**——漏一个 if 就串数据。这是合规红线。

---

## 4. Action 执行层：工具 + 代码 + 多 Agent 编排

### 4.1 工具协议与代码执行

| 维度 | 选型 | 要点 |
|------|------|------|
| **工具协议** | MCP（Model Context Protocol）—— 2026 事实标准 | 每个工具暴露 JSON Schema，Agent 通过 MCP client 调用；老系统用 OpenAPI → MCP 适配层 |
| **代码执行沙箱** | E2B / Pyodide / gVisor 隔离 | 禁止网络出口白名单外访问；代码生成 → 静态检查（Bandit/ESLint）→ 沙箱跑 → 结果回收，全程流式 |

### 4.2 多 Agent 编排选型

| 编排框架 | 适用场景 | 特点 |
|---------|---------|------|
| **LangGraph** | 有状态、可断点续跑的流程（客服、报销审批） | StateGraph + Checkpoint |
| **AutoGen GroupChat** | 角色扮演式协作（一个提方案、一个挑刺、一个汇总） | 对话驱动 |
| **CrewAI** | 轻量任务制（检索 Agent / 写稿 Agent / 校验 Agent 组队） | 角色化声明式 |

> [!tip] 编排层原则
> 编排层**只做任务分发和结果聚合**，每个子 Agent 内部再跑自己的 Brain + Memory 小循环。不要把业务逻辑写在编排层。

> [!seealso] 框架深度对比
> 三个框架的代码级对比和选型决策，见 [[02-LangGraph与CrewAI实战对比|LangGraph 与 CrewAI 实战对比]] 和 [[06-生产级框架选型批判性决策指南|生产级框架选型批判性决策指南]]。

### 4.3 Human-in-the-loop

> [!warning] 企业落地硬开关
> 敏感动作（发邮件、提交工单、转账）必须挂 `interrupt_before` 节点，等人确认再继续。这是企业落地的**硬开关**，不是可选项。

LangGraph 原生支持 `interrupt_before` 机制，CrewAI 需自行实现。详见 [[02-LangGraph与CrewAI实战对比]]。

---

## 5. Governance 治理层：企业级分水岭

> [!important] 纯技术 Demo 可以没有这层，进生产必须有

| 维度 | 落地方案 | 关键点 |
|------|---------|--------|
| **权限** | RBAC + ABAC 双控，工具调用前过 OpenPolicyAgent（Rego 规则） | "能查 KG" ≠ "能查 HR 子图" |
| **溯源** | 每次生成保留 `(query, retrieved_chunks_with_score, model_version, tool_calls, prompt_hash)`，存 ClickHouse | 出事能复现。用 LangSmith / Arize Phoenix / 自研 trace 服务接 OpenTelemetry |
| **评测** | 离线：Ragas / DeepEval（事实一致性、上下文召回）<br>在线：用户点赞/纠正作为 reward 信号回灌 Memory 和 Planner 微调 | 离线 + 在线双轨 |
| **合规遗忘** | GDPR / 个保法要求"删除我的数据"——必须能 cascade 清：向量库 chunk 删、KG 边断、情景记忆事件清、trace 日志匿名化 | Mem0 的 `delete_user` 要自己扩写到全链路 |
| **成本治理** | 按 user/agent/tool 维度计量 token 和调用次数 | 超预算自动降级（复杂 Planner 换小模型） |

> [!danger] 合规遗忘是最容易漏的
> 很多团队只做了向量库删除，忘了 KG 断边和 trace 日志匿名化。GDPR 罚款按"数据泄露事件"计，一个未清理的 trace 日志可能就是一张罚单。

---

## 6. 四层串联：一次请求的生命周期

```
用户 Query
  → Governance:  鉴权 + 预算检查
  → Brain.Planner:  拆 plan ("先查客户合同KG, 再调CRM API, 最后生成摘要")
  → Memory:  读用户语义记忆 + 混合检索相关文档 (RAG 隐身调用)
  → Brain.Solver:  带 plan + context 递归推理 (ToT 2 层)
  → Action:  并行调 KG 查询工具 + CRM MCP 工具 + 代码沙箱算账
  → Memory:  抽取本轮新事实, 冲突检测, 写回情景/语义层
  → Brain:  合成最终回答
  → Governance:  溯源落盘 + 敏感词/泄露扫描
  → 返回 + 前端展示引用片段 (可点击溯源)
```

> [!note] 关键设计点
> - RAG 在这个流程里是**隐身调用**——Memory 层自动决定检索什么、检索多少，Agent 业务代码不感知 RAG 的存在
> - Action 层的三个工具调用是**并行**的，不是串行——Planner 拆 plan 时就标注了哪些步骤可并行
> - Memory 写回在 Action 之后、Brain 合成之前——确保本轮新事实能被合成阶段使用

---

## 7. 自研 vs 托管：2026 年的现实选择

| 场景 | Brain | Memory | Action | Governance | 周期 |
|------|-------|--------|--------|-----------|------|
| **创业/中型公司** | 托管 LLM API | Mem0 管记忆 + Qdrant + Neo4j | LangGraph 编排 + MCP | LangSmith 观测 | **3 人月** MVP |
| **金融/政务/大厂** | 自部 Qwen3-235B 类权重 | **自研**（必须掌握写回和遗忘代码） | MCP 网关自研 | 治理层过等保 | **6-12 个月**，数据和合规不出门 |

> [!tip] 别自研的部分
> Rerank 模型、Embedding 模型、沙箱运行时——这些用开源权重微调比从零写划算 **10 倍**。自研精力应该花在 Memory 写回流水线和 Governance 治理层上，这才是核心竞争力。

---

## 8. 深度分析

### 8.1 四层架构与既有架构的关系

本篇的四层（Brain/Memory/Action/Governance）是从**工程落地视角**的切分，与知识库既有架构笔记的对应关系：

| 本篇四层 | 对应既有笔记 | 视角差异 |
|---------|------------|---------|
| **Brain** | [[01-Agent架构演进]] | 01 讲历史演进（Function Calling → ReAct → ToT），本篇讲 2026 年怎么用 LangGraph StateGraph 落地 ToT |
| **Memory** | [[10-AgentOS架构与运行时]] §分层记忆 | 10 从 OS 类比角度讲 L1/L2/L3 记忆，本篇从工程拼法角度讲 Working/Episodic/Semantic + 写回流水线 |
| **Action** | [[02-Multi-Agent协作架构]] + [[03-Agent协议与通信架构]] | 02/03 讲协作模式和协议设计，本篇讲具体编排框架选型（LangGraph/AutoGen/CrewAI）+ MCP 工具网关 + 代码沙箱 |
| **Governance** | [[07-Agent安全防护体系]] | 07 专注安全攻防，本篇从企业治理角度补充 RBAC/溯源/评测/合规遗忘/成本治理 |

### 8.2 为什么是四层而不是三层或五层

```
三层方案（缺 Governance）：
  Brain + Memory + Action
  → 纯技术 Demo 够用，进生产必塌（无权限、无溯源、无合规）

五层方案（过度拆分）：
  Brain + Memory + Action + Governance + Observability
  → Observability 本质是 Governance 的子域，拆出来增加接口复杂度
  → 四层是工程上的最小完备集
```

> [!important] 四层是稳态，不是终态
> 四层的接口边界（MCP / OpenTelemetry / Rerank 协议）已经稳定，但每层内部实现仍在快速演进。2026 年的最大变化是 Memory 层从"向量库 + 检索脚本"升级为"MemoryOS 托管服务"——未来可能像数据库一样成为独立基础设施。

### 8.3 与框架选型的关系

本篇的四层架构是**框架无关**的——无论用 LangGraph、CrewAI 还是 MAF，都需要这四层。框架选型决定的是 Action 层的编排方式和 Brain 层的推理循环实现，但 Memory 和 Governance 层基本需要自研或独立采购。

> [!seealso] 选型方法论
> 框架选型不能只看 Action 层能力，必须综合考虑四层全栈。详见 [[06-生产级框架选型批判性决策指南]] 的五模式批判性决策——特别是"前期死后分析"部分，识别每个选型最可能的死因。

---

## 9. Checklist

### Brain 层
- [ ] Planner 和 Solver 是否分模型部署？主干 LLM 不做元思考
- [ ] ToT 递归深度是否配了 token 预算控制器？超限强制收敛
- [ ] Planner 输出是否做了 JSON Schema 校验？拒绝自由文本 plan 进执行层
- [ ] 上下文窗口管理是否三策组合（Caching + Sliding + Span 保留）？
- [ ] RLM 的推理痕迹是否写回 Memory 层？下一轮能读到"上轮错因"

### Memory 层
- [ ] 读路径是否三路混合检索（向量 + BM25 + KG）？RRF 融合后 Rerank
- [ ] Embedding 模型是否按数据类型分路（长文档/代码/表格各用专用模型）？
- [ ] 写回流水线是否有冲突检测？和已有记忆余弦比对
- [ ] Memory Extractor 是否和主干模型隔离部署？有人工抽检通道
- [ ] 多租户隔离是否在存储层做（partition key / tenant_label）？不靠应用层过滤
- [ ] 遗忘 API 是否 cascade 清理（向量 + KG + 情景 + trace）？

### Action 层
- [ ] 工具协议是否统一走 MCP？老系统有 OpenAPI → MCP 适配层
- [ ] 代码沙箱是否禁止白名单外网络出口？生成 → 静态检查 → 沙箱 → 回收
- [ ] 编排框架是否按场景选（LangGraph 有状态 / AutoGen 角色扮演 / CrewAI 轻量）？
- [ ] 敏感动作是否挂 `interrupt_before`？Human-in-the-loop 是硬开关

### Governance 层
- [ ] 权限是否 RBAC + ABAC 双控？工具调用前过 OpenPolicyAgent
- [ ] 溯源是否保留完整链路（query + chunks + model_version + tool_calls + prompt_hash）？
- [ ] 评测是否离线 + 在线双轨？在线 reward 信号回灌 Memory 和 Planner
- [ ] 合规遗忘是否 cascade 清理全链路？trace 日志也匿名化
- [ ] 成本治理是否按 user/agent/tool 维度计量？超预算自动降级

### 架构整体
- [ ] 四层接口是否协议化（MCP / OpenTelemetry / Rerank 协议）？内部实现可替换
- [ ] 是否区分了自研核心（Memory 写回 + Governance）和采购非核心（Rerank / Embedding / 沙箱）？
- [ ] MVP 周期是否控制在 3 人月（创业）或 6-12 个月（金融政务）？

---

## 10. 延伸阅读

### 本目录关联
- [[01-Agent架构演进]] — Brain 层的推理模式演进史（Function Calling → ReAct → ToT）
- [[02-Multi-Agent协作架构]] — Action 层多 Agent 协作模式的理论基础
- [[03-Agent协议与通信架构]] — MCP / A2A 协议设计，Action 层的协议基础设施
- [[07-Agent安全防护体系]] — Governance 层的安全攻防深度展开
- [[09-MCP协议2026演进与无状态传输]] — MCP 协议 2026 最新演进，Action 层工具协议
- [[10-AgentOS架构与运行时]] — Memory 层分层记忆架构的 OS 视角，以及无状态 Agent + Sidecar 设计

### 跨目录关联
- [[05-Agent脚手架与快速启动工具生态|../09-Agent框架生态/05-Agent脚手架与快速启动工具生态]] — Action 层编排框架的脚手架 CLI 工具（AgentStack / AgentX-Kit / Mastra）
- [[06-生产级框架选型批判性决策指南|../09-Agent框架生态/06-生产级框架选型批判性决策指南]] — 四层架构中 Action 层框架选型的批判性决策方法论
- [[01-框架格局总览2026|../09-Agent框架生态/01-框架格局总览2026]] — 2026 框架版图全景

### 外部资源
- [LangGraph StateGraph 文档](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [Mem0 文档](https://docs.mem0.ai/)
- [OpenPolicyAgent Rego 规则](https://www.openpolicyagent.org/docs/latest/policy-language/)
- [bge-m3 模型](https://huggingface.co/BAAI/bge-m3)
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

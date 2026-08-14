# AgentOS 技术组件选型指南

> AgentOS 每层用什么组件？vLLM vs SGLang、Qdrant vs Milvus、NATS vs Kafka、LangSmith vs Langfuse——2026 年生产级选型对比与落地建议

## 元数据
- **难度**: ⭐⭐⭐⭐
- **前置知识**: [[10-AgentOS架构与运行时]], [[11-生产级Agent四层架构工程落地]]
- **关联文件**: [[10-AgentOS架构与运行时]], [[11-生产级Agent四层架构工程落地]], [[13-十万用户级企业架构方案]], [[14-百万用户级超级架构方案]], [[15-亿级用户终极架构方案]], [[03-Agent协议与通信架构]], [[07-Agent安全防护体系]], [[09-MCP协议2026演进与无状态传输]], [[05-Agent脚手架与快速启动工具生态|../09-Agent框架生态/05-Agent脚手架与快速启动工具生态]]
- **最后更新**: 2026-08-14

---

## 目录

- [1. AgentOS 技术组件全景](#1-agentos-技术组件全景)
- [2. LLM 推理引擎层](#2-llm-推理引擎层)
  - [2.1 vLLM](#21-vllm)
  - [2.2 SGLang](#22-sglang)
  - [2.3 TensorRT-LLM](#23-tensorrt-llm)
  - [2.4 TGI / Triton / Ollama](#24-tgi--triton--ollama)
  - [2.5 推理引擎选型矩阵](#25-推理引擎选型矩阵)
- [3. LLM 路由层](#3-llm-路由层)
  - [3.1 三层路由架构](#31-三层路由架构)
  - [3.2 vLLM Production Stack](#32-vllm-production-stack)
  - [3.3 llm-d（CNCF Sandbox）](#33-llm-dcncf-sandbox)
  - [3.4 SAAR 会话感知路由](#34-saar-会话感知路由)
- [4. 记忆与存储层](#4-记忆与存储层)
  - [4.1 向量数据库选型](#41-向量数据库选型)
  - [4.2 工作记忆：Redis](#42-工作记忆redis)
  - [4.3 情景记忆：时序库](#43-情景记忆时序库)
  - [4.4 托管记忆服务](#44-托管记忆服务)
- [5. 消息总线](#5-消息总线)
- [6. 可观测性](#6-可观测性)
- [7. 代码沙箱](#7-代码沙箱)
- [8. 注册中心](#8-注册中心)
- [9. 安全治理组件](#9-安全治理组件)
- [10. 选型决策矩阵](#10-选型决策矩阵)
- [11. 深度分析](#11-深度分析)
- [12. Checklist](#12-checklist)
- [13. 延伸阅读](#13-延伸阅读)

---

## 1. AgentOS 技术组件全景

```
┌────────────────────────────────────────────────────────────────┐
│                     可观测性层                                   │
│  LangSmith / Langfuse / Arize Phoenix / MLflow / Datadog       │
├────────────────────────────────────────────────────────────────┤
│                     安全治理层                                   │
│  OpenPolicyAgent / Guardrails / 审计日志 / PII 脱敏              │
├────────────────────────────────────────────────────────────────┤
│                     编排与执行层                                 │
│  LangGraph / CrewAI / AutoGen  +  MCP Proxy / 代码沙箱          │
├────────────────────────────────────────────────────────────────┤
│                     记忆与存储层                                 │
│  向量库(Qdrant/Milvus) + Redis(工作记忆) + TimescaleDB(情景)    │
│  + Mem0/Zep(托管记忆) + Neo4j(KG)                               │
├────────────────────────────────────────────────────────────────┤
│                     LLM 路由层                                   │
│  vLLM Production Stack / llm-d / SAAR / K8s Gateway             │
├────────────────────────────────────────────────────────────────┤
│                     LLM 推理引擎层                               │
│  vLLM / SGLang / TensorRT-LLM / TGI / Triton                   │
├────────────────────────────────────────────────────────────────┤
│                     消息与协调层                                 │
│  NATS JetStream / Kafka / Redis Streams                         │
├────────────────────────────────────────────────────────────────┤
│                     注册与发现层                                 │
│  etcd / Consul / ZooKeeper                                      │
└────────────────────────────────────────────────────────────────┘
```

> [!abstract] 选型原则
> 接口稳定（OpenTelemetry / MCP / OpenAI API 兼容），实现可替换。每层选型独立，不跨层耦合。本文只覆盖 [[10-AgentOS架构与运行时]] 未展开的组件级选型对比。

---

## 2. LLM 推理引擎层

### 2.1 vLLM

**定位**：最广泛采用的开源推理引擎，事实标准。

| 维度 | 详情 |
|------|------|
| 核心创新 | PagedAttention（类虚拟内存管理 KV Cache）、Continuous Batching（请求随到随批） |
| API 兼容 | OpenAI 兼容，drop-in 替换 |
| 监控 | 内置 Prometheus `/metrics`（tokens/s, KV cache 利用率, TTFT, 队列深度） |
| 模型支持 | 新架构权重发布后通常**当天**支持 |
| 适用 | 通用推理、快速迭代、需要最广社区支持 |
| 短板 | 纯吞吐不及 TensorRT-LLM；结构化生成不如 SGLang |

### 2.2 SGLang

**定位**：LMSYS 出品，针对 Agent 工作负载优化的推理引擎。2026 年三大参考引擎之一（与 vLLM、TRT-LLM 并列），xAI Grok 推理 fleet 默认引擎。

| 维度 | 详情 |
|------|------|
| 核心创新 | **RadixAttention**——基数树索引整个 KV 池，跨无关请求自动共享前缀块，Agent/多租户场景吞吐 2-5x |
| 结构化生成 | 一等公民：`regex=` / `choices=` / JSON Schema 约束解码（XGrammar，单步微秒级 logit mask 更新） |
| Python DSL | 多调用程序（fork/join/并行 gen）编译为批量调度原语 |
| 高级能力 | FlashInfer 内核、FP8/FP4 量化、TP/EP 并行、EAGLE-2/Medusa 投机解码、多 LoRA 热切换 |
| 模型支持 | 新权重发布后 1-2 周支持（慢于 vLLM，快于 TRT-LLM） |
| 适用 | **Agent 循环**（共享工具脚手架）、多租户 chat（重叠 system prompt）、批量评估（共享 few-shot）、JSON 约束输出 |

> [!tip] vLLM vs SGLang 关键差异
> vLLM 优化广度和开发者体验；SGLang 优化 prompt 结构主导成本的场景。**Agent 工作负载（多轮 + 共享前缀 + 结构化输出）首选 SGLang**。

### 2.3 TensorRT-LLM

**定位**：NVIDIA 优化推理引擎，极限吞吐。

| 维度 | 详情 |
|------|------|
| 核心优势 | 编译为高度优化 GPU kernel，原始吞吐最高 |
| 代价 | 需要提前编译，模型迭代灵活性低 |
| 模型支持 | 月级更新周期，最慢 |
| 适用 | 固定模型 + 极致吞吐需求 + NVIDIA 硬件 |

### 2.4 TGI / Triton / Ollama

| 引擎 | 定位 | 适用 |
|------|------|------|
| **TGI**（HuggingFace） | 与 HF Hub 紧密集成 | 已用 HF 生态的组织 |
| **Triton**（NVIDIA） | 通用模型服务器，多框架（PyTorch/TF/ONNX/TRT） | 异构模型类型 |
| **Ollama** | 开发者友好，本地运行 | **仅本地开发**，无批量/监控/水平扩展，不生产级 |

### 2.5 推理引擎选型矩阵

| 维度 | vLLM | SGLang | TensorRT-LLM | TGI |
|------|------|--------|-------------|-----|
| **通用推理** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Agent 工作负载** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **结构化输出** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **模型迭代速度** | 当天 | 1-2 周 | 月级 | 1-2 周 |
| **社区规模** | 最大 | 快速增长 | 中 | 中 |
| **部署复杂度** | 低 | 低 | 高（需编译） | 低 |
| **硬件中立** | ✅ NVIDIA/AMD/Intel | ✅ NVIDIA/AMD | ❌ NVIDIA only | ✅ |

> [!example] 典型组合
> - **创业/MVP**：vLLM（社区最大，踩坑有人帮）
> - **Agent 密集型生产**：SGLang（RadixAttention 对 Agent 循环收益巨大）
> - **固定模型 + 极致吞吐**：TensorRT-LLM（编译一次跑半年）
> - **本地开发**：Ollama（别上生产）

---

## 3. LLM 路由层

单机推理不需要路由层。当你从 1 个 vLLM 实例扩展到多个时，路由层决定每个请求去哪。

### 3.1 三层路由架构

```
┌─────────────────────────────────────────────────────┐
│ L7: 推理感知路由（vLLM Stack / llm-d / SAAR）         │  ← 最智能
│   根据 KV cache 状态、前缀哈希、队列深度选 Pod          │
├─────────────────────────────────────────────────────┤
│ L7: API Gateway / Ingress                            │
│   HTTP 感知，按 model name / header 路由              │
├─────────────────────────────────────────────────────┤
│ L4: K8s Service (Round Robin)                        │  ← 最简单
│   TCP 负载均衡，不知道 Pod 内部状态                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 vLLM Production Stack

**定位**：vLLM 官方 K8s 生产部署参考栈（2025-01 发布）。

| 能力 | 说明 |
|------|------|
| 前缀缓存感知路由 | 请求路由到已缓存对应 KV 的 Pod，大幅提升缓存命中率 |
| KV Cache 卸载 | 集成 LMCache，GPU → CPU 两级缓存 |
| 一键部署 | 单条 Helm 命令拉起完整服务栈 |
| 开箱监控 | 内置 Prometheus + Grafana（TTFT / KV 命中率 / 吞吐） |
| 性能 | 相比裸 vLLM + KServe：吞吐 2-5x，延迟 3-10x 更低 |
| 限制 | KV Cache 卸载只到 CPU（无 NVMe 层）；Prefill/Decode 分离实验阶段 |
| 适用 | 已用 vLLM + 中小规模集群（几十张 GPU） |

### 3.3 llm-d（CNCF Sandbox）

**定位**：IBM/Red Hat/Google 联合捐献给 CNCF 的 K8s 原生分布式推理中间件（2026-03 进入 Sandbox）。底层默认 vLLM。

| 能力 | 说明 |
|------|------|
| **Prefill/Decode 分离** | Prefill Pod（算力优先）和 Decode Pod（带宽优先）独立部署、独立扩缩容 |
| **三级 KV Cache 卸载** | GPU HBM → CPU DRAM → NVMe SSD，支持 128K+ 长上下文 |
| **EPP 前缀感知路由** | Endpoint Picker 根据 prompt 前缀哈希路由到缓存命中 Pod |
| **MoE 宽专家并行** | K8s LeaderWorkerSet 跨节点分布 MoE Expert |
| **Scale-to-Zero** | 无流量时自动缩至 0 Pod |
| **硬件中立** | NVIDIA / AMD / Intel Gaudi / Google TPU |
| 性能（v0.5, Qwen3-32B, 16×H100） | 吞吐 ~120K tokens/s，GPU 利用率 40-60% → **80%+** |
| 适用 | 超大规模集群（百 GPU+）、多租户 SaaS、长上下文、异构硬件 |
| 限制 | CNCF Sandbox 阶段，API 可能有 breaking changes |

### 3.4 SAAR 会话感知路由

**定位**：vLLM Semantic Router 的会话感知模型选择策略（2026-06 发布）。解决长程 Agent 的路由问题。

| 维度 | 详情 |
|------|------|
| 核心问题 | 单轮 prompt 路由器不理解会话轨迹，Agent 中途切模型会破坏上下文连续性 |
| 解决方案 | 保持语义路由 + 新增路由器拥有的会话记忆 + 工具循环硬锁 + 前缀缓存感知切换定价 + 可重放 trace |
| 性能 | 21,600 确定性轮次：模型切换减少 **79.29%**，消除 3,836 次不安全切换，物理模型成本降低 **78.71%** |
| 关键规则 | **有时路由器绝不能切换**——工具循环中、非可移植 provider 状态存在时，硬锁不切 |
| 适用 | 长程 Agent（编码/研究 Agent），多模型混用场景 |

> [!important] 路由层选型要点
> - <10 GPU：vLLM Production Stack（简单可靠）
> - 10-100 GPU：vLLM Production Stack + 前缀缓存路由
> - 100+ GPU / 多租户 / 长上下文：llm-d（Prefill/Decode 分离 + 三级缓存）
> - 多模型混用 Agent：SAAR（会话感知，防止不安全切换）

---

## 4. 记忆与存储层

### 4.1 向量数据库选型

| 维度 | Qdrant | Milvus 2.4+ | Pinecone | Weaviate | pgvector |
|------|--------|-------------|----------|----------|----------|
| **开源** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **自托管** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **托管云** | ✅ | ✅ (Zilliz) | ✅ | ✅ | ✅ (云 PG) |
| **多租户** | ✅ payload 分区 | ✅ partition key | ✅ namespace | ✅ | 需应用层 |
| **混合检索** | ✅ 向量+BM25+过滤 | ✅ | ✅ | ✅ | ✅ (pg_trgm) |
| **性能** | 高（Rust） | 高（Go+C++） | 高（托管） | 中高 | 中（PG 扩展） |
| **适用** | 性能敏感 + 轻量易用 | 大规模生产 | 托管生产规模 | 语义搜索+GraphQL | 已有 PG 栈，精益部署 |

> [!tip] 选型建议
> - **从零开始 + 性能优先**：Qdrant（Rust 写的，轻量快）
> - **大规模生产 + 已有运维团队**：Milvus（成熟度高）
> - **不想运维**：Pinecone（纯托管）
> - **已有 PostgreSQL**：pgvector（不引入新组件，适合 MVP）
> - **Agent 多租户场景**：Qdrant 或 Milvus（原生支持分区，不靠应用层过滤）

### 4.2 工作记忆：Redis

**定位**：Agent 记忆栈中覆盖最广的单一组件——一个 Redis 部署可同时处理三种记忆模式。

| 记忆模式 | Redis 实现方式 | 其他方案对比 |
|---------|---------------|-------------|
| **短期回忆**（当前会话上下文） | ✅ 内存原生数据结构（Hash / Sorted Set） | Pinecone ❌ / MongoDB 磁盘延迟高 |
| **长期检索**（跨会话语义记忆） | ✅ Redis Query Engine（HNSW / FLAT / SVS-VAMANA 索引） | Pinecone 更专注但只覆盖此模式 |
| **操作状态**（队列/锁/计数器/限流） | ✅ 原生数据结构 + Streams + Pub/Sub | 向量库均 ❌ |

| 性能指标 | Redis 数据 |
|---------|-----------|
| 十亿向量基准 | 90% 精度，~200ms 中位延迟（50 并发查询） |
| 读写延迟 | 亚毫秒级 |
| 语义缓存 | LangCache（语义等价缓存，非精确匹配） |

> [!note] Redis vs 专用向量库
> Redis 一个部署 = 向量库 + 缓存 + 状态存储。Pinecone/MongoDB/Weaviate 各自只覆盖 1-2 种模式，实际生产中需要拼装多个系统。**小规模 Agent（< 千万向量）可只用 Redis 搞定全栈记忆**。

### 4.3 情景记忆：时序库

| 方案 | 适用 | 特点 |
|------|------|------|
| **TimescaleDB** | 已有 PG 栈 | PG 扩展，时序查询原生支持，和 pgvector 同库 |
| **ClickHouse** | 大规模事件流 | 列式存储，分析查询极快，Langfuse 自托管也用它 |

情景记忆 Schema：`(user_id, ts, event_json, agent_id, session_id)`

### 4.4 托管记忆服务

| 服务 | 定位 | 核心接口 |
|------|------|---------|
| **Mem0** | 记忆即服务，三层流水线托管版 | `add_memory` / `search_memory` / `delete_user` |
| **Zep** | 长期记忆 + 时间感知 | 时序图 + 向量检索 |
| **MemoryOS** | 分层记忆 OS 抽象 | Working / Episodic / Semantic 统一接口 |

> [!warning] 托管 vs 自研
> Mem0 / Zep / MemoryOS 本质是「抽取器 + 冲突检测 + 分层写入」三条流水线的托管版。自研就照这个画——核心是 Memory Extractor（小模型抽事实）+ 冲突检测（余弦比对）+ 分层写入。详见 [[11-生产级Agent四层架构工程落地]] §3.2。

---

## 5. 消息总线

| 维度 | NATS JetStream | Kafka | Redis Streams |
|------|---------------|-------|---------------|
| **定位** | 轻量高性能消息系统 | 分布式事件流平台 | Redis 原生流 |
| **吞吐** | 100K+ msg/s | 百万级 msg/s | 高（受限于单节点） |
| **持久化** | ✅ JetStream | ✅ 不可变日志 | ✅ AOF/RDB |
| **复杂度** | 低（单二进制） | 高（Zookeeper/KRaft + Broker 集群） | 低（已有 Redis 即可用） |
| **运维成本** | 极低 | 高 | 低 |
| **生态** | 原生 K8s 友好 | 最成熟 | 已有 Redis 零额外成本 |
| **适用** | Agent 间通信（[[10-AgentOS架构与运行时]] 默认选型） | 大规模事件流 + 审计日志 | 已有 Redis + 小规模 Agent 通信 |

> [!tip] 选型建议
> - **Agent 间通信（< 100 节点）**：NATS JetStream（轻量、K8s 原生、[[10-AgentOS架构与运行时]] 默认）
> - **审计日志 / 事件溯源**：Kafka（不可变日志 + 成熟生态）
> - **已有 Redis + 不想引入新组件**：Redis Streams（够用）
> - **混合方案**：NATS 做 Agent 通信 + Kafka 做审计日志（[[10-AgentOS架构与运行时]] 架构图即此方案）

---

## 6. 可观测性

> [!important] 2026 关键变化
> 所有主流平台都已支持 OpenTelemetry / OpenInference，**instrumentation 不再是锁定决策**——你可以 emit 一次 trace，后续路由到不同后端。真正的决策是「在 OTel trace 之上你要什么层」。

### 6.1 平台对比矩阵

| 维度 | LangSmith | Langfuse | Arize Phoenix | MLflow | Braintrust | Datadog |
|------|-----------|----------|---------------|--------|------------|---------|
| **开源** | ❌ 专有 | ✅ MIT | ✅ ELv2 | ✅ Apache 2.0 | ❌ 专有 | ❌ 专有 |
| **自托管** | Enterprise only | ✅（PG+ClickHouse+Redis+对象存储） | ✅ 单容器 | ✅ 简单 | ❌ | ❌ |
| **OTel 原生** | 部分（ingest） | 部分（ingest） | ✅ | ✅ | 部分（ingest） | ✅ |
| **LangChain 集成** | 最深（原生） | 60+ 框架 | 40+ (OpenInference) | 60+ 框架 | 50+ 框架 | LLM 附加 |
| **评估（evals）** | 在线+离线 | 在线 judge | 离线库 + 在线(AX) | 完整 eval 平台 | 评估优先设计 | 基础 |
| **Agent 部署** | ✅ 内置 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **数据保留** | 14天免费/400天付费 | 30天免费/3年付费 | 7天免费/15天付费 | 无限（自托管） | 14天/30天 | 15天 |
| **ClickHouse 原生** | SmithDB（自研） | ✅ | ❌ | ❌ | ❌ | ❌ |
| **成本治理** | ✅ | ❌ | ❌ | ✅ AI Gateway | ✅ Gateway | ✅ |

### 6.2 选型建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| **LangChain/LangGraph 深度用户** | LangSmith | 原生集成最深，traces 秒级可见 |
| **开源自托管 + 数据主权** | Langfuse | MIT 协议 + ClickHouse 原生，生产级自托管 |
| **快速本地调试 + 离线 evals** | Arize Phoenix | `pip install` + `register()` 即用，单容器 |
| **完整 AI 工程平台** | MLflow | tracing + eval + prompt 优化 + 治理，无企业付费墙 |
| **评估优先开发** | Braintrust | 回归测试 + 非技术友好 |
| **已有 Datadog APM** | Datadog Agent Obs | 与现有基础设施监控关联 |

> [!danger] Phoenix 的扩展天花板
> Phoenix 单容器自托管很方便，但 >2 亿 span 后会遇到 ops 瓶颈：Postgres 无分片支持、BulkInserter 连接池耗尽风险（GitHub #12358）、大数据集导出 504（#11873）。大规模生产请用 Langfuse（ClickHouse 后端）或 LangSmith（SmithDB 自研）。

---

## 7. 代码沙箱

| 方案 | 隔离级别 | 适用 | 特点 |
|------|---------|------|------|
| **E2B** | 云沙箱 | 生产级代码执行 | 托管服务，毫秒级启动，预装环境 |
| **gVisor** | 内核级 | K8s Pod 安全隔离 | Google 出品，系统调用拦截，[[10-AgentOS架构与运行时]] 默认 |
| **Firecracker** | microVM | 强隔离 + 多租户 | AWS 出品，Lambda 底层，启动 125ms |
| **Pyodide** | WASM 沙箱 | 浏览器端 / 轻量 Python | 无网络出口，安全但能力有限 |

> [!tip] 选型建议
> - **生产级 Agent 代码执行**：E2B（托管，省心）
> - **K8s 自建 + 安全优先**：gVisor（[[10-AgentOS架构与运行时]] 默认）
> - **强隔离多租户 SaaS**：Firecracker microVM
> - **浏览器端 / 无服务端**：Pyodide（WASM）

---

## 8. 注册中心

| 维度 | etcd | Consul | ZooKeeper |
|------|------|--------|-----------|
| **一致性** | 强一致（Raft） | 强一致（Raft） | 强一致（ZAB） |
| **Watch 机制** | ✅ | ✅（long polling） | ✅（watcher） |
| **Lease/TTL** | ✅ 原生 | ✅（session + check） | ✅（ephemeral node） |
| **K8s 原生** | ✅（K8s 底层存储） | ❌（需额外集成） | ❌ |
| **运维复杂度** | 中 | 低（自带 UI） | 高（JVM 栈 + 慢） |
| **适用** | AgentOS 首选（[[10-AgentOS架构与运行时]] 默认） | 服务发现 + 健康检查 | 老系统兼容 |

> [!note] 为什么 AgentOS 用 etcd
> 1. **强一致性**：调度决策基于准确的 Agent 列表
> 2. **Watch 机制**：调度器和路由实时感知 Agent 变化
> 3. **Lease 机制**：心跳超时自动清理（Agent 宕机自动标记 DOWN）
> 4. **K8s 原生**：已在 K8s 集群中运行，零额外组件
> 5. **数据量小**：千/万级 Agent 元数据，不是大 value

---

## 9. 安全治理组件

| 维度 | 选型 | 用途 |
|------|------|------|
| **策略引擎** | OpenPolicyAgent（Rego 规则） | RBAC + ABAC 双控，工具调用前过策略 |
| **Guardrails** | NeMo Guardrails / Llama Guard / 自研 | Prompt 注入防护、PII 脱敏、输出过滤 |
| **审计日志** | Kafka 不可变日志 + ClickHouse 查询 | 每次工具调用可追溯 `(agent_id, intent, params, result)` |
| **身份认证** | OAuth2 / JWT / Entra ID | Agent 身份证书 + 意图级权限 |
| **密钥管理** | HashiCorp Vault / KMS | API Key / 模型凭证管理 |

> [!seealso] 安全深度
> 安全攻防的完整设计见 [[07-Agent安全防护体系]]，本篇只列组件选型。

---

## 10. 选型决策矩阵

### 10.1 按规模选型

| 组件层 | 创业/MVP（< 10 GPU） | 中型（10-100 GPU） | 大型（100+ GPU） |
|--------|---------------------|-------------------|-----------------|
| **推理引擎** | vLLM | vLLM / SGLang | SGLang（Agent）+ TRT-LLM（固定模型） |
| **路由层** | K8s Service（L4） | vLLM Production Stack | llm-d + SAAR |
| **向量库** | pgvector / Redis | Qdrant | Milvus / Qdrant 集群 |
| **工作记忆** | Redis | Redis | Redis 集群 |
| **情景记忆** | TimescaleDB | TimescaleDB | ClickHouse |
| **消息总线** | Redis Streams | NATS JetStream | NATS + Kafka（审计） |
| **可观测性** | Phoenix（单容器） | Langfuse（自托管） | LangSmith / Langfuse 企业版 |
| **沙箱** | Pyodide | gVisor | Firecracker |
| **注册中心** | etcd | etcd | etcd 集群 |

### 10.2 按场景选型

| 场景 | 推理引擎 | 路由 | 向量库 | 可观测 |
|------|---------|------|--------|--------|
| **Agent 密集型** | SGLang（RadixAttention） | SAAR（会话感知） | Qdrant（多租户） | LangSmith（Agent trace） |
| **RAG 知识库** | vLLM | vLLM Stack（前缀路由） | Milvus（大规模） | Langfuse（自托管） |
| **多模型混用** | vLLM + SGLang 混部 | SAAR | Redis（全栈记忆） | MLflow（多模型管理） |
| **金融/政企合规** | vLLM（自部署） | llm-d（K8s 原生） | Qdrant（自托管） | Langfuse（MIT + 自托管） |

---

## 11. 深度分析

### 11.1 为什么 SGLang 在 Agent 场景胜出

```
Agent 循环的 prompt 结构：
  Turn 1: [system_prompt][tool_scaffolding][user_query] → LLM
  Turn 2: [system_prompt][tool_scaffolding][tool_result_1][user_query] → LLM
  Turn 3: [system_prompt][tool_scaffolding][tool_result_1][tool_result_2][user_query] → LLM
           ↑────────────── 共享前缀 ──────────────↑

vLLM:  每轮重新计算 system_prompt + tool_scaffolding（除非命中 prefix cache）
SGLang: RadixAttention 自动跨轮共享前缀 KV 块，无需显式缓存管理

结果：Agent 工作负载吞吐 2-5x（来自 SGLang 官方基准）
```

### 11.2 Redis 全栈记忆 vs 多组件拼装

```
方案 A：Redis 全栈（小规模）
  Redis ─→ 工作记忆（Hash/Sorted Set）
       ─→ 语义记忆（HNSW 向量索引）
       ─→ 操作状态（Streams/Pub-Sub/锁）
  优点：1 个部署 = 3 种模式，无跨系统同步
  限制：< 千万向量级别，超大规模需要专用向量库

方案 B：多组件拼装（大规模）
  Redis ────→ 工作记忆 + 操作状态
  Qdrant ───→ 语义记忆（向量检索）
  TimescaleDB → 情景记忆（事件流）
  Neo4j ────→ 关系记忆（KG）
  优点：每个组件最优解
  代价：4 个系统 = 4 个故障点 + 跨系统一致性
```

> [!important] 决策规则
> - 向量规模 < 1000 万 + 团队 < 5 人 → **Redis 全栈**（省运维）
> - 向量规模 > 1000 万 或 需要专用能力（KG / 全文搜索）→ **多组件拼装**

### 11.3 可观测性的 OTel 标准化

2026 年最大的变化是 **OpenTelemetry + OpenInference 成为事实标准**：

```
OpenInference 定义的 10 种 span kind：
  LLM / EMBEDDING / RETRIEVER / RERANKER / TOOL / CHAIN / AGENT / GUARDRAIL / EVALUATOR / PROMPT

所有主流框架（LangChain/LangGraph/LlamaIndex/DSPy/OpenAI Agents SDK/CrewAI）
→ 自动 emit OTel trace
→ 可路由到任意后端（LangSmith/Langfuse/Phoenix/MLflow/Datadog）
```

这意味着：
1. **instrumentation 不再是锁定决策**——先埋点，后选后端
2. **迁移成本大幅降低**——换可观测平台不需要改业务代码
3. **选型焦点从「能不能集成」转向「trace 之上要什么能力」**——evals、成本治理、Agent 部署

---

## 12. Checklist

### 推理引擎
- [ ] Agent 工作负载是否考虑了 SGLang（RadixAttention 前缀共享）？
- [ ] 固定模型 + 极致吞吐是否评估了 TensorRT-LLM（需提前编译）？
- [ ] 本地开发是否用 Ollama（别上生产）？

### 路由层
- [ ] 多实例部署时是否用了推理感知路由（不只是 K8s Service L4）？
- [ ] 前缀缓存感知路由是否启用（vLLM v0.9.2+ 默认开启前缀缓存）？
- [ ] 长程 Agent 多模型混用是否考虑了 SAAR（防止不安全切换）？

### 记忆与存储
- [ ] 向量库是否在存储层做多租户隔离（partition key），不靠应用层过滤？
- [ ] 小规模是否考虑了 Redis 全栈记忆（避免多组件拼装）？
- [ ] 情景记忆是否用了时序库（TimescaleDB / ClickHouse），不是关系库？
- [ ] 记忆写回流水线是否有冲突检测 + 人工抽检通道？

### 消息总线
- [ ] Agent 间通信是否用了 NATS（K8s 原生 + 轻量）？
- [ ] 审计日志是否用了 Kafka（不可变日志 + 成熟生态）？
- [ ] 已有 Redis + 小规模是否用 Redis Streams（省组件）？

### 可观测性
- [ ] 是否用了 OpenTelemetry / OpenInference 埋点（保证后端可换）？
- [ ] 自托管是否考虑了 Langfuse（MIT + ClickHouse）而非 Phoenix（>2 亿 span 有瓶颈）？
- [ ] 评估能力是否在线 + 离线双轨（不只看 trace，还要能 eval）？
- [ ] 成本治理是否按 user/agent/tool 维度计量 token？

### 沙箱与安全
- [ ] 代码执行是否在沙箱内（gVisor / E2B / Firecracker）？
- [ ] 沙箱是否禁止白名单外网络出口？
- [ ] 工具调用前是否过 OpenPolicyAgent 策略引擎？
- [ ] 审计日志是否不可变（Kafka）+ 可查询（ClickHouse）？

---

## 13. 延伸阅读

### 本目录关联
- [[10-AgentOS架构与运行时]] — AgentOS 架构设计、Registry/Scheduler/Operator 实现
- [[11-生产级Agent四层架构工程落地]] — Brain/Memory/Action/Governance 四层工程拼法
- [[03-Agent协议与通信架构]] — MCP/A2A 协议，Agent 间通信基础设施
- [[07-Agent安全防护体系]] — 安全攻防深度展开
- [[09-MCP协议2026演进与无状态传输]] — MCP 协议 2026 最新演进

### 跨目录关联
- [[05-Agent脚手架与快速启动工具生态|../09-Agent框架生态/05-Agent脚手架与快速启动工具生态]] — 编排框架选型（LangGraph/CrewAI/AutoGen）
- [[02-向量数据库架构深度|../05-AI基础设施/05-向量数据库架构深度]] — 向量数据库架构原理
- [[02-模型服务与推理优化|../05-AI基础设施/02-模型服务与推理优化]] — 推理优化技术
- [[02-可观测性与治理|../06-LLMOps体系/02-可观测性与治理]] — LLMOps 可观测性治理

### 外部资源
- [vLLM 文档](https://docs.vllm.ai/)
- [SGLang 文档](https://docs.sglang.ai/)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)
- [llm-d（CNCF）](https://github.com/llm-d/llm-d)
- [SAAR: Session-Aware Agentic Routing](https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing)
- [Qdrant AI Agents](https://qdrant.tech/ai-agents/)
- [Redis Agent Memory](https://redis.io/blog/best-databases-for-agent-memory/)
- [Langfuse](https://langfuse.com/)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [OpenInference 规范](https://github.com/Arize-ai/openinference)

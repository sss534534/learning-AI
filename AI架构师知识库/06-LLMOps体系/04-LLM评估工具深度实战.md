# LLM 评估工具深度实战

> RAGAS 不是唯一的答案。DeepEval、TruLens、Phoenix、LangKit 各有侧重。
> 行业顶尖架构师需要知道在什么场景用什么工具、如何组合。

## 目录

1. [评估工具生态全景](#1-评估工具生态全景)
2. [DeepEval：超越 RAGAS 的全能评估框架](#2-deepeval超越-ragas-的全能评估框架)
3. [TruLens：RAG 应用的可观测性评估](#3-trulensrag-应用的可观测性评估)
4. [Phoenix (Arize)：LLM 可观测性与评估一体化](#4-phoenix-arizellm-可观测性与评估一体化)
5. [LangKit：文本质量与安全评估](#5-langkit文本质量与安全评估)
6. [工具对比与选型指南](#6-工具对比与选型指南)
7. [生产级评估架构](#7-生产级评估架构)

---

## 1. 评估工具生态全景

```
                    LLM 评估工具体系
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼──────┐    ┌──────▼──────┐
   │ RAG评估  │      │ 通用LLM评估  │    │ 可观测性     │
   │          │      │              │    │ + 评估      │
   │ · RAGAS  │      │ · DeepEval   │    │ · Phoenix   │
   │ · TruLens │     │ · LangKit    │    │ · LangSmith │
   │ · ARES   │      │ · Promptfoo  │    │ · Braintrust│
   └─────────┘      └──────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │  基准测试    │
                    │              │
                    │ · MT-Bench  │
                    │ · MMLU      │
                    │ · HumanEval │
                    │ · SWE-bench │
                    └─────────────┘
```

---

## 2. DeepEval：超越 RAGAS 的全能评估框架

### 2.1 为什么需要 DeepEval

RAGAS 专注 RAG 场景的 4 个核心指标。DeepEval 覆盖更广：

| RAGAS | DeepEval |
|-------|----------|
| 仅 RAG 评估 | RAG + 对话 + 安全 + 幻觉 + 偏见 |
| 需要自行集成 CI/CD | 原生 CI/CD 支持 |
| 无内置基准测试 | 集成 MMLU/HellaSwag/BIG-bench |
| 有限的自定义指标 | 灵活的自定义指标框架 |

### 2.2 快速上手

```python
# 安装
# pip install deepeval

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase

# 定义测试用例
test_case = LLMTestCase(
    input="解释一下OSPF协议的工作原理",
    actual_output="OSPF是一种链路状态路由协议...(详细解释)",
    expected_output="OSPF通过交换LSA来构建网络拓扑图...",
    context=["OSPF协议规范RFC 2328...", "链路状态路由协议对比..."],
    retrieval_context=["检索到的文档1...", "检索到的文档2..."]
)

# 多指标并行评估
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    HallucinationMetric(threshold=0.3),  # 幻觉分数越低越好
    BiasMetric(threshold=0.3),
    ToxicityMetric(threshold=0.1),
]

results = evaluate([test_case], metrics)
print(results)
```

### 2.3 G-Eval: 自定义评估指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# 用自然语言定义评估标准
correctness_metric = GEval(
    name="Correctness",
    criteria="判断实际输出是否与预期输出在关键事实上一致",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8
)

# 针对网管场景的自定义指标
troubleshooting_quality = GEval(
    name="TroubleshootingQuality",
    criteria="""
    评估网络故障诊断回复的质量：
    1. 是否准确定位了根因（0-3分）
    2. 是否给出了可操作的修复步骤（0-3分）
    3. 是否说明了影响范围（0-2分）
    4. 是否给出了预防建议（0-2分）
    总分10分，6分及格
    """,
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6
)
```

### 2.4 CI/CD 集成

```python
# deepeval 原生支持 pytest 集成
# test_agent.py
import pytest
from deepeval import assert_test

@pytest.mark.parametrize("test_case", load_test_cases("agent_eval_set.json"))
def test_agent_response(test_case):
    result = agent.run(test_case.input)
    test_case.actual_output = result.answer
    test_case.retrieval_context = result.retrieved_docs
    
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        HallucinationMetric(threshold=0.3),
    ])

# 在 CI 中运行
# $ deepeval test run test_agent.py
# 输出: ✅ 45/50 通过 | ❌ 5 退化 | 准确率: 90%
```

---

## 3. TruLens：RAG 应用的可观测性评估

### 3.1 TruLens 核心概念

TruLens 不是纯评估工具——它是 **RAG 应用的可观测性 + 评估**一体化平台：

```
                    TruLens 三件套
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
  ┌─────▼─────┐   ┌───────▼───────┐  ┌──────▼──────┐
  │ Feedback   │   │   Tracking    │  │  Dashboard  │
  │ Functions  │   │               │  │             │
  │            │   │ 记录完整的     │  │ 可视化追踪   │
  │ 评估函数    │   │ RAG 调用链     │  │ 和对比       │
  │ (RAG三连)  │   │ (App + Eval)  │  │             │
  └───────────┘   └───────────────┘  └─────────────┘
```

### 3.2 RAG 三连评估

```python
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider import OpenAI

# 初始化
tru = Tru()
provider = OpenAI()

# 定义反馈函数 (Feedback Functions)
f_groundedness = Feedback(
    Groundedness(groundedness_provider=provider),
    name="Groundedness"  # 回答是否基于检索到的文档
).on_output().on(TruLlama.select_source_nodes().text)

f_answer_relevance = Feedback(
    provider.relevance,
    name="Answer Relevance"  # 回答与问题的相关性
).on_input_output()

f_context_relevance = Feedback(
    provider.qs_relevance,
    name="Context Relevance"  # 检索文档与问题的相关性
).on_input().on(TruLlama.select_source_nodes().text)

# 包装你的RAG应用
tru_recorder = TruLlama(
    your_rag_app,
    app_id="RAG-v2.3",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# 运行并自动记录
with tru_recorder as recording:
    response = your_rag_app.query("OSPF和BGP的区别是什么？")

# 启动Dashboard查看结果
# $ tru run dashboard
# 浏览器打开 http://localhost:8501
```

### 3.3 TruLens vs RAGAS

| 维度 | RAGAS | TruLens |
|------|-------|---------|
| 评估方式 | 批量评估（离线） | 实时+批量 |
| 可视化 | 无内置 | Streamlit Dashboard |
| 追踪 | 需要自行集成 | 自动追踪RAG管道 |
| 生产集成 | 需要额外工作 | 原生支持 |
| 学习曲线 | 低 | 中 |
| 适用阶段 | 开发阶段 | 开发+生产 |

---

## 4. Phoenix (Arize)：LLM 可观测性与评估一体化

### 4.1 Phoenix 的独特定位

Phoenix 不是传统评估工具——它先是**可观测性平台**（OpenTelemetry 原生），然后才是评估：

```
传统评估:  写测试用例 → 跑评估 → 看分数 → 结束
Phoenix:  自动追踪 → 实时监控 → 发现问题 → 针对性评估 → 持续优化
```

### 4.2 快速集成

```python
# pip install arize-phoenix

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. 启动 Phoenix 服务
session = px.launch_app()

# 2. 自动插桩 LangChain/LlamaIndex
LangChainInstrumentor().instrument()

# 3. 正常运行你的Agent
agent.run("查询设备A的告警")

# 4. Phoenix自动记录所有trace → 在Dashboard中查看
# http://localhost:6006

# 5. 从生产数据中提取测试用例
from phoenix.experiments import run_experiment

run_experiment(
    dataset=session.get_dataset("production-traces-last-7-days"),
    evaluators=[
        HallucinationEvaluator(),
        RelevanceEvaluator(),
        ToxicityEvaluator(),
    ]
)
```

### 4.3 Phoenix 核心能力

| 能力 | 说明 |
|------|------|
| **LLM Trace 可视化** | 完整的Agent执行链路追踪，每步耗时/Token/工具调用 |
| **Embedding 漂移检测** | 检测检索向量空间是否偏离训练分布 |
| **自动评估集生成** | 从生产trace中自动提取异常case组成评估集 |
| **A/B 实验对比** | 对比不同 Prompt/模型版本的评估结果 |
| **异常检测** | 自动发现退化、幻觉率飙升、延迟异常 |

---

## 5. LangKit：文本质量与安全评估

### 5.1 核心功能

LangKit 专注文本质量的统计评估（非 LLM-as-Judge）：

```python
from langkit import (
    textstat,          # 可读性指标
    regexes,           # 正则模式匹配
    sentiment,         # 情感分析
    toxicity,          # 毒性检测
    themes,            # 主题分类
    input_output,      # 输入输出对比
)

# 评估文本质量
result = textstat.flesch_reading_ease("你的Agent生成的文本内容")
# → 返回可读性分数、年级水平、句子复杂度

# 检测敏感内容
result = toxicity.toxicity_metric("文本内容")
# → 返回毒性分数、威胁等级、侮辱等级

# 检测幻觉（简单方法）
result = input_output.refusal_similarity(input_text, output_text)
# → 检测Agent是否在"拒绝回答"而非"产生幻觉"
```

### 5.2 组合使用

```python
class ProductionQualityGuard:
    """生产环境质量守护"""
    
    def __init__(self):
        self.checks = [
            ("Hallucination", lambda x: hallucination_score(x) < 0.3),
            ("Toxicity",      lambda x: toxicity_score(x) < 0.1),
            ("Readability",   lambda x: readability_score(x) > 60),
            ("Refusal",       lambda x: not is_refusal(x)),
        ]
    
    def validate(self, output: str) -> dict:
        results = {}
        for name, check in self.checks:
            results[name] = check(output)
        
        if not all(results.values()):
            failed = [k for k, v in results.items() if not v]
            return {"pass": False, "failed_checks": failed}
        
        return {"pass": True}
```

---

## 6. 工具对比与选型指南

### 6.1 全面对比

| 维度 | RAGAS | DeepEval | TruLens | Phoenix | LangKit |
|------|-------|----------|---------|---------|---------|
| RAG评估 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★☆ | ★☆☆☆☆ |
| 通用LLM评估 | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| 安全/幻觉 | ★★☆☆☆ | ★★★★★ | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |
| 可观测性 | ★☆☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★☆☆☆☆ |
| CI/CD集成 | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| 可视化 | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★★ | ★☆☆☆☆ |
| 学习曲线 | 低 | 中 | 中 | 高 | 低 |

### 6.2 场景化推荐

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| RAG应用开发阶段 | RAGAS + DeepEval | RAGAS做核心指标，DeepEval补充安全评估 |
| RAG应用生产运行 | TruLens | 实时追踪 + 自动评估 |
| Agent系统全面评估 | DeepEval + Phoenix | DeepEval做批量评估，Phoenix做可观测性 |
| 安全合规审查 | DeepEval + LangKit | 毒性/偏见/幻觉多维度检测 |
| CI/CD流水线 | DeepEval | 原生pytest集成 |
| 需要统一平台 | Phoenix | 可观测性+评估一体化 |

---

## 7. 生产级评估架构

### 7.1 评估流水线设计

```
┌─────────────────────────────────────────────────────────────┐
│                    生产级评估架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  开发阶段                  CI/CD阶段            生产阶段       │
│  ┌──────────┐       ┌──────────────┐    ┌──────────────┐    │
│  │ RAGAS    │       │ DeepEval     │    │ Phoenix      │    │
│  │ (RAG指标)│  ───→ │ (全面回归)    │───→│ (实时监控)    │    │
│  │          │       │              │    │              │    │
│  │ TruLens  │       │ - RAG指标    │    │ - Trace追踪  │    │
│  │ (在线调试)│       │ - 幻觉检测    │    │ - 漂移检测   │    │
│  └──────────┘       │ - 安全指标    │    │ - 自动评估   │    │
│                     │ - 回归基线    │    │ - 告警通知   │    │
│                     └──────┬───────┘    └──────┬───────┘    │
│                            │                   │            │
│                     ┌──────▼───────┐    ┌──────▼───────┐    │
│                     │ 评估报告      │    │ 自动回滚      │    │
│                     │ 通过/失败     │    │ (退化>阈值)   │    │
│                     └──────────────┘    └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 评估集管理

```python
class EvaluationDatasetManager:
    """评估数据集管理"""
    
    def __init__(self):
        self.datasets = {
            "core_regression": 200,      # 核心功能回归
            "edge_cases": 50,            # 边界场景
            "safety_tests": 100,         # 安全测试
            "production_samples": 500,   # 生产采样
        }
    
    def update_from_production(self, phoenix_client):
        """从生产环境自动补充评估集"""
        # 1. 从 Phoenix 获取最近 7 天的异常 case
        anomalies = phoenix_client.get_anomalies(days=7)
        
        # 2. 人工审核后加入评估集
        # 3. 去重（避免重复case稀释评估质量）
        # 4. 平衡各类场景的占比
        
    def health_check(self) -> dict:
        """评估集健康检查"""
        return {
            "total_cases": sum(self.datasets.values()),
            "last_updated": self.last_update_time,
            "category_balance": self.check_balance(),
            "stale_cases": self.find_stale(days=30),  # 30天未使用的case
        }
```

---

*最后更新：2026-05-29*

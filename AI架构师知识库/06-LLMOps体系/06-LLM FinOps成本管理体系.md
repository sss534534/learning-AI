# LLM FinOps 成本管理体系

> FinOps = Finance + DevOps。LLM 场景下，成本可能占运营费用的 60-80%。
> 没有体系化的成本管理，AI 应用规模越大越亏钱。

## 元数据

- **难度**: ⭐⭐
- **前置知识**: [LLM 基础](/01-LLM基础理论/01-Transformer架构详解.md)、[成本管理基础](/08-架构模式/02-AI经济学与架构决策.md)
- **关联文件**: [成本优化与 A-B 测试](/06-LLMOps体系/03-成本优化与A-B测试.md)、[LLM 语义缓存架构](/05-AI基础设施/03-LLM语义缓存架构.md)、[模型服务与推理优化](/05-AI基础设施/02-模型服务与推理优化.md)、[AI 经济学与架构决策](/08-架构模式/02-AI经济学与架构决策.md)、[LLM SRE 与生产运维](/06-LLMOps体系/05-LLM SRE与生产运维.md)、[可观测性与治理](/06-LLMOps体系/02-可观测性与治理.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [LLM FinOps 框架](#1-llm-finops-框架)
2. [TCO 全成本模型](#2-tco-全成本模型)
3. [成本归因与追踪](#3-成本归因与追踪)
4. [成本预测与预算管理](#4-成本预测与预算管理)
5. [ROI 决策框架](#5-roi-决策框架)
6. [多云成本优化](#6-多云成本优化)
7. [持续成本优化闭环](#7-持续成本优化闭环)
8. [深度分析：成本权衡与架构决策](#8-深度分析成本权衡与架构决策)
9. [2026 前沿：AI Credits 与 Agent 计量经济](#9-2026-前沿ai-credits-与-agent-计量经济)
10. [Checklist：成本管理成熟度](#10-checklist成本管理成熟度)
11. [延伸阅读](#11-延伸阅读)

---

## 1. LLM FinOps 框架

### 1.1 三层 FinOps 循环

```
┌──────────────────────────────────────────────────────────┐
│                  LLM FinOps 循环                          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐      │
│  │            Inform (可见性)                        │      │
│  │  · 成本归因到团队/功能/用户                        │      │
│  │  · 实时成本 Dashboard                            │      │
│  │  · Token消耗趋势分析                              │      │
│  └──────────────────────┬──────────────────────────┘      │
│                         │                                 │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │            Optimize (优化)                        │      │
│  │  · 模型级联降本                                  │      │
│  │  · 语义缓存利用                                  │      │
│  │  · Prompt 精简                                   │      │
│  │  · 推理优化（量化/批处理）                        │      │
│  └──────────────────────┬──────────────────────────┘      │
│                         │                                 │
│                         ▼                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │            Operate (运营)                         │      │
│  │  · 预算设定与审批                                │      │
│  │  · 成本异常告警                                  │      │
│  │  · 成本预测与规划                                │      │
│  │  · ROI 回顾与决策                                │      │
│  └─────────────────────────────────────────────────┘      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 1.2 成本构成

```
LLM 总运营成本 = API调用成本 + 自建推理成本 + 运维人力 + 基础设施

API调用:
  - 输入Token费用 ($/1M tokens)
  - 输出Token费用 ($/1M tokens)
  - 微调训练费用
  
自建推理:
  - GPU租赁/折旧 (按小时)
  - GPU空闲成本 (利用率 < 50% = 浪费)
  - 电力 + 冷却
  - 网络带宽
  
运维:
  - SRE人力
  - 监控工具许可
  - 安全审计
```

---

## 2. TCO 全成本模型

### 2.1 API vs 自建模型 TCO 对比

```python
class LLMTCOModel:
    """LLM 总拥有成本模型"""
    
    def __init__(self):
        self.api_pricing = {
            "gpt-4o":       {"input": 2.50,  "output": 10.00},  # $/1M tokens
            "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
            "claude-3-opus":{"input": 15.00, "output": 75.00},
            "qwen-max":     {"input": 0.55,  "output": 2.20},
        }
    
    def calculate_api_cost(self, model: str, daily_requests: int, 
                           avg_input_tokens: int, avg_output_tokens: int) -> dict:
        """计算API月度成本"""
        daily_input_tokens = daily_requests * avg_input_tokens
        daily_output_tokens = daily_requests * avg_output_tokens
        
        pricing = self.api_pricing[model]
        daily_cost = (
            daily_input_tokens / 1_000_000 * pricing["input"] +
            daily_output_tokens / 1_000_000 * pricing["output"]
        )
        
        return {
            "daily_cost": round(daily_cost, 2),
            "monthly_cost": round(daily_cost * 30, 2),
            "annual_cost": round(daily_cost * 365, 2),
            "cost_per_request": round(daily_cost / daily_requests, 4),
        }
    
    def calculate_self_hosted_cost(self, gpu_type: str, num_gpus: int,
                                    utilization: float) -> dict:
        """计算自建模型月度成本"""
        gpu_pricing = {
            "A100-80G": {"hourly_cloud": 3.50, "monthly_purchase_amortized": 1200},
            "H100":     {"hourly_cloud": 5.50, "monthly_purchase_amortized": 2500},
        }
        
        pricing = gpu_pricing[gpu_type]
        
        # 云租赁
        hourly_cost = pricing["hourly_cloud"] * num_gpus
        monthly_cloud = hourly_cost * 24 * 30 * utilization
        
        # 自建摊销（假设3年折旧）
        monthly_owned = pricing["monthly_purchase_amortized"] * num_gpus
        
        # 电力 + 冷却 (约GPU成本的15%)
        power_cooling = monthly_owned * 0.15
        
        return {
            "cloud_monthly": round(monthly_cloud, 2),
            "owned_monthly": round(monthly_owned, 2),
            "power_cooling": round(power_cooling, 2),
            "total_cloud": round(monthly_cloud + power_cooling, 2),
            "total_owned": round(monthly_owned + power_cooling, 2),
        }
```

### 2.2 TCO 决策矩阵

| 场景 | 日请求量 | QPS | 推荐方案 | 月成本估算 |
|------|---------|-----|---------|-----------|
| 小规模 | < 10K | < 1 | API直接调用 | $50-200 |
| 中规模 | 10K-500K | 1-10 | API + 缓存 | $500-3000 |
| 大规模 | 500K-5M | 10-100 | 自建推理 | $3000-15000 |
| 超大规模 | > 5M | > 100 | 自建集群 | $15000+ |

**临界点估算**：当 API 月费 > 2× 自建月费时，考虑转向自建。

---

## 3. 成本归因与追踪

### 3.1 成本标签体系

```python
class CostAttribution:
    """成本归因系统"""
    
    def __init__(self):
        self.dimensions = {
            "team":        ["platform", "business", "devops", "security"],
            "feature":     ["chat", "search", "recommendation", "agent"],
            "model":       ["gpt-4o", "qwen-30b", "claude-3"],
            "user_tier":   ["free", "pro", "enterprise"],
            "request_type":["simple_query", "rag", "agent_task", "code_gen"],
        }
    
    def attribute_cost(self, request: dict, response: dict) -> dict:
        """为每次 LLM 调用打标签"""
        cost = self.calculate_cost(request, response)
        
        return {
            "cost": cost,
            "input_tokens": request["usage"]["input_tokens"],
            "output_tokens": response["usage"]["output_tokens"],
            "model": request["model"],
            "team": request["metadata"]["team"],
            "feature": request["metadata"]["feature"],
            "user_tier": request["metadata"]["user_tier"],
            "request_type": self.classify_request(request),
            "timestamp": time.time(),
            "cache_hit": request.get("cache_hit", False),
        }
    
    def get_cost_breakdown(self, start_time, end_time, group_by: str) -> pd.DataFrame:
        """成本分组统计"""
        records = self.query_cost_records(start_time, end_time)
        df = pd.DataFrame(records)
        
        return df.groupby(group_by).agg({
            "cost": "sum",
            "input_tokens": "sum",
            "output_tokens": "sum",
            "request_count": "count",
            "cache_hit": "mean",  # 缓存命中率
        }).sort_values("cost", ascending=False)
```

### 3.2 成本 Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM 成本 Dashboard                         │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ 本月总成本 │ 日均成本   │ Token消耗  │ 缓存节省   │ 预算使用率       │
│ $4,532   │ $151     │ 89M/月   │ $1,280   │ 68% [████████░░] │
│ +12% MoM │ -3% WoW  │ +8% MoM  │ 22% 节省  │                 │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                                                              │
│ 按团队                  按模型                   按功能       │
│ Platform:   $2,100 ██  GPT-4o:  $1,800 ██  Chat: $1,500 ██ │
│ Business:   $1,400 ██  Qwen:    $1,400 ██  Agent:$1,200 ██ │
│ DevOps:       $632 █   Claude:  $  832 █   RAG:   $800 █   │
│ Security:     $400 ░   Mini:    $  500 ░   Other:  $ 32 ░  │
│                                                              │
│ 成本趋势 (7天)              请求类型成本分布                    │
│ ┌─────────────────────┐    ┌─────────────────────────────┐   │
│ │ ▁▂▃▂▃▄▃             │    │ Simple:  15% ($680)  ███    │   │
│ │ M T W T F S S       │    │ RAG:     35% ($1,586)█████   │   │
│ └─────────────────────┘    │ Agent:   45% ($2,039)███████  │   │
│                            │ Code:     5% ($227)  █        │   │
│                            └─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 成本预测与预算管理

### 4.1 成本预测模型

```python
class CostPredictor:
    """LLM 成本预测器"""
    
    def predict_monthly_cost(self, 
                             current_month_cost: float,
                             user_growth_rate: float,
                             avg_tokens_per_request_growth: float,
                             model_mix_shift: float = 0.0) -> dict:
        """
        预测未来6个月成本
        
        user_growth_rate: 用户增长率 (如 0.15 = 15% MoM)
        avg_tokens_per_request_growth: 每次请求Token数增长
        model_mix_shift: 模型组合变化影响 (正值=更多用贵模型)
        """
        predictions = []
        current = current_month_cost
        
        for month in range(1, 7):
            # 复合增长
            growth = (1 + user_growth_rate) * (1 + avg_tokens_per_request_growth) * (1 + model_mix_shift)
            predicted = current * (growth ** month)
            
            predictions.append({
                "month": month,
                "predicted_cost": round(predicted, 2),
                "growth_factor": round(growth, 3),
                "confidence_interval": (
                    round(predicted * 0.8, 2),  # 下限
                    round(predicted * 1.3, 2),  # 上限
                )
            })
        
        return predictions
    
    def get_budget_alert(self, current_cost: float, monthly_budget: float, 
                          days_elapsed: int) -> dict:
        """预算预警"""
        daily_budget = monthly_budget / 30
        expected_cost = daily_budget * days_elapsed
        burn_rate = current_cost / expected_cost if expected_cost > 0 else 0
        
        if burn_rate > 1.2:
            return {
                "level": "CRITICAL",
                "message": f"成本超出预期 20%+，当前 burn rate: {burn_rate:.2f}x",
                "projected_overspend": round(current_cost * 30 / days_elapsed - monthly_budget, 2)
            }
        elif burn_rate > 1.0:
            return {
                "level": "WARNING",
                "message": f"成本略超预期，burn rate: {burn_rate:.2f}x"
            }
        return {"level": "OK"}
```

### 4.2 预算管理流程

```
月初: 设定团队/项目预算
每周: 自动发送预算使用率报告
超80%: 自动预警
超100%: 冻结非关键请求 + 审批额外预算
月末: 成本复盘 + 更新下月预算
```

---

## 5. ROI 决策框架

### 5.1 模型升级 ROI 计算

```
问题: 是否从 GPT-4o-mini ($0.15/$0.60) 升级到 GPT-4o ($2.50/$10.00)？

ROI = (质量提升价值 - 成本增量) / 成本增量

质量提升价值需量化:
- 对客服场景: 首次解决率每提升 1% = 节省人工成本 $X/月
- 对代码生成: 采纳率每提升 1% = 节省开发时间 Y 小时/月
- 对搜索: 准确率每提升 1% = 用户留存提升 Z%

决策表格:
┌──────────────┬──────────┬──────────┬──────────┐
│              │ 成本增量  │ 质量提升   │ ROI      │
├──────────────┼──────────┼──────────┼──────────┤
│ 客服场景      │ +$800/月 │ 首次解决率 │ +120%    │
│              │          │ +15%      │ 值得升级  │
├──────────────┼──────────┼──────────┼──────────┤
│ 简单翻译      │ +$800/月 │ 质量提升   │ -60%     │
│              │          │ +2%       │ 不值得    │
└──────────────┴──────────┴──────────┴──────────┘
```

### 5.2 ROI 决策矩阵

```python
def model_upgrade_roi(current_model, candidate_model, 
                       monthly_requests, quality_impact_estimate):
    """
    quality_impact_estimate: {
        "accuracy_gain": 0.05,       # 准确率提升 5%
        "revenue_per_accuracy": 100, # 每1%准确率提升 = $100/月收入
        "cost_saving_per_accuracy": 50, # 每1%准确率提升 = $50/月节省
    }
    """
    current_cost = estimate_monthly_cost(current_model, monthly_requests)
    candidate_cost = estimate_monthly_cost(candidate_model, monthly_requests)
    cost_delta = candidate_cost - current_cost
    
    impact = quality_impact_estimate
    total_benefit = (
        impact["accuracy_gain"] * 100 * impact["revenue_per_accuracy"] +
        impact["accuracy_gain"] * 100 * impact["cost_saving_per_accuracy"]
    )
    
    roi = (total_benefit - cost_delta) / cost_delta if cost_delta > 0 else float('inf')
    
    return {
        "current_monthly_cost": current_cost,
        "candidate_monthly_cost": candidate_cost,
        "cost_delta": cost_delta,
        "estimated_benefit": total_benefit,
        "roi": roi,
        "recommendation": "UPGRADE" if roi > 0.5 else "KEEP_CURRENT"
    }
```

---

## 6. 多云成本优化

### 6.1 成本套利策略

```python
class MultiCloudCostOptimizer:
    """多云成本优化器"""
    
    def __init__(self):
        self.providers = {
            "openai":       {"gpt-4o": 0.00250, "gpt-4o-mini": 0.00015},
            "azure":        {"gpt-4o": 0.00250, "gpt-4o-mini": 0.00015},  # 预留容量折扣
            "deepseek":     {"deepseek-v2": 0.00014},  # 极低成本
            "qwen":         {"qwen-max": 0.00055, "qwen-turbo": 0.00008},
            "self_hosted":  {"qwen-30b": self.calc_self_hosted_unit_cost()},
        }
    
    def optimal_route(self, request: dict) -> str:
        """为每个请求选择最优成本的供应商"""
        request_type = self.classify(request)
        
        # 简单任务 → 最便宜的模型
        if request_type == "simple_query":
            return "qwen-turbo"  # $0.00008/1K tokens
        
        # 中等任务 → 性价比最高的模型
        if request_type == "rag":
            return "deepseek-v2"  # $0.00014/1K tokens
        
        # 复杂任务 → 质量优先但选便宜的供应商
        if request_type == "agent_task":
            return "azure-gpt-4o"  # 预留容量折扣
        
        # 关键任务 → 质量最优
        if request_type == "critical":
            return "openai-gpt-4o"
    
    def calc_self_hosted_unit_cost(self) -> float:
        """计算自建模型的单位Token成本"""
        # GPU: A100-80G, $3.50/h
        # 吞吐: 2000 tokens/s
        # 利用率: 70%
        
        hourly_tokens = 2000 * 3600 * 0.70  # 5,040,000 tokens/h
        hourly_cost = 3.50
        
        cost_per_1k = (hourly_cost / hourly_tokens) * 1000
        return cost_per_1k  # ≈ $0.00069/1K tokens
```

### 6.2 多云配置

```yaml
routing_strategy:
  default: "cost_optimized"  # cost_optimized | quality_first | hybrid
  
  models:
    - name: "gpt-4o"
      provider: "openai"
      priority: 1
      max_cost_per_month: 2000
      
    - name: "gpt-4o"
      provider: "azure"
      priority: 2
      reservation_discount: 0.15  # 15% 预留折扣
      
    - name: "qwen-max"
      provider: "alibaba"
      priority: 3
      
  fallback_chain:
    - condition: "cost_budget_exhausted"
      action: "switch_to_cheaper_model"
    - condition: "provider_outage"
      action: "switch_to_next_provider"
```

---

## 7. 持续成本优化闭环

### 7.1 优化策略优先级

| 优先级 | 策略 | 预期节省 | 风险 |
|--------|------|---------|------|
| P0 | 语义缓存 | 30-50% | 低 |
| P0 | Prompt 精简 | 20-30% | 低 |
| P1 | 模型级联 (小模型处理简单任务) | 40-60% | 中 |
| P1 | 输出 Token 限制 | 15-25% | 低 |
| P2 | 批处理推理 | 20-30% | 中 |
| P2 | 模型量化 (INT8/INT4) | 30-50% | 中 |
| P3 | 多供应商成本套利 | 10-20% | 低 |

### 7.2 月/季/年成本优化循环

```
月度:
  - 成本回顾 Dashboard
  - 识别 Top 5 成本来源
  - 调整缓存策略
  
季度:
  - TCO 模型重新校准
  - 评估模型升级/降级 ROI
  - 供应商合同重新谈判
  
年度:
  - 年度成本预测
  - 自建 vs API 重新评估
  - 预算规划
```

---

## 8. 深度分析：成本权衡与架构决策

### 8.1 API 调用 vs 自建推理 vs 缓存的成本权衡

LLM 场景下的成本优化并非简单的"选最便宜的方案"，而是需要在延迟、质量、可靠性之间做多维度权衡。

| 维度 | API 调用 | 自建推理 | 语义缓存 |
|------|---------|---------|---------|
| 边际成本 | 高（按 token 计费） | 低（固定成本摊销） | 极低（仅存储） |
| 初始投入 | 零 | 高（GPU + 运维） | 低 |
| 延迟 | 中（网络开销） | 低（本地部署） | 极低（直接命中） |
| 质量天花板 | 高（商业模型） | 中（开源模型） | 取决于缓存内容 |
| 弹性 | 好 | 差 | 中 |

**核心结论**：
- 日请求量 < 100K：优先 API + 缓存组合
- 日请求量 100K-1M：语义缓存是关键杠杆，命中率决定 ROI
- 日请求量 > 1M：自建推理的经济性开始显现，但需考虑 GPU 利用率

### 8.2 AI Credits 经济模型与架构影响

2026 年 GitHub Copilot 转向 Credits 计费标志着 Agentic 服务进入计量经济时代。这对架构设计有深远影响：

**Credits 模型的定价逻辑**：
```
Task Cost = Σ(Model_Credits_per_Token × Tokens_Consumed) + Base_Fee
```
其中 Model_Credits_per_Token 由模型复杂度和供需关系决定，GPT-4o 消耗的 credits 是 GPT-4o-mini 的 10-20 倍。

**架构决策影响**：
1. **模型级联成为必需**：简单任务必须走低成本模型，否则 credits 消耗不可持续
2. **缓存策略更激进**：context caching 可以大幅减少重复 token 消耗
3. **预算控制前移**：从月末对账变为请求级的 budget 检查
4. **混合计量**：固定月费 + 按量 credits，需要精细的用量预估

### 8.3 主流 LLM 供应商 TCO 对比

| 供应商 | 代表模型 | Input $/1M | Output $/1M | 自建等价成本 | 适用场景 |
|--------|---------|-----------|------------|------------|---------|
| OpenAI | GPT-4o | $2.50 | $10.00 | $1.20-1.80 | 通用/复杂推理 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | $0.08-0.12 | 简单任务 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | $1.50-2.50 | 长上下文/代码 |
| DeepSeek | DeepSeek-V2 | $0.14 | $0.28 | $0.04-0.08 | 高性价比 |
| Alibaba | Qwen-Max | $0.55 | $2.20 | $0.30-0.50 | 中文场景 |
| Meta (自建) | Llama 3 70B | — | — | $0.08-0.15 | 高吞吐场景 |

> 自建等价成本 = GPU 折旧摊销 + 电力 + 运维，假设利用率 > 60%。

### 8.4 关键争论与前沿思考

| 争论点 | 正方 | 反方 |
|--------|------|------|
| 预留容量 vs 按需 | 预留可节省 15-30% | 容量规划不准会导致浪费 |
| 多模型级联 vs 单模型 | 级联可降本 40-60% | 系统复杂度增加，延迟抖动 |
| 语义缓存 vs 无缓存 | 命中率高时节省 30-50% | 冷启动阶段 TCO 反而更高 |
| 推理优化投入 ROI | 量化/剪枝可降本 2-3x | 优化投入本身需要工程成本 |
| 多云套利 vs 单一供应商 | 套利可降本 10-20% | 多供应商管理复杂度高 |

**共识**：2026 年的趋势是"精细化成本运营"——没有银弹，必须组合多种策略。

---

## 9. 2026 前沿：AI Credits 与 Agent 计量经济

### 9.1 GitHub AI Credits（2026-06-01）

2026 年 6 月 1 日，GitHub Copilot 所有计划转为使用量计费（Usage-Based Billing），这是 Agentic Coding 成本模型的分水岭事件。

**定价模型：**

| 项目 | 内容 |
|------|------|
| 计价单位 | AI Credits（1 credit = $0.01） |
| 计量范围 | input token + output token + cached token |
| 模型差异 | 不同模型每 token 消耗不同 credits |
| 代码补全 | 无限量（不计入 credits） |

**套餐对比：**

| 套餐 | 月费 | 每月 Credits | 适用 |
|------|------|-------------|------|
| Pro | $10 | 1,500 | 个人开发者 |
| Pro+ | $30 | 7,000 | 重度用户 |
| Max | 自定义 | 20,000 | 企业团队 |

**行业信号：** Agentic Coding 是计量资源，按计算定价而非固定席位。这意味着大规模使用 Agent 编写代码时，成本将线性增长。

### 9.2 Agent 成本归因框架

```
┌────────────────────────────────────────────────────────┐
│                 Agent 成本归因模型                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  层级 1：模型成本（直接 token 消耗）                     │
│  ├── Input tokens（prompt + context）                   │
│  ├── Output tokens（生成内容）                          │
│  ├── Thinking tokens（推理模型）                        │
│  └── Cache tokens（上下文缓存）                         │
│                                                        │
│  层级 2：Agent 基础设施成本                             │
│  ├── Agent 运行时（框架 + 编排）                        │
│  ├── 工具执行（API 调用费用）                           │
│  ├── 记忆存储（向量数据库）                             │
│  └── 可观测性（tracing + 监控）                         │
│                                                        │
│  层级 3：治理成本                                       │
│  ├── 安全审查 + 合规审计                                │
│  ├── Agent 身份管理                                    │
│  └── 成本治理工具                                      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 9.3 每任务成本核算

```python
def estimate_task_cost(task: dict, model_pricing: dict) -> dict:
    """估算单个 Agent 任务的总成本"""

    # 层级 1：模型成本
    model_cost = (
        task["input_tokens"] * model_pricing["input_per_1k"] / 1000 +
        task["output_tokens"] * model_pricing["output_per_1k"] / 1000 +
        task["think_tokens"] * model_pricing["think_per_1k"] / 1000
    )

    # 层级 2：基础设施
    infra_cost = (
        task["tool_calls"] * 0.001 +          # 工具调用
        task["vector_searches"] * 0.0005 +     # 向量搜索
        task["agent_steps"] * 0.0002           # 编排开销
    )

    # 层级 3：治理（固定比例）
    governance_cost = (model_cost + infra_cost) * 0.15

    return {
        "total": model_cost + infra_cost + governance_cost,
        "breakdown": {
            "model": model_cost,
            "infra": infra_cost,
            "governance": governance_cost
        }
    }
```

### 9.4 成本优化策略（2026 更新）

| 策略 | 节省潜力 | 说明 |
|------|----------|------|
| 模型级联 | 40-60% | 简单任务用小模型，困难任务升级 |
| 语义缓存 | 30-50% | 缓存相似请求的响应 |
| Prompt 精简 | 20-30% | 减少输入 token |
| 推理预算控制 | 15-40% | 限制 thinking tokens 上限 |
| 多 Agent 复用 | 25-50% | 共享中间结果，避免重复计算 |
| 本地小模型 | 60-80% | 隐私敏感或高频率任务使用本地模型 |

---

## 10. Checklist：成本管理成熟度

### 10.1 成本归因（Cost Attribution）

- [ ] 所有 LLM 调用是否带有团队/功能/用户层级的标签？
- [ ] 是否能按模型、功能、团队维度生成成本报表？
- [ ] 是否区分了缓存命中与未命中的成本？
- [ ] 是否对 Agent 的多步调用做了任务级归因？
- [ ] 是否有成本基准线（baseline）用于对比？

### 10.2 预算管理（Budget Management）

- [ ] 是否为每个团队/项目设定了月度预算？
- [ ] 是否有自动化的预算预警机制（80% / 100%）？
- [ ] 是否支持按日的 burn rate 追踪？
- [ ] 预算超额时是否有自动化降级或阻断策略？
- [ ] 是否定期（月度）进行成本复盘？

### 10.3 优化策略（Optimization Strategies）

- [ ] 是否实施了语义缓存？缓存命中率是否持续监控？
- [ ] 是否实施模型级联路由？简单请求是否走低成本模型？
- [ ] 是否对 Prompt 长度做了优化（精简 context）？
- [ ] 是否限制了输出 Token 上限？
- [ ] 是否评估了自建推理 vs API 的 TCO 转折点？
- [ ] 是否实施了推理预算控制（thinking tokens 上限）？

### 10.4 监控告警（Monitoring & Alerting）

- [ ] 是否有实时成本 Dashboard？
- [ ] 是否设置了日/周/月成本异常告警？
- [ ] 是否能按用户/功能/模型维度钻取成本异常？
- [ ] 是否有成本预测模型（预测未来 30 天支出）？
- [ ] 成本告警是否关联到 On-Call 流程？
- [ ] 是否定期进行 TCO 模型校准？

---

## 11. 延伸阅读

### 关联文件

- [成本优化与 A-B 测试](../06-LLMOps体系/03-成本优化与A-B测试.md)
- [LLM 语义缓存架构](../05-AI基础设施/03-LLM语义缓存架构.md)
- [模型服务与推理优化](../05-AI基础设施/02-模型服务与推理优化.md)
- [AI 经济学与架构决策](../08-架构模式/02-AI经济学与架构决策.md)
- [LLM SRE 与生产运维](../06-LLMOps体系/05-LLM SRE与生产运维.md)
- [可观测性与治理](../06-LLMOps体系/02-可观测性与治理.md)
- [推理优化技术](../02-模型工程化/02-推理优化技术.md)

### 推荐资源

1. **Cloud FinOps** — O'Reilly, 2023. 云成本管理圣经，FinOps 基金会官方指南。
2. **LLM Inference Performance Engineering** — Databricks, 2024. 推理性能与成本的系统性分析。
3. **The Economics of Large Language Models** — OpenAI Research, 2023. 大语言模型经济学的开创性讨论。
4. **AI Cost Optimization at Scale** — Meta Engineering Blog, 2024. Meta 在 LLM 成本优化上的工程实践。
5. **TCO Analysis for LLM Deployment Patterns** — Microsoft Azure Architecture Center, 2024.

---

*最后更新：2026-06-12*

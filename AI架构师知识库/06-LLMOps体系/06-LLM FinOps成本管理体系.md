# LLM FinOps 成本管理体系

> FinOps = Finance + DevOps。LLM 场景下，成本可能占运营费用的 60-80%。
> 没有体系化的成本管理，AI 应用规模越大越亏钱。

## 目录

1. [LLM FinOps 框架](#1-llm-finops-框架)
2. [TCO 全成本模型](#2-tco-全成本模型)
3. [成本归因与追踪](#3-成本归因与追踪)
4. [成本预测与预算管理](#4-成本预测与预算管理)
5. [ROI 决策框架](#5-roi-决策框架)
6. [多云成本优化](#6-多云成本优化)
7. [持续成本优化闭环](#7-持续成本优化闭环)

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

*最后更新：2026-05-29*

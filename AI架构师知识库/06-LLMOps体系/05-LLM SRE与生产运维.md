# LLM SRE 与生产运维

> 传统 SRE 管 CPU/内存/磁盘，LLM SRE 还要管 Token/幻觉/概率性故障。
> 这是目前 AI 领域最稀缺的系统性知识之一。

## 目录

1. [LLM SRE 核心概念](#1-llm-sre-核心概念)
2. [SLO/SLI 设计](#2-slosli-设计)
3. [GPU 容量规划](#3-gpu-容量规划)
4. [灰度发布策略](#4-灰度发布策略)
5. [灾难恢复与灾备](#5-灾难恢复与灾备)
6. [On-Call 手册](#6-on-call-手册)
7. [生产检查清单](#7-生产检查清单)

---

## 1. LLM SRE 核心概念

### 1.1 与传统 SRE 的关键差异

| 维度 | 传统 SRE | LLM SRE |
|------|---------|---------|
| 主要资源 | CPU/RAM/Disk | GPU/VRAM/Token配额 |
| 故障模式 | 确定性故障 | **概率性故障**（幻觉、随机输出） |
| 性能指标 | 延迟/吞吐/QPS | + TTFT/TPOT/Token/s |
| 质量指标 | 错误率 | + 幻觉率/相关性/安全性 |
| 容量单位 | 请求数/秒 | Token数/秒 + 并发GPU数 |
| 供应商依赖 | 低 | **高**（API供应商宕机影响） |

### 1.2 LLM SRE 责任矩阵

```
┌────────────────────────────────────────────────────────────────┐
│                       LLM SRE 责任矩阵                          │
├────────────┬──────────────┬──────────────┬────────────────────┤
│  可靠性     │  性能        │  成本        │  质量              │
├────────────┼──────────────┼──────────────┼────────────────────┤
│ · 可用性    │ · TTFT < 2s  │ · Token预算   │ · 幻觉率 < 2%      │
│ · 故障转移  │ · 吞吐达标    │ · 成本归因    │ · 安全合规         │
│ · 灾备恢复  │ · 并发容量    │ · 预算预警    │ · 评估回归         │
│ · 数据一致  │ · GPU利用率   │ · 资源优化    │ · A/B验证          │
└────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## 2. SLO/SLI 设计

### 2.1 SLI 指标体系

```yaml
sli_definitions:
  # 延迟指标
  - name: "ttft_p99"  # Time To First Token
    description: "首个Token返回时间P99"
    target: "< 2000ms"
    measurement: "prometheus_histogram_quantile(0.99, ttft_seconds)"
    
  - name: "tpot_p99"  # Time Per Output Token
    description: "每个输出Token的平均时间P99"
    target: "< 50ms"
    
  - name: "e2e_latency_p95"  # 端到端延迟
    description: "从请求到完成响应的P95时间"
    target: "< 10s"
  
  # 可用性指标
  - name: "availability"
    description: "服务可用性 (非5xx响应)"
    target: "99.9%"
    measurement: "sum(rate(http_requests_total{status!~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"
  
  # 质量指标
  - name: "hallucination_rate"
    description: "输出幻觉率 (需要评估系统支持)"
    target: "< 2%"
    measurement: "phoenix_hallucination_rate"
    
  - name: "refusal_rate"
    description: "Agent 拒绝执行率"
    target: "< 5%"
    warning: "> 10% → 检查 System Prompt 或工具可用性"
  
  # 容量指标
  - name: "gpu_utilization"
    description: "GPU 利用率"
    target: "60-80%"
    warning: "> 90% → 准备扩容; < 30% → 降配"
    
  - name: "token_burn_rate"
    description: "Token消耗速率"
    target: "低于预算的日均消耗"
    warning: "超过日均 120% → 告警"
```

### 2.2 SLO 与错误预算

```python
class LLMErrorBudget:
    """LLM 错误预算管理"""
    
    def __init__(self):
        self.slos = {
            "availability": {
                "target": 0.999,  # 99.9%
                "window": "30d",
                "max_unavailable_minutes": 43.2,  # 30天 * 0.001 * 24 * 60
            },
            "ttft_p99": {
                "target": 0.99,   # P99 < 2s
                "window": "7d",
            }
        }
    
    def check_budget(self, sli_name: str) -> dict:
        """检查错误预算剩余"""
        slo = self.slos[sli_name]
        current_error_rate = self.get_current_error_rate(sli_name)
        
        if sli_name == "availability":
            budget_burned = self.get_downtime_minutes(window=slo["window"])
            budget_remaining = slo["max_unavailable_minutes"] - budget_burned
            
            if budget_remaining < 0:
                return {
                    "status": "BUDGET_EXHAUSTED",
                    "action": "🚨 冻结所有非紧急发布，全力修复可靠性",
                    "budget_remaining_minutes": budget_remaining
                }
            elif budget_remaining < slo["max_unavailable_minutes"] * 0.2:
                return {
                    "status": "BUDGET_LOW",
                    "action": "⚠️ 仅允许P0修复发布",
                    "budget_remaining_minutes": budget_remaining
                }
        
        return {"status": "OK"}
```

### 2.3 SRE 仪表盘设计

```
┌─────────────────────────────────────────────────────────┐
│                   LLM SRE Dashboard                      │
├───────────────┬──────────────┬──────────────┬───────────┤
│ 服务可用性     │ 延迟SLO       │ 错误预算      │ Token消耗  │
│ 99.97% ▼      │ TTFT P99:    │ 剩余: 38min   │ 今日:      │
│ 过去30天       │ 1.8s [████░░]│ 消耗: 12%     │ 2.3M/3M   │
├───────────────┴──────────────┴──────────────┴───────────┤
│                                                         │
│ 实时流量                    GPU 利用率                    │
│ ┌────────────────┐        ┌────────────────┐            │
│ │ ████░░░░ 45 QPS│        │ GPU0: 72% ████░│            │
│ │ █████░░░ 并发8  │        │ GPU1: 68% ███░░│            │
│ └────────────────┘        └────────────────┘            │
│                                                         │
│ 语义缓存命中率              质量指标                       │
│ ┌────────────────┐        ┌────────────────┐            │
│ │ 42% ████░░░░░░ │        │ 幻觉: 1.2% ✓   │            │
│ │ 节省 $450/天   │        │ 毒性: 0.01% ✓  │            │
│ └────────────────┘        └────────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. GPU 容量规划

### 3.1 容量估算模型

```
核心公式: Required_GPUs = ceil(QPS × avg_tokens_per_request / GPU_throughput)

其中:
  GPU_throughput = batch_size × tokens_per_second_per_batch
  
示例:
  QPS = 50 请求/秒
  avg_tokens_per_request = 500 (输入+输出)
  batch_size = 8
  tokens_per_second_per_batch = 200 (A100 for Qwen-30B)
  
  total_tokens_per_second = 50 × 500 = 25,000
  GPU_throughput = 8 × 200 = 1,600 t/s per GPU
  Required_GPUs = ceil(25,000 / 1,600) = 16 GPUs
```

### 3.2 各模型 GPU 需求参考

| 模型 | FP16 VRAM | INT8 VRAM | 推荐GPU | 每卡并发 |
|------|----------|----------|---------|---------|
| Qwen2.5-7B | 14GB | 8GB | A10 (24GB) | 2-3 instances |
| Qwen2.5-30B | 60GB | 35GB | A100-80G | 2 instances |
| DeepSeek-V2 | 48GB | 24GB | A100-80G | 3-4 instances |
| Llama-3-70B | 140GB | 80GB | 2×A100-80G | 1 instance |

### 3.3 容量规划时间线

```
T-6月: 基于业务增长预测年度Token消耗
T-3月: 确认GPU采购/云资源预留
T-1月: 压力测试 + 验证容量模型
T-1周: 预热新节点 + 灰度接入
T-0:   上线
T+1月: 回顾实际 vs 预测，校准模型
```

---

## 4. 灰度发布策略

### 4.1 LLM 灰度发布的三层模型

```
┌──────────────────────────────────────────────────────────┐
│                     三层灰度发布                           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Layer 1: 流量灰度                                        │
│  1% → 5% → 10% → 25% → 50% → 100%                        │
│  每步观察 30分钟                                          │
│                                                           │
│  Layer 2: 用户灰度                                        │
│  内部用户 → 友好客户 → 普通用户 → 全部用户                   │
│                                                           │
│  Layer 3: 功能灰度                                        │
│  只读查询 → 简单工具 → 复杂工具 → 全部能力                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 4.2 自动金丝雀分析

```python
class CanaryAnalyzer:
    """LLM 金丝雀发布分析器"""
    
    def __init__(self):
        self.metrics = [
            "ttft_p99",           # 延迟
            "error_rate",         # 错误率
            "hallucination_rate", # 幻觉率
            "token_per_request",  # Token消耗
            "user_satisfaction",  # 用户满意度
        ]
        
        self.thresholds = {
            "ttft_p99":            {"max_degradation": 1.2},  # 不超过基线 120%
            "error_rate":          {"max_increase": 0.01},    # 不增加超过 1%
            "hallucination_rate":  {"max_increase": 0.005},   # 不增加超过 0.5%
        }
    
    def analyze(self, canary_metrics: dict, baseline_metrics: dict) -> dict:
        results = {"pass": True, "checks": []}
        
        for metric in self.metrics:
            if metric not in self.thresholds:
                continue
            
            canary_value = canary_metrics.get(metric)
            baseline_value = baseline_metrics.get(metric)
            
            if not canary_value or not baseline_value:
                continue
            
            threshold = self.thresholds[metric]
            ratio = canary_value / baseline_value if baseline_value > 0 else float('inf')
            
            check = {
                "metric": metric,
                "canary": canary_value,
                "baseline": baseline_value,
                "ratio": ratio,
                "pass": True
            }
            
            if "max_degradation" in threshold and ratio > threshold["max_degradation"]:
                check["pass"] = False
                results["pass"] = False
                check["reason"] = f"退化 {ratio:.2f}x > 阈值 {threshold['max_degradation']}x"
            
            results["checks"].append(check)
        
        return results
    
    def should_rollback(self, analysis: dict) -> bool:
        """判断是否应该自动回滚"""
        if not analysis["pass"]:
            # 检查是否有严重退化
            severe = [c for c in analysis["checks"] 
                      if not c["pass"] and c.get("ratio", 0) > 2.0]
            if severe:
                return True  # 自动回滚
        return False
```

### 4.3 灰度发布 SOP

```
1. 评估基线 (0% 流量, 30分钟)
2. 1% 流量金丝雀 (30分钟) → 自动分析
3. 5% 流量 (30分钟) → 自动分析
4. 10% 流量 (30分钟) → 自动分析
5. 25% 流量 (60分钟) → 人工确认
6. 50% 流量 (60分钟) → 自动分析
7. 100% 全量 → 监控 24h

任一阶段分析失败 → 触发回滚条件检查 → 
  · 严重退化 (ratio > 2x) → 自动回滚
  · 中等退化 → 人工介入
  · 轻微退化 → 继续观察 (不阻塞)
```

---

## 5. 灾难恢复与灾备

### 5.1 故障场景与恢复

| 故障场景 | RTO | RPO | 恢复策略 |
|---------|-----|-----|---------|
| 单GPU故障 | < 1min | 0 | 自动切换到备用GPU |
| 单模型服务宕机 | < 2min | 0 | K8s自动重启 + 切到备模型 |
| API供应商宕机 | < 30s | 0 | 自动fallback到其他供应商 |
| 数据中心故障 | < 15min | < 5min | 跨Region多活切换 |
| 全量数据丢失 | < 4h | < 1h | 从备份恢复 |

### 5.2 灾备架构

```
┌──────────────────────┐      ┌──────────────────────┐
│    Region A (主)      │      │    Region B (备)      │
│                       │      │                       │
│  ┌─────────────────┐ │      │  ┌─────────────────┐  │
│  │ 负载均衡         │ │      │  │ 负载均衡 (Standby)│ │
│  └────────┬────────┘ │      │  └────────┬────────┘  │
│           │          │      │           │           │
│  ┌────────▼────────┐ │      │  ┌────────▼────────┐  │
│  │ API Gateway     │─┼──────┼──│ API Gateway     │  │
│  └────────┬────────┘ │      │  └────────┬────────┘  │
│           │          │      │           │           │
│  ┌────────▼────────┐ │      │  ┌────────▼────────┐  │
│  │ GPU 集群 8×A100 │ │      │  │ GPU 集群 2×A100 │  │
│  └─────────────────┘ │      │  └─────────────────┘  │
│                       │      │                       │
│  ┌─────────────────┐ │      │  ┌─────────────────┐  │
│  │ MySQL + Redis   │─┼──────┼──│ MySQL + Redis   │  │
│  │ (主)            │ │ 复制  │  │ (从)            │  │
│  └─────────────────┘ │      │  └─────────────────┘  │
│                       │      │                       │
└──────────────────────┘      └──────────────────────┘
```

---

## 6. On-Call 手册

### 6.1 告警分级

```yaml
alerts:
  P0_Critical:  # 🚨 5分钟内响应
    - name: "服务完全不可用"
      condition: "availability < 99% for 2min"
    - name: "幻觉率飙升"
      condition: "hallucination_rate > 10% for 5min"
    - name: "安全事件"
      condition: "jailbreak_attempt_block_rate < 90%"
  
  P1_High:  # ⚠️ 15分钟内响应
    - name: "TTFT 严重退化"
      condition: "ttft_p99 > 10s for 10min"
    - name: "GPU 全部 100%"
      condition: "gpu_utilization > 95% on all GPUs for 5min"
    - name: "供应商 API 全部不可用"
      condition: "supplier_availability < 50%"
  
  P2_Medium:  # 1小时内处理
    - name: "评估分数下降"
      condition: "ragas_score decreased by 10% from baseline"
    - name: "Token消耗异常"
      condition: "hourly_token_consumption > 200% of average"
```

### 6.2 P0 故障处理 Runbook

```
步骤 1: 确认故障范围
  - 检查 Dashboard: 哪些指标异常？
  - 检查 影响范围: 全部用户？特定区域？
  - 检查 最近变更: 是否有发布？配置变更？

步骤 2: 止损
  - 如果最近有发布 → 立即回滚 (一键回滚脚本)
  - 如果供应商故障 → 切换到备用供应商
  - 如果自建模型故障 → 切换到 API 备路

步骤 3: 恢复
  - 确认服务恢复
  - 通知受影响用户
  - 创建 P1 Incident

步骤 4: 复盘
  - 24h 内完成 RCA (Root Cause Analysis)
  - 更新 Runbook
  - 创建 Action Items
```

---

## 7. 生产检查清单

### 上线前检查

- [ ] SLO 定义完成并写入 SLA
- [ ] 监控 Dashboard 上线
- [ ] 告警规则配置并测试
- [ ] 灰度发布脚本就绪
- [ ] 自动回滚脚本就绪
- [ ] GPU 容量规划文档更新
- [ ] On-Call 排班确认
- [ ] Runbook 已更新
- [ ] 灾备演练完成
- [ ] 负载测试通过 (1.5x 峰值 QPS)
- [ ] 安全审计通过

---

*最后更新：2026-05-29*

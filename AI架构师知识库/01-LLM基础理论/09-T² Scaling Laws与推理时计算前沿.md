# T² Scaling Laws 与推理时计算前沿

> Train-to-Test Scaling Laws：在训练 + 推理总预算约束下联合优化模型规模和推理算力

## 1. T² Scaling Laws（Train-to-Test）

### 1.1 核心发现

传统 Chinchilla Scaling Laws 只优化训练预算（模型大小 N × 训练 token D）。但 2026 年的研究表明，当**推理时算力**被纳入预算约束时，最优解发生根本性转变。

**T² Scaling Laws（arXiv 2604.01411, 2026）** 的核心发现：

```
传统 Chinchilla:
  目标：给定训练预算 C_train = 6ND，最小化 loss
  假设：推理算力无限且免费
  
T² Scaling Laws:
  目标：给定总预算 C_total = C_train + C_inf，最小化 loss
  约束：推理预算 C_inf = 2Nk（k = 推理时采样次数）
  结论：模型更小、更过度训练、搭配足够推理时采样
```

### 1.2 方法

采用两种互补建模方法，结论高度一致：

| 方法 | 建模目标 | 输入 | 输出 |
|------|----------|------|------|
| Approach 1 | NLL（负对数似然） | N, D, k | 预测 loss |
| Approach 2 | pass@k | N, D, k | 预测准确率 |

两者都表明：**当 test-time compute 纳入训练决策后，最优模型比 Chinchilla 推荐的更小更过度训练。**

### 1.3 实践建议

> 如果你知道 test-time scaling 预算（推理时可通过重复采样投入多少算力），就应该训练一个更小的模型更长时间。

具体步骤：
1. 确定推理预算 C_inf（每请求允许的推理 FLOPs）
2. 使用 T² Scaling Laws 查找最优 (N, D, k) 组合
3. 训练更小模型 + 更多数据 + 推理时多次采样

## 2. Test-Time Compute 三阶段

推理时算力的投入效果呈现明显的三阶段特征：

```
准确率
  │
  ├── 线性区 ── 饱和区 ──── 突破区
  │
  │  线性区（Linear Regime）
  │   · 低预算、简单到中等任务
  │   · 每加倍推理算力 ≈ 等比例收益
  │
  │      饱和区（Saturation Regime）
  │       · 中等预算，任务可解
  │       · 接近准确率天花板
  │       · 额外算力大部分浪费
  │
  │               突破区（Breakthrough Regime）
  │                · 极高预算，困难任务
  │                · 偶尔解锁低预算无法解决的问题
  │                · 离散跳跃而非平滑增长
  │
  └───────────────────────────────────→ 推理算力（log）
```

**生产启示**：大多数生产流量处于饱和区，这意味额外的 thinking tokens 大部分是浪费的。只有在真正困难的问题上才值得投入高预算。

## 3. Reasoning Floor（推理地板）

### 3.1 核心发现

**推理地板**（Reasoning Floor）是一个关键的性能天花板：

> 通用模型（non-reasoning models）即使增加 10 倍的推理计算，也无法达到推理优化权重（reasoning-optimized weights）的基线性能。

### 3.2 实证结果

| 对比 | 推理预算 | 表现 |
|------|----------|------|
| Llama-3.3-70B Instruct（通用） | N=256 采样 | 低于基线 |
| R1-Distill Llama-70B（推理优化） | N=1 | 基线 |

**结论：** 通过 RL 内化推理协议，比外部搜索方法更有效。推理优化权重提供的是"内建的推理能力"，外部搜索无法弥补这个差距。

## 4. Parallel Thinking 与 VGS

### 4.1 Parallel Thinking

并行生成多个 rollout 并聚合结果，相比顺序搜索大幅降低延迟：

```
Sequential: 思考 → 思考 → 思考 → 回答（延迟 = sum）
Parallel:   思考 ─┐
            思考 ─┼─→ 聚合 → 回答（延迟 = max）
            思考 ─┘
```

### 4.2 Value-Guided Search（VGS）

使用值模型引导树搜索，在关键决策点上进行深入探索：

1. 训练值模型（使用任务的 reward signal）
2. 推理时在不确定的分支上执行树搜索
3. 值模型评估每条路径的期望收益
4. 选择最优路径生成答案

## 5. IBM ASTER / I3 Agent

### 5.1 ASTER（自动化测试生成）

| 指标 | 效果 |
|------|------|
| 模型 | Devstral 24B |
| 行覆盖率提升 | +20% ~ +45% |
| Token 消耗 | 比 SOTA coding agent 低 15x |
| 应用 | 75 个内部 Java 应用（最高 67k 行） |

### 5.2 I3 Agent（IT 事件响应）

| 指标 | 效果 |
|------|------|
| 对比基准 | ReAct + GPT-5.1 |
| 性能提升 | 最高 4.0x |
| Token 消耗 | 降低 30x |
| K8s 诊断 token | 降低 3.7x |
| Bug 修复 token | 降低 5.9x |

**核心结论：** 结构化 Agent 逻辑 > 裸 ReAct + Frontier Model。架构，而不是参数量，成为决定性变量。

## 参考资料

- T² Scaling Laws: arXiv 2604.01411（2026）
- Think Deep, Think Fast: Inference-Time Scaling And The Reasoning Floor（2026）
- IBM Research: ASTER & I3 Agent（2026-06）
- Effective Frontiers: A Unification of Neural Scaling Laws, arXiv 2602.02593

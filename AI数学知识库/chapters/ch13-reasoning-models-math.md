# 第十三章：推理模型的数学理论

> 2024-2025年，以OpenAI o1、DeepSeek-R1为代表的推理模型标志着大模型发展的新范式——从"训练时计算扩展"转向"推理时计算扩展"。这些模型通过在推理阶段投入更多计算资源，实现了复杂推理能力的涌现。本章将系统阐述推理模型背后的数学理论，包括**Test-Time Compute Scaling**、**GRPO算法**、**思维链涌现理论**、**过程奖励模型**和**推理预算优化**等核心内容。

## 目录

1. [Test-Time Compute Scaling理论](#1-test-time-compute-scaling理论)
2. [GRPO算法数学推导](#2-grpo算法数学推导)
3. [思维链涌现的数学模型](#3-思维链涌现的数学模型)
4. [过程奖励模型（PRM）数学](#4-过程奖励模型prm数学)
5. [推理预算优化](#5-推理预算优化)

---

## 1. Test-Time Compute Scaling理论

### 1.1 推理时计算扩展的基本概念

**定义 13.1（推理时计算）** 推理时计算（Test-Time Compute）是指在模型推理阶段投入的额外计算资源，用于提升输出质量。形式化地，给定输入 $x$ 和模型 $\pi_\theta$，推理时计算定义为：

$$
C_{\text{test}}(x) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \text{FLOPs}(\text{generate}(y|x)) \right]
$$

**与传统训练时计算的区别：**

| 计算类型 | 发生阶段 | 目标 | 可扩展性 |
|:---:|:---:|:---:|:---:|
| 训练时计算 | 训练阶段 | 优化模型参数 $\theta$ | 受数据量限制 |
| 推理时计算 | 推理阶段 | 优化输出质量 | 理论上无上限 |

**核心洞察：** 推理时计算提供了一种新的扩展维度——通过增加推理计算来换取性能提升，而非单纯增大模型规模。

### 1.2 推理预算与性能的幂律关系

**假设 13.1（Test-Time Scaling假设）** 对于给定任务，模型性能 $P$ 与推理计算预算 $C$ 满足幂律关系：

$$
P(C) = P_{\infty} - \alpha \cdot C^{-\beta}
$$

其中：
- $P_{\infty}$：性能上界（无限计算时的性能）
- $\alpha > 0$：性能差距系数
- $\beta > 0$：缩放指数

**定理 13.1（Test-Time Scaling定律）** 在以下条件下，推理时计算扩展满足对数缩放律：

$$
\text{Accuracy}(C) \approx a \log(C) + b
$$

**条件：**
1. 任务具有可分解的推理结构
2. 模型具备足够的推理能力
3. 验证器能够准确评估中间步骤

**证明：** 假设任务可以分解为 $n$ 个子问题，每个子问题的成功概率为 $p$。通过并行采样 $k$ 次，至少一个正确答案的概率为：

$$
P_{\text{success}}(k) = 1 - (1-p)^k
$$

当 $p$ 较小且 $k$ 较大时，使用近似 $(1-p)^k \approx e^{-pk}$：

$$
P_{\text{success}}(k) \approx 1 - e^{-pk}
$$

令 $C = k \cdot c_0$（$c_0$ 为单次采样成本），则：

$$
P_{\text{success}}(C) \approx 1 - e^{-pC/c_0}
$$

对于中等难度任务（$p$ 适中），在 $C$ 较大时：

$$
P_{\text{success}}(C) \approx 1 - \frac{c_0}{pC} = 1 - O\left(\frac{1}{C}\right)
$$

对于困难任务（$p$ 很小），需要使用Best-of-N策略配合验证器。设验证器准确率为 $q$，则：

$$
P_{\text{success}}(N) = \sum_{i=1}^{N} \binom{N}{i} p^i (1-p)^{N-i} \cdot q
$$

当 $Np \gg 1$ 时，正确样本数量服从泊松分布 $\text{Poisson}(Np)$，至少一个正确样本的概率约为 $1 - e^{-Np}$。

$\blacksquare$

### 1.3 最优计算分配策略

#### 1.3.1 并行采样 vs 串行验证

**问题设定：** 给定总计算预算 $C_{\text{total}}$，如何分配并行采样数 $N$ 和串行验证深度 $D$？

$$
C_{\text{total}} = N \cdot c_{\text{sample}} + D \cdot c_{\text{verify}}
$$

**定义 13.2（并行采样策略）** 生成 $N$ 个独立候选答案，使用验证器选择最优：

$$
y^* = \arg\max_{y_i} V(x, y_i), \quad i = 1, \ldots, N
$$

**定义 13.3（串行验证策略）** 迭代改进单个答案，进行 $D$ 轮修正：

$$
y^{(d+1)} = \text{Refine}(x, y^{(d)}, \text{Feedback}(x, y^{(d)}))
$$

**定理 13.2（最优分配定理）** 设采样成功率 $p$，验证准确率 $q$，修正成功率 $r$。最优分配策略为：

$$
\frac{N^*}{D^*} = \sqrt{\frac{c_{\text{verify}} \cdot r}{c_{\text{sample}} \cdot p \cdot q}}
$$

**证明：** 定义总成功率函数：

$$
S(N, D) = \underbrace{(1 - (1-p)^N) \cdot q}_{\text{并行贡献}} + \underbrace{(1 - (1-r)^D) \cdot (1 - (1-p)^N) \cdot (1-q)}_{\text{串行贡献}}
$$

简化模型：假设 $p, r \ll 1$，使用近似 $(1-p)^N \approx e^{-Np}$：

$$
S(N, D) \approx (1 - e^{-Np}) \cdot q + (1 - e^{-Dr}) \cdot (1 - e^{-Np}) \cdot (1-q)
$$

在预算约束 $N \cdot c_s + D \cdot c_v = C$ 下最大化 $S(N, D)$。使用拉格朗日乘数法：

$$
\mathcal{L} = S(N, D) - \lambda(N \cdot c_s + D \cdot c_v - C)
$$

对 $N$ 和 $D$ 分别求偏导并令其为零：

$$
\frac{\partial S}{\partial N} = p e^{-Np} \cdot q + p e^{-Np} \cdot (1 - e^{-Dr}) \cdot (1-q) = \lambda c_s
$$

$$
\frac{\partial S}{\partial D} = r e^{-Dr} \cdot (1 - e^{-Np}) \cdot (1-q) = \lambda c_v
$$

两式相除：

$$
\frac{p e^{-Np} \cdot [q + (1 - e^{-Dr})(1-q)]}{r e^{-Dr} \cdot (1 - e^{-Np}) \cdot (1-q)} = \frac{c_s}{c_v}
$$

当 $Np, Dr$ 适中时，近似解为：

$$
\frac{N}{D} \approx \sqrt{\frac{c_v \cdot r}{c_s \cdot p \cdot q}}
$$

$\blacksquare$

#### 1.3.2 计算最优扩展曲线

**定义 13.4（计算最优曲线）** 给定性能目标 $P_{\text{target}}$，最小计算预算为：

$$
C^*(P_{\text{target}}) = \min_{N, D} \{ N \cdot c_s + D \cdot c_v : S(N, D) \geq P_{\text{target}} \}
$$

**计算最优扩展的三种模式：**

```
性能
  ↑
  │                    ╭─────────────── 模式3：验证主导
  │                   ╱
  │                  ╱
  │                 ╱
  │                ╱
  │      ╭────────╯ 模式2：混合策略
  │     ╱
  │    ╱
  │   ╱ 模式1：采样主导
  │  ╱
  └──┴────────────────────────────────→ 计算预算
```

**模式分析：**

| 模式 | 条件 | 策略 | 适用场景 |
|:---:|:---:|:---:|:---:|
| 采样主导 | $p \cdot q > r$ | 增大 $N$，固定 $D=1$ | 选择题、简单推理 |
| 混合策略 | $p \cdot q \approx r$ | 同时增大 $N$ 和 $D$ | 中等难度推理 |
| 验证主导 | $p \cdot q < r$ | 固定 $N$，增大 $D$ | 复杂推理、数学证明 |

### 1.4 推理效率的理论上限

**定理 13.3（推理效率上界定理）** 对于任意推理策略，在计算预算 $C$ 下的性能上界为：

$$
P(C) \leq P_{\text{optimal}} \cdot \left(1 - e^{-\lambda C}\right)
$$

其中 $\lambda$ 是任务固有难度参数，$P_{\text{optimal}}$ 是最优策略的性能。

**证明：** 考虑信息论视角。设任务的熵为 $H(Y|X)$，每次推理步骤提供的信息量为 $I$。达到目标精度需要的信息总量为：

$$
I_{\text{total}} = H(Y|X) - H(Y|X, \hat{Y})
$$

其中 $\hat{Y}$ 是模型输出。每单位计算提供的信息量上界为：

$$
I_{\text{per\_compute}} \leq \log_2(|\mathcal{Y}|) \cdot \eta
$$

其中 $\eta$ 是计算效率因子。因此：

$$
C \geq \frac{I_{\text{total}}}{I_{\text{per\_compute}}} = \frac{H(Y|X) - H(Y|X, \hat{Y})}{\log_2(|\mathcal{Y}|) \cdot \eta}
$$

整理得性能上界：

$$
H(Y|X, \hat{Y}) \geq H(Y|X) - C \cdot \log_2(|\mathcal{Y}|) \cdot \eta
$$

使用Fano不等式，错误率 $P_e$ 满足：

$$
H(P_e) + P_e \log_2(|\mathcal{Y}|-1) \geq H(Y|X, \hat{Y})
$$

因此：

$$
P_e \geq f^{-1}\left(H(Y|X) - C \cdot \log_2(|\mathcal{Y}|) \cdot \eta\right)
$$

其中 $f(P_e) = H(P_e) + P_e \log_2(|\mathcal{Y}|-1)$。

$\blacksquare$

### 1.5 代码示例：Test-Time Scaling实现

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TestTimeConfig:
    num_samples: int = 16
    verification_depth: int = 3
    temperature: float = 0.7
    compute_budget: float = 100.0
    sample_cost: float = 1.0
    verify_cost: float = 2.0

class TestTimeScaler:
    def __init__(self, model, verifier, config: TestTimeConfig):
        self.model = model
        self.verifier = verifier
        self.config = config
    
    def optimal_allocation(self, difficulty: float) -> Tuple[int, int]:
        """
        根据任务难度计算最优N和D分配
        
        difficulty: 任务难度估计 (0-1)
        """
        c_s = self.config.sample_cost
        c_v = self.config.verify_cost
        
        p = 1 - difficulty
        q = 0.9
        r = 0.3 * (1 - difficulty)
        
        ratio = (c_v * r / (c_s * p * q)) ** 0.5
        
        total = self.config.compute_budget / (c_s + ratio * c_v)
        N = int(total)
        D = int(total * ratio)
        
        return max(1, N), max(1, D)
    
    def parallel_sampling(self, prompt: str, n: int) -> List[str]:
        """并行采样N个候选答案"""
        responses = []
        for _ in range(n):
            output = self.model.generate(
                prompt,
                temperature=self.config.temperature,
                do_sample=True
            )
            responses.append(output)
        return responses
    
    def serial_verification(self, prompt: str, response: str, depth: int) -> str:
        """串行验证和修正"""
        current = response
        for _ in range(depth):
            feedback = self.verifier.get_feedback(prompt, current)
            if feedback.is_correct:
                break
            current = self.model.refine(prompt, current, feedback)
        return current
    
    def compute_optimal_inference(self, prompt: str, difficulty: float = 0.5) -> str:
        """计算最优推理"""
        N, D = self.optimal_allocation(difficulty)
        
        candidates = self.parallel_sampling(prompt, N)
        
        refined = []
        for candidate in candidates:
            refined.append(self.serial_verification(prompt, candidate, D))
        
        scores = [self.verifier.score(prompt, r) for r in refined]
        best_idx = scores.index(max(scores))
        
        return refined[best_idx]

class Verifier:
    def __init__(self, model):
        self.model = model
    
    def score(self, prompt: str, response: str) -> float:
        """对回答进行评分"""
        verification_prompt = f"""
        问题: {prompt}
        回答: {response}
        
        请评估上述回答的正确性，给出0-1的分数。
        """
        score = self.model.generate(verification_prompt)
        return float(score)
    
    def get_feedback(self, prompt: str, response: str):
        """获取修正反馈"""
        feedback_prompt = f"""
        问题: {prompt}
        当前回答: {response}
        
        请指出回答中的错误并提供修正建议。
        """
        feedback = self.model.generate(feedback_prompt)
        return Feedback(feedback)
```

---

## 2. GRPO算法数学推导

### 2.1 GRPO概述

**GRPO（Group Relative Policy Optimization）** 是DeepSeek-R1等推理模型使用的核心训练算法，是对PPO的重要改进。其核心创新在于：

1. **无需Critic模型**：通过组内相对奖励估计优势函数
2. **降低方差**：利用组内比较减少梯度估计方差
3. **简化训练**：避免价值函数训练的不稳定性

### 2.2 从PPO到GRPO的动机

#### 2.2.1 PPO的局限性

PPO需要训练一个价值函数 $V_\phi(s)$ 来估计优势函数：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

**问题：**
1. 价值函数训练需要额外参数和计算
2. 价值函数估计误差会传播到策略更新
3. 在语言模型场景下，状态空间巨大，价值函数难以准确估计

#### 2.2.2 GRPO的核心思想

**关键洞察：** 对于同一提示 $x$，生成一组回复 $\{y_1, y_2, \ldots, y_G\}$，通过组内比较估计每个回复的相对质量。

### 2.3 GRPO的数学框架

#### 2.3.1 组相对优势函数

**定义 13.5（组相对优势）** 给定提示 $x$ 和组大小 $G$，生成 $G$ 个回复 $\{y_1, \ldots, y_G\}$，每个回复的优势定义为：

$$
\hat{A}_i = \frac{r(x, y_i) - \mu_r}{\sigma_r}
$$

其中：

$$
\mu_r = \frac{1}{G} \sum_{j=1}^{G} r(x, y_j), \quad \sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r(x, y_j) - \mu_r)^2}
$$

**定理 13.4（组相对优势的无偏性）** 当奖励函数 $r(x, y)$ 满足 $\mathbb{E}[r(x, y)] = \bar{r}(x)$ 时，组相对优势是真实优势的无偏估计：

$$
\mathbb{E}[\hat{A}_i] = \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r(x)}
$$

**证明：** 由定义：

$$
\mathbb{E}[\hat{A}_i] = \mathbb{E}\left[\frac{r(x, y_i) - \mu_r}{\sigma_r}\right]
$$

由于 $\mu_r$ 是 $G$ 个独立样本的平均值：

$$
\mathbb{E}[\mu_r] = \frac{1}{G} \sum_{j=1}^{G} \mathbb{E}[r(x, y_j)] = \bar{r}(x)
$$

因此：

$$
\mathbb{E}[\hat{A}_i] = \frac{r(x, y_i) - \bar{r}(x)}{\mathbb{E}[\sigma_r]}
$$

$\blacksquare$

#### 2.3.2 GRPO目标函数

**定义 13.6（GRPO目标函数）** GRPO的目标函数为：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_\theta} \left[ \frac{1}{G} \sum_{i=1}^{G} \mathcal{L}_i(\theta) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

其中单个样本的损失为：

$$
\mathcal{L}_i(\theta) = -\hat{A}_i \cdot \min\left( \rho_i(\theta), \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \right)
$$

概率比为：

$$
\rho_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}
$$

#### 2.3.3 GRPO的完整推导

**步骤1：策略梯度基准**

由策略梯度定理，梯度估计为：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot A(x, y) \right]
$$

减去一个基线 $b(x)$ 不改变期望：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot (A(x, y) - b(x)) \right]
$$

**步骤2：组均值作为基线**

GRPO选择组均值作为基线：

$$
b(x) = \frac{1}{G} \sum_{j=1}^{G} r(x, y_j) = \mu_r
$$

**步骤3：组内比较**

对于组内第 $i$ 个样本，相对优势为：

$$
\hat{A}_i = r(x, y_i) - \mu_r
$$

归一化后：

$$
\hat{A}_i = \frac{r(x, y_i) - \mu_r}{\sigma_r + \epsilon}
$$

**步骤4：PPO裁剪**

应用PPO的裁剪机制防止过大更新：

$$
\mathcal{L}_i = -\hat{A}_i \cdot \min\left( \rho_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \right)
$$

### 2.4 梯度估计方差分析

**定理 13.5（GRPO方差缩减定理）** 设奖励方差为 $\sigma_r^2$，组大小为 $G$，则GRPO的梯度估计方差为：

$$
\text{Var}(\nabla_\theta \mathcal{L}_{\text{GRPO}}) = \frac{\sigma_r^2}{G} \cdot \left(1 - \frac{1}{G}\right) \cdot \mathbb{E}\left[\|\nabla_\theta \log \pi_\theta\|^2\right]
$$

相比于不使用基线的方法，方差缩减比例为：

$$
\text{Reduction} = 1 - \frac{1}{G}
$$

**证明：** 设 $g_i = \nabla_\theta \log \pi_\theta(y_i|x)$，则GRPO梯度估计为：

$$
\hat{g} = \frac{1}{G} \sum_{i=1}^{G} g_i \cdot \frac{r_i - \mu_r}{\sigma_r}
$$

计算方差：

$$
\begin{aligned}
\text{Var}(\hat{g}) &= \mathbb{E}\left[\left\|\hat{g} - \mathbb{E}[\hat{g}]\right\|^2\right] \\
&= \frac{1}{G^2} \sum_{i,j} \mathbb{E}\left[(g_i \cdot \frac{r_i - \mu_r}{\sigma_r}) \cdot (g_j \cdot \frac{r_j - \mu_r}{\sigma_r})\right]
\end{aligned}
$$

由于不同样本独立：

$$
\text{Var}(\hat{g}) = \frac{1}{G^2} \sum_{i=1}^{G} \mathbb{E}\left[\|g_i\|^2 \cdot \frac{(r_i - \mu_r)^2}{\sigma_r^2}\right]
$$

计算 $(r_i - \mu_r)^2$ 的期望：

$$
\mathbb{E}[(r_i - \mu_r)^2] = \mathbb{E}[r_i^2] - 2\mathbb{E}[r_i \mu_r] + \mathbb{E}[\mu_r^2]
$$

由于 $\mu_r = \frac{1}{G}\sum_j r_j$：

$$
\mathbb{E}[\mu_r^2] = \frac{\sigma_r^2}{G} + \bar{r}^2
$$

$$
\mathbb{E}[r_i \mu_r] = \frac{\sigma_r^2}{G} + \bar{r}^2
$$

因此：

$$
\mathbb{E}[(r_i - \mu_r)^2] = \sigma_r^2 - \frac{\sigma_r^2}{G} = \sigma_r^2 \left(1 - \frac{1}{G}\right)
$$

最终：

$$
\text{Var}(\hat{g}) = \frac{1}{G} \cdot \sigma_r^2 \left(1 - \frac{1}{G}\right) \cdot \mathbb{E}\left[\|g_i\|^2 / \sigma_r^2\right]
$$

$\blacksquare$

### 2.5 GRPO vs PPO的理论比较

| 特性 | PPO | GRPO |
|:---:|:---:|:---:|
| 价值函数 | 需要训练 $V_\phi$ | 不需要 |
| 参数量 | 策略 + 价值网络 | 仅策略网络 |
| 优势估计 | GAE（时序差分） | 组相对比较 |
| 方差控制 | 价值函数基线 | 组均值基线 |
| 计算开销 | 中等 | 较高（组采样） |
| 训练稳定性 | 依赖价值函数质量 | 更稳定 |

**GRPO的优势分析：**

1. **简化架构**：无需价值函数，减少50%参数
2. **稳定训练**：避免价值函数训练的不稳定性
3. **天然正则化**：组内比较自动提供相对奖励

**GRPO的劣势分析：**

1. **计算开销**：每次需要生成 $G$ 个样本
2. **内存需求**：需要存储 $G$ 个完整序列
3. **组大小权衡**：$G$ 太小方差大，$G$ 太大计算贵

### 2.6 代码示例：GRPO实现

```python
import torch
import torch.nn.functional as F
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_coef: float = 0.04
    learning_rate: float = 1e-6
    max_length: int = 2048

class GRPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, config: GRPOConfig):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=config.learning_rate
        )
    
    def generate_group(self, prompt: str, group_size: int) -> List[str]:
        """生成一组回复"""
        responses = []
        for _ in range(group_size):
            response = self.policy.generate(
                prompt,
                temperature=0.7,
                do_sample=True,
                max_length=self.config.max_length
            )
            responses.append(response)
        return responses
    
    def compute_log_probs(self, model, prompt: str, response: str) -> torch.Tensor:
        """计算对数概率"""
        input_ids = self.policy.tokenizer(
            prompt + response, 
            return_tensors='pt'
        )['input_ids'].to(self.policy.device)
        
        prompt_length = len(self.policy.tokenizer(prompt)['input_ids'])
        
        with torch.no_grad() if model != self.policy else torch.enable_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, prompt_length-1:-1, :]
            labels = input_ids[:, prompt_length:]
            
            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='none'
            )
            
        return log_probs.sum(dim=-1)
    
    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算组相对优势"""
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages
    
    def grpo_loss(
        self, 
        prompt: str, 
        responses: List[str],
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """计算GRPO损失"""
        G = len(responses)
        
        rewards = torch.tensor([
            self.reward_model.score(prompt, r) 
            for r in responses
        ], device=self.policy.device)
        
        advantages = self.compute_group_advantages(rewards)
        
        total_loss = 0.0
        for i, response in enumerate(responses):
            new_log_prob = self.compute_log_probs(self.policy, prompt, response)
            ref_log_prob = self.compute_log_probs(self.ref_model, prompt, response)
            
            ratio = torch.exp(new_log_prob - old_log_probs[i])
            
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            )
            
            policy_loss = -torch.min(
                ratio * advantages[i],
                clipped_ratio * advantages[i]
            )
            
            kl_penalty = self.config.kl_coef * (new_log_prob - ref_log_prob)
            
            total_loss = total_loss + policy_loss + kl_penalty
        
        return total_loss / G
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """单步训练"""
        total_loss = 0.0
        
        for prompt in prompts:
            responses = self.generate_group(prompt, self.config.group_size)
            
            old_log_probs = torch.stack([
                self.compute_log_probs(self.policy, prompt, r).detach()
                for r in responses
            ])
            
            loss = self.grpo_loss(prompt, responses, old_log_probs)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
        
        return {'loss': total_loss / len(prompts)}

class RewardModel:
    def __init__(self, model):
        self.model = model
    
    def score(self, prompt: str, response: str) -> float:
        """对回答进行评分"""
        scoring_prompt = f"""
        问题: {prompt}
        回答: {response}
        
        请对上述回答的质量进行评分（0-10分）：
        """
        score_text = self.model.generate(scoring_prompt, max_length=10)
        try:
            return float(score_text.strip()) / 10.0
        except:
            return 0.5
```

---

## 3. 思维链涌现的数学模型

### 3.1 思维链现象的形式化

**定义 13.7（思维链）** 思维链（Chain-of-Thought, CoT）是指模型在给出最终答案前，生成一系列中间推理步骤的过程。形式化地：

$$
\text{CoT}: x \rightarrow (r_1, r_2, \ldots, r_k, y)
$$

其中 $r_i$ 是第 $i$ 个推理步骤，$y$ 是最终答案。

**思维链的数学表示：**

$$
P(y|x) = \sum_{r_1, \ldots, r_k} P(r_1|x) \prod_{i=2}^{k} P(r_i|x, r_{<i}) \cdot P(y|x, r_{1:k})
$$

### 3.2 思维链长度与模型规模的关系

**假设 13.2（CoT缩放假设）** 有效思维链长度 $L_{\text{CoT}}$ 与模型参数量 $N$ 满足：

$$
L_{\text{CoT}} \propto \log(N)
$$

**定理 13.6（CoT能力涌现定理）** 设任务复杂度为 $C$（所需推理步骤数），模型参数量为 $N$。模型能够解决该任务的必要条件为：

$$
N \geq N_0 \cdot e^{\alpha C}
$$

其中 $N_0$ 是基础参数量，$\alpha > 0$ 是缩放系数。

**证明：** 假设模型每层可以执行一个基本推理操作，$L$ 层模型可以执行 $L$ 步推理。对于需要 $C$ 步推理的任务：

$$
\text{Capacity}(N) \geq C
$$

模型容量与参数量的关系（基于Scaling Laws）：

$$
\text{Capacity}(N) \approx \beta \log(N)
$$

因此：

$$
\beta \log(N) \geq C \implies N \geq e^{C/\beta}
$$

令 $\alpha = 1/\beta$，$N_0 = 1$，得证。

$\blacksquare$

### 3.3 涌现能力的阈值模型

**定义 13.8（涌现阈值）** 涌现阈值 $N_{\text{emerge}}$ 是模型突然获得某项能力的参数量临界点。

**涌现的相变模型：**

将模型能力视为相变过程，使用Ising模型类比：

$$
P(\text{success}|N) = \frac{1}{1 + e^{-\alpha(N - N_c)}}
$$

其中 $N_c$ 是临界参数量，$\alpha$ 控制相变陡峭程度。

**定理 13.7（涌现阈值估计）** 对于具有 $K$ 个子任务的任务，涌现阈值为：

$$
N_{\text{emerge}} = \max_{k=1,\ldots,K} N_k + \gamma \cdot \text{Var}(\{N_k\})
$$

其中 $N_k$ 是第 $k$ 个子任务的阈值，$\gamma$ 是协同系数。

**证明：** 任务成功需要所有子任务成功：

$$
P(\text{success}) = \prod_{k=1}^{K} P_k(\text{success})
$$

假设每个子任务的成功概率服从sigmoid：

$$
P_k(\text{success}) = \frac{1}{1 + e^{-\alpha_k(N - N_k)}}
$$

当 $N \gg N_k$ 时，$P_k \approx 1$；当 $N \ll N_k$ 时，$P_k \approx 0$。

总成功概率在 $N = \max_k N_k$ 附近快速上升，但子任务间的依赖关系会增加阈值：

$$
N_{\text{emerge}} \approx \max_k N_k + \text{协同修正}
$$

协同修正与子任务阈值的方差成正比。

$\blacksquare$

### 3.4 推理深度的数学表达

**定义 13.9（推理深度）** 推理深度 $D$ 是指从输入到输出所需的逻辑推理步数。

**推理深度的信息论刻画：**

$$
D(x, y) = \frac{I(X; Y)}{H(Y|X)} \cdot \frac{1}{\text{CompressionRatio}}
$$

其中：
- $I(X; Y)$：输入输出的互信息
- $H(Y|X)$：条件熵
- $\text{CompressionRatio}$：信息压缩比

**定理 13.8（推理深度下界）** 对于任意推理任务，所需的最小推理深度满足：

$$
D_{\min} \geq \frac{H(Y|X)}{\max_{\pi} I(\text{Step}_i; \text{Context})}
$$

**证明：** 设每步推理提供的信息量为 $I_i$，总信息需求为 $H(Y|X)$。由信息论：

$$
\sum_{i=1}^{D} I_i \geq H(Y|X)
$$

由于 $I_i \leq \max_{\pi} I(\text{Step}_i; \text{Context})$：

$$
D \cdot \max_{\pi} I(\text{Step}_i; \text{Context}) \geq H(Y|X)
$$

因此：

$$
D \geq \frac{H(Y|X)}{\max_{\pi} I(\text{Step}_i; \text{Context})}
$$

$\blacksquare$

### 3.5 思维链的优化目标

**CoT训练的目标函数：**

$$
\mathcal{L}_{\text{CoT}} = -\mathbb{E}_{(x, r, y) \sim \mathcal{D}} \left[ \log P_\theta(r, y|x) \right] + \lambda \cdot \mathcal{R}(r)
$$

其中 $\mathcal{R}(r)$ 是推理步骤的正则化项：

$$
\mathcal{R}(r) = -\sum_{i=1}^{k} \text{Validity}(r_i) + \alpha \cdot \text{Conciseness}(r)
$$

**自洽性目标（Self-Consistency）：**

$$
\mathcal{L}_{\text{SC}} = -\mathbb{E}_x \left[ \log \sum_{r: f(r)=y^*} P_\theta(r, y^*|x) \right]
$$

其中 $f(r)$ 是从推理路径提取答案的函数，$y^*$ 是正确答案。

### 3.6 代码示例：思维链推理

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CoTConfig:
    max_reasoning_steps: int = 10
    temperature: float = 0.7
    self_consistency_samples: int = 5

class ChainOfThoughtReasoner:
    def __init__(self, model, config: CoTConfig):
        self.model = model
        self.config = config
    
    def generate_reasoning_step(
        self, 
        prompt: str, 
        previous_steps: List[str]
    ) -> str:
        """生成单个推理步骤"""
        context = prompt + "\n".join(previous_steps)
        context += "\n下一步推理："
        
        step = self.model.generate(
            context,
            temperature=self.config.temperature,
            max_length=200
        )
        return step
    
    def extract_answer(self, reasoning: str) -> str:
        """从推理过程中提取最终答案"""
        extract_prompt = f"""
        推理过程：
        {reasoning}
        
        请从上述推理中提取最终答案：
        """
        answer = self.model.generate(extract_prompt, max_length=50)
        return answer.strip()
    
    def cot_inference(self, question: str) -> Tuple[str, str]:
        """完整的思维链推理"""
        prompt = f"问题：{question}\n让我们一步步思考：\n"
        
        reasoning_steps = []
        for _ in range(self.config.max_reasoning_steps):
            step = self.generate_reasoning_step(prompt, reasoning_steps)
            reasoning_steps.append(step)
            
            if "答案是" in step or "因此" in step:
                break
        
        full_reasoning = "\n".join(reasoning_steps)
        answer = self.extract_answer(full_reasoning)
        
        return full_reasoning, answer
    
    def self_consistency_inference(
        self, 
        question: str
    ) -> Tuple[str, str, float]:
        """自洽性推理"""
        results = []
        
        for _ in range(self.config.self_consistency_samples):
            reasoning, answer = self.cot_inference(question)
            results.append((reasoning, answer))
        
        answer_counts = {}
        for _, answer in results:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        best_answer = max(answer_counts, key=answer_counts.get)
        confidence = answer_counts[best_answer] / len(results)
        
        best_reasoning = next(
            r for r, a in results if a == best_answer
        )
        
        return best_reasoning, best_answer, confidence

class CoTTrainer:
    def __init__(self, model, tokenizer, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def compute_cot_loss(
        self, 
        question: str, 
        reasoning: str, 
        answer: str
    ) -> torch.Tensor:
        """计算CoT训练损失"""
        full_text = f"问题：{question}\n推理：{reasoning}\n答案：{answer}"
        
        inputs = self.tokenizer(
            full_text, 
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        labels = inputs['input_ids'].clone()
        
        question_ids = self.tokenizer(
            f"问题：{question}\n推理：",
            return_tensors='pt'
        )['input_ids']
        question_length = question_ids.shape[1]
        
        labels[:, :question_length] = -100
        
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        
        return outputs.loss
    
    def train_step(
        self, 
        questions: List[str], 
        reasonings: List[str], 
        answers: List[str]
    ) -> float:
        """单步训练"""
        total_loss = 0.0
        
        for q, r, a in zip(questions, reasonings, answers):
            loss = self.compute_cot_loss(q, r, a)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss / len(questions)
```

---

## 4. 过程奖励模型（PRM）数学

### 4.1 过程监督 vs 结果监督

**定义 13.10（结果奖励模型，ORM）** 结果奖励模型仅对最终输出评分：

$$
r_{\text{ORM}}(x, y) = \text{Score}(x, y)
$$

**定义 13.11（过程奖励模型，PRM）** 过程奖励模型对推理的每一步评分：

$$
r_{\text{PRM}}(x, r_1, r_2, \ldots, r_k) = \sum_{i=1}^{k} s(x, r_{\leq i})
$$

其中 $s(x, r_{\leq i})$ 是第 $i$ 步的步骤奖励。

**两种监督方式的比较：**

| 特性 | ORM | PRM |
|:---:|:---:|:---:|
| 监督粒度 | 整体输出 | 每个步骤 |
| 信号稀疏性 | 高（只有最终信号） | 低（每步有信号） |
| 训练数据需求 | 较少 | 较多（需要步骤标注） |
| 信用分配 | 困难 | 直接 |
| 可解释性 | 低 | 高 |

### 4.2 PRM的数学框架

#### 4.2.1 步骤价值函数

**定义 13.12（步骤价值函数）** 给定部分推理路径 $r_{<i}$，步骤价值函数定义为从该状态到达正确答案的期望概率：

$$
V(x, r_{<i}) = \mathbb{E}_{r_i, \ldots, r_k \sim \pi} \left[ \mathbf{1}[\text{Correct}(x, r_{1:k})] \right]
$$

**步骤奖励与步骤价值的关系：**

$$
s(x, r_{\leq i}) = V(x, r_{\leq i}) - V(x, r_{<i})
$$

#### 4.2.2 PRM训练目标

**数据格式：** PRM训练数据包含步骤级标注：

$$
\mathcal{D} = \{(x, r_1, \ldots, r_k, \ell_1, \ldots, \ell_k)\}
$$

其中 $\ell_i \in \{0, 1\}$ 表示第 $i$ 步是否正确。

**训练目标：**

$$
\mathcal{L}_{\text{PRM}} = -\sum_{i=1}^{k} \left[ \ell_i \log \sigma(s_\phi(x, r_{\leq i})) + (1-\ell_i) \log(1 - \sigma(s_\phi(x, r_{\leq i}))) \right]
$$

**定理 13.9（PRM的信用分配）** PRM提供精确的步骤级信用分配：

$$
\frac{\partial \mathcal{L}_{\text{PRM}}}{\partial s_\phi(x, r_{\leq i})} = \sigma(s_\phi(x, r_{\leq i})) - \ell_i
$$

**证明：** 由交叉熵损失的定义：

$$
\mathcal{L}_i = -\ell_i \log p_i - (1-\ell_i) \log(1-p_i)
$$

其中 $p_i = \sigma(s_\phi(x, r_{\leq i}))$。对 $s_\phi$ 求导：

$$
\frac{\partial \mathcal{L}_i}{\partial s_\phi} = -\ell_i \cdot \frac{1}{p_i} \cdot p_i(1-p_i) + (1-\ell_i) \cdot \frac{1}{1-p_i} \cdot p_i(1-p_i)
$$

简化：

$$
\frac{\partial \mathcal{L}_i}{\partial s_\phi} = -\ell_i(1-p_i) + (1-\ell_i)p_i = p_i - \ell_i
$$

$\blacksquare$

### 4.3 推理路径评分

#### 4.3.1 路径价值估计

**定义 13.13（路径价值）** 完整推理路径的价值定义为：

$$
V_{\text{path}}(x, r_{1:k}) = \prod_{i=1}^{k} \sigma(s_\phi(x, r_{\leq i}))
$$

或使用对数形式：

$$
\log V_{\text{path}}(x, r_{1:k}) = \sum_{i=1}^{k} \log \sigma(s_\phi(x, r_{\leq i}))
$$

#### 4.3.2 最优路径选择

**问题：** 给定多条候选推理路径，选择最优路径：

$$
r^* = \arg\max_{r \in \mathcal{R}} V_{\text{path}}(x, r)
$$

**束搜索策略：**

使用PRM指导的束搜索：

$$
\mathcal{B}_{i+1} = \text{TopK}\left\{ (r_{<i}, r_i) : r_{<i} \in \mathcal{B}_i, r_i \in \text{Candidates} \right\}
$$

排序依据：

$$
\text{Score}(r_{<i}, r_i) = \log P(r_i|x, r_{<i}) + \alpha \cdot s_\phi(x, r_{\leq i})
$$

### 4.4 PRM训练的数学分析

#### 4.4.1 方差分析

**定理 13.10（PRM方差优势）** 相比ORM，PRM的梯度估计方差降低比例为：

$$
\text{Variance Reduction} = 1 - \frac{1}{k}
$$

其中 $k$ 是推理步骤数。

**证明：** ORM的梯度方差主要来自最终奖励的稀疏性：

$$
\text{Var}(\nabla \mathcal{L}_{\text{ORM}}) \propto \text{Var}(r_{\text{final}})
$$

PRM将奖励分解为 $k$ 个步骤奖励：

$$
\text{Var}(\nabla \mathcal{L}_{\text{PRM}}) \propto \frac{1}{k} \sum_{i=1}^{k} \text{Var}(s_i)
$$

假设各步骤奖励方差相近：

$$
\text{Var}(\nabla \mathcal{L}_{\text{PRM}}) \approx \frac{1}{k} \text{Var}(\nabla \mathcal{L}_{\text{ORM}})
$$

$\blacksquare$

#### 4.4.2 PRM与强化学习的结合

**PRM-guided策略优化：**

$$
\mathcal{L}_{\text{PRM-PPO}} = \mathbb{E}_{x, r \sim \pi_\theta} \left[ \sum_{i=1}^{k} \nabla_\theta \log \pi_\theta(r_i|x, r_{<i}) \cdot \hat{A}_i \right]
$$

其中优势函数由PRM提供：

$$
\hat{A}_i = s_\phi(x, r_{\leq i}) - V_\phi(x, r_{<i})
$$

### 4.5 代码示例：PRM实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PRMConfig:
    hidden_size: int = 768
    num_labels: int = 1
    dropout: float = 0.1

class ProcessRewardModel(nn.Module):
    def __init__(self, base_model, config: PRMConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        self.step_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: List[List[int]]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            step_positions: 每个样本的步骤结束位置列表
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        step_scores = []
        for i, positions in enumerate(step_positions):
            sample_scores = []
            for pos in positions:
                step_hidden = hidden_states[i, pos, :]
                score = self.step_scorer(step_hidden)
                sample_scores.append(score)
            step_scores.append(torch.stack(sample_scores))
        
        return step_scores
    
    def compute_loss(
        self,
        step_scores: List[torch.Tensor],
        step_labels: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算PRM损失"""
        total_loss = 0.0
        num_steps = 0
        
        for scores, labels in zip(step_scores, step_labels):
            scores = scores.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                scores, labels.float(), reduction='sum'
            )
            total_loss += loss
            num_steps += len(labels)
        
        return total_loss / num_steps

class PRMTrainer:
    def __init__(
        self, 
        prm: ProcessRewardModel, 
        lr: float = 1e-5
    ):
        self.prm = prm
        self.optimizer = torch.optim.AdamW(prm.parameters(), lr=lr)
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: List[List[int]],
        step_labels: List[torch.Tensor]
    ) -> float:
        """单步训练"""
        step_scores = self.prm(input_ids, attention_mask, step_positions)
        loss = self.prm.compute_loss(step_scores, step_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

class PRMGuidedDecoder:
    def __init__(
        self, 
        model, 
        prm: ProcessRewardModel,
        beam_size: int = 4,
        prm_weight: float = 0.5
    ):
        self.model = model
        self.prm = prm
        self.beam_size = beam_size
        self.prm_weight = prm_weight
    
    def decode_with_prm(
        self, 
        prompt: str,
        max_steps: int = 10
    ) -> Tuple[str, List[float]]:
        """使用PRM指导解码"""
        current_text = prompt
        step_scores = []
        
        for step in range(max_steps):
            candidates = []
            
            for _ in range(self.beam_size):
                next_token = self.model.generate(
                    current_text,
                    temperature=0.7,
                    max_length=50
                )
                candidates.append(next_token)
            
            best_score = float('-inf')
            best_candidate = None
            
            for candidate in candidates:
                lm_score = self.model.score(current_text, candidate)
                
                prm_score = self.prm.score_step(prompt, current_text, candidate)
                
                combined_score = (
                    (1 - self.prm_weight) * lm_score + 
                    self.prm_weight * prm_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            current_text = best_candidate
            step_scores.append(best_score)
            
            if self.is_complete(current_text):
                break
        
        return current_text, step_scores
    
    def is_complete(self, text: str) -> bool:
        """检查推理是否完成"""
        completion_markers = ["答案是", "因此", "最终结果"]
        return any(marker in text for marker in completion_markers)
    
    def score_path(
        self, 
        prompt: str, 
        reasoning_path: str
    ) -> float:
        """对完整推理路径评分"""
        steps = self.extract_steps(reasoning_path)
        
        scores = []
        current = prompt
        for step in steps:
            score = self.prm.score_step(prompt, current, step)
            scores.append(score)
            current += step
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def extract_steps(self, reasoning: str) -> List[str]:
        """提取推理步骤"""
        steps = reasoning.split("\n")
        return [s.strip() for s in steps if s.strip()]
```

---

## 5. 推理预算优化

### 5.1 推理预算优化问题

**定义 13.14（推理预算优化问题）** 给定计算预算 $C$ 和性能目标 $P_{\text{target}}$，寻找最优推理策略 $\pi^*$：

$$
\pi^* = \arg\min_{\pi} C(\pi) \quad \text{s.t.} \quad P(\pi) \geq P_{\text{target}}
$$

或等价地：

$$
\pi^* = \arg\max_{\pi} P(\pi) \quad \text{s.t.} \quad C(\pi) \leq C_{\text{budget}}
$$

### 5.2 预算分配的优化理论

#### 5.2.1 多阶段推理的预算分配

**问题设定：** 推理过程分为 $K$ 个阶段，每个阶段的计算成本为 $c_k$，对最终性能的贡献为 $\Delta P_k$。

**优化目标：**

$$
\max_{\{n_k\}} \sum_{k=1}^{K} \Delta P_k(n_k) \quad \text{s.t.} \quad \sum_{k=1}^{K} n_k \cdot c_k \leq C
$$

其中 $n_k$ 是第 $k$ 阶段的迭代次数。

**定理 13.11（最优预算分配）** 当 $\Delta P_k(n_k)$ 为凹函数时，最优分配满足边际效益相等：

$$
\frac{\partial \Delta P_1}{\partial n_1} \cdot \frac{1}{c_1} = \frac{\partial \Delta P_2}{\partial n_2} \cdot \frac{1}{c_2} = \cdots = \frac{\partial \Delta P_K}{\partial n_K} \cdot \frac{1}{c_K}
$$

**证明：** 构造拉格朗日函数：

$$
\mathcal{L} = \sum_{k=1}^{K} \Delta P_k(n_k) - \lambda \left( \sum_{k=1}^{K} n_k \cdot c_k - C \right)
$$

对 $n_k$ 求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial n_k} = \frac{\partial \Delta P_k}{\partial n_k} - \lambda \cdot c_k = 0
$$

因此：

$$
\frac{\partial \Delta P_k}{\partial n_k} = \lambda \cdot c_k
$$

即：

$$
\frac{1}{c_k} \frac{\partial \Delta P_k}{\partial n_k} = \lambda \quad \forall k
$$

$\blacksquare$

#### 5.2.2 采样与验证的预算权衡

**模型：** 设采样成本 $c_s$，验证成本 $c_v$，采样成功率 $p$，验证准确率 $q$。

**总成功率：**

$$
P_{\text{success}}(N, V) = (1 - (1-p)^N) \cdot q^V
$$

**预算约束：**

$$
N \cdot c_s + V \cdot c_v = C
$$

**定理 13.12（采样-验证权衡）** 最优分配为：

$$
\frac{N^*}{V^*} = \frac{c_v}{c_s} \cdot \frac{\log(1/q)}{\log(1/(1-p))}
$$

**证明：** 将 $V = (C - N \cdot c_s) / c_v$ 代入成功率：

$$
P(N) = (1 - (1-p)^N) \cdot q^{(C - N \cdot c_s) / c_v}
$$

对 $N$ 求导：

$$
\frac{dP}{dN} = -\log(1-p) \cdot (1-p)^N \cdot q^V - \frac{c_s}{c_v} \log q \cdot (1 - (1-p)^N) \cdot q^V
$$

令导数为零：

$$
-\log(1-p) \cdot (1-p)^N = \frac{c_s}{c_v} \log q \cdot (1 - (1-p)^N)
$$

当 $p$ 较小时，$(1-p)^N \approx 1$：

$$
-\log(1-p) \approx -\frac{c_s}{c_v} \log q \cdot \frac{1 - (1-p)^N}{(1-p)^N}
$$

近似解：

$$
N^* \approx \frac{C}{c_s} \cdot \frac{\log(1/q)}{\log(1/q) + \frac{c_s}{c_v}\log(1/(1-p))}
$$

$\blacksquare$

### 5.3 早停策略的数学分析

#### 5.3.1 早停条件

**定义 13.15（置信度早停）** 当模型对当前答案的置信度超过阈值时停止：

$$
\text{Stop if } \max_y P(y|x, \text{reasoning}) \geq \tau
$$

**定义 13.16（收敛早停）** 当连续 $k$ 步推理的输出变化小于阈值时停止：

$$
\text{Stop if } \|\text{Embedding}(r_i) - \text{Embedding}(r_{i-1})\| < \epsilon \text{ for } i = t-k+1, \ldots, t
$$

#### 5.3.2 早停的最优阈值

**定理 13.13（最优早停阈值）** 设继续推理的成本为 $c$，正确率提升期望为 $\Delta P$，错误率降低期望为 $\Delta E$。最优早停阈值为：

$$
\tau^* = \frac{c}{c + \Delta E \cdot L_{\text{error}} - \Delta P \cdot G_{\text{correct}}}
$$

其中 $L_{\text{error}}$ 是错误答案的损失，$G_{\text{correct}}$ 是正确答案的收益。

**证明：** 定义期望价值函数：

$$
V(\tau) = P(\text{correct}|\tau) \cdot G - (1 - P(\text{correct}|\tau)) \cdot L - C(\tau)
$$

其中 $C(\tau)$ 是期望计算成本。对 $\tau$ 求导并令其为零：

$$
\frac{dV}{d\tau} = \frac{dP}{d\tau} \cdot (G + L) - \frac{dC}{d\tau} = 0
$$

因此：

$$
\frac{dP}{d\tau} = \frac{dC/d\tau}{G + L}
$$

假设 $P(\tau) \approx \tau$（置信度校准），$C(\tau) \approx c \cdot (1-\tau)$：

$$
\frac{dP}{d\tau} = 1, \quad \frac{dC}{d\tau} = -c
$$

最优条件：

$$
1 = \frac{-c}{G + L} \implies \tau^* = \frac{c}{G + L}
$$

更精细的分析考虑正确和错误的非对称性：

$$
\tau^* = \frac{c}{c + \Delta E \cdot L - \Delta P \cdot G}
$$

$\blacksquare$

### 5.4 推理-成本权衡曲线

**定义 13.17（Pareto前沿）** 推理-成本权衡的Pareto前沿定义为：

$$
\mathcal{F} = \{(C, P) : \nexists (C', P') \text{ s.t. } C' \leq C \text{ and } P' \geq P\}
$$

**Pareto前沿的参数化：**

$$
P(C) = P_{\max} - \alpha \cdot e^{-\beta C}
$$

其中 $\alpha, \beta$ 是任务相关参数。

**权衡效率指标：**

$$
\eta = \frac{dP}{dC} = \alpha \beta \cdot e^{-\beta C}
$$

### 5.5 自适应预算分配

#### 5.5.1 基于难度的自适应

**问题：** 不同问题需要不同的计算预算。

**难度估计：**

$$
\text{Difficulty}(x) = 1 - \max_y P_{\text{initial}}(y|x)
$$

**自适应预算：**

$$
C(x) = C_{\min} + (C_{\max} - C_{\min}) \cdot \text{Difficulty}(x)^\gamma
$$

#### 5.5.2 动态预算调整

**算法 13.1（动态预算调整）**

```
1. 初始化预算 C_0
2. 对于每个问题 x:
   a. 估计难度 d(x)
   b. 分配预算 C(x) = f(d(x))
   c. 执行推理
   d. 如果未达到目标:
      - 增加预算: C(x) := C(x) * (1 + α)
      - 继续推理
   e. 如果提前完成:
      - 记录节省的预算
```

### 5.6 代码示例：推理预算优化

```python
import torch
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

class StopReason(Enum):
    CONFIDENCE = "confidence"
    CONVERGENCE = "convergence"
    BUDGET = "budget"
    MAX_STEPS = "max_steps"

@dataclass
class BudgetConfig:
    min_budget: float = 10.0
    max_budget: float = 100.0
    confidence_threshold: float = 0.95
    convergence_threshold: float = 0.01
    max_steps: int = 20
    sample_cost: float = 1.0
    verify_cost: float = 2.0

class ReasoningBudgetOptimizer:
    def __init__(self, model, verifier, config: BudgetConfig):
        self.model = model
        self.verifier = verifier
        self.config = config
    
    def estimate_difficulty(self, prompt: str) -> float:
        """估计问题难度"""
        initial_response = self.model.generate(
            prompt, 
            temperature=0.0,
            max_length=100
        )
        
        confidence = self.model.get_confidence(prompt, initial_response)
        
        difficulty = 1 - confidence
        return difficulty
    
    def allocate_budget(self, difficulty: float) -> float:
        """根据难度分配预算"""
        gamma = 1.5
        budget = (
            self.config.min_budget + 
            (self.config.max_budget - self.config.min_budget) * (difficulty ** gamma)
        )
        return budget
    
    def optimal_sample_verify_ratio(self, success_rate: float, verify_accuracy: float) -> float:
        """计算最优采样-验证比例"""
        c_s = self.config.sample_cost
        c_v = self.config.verify_cost
        p = success_rate
        q = verify_accuracy
        
        if p <= 0 or q <= 0:
            return 1.0
        
        ratio = (c_v / c_s) * (np.log(1/q) / np.log(1/(1-p)))
        return max(0.1, min(10.0, ratio))
    
    def check_early_stop(
        self, 
        current_step: int,
        confidence: float,
        prev_embedding: torch.Tensor,
        curr_embedding: torch.Tensor
    ) -> Tuple[bool, StopReason]:
        """检查是否应该早停"""
        if confidence >= self.config.confidence_threshold:
            return True, StopReason.CONFIDENCE
        
        if prev_embedding is not None:
            convergence = torch.norm(curr_embedding - prev_embedding).item()
            if convergence < self.config.convergence_threshold:
                return True, StopReason.CONVERGENCE
        
        if current_step >= self.config.max_steps:
            return True, StopReason.MAX_STEPS
        
        return False, None
    
    def optimize_inference(
        self, 
        prompt: str
    ) -> Tuple[str, float, dict]:
        """优化推理过程"""
        difficulty = self.estimate_difficulty(prompt)
        budget = self.allocate_budget(difficulty)
        
        ratio = self.optimal_sample_verify_ratio(
            success_rate=1 - difficulty,
            verify_accuracy=0.9
        )
        
        num_samples = max(1, int(np.sqrt(budget / self.config.sample_cost / ratio)))
        num_verifications = max(1, int(num_samples * ratio))
        
        candidates = []
        for _ in range(num_samples):
            response = self.model.generate(prompt, temperature=0.7)
            candidates.append(response)
        
        best_response = None
        best_score = float('-inf')
        total_cost = 0
        
        prev_embedding = None
        for step, candidate in enumerate(candidates[:num_verifications]):
            score = self.verifier.score(prompt, candidate)
            total_cost += self.config.verify_cost
            
            if score > best_score:
                best_score = score
                best_response = candidate
            
            confidence = torch.sigmoid(torch.tensor(score))
            curr_embedding = self.model.get_embedding(candidate)
            
            should_stop, reason = self.check_early_stop(
                step, confidence.item(), prev_embedding, curr_embedding
            )
            
            if should_stop:
                break
            
            prev_embedding = curr_embedding
        
        total_cost += num_samples * self.config.sample_cost
        
        stats = {
            'difficulty': difficulty,
            'allocated_budget': budget,
            'actual_cost': total_cost,
            'num_samples': num_samples,
            'num_verifications': min(len(candidates), num_verifications),
            'best_score': best_score
        }
        
        return best_response, best_score, stats

class AdaptiveBudgetScheduler:
    def __init__(self, initial_budget: float = 50.0, lr: float = 0.1):
        self.budget = initial_budget
        self.lr = lr
        self.history = []
    
    def update(self, success: bool, cost: float, performance: float):
        """根据结果更新预算"""
        self.history.append({
            'success': success,
            'cost': cost,
            'performance': performance
        })
        
        if len(self.history) < 10:
            return
        
        recent = self.history[-10:]
        success_rate = sum(h['success'] for h in recent) / len(recent)
        avg_cost = sum(h['cost'] for h in recent) / len(recent)
        
        if success_rate < 0.7:
            self.budget *= (1 + self.lr)
        elif success_rate > 0.9 and avg_cost < self.budget * 0.7:
            self.budget *= (1 - self.lr * 0.5)
    
    def get_budget(self) -> float:
        return self.budget

def compute_pareto_frontier(
    costs: List[float], 
    performances: List[float]
) -> List[Tuple[float, float]]:
    """计算Pareto前沿"""
    points = list(zip(costs, performances))
    points.sort(key=lambda x: x[0])
    
    frontier = []
    max_perf = float('-inf')
    
    for cost, perf in points:
        if perf > max_perf:
            frontier.append((cost, perf))
            max_perf = perf
    
    return frontier
```

---

## 本章小结

本章系统阐述了推理模型的数学理论，涵盖了2024-2025年推理模型发展的核心数学基础：

1. **Test-Time Compute Scaling理论**：推理时计算扩展的幂律关系、最优计算分配策略、推理效率的理论上限
2. **GRPO算法数学推导**：组相对优势函数、GRPO目标函数、梯度估计方差分析、与PPO的理论比较
3. **思维链涌现的数学模型**：思维链长度与模型规模的关系、涌现能力的阈值模型、推理深度的数学表达
4. **过程奖励模型（PRM）数学**：过程监督vs结果监督的数学框架、PRM训练目标、推理路径评分
5. **推理预算优化**：预算分配的优化问题、早停策略的数学分析、推理-成本权衡

**核心脉络：** 推理模型代表了大模型发展的新范式——从"训练时计算扩展"转向"推理时计算扩展"。这一转变的核心数学基础是：通过在推理阶段投入更多计算资源，可以在不增加模型参数的情况下提升推理能力。GRPO算法通过组相对比较消除了对价值函数的需求，PRM提供了精确的步骤级信用分配，而推理预算优化理论则为计算资源的高效利用提供了指导。

**关键公式速查：**

| 公式 | 表达 |
|:---:|:---:|
| Test-Time Scaling | $P(C) = P_{\infty} - \alpha \cdot C^{-\beta}$ |
| 最优分配比 | $\frac{N^*}{D^*} = \sqrt{\frac{c_v \cdot r}{c_s \cdot p \cdot q}}$ |
| GRPO优势 | $\hat{A}_i = \frac{r(x, y_i) - \mu_r}{\sigma_r}$ |
| GRPO方差缩减 | $\text{Reduction} = 1 - \frac{1}{G}$ |
| 涌现阈值 | $N_{\text{emerge}} = N_0 \cdot e^{\alpha C}$ |
| PRM损失 | $\mathcal{L}_{\text{PRM}} = -\sum_i [\ell_i \log \sigma(s_i) + (1-\ell_i)\log(1-\sigma(s_i))]$ |
| 最优早停阈值 | $\tau^* = \frac{c}{c + \Delta E \cdot L - \Delta P \cdot G}$ |

**未来展望：** 推理模型的数学理论仍在快速发展中，未来可能的研究方向包括：
- 更精细的推理时计算扩展定律
- 多模态推理的统一数学框架
- 推理效率与模型架构的联合优化
- 分布式推理的理论基础

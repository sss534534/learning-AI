# 第四章：信息论

> 信息论是20世纪最伟大的科学成就之一，由克劳德·香农在1948年创立。它不仅奠定了现代通信理论的基础，更深刻影响了机器学习和人工智能的发展。在大模型时代，**交叉熵**是训练语言模型的核心损失函数，**KL散度**驱动着知识蒸馏和变分推断。本章将深入讲解信息论的核心概念及其在大型语言模型中的应用。

## 目录

1. [信息与熵](#1-信息与熵)
2. [交叉熵](#2-交叉熵)
3. [KL散度](#3-kl散度)
4. [互信息](#4-互信息)
5. [信息论在机器学习中的应用](#5-信息论在机器学习中的应用)
6. [信息论在大模型中的应用](#6-信息论在大模型中的应用)

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [第三章：概率论与统计学](./ch03-probability.md)
- **关联文件**: [第九章：Transformer架构](./ch09-transformer.md), [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md)
- **最后更新**: 2026-06-12
---

## 1. 信息与熵

### 1.1 信息的度量

**信息量** 是衡量消息"惊喜程度"的数学量。

**直观理解：**
- "太阳从东方升起" → 信息量很低（必然事件，没什么惊喜）
- "明天股市暴涨" → 信息量很高（稀有事件，非常惊喜）

**信息的数学定义：**

事件 $x$ 的自信息（Self-Information）：
$$
I(x) = -\log P(x) = \log \frac{1}{P(x)}
$$

**为什么是对数？**
1. **可加性**：两个独立事件的信息量可加
   $I(x,y) = I(x) + I(y)$
2. **单调性**：概率越小的事件，信息量越大
3. **数学上简洁**：对数变换将乘法变为加法

**对数底数的选择：**

| 底数 | 单位 | 常见应用 |
|------|------|----------|
| 2 | 比特（bit） | 计算机科学、信息论 |
| e | 纳特（nat） | 数学、自然科学 |
| 10 | 哈特利（Hartley） | 通信工程 |

**换算关系：**
$$
1 \text{ nat} = \log_2 e \approx 1.443 \text{ bits}
$$

### 1.2 熵的定义

**熵（Entropy）** 是随机变量不确定性的度量，是自信息的期望值：

$$
H(X) = E[I(X)] = E[-\log P(X)] = -\sum_{x} P(x) \log P(x)
$$

**更直观的理解：**
- 熵是编码一个随机变量所需比特数的期望下界
- 熵越高，预测越困难，不确定性越大

```python
import numpy as np
import torch
import torch.nn.functional as F

def entropy(probs):
    """计算离散分布的熵（以比特为单位）"""
    # 过滤掉概率为0的情况
    probs = probs[probs > 0]
    return -torch.sum(probs * torch.log2(probs))

# 示例：比较不同分布的熵
uniform_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
biased_probs = torch.tensor([0.9, 0.05, 0.03, 0.02])

print(f"均匀分布熵: {entropy(uniform_probs):.4f} bits")  # 2.0 bits
print(f"偏斜分布熵: {entropy(biased_probs):.4f} bits")  # 低熵
```

### 1.3 熵的性质

**性质1：熵的非负性**
$$
H(X) \geq 0
$$
当且仅当 $P(X=x_0) = 1$ 时取等号（确定性分布）。

**性质2：最大值**
对于固定的取值数量 $n$，均匀分布熵最大：
$$
H(X) \leq \log n
$$

**性质3：联合熵**
$$
H(X, Y) = -\sum_x \sum_y P(x, y) \log P(x, y)
$$

**性质4：条件熵**
在已知 $Y$ 的条件下，$X$ 的剩余不确定性：
$$
H(X|Y) = -\sum_x \sum_y P(x, y) \log P(x|y)
$$

### 1.4 熵的图示理解

```
                    熵的可视化
                    
    P(成功)        熵值        概率分布
       ↓
     1.0 │ 0.0 ████████████████████ ▓▓▓▓▓▓▓▓▓▓
         │ 0.0 │                   │ 
     0.8 │ 0.2 ████████████████    │ ▓▓▓▓▓▓▓▓
         │     │                   │        
     0.6 │ 0.4 █████████████       │ ▓▓▓▓▓▓▓
         │     │                   │        
     0.5 │ 0.5 ████████████        │ ▓▓▓▓▓▓  ← 最大熵
         │     │                   │        
     0.4 │ 0.6 █████████          │ ▓▓▓▓▓
         │     │                   │        
     0.2 │ 0.8 ████               │ ▓▓▓
         │     │                   │        
     0.0 │ 1.0 ░░░░░░░░░░░░░░░░░░░ │ ░  ← 零熵
         └────────────────────────────────────
              0.0  0.2  0.4  0.6  0.8  1.0
                          熵 (bits)
                          
    H = 0        H最大         H = 0
    (确定)      (0.5,0.5)      (确定)
```

### 1.5 二元熵函数

对于伯努利分布 $P(X=1) = p$：
$$
H(p) = -p \log_2 p - (1-p) \log_2 (1-p)
$$

```python
import matplotlib.pyplot as plt
import numpy as np

p = np.linspace(0.001, 0.999, 100)
h = -p * np.log2(p) - (1-p) * np.log2(1-p)

plt.plot(p, h)
plt.xlabel('P(X=1)')
plt.ylabel('H(X) (bits)')
plt.title('二元熵函数')
plt.axhline(y=1, color='r', linestyle='--', label='最大熵=1 bit')
plt.legend()
plt.grid(True)
```

### 1.6 熵的几何意义

**最大熵原理：** 在所有满足已知约束的分布中，熵最大的分布是最"公平"的分布。

**应用：**
- 均匀分布是满足支撑约束的最大熵分布
- 高斯分布是满足均值和方差约束的最大熵分布
- 指数分布是满足均值约束的最大熵分布

---

## 2. 交叉熵

### 2.1 交叉熵的定义

**交叉熵（Cross-Entropy）** 衡量用错误分布 $q$ 来编码真实分布 $p$ 所需的信息量：

$$
H(P, Q) = -\sum_x P(x) \log Q(x)
$$

**对比：**
- $H(P) = -\sum_x P(x) \log P(x)$：用真实分布 $P$ 编码
- $H(P, Q) = -\sum_x P(x) \log Q(x)$：用假设分布 $Q$ 编码

**直觉：** 如果 $Q$ 越接近 $P$，交叉熵越接近真实熵。

### 2.2 交叉熵与熵、KL散度的关系

**核心恒等式：**
$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

其中 $D_{KL}(P \| Q)$ 是 KL 散度。

**推导：**
$$
\begin{aligned}
H(P, Q) &= -\sum_x P(x) \log Q(x) \\
&= -\sum_x P(x) \log \left(P(x) \cdot \frac{Q(x)}{P(x)}\right) \\
&= -\sum_x P(x) \log P(x) - \sum_x P(x) \log \frac{Q(x)}{P(x)} \\
&= H(P) + D_{KL}(P \| Q)
\end{aligned}
$$

### 2.3 交叉熵作为损失函数

在机器学习中，交叉熵损失是分类问题的标准损失函数。

**二分类交叉熵：**
$$
\mathcal{L}_{CE} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

**多分类交叉熵：**
$$
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$

```python
import torch
import torch.nn.functional as F

# 真实标签（one-hot编码）
y_true = torch.tensor([0, 0, 1])  # 第三类是正确答案

# 模型预测（未归一化的logits）
logits = torch.tensor([2.0, 1.0, 4.0])

# 方法1：直接使用cross_entropy（推荐，数值稳定）
loss = F.cross_entropy(logits, y_true.argmax())
print(f"交叉熵损失: {loss.item():.4f}")

# 方法2：手动计算（Softmax + NLL）
probs = F.softmax(logits, dim=-1)
loss_manual = -torch.log(probs[y_true.argmax()])
print(f"手动计算: {loss_manual.item():.4f}")

# 验证两者相等
print(f"\n预测概率分布: {probs}")
print(f"正确答案概率: {probs[2].item():.4f}")
print(f"-log(正确答案概率): {-torch.log(probs[2]).item():.4f}")
```

### 2.4 交叉熵损失的数值稳定性

**问题：** 当预测概率接近0时，$\log(\hat{y})$ 会变成负无穷。

**解决方案：** 使用 log-sum-exp 技巧

```python
def stable_cross_entropy(logits, targets):
    """
    数值稳定的交叉熵计算
    """
    # logits: 未归一化的对数概率
    # 应用 log-sum-exp 技巧
    max_logit = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logit
    
    # 计算 log(sum(exp(logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
    
    # 计算交叉熵
    ce = max_logit + log_sum_exp - logits.gather(-1, targets.unsqueeze(-1))
    
    return ce.mean()

# 示例
logits = torch.tensor([[100.0, 1.0, 0.0]])
targets = torch.tensor([0])

loss = stable_cross_entropy(logits, targets)
print(f"数值稳定的交叉熵: {loss.item():.4f}")
```

---

## 3. KL散度

### 3.1 KL散度的定义

**KL散度（Kullback-Leibler Divergence）** 衡量两个概率分布 $P$ 和 $Q$ 之间的"距离"：

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

**物理意义：**
- 使用分布 $Q$ 编码分布 $P$ 的额外信息量
- $D_{KL}(P \| Q) \geq 0$，当且仅当 $P = Q$ 时等于0

### 3.2 KL散度的性质

**性质1：非负性**
$$
D_{KL}(P \| Q) \geq 0
$$

**性质2：非对称性**
$$
D_{KL}(P \| Q) \neq D_{KL}(Q \| P)
$$

**重要提示：** KL散度不是真正的距离度量（不满足对称性和三角不等式）！

**性质3：链式法则**
$$
D_{KL}(P(X,Y) \| Q(X,Y)) = D_{KL}(P(X) \| Q(X)) + D_{KL}(P(Y|X) \| Q(Y|X))
$$

```python
import torch

def kl_divergence(p, q):
    """
    计算KL散度 D(P||Q)
    p, q: 概率分布张量（已归一化）
    """
    # 确保数值稳定
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    
    return torch.sum(p * torch.log(p / q))

# 示例
p = torch.tensor([0.5, 0.3, 0.2])
q = torch.tensor([0.4, 0.4, 0.2])

d_kl = kl_divergence(p, q)
print(f"D_KL(P||Q) = {d_kl:.4f} nats")

# 注意：D_KL(P||Q) ≠ D_KL(Q||P)
d_kl_reverse = kl_divergence(q, p)
print(f"D_KL(Q||P) = {d_kl_reverse:.4f} nats")
print(f"D_KL(P||Q) ≠ D_KL(Q||P)")
```

### 3.3 KL散度的直观理解

```
        KL散度的直观理解
        
    P分布: ████████████████    ████░░░░░
           高概率区域        低概率区域
    
    Q分布(接近P): ████████████████    ████░░░░
                  匹配较好          略有偏差
                  
    Q分布(远离P): ████████░░░░░    ████████████
                  高估低概率区域   低估高概率区域
                  
    Q分布(远离P): ████░░░░░░░    ████████████████
                  低估高概率区域   高估低概率区域
                  
    绿色区域：P概率高，Q概率低 → KL散度贡献大
    红色区域：P概率低，Q概率高 → KL散度贡献小
```

### 3.4 KL散度与交叉熵的关系

**关系图：**
```
                    H(P, Q)
                   /
                  /
                 /
                /
    H(P) ──────●─────────────────→ 分布Q
              /
             /
            /
    D_KL(P||Q)
    
    H(P, Q) = H(P) + D_KL(P||Q)
```

**训练神经网络时的意义：**
- $H(P)$ 是固定的（由数据决定）
- 最小化交叉熵 = 最小化KL散度
- 当 $D_{KL}(P\|Q) \to 0$ 时，模型分布 $Q$ 趋近真实分布 $P$

---

## 4. 互信息

### 4.1 互信息的定义

**互信息（Mutual Information）** 衡量两个随机变量之间的信息共享程度：

$$
I(X; Y) = D_{KL}(P(X, Y) \| P(X)P(Y)) = \sum_x \sum_y P(x, y) \log \frac{P(x, y)}{P(x)P(y)}
$$

### 4.2 互信息的等价形式

**形式1：熵差**
$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

**形式2：联合熵**
$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

**形式3：自信息**
$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

### 4.3 互信息的性质

| 性质 | 公式 |
|------|------|
| 非负性 | $I(X; Y) \geq 0$ |
| 对称性 | $I(X; Y) = I(Y; X)$ |
| 自信息上界 | $I(X; Y) \leq \min(H(X), H(Y))$ |
| 独立时为零 | $I(X; Y) = 0$ 当且仅当 $X \perp Y$ |

### 4.4 互信息在特征选择中的应用

```python
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# 示例：计算特征与目标变量的互信息
X = np.random.randn(100, 10)  # 100个样本，10个特征
y = X[:, 0] * 2 + X[:, 1] ** 2 + np.random.randn(100) * 0.1

# 计算每个特征与y的互信息
mi_scores = mutual_info_classif(X, y)
print("互信息分数:", mi_scores)

# 互信息高的特征与y相关性更强
```

---

## 5. 信息论在机器学习中的应用

### 5.1 最大熵原理与逻辑回归

逻辑回归的假设分布（伯努利分布）：
$$
P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-w^T x}}
$$

**最大熵解释：** 逻辑回归在满足线性约束的条件下，选择熵最大的分布，即最"不确定"的分布。

### 5.2 决策树中的信息增益

**信息增益（Information Gain）** 是特征选择的标准：

$$
IG(Y, X) = H(Y) - H(Y|X)
$$

```python
def information_gain(y, split_indices):
    """计算信息增益"""
    parent_entropy = entropy(y)
    
    left_entropy = entropy(y[split_indices])
    right_entropy = entropy(y[~split_indices])
    
    n = len(y)
    n_left = len(y[split_indices])
    n_right = n - n_left
    
    weighted_child_entropy = (n_left/n) * left_entropy + (n_right/n) * right_entropy
    
    return parent_entropy - weighted_child_entropy

# 示例
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# 假设按第0个特征的均值分割
threshold = X[:, 0].mean()
split = X[:, 0] <= threshold

ig = information_gain(torch.tensor(y, dtype=torch.float32), torch.tensor(split))
print(f"信息增益: {ig:.4f}")
```

### 5.3 变分推断中的信息论

变分推断最小化：
$$
D_{KL}(q(z) \| p(z|D)) = \int q(z) \log \frac{q(z)}{p(z|D)} dz
$$

这等价于最大化证据下界（ELBO）：
$$
\mathcal{L} = -D_{KL}(q(z) \| p(z)) + E_{q(z)}[\log p(D|z)]
$$

### 5.4 互信息正则化

**InfoMax** 原则：最大化输入与表示之间的互信息

```python
def info_nce_loss(z1, z2, temperature=0.1):
    """
    InfoNCE损失 - 对比学习的核心
    最大化正样本对之间的互信息
    """
    # z1, z2: 同一图像的两个增强视图
    # 负样本来自其他图像
    
    # 计算相似度
    sim = torch.mm(z1, z2.T) / temperature
    
    # 对角线是正样本，其他是负样本
    exp_sim = torch.exp(sim)
    
    # 分母：每个样本的正样本 + 所有负样本
    pos_sim = torch.diag(exp_sim)
    denominator = exp_sim.sum(dim=1)
    
    # InfoNCE损失
    loss = -torch.log(pos_sim / denominator).mean()
    
    return loss
```

---

## 6. 信息论在大模型中的应用

### 6.1 语言模型训练：交叉熵损失

大语言模型的核心目标是预测下一个词，这正是交叉熵损失的应用场景。

**语言模型的交叉熵损失：**
$$
\mathcal{L}_{CE} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(w_t | w_1, w_2, \ldots, w_{t-1})
$$

```python
import torch
import torch.nn as nn

class LanguageModelLoss(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
    
    def forward(self, logits, targets):
        """
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len] (词ID)
        """
        # 调整维度以匹配交叉熵
        # targets 中的每个位置对应 logits 中该位置的正确答案
        # 因此我们需要将 logits 的最后两个维度展平
        # [B, S, V] -> [B*S, V]
        # [B, S] -> [B*S]
        
        B, S, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.view(-1, V),  # [B*S, V]
            targets.view(-1),   # [B*S]
            reduction='mean'
        )
        return loss

# 示例
vocab_size = 50000
batch_size = 4
seq_len = 128

model = LanguageModelLoss(vocab_size)
logits = torch.randn(batch_size, seq_len, vocab_size)  # 模型输出
targets = torch.randint(0, vocab_size, (batch_size, seq_len))  # 目标词

loss = model(logits, targets)
print(f"语言模型交叉熵损失: {loss.item():.4f}")
```

### 6.2 困惑度：信息论视角

**困惑度（Perplexity）** 与熵的关系：

$$
\text{PP}(w_1^T) = 2^{-\frac{1}{T} \sum_{t=1}^{T} \log_2 P(w_t | w_1^{t-1})} = 2^{H_{\text{cross}}}
$$

其中 $H_{\text{cross}}$ 是交叉熵（以2为底）。

**直观理解：**
- 困惑度 = 模型的"分支因子"
- 如果 PP = 10，意味着模型在每一步平均有10个等可能的词
- 困惑度越低，语言模型越好

```python
def perplexity(log_probs):
    """
    计算困惑度
    log_probs: 对数概率，形状 [seq_len]
    """
    # 平均对数似然
    avg_log_prob = log_probs.mean()
    
    # 困惑度 = exp(-平均对数似然)
    perplexity = torch.exp(-avg_log_prob)
    
    return perplexity

# 示例
log_probs = torch.tensor([-0.5, -1.0, -0.8, -0.3])
pp = perplexity(log_probs)
print(f"困惑度: {pp.item():.2f}")
print(f"相当于每步有 {pp.item():.1f} 个等可能的词")
```

### 6.3 知识蒸馏：KL散度的应用

**知识蒸馏（Knowledge Distillation）** 使用大模型的软概率作为"教师"信号训练小模型。

**蒸馏损失：**
$$
\mathcal{L}_{\text{distil}} = D_{KL}(P_{\text{teacher}} \| P_{\text{student}})
$$

```python
def distillation_loss(student_logits, teacher_probs, temperature=2.0):
    """
    知识蒸馏损失
    """
    # 学生模型的软目标（带温度缩放）
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # 教师模型的软概率
    teacher_soft = teacher_probs
    
    # KL散度（注意顺序！）
    # 对于蒸馏，我们使用 D_KL(teacher || student)
    # 因为教师分布是固定的"真实"分布
    kl_div = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)  # 乘以T²补偿温度缩放
    
    return kl_div

def combined_loss(
    hard_loss,          # 硬目标损失（真实标签）
    soft_loss,          # 软目标损失（知识蒸馏）
    alpha=0.5           # 混合权重
):
    """
    组合损失函数
    α * hard_loss + (1-α) * soft_loss
    """
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

### 6.4 对比学习中的InfoNCE

**InfoNCE** 是对比学习的目标函数，本质上是互信息的下界估计：

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z, z^+)/\tau)}{\sum_{z'} \exp(\text{sim}(z, z')/\tau)}
$$

```python
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
    
    def forward(self, z1, z2):
        """
        SimCLR对比学习损失
        z1, z2: [batch_size, hidden_dim]，同一批数据的两个视图
        """
        batch_size = z1.shape[0]
        
        # L2归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 计算相似度矩阵
        sim = torch.mm(z1, z2.T) / self.tau  # [B, B]
        
        # 对角线是正样本对
        labels = torch.arange(batch_size, device=z1.device)
        
        # InfoNCE损失（双向）
        loss_1 = F.cross_entropy(sim, labels)
        loss_2 = F.cross_entropy(sim.T, labels)
        
        return (loss_1 + loss_2) / 2
```

### 6.5 最大互信息（GAN的核心）

**GAN的训练目标** 可以从信息论角度理解：

生成器 G 最大化 $I(z; G(z))$，即隐变量与生成样本之间的互信息越大越好。

### 6.6 ELBO与变分推断

**变分自编码器（VAE）** 的证据下界（ELBO）：

$$
\mathcal{L}_{\text{ELBO}} = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

**两项的信息论意义：**
1. **重构项**：$-E_{q}[\log P(x|z)]$ - 最大化期望对数似然
2. **正则项**：$D_{KL}(q\|p)$ - 确保后验接近先验

```python
class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_recon, x, mu, logvar):
        """
        VAE的ELBO损失
        """
        # 重构损失（负对数似然）
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL散度损失
        # D_KL(N(μ,σ) || N(0,1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # ELBO（要最小化，所以取负）
        loss = recon_loss + kl_loss
        
        return loss, recon_loss, kl_loss
```

---

## 本章小结

信息论提供了度量信息和不确定性的统一框架。关键要点：

1. **熵** 衡量随机变量的不确定性，是编码长度的理论下界
2. **交叉熵** 衡量用错误分布编码真实分布的代价
3. **KL散度** 衡量两个分布的差异，是知识蒸馏的数学基础
4. **互信息** 衡量变量之间的信息共享程度
5. 在大模型中，**交叉熵损失**是训练语言模型的核心
6. **困惑度**是评估语言模型的标准指标
7. **InfoNCE**和**知识蒸馏**都建立在信息论基础之上

## 深度分析

信息论为 LLM 提供了核心的训练目标和评估指标。交叉熵损失函数是所有自回归语言模型的标准训练目标——最小化交叉熵等价于最小化模型分布与数据真实分布之间的 KL 散度。从信息论视角看，LLM 训练的本质是通过有限的参数容量压缩训练数据中的信息，找到最低编码代价的表示。Perplexity（困惑度）作为模型评估标准，其数学本质是 $2^{H}$，即模型在每个位置平均需要多少比特来编码下一个 Token。

KL 散度在当代大模型技术中扮演着多重角色。知识蒸馏（Knowledge Distillation）通过最小化教师模型和学生模型输出分布之间的 KL 散度来传递知识；RLHF（Reinforcement Learning from Human Feedback）中的 PPO 优化包含 KL 惩罚项以防止策略偏离参考模型太远；在 LLM 对齐微调中，DPO（Direct Preference Optimization）也内在地使用了 KL 散度约束。理解互信息的概念有助于把握对比学习（如 SimCSE、InfoNCE）在句子表示学习中的工作原理。

## 核心概念检查

- [ ] 你能推导交叉熵损失与 KL 散度之间的关系？
- [ ] 你能解释为什么最小化交叉熵等价于最小化 KL 散度？
- [ ] 你能说明信息论中困惑度（Perplexity）与熵的关系？
- [ ] 你能解释知识蒸馏中 Temperature 缩放对 KL 散度计算的影响？
- [ ] 你能分析 RLHF 中 KL 惩罚项 $-\beta D_{KL}(\pi_{\theta} \| \pi_{ref})$ 的作用？
- [ ] 你能描述信息熵与 LLM 输出确定性之间的关系（高熵 vs 低熵生成）？
- [ ] 你能计算一个简单语言模型的交叉熵损失并推导其梯度？
- [ ] 你能说明互信息 $I(X;Y)$ 在对比学习损失 InfoNCE 中的具体体现？
- [ ] 你能解释最大熵原理如何与 Softmax 输出层相联系？
- [ ] 你能分析自信息 $I(x)=-\log P(x)$ 在 LLM 输出概率校准中的意义？

## 延伸阅读

- [第三章：概率论与统计学](./ch03-probability.md) - 信息熵的概率基础
- [第九章：Transformer架构](./ch09-transformer.md) - 语言模型中的交叉熵训练
- [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md) - 知识蒸馏中的 KL 散度
- [第五章：数值优化](./ch05-optimization.md) - 优化算法与损失函数

**最后更新**: 2026-06-12

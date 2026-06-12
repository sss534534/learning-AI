# 第三章：概率论与统计学

> 概率论是理解人工智能不确定性的数学语言。在大模型时代，概率论的重要性更加凸显：语言模型本质上是对下一个词的概率分布进行建模，生成式AI的核心是学习数据的概率分布。本章将深入讲解概率论的核心概念，并详细阐述其在大型语言模型中的应用。

## 目录

1. [概率基础](#1-概率基础)
2. [随机变量与概率分布](#2-随机变量与概率分布)
3. [多维随机变量](#3-多维随机变量)
4. [贝叶斯定理](#4-贝叶斯定理)
5. [估计理论](#5-估计理论)
6. [采样策略](#6-采样策略)
7. [概率在大模型中的应用](#7-概率在大模型中的应用)

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [第一章：线性代数](./ch01-linear-algebra.md)
- **关联文件**: [第四章：信息论](./ch04-information-theory.md), [第九章：Transformer架构](./ch09-transformer.md)
- **最后更新**: 2026-06-12
---

## 1. 概率基础

### 1.1 概率的定义

**概率（Probability）** 是描述某事件发生可能性大小的数值。

**公理化定义（Kolmogorov公理）：**
设 $\Omega$ 为样本空间，$\mathcal{F}$ 为事件域，概率函数 $P$ 满足：

1. **非负性**：$P(A) \geq 0$（任何事件概率非负）
2. **规范性**：$P(\Omega) = 1$（必然事件概率为1）
3. **可列可加性**：若 $A_1, A_2, \ldots$ 互不相容，则 $P(\bigcup_{i=1}^{\infty} A_i) = \sum_{i=1}^{\infty} P(A_i)$

### 1.2 概率的常见解释

| 解释类型 | 含义 | 应用场景 |
|----------|------|----------|
| **古典概型** | 等可能结果 | 掷骰子、抽牌 |
| **频率学派** | 大量重复试验的频率 | 统计推断 |
| **贝叶斯学派** | 主观信念的度量 | 贝叶斯推断 |

### 1.3 条件概率

**条件概率** 是在已知某事件发生的条件下，另一事件发生的概率：

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

**直观理解：** 在 B 已经发生的世界里，A 发生的概率。

**示例：** 在已知今天下雨的条件下，明天也下雨的概率。

### 1.4 乘法公式

由条件概率公式可得：
$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

**推广到多个事件：**
$$
P(A \cap B \cap C) = P(A) \cdot P(B|A) \cdot P(C|A \cap B)
$$

### 1.5 全概率公式

设 $B_1, B_2, \ldots, B_n$ 是样本空间的一个划分，则：
$$
P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)
$$

**几何直观：**
```
         A
    ┌─────────────┐
    │    B₁       │    B₂       B₃      B₄
    │  ┌───┐      │ ┌───┐  ┌───┐  ┌───┐
    │  │   │      │ │   │  │   │  │   │
    │  └───┘      │ └───┘  └───┘  └───┘
    │   P(A∩B₁)    │ P(A∩B₂)  P(A∩B₃) P(A∩B₄)
    └─────────────┘

P(A) = P(A∩B₁) + P(A∩B₂) + P(A∩B₃) + P(A∩B₄)
     = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + ...
```

---

## 2. 随机变量与概率分布

### 2.1 随机变量的定义

**随机变量（Random Variable）** 是从样本空间到实数的函数，将每个基本事件映射为一个数值。

**类型：**
- **离散随机变量**：取值可数（0, 1, 2, ...）
- **连续随机变量**：取值连续（任何实数值）

```python
import numpy as np
import matplotlib.pyplot as plt

# 离散分布示例：投掷骰子
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

plt.bar(outcomes, probabilities)
plt.title("均匀分布 (骰子)")
plt.xlabel("点数")
plt.ylabel("概率")
```

### 2.2 离散型分布

#### 2.2.1 伯努利分布（Bernoulli Distribution）

**定义：** 一次试验只有两种结果（成功/失败）

$$
P(X = x) = \begin{cases} 
1 - p & \text{if } x = 0 \\
p & \text{if } x = 1 
\end{cases}
$$

**期望：** $E[X] = p$
**方差：** $\text{Var}(X) = p(1-p)$

```python
import torch
from torch.distributions import Bernoulli

p = 0.7  # 成功概率
dist = Bernoulli(probs=p)
sample = dist.sample()  # 采样一个值
print(f"伯努利采样: {sample}")  # tensor([1.]) 或 tensor([0.])
```

#### 2.2.2 二项分布（Binomial Distribution）

**定义：** n 次独立伯努利试验中成功次数的分布

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n
$$

**期望：** $E[X] = np$
**方差：** $\text{Var}(X) = np(1-p)$

```python
from torch.distributions import Binomial

n, p = 10, 0.5  # 10次试验，成功概率0.5
dist = Binomial(total_count=n, probs=p)
samples = dist.sample((100,))  # 采样100次
print(f"二项分布样本均值: {samples.mean()}")
```

#### 2.2.3 分类分布（Categorical Distribution）

**定义：** K 个离散类别上的概率分布

$$
P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1
$$

这是**多分类问题**和**语言模型输出**的核心分布！

```python
from torch.distributions import Categorical

probs = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 4个类别的概率
dist = Categorical(probs=probs)
category = dist.sample()  # 采样一个类别
print(f"采样类别: {category}")  # tensor(2) 之类的
```

#### 2.2.4 泊松分布（Poisson Distribution）

**定义：** 单位时间内随机事件发生次数的分布

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

**期望：** $E[X] = \lambda$
**方差：** $\text{Var}(X) = \lambda$

### 2.3 连续型分布

#### 2.3.1 均匀分布（Uniform Distribution）

**定义：** 在区间 [a, b] 上等概率取值

$$
P(x) = \begin{cases} 
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

**期望：** $E[X] = \frac{a+b}{2}$
**方差：** $\text{Var}(X) = \frac{(b-a)^2}{12}$

#### 2.3.2 正态分布（Normal/Gaussian Distribution）

**定义：** 最重要的连续分布，自然现象中普遍存在

$$
P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**参数：**
- $\mu$：均值（位置参数）
- $\sigma^2$：方差（尺度参数）

**期望：** $E[X] = \mu$
**方差：** $\text{Var}(X) = \sigma^2$

**标准正态分布：** $\mu = 0, \sigma = 1$

```python
import torch
from torch.distributions import Normal

mu, sigma = 0, 1
dist = Normal(mu, sigma)
samples = dist.sample((1000,))
print(f"样本均值: {samples.mean():.3f}")  # ≈ 0
print(f"样本标准差: {samples.std():.3f}")  # ≈ 1
```

#### 2.3.3 指数分布（Exponential Distribution）

**定义：** 描述事件等待时间的分布

$$
P(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**期望：** $E[X] = \frac{1}{\lambda}$
**方差：** $\text{Var}(X) = \frac{1}{\lambda^2}$

### 2.4 分布函数

#### 2.4.1 概率质量函数（PMF）

离散随机变量取特定值的概率：
$$
p(x) = P(X = x)
$$

#### 2.4.2 概率密度函数（PDF）

连续随机变量满足：
$$
P(a \leq X \leq b) = \int_a^b p(x) \, dx
$$

**注意：** 对于连续变量，$P(X = x) = 0$，但 $p(x)$ 可以非零！

#### 2.4.3 累积分布函数（CDF）

$$
F(x) = P(X \leq x) = \begin{cases} 
\sum_{k \leq x} p(k) & \text{离散} \\
\int_{-\infty}^{x} p(t) \, dt & \text{连续}
\end{cases}
$$

### 2.5 矩与特征数

| 名称 | 公式 | 含义 |
|------|------|------|
| **一阶原点矩** | $E[X]$ | 均值 |
| **二阶中心矩** | $E[(X-\mu)^2]$ | 方差 |
| **三阶中心矩** | $E[(X-\mu)^3]$ | 偏度（偏斜程度） |
| **四阶中心矩** | $E[(X-\mu)^4]$ | 峰度（尖峰/平坦程度） |
| **协方差** | $E[(X-\mu_X)(Y-\mu_Y)]$ | 两个变量的关联 |

---

## 3. 多维随机变量

### 3.1 联合分布

**联合概率分布** 描述多个随机变量同时取值的概率：

**离散：**
$$
P(X = x, Y = y) = p(x, y)
$$

**连续：**
$$
P(a \leq X \leq b, c \leq Y \leq d) = \int_a^b \int_c^d p(x, y) \, dy \, dx
$$

### 3.2 边缘分布

从联合分布中得到单个变量的分布：

**离散：**
$$
P(X = x) = \sum_{y} P(X = x, Y = y)
$$

**连续：**
$$
p_X(x) = \int_{-\infty}^{+\infty} p(x, y) \, dy
$$

```python
import torch

# 定义联合分布
joint_probs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

# 计算边缘分布（对第二个维度求和）
margin_x = joint_probs.sum(dim=1)
margin_y = joint_probs.sum(dim=0)

print(f"P(X=0) = {margin_x[0]}")  # 0.3
print(f"P(X=1) = {margin_x[1]}")  # 0.7
```

### 3.3 条件分布

**条件概率分布** 是在已知另一变量取值时该变量的分布：

**离散：**
$$
P(X = x | Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)} = \frac{p(x, y)}{p_Y(y)}
$$

**连续：**
$$
p_{X|Y}(x|y) = \frac{p(x, y)}{p_Y(y)}
$$

### 3.4 独立性

**独立性的定义：**
$$
P(X = x, Y = y) = P(X = x) \cdot P(Y = y)
$$

或等价的：
$$
P(X = x | Y = y) = P(X = x)
$$

**深度学习中的独立性假设：**
- 条件独立假设在朴素贝叶斯、因子图模型中广泛应用
- 变分推断中使用平均场近似（假设隐变量相互独立）

### 3.5 协方差与相关系数

**协方差（Covariance）：**
$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

**相关系数（Correlation Coefficient）：**
$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y} = \frac{E[XY] - E[X]E[Y]}{\sqrt{E[X^2] - E[X]^2} \cdot \sqrt{E[Y^2] - E[Y]^2}}
$$

**性质：**
- $-1 \leq \rho_{XY} \leq 1$
- $\rho = 0$：不相关（但不一定独立）
- $|\rho| = 1$：完全线性相关

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

cov = np.cov(X, Y)[0, 1]  # 协方差
corr = np.corrcoef(X, Y)[0, 1]  # 相关系数

print(f"协方差: {cov}")   # 5.0
print(f"相关系数: {corr}") # 1.0（完全线性相关）
```

### 3.6 多元正态分布

**多元正态分布** 是高维数据分析的基础：

$$
p(\vec{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\vec{x}-\vec{\mu})^T \Sigma^{-1} (\vec{x}-\vec{\mu})\right)
$$

其中：
- $\vec{\mu}$：均值向量
- $\Sigma$：协方差矩阵

```python
from torch.distributions import MultivariateNormal

# 二元正态分布
mean = torch.zeros(2)
cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

dist = MultivariateNormal(mean, covariance_matrix=cov)
sample = dist.sample()
print(f"多元正态采样: {sample}")
```

---

## 4. 贝叶斯定理

### 4.1 贝叶斯定理的定义

**贝叶斯定理** 是概率论中最重要的公式之一：

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

**在机器学习中的形式：**
$$
P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}
$$

**术语：**
| 符号 | 名称 | 含义 |
|------|------|------|
| $P(\theta)$ | 先验分布 | 在看到数据前对参数的信念 |
| $P(D|\theta)$ | 似然函数 | 在参数为 $\theta$ 时观察到数据 D 的概率 |
| $P(\theta|D)$ | 后验分布 | 看到数据后对参数的更新信念 |
| $P(D)$ | 证据 | 归一化常数，保证后验是合法的概率分布 |

### 4.2 贝叶斯定理的推导

由条件概率的定义：
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B|A) = \frac{P(A \cap B)}{P(A)}
$$

消去 $P(A \cap B)$：
$$
P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

即得贝叶斯定理。

### 4.3 贝叶斯推断的直观理解

```
    贝叶斯更新流程
    
    先验 P(θ)
       │
       │  看到数据 D
       │
       ▼
    计算似然 P(D|θ)
       │
       │  应用贝叶斯公式
       │
       ▼
    后验 P(θ|D)
       │
       │  成为新的先验，继续学习
       ▼
    ...
```

**示例：** 疾病检测
- 先验：人群中患某病的概率 $P(\text{病}) = 0.01$
- 似然：患病检测呈阳性的概率 $P(\text{+| 病}) = 0.99$
- 检测准确性：$P(\text{+| 无病}) = 0.05$

计算检测阳性时真正患病的概率：
$$
P(\text{病} | \text{+}) = \frac{P(\text{+| 病}) \cdot P(\text{病})}{P(\text{+})}
$$

其中：
$$
P(\text{+}) = P(\text{+| 病})P(\text{病}) + P(\text{+| 无病})P(\text{无病}) = 0.99 \times 0.01 + 0.05 \times 0.99 \approx 0.0594
$$

所以：
$$
P(\text{病} | \text{+}) = \frac{0.99 \times 0.01}{0.0594} \approx 0.167
$$

**结论：** 即使检测呈阳性，真正患病的概率只有约16.7%！

### 4.4 最大后验估计（MAP）

**MAP估计** 是找到使后验概率最大的参数值：
$$
\theta_{\text{MAP}} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta) \cdot P(\theta)
$$

**与最大似然估计（MLE）的对比：**
- MLE：$\theta_{\text{MLE}} = \arg\max_\theta P(D|\theta)$
- MAP = MLE + 先验正则化

```python
import torch
from torch.distributions import Normal

# 数据
data = torch.tensor([2.1, 2.4, 2.5, 2.3, 2.2])

# 先验：均值~N(0, 10)
prior_mean = torch.tensor(0.0)
prior_std = torch.tensor(3.16)  # 方差10

# 似然参数
def log_likelihood(mu):
    return Normal(mu, 0.5).log_prob(data).sum()

# 先验
def log_prior(mu):
    return Normal(prior_mean, prior_std).log_prob(mu)

# MAP: 最大化后验
def log_posterior(mu):
    return log_likelihood(mu) + log_prior(mu)

# 简单梯度下降
mu = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.Adam([mu], lr=0.1)

for _ in range(100):
    optimizer.zero_grad()
    loss = -log_posterior(mu)
    loss.backward()
    optimizer.step()

print(f"MAP估计: μ = {mu.item():.3f}")
```

### 4.5 共轭先验

**共轭先验** 的数学性质：先验分布和后验分布属于同一分布族。

| 似然分布 | 共轭先验 | 后验分布 |
|----------|----------|----------|
| 伯努利 | Beta分布 | Beta分布 |
| 多项式/分类 | Dirichlet分布 | Dirichlet分布 |
| 正态（已知方差） | 正态分布 | 正态分布 |
| 正态（已知均值） | Inverse-Gamma | Inverse-Gamma |
| 泊松 | Gamma分布 | Gamma分布 |

**Beta-Bernoulli共轭示例：**
```python
from torch.distributions import Beta, Bernoulli

# Beta先验参数
alpha, beta_param = 2, 2  # 相当于先验观察到2次成功和2次失败

# 观测数据：5次试验，3次成功
successes, trials = 3, 5

# 后验分布
alpha_post = alpha + successes
beta_post = beta_param + (trials - successes)

posterior = Beta(alpha_post, beta_post)
print(f"后验Beta({alpha_post}, {beta_post})")

# MAP估计
map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
print(f"MAP估计: {map_estimate:.3f}")
```

### 4.6 贝叶斯方法在大模型中的应用

#### 4.6.1 贝叶斯神经网络

传统神经网络：每个权重是固定值
贝叶斯神经网络：每个权重是分布

**优势：**
- 自然提供不确定性估计
- 防止过拟合（权重平均）
- 可以增量学习

```python
# 简化的贝叶斯线性回归
import torch
from torch.distributions import Normal

# 定义权重分布的先验
W_prior = Normal(0, 1)  # 先验：均值0，标准差1

# 给定数据，更新权重分布
# 实际中用变分推断或MCMC
```

#### 4.6.2 变分推断

当后验分布难以精确计算时，使用变分推断近似：
$$
q(\theta) \approx P(\theta|D)
$$

目标是最小化 $KL(q(\theta) \| P(\theta|D))$。

---

## 5. 估计理论

### 5.1 点估计

**点估计** 用一个具体的数值估计未知参数：

$$
\hat{\theta} = \hat{\theta}(X_1, X_2, \ldots, X_n)
$$

**常见估计方法：**

#### 5.1.1 矩估计法（MME）

用样本矩匹配总体矩：
$$
E[X^k] = \frac{1}{n}\sum_{i=1}^{n} X_i^k
$$

#### 5.1.2 最大似然估计（MLE）

找到使观测数据出现概率最大的参数值：
$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(D|\theta)
$$

**对数似然：** 乘积变求和，便于计算
$$
\ell(\theta) = \log P(D|\theta) = \sum_{i=1}^{n} \log P(x_i|\theta)
$$

```python
import torch
from torch.distributions import Normal

# 生成数据
true_mu, true_sigma = 3.0, 1.5
data = Normal(true_mu, true_sigma).sample((100,))

# MLE估计正态分布参数
sample_mean = data.mean()
sample_std = data.std(unbiased=True)

print(f"真实参数: μ={true_mu}, σ={true_sigma}")
print(f"MLE估计: μ̂={sample_mean:.3f}, σ̂={sample_std:.3f}")
```

### 5.2 估计量的评价标准

| 标准 | 定义 | 含义 |
|------|------|------|
| **无偏性** | $E[\hat{\theta}] = \theta$ | 估计量期望等于真实值 |
| **有效性** | $\text{Var}(\hat{\theta}_1) < \text{Var}(\hat{\theta}_2)$ | 方差越小越有效 |
| **一致性** | $\hat{\theta}_n \to \theta$ 当 $n \to \infty$ | 样本越大越准确 |

### 5.3 置信区间

**置信区间** 给出参数估计的不确定性范围：

对于 95% 置信区间：
$$
P(\hat{\theta} - 1.96 \cdot \text{SE} \leq \theta \leq \hat{\theta} + 1.96 \cdot \text{SE}) = 0.95
$$

```python
import scipy.stats as stats
import numpy as np

# 数据
data = np.array([2.1, 2.4, 2.5, 2.3, 2.2, 2.6, 2.0, 2.3])

# 计算95%置信区间
n = len(data)
mean = data.mean()
se = stats.sem(data)  # 标准误差
ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)

print(f"均值: {mean:.3f}")
print(f"95%置信区间: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 5.4 假设检验

**假设检验** 检验关于总体的假设：

1. **原假设** $H_0$：通常是我们想要拒绝的假设
2. **备择假设** $H_1$：通常是我们想要证明的假设

**p值：** 在原假设成立的前提下，观察到当前数据或更极端数据的概率。

```python
from scipy import stats

# t检验示例：检验两组数据是否有显著差异
group1 = [80, 85, 78, 92, 88]
group2 = [75, 82, 79, 86, 81]

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t统计量: {t_stat:.3f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print("拒绝原假设：两组有显著差异")
else:
    print("无法拒绝原假设")
```

---

## 6. 采样策略

### 6.1 采样的重要性

在大模型中，**采样** 是生成过程的核心：

```
    模型输出logits
         │
         ▼
    [转换为概率分布] (Softmax)
         │
         ▼
    [选择采样策略]
         │
    ┌────┴────┐
    ▼         ▼
  贪婪     随机采样
  采样         │
          ┌────┴────┐
          ▼    ▼    ▼
        Top-k  Top-p Temperature
```

### 6.2 Softmax函数

**Softmax** 将任意实数向量转换为概率分布：

$$
P(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

**性质：**
- 输出值在 (0, 1) 之间
- 输出和为 1
- 保持相对大小关系

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, -1.0])

# 方法1：PyTorch内置
probs = F.softmax(logits, dim=-1)
print(f"概率分布: {probs}")

# 方法2：手动实现
def softmax(x):
    exp_x = torch.exp(x - torch.max(x))  # 数值稳定化
    return exp_x / torch.sum(exp_x)

probs_manual = softmax(logits)
print(f"手动计算: {probs_manual}")
```

**数值稳定性问题：**
当 $x_i$ 很大时，$e^{x_i}$ 可能溢出。解决方案：
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

### 6.3 贪婪采样（Greedy Sampling）

**策略：** 总是选择概率最高的词

$$
\hat{y} = \arg\max_i P(y_i)
$$

**优点：** 确定性强，总是选择最可能的
**缺点：** 缺乏多样性，容易陷入重复循环

```python
def greedy_sample(probs):
    return torch.argmax(probs).item()

# 示例
probs = torch.tensor([0.1, 0.4, 0.3, 0.2])
chosen = greedy_sample(probs)
print(f"贪婪选择: 索引 {chosen}")  # 索引 1
```

### 6.4 随机采样（Random Sampling）

**策略：** 根据概率分布随机选择

```python
def random_sample(probs):
    return torch.multinomial(probs, 1).item()

# 示例
probs = torch.tensor([0.1, 0.4, 0.3, 0.2])
chosen = random_sample(probs)  # 按概率随机选择
```

### 6.5 Top-k采样

**策略：** 只从前k个最高概率的词中采样

```python
def top_k_sample(probs, k=5):
    # 获取top-k的索引和概率
    top_k_probs, top_k_indices = torch.topk(probs, k)
    
    # 从top-k中按概率采样
    chosen_idx = torch.multinomial(top_k_probs, 1).item()
    
    # 返回原始索引
    return top_k_indices[chosen_idx].item()

# 示例
probs = torch.tensor([0.15, 0.35, 0.25, 0.15, 0.05, 0.03, 0.02])
chosen = top_k_sample(probs, k=3)
print(f"Top-3采样: 索引 {chosen}")
```

### 6.6 Top-p采样（Nucleus Sampling）

**策略：** 从累积概率超过p的最小词集合中采样

```python
def top_p_sample(probs, p=0.9):
    # 按概率排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的位置
    # (保留概率质量在top-p内的词)
    nucleus_size = torch.sum(cumsum_probs <= p).item() + 1
    
    # 从nucleus中采样
    nucleus_probs = sorted_probs[:nucleus_size]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()  # 重新归一化
    
    chosen_idx = torch.multinomial(nucleus_probs, 1).item()
    return sorted_indices[chosen_idx].item()

# 示例
probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
chosen = top_p_sample(probs, p=0.9)
print(f"Top-p采样: 索引 {chosen}")
```

### 6.7 Temperature采样

**策略：** 通过调整Softmax的"温度"控制随机性

$$
P_T(x_i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}
$$

**Temperature的效果：**
| Temperature | 效果 | 适合场景 |
|-------------|------|----------|
| T < 1 | 更集中（低熵） | 确定性、精确任务 |
| T = 1 | 标准Softmax | 平衡 |
| T > 1 | 更分散（高熵） | 创造性、多样性 |

```python
def temperature_sample(logits, temperature=1.0):
    # 应用temperature
    adjusted_logits = logits / temperature
    
    # Softmax + 采样
    probs = F.softmax(adjusted_logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# 示例
logits = torch.tensor([2.0, 1.0, 0.5, -1.0])

for temp in [0.5, 1.0, 2.0]:
    chosen = temperature_sample(logits, temperature=temp)
    print(f"Temperature={temp}: 选择索引 {chosen}")
```

### 6.8 Beam Search（束搜索）

**策略：** 维护k个最可能的候选序列

```python
def beam_search(log_probs, beam_width=3, max_length=10):
    """
    简化的Beam Search实现
    log_probs: 每个位置的词表概率对数
    """
    batch_size, vocab_size = log_probs.shape
    
    # 初始化：选择beam_width个初始词
    log_probs_start = log_probs[0]
    top_k_probs, top_k_indices = torch.topk(log_probs_start, beam_width)
    
    beams = [(top_k_indices, top_k_probs)]  # [(序列, 累积对数概率)]
    
    for step in range(1, max_length):
        all_candidates = []
        
        for seq_indices, seq_score in beams:
            # 获取当前词的下一个词概率
            next_probs = log_probs[step]
            
            # 扩展所有beam
            top_k_probs, top_k_indices = torch.topk(next_probs, beam_width)
            
            for prob, word_idx in zip(top_k_probs, top_k_indices):
                new_seq = seq_indices.tolist() + [word_idx.item()]
                new_score = seq_score + prob
                all_candidates.append((new_seq, new_score))
        
        # 选择top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
    
    # 返回最佳序列
    best_seq, _ = beams[0]
    return best_seq
```

---

## 7. 概率在大模型中的应用

### 7.1 语言模型概率建模

**语言模型** 学习下一个词的条件概率：

$$
P(w_t | w_1, w_2, \ldots, w_{t-1})
$$

**交叉熵损失：**
$$
\mathcal{L} = -\sum_{t} \log P(w_t | w_1, \ldots, w_{t-1})
$$

```python
import torch
import torch.nn.functional as F

# 假设模型输出
logits = torch.randn(1, 10000)  # batch=1, vocab_size=10000

# 目标词ID
target_id = 42

# 计算交叉熵损失
loss = F.cross_entropy(logits, torch.tensor([target_id]))
print(f"交叉熵损失: {loss.item():.4f}")

# 对应于 -log P(目标词)
prob = F.softmax(logits, dim=-1)[0, target_id]
print(f"目标词概率: {prob.item():.6f}")
print(f"-log P = {-torch.log(prob).item():.4f}")
```

### 7.2 困惑度（Perplexity）

**困惑度** 是评估语言模型的标准指标：

$$
\text{PP} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i | w_1, \ldots, w_{i-1})}
$$

**直观理解：** 模型预测的不确定性（越低越好）

```python
def calculate_perplexity(model, tokenizer, text):
    """计算文本的困惑度"""
    model.eval()
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids']
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    # 计算负对数似然
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='mean'
    )
    
    perplexity = torch.exp(loss).item()
    return perplexity

# 示例
text = "The quick brown fox jumps over the lazy dog"
ppl = calculate_perplexity(model, tokenizer, text)
print(f"困惑度: {ppl:.2f}")
```

### 7.3 Dropout的随机性

**Dropout** 在训练时随机丢弃神经元：

```python
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(p=0.1)  # 10%概率丢弃
        self.linear2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x, training=True):
        x = self.linear1(x)
        x = torch.relu(x)
        if training:
            x = self.dropout(x)  # 训练时随机丢弃
        x = self.linear2(x)
        return x
```

**Dropout的数学解释：**
训练时：$y = f(W_1 \cdot \text{mask} \cdot x)$
推断时：$y = f((1-p) \cdot W_1 \cdot x)$ 或使用多个mask的平均

### 7.4 变分自编码器（VAE）

**VAE** 使用概率建模进行生成：

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = torch.relu(self.encoder(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    # 重构损失
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

### 7.5 大模型输出概率控制

```python
def controlled_generation(
    model,
    input_ids,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50
):
    """带概率控制的文本生成"""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature
        
        # Top-p掩码
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除超过top_p的词
            sorted_mask = cumsum_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            
            logits[sorted_indices[sorted_mask]] = float('-inf')
        
        # Top-k掩码
        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k).values[-1]
            logits[logits < top_k_vals] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return generated
```

---

## 本章小结

概率论是理解AI不确定性和生成过程的核心数学工具。关键要点：

1. **概率基础** 提供了描述不确定性的数学语言
2. **概率分布** 是建模数据生成过程的工具
3. **贝叶斯定理** 提供了更新信念的框架
4. **估计理论** 教我们如何从数据中学习参数
5. **采样策略** 是大模型生成文本的核心机制
6. **困惑度** 是评估语言模型的标准指标

## 深度分析

现代 LLM 本质上是一个概率建模系统。预训练阶段的目标是最大化下一个 Token 的对数似然 $\sum \log P(w_t|w_{<t})$，推理时的文本生成过程完全依赖概率采样——Top-k、Top-p（Nucleus）和 Temperature 策略直接控制生成文本的多样性与确定性之间的平衡。理解这些采样策略背后的概率图模型和条件分布知识，是调试 LLM 生成质量（如重复、幻觉问题）的关键。

贝叶斯定理在大模型中有更深层的含义。贝叶斯神经网络将权重视为分布而非点估计，能够提供不确定性量化。在 LLM 微调中，参数高效微调方法（如 LoRA）的初始化策略、早停法和权重衰减正则化，都可以从贝叶斯先验的角度理解。此外，变分推断中的 ELBO 目标直接关联到 VAE 和扩散模型的训练，这些生成模型与 LLM 的交叉融合正成为多模态 AI 的重要方向。

## 核心概念检查

- [ ] 你能解释语言模型中的下一个 Token 预测如何用条件概率表示？
- [ ] 你能推导 Softmax 函数与 Temperature 参数 $T$ 对概率分布熵的影响？
- [ ] 你能比较 Top-k、Top-p 和 Temperature 采样的优缺点及适用场景？
- [ ] 你能解释困惑度（Perplexity）与交叉熵之间的数学关系？
- [ ] 你能说明 MLE 和 MAP 估计的区别以及先验概率的正则化效果？
- [ ] 你能描述 Dropout 作为贝叶斯近似的理论解释？
- [ ] 你能分析自回归生成中 Beam Search 的搜索策略及其多样性问题？
- [ ] 你能解释条件概率中"已知之前所有 Token"对 LLM 生成质量的影响？
- [ ] 你能说明分类分布（Categorical Distribution）与 Softmax 的关系？
- [ ] 你能描述变分推断中 KL 散度最小化的几何意义？

## 延伸阅读

- [第四章：信息论](./ch04-information-theory.md) - 交叉熵与 KL 散度的概率基础
- [第二章：微积分](./ch02-calculus.md) - 概率密度函数中的积分运算
- [第九章：Transformer架构](./ch09-transformer.md) - 自回归生成中的概率链
- [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md) - Dropout 的正则化原理

**最后更新**: 2026-06-12

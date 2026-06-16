# 第十七章：概率与统计

> 概率论为机器学习提供了描述不确定性的数学框架，统计推断则让我们从数据中学习规律。

---

## 目录

1. [概率基础](#1-概率基础)
2. [随机变量与分布](#2-随机变量与分布)
3. [条件概率与贝叶斯定理](#3-条件概率与贝叶斯定理)
4. [期望与方差](#4-期望与方差)
5. [最大似然估计](#5-最大似然估计)
6. [贝叶斯推断](#6-贝叶斯推断)
7. [概率图模型](#7-概率图模型)

---

## 1. 概率基础

### 1.1 概率公理

1. $0 \leq P(A) \leq 1$
2. $P(\Omega) = 1$
3. 互斥事件的并：$P(\bigcup_i A_i) = \sum_i P(A_i)$

### 1.2 基本规则

| 规则 | 公式 |
|------|------|
| 加法 | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| 乘法 | $P(A \cap B) = P(A)P(B|A) = P(B)P(A|B)$ |
| 补集 | $P(\bar{A}) = 1 - P(A)$ |

---

## 2. 随机变量与分布

### 2.1 离散分布

| 分布 | 概率质量函数 | 期望 | 方差 | 应用 |
|------|------------|------|------|------|
| 伯努利 | $P(X=1)=p$ | $p$ | $p(1-p)$ | 二分类 |
| 二项 | $P(X=k)= \binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $n$次试验 |
| 泊松 | $P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 事件计数 |

### 2.2 连续分布

| 分布 | 概率密度函数 | 期望 | 方差 | 应用 |
|------|------------|------|------|------|
| 均匀 | $f(x)=\frac{1}{b-a}$ | $(a+b)/2$ | $(b-a)^2/12$ | 无先验信息 |
| 正态 | $f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | 自然现象、误差 |
| 指数 | $f(x)=\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | 等待时间 |
| Beta | $f(x) \propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | - | 概率的先验 |

### 2.3 正态分布的重要性

**中心极限定理（CLT）：** 独立同分布随机变量的均值趋近正态分布。

$$\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu, \frac{\sigma^2}{n})$$

这是统计推断的基石——无论原始分布如何，样本均值的分布都是正态的。

```python
import numpy as np
import matplotlib.pyplot as plt

# 验证CLT：从均匀分布采样，均值分布趋近正态
n_samples = 10000
n_each = 30
means = [np.mean(np.random.uniform(0, 1, n_each)) for _ in range(n_samples)]
```

---

## 3. 条件概率与贝叶斯定理

### 3.1 条件概率

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### 3.2 贝叶斯定理

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

- $P(A)$：先验概率
- $P(B|A)$：似然
- $P(A|B)$：后验概率
- $P(B)$：证据

### 3.3 朴素贝叶斯分类器

假设特征条件独立：

$$P(y|x_1,\ldots,x_n) \propto P(y)\prod_{i=1}^n P(x_i|y)$$

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
y = np.array([0, 0, 1, 1])
clf = GaussianNB()
clf.fit(X, y)
print(clf.predict([[2.0, 3.0]]))
```

---

## 4. 期望与方差

### 4.1 期望

离散：$\mathbb{E}[X] = \sum_i x_i P(X=x_i)$

连续：$\mathbb{E}[X] = \int x f_X(x) dx$

**线性性质：** $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

### 4.2 方差

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### 4.3 协方差与相关系数

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

### 4.4 在机器学习中的应用

```python
# 特征标准化（使均值为0，方差为1）
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
```

---

## 5. 最大似然估计

### 5.1 定义

给定数据 $\mathcal{D} = \{x_1, \ldots, x_n\}$，MLE选择使数据出现概率最大的参数：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D}|\theta) = \arg\max_\theta \prod_{i=1}^n P(x_i|\theta)$$

### 5.2 MSE = MLE under Gaussian

均方误差损失等价于高斯噪声假设下的最大似然估计：

$$L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 \iff y \sim \mathcal{N}(\hat{y}, \sigma^2)$$

### 5.3 交叉熵 = MLE under Categorical

交叉熵损失等价于类别分布假设下的MLE。

### 5.4 MLE示例：高斯分布

```python
def mle_gaussian(X):
    mu = np.mean(X)
    sigma = np.std(X)
    return mu, sigma

X = np.random.randn(1000) * 2 + 5
mu_hat, sigma_hat = mle_gaussian(X)
print(f"MLE: μ={mu_hat:.3f}, σ={sigma_hat:.3f}")
```

---

## 6. 贝叶斯推断

### 6.1 贝叶斯 vs 频率学派

| 方面 | 频率学派 | 贝叶斯学派 |
|------|---------|------------|
| 参数 | 固定未知常数 | 随机变量 |
| 概率 | 长期频率 | 信念程度 |
| 先验 | 不使用 | 必须指定 |
| 结果 | 点估计 + 置信区间 | 后验分布 |

### 6.2 共轭先验

选择先验使得后验与先验同分布族：

| 似然 | 共轭先验 | 后验 |
|------|----------|------|
| 伯努利 | Beta | Beta |
| 泊松 | Gamma | Gamma |
| 正态（已知方差） | 正态 | 正态 |

### 6.3 贝叶斯线性回归

$$P(\mathbf{w}|\mathbf{X}, \mathbf{y}) \propto P(\mathbf{y}|\mathbf{X}, \mathbf{w})P(\mathbf{w})$$

```python
# 贝叶斯线性回归的闭式解
def bayesian_linear_regression(X, y, alpha=1.0, beta=1.0):
    """alpha: 先验精度, beta: 似然精度"""
    S_N = np.linalg.inv(alpha * np.eye(X.shape[1]) + beta * X.T @ X)
    mu_N = beta * S_N @ X.T @ y
    return mu_N, S_N  # 后验均值与协方差
```

### 6.4 贝叶斯在深度学习中

- **Dropout** 可被视为贝叶斯近似的实现
- **BNN（贝叶斯神经网络）** 对权重学习分布而非点估计
- **不确定性量化：** 偶然不确定性（Aleatoric）+ 认知不确定性（Epistemic）

---

## 7. 概率图模型

### 7.1 图模型基础

| 类型 | 边 | 含义 | 示例 |
|------|-----|------|------|
| 贝叶斯网络 | 有向 | $P(X|Parents)$ | 朴素贝叶斯、HMM |
| 马尔可夫网络 | 无向 | 势函数 $\phi(C)$ | CRF、受限玻尔兹曼机 |

### 7.2 隐马尔可夫模型（HMM）

隐藏状态 $z_t$，观测 $x_t$：

$$P(\mathbf{z}, \mathbf{x}) = P(z_1)\prod_{t=2}^T P(z_t|z_{t-1})\prod_{t=1}^T P(x_t|z_t)$$

**三个基本问题：**
1. **评估：** Forward算法计算 $P(\mathbf{x}|\theta)$
2. **解码：** Viterbi算法找最可能状态序列
3. **学习：** Baum-Welch算法估计参数

### 7.3 条件随机场（CRF）

$$P(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})}\exp\left(\sum_i\sum_k \lambda_k f_k(y_i, y_{i-1}, \mathbf{x})\right)\]

在序列标注（NER、POS Tagging）中广泛应用。

---

## 延伸阅读

- *Probability Theory: The Logic of Science* (Jaynes) — 贝叶斯学派经典
- *Pattern Recognition and Machine Learning* (Bishop) — ML中的概率方法
- *Machine Learning: A Probabilistic Perspective* (Murphy) — 全面概率视角
- SciPy 统计文档: `scipy.stats`

---

*最后更新：2026-06-15*

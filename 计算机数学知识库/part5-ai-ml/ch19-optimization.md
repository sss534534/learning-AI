# 第十九章：优化理论

> 优化理论是机器学习的引擎——几乎所有模型训练都归结为最小化（或最大化）某个目标函数。

---

## 目录

1. [优化基础](#1-优化基础)
2. [无约束优化](#2-无约束优化)
3. [梯度下降法](#3-梯度下降法)
4. [凸优化](#4-凸优化)
5. [对偶问题](#5-对偶问题)
6. [随机优化](#6-随机优化)
7. [约束优化](#7-约束优化)

---

## 1. 优化基础

### 1.1 优化问题形式

$$\min_{\mathbf{x} \in \mathcal{D}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \; h_j(\mathbf{x}) = 0$$

- $f$：目标函数
- $g_i$：不等式约束
- $h_j$：等式约束
- $\mathcal{D}$：可行域

### 1.2 局部与全局最优

- **局部最优：** 在邻域内函数值最小
- **全局最优：** 在整个可行域内函数值最小
- **凸优化**的优美性质：局部最优 = 全局最优

### 1.3 最优性条件

**一阶必要条件：** $\nabla f(\mathbf{x}^*) = 0$

**二阶必要条件：** $\nabla^2 f(\mathbf{x}^*) \succeq 0$（海森矩阵半正定）

---

## 2. 无约束优化

### 2.1 解析解

对于某些问题可直接求导得到闭式解：

**线性回归（最小二乘）：**
$$\min_\mathbf{w} \|\mathbf{y} - X\mathbf{w}\|_2^2 \Rightarrow \mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$$

### 2.2 牛顿法

$$\mathbf{x}_{t+1} = \mathbf{x}_t - [\nabla^2 f(\mathbf{x}_t)]^{-1}\nabla f(\mathbf{x}_t)$$

**优点：** 二阶收敛，迭代次数少
**缺点：** 每步需计算海森逆，昂贵（$O(n^3)$）

### 2.3 拟牛顿法（BFGS）

用一阶信息近似海森矩阵：

$$B_{t+1} = B_t + \frac{y_t y_t^T}{y_t^T s_t} - \frac{B_t s_t s_t^T B_t^T}{s_t^T B_t s_t}$$

其中 $s_t = \mathbf{x}_{t+1} - \mathbf{x}_t$, $y_t = \nabla f(\mathbf{x}_{t+1}) - \nabla f(\mathbf{x}_t)$

---

## 3. 梯度下降法

### 3.1 基本原理

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta_t \nabla f(\mathbf{x}_t)$$

```python
def gradient_descent(grad_fn, x0, lr=0.01, n_iters=100):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_iters):
        x -= lr * grad_fn(x)
        trajectory.append(x.copy())
    return x, np.array(trajectory)
```

### 3.2 学习率调度

| 调度策略 | 公式 | 特性 |
|---------|------|------|
| 固定 | $\eta_t = \eta$ | 简单但需手动调 |
| 步衰减 | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k\rfloor}$ | 常见，$\gamma$ 通常 0.1 |
| 余弦退火 | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1+\cos(\frac{t}{T}\pi))$ | 平滑下降 |

### 3.3 动量方法

SGD + 动量可理解为物理中的惯性：

$$v_{t+1} = \mu v_t + \eta \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - v_{t+1}$$

- 加速收敛
- 抑制振荡
- 逃离局部极小

---

## 4. 凸优化

### 4.1 凸集与凸函数

**凸集：** 集合内任意两点的连线仍在集合内。

**凸函数：** $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$

### 4.2 凸函数的判定

- 一阶条件：$f(y) \geq f(x) + \nabla f(x)^T(y - x)$
- 二阶条件：$\nabla^2 f(x) \succeq 0$（海森半正定）

### 4.3 常见凸函数

| 函数 | 凸性 | 应用 |
|------|------|------|
| $x^2$ | 凸 | L2正则化 |
| $\log(1+e^{-x})$ | 凸 | 逻辑回归损失 |
| $\|x\|$ | 凸 | L1/L2正则化 |
| $\max(0, 1-x)$ | 凸 | Hinge损失（SVM） |

### 4.4 Jensen不等式

对于凸函数 $f$：

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

这是EM算法、VAE等许多ML方法的理论基础。

---

## 5. 对偶问题

### 5.1 Lagrange乘子法

对于约束优化：
$$\min f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, h_j(x) = 0$$

**Lagrange函数：**
$$L(x, \lambda, \mu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \mu_j h_j(x)$$

### 5.2 对偶问题

原问题（Primal）：
$$p^* = \min_x \max_{\lambda \geq 0, \mu} L(x, \lambda, \mu)$$

对偶问题（Dual）：
$$d^* = \max_{\lambda \geq 0, \mu} \min_x L(x, \lambda, \mu)$$

**弱对偶：** $d^* \leq p^*$
**强对偶（Slater条件）：** $d^* = p^*$

### 5.3 SVM中的对偶

SVM原问题是凸二次规划，对偶形式只需要计算内积——引出了核技巧：

$$L(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

---

## 6. 随机优化

### 6.1 SGD

用随机采样的梯度估计代替真实梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \nabla L_i(\mathbf{w}_t)$$

其中 $L_i$ 是第 $i$ 个样本的损失。

### 6.2 优化器演进

```
SGD → Momentum → NAG → AdaGrad → RMSProp → Adam → AdamW → Lion → Sophia
```

| 优化器 | 特点 | 适用场景 |
|--------|------|---------|
| SGD | 基础，需精细调学习率 | 简单任务 |
| SGD+Momentum | 加速收敛，抑制振荡 | 大多数任务 |
| AdaGrad | 自适应学习率，稀疏特征好 | NLP，稀疏数据 |
| RMSProp | 解决AdaGrad学习率消失 | 非平稳目标 |
| **Adam** | **自适应 + Momentum** | **通用（推荐）** |
| AdamW | Adam + 解耦权重衰减 | Transformer训练 |
| Lion | 符号优化，节省显存 | 大模型训练 |
| Sophia | 二阶信息，比Adam快2x | 大模型预训练 |

### 6.3 Adam优化器

```python
def adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v
```

### 6.4 学习率与收敛

| 学习率 | 行为 | 后果 |
|--------|------|------|
| 过大 | 震荡/发散 | 不收敛 |
| 适中 | 快速下降 | 理想 |
| 过小 | 缓慢下降 | 收敛慢或陷入局部 |
| 余弦调度 | 先大后小平滑下降 | 实践效果好 |

---

## 7. 约束优化

### 7.1 投影梯度法

执行梯度步后，投影回可行域：

$$\mathbf{x}_{t+1} = \Pi_\mathcal{C}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))$$

### 7.2 罚函数法

将约束以罚项形式加入目标：

$$\min_x f(x) + \rho \sum_i [g_i(x)]_+^2$$

### 7.3 在机器学习中的应用

| 场景 | 约束类型 | 处理方法 |
|------|---------|---------|
| L1正则化（Lasso） | $\|w\|_1 \leq t$ | 近端梯度下降 |
| L2正则化（Ridge） | $\|w\|_2^2 \leq t$ | 权重衰减 |
| 梯度裁剪 | $\|g\|_2 \leq c$ | 投影梯度 |
| 联邦学习约束 | 本地更新距离限制 | 近端项（FedProx） |

```python
# L1正则化的近端梯度（软阈值）
def soft_threshold(w, lambda_):
    return np.sign(w) * np.maximum(np.abs(w) - lambda_, 0)

# L2正则化（权重衰减）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## 延伸阅读

- *Convex Optimization* (Boyd & Vandenberghe) — 凸优化标准教材（免费PDF）
- *Numerical Optimization* (Nocedal & Wright) — 数值优化经典
- *Optimization Methods for Large-Scale Machine Learning* (Bottou et al.) — 大规模ML优化综述
- PyTorch优化器文档: `torch.optim`

---

*最后更新：2026-06-15*

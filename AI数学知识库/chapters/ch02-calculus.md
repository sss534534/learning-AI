# 第二章：微积分

> 微积分是深度学习的数学支柱之一。反向传播算法——训练神经网络的核心机制——本质上是链式法则的工程实现。本章将深入讲解微积分的核心概念，特别是**偏导数**、**梯度**和**链式法则**，并详细阐述它们在神经网络训练中的应用。

## 目录

1. [导数基础](#1-导数基础)
2. [偏导数与梯度](#2-偏导数与梯度)
3. [链式法则](#3-链式法则)
4. [高阶导数](#4-高阶导数)
5. [积分基础](#5-积分基础)
6. [梯度下降法](#6-梯度下降法)
7. [反向传播算法详解](#7-反向传播算法详解)
8. [梯度相关问题与解决方案](#8-梯度相关问题与解决方案)

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [第一章：线性代数](./ch01-linear-algebra.md)
- **关联文件**: [第五章：数值优化](./ch05-optimization.md), [第六章：神经网络基础](./ch06-neural-networks.md)
- **最后更新**: 2026-06-12
---

## 1. 导数基础

### 1.1 导数的定义

**导数（Derivative）** 描述了函数的瞬时变化率。

几何定义：
$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = \lim_{h \to 0} \frac{\Delta y}{\Delta x}
$$

**物理意义：**
- 几何：函数曲线的切线斜率
- 物理：瞬时速度（位置对时间的导数）

### 1.2 导数的几何意义

```
        f(x)
         │
    f(x+h)│    ●
         │   /│
         │  / │
         │ /  │  Δy = f(x+h) - f(x)
         │/   │
    f(x) ├──●────────────────
         x   x+h
         
    导数 = 切线斜率 = Δy/Δx 当 Δx → 0
```

### 1.3 基本求导法则

| 函数类型 | 函数 f(x) | 导数 f'(x) |
|----------|-----------|------------|
| 常数 | $c$ | $0$ |
| 幂函数 | $x^n$ | $nx^{n-1}$ |
| 指数函数 | $e^x$ | $e^x$ |
| 指数函数 | $a^x$ | $a^x \ln a$ |
| 对数函数 | $\ln x$ | $\frac{1}{x}$ |
| 对数函数 | $\log_a x$ | $\frac{1}{x \ln a}$ |
| 正弦函数 | $\sin x$ | $\cos x$ |
| 余弦函数 | $\cos x$ | $-\sin x$ |
| 正切函数 | $\tan x$ | $\sec^2 x$ |

### 1.4 求导法则

#### 1.4.1 四则运算

**加法法则：**
$$
(f + g)' = f' + g'
$$

**乘法法则（乘积法则）：**
$$
(f \cdot g)' = f' \cdot g + f \cdot g'
$$

**除法法则（商法则）：**
$$
\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}
$$

#### 1.4.2 复合函数求导（链式法则）

$$
[f(g(x))]' = f'(g(x)) \cdot g'(x)
$$

这是深度学习中最重要的求导法则！

```python
import sympy as sp

x = sp.symbols('x')
f = sp.sin(sp.cos(x))

# 求导
df = sp.diff(f, x)
print(df)  # -sin(x)*cos(cos(x))
```

### 1.5 导数与函数性质

| 导数符号 | 函数性质 | 图像特征 |
|----------|----------|----------|
| $f'(x) > 0$ | 函数递增 | 曲线上升 |
| $f'(x) < 0$ | 函数递减 | 曲线下降 |
| $f'(x) = 0$ | 极值点（临界点） | 切线水平 |
| $f''(x) > 0$ | 向下凹（凹向上） | 碗状 |
| $f''(x) < 0$ | 向上凸（凹向下） | 倒碗状 |

---

## 2. 偏导数与梯度

### 2.1 偏导数的定义

**偏导数（Partial Derivative）** 是多变量函数对其中一个变量的导数，将其他变量视为常数。

对于函数 $f(x, y)$：

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}
$$

**几何意义：**
- $\frac{\partial f}{\partial x}$：在 y 方向固定时，f 沿 x 方向的变化率
- $\frac{\partial f}{\partial y}$：在 x 方向固定时，f 沿 y 方向的变化率

### 2.2 偏导数的计算示例

**示例：计算 $f(x, y) = x^2 + 3xy + y^2$ 的偏导数**

对 x 求偏导（y 视为常数）：
$$
\frac{\partial f}{\partial x} = 2x + 3y
$$

对 y 求偏导（x 视为常数）：
$$
\frac{\partial f}{\partial y} = 3x + 2y
$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

f = x**2 + 3*x*y + y**2
f.backward()

print(f"∂f/∂x = {x.grad}")  # tensor([13.])
print(f"∂f/∂y = {y.grad}")  # tensor([10.])
```

### 2.3 梯度的定义

**梯度（Gradient）** 是偏导数的向量，将所有偏导数组合成一个向量：

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

**梯度指向函数增长最快的方向。**

### 2.4 梯度的几何意义

```
                        ∇f (梯度方向)
                          ↑
                          │
                    ╱╱╱╱╱╱╱╱│
                  ╱╱╱╱╱╱╱╱  │
                ╱╱╱╱╱╱╱╱    │
              ╱╱╱╱╱╱╱╱      │  梯度向量
            ╱╱╱╱╱╱╱╱        │  ⬆️指向最陡上升方向
          ╱╱╱╱╱╱╱╱          │
        ╱╱╱╱╱╱╱╱            │
      ╱╱╱╱╱╱╱╱              │
    ────────────────────────→ 等高线（水平集）
    
    梯度 ∇f 垂直于等高线，指向函数值增加的方向
```

### 2.5 梯度与方向导数

**方向导数** 表示函数沿任意方向的变化率：

$$
D_{\vec{u}}f = \nabla f \cdot \vec{u} = \|\nabla f\| \cos\theta
$$

其中 $\theta$ 是梯度与方向向量的夹角。

**关键结论：**
- 当 $\theta = 0$（同方向），方向导数最大 = $\|\nabla f\|$
- 当 $\theta = \pi$（相反方向），方向导数最小 = $-\|\nabla f\|$
- 当 $\theta = \pi/2$（垂直方向），方向导数为 0

### 2.6 梯度为零的条件

当 $\nabla f = \vec{0}$ 时，函数达到：
- **极值点**（局部/全局最大或最小）
- **鞍点**（一个方向最大，另一个方向最小）

在深度学习中，鞍点是训练的主要挑战之一。

---

## 3. 链式法则

### 3.1 链式法则的定义

链式法则是复合函数求导的通用规则，是反向传播算法的数学基础。

**一元函数链式法则：**
$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

如果 $y = f(g(x))$，设 $u = g(x)$，则：
$$
\frac{dy}{dx} = f'(u) \cdot u' = f'(g(x)) \cdot g'(x)
$$

### 3.2 多元函数链式法则

对于 $z = f(x, y)$，其中 $x = g(t)$，$y = h(t)$：

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial z}{\partial y} \cdot \frac{dy}{dt}
$$

### 3.3 链式法则的图示理解

```
    输入层          隐藏层          输出层
    
      x ──────→ u ──────→ v ──────→ y
      
      x ──────→ u ──────→ v ──────→ y
             ∂u/∂x    ∂v/∂u    ∂y/∂v
      
    dy/dx = (dy/dv) · (dv/du) · (du/dx)
```

### 3.4 链式法则在神经网络中的应用

考虑一个简单的三层神经网络：

$$
y = f_3(f_2(f_1(x)))
$$

令：
- $h_1 = f_1(x)$（第一层输出）
- $h_2 = f_2(h_1)$（第二层输出）
- $y = f_3(h_2)$（最终输出）

对 $x$ 求导：
$$
\frac{dy}{dx} = \frac{dy}{dh_2} \cdot \frac{dh_2}{dh_1} \cdot \frac{dh_1}{dx}
$$

### 3.5 链式法则的矩阵形式（批量计算）

在神经网络中，我们需要高效地计算雅可比矩阵（Jacobian）：

$$
\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Y}\right) \cdot \frac{\partial Y}{\partial X}
$$

**雅可比矩阵：** 对于函数 $\vec{y} = f(\vec{x})$：
$$
J = \frac{\partial \vec{y}}{\partial \vec{x}} = \begin{bmatrix} 
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots \\ 
\vdots & \vdots & \ddots 
\end{bmatrix}
$$

---

## 4. 高阶导数

### 4.1 二阶导数

**二阶导数** 是导数的导数：

$$
f''(x) = \frac{d}{dx}\left(\frac{df}{dx}\right)
$$

**物理意义：**
- 位置 s(t) → 一阶导数 v(t) = s'(t)（速度）
- 速度 v(t) → 二阶导数 a(t) = v'(t) = s''(t)（加速度）

### 4.2 二阶偏导数

对于多变量函数，有多个二阶偏导数：

$$
\frac{\partial^2 f}{\partial x^2}, \quad \frac{\partial^2 f}{\partial y^2}, \quad 
\frac{\partial^2 f}{\partial x \partial y}, \quad \frac{\partial^2 f}{\partial y \partial x}
$$

**混合偏导数**（如果函数光滑）：
$$
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}
$$

### 4.3 Hessian矩阵

**Hessian矩阵** 是二阶偏导数的矩阵形式：

$$
H = \begin{bmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ 
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ 
\vdots & \vdots & \ddots 
\end{bmatrix}
$$

**Hessian在优化中的作用：**
- Hessian 描述了函数的局部曲率
- 用于二阶优化方法（如牛顿法）
- $H$ 正定 → 局部极小值
- $H$ 负定 → 局部极大值
- $H$ 不定 → 鞍点

### 4.4 泰勒展开

**泰勒展开** 用多项式近似任意光滑函数：

$$
f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
$$

**二阶泰勒展开：**
$$
f(x) \approx f(a) + \nabla f(a)^T(x-a) + \frac{1}{2}(x-a)^T H(a)(x-a)
$$

在优化中，这用于：
- 分析优化算法的收敛性
- 理解梯度下降在曲面上的行为

---

## 5. 积分基础

### 5.1 积分的定义

**不定积分（反导数）：**
$$
\int f(x) \, dx = F(x) + C \quad \text{其中} \quad F'(x) = f(x)
$$

**定积分（面积）：**
$$
\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x
$$

定积分表示曲线 $y = f(x)$ 在区间 $[a, b]$ 与 x 轴围成的有向面积。

### 5.2 基本积分公式

| 函数 | 积分 |
|------|------|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$（$n \neq -1$） |
| $\frac{1}{x}$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |

### 5.3 积分在概率中的应用

概率密度函数 $p(x)$ 的积分等于 1：
$$
\int_{-\infty}^{+\infty} p(x) \, dx = 1
$$

**期望值的计算：**
$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot p(x) \, dx
$$

### 5.4 数值积分

在实际计算中，常用数值方法近似积分：

```python
import scipy.integrate as integrate

# 数值积分
result, error = integrate.quad(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx = {result}")  # ≈ 0.333

# 蒙特卡洛积分
import numpy as np

def monte_carlo_integral(f, a, b, n=10000):
    x = np.random.uniform(a, b, n)
    y = f(x)
    return (b - a) * np.mean(y)

result = monte_carlo_integral(lambda x: x**2, 0, 1)
print(f"蒙特卡洛: {result}")
```

---

## 6. 梯度下降法

### 6.1 优化问题的定义

机器学习的核心优化问题：
$$
\min_{\theta} \mathcal{L}(\theta)
$$

其中：
- $\theta$ 是模型参数
- $\mathcal{L}$ 是损失函数

### 6.2 梯度下降法原理

**核心思想：** 沿着梯度的反方向移动，因为梯度指向函数增长最快的方向，所以负梯度方向是函数下降最快的方向。

**更新规则：**
$$
\theta_{t+1} = \theta_t - \alpha \cdot \nabla \mathcal{L}(\theta_t)
$$

其中 $\alpha$ 是学习率（步长）。

### 6.3 梯度下降的几何直观

```
                    损失函数曲面
                         ⬆️
                        /│\
                       / │ \
                      /  │  \
                     /   │   \
                    /    │    \
                   /     │     \
                  ───────┼───────→ 参数空间
                        │
                        ↓
                        
    梯度下降：沿最陡下降方向迭代移动到局部最小值
```

### 6.4 学习率的影响

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(grad_func, x0, alpha, iterations):
    """梯度下降算法"""
    x = [x0]
    for _ in range(iterations):
        x.append(x[-1] - alpha * grad_func(x[-1]))
    return np.array(x)

# 示例：最小化 f(x) = x^2
grad_f = lambda x: 2 * x  # f'(x) = 2x

x0 = 5.0
alphas = [0.1, 0.4, 0.6, 0.9]

for alpha in alphas:
    trajectory = gradient_descent(grad_f, x0, alpha, 20)
    print(f"学习率 α={alpha}: 最终位置 x={trajectory[-1]:.6f}")
```

**学习率太小的后果：** 收敛速度极慢
**学习率太大的后果：** 可能越过最优解，甚至发散

### 6.5 梯度下降的类型

| 类型 | 更新公式 | 特点 |
|------|----------|------|
| **批量梯度下降** (BGD) | $\theta = \theta - \alpha \nabla \mathcal{L}_{全部}$ | 稳定但慢 |
| **随机梯度下降** (SGD) | $\theta = \theta - \alpha \nabla \mathcal{L}_{单个样本}$ | 快但不稳定 |
| **小批量梯度下降** (Mini-batch GD) | $\theta = \theta - \alpha \nabla \mathcal{L}_{batch}$ | 平衡速度和稳定性 |

```python
# 小批量梯度下降示例
batch_size = 32
for batch in dataloader:
    # 计算这个批量的梯度
    loss = model(batch)
    loss.backward()
    
    # 更新参数
    optimizer.step()
    optimizer.zero_grad()
```

### 6.6 收敛性分析

**梯度下降收敛的条件：**
- 损失函数是凸函数（或满足某些条件）
- 学习率足够小但不为零

**收敛速度：**
- 批量GD：$O(1/t)$（t是迭代次数）
- 使用动量/Adam等方法可以加速收敛

---

## 7. 反向传播算法详解

### 7.1 反向传播的本质

**反向传播（Backpropagation）** = 链式法则 + 计算图

核心思想：
1. **前向传播**：计算每个节点的输出
2. **反向传播**：从后向前计算每个参数的梯度

### 7.2 计算图与自动微分

**计算图** 记录了所有运算的依赖关系：

```
    x ──→ [乘法] ──→ u ──→ [ReLU] ──→ v ──→ [乘法] ──→ y
           ↑                          ↑              ↑
           w                          b              y_target
```

```python
import torch

# PyTorch自动微分
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 前向传播
u = w * x
v = u + b
y = v ** 2

# 反向传播
y.backward()

print(f"dy/dw = {w.grad}")  # = dy/dy * dy/dv * dv/du * du/dw
print(f"dy/dx = {x.grad}")  # = dy/dy * dy/dv * dv/du * du/dx
```

### 7.3 链式法则的完整推导

以简单的神经网络为例：

$$
\mathcal{L} = (y - \hat{y})^2
$$

其中 $\hat{y} = \sigma(w_2 \cdot \sigma(w_1 \cdot x + b_1) + b_2)$

**反向传播计算梯度：**

**第一步：对输出层参数求导**
$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)
$$

$$
\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2}
$$

**第二步：对隐藏层参数求导**
$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial v_2} \cdot \frac{\partial v_2}{\partial h} \cdot \frac{\partial h}{\partial w_1}
$$

### 7.4 反向传播的矩阵形式

对于全连接层 $Y = XW + b$：

**前向传播：**
```python
Z = X @ W + b        # 线性变换
Y = activation(Z)    # 激活函数
```

**反向传播：**
```python
dY = loss.backward()           # 输出梯度
dZ = dY * activation_grad(Z)   # 激活函数梯度
dW = X.T @ dZ                   # 权重梯度
dX = dZ @ W.T                   # 输入梯度（传给上一层）
db = dZ.sum(axis=0)            # 偏置梯度
```

### 7.5 具体计算示例

**示例：两层全连接网络**

```
输入 x (3,) → 线性层 W1 (3×4) → h (4,) → ReLU → a (4,) 
    → 线性层 W2 (4×2) → z (2,) → Softmax → y_pred (2,)
```

```python
import torch
import torch.nn.functional as F

# 假设
x = torch.randn(1, 3)           # 输入
y_target = torch.tensor([1])    # 目标类别
W1 = torch.randn(3, 4, requires_grad=True)
W2 = torch.randn(4, 2, requires_grad=True)
b1 = torch.randn(1, 4, requires_grad=True)
b2 = torch.randn(1, 2, requires_grad=True)

# 前向传播
h = x @ W1 + b1
a = F.relu(h)
z = a @ W2 + b2
y_pred = F.softmax(z, dim=-1)

# 计算交叉熵损失
loss = F.cross_entropy(z, y_target)

# 反向传播
loss.backward()

print(f"W1梯度形状: {W1.grad.shape}")  # [3, 4]
print(f"W2梯度形状: {W2.grad.shape}")  # [4, 2]
```

### 7.6 PyTorch自动微分机制

```python
import torch

# 创建一个需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 构建计算图
z = x ** 2 + 2 * x * y + y ** 2  # z = (x + y)^2

# 反向传播
z.backward()

print(f"dz/dx = {x.grad}")  # 2*x + 2*y = 2*2 + 2*3 = 10
print(f"dz/dy = {y.grad}")  # 2*x + 2*y = 10
```

### 7.7 反向传播算法流程总结

```
┌─────────────────────────────────────────────────────────────┐
│                     反向传播算法                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 前向传播                                                 │
│     输入 → Layer1 → Layer2 → ... → LayerL → 输出            │
│                                                             │
│  2. 计算损失                                                │
│     L = Loss(y_pred, y_true)                                │
│                                                             │
│  3. 反向传播梯度                                            │
│     ∂L/∂y_pred → ∂L/∂LayerL → ∂L/∂Layer(L-1) → ... → ∂L/∂Layer1│
│                                                             │
│  4. 更新参数                                                │
│     θ = θ - α * ∂L/∂θ                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 梯度相关问题与解决方案

### 8.1 梯度消失问题

**问题描述：**
当网络层数很深时，梯度在反向传播过程中逐层减小，最终变得几乎为0。

**原因分析：**
- 梯度连乘：$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial y} \cdot \prod_{i=2}^{L} \frac{\partial y_i}{\partial y_{i-1}}$
- 如果每层的雅可比矩阵的奇异值都小于1，梯度会指数级衰减

**解决方案：**

| 方法 | 原理 | 应用 |
|------|------|------|
| ReLU激活函数 | 导数是0或1 | 避免梯度消失 |
| 残差连接 | 梯度可以直接回传 | ResNet |
| LSTM/GRU | 门控机制保留梯度 | 序列模型 |
| 归一化层 | 稳定梯度范围 | BatchNorm |
| 预训练 + 微调 | 提供良好的初始梯度 | 迁移学习 |

```python
# 梯度消失示例
import torch

x = torch.tensor([0.1], requires_grad=True)
for i in range(10):
    w = torch.tensor([0.1], requires_grad=True)
    x = x * w  # 连续乘法
x.backward()
print(f"最终梯度: {x.grad}")  # ≈ 0.1^10 ≈ 0 (梯度消失!)
```

### 8.2 梯度爆炸问题

**问题描述：**
梯度在反向传播过程中逐层增大，最终变得非常大。

**原因分析：**
- 梯度连乘中，如果雅可比矩阵的奇异值大于1，梯度会指数级增长

**解决方案：**

| 方法 | 原理 | 应用 |
|------|------|------|
| **梯度裁剪** | 将梯度限制在某个范围内 | 训练稳定 |
| 权重正则化 | 防止权重过大 | L2正则化 |
| 合理的权重初始化 | 保持梯度方差稳定 | Xavier/Kaiming |
| Batch Normalization | 归一化激活值 | 训练稳定 |

```python
# 梯度裁剪实现
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 8.3 权重初始化

**Xavier初始化：**
$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)
$$

**Kaiming初始化（He初始化）：**
$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$

```python
# PyTorch中的初始化
import torch.nn.init as init

# Xavier初始化
init.xavier_uniform_(layer.weight)

# Kaiming初始化（适合ReLU）
init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 8.4 梯度检验

数值梯度检验（Numerical Gradient Check）：
$$
\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}
$$

```python
def gradient_check(model, x, y, eps=1e-7):
    """梯度检验"""
    model.eval()
    
    # 计算解析梯度
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    
    analytical_grad = model.weight.grad.data.clone()
    
    # 计算数值梯度
    model.zero_grad()
    numerical_grad = []
    for i in range(model.weight.numel()):
        old_val = model.weight.data.view(-1)[i].item()
        
        model.weight.data.view(-1)[i] = old_val + eps
        pred_plus = model(x)
        loss_plus = criterion(pred_plus, y).item()
        
        model.weight.data.view(-1)[i] = old_val - eps
        pred_minus = model(x)
        loss_minus = criterion(pred_minus, y).item()
        
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        numerical_grad.append(num_grad)
        
        model.weight.data.view(-1)[i] = old_val
    
    # 比较
    analytical = analytical_grad.view(-1)
    numerical = torch.tensor(numerical_grad)
    
    relative_error = (analytical - numerical).norm() / (analytical.norm() + numerical.norm())
    return relative_error.item()
```

---

## 本章小结

微积分是理解深度学习训练机制的数学基础。关键要点：

1. **偏导数** 让我们能够处理多变量函数的优化问题
2. **梯度** 指向函数增长最快的方向，是优化的核心工具
3. **链式法则** 是反向传播算法的数学基础
4. **梯度下降** 是训练神经网络的基本方法
5. **反向传播** 通过链式法则高效计算梯度
6. **梯度消失/爆炸** 是深度网络训练的主要挑战

## 深度分析

微积分是深度学习训练的引擎——反向传播（Backpropagation）本质上就是链式法则在计算图上的工程实现。当 LLM 在数百亿参数上进行梯度更新时，PyTorch 的 autograd 系统正是通过自动应用链式法则来逐层计算梯度的。理解偏导数与梯度的关系，有助于工程师把握学习率调谐、梯度裁剪和混合精度训练中的数值稳定性问题。

梯度消失和梯度爆炸是大模型训练中最棘手的挑战之一。深层 Transformer 中，梯度需要经过数十甚至上百层回传，每一层的雅可比矩阵的奇异值决定了梯度的缩放因子。这直接解释了为什么 Pre-LayerNorm 架构比 Post-LayerNorm 更稳定（梯度能更顺畅地流动），以及为什么残差连接对于训练深度网络必不可少。Hessian 矩阵刻画了损失曲面的局部曲率，二阶优化方法（如 K-FAC）虽然在大模型中因计算成本未能普及，但其思想影响了 Adam 等自适应优化器的设计。

## 核心概念检查

- [ ] 你能解释反向传播算法中链式法则的具体应用步骤？
- [ ] 你能手动推导一个两层 MLP 的梯度表达式？
- [ ] 你能说明偏导数、梯度和方向导数之间的关系？
- [ ] 你能解释梯度消失的数学原因（雅可比矩阵的奇异值）？
- [ ] 你能分析为什么 ReLU 比 Sigmoid 更能缓解梯度消失问题？
- [ ] 你能说明学习率过大或过小时梯度下降的行为差异？
- [ ] 你能用泰勒展开解释梯度下降法的一阶近似本质？
- [ ] 你能描述自动微分（Autograd）中前向模式和反向模式的区别？
- [ ] 你能解释为什么残差连接 $y=x+F(x)$ 的梯度包含恒等映射项？
- [ ] 你能说明梯度裁剪 clip_grad_norm_ 的数学原理及其对训练稳定性的影响？

## 延伸阅读

- [第一章：线性代数](./ch01-linear-algebra.md) - 向量微积分与雅可比矩阵
- [第五章：数值优化](./ch05-optimization.md) - 梯度下降与自适应优化器
- [第六章：神经网络基础](./ch06-neural-networks.md) - 激活函数与梯度传播
- [第三章：概率论与统计学](./ch03-probability.md) - 梯度估计的随机性

**最后更新**: 2026-06-12

# 第十六章：微积分

> 微积分是研究变化率和累积量的数学分支，在机器学习中既是反向传播的理论基础，也是优化算法的核心工具。

---

## 目录

1. [导数与微分](#1-导数与微分)
2. [偏导数与梯度](#2-偏导数与梯度)
3. [链式法则](#3-链式法则)
4. [反向传播算法](#4-反向传播算法)
5. [多元函数优化](#5-多元函数优化)
6. [雅克比矩阵与海森矩阵](#6-雅克比矩阵与海森矩阵)

---

## 1. 导数与微分

### 1.1 导数的定义

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**几何意义：** 函数在某点的切线斜率。

### 1.2 基本导数公式

| 函数 | 导数 |
|------|------|
| $c$（常数） | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |

### 1.3 求导法则

**乘法法则：** $(fg)' = f'g + fg'$

**除法法则：** $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$

**链式法则：** $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$

---

## 2. 偏导数与梯度

### 2.1 偏导数

固定其他变量，对目标变量求导：

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### 2.2 梯度

梯度是偏导数组成的向量：

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**性质：** 梯度方向是函数增长最快的方向。

### 2.3 梯度示例：线性回归损失

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n (y_i - (\mathbf{w} \cdot \mathbf{x}_i + b))^2$$

$$\frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{n}\sum_{i=1}^n (y_i - \hat{y}_i)\mathbf{x}_i$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^n (y_i - \hat{y}_i)$$

---

## 3. 链式法则

### 3.1 单变量链式法则

若 $y = f(u), u = g(x)$，则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

### 3.2 多变量链式法则

若 $z = f(x, y), x = g(t), y = h(t)$，则：

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

### 3.3 向量链式法则

若 $z = f(\mathbf{x}), \mathbf{x} = g(t)$：

$$\frac{dz}{dt} = \nabla f(\mathbf{x})^T \cdot \frac{d\mathbf{x}}{dt}$$

---

## 4. 反向传播算法

### 4.1 计算图

```
        x ──┐
            ├──→(+)──→ a ──→(σ)──→ ŷ ──→(L)──→ loss
        w ──┘          ↑
                       b
```

### 4.2 反向传播推导

以 $L = \frac{1}{2}(y - \hat{y})^2, \hat{y} = \sigma(wx + b)$ 为例：

```python
import numpy as np

# 前向传播
x, y, w, b = 1.0, 0.5, 0.3, 0.1
z = w * x + b        # 线性变换
y_pred = 1 / (1 + np.exp(-z))  # sigmoid
loss = 0.5 * (y - y_pred)**2   # MSE

# 反向传播
dL_dy_pred = -(y - y_pred)       # dL/dŷ
dy_pred_dz = y_pred * (1 - y_pred)  # dŷ/dz (sigmoid导数)
dz_dw = x                        # dz/dw
dz_db = 1                        # dz/db

# 链式法则
dL_dw = dL_dy_pred * dy_pred_dz * dz_dw  # 梯度 wrt w
dL_db = dL_dy_pred * dy_pred_dz * dz_db  # 梯度 wrt b

# 更新参数
w -= 0.1 * dL_dw
b -= 0.1 * dL_db
```

### 4.3 自动微分

现代框架（PyTorch/TensorFlow）自动计算梯度：

```python
import torch

x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(0.3, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

z = w * x + b
y_pred = torch.sigmoid(z)
loss = 0.5 * (0.5 - y_pred)**2

loss.backward()  # 自动反向传播
print(w.grad)    # 自动计算的梯度
```

---

## 5. 多元函数优化

### 5.1 梯度下降法

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

```python
def gradient_descent(f_grad, x0, lr=0.01, n_iter=100):
    x = x0.copy()
    for i in range(n_iter):
        grad = f_grad(x)
        x -= lr * grad
    return x
```

### 5.2 牛顿法

使用二阶信息加速收敛：

$$\mathbf{x}_{t+1} = \mathbf{x}_t - [Hf(\mathbf{x}_t)]^{-1}\nabla f(\mathbf{x}_t)$$

其中 $Hf$ 是海森矩阵。

### 5.3 梯度下降变体

| 变体 | 更新规则 | 特性 |
|------|---------|------|
| SGD | $\theta_{t+1} = \theta_t - \eta g_t$ | 基础，随机性可逃离鞍点 |
| Momentum | $v_{t+1} = \gamma v_t + \eta \nabla f(\theta_t)$ | 加速收敛，抑制振荡 |
| Adam | 自适应学习率 + Momentum | 最常用，适合大多数任务 |

---

## 6. 雅克比矩阵与海森矩阵

### 6.1 雅克比矩阵

向量值函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的雅克比矩阵：

$$J_f = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**在神经网络中：** 每层的雅克比矩阵连接了输入变化与输出变化。

### 6.2 海森矩阵

标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的海森矩阵：

$$H_f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$$

**特征值分析：**
- 所有特征值 > 0：局部最小值
- 所有特征值 < 0：局部最大值
- 混合符号：鞍点

### 6.3 在实际中的应用

```python
# 使用自动微分计算雅克比
import torch

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = f(x)

# 计算雅克比
J = torch.stack([torch.autograd.grad(y[i], x, retain_graph=True)[0] for i in range(2)])
print("雅克比矩阵:\n", J)
```

---

## 延伸阅读

- *Calculus* (Spivak) — 严格但清晰的微积分教材
- *The Calculus of Variations* (Gelfand & Fomin) — 变分法基础
- *Deep Learning* (Goodfellow et al.) 第4-6章 — 深度学习中的微积分
- 3Blue1Brown 微积分系列 — 直观可视化理解

---

*最后更新：2026-06-15*

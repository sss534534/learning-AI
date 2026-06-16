# 第十二章：数值计算

> 数值计算研究如何用计算机高效、精确地求解数学问题，是连接数学理论与工程实践的桥梁。

---

## 目录

1. [浮点数与精度问题](#1-浮点数与精度问题)
2. [数值稳定性](#2-数值稳定性)
3. [线性方程组求解](#3-线性方程组求解)
4. [数值积分与微分](#4-数值积分与微分)
5. [常微分方程数值解](#5-常微分方程数值解)
6. [插值与拟合](#6-插值与拟合)

---

## 1. 浮点数与精度问题

### 1.1 IEEE 754 浮点标准

```python
import numpy as np

# 单精度（float32）vs 双精度（float64）
print(np.finfo(np.float32).eps)  # 1.19e-7
print(np.finfo(np.float64).eps)  # 2.22e-16
```

**浮点数表示：** $(-1)^s \times M \times 2^E$

| 精度 | 符号位 | 指数位 | 尾数位 | 机器精度 |
|------|--------|--------|--------|----------|
| 单精度 | 1 | 8 | 23 | ~1.19e-7 |
| 双精度 | 1 | 11 | 52 | ~2.22e-16 |

### 1.2 常见精度陷阱

```python
# 大数吃小数
a = 1e16
b = 1.0
print(a + b == a)  # True — 1.0 被吃掉

# 相消相减
import math
x = 1e-8
print(math.cos(x) - 1.0)  # 灾难性抵消
print(-2 * math.sin(x/2)**2)  # 数值稳定形式
```

---

## 2. 数值稳定性

### 2.1 条件数

条件数衡量输出对输入误差的敏感度：

$$\kappa(f) = \|J_f\| \cdot \frac{\|x\|}{\|f(x)\|}$$

**解读：**
- $\kappa \approx 1$：良态问题
- $\kappa \gg 1$：病态问题

### 2.2 前向与后向误差

| 误差类型 | 定义 | 意义 |
|---------|------|------|
| 前向误差 | $\|\tilde{y} - y\|$ | 输出偏离精确解的程度 |
| 后向误差 | $\min\{\|\Delta x\|: f(x+\Delta x) = \tilde{y}\}$ | 输入需要修正多少 |

---

## 3. 线性方程组求解

### 3.1 直接法：LU 分解

$$A = LU, \quad Ax = b \Rightarrow Ly = b, Ux = y$$

```python
import numpy as np
from scipy.linalg import lu, solve

A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

# LU 分解
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)

# 求解
x = solve(A, b)
print("解:", x)
```

### 3.2 迭代法：共轭梯度法

适用于大规模稀疏矩阵，复杂度 $O(n \cdot \sqrt{\kappa})$：

```python
def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-10):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rsold = r @ r

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
```

### 3.3 矩阵分解对比

| 分解 | 适用场景 | 复杂度 | 特性 |
|-----|---------|--------|------|
| LU | 一般方阵 | $O(n^3)$ | 高斯消元矩阵形式 |
| Cholesky | 对称正定 | $O(\frac{1}{3}n^3)$ | 比LU快2倍 |
| QR | 最小二乘 | $O(mn^2)$ | 数值稳定 |
| SVD | 任意矩阵 | $O(mn^2)$ | 最稳定，最慢 |

---

## 4. 数值积分与微分

### 4.1 牛顿-柯特斯公式

**梯形法则：**
$$\int_a^b f(x)\,dx \approx \frac{h}{2}[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)], \quad h = \frac{b-a}{n}$$

**辛普森法则：**
$$\int_a^b f(x)\,dx \approx \frac{h}{3}[f(a) + 4\sum_{i=1}^{n/2}f(x_{2i-1}) + 2\sum_{i=1}^{n/2-1}f(x_{2i}) + f(b)]$$

```python
import numpy as np

def simpson(f, a, b, n=100):
    """辛普森数值积分"""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
```

### 4.2 数值微分

**中心差分：**
$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}, \quad \text{误差 } O(h^2)$$

---

## 5. 常微分方程数值解

### 5.1 欧拉法

$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$

### 5.2 RK4（经典龙格-库塔法）

```python
def rk4(f, t0, y0, h, n):
    """四阶龙格-库塔法"""
    t, y = t0, y0
    ts, ys = [t], [y]
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y += h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        ts.append(t)
        ys.append(y)
    return np.array(ts), np.array(ys)
```

---

## 6. 插值与拟合

### 6.1 拉格朗日插值

$$L_n(x) = \sum_{i=0}^n y_i \cdot l_i(x), \quad l_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

### 6.2 最小二乘拟合

$$\min_\beta \|y - X\beta\|_2^2 \Rightarrow \beta = (X^TX)^{-1}X^Ty$$

```python
import numpy as np

def least_squares(X, y):
    """最小二乘拟合"""
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

### 6.3 插值方法对比

| 方法 | 特性 | 适用场景 |
|-----|------|---------|
| 拉格朗日 | 高次可能振荡（Runge现象） | 低次插值 |
| 样条插值 | 分段低次，光滑连续 | 曲线拟合/图形学 |
| 最小二乘 | 噪声数据的最佳拟合 | 数据分析/回归 |

---

## 延伸阅读

- *Numerical Recipes* (Press et al.) — 数值计算圣经
- *Matrix Computations* (Golub & Van Loan) — 矩阵计算经典
- SciPy 文档: `scipy.linalg`, `scipy.integrate`, `scipy.interpolate`

---

*最后更新：2026-06-15*

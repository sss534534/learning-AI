# 第二章：复变函数

> 复变函数是量子力学的数学语言，复数在量子态的描述中扮演着核心角色。

---

## 目录

1. [复数基础](#1-复数基础)
2. [复向量空间](#2-复向量空间)
3. [复内积与酉变换](#3-复内积与酉变换)
4. [复矩阵的性质](#4-复矩阵的性质)
5. [在量子计算中的应用](#5-在量子计算中的应用)

---

## 1. 复数基础

### 1.1 复数的表示

$$z = a + bi, \quad i^2 = -1$$

**极坐标形式：** $z = re^{i\theta} = r(\cos\theta + i\sin\theta)$

**共轭：** $\bar{z} = a - bi$

**模：** $|z| = \sqrt{a^2 + b^2}$

### 1.2 复数的运算

| 运算 | 公式 |
|------|------|
| 加法 | $(a+bi) + (c+di) = (a+c) + (b+d)i$ |
| 乘法 | $(a+bi)(c+di) = (ac-bd) + (ad+bc)i$ |
| 除法 | $\frac{a+bi}{c+di} = \frac{(a+bi)(c-di)}{c^2+d^2}$ |
| 乘方 | $(re^{i\theta})^n = r^n e^{in\theta}$ |

### 1.3 欧拉公式

$$e^{i\theta} = \cos\theta + i\sin\theta$$

这是量子力学中最核心的公式之一，连接了指数函数和三角函数。

---

## 2. 复向量空间

### 2.1 复向量

量子态表示为复向量：

$$|\psi\rangle = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}, \quad \alpha, \beta \in \mathbb{C}$$

**归一化条件：** $|\alpha|^2 + |\beta|^2 = 1$

### 2.2 复内积

$$\langle \phi | \psi \rangle = \sum_i \bar{\phi}_i \psi_i$$

**性质：**
- 共轭对称：$\langle \phi|\psi \rangle = \overline{\langle \psi|\phi \rangle}$
- 正定性：$\langle \psi|\psi \rangle \geq 0$，等号仅当 $|\psi\rangle = 0$

---

## 3. 复内积与酉变换

### 3.1 酉矩阵

$U^\dagger U = UU^\dagger = I$，其中 $U^\dagger = (U^*)^T$ 是共轭转置。

**酉矩阵的性质：**
- 保内积：$\langle U\phi|U\psi \rangle = \langle \phi|\psi \rangle$
- 特征值的模为1：$|\lambda| = 1$
- 行列式的模为1：$|\det(U)| = 1$

### 3.2 常见量子酉门

| 量子门 | 矩阵 | 作用 |
|--------|------|------|
| Hadamard | $\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$ | 创建叠加态 |
| Pauli-X | $\begin{bmatrix}0&1\\1&0\end{bmatrix}$ | 量子NOT |
| Pauli-Z | $\begin{bmatrix}1&0\\0&-1\end{bmatrix}$ | 相位翻转 |
| $S$ | $\begin{bmatrix}1&0\\0&i\end{bmatrix}$ | 相位门 |
| $T$ | $\begin{bmatrix}1&0\\0&e^{i\pi/4}\end{bmatrix}$ | $\pi/4$相位门 |

---

## 4. 复矩阵的性质

### 4.1 Hermite矩阵

$H^\dagger = H$（共轭转置等于自身）

**性质：**
- 特征值为实数
- 特征向量正交
- 可被酉对角化：$H = U\Lambda U^\dagger$

### 4.2 谱分解

任何正规矩阵（$AA^\dagger = A^\dagger A$）可对角化：

$$A = \sum_i \lambda_i |v_i\rangle\langle v_i|$$

量子力学中可观测量都是 Hermite 矩阵，测量结果对应其特征值。

### 4.3 Kronecker积

描述多量子比特系统：

$$A \otimes B = \begin{bmatrix}
a_{11}B & a_{12}B & \cdots \\
a_{21}B & a_{22}B & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

```python
import numpy as np
# 两量子比特门的Kronecker积
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
H2 = np.kron(H, H)  # 两比特Hadamard
```

---

## 5. 在量子计算中的应用

- 量子比特的态矢量是复向量
- 量子门的矩阵表示是酉矩阵
- 测量算符是 Hermite 矩阵
- 量子态的相位 $\alpha = |\alpha|e^{i\phi}$ 用复数表示
- 量子干涉来源于复数幅度的叠加

---

## 延伸阅读

- *Complex Analysis* (Ahlfors) — 复变函数经典教材
- Nielsen & Chuang 第2章 — 量子计算的数学基础
- *Linear Algebra Done Right* (Axler) — 线性代数严格论述

---

*最后更新：2026-06-15*

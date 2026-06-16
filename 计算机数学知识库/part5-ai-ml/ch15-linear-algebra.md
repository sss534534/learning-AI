# 第十五章：线性代数

> 线性代数是AI和机器学习的核心数学语言，向量、矩阵、特征分解和张量运算构成了深度学习的基础。

---

## 目录

1. [向量基础](#1-向量基础)
2. [矩阵运算](#2-矩阵运算)
3. [线性变换](#3-线性变换)
4. [特征值与特征向量](#4-特征值与特征向量)
5. [矩阵分解](#5-矩阵分解)
6. [张量运算](#6-张量运算)
7. [在深度学习中的应用](#7-在深度学习中的应用)

---

## 1. 向量基础

### 1.1 向量定义与运算

向量 $\mathbf{v} \in \mathbb{R}^n$ 是有序数字集合。

**基本运算：**
- 加法：$\mathbf{v} + \mathbf{w} = (v_1 + w_1, \ldots, v_n + w_n)$
- 数乘：$c\mathbf{v} = (cv_1, \ldots, cv_n)$
- 点积：$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_i w_i = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta$
- 范数：$\|\mathbf{v}\|_p = (\sum |v_i|^p)^{1/p}$

### 1.2 常用范数

| 范数 | 公式 | 几何意义 |
|------|------|----------|
| L1 | $\|\mathbf{v}\|_1 = \sum\|v_i|$ | 曼哈顿距离 |
| L2 | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | 欧氏距离 |
| L∞ | $\|\mathbf{v}\|_\infty = \max\|v_i|$ | 切比雪夫距离 |

### 1.3 线性无关与基

向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 线性无关当且仅当：
$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

---

## 2. 矩阵运算

### 2.1 矩阵乘法

$$(AB)_{ij} = \sum_{k} A_{ik}B_{kj}$$

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # 等价于 np.matmul(A, B)
print(C.shape)  # (3, 5)
```

### 2.2 矩阵性质

| 性质 | 定义 | 条件 |
|------|------|------|
| 对称 | $A = A^T$ | $A_{ij} = A_{ji}$ |
| 正交 | $A^TA = AA^T = I$ | 列向量正交单位化 |
| 正定 | $\mathbf{x}^TA\mathbf{x} > 0$ | 所有特征值 > 0 |
| 半正定 | $\mathbf{x}^TA\mathbf{x} \geq 0$ | 所有特征值 ≥ 0 |

### 2.3 逆矩阵与伪逆

$$A^{-1}A = AA^{-1} = I \quad \text{（方阵可逆时）}$$

**伪逆（Moore-Penrose）：**
$$A^+ = \lim_{\alpha \to 0^+} (A^TA + \alpha I)^{-1}A^T$$

```python
# 求解线性系统 Ax = b
A = np.random.randn(5, 3)
b = np.random.randn(5)
x = np.linalg.lstsq(A, b, rcond=None)[0]  # 使用伪逆
```

---

## 3. 线性变换

### 3.1 变换的矩阵表示

线性变换 $T: \mathbb{R}^n \to \mathbb{R}^m$ 可以用矩阵 $A \in \mathbb{R}^{m \times n}$ 表示：
$$T(\mathbf{x}) = A\mathbf{x}$$

### 3.2 常见变换矩阵

| 变换 | 矩阵 | 效果 |
|------|------|------|
| 旋转（2D, $\theta$） | $\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ | 逆时针旋转 |
| 缩放 | $\begin{bmatrix}s_x & 0 \\ 0 & s_y\end{bmatrix}$ | 各轴缩放 |
| 剪切 | $\begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}$ | 水平剪切 |

### 3.3 核与像

- **核（Null Space）：** $\ker(T) = \{\mathbf{x} : T(\mathbf{x}) = \mathbf{0}\}$
- **像（Image）：** $\text{Im}(T) = \{T(\mathbf{x}) : \mathbf{x} \in \mathbb{R}^n\}$
- **维数定理：** $\dim(\ker(T)) + \dim(\text{Im}(T)) = n$

---

## 4. 特征值与特征向量

### 4.1 定义

$$A\mathbf{v} = \lambda\mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}$$

$\lambda$ 是特征值，$\mathbf{v}$ 是对应的特征向量。

### 4.2 特征分解

$$A = V\Lambda V^{-1}$$

其中 $V$ 的列是特征向量，$\Lambda$ 是对角矩阵（特征值）。

### 4.3 对称矩阵的谱定理

若 $A$ 是对称矩阵，则：
- 所有特征值为实数
- 特征向量相互正交
- $A = Q\Lambda Q^T$（$Q$ 是正交矩阵）

```python
A = np.random.randn(5, 5)
A = A @ A.T  # 构造对称矩阵
eigvals, eigvecs = np.linalg.eigh(A)  # 对称矩阵专用
print("特征值:", eigvals)
```

---

## 5. 矩阵分解

### 5.1 SVD（奇异值分解）

任何矩阵 $A \in \mathbb{R}^{m \times n}$ 可分解为：
$$A = U\Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$：左奇异向量（正交）
- $\Sigma \in \mathbb{R}^{m \times n}$：奇异值（对角）
- $V \in \mathbb{R}^{n \times n}$：右奇异向量（正交）

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
```

### 5.2 低秩近似（截断SVD）

保留前 $k$ 个最大奇异值：
$$A_k = U_k \Sigma_k V_k^T \approx A$$

```python
k = 2
U_k = U[:, :k]
Sigma_k = np.diag(s[:k])
Vt_k = Vt[:k, :]
A_k = U_k @ Sigma_k @ Vt_k
```

### 5.3 分解对比

| 分解 | 条件 | 复杂度 | 应用 |
|-----|------|--------|------|
| LU | 方阵 | $O(n^3)$ | 解线性方程组 |
| QR | 任意 | $O(mn^2)$ | 最小二乘 |
| 特征分解 | 可对角化 | $O(n^3)$ | 谱分析 |
| SVD | 任意 | $O(mn^2)$ | 降维、推荐系统、PCA |

---

## 6. 张量运算

### 6.1 张量定义

张量是多维数组的推广：
- 标量：0阶张量
- 向量：1阶张量
- 矩阵：2阶张量
- 3阶及以上：高阶张量

### 6.2 张量操作

```python
import torch

# 创建张量
t = torch.randn(2, 3, 4)  # 3阶张量

# 基本操作
t_sum = t.sum()           # 所有元素和
t_mean = t.mean(dim=0)    # 沿第一维求均值
t_reshape = t.view(2, 12) # 重塑

# 张量乘法（爱因斯坦求和）
result = torch.einsum('ijk,ikl->ijl', t, torch.randn(3, 4, 5))
```

---

## 7. 在深度学习中的应用

### 7.1 全连接层

$$h = \sigma(Wx + b)$$

$W$ 是权重矩阵，本质是一个线性变换，$\sigma$ 引入非线性。

### 7.2 注意力机制中的线性代数

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$：查询与键的相似度矩阵（矩阵乘法）
- $\text{softmax}$ 按行归一化
- 与 $V$ 相乘得到加权和

### 7.3 词嵌入与SVD

```python
# 使用SVD进行词嵌入降维
co_occurrence_matrix = ...  # 词共现矩阵
U, s, Vt = np.linalg.svd(co_occurrence_matrix, full_matrices=False)
word_embeddings = U[:, :50] @ np.diag(s[:50])  # 50维嵌入
```

### 7.4 PCA 与 SVD

PCA可以通过SVD高效实现：

```python
def pca(X, n_components):
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T
```

---

## 延伸阅读

- *Linear Algebra and Its Applications* (Strang) — 最经典的线性代数教材
- *Matrix Analysis* (Horn & Johnson) — 矩阵分析参考书
- NumPy 线性代数文档: `np.linalg`
- 3Blue1Brown 线性代数系列 — 直观可视化理解

---

*最后更新：2026-06-15*

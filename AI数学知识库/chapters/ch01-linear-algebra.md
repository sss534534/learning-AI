# 第一章：线性代数

> 线性代数是现代人工智能和深度学习的数学基石。从神经网络的矩阵运算到Transformer的注意力机制，从Embedding向量化表示到模型的参数存储，线性代数无处不在。本章将深入讲解线性代数的核心概念，并详细阐述其在大型语言模型中的应用。

## 目录

1. [向量与向量空间](#1-向量与向量空间)
2. [矩阵基础](#2-矩阵基础)
3. [矩阵运算详解](#3-矩阵运算详解)
4. [矩阵变换与逆矩阵](#4-矩阵变换与逆矩阵)
5. [特征值与特征向量](#5-特征值与特征向量)
6. [奇异值分解（SVD）](#6-奇异值分解svd)
7. [张量运算](#7-张量运算)
8. [线性代数在大模型中的应用](#8-线性代数在大模型中的应用)

---

## 1. 向量与向量空间

### 1.1 向量的定义

**向量（Vector）** 是线性代数中最基本的概念之一，它表示既有大小又有方向的量。

在数学上，我们用有序数组表示向量：

$$
\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

**向量的几个关键属性：**

- **维度（Dimension）**：向量中元素的数量，如 $\vec{v} \in \mathbb{R}^n$ 表示 n 维向量
- **范数（Norm）**：向量的"长度"，记作 $\|\vec{v}\|$
- **方向**：向量在空间中的指向

### 1.2 向量的范数

向量的范数是衡量向量"长度"的数学工具，最常用的是 $L_p$ 范数：

$$
\|\vec{v}\|_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}
$$

**常见的范数：**

| 范数类型 | 公式 | 说明 |
|----------|------|------|
| $L_1$ 范数（曼哈顿距离） | $\|\vec{v}\|_1 = \sum_i \|v_i\|$ | 各分量绝对值之和 |
| $L_2$ 范数（欧几里得距离） | $\|\vec{v}\|_2 = \sqrt{\sum_i v_i^2}$ | 最常用的向量长度 |
| $L_\infty$ 范数 | $\|\vec{v}\|_\infty = \max_i \|v_i\|$ | 最大分量的绝对值 |

**$L_2$ 范数的几何意义：**

```python
import numpy as np

v = np.array([3, 4])
l2_norm = np.linalg.norm(v)
print(f"向量 v = {v}")
print(f"L2范数 = {l2_norm}")  # 输出: 5.0
```

### 1.3 向量运算

#### 1.3.1 向量加法

$$
\vec{u} + \vec{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}
$$

#### 1.3.2 向量数乘

$$
c \cdot \vec{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}
$$

#### 1.3.3 向量点积（内积）

向量点积是深度学习中最重要的运算之一：

$$
\vec{u} \cdot \vec{v} = \sum_{i=1}^{n} u_i \cdot v_i = \|\vec{u}\| \cdot \|\vec{v}\| \cdot \cos\theta
$$

其中 $\theta$ 是两个向量之间的夹角。

**点积的物理意义：**
- 当 $\cos\theta = 1$（同方向），点积最大
- 当 $\cos\theta = 0$（垂直），点积为 0
- 当 $\cos\theta = -1$（相反方向），点积最小

```python
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot_product = np.dot(u, v)  # 或 u @ v
print(f"点积: {dot_product}")  # 输出: 32
```

#### 1.3.4 向量叉积（外积）

在三维空间中，向量叉积产生一个垂直于原来两个向量的新向量：

$$
\vec{u} \times \vec{v} = \begin{bmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{bmatrix}
$$

#### 1.3.5 向量外积（Outer Product）

向量外积产生一个矩阵：

$$
\vec{u} \otimes \vec{v} = \vec{u} \vec{v}^T = \begin{bmatrix} u_1 v_1 & u_1 v_2 & \cdots \\ u_2 v_1 & u_2 v_2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

**外积在注意力机制中的应用：**

在Transformer的注意力计算中，会大量使用向量外积来构建注意力矩阵。

### 1.4 向量空间

**向量空间（Vector Space）** 是满足特定运算规则的向量的集合。

**向量空间必须满足的性质：**

1. **加法封闭性**：$\vec{u}, \vec{v} \in V \Rightarrow \vec{u} + \vec{v} \in V$
2. **数乘封闭性**：$c \in \mathbb{R}, \vec{v} \in V \Rightarrow c\vec{v} \in V$
3. **存在零向量**：$\vec{0} \in V$
4. **存在加法逆元**：$\vec{v} \in V \Rightarrow -\vec{v} \in V$

### 1.5 线性相关与线性无关

**线性组合：**
给定一组标量 $c_1, c_2, \ldots, c_k$，向量 $\vec{v}$ 可以表示为：
$$
\vec{v} = c_1 \vec{v}_1 + c_2 \vec{v}_2 + \cdots + c_k \vec{v}_k
$$

**线性相关：** 存在不全为零的标量使得 $c_1 \vec{v}_1 + c_2 \vec{v}_2 + \cdots + c_k \vec{v}_k = \vec{0}$

**线性无关：** 只有当所有 $c_i = 0$ 时，上式才成立

**几何意义：** 线性无关的向量指向不同的方向，它们张成（span）一个更大的空间。

---

## 2. 矩阵基础

### 2.1 矩阵的定义

**矩阵（Matrix）** 是一个按照长方形阵列排列的复数或实数集合：

$$
A_{m \times n} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn} 
\end{bmatrix}
$$

**关键属性：**

| 属性 | 符号 | 说明 |
|------|------|------|
| 形状 | $m \times n$ | m 行 n 列 |
| 元素 | $a_{ij}$ | 第 i 行第 j 列的元素 |

### 2.2 特殊矩阵

| 矩阵类型 | 定义 | 示例 |
|----------|------|------|
| **方阵** | 行数 = 列数 ($n \times n$) | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| **行向量** | $1 \times n$ | $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ |
| **列向量** | $n \times 1$ | $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ |
| **零矩阵** | 所有元素为 0 | $\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$ |
| **单位矩阵** | 对角线为1，其余为0 | $I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ |
| **对角矩阵** | 非对角线元素为0 | $\text{diag}(d_1, d_2, \ldots, d_n)$ |
| **对称矩阵** | $A = A^T$ | $A_{ij} = A_{ji}$ |
| **反对称矩阵** | $A = -A^T$ | $A_{ij} = -A_{ji}$ |

### 2.3 单位矩阵的性质

单位矩阵 $I_n$ 是 $n \times n$ 的方阵，具有以下性质：

$$
I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}
$$

**核心性质：**
$$
A \cdot I = I \cdot A = A
$$

对于任意矩阵 A，乘以单位矩阵不改变其值。

---

## 3. 矩阵运算详解

### 3.1 矩阵加法

两个形状相同的矩阵可以相加：

$$
C_{ij} = A_{ij} + B_{ij}
$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B  # [[6, 8], [10, 12]]
```

### 3.2 矩阵数乘

矩阵与标量相乘：

$$
(cA)_{ij} = c \cdot a_{ij}
$$

### 3.3 矩阵乘法（核心运算）

**矩阵乘法的定义：**

设 $A_{m \times n}$ 和 $B_{n \times p}$，则 $C = A \cdot B$ 是 $m \times p$ 的矩阵：

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

**几何理解：** 矩阵乘法的本质是**线性变换的复合**。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # 矩阵乘法
# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
# [[19, 22], [43, 50]]
```

**矩阵乘法的性质：**

| 性质 | 公式 | 说明 |
|------|------|------|
| 结合律 | $(AB)C = A(BC)$ | 计算顺序无关 |
| 分配律 | $A(B+C) = AB + AC$ | 矩阵乘法分配于加法 |
| 单位元 | $AI = IA = A$ | 单位矩阵是乘法单位元 |
| **不满足交换律** | $AB \neq BA$ | 矩阵乘法通常不可交换 |

### 3.4 矩阵转置

矩阵转置交换行列：

$$
(A^T)_{ij} = A_{ji}
$$

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A.T  # [[1, 4], [2, 5], [3, 6]]
```

**转置的性质：**
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$
- $(A + B)^T = A^T + B^T$

### 3.5 矩阵求逆

**逆矩阵的定义：**

对于 $n \times n$ 的方阵 $A$，如果存在矩阵 $B$ 使得：
$$
AB = BA = I_n
$$
则称 $B$ 是 $A$ 的逆矩阵，记作 $A^{-1}$。

**可逆的条件：**
- $A$ 必须是方阵
- $\det(A) \neq 0$（行列式不为零）
- $A$ 的秩等于 n（满秩）

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # 接近单位矩阵 [[1, 0], [0, 1]]
```

### 3.6 哈达玛积（元素wise乘法）

对应元素相乘：
$$
(A \odot B)_{ij} = A_{ij} \cdot B_{ij}
$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A * B  # [[5, 12], [21, 32]]
```

### 3.7 克罗内克积

对于任意大小的矩阵：
$$
A_{m \times n} \otimes B_{p \times q} = \begin{bmatrix} a_{11}B & a_{12}B & \cdots \\ a_{21}B & a_{22}B & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}_{mp \times nq}
$$

克罗内克积在注意力机制的并行计算中有重要应用。

---

## 4. 矩阵变换与逆矩阵

### 4.1 行列式

**行列式的定义（2x2）：**
$$
\det(A) = \det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

**行列式的意义：**
- 几何上：行列式表示矩阵对应的线性变换对面积的缩放因子
- 如果 $\det(A) = 0$，则该变换将平面压缩到低维空间（不可逆）
- 如果 $\det(A) < 0$，则发生"翻转"（定向改变）

**行列式的性质：**
- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
print(f"行列式: {det}")  # -2.0
```

### 4.2 矩阵的秩

**秩（Rank）** 是矩阵中线性无关的行（或列）的最大数量。

**秩的性质：**
- $\text{rank}(A) \leq \min(m, n)$
- $\text{rank}(A) = \text{rank}(A^T)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

**秩的几何意义：**
- 秩 = 变换后的空间维度
- $\text{rank}(A) < n$ 意味着变换会压缩空间（信息损失）

```python
A = np.array([[1, 2], [2, 4]])  # 第二行是第一行的2倍
np.linalg.matrix_rank(A)  # 返回 1
```

### 4.3 线性方程组与矩阵

线性方程组可以写成矩阵形式：
$$
A\vec{x} = \vec{b}
$$

**解的存在性：**
- 如果 $\text{rank}(A) = \text{rank}([A|\vec{b}]) = n$，则唯一解存在
- 如果 $\text{rank}(A) = \text{rank}([A|\vec{b}]) < n$，则无穷多解
- 如果 $\text{rank}(A) < \text{rank}([A|\vec{b}])$，则无解

**求解方法：**
- 高斯消元法
- 矩阵求逆（如果 A 可逆）：$\vec{x} = A^{-1}\vec{b}$

---

## 5. 特征值与特征向量

### 5.1 定义

对于 $n \times n$ 的方阵 $A$，如果存在非零向量 $\vec{v}$ 和标量 $\lambda$ 使得：
$$
A\vec{v} = \lambda\vec{v}
$$

则：
- $\vec{v}$ 是矩阵 $A$ 的**特征向量（Eigenvector）**
- $\lambda$ 是矩阵 $A$ 的**特征值（Eigenvalue）**

**几何意义：**
特征向量在矩阵变换后仅发生伸缩（不改变方向），特征值表示伸缩的比例。

### 5.2 特征值与特征向量的计算

特征值满足**特征方程**：
$$
\det(A - \lambda I) = 0
$$

**计算步骤：**

1. 计算 $A - \lambda I$
2. 计算行列式 $\det(A - \lambda I)$
3. 解方程得到特征值 $\lambda$
4. 对每个 $\lambda$，解 $(A - \lambda I)\vec{v} = 0$ 得到特征向量

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
# [5. 2.]

print("特征向量:")
# eigenvectors[:,0] 对应特征值5的特征向量
# eigenvectors[:,1] 对应特征值2的特征向量
```

### 5.3 特征值的性质

| 性质 | 公式 | 说明 |
|------|------|------|
| 特征值之和 | $\sum \lambda_i = \text{tr}(A)$ | 等于矩阵的迹 |
| 特征值之积 | $\prod \lambda_i = \det(A)$ | 等于矩阵的行列式 |
| 相似矩阵 | $\det(A) = \det(B)$ | 相似矩阵有相同特征值 |

### 5.4 特征分解

**特征分解（Eigendecomposition）** 将矩阵分解为特征值和特征向量：

$$
A = Q \Lambda Q^{-1}
$$

其中：
- $Q$ 的列是特征向量
- $\Lambda$ 是对角矩阵，对角线是特征值

**特征分解的条件：**
- 矩阵必须可对角化
- 有 n 个线性无关的特征向量

### 5.5 特征值分解在降维中的应用（PCA）

**主成分分析（PCA）** 正是基于特征值分解：

1. 计算协方差矩阵的特征值和特征向量
2. 选择最大的 k 个特征值对应的特征向量
3. 将数据投影到这 k 个特征向量张成的空间中

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("主成分方差比例:", pca.explained_variance_ratio_)
```

---

## 6. 奇异值分解（SVD）

### 6.1 定义

**奇异值分解（Singular Value Decomposition）** 是最重要的高等线性代数概念之一。

对于任意 $m \times n$ 的矩阵 $A$，可以分解为：
$$
A = U \Sigma V^T
$$

其中：
- $U$：$m \times m$ 的正交矩阵（左奇异向量）
- $\Sigma$：$m \times n$ 的对角矩阵（奇异值）
- $V^T$：$n \times n$ 的正交矩阵（右奇异向量）

**关键点：**
- $U$ 和 $V$ 的列向量都是单位正交的
- $\Sigma$ 的对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 是奇异值

### 6.2 几何意义

SVD 将任意线性变换分解为三个简单的几何变换：

```
原始矩阵 A
     ↓
 [旋转/反射] U^T   ← 左奇异向量
     ↓
 [缩放] Σ        ← 奇异值（在各个方向缩放）
     ↓
 [旋转/反射] V    ← 右奇异向量
```

1. **旋转/反射**（$V^T$）：将坐标系旋转到新的方向
2. **缩放**（$\Sigma$）：在正交方向上按奇异值缩放
3. **旋转/反射**（$U$）：再次旋转到最终方向

### 6.3 SVD的计算

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, VT = np.linalg.svd(A)

print("U 形状:", U.shape)
print("奇异值 S:", S)
print("V^T 形状:", VT.shape)
```

### 6.4 截断SVD与降维

SVD 可以用于**数据压缩**和**降维**：

选择前 k 个最大的奇异值：

$$
A \approx U_k \Sigma_k V_k^T
$$

**降维误差：**
$$
\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2
$$

其中 $r$ 是矩阵的秩。

### 6.5 SVD与特征值的关系

对于 $A^TA$ 和 $AA^T$：
- $A^TA$ 的特征值是 $\sigma^2$
- $AA^T$ 的特征值也是 $\sigma^2$

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, VT = np.linalg.svd(A)

# 验证：AA^T = U Σ Σ^T U^T
# S**2 就是 AA^T 的特征值
print("S^2 (AA^T 的特征值):", S**2)
```

---

## 7. 张量运算

### 7.1 张量的定义

**张量（Tensor）** 是向量和矩阵的高维推广：

| 维度 | 名称 | 示例 |
|------|------|------|
| 0维 | 标量 | $5$ |
| 1维 | 向量 | $[1, 2, 3]$ |
| 2维 | 矩阵 | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| 3维+ | 张量 | 多维数组 |

在深度学习中：
- 一批图像：B×H×W×C（批次、高度、宽度、通道）
- 一批文本：B×L×D（批次、序列长度、嵌入维度）

### 7.2 张量基本运算

```python
import torch

# 创建张量
a = torch.randn(32, 64, 512)  # 批次32，序列长度64，嵌入维度512

# 张量乘法
b = torch.randn(32, 512, 768)
c = torch.bmm(a, b)  # 批次矩阵乘法，结果形状 [32, 64, 768]

# 张量转置
c_t = c.transpose(1, 2)  # 形状变为 [32, 768, 64]

# 张量重塑
d = c.reshape(32, -1)  # 展平为 [32, 49152]
```

### 7.3 注意力机制中的张量运算

在Transformer的Multi-Head Attention中：

```python
import torch
import torch.nn.functional as F

# 输入形状: [batch, seq_len, d_model]
X = torch.randn(32, 128, 512)

# 投影得到 Q, K, V
W_q = torch.randn(512, 512)
W_k = torch.randn(512, 512)
W_v = torch.randn(512, 512)

Q = X @ W_q  # [32, 128, 512]
K = X @ W_k  # [32, 128, 512]
V = X @ W_v  # [32, 128, 512]

# 计算注意力分数
scores = Q @ K.transpose(-2, -1)  # [32, 128, 128]
scores = scores / (512 ** 0.5)   # 缩放

# Softmax
attn_weights = F.softmax(scores, dim=-1)

# 加权求和
context = attn_weights @ V  # [32, 128, 512]
```

---

## 8. 线性代数在大模型中的应用

### 8.1 Embedding向量化表示

**Embedding** 是将离散符号映射到连续向量空间的技术。

```python
import torch
import torch.nn as nn

vocab_size = 50000
embedding_dim = 512

# 词嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 输入词ID
input_ids = torch.tensor([1234, 5678, 9012])

# 获取词向量
vectors = embedding(input_ids)
print("输出形状:", vectors.shape)  # [3, 512]

# 词的数学表示
# "hello" → [0.1, -0.2, 0.3, ..., 0.05]
# "world" → [0.8, 0.1, -0.4, ..., -0.2]
```

**数学原理：**
- 词嵌入矩阵 $E$ 的形状为 $V \times D$
- 每个词 $w_i$ 对应嵌入矩阵的一行 $E_i$
- 嵌入操作本质上是查表：$\text{embedding}(w_i) = E_i$

### 8.2 Transformer中的矩阵运算

#### 8.2.1 Self-Attention的矩阵形式

**核心公式：**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**完整矩阵推导：**

设输入序列 $X \in \mathbb{R}^{L \times D}$，则：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\text{Attention} = \text{softmax}\left(\frac{XW_QW_K^TX^T}{\sqrt{d_k}}\right)XW_V
$$

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, D = x.shape
        
        # 线性投影
        Q = self.W_q(x)  # [B, L, D]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分头
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ V
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(context)
```

#### 8.2.2 Feed-Forward Network (FFN)

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

通常 $d_{ff} = 4 \times d_{model}$，例如：
- BERT-base: $d_{model} = 768, d_{ff} = 3072$
- GPT-3: $d_{model} = 12288, d_{ff} = 49152$

### 8.3 位置编码（Positional Encoding）

#### 8.3.1 绝对位置编码（正弦/余弦）

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-math.log(10000.0) / d_model))
    
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    
    return PE
```

#### 8.3.2 旋转位置编码（RoPE）

RoPE 通过旋转操作编码位置信息：

$$
R_{\Theta,m}^d x = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_i^{(1)} \\ x_i^{(2)} \end{bmatrix}
$$

这是 LLaMA 等模型使用的位置编码方式。

### 8.4 Layer Normalization

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中：
- $\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$（均值）
- $\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2$（方差）
- $\gamma, \beta$ 是可学习的缩放和偏移参数

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### 8.5 模型参数存储

以 GPT-3 为例的参数规模分析：

| 组件 | 参数量 | 计算方式 |
|------|--------|----------|
| Embedding | $V \times D$ | 50000 × 12288 ≈ 6.1亿 |
| Attention W_Q, W_K, W_V | $3 \times D^2$ | 3 × 12288² ≈ 4.5亿 |
| Attention W_O | $D^2$ | 12288² ≈ 1.5亿 |
| FFN W_1, W_2 | $2 \times D \times d_{ff}$ | 2 × 12288 × 49152 ≈ 12亿 |
| LayerNorm | $2 \times D$ | 2 × 12288 × 96 ≈ 240万 |
| **总计** | 1750亿 | - |

**矩阵运算视角：**
- 每个 Transformer 层的核心计算都是矩阵乘法
- $QK^T$ 是 $D \times D$ 的矩阵乘法
- FFN 涉及 $D \times 4D$ 和 $4D \times D$ 的矩阵乘法

---

## 本章小结

线性代数是理解和实现深度学习模型的数学基础。关键要点：

1. **向量与矩阵运算** 是神经网络计算的基本单元
2. **矩阵乘法** 是信息传递和特征提取的核心操作
3. **特征值/特征向量** 揭示了矩阵变换的本质
4. **SVD** 提供了数据降维和压缩的数学工具
5. **张量运算** 使得处理多维数据成为可能
6. 在大模型中，这些运算被大量组合用于实现注意力机制、位置编码等核心功能

**下一章：** 我们将学习微积分，特别是**链式法则**和**梯度**在反向传播中的应用。

# 第一章：线性代数深度

> 线性代数是量子计算的数学基石。本章深入探讨量子计算所需的线性代数核心概念，包括向量空间、内积空间、线性算子、酉变换等。

---

## 1.1 向量空间

### 1.1.1 向量空间的定义

**定义（向量空间）**：域 $\mathbb{F}$ 上的向量空间 $V$ 是一个集合，其上定义了两种运算：
- 向量加法：$+: V \times V \rightarrow V$
- 标量乘法：$\cdot: \mathbb{F} \times V \rightarrow V$

满足以下8条公理：
1. 交换律：$u + v = v + u$
2. 结合律：$(u + v) + w = u + (v + w)$
3. 零向量存在：$\exists 0 \in V, u + 0 = u$
4. 加法逆元存在：$\forall u, \exists -u, u + (-u) = 0$
5. 分配律1：$a(u + v) = au + av$
6. 分配律2：$(a + b)u = au + bu$
7. 结合律：$a(bu) = (ab)u$
8. 单位元：$1u = u$

在量子计算中，我们主要关注复数域 $\mathbb{C}$ 上的有限维向量空间。

### 1.1.2 基与维度

**定义（线性无关）**：向量集合 $\{v_1, v_2, \dots, v_n\}$ 线性无关，当且仅当
$$a_1v_1 + a_2v_2 + \dots + a_nv_n = 0 \implies a_1 = a_2 = \dots = a_n = 0$$

**定义（基）**：向量空间 $V$ 的基是一个线性无关的向量集合，且能张成 $V$。

**定理**：向量空间的所有基具有相同的基数，称为空间的维度，记为 $\dim(V)$。

### 1.1.3 狄拉克（Dirac）记号

在量子力学和量子计算中，我们使用狄拉克记号：
- 右矢（ket）：$|v\rangle$ 表示向量
- 左矢（bra）：$\langle v|$ 表示对偶向量
- 内积：$\langle v|w\rangle$

对于 $n$ 维复向量空间 $\mathbb{C}^n$，标准基为：
$$|0\rangle = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad |1\rangle = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \dots, \quad |n-1\rangle = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

任意向量 $|v\rangle$ 可表示为：
$$|v\rangle = \sum_{i=0}^{n-1} v_i |i\rangle$$

---

## 1.2 内积空间

### 1.2.1 内积的定义

**定义（内积）**：复向量空间 $V$ 上的内积是映射 $\langle \cdot | \cdot \rangle: V \times V \rightarrow \mathbb{C}$，满足：
1. 共轭对称性：$\langle v|w\rangle = \langle w|v\rangle^*$
2. 线性性：$\langle v|aw_1 + bw_2\rangle = a\langle v|w_1\rangle + b\langle v|w_2\rangle$
3. 正定性：$\langle v|v\rangle \geq 0$，当且仅当 $v = 0$ 时等号成立

**定义（希尔伯特空间）**：完备的内积空间称为希尔伯特空间。有限维内积空间自动是完备的。

### 1.2.2 正交与归一

**定义（正交）**：若 $\langle v|w\rangle = 0$，则称 $|v\rangle$ 与 $|w\rangle$ 正交。

**定义（归一）**：若 $\langle v|v\rangle = 1$，则称 $|v\rangle$ 是归一的。

**定义（正交归一基）**：若基 $\{|e_i\rangle\}$ 满足 $\langle e_i|e_j\rangle = \delta_{ij}$，则称为正交归一基。

**Gram-Schmidt正交化过程**：将任意基 $\{v_1, \dots, v_n\}$ 转换为正交归一基：
$$|e_1\rangle = \frac{|v_1\rangle}{\sqrt{\langle v_1|v_1\rangle}}$$
$$|e_k\rangle = \frac{|v_k\rangle - \sum_{i=1}^{k-1} \langle e_i|v_k\rangle |e_i\rangle}{\sqrt{\langle \dots | \dots \rangle}}$$

---

## 1.3 线性算子与矩阵表示

### 1.3.1 线性算子

**定义（线性算子）**：映射 $A: V \rightarrow W$ 是线性的，若
$$A(av + bw) = aA(v) + bA(w)$$

**定义（算子的矩阵表示）**：给定 $V$ 的基 $\{|v_i\rangle\}$ 和 $W$ 的基 $\{|w_j\rangle\}$，算子 $A$ 的矩阵表示 $[A]$ 的元素为
$$[A]_{ji} = \langle w_j|A|v_i\rangle$$

### 1.3.2 重要的算子类型

**定义（Hermite算子/自伴算子）**：算子 $A$ 是Hermite的，若
$$A^\dagger = A$$
其中 $A^\dagger$ 是 $A$ 的厄米共轭，满足 $\langle v|A^\dagger|w\rangle = \langle w|A|v\rangle^*$。

**性质**：
- Hermite算子的特征值都是实数
- Hermite算子对应于量子力学中的可观测量

**定义（酉算子）**：算子 $U$ 是酉的，若
$$U^\dagger U = U U^\dagger = I$$

**性质**：
- 酉算子保持内积：$\langle Uv|Uw\rangle = \langle v|w\rangle$
- 酉算子对应于量子态的演化
- 酉算子的特征值都是模为1的复数

**定义（正规算子）**：算子 $N$ 是正规的，若
$$N^\dagger N = N N^\dagger$$

所有Hermite算子和酉算子都是正规算子。

---

## 1.4 特征值与特征向量

### 1.4.1 基本定义

**定义（特征值与特征向量）**：对于算子 $A$，若存在非零向量 $|v\rangle$ 和标量 $\lambda$ 使得
$$A|v\rangle = \lambda|v\rangle$$
则 $\lambda$ 是 $A$ 的特征值，$|v\rangle$ 是对应的特征向量。

**谱定理**：正规算子可以酉对角化，即存在酉算子 $U$ 使得
$$U^\dagger N U = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$$
其中 $\lambda_i$ 是 $N$ 的特征值。

### 1.4.2 迹与行列式

**定义（迹）**：算子 $A$ 的迹是其矩阵表示的对角线元素之和：
$$\text{tr}(A) = \sum_i \langle i|A|i\rangle$$

**性质**：
- $\text{tr}(AB) = \text{tr}(BA)$
- $\text{tr}(A) = \sum_i \lambda_i$（特征值之和）

**定义（行列式）**：算子 $A$ 的行列式是其矩阵表示的行列式，满足
$$\det(A) = \prod_i \lambda_i$$

---

## 1.5 张量代数与直积

### 1.5.1 张量积空间

**定义（张量积）**：给定两个向量空间 $V$ 和 $W$，它们的张量积 $V \otimes W$ 是由所有形式为 $|v\rangle \otimes |w\rangle$ 的元素张成的向量空间，满足：
1. $(a|v_1\rangle + b|v_2\rangle) \otimes |w\rangle = a|v_1\rangle \otimes |w\rangle + b|v_2\rangle \otimes |w\rangle$
2. $|v\rangle \otimes (a|w_1\rangle + b|w_2\rangle) = a|v\rangle \otimes |w_1\rangle + b|v\rangle \otimes |w_2\rangle$

若 $\{|v_i\rangle\}$ 是 $V$ 的基，$\{|w_j\rangle\}$ 是 $W$ 的基，则 $\{|v_i\rangle \otimes |w_j\rangle\}$ 是 $V \otimes W$ 的基。

### 1.5.2 算子的张量积

**定义（算子的张量积）**：给定算子 $A: V \rightarrow V$ 和 $B: W \rightarrow W$，它们的张量积 $A \otimes B$ 定义为
$$(A \otimes B)(|v\rangle \otimes |w\rangle) = A|v\rangle \otimes B|w\rangle$$

矩阵表示：若 $A$ 是 $m \times m$ 矩阵，$B$ 是 $n \times n$ 矩阵，则
$$A \otimes B = \begin{bmatrix} A_{11}B & \dots & A_{1m}B \\ \vdots & \ddots & \vdots \\ A_{m1}B & \dots & A_{mm}B \end{bmatrix}$$

### 1.5.3 多体量子系统

在量子计算中，$n$ 量子比特系统的状态空间是
$$(\mathbb{C}^2)^{\otimes n} = \mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \dots \otimes \mathbb{C}^2$$

计算基为
$$|x_1\rangle \otimes |x_2\rangle \otimes \dots \otimes |x_n\rangle \equiv |x_1x_2\dots x_n\rangle$$
其中 $x_i \in \{0, 1\}$。

---

## 1.6 投影算子与谱分解

### 1.6.1 投影算子

**定义（投影算子）**：算子 $P$ 是投影算子，若
$$P^2 = P \quad \text{且} \quad P^\dagger = P$$

**正交投影**：若 $P$ 是投影到子空间 $S$ 的投影，则 $I - P$ 是投影到正交补空间 $S^\perp$ 的投影。

### 1.6.2 谱分解

**谱分解定理**：正规算子 $N$ 可以表示为
$$N = \sum_i \lambda_i P_i$$
其中 $\lambda_i$ 是特征值，$P_i$ 是投影到特征空间的正交投影算子，满足
$$P_i P_j = \delta_{ij} P_i, \quad \sum_i P_i = I$$

对于Hermite算子 $H$，其谱分解对应于量子可观测量的测量结果。

---

## 1.7 矩阵分解

### 1.7.1 奇异值分解（SVD）

**定理（SVD）**：任意 $m \times n$ 复矩阵 $A$ 可以分解为
$$A = U \Sigma V^\dagger$$
其中：
- $U$ 是 $m \times m$ 酉矩阵
- $V$ 是 $n \times n$ 酉矩阵
- $\Sigma$ 是 $m \times n$ 对角矩阵，对角元素 $\sigma_i \geq 0$ 为奇异值

### 1.7.2 极分解

**定理（极分解）**：任意方阵 $A$ 可以分解为
$$A = U P$$
其中 $U$ 是酉矩阵，$P$ 是半正定Hermite矩阵。

---

## 1.8 量子计算中的重要算子

### 1.8.1 Pauli矩阵

Pauli矩阵是量子计算中最重要的算子：
$$\sigma_0 = I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$
$$\sigma_1 = X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
$$\sigma_2 = Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}$$
$$\sigma_3 = Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**性质**：
- $\sigma_i^\dagger = \sigma_i$（Hermite）
- $\sigma_i^2 = I$
- $\{\sigma_i, \sigma_j\} = 2\delta_{ij}I$（反对易关系）
- $[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k$（对易关系）

### 1.8.2 Hadamard矩阵

$$H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

**性质**：
- $H$ 是酉矩阵
- $H^2 = I$
- 变换基：$H|0\rangle = |+\rangle, \quad H|1\rangle = |-\rangle$

---

## 本章小结

本章深入介绍了量子计算所需的线性代数基础，包括：
- 向量空间与狄拉克记号
- 内积空间与希尔伯特空间
- 线性算子与矩阵表示
- 酉算子与Hermite算子
- 张量积与多体系统
- 谱分解与矩阵分解

这些概念是理解量子计算的核心数学工具。

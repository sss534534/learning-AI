# 第三章：群论与表示论

> 群论是描述对称性的数学工具，在量子力学中用于分类粒子和理解守恒律，在量子计算中用于量子纠错码的设计。

---

## 目录

1. [群的基本概念](#1-群的基本概念)
2. [李群与李代数](#2-李群与李代数)
3. [量子力学中的对称性](#3-量子力学中的对称性)
4. [表示论基础](#4-表示论基础)

---

## 1. 群的基本概念

### 1.1 群的定义

集合 $G$ 上的二元运算 $\cdot$ 满足：
- **封闭性**：$\forall a,b \in G, a\cdot b \in G$
- **结合律**：$(a\cdot b)\cdot c = a\cdot(b\cdot c)$
- **单位元**：$\exists e, \forall a, e\cdot a = a\cdot e = a$
- **逆元**：$\forall a, \exists a^{-1}, a\cdot a^{-1} = a^{-1}\cdot a = e$

### 1.2 常见群

| 群 | 元素 | 运算 |
|------|------|------|
| $\mathbb{Z}_n$ | $\{0,1,\ldots,n-1\}$ | 模n加法 |
| $S_n$ | n个元素的排列 | 置换复合 |
| $SU(2)$ | 2×2酉矩阵，行列式为1 | 矩阵乘法 |
| $U(1)$ | $e^{i\theta}$ | 复数乘法 |

### 1.3 Pauli群

量子纠错中的重要群：

$$P_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$$

$n=1$ 时的单量子比特Pauli群包含16个元素。

---

## 2. 李群与李代数

### 2.1 李群

连续对称性的数学描述。量子力学中的旋转对称由 $SU(2)$ 描述。

### 2.2 李代数

李群的切空间，由生成元及其对易关系定义：

$$[J_i, J_j] = i\epsilon_{ijk} J_k$$

**Pauli矩阵作为生成元：**
- $X = \begin{bmatrix}0&1\\1&0\end{bmatrix}$
- $Y = \begin{bmatrix}0&-i\\i&0\end{bmatrix}$
- $Z = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$

**对易关系：** $[X, Y] = 2iZ$，$[Y, Z] = 2iX$，$[Z, X] = 2iY$

---

## 3. 量子力学中的对称性

### 3.1 Wigner定理

对称性变换由酉算符或反酉算符表示。

### 3.2 守恒律

Noether定理：每个连续对称性对应一个守恒量。

| 对称性 | 守恒量 |
|--------|--------|
| 时间平移 | 能量 |
| 空间平移 | 动量 |
| 旋转 | 角动量 |
| 规范变换 | 电荷 |

---

## 4. 表示论基础

### 4.1 群表示

将群元素映射到线性算符：
$$\rho: G \to GL(V), \quad \rho(g_1g_2) = \rho(g_1)\rho(g_2)$$

### 4.2 量子纠错中的表示

Stabilizer码使用Pauli群的表示：

$$S = \langle g_1, g_2, \ldots, g_{n-k} \rangle$$

其中 $g_i$ 是对易的Pauli算符，共同特征空间编码量子信息。

---

## 延伸阅读

- *Group Theory and Physics* (Sternberg) — 群论与物理
- Nielsen & Chuang 第10章 — 量子纠错
- *Lie Algebras in Particle Physics* (Georgi) — 李代数

---

*最后更新：2026-06-15*

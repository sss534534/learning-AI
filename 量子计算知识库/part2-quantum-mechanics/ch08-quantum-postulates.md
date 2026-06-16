# 第八章：量子力学公设

> 量子力学的五大公设构成了量子理论的形式化基础，描述了量子系统的状态、演化、测量和复合系统。

---

## 目录

1. [公设一：态空间](#1-公设一态空间)
2. [公设二：演化](#2-公设二演化)
3. [公设三：测量](#3-公设三测量)
4. [公设四：复合系统](#4-公设四复合系统)
5. [公设五：全同粒子](#5-公设五全同粒子)

---

## 1. 公设一：态空间

### 1.1 表述

> 任何孤立物理系统都关联一个复内积空间（Hilbert空间），系统的状态由该空间的单位向量描述。

**量子比特：** 最简单的量子系统，Hilbert空间 $\mathcal{H} = \mathbb{C}^2$。

$$|\psi\rangle = \alpha |0\rangle + \beta |1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

### 1.2 态的表示

| 表示法 | 形式 | 说明 |
|--------|------|------|
| Dirac符号 | $|\psi\rangle$ | 抽象态矢量 |
| 列向量 | $\begin{bmatrix}\alpha\\\beta\end{bmatrix}$ | 具体表示 |
| Bloch球 | $(\theta, \phi)$ | 可视化（单比特） |

---

## 2. 公设二：演化

### 2.1 表述

> 封闭量子系统的演化由酉变换描述：$|\psi'\rangle = U|\psi\rangle$

### 2.2 薛定谔方程

连续时间演化：

$$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = H|\psi(t)\rangle$$

其中 $H$ 是系统的Hamiltonian（能量算符）。

### 2.3 演化算符

$$U(t) = e^{-iHt/\hbar}$$

**性质：**
- $U^\dagger U = I$（酉性）
- $U(t_1+t_2) = U(t_1)U(t_2)$（半群性）
- 量子门操作是离散的酉变换

---

## 3. 公设三：测量

### 3.1 表述

> 量子测量由一组测量算符 $\{M_m\}$ 描述，满足完备性 $\sum_m M_m^\dagger M_m = I$。

**结果 $m$ 的概率：** $P(m) = \langle \psi | M_m^\dagger M_m | \psi \rangle$

**测量后状态：** $\frac{M_m |\psi\rangle}{\sqrt{P(m)}}$

### 3.2 投影测量 vs POVM

| 测量类型 | 算符性质 | 信息量 |
|---------|---------|--------|
| 投影测量 (PVM) | $M_m$ 是正交投影 | 最大化 |
| POVM | $E_m = M_m^\dagger M_m$ | 仅概率 |
| 广义测量 | 任意 Kraus 算符 | 交互作用 |

---

## 4. 公设四：复合系统

### 4.1 表述

> 复合系统的态空间是子系统态空间的张量积：$\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B$

### 4.2 纠缠态

不可分解为张量积的状态：

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

这是最大纠缠的Bell态。

### 4.3 约化密度矩阵

对子系统 $B$ 求迹：

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

---

## 5. 公设五：全同粒子

> 多粒子系统的态在粒子交换下必须是对称的（玻色子）或反对称的（费米子）。

这导致了量子统计效应，如Pauli不相容原理。

---

## 延伸阅读

- *Principles of Quantum Mechanics* (Shankar) — 经典QM教材
- Nielsen & Chuang 第2章 — 量子计算视角
- *Quantum Mechanics* (Sakurai) — 进阶教材

---

*最后更新：2026-06-15*

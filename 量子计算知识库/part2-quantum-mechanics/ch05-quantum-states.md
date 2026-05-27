# 第五章：量子态与希尔伯特空间

> 量子态是量子力学的核心概念。本章深入探讨量子态的数学表示、希尔伯特空间公理、叠加原理以及态的演化。

---

## 5.1 希尔伯特空间公理

### 5.1.1 量子态的公设

**量子力学第一公设**：任一孤立物理系统都与一个称为系统状态空间的复内积向量空间（即希尔伯特空间）相关联。系统完全由态向量描述，态向量是状态空间中的一个单位向量。

**关键要点**：
1. 量子态由希尔伯特空间中的归一化向量表示
2. 相差一个全局相位因子的态向量描述相同的物理状态
3. 即 $|\psi\rangle$ 和 $e^{i\theta}|\psi\rangle$ 在物理上不可区分

### 5.1.2 希尔伯特空间的性质

希尔伯特空间 $\mathcal{H}$ 满足：
- 是复数域 $\mathbb{C}$ 上的向量空间
- 定义了内积 $\langle \cdot | \cdot \rangle$
- 是完备的（所有柯西序列都收敛）

对于有限维空间，完备性自动满足。

---

## 5.2 态矢量与算符

### 5.2.1 态矢量的表示

在给定正交归一基 $\{|i\rangle\}$ 下，任意态矢量 $|\psi\rangle$ 可展开为：
$$|\psi\rangle = \sum_i c_i |i\rangle$$
其中 $c_i = \langle i|\psi\rangle$ 是复系数，称为概率幅。

**归一化条件**：
$$\langle \psi|\psi\rangle = \sum_i |c_i|^2 = 1$$

### 5.2.2 线性算符

算符是将希尔伯特空间映射到自身的线性变换：
$$A: \mathcal{H} \rightarrow \mathcal{H}$$

**矩阵表示**：在基 $\{|i\rangle\}$ 下，算符 $A$ 的矩阵元素为
$$A_{ij} = \langle i|A|j\rangle$$

### 5.2.3 期望值

对于可观测量（Hermite算子）$A$，在态 $|\psi\rangle$ 中的期望值为
$$\langle A\rangle = \langle \psi|A|\psi\rangle$$

**物理意义**：期望值是对 $A$ 进行多次测量结果的平均值。

---

## 5.3 叠加原理

### 5.3.1 量子叠加

**叠加原理**：若 $|\psi_1\rangle$ 和 $|\psi_2\rangle$ 是系统的两个可能状态，则它们的任意线性组合
$$|\psi\rangle = \alpha|\psi_1\rangle + \beta|\psi_2\rangle$$
也是系统的一个可能状态，其中 $|\alpha|^2 + |\beta|^2 = 1$。

**与经典叠加的区别**：
- 经典叠加：系统确实处于某个状态，只是我们不知道
- 量子叠加：系统同时处于多个状态，直到测量发生

### 5.3.2 双缝实验演示

双缝实验是量子叠加原理的经典演示：
- 粒子同时通过两条缝
- 产生干涉图样
- 当观测粒子通过哪条缝时，干涉图样消失

**数学描述**：
$$|\psi\rangle = \frac{1}{\sqrt{2}} (|\text{缝1}\rangle + |\text{缝2}\rangle)$$

---

## 5.4 态的演化

### 5.4.1 薛定谔方程

**量子力学第二公设**：封闭量子系统的演化由薛定谔方程描述：
$$i\hbar \frac{d}{dt}|\psi(t)\rangle = H|\psi(t)\rangle$$
其中 $H$ 是哈密顿量（能量算符），$\hbar$ 是约化普朗克常数。

### 5.4.2 幺正演化

薛定谔方程的解可以表示为
$$|\psi(t)\rangle = U(t, t_0)|\psi(t_0)\rangle$$
其中演化算符 $U(t, t_0)$ 是幺正的：
$$U(t, t_0) = e^{-iH(t-t_0)/\hbar}$$

**幺正性的重要性**：
- 保持内积不变：$\langle \psi(t)|\phi(t)\rangle = \langle \psi(t_0)|\phi(t_0)\rangle$
- 保持概率归一化
- 时间演化是可逆的

### 5.4.3 离散时间演化

在量子计算中，我们通常考虑离散时间步的演化：
$$|\psi_{t+1}\rangle = U|\psi_t\rangle$$
其中 $U$ 是一个幺正量子门或量子电路。

---

## 5.5 密度算子

### 5.5.1 纯态与混合态

**纯态**：可以用单一态矢量 $|\psi\rangle$ 描述的量子态。

**混合态**：不能用单一态矢量描述，必须用态的统计系综描述：
$$\{p_i, |\psi_i\rangle\}$$
其中 $p_i$ 是处于态 $|\psi_i\rangle$ 的概率，满足 $\sum_i p_i = 1$。

### 5.5.2 密度算子的定义

**密度算子**：对于系综 $\{p_i, |\psi_i\rangle\}$，密度算子为
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

**性质**：
1. $\rho$ 是半正定的：$\langle v|\rho|v\rangle \geq 0$
2. $\rho$ 是迹为1的：$\text{tr}(\rho) = 1$
3. $\rho$ 是Hermite的：$\rho^\dagger = \rho$

### 5.5.3 纯态与混合态的判别

**定理**：密度算子 $\rho$ 描述纯态当且仅当
$$\text{tr}(\rho^2) = 1$$

对于混合态，有
$$\text{tr}(\rho^2) < 1$$

### 5.5.4 用密度算子计算期望值

对于可观测量 $A$，期望值为
$$\langle A\rangle = \text{tr}(A\rho)$$

### 5.5.5 密度算子的演化

密度算子的演化由幺正变换给出：
$$\rho \rightarrow U\rho U^\dagger$$

---

## 5.6 复合系统

### 5.6.1 张量积态

考虑两个量子系统 $A$ 和 $B$，状态空间分别为 $\mathcal{H}_A$ 和 $\mathcal{H}_B$。复合系统的状态空间为
$$\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$$

**可分态**：可以表示为张量积形式的态
$$|\psi\rangle = |\psi_A\rangle \otimes |\psi_B\rangle$$

### 5.6.2 密度算子的偏迹

**偏迹**：用于从复合系统的密度算子得到子系统的约化密度算子。

对于 $\rho_{AB} \in \mathcal{H}_A \otimes \mathcal{H}_B$，系统 $A$ 的约化密度算子为
$$\rho_A = \text{tr}_B(\rho_{AB})$$

**定义**：
$$\text{tr}_B(|a_1\rangle\langle a_2| \otimes |b_1\rangle\langle b_2|) = |a_1\rangle\langle a_2| \text{tr}(|b_1\rangle\langle b_2|)$$

**物理意义**：$\rho_A$ 包含了仅对系统 $A$ 进行测量所能得到的所有信息。

---

## 5.7 Schmidt分解

### 5.7.1 Schmidt分解定理

**Schmidt分解定理**：任意两体纯态 $|\psi\rangle_{AB}$ 可以表示为
$$|\psi\rangle_{AB} = \sum_i \sqrt{\lambda_i} |i\rangle_A \otimes |i\rangle_B$$
其中：
- $\{|i\rangle_A\}$ 和 $\{|i\rangle_B\}$ 分别是 $\mathcal{H}_A$ 和 $\mathcal{H}_B$ 的正交归一基
- $\lambda_i \geq 0$ 是Schmidt系数，满足 $\sum_i \lambda_i = 1$

### 5.7.2 Schmidt数

**Schmidt数**：非零Schmidt系数的个数。

**性质**：
- Schmidt数 = 1 $\iff$ 态是可分的
- Schmidt数 > 1 $\iff$ 态是纠缠的

Schmidt数是两体纠缠程度的一个重要度量。

---

## 5.8 量子态的距离

### 5.8.1 迹距离

**迹距离**：两个密度算子 $\rho$ 和 $\sigma$ 之间的迹距离定义为
$$D(\rho, \sigma) = \frac{1}{2} \text{tr}|\rho - \sigma|$$
其中 $|A| = \sqrt{A^\dagger A}$。

**性质**：
- $0 \leq D(\rho, \sigma) \leq 1$
- $D(\rho, \sigma) = 0$ 当且仅当 $\rho = \sigma$
- 对幺正变换不变：$D(U\rho U^\dagger, U\sigma U^\dagger) = D(\rho, \sigma)$

### 5.8.2 保真度

**保真度**：两个量子态 $\rho$ 和 $\sigma$ 之间的保真度定义为
$$F(\rho, \sigma) = \text{tr}\sqrt{\sqrt{\rho} \sigma \sqrt{\rho}}$$

对于纯态 $|\psi\rangle$ 和 $|\phi\rangle$，保真度简化为
$$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|$$

**性质**：
- $0 \leq F(\rho, \sigma) \leq 1$
- $F(\rho, \sigma) = 1$ 当且仅当 $\rho = \sigma$
- 对幺正变换不变

---

## 本章小结

本章深入介绍了量子态的核心概念，包括：
- 希尔伯特空间公理
- 态矢量与算符
- 量子叠加原理
- 态的幺正演化
- 密度算子与混合态
- 复合系统与偏迹
- Schmidt分解
- 量子态的距离度量

这些概念是理解量子计算和量子信息的基础。

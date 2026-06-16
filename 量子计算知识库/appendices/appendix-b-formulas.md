# 附录B：常用公式汇总

> 量子计算常用公式速查，按主题分类。

---

## 目录

1. [线性代数](#1-线性代数)
2. [量子力学](#2-量子力学)
3. [量子信息](#3-量子信息)
4. [量子算法](#4-量子算法)
5. [量子纠错](#5-量子纠错)
6. [数值常数](#6-数值常数)

---

## 1. 线性代数

### 1.1 基本公式

- **欧拉公式**：$e^{i\theta} = \cos\theta + i\sin\theta$
- **逆矩阵**：$A^{-1}A = AA^{-1} = I$
- **特征方程**：$A\vec{v} = \lambda\vec{v}$

### 1.2 张量积性质

- $(A \otimes B)(C \otimes D) = AC \otimes BD$
- $(A \otimes B)^\dagger = A^\dagger \otimes B^\dagger$
- $\text{Tr}(A \otimes B) = \text{Tr}(A) \cdot \text{Tr}(B)$
- $\text{det}(A \otimes B) = \text{det}(A)^{\dim(B)} \cdot \text{det}(B)^{\dim(A)}$

### 1.3 矩阵分解

- **谱分解**：$H = \sum_i \lambda_i |i\rangle\langle i|$
- **奇异值分解**：$A = U\Sigma V^\dagger$
- **极分解**：$A = UP$（$U$ 幺正，$P$ 正定）

---

## 2. 量子力学

### 2.1 核心方程

- **薛定谔方程**：$i\hbar\frac{\partial}{\partial t}|\psi\rangle = H|\psi\rangle$
- **定态薛定谔方程**：$H|\psi\rangle = E|\psi\rangle$
- **时间演化算符**：$U(t) = e^{-iHt/\hbar}$

### 2.2 测量公理

- **Born规则**：$P(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$
- **归一化**：$\sum_m M_m^\dagger M_m = I$
- **期望值**：$\langle A \rangle = \langle\psi|A|\psi\rangle$

### 2.3 不确定性关系

- **Heisenberg不确定性**：$\Delta A \cdot \Delta B \geq \frac{1}{2}|\langle[A,B]\rangle|$
- **能量-时间**：$\Delta E \cdot \Delta t \geq \frac{\hbar}{2}$

### 2.4 密度矩阵

- **定义**：$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$
- **演化**：$\rho(t) = U(t)\rho(0)U^\dagger(t)$
- **混合态条件**：$\text{Tr}(\rho^2) < 1$

---

## 3. 量子信息

### 3.1 熵

- **冯·诺依曼熵**：$S(\rho) = -\text{Tr}(\rho\log\rho)$
- **香农熵**：$H(X) = -\sum_i p(x_i)\log p(x_i)$
- **条件熵**：$S(A|B) = S(AB) - S(B)$
- **相对熵**：$S(\rho\|\sigma) = \text{Tr}(\rho\log\rho) - \text{Tr}(\rho\log\sigma)$

### 3.2 信息度量

- **互信息**：$I(A:B) = S(A) + S(B) - S(AB)$
- **Holevo界**：$\chi = S(\rho) - \sum_i p_i S(\rho_i)$
- **保真度**：$F(\rho,\sigma) = \text{Tr}\sqrt{\rho^{1/2}\sigma\rho^{1/2}}$

### 3.3 量子信道

- **Kraus表示**：$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$
- **完全正定性**：$\sum_k E_k^\dagger E_k = I$

### 3.4 纠缠度量

- **部分转置判据**：$\rho^{T_B} \geq 0$（可分离态）
- **纠缠熵**：$E(|\psi\rangle) = S(\text{Tr}_A|\psi\rangle\langle\psi|)$
- **并发度**：$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$

---

## 4. 量子算法

### 4.1 Grover搜索

- **迭代次数**：$O(\sqrt{N})$
- **每次迭代**：$G = (2|\psi\rangle\langle\psi| - I)O$
- **成功概率**：$\sin^2((2k+1)\theta)$，其中 $\theta = \arcsin(1/\sqrt{N})$

### 4.2 Shor算法

- **周期寻找**：$f(x+r) = f(x)$
- **连续分数展开**：从QFT结果提取有理数
- **复杂度**：$O((\log N)^2(\log\log N)(\log\log\log N))$

### 4.3 量子傅里叶变换

- **变换公式**：$|j\rangle \to \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} \omega_N^{jk}|k\rangle$
- **旋转角度**：$\omega_N = e^{2\pi i/N}$
- **深度**：$O((\log N)^2)$ 门

### 4.4 振幅估计

- **估计精度**：$|\tilde{a} - a| \leq \frac{2\pi}{M} + \frac{\pi^2}{M^2}$
- **复杂度**：$O(1/\epsilon)$ 相比经典 $O(1/\epsilon^2)$

---

## 5. 量子纠错

### 5.1 重复码

- **3比特码**：$|0_L\rangle = |000\rangle$，$|1_L\rangle = |111\rangle$
- **可纠正错误**：单量子比特翻转 (X error)

### 5.2 Shor码

- **9量子比特**：结合相位翻转和比特翻转保护
- **$|0_L\rangle$**：$\frac{(|000\rangle+|111\rangle)^{\otimes 3}}{2\sqrt{2}}$

### 5.3 Surface码

- **码距与保真度**：$P_{\text{fail}} \propto \left(\frac{p}{p_{\text{th}}}\right)^{d/2}$
- **阈值**：$p_{\text{th}} \approx 1\%$
- **编码率**：$k/n \to 0$（随 $d$ 增长）

### 5.4 容错阈值定理

> 如果物理错误率低于某个阈值，则可以通过量子纠错实现任意精度的量子计算。

---

## 6. 数值常数

| 常数 | 符号 | 数值 |
|------|------|------|
| 普朗克常数 | $h$ | $6.626 \times 10^{-34} \text{ J·s}$ |
| 约化普朗克常数 | $\hbar$ | $1.055 \times 10^{-34} \text{ J·s}$ |
| 精细结构常数 | $\alpha$ | $1/137.036$ |
| Bohr半径 | $a_0$ | $0.529 \times 10^{-10} \text{ m}$ |
| Hartree能量 | $E_h$ | $4.360 \times 10^{-18} \text{ J}$ |
| Boltzmann常数 | $k_B$ | $1.381 \times 10^{-23} \text{ J/K}$ |

---

*最后更新：2026-06-15*

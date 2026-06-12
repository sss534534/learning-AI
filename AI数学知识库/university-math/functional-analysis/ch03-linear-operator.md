# 第3章 线性算子


## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


## 3.1 线性算子的定义与例子

### 3.1.1 线性算子的定义

**定义3.1.1（线性算子）** 设 $X, Y$ 为同一数域 $\mathbb{K}$（$\mathbb{K}=\mathbb{R}$ 或 $\mathbb{C}$）上的线性空间，$D(T) \subset X$ 为线性子空间。若映射 $T: D(T) \to Y$ 满足：

1. **可加性**：对任意 $x_1, x_2 \in D(T)$，有 $T(x_1 + x_2) = Tx_1 + Tx_2$；
2. **齐次性**：对任意 $x \in D(T)$ 和 $\alpha \in \mathbb{K}$，有 $T(\alpha x) = \alpha Tx$，

则称 $T$ 为**线性算子**。$D(T)$ 称为 $T$ 的**定义域**，$R(T) = \{Tx \mid x \in D(T)\}$ 称为 $T$ 的**值域**。

当 $Y = \mathbb{K}$ 时，线性算子 $T: D(T) \to \mathbb{K}$ 称为**线性泛函**。

### 3.1.2 微分算子

**例3.1.1（微分算子）** 设 $X = C^1[a, b]$ 为 $[a, b]$ 上连续可微函数空间，$Y = C[a, b]$ 为连续函数空间。定义微分算子 $D: X \to Y$ 为：

$$
(Df)(t) = f'(t), \quad \forall f \in C^1[a, b]
$$

显然，$D$ 是线性算子。

更一般地，考虑 $n$ 阶微分算子：

$$
L = a_n(t) \frac{d^n}{dt^n} + a_{n-1}(t) \frac{d^{n-1}}{dt^{n-1}} + \cdots + a_1(t) \frac{d}{dt} + a_0(t)
$$

其中 $a_k(t) \in C[a, b]$，则 $L: C^n[a, b] \to C[a, b]$ 是线性算子。

### 3.1.3 积分算子

**例3.1.2（积分算子）** 设 $K(t, s)$ 为 $[a, b] \times [a, b]$ 上的连续函数，定义积分算子 $T: C[a, b] \to C[a, b]$ 为：

$$
(Tf)(t) = \int_a^b K(t, s) f(s) ds, \quad \forall f \in C[a, b]
$$

**证明线性性**：对任意 $f, g \in C[a, b]$ 和 $\alpha, \beta \in \mathbb{K}$，有

$$
\begin{aligned}
T(\alpha f + \beta g)(t) &= \int_a^b K(t, s) [\alpha f(s) + \beta g(s)] ds \\
&= \alpha \int_a^b K(t, s) f(s) ds + \beta \int_a^b K(t, s) g(s) ds \\
&= \alpha (Tf)(t) + \beta (Tg)(t)
\end{aligned}
$$

故 $T$ 是线性算子。

**例3.1.3（Fredholm积分算子）** 核函数 $K(t, s)$ 称为Fredholm核，对应的积分算子称为Fredholm积分算子。当 $K(t, s) = k(t-s)$ 时，称为**卷积算子**，记为 $k * f$。

---

## 3.2 有界线性算子与算子范数、算子空间 $B(X, Y)$

### 3.2.1 有界线性算子

**定义3.2.1（有界线性算子）** 设 $X, Y$ 为赋范线性空间，$T: D(T) \to Y$ 为线性算子。若存在常数 $M \geq 0$，使得对任意 $x \in D(T)$，有

$$
\|Tx\|_Y \leq M \|x\|_X
$$

则称 $T$ 为**有界线性算子**。

**定理3.2.1** 设 $X, Y$ 为赋范线性空间，$T: X \to Y$ 为线性算子，则以下命题等价：

1. $T$ 是有界的；
2. $T$ 在 $X$ 上连续；
3. $T$ 在某一点 $x_0 \in X$ 连续。

**证明思路**：
- (1) $\Rightarrow$ (2)：由 $\|Tx - Ty\| \leq M\|x - y\|$ 直接得连续性。
- (2) $\Rightarrow$ (3)：显然。
- (3) $\Rightarrow$ (1)：由在 $x_0$ 连续，取 $\varepsilon = 1$，存在 $\delta > 0$，当 $\|x - x_0\| < \delta$ 时，$\|Tx - Tx_0\| < 1$。对任意 $z \neq 0$，令 $x = x_0 + \frac{\delta}{2\|z\|}z$，则 $\|x - x_0\| = \delta/2 < \delta$，故 $\|Tz\| \leq \frac{2}{\delta}\|z\|$，即 $T$ 有界。

### 3.2.2 算子范数

**定义3.2.2（算子范数）** 设 $T: X \to Y$ 为有界线性算子，定义 $T$ 的**范数**为：

$$
\|T\| = \sup_{\substack{x \in X \\ x \neq 0}} \frac{\|Tx\|}{\|x\|} = \sup_{\substack{x \in X \\ \|x\| \leq 1}} \|Tx\| = \sup_{\substack{x \in X \\ \|x\| = 1}} \|Tx\|
$$

**定理3.2.2** 算子范数满足范数公理：

1. **非负性**：$\|T\| \geq 0$，且 $\|T\| = 0$ 当且仅当 $T = 0$；
2. **齐次性**：$\|\alpha T\| = |\alpha| \|T\|$；
3. **三角不等式**：$\|T_1 + T_2\| \leq \|T_1\| + \|T_2\|$；
4. **次可乘性**：若 $T_1: Y \to Z$，$T_2: X \to Y$ 均为有界线性算子，则 $\|T_1 T_2\| \leq \|T_1\| \|T_2\|$。

**例3.2.1** 考虑积分算子 $T: C[a, b] \to C[a, b]$，$(Tf)(t) = \int_a^b K(t, s) f(s) ds$，其中 $K \in C([a, b] \times [a, b])$，则

$$
\|T\| = \max_{t \in [a, b]} \int_a^b |K(t, s)| ds
$$

### 3.2.3 算子空间 $B(X, Y)$

**定义3.2.3** 设 $X, Y$ 为赋范线性空间，记 $B(X, Y)$ 为所有从 $X$ 到 $Y$ 的有界线性算子构成的集合。在 $B(X, Y)$ 中定义加法和数乘：

$$
(T_1 + T_2)x = T_1x + T_2x, \quad (\alpha T)x = \alpha Tx
$$

则 $B(X, Y)$ 成为线性空间，配以算子范数后成为赋范线性空间，称为**有界线性算子空间**。

**定理3.2.3** 若 $Y$ 是Banach空间，则 $B(X, Y)$ 也是Banach空间。

**证明思路**：设 $\{T_n\}$ 为 $B(X, Y)$ 中的Cauchy列，则对任意 $x \in X$，$\{T_n x\}$ 是 $Y$ 中的Cauchy列。由 $Y$ 完备，可定义 $Tx = \lim_{n \to \infty} T_n x$。验证 $T$ 是有界线性算子且 $T_n \to T$。

当 $Y = X$ 时，记 $B(X) = B(X, X)$，此时 $B(X)$ 是Banach代数。

---

## 3.3 线性泛函、对偶空间、里茨表示定理

### 3.3.1 线性泛函

**定义3.3.1（线性泛函）** 设 $X$ 为数域 $\mathbb{K}$ 上的线性空间，若线性泛函 $f: X \to \mathbb{K}$ 是有界的，即存在 $M \geq 0$，使得

$$
|f(x)| \leq M \|x\|, \quad \forall x \in X
$$

则称 $f$ 为**有界线性泛函**，其范数定义为

$$
\|f\| = \sup_{\|x\| = 1} |f(x)|
$$

### 3.3.2 对偶空间

**定义3.3.2（对偶空间）** 设 $X$ 为赋范线性空间，$X$ 上所有有界线性泛函构成的Banach空间称为 $X$ 的**对偶空间**，记为 $X^*$。

**例3.3.1**
1. $(\mathbb{R}^n)^* \cong \mathbb{R}^n$：对任意 $f \in (\mathbb{R}^n)^*$，存在唯一的 $a \in \mathbb{R}^n$，使得 $f(x) = \langle a, x \rangle$，且 $\|f\| = \|a\|$。
2. $(l^p)^* \cong l^q$，其中 $1 \leq p < \infty$，$\frac{1}{p} + \frac{1}{q} = 1$：对任意 $f \in (l^p)^*$，存在唯一的 $y = (y_n) \in l^q$，使得 $f(x) = \sum_{n=1}^\infty x_n y_n$，且 $\|f\| = \|y\|_q$。
3. $(L^p[a, b])^* \cong L^q[a, b]$，其中 $1 \leq p < \infty$，$\frac{1}{p} + \frac{1}{q} = 1$：对任意 $f \in (L^p[a, b])^*$，存在唯一的 $g \in L^q[a, b]$，使得 $f(x) = \int_a^b x(t) g(t) dt$，且 $\|f\| = \|g\|_q$。

### 3.3.3 里茨表示定理（希尔伯特空间）

**定理3.3.1（里茨表示定理）** 设 $H$ 为希尔伯特空间，则对任意 $f \in H^*$，存在唯一的 $y \in H$，使得

$$
f(x) = \langle x, y \rangle, \quad \forall x \in H
$$

且 $\|f\| = \|y\|$。

**证明思路**：
1. **存在性**：若 $f = 0$，取 $y = 0$。否则，令 $N(f) = \{x \in H \mid f(x) = 0\}$ 为 $f$ 的零空间，则 $N(f)$ 是 $H$ 的闭子空间。取 $z \in N(f)^\perp$，$z \neq 0$，令 $y = \frac{\overline{f(z)}}{\|z\|^2} z$，验证 $f(x) = \langle x, y \rangle$。
2. **唯一性**：若 $\langle x, y_1 \rangle = \langle x, y_2 \rangle$ 对所有 $x$ 成立，则 $\langle x, y_1 - y_2 \rangle = 0$，取 $x = y_1 - y_2$ 得 $y_1 = y_2$。
3. **范数相等**：由Cauchy-Schwarz不等式，$|f(x)| \leq \|y\|\|x\|$，故 $\|f\| \leq \|y\|$。又 $f(y) = \langle y, y \rangle = \|y\|^2$，故 $\|f\| \geq \|y\|$，从而 $\|f\| = \|y\|$。

里茨表示定理建立了希尔伯特空间与其对偶空间之间的共轭线性同构。

---

## 3.4 自伴算子、正规算子、酉算子

### 3.4.1 伴随算子

**定义3.4.1（伴随算子）** 设 $H$ 为希尔伯特空间，$T \in B(H)$。由里茨表示定理，对任意 $y \in H$，映射 $x \mapsto \langle Tx, y \rangle$ 是有界线性泛函，故存在唯一的 $T^* y \in H$，使得

$$
\langle Tx, y \rangle = \langle x, T^* y \rangle, \quad \forall x, y \in H
$$

称 $T^*$ 为 $T$ 的**伴随算子**。

**定理3.4.1** 伴随算子具有以下性质：

1. $(T_1 + T_2)^* = T_1^* + T_2^*$；
2. $(\alpha T)^* = \overline{\alpha} T^*$；
3. $(T_1 T_2)^* = T_2^* T_1^*$；
4. $(T^*)^* = T$；
5. $\|T^*\| = \|T\|$；
6. $\|T^* T\| = \|T\|^2$。

### 3.4.2 自伴算子

**定义3.4.2（自伴算子）** 设 $T \in B(H)$，若 $T^* = T$，则称 $T$ 为**自伴算子**（或Hermite算子）。

等价地，$T$ 是自伴算子当且仅当对任意 $x, y \in H$，有

$$
\langle Tx, y \rangle = \langle x, Ty \rangle
$$

**定理3.4.2** 设 $T$ 为自伴算子，则

1. $\|T\| = \sup_{\|x\| = 1} |\langle Tx, x \rangle|$；
2. $T$ 的特征值均为实数；
3. $T$ 的对应于不同特征值的特征向量相互正交。

**例3.4.1** $n \times n$ Hermite矩阵是自伴算子；实对称矩阵是自伴算子。

### 3.4.3 正规算子

**定义3.4.3（正规算子）** 设 $T \in B(H)$，若 $T^* T = T T^*$，则称 $T$ 为**正规算子**。

显然，自伴算子是正规算子。

**定理3.4.3** 设 $T$ 为正规算子，则

1. $\|Tx\| = \|T^* x\|$ 对所有 $x \in H$ 成立；
2. $N(T) = N(T^*)$，$R(T)^\perp = N(T)$；
3. 若 $\lambda$ 是 $T$ 的特征值，则 $\overline{\lambda}$ 是 $T^*$ 的特征值；
4. 对应于不同特征值的特征向量相互正交。

### 3.4.4 酉算子

**定义3.4.4（酉算子）** 设 $U \in B(H)$，若 $U^* U = U U^* = I$，则称 $U$ 为**酉算子**。

等价地，$U$ 是酉算子当且仅当 $U$ 是满射且保内积，即

$$
\langle Ux, Uy \rangle = \langle x, y \rangle, \quad \forall x, y \in H
$$

**定理3.4.4** 酉算子具有以下性质：

1. $\|Ux\| = \|x\|$ 对所有 $x \in H$ 成立（保距）；
2. $U$ 的特征值的模均为1；
3. 酉算子的乘积仍是酉算子；
4. $U^* = U^{-1}$。

**例3.4.2** Fourier变换是酉算子；有限维空间上的正交矩阵（实）或酉矩阵（复）是酉算子。

---

## 3.5 谱论初步

### 3.5.1 预解集与谱

**定义3.5.1（预解集与谱）** 设 $X$ 为复Banach空间，$T \in B(X)$，$\lambda \in \mathbb{C}$。

1. 若 $\lambda I - T$ 是双射且 $(\lambda I - T)^{-1} \in B(X)$，则称 $\lambda$ 为 $T$ 的**正则值**，正则值的全体称为 $T$ 的**预解集**，记为 $\rho(T)$。称 $R(\lambda, T) = (\lambda I - T)^{-1}$ 为 $T$ 的**预解算子**。
2. 若 $\lambda$ 不是正则值，则称 $\lambda$ 为 $T$ 的**谱点**，谱点的全体称为 $T$ 的**谱**，记为 $\sigma(T)$。

谱 $\sigma(T)$ 可分为三部分：

1. **点谱** $\sigma_p(T)$：$\lambda I - T$ 不是单射，即存在非零 $x \in X$，使得 $Tx = \lambda x$，称 $\lambda$ 为**特征值**，$x$ 为对应的**特征向量**。
2. **连续谱** $\sigma_c(T)$：$\lambda I - T$ 是单射但不是满射，且值域 $R(\lambda I - T)$ 在 $X$ 中稠密。
3. **剩余谱** $\sigma_r(T)$：$\lambda I - T$ 是单射，但值域 $R(\lambda I - T)$ 在 $X$ 中不稠密。

### 3.5.2 特征值与特征向量

**定义3.5.2（特征值与特征向量）** 设 $T \in B(X)$，若存在 $\lambda \in \mathbb{C}$ 和非零 $x \in X$，使得

$$
Tx = \lambda x
$$

则称 $\lambda$ 为 $T$ 的**特征值**，$x$ 为 $T$ 对应于 $\lambda$ 的**特征向量**。所有对应于 $\lambda$ 的特征向量与零向量构成的线性子空间称为 $\lambda$ 的**特征子空间**，记为 $E_\lambda(T)$。

**定理3.5.1** 设 $T \in B(X)$，则

1. 谱 $\sigma(T)$ 是 $\mathbb{C}$ 中的非空有界闭集，且 $\sigma(T) \subset \{ \lambda \in \mathbb{C} \mid |\lambda| \leq \|T\| \}$；
2. 谱半径 $r(T) = \sup_{\lambda \in \sigma(T)} |\lambda| = \lim_{n \to \infty} \|T^n\|^{1/n}$（Gelfand公式）。

### 3.5.3 自伴算子的谱分解

**定理3.5.2（谱定理-有限维情形）** 设 $A$ 为 $n \times n$ Hermite矩阵，则存在酉矩阵 $U$，使得

$$
U^* A U = \Lambda = \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_n)
$$

其中 $\lambda_1, \lambda_2, \cdots, \lambda_n$ 为 $A$ 的特征值（按重数计）。等价地，$A$ 可表示为

$$
A = \sum_{k=1}^n \lambda_k P_k
$$

其中 $P_k$ 是投影到特征子空间 $E_{\lambda_k}$ 的正交投影算子，满足 $P_k P_j = 0$（$k \neq j$）且 $\sum_{k=1}^n P_k = I$。

**定理3.5.3（谱定理-紧自伴算子）** 设 $H$ 为可分希尔伯特空间，$T$ 为紧自伴算子，则

1. $T$ 的谱 $\sigma(T)$ 最多是可数集，除0外无聚点；
2. 非零谱点均为特征值，且对应的特征子空间是有限维的；
3. 存在正交归一基 $\{e_n\}$，使得 $T e_n = \lambda_n e_n$，且 $T$ 可表示为

$$
T x = \sum_{n=1}^\infty \lambda_n \langle x, e_n \rangle e_n, \quad \forall x \in H
$$

### 3.5.4 一般自伴算子的谱分解

**定理3.5.4（谱定理-有界自伴算子）** 设 $T$ 为希尔伯特空间 $H$ 上的有界自伴算子，则存在唯一的谱测度（或投影值测度）$E$，使得

$$
T = \int_{m-0}^M \lambda dE(\lambda)
$$

其中 $m = \inf_{\|x\|=1} \langle Tx, x \rangle$，$M = \sup_{\|x\|=1} \langle Tx, x \rangle$。

对任意有界Borel可测函数 $f$，可定义算子函数

$$
f(T) = \int_{m-0}^M f(\lambda) dE(\lambda)
$$

---

## 3.6 线性算子在AI中的应用

### 3.6.1 卷积算子

**定义3.6.1（卷积）** 设 $f, g \in L^1(\mathbb{R}^n)$，定义卷积 $f * g$ 为

$$
(f * g)(x) = \int_{\mathbb{R}^n} f(x - y) g(y) dy = \int_{\mathbb{R}^n} f(y) g(x - y) dy
$$

**卷积算子** $T_g: L^2(\mathbb{R}^n) \to L^2(\mathbb{R}^n)$ 定义为 $T_g f = f * g$。

**性质**：
1. 卷积算子是线性算子；
2. 由Fourier变换的卷积定理，$\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$，故卷积算子在频域上等价于逐点乘法算子；
3. 卷积算子是平移不变的：$T_g (\tau_h f) = \tau_h (T_g f)$，其中 $\tau_h f(x) = f(x - h)$。

**AI中的应用**：
- **卷积神经网络(CNN)**：卷积层通过卷积算子提取局部特征，实现平移不变性。
- **图像滤波**：高斯滤波、边缘检测等均通过卷积实现。
- **信号处理**：降噪、平滑等操作。

### 3.6.2 注意力算子

**定义3.6.2（注意力机制）** 在Transformer模型中，自注意力算子定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$（Query）、$K \in \mathbb{R}^{n \times d_k}$（Key）、$V \in \mathbb{R}^{n \times d_v}$（Value）。

**作为线性算子的理解**：
1. 线性变换：首先通过线性变换将输入投影到 $Q, K, V$ 空间，即 $Q = X W_Q$，$K = X W_K$，$V = X W_V$，其中 $W_Q, W_K, W_V$ 为可学习的参数矩阵。
2. 注意力权重：计算 $A = \text{softmax}(Q K^T / \sqrt{d_k})$，$A$ 可以看作是一个注意力矩阵，$A_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的注意力权重。
3. 加权求和：输出为 $A V$，即每个位置的输出是所有位置Value的加权和。

**定理3.6.1** 自注意力算子可以表示为序列空间上的线性算子的组合，具有全局感受野，能够捕获长程依赖关系。

**AI中的应用**：
- **自然语言处理**：Transformer、BERT、GPT等模型的核心组件。
- **计算机视觉**：Vision Transformer(ViT)、DETR等。
- **图神经网络**：图注意力网络(GAT)。

### 3.6.3 图拉普拉斯算子

**定义3.6.3（图拉普拉斯矩阵）** 设 $G = (V, E)$ 为无向图，$V = \{v_1, \cdots, v_n\}$ 为顶点集，$E$ 为边集。定义：

1. **邻接矩阵** $A \in \mathbb{R}^{n \times n}$：$A_{ij} = 1$ 若 $(v_i, v_j) \in E$，否则为0。
2. **度矩阵** $D \in \mathbb{R}^{n \times n}$：对角矩阵，$D_{ii} = \sum_{j=1}^n A_{ij}$ 为顶点 $v_i$ 的度。
3. **（未归一化）图拉普拉斯矩阵**：$L = D - A$。
4. **归一化图拉普拉斯矩阵**：$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$ 或 $L_{\text{rw}} = I - D^{-1} A$。

**图拉普拉斯算子的性质**：

**定理3.6.2** 图拉普拉斯矩阵 $L$ 具有以下性质：

1. $L$ 是对称半正定矩阵；
2. $L$ 的所有特征值非负；
3. 最小特征值为0，对应的特征向量为全1向量 $\mathbf{1}$；
4. 特征值0的重数等于图的连通分支数；
5. 对任意向量 $f \in \mathbb{R}^n$，有 $f^T L f = \frac{1}{2} \sum_{i,j} A_{ij} (f_i - f_j)^2$。

**谱图理论**：利用图拉普拉斯的谱性质进行图分析，包括图划分、图嵌入等。

**AI中的应用**：

1. **图神经网络(GNN)**：
   - GCN（图卷积网络）：利用归一化拉普拉斯矩阵的谱分解定义图卷积。
   - 图滤波：$y = \sum_{k=0}^K w_k L^k x$。

2. **谱聚类**：利用图拉普拉斯的特征向量进行数据聚类。

3. **图嵌入**：通过拉普拉斯特征映射将图顶点嵌入到低维空间。

4. **流形学习**：将数据视为高维空间中的流形，利用图拉普拉斯构造图上的光滑函数。

---

## 习题

1. 证明：赋范空间上的线性算子连续当且仅当它将有界集映为有界集。

2. 设 $T: l^2 \to l^2$ 定义为 $(Tx)_n = x_{n+1}$（右移算子），求 $T$ 的伴随算子 $T^*$。

3. 证明：自伴算子的谱均为实数。

4. 设 $A$ 为 $n \times n$ 实对称矩阵，证明：存在正交矩阵 $Q$，使得 $Q^T A Q$ 为对角矩阵。

5. 考虑图拉普拉斯矩阵 $L = D - A$，证明：对任意向量 $f$，有 $f^T L f = \frac{1}{2} \sum_{i,j} A_{ij} (f_i - f_j)^2$。

---

## 参考文献

1. 张恭庆, 郭懋正. 泛函分析讲义(上册). 北京大学出版社, 1987.
2. 郭懋正. 泛函分析讲义(下册). 北京大学出版社, 1990.
3. Conway, J. B. A Course in Functional Analysis. Springer, 2007.
4. Rudin, W. Functional Analysis. McGraw-Hill, 1991.
5. Shalev-Shwartz, S., & Ben-David, S. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.
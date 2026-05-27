# 第1章 度量空间

## 1.1 度量空间的定义与例子

### 1.1.1 度量空间的定义

**定义1.1.1（度量空间）** 设 $X$ 为非空集合，若映射 $d: X \times X \rightarrow \mathbb{R}$ 满足以下条件：

1. **非负性**：对任意 $x, y \in X$，有 $d(x, y) \geq 0$，且 $d(x, y) = 0$ 当且仅当 $x = y$；
2. **对称性**：对任意 $x, y \in X$，有 $d(x, y) = d(y, x)$；
3. **三角不等式**：对任意 $x, y, z \in X$，有 $d(x, y) \leq d(x, z) + d(z, y)$。

则称 $(X, d)$ 为**度量空间**（Metric Space），$d$ 称为 $X$ 上的**度量**（Metric）或**距离函数**。

注：三角不等式是度量空间最重要的性质，它刻画了"两点之间直线最短"的几何直觉。

### 1.1.2 欧氏空间

**例子1.1.1（n维欧氏空间 $\mathbb{R}^n$）** 设 $X = \mathbb{R}^n$，对任意 $x = (x_1, x_2, \dots, x_n), y = (y_1, y_2, \dots, y_n) \in \mathbb{R}^n$，定义：

$$d_2(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

易验证 $(\mathbb{R}^n, d_2)$ 是度量空间，称为**n维欧氏空间**（Euclidean Space）。

**定理1.1.1（柯西-施瓦茨不等式）** 对任意实数 $a_1, a_2, \dots, a_n$ 和 $b_1, b_2, \dots, b_n$，有：

$$\left(\sum_{i=1}^n a_i b_i\right)^2 \leq \left(\sum_{i=1}^n a_i^2\right)\left(\sum_{i=1}^n b_i^2\right)$$

**证明思路**：考虑关于 $t$ 的二次函数 $\sum_{i=1}^n (a_i t + b_i)^2 \geq 0$，其判别式非正即得。

利用柯西-施瓦茨不等式可证 $d_2$ 满足三角不等式。

**其他常用度量**：
- $d_1(x, y) = \sum_{i=1}^n |x_i - y_i|$（曼哈顿距离）
- $d_\infty(x, y) = \max_{1 \leq i \leq n} |x_i - y_i|$（切比雪夫距离）

### 1.1.3 连续函数空间

**例子1.1.2（连续函数空间 $C[a, b]$）** 设 $[a, b]$ 为闭区间，$C[a, b]$ 表示 $[a, b]$ 上所有实值连续函数构成的集合。对任意 $f, g \in C[a, b]$，定义：

$$d_\infty(f, g) = \max_{x \in [a, b]} |f(x) - g(x)|$$

称 $d_\infty$ 为**一致度量**或**上确界度量**。

**定理1.1.2** $(C[a, b], d_\infty)$ 是度量空间。

**证明**：
1. 非负性显然，$d_\infty(f, g) = 0$ 当且仅当对所有 $x \in [a, b]$，$f(x) = g(x)$，即 $f = g$。
2. 对称性由绝对值的对称性立得。
3. 三角不等式：对任意 $x \in [a, b]$，$|f(x) - g(x)| \leq |f(x) - h(x)| + |h(x) - g(x)| \leq d_\infty(f, h) + d_\infty(h, g)$，取最大值即得 $d_\infty(f, g) \leq d_\infty(f, h) + d_\infty(h, g)$。

### 1.1.4 序列空间

**例子1.1.3（$\ell^p$ 空间，$1 \leq p < \infty$）** 设 $\ell^p$ 表示所有满足 $\sum_{i=1}^\infty |x_i|^p < \infty$ 的实序列 $x = (x_1, x_2, \dots)$ 构成的集合。定义：

$$d_p(x, y) = \left(\sum_{i=1}^\infty |x_i - y_i|^p\right)^{1/p}$$

**定理1.1.3（闵可夫斯基不等式）** 对任意 $1 \leq p < \infty$，以及任意序列 $x, y \in \ell^p$，有：

$$\left(\sum_{i=1}^\infty |x_i + y_i|^p\right)^{1/p} \leq \left(\sum_{i=1}^\infty |x_i|^p\right)^{1/p} + \left(\sum_{i=1}^\infty |y_i|^p\right)^{1/p}$$

由闵可夫斯基不等式可知 $d_p$ 满足三角不等式，故 $(\ell^p, d_p)$ 是度量空间。

**例子1.1.4（$\ell^\infty$ 空间）** 设 $\ell^\infty$ 表示所有有界实序列构成的集合，定义：

$$d_\infty(x, y) = \sup_{i \geq 1} |x_i - y_i|$$

易证 $(\ell^\infty, d_\infty)$ 是度量空间。

---

## 1.2 开集、闭集、内部、闭包、边界

### 1.2.1 开球与闭球

**定义1.2.1（开球）** 设 $(X, d)$ 为度量空间，$x_0 \in X$，$r > 0$，称集合

$$B_r(x_0) = \{x \in X \mid d(x, x_0) < r\}$$

为以 $x_0$ 为中心、$r$ 为半径的**开球**（Open Ball）。

**定义1.2.2（闭球）** 类似地，称集合

$$\overline{B}_r(x_0) = \{x \in X \mid d(x, x_0) \leq r\}$$

为以 $x_0$ 为中心、$r$ 为半径的**闭球**（Closed Ball）。

### 1.2.2 开集

**定义1.2.3（开集）** 设 $(X, d)$ 为度量空间，$G \subset X$。若对任意 $x \in G$，存在 $\varepsilon > 0$，使得 $B_\varepsilon(x) \subset G$，则称 $G$ 为**开集**（Open Set）。

**定理1.2.1（开集的性质）** 度量空间 $(X, d)$ 中的开集具有以下性质：

1. $\emptyset$ 和 $X$ 都是开集；
2. 任意多个开集的并集是开集；
3. 有限多个开集的交集是开集。

**证明思路**：
1. 显然。
2. 设 $\{G_\alpha\}_{\alpha \in I}$ 是一族开集，令 $G = \bigcup_{\alpha \in I} G_\alpha$。对任意 $x \in G$，存在 $\alpha_0 \in I$ 使得 $x \in G_{\alpha_0}$，由开集定义存在开球含于 $G_{\alpha_0}$，故含于 $G$。
3. 设 $G_1, G_2, \dots, G_n$ 为开集，令 $G = \bigcap_{i=1}^n G_i$。对任意 $x \in G$，对每个 $i$ 存在 $r_i > 0$ 使得 $B_{r_i}(x) \subset G_i$，取 $r = \min\{r_1, \dots, r_n\}$，则 $B_r(x) \subset G$。

**注**：无限多个开集的交集不一定是开集。例如在 $\mathbb{R}$ 中，$\bigcap_{n=1}^\infty (-1/n, 1/n) = \{0\}$ 不是开集。

### 1.2.3 内部、闭包、边界

**定义1.2.4（内部点与内部）** 设 $(X, d)$ 为度量空间，$A \subset X$，$x \in A$。若存在 $\varepsilon > 0$ 使得 $B_\varepsilon(x) \subset A$，则称 $x$ 为 $A$ 的**内部点**。$A$ 的所有内部点构成的集合称为 $A$ 的**内部**，记为 $\mathring{A}$ 或 $\text{int}(A)$。

**定义1.2.5（接触点与闭包）** 设 $A \subset X$，$x \in X$。若对任意 $\varepsilon > 0$，$B_\varepsilon(x) \cap A \neq \emptyset$，则称 $x$ 为 $A$ 的**接触点**。$A$ 的所有接触点构成的集合称为 $A$ 的**闭包**，记为 $\overline{A}$。

**定义1.2.6（边界点与边界）** 设 $A \subset X$，$x \in X$。若对任意 $\varepsilon > 0$，$B_\varepsilon(x) \cap A \neq \emptyset$ 且 $B_\varepsilon(x) \cap (X \setminus A) \neq \emptyset$，则称 $x$ 为 $A$ 的**边界点**。$A$ 的所有边界点构成的集合称为 $A$ 的**边界**，记为 $\partial A$。

### 1.2.4 闭集

**定义1.2.7（闭集）** 设 $(X, d)$ 为度量空间，$F \subset X$。若 $X \setminus F$ 是开集，则称 $F$ 为**闭集**（Closed Set）。

**定理1.2.2** 集合 $F \subset X$ 是闭集当且仅当 $\overline{F} = F$。

**证明思路**：利用闭集的定义与闭包的性质，通过补集运算证明等价性。

**定理1.2.3（闭集的性质）** 度量空间 $(X, d)$ 中的闭集具有以下性质：

1. $\emptyset$ 和 $X$ 都是闭集；
2. 任意多个闭集的交集是闭集；
3. 有限多个闭集的并集是闭集。

---

## 1.3 序列的收敛性、柯西序列、完备度量空间

### 1.3.1 序列的收敛性

**定义1.3.1（序列收敛）** 设 $(X, d)$ 为度量空间，$\{x_n\}_{n=1}^\infty \subset X$，$x \in X$。若对任意 $\varepsilon > 0$，存在正整数 $N$，使得当 $n > N$ 时，有 $d(x_n, x) < \varepsilon$，则称序列 $\{x_n\}$ **收敛**于 $x$，记为 $\lim_{n \to \infty} x_n = x$ 或 $x_n \to x \ (n \to \infty)$。

**定理1.3.1（收敛序列的性质）** 设 $(X, d)$ 为度量空间，$\{x_n\}$ 收敛于 $x$，则：

1. 极限唯一；
2. $\{x_n\}$ 是有界集；
3. $\{x_n\}$ 的任一子序列也收敛于 $x$。

**证明**：
1. 设 $x_n \to x$ 且 $x_n \to y$，则 $d(x, y) \leq d(x, x_n) + d(x_n, y) \to 0$，故 $x = y$。
2. 取 $\varepsilon = 1$，存在 $N$，当 $n > N$ 时 $d(x_n, x) < 1$，令 $r = \max\{1, d(x_1, x), \dots, d(x_N, x)\}$，则所有 $x_n \in \overline{B}_r(x)$。
3. 由定义立得。

### 1.3.2 柯西序列

**定义1.3.2（柯西序列）** 设 $(X, d)$ 为度量空间，$\{x_n\}_{n=1}^\infty \subset X$。若对任意 $\varepsilon > 0$，存在正整数 $N$，使得当 $m, n > N$ 时，有 $d(x_m, x_n) < \varepsilon$，则称 $\{x_n\}$ 为**柯西序列**（Cauchy Sequence）。

**定理1.3.2** 收敛序列必为柯西序列。

**证明**：设 $x_n \to x$，则对任意 $\varepsilon > 0$，存在 $N$，当 $n > N$ 时 $d(x_n, x) < \varepsilon/2$。故当 $m, n > N$ 时，$d(x_m, x_n) \leq d(x_m, x) + d(x, x_n) < \varepsilon/2 + \varepsilon/2 = \varepsilon$。

**注**：柯西序列不一定收敛。例如在有理数集 $\mathbb{Q}$ 中，考虑序列 $x_n = (1 + 1/n)^n$，它是柯西序列，但在 $\mathbb{Q}$ 中不收敛（其极限为 $e \notin \mathbb{Q}$）。

### 1.3.3 完备度量空间

**定义1.3.3（完备度量空间）** 若度量空间 $(X, d)$ 中的每个柯西序列都收敛于 $X$ 中的点，则称 $(X, d)$ 为**完备度量空间**（Complete Metric Space）。

**例子1.3.1**
1. $(\mathbb{R}^n, d_2)$ 是完备的（由实数完备性保证）；
2. $(C[a, b], d_\infty)$ 是完备的（一致收敛的连续函数列的极限仍连续）；
3. $(\ell^p, d_p)$（$1 \leq p \leq \infty$）是完备的。

**定理1.3.3** 完备度量空间的闭子空间也是完备的。

### 1.3.4 巴拿赫空间与希尔伯特空间简介

**定义1.3.4（赋范线性空间）** 设 $X$ 是实数域或复数域上的线性空间，若映射 $\|\cdot\|: X \rightarrow \mathbb{R}$ 满足：

1. $\|x\| \geq 0$，且 $\|x\| = 0$ 当且仅当 $x = 0$；
2. $\|\alpha x\| = |\alpha| \|x\|$（齐次性）；
3. $\|x + y\| \leq \|x\| + \|y\|$（三角不等式）。

则称 $(X, \|\cdot\|)$ 为**赋范线性空间**（Normed Linear Space），$\|\cdot\|$ 称为**范数**。

注：赋范线性空间自然成为度量空间，取 $d(x, y) = \|x - y\|$ 即可。

**定义1.3.5（巴拿赫空间）** 完备的赋范线性空间称为**巴拿赫空间**（Banach Space）。

**例子1.3.2**
1. $\mathbb{R}^n$ 以欧氏范数 $\|x\|_2 = \sqrt{\sum x_i^2}$ 构成巴拿赫空间；
2. $C[a, b]$ 以 $\|f\|_\infty = \max |f(x)|$ 构成巴拿赫空间；
3. $\ell^p$ 以 $\|x\|_p = (\sum |x_i|^p)^{1/p}$ 构成巴拿赫空间。

**定义1.3.6（内积空间）** 设 $X$ 是实数域或复数域上的线性空间，若映射 $\langle \cdot, \cdot \rangle: X \times X \rightarrow \mathbb{K}$ 满足：

1. $\langle x, y \rangle = \overline{\langle y, x \rangle}$（共轭对称性）；
2. $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$（线性性）；
3. $\langle x, x \rangle \geq 0$，且 $\langle x, x \rangle = 0$ 当且仅当 $x = 0$（正定性）。

则称 $(X, \langle \cdot, \cdot \rangle)$ 为**内积空间**（Inner Product Space）。

注：内积空间自然成为赋范线性空间，取 $\|x\| = \sqrt{\langle x, x \rangle}$ 即可。

**定义1.3.7（希尔伯特空间）** 完备的内积空间称为**希尔伯特空间**（Hilbert Space）。

**例子1.3.3**
1. $\mathbb{R}^n$ 以 $\langle x, y \rangle = \sum x_i y_i$ 构成希尔伯特空间；
2. $\ell^2$ 以 $\langle x, y \rangle = \sum x_i y_i$ 构成希尔伯特空间。

---

## 1.4 连续映射、压缩映射原理

### 1.4.1 连续映射

**定义1.4.1（连续映射）** 设 $(X, d_X)$ 和 $(Y, d_Y)$ 为两个度量空间，映射 $f: X \rightarrow Y$，$x_0 \in X$。若对任意 $\varepsilon > 0$，存在 $\delta > 0$，使得当 $d_X(x, x_0) < \delta$ 时，有 $d_Y(f(x), f(x_0)) < \varepsilon$，则称 $f$ 在 $x_0$ 处**连续**。若 $f$ 在 $X$ 中每一点都连续，则称 $f$ 是**连续映射**（Continuous Mapping）。

**定理1.4.1（连续的等价刻画）** 设 $f: X \rightarrow Y$ 为度量空间之间的映射，则以下命题等价：

1. $f$ 是连续映射；
2. 对 $Y$ 中任意开集 $G$，$f^{-1}(G)$ 是 $X$ 中的开集；
3. 对 $Y$ 中任意闭集 $F$，$f^{-1}(F)$ 是 $X$ 中的闭集；
4. 对 $X$ 中任意序列 $\{x_n\}$，若 $x_n \to x$，则 $f(x_n) \to f(x)$。

### 1.4.2 压缩映射原理

**定义1.4.2（压缩映射）** 设 $(X, d)$ 为度量空间，映射 $T: X \rightarrow X$。若存在常数 $0 \leq k < 1$，使得对任意 $x, y \in X$，有

$$d(Tx, Ty) \leq k \, d(x, y)$$

则称 $T$ 为**压缩映射**（Contraction Mapping），$k$ 称为**压缩系数**。

**定理1.4.2（压缩映射原理/不动点定理）** 设 $(X, d)$ 是完备度量空间，$T: X \rightarrow X$ 是压缩映射，则 $T$ 在 $X$ 中存在唯一的不动点，即存在唯一的 $x^* \in X$，使得 $T x^* = x^*$。

**证明**：

**存在性**：任取 $x_0 \in X$，构造迭代序列 $x_{n+1} = T x_n$，$n = 0, 1, 2, \dots$。

首先证明 $\{x_n\}$ 是柯西序列。对任意 $m > n$，有：

$$
\begin{aligned}
d(x_m, x_n) &\leq d(x_m, x_{m-1}) + d(x_{m-1}, x_{m-2}) + \dots + d(x_{n+1}, x_n) \\
&\leq k^{m-1} d(x_1, x_0) + k^{m-2} d(x_1, x_0) + \dots + k^n d(x_1, x_0) \\
&= d(x_1, x_0) \cdot \frac{k^n - k^m}{1 - k} \\
&\leq d(x_1, x_0) \cdot \frac{k^n}{1 - k}
\end{aligned}
$$

因 $0 \leq k < 1$，故 $k^n \to 0$，因此 $\{x_n\}$ 是柯西序列。由 $X$ 的完备性，存在 $x^* \in X$ 使得 $x_n \to x^*$。

由 $T$ 的连续性（压缩映射必连续），有：

$$T x^* = \lim_{n \to \infty} T x_n = \lim_{n \to \infty} x_{n+1} = x^*$$

故 $x^*$ 是不动点。

**唯一性**：设 $x^*$ 和 $x^{**}$ 都是不动点，则：

$$d(x^*, x^{**}) = d(T x^*, T x^{**}) \leq k \, d(x^*, x^{**})$$

因 $0 \leq k < 1$，故必有 $d(x^*, x^{**}) = 0$，即 $x^* = x^{**}$。

**注**：压缩映射原理不仅证明了不动点的存在唯一性，还提供了构造性的迭代求法，误差估计为 $d(x_n, x^*) \leq \frac{k^n}{1 - k} d(x_1, x_0)$。

---

## 1.5 度量空间在AI中的应用

### 1.5.1 词向量空间

在自然语言处理（NLP）中，词向量（Word Embedding）技术将词语映射为高维欧氏空间中的向量，使得语义相近的词在空间中距离较近。

**定义1.5.1（词向量空间）** 设 $V$ 为词汇表，映射 $\phi: V \rightarrow \mathbb{R}^d$ 将每个词 $w \in V$ 映射为一个 $d$ 维实向量 $\phi(w)$，称为 $w$ 的**词向量**。$\mathbb{R}^d$ 配以欧氏距离或余弦距离构成词向量空间。

**余弦相似度**：两个词向量 $u, v$ 的余弦相似度定义为：

$$\text{sim}(u, v) = \frac{\langle u, v \rangle}{\|u\| \|v\|}$$

对应的余弦距离为 $d_{\text{cos}}(u, v) = 1 - \text{sim}(u, v)$。

**例子1.5.1** Word2Vec、GloVe、BERT 等模型均通过不同方式学习词向量。例如，在训练良好的词向量空间中，常有：

$$\phi(\text{king}) - \phi(\text{man}) + \phi(\text{woman}) \approx \phi(\text{queen})$$

### 1.5.2 嵌入空间

**定义1.5.2（嵌入）** 设 $(X, d_X)$ 为原始数据空间（如图像、文本），$(Y, d_Y)$ 为低维（或结构良好的）度量空间，映射 $f: X \rightarrow Y$ 称为**嵌入**（Embedding），若 $f$ 保持某种几何或语义结构。

**例子1.5.2（图像嵌入）** 在计算机视觉中，卷积神经网络（CNN）如 ResNet 可将图像映射为特征向量，构成嵌入空间。相似图像在嵌入空间中距离较近，可用于图像检索、聚类等任务。

**例子1.5.3（流形假设）** 许多高维数据（如图像）被认为分布在低维流形上。嵌入方法（如 t-SNE、UMAP）试图将高维数据映射到低维空间，同时保持数据的局部邻域结构。

### 1.5.3 梯度流空间

在深度学习中，优化过程发生在参数空间中，梯度下降法沿着损失函数的负梯度方向更新参数，形成梯度流。

**定义1.5.3（参数空间）** 设神经网络参数为 $\theta = (\theta_1, \theta_2, \dots, \theta_n) \in \mathbb{R}^n$，$\mathbb{R}^n$ 配以欧氏度量构成**参数空间**。损失函数 $L: \mathbb{R}^n \rightarrow \mathbb{R}$ 定义在该空间上。

**梯度下降**：参数更新规则为：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

其中 $\eta > 0$ 为学习率。该迭代过程在参数空间中形成一条轨迹，称为**梯度流**。

**度量的选择**：在优化中，有时会使用非欧氏度量，如：
- 自适应优化器（Adam、RMSprop）使用对角尺度化的度量；
- 自然梯度法使用Fisher信息矩阵诱导的度量。

**不动点观点**：训练收敛时，梯度 $\nabla L(\theta^*) = 0$，此时 $\theta^*$ 是梯度下降迭代的不动点（在连续时间下是梯度流的平衡点）。

---

## 习题

1. 证明离散度量空间（$d(x, y) = 1$ 当 $x \neq y$，$d(x, x) = 0$）是完备的。
2. 设 $(X, d)$ 为度量空间，证明 $\mathring{A} = X \setminus \overline{X \setminus A}$。
3. 在 $C[0, 1]$ 中，考虑序列 $f_n(x) = x^n$，它是柯西序列吗？
4. 应用压缩映射原理证明方程 $x = \cos x$ 在 $[0, \pi/2]$ 上有唯一解。
5. 思考：在词向量空间中，为什么余弦距离比欧氏距离更常用？

# 附录A：量子符号速查表

> 常用量子计算符号及其含义。

---

## 1. 量子态与向量符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $|\psi\rangle$ | Dirac ket | 量子态 |
| $\langle \phi|$ | Dirac bra | 对偶态 |
| $\langle \phi|\psi\rangle$ | 内积 | 幅度 |
| $|\phi\rangle\langle \psi|$ | 外积 | 算符 |
| $\otimes$ | 张量积 | 复合系统 |
| $\oplus$ | 直和 | 子空间分解 |
| $\| \cdot \|$ | 范数 | 向量长度 |
| $|\psi\rangle^{\otimes n}$ | $n$ 重张量积 | $n$ 量子比特 |
| $\langle \psi|A|\psi\rangle$ | 期望值 | 可观测量 |

## 2. 矩阵与算符

| 符号 | 含义 | 说明 |
|------|------|------|
| $\dagger$ | Hermite共轭 | 共轭转置 |
| $U^\dagger U = I$ | 幺正性 | 量子门条件 |
| $[A,B]$ | 对易子 | $AB-BA$ |
| $\{A,B\}$ | 反对易子 | $AB+BA$ |
| $\text{Tr}$ | 迹 | 矩阵对角和 |
| $\det$ | 行列式 | 方阵标量值 |
| $\text{spec}(A)$ | 谱 | 特征值集合 |
| $\|\|A\|\|$ | 算子范数 | 最大奇异值 |

## 3. 特殊空间与群

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbb{C}^n$ | 复空间 | 量子态空间 |
| $\mathbb{C}^{2^n}$ | $n$ 量子比特空间 | 维数 $2^n$ |
| $U(n)$ | 酉群 | $n$ 维酉矩阵 |
| $SU(n)$ | 特殊酉群 | 行列式 $=1$ |
| $O(n)$ | 正交群 | 实正交矩阵 |
| $\mathcal{H}$ | 希尔伯特空间 | 完备内积空间 |

## 4. 量子信息与协议

| 符号 | 含义 | 说明 |
|------|------|------|
| $I$ | 量子信息 | 比特或互信息 |
| $S(\rho)$ | 冯·诺依曼熵 | $-\text{Tr}(\rho\log\rho)$ |
| $H(X)$ | 香农熵 | $-\sum p(x)\log p(x)$ |
| $I(X:Y)$ | 互信息 | 相关性度量 |
| $F(\rho,\sigma)$ | 保真度 | 态接近程度 |
| $\chi$ | Holevo量 | 信道容量上界 |
| $E_0, E_1$ | Kraus算符 | 量子操作表示 |

## 5. 量子计算模型

| 符号 | 含义 | 说明 |
|------|------|------|
| $|+\rangle$ | 叠加态 | $(|0\rangle+|1\rangle)/\sqrt{2}$ |
| $|-\rangle$ | 叠加态 | $(|0\rangle-|1\rangle)/\sqrt{2}$ |
| $|\Phi^+\rangle$ | Bell态 | $(|00\rangle+|11\rangle)/\sqrt{2}$ |
| $|\Phi^-\rangle$ | Bell态 | $(|00\rangle-|11\rangle)/\sqrt{2}$ |
| $|\Psi^+\rangle$ | Bell态 | $(|01\rangle+|10\rangle)/\sqrt{2}$ |
| $|\Psi^-\rangle$ | Bell态 | $(|01\rangle-|10\rangle)/\sqrt{2}$ |
| $|W\rangle$ | W态 | $\frac{1}{\sqrt{3}}(|001\rangle+|010\rangle+|100\rangle)$ |
| $|GHZ\rangle$ | GHZ态 | $(|00\cdots0\rangle+|11\cdots1\rangle)/\sqrt{2}$ |

## 6. 量子门符号

| 符号 | 名称 | 矩阵表示 |
|------|------|---------|
| $X$ | Pauli-X (NOT) | $\begin{bmatrix}0&1\\1&0\end{bmatrix}$ |
| $Y$ | Pauli-Y | $\begin{bmatrix}0&-i\\i&0\end{bmatrix}$ |
| $Z$ | Pauli-Z | $\begin{bmatrix}1&0\\0&-1\end{bmatrix}$ |
| $H$ | Hadamard | $\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$ |
| $S$ | 相位门 | $\begin{bmatrix}1&0\\0&i\end{bmatrix}$ |
| $T$ | $\pi/8$门 | $\begin{bmatrix}1&0\\0&e^{i\pi/4}\end{bmatrix}$ |
| CNOT | 受控非 | $\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{bmatrix}$ |
| SWAP | 交换门 | 交换两个量子比特 |

---

*最后更新：2026-06-15*

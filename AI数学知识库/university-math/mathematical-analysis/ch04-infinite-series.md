# 第4章 无穷级数

## 4.1 数项级数的基本概念

### 4.1.1 级数的定义

**定义4.1** 给定一个数列 $\{a_n\}$，表达式
$$
\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots
$$
称为**无穷级数**，简称**级数**，其中 $a_n$ 称为级数的**通项**或**一般项**。

**定义4.2** 级数 $\sum_{n=1}^{\infty} a_n$ 的前 $n$ 项之和
$$
S_n = a_1 + a_2 + \cdots + a_n = \sum_{k=1}^{n} a_k
$$
称为级数的**前 $n$ 项部分和**。数列 $\{S_n\}$ 称为级数的**部分和数列**。

### 4.1.2 收敛与发散

**定义4.3** 若级数 $\sum_{n=1}^{\infty} a_n$ 的部分和数列 $\{S_n\}$ 收敛于极限 $S$，即
$$
\lim_{n \to \infty} S_n = S
$$
则称级数**收敛**，并称 $S$ 为级数的**和**，记为
$$
S = \sum_{n=1}^{\infty} a_n
$$
若 $\{S_n\}$ 发散，则称级数**发散**。

**例4.1** 讨论几何级数（等比级数）
$$
\sum_{n=0}^{\infty} ar^n = a + ar + ar^2 + \cdots + ar^n + \cdots \quad (a \neq 0)
$$
的敛散性。

**解** 当 $r \neq 1$ 时，
$$
S_n = a + ar + ar^2 + \cdots + ar^{n-1} = a \cdot \frac{1 - r^n}{1 - r}
$$
- 若 $|r| < 1$，则 $\lim_{n \to \infty} r^n = 0$，故 $\lim_{n \to \infty} S_n = \frac{a}{1 - r}$，级数收敛，其和为 $\frac{a}{1 - r}$；
- 若 $|r| > 1$，则 $\lim_{n \to \infty} |r|^n = +\infty$，故 $\lim_{n \to \infty} S_n = \infty$，级数发散；
- 若 $r = 1$，则 $S_n = na \to \infty$（$n \to \infty$），级数发散；
- 若 $r = -1$，则 $S_n = \frac{a}{2}[1 + (-1)^{n-1}]$，极限不存在，级数发散。

综上，几何级数 $\sum_{n=0}^{\infty} ar^n$ 当且仅当 $|r| < 1$ 时收敛，其和为 $\frac{a}{1 - r}$。

### 4.1.3 级数收敛的必要条件

**定理4.1（级数收敛的必要条件）** 若级数 $\sum_{n=1}^{\infty} a_n$ 收敛，则
$$
\lim_{n \to \infty} a_n = 0
$$

**证明** 设级数的和为 $S$，即 $\lim_{n \to \infty} S_n = S$。因为 $a_n = S_n - S_{n-1}$，所以
$$
\lim_{n \to \infty} a_n = \lim_{n \to \infty} (S_n - S_{n-1}) = \lim_{n \to \infty} S_n - \lim_{n \to \infty} S_{n-1} = S - S = 0
$$

**注** 该条件只是必要条件，而非充分条件。即 $\lim_{n \to \infty} a_n = 0$ 不能推出级数收敛。

**例4.2** 调和级数
$$
\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n} + \cdots
$$
虽然 $\lim_{n \to \infty} \frac{1}{n} = 0$，但级数是发散的。

**证明** 考虑部分和
$$
S_n = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n}
$$
对于任意正整数 $k$，有
$$
S_{2^k} = 1 + \frac{1}{2} + \left( \frac{1}{3} + \frac{1}{4} \right) + \left( \frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8} \right) + \cdots + \left( \frac{1}{2^{k-1}+1} + \cdots + \frac{1}{2^k} \right)
$$
每一组的项数分别为 $1, 1, 2, 4, \cdots, 2^{k-1}$，且每组的最后一项是该组中最小的项，因此
$$
S_{2^k} > 1 + \frac{1}{2} + 2 \cdot \frac{1}{4} + 4 \cdot \frac{1}{8} + \cdots + 2^{k-1} \cdot \frac{1}{2^k} = 1 + \underbrace{\frac{1}{2} + \frac{1}{2} + \cdots + \frac{1}{2}}_{k \text{ 项}} = 1 + \frac{k}{2}
$$
当 $k \to \infty$ 时，$S_{2^k} \to \infty$，故调和级数发散。

### 4.1.4 收敛级数的基本性质

**性质4.1** 若级数 $\sum_{n=1}^{\infty} a_n$ 收敛，其和为 $S$，则对任意常数 $c$，级数 $\sum_{n=1}^{\infty} ca_n$ 也收敛，其和为 $cS$。

**性质4.2** 若级数 $\sum_{n=1}^{\infty} a_n$ 和 $\sum_{n=1}^{\infty} b_n$ 都收敛，其和分别为 $S$ 和 $T$，则级数 $\sum_{n=1}^{\infty} (a_n \pm b_n)$ 也收敛，其和为 $S \pm T$。

**性质4.3** 在级数中添加、去掉或改变有限项，不改变级数的敛散性。

**性质4.4** 收敛级数的项任意加括号后所成的新级数仍收敛，且其和不变。

**注** 性质4.4的逆命题不成立，即加括号后级数收敛，原级数未必收敛。

---

## 4.2 正项级数收敛性判别法

若级数 $\sum_{n=1}^{\infty} a_n$ 的各项都是非负的，即 $a_n \geq 0$（$n = 1, 2, 3, \cdots$），则称该级数为**正项级数**。

### 4.2.1 基本定理

**定理4.2（基本定理）** 正项级数 $\sum_{n=1}^{\infty} a_n$ 收敛的充分必要条件是其部分和数列 $\{S_n\}$ 有上界。

**证明** 由于 $a_n \geq 0$，故 $\{S_n\}$ 是单调不减数列。根据单调有界数列收敛准则，$\{S_n\}$ 收敛当且仅当它有上界。

### 4.2.2 比较判别法

**定理4.3（比较判别法）** 设 $\sum_{n=1}^{\infty} a_n$ 和 $\sum_{n=1}^{\infty} b_n$ 都是正项级数，且存在正整数 $N$，使得当 $n \geq N$ 时，有 $a_n \leq b_n$。则：
- 若 $\sum_{n=1}^{\infty} b_n$ 收敛，则 $\sum_{n=1}^{\infty} a_n$ 也收敛；
- 若 $\sum_{n=1}^{\infty} a_n$ 发散，则 $\sum_{n=1}^{\infty} b_n$ 也发散。

**定理4.4（比较判别法的极限形式）** 设 $\sum_{n=1}^{\infty} a_n$ 和 $\sum_{n=1}^{\infty} b_n$ 都是正项级数，且
$$
\lim_{n \to \infty} \frac{a_n}{b_n} = l \quad (0 \leq l \leq +\infty)
$$
则：
- 若 $0 < l < +\infty$，则 $\sum_{n=1}^{\infty} a_n$ 与 $\sum_{n=1}^{\infty} b_n$ 同时收敛或同时发散；
- 若 $l = 0$，且 $\sum_{n=1}^{\infty} b_n$ 收敛，则 $\sum_{n=1}^{\infty} a_n$ 也收敛；
- 若 $l = +\infty$，且 $\sum_{n=1}^{\infty} b_n$ 发散，则 $\sum_{n=1}^{\infty} a_n$ 也发散。

**例4.3** 讨论 $p$-级数
$$
\sum_{n=1}^{\infty} \frac{1}{n^p} = 1 + \frac{1}{2^p} + \frac{1}{3^p} + \cdots + \frac{1}{n^p} + \cdots
$$
的敛散性。

**解** 当 $p = 1$ 时，$p$-级数就是调和级数，发散。

当 $p \leq 0$ 时，$\lim_{n \to \infty} \frac{1}{n^p} \neq 0$，级数发散。

当 $p > 1$ 时，对 $n \geq 2$，有
$$
\frac{1}{n^p} < \int_{n-1}^{n} \frac{1}{x^p} dx
$$
因此
$$
S_n = 1 + \sum_{k=2}^{n} \frac{1}{k^p} < 1 + \int_{1}^{n} \frac{1}{x^p} dx = 1 + \frac{1}{p-1} \left( 1 - \frac{1}{n^{p-1}} \right) < 1 + \frac{1}{p-1}
$$
即 $S_n$ 有上界，故级数收敛。

综上，$p$-级数当且仅当 $p > 1$ 时收敛。

### 4.2.3 比值判别法（达朗贝尔判别法）

**定理4.5（比值判别法）** 设 $\sum_{n=1}^{\infty} a_n$ 是正项级数，且
$$
\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = \rho
$$
则：
- 若 $\rho < 1$，则级数收敛；
- 若 $\rho > 1$（或 $\rho = +\infty$），则级数发散；
- 若 $\rho = 1$，则判别法失效。

**例4.4** 判别级数 $\sum_{n=1}^{\infty} \frac{2^n n!}{n^n}$ 的敛散性。

**解**
$$
\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = \lim_{n \to \infty} \frac{2^{n+1} (n+1)!}{(n+1)^{n+1}} \cdot \frac{n^n}{2^n n!} = \lim_{n \to \infty} \frac{2}{\left( 1 + \frac{1}{n} \right)^n} = \frac{2}{e} < 1
$$
故级数收敛。

### 4.2.4 根值判别法（柯西判别法）

**定理4.6（根值判别法）** 设 $\sum_{n=1}^{\infty} a_n$ 是正项级数，且
$$
\lim_{n \to \infty} \sqrt[n]{a_n} = \rho
$$
则：
- 若 $\rho < 1$，则级数收敛；
- 若 $\rho > 1$（或 $\rho = +\infty$），则级数发散；
- 若 $\rho = 1$，则判别法失效。

**例4.5** 判别级数 $\sum_{n=1}^{\infty} \left( \frac{n}{2n+1} \right)^n$ 的敛散性。

**解**
$$
\lim_{n \to \infty} \sqrt[n]{a_n} = \lim_{n \to \infty} \frac{n}{2n+1} = \frac{1}{2} < 1
$$
故级数收敛。

### 4.2.5 积分判别法（柯西积分判别法）

**定理4.7（积分判别法）** 设 $f(x)$ 在 $[1, +\infty)$ 上非负、递减且连续，则正项级数 $\sum_{n=1}^{\infty} f(n)$ 与反常积分 $\int_{1}^{+\infty} f(x) dx$ 同时收敛或同时发散。

---

## 4.3 任意项级数

### 4.3.1 绝对收敛与条件收敛

**定义4.4** 设 $\sum_{n=1}^{\infty} a_n$ 是任意项级数。
- 若正项级数 $\sum_{n=1}^{\infty} |a_n|$ 收敛，则称级数 $\sum_{n=1}^{\infty} a_n$ **绝对收敛**；
- 若 $\sum_{n=1}^{\infty} |a_n|$ 发散，但 $\sum_{n=1}^{\infty} a_n$ 收敛，则称级数 $\sum_{n=1}^{\infty} a_n$ **条件收敛**。

**定理4.8** 若级数 $\sum_{n=1}^{\infty} |a_n|$ 收敛，则级数 $\sum_{n=1}^{\infty} a_n$ 必收敛。

**证明** 令 $b_n = \frac{1}{2}(|a_n| + a_n)$，$c_n = \frac{1}{2}(|a_n| - a_n)$，则 $b_n \geq 0$，$c_n \geq 0$，且 $b_n \leq |a_n|$，$c_n \leq |a_n|$。由比较判别法，$\sum b_n$ 和 $\sum c_n$ 都收敛。又 $a_n = b_n - c_n$，故 $\sum a_n$ 收敛。

### 4.3.2 交错级数与莱布尼茨判别法

**定义4.5** 形如
$$
\sum_{n=1}^{\infty} (-1)^{n-1} a_n = a_1 - a_2 + a_3 - a_4 + \cdots \quad (a_n > 0)
$$
的级数称为**交错级数**。

**定理4.9（莱布尼茨判别法）** 若交错级数 $\sum_{n=1}^{\infty} (-1)^{n-1} a_n$ 满足：
1. $\lim_{n \to \infty} a_n = 0$；
2. 数列 $\{a_n\}$ 单调递减，即 $a_n \geq a_{n+1}$（$n = 1, 2, 3, \cdots$），
则级数收敛，且其和 $S \leq a_1$，余项 $r_n = S - S_n$ 满足 $|r_n| \leq a_{n+1}$。

**例4.6** 交错调和级数
$$
\sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots
$$
是条件收敛的。

**解** 由莱布尼茨判别法，级数收敛。但 $\sum_{n=1}^{\infty} \left| \frac{(-1)^{n-1}}{n} \right| = \sum \frac{1}{n}$ 发散，故原级数条件收敛。

---

## 4.4 函数项级数与一致收敛性

### 4.4.1 函数项级数的概念

**定义4.6** 设 $\{u_n(x)\}$ 是定义在区间 $I$ 上的函数列，则表达式
$$
\sum_{n=1}^{\infty} u_n(x) = u_1(x) + u_2(x) + u_3(x) + \cdots
$$
称为定义在 $I$ 上的**函数项级数**。

对每一个固定的 $x_0 \in I$，若数项级数 $\sum u_n(x_0)$ 收敛，则称 $x_0$ 为函数项级数的**收敛点**；否则称为**发散点**。收敛点的全体称为**收敛域**。

在收敛域上，函数项级数的和是 $x$ 的函数，记为 $S(x)$，称为**和函数**。

### 4.4.2 一致收敛性

**定义4.7** 设函数项级数 $\sum_{n=1}^{\infty} u_n(x)$ 在区间 $I$ 上的和函数为 $S(x)$，部分和为 $S_n(x)$。若对任意 $\varepsilon > 0$，存在正整数 $N$，使得当 $n > N$ 时，对所有 $x \in I$，都有
$$
|S(x) - S_n(x)| < \varepsilon
$$
则称函数项级数 $\sum_{n=1}^{\infty} u_n(x)$ 在 $I$ 上**一致收敛**于 $S(x)$。

### 4.4.3 魏尔斯特拉斯判别法（M判别法）

**定理4.10（魏尔斯特拉斯判别法）** 若函数项级数 $\sum_{n=1}^{\infty} u_n(x)$ 在区间 $I$ 上满足：
1. 存在正项级数 $\sum_{n=1}^{\infty} M_n$，使得对所有 $x \in I$ 和 $n \in \mathbb{N}^*$，有 $|u_n(x)| \leq M_n$；
2. 正项级数 $\sum_{n=1}^{\infty} M_n$ 收敛，
则函数项级数 $\sum_{n=1}^{\infty} u_n(x)$ 在 $I$ 上一致收敛。

### 4.4.4 一致收敛级数的性质

**定理4.11** 若函数项级数 $\sum u_n(x)$ 在 $[a, b]$ 上一致收敛于 $S(x)$，且每一项 $u_n(x)$ 都在 $[a, b]$ 上连续，则：
1. 和函数 $S(x)$ 在 $[a, b]$ 上连续；
2. 级数可逐项积分：
$$
\int_{a}^{b} S(x) dx = \sum_{n=1}^{\infty} \int_{a}^{b} u_n(x) dx
$$

**定理4.12** 若函数项级数 $\sum u_n(x)$ 在 $[a, b]$ 上收敛于 $S(x)$，每一项 $u_n(x)$ 都有连续导数，且导数级数 $\sum u_n'(x)$ 在 $[a, b]$ 上一致收敛，则和函数 $S(x)$ 可导，且
$$
S'(x) = \sum_{n=1}^{\infty} u_n'(x)
$$

---

## 4.5 幂级数

### 4.5.1 幂级数的概念

**定义4.8** 形如
$$
\sum_{n=0}^{\infty} a_n (x - x_0)^n = a_0 + a_1 (x - x_0) + a_2 (x - x_0)^2 + \cdots
$$
的函数项级数称为**幂级数**，其中 $a_0, a_1, a_2, \cdots$ 称为**幂级数的系数**。

当 $x_0 = 0$ 时，幂级数简化为
$$
\sum_{n=0}^{\infty} a_n x^n = a_0 + a_1 x + a_2 x^2 + \cdots
$$

### 4.5.2 收敛半径与收敛区间

**定理4.13（阿贝尔定理）** 若幂级数 $\sum_{n=0}^{\infty} a_n x^n$ 在 $x = x_0 \neq 0$ 处收敛，则对满足 $|x| < |x_0|$ 的一切 $x$，幂级数绝对收敛；若幂级数在 $x = x_0$ 处发散，则对满足 $|x| > |x_0|$ 的一切 $x$，幂级数发散。

**定义4.9** 若存在正数 $R$，使得幂级数 $\sum a_n x^n$ 当 $|x| < R$ 时绝对收敛，当 $|x| > R$ 时发散，则称 $R$ 为幂级数的**收敛半径**。开区间 $(-R, R)$ 称为**收敛区间**。

**定理4.14** 设幂级数 $\sum_{n=0}^{\infty} a_n x^n$ 的系数满足
$$
\lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right| = \rho \quad \text{或} \quad \lim_{n \to \infty} \sqrt[n]{|a_n|} = \rho
$$
则收敛半径
$$
R = \begin{cases}
\frac{1}{\rho}, & 0 < \rho < +\infty, \\
+\infty, & \rho = 0, \\
0, & \rho = +\infty.
\end{cases}
$$

### 4.5.3 幂级数的运算性质

设幂级数 $\sum a_n x^n$ 和 $\sum b_n x^n$ 的收敛半径分别为 $R_1$ 和 $R_2$，记 $R = \min\{R_1, R_2\}$，则在 $(-R, R)$ 内：
1. $\sum a_n x^n \pm \sum b_n x^n = \sum (a_n \pm b_n) x^n$
2. $\left( \sum a_n x^n \right) \left( \sum b_n x^n \right) = \sum_{n=0}^{\infty} \left( \sum_{k=0}^{n} a_k b_{n-k} \right) x^n$

**定理4.15** 设幂级数 $\sum_{n=0}^{\infty} a_n x^n$ 的收敛半径为 $R > 0$，和函数为 $S(x)$，则：
1. $S(x)$ 在 $(-R, R)$ 内连续；
2. $S(x)$ 在 $(-R, R)$ 内可导，且可逐项求导：
$$
S'(x) = \sum_{n=1}^{\infty} n a_n x^{n-1}
$$
逐项求导后所得幂级数的收敛半径仍为 $R$；
3. $S(x)$ 在 $(-R, R)$ 内可积，且可逐项积分：
$$
\int_{0}^{x} S(t) dt = \sum_{n=0}^{\infty} \frac{a_n}{n+1} x^{n+1}
$$
逐项积分后所得幂级数的收敛半径仍为 $R$。

### 4.5.4 泰勒级数与麦克劳林级数

**定义4.10** 设函数 $f(x)$ 在点 $x_0$ 的某邻域内具有任意阶导数，则幂级数
$$
\sum_{n=0}^{\infty} \frac{f^{(n)}(x_0)}{n!} (x - x_0)^n
$$
称为函数 $f(x)$ 在点 $x_0$ 处的**泰勒级数**。

当 $x_0 = 0$ 时，泰勒级数称为**麦克劳林级数**：
$$
\sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} x^n
$$

**定理4.16** 设函数 $f(x)$ 在点 $x_0$ 的某邻域 $U(x_0)$ 内具有任意阶导数，则 $f(x)$ 在该邻域内能展开成泰勒级数的充分必要条件是：在该邻域内，$f(x)$ 的泰勒公式的余项 $R_n(x)$ 满足
$$
\lim_{n \to \infty} R_n(x) = 0, \quad \forall x \in U(x_0)
$$

### 4.5.5 常见函数的麦克劳林展开式

1. $e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}, \quad x \in (-\infty, +\infty)$
2. $\sin x = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1}, \quad x \in (-\infty, +\infty)$
3. $\cos x = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!} x^{2n}, \quad x \in (-\infty, +\infty)$
4. $\ln(1+x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{n} x^n, \quad x \in (-1, 1]$
5. $(1+x)^\alpha = \sum_{n=0}^{\infty} \binom{\alpha}{n} x^n, \quad x \in (-1, 1)$，其中 $\binom{\alpha}{n} = \frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}$
6. $\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n, \quad x \in (-1, 1)$

---

## 4.6 傅里叶级数简介

### 4.6.1 三角级数与三角函数系

**定义4.11** 形如
$$
\frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)
$$
的级数称为**三角级数**，其中 $a_0, a_n, b_n$（$n = 1, 2, 3, \cdots$）为常数。

三角函数系
$$
1, \cos x, \sin x, \cos 2x, \sin 2x, \cdots, \cos nx, \sin nx, \cdots
$$
在区间 $[-\pi, \pi]$ 上是**正交**的，即任意两个不同函数的乘积在 $[-\pi, \pi]$ 上的积分等于零。

### 4.6.2 傅里叶系数与傅里叶级数

**定义4.12** 设 $f(x)$ 是周期为 $2\pi$ 的函数，且在 $[-\pi, \pi]$ 上可积，则
$$
\begin{cases}
a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos nx dx, & n = 0, 1, 2, \cdots, \\
b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin nx dx, & n = 1, 2, 3, \cdots
\end{cases}
$$
称为 $f(x)$ 的**傅里叶系数**，由这些系数构成的三角级数称为 $f(x)$ 的**傅里叶级数**，记为
$$
f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)
$$

### 4.6.3 收敛定理（狄利克雷定理）

**定理4.17（狄利克雷定理）** 设 $f(x)$ 是周期为 $2\pi$ 的函数，且在 $[-\pi, \pi]$ 上满足：
1. 连续或只有有限个第一类间断点；
2. 只有有限个极值点，
则 $f(x)$ 的傅里叶级数收敛，且：
- 在 $f(x)$ 的连续点 $x$ 处，级数收敛于 $f(x)$；
- 在 $f(x)$ 的间断点 $x$ 处，级数收敛于 $\frac{f(x-0) + f(x+0)}{2}$。

---

## 4.7 级数在AI中的应用

### 4.7.1 泰勒展开在优化中的应用

在机器学习和深度学习中，优化算法（如梯度下降、牛顿法）广泛使用泰勒展开来近似目标函数。

设 $f(x)$ 是可微函数，在点 $x_k$ 处的泰勒展开：

**一阶泰勒展开（线性近似）**：
$$
f(x) \approx f(x_k) + \nabla f(x_k)^T (x - x_k)
$$
用于梯度下降法，沿着负梯度方向搜索最小值。

**二阶泰勒展开（二次近似）**：
$$
f(x) \approx f(x_k) + \nabla f(x_k)^T (x - x_k) + \frac{1}{2} (x - x_k)^T \nabla^2 f(x_k) (x - x_k)
$$
用于牛顿法，利用海森矩阵信息加速收敛。

### 4.7.2 注意力机制的级数表达

Transformer模型中的自注意力机制可以用级数形式理解。注意力权重的计算为：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

softmax函数可展开为指数级数：
$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} = \frac{1 + z_i + \frac{z_i^2}{2!} + \cdots}{\sum_{j=1}^{n} \left( 1 + z_j + \frac{z_j^2}{2!} + \cdots \right)}
$$

此外，注意力机制可以看作是一种加权和，即：
$$
\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} v_j
$$
其中 $\alpha_{ij}$ 是注意力权重，这本质上是一个级数形式的加权求和。

### 4.7.3 神经网络激活函数的级数展开

许多激活函数可以表示为幂级数展开：

**Sigmoid函数**：
$$
\sigma(x) = \frac{1}{1 + e^{-x}} = \sum_{n=0}^{\infty} (-1)^n e^{-nx}, \quad x > 0
$$

**双曲正切函数**：
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2 \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1} x^{2n+1}, \quad |x| < \frac{\pi}{2}
$$

**ReLU函数的平滑近似**（如Softplus）：
$$
\text{softplus}(x) = \ln(1 + e^x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{n} e^{-(n-1)x}, \quad x > 0
$$

### 4.7.4 生成模型中的级数应用

在生成模型中，级数展开可用于概率分布的近似和采样。例如，变分推断中使用的指数族分布的矩可以通过级数展开计算。

此外，傅里叶级数在信号处理、图像处理以及最近的NeRF（神经辐射场）等视觉任务中有着重要应用，用于表示复杂的函数和数据分布。

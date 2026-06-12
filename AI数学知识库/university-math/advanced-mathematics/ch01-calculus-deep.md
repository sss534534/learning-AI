# 第1章 一元微积分深化


## 元数据
- **难度**: ⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


## 1.1 高阶导数与莱布尼茨公式

### 1.1.1 高阶导数的定义

**定义1（高阶导数）**：设函数 $y = f(x)$ 在区间 $I$ 上可导，若其导函数 $f'(x)$ 在点 $x_0 \in I$ 处也可导，则称 $f'(x)$ 在 $x_0$ 处的导数为 $f(x)$ 在 $x_0$ 处的**二阶导数**，记作 $f''(x_0)$，即
$$
f''(x_0) = \lim_{\Delta x \to 0} \frac{f'(x_0 + \Delta x) - f'(x_0)}{\Delta x}
$$

类似地，若二阶导函数 $f''(x)$ 可导，则称其导数为**三阶导数**，记作 $f'''(x)$。一般地，$n-1$ 阶导函数的导数称为**n阶导数**，记作 $f^{(n)}(x)$。

**记号约定**：
- 二阶导数：$f''(x),\ y'',\ \frac{d^2y}{dx^2},\ \frac{d^2f}{dx^2}$
- n阶导数：$f^{(n)}(x),\ y^{(n)},\ \frac{d^ny}{dx^n},\ \frac{d^nf}{dx^n}$

### 1.1.2 常见函数的高阶导数公式

**定理1（幂函数的高阶导数）**：设 $f(x) = x^\alpha$，其中 $\alpha$ 为实数，则
$$
f^{(n)}(x) = \alpha(\alpha - 1)\cdots(\alpha - n + 1)x^{\alpha - n}
$$
特别地，当 $\alpha = k$ 为正整数时，
$$
(x^k)^{(n)} =
\begin{cases}
k(k-1)\cdots(k-n+1)x^{k-n}, & n \leq k \\\\
k!, & n = k \\\\
0, & n > k
\end{cases}
$$

**定理2（指数函数的高阶导数）**：
$$
(e^x)^{(n)} = e^x, \quad (a^x)^{(n)} = a^x (\ln a)^n
$$

**定理3（三角函数的高阶导数）**：
$$
(\sin x)^{(n)} = \sin\left(x + \frac{n\pi}{2}\right), \quad (\cos x)^{(n)} = \cos\left(x + \frac{n\pi}{2}\right)
$$

**定理4（对数函数的高阶导数）**：
$$
(\ln x)^{(n)} = (-1)^{n-1} \frac{(n-1)!}{x^n}, \quad \left(\frac{1}{x}\right)^{(n)} = (-1)^n \frac{n!}{x^{n+1}}
$$

### 1.1.3 莱布尼茨公式

**定理5（莱布尼茨公式）**：设函数 $u(x)$ 和 $v(x)$ 都具有n阶导数，则它们的乘积也具有n阶导数，且
$$
(uv)^{(n)} = \sum_{k=0}^n \binom{n}{k} u^{(k)}(x) v^{(n-k)}(x)
$$
其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 为二项式系数，约定 $u^{(0)} = u$，$v^{(0)} = v$。

**例1**：求 $y = x^2 e^{2x}$ 的5阶导数。

解：设 $u = e^{2x}$，$v = x^2$，则
$$
u^{(k)} = 2^k e^{2x}, \quad v' = 2x, \quad v'' = 2, \quad v^{(k)} = 0 \ (k \geq 3)
$$

由莱布尼茨公式：
$$
\begin{aligned}
y^{(5)} &= \sum_{k=0}^5 \binom{5}{k} u^{(k)} v^{(5-k)} \\\\
&= \binom{5}{0} u v^{(5)} + \binom{5}{1} u' v^{(4)} + \binom{5}{2} u'' v'' + \binom{5}{3} u''' v' + \binom{5}{4} u^{(4)} v + \binom{5}{5} u^{(5)} v^{(0)} \\\\
&= 10 \cdot 2^2 e^{2x} \cdot 2 + 10 \cdot 2^3 e^{2x} \cdot 2x + 5 \cdot 2^4 e^{2x} \cdot x^2 + 1 \cdot 2^5 e^{2x} \cdot x^2 \\\\
&= e^{2x}(32x^2 + 160x + 80)
\end{aligned}
$$

## 1.2 泰勒公式与麦克劳林展开

### 1.2.1 泰勒多项式

**定义2（泰勒多项式）**：设函数 $f(x)$ 在点 $x_0$ 处具有n阶导数，称多项式
$$
P_n(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n
$$
为 $f(x)$ 在点 $x_0$ 处的**n阶泰勒多项式**。

### 1.2.2 泰勒公式

**定理6（泰勒公式 - 皮亚诺余项）**：设函数 $f(x)$ 在点 $x_0$ 处具有n阶导数，则
$$
f(x) = P_n(x) + R_n(x)
$$
其中
$$
R_n(x) = o((x - x_0)^n) \quad (x \to x_0)
$$
$R_n(x)$ 称为**皮亚诺余项**。

**定理7（泰勒公式 - 拉格朗日余项）**：设函数 $f(x)$ 在包含 $x_0$ 的区间 $I$ 上具有 $n+1$ 阶导数，则对任意 $x \in I$，有
$$
f(x) = P_n(x) + R_n(x)
$$
其中
$$
R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - x_0)^{n+1}
$$
$\xi$ 介于 $x_0$ 与 $x$ 之间。$R_n(x)$ 称为**拉格朗日余项**。

### 1.2.3 麦克劳林展开

当 $x_0 = 0$ 时，泰勒公式称为**麦克劳林公式**：
$$
f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \cdots + \frac{f^{(n)}(0)}{n!}x^n + R_n(x)
$$

## 1.3 常见函数的泰勒展开式

### 1.3.1 基本初等函数的麦克劳林展开

**1. 指数函数**：
$$
e^x = 1 + x + \frac{x^2}{2!} + \cdots + \frac{x^n}{n!} + o(x^n)
$$
拉格朗日余项：$R_n(x) = \frac{e^\xi}{(n+1)!}x^{n+1}$，$\xi$ 介于0与x之间。

**2. 正弦函数**：
$$
\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots + (-1)^m \frac{x^{2m+1}}{(2m+1)!} + o(x^{2m+2})
$$
拉格朗日余项：$R_n(x) = \frac{\sin\left(\xi + \frac{(2m+2)\pi}{2}\right)}{(2m+2)!}x^{2m+2}$

**3. 余弦函数**：
$$
\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots + (-1)^m \frac{x^{2m}}{(2m)!} + o(x^{2m+1})
$$

**4. 对数函数**：
$$
\ln(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots + (-1)^{n-1} \frac{x^n}{n} + o(x^n)
$$

**5. 幂函数**：
$$
(1 + x)^\alpha = 1 + \alpha x + \frac{\alpha(\alpha - 1)}{2!}x^2 + \cdots + \frac{\alpha(\alpha - 1)\cdots(\alpha - n + 1)}{n!}x^n + o(x^n)
$$

**例2**：求 $\lim_{x \to 0} \frac{e^x - 1 - x - \frac{x^2}{2}}{x^3}$。

解：使用泰勒展开，$e^x = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + o(x^3)$，因此
$$
e^x - 1 - x - \frac{x^2}{2} = \frac{x^3}{6} + o(x^3)
$$
故
$$
\lim_{x \to 0} \frac{e^x - 1 - x - \frac{x^2}{2}}{x^3} = \lim_{x \to 0} \frac{\frac{x^3}{6} + o(x^3)}{x^3} = \frac{1}{6}
$$

## 1.4 反常积分

### 1.4.1 无穷限反常积分

**定义3（无穷限反常积分）**：

1. 设函数 $f(x)$ 在 $[a, +\infty)$ 上连续，定义
$$
\int_a^{+\infty} f(x) dx = \lim_{b \to +\infty} \int_a^b f(x) dx
$$
若极限存在，则称反常积分**收敛**，否则称**发散**。

2. 设函数 $f(x)$ 在 $(-\infty, b]$ 上连续，定义
$$
\int_{-\infty}^b f(x) dx = \lim_{a \to -\infty} \int_a^b f(x) dx
$$

3. 设函数 $f(x)$ 在 $(-\infty, +\infty)$ 上连续，定义
$$
\int_{-\infty}^{+\infty} f(x) dx = \int_{-\infty}^c f(x) dx + \int_c^{+\infty} f(x) dx
$$
其中 $c$ 为任意实数，当且仅当右边两个反常积分都收敛时，左边的反常积分收敛。

**例3**：讨论 $\int_1^{+\infty} \frac{1}{x^p} dx$ 的敛散性。

解：当 $p \neq 1$ 时，
$$
\int_1^{+\infty} \frac{1}{x^p} dx = \lim_{b \to +\infty} \left. \frac{x^{1-p}}{1-p} \right|_1^b = \lim_{b \to +\infty} \frac{b^{1-p} - 1}{1-p}
$$
- 若 $p > 1$，则 $1-p < 0$，$\lim_{b \to +\infty} b^{1-p} = 0$，积分收敛于 $\frac{1}{p-1}$。
- 若 $p < 1$，则 $1-p > 0$，$\lim_{b \to +\infty} b^{1-p} = +\infty$，积分发散。

当 $p = 1$ 时，
$$
\int_1^{+\infty} \frac{1}{x} dx = \lim_{b \to +\infty} \ln b = +\infty
$$
积分发散。

综上，当 $p > 1$ 时收敛，当 $p \leq 1$ 时发散。

### 1.4.2 无界函数反常积分

**定义4（瑕积分）**：设函数 $f(x)$ 在 $(a, b]$ 上连续，且在 $x = a$ 附近无界（$a$ 为**瑕点**），定义
$$
\int_a^b f(x) dx = \lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f(x) dx
$$
若极限存在，则称反常积分**收敛**，否则称**发散**。

类似地，若 $b$ 为瑕点，则
$$
\int_a^b f(x) dx = \lim_{\varepsilon \to 0^+} \int_a^{b-\varepsilon} f(x) dx
$$

**例4**：讨论 $\int_0^1 \frac{1}{x^q} dx$ 的敛散性。

解：当 $q \neq 1$ 时，
$$
\int_0^1 \frac{1}{x^q} dx = \lim_{\varepsilon \to 0^+} \left. \frac{x^{1-q}}{1-q} \right|_\varepsilon^1 = \lim_{\varepsilon \to 0^+} \frac{1 - \varepsilon^{1-q}}{1-q}
$$
- 若 $q < 1$，则 $1-q > 0$，$\lim_{\varepsilon \to 0^+} \varepsilon^{1-q} = 0$，积分收敛于 $\frac{1}{1-q}$。
- 若 $q > 1$，则 $1-q < 0$，$\lim_{\varepsilon \to 0^+} \varepsilon^{1-q} = +\infty$，积分发散。

当 $q = 1$ 时，
$$
\int_0^1 \frac{1}{x} dx = \lim_{\varepsilon \to 0^+} (-\ln \varepsilon) = +\infty
$$
积分发散。

综上，当 $q < 1$ 时收敛，当 $q \geq 1$ 时发散。

## 1.5 伽马函数与贝塔函数

### 1.5.1 伽马函数

**定义5（伽马函数）**：
$$
\Gamma(s) = \int_0^{+\infty} x^{s-1} e^{-x} dx, \quad s > 0
$$

**定理8（伽马函数的性质）**：

1. **递推公式**：$\Gamma(s+1) = s\Gamma(s)$
   证明：
   $$
   \begin{aligned}
   \Gamma(s+1) &= \int_0^{+\infty} x^s e^{-x} dx = -\int_0^{+\infty} x^s d(e^{-x}) \\\\
   &= -x^s e^{-x} \bigg|_0^{+\infty} + s \int_0^{+\infty} x^{s-1} e^{-x} dx = s\Gamma(s)
   \end{aligned}
   $$

2. **特殊值**：$\Gamma(1) = 1$，因此对正整数 $n$，$\Gamma(n+1) = n!$

3. **余元公式**：$\Gamma(s)\Gamma(1-s) = \frac{\pi}{\sin \pi s}$，特别地 $\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}$

### 1.5.2 贝塔函数

**定义6（贝塔函数）**：
$$
B(p, q) = \int_0^1 x^{p-1} (1-x)^{q-1} dx, \quad p > 0, q > 0
$$

**定理9（贝塔函数与伽马函数的关系）**：
$$
B(p, q) = \frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}
$$

**定理10（贝塔函数的性质）**：

1. 对称性：$B(p, q) = B(q, p)$
2. 递推公式：$B(p+1, q) = \frac{p}{p+q} B(p, q)$

**例5**：计算 $\int_0^{+\infty} x^2 e^{-x} dx$。

解：由伽马函数的定义和性质，
$$
\int_0^{+\infty} x^2 e^{-x} dx = \Gamma(3) = 2! = 2
$$

## 1.6 反常积分敛散性判别法

### 1.6.1 比较判别法

**定理11（比较判别法 - 无穷限）**：设函数 $f(x), g(x)$ 在 $[a, +\infty)$ 上非负连续，且 $0 \leq f(x) \leq g(x)$，则
- 若 $\int_a^{+\infty} g(x) dx$ 收敛，则 $\int_a^{+\infty} f(x) dx$ 收敛；
- 若 $\int_a^{+\infty} f(x) dx$ 发散，则 $\int_a^{+\infty} g(x) dx$ 发散。

**定理12（极限形式的比较判别法）**：设 $f(x), g(x)$ 在 $[a, +\infty)$ 上非负连续，且
$$
\lim_{x \to +\infty} \frac{f(x)}{g(x)} = l
$$
则
- 当 $0 < l < +\infty$ 时，$\int_a^{+\infty} f(x) dx$ 与 $\int_a^{+\infty} g(x) dx$ 同敛散；
- 当 $l = 0$ 时，若 $\int_a^{+\infty} g(x) dx$ 收敛，则 $\int_a^{+\infty} f(x) dx$ 收敛；
- 当 $l = +\infty$ 时，若 $\int_a^{+\infty} g(x) dx$ 发散，则 $\int_a^{+\infty} f(x) dx$ 发散。

### 1.6.2 狄利克雷判别法

**定理13（狄利克雷判别法）**：设函数 $f(x), g(x)$ 在 $[a, +\infty)$ 上满足：
1. $F(A) = \int_a^A f(x) dx$ 在 $[a, +\infty)$ 上有界；
2. $g(x)$ 在 $[a, +\infty)$ 上单调且 $\lim_{x \to +\infty} g(x) = 0$，

则反常积分 $\int_a^{+\infty} f(x)g(x) dx$ 收敛。

### 1.6.3 阿贝尔判别法

**定理14（阿贝尔判别法）**：设函数 $f(x), g(x)$ 在 $[a, +\infty)$ 上满足：
1. $\int_a^{+\infty} f(x) dx$ 收敛；
2. $g(x)$ 在 $[a, +\infty)$ 上单调有界，

则反常积分 $\int_a^{+\infty} f(x)g(x) dx$ 收敛。

**例6**：证明 $\int_1^{+\infty} \frac{\sin x}{x} dx$ 收敛。

证明：令 $f(x) = \sin x$，$g(x) = \frac{1}{x}$，则
- $F(A) = \int_1^A \sin x dx = \cos 1 - \cos A$，故 $|F(A)| \leq 2$，有界；
- $g(x) = \frac{1}{x}$ 单调递减且 $\lim_{x \to +\infty} g(x) = 0$。

由狄利克雷判别法，$\int_1^{+\infty} \frac{\sin x}{x} dx$ 收敛。

## 1.7 一元微积分深化在AI中的应用

### 1.7.1 泰勒展开近似

在机器学习和深度学习中，泰勒展开是优化算法（如梯度下降、牛顿法）的理论基础。

**牛顿法**：利用二阶泰勒展开近似目标函数。设目标函数为 $f(x)$，在当前点 $x_k$ 处进行二阶泰勒展开：
$$
f(x) \approx f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2
$$
令导数为零，求得下一个迭代点：
$$
x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}
$$

**例7（逻辑回归中的牛顿法）**：逻辑回归的损失函数为
$$
J(\theta) = -\sum_{i=1}^m [y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]
$$
其中 $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$。

利用牛顿法求解时，迭代公式为
$$
\theta_{k+1} = \theta_k - H^{-1} \nabla J(\theta_k)
$$
其中 $H$ 为海森矩阵。

### 1.7.2 数值积分

在机器学习中，经常需要计算积分，特别是在贝叶斯方法中。当解析积分不可行时，使用数值积分方法。

**梯形法**：将区间 $[a, b]$ 等分为 $n$ 份，分点为 $x_i = a + i\Delta x$，$\Delta x = \frac{b-a}{n}$，则
$$
\int_a^b f(x) dx \approx \frac{\Delta x}{2} \left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]
$$

**辛普森法**：
$$
\int_a^b f(x) dx \approx \frac{\Delta x}{3} \left[ f(x_0) + 4\sum_{i=1,3,\cdots}^{n-1} f(x_i) + 2\sum_{i=2,4,\cdots}^{n-2} f(x_i) + f(x_n) \right]
$$
要求 $n$ 为偶数。

**例8（高斯积分）**：在概率图模型中，经常需要计算高斯分布的积分。设 $X \sim N(\mu, \sigma^2)$，则
$$
E[X^2] = \int_{-\infty}^{+\infty} x^2 \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = \mu^2 + \sigma^2
$$
这个结果可以通过变量代换和伽马函数性质验证。

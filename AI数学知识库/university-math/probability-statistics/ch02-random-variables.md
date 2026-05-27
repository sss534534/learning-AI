# 第二章 随机变量与概率分布

## 2.1 随机变量的定义、分布函数、离散型与连续型随机变量

### 2.1.1 随机变量的定义

**定义 2.1（随机变量）**：设随机试验的样本空间为 $\Omega = \{\omega\}$，若对于每一个样本点 $\omega \in \Omega$，都有唯一的实数 $X(\omega)$ 与之对应，则称定义在 $\Omega$ 上的实值函数 $X = X(\omega)$ 为随机变量。

随机变量通常用大写字母 $X, Y, Z, \cdots$ 表示，其可能取值用小写字母 $x, y, z, \cdots$ 表示。

**示例 2.1**：
- 抛一枚硬币，定义 $X$ 为：$X(H) = 1$，$X(T) = 0$，则 $X$ 是随机变量
- 掷一颗骰子，定义 $X$ 为出现的点数，则 $X \in \{1, 2, 3, 4, 5, 6\}$
- 记录某电话交换台一分钟内接到的呼唤次数，定义 $X$ 为呼唤次数，则 $X \in \{0, 1, 2, \cdots\}$

### 2.1.2 分布函数

**定义 2.2（分布函数）**：设 $X$ 是一个随机变量，$x$ 是任意实数，函数
$$F(x) = P(X \leq x), \quad x \in \mathbb{R}$$
称为 $X$ 的分布函数。

**分布函数的性质**：
1. **单调性**：若 $x_1 < x_2$，则 $F(x_1) \leq F(x_2)$
2. **有界性**：$0 \leq F(x) \leq 1$，且
   $$F(-\infty) = \lim_{x \to -\infty} F(x) = 0, \quad F(+\infty) = \lim_{x \to +\infty} F(x) = 1$$
3. **右连续性**：$F(x + 0) = F(x)$

**利用分布函数计算概率**：
- $P(a < X \leq b) = F(b) - F(a)$
- $P(X > a) = 1 - F(a)$
- $P(X = a) = F(a) - F(a - 0)$

### 2.1.3 离散型随机变量

**定义 2.3（离散型随机变量）**：若随机变量 $X$ 的所有可能取值只有有限个或可列个，则称 $X$ 为离散型随机变量。

**定义 2.4（概率质量函数）**：设离散型随机变量 $X$ 的所有可能取值为 $x_k$（$k = 1, 2, \cdots$），称
$$P(X = x_k) = p_k, \quad k = 1, 2, \cdots$$
为 $X$ 的概率质量函数（PMF），或称为分布律。

**分布律的性质**：
1. $p_k \geq 0, \quad k = 1, 2, \cdots$
2. $\sum_{k=1}^{\infty} p_k = 1$

**示例 2.2**：掷一颗骰子，$X$ 为出现的点数，其分布律为
$$P(X = k) = \frac{1}{6}, \quad k = 1, 2, 3, 4, 5, 6$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 离散型随机变量：骰子点数
x = np.arange(1, 7)
p = np.ones(6) / 6

# 绘制概率质量函数
plt.stem(x, p, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title('Probability Mass Function of Dice Roll')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.1.4 连续型随机变量

**定义 2.5（连续型随机变量）**：设随机变量 $X$ 的分布函数为 $F(x)$，若存在非负可积函数 $f(x)$，使得对任意实数 $x$，有
$$F(x) = \int_{-\infty}^{x} f(t) dt$$
则称 $X$ 为连续型随机变量，$f(x)$ 称为 $X$ 的概率密度函数（PDF）。

**概率密度函数的性质**：
1. $f(x) \geq 0$
2. $\int_{-\infty}^{+\infty} f(x) dx = 1$

**利用概率密度函数计算概率**：
- $P(a < X \leq b) = \int_{a}^{b} f(x) dx$
- 对于任意实数 $a$，$P(X = a) = 0$

**分布函数与密度函数的关系**：
- 若 $f(x)$ 在点 $x$ 处连续，则 $F'(x) = f(x)$

**示例 2.3**：设 $X$ 在区间 $[0, 1]$ 上均匀分布，其概率密度函数为
$$f(x) = \begin{cases} 1, & 0 \leq x \leq 1 \\ 0, & \text{其他} \end{cases}$$
则
$$P(0.2 < X \leq 0.5) = \int_{0.2}^{0.5} 1 dx = 0.3$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# 连续型随机变量：[0,1]上的均匀分布
x = np.linspace(-0.5, 1.5, 1000)
pdf = uniform.pdf(x, loc=0, scale=1)
cdf = uniform.cdf(x, loc=0, scale=1)

# 绘制概率密度函数和分布函数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('f(x)')
ax1.set_title('Probability Density Function')
ax1.grid(True, alpha=0.3)

ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.set_xlabel('X')
ax2.set_ylabel('F(x)')
ax2.set_title('Cumulative Distribution Function')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2.2 常见离散分布

### 2.2.1 0-1分布

**定义 2.6（0-1分布）**：若随机变量 $X$ 的分布律为
$$P(X = k) = p^k (1 - p)^{1 - k}, \quad k = 0, 1$$
其中 $0 < p < 1$，则称 $X$ 服从参数为 $p$ 的0-1分布（或伯努利分布），记为 $X \sim B(1, p)$。

**概率质量函数**：
$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

**期望与方差**：
$$E(X) = p, \quad D(X) = p(1 - p)$$

**示例 2.4**：抛一枚硬币，设 $X = 1$ 表示正面，$X = 0$ 表示反面，$p = 0.5$，则 $X$ 服从0-1分布。

**代码示例**：
```python
import numpy as np
from scipy.stats import bernoulli

# 0-1分布参数
p = 0.5

# 概率质量函数
x = [0, 1]
pmf = bernoulli.pmf(x, p)
print(f"P(X=0) = {pmf[0]:.4f}, P(X=1) = {pmf[1]:.4f}")

# 期望和方差
mean, var = bernoulli.stats(p, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")

# 随机样本
samples = bernoulli.rvs(p, size=1000)
print(f"样本均值: {np.mean(samples):.4f}")
print(f"样本方差: {np.var(samples):.4f}")
```

### 2.2.2 二项分布

**定义 2.7（二项分布）**：若随机变量 $X$ 的分布律为
$$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k = 0, 1, 2, \cdots, n$$
其中 $n$ 为正整数，$0 < p < 1$，则称 $X$ 服从参数为 $(n, p)$ 的二项分布，记为 $X \sim B(n, p)$。

**二项分布的背景**：n重伯努利试验中事件A发生的次数服从二项分布。

**期望与方差**：
$$E(X) = np, \quad D(X) = np(1 - p)$$

**示例 2.5**：某人进行射击，每次命中率为0.6，独立射击5次，求恰好命中3次的概率。

设命中次数为 $X$，则 $X \sim B(5, 0.6)$，
$$P(X = 3) = \binom{5}{3} (0.6)^3 (0.4)^2 = 10 \times 0.216 \times 0.16 = 0.3456$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 二项分布参数
n = 10
p = 0.5

# 概率质量函数
x = np.arange(0, n + 1)
pmf = binom.pmf(x, n, p)

# 绘制
plt.stem(x, pmf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title(f'Binomial Distribution B({n}, {p})')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = binom.stats(n, p, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.2.3 泊松分布

**定义 2.8（泊松分布）**：若随机变量 $X$ 的分布律为
$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \cdots$$
其中 $\lambda > 0$，则称 $X$ 服从参数为 $\lambda$ 的泊松分布，记为 $X \sim P(\lambda)$。

**期望与方差**：
$$E(X) = \lambda, \quad D(X) = \lambda$$

**泊松定理**：设 $np_n \to \lambda$（$n \to \infty$，$\lambda > 0$ 为常数），则
$$\lim_{n \to \infty} \binom{n}{k} p_n^k (1 - p_n)^{n - k} = \frac{e^{-\lambda} \lambda^k}{k!}$$

**泊松分布的应用场景**：
- 某段时间内电话交换台接到的呼唤次数
- 某地区某段时间内发生的交通事故数
- 某容器内的细菌数

**示例 2.6**：设某电话交换台每分钟接到的呼唤次数 $X \sim P(4)$，求一分钟内恰好接到3次呼唤的概率。

$$P(X = 3) = \frac{e^{-4} 4^3}{3!} = \frac{e^{-4} \times 64}{6} \approx 0.1954$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 泊松分布参数
lambda_ = 4

# 概率质量函数
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_)

# 绘制
plt.stem(x, pmf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title(f'Poisson Distribution P({lambda_})')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = poisson.stats(lambda_, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.2.4 几何分布

**定义 2.9（几何分布）**：若随机变量 $X$ 的分布律为
$$P(X = k) = (1 - p)^{k - 1} p, \quad k = 1, 2, 3, \cdots$$
其中 $0 < p < 1$，则称 $X$ 服从参数为 $p$ 的几何分布。

**几何分布的背景**：在伯努利试验中，首次成功时的试验次数服从几何分布。

**期望与方差**：
$$E(X) = \frac{1}{p}, \quad D(X) = \frac{1 - p}{p^2}$$

**无记忆性**：对于任意正整数 $m, n$，有
$$P(X > m + n \mid X > m) = P(X > n)$$

**示例 2.7**：某人进行射击，每次命中率为0.2，求首次命中时射击次数不超过5次的概率。

设首次命中次数为 $X$，则 $X$ 服从几何分布，
$$P(X \leq 5) = \sum_{k=1}^5 (0.8)^{k - 1} \times 0.2 = 0.2 + 0.16 + 0.128 + 0.1024 + 0.08192 = 0.67232$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# 几何分布参数
p = 0.2

# 概率质量函数
x = np.arange(1, 15)
pmf = geom.pmf(x, p)

# 绘制
plt.stem(x, pmf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title(f'Geometric Distribution (p = {p})')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = geom.stats(p, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.2.5 超几何分布

**定义 2.10（超几何分布）**：若随机变量 $X$ 的分布律为
$$P(X = k) = \frac{\binom{M}{k} \binom{N - M}{n - k}}{\binom{N}{n}}, \quad k = 0, 1, \cdots, \min(M, n)$$
其中 $N, M, n$ 为正整数，且 $n \leq N$，则称 $X$ 服从参数为 $(N, M, n)$ 的超几何分布，记为 $X \sim H(N, M, n)$。

**超几何分布的背景**：从含有 $M$ 件次品的 $N$ 件产品中不放回地抽取 $n$ 件，其中次品件数服从超几何分布。

**期望与方差**：
$$E(X) = n \frac{M}{N}, \quad D(X) = n \frac{M}{N} \frac{N - M}{N} \frac{N - n}{N - 1}$$

**超几何分布与二项分布的关系**：当 $N$ 很大，$n$ 相对较小时，超几何分布近似于二项分布 $B(n, \frac{M}{N})$。

**示例 2.8**：设有100件产品，其中有10件次品，从中不放回地抽取20件，求恰好取到3件次品的概率。

设次品件数为 $X$，则 $X \sim H(100, 10, 20)$，
$$P(X = 3) = \frac{\binom{10}{3} \binom{90}{17}}{\binom{100}{20}} \approx 0.1937$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# 超几何分布参数
N = 100  # 总体大小
M = 10   # 总体中的成功数
n = 20   # 抽取次数

# 概率质量函数
x = np.arange(0, min(M, n) + 1)
pmf = hypergeom.pmf(x, N, M, n)

# 绘制
plt.stem(x, pmf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title(f'Hypergeometric Distribution H({N}, {M}, {n})')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = hypergeom.stats(N, M, n, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

---

## 2.3 常见连续分布

### 2.3.1 均匀分布

**定义 2.11（均匀分布）**：若随机变量 $X$ 的概率密度函数为
$$f(x) = \begin{cases} \frac{1}{b - a}, & a \leq x \leq b \\ 0, & \text{其他} \end{cases}$$
则称 $X$ 在区间 $[a, b]$ 上服从均匀分布，记为 $X \sim U(a, b)$。

**分布函数**：
$$F(x) = \begin{cases} 0, & x < a \\ \frac{x - a}{b - a}, & a \leq x < b \\ 1, & x \geq b \end{cases}$$

**期望与方差**：
$$E(X) = \frac{a + b}{2}, \quad D(X) = \frac{(b - a)^2}{12}$$

**示例 2.9**：设 $X \sim U(0, 1)$，求 $P(0.2 < X \leq 0.5)$。

$$P(0.2 < X \leq 0.5) = \int_{0.2}^{0.5} 1 dx = 0.3$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# 均匀分布参数
a = 0
b = 1

# 概率密度函数和分布函数
x = np.linspace(a - 0.5, b + 0.5, 1000)
pdf = uniform.pdf(x, loc=a, scale=b - a)
cdf = uniform.cdf(x, loc=a, scale=b - a)

# 绘制
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('f(x)')
ax1.set_title(f'Uniform Distribution U({a}, {b}) - PDF')
ax1.grid(True, alpha=0.3)

ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.set_xlabel('X')
ax2.set_ylabel('F(x)')
ax2.set_title(f'Uniform Distribution U({a}, {b}) - CDF')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 期望和方差
mean, var = uniform.stats(loc=a, scale=b - a, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.3.2 指数分布

**定义 2.12（指数分布）**：若随机变量 $X$ 的概率密度函数为
$$f(x) = \begin{cases} \lambda e^{-\lambda x}, & x > 0 \\ 0, & x \leq 0 \end{cases}$$
其中 $\lambda > 0$，则称 $X$ 服从参数为 $\lambda$ 的指数分布，记为 $X \sim Exp(\lambda)$。

**分布函数**：
$$F(x) = \begin{cases} 0, & x \leq 0 \\ 1 - e^{-\lambda x}, & x > 0 \end{cases}$$

**期望与方差**：
$$E(X) = \frac{1}{\lambda}, \quad D(X) = \frac{1}{\lambda^2}$$

**无记忆性**：对于任意 $s, t > 0$，有
$$P(X > s + t \mid X > s) = P(X > t)$$

**指数分布的应用场景**：
- 电子元件的寿命
- 电话通话时间
- 排队系统中的服务时间

**示例 2.10**：设某电子元件的寿命 $X \sim Exp(0.001)$（单位：小时），求该元件寿命超过1000小时的概率。

$$P(X > 1000) = 1 - F(1000) = 1 - (1 - e^{-0.001 \times 1000}) = e^{-1} \approx 0.3679$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 指数分布参数
lambda_ = 0.001
scale = 1 / lambda_

# 概率密度函数和分布函数
x = np.linspace(0, 5000, 1000)
pdf = expon.pdf(x, scale=scale)
cdf = expon.cdf(x, scale=scale)

# 绘制
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('f(x)')
ax1.set_title(f'Exponential Distribution Exp({lambda_}) - PDF')
ax1.grid(True, alpha=0.3)

ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.set_xlabel('X')
ax2.set_ylabel('F(x)')
ax2.set_title(f'Exponential Distribution Exp({lambda_}) - CDF')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 期望和方差
mean, var = expon.stats(scale=scale, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.3.3 正态分布

**定义 2.13（正态分布）**：若随机变量 $X$ 的概率密度函数为
$$f(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x - \mu)^2}{2\sigma^2}}, \quad x \in \mathbb{R}$$
其中 $\mu$ 为实数，$\sigma > 0$，则称 $X$ 服从参数为 $(\mu, \sigma^2)$ 的正态分布，记为 $X \sim N(\mu, \sigma^2)$。

**正态分布的性质**：
1. 密度函数曲线关于 $x = \mu$ 对称
2. 在 $x = \mu$ 处取得最大值 $\frac{1}{\sqrt{2\pi} \sigma}$
3. 曲线在 $x = \mu \pm \sigma$ 处有拐点
4. 当 $x \to \pm \infty$ 时，曲线以 $x$ 轴为渐近线

**期望与方差**：
$$E(X) = \mu, \quad D(X) = \sigma^2$$

**定义 2.14（标准正态分布）**：参数 $\mu = 0$，$\sigma^2 = 1$ 的正态分布称为标准正态分布，记为 $Z \sim N(0, 1)$。其概率密度函数和分布函数分别记为 $\varphi(x)$ 和 $\Phi(x)$。

**标准正态分布的性质**：
1. $\varphi(-x) = \varphi(x)$
2. $\Phi(-x) = 1 - \Phi(x)$

**标准化变换**：若 $X \sim N(\mu, \sigma^2)$，则
$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

**示例 2.11**：设 $X \sim N(1, 4)$，求 $P(0 < X \leq 3)$。

标准化得 $Z = \frac{X - 1}{2} \sim N(0, 1)$，
$$P(0 < X \leq 3) = P\left(\frac{0 - 1}{2} < Z \leq \frac{3 - 1}{2}\right) = P(-0.5 < Z \leq 1)$$
$$= \Phi(1) - \Phi(-0.5) = \Phi(1) - (1 - \Phi(0.5)) \approx 0.8413 - 1 + 0.6915 = 0.5328$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 正态分布参数
mu = 1
sigma2 = 4
sigma = np.sqrt(sigma2)

# 概率密度函数和分布函数
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)
cdf = norm.cdf(x, loc=mu, scale=sigma)

# 绘制
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('f(x)')
ax1.set_title(f'Normal Distribution N({mu}, {sigma2}) - PDF')
ax1.grid(True, alpha=0.3)

ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.set_xlabel('X')
ax2.set_ylabel('F(x)')
ax2.set_title(f'Normal Distribution N({mu}, {sigma2}) - CDF')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 期望和方差
mean, var = norm.stats(loc=mu, scale=sigma, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")

# 计算概率 P(0 < X <= 3)
prob = norm.cdf(3, loc=mu, scale=sigma) - norm.cdf(0, loc=mu, scale=sigma)
print(f"P(0 < X <= 3) = {prob:.4f}")
```

### 2.3.4 伽马分布

**定义 2.15（伽马函数）**：$\Gamma(\alpha) = \int_{0}^{+\infty} t^{\alpha - 1} e^{-t} dt$，$\alpha > 0$。

**伽马函数的性质**：
1. $\Gamma(1) = 1$，$\Gamma(\frac{1}{2}) = \sqrt{\pi}$
2. $\Gamma(\alpha + 1) = \alpha \Gamma(\alpha)$
3. 对于正整数 $n$，$\Gamma(n + 1) = n!$

**定义 2.16（伽马分布）**：若随机变量 $X$ 的概率密度函数为
$$f(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, & x > 0 \\ 0, & x \leq 0 \end{cases}$$
其中 $\alpha > 0$，$\beta > 0$，则称 $X$ 服从参数为 $(\alpha, \beta)$ 的伽马分布，记为 $X \sim \Gamma(\alpha, \beta)$。

**期望与方差**：
$$E(X) = \frac{\alpha}{\beta}, \quad D(X) = \frac{\alpha}{\beta^2}$$

**伽马分布的特殊情形**：
1. 当 $\alpha = 1$ 时，伽马分布退化为指数分布 $Exp(\beta)$
2. 当 $\alpha = \frac{n}{2}$，$\beta = \frac{1}{2}$ 时，伽马分布退化为自由度为 $n$ 的卡方分布 $\chi^2(n)$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 伽马分布参数
alpha = 2
beta = 1
scale = 1 / beta

# 概率密度函数
x = np.linspace(0, 10, 1000)
pdf = gamma.pdf(x, a=alpha, scale=scale)

# 绘制
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('X')
plt.ylabel('f(x)')
plt.title(f'Gamma Distribution Γ({alpha}, {beta})')
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = gamma.stats(a=alpha, scale=scale, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.3.5 卡方分布

**定义 2.17（卡方分布）**：设 $X_1, X_2, \cdots, X_n$ 独立同服从 $N(0, 1)$，则称
$$\chi^2 = X_1^2 + X_2^2 + \cdots + X_n^2$$
服从自由度为 $n$ 的卡方分布，记为 $\chi^2 \sim \chi^2(n)$。

**概率密度函数**：
$$f(x) = \begin{cases} \frac{1}{2^{n/2} \Gamma(n/2)} x^{n/2 - 1} e^{-x/2}, & x > 0 \\ 0, & x \leq 0 \end{cases}$$

**卡方分布的性质**：
1. 可加性：若 $\chi_1^2 \sim \chi^2(n_1)$，$\chi_2^2 \sim \chi^2(n_2)$，且独立，则 $\chi_1^2 + \chi_2^2 \sim \chi^2(n_1 + n_2)$
2. 期望与方差：
   $$E(\chi^2) = n, \quad D(\chi^2) = 2n$$

**示例 2.12**：设 $\chi^2 \sim \chi^2(10)$，求 $E(\chi^2)$ 和 $D(\chi^2)$。

$$E(\chi^2) = 10, \quad D(\chi^2) = 20$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 卡方分布参数
n = 10  # 自由度

# 概率密度函数
x = np.linspace(0, 30, 1000)
pdf = chi2.pdf(x, df=n)

# 绘制
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('X')
plt.ylabel('f(x)')
plt.title(f'Chi-Square Distribution χ²({n})')
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
mean, var = chi2.stats(df=n, moments='mv')
print(f"期望 E(X) = {mean:.4f}")
print(f"方差 D(X) = {var:.4f}")
```

### 2.3.6 t分布

**定义 2.18（t分布）**：设 $X \sim N(0, 1)$，$Y \sim \chi^2(n)$，且 $X$ 与 $Y$ 独立，则称
$$t = \frac{X}{\sqrt{Y / n}}$$
服从自由度为 $n$ 的t分布，记为 $t \sim t(n)$。

**t分布的性质**：
1. 密度函数曲线关于 $t = 0$ 对称
2. 当 $n \to \infty$ 时，t分布趋近于标准正态分布
3. 当 $n > 1$ 时，$E(t) = 0$；当 $n > 2$ 时，$D(t) = \frac{n}{n - 2}$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# t分布参数
n = 5  # 自由度

# 概率密度函数（对比标准正态分布）
x = np.linspace(-4, 4, 1000)
pdf_t = t.pdf(x, df=n)
pdf_norm = norm.pdf(x, loc=0, scale=1)

# 绘制
plt.plot(x, pdf_t, 'b-', linewidth=2, label=f't({n})')
plt.plot(x, pdf_norm, 'r--', linewidth=2, label='N(0,1)')
plt.xlabel('X')
plt.ylabel('f(x)')
plt.title(f't Distribution vs Standard Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
if n > 1:
    mean = t.stats(df=n, moments='m')
    print(f"期望 E(X) = {mean:.4f}")
if n > 2:
    var = t.stats(df=n, moments='v')
    print(f"方差 D(X) = {var:.4f}")
```

### 2.3.7 F分布

**定义 2.19（F分布）**：设 $X \sim \chi^2(n_1)$，$Y \sim \chi^2(n_2)$，且 $X$ 与 $Y$ 独立，则称
$$F = \frac{X / n_1}{Y / n_2}$$
服从自由度为 $(n_1, n_2)$ 的F分布，记为 $F \sim F(n_1, n_2)$，其中 $n_1$ 称为第一自由度，$n_2$ 称为第二自由度。

**F分布的性质**：
1. 若 $F \sim F(n_1, n_2)$，则 $\frac{1}{F} \sim F(n_2, n_1)$
2. 当 $n_2 > 2$ 时，$E(F) = \frac{n_2}{n_2 - 2}$
3. 当 $n_2 > 4$ 时，$D(F) = \frac{2n_2^2(n_1 + n_2 - 2)}{n_1(n_2 - 2)^2(n_2 - 4)}$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# F分布参数
n1 = 5  # 第一自由度
n2 = 10  # 第二自由度

# 概率密度函数
x = np.linspace(0, 5, 1000)
pdf = f.pdf(x, dfn=n1, dfd=n2)

# 绘制
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('X')
plt.ylabel('f(x)')
plt.title(f'F Distribution F({n1}, {n2})')
plt.grid(True, alpha=0.3)
plt.show()

# 期望和方差
if n2 > 2:
    mean = f.stats(dfn=n1, dfd=n2, moments='m')
    print(f"期望 E(X) = {mean:.4f}")
if n2 > 4:
    var = f.stats(dfn=n1, dfd=n2, moments='v')
    print(f"方差 D(X) = {var:.4f}")
```

---

## 2.4 随机变量函数的分布

### 2.4.1 离散型随机变量函数的分布

设 $X$ 是离散型随机变量，其分布律为 $P(X = x_k) = p_k$，$k = 1, 2, \cdots$，$Y = g(X)$ 是 $X$ 的函数。

**求解方法**：
1. 先确定 $Y$ 的所有可能取值 $y_j$
2. 对于每个 $y_j$，求 $P(Y = y_j) = \sum_{k: g(x_k) = y_j} p_k$

**示例 2.13**：设 $X$ 的分布律为
$$\begin{array}{c|cccc}
X & -2 & -1 & 0 & 1 \\
\hline
P & 0.1 & 0.2 & 0.3 & 0.4 \\
\end{array}$$
求 $Y = X^2$ 的分布律。

$Y$ 的可能取值为 $0, 1, 4$，
$$P(Y = 0) = P(X = 0) = 0.3$$
$$P(Y = 1) = P(X = -1) + P(X = 1) = 0.2 + 0.4 = 0.6$$
$$P(Y = 4) = P(X = -2) = 0.1$$

**代码示例**：
```python
import numpy as np

# 离散型随机变量X的分布
x_values = np.array([-2, -1, 0, 1])
p_values = np.array([0.1, 0.2, 0.3, 0.4])

# 定义函数 Y = g(X)
def g(x):
    return x**2

# 计算Y的分布
y_values = g(x_values)
unique_y = np.unique(y_values)
y_probs = []

for y in unique_y:
    prob = np.sum(p_values[y_values == y])
    y_probs.append(prob)

# 打印结果
print("Y的分布律：")
for y, prob in zip(unique_y, y_probs):
    print(f"P(Y = {y}) = {prob:.4f}")
```

### 2.4.2 连续型随机变量函数的分布

设 $X$ 是连续型随机变量，其概率密度函数为 $f_X(x)$，$Y = g(X)$ 是 $X$ 的函数。

**方法一：分布函数法**
1. 先求 $Y$ 的分布函数 $F_Y(y) = P(Y \leq y) = P(g(X) \leq y)$
2. 再对 $F_Y(y)$ 求导，得到 $Y$ 的概率密度函数 $f_Y(y) = F_Y'(y)$

**定理 2.1（严格单调函数的分布）**：设 $X$ 是连续型随机变量，其概率密度函数为 $f_X(x)$，函数 $y = g(x)$ 严格单调可导，其反函数为 $x = h(y)$，则 $Y = g(X)$ 的概率密度函数为
$$f_Y(y) = \begin{cases} f_X(h(y)) |h'(y)|, & \alpha < y < \beta \\ 0, & \text{其他} \end{cases}$$
其中 $\alpha = \min\{g(-\infty), g(+\infty)\}$，$\beta = \max\{g(-\infty), g(+\infty)\}$。

**示例 2.14**：设 $X \sim N(\mu, \sigma^2)$，求 $Y = aX + b$（$a \neq 0$）的分布。

方法一：分布函数法
$$F_Y(y) = P(Y \leq y) = P(aX + b \leq y) = P\left(X \leq \frac{y - b}{a}\right) \quad (a > 0)$$
$$= F_X\left(\frac{y - b}{a}\right)$$
求导得
$$f_Y(y) = f_X\left(\frac{y - b}{a}\right) \cdot \frac{1}{a} = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{((y - b)/a - \mu)^2}{2\sigma^2}} \cdot \frac{1}{a}$$
$$= \frac{1}{\sqrt{2\pi} |a| \sigma} e^{-\frac{(y - (a\mu + b))^2}{2a^2\sigma^2}}$$
因此，$Y \sim N(a\mu + b, a^2\sigma^2)$。

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# X ~ N(mu, sigma^2)
mu = 1
sigma2 = 4
sigma = np.sqrt(sigma2)

# 变换 Y = aX + b
a = 2
b = 3

# 理论上 Y ~ N(a*mu + b, a^2*sigma^2)
mu_y = a * mu + b
sigma2_y = a**2 * sigma2
sigma_y = np.sqrt(sigma2_y)

# 生成样本验证
x_samples = norm.rvs(loc=mu, scale=sigma, size=10000)
y_samples = a * x_samples + b

# 绘制对比
x = np.linspace(mu_y - 4*sigma_y, mu_y + 4*sigma_y, 1000)
pdf_theory = norm.pdf(x, loc=mu_y, scale=sigma_y)

plt.hist(y_samples, bins=50, density=True, alpha=0.5, label='Samples')
plt.plot(x, pdf_theory, 'r-', linewidth=2, label='Theoretical PDF')
plt.xlabel('Y')
plt.ylabel('f(y)')
plt.title(f'Y = {a}X + {b}, X ~ N({mu}, {sigma2})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"理论期望: {mu_y:.4f}, 样本均值: {np.mean(y_samples):.4f}")
print(f"理论方差: {sigma2_y:.4f}, 样本方差: {np.var(y_samples):.4f}")
```

---

## 2.5 随机变量在AI中的应用

### 2.5.1 正态分布假设

**中心极限定理**：设 $X_1, X_2, \cdots, X_n$ 独立同分布，且 $E(X_i) = \mu$，$D(X_i) = \sigma^2 > 0$，则当 $n$ 充分大时，
$$\frac{\sum_{i=1}^n X_i - n\mu}{\sqrt{n} \sigma} \sim N(0, 1)$$

**在机器学习中的应用**：
1. **神经网络的权重初始化**：使用正态分布初始化权重，如 He 初始化和 Xavier 初始化
2. **误差项的假设**：在线性回归中，通常假设误差项服从正态分布
3. **梯度下降的噪声**：小批量梯度下降中的噪声可以用正态分布建模

**示例 2.15（Xavier 初始化）**：对于具有 $n_{in}$ 个输入神经元和 $n_{out}$ 个输出神经元的全连接层，权重 $W$ 服从
$$W \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt

def xavier_initialization(n_in, n_out, size=None):
    """Xavier 初始化"""
    scale = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(loc=0.0, scale=scale, size=size)

def he_initialization(n_in, size=None):
    """He 初始化"""
    scale = np.sqrt(2.0 / n_in)
    return np.random.normal(loc=0.0, scale=scale, size=size)

# 初始化权重
n_in = 100
n_out = 50
xavier_weights = xavier_initialization(n_in, n_out, size=10000)
he_weights = he_initialization(n_in, size=10000)

# 绘制
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(xavier_weights, bins=50, density=True, alpha=0.5)
ax1.set_xlabel('Weight Value')
ax1.set_ylabel('Density')
ax1.set_title('Xavier Initialization')
ax1.grid(True, alpha=0.3)

ax2.hist(he_weights, bins=50, density=True, alpha=0.5)
ax2.set_xlabel('Weight Value')
ax2.set_ylabel('Density')
ax2.set_title('He Initialization')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.5.2 激活函数分布

**常见激活函数**：
1. **Sigmoid**：$\sigma(x) = \frac{1}{1 + e^{-x}}$
2. **Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
3. **ReLU**：$\text{ReLU}(x) = \max(0, x)$
4. **Leaky ReLU**：$\text{LeakyReLU}(x) = \max(\alpha x, x)$

**输入服从正态分布时的输出分布**：
- 对于 $\tanh$ 激活函数，当输入方差较小时，输出近似正态分布
- 对于 ReLU 激活函数，输出是半正态分布

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 输入 X ~ N(0, 1)
x = np.random.normal(loc=0, scale=1, size=10000)

# 计算激活后的输出
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 绘制
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(x, bins=50, density=True, alpha=0.5)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_title('Input: N(0, 1)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(y_sigmoid, bins=50, density=True, alpha=0.5)
axes[0, 1].set_xlabel('σ(x)')
axes[0, 1].set_title('Sigmoid Output')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(y_tanh, bins=50, density=True, alpha=0.5)
axes[1, 0].set_xlabel('tanh(x)')
axes[1, 0].set_title('Tanh Output')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(y_relu, bins=50, density=True, alpha=0.5)
axes[1, 1].set_xlabel('ReLU(x)')
axes[1, 1].set_title('ReLU Output')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.5.3 噪声注入

**在深度学习中注入噪声的目的**：
1. **正则化**：防止过拟合，提高模型的泛化能力
2. **数据增强**：增加训练数据的多样性
3. **鲁棒性**：提高模型对输入扰动的抵抗能力

**常见的噪声类型**：
1. **高斯噪声**：$\tilde{x} = x + \epsilon$，其中 $\epsilon \sim N(0, \sigma^2)$
2. **dropout**：随机将部分神经元的输出置为0
3. **输入噪声**：在输入数据中添加噪声

**示例 2.16（高斯噪声注入）**：
$$\tilde{x}_i = x_i + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)$$

**代码示例**：
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义添加高斯噪声的函数
def add_gaussian_noise(X, sigma=0.1):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

# 训练模型（无噪声）
model_no_noise = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model_no_noise.fit(X_train, y_train)
y_pred_no_noise = model_no_noise.predict(X_test)
acc_no_noise = accuracy_score(y_test, y_pred_no_noise)

# 训练模型（添加噪声）
sigma_noise = 0.05
X_train_noisy = add_gaussian_noise(X_train, sigma=sigma_noise)
model_noisy = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model_noisy.fit(X_train_noisy, y_train)
y_pred_noisy = model_noisy.predict(X_test)
acc_noisy = accuracy_score(y_test, y_pred_noisy)

# 绘制结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6)
ax1.set_title('Original Training Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(True, alpha=0.3)

ax2.scatter(X_train_noisy[:, 0], X_train_noisy[:, 1], c=y_train, cmap='viridis', alpha=0.6)
ax2.set_title(f'Training Data with Gaussian Noise (σ={sigma_noise})')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"无噪声模型测试准确率: {acc_no_noise:.4f}")
print(f"添加噪声模型测试准确率: {acc_noisy:.4f}")
```

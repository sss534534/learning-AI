# 第4章 数字特征

## 4.1 数学期望

### 4.1.1 定义

**定义4.1（离散型随机变量的数学期望）**：设离散型随机变量 $X$ 的概率分布为
$$P(X = x_k) = p_k, \quad k = 1, 2, \dots$$
若级数 $\sum_{k=1}^{\infty} |x_k| p_k$ 收敛，则称级数 $\sum_{k=1}^{\infty} x_k p_k$ 的和为随机变量 $X$ 的**数学期望**，记为 $E(X)$ 或 $\mu_X$，即
$$E(X) = \sum_{k=1}^{\infty} x_k p_k$$

**定义4.2（连续型随机变量的数学期望）**：设连续型随机变量 $X$ 的概率密度函数为 $f(x)$，若积分 $\int_{-\infty}^{+\infty} |x| f(x) dx$ 收敛，则称积分 $\int_{-\infty}^{+\infty} x f(x) dx$ 的值为随机变量 $X$ 的**数学期望**，记为 $E(X)$ 或 $\mu_X$，即
$$E(X) = \int_{-\infty}^{+\infty} x f(x) dx$$

**定义4.3（随机变量函数的数学期望）**：设 $Y = g(X)$ 是随机变量 $X$ 的函数（$g$ 是连续函数）：

1. 若 $X$ 是离散型，概率分布为 $P(X = x_k) = p_k$，则
   $$E(Y) = E[g(X)] = \sum_{k=1}^{\infty} g(x_k) p_k$$
   （当级数绝对收敛时）

2. 若 $X$ 是连续型，概率密度为 $f(x)$，则
   $$E(Y) = E[g(X)] = \int_{-\infty}^{+\infty} g(x) f(x) dx$$
   （当积分绝对收敛时）

### 4.1.2 数学期望的性质

**定理4.1（数学期望的性质）**：设以下所涉及的数学期望都存在，则：

1. **线性性质**：
   - 设 $c$ 为常数，则 $E(c) = c$
   - 设 $c$ 为常数，$X$ 为随机变量，则 $E(cX) = cE(X)$
   - 设 $X, Y$ 为任意两个随机变量，则 $E(X + Y) = E(X) + E(Y)$
   - 推广：$E\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} E(X_i)$

2. **乘积性质**：
   - 若 $X, Y$ 相互独立，则 $E(XY) = E(X)E(Y)$
   - 推广：若 $X_1, X_2, \dots, X_n$ 相互独立，则 $E\left(\prod_{i=1}^{n} X_i\right) = \prod_{i=1}^{n} E(X_i)$

### 4.1.3 计算示例

**例4.1（二项分布的期望）**：设 $X \sim B(n, p)$，则 $E(X) = np$。

**证明**：$X$ 可表示为 $n$ 个独立的 0-1 分布变量之和：$X = X_1 + X_2 + \dots + X_n$，其中 $X_i \sim B(1, p)$，$E(X_i) = p$，故
$$E(X) = \sum_{i=1}^{n} E(X_i) = np$$

**例4.2（正态分布的期望）**：设 $X \sim N(\mu, \sigma^2)$，则 $E(X) = \mu$。

**证明**：
$$E(X) = \int_{-\infty}^{+\infty} x \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx$$
令 $t = \frac{x - \mu}{\sigma}$，则 $x = \mu + \sigma t$，$dx = \sigma dt$，代入得
$$E(X) = \int_{-\infty}^{+\infty} (\mu + \sigma t) \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \mu \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt + \sigma \int_{-\infty}^{+\infty} t \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \mu$$

### 4.1.4 代码示例

```python
import numpy as np
import scipy.stats as stats

# 离散型：二项分布的期望
n, p = 10, 0.3
binom_dist = stats.binom(n, p)
print(f"二项分布 B({n}, {p}) 的期望: {binom_dist.mean()}")

# 连续型：正态分布的期望
mu, sigma = 5, 2
norm_dist = stats.norm(mu, sigma)
print(f"正态分布 N({mu}, {sigma}^2) 的期望: {norm_dist.mean()}")

# 随机变量函数的期望
# 计算 E[X^2] 对于 X ~ N(0, 1)
def expected_function(func, dist, num_samples=100000):
    samples = dist.rvs(num_samples)
    return np.mean(func(samples))

norm_standard = stats.norm(0, 1)
e_x2 = expected_function(lambda x: x**2, norm_standard)
print(f"E[X^2] 对于 X ~ N(0, 1): {e_x2}")
```

---

## 4.2 方差与标准差

### 4.2.1 定义

**定义4.4（方差）**：设 $X$ 是一个随机变量，若 $E\left\{[X - E(X)]^2\right\}$ 存在，则称它为 $X$ 的**方差**，记为 $D(X)$ 或 $\text{Var}(X)$，即
$$D(X) = \text{Var}(X) = E\left\{[X - E(X)]^2\right\}$$

**定义4.5（标准差）**：方差的算术平方根 $\sqrt{D(X)}$ 称为 $X$ 的**标准差**或**均方差**，记为 $\sigma(X)$。

**常用计算公式**：
$$D(X) = E(X^2) - [E(X)]^2$$

### 4.2.2 方差的性质

**定理4.2（方差的性质）**：设以下所涉及的方差都存在，则：

1. 设 $c$ 为常数，则 $D(c) = 0$
2. 设 $c$ 为常数，$X$ 为随机变量，则 $D(cX) = c^2 D(X)$
3. 设 $X, Y$ 为两个随机变量，则
   $$D(X + Y) = D(X) + D(Y) + 2E\{[X - E(X)][Y - E(Y)]\}$$
   若 $X, Y$ 相互独立，则 $D(X + Y) = D(X) + D(Y)$
4. 推广：若 $X_1, X_2, \dots, X_n$ 相互独立，则
   $$D\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} D(X_i)$$
5. $D(X) = 0$ 的充要条件是 $X$ 以概率 1 取常数 $c$，即 $P(X = c) = 1$，其中 $c = E(X)$

### 4.2.3 切比雪夫不等式

**定理4.3（切比雪夫不等式）**：设随机变量 $X$ 具有数学期望 $E(X) = \mu$，方差 $D(X) = \sigma^2$，则对任意正数 $\varepsilon$，不等式
$$P(|X - \mu| \geq \varepsilon) \leq \frac{\sigma^2}{\varepsilon^2}$$
成立。

**等价形式**：
$$P(|X - \mu| < \varepsilon) \geq 1 - \frac{\sigma^2}{\varepsilon^2}$$

**意义**：切比雪夫不等式给出了在未知分布的情况下，对随机变量偏离其均值的概率的估计。

### 4.2.4 计算示例

**例4.3（二项分布的方差）**：设 $X \sim B(n, p)$，则 $D(X) = np(1 - p)$。

**例4.4（正态分布的方差）**：设 $X \sim N(\mu, \sigma^2)$，则 $D(X) = \sigma^2$。

### 4.2.5 代码示例

```python
import numpy as np
import scipy.stats as stats

# 二项分布的方差
n, p = 10, 0.3
binom_dist = stats.binom(n, p)
print(f"二项分布 B({n}, {p}) 的方差: {binom_dist.var()}")
print(f"标准差: {binom_dist.std()}")

# 正态分布的方差
mu, sigma = 5, 2
norm_dist = stats.norm(mu, sigma)
print(f"正态分布 N({mu}, {sigma}^2) 的方差: {norm_dist.var()}")

# 切比雪夫不等式验证
np.random.seed(42)
X = norm_dist.rvs(100000)
epsilon = 3
count = np.sum(np.abs(X - mu) >= epsilon)
prob_empirical = count / len(X)
prob_chebyshev = sigma**2 / epsilon**2
print(f"P(|X - mu| >= {epsilon}) 经验值: {prob_empirical}")
print(f"切比雪夫上界: {prob_chebyshev}")
print(f"验证: {prob_empirical <= prob_chebyshev}")
```

---

## 4.3 协方差、相关系数与协方差矩阵

### 4.3.1 协方差

**定义4.6（协方差）**：设 $(X, Y)$ 是二维随机变量，若 $E\left\{[X - E(X)][Y - E(Y)]\right\}$ 存在，则称它为 $X$ 与 $Y$ 的**协方差**，记为 $\text{Cov}(X, Y)$，即
$$\text{Cov}(X, Y) = E\left\{[X - E(X)][Y - E(Y)]\right\}$$

**常用计算公式**：
$$\text{Cov}(X, Y) = E(XY) - E(X)E(Y)$$

**协方差的性质**：
1. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
2. $\text{Cov}(aX, bY) = ab\text{Cov}(X, Y)$，其中 $a, b$ 为常数
3. $\text{Cov}(X_1 + X_2, Y) = \text{Cov}(X_1, Y) + \text{Cov}(X_2, Y)$
4. $D(X + Y) = D(X) + D(Y) + 2\text{Cov}(X, Y)$

### 4.3.2 相关系数

**定义4.7（相关系数）**：设 $(X, Y)$ 是二维随机变量，$D(X) > 0, D(Y) > 0$，称
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}$$
为 $X$ 与 $Y$ 的**相关系数**。

**相关系数的性质**：
1. $|\rho_{XY}| \leq 1$
2. $|\rho_{XY}| = 1$ 的充要条件是存在常数 $a, b$，使得 $P(Y = aX + b) = 1$
3. 若 $X, Y$ 相互独立，则 $\rho_{XY} = 0$（反之不成立）

**定义4.8（不相关）**：若 $\rho_{XY} = 0$，则称 $X$ 与 $Y$ **不相关**。

**注**：独立一定不相关，但不相关不一定独立。

### 4.3.3 协方差矩阵

**定义4.9（协方差矩阵）**：设 $n$ 维随机变量 $(X_1, X_2, \dots, X_n)$ 的二阶矩都存在，记
$$c_{ij} = \text{Cov}(X_i, X_j) = E\left\{[X_i - E(X_i)][X_j - E(X_j)]\right\}, \quad i, j = 1, 2, \dots, n$$
则称矩阵
$$\Sigma = \begin{pmatrix}
c_{11} & c_{12} & \dots & c_{1n} \\
c_{21} & c_{22} & \dots & c_{2n} \\
\vdots & \vdots & & \vdots \\
c_{n1} & c_{n2} & \dots & c_{nn}
\end{pmatrix}$$
为 $n$ 维随机变量的**协方差矩阵**。

**协方差矩阵的性质**：
1. 对称性：$\Sigma^T = \Sigma$
2. 半正定性：对任意实向量 $\boldsymbol{a}$，有 $\boldsymbol{a}^T \Sigma \boldsymbol{a} \geq 0$

### 4.3.4 代码示例

```python
import numpy as np
import scipy.stats as stats

# 生成二维正态分布数据
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # 协方差矩阵
X, Y = np.random.multivariate_normal(mean, cov, 10000).T

# 计算协方差
cov_xy = np.cov(X, Y)[0, 1]
print(f"协方差 Cov(X, Y): {cov_xy}")

# 计算相关系数
rho_xy = np.corrcoef(X, Y)[0, 1]
print(f"相关系数 ρ_XY: {rho_xy}")

# 验证 Cov(X, Y) = E[XY] - E[X]E[Y]
e_xy = np.mean(X * Y)
e_x = np.mean(X)
e_y = np.mean(Y)
cov_verify = e_xy - e_x * e_y
print(f"验证协方差公式: {cov_verify}")

# 协方差矩阵
print("协方差矩阵:")
print(np.cov(X, Y))
```

---

## 4.4 矩、矩母函数与特征函数

### 4.4.1 矩

**定义4.10（原点矩）**：设 $X$ 为随机变量，若 $E(X^k)$ 存在（$k = 1, 2, \dots$），则称它为 $X$ 的**$k$ 阶原点矩**，记为 $\mu_k'$，即
$$\mu_k' = E(X^k)$$

**定义4.11（中心矩）**：设 $X$ 为随机变量，若 $E\left\{[X - E(X)]^k\right\}$ 存在（$k = 1, 2, \dots$），则称它为 $X$ 的**$k$ 阶中心矩**，记为 $\mu_k$，即
$$\mu_k = E\left\{[X - E(X)]^k\right\}$$

**注**：
- 1 阶原点矩就是数学期望：$\mu_1' = E(X)$
- 2 阶中心矩就是方差：$\mu_2 = D(X)$

**定义4.12（混合矩）**：设 $(X, Y)$ 为二维随机变量，若 $E(X^k Y^l)$ 存在（$k, l = 1, 2, \dots$），则称它为 $X$ 和 $Y$ 的**$k+l$ 阶混合原点矩**；若 $E\left\{[X - E(X)]^k [Y - E(Y)]^l\right\}$ 存在，则称它为 $X$ 和 $Y$ 的**$k+l$ 阶混合中心矩**。

### 4.4.2 矩母函数

**定义4.13（矩母函数）**：设 $X$ 为随机变量，若 $E(e^{tX})$ 在 $t = 0$ 的某邻域内存在，则称
$$M_X(t) = E(e^{tX})$$
为 $X$ 的**矩母函数**。

**矩母函数的性质**：
1. $M_X(0) = 1$
2. 矩母函数唯一确定随机变量的分布
3. 若 $X, Y$ 相互独立，则 $M_{X+Y}(t) = M_X(t) M_Y(t)$
4. 矩生成性质：$M_X^{(k)}(0) = E(X^k)$

### 4.4.3 特征函数

**定义4.14（特征函数）**：设 $X$ 为随机变量，称
$$\phi_X(t) = E(e^{itX})$$
为 $X$ 的**特征函数**，其中 $i = \sqrt{-1}$ 为虚数单位。

**特征函数的性质**：
1. $|\phi_X(t)| \leq 1$
2. $\phi_X(-t) = \overline{\phi_X(t)}$（共轭）
3. 特征函数一致连续
4. 特征函数唯一确定随机变量的分布
5. 若 $X, Y$ 相互独立，则 $\phi_{X+Y}(t) = \phi_X(t) \phi_Y(t)$
6. 若 $E(X^k)$ 存在，则 $\phi_X^{(k)}(0) = i^k E(X^k)$

### 4.4.4 代码示例

```python
import numpy as np
import scipy.stats as stats

# 计算高阶矩
norm_dist = stats.norm(0, 1)
X = norm_dist.rvs(100000)

# 各阶原点矩
for k in range(1, 5):
    moment = np.mean(X**k)
    print(f"{k} 阶原点矩: {moment}")

# 各阶中心矩
mean = np.mean(X)
for k in range(2, 5):
    central_moment = np.mean((X - mean)**k)
    print(f"{k} 阶中心矩: {central_moment}")

# 使用 scipy 计算矩
print("\n使用 scipy.stats 计算:")
print(f"偏度 (3阶标准化矩): {stats.skew(X)}")
print(f"峰度 (4阶标准化矩 - 3): {stats.kurtosis(X)}")
```

---

## 4.5 常见分布的数字特征总结

| 分布名称 | 参数 | 概率分布/密度函数 | 数学期望 $E(X)$ | 方差 $D(X)$ |
|---------|------|------------------|-----------------|------------|
| 0-1 分布 | $0 < p < 1$ | $P(X=1)=p, P(X=0)=1-p$ | $p$ | $p(1-p)$ |
| 二项分布 | $n \geq 1, 0 < p < 1$ | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| 泊松分布 | $\lambda > 0$ | $P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| 几何分布 | $0 < p < 1$ | $P(X=k)=(1-p)^{k-1}p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |
| 均匀分布 | $a < b$ | $f(x)=\frac{1}{b-a}, x \in [a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| 指数分布 | $\lambda > 0$ | $f(x)=\lambda e^{-\lambda x}, x \geq 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| 正态分布 | $\mu, \sigma > 0$ | $f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| $\chi^2$ 分布 | $n$ | - | $n$ | $2n$ |
| $t$ 分布 | $n$ | - | $0(n>1)$ | $\frac{n}{n-2}(n>2)$ |
| $F$ 分布 | $n_1, n_2$ | - | $\frac{n_2}{n_2-2}(n_2>2)$ | - |

---

## 4.6 数字特征在AI中的应用

### 4.6.1 损失函数设计

**均方误差（MSE）**：
$$\text{MSE} = E\left[(Y - \hat{Y})^2\right] = D(Y - \hat{Y}) + [E(Y - \hat{Y})]^2$$

**交叉熵损失**：利用对数似然，与信息论中的熵概念相关。

### 4.6.2 正则化

**L2 正则化（权重衰减）**：在损失函数中加入权重的平方和，等价于对权重参数施加零均值的高斯先验，利用方差控制模型复杂度。

### 4.6.3 模型诊断

- **偏差-方差分解**：
  $$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$
- **协方差矩阵**：用于主成分分析（PCA）降维
- **相关系数**：特征选择，去除高度相关的特征

### 4.6.4 代码示例：偏差-方差分解

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 生成数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * X + np.sin(X)
y = y_true + np.random.normal(0, 0.5, size=y_true.shape)

# 不同复杂度的模型
degrees = [1, 3, 10]
models = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    models.append((poly, model))

# 可视化
plt.figure(figsize=(12, 4))
for i, (poly, model) in enumerate(models):
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, alpha=0.5, label='数据')
    y_pred = model.predict(poly.transform(X))
    plt.plot(X, y_pred, 'r-', label=f'度{degrees[i]}')
    plt.plot(X, y_true, 'g--', label='真实函数')
    plt.legend()
    plt.title(f'模型复杂度: {degrees[i]}')

plt.tight_layout()
plt.savefig('bias-variance.png', dpi=100)
plt.show()
```

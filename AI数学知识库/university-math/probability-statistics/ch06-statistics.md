# 第六章 数理统计基础


## 元数据
- **难度**: ⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


## 1. 数理统计基本概念

### 1.1 总体与样本

**定义 1.1（总体）**：研究对象的全体称为总体（或母体），记为 $X$。总体中的每个元素称为个体。总体通常用一个随机变量表示，其分布称为总体分布。

**定义 1.2（样本）**：从总体 $X$ 中抽取的 $n$ 个个体 $X_1, X_2, \dots, X_n$ 称为样本（或子样），$n$ 称为样本容量。若 $X_1, X_2, \dots, X_n$ 相互独立且与总体 $X$ 同分布，则称为简单随机样本。

**定义 1.3（样本值）**：样本 $X_1, X_2, \dots, X_n$ 的观察值 $x_1, x_2, \dots, x_n$ 称为样本值。

### 1.2 统计量

**定义 1.4（统计量）**：设 $X_1, X_2, \dots, X_n$ 是来自总体 $X$ 的一个样本，$g(X_1, X_2, \dots, X_n)$ 是 $X_1, X_2, \dots, X_n$ 的函数，若 $g$ 中不含任何未知参数，则称 $g(X_1, X_2, \dots, X_n)$ 为一个统计量。

**常用统计量**：

1. **样本均值**：
   $$
   \bar{X} = \frac{1}{n} \sum_{i=1}^n X_i
   $$

2. **样本方差**：
   $$
   S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2
   $$
   其中 $n-1$ 称为自由度。

3. **样本标准差**：
   $$
   S = \sqrt{S^2}
   $$

4. **样本 $k$ 阶原点矩**：
   $$
   A_k = \frac{1}{n} \sum_{i=1}^n X_i^k, \quad k = 1, 2, \dots
   $$

5. **样本 $k$ 阶中心矩**：
   $$
   B_k = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^k, \quad k = 1, 2, \dots
   $$

6. **顺序统计量**：将样本 $X_1, X_2, \dots, X_n$ 按从小到大排序为 $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$，则 $X_{(k)}$ 称为第 $k$ 个顺序统计量。

**定理 1.1**：设总体 $X$ 具有数学期望 $E(X) = \mu$ 和方差 $D(X) = \sigma^2$，$X_1, X_2, \dots, X_n$ 是来自总体 $X$ 的样本，则：
$$
E(\bar{X}) = \mu, \quad D(\bar{X}) = \frac{\sigma^2}{n}, \quad E(S^2) = \sigma^2
$$

### 1.3 经验分布

**定义 1.5（经验分布函数）**：设 $x_1, x_2, \dots, x_n$ 是总体 $X$ 的一个样本值，将它们按从小到大排序为 $x_{(1)} \leq x_{(2)} \leq \dots \leq x_{(n)}$，定义函数：
$$
F_n(x) =
\begin{cases}
0, & x < x_{(1)} \\
\frac{k}{n}, & x_{(k)} \leq x < x_{(k+1)}, \quad k = 1, 2, \dots, n-1 \\
1, & x \geq x_{(n)}
\end{cases}
$$
称 $F_n(x)$ 为经验分布函数。

**定理 1.2（格列汶科定理）**：设总体 $X$ 的分布函数为 $F(x)$，经验分布函数为 $F_n(x)$，则有：
$$
P\left\{ \lim_{n \to \infty} \sup_{-\infty < x < +\infty} |F_n(x) - F(x)| = 0 \right\} = 1
$$
即 $F_n(x)$ 以概率 1 一致收敛于 $F(x)$。

### 1.4 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 生成样本数据
np.random.seed(42)
population = np.random.normal(loc=5, scale=2, size=10000)  # 总体
sample = np.random.choice(population, size=100, replace=False)  # 样本

# 2. 计算统计量
sample_mean = np.mean(sample)
sample_variance = np.var(sample, ddof=1)
sample_std = np.std(sample, ddof=1)

print(f"样本均值: {sample_mean:.4f}")
print(f"样本方差: {sample_variance:.4f}")
print(f"样本标准差: {sample_std:.4f}")

# 3. 经验分布函数
def empirical_cdf(sample, x):
    return np.sum(sample <= x) / len(sample)

x_values = np.linspace(min(sample)-1, max(sample)+1, 1000)
ecdf_values = [empirical_cdf(sample, x) for x in x_values]

# 4. 绘制经验分布函数与理论分布函数
plt.figure(figsize=(10, 6))
plt.plot(x_values, ecdf_values, label='经验分布函数', linewidth=2)
plt.plot(x_values, stats.norm.cdf(x_values, loc=5, scale=2), label='理论分布函数', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('经验分布函数与理论分布函数')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. 常用统计分布

### 2.1 $\chi^2$（卡方）分布

**定义 2.1（$\chi^2$ 分布）**：设 $X_1, X_2, \dots, X_n$ 相互独立且都服从标准正态分布 $N(0,1)$，则称随机变量
$$
\chi^2 = X_1^2 + X_2^2 + \dots + X_n^2
$$
服从自由度为 $n$ 的 $\chi^2$ 分布，记为 $\chi^2 \sim \chi^2(n)$。

**性质 2.1**：

1. 若 $\chi^2 \sim \chi^2(n)$，则
   $$
   E(\chi^2) = n, \quad D(\chi^2) = 2n
   $$

2. 可加性：若 $\chi_1^2 \sim \chi^2(n_1)$，$\chi_2^2 \sim \chi^2(n_2)$，且相互独立，则
   $$
   \chi_1^2 + \chi_2^2 \sim \chi^2(n_1 + n_2)
   $$

3. 上 $\alpha$ 分位点：设 $\chi^2 \sim \chi^2(n)$，对于给定的 $\alpha(0 < \alpha < 1)$，称满足条件
   $$
   P\{\chi^2 > \chi^2_\alpha(n)\} = \alpha
   $$
   的点 $\chi^2_\alpha(n)$ 为 $\chi^2(n)$ 分布的上 $\alpha$ 分位点。

**定理 2.1**：设 $X_1, X_2, \dots, X_n$ 是来自正态总体 $N(\mu, \sigma^2)$ 的样本，则：
1. $\bar{X}$ 与 $S^2$ 相互独立；
2. $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$。

### 2.2 $t$ 分布

**定义 2.2（$t$ 分布）**：设 $X \sim N(0,1)$，$Y \sim \chi^2(n)$，且 $X$ 与 $Y$ 相互独立，则称随机变量
$$
T = \frac{X}{\sqrt{Y/n}}
$$
服从自由度为 $n$ 的 $t$ 分布（或学生氏分布），记为 $T \sim t(n)$。

**性质 2.2**：

1. $t$ 分布的概率密度函数关于 $t=0$ 对称；
2. $\lim_{n \to \infty} t(n) = N(0,1)$，即当 $n$ 很大时，$t$ 分布近似于标准正态分布；
3. 上 $\alpha$ 分位点：设 $T \sim t(n)$，对于给定的 $\alpha(0 < \alpha < 1)$，称满足条件
   $$
   P\{T > t_\alpha(n)\} = \alpha
   $$
   的点 $t_\alpha(n)$ 为 $t(n)$ 分布的上 $\alpha$ 分位点。由对称性知 $t_{1-\alpha}(n) = -t_\alpha(n)$。

**定理 2.2**：设 $X_1, X_2, \dots, X_n$ 是来自正态总体 $N(\mu, \sigma^2)$ 的样本，则
$$
\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)
$$

**定理 2.3**：设 $X_1, X_2, \dots, X_{n_1}$ 和 $Y_1, Y_2, \dots, Y_{n_2}$ 分别是来自正态总体 $N(\mu_1, \sigma^2)$ 和 $N(\mu_2, \sigma^2)$ 的样本，且这两个样本相互独立，则
$$
\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{S_w \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1 + n_2 - 2)
$$
其中
$$
S_w^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}, \quad S_w = \sqrt{S_w^2}
$$

### 2.3 $F$ 分布

**定义 2.3（$F$ 分布）**：设 $U \sim \chi^2(n_1)$，$V \sim \chi^2(n_2)$，且 $U$ 与 $V$ 相互独立，则称随机变量
$$
F = \frac{U/n_1}{V/n_2}
$$
服从自由度为 $(n_1, n_2)$ 的 $F$ 分布，记为 $F \sim F(n_1, n_2)$，其中 $n_1$ 称为第一自由度，$n_2$ 称为第二自由度。

**性质 2.3**：

1. 若 $F \sim F(n_1, n_2)$，则 $\frac{1}{F} \sim F(n_2, n_1)$；
2. 上 $\alpha$ 分位点：设 $F \sim F(n_1, n_2)$，对于给定的 $\alpha(0 < \alpha < 1)$，称满足条件
   $$
   P\{F > F_\alpha(n_1, n_2)\} = \alpha
   $$
   的点 $F_\alpha(n_1, n_2)$ 为 $F(n_1, n_2)$ 分布的上 $\alpha$ 分位点；
3. $F_{1-\alpha}(n_1, n_2) = \frac{1}{F_\alpha(n_2, n_1)}$。

**定理 2.4**：设 $X_1, X_2, \dots, X_{n_1}$ 和 $Y_1, Y_2, \dots, Y_{n_2}$ 分别是来自正态总体 $N(\mu_1, \sigma_1^2)$ 和 $N(\mu_2, \sigma_2^2)$ 的样本，且这两个样本相互独立，则
$$
\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} \sim F(n_1 - 1, n_2 - 1)
$$

### 2.4 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 卡方分布
x = np.linspace(0, 20, 1000)
dfs = [1, 4, 10, 20]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
for df in dfs:
    plt.plot(x, stats.chi2.pdf(x, df), label=f'自由度 {df}')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('卡方分布概率密度函数')
plt.legend()
plt.grid(True)

# 2. t分布
x = np.linspace(-4, 4, 1000)
dfs = [1, 4, 10, 30]

plt.subplot(1, 3, 2)
for df in dfs:
    plt.plot(x, stats.t.pdf(x, df), label=f'自由度 {df}')
plt.plot(x, stats.norm.pdf(x), label='标准正态', linestyle='--')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('t分布概率密度函数')
plt.legend()
plt.grid(True)

# 3. F分布
x = np.linspace(0, 5, 1000)
df_pairs = [(2, 10), (5, 10), (10, 10), (20, 10)]

plt.subplot(1, 3, 3)
for dfn, dfd in df_pairs:
    plt.plot(x, stats.f.pdf(x, dfn, dfd), label=f'自由度 ({dfn}, {dfd})')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('F分布概率密度函数')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. 分位点示例
print("卡方分布上0.05分位点 (自由度10):", stats.chi2.ppf(0.95, 10))
print("t分布上0.05分位点 (自由度10):", stats.t.ppf(0.95, 10))
print("F分布上0.05分位点 (自由度5,10):", stats.f.ppf(0.95, 5, 10))
```

---

## 3. 参数估计

### 3.1 矩估计

**定义 3.1（矩估计法）**：设总体 $X$ 的分布函数为 $F(x; \theta_1, \theta_2, \dots, \theta_k)$，其中 $\theta_1, \theta_2, \dots, \theta_k$ 是未知参数。假设总体 $X$ 的前 $k$ 阶矩存在，令
$$
\mu_l = A_l, \quad l = 1, 2, \dots, k
$$
其中 $\mu_l = E(X^l)$ 是总体的 $l$ 阶原点矩，$A_l = \frac{1}{n} \sum_{i=1}^n X_i^l$ 是样本的 $l$ 阶原点矩。解这 $k$ 个方程组成的方程组，得到的解 $\hat{\theta}_1, \hat{\theta}_2, \dots, \hat{\theta}_k$ 分别作为 $\theta_1, \theta_2, \dots, \theta_k$ 的估计量，这种估计方法称为矩估计法，所得估计量称为矩估计量。

**例 3.1**：设总体 $X \sim N(\mu, \sigma^2)$，其中 $\mu, \sigma^2$ 为未知参数，求 $\mu, \sigma^2$ 的矩估计量。

**解**：
$$
\mu_1 = E(X) = \mu, \quad \mu_2 = E(X^2) = D(X) + [E(X)]^2 = \sigma^2 + \mu^2
$$
令
$$
\begin{cases}
\mu = A_1 \\
\sigma^2 + \mu^2 = A_2
\end{cases}
$$
解得
$$
\hat{\mu} = A_1 = \bar{X}, \quad \hat{\sigma}^2 = A_2 - A_1^2 = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2
$$

### 3.2 最大似然估计

**定义 3.2（似然函数）**：设总体 $X$ 的概率密度函数（或分布律）为 $f(x; \theta)$，$\theta$ 为未知参数，$X_1, X_2, \dots, X_n$ 是来自总体 $X$ 的样本，则称
$$
L(\theta) = \prod_{i=1}^n f(X_i; \theta)
$$
为样本的似然函数。

**定义 3.3（最大似然估计）**：若存在统计量 $\hat{\theta} = \hat{\theta}(X_1, X_2, \dots, X_n)$，使得
$$
L(\hat{\theta}) = \max_{\theta} L(\theta)
$$
则称 $\hat{\theta}$ 为 $\theta$ 的最大似然估计量，记为 $\hat{\theta}_{MLE}$。

**求解步骤**：
1. 写出似然函数 $L(\theta)$；
2. 取对数得对数似然函数 $\ln L(\theta)$；
3. 对 $\theta$ 求导，令导数为零，解方程得到最大似然估计。

**性质 3.1（不变性）**：若 $\hat{\theta}$ 是 $\theta$ 的最大似然估计，$g(\theta)$ 是 $\theta$ 的连续函数，则 $g(\hat{\theta})$ 是 $g(\theta)$ 的最大似然估计。

**例 3.2**：设总体 $X \sim N(\mu, \sigma^2)$，求 $\mu, \sigma^2$ 的最大似然估计。

**解**：似然函数为
$$
L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left\{ -\frac{(x_i - \mu)^2}{2\sigma^2} \right\}
$$
取对数得
$$
\ln L(\mu, \sigma^2) = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
$$
求偏导并令其为零
$$
\frac{\partial \ln L}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0, \quad \frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^n (x_i - \mu)^2 = 0
$$
解得
$$
\hat{\mu} = \bar{X}, \quad \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2
$$

### 3.3 贝叶斯估计

**定义 3.4（先验分布与后验分布）**：设总体 $X$ 的概率密度为 $f(x|\theta)$，其中 $\theta$ 为未知参数，$\theta$ 具有概率密度 $\pi(\theta)$（先验分布）。给定样本 $X_1, X_2, \dots, X_n$ 后，$\theta$ 的条件概率密度
$$
\pi(\theta|x_1, x_2, \dots, x_n) = \frac{f(x_1, x_2, \dots, x_n|\theta) \pi(\theta)}{\int f(x_1, x_2, \dots, x_n|\theta) \pi(\theta) d\theta}
$$
称为 $\theta$ 的后验分布。

**定义 3.5（贝叶斯估计）**：常用的贝叶斯估计有：
1. **后验期望估计**：$\hat{\theta} = E(\theta|X_1, X_2, \dots, X_n)$
2. **后验中位数估计**：取后验分布的中位数作为 $\theta$ 的估计
3. **后验众数估计**：取后验分布的众数作为 $\theta$ 的估计

### 3.4 代码示例

```python
import numpy as np
from scipy import stats

# 生成样本数据
np.random.seed(42)
sample = np.random.normal(loc=5, scale=2, size=100)

# 1. 矩估计
mu_moment = np.mean(sample)
sigma2_moment = np.var(sample, ddof=0)  # 除以n

# 2. 最大似然估计
mu_mle = np.mean(sample)
sigma2_mle = np.var(sample, ddof=0)  # 正态分布下MLE与矩估计相同

print("矩估计:")
print(f"  mu = {mu_moment:.4f}")
print(f"  sigma2 = {sigma2_moment:.4f}")

print("\n最大似然估计:")
print(f"  mu = {mu_mle:.4f}")
print(f"  sigma2 = {sigma2_mle:.4f}")

# 3. 贝叶斯估计 (正态-正态模型)
# 先验分布: mu ~ N(mu0, tau2)
mu0 = 0  # 先验均值
tau2 = 10  # 先验方差
sigma2 = 4  # 已知总体方差

n = len(sample)
x_bar = np.mean(sample)

# 后验分布参数
mu_post = (n * x_bar / sigma2 + mu0 / tau2) / (n / sigma2 + 1 / tau2)
sigma2_post = 1 / (n / sigma2 + 1 / tau2)

print("\n贝叶斯估计 (后验期望):")
print(f"  mu = {mu_post:.4f}")
print(f"  后验方差 = {sigma2_post:.4f}")

# 4. 使用scipy的MLE函数
def normal_log_likelihood(theta, data):
    mu, sigma = theta
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

from scipy.optimize import minimize
result = minimize(normal_log_likelihood, [0, 1], args=(sample,), bounds=[(None, None), (1e-6, None)])
mu_mle_scipy, sigma_mle_scipy = result.x

print("\nScipy优化MLE:")
print(f"  mu = {mu_mle_scipy:.4f}")
print(f"  sigma = {sigma_mle_scipy:.4f}")
```

---

## 4. 估计量评价标准

### 4.1 无偏性

**定义 4.1（无偏估计）**：设 $\hat{\theta} = \hat{\theta}(X_1, X_2, \dots, X_n)$ 是未知参数 $\theta$ 的估计量，若
$$
E(\hat{\theta}) = \theta
$$
则称 $\hat{\theta}$ 是 $\theta$ 的无偏估计量。

**定义 4.2（渐近无偏估计）**：若
$$
\lim_{n \to \infty} E(\hat{\theta}) = \theta
$$
则称 $\hat{\theta}$ 是 $\theta$ 的渐近无偏估计量。

**例 4.1**：对于正态总体 $N(\mu, \sigma^2)$，有
- $E(\bar{X}) = \mu$，故 $\bar{X}$ 是 $\mu$ 的无偏估计；
- $E(S^2) = \sigma^2$，故 $S^2$ 是 $\sigma^2$ 的无偏估计；
- $E\left(\frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2\right) = \frac{n-1}{n} \sigma^2 \neq \sigma^2$，但它是渐近无偏的。

### 4.2 有效性

**定义 4.3（有效性）**：设 $\hat{\theta}_1$ 和 $\hat{\theta}_2$ 都是 $\theta$ 的无偏估计量，若
$$
D(\hat{\theta}_1) < D(\hat{\theta}_2)
$$
则称 $\hat{\theta}_1$ 比 $\hat{\theta}_2$ 有效。

**定义 4.4（Cramér-Rao 下界）**：设总体 $X$ 的概率密度为 $f(x; \theta)$，$\theta \in \Theta$，满足正则条件，$\hat{\theta}$ 是 $\theta$ 的无偏估计量，则
$$
D(\hat{\theta}) \geq \frac{1}{nI(\theta)}
$$
其中
$$
I(\theta) = E\left[ \left( \frac{\partial \ln f(X; \theta)}{\partial \theta} \right)^2 \right] = -E\left[ \frac{\partial^2 \ln f(X; \theta)}{\partial \theta^2} \right]
$$
称为 Fisher 信息。

**定义 4.5（有效估计）**：若无偏估计量 $\hat{\theta}$ 的方差达到 Cramér-Rao 下界，则称 $\hat{\theta}$ 为 $\theta$ 的有效估计量。

### 4.3 相合性

**定义 4.6（相合估计）**：设 $\hat{\theta}_n = \hat{\theta}(X_1, X_2, \dots, X_n)$ 是 $\theta$ 的估计量，若对任意 $\varepsilon > 0$，有
$$
\lim_{n \to \infty} P\{|\hat{\theta}_n - \theta| < \varepsilon\} = 1
$$
即 $\hat{\theta}_n \xrightarrow{P} \theta$，则称 $\hat{\theta}_n$ 是 $\theta$ 的相合估计量（或一致估计量）。

**定理 4.1**：若 $\hat{\theta}_n$ 是 $\theta$ 的无偏估计量，且 $\lim_{n \to \infty} D(\hat{\theta}_n) = 0$，则 $\hat{\theta}_n$ 是 $\theta$ 的相合估计量。

### 4.4 渐近正态性

**定义 4.7（渐近正态性）**：若存在序列 $\{\mu_n\}$ 和 $\{\sigma_n^2\}$，使得
$$
\frac{\hat{\theta}_n - \mu_n}{\sigma_n} \xrightarrow{d} N(0,1)
$$
则称 $\hat{\theta}_n$ 具有渐近正态性。

**定理 4.2**：最大似然估计量在正则条件下具有渐近正态性，即
$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} N(0, \frac{1}{I(\theta)})
$$

### 4.5 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def evaluate_estimators(true_mu=5, true_sigma=2, sample_size=100, num_trials=10000):
    np.random.seed(42)
    
    estimates_mu = []
    estimates_s2_unbiased = []
    estimates_s2_biased = []
    
    for _ in range(num_trials):
        sample = np.random.normal(true_mu, true_sigma, sample_size)
        estimates_mu.append(np.mean(sample))
        estimates_s2_unbiased.append(np.var(sample, ddof=1))
        estimates_s2_biased.append(np.var(sample, ddof=0))
    
    estimates_mu = np.array(estimates_mu)
    estimates_s2_unbiased = np.array(estimates_s2_unbiased)
    estimates_s2_biased = np.array(estimates_s2_biased)
    
    # 计算偏差
    bias_mu = np.mean(estimates_mu) - true_mu
    bias_s2_unbiased = np.mean(estimates_s2_unbiased) - true_sigma**2
    bias_s2_biased = np.mean(estimates_s2_biased) - true_sigma**2
    
    # 计算方差
    var_mu = np.var(estimates_mu)
    var_s2_unbiased = np.var(estimates_s2_unbiased)
    var_s2_biased = np.var(estimates_s2_biased)
    
    # 计算均方误差
    mse_mu = np.mean((estimates_mu - true_mu)**2)
    mse_s2_unbiased = np.mean((estimates_s2_unbiased - true_sigma**2)**2)
    mse_s2_biased = np.mean((estimates_s2_biased - true_sigma**2)**2)
    
    print("μ的估计:")
    print(f"  偏差: {bias_mu:.6f}")
    print(f"  方差: {var_mu:.6f}")
    print(f"  均方误差: {mse_mu:.6f}")
    
    print("\nσ²的无偏估计 (样本方差 S²):")
    print(f"  偏差: {bias_s2_unbiased:.6f}")
    print(f"  方差: {var_s2_unbiased:.6f}")
    print(f"  均方误差: {mse_s2_unbiased:.6f}")
    
    print("\nσ²的有偏估计 (除以n):")
    print(f"  偏差: {bias_s2_biased:.6f}")
    print(f"  方差: {var_s2_biased:.6f}")
    print(f"  均方误差: {mse_s2_biased:.6f}")
    
    # 绘制估计分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(estimates_mu, bins=50, density=True, alpha=0.7, label='估计分布')
    x = np.linspace(true_mu - 3*np.sqrt(var_mu), true_mu + 3*np.sqrt(var_mu), 100)
    axes[0].plot(x, stats.norm.pdf(x, true_mu, np.sqrt(var_mu)), 'r-', label='正态近似')
    axes[0].axvline(true_mu, color='k', linestyle='--', label='真实值')
    axes[0].set_xlabel('μ̂')
    axes[0].set_title('μ的估计分布')
    axes[0].legend()
    
    axes[1].hist(estimates_s2_unbiased, bins=50, density=True, alpha=0.7, label='无偏估计')
    axes[1].axvline(true_sigma**2, color='k', linestyle='--', label='真实值')
    axes[1].set_xlabel('σ̂²')
    axes[1].set_title('σ²的无偏估计分布')
    axes[1].legend()
    
    axes[2].hist(estimates_s2_biased, bins=50, density=True, alpha=0.7, label='有偏估计')
    axes[2].axvline(true_sigma**2, color='k', linestyle='--', label='真实值')
    axes[2].set_xlabel('σ̂²')
    axes[2].set_title('σ²的有偏估计分布')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

evaluate_estimators()
```

---

## 5. 假设检验

### 5.1 基本思想

**定义 5.1（假设检验）**：关于总体分布的未知参数或分布形式的陈述称为统计假设，记为 $H_0$（原假设）和 $H_1$（备择假设）。根据样本信息判断是否拒绝 $H_0$ 的过程称为假设检验。

**两类错误**：
- **第一类错误（弃真）**：$H_0$ 为真时拒绝 $H_0$，概率记为 $\alpha$（显著性水平）；
- **第二类错误（取伪）**：$H_0$ 为假时接受 $H_0$，概率记为 $\beta$。

**检验步骤**：
1. 提出原假设 $H_0$ 和备择假设 $H_1$；
2. 选择检验统计量，确定其在 $H_0$ 成立时的分布；
3. 给定显著性水平 $\alpha$，确定拒绝域；
4. 计算检验统计量的值，判断是否落在拒绝域内，作出决策。

### 5.2 显著性检验

**定义 5.2（p值）**：在原假设 $H_0$ 成立的条件下，出现当前样本观察值或更极端情况的概率称为 p 值。

**决策规则**：
- 若 p 值 ≤ α，则拒绝 $H_0$；
- 若 p 值 > α，则不拒绝 $H_0$。

### 5.3 t 检验

**5.3.1 单样本 t 检验**

检验假设：$H_0: \mu = \mu_0$，$H_1: \mu \neq \mu_0$（或单侧）

检验统计量：
$$
T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1) \quad (H_0成立时)
$$

拒绝域：$|T| \geq t_{\alpha/2}(n-1)$（双侧）

**5.3.2 两独立样本 t 检验**

设两总体 $N(\mu_1, \sigma_1^2)$ 和 $N(\mu_2, \sigma_2^2)$，检验 $H_0: \mu_1 = \mu_2$

当 $\sigma_1^2 = \sigma_2^2 = \sigma^2$ 时，检验统计量：
$$
T = \frac{\bar{X} - \bar{Y}}{S_w \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1 + n_2 - 2)
$$

**5.3.3 配对样本 t 检验**

设 $D_i = X_i - Y_i$，检验 $H_0: \mu_D = 0$

检验统计量：
$$
T = \frac{\bar{D}}{S_D/\sqrt{n}} \sim t(n-1)
$$

### 5.4 $\chi^2$ 检验

**5.4.1 拟合优度检验**

检验假设：$H_0$：总体服从某已知分布

检验统计量：
$$
\chi^2 = \sum_{i=1}^k \frac{(n_i - np_i)^2}{np_i} \sim \chi^2(k - r - 1)
$$
其中 $n_i$ 为实际频数，$np_i$ 为理论频数，$r$ 为估计的参数个数。

**5.4.2 独立性检验**

列联表检验，检验两个分类变量是否独立。

检验统计量：
$$
\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(n_{ij} - n_{i.}n_{.j}/n)^2}{n_{i.}n_{.j}/n} \sim \chi^2((r-1)(c-1))
$$

### 5.5 F 检验

**5.5.1 方差齐性检验**

检验假设：$H_0: \sigma_1^2 = \sigma_2^2$

检验统计量：
$$
F = \frac{S_1^2}{S_2^2} \sim F(n_1 - 1, n_2 - 1) \quad (H_0成立时)
$$

**5.5.2 单因素方差分析**

检验 $H_0: \mu_1 = \mu_2 = \dots = \mu_k$

总平方和 $SST = SSE + SSB$，其中
- $SST = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{ij} - \bar{X})^2$
- $SSE = \sum_{i=1}^k \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2$（组内平方和）
- $SSB = \sum_{i=1}^k n_i (\bar{X}_i - \bar{X})^2$（组间平方和）

检验统计量：
$$
F = \frac{SSB/(k-1)}{SSE/(n-k)} \sim F(k-1, n-k)
$$

### 5.6 代码示例

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# 1. 单样本t检验
sample1 = np.random.normal(loc=5.2, scale=2, size=30)
t_stat, p_value = stats.ttest_1samp(sample1, popmean=5)
print("单样本t检验:")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 2. 两独立样本t检验
sample2 = np.random.normal(loc=4.5, scale=2, size=30)
t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=True)
print("\n两独立样本t检验:")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 3. 配对样本t检验
sample3 = np.random.normal(loc=0, scale=1, size=30)
sample4 = sample3 + np.random.normal(loc=0.5, scale=0.5, size=30)
t_stat, p_value = stats.ttest_rel(sample4, sample3)
print("\n配对样本t检验:")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 4. 卡方拟合优度检验
observed = np.array([28, 22, 25, 25])  # 观察频数
expected = np.array([25, 25, 25, 25])  # 期望频数
chi2_stat, p_value = stats.chisquare(observed, expected)
print("\n卡方拟合优度检验:")
print(f"  卡方统计量: {chi2_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 5. 卡方独立性检验
contingency_table = np.array([[20, 15, 10], [10, 15, 20]])
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print("\n卡方独立性检验:")
print(f"  卡方统计量: {chi2_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  自由度: {dof}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 6. F检验（方差齐性检验）
group1 = np.random.normal(5, 2, 30)
group2 = np.random.normal(6, 3, 30)
f_stat, p_value = stats.f_oneway(group1, group2)
print("\nF检验（方差分析）:")
print(f"  F统计量: {f_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否拒绝H0 (α=0.05): {p_value < 0.05}")

# 使用bartlett检验方差齐性
stat, p_value = stats.bartlett(group1, group2)
print("\nBartlett方差齐性检验:")
print(f"  统计量: {stat:.4f}")
print(f"  p值: {p_value:.4f}")
```

---

## 6. 数理统计在AI中的应用

### 6.1 模型评估

**6.1.1 性能指标估计**

设模型在测试集上的预测结果为 $\hat{y}_1, \dots, \hat{y}_n$，真实标签为 $y_1, \dots, y_n$。

- **准确率（Accuracy）**：
  $$
  \hat{\text{Acc}} = \frac{1}{n} \sum_{i=1}^n I(\hat{y}_i = y_i)
  $$

- **精确率（Precision）、召回率（Recall）、F1分数** 等均为统计量，可计算其置信区间。

**6.1.2 交叉验证**

k折交叉验证将数据分为k份，每次用k-1份训练，1份测试，重复k次，得到k个性能估计值，取平均作为最终估计。

### 6.2 置信区间

**定义 6.1（置信区间）**：设 $\theta$ 是总体的未知参数，$X_1, X_2, \dots, X_n$ 是样本，若对于给定的 $\alpha(0 < \alpha < 1)$，存在统计量 $\underline{\theta}$ 和 $\overline{\theta}$，使得
$$
P\{\underline{\theta} < \theta < \overline{\theta}\} = 1 - \alpha
$$
则称随机区间 $(\underline{\theta}, \overline{\theta})$ 为 $\theta$ 的置信水平为 $1 - \alpha$ 的置信区间。

**正态总体参数的置信区间**：

1. $\mu$ 的置信区间（$\sigma^2$ 未知）：
   $$
   \left( \bar{X} - t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}} \right)
   $$

2. $\sigma^2$ 的置信区间：
   $$
   \left( \frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}, \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)} \right)
   $$

### 6.3 A/B测试

**定义 6.2（A/B测试）**：通过随机对照试验比较两个版本（A和B）的性能，判断哪个更好。

**假设检验框架**：
- $H_0$：两版本无差异（$\mu_A = \mu_B$）
- $H_1$：两版本有差异（$\mu_A \neq \mu_B$）

**常用方法**：
- 连续指标：两样本t检验
- 二分类指标（转化率等）：卡方检验或Z检验

### 6.4 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

np.random.seed(42)

# 1. 模型评估与置信区间
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = LogisticRegression(max_iter=1000)

# 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("交叉验证准确率:", scores)
print(f"平均准确率: {np.mean(scores):.4f}")
print(f"标准差: {np.std(scores):.4f}")

# 计算准确率的置信区间
mean_score = np.mean(scores)
std_score = np.std(scores, ddof=1)
n_folds = len(scores)
confidence = 0.95
t_critical = stats.t.ppf((1 + confidence) / 2, n_folds - 1)
margin_of_error = t_critical * (std_score / np.sqrt(n_folds))

confidence_interval = (mean_score - margin_of_error, mean_score + margin_of_error)
print(f"\n{int(confidence*100)}% 置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")

# 2. 单样本均值的置信区间
sample = np.random.normal(loc=5, scale=2, size=100)
mean = np.mean(sample)
std = np.std(sample, ddof=1)
n = len(sample)

confidence = 0.95
t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
margin_of_error = t_critical * (std / np.sqrt(n))

confidence_interval = (mean - margin_of_error, mean + margin_of_error)
print(f"\n单样本均值 {int(confidence*100)}% 置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")

# 3. A/B测试模拟
np.random.seed(42)

# A组: 旧版本
n_a = 1000
conversions_a = np.random.binomial(1, 0.10, n_a)

# B组: 新版本
n_b = 1000
conversions_b = np.random.binomial(1, 0.12, n_b)

rate_a = np.mean(conversions_a)
rate_b = np.mean(conversions_b)

print(f"\nA/B测试结果:")
print(f"  A组转化率: {rate_a:.4f}")
print(f"  B组转化率: {rate_b:.4f}")
print(f"  差异: {rate_b - rate_a:.4f}")

# 使用卡方检验
contingency = np.array([
    [np.sum(conversions_a), n_a - np.sum(conversions_a)],
    [np.sum(conversions_b), n_b - np.sum(conversions_b)]
])
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency, correction=False)
print(f"\n卡方检验:")
print(f"  卡方统计量: {chi2_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否显著 (α=0.05): {p_value < 0.05}")

# 使用Z检验（比例差异）
from statsmodels.stats.proportion import proportions_ztest

count = np.array([np.sum(conversions_a), np.sum(conversions_b)])
nobs = np.array([n_a, n_b])
z_stat, p_value = proportions_ztest(count, nobs)
print(f"\nZ检验:")
print(f"  Z统计量: {z_stat:.4f}")
print(f"  p值: {p_value:.4f}")
print(f"  是否显著 (α=0.05): {p_value < 0.05}")

# 绘制置信区间
fig, ax = plt.subplots(figsize=(10, 6))

# A组
se_a = np.sqrt(rate_a * (1 - rate_a) / n_a)
ci_a = (rate_a - 1.96 * se_a, rate_a + 1.96 * se_a)

# B组
se_b = np.sqrt(rate_b * (1 - rate_b) / n_b)
ci_b = (rate_b - 1.96 * se_b, rate_b + 1.96 * se_b)

ax.bar(['A', 'B'], [rate_a, rate_b], yerr=[rate_a - ci_a[0], rate_b - ci_b[0]], capsize=10, alpha=0.7)
ax.set_ylabel('转化率')
ax.set_title('A/B测试结果与95%置信区间')
ax.grid(axis='y', alpha=0.3)
plt.show()
```

---

## 参考文献

1. 盛骤, 谢式千, 潘承毅. 概率论与数理统计教程[M]. 高等教育出版社.
2. 茆诗松, 程依明, 濮晓龙. 概率论与数理统计教程[M]. 高等教育出版社.
3. Hogg, R. V., McKean, J. W., & Craig, A. T. (2013). Introduction to Mathematical Statistics. Pearson.
4. Wasserman, L. (2013). All of Statistics: A Concise Course in Statistical Inference. Springer.

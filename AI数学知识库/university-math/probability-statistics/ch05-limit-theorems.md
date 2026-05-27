# 第5章 大数定律与中心极限定理

## 5.1 随机变量序列的收敛性

### 5.1.1 依概率收敛

**定义5.1.1（依概率收敛）**

设 $\{X_n\}$ 是随机变量序列，$X$ 是随机变量。若对任意 $\varepsilon > 0$，有
$$\lim_{n \to \infty} P(|X_n - X| \geq \varepsilon) = 0$$
则称 $\{X_n\}$ **依概率收敛**于 $X$，记作 $X_n \xrightarrow{P} X$。

**性质：**

1. 若 $X_n \xrightarrow{P} X$，$Y_n \xrightarrow{P} Y$，则
   - $X_n + Y_n \xrightarrow{P} X + Y$
   - $X_n Y_n \xrightarrow{P} X Y$
   - 若 $P(Y \neq 0) = 0$，则 $X_n / Y_n \xrightarrow{P} X / Y$

2. 若 $g$ 是连续函数，则 $X_n \xrightarrow{P} X$ 蕴含 $g(X_n) \xrightarrow{P} g(X)$。

### 5.1.2 依分布收敛

**定义5.1.2（依分布收敛）**

设 $\{X_n\}$ 是随机变量序列，$F_n(x)$ 是 $X_n$ 的分布函数，$F(x)$ 是随机变量 $X$ 的分布函数。若在 $F(x)$ 的所有连续点 $x$ 处，有
$$\lim_{n \to \infty} F_n(x) = F(x)$$
则称 $\{X_n\}$ **依分布收敛**于 $X$，记作 $X_n \xrightarrow{d} X$。

**性质：**

1. 若 $X_n \xrightarrow{P} X$，则 $X_n \xrightarrow{d} X$。反之不成立。
2. 若 $X_n \xrightarrow{d} c$（常数），则 $X_n \xrightarrow{P} c$。

### 5.1.3 几乎处处收敛

**定义5.1.3（几乎处处收敛）**

设 $\{X_n\}$ 是随机变量序列，$X$ 是随机变量。若
$$P\left(\lim_{n \to \infty} X_n = X\right) = 1$$
则称 $\{X_n\}$ **几乎处处收敛**于 $X$，记作 $X_n \xrightarrow{a.s.} X$。

**性质：**

1. 几乎处处收敛蕴含依概率收敛。
2. 依概率收敛不一定蕴含几乎处处收敛，但存在子序列几乎处处收敛。

## 5.2 大数定律

### 5.2.1 切比雪夫大数定律

**定理5.2.1（切比雪夫大数定律）**

设 $\{X_n\}$ 是独立随机变量序列，若存在常数 $C$，使得 $D(X_i) \leq C$ 对所有 $i$ 成立，则
$$\frac{1}{n} \sum_{i=1}^n (X_i - E(X_i)) \xrightarrow{P} 0$$
即对任意 $\varepsilon > 0$，有
$$\lim_{n \to \infty} P\left(\left|\frac{1}{n} \sum_{i=1}^n X_i - \frac{1}{n} \sum_{i=1}^n E(X_i)\right| \geq \varepsilon\right) = 0$$

**证明思路：**

利用切比雪夫不等式：
$$P\left(\left|\frac{1}{n} \sum_{i=1}^n X_i - \frac{1}{n} \sum_{i=1}^n E(X_i)\right| \geq \varepsilon\right) \leq \frac{D\left(\frac{1}{n} \sum_{i=1}^n X_i\right)}{\varepsilon^2}$$
由于独立性，$D\left(\frac{1}{n} \sum_{i=1}^n X_i\right) = \frac{1}{n^2} \sum_{i=1}^n D(X_i) \leq \frac{C}{n}$，故当 $n \to \infty$ 时，上界趋于0。

### 5.2.2 伯努利大数定律

**定理5.2.2（伯努利大数定律）**

设 $n_A$ 是 $n$ 重伯努利试验中事件 $A$ 发生的次数，$p$ 是每次试验中 $A$ 发生的概率，则
$$\frac{n_A}{n} \xrightarrow{P} p$$

**证明思路：**

令 $X_i = I\{第i次试验A发生\}$，则 $n_A = \sum_{i=1}^n X_i$，且 $E(X_i) = p$，$D(X_i) = p(1-p) \leq 1/4$。由切比雪夫大数定律即得。

**代码示例：**

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_lln(p, n_trials, n_experiments=1000):
    """
    伯努利大数定律演示
    p: 事件发生概率
    n_trials: 试验次数
    n_experiments: 重复实验次数
    """
    proportions = []
    for _ in range(n_experiments):
        trials = np.random.binomial(1, p, n_trials)
        prop = np.mean(trials)
        proportions.append(prop)
    
    return proportions

p = 0.5
n_list = [10, 100, 1000, 10000]

plt.figure(figsize=(12, 8))
for i, n in enumerate(n_list, 1):
    props = bernoulli_lln(p, n)
    plt.subplot(2, 2, i)
    plt.hist(props, bins=30, density=True, alpha=0.7)
    plt.axvline(p, color='red', linestyle='--', label=f'p={p}')
    plt.xlabel('Proportion')
    plt.ylabel('Density')
    plt.title(f'n = {n}')
    plt.legend()
plt.tight_layout()
plt.show()
```

### 5.2.3 辛钦大数定律

**定理5.2.3（辛钦大数定律）**

设 $\{X_n\}$ 是独立同分布随机变量序列，且 $E(X_i) = \mu$ 存在，则
$$\frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu$$

**注：** 辛钦大数定律不要求方差存在，只要求期望存在。

**代码示例：**

```python
import numpy as np

def khinchin_lln(distribution, params, n_samples, n_experiments=1000):
    """
    辛钦大数定律演示
    distribution: 分布函数，如 np.random.normal, np.random.exponential
    params: 分布参数
    n_samples: 样本量
    n_experiments: 重复实验次数
    """
    sample_means = []
    for _ in range(n_experiments):
        samples = distribution(*params, size=n_samples)
        mean = np.mean(samples)
        sample_means.append(mean)
    return sample_means

# 指数分布，期望为1/λ
lambda_param = 2
true_mean = 1 / lambda_param

n_list = [10, 100, 1000]

for n in n_list:
    means = khinchin_lln(np.random.exponential, [true_mean], n)
    print(f"n = {n}:")
    print(f"  样本均值的均值: {np.mean(means):.4f}")
    print(f"  样本均值的标准差: {np.std(means):.4f}")
    print(f"  理论均值: {true_mean:.4f}")
    print()
```

## 5.3 中心极限定理

### 5.3.1 棣莫弗-拉普拉斯定理

**定理5.3.1（棣莫弗-拉普拉斯定理）**

设 $n_A \sim B(n, p)$，则对任意实数 $x$，有
$$\lim_{n \to \infty} P\left(\frac{n_A - np}{\sqrt{np(1-p)}} \leq x\right) = \Phi(x)$$
其中 $\Phi(x)$ 是标准正态分布的分布函数。

**证明思路：**

二项分布可表示为独立伯努利变量之和，利用特征函数方法证明其极限为正态分布特征函数。

**代码示例：**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

def de_moivre_laplace(n, p):
    """
    棣莫弗-拉普拉斯定理演示
    """
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    
    # 正态近似
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    x_norm = np.linspace(0, n, 1000)
    pdf_norm = norm.pdf(x_norm, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, pmf, alpha=0.6, label='Binomial PMF')
    plt.plot(x_norm, pdf_norm, 'r-', lw=2, label='Normal Approximation')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title(f'De Moivre-Laplace Theorem (n={n}, p={p})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

de_moivre_laplace(n=50, p=0.3)
```

### 5.3.2 莱维-林德伯格定理

**定理5.3.2（莱维-林德伯格中心极限定理）**

设 $\{X_n\}$ 是独立同分布随机变量序列，且 $E(X_i) = \mu$，$D(X_i) = \sigma^2 > 0$ 存在，则对任意实数 $x$，有
$$\lim_{n \to \infty} P\left(\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \leq x\right) = \Phi(x)$$

**等价形式：**

$$\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

其中 $\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i$。

**代码示例：**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def levy_lindberg(distribution, params, n_samples, n_experiments=10000):
    """
    莱维-林德伯格中心极限定理演示
    """
    # 计算理论均值和方差
    if distribution == np.random.exponential:
        true_mean = params[0]
        true_std = params[0]
    elif distribution == np.random.uniform:
        true_mean = (params[0] + params[1]) / 2
        true_std = (params[1] - params[0]) / np.sqrt(12)
    elif distribution == np.random.poisson:
        true_mean = params[0]
        true_std = np.sqrt(params[0])
    else:
        raise ValueError("Unknown distribution")
    
    # 生成标准化样本均值
    standardized_means = []
    for _ in range(n_experiments):
        samples = distribution(*params, size=n_samples)
        sample_mean = np.mean(samples)
        z = (sample_mean - true_mean) / (true_std / np.sqrt(n_samples))
        standardized_means.append(z)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(standardized_means, bins=50, density=True, alpha=0.6, label='Standardized Means')
    
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.pdf(x), 'r-', lw=2, label='N(0,1) PDF')
    
    plt.xlabel('z')
    plt.ylabel('Density')
    plt.title(f'Levy-Lindberg CLT (n={n_samples})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 演示：均匀分布 U(0,1)
levy_lindberg(np.random.uniform, [0, 1], n_samples=30)

# 演示：指数分布 Exp(1)
levy_lindberg(np.random.exponential, [1], n_samples=50)
```

## 5.4 在AI中的应用

### 5.4.1 批量梯度下降

**问题背景：**

在机器学习中，我们通常需要最小化经验风险：
$$R_n(\theta) = \frac{1}{n} \sum_{i=1}^n L(f(x_i; \theta), y_i)$$
其中 $L$ 是损失函数。

**大数定律的应用：**

当 $n$ 很大时，经验风险 $R_n(\theta)$ 依概率收敛于期望风险：
$$R(\theta) = E_{(x,y) \sim P}[L(f(x; \theta), y)]$$

**中心极限定理的应用：**

批量梯度下降中，梯度估计
$$g_n(\theta) = \frac{1}{n} \sum_{i=1}^n \nabla_\theta L(f(x_i; \theta), y_i)$$
当 $n$ 较大时，近似服从正态分布，可用于构造置信区间和收敛性分析。

**代码示例：**

```python
import numpy as np
import matplotlib.pyplot as plt

def batch_gradient_descent(X, y, learning_rate=0.01, n_epochs=100, batch_size=32):
    """
    批量梯度下降演示
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    losses = []
    
    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 计算预测和损失
            y_pred = X_batch.dot(theta)
            loss = np.mean((y_pred - y_batch) ** 2)
            losses.append(loss)
            
            # 计算梯度（批量估计）
            gradient = 2 * X_batch.T.dot(y_pred - y_batch) / len(X_batch)
            
            # 更新参数
            theta -= learning_rate * gradient
    
    return theta, losses

# 生成数据
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 1)
y = 3 * X.flatten() + 2 + np.random.randn(n_samples) * 0.5
X_b = np.c_[np.ones(n_samples), X]

# 运行批量梯度下降
theta, losses = batch_gradient_descent(X_b, y, learning_rate=0.1, n_epochs=50, batch_size=64)

print(f"拟合参数: {theta}")
print(f"真实参数: [2, 3]")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Batch Gradient Descent - Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.4.2 统计推断

**置信区间构造：**

对于参数 $\theta$ 的估计量 $\hat{\theta}_n$，若 $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)$，则渐近置信区间为：
$$\hat{\theta}_n \pm z_{\alpha/2} \frac{\hat{\sigma}}{\sqrt{n}}$$

**代码示例：**

```python
import numpy as np
from scipy.stats import norm

def confidence_interval(data, confidence=0.95):
    """
    构造均值的置信区间
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    standard_error = std / np.sqrt(n)
    
    z = norm.ppf((1 + confidence) / 2)
    margin = z * standard_error
    
    return mean, mean - margin, mean + margin

# 示例：从正态分布采样
np.random.seed(42)
data = np.random.normal(5, 2, size=100)

mean, ci_low, ci_high = confidence_interval(data)
print(f"样本均值: {mean:.4f}")
print(f"95% 置信区间: [{ci_low:.4f}, {ci_high:.4f}]")
```

### 5.4.3 蒙特卡洛方法

**基本思想：**

用样本均值估计期望：
$$\theta = E[g(X)] \approx \frac{1}{n} \sum_{i=1}^n g(X_i)$$

**中心极限定理给出误差估计：**
$$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)$$
其中 $\sigma^2 = D[g(X)]$。

**代码示例：计算圆周率 $\pi$**

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(n_samples):
    """
    蒙特卡洛方法估计圆周率π
    """
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside)
    
    # 可视化
    plt.figure(figsize=(8, 8))
    plt.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5, label='Inside')
    plt.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5, label='Outside')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Monte Carlo Estimation of π (n={n_samples})\nEstimate: {pi_estimate:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pi_estimate

# 运行
pi_est = monte_carlo_pi(n_samples=10000)
print(f"π的估计值: {pi_est:.6f}")
print(f"π的真实值: {np.pi:.6f}")

# 收敛性分析
n_list = [100, 1000, 10000, 100000, 1000000]
pi_estimates = []

for n in n_list:
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = (x**2 + y**2) <= 1
    pi_est = 4 * np.mean(inside)
    pi_estimates.append(pi_est)

plt.figure(figsize=(10, 6))
plt.semilogx(n_list, pi_estimates, 'o-', label='Estimate')
plt.axhline(y=np.pi, color='red', linestyle='--', label='True π')
plt.xlabel('Number of Samples')
plt.ylabel('Estimate of π')
plt.title('Convergence of Monte Carlo Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 习题

1. 设 $X_n \xrightarrow{P} X$，证明 $X_n^2 \xrightarrow{P} X^2$。
2. 设 $\{X_n\}$ 独立同服从参数为 $\lambda$ 的泊松分布，用中心极限定理近似计算 $P(\sum_{i=1}^{100} X_i \leq 120)$。
3. 编写蒙特卡洛程序计算积分 $\int_0^1 e^{-x^2} dx$，并给出误差估计。

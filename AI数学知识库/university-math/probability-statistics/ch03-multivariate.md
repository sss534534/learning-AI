# 第三章 多维随机变量

## 1. 多维随机变量及其分布函数

### 1.1 多维随机变量的定义

**定义 3.1.1** 设 $E$ 是一个随机试验，样本空间为 $S = \\{e\\}$，$X_1 = X_1(e), X_2 = X_2(e), \\dots, X_n = X_n(e)$ 是定义在 $S$ 上的随机变量，由它们构成的 $n$ 维向量 $(X_1, X_2, \\dots, X_n)$ 叫做 $n$ 维随机变量或 $n$ 维随机向量。

### 1.2 联合分布函数

**定义 3.1.2** 设 $(X, Y)$ 是二维随机变量，对于任意实数 $x, y$，二元函数：

$$F(x, y) = P\\{(X \\leq x) \\cap (Y \\leq y)\\} \\triangleq P\\{X \\leq x, Y \\leq y\\}$$

称为二维随机变量 $(X, Y)$ 的**联合分布函数**，或简称分布函数。

**基本性质**：
1. $F(x, y)$ 是 $x$ 和 $y$ 的不减函数
2. $0 \\leq F(x, y) \\leq 1$，且 $F(-\\infty, y) = F(x, -\\infty) = F(-\\infty, -\\infty) = 0$，$F(+\\infty, +\\infty) = 1$
3. $F(x, y)$ 关于 $x$ 右连续，关于 $y$ 也右连续
4. 对于任意 $(x_1, y_1), (x_2, y_2)$，其中 $x_1 < x_2, y_1 < y_2$，有：
   $$F(x_2, y_2) - F(x_2, y_1) - F(x_1, y_2) + F(x_1, y_1) \\geq 0$$

### 1.3 边缘分布函数

**定义 3.1.3** 二维随机变量 $(X, Y)$ 作为一个整体，具有分布函数 $F(x, y)$。而 $X$ 和 $Y$ 都是随机变量，各自也有分布函数，将它们分别记为 $F_X(x), F_Y(y)$，依次称为二维随机变量 $(X, Y)$ 关于 $X$ 和关于 $Y$ 的**边缘分布函数**。

**计算公式**：
$$F_X(x) = P\\{X \\leq x\\} = P\\{X \\leq x, Y < +\\infty\\} = F(x, +\\infty)$$
$$F_Y(y) = P\\{Y \\leq y\\} = P\\{X < +\\infty, Y \\leq y\\} = F(+\\infty, y)$$

---

## 2. 离散型多维随机变量的联合分布律、边缘分布律

### 2.1 联合分布律

**定义 3.2.1** 如果二维随机变量 $(X, Y)$ 全部可能取到的值是有限对或可列无限多对，则称 $(X, Y)$ 是**离散型的二维随机变量**。

设 $(X, Y)$ 所有可能取的值为 $(x_i, y_j), i, j = 1, 2, \\dots$，记：

$$P\\{X = x_i, Y = y_j\\} = p_{ij}, \\quad i, j = 1, 2, \\dots$$

则称 $P\\{X = x_i, Y = y_j\\} = p_{ij}$ 为二维离散型随机变量 $(X, Y)$ 的**联合分布律**。

**性质**：
1. $p_{ij} \\geq 0, i, j = 1, 2, \\dots$
2. $\\sum_{i=1}^{\\infty} \\sum_{j=1}^{\\infty} p_{ij} = 1$

### 2.2 边缘分布律

对于离散型二维随机变量 $(X, Y)$，关于 $X$ 的边缘分布律为：

$$P\\{X = x_i\\} = \\sum_{j=1}^{\\infty} p_{ij} \\triangleq p_{i\\cdot}, \\quad i = 1, 2, \\dots$$

关于 $Y$ 的边缘分布律为：

$$P\\{Y = y_j\\} = \\sum_{i=1}^{\\infty} p_{ij} \\triangleq p_{\\cdot j}, \\quad j = 1, 2, \\dots$$

### 2.3 例子：二维两点分布

设随机变量 $X$ 和 $Y$ 只能取 0 和 1 两个值，其联合分布律为：

| $X \\setminus Y$ | 0 | 1 |
|----------------|----|----|
| 0 | $p_{00}$ | $p_{01}$ |
| 1 | $p_{10}$ | $p_{11}$ |

其中 $p_{00} + p_{01} + p_{10} + p_{11} = 1$。

边缘分布律为：
- $P\\{X = 0\\} = p_{00} + p_{01}, P\\{X = 1\\} = p_{10} + p_{11}$
- $P\\{Y = 0\\} = p_{00} + p_{10}, P\\{Y = 1\\} = p_{01} + p_{11}$

### 2.4 代码示例

```python
import numpy as np

# 定义联合分布律
def create_joint_distribution(probs):
    """
    创建二维离散型随机变量的联合分布律
    
    参数:
        probs: 字典 {(x, y): p}
    """
    return probs

# 计算边缘分布律
def marginal_distribution(joint_dist, variable='X'):
    """
    计算边缘分布律
    
    参数:
        joint_dist: 联合分布律
        variable: 'X' 或 'Y'
    """
    marginals = {}
    for (x, y), p in joint_dist.items():
        key = x if variable == 'X' else y
        if key in marginals:
            marginals[key] += p
        else:
            marginals[key] = p
    return marginals

# 例子：二维两点分布
joint_dist = {(0, 0): 0.4, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.1}
print("联合分布律:", joint_dist)
print("X的边缘分布:", marginal_distribution(joint_dist, 'X'))
print("Y的边缘分布:", marginal_distribution(joint_dist, 'Y'))
```

---

## 3. 连续型多维随机变量的联合概率密度、边缘概率密度

### 3.1 联合概率密度

**定义 3.3.1** 对于二维随机变量 $(X, Y)$ 的分布函数 $F(x, y)$，如果存在非负可积函数 $f(x, y)$，使对于任意 $x, y$ 有：

$$F(x, y) = \\int_{-\\infty}^{y} \\int_{-\\infty}^{x} f(u, v) \\, du \\, dv$$

则称 $(X, Y)$ 是**连续型二维随机变量**，函数 $f(x, y)$ 称为二维随机变量 $(X, Y)$ 的**联合概率密度**。

**性质**：
1. $f(x, y) \\geq 0$
2. $\\int_{-\\infty}^{+\\infty} \\int_{-\\infty}^{+\\infty} f(x, y) \\, dx \\, dy = 1$
3. 设 $G$ 是 $xOy$ 平面上的区域，则点 $(X, Y)$ 落在 $G$ 内的概率为：
   $$P\\{(X, Y) \\in G\\} = \\iint_G f(x, y) \\, dx \\, dy$$
4. 若 $f(x, y)$ 在点 $(x, y)$ 连续，则有：
   $$\\frac{\\partial^2 F(x, y)}{\\partial x \\partial y} = f(x, y)$$

### 3.2 边缘概率密度

**定义 3.3.2** 对于连续型随机变量 $(X, Y)$，设它的概率密度为 $f(x, y)$，则 $X$ 的概率密度 $f_X(x)$ 称为关于 $X$ 的边缘概率密度，其计算公式为：

$$f_X(x) = \\int_{-\\infty}^{+\\infty} f(x, y) \\, dy$$

同理，关于 $Y$ 的边缘概率密度：

$$f_Y(y) = \\int_{-\\infty}^{+\\infty} f(x, y) \\, dx$$

### 3.3 例子：二维均匀分布

设 $G$ 是平面上的有界区域，其面积为 $A$。若二维随机变量 $(X, Y)$ 具有概率密度：

$$f(x, y) = \\begin{cases} 
\\dfrac{1}{A}, & (x, y) \\in G \\\\
0, & \\text{其他}
\\end{cases}$$

则称 $(X, Y)$ 在 $G$ 上服从**均匀分布**。

### 3.4 例子：二维正态分布

**定义 3.3.3** 设二维随机变量 $(X, Y)$ 的概率密度为：

$$
f(x, y) = \\frac{1}{2\\pi\\sigma_1\\sigma_2\\sqrt{1-\\rho^2}} \\exp\\left\\{ -\\frac{1}{2(1-\\rho^2)} \\left[ \\frac{(x-\\mu_1)^2}{\\sigma_1^2} - 2\\rho\\frac{(x-\\mu_1)(y-\\mu_2)}{\\sigma_1\\sigma_2} + \\frac{(y-\\mu_2)^2}{\\sigma_2^2} \\right] \\right\\}
$$

其中 $\\mu_1, \\mu_2, \\sigma_1 > 0, \\sigma_2 > 0, -1 < \\rho < 1$ 都是常数，则称 $(X, Y)$ 服从参数为 $\\mu_1, \\mu_2, \\sigma_1, \\sigma_2, \\rho$ 的**二维正态分布**，记为 $(X, Y) \\sim N(\\mu_1, \\mu_2, \\sigma_1^2, \\sigma_2^2, \\rho)$。

**边缘分布**：
- $X \\sim N(\\mu_1, \\sigma_1^2)$
- $Y \\sim N(\\mu_2, \\sigma_2^2)$

### 3.5 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 二维正态分布示例
def bivariate_normal_example():
    # 参数设置
    mu = [0, 0]
    sigma = [[1, 0.5], [0.5, 1]]
    
    # 创建网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算概率密度
    rv = multivariate_normal(mu, sigma)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(contour)
    plt.title('二维正态分布概率密度')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(1, 2, 2)
    plt.plot(x, rv.pdf(np.dstack((x, np.zeros_like(x)))), label='X的边缘分布')
    plt.plot(y, rv.pdf(np.dstack((np.zeros_like(y), y))), label='Y的边缘分布')
    plt.title('边缘概率密度')
    plt.xlabel('值')
    plt.ylabel('概率密度')
    plt.legend()
    
    plt.show()

bivariate_normal_example()
```

---

## 4. 条件分布、随机变量的独立性

### 4.1 离散型随机变量的条件分布律

**定义 3.4.1** 设 $(X, Y)$ 是二维离散型随机变量，对于固定的 $j$，若 $P\\{Y = y_j\\} > 0$，则称：

$$P\\{X = x_i | Y = y_j\\} = \\frac{P\\{X = x_i, Y = y_j\\}}{P\\{Y = y_j\\}} = \\frac{p_{ij}}{p_{\\cdot j}}, \\quad i = 1, 2, \\dots$$

为在 $Y = y_j$ 条件下随机变量 $X$ 的条件分布律。

同理，在 $X = x_i$ 条件下随机变量 $Y$ 的条件分布律：

$$P\\{Y = y_j | X = x_i\\} = \\frac{p_{ij}}{p_{i\\cdot}}, \\quad j = 1, 2, \\dots$$

### 4.2 连续型随机变量的条件概率密度

**定义 3.4.2** 设二维连续型随机变量 $(X, Y)$ 的概率密度为 $f(x, y)$，$(X, Y)$ 关于 $Y$ 的边缘概率密度为 $f_Y(y)$。若对于固定的 $y$，$f_Y(y) > 0$，则称：

$$f_{X|Y}(x|y) = \\frac{f(x, y)}{f_Y(y)}$$

为在 $Y = y$ 条件下 $X$ 的条件概率密度。

同理，在 $X = x$ 条件下 $Y$ 的条件概率密度：

$$f_{Y|X}(y|x) = \\frac{f(x, y)}{f_X(x)}$$

条件分布函数：

$$F_{X|Y}(x|y) = P\\{X \\leq x | Y = y\\} = \\int_{-\\infty}^{x} \\frac{f(u, y)}{f_Y(y)} \\, du$$

### 4.3 随机变量的独立性

**定义 3.4.3** 设 $F(x, y)$ 及 $F_X(x), F_Y(y)$ 分别是二维随机变量 $(X, Y)$ 的分布函数及边缘分布函数。若对于所有 $x, y$ 有：

$$P\\{X \\leq x, Y \\leq y\\} = P\\{X \\leq x\\}P\\{Y \\leq y\\}$$

即

$$F(x, y) = F_X(x)F_Y(y)$$

则称随机变量 $X$ 和 $Y$ 是**相互独立的**。

**判别准则**：
1. **离散型**：$X$ 和 $Y$ 相互独立 $\\iff p_{ij} = p_{i\\cdot} p_{\\cdot j}$ 对所有 $i, j$ 成立
2. **连续型**：$X$ 和 $Y$ 相互独立 $\\iff f(x, y) = f_X(x)f_Y(y)$ 几乎处处成立

**定理 3.4.1** 若 $(X, Y)$ 服从二维正态分布，则 $X$ 和 $Y$ 相互独立的充要条件是 $\\rho = 0$。

### 4.4 代码示例

```python
import numpy as np

# 检查离散型随机变量的独立性
def check_independence_discrete(joint_dist):
    """
    检查离散型二维随机变量是否独立
    
    参数:
        joint_dist: 联合分布律 {(x, y): p}
    """
    # 计算边缘分布
    x_marginal = {}
    y_marginal = {}
    for (x, y), p in joint_dist.items():
        x_marginal[x] = x_marginal.get(x, 0) + p
        y_marginal[y] = y_marginal.get(y, 0) + p
    
    # 检查是否满足 p_ij = p_i * p_j
    is_independent = True
    for (x, y), p_joint in joint_dist.items():
        p_independent = x_marginal[x] * y_marginal[y]
        if not np.isclose(p_joint, p_independent):
            is_independent = False
            print(f"({x}, {y})处不独立: 联合概率={p_joint}, 独立乘积={p_independent}")
            break
    
    return is_independent

# 测试独立性
independent_dist = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}
dependent_dist = {(0, 0): 0.4, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.1}

print("第一个分布是否独立:", check_independence_discrete(independent_dist))
print("第二个分布是否独立:", check_independence_discrete(dependent_dist))
```

---

## 5. 多维随机变量函数的分布

### 5.1 离散型情形

设 $(X, Y)$ 是离散型随机变量，其分布律为 $P\\{X = x_i, Y = y_j\\} = p_{ij}$，则 $Z = g(X, Y)$ 的分布律为：

$$P\\{Z = z_k\\} = \\sum_{g(x_i, y_j) = z_k} p_{ij}, \\quad k = 1, 2, \\dots$$

### 5.2 连续型情形 - 和的分布

设 $(X, Y)$ 是连续型随机变量，其概率密度为 $f(x, y)$，则 $Z = X + Y$ 的概率密度为：

$$f_Z(z) = \\int_{-\\infty}^{+\\infty} f(z - y, y) \\, dy$$

或

$$f_Z(z) = \\int_{-\\infty}^{+\\infty} f(x, z - x) \\, dx$$

当 $X$ 和 $Y$ 相互独立时，上述公式成为卷积公式：

$$f_Z(z) = \\int_{-\\infty}^{+\\infty} f_X(z - y)f_Y(y) \\, dy = \\int_{-\\infty}^{+\\infty} f_X(x)f_Y(z - x) \\, dx$$

### 5.3 连续型情形 - 商的分布

设 $(X, Y)$ 是连续型随机变量，其概率密度为 $f(x, y)$，则 $Z = Y/X$ 的概率密度为：

$$f_{Y/X}(z) = \\int_{-\\infty}^{+\\infty} |x|f(x, xz) \\, dx$$

### 5.4 极值分布

设 $X_1, X_2, \\dots, X_n$ 相互独立，且它们的分布函数分别为 $F_{X_i}(x)$，记：

$$M = \\max(X_1, X_2, \\dots, X_n)$$
$$N = \\min(X_1, X_2, \\dots, X_n)$$

则它们的分布函数分别为：

$$F_M(z) = P\\{M \\leq z\\} = \\prod_{i=1}^n F_{X_i}(z)$$
$$F_N(z) = P\\{N \\leq z\\} = 1 - \\prod_{i=1}^n [1 - F_{X_i}(z)]$$

当 $X_1, \\dots, X_n$ 同分布时：
$$F_M(z) = [F_X(z)]^n$$
$$F_N(z) = 1 - [1 - F_X(z)]^n$$

### 5.5 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 两个独立正态分布之和的分布
def sum_of_normals_example():
    # X ~ N(0, 1), Y ~ N(1, 4)
    x_samples = np.random.normal(0, 1, 10000)
    y_samples = np.random.normal(1, 2, 10000)
    z_samples = x_samples + y_samples
    
    # 理论上 Z ~ N(1, 5)
    z = np.linspace(-5, 8, 100)
    theoretical_pdf = norm.pdf(z, 1, np.sqrt(5))
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(z_samples, bins=50, density=True, alpha=0.6, label='样本直方图')
    plt.plot(z, theoretical_pdf, 'r-', linewidth=2, label='理论分布 N(1, 5)')
    plt.title('两个独立正态分布之和的分布')
    plt.xlabel('Z = X + Y')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True)
    plt.show()

sum_of_normals_example()
```

---

## 6. 协方差、相关系数、协方差矩阵、多元正态分布

### 6.1 协方差

**定义 3.6.1** 设 $(X, Y)$ 是二维随机变量，若 $E\\{[X - E(X)][Y - E(Y)]\\}$ 存在，则称它为随机变量 $X$ 与 $Y$ 的**协方差**，记为 $\\text{Cov}(X, Y)$，即：

$$\\text{Cov}(X, Y) = E\\{[X - E(X)][Y - E(Y)]\\}$$

**计算公式**：
$$\\text{Cov}(X, Y) = E(XY) - E(X)E(Y)$$

**性质**：
1. $\\text{Cov}(X, Y) = \\text{Cov}(Y, X)$
2. $\\text{Cov}(aX, bY) = ab\\text{Cov}(X, Y)$，其中 $a, b$ 是常数
3. $\\text{Cov}(X_1 + X_2, Y) = \\text{Cov}(X_1, Y) + \\text{Cov}(X_2, Y)$
4. $D(X + Y) = D(X) + D(Y) + 2\\text{Cov}(X, Y)$

### 6.2 相关系数

**定义 3.6.2** 设 $(X, Y)$ 是二维随机变量，若 $D(X) > 0, D(Y) > 0$，则称：

$$\\rho_{XY} = \\frac{\\text{Cov}(X, Y)}{\\sqrt{D(X)}\\sqrt{D(Y)}}$$

为随机变量 $X$ 与 $Y$ 的**相关系数**。

**性质**：
1. $|\\rho_{XY}| \\leq 1$
2. $|\\rho_{XY}| = 1$ 的充要条件是，存在常数 $a, b$ 使得 $P\\{Y = aX + b\\} = 1$
3. 若 $\\rho_{XY} = 0$，称 $X$ 与 $Y$ 不相关

**重要结论**：若 $X$ 与 $Y$ 相互独立，则 $X$ 与 $Y$ 不相关；反之不成立。

### 6.3 协方差矩阵

**定义 3.6.3** 设 $n$ 维随机变量 $(X_1, X_2, \\dots, X_n)$ 的二阶中心矩都存在，记：

$$c_{ij} = \\text{Cov}(X_i, X_j) = E\\{[X_i - E(X_i)][X_j - E(X_j)]\\}, \\quad i, j = 1, 2, \\dots, n$$

则称矩阵：

$$C = \\begin{pmatrix}
c_{11} & c_{12} & \\dots & c_{1n} \\\\
c_{21} & c_{22} & \\dots & c_{2n} \\\\
\\vdots & \\vdots & & \\vdots \\\\
c_{n1} & c_{n2} & \\dots & c_{nn}
\\end{pmatrix}$$

为 $n$ 维随机变量 $(X_1, X_2, \\dots, X_n)$ 的**协方差矩阵**。

**性质**：
1. 协方差矩阵是对称矩阵：$c_{ij} = c_{ji}$
2. 协方差矩阵是非负定矩阵

### 6.4 多元正态分布

**定义 3.6.4** 设 $n$ 维随机变量 $\\boldsymbol{X} = (X_1, X_2, \\dots, X_n)^T$ 的概率密度为：

$$f(x_1, x_2, \\dots, x_n) = \\frac{1}{(2\\pi)^{n/2}|C|^{1/2}} \\exp\\left\\{ -\\frac{1}{2}(\\boldsymbol{x} - \\boldsymbol{\\mu})^T C^{-1} (\\boldsymbol{x} - \\boldsymbol{\\mu}) \\right\\}$$

其中 $\\boldsymbol{\\mu} = (\\mu_1, \\mu_2, \\dots, \\mu_n)^T$ 是均值向量，$C$ 是协方差矩阵，则称 $\\boldsymbol{X}$ 服从**$n$ 元正态分布**，记为 $\\boldsymbol{X} \\sim N(\\boldsymbol{\\mu}, C)$。

**重要性质**：
1. $n$ 元正态分布的任一线性组合是一元正态分布
2. $n$ 元正态变量的任一部分仍服从正态分布
3. $n$ 元正态变量的分量相互独立的充要条件是它们两两不相关

### 6.5 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 协方差和相关系数计算
def calculate_covariance_correlation(X, Y):
    """
    计算协方差和相关系数
    """
    n = len(X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    
    # 计算协方差
    covariance = np.sum((X - mean_X) * (Y - mean_Y)) / n
    
    # 计算标准差
    std_X = np.sqrt(np.sum((X - mean_X) ** 2) / n)
    std_Y = np.sqrt(np.sum((Y - mean_Y) ** 2) / n)
    
    # 计算相关系数
    correlation = covariance / (std_X * std_Y)
    
    return covariance, correlation

# 生成数据并验证
np.random.seed(42)
X = np.random.normal(0, 1, 1000)
Y = 2 * X + np.random.normal(0, 0.5, 1000)  # Y = 2X + 噪声

cov, corr = calculate_covariance_correlation(X, Y)
print(f"协方差: {cov:.4f}")
print(f"相关系数: {corr:.4f}")

# 协方差矩阵
data = np.column_stack((X, Y))
cov_matrix = np.cov(data, rowvar=False, bias=True)
print("协方差矩阵:")
print(cov_matrix)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.5)
plt.title(f'Y = 2X + 噪声 (相关系数 ρ = {corr:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
```

---

## 7. 多维随机变量在AI中的应用

### 7.1 多元正态分布与异常检测

在异常检测中，我们可以利用多元正态分布来建模正常数据的分布，然后计算新样本在该分布下的概率，概率低于阈值时判定为异常。

**算法步骤**：
1. 用正常数据估计均值向量 $\\boldsymbol{\\mu}$ 和协方差矩阵 $C$
2. 对于新样本 $\\boldsymbol{x}$，计算概率密度 $p(\\boldsymbol{x})$
3. 若 $p(\\boldsymbol{x}) < \\varepsilon$，则判定为异常

### 7.2 高斯混合模型(GMM)

高斯混合模型是一种概率模型，用于表示总体分布由多个高斯分布混合而成：

$$p(\\boldsymbol{x}) = \\sum_{k=1}^K \\pi_k \\mathcal{N}(\\boldsymbol{x} | \\boldsymbol{\\mu}_k, C_k)$$

其中 $\\pi_k \\geq 0$ 是混合系数，且 $\\sum_{k=1}^K \\pi_k = 1$。

**EM算法**：用于估计GMM的参数，包括E步和M步的迭代。

### 7.3 多标签分类

在多标签分类中，每个样本可以同时属于多个类别。我们可以使用多元正态分布来建模标签之间的相关性。

### 7.4 代码示例：高斯混合模型

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ = make_blobs(n_samples=1000, centers=3, random_state=42, cluster_std=1.0)

# 拟合GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# 可视化结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('原始数据')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title('GMM聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.colorbar(scatter, label='聚类标签')

# 画出高斯分布的等高线
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T

for i in range(3):
    rv = multivariate_normal(gmm.means_[i], gmm.covariances_[i])
    Z = rv.pdf(XX).reshape(X_grid.shape)
    plt.contour(X_grid, Y_grid, Z, colors='black', alpha=0.5)

plt.show()

print("GMM参数:")
print("均值向量:")
print(gmm.means_)
print("混合系数:")
print(gmm.weights_)
```

### 7.5 代码示例：多元正态分布异常检测

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def multivariate_gaussian_anomaly_detection(X_train, X_test, epsilon=0.01):
    """
    基于多元正态分布的异常检测
    
    参数:
        X_train: 训练数据(正常样本)
        X_test: 测试数据
        epsilon: 概率阈值
    """
    # 估计均值和协方差
    mu = np.mean(X_train, axis=0)
    Sigma = np.cov(X_train.T)
    
    # 计算训练数据的概率密度
    rv = multivariate_normal(mu, Sigma)
    p_train = rv.pdf(X_train)
    
    # 如果没有提供epsilon，使用训练数据的10分位数
    if epsilon is None:
        epsilon = np.percentile(p_train, 10)
    
    # 检测测试数据中的异常
    p_test = rv.pdf(X_test)
    anomalies = p_test < epsilon
    
    return anomalies, p_test, mu, Sigma, epsilon

# 生成数据
np.random.seed(42)
# 正常数据
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 500)
# 异常数据
X_anomalies = np.array([[-3, 3], [3, -3], [3.5, 3.5], [-3.5, -3.5]])
X_test = np.vstack([X_normal[:100], X_anomalies])

# 异常检测
anomalies, p_test, mu, Sigma, epsilon = multivariate_gaussian_anomaly_detection(
    X_normal, X_test, epsilon=None
)

# 可视化
plt.figure(figsize=(10, 8))

# 绘制等高线
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X_grid, Y_grid = np.meshgrid(x, y)
rv = multivariate_normal(mu, Sigma)
Z = rv.pdf(np.dstack((X_grid, Y_grid)))
contour = plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='viridis', alpha=0.5)
plt.colorbar(contour, label='概率密度')

# 绘制正常点和异常点
normal_points = X_test[~anomalies]
anomaly_points = X_test[anomalies]
plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='正常', alpha=0.6)
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='red', s=100, marker='x', label='异常')

plt.title('多元正态分布异常检测')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True)
plt.show()

print(f"检测到 {np.sum(anomalies)} 个异常点")
```

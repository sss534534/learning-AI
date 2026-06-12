# 第一章 极限与连续


## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


## 1. 数列极限

### 1.1 数列极限的定义（ε-N语言）

**定义**（数列极限）：设 $\{a_n\}$ 为一数列，如果存在常数 $a$，对于任意给定的正数 $\varepsilon$（无论它多么小），总存在正整数 $N$，使得当 $n > N$ 时，不等式
$$|a_n - a| < \varepsilon$$
都成立，那么就称常数 $a$ 是数列 $\{a_n\}$ 的极限，或者称数列 $\{a_n\}$ 收敛于 $a$，记作
$$\lim_{n \to \infty} a_n = a$$
或
$$a_n \to a \quad (n \to \infty)$$

如果不存在这样的常数 $a$，就说数列 $\{a_n\}$ 没有极限，或者说数列 $\{a_n\}$ 是发散的。

**几何解释**：对于任意给定的 $\varepsilon > 0$，在开区间 $(a-\varepsilon, a+\varepsilon)$ 之外数列 $\{a_n\}$ 的项至多只有有限个（$N$ 个）。

### 1.2 数列极限的性质

**定理1.1（唯一性）**：如果数列 $\{a_n\}$ 收敛，那么它的极限唯一。

**证明思路**：反证法。假设存在两个不同的极限 $a$ 和 $b$，取 $\varepsilon = \frac{|a-b|}{2}$，根据极限定义会导出矛盾。

**定理1.2（有界性）**：如果数列 $\{a_n\}$ 收敛，那么数列 $\{a_n\}$ 一定有界。

**证明思路**：取 $\varepsilon = 1$，则存在 $N$，当 $n > N$ 时，$|a_n - a| < 1$，即 $|a_n| < |a| + 1$。而前 $N$ 项是有限个数，因此可以取最大值作为界。

**定理1.3（保号性）**：如果 $\lim_{n \to \infty} a_n = a$，且 $a > 0$（或 $a < 0$），那么存在正整数 $N$，当 $n > N$ 时，都有 $a_n > 0$（或 $a_n < 0$）。

**推论**：如果从某项起 $a_n \geq 0$（或 $a_n \leq 0$），且 $\lim_{n \to \infty} a_n = a$，那么 $a \geq 0$（或 $a \leq 0$）。

**定理1.4（四则运算）**：设 $\lim_{n \to \infty} a_n = a$，$\lim_{n \to \infty} b_n = b$，则
1. $\lim_{n \to \infty} (a_n \pm b_n) = a \pm b$
2. $\lim_{n \to \infty} (a_n \cdot b_n) = a \cdot b$
3. $\lim_{n \to \infty} \frac{a_n}{b_n} = \frac{a}{b} \quad (b \neq 0)$

### 1.3 数列收敛准则

#### 1.3.1 夹逼定理

**定理1.5（夹逼定理）**：如果数列 $\{a_n\}, \{b_n\}, \{c_n\}$ 满足下列条件：
1. 存在正整数 $N_0$，当 $n > N_0$ 时，有 $b_n \leq a_n \leq c_n$
2. $\lim_{n \to \infty} b_n = \lim_{n \to \infty} c_n = a$

那么数列 $\{a_n\}$ 的极限存在，且 $\lim_{n \to \infty} a_n = a$。

**例子**：证明 $\lim_{n \to \infty} \sqrt[n]{n} = 1$

**证明**：令 $a_n = \sqrt[n]{n} - 1$，则 $a_n \geq 0$，且
$$n = (1 + a_n)^n \geq 1 + na_n + \frac{n(n-1)}{2}a_n^2 > \frac{n(n-1)}{2}a_n^2$$
因此
$$0 \leq a_n < \sqrt{\frac{2}{n-1}}$$
由夹逼定理，$\lim_{n \to \infty} a_n = 0$，故 $\lim_{n \to \infty} \sqrt[n]{n} = 1$。

#### 1.3.2 单调有界收敛准则

**定理1.6（单调有界收敛准则）**：单调有界数列必有极限。

**说明**：
- 单调递增且有上界的数列必有极限
- 单调递减且有下界的数列必有极限

**例子**：证明数列 $a_n = \left(1 + \frac{1}{n}\right)^n$ 收敛。

**证明思路**：利用二项式展开证明数列单调递增且有上界3。

### 1.4 数列极限的代码示例

```python
def sequence_limit_example():
    """示例：计算数列极限"""
    
    # 数列1: a_n = (1 + 1/n)^n，收敛到e ≈ 2.71828
    def a1(n):
        return (1 + 1/n) ** n
    
    # 数列2: a_n = sqrt(n^(1/n)) = n^(1/(2n))，收敛到1
    def a2(n):
        return n ** (1/(2*n))
    
    # 数列3: a_n = sin(n)/n，收敛到0
    def a3(n):
        import math
        return math.sin(n) / n
    
    # 计算不同n值处的数列项
    n_values = [10, 100, 1000, 10000, 100000]
    
    print("数列1: a_n = (1 + 1/n)^n")
    for n in n_values:
        print(f"n = {n:6d}, a_n = {a1(n):.10f}")
    
    print("\n数列2: a_n = n^(1/(2n))")
    for n in n_values:
        print(f"n = {n:6d}, a_n = {a2(n):.10f}")
    
    print("\n数列3: a_n = sin(n)/n")
    for n in n_values:
        print(f"n = {n:6d}, a_n = {a3(n):.10f}")

if __name__ == "__main__":
    sequence_limit_example()
```

---

## 2. 函数极限

### 2.1 函数极限的定义（ε-δ语言）

**定义**（函数在有限点的极限）：设函数 $f(x)$ 在点 $x_0$ 的某一去心邻域内有定义。如果存在常数 $A$，对于任意给定的正数 $\varepsilon$（无论它多么小），总存在正数 $\delta$，使得当 $x$ 满足不等式 $0 < |x - x_0| < \delta$ 时，对应的函数值 $f(x)$ 都满足不等式
$$|f(x) - A| < \varepsilon$$
那么常数 $A$ 就叫做函数 $f(x)$ 当 $x \to x_0$ 时的极限，记作
$$\lim_{x \to x_0} f(x) = A$$
或
$$f(x) \to A \quad (x \to x_0)$$

**定义**（函数在无穷远处的极限）：设函数 $f(x)$ 当 $|x|$ 大于某一正数时有定义。如果存在常数 $A$，对于任意给定的正数 $\varepsilon$，总存在正数 $X$，使得当 $x$ 满足不等式 $|x| > X$ 时，对应的函数值 $f(x)$ 都满足不等式
$$|f(x) - A| < \varepsilon$$
那么常数 $A$ 就叫做函数 $f(x)$ 当 $x \to \infty$ 时的极限，记作
$$\lim_{x \to \infty} f(x) = A$$

### 2.2 单侧极限

**定义**（左极限）：如果当 $x \to x_0^-$（即 $x$ 从 $x_0$ 的左侧趋于 $x_0$）时，函数 $f(x)$ 趋于 $A$，则称 $A$ 为 $f(x)$ 在 $x_0$ 处的左极限，记作
$$\lim_{x \to x_0^-} f(x) = A \quad \text{或} \quad f(x_0^-) = A$$

**定义**（右极限）：如果当 $x \to x_0^+$（即 $x$ 从 $x_0$ 的右侧趋于 $x_0$）时，函数 $f(x)$ 趋于 $A$，则称 $A$ 为 $f(x)$ 在 $x_0$ 处的右极限，记作
$$\lim_{x \to x_0^+} f(x) = A \quad \text{或} \quad f(x_0^+) = A$$

**定理2.1**：函数 $f(x)$ 当 $x \to x_0$ 时极限存在的充分必要条件是左极限和右极限都存在并且相等，即
$$\lim_{x \to x_0} f(x) = A \iff f(x_0^-) = f(x_0^+) = A$$

### 2.3 函数极限的性质

**定理2.2（唯一性）**：如果 $\lim_{x \to x_0} f(x)$ 存在，那么这极限唯一。

**定理2.3（局部有界性）**：如果 $\lim_{x \to x_0} f(x) = A$，那么存在常数 $M > 0$ 和 $\delta > 0$，使得当 $0 < |x - x_0| < \delta$ 时，有 $|f(x)| \leq M$。

**定理2.4（局部保号性）**：如果 $\lim_{x \to x_0} f(x) = A$，且 $A > 0$（或 $A < 0$），那么存在常数 $\delta > 0$，使得当 $0 < |x - x_0| < \delta$ 时，有 $f(x) > 0$（或 $f(x) < 0$）。

**定理2.5（四则运算）**：设 $\lim_{x \to x_0} f(x) = A$，$\lim_{x \to x_0} g(x) = B$，则
1. $\lim_{x \to x_0} [f(x) \pm g(x)] = A \pm B$
2. $\lim_{x \to x_0} [f(x) \cdot g(x)] = A \cdot B$
3. $\lim_{x \to x_0} \frac{f(x)}{g(x)} = \frac{A}{B} \quad (B \neq 0)$

**定理2.6（复合函数的极限运算法则）**：设函数 $y = f[g(x)]$ 是由函数 $u = g(x)$ 与函数 $y = f(u)$ 复合而成，$f[g(x)]$ 在点 $x_0$ 的某去心邻域内有定义。如果 $\lim_{x \to x_0} g(x) = u_0$，$\lim_{u \to u_0} f(u) = A$，且存在 $\delta_0 > 0$，当 $x \in \mathring{U}(x_0, \delta_0)$ 时，有 $g(x) \neq u_0$，则
$$\lim_{x \to x_0} f[g(x)] = \lim_{u \to u_0} f(u) = A$$

### 2.4 函数极限的代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def function_limit_example():
    """示例：函数极限的可视化"""
    
    # 函数1: f(x) = sin(x)/x，x→0时极限为1
    def f1(x):
        return np.sin(x) / x if x != 0 else 1
    
    # 函数2: f(x) = (1 + 1/x)^x，x→∞时极限为e
    def f2(x):
        return (1 + 1/x) ** x
    
    # 函数3: f(x) = (e^x - 1)/x，x→0时极限为1
    def f3(x):
        return (np.exp(x) - 1) / x if x != 0 else 1
    
    # 绘制函数图像
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: sin(x)/x
    x1 = np.linspace(-2*np.pi, 2*np.pi, 400)
    y1 = [f1(x) for x in x1]
    axes[0].plot(x1, y1, label='f(x) = sin(x)/x')
    axes[0].axhline(y=1, color='r', linestyle='--', label='极限 y=1')
    axes[0].set_title('lim(x→0) sin(x)/x = 1')
    axes[0].legend()
    axes[0].grid(True)
    
    # 图2: (1+1/x)^x
    x2 = np.linspace(1, 50, 400)
    y2 = [f2(x) for x in x2]
    axes[1].plot(x2, y2, label='f(x) = (1 + 1/x)^x')
    axes[1].axhline(y=np.e, color='r', linestyle='--', label=f'极限 y=e≈{np.e:.5f}')
    axes[1].set_title('lim(x→∞) (1 + 1/x)^x = e')
    axes[1].legend()
    axes[1].grid(True)
    
    # 图3: (e^x - 1)/x
    x3 = np.linspace(-2, 2, 400)
    y3 = [f3(x) for x in x3]
    axes[2].plot(x3, y3, label='f(x) = (e^x - 1)/x')
    axes[2].axhline(y=1, color='r', linestyle='--', label='极限 y=1')
    axes[2].set_title('lim(x→0) (e^x - 1)/x = 1')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('function_limits.png')
    print("函数极限图像已保存为 function_limits.png")
    
    # 数值计算
    print("\n数值验证：")
    print(f"f1(0.1) = {f1(0.1):.10f}")
    print(f"f1(0.01) = {f1(0.01):.10f}")
    print(f"f2(1000) = {f2(1000):.10f}")
    print(f"e = {np.e:.10f}")

if __name__ == "__main__":
    function_limit_example()
```

---

## 3. 函数连续性

### 3.1 函数连续性的定义

**定义**（函数在点连续）：设函数 $y = f(x)$ 在点 $x_0$ 的某一邻域内有定义，如果
$$\lim_{x \to x_0} f(x) = f(x_0)$$
则称函数 $f(x)$ 在点 $x_0$ 连续。

**等价定义**（增量形式）：设函数 $y = f(x)$ 在点 $x_0$ 的某一邻域内有定义，令 $\Delta x = x - x_0$，$\Delta y = f(x) - f(x_0)$，如果
$$\lim_{\Delta x \to 0} \Delta y = 0$$
则称函数 $f(x)$ 在点 $x_0$ 连续。

**定义**（单侧连续）：
- 如果 $\lim_{x \to x_0^-} f(x) = f(x_0)$，则称 $f(x)$ 在点 $x_0$ 左连续
- 如果 $\lim_{x \to x_0^+} f(x) = f(x_0)$，则称 $f(x)$ 在点 $x_0$ 右连续

**定理3.1**：函数 $f(x)$ 在点 $x_0$ 连续的充分必要条件是它在点 $x_0$ 左连续且右连续。

**定义**（区间上连续）：
- 如果函数 $f(x)$ 在开区间 $(a, b)$ 内每一点都连续，则称 $f(x)$ 在开区间 $(a, b)$ 内连续
- 如果函数 $f(x)$ 在开区间 $(a, b)$ 内连续，且在左端点 $a$ 右连续，在右端点 $b$ 左连续，则称 $f(x)$ 在闭区间 $[a, b]$ 上连续

### 3.2 间断点分类

**定义**（间断点）：设函数 $f(x)$ 在点 $x_0$ 的某去心邻域内有定义。如果函数 $f(x)$ 有下列三种情形之一：
1. 在 $x = x_0$ 没有定义
2. 虽在 $x = x_0$ 有定义，但 $\lim_{x \to x_0} f(x)$ 不存在
3. 虽在 $x = x_0$ 有定义，且 $\lim_{x \to x_0} f(x)$ 存在，但 $\lim_{x \to x_0} f(x) \neq f(x_0)$

则函数 $f(x)$ 在点 $x_0$ 为不连续，而点 $x_0$ 称为函数 $f(x)$ 的间断点。

**第一类间断点**：设 $x_0$ 是函数 $f(x)$ 的间断点，如果 $f(x)$ 在 $x_0$ 处的左、右极限都存在，则称 $x_0$ 是第一类间断点。
- **可去间断点**：左、右极限都存在且相等，但不等于 $f(x_0)$ 或 $f(x)$ 在 $x_0$ 无定义
- **跳跃间断点**：左、右极限都存在但不相等

**第二类间断点**：除第一类间断点以外的其他间断点。
- 无穷间断点：极限为无穷大
- 振荡间断点：极限不存在但也不是无穷大

**例子**：
1. $f(x) = \frac{\sin x}{x}$ 在 $x = 0$ 处是可去间断点
2. $f(x) = \begin{cases} x-1, & x < 0 \\ 0, & x = 0 \\ x+1, & x > 0 \end{cases}$ 在 $x = 0$ 处是跳跃间断点
3. $f(x) = \frac{1}{x}$ 在 $x = 0$ 处是无穷间断点
4. $f(x) = \sin\frac{1}{x}$ 在 $x = 0$ 处是振荡间断点

### 3.3 一致连续性

**定义**（一致连续）：设函数 $f(x)$ 在区间 $I$ 上有定义。如果对于任意给定的正数 $\varepsilon$，总存在正数 $\delta$，使得对于区间 $I$ 上的任意两点 $x_1, x_2$，当 $|x_1 - x_2| < \delta$ 时，就有
$$|f(x_1) - f(x_2)| < \varepsilon$$
那么称函数 $f(x)$ 在区间 $I$ 上是一致连续的。

**说明**：
- 一致连续比连续要求更强
- 在某区间一致连续的函数必定在该区间上连续
- 但连续函数未必一致连续（如 $f(x) = \frac{1}{x}$ 在 $(0, 1)$ 上连续但不一致连续）

**定理3.2（Cantor定理）**：如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，那么它在该区间上一致连续。

### 3.4 函数连续性的代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def continuity_examples():
    """示例：函数连续性与间断点"""
    
    # 连续函数示例: f(x) = sin(x)
    def continuous_func(x):
        return np.sin(x)
    
    # 可去间断点: f(x) = sin(x)/x
    def removable_discontinuity(x):
        return np.sin(x) / x if x != 0 else np.nan
    
    # 跳跃间断点: 分段函数
    def jump_discontinuity(x):
        return np.where(x < 0, x - 1, np.where(x > 0, x + 1, 0))
    
    # 无穷间断点: f(x) = 1/x
    def infinite_discontinuity(x):
        return np.where(x != 0, 1/x, np.nan)
    
    # 绘制图像
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 连续函数
    x1 = np.linspace(-2*np.pi, 2*np.pi, 400)
    y1 = continuous_func(x1)
    axes[0, 0].plot(x1, y1)
    axes[0, 0].set_title('连续函数: f(x) = sin(x)')
    axes[0, 0].grid(True)
    
    # 图2: 可去间断点
    x2 = np.linspace(-3, 3, 400)
    y2 = [removable_discontinuity(x) for x in x2]
    axes[0, 1].plot(x2, y2)
    axes[0, 1].scatter([0], [1], color='red', s=50, label='补充定义处')
    axes[0, 1].set_title('可去间断点: f(x) = sin(x)/x')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 图3: 跳跃间断点
    x3 = np.linspace(-2, 2, 400)
    y3 = jump_discontinuity(x3)
    axes[1, 0].plot(x3, y3)
    axes[1, 0].scatter([0], [0], color='red', s=50)
    axes[1, 0].set_title('跳跃间断点: 分段函数')
    axes[1, 0].grid(True)
    
    # 图4: 无穷间断点
    x4 = np.linspace(-2, 2, 400)
    y4 = infinite_discontinuity(x4)
    axes[1, 1].plot(x4, y4)
    axes[1, 1].set_ylim([-10, 10])
    axes[1, 1].set_title('无穷间断点: f(x) = 1/x')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('continuity_examples.png')
    print("函数连续性图像已保存为 continuity_examples.png")

if __name__ == "__main__":
    continuity_examples()
```

---

## 4. 闭区间上连续函数的性质

### 4.1 最值定理

**定理4.1（有界性定理）**：如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，那么它在该区间上有界，即存在常数 $M > 0$，使得对于一切 $x \in [a, b]$，都有 $|f(x)| \leq M$。

**定理4.2（最值定理）**：如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，那么它在该区间上一定有最大值和最小值，即存在 $\xi, \eta \in [a, b]$，使得
$$f(\xi) = \max_{x \in [a, b]} f(x), \quad f(\eta) = \min_{x \in [a, b]} f(x)$$

### 4.2 介值定理与零点定理

**定理4.3（介值定理）**：如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，且 $f(a) \neq f(b)$，则对于 $f(a)$ 与 $f(b)$ 之间的任意一个数 $C$，在开区间 $(a, b)$ 内至少有一点 $\xi$，使得
$$f(\xi) = C$$

**几何解释**：连续曲线 $y = f(x)$ 与水平直线 $y = C$ 至少有一个交点。

**定理4.4（零点定理）**：如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，且 $f(a)$ 与 $f(b)$ 异号（即 $f(a) \cdot f(b) < 0$），则在开区间 $(a, b)$ 内至少有一点 $\xi$，使得
$$f(\xi) = 0$$

**推论**：在闭区间上连续的函数必取得介于最大值 $M$ 和最小值 $m$ 之间的任何值。

**例子**：证明方程 $x^3 - 4x^2 + 1 = 0$ 在区间 $(0, 1)$ 内至少有一个根。

**证明**：设 $f(x) = x^3 - 4x^2 + 1$，则 $f(x)$ 在 $[0, 1]$ 上连续，且
$$f(0) = 1 > 0, \quad f(1) = 1 - 4 + 1 = -2 < 0$$
由零点定理，存在 $\xi \in (0, 1)$，使得 $f(\xi) = 0$。

### 4.3 闭区间上连续函数性质的代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def closed_interval_properties():
    """示例：闭区间上连续函数的性质"""
    
    # 示例函数: f(x) = x^3 - 4x^2 + 1
    def f(x):
        return x**3 - 4*x**2 + 1
    
    # 二分法求根
    def bisection_method(f, a, b, tol=1e-8, max_iter=100):
        """使用二分法求方程f(x)=0在区间(a,b)内的根"""
        if f(a) * f(b) >= 0:
            raise ValueError("函数在区间端点处同号")
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            
            if abs(fc) < tol or (b - a) / 2 < tol:
                return c, i+1
            
            if f(a) * fc < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2, max_iter
    
    # 绘制函数图像
    x = np.linspace(-1, 4, 400)
    y = f(x)
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label='f(x) = x^3 - 4x^2 + 1')
    plt.axhline(y=0, color='black', linestyle='--')
    
    # 标记区间 [0, 1]
    x_interval = np.linspace(0, 1, 100)
    y_interval = f(x_interval)
    plt.fill_between(x_interval, y_interval, alpha=0.3, label='区间 [0, 1]')
    plt.scatter([0, 1], [f(0), f(1)], color='red', s=100, zorder=5)
    plt.text(0, f(0)+0.2, f'f(0)={f(0)}', ha='center')
    plt.text(1, f(1)+0.2, f'f(1)={f(1)}', ha='center')
    
    # 用二分法求根
    root, iterations = bisection_method(f, 0, 1)
    plt.scatter([root], [0], color='green', s=100, zorder=5, label=f'根 x≈{root:.8f}')
    
    plt.title('零点定理示例')
    plt.legend()
    plt.grid(True)
    plt.savefig('zero_point_theorem.png')
    print("零点定理图像已保存为 zero_point_theorem.png")
    print(f"使用二分法，迭代 {iterations} 次后求得根为 {root:.10f}")
    print(f"验证: f({root:.10f}) = {f(root):.10f}")
    
    # 找闭区间上的最大值和最小值
    x2 = np.linspace(-1, 2, 1000)
    y2 = f(x2)
    max_val = np.max(y2)
    min_val = np.min(y2)
    max_x = x2[np.argmax(y2)]
    min_x = x2[np.argmin(y2)]
    
    print(f"\n在区间 [-1, 2] 上：")
    print(f"最大值: f({max_x:.4f}) = {max_val:.6f}")
    print(f"最小值: f({min_x:.4f}) = {min_val:.6f}")

if __name__ == "__main__":
    closed_interval_properties()
```

---

## 5. 重要极限与等价无穷小

### 5.1 两个重要极限

**第一个重要极限**：
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

**证明思路**：利用几何图形（单位圆）构造不等式 $\cos x < \frac{\sin x}{x} < 1$，再用夹逼定理。

**第二个重要极限**：
$$\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e$$
或
$$\lim_{x \to 0} (1 + x)^{\frac{1}{x}} = e$$

其中 $e \approx 2.718281828459045$ 是自然对数的底数。

### 5.2 无穷小与无穷大

**定义**（无穷小）：如果函数 $f(x)$ 当 $x \to x_0$（或 $x \to \infty$）时的极限为零，那么称函数 $f(x)$ 为当 $x \to x_0$（或 $x \to \infty$）时的无穷小。

**定义**（无穷大）：如果当 $x \to x_0$（或 $x \to \infty$）时，函数 $f(x)$ 的绝对值无限增大，那么称函数 $f(x)$ 为当 $x \to x_0$（或 $x \to \infty$）时的无穷大，记作
$$\lim_{x \to x_0} f(x) = \infty \quad \text{或} \quad \lim_{x \to \infty} f(x) = \infty$$

**定理5.1**：在自变量的同一变化过程中，如果 $f(x)$ 为无穷大，则 $\frac{1}{f(x)}$ 为无穷小；反之，如果 $f(x)$ 为无穷小，且 $f(x) \neq 0$，则 $\frac{1}{f(x)}$ 为无穷大。

### 5.3 无穷小的比较

**定义**：设 $\alpha$ 和 $\beta$ 都是在同一个自变量的变化过程中的无穷小，且 $\alpha \neq 0$。

1. 如果 $\lim \frac{\beta}{\alpha} = 0$，则称 $\beta$ 是比 $\alpha$ 高阶的无穷小，记作 $\beta = o(\alpha)$
2. 如果 $\lim \frac{\beta}{\alpha} = \infty$，则称 $\beta$ 是比 $\alpha$ 低阶的无穷小
3. 如果 $\lim \frac{\beta}{\alpha} = c \neq 0$，则称 $\beta$ 与 $\alpha$ 是同阶无穷小
4. 如果 $\lim \frac{\beta}{\alpha^k} = c \neq 0$，则称 $\beta$ 是关于 $\alpha$ 的 $k$ 阶无穷小
5. 如果 $\lim \frac{\beta}{\alpha} = 1$，则称 $\beta$ 与 $\alpha$ 是等价无穷小，记作 $\beta \sim \alpha$

### 5.4 常用等价无穷小

当 $x \to 0$ 时，常用的等价无穷小：

1. $\sin x \sim x$
2. $\tan x \sim x$
3. $\arcsin x \sim x$
4. $\arctan x \sim x$
5. $1 - \cos x \sim \frac{1}{2}x^2$
6. $\ln(1 + x) \sim x$
7. $e^x - 1 \sim x$
8. $a^x - 1 \sim x\ln a$
9. $(1 + x)^\alpha - 1 \sim \alpha x$

**定理5.2（等价无穷小代换定理）**：设 $\alpha \sim \alpha'$，$\beta \sim \beta'$，且 $\lim \frac{\beta'}{\alpha'}$ 存在，则
$$\lim \frac{\beta}{\alpha} = \lim \frac{\beta'}{\alpha'}$$

**例子**：计算 $\lim_{x \to 0} \frac{\tan x - \sin x}{x^3}$

**解**：
$$\tan x - \sin x = \tan x(1 - \cos x)$$
当 $x \to 0$ 时，$\tan x \sim x$，$1 - \cos x \sim \frac{1}{2}x^2$，因此
$$\lim_{x \to 0} \frac{\tan x - \sin x}{x^3} = \lim_{x \to 0} \frac{x \cdot \frac{1}{2}x^2}{x^3} = \frac{1}{2}$$

### 5.5 重要极限与等价无穷小的代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

def important_limits():
    """示例：重要极限与等价无穷小"""
    
    # 第一个重要极限: sin(x)/x → 1 (x→0)
    x = np.linspace(-1, 1, 400)
    x[x == 0] = np.finfo(float).eps  # 避免除以0
    y1 = np.sin(x) / x
    
    # 等价无穷小示例: sin(x) ~ x, tan(x) ~ x, 1 - cos(x) ~ 0.5x²
    y_sin = np.sin(x)
    y_tan = np.tan(x)
    y_1cos = 1 - np.cos(x)
    y_x = x
    y_x2 = 0.5 * x**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 第一个重要极限
    axes[0].plot(x, y1, label='sin(x)/x')
    axes[0].axhline(y=1, color='r', linestyle='--', label='极限 y=1')
    axes[0].set_title('第一个重要极限: lim(x→0) sin(x)/x = 1')
    axes[0].legend()
    axes[0].grid(True)
    
    # 图2: 等价无穷小
    axes[1].plot(x, y_sin, label='sin(x)')
    axes[1].plot(x, y_tan, label='tan(x)')
    axes[1].plot(x, y_x, label='x', linestyle='--')
    axes[1].plot(x, y_1cos, label='1 - cos(x)')
    axes[1].plot(x, y_x2, label='0.5x²', linestyle='--')
    axes[1].set_title('等价无穷小 (x→0)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].set_ylim([-0.5, 0.5])
    
    plt.tight_layout()
    plt.savefig('important_limits.png')
    print("重要极限图像已保存为 important_limits.png")
    
    # 数值验证
    print("\n数值验证等价无穷小:")
    for h in [0.1, 0.01, 0.001, 0.0001]:
        print(f"\nx = {h}:")
        print(f"  sin(x)/x = {np.sin(h)/h:.10f}")
        print(f"  tan(x)/x = {np.tan(h)/h:.10f}")
        print(f"  (1 - cos(x))/(0.5x²) = {(1 - np.cos(h))/(0.5*h**2):.10f}")
        print(f"  (e^x - 1)/x = {(np.exp(h) - 1)/h:.10f}")
        print(f"  ln(1+x)/x = {np.log(1+h)/h:.10f}")

if __name__ == "__main__":
    important_limits()
```

---

## 6. 数学分析在AI中的应用

### 6.1 梯度消失问题

**问题背景**：在深度神经网络训练中，当网络层数很深时，梯度在反向传播过程中会逐渐消失（变得趋近于0），导致网络无法学习。

**数学原理**：假设我们有一个 $L$ 层的神经网络，每层的激活函数为 $\sigma$，权重为 $W_i$。根据链式法则，损失函数 $L$ 对第 $l$ 层权重 $W_l$ 的梯度为：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial y_L} \prod_{k=l}^{L-1} \sigma'(z_k) W_{k+1}$$

其中 $z_k = W_k a_{k-1} + b_k$ 是第 $k$ 层的线性输出。

如果使用Sigmoid激活函数，其导数 $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \in (0, 0.25]$。当网络很深时，多个小于1的因子相乘会导致梯度迅速衰减至0，即出现**梯度消失**。

**代码示例**：梯度消失问题的数值演示

```python
import numpy as np

def vanishing_gradient_demo():
    """演示梯度消失问题"""
    
    # Sigmoid激活函数及其导数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # ReLU激活函数及其导数
    def relu(x):
        return np.maximum(0, x)
    
    def relu_deriv(x):
        return np.where(x > 0, 1, 0)
    
    # 模拟10层网络的梯度传播
    n_layers = 10
    x = np.random.randn()  # 初始输入
    
    print("=== 使用Sigmoid激活函数 ===")
    gradient = 1.0
    z = x
    for i in range(n_layers):
        gradient *= sigmoid_deriv(z) * 0.5  # 假设权重的影响为0.5
        z = sigmoid(z) * 0.5  # 前向传播
        print(f"第{i+1}层后的梯度大小: {gradient:.10f}")
    
    print("\n=== 使用ReLU激活函数 ===")
    gradient = 1.0
    z = x
    for i in range(n_layers):
        gradient *= relu_deriv(z) * 0.5
        z = relu(z) * 0.5
        print(f"第{i+1}层后的梯度大小: {gradient:.10f}")

if __name__ == "__main__":
    vanishing_gradient_demo()
```

### 6.2 激活函数的连续性与可导性

**激活函数的要求**：
1. 非线性：否则多层网络等价于单层网络
2. 连续可导：保证梯度可以反向传播
3. 不饱和：避免梯度消失
4. 计算高效

**常见激活函数**：

1. **Sigmoid函数**：
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   - 优点：输出在 $(0, 1)$ 之间，可解释为概率
   - 缺点：饱和时梯度消失；输出不以0为中心

2. **Tanh函数**：
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
   - 输出在 $(-1, 1)$ 之间，以0为中心
   - 仍然存在梯度消失问题

3. **ReLU函数**：
   $$\text{ReLU}(x) = \max(0, x)$$
   - 优点：计算简单；缓解梯度消失问题；加速训练
   - 缺点：存在"死神经元"问题（某些神经元永远不会被激活）

4. **Leaky ReLU**：
   $$\text{LeakyReLU}(x) = \begin{cases} x, & x \geq 0 \\ \alpha x, & x < 0 \end{cases}$$
   - 解决了"死神经元"问题

**代码示例**：激活函数的可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def activation_functions():
    """可视化常见激活函数"""
    
    x = np.linspace(-5, 5, 400)
    
    # Sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Tanh
    def tanh(x):
        return np.tanh(x)
    
    # ReLU
    def relu(x):
        return np.maximum(0, x)
    
    # Leaky ReLU
    def leaky_relu(x, alpha=0.1):
        return np.where(x >= 0, x, alpha * x)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x))
    axes[0, 0].set_title('Sigmoid 函数')
    axes[0, 0].grid(True)
    
    # Tanh
    axes[0, 1].plot(x, tanh(x))
    axes[0, 1].set_title('Tanh 函数')
    axes[0, 1].grid(True)
    
    # ReLU
    axes[1, 0].plot(x, relu(x))
    axes[1, 0].set_title('ReLU 函数')
    axes[1, 0].grid(True)
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu(x))
    axes[1, 1].set_title('Leaky ReLU 函数 (α=0.1)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    print("激活函数图像已保存为 activation_functions.png")

if __name__ == "__main__":
    activation_functions()
```

### 6.3 数值稳定性与初始化

**问题**：在神经网络中，权重初始化对训练非常重要。如果初始权重过大，会导致激活值饱和；如果太小，会导致梯度信号太小。

**Xavier/Glorot初始化**：对于使用Sigmoid或Tanh激活函数的层，权重初始化应满足：
$$\text{Var}(W_i) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
其中 $n_{\text{in}}$ 和 $n_{\text{out}}$ 分别是输入和输出的维度。

**He初始化**：对于使用ReLU激活函数的层，推荐使用：
$$\text{Var}(W_i) = \frac{2}{n_{\text{in}}}$$

**数学原理**：这样的初始化可以保证在网络的前向和反向传播过程中，每一层的激活值和梯度的方差保持相对稳定，避免信号的指数级增长或衰减。

**代码示例**：权重初始化的演示

```python
import numpy as np
import matplotlib.pyplot as plt

def weight_initialization():
    """演示不同的权重初始化方法"""
    
    def simulate_forward_pass(init_method, n_layers=10, n_neurons=512):
        """模拟前向传播，观察激活值的变化"""
        np.random.seed(42)
        
        activations = []
        x = np.random.randn(n_neurons)  # 输入
        activations.append(x.copy())
        
        for i in range(n_layers):
            # 根据不同方法初始化权重
            if init_method == 'small_random':
                W = np.random.randn(n_neurons, n_neurons) * 0.01
            elif init_method == 'xavier':
                W = np.random.randn(n_neurons, n_neurons) * np.sqrt(1.0 / n_neurons)
            elif init_method == 'he':
                W = np.random.randn(n_neurons, n_neurons) * np.sqrt(2.0 / n_neurons)
            else:
                raise ValueError("Unknown initialization method")
            
            # 前向传播：使用tanh激活函数
            z = np.dot(W, x)
            x = np.tanh(z)
            activations.append(x.copy())
        
        return activations
    
    # 模拟三种初始化方法
    methods = ['small_random', 'xavier', 'he']
    results = {}
    
    for method in methods:
        results[method] = simulate_forward_pass(method)
    
    # 绘制激活值的分布变化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, method in enumerate(methods):
        activations = results[method]
        means = [np.mean(act) for act in activations]
        stds = [np.std(act) for act in activations]
        layers = range(len(activations))
        
        axes[i].plot(layers, means, label='均值')
        axes[i].plot(layers, stds, label='标准差')
        axes[i].set_title(f'{method} 初始化')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xlabel('层数')
        axes[i].set_ylabel('值')
    
    plt.tight_layout()
    plt.savefig('weight_initialization.png')
    print("权重初始化实验图像已保存为 weight_initialization.png")
    
    print("\n不同初始化方法的激活值标准差：")
    for method in methods:
        print(f"\n{method}:")
        for layer, act in enumerate(results[method]):
            print(f"  层 {layer}: std = {np.std(act):.6f}")

if __name__ == "__main__":
    weight_initialization()
```

### 6.4 极限思想在优化算法中的应用

**梯度下降法**：梯度下降是机器学习中最基本的优化算法，其核心思想是利用极限和导数的概念来找到目标函数的最小值。

**梯度下降的迭代公式**：
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

其中：
- $\theta_t$ 是第 $t$ 步的参数
- $\eta$ 是学习率
- $\nabla_\theta L(\theta_t)$ 是损失函数在 $\theta_t$ 处的梯度

**收敛性**：当学习率选择合适，且损失函数是凸函数时，梯度下降保证收敛到全局最优解。这个收敛过程可以看作是数列极限的一个实例。

**代码示例**：梯度下降优化演示

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_demo():
    """演示梯度下降优化过程"""
    
    # 目标函数: f(x) = x²
    def f(x):
        return x**2
    
    # 梯度: f'(x) = 2x
    def grad_f(x):
        return 2 * x
    
    # 梯度下降算法
    def gradient_descent(initial_x, learning_rate, num_iterations):
        x_history = [initial_x]
        f_history = [f(initial_x)]
        
        x = initial_x
        for i in range(num_iterations):
            x = x - learning_rate * grad_f(x)
            x_history.append(x)
            f_history.append(f(x))
        
        return x_history, f_history
    
    # 运行梯度下降
    initial_x = 5.0
    learning_rate = 0.1
    num_iterations = 50
    
    x_history, f_history = gradient_descent(initial_x, learning_rate, num_iterations)
    
    # 绘制优化过程
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 函数和迭代过程
    x = np.linspace(-6, 6, 400)
    axes[0].plot(x, f(x), label='f(x) = x²')
    axes[0].scatter(x_history, f_history, c='red', s=50, label='迭代点')
    axes[0].plot(x_history, f_history, 'r--', alpha=0.5)
    axes[0].set_title('梯度下降优化过程')
    axes[0].legend()
    axes[0].grid(True)
    
    # 图2: 收敛曲线
    axes[1].plot(range(len(f_history)), f_history)
    axes[1].set_title('损失函数收敛曲线')
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('f(x)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('gradient_descent.png')
    print("梯度下降图像已保存为 gradient_descent.png")
    
    print("\n梯度下降收敛过程:")
    for i in range(0, num_iterations+1, 5):
        print(f"迭代 {i:2d}: x = {x_history[i]:.6f}, f(x) = {f_history[i]:.6f}")

if __name__ == "__main__":
    gradient_descent_demo()
```

---

## 参考文献

1. 华东师范大学数学系. 《数学分析》（第五版）. 高等教育出版社, 2019.
2. 同济大学数学系. 《高等数学》（第七版）. 高等教育出版社, 2014.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV.
5. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.

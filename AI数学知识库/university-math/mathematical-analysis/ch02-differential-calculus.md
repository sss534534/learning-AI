# 第二章 微分学

## §1 导数的概念

### 1.1 导数的定义

**定义1（导数）** 设函数 $y = f(x)$ 在点 $x_0$ 的某邻域内有定义，当自变量 $x$ 在 $x_0$ 处取得增量 $\Delta x$（点 $x_0 + \Delta x$ 仍在该邻域内）时，相应地，函数取得增量 $\Delta y = f(x_0 + \Delta x) - f(x_0)$。如果极限

$$
\lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}
$$

存在，则称函数 $y = f(x)$ 在点 $x_0$ 处**可导**，并称这个极限为函数 $y = f(x)$ 在点 $x_0$ 处的**导数**，记为 $f'(x_0)$，即

$$
f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}
$$

也可记作 $y'|_{x=x_0}$，$\frac{dy}{dx}|_{x=x_0}$ 或 $\frac{df(x)}{dx}|_{x=x_0}$。

**定义2（单侧导数）**
- 左导数：$f'_-(x_0) = \lim_{\Delta x \to 0^-} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$
- 右导数：$f'_+(x_0) = \lim_{\Delta x \to 0^+} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$

**定理1** 函数 $f(x)$ 在点 $x_0$ 处可导的充要条件是左导数 $f'_-(x_0)$ 和右导数 $f'_+(x_0)$ 都存在且相等。

### 1.2 导数的几何意义

函数 $y = f(x)$ 在点 $x_0$ 处的导数 $f'(x_0)$ 在几何上表示曲线 $y = f(x)$ 在点 $M(x_0, f(x_0))$ 处的切线斜率，即

$$
f'(x_0) = \tan \alpha
$$

其中 $\alpha$ 是切线的倾角。

**切线方程**：$y - f(x_0) = f'(x_0)(x - x_0)$

**法线方程**：$y - f(x_0) = -\frac{1}{f'(x_0)}(x - x_0)$（当 $f'(x_0) \neq 0$ 时）

### 1.3 函数的可导性与连续性的关系

**定理2** 如果函数 $y = f(x)$ 在点 $x_0$ 处可导，则它在点 $x_0$ 处连续。

**证明思路**：由可导定义，$\lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = f'(x_0)$ 存在，因此 $\Delta y = \frac{\Delta y}{\Delta x} \cdot \Delta x \to 0$（当 $\Delta x \to 0$ 时），故函数在 $x_0$ 处连续。

注：连续不一定可导，例如 $f(x) = |x|$ 在 $x = 0$ 处连续但不可导。

### 1.4 基本初等函数的导数公式

1. $(C)' = 0$（$C$ 为常数）
2. $(x^\mu)' = \mu x^{\mu-1}$
3. $(\sin x)' = \cos x$
4. $(\cos x)' = -\sin x$
5. $(\tan x)' = \sec^2 x$
6. $(\cot x)' = -\csc^2 x$
7. $(\sec x)' = \sec x \tan x$
8. $(\csc x)' = -\csc x \cot x$
9. $(a^x)' = a^x \ln a$（$a > 0, a \neq 1$）
10. $(e^x)' = e^x$
11. $(\log_a x)' = \frac{1}{x \ln a}$（$a > 0, a \neq 1$）
12. $(\ln x)' = \frac{1}{x}$
13. $(\arcsin x)' = \frac{1}{\sqrt{1-x^2}}$
14. $(\arccos x)' = -\frac{1}{\sqrt{1-x^2}}$
15. $(\arctan x)' = \frac{1}{1+x^2}$
16. $(\text{arccot}\, x)' = -\frac{1}{1+x^2}$

## §2 函数的求导法则

### 2.1 函数的和、差、积、商的求导法则

**定理3** 设 $u = u(x)$，$v = v(x)$ 都在点 $x$ 处可导，则
1. $(u \pm v)' = u' \pm v'$
2. $(uv)' = u'v + uv'$
3. $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$（$v \neq 0$）

**证明思路**（以乘积法则为例）：
$$
\begin{aligned}
(uv)' &= \lim_{\Delta x \to 0} \frac{u(x+\Delta x)v(x+\Delta x) - u(x)v(x)}{\Delta x} \\
&= \lim_{\Delta x \to 0} \left[ \frac{u(x+\Delta x)-u(x)}{\Delta x} v(x+\Delta x) + u(x) \frac{v(x+\Delta x)-v(x)}{\Delta x} \right] \\
&= u'v + uv'
\end{aligned}
$$

**例1** 求 $f(x) = x^3 + 2x^2 - 3x + 1$ 的导数。
**解**：$f'(x) = 3x^2 + 4x - 3$

### 2.2 反函数的求导法则

**定理4** 设 $x = f(y)$ 在区间 $I_y$ 内单调、可导且 $f'(y) \neq 0$，则它的反函数 $y = f^{-1}(x)$ 在对应区间 $I_x$ 内也可导，且

$$
[f^{-1}(x)]' = \frac{1}{f'(y)} \quad \text{或} \quad \frac{dy}{dx} = \frac{1}{\frac{dx}{dy}}
$$

**证明思路**：由反函数定义，$\Delta x = f(y+\Delta y) - f(y)$，当 $\Delta x \to 0$ 时，$\Delta y \to 0$，因此

$$
\frac{\Delta y}{\Delta x} = \frac{1}{\frac{\Delta x}{\Delta y}} \implies \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = \frac{1}{\lim_{\Delta y \to 0} \frac{\Delta x}{\Delta y}} = \frac{1}{f'(y)}
$$

### 2.3 复合函数的求导法则

**定理5（链式法则）** 设 $y = f(u)$，$u = g(x)$，且 $g(x)$ 在点 $x$ 处可导，$f(u)$ 在对应点 $u = g(x)$ 处可导，则复合函数 $y = f[g(x)]$ 在点 $x$ 处可导，且

$$
\frac{dy}{dx} = f'(u) \cdot g'(x) \quad \text{或} \quad \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

**证明思路**：由可导定义，$\Delta y = f'(u)\Delta u + \alpha \Delta u$，其中 $\alpha \to 0$（当 $\Delta u \to 0$ 时），因此

$$
\frac{\Delta y}{\Delta x} = f'(u) \frac{\Delta u}{\Delta x} + \alpha \frac{\Delta u}{\Delta x}
$$

令 $\Delta x \to 0$，取极限即得。

**例2** 求 $y = e^{\sin x^2}$ 的导数。
**解**：令 $y = e^u$，$u = \sin v$，$v = x^2$，则

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx} = e^u \cdot \cos v \cdot 2x = 2x e^{\sin x^2} \cos x^2
$$

### 2.4 高阶导数

**定义3** 函数 $y = f(x)$ 的导数 $y' = f'(x)$ 仍然是 $x$ 的函数，我们把 $y' = f'(x)$ 的导数叫做函数 $y = f(x)$ 的**二阶导数**，记作 $y''$，$f''(x)$ 或 $\frac{d^2 y}{dx^2}$。

类似地，二阶导数的导数叫做三阶导数，记为 $y'''$，$f'''(x)$ 或 $\frac{d^3 y}{dx^3}$。一般地，$n$ 阶导数记为 $y^{(n)}$，$f^{(n)}(x)$ 或 $\frac{d^n y}{dx^n}$。

**例3** 求 $y = e^x$ 的 $n$ 阶导数。
**解**：$y' = e^x$，$y'' = e^x$，...，$y^{(n)} = e^x$

**莱布尼茨公式** 设 $u(x)$，$v(x)$ 都有 $n$ 阶导数，则

$$
(uv)^{(n)} = \sum_{k=0}^{n} \binom{n}{k} u^{(n-k)} v^{(k)}
$$

其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 为二项式系数。

## §3 微分

### 3.1 微分的定义

**定义4** 设函数 $y = f(x)$ 在某区间内有定义，$x_0$ 及 $x_0 + \Delta x$ 在这区间内，如果函数的增量

$$
\Delta y = f(x_0 + \Delta x) - f(x_0)
$$

可表示为

$$
\Delta y = A \Delta x + o(\Delta x)
$$

其中 $A$ 是不依赖于 $\Delta x$ 的常数，而 $o(\Delta x)$ 是比 $\Delta x$ 高阶的无穷小，那么称函数 $y = f(x)$ 在点 $x_0$ 是**可微的**，而 $A \Delta x$ 叫做函数 $y = f(x)$ 在点 $x_0$ 相应于自变量增量 $\Delta x$ 的**微分**，记作 $dy$，即

$$
dy = A \Delta x
$$

### 3.2 可微与可导的关系

**定理6** 函数 $y = f(x)$ 在点 $x_0$ 可微的充要条件是函数 $y = f(x)$ 在点 $x_0$ 可导，且当 $f(x)$ 在点 $x_0$ 可微时，其微分一定是

$$
dy = f'(x_0) \Delta x
$$

**证明思路**：
- 必要性：若可微，则 $\Delta y = A \Delta x + o(\Delta x)$，两边除以 $\Delta x$ 得 $\frac{\Delta y}{\Delta x} = A + \frac{o(\Delta x)}{\Delta x}$，令 $\Delta x \to 0$，得 $f'(x_0) = A$。
- 充分性：若可导，则 $\lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = f'(x_0)$，因此 $\frac{\Delta y}{\Delta x} = f'(x_0) + \alpha$，其中 $\alpha \to 0$（当 $\Delta x \to 0$ 时），故 $\Delta y = f'(x_0) \Delta x + \alpha \Delta x = f'(x_0) \Delta x + o(\Delta x)$，即可微。

通常把自变量的增量 $\Delta x$ 称为自变量的微分，记为 $dx$，即 $dx = \Delta x$，因此微分可写成

$$
dy = f'(x) dx \quad \text{或} \quad \frac{dy}{dx} = f'(x)
$$

这说明导数是函数的微分与自变量的微分之商，因此导数也称为**微商**。

### 3.3 微分的几何意义

函数 $y = f(x)$ 在点 $x_0$ 的微分 $dy = f'(x_0) \Delta x$ 在几何上表示曲线 $y = f(x)$ 在点 $M(x_0, f(x_0))$ 处的切线上的纵坐标的增量。

当 $|\Delta x|$ 很小时，$dy \approx \Delta y$，即用切线段近似代替曲线段，这是**以直代曲**的思想。

### 3.4 基本初等函数的微分公式

由 $dy = f'(x) dx$，可得到基本初等函数的微分公式：

1. $d(C) = 0$
2. $d(x^\mu) = \mu x^{\mu-1} dx$
3. $d(\sin x) = \cos x dx$
4. $d(\cos x) = -\sin x dx$
5. $d(\tan x) = \sec^2 x dx$
6. $d(\cot x) = -\csc^2 x dx$
7. $d(a^x) = a^x \ln a dx$
8. $d(e^x) = e^x dx$
9. $d(\log_a x) = \frac{1}{x \ln a} dx$
10. $d(\ln x) = \frac{1}{x} dx$

### 3.5 微分在近似计算中的应用

当 $|\Delta x|$ 很小时，有近似公式：

$$
f(x_0 + \Delta x) \approx f(x_0) + f'(x_0) \Delta x
$$

特别地，当 $x_0 = 0$ 时，有

$$
f(x) \approx f(0) + f'(0) x
$$

常用近似公式（当 $|x|$ 很小时）：
1. $\sin x \approx x$
2. $\tan x \approx x$
3. $e^x \approx 1 + x$
4. $\ln(1+x) \approx x$
5. $(1+x)^\alpha \approx 1 + \alpha x$

**例4** 计算 $\sqrt{1.02}$ 的近似值。
**解**：令 $f(x) = \sqrt{x}$，$x_0 = 1$，$\Delta x = 0.02$，则 $f'(x) = \frac{1}{2\sqrt{x}}$，因此

$$
\sqrt{1.02} \approx \sqrt{1} + \frac{1}{2\sqrt{1}} \cdot 0.02 = 1 + 0.01 = 1.01
$$

## §4 微分中值定理

### 4.1 罗尔定理

**定理7（罗尔定理）** 如果函数 $f(x)$ 满足：
1. 在闭区间 $[a, b]$ 上连续；
2. 在开区间 $(a, b)$ 内可导；
3. 在区间端点处的函数值相等，即 $f(a) = f(b)$，

那么在 $(a, b)$ 内至少存在一点 $\xi$（$a < \xi < b$），使得 $f'(\xi) = 0$。

**证明思路**：由连续函数的最值定理，$f(x)$ 在 $[a, b]$ 上必取得最大值 $M$ 和最小值 $m$。
- 若 $M = m$，则 $f(x)$ 在 $[a, b]$ 上为常数，故 $f'(x) \equiv 0$，任取 $\xi \in (a, b)$ 即可。
- 若 $M > m$，由于 $f(a) = f(b)$，故 $M$ 和 $m$ 中至少有一个不在端点处取得，不妨设 $M = f(\xi)$，$\xi \in (a, b)$，由费马引理（可导的极值点导数为0）知 $f'(\xi) = 0$。

### 4.2 拉格朗日中值定理

**定理8（拉格朗日中值定理）** 如果函数 $f(x)$ 满足：
1. 在闭区间 $[a, b]$ 上连续；
2. 在开区间 $(a, b)$ 内可导，

那么在 $(a, b)$ 内至少存在一点 $\xi$（$a < \xi < b$），使等式

$$
f(b) - f(a) = f'(\xi)(b - a)
$$

成立。

**证明思路**：构造辅助函数

$$
F(x) = f(x) - f(a) - \frac{f(b) - f(a)}{b - a}(x - a)
$$

验证 $F(x)$ 满足罗尔定理的三个条件，故存在 $\xi \in (a, b)$ 使 $F'(\xi) = 0$，即得结论。

拉格朗日中值定理的其他形式：
- $f(x + \Delta x) - f(x) = f'(x + \theta \Delta x) \Delta x$（$0 < \theta < 1$）
- $\Delta y = f'(x + \theta \Delta x) \Delta x$（有限增量公式）

**推论1** 如果函数 $f(x)$ 在区间 $I$ 上的导数恒为零，那么 $f(x)$ 在区间 $I$ 上是一个常数。

**推论2** 如果函数 $f(x)$ 与 $g(x)$ 在区间 $I$ 上每一点的导数都相等，那么这两个函数在区间 $I$ 上最多相差一个常数，即 $f(x) - g(x) = C$（$C$ 为常数）。

**例5** 证明当 $x > 0$ 时，$\frac{x}{1+x} < \ln(1+x) < x$。

**证明**：设 $f(t) = \ln(1+t)$，在 $[0, x]$ 上应用拉格朗日中值定理，有

$$
\ln(1+x) - \ln 1 = \frac{1}{1+\xi}(x - 0) \quad (0 < \xi < x)
$$

即 $\ln(1+x) = \frac{x}{1+\xi}$。由于 $0 < \xi < x$，故 $\frac{1}{1+x} < \frac{1}{1+\xi} < 1$，因此

$$
\frac{x}{1+x} < \frac{x}{1+\xi} < x \implies \frac{x}{1+x} < \ln(1+x) < x
$$

### 4.3 柯西中值定理

**定理9（柯西中值定理）** 如果函数 $f(x)$ 及 $F(x)$ 满足：
1. 在闭区间 $[a, b]$ 上连续；
2. 在开区间 $(a, b)$ 内可导；
3. 对任一 $x \in (a, b)$，$F'(x) \neq 0$，

那么在 $(a, b)$ 内至少存在一点 $\xi$，使等式

$$
\frac{f(b) - f(a)}{F(b) - F(a)} = \frac{f'(\xi)}{F'(\xi)}
$$

成立。

**证明思路**：构造辅助函数

$$
\Phi(x) = f(x) - f(a) - \frac{f(b) - f(a)}{F(b) - F(a)}[F(x) - F(a)]
$$

验证 $\Phi(x)$ 满足罗尔定理的条件，故存在 $\xi \in (a, b)$ 使 $\Phi'(\xi) = 0$，即得结论。

注：当 $F(x) = x$ 时，柯西中值定理就是拉格朗日中值定理，因此拉格朗日中值定理是柯西中值定理的特例。

## §5 泰勒公式

### 5.1 泰勒中值定理

**定理10（泰勒中值定理）** 如果函数 $f(x)$ 在包含 $x_0$ 的某个开区间 $(a, b)$ 内具有直到 $n+1$ 阶的导数，则对任一 $x \in (a, b)$，有

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n + R_n(x)
$$

其中

$$
R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - x_0)^{n+1}
$$

这里 $\xi$ 是介于 $x_0$ 与 $x$ 之间的某个值。

上式称为**泰勒公式**，$R_n(x)$ 称为**拉格朗日型余项**。

多项式

$$
P_n(x) = f(x_0) + f'(x_0)(x - x_0) + \cdots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n
$$

称为函数 $f(x)$ 在 $x_0$ 处的**泰勒多项式**。

**证明思路**：对函数 $R_n(x) = f(x) - P_n(x)$ 和 $Q_n(x) = (x - x_0)^{n+1}$ 连续应用 $n+1$ 次柯西中值定理，即得结论。

当 $n = 0$ 时，泰勒公式就是拉格朗日中值公式，因此泰勒中值定理是拉格朗日中值定理的推广。

### 5.2 麦克劳林公式

当 $x_0 = 0$ 时，泰勒公式称为**麦克劳林公式**：

$$
f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \cdots + \frac{f^{(n)}(0)}{n!}x^n + \frac{f^{(n+1)}(\theta x)}{(n+1)!}x^{n+1} \quad (0 < \theta < 1)
$$

### 5.3 常用初等函数的麦克劳林公式

1. $e^x = 1 + x + \frac{x^2}{2!} + \cdots + \frac{x^n}{n!} + \frac{e^{\theta x}}{(n+1)!}x^{n+1}$
2. $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots + (-1)^m \frac{x^{2m+1}}{(2m+1)!} + \frac{\sin[\theta x + (m+1)\pi]}{(2m+2)!}x^{2m+2}$
3. $\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots + (-1)^m \frac{x^{2m}}{(2m)!} + \frac{\cos[\theta x + (m+1)\pi]}{(2m+2)!}x^{2m+2}$
4. $\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots + (-1)^{n-1} \frac{x^n}{n} + \frac{(-1)^n}{(n+1)(1+\theta x)^{n+1}}x^{n+1}$
5. $(1+x)^\alpha = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2!}x^2 + \cdots + \frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}x^n + \frac{\alpha(\alpha-1)\cdots(\alpha-n)}{(n+1)!}(1+\theta x)^{\alpha-n-1}x^{n+1}$

其中 $0 < \theta < 1$。

**例6** 求 $\sin x$ 的四阶麦克劳林公式。
**解**：$f(x) = \sin x$，$f(0) = 0$；$f'(x) = \cos x$，$f'(0) = 1$；$f''(x) = -\sin x$，$f''(0) = 0$；$f'''(x) = -\cos x$，$f'''(0) = -1$；$f^{(4)}(x) = \sin x$，$f^{(4)}(0) = 0$；$f^{(5)}(x) = \cos x$，因此

$$
\sin x = x - \frac{x^3}{6} + \frac{\cos(\theta x)}{120}x^5 \quad (0 < \theta < 1)
$$

## §6 函数的单调性与极值

### 6.1 函数单调性的判定法

**定理11** 设函数 $y = f(x)$ 在 $[a, b]$ 上连续，在 $(a, b)$ 内可导。
1. 如果在 $(a, b)$ 内 $f'(x) > 0$，那么函数 $y = f(x)$ 在 $[a, b]$ 上单调增加；
2. 如果在 $(a, b)$ 内 $f'(x) < 0$，那么函数 $y = f(x)$ 在 $[a, b]$ 上单调减少。

**证明思路**：任取 $x_1, x_2 \in [a, b]$，且 $x_1 < x_2$，应用拉格朗日中值定理，得

$$
f(x_2) - f(x_1) = f'(\xi)(x_2 - x_1) \quad (x_1 < \xi < x_2)
$$

由 $f'(\xi)$ 的符号可知 $f(x_2) - f(x_1)$ 的符号，即得单调性。

### 6.2 函数的极值及其求法

**定义5（极值）** 设函数 $f(x)$ 在点 $x_0$ 的某邻域 $U(x_0)$ 内有定义，如果对于去心邻域 $\mathring{U}(x_0)$ 内的任一 $x$，有

$$
f(x) < f(x_0) \quad \text{（或 } f(x) > f(x_0) \text{）}
$$

那么就称 $f(x_0)$ 是函数 $f(x)$ 的一个**极大值**（或**极小值**）。极大值与极小值统称为**极值**，使函数取得极值的点 $x_0$ 称为**极值点**。

**定理12（必要条件）** 设函数 $f(x)$ 在点 $x_0$ 处可导，且在 $x_0$ 处取得极值，则 $f'(x_0) = 0$。

使导数为零的点称为**驻点**。可导函数的极值点必定是驻点，但驻点不一定是极值点。

**定理13（第一充分条件）** 设函数 $f(x)$ 在点 $x_0$ 处连续，且在 $x_0$ 的某去心邻域 $\mathring{U}(x_0, \delta)$ 内可导。
1. 若 $x \in (x_0 - \delta, x_0)$ 时，$f'(x) > 0$，而 $x \in (x_0, x_0 + \delta)$ 时，$f'(x) < 0$，则 $f(x)$ 在 $x_0$ 处取得极大值；
2. 若 $x \in (x_0 - \delta, x_0)$ 时，$f'(x) < 0$，而 $x \in (x_0, x_0 + \delta)$ 时，$f'(x) > 0$，则 $f(x)$ 在 $x_0$ 处取得极小值；
3. 若 $x \in \mathring{U}(x_0, \delta)$ 时，$f'(x)$ 的符号保持不变，则 $f(x)$ 在 $x_0$ 处没有极值。

**定理14（第二充分条件）** 设函数 $f(x)$ 在点 $x_0$ 处具有二阶导数且 $f'(x_0) = 0$，$f''(x_0) \neq 0$，则
1. 当 $f''(x_0) < 0$ 时，函数 $f(x)$ 在 $x_0$ 处取得极大值；
2. 当 $f''(x_0) > 0$ 时，函数 $f(x)$ 在 $x_0$ 处取得极小值。

**证明思路**：由二阶导数的定义

$$
f''(x_0) = \lim_{x \to x_0} \frac{f'(x) - f'(x_0)}{x - x_0} = \lim_{x \to x_0} \frac{f'(x)}{x - x_0}
$$

若 $f''(x_0) < 0$，则当 $x$ 接近 $x_0$ 时，$\frac{f'(x)}{x - x_0} < 0$，因此 $f'(x)$ 在 $x_0$ 左侧正，右侧负，由第一充分条件知 $f(x)$ 在 $x_0$ 处取得极大值。

**例7** 求函数 $f(x) = x^3 - 3x$ 的极值。
**解**：$f'(x) = 3x^2 - 3 = 3(x-1)(x+1)$，令 $f'(x) = 0$，得驻点 $x = -1$ 和 $x = 1$。

$f''(x) = 6x$，$f''(-1) = -6 < 0$，故 $f(-1) = 2$ 是极大值；$f''(1) = 6 > 0$，故 $f(1) = -2$ 是极小值。

### 6.3 最大值与最小值问题

设函数 $f(x)$ 在闭区间 $[a, b]$ 上连续，则其最大值和最小值可通过以下步骤求得：
1. 求出 $f(x)$ 在 $(a, b)$ 内的驻点和不可导点；
2. 计算 $f(x)$ 在上述各点及端点 $a, b$ 处的函数值；
3. 比较这些函数值，其中最大的就是最大值，最小的就是最小值。

## §7 曲线的凹凸性与拐点

### 7.1 曲线的凹凸性

**定义6** 设 $f(x)$ 在区间 $I$ 上连续，如果对 $I$ 上任意两点 $x_1, x_2$，恒有

$$
f\left( \frac{x_1 + x_2}{2} \right) < \frac{f(x_1) + f(x_2)}{2}
$$

那么称 $f(x)$ 在 $I$ 上的图形是**凹的**（或**凹弧**）；如果恒有

$$
f\left( \frac{x_1 + x_2}{2} \right) > \frac{f(x_1) + f(x_2)}{2}
$$

那么称 $f(x)$ 在 $I$ 上的图形是**凸的**（或**凸弧**）。

**定理15** 设 $f(x)$ 在 $[a, b]$ 上连续，在 $(a, b)$ 内具有一阶和二阶导数，那么
1. 若在 $(a, b)$ 内 $f''(x) > 0$，则 $f(x)$ 在 $[a, b]$ 上的图形是凹的；
2. 若在 $(a, b)$ 内 $f''(x) < 0$，则 $f(x)$ 在 $[a, b]$ 上的图形是凸的。

### 7.2 拐点

**定义7** 连续曲线 $y = f(x)$ 上凹弧与凸弧的分界点称为这曲线的**拐点**。

**拐点的求法**：
1. 求 $f''(x)$；
2. 令 $f''(x) = 0$，解出这方程在区间 $I$ 内的实根，并求出在区间 $I$ 内 $f''(x)$ 不存在的点；
3. 对于上述求得的点 $x_0$，检查 $f''(x)$ 在 $x_0$ 左、右两侧邻近的符号，如果两侧符号相反，则点 $(x_0, f(x_0))$ 是拐点；如果两侧符号相同，则不是拐点。

**例8** 求曲线 $y = x^3 - 3x^2 + 1$ 的凹凸区间及拐点。
**解**：$y' = 3x^2 - 6x$，$y'' = 6x - 6 = 6(x - 1)$。

令 $y'' = 0$，得 $x = 1$。

当 $x < 1$ 时，$y'' < 0$，曲线在 $(-\infty, 1]$ 上是凸的；
当 $x > 1$ 时，$y'' > 0$，曲线在 $[1, +\infty)$ 上是凹的；
拐点为 $(1, 1^3 - 3 \cdot 1^2 + 1) = (1, -1)$。

## §8 洛必达法则

### 8.1 $\frac{0}{0}$ 型未定式

**定理16（洛必达法则）** 设
1. 当 $x \to a$（或 $x \to \infty$）时，函数 $f(x)$ 及 $F(x)$ 都趋于零；
2. 在点 $a$ 的某去心邻域内（或当 $|x| > N$ 时），$f'(x)$ 及 $F'(x)$ 都存在且 $F'(x) \neq 0$；
3. $\lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f'(x)}{F'(x)}$ 存在（或为无穷大），

那么

$$
\lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f(x)}{F(x)} = \lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f'(x)}{F'(x)}
$$

**证明思路**（以 $x \to a$ 为例）：补充定义 $f(a) = F(a) = 0$，则 $f(x)$ 和 $F(x)$ 在点 $a$ 的某邻域内连续，应用柯西中值定理，得

$$
\frac{f(x)}{F(x)} = \frac{f(x) - f(a)}{F(x) - F(a)} = \frac{f'(\xi)}{F'(\xi)}
$$

其中 $\xi$ 在 $a$ 与 $x$ 之间，令 $x \to a$，则 $\xi \to a$，故得结论。

### 8.2 $\frac{\infty}{\infty}$ 型未定式

**定理17** 设
1. 当 $x \to a$（或 $x \to \infty$）时，函数 $f(x)$ 及 $F(x)$ 都趋于无穷大；
2. 在点 $a$ 的某去心邻域内（或当 $|x| > N$ 时），$f'(x)$ 及 $F'(x)$ 都存在且 $F'(x) \neq 0$；
3. $\lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f'(x)}{F'(x)}$ 存在（或为无穷大），

那么

$$
\lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f(x)}{F(x)} = \lim_{\substack{x \to a \\ (x \to \infty)}} \frac{f'(x)}{F'(x)}
$$

### 8.3 其他类型的未定式

其他类型的未定式（如 $0 \cdot \infty$，$\infty - \infty$，$0^0$，$1^\infty$，$\infty^0$ 等）可以通过适当的变形转化为 $\frac{0}{0}$ 或 $\frac{\infty}{\infty}$ 型，再应用洛必达法则。

**例9** 求 $\lim_{x \to 0} \frac{\sin x}{x}$。
**解**：这是 $\frac{0}{0}$ 型，应用洛必达法则

$$
\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1
$$

**例10** 求 $\lim_{x \to +\infty} \frac{\ln x}{x^n}$（$n > 0$）。
**解**：这是 $\frac{\infty}{\infty}$ 型，应用洛必达法则

$$
\lim_{x \to +\infty} \frac{\ln x}{x^n} = \lim_{x \to +\infty} \frac{\frac{1}{x}}{n x^{n-1}} = \lim_{x \to +\infty} \frac{1}{n x^n} = 0
$$

**例11** 求 $\lim_{x \to 0^+} x \ln x$。
**解**：这是 $0 \cdot \infty$ 型，变形为 $\frac{\infty}{\infty}$ 型

$$
\lim_{x \to 0^+} x \ln x = \lim_{x \to 0^+} \frac{\ln x}{\frac{1}{x}} = \lim_{x \to 0^+} \frac{\frac{1}{x}}{-\frac{1}{x^2}} = \lim_{x \to 0^+} (-x) = 0
$$

## §9 微分学在AI中的应用

### 9.1 激活函数及其导数

在人工神经网络中，激活函数是关键组成部分，它将线性变换转换为非线性变换，使神经网络能够学习复杂的非线性模式。常见的激活函数及其导数如下：

#### 9.1.1 Sigmoid函数

**定义**：$\sigma(x) = \frac{1}{1 + e^{-x}}$

**性质**：
- 定义域：$\mathbb{R}$，值域：$(0, 1)$
- 单调递增，处处可导
- 当 $x \to +\infty$ 时，$\sigma(x) \to 1$；当 $x \to -\infty$ 时，$\sigma(x) \to 0$

**导数**：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**证明**：

$$
\begin{aligned}
\sigma'(x) &= \frac{d}{dx} \left( \frac{1}{1 + e^{-x}} \right) = \frac{e^{-x}}{(1 + e^{-x})^2} \\
&= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x)(1 - \sigma(x))
\end{aligned}
$$

#### 9.1.2 Tanh函数

**定义**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$

**性质**：
- 定义域：$\mathbb{R}$，值域：$(-1, 1)$
- 单调递增，处处可导，奇函数
- 输出以0为中心

**导数**：$\tanh'(x) = 1 - \tanh^2(x)$

#### 9.1.3 ReLU函数

**定义**：$\text{ReLU}(x) = \max(0, x) = \begin{cases} x, & x \geq 0 \\ 0, & x < 0 \end{cases}$

**性质**：
- 计算简单，收敛速度快
- 单侧线性，当 $x > 0$ 时不饱和
- 可能出现"死亡ReLU"问题

**导数**：$\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x < 0 \end{cases}$（在 $x = 0$ 处不可导，通常取0或1）

#### 9.1.4 Leaky ReLU函数

**定义**：$\text{LeakyReLU}(x) = \begin{cases} x, & x \geq 0 \\ \alpha x, & x < 0 \end{cases}$，其中 $\alpha > 0$（通常取0.01）

**导数**：$\text{LeakyReLU}'(x) = \begin{cases} 1, & x > 0 \\ \alpha, & x < 0 \end{cases}$

#### 9.1.5 Softmax函数

**定义**：对于向量 $\boldsymbol{x} = (x_1, x_2, \dots, x_n)^T$，

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} \quad (i = 1, 2, \dots, n)
$$

**性质**：
- 输出为概率分布，$\sum_{i=1}^n \text{softmax}(\boldsymbol{x})_i = 1$
- 常用于多分类问题的输出层

**导数**：设 $a_i = \text{softmax}(\boldsymbol{x})_i$，则

$$
\frac{\partial a_i}{\partial x_j} = \begin{cases} a_i(1 - a_i), & i = j \\ -a_i a_j, & i \neq j \end{cases}
$$

### 9.2 梯度下降法的数学基础

梯度下降法是机器学习中最常用的优化算法之一，用于最小化损失函数 $L(\boldsymbol{\theta})$，其中 $\boldsymbol{\theta} = (\theta_1, \theta_2, \dots, \theta_n)^T$ 是参数向量。

#### 9.2.1 梯度的定义

函数 $L(\boldsymbol{\theta})$ 在点 $\boldsymbol{\theta}$ 处的**梯度**是一个向量，其方向是函数在该点处增长最快的方向，其模是该方向上的变化率，记为

$$
\nabla L(\boldsymbol{\theta}) = \left( \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots, \frac{\partial L}{\partial \theta_n} \right)^T
$$

#### 9.2.2 梯度下降的基本思想

由于负梯度方向 $-\nabla L(\boldsymbol{\theta})$ 是函数 $L(\boldsymbol{\theta})$ 在点 $\boldsymbol{\theta}$ 处下降最快的方向，因此我们可以沿着该方向更新参数：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)
$$

其中 $\eta > 0$ 称为**学习率**，$t$ 为迭代步数。

#### 9.2.3 泰勒展开与梯度下降

考虑损失函数 $L(\boldsymbol{\theta})$ 在点 $\boldsymbol{\theta}_t$ 处的一阶泰勒展开：

$$
L(\boldsymbol{\theta}_t + \Delta \boldsymbol{\theta}) \approx L(\boldsymbol{\theta}_t) + \nabla L(\boldsymbol{\theta}_t)^T \Delta \boldsymbol{\theta}
$$

我们希望找到 $\Delta \boldsymbol{\theta}$ 使得 $L(\boldsymbol{\theta}_t + \Delta \boldsymbol{\theta})$ 最小。令 $\Delta \boldsymbol{\theta} = -\eta \nabla L(\boldsymbol{\theta}_t)$，则

$$
L(\boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)) \approx L(\boldsymbol{\theta}_t) - \eta \|\nabla L(\boldsymbol{\theta}_t)\|^2
$$

只要 $\eta > 0$ 足够小，就有 $L(\boldsymbol{\theta}_{t+1}) < L(\boldsymbol{\theta}_t)$，保证损失函数下降。

#### 9.2.4 学习率的选择

学习率 $\eta$ 是梯度下降法的关键超参数：
- 过小：收敛速度慢
- 过大：可能导致振荡甚至发散

可以采用自适应学习率方法（如Adam、AdaGrad、RMSProp等）动态调整学习率。

#### 9.2.5 梯度下降的变体

1. **批量梯度下降（BGD）**：使用整个数据集计算梯度
   $$
   \nabla L(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^m \nabla L_i(\boldsymbol{\theta})
   $$
   其中 $m$ 是样本数，$L_i$ 是第 $i$ 个样本的损失。

2. **随机梯度下降（SGD）**：每次只使用一个样本计算梯度
   $$
   \nabla L(\boldsymbol{\theta}) \approx \nabla L_i(\boldsymbol{\theta})
   $$

3. **小批量梯度下降（Mini-batch GD）**：每次使用一小批样本计算梯度（通常取32、64、128等）

#### 9.2.6 示例：线性回归的梯度下降

对于线性回归模型 $h_\boldsymbol{\theta}(x) = \boldsymbol{\theta}^T x$，损失函数为均方误差：

$$
L(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^m (h_\boldsymbol{\theta}(x^{(i)}) - y^{(i)})^2
$$

计算梯度：

$$
\frac{\partial L}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\boldsymbol{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

更新规则：

$$
\theta_j := \theta_j - \eta \cdot \frac{1}{m} \sum_{i=1}^m (h_\boldsymbol{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

### 9.3 反向传播算法

反向传播（Backpropagation）算法是训练神经网络的核心方法，它利用链式法则高效计算损失函数关于各层权重的梯度。

#### 9.3.1 链式法则

对于复合函数 $L = f(g(h(\boldsymbol{\theta})))$，有

$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial \boldsymbol{\theta}}
$$

#### 9.3.2 前向传播

设神经网络有 $L$ 层，第 $l$ 层的输入为 $\boldsymbol{a}^{(l-1)}$，线性变换为 $\boldsymbol{z}^{(l)} = W^{(l)} \boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)}$，激活函数为 $\sigma$，输出为 $\boldsymbol{a}^{(l)} = \sigma(\boldsymbol{z}^{(l)})$。

损失函数为 $L = \frac{1}{2} \|\boldsymbol{y} - \boldsymbol{a}^{(L)}\|^2$。

#### 9.3.3 反向传播

定义误差项 $\delta^{(l)} = \frac{\partial L}{\partial \boldsymbol{z}^{(l)}}$，则：

1. 输出层误差：
   $$
   \delta^{(L)} = (\boldsymbol{a}^{(L)} - \boldsymbol{y}) \odot \sigma'(\boldsymbol{z}^{(L)})
   $$
   其中 $\odot$ 表示逐元素乘积。

2. 隐藏层误差（反向传播）：
   $$
   \delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(\boldsymbol{z}^{(l)})
   $$

3. 梯度计算：
   $$
   \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (\boldsymbol{a}^{(l-1)})^T, \quad \frac{\partial L}{\partial \boldsymbol{b}^{(l)}} = \delta^{(l)}
   $$

4. 参数更新：
   $$
   W^{(l)} := W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}, \quad \boldsymbol{b}^{(l)} := \boldsymbol{b}^{(l)} - \eta \frac{\partial L}{\partial \boldsymbol{b}^{(l)}}
   $$

### 9.4 牛顿法与海森矩阵

在优化问题中，除了梯度下降法，牛顿法也是一种重要的方法，它利用二阶导数信息（海森矩阵）收敛更快。

**定义（海森矩阵）** 函数 $L(\boldsymbol{\theta})$ 的**海森矩阵**$H$ 是二阶偏导数构成的矩阵：

$$
H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
$$

**牛顿法更新公式**：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_t^{-1} \nabla L(\boldsymbol{\theta}_t)
$$

牛顿法的收敛速度比梯度下降法快，但计算海森矩阵的逆在高维情况下非常昂贵，因此在深度学习中较少直接使用，但其思想启发了许多拟牛顿法（如BFGS）和二阶优化方法。

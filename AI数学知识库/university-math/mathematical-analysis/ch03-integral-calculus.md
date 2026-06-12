# 第3章：积分学基础


## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


积分学是数学分析的核心组成部分，与微分学共同构成微积分的完整体系。本章系统介绍不定积分、定积分、微积分基本定理及其应用。

---

## 目录

1. [不定积分](#1-不定积分)
2. [定积分](#2-定积分)
3. [微积分基本定理](#3-微积分基本定理)
4. [定积分的应用](#4-定积分的应用)
5. [反常积分](#5-反常积分)
6. [积分在AI中的应用](#6-积分在ai中的应用)

---

## 1. 不定积分

### 1.1 原函数与不定积分的定义

**定义1.1（原函数）** 设函数 $f(x)$ 在区间 $I$ 上有定义，如果存在函数 $F(x)$，使得对任意 $x \in I$，都有
$$F'(x) = f(x) \quad \text{或} \quad dF(x) = f(x)dx$$
则称 $F(x)$ 为 $f(x)$ 在区间 $I$ 上的一个**原函数**。

**定义1.2（不定积分）** 函数 $f(x)$ 在区间 $I$ 上的所有原函数的集合称为 $f(x)$ 在 $I$ 上的**不定积分**，记作
$$\int f(x)dx$$
其中 $\int$ 称为积分号，$f(x)$ 称为被积函数，$f(x)dx$ 称为被积表达式，$x$ 称为积分变量。

**定理1.1** 若 $F(x)$ 是 $f(x)$ 在区间 $I$ 上的一个原函数，则 $f(x)$ 的不定积分为
$$\int f(x)dx = F(x) + C$$
其中 $C$ 为任意常数，称为积分常数。

**证明思路**：
- 若 $F(x)$ 是原函数，则对任意常数 $C$，$F(x) + C$ 也是原函数。
- 若 $G(x)$ 是另一个原函数，则 $(G(x) - F(x))' = 0$，故 $G(x) - F(x) = C$。

**例1.1** 求 $\int x^2 dx$。

**解**：因为 $\left(\frac{x^3}{3}\right)' = x^2$，所以
$$\int x^2 dx = \frac{x^3}{3} + C$$

### 1.2 不定积分的性质

**性质1.1** 不定积分与导数（微分）的互逆关系：
$$\frac{d}{dx}\int f(x)dx = f(x) \quad \text{或} \quad d\int f(x)dx = f(x)dx$$
$$\int F'(x)dx = F(x) + C \quad \text{或} \quad \int dF(x) = F(x) + C$$

**性质1.2（线性性）** 设 $k_1, k_2$ 为常数，则
$$\int [k_1f(x) + k_2g(x)]dx = k_1\int f(x)dx + k_2\int g(x)dx$$

### 1.3 基本积分公式

1. $\int kdx = kx + C$
2. $\int x^\alpha dx = \frac{x^{\alpha+1}}{\alpha+1} + C \ (\alpha \neq -1)$
3. $\int \frac{1}{x}dx = \ln|x| + C$
4. $\int e^x dx = e^x + C$
5. $\int a^x dx = \frac{a^x}{\ln a} + C \ (a > 0, a \neq 1)$
6. $\int \sin x dx = -\cos x + C$
7. $\int \cos x dx = \sin x + C$
8. $\int \sec^2 x dx = \tan x + C$
9. $\int \csc^2 x dx = -\cot x + C$
10. $\int \sec x \tan x dx = \sec x + C$
11. $\int \csc x \cot x dx = -\csc x + C$
12. $\int \frac{1}{\sqrt{1-x^2}}dx = \arcsin x + C$
13. $\int \frac{1}{1+x^2}dx = \arctan x + C$

**例1.2** 求 $\int (2x^3 - 3\sin x + 5e^x)dx$。

**解**：
$$
\begin{align*}
\int (2x^3 - 3\sin x + 5e^x)dx &= 2\int x^3dx - 3\int \sin xdx + 5\int e^xdx \\
&= 2 \cdot \frac{x^4}{4} - 3(-\cos x) + 5e^x + C \\
&= \frac{x^4}{2} + 3\cos x + 5e^x + C
\end{align*}
$$

### 1.4 换元积分法

#### 1.4.1 第一类换元法（凑微分法）

**定理1.2（第一类换元法）** 设 $f(u)$ 具有原函数 $F(u)$，$u = \varphi(x)$ 可导，则
$$\int f[\varphi(x)]\varphi'(x)dx = \int f(u)du = F(u) + C = F[\varphi(x)] + C$$

**证明思路**：利用复合函数求导法则验证。

**例1.3** 求 $\int 2xe^{x^2}dx$。

**解**：令 $u = x^2$，则 $du = 2xdx$，
$$\int 2xe^{x^2}dx = \int e^u du = e^u + C = e^{x^2} + C$$

#### 1.4.2 第二类换元法

**定理1.3（第二类换元法）** 设 $x = \psi(t)$ 单调、可导且 $\psi'(t) \neq 0$，又设 $f[\psi(t)]\psi'(t)$ 具有原函数 $\Phi(t)$，则
$$\int f(x)dx = \int f[\psi(t)]\psi'(t)dt = \Phi(t) + C = \Phi[\psi^{-1}(x)] + C$$
其中 $t = \psi^{-1}(x)$ 是 $x = \psi(t)$ 的反函数。

**常用代换**：
- 三角代换：$\sqrt{a^2 - x^2}$ 用 $x = a\sin t$；$\sqrt{x^2 + a^2}$ 用 $x = a\tan t$；$\sqrt{x^2 - a^2}$ 用 $x = a\sec t$
- 倒代换：$x = \frac{1}{t}$
- 根式代换：$\sqrt[n]{ax + b} = t$

**例1.4** 求 $\int \sqrt{a^2 - x^2}dx \ (a > 0)$。

**解**：令 $x = a\sin t$，$-\frac{\pi}{2} \leq t \leq \frac{\pi}{2}$，则 $dx = a\cos t dt$，
$$
\begin{align*}
\int \sqrt{a^2 - x^2}dx &= \int a\cos t \cdot a\cos t dt = a^2\int \cos^2 t dt \\
&= a^2\int \frac{1 + \cos 2t}{2}dt = \frac{a^2}{2}\left(t + \frac{\sin 2t}{2}\right) + C \\
&= \frac{a^2}{2}t + \frac{a^2}{2}\sin t \cos t + C \\
&= \frac{a^2}{2}\arcsin\frac{x}{a} + \frac{x}{2}\sqrt{a^2 - x^2} + C
\end{align*}
$$

### 1.5 分部积分法

**定理1.4（分部积分法）** 设函数 $u = u(x)$ 和 $v = v(x)$ 具有连续导数，则
$$\int udv = uv - \int vdu$$

**证明思路**：由乘积的微分法则 $d(uv) = udv + vdu$，两边积分即得。

**分部积分的选择原则**（LIATE）：
- L: 对数函数
- I: 反三角函数
- A: 代数函数
- T: 三角函数
- E: 指数函数
优先选择前者作为 $u$。

**例1.5** 求 $\int x e^x dx$。

**解**：令 $u = x$，$dv = e^x dx$，则 $du = dx$，$v = e^x$，
$$\int x e^x dx = x e^x - \int e^x dx = x e^x - e^x + C$$

**例1.6** 求 $\int e^x \sin x dx$。

**解**：令 $I = \int e^x \sin x dx$，
$$
\begin{align*}
I &= \int \sin x d(e^x) = e^x \sin x - \int e^x \cos x dx \\
&= e^x \sin x - \int \cos x d(e^x) = e^x \sin x - e^x \cos x - \int e^x \sin x dx \\
&= e^x(\sin x - \cos x) - I
\end{align*}
$$
移项得 $2I = e^x(\sin x - \cos x) + C$，故
$$I = \frac{e^x}{2}(\sin x - \cos x) + C$$

---

## 2. 定积分

### 2.1 黎曼积分的定义

**定义2.1（分割）** 设 $[a, b]$ 为闭区间，在 $[a, b]$ 中任意插入 $n-1$ 个分点：
$$a = x_0 < x_1 < x_2 < \cdots < x_{n-1} < x_n = b$$
把 $[a, b]$ 分成 $n$ 个小区间 $[x_{i-1}, x_i]$，记 $\Delta x_i = x_i - x_{i-1}$，称 $T = \{x_0, x_1, \cdots, x_n\}$ 为 $[a, b]$ 的一个**分割**，记 $\|T\| = \max_{1 \leq i \leq n} \Delta x_i$ 为分割的**细度**。

**定义2.2（黎曼和）** 设 $f(x)$ 在 $[a, b]$ 上有定义，对分割 $T$，任取 $\xi_i \in [x_{i-1}, x_i]$，作和
$$\sigma(T, \xi) = \sum_{i=1}^n f(\xi_i)\Delta x_i$$
称为 $f(x)$ 在 $[a, b]$ 上的一个**黎曼和**。

**定义2.3（黎曼积分）** 设 $f(x)$ 在 $[a, b]$ 上有定义，若存在常数 $I$，使得对任意 $\varepsilon > 0$，存在 $\delta > 0$，对任意分割 $T$ 及任意取点 $\{\xi_i\}$，只要 $\|T\| < \delta$，就有
$$\left|\sum_{i=1}^n f(\xi_i)\Delta x_i - I\right| < \varepsilon$$
则称 $f(x)$ 在 $[a, b]$ 上**可积**，称 $I$ 为 $f(x)$ 在 $[a, b]$ 上的**定积分**，记作
$$\int_a^b f(x)dx = I$$
其中 $a, b$ 分别称为积分下限和上限。

**几何意义**：当 $f(x) \geq 0$ 时，$\int_a^b f(x)dx$ 表示曲线 $y = f(x)$，直线 $x = a, x = b$ 及 $x$ 轴围成的曲边梯形的面积。

### 2.2 定积分的性质

**规定**：
$$\int_a^a f(x)dx = 0, \quad \int_b^a f(x)dx = -\int_a^b f(x)dx$$

**性质2.1（线性性）**
$$\int_a^b [k_1f(x) + k_2g(x)]dx = k_1\int_a^b f(x)dx + k_2\int_a^b g(x)dx$$

**性质2.2（区间可加性）** 对任意三点 $a, b, c$，有
$$\int_a^b f(x)dx = \int_a^c f(x)dx + \int_c^b f(x)dx$$

**性质2.3（保号性）** 若在 $[a, b]$ 上 $f(x) \geq 0$，则
$$\int_a^b f(x)dx \geq 0$$

**推论2.1（单调性）** 若在 $[a, b]$ 上 $f(x) \geq g(x)$，则
$$\int_a^b f(x)dx \geq \int_a^b g(x)dx$$

**推论2.2（绝对可积性）** 若 $f(x)$ 在 $[a, b]$ 上可积，则 $|f(x)|$ 也可积，且
$$\left|\int_a^b f(x)dx\right| \leq \int_a^b |f(x)|dx$$

**性质2.4（积分中值定理）** 若 $f(x)$ 在 $[a, b]$ 上连续，则存在 $\xi \in [a, b]$，使得
$$\int_a^b f(x)dx = f(\xi)(b - a)$$

**证明思路**：利用连续函数的介值定理。

### 2.3 可积条件

**定理2.1（可积的必要条件）** 若 $f(x)$ 在 $[a, b]$ 上可积，则 $f(x)$ 在 $[a, b]$ 上有界。

**定义2.4（达布和）** 设 $f(x)$ 在 $[a, b]$ 上有界，对分割 $T$，记
$$M_i = \sup_{x \in [x_{i-1}, x_i]} f(x), \quad m_i = \inf_{x \in [x_{i-1}, x_i]} f(x)$$
称
$$S(T) = \sum_{i=1}^n M_i\Delta x_i, \quad s(T) = \sum_{i=1}^n m_i\Delta x_i$$
分别为 $f(x)$ 关于分割 $T$ 的**达布上和**与**达布下和**。

**定理2.2（可积的充要条件）** 有界函数 $f(x)$ 在 $[a, b]$ 上可积的充要条件是
$$\lim_{\|T\| \to 0} [S(T) - s(T)] = 0$$
即对任意 $\varepsilon > 0$，存在分割 $T$，使得
$$S(T) - s(T) < \varepsilon$$

**定理2.3** 以下三类函数在 $[a, b]$ 上可积：
1. 闭区间上的连续函数
2. 闭区间上只有有限个间断点的有界函数
3. 闭区间上的单调有界函数

**例2.1** 证明 $\int_0^1 x^2 dx = \frac{1}{3}$。

**证明**：将 $[0, 1]$ 等分为 $n$ 份，取 $\xi_i = \frac{i}{n}$，则
$$
\sum_{i=1}^n f(\xi_i)\Delta x_i = \sum_{i=1}^n \left(\frac{i}{n}\right)^2 \cdot \frac{1}{n} = \frac{1}{n^3}\sum_{i=1}^n i^2 = \frac{1}{n^3} \cdot \frac{n(n+1)(2n+1)}{6} \to \frac{1}{3} \quad (n \to \infty)
$$
由可积性，得 $\int_0^1 x^2 dx = \frac{1}{3}$。

---

## 3. 微积分基本定理

### 3.1 变上限积分函数

**定义3.1（变上限积分）** 设 $f(x)$ 在 $[a, b]$ 上可积，对任意 $x \in [a, b]$，定义
$$\Phi(x) = \int_a^x f(t)dt$$
称为**变上限积分函数**。

**定理3.1（原函数存在定理）** 若 $f(x)$ 在 $[a, b]$ 上连续，则变上限积分 $\Phi(x) = \int_a^x f(t)dt$ 在 $[a, b]$ 上可导，且
$$\Phi'(x) = \frac{d}{dx}\int_a^x f(t)dt = f(x)$$

**证明思路**：
$$\Phi'(x) = \lim_{\Delta x \to 0} \frac{\Phi(x+\Delta x) - \Phi(x)}{\Delta x} = \lim_{\Delta x \to 0} \frac{1}{\Delta x}\int_x^{x+\Delta x} f(t)dt = \lim_{\Delta x \to 0} f(\xi) = f(x)$$
其中 $\xi$ 在 $x$ 与 $x+\Delta x$ 之间，利用积分中值定理。

**推论3.1** 若 $f(x)$ 在 $[a, b]$ 上连续，则 $\Phi(x) = \int_a^x f(t)dt$ 是 $f(x)$ 在 $[a, b]$ 上的一个原函数。

### 3.2 牛顿-莱布尼茨公式

**定理3.2（牛顿-莱布尼茨公式）** 若 $f(x)$ 在 $[a, b]$ 上连续，$F(x)$ 是 $f(x)$ 的一个原函数，则
$$\int_a^b f(x)dx = F(b) - F(a) \triangleq F(x)\bigg|_a^b$$

**证明**：由定理3.1，$\Phi(x) = \int_a^x f(t)dt$ 是 $f(x)$ 的一个原函数，故 $\Phi(x) - F(x) = C$。令 $x = a$，得 $C = -F(a)$，因此 $\Phi(x) = F(x) - F(a)$。令 $x = b$，得
$$\int_a^b f(x)dx = F(b) - F(a)$$

**例3.1** 计算 $\int_0^{\pi/2} \sin x dx$。

**解**：因为 $(-\cos x)' = \sin x$，所以
$$\int_0^{\pi/2} \sin x dx = -\cos x\bigg|_0^{\pi/2} = -\cos\frac{\pi}{2} + \cos 0 = 1$$

**例3.2** 计算 $\frac{d}{dx}\int_{x^2}^{x^3} \sin t^2 dt$。

**解**：利用链式法则和变上限积分求导：
$$
\begin{align*}
\frac{d}{dx}\int_{x^2}^{x^3} \sin t^2 dt &= \frac{d}{dx}\left(\int_{0}^{x^3} \sin t^2 dt - \int_{0}^{x^2} \sin t^2 dt\right) \\
&= \sin(x^3)^2 \cdot 3x^2 - \sin(x^2)^2 \cdot 2x \\
&= 3x^2 \sin x^6 - 2x \sin x^4
\end{align*}
$$

### 3.3 定积分的换元法与分部积分法

**定理3.3（定积分换元法）** 设 $f(x)$ 在 $[a, b]$ 上连续，函数 $x = \varphi(t)$ 满足：
1. $\varphi(\alpha) = a$，$\varphi(\beta) = b$
2. $\varphi(t)$ 在 $[\alpha, \beta]$（或 $[\beta, \alpha]$）上有连续导数
3. 当 $t$ 在 $[\alpha, \beta]$ 上变化时，$\varphi(t)$ 在 $[a, b]$ 上变化

则
$$\int_a^b f(x)dx = \int_\alpha^\beta f[\varphi(t)]\varphi'(t)dt$$

**例3.3** 计算 $\int_0^a \sqrt{a^2 - x^2}dx \ (a > 0)$。

**解**：令 $x = a\sin t$，则 $dx = a\cos t dt$，当 $x = 0$ 时 $t = 0$，$x = a$ 时 $t = \frac{\pi}{2}$，
$$
\begin{align*}
\int_0^a \sqrt{a^2 - x^2}dx &= \int_0^{\pi/2} a\cos t \cdot a\cos t dt = a^2\int_0^{\pi/2} \cos^2 t dt \\
&= a^2 \cdot \frac{1}{2} \cdot \frac{\pi}{2} = \frac{\pi a^2}{4}
\end{align*}
$$

**定理3.4（定积分分部积分法）** 设 $u(x), v(x)$ 在 $[a, b]$ 上有连续导数，则
$$\int_a^b u(x)v'(x)dx = u(x)v(x)\bigg|_a^b - \int_a^b v(x)u'(x)dx$$

**例3.4** 计算 $\int_0^1 x e^x dx$。

**解**：
$$
\begin{align*}
\int_0^1 x e^x dx &= \int_0^1 x d(e^x) = x e^x\bigg|_0^1 - \int_0^1 e^x dx \\
&= e - e^x\bigg|_0^1 = e - (e - 1) = 1
\end{align*}
$$

---

## 4. 定积分的应用

### 4.1 平面图形的面积

#### 4.1.1 直角坐标情形

**公式4.1** 由曲线 $y = f(x)$，直线 $x = a, x = b$ 及 $x$ 轴围成的图形面积：
$$A = \int_a^b |f(x)|dx$$

**公式4.2** 由曲线 $y = f(x), y = g(x)$，直线 $x = a, x = b$ 围成的图形面积：
$$A = \int_a^b |f(x) - g(x)|dx$$

**例4.1** 求抛物线 $y = x^2$ 与 $y^2 = x$ 围成的图形面积。

**解**：联立方程得交点 $(0, 0)$ 和 $(1, 1)$，
$$A = \int_0^1 (\sqrt{x} - x^2)dx = \left(\frac{2}{3}x^{3/2} - \frac{x^3}{3}\right)\bigg|_0^1 = \frac{1}{3}$$

#### 4.1.2 极坐标情形

**公式4.3** 由曲线 $r = r(\theta)$ 及射线 $\theta = \alpha, \theta = \beta$ 围成的曲边扇形面积：
$$A = \frac{1}{2}\int_\alpha^\beta r^2(\theta)d\theta$$

**例4.2** 求心形线 $r = a(1 + \cos\theta)$ 围成的面积。

**解**：利用对称性，
$$
\begin{align*}
A &= 2 \cdot \frac{1}{2}\int_0^\pi a^2(1 + \cos\theta)^2 d\theta = a^2\int_0^\pi (1 + 2\cos\theta + \cos^2\theta)d\theta \\
&= a^2\int_0^\pi \left(\frac{3}{2} + 2\cos\theta + \frac{\cos 2\theta}{2}\right)d\theta = \frac{3}{2}\pi a^2
\end{align*}
$$

### 4.2 体积

#### 4.2.1 旋转体体积

**公式4.4（圆盘法）** 由曲线 $y = f(x)$，直线 $x = a, x = b$ 及 $x$ 轴围成的图形绕 $x$ 轴旋转一周的体积：
$$V = \pi \int_a^b f^2(x)dx$$

**公式4.5（壳层法）** 绕 $y$ 轴旋转：
$$V = 2\pi \int_a^b x |f(x)|dx$$

**例4.3** 求圆 $x^2 + (y - b)^2 = a^2 \ (b > a)$ 绕 $x$ 轴旋转的体积（圆环体）。

**解**：上半圆 $y = b + \sqrt{a^2 - x^2}$，下半圆 $y = b - \sqrt{a^2 - x^2}$，
$$
\begin{align*}
V &= \pi \int_{-a}^a \left[(b + \sqrt{a^2 - x^2})^2 - (b - \sqrt{a^2 - x^2})^2\right]dx \\
&= \pi \int_{-a}^a 4b\sqrt{a^2 - x^2}dx = 4\pi b \cdot \frac{\pi a^2}{2} = 2\pi^2 a^2 b
\end{align*}
$$

#### 4.2.2 平行截面面积已知的立体体积

**公式4.6** 设立体在 $x = a$ 到 $x = b$ 之间，垂直于 $x$ 轴的截面面积为 $A(x)$，则体积：
$$V = \int_a^b A(x)dx$$

### 4.3 平面曲线的弧长

#### 4.3.1 直角坐标情形

**公式4.7** 设曲线 $y = f(x)$ 在 $[a, b]$ 上有连续导数，则弧长：
$$s = \int_a^b \sqrt{1 + [f'(x)]^2}dx$$

#### 4.3.2 参数方程情形

**公式4.8** 设曲线参数方程为 $\begin{cases}x = \varphi(t) \\ y = \psi(t)\end{cases}, t \in [\alpha, \beta]$，则弧长：
$$s = \int_\alpha^\beta \sqrt{[\varphi'(t)]^2 + [\psi'(t)]^2}dt$$

#### 4.3.3 极坐标情形

**公式4.9** 设曲线极坐标方程为 $r = r(\theta), \theta \in [\alpha, \beta]$，则弧长：
$$s = \int_\alpha^\beta \sqrt{r^2(\theta) + [r'(\theta)]^2}d\theta$$

**例4.4** 求悬链线 $y = \frac{e^x + e^{-x}}{2}$ 从 $x = 0$ 到 $x = a$ 的弧长。

**解**：$y' = \frac{e^x - e^{-x}}{2}$，
$$
\begin{align*}
s &= \int_0^a \sqrt{1 + \left(\frac{e^x - e^{-x}}{2}\right)^2}dx = \int_0^a \frac{e^x + e^{-x}}{2}dx \\
&= \frac{e^x - e^{-x}}{2}\bigg|_0^a = \frac{e^a - e^{-a}}{2}
\end{align*}
$$

### 4.4 物理应用

#### 4.4.1 变力做功

**公式4.10** 设变力 $F(x)$ 沿 $x$ 轴方向作用，物体从 $x = a$ 移动到 $x = b$，则做功：
$$W = \int_a^b F(x)dx$$

**例4.5** 设弹簧伸长 $x$ 时所需力 $F(x) = kx$，求从 $x = 0$ 伸长到 $x = a$ 的做功。

**解**：
$$W = \int_0^a kx dx = \frac{1}{2}ka^2$$

#### 4.4.2 液体静压力

**公式4.11** 设液体密度为 $\rho$，深度为 $h$ 处的压强 $p = \rho g h$。垂直放置在液体中的平面片，其一侧所受压力：
$$P = \rho g \int_a^b x f(x)dx$$
其中 $x$ 为深度，$f(x)$ 为深度 $x$ 处的宽度。

#### 4.4.3 引力、质心、转动惯量

- **质心坐标**（平面薄板）：
  $$\bar{x} = \frac{\int_a^b x f(x)dx}{\int_a^b f(x)dx}, \quad \bar{y} = \frac{\frac{1}{2}\int_a^b f^2(x)dx}{\int_a^b f(x)dx}$$

---

## 5. 反常积分

### 5.1 无穷限反常积分

**定义5.1** 设函数 $f(x)$ 在 $[a, +\infty)$ 上有定义，且在任意有限区间 $[a, A]$ 上可积，若极限
$$\lim_{A \to +\infty} \int_a^A f(x)dx$$
存在，则称此极限为 $f(x)$ 在 $[a, +\infty)$ 上的**无穷限反常积分**，记作
$$\int_a^{+\infty} f(x)dx = \lim_{A \to +\infty} \int_a^A f(x)dx$$
此时称反常积分**收敛**，否则称**发散**。

类似定义：
$$\int_{-\infty}^b f(x)dx = \lim_{A \to -\infty} \int_A^b f(x)dx$$
$$\int_{-\infty}^{+\infty} f(x)dx = \int_{-\infty}^c f(x)dx + \int_c^{+\infty} f(x)dx$$

**例5.1** 讨论 $p$-积分 $\int_a^{+\infty} \frac{dx}{x^p} \ (a > 0)$ 的敛散性。

**解**：
- 当 $p \neq 1$ 时，
  $$\int_a^{+\infty} \frac{dx}{x^p} = \lim_{A \to +\infty} \frac{x^{1-p}}{1-p}\bigg|_a^A = \begin{cases}
  \frac{a^{1-p}}{p-1}, & p > 1 \quad (\text{收敛}) \\
  +\infty, & p < 1 \quad (\text{发散})
  \end{cases}$$
- 当 $p = 1$ 时，
  $$\int_a^{+\infty} \frac{dx}{x} = \lim_{A \to +\infty} \ln x\bigg|_a^A = +\infty \quad (\text{发散})$$
综上，当 $p > 1$ 时收敛，$p \leq 1$ 时发散。

**定理5.1（比较判别法）** 设 $0 \leq f(x) \leq g(x)$ 在 $[a, +\infty)$ 上成立：
- 若 $\int_a^{+\infty} g(x)dx$ 收敛，则 $\int_a^{+\infty} f(x)dx$ 收敛
- 若 $\int_a^{+\infty} f(x)dx$ 发散，则 $\int_a^{+\infty} g(x)dx$ 发散

**定理5.2（绝对收敛）** 若 $\int_a^{+\infty} |f(x)|dx$ 收敛，则 $\int_a^{+\infty} f(x)dx$ 收敛，称其为**绝对收敛**。

### 5.2 无界函数反常积分（瑕积分）

**定义5.2** 设函数 $f(x)$ 在 $(a, b]$ 上有定义，在点 $a$ 附近无界，且在任意区间 $[a+\varepsilon, b]$ 上可积，若极限
$$\lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f(x)dx$$
存在，则称此极限为 $f(x)$ 在 $(a, b]$ 上的**瑕积分**，仍记作
$$\int_a^b f(x)dx = \lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f(x)dx$$
点 $a$ 称为**瑕点**。

类似定义瑕点在 $b$ 或内部的情形。

**例5.2** 讨论瑕积分 $\int_a^b \frac{dx}{(x - a)^p}$ 的敛散性。

**解**：
- 当 $p \neq 1$ 时，
  $$\int_a^b \frac{dx}{(x - a)^p} = \lim_{\varepsilon \to 0^+} \frac{(x - a)^{1-p}}{1-p}\bigg|_{a+\varepsilon}^b = \begin{cases}
  \frac{(b - a)^{1-p}}{1 - p}, & p < 1 \quad (\text{收敛}) \\
  +\infty, & p > 1 \quad (\text{发散})
  \end{cases}$$
- 当 $p = 1$ 时，
  $$\int_a^b \frac{dx}{x - a} = \lim_{\varepsilon \to 0^+} \ln(x - a)\bigg|_{a+\varepsilon}^b = +\infty \quad (\text{发散})$$
综上，当 $p < 1$ 时收敛，$p \geq 1$ 时发散。

### 5.3 Gamma函数与Beta函数

**定义5.3（Gamma函数）**
$$\Gamma(s) = \int_0^{+\infty} x^{s-1}e^{-x}dx, \quad s > 0$$

**性质**：
1. $\Gamma(s + 1) = s\Gamma(s)$
2. $\Gamma(n + 1) = n!$（$n$ 为正整数）
3. $\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}$

**定义5.4（Beta函数）**
$$B(p, q) = \int_0^1 x^{p-1}(1 - x)^{q-1}dx, \quad p > 0, q > 0$$

**关系**：
$$B(p, q) = \frac{\Gamma(p)\Gamma(q)}{\Gamma(p + q)}$$

---

## 6. 积分在AI中的应用

积分在人工智能中有着广泛的应用，特别是在概率论、统计学和机器学习中。

### 6.1 概率密度函数的积分

**定义6.1（概率密度函数）** 设 $X$ 为连续型随机变量，若存在非负可积函数 $f(x)$，使得对任意实数 $a \leq b$，有
$$P(a \leq X \leq b) = \int_a^b f(x)dx$$
则称 $f(x)$ 为 $X$ 的**概率密度函数**（PDF）。

**性质**：
1. 非负性：$f(x) \geq 0$
2. 归一性：$\int_{-\infty}^{+\infty} f(x)dx = 1$

**例6.1（正态分布）** 正态分布的概率密度函数：
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$
其归一性验证：
$$\int_{-\infty}^{+\infty} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}dx = 1$$
令 $t = \frac{x - \mu}{\sigma}$，则 $dx = \sigma dt$，
$$\int_{-\infty}^{+\infty} f(x)dx = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{+\infty} e^{-\frac{t^2}{2}}dt = 1$$

### 6.2 期望与方差的积分计算

**定义6.2（数学期望）** 设连续型随机变量 $X$ 的概率密度为 $f(x)$，则其数学期望为
$$E[X] = \int_{-\infty}^{+\infty} x f(x)dx$$
若 $g(X)$ 是 $X$ 的函数，则
$$E[g(X)] = \int_{-\infty}^{+\infty} g(x) f(x)dx$$

**定义6.3（方差）**
$$Var(X) = E[(X - E[X])^2] = \int_{-\infty}^{+\infty} (x - E[X])^2 f(x)dx$$

**例6.2** 计算正态分布 $N(\mu, \sigma^2)$ 的期望和方差。

**解**：
$$
\begin{align*}
E[X] &= \int_{-\infty}^{+\infty} x \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}dx \\
&= \int_{-\infty}^{+\infty} (\mu + \sigma t) \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}dt \\
&= \mu \cdot \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{+\infty} e^{-\frac{t^2}{2}}dt + \frac{\sigma}{\sqrt{2\pi}} \int_{-\infty}^{+\infty} t e^{-\frac{t^2}{2}}dt \\
&= \mu
\end{align*}
$$

$$
\begin{align*}
Var(X) &= \int_{-\infty}^{+\infty} (x - \mu)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}dx \\
&= \sigma^2 \int_{-\infty}^{+\infty} t^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}dt \\
&= \sigma^2
\end{align*}
$$

### 6.3 累积分布函数

**定义6.4（累积分布函数）**
$$F(x) = P(X \leq x) = \int_{-\infty}^x f(t)dt$$

**性质**：$F'(x) = f(x)$（几乎处处）。

### 6.4 最大似然估计中的积分

在最大似然估计中，对数似然函数常涉及积分运算：
$$\mathcal{L}(\theta) = \prod_{i=1}^n f(x_i; \theta), \quad \ln\mathcal{L}(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)$$

**例6.3（高斯混合模型）** 高斯混合模型的似然函数：
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \sigma_k^2)$$
其中 $\pi_k \geq 0$，$\sum_{k=1}^K \pi_k = 1$。边缘似然：
$$\ln p(X) = \sum_{i=1}^n \ln\left(\sum_{k=1}^K \pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2)\right)$$

### 6.5 信息论中的积分

**定义6.5（微分熵）** 连续型随机变量 $X$ 的微分熵：
$$h(X) = -\int_{-\infty}^{+\infty} f(x) \log f(x)dx$$

**定义6.6（KL散度）** 两个概率分布 $p(x)$ 和 $q(x)$ 之间的KL散度：
$$D_{KL}(p\|q) = \int_{-\infty}^{+\infty} p(x) \log \frac{p(x)}{q(x)}dx$$

**例6.4** 两个正态分布 $p = \mathcal{N}(\mu_1, \sigma_1^2)$ 和 $q = \mathcal{N}(\mu_2, \sigma_2^2)$ 之间的KL散度：
$$D_{KL}(p\|q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 6.6 变分推断中的积分

在变分推断中，我们需要最小化：
$$\mathcal{L}(q) = \int q(z) \log \frac{q(z)}{p(z|x)}dz = D_{KL}(q(z)\|p(z|x))$$
或最大化证据下界（ELBO）：
$$\text{ELBO} = \int q(z) \log \frac{p(x, z)}{q(z)}dz = E_q[\log p(x, z)] + H(q)$$

---

## 习题

1. 计算下列不定积分：
   (1) $\int \frac{dx}{x^2(1 + x^2)}$
   (2) $\int e^{ax} \cos bx dx$
   (3) $\int \frac{\ln x}{x^3}dx$

2. 计算下列定积分：
   (1) $\int_0^{\pi/2} \sin^n x dx$
   (2) $\int_0^1 \arcsin x dx$
   (3) $\int_0^{\ln 2} \sqrt{e^x - 1}dx$

3. 讨论下列反常积分的敛散性：
   (1) $\int_0^{+\infty} e^{-ax} \sin bx dx$
   (2) $\int_0^1 \frac{\ln x}{1 - x}dx$

4. 求曲线 $y = \sin x$ 与 $y = \cos x$ 在 $[0, \pi]$ 内围成的面积。

5. 设随机变量 $X$ 服从指数分布，概率密度为 $f(x) = \lambda e^{-\lambda x} \ (x \geq 0, \lambda > 0)$，求 $E[X]$ 和 $Var(X)$。

---

**参考文献**：
1. 华东师范大学数学系. 数学分析（第四版）. 高等教育出版社, 2010.
2. 菲赫金哥尔茨. 微积分学教程. 人民教育出版社, 1978.
3. Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.

---

*最后更新：2026-05-13*

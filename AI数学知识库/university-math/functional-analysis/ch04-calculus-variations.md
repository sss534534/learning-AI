# 第4章 变分法


## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: 高等数学基础
- **关联文件**: 大学数学统一参考
- **最后更新**: 2026-06-12

---


变分法是数学分析的一个分支，主要研究泛函的极值问题。它在物理、工程、经济学和人工智能等领域有着广泛的应用。本章将介绍变分法的基本概念、主要理论以及在AI中的应用。

---

## 4.1 变分法的基本概念

### 4.1.1 泛函

**定义 4.1.1（泛函）**
设 $C$ 是一个由函数组成的集合（函数空间），如果对每个函数 $y(x) \in C$，都有一个确定的实数 $J[y]$ 与之对应，则称 $J[y]$ 是定义在 $C$ 上的**泛函**，记作：
$$J[y] = \int_{x_1}^{x_2} F(x, y(x), y'(x)) dx$$
其中 $F$ 称为泛函的**核函数**，它是 $x, y, y'$ 的已知函数，且具有二阶连续偏导数。

**例 4.1.1（最速降线问题）**
考虑在重力场中，质点从点 $A(x_1, y_1)$ 滑到点 $B(x_2, y_2)$ 的时间泛函：
$$J[y] = \int_{x_1}^{x_2} \sqrt{\frac{1 + y'^2}{2g(y_1 - y)}} dx$$
其中 $g$ 是重力加速度。

**例 4.1.2（两点间最短路径）**
平面上两点 $A(x_1, y_1)$ 和 $B(x_2, y_2)$ 之间的弧长泛函：
$$J[y] = \int_{x_1}^{x_2} \sqrt{1 + y'^2} dx$$
其极值函数为直线 $y = kx + b$。

### 4.1.2 变分

**定义 4.1.2（函数的变分）**
设 $y(x)$ 是容许函数，$\eta(x)$ 是任意光滑函数，满足边界条件 $\eta(x_1) = \eta(x_2) = 0$，则函数的**变分**定义为：
$$\delta y = \varepsilon \eta(x)$$
其中 $\varepsilon$ 是一个小参数。

**定义 4.1.3（泛函的变分）**
泛函 $J[y]$ 在 $y(x)$ 处的**一阶变分**定义为泛函增量的主部，即：
$$\delta J = \left. \frac{d}{d\varepsilon} J[y + \varepsilon \eta] \right|_{\varepsilon = 0}$$

**定理 4.1.1（一阶变分公式）**
对于泛函 $J[y] = \int_{x_1}^{x_2} F(x, y, y') dx$，其一阶变分为：
$$\delta J = \int_{x_1}^{x_2} \left( \frac{\partial F}{\partial y} - \frac{d}{dx} \frac{\partial F}{\partial y'} \right) \eta(x) dx$$

**证明：**
将 $J[y + \varepsilon \eta]$ 展开为：
$$J[y + \varepsilon \eta] = \int_{x_1}^{x_2} F(x, y + \varepsilon \eta, y' + \varepsilon \eta') dx$$
对 $\varepsilon$ 求导并令 $\varepsilon = 0$：
$$\left. \frac{dJ}{d\varepsilon} \right|_{\varepsilon = 0} = \int_{x_1}^{x_2} \left( \frac{\partial F}{\partial y} \eta + \frac{\partial F}{\partial y'} \eta' \right) dx$$
对第二项分部积分：
$$\int_{x_1}^{x_2} \frac{\partial F}{\partial y'} \eta' dx = \left. \frac{\partial F}{\partial y'} \eta \right|_{x_1}^{x_2} - \int_{x_1}^{x_2} \eta \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) dx$$
由于 $\eta(x_1) = \eta(x_2) = 0$，边界项为零，因此得证。

### 4.1.3 极值条件

**定理 4.1.2（泛函极值的必要条件）**
若泛函 $J[y]$ 在 $y(x)$ 处取得极值，则其一阶变分为零：
$$\delta J = 0$$

**定义 4.1.4（平稳函数）**
满足 $\delta J = 0$ 的函数 $y(x)$ 称为泛函 $J[y]$ 的**平稳函数**或**极值曲线**。

---

## 4.2 欧拉-拉格朗日方程

### 4.2.1 基本形式的推导

**定理 4.2.1（欧拉-拉格朗日方程）**
泛函 $J[y] = \int_{x_1}^{x_2} F(x, y, y') dx$ 取得极值的必要条件是 $y(x)$ 满足：
$$\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0$$
这个方程称为**欧拉-拉格朗日方程**（Euler-Lagrange equation）。

**证明：**
由定理 4.1.2，极值必要条件是 $\delta J = 0$，即：
$$\int_{x_1}^{x_2} \left( \frac{\partial F}{\partial y} - \frac{d}{dx} \frac{\partial F}{\partial y'} \right) \eta(x) dx = 0$$
对任意满足边界条件的 $\eta(x)$ 成立。根据变分法基本引理（若 $\int_{x_1}^{x_2} \Phi(x) \eta(x) dx = 0$ 对任意 $\eta(x)$ 成立，则 $\Phi(x) \equiv 0$），可得欧拉-拉格朗日方程。

**展开形式：**
将全导数展开，欧拉-拉格朗日方程可写为：
$$F_y - F_{y'x} - F_{y'y} y' - F_{y'y'} y'' = 0$$
这是一个二阶常微分方程。

### 4.2.2 几种特殊情形

**情形 1：$F$ 不显含 $y'$**
若 $F = F(x, y)$，则欧拉-拉格朗日方程简化为：
$$F_y(x, y) = 0$$
这是一个代数方程。

**情形 2：$F$ 不显含 $y$**
若 $F = F(x, y')$，则欧拉-拉格朗日方程简化为：
$$\frac{d}{dx} F_{y'}(x, y') = 0$$
积分得**首次积分**：
$$F_{y'}(x, y') = C$$
其中 $C$ 为常数。

**情形 3：$F$ 不显含 $x$**
若 $F = F(y, y')$，考虑表达式：
$$\frac{d}{dx} \left( F - y' F_{y'} \right) = F_y y' + F_{y'} y'' - y'' F_{y'} - y' \frac{d}{dx} F_{y'} = y' \left( F_y - \frac{d}{dx} F_{y'} \right)$$
由欧拉-拉格朗日方程，上式为零，因此有**首次积分**：
$$F(y, y') - y' F_{y'}(y, y') = C$$

**例 4.2.1（最速降线问题求解）**
最速降线的泛函为：
$$J[y] = \int_{0}^{a} \sqrt{\frac{1 + y'^2}{y}} dx \quad (y(0) = 0, y(a) = b)$$
这里 $F$ 不显含 $x$，使用情形 3 的首次积分：
$$\sqrt{\frac{1 + y'^2}{y}} - y' \cdot \frac{y'}{\sqrt{y(1 + y'^2)}} = C$$
化简得：
$$\frac{1}{\sqrt{y(1 + y'^2)}} = C$$
令 $C = 1/\sqrt{2r}$，则：
$$y(1 + y'^2) = 2r$$
用参数法求解，令 $y' = \cot \theta$，则 $y = 2r \sin^2 \theta = r(1 - \cos 2\theta)$，进一步可得 $x = r(2\theta - \sin 2\theta) + C_1$。由边界条件 $y(0) = 0$ 得 $C_1 = 0$。令 $\phi = 2\theta$，则参数方程为：
$$\begin{cases}
x = r(\phi - \sin \phi) \\
y = r(1 - \cos \phi)
\end{cases}$$
这是**摆线**（旋轮线）的参数方程。

### 4.2.3 含高阶导数的泛函

对于含高阶导数的泛函：
$$J[y] = \int_{x_1}^{x_2} F(x, y, y', y'', \dots, y^{(n)}) dx$$
其欧拉-拉格朗日方程为：
$$F_y - \frac{d}{dx} F_{y'} + \frac{d^2}{dx^2} F_{y''} - \dots + (-1)^n \frac{d^n}{dx^n} F_{y^{(n)}} = 0$$

### 4.2.4 多元函数的泛函

设 $u = u(x, y)$ 是二元函数，泛函：
$$J[u] = \iint_D F(x, y, u, u_x, u_y) dxdy$$
其欧拉-拉格朗日方程为：
$$F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} = 0$$

**例 4.2.2（最小曲面问题）**
曲面面积泛函：
$$J[u] = \iint_D \sqrt{1 + u_x^2 + u_y^2} dxdy$$
对应的欧拉-拉格朗日方程为**极小曲面方程**：
$$\frac{\partial}{\partial x} \left( \frac{u_x}{\sqrt{1 + u_x^2 + u_y^2}} \right) + \frac{\partial}{\partial y} \left( \frac{u_y}{\sqrt{1 + u_x^2 + u_y^2}} \right) = 0$$

---

## 4.3 带约束的变分问题

### 4.3.1 完整约束

**问题描述：**
在约束条件 $G(x, y, y') = 0$ 下，求泛函 $J[y] = \int_{x_1}^{x_2} F(x, y, y') dx$ 的极值。

**定理 4.3.1（拉格朗日乘子法）**
构造增广泛函：
$$J^*[y, \lambda] = \int_{x_1}^{x_2} \left[ F(x, y, y') + \lambda(x) G(x, y, y') \right] dx$$
其中 $\lambda(x)$ 称为**拉格朗日乘子**。极值的必要条件是 $J^*$ 的变分为零，即：
$$\begin{cases}
F_y + \lambda G_y - \frac{d}{dx}(F_{y'} + \lambda G_{y'}) = 0 \\
G(x, y, y') = 0
\end{cases}$$

### 4.3.2 等周问题

**定义 4.3.1（等周问题）**
在积分约束 $\int_{x_1}^{x_2} G(x, y, y') dx = l$（常数）下，求泛函 $J[y] = \int_{x_1}^{x_2} F(x, y, y') dx$ 的极值，称为**等周问题**。

**定理 4.3.2（等周问题的欧拉-拉格朗日方程）**
构造增广泛函：
$$J^*[y, \lambda] = \int_{x_1}^{x_2} \left[ F(x, y, y') + \lambda G(x, y, y') \right] dx$$
其中 $\lambda$ 为常数拉格朗日乘子。极值的必要条件是：
$$(F + \lambda G)_y - \frac{d}{dx}(F + \lambda G)_{y'} = 0$$

**例 4.3.1（经典等周问题）**
在周长固定的条件下，求面积最大的平面闭曲线。设曲线参数方程为 $(x(s), y(s))$，周长约束为 $\int_0^L \sqrt{x'^2 + y'^2} ds = L$，面积泛函为 $A = \frac{1}{2} \int_0^L (x y' - y x') ds$。构造增广泛函后可证明，极值曲线为圆。

### 4.3.3 多个约束条件

对于 $m$ 个约束条件 $G_1 = 0, G_2 = 0, \dots, G_m = 0$，引入 $m$ 个拉格朗日乘子 $\lambda_1(x), \dots, \lambda_m(x)$，构造增广泛函：
$$J^* = \int_{x_1}^{x_2} \left( F + \sum_{i=1}^m \lambda_i G_i \right) dx$$
极值条件为 $J^*$ 的变分为零。

---

## 4.4 最优控制问题

### 4.4.1 问题的表述

**最优控制问题的标准形式：**
给定状态方程：
$$\dot{x}(t) = f(x(t), u(t), t), \quad x(t_0) = x_0$$
其中 $x(t) \in \mathbb{R}^n$ 是状态向量，$u(t) \in \mathbb{R}^m$ 是控制向量。

目标是在容许控制集 $U$ 中选择 $u(t)$，使性能指标泛函：
$$J[u] = \phi(x(t_f), t_f) + \int_{t_0}^{t_f} L(x(t), u(t), t) dt$$
取极小值（或极大值），其中 $t_f$ 是终端时刻，$\phi$ 是终端代价，$L$ 是运行代价。

### 4.4.2 庞特里亚金极大值原理

**定义 4.4.1（哈密顿函数）**
定义**哈密顿函数**（Hamiltonian）：
$$H(x, u, \lambda, t) = L(x, u, t) + \lambda^T(t) f(x, u, t)$$
其中 $\lambda(t) \in \mathbb{R}^n$ 称为**协态向量**。

**定理 4.4.1（庞特里亚金极大值原理）**
设 $u^*(t)$ 是最优控制，$x^*(t)$ 是对应的最优状态轨线，则存在协态向量 $\lambda^*(t)$，满足：

1. **正则方程：**
   $$\begin{cases}
   \dot{x}^* = \frac{\partial H}{\partial \lambda}(x^*, u^*, \lambda^*, t) = f(x^*, u^*, t) \\
   \dot{\lambda}^* = -\frac{\partial H}{\partial x}(x^*, u^*, \lambda^*, t)
   \end{cases}$$

2. **极大值条件：**
   对所有容许控制 $u(t)$，有：
   $$H(x^*(t), u^*(t), \lambda^*(t), t) \geq H(x^*(t), u(t), \lambda^*(t), t)$$
   即最优控制使哈密顿函数取极大值（或极小值，依问题而定）。

3. **横截条件：**
   - 若 $x(t_f)$ 自由，则 $\lambda(t_f) = \frac{\partial \phi}{\partial x}(x(t_f), t_f)$
   - 若 $x(t_f)$ 固定，则 $\lambda(t_f)$ 自由
   - 若 $t_f$ 自由，则 $H(x(t_f), u(t_f), \lambda(t_f), t_f) + \frac{\partial \phi}{\partial t}(x(t_f), t_f) = 0$

**注：** 对于极小化问题，极大值条件变为极小值条件。

**例 4.4.1（线性二次调节器）**
考虑线性系统 $\dot{x} = Ax + Bu$，性能指标：
$$J[u] = \frac{1}{2} x^T(t_f) S x(t_f) + \frac{1}{2} \int_{t_0}^{t_f} (x^T Q x + u^T R u) dt$$
其中 $Q \succeq 0, R \succ 0, S \succeq 0$。哈密顿函数为：
$$H = \frac{1}{2}(x^T Q x + u^T R u) + \lambda^T (Ax + Bu)$$
由极大值条件，$\frac{\partial H}{\partial u} = R u + B^T \lambda = 0$，解得 $u^* = -R^{-1} B^T \lambda$。代入正则方程可得 Riccati 微分方程，最优控制为状态反馈 $u^* = -K(t) x$。

---

## 4.5 变分法在AI中的应用

### 4.5.1 变分推断

**问题背景：**
在贝叶斯推断中，我们需要计算后验分布 $p(z | x)$，其中 $z$ 是隐变量，$x$ 是观测数据。当精确推断不可行时，使用**变分推断**（Variational Inference），用简单的分布 $q(z)$ 近似真实后验 $p(z | x)$。

**变分自由能：**
定义**证据下界**（Evidence Lower Bound, ELBO）：
$$\mathcal{L}(q) = \mathbb{E}_{q(z)} [\log p(x, z)] - \mathbb{E}_{q(z)} [\log q(z)]$$
可以证明：
$$\log p(x) = \mathcal{L}(q) + \text{KL}(q(z) \| p(z | x))$$
其中 $\text{KL}(q \| p)$ 是 Kullback-Leibler 散度。由于 KL 散度非负，$\mathcal{L}(q)$ 是 $\log p(x)$ 的下界。

**变分优化：**
最大化 ELBO 等价于最小化 $\text{KL}(q(z) \| p(z | x))$。利用变分法，对 $q(z)$ 求导可得更新规则。在平均场变分推断中，假设 $q(z) = \prod_i q_i(z_i)$，可得到坐标上升更新公式。

### 4.5.2 强化学习策略梯度

**策略梯度定理：**
在强化学习中，设策略为 $\pi_\theta(a | s)$，目标是最大化期望回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$
其中 $\tau = (s_0, a_0, s_1, a_1, \dots)$ 是轨迹，$R(\tau)$ 是轨迹的总回报。

**策略梯度：**
利用变分思想，策略梯度可写为：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$
证明思路：将期望写为积分，交换微分与积分顺序，利用 $\nabla \log \pi = \nabla \pi / \pi$。

**REINFORCE 算法：**
基于策略梯度定理，REINFORCE 算法使用蒙特卡洛估计：
$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_{i,t} | s_{i,t}) \cdot R(\tau_i)$$

**优势 Actor-Critic：**
引入优势函数 $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$，策略梯度可改进为：
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a | s) \cdot A^\pi(s, a) \right]$$
这减小了梯度估计的方差。

### 4.5.3 扩散模型变分原理

**扩散模型简介：**
扩散模型是一类生成模型，通过逐步向数据添加高斯噪声，然后学习逆向去噪过程。

**变分下界：**
设前向扩散过程为 $q(x_{1:T} | x_0)$，逆向过程为 $p_\theta(x_{0:T})$，则数据对数似然的变分下界为：
$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} | x_0)} \left[ \log p_\theta(x_{0:T}) - \log q(x_{1:T} | x_0) \right]$$

**简化的训练目标：**
通过变分推导，可将训练目标简化为去噪分数匹配：
$$\mathcal{L}_\theta = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$
其中 $\epsilon$ 是添加的噪声，$\epsilon_\theta(x_t, t)$ 是模型预测的噪声。

**与变分法的联系：**
扩散模型的训练本质上是在变分框架下最大化数据对数似然的下界，通过优化变分自由能来学习生成模型。

---

## 习题

1. 求泛函 $J[y] = \int_0^1 (y'^2 + 12xy) dx$ 的极值曲线，边界条件 $y(0) = 0, y(1) = 1$。

2. 用欧拉-拉格朗日方程证明：在所有连接两点的曲线中，直线段最短。

3. 求解等周问题：在长度为 $L$ 的所有闭曲线中，求面积最大的曲线。

4. 考虑线性二次调节器问题 $\dot{x} = -x + u$，性能指标 $J = \frac{1}{2} \int_0^\infty (x^2 + u^2) dt$，求最优控制律。

5. 证明变分推断中 ELBO 与 KL 散度的关系：$\log p(x) = \mathcal{L}(q) + \text{KL}(q(z) \| p(z | x))$。

---

## 参考文献

1. Gelfand, I. M., & Fomin, S. V. (2000). *Calculus of Variations*. Dover Publications.
2. Pontryagin, L. S., et al. (1962). *The Mathematical Theory of Optimal Processes*. Interscience.
3. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*.
4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
5. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
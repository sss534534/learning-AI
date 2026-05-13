# 第十一章：强化学习与大模型对齐

> 强化学习（Reinforcement Learning, RL）是机器学习的三大范式之一，其核心思想是智能体通过与环境交互获得奖励信号来学习最优策略。近年来，强化学习在大模型对齐领域发挥了关键作用——从RLHF到DPO，从人类反馈到AI反馈，强化学习为让大模型"听话"且"安全"提供了数学基础。本章将系统讲解强化学习的数学理论，并深入推导RLHF、DPO等对齐方法的核心公式。

## 目录

1. [强化学习基础](#1-强化学习基础)
2. [策略梯度方法](#2-策略梯度方法)
3. [Actor-Critic框架](#3-actor-critic框架)
4. [RLHF的数学框架](#4-rlhf的数学框架)
5. [DPO的理论](#5-dpo的理论)
6. [其他对齐方法](#6-其他对齐方法)

---

## 1. 强化学习基础

### 1.1 马尔可夫决策过程（MDP）

**定义 11.1（马尔可夫决策过程）** 一个马尔可夫决策过程是一个五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$，其中：

- $\mathcal{S}$：状态空间（有限或无限）
- $\mathcal{A}$：动作空间（有限或无限）
- $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$：状态转移概率，$P(s'|s,a) = \Pr(s_{t+1}=s' | s_t=s, a_t=a)$
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$：奖励函数，$R(s,a) = \mathbb{E}[r_{t+1} | s_t=s, a_t=a]$
- $\gamma \in [0,1)$：折扣因子

**马尔可夫性质：** 状态转移满足马尔可夫性，即未来只依赖于当前状态和动作，与历史无关：

$$
\Pr(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = \Pr(s_{t+1} | s_t, a_t)
$$

**MDP的动态过程：** 智能体在每个时间步 $t$：
1. 观察当前状态 $s_t \in \mathcal{S}$
2. 根据策略选择动作 $a_t \in \mathcal{A}$
3. 环境返回奖励 $r_{t+1} = R(s_t, a_t)$
4. 环境转移到新状态 $s_{t+1} \sim P(\cdot|s_t, a_t)$

### 1.2 策略与价值函数

**定义 11.2（策略）** 策略 $\pi$ 是从状态到动作的映射：

- 确定性策略：$\pi: \mathcal{S} \to \mathcal{A}$
- 随机性策略：$\pi: \mathcal{S} \times \mathcal{A} \to [0,1]$，$\pi(a|s) = \Pr(a_t=a | s_t=s)$

**定义 11.3（状态价值函数）** 给定策略 $\pi$，状态 $s$ 的价值函数定义为从该状态出发的期望累积折扣奖励：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \bigg| s_0 = s \right]
$$

**定义 11.4（动作价值函数）** 给定策略 $\pi$，在状态 $s$ 采取动作 $a$ 的价值函数：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \bigg| s_0 = s, a_0 = a \right]
$$

**两者之间的关系：**

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)
$$

$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^\pi(s')
$$

### 1.3 贝尔曼方程

**定理 11.1（贝尔曼期望方程）** 价值函数满足递归关系：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^\pi(s') \right]
$$

$$
Q^\pi(s, a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s', a')
$$

**证明：** 由价值函数的定义展开：

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi \left[ r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots \bigg| s_0 = s \right] \\
&= \mathbb{E}_\pi \left[ r_1 + \gamma (r_2 + \gamma r_3 + \cdots) \bigg| s_0 = s \right] \\
&= \mathbb{E}_\pi \left[ r_1 + \gamma V^\pi(s_1) \bigg| s_0 = s \right] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a) + \gamma V^\pi(s') \right] \\
&= \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
\end{aligned}
$$

$\blacksquare$

**定义 11.5（最优价值函数）** 最优状态价值函数和最优动作价值函数：

$$
V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

**定理 11.2（贝尔曼最优方程）**

$$
V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^*(s') \right]
$$

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \max_{a' \in \mathcal{A}} Q^*(s',a')
$$

**贝尔曼算子：** 定义贝尔曼算子 $T^\pi$ 和 $T^*$：

$$
(T^\pi V)(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]
$$

$$
(T^* V)(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]
$$

**定理 11.3（压缩映射定理）** 当 $\gamma < 1$ 时，$T^\pi$ 和 $T^*$ 都是 $\|\cdot\|_\infty$ 意义下的压缩映射，即：

$$
\|T^\pi V_1 - T^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty
$$

因此由Banach不动点定理，贝尔曼方程存在唯一解，且可通过迭代收敛到该解。

### 1.4 代码示例：MDP与价值迭代

```python
import numpy as np

class MDP:
    def __init__(self, n_states, n_actions, transitions, rewards, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transitions
        self.R = rewards
        self.gamma = gamma

    def value_iteration(self, theta=1e-6):
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                V[s] = max(
                    self.R[s, a] + self.gamma * np.dot(self.P[s, a], V)
                    for a in range(self.n_actions)
                )
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            policy[s] = np.argmax([
                self.R[s, a] + self.gamma * np.dot(self.P[s, a], V)
                for a in range(self.n_actions)
            ])
        return V, policy

    def policy_evaluation(self, policy, theta=1e-6):
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                a = policy[s]
                V[s] = self.R[s, a] + self.gamma * np.dot(self.P[s, a], V)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    def policy_iteration(self):
        policy = np.zeros(self.n_states, dtype=int)
        while True:
            V = self.policy_evaluation(policy)
            policy_stable = True
            for s in range(self.n_states):
                old_action = policy[s]
                policy[s] = np.argmax([
                    self.R[s, a] + self.gamma * np.dot(self.P[s, a], V)
                    for a in range(self.n_actions)
                ])
                if old_action != policy[s]:
                    policy_stable = False
            if policy_stable:
                break
        return V, policy
```

---

## 2. 策略梯度方法

### 2.1 策略梯度定理

**目标函数：** 策略梯度的目标是最大化期望累积奖励。常用的目标函数有三种：

- **起始价值：** $J(\theta) = V^{\pi_\theta}(s_0)$
- **平均奖励：** $J(\theta) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=0}^{T} r_t\right]$
- **情节奖励：** $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$

其中 $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$ 表示一条轨迹。

**定理 11.4（策略梯度定理）** 对于任意可微策略 $\pi_\theta$，目标函数的梯度为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

**证明：** 考虑情节奖励目标函数 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$，其中 $R(\tau) = \sum_{t=0}^{T} \gamma^t r_{t+1}$。

轨迹的概率：

$$
\pi_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$

对目标函数求梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_\tau \pi_\theta(\tau) R(\tau) \\
&= \sum_\tau \nabla_\theta \pi_\theta(\tau) R(\tau) \\
&= \sum_\tau \pi_\theta(\tau) \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)} R(\tau) \\
&= \sum_\tau \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) R(\tau) \right]
\end{aligned}
$$

计算 $\nabla_\theta \log \pi_\theta(\tau)$：

$$
\begin{aligned}
\nabla_\theta \log \pi_\theta(\tau) &= \nabla_\theta \left[ \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log P(s_{t+1}|s_t,a_t) \right] \\
&= \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
\end{aligned}
$$

因为 $p(s_0)$ 和 $P(s_{t+1}|s_t,a_t)$ 不依赖于 $\theta$。

因此：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$

进一步，利用因果性（$t$ 时刻的动作不影响之前的奖励），可以简化为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$ 是从时间步 $t$ 开始的折扣回报。

由于 $\mathbb{E}_{\tau \sim \pi_\theta}[G_t | s_t, a_t] = Q^{\pi_\theta}(s_t, a_t)$，最终得到：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

$\blacksquare$

**关键洞察：** 策略梯度定理的核心思想是 **"对数导数技巧"**（log-derivative trick），将期望的梯度转化为对数概率梯度的期望：

$$
\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[f(x) \nabla_\theta \log p_\theta(x)]
$$

### 2.2 REINFORCE算法

**REINFORCE** 是最基础的策略梯度算法，使用蒙特卡洛采样估计梯度。

**算法 11.1（REINFORCE）**

1. 用当前策略 $\pi_\theta$ 采样轨迹 $\tau = (s_0, a_0, r_1, \ldots, s_T)$
2. 计算每个时间步的回报 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$
3. 更新参数：$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$

**梯度估计的无偏性：** REINFORCE的梯度估计是无偏的，因为：

$$
\mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] = \nabla_\theta J(\theta)
$$

**方差分析：** REINFORCE的主要问题是方差很大。回报 $G_t$ 的方差随时间步长指数增长：

$$
\text{Var}(G_t) = \text{Var}\left(\sum_{k=t}^{T} \gamma^{k-t} r_{k+1}\right) \leq \frac{\sigma_r^2}{(1-\gamma^2)}
$$

其中 $\sigma_r^2$ 是单步奖励的方差。

### 2.3 方差缩减技术

#### 2.3.1 基线方法

**定理 11.5** 在策略梯度中减去一个基线 $b(s_t)$ 不改变梯度的期望值：

$$
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right] = \nabla_\theta J(\theta)
$$

**证明：** 只需证明 $\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right] = 0$：

$$
\begin{aligned}
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right] &= \sum_{s_t} d^{\pi}(s_t) b(s_t) \sum_{a_t} \nabla_\theta \pi_\theta(a_t|s_t) \\
&= \sum_{s_t} d^{\pi}(s_t) b(s_t) \nabla_\theta \sum_{a_t} \pi_\theta(a_t|s_t) \\
&= \sum_{s_t} d^{\pi}(s_t) b(s_t) \nabla_\theta 1 \\
&= 0
\end{aligned}
$$

其中 $d^{\pi}(s)$ 是策略 $\pi$ 下的状态访问分布。$\blacksquare$

**最优基线：** 使方差最小的基线为：

$$
b^*(s) = \frac{\mathbb{E}_{a \sim \pi_\theta}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 G_t\right]}{\mathbb{E}_{a \sim \pi_\theta}\left[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\right]}
$$

实践中通常取 $b(s) = V^{\pi_\theta}(s)$ 作为基线。

#### 2.3.2 优势函数

**定义 11.6（优势函数）** 优势函数衡量动作 $a$ 相对于平均水平的好坏：

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
$$

**优势函数的性质：**

$$
\mathbb{E}_{a \sim \pi(\cdot|s)}[A^{\pi}(s,a)] = 0
$$

使用优势函数的策略梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$

优势函数的优势在于：$A^{\pi}(s,a) > 0$ 表示该动作优于平均，应增加其概率；$A^{\pi}(s,a) < 0$ 表示该动作劣于平均，应降低其概率。

### 2.4 代码示例：REINFORCE算法

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        dist = self.policy(torch.FloatTensor(state))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update(self, log_probs, returns):
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

---

## 3. Actor-Critic框架

### 3.1 优势函数估计（GAE）

**问题：** 优势函数 $A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)$ 的精确计算需要无穷步的回报，实践中需要估计。

**时序差分残差（TD残差）：** 定义 $\delta_t^V$ 为：

$$
\delta_t^V = r_{t+1} + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)
$$

**引理 11.1** 优势函数可以表示为TD残差的加权和：

$$
A^{\pi}(s_t, a_t) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

其中 $\lambda \in [0,1]$ 是GAE参数。

**证明：** 定义 $k$ 步回报的优势估计：

$$
\begin{aligned}
\hat{A}_t^{(1)} &= \delta_t^V = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t^{(2)} &= r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) - V(s_t) \\
&= \delta_t^V + \gamma \delta_{t+1}^V \\
\hat{A}_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V
\end{aligned}
$$

GAE是这些估计的指数加权平均：

$$
\begin{aligned}
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} &= (1-\lambda)\left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots\right) \\
&= (1-\lambda)\sum_{l=0}^{\infty} \lambda^l \hat{A}_t^{(l+1)} \\
&= (1-\lambda)\sum_{l=0}^{\infty} \lambda^l \sum_{m=0}^{l} \gamma^m \delta_{t+m}^V
\end{aligned}
$$

交换求和顺序：

$$
\begin{aligned}
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} &= (1-\lambda)\sum_{m=0}^{\infty} \gamma^m \delta_{t+m}^V \sum_{l=m}^{\infty} \lambda^l \\
&= (1-\lambda)\sum_{m=0}^{\infty} \gamma^m \delta_{t+m}^V \cdot \frac{\lambda^m}{1-\lambda} \\
&= \sum_{m=0}^{\infty} (\gamma\lambda)^m \delta_{t+m}^V
\end{aligned}
$$

$\blacksquare$

**GAE的偏差-方差权衡：**

| $\lambda$ 值 | 特点 | 等价于 |
|:---:|:---:|:---:|
| $\lambda = 0$ | 低方差、高偏差 | TD(0) |
| $\lambda = 1$ | 高方差、低偏差 | 蒙特卡洛 |

实践中通常取 $\lambda = 0.95$。

### 3.2 A2C/A3C算法

**A2C（Advantage Actor-Critic）** 同时训练策略网络（Actor）和价值网络（Critic）。

**Actor的目标：** 最大化策略梯度目标

$$
L_{\text{actor}} = \mathbb{E}_t \left[ \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]
$$

**Critic的目标：** 最小化价值函数的预测误差

$$
L_{\text{critic}} = \mathbb{E}_t \left[ \left( V_\phi(s_t) - G_t \right)^2 \right]
$$

**A3C（Asynchronous Advantage Actor-Critic）** 是A2C的异步版本，多个工作线程并行与环境交互，异步更新全局网络参数。

**A3C与A2C的区别：**

| 特性 | A2C | A3C |
|:---:|:---:|:---:|
| 更新方式 | 同步 | 异步 |
| 数据利用 | 等待所有线程完成 | 即时更新 |
| 训练稳定性 | 更稳定 | 可能不稳定 |
| 实现复杂度 | 简单 | 较复杂 |

**熵正则化：** 为鼓励探索，通常在Actor目标中加入策略熵的正则项：

$$
L_{\text{actor}} = \mathbb{E}_t \left[ \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t + c_{\text{ent}} H(\pi_\theta(\cdot|s_t)) \right]
$$

其中 $H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ 是策略的熵。

### 3.3 PPO算法的数学推导

**PPO（Proximal Policy Optimization）** 是目前RLHF中最常用的算法，其核心思想是限制策略更新的幅度，保证训练的稳定性。

#### 3.3.1 从策略梯度到信赖域方法

**问题：** 策略梯度的大步更新可能导致策略崩溃。TRPO通过约束KL散度来限制更新步长：

$$
\max_\theta \mathbb{E}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}_t\left[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]\right] \leq \delta
$$

TRPO的约束优化问题求解复杂，PPO通过裁剪替代目标简化了这一过程。

#### 3.3.2 重要性采样比

定义概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

则策略梯度目标可以重写为：

$$
L^{\text{IS}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right]
$$

**问题：** 当 $r_t(\theta)$ 远离1时，重要性采样估计的方差急剧增大，且可能导致策略更新过大。

#### 3.3.3 PPO-Clip目标函数

**定义 11.7（PPO裁剪目标函数）**

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

**裁剪机制分析：**

| 情况 | 条件 | 效果 |
|:---:|:---:|:---:|
| $\hat{A}_t > 0$（好动作） | $r_t(\theta)$ 上限被裁剪为 $1+\epsilon$ | 防止过度增加好动作的概率 |
| $\hat{A}_t < 0$（坏动作） | $r_t(\theta)$ 下限被裁剪为 $1-\epsilon$ | 防止过度减少坏动作的概率 |

**详细推导：**

当 $\hat{A}_t > 0$ 时，目标函数为：

$$
L_+ = \min(r_t(\theta) \hat{A}_t, (1+\epsilon) \hat{A}_t)
$$

- 若 $r_t(\theta) \leq 1+\epsilon$：$L_+ = r_t(\theta) \hat{A}_t$，正常梯度更新
- 若 $r_t(\theta) > 1+\epsilon$：$L_+ = (1+\epsilon) \hat{A}_t$，梯度为0，停止增大

当 $\hat{A}_t < 0$ 时，目标函数为：

$$
L_- = \min(r_t(\theta) \hat{A}_t, (1-\epsilon) \hat{A}_t)
$$

注意 $\hat{A}_t < 0$，所以 $\min$ 取更负的值：
- 若 $r_t(\theta) \geq 1-\epsilon$：$L_- = r_t(\theta) \hat{A}_t$，正常梯度更新
- 若 $r_t(\theta) < 1-\epsilon$：$L_- = (1-\epsilon) \hat{A}_t$，梯度为0，停止减小

#### 3.3.4 PPO的完整目标函数

PPO的完整目标函数结合了策略目标、价值函数损失和熵奖励：

$$
L(\theta) = \mathbb{E}_t \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H(\pi_\theta(\cdot|s_t)) \right]
$$

其中：
- $L^{\text{VF}}(\theta) = \left( V_\theta(s_t) - G_t \right)^2$ 是价值函数损失
- $c_1 = 0.5$，$c_2 = 0.01$ 是超参数
- $H(\pi_\theta(\cdot|s_t))$ 是策略熵，鼓励探索

**PPO与TRPO的关系：** PPO-Clip可以看作TRPO信赖域约束的一阶近似。当 $\epsilon$ 足够小时，裁剪目标函数等价于在信赖域内优化。

**定理 11.6** 设 $\epsilon^* = \max_\theta \epsilon$ 使得 $L^{\text{CLIP}}(\theta) = L^{\text{IS}}(\theta)$，则 $\epsilon^*$ 对应的KL散度约束与TRPO的约束等价。

### 3.4 代码示例：PPO算法

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return Categorical(logits=logits), value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values[:-1])
        return advantages, returns

    def ppo_loss(self, states, actions, old_log_probs, advantages, returns):
        dist, values = self.model(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values.squeeze(), returns)

        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
        return loss

    def update(self, buffer, epochs=10, batch_size=64):
        states = torch.FloatTensor(buffer['states'])
        actions = torch.LongTensor(buffer['actions'])
        old_log_probs = torch.FloatTensor(buffer['log_probs'])
        advantages = buffer['advantages']
        returns = buffer['returns']

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), batch_size):
                idx = indices[start:start+batch_size]
                loss = self.ppo_loss(
                    states[idx], actions[idx], old_log_probs[idx],
                    advantages[idx], returns[idx]
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
```

---

## 4. RLHF的数学框架

### 4.1 问题设定

在大模型对齐中，强化学习的设定与传统RL有所不同：

- **状态 $s$**：提示词（prompt）$x$
- **动作 $a$**：生成的回复 $y$
- **策略 $\pi_\theta$**：语言模型 $\pi_\theta(y|x)$
- **奖励 $r$**：由奖励模型给出的标量分数 $r_\phi(x, y)$
- **转移**：确定性（生成token后确定下一个状态）

**目标：** 在保持与参考策略（SFT模型）$\pi_{\text{ref}}$ 接近的前提下，最大化奖励模型的评分。

### 4.2 奖励模型训练（Bradley-Terry模型）

#### 4.2.1 Bradley-Terry模型

**定义 11.8（Bradley-Terry模型）** 给定两个回复 $y_1, y_2$，人类偏好 $y_1 \succ y_2$ 的概率为：

$$
P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2)) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))}
$$

其中 $\sigma(\cdot)$ 是sigmoid函数，$r(x, y)$ 是奖励函数。

**Bradley-Terry模型的性质：**

1. **传递性：** 若 $P(y_1 \succ y_2) > 0.5$ 且 $P(y_2 \succ y_3) > 0.5$，则 $P(y_1 \succ y_3) > 0.5$
2. **对称性：** $P(y_1 \succ y_2) + P(y_2 \succ y_1) = 1$
3. **Luce选择公理：** 偏好概率与奖励值的指数成正比

#### 4.2.2 奖励模型的训练

给定偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，其中 $y_w$ 是被偏好的回复（winner），$y_l$ 是不被偏好的回复（loser）。

**最大似然目标：**

$$
\begin{aligned}
\mathcal{L}_R(r_\phi) &= -\sum_{i=1}^N \log P(y_w^{(i)} \succ y_l^{(i)} | x^{(i)}) \\
&= -\sum_{i=1}^N \log \sigma\left(r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)})\right)
\end{aligned}
$$

**梯度：**

$$
\nabla_\phi \mathcal{L}_R = -\sum_{i=1}^N \left(1 - \sigma\left(r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)})\right)\right) \nabla_\phi \left(r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)})\right)
$$

**直觉：** 当奖励模型正确预测偏好（$r_\phi(x, y_w) \gg r_\phi(x, y_l)$）时，$\sigma(\cdot) \approx 1$，梯度接近0；当预测错误时，梯度较大，推动模型修正。

#### 4.2.3 奖励模型的实现

奖励模型通常基于预训练语言模型，在最后一层隐藏状态上添加一个线性头：

$$
r_\phi(x, y) = V_\phi(h_\phi(x, y))
$$

其中 $h_\phi$ 是语言模型的最后一层隐藏状态，$V_\phi$ 是线性投影层 $V_\phi: \mathbb{R}^d \to \mathbb{R}$。

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(last_hidden)
        return reward.squeeze(-1)

def reward_model_loss(reward_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    r_chosen = reward_model(chosen_ids, chosen_mask)
    r_rejected = reward_model(rejected_ids, rejected_mask)
    loss = -torch.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

### 4.3 KL散度约束的优化问题

#### 4.3.1 KL约束的必要性

直接最大化奖励会导致：

1. **奖励黑客（Reward Hacking）**：模型学会利用奖励模型的漏洞
2. **模式坍塌（Mode Collapse）**：模型丧失多样性
3. **语言退化**：生成不自然、重复的文本

因此需要约束策略与参考策略的KL散度。

#### 4.3.2 KL约束的优化问题

**问题 11.1（KL约束优化）**

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)] \right]
$$

其中 $\beta > 0$ 控制KL惩罚的强度。

**KL散度的展开：**

$$
\text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)] = \sum_y \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

#### 4.3.3 最优解的形式

**定理 11.7** KL约束优化问题的最优策略具有如下形式：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是配分函数。

**证明：** 使用拉格朗日方法。构造拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda) = \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta \text{KL}[\pi \| \pi_{\text{ref}}] - \lambda\left(\sum_y \pi(y|x) - 1\right)
$$

对 $\pi(y|x)$ 求变分导数并令其为0：

$$
\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x,y) - \beta \left(\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + 1\right) - \lambda = 0
$$

解得：

$$
\begin{aligned}
\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} &= \frac{1}{\beta}(r(x,y) - \lambda - \beta) \\
\pi(y|x) &= \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y) - \lambda - \beta}{\beta}\right)
\end{aligned}
$$

由归一化条件 $\sum_y \pi(y|x) = 1$，令 $Z(x) = \exp\left(\frac{\lambda + \beta}{\beta}\right)$，得：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

$\blacksquare$

**关键推论：** 最优策略可以表示为参考策略乘以奖励的指数加权，这意味着：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

即奖励函数（至多相差一个与 $y$ 无关的常数）可以由最优策略和参考策略的对数比恢复。这是DPO的理论基础。

### 4.4 PPO在RLHF中的应用

#### 4.4.1 RLHF中的PPO目标函数

在RLHF中，PPO的目标函数修改为：

$$
L_{\text{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中概率比为：

$$
r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})}
$$

**优势函数的计算：**

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}
$$

其中TD残差为：

$$
\delta_t = r_\phi(x, y) - \beta \left(\log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t})\right) + \gamma V(s_{t+1}) - V(s_t)
$$

这里KL惩罚项被整合到了奖励中。

#### 4.4.2 RLHF-PPO的完整训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RLHF训练流程                              │
│                                                             │
│  1. SFT阶段：在指令数据上微调基础模型 → π_ref (参考策略)     │
│                                                             │
│  2. RM阶段：在偏好数据上训练奖励模型 → r_φ                   │
│     L_RM = -E[log σ(r(x,y_w) - r(x,y_l))]                 │
│                                                             │
│  3. RL阶段：用PPO优化策略模型 → π_θ                         │
│     reward = r_φ(x,y) - β·KL[π_θ || π_ref]                │
│     使用PPO-Clip更新策略                                     │
│                                                             │
│  循环：生成 → 评分 → 优势估计 → PPO更新                      │
└─────────────────────────────────────────────────────────────┘
```

```python
class RLHFTrainer:
    def __init__(self, policy_model, ref_model, reward_model, value_model,
                 tokenizer, beta=0.04, clip_eps=0.2, gamma=1.0, lam=0.95):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

    def compute_reward_with_kl(self, prompts, responses):
        r = self.reward_model(prompts, responses)
        log_pi = self.policy.log_prob(prompts, responses)
        with torch.no_grad():
            log_ref = self.ref_model.log_prob(prompts, responses)
        kl_penalty = self.beta * (log_pi - log_ref).sum(-1)
        return r - kl_penalty

    def train_step(self, prompts):
        responses = self.policy.generate(prompts)
        rewards = self.compute_reward_with_kl(prompts, responses)
        values = self.value_model(prompts, responses)

        advantages, returns = self.compute_gae(rewards, values)
        old_log_probs = self.policy.log_prob(prompts, responses).detach()

        for _ in range(self.ppo_epochs):
            new_log_probs = self.policy.log_prob(prompts, responses)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(self.value_model(prompts, responses), returns)

            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

---

## 5. DPO的理论

### 5.1 从RLHF到DPO的等价推导

**DPO（Direct Preference Optimization）** 的核心洞察是：RLHF中的奖励函数可以被隐式地表示为策略和参考策略的对数比，从而绕过显式的奖励模型训练和RL优化。

#### 5.1.1 奖励函数的隐式表示

由定理11.7，KL约束优化的最优策略为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

两边取对数并整理：

$$
\begin{aligned}
\log \pi^*(y|x) &= \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x) \\
\frac{1}{\beta} r(x, y) &= \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x) \\
r(x, y) &= \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
\end{aligned}
$$

**关键观察：** $\log Z(x)$ 只依赖于 $x$，不依赖于 $y$。在Bradley-Terry模型中，偏好概率只依赖于奖励的差值：

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

因此 $\beta \log Z(x)$ 项在差值中被消去：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

**这表明：** 我们不需要显式地学习奖励函数 $r(x,y)$，可以直接用策略模型参数化偏好。

#### 5.1.2 偏好数据的似然函数

将隐式奖励代入Bradley-Terry模型：

$$
\begin{aligned}
P(y_w \succ y_l | x) &= \sigma\left(r(x, y_w) - r(x, y_l)\right) \\
&= \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \\
&= \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \cdot \frac{\pi_{\text{ref}}(y_l|x)}{\pi^*(y_l|x)}\right)
\end{aligned}
$$

### 5.2 DPO损失函数推导

**定义 11.9（DPO损失函数）** 给定偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$，DPO的损失函数为：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

**推导过程：**

1. 从RLHF的KL约束优化问题出发，最优策略满足：

$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

2. 将此代入Bradley-Terry偏好模型的最大似然目标：

$$
\mathcal{L} = -\mathbb{E}\left[\log \sigma(r^*(x, y_w) - r^*(x, y_l))\right]
$$

3. 奖励差值中 $\beta \log Z(x)$ 被消去：

$$
r^*(x, y_w) - r^*(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

4. 用可训练策略 $\pi_\theta$ 替换最优策略 $\pi^*$，得到DPO损失函数。

**DPO梯度的分析：**

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\text{DPO}} &= -\mathbb{E}\left[\left(1 - \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l))\right) \beta \left(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right)\right]
\end{aligned}
$$

其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是隐式奖励。

**梯度直觉：**
- 当模型正确偏好 $y_w$ 时，$\sigma(\cdot) \approx 1$，梯度接近0
- 当模型错误偏好 $y_l$ 时，$\sigma(\cdot) \approx 0$，梯度增大 $\log \pi_\theta(y_w|x)$ 并减小 $\log \pi_\theta(y_l|x)$
- 参考模型 $\pi_{\text{ref}}$ 不参与梯度计算，起到锚定作用

### 5.3 DPO与RLHF的等价性

**定理 11.8（DPO-RLHF等价性）** 在以下条件下，DPO的最优解与KL约束RLHF的最优解等价：

1. 奖励模型族足够丰富，可以表示 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + c(x)$
2. 偏好数据由同一奖励函数生成
3. 策略族包含最优策略

**证明思路：**

1. KL约束RLHF的最优策略 $\pi^*$ 满足 $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$
2. 此策略对应的隐式奖励为 $\hat{r}(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$
3. 在Bradley-Terry模型下，$\hat{r}$ 与真实奖励 $r$ 产生相同的偏好概率（因为差值相同）
4. 因此最大化DPO似然等价于找到产生正确偏好的策略，即RLHF的最优策略

$\blacksquare$

### 5.4 DPO的变体与改进

#### 5.4.1 IPO（Identity Preference Optimization）

IPO用二次损失替代log-sigmoid损失，避免了对偏好概率的过度自信：

$$
\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]
$$

#### 5.4.2 KTO（Kahneman-Tversky Optimization）

KTO不需要成对偏好数据，只需要二元反馈（好/坏），基于前景理论：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)} \left[ \lambda_w \cdot \mathbf{1}[y \sim y_w] \cdot (1 - v(x,y)) + \lambda_l \cdot \mathbf{1}[y \sim y_l] \cdot v(x,y) \right]
$$

其中 $v(x,y) = \sigma(\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0)$ 是价值函数，$z_0$ 是参考点。

#### 5.4.3 ORPO（Odds Ratio Preference Optimization）

ORPO将SFT和对齐合并为一个训练阶段，使用胜率比：

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{OR}}
$$

其中胜率比损失为：

$$
\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \frac{\text{odds}(\pi_\theta(y_w|x))}{\text{odds}(\pi_\theta(y_l|x))}\right)
$$

### 5.5 代码示例：DPO训练

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards

class DPOTrainer:
    def __init__(self, policy_model, ref_model, tokenizer, beta=0.1, lr=1e-6):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def get_log_probs(self, model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        per_token_logps = -outputs.loss.unsqueeze(-1)
        return per_token_logps.sum(-1)

    def train_step(self, batch):
        chosen_ids = batch['chosen_input_ids']
        chosen_mask = batch['chosen_attention_mask']
        rejected_ids = batch['rejected_input_ids']
        rejected_mask = batch['rejected_attention_mask']

        policy_chosen_logps = self.get_log_probs(
            self.policy, chosen_ids, chosen_mask, chosen_ids)
        policy_rejected_logps = self.get_log_probs(
            self.policy, rejected_ids, rejected_mask, rejected_ids)

        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, chosen_ids, chosen_mask, chosen_ids)
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, rejected_ids, rejected_mask, rejected_ids)

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }
```

---

## 6. 其他对齐方法

### 6.1 RLAIF（AI反馈强化学习）

#### 6.1.1 动机与定义

RLAIF（Reinforcement Learning from AI Feedback）用AI模型替代人类标注者提供偏好反馈，解决了RLHF中人类标注成本高、一致性差的问题。

**RLAIF与RLHF的对比：**

| 特性 | RLHF | RLAIF |
|:---:|:---:|:---:|
| 反馈来源 | 人类标注者 | AI模型 |
| 标注成本 | 高 | 低 |
| 一致性 | 受标注者主观影响 | 可控 |
| 可扩展性 | 受限于人力 | 几乎无限 |
| 潜在问题 | 标注者偏差 | AI偏差放大 |

#### 6.1.2 RLAIF的数学框架

RLAIF的核心是用AI模型（如GPT-4）生成偏好标签。给定提示 $x$ 和两个回复 $y_1, y_2$：

$$
P_{\text{AI}}(y_1 \succ y_2 | x) = \text{AI-Model}(y_1 \text{ is better than } y_2 | x, y_1, y_2)
$$

然后用与RLHF相同的Bradley-Terry模型训练奖励模型：

$$
\mathcal{L}_{\text{RLAIF}} = -\mathbb{E}_{(x, y_1, y_2)} \left[ P_{\text{AI}}(y_1 \succ y_2 | x) \log \sigma(r(x, y_1) - r(x, y_2)) + P_{\text{AI}}(y_2 \succ y_1 | x) \log \sigma(r(x, y_2) - r(x, y_1)) \right]
$$

**软标签 vs 硬标签：** RLAIF可以使用AI模型输出的概率作为软标签，而非二元硬标签，提供更丰富的信号。

#### 6.1.3 自我RLAIF

更激进的方案是让被训练的模型自己提供反馈（Self-RLAIF）：

$$
\text{Policy} \xrightarrow{\text{生成}} (y_1, y_2) \xrightarrow{\text{自我评判}} P(y_1 \succ y_2) \xrightarrow{\text{训练}} \text{Policy}'
$$

**循环偏差问题：** 自我RLAIF可能导致偏差累积，需要通过以下方式缓解：
- 使用不同温度采样增加多样性
- 引入外部验证信号
- 限制迭代轮次

### 6.2 Constitutional AI（CAI）

#### 6.2.1 基本框架

Constitutional AI由Anthropic提出，通过一组"宪法原则"指导AI的自我改进：

```
┌────────────────────────────────────────────────────┐
│              Constitutional AI流程                   │
│                                                    │
│  阶段1：监督学习（SL）                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ 有害提示  │ → │ AI生成回复 │ → │ 宪法评判  │     │
│  └──────────┘    └──────────┘    └─────┬────┘     │
│                                        │          │
│                                        ▼          │
│                               ┌──────────────┐    │
│                               │ 修订后的回复  │    │
│                               └──────┬───────┘    │
│                                      │            │
│                                      ▼            │
│                               ┌──────────────┐    │
│                               │ SFT微调模型  │    │
│                               └──────────────┘    │
│                                                    │
│  阶段2：RLAIF                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ 提示     │ → │ 两个回复  │ → │ 宪法评判  │     │
│  └──────────┘    └──────────┘    └─────┬────┘     │
│                                        │          │
│                                        ▼          │
│                               ┌──────────────┐    │
│                               │ 偏好数据     │    │
│                               └──────┬───────┘    │
│                                      │            │
│                                      ▼            │
│                               ┌──────────────┐    │
│                               │ RL优化       │    │
│                               └──────────────┘    │
└────────────────────────────────────────────────────┘
```

#### 6.2.2 宪法评判的数学形式

给定宪法原则集合 $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$，评判过程为：

$$
P_{\text{CAI}}(y_1 \succ y_2 | x, \mathcal{C}) = \prod_{k=1}^{K} P(y_1 \succ y_2 | x, c_k)^{\alpha_k}
$$

其中 $\alpha_k$ 是第 $k$ 条原则的权重。

**评判提示模板：**

```
请根据以下原则评判哪个回复更好：

原则：{constitution_principle}

提示：{prompt}
回复A：{response_1}
回复B：{response_2}

请选择更好的回复并说明理由。
```

#### 6.2.3 CAI与RLHF的理论关系

CAI可以看作RLHF的推广，其中奖励信号不是来自人类，而是来自AI模型基于宪法原则的评判：

$$
r_{\text{CAI}}(x, y) = \mathbb{E}_{c \sim \mathcal{C}} \left[ r_{\text{AI}}(x, y | c) \right]
$$

CAI的优势在于：
1. **可解释性**：宪法原则是显式的、可审查的
2. **可控性**：修改宪法原则即可改变模型行为
3. **一致性**：AI评判比人类标注更一致

### 6.3 KTO的理论基础

#### 6.3.1 从前景理论到KTO

KTO（Kahneman-Tversky Optimization）的灵感来自行为经济学中的前景理论（Prospect Theory）。

**前景理论的核心思想：**

1. **参考依赖（Reference Dependence）**：人们根据相对于参考点的变化评估结果，而非绝对值
2. **损失厌恶（Loss Aversion）**：损失的痛苦大于等量收益的快乐
3. **边际递减（Diminishing Sensitivity）**：远离参考点后，边际效用递减

**价值函数：**

$$
v(z) = \begin{cases} z^\alpha & \text{if } z \geq 0 \\ -\lambda(-z)^\beta & \text{if } z < 0 \end{cases}
$$

其中 $\alpha, \beta \in (0,1)$ 控制边际递减，$\lambda > 1$ 控制损失厌恶。

#### 6.3.2 KTO损失函数

KTO不需要成对偏好数据 $(y_w, y_l)$，只需要非成对的信号 $y$ 是否可接受：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)} \left[ w(x,y) \cdot \left(1 - v\left(\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0\right)\right) \right]
$$

其中：
- $z_0$ 是参考点（通常取0或期望KL散度）
- $v(\cdot)$ 是前景理论的价值函数
- $w(x,y)$ 是权重：

$$
w(x,y) = \begin{cases} \lambda_w & \text{if } y \text{ is desirable} \\ \lambda_l & \text{if } y \text{ is undesirable} \end{cases}
$$

**KTO的具体形式：** 在实践中，KTO使用简化的价值函数：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)} \left[ \begin{cases} \lambda_w \left(1 - \sigma\left(\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0\right)\right) & \text{if } y \text{ is desirable} \\ \lambda_l \cdot \sigma\left(\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_0\right) & \text{if } y \text{ is undesirable} \end{cases} \right]
$$

#### 6.3.3 KTO的理论优势

**定理 11.9** KTO在以下条件下是DPO的保守估计：

1. 当偏好数据完全一致（$y_w$ 总是优于 $y_l$）时，KTO退化为DPO
2. 当偏好数据有噪声时，KTO比DPO更鲁棒

**KTO的优势：**

| 特性 | DPO | KTO |
|:---:|:---:|:---:|
| 数据需求 | 成对偏好 | 非成对信号 |
| 数据量 | 较少（配对困难） | 较多（容易获取） |
| 对噪声鲁棒性 | 较弱 | 较强（前景理论） |
| 损失厌恶 | 无 | 有 |

### 6.4 对齐方法的理论比较

**统一视角：** 所有对齐方法都可以看作在解决以下优化问题：

$$
\max_{\pi_\theta} \mathbb{E}_{x,y \sim \pi_\theta} \left[ f(r(x,y)) \right] - \beta \cdot D(\pi_\theta \| \pi_{\text{ref}})
$$

其中 $f$ 是效用函数，$D$ 是散度度量。

| 方法 | 效用函数 $f$ | 散度 $D$ | 数据类型 |
|:---:|:---:|:---:|:---:|
| RLHF | $r(x,y)$ | KL散度 | 偏好对 |
| DPO | 隐式（对数比） | 隐式KL | 偏好对 |
| IPO | 二次损失 | KL散度 | 偏好对 |
| KTO | 前景理论价值函数 | 隐式KL | 二元信号 |
| RLAIF | AI评分 | KL散度 | AI偏好对 |
| CAI | 宪法评判 | KL散度 | AI偏好对 |

### 6.5 代码示例：KTO训练

```python
import torch
import torch.nn.functional as F

def kto_loss(policy_logps, reference_logps, desirable_mask, beta=0.1,
             lambda_w=1.0, lambda_l=1.5):
    kl = policy_logps - reference_logps
    rewards = beta * kl

    desirable_loss = lambda_w * (1 - F.sigmoid(rewards[desirable_mask]))
    undesirable_loss = lambda_l * F.sigmoid(rewards[~desirable_mask])

    if desirable_mask.sum() > 0 and (~desirable_mask).sum() > 0:
        loss = (desirable_loss.mean() + undesirable_loss.mean()) / 2
    elif desirable_mask.sum() > 0:
        loss = desirable_loss.mean()
    else:
        loss = undesirable_loss.mean()

    return loss

class KTOTrainer:
    def __init__(self, policy_model, ref_model, beta=0.1, lr=1e-6,
                 lambda_w=1.0, lambda_l=1.5):
        self.policy = policy_model
        self.ref_model = ref_model
        self.beta = beta
        self.lambda_w = lambda_w
        self.lambda_l = lambda_l
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        desirable = batch['desirable']

        policy_logps = self.policy(input_ids, attention_mask).log_probs.sum(-1)
        with torch.no_grad():
            ref_logps = self.ref_model(input_ids, attention_mask).log_probs.sum(-1)

        desirable_mask = torch.BoolTensor(desirable)
        loss = kto_loss(policy_logps, ref_logps, desirable_mask,
                        beta=self.beta, lambda_w=self.lambda_w, lambda_l=self.lambda_l)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {'loss': loss.item()}
```

---

## 本章小结

本章系统介绍了强化学习与大模型对齐的数学理论：

1. **强化学习基础**：MDP的形式化定义、策略与价值函数、贝尔曼方程及其压缩映射性质
2. **策略梯度方法**：策略梯度定理的完整推导、REINFORCE算法、方差缩减技术（基线方法、优势函数）
3. **Actor-Critic框架**：GAE的推导与偏差-方差权衡、A2C/A3C算法、PPO裁剪目标函数的数学推导
4. **RLHF的数学框架**：Bradley-Terry偏好模型、KL散度约束优化的最优解形式、PPO在RLHF中的应用
5. **DPO的理论**：从RLHF最优策略到隐式奖励的等价推导、DPO损失函数的完整推导、DPO与RLHF的等价性证明
6. **其他对齐方法**：RLAIF的AI反馈框架、Constitutional AI的宪法评判机制、KTO的前景理论基础

**核心脉络：** 从MDP → 策略梯度 → Actor-Critic → PPO → RLHF → DPO → KTO，强化学习的理论发展始终围绕"如何在保持稳定性的前提下最大化奖励"这一核心问题。DPO的突破在于将RLHF的两阶段过程（奖励模型 + RL优化）统一为单一的偏好优化问题，而KTO进一步将数据需求从成对偏好降低为二元信号，使对齐更加高效。

**关键公式速查：**

| 公式 | 表达 |
|:---:|:---:|
| 贝尔曼方程 | $V^\pi(s) = \sum_a \pi(a|s)[R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V^\pi(s')]$ |
| 策略梯度 | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a\|s) A^\pi(s,a)]$ |
| GAE | $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$ |
| PPO-Clip | $L^{\text{CLIP}} = \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)]$ |
| RLHF最优策略 | $\pi^*(y\|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y\|x) \exp(r(x,y)/\beta)$ |
| DPO损失 | $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \beta \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)})]$ |

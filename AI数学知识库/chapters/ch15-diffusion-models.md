# 第十五章：扩散模型数学基础

> 扩散模型（Diffusion Models）是近年来生成式AI领域最具革命性的突破之一，从DALL-E 2、Stable Diffusion到Midjourney，扩散模型已成为图像生成的主流技术。本章将深入讲解扩散模型的数学基础，包括前向扩散过程、逆向去噪过程、DDPM数学推导、Flow Matching以及条件生成数学。

## 目录

1. [前向扩散过程](#1-前向扩散过程)
2. [逆向去噪过程](#2-逆向去噪过程)
3. [DDPM数学推导](#3-ddpm数学推导)
4. [Flow Matching](#4-flow-matching)
5. [条件生成数学](#5-条件生成数学)

---

## 1. 前向扩散过程

### 1.1 扩散模型的基本思想

**扩散模型** 的灵感来自非平衡热力学：通过逐步向数据添加噪声，将复杂数据分布转化为简单的高斯分布，然后学习逆向过程以生成新样本。

```
    前向过程（加噪）                    逆向过程（去噪）
    
    x₀ (真实数据)                      x_T (纯噪声)
         │                                  │
         ▼                                  ▼
    x₁ = √(1-β₁)x₀ + √β₁ε₁            x_{T-1} = μ_θ(x_T, T)
         │                                  │
         ▼                                  ▼
        ...                              ...
         │                                  │
         ▼                                  ▼
    x_T (纯噪声)                       x₀ (生成数据)
```

### 1.2 离散时间马尔可夫扩散过程

**前向扩散过程** 是一个固定的马尔可夫链，逐步向数据添加高斯噪声：

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

其中 $\beta_t$ 是噪声调度参数，$t = 1, 2, \ldots, T$。

**马尔可夫性质：**
$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})
$$

### 1.3 重参数化与闭式解

**关键洞察：** 通过重参数化技巧，可以直接从 $\mathbf{x}_0$ 采样任意时刻的 $\mathbf{x}_t$。

定义：
$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

**闭式采样公式：**
$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

**推导过程：**

由递推关系：
$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

展开：
$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}\right) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}
\end{aligned}
$$

由于两个独立高斯的和仍为高斯：
$$
\sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}\left(\mathbf{0}, \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)\mathbf{I}\right)
$$

简化：
$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1} = 1 - \bar{\alpha}_t
$$

因此：
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

```python
import torch

def forward_diffusion_sample(x_0, t, betas):
    """
    前向扩散采样：从x_0直接采样x_t
    
    x_0: 原始数据 [B, C, H, W]
    t: 时间步 [B]
    betas: 噪声调度 [T]
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 获取对应时间步的alpha_bar
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    # 采样噪声
    noise = torch.randn_like(x_0)
    
    # 重参数化采样
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    return x_t, noise
```

### 1.4 连续时间随机微分方程（SDE）

当时间步数 $T \to \infty$ 时，离散扩散过程收敛到连续SDE：

**前向SDE：**
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$

其中：
- $\mathbf{f}(\mathbf{x}, t)$：漂移系数（drift coefficient）
- $g(t)$：扩散系数（diffusion coefficient）
- $\mathbf{w}$：标准维纳过程（布朗运动）

**常用SDE形式：**

#### 1.4.1 Variance Preserving (VP) SDE

对应DDPM的连续极限：
$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}
$$

**特点：** 保持方差不变，$\text{Var}(\mathbf{x}_T) \approx 1$

#### 1.4.2 Variance Exploding (VE) SDE

对应Score Matching模型：
$$
d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,d\mathbf{w}
$$

**特点：** 方差随时间增长，$\text{Var}(\mathbf{x}_T) \to \infty$

#### 1.4.3 子VP SDE (Sub-VP SDE)

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)ds}\right)}\,d\mathbf{w}
$$

```python
import torch

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        """线性噪声调度"""
        return self.beta_min + (self.beta_max - self.beta_min) * t
    
    def alpha_bar(self, t):
        """累积alpha"""
        return torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
    
    def drift(self, x, t):
        """漂移系数 f(x,t) = -0.5 * beta(t) * x"""
        return -0.5 * self.beta(t)[:, None, None, None] * x
    
    def diffusion(self, t):
        """扩散系数 g(t) = sqrt(beta(t))"""
        return torch.sqrt(self.beta(t))
    
    def marginal_prob(self, x_0, t):
        """边际分布的均值和标准差"""
        mean = torch.sqrt(self.alpha_bar(t))[:, None, None, None] * x_0
        std = torch.sqrt(1 - self.alpha_bar(t))
        return mean, std
```

### 1.5 噪声调度的数学设计

噪声调度 $\{\beta_t\}_{t=1}^T$ 的选择对模型性能至关重要。

#### 1.5.1 线性调度（Linear Schedule）

$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})
$$

**DDPM默认参数：** $\beta_1 = 10^{-4}$, $\beta_T = 0.02$

```python
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)
```

**问题：** 在高分辨率图像上，线性调度在早期时间步破坏信息过快。

#### 1.5.2 余弦调度（Cosine Schedule）

Improved DDPM提出的改进调度：
$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)^2
$$

其中 $s$ 是偏移量，通常取 $s = 0.008$。

```python
def cosine_beta_schedule(T, s=0.008):
    """
    余弦噪声调度
    
    T: 总时间步数
    s: 偏移量，防止β_t在t=0时过小
    """
    t = torch.linspace(0, 1, T + 1)
    f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    
    # 从alpha_bar计算beta
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    
    # 裁剪防止数值问题
    betas = torch.clip(betas, 0, 0.999)
    return betas
```

**优势：** 
- 噪声添加更加平滑
- 避免早期过多信息损失
- 在图像生成中效果更好

#### 1.5.3 Sigmoid调度

$$
\beta_t = \beta_{\max} \cdot \sigma\left(\frac{t - T/2}{\tau}\right)
$$

其中 $\sigma$ 是sigmoid函数，$\tau$ 控制过渡的平滑程度。

```python
def sigmoid_beta_schedule(T, beta_start=1e-4, beta_end=0.02, tau=10):
    """
    Sigmoid噪声调度
    
    tau: 控制过渡的平滑程度
    """
    t = torch.linspace(0, 1, T)
    betas = beta_start + (beta_end - beta_start) * torch.sigmoid(tau * (t - 0.5))
    return betas
```

#### 1.5.4 噪声调度对比

| 调度类型 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| 线性 | 简单直观 | 早期信息损失快 | 低分辨率图像 |
| 余弦 | 平滑过渡 | 需要调参 | 高分辨率图像 |
| Sigmoid | 灵活可控 | 可能不稳定 | 特定任务 |

```python
import matplotlib.pyplot as plt

T = 1000
linear_betas = linear_beta_schedule(T)
cosine_betas = cosine_beta_schedule(T)
sigmoid_betas = sigmoid_beta_schedule(T)

# 计算累积alpha
def compute_alpha_bar(betas):
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(linear_betas, label='Linear')
plt.plot(cosine_betas, label='Cosine')
plt.plot(sigmoid_betas, label='Sigmoid')
plt.xlabel('Timestep')
plt.ylabel('β_t')
plt.title('Noise Schedule Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(compute_alpha_bar(linear_betas), label='Linear')
plt.plot(compute_alpha_bar(cosine_betas), label='Cosine')
plt.plot(compute_alpha_bar(sigmoid_betas), label='Sigmoid')
plt.xlabel('Timestep')
plt.ylabel='ᾱ_t')
plt.title('Cumulative Product of α')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 2. 逆向去噪过程

### 2.1 逆向过程的数学形式

**逆向过程** 是我们需要学习的生成过程，从纯噪声逐步恢复数据：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

**完整的逆向链：**
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

其中 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ 是标准高斯先验。

### 2.2 逆向SDE推导

**关键定理：** 给定前向SDE，存在对应的逆向SDE：

$$
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t) d\bar{\mathbf{w}}
$$

其中：
- $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$：分数函数（Score Function）
- $\bar{\mathbf{w}}$：逆向时间的维纳过程
- $dt$：负时间步（逆向）

**推导思路（Anderson, 1982）：**

前向SDE的转移概率密度 $p_{t|s}(\mathbf{x}_t | \mathbf{x}_s)$ 满足Fokker-Planck方程。逆向过程的转移概率可以通过贝叶斯定理得到：

$$
p_{s|t}(\mathbf{x}_s | \mathbf{x}_t) = \frac{p_{t|s}(\mathbf{x}_t | \mathbf{x}_s) p_s(\mathbf{x}_s)}{p_t(\mathbf{x}_t)}
$$

取对数并对 $\mathbf{x}_s$ 求导，可以得到逆向SDE的形式。

### 2.3 Score Matching理论

**分数函数** 定义为对数概率密度的梯度：
$$
\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

**为什么学习分数函数？**

1. **避免归一化常数：** 概率密度 $p(\mathbf{x}) = \frac{e^{-E(\mathbf{x})}}{Z}$ 中 $Z$ 难以计算，但分数函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$ 不依赖 $Z$

2. **直接用于采样：** 朗之万动力学（Langevin Dynamics）可以利用分数函数采样

**显式分数匹配目标：**
$$
\mathcal{L}_{\text{ESM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})}\left[\frac{1}{2}\|\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})\|^2\right]
$$

**问题：** 需要知道真实分数 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$，这在实际中不可用。

### 2.4 去噪分数匹配（Denoising Score Matching, DSM）

**核心思想：** 通过向数据添加噪声，构造一个可计算的目标。

**DSM目标函数：**
$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}}\left[\frac{1}{2}\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) - \frac{-\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]
$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

**关键定理（Vincent, 2011）：**
$$
\mathbb{E}_{q(\tilde{\mathbf{x}}|\mathbf{x})}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right] = \mathbb{E}_{q(\tilde{\mathbf{x}})}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}})\|^2\right] + C
$$

这意味着我们可以用条件分数 $\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x})$ 替代边际分数。

**对于高斯噪声：**
$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2\mathbf{I})
$$

条件分数：
$$
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} = \frac{-\boldsymbol{\epsilon}}{\sigma}
$$

```python
def denoising_score_matching_loss(score_model, x, sigma, noise=None):
    """
    去噪分数匹配损失
    
    score_model: 分数网络 s_θ(x, t)
    x: 干净数据 [B, C, H, W]
    sigma: 噪声水平
    """
    if noise is None:
        noise = torch.randn_like(x)
    
    # 加噪
    x_noisy = x + sigma * noise
    
    # 预测分数
    score_pred = score_model(x_noisy, sigma)
    
    # 目标分数: -epsilon / sigma
    score_target = -noise / sigma
    
    # MSE损失
    loss = 0.5 * torch.mean((score_pred - score_target) ** 2)
    
    return loss
```

### 2.5 Tweedie公式

**Tweedie公式** 是统计估计中的重要结果，在扩散模型中用于估计后验均值。

**单变量形式：**
$$
\mathbb{E}[\mu | x] = x + \sigma^2 \frac{d}{dx}\log p(x)
$$

**多变量形式：**
$$
\mathbb{E}[\boldsymbol{\mu} | \mathbf{x}] = \mathbf{x} + \Sigma \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

**在扩散模型中的应用：**

给定 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$，后验均值：

$$
\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t + (1-\bar{\alpha}_t)\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)\right)
$$

**推导：**

由Tweedie公式，对于 $\mathbf{x}_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$：

$$
\mathbb{E}[\sqrt{\bar{\alpha}_t}\mathbf{x}_0 | \mathbf{x}_t] = \mathbf{x}_t + (1-\bar{\alpha}_t)\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)
$$

因此：
$$
\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \frac{1}{\sqrt{\bar{\alpha}_t}}\mathbf{x}_t + \frac{1-\bar{\alpha}_t}{\sqrt{\bar{\alpha}_t}}\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)
$$

**与噪声预测的关系：**

由于 $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) \propto -\boldsymbol{\epsilon}$：

$$
\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)
$$

```python
def tweedie_estimate(x_t, t, score_model, alphas_cumprod):
    """
    使用Tweedie公式估计x_0
    
    x_t: 噪声数据
    t: 时间步
    score_model: 分数模型
    alphas_cumprod: 累积alpha
    """
    alpha_bar = alphas_cumprod[t]
    
    # 预测分数
    score = score_model(x_t, t)
    
    # Tweedie估计
    x_0_est = (x_t + (1 - alpha_bar) * score) / torch.sqrt(alpha_bar)
    
    return x_0_est
```

### 2.6 后验分布推导

**目标：** 推导 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ 的解析形式。

**应用贝叶斯定理：**
$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}
$$

**计算各项：**

1. $q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

2. $q(\mathbf{x}_{t-1} | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$

3. $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$

**后验分布结果：**
$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

其中：
$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

**用噪声表示的后验均值：**

代入 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$：

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \boldsymbol{\epsilon}) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)
$$

```python
def posterior_mean_variance(x_t, t, x_0, betas):
    """
    计算后验分布的均值和方差
    
    x_t: 当前时刻数据
    t: 时间步
    x_0: 原始数据（或估计值）
    betas: 噪声调度
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    # 后验均值系数
    posterior_mean_coef1 = betas[t] * torch.sqrt(alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
    posterior_mean_coef2 = (1 - alphas_cumprod_prev[t]) * torch.sqrt(alphas[t]) / (1 - alphas_cumprod[t])
    
    # 后验均值
    posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
    
    # 后验方差
    posterior_variance = betas[t] * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
    
    return posterior_mean, posterior_variance
```

---

## 3. DDPM数学推导

### 3.1 变分下界（ELBO）推导

**目标：** 最大化数据的对数似然 $\log p_\theta(\mathbf{x}_0)$。

**变分下界：**
$$
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\right] = -\mathcal{L}
$$

**展开ELBO：**
$$
\mathcal{L} = \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{\mathbb{E}_q\left[D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]}_{L_{t-1}} \underbrace{- \mathbb{E}_q\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]}_{L_0}
$$

**各项含义：**
- $L_T$：先验匹配项，当 $\bar{\alpha}_T \approx 0$ 时接近0
- $L_{t-1}$：去噪匹配项，需要学习
- $L_0$：重构项

### 3.2 KL散度计算

**高斯分布之间的KL散度：**
$$
D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2\mathbf{I}) \| \mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2\mathbf{I})) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

**应用到扩散模型：**

设 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$，则：

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2\right] + C
$$

### 3.3 参数化与简化损失

**关键洞察：** 不直接预测均值，而是预测噪声 $\boldsymbol{\epsilon}$。

**参数化：**
$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

**代入KL散度：**
$$
\begin{aligned}
L_{t-1} &= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right) - \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)\right\|^2\right] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
\end{aligned}
$$

**简化损失（DDPM）：**

忽略权重系数，得到简化的训练目标：
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

**为什么简化有效？**

1. **权重分析：** $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}$ 在不同时间步变化不大
2. **实践效果：** 简化损失在图像生成中效果更好
3. **等价性：** 相当于对原始ELBO的重加权

```python
def ddpm_loss(model, x_0, t, noise=None):
    """
    DDPM简化损失函数
    
    model: 噪声预测网络 ε_θ(x_t, t)
    x_0: 干净数据 [B, C, H, W]
    t: 时间步 [B]
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # 前向扩散
    x_t, _ = forward_diffusion_sample(x_0, t, betas)
    
    # 预测噪声
    noise_pred = model(x_t, t)
    
    # 简化损失
    loss = F.mse_loss(noise_pred, noise)
    
    return loss
```

### 3.4 完整DDPM训练算法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = T
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 前向扩散系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # 后验方差
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        # 后验均值系数
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """前向扩散采样 x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_losses(self, x_0, t, noise=None):
        """训练损失"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t)
        
        return F.mse_loss(noise_pred, noise)
    
    def p_sample(self, x_t, t):
        """单步去噪"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
        
        # 预测噪声
        noise_pred = self.model(x_t, t)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            # 添加噪声
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, shape, device):
        """完整采样过程"""
        b = shape[0]
        
        # 从纯噪声开始
        x_t = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.T)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch)
        
        return x_t
```

### 3.5 DDPM采样过程

**算法流程：**

```
输入：训练好的模型 ε_θ，噪声调度 {β_t}
输出：生成样本 x_0

1. x_T ~ N(0, I)
2. for t = T, T-1, ..., 1 do
3.     z ~ N(0, I) if t > 1, else z = 0
4.     x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + √β_t · z
5. end for
6. return x_0
```

```python
@torch.no_grad()
def p_sample_loop(model, shape, T, betas, alphas, alphas_cumprod):
    """
    DDPM完整采样循环
    
    model: 噪声预测网络
    shape: 生成样本形状 [B, C, H, W]
    T: 总时间步
    """
    device = next(model.parameters()).device
    b = shape[0]
    
    # 从纯噪声开始
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(T)):
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        
        # 预测噪声
        noise_pred = model(x, t_batch)
        
        # 计算去噪后的均值
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        
        if t > 0:
            # 添加噪声
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    
    return x
```

### 3.6 DDIM加速采样

**DDIM（Denoising Diffusion Implicit Models）** 提供了一种非马尔可夫的采样方式，可以大幅减少采样步数。

**DDIM采样公式：**
$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的}\mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
$$

```python
@torch.no_grad()
def ddim_sample(model, x_T, seq, alphas_cumprod, eta=0.0):
    """
    DDIM采样
    
    model: 噪声预测网络
    x_T: 初始噪声
    seq: 采样时间步序列（可以是子集）
    alphas_cumprod: 累积alpha
    eta: 随机性参数，0表示确定性采样
    """
    x = x_T
    
    for i in range(len(seq) - 1):
        t = seq[i]
        t_prev = seq[i + 1]
        
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # 预测噪声
        noise_pred = model(x, torch.tensor([t], device=x.device))
        
        # 预测x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # 计算方向向量
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t) / (1 - alpha_bar_t_prev) * (1 - alpha_bar_t_prev)) * noise_pred
        
        # 随机噪声
        if eta > 0:
            noise = torch.randn_like(x)
            sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
        else:
            noise = 0
            sigma = 0
        
        # 更新
        x = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt + sigma * noise
    
    return x
```

---

## 4. Flow Matching

### 4.1 连续归一化流（CNF）

**连续归一化流** 通过ODE定义概率分布的变换：

$$
\frac{d\mathbf{x}}{dt} = \mathbf{v}_\theta(\mathbf{x}, t)
$$

**概率密度变换：**
$$
\log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \nabla \cdot \mathbf{v}_\theta(\mathbf{x}(t), t) \, dt
$$

**与离散流的对比：**

| 特性 | 离散流 | 连续流（CNF） |
|------|--------|---------------|
| 变换方式 | 多步离散变换 | 单步连续ODE |
| 雅可比行列式 | 需要显式计算 | 通过积分隐式计算 |
| 可逆性 | 需要特殊设计 | 自动可逆 |
| 表达能力 | 受限于步数 | 理论上无限 |

### 4.2 Flow Matching目标函数

**Flow Matching** 是一种训练CNF的高效方法。

**目标函数：**
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|\mathbf{v}_\theta(\mathbf{x}(t), t) - \mathbf{u}_t(\mathbf{x}_0, \mathbf{x}_1)\|^2\right]
$$

其中 $\mathbf{u}_t(\mathbf{x}_0, \mathbf{x}_1)$ 是条件向量场。

**条件Flow Matching：**
$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0 \sim p_0, \mathbf{x}_1 \sim p_1}\left[\|\mathbf{v}_\theta(\mathbf{x}(t), t) - \mathbf{u}_t(\mathbf{x} | \mathbf{x}_0, \mathbf{x}_1)\|^2\right]
$$

**关键定理：** $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$，即条件Flow Matching的梯度与边际Flow Matching相同。

### 4.3 最优传输路径

**最优传输路径** 是Flow Matching中常用的路径设计。

**线性插值路径：**
$$
\mathbf{x}(t) = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
$$

**对应的向量场：**
$$
\mathbf{u}_t(\mathbf{x} | \mathbf{x}_0, \mathbf{x}_1) = \frac{d\mathbf{x}}{dt} = \mathbf{x}_1 - \mathbf{x}_0
$$

**概率路径：**
$$
p_t(\mathbf{x} | \mathbf{x}_0, \mathbf{x}_1) = \mathcal{N}(\mathbf{x}; (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \sigma^2\mathbf{I})
$$

```python
def optimal_transport_path(x_0, x_1, t):
    """
    最优传输路径
    
    x_0: 源分布样本
    x_1: 目标分布样本
    t: 时间 [0, 1]
    """
    # 线性插值
    x_t = (1 - t) * x_0 + t * x_1
    
    # 向量场（速度）
    v_t = x_1 - x_0
    
    return x_t, v_t

def flow_matching_loss(velocity_model, x_0, x_1, t):
    """
    Flow Matching损失
    
    velocity_model: 速度场网络 v_θ(x, t)
    x_0: 噪声样本
    x_1: 数据样本
    t: 时间
    """
    # 计算路径和目标速度
    x_t, v_target = optimal_transport_path(x_0, x_1, t)
    
    # 预测速度
    v_pred = velocity_model(x_t, t)
    
    # MSE损失
    loss = torch.mean((v_pred - v_target) ** 2)
    
    return loss
```

### 4.4 Flow Matching与扩散模型的关系

**扩散模型作为特殊的Flow：**

VP-SDE可以转化为概率流ODE：
$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)\mathbf{x} - \frac{1}{2}\beta(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})
$$

**等价性分析：**

| 方法 | 训练目标 | 采样方式 |
|------|----------|----------|
| DDPM | 噪声预测 $\boldsymbol{\epsilon}_\theta$ | 随机采样 |
| Score Matching | 分数函数 $\mathbf{s}_\theta$ | Langevin/ODE |
| Flow Matching | 速度场 $\mathbf{v}_\theta$ | ODE求解 |

**统一视角：**

$$
\mathbf{v}_\theta(\mathbf{x}, t) = \frac{d\mathbf{x}}{dt} \leftrightarrow \boldsymbol{\epsilon}_\theta(\mathbf{x}, t) = -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}}\log p_t(\mathbf{x})
$$

```python
class FlowMatching:
    def __init__(self, velocity_model, sigma=0.0):
        self.model = velocity_model
        self.sigma = sigma
    
    def compute_loss(self, x_1):
        """
        计算Flow Matching损失
        
        x_1: 数据样本
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # 采样时间
        t = torch.rand(batch_size, device=device)
        
        # 采样源分布（标准高斯）
        x_0 = torch.randn_like(x_1)
        
        # 条件路径
        x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
        
        # 添加噪声（可选）
        if self.sigma > 0:
            noise = torch.randn_like(x_t) * self.sigma
            x_t = x_t + noise
        
        # 目标速度
        v_target = x_1 - x_0
        
        # 预测速度
        v_pred = self.model(x_t, t)
        
        return F.mse_loss(v_pred, v_target)
    
    @torch.no_grad()
    def sample(self, shape, device, steps=100):
        """
        ODE采样
        """
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = i / steps
            t_tensor = torch.full((shape[0],), t, device=device)
            
            v = self.model(x, t_tensor)
            x = x + v * dt
        
        return x
```

### 4.5 Rectified Flow

**Rectified Flow** 是Flow Matching的一个重要变体，旨在学习"直线路径"。

**核心思想：** 通过"回流"（reflow）操作，使路径越来越直。

**目标：**
$$
\min_\theta \mathbb{E}\left[\|\mathbf{x}_1 - \mathbf{x}_0 - \mathbf{v}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

其中 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$。

**Reflow过程：**
1. 用当前模型生成样本对 $(\mathbf{x}_0, \mathbf{x}_1')$
2. 在新的样本对上重新训练
3. 重复直到路径足够直

```python
class RectifiedFlow:
    def __init__(self, model, steps=100):
        self.model = model
        self.steps = steps
    
    def train_step(self, x_1, optimizer):
        """训练一步"""
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        
        # 线性插值
        x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
        
        # 目标：直接指向x_1
        v_target = x_1 - x_0
        
        # 预测
        v_pred = self.model(x_t, t)
        
        loss = F.mse_loss(v_pred, v_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, x_0, steps=None):
        """Euler方法采样"""
        if steps is None:
            steps = self.steps
        
        x = x_0
        dt = 1.0 / steps
        
        for i in range(steps):
            t = i / steps
            t_tensor = torch.full((x.shape[0],), t, device=x.device)
            v = self.model(x, t_tensor)
            x = x + v * dt
        
        return x
```

---

## 5. 条件生成数学

### 5.1 条件扩散模型

**条件生成** 的目标是学习条件分布 $p(\mathbf{x} | \mathbf{y})$，其中 $\mathbf{y}$ 是条件信息（如类别标签、文本描述等）。

**条件逆向过程：**
$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{y}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t, \mathbf{y}), \sigma_t^2 \mathbf{I})
$$

**条件训练目标：**
$$
\mathcal{L}_{\text{conditional}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}, \mathbf{y}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{y})\|^2\right]
$$

### 5.2 Classifier Guidance

**核心思想：** 使用一个独立的分类器来引导生成过程。

**数学推导：**

由贝叶斯定理：
$$
\log p(\mathbf{x}_t | \mathbf{y}) = \log p(\mathbf{y} | \mathbf{x}_t) + \log p(\mathbf{x}_t) - \log p(\mathbf{y})
$$

对 $\mathbf{x}_t$ 求梯度：
$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{y} | \mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
$$

**条件分数函数：**
$$
\mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{y}) = \mathbf{s}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{y} | \mathbf{x}_t)
$$

**引导采样：**
$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{y}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{y} | \mathbf{x}_t)
$$

**引入引导强度 $s$：**
$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{y}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{y} | \mathbf{x}_t)
$$

```python
def classifier_guidance_sample(model, classifier, x_t, t, y, guidance_scale=1.0):
    """
    Classifier Guidance采样
    
    model: 无条件扩散模型
    classifier: 噪声图像分类器
    x_t: 当前噪声图像
    t: 时间步
    y: 目标类别
    guidance_scale: 引导强度
    """
    x_t.requires_grad_(True)
    
    # 分类器预测
    logits = classifier(x_t, t)
    log_prob = F.log_softmax(logits, dim=-1)[:, y]
    
    # 计算梯度
    grad = torch.autograd.grad(log_prob.sum(), x_t)[0]
    
    # 无条件噪声预测
    noise_pred = model(x_t.detach(), t)
    
    # 引导后的噪声预测
    alpha_bar_t = alphas_cumprod[t]
    noise_guided = noise_pred - guidance_scale * torch.sqrt(1 - alpha_bar_t) * grad
    
    return noise_guided
```

### 5.3 Classifier-Free Guidance

**核心思想：** 同时训练条件和无条件模型，避免额外的分类器。

**条件和无条件模型结合：**
$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{y}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{y}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing))
$$

其中 $\varnothing$ 表示空条件（无条件）。

**数学推导：**

从条件分数的角度：
$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log \frac{p(\mathbf{x}_t | \mathbf{y})}{p(\mathbf{x}_t)}
$$

引导强度 $s$ 放大条件信息：
$$
\nabla_{\mathbf{x}_t} \log \tilde{p}(\mathbf{x}_t | \mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + s \cdot \nabla_{\mathbf{x}_t} \log \frac{p(\mathbf{x}_t | \mathbf{y})}{p(\mathbf{x}_t)}
$$

**训练策略：**
- 以概率 $p_{\text{uncond}}$（通常10%）随机丢弃条件 $\mathbf{y}$
- 用同一个模型学习条件和无条件生成

```python
class ClassifierFreeGuidanceModel(nn.Module):
    def __init__(self, model, p_uncond=0.1):
        super().__init__()
        self.model = model
        self.p_uncond = p_uncond
    
    def forward(self, x_t, t, y=None, guidance_scale=1.0):
        """
        前向传播，支持Classifier-Free Guidance
        
        x_t: 噪声图像
        t: 时间步
        y: 条件（如文本embedding）
        guidance_scale: 引导强度
        """
        if self.training:
            # 训练时随机丢弃条件
            if torch.rand(1).item() < self.p_uncond:
                y = None
            return self.model(x_t, t, y)
        else:
            # 推理时应用引导
            if guidance_scale == 1.0 or y is None:
                return self.model(x_t, t, y)
            
            # 条件预测
            noise_cond = self.model(x_t, t, y)
            
            # 无条件预测
            noise_uncond = self.model(x_t, t, None)
            
            # 引导
            noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            return noise_guided

def classifier_free_guidance_sample(model, x_t, t, y, guidance_scale=7.5):
    """
    Classifier-Free Guidance采样
    
    model: 支持条件和无条件的模型
    x_t: 当前噪声图像
    t: 时间步
    y: 条件
    guidance_scale: 引导强度（通常5-15）
    """
    # 条件预测
    noise_cond = model(x_t, t, y)
    
    # 无条件预测
    noise_uncond = model(x_t, t, None)
    
    # CFG组合
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    return noise_guided
```

### 5.4 条件分数函数的完整推导

**设定：**
- 数据分布：$p_0(\mathbf{x})$
- 条件分布：$p(\mathbf{y}|\mathbf{x})$
- 目标：学习 $p_0(\mathbf{x}|\mathbf{y})$

**扩散过程中的条件分布：**

由贝叶斯定理：
$$
p_t(\mathbf{x}_t | \mathbf{y}) = \int p_t(\mathbf{x}_t | \mathbf{x}_0) p_0(\mathbf{x}_0 | \mathbf{y}) d\mathbf{x}_0
$$

**条件分数函数：**
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{y}) = \frac{\int \nabla_{\mathbf{x}_t} p_t(\mathbf{x}_t | \mathbf{x}_0) p_0(\mathbf{x}_0 | \mathbf{y}) d\mathbf{x}_0}{\int p_t(\mathbf{x}_t | \mathbf{x}_0) p_0(\mathbf{x}_0 | \mathbf{y}) d\mathbf{x}_0}
$$

**对于高斯扩散：**
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t}
$$

**Classifier-Free Guidance的分数形式：**
$$
\mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{y}) = \mathbf{s}_\theta(\mathbf{x}_t, t, \varnothing) + s \cdot (\mathbf{s}_\theta(\mathbf{x}_t, t, \mathbf{y}) - \mathbf{s}_\theta(\mathbf{x}_t, t, \varnothing))
$$

### 5.5 引导强度的影响分析

**引导强度 $s$ 的作用：**

| $s$ 值 | 效果 | 生成质量 | 多样性 |
|--------|------|----------|--------|
| $s = 1$ | 标准条件生成 | 平衡 | 高 |
| $s > 1$ | 增强条件影响 | 更符合条件 | 降低 |
| $s \gg 1$ | 过度引导 | 可能失真 | 很低 |

**最优引导强度的选择：**

```python
def find_optimal_guidance_scale(model, val_loader, scales=[1.0, 3.0, 5.0, 7.5, 10.0, 15.0]):
    """
    寻找最优引导强度
    """
    results = {}
    
    for scale in scales:
        total_loss = 0
        total_samples = 0
        
        for batch in val_loader:
            x_0, y = batch
            
            # 生成样本
            x_gen = sample_with_guidance(model, y, guidance_scale=scale)
            
            # 计算FID或其他指标
            loss = compute_fid(x_gen, x_0)
            total_loss += loss
            total_samples += 1
        
        results[scale] = total_loss / total_samples
        print(f"Scale {scale}: FID = {results[scale]:.2f}")
    
    best_scale = min(results, key=results.get)
    return best_scale, results
```

### 5.6 完整的条件生成实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiffusion(nn.Module):
    def __init__(self, unet, text_encoder, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.T = T
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # CFG参数
        self.p_uncond = 0.1
    
    def forward(self, x_0, text, t=None):
        """
        训练前向传播
        
        x_0: 干净图像 [B, C, H, W]
        text: 文本描述 [B, seq_len]
        t: 时间步（可选）
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # 采样时间步
        if t is None:
            t = torch.randint(0, self.T, (B,), device=device)
        
        # 编码文本
        text_emb = self.text_encoder(text)
        
        # 随机丢弃条件
        if self.training:
            mask = torch.rand(B, device=device) < self.p_uncond
            text_emb[mask] = 0
        
        # 采样噪声
        noise = torch.randn_like(x_0)
        
        # 前向扩散
        alpha_bar = self.alphas_cumprod[t][:, None, None, None]
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        # 预测噪声
        noise_pred = self.unet(x_t, t, text_emb)
        
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, text, shape, guidance_scale=7.5, device='cuda'):
        """
        条件采样
        
        text: 文本描述
        shape: 生成形状 [B, C, H, W]
        guidance_scale: CFG引导强度
        """
        B = shape[0]
        
        # 编码文本
        text_emb = self.text_encoder(text)
        
        # 从纯噪声开始
        x_t = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # 条件预测
            noise_cond = self.unet(x_t, t_batch, text_emb)
            
            # 无条件预测
            noise_uncond = self.unet(x_t, t_batch, torch.zeros_like(text_emb))
            
            # CFG
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # 去噪
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
        
        return x_t
```

---

## 本章小结

扩散模型是现代生成式AI的核心技术，其数学基础涵盖多个重要领域：

1. **前向扩散过程** 通过马尔可夫链和SDE描述噪声添加过程，噪声调度设计影响生成质量

2. **逆向去噪过程** 基于Score Matching理论和Tweedie公式，学习从噪声恢复数据

3. **DDPM数学推导** 基于变分下界，简化损失函数使训练高效稳定

4. **Flow Matching** 提供了统一ODE视角，最优传输路径实现高效采样

5. **条件生成数学** Classifier Guidance和Classifier-Free Guidance是条件生成的两大范式

**关键公式总结：**

| 概念 | 核心公式 |
|------|----------|
| 前向扩散 | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ |
| 后验均值 | $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon})$ |
| DDPM损失 | $\mathcal{L} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2]$ |
| 分数函数 | $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ |
| 逆向SDE | $d\mathbf{x} = [\mathbf{f} - g^2\nabla_{\mathbf{x}}\log p_t]dt + g\,d\bar{\mathbf{w}}$ |
| CFG | $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\varnothing} + s(\boldsymbol{\epsilon}_\mathbf{y} - \boldsymbol{\epsilon}_{\varnothing})$ |

**下一章：** 我们将学习**强化学习数学基础**，包括马尔可夫决策过程、策略梯度方法和PPO算法的数学推导。

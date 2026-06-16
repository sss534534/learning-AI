# 第二十五章：金融与优化

> 量子计算在金融领域的应用包括投资组合优化、风险管理、期权定价、欺诈检测等问题。量子算法在特定金融计算任务上可以提供多项式到指数级的加速。

---

## 目录

1. [金融量子计算概述](#1-金融量子计算概述)
2. [投资组合优化](#2-投资组合优化)
3. [风险分析](#3-风险分析)
4. [期权定价](#4-期权定价)
5. [欺诈检测与信用评估](#5-欺诈检测与信用评估)
6. [量子优化理论基础](#6-量子优化理论基础)
7. [本章小结](#7-本章小结)

---

## 1. 金融量子计算概述

### 1.1 金融计算的挑战

金融行业面临大量计算密集型问题：

| 问题 | 传统方法 | 计算瓶颈 |
|------|---------|---------|
| 投资组合优化 | 二次规划 | $O(N^3)$ 矩阵运算 |
| 蒙特卡洛模拟 | 随机采样 | $O(1/\epsilon^2)$ 收敛 |
| 风险价值(VaR) | 分位数估计 | 大量场景计算 |
| 期权定价 | PDE求解 | 高维时指数爆炸 |

### 1.2 量子优势来源

- **振幅估计**：对蒙特卡洛模拟实现二次加速
- **量子退火**：高效求解组合优化问题
- **量子线性代数**：HHL算法实现矩阵运算指数加速
- **量子核方法**：高维特征空间分类

---

## 2. 投资组合优化

### 2.1 Markowitz均值-方差模型

经典的投资组合选择模型：

$$\max_w \left(\mu^T w - \gamma w^T \Sigma w\right)$$

约束：$\sum_i w_i = 1$，$w_i \geq 0$

其中 $w$ 是资产权重，$\mu$ 是预期收益，$\Sigma$ 是协方差矩阵，$\gamma$ 是风险厌恶系数。

### 2.2 离散化与量子表述

实际投资中，头寸通常是离散的（例如：买入整手股票）。这转化为二次无约束二元优化 (QUBO)：

$$E(x) = \sum_i a_i x_i + \sum_{i,j} b_{ij} x_i x_j$$

其中 $x_i \in \{0,1\}$。

### 2.3 QAOA求解投资组合优化

```python
# QAOA 投资组合优化
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator

# 构建投资组合优化问题
mod = QuadraticProgram("portfolio")
n_assets = 4

# 决策变量：是否持有资产 i
for i in range(n_assets):
    mod.binary_var(f"x{i}")

# 目标：最大化收益 - 风险惩罚
expected_returns = [0.15, 0.12, 0.10, 0.08]
cov_matrix = [
    [0.1, 0.02, 0.01, 0.005],
    [0.02, 0.12, 0.03, 0.01],
    [0.01, 0.03, 0.15, 0.02],
    [0.005, 0.01, 0.02, 0.08]
]

# 构建目标函数的线性部分
linear = {f"x{i}": expected_returns[i] for i in range(n_assets)}

# 构建二次部分（风险惩罚）
gamma = 0.5
quadratic = {}
for i in range(n_assets):
    for j in range(n_assets):
        quadratic[(f"x{i}", f"x{j}")] = -gamma * cov_matrix[i][j]

mod.maximize(linear=linear, quadratic=quadratic)

# 约束：至少持有一半资产
min_holdings = 2
mod.linear_constraint(
    {f"x{i}": 1 for i in range(n_assets)},
    ">=", min_holdings, "min_assets"
)

# QAOA 求解
qaoa = MinimumEigenOptimizer(
    QAOA(optimizer=COBYLA(), 
         quantum_instance=AerSimulator())
)
result = qaoa.solve(mod)
print(f"最优组合: {result.x}")
print(f"最优值: {result.fval:.4f}")
```

### 2.4 量子退火方法

D-Wave 等量子退火器可以直接求解 QUBO 问题：

```python
# 使用 D-Wave Ocean SDK 的量子退火示例
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# 定义 QUBO
Q = {(0, 0): -2, (1, 1): -2, (2, 2): -2,
     (0, 1): 1, (0, 2): 1, (1, 2): 1}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# 在 D-Wave 量子退火器上运行
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=1000)
print(sampleset.first)
```

### 2.5 经典 vs 量子对比

| 方法 | 适用规模 | 解质量 | 速度 |
|------|---------|-------|------|
| 经典二次规划 | 任意 | 全局最优 | 中等 |
| QAOA | NISQ规模 | 近似最优 | 可能快 |
| 量子退火 | 数千变量 | 近似最优 | 快 |
| 经典启发式 | 超大规模 | 近似 | 最快 |

---

## 3. 风险分析

### 3.1 风险价值 (VaR)

VaR 是在给定置信水平下的最大可能损失：

$$\text{VaR}_\alpha(X) = \inf\{t \in \mathbb{R} : P(X > t) \leq 1 - \alpha\}$$

### 3.2 量子振幅估计加速

**经典蒙特卡洛**：需要 $O(1/\epsilon^2)$ 次采样达到精度 $\epsilon$。

**量子振幅估计 (QAE)**：
- 使用 Grover-like 迭代
- 达到 $O(1/\epsilon)$ 的收敛速度
- 实现二次加速

```python
# 量子振幅估计计算 VaR（概念代码）
from qiskit_algorithms import AmplitudeEstimation
from qiskit_finance.applications import EuropeanCallDeltaObjective

# 构建量子电路表示损失分布
n_qubits = 5  # 用 5 个量子比特表示 32 个场景
state_preparation = build_loss_distribution_circuit(portfolio_data)

# 目标：估计损失超过阈值的概率
objective = EuropeanCallDeltaObjective(
    target_loss=0.05,  # 5% 损失阈值
    bounds=[0, 0.2]    # 损失范围 0-20%
)

# 振幅估计
ae = AmplitudeEstimation(
    num_eval_qubits=3,  # 精度控制
    quantum_instance=AerSimulator()
)
result = ae.estimate(state_preparation, objective)
print(f"VaR(95%): {result.estimation:.4f}")
```

### 3.3 VaR 方法对比

| 方法 | 复杂度 | 精度 | 适用场景 |
|------|-------|------|---------|
| 历史模拟 | $O(N)$ | 受限于历史 | 常见 |
| 方差-协方差 | $O(N^3)$ | 正态假设 | 简单 |
| 经典蒙特卡洛 | $O(1/\epsilon^2)$ | 可调 | 广泛 |
| 量子蒙特卡洛 | $O(1/\epsilon)$ | 可调 | 大规模场景 |

### 3.4 信用风险评估

量子算法在信用风险评估中的应用：

| 应用 | 经典算法 | 量子算法 | 优势 |
|------|---------|---------|------|
| 违约概率 | 逻辑回归 | 量子SVM | 高维特征 |
| 损失分布 | 蒙特卡洛 | 量子振幅估计 | 二次加速 |
| 压力测试 | 场景模拟 | 量子并行采样 | 更多场景 |

---

## 4. 期权定价

### 4.1 经典期权定价

Black-Scholes 公式：
$$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$

其中 $d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$，$d_2 = d_1 - \sigma\sqrt{T}$。

### 4.2 量子蒙特卡洛

路径相关的奇异期权需要蒙特卡洛模拟：

$$C = e^{-rT}\mathbb{E}[f(S_T)] \approx e^{-rT}\frac{1}{M}\sum_{i=1}^M f(S_i(T))$$

### 4.3 量子振幅估计定价

```python
# 量子蒙特卡洛期权定价（概念实现）
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications import EuropeanCallPricing
from qiskit_algorithms import AmplitudeEstimation

# 构建标的资产价格的概率分布
n_qubits = 6  # 精度
dist = LogNormalDistribution(
    num_qubits=n_qubits,
    mu=0.1,      # 预期收益率
    sigma=0.2,   # 波动率
    bounds=(0, 200)  # 价格范围 0-200
)

# 欧式看涨期权定价
european_call = EuropeanCallPricing(
    num_state_qubits=n_qubits,
    strike_price=105,
    rescaling_factor=0.25,
    bounds=(0, 200)
)

# 振幅估计
state_preparation = european_call.state_preparation
objective = european_call.value

ae = AmplitudeEstimation(
    num_eval_qubits=3,
    quantum_instance=AerSimulator()
)
result = ae.estimate(state_preparation, objective)
estimated_price = european_call.interpret(result)
print(f"量子估计期权价格: {estimated_price:.4f}")
```

### 4.4 加速对比

| 期权类型 | 经典MC时间 | 量子AE时间 | 加速比 |
|---------|-----------|-----------|-------|
| 欧式期权 | $10^6$ 路径 | $1000$ 次迭代 | $1000\times$ |
| 亚式期权 | $10^7$ 路径 | $3162$ 次迭代 | $3162\times$ |
| 篮子期权 | $10^8$ 路径 | $10000$ 次迭代 | $10000\times$ |

---

## 5. 欺诈检测与信用评估

### 5.1 量子SVM

量子核方法在高维特征空间中计算核函数：

$$K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$

量子计算机天然适合计算内积，可以在高维空间中实现高效的分类。

### 5.2 量子生成模型

使用量子波尔兹曼机 (QBM) 生成合成交易数据：
- 解决欺诈检测中的类别不平衡
- 生成符合真实分布的样本
- 隐私保护（差分隐私训练）

### 5.3 应用总结

| 应用场景 | 量子方法 | 预期收益 |
|---------|---------|---------|
| 实时欺诈检测 | 量子核SVM | 更快分类 |
| 信用评分 | 量子神经网络 | 更准确 |
| 反洗钱 | 量子聚类 | 新模式发现 |
| 客户分群 | 量子k-means | 更多特征 |

---

## 6. 量子优化理论基础

### 6.1 QAOA 原理

量子近似优化算法 (QAOA) 的核心思想是交替应用问题哈密顿量和混合哈密顿量：

$$|\psi(\gamma, \beta)\rangle = e^{-i\beta_p B} e^{-i\gamma_p C} \cdots e^{-i\beta_1 B} e^{-i\gamma_1 C} |s\rangle$$

### 6.2 组合优化与QUBO

任何组合优化问题都可以转化为 QUBO 形式：

$$f(x) = \sum_i a_i x_i + \sum_{i<j} b_{ij} x_i x_j$$

### 6.3 约束处理方法

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| 惩罚项 | 约束违反加入目标函数 | 硬约束 |
| 拉格朗日乘子 | 约束作为额外项 | 可调约束 |
| 变量编码 | 特殊编码满足约束 | 特定约束 |

---

## 7. 本章小结

- 量子计算在金融领域的主要优势在于蒙特卡洛模拟的二次加速和组合优化的指数级加速
- 投资组合优化可通过 QUBO 表述用 QAOA 或量子退火求解
- 量子振幅估计是金融量子计算的核心工具，能显著加速风险分析和期权定价
- 欺诈检测和信用评估可从量子核方法和生成模型中受益
- 当前金融量子应用受限于 NISQ 硬件，但在中期（5-10年）有望实现实际优势

---

## 延伸阅读

- *Quantum Computing for Finance* (Herman et al., 2022) — 综述 ⭐⭐⭐⭐⭐
- *Qiskit Finance* 官方文档 — 实践参考 ⭐⭐⭐⭐
- *Quantum Amplitude Amplification and Estimation* (Brassard et al., 2002) — 理论基础 ⭐⭐⭐⭐⭐
- *Quantum Risk Analysis* (Woerner & Egger, 2019) — VaR量子方法 ⭐⭐⭐⭐
- arXiv: 量子金融综述 ⭐⭐⭐⭐

---

*最后更新：2026-06-15*

# 第十二章：图神经网络与知识图谱

> 图神经网络（Graph Neural Network, GNN）是处理图结构数据的深度学习范式，其核心思想是通过消息传递机制在图的拓扑结构上聚合邻居信息来学习节点表示。知识图谱作为图结构数据的重要实例，其表示学习与推理技术与大模型的结合正在催生新一代知识增强AI系统。本章将系统讲解图论基础、图神经网络的数学原理、知识图谱表示学习与推理，以及图神经网络在大模型中的前沿应用。

## 目录

1. [图论基础](#1-图论基础)
2. [图神经网络基础](#2-图神经网络基础)
3. [知识图谱表示学习](#3-知识图谱表示学习)
4. [知识图谱推理](#4-知识图谱推理)
5. [图神经网络在大模型中的应用](#5-图神经网络在大模型中的应用)

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [ch01-linear-algebra.md], [ch02-matrix-decomposition.md]
- **关联文件**: [ch10-llm-training.md], [ch16-multimodal-alignment.md]
- **最后更新**: 2026-06-12

---

## 1. 图论基础

### 1.1 图的定义与基本概念

**定义 12.1（图）** 一个图 $G = (V, E)$ 由顶点集 $V = \{v_1, v_2, \ldots, v_n\}$ 和边集 $E \subseteq V \times V$ 组成。对于有向图，边 $e = (v_i, v_j)$ 表示从 $v_i$ 指向 $v_j$ 的有向边；对于无向图，边无方向性。

**图的属性：**
- 节点数 $n = |V|$
- 边数 $m = |E|$
- 节点 $v_i$ 的邻居集 $\mathcal{N}(v_i) = \{v_j : (v_i, v_j) \in E\}$
- 节点 $v_i$ 的度 $d_i = |\mathcal{N}(v_i)|$

**属性图（Attributed Graph）：** 实际应用中，图通常带有丰富的属性信息：

$$
G = (V, E, \mathbf{X}, \mathbf{E})
$$

其中 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 为节点特征矩阵，$\mathbf{E} \in \mathbb{R}^{m \times d_e}$ 为边特征矩阵。

### 1.2 图的矩阵表示

**邻接矩阵（Adjacency Matrix）**

**定义 12.2（邻接矩阵）** 给定图 $G$，其邻接矩阵 $\mathbf{A} \in \{0,1\}^{n \times n}$ 定义为：

$$
A_{ij} = \begin{cases} 1, & \text{若 } (v_i, v_j) \in E \\ 0, & \text{否则} \end{cases}
$$

对于带权图，$A_{ij} = w_{ij}$ 表示边 $(v_i, v_j)$ 的权重。

**邻接矩阵的性质：**
- 无向图的邻接矩阵是对称的：$\mathbf{A} = \mathbf{A}^\top$
- 邻接矩阵的 $k$ 次幂 $\mathbf{A}^k$ 的元素 $(\mathbf{A}^k)_{ij}$ 表示从 $v_i$ 到 $v_j$ 长度为 $k$ 的路径数

**度矩阵（Degree Matrix）**

**定义 12.3（度矩阵）** 度矩阵 $\mathbf{D} \in \mathbb{R}^{n \times n}$ 是对角矩阵：

$$
D_{ii} = d_i = \sum_{j=1}^{n} A_{ij}
$$

**拉普拉斯矩阵（Laplacian Matrix）**

**定义 12.4（组合拉普拉斯矩阵）** 组合拉普拉斯矩阵（Combinatorial Laplacian）定义为：

$$
\mathbf{L} = \mathbf{D} - \mathbf{A}
$$

**定义 12.5（归一化拉普拉斯矩阵）** 对称归一化拉普拉斯矩阵定义为：

$$
\mathbf{L}_{\text{sym}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
$$

随机游走归一化拉普拉斯矩阵定义为：

$$
\mathbf{L}_{\text{rw}} = \mathbf{D}^{-1} \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}
$$

**拉普拉斯矩阵的关键性质：**

**定理 12.1** 对于无向图 $G$，拉普拉斯矩阵 $\mathbf{L}$ 具有以下性质：

1. **半正定性：** 对任意向量 $\mathbf{x} \in \mathbb{R}^n$，

$$
\mathbf{x}^\top \mathbf{L} \mathbf{x} = \frac{1}{2} \sum_{(v_i, v_j) \in E} (x_i - x_j)^2 \geq 0
$$

2. **特征值非负：** $\mathbf{L}$ 的所有特征值 $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$

3. **零特征值重数：** $\lambda_1 = 0$ 的重数等于图的连通分量数

**证明（性质1）：**

$$
\begin{aligned}
\mathbf{x}^\top \mathbf{L} \mathbf{x} &= \mathbf{x}^\top (\mathbf{D} - \mathbf{A}) \mathbf{x} \\
&= \sum_{i=1}^{n} d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j \\
&= \sum_{i=1}^{n} \sum_{j \in \mathcal{N}(i)} x_i^2 - \sum_{(v_i,v_j) \in E} 2 x_i x_j \\
&= \sum_{(v_i,v_j) \in E} (x_i^2 + x_j^2 - 2x_i x_j) \\
&= \frac{1}{2} \sum_{(v_i,v_j) \in E} (x_i - x_j)^2
\end{aligned}
$$

$\blacksquare$

### 1.3 谱图理论

**特征分解：** 拉普拉斯矩阵可以进行特征分解：

$$
\mathbf{L} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top
$$

其中 $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n] \in \mathbb{R}^{n \times n}$ 是正交特征向量矩阵，$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 是特征值对角矩阵。

**图傅里叶变换（Graph Fourier Transform, GFT）**

类比经典信号处理中的傅里叶变换，拉普拉斯矩阵的特征向量构成了图上的"频率基"。

**定义 12.6（图傅里叶变换）** 给定图信号 $\mathbf{x} \in \mathbb{R}^n$：

- **图傅里叶正变换：**

$$
\hat{\mathbf{x}} = \mathbf{U}^\top \mathbf{x}
$$

其中 $\hat{x}_k = \mathbf{u}_k^\top \mathbf{x}$ 是信号 $\mathbf{x}$ 在第 $k$ 个频率分量上的系数。

- **图傅里叶逆变换：**

$$
\mathbf{x} = \mathbf{U} \hat{\mathbf{x}} = \sum_{k=1}^{n} \hat{x}_k \mathbf{u}_k
$$

**频率的物理意义：** 特征值 $\lambda_k$ 对应图上的"频率"，$\lambda_k$ 越小表示该分量越平滑（低频），$\lambda_k$ 越大表示变化越剧烈（高频）。

**图信号的平滑度：** 图信号 $\mathbf{x}$ 的平滑度可以用拉普拉斯二次型度量：

$$
S(\mathbf{x}) = \mathbf{x}^\top \mathbf{L} \mathbf{x} = \sum_{k=1}^{n} \lambda_k \hat{x}_k^2
$$

低频分量（小 $\lambda_k$）对平滑度的贡献小，高频分量（大 $\lambda_k$）对平滑度的贡献大。

**谱卷积（Spectral Convolution）**

**定义 12.7（谱卷积）** 图上两个信号 $\mathbf{x}$ 和 $\mathbf{y}$ 的谱卷积定义为：

$$
\mathbf{x} *_{\mathcal{G}} \mathbf{y} = \mathbf{U} \left( (\mathbf{U}^\top \mathbf{x}) \odot (\mathbf{U}^\top \mathbf{y}) \right)
$$

其中 $\odot$ 表示逐元素乘法（Hadamard积）。

等价地，定义谱滤波器 $g_\theta = \text{diag}(\theta_1, \theta_2, \ldots, \theta_n)$，滤波操作为：

$$
\mathbf{x} *_{\mathcal{G}} g_\theta = \mathbf{U} g_\theta \mathbf{U}^\top \mathbf{x} = \mathbf{U} \text{diag}(\theta_1, \ldots, \theta_n) \mathbf{U}^\top \mathbf{x}
$$

**谱卷积的问题：** 直接计算需要对拉普拉斯矩阵做特征分解，复杂度为 $O(n^3)$；且滤波器参数 $\theta$ 与图规模相关，不可迁移。

**Chebyshev多项式近似：** Hammond等人提出用Chebyshev多项式近似谱滤波器：

$$
g_\theta *_{\mathcal{G}} \mathbf{x} \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\mathbf{L}}) \mathbf{x}
$$

其中 $\tilde{\mathbf{L}} = \frac{2}{\lambda_{\max}} \mathbf{L} - \mathbf{I}$ 是缩放后的拉普拉斯矩阵，$T_k$ 是第 $k$ 阶Chebyshev多项式，满足递推关系：

$$
\begin{cases}
T_0(\tilde{\mathbf{L}}) = \mathbf{I} \\
T_1(\tilde{\mathbf{L}}) = \tilde{\mathbf{L}} \\
T_k(\tilde{\mathbf{L}}) = 2\tilde{\mathbf{L}} T_{k-1}(\tilde{\mathbf{L}}) - T_{k-2}(\tilde{\mathbf{L}})
\end{cases}
$$

**复杂度分析：** Chebyshev近似将复杂度从 $O(n^3)$ 降低到 $O(K|\mathcal{E}|)$，其中 $K$ 是多项式阶数，$|\mathcal{E}|$ 是边数，且滤波器参数 $\{\theta_k\}_{k=0}^{K}$ 与图规模无关，可迁移到不同图上。

### 1.4 代码示例：图的矩阵表示与谱分析

```python
import numpy as np
from scipy.linalg import eigh

class GraphSpectral:
    def __init__(self, adj_matrix):
        self.A = np.array(adj_matrix, dtype=np.float64)
        self.n = self.A.shape[0]
        self.D = np.diag(self.A.sum(axis=1))
        self.L = self.D - self.A
        self.D_inv_sqrt = np.diag(1.0 / np.sqrt(self.A.sum(axis=1) + 1e-10))
        self.L_sym = self.D_inv_sqrt @ self.L @ self.D_inv_sqrt

    def eigen_decomposition(self):
        eigenvalues, eigenvectors = eigh(self.L_sym)
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def graph_fourier_transform(self, signal):
        _, U = self.eigen_decomposition()
        return U.T @ signal

    def inverse_gft(self, spectrum):
        _, U = self.eigen_decomposition()
        return U @ spectrum

    def spectral_filter(self, signal, theta):
        eigenvalues, U = self.eigen_decomposition()
        g = np.diag(theta)
        return U @ g @ U.T @ signal

    def chebyshev_filter(self, signal, theta_coeffs, lambda_max=2.0):
        L_tilde = (2.0 / lambda_max) * self.L_sym - np.eye(self.n)
        K = len(theta_coeffs)
        T_prev2 = np.eye(self.n)
        T_prev1 = L_tilde
        result = theta_coeffs[0] * T_prev2
        if K > 1:
            result += theta_coeffs[1] * T_prev1
        for k in range(2, K):
            T_k = 2 * L_tilde @ T_prev1 - T_prev2
            result += theta_coeffs[k] * T_k
            T_prev2 = T_prev1
            T_prev1 = T_k
        return result @ signal

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])
gs = GraphSpectral(A)
eigenvalues, _ = gs.eigen_decomposition()
print(f"拉普拉斯特征值: {eigenvalues}")
signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
spectrum = gs.graph_fourier_transform(signal)
print(f"图傅里叶变换: {spectrum}")
```

---

## 2. 图神经网络基础

### 2.1 消息传递框架（Message Passing Framework）

消息传递神经网络（MPNN）由Gilmer等人于2017年提出，统一了多种GNN变体的框架。

**定义 12.8（消息传递框架）** 给定图 $G = (V, E, \mathbf{X})$，消息传递的第 $k$ 层更新规则为：

$$
\mathbf{h}_v^{(k)} = \phi^{(k)} \left( \mathbf{h}_v^{(k-1)}, \bigoplus_{u \in \mathcal{N}(v)} \psi^{(k)} \left( \mathbf{h}_v^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{e}_{vu} \right) \right)
$$

其中：
- $\mathbf{h}_v^{(k)} \in \mathbb{R}^{d_k}$ 是节点 $v$ 在第 $k$ 层的隐藏表示
- $\psi^{(k)}$ 是消息函数（Message Function），定义从邻居 $u$ 传递到节点 $v$ 的消息
- $\bigoplus$ 是聚合函数（Aggregation Function），如求和、均值、最大值等
- $\phi^{(k)}$ 是更新函数（Update Function），结合旧状态和聚合消息生成新状态
- $\mathbf{e}_{vu}$ 是边 $(v, u)$ 的特征

**消息传递的三个阶段：**

1. **消息生成（Message）：** 对每条边 $(v, u)$ 计算消息

$$
\mathbf{m}_{vu}^{(k)} = \psi^{(k)} \left( \mathbf{h}_v^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{e}_{vu} \right)
$$

2. **消息聚合（Aggregate）：** 对每个节点聚合来自所有邻居的消息

$$
\mathbf{m}_v^{(k)} = \bigoplus_{u \in \mathcal{N}(v)} \mathbf{m}_{vu}^{(k)}
$$

3. **状态更新（Update）：** 结合自身状态和聚合消息更新节点表示

$$
\mathbf{h}_v^{(k)} = \phi^{(k)} \left( \mathbf{h}_v^{(k-1)}, \mathbf{m}_v^{(k)} \right)
$$

**聚合函数的要求：** 聚合函数 $\bigoplus$ 需要满足：
- **排列不变性（Permutation Invariance）：** $\bigoplus_{u \in \mathcal{N}(v)} \mathbf{m}_{vu}$ 的结果不依赖邻居的顺序
- 常见选择：$\text{SUM}$、$\text{MEAN}$、$\text{MAX}$

**读出函数（Readout/Pooling）：** 对于图级任务，需要将所有节点表示聚合为图表示：

$$
\hat{\mathbf{y}} = \rho \left( \{ \mathbf{h}_v^{(K)} : v \in V \} \right)
$$

其中 $\rho$ 是读出函数，如全局均值池化 $\rho = \frac{1}{n} \sum_{v \in V} \mathbf{h}_v^{(K)}$。

### 2.2 图卷积网络（GCN）

**谱方法推导：**

Kipf和Welling（2017）从谱图理论出发推导GCN。回顾谱卷积：

$$
\mathbf{x} *_{\mathcal{G}} g_\theta = \mathbf{U} \text{diag}(\theta) \mathbf{U}^\top \mathbf{x}
$$

使用Chebyshev多项式的一阶近似（$K=1$），并令 $\lambda_{\max} \approx 2$：

$$
g_\theta *_{\mathcal{G}} \mathbf{x} \approx \theta_0 \mathbf{x} + \theta_1 (\mathbf{L} - \mathbf{I}) \mathbf{x}
$$

由于 $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$（归一化拉普拉斯），代入得：

$$
g_\theta *_{\mathcal{G}} \mathbf{x} \approx \theta_0 \mathbf{x} - \theta_1 \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} \mathbf{x}
$$

进一步约束 $\theta = \theta_0 = -\theta_1$（减少参数，缓解过拟合）：

$$
g_\theta *_{\mathcal{G}} \mathbf{x} \approx \theta \left( \mathbf{I} + \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} \right) \mathbf{x}
$$

**重归一化技巧：** $\mathbf{I} + \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ 的特征值范围为 $[0, 2]$，多层堆叠时会导致数值不稳定（梯度消失/爆炸）。引入重归一化：

$$
\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_n, \quad \tilde{\mathbf{D}}_{ii} = \sum_{j} \tilde{A}_{ij}
$$

**GCN的层传播规则：**

$$
\mathbf{H}^{(k)} = \sigma \left( \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(k-1)} \mathbf{W}^{(k)} \right)
$$

其中：
- $\mathbf{H}^{(0)} = \mathbf{X} \in \mathbb{R}^{n \times d}$ 是输入特征矩阵
- $\mathbf{W}^{(k)} \in \mathbb{R}^{d_{k-1} \times d_k}$ 是第 $k$ 层的可学习权重
- $\sigma$ 是非线性激活函数（如ReLU）
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_n$ 是加了自环的邻接矩阵

**空间视角理解：** GCN的传播规则可以从消息传递框架理解：

$$
\mathbf{h}_v^{(k)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{W}^{(k)} \mathbf{h}_u^{(k-1)} \right)
$$

即每个节点聚合自身和邻居的特征，按度数归一化后线性变换。

**GCN的局限性：**
1. **过平滑（Over-smoothing）：** 多层GCN堆叠后，节点表示趋于一致
2. **无法区分不同邻居的重要性**（所有邻居使用相同权重）
3. **转导学习（Transductive）：** 需要知道全图结构

### 2.3 GraphSAGE：采样聚合方法

GraphSAGE（SAmple and aggreGatE）由Hamilton等人（2017）提出，解决GCN的转导学习限制，支持归纳学习。

**核心思想：** 不使用全图邻居，而是对每个节点采样固定数量的邻居，然后聚合采样邻居的信息。

**GraphSAGE的层更新规则：**

$$
\mathbf{h}_v^{(k)} = \sigma \left( \mathbf{W}^{(k)} \cdot \text{CONCAT} \left( \mathbf{h}_v^{(k-1)}, \text{AGGREGATE}^{(k)} \left( \{ \mathbf{h}_u^{(k-1)} : u \in \mathcal{S}_{\mathcal{N}(v)} \} \right) \right) \right)
$$

其中 $\mathcal{S}_{\mathcal{N}(v)}$ 是从邻居 $\mathcal{N}(v)$ 中采样的子集。

**三种聚合器：**

1. **均值聚合器（Mean Aggregator）：**

$$
\text{AGGREGATE}^{\text{mean}} = \frac{1}{|\mathcal{S}_{\mathcal{N}(v)}|} \sum_{u \in \mathcal{S}_{\mathcal{N}(v)}} \mathbf{h}_u^{(k-1)}
$$

2. **LSTM聚合器（LSTM Aggregator）：** 将邻居节点随机排列后输入LSTM，利用LSTM的非线性能力，但结果依赖输入顺序。

3. **池化聚合器（Pooling Aggregator）：**

$$
\text{AGGREGATE}^{\text{pool}} = \max \left( \{ \sigma(\mathbf{W}_{\text{pool}} \mathbf{h}_u^{(k-1)} + \mathbf{b}) : u \in \mathcal{S}_{\mathcal{N}(v)} \} \right)
$$

其中 $\max$ 是逐元素最大值操作。

**无监督损失函数：** GraphSAGE可以使用基于图结构的无监督损失：

$$
\mathcal{J}_G(\mathbf{z}_v) = -\log \left( \sigma(\mathbf{z}_v^\top \mathbf{z}_u) \right) - Q \cdot \mathbb{E}_{v_n \sim P_n} \log \left( \sigma(-\mathbf{z}_v^\top \mathbf{z}_{v_n}) \right)
$$

其中 $\mathbf{z}_v$ 是节点 $v$ 的输出表示，$u$ 是 $v$ 的邻居，$v_n$ 是负采样节点，$Q$ 是负样本数，$P_n$ 是负采样分布。

### 2.4 图注意力网络（GAT）

GAT（Graph Attention Network）由Veličković等人（2018）提出，通过注意力机制为不同邻居分配不同权重。

**注意力系数计算：**

**步骤1：线性变换**

$$
\mathbf{h}_u' = \mathbf{W} \mathbf{h}_u, \quad \mathbf{h}_v' = \mathbf{W} \mathbf{h}_v
$$

其中 $\mathbf{W} \in \mathbb{R}^{d' \times d}$ 是共享权重矩阵。

**步骤2：计算注意力分数**

$$
e_{vu} = \text{LeakyReLU} \left( \mathbf{a}^\top [\mathbf{h}_v' \| \mathbf{h}_u'] \right)
$$

其中 $\mathbf{a} \in \mathbb{R}^{2d'}$ 是注意力向量，$\|$ 表示拼接操作。

**步骤3：归一化注意力系数**

$$
\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}
$$

**步骤4：加权聚合**

$$
\mathbf{h}_v^{(k)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}^{(k)} \mathbf{h}_u^{(k-1)} \right)
$$

**多头注意力（Multi-Head Attention）：** 类似Transformer，GAT使用多头注意力增强表达能力：

$$
\mathbf{h}_v^{(k)} = \Bigg\|_{i=1}^{H} \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(i)} \mathbf{W}^{(i)} \mathbf{h}_u^{(k-1)} \right)
$$

其中 $\|$ 表示拼接，$H$ 是注意力头数，$\alpha_{vu}^{(i)}$ 是第 $i$ 个头的注意力系数。

对于最后一层，通常使用平均而非拼接：

$$
\mathbf{h}_v^{(K)} = \sigma \left( \frac{1}{H} \sum_{i=1}^{H} \sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(i)} \mathbf{W}^{(i)} \mathbf{h}_u^{(K-1)} \right)
$$

**GAT与GCN的关系：** GCN可以看作GAT的特例——当注意力权重固定为 $\alpha_{vu} = \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}}$ 时，GAT退化为GCN。

**GAT的优势：**
1. 不同邻居获得不同权重，表达力更强
2. 注意力权重可解释
3. 适用于归纳学习（不需要全图结构）
4. 计算复杂度为 $O(|\mathcal{E}|)$，可并行

### 2.5 代码示例：GCN与GAT实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, H, A_tilde, D_tilde_inv_sqrt):
        support = H @ self.W
        norm_A = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        output = norm_A @ support + self.bias
        return output

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, X, A):
        n = A.size(0)
        A_tilde = A + torch.eye(n, device=A.device)
        D_tilde = torch.diag(A_tilde.sum(dim=1))
        D_tilde_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_tilde.diag()))
        H = F.relu(self.layer1(X, A_tilde, D_tilde_inv_sqrt))
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = self.layer2(H, A_tilde, D_tilde_inv_sqrt)
        return F.log_softmax(H, dim=1)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.a = nn.Parameter(torch.randn(2 * out_features, 1) * 0.01)
        self.dropout = dropout
        self.alpha = alpha
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, H, A):
        n = H.size(0)
        Wh = H @ self.W
        Wh1 = Wh @ self.a[:self.a.size(0)//2, :]
        Wh2 = Wh @ self.a[self.a.size(0)//2:, :]
        e = self.leaky_relu(Wh1 + Wh2.T)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        return attention @ Wh

class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads=8, dropout=0.6):
        super().__init__()
        self.attentions = nn.ModuleList([
            GATLayer(in_features, hidden_features, dropout) for _ in range(num_heads)
        ])
        self.out_att = GATLayer(hidden_features * num_heads, out_features, dropout)
        self.dropout = dropout

    def forward(self, X, A):
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = torch.cat([att(X, A) for att in self.attentions], dim=1)
        X = F.elu(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.out_att(X, A)
        return F.log_softmax(X, dim=1)
```

---

## 3. 知识图谱表示学习

### 3.1 知识图谱基础

**定义 12.9（知识图谱）** 知识图谱是一个多关系图 $G = (\mathcal{E}, \mathcal{R}, \mathcal{T})$，其中：
- $\mathcal{E}$ 是实体集合
- $\mathcal{R}$ 是关系集合
- $\mathcal{T} = \{(h, r, t) : h, t \in \mathcal{E}, r \in \mathcal{R}\}$ 是三元组集合

每个三元组 $(h, r, t)$ 表示头实体 $h$ 通过关系 $r$ 与尾实体 $t$ 相关联。

**知识图谱表示学习的目标：** 学习实体和关系的低维向量表示，使得真实三元组的得分高于虚假三元组。

### 3.2 TransE：平移嵌入

TransE（Bordes等人，2013）是最简单的知识图谱嵌入模型，其核心假设是：**关系是实体之间的平移**。

**得分函数：**

$$
f_r(h, t) = -\| \mathbf{h} + \mathbf{r} - \mathbf{t} \|
$$

理想情况下，若 $(h, r, t)$ 成立，则 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$。

**训练目标：** 使用margin-based排序损失：

$$
\mathcal{L} = \sum_{(h,r,t) \in \mathcal{T}} \sum_{(h',r,t') \in \mathcal{T}'} \max \left( 0, \gamma - f_r(h,t) + f_r(h',t') \right)
$$

其中 $\gamma > 0$ 是margin超参数，$\mathcal{T}'$ 是负采样三元组集合（通过替换头实体或尾实体生成）：

$$
\mathcal{T}' = \{(h', r, t) : h' \neq h\} \cup \{(h, r, t') : t' \neq t\}
$$

**TransE的局限性：** 无法很好地处理1-to-N、N-to-1和N-to-N关系。例如，对于关系"国籍"，一个头实体可能对应多个尾实体（一个演员可能有多个国籍），此时 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}_1$ 和 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}_2$ 要求 $\mathbf{t}_1 \approx \mathbf{t}_2$，但不同实体应有不同表示。

### 3.3 TransH：超平面投影

TransH（Wang等人，2014）通过将实体投影到关系特定的超平面上解决TransE的局限性。

**投影操作：** 对于关系 $r$，定义法向量 $\mathbf{w}_r$（$\|\mathbf{w}_r\|_2 = 1$），实体在超平面上的投影为：

$$
\mathbf{h}_\perp = \mathbf{h} - \mathbf{w}_r^\top \mathbf{h} \cdot \mathbf{w}_r
$$

$$
\mathbf{t}_\perp = \mathbf{t} - \mathbf{w}_r^\top \mathbf{t} \cdot \mathbf{w}_r
$$

**得分函数：**

$$
f_r(h, t) = -\| \mathbf{h}_\perp + \mathbf{d}_r - \mathbf{t}_\perp \|
$$

其中 $\mathbf{d}_r$ 是关系 $r$ 在超平面上的平移向量。

**投影的几何解释：** $\mathbf{h}_\perp = \mathbf{h} - (\mathbf{w}_r^\top \mathbf{h}) \mathbf{w}_r$ 是 $\mathbf{h}$ 在法向量为 $\mathbf{w}_r$ 的超平面上的正交投影。这使得同一实体在不同关系下有不同的投影表示，从而允许一个实体通过同一关系与多个不同实体关联。

**约束条件：**

$$
\|\mathbf{w}_r\|_2 = 1, \quad \mathbf{w}_r^\top \mathbf{d}_r = 0, \quad \|\mathbf{h}\|_2 \leq 1, \quad \|\mathbf{t}\|_2 \leq 1
$$

### 3.4 TransR：关系空间投影

TransR（Lin等人，2015）进一步将实体和关系映射到不同的空间中。

**投影操作：** 对于关系 $r$，定义投影矩阵 $\mathbf{M}_r \in \mathbb{R}^{k \times d}$（$d$ 是实体空间维度，$k$ 是关系空间维度）：

$$
\mathbf{h}_r = \mathbf{M}_r \mathbf{h}, \quad \mathbf{t}_r = \mathbf{M}_r \mathbf{t}
$$

**得分函数：**

$$
f_r(h, t) = -\| \mathbf{h}_r + \mathbf{r} - \mathbf{t}_r \|_2
$$

**TransR的优势：** 实体空间和关系空间解耦，关系投影矩阵 $\mathbf{M}_r$ 提供了更灵活的映射能力。

**TransR的参数量：** 每个关系需要 $k \times d$ 个参数（投影矩阵），当关系数量较多时参数量较大。

**Trans系列模型的统一视角：**

| 模型 | 投影方式 | 得分函数 | 参数量/关系 |
|------|---------|---------|-----------|
| TransE | 无投影 | $-\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|$ | $d$ |
| TransH | 超平面投影 | $-\|\mathbf{h}_\perp+\mathbf{d}_r-\mathbf{t}_\perp\|$ | $2d$ |
| TransR | 矩阵投影 | $-\|\mathbf{M}_r\mathbf{h}+\mathbf{r}-\mathbf{M}_r\mathbf{t}\|$ | $kd+d$ |

### 3.5 RotatE：旋转嵌入

RotatE（Sun等人，2019）将知识图谱嵌入建模为复数空间中的旋转操作，能够建模包括对称/反对称、逆、组合等多种关系模式。

**核心思想：** 将每个实体映射到复数向量 $\mathbf{h}, \mathbf{t} \in \mathbb{C}^d$，每个关系映射为复数向量 $\mathbf{r} \in \mathbb{C}^d$，满足 $|r_i| = 1$（即 $\mathbf{r}$ 的每个分量是单位复数，对应旋转）。

**得分函数：**

$$
f_r(h, t) = -\| \mathbf{h} \odot \mathbf{r} - \mathbf{t} \|
$$

其中 $\odot$ 是逐元素复数乘法（Hadamard积）：

$$
(h \odot r)_i = h_i \cdot r_i = |h_i| e^{i(\theta_{h_i} + \theta_{r_i})}
$$

**关系模式的建模能力：**

1. **对称/反对称关系：** 若 $r$ 是对称的（$r(x,y) \Rightarrow r(y,x)$），则 $\mathbf{r} = e^{i\pi} = -1$（旋转 $\pi$），此时 $\mathbf{h} \odot (-1) = -\mathbf{h} = \mathbf{t}$ 且 $\mathbf{t} \odot (-1) = -\mathbf{t} = \mathbf{h}$，要求 $\mathbf{h} = -\mathbf{t}$，即对称关系要求头尾实体互为相反数。若 $r$ 是反对称的，则 $\mathbf{r} \neq -1$。

2. **逆关系：** 若 $r_1$ 是 $r_2$ 的逆关系，则 $\mathbf{r}_1 \odot \mathbf{r}_2 = \mathbf{1}$（两次旋转回到原点），即 $\mathbf{r}_1 = \overline{\mathbf{r}_2}$。

3. **组合关系：** 若 $r_1(x,z) \wedge r_2(z,y) \Rightarrow r_3(x,y)$，则 $\mathbf{r}_1 \odot \mathbf{r}_2 = \mathbf{r}_3$（旋转的复合仍是旋转）。

**自对抗负采样损失：** RotatE提出自对抗负采样（Self-Adversarial Negative Sampling）：

$$
\mathcal{L} = -\log \sigma(\gamma - f_r(h,t)) - \sum_{i=1}^{n} p(h_i', r, t_i') \log \sigma(f_r(h_i', t_i') - \gamma)
$$

其中负样本的采样概率按得分加权：

$$
p(h_j', r, t_j' | \{(h,r,t)\}) = \frac{\exp \alpha f_r(h_j', t_j')}{\sum_i \exp \alpha f_r(h_i', t_i')}
$$

$\alpha$ 是温度参数，这使得模型更关注"困难"负样本。

### 3.6 双曲空间嵌入

双曲空间具有"指数级增长的体积"，天然适合建模层次结构和树形结构，是知识图谱嵌入的理想空间。

**Poincaré球模型：** $n$ 维Poincaré球定义为：

$$
\mathbb{B}^n = \{ \mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| < 1 \}
$$

**度规张量：** Poincaré球上的黎曼度规为：

$$
g_\mathbf{x} = \lambda_\mathbf{x}^2 g^E, \quad \lambda_\mathbf{x} = \frac{2}{1 - \|\mathbf{x}\|^2}
$$

其中 $g^E$ 是欧氏度规，$\lambda_\mathbf{x}$ 是共形因子。

**双曲距离：** Poincaré球上两点 $\mathbf{x}, \mathbf{y} \in \mathbb{B}^n$ 之间的双曲距离为：

$$
d_\mathbb{B}(\mathbf{x}, \mathbf{y}) = \text{arccosh} \left( 1 + 2 \frac{\|\mathbf{x} - \mathbf{y}\|^2}{(1 - \|\mathbf{x}\|^2)(1 - \|\mathbf{y}\|^2)} \right)
$$

**Möbius加法：** 双曲空间中的"加法"操作通过Möbius加法定义：

$$
\mathbf{x} \oplus_\mathbb{B} \mathbf{y} = \frac{(1 + 2\langle \mathbf{x}, \mathbf{y} \rangle + \|\mathbf{y}\|^2)\mathbf{x} + (1 - \|\mathbf{x}\|^2)\mathbf{y}}{1 + 2\langle \mathbf{x}, \mathbf{y} \rangle + \|\mathbf{x}\|^2 \|\mathbf{y}\|^2}
$$

**双曲空间中的TransE（MuRP）：** 在双曲空间中，关系建模为Möbius旋转和平移：

$$
f_r(h, t) = -d_\mathbb{B}(\mathbf{R}_r \otimes \mathbf{h}, \mathbf{t} \oplus_\mathbb{B} \mathbf{r})
$$

其中 $\mathbf{R}_r$ 是关系特定的对角旋转矩阵，$\otimes$ 是Möbius旋转操作。

**双曲空间的优势：**
- 树形结构可以用低维双曲嵌入近乎无损地表示
- 层次结构中，根节点靠近原点，叶节点靠近边界
- 相比欧氏空间，双曲空间可以用更少的维度编码层次信息

### 3.7 代码示例：知识图谱嵌入模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=128, margin=1.0, norm=1):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.margin = margin
        self.norm = norm
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = F.normalize(self.entity_emb(h), p=2, dim=-1)
        r_emb = F.normalize(self.relation_emb(r), p=2, dim=-1)
        t_emb = F.normalize(self.entity_emb(t), p=2, dim=-1)
        score = -torch.norm(h_emb + r_emb - t_emb, p=self.norm, dim=-1)
        return score

    def loss(self, pos_score, neg_score):
        return F.relu(self.margin - pos_score + neg_score).mean()

class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=128, margin=12.0, epsilon=2.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.margin = margin
        self.epsilon = epsilon
        entity_range = nn.Parameter(torch.arange(0, num_entities), requires_grad=False)
        self.register_buffer('entity_range', entity_range)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        pi = 3.14159265358979323846
        r_phase = r_emb / (self.epsilon + 1e-8) * pi
        r_real = torch.cos(r_phase)
        r_imag = torch.sin(r_phase)
        h_real, h_imag = h_emb, torch.zeros_like(h_emb)
        t_real, t_imag = t_emb, torch.zeros_like(t_emb)
        hr_real = h_real * r_real - h_imag * r_imag
        hr_imag = h_real * r_imag + h_imag * r_real
        diff_real = hr_real - t_real
        diff_imag = hr_imag - t_imag
        score = -torch.sqrt(diff_real ** 2 + diff_imag ** 2 + 1e-12).sum(dim=-1)
        return score

    def loss(self, pos_score, neg_score, alpha=1.0):
        pos_loss = -F.logsigmoid(self.margin + pos_score)
        neg_weights = F.softmax(neg_score * alpha, dim=-1).detach()
        neg_loss = (neg_weights * -F.logsigmoid(-neg_score - self.margin)).sum(dim=-1)
        return (pos_loss + neg_loss).mean()

class PoincareEmbedding(nn.Module):
    def __init__(self, num_entities, dim=64, epsilon=1e-5):
        super().__init__()
        self.emb = nn.Embedding(num_entities, dim)
        self.epsilon = epsilon
        nn.init.xavier_uniform_(self.emb.weight)

    def poincare_distance(self, x, y):
        x_norm = torch.clamp(torch.sum(x ** 2, dim=-1), 0, 1 - self.epsilon)
        y_norm = torch.clamp(torch.sum(y ** 2, dim=-1), 0, 1 - self.epsilon)
        diff_norm = torch.sum((x - y) ** 2, dim=-1)
        numerator = 2 * diff_norm
        denominator = (1 - x_norm) * (1 - y_norm)
        cosh = 1 + numerator / (denominator + 1e-10)
        return torch.arccosh(torch.clamp(cosh, min=1.0 + 1e-10))

    def project_to_ball(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * (1 - self.epsilon) / norm.clamp(min=1 - self.epsilon)

    def forward(self, h, t):
        h_emb = self.project_to_ball(self.emb(h))
        t_emb = self.project_to_ball(self.emb(t))
        return -self.poincare_distance(h_emb, t_emb)
```

---

## 4. 知识图谱推理

### 4.1 路径推理

路径推理（Path Reasoning）利用知识图谱中连接两个实体的路径进行推理。

**关系路径：** 给定实体 $e_1$ 和 $e_k$，关系路径定义为：

$$
P = e_1 \xrightarrow{r_1} e_2 \xrightarrow{r_2} \cdots \xrightarrow{r_{k-1}} e_k
$$

**路径推理的直觉：** 若 $(e_1, r_1, e_2)$ 和 $(e_2, r_2, e_3)$ 成立，则可能推出 $(e_1, r_3, e_3)$ 成立，记为 $r_1 \circ r_2 \Rightarrow r_3$。

**Path Ranking Algorithm（PRA）：** Lao等人（2011）提出的路径排序算法：

1. **路径生成：** 为每个目标关系 $r$ 枚举所有长度 $\leq L$ 的关系路径 $\pi = r_1 \circ r_2 \circ \cdots \circ r_l$

2. **路径特征计算：** 对每个实体对 $(h, t)$ 和路径 $\pi$，计算路径特征值：

$$
s(h, \pi, t) = P(t | h, \pi) = \prod_{i=1}^{l} P(e_{i+1} | e_i, r_i)
$$

对于随机游走解释：

$$
s(h, \pi, t) = \sum_{e_2, \ldots, e_{l-1}} \prod_{i=1}^{l} \frac{1}{d_{\text{out}}(e_i, r_i)} \cdot \mathbf{1}[(e_i, r_i, e_{i+1}) \in \mathcal{T}]
$$

其中 $d_{\text{out}}(e_i, r_i)$ 是实体 $e_i$ 沿关系 $r_i$ 的出度。

3. **分类器训练：** 使用逻辑回归预测目标关系：

$$
P(r(h,t) = 1) = \sigma \left( w_0 + \sum_{\pi} w_\pi \cdot s(h, \pi, t) \right)
$$

**神经路径推理（Neural Reasoning over Paths）：** 将路径编码为向量表示：

$$
\mathbf{p} = f(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_l)
$$

其中 $f$ 可以是RNN、LSTM或简单的加权求和。路径的可靠性得分：

$$
s(h, r, t) = \text{sim}(\mathbf{p}, \mathbf{r})
$$

### 4.2 规则学习

规则学习从知识图谱中自动发现逻辑规则，用于推理和知识补全。

**Horn规则：** 知识图谱规则学习的标准形式是Horn子句：

$$
r_1(X, Z_1) \wedge r_2(Z_1, Z_2) \wedge \cdots \wedge r_l(Z_{l-1}, Y) \Rightarrow r(X, Y)
$$

简记为 $B \Rightarrow r(X,Y)$，其中 $B$ 是规则体（Body），$r(X,Y)$ 是规则头（Head）。

**规则质量度量：**

1. **支持度（Support）：** 规则在知识图谱中的实例数

$$
\text{Supp}(B \Rightarrow r) = |\{(x,y) : B(x,y) \text{ 在KG中成立} \wedge r(x,y) \text{ 在KG中成立}\}|
$$

2. **置信度（Confidence）：** 规则体成立时规则头也成立的概率

$$
\text{Conf}(B \Rightarrow r) = \frac{\text{Supp}(B \Rightarrow r)}{|\{(x,y) : B(x,y) \text{ 在KG中成立}\}|}
$$

3. **PCA置信度（Partial Completeness Assumption）：** 考虑知识图谱不完整的修正置信度

$$
\text{PCA\_Conf}(B \Rightarrow r) = \frac{\text{Supp}(B \Rightarrow r)}{|\{(x,y) : B(x,y) \text{ 在KG中成立}\}| - \text{Supp}(B \Rightarrow \neg r)}
$$

**AMIE规则学习系统：** AMIE（AMIE+）通过枚举和剪枝策略高效发现高质量规则：

1. 从空规则体开始，逐步添加原子（atom）
2. 使用置信度下界进行剪枝
3. 支持闭路径（Closed Path）约束保证规则质量

**神经规则学习（Neural Rule Learning）：** 将规则学习建模为可微过程：

$$
\text{Rule\_Score}(B \Rightarrow r) = \sigma \left( \mathbf{w}_r^\top \phi(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_l) + b \right)
$$

其中 $\phi$ 是路径编码器（如LSTM），$\mathbf{w}_r$ 是关系特定的权重向量。

### 4.3 知识图谱与LLM的结合

**知识增强检索（Knowledge-Enhanced Retrieval）：** 将知识图谱作为外部知识源增强大模型的生成能力。

**检索增强生成（RAG）与知识图谱的结合：**

1. **实体链接：** 将查询中的实体链接到知识图谱中的节点

$$
\text{Link}(q) = \arg\max_{e \in \mathcal{E}} \text{sim}(\text{Enc}(q), \mathbf{e})
$$

2. **子图检索：** 从知识图谱中检索与查询相关的子图

$$
\mathcal{G}_q = \text{Extract}(G, \text{Link}(q), k\text{-hop})
$$

3. **知识序列化：** 将检索到的子图转换为文本序列

$$
\text{Serialize}(\mathcal{G}_q) = \{(h_i, r_i, t_i)\}_{i=1}^{|\mathcal{T}_q|}
$$

4. **增强生成：** 将序列化的知识作为上下文输入LLM

$$
\hat{y} = \text{LLM}(\text{Serialize}(\mathcal{G}_q) \| q)
$$

**KG-LLM联合训练：** 将知识图谱嵌入与语言模型联合训练：

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda \mathcal{L}_{\text{KG}}
$$

其中 $\mathcal{L}_{\text{LM}}$ 是语言模型损失，$\mathcal{L}_{\text{KG}}$ 是知识图谱嵌入损失：

$$
\mathcal{L}_{\text{KG}} = \sum_{(h,r,t) \in \mathcal{T}} \max(0, \gamma + f_r(h,t) - f_r(h',t'))
$$

### 4.4 代码示例：路径推理与规则评估

```python
import numpy as np
from collections import defaultdict

class PathReasoner:
    def __init__(self, triples):
        self.triples = triples
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for h, r, t in triples:
            self.outgoing[(h, r)].append(t)
            self.incoming[(t, r)].append(h)

    def find_paths(self, source, max_length=3, max_paths=100):
        paths = []
        queue = [(source, [], [source])]
        while queue and len(paths) < max_paths:
            current, rel_path, entity_path = queue.pop(0)
            if len(rel_path) >= max_length:
                continue
            for r in set(r for h, r in self.outgoing if h == current):
                for t in self.outgoing[(current, r)]:
                    new_rel_path = rel_path + [r]
                    new_entity_path = entity_path + [t]
                    if len(new_rel_path) > 0:
                        paths.append((tuple(new_rel_path), new_entity_path))
                    queue.append((t, new_rel_path, new_entity_path))
        return paths

    def compute_path_feature(self, h, path_relations, t):
        current_entities = {h}
        for r in path_relations:
            next_entities = set()
            for e in current_entities:
                if (e, r) in self.outgoing:
                    next_entities.update(self.outgoing[(e, r)])
            current_entities = next_entities
            if not current_entities:
                return 0.0
        return 1.0 if t in current_entities else 0.0

    def compute_random_walk_prob(self, h, path_relations, t):
        current = {h: 1.0}
        for r in path_relations:
            next_dist = defaultdict(float)
            for e, prob in current.items():
                neighbors = self.outgoing.get((e, r), [])
                if neighbors:
                    transition = 1.0 / len(neighbors)
                    for n in neighbors:
                        next_dist[n] += prob * transition
            current = dict(next_dist)
            if not current:
                return 0.0
        return current.get(t, 0.0)

class RuleEvaluator:
    def __init__(self, triples):
        self.triples_set = set(triples)
        self.hr_to_t = defaultdict(set)
        for h, r, t in triples:
            self.hr_to_t[(h, r)].add(t)

    def compute_confidence(self, body_triples_func, head_relation):
        body_count = 0
        support_count = 0
        all_entities = set()
        for h, r, t in self.triples_set:
            all_entities.add(h)
            all_entities.add(t)
        for x in all_entities:
            for y in all_entities:
                if body_triples_func(x, y, self):
                    body_count += 1
                    if y in self.hr_to_t.get((x, head_relation), set()):
                        support_count += 1
        if body_count == 0:
            return 0.0
        return support_count / body_count
```

---

## 5. 图神经网络在大模型中的应用

### 5.1 知识图谱增强RAG

传统RAG基于文本检索，知识图谱增强RAG（KG-RAG）利用图结构提供更精确、更结构化的知识检索。

**KG-RAG的流程：**

1. **查询理解：** 将自然语言查询解析为结构化查询

$$
q \xrightarrow{\text{NLU}} (e_q, r_q)
$$

2. **实体链接与消歧：** 将查询实体映射到知识图谱

$$
e_q \xrightarrow{\text{Link}} e \in \mathcal{E}
$$

3. **子图检索：** 检索与查询实体相关的子图

$$
\mathcal{G}_q = \{ v : v \in \mathcal{N}_k(e) \}
$$

其中 $\mathcal{N}_k(e)$ 表示实体 $e$ 的 $k$-hop邻居。

4. **GNN编码：** 使用GNN对子图进行编码

$$
\mathbf{z}_v = \text{GNN}(\mathcal{G}_q, \mathbf{X})_v
$$

5. **相关性排序：** 对检索到的知识进行排序

$$
\text{score}(v) = \text{sim}(\mathbf{z}_q, \mathbf{z}_v)
$$

6. **增强生成：** 将排序后的知识注入LLM

**KG-RAG的数学框架：**

$$
P(y | q) = \sum_{\mathcal{K}} P(y | q, \mathcal{K}) P(\mathcal{K} | q, G)
$$

其中 $\mathcal{K}$ 是从知识图谱 $G$ 中检索的知识片段，$P(\mathcal{K} | q, G)$ 由GNN编码和相似度计算决定：

$$
P(\mathcal{K} | q, G) = \frac{\exp(\text{sim}(\mathbf{z}_q, \mathbf{z}_\mathcal{K}) / \tau)}{\sum_{\mathcal{K}'} \exp(\text{sim}(\mathbf{z}_q, \mathbf{z}_{\mathcal{K}'}) / \tau)}
$$

**与文本RAG的对比：**

| 特性 | 文本RAG | KG-RAG |
|------|---------|--------|
| 知识表示 | 非结构化文本 | 结构化三元组 |
| 检索方式 | 向量相似度 | 图遍历+GNN编码 |
| 推理能力 | 有限 | 支持多跳推理 |
| 可解释性 | 较弱 | 可追溯推理路径 |
| 知识更新 | 重新索引 | 增量更新三元组 |

### 5.2 图结构化知识注入

将知识图谱的结构化信息注入到大模型中，使模型具备结构化知识理解能力。

**方法1：知识图谱嵌入作为软提示（KG Embedding as Soft Prompt）**

将知识图谱嵌入向量作为软提示拼接到输入嵌入中：

$$
\mathbf{E}_{\text{input}} = [\mathbf{E}_{\text{token}}; \mathbf{E}_{\text{KG}}]
$$

其中 $\mathbf{E}_{\text{KG}}$ 是从知识图谱中检索的相关实体的嵌入向量，通过投影层对齐到语言模型的嵌入空间：

$$
\mathbf{E}_{\text{KG}} = \text{Proj}(\mathbf{z}_e) = \mathbf{W}_p \mathbf{z}_e + \mathbf{b}_p
$$

$\mathbf{z}_e$ 是GNN编码的实体表示，$\mathbf{W}_p \in \mathbb{R}^{d_{\text{LM}} \times d_{\text{GNN}}}$ 是投影矩阵。

**方法2：知识感知注意力（Knowledge-Aware Attention）**

在Transformer的注意力计算中融入知识图谱信息：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{KG}}\right) V
$$

其中 $\mathbf{M}_{\text{KG}}$ 是知识图谱结构信息的偏置矩阵：

$$
M_{\text{KG},ij} = \begin{cases} \beta \cdot \text{sim}(\mathbf{z}_{e_i}, \mathbf{z}_{e_j}), & \text{若 } e_i, e_j \text{ 在KG中相连} \\ 0, & \text{否则} \end{cases}
$$

$\beta$ 是可学习的标量参数，控制知识图谱偏置的强度。

**方法3：图-文本联合编码（Graph-Text Joint Encoding）**

构建统一的图-文本异构图，其中节点包括文本token和知识图谱实体，边包括文本顺序边、知识图谱关系边和跨模态链接边：

$$
\mathbf{h}_v^{(k)} = \phi^{(k)} \left( \mathbf{h}_v^{(k-1)}, \bigoplus_{u \in \mathcal{N}(v)} \alpha_{vu}^{(k)} \psi^{(k)}(\mathbf{h}_v^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{e}_{vu}) \right)
$$

其中 $\alpha_{vu}^{(k)}$ 由异构注意力计算：

$$
\alpha_{vu}^{(k)} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}_{\tau(v)\tau(u)}^\top [\mathbf{W}^{(k)} \mathbf{h}_v^{(k-1)} \| \mathbf{W}^{(k)} \mathbf{h}_u^{(k-1)}]))}{\sum_{u' \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}_{\tau(v)\tau(u')}^\top [\mathbf{W}^{(k)} \mathbf{h}_v^{(k-1)} \| \mathbf{W}^{(k)} \mathbf{h}_{u'}^{(k-1)}]))}
$$

$\tau(v)$ 表示节点 $v$ 的类型（token或entity），$\mathbf{a}_{\tau(v)\tau(u)}$ 是类型对特定的注意力向量。

### 5.3 多跳推理

多跳推理（Multi-hop Reasoning）是知识图谱与大模型结合的核心能力之一，指需要经过多条边/多个推理步骤才能得出结论的推理过程。

**多跳推理的形式化定义：** 给定查询 $q$ 和知识图谱 $G$，多跳推理需要找到长度为 $L$ 的推理链：

$$
e_0 \xrightarrow{r_1} e_1 \xrightarrow{r_2} e_2 \xrightarrow{\cdots} \xrightarrow{r_L} e_L
$$

使得 $e_0$ 是查询中的源实体，$e_L$ 是答案实体。

**基于GNN的多跳推理：**

**推理链的GNN编码：** 使用多层GNN，每一层对应一跳推理：

$$
\mathbf{h}_v^{(l)} = \text{GNNLayer}^{(l)}(\mathbf{h}_v^{(l-1)}, \{ \mathbf{h}_u^{(l-1)} : u \in \mathcal{N}(v) \})
$$

第 $l$ 层的输出 $\mathbf{h}_v^{(l)}$ 编码了从起始节点出发经过 $l$ 跳可达的信息。

**查询感知的注意力：** 引入查询向量 $\mathbf{q}$ 引导推理方向：

$$
\alpha_{vu}^{(l)} = \frac{\exp(\mathbf{q}^\top \mathbf{W}_a [\mathbf{h}_v^{(l-1)} \| \mathbf{h}_u^{(l-1)} \| \mathbf{r}_{vu}])}{\sum_{u' \in \mathcal{N}(v)} \exp(\mathbf{q}^\top \mathbf{W}_a [\mathbf{h}_v^{(l-1)} \| \mathbf{h}_{u'}^{(l-1)} \| \mathbf{r}_{vu'}])}
$$

$$
\mathbf{h}_v^{(l)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(l)} \mathbf{W}^{(l)} \mathbf{h}_u^{(l-1)} \right)
$$

**答案预测：** 最终答案通过计算候选实体与查询的匹配度得到：

$$
P(e = \text{answer} | q, G) = \text{softmax}(\mathbf{q}^\top \mathbf{W}_o \mathbf{H}^{(L)})
$$

其中 $\mathbf{H}^{(L)} \in \mathbb{R}^{n \times d}$ 是所有节点在第 $L$ 层的表示矩阵。

**基于LLM的多跳推理：** 将多跳推理分解为多步生成过程：

$$
P(e_L | q, G) = \prod_{l=1}^{L} P(e_l | e_{l-1}, r_l, q, G)
$$

每一步由LLM生成中间推理步骤：

$$
P(e_l | e_{l-1}, r_l, q, G) = \text{LLM}(e_{l-1}, r_l, \text{Context}(e_{l-1}, G), q)
$$

其中 $\text{Context}(e_{l-1}, G)$ 是从知识图谱中检索的 $e_{l-1}$ 的邻居信息。

**思维链与图推理的结合（CoT + Graph Reasoning）：**

$$
\begin{aligned}
&\text{Step 1: } q \xrightarrow{\text{LLM}} \text{decompose}(q) = \{q_1, q_2, \ldots, q_L\} \\
&\text{Step 2: } q_l \xrightarrow{\text{KG}} \text{retrieve}(q_l, G) = \mathcal{K}_l \\
&\text{Step 3: } \mathcal{K}_l \xrightarrow{\text{LLM}} \text{reason}(q_l, \mathcal{K}_l) = a_l \\
&\text{Step 4: } \{a_1, a_2, \ldots, a_L\} \xrightarrow{\text{LLM}} \text{synthesize} = a
\end{aligned}
$$

**推理路径的可解释性：** 图结构天然提供了推理路径的可解释性——每一步推理都可以追溯到知识图谱中的具体三元组，形成可审计的推理链。

### 5.4 代码示例：知识图谱增强RAG系统

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KGRAGEncoder(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.relation_proj = nn.Linear(relation_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, entity_features, adj_matrix, query_entity_idx):
        H = self.entity_proj(entity_features)
        for gnn_layer in self.gnn_layers:
            H = F.relu(gnn_layer(H, adj_matrix))
        query_repr = H[query_entity_idx]
        scores = torch.matmul(H, query_repr.unsqueeze(-1)).squeeze(-1)
        scores = scores / (torch.norm(H, dim=-1) * torch.norm(query_repr, dim=-1, keepdim=True) + 1e-8)
        return scores, H

class KnowledgeAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, kg_bias_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.kg_proj = nn.Linear(kg_bias_dim, n_heads)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, X, kg_bias=None, mask=None):
        B, L, _ = X.shape
        Q = self.W_q(X).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if kg_bias is not None:
            kg_attn = self.kg_proj(kg_bias)
            attn = attn + self.beta * kg_attn.unsqueeze(1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return out

class MultiHopReasoner(nn.Module):
    def __init__(self, entity_dim, hidden_dim, num_hops=3):
        super().__init__()
        self.num_hops = num_hops
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.hop_layers = nn.ModuleList()
        for _ in range(num_hops):
            self.hop_layers.append(nn.ModuleDict({
                'message': nn.Linear(hidden_dim * 2, hidden_dim),
                'update': nn.GRUCell(hidden_dim, hidden_dim),
                'query_attn': nn.Linear(hidden_dim, 1)
            }))
        self.answer_proj = nn.Linear(hidden_dim, 1)

    def forward(self, entity_features, adj_list, query_vec):
        n = entity_features.size(0)
        H = self.entity_proj(entity_features)
        h_query = query_vec
        for hop in range(self.num_hops):
            messages = []
            for v in range(n):
                neighbors = adj_list[v]
                if len(neighbors) == 0:
                    messages.append(torch.zeros_like(H[v]))
                    continue
                neighbor_feats = H[neighbors]
                combined = torch.cat([
                    H[v].unsqueeze(0).expand(len(neighbors), -1),
                    neighbor_feats
                ], dim=-1)
                msg = F.relu(self.hop_layers[hop]['message'](combined))
                attn_logits = self.hop_layers[hop]['query_attn'](
                    torch.cat([msg, h_query.unsqueeze(0).expand(len(neighbors), -1)], dim=-1)
                )
                attn_weights = F.softmax(attn_logits, dim=0)
                agg_msg = (attn_weights * msg).sum(dim=0)
                messages.append(agg_msg)
            messages = torch.stack(messages)
            H = self.hop_layers[hop]['update'](messages, H)
        scores = self.answer_proj(H).squeeze(-1)
        return scores, H
```

---

## 本章小结

本章系统介绍了图神经网络与知识图谱的数学基础和核心方法：

1. **图论基础：** 图的矩阵表示（邻接矩阵、度矩阵、拉普拉斯矩阵）是GNN的理论基石；谱图理论通过拉普拉斯矩阵的特征分解定义了图傅里叶变换和谱卷积，Chebyshev多项式近似将谱卷积转化为高效的空域操作。

2. **图神经网络基础：** 消息传递框架统一了GNN的设计范式；GCN从谱方法推导出简洁的层传播规则，GraphSAGE通过采样聚合支持归纳学习，GAT通过注意力机制实现自适应邻居加权。

3. **知识图谱表示学习：** TransE/TransH/TransR系列模型从简单平移到超平面投影再到关系空间投影，逐步增强了对复杂关系的建模能力；RotatE通过复数旋转统一建模对称、逆、组合等关系模式；双曲空间嵌入利用其指数增长的体积特性天然适合编码层次结构。

4. **知识图谱推理：** 路径推理利用关系路径进行多步推理，规则学习从数据中自动发现逻辑规则，与LLM的结合使得知识图谱推理能力得到进一步增强。

5. **图神经网络在大模型中的应用：** 知识图谱增强RAG提供结构化的知识检索，图结构化知识注入通过软提示、知识感知注意力和图-文本联合编码等方式将图信息融入LLM，多跳推理结合GNN和LLM实现可解释的复杂推理。

**关键公式速查：**

| 方法 | 核心公式 |
|------|---------|
| GCN | $\mathbf{H}^{(k)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(k-1)}\mathbf{W}^{(k)})$ |
| GAT | $\mathbf{h}_v^{(k)} = \sigma(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}\mathbf{W}\mathbf{h}_u^{(k-1)})$ |
| TransE | $f_r(h,t) = -\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|$ |
| RotatE | $f_r(h,t) = -\|\mathbf{h} \odot \mathbf{r} - \mathbf{t}\|$ |
| 图傅里叶变换 | $\hat{\mathbf{x}} = \mathbf{U}^\top\mathbf{x}$ |
| 双曲距离 | $d_\mathbb{B}(\mathbf{x},\mathbf{y}) = \text{arccosh}(1 + 2\frac{\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)})$ |

---

## 深度分析

### 图数据的归纳偏置

GNN的核心设计哲学是"图的对称性决定架构"。与CNN利用平移不变性、Transformer利用排列不变性类似，GNN显式编码了图结构的置换等变性。这一归纳偏置使得GNN在分子性质预测、社交网络分析等图结构数据上天然优于通用架构。

### 消息传递的扩展

消息传递范式是GNN的基础，但存在表达力上限。2024-2026年的前沿工作通过引入全局注意力或高阶邻域聚合突破了这一上限。GraphRAG的兴起更是将GNN与检索增强生成结合——知识图谱的结构信息通过GNN编码后注入检索过程。

---

## GNN实践Checklist

- [ ] 理解消息传递范式（message/aggregate/update）
- [ ] 掌握GCN、GAT、GraphSAGE的核心差异
- [ ] 理解过平滑（over-smoothing）问题的数学原因
- [ ] 能够实现图级别的readout操作
- [ ] 理解Graph Transformer与vanilla Transformer的区别
- [ ] 掌握图数据增强策略（dropout/edge masking）
- [ ] 了解GNN与知识图谱结合的典型模式
- [ ] 能够诊断GNN的过拟合和欠拟合
- [ ] 理解WL test与GNN表达力的关系
- [ ] 了解GNN在大规模图上的可扩展训练方法

---

## 延伸阅读

- [线性代数](ch01-linear-algebra.md)
- [神经网络](ch06-neural-networks.md)
- [深度学习技巧](ch07-deep-learning-techniques.md)
- [注意力机制](ch08-attention-mechanism.md)
- [Transformer架构](ch09-transformer.md)

---

*最后更新：2026-06-12*

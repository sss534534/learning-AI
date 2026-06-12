# 第六章：神经网络基础

> 神经网络是深度学习的核心模型，其设计灵感来源于生物神经元的工作原理。本章将深入讲解神经网络的基本概念，包括**感知机**、**多层感知机**、**激活函数**和**网络架构**，为理解大型语言模型奠定基础。

## 目录

1. [神经元与感知机](#1-神经元与感知机)
2. [多层感知机](#2-多层感知机)
3. [激活函数](#3-激活函数)
4. [损失函数](#4-损失函数)
5. [网络架构设计](#5-网络架构设计)

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [第一章：线性代数](./ch01-linear-algebra.md), [第二章：微积分](./ch02-calculus.md)
- **关联文件**: [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md), [第八章：注意力机制](./ch08-attention-mechanism.md), [第九章：Transformer架构](./ch09-transformer.md)
- **最后更新**: 2026-06-12
---

## 1. 神经元与感知机

### 1.1 生物神经元与人工神经元

**生物神经元结构：**
```
    树突（输入）→ 细胞体 → 轴突（输出）
                    ↑
               细胞核（处理）
```

**人工神经元数学模型：**
$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)
$$

其中：
- $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$：输入向量
- $\mathbf{w} = (w_1, w_2, \ldots, w_n)^T$：权重向量
- $b$：偏置
- $f(\cdot)$：激活函数
- $y$：输出

### 1.2 感知机模型

**感知机（Perceptron）** 是最简单的神经网络模型：

```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # 使用sigmoid激活
```

**决策边界：** 感知机是一个线性分类器，其决策边界是超平面：
$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

### 1.3 感知机的局限性与解决

**问题：** 单层感知机无法解决XOR问题。

**XOR问题可视化：**
```
    XOR真值表：
    ┌─────┬─────┬───────┐
    │ x₁  │ x₂  │ x₁⊕x₂ │
    ├─────┼─────┼───────┤
    │  0  │  0  │   0   │
    │  0  │  1  │   1   │
    │  1  │  0  │   1   │
    │  1  │  1  │   0   │
    └─────┴─────┴───────┘
```

**解决方案：** 多层感知机（MLP）通过隐藏层实现非线性分类。

---

## 2. 多层感知机

### 2.1 MLP结构

**多层感知机（MLP）** 由输入层、隐藏层和输出层组成：

```
    输入层    隐藏层1    隐藏层2    输出层
      │          │          │          │
    ○─○─○──→  ○─○─○──→  ○─○─○──→  ○─○
      │          │          │          │
    ○─○─○──→  ○─○─○──→  ○─○─○──→  ○─○
      │          │          │          │
    ○─○─○──→  ○─○─○──→  ○─○─○──→  ○─○
      │          │          │          │
    x₁      h₁¹       h₂¹       ŷ₁
    x₂      h₁²       h₂²       ŷ₂
    x₃      h₁³       h₂³       ŷ₃
```

### 2.2 前向传播

**数学表达式：**

第 $l$ 层的输出：
$$
\mathbf{h}^{(l)} = f\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)
$$

**完整的前向传播：**
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()  # 激活函数
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 2.3 万能近似定理

**万能近似定理（Universal Approximation Theorem）：**

一个具有足够多隐藏层神经元的前馈神经网络，可以以任意精度逼近任意连续函数。

**数学表述：**
对于任意连续函数 $g: [0,1]^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在一个单隐藏层的神经网络 $f$，使得：
$$
\|f(\mathbf{x}) - g(\mathbf{x})\| < \epsilon, \quad \forall \mathbf{x} \in [0,1]^n
$$

**局限性：**
- 定理只保证存在性，不保证可学习性
- 实际上需要极多的神经元
- 深层网络通常比浅层网络更高效

---

## 3. 激活函数

### 3.1 为什么需要激活函数？

**核心原因：** 引入非线性，否则多层网络等价于单层线性变换。

**无激活函数的情况：**
$$
\mathbf{h}^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} = \mathbf{W}^{(2)} (\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}) = \tilde{\mathbf{W}} \mathbf{x} + \tilde{\mathbf{b}}
$$

这仍然是一个线性变换！

### 3.2 Sigmoid函数

**数学公式：**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**特点：**
- 输出范围：$(0, 1)$
- 曾经是最流行的激活函数
- 缺点：梯度消失、计算慢、输出不以0为中心

```python
import torch
import torch.nn.functional as F

x = torch.randn(5)
sigmoid_x = torch.sigmoid(x)
print(f"Sigmoid: {sigmoid_x}")
```

### 3.3 Tanh函数

**数学公式：**
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
$$

**特点：**
- 输出范围：$(-1, 1)$
- 输出以0为中心
- 梯度比Sigmoid更陡峭

```python
x = torch.randn(5)
tanh_x = torch.tanh(x)
print(f"Tanh: {tanh_x}")
```

### 3.4 ReLU函数

**数学公式：**
$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} 0 & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases}
$$

**特点：**
- 计算极快
- 缓解梯度消失问题
- 稀疏激活
- **缺点：Dying ReLU问题**（神经元可能永远不被激活）

```python
x = torch.randn(5)
relu_x = F.relu(x)
print(f"ReLU: {relu_x}")
```

### 3.5 Leaky ReLU

**数学公式：**
$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

**特点：** 解决Dying ReLU问题，允许负值有小的梯度。

```python
x = torch.randn(5)
leaky_relu = F.leaky_relu(x, negative_slope=0.01)
```

### 3.6 GELU（Gaussian Error Linear Unit）

**数学公式：**
$$
\text{GELU}(x) = x \cdot \Phi(x)
$$
其中 $\Phi(x)$ 是标准正态分布的CDF。

**近似形式：**
$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

**特点：**
- Transformer架构的标准激活函数
- BERT、GPT等模型使用

```python
x = torch.randn(5)
gelu_x = F.gelu(x)  # PyTorch内置
print(f"GELU: {gelu_x}")
```

### 3.7 SiLU（Swish）

**数学公式：**
$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**特点：** 自门控激活函数，在某些任务上优于ReLU。

### 3.8 激活函数对比

| 激活函数 | 公式 | 输出范围 | 优点 | 缺点 |
|----------|------|----------|------|------|
| Sigmoid | $1/(1+e^{-x})$ | (0,1) | 输出概率 | 梯度消失 |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | (-1,1) | 以0为中心 | 梯度消失 |
| ReLU | $\max(0,x)$ | [0,+∞) | 计算快、不梯度消失 | Dying ReLU |
| LeakyReLU | $\max(\alpha x,x)$ | (-∞,+∞) | 解决Dying ReLU | 超参数α |
| GELU | $x\Phi(x)$ | (-∞,+∞) | Transformer标配 | 计算稍慢 |

---

## 4. 损失函数

### 4.1 损失函数的定义

**损失函数** 衡量模型预测与真实值之间的差距。

**数学定义：**
$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f(x_i; \theta))
$$

### 4.2 回归任务损失函数

#### 4.2.1 均方误差（MSE）

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

```python
criterion = nn.MSELoss()
loss = criterion(predictions, targets)
```

#### 4.2.2 平均绝对误差（MAE）

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

```python
criterion = nn.L1Loss()
loss = criterion(predictions, targets)
```

### 4.3 分类任务损失函数

#### 4.3.1 二分类交叉熵

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]
$$

```python
criterion = nn.BCEWithLogitsLoss()  # 带sigmoid的BCE
loss = criterion(logits, targets)
```

#### 4.3.2 多分类交叉熵

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}
$$

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target_class_indices)
```

---

## 5. 网络架构设计

### 5.1 全连接层

**全连接层（FC/Dense）：** 每个神经元与前一层的所有神经元相连。

```python
nn.Linear(in_features, out_features, bias=True)
```

### 5.2 常见网络架构

| 架构 | 特点 | 应用 |
|------|------|------|
| MLP | 全连接、非循环 | 分类、回归 |
| CNN | 局部连接、权重共享 | 图像处理 |
| RNN/LSTM/GRU | 循环、时序建模 | 序列数据 |
| Transformer | 自注意力、并行 | NLP、多模态 |

### 5.3 残差连接

**残差网络（ResNet）** 通过跳跃连接缓解梯度消失：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)  # 残差连接

---

## 本章小结

神经网络是现代深度学习的基石模型。关键要点：

1. **感知机**是最简单的线性分类器，但无法处理非线性问题
2. **多层感知机**通过隐藏层引入非线性，可以逼近任意连续函数
3. **激活函数**（ReLU、GELU、SwiGLU）是引入非线性的关键
4. **损失函数**（交叉熵、MSE）定义了优化目标
5. **残差连接**解决了深层网络的梯度消失问题

## 深度分析

神经网络是连接数学理论与实际 AI 系统的桥梁。从机器学习的角度看，神经网络可被视为一个可微分的函数逼近器——万能近似定理保证了足够宽的单隐藏层网络可以逼近任意连续函数，但现代深度学习更倾向于深而窄的结构，因为深层网络能以更少的参数实现更高的表示效率。LLM 中的 Feed-Forward Network 层本质上是两层 MLP，中间使用 GELU 或 SwiGLU 激活函数。

激活函数的演进反映了深度学习对训练稳定性的追求。早期 Sigmoid/Tanh 因梯度饱和问题被 ReLU 取代，后者因 Dying ReLU 问题催生了 LeakyReLU、PReLU 等变体，最终 GELU 和 SwiGLU 成为 Transformer 架构的标准选择。理解这些激活函数的导数特性（如 GELU 近似标准正态 CDF 的平滑性质）对于把握 LLM 训练中的梯度行为和数据分布变化至关重要。

## 核心概念检查

- [ ] 你能解释万能近似定理的数学表述及其局限性？
- [ ] 你能说明为什么多层线性层叠加等价于单层线性层？
- [ ] 你能分析 ReLU、GELU 和 SwiGLU 激活函数的梯度特性差异？
- [ ] 你能推导交叉熵损失函数关于 Softmax 输入的梯度？
- [ ] 你能解释残差连接 $y=x+F(x)$ 为何能缓解梯度消失？
- [ ] 你能比较 Sigmoid 和 Tanh 的梯度饱和区间与网络深度的影响？
- [ ] 你能设计一个 MLP，手动计算其单次前向和反向传播？
- [ ] 你能说明嵌入层（Embedding）为什么可以视为特殊的全连接层？
- [ ] 你能衡量模型宽度与深度对表示能力的不同影响？
- [ ] 你能解释 Dropout 在训练和推理时的缩放策略及其数学推导？

## 延伸阅读

- [第一章：线性代数](./ch01-linear-algebra.md) - 神经网络计算的矩阵基础
- [第二章：微积分](./ch02-calculus.md) - 神经网络中的链式法则
- [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md) - 深度网络训练技术
- [第八章：注意力机制](./ch08-attention-mechanism.md) - 从全连接到注意力

**最后更新**: 2026-06-12
```

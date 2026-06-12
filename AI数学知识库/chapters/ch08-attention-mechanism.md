# 第八章：注意力机制

> 注意力机制（Attention Mechanism）是深度学习领域的革命性突破，2017年Google在论文《Attention Is All You Need》中提出了完全基于注意力机制的Transformer架构，彻底改变了自然语言处理乃至整个人工智能领域。本章将深入讲解注意力机制的数学原理，包括**Scaled Dot-Product Attention**、**Multi-Head Attention**，以及在大模型中的应用。

## 目录

1. [注意力机制概述](#1-注意力机制概述)
2. [Scaled Dot-Product Attention](#2-scaled-dot-product-attention)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [注意力机制变体](#4-注意力机制变体)
5. [注意力机制的计算复杂度](#5-注意力机制的计算复杂度)

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [第一章：线性代数](./ch01-linear-algebra.md), [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md)
- **关联文件**: [第九章：Transformer架构](./ch09-transformer.md), [第六章：神经网络基础](./ch06-neural-networks.md)
- **最后更新**: 2026-06-12
---

## 1. 注意力机制概述

### 1.1 从人类注意力到机器注意力

**人类注意力的特点：**
- 选择性：关注重要信息，忽略无关信息
- 动态性：根据任务需求调整关注点
- 资源有限：一次只能关注有限的信息

**机器注意力的目标：**
- 建立序列中不同位置之间的依赖关系
- 捕捉长距离依赖（Long-range Dependencies）
- 动态调整信息权重

### 1.2 注意力机制的形式化定义

**注意力机制的本质：** 对输入的不同部分加权求和。

给定查询（Query）和键值对（Key-Value）：
$$
\text{Attention}(Q, K, V) = \sum_{i=1}^{n} \alpha_i \cdot v_i
$$

其中权重 $\alpha_i$ 由查询和键计算得到：
$$
\alpha_i = \text{softmax}\left(\frac{q \cdot k_i}{\sqrt{d_k}}\right)
$$

### 1.3 为什么需要注意力机制？

**传统RNN的问题：**
- 信息需要经过多个时间步传递
- 梯度消失/爆炸问题
- 难以捕捉长距离依赖

**注意力机制的优势：**
- 直接建立任意位置之间的联系
- 并行计算，效率高
- 可解释性强（可视化注意力权重）

---

## 2. Scaled Dot-Product Attention

### 2.1 核心公式

**Scaled Dot-Product Attention** 是Transformer中使用的注意力机制：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**各部分含义：**
- $Q \in \mathbb{R}^{L \times d_k}$：查询矩阵（L个查询向量）
- $K \in \mathbb{R}^{L \times d_k}$：键矩阵（L个键向量）
- $V \in \mathbb{R}^{L \times d_v}$：值矩阵（L个值向量）
- $\sqrt{d_k}$：缩放因子

### 2.2 计算过程详解

**步骤1：计算注意力分数**
$$
S = QK^T \in \mathbb{R}^{L \times L}
$$

**步骤2：缩放**
$$
S_{\text{scaled}} = \frac{S}{\sqrt{d_k}}
$$

**步骤3：Softmax归一化**
$$
\alpha = \text{softmax}(S_{\text{scaled}}) \in \mathbb{R}^{L \times L}
$$

**步骤4：加权求和**
$$
\text{Attention} = \alpha V \in \mathbb{R}^{L \times d_v}
$$

### 2.3 缩放因子的必要性

**问题：** 当 $d_k$ 很大时，点积的值可能变得很大，导致Softmax进入饱和区域。

**分析：**
假设 $q$ 和 $k$ 的各分量是独立的随机变量，均值为0，方差为1，则：
$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

由于各分量独立：
- $E[q \cdot k] = 0$
- $\text{Var}(q \cdot k) = d_k$

**因此：** 点积的方差与 $d_k$ 成正比，当 $d_k$ 较大时，点积可能变得很大。

**解决方案：** 除以 $\sqrt{d_k}$ 使方差回到1。

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    Q: [batch, num_heads, seq_len_q, d_k]
    K: [batch, num_heads, seq_len_k, d_k]
    V: [batch, num_heads, seq_len_v, d_v]
    """
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 2.4 掩码机制

**Padding Mask：** 处理不同长度的序列
```python
def create_padding_mask(seq, pad_idx=0):
    """创建padding掩码"""
    # seq: [batch, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

**Causal Mask（未来信息掩码）：** 防止看到未来信息
```python
def create_causal_mask(seq_len):
    """创建因果掩码"""
    # 上三角为False（下限为-inf）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

---

## 3. Multi-Head Attention

### 3.1 核心思想

**Multi-Head Attention** 让模型同时关注不同位置的不同表示子空间：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

其中每个头：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 3.2 完整数学表达式

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

其中投影矩阵的维度：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

### 3.3 为什么使用多头注意力？

**直观理解：** 不同的头可以关注不同类型的信息

| 头类型 | 关注的信息 |
|--------|------------|
| 句法头 | 句法依赖关系 |
| 语义头 | 语义相似性 |
| 位置头 | 位置关系 |
| 任务头 | 任务相关信息 |

### 3.4 完整实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性投影
        Q = self.W_Q(Q)  # [B, L, D]
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 分头
        # [B, L, D] -> [B, L, num_heads, d_k] -> [B, num_heads, L, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)  # [B, num_heads, L, d_k]
        
        # 合并多头
        # [B, num_heads, L, d_k] -> [B, L, num_heads, d_k] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = self.W_O(context)
        
        return output, attention_weights
```

### 3.5 不同类型的注意力

| 类型 | 说明 | 应用 |
|------|------|------|
| **Self-Attention** | Q=K=V | Transformer编码器、解码器 |
| **Cross-Attention** | Q≠K=V | 解码器中看编码器输出 |
| **Causal Attention** | 带掩码的Self | GPT等语言模型 |
| **Bi-directional** | 双向注意力 | BERT |

```python
# Self-Attention（编码器）
encoder_output = multi_head_attention(x, x, x, mask)  # Q=K=V=x

# Cross-Attention（解码器）
decoder_output = multi_head_attention(
    decoder_hidden,      # Q: 解码器状态
    encoder_output,      # K, V: 编码器输出
    encoder_output,
    mask
)
```

---

## 4. 注意力机制变体

### 4.1 线性注意力（Linear Attention）

**问题：** 标准注意力的复杂度是 $O(L^2)$，当序列很长时计算量巨大。

**线性注意力** 通过核方法将复杂度降为 $O(L)$：

$$
\text{Attention}(x_i, x_j) = \frac{\phi(x_i)^T \phi(x_j)}{\sum_j \phi(x_i)^T \phi(x_j)}
$$

### 4.2 Flash Attention

**核心思想：** 利用GPU的SRAM加速注意力计算，减少HBM访问。

**关键技术：**
- 分块计算（Tile）
- 融合核（Fused Kernel）
- 在线Softmax

```python
# 使用Flash Attention
from flash_attn import flash_attn_func

# Q, K, V: [B, S, H, D]
output = flash_attn_func(
    Q, K, V,
    causal=True  # 是否应用因果掩码
)
```

### 4.3 Sparse Attention

**核心思想：** 只计算部分注意力连接，而非全连接。

**常见模式：**
- Global + Local：某些位置与所有位置相连，其他只与局部相连
- Random：随机连接
- Stride：固定间隔连接

---

## 5. 注意力机制的计算复杂度

### 5.1 复杂度分析

| 操作 | 复杂度 |
|------|--------|
| $QK^T$ | $O(L^2 \cdot d)$ |
| Softmax | $O(L^2)$ |
| $\alpha V$ | $O(L^2 \cdot d)$ |
| **总计** | $O(L^2 \cdot d)$ |

### 5.2 长序列问题的解决方案

| 方法 | 复杂度 | 特点 |
|------|--------|------|
| 标准Attention | $O(L^2)$ | 完整但慢 |
| Flash Attention | $O(L^2)$ 但内存优化 | 工业级实现 |
| Linear Attention | $O(L)$ | 近似，快速 |
| Longformer | $O(L)$ | 稀疏+局部+全局 |
| BigBird | $O(L)$ | 稀疏组合 |
| Performer | $O(L)$ | 随机特征近似 |

---

## 本章小结

注意力机制是现代大模型的核心：

1. **Scaled Dot-Product Attention** 是Transformer的基础
2. **Multi-Head Attention** 让模型关注不同表示空间
3. **掩码机制** 控制信息流动（因果、padding）
4. **Flash Attention** 解决了长序列的计算效率问题

## 深度分析

注意力机制是 Transformer 架构的核心创新，其数学形式 $\text{softmax}(QK^T/\sqrt{d_k})V$ 优雅地解决了序列建模中的长距离依赖问题。与 RNN 的串行计算不同，注意力层可以并行计算所有位置的交互，这使得 GPU 的矩阵计算能力得到充分发挥。缩放因子 $\sqrt{d_k}$ 的设置体现了数值稳定性的考量——当 $d_k$ 较大时，点积的方差随维度线性增长，不缩放的话 Softmax 会进入梯度极小的饱和区。

Multi-Head Attention 的设计让模型能够从不同的表示子空间中捕捉信息——某些头关注语法依赖，某些头关注语义相似性，还有一些头关注位置关系。这种并行子空间的学习机制直接影响了 MoE（Mixture of Experts）等后续架构。Flash Attention 通过分块计算和在线 Softmax 将注意力计算的内存复杂度从 $O(L^2)$ 降低到 $O(L)$，使得处理 128K+ 长上下文成为可能，是当前 LLM 推理优化的核心技术之一。

## 核心概念检查

- [ ] 你能推导 Scaled Dot-Product Attention 的完整前向计算过程？
- [ ] 你能解释为什么缩放因子 $\sqrt{d_k}$ 对注意力梯度的数值稳定至关重要？
- [ ] 你能说明 Multi-Head Attention 中分头和合并的矩阵变换过程？
- [ ] 你能分析 Self-Attention 的计算复杂度 $O(L^2 \cdot d)$ 的来源？
- [ ] 你能解释 Causal Mask 和 Padding Mask 在注意力计算中的实现？
- [ ] 你能说明 Flash Attention 通过分块（Tiling）和在线 Softmax 减少 HBM 访问的原理？
- [ ] 你能比较标准 Attention、Linear Attention 和 Sparse Attention 的复杂度与适用场景？
- [ ] 你能描述 Cross-Attention 与 Self-Attention 在 Query/Key/Value 来源上的区别？
- [ ] 你能分析注意力头数 $h$ 与每头维度 $d_k$ 的权衡对模型容量的影响？
- [ ] 你能解释 Grouped Query Attention（GQA）相比标准 Multi-Head Attention 在推理时的 KV Cache 优势？

## 延伸阅读

- [第九章：Transformer架构](./ch09-transformer.md) - 注意力机制在 Transformer 中的完整应用
- [第一章：线性代数](./ch01-linear-algebra.md) - 注意力计算中的矩阵运算
- [第七章：深度学习关键技术](./ch07-deep-learning-techniques.md) - Transformer Block 中的关键技术组合
- [第三章：概率论与统计学](./ch03-probability.md) - Softmax 的概率解释

**最后更新**: 2026-06-12

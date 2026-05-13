# 第九章：Transformer架构与位置编码

> Transformer架构是现代大语言模型的基础。本章将深入讲解Transformer的完整结构，包括**位置编码**、**编码器**、**解码器**，以及主流模型架构（GPT、BERT、LLaMA等）的特点。

## 目录

1. [Transformer整体架构](#1-transformer整体架构)
2. [位置编码](#2-位置编码)
3. [Transformer编码器](#3-transformer编码器)
4. [Transformer解码器](#4-transformer解码器)
5. [主流模型架构对比](#5-主流模型架构对比)

---

## 1. Transformer整体架构

### 1.1 经典Transformer结构

```
                    Transformer架构
    
    输入                    编码器栈                    解码器栈                    输出
    ┌────┐          ┌────────────────┐         ┌────────────────┐          ┌────┐
    │    │          │                │         │                │          │    │
x ──┼──► │Embedding │                │         │                │          │    │
    │    │          │   ┌────────┐   │         │   ┌────────┐   │          │    │
    │    │          │   │Self-   │   │         │   │Self-   │   │          │    │
    │    │          │   │Attention│   │         │   │Attention│   │          │    │
    │    │          │   └────────┘   │         │   └────────┘   │          │    │
    │    │          │       ↓         │         │       ↓         │          │    │
    │    │          │   ┌────────┐   │         │   ┌────────┐   │          │    │
    │    │          │   │  FFN   │   │         │   │Cross-  │   │          │    │
    │    │          │   └────────┘   │         │   │Attention│   │          │    │
    │    │          │       ↓         │         │   └────────┘   │          │    │
    │    │          │  ┌────────┐   │         │       ↓         │          │    │
    │    │          │  │  FFN   │   │         │   ┌────────┐   │          │    │
    │    │          │  └────────┘   │         │   │  FFN   │   │          │    │
    │    │          │       ↓         │         │   └────────┘   │          │    │
    │    │          └────────────────┘         │       ↓         │          │    │
    │    │                                       └────────────────┘          │    │
    │    │                                                  │                │    │
    └────┼──────────────────────────────────────────────────┘                │    │
         │                                                       Output      │    │
    Input                                                          └────────►│Embedding│──► Output
    Tokens                                                                   └────┘
    
    N× Encoder层                    N× Decoder层
```

### 1.2 编码器与解码器的区别

| 组件 | 编码器 | 解码器 |
|------|--------|--------|
| Self-Attention | 双向 | 因果（Causal） |
| Cross-Attention | 无 | 有（看编码器输出） |
| 掩码 | Padding Mask | Padding + Causall Mask |
| 用途 | 理解输入 | 生成输出 |

---

## 2. 位置编码

### 2.1 为什么需要位置编码？

**问题：** 自注意力机制是位置无关的（Permutation Invariant）

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

输入的词序变化不会影响输出——但语言中词序很重要！

**解决方案：** 在输入中加入位置信息

$$
x'_i = x_i + p_i
$$

### 2.2 绝对位置编码（正弦/余弦）

**原始Transformer论文中的位置编码：**

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**特点：**
- 不同频率的正弦波组合
- 每个位置有唯一的编码
- 可以推广到训练时未见过的更长序列

```python
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # 计算频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 奇偶位置分别使用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### 2.3 旋转位置编码（RoPE）

**RoPE** 是LLaMA等模型使用的新型位置编码。

**核心思想：** 通过旋转操作将位置信息注入到Query和Key向量中。

**二维情况（简化版）：**
$$
\begin{bmatrix} q'_x \\ q'_y \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_x \\ q_y \end{bmatrix}
$$

**广义RoPE（多维）：**
$$
f_q(x_m, m) = W_q x_m \cdot R^m_{\Theta}
$$
$$
f_k(x_n, n) = W_k x_n \cdot R^n_{\Theta}
$$

其中 $R^m_{\Theta}$ 是旋转矩阵。

**优势：**
- 可以处理任意长度的序列
- 计算高效
- 在LLaMA、GLM等模型中广泛使用

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    应用旋转位置编码
    
    q, k: [batch, num_heads, seq_len, d_k]
    cos, sin: [batch, seq_len, d_k]
    """
    # 分离奇偶维度
    q1 = q[..., 0::2]
    q2 = q[..., 1::2]
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]
    
    # 旋转操作
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    return q_rot, k_rot
```

### 2.4 相对位置编码

**核心思想：** 编码 token 之间的相对位置，而非绝对位置。

**代表工作：** Shaw et al., 2018; T5; DeBERTa

$$
\text{Attention}_{\text{rel}} = \text{softmax}\left(\frac{QK^T + S_{\text{rel}}}{\sqrt{d}}\right)V
$$

其中 $S_{\text{rel}}$ 是相对位置偏置矩阵。

### 2.5 位置编码对比

| 类型 | 代表模型 | 特点 |
|------|----------|------|
| **绝对正弦** | 原始Transformer | 简单，但无法外推 |
| **可学习绝对** | BERT | 需指定最大长度 |
| **旋转（RoPE）** | LLaMA, GLM | 可处理任意长度 |
| **相对** | T5, DeBERTa | 更灵活 |

---

## 3. Transformer编码器

### 3.1 编码器层结构

```
    编码器层
    
    输入 x
       │
       ▼
    ┌─────────────────┐
    │ LayerNorm       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Multi-Head      │
    │ Self-Attention  │───→ 注意力权重
    └────────┬────────┘
             │
             ▼
         x + attention(x)  (残差连接)
             │
             ▼
    ┌─────────────────┐
    │ LayerNorm       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feed-Forward    │
    │ Network (FFN)    │
    └────────┬────────┘
             │
             ▼
         x + ffn(x)    (残差连接)
             │
             ▼
         输出
```

### 3.2 前馈网络（FFN）

FFN是Transformer中参数最多的部分：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**维度配置：**
- BERT-base: $d_{\text{model}} = 768$, $d_{ff} = 3072$
- BERT-large: $d_{\text{model}} = 1024$, $d_{ff} = 4096$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### 3.3 Pre-LN vs Post-LN

**Post-LN（原始Transformer）：**
```python
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Pre-LN（常用）：**
```python
x = x + MultiHeadAttention(x)
x = x + FFN(x)
x = LayerNorm(x)  # 只在最后做一次LayerNorm
```

**Pre-LN的优势：** 训练更稳定，是目前的主流选择。

---

## 4. Transformer解码器

### 4.1 解码器层结构

```
    解码器层
    
    输入 y（已经生成的部分）
       │
       ▼
    ┌─────────────────┐
    │ LayerNorm       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Masked          │
    │ Self-Attention  │───→ 只能看到当前位置及之前
    └────────┬────────┘
             │
             ▼
         y + masked_attn(y)
             │
             ▼
    ┌─────────────────┐
    │ LayerNorm       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Cross-Attention │←─── 编码器输出
    └────────┬────────┘
             │
             ▼
         y + cross_attn(y, encoder_output)
             │
             ▼
    ┌─────────────────┐
    │ LayerNorm       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feed-Forward    │
    └────────┬────────┘
             │
             ▼
         y + FFN(y)
             │
             ▼
         输出
```

### 4.2 因果掩码（Causal Mask）

确保生成时只能看到之前的内容：

```python
def create_causal_mask(seq_len, device):
    """
    创建因果掩码
    形状: [1, 1, seq_len, seq_len]
    """
    # 创建一个下三角形矩阵
    # 位置(i,j)，如果j>i则掩蔽
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), 
        diagonal=1
    ).bool()
    
    return mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度

# 示例
seq_len = 5
mask = create_causal_mask(seq_len, 'cpu')
print(mask.squeeze())
# tensor([[ True,  True,  True,  True,  True],
#         [False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True]])
# True表示需要掩蔽（不能看到）
```

---

## 5. 主流模型架构对比

### 5.1 GPT系列（Decoder-only）

**架构特点：**
- 只使用解码器
- 因果注意力（Causal Attention）
- 下一个词预测目标

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(torch.arange(x.size(1), device=x.device))
        
        causal_mask = create_causal_mask(x.size(1), x.device)
        
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)
        
        x = self.final_norm(x)
        return self.lm_head(x)
```

### 5.2 BERT（Encoder-only）

**架构特点：**
- 只使用编码器
- 双向注意力（Bidirectional Attention）
- MLM（Masked Language Model）+ NSP（Next Sentence Prediction）目标

### 5.3 LLaMA（改进的Decoder）

**关键改进：**
1. **RMSNorm** 替代 LayerNorm
2. **RoPE** 替代绝对位置编码
3. **SwiGLU** 激活函数替代 GELU
4. **分组查询注意力（GQA）**

```python
class LLaMAMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # SwiGLU激活
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### 5.4 模型架构对比表

| 模型 | 架构 | 位置编码 | 注意力 | 激活函数 |
|------|------|----------|--------|----------|
| GPT-2 | Decoder | 绝对 | GQA | GELU |
| GPT-3 | Decoder | 绝对 | GQA | GELU |
| BERT | Encoder | 可学习 | MHA | GELU |
| LLaMA | Decoder | RoPE | GQA | SwiGLU |
| LLaMA-2 | Decoder | RoPE | GQA | SwiGLU |
| ChatGLM | Decoder | RoPE | GQA | SwiGLU |

---

## 本章小结

Transformer架构的核心组件：

1. **位置编码** 解决序列顺序问题（绝对/相对/旋转）
2. **编码器** 理解输入（双向注意力）
3. **解码器** 生成输出（因果注意力+交叉注意力）
4. **残差连接+LayerNorm** 保证训练稳定性
5. **FFN** 提供非线性变换能力

**下一章：** 我们将学习**大模型训练与优化**，包括预训练、SFT、RLHF等关键技术。

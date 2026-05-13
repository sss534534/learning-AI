# Transformer架构详解

> 理解现代大模型的基石架构

## 1. 架构概览

Transformer架构由Google在2017年论文《Attention Is All You Need》中提出，彻底改变了NLP领域。其核心创新是**完全基于注意力机制**，摒弃了传统的RNN/CNN结构。

### 1.1 整体结构

```
输入 → [Embedding + Positional Encoding] → 
      [Encoder × N] → [Decoder × N] → 输出
```

**Encoder-Decoder架构特点：**
- **Encoder**: 将输入序列编码为上下文表示（双向注意力）
- **Decoder**: 基于编码器输出自回归生成序列（单向注意力+交叉注意力）

**典型模型演进：**
| 模型 | 架构 | 特点 |
|------|------|------|
| BERT | Encoder-only | 双向编码，适合理解任务 |
| GPT系列 | Decoder-only | 单向生成，适合生成任务 |
| T5 | Encoder-Decoder | 统一框架，适合翻译/摘要 |

### 1.2 核心组件

```
Transformer Block
├── Multi-Head Self-Attention
├── Add & Norm (残差连接 + LayerNorm)
├── Feed-Forward Network (FFN)
└── Add & Norm
```

---

## 2. 自注意力机制（Self-Attention）

### 2.1 核心思想

> 每个token都能"看到"序列中所有其他token，并根据相关性加权聚合信息

**类比理解：**
- 想象你在阅读一句话时，眼睛会自然地在关键词之间跳跃
- "它"这个词会特别关注前文提到的名词
- Self-Attention就是让这个"关注"过程可计算、可学习

### 2.2 Query-Key-Value机制

**数学表达：**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**三个向量的含义：**

| 向量 | 作用 | 类比 |
|------|------|------|
| **Query (Q)** | 当前token的"查询意图" | 你提出的问题 |
| **Key (K)** | 每个token的"身份标识" | 图书馆的索引卡片 |
| **Value (V)** | 每个token的"实际内容" | 书中的实际内容 |

**计算流程：**

1. **相似度计算**: Q × K^T → 得到注意力分数（点积）
2. **缩放**: 除以√d_k（防止softmax梯度消失）
3. **归一化**: softmax → 得到注意力权重（和为1）
4. **加权求和**: 权重 × V → 输出上下文向量

### 2.3 多头注意力（Multi-Head Attention）

**核心思想：** 使用多组Q-K-V，让模型从不同"角度"理解关系

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**为什么需要多头？**

| Head | 关注的关系类型 | 示例 |
|------|----------------|------|
| Head 1 | 句法关系 | 主语-谓语 |
| Head 2 | 语义关系 | 同义词 |
| Head 3 | 指代关系 | 代词-先行词 |
| Head 4 | 位置关系 | 相邻词 |

**典型配置：**
- GPT-3: 96 heads, d_k = d_v = 128
- LLaMA-2: 32 heads, d_k = d_v = 128

---

## 3. 位置编码（Positional Encoding）

### 3.1 为什么需要位置编码？

> Self-Attention是**位置无关**的："我爱猫"和"猫爱我"计算出的注意力分数相同

**解决方案：** 为每个位置添加唯一的编码，让模型感知顺序

### 3.2 绝对位置编码

**原始Transformer使用正弦/余弦函数：**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**优点：**
- 无需学习参数
- 可外推到训练时未见过的长度
- 相对位置可表示（PE(pos+k)可由PE(pos)线性变换得到）

### 3.3 相对位置编码

**RoPE (Rotary Position Embedding)** - 现代大模型主流方案

**核心思想：** 通过旋转矩阵编码相对位置

```
R_Θ^d × x = 
[x_1, x_2, x_3, x_4, ...]  →  [x_1cosθ - x_2sinθ, x_1sinθ + x_2cosθ, ...]
```

**优势：**
- 更好的长文本外推能力
- 相对位置信息更明确
- 被LLaMA、ChatGLM、Qwen等主流模型采用

### 3.4 ALiBi (Attention with Linear Biases)

**核心思想：** 直接在注意力分数上添加距离惩罚

```
AttentionScore = QK^T/√d_k + m × [- (i-j)]
```

**优势：**
- 极强的长文本外推能力
- 无需修改位置编码
- BLOOM、MPT等模型使用

---

## 4. 前馈网络与归一化

### 4.1 Feed-Forward Network (FFN)

**结构：**
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**特点：**
- 两个线性变换夹一个ReLU/GELU激活
- 中间维度通常是输入的4倍（如d_model=512, d_ff=2048）
- **每个token独立计算**（无token间交互）

**现代变体 - SwiGLU：**
```
SwiGLU(x) = (xW_1 ⊗ Swish(xW_2))W_3
```
- 被LLaMA、PaLM等采用
- 性能更好但参数量增加

### 4.2 归一化层

**LayerNorm vs RMSNorm：**

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 公式 | (x - μ) / √(σ² + ε) | x / √(mean(x²) + ε) |
| 去均值 | 是 | 否 |
| 计算量 | 较大 | 较小 |
| 使用模型 | BERT、GPT-2 | LLaMA、Qwen |

**为什么用RMSNorm？**
- 去均值操作在深层网络中作用有限
- 减少计算量，加速训练
- 现代大模型普遍采用

### 4.3 残差连接（Residual Connection）

```
output = LayerNorm(x + Sublayer(x))
```

**作用：**
- 缓解梯度消失，支持深层网络（100+层）
- 保留原始信息，注意力层专注学习"差异"

---

## 5. 解码策略

### 5.1 贪心解码（Greedy Decoding）

**策略：** 每步选择概率最高的token

**优点：** 简单、快速
**缺点：** 容易陷入局部最优，生成内容单调

### 5.2 Beam Search

**策略：** 每步保留top-k个候选序列

**参数：**
- beam_width: 候选序列数量（通常4-10）
- length_penalty: 长度惩罚（避免过短/过长）

**适用场景：** 机器翻译、摘要（有明确参考答案的任务）

### 5.3 采样策略

**Temperature Sampling：**
```
p_i = exp(z_i/T) / Σexp(z_j/T)
```
- T→0: 接近贪心解码
- T→∞: 均匀分布
- **推荐值：** 0.7-1.0

**Top-k Sampling：**
- 只从概率最高的k个token中采样
- **推荐值：** k=40-50

**Top-p (Nucleus) Sampling：**
- 从累积概率达到p的最小集合中采样
- **推荐值：** p=0.9-0.95

**组合策略：**
```
Temperature=0.7 + Top-p=0.9  # 最常用配置
```

---

## 6. 模型规模与计算量

### 6.1 参数量估算

```
总参数量 ≈ 4 × d_model² × n_layers
```

**典型配置：**

| 模型 | d_model | n_layers | n_heads | 参数量 |
|------|---------|----------|---------|--------|
| GPT-2 | 768 | 12 | 12 | 117M |
| GPT-3 | 12288 | 96 | 96 | 175B |
| LLaMA-2-7B | 4096 | 32 | 32 | 7B |
| LLaMA-2-70B | 8192 | 80 | 64 | 70B |

### 6.2 计算复杂度

**Self-Attention复杂度：**
- 时间：O(n² × d) - 序列长度的平方
- 空间：O(n²) - 注意力矩阵存储

**优化方向：**
- 稀疏注意力（Sparse Attention）
- 线性注意力（Linear Attention）
- FlashAttention（IO优化）

---

## 7. 架构师关注点

### 7.1 设计决策清单

**选择模型架构时考虑：**

- [ ] 任务类型：理解(Encoder) vs 生成(Decoder) vs 两者兼顾
- [ ] 序列长度：是否需要长文本支持（>4K tokens）
- [ ] 推理延迟：实时场景需要优化注意力计算
- [ ] 显存预算：模型规模与部署成本的平衡

### 7.2 关键指标

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| FLOPs | 浮点运算次数 | 模型压缩、量化 |
| Memory Bandwidth | 显存带宽瓶颈 | FlashAttention、KV Cache优化 |
| Latency | 首Token延迟 | 并行解码、投机解码 |
| Throughput | 吞吐量 | Continuous Batching |

### 7.3 常见陷阱

1. **忽视位置编码的影响**
   - 长文本外推能力取决于位置编码设计
   - 直接外推可能导致性能急剧下降

2. **注意力熵崩溃**
   - 深层网络注意力趋于均匀分布
   - 解决方案：LayerNorm前置、残差缩放

3. **KV Cache显存爆炸**
   - 长序列生成时显存线性增长
   - 解决方案：PagedAttention、量化缓存

---

## 8. 延伸阅读

### 必读论文
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
2. [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE位置编码
3. [LLaMA 2](https://arxiv.org/abs/2307.09288) - 开源大模型架构细节

### 实践资源
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)

---

*最后更新：2026-05-07*

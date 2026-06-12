# 第七章：深度学习关键技术

> 本章深入讲解深度学习中的关键技术，包括**正则化方法**、**归一化技术**、**注意力机制**（基础版）、**残差连接**等。这些技术是构建现代大型语言模型的基础组件。

## 目录

1. [正则化技术](#1-正则化技术)
2. [归一化技术](#2-归一化技术)
3. [深度学习关键技术综合应用](#3-深度学习关键技术综合应用)

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [第六章：神经网络基础](./ch06-neural-networks.md), [第二章：微积分](./ch02-calculus.md)
- **关联文件**: [第八章：注意力机制](./ch08-attention-mechanism.md), [第九章：Transformer架构](./ch09-transformer.md)
- **最后更新**: 2026-06-12
---

## 1. 正则化技术

### 1.1 正则化的目的

**正则化** 是防止过拟合、提高模型泛化能力的技术。

**过拟合的表现：**
- 训练损失持续下降，验证损失上升
- 模型在训练集上表现很好，在测试集上表现差

### 1.2 L1和L2正则化

**L2正则化（权重衰减）：**
$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_0 + \frac{\lambda}{2} \sum_w w^2
$$

**L1正则化：**
$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_0 + \lambda \sum_w |w|
$$

```python
# PyTorch实现
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2正则化
)
```

### 1.3 Dropout

**Dropout** 在训练时随机丢弃神经元：

```python
class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, training=True):
        if not training:
            return x
        
        # 随机创建掩码
        mask = torch.rand_like(x) > self.p
        return x * mask / (1 - self.p)  # 缩放保持期望

# 使用
self.dropout = nn.Dropout(p=0.1)
x = self.dropout(x, training=self.training)
```

### 1.4 Dropout的数学原理

**训练时：** 每个神经元以概率 $p$ 被保留
$$
\hat{y} = \frac{1}{1-p} \cdot \mathbf{w}^T \cdot (\mathbf{x} \odot \mathbf{m})
$$
其中 $\mathbf{m}$ 是伯努利掩码。

**测试时：** 使用完整网络，但权重乘以 $(1-p)$
$$
\hat{y}_{\text{test}} = (1-p) \cdot \mathbf{w}^T \mathbf{x}
$$

### 1.5 Dropout变体

| 方法 | 特点 | 应用场景 |
|------|------|----------|
| **Standard Dropout** | 随机丢弃 | 通用 |
| **Dropout** | 丢弃整个通道 | CNN |
| **Spatial Dropout** | 丢弃整个特征图 | CNN |
| **DropBlock** | 丢弃连续区域 | CNN |
| **Variational Dropout** | 自适应丢弃率 | RNN/LSTM |
| **Dropout** | 丢弃注意力头 | Transformer |

---

## 2. 归一化技术

### 2.1 归一化的必要性

**内部协变量偏移（Internal Covariate Shift）：**
- 每层输入的分布随前层参数变化而变化
- 导致训练困难，需要小的学习率

**归一化的作用：**
- 稳定输入分布
- 加速收敛
- 允许更大的学习率

### 2.2 Batch Normalization

**核心思想：** 对每个batch的每个特征维度分别归一化。

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i \quad \text{（均值）}
$$
$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2 \quad \text{（方差）}
$$
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \text{（归一化）}
$$
$$
y_i = \gamma \hat{x}_i + \beta \quad \text{（仿射变换）}
$$

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新running statistics
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 2.3 Layer Normalization

**核心思想：** 对每个样本的所有特征归一化。

$$
\mu_i = \frac{1}{H} \sum_{j=1}^{H} x_{ij} \quad \text{（均值）}
$$
$$
\sigma_i^2 = \frac{1}{H} \sum_{j=1}^{H} (x_{ij} - \mu_i)^2 \quad \text{（方差）}
$$
$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$
$$
y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j
$$

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        mean = x.mean(dim=-1, keepdim=True)  # 沿最后一维
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 2.4 BatchNorm vs LayerNorm

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch维度 | 特征维度 |
| batch依赖性 | 强（训练/推理不同） | 无 |
| RNN中使用 | 不适合 | 适合 |
| Transformer中使用 | 不适合 | 适合 |
| 小batch效果 | 差 | 好 |

```
    BatchNorm vs LayerNorm 可视化
    
    BatchNorm (2D input [N, C]):
    
    ┌────────────────────────────────────┐
    │  batch 0: [x₁₁, x₁₂, x₁₃]          │
    │  batch 1: [x₂₁, x₂₂, x₂₃]          │
    │  batch 2: [x₃₁, x₃₂, x₃₃]          │
    └────────────────────────────────────┘
    
    归一化: 对每个通道（列）单独归一化
    ┌────────────────────────────────────┐
    │  Channel 0: normalize(x₁₁,x₂₁,x₃₁) │
    │  Channel 1: normalize(x₁₂,x₂₂,x₃₂) │
    │  Channel 2: normalize(x₁₃,x₂₃,x₃₃) │
    └────────────────────────────────────┘
    
    LayerNorm (2D input [N, C]):
    
    ┌────────────────────────────────────┐
    │  Sample 0: normalize(x₁₁,x₁₂,x₁₃) │
    │  Sample 1: normalize(x₂₁,x₂₂,x₂₃) │
    │  Sample 2: normalize(x₃₁,x₃₂,x₃₃) │
    └────────────────────────────────────┘
    
    归一化: 对每个样本（行）单独归一化
```

### 2.5 RMSNorm

**核心思想：** 只使用RMS统计量，忽略均值。

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H} \sum_{j=1}^{H} x_j^2}
$$

**优势：** 计算更快，效果与LayerNorm相当。

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
```

### 2.6 归一化方法对比

| 方法 | 归一化维度 | batch依赖 | 应用场景 |
|------|-----------|-----------|----------|
| BatchNorm | (N,) | 是 | CNN、图像 |
| LayerNorm | (C,) | 否 | RNN、Transformer |
| InstanceNorm | (N, H, W) | 否 | 风格迁移 |
| GroupNorm | (N, C/g, H, W) | 否 | 小batch CNN |
| RMSNorm | (C,) | 否 | 高效Transformer |

---

## 3. 深度学习关键技术综合应用

### 3.1 Transformer Block实现

以下是BERT/GPT中使用的完整Transformer Block：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LN 架构（更稳定）
        # 残差连接在归一化之前
        
        # 自注意力 + 残差
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

### 3.2 完整编码器实现

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # 输出归一化
        self.final_norm = LayerNorm(d_model)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入 + 位置嵌入
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # 通过所有Transformer块
        for block in self.blocks:
            x = block(x)
        
        return self.final_norm(x)
```

### 3.3 关键技术总结

| 技术 | 作用 | 位置 |
|------|------|------|
| **词嵌入** | 离散到连续的映射 | 输入层 |
| **位置编码** | 注入序列位置信息 | 输入层 |
| **LayerNorm** | 稳定训练 | 每个子层后 |
| **残差连接** | 缓解梯度问题 | 每个子层 |
| **GELU** | 非线性激活 | FFN层 |
| **Dropout** | 正则化 | 多个位置 |

---

## 本章小结

深度学习的关键技术包括：

1. **正则化方法**（L1/L2、Dropout）防止过拟合
2. **归一化技术**（BatchNorm、LayerNorm、RMSNorm）稳定训练
3. **残差连接** 缓解深层网络的梯度问题
4. 这些技术组合使用，构成了现代Transformer架构的基础

## 深度分析

归一化技术是深层神经网络训练的基石。在 LLM 中，LayerNorm 是 Transformer 的标准配置——它对每个 Token 的隐藏状态独立归一化，不受 Batch Size 影响，因此非常适合序列建模。值得注意的是，LLaMA 等新一代模型采用 RMSNorm（移除均值计算）替代 LayerNorm，在保持效果的同时降低了约 15% 的归一化计算量。Pre-LN（在子层之前归一化）架构相比 Post-LN（原始 Transformer 设计）显著提升了训练稳定性，是目前的主流选择。

正则化技术在大模型中有新的演化。Dropout 在 GPT-3 等早期大模型中仍有使用，但在 LLaMA 等最新模型中已基本被移除，因为足够大规模的数据和训练本身就具有正则化效果。残差连接与归一化的协同工作使得训练 100 层以上的深度 Transformer 成为可能——梯度可以通过残差捷径直接回传，归一化则保证了每一层的激活值在合理范围内。理解这些技术的相互作用是设计稳定训练配置的关键。

## 核心概念检查

- [ ] 你能比较 BatchNorm、LayerNorm 和 RMSNorm 的数学公式及适用场景？
- [ ] 你能解释为什么 Transformer 使用 LayerNorm 而非 BatchNorm？
- [ ] 你能推导 Pre-LN 和 Post-LN 架构中梯度的流动路径差异？
- [ ] 你能说明 Dropout 在训练和推理时的行为差异及其数学期望？
- [ ] 你能分析残差连接 $y=x+F(x)$ 在反向传播中创建梯度捷径的原理？
- [ ] 你能解释 RMSNorm 移除均值计算对梯度的影响？
- [ ] 你能说明权重衰减（Weight Decay）在大模型预训练中的作用？
- [ ] 你能描述完整的 Transformer Block 中各组件（LayerNorm、Attention、FFN、残差）的排列顺序？
- [ ] 你能比较 Label Smoothing 与 Dropout 两种正则化策略的区别？
- [ ] 你能分析为什么最新的 LLM 倾向于减少或移除 Dropout？

## 延伸阅读

- [第六章：神经网络基础](./ch06-neural-networks.md) - 神经网络组件基础
- [第八章：注意力机制](./ch08-attention-mechanism.md) - Transformer 中的注意力
- [第九章：Transformer架构](./ch09-transformer.md) - Pre-LN 与 Post-LN 架构
- [第五章：数值优化](./ch05-optimization.md) - 正则化与训练优化

**最后更新**: 2026-06-12

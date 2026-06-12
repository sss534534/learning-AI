# 第一章：预训练技术

> 预训练是构建大语言模型的第一阶段，决定了模型的基础能力。本章将深入讲解预训练的数据准备、训练目标、分布式训练策略以及显存优化技术。

## 目录

1. [预训练概述](#1-预训练概述)
2. [数据准备与处理](#2-数据准备与处理)
3. [训练目标与损失函数](#3-训练目标与损失函数)
4. [分布式训练策略](#4-分布式训练策略)
5. [显存优化技术](#5-显存优化技术)

---

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: 无
- **关联文件**: ../chapters/ch02-finetuning-alignment.md, ../appendices/appendix-a-glossary.md
- **最后更新**: 2026-06-12
---

## 1. 预训练概述

### 1.1 预训练的目标

预训练的核心目标是让模型学习通用的语言表示能力：

- **语言理解**：理解词汇、语法、语义
- **知识获取**：从海量文本中学习事实知识
- **推理能力**：学习基本的逻辑和推理模式
- **生成能力**：学会流畅地生成文本

### 1.2 预训练 vs 微调

| 阶段 | 数据规模 | 计算资源 | 训练目标 | 输出 |
|------|----------|----------|----------|------|
| **预训练** | 万亿级tokens | 数千GPU | 通用语言能力 | Base模型 |
| **微调** | 百万级样本 | 数十GPU | 特定任务能力 | 专用模型 |

### 1.3 预训练流程概览

```
原始语料
    │
    ▼
┌─────────────┐
│  数据清洗    │ → 去重、过滤、格式化
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Tokenization│ → BPE、WordPiece、SentencePiece
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   预训练     │ → 自回归/自编码训练
└──────┬──────┘
       │
       ▼
   Base模型
```

---

## 2. 数据准备与处理

### 2.1 数据来源

| 数据类型 | 占比 | 特点 | 代表数据集 |
|----------|------|------|------------|
| **网页数据** | 60-80% | 量大、质量参差 | Common Crawl, C4 |
| **书籍** | 10-20% | 长文本、连贯 | BooksCorpus, Gutenberg |
| **代码** | 10-20% | 结构化、逻辑性强 | GitHub, StackOverflow |
| **学术论文** | 5-10% | 专业化 | arXiv, PubMed |
| **对话数据** | 5-10% | 交互性 | Reddit, Wikipedia Talk |

### 2.2 数据清洗流程

```python
import re
import hashlib
from typing import List, Dict

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.min_length = 100  # 最小长度
        self.max_length = 100000  # 最大长度
        self.min_words = 50  # 最小词数
        
    def clean_text(self, text: str) -> str:
        """清洗单条文本"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 去除控制字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def filter_quality(self, text: str) -> bool:
        """质量过滤"""
        # 长度检查
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # 词数检查
        words = text.split()
        if len(words) < self.min_words:
            return False
        
        # 重复度检查
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # 重复度过高
            return False
        
        # 符号比例检查
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if symbol_ratio > 0.5:
            return False
        
        return True
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """去重（基于MinHash）"""
        seen_hashes = set()
        unique_texts = []
        
        for text in texts:
            # 计算文本指纹
            text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        return unique_texts

# 使用示例
cleaner = DataCleaner()
raw_texts = [...]  # 原始文本

cleaned_texts = [cleaner.clean_text(t) for t in raw_texts]
filtered_texts = [t for t in cleaned_texts if cleaner.filter_quality(t)]
unique_texts = cleaner.deduplicate(filtered_texts)
```

### 2.3 Tokenization

#### 2.3.1 BPE（Byte Pair Encoding）

BPE通过合并高频子词对来构建词表：

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 创建BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 设置预分词器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 训练
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
)

# 从文件训练
files = ["train.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("tokenizer.json")
```

#### 2.3.2 SentencePiece

SentencePiece将文本视为字节序列，无需预分词：

```python
import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='train.txt',
    model_prefix='spm',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'
)

# 加载
sp = spm.SentencePieceProcessor()
sp.load('spm.model')

# 编码/解码
tokens = sp.encode("Hello World", out_type=str)
text = sp.decode(tokens)
```

### 2.4 数据混合策略

```python
class DataMixer:
    """数据混合器"""
    
    def __init__(self, data_sources: Dict[str, float]):
        """
        data_sources: {数据源名称: 采样比例}
        例如: {"web": 0.6, "books": 0.2, "code": 0.2}
        """
        self.data_sources = data_sources
        
    def create_mixed_dataset(self, datasets: Dict[str, List]):
        """创建混合数据集"""
        mixed_data = []
        
        for source_name, ratio in self.data_sources.items():
            if source_name in datasets:
                data = datasets[source_name]
                # 按比例采样
                sample_size = int(len(data) * ratio / sum(self.data_sources.values()))
                sampled = random.sample(data, min(sample_size, len(data)))
                mixed_data.extend(sampled)
        
        # 随机打乱
        random.shuffle(mixed_data)
        
        return mixed_data
```

---

## 3. 训练目标与损失函数

### 3.1 自回归语言建模（Causal LM）

**目标：** 预测下一个词

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})
$$

```python
import torch
import torch.nn as nn

class CausalLMTrainer:
    """自回归语言模型训练器"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def compute_loss(self, input_ids, labels=None):
        """
        计算因果语言模型损失
        
        input_ids: [batch_size, seq_len]
        labels: [batch_size, seq_len]，与input_ids相同（预测下一个token）
        """
        # 前向传播
        outputs = self.model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        if labels is None:
            # 自监督：labels就是input_ids右移一位
            labels = input_ids.clone()
        
        # 计算损失（忽略padding）
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # 预测位置：logits[:, :-1, :] 预测 labels[:, 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
```

### 3.2 掩码语言建模（MLM）

BERT使用的训练目标：

```python
class MLMTrainer:
    """掩码语言模型训练器"""
    
    def __init__(self, model, tokenizer, mlm_probability=0.15):
        self.model = model
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
    def mask_tokens(self, inputs):
        """
        随机掩码token
        80% [MASK], 10% 随机token, 10% 不变
        """
        labels = inputs.clone()
        
        # 创建掩码矩阵
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 不掩码特殊token
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算掩码位置的损失
        
        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% 随机token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # 10% 不变（已经在原始inputs中）
        
        return inputs, labels
    
    def compute_loss(self, input_ids):
        """计算MLM损失"""
        masked_input_ids, labels = self.mask_tokens(input_ids.clone())
        
        outputs = self.model(masked_input_ids)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss
```

### 3.3 多任务学习

```python
class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, model, tasks_config):
        self.model = model
        self.tasks_config = tasks_config
        
    def compute_loss(self, batch, task_type):
        """根据任务类型计算损失"""
        if task_type == "causal_lm":
            return self.causal_lm_loss(batch)
        elif task_type == "mlm":
            return self.mlm_loss(batch)
        elif task_type == "span_corruption":
            return self.span_corruption_loss(batch)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
```

---

## 4. 分布式训练策略

### 4.1 数据并行（Data Parallelism）

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式训练"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
def train_with_ddp():
    setup_distributed()
    
    # 创建模型
    model = YourModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 数据加载器（使用DistributedSampler）
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            loss = model(batch)
            loss.backward()
            
            optimizer.step()
```

### 4.2 DeepSpeed ZeRO

```python
# deepspeed_config.json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "gradient_clipping": 1.0
}
```

```python
import deepspeed

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

# 训练
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 4.3 FSDP（Fully Sharded Data Parallel）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=torch.bfloat16,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True
)
```

---

## 5. 显存优化技术

### 5.1 梯度检查点（Gradient Checkpointing）

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpoint(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x):
        # 使用checkpoint节省显存
        x = checkpoint(self.attention, x, use_reentrant=False)
        x = checkpoint(self.ffn, x, use_reentrant=False)
        return x
```

### 5.2 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

### 5.3 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5.4 8-bit优化器

```python
import bitsandbytes as bnb

# 使用8-bit AdamW
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### 5.5 Flash Attention

```python
from flash_attn import flash_attn_func

# 使用Flash Attention替代标准attention
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,
    softmax_scale=None
)
```

---

## 深度分析

预训练技术是构建大语言模型的基石，其核心挑战在于如何从海量、异构的互联网数据中提取高质量的语义知识。本章所涵盖的数据清洗（去重、过滤）、Tokenization（BPE、SentencePiece）以及数据混合策略，构成了现代预训练数据管线的标准范式。实践中，数据的质量往往比数量更为关键——例如，C4数据集对Common Crawl的严格过滤显著提升了模型的收敛速度与下游任务表现。此外，Tokenization的选择直接影响模型的词汇覆盖率和推理效率：BPE在平衡词表大小与表示粒度上表现优异，而SentencePiece则通过将文本视为字节序列消除了对预分词的依赖。

分布式训练策略与显存优化技术是预训练工程化的核心支柱。从数据并行（DDP）到ZeRO优化（Stage 1-3）再到FSDP，显存效率的提升使得在有限硬件上训练大规模模型成为可能。梯度检查点、混合精度训练和Flash Attention等技术的组合使用，可以将单GPU所能容纳的模型规模扩大数倍。值得注意的是，这些优化技术并非孤立使用——例如，QLoRA（参见[微调与对齐](../chapters/ch02-finetuning-alignment.md)）正是4-bit量化与LoRA的融合，在预训练和微调场景下均展现出极佳的性价比。理解这些底层优化原理，对于后续掌握模型部署与推理优化（参见[评估与部署](../chapters/ch03-evaluation-deployment.md)）同样至关重要。

---

## Checklist

- [ ] 理解预训练与微调的本质区别（数据规模、计算资源、训练目标）
- [ ] 掌握数据清洗流程：去重、过滤、质量评估
- [ ] 熟悉 BPE 和 SentencePiece 两种 Tokenization 方法的原理与实现
- [ ] 了解数据混合策略中不同来源数据的配比原则
- [ ] 理解自回归（Causal LM）与掩码（MLM）两种训练目标的数学表达
- [ ] 掌握分布式训练三大范式：数据并行、ZeRO、FSDP
- [ ] 熟悉至少三种显存优化技术：梯度检查点、混合精度、梯度累积
- [ ] 能够配置 DeepSpeed ZeRO Stage 2 或 3 的 JSON 配置文件
- [ ] 了解 Flash Attention 的原理及其对长序列训练的意义
- [ ] 明确本章技术与后续微调、部署章节的关联

---

## 本章小结

预训练是大模型构建的基础阶段，关键要点：

1. **数据质量** 是预训练成功的关键，需要严格的清洗和去重
2. **Tokenization** 影响模型的词汇理解和生成能力
3. **训练目标** 决定了模型的能力方向（自回归/自编码）
4. **分布式训练** 使得训练大模型成为可能
5. **显存优化** 技术（检查点、混合精度、8-bit）大幅降低训练成本

**下一章：** 我们将学习模型微调与对齐技术，包括SFT、LoRA、RLHF和DPO。

---

## 延伸阅读

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — 参数高效微调的开创性工作
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — 高效注意力机制的原始论文
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 预训练规模法则，理解数据量与模型大小的关系
- [DeepSpeed: System Optimizations Enable Training Very Large Models](https://arxiv.org/abs/2205.12816) — DeepSpeed 系统优化详解
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — 预训练数据集的代表性工作

---

*最后更新: 2026-06-12*

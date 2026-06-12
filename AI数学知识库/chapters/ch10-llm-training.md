# 第十章：大模型训练与优化

> 训练大型语言模型是一个复杂的工程问题，涉及预训练、有监督微调、人类反馈强化学习等多个阶段。本章将深入讲解大模型训练的核心技术，包括**预训练**、**有监督微调（SFT）**、**RLHF**、**分布式训练**等。

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: [ch04-probability.md], [ch06-calculus.md]
- **关联文件**: [ch11-reinforcement-learning.md], [ch17-advanced-optimizers.md]
- **最后更新**: 2026-06-12

---

## 目录

1. [预训练阶段](#1-预训练阶段)
2. [有监督微调（SFT）](#2-有监督微调sft)
3. [人类反馈强化学习（RLHF）](#3-人类反馈强化学习rlhf)
4. [分布式训练策略](#4-分布式训练策略)
5. [大模型训练的关键技术](#5-大模型训练的关键技术)

---

## 1. 预训练阶段

### 1.1 预训练目标

预训练的核心目标是让模型学习通用的语言表示。

**语言模型目标：**
$$
\mathcal{L}_{\text{pretrain}} = -\sum_{t=1}^{T} \log P_\theta(w_t | w_1, \ldots, w_{t-1})
$$

### 1.2 训练数据

| 数据类型 | 特点 | 示例 |
|----------|------|------|
| 网页数据 | 量大、质量参差 | Common Crawl |
| 书籍 | 长文本、连贯 | BooksCorpus |
| 代码 | 结构化、有逻辑 | GitHub |
| 学术论文 | 专业化 | arXiv |
| 对话数据 | 交互性 | Reddit |
| 新闻 | 时效性强 | News |

**数据处理流程：**
```
原始数据 → 清洗 → 去重 → 质量过滤 → 格式转换 → 分词 → 数据集
```

```python
def preprocess_data(text, tokenizer, max_length=2048):
    """预处理训练数据"""
    # 分词
    tokens = tokenizer.encode(text)
    
    # 截断
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    return tokens

def create_training_examples(texts, tokenizer, stride=512):
    """创建滑动窗口训练样本"""
    examples = []
    
    for text in texts:
        tokens = preprocess_data(text, tokenizer)
        
        # 滑动窗口
        for i in range(0, len(tokens) - 1, stride):
            input_ids = tokens[i:i+max_length]
            labels = input_ids.copy()
            examples.append(input_ids, labels)
    
    return examples
```

### 1.3 训练配置

**典型训练配置：**

| 模型 | 参数规模 | 上下文长度 | 训练tokens | GPU数量 |
|------|----------|------------|------------|---------|
| GPT-3 | 175B | 2048 | 300B | 10000+ |
| LLaMA-2 | 70B | 4096 | 2T | 2048 |
| ChatGLM | 6B | 2048 | 1T | 96 |

**学习率调度策略：**
```python
# 余弦退火 + Warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=1000000,
    min_lr_ratio=0.1
)
```

---

## 2. 有监督微调（SFT）

### 2.1 SFT概述

预训练模型通过SFT获得指令跟随能力。

**数据集格式：**
```json
{
    "instruction": "解释什么是量子纠缠",
    "input": "",
    "output": "量子纠缠是量子力学中..."
}
```

### 2.2 SFT训练配置

**与预训练的区别：**
- 批量更小
- 学习率更低
- 使用Instruction数据集
- 训练步数更少

```python
class SupervisedFinetuner:
    def __init__(self, model, tokenizer, train_data, val_data):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        
        # SFT专用配置
        self.train_config = {
            'batch_size': 8,        # 远小于预训练
            'learning_rate': 2e-5,   # 较小学习率
            'epochs': 3,
            'warmup_ratio': 0.03,
            'gradient_accumulation': 4
        }
    
    def collate_fn(self, batch):
        """数据整理函数"""
        instructions = [item['instruction'] for item in batch]
        inputs = [item['input'] for item in batch]
        outputs = [item['output'] for item in batch]
        
        # 构建prompt模板
        prompts = [
            f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        # 分词
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        
        # 构建labels（计算loss的位置）
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return encodings['input_ids'], labels, encodings['attention_mask']
```

---

## 3. 人类反馈强化学习（RLHF）

### 3.1 RLHF概述

**RLHF** 让模型学习符合人类偏好的响应。

**三阶段流程：**
```
    阶段1：预训练      阶段2：SFT        阶段3：RLHF
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │  Base LLM │ → │ SFT Model │ → │ Reward Model │
    └──────────┘    └──────────┘    └──────┬───────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  PPO优化     │
                                    └──────┬───────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  对齐模型    │
                                    └──────────────┘
```

### 3.2 Reward Model（奖励模型）

**训练目标：** 学习人类偏好

$$
\mathcal{L}_R = -E_{(x, y_1, y_2, \偏好)}[\log \sigma(r(x, y_{\text{chosen}}) - r(x, y_{\text{rejected}}))]
$$

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # 添加奖励头
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用最后一层[CLS]或最后位置
        hidden_states = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(hidden_states)
        
        return reward

def train_reward_model(reward_model, chosen_data, rejected_data):
    """训练奖励模型"""
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    for epoch in range(num_epochs):
        for batch_chosen, batch_rejected in dataloader:
            # 计算奖励
            r_chosen = reward_model(**batch_chosen)
            r_rejected = reward_model(**batch_rejected)
            
            # 偏好损失
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 3.3 PPO算法

**PPO（Proximal Policy Optimization）** 是RLHF的核心算法。

**策略梯度目标：**
$$
\mathcal{L}^{\text{CLIP}} = -E_t[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是概率比。

```python
class PPOTrainer:
    def __init__(self, model, ref_model, reward_model):
        self.model = model
        self.ref_model = ref_model  # SFT模型（参考）
        self.reward_model = reward_model
        self.ppo_config = {
            'epsilon': 0.2,        # PPO裁剪参数
            'kl_coef': 0.04,       # KL散度系数
            'vf_coef': 0.1,        # Value函数系数
            'gamma': 1.0,          # 折扣因子
            'lam': 0.95            # GAE参数
        }
    
    def compute_kl_divergence(self, log_probs, ref_log_probs):
        """计算KL散度"""
        return (log_probs - ref_log_probs).mean()
    
    def ppo_loss(self, log_probs, old_log_probs, advantages):
        """PPO裁剪损失"""
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 裁剪
        clipped_ratio = ratio.clamp(
            1 - self.ppo_config['epsilon'],
            1 + self.ppo_config['epsilon']
        )
        
        # 最终损失
        loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return loss
    
    def train_step(self, query_data, response_data):
        """PPO训练步骤"""
        # 生成响应
        responses = self.model.generate(query_data)
        
        # 计算奖励
        rewards = self.reward_model(responses)
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            ref_log_probs = self.ref_model(response_data).log_probs
        
        # 计算当前模型的对数概率
        outputs = self.model(response_data)
        log_probs = outputs.log_probs
        
        # 计算KL散度惩罚
        kl_div = self.compute_kl_divergence(log_probs, ref_log_probs)
        
        # PPO损失
        advantages = rewards  # 简化版
        ppo_loss = self.ppo_loss(log_probs, ref_log_probs, advantages)
        
        # 最终损失
        total_loss = ppo_loss + self.ppo_config['kl_coef'] * kl_div
        
        return total_loss
```

### 3.4 DPO（Direct Preference Optimization）

**DPO** 简化了RLHF，不需要单独的奖励模型。

**目标函数：**
$$
\mathcal{L}_{\text{DPO}} = -E_{(x, y_w, y_l)}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]
$$

```python
def dpo_loss(policy_logps, reference_logps, chosen_logps, rejected_logps, beta=0.1):
    """
    DPO损失函数
    
    policy_logps: 策略模型的对数概率
    reference_logps: 参考模型的对数概率
    chosen_logps: 被选中的响应对数概率
    rejected_logps: 被拒绝的响应对数概率
    """
    # 计算优势
    policy_chosen = (chosen_logps - policy_logps).mean(dim=-1)
    policy_rejected = (rejected_logps - policy_logps).mean(dim=-1)
    
    reference_chosen = (chosen_logps - reference_logps).mean(dim=-1)
    reference_rejected = (rejected_logps - reference_logps).mean(dim=-1)
    
    # DPO公式
    logits = beta * (policy_chosen - policy_rejected) - beta * (reference_chosen - reference_rejected)
    
    return -F.logsigmoid(logits).mean()
```

---

## 4. 分布式训练策略

### 4.1 数据并行（Data Parallelism）

将数据分片，每个GPU有完整模型：

```python
# DeepSpeed ZeRO
ds_config = {
    "train_batch_size": 64,
    "zero_optimization": {
        "stage": 3,  # 完整分片
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### 4.2 流水线并行（Pipeline Parallelism）

将模型分层分布在不同GPU：

```python
# Megatron-LM流水线并行
model = TransformerLayer()
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### 4.3 张量并行（Tensor Parallelism）

将单个层的参数分片：

```python
# Megatron-LM张量并行
# Linear层改为列分片
ColumnParallelLinear = megatron_core.parallel_state.get_tensor_model_parallel_module()
```

---

## 5. 大模型训练的关键技术

### 5.1 梯度检查点（Gradient Checkpointing）

用计算换显存：

```python
# PyTorch实现
model.register_forward_pre_hook(checkpoint_hook)
```

### 5.2 混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5.3 分布式优化器

```python
# 使用FSDP（Fully Sharded Data Parallel）
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

---

## 本章小结

大模型训练的核心技术：

1. **预训练** 学习通用语言能力
2. **SFT** 获得指令跟随能力
3. **RLHF/DPO** 对齐人类偏好
4. **分布式训练** 使训练超大模型成为可能
5. **混合精度、梯度检查点** 优化显存使用

本知识库完整覆盖了AI和大模型所需的数学知识体系。

---

## 深度分析

### 训练范式的数学本质

大模型训练的三个阶段（预训练、SFT、RLHF）对应不同的数学优化目标。预训练本质是在海量数据上最大化对数似然——信息论中交叉熵最小化的直接应用。SFT将通用语言模型适配到特定任务分布，本质是条件概率分布的迁移学习。RLHF则引入偏好排序的数学框架，将人类价值观编码为奖励函数。

### 关键权衡

训练效率与模型质量之间存在根本性张力。更大的batch size提高吞吐但可能降低泛化性能；更长的训练步数提升收敛但增加成本；混合精度训练加速计算但引入数值误差。

---

## 训练实践Checklist

- [ ] 理解预训练的损失函数设计（CLM/MLM/PLM）
- [ ] 掌握学习率调度策略（cosine/warmup/decay）
- [ ] 能够配置分布式训练策略（DDP/FSDP/DeepSpeed）
- [ ] 理解混合精度训练（FP16/BF16/FP8）的数值范围
- [ ] 掌握梯度积累和梯度裁剪的配置方法
- [ ] 能够诊断训练不收敛的常见原因
- [ ] 理解SFT数据配比对模型行为的影响
- [ ] 掌握RLHF中的奖励模型训练技巧
- [ ] 了解DPO与RLHF的数学等价性
- [ ] 能够评估训练效率（MFU/吞吐/显存利用率）

---

## 延伸阅读

- [线性代数](ch01-linear-algebra.md)
- [概率论](ch03-probability.md)
- [神经网络](ch06-neural-networks.md)
- [优化理论](ch05-optimization.md)
- [强化学习](ch11-reinforcement-learning.md)

---

*最后更新：2026-06-12*

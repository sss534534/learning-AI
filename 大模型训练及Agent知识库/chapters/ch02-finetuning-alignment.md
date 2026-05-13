# 第二章：模型微调与对齐

> 预训练后的Base模型需要通过微调和对齐来获得特定任务能力和人类偏好。本章将深入讲解有监督微调（SFT）、参数高效微调（PEFT）、RLHF、DPO等关键技术。

## 目录

1. [有监督微调（SFT）](#1-有监督微调sft)
2. [参数高效微调（PEFT）](#2-参数高效微调peft)
3. [人类反馈强化学习（RLHF）](#3-人类反馈强化学习rlhf)
4. [直接偏好优化（DPO）](#4-直接偏好优化dpo)
5. [模型合并与集成](#5-模型合并与集成)

---

## 1. 有监督微调（SFT）

### 1.1 SFT概述

SFT通过指令数据让模型学习遵循人类指令的能力。

**数据格式：**
```json
{
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

### 1.2 指令模板设计

```python
class InstructionTemplate:
    """指令模板"""
    
    def __init__(self, template_type="alpaca"):
        self.template_type = template_type
        
        self.templates = {
            "alpaca": {
                "prompt": "Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n"
                         "### Response:\n",
                "response": "{output}"
            },
            "chatml": {
                "system": "<|im_start|>system\n{system}<|im_end|>\n",
                "user": "<|im_start|>user\n{instruction}<|im_end|>\n",
                "assistant": "<|im_start|>assistant\n{output}<|im_end|>\n"
            },
            "llama2": {
                "system": "<<SYS>>\n{system}\n<</SYS>>\n\n",
                "user": "[INST] {instruction} [/INST] ",
                "assistant": "{output}"
            }
        }
    
    def format(self, instruction, output="", system=""):
        """格式化指令"""
        template = self.templates[self.template_type]
        
        if self.template_type == "alpaca":
            prompt = template["prompt"].format(instruction=instruction)
            if output:
                return prompt + template["response"].format(output=output)
            return prompt
        
        elif self.template_type == "chatml":
            result = ""
            if system:
                result += template["system"].format(system=system)
            result += template["user"].format(instruction=instruction)
            if output:
                result += template["assistant"].format(output=output)
            return result
        
        elif self.template_type == "llama2":
            result = ""
            if system:
                result += template["system"].format(system=system)
            result += template["user"].format(instruction=instruction)
            if output:
                result += template["assistant"].format(output=output)
            return result

# 使用示例
template = InstructionTemplate("chatml")
formatted = template.format(
    instruction="你好",
    output="你好！有什么我可以帮助你的吗？",
    system="你是一个 helpful 的助手"
)
print(formatted)
```

### 1.3 SFT训练流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 准备数据
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def formatting_prompts_func(example):
    """格式化数据"""
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n"
        if example['input'][i]:
            text += f"### Input:\n{example['input'][i]}\n\n"
        text += f"### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

# 训练参数
training_args = TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="paged_adamw_8bit"
)

# 创建Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=2048,
    args=training_args
)

# 训练
trainer.train()
```

---

## 2. 参数高效微调（PEFT）

### 2.1 LoRA（Low-Rank Adaptation）

LoRA通过在原始权重旁添加低秩矩阵来微调模型。

**原理：**
$$
W' = W + \Delta W = W + BA
$$
其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA配置
lora_config = LoraConfig(
    r=16,  # 低秩维度
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出示例: trainable params: 33,554,432 || all params: 6,771,970,048 || trainable%: 0.4956
```

### 2.2 QLoRA（Quantized LoRA）

QLoRA结合4-bit量化和LoRA，进一步降低显存需求。

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 应用LoRA
model = get_peft_model(model, lora_config)
```

### 2.3 其他PEFT方法

| 方法 | 原理 | 显存节省 | 适用场景 |
|------|------|----------|----------|
| **Prefix Tuning** | 在输入前添加可训练前缀 | 高 | 生成任务 |
| **P-Tuning** | 使用连续提示 | 高 | NLU任务 |
| **Prompt Tuning** | 软提示微调 | 极高 | 少样本场景 |
| **IA³** | 学习缩放向量 | 中 | 多种任务 |
| **Adapter** | 插入小型适配层 | 中 | 多任务 |

---

## 3. 人类反馈强化学习（RLHF）

### 3.1 RLHF三阶段流程

```
阶段1: 预训练          阶段2: SFT            阶段3: RLHF
┌──────────┐         ┌──────────┐         ┌──────────────┐
│ Base LLM │   →    │ SFT Model│   →    │ Reward Model │
└──────────┘         └──────────┘         └──────┬───────┘
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

### 3.2 奖励模型训练

```python
from transformers import AutoModelForSequenceClassification

# 创建奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "sft_model",
    num_labels=1  # 回归任务
)

def compute_reward_loss(chosen_rewards, rejected_rewards):
    """
    计算奖励模型损失
    chosen_rewards: 被选中的响应的奖励
    rejected_rewards: 被拒绝的响应的奖励
    """
    # 希望 chosen 的奖励 > rejected 的奖励
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss

# 训练奖励模型
for batch in dataloader:
    chosen_rewards = reward_model(batch["chosen_input_ids"])
    rejected_rewards = reward_model(batch["rejected_input_ids"])
    
    loss = compute_reward_loss(chosen_rewards, rejected_rewards)
    loss.backward()
    optimizer.step()
```

### 3.3 PPO训练

```python
from trl import PPOTrainer, PPOConfig

# PPO配置
ppo_config = PPOConfig(
    model_name="sft_model",
    learning_rate=1.41e-5,
    batch_size=256,
    mini_batch_size=64,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,
    ppo_epochs=4,
    seed=0,
    init_kl_coef=0.2,
    adap_kl_ctrl=True
)

# 创建PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset
)

# PPO训练循环
for epoch in range(num_epochs):
    for batch in ppo_trainer.dataloader:
        # 生成响应
        queries = batch["query"]
        response_tensors = ppo_trainer.generate(queries)
        
        # 计算奖励
        rewards = reward_model(response_tensors)
        
        # PPO更新
        stats = ppo_trainer.step(queries, response_tensors, rewards)
        
        # 日志
        ppo_trainer.log_stats(stats, batch, rewards)
```

---

## 4. 直接偏好优化（DPO）

### 4.1 DPO原理

DPO直接优化策略模型，无需单独的奖励模型。

**损失函数：**
$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

```python
from trl import DPOTrainer, DPOConfig

# DPO配置
dpo_config = DPOConfig(
    beta=0.1,  # 温度参数
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    warmup_steps=100
)

# 准备偏好数据
# 格式: {"prompt": "...", "chosen": "...", "rejected": "..."}

# 创建DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 参考模型（通常是SFT模型）
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer
)

# 训练
dpo_trainer.train()
```

### 4.2 DPO vs RLHF

| 特性 | RLHF | DPO |
|------|------|-----|
| 奖励模型 | 需要单独训练 | 不需要 |
| 稳定性 | 需要 careful tuning | 更稳定 |
| 计算成本 | 高 | 低 |
| 超参数 | 多（PPO相关） | 少（主要是β） |
| 效果 | 通常更好 | 接近RLHF |

---

## 5. 模型合并与集成

### 5.1 模型合并（Model Merging）

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载并合并LoRA权重
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")
model = model.merge_and_unload()  # 合并权重

# 保存完整模型
model.save_pretrained("merged_model")
```

### 5.2 任务向量合并

```python
def merge_models_with_task_vectors(base_model, *finetuned_models, weights=None):
    """
    使用任务向量合并多个微调模型
    """
    if weights is None:
        weights = [1.0 / len(finetuned_models)] * len(finetuned_models)
    
    # 获取基础模型状态
    base_state = base_model.state_dict()
    
    # 计算任务向量并合并
    merged_state = {}
    for key in base_state.keys():
        merged_state[key] = base_state[key].clone()
        
        for ft_model, weight in zip(finetuned_models, weights):
            ft_state = ft_model.state_dict()
            # 任务向量 = 微调权重 - 基础权重
            task_vector = ft_state[key] - base_state[key]
            merged_state[key] += weight * task_vector
    
    # 加载合并后的权重
    base_model.load_state_dict(merged_state)
    return base_model
```

### 5.3 TIES合并

```python
def ties_merging(base_model, *finetuned_models, reset_thresh=20, merge_func="dis-mean"):
    """
    TIES合并：修剪、选择符号、合并
    """
    base_state = base_model.state_dict()
    
    # 步骤1: 修剪（Trim）
    trimmed_vectors = []
    for ft_model in finetuned_models:
        ft_state = ft_model.state_dict()
        task_vector = {k: ft_state[k] - base_state[k] for k in base_state.keys()}
        
        # 保留最重要的reset_thresh%参数
        for key in task_vector:
            flat = task_vector[key].flatten()
            k = int(len(flat) * reset_thresh / 100)
            topk_values, topk_indices = torch.topk(torch.abs(flat), k)
            mask = torch.zeros_like(flat)
            mask[topk_indices] = 1
            task_vector[key] = (task_vector[key] * mask.view(task_vector[key].shape))
        
        trimmed_vectors.append(task_vector)
    
    # 步骤2: 选择符号（Elect Sign）
    elected_signs = {}
    for key in base_state.keys():
        # 多数投票决定符号
        signs = [torch.sign(tv[key]) for tv in trimmed_vectors]
        elected_signs[key] = torch.sign(sum(signs))
    
    # 步骤3: 不相交合并（Disjoint Merge）
    merged_state = base_state.copy()
    for key in base_state.keys():
        # 只保留与选定符号一致的参数
        for tv in trimmed_vectors:
            mask = torch.sign(tv[key]) == elected_signs[key]
            merged_state[key] += tv[key] * mask
    
    base_model.load_state_dict(merged_state)
    return base_model
```

---

## 本章小结

模型微调与对齐是将Base模型转化为可用产品的关键步骤：

1. **SFT** 让模型学会遵循指令
2. **LoRA/QLoRA** 大幅降低微调成本
3. **RLHF** 通过人类反馈对齐模型
4. **DPO** 提供了一种更简单的对齐方法
5. **模型合并** 可以组合多个微调模型的能力

**下一章：** 我们将学习模型评估与部署技术。

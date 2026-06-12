# 04-微调与PEFT深度实战

> 从全量微调到参数高效微调（LoRA/QLoRA/DoRA），覆盖数据策略、超参搜索、评估闭环、生产部署，构建完整的微调工程能力。

## 元数据
- **难度**: ⭐⭐
- **前置知识**: [01-模型量化技术.md](./01-模型量化技术.md)（QLoRA部分）
- **关联文件**: [05-AI性能工程与基准测试.md](./05-AI性能工程与基准测试.md)
- **最后更新**: 2026-06-12
---

---

## 1. 微调全景：什么时候需要微调？

### 1.1 决策框架

```
需要微调吗？
├── 任务是否在模型训练数据分布外？
│   ├── 是 → 考虑微调
│   └── 否 → 尝试 Prompt Engineering / Few-shot
├── 对延迟/成本敏感？
│   ├── 是 → 微调小模型替代大模型
│   └── 否 → 直接用最强模型 + 长Prompt
├── 需要领域深度知识？
│   ├── 是 → 必须微调（法律、医疗、金融专用术语）
│   └── 否 → RAG可能就够
└── 数据量是否充足？
    ├── < 100条 → Few-shot Prompting
    ├── 100-1000条 → LoRA微调
    ├── 1000-10000条 → QLoRA / 全量微调
    └── > 10000条 → 全量微调（考虑继续预训练）
```

### 1.2 微调 vs RAG vs Prompt Engineering

| 维度 | Prompt Engineering | RAG | 微调 |
|------|-------------------|-----|------|
| 实施成本 | 极低 | 中（向量库+检索管道） | 高（GPU+数据+训练） |
| 知识更新 | 实时 | 实时（更新向量库） | 需重新训练 |
| 推理延迟 | 不变 | 略增（检索开销） | 不变/可能降低 |
| 风格控制 | 弱 | 弱 | 强 |
| 领域术语 | 一般 | 一般 | 优秀 |
| 幻觉控制 | 依赖指令 | 较好（有检索依据） | 训练数据决定 |
| 典型场景 | 通用任务、快速验证 | 知识密集型问答 | 领域特化、风格迁移 |

---

## 2. 全量微调（Full Fine-Tuning）

### 2.1 数学原理

全量微调更新模型所有参数 $W$：

$$\min_{\Delta W} \mathcal{L}(f_{W + \Delta W}(x), y)$$

**关键挑战：**
- 7B模型 = ~14GB参数 × 4（optimizer states）= **56GB+ 显存**
- 70B模型 = 无法在单卡上进行全量微调

### 2.2 训练配置模板（DeepSpeed ZeRO-3）

```yaml
# deepspeed_config.yaml
train_batch_size: 128
train_micro_batch_size_per_gpu: 4
gradient_accumulation_steps: 8

zero_optimization:
  stage: 3
  offload_optimizer:
    device: cpu
    pin_memory: true
  offload_param:
    device: cpu
    pin_memory: true
  overlap_comm: true
  contiguous_gradients: true
  reduce_bucket_size: 5e8
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6

bf16:
  enabled: true

optimizer:
  type: AdamW
  params:
    lr: 2e-5
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.1

scheduler:
  type: WarmupDecayLR
  params:
    warmup_min_lr: 0
    warmup_max_lr: 2e-5
    warmup_num_steps: 100
    total_num_steps: 1000
```

### 2.3 全量微调的陷阱

| 陷阱 | 现象 | 解决方案 |
|------|------|---------|
| 灾难性遗忘 | 微调后通用能力下降 | 混合通用数据(10-20%) + EMA |
| 过拟合 | 训练loss降但验证loss升 | 早停 + 权重衰减 + 数据增强 |
| 训练不稳定 | loss震荡/NaN | 梯度裁剪(1.0) + Warmup + BF16 |
| 算力浪费 | 训练很久但效果不如LoRA | 先用LoRA验证可行性，再决定是否全量 |

---

## 3. LoRA：低秩适配器

### 3.1 核心思想

不修改原始权重 $W_0 \in \mathbb{R}^{d \times k}$，而是训练一个低秩增量：

$$h = W_0 x + \frac{\alpha}{r} \cdot BA x$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**关键洞察：** 微调过程中的权重更新矩阵 $\Delta W$ 具有**内在低秩性**（Aghajanyan et al., 2021），因此用低秩分解即可捕获大部分任务适配信息。

### 3.2 LoRA配置参数决策树

```
选择 rank (r)：
├── 简单分类任务 → r=4~8
├── 指令遵循/对话 → r=16~32
├── 代码生成 → r=32~64
└── 数学推理/复杂推理 → r=64~128

选择 alpha：
├── alpha = 2*r（标准）
├── 需要更强的适配 → alpha = 4*r
└── 担心过拟合 → alpha = r

选择 target_modules：
├── Llama/Qwen系列 → ["q_proj", "v_proj", "k_proj", "o_proj"]
├── 更强效果 → 加上 ["gate_proj", "up_proj", "down_proj"]
├── 节省显存 → 仅 ["q_proj", "v_proj"]
└── Mistral/Mixtral → 同上，注意 attention 命名差异

dropout：
├── 数据量 > 1000 → 0.05
├── 数据量 < 500 → 0.1
└── 数据量 > 10000 → 0（不需要）
```

### 3.3 LoRA训练代码（HuggingFace PEFT）

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

# ===== 1. 加载基座模型 =====
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ===== 2. LoRA 配置 =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank
    lora_alpha=32,                 # alpha = 2*r
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 41,943,040 || all params: 7,615,600,640 || trainable%: 0.55%

# ===== 3. 训练参数 =====
training_args = TrainingArguments(
    output_dir="./qwen-lora-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,            # LoRA通常比全量微调高一个数量级
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,   # 节省显存
    lr_scheduler_type="cosine",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# ===== 4. 训练 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
trainer.train()

# ===== 5. 保存 =====
model.save_pretrained("./qwen-lora-adapter")
tokenizer.save_pretrained("./qwen-lora-adapter")
```

### 3.4 LoRA推理合并

```python
from peft import PeftModel

# 方式1：加载base + adapter（灵活，可热切换adapter）
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./qwen-lora-adapter")
model = model.merge_and_unload()  # 合并权重，消除推理开销

# 方式2：直接加载合并后的模型（用于serving）
model.save_pretrained("./qwen-lora-merged")
```

---

## 4. QLoRA：4-bit量化的LoRA

### 4.1 技术原理

QLoRA = **NF4量化** + **双重量化** + **分页优化器**

| 组件 | 作用 | 效果 |
|------|------|------|
| NF4 (4-bit NormalFloat) | 信息论最优4-bit数据类型 | 相比FP4精度损失更小 |
| 双重量化 | 对量化常数量化 | 额外节省0.37 bits/参数 |
| 分页优化器 | GPU OOM时自动offload到CPU | 允许在单张24GB显卡上微调65B模型 |

### 4.2 QLoRA配置模板

```python
from transformers import BitsAndBytesConfig

# NF4量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA配置同上，但注意：
# 1. QLoRA需要在所有线性层应用LoRA（因为原始权重是4-bit的）
# 2. 学习率可以比标准LoRA稍低（1e-4 ~ 2e-4）
lora_config = LoraConfig(
    r=64,                    # QLoRA可以用更大的rank
    lora_alpha=128,
    target_modules="all-linear",  # 或手动指定所有线性层
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

### 4.3 显存对比（Qwen2.5-7B）

| 方法 | 显存占用 | 训练速度(相对) | 效果(相对全量) |
|------|---------|---------------|---------------|
| 全量微调 (BF16) | ~56 GB | 1.0x | 100% |
| LoRA (BF16, r=16) | ~18 GB | 1.5x | 95-98% |
| QLoRA (NF4, r=64) | ~10 GB | 1.2x | 93-97% |
| QLoRA (NF4, r=16) | ~8 GB | 1.3x | 90-95% |

---

## 5. DoRA：权重分解LoRA

### 5.1 动机

LoRA将更新约束在低秩子空间，限制了对**幅度**和**方向**的解耦学习能力。DoRA（Weight-Decomposed Low-Rank Adaptation）将预训练权重分解为幅度和方向：

$$W = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c}$$

其中 $m$ 是可学习的幅度向量，$V$ 是方向矩阵，$\Delta V = BA$ 是低秩方向更新。

### 5.2 DoRA vs LoRA

```python
from peft import LoraConfig  # DoRA复用LoraConfig，设置use_dora=True

dora_config = LoraConfig(
    use_dora=True,           # 启用DoRA
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```

**效果对比（来自论文）：**

| 任务 | LoRA (r=16) | DoRA (r=16) | 提升 |
|------|------------|------------|------|
| 常识推理 | 74.5% | 76.2% | +1.7% |
| 数学推理 | 52.3% | 55.8% | +3.5% |
| 代码生成 | 48.7% | 50.1% | +1.4% |
| 多任务平均 | 68.1% | 70.4% | +2.3% |

**代价：** 额外约10%的训练时间和5%的显存。

---

## 6. 数据工程：微调数据的质量决定上限

### 6.1 数据质量标准

```python
# 数据质量检查清单
quality_checks = {
    "格式一致性": "所有样本必须符合统一格式（如ChatML）",
    "指令多样性": "避免重复或高度相似的指令",
    "回答准确性": "事实性错误是微调的最大杀手",
    "长度分布": "控制输入输出长度，避免极端样本",
    "去污染": "移除与评测集重叠的样本",
    "安全过滤": "移除有害、偏见、敏感内容",
    "难度分级": "简单/中等/困难样本保持合理比例",
}
```

### 6.2 数据格式（ChatML）

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的金融分析师，擅长解读财报和宏观经济数据。"},
    {"role": "user", "content": "分析茅台2025年Q1财报的核心亮点和风险点。"},
    {"role": "assistant", "content": "茅台2025年Q1财报核心分析如下：\n\n**亮点：**\n1. 营收同比增长12.3%至468亿元..."}
  ]
}
```

### 6.3 数据配比策略

```
微调数据集配方（以10000条为例）：
├── 70% - 目标任务数据（核心能力）
├── 15% - 通用指令数据（防止遗忘）
├── 10% - 安全对齐数据（拒绝有害请求）
├── 3% - 数学/推理数据（维持推理能力）
└── 2% - 高质量few-shot示例（格式引导）
```

### 6.4 数据增强技术

```python
# Self-Instruct 式数据增强
def generate_synthetic_data(seed_instructions, model, n=100):
    """
    基于种子指令，用强模型生成多样性训练数据
    """
    prompts = [
        f"基于以下任务模板，生成5个变体：\n{seed}\n要求：改变领域、难度或角度"
        for seed in seed_instructions * (n // 5)
    ]
    # 用GPT-4/DeepSeek等强模型生成
    # 注意：需要人工或自动质量过滤
    return generated_samples

# Evol-Instruct: 逐步增加复杂度
def evolve_instruction(instruction, model):
    """将简单指令进化为更复杂的版本"""
    evolution_prompts = [
        "给这个任务增加约束条件",
        "要求更深层的推理",
        "增加多步操作",
        "引入反直觉的陷阱",
    ]
    # 随机选择一个进化方向
    # ...
```

### 6.5 自动化质量过滤

```python
def auto_filter(dataset, quality_model):
    """使用评分模型自动过滤低质量数据"""
    filtered = []
    for sample in dataset:
        score = quality_model.evaluate(sample, criteria=[
            "instruction_clarity",     # 指令清晰度 (1-5)
            "response_correctness",    # 回答正确性 (1-5)
            "response_helpfulness",    # 回答有用性 (1-5)
            "format_compliance",       # 格式合规 (1-5)
        ])
        if all(v >= 3 for v in score.values()):
            filtered.append(sample)
    return filtered
```

---

## 7. 超参数调优

### 7.1 LoRA超参数搜索空间

| 超参数 | 搜索范围 | 推荐起点 |
|--------|---------|---------|
| rank (r) | {4, 8, 16, 32, 64, 128} | 16 |
| alpha | {r, 2r, 4r} | 2r |
| learning_rate | {1e-5, 5e-5, 1e-4, 2e-4, 5e-4} | 2e-4 |
| batch_size | {1, 2, 4, 8, 16} | 4 |
| epochs | {1, 2, 3, 5, 10} | 3 |
| warmup_ratio | {0.03, 0.05, 0.1} | 0.03 |
| lora_dropout | {0, 0.05, 0.1} | 0.05 |

### 7.2 Optuna自动搜索

```python
import optuna

def objective(trial):
    r = trial.suggest_categorical("r", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    epochs = trial.suggest_int("epochs", 1, 5)

    config = LoraConfig(r=r, lora_alpha=2*r,
                        lora_dropout=dropout, ...)

    model = get_peft_model(base_model, config)
    trainer = Trainer(model=model, args=TrainingArguments(
        learning_rate=lr, num_train_epochs=epochs, ...
    ))
    trainer.train()
    eval_result = trainer.evaluate()

    return eval_result["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print(f"Best params: {study.best_params}")
print(f"Best loss: {study.best_value}")
```

### 7.3 学习率调度策略对比

| 调度器 | 适用场景 | 特点 |
|--------|---------|------|
| Cosine | 通用 | 平滑衰减，最常用 |
| Linear | 小数据集 | 控制简单的衰减曲线 |
| Constant + Warmup | 持续预训练 | 需要长时间稳定学习 |
| Cosine with Restarts | 多次微调迭代 | 周期性重置，跳出局部最优 |

---

## 8. 评估闭环

### 8.1 多维度评估框架

```python
class FineTuningEvaluator:
    """微调评估器：覆盖能力保持 + 任务提升两个维度"""

    def __init__(self, base_model, finetuned_model):
        self.base = base_model
        self.finetuned = finetuned_model
        self.results = {}

    def evaluate_all(self):
        # 1. 目标任务评估
        self.results["target_task"] = self._eval_target_task()

        # 2. 通用能力保持（防遗忘）
        self.results["general"] = self._eval_general_ability()

        # 3. 安全性评估
        self.results["safety"] = self._eval_safety()

        # 4. 效率评估
        self.results["efficiency"] = self._eval_efficiency()

        return self._compute_score()

    def _eval_target_task(self):
        """目标任务：准确率、F1、BLEU/ROUGE等"""
        metrics = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "exact_match": 0.0,
            "human_preference_win_rate": 0.0,  # 人工偏好胜率
        }
        # 在目标任务测试集上评估
        return metrics

    def _eval_general_ability(self):
        """通用能力：在标准基准上的退化程度"""
        benchmarks = {
            "mmlu": None,          # 知识广度
            "gsm8k": None,         # 数学推理
            "humaneval": None,     # 代码生成
            "mt_bench": None,      # 多轮对话
        }
        for name in benchmarks:
            base_score = self._benchmark(self.base, name)
            ft_score = self._benchmark(self.finetuned, name)
            # 退化率 = (ft - base) / base
            degradation = (ft_score - base_score) / base_score * 100
            benchmarks[name] = {
                "base": base_score,
                "finetuned": ft_score,
                "degradation_pct": degradation,
            }
        return benchmarks

    def _eval_safety(self):
        """安全评估：拒绝率、有害输出率"""
        # 使用安全基准数据集（如 BeaverTails, SafeRLHF）
        safety_metrics = {
            "refusal_rate": 0.0,         # 对有害请求的拒绝率
            "over_refusal_rate": 0.0,    # 对无害请求的过度拒绝率
            "toxicity_score": 0.0,       # 毒性评分（越低越好）
        }
        return safety_metrics

    def _compute_score(self):
        """综合评分：target_task提升 + general退化惩罚"""
        target_gain = self.results["target_task"]["accuracy"]
        general_degradation = sum(
            abs(m["degradation_pct"])
            for m in self.results["general"].values()
        ) / len(self.results["general"])

        # 综合分数 = 任务提升 - 退化惩罚
        composite_score = target_gain - 0.5 * (general_degradation / 100)
        return composite_score
```

### 8.2 人工评估协议

```
评估维度（1-5分）：
1. 准确性：回答是否事实正确
2. 完整性：是否覆盖了问题的所有方面
3. 相关性：是否紧扣用户问题
4. 格式合规：是否遵循了输出格式要求
5. 安全性：是否避免了有害/偏见内容

盲评流程：
- 随机打乱base模型和微调模型的输出
- 至少3位评估者独立打分
- 计算 Inter-rater Reliability (Cohen's Kappa > 0.6)
- 对比胜率时使用 Bradley-Terry 模型
```

---

## 9. 进阶技术

### 9.1 RAFT：检索增强微调

RAFT（Retrieval Augmented Fine-Tuning）在微调时模拟RAG环境，训练模型学会利用检索文档：

```python
def raft_training_sample(query, documents, answer):
    """
    RAFT训练样本构造：
    1. 混合相关和不相关文档
    2. 在相关文档中插入干扰段落
    3. 训练模型从噪声中提取关键信息
    """
    # 正例文档（相关）
    positive_docs = documents["relevant"]  # 80%
    # 负例文档（不相关）
    negative_docs = documents["irrelevant"]  # 20%

    # 构造上下文
    context = []
    for doc in positive_docs + negative_docs:
        context.append(f"<DOCUMENT>{doc}</DOCUMENT>")

    # 对正例文档，插入引用标记
    formatted_answer = answer  # 包含引用标注：<CITE>doc_id</CITE>

    instruction = f"""基于以下文档回答问题。如果文档中没有相关信息，请明确说明。
{''.join(context)}

问题：{query}"""

    return {"instruction": instruction, "output": formatted_answer}
```

### 9.2 DPO：直接偏好优化（替代RLHF）

```python
# DPO 训练数据格式
dpo_sample = {
    "prompt": "解释量子纠缠",
    "chosen": "量子纠缠是指两个或多个粒子...（正确、清晰、有帮助的回答）",
    "rejected": "量子纠缠就是两个粒子谈恋爱...（错误或低质量的回答）"
}

# DPO训练代码片段
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                     # 温度参数，控制与参考模型的偏离程度
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=2048,
    max_prompt_length=1024,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,          # 参考模型（通常是基座模型或SFT模型）
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

**DPO vs RLHF 对比：**

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要奖励模型 | 是 | 否 |
| 训练稳定性 | 低（PPO难调） | 高 |
| 显存占用 | 高（4个模型） | 中（2个模型） |
| 效果上限 | 理论上更高 | 接近PPO |
| 适用场景 | 复杂对齐任务 | 标准偏好对齐 |

### 9.3 ORPO：一步式对齐

ORPO（Odds Ratio Preference Optimization）将SFT和偏好对齐合并为一步训练，不需要参考模型：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$$

优势：训练速度比SFT→DPO两阶段快2倍，显存只需一个模型。

### 9.4 Multi-LoRA Serving

```python
# 生产环境中同时服务多个LoRA adapter
# 使用 vLLM 或 S-LoRA

# vLLM Multi-LoRA配置
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=10,           # 最多同时加载10个LoRA
    max_cpu_loras=50,       # CPU上可缓存50个
)

# 为不同tenant/任务加载不同adapter
lora_request_1 = LoRARequest("finance_lora", 1, "/path/to/finance-lora")
lora_request_2 = LoRARequest("medical_lora", 2, "/path/to/medical-lora")

# 推理时指定adapter
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request_1,  # 使用金融adapter
)
```

---

## 10. 生产部署检查清单

### 10.1 上线前检查

```
[ ] 微调数据：通过质量过滤和人工抽检
[ ] 评测报告：目标提升率 + 通用能力退化率 + 安全性
[ ] A/B测试：在5-10%流量上对比base模型
[ ] 回滚方案：保留base模型服务，随时切换
[ ] 监控就绪：目标指标 + 通用指标 + 延迟/成本
[ ] 适配器版本管理：Git LFS + 模型注册表
[ ] 灰度发布：1% → 5% → 25% → 100%
[ ] 人工评估：至少3位评估者盲评确认提升
```

### 10.2 常见故障处理

| 故障 | 排查方向 | 应急措施 |
|------|---------|---------|
| 微调后效果变差 | 数据质量/过拟合/超参不当 | 回滚到base模型，重新检查数据 |
| 通用能力崩了 | 任务数据占比过高/未混合通用数据 | 加入15%通用数据重新训练 |
| 推理变慢 | adapter未合并/merge_and_unload缺失 | 使用merge_and_unload合并权重 |
| 生产OOM | adapter大小超预期 | 降低rank/使用QLoRA/增加GPU |
| 安全拒答失效 | 安全数据过少/被任务数据覆盖 | 增加安全数据比例至15% |

---

## 11. 工具链速查

| 工具 | 用途 | 命令示例 |
|------|------|---------|
| **axolotl** | 一站式微调 | `axolotl train config.yml` |
| **LLaMA-Factory** | 中文友好微调UI | Web界面操作，支持100+模型 |
| **Unsloth** | 极速微调(2-5x加速) | `from unsloth import FastLanguageModel` |
| **trl** | RLHF/DPO训练 | `from trl import SFTTrainer, DPOTrainer` |
| **PEFT** | LoRA/QLoRA/DoRA实现 | `from peft import LoraConfig, get_peft_model` |
| **DeepSpeed** | 分布式训练加速 | `deepspeed --num_gpus=4 train.py` |
| **wandb / MLflow** | 实验追踪 | `wandb.init(project="lora-finetune")` |

---

## 总结

| 层级 | 技术 | 适用场景 |
|------|------|---------|
| **入门** | LoRA (r=8~16) | 快速验证、小数据集、风格迁移 |
| **进阶** | QLoRA (r=32~64) | 大模型低资源微调、个人开发者 |
| **高级** | DoRA + DPO | 追求最优效果、偏好对齐 |
| **专家** | Multi-LoRA Serving + RAFT | 多租户生产系统、检索增强微调 |

微调不是魔法，是**数据工程 + 超参科学 + 评估闭环**的系统工程。掌握了这三者，才算真正掌握了微调。

## 深度分析

微调的核心矛盾在于"适应"与"遗忘"之间的平衡。全量微调可以充分适配目标任务，但计算成本和灾难性遗忘风险都很高；LoRA通过低秩假设大幅降低可训练参数量，但其表达能力受限于秩r的选择。架构师需要理解的是，LoRA并非"劣化版的全量微调"，而是一种有不同归纳偏好的方法——低秩更新天然具有正则化效果，在小数据场景下往往优于全量微调。QLoRA的NF4量化进一步降低了资源门槛，使得单卡24GB即可微调65B模型，这改变了微调的经济学。

数据质量是微调效果的上限。从实践角度看，1000条精心筛选和增强的数据往往优于10000条含噪声数据。Self-Instruct和Evol-Instruct等合成数据方法虽然能大幅扩充数据量，但质量控制才是真正的瓶颈。架构师应建立自动化的数据质量过滤管道，至少覆盖格式一致性、事实正确性和安全合规三个维度，并在数据准备阶段投入与训练阶段同等的时间预算。

微调正在从"一次性项目"演变为"持续迭代的工程系统"。Multi-LoRA Serving使一个基座模型可以同时服务多个领域适配器，DPO和ORPO等偏好对齐方法让对齐训练更加简化。架构师应将微调嵌入到持续集成体系中，建立自动化的评测闭环——不仅仅是目标任务的提升率，还要监控通用能力退化、安全性和推理成本三个维度，做到"不伤害"是微调上线的最低标准。

## Checklist

- [ ] 决策前回答：RAG/PE是否已充分探索？是否需要微调？
- [ ] 使用LoRA作为默认方案，全量微调仅在有明确证据时使用
- [ ] 数据质量检查：格式统一、事实正确、安全过滤、去污染
- [ ] 微调数据配比：70%任务数据 + 15%通用数据 + 10%安全数据
- [ ] QLoRA在资源受限场景下优先选择（NF4 + 双重量化）
- [ ] DoRA在数学/代码等复杂推理任务上替代LoRA（+2-3%效果）
- [ ] 超参搜索：至少覆盖 r/lr/dropout/batch_size 四个维度
- [ ] 评估闭环：目标任务提升率 + 通用能力退化率 + 安全性
- [ ] 生产部署前完成A/B测试（5-10%流量对比base模型）
- [ ] 建立灰度发布和回滚机制（1%→5%→25%→100%）

## 延伸阅读

- [01-模型量化技术.md](./01-模型量化技术.md) — QLoRA中NF4量化和双重量化的技术原理
- [05-AI性能工程与基准测试.md](./05-AI性能工程与基准测试.md) — 微调后的模型性能基准测试
- LoRA: Low-Rank Adaptation of Large Language Models（Hu et al., 2021）
- QLoRA: Efficient Finetuning of Quantized Language Models（Dettmers et al., 2023）
- DoRA: Weight-Decomposed Low-Rank Adaptation（Liu et al., 2024）

---

*最后更新：2026-06-12*

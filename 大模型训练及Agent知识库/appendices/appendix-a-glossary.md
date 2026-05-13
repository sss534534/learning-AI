# 附录 A：术语表

## 目录

1. [预训练相关术语](#1-预训练相关术语)
2. [微调与对齐相关术语](#2-微调与对齐相关术语)
3. [Agent系统相关术语](#3-agent系统相关术语)
4. [部署与优化相关术语](#4-部署与优化相关术语)

---

## 1. 预训练相关术语

### 1.1 基础概念

| 术语 | 英文 | 解释 |
|------|------|------|
| **预训练** | Pre-training | 在大规模无标注数据上训练模型，使其学习通用语言表示能力的过程 |
| **基础模型** | Base Model | 经过预训练但尚未经过微调的模型，具有通用能力 |
| **自回归语言模型** | Causal LM | 从左到右逐词预测的语言模型，如GPT系列 |
| **掩码语言模型** | Masked LM | 预测被掩码位置的语言模型，如BERT |
| **大语言模型** | LLM (Large Language Model) | 参数量巨大（通常数十亿以上）的语言模型 |

### 1.2 数据与Tokenization

| 术语 | 英文 | 解释 |
|------|------|------|
| **Token** | Token | 文本的最小处理单位，可以是字、词或子词 |
| **Tokenization** | Tokenization | 将文本分割为token的过程 |
| **字节对编码** | BPE (Byte Pair Encoding) | 一种常用的子词tokenization算法 |
| **SentencePiece** | SentencePiece | 一种不依赖预分词的tokenization库 |
| **词表** | Vocabulary | 模型使用的所有token的集合 |

### 1.3 训练技术

| 术语 | 英文 | 解释 |
|------|------|------|
| **数据并行** | Data Parallelism | 将数据切分到多个GPU，每个GPU有完整模型副本 |
| **张量并行** | Tensor Parallelism | 将模型参数切分到多个GPU |
| **流水线并行** | Pipeline Parallelism | 将模型层切分到多个GPU |
| **梯度检查点** | Gradient Checkpointing | 用计算换显存的技术，不存储中间激活值 |
| **混合精度训练** | Mixed Precision Training | 同时使用FP16/FP32进行训练 |

---

## 2. 微调与对齐相关术语

### 2.1 微调技术

| 术语 | 英文 | 解释 |
|------|------|------|
| **有监督微调** | SFT (Supervised Fine-tuning) | 使用标注数据对模型进行微调 |
| **指令微调** | Instruction Tuning | 使用指令-响应对进行微调 |
| **参数高效微调** | PEFT (Parameter-Efficient Fine-tuning) | 只训练少量参数的微调方法 |
| **低秩适应** | LoRA (Low-Rank Adaptation) | 通过低秩矩阵进行微调的方法 |
| **量化LoRA** | QLoRA (Quantized LoRA) | 结合4-bit量化的LoRA |

### 2.2 对齐技术

| 术语 | 英文 | 解释 |
|------|------|------|
| **人类反馈强化学习** | RLHF (Reinforcement Learning from Human Feedback) | 使用人类反馈训练奖励模型，再用PPO优化策略 |
| **直接偏好优化** | DPO (Direct Preference Optimization) | 直接使用偏好数据优化策略模型 |
| **近端策略优化** | PPO (Proximal Policy Optimization) | 一种常用的强化学习算法 |
| **奖励模型** | Reward Model | 预测人类偏好的模型 |
| **价值模型** | Value Model | 预测状态价值的模型 |

### 2.3 模型集成

| 术语 | 英文 | 解释 |
|------|------|------|
| **模型合并** | Model Merging | 将多个模型的权重合并 |
| **任务向量** | Task Vector | 微调模型与基础模型的权重差 |
| **TIES合并** | TIES Merging | 修剪-选择符号-合并的模型合并方法 |
| **模型集成** | Model Ensemble | 组合多个模型的预测 |

---

## 3. Agent系统相关术语

### 3.1 基础概念

| 术语 | 英文 | 解释 |
|------|------|------|
| **智能体** | Agent | 能够感知环境并采取行动的实体 |
| **大语言模型智能体** | LLM Agent | 以LLM为核心的智能体 |
| **思维链** | CoT (Chain-of-Thought) | 让模型逐步推理的提示方法 |
| **推理-行动** | ReAct | 结合推理和行动的Agent模式 |
| **规划与执行** | Plan-and-Execute | 将任务分解为计划并执行的模式 |

### 3.2 组件

| 术语 | 英文 | 解释 |
|------|------|------|
| **工具调用** | Tool Calling / Function Calling | Agent调用外部工具的能力 |
| **记忆系统** | Memory System | Agent存储和检索信息的系统 |
| **短期记忆** | Short-term Memory | 保存当前对话上下文的记忆 |
| **长期记忆** | Long-term Memory | 保存历史知识的记忆 |
| **嵌入** | Embedding | 将文本转换为向量表示 |

### 3.3 多Agent系统

| 术语 | 英文 | 解释 |
|------|------|------|
| **多智能体系统** | Multi-Agent System | 多个Agent协作的系统 |
| **层级协作** | Hierarchical Collaboration | 有主从关系的协作模式 |
| **工作组协作** | Team Collaboration | 平等协商的协作模式 |
| **流水线协作** | Pipeline Collaboration | 顺序处理的协作模式 |
| **通信协议** | Communication Protocol | Agent之间通信的规则 |

---

## 4. 部署与优化相关术语

### 4.1 量化

| 术语 | 英文 | 解释 |
|------|------|------|
| **量化** | Quantization | 将模型权重从高精度降低到低精度 |
| **8-bit量化** | 8-bit Quantization | 将权重量化为8位整数 |
| **4-bit量化** | 4-bit Quantization | 将权重量化为4位整数 |
| **NF4** | Normalized Float 4 | 一种4-bit浮点格式 |
| **GPTQ** | GPTQ | 一种校准数据驱动的量化方法 |

### 4.2 推理优化

| 术语 | 英文 | 解释 |
|------|------|------|
| **推理** | Inference | 模型生成输出的过程 |
| **Flash Attention** | Flash Attention | 一种高效的注意力机制实现 |
| **连续批处理** | Continuous Batching | 动态处理请求的批处理方式 |
| **PagedAttention** | PagedAttention | vLLM使用的注意力机制 |
| **KV缓存** | KV Cache | 缓存键值对以加速自回归生成 |

### 4.3 部署

| 术语 | 英文 | 解释 |
|------|------|------|
| **推理引擎** | Inference Engine | 运行模型推理的软件 |
| **vLLM** | vLLM | 高性能LLM推理和服务库 |
| **TGI** | Text Generation Inference | HuggingFace的模型服务框架 |
| **TensorRT-LLM** | TensorRT-LLM | NVIDIA的LLM推理优化库 |
| **OpenAI API兼容** | OpenAI API Compatible | 与OpenAI API格式兼容的接口 |

---

## 常用缩写速查

| 缩写 | 全称 | 含义 |
|------|------|------|
| **API** | Application Programming Interface | 应用程序编程接口 |
| **BLEU** | Bilingual Evaluation Understudy | 机器翻译评估指标 |
| **CoT** | Chain-of-Thought | 思维链 |
| **DDP** | Distributed Data Parallel | 分布式数据并行 |
| **DPO** | Direct Preference Optimization | 直接偏好优化 |
| **FSDP** | Fully Sharded Data Parallel | 完全分片数据并行 |
| **GPT** | Generative Pre-trained Transformer | 生成式预训练Transformer |
| **GPU** | Graphics Processing Unit | 图形处理单元 |
| **JSON** | JavaScript Object Notation | 一种数据交换格式 |
| **KV** | Key-Value | 键值对 |
| **LM** | Language Model | 语言模型 |
| **LoRA** | Low-Rank Adaptation | 低秩适应 |
| **MLM** | Masked Language Model | 掩码语言模型 |
| **NLU** | Natural Language Understanding | 自然语言理解 |
| **PEFT** | Parameter-Efficient Fine-tuning | 参数高效微调 |
| **PPO** | Proximal Policy Optimization | 近端策略优化 |
| **QLoRA** | Quantized LoRA | 量化LoRA |
| **RAG** | Retrieval-Augmented Generation | 检索增强生成 |
| **ReAct** | Reasoning + Acting | 推理+行动 |
| **RL** | Reinforcement Learning | 强化学习 |
| **RLHF** | RL from Human Feedback | 人类反馈强化学习 |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | 文本摘要评估指标 |
| **SFT** | Supervised Fine-tuning | 有监督微调 |
| **TGI** | Text Generation Inference | 文本生成推理服务 |
| **TIES** | Trimming, Electing Signs, and Disjoint Merging | 一种模型合并方法 |
| **UI** | User Interface | 用户界面 |
| **URL** | Uniform Resource Locator | 统一资源定位符 |
| **vLLM** | Virtual Large Language Model | 高性能LLM推理库 |

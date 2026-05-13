# 附录 C：资源推荐

## 目录

1. [论文推荐](#1-论文推荐)
2. [开源项目](#2-开源项目)
3. [学习资源](#3-学习资源)
4. [工具与库](#4-工具与库)

---

## 1. 论文推荐

### 1.1 基础模型论文

| 论文 | 年份 | 机构 | 简介 |
|------|------|------|------|
| **Attention Is All You Need** | 2017 | Google | Transformer架构开山之作 |
| **BERT: Pre-training of Deep Bidirectional Transformers** | 2019 | Google | 双向预训练语言模型 |
| **GPT-3: Language Models are Few-Shot Learners** | 2020 | OpenAI | 大语言模型少样本学习 |
| **PaLM: Scaling Language Modeling with Pathways** | 2022 | Google | 路径语言模型 |
| **LLaMA: Open and Efficient Foundation Language Models** | 2023 | Meta | 开源大语言模型 |

### 1.2 预训练与微调论文

| 论文 | 年份 | 主题 |
|------|------|------|
| **LoRA: Low-Rank Adaptation of Large Language Models** | 2021 | 参数高效微调 |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | 2023 | 量化LoRA微调 |
| **Training language models to follow instructions** | 2022 | 指令微调 |
| **Scaling Instruction-Finetuned Language Models** | 2022 | 大规模指令微调 |

### 1.3 对齐论文

| 论文 | 年份 | 主题 |
|------|------|------|
| **Training a Helpful and Harmless Assistant with RLHF** | 2022 | RLHF |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | 2023 | DPO |
| **RRHF: Rank Responses to Align Language Models** | 2023 | 排名反馈 |
| **KTO: Knowledge Transfer from Optimizing Preferences** | 2024 | Kahneman-Tversky优化 |

### 1.4 Agent论文

| 论文 | 年份 | 主题 |
|------|------|------|
| **ReAct: Synergizing Reasoning and Acting** | 2023 | 推理-行动Agent |
| **WebGPT: Browser-assisted question-answering** | 2021 | 网页浏览Agent |
| **AutoGPT: An Autonomous GPT-4 Experiment** | 2023 | 自主Agent |
| **HuggingGPT: Solving AI Tasks with ChatGPT** | 2023 | 工具调用Agent |
| **Self-Refine: Iterative Refinement with Self-Feedback** | 2023 | 自我反思Agent |

### 1.5 推理优化论文

| 论文 | 年份 | 主题 |
|------|------|------|
| **FlashAttention: Fast and Memory-Efficient Exact Attention** | 2022 | Flash Attention |
| **FlashAttention-2: Faster Attention with Better Parallelism** | 2023 | Flash Attention v2 |
| **GPTQ: Accurate Post-Training Quantization** | 2022 | GPTQ量化 |
| **SqueezeLLM: Dense-and-Sparse Quantization** | 2023 | 稠密稀疏量化 |

---

## 2. 开源项目

### 2.1 模型仓库

| 项目 | 链接 | 简介 |
|------|------|------|
| **HuggingFace Hub** | https://huggingface.co | 最大的开源模型仓库 |
| **ModelScope** | https://modelscope.cn | 阿里的模型仓库 |
| **OpenLLaMA** | https://github.com/openlm-research/open_llama | 开源LLaMA复现 |
| **Falcon** | https://huggingface.co/tiiuae | TII的开源模型系列 |
| **Mistral** | https://mistral.ai | Mistral开源模型 |
| **Qwen** | https://huggingface.co/Qwen | 通义千问开源模型 |
| **Llama 3** | https://llama.meta.com | Meta的最新开源模型 |

### 2.2 训练框架

| 项目 | 链接 | 简介 |
|------|------|------|
| **Transformers** | https://github.com/huggingface/transformers | HuggingFace的Transformers库 |
| **PEFT** | https://github.com/huggingface/peft | 参数高效微调库 |
| **TRL** | https://github.com/huggingface/trl | Transformer强化学习库 |
| **Axolotl** | https://github.com/OpenAccess-AI-Collective/axolotl | 简化LLM训练的框架 |
| **LLaMA Factory** | https://github.com/hiyouga/LLaMA-Factory | LLaMA微调框架 |
| **Megatron-LM** | https://github.com/NVIDIA/Megatron-LM | NVIDIA的大规模训练框架 |
| **DeepSpeed** | https://github.com/microsoft/DeepSpeed | 微软的深度学习优化库 |

### 2.3 Agent框架

| 项目 | 链接 | 简介 |
|------|------|------|
| **LangChain** | https://github.com/langchain-ai/langchain | 最流行的LLM应用框架 |
| **AutoGen** | https://github.com/microsoft/autogen | 微软的多Agent框架 |
| **Semantic Kernel** | https://github.com/microsoft/semantic-kernel | 微软的语义内核 |
| **LlamaIndex** | https://github.com/run-llama/llama_index | 数据连接框架 |
| **CrewAI** | https://github.com/joaomdmoura/crewAI | 多Agent协作框架 |
| **LangGraph** | https://github.com/langchain-ai/langgraph | LangChain的状态图框架 |

### 2.4 推理服务

| 项目 | 链接 | 简介 |
|------|------|------|
| **vLLM** | https://github.com/vllm-project/vllm | 高性能LLM推理和服务 |
| **Text Generation Inference** | https://github.com/huggingface/text-generation-inference | HuggingFace的推理服务 |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA的LLM优化库 |
| **CTranslate2** | https://github.com/OpenNMT/CTranslate2 | 高效的Transformer推理引擎 |
| **Ollama** | https://github.com/ollama/ollama | 本地运行LLM的工具 |

### 2.5 量化工具

| 项目 | 链接 | 简介 |
|------|------|------|
| **AutoGPTQ** | https://github.com/AutoGPTQ/AutoGPTQ | GPTQ量化实现 |
| **ExLlama** | https://github.com/turboderp/exllama | 高效的4-bit推理 |
| **ExLlamaV2** | https://github.com/turboderp/exllamav2 | ExLlama的改进版 |
| **GPTQ-for-LLaMA** | https://github.com/oobabooga/GPTQ-for-LLaMA | LLaMA的GPTQ量化 |
| **bitsandbytes** | https://github.com/TimDettmers/bitsandbytes | 8-bit优化库 |

---

## 3. 学习资源

### 3.1 课程与教程

| 资源 | 链接 | 简介 |
|------|------|------|
| **HuggingFace Course** | https://huggingface.co/learn | 免费的NLP和LLM课程 |
| **Stanford CS229** | https://cs229.stanford.edu | 机器学习课程 |
| **Stanford CS230** | https://cs230.stanford.edu | 深度学习课程 |
| **Stanford CS224N** | https://web.stanford.edu/class/cs224n | NLP课程 |
| **Fast.ai** | https://course.fast.ai | 实用深度学习课程 |
| **Full Stack LLM Bootcamp** | https://fullstackdeeplearning.com/llm-bootcamp | LLM全栈课程 |

### 3.2 博客与社区

| 资源 | 链接 | 简介 |
|------|------|------|
| **Towards Data Science** | https://towardsdatascience.com | Medium上的DS博客 |
| **The Batch** | https://www.deeplearning.ai/the-batch | DeepLearning.AI的新闻通讯 |
| **Lil'Log** | https://lilianweng.github.io | Lilian Weng的技术博客 |
| **Jay Alammar's Blog** | https://jalammar.github.io | 图解ML的博客 |
| **HuggingFace Blog** | https://huggingface.co/blog | HuggingFace官方博客 |
| **Reddit r/MachineLearning** | https://www.reddit.com/r/MachineLearning | ML社区 |
| **Reddit r/LocalLLaMA** | https://www.reddit.com/r/LocalLLaMA | 本地LLM社区 |

### 3.3 视频资源

| 资源 | 链接 | 简介 |
|------|------|------|
| **3Blue1Brown** | https://www.3blue1brown.com | 可视化数学和ML |
| **Andrej Karpathy** | https://www.youtube.com/@AndrejKarpathy | 神经网络教学 |
| **Yannic Kilcher** | https://www.youtube.com/@YannicKilcher | 论文解读 |
| **HuggingFace YouTube** | https://www.youtube.com/@HuggingFace | HuggingFace官方视频 |
| **Coursera ML** | https://www.coursera.org/learn/machine-learning | Andrew Ng的ML课程 |

### 3.4 中文资源

| 资源 | 链接 | 简介 |
|------|------|------|
| **李沐老师视频** | https://space.bilibili.com/1567748478 | 动手学深度学习 |
| **张俊林博客** | https://weibo.com/zhangjunlin | NLP专家博客 |
| **DataWhale** | https://datawhale.club | 开源学习社区 |
| **HuggingFace中文** | https://huggingface.co/docs/transformers/zh | 中文文档 |
| **超神经** | https://www.jiqizhixin.com | AI资讯和教程 |

---

## 4. 工具与库

### 4.1 核心库

| 库 | 用途 | 安装 |
|----|------|------|
| **PyTorch** | 深度学习框架 | `pip install torch` |
| **Transformers** | 预训练模型 | `pip install transformers` |
| **Tokenizers** | 快速tokenization | `pip install tokenizers` |
| **Datasets** | 数据集处理 | `pip install datasets` |
| **Accelerate** | 分布式训练 | `pip install accelerate` |
| **PEFT** | 参数高效微调 | `pip install peft` |
| **TRL** | 强化学习 | `pip install trl` |
| **BitsAndBytes** | 8-bit优化 | `pip install bitsandbytes` |

### 4.2 向量数据库

| 数据库 | 用途 | 链接 |
|--------|------|------|
| **Chroma** | 轻量级向量数据库 | https://www.trychroma.com |
| **Milvus** | 企业级向量数据库 | https://milvus.io |
| **Qdrant** | 向量相似度搜索 | https://qdrant.tech |
| **Weaviate** | 向量搜索引擎 | https://weaviate.io |
| **pgvector** | PostgreSQL扩展 | https://github.com/pgvector/pgvector |
| **FAISS** | Facebook的向量搜索库 | https://github.com/facebookresearch/faiss |

### 4.3 开发工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **Jupyter Lab** | 交互式开发 | https://jupyter.org |
| **VS Code** | 代码编辑器 | https://code.visualstudio.com |
| **Git** | 版本控制 | https://git-scm.com |
| **Docker** | 容器化 | https://www.docker.com |
| **WandB** | 实验跟踪 | https://wandb.ai |
| **MLflow** | ML生命周期管理 | https://mlflow.org |

### 4.4 评估工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **LangSmith** | LLM应用追踪 | https://www.langchain.com/langsmith |
| **LangFuse** | LLM工程平台 | https://langfuse.com |
| **Helm** | LLM综合评估 | https://crfm.stanford.edu/helm |
| **Eleuther LM Evaluation Harness** | 模型评估 | https://github.com/EleutherAI/lm-evaluation-harness |
| **HuggingFace Evaluate** | 评估库 | https://huggingface.co/docs/evaluate |

---

## 快速开始建议

对于初学者，建议按以下顺序学习：

1. **基础知识** - 学习Python、PyTorch、深度学习基础
2. **HuggingFace生态** - 学习Transformers、Datasets等库
3. **微调实践** - 使用LoRA微调一个小型模型
4. **Agent开发** - 使用LangChain构建简单的Agent
5. **进阶优化** - 学习量化、推理优化等技术
6. **大规模训练** - 学习分布式训练、DeepSpeed等

对于进阶学习者，可以关注最新论文和开源项目，参与社区讨论和贡献。

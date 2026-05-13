# LLM服务与API选型

> 主流LLM供应商与私有化部署方案对比

## 1. LLM服务类型

### 1.1 服务类型对比

| 类型 | 代表 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **闭源API** | OpenAI/Anthropic | 效果好、无需运维 | 数据外泄风险、成本高 | 通用任务、快速启动 |
| **开源API** | 阿里云/DeepSeek | 中文好、性价比高 | 效果略逊于顶级模型 | 中文场景、成本敏感 |
| **私有化部署** | vLLM/自部署 | 数据安全、可控 | 运维成本高 | 敏感数据、高频调用 |
| **混合模式** | 网关路由 | 灵活、成本优化 | 架构复杂 | 企业级生产 |

### 1.2 选型决策树

```
数据敏感度?
├── 高度敏感（金融/医疗/政务）
│   └── 私有化部署（Qwen-72B自托管）
├── 中度敏感（企业内部）
│   └── 国产API（阿里云/智谱）
└── 一般（公开信息）
    ├── 追求效果 → OpenAI/Anthropic
    └── 追求性价比 → DeepSeek/阿里云

调用频率?
├── 高频（>100万Token/天）
│   └── 私有化部署或混合模式
└── 低频
    └── API调用
```

---

## 2. 闭源API服务

### 2.1 OpenAI

**模型矩阵：**

| 模型 | 上下文 | 特点 | 输入价格 | 输出价格 |
|------|--------|------|----------|----------|
| GPT-4o | 128K | 多模态、最强通用 | $2.5/1M | $10/1M |
| GPT-4o-mini | 128K | 轻量、便宜 | $0.15/1M | $0.6/1M |
| GPT-4-turbo | 128K | 长上下文 | $10/1M | $30/1M |
| o1-preview | 128K | 推理模型 | $15/1M | $60/1M |
| o1-mini | 128K | 轻量推理 | $3/1M | $12/1M |

**核心能力：**
- Function Calling（工具调用）
- JSON Mode（结构化输出）
- Vision（图像理解）
- Assistants API（对话管理）

**适用场景：**
- 复杂推理任务
- 多模态应用
- 快速原型开发
- 国际化应用

---

### 2.2 Anthropic Claude

**模型矩阵：**

| 模型 | 上下文 | 特点 | 输入价格 | 输出价格 |
|------|--------|------|----------|----------|
| Claude-3.5-Sonnet | 200K | 编程强、推理好 | $3/1M | $15/1M |
| Claude-3.5-Haiku | 200K | 快速、便宜 | $0.25/1M | $1.25/1M |
| Claude-3-Opus | 200K | 最强推理 | $15/1M | $75/1M |

**核心优势：**
- 超长上下文（200K）
- 代码能力突出
- 安全性高（ Constitutional AI）
- 幻觉率低

**适用场景：**
- 代码生成与审查
- 长文档分析
- 需要高安全性的场景

---

### 2.3 Google Gemini

**模型矩阵：**

| 模型 | 上下文 | 特点 | 价格 |
|------|--------|------|------|
| Gemini-1.5-Pro | 1M-2M | 超长上下文 | $3.5/1M |
| Gemini-1.5-Flash | 1M | 快速、便宜 | $0.35/1M |
| Gemini-1.0-Pro | 32K | 通用 | $0.5/1M |

**核心优势：**
- 业界最长上下文（1M+ tokens）
- 视频理解能力
- Google生态集成

**适用场景：**
- 超长文档/视频分析
- Google Cloud用户

---

## 3. 国产API服务

### 3.1 阿里云通义千问

**模型矩阵：**

| 模型 | 上下文 | 特点 | 价格 |
|------|--------|------|------|
| qwen-max | 32K | 最强中文 | ¥40/1K tokens |
| qwen-plus | 128K | 平衡选择 | ¥8/1K tokens |
| qwen-turbo | 8K | 快速便宜 | ¥2/1K tokens |
| qwen-coder | 128K | 代码专用 | ¥4/1K tokens |
| qwen-vl | 32K | 多模态 | ¥20/1K tokens |

**核心优势：**
- 中文理解能力强
- 全尺寸覆盖（0.5B-110B）
- 开源可私有化部署
- 阿里云生态集成

**适用场景：**
- 中文为主的应用
- 阿里云用户
- 需要私有化部署

---

### 3.2 DeepSeek

**模型矩阵：**

| 模型 | 上下文 | 特点 | 价格 |
|------|--------|------|------|
| DeepSeek-V3 | 64K | 性价比之王 | ¥2/1M tokens |
| DeepSeek-Coder | 64K | 代码专用 | ¥2/1M tokens |
| DeepSeek-MoE | 64K | 混合专家 | ¥2/1M tokens |

**核心优势：**
- 极致性价比
- 开源可私有化
- 代码能力优秀

**适用场景：**
- 成本敏感场景
- 高频调用
- 代码生成

---

### 3.3 智谱AI（GLM）

**模型矩阵：**

| 模型 | 上下文 | 特点 | 价格 |
|------|--------|------|------|
| GLM-4 | 128K | 旗舰模型 | ¥100/1M tokens |
| GLM-4-Air | 128K | 轻量快速 | ¥1/1M tokens |
| GLM-4-Flash | 128K | 免费版 | 免费 |

**核心优势：**
- 清华背景
- Function Call支持好
- 有免费额度

---

### 3.4 月之暗面（Moonshot）

**模型矩阵：**

| 模型 | 上下文 | 特点 | 价格 |
|------|--------|------|------|
| Kimi-K1.5 | 200K | 长上下文 | ¥12/1M tokens |
| Moonshot-v1-8k | 8K | 轻量 | ¥6/1M tokens |

**核心优势：**
- 长上下文（200K）
- 文档理解能力强

---

## 4. 私有化部署

### 4.1 部署方案对比

| 方案 | 资源需求 | 难度 | 适用场景 |
|------|----------|------|----------|
| **vLLM** | 中 | 低 | 生产推荐 |
| **TensorRT-LLM** | 高 | 高 | 极致性能 |
| **llama.cpp** | 低 | 低 | 端侧/CPU |
| **TGI** | 中 | 低 | 快速原型 |
| **Ollama** | 低 | 极低 | 开发测试 |

### 4.2 vLLM部署

**核心特性：**
- PagedAttention（高效KV Cache管理）
- Continuous Batching（高吞吐）
- 多GPU张量并行
- OpenAI兼容API

**部署示例：**
```bash
# 单卡部署
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-72B \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-num-seqs 256

# Docker部署
docker run --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen-72B \
    --quantization awq
```

**性能优化参数：**
```bash
--max-num-batched-tokens 4096  # 最大批处理token数
--max-num-seqs 256             # 最大并发序列数
--gpu-memory-utilization 0.9   # GPU显存利用率
--swap-space 4                 # CPU swap空间(GB)
```

---

### 4.3 Ollama（开发测试）

**一键运行本地模型：**
```bash
# 安装
curl -fsSL https://ollama.com/install.sh | sh

# 运行模型
ollama run qwen:72b
ollama run llama3:70b
ollama run deepseek-coder:33b

# REST API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen:72b",
  "prompt": "你好"
}'
```

**适用场景：**
- 本地开发测试
- 离线环境
- 快速验证模型效果

---

## 5. API调用最佳实践

### 5.1 多供应商封装

```python
from abc import ABC, abstractmethod
from typing import Iterator

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        pass
    
    @abstractmethod
    def stream_chat(self, messages: list, **kwargs) -> Iterator[str]:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o"),
            messages=messages,
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content

class QwenProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "qwen-max"),
            messages=messages
        )
        return response.choices[0].message.content

# 使用
providers = {
    "openai": OpenAIProvider(os.getenv("OPENAI_KEY")),
    "qwen": QwenProvider(os.getenv("DASHSCOPE_KEY"))
}

def chat(provider_name: str, messages: list):
    return providers[provider_name].chat(messages)
```

### 5.2 成本控制策略

```python
class CostOptimizedLLM:
    """成本优化的LLM调用"""
    
    def __init__(self):
        self.cheap_model = "gpt-4o-mini"      # $0.15/1M
        self.expensive_model = "gpt-4o"       # $2.5/1M
    
    def chat(self, messages, complexity="auto"):
        # 自动判断复杂度
        if complexity == "auto":
            complexity = self._estimate_complexity(messages)
        
        # 简单任务用便宜模型
        if complexity == "simple":
            return self._call_model(self.cheap_model, messages)
        
        # 复杂任务用强模型
        return self._call_model(self.expensive_model, messages)
    
    def _estimate_complexity(self, messages):
        last_msg = messages[-1]["content"]
        
        # 简单判断规则
        if len(last_msg) < 100 and "?" not in last_msg:
            return "simple"
        if any(kw in last_msg for kw in ["分析", "对比", "详细", "复杂"]):
            return "complex"
        
        return "medium"
```

---

## 6. 开发者Checklist

### 6.1 LLM选型Checklist

- [ ] 评估数据安全要求
- [ ] 评估预算限制
- [ ] 评估调用频率
- [ ] 测试候选模型效果
- [ ] 评估延迟要求
- [ ] 检查供应商稳定性
- [ ] 准备备选方案
- [ ] 实现成本监控

### 6.2 常见陷阱

**陷阱1：单供应商依赖**
- 问题：供应商故障导致服务中断
- 解决：多供应商备份

**陷阱2：成本失控**
- 问题：Token用量暴涨
- 解决：预算告警+模型路由

**陷阱3：忽视延迟**
- 问题：API响应慢影响体验
- 解决：流式输出+降级策略

---

*最后更新：2026-05-07*

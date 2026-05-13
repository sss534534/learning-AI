# 第三章：大模型评估与部署

> 模型训练完成后，需要进行全面评估以确保其性能和可靠性，然后通过优化技术和部署框架将其投入生产环境。本章将深入讲解评估指标、量化技术、推理优化以及服务部署方案。

## 目录

1. [评估指标](#1-评估指标)
2. [量化技术](#2-量化技术)
3. [推理优化](#3-推理优化)
4. [服务部署](#4-服务部署)

---

## 1. 评估指标

### 1.1 自动评估指标

#### 1.1.1 Perplexity（困惑度）

Perplexity衡量模型对文本的预测能力，值越小越好。

$$
\text{PPL} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, \ldots, w_{i-1})}
$$

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, texts, device="cuda"):
    """
    计算模型在文本上的困惑度
    """
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            labels = inputs.input_ids.clone()
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # 计算有效token数（忽略padding）
            num_tokens = (labels != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

# 使用示例
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

test_texts = [
    "大语言模型在自然语言处理领域取得了重大突破。",
    "机器学习算法能够从数据中学习模式和规律。"
]

ppl = compute_perplexity(model, tokenizer, test_texts)
print(f"Perplexity: {ppl:.2f}")
```

#### 1.1.2 BLEU（双语评估辅助工具）

BLEU主要用于机器翻译评估，衡量生成文本与参考文本的n-gram重叠度。

$$
\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

def compute_bleu(reference, hypothesis, n=4):
    """
    计算BLEU分数
    reference: 参考文本列表，每个参考文本是token列表
    hypothesis: 生成文本，token列表
    """
    smoothie = SmoothingFunction().method4
    weights = tuple([1.0/n] * n)
    return sentence_bleu(reference, hypothesis, weights=weights, smoothing_function=smoothie)

# 使用示例
reference = [["大", "语言", "模型", "在", "自然语言处理", "领域", "取得", "了", "重大", "突破"]]
hypothesis = ["大", "语言", "模型", "在", "NLP", "领域", "取得", "了", "重要", "进展"]

bleu_score = compute_bleu(reference, hypothesis)
print(f"BLEU-4: {bleu_score:.4f}")
```

#### 1.1.3 ROUGE（面向召回的摘要评估辅助工具）

ROUGE用于文本摘要评估，包括ROUGE-1、ROUGE-2、ROUGE-L等。

```python
from rouge import Rouge

def compute_rouge(reference, hypothesis):
    """
    计算ROUGE分数
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

# 使用示例
reference = "大语言模型在自然语言处理领域取得了重大突破，为各种应用提供了强大的技术支持。"
hypothesis = "大语言模型在NLP领域取得了重要进展，为多种应用提供了有力支撑。"

rouge_scores = compute_rouge(reference, hypothesis)
print("ROUGE Scores:")
print(f"  ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"  ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
print(f"  ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
```

#### 1.1.4 METEOR

METEOR结合了精确匹配、词形还原和同义词匹配。

```python
from nltk.translate import meteor_score

def compute_meteor(references, hypothesis):
    """
    计算METEOR分数
    references: 参考文本列表
    hypothesis: 生成文本
    """
    return meteor_score.meteor_score(references, hypothesis)

# 使用示例
references = ["大语言模型在自然语言处理领域取得了重大突破"]
hypothesis = "大语言模型在NLP领域取得了重要进展"

meteor = compute_meteor(references, hypothesis)
print(f"METEOR: {meteor:.4f}")
```

#### 1.1.5 BERTScore

BERTScore使用预训练语言模型计算语义相似度。

```python
from bert_score import score

def compute_bertscore(references, candidates, model_type="bert-base-chinese"):
    """
    计算BERTScore
    """
    P, R, F1 = score(candidates, references, model_type=model_type, lang="zh", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# 使用示例
references = ["大语言模型在自然语言处理领域取得了重大突破"]
candidates = ["大语言模型在NLP领域取得了重要进展"]

bertscore = compute_bertscore(references, candidates)
print(f"BERTScore F1: {bertscore['f1']:.4f}")
```

### 1.2 人类评估

#### 1.2.1 评估维度设计

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class EvaluationDimension:
    name: str
    description: str
    scale: tuple = (1, 5)  # 评分范围

@dataclass
class EvaluationResult:
    prompt: str
    model_output: str
    dimensions: dict
    overall_score: float
    comments: Optional[str] = None

class HumanEvaluationFramework:
    """人类评估框架"""
    
    def __init__(self):
        self.dimensions = [
            EvaluationDimension(
                name="有用性",
                description="输出是否有用，是否回答了问题"
            ),
            EvaluationDimension(
                name="准确性",
                description="输出是否准确，有无错误信息"
            ),
            EvaluationDimension(
                name="连贯性",
                description="输出是否逻辑连贯，语言流畅"
            ),
            EvaluationDimension(
                name="完整性",
                description="输出是否完整，是否涵盖所有要点"
            ),
            EvaluationDimension(
                name="无害性",
                description="输出是否安全，有无有害内容"
            )
        ]
    
    def create_evaluation_task(self, prompt: str, model_output: str) -> dict:
        """创建评估任务"""
        return {
            "prompt": prompt,
            "model_output": model_output,
            "dimensions": [
                {
                    "name": dim.name,
                    "description": dim.description,
                    "scale": dim.scale
                }
                for dim in self.dimensions
            ]
        }
    
    def save_evaluation_result(self, result: EvaluationResult, filepath: str):
        """保存评估结果"""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "prompt": result.prompt,
                "model_output": result.model_output,
                "dimensions": result.dimensions,
                "overall_score": result.overall_score,
                "comments": result.comments
            }, ensure_ascii=False) + "\n")
    
    def analyze_results(self, results: List[EvaluationResult]) -> dict:
        """分析评估结果"""
        dimension_scores = {dim.name: [] for dim in self.dimensions}
        overall_scores = []
        
        for result in results:
            overall_scores.append(result.overall_score)
            for dim_name, score in result.dimensions.items():
                dimension_scores[dim_name].append(score)
        
        return {
            "overall_avg": sum(overall_scores) / len(overall_scores),
            "dimensions": {
                dim_name: sum(scores) / len(scores)
                for dim_name, scores in dimension_scores.items()
            }
        }
```

#### 1.2.2 评估示例

```python
# 创建评估框架
evaluator = HumanEvaluationFramework()

# 示例评估结果
result = EvaluationResult(
    prompt="解释什么是大语言模型",
    model_output="大语言模型是一种基于深度学习的自然语言处理模型...",
    dimensions={
        "有用性": 5,
        "准确性": 4,
        "连贯性": 5,
        "完整性": 4,
        "无害性": 5
    },
    overall_score=4.6,
    comments="回答清晰准确，但可以增加更多应用场景"
)

# 保存结果
evaluator.save_evaluation_result(result, "evaluation_results.jsonl")
```

### 1.3 评估指标对比

| 指标 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **Perplexity** | 语言建模 | 计算简单，客观 | 不能直接衡量下游任务性能 |
| **BLEU** | 机器翻译 | 广泛使用，计算快 | 对词序敏感，不考虑语义 |
| **ROUGE** | 文本摘要 | 关注召回率 | 不考虑精确匹配 |
| **METEOR** | 机器翻译 | 考虑同义词 | 计算复杂 |
| **BERTScore** | 通用生成 | 考虑语义相似度 | 计算成本高 |
| **人类评估** | 所有任务 | 最可靠 | 成本高，主观性强 |

---

## 2. 量化技术

### 2.1 8-bit量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_8bit_model(model_name):
    """
    加载8-bit量化模型
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # 离群值阈值
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 使用示例
model_name = "meta-llama/Llama-2-7b-hf"
model_8bit, tokenizer = load_8bit_model(model_name)

# 推理
prompt = "解释什么是机器学习："
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model_8bit.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2.2 4-bit量化

#### 2.2.1 NF4量化（Normalized Float 4）

```python
def load_4bit_model(model_name, quant_type="nf4", compute_dtype=torch.bfloat16):
    """
    加载4-bit量化模型
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # 双重量化
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# 使用示例
model_4bit, tokenizer = load_4bit_model(model_name)

# 显存对比
print(f"Model size (4-bit): {model_4bit.get_memory_footprint() / 1024**3:.2f} GB")
```

#### 2.2.2 GPTQ量化

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np

def quantize_with_gptq(model_name, output_dir, bits=4, group_size=128):
    """
    使用GPTQ量化模型
    """
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        damp_percent=0.1,
    )
    
    # 加载模型
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 准备校准数据
    examples = [
        "大语言模型是人工智能领域的重要突破。",
        "机器学习算法能够从数据中学习规律。",
        "深度学习在图像识别和自然语言处理方面表现出色。"
    ]
    
    # 量化
    model.quantize(
        tokenizer(examples, return_tensors="pt", padding=True, truncation=True),
        batch_size=1
    )
    
    # 保存量化模型
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def load_gptq_model(model_dir):
    """
    加载GPTQ量化模型
    """
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device_map="auto",
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# 使用示例
output_dir = "./llama-7b-gptq-4bit"
quantize_with_gptq(model_name, output_dir)
model_gptq, tokenizer = load_gptq_model(output_dir)
```

### 2.3 量化技术对比

| 方法 | 精度 | 显存节省 | 推理速度 | 质量损失 |
|------|------|----------|----------|----------|
| **FP16** | 16-bit | ~50% | 快 | 无 |
| **8-bit** | 8-bit | ~75% | 快 | 很小 |
| **4-bit (NF4)** | 4-bit | ~87.5% | 中 | 小 |
| **GPTQ** | 4-bit | ~87.5% | 快 | 很小 |
| **AWQ** | 4-bit | ~87.5% | 很快 | 很小 |

---

## 3. 推理优化

### 3.1 Flash Attention

```python
from flash_attn import flash_attn_func
import torch

class FlashAttentionModel(torch.nn.Module):
    """使用Flash Attention的模型"""
    
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Flash Attention
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            softmax_scale=None
        )
        
        # 重塑回原始格式
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.o_proj(attn_output)
        return output

# 使用示例
model = FlashAttentionModel(d_model=512, num_heads=8).cuda()
x = torch.randn(2, 128, 512).cuda()  # [batch, seq_len, d_model]
output = model(x)
print(f"Output shape: {output.shape}")
```

### 3.2 vLLM（推理优化引擎）

```python
from vllm import LLM, SamplingParams

def vllm_inference(model_name, prompts):
    """
    使用vLLM进行推理
    """
    # 创建LLM实例
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # 张量并行大小
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    
    # 批量推理
    outputs = llm.generate(prompts, sampling_params)
    
    # 处理输出
    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append({"prompt": prompt, "response": generated_text})
    
    return results

# 使用示例
model_name = "meta-llama/Llama-2-7b-hf"
prompts = [
    "解释什么是大语言模型：",
    "机器学习的主要算法有哪些："
]

results = vllm_inference(model_name, prompts)
for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}\n")
```

### 3.3 TensorRT-LLM

```python
import tensorrt_llm
from tensorrt_llm import ModelConfig, BuildConfig, RuntimeConfig
from tensorrt_llm.builder import Builder
from tensorrt_llm.models import LLaMAForCausalLM

def build_tensorrt_llm_engine(model_name, output_dir, max_batch_size=8, max_input_len=1024, max_output_len=256):
    """
    构建TensorRT-LLM引擎
    """
    # 模型配置
    model_config = ModelConfig.from_pretrained(model_name)
    
    # 构建配置
    build_config = BuildConfig()
    build_config.max_batch_size = max_batch_size
    build_config.max_input_len = max_input_len
    build_config.max_output_len = max_output_len
    build_config.plugin_config.set_gpt_attention_plugin(dtype="float16")
    build_config.plugin_config.set_gemm_plugin(dtype="float16")
    build_config.plugin_config.set_rmsnorm_plugin(dtype="float16")
    
    # 创建模型
    model = LLaMAForCausalLM.from_pretrained(model_name)
    
    # 构建引擎
    builder = Builder()
    engine = builder.build_engine(model, model_config, build_config)
    
    # 保存引擎
    tensorrt_llm.save(engine, output_dir)
    
    return engine

def run_tensorrt_llm_inference(engine_dir, tokenizer, prompts):
    """
    使用TensorRT-LLM引擎进行推理
    """
    # 加载引擎
    engine = tensorrt_llm.load(engine_dir)
    
    # 创建运行时
    runtime = tensorrt_llm.Runtime(engine)
    
    # 运行时配置
    runtime_config = RuntimeConfig()
    runtime_config.max_batch_size = 8
    
    # 准备输入
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # 推理
    outputs = runtime.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        temperature=0.7,
    )
    
    # 解码
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results
```

### 3.4 推理优化技术对比

| 技术 | 速度提升 | 显存优化 | 兼容性 | 复杂度 |
|------|----------|----------|--------|--------|
| **Flash Attention** | 2-4x | 显著 | 好 | 低 |
| **vLLM** | 10-20x | 高 | 中 | 中 |
| **TensorRT-LLM** | 10-30x | 高 | 低 | 高 |
| **CTranslate2** | 5-10x | 中 | 好 | 中 |
| **ExLlama** | 5-15x | 高 | 低 | 中 |

---

## 4. 服务部署

### 4.1 FastAPI轻量级部署

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="LLM Inference Service", version="1.0.0")

# 全局变量存储模型
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="输入提示")
    max_tokens: int = Field(default=256, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p采样")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k采样")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="重复惩罚")

class InferenceResponse(BaseModel):
    prompt: str
    response: str
    inference_time: float

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, tokenizer
    
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

@app.post("/v1/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    """生成文本"""
    import time
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    # 准备输入
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    inference_time = time.time() - start_time
    
    return InferenceResponse(
        prompt=request.prompt,
        response=response,
        inference_time=inference_time
    )

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4.2 Text Generation Inference (TGI)

```python
# TGI配置文件（config.yaml）
"""
model: meta-llama/Llama-2-7b-hf
max_input_length: 1024
max_total_tokens: 2048
max_batch_prefill_tokens: 4096
max_batch_total_tokens: 4096
hostname: 0.0.0.0
port: 8080
num_shard: 1
quantize: bitsandbytes
dtype: float16
"""

import requests
import json

class TGIClient:
    """TGI客户端"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def generate(self, prompt, **kwargs):
        """
        调用TGI服务生成文本
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "do_sample": kwargs.get("do_sample", True),
                "return_full_text": kwargs.get("return_full_text", False)
            }
        }
        
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"TGI request failed: {response.text}")
        
        result = response.json()
        return result["generated_text"]
    
    def generate_stream(self, prompt, **kwargs):
        """
        流式生成
        """
        url = f"{self.base_url}/generate_stream"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "do_sample": kwargs.get("do_sample", True)
            }
        }
        
        response = requests.post(url, json=payload, stream=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "token" in data:
                        yield data["token"]["text"]

# 使用示例
client = TGIClient("http://localhost:8080")

# 普通生成
response = client.generate("解释什么是大语言模型：", max_tokens=256)
print(response)

# 流式生成
print("Streaming response:")
for token in client.generate_stream("解释什么是机器学习："):
    print(token, end="", flush=True)
```

### 4.3 vLLM服务部署

```python
# 使用vLLM启动API服务
"""
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
"""

from openai import OpenAI

class VLLMClient:
    """vLLM OpenAI兼容客户端"""
    
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy"  # vLLM不需要真实API key
        )
    
    def chat_completion(self, messages, model="gpt-3.5-turbo", **kwargs):
        """
        聊天补全
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            stream=kwargs.get("stream", False)
        )
        
        if kwargs.get("stream", False):
            return (chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content)
        else:
            return response.choices[0].message.content

# 使用示例
client = VLLMClient()

# 聊天
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "解释什么是大语言模型？"}
]

response = client.chat_completion(messages)
print(response)

# 流式聊天
print("\nStreaming chat:")
for token in client.chat_completion(messages, stream=True):
    print(token, end="", flush=True)
```

### 4.4 部署方案对比

| 方案 | 易用性 | 性能 | 扩展性 | 特性丰富度 |
|------|--------|------|--------|------------|
| **FastAPI** | 高 | 中 | 中 | 低 |
| **TGI** | 中 | 高 | 高 | 中 |
| **vLLM** | 中 | 很高 | 高 | 中 |
| **Ray Serve** | 中 | 高 | 很高 | 高 |
| **KServe** | 低 | 高 | 很高 | 高 |

---

## 本章小结

大模型评估与部署是将训练好的模型转化为生产服务的关键环节：

1. **评估指标** 涵盖自动指标（Perplexity、BLEU、ROUGE、BERTScore）和人类评估，全方位衡量模型性能
2. **量化技术**（8-bit、4-bit、GPTQ）大幅降低显存需求，使大模型能在普通硬件上运行
3. **推理优化**（Flash Attention、vLLM、TensorRT-LLM）显著提升推理速度和吞吐量
4. **服务部署**（FastAPI、TGI、vLLM）提供从简单到复杂的多种部署方案，满足不同场景需求

**下一章：** 我们将学习Agent架构，包括工具调用、规划、记忆等核心组件。

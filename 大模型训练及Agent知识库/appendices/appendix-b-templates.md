# 附录 B：代码模板

## 目录

1. [预训练模板](#1-预训练模板)
2. [微调与对齐模板](#2-微调与对齐模板)
3. [Agent系统模板](#3-agent系统模板)
4. [部署模板](#4-部署模板)

---

## 元数据
- **难度**: ⭐
- **关联文件**: ../chapters/ch01-pretraining.md, ../chapters/ch02-finetuning-alignment.md, ../chapters/ch03-evaluation-deployment.md, ../chapters/ch04-agent-architecture.md
- **最后更新**: 2026-06-12
---

## 1. 预训练模板

### 1.1 基础训练循环模板

```python
"""
基础预训练循环模板
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreTrainer:
    """预训练器"""
    
    def __init__(
        self,
        model_name: str,
        train_dataset,
        val_dataset=None,
        config: dict = None
    ):
        """
        初始化预训练器
        
        Args:
            model_name: 模型名称或路径
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 配置字典
        """
        self.config = config or self._default_config()
        
        # 加载模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 优化器和调度器
        self.optimizer = None
        self.lr_scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
    
    @staticmethod
    def _default_config():
        """默认配置"""
        return {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "num_train_epochs": 3,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "output_dir": "./output",
            "fp16": False
        }
    
    def setup_optimizers(self):
        """设置优化器和调度器"""
        # 优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"]
        )
        
        # 学习率调度器
        num_update_steps_per_epoch = len(self.train_dataset) // (
            self.config["batch_size"] * self.config["gradient_accumulation_steps"]
        )
        max_train_steps = self.config["num_train_epochs"] * num_update_steps_per_epoch
        
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=max_train_steps
        )
    
    def train(self):
        """训练循环"""
        self.setup_optimizers()
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True
        )
        
        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if self.config["fp16"] else None
        
        logger.info("开始训练...")
        
        for epoch in range(self.config["num_train_epochs"]):
            self.epoch = epoch
            self.model.train()
            
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                with torch.cuda.amp.autocast(enabled=self.config["fp16"]):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config["gradient_accumulation_steps"]
                
                # 反向传播
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                    if scaler:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.config["logging_steps"] == 0:
                        avg_loss = total_loss / self.config["logging_steps"]
                        logger.info(
                            f"Step {self.global_step}, Loss: {avg_loss:.4f}"
                        )
                        total_loss = 0
                    
                    # 保存
                    if self.global_step % self.config["save_steps"] == 0:
                        self.save_checkpoint()
                
                progress_bar.set_postfix({"loss": loss.item()})
            
            # 每个epoch结束后验证
            if self.val_dataset is not None:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        logger.info("训练完成！")
    
    def validate(self):
        """验证"""
        self.model.eval()
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"]
        )
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_dataloader)
    
    def save_checkpoint(self, suffix: str = None):
        """保存检查点"""
        if suffix:
            save_dir = f"{self.config['output_dir']}/checkpoint_{suffix}"
        else:
            save_dir = f"{self.config['output_dir']}/checkpoint_{self.global_step}"
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        logger.info(f"模型已保存到: {save_dir}")
```

---

### 1.2 数据处理模板

```python
"""
数据处理模板
"""
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import random


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, tokenizer_name: str, max_seq_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
    
    def load_and_process_dataset(self, dataset_path: str, **kwargs) -> Dataset:
        """
        加载并处理数据集
        
        Args:
            dataset_path: 数据集路径或名称
            **kwargs: 其他参数
            
        Returns:
            处理后的数据集
        """
        # 加载数据集
        dataset = load_dataset(dataset_path, **kwargs)
        
        # 处理
        processed_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        # 这里需要根据具体数据集格式修改
        texts = examples["text"]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            return_overflowing_tokens=False,
            padding="max_length"
        )
        
        # 因果语言模型的labels就是input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def filter_by_length(self, dataset: Dataset, min_len: int = 100, max_len: int = None):
        """按长度过滤"""
        max_len = max_len or self.max_seq_length
        
        def filter_fn(example):
            return min_len <= len(example["input_ids"]) <= max_len
        
        return dataset.filter(filter_fn)
    
    def deduplicate(self, dataset: Dataset, key: str = "text"):
        """去重"""
        seen = set()
        
        def filter_fn(example):
            text = example[key]
            if text in seen:
                return False
            seen.add(text)
            return True
        
        return dataset.filter(filter_fn)
    
    def train_val_split(self, dataset: Dataset, val_ratio: float = 0.1):
        """训练验证集划分"""
        return dataset.train_test_split(test_size=val_ratio)
```

---

## 2. 微调与对齐模板

### 2.1 LoRA微调模板

```python
"""
LoRA微调模板
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset


def train_with_lora(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./lora_output",
    lora_r: int = 16,
    lora_alpha: int = 32,
    **kwargs
):
    """
    使用LoRA微调模型
    
    Args:
        model_name: 基础模型名称
        dataset_name: 数据集名称
        output_dir: 输出目录
        lora_r: LoRA秩
        lora_alpha: LoRA alpha
        **kwargs: 其他参数
    """
    # 4-bit量化配置（可选）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 加载数据集
    dataset = load_dataset(dataset_name, split="train")
    
    # 格式化函数（根据数据集修改）
    def formatting_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction: {example['instruction'][i]}\n"
            if example['input'][i]:
                text += f"### Input: {example['input'][i]}\n"
            text += f"### Response: {example['output'][i]}"
            output_texts.append(text)
        return output_texts
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        **kwargs
    )
    
    # 创建Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=2048
    )
    
    # 训练
    trainer.train()
    
    # 保存
    trainer.model.save_pretrained(output_dir)
    
    return trainer.model


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    """
    合并LoRA权重
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        output_path: 输出路径
    """
    from peft import PeftModel
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并
    model = model.merge_and_unload()
    
    # 保存
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
```

---

## 3. Agent系统模板

### 3.1 基础Agent模板

```python
"""
基础Agent模板
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class Message:
    """消息"""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_responses: Optional[List[Dict]] = None


class BaseAgent:
    """基础Agent"""
    
    def __init__(
        self,
        system_prompt: str = None,
        tools: List[Dict] = None
    ):
        """
        初始化Agent
        
        Args:
            system_prompt: 系统提示词
            tools: 可用工具列表
        """
        self.system_prompt = system_prompt or "你是一个有用的AI助手。"
        self.tools = tools or []
        self.messages: List[Message] = []
        
        # 初始化系统消息
        if self.system_prompt:
            self.messages.append(Message(
                role="system",
                content=self.system_prompt
            ))
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str, tool_calls: List[Dict] = None):
        """添加助手消息"""
        self.messages.append(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        ))
    
    def add_tool_message(self, tool_name: str, tool_id: str, result: Any):
        """添加工具结果消息"""
        tool_response = {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "result": result
        }
        self.messages.append(Message(
            role="tool",
            content=json.dumps(result, ensure_ascii=False),
            tool_responses=[tool_response]
        ))
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        history = []
        for msg in self.messages:
            item = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = msg.tool_calls
            history.append(item)
        return history
    
    def reset(self):
        """重置对话"""
        self.messages = []
        if self.system_prompt:
            self.messages.append(Message(
                role="system",
                content=self.system_prompt
            ))
    
    def think(self, user_input: str) -> str:
        """
        思考并生成响应（需要子类实现）
        
        Args:
            user_input: 用户输入
            
        Returns:
            Agent响应
        """
        raise NotImplementedError
```

---

### 3.2 ReAct Agent模板

```python
"""
ReAct Agent模板
"""
import re
from typing import Dict, Any, Callable, List


class ReActAgent:
    """推理-行动Agent"""
    
    def __init__(
        self,
        llm: Callable,
        tools: Dict[str, Callable],
        system_prompt: str = None
    ):
        """
        初始化ReAct Agent
        
        Args:
            llm: LLM调用函数
            tools: 工具字典 {name: function}
            system_prompt: 系统提示词
        """
        self.llm = llm
        self.tools = tools
        
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or '无描述'}"
            for name, func in self.tools.items()
        ])
        
        return f"""你是一个能够使用工具的AI助手。

可用工具：
{tool_descriptions}

请使用以下格式进行思考和行动：
Thought: 你的思考
Action: 工具名(参数)
Observation: 工具执行结果
...（重复思考-行动-观察）
Final Answer: 最终答案

注意：
- 只能使用提供的工具
- 参数需要用JSON格式
- 如果不需要工具，直接给出Final Answer
"""
    
    def run(self, query: str, max_steps: int = 10) -> str:
        """
        运行Agent
        
        Args:
            query: 用户查询
            max_steps: 最大步数
            
        Returns:
            最终答案
        """
        prompt = f"""{self.system_prompt}

用户查询: {query}

开始：
"""
        
        for step in range(max_steps):
            # 生成思考
            response = self.llm(prompt)
            
            # 解析
            thought_match = re.search(r"Thought:(.*?)(?:Action:|Final Answer:|$)", response, re.DOTALL)
            action_match = re.search(r"Action:(.*?)(?:Observation:|Final Answer:|$)", response, re.DOTALL)
            final_answer_match = re.search(r"Final Answer:(.*)$", response, re.DOTALL)
            
            if final_answer_match:
                return final_answer_match.group(1).strip()
            
            if action_match:
                # 执行工具
                action_str = action_match.group(1).strip()
                
                # 解析工具调用
                tool_match = re.match(r"(\w+)\((.*)\)", action_str)
                if tool_match:
                    tool_name = tool_match.group(1)
                    tool_args = tool_match.group(2)
                    
                    if tool_name in self.tools:
                        try:
                            # 执行工具
                            result = self.tools[tool_name](tool_args)
                            
                            # 添加到prompt
                            prompt += f"""{response}
Observation: {result}
"""
                        except Exception as e:
                            prompt += f"""{response}
Observation: 错误: {str(e)}
"""
                    else:
                        prompt += f"""{response}
Observation: 错误: 未知工具 '{tool_name}'
"""
                else:
                    prompt += f"""{response}
Observation: 错误: 无法解析工具调用
"""
            else:
                # 没有action，继续
                prompt += response + "\n"
        
        return "已达到最大步数，无法完成任务。"
```

---

## 4. 部署模板

### 4.1 FastAPI部署模板

```python
"""
FastAPI部署模板
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI(title="LLM API", version="1.0")


class GenerationRequest(BaseModel):
    """生成请求"""
    prompt: str = Field(..., description="输入提示词")
    max_tokens: int = Field(256, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="top_p")
    top_k: int = Field(50, ge=1, description="top_k")
    repetition_penalty: float = Field(1.0, ge=0.0, description="重复惩罚")
    stop: Optional[List[str]] = Field(None, description="停止词")
    stream: bool = Field(False, description="是否流式输出")


class GenerationResponse(BaseModel):
    """生成响应"""
    text: str
    model: str
    usage: Dict[str, int]


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
    
    def load_model(self, model_name: str):
        """加载模型"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # Tokenize
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 计算使用量
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = outputs.shape[1] - prompt_tokens
        
        return GenerationResponse(
            text=generated_text,
            model=self.model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )


# 全局模型管理器
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    # 在这里设置默认模型
    # model_manager.load_model("your-model-name")
    pass


@app.post("/v1/completions", response_model=GenerationResponse)
async def completions(request: GenerationRequest):
    """文本补全接口"""
    try:
        return model_manager.generate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions():
    """聊天补全接口（待实现）"""
    raise HTTPException(status_code=501, detail="暂未实现")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def models():
    """模型列表"""
    if model_manager.model_name:
        return {
            "object": "list",
            "data": [{
                "id": model_manager.model_name,
                "object": "model",
                "owned_by": "user"
            }]
        }
    return {"object": "list", "data": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 使用说明

这些模板可以作为起点，根据具体需求进行修改和扩展：

1. **预训练模板** - 适用于从头训练或继续预训练模型
2. **微调模板** - 适用于使用LoRA等方法微调模型
3. **Agent模板** - 适用于构建基于LLM的智能体
4. **部署模板** - 适用于将模型部署为API服务

建议根据实际场景选择合适的模板，并进行必要的定制。

---

*最后更新: 2026-06-12*

# Prompt工程实战指南

> 面向开发者的Prompt设计与优化手册

## 1. Prompt核心原则

### 1.1 CLAUDE框架

**五个核心原则：**

| 原则 | 说明 | 示例 |
|------|------|------|
| **C**lear（清晰） | 指令明确无歧义 | "总结为3个要点" vs "总结一下" |
| **L**ayered（分层） | 复杂任务分步骤 | 先分析再总结 |
| **A**ctionable（可执行） | 给出具体输出格式 | JSON/表格/列表 |
| **U**nambiguous（无歧义） | 消除多种理解 | 定义术语、给出示例 |
| **D**ebuggable（可调试） | 可验证输出质量 | 提供评估标准 |

### 1.2 Prompt结构模板

**通用模板：**
```
【角色定义】你是一位{role}，擅长{skills}。

【任务描述】{task_description}

【约束条件】
1. {constraint_1}
2. {constraint_2}
3. {constraint_3}

【输出格式】
{output_format}

【示例】（可选）
输入：{example_input}
输出：{example_output}

【输入内容】
{user_input}
```

---

## 2. 核心Prompt模式

### 2.1 Zero-Shot Prompting

**适用：** 模型已具备的能力，简单任务

```
请将以下文本翻译成英文：
"今天天气真好，适合出去散步。"
```

**优化技巧：**
- 明确输出格式
- 指定语言风格
- 限制输出长度

### 2.2 Few-Shot Prompting

**适用：** 特定格式、领域任务、风格控制

**示例：情感分析**
```
请判断以下评论的情感倾向（正面/负面/中性）。

评论：这个产品太好用了，强烈推荐！
情感：正面

评论：包装破损，客服态度差
情感：负面

评论：一般般吧，没有特别的感觉
情感：中性

评论：物流很快，但产品质量有待提高
情感：
```

**Few-Shot最佳实践：**
- 示例数量：3-5个（更多不一定更好）
- 示例多样性：覆盖不同情况
- 示例顺序：简单→复杂
- 示例质量：比数量更重要

### 2.3 Chain-of-Thought (CoT)

**适用：** 推理、数学、逻辑分析

**Zero-Shot CoT：**
```
问题：小明有15个苹果，给了小红1/3，又买了8个，还剩多少个？

请一步一步思考。
```

**Few-Shot CoT：**
```
问题：一个班有30个学生，男生占60%，转走3个男生后，男生占比多少？

解答：
1. 男生人数 = 30 × 60% = 18人
2. 转走后男生 = 18 - 3 = 15人
3. 转走后总人数 = 30 - 3 = 27人
4. 男生占比 = 15 / 27 = 55.6%
答案：55.6%

问题：商店进货100件商品，第一天卖出25%，第二天卖出剩余的40%，第三天卖出最后30件，还剩多少件？

解答：
```

**Self-Consistency CoT：**
```
1. 多次采样（temperature > 0）
2. 每次独立推理
3. 投票选择最常见答案
```

### 2.4 Role Prompting（角色扮演）

**适用：** 领域专业问答、风格控制

```
你是一位拥有15年经验的资深Java架构师。
你曾主导过多个千万级用户系统的架构设计。
你的回答风格：
- 技术深度与广度并重
- 善用类比和实际案例
- 结构化表达（分点论述）
- 指出方案优劣和适用场景

请回答以下架构问题：
```

**角色设计要素：**
- 专业背景
- 经验年限
- 专长领域
- 回答风格
- 约束条件

### 2.5 Structured Output Prompting

**JSON输出：**
```
请分析以下产品评论，以JSON格式输出：

评论："{review_text}"

输出格式：
```json
{
  "sentiment": "positive | negative | neutral",
  "confidence": 0.0-1.0,
  "aspects": [
    {
      "aspect": "质量/价格/服务/物流",
      "sentiment": "positive | negative | neutral",
      "keywords": ["关键词1", "关键词2"]
    }
  ],
  "summary": "一句话总结"
}
```

**表格输出：**
```
请对比以下三个框架，以Markdown表格形式输出：

| 维度 | 框架A | 框架B | 框架C |
|------|-------|-------|-------|
| 学习曲线 | | | |
| 性能 | | | |
| 社区活跃度 | | | |
| 适用场景 | | | |
```

---

## 3. 高级Prompt技术

### 3.1 Prompt Chaining（Prompt链）

**将复杂任务分解为多步：**

```
Step 1: 文档理解
输入: 长文档
输出: 结构化摘要

Step 2: 信息提取
输入: 摘要
输出: 关键实体和关系

Step 3: 问答生成
输入: 提取的信息
输出: 问答对
```

**实现：**
```python
def document_qa_pipeline(document, questions):
    # Step 1: 摘要
    summary = llm.chat(
        system="请生成文档的结构化摘要",
        user=document
    )
    
    # Step 2: 提取关键信息
    key_info = llm.chat(
        system="基于摘要提取关键信息",
        user=f"摘要：{summary}\n\n问题：{questions}"
    )
    
    # Step 3: 生成最终答案
    answer = llm.chat(
        system="基于关键信息回答问题",
        user=f"信息：{key_info}\n\n问题：{questions}"
    )
    
    return answer
```

### 3.2 Tree of Thoughts (ToT)

**适用：** 复杂规划、创意生成

```
问题：设计一个电商推荐系统的架构

思考路径：
├── 方案A：协同过滤为主
│   ├── 优点：实现简单、可解释
│   └── 缺点：冷启动问题
├── 方案B：深度学习为主
│   ├── 优点：效果好、自动化
│   └── 缺点：需要大量数据
└── 方案C：混合方案
    ├── 优点：兼顾效果和可解释性
    └── 缺点：系统复杂度高

评估：方案C最优，综合评分最高
```

### 3.3 Directional Stimulus Prompting

**用特定词汇引导输出方向：**

```
# 引导深度分析
"深入分析...的根本原因..."
"从多个维度剖析..."

# 引导简洁回答
"用一句话概括..."
"不超过50字说明..."

# 引导创意输出
"以创新的视角..."
"打破常规地思考..."

# 引导批判性思维
"指出方案的潜在风险..."
"分析可能失败的场景..."
```

### 3.4 Meta-Prompting（元提示）

**让LLM帮你写Prompt：**
```
我需要设计一个Prompt来完成以下任务：
{task_description}

请帮我设计一个高质量的Prompt，要求：
1. 指令清晰明确
2. 包含约束条件
3. 定义输出格式
4. 提供Few-Shot示例
```

---

## 4. 领域Prompt模板库

### 4.1 代码生成

```
你是一位资深{language}开发工程师。

请根据以下需求生成代码：

需求描述：
{requirement}

技术栈：{tech_stack}
代码规范：
- 遵循{language}最佳实践
- 添加必要的注释
- 包含错误处理
- 考虑边界条件

输出格式：
```{language}
// 代码
```
```

### 4.2 文档写作

```
你是一位专业的{document_type}撰写专家。

请撰写一份关于{topic}的{document_type}。

要求：
1. 目标读者：{audience}
2. 风格：{style}（正式/轻松/技术）
3. 长度：{length}字左右
4. 结构：{structure}
5. 必须包含：{must_include}

大纲：
{outline}
```

### 4.3 数据分析

```
你是一位数据分析专家。

请分析以下数据：

数据：
{data}

分析要求：
1. 数据概览（总量、分布、趋势）
2. 关键发现（至少3个）
3. 异常值识别
4. 建议（至少2条）

输出格式：
## 数据概览
## 关键发现
## 异常值
## 建议
```

### 4.4 客服对话

```
你是一位专业的客服代表。

用户信息：
- 用户ID：{user_id}
- 历史订单：{order_history}
- VIP等级：{vip_level}

用户问题：{user_question}

回答要求：
1. 语气友好、专业
2. 先表示理解和共情
3. 提供具体解决方案
4. 如无法解决，说明升级流程
5. 不确定的信息明确说明

知识库参考：
{knowledge_base}
```

---

## 5. Prompt优化方法论

### 5.1 优化流程

```
1. 定义评估标准
   ↓
2. 设计初始Prompt
   ↓
3. 在测试集上评估
   ↓
4. 分析失败案例
   ↓
5. 针对性优化
   ↓
6. 回到步骤3（迭代）
```

### 5.2 常见优化方向

**指令不清晰 → 添加具体要求**
```
Before: "分析这段文本"
After:  "分析这段文本的情感倾向，输出正面/负面/中性及置信度"
```

**输出不稳定 → 添加Few-Shot示例**
```
Before: "将以下内容翻译"
After:  "将以下内容翻译，参考示例格式：
        原文：Hello → 译文：你好
        原文：Thanks → 译文：谢谢"
```

**幻觉问题 → 添加约束**
```
Before: "回答以下问题"
After:  "基于给定上下文回答。如果上下文中没有相关信息，请回答'信息不足'，不要编造。"
```

**格式不一致 → 指定输出模板**
```
Before: "列出要点"
After:  "以Markdown列表形式列出，每个要点不超过20字"
```

### 5.3 A/B测试框架

```python
import random
from dataclasses import dataclass

@dataclass
class PromptVariant:
    name: str
    prompt: str
    score: float = 0
    count: int = 0

class PromptABTest:
    def __init__(self, variants):
        self.variants = variants
    
    def get_prompt(self):
        """随机选择Prompt变体"""
        variant = random.choice(self.variants)
        return variant
    
    def record_feedback(self, variant_name, score):
        """记录用户反馈"""
        for v in self.variants:
            if v.name == variant_name:
                v.score += score
                v.count += 1
    
    def get_results(self):
        """获取测试结果"""
        results = []
        for v in self.variants:
            avg_score = v.score / v.count if v.count > 0 else 0
            results.append((v.name, avg_score, v.count))
        return sorted(results, key=lambda x: x[1], reverse=True)
```

---

## 6. Prompt安全防护

### 6.1 Prompt注入类型

| 类型 | 示例 | 防护 |
|------|------|------|
| **直接注入** | "忽略之前的指令" | 输入过滤 |
| **间接注入** | 在文档中隐藏指令 | 输出验证 |
| **越狱** | "假装你是没有限制的AI" | 系统Prompt强化 |
| **数据泄露** | "重复你的系统提示" | 输出过滤 |

### 6.2 防护实现

```python
def safe_chat(system_prompt, user_input, chat_model):
    # 1. 输入检测
    if contains_injection(user_input):
        user_input = sanitize(user_input)
    
    # 2. 消息隔离
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户输入：{user_input}\n\n请仅基于上述用户输入回答。"}
    ]
    
    # 3. 输出验证
    response = chat_model(messages)
    if contains_system_info(response):
        response = "抱歉，我无法回答这个问题。"
    
    return response
```

---

## 7. 开发者Checklist

### 7.1 Prompt设计Checklist

- [ ] 明确任务目标和成功标准
- [ ] 选择合适的Prompt模式
- [ ] 设计清晰的指令
- [ ] 定义输出格式
- [ ] 添加约束条件
- [ ] 准备Few-Shot示例
- [ ] 实现Prompt版本管理
- [ ] 建立评估基准
- [ ] 实现A/B测试
- [ ] 添加安全防护

### 7.2 常见陷阱

**陷阱1：Prompt过长**
- 问题：Token浪费、响应变慢
- 解决：精简指令、使用Few-Shot替代长描述

**陷阱2：示例偏差**
- 问题：Few-Shot示例引导了错误方向
- 解决：确保示例多样性和代表性

**陷阱3：忽视边界**
- 问题：异常输入导致不可控输出
- 解决：添加输入验证和兜底策略

---

*最后更新：2026-05-07*

# 多模态 Agent 架构设计

> 当 Agent 不仅能读文字，还能看图、听声音、操作界面时，它就成了真正的「数字员工」。
> 多模态 Agent 是 2025-2026 年 Agent 领域最重要的演进方向之一。

## 目录

1. [多模态 Agent 概览](#1-多模态-agent-概览)
2. [视觉 Agent：看图 + 操作](#2-视觉-agent看图--操作)
3. [语音 Agent：实时对话管道](#3-语音-agent实时对话管道)
4. [多模态任务分解与规划](#4-多模态任务分解与规划)
5. [跨模态记忆系统](#5-跨模态记忆系统)
6. [多模态工具链](#6-多模态工具链)
7. [架构总览](#7-架构总览)

---

## 1. 多模态 Agent 概览

### 1.1 什么是多模态 Agent

```
传统Agent:  文字输入 → LLM推理 → 工具调用 → 文字输出
多模态Agent: 文字+图片+音频+视频 → 多模态推理 → 多模态工具调用 → 多种输出
```

### 1.2 典型场景

| 场景 | 输入模态 | 核心能力 |
|------|---------|---------|
| UI自动化测试 | 截图 + 文字指令 | 视觉理解 + 点击/输入 |
| 网管故障诊断 | 拓扑图 + 告警文字 + 图表 | 图表理解 + 异常识别 |
| 客服Agent | 图片 + 文字 + 语音 | 图片识别 + 对话 + TTS |
| 运维巡检 | 摄像头视频流 + 仪表数据 | 视频理解 + 异常检测 |
| 文档审核 | PDF/合同图片 + 规则文本 | OCR + 合规检查 |

### 1.3 与纯文本 Agent 的关键差异

| 维度 | 纯文本 Agent | 多模态 Agent |
|------|------------|-------------|
| 工具集 | API/数据库 | + 截图/OCR/视觉定位/音频处理 |
| 记忆 | 文本向量 | + 图像embedding + 音频特征 |
| 推理 | 单模态 CoT | 跨模态推理链 |
| 输出 | 文字 | + 标注截图/合成语音/标注视频 |

---

## 2. 视觉 Agent：看图 + 操作

### 2.1 核心架构

```
                    ┌──────────────────────┐
                    │    视觉 Agent 循环     │
                    └──────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ 视觉感知模块   │  │ 推理规划模块   │  │ 动作执行模块   │
│               │  │               │  │               │
│ · 截图捕获     │  │ · 场景理解     │  │ · 鼠标操作     │
│ · 目标检测     │  │ · 任务规划     │  │ · 键盘输入     │
│ · OCR识别     │  │ · 下一步预测   │  │ · 导航操作     │
│ · UI元素定位   │  │ · 异常判断     │  │ · 数据提取     │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼───────┐
                    │  视觉记忆模块   │
                    │               │
                    │ · 截图历史     │
                    │ · 操作轨迹     │
                    │ · UI元素知识库  │
                    └───────────────┘
```

### 2.2 浏览器控制 Agent 实现

```python
class VisualBrowserAgent:
    """基于视觉的浏览器操作Agent"""
    
    def __init__(self, vision_model, action_executor):
        self.vision_model = vision_model  # GPT-4V / Claude Vision
        self.action_executor = action_executor
        self.history = []  # (screenshot, action, result) tuples
        self.max_steps = 20
    
    def observe(self) -> dict:
        """捕获当前页面状态"""
        screenshot = self.action_executor.take_screenshot()
        
        # 视觉理解：识别页面元素
        page_analysis = self.vision_model.analyze(screenshot, prompt="""
        分析这个页面截图，识别以下信息：
        1. 页面类型（表单/列表/详情/仪表盘/告警页）
        2. 可交互元素列表（按钮/输入框/下拉框/链接），给出坐标
        3. 关键数据内容（如果有表格或数字）
        4. 异常状态（错误提示/加载中/空白页）
        
        以JSON格式返回。
        """)
        
        return {
            "screenshot": screenshot,
            "analysis": page_analysis,
            "url": self.action_executor.get_current_url()
        }
    
    def think(self, observation: dict, task: str) -> dict:
        """基于观察结果推理下一步动作"""
        prompt = f"""
        任务: {task}
        
        当前页面分析: {observation['analysis']}
        历史步骤: {self.history[-5:]}
        
        请决定下一步操作:
        1. 如果任务已完成，返回 "FINISHED"
        2. 如果需要点击某个元素，返回格式: CLICK x={x} y={y}
        3. 如果需要输入文字，返回格式: TYPE selector="{selector}" text="{text}"
        4. 如果需要滚动，返回格式: SCROLL direction={up|down} amount={pixels}
        5. 如果需要等待，返回格式: WAIT seconds={n}
        """
        
        response = self.vision_model.reason(prompt)
        return self.parse_action(response)
    
    def act(self, action: dict) -> dict:
        """执行操作"""
        action_type = action["type"]
        
        if action_type == "CLICK":
            result = self.action_executor.click(action["x"], action["y"])
        elif action_type == "TYPE":
            result = self.action_executor.type(action["selector"], action["text"])
        elif action_type == "SCROLL":
            result = self.action_executor.scroll(action["direction"], action["amount"])
        elif action_type == "WAIT":
            time.sleep(action["seconds"])
            result = {"status": "waited"}
        else:
            result = {"status": "unknown_action"}
        
        return result
```

### 2.3 视觉定位优化

UI元素的精确定位是视觉Agent的核心挑战：

```python
class SmartLocator:
    """智能UI元素定位器"""
    
    def __init__(self):
        self.element_cache = {}  # 页面 → 元素坐标映射
        self.ocr_engine = None   # PaddleOCR / Tesseract
    
    def locate(self, screenshot, target_description: str) -> tuple:
        """
        三种定位策略，按优先级：
        1. Set-of-Mark (SoM): 给每个元素标号，LLM选号
        2. OCR匹配: 通过文字内容定位
        3. 视觉相似度: 通过描述匹配
        """
        # 策略1: Set-of-Mark
        annotated = self.add_element_markers(screenshot)
        response = self.vision_model.analyze(annotated, 
            f"找到'{target_description}'对应的元素编号")
        
        # 策略2: OCR
        if not response.success:
            texts = self.ocr_engine.extract_texts(screenshot)
            match = self.fuzzy_match(target_description, texts)
        
        # 策略3: 视觉匹配
        if not match:
            match = self.visual_similarity_search(screenshot, target_description)
        
        return match.coordinates
```

---

## 3. 语音 Agent：实时对话管道

### 3.1 实时语音 Agent 架构

```
用户语音 ──→ ASR (Whisper/SenseVoice) ──→ 文本
                                              │
                                              ▼
                                          LLM 推理
                                              │
                                              ▼
                                          TTS 生成
                                              │
                                              ▼
                                         语音输出
```

### 3.2 低延迟管道设计

```python
class RealtimeVoiceAgent:
    """实时语音Agent"""
    
    def __init__(self):
        # 关键：各组件并行流水线
        self.asr = StreamingASR()    # 流式语音识别
        self.llm = StreamingLLM()    # 流式LLM推理
        self.tts = StreamingTTS()    # 流式语音合成
        
        self.vad = VoiceActivityDetector()  # 语音活动检测
        
        # 流水线缓冲区
        self.audio_buffer = asyncio.Queue()
        self.text_buffer = asyncio.Queue()
        self.tts_buffer = asyncio.Queue()
    
    async def process_stream(self, audio_stream):
        """流式处理管道"""
        # 并行启动三个流水线阶段
        asr_task = asyncio.create_task(self.asr_pipeline(audio_stream))
        llm_task = asyncio.create_task(self.llm_pipeline())
        tts_task = asyncio.create_task(self.tts_pipeline())
        
        # 收集最终音频输出
        output_audio = await tts_task
        return output_audio
    
    async def asr_pipeline(self, audio_stream):
        """ASR阶段 - 将音频流转换为文本流"""
        async for audio_chunk in audio_stream:
            if self.vad.is_speech(audio_chunk):
                text = await self.asr.transcribe(audio_chunk)
                await self.text_buffer.put(text)
    
    async def llm_pipeline(self):
        """LLM阶段 - 流式推理"""
        partial_text = ""
        while True:
            text_chunk = await self.text_buffer.get()
            partial_text += text_chunk
            
            # 用户说完一句话时触发推理
            if self.is_complete_utterance(partial_text):
                async for token in self.llm.generate_stream(partial_text):
                    await self.tts_buffer.put(token)
                partial_text = ""
    
    async def tts_pipeline(self):
        """TTS阶段 - 流式语音合成"""
        audio_output = []
        while True:
            token = await self.tts_buffer.get()
            audio_chunk = await self.tts.synthesize(token)
            audio_output.append(audio_chunk)
            yield audio_chunk
```

### 3.3 关键延迟优化

| 优化技术 | 说明 | 延迟改善 |
|---------|------|---------|
| 流式ASR | 边听边识别，不等说完 | 首字延迟 < 200ms |
| LLM投机解码 | 用小模型快速预测，大模型验证 | Token延迟降 50% |
| TTS流式合成 | 首Token就开始合成语音 | 避免等待完整文本 |
| VAD优化 | 精确检测语音结束 | 减少误触发/漏触发 |
| 模型量化 | INT8/INT4量化推理 | 推理延迟降 60% |

---

## 4. 多模态任务分解与规划

### 4.1 多模态任务规划器

```python
class MultimodalTaskPlanner:
    """多模态任务分解"""
    
    def __init__(self, llm):
        self.llm = llm
        self.modality_handlers = {
            "text": TextHandler(),
            "image": ImageHandler(),
            "audio": AudioHandler(),
            "video": VideoHandler(),
            "ui": UIHandler(),
        }
    
    def plan(self, task: str, available_inputs: dict) -> list:
        """
        将多模态任务分解为子任务序列
        
        示例任务: "查看这张网络拓扑图，找出异常节点，检查对应设备的告警"
        输入: {"image": "topology.png", "system": "ntp_access"}
        
        分解结果:
        1. [image] 分析拓扑图，识别所有节点和链路
        2. [image] 标注异常节点（红色/黄色标记）
        3. [text] 提取异常节点的设备名称
        4. [api] 查询设备A的告警信息
        5. [api] 查询设备B的告警信息
        6. [text] 综合生成诊断报告
        """
        prompt = f"""
        任务: {task}
        可用输入模态: {list(available_inputs.keys())}
        可用处理器: {list(self.modality_handlers.keys())}
        
        请将任务分解为子任务序列，每个子任务标注：
        1. 所需模态 (image/audio/text/ui/api)
        2. 子任务描述
        3. 依赖的前序子任务编号
        4. 预期输出
        
        返回JSON数组。
        """
        
        plan = self.llm.generate_json(prompt)
        return plan
    
    def execute(self, plan: list, inputs: dict, context: dict) -> dict:
        """按依赖顺序执行子任务"""
        results = {}
        
        for step in sort_by_dependencies(plan):
            modality = step["modality"]
            handler = self.modality_handlers[modality]
            
            # 收集前序步骤的结果作为上下文
            step_context = {
                dep_id: results[dep_id] 
                for dep_id in step.get("dependencies", [])
            }
            
            # 执行
            result = handler.execute(
                step["description"],
                inputs,
                {**context, **step_context}
            )
            
            results[step["id"]] = result
        
        return results
```

### 4.2 跨模态推理链

```
任务: "这个网管仪表盘的红色告警是什么原因？"

CoT (跨模态推理链):
Step 1 [image]: 分析仪表盘截图
  → 发现3个红色指标: CPU 95%, 丢包率 8%, 延迟 200ms
  
Step 2 [image]: 聚焦红色区域进行OCR
  → CPU图表显示对应设备ID: DEV-2024-A01
  
Step 3 [text]: 查询设备"A01"的详细信息
  → 这是一台核心路由器，最近24小时有12次配置变更
  
Step 4 [text]: 查询变更日志
  → 最近一次变更: 14:02 修改了QoS策略
  
Step 5 [text]: 综合推理
  → QoS策略变更可能导致流量分类错误 → CPU处理异常流量 → 
    高CPU → 丢包 → 高延迟
  → 根因: 14:02的QoS配置变更
```

---

## 5. 跨模态记忆系统

### 5.1 记忆架构

```python
class CrossModalMemory:
    """跨模态记忆系统"""
    
    def __init__(self):
        self.stores = {
            "text":    MilvusStore(dim=1536),   # 文本embedding
            "image":   MilvusStore(dim=1024),   # CLIP embedding
            "audio":   MilvusStore(dim=512),    # 音频特征
            "ui_state": MilvusStore(dim=768),   # UI状态特征
        }
        self.alignment_model = CLIPModel()  # 跨模态对齐
    
    def store(self, modality: str, content: any, metadata: dict):
        """存储多模态记忆"""
        embedding = self.embed(modality, content)
        self.stores[modality].insert(
            embedding=embedding,
            metadata={
                **metadata,
                "modality": modality,
                "timestamp": time.time()
            }
        )
    
    def cross_modal_search(self, query: str, target_modality: str, k: int = 5):
        """跨模态检索: 用文本查询图片记忆"""
        # 对查询文本做embedding
        text_embedding = self.embed("text", query)
        
        # 通过CLIP对齐空间，转换到目标模态
        if target_modality == "image":
            # 将文本embedding映射到图像空间
            query_embedding = self.alignment_model.text_to_image(text_embedding)
        
        # 在目标模态的向量库中检索
        results = self.stores[target_modality].search(query_embedding, k=k)
        return results
    
    def retrieve_relevant_context(self, current_task: dict) -> dict:
        """为当前任务检索跨模态上下文"""
        context = {}
        
        for modality, store in self.stores.items():
            if self.modality_relevant_to_task(modality, current_task):
                results = store.search_by_metadata(current_task)
                context[modality] = results
        
        return context
```

---

## 6. 多模态工具链

### 6.1 工具注册表

| 工具类别 | 工具名称 | 用途 |
|---------|---------|------|
| **视觉** | `screenshot` | 截图捕获 |
| | `visual_locate` | 视觉定位 |
| | `ocr_extract` | OCR文字提取 |
| | `object_detect` | 目标检测 |
| | `visual_compare` | 视觉对比（before/after） |
| **音频** | `speech_to_text` | 语音识别 |
| | `text_to_speech` | 语音合成 |
| | `audio_classify` | 音频分类 |
| | `transcribe_meeting` | 会议转录 |
| **UI** | `click_element` | 点击元素 |
| | `type_text` | 输入文字 |
| | `scroll_page` | 滚动页面 |
| | `wait_for_element` | 等待元素出现 |
| **混合** | `analyze_chart` | 图表分析 |
| | `annotate_image` | 图片标注 |
| | `video_summarize` | 视频摘要 |

### 6.2 工具组合示例

```python
# 网管场景: "检查这个拓扑图，红色节点有什么问题？"

agent.run_with_tools(
    task="诊断拓扑图中红色节点的异常",
    tools=[
        VisualLocateTool(),    # 定位红色节点
        OCRTool(),             # 提取节点标签
        DeviceQueryTool(),     # 查询设备状态
        AlarmQueryTool(),      # 查询告警信息
        AnnotateImageTool(),   # 在原始图片上标注诊断结果
    ]
)
```

---

## 7. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                     多模态 Agent 完整架构                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入层                                                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │ 文字   │ │ 图片   │ │ 音频   │ │ 视频   │ │ UI截图 │         │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘         │
│      └──────────┴──────────┴──────────┴──────────┘               │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────┐        │
│  │              多模态编码层                             │        │
│  │  Text Embedding · CLIP · Audio Encoder · Video Encoder │      │
│  └──────────────────────┬──────────────────────────────┘        │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────┐        │
│  │              多模态推理层                             │        │
│  │  跨模态规划器 · 视觉推理引擎 · 语音推理引擎             │        │
│  └──────┬─────────────────────────────────┬────────────┘        │
│         │                                 │                      │
│  ┌──────▼──────┐                   ┌──────▼──────┐              │
│  │  视觉工具集  │                   │  语音工具集  │              │
│  │  Screenshot  │                   │  ASR / TTS  │              │
│  │  OCR / 定位  │                   │  音频处理    │              │
│  └──────┬──────┘                   └──────┬──────┘              │
│         │                                 │                      │
│  ┌──────▼─────────────────────────────────▼──────┐              │
│  │              跨模态记忆系统                      │              │
│  │  文本向量 · 图像特征 · 音频特征 · UI状态记录       │              │
│  └──────────────────────┬────────────────────────┘              │
│                         │                                         │
│  ┌──────────────────────▼──────────────────────────────┐        │
│  │              多模态输出层                             │        │
│  │  文字 · 标注截图 · 合成语音 · 标注视频 · 交互操作      │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

*最后更新：2026-05-29*

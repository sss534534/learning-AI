# Multi-Agent协作架构

## 目录
1. [Multi-Agent架构概览](#multi-agent架构概览)
2. [协作模式设计](#协作模式设计)
3. [通信与协调机制](#通信与协调机制)
4. [任务分解与分配](#任务分解与分配)
5. [记忆共享与状态管理](#记忆共享与状态管理)
6. [冲突解决与一致性](#冲突解决与一致性)
7. [实战案例与代码](#实战案例与代码)

---

## Multi-Agent架构概览

### 从单Agent到Multi-Agent的演进

```
单Agent架构:
┌─────────────────────────────────────┐
│           用户请求                   │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│  ┌─────────┐  ┌─────────┐          │
│  │ 规划模块 │→│ 工具调用 │          │
│  └─────────┘  └─────────┘          │
│       ↑            ↓               │
│  ┌─────────┐  ┌─────────┐          │
│  │ 记忆模块 │←│ LLM核心  │          │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘

Multi-Agent架构:
┌─────────────────────────────────────────────────────────┐
│                      协调器/编排器                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Agent A │  │ Agent B │  │ Agent C │  │ Agent D │    │
│  │ (研究)   │  │ (分析)   │  │ (执行)   │  │ (验证)   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       └─────────────┴─────────────┴─────────────┘       │
│                      共享总线/消息队列                    │
└─────────────────────────────────────────────────────────┘
```

### Multi-Agent适用场景

| 场景特征 | 单Agent | Multi-Agent | 原因 |
|----------|---------|-------------|------|
| 任务复杂度 | 简单-中等 | 复杂 | 复杂任务需要分工协作 |
| 专业领域数 | 1-2个 | 3+个 | 不同领域需要不同专家 |
| 并行需求 | 低 | 高 | 多Agent可并行处理 |
| 容错要求 | 一般 | 高 | 单点故障风险分散 |
| 可解释性 | 中等 | 高 | 决策过程更透明 |

### 架构模式对比

| 模式 | 结构 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **层级式** | 树状结构，有中心协调 | 企业工作流、审批流程 | 控制清晰、责任明确 | 单点瓶颈 |
| **网状式** | 点对点全连接 | 创意生成、头脑风暴 | 信息流通快 | 复杂度O(n²) |
| **星型式** | 中心节点+边缘节点 | 客服系统、任务分发 | 易于管理 | 中心节点压力大 |
| **流水线式** | 链式处理 | 数据处理、ETL | 简单高效 | 灵活性差 |
| **市场式** | 拍卖/竞标机制 | 资源分配、负载均衡 | 自适应 | 实现复杂 |

---

## 协作模式设计

### 1. 层级协作模式 (Hierarchical)

```python
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """任务定义"""
    id: str
    description: str
    task_type: str
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    assigned_to: Optional[str] = None
    
@dataclass
class Agent:
    """Agent定义"""
    id: str
    name: str
    role: str
    capabilities: List[str]
    llm_client: any
    parent: Optional['Agent'] = None
    children: List['Agent'] = field(default_factory=list)
    
class HierarchicalOrchestrator:
    """层级式编排器"""
    
    def __init__(self, root_agent: Agent):
        self.root = root_agent
        self.task_queue = asyncio.PriorityQueue()
        self.results = {}
        self.agent_registry = self._build_registry()
    
    def _build_registry(self) -> Dict[str, Agent]:
        """构建Agent注册表"""
        registry = {}
        
        def traverse(agent: Agent):
            registry[agent.id] = agent
            for child in agent.children:
                traverse(child)
        
        traverse(self.root)
        return registry
    
    async def submit_task(self, task: Task) -> str:
        """提交任务到根节点"""
        await self.task_queue.put((task.priority, task))
        return await self._process_hierarchy(self.root, task)
    
    async def _process_hierarchy(self, agent: Agent, task: Task) -> str:
        """层级处理任务"""
        # 1. 判断当前Agent是否能处理
        if self._can_handle(agent, task):
            return await self._execute_task(agent, task)
        
        # 2. 否则分配给子Agent
        capable_children = [
            child for child in agent.children
            if any(cap in task.task_type for cap in child.capabilities)
        ]
        
        if not capable_children:
            raise ValueError(f"No agent can handle task: {task.task_type}")
        
        # 3. 选择最优子Agent（负载均衡）
        selected = self._select_best_agent(capable_children)
        
        # 4. 分解任务（如果需要）
        if len(capable_children) > 1 and task.task_type == "complex":
            subtasks = self._decompose_task(task, capable_children)
            
            # 并行执行子任务
            results = await asyncio.gather(*[
                self._process_hierarchy(child, subtask)
                for child, subtask in zip(capable_children, subtasks)
            ])
            
            # 合并结果
            return await self._merge_results(agent, task, results)
        else:
            return await self._process_hierarchy(selected, task)
    
    def _can_handle(self, agent: Agent, task: Task) -> bool:
        """判断Agent是否能处理任务"""
        return any(cap in task.task_type for cap in agent.capabilities)
    
    def _select_best_agent(self, agents: List[Agent]) -> Agent:
        """选择最佳Agent（可扩展负载均衡逻辑）"""
        # 简化：选择第一个
        return agents[0]
    
    def _decompose_task(self, task: Task, agents: List[Agent]) -> List[Task]:
        """分解任务"""
        subtasks = []
        for i, agent in enumerate(agents):
            subtask = Task(
                id=f"{task.id}_sub_{i}",
                description=f"{task.description} - Part {i+1}",
                task_type=agent.capabilities[0],
                priority=task.priority,
                dependencies=[]
            )
            subtasks.append(subtask)
        return subtasks
    
    async def _execute_task(self, agent: Agent, task: Task) -> str:
        """执行具体任务"""
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_to = agent.id
        
        # 调用LLM执行任务
        prompt = f"""作为{agent.role}，请完成以下任务：
        
任务：{task.description}

请提供详细的执行结果。"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        task.status = TaskStatus.COMPLETED
        task.result = {"output": result, "agent": agent.id}
        
        return result
    
    async def _merge_results(self, agent: Agent, task: Task, results: List[str]) -> str:
        """合并子任务结果"""
        prompt = f"""作为{agent.role}，请整合以下子任务的结果：

原始任务：{task.description}

子任务结果：
{chr(10).join(f"子任务{i+1}：{r[:500]}..." for i, r in enumerate(results))}

请提供一个完整、连贯的整合结果。"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

# 使用示例：企业报告生成系统
async def create_report_system():
    """创建层级式报告生成系统"""
    
    # 创建Agent层级
    data_collector = Agent(
        id="data_001",
        name="数据收集员",
        role="数据收集专家",
        capabilities=["data_collection", "web_search"],
        llm_client=openai_client
    )
    
    analyst = Agent(
        id="analyst_001",
        name="数据分析师",
        role="数据分析专家",
        capabilities=["data_analysis", "statistics"],
        llm_client=openai_client
    )
    
    writer = Agent(
        id="writer_001",
        name="报告撰写员",
        role="报告撰写专家",
        capabilities=["writing", "formatting"],
        llm_client=openai_client
    )
    
    # 创建根协调器
    manager = Agent(
        id="manager_001",
        name="项目经理",
        role="项目协调专家",
        capabilities=["coordination", "planning"],
        llm_client=openai_client,
        children=[data_collector, analyst, writer]
    )
    
    data_collector.parent = manager
    analyst.parent = manager
    writer.parent = manager
    
    # 初始化编排器
    orchestrator = HierarchicalOrchestrator(manager)
    
    # 提交任务
    task = Task(
        id="report_001",
        description="生成2024年Q1市场分析报告",
        task_type="complex",
        priority=1
    )
    
    result = await orchestrator.submit_task(task)
    return result
```

### 2. 工作组协作模式 (Workgroup)

```python
class WorkgroupOrchestrator:
    """工作组协作模式 - 扁平化团队协作"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.id: agent for agent in agents}
        self.message_bus = asyncio.Queue()
        self.shared_context = {}
        self.discussion_history = []
    
    async def collaborative_solve(
        self, 
        problem: str,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8
    ) -> Dict:
        """
        协作解决问题
        
        Args:
            problem: 待解决的问题
            max_rounds: 最大讨论轮数
            consensus_threshold: 共识阈值
        """
        # 初始化讨论
        self.discussion_history = []
        self.shared_context = {
            "problem": problem,
            "current_solution": None,
            "agreement_scores": {}
        }
        
        for round_num in range(max_rounds):
            print(f"\n=== 讨论轮次 {round_num + 1} ===")
            
            # 每个Agent发表观点
            round_contributions = []
            for agent_id, agent in self.agents.items():
                contribution = await self._agent_contribute(agent, round_num)
                round_contributions.append({
                    "agent": agent_id,
                    "role": agent.role,
                    "content": contribution
                })
                
                self.discussion_history.append({
                    "round": round_num,
                    "agent": agent_id,
                    "content": contribution
                })
            
            # 评估共识度
            consensus = await self._evaluate_consensus(round_contributions)
            
            if consensus["score"] >= consensus_threshold:
                return {
                    "solution": consensus["synthesis"],
                    "consensus_score": consensus["score"],
                    "rounds": round_num + 1,
                    "discussion": self.discussion_history
                }
            
            # 更新共享上下文
            self.shared_context["current_solution"] = consensus["synthesis"]
        
        # 达到最大轮数，返回最佳方案
        final_consensus = await self._evaluate_consensus([])
        return {
            "solution": final_consensus["synthesis"],
            "consensus_score": final_consensus["score"],
            "rounds": max_rounds,
            "status": "max_rounds_reached",
            "discussion": self.discussion_history
        }
    
    async def _agent_contribute(self, agent: Agent, round_num: int) -> str:
        """Agent发表贡献"""
        # 构建上下文
        context = self._build_agent_context(agent, round_num)
        
        prompt = f"""你是{agent.name}，角色是{agent.role}。

{context}

请基于你的专业角度，对当前问题发表你的观点和建议。
保持简洁，重点突出你的专业见解。"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _build_agent_context(self, agent: Agent, round_num: int) -> str:
        """构建Agent上下文"""
        context_parts = [
            f"问题：{self.shared_context['problem']}",
            f"你的专长：{', '.join(agent.capabilities)}"
        ]
        
        if round_num > 0:
            # 添加历史讨论
            recent_discussions = self.discussion_history[-len(self.agents):]
            context_parts.append("\n之前的讨论：")
            for d in recent_discussions:
                context_parts.append(f"- {d['agent']}: {d['content'][:200]}...")
        
        if self.shared_context.get("current_solution"):
            context_parts.append(f"\n当前方案：{self.shared_context['current_solution'][:300]}...")
        
        return "\n".join(context_parts)
    
    async def _evaluate_consensus(self, contributions: List[Dict]) -> Dict:
        """评估共识度"""
        if not contributions:
            # 基于历史讨论综合
            prompt = f"""基于以下讨论历史，综合出一个最终方案：

{chr(10).join(f"{d['agent']} ({d['round']}轮): {d['content'][:300]}" 
              for d in self.discussion_history[-10:])}

请：
1. 综合各方观点形成一个完整方案
2. 评估各方对这个方案的一致程度（0-1分数）
3. 指出剩余的分歧点

以JSON格式返回：
{{
    "synthesis": "综合方案",
    "consensus_score": 0.85,
    "remaining_disagreements": ["分歧点1", "分歧点2"]
}}"""
        else:
            prompt = f"""基于以下各方贡献，评估共识度：

{chr(10).join(f"{c['agent']} ({c['role']}): {c['content'][:300]}" for c in contributions)}

请：
1. 综合形成一个方案
2. 评估共识度（0-1）

以JSON格式返回：
{{
    "synthesis": "综合方案",
    "consensus_score": 0.85
}}"""
        
        # 使用任意一个Agent的客户端
        agent = list(self.agents.values())[0]
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response.choices[0].message.content)
```

### 3. 流水线协作模式 (Pipeline)

```python
from typing import Callable

class PipelineStage:
    """流水线阶段"""
    
    def __init__(
        self, 
        name: str,
        agent: Agent,
        process_func: Callable,
        output_schema: Dict = None
    ):
        self.name = name
        self.agent = agent
        self.process_func = process_func
        self.output_schema = output_schema
        self.next_stage: Optional['PipelineStage'] = None
        self.error_handler: Optional['PipelineStage'] = None

class PipelineOrchestrator:
    """流水线编排器"""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.stage_map: Dict[str, PipelineStage] = {}
    
    def add_stage(self, stage: PipelineStage):
        """添加阶段"""
        if self.stages:
            self.stages[-1].next_stage = stage
        self.stages.append(stage)
        self.stage_map[stage.name] = stage
    
    def set_error_handler(self, stage_name: str, handler_stage: PipelineStage):
        """设置错误处理"""
        if stage_name in self.stage_map:
            self.stage_map[stage_name].error_handler = handler_stage
    
    async def execute(self, input_data: Dict, context: Dict = None) -> Dict:
        """执行流水线"""
        if not self.stages:
            raise ValueError("No stages in pipeline")
        
        current_data = input_data
        execution_trace = []
        context = context or {}
        
        for stage in self.stages:
            try:
                print(f"执行阶段: {stage.name}")
                
                # 执行阶段处理
                result = await self._execute_stage(stage, current_data, context)
                
                execution_trace.append({
                    "stage": stage.name,
                    "status": "success",
                    "input": str(current_data)[:200],
                    "output": str(result)[:200]
                })
                
                current_data = result
                
            except Exception as e:
                print(f"阶段 {stage.name} 失败: {e}")
                
                execution_trace.append({
                    "stage": stage.name,
                    "status": "failed",
                    "error": str(e)
                })
                
                # 尝试错误处理
                if stage.error_handler:
                    print(f"触发错误处理: {stage.error_handler.name}")
                    error_result = await self._execute_stage(
                        stage.error_handler,
                        {"error": str(e), "failed_data": current_data},
                        context
                    )
                    current_data = error_result
                else:
                    raise
        
        return {
            "result": current_data,
            "trace": execution_trace,
            "stages_completed": len([t for t in execution_trace if t["status"] == "success"])
        }
    
    async def _execute_stage(
        self, 
        stage: PipelineStage, 
        data: Dict, 
        context: Dict
    ) -> Dict:
        """执行单个阶段"""
        return await stage.process_func(stage.agent, data, context)
    
    async def execute_parallel(
        self,
        inputs: List[Dict],
        max_concurrency: int = 5
    ) -> List[Dict]:
        """并行执行多个输入"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def execute_with_limit(inp):
            async with semaphore:
                return await self.execute(inp)
        
        return await asyncio.gather(*[execute_with_limit(inp) for inp in inputs])

# 使用示例：内容审核流水线
def create_content_pipeline():
    """创建内容审核流水线"""
    
    # 定义各阶段处理函数
    async def preprocess(agent, data, context):
        """预处理阶段"""
        content = data.get("content", "")
        # 文本清洗、分块等
        return {
            "original": content,
            "cleaned": content.strip(),
            "word_count": len(content.split())
        }
    
    async def safety_check(agent, data, context):
        """安全检查阶段"""
        prompt = f"""检查以下内容是否包含违规信息：

{data['cleaned']}

请识别：
1. 是否包含敏感词
2. 是否涉及违规内容
3. 风险等级（低/中/高）

以JSON格式返回：
{{
    "is_safe": true/false,
    "risk_level": "low/medium/high",
    "violations": ["违规类型1", "违规类型2"]
}}"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        check_result = json.loads(response.choices[0].message.content)
        
        if not check_result.get("is_safe", True):
            raise ValueError(f"内容不安全: {check_result.get('violations', [])}")
        
        return {**data, "safety_check": check_result}
    
    async def quality_score(agent, data, context):
        """质量评分阶段"""
        prompt = f"""评估以下内容质量：

{data['cleaned']}

请从以下维度评分（1-10）：
1. 原创性
2. 可读性
3. 信息密度
4. 结构清晰度

以JSON格式返回评分。"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        scores = json.loads(response.choices[0].message.content)
        
        avg_score = sum(scores.values()) / len(scores)
        
        return {
            **data,
            "quality_scores": scores,
            "overall_score": avg_score
        }
    
    async def tag_and_categorize(agent, data, context):
        """标签分类阶段"""
        prompt = f"""为内容生成标签和分类：

{data['cleaned']}

请提供：
1. 主要分类
2. 关键词标签（5-10个）
3. 目标受众

以JSON格式返回。"""
        
        response = await agent.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        tags = json.loads(response.choices[0].message.content)
        
        return {**data, "metadata": tags}
    
    async def error_recovery(agent, data, context):
        """错误恢复处理"""
        error = data.get("error", "")
        failed_data = data.get("failed_data", {})
        
        # 尝试修复或降级处理
        return {
            "status": "recovered",
            "error": error,
            "recovered_data": failed_data,
            "requires_manual_review": True
        }
    
    # 创建Agent
    preprocessor = Agent("pre_001", "预处理器", "预处理", ["text_cleaning"], openai_client)
    safety_agent = Agent("safe_001", "安全审核员", "安全审核", ["content_moderation"], openai_client)
    quality_agent = Agent("qual_001", "质量评估员", "质量评估", ["quality_analysis"], openai_client)
    tagger = Agent("tag_001", "标签生成器", "标签生成", ["classification"], openai_client)
    
    # 构建流水线
    pipeline = PipelineOrchestrator()
    
    pipeline.add_stage(PipelineStage("preprocess", preprocessor, preprocess))
    pipeline.add_stage(PipelineStage("safety_check", safety_agent, safety_check))
    pipeline.add_stage(PipelineStage("quality_score", quality_agent, quality_score))
    pipeline.add_stage(PipelineStage("tag", tagger, tag_and_categorize))
    
    # 设置错误处理
    error_stage = PipelineStage("error_handler", preprocessor, error_recovery)
    pipeline.set_error_handler("safety_check", error_stage)
    
    return pipeline
```

---

## 通信与协调机制

### 1. 消息总线设计

```python
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    TASK = "task"           # 任务分配
    RESULT = "result"       # 任务结果
    QUERY = "query"         # 查询请求
    RESPONSE = "response"   # 查询响应
    BROADCAST = "broadcast" # 广播消息
    HEARTBEAT = "heartbeat" # 心跳

@dataclass
class Message:
    """消息定义"""
    id: str
    type: MessageType
    sender: str
    receiver: Optional[str]  # None表示广播
    payload: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 用于关联请求-响应
    priority: int = 1

class MessageBus:
    """异步消息总线"""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.max_history = 1000
    
    def register_agent(self, agent_id: str):
        """注册Agent"""
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.PriorityQueue()
    
    def subscribe(self, message_type: MessageType, handler: Callable):
        """订阅消息类型"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(handler)
    
    async def send(self, message: Message):
        """发送消息"""
        # 记录历史
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        # 广播或单播
        if message.receiver is None:
            # 广播给所有订阅者
            handlers = self.subscribers.get(message.type, [])
            for handler in handlers:
                asyncio.create_task(handler(message))
        else:
            # 发送到指定队列
            if message.receiver in self.queues:
                await self.queues[message.receiver].put(
                    (message.priority, message)
                )
    
    async def receive(self, agent_id: str, timeout: float = None) -> Optional[Message]:
        """接收消息"""
        if agent_id not in self.queues:
            return None
        
        try:
            _, message = await asyncio.wait_for(
                self.queues[agent_id].get(),
                timeout=timeout
            )
            return message
        except asyncio.TimeoutError:
            return None
    
    async def request_response(
        self,
        request: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """请求-响应模式"""
        correlation_id = str(uuid.uuid4())
        request.correlation_id = correlation_id
        
        # 发送请求
        await self.send(request)
        
        # 等待响应
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            # 检查历史消息中的响应
            for msg in reversed(self.message_history):
                if (msg.correlation_id == correlation_id and 
                    msg.type == MessageType.RESPONSE):
                    return msg
            await asyncio.sleep(0.1)
        
        return None
```

### 2. 状态同步机制

```python
class SharedStateManager:
    """共享状态管理器"""
    
    def __init__(self):
        self.state: Dict[str, any] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.version_counter = 0
        self.state_versions: Dict[str, int] = {}
    
    async def get(self, key: str, default=None):
        """获取状态值"""
        async with self._get_lock(key):
            return self.state.get(key, default)
    
    async def set(self, key: str, value: any, notify: bool = True):
        """设置状态值"""
        async with self._get_lock(key):
            self.state[key] = value
            self.version_counter += 1
            self.state_versions[key] = self.version_counter
        
        if notify:
            await self._notify_change(key, value)
    
    async def update(self, key: str, updater: Callable, notify: bool = True):
        """原子更新"""
        async with self._get_lock(key):
            current = self.state.get(key)
            new_value = updater(current)
            self.state[key] = new_value
            self.version_counter += 1
            self.state_versions[key] = self.version_counter
            
            if notify:
                await self._notify_change(key, new_value)
            
            return new_value
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """获取锁"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    async def _notify_change(self, key: str, value: any):
        """通知订阅者"""
        handlers = self.subscribers.get(key, [])
        for handler in handlers:
            asyncio.create_task(handler(key, value))
    
    def subscribe(self, key: str, handler: Callable):
        """订阅状态变化"""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(handler)
    
    async def get_state_snapshot(self) -> Dict:
        """获取状态快照"""
        return {
            "state": dict(self.state),
            "versions": dict(self.state_versions),
            "global_version": self.version_counter
        }
```

---

## 任务分解与分配

### 智能任务分解器

```python
class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def decompose(
        self, 
        task: str,
        available_capabilities: List[str],
        max_depth: int = 3
    ) -> Dict:
        """
        智能分解任务
        
        Returns:
            任务树结构
        """
        prompt = f"""将以下复杂任务分解为可执行的子任务。

原始任务：{task}

可用能力：{', '.join(available_capabilities)}

请分析：
1. 任务的主要组成部分
2. 各部分之间的依赖关系
3. 每个子任务所需的能力
4. 执行顺序

以JSON格式返回任务树：
{{
    "root_task": "原始任务",
    "subtasks": [
        {{
            "id": "task_1",
            "description": "子任务描述",
            "required_capabilities": ["cap1", "cap2"],
            "dependencies": [],
            "estimated_effort": "low/medium/high",
            "can_parallel": true/false
        }}
    ],
    "execution_order": ["task_1", "task_2", ...]
}}"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        decomposition = json.loads(response.choices[0].message.content)
        
        # 验证分解结果
        validated = self._validate_decomposition(decomposition, available_capabilities)
        
        return validated
    
    def _validate_decomposition(self, decomposition: Dict, capabilities: List[str]) -> Dict:
        """验证分解结果"""
        subtasks = decomposition.get("subtasks", [])
        
        for task in subtasks:
            required = set(task.get("required_capabilities", []))
            available = set(capabilities)
            
            missing = required - available
            if missing:
                task["missing_capabilities"] = list(missing)
                task["warning"] = f"缺少能力: {missing}"
        
        return decomposition

class TaskAllocator:
    """任务分配器"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.agent_load = {agent.id: 0 for agent in agents}
    
    def allocate(self, task: Dict, strategy: str = "capability_best") -> Optional[Agent]:
        """
        分配任务给Agent
        
        Strategies:
        - capability_best: 选择能力最匹配的
        - load_balanced: 负载均衡
        - fastest: 选择历史执行最快的
        - cost_optimal: 成本最优
        """
        required_caps = set(task.get("required_capabilities", []))
        
        # 筛选有能力的Agent
        capable_agents = [
            agent for agent in self.agents
            if required_caps.issubset(set(agent.capabilities))
        ]
        
        if not capable_agents:
            return None
        
        if strategy == "capability_best":
            # 选择能力匹配度最高的
            return max(capable_agents, 
                      key=lambda a: len(set(a.capabilities) & required_caps))
        
        elif strategy == "load_balanced":
            # 选择负载最低的
            return min(capable_agents, key=lambda a: self.agent_load.get(a.id, 0))
        
        elif strategy == "cost_optimal":
            # 简化：选择能力最少的（假设能力越少成本越低）
            return min(capable_agents, key=lambda a: len(a.capabilities))
        
        return capable_agents[0]
    
    def update_load(self, agent_id: str, delta: int):
        """更新Agent负载"""
        self.agent_load[agent_id] = self.agent_load.get(agent_id, 0) + delta
```

---

## 实战案例与代码

### 完整Multi-Agent系统：智能研发团队

```python
class IntelligentRDTeam:
    """智能研发团队 - Multi-Agent协作系统"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.message_bus = MessageBus()
        self.state_manager = SharedStateManager()
        self.orchestrator = None
        self._setup_team()
    
    def _setup_team(self):
        """设置研发团队Agent"""
        # 创建各角色Agent
        self.product_manager = Agent(
            id="pm_001",
            name="产品经理",
            role="产品规划与需求分析",
            capabilities=["requirement_analysis", "prd_writing", "priority_management"],
            llm_client=self.llm_client
        )
        
        self.architect = Agent(
            id="arch_001",
            name="架构师",
            role="系统架构设计",
            capabilities=["architecture_design", "tech_selection", "api_design"],
            llm_client=self.llm_client
        )
        
        self.senior_dev = Agent(
            id="dev_001",
            name="高级开发",
            role="核心功能开发",
            capabilities=["coding", "code_review", "refactoring"],
            llm_client=self.llm_client
        )
        
        self.qa_engineer = Agent(
            id="qa_001",
            name="测试工程师",
            role="质量保证",
            capabilities=["test_design", "automation", "bug_analysis"],
            llm_client=self.llm_client
        )
        
        self.devops = Agent(
            id="ops_001",
            name="运维工程师",
            role="部署与运维",
            capabilities=["ci_cd", "monitoring", "troubleshooting"],
            llm_client=self.llm_client
        )
        
        # 注册到消息总线
        for agent in [self.product_manager, self.architect, self.senior_dev, 
                      self.qa_engineer, self.devops]:
            self.message_bus.register_agent(agent.id)
        
        # 设置层级关系
        self.product_manager.children = [self.architect]
        self.architect.children = [self.senior_dev, self.qa_engineer]
        self.senior_dev.children = [self.devops]
        
        # 创建编排器
        self.orchestrator = HierarchicalOrchestrator(self.product_manager)
    
    async def develop_feature(self, requirement: str) -> Dict:
        """开发新功能"""
        print("=" * 50)
        print(f"开始开发功能: {requirement}")
        print("=" * 50)
        
        # 阶段1: 需求分析
        print("\n[阶段1] 需求分析...")
        prd = await self._create_prd(requirement)
        await self.state_manager.set("prd", prd)
        
        # 阶段2: 架构设计
        print("\n[阶段2] 架构设计...")
        architecture = await self._design_architecture(prd)
        await self.state_manager.set("architecture", architecture)
        
        # 阶段3: 并行开发与测试设计
        print("\n[阶段3] 并行开发与测试...")
        dev_task = self._implement_code(architecture)
        test_task = self._design_tests(prd)
        
        code, test_plan = await asyncio.gather(dev_task, test_task)
        
        await self.state_manager.set("code", code)
        await self.state_manager.set("test_plan", test_plan)
        
        # 阶段4: 代码审查
        print("\n[阶段4] 代码审查...")
        review_result = await self._code_review(code, test_plan)
        await self.state_manager.set("review", review_result)
        
        # 阶段5: 部署
        print("\n[阶段5] 部署...")
        deployment = await self._deploy(code)
        
        return {
            "prd": prd,
            "architecture": architecture,
            "code": code,
            "test_plan": test_plan,
            "review": review_result,
            "deployment": deployment,
            "status": "completed"
        }
    
    async def _create_prd(self, requirement: str) -> str:
        """创建产品需求文档"""
        prompt = f"""作为产品经理，请将以下需求转化为详细的产品需求文档(PRD)：

需求：{requirement}

PRD应包含：
1. 背景与目标
2. 用户故事
3. 功能需求（详细描述）
4. 非功能需求
5. 验收标准
6. 优先级划分

请输出完整的PRD文档。"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _design_architecture(self, prd: str) -> str:
        """设计架构"""
        prompt = f"""作为架构师，请基于以下PRD设计系统架构：

{prd[:2000]}...

请提供：
1. 整体架构图（文字描述）
2. 核心组件设计
3. 数据模型
4. API设计
5. 技术选型及理由
6. 风险与应对"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _implement_code(self, architecture: str) -> str:
        """实现代码"""
        prompt = f"""作为高级开发工程师，请基于以下架构实现核心代码：

{architecture[:2000]}...

请提供：
1. 核心模块的代码实现（Python）
2. 关键算法实现
3. 必要的注释说明

代码要求：
- 遵循PEP8规范
- 包含类型提示
- 有适当的错误处理"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _design_tests(self, prd: str) -> str:
        """设计测试"""
        prompt = f"""作为测试工程师，请基于以下PRD设计测试方案：

{prd[:2000]}...

请提供：
1. 测试策略
2. 测试用例（含边界条件）
3. 自动化测试建议
4. 性能测试方案"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _code_review(self, code: str, test_plan: str) -> str:
        """代码审查"""
        prompt = f"""作为架构师，请对以下代码进行审查：

代码：
{code[:3000]}...

测试方案：
{test_plan[:1000]}...

请提供：
1. 代码质量评估
2. 潜在问题
3. 改进建议
4. 是否通过审查（通过/有条件通过/不通过）"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _deploy(self, code: str) -> str:
        """部署"""
        prompt = f"""作为运维工程师，请为以下代码设计部署方案：

代码概览：
{code[:1500]}...

请提供：
1. 部署架构
2. CI/CD流程
3. 监控方案
4. 回滚策略"""
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

# 使用示例
async def main():
    from openai import OpenAI
    
    client = OpenAI(api_key="your-api-key")
    team = IntelligentRDTeam(client)
    
    result = await team.develop_feature(
        "开发一个支持多Agent协作的任务管理系统，"
        "包含任务分配、进度跟踪、冲突解决功能"
    )
    
    print("\n" + "=" * 50)
    print("开发完成！")
    print(f"PRD长度: {len(result['prd'])} 字符")
    print(f"代码长度: {len(result['code'])} 字符")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 总结

### Multi-Agent设计原则

1. **单一职责**: 每个Agent专注于特定领域
2. **明确接口**: 定义清晰的通信协议和消息格式
3. **容错设计**: 单点故障不影响整体系统
4. **可观测性**: 完整的执行追踪和状态监控
5. **动态扩展**: 支持Agent的动态加入和退出

### 模式选择指南

| 场景 | 推荐模式 | 关键考量 |
|------|----------|----------|
| 企业审批流程 | 层级式 | 权限控制、审计追踪 |
| 头脑风暴/创意 | 工作组式 | 充分讨论、共识达成 |
| 数据处理ETL | 流水线式 | 效率、可复用性 |
| 客服系统 | 星型式 | 快速响应、负载均衡 |
| 资源调度 | 市场式 | 自适应性、公平性 |

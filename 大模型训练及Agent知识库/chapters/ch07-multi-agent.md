# 第七章：多Agent协作系统

> 多Agent协作系统是由多个智能体组成的复杂系统，各Agent通过协作完成单个Agent无法或难以完成的任务。本章将深入讲解多Agent协作模式、通信协议、任务分配与调度以及冲突解决机制。

## 目录

1. [协作模式](#1-协作模式)
2. [通信协议](#2-通信协议)
3. [任务分配与调度](#3-任务分配与调度)
4. [冲突解决机制](#4-冲突解决机制)

---

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: ../chapters/ch05-tool-calling.md, ../chapters/ch06-memory-system.md
- **关联文件**: ../chapters/ch08-agent-frameworks.md
- **最后更新**: 2026-06-12
---

## 1. 协作模式

### 1.1 层级式协作

**层级式协作（Hierarchical Collaboration）** 中，Agent按等级组织，上级Agent负责规划和协调，下级Agent负责执行具体任务。

```
┌─────────────────────────────────────────────────────────┐
│                   层级式协作架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│              ┌─────────────────┐                       │
│              │  主协调Agent    │                       │
│              │  (Coordinator)  │                       │
│              └────────┬────────┘                       │
│                       │                               │
│           ┌───────────┼───────────┐                   │
│           │           │           │                   │
│           ▼           ▼           ▼                   │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│    │ 子Agent1 │ │ 子Agent2 │ │ 子Agent3 │            │
│    └──────────┘ └──────────┘ └──────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**层级式协作的特点：**
- 决策集中化，控制清晰
- 适合结构化、可分解的任务
- 上级故障可能导致系统瘫痪

```python
from typing import List, Dict, Any
import json

class HierarchicalAgent:
    """层级式协作Agent基类"""
    
    def __init__(self, name: str, role: str, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.status = "idle"
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        raise NotImplementedError

class CoordinatorAgent(HierarchicalAgent):
    """主协调Agent"""
    
    def __init__(self, name: str):
        super().__init__(name, "coordinator", ["planning", "coordination"])
        self.sub_agents: List[HierarchicalAgent] = []
    
    def add_sub_agent(self, agent: HierarchicalAgent):
        """添加子Agent"""
        self.sub_agents.append(agent)
    
    def decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """任务分解"""
        decomposition_prompt = f"""Decompose the following task into subtasks:
Task: {task}

Return a JSON list of subtasks, each with 'name', 'description', and 'required_capability'.
"""
        # 实际实现会调用LLM进行任务分解
        # 这里返回示例
        return [
            {
                "name": "task1",
                "description": "Collect information",
                "required_capability": "research"
            },
            {
                "name": "task2", 
                "description": "Analyze data",
                "required_capability": "analysis"
            },
            {
                "name": "task3",
                "description": "Generate report",
                "required_capability": "writing"
            }
        ]
    
    def assign_subtask(self, subtask: Dict[str, Any]) -> HierarchicalAgent:
        """分配子任务"""
        for agent in self.sub_agents:
            if subtask["required_capability"] in agent.capabilities:
                return agent
        return None
    
    def coordinate(self, task: str) -> Dict[str, Any]:
        """协调执行"""
        print(f"[{self.name}] Coordinating task: {task}")
        
        # 1. 任务分解
        subtasks = self.decompose_task(task)
        print(f"[{self.name}] Decomposed into {len(subtasks)} subtasks")
        
        # 2. 任务分配与执行
        results = {}
        for subtask in subtasks:
            agent = self.assign_subtask(subtask)
            if agent:
                print(f"[{self.name}] Assigning '{subtask['name']}' to {agent.name}")
                result = agent.execute(subtask)
                results[subtask["name"]] = result
            else:
                print(f"[{self.name}] No suitable agent for '{subtask['name']}'")
        
        # 3. 结果整合
        final_result = self._integrate_results(results)
        return final_result
    
    def _integrate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """整合结果"""
        return {
            "status": "completed",
            "subtask_results": results,
            "summary": "All subtasks completed"
        }

class WorkerAgent(HierarchicalAgent):
    """工作Agent"""
    
    def __init__(self, name: str, capabilities: List[str]):
        super().__init__(name, "worker", capabilities)
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        self.status = "working"
        print(f"[{self.name}] Executing: {task['description']}")
        
        # 模拟执行
        result = {
            "task": task["name"],
            "agent": self.name,
            "status": "completed",
            "output": f"Result of {task['name']}"
        }
        
        self.status = "idle"
        return result

# 使用示例
# 创建协调者
coordinator = CoordinatorAgent("MainCoordinator")

# 创建工作Agent
researcher = WorkerAgent("Researcher", ["research", "data_collection"])
analyst = WorkerAgent("Analyst", ["analysis", "data_processing"])
writer = WorkerAgent("Writer", ["writing", "reporting"])

# 添加子Agent
coordinator.add_sub_agent(researcher)
coordinator.add_sub_agent(analyst)
coordinator.add_sub_agent(writer)

# 协调执行任务
result = coordinator.coordinate("Prepare a market research report")
print("\nFinal Result:", result)
```

### 1.2 工作组式协作

**工作组式协作（Team Collaboration）** 中，多个Agent平等协作，共同完成任务，通过协商达成一致。

| 特性 | 层级式 | 工作组式 |
|------|--------|----------|
| 决策方式 | 集中式 | 分布式 |
| Agent关系 | 主从 | 平等 |
| 通信模式 | 自上而下 | 点对点 |
| 容错性 | 低（单点故障） | 高 |
| 适用场景 | 结构化任务 | 复杂、动态任务 |

```python
from typing import List, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class Message:
    """消息类"""
    sender: str
    receiver: str
    content: str
    msg_type: str
    timestamp: float

class TeamAgent:
    """工作组Agent"""
    
    def __init__(self, name: str, expertise: List[str]):
        self.name = name
        self.expertise = expertise
        self.team: List['TeamAgent'] = []
        self.message_queue: List[Message] = []
        self.beliefs: Dict[str, Any] = {}
        self.goals: List[str] = []
    
    def join_team(self, team: List['TeamAgent']):
        """加入团队"""
        self.team = team
        if self not in team:
            team.append(self)
    
    def send_message(self, receiver: str, content: str, msg_type: str = "inform"):
        """发送消息"""
        msg = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            timestamp=time.time()
        )
        
        if receiver == "all":
            for agent in self.team:
                if agent.name != self.name:
                    agent.receive_message(msg)
        else:
            for agent in self.team:
                if agent.name == receiver:
                    agent.receive_message(msg)
    
    def receive_message(self, msg: Message):
        """接收消息"""
        self.message_queue.append(msg)
        print(f"[{self.name}] Received message from {msg.sender}: {msg.content}")
    
    def process_messages(self):
        """处理消息队列"""
        for msg in self.message_queue:
            self._handle_message(msg)
        self.message_queue.clear()
    
    def _handle_message(self, msg: Message):
        """处理单条消息"""
        if msg.msg_type == "proposal":
            self._handle_proposal(msg)
        elif msg.msg_type == "vote":
            self._handle_vote(msg)
        elif msg.msg_type == "inform":
            self._handle_inform(msg)
    
    def _handle_proposal(self, msg: Message):
        """处理提议"""
        # 评估提议
        # 发送投票
        self.send_message(msg.sender, "agree", "vote")
    
    def _handle_vote(self, msg: Message):
        """处理投票"""
        pass
    
    def _handle_inform(self, msg: Message):
        """处理信息"""
        # 更新信念
        self.beliefs[msg.sender] = msg.content
    
    def propose(self, proposal: str):
        """发起提议"""
        print(f"[{self.name}] Proposing: {proposal}")
        self.send_message("all", proposal, "proposal")
    
    def collaborate_on_task(self, task: str) -> Dict[str, Any]:
        """协作完成任务"""
        print(f"[{self.name}] Starting collaboration on: {task}")
        
        # 1. 广播任务
        self.send_message("all", f"Let's work on: {task}", "inform")
        
        # 2. 各Agent根据专长认领任务
        for agent in self.team:
            if any(exp in task.lower() for exp in agent.expertise):
                agent.send_message("all", f"I can handle this part", "inform")
        
        # 3. 消息处理循环
        for _ in range(3):
            for agent in self.team:
                agent.process_messages()
            time.sleep(0.1)
        
        return {"status": "collaborated", "task": task}

# 使用示例
# 创建团队成员
agent1 = TeamAgent("Alice", ["design", "planning"])
agent2 = TeamAgent("Bob", ["coding", "implementation"])
agent3 = TeamAgent("Charlie", ["testing", "review"])

# 加入团队
team = [agent1, agent2, agent3]
agent1.join_team(team)
agent2.join_team(team)
agent3.join_team(team)

# 协作
result = agent1.collaborate_on_task("Design and implement a new feature")
```

### 1.3 流水线式协作

**流水线式协作（Pipeline Collaboration）** 中，任务被分解为多个步骤，每个Agent负责一个特定步骤，按顺序传递。

```
任务输入 → [AgentA:预处理] → [AgentB:分析] → [AgentC:生成] → 任务输出
```

```python
from typing import List, Any, Callable

class PipelineStage:
    """流水线阶段"""
    
    def __init__(self, name: str, processor: Callable[[Any], Any]):
        self.name = name
        self.processor = processor
    
    def process(self, data: Any) -> Any:
        """处理数据"""
        print(f"[Pipeline] Stage '{self.name}' processing...")
        return self.processor(data)

class PipelineAgent:
    """流水线协作Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
    
    def add_stage(self, name: str, processor: Callable[[Any], Any]):
        """添加流水线阶段"""
        self.stages.append(PipelineStage(name, processor))
    
    def execute(self, input_data: Any) -> Any:
        """执行流水线"""
        print(f"[{self.name}] Starting pipeline execution")
        
        current_data = input_data
        for stage in self.stages:
            current_data = stage.process(current_data)
        
        print(f"[{self.name}] Pipeline completed")
        return current_data

# 使用示例
def preprocess(text: str) -> str:
    """预处理：清理文本"""
    return text.strip().lower()

def analyze(text: str) -> Dict[str, Any]:
    """分析：提取关键词"""
    words = text.split()
    return {
        "original": text,
        "word_count": len(words),
        "keywords": [w for w in words if len(w) > 3]
    }

def generate(data: Dict[str, Any]) -> str:
    """生成：创建摘要"""
    return f"Text has {data['word_count']} words. Key topics: {', '.join(data['keywords'][:5])}"

# 创建流水线
pipeline = PipelineAgent("TextProcessingPipeline")
pipeline.add_stage("Preprocess", preprocess)
pipeline.add_stage("Analyze", analyze)
pipeline.add_stage("Generate", generate)

# 执行
input_text = "  This is a sample text for demonstrating the pipeline collaboration pattern.  "
result = pipeline.execute(input_text)
print("\nFinal Output:", result)
```

---

## 2. 通信协议

### 2.1 消息传递

**消息传递（Message Passing）** 是Agent间最基本的通信方式，通过发送和接收消息进行交互。

```python
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time

class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"

@dataclass
class AgentMessage:
    """Agent消息"""
    message_id: str
    sender: str
    receiver: str
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    in_reply_to: Optional[str] = None

class MessageQueue:
    """消息队列"""
    
    def __init__(self):
        self.queue: List[AgentMessage] = []
    
    def send(self, message: AgentMessage):
        """发送消息"""
        self.queue.append(message)
        print(f"[MessageQueue] {message.sender} → {message.receiver}: {message.msg_type.value}")
    
    def receive(self, receiver: str) -> Optional[AgentMessage]:
        """接收消息"""
        for i, msg in enumerate(self.queue):
            if msg.receiver == receiver or msg.receiver == "all":
                return self.queue.pop(i)
        return None
    
    def get_all_for(self, receiver: str) -> List[AgentMessage]:
        """获取所有给指定接收者的消息"""
        messages = []
        remaining = []
        for msg in self.queue:
            if msg.receiver == receiver or msg.receiver == "all":
                messages.append(msg)
            else:
                remaining.append(msg)
        self.queue = remaining
        return messages

class CommunicatingAgent:
    """支持通信的Agent"""
    
    def __init__(self, name: str, message_queue: MessageQueue):
        self.name = name
        self.message_queue = message_queue
        self.message_id_counter = 0
        self.handlers: Dict[MessageType, Callable] = {}
    
    def _generate_message_id(self) -> str:
        """生成消息ID"""
        self.message_id_counter += 1
        return f"{self.name}-{self.message_id_counter}-{int(time.time())}"
    
    def send_message(self, receiver: str, msg_type: MessageType, 
                    content: Dict[str, Any], in_reply_to: Optional[str] = None):
        """发送消息"""
        msg = AgentMessage(
            message_id=self._generate_message_id(),
            sender=self.name,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            timestamp=time.time(),
            in_reply_to=in_reply_to
        )
        self.message_queue.send(msg)
        return msg.message_id
    
    def receive_messages(self) -> List[AgentMessage]:
        """接收消息"""
        return self.message_queue.get_all_for(self.name)
    
    def register_handler(self, msg_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.handlers[msg_type] = handler
    
    def process_messages(self):
        """处理收到的消息"""
        messages = self.receive_messages()
        for msg in messages:
            if msg.msg_type in self.handlers:
                self.handlers[msg.msg_type](msg)
            else:
                print(f"[{self.name}] No handler for {msg.msg_type}")

# 使用示例
# 创建消息队列
mq = MessageQueue()

# 创建Agent
alice = CommunicatingAgent("Alice", mq)
bob = CommunicatingAgent("Bob", mq)

# 注册处理器
def handle_request(msg: AgentMessage):
    print(f"[Bob] Received request from {msg.sender}: {msg.content}")
    # 发送响应
    bob.send_message(
        msg.sender,
        MessageType.RESPONSE,
        {"result": "processed", "data": msg.content},
        in_reply_to=msg.message_id
    )

def handle_response(msg: AgentMessage):
    print(f"[Alice] Received response from {msg.sender}: {msg.content}")

bob.register_handler(MessageType.REQUEST, handle_request)
alice.register_handler(MessageType.RESPONSE, handle_response)

# Alice发送请求给Bob
alice.send_message("Bob", MessageType.REQUEST, {"task": "calculate", "value": 42})

# 处理消息
print()
bob.process_messages()
print()
alice.process_messages()
```

### 2.2 共享状态

**共享状态（Shared State）** 通过共享数据存储实现Agent间的间接通信。

```python
from typing import Dict, Any, List, Optional
from threading import Lock
from dataclasses import dataclass, field

@dataclass
class SharedState:
    """共享状态"""
    data: Dict[str, Any] = field(default_factory=dict)
    subscribers: List[str] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def get(self, key: str) -> Optional[Any]:
        """获取状态"""
        with self._lock:
            return self.data.get(key)
    
    def set(self, key: str, value: Any, publisher: str):
        """设置状态"""
        with self._lock:
            old_value = self.data.get(key)
            self.data[key] = value
            print(f"[SharedState] {publisher} updated '{key}': {old_value} → {value}")
    
    def subscribe(self, agent_name: str):
        """订阅状态变化"""
        with self._lock:
            if agent_name not in self.subscribers:
                self.subscribers.append(agent_name)
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有状态"""
        with self._lock:
            return self.data.copy()

class SharedStateAgent:
    """使用共享状态的Agent"""
    
    def __init__(self, name: str, shared_state: SharedState):
        self.name = name
        self.shared_state = shared_state
        self.shared_state.subscribe(name)
        self.local_cache: Dict[str, Any] = {}
    
    def write_state(self, key: str, value: Any):
        """写入共享状态"""
        self.shared_state.set(key, value, self.name)
    
    def read_state(self, key: str) -> Optional[Any]:
        """读取共享状态"""
        return self.shared_state.get(key)
    
    def sync_state(self):
        """同步状态到本地"""
        self.local_cache = self.shared_state.get_all()

# 使用示例
# 创建共享状态
shared_state = SharedState()

# 创建Agent
agent1 = SharedStateAgent("Agent1", shared_state)
agent2 = SharedStateAgent("Agent2", shared_state)
agent3 = SharedStateAgent("Agent3", shared_state)

# Agent1写入状态
agent1.write_state("task_status", "started")
agent1.write_state("data", [1, 2, 3])

# Agent2读取并更新
print("\nAgent2 reads task_status:", agent2.read_state("task_status"))
agent2.write_state("task_status", "processing")
agent2.write_state("analysis_result", {"sum": 6, "count": 3})

# Agent3读取
print("\nAgent3 reads all:")
agent3.sync_state()
for key, value in agent3.local_cache.items():
    print(f"  {key}: {value}")
```

---

## 3. 任务分配与调度

### 3.1 任务分配策略

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """任务"""
    task_id: str
    description: str
    required_skills: List[str]
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    estimated_time: float = 1.0

@dataclass
class AgentCapability:
    """Agent能力"""
    name: str
    skills: List[str]
    workload: float = 0.0
    max_workload: float = 10.0
    efficiency: Dict[str, float] = None
    
    def __post_init__(self):
        if self.efficiency is None:
            self.efficiency = {skill: 1.0 for skill in self.skills}

class TaskAllocator:
    """任务分配器"""
    
    def __init__(self, agents: List[AgentCapability]):
        self.agents = agents
        self.tasks: List[Task] = []
    
    def add_task(self, task: Task):
        """添加任务"""
        self.tasks.append(task)
    
    def _score_agent_for_task(self, agent: AgentCapability, task: Task) -> float:
        """计算Agent对任务的适合度分数"""
        score = 0.0
        
        # 1. 技能匹配
        matching_skills = set(agent.skills) & set(task.required_skills)
        skill_score = len(matching_skills) / len(task.required_skills)
        score += skill_score * 0.5
        
        # 2. 工作负载
        workload_score = 1.0 - (agent.workload / agent.max_workload)
        score += workload_score * 0.3
        
        # 3. 效率
        if task.required_skills:
            efficiency_scores = [agent.efficiency.get(s, 0.5) for s in task.required_skills]
            efficiency_score = sum(efficiency_scores) / len(efficiency_scores)
            score += efficiency_score * 0.2
        
        return score
    
    def assign_round_robin(self) -> Dict[str, List[Task]]:
        """轮询分配"""
        assignments = {agent.name: [] for agent in self.agents}
        agent_idx = 0
        
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        
        for task in pending_tasks:
            # 找到有技能的Agent
            found = False
            for _ in range(len(self.agents)):
                agent = self.agents[agent_idx]
                agent_idx = (agent_idx + 1) % len(self.agents)
                
                if any(skill in agent.skills for skill in task.required_skills):
                    task.assigned_to = agent.name
                    task.status = TaskStatus.ASSIGNED
                    assignments[agent.name].append(task)
                    found = True
                    break
            
            if not found:
                print(f"[Allocator] No suitable agent for task {task.task_id}")
        
        return assignments
    
    def assign_by_capability(self) -> Dict[str, List[Task]]:
        """按能力分配"""
        assignments = {agent.name: [] for agent in self.agents}
        
        # 按优先级排序任务
        pending_tasks = sorted(
            [t for t in self.tasks if t.status == TaskStatus.PENDING],
            key=lambda t: -t.priority
        )
        
        for task in pending_tasks:
            best_agent = None
            best_score = -1
            
            for agent in self.agents:
                # 检查是否有必要技能
                if not any(skill in agent.skills for skill in task.required_skills):
                    continue
                
                # 检查工作负载
                if agent.workload + task.estimated_time > agent.max_workload:
                    continue
                
                score = self._score_agent_for_task(agent, task)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                task.assigned_to = best_agent.name
                task.status = TaskStatus.ASSIGNED
                best_agent.workload += task.estimated_time
                assignments[best_agent.name].append(task)
                print(f"[Allocator] Assigned {task.task_id} to {best_agent.name} (score: {best_score:.2f})")
            else:
                print(f"[Allocator] No suitable agent for task {task.task_id}")
        
        return assignments

# 使用示例
# 创建Agent
agents = [
    AgentCapability(
        name="AgentA",
        skills=["coding", "debugging"],
        efficiency={"coding": 0.9, "debugging": 0.8}
    ),
    AgentCapability(
        name="AgentB",
        skills=["design", "coding"],
        efficiency={"design": 0.95, "coding": 0.7}
    ),
    AgentCapability(
        name="AgentC",
        skills=["testing", "documentation"],
        efficiency={"testing": 0.85, "documentation": 0.9}
    )
]

# 创建任务分配器
allocator = TaskAllocator(agents)

# 添加任务
tasks = [
    Task("T1", "Implement feature X", ["coding"], priority=3),
    Task("T2", "Design system architecture", ["design"], priority=2),
    Task("T3", "Write documentation", ["documentation"], priority=1),
    Task("T4", "Debug issue Y", ["debugging", "coding"], priority=3),
    Task("T5", "Test feature X", ["testing"], priority=2)
]

for task in tasks:
    allocator.add_task(task)

# 按能力分配
print("=== Capability-based Assignment ===")
assignments = allocator.assign_by_capability()
for agent_name, agent_tasks in assignments.items():
    print(f"\n{agent_name}:")
    for task in agent_tasks:
        print(f"  - {task.task_id}: {task.description}")
```

### 3.2 任务调度

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
import time

class ScheduleStrategy(Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_JOB_FIRST = "sjf"
    EARLIEST_DEADLINE_FIRST = "edf"

@dataclass
class ScheduledTask(Task):
    """带调度信息的任务"""
    deadline: Optional[float] = None
    submit_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY):
        self.strategy = strategy
        self.task_queue: List[ScheduledTask] = []
        self.completed_tasks: List[ScheduledTask] = []
    
    def submit_task(self, task: ScheduledTask):
        """提交任务"""
        task.submit_time = time.time()
        self.task_queue.append(task)
        print(f"[Scheduler] Submitted task {task.task_id}")
    
    def _get_priority_key(self, task: ScheduledTask) -> tuple:
        """获取优先级键"""
        if self.strategy == ScheduleStrategy.FIFO:
            return (task.submit_time,)
        elif self.strategy == ScheduleStrategy.PRIORITY:
            return (-task.priority, task.submit_time)  # 高优先级优先
        elif self.strategy == ScheduleStrategy.SHORTEST_JOB_FIRST:
            return (task.estimated_time, -task.priority)
        elif self.strategy == ScheduleStrategy.EARLIEST_DEADLINE_FIRST:
            return (task.deadline or float('inf'), -task.priority)
        return (task.submit_time,)
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        """获取下一个要执行的任务"""
        if not self.task_queue:
            return None
        
        # 按策略排序
        self.task_queue.sort(key=self._get_priority_key)
        task = self.task_queue.pop(0)
        task.status = TaskStatus.IN_PROGRESS
        task.start_time = time.time()
        return task
    
    def complete_task(self, task: ScheduledTask):
        """完成任务"""
        task.status = TaskStatus.COMPLETED
        task.end_time = time.time()
        self.completed_tasks.append(task)
        print(f"[Scheduler] Completed task {task.task_id} in {task.end_time - task.start_time:.2f}s")
    
    def get_schedule(self) -> List[ScheduledTask]:
        """获取调度队列"""
        return sorted(self.task_queue, key=self._get_priority_key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.completed_tasks:
            return {"total_completed": 0}
        
        total_time = sum(t.end_time - t.start_time for t in self.completed_tasks)
        avg_time = total_time / len(self.completed_tasks)
        
        return {
            "total_completed": len(self.completed_tasks),
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "pending_tasks": len(self.task_queue)
        }

# 使用示例
# 创建调度器
scheduler = TaskScheduler(strategy=ScheduleStrategy.PRIORITY)

# 提交任务
now = time.time()
scheduler.submit_task(ScheduledTask("T1", "High priority task", ["coding"], priority=3, estimated_time=2.0))
scheduler.submit_task(ScheduledTask("T2", "Medium priority task", ["design"], priority=2, estimated_time=3.0))
scheduler.submit_task(ScheduledTask("T3", "Low priority task", ["testing"], priority=1, estimated_time=1.0))
scheduler.submit_task(ScheduledTask("T4", "Another high priority", ["coding"], priority=3, estimated_time=1.5))

# 模拟执行
print("\n=== Executing Tasks ===")
while True:
    task = scheduler.get_next_task()
    if not task:
        break
    
    print(f"\nExecuting: {task.task_id} (Priority: {task.priority})")
    time.sleep(task.estimated_time * 0.1)  # 模拟执行
    scheduler.complete_task(task)

# 统计信息
print("\n=== Statistics ===")
stats = scheduler.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value}")
```

---

## 4. 冲突解决机制

### 4.1 冲突类型与检测

```python
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

class ConflictType(Enum):
    """冲突类型"""
    RESOURCE_CONFLICT = "resource_conflict"
    TASK_CONFLICT = "task_conflict"
    GOAL_CONFLICT = "goal_conflict"
    BELIEF_CONFLICT = "belief_conflict"

@dataclass
class Conflict:
    """冲突"""
    conflict_id: str
    conflict_type: ConflictType
    agents_involved: List[str]
    description: str
    resources: Optional[List[str]] = None
    severity: int = 1  # 1-5

class ConflictDetector:
    """冲突检测器"""
    
    def __init__(self):
        self.resource_usage: Dict[str, List[str]] = {}
        self.agent_goals: Dict[str, Set[str]] = {}
        self.agent_beliefs: Dict[str, Dict[str, Any]] = {}
    
    def register_resource_usage(self, resource: str, agent: str):
        """注册资源使用"""
        if resource not in self.resource_usage:
            self.resource_usage[resource] = []
        self.resource_usage[resource].append(agent)
    
    def register_goal(self, agent: str, goal: str):
        """注册Agent目标"""
        if agent not in self.agent_goals:
            self.agent_goals[agent] = set()
        self.agent_goals[agent].add(goal)
    
    def register_belief(self, agent: str, key: str, value: Any):
        """注册Agent信念"""
        if agent not in self.agent_beliefs:
            self.agent_beliefs[agent] = {}
        self.agent_beliefs[agent][key] = value
    
    def detect_resource_conflicts(self) -> List[Conflict]:
        """检测资源冲突"""
        conflicts = []
        conflict_id = 0
        
        for resource, agents in self.resource_usage.items():
            if len(agents) > 1:
                conflict_id += 1
                conflicts.append(Conflict(
                    conflict_id=f"RC-{conflict_id}",
                    conflict_type=ConflictType.RESOURCE_CONFLICT,
                    agents_involved=agents,
                    description=f"Multiple agents trying to use resource: {resource}",
                    resources=[resource],
                    severity=min(2 + len(agents), 5)
                ))
        
        return conflicts
    
    def detect_goal_conflicts(self, incompatible_goals: List[Set[str]]) -> List[Conflict]:
        """检测目标冲突"""
        conflicts = []
        conflict_id = 0
        
        all_agents = list(self.agent_goals.keys())
        
        for i in range(len(all_agents)):
            for j in range(i + 1, len(all_agents)):
                agent1 = all_agents[i]
                agent2 = all_agents[j]
                
                goals1 = self.agent_goals.get(agent1, set())
                goals2 = self.agent_goals.get(agent2, set())
                
                # 检查是否有不兼容的目标组合
                for incomp_set in incompatible_goals:
                    if (incomp_set & goals1) and (incomp_set & goals2):
                        conflict_id += 1
                        conflicts.append(Conflict(
                            conflict_id=f"GC-{conflict_id}",
                            conflict_type=ConflictType.GOAL_CONFLICT,
                            agents_involved=[agent1, agent2],
                            description=f"Agents have incompatible goals",
                            severity=3
                        ))
        
        return conflicts
    
    def detect_belief_conflicts(self) -> List[Conflict]:
        """检测信念冲突"""
        conflicts = []
        conflict_id = 0
        
        all_agents = list(self.agent_beliefs.keys())
        all_keys = set()
        for beliefs in self.agent_beliefs.values():
            all_keys.update(beliefs.keys())
        
        for key in all_keys:
            agents_with_key = [a for a in all_agents if key in self.agent_beliefs[a]]
            if len(agents_with_key) < 2:
                continue
            
            # 检查是否有不同的信念值
            values = {}
            for agent in agents_with_key:
                val = self.agent_beliefs[agent][key]
                val_str = str(val)
                if val_str not in values:
                    values[val_str] = []
                values[val_str].append(agent)
            
            if len(values) > 1:
                conflict_id += 1
                conflicts.append(Conflict(
                    conflict_id=f"BC-{conflict_id}",
                    conflict_type=ConflictType.BELIEF_CONFLICT,
                    agents_involved=agents_with_key,
                    description=f"Disagreement on belief '{key}': {values}",
                    severity=2
                ))
        
        return conflicts
    
    def detect_all_conflicts(self, incompatible_goals: Optional[List[Set[str]]] = None) -> List[Conflict]:
        """检测所有冲突"""
        conflicts = []
        conflicts.extend(self.detect_resource_conflicts())
        if incompatible_goals:
            conflicts.extend(self.detect_goal_conflicts(incompatible_goals))
        conflicts.extend(self.detect_belief_conflicts())
        return conflicts

# 使用示例
detector = ConflictDetector()

# 注册资源使用
detector.register_resource_usage("database", "AgentA")
detector.register_resource_usage("database", "AgentB")
detector.register_resource_usage("printer", "AgentC")
detector.register_resource_usage("printer", "AgentA")

# 注册目标
detector.register_goal("AgentA", "maximize_profit")
detector.register_goal("AgentA", "improve_quality")
detector.register_goal("AgentB", "minimize_cost")
detector.register_goal("AgentB", "fast_delivery")

# 注册信念
detector.register_belief("AgentA", "market_trend", "up")
detector.register_belief("AgentB", "market_trend", "down")
detector.register_belief("AgentC", "market_trend", "stable")

# 检测冲突
print("=== Detected Conflicts ===")
incompatible = [{"maximize_profit", "minimize_cost"}]
conflicts = detector.detect_all_conflicts(incompatible)

for conflict in conflicts:
    print(f"\n[{conflict.conflict_id}] {conflict.conflict_type.value}")
    print(f"  Agents: {conflict.agents_involved}")
    print(f"  Description: {conflict.description}")
    print(f"  Severity: {conflict.severity}/5")
```

### 4.2 冲突解决策略

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

class ResolutionStrategy(Enum):
    """解决策略"""
    PRIORITY = "priority"
    VOTING = "voting"
    NEGOTIATION = "negotiation"
    ARBITRATION = "arbitration"
    COMPROMISE = "compromise"

@dataclass
class ResolutionResult:
    """解决结果"""
    conflict_id: str
    resolved: bool
    strategy: ResolutionStrategy
    winner: Optional[str] = None
    agreement: Optional[Any] = None
    explanation: str = ""

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.agent_priorities: Dict[str, int] = {}
        self.arbitrator: Optional[str] = None
    
    def set_agent_priority(self, agent: str, priority: int):
        """设置Agent优先级"""
        self.agent_priorities[agent] = priority
    
    def set_arbitrator(self, arbitrator: str):
        """设置仲裁者"""
        self.arbitrator = arbitrator
    
    def resolve_by_priority(self, conflict: Conflict) -> ResolutionResult:
        """按优先级解决"""
        if not self.agent_priorities:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=False,
                strategy=ResolutionStrategy.PRIORITY,
                explanation="No priorities set"
            )
        
        # 选择优先级最高的Agent
        agents = conflict.agents_involved
        priorities = [(self.agent_priorities.get(a, 0), a) for a in agents]
        priorities.sort(reverse=True)
        winner = priorities[0][1]
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            resolved=True,
            strategy=ResolutionStrategy.PRIORITY,
            winner=winner,
            explanation=f"{winner} has highest priority"
        )
    
    def resolve_by_voting(self, conflict: Conflict, 
                        voters: List[str], preferences: Dict[str, str]) -> ResolutionResult:
        """投票解决"""
        votes = {}
        for voter in voters:
            choice = preferences.get(voter)
            if choice:
                votes[choice] = votes.get(choice, 0) + 1
        
        if not votes:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=False,
                strategy=ResolutionStrategy.VOTING,
                explanation="No valid votes"
            )
        
        winner = max(votes.items(), key=lambda x: x[1])[0]
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            resolved=True,
            strategy=ResolutionStrategy.VOTING,
            winner=winner,
            explanation=f"Won with {votes[winner]} votes"
        )
    
    def resolve_by_negotiation(self, conflict: Conflict,
                             utility_functions: Dict[str, Callable]) -> ResolutionResult:
        """协商解决"""
        agents = conflict.agents_involved
        
        # 寻找对双方都可接受的方案
        # 简化实现：随机选择一个折衷方案
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            resolved=True,
            strategy=ResolutionStrategy.NEGOTIATION,
            agreement="Compromise reached",
            explanation="Negotiated settlement"
        )
    
    def resolve_by_arbitration(self, conflict: Conflict) -> ResolutionResult:
        """仲裁解决"""
        if not self.arbitrator:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=False,
                strategy=ResolutionStrategy.ARBITRATION,
                explanation="No arbitrator set"
            )
        
        # 模拟仲裁者决策
        winner = conflict.agents_involved[0] if conflict.agents_involved else None
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            resolved=True,
            strategy=ResolutionStrategy.ARBITRATION,
            winner=winner,
            explanation=f"Arbitrated by {self.arbitrator}"
        )
    
    def resolve(self, conflict: Conflict, 
               strategy: Optional[ResolutionStrategy] = None,
               **kwargs) -> ResolutionResult:
        """解决冲突"""
        # 根据冲突严重程度选择策略
        if strategy is None:
            if conflict.severity >= 4:
                strategy = ResolutionStrategy.ARBITRATION
            elif conflict.severity >= 2:
                strategy = ResolutionStrategy.PRIORITY
            else:
                strategy = ResolutionStrategy.VOTING
        
        if strategy == ResolutionStrategy.PRIORITY:
            return self.resolve_by_priority(conflict)
        elif strategy == ResolutionStrategy.VOTING:
            return self.resolve_by_voting(conflict, 
                                         kwargs.get("voters", []),
                                         kwargs.get("preferences", {}))
        elif strategy == ResolutionStrategy.NEGOTIATION:
            return self.resolve_by_negotiation(conflict, 
                                              kwargs.get("utility_functions", {}))
        elif strategy == ResolutionStrategy.ARBITRATION:
            return self.resolve_by_arbitration(conflict)
        else:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=False,
                strategy=strategy,
                explanation="Unknown strategy"
            )

# 使用示例
resolver = ConflictResolver()

# 设置优先级
resolver.set_agent_priority("AgentA", 3)
resolver.set_agent_priority("AgentB", 2)
resolver.set_agent_priority("AgentC", 1)

# 设置仲裁者
resolver.set_arbitrator("SuperAgent")

# 创建示例冲突
conflict = Conflict(
    conflict_id="RC-1",
    conflict_type=ConflictType.RESOURCE_CONFLICT,
    agents_involved=["AgentA", "AgentB"],
    description="Conflict over database access",
    resources=["database"],
    severity=3
)

# 解决冲突
print("=== Resolving Conflict ===")

# 按优先级解决
result = resolver.resolve(conflict, ResolutionStrategy.PRIORITY)
print(f"\nPriority resolution:")
print(f"  Resolved: {result.resolved}")
print(f"  Winner: {result.winner}")
print(f"  Explanation: {result.explanation}")

# 投票解决
voters = ["AgentA", "AgentB", "AgentC"]
preferences = {"AgentA": "AgentA", "AgentB": "AgentB", "AgentC": "AgentA"}
result = resolver.resolve(conflict, ResolutionStrategy.VOTING,
                         voters=voters, preferences=preferences)
print(f"\nVoting resolution:")
print(f"  Resolved: {result.resolved}")
print(f"  Winner: {result.winner}")
print(f"  Explanation: {result.explanation}")
```

---

## 深度分析

多Agent协作系统代表了从"单一智能体"到"智能体社会"的范式跃迁。当单个Agent难以应对复杂、跨领域的任务时，通过多个专业化Agent的协作可以实现能力的组合涌现。本章介绍的三种协作模式——层级式、工作组式和流水线式——分别对应了组织结构中常见的集中控制、民主协商和流水线生产三种模式。层级式适合任务分解清晰的场景，但存在单点故障风险；工作组式灵活且容错性强，但协商开销较大；流水线式吞吐量高，但延迟受最慢环节制约。实际系统往往需要根据任务特性动态选择合适的协作模式，甚至在同一系统中混合多种模式。

通信协议是决定多Agent系统效率和可扩展性的关键设计维度。消息传递（Message Passing）提供了Agent间的显式通信通道，适合需要精确控制交互流程的场景，但其异步特性和网络开销在高频交互场景下可能成为瓶颈。共享状态（Shared State）则通过共享数据存储实现隐式协作，减少了消息传递的开销，但引入了数据一致性和并发控制的挑战。在实际工程中，建议采用混合通信模式：关键决策通过消息传递保证可靠性，非敏感的中间状态通过共享存储提高效率。

任务分配与冲突解决是多Agent系统从"能用"走向"好用"必须跨越的两道门槛。任务分配本质上是一个优化问题——在Agent能力、工作负载、任务优先级等多个约束条件下寻找最优匹配。轮询分配简单但低效，按能力分配效果更优但需要准确的Agent画像。冲突解决则需要应对资源争用、目标矛盾、信念不一致等多类冲突场景，优先级仲裁适用于有明确等级的系统，投票和协商更适合平等协作的环境。一个值得关注的方向是引入元Agent（Meta-Agent）来动态监控和优化整个系统的协作效率，实现真正的自适应多Agent管理。

---

## Checklist

- [ ] 理解三种协作模式（层级式、工作组式、流水线式）的适用场景
- [ ] 实现层级式协作系统，包括Coordinator和Worker的交互
- [ ] 实现工作组式协作系统，包括消息传递和协商机制
- [ ] 理解消息队列和消息类型（REQUEST/RESPONSE/INFORM等）的设计
- [ ] 实现基于共享状态的隐式协作模式
- [ ] 掌握任务分配策略：轮询分配和按能力分配
- [ ] 实现任务调度器，支持FIFO/PRIORITY/SJF/EDF多种策略
- [ ] 理解冲突类型（资源/目标/信念冲突）及其检测方法
- [ ] 实现冲突解决策略：优先级/投票/协商/仲裁
- [ ] 设计和测试一个包含3个以上Agent的多Agent协作系统

---

## 延伸阅读

- [第五章：工具调用与Function Calling](../chapters/ch05-tool-calling.md) - 多Agent系统中的工具调用
- [第六章：Agent记忆系统](../chapters/ch06-memory-system.md) - 多Agent记忆共享策略
- [第八章：Agent框架与实践](../chapters/ch08-agent-frameworks.md) - AutoGen框架的多Agent实现
- AutoGen官方文档 - https://microsoft.github.io/autogen/
- Multi-Agent Systems: A Survey - 多Agent系统综述

---

## 本章小结

多Agent协作系统是构建复杂智能系统的关键：

1. **协作模式**：层级式适合结构化任务，工作组式适合动态复杂任务，流水线式适合可分解的流程任务
2. **通信协议**：消息传递提供显式通信，共享状态提供隐式协作
3. **任务分配与调度**：根据Agent能力和任务特性进行智能分配，多种调度策略优化执行效率
4. **冲突解决**：识别冲突类型，采用优先级、投票、协商、仲裁等策略解决冲突

**下一章：** 我们将学习Agent框架与实践。

---

*最后更新: 2026-06-12*

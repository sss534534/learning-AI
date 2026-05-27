**多Agent API调用系统**

**详细设计文档**

(面向研发人员)

版本: V1.0

日期: 2026年5月

配套文档: 架构设计文档 V1.0

一、项目结构与模块划分

1.1 技术栈

  ------------ ---------------------------- --------------
  **领域**     **技术选型**                 **版本要求**
  开发语言     TypeScript                   \>= 5.2
  运行时       Node.js                      \>= 20 LTS
  Web框架      Fastify                      \>= 4.x
  消息队列     Kafka (kafkajs)              \>= 2.x
  缓存         Redis (ioredis)              \>= 7.x
  向量数据库   Milvus                       \>= 2.3
  LLM SDK      OpenAI SDK / Anthropic SDK   最新版
  任务调度     BullMQ                       \>= 5.x
  日志         Pino                         \>= 8.x
  测试         Vitest + Supertest           \>= 1.x
  ------------ ---------------------------- --------------

1.2 目录结构

multi-agent-api-system/

├── src/

│ ├── gateway/ \# 接入层

│ │ ├── auth.ts \# 认证中间件

│ │ ├── rate-limiter.ts \# 限流

│ │ ├── session.ts \# 会话管理

│ │ └── routes.ts \# 路由定义

│ │

│ ├── orchestrator/ \# 编排层

│ │ ├── orchestrator.ts \# 编排器核心

│ │ ├── planner.ts \# 任务规划器

│ │ ├── dag-executor.ts \# DAG执行引擎

│ │ └── state-manager.ts \# 状态管理

│ │

│ ├── agents/ \# Agent层

│ │ ├── base-agent.ts \# Agent基类

│ │ ├── data-agent.ts \# 数据Agent

│ │ ├── auth-agent.ts \# 认证Agent

│ │ ├── notify-agent.ts \# 通知Agent

│ │ └── agent-registry.ts \# Agent注册中心

│ │

│ ├── adapter/ \# 适配层

│ │ ├── tool-registry.ts \# 工具注册中心

│ │ ├── openapi-loader.ts \# OpenAPI加载器

│ │ ├── api-adapter.ts \# API适配器

│ │ ├── param-transformer.ts \# 参数转换器

│ │ ├── error-handler.ts \# 错误处理器

│ │ └── response-trimmer.ts \# 响应裁剪器

│ │

│ ├── messaging/ \# 消息通信

│ │ ├── kafka-client.ts \# Kafka客户端

│ │ ├── message-types.ts \# 消息类型定义

│ │ └── message-bus.ts \# 消息总线

│ │

│ ├── knowledge/ \# 知识增强

│ │ ├── tool-graph.ts \# 工具知识图谱

│ │ ├── rag-retriever.ts \# RAG检索器

│ │ └── embedding.ts \# 向量化服务

│ │

│ ├── security/ \# 安全模块

│ │ ├── rbac.ts \# 角色权限控制

│ │ ├── param-validator.ts \# 参数校验

│ │ └── audit-logger.ts \# 审计日志

│ │

│ ├── observability/ \# 可观测性

│ │ ├── metrics.ts \# 指标采集

│ │ ├── tracing.ts \# 链路追踪

│ │ └── alerting.ts \# 告警规则

│ │

│ ├── shared/ \# 共享模块

│ │ ├── types.ts \# 类型定义

│ │ ├── config.ts \# 配置管理

│ │ ├── errors.ts \# 错误定义

│ │ └── utils.ts \# 工具函数

│ │

│ └── index.ts \# 入口文件

├── tests/ \# 测试

├── configs/ \# 配置文件

├── openapi-specs/ \# OpenAPI定义文件

└── package.json

二、核心类型定义

2.1 基础枚举与常量

2.1.1 AgentType 枚举

// src/shared/types.ts

export enum AgentType {

ORCHESTRATOR = \'orchestrator\',

PLANNER = \'planner\',

DATA = \'data\_agent\',

AUTH = \'auth\_agent\',

NOTIFY = \'notify\_agent\',

}

2.1.2 RiskLevel 枚举

export enum RiskLevel {

READ = \'READ\',

WRITE = \'WRITE\',

DELETE = \'DELETE\',

DANGEROUS = \'DANGEROUS\',

}

2.1.3 TaskStatus 枚举

export enum TaskStatus {

PENDING = \'pending\',

RUNNING = \'running\',

COMPLETED = \'completed\',

FAILED = \'failed\',

CANCELLED = \'cancelled\',

WAITING\_CONFIRM = \'waiting\_confirm\',

}

2.1.4 MessageType 枚举

export enum MessageType {

TASK\_ASSIGN = \'task\_assign\',

TASK\_RESULT = \'task\_result\',

TASK\_ERROR = \'task\_error\',

STATE\_UPDATE = \'state\_update\',

CONFIRM\_REQ = \'confirmation\_request\',

CONFIRM\_RESP = \'confirmation\_response\',

HEARTBEAT = \'heartbeat\',

}

2.2 核心接口定义

2.2.1 ToolDefinition - 工具定义

export interface ToolDefinition {

name: string;

description: string;

parameters: JSONSchema7;

agent: AgentType;

risk\_level: RiskLevel;

requires\_confirmation: boolean;

capabilities: string\[\];

api\_spec: {

method: \'GET\' \| \'POST\' \| \'PUT\' \| \'PATCH\' \| \'DELETE\';

path: string;

base\_url?: string;

headers?: Record\<string, string\>;

auth\_type: \'bearer\' \| \'api\_key\' \| \'basic\' \| \'none\';

timeout\_ms: number;

retry\_policy: RetryPolicy;

};

response\_schema?: JSONSchema7;

response\_trim\_paths?: string\[\];

examples?: ToolExample\[\];

}

2.2.2 ExecutionPlan - 执行计划

export interface ExecutionPlan {

plan\_id: string;

goal: string;

tasks: TaskNode\[\];

edges: DependencyEdge\[\];

context: ExecutionContext;

created\_at: string;

estimated\_duration\_ms?: number;

}

export interface TaskNode {

task\_id: string;

agent: AgentType;

tool\_name: string;

parameters: Record\<string, any\>;

dependencies: string\[\];

condition?: ConditionExpr;

retry\_policy: RetryPolicy;

timeout\_ms: number;

priority: number;

}

export interface DependencyEdge {

from\_task\_id: string;

to\_task\_id: string;

data\_mapping?: Record\<string, string\>;

}

export interface ConditionExpr {

field: string;

operator: \'eq\' \| \'neq\' \| \'gt\' \| \'lt\' \| \'in\' \| \'contains\';

value: any;

}

export interface RetryPolicy {

max\_retries: number;

backoff\_ms: number;

backoff\_multiplier: number;

retryable\_errors: string\[\];

}

export interface ExecutionContext {

session\_id: string;

user\_id: string;

role: string;

shared\_memory: Record\<string, any\>;

metadata: Record\<string, string\>;

}

2.2.3 AgentMessage - Agent间消息

export interface AgentMessage {

message\_id: string;

correlation\_id: string;

timestamp: string;

from: AgentType;

to: AgentType;

type: MessageType;

payload: Record\<string, any\>;

context\_ref?: string;

reply\_to?: string;

ttl\_ms?: number;

}

2.2.4 ToolExecutionResult - 工具执行结果

export interface ToolExecutionResult {

success: boolean;

tool\_name: string;

task\_id: string;

data?: any;

error?: {

code: string;

message: string;

http\_status?: number;

retryable: boolean;

details?: any;

};

duration\_ms: number;

trimmed: boolean;

metadata?: Record\<string, string\>;

}

三、Agent层详细设计

3.1 BaseAgent 抽象基类

所有Agent继承自BaseAgent，实现统一的生命周期管理和消息处理框架。

// src/agents/base-agent.ts

import { EventEmitter } from \'events\';

export abstract class BaseAgent extends EventEmitter {

protected agentType: AgentType;

protected tools: Map\<string, ToolDefinition\>;

protected llmClient: LLMClient;

protected messageBus: MessageBus;

protected logger: Logger;

constructor(config: AgentConfig) {

super();

this.agentType = config.type;

this.tools = new Map();

this.llmClient = createLLMClient(config.llm);

this.messageBus = config.messageBus;

this.logger = createLogger(\`agent:\${config.type}\`);

}

// 注册工具

registerTool(tool: ToolDefinition): void {

if (tool.agent !== this.agentType) {

throw new Error(

\`Tool \${tool.name} belongs to \${tool.agent}, \` +

\`not \${this.agentType}\`

);

}

this.tools.set(tool.name, tool);

}

// 获取Function Calling格式的工具描述

getToolDescriptions(): FunctionTool\[\] {

return Array.from(this.tools.values()).map(tool =\> ({

type: \'function\',

function: {

name: tool.name,

description: tool.description,

parameters: tool.parameters,

},

}));

}

// 消息处理入口

async handleMessage(msg: AgentMessage): Promise\<AgentMessage\> {

this.logger.info({ msgId: msg.message\_id }, \'Received message\');

try {

const result = await this.process(msg);

return this.buildResponse(msg, result);

} catch (err) {

this.logger.error({ err, msgId: msg.message\_id }, \'Processing failed\');

return this.buildErrorResponse(msg, err);

}

}

// 子类实现具体处理逻辑

protected abstract process(msg: AgentMessage): Promise\<any\>;

// 构建响应消息

protected buildResponse(

original: AgentMessage, result: any

): AgentMessage {

return {

message\_id: generateUUID(),

correlation\_id: original.correlation\_id,

timestamp: new Date().toISOString(),

from: this.agentType,

to: original.from,

type: MessageType.TASK\_RESULT,

payload: result,

reply\_to: original.message\_id,

};

}

// 构建错误响应

protected buildErrorResponse(

original: AgentMessage, err: Error

): AgentMessage {

return {

message\_id: generateUUID(),

correlation\_id: original.correlation\_id,

timestamp: new Date().toISOString(),

from: this.agentType,

to: original.from,

type: MessageType.TASK\_ERROR,

payload: {

code: err.name,

message: err.message,

retryable: isRetryableError(err),

},

reply\_to: original.message\_id,

};

}

// 启动Agent

async start(): Promise\<void\> {

await this.messageBus.subscribe(

this.agentType,

(msg) =\> this.handleMessage(msg)

);

this.logger.info(\`Agent \${this.agentType} started\`);

}

// 停止Agent

async stop(): Promise\<void\> {

await this.messageBus.unsubscribe(this.agentType);

this.logger.info(\`Agent \${this.agentType} stopped\`);

}

}

3.2 DataAgent 实现

DataAgent负责所有数据CRUD操作，管理分页、批量操作和数据缓存。

// src/agents/data-agent.ts

export class DataAgent extends BaseAgent {

private apiAdapter: ApiAdapter;

private cache: RedisClient;

private securityGuard: SecurityGuard;

constructor(config: DataAgentConfig) {

super(config);

this.apiAdapter = new ApiAdapter(config.adapter);

this.cache = createRedisClient(config.redis);

this.securityGuard = new SecurityGuard(config.security);

}

protected async process(msg: AgentMessage): Promise\<any\> {

const { tool\_name, parameters } = msg.payload;

const tool = this.tools.get(tool\_name);

if (!tool) {

throw new ToolNotFoundError(tool\_name);

}

// 1. 安全检查

await this.securityGuard.check(msg.context\_ref, tool, parameters);

// 2. 参数转换与校验

const transformed = ParamTransformer.transform(

parameters, tool.parameters

);

// 3. 检查缓存（仅GET请求）

if (tool.api\_spec.method === \'GET\') {

const cached = await this.cache.get(

this.buildCacheKey(tool\_name, transformed)

);

if (cached) return JSON.parse(cached);

}

// 4. 执行API调用

const result = await this.apiAdapter.execute(tool, transformed);

// 5. 缓存结果（仅GET请求）

if (tool.api\_spec.method === \'GET\' && result.success) {

await this.cache.set(

this.buildCacheKey(tool\_name, transformed),

JSON.stringify(result),

\'EX\', 300 // 5分钟TTL

);

}

// 6. 响应裁剪

return ResponseTrimmer.trim(result, tool.response\_trim\_paths);

}

private buildCacheKey(

toolName: string, params: any

): string {

return \`tool:\${toolName}:\${hashParams(params)}\`;

}

}

3.3 AgentRegistry - Agent注册中心

// src/agents/agent-registry.ts

export class AgentRegistry {

private agents: Map\<AgentType, BaseAgent\>;

private toolRegistry: ToolRegistry;

constructor(toolRegistry: ToolRegistry) {

this.agents = new Map();

this.toolRegistry = toolRegistry;

}

// 注册Agent实例

register(agent: BaseAgent): void {

this.agents.set(agent.agentType, agent);

}

// 按Agent类型分配工具

assignTools(): void {

for (const tool of this.toolRegistry.getAll()) {

const agent = this.agents.get(tool.agent);

if (agent) {

agent.registerTool(tool);

} else {

throw new Error(

\`No agent registered for type \${tool.agent}\`

);

}

}

}

// 获取指定Agent的工具描述

getToolDescriptions(agentType: AgentType): FunctionTool\[\] {

return this.agents.get(agentType)?.getToolDescriptions() ?? \[\];

}

// 启动所有Agent

async startAll(): Promise\<void\> {

this.assignTools();

for (const agent of this.agents.values()) {

await agent.start();

}

}

}

四、编排层详细设计

4.1 Orchestrator 核心逻辑

Orchestrator是系统入口，负责意图识别、任务分解、Agent调度和结果整合。

// src/orchestrator/orchestrator.ts

export class Orchestrator {

private planner: Planner;

private dagExecutor: DAGExecutor;

private stateManager: StateManager;

private messageBus: MessageBus;

private llmClient: LLMClient;

private logger: Logger;

async handleRequest(req: UserRequest): Promise\<UserResponse\> {

const sessionId = req.session\_id;

const traceId = generateTraceId();

// Step 1: 意图识别

const intent = await this.recognizeIntent(req.query, traceId);

// Step 2: 生成执行计划

const plan = await this.planner.createPlan(

intent,

{ session\_id: sessionId, user\_id: req.user\_id, role: req.role }

);

// Step 3: 持久化计划

await this.stateManager.savePlan(plan);

// Step 4: 执行DAG

const results = await this.dagExecutor.execute(

plan,

{ onTaskComplete: (taskId, result) =\>

this.stateManager.saveTaskResult(taskId, result)

}

);

// Step 5: 整合结果

const response = await this.integrateResults(

req.query, results, plan

);

// Step 6: 记录审计日志

await this.stateManager.saveAuditLog({

session\_id: sessionId,

trace\_id: traceId,

intent,

plan\_id: plan.plan\_id,

status: \'success\',

duration\_ms: results.total\_duration\_ms,

});

return response;

}

private async recognizeIntent(

query: string, traceId: string

): Promise\<Intent\> {

const response = await this.llmClient.chat({

messages: \[

{ role: \'system\', content: INTENT\_PROMPT },

{ role: \'user\', content: query },

\],

response\_format: { type: \'json\_object\' },

metadata: { trace\_id: traceId },

});

return IntentSchema.parse(JSON.parse(response.content));

}

private async integrateResults(

query: string,

results: DAGExecutionResult,

plan: ExecutionPlan

): Promise\<UserResponse\> {

const summary = await this.llmClient.chat({

messages: \[

{ role: \'system\', content: SUMMARY\_PROMPT },

{ role: \'user\', content: JSON.stringify({

query,

plan: plan.goal,

results: results.taskResults,

}) },

\],

});

return {

content: summary.content,

plan\_id: plan.plan\_id,

task\_count: plan.tasks.length,

success\_count: results.successCount,

total\_duration\_ms: results.total\_duration\_ms,

};

}

}

4.2 Planner - 任务规划器

Planner根据用户意图和可用工具，生成最优执行计划（DAG）。

// src/orchestrator/planner.ts

export class Planner {

private toolRegistry: ToolRegistry;

private knowledgeGraph: ToolKnowledgeGraph;

private llmClient: LLMClient;

async createPlan(

intent: Intent,

context: ExecutionContext

): Promise\<ExecutionPlan\> {

// 1. 通过知识图谱检索相关工具

const candidateTools = await this.knowledgeGraph.search(

intent.goal,

{ limit: 20 }

);

// 2. 过滤权限允许的工具

const allowedTools = candidateTools.filter(t =\>

this.checkPermission(context.role, t)

);

// 3. LLM生成执行计划

const planPrompt = this.buildPlanPrompt(

intent, allowedTools, context

);

const response = await this.llmClient.chat({

messages: \[

{ role: \'system\', content: PLANNER\_SYSTEM\_PROMPT },

{ role: \'user\', content: planPrompt },

\],

tools: this.buildPlanningTools(),

});

// 4. 解析并验证计划

const plan = this.parsePlan(response);

this.validatePlan(plan, allowedTools);

return plan;

}

private validatePlan(

plan: ExecutionPlan,

availableTools: ToolDefinition\[\]

): void {

const toolNames = new Set(availableTools.map(t =\> t.name));

for (const task of plan.tasks) {

if (!toolNames.has(task.tool\_name)) {

throw new PlanValidationError(

\`Task \${task.task\_id} references unknown tool: \${task.tool\_name}\`

);

}

// 检查循环依赖

if (this.hasCircularDependency(plan)) {

throw new PlanValidationError(\'Circular dependency detected\');

}

}

}

private hasCircularDependency(plan: ExecutionPlan): boolean {

const visited = new Set\<string\>();

const recursionStack = new Set\<string\>();

const dfs = (taskId: string): boolean =\> {

visited.add(taskId);

recursionStack.add(taskId);

const task = plan.tasks.find(t =\> t.task\_id === taskId);

if (!task) return false;

for (const dep of task.dependencies) {

if (!visited.has(dep)) {

if (dfs(dep)) return true;

} else if (recursionStack.has(dep)) {

return true;

}

}

recursionStack.delete(taskId);

return false;

};

return plan.tasks.some(t =\> dfs(t.task\_id));

}

}

4.3 DAGExecutor - DAG执行引擎

DAGExecutor负责按照DAG结构调度任务执行，支持并行执行无依赖的任务。

// src/orchestrator/dag-executor.ts

export class DAGExecutor {

private messageBus: MessageBus;

private stateManager: StateManager;

private logger: Logger;

async execute(

plan: ExecutionPlan,

callbacks?: { onTaskComplete?: (id: string, r: any) =\> void }

): Promise\<DAGExecutionResult\> {

const startTime = Date.now();

const taskResults = new Map\<string, ToolExecutionResult\>();

const taskStatus = new Map\<string, TaskStatus\>();

// 初始化所有任务状态

for (const task of plan.tasks) {

taskStatus.set(task.task\_id, TaskStatus.PENDING);

}

// 主执行循环

while (true) {

// 找出所有可执行的任务（依赖已满足）

const readyTasks = plan.tasks.filter(t =\>

taskStatus.get(t.task\_id) === TaskStatus.PENDING &&

t.dependencies.every(dep =\>

taskStatus.get(dep) === TaskStatus.COMPLETED

)

);

if (readyTasks.length === 0) {

// 检查是否全部完成

const allDone = plan.tasks.every(t =\>

\[TaskStatus.COMPLETED, TaskStatus.FAILED,

TaskStatus.CANCELLED\].includes(

taskStatus.get(t.task\_id)!

)

);

if (allDone) break;

// 检查是否有失败导致死锁

const hasFailed = plan.tasks.some(t =\>

taskStatus.get(t.task\_id) === TaskStatus.FAILED

);

if (hasFailed) break;

await sleep(100); // 等待进行中的任务

continue;

}

// 并行执行就绪任务

const promises = readyTasks.map(async (task) =\> {

taskStatus.set(task.task\_id, TaskStatus.RUNNING);

// 检查条件表达式

if (task.condition && !this.evaluateCondition(

task.condition, taskResults

)) {

taskStatus.set(task.task\_id, TaskStatus.CANCELLED);

return;

}

try {

// 解析参数中的引用（如 \$task\_1.result.id）

const resolvedParams = this.resolveParams(

task.parameters, taskResults

);

// 发送消息给目标Agent

const result = await this.messageBus.request({

from: AgentType.ORCHESTRATOR,

to: task.agent,

type: MessageType.TASK\_ASSIGN,

payload: {

tool\_name: task.tool\_name,

parameters: resolvedParams,

task\_id: task.task\_id,

},

correlation\_id: plan.plan\_id,

timeout\_ms: task.timeout\_ms,

});

taskResults.set(task.task\_id, result);

taskStatus.set(task.task\_id, TaskStatus.COMPLETED);

callbacks?.onTaskComplete?.(task.task\_id, result);

} catch (err) {

this.logger.error(

{ taskId: task.task\_id, err }, \'Task failed\'

);

taskResults.set(task.task\_id, {

success: false,

tool\_name: task.tool\_name,

task\_id: task.task\_id,

error: {

code: err.name,

message: err.message,

retryable: isRetryableError(err),

},

duration\_ms: 0,

trimmed: false,

});

// 重试逻辑

if (isRetryableError(err)) {

// \... 重试逻辑（略）

} else {

taskStatus.set(task.task\_id, TaskStatus.FAILED);

}

}

});

await Promise.allSettled(promises);

}

return {

plan\_id: plan.plan\_id,

taskResults,

successCount: Array.from(taskResults.values())

.filter(r =\> r.success).length,

failCount: Array.from(taskResults.values())

.filter(r =\> !r.success).length,

total\_duration\_ms: Date.now() - startTime,

};

}

// 解析参数引用: \$task\_1.result.id -\> 实际值

private resolveParams(

params: Record\<string, any\>,

results: Map\<string, ToolExecutionResult\>

): Record\<string, any\> {

const resolved = JSON.parse(JSON.stringify(params));

const resolveValue = (val: any): any =\> {

if (typeof val === \'string\' && val.startsWith(\'\$\')) {

const \[taskId, \...paths\] = val.slice(1).split(\'.\');

const result = results.get(taskId);

if (!result?.data) throw new ReferenceError(

\`Cannot resolve \${val}: task \${taskId} has no result\`

);

return paths.reduce((obj, key) =\> obj?.\[key\], result.data);

}

if (Array.isArray(val)) return val.map(resolveValue);

if (typeof val === \'object\' && val !== null) {

return Object.fromEntries(

Object.entries(val).map((\[k, v\]) =\> \[k, resolveValue(v)\])

);

}

return val;

};

return resolveValue(resolved);

}

}

五、适配层详细设计

5.1 OpenAPI 加载器

从OpenAPI JSON文档自动提取并转换为工具定义。

// src/adapter/openapi-loader.ts

export class OpenAPILoader {

/\*\*

\* 将OpenAPI文档转换为工具定义列表

\* 支持路径参数、查询参数、请求体、响应Schema

\*/

async load(specPath: string): Promise\<ToolDefinition\[\]\> {

const spec = await this.parseSpec(specPath);

const tools: ToolDefinition\[\] = \[\];

for (const \[path, methods\] of Object.entries(spec.paths)) {

for (const \[method, operation\] of Object.entries(methods)) {

const tool = this.convertOperation(

path, method, operation, spec

);

tools.push(tool);

}

}

return tools;

}

private convertOperation(

path: string,

method: string,

op: OpenAPIOperation,

spec: OpenAPISpec

): ToolDefinition {

// 1. 生成工具名称

const name = op.operationId

\|\| this.generateName(path, method);

// 2. 合并参数

const params = this.mergeParameters(

op.parameters \|\| \[\],

spec.paths\[path\].parameters \|\| \[\]

);

// 3. 转换为JSON Schema

const properties: Record\<string, any\> = {};

const required: string\[\] = \[\];

for (const p of params) {

properties\[p.name\] = this.convertToJSONSchema(p);

if (p.required) required.push(p.name);

}

// 4. 处理请求体

if (op.requestBody) {

const schema = this.extractSchema(op.requestBody);

Object.assign(properties, schema.properties);

required.push(\...(schema.required \|\| \[\]));

}

// 5. 推断风险等级

const riskLevel = this.inferRiskLevel(method);

// 6. 提取响应裁剪路径

const responseTrimPaths = this.extractTrimPaths(

op.responses

);

return {

name: this.toSnakeCase(name),

description: this.buildDescription(op),

parameters: {

type: \'object\',

properties,

required: \[\...new Set(required)\],

},

agent: this.mapToAgent(path, method),

risk\_level: riskLevel,

requires\_confirmation: riskLevel === RiskLevel.DELETE

\|\| riskLevel === RiskLevel.DANGEROUS,

capabilities: this.extractCapabilities(op),

api\_spec: {

method: method.toUpperCase() as any,

path: this.normalizePath(path),

auth\_type: this.extractAuthType(op, spec),

timeout\_ms: 30000,

retry\_policy: {

max\_retries: method === \'get\' ? 2 : 0,

backoff\_ms: 1000,

backoff\_multiplier: 2,

retryable\_errors: \[\'502\', \'503\', \'429\'\],

},

},

response\_trim\_paths: responseTrimPaths,

};

}

private inferRiskLevel(method: string): RiskLevel {

const map: Record\<string, RiskLevel\> = {

get: RiskLevel.READ,

post: RiskLevel.WRITE,

put: RiskLevel.WRITE,

patch: RiskLevel.WRITE,

delete: RiskLevel.DELETE,

};

return map\[method\] \|\| RiskLevel.READ;

}

private mapToAgent(path: string, method: string): AgentType {

// 根据路径前缀和HTTP方法映射到Agent

if (path.includes(\'/auth\') \|\| path.includes(\'/login\'))

return AgentType.AUTH;

if (path.includes(\'/notify\') \|\| path.includes(\'/email\') \|\| path.includes(\'/sms\'))

return AgentType.NOTIFY;

return AgentType.DATA;

}

}

5.2 ApiAdapter - API适配器

// src/adapter/api-adapter.ts

export class ApiAdapter {

private authManager: AuthManager;

private logger: Logger;

private metrics: MetricsClient;

async execute(

tool: ToolDefinition,

params: Record\<string, any\>

): Promise\<ToolExecutionResult\> {

const startTime = Date.now();

const { api\_spec } = tool;

try {

// 1. 注入认证信息

const headers = { \...api\_spec.headers };

if (api\_spec.auth\_type !== \'none\') {

const auth = await this.authManager.getAuth(

api\_spec.auth\_type

);

Object.assign(headers, auth.headers);

}

// 2. 构建请求URL（替换路径参数）

let url = api\_spec.base\_url + api\_spec.path;

for (const \[key, value\] of Object.entries(params)) {

url = url.replace(\`{\${key}}\`, encodeURIComponent(String(value)));

}

// 3. 分离路径参数和查询参数

const pathParams = new Set(

(api\_spec.path.match(/\\{(\[\^}\]+)\\}/g) \|\| \[\])

.map(m =\> m.slice(1, -1))

);

const queryParams = Object.fromEntries(

Object.entries(params).filter((\[k\]) =\> !pathParams.has(k))

);

// 4. 构建请求选项

const fetchOptions: RequestInit = {

method: api\_spec.method,

headers: { \'Content-Type\': \'application/json\', \...headers },

};

if (\[\'POST\', \'PUT\', \'PATCH\'\].includes(api\_spec.method)) {

fetchOptions.body = JSON.stringify(queryParams);

} else if (Object.keys(queryParams).length \> 0) {

const searchParams = new URLSearchParams(

queryParams as any

);

url += \'?\' + searchParams.toString();

}

// 5. 执行请求（带超时）

const controller = new AbortController();

const timeout = setTimeout(

() =\> controller.abort(),

api\_spec.timeout\_ms

);

const response = await fetch(url, {

\...fetchOptions,

signal: controller.signal,

});

clearTimeout(timeout);

// 6. 处理响应

const duration\_ms = Date.now() - startTime;

this.metrics.recordToolCall(tool.name, duration\_ms, response.status);

if (!response.ok) {

return {

success: false,

tool\_name: tool.name,

task\_id: \'\',

error: ErrorHandler.classify(

response.status, await response.text()

),

duration\_ms,

trimmed: false,

};

}

const data = await response.json();

return {

success: true,

tool\_name: tool.name,

task\_id: \'\',

data,

duration\_ms,

trimmed: false,

};

} catch (err) {

const duration\_ms = Date.now() - startTime;

this.metrics.recordToolCall(tool.name, duration\_ms, 0);

if (err.name === \'AbortError\') {

return {

success: false,

tool\_name: tool.name,

task\_id: \'\',

error: {

code: \'TIMEOUT\',

message: \`Request timed out after \${api\_spec.timeout\_ms}ms\`,

retryable: true,

},

duration\_ms,

trimmed: false,

};

}

throw err;

}

}

}

5.3 ErrorHandler - 错误处理器

// src/adapter/error-handler.ts

export class ErrorHandler {

static classify(

httpStatus: number, responseBody: string

): ToolExecutionResult\[\'error\'\] {

const errorMap: Record\<number, { code: string; retryable: boolean }\> = {

400: { code: \'BAD\_REQUEST\', retryable: false },

401: { code: \'UNAUTHORIZED\', retryable: false },

403: { code: \'FORBIDDEN\', retryable: false },

404: { code: \'NOT\_FOUND\', retryable: false },

429: { code: \'RATE\_LIMITED\', retryable: true },

500: { code: \'SERVER\_ERROR\', retryable: true },

502: { code: \'BAD\_GATEWAY\', retryable: true },

503: { code: \'SERVICE\_UNAVAILABLE\', retryable: true },

};

const mapped = errorMap\[httpStatus\];

let message = responseBody;

try {

const parsed = JSON.parse(responseBody);

message = parsed.message \|\| parsed.error \|\| responseBody;

} catch { /\* use raw response \*/ }

return {

code: mapped?.code \|\| \`HTTP\_\${httpStatus}\`,

message,

http\_status: httpStatus,

retryable: mapped?.retryable \|\| false,

};

}

/\*\*

\* 将错误翻译为自然语言，供LLM理解

\*/

static toNaturalLanguage(error: ToolExecutionResult\[\'error\'\]): string {

const translations: Record\<string, string\> = {

BAD\_REQUEST: \'请求参数有误，请检查参数格式和取值范围\',

UNAUTHORIZED: \'认证失败，Token可能已过期，请重新登录\',

FORBIDDEN: \'权限不足，当前角色无权执行此操作\',

NOT\_FOUND: \'请求的资源不存在，请确认ID是否正确\',

RATE\_LIMITED: \'请求过于频繁，请稍后重试\',

SERVER\_ERROR: \'服务端异常，请稍后重试\',

BAD\_GATEWAY: \'网关异常，请稍后重试\',

SERVICE\_UNAVAILABLE: \'服务暂时不可用，请稍后重试\',

TIMEOUT: \'请求超时，请检查网络或稍后重试\',

};

return translations\[error?.code \|\| \'\'\]

\|\| \`未知错误: \${error?.message \|\| \'\'}\`;

}

}

5.4 ResponseTrimmer - 响应裁剪器

// src/adapter/response-trimmer.ts

export class ResponseTrimmer {

/\*\*

\* 按指定路径裁剪响应，只保留关键字段

\* 减少返回给LLM的token数量

\*/

static trim(

result: ToolExecutionResult,

trimPaths?: string\[\]

): ToolExecutionResult {

if (!trimPaths \|\| trimPaths.length === 0 \|\| !result.data) {

return result;

}

const trimmed = {};

for (const path of trimPaths) {

const value = this.getByPath(result.data, path);

if (value !== undefined) {

this.setByPath(trimmed, path, value);

}

}

return { \...result, data: trimmed, trimmed: true };

}

/\*\*

\* 智能裁剪：自动移除元数据字段，保留业务数据

\*/

static smartTrim(data: any, maxDepth: number = 3): any {

const META\_KEYS = new Set(\[

\'metadata\', \'meta\', \'pagination\', \'links\',

\'\_links\', \'\_meta\', \'total\_pages\', \'total\_count\',

\]);

if (Array.isArray(data)) {

return data.slice(0, 50).map(item =\>

this.smartTrim(item, maxDepth - 1)

);

}

if (typeof data === \'object\' && data !== null) {

if (maxDepth \<= 0) return \'\[truncated\]\';

const result: any = {};

for (const \[key, value\] of Object.entries(data)) {

if (META\_KEYS.has(key)) continue;

result\[key\] = this.smartTrim(value, maxDepth - 1);

}

return result;

}

return data;

}

private static getByPath(obj: any, path: string): any {

return path.split(\'.\').reduce((o, k) =\> o?.\[k\], obj);

}

private static setByPath(obj: any, path: string, value: any): void {

const keys = path.split(\'.\');

const last = keys.pop()!;

const target = keys.reduce((o, k) =\> {

o\[k\] = o\[k\] \|\| {};

return o\[k\];

}, obj);

target\[last\] = value;

}

}

六、消息通信详细设计

6.1 MessageBus 消息总线

// src/messaging/message-bus.ts

export class MessageBus {

private kafka: Kafka;

private producer: Producer;

private consumers: Map\<AgentType, Consumer\>;

private pendingRequests: Map\<string, {

resolve: (msg: AgentMessage) =\> void;

reject: (err: Error) =\> void;

timer: NodeJS.Timeout;

}\>;

constructor(config: KafkaConfig) {

this.kafka = new Kafka(config);

this.producer = this.kafka.producer();

this.consumers = new Map();

this.pendingRequests = new Map();

}

/\*\*

\* 发送消息（单向）

\*/

async send(message: AgentMessage): Promise\<void\> {

const topic = \`agent.\${message.to}\`;

await this.producer.send({

topic,

messages: \[{

key: message.message\_id,

value: JSON.stringify(message),

}\],

});

}

/\*\*

\* 请求-响应模式（同步等待）

\*/

async request(

message: Partial\<AgentMessage\>,

options?: { timeout\_ms?: number }

): Promise\<AgentMessage\> {

const fullMessage: AgentMessage = {

message\_id: generateUUID(),

correlation\_id: generateUUID(),

timestamp: new Date().toISOString(),

from: message.from!,

to: message.to!,

type: message.type!,

payload: message.payload \|\| {},

reply\_to: \`reply.\${message.from}.\${generateUUID()}\`,

ttl\_ms: options?.timeout\_ms \|\| 30000,

};

return new Promise((resolve, reject) =\> {

const timer = setTimeout(() =\> {

this.pendingRequests.delete(fullMessage.correlation\_id);

reject(new Error(

\`Agent \${fullMessage.to} did not respond within \${options?.timeout\_ms \|\| 30000}ms\`

));

}, fullMessage.ttl\_ms);

this.pendingRequests.set(fullMessage.correlation\_id, {

resolve, reject, timer,

});

this.send(fullMessage).catch(reject);

});

}

/\*\*

\* 订阅消息

\*/

async subscribe(

agentType: AgentType,

handler: (msg: AgentMessage) =\> Promise\<AgentMessage \| void\>

): Promise\<void\> {

const topic = \`agent.\${agentType}\`;

const consumer = this.kafka.consumer({

groupId: \`agent-group-\${agentType}\`,

});

await consumer.connect();

await consumer.subscribe({ topic, fromBeginning: false });

await consumer.run({

eachMessage: async ({ message }) =\> {

const msg = JSON.parse(message.value!.toString());

// 处理回复消息（request-response模式）

if (msg.reply\_to && this.pendingRequests.has(msg.correlation\_id)) {

const pending = this.pendingRequests.get(msg.correlation\_id)!;

clearTimeout(pending.timer);

this.pendingRequests.delete(msg.correlation\_id);

pending.resolve(msg);

return;

}

// 正常处理

const response = await handler(msg);

if (response) await this.send(response);

},

});

this.consumers.set(agentType, consumer);

}

}

6.2 Kafka Topic 规划

  --------------------- ------------ ------------ ---------------
  **Topic**             **分区数**   **副本数**   **说明**
  agent.orchestrator    3            3            编排器消息
  agent.data\_agent     6            3            数据Agent消息
  agent.auth\_agent     3            3            认证Agent消息
  agent.notify\_agent   3            3            通知Agent消息
  agent.audit           3            3            审计日志Topic
  --------------------- ------------ ------------ ---------------

七、安全模块详细设计

7.1 RBAC 权限控制

// src/security/rbac.ts

export class SecurityGuard {

private rolePermissions: Map\<string, Set\<string\>\>;

async check(

contextRef: string,

tool: ToolDefinition,

params: Record\<string, any\>

): Promise\<SecurityCheckResult\> {

const context = await this.loadContext(contextRef);

const role = context.role;

// 1. 工具级权限检查

const allowed = this.rolePermissions.get(role);

if (!allowed?.has(tool.name)) {

return { allowed: false, reason: \`Role \${role} cannot access tool \${tool.name}\` };

}

// 2. 参数级校验

const paramResult = this.validateParams(tool, params);

if (!paramResult.valid) {

return { allowed: false, reason: paramResult.reason };

}

// 3. 敏感操作确认检查

if (tool.requires\_confirmation) {

const confirmed = await this.checkConfirmation(

contextRef, tool.name

);

if (!confirmed) {

return {

allowed: false,

reason: \'Requires user confirmation\',

requires\_confirmation: true,

confirmation\_message: this.buildConfirmMessage(tool, params),

};

}

}

return { allowed: true };

}

private validateParams(

tool: ToolDefinition,

params: Record\<string, any\>

): { valid: boolean; reason?: string } {

// 数值范围检查

for (const \[key, schema\] of Object.entries(tool.parameters.properties \|\| {})) {

if (params\[key\] !== undefined) {

if (schema.minimum !== undefined && params\[key\] \< schema.minimum)

return { valid: false, reason: \`\${key} must be \>= \${schema.minimum}\` };

if (schema.maximum !== undefined && params\[key\] \> schema.maximum)

return { valid: false, reason: \`\${key} must be \<= \${schema.maximum}\` };

if (schema.maxLength !== undefined && String(params\[key\]).length \> schema.maxLength)

return { valid: false, reason: \`\${key} exceeds max length \${schema.maxLength}\` };

}

}

return { valid: true };

}

}

7.2 审计日志

// src/security/audit-logger.ts

export class AuditLogger {

private kafka: Producer;

async log(entry: AuditEntry): Promise\<void\> {

await this.kafka.send({

topic: \'agent.audit\',

messages: \[{

key: entry.trace\_id,

value: JSON.stringify({

\...entry,

timestamp: new Date().toISOString(),

}),

}\],

});

}

}

export interface AuditEntry {

trace\_id: string;

session\_id: string;

user\_id: string;

action: string; // \'tool\_call\' \| \'plan\_create\' \| \'auth\_check\'

tool\_name?: string;

parameters?: Record\<string, any\>; // 敏感参数脱敏

result: \'success\' \| \'failure\' \| \'denied\';

risk\_level: RiskLevel;

duration\_ms?: number;

error\_code?: string;

ip\_address?: string;

}

八、知识增强模块

8.1 ToolKnowledgeGraph - 工具知识图谱

// src/knowledge/tool-graph.ts

export class ToolKnowledgeGraph {

private milvus: MilvusClient;

private embedding: EmbeddingService;

/\*\*

\* 构建工具知识图谱索引

\* 将工具描述向量化并存入Milvus

\*/

async buildIndex(tools: ToolDefinition\[\]): Promise\<void\> {

const vectors = await Promise.all(

tools.map(async tool =\> {

const text = this.buildIndexText(tool);

const embedding = await this.embedding.embed(text);

return {

id: tool.name,

vector: embedding,

metadata: {

name: tool.name,

agent: tool.agent,

risk\_level: tool.risk\_level,

capabilities: tool.capabilities,

description: tool.description,

},

};

})

);

await this.milvus.upsert({

collection: \'tool\_knowledge\',

data: vectors,

});

}

/\*\*

\* 语义搜索：根据用户意图查找最相关的工具

\*/

async search(

query: string,

options?: { limit?: number; agent?: AgentType }

): Promise\<ToolDefinition\[\]\> {

const queryVector = await this.embedding.embed(query);

const filter = options?.agent

? \`agent == \'\${options.agent}\'\`

: undefined;

const results = await this.milvus.search({

collection: \'tool\_knowledge\',

vector: queryVector,

top\_k: options?.limit \|\| 10,

filter,

});

return results.map(r =\> r.metadata as ToolDefinition);

}

private buildIndexText(tool: ToolDefinition): string {

return \[

tool.name,

tool.description,

\...tool.capabilities,

\`HTTP \${tool.api\_spec.method} \${tool.api\_spec.path}\`,

\].join(\' \');

}

}

九、配置与部署

9.1 配置文件结构

// configs/app.config.ts

export const appConfig = {

server: { port: 3000, host: \'0.0.0.0\' },

llm: {

provider: \'openai\', // \'openai\' \| \'anthropic\' \| \'gemini\'

model: \'gpt-4o\',

maxTokens: 4096,

temperature: 0.1,

},

kafka: {

brokers: \[\'localhost:9092\'\],

clientId: \'multi-agent-api\',

},

redis: {

host: \'localhost\',

port: 6379,

password: process.env.REDIS\_PASSWORD,

db: 0,

},

milvus: {

address: \'localhost:19530\',

collection: \'tool\_knowledge\',

dimension: 1536,

},

agents: {

data: { instances: 3, concurrency: 10 },

auth: { instances: 2, concurrency: 5 },

notify: { instances: 2, concurrency: 5 },

},

security: {

jwtSecret: process.env.JWT\_SECRET,

tokenExpiryMs: 3600000,

maxBatchSize: 100,

maxDeletePerRequest: 10,

},

observability: {

metricsPort: 9090,

logLevel: \'info\',

traceSampleRate: 0.1,

},

} as const;

9.2 Docker Compose 部署

\# docker-compose.yml

version: \'3.8\'

services:

api-gateway:

build: .

ports: \[\'3000:3000\'\]

depends\_on: \[kafka, redis, milvus\]

environment:

\- NODE\_ENV=production

\- KAFKA\_BROKERS=kafka:9092

\- REDIS\_HOST=redis

deploy:

replicas: 2

data-agent:

build: .

command: node dist/agents/data-agent-worker.js

depends\_on: \[kafka, redis\]

environment:

\- AGENT\_TYPE=data\_agent

\- KAFKA\_BROKERS=kafka:9092

deploy:

replicas: 3

auth-agent:

build: .

command: node dist/agents/auth-agent-worker.js

depends\_on: \[kafka\]

environment:

\- AGENT\_TYPE=auth\_agent

deploy:

replicas: 2

notify-agent:

build: .

command: node dist/agents/notify-agent-worker.js

depends\_on: \[kafka\]

environment:

\- AGENT\_TYPE=notify\_agent

deploy:

replicas: 2

kafka:

image: confluentinc/cp-kafka:7.5.0

ports: \[\'9092:9092\'\]

redis:

image: redis:7-alpine

ports: \[\'6379:6379\'\]

milvus:

image: milvusdb/milvus:v2.3.0

ports: \[\'19530:19530\'\]

prometheus:

image: prom/prometheus

ports: \[\'9090:9090\'\]

volumes:

\- ./configs/prometheus.yml:/etc/prometheus/prometheus.yml

十、测试策略

10.1 测试分层

  -------------- ------------------- ------------------------------
  **测试层级**   **覆盖范围**        **工具/方法**
  单元测试       各模块独立逻辑      Vitest + 类型校验
  集成测试       模块间交互          Testcontainers (Kafka/Redis)
  E2E测试        完整请求链路        Supertest + Mock LLM
  工具描述测试   LLM工具选择准确性   预设意图-工具映射测试集
  -------------- ------------------- ------------------------------

10.2 关键测试用例

// tests/orchestrator/planner.test.ts

describe(\'Planner\', () =\> {

it(\'should generate correct plan for simple query\', async () =\> {

const planner = new Planner(mockToolRegistry, mockGraph);

const intent = { goal: \'query customer orders\', entities: { customer\_id: \'CUS-001\' } };

const plan = await planner.createPlan(intent, mockContext);

expect(plan.tasks).toHaveLength(2);

expect(plan.tasks\[0\].tool\_name).toBe(\'get\_customer\_info\');

expect(plan.tasks\[1\].tool\_name).toBe(\'list\_orders\');

expect(plan.tasks\[1\].dependencies).toContain(plan.tasks\[0\].task\_id);

});

it(\'should detect circular dependencies\', () =\> {

const plan: ExecutionPlan = {

plan\_id: \'test\',

goal: \'test\',

tasks: \[

{ task\_id: \'1\', dependencies: \[\'2\'\], \... },

{ task\_id: \'2\', dependencies: \[\'1\'\], \... },

\],

edges: \[\],

context: mockContext,

created\_at: new Date().toISOString(),

};

expect(() =\> planner.validatePlan(plan, \[\])).toThrow(\'Circular dependency\');

});

});

// tests/adapter/error-handler.test.ts

describe(\'ErrorHandler\', () =\> {

it(\'should classify 429 as retryable\', () =\> {

const error = ErrorHandler.classify(429, \'{\"message\": \"Too many requests\"}\');

expect(error.retryable).toBe(true);

expect(error.code).toBe(\'RATE\_LIMITED\');

});

it(\'should translate 403 to natural language\', () =\> {

const error = ErrorHandler.classify(403, \'Forbidden\');

const message = ErrorHandler.toNaturalLanguage(error);

expect(message).toContain(\'权限不足\');

});

});

10.3 Mock LLM 测试策略

在测试环境中，使用Mock LLM替代真实LLM调用，确保测试的确定性和速度。

// tests/helpers/mock-llm.ts

export class MockLLMClient implements LLMClient {

private responses: Map\<string, any\>;

constructor() {

this.responses = new Map();

}

// 注册预设响应

mockResponse(key: string, response: any): void {

this.responses.set(key, response);

}

async chat(request: ChatRequest): Promise\<ChatResponse\> {

const userMsg = request.messages.find(m =\> m.role === \'user\');

const key = userMsg?.content \|\| \'\';

// 匹配预设响应

for (const \[pattern, response\] of this.responses) {

if (key.includes(pattern)) {

return typeof response === \'string\'

? { content: response, usage: { prompt\_tokens: 10, completion\_tokens: 20 } }

: response;

}

}

// 默认响应

return { content: \'{}\', usage: { prompt\_tokens: 0, completion\_tokens: 0 } };

}

}

// 使用示例

const mockLLM = new MockLLMClient();

mockLLM.mockResponse(\'query customer\', {

content: JSON.stringify({

tool\_name: \'get\_customer\_info\',

parameters: { customer\_id: \'CUS-001\' },

}),

});

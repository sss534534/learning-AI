# LLM + CLI 应用开发知识库

> 方法论 + 六大核心领域 + 最佳实践 + 反模式 + 检查清单

---

## 核心认知

**LLM + CLI 是一种特殊的人机交互范式。** 它不同于 Web/App 的 GUI 交互，也不同于纯 API 调用。核心矛盾在于：终端是线性文本流，但 LLM 的输出天生非线性（工具调用、代码块、流式思考交织在一起）。

经典案例：Claude Code、CodeBuddy Code、aider、Warp、Ghostty AI、Shell-GPT。它们共同验证了一个原则——**在终端里，非阻塞体验 > 信息完整性。**

---

## 第一部分：六大核心领域

### 1. 交互模式设计

CLI 下 LLM 交互有四种模式，选择取决于**用户是否守在终端前**。

| 模式 | 交互方式 | 人在终端前？ | 典型场景 |
|------|----------|-------------|----------|
| **Repl** | 持续对话，状态保持 | 是 | 编码助手、数据分析 |
| **Oneshot** | 一句话任务，用完即走 | 是 | `ai "fix this bug"` |
| **Pipe** | stdin/stdout 管道流 | 可能不在 | `cat log.txt \| ai "分析错误"` |
| **Daemon** | 后台常驻，watch 触发 | 不在 | CI 自动修 bug、监控告警处理 |

#### 模式路由决策树

```
用户是否要持续对话？
  ├─ 是 → Repl 模式
  │   ├─ 需要文件系统访问？ → 加载工作目录上下文
  │   └─ 纯问答？ → 轻量 Repl，不加载项目
  └─ 否 → 人在终端前吗？
      ├─ 是 → Oneshot 模式
      └─ 否 → Pipe 或 Daemon 模式
          ├─ 有输入流？ → Pipe
          └─ 定时触发？ → Daemon
```

#### Repl 模式的最小骨架

```python
class ReplSession:
    def __init__(self, workdir: str, model: str):
        self.workdir = workdir
        self.model = model
        self.history: list[dict] = []
        self.agent = AgentLoop(model=model)

    async def start(self):
        """交互主循环"""
        print(f"\033[36m会话已启动 | 工作目录: {self.workdir}\033[0m")

        while True:
            try:
                user_input = await self._read_input()
                if user_input in ("exit", "quit", "/q"):
                    break
                if not user_input.strip():
                    continue

                await self._process(user_input)
            except KeyboardInterrupt:
                print("\n\033[33m收到中断，正在停止...\033[0m")
                self.agent.cancel()
            except EOFError:
                break

    async def _process(self, user_input: str):
        # 流式打印 Agent 的思考过程
        async for chunk in self.agent.stream(user_input):
            if chunk.type == "thought":
                print(f"\033[90m  {chunk.content}\033[0m")
            elif chunk.type == "tool_call":
                print(f"\033[34m  → {chunk.tool_name}({chunk.args})\033[0m")
            elif chunk.type == "output":
                print(chunk.content, end="", flush=True)
            elif chunk.type == "tool_result":
                lines = str(chunk.result)[:200]
                print(f"\033[90m    ← {lines}\033[0m")
```

**Oneshot 模式**：

```python
async def oneshot(prompt: str, workdir: str = "."):
    result = await AgentLoop(model="gpt-4o").run(prompt)
    # 只输出最终结果，不输出中间过程
    print(result.content)
```

**Pipe 模式**：

```python
async def pipe_mode():
    stdin_data = sys.stdin.read()
    if not stdin_data.strip():
        print("Usage: cat data.txt | ai '分析这些数据'", file=sys.stderr)
        sys.exit(1)
    # prompt 来自 argv，input 来自 stdin
    prompt = " ".join(sys.argv[1:])
    result = await AgentLoop().run(f"{prompt}\n\n输入数据:\n{stdin_data}")
    print(result.content)
```

---

### 2. 终端 UX 设计

这是 CLI 应用区别于 API 调用的核心。在终端里，**用户看见的第一帧决定了体验的 80%。**

#### 流式输出的分层渲染

```
Layer 1: 思考过程 (Thought)
  └─ 灰色、缩进、紧跟 spinner
     └─ 示例: "⋯ 分析告警类型..."

Layer 2: 工具调用 (Tool Call)
  └─ 蓝色箭头、紧凑单行
     └─ 示例: "→ search_alerts(severity="P0")"

Layer 3: 工具结果 (Observation)
  └─ 灰色、截断到 200 字符、可展开
     └─ 示例: "← 找到 3 条 P0 告警 [展开...]"

Layer 4: 最终回答 (Answer)
  └─ 正常颜色、完整渲染、支持 Markdown
```

#### 色彩语义规范

```python
class TerminalColors:
    """CLI 色彩语义——永远用语义而非固定颜色"""
    INFO = "\033[36m"       # 青色：信息、状态
    SUCCESS = "\033[32m"    # 绿色：成功、完成
    WARNING = "\033[33m"    # 黄色：警告、需要注意
    ERROR = "\033[31m"      # 红色：错误、失败
    DIM = "\033[90m"        # 灰色：次要信息、日志
    BOLD = "\033[1m"        # 加粗：强调
    RESET = "\033[0m"

    @staticmethod
    def spinner(frames: list[str] = None):
        """旋转指示器——用于等待 LLM 首 token"""
        return Spinner(frames or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
```

#### 进度指示策略

```python
class ProgressIndicator:
    """不同阶段的进度指示"""

    @staticmethod
    def waiting_for_llm():
        """等待 LLM 响应（首 token 前）"""
        return Spinner("dots")  # 旋转点

    @staticmethod
    def tool_executing(tool_name: str):
        """工具执行中"""
        return f"\033[34m  → {tool_name} \033[90m执行中...\033[0m"

    @staticmethod
    def step_counter(current: int, total: int):
        """步骤计数"""
        return f"\033[90m[{current}/{total}]\033[0m"

    @staticmethod
    def thinking():
        """模型在推理（非工具调用）"""
        return f"\033[90m  ⋯\033[0m"
```

#### 折叠区域（工具调用展开/折叠）

终端空间有限，工具调用的详细输入输出默认折叠：

```python
class CollapsibleSection:
    """可折叠区域——节省终端空间"""
    def __init__(self, title: str):
        self.title = title
        self.expanded = False

    def render_summary(self):
        return f"\033[90m  ── {self.title} [按 Enter 展开] ──\033[0m"

    def render_detail(self, content: str):
        return f"\033[90m  ┌ {self.title}:\n" \
               f"  │ {content[:500]}\n" \
               f"  └──\033[0m"
```

#### 核心 UX 原则

```
1. 首 token 延迟 (TTFT) < 500ms：用户看到响应的速度决定"够快"还是"卡"
2. 流式逐字输出：一个字符一个字符地渲染，不等完整回复
3. 工具调用非阻塞展示：调用工具时立即显示"正在调用 xxx..."，不等返回
4. 可中断：Ctrl+C 必须立即停止 LLM 调用和工具执行
5. 差异化更新：不要清屏重绘整个界面，只更新变化的部分
```

---

### 3. 工具-沙箱模型

这是 LLM + CLI 最特殊的地方——Agent 的能力本质上是**操作系统调用**，而不只是 API 调用。这意味着安全风险等比放大。

#### 三级权限模型

```python
from enum import Enum

class ToolPermission(Enum):
    READ = "read"        # 只读：搜索、查询、读取文件内容
    WRITE = "write"      # 写入：创建/修改文件、安装依赖
    DANGEROUS = "dangerous"  # 危险：删除、执行命令、网络请求、发消息

class ToolRegistry:
    """工具注册表——每个工具必须声明权限级别"""

    tools = {
        "read_file":      ToolPermission.READ,
        "search_code":    ToolPermission.READ,
        "list_directory": ToolPermission.READ,
        "write_file":     ToolPermission.WRITE,
        "run_command":    ToolPermission.DANGEROUS,
        "delete_file":    ToolPermission.DANGEROUS,
        "send_message":   ToolPermission.DANGEROUS,
    }

    @classmethod
    def needs_confirmation(cls, tool_name: str) -> bool:
        return cls.tools.get(tool_name) == ToolPermission.DANGEROUS
```

#### 沙箱实现策略

```python
class Sandbox:
    """执行环境沙箱——逐层防护"""

    def __init__(self, allowed_paths: list[str], timeout: int = 30):
        self.allowed_paths = set(allowed_paths)
        self.timeout = timeout

    def validate_file_access(self, path: str, mode: str) -> bool:
        """验证文件访问权限"""
        resolved = Path(path).resolve()
        # 检查是否在白名单路径内
        if not any(resolved.is_relative_to(allowed) for allowed in self.allowed_paths):
            raise SandboxError(f"Access denied: {path}")
        # 禁止访问敏感路径
        forbidden = ["/etc/passwd", "~/.ssh", "~/.aws", "AppData/Roaming"]
        if any(f in str(resolved) for f in forbidden):
            raise SandboxError(f"Sensitive path blocked: {path}")
        return True

    def validate_command(self, cmd: list[str]) -> bool:
        """验证命令安全性"""
        # 禁止交互式命令
        interactive_commands = ["vim", "nano", "less", "more", "ssh", "sudo"]
        if cmd[0] in interactive_commands:
            raise SandboxError(f"Interactive command blocked: {cmd[0]}")
        # 禁止网络下载命令（除非显式允许）
        network_commands = ["curl", "wget", "nc", "telnet"]
        if cmd[0] in network_commands and not self.allow_network:
            raise SandboxError(f"Network command requires confirmation: {cmd[0]}")
        return True

    async def execute(self, cmd: str, cwd: str) -> CommandResult:
        """在沙箱中执行命令"""
        cmd_parts = shlex.split(cmd)
        self.validate_command(cmd_parts)

        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # 禁止交互
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise SandboxError(f"Command timed out after {self.timeout}s: {cmd}")

        return CommandResult(
            returncode=process.returncode,
            stdout=stdout.decode()[:10000],  # 截断过长的输出
            stderr=stderr.decode()[:5000],
        )
```

#### 确认流程

所有 DANGEROUS 操作必须经过用户确认：

```python
class ConfirmationFlow:
    """危险操作的确认流程"""

    @staticmethod
    async def confirm(
        tool_name: str,
        params: dict,
        reasoning: str
    ) -> bool:
        print(f"\n\033[33m⚠️  确认执行危险操作:\033[0m")
        print(f"   工具: {tool_name}")
        print(f"   参数: {json.dumps(params, indent=2)}")
        print(f"   原因: {reasoning}")
        print(f"\n\033[33m   [y] 确认执行  [n] 拒绝  [s] 跳过  [a] 本次会话全部允许\033[0m")

        response = input("   > ").strip().lower()

        if response == "a":
            # 缓存本次会话的允许
            SessionCache.set(f"allow_{tool_name}", True)
            return True
        return response == "y"
```

---

### 4. 流式输出与渲染

CLI 最大的技术挑战是**增量渲染**——如何在终端中实时显示 LLM 的生成过程。

#### SSE 流式处理

```python
import aiohttp

async def stream_llm_response(
    messages: list[dict],
    model: str = "gpt-4o"
) -> AsyncGenerator[StreamChunk, None]:
    """流式获取 LLM 响应"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            tool_call_buffer = {}
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])

                    if data == "[DONE]":
                        break

                    delta = data["choices"][0]["delta"]

                    # 文本内容 → 直接渲染
                    if delta.get("content"):
                        yield StreamChunk(type="text", content=delta["content"])

                    # 工具调用 → 缓冲后渲染
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc["index"]
                            if idx not in tool_call_buffer:
                                tool_call_buffer[idx] = {
                                    "name": "", "arguments": ""
                                }
                            if tc.get("function", {}).get("name"):
                                tool_call_buffer[idx]["name"] = tc["function"]["name"]
                                yield StreamChunk(
                                    type="tool_name",
                                    content=tc["function"]["name"]
                                )
                            if tc.get("function", {}).get("arguments"):
                                tool_call_buffer[idx]["arguments"] += tc["function"]["arguments"]
```

#### Markdown 增量渲染

终端中的 Markdown 渲染需要在内容不全的情况下也能正确展示：

```python
class IncrementalMarkdownRenderer:
    """增量 Markdown 渲染器——处理不完整的标记"""

    def __init__(self):
        self.buffer = ""
        self.in_code_block = False
        self.code_lang = ""

    def feed(self, chunk: str) -> str:
        """输入 chunk，返回可渲染的文本"""
        self.buffer += chunk
        return self._render_current()

    def _render_current(self) -> str:
        """基于当前 buffer 渲染——处理不完整的标记"""
        result = []

        # 处理代码块（即使未闭合也能渲染）
        lines = self.buffer.split("\n")
        in_fence = False

        for i, line in enumerate(lines):
            if line.startswith("```"):
                if not in_fence:
                    in_fence = True
                    lang = line[3:].strip()
                    result.append(f"\033[90m┌─ {lang or 'code'} ─\033[0m")
                else:
                    in_fence = False
                    result.append(f"\033[90m└──\033[0m")
            elif in_fence:
                result.append(f"\033[90m│ {line}\033[0m")
            else:
                result.append(line)

        # 如果 buffer 末尾有未闭合的代码块，加省略提示
        if in_fence:
            result.append(f"\033[90m│ ...\033[0m")

        return "\n".join(result)
```

#### 差异化更新

不要每次都清屏重绘。只刷新变化的部分：

```python
class DiffRenderer:
    """差异化终端渲染——只更新变化区域"""

    def __init__(self):
        self.last_output = ""

    def render(self, new_output: str) -> str:
        """只输出与上次不同的部分"""
        if new_output == self.last_output:
            return ""

        # 找到公共前缀
        common_len = 0
        for a, b in zip(self.last_output, new_output):
            if a == b:
                common_len += 1
            else:
                break

        diff = new_output[common_len:]
        self.last_output = new_output
        return diff
```

---

### 5. 安全护栏

LLM + CLI 的 #1 安全威胁是 **Shell 注入**。LLM 生成的命令包含用户输入时，注入风险极高。

#### Shell 注入防护

```python
import shlex

class ShellSafety:
    """Shell 命令安全——防止注入攻击"""

    # 危险模式列表
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",           # 递归删除根目录
        r">\s*/dev/",              # 覆盖设备文件
        r"mkfs\.",                 # 格式化文件系统
        r"dd\s+if=",               # 裸磁盘操作
        r"chmod\s+777",            # 不安全的权限
        r"eval\s+",                # 二次解析
        r"\$\(.+\)",               # 命令替换
        r"`.+`",                   # 反引号命令替换
        r";\s*\w+",                # 命令链接
        r"\|\s*\w+",               # 管道到未知命令
        r"sudo\s+",                # 提权
    ]

    @classmethod
    def validate(cls, command: str) -> tuple[bool, str]:
        """检查命令是否有注入风险"""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Command matches dangerous pattern: {pattern}"
        return True, "OK"

    @staticmethod
    def parameterize(template: str, **kwargs) -> str:
        """参数化命令构建——禁止字符串拼接"""
        # 所有参数通过 shlex.quote() 转义
        safe_kwargs = {k: shlex.quote(str(v)) for k, v in kwargs.items()}
        return template.format(**safe_kwargs)

# ✅ 安全：参数化
cmd = ShellSafety.parameterize(
    "git log --author={author} --since={date}",
    author="Zhang San",
    date="2026-01-01"
)
# 结果: "git log --author='Zhang San' --since=2026-01-01"

# ❌ 危险：字符串拼接
cmd = f"git log --author='{user_input}'"  # user_input 可以包含 '; rm -rf /
```

#### 输出安全

```python
class OutputSanitizer:
    """输出清理——防止 ANSI 注入和敏感信息泄露"""

    SENSITIVE_PATTERNS = [
        (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
        (r"sk-[0-9a-zA-Z]{48}", "OPENAI_API_KEY"),
        (r"github_pat_[0-9a-zA-Z_]{36}", "GITHUB_TOKEN"),
        (r"-----BEGIN.*?PRIVATE KEY-----", "PRIVATE_KEY"),
        (r"ghp_[0-9a-zA-Z]{36}", "GITHUB_TOKEN"),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        for pattern, label in cls.SENSITIVE_PATTERNS:
            text = re.sub(pattern, f"[REDACTED:{label}]", text)
        return text

    @staticmethod
    def strip_ansi_control(text: str) -> str:
        """移除可能被注入的 ANSI 控制序列（保留已知安全的）"""
        safe_codes = {"\033[0m", "\033[1m", "\033[3", "\033[9"}
        # 简化处理：移除所有未知的转义序列
        return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)
```

#### 审计日志

```python
@dataclass
class AuditEntry:
    timestamp: str
    session_id: str
    user: str
    action: str           # "tool_call" | "command_exec" | "file_write"
    tool_name: str | None
    params: dict | None
    command: str | None
    result_summary: str   # 前 200 字符
    success: bool
    sandbox_blocked: bool

class AuditLogger:
    """所有操作必须落审计日志"""

    def log(self, entry: AuditEntry):
        # 写入不可变日志（append-only）
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def query(self, session_id: str) -> list[AuditEntry]:
        """查询某次会话的所有操作"""
        # 用于排查问题和安全审计
        pass
```

---

### 6. 工程化实践

#### Session 持久化

CLI 随时可能被 Ctrl+C 中断或终端关闭。会话状态必须可恢复：

```python
@dataclass
class SessionState:
    session_id: str
    created_at: str
    workdir: str
    model: str
    history: list[dict]
    agent_state: dict       # Agent 的内部状态（步骤计数、中间结果）
    tool_allowlist: set     # 本次会话用户允许的危险工具

class SessionManager:
    """会话管理器——支持中断恢复"""

    def save(self, state: SessionState):
        path = f"~/.cli-ai/sessions/{state.session_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(state), f, default=str)

    def resume(self, session_id: str) -> SessionState | None:
        path = f"~/.cli-ai/sessions/{session_id}.json"
        if not os.path.exists(path):
            return None

        with open(path) as f:
            data = json.load(f)

        state = SessionState(**data)
        print(f"\033[36m恢复会话 {session_id} ({len(state.history)} 轮历史)\033[0m")
        return state
```

#### 跨平台兼容

```python
class PlatformAdapter:
    """跨平台适配——Windows/Linux/macOS"""

    @staticmethod
    def get_shell() -> str:
        if sys.platform == "win32":
            return "powershell.exe"  # 或 "cmd.exe"
        return os.environ.get("SHELL", "/bin/bash")

    @staticmethod
    def get_home() -> Path:
        return Path.home()

    @staticmethod
    def is_writable(path: Path) -> bool:
        """跨平台检查文件可写性"""
        if sys.platform == "win32":
            return os.access(path, os.W_OK)
        return path.is_file() and os.access(path, os.W_OK)

    @staticmethod
    def normalize_path(path: str) -> str:
        """标准化路径分隔符"""
        return str(Path(path))

    @staticmethod
    def escape_args(args: list[str]) -> str:
        """跨平台参数转义"""
        if sys.platform == "win32":
            # Windows 需要特殊的引号处理
            return " ".join(f'"{a}"' if " " in a else a for a in args)
        return shlex.join(args)
```

#### 离线降级

```python
class OfflineFallback:
    """LLM 不可达时的降级策略"""

    def __init__(self):
        self.online = True

    async def check_connectivity(self) -> bool:
        try:
            # 快速健康检查
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{API_BASE}/health", timeout=3) as r:
                    self.online = r.status == 200
        except Exception:
            self.online = False
        return self.online

    async def run(self, prompt: str) -> str:
        if self.online:
            return await self.agent.run(prompt)

        # 离线降级
        print("\033[33m⚠️  LLM 服务不可达，使用本地模式\033[0m")
        # 策略 1: 本地小模型 (Ollama / llama.cpp)
        try:
            return await local_llm.run(prompt)
        except Exception:
            pass

        # 策略 2: 规则引擎（匹配已知命令模板）
        return self._rule_based_fallback(prompt)

    def _rule_based_fallback(self, prompt: str) -> str:
        """基于规则的降级——匹配已知命令模式"""
        patterns = {
            r"当前分支": "git branch --show-current",
            r"最近提交": "git log --oneline -5",
            r"文件列表": "ls -la",
        }
        for pattern, cmd in patterns.items():
            if re.search(pattern, prompt):
                return f"离线模式推荐命令: {cmd}"
        return "离线模式无法处理此请求，请稍后重试。"
```

---

## 第二部分：六大黄金法则

### 法则一：流式优先

**首 token 必须在 500ms 内出现，否则用户会觉得"卡了"。**

```python
# ✅ 流式输出——用户立刻看到响应
async for chunk in agent.stream(prompt):
    print(chunk, end="", flush=True)

# ❌ 阻塞等待——用户盯着空白终端 10 秒
result = await agent.run(prompt)
print(result)
```

关键指标：
- TTFT (Time To First Token) < 500ms
- 工具调用开始前显示 spinner
- 长任务显示进度（`[3/8] 正在分析...`）

### 法则二：确认不可绕过

**任何破坏性操作必须经过用户确认，且确认流程不能被绕过。**

```python
# ✅ 确认流程是硬编码的，不受 prompt 控制
if tool.permission == DANGEROUS:
    confirmed = await ConfirmationFlow.confirm(...)
    if not confirmed:
        return "操作被用户拒绝"

# ❌ 把确认交给 LLM 判断——LLM 可能被 prompt 注入欺骗
if llm_thinks_its_safe:  # 危险！
    execute()
```

### 法则三：沙箱即默认

**Agent 默认只有最小权限。** 每次扩大权限都需要用户显式批准。

```
默认：只读当前工作目录
  → 用户批准后：可写入工作目录
  → 用户再批准：可执行命令
  → 用户再批准：可访问网络
  → 用户再批准：可访问其他目录
```

### 法则四：状态可恢复

**会话随时可能中断（Ctrl+C / 终端关闭 / 网络断开）。**

```
必须持久化的状态：
  - 对话历史（最近 N 轮）
  - Agent 当前步骤（避免重复工作）
  - 用户的确认缓存（本次会话允许的操作）
  - 中间生成的文件列表

不需要持久化的：
  - 完整的工具调用日志（存审计）
  - 原始 LLM 响应（可从对话历史重建）
```

### 法则五：输入即代码

**用户输入和 LLM 输出都可能包含恶意内容。永远参数化，永远不拼接。**

```python
# ✅ 参数化 Shell 调用
subprocess.run(["git", "log", "--author", user_input])

# ❌ 字符串拼接——注入风险
os.system(f"git log --author='{user_input}'")
```

### 法则六：审计全覆盖

**每一条命令执行、每一个文件写入、每一次 LLM 调用，都必须落审计日志。** 出问题时，审计日志是唯一的真相来源。

---

## 第三部分：常见反模式

| 反模式 | 症状 | 正确做法 |
|--------|------|----------|
| **假装思考** | 没有实际计算却显示"正在分析..." 30 秒 | 只在真正等待 LLM 时显示 spinner，其余显示具体进度 |
| **阻塞等待** | 等 LLM 完全回复才显示任何内容 | 首 token 到达立即渲染，边生成边显示 |
| **裸奔执行** | Agent 生成的命令直接 `os.system()` | 参数化 + 沙箱 + 确认三重防护 |
| **状态遗忘** | 重开会话后 Agent 从零开始 | Session 持久化，支持 `--resume` |
| **过度自动化** | 简单的 `ls` 都要 Agent 处理 | 区分"需要 LLM"和"直接执行"，用规则引擎处理已知命令 |
| **ANSI 轰炸** | 每帧都清屏重绘整个界面 | 差异化更新，只修改变化区域 |
| **Prompt 即代码** | 把所有逻辑写在 system prompt 里 | 安全约束放在代码层（不可绕过），prompt 只管行为 |
| **无语境感知** | 不读工作目录就直接生成命令 | 启动时收集项目上下文（文件结构、git 状态、语言/框架） |

---

## 第四部分：生产部署检查清单

```
□ 1. 首 token 延迟 < 500ms？（流式输出，不等完整回复）
□ 2. 所有工具调用都有 spinner/进度指示吗？
□ 3. DANGEROUS 操作都有确认流程吗？（不可绕过）
□ 4. 命令执行用了参数化而非字符串拼接吗？
□ 5. 文件访问限制在工作目录内吗？
□ 6. 命令有 timeout 吗？（推荐 30-120s）
□ 7. Session 可以 --resume 恢复吗？
□ 8. 所有操作有审计日志吗？
□ 9. Ctrl+C 能立即中断 LLM 调用和工具执行吗？
□ 10. 输出过滤了敏感信息（API key、token）吗？
□ 11. 跨平台兼容测试过了吗（Windows/Linux/macOS）？
□ 12. LLM 不可达时有离线降级策略吗？
□ 13. 相同 prompt 有结果缓存吗（减少重复 LLM 调用）？
□ 14. 支持非交互模式（--oneshot / pipe）吗？
```

---

## 核心思想

> **LLM + CLI 的本质是一个能调用操作系统的对话界面。它的核心矛盾不是"LLM 够不够聪明"，而是"如何在终端这个受限媒介上，安全、流畅地暴露系统能力给 LLM"。**

三条铁律：

1. **流式 > 完整。** 用户看见 80% 正确的流式输出，远好于等 10 秒看到 100% 正确的完整输出。
2. **安全在代码层，不在 prompt 层。** 确认流程、权限检查、注入防护——这些必须是硬编码的，不能让 LLM 自己决定安不安全。
3. **Session 是状态，不是文件。** 会话应该像浏览器标签页一样——关了可以恢复，切换不丢失。

---

*最后更新: 2026-05-25*

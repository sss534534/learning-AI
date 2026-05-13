# GPU集群调度

> AI工作负载的调度策略与资源管理

## 1. AI工作负载特点

### 1.1 与CPU工作负载的差异

| 维度 | CPU工作负载 | GPU工作负载 |
|------|-------------|-------------|
| **资源粒度** | Pod/容器级别 | GPU卡级别 |
| **调度单位** |  milli-cores | 整卡或部分（MIG） |
| **任务类型** | 多样化 | 主要为批处理（训练/推理） |
| **运行时长** | 秒级到小时级 | 分钟级到天数 |
| **故障影响** | 单个Pod | 整卡或节点 |
| **拓扑敏感** | 低 | 高（NVLink/PCIe） |

### 1.2 GPU任务分类

**训练任务（Training）：**
- 长时间运行（小时到天）
- 需要多卡/多节点并行
- 对网络带宽要求高
- 故障成本高（需checkpoint）

**推理任务（Inference）：**
- 实时或准实时
- 延迟敏感
- 需要弹性伸缩
- 资源可共享

**微调任务（Fine-tuning）：**
- 中等时长（分钟到小时）
- 资源需求介于两者之间
- 需要快速启动

---

## 2. GPU调度核心概念

### 2.1 GPU共享技术

**MIG（Multi-Instance GPU）：**
```
A100 40GB → 分割为7个5GB实例
- 硬件级隔离
- 独立显存和计算单元
- 适合推理场景
```

**MPS（Multi-Process Service）：**
```
多个进程共享GPU上下文
- 软件级共享
- 显存共享但计算分时
- 适合小模型推理
```

**Time-Slicing：**
```
多个任务分时使用GPU
- Kubernetes默认方式
- 简单但无隔离
- 仅适合开发测试
```

### 2.2 GPU拓扑

**典型节点拓扑：**
```
┌─────────────────────────────────────────┐
│              Node (DGX A100)            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │GPU0 │←→│GPU1 │←→│GPU2 │←→│GPU3 │      │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘      │
│     └───────┘ └───────┘              │
│         NVSwitch                      │
│     ┌───────┐ ┌───────┐              │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐      │
│  │GPU4 │←→│GPU5 │←→│GPU6 │←→│GPU7 │      │
│  └─────┘ └─────┘ └─────┘ └─────┘      │
└─────────────────────────────────────────┘
```

**拓扑感知调度：**
- 优先调度到同一NVSwitch下的GPU
- 减少跨节点通信
- 提升分布式训练效率

---

## 3. Kubernetes GPU调度

### 3.1 标准Device Plugin机制

**架构：**
```
Kubelet → Device Plugin → NVIDIA Driver → GPU
              ↓
         上报GPU资源
              ↓
    Scheduler调度Pod到节点
```

**资源声明：**
```yaml
resources:
  limits:
    nvidia.com/gpu: 2  # 请求2个GPU
```

**局限：**
- 不支持GPU共享
- 无拓扑感知
- 无队列管理

### 3.2 NVIDIA GPU Operator

**组件：**
```
GPU Operator
├── NVIDIA Driver
├── Container Toolkit
├── Device Plugin
├── DCGM Exporter (监控)
├── MIG Manager
└── Feature Discovery
```

**安装：**
```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
```

### 3.3 调度扩展

**Scheduler Extender：**
```
默认Scheduler → 过滤节点 → Extender二次过滤 → 绑定Pod
                      ↑
                 自定义调度逻辑
```

**Scheduler Framework：**
```
更灵活的插件机制
- PreFilter
- Filter
- PostFilter
- PreScore
- Score
- Reserve
- Permit
- PreBind
- Bind
- PostBind
```

---

## 4. 批调度系统

### 4.1 Volcano

**华为开源的K8s批调度系统**

**核心特性：**

**1. Gang Scheduling（组调度）**
```
All-or-Nothing：
- 任务需要4个GPU
- 只有3个GPU可用
- 不调度，等待4个都可用

避免资源死锁，提升利用率
```

**2. 队列管理（Queue）**
```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: default
spec:
  weight: 1          # 权重
  capability:
    cpu: 1000
    memory: 2000Gi
    nvidia.com/gpu: 32
```

**3. 优先级与抢占**
```
高优先级任务可抢占低优先级任务
- 优雅终止（Graceful Eviction）
- 支持checkpoint恢复
```

**4. 拓扑感知调度**
```
自动识别GPU拓扑
优先调度到高带宽连接的GPU
```

**Job定义：**
```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llm-training
spec:
  schedulerName: volcano
  minAvailable: 4    # Gang Scheduling
  queue: default
  tasks:
    - replicas: 4
      name: worker
      template:
        spec:
          containers:
            - name: training
              image: training:latest
              resources:
                limits:
                  nvidia.com/gpu: 1
```

### 4.2 Kueue

**K8s原生作业队列管理**

**特点：**
- 原生K8s风格
- 与标准Scheduler集成
- 支持资源配额和抢占

**概念模型：**
```
ResourceFlavor → ClusterQueue → LocalQueue → Workload
     ↓                ↓              ↓          ↓
   资源定义         集群队列       命名空间队列  具体任务
```

**配置示例：**
```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-queue
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: a100
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 32
```

### 4.3 Yunikorn

**Apache开源调度器**

**特点：**
- 支持YARN和K8s
- 丰富的队列策略
- 应用感知调度

---

## 5. 推理服务调度

### 5.1 弹性伸缩

**HPA（Horizontal Pod Autoscaler）：**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Pods
      pods:
        metric:
          name: gpu_utilization
        target:
          type: AverageValue
          averageValue: "70"
```

**自定义指标：**
- GPU利用率
- 请求队列长度
- P99延迟
- Token生成速率

### 5.2 多模型服务调度

**模型路由：**
```
请求 → 路由层 → 模型A服务
              → 模型B服务
              → 模型C服务
```

**调度策略：**
- 基于模型负载
- 基于请求特征
- 基于成本优化

### 5.3 冷启动优化

**问题：** 大模型加载时间长（秒到分钟）

**解决方案：**

1. **预加载（Pre-warming）**
   ```
   保持最小副本数运行
   新请求到达时可立即响应
   ```

2. **模型缓存**
   ```
   共享模型权重存储
   新Pod快速挂载
   ```

3. **分层加载**
   ```
   先加载关键层，服务启动
   后台异步加载剩余层
   ```

---

## 6. 网络与存储优化

### 6.1 网络优化

**RDMA（Remote Direct Memory Access）：**
```
特点：
- 绕过CPU，直接内存访问
- 低延迟（微秒级）
- 高带宽（100-400Gbps）

适用：
- 分布式训练
- 多节点推理
```

**GPUDirect RDMA：**
```
GPU内存 ↔ 网卡 ↔ 远程GPU内存
无需经过CPU和系统内存
```

**网络拓扑：**
```
推荐：
- 训练集群：全连接（Fat Tree）
- 推理集群：收敛比可更高
```

### 6.2 存储优化

**Checkpoint存储：**
```
挑战：
- 大模型checkpoint可达数百GB
- 频繁写入影响训练

方案：
- 异步checkpoint
- 分层存储（热数据SSD，冷数据对象存储）
- 增量checkpoint
```

**模型加载优化：**
```
- 模型权重预加载到节点
- 本地SSD缓存
- 并行加载
```

---

## 7. 监控与运维

### 7.1 GPU监控指标

**DCGM（Data Center GPU Manager）：**

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| GPU利用率 | 计算单元使用率 | < 50% |
| 显存使用 | GPU内存占用 | > 90% |
| 温度 | GPU核心温度 | > 80°C |
| 功耗 | 实际功耗 | > 额定值90% |
| Xid错误 | 硬件错误码 | > 0 |

**Prometheus + Grafana：**
```yaml
# ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  endpoints:
    - port: metrics
```

### 7.2 日志与追踪

**分布式训练追踪：**
```
- 记录每个step的耗时
- 追踪通信开销
- 识别慢节点
```

**推理服务追踪：**
```
- 请求全链路追踪
- Token生成延迟分析
- 队列等待时间
```

### 7.3 故障处理

**常见故障：**

| 故障 | 现象 | 处理 |
|------|------|------|
| GPU掉卡 | nvidia-smi看不到卡 | 重启节点或驱动 |
| ECC错误 | 显存纠错 | 隔离或更换 |
| 温度高 | 降频 | 检查散热 |
| 通信超时 | NCCL错误 | 检查网络 |

**自动恢复：**
```
- 健康检查
- 自动驱逐故障Pod
- 重新调度
- 从checkpoint恢复
```

---

## 8. 架构师最佳实践

### 8.1 集群设计Checklist

**硬件规划：**
- [ ] 计算：GPU型号、数量、拓扑
- [ ] 网络：带宽、延迟、拓扑
- [ ] 存储：容量、IOPS、吞吐量
- [ ] 散热：功耗、冷却能力

**软件栈：**
- [ ] 操作系统：驱动版本、内核参数
- [ ] 容器运行时：nvidia-docker、containerd
- [ ] 调度器：Volcano/Kueue/标准Scheduler
- [ ] 监控：DCGM、Prometheus、Grafana

**运维流程：**
- [ ] 故障检测与自动恢复
- [ ] 资源配额管理
- [ ] 成本分摊与优化
- [ ] 安全隔离

### 8.2 成本优化策略

**训练场景：**
- 抢占式实例（Spot Instance）
- 自动checkpoint和恢复
- 训练时间预估和调度

**推理场景：**
- 自动扩缩容
- 模型量化降低资源需求
- 请求批处理

### 8.3 常见陷阱

**陷阱1：忽视拓扑**
- 问题：跨NVSwitch通信慢
- 解决：拓扑感知调度

**陷阱2：资源碎片化**
- 问题：小任务占用大节点
- 解决：Gang Scheduling

**陷阱3：缺乏隔离**
- 问题：训练任务影响推理
- 解决：资源池分离

---

*最后更新：2026-05-07*

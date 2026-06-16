# Kubernetes 核心

> Kubernetes（K8s）是容器编排的事实标准，提供部署、扩展、管理容器化应用的自动化平台。

---

## 目录

1. [核心概念](#1-核心概念)
2. [架构组件](#2-架构组件)
3. [Pod与工作负载](#3-pod与工作负载)
4. [服务发现与网络](#4-服务发现与网络)
5. [存储与配置](#5-存储与配置)
6. [安全与RBAC](#6-安全与rbac)
7. [集群管理](#7-集群管理)

---

## 1. 核心概念

### 1.1 声明式API

用户描述期望状态（Desired State），K8s持续弥合当前状态到期望状态。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
```

### 1.2 控制循环

```
Replicas: 3 (期望)
  ↓
Controller: 发现当前只有2个Pod
  ↓
Action: 创建第3个Pod
  ↓
状态收敛到期望
```

---

## 2. 架构组件

### 2.1 控制面（Control Plane）

| 组件 | 功能 |
|------|------|
| kube-apiserver | 所有操作的入口 |
| etcd | 集群状态存储（键值对） |
| kube-scheduler | Pod调度到合适节点 |
| kube-controller-manager | 控制器集合 |

### 2.2 数据面（Data Plane）

| 组件 | 功能 |
|------|------|
| kubelet | 节点代理，管理Pod |
| kube-proxy | 网络代理，负载均衡 |
| 容器运行时 | containerd / CRI-O |

### 2.3 请求流程

```
kubectl → API Server → etcd ← Controller Manager ←→ API Server
              ↓                ↓
         Scheduler        kubelet (各节点)
              ↓                ↓
         API Server  →  创建Pod → 容器运行时
```

---

## 3. Pod与工作负载

### 3.1 Pod

最小的调度单元，一个或多个共享网络/存储的容器：

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    image: myapp:1.0
  - name: sidecar
    image: envoy:latest  # 共享网络栈
```

### 3.2 工作负载类型

| 类型 | 用途 | 特点 |
|------|------|------|
| Deployment | 无状态应用 | 滚动更新，扩缩容 |
| StatefulSet | 有状态应用 | 稳定网络标识，有序部署 |
| DaemonSet | 每个节点运行一个 | 日志收集，监控 |
| Job/CronJob | 批处理任务 | 运行完成即退出 |

### 3.3 滚动更新策略

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1       # 最大不可用Pod数
    maxSurge: 1             # 最大超过期望Pod数
```

---

## 4. 服务发现与网络

### 4.1 Service

```yaml
apiVersion: v1
kind: Service
spec:
  type: ClusterIP  # ClusterIP | NodePort | LoadBalancer
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
```

### 4.2 Ingress

7层路由，将外部流量引入集群：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
spec:
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

### 4.3 网络模型

```
每个Pod拥有唯一IP
Pod间可直接通信（无需NAT）
节点与Pod间可直接通信
kube-proxy 实现 Service 负载均衡
（iptables/IPVS/eBPF 模式）
```

---

## 5. 存储与配置

### 5.1 存储类型

| 类型 | 用途 | 生命周期 |
|------|------|----------|
| emptyDir | 临时存储 | Pod生命周期 |
| HostPath | 节点本地存储 | 节点生命周期 |
| PVC | 持久化存储 | 独立于Pod |
| ConfigMap | 配置注入 | 独立于Pod |
| Secret | 敏感信息注入 | 独立于Pod |

### 5.2 PVC示例

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: ssd
```

---

## 6. 安全与RBAC

### 6.1 RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
kind: RoleBinding
subjects:
- kind: User
  name: developer
roleRef:
  kind: Role
  name: pod-reader
```

### 6.2 Pod安全

| 机制 | 说明 |
|------|------|
| PodSecurityContext | 容器运行用户/组 |
| Seccomp | 限制系统调用 |
| AppArmor | 强制访问控制 |
| NetworkPolicy | 网络隔离 |

---

## 7. 集群管理

### 7.1 节点管理

```bash
# 节点维护
kubectl cordon node-1      # 标记不可调度
kubectl drain node-1       # 驱逐Pod
# 维护完成后
kubectl uncordon node-1    # 恢复调度
```

### 7.2 监控与排障

```bash
kubectl get events --watch
kubectl describe pod <pod-name>
kubectl logs -f <pod-name>
kubectl exec -it <pod-name> -- sh
kubectl top node  # 节点资源使用
```

### 7.3 集群运维工具

| 工具 | 功能 |
|------|------|
| kubeadm | 集群初始化 |
| Helm | 包管理 |
| Kustomize | 声明式配置管理 |
| ArgoCD | GitOps部署 |
| Prometheus + Grafana | 监控 |
| Velero | 备份恢复 |

---

## 延伸阅读

- Kubernetes官方文档: https://kubernetes.io/docs/
- *Kubernetes in Action* (Marko Lukša) — K8s实战经典
- *Kubernetes: Up and Running* (Kelsey Hightower) — 入门推荐
- CKAD/CKA 认证指南
- kubectl cheat sheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/

---

*最后更新：2026-06-15*

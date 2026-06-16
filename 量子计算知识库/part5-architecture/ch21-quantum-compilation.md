# 第二十一章：量子编译

> 量子编译将高层次的量子算法转化为可在特定量子硬件上执行的低层次指令序列，包含优化、映射和调度。

---

## 目录

1. [量子编译流程](#1-量子编译流程)
2. [量子电路优化](#2-量子电路优化)
3. [门分解](#3-门分解)
4. [布局与路由](#4-布局与路由)

---

## 1. 量子编译流程

```
高级算法 → 电路生成 → 优化 → 映射 → 调度 → 硬件指令
```

**主要挑战：**
- 有限量子比特连通性
- 有限的本征门集
- 噪声和错误率不同
- 量子比特退相干时间限制

---

## 2. 量子电路优化

| 优化技术 | 描述 | 效果 |
|---------|------|------|
| 门合并 | 相邻可逆门消除 | 减少门数 |
| 门移动 | 交换不重叠门 | 减少深度 |
| 模板匹配 | 模式替换 | 多个优化 |
| 代数优化 | ZX-calculus | 全局优化 |

```python
# Qiskit transpiler 示例
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation

qc = QuantumCircuit(2)
qc.h(0)
qc.h(0)  # 抵消
qc.cx(0, 1)
qc.cx(0, 1)  # 抵消

pm = PassManager([Optimize1qGates(), CXCancellation()])
qc_opt = pm.run(qc)
```

---

## 3. 门分解

### 3.1 Solovay-Kitaev定理

> 任何单量子比特门可在 $O(\log^c(1/\epsilon))$ 步内用有限门集近似到精度 $\epsilon$。

### 3.2 常见分解

| 目标门 | 基础门分解 |
|--------|-----------|
| $T$ | $H S^\dagger H$（近似） |
| $R_z(\theta)$ | $T^k$ 序列（近似） |
| CNOT | 特定硬件本征门 |

---

## 4. 布局与路由

### 4.1 耦合图映射

实际硬件有有限连通性：

```
量子芯片耦合图示例（IBM Q）：
0 ─ 1 ─ 2
    │
    3 ─ 4
```

两比特门只能在相连的量子比特上执行。

### 4.2 SWAP插入

当需要连接非相邻量子比特时插入SWAP门：

```python
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap

coupling = CouplingMap([[0,1], [1,2], [1,3], [3,4]])
pm = PassManager([
    SabreLayout(coupling),
    SabreSwap(coupling)
])
```

---

## 延伸阅读

- Qiskit Transpiler 文档
- *Quantum Compilation* (Amy) — 综述
- arXiv: 量子编译综述

---

*最后更新：2026-06-15*

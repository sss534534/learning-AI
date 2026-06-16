# 附录D：量子编程入门

> 量子编程入门指南，涵盖主流框架的安装、基础示例、调试技巧和最佳实践。

---

## 目录

1. [环境安装](#1-环境安装)
2. [Qiskit编程](#2-qiskit编程)
3. [Cirq编程](#3-cirq编程)
4. [PennyLane编程](#4-pennylane编程)
5. [电路可视化](#5-电路可视化)
6. [在真实硬件上运行](#6-在真实硬件上运行)
7. [调试与最佳实践](#7-调试与最佳实践)

---

## 1. 环境安装

### 1.1 Qiskit

```bash
# 安装 Qiskit 核心包
pip install qiskit qiskit-aer

# 可选：额外模块
pip install qiskit-ibm-runtime     # IBM 硬件访问
pip install qiskit-nature           # 量子化学
pip install qiskit-finance          # 金融计算
pip install qiskit-optimization     # 优化问题
pip install qiskit-machine-learning # 机器学习

# 验证安装
python -c "import qiskit; print(qiskit.__version__)"
```

### 1.2 Cirq

```bash
pip install cirq
python -c "import cirq; print(cirq.__version__)"
```

### 1.3 PennyLane

```bash
pip install pennylane
python -c "import pennylane as qml; print(qml.__version__)"
```

---

## 2. Qiskit编程

### 2.1 Hello Quantum

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# 创建 2 量子比特、2 经典比特的电路
qc = QuantumCircuit(2, 2)

# 构建 Bell 态电路
qc.h(0)           # 在第一个量子比特上施加 Hadamard 门
qc.cx(0, 1)       # CNOT：以 q0 为控制，q1 为目标
qc.measure([0, 1], [0, 1])  # 测量两个量子比特

print(qc.draw())  # 输出 ASCII 电路图

# 运行模拟
sim = AerSimulator()
result = sim.run(qc, shots=1024).result()
counts = result.get_counts()
print(counts)
# 预期输出：{'00': ~512, '11': ~512}
```

### 2.2 量子纠缠与Bell不等式

```python
def create_bell_state(qc, q0, q1):
    """在 q0, q1 上创建 Bell 态 |Φ⁺⟩"""
    qc.h(q0)
    qc.cx(q0, q1)

# Bell 态测量验证
qc = QuantumCircuit(2, 2)
create_bell_state(qc, 0, 1)
qc.measure_all()

sim = AerSimulator()
result = sim.run(qc, shots=10000).result()
counts = result.get_counts()

# Bell 态只产生 00 或 11（理想情况）
assert '01' not in counts or counts['01'] == 0
assert '10' not in counts or counts['10'] == 0
print(f"关联度验证: 仅 '00' 和 '11' 出现 ✓")
```

### 2.3 量子傅里叶变换

```python
from qiskit.circuit.library import QFT
import numpy as np

def qft_example(n_qubits=3):
    """实现 n 量子比特 QFT"""
    qc = QuantumCircuit(n_qubits)
    
    # 制备输入态 |+++⟩
    for i in range(n_qubits):
        qc.h(i)
    
    # QFT
    qft_circ = QFT(num_qubits=n_qubits, approximation_degree=0)
    qc = qc.compose(qft_circ)
    
    # 逆 QFT
    iqft_circ = QFT(num_qubits=n_qubits, inverse=True)
    qc = qc.compose(iqft_circ)
    
    # 应回到 |000⟩
    qc.measure_all()
    return qc

qc = qft_example(3)
sim = AerSimulator()
result = sim.run(qc, shots=1024).result()
print(f"QFT + IQFT 应回到 |000⟩: {result.get_counts()}")
```

### 2.4 噪声模拟

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# 构建噪声模型
noise_model = NoiseModel()
dep_error = depolarizing_error(0.01, 1)  # 1% 退极化
noise_model.add_all_qubit_quantum_error(dep_error, ['h', 'cx'])

# 带噪声运行
sim_noisy = AerSimulator(noise_model=noise_model)
result_noisy = sim_noisy.run(qc, shots=1024).result()
print(f"含噪声结果: {result_noisy.get_counts()}")
```

---

## 3. Cirq编程

### 3.1 基本示例

```python
import cirq

# 创建量子比特
q0, q1 = cirq.LineQubit.range(2)

# 构建电路
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

print(circuit)

# 模拟运行
sim = cirq.Simulator()
result = sim.run(circuit, repetitions=1000)
print(result.histogram(key='result'))
```

### 3.2 参数化电路

```python
import sympy

# 定义参数
theta = sympy.Symbol('theta')

# 参数化电路
q = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.XPowGate(exponent=theta)(q),
    cirq.measure(q, key='m')
)

# 不同参数值运行
for val in [0.0, 0.5, 1.0]:
    params = cirq.ParamResolver({theta: val})
    result = cirq.Simulator().run(circuit, params, repetitions=100)
    p1 = result.histogram(key='m')[1] / 100
    print(f"θ={val:.1f}π, P(|1⟩)={p1:.2f}")
```

---

## 4. PennyLane编程

### 4.1 量子节点 (QNode)

```python
import pennylane as qml
import numpy as np

# 定义设备
dev = qml.device('default.qubit', wires=2)

# 定义量子节点
@qml.qnode(dev)
def bell_state_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

print(bell_state_circuit())
# 输出: [0.707+0j, 0+0j, 0+0j, 0.707+0j]
# 对应: 1/√2(|00⟩ + |11⟩)
```

### 4.2 变分量子分类器

```python
@qml.qnode(dev)
def variational_circuit(params, x):
    """变分量子分类器"""
    # 数据编码
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    
    # 变分层
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[0], wires=0)
    qml.RX(params[1], wires=1)
    
    # 测量
    return qml.expval(qml.PauliZ(0))

# 训练
params = np.array([0.5, 0.3])
x = np.array([0.8, 0.6])
result = variational_circuit(params, x)
print(f"输出期望值: {result:.4f}")

# 自动微分
dc = qml.grad(variational_circuit)(params, x)
print(f"梯度: {dc}")
```

### 4.3 优化示例

```python
def cost(params):
    return (variational_circuit(params, x) - 1.0) ** 2

opt = qml.AdamOptimizer(stepsize=0.1)
params = np.random.randn(2)

for step in range(50):
    params = opt.step(cost, params)
    if step % 10 == 0:
        print(f"Step {step}: cost = {cost(params):.6f}")
```

---

## 5. 电路可视化

### 5.1 Qiskit绘图

```python
from qiskit.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.measure_all()

# ASCII 文本表示
print(qc.draw())

# mpl 图像表示
fig = circuit_drawer(qc, output='mpl', style='iqp')
plt.show()

# 结果直方图
result = AerSimulator().run(qc, shots=1024).result()
fig = plot_histogram(result.get_counts())
plt.show()
```

### 5.2 Cirq绘图

```python
import cirq

q = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q[0]),
    cirq.CNOT(q[0], q[1]),
    cirq.CNOT(q[0], q[2]),
    cirq.measure(*q, key='result')
)

# ASCII
print(circuit)

# 使用 cirq.to_json 导出
json_str = cirq.to_json(circuit)
```

---

## 6. 在真实硬件上运行

### 6.1 IBM Quantum

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# 保存 API 令牌（首次运行）
# QiskitRuntimeService.save_account(token='YOUR_IBM_QUANTUM_TOKEN')

# 连接服务
service = QiskitRuntimeService()
backend = service.get_backend('ibm_brisbane')  # 127 量子比特

# 创建 Bell 态电路
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 在真实硬件上运行
sampler = Sampler(backend)
job = sampler.run([qc], shots=4000)
print(f"Job ID: {job.job_id()}")

# 等待结果
result = job.result()
print(f"测量结果: {result.quasi_dists}")
```

### 6.2 选择后端的最佳实践

```python
# 查询可用后端
for backend in service.backends():
    config = backend.configuration()
    print(f"{backend.name}: {config.n_qubits} qubits, "
          f"1Q error={config.single_qubit_gate_error:.3e}, "
          f"2Q error={config.two_qubit_gate_error:.3e}")
```

---

## 7. 调试与最佳实践

### 7.1 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 测量结果占比不等 | 非标准化态 | 检查电路逻辑 |
| 模拟与硬件不一致 | 噪声影响 | 应用误差缓解 |
| 电路太深无法运行 | 退相干 | 优化电路深度 |
| 量子比特连通性错误 | 映射问题 | 使用 transpiler |

### 7.2 调试技巧

```python
# 1. 逐步检查中间态
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
# 保存中间态快照
qc.save_statevector(label='after_h')
qc.cx(0, 1)
qc.save_statevector(label='after_cx')

sim = AerSimulator()
result = sim.run(qc).result()
data = result.data()
print("After H:", data['after_h'])
print("After CX:", data['after_cx'])

# 2. 检查深度和门数
print(f"Depth: {qc.depth()}")
print(f"Operations: {qc.count_ops()}")

# 3. 使用 transpiler 信息
from qiskit import transpile
qc_transpiled = transpile(qc, basis_gates=['u1','u2','u3','cx'])
print(f"Transpiled depth: {qc_transpiled.depth()}")
```

### 7.3 性能优化

- **最小化电路深度**：减少门数 = 减少噪声影响
- **使用并行门**：在不重叠的量子比特上同时执行操作
- **选择合适的基础门集**：不同硬件有不同原生门
- **利用电路简化**：合并相邻的可抵消门
- **对称性验证**：利用物理对称性检测错误

### 7.4 学习路径

1. 从模拟器开始，理解量子逻辑
2. 在有噪声的模拟器上测试鲁棒性
3. 在免费的真实硬件上运行小规模实验
4. 学习误差缓解技术
5. 逐步增加算法复杂度

---

*最后更新：2026-06-15*

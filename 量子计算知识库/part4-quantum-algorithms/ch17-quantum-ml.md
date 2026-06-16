# 第十七章：量子机器学习

> 量子机器学习（QML）结合量子计算和机器学习，利用量子并行性和纠缠来加速经典ML算法或实现新的学习范式。

---

## 目录

1. [量子数据与量子特征](#1-量子数据与量子特征)
2. [量子支持向量机](#2-量子支持向量机)
3. [量子神经网络](#3-量子神经网络)
4. [量子核方法](#4-量子核方法)

---

## 1. 量子数据与量子特征

### 1.1 量子特征映射

将经典数据 $x \in \mathbb{R}^n$ 映射到量子态：

$$|\Phi(x)\rangle = U_{\Phi(x)}|0\rangle^{\otimes n}$$

```python
# 使用ZZFeatureMap编码数据
from qiskit.circuit.library import ZZFeatureMap

n_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
feature_map.draw()
```

### 1.2 振幅编码 vs 角度编码

| 编码 | 描述 | 量子比特数 | 优点 |
|------|------|-----------|------|
| 角度编码 | 特征作为旋转角度 | $n$ | 简单 |
| 振幅编码 | 特征作为振幅 | $\log n$ | 高效 |
| 基编码 | 特征作为二进制位 | 特征位数 | 直观 |

---

## 2. 量子支持向量机

### 2.1 量子核估计

经典SVM核函数被量子核替代：

$$K(x_i, x_j) = |\langle \Phi(x_i) | \Phi(x_j) \rangle|^2$$

量子核在高维Hilbert空间中计算，可能提供指数级优势。

### 2.2 实现

```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel

kernel = FidelityQuantumKernel(feature_map=feature_map)
kernel_matrix = kernel.evaluate(x_train)
```

---

## 3. 量子神经网络

### 3.1 变分量子电路（VQC）

```mermaid
graph LR
    Data[经典数据] --> Encoding[量子编码]
    Encoding --> VarCircuit[变分层]
    VarCircuit --> Measurement[测量]
    Measurement --> Classical[经典优化]
    Classical -->|更新参数| VarCircuit
```

```python
from qiskit.circuit.library import RealAmplitudes

# 变分层
ansatz = RealAmplitudes(num_qubits=4, reps=3)
ansatz.draw()
```

### 3.2 参数化量子电路

$$|\psi(\theta)\rangle = U_L(\theta_L) \cdots U_2(\theta_2)U_1(\theta_1)|0\rangle$$

**参数平移规则（梯度计算）：**

$$\frac{\partial \langle f \rangle}{\partial \theta_i} = \frac{\langle f \rangle_{\theta_i + \pi/2} - \langle f \rangle_{\theta_i - \pi/2}}{2}$$

### 3.3 贫瘠高原问题

随机量子电路的梯度随量子比特数指数级衰减 → 需要特殊初始化策略。

---

## 4. 量子核方法

### 4.1 量子核的优势

| 经典核 | 量子核 |
|--------|--------|
| 多项式/高斯核 | Hilbert空间内积 |
| 特征空间固定 | 可学习特征映射 |
| 维度有限 | 维度指数级 |

### 4.2 应用场景

- 小样本学习（药物发现、材料科学）
- 高维数据分类
- 量子态分类

---

## 延伸阅读

- *Quantum Machine Learning* (Schuld & Petruccione) — QML教材
- Qiskit ML文档: https://qiskit-community.github.io/qiskit-machine-learning/
- *Supervised Learning with Quantum Computers* (Schuld & Petruccione)

---

*最后更新：2026-06-15*

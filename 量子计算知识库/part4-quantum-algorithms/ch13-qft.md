# 第十三章：量子傅里叶变换

> 量子傅里叶变换（QFT）是许多重要量子算法的核心，包括Shor算法。本章深入探讨经典傅里叶变换、量子傅里叶变换的电路构造、性质以及相位估计。

---

## 13.1 经典傅里叶变换回顾

### 13.1.1 离散傅里叶变换

离散傅里叶变换（DFT）将向量 $(x_0, x_1, \dots, x_{N-1})$ 变换为 $(y_0, y_1, \dots, y_{N-1})$，其中
$$y_k = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j e^{2\pi i j k / N}$$

逆离散傅里叶变换：
$$x_j = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} y_k e^{-2\pi i j k / N}$$

### 13.1.2 快速傅里叶变换

快速傅里叶变换（FFT）可以在 $O(N\log N)$ 时间内计算DFT。

---

## 13.2 量子傅里叶变换

### 13.2.1 定义

量子傅里叶变换对计算基态 $|j\rangle$ 的作用是
$$QFT|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i j k / N} |k\rangle$$

其中 $N = 2^n$。

写成二进制形式：
$$QFT|j_1j_2\cdots j_n\rangle = \frac{1}{2^{n/2}} (|0\rangle + e^{2\pi i 0.j_n}|1\rangle) \otimes (|0\rangle + e^{2\pi i 0.j_{n-1}j_n}|1\rangle) \otimes \dots \otimes (|0\rangle + e^{2\pi i 0.j_1j_2\cdots j_n}|1\rangle)$$

其中 $0.j_kj_{k+1}\cdots j_n = j_k/2 + j_{k+1}/4 + \dots + j_n/2^{n-k+1}$ 是二进制小数。

### 13.2.2 单量子比特和两量子比特例子

**单量子比特QFT**（即Hadamard门）：
$$QFT|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} = H|0\rangle$$
$$QFT|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} = H|1\rangle$$

**两量子比特QFT**：
$$QFT|00\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$
$$QFT|01\rangle = \frac{1}{2}(|00\rangle + |01\rangle - |10\rangle - |11\rangle)$$
$$QFT|10\rangle = \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$
$$QFT|11\rangle = \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)$$

---

## 13.3 QFT的电路构造

### 13.3.1 受控相位门

定义受控相位门 $R_k$：
$$R_k = \begin{bmatrix} 1 & 0 \\ 0 & e^{2\pi i / 2^k} \end{bmatrix}$$

### 13.3.2 QFT电路

QFT可以用Hadamard门和受控相位门构造。

**n量子比特QFT电路**：
1. 对第一个量子比特应用Hadamard门
2. 对第一个量子比特和第二个量子比特应用受控 $R_2$ 门（第一个为控制，第二个为目标）
3. ...
4. 对第一个量子比特和第n个量子比特应用受控 $R_n$ 门
5. 对第二个量子比特应用Hadamard门
6. ...
7. 最后交换量子比特的顺序（可选）

### 13.3.3 电路的高效性

QFT可以用 $O(n^2)$ 个量子门实现，而经典FFT需要 $O(N\log N) = O(n2^n)$ 步。

这是量子指数加速的一个来源。

---

## 13.4 QFT的性质

### 13.4.1 幺正性

QFT是幺正变换，因为它的矩阵表示是幺正矩阵。

### 13.4.2 线性性

QFT是线性变换：
$$QFT(\alpha|j\rangle + \beta|k\rangle) = \alpha QFT|j\rangle + \beta QFT|k\rangle$$

### 13.4.3 周期性

QFT可以用来检测周期函数的周期，这是Shor算法的关键。

---

## 13.5 相位估计

### 13.5.1 问题描述

**相位估计问题**：给定幺正算子 $U$ 和它的一个本征态 $|u\rangle$，满足 $U|u\rangle = e^{2\pi i\phi}|u\rangle$，估计相位 $\phi$。

### 13.5.2 相位估计算法

**相位估计算法**：
1. 准备两个寄存器：控制寄存器（n量子比特，初始化为 $|0\rangle$）和目标寄存器（初始化为 $|u\rangle$）
2. 对控制寄存器的每个量子比特应用Hadamard门
3. 对控制寄存器的第j个量子比特，应用受控 $U^{2^{j-1}}$ 门
4. 对控制寄存器应用逆QFT
5. 测量控制寄存器，得到相位 $\phi$ 的n比特估计

### 13.5.3 工作原理

经过步骤2和3后，控制寄存器的状态为：
$$\frac{1}{2^{n/2}} \sum_{k=0}^{2^n-1} e^{2\pi i\phi k} |k\rangle$$

这正是状态 $|2^n\phi\rangle$ 的QFT。应用逆QFT得到 $|2^n\phi\rangle$，测量得到 $2^n\phi$ 的整数近似，从而估计 $\phi$。

---

## 13.6 应用

### 13.6.1 周期发现

相位估计可以用来发现幺正算子的幂次的周期，这是Shor算法的核心。

### 13.6.2 阶发现

对于函数 $f(x) = a^x \mod N$，相位估计可以用来发现它的阶（最小的r使得 $a^r \equiv 1 \mod N$）。

---

## 本章小结

本章深入探讨了量子傅里叶变换，包括：
- 经典傅里叶变换回顾
- 量子傅里叶变换的定义
- QFT的电路构造
- QFT的性质
- 相位估计
- 应用（周期发现、阶发现）

QFT是许多重要量子算法的核心。

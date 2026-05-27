# 第九章：量子比特

> 量子比特（qubit）是量子计算的基本信息单元。本章深入探讨量子比特的数学描述、Bloch球表示、单量子比特门以及多量子比特系统。

---

## 9.1 量子比特的定义

### 9.1.1 经典比特与量子比特

**经典比特**：可以处于 $0$ 或 $1$ 两个状态之一。

**量子比特**：可以处于状态 $|0\rangle$、$|1\rangle$，或者它们的任意线性组合
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$
其中 $\alpha, \beta \in \mathbb{C}$，满足 $|\alpha|^2 + |\beta|^2 = 1$。

### 9.1.2 概率解释

测量量子比特时，得到结果 $0$ 的概率为 $|\alpha|^2$，得到结果 $1$ 的概率为 $|\beta|^2$。

测量后，量子比特坍缩到测量结果对应的状态。

---

## 9.2 Bloch球表示

### 9.2.1 Bloch球参数化

任意单量子比特态可以参数化为
$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$
其中 $0 \leq \theta \leq \pi$，$0 \leq \phi < 2\pi$。

这对应于Bloch球上的一个点，球坐标为 $(\theta, \phi)$。

### 9.2.2 Bloch球的性质

- Bloch球是半径为1的球
- 球面上的点对应纯态
- 球内的点对应混合态
- $|0\rangle$ 在北极，$|1\rangle$ 在南极
- $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$ 在 $x$ 轴正方向
- $|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$ 在 $x$ 轴负方向
- $|i\rangle = (|0\rangle + i|1\rangle)/\sqrt{2}$ 在 $y$ 轴正方向
- $|-i\rangle = (|0\rangle - i|1\rangle)/\sqrt{2}$ 在 $y$ 轴负方向

### 9.2.3 密度算子与Bloch球

对于混合态，密度算子可以表示为
$$\rho = \frac{I + \vec{r} \cdot \vec{\sigma}}{2}$$
其中 $\vec{r}$ 是Bloch向量，$|\vec{r}| \leq 1$。

---

## 9.3 单量子比特门

### 9.3.1 Pauli门

Pauli门是最重要的单量子比特门：

**Pauli-X门（量子非门）**：
$$X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
作用：$X|0\rangle = |1\rangle$，$X|1\rangle = |0\rangle$
Bloch球：绕 $x$ 轴旋转 $\pi$ 弧度

**Pauli-Y门**：
$$Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}$$
Bloch球：绕 $y$ 轴旋转 $\pi$ 弧度

**Pauli-Z门**：
$$Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$
作用：$Z|0\rangle = |0\rangle$，$Z|1\rangle = -|1\rangle$
Bloch球：绕 $z$ 轴旋转 $\pi$ 弧度

### 9.3.2 Hadamard门

Hadamard门：
$$H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

作用：
$$H|0\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
$$H|1\rangle = |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

Bloch球：绕 $x+z$ 轴旋转 $\pi$ 弧度，或者等价地绕 $x$ 轴旋转 $\pi/2$ 再绕 $z$ 轴旋转 $\pi$。

### 9.3.3 相位门

**S门（相位门）**：
$$S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}$$

**T门（$\pi/8$ 门）**：
$$T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}$$

**一般相位门**：
$$R_z(\theta) = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}$$
绕 $z$ 轴旋转 $\theta$ 弧度。

### 9.3.4 旋转门

**绕 $x$ 轴旋转**：
$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}X = \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}$$

**绕 $y$ 轴旋转**：
$$R_y(\theta) = e^{-i\theta Y/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}Y = \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}$$

**绕 $z$ 轴旋转**：
$$R_z(\theta) = e^{-i\theta Z/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}Z = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}$$

### 9.3.5 单量子比特门的一般形式

任意单量子比特幺正算子可以表示为
$$U = e^{i\alpha}R_n(\theta)$$
其中 $\alpha$ 是全局相位，$R_n(\theta)$ 是绕单位向量 $\hat{n}$ 旋转 $\theta$ 弧度的旋转算子。

**Z-Y分解**：任意单量子比特幺正算子可以分解为
$$U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$$

---

## 9.4 多量子比特系统

### 9.4.1 张量积

$n$ 量子比特系统的状态空间是
$$(\mathbb{C}^2)^{\otimes n} = \mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \dots \otimes \mathbb{C}^2$$

其维数为 $2^n$。

### 9.4.2 计算基

计算基为
$$|x_1x_2\cdots x_n\rangle = |x_1\rangle \otimes |x_2\rangle \otimes \dots \otimes |x_n\rangle$$
其中 $x_i \in \{0, 1\}$。

任意 $n$ 量子比特态可以表示为
$$|\psi\rangle = \sum_{x \in \{0,1\}^n} \alpha_x |x\rangle$$
其中 $\sum_x |\alpha_x|^2 = 1$。

### 9.4.3 乘积态与纠缠态

**乘积态**：可以表示为各个单量子比特态的张量积
$$|\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \dots \otimes |\psi_n\rangle$$

**纠缠态**：不能表示为乘积态的态

例子：Bell态
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

---

## 9.5 量子寄存器

### 9.5.1 量子寄存器的概念

量子寄存器是一组量子比特，用于存储量子信息。

### 9.5.2 基态的整数表示

通常将量子寄存器的基态与整数对应：
$$|x\rangle = |x_{n-1}\cdots x_1x_0\rangle$$
其中 $x = x_{n-1}2^{n-1} + \dots + x_12 + x_0$。

### 9.5.3 量子寄存器的操作

可以对量子寄存器执行各种幺正操作，如量子傅里叶变换、模加法等。

---

## 9.6 测量

### 9.6.1 单量子比特测量

在计算基下测量单量子比特态 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$：
- 得到结果 0 的概率：$|\alpha|^2$
- 得到结果 1 的概率：$|\beta|^2$
- 测量后状态坍缩到对应的基态

### 9.6.2 多量子比特测量

可以测量部分量子比特：

对于两体态
$$|\psi\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle + \delta|11\rangle$$

测量第一个量子比特得到 0 的概率：$|\alpha|^2 + |\beta|^2$，测量后状态为
$$\frac{\alpha|00\rangle + \beta|01\rangle}{\sqrt{|\alpha|^2 + |\beta|^2}}$$

---

## 本章小结

本章深入探讨了量子比特，包括：
- 量子比特的定义与经典比特的区别
- Bloch球表示
- 单量子比特门（Pauli门、Hadamard门、相位门、旋转门）
- 多量子比特系统与张量积
- 量子寄存器
- 量子测量

量子比特是量子计算的基础单元。

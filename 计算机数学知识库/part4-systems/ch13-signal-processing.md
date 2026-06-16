# 第十三章：信号与信息处理

> 信号处理研究如何从物理世界的连续信号中提取、变换和恢复信息，是通信、音频、图像、AI等领域的基础。

---

## 目录

1. [信号基础](#1-信号基础)
2. [傅里叶级数](#2-傅里叶级数)
3. [傅里叶变换](#3-傅里叶变换)
4. [离散傅里叶变换与FFT](#4-离散傅里叶变换与fft)
5. [小波变换](#5-小波变换)
6. [信号处理在计算机中的应用](#6-信号处理在计算机中的应用)

---

## 1. 信号基础

### 1.1 信号分类

| 类型 | 定义 | 例 |
|------|------|----|
| 连续 | 定义在连续时间上 | 语音模拟信号 |
| 离散 | 定义在离散时间点上 | 数字音频 |
| 周期 | $x(t+T) = x(t)$ | 正弦波 |
| 非周期 | 不重复 | 语音片段 |

### 1.2 基本操作

- 时移：$x(t - t_0)$
- 缩放：$x(at)$
- 反转：$x(-t)$
- 卷积：$(x * h)(t) = \int x(\tau)h(t-\tau)d\tau$

---

## 2. 傅里叶级数

### 2.1 周期信号的三角级数表示

任何周期信号（满足Dirichlet条件）可分解为：

$$f(t) = a_0 + \sum_{n=1}^\infty [a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t)]$$

### 2.2 系数公式

$$a_0 = \frac{1}{T}\int_T f(t)dt, \quad \omega_0 = \frac{2\pi}{T}$$

$$a_n = \frac{2}{T}\int_T f(t)\cos(n\omega_0 t)dt, \quad b_n = \frac{2}{T}\int_T f(t)\sin(n\omega_0 t)dt$$

### 2.3 指数形式

更简洁的复数形式：

$$f(t) = \sum_{n=-\infty}^\infty c_n e^{jn\omega_0 t}, \quad c_n = \frac{1}{T}\int_T f(t)e^{-jn\omega_0 t}dt$$

---

## 3. 傅里叶变换

### 3.1 定义

将时域信号转换到频域：

$$F(\omega) = \mathcal{F}\{f(t)\} = \int_{-\infty}^\infty f(t)e^{-j\omega t}dt$$

**逆变换：**
$$f(t) = \mathcal{F}^{-1}\{F(\omega)\} = \frac{1}{2\pi}\int_{-\infty}^\infty F(\omega)e^{j\omega t}d\omega$$

### 3.2 重要性质

| 性质 | 时域 | 频域 |
|------|------|------|
| 线性 | $af(t) + bg(t)$ | $aF(\omega) + bG(\omega)$ |
| 时移 | $f(t-t_0)$ | $F(\omega)e^{-j\omega t_0}$ |
| 频移 | $f(t)e^{j\omega_0 t}$ | $F(\omega - \omega_0)$ |
| 卷积 | $f(t)*g(t)$ | $F(\omega)G(\omega)$ |
| 对偶 | $F(t)$ | $2\pi f(-\omega)$ |

### 3.3 常见变换对

| $f(t)$ | $F(\omega)$ |
|--------|-------------|
| $\delta(t)$ | $1$ |
| $1$ | $2\pi\delta(\omega)$ |
| $e^{-at}u(t), a>0$ | $\frac{1}{a+j\omega}$ |
| $\cos(\omega_0 t)$ | $\pi[\delta(\omega+\omega_0) + \delta(\omega-\omega_0)]$ |

---

## 4. 离散傅里叶变换与FFT

### 4.1 DFT 定义

对 $N$ 点离散信号：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi}{N}kn}, \quad k = 0,1,\ldots,N-1$$

### 4.2 朴素DFT的问题

直接计算复杂度 $O(N^2)$ — 对实用 $N$ 不可行。

### 4.3 FFT：Cooley-Tukey算法

**核心思想：分治**

$$X[k] = \sum_{n=0}^{N/2-1} x[2n] \cdot e^{-j\frac{2\pi}{N/2}kn} + W_N^k \sum_{n=0}^{N/2-1} x[2n+1] \cdot e^{-j\frac{2\pi}{N/2}kn}$$

复杂度从 $O(N^2)$ 降至 $O(N \log N)$。

```python
import numpy as np

def fft(x):
    """递归FFT实现（教学用）"""
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N//2] * odd,
                           even + factor[N//2:] * odd])

# 验证
x = np.random.randn(1024)
assert np.allclose(fft(x), np.fft.fft(x))
```

### 4.4 FFT的应用

```python
# 快速卷积（时域卷积 = 频域点乘）
def fast_convolution(x, h):
    X = np.fft.fft(x, n=len(x)+len(h)-1)
    H = np.fft.fft(h, n=len(x)+len(h)-1)
    return np.fft.ifft(X * H).real
```

---

## 5. 小波变换

### 5.1 傅里叶的局限性

- 傅里叶变换丢失时间信息
- 短时傅里叶变换（STFT）在时间和频率分辨率间存在折中

### 5.2 连续小波变换（CWT）

$$\text{CWT}_f(a,b) = \frac{1}{\sqrt{|a|}}\int f(t)\,\psi^*\left(\frac{t-b}{a}\right)dt$$

- $a$：尺度（频率）
- $b$：平移（时间）
- $\psi$：母小波

### 5.3 常用小波

| 小波 | 特点 | 应用 |
|------|------|------|
| Haar | 最简单，不连续 | 教学/快速变换 |
| Daubechies | 紧支撑，正交 | 图像压缩(JPEG 2000) |
| Morlet | 复值，连续光滑 | 时频分析 |

### 5.4 离散小波变换（DWT）

```python
import pywt

# 单层分解
coeffs = pywt.dwt([1, 2, 3, 4], 'db1')
cA, cD = coeffs  # 近似系数，细节系数

# 多层分解
coeffs = pywt.wavedec([1, 2, 3, 4, 5, 6, 7, 8], 'db1', level=3)
```

---

## 6. 信号处理在计算机中的应用

### 6.1 图像处理

| 操作 | 数学本质 |
|------|---------|
| 模糊 | 卷积低通滤波核 |
| 边缘检测Sobel | 离散微分算子 |
| JPEG压缩 | DCT + 量化 + 熵编码 |
| JPEG 2000 | DWT + 量化 + 算术编码 |

### 6.2 音频处理

| 操作 | 数学本质 |
|------|---------|
| MFCC特征 | FFT → Mel滤波器组 → DCT |
| 降噪 | STFT → 阈值处理 → ISTFT |
| 卷积混响 | 信号与冲击响应卷积 |

### 6.3 卷积神经网络中的信号处理

- 卷积操作本质上与信号处理中的卷积一致
- 池化 = 下采样
- 频域视角：CNN在低频学习全局结构，高频学习局部纹理

---

## 延伸阅读

- *The Scientist and Engineer's Guide to Digital Signal Processing* (Smith)
- *Discrete-Time Signal Processing* (Oppenheim)
- NumPy FFT 文档: `np.fft`
- SciPy 信号处理: `scipy.signal`

---

*最后更新：2026-06-15*

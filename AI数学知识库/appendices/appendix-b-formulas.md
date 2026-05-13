# 附录B：常用公式汇总

## 线性代数公式

### 向量运算

$$
\vec{a} \cdot \vec{b} = \sum_{i} a_i b_i = \|\vec{a}\| \|\vec{b}\| \cos\theta
$$

$$
\|\vec{v}\|_p = \left(\sum_i |v_i|^p\right)^{1/p}
$$

$$
(\vec{u} \otimes \vec{v})_{ij} = u_i v_j
$$

### 矩阵运算

$$
(AB)^T = B^T A^T
$$

$$
(AB)^{-1} = B^{-1} A^{-1}
$$

$$
\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)
$$

### 特征值与特征向量

$$
A\vec{v} = \lambda \vec{v}
$$

$$
\det(A - \lambda I) = 0
$$

### SVD

$$
A = U \Sigma V^T
$$

$$
\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sum_i \sigma_i^2}
$$

---

## 微积分公式

### 导数

$$
\frac{d}{dx} x^n = nx^{n-1}
$$

$$
\frac{d}{dx} e^x = e^x
$$

$$
\frac{d}{dx} \ln x = \frac{1}{x}
$$

$$
\frac{d}{dx} \sin x = \cos x
$$

$$
\frac{d}{dx} \cos x = -\sin x
$$

### 链式法则

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

### 积分

$$
\int x^n dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
$$

$$
\int e^x dx = e^x + C
$$

$$
\int \frac{1}{x} dx = \ln|x| + C
$$

---

## 概率论公式

### 贝叶斯定理

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
$$

### 期望与方差

$$
E[X] = \sum_x x P(X=x)
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

### 常用分布

**正态分布：**
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

**伯努利分布：**
$$
P(X=1) = p, \quad P(X=0) = 1-p
$$

---

## 信息论公式

### 熵

$$
H(X) = -\sum_x P(x) \log P(x)
$$

### 交叉熵

$$
H(P, Q) = -\sum_x P(x) \log Q(x)
$$

### KL散度

$$
D_{KL}(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

### 关系

$$
H(P, Q) = H(P) + D_{KL}(P\|Q)
$$

---

## 深度学习公式

### 激活函数

**Sigmoid：**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Tanh：**
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**ReLU：**
$$
\text{ReLU}(x) = \max(0, x)
$$

**GELU：**
$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

### 损失函数

**MSE：**
$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2
$$

**交叉熵：**
$$
\mathcal{L}_{\text{CE}} = -\sum_i y_i \log \hat{y}_i
$$

### 注意力机制

**Scaled Dot-Product Attention：**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 优化器

**SGD：**
$$
\theta_{t+1} = \theta_t - \alpha \nabla \mathcal{L}
$$

**Adam：**
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

---

## 位置编码

### 正弦位置编码

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### RoPE（二维）

$$
R_{\Theta,m}^d = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
$$

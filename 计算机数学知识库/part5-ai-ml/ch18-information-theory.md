# 第十八章：信息论

> 信息论量化了信息的度量、存储和通信，是理解机器学习损失函数、模型评估和特征选择的理论基础。

---

## 目录

1. [信息熵](#1-信息熵)
2. [联合熵与条件熵](#2-联合熵与条件熵)
3. [互信息](#3-互信息)
4. [KL散度与交叉熵](#4-kl散度与交叉熵)
5. [信道容量](#5-信道容量)
6. [编码理论](#6-编码理论)
7. [在机器学习中的应用](#7-在机器学习中的应用)

---

## 1. 信息熵

### 1.1 自信息

事件 $x$ 发生所提供的信息量：

$$I(x) = -\log P(x)$$

- 概率越小，信息量越大
- 必然事件 $P(x)=1$ 的信息量为 0

### 1.2 熵的定义

离散随机变量 $X$ 的熵：

$$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x) = \mathbb{E}[-\log P(X)]$$

**直观理解：** 熵是平均信息量，也是不确定性的度量。

### 1.3 举例

```python
import numpy as np

def entropy(p):
    p = np.array(p)
    return -np.sum(p * np.log2(p + 1e-10))

# 公平硬币
print(entropy([0.5, 0.5]))   # 1.0 bit

# 确定性事件
print(entropy([1.0, 0.0]))   # 0.0 bit

# 不均匀
print(entropy([0.9, 0.1]))   # 0.469 bit
```

### 1.4 最大熵原理

在约束条件下，应选择熵最大的分布——对未知做最少的假设。

---

## 2. 联合熵与条件熵

### 2.1 联合熵

$$H(X, Y) = -\sum_{x, y} P(x, y) \log P(x, y)$$

### 2.2 条件熵

知道 $Y$ 后 $X$ 的剩余不确定性：

$$H(X|Y) = -\sum_{x, y} P(x, y) \log P(x|y) = H(X, Y) - H(Y)$$

**链式法则：**
$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

---

## 3. 互信息

### 3.1 定义

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

**意义：** 知道 $Y$ 减少的 $X$ 的不确定性。

### 3.2 互信息图示

```
    H(X)           H(Y)
┌────────┐   ┌────────┐
│        │   │        │
│ H(X|Y) │ I │ H(Y|X) │
│        │   │        │
└────────┘   └────────┘
    H(X,Y) = H(X|Y) + I + H(Y|X)
```

### 3.3 互信息与相关性

- 相关系数仅度量线性关系
- 互信息度量任意依赖关系（包括非线性）

```python
from sklearn.feature_selection import mutual_info_classif

# 使用互信息进行特征选择
X = np.random.randn(100, 5)
y = (X[:, 0] + 0.5 * X[:, 1]**2 > 0).astype(int)
mi = mutual_info_classif(X, y)
print("互信息得分:", mi)
```

---

## 4. KL散度与交叉熵

### 4.1 KL散度

衡量两个分布 $P$ 和 $Q$ 的差异：

$$D_{KL}(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

**性质：**
- $D_{KL}(P\|Q) \geq 0$，等号成立当且仅当 $P = Q$
- 不对称：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$

### 4.2 交叉熵

$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{KL}(P\|Q)$$

### 4.3 作为损失函数

在分类问题中，交叉熵损失：

$$L = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}$$

```python
import torch.nn.functional as F

# 交叉熵损失 = LogSoftmax + NLLLoss
loss = F.cross_entropy(logits, targets)
```

**为什么用交叉熵而非MSE：**
- 梯度更大，训练更快
- 概率解释：等价于分类分布下的MLE
- 输出自动约束为概率分布（softmax）

### 4.4 KL散度在蒸馏中的应用

知识蒸馏损失：

$$L_{\text{KD}} = \alpha \cdot H(y_{\text{hard}}, y_{\text{student}}) + \beta \cdot D_{KL}(y_{\text{teacher}}^\tau \| y_{\text{student}}^\tau)$$

其中 $\tau$ 是温度参数。

---

## 5. 信道容量

### 5.1 定义

信道容量是信道能可靠传输的最大信息率：

$$C = \max_{P(X)} I(X; Y)$$

### 5.2 二元对称信道

$$C = 1 - H(p) = 1 + p\log p + (1-p)\log(1-p)$$

其中 $p$ 是误码率。

### 5.3 香农编码定理

> 若信息传输速率 $R < C$，则存在编码使得错误概率任意小。
> 若 $R > C$，则不可能可靠通信。

---

## 6. 编码理论

### 6.1 前缀编码

没有任何码字是另一码字的前缀 → 唯一可解码。

### 6.2 霍夫曼编码

最优前缀编码，平均码长最小：

```python
import heapq

def huffman(freq):
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: len(p[1]))
```

### 6.3 香农的信源编码定理

> 平均码长不小于信源的熵 $H(X)$ 且可任意接近。

---

## 7. 在机器学习中的应用

### 7.1 决策树与信息增益

$$IG(Y|X) = H(Y) - H(Y|X) = I(Y; X)$$

```python
from sklearn.tree import DecisionTreeClassifier

# 决策树内置使用信息增益（ID3算法）
clf = DecisionTreeClassifier(criterion='entropy')
```

### 7.2 VAE与ELBO

变分自编码器优化证据下界（ELBO）：

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|p(z))$$

第一项：重建损失；第二项：KL正则化项。

### 7.3 扩散模型的噪声调度

前向扩散过程添加噪声，其信息论解释：逐步移除数据信息，最终趋近纯噪声分布。

### 7.4 互信息最大化（InfoMax）

对比学习（如CLIP、SimCLR）的核心思想：
- 正样本对互信息最大化
- 负样本对互信息最小化

```python
# SimCLR 的 NT-Xent 损失（归一化温度标度交叉熵损失）
def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j: 正样本对的表示
    # 互信息最大化 = 正样本对相似度最大化
    ...
```

---

## 延伸阅读

- *Elements of Information Theory* (Cover & Thomas) — 信息论标准教材
- *Information Theory, Inference, and Learning Algorithms* (MacKay) — 信息论与ML结合
- *Deep Learning* (Goodfellow et al.) 第3章 — 概率与信息论

---

*最后更新：2026-06-15*

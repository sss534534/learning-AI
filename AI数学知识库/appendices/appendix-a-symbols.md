# 附录A：数学符号速查表

## 元数据
- **难度**: ⭐
- **前置知识**: 无
- **关联文件**: 全部章节
- **最后更新**: 2026-06-12
---
## 常用符号

| 符号 | 含义 | 示例 |
|------|------|------|
| $\mathbb{R}^n$ | n维实数向量空间 | $\vec{x} \in \mathbb{R}^n$ |
| $\mathbb{R}^{m \times n}$ | m×n 实数矩阵空间 | $A \in \mathbb{R}^{m \times n}$ |
| $\nabla$ | 梯度算子 | $\nabla f$ |
| $\partial$ | 偏导数 | $\frac{\partial f}{\partial x}$ |
| $\Sigma$ | 求和 | $\sum_{i=1}^n x_i$ |
| $\prod$ | 连乘 | $\prod_{i=1}^n x_i$ |
| $\|\vec{x}\|$ | 向量范数 | $\|\vec{x}\|_2$ |
| $\|\vec{x}\|_p$ | Lp范数 | $\|\vec{x}\|_1$ |
| $E[X]$ | 期望 | $E[X] = \sum x P(X=x)$ |
| $\text{Var}(X)$ | 方差 | $\text{Var}(X) = E[(X-E[X])^2]$ |
| $\text{Cov}(X,Y)$ | 协方差 | $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$ |
| $P(A\|B)$ | 条件概率 | B条件下A的概率 |
| $p(x\|y)$ | 条件概率密度 | $f_{X|Y}(x\|y)$ |

## 线性代数符号

| 符号 | 含义 |
|------|------|
| $\vec{x}$ | 向量 |
| $A$ | 矩阵 |
| $A^T$ | 矩阵转置 |
| $A^{-1}$ | 矩阵逆 |
| $A^\dagger$ | 矩阵伪逆 |
| $I$ | 单位矩阵 |
| $0$ | 零矩阵 |
| $\text{tr}(A)$ | 矩阵的迹 |
| $\det(A)$ | 行列式 |
| $\text{rank}(A)$ | 矩阵的秩 |
| $\lambda$ | 特征值 |
| $\vec{v}$ | 特征向量 |
| $U, \Sigma, V^T$ | SVD分解 |
| $\odot$ | 哈达玛积（元素乘法） |
| $\otimes$ | 克罗内克积 |

## 微积分符号

| 符号 | 含义 |
|------|------|
| $\frac{df}{dx}$ | 导数 |
| $\frac{\partial f}{\partial x}$ | 偏导数 |
| $\nabla f$ | 梯度 |
| $\nabla^2 f$ | 拉普拉斯算子 |
| $\oint$ | 曲线积分 |
| $\iint$ | 曲面积分 |

## 概率论符号

| 符号 | 含义 |
|------|------|
| $X, Y$ | 随机变量 |
| $p(x)$ | 概率质量函数 |
| $f(x)$ | 概率密度函数 |
| $F(x)$ | 累积分布函数 |
| $N(\mu, \sigma^2)$ | 正态分布 |
| $Bernoulli(p)$ | 伯努利分布 |
| $Beta(\alpha, \beta)$ | Beta分布 |
| $\mathcal{N}(0, I)$ | 标准正态分布 |

## 信息论符号

| 符号 | 含义 |
|------|------|
| $H(X)$ | 熵 |
| $H(P, Q)$ | 交叉熵 |
| $D_{KL}(P\|Q)$ | KL散度 |
| $I(X;Y)$ | 互信息 |

## 深度学习符号

| 符号 | 含义 |
|------|------|
| $\theta$ | 模型参数 |
| $\mathcal{L}$ | 损失函数 |
| $\alpha$ | 学习率 |
| $\beta_1, \beta_2$ | Adam动量参数 |
| $\sigma$ | Sigmoid函数 |
| $\text{softmax}(x)_i$ | Softmax函数 |
| $\text{ReLU}(x)$ | ReLU函数 |
| $\text{GELU}(x)$ | GELU函数 |

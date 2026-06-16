# 第四章：概率与统计

> 量子力学本质上是概率性的，Born规则将量子态的幅度平方解释为概率，量子统计则处理混合态。

---

## 目录

1. [经典概率回顾](#1-经典概率回顾)
2. [量子概率](#2-量子概率)
3. [测量统计](#3-测量统计)
4. [量子信息论基础](#4-量子信息论基础)

---

## 1. 经典概率回顾

### 1.1 概率公理

Kolmogorov 公理：概率是归一化的测度。

### 1.2 随机变量与期望

经典比特与概率：

$$P(X=0) = p, \quad P(X=1) = 1-p$$

---

## 2. 量子概率

### 2.1 Born规则

量子态 $|\psi\rangle$ 下测量 $M$ 得到结果 $m$ 的概率：

$$P(m) = \langle \psi | M_m^\dagger M_m | \psi \rangle$$

### 2.2 量子 vs 经典概率

| 特性 | 经典概率 | 量子概率 |
|------|---------|---------|
| 状态 | 概率分布 | 态矢量/密度矩阵 |
| 叠加 | 混合（or） | 相干（and） |
| 干涉 | ❌ | ✅ |
| 测量 | 更新分布 | 坍缩 |

### 2.3 密度矩阵

混合态的完整描述：

$$\rho = \sum_i p_i |\psi_i\rangle\langle \psi_i|$$

**性质：**
- $\text{Tr}(\rho) = 1$
- $\rho \succeq 0$（半正定）
- 纯态：$\rho^2 = \rho$，混合态：$\rho^2 < \rho$

---

## 3. 测量统计

### 3.1 期望值

$$\langle M \rangle = \text{Tr}(M\rho)$$

### 3.2 方差

$$\Delta M^2 = \langle M^2 \rangle - \langle M \rangle^2$$

### 3.3 海森堡不确定性原理

$$\Delta A \cdot \Delta B \geq \frac{1}{2} |\langle [A, B] \rangle|$$

例如位置-动量：$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$

---

## 4. 量子信息论基础

### 4.1 冯·诺依曼熵

量子系统的信息度量：

$$S(\rho) = -\text{Tr}(\rho \log \rho)$$

### 4.2 Holevo界

量子信息传输的基本极限：

$$\chi \leq S(\rho) - \sum_i p_i S(\rho_i)$$

---

## 延伸阅读

- *Quantum Probability* (Meyer) — 量子概率
- Nielsen & Chuang 第2、11章 — 量子信息论
- *Quantum Information and Quantum Computing* (Preskill) — 讲义

---

*最后更新：2026-06-15*

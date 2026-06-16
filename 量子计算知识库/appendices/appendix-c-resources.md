# 附录C：量子计算资源推荐

> 量子计算学习资源推荐，按类型和难度分级。

---

## 目录

1. [必读书籍](#1-必读书籍)
2. [在线课程](#2-在线课程)
3. [开源框架与工具](#3-开源框架与工具)
4. [经典论文](#4-经典论文)
5. [学术机构与会议](#5-学术机构与会议)
6. [社区与博客](#6-社区与博客)

---

## 1. 必读书籍

### 1.1 入门级 ⭐

| 书名 | 作者 | 推荐指数 | 简评 |
|------|------|---------|------|
| Quantum Computing for Computer Scientists | Yanofsky & Mannucci | ⭐⭐⭐⭐ | 计算机科学视角，数学要求低 |
| Quantum Computing: A Gentle Introduction | Rieffel & Polak | ⭐⭐⭐⭐ | 循序渐进，适合自学 |
| Dancing with Qubits | Robert Sutor | ⭐⭐⭐⭐ | IBM出品，实践导向 |
| Programming Quantum Computers | Johnston et al. | ⭐⭐⭐⭐ | 代码驱动学习 |

### 1.2 进阶级 ⭐⭐

| 书名 | 作者 | 推荐指数 | 简评 |
|------|------|---------|------|
| Quantum Computing for Everyone | Chris Bernhardt | ⭐⭐⭐⭐ | 数学基础扎实 |
| Quantum Computing: An Applied Approach | Jack Hidary | ⭐⭐⭐⭐ | 应用导向，含大量代码 |
| Learn Quantum Computing with Python and Qiskit | Loredo et al. | ⭐⭐⭐⭐ | 实践入门最佳选择 |

### 1.3 高级/参考 ⭐⭐⭐

| 书名 | 作者 | 推荐指数 | 简评 |
|------|------|---------|------|
| Quantum Computation and Quantum Information | Nielsen & Chuang | ⭐⭐⭐⭐⭐ | 量子计算"圣经" |
| Quantum Computing Since Democritus | Scott Aaronson | ⭐⭐⭐⭐⭐ | 深度与趣味兼备 |
| Principles of Quantum Mechanics | R. Shankar | ⭐⭐⭐⭐ | 量子力学标准教材 |

### 1.4 专业方向

| 领域 | 推荐书籍 | 作者 |
|------|---------|------|
| 量子纠错 | *Quantum Error Correction* | Lidar & Brun |
| 量子信息 | *Quantum Information Theory* | Wilde |
| 量子机器学习 | *Quantum Machine Learning* | Schuld & Petruccione |
| 量子密码学 | *Quantum Cryptography* | Pirandola et al. |

---

## 2. 在线课程

### 2.1 系统性课程

| 课程名称 | 平台 | 难度 | 备注 |
|---------|------|------|------|
| Quantum Computing for the Very Curious | Qiskit | ⭐ | 互动式入门 |
| Qiskit Summer School | IBM | ⭐⭐ | 全球知名，每年更新 |
| Quantum Machine Learning | edX (Xanadu) | ⭐⭐ | 量子ML方向 |
| 量子计算基础 | MIT OCW | ⭐⭐⭐ | 理论严谨 |

### 2.2 专项课程

- **Coursera**: *Quantum Computing* Specialization (UIUC)
- **edX**: *Quantum Cryptography* (Caltech)
- **Qiskit Textbook**: 在线免费教材 + 互动练习
- **IBM Quantum Learning**: 认证课程路径

---

## 3. 开源框架与工具

### 3.1 编程框架

| 框架 | 语言 | 开发者 | 特点 |
|------|------|-------|------|
| Qiskit | Python | IBM | 功能最全，生态最大，企业支持 |
| Cirq | Python | Google | NISQ优化，集成Google硬件 |
| PennyLane | Python | Xanadu | 量子ML，自动微分 |
| QuEST | C | Oxford | 高性能模拟器 |
| ProjectQ | Python | ETH Zurich | 可扩展编译器 |
| Braket SDK | Python | AWS | 多硬件后端 |
| Q# | .NET | Microsoft | 领域特定语言 |

### 3.2 模拟器

| 工具 | 类型 | 最大量子比特 | 特点 |
|------|------|-------------|------|
| Qiskit Aer | 本地 | 32+ | 噪声模拟支持 |
| QuEST | C库 | 40+ | 高性能 |
| qsim | Google | 40+ | 高性能 |
| Atos QLM | 硬件加速 | 40+ | 企业级 |
| IBM Quantum | 云端 | 127+ | 真实硬件访问 |

### 3.3 专业工具

| 工具 | 用途 | 开发者 |
|------|------|-------|
| Qiskit Nature | 量子化学 | IBM |
| Qiskit Finance | 金融计算 | IBM |
| Qiskit Optimization | 组合优化 | IBM |
| OpenFermion | 电子结构 | Google |
| Mitiq | 误差缓解 | Unitary Fund |
| Tket | 电路优化 | Cambridge Quantum |
| Stim | 纠错模拟 | Google |

---

## 4. 经典论文

### 4.1 里程碑论文

| 年份 | 论文 | 作者 | 意义 |
|------|------|------|------|
| 1982 | Simulating Physics with Computers | Feynman | 量子计算起源 |
| 1985 | Quantum Theory, the Church-Turing Principle... | Deutsch | 量子图灵机 |
| 1994 | Algorithms for Quantum Computation | Shor | Shor算法 |
| 1996 | A Fast Quantum Mechanical Algorithm... | Grover | Grover搜索 |
| 1995 | Scheme for Reducing Decoherence... | Shor | 量子纠错 |

### 4.2 综述文章

- *Quantum computing* (Ladd et al., 2010) — Nature ⭐⭐⭐⭐⭐
- *Quantum machine learning* (Biamonte et al., 2017) — Nature ⭐⭐⭐⭐
- *Quantum optimization* (Farhi et al., 2014) — arXiv ⭐⭐⭐⭐
- *Quantum chemistry in the age of quantum computing* (McArdle et al., 2020) — ⭐⭐⭐⭐⭐

---

## 5. 学术机构与会议

### 5.1 主要研究机构

| 机构 | 国家 | 重点方向 |
|------|------|---------|
| IBM Quantum | 美国 | 超导量子比特、Qiskit生态 |
| Google Quantum AI | 美国 | 超导、量子优势 |
| Xanadu | 加拿大 | 光量子计算 |
| D-Wave | 加拿大 | 量子退火 |
| Oxford Quantum | 英国 | 离子阱、量子网络 |
| 中国科学技术大学 | 中国 | 光量子、超导 |

### 5.2 重要会议

| 会议 | 领域 | 频率 |
|------|------|------|
| QIP (Quantum Information Processing) | 理论 | 每年 |
| TQC (Theory of Quantum Computation) | 理论 | 每年 |
| IEEE QCE / Quantum Week | 全领域 | 每年 |
| AQIS (Asian Quantum Information) | 亚洲 | 每年 |
| QCrpt | 量子密码学 | 每年 |

---

## 6. 社区与博客

### 6.1 社区

- **Quantum Computing Stack Exchange** — Q&A 社区 ⭐⭐⭐⭐⭐
- **Quantum Open Source Foundation** — 开源项目 ⭐⭐⭐⭐
- **Unitary Fund** — 量子开源资助 ⭐⭐⭐⭐
- **Qiskit Slack** — IBM 开发者社区 ⭐⭐⭐⭐

### 6.2 博客

- **Scott Aaronson's Blog (Shtetl-Optimized)** — 深度讨论 ⭐⭐⭐⭐⭐
- **Quantum Zeitgeist** — 新闻与趋势 ⭐⭐⭐
- **IBM Quantum Blog** — 企业进展 ⭐⭐⭐
- **Google Quantum AI Blog** — 前沿研究 ⭐⭐⭐⭐

### 6.3 视频资源

- **Qiskit YouTube** — 教程与讲座
- **Microsoft Quantum** — Q# 教程
- **Quantum Computing Report** — 行业分析
- **MinutePhysics / 3Blue1Brown** — 直观解释

---

*最后更新：2026-06-15*

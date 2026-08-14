<div align="center">

# 🧠 AI 学习知识库

### 从数学基础到亿级 Agent 架构 — 一站式 AI 知识体系

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Last Updated](https://img.shields.io/badge/Updated-2026--08-14-brightgreen.svg)]()
[![Docs](https://img.shields.io/badge/Docs-300%2B%20Articles-orange.svg)]()
[![Knowledge Base](https://img.shields.io/badge/Modules-9%20Knowledge%20Bases-purple.svg)]()

**系统架构师转型 AI 架构师的完整知识体系 · 紧跟 2026 年最前沿**

</div>

---

## ✨ 项目简介

这是一个**个人 AI / LLM / Agent 深度学习知识库**，覆盖从数学基础、LLM 理论、模型工程化、Agent 系统架构到亿级用户生产方案的全链路知识体系。

不是教程的堆砌，而是**经过工程实践检验的架构决策指南**——每一篇都包含选型对比、生产坑位、批判性分析。

### 为什么值得看？

- **🔥 前沿性**：覆盖 2026 年最新技术（A2A 协议、MCP v1.0、AgentScope、SGLang、联邦学习、Cell-based 架构）
- **🏗️ 系统性**：从 Transformer 原理到亿级用户架构，完整知识链路
- **⚔️ 批判性**：采用「The Fool」五模式批判性思维框架做选型决策
- **📊 实战性**：包含 10 万 / 100 万 / 1 亿三级用户规模的完整架构方案
- **🎨 可视化**：关键架构图采用 HTML 卡片式设计，Obsidian 完美渲染

---

## 📚 知识库导航

### 核心知识库

| 模块 | 定位 | 文档数 | 亮点 |
|------|------|--------|------|
| 🏛️ **[AI 架构师知识库](./AI架构师知识库/)** | 系统架构层 | 60+ | LLM 理论 → Agent 架构 → 亿级方案 |
| 💻 **[AI 应用工程师知识库](./AI应用工程师知识库/)** | 应用开发层 | 20+ | API 集成 → Prompt 工程 → Agent 实战 |
| 📐 **[AI 数学知识库](./AI数学知识库/)** | 理论基础 | 30+ | 线性代数 → 优化论 → Transformer 数学 |
| 🤖 **[大模型训练及 Agent 知识库](./大模型训练及Agent知识库/)** | 训练与 Agent | 15+ | 预训练 → 微调对齐 → 多 Agent 系统 |
| 🏗️ **[架构与云原生知识库](./架构与云原生知识库/)** | 基础设施 | 25+ | 微服务 → K8s → 服务网格 |
| ⚛️ **[量子计算知识库](./量子计算知识库/)** | 前沿探索 | 25+ | 量子算法 → 量子 ML |
| 🔢 **[计算机数学知识库](./计算机数学知识库/)** | CS 数学 | 20+ | 离散数学 → 算法 → 密码学 |

### 特色专栏

| 专栏 | 说明 |
|------|------|
| 🧪 **[AI 实战](./AI实战/)** | 单 Agent / 多 Agent / 知识库实战笔记 |
| 📂 **[AI 项目](./AI项目/)** | 多 Agent API 调用系统设计方案 |
| 🎯 **[Agent 优化](./agent-optimization/)** | Agent 效果优化知识库（Prompt / 规划 / 工具 / 评估） |
| 🎓 **[AI 时代素养](./AI时代素养知识库/)** | AI 时代的教育、职业、认知、创造力重塑 |

---

## 🔥 亮点内容速览

### Agent 系统架构完整链路（04-Agent系统架构/）

```
演进脉络 → OS 运行时 → 四层工程落地 → 组件选型 → 规模化方案 → 框架选型
   01         10          11             12        13/14/15         16
```

| 文档 | 核心内容 |
|------|---------|
| [11-生产级 Agent 四层架构](./AI架构师知识库/04-Agent系统架构/11-生产级Agent四层架构工程落地.md) | Brain / Memory / Action / Governance 四层工程落地 |
| [12-AgentOS 技术组件选型](./AI架构师知识库/04-Agent系统架构/12-AgentOS技术组件选型指南.md) | vLLM / SGLang / Qdrant / NATS / E2B / OPA 选型对比 |
| [13-十万用户级架构](./AI架构师知识库/04-Agent系统架构/13-十万用户级企业架构方案.md) | 单 Region 七层架构（含 HTML 架构图）|
| [14-百万用户级架构](./AI架构师知识库/04-Agent系统架构/14-百万用户级超级架构方案.md) | Cell-based 多地域 + 双层调度 + 联邦记忆（含 HTML 架构图）|
| [15-亿级用户终极架构](./AI架构师知识库/04-Agent系统架构/15-亿级用户终极架构方案.md) | 云 + 边三层 + 联邦学习 + GPU 产能策略（含 HTML 架构图）|
| [16-Agent 编排框架选型](./AI架构师知识库/04-Agent系统架构/16-Agent编排层框架生态选型.md) | 12+ 框架深度对比（LangGraph / CrewAI / AgentScope / Mastra / Dapr）|

### 架构方案三级跃迁

| 规模 | 架构 DNA | GPU 规模 | 月成本 |
|------|---------|---------|--------|
| **10 万用户** | 单 Region × 多 AZ 七层架构 | 8-32 × H100 | $80K-$300K |
| **100 万用户** | Cell-based 多地域 + 联邦记忆 + 双层调度 | 200-600 × H100 | $1.2M-$3.5M |
| **1 亿用户** | 云 + 边三层 + 联邦学习 + 三层控制平面 | 2 万-6 万 × H100 | $80M-$200M |

> 每次跃迁都不是线性放大，而是架构 DNA 的质变。

---

## 🗺️ 学习路径

### 按背景选择

```
后端/微服务转 AI:
  数学基础 → LLM 理论 → 模型工程化 → AI 基础设施 → RAG → Agent → LLMOps → 架构案例

算法/ML 转架构:
  LLM 理论 → 模型工程化 → RAG → AI 基础设施 → Agent → LLMOps → 架构模式

全栈/创业/产品转 AI:
  RAG → Agent → 架构案例 → 框架生态 → LLM 基础 → 架构模式 → 模型工程化
```

### 6 个月规划

| 阶段 | 模块 | 目标 | 周期 |
|------|------|------|------|
| **基础夯实** | LLM 基础 → RAG 架构 | Transformer / 推理模型 / RAG 核心 | 第 1-2 月 |
| **架构实战** | 模型工程化 → Agent 系统 | 部署 / 量化 / Agent 设计 | 第 3-4 月 |
| **生产级能力** | AI 基础设施 → LLMOps | GPU 调度 / SRE / FinOps / 合规 | 第 5-6 月 |
| **贯穿全程** | 架构案例 → 架构模式 | 案例验证模式，模式指导案例 | 持续 |

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **文档** | Markdown 95%+（Obsidian Flavored）|
| **可视化** | HTML 卡片式架构图 / Mermaid / matplotlib |
| **代码示例** | Python / TypeScript / Java / Go |
| **架构图** | 内联 CSS + Flexbox，无 JS 依赖 |
| **知识管理** | Obsidian（wikilink + callout + embed）|

---

## 📊 项目统计

<div align="center">

| 指标 | 数量 |
|------|------|
| 知识库模块 | 9 个 |
| Markdown 文档 | 300+ 篇 |
| 架构案例 | 8 个生产级案例 |
| Agent 架构文档 | 16 篇深度文档 |
| 用户规模方案 | 3 级（10 万 / 100 万 / 1 亿）|
| 框架对比覆盖 | 12+ 主流框架 |

</div>

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/sss534534/learning-AI.git
cd learning-AI
```

### 2. 用 Obsidian 打开（推荐）

1. 安装 [Obsidian](https://obsidian.md/)
2. 打开仓库根目录作为 Vault
3. 享受 wikilink + callout + HTML 架构图的完美渲染

### 3. 直接浏览

任何 Markdown 编辑器或 GitHub 网页端均可直接阅读。

---

## 📝 更新日志

- **2026-08-14**: 🔥 Agent 系统架构知识体系完整构建
  - 新增 8 篇深度架构文档（11-16 号 + 框架生态 05/06）
  - 覆盖四层架构 → 组件选型 → 10 万/100 万/1 亿三级规模化方案
  - 12+ Agent 编排框架深度对比（LangGraph / CrewAI / AgentScope / Mastra / Dapr）
  - 关键架构图升级为 HTML 卡片式设计
- **2026-06-16**: 知识库结构优化，新增 09-Agent 框架生态和 10-AI 工程化前沿
- **2026-05-29**: P0/P1 缺口补齐，8.5→9.5 冲刺
- **2026-05-12**: 前沿技术大升级（推理模型 / SGLang / GraphRAG / MCP）
- **2026-05-07**: 知识库初始版本创建

---

## 📄 License

本项目采用 [MIT License](LICENSE)。

知识内容欢迎引用，请注明出处。

---

<div align="center">

**⭐ 如果这个知识库对你有帮助，欢迎 Star**

*持续更新 · 紧跟 AI 技术前沿 · 从理论到生产*

</div>

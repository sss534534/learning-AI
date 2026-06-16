# 附录A：数学符号速查表

> 整理计算机数学中常用的数学符号，方便快速查阅。

## 元数据

- **难度**: ⭐
- **前置知识**: 所有章节
- **关联文件**: [../index.md](../index.md)
- **最后更新**: 2026-06-12

---

## 1. 逻辑符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| ¬ | 否定/非 | not | ¬P |
| ∧ | 合取/且 | and | P ∧ Q |
| ∨ | 析取/或 | or | P ∨ Q |
| → | 蕴含 | if...then | P → Q |
| ↔ | 等价/当且仅当 | iff | P ↔ Q |
| ∀ | 全称量词 | 对所有 | ∀x, P(x) |
| ∃ | 存在量词 | 存在 | ∃x, P(x) |
| ∴ | 所以/因此 | therefore | ∴ Q |
| ⇒ | 蕴含 | implies | A ⇒ B |
| ⇔ | 等价 | iff | A ⇔ B |

---

## 2. 集合论符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| ∈ | 属于 | is an element of | x ∈ A |
| ∉ | 不属于 | is not an element of | x ∉ A |
| ⊆ | 子集 | subset | A ⊆ B |
| ⊂ | 真子集 | proper subset | A ⊂ B |
| ∪ | 并集 | union | A ∪ B |
| ∩ | 交集 | intersection | A ∩ B |
| \ 或 - | 差集 | set difference | A \ B |
| × | 笛卡尔积 | Cartesian product | A × B |
| ∅ | 空集 | empty set | ∅ |
| 𝒫(A) 或 2^A | 幂集 | power set | 𝒫(A) |
| |A| | 基数 | cardinality | |A| |
| ℕ | 自然数集 | natural numbers | ℕ = {0, 1, 2, ...} |
| ℤ | 整数集 | integers | ℤ = {..., -1, 0, 1, ...} |
| ℚ | 有理数集 | rational numbers | |
| ℝ | 实数集 | real numbers | |

---

## 3. 关系与函数

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| f: A→B | 函数 | function from A to B | f: ℕ→ℕ |
| f(x) | 函数值 | value of f at x | f(x) = x² |
| ∘ | 复合 | composition | (f∘g)(x) = f(g(x)) |
| ⌈x⌉ | 上取整 | ceiling | ⌈2.3⌉ = 3 |
| ⌊x⌋ | 下取整 | floor | ⌊2.3⌋ = 2 |
| x mod y | 模运算 | modulo | 7 mod 3 = 1 |

---

## 4. 组合数学符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| n! | 阶乘 | factorial | 5! = 120 |
| C(n,k) 或 ⎛n⎞ | 组合数 | binomial coefficient | C(5,2)=10 |
| | | | ⎝k⎠ |
| P(n,k) | 排列数 | permutation | P(5,2)=20 |
| Σ | 求和 | summation | Σ_{i=1}^n i |
| Π | 求积 | product | Π_{i=1}^n i |

---

## 5. 线性代数符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| 𝐯 或 v | 向量 | vector | 𝐯 ∈ ℝⁿ |
| A 或 [a_ij] | 矩阵 | matrix | A ∈ ℝ^{m×n} |
| A^T 或 A' | 转置 | transpose | (A^T)_{ij}=A_{ji} |
| A^{-1} | 逆矩阵 | inverse | AA^{-1}=I |
| det(A) 或 |A| | 行列式 | determinant |
| tr(A) | 迹 | trace | tr(A)=Σa_{ii} |
| 𝐯·𝐰 或 ⟨𝐯,𝐰⟩ | 内积/点积 | dot product |
| 𝐯×𝐰 | 叉积 | cross product | (3D向量) |
| ||𝐯|| | 范数 | norm | ||𝐯||₂ |
| 0 或 O | 零向量/零矩阵 | zero vector/matrix |
| I 或 I_n | 单位矩阵 | identity matrix | I_n ∈ ℝ^{n×n} |
| λ | 特征值 | eigenvalue | A𝐯=λ𝐯 |
| 𝐯 | 特征向量 | eigenvector |

---

## 6. 微积分符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| f'(x) 或 df/dx | 导数 | derivative |
| f''(x) 或 d²f/dx² | 二阶导数 | second derivative |
| ∂f/∂x | 偏导数 | partial derivative |
| ∇f | 梯度 | gradient | ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ) |
| ∫f(x)dx | 不定积分 | indefinite integral |
| ∫_a^b f(x)dx | 定积分 | definite integral |
| lim_{x→a} f(x) | 极限 | limit |
| ≈ | 约等于 | approximately equal | π ≈ 3.1416 |

---

## 7. 算法复杂度符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| O(g(n)) | 大O | 上界 | f(n) = O(n²) |
| Ω(g(n)) | 大Omega | 下界 | f(n) = Ω(n) |
| Θ(g(n)) | 大Theta | 紧确界 | f(n) = Θ(n log n) |
| o(g(n)) | 小o | 非紧上界 | f(n) = o(n²) |
| ω(g(n)) | 小omega | 非紧下界 | f(n) = ω(n) |

---

## 8. 数论符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| a | b | a整除b | 3 | 6 |
| gcd(a,b) | 最大公约数 | greatest common divisor | gcd(12,18)=6 |
| lcm(a,b) | 最小公倍数 | least common multiple | lcm(12,18)=36 |
| a ≡ b mod m | 同余 | congruent modulo m | 7 ≡ 1 mod 3 |
| φ(n) | 欧拉函数 | Euler's totient function | φ(6)=2 |

---

## 9. 概率与统计符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| P(A) | 概率 | probability of A |
| P(A\|B) | 条件概率 | conditional probability |
| E[X] 或 μ | 期望 | expectation |
| Var(X) 或 σ² | 方差 | variance |
| X ~ D | 服从分布 | X has distribution D |
| N(μ,σ²) | 正态分布 | normal distribution |
| Cov(X,Y) | 协方差 | covariance |
| Corr(X,Y) | 相关系数 | correlation coefficient |

---

## 10. 其他常用符号

| 符号 | 名称 | 含义 | 示例 |
|-----|------|-----|------|
| = | 等于 | equal | a = b |
| ≠ | 不等于 | not equal | a ≠ b |
| < | 小于 | less than | a < b |
| > | 大于 | greater than | a > b |
| ≤ | 小于等于 | less than or equal | a ≤ b |
| ≥ | 大于等于 | greater than or equal | a ≥ b |
| := 或 ≡ | 定义为 | is defined as | x := 5 |
| ∞ | 无穷大 | infinity |
| ∋ | 使得 | such that | ∋ x > 0 |
| ∵ | 因为 | because | ∵ x > 0 |
| | | 使得/条件 | {x | x > 0} |

*最后更新：2026-06-12*


# 第七章：集合与关系

> 集合论是现代数学的基础，关系理论是理解数据结构和算法的关键。本章深入探讨集合操作、关系类型及其在计算机科学中的应用。

---

## 目录

1. [集合基础](#1-集合基础)
2. [集合操作](#2-集合操作)
3. [关系与函数](#3-关系与函数)
4. [等价关系](#4-等价关系)
5. [偏序关系](#5-偏序关系)
6. [格与布尔代数](#6-格与布尔代数)

---

## 1. 集合基础

### 1.1 集合的定义

```
集合是由确定的、互不相同的对象组成的整体。
- 元素：集合中的对象
- 表示：A = {a, b, c} 或 A = {x | P(x)}（集合构造器）
- 基数：|A| 表示集合A的元素个数
```

### 1.2 特殊集合

```
- 空集：∅ = {}
- 全集：包含所有讨论对象的集合，记为U
- 自然数集：N = {0, 1, 2, ...}
- 整数集：Z = {..., -2, -1, 0, 1, 2, ...}
- 有理数集：Q
- 实数集：R
```

### 1.3 集合的性质

```python
def is_subset(A, B):
    """判断A是否是B的子集"""
    return all(x in B for x in A)

def is_proper_subset(A, B):
    """判断A是否是B的真子集"""
    return is_subset(A, B) and A != B

def set_equality(A, B):
    """判断两个集合是否相等"""
    return is_subset(A, B) and is_subset(B, A)
```

---

## 2. 集合操作

### 2.1 基本操作

```
并集（Union）：A ∪ B = {x | x ∈ A 或 x ∈ B}
交集（Intersection）：A ∩ B = {x | x ∈ A 且 x ∈ B}
差集（Difference）：A - B = {x | x ∈ A 且 x ∉ B}
补集（Complement）：A' = U - A = {x | x ∉ A}
对称差（Symmetric Difference）：A △ B = (A - B) ∪ (B - A)
```

### 2.2 Python实现

```python
def union(A, B):
    return A | B

def intersection(A, B):
    return A & B

def difference(A, B):
    return A - B

def symmetric_difference(A, B):
    return A ^ B

def complement(A, universe):
    return universe - A
```

### 2.3 集合恒等式

```
交换律：
A ∪ B = B ∪ A
A ∩ B = B ∩ A

结合律：
(A ∪ B) ∪ C = A ∪ (B ∪ C)
(A ∩ B) ∩ C = A ∩ (B ∩ C)

分配律：
A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)

德摩根定律：
(A ∪ B)' = A' ∩ B'
(A ∩ B)' = A' ∪ B'
```

### 2.4 幂集

```python
def power_set(S):
    """计算集合的幂集"""
    if not S:
        return [set()]
    element = S.pop()
    subsets = power_set(S)
    return subsets + [subset | {element} for subset in subsets]
```

### 2.5 笛卡尔积

```python
def cartesian_product(A, B):
    """计算笛卡尔积 A × B"""
    return {(a, b) for a in A for b in B}
```

---

## 3. 关系与函数

### 3.1 关系的定义

```
设A和B是集合，A × B的子集R称为从A到B的二元关系。
- 如果A = B，则称R为A上的关系
- 若(x, y) ∈ R，记为xRy
```

### 3.2 关系的表示

```python
# 关系可以用集合、矩阵或图表示

def relation_matrix(R, A, B):
    """生成关系矩阵"""
    matrix = []
    for a in A:
        row = []
        for b in B:
            row.append(1 if (a, b) in R else 0)
        matrix.append(row)
    return matrix
```

### 3.3 关系的性质

```
自反性（Reflexive）：∀x ∈ A, (x, x) ∈ R
反自反性（Irreflexive）：∀x ∈ A, (x, x) ∉ R
对称性（Symmetric）：∀x, y ∈ A, 若xRy则yRx
反对称性（Antisymmetric）：∀x, y ∈ A, 若xRy且yRx则x = y
传递性（Transitive）：∀x, y, z ∈ A, 若xRy且yRz则xRz
```

### 3.4 函数的定义

```
设f是从A到B的关系，如果对每个a ∈ A，存在唯一的b ∈ B使得(a, b) ∈ f，则f是函数。
- 定义域：dom(f) = A
- 值域：ran(f) = {b ∈ B | ∃a ∈ A, f(a) = b}
```

### 3.5 特殊函数类型

```
单射（Injective）：∀a1, a2 ∈ A, 若f(a1) = f(a2)则a1 = a2
满射（Surjective）：ran(f) = B
双射（Bijective）：既是单射又是满射
```

---

## 4. 等价关系

### 4.1 等价关系的定义

```
关系R是等价关系当且仅当R是自反的、对称的、传递的。
```

### 4.2 等价类

```
设R是A上的等价关系，对于a ∈ A，集合
[a]_R = {x ∈ A | xRa}
称为a关于R的等价类。

性质：
1. a ∈ [a]_R
2. 若aRb，则[a]_R = [b]_R
3. 任意两个等价类要么相等，要么不相交
```

### 4.3 划分

```
集合A的划分是A的非空子集族{Ai}，满足：
1. ∪Ai = A
2. 若i ≠ j，则Ai ∩ Aj = ∅

定理：A上的等价关系R确定A的一个划分，反之亦然。
```

### 4.4 应用示例

```python
def find_equivalence_classes(A, R):
    """找出所有等价类"""
    classes = []
    visited = set()
    for a in A:
        if a not in visited:
            cls = {x for x in A if (a, x) in R}
            classes.append(cls)
            visited.update(cls)
    return classes
```

---

## 5. 偏序关系

### 5.1 偏序的定义

```
关系R是偏序关系当且仅当R是自反的、反对称的、传递的。
- 记为 ≤，称(A, ≤)为偏序集
```

### 5.2 哈斯图（Hasse Diagram）

```
哈斯图是偏序集的图形表示：
1. 用顶点表示元素
2. 若x ≤ y且不存在z使得x ≤ z ≤ y，则从x到y画一条向上的边
3. 省略自反边和传递边
```

### 5.3 特殊元素

```
极大元（Maximal Element）：∀y ∈ A, 若x ≤ y则x = y
极小元（Minimal Element）：∀y ∈ A, 若y ≤ x则x = y
最大元（Greatest Element）：∀y ∈ A, y ≤ x
最小元（Least Element）：∀y ∈ A, x ≤ y
```

### 5.4 上界与下界

```
上界（Upper Bound）：∀a ∈ S, a ≤ u
下界（Lower Bound）：∀a ∈ S, l ≤ a
最小上界（LUB）：S的所有上界中的最小元
最大下界（GLB）：S的所有下界中的最大元
```

---

## 6. 格与布尔代数

### 6.1 格的定义

```
偏序集(L, ≤)是格，如果L中任意两个元素都有最小上界和最大下界。
- 记a ∨ b = LUB(a, b)（并运算）
- 记a ∧ b = GLB(a, b)（交运算）
```

### 6.2 格的性质

```
交换律：a ∨ b = b ∨ a，a ∧ b = b ∧ a
结合律：(a ∨ b) ∨ c = a ∨ (b ∨ c)
吸收律：a ∨ (a ∧ b) = a，a ∧ (a ∨ b) = a
幂等律：a ∨ a = a，a ∧ a = a
```

### 6.3 布尔代数

```
布尔代数是一个有补分配格(B, ∨, ∧, ', 0, 1)，满足：
1. 有界性：存在0和1，∀a ∈ B, 0 ≤ a ≤ 1
2. 分配性：a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c)
3. 有补性：∀a ∈ B, 存在a'使得a ∨ a' = 1且a ∧ a' = 0
```

### 6.4 布尔代数的性质

```python
class BooleanAlgebra:
    def __init__(self, elements):
        self.elements = elements
    
    def meet(self, a, b):
        """交运算"""
        return a & b
    
    def join(self, a, b):
        """并运算"""
        return a | b
    
    def complement(self, a):
        """补运算"""
        return ~a & 0xFF  # 假设8位布尔代数
```

### 6.5 布尔表达式与电路

```
布尔代数在数字电路中的应用：
- AND门：a ∧ b
- OR门：a ∨ b
- NOT门：a'
- XOR门：a ⊕ b = (a ∧ b') ∨ (a' ∧ b)
```

---

## 本章小结

集合与关系是数据结构的数学基础：
1. 集合操作是数据库查询和算法设计的基础
2. 等价关系用于分类和分组
3. 偏序关系用于排序和优先级处理
4. 格与布尔代数是计算机逻辑和数字电路的核心

这些概念在算法设计、数据库系统、编译器设计等领域都有广泛应用。
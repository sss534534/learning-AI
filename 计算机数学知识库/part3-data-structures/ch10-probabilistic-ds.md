# 第十章：概率数据结构

> 概率数据结构是一类基于概率理论的数据结构，通过牺牲一定的准确性来换取空间和时间效率的大幅提升。本章深入探讨各类概率数据结构的原理与应用。

---

## 目录

1. [布隆过滤器](#1-布隆过滤器)
2. [Count-Min Sketch](#2-count-min-sketch)
3. [HyperLogLog](#3-hyperloglog)
4. [跳表](#4-跳表)
5. [Treap](#5-treap)
6. [概率数据结构对比](#6-概率数据结构对比)

---

## 1. 布隆过滤器

### 1.1 原理

```
布隆过滤器是一种空间效率极高的概率数据结构，用于判断元素是否在集合中。

工作原理：
1. 使用k个独立的哈希函数
2. 将元素哈希到m位的位数组中
3. 查询时检查所有k个位置是否都为1

特点：
- 假阳性（False Positive）：可能误判元素存在
- 假阴性（False Negative）：永远不会误判元素不存在
```

### 1.2 数学分析

```
假阳性概率：
ε ≈ (1 - e^(-kn/m))^k

最优哈希函数数量：
k = (m/n) × ln2

空间复杂度：
m = -(n × lnε) / (ln2)^2

其中：
- n：预计元素数量
- m：位数组大小
- k：哈希函数数量
- ε：可接受的假阳性率
```

### 1.3 Python实现

```python
import math

class BloomFilter:
    def __init__(self, expected_items, false_positive_rate=0.01):
        self.n = expected_items
        self.epsilon = false_positive_rate
        self.m = self._calculate_size()
        self.k = self._calculate_hash_count()
        self.bit_array = [0] * self.m
    
    def _calculate_size(self):
        """计算最佳位数组大小"""
        return int(-(self.n * math.log(self.epsilon)) / (math.log(2) ** 2))
    
    def _calculate_hash_count(self):
        """计算最佳哈希函数数量"""
        return int((self.m / self.n) * math.log(2))
    
    def _hash(self, item, i):
        """第i个哈希函数"""
        hash_val = hash(str(item) + str(i))
        return abs(hash_val) % self.m
    
    def add(self, item):
        """添加元素"""
        for i in range(self.k):
            idx = self._hash(item, i)
            self.bit_array[idx] = 1
    
    def contains(self, item):
        """判断元素是否可能存在"""
        for i in range(self.k):
            idx = self._hash(item, i)
            if self.bit_array[idx] == 0:
                return False
        return True
```

---

## 2. Count-Min Sketch

### 2.1 原理

```
Count-Min Sketch是一种用于频率估计的概率数据结构。

工作原理：
1. 使用d个哈希表，每个哈希表有w个桶
2. 插入时，每个哈希函数将元素映射到对应哈希表的桶中并计数+1
3. 查询时，取所有哈希表中对应桶的最小值作为估计值

特点：
- 空间复杂度：O(d × w)
- 频率估计可能偏高（保守估计）
- 支持增量更新
```

### 2.2 数学分析

```
误差保证：
对于任意元素i，估计频率f̂(i)满足：
f̂(i) ≥ f(i)
Pr[f̂(i) - f(i) > εF] ≤ δ

参数选择：
w = ⌈e/ε⌉
d = ⌈ln(1/δ)⌉

其中：
- ε：相对误差
- δ：置信度
- F：总频率
```

### 2.3 Python实现

```python
import math
import hashlib

class CountMinSketch:
    def __init__(self, epsilon=0.01, delta=0.01):
        self.epsilon = epsilon
        self.delta = delta
        self.w = int(math.ceil(math.e / epsilon))
        self.depth = int(math.ceil(math.log(1 / delta)))
        self.table = [[0] * self.w for _ in range(self.depth)]
    
    def _hash(self, item, i):
        """第i个哈希函数"""
        h = hashlib.md5(f"{item}{i}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.w
    
    def update(self, item, count=1):
        """增加元素频率"""
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i][idx] += count
    
    def query(self, item):
        """查询元素频率"""
        min_count = float('inf')
        for i in range(self.depth):
            idx = self._hash(item, i)
            min_count = min(min_count, self.table[i][idx])
        return min_count
```

### 2.4 应用场景

```
- 数据流中的频率统计
- 热门元素追踪
- 网络流量监控
- 数据库查询优化
```

---

## 3. HyperLogLog

### 3.1 原理

```
HyperLogLog是一种用于基数估计的概率数据结构，能够以极低的空间复杂度估算集合的唯一元素数量。

核心思想：
1. 使用哈希函数将元素映射到均匀分布的范围
2. 观察哈希值前导零的数量
3. 利用前导零的最大数量来估计基数
```

### 3.2 数学分析

```
基数估计公式：
E = α_m × m × 2^R

其中：
- α_m：修正系数，取决于m
- m：桶的数量
- R：所有桶中最大的前导零数量

空间复杂度：O(log log n)
相对标准误差：σ ≈ 1.04 / √m
```

### 3.3 Python实现

```python
import math
import hashlib

class HyperLogLog:
    def __init__(self, relative_error=0.01):
        self.relative_error = relative_error
        self.m = self._calculate_buckets()
        self.alpha = self._calculate_alpha()
        self.buckets = [0] * self.m
    
    def _calculate_buckets(self):
        """计算桶的数量"""
        return 1 << (int(math.ceil(math.log((1.04 / self.relative_error) ** 2, 2))))
    
    def _calculate_alpha(self):
        """计算修正系数"""
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)
    
    def _hash(self, item):
        """哈希函数"""
        h = hashlib.sha256(str(item).encode()).digest()
        return int.from_bytes(h, 'big')
    
    def _count_leading_zeros(self, x):
        """计算前导零数量"""
        if x == 0:
            return 32
        return x.bit_length() - 1
    
    def add(self, item):
        """添加元素"""
        h = self._hash(item)
        idx = h & (self.m - 1)
        remaining = h >> self.m.bit_length()
        leading_zeros = self._count_leading_zeros(remaining)
        if leading_zeros > self.buckets[idx]:
            self.buckets[idx] = leading_zeros
    
    def estimate(self):
        """估算基数"""
        z = sum(2 ** (-b) for b in self.buckets)
        E = self.alpha * self.m * self.m / z
        
        # 小基数修正
        if E <= 5 * self.m / 2:
            v = sum(1 for b in self.buckets if b == 0)
            if v != 0:
                return self.m * math.log(self.m / v)
        
        # 大基数修正
        if E > (1 << 32) / 30:
            return -(1 << 32) * math.log(1 - E / (1 << 32))
        
        return E
```

### 3.4 应用场景

```
- 网站UV统计
- 数据库distinct count优化
- 网络流中的唯一元素计数
- 分布式系统中的基数合并
```

---

## 4. 跳表

### 4.1 原理

```
跳表（Skip List）是一种基于概率的数据结构，提供O(log n)的查找、插入和删除操作。

核心思想：
1. 在有序链表上建立多层索引
2. 第i层的节点以概率p出现在第i+1层
3. 查找时从顶层开始快速定位

特点：
- 平均O(log n)时间复杂度
- 无需复杂的平衡操作
- 实现简单
```

### 4.2 数学分析

```
期望高度：H ≈ log_{1/p}(n)

期望节点数：
每个元素期望出现在1/(1-p)层
总节点数 ≈ n / (1-p)

时间复杂度：
- 查找：O(log n)
- 插入：O(log n)
- 删除：O(log n)
```

### 4.3 Python实现

```python
import random

class SkipListNode:
    def __init__(self, value, level):
        self.value = value
        self.next = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.head = SkipListNode(-float('inf'), max_level)
        self.level = 0
    
    def _random_level(self):
        """随机生成节点层数"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def insert(self, value):
        """插入元素"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        # 从顶层开始查找
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                current = current.next[i]
            update[i] = current
        
        # 生成随机层数
        new_level = self._random_level()
        
        # 更新跳表高度
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level
        
        # 创建新节点
        new_node = SkipListNode(value, new_level)
        
        # 插入节点
        for i in range(new_level + 1):
            new_node.next[i] = update[i].next[i]
            update[i].next[i] = new_node
    
    def search(self, value):
        """查找元素"""
        current = self.head
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                current = current.next[i]
        current = current.next[0]
        return current and current.value == value
    
    def delete(self, value):
        """删除元素"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].value < value:
                current = current.next[i]
            update[i] = current
        
        current = current.next[0]
        
        if current and current.value == value:
            for i in range(self.level + 1):
                if update[i].next[i] != current:
                    break
                update[i].next[i] = current.next[i]
            
            # 更新跳表高度
            while self.level > 0 and not self.head.next[self.level]:
                self.level -= 1
```

---

## 5. Treap

### 5.1 原理

```
Treap（Tree + Heap）是一种结合了二叉搜索树和堆性质的数据结构。

核心思想：
1. 每个节点包含关键字和随机优先级
2. 按关键字满足BST性质
3. 按优先级满足堆性质（最大堆或最小堆）

特点：
- 期望O(log n)时间复杂度
- 通过旋转维护平衡
- 随机化保证期望性能
```

### 5.2 Python实现

```python
import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.randint(0, 1000000)
        self.left = None
        self.right = None

class Treap:
    def __init__(self):
        self.root = None
    
    def _rotate_right(self, y):
        """右旋"""
        x = y.left
        T2 = x.right
        
        x.right = y
        y.left = T2
        
        return x
    
    def _rotate_left(self, x):
        """左旋"""
        y = x.right
        T2 = y.left
        
        y.left = x
        x.right = T2
        
        return y
    
    def insert(self, key):
        """插入节点"""
        self.root = self._insert(self.root, key)
    
    def _insert(self, node, key):
        if not node:
            return TreapNode(key)
        
        if key < node.key:
            node.left = self._insert(node.left, key)
            if node.left.priority > node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert(node.right, key)
            if node.right.priority > node.priority:
                node = self._rotate_left(node)
        
        return node
    
    def search(self, key):
        """查找节点"""
        return self._search(self.root, key)
    
    def _search(self, node, key):
        if not node:
            return False
        if key == node.key:
            return True
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)
    
    def delete(self, key):
        """删除节点"""
        self.root = self._delete(self.root, key)
    
    def _delete(self, node, key):
        if not node:
            return node
        
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # 找到要删除的节点
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # 比较子节点优先级
            if node.left.priority > node.right.priority:
                node = self._rotate_right(node)
                node.right = self._delete(node.right, key)
            else:
                node = self._rotate_left(node)
                node.left = self._delete(node.left, key)
        
        return node
```

---

## 6. 概率数据结构对比

### 6.1 时间空间复杂度对比

```
| 数据结构 | 空间复杂度 | 查询时间 | 插入时间 | 特点 |
|---------|-----------|---------|---------|------|
| 布隆过滤器 | O(m) | O(k) | O(k) | 存在性检测，有假阳性 |
| Count-Min Sketch | O(d×w) | O(d) | O(d) | 频率估计，保守估计 |
| HyperLogLog | O(log log n) | O(1) | O(1) | 基数估计，高精度 |
| 跳表 | O(n) | O(log n) | O(log n) | 有序集合，可范围查询 |
| Treap | O(n) | O(log n) | O(log n) | 有序集合，随机化平衡 |
```

### 6.2 应用场景总结

```
- 存在性检测：布隆过滤器
- 频率统计：Count-Min Sketch
- 基数估计：HyperLogLog
- 有序集合：跳表、Treap
- 大数据流处理：所有概率数据结构
```

---

## 本章小结

概率数据结构是处理大规模数据的利器：
1. 布隆过滤器用于快速存在性检测
2. Count-Min Sketch用于频率估计
3. HyperLogLog用于基数估计
4. 跳表和Treap是随机化的有序集合

这些数据结构在大数据、分布式系统、网络流量监控等领域有广泛应用，通过牺牲少量准确性换取巨大的空间和时间效率提升。
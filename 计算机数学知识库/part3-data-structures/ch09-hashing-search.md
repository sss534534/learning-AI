# 第九章：哈希与查找

> 哈希表是一种高效的数据结构，通过哈希函数将键映射到值，支持平均O(1)时间的查找、插入和删除。

## 元数据

- **难度**: ⭐⭐
- **前置知识**: ch04-ch06
- **关联文件**: [ch08-trees-heaps.md](ch08-trees-heaps.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [哈希函数](#1-哈希函数)
2. [哈希冲突解决](#2-哈希冲突解决)
3. [哈希表实现](#3-哈希表实现)
4. [布隆过滤器](#4-布隆过滤器)
5. [查找算法](#5-查找算法)
6. [平衡树与哈希表的比较](#6-平衡树与哈希表的比较)

---

## 1. 哈希函数

### 1.1 哈希函数的性质
```
1. 一致性：相同的输入总是产生相同的输出
2. 高效性：计算要快
3. 均匀分布：输出应均匀分布在哈希空间中
4. 雪崩效应：输入的微小变化会导致输出的显著变化
```

### 1.2 常见哈希函数

#### 除法散列法
```
h(k) = k mod m
选择m为素数，且不接近2的幂。
```

#### 乘法散列法
```
h(k) = floor(m × (k × A mod 1))
A是一个在(0,1)之间的常数，Knuth建议A ≈ (√5 - 1)/2 ≈ 0.618。
```

#### 全域哈希
```
从一组哈希函数中随机选择一个，
使得对于任意键k ≠ l，概率Pr[h(k) = h(l)] ≤ 1/m。
```

### 1.3 Python实现
```python
import math

def hash_division(k, m):
    """除法散列法"""
    return k % m

def hash_multiplication(k, m, A=None):
    """乘法散列法"""
    if A is None:
        A = (math.sqrt(5) - 1) / 2  # Knuth建议的常数
    return int(m * ((k * A) % 1))
```

### 1.4 字符串哈希
```python
def hash_string(s, m, base=31):
    """字符串哈希（多项式滚动哈希）"""
    hash_val = 0
    for c in s:
        hash_val = (hash_val * base + ord(c)) % m
    return hash_val
```

---

## 2. 哈希冲突解决

### 2.1 链地址法（Separate Chaining）
```
每个哈希桶维护一个链表，
冲突的元素插入到对应的链表中。

优点：
- 实现简单
- 删除操作方便
- 不需要知道元素数量

缺点：
- 额外的链表开销
- 可能导致链表过长
```

### 2.2 开放寻址法（Open Addressing）

#### 线性探测（Linear Probing）
```
h(k, i) = (h'(k) + i) mod m
i = 0, 1, ..., m-1

问题：一次群集（Primary Clustering）
```

#### 二次探测（Quadratic Probing）
```
h(k, i) = (h'(k) + c1*i + c2*i²) mod m
常使用c1 = c2 = 0.5，或c1 = 1, c2 = 3等。

问题：二次群集（Secondary Clustering）
```

#### 双重哈希（Double Hashing）
```
h(k, i) = (h1(k) + i * h2(k)) mod m
h2(k)必须与m互质，常选择h2(k) = 1 + (k mod (m-1))。
```

---

## 3. 哈希表实现

### 3.1 链地址法实现
```python
class HashTableChaining:
    """链地址法哈希表"""
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        """插入键值对"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, value)
                return
        self.table[idx].append((key, value))
    
    def get(self, key):
        """查找键"""
        idx = self._hash(key)
        for k, v in self.table[idx]:
            if k == key:
                return v
        return None
    
    def remove(self, key):
        """删除键"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                del self.table[idx][i]
                return
```

### 3.2 线性探测法实现
```python
class HashTableLinearProbing:
    """线性探测法哈希表"""
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        """插入键值对"""
        idx = self._hash(key)
        while self.keys[idx] is not None and self.keys[idx] != key:
            idx = (idx + 1) % self.size
        self.keys[idx] = key
        self.values[idx] = value
    
    def get(self, key):
        """查找键"""
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.size
        return None
```

---

## 4. 布隆过滤器

### 4.1 布隆过滤器原理
```
布隆过滤器是一种空间效率很高的数据结构，
用于判断一个元素是否在集合中。

特点：
- 可能存在假阳性（False Positive）
- 不会出现假阴性（False Negative）
- 空间效率很高
```

### 4.2 数学分析
```
假阳性概率：
ε ≈ (1 - e^(-kn/m))^k

最优k选择：
k = (m/n) × ln2

其中：
- m：位数组大小
- n：元素数量
- k：哈希函数数量
```

### 4.3 Python实现
```python
import bitarray
import math

class BloomFilter:
    """布隆过滤器"""
    def __init__(self, expected_items, false_positive_rate=0.01):
        self.m = self._calculate_size(expected_items, false_positive_rate)
        self.k = self._calculate_hash_functions(self.m, expected_items)
        self.bit_array = bitarray.bitarray(self.m)
        self.bit_array.setall(0)
    
    def _calculate_size(self, n, p):
        """计算最佳位数组大小"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    def _calculate_hash_functions(self, m, n):
        """计算最佳哈希函数数量"""
        k = (m / n) * math.log(2)
        return int(k)
    
    def _hash(self, item, i):
        """第i个哈希函数"""
        return hash((item, i)) % self.m
    
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

## 5. 查找算法

### 5.1 顺序查找
```python
def linear_search(arr, target):
    """顺序查找"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

### 5.2 二分查找
```python
def binary_search(arr, target):
    """二分查找（迭代版）"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    """二分查找（递归版）"""
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### 5.3 插值查找
```python
def interpolation_search(arr, target):
    """插值查找"""
    left, right = 0, len(arr) - 1
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            return left if arr[left] == target else -1
        # 插值公式
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    return -1
```

---

## 6. 平衡树与哈希表的比较

### 6.1 时间复杂度比较
```
| 操作       | 哈希表（平均） | 哈希表（最坏） | 平衡树 |
|------------|---------------|---------------|--------|
| 查找       | O(1)          | O(n)          | O(log n) |
| 插入       | O(1)          | O(n)          | O(log n) |
| 删除       | O(1)          | O(n)          | O(log n) |
```

### 6.2 优缺点比较
```
哈希表：
优点：
- 平均情况速度快
- 实现简单
- 适用于大量数据

缺点：
- 最坏情况性能差
- 不支持顺序遍历
- 空间可能有浪费
- 需要好的哈希函数

平衡树（如AVL、红黑树）：
优点：
- 最坏情况性能好（O(log n)）
- 支持顺序遍历
- 可以维护顺序关系

缺点：
- 平均情况比哈希表慢
- 实现复杂
- 树平衡的开销
```

---

## 本章小结

哈希与查找是数据结构中的重要内容，本章要点：
1. 哈希函数的设计与性质
2. 哈希冲突的解决方法（链地址法、开放寻址法）
3. 布隆过滤器的原理与实现
4. 各类查找算法（顺序查找、二分查找、插值查找）
5. 哈希表与平衡树的比较

*最后更新：2026-06-12*


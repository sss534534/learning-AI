# 第十一章：高级数据结构

> 高级数据结构是为解决特定问题而设计的复杂数据结构，能够在特定场景下提供最优的时间复杂度。本章深入探讨各类前沿数据结构的设计原理与应用。

## 元数据

- **难度**: ⭐⭐⭐
- **前置知识**: ch07-ch10
- **关联文件**: [ch10-probabilistic-ds.md](ch10-probabilistic-ds.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [线段树（Segment Tree）](#1-线段树)
2. [树状数组（Fenwick Tree）](#2-树状数组)
3. [伸展树（Splay Tree）](#3-伸展树)
4. [Link-Cut Tree](#4-link-cut-tree)
5. [后缀自动机（Suffix Automaton）](#5-后缀自动机)
6. [持久化数据结构](#6-持久化数据结构)
7. [高级数据结构对比](#7-高级数据结构对比)

---

## 1. 线段树（Segment Tree）

### 1.1 原理

```
线段树是一种用于区间查询和区间更新的数据结构。

核心思想：
1. 将数组表示为二叉树，每个节点代表一个区间
2. 叶子节点对应数组的单个元素
3. 内部节点存储对应区间的聚合信息（如和、最小值、最大值等）

支持操作：
- 区间查询：查询某个区间的聚合值
- 单点更新：更新某个位置的值
- 区间更新：更新某个区间内所有元素
```

### 1.2 结构

```
数组 [1, 3, 5, 7, 9, 11] 对应的线段树：
                    [36]           (和)
                   /    \
                [9]      [27]
               /  \     /   \
             [4] [5] [16] [11]
             /\        /\
           [1][3]   [7][9]
```

### 1.3 Python实现

```python
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        
        # 初始化叶子节点
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # 构建树
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
    
    def update(self, pos, value):
        """单点更新"""
        pos += self.size
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            self.tree[pos] = self.tree[2*pos] + self.tree[2*pos+1]
            pos >>= 1
    
    def query(self, l, r):
        """区间查询 [l, r)"""
        res = 0
        l += self.size
        r += self.size
        while l < r:
            if l % 2 == 1:
                res += self.tree[l]
                l += 1
            if r % 2 == 1:
                r -= 1
                res += self.tree[r]
            l >>= 1
            r >>= 1
        return res
```

### 1.4 进阶：区间更新与延迟标记

```python
class SegmentTreeRange:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
    
    def push(self, node, l, r):
        """向下传递延迟标记"""
        if self.lazy[node] != 0 and node < self.size:
            mid = (l + r) // 2
            self.tree[2*node] += self.lazy[node] * (mid - l)
            self.lazy[2*node] += self.lazy[node]
            self.tree[2*node+1] += self.lazy[node] * (r - mid)
            self.lazy[2*node+1] += self.lazy[node]
            self.lazy[node] = 0
    
    def range_add(self, a, b, val, node=1, l=0, r=None):
        """区间加法"""
        if r is None:
            r = self.size
        if a >= r or b <= l:
            return
        if a <= l and r <= b:
            self.tree[node] += val * (r - l)
            self.lazy[node] += val
            return
        self.push(node, l, r)
        mid = (l + r) // 2
        self.range_add(a, b, val, 2*node, l, mid)
        self.range_add(a, b, val, 2*node+1, mid, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]
    
    def range_query(self, a, b, node=1, l=0, r=None):
        """区间查询"""
        if r is None:
            r = self.size
        if a >= r or b <= l:
            return 0
        if a <= l and r <= b:
            return self.tree[node]
        self.push(node, l, r)
        mid = (l + r) // 2
        return self.range_query(a, b, 2*node, l, mid) + self.range_query(a, b, 2*node+1, mid, r)
```

---

## 2. 树状数组（Fenwick Tree）

### 2.1 原理

```
树状数组（Binary Indexed Tree）是一种高效的前缀和数据结构。

核心思想：
1. 利用二进制表示将数组分成若干区间
2. 每个节点存储特定区间的和
3. 通过二进制位运算快速定位和更新

支持操作：
- 单点更新：O(log n)
- 前缀和查询：O(log n)
- 区间和查询：O(log n)
```

### 2.2 结构

```
数组索引的二进制表示决定了节点的覆盖范围：
- idx = 8 (1000)：覆盖 [1, 8]
- idx = 6 (0110)：覆盖 [5, 6]
- idx = 5 (0101)：覆盖 [5, 5]
```

### 2.3 Python实现

```python
class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, idx, delta):
        """单点更新：idx从1开始"""
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        """前缀和查询：[1, idx]"""
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res
    
    def range_query(self, l, r):
        """区间和查询：[l, r]"""
        return self.query(r) - self.query(l - 1)
```

### 2.4 二维树状数组

```python
class FenwickTree2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, x, y, delta):
        """单点更新"""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i
    
    def query(self, x, y):
        """前缀和查询：[1, x] x [1, y]"""
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return res
    
    def range_query(self, x1, y1, x2, y2):
        """矩形区域和查询"""
        return self.query(x2, y2) - self.query(x1-1, y2) - self.query(x2, y1-1) + self.query(x1-1, y1-1)
```

---

## 3. 伸展树（Splay Tree）

### 3.1 原理

```
伸展树是一种自调整二叉搜索树，通过伸展操作将访问的节点移动到根节点。

核心思想：
1. 每次访问节点后，通过一系列旋转将其移动到根
2. 摊还分析保证整体O(log n)的时间复杂度
3. 具有局部性优化：最近访问的节点更容易被再次访问

旋转类型：
- Zig：单旋转（当节点是根的直接子节点）
- Zig-Zig：双旋转（节点和父节点同向）
- Zig-Zag：双旋转（节点和父节点异向）
```

### 3.2 Python实现

```python
class SplayNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    def __init__(self):
        self.root = None
    
    def _rotate(self, x):
        """旋转操作"""
        y = x.parent
        z = y.parent
        
        if y.left == x:
            y.left = x.right
            if x.right:
                x.right.parent = y
            x.right = y
        else:
            y.right = x.left
            if x.left:
                x.left.parent = y
            x.left = y
        
        y.parent = x
        x.parent = z
        
        if z:
            if z.left == y:
                z.left = x
            else:
                z.right = x
    
    def _splay(self, x):
        """伸展操作"""
        while x.parent:
            y = x.parent
            z = y.parent
            
            if not z:
                # Zig
                self._rotate(x)
            elif (z.left == y) == (y.left == x):
                # Zig-Zig
                self._rotate(y)
                self._rotate(x)
            else:
                # Zig-Zag
                self._rotate(x)
                self._rotate(x)
        
        self.root = x
    
    def insert(self, key):
        """插入节点"""
        if not self.root:
            self.root = SplayNode(key)
            return
        
        current = self.root
        parent = None
        
        while current:
            parent = current
            if key < current.key:
                current = current.left
            else:
                current = current.right
        
        new_node = SplayNode(key)
        new_node.parent = parent
        
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        
        self._splay(new_node)
    
    def search(self, key):
        """查找节点"""
        current = self.root
        while current:
            if key == current.key:
                self._splay(current)
                return True
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return False
    
    def delete(self, key):
        """删除节点"""
        if not self.search(key):
            return
        
        root = self.root
        
        if not root.left:
            self.root = root.right
            if self.root:
                self.root.parent = None
        elif not root.right:
            self.root = root.left
            if self.root:
                self.root.parent = None
        else:
            left_max = root.left
            while left_max.right:
                left_max = left_max.right
            
            self._splay(left_max)
            left_max.right = root.right
            root.right.parent = left_max
            self.root = left_max
            self.root.parent = None
```

---

## 4. Link-Cut Tree

### 4.1 原理

```
Link-Cut Tree（动态树）是一种用于维护动态树森林的数据结构。

核心思想：
1. 将树分解成若干条路径（preferred path）
2. 每条路径用splay tree表示
3. 通过路径切割和链接操作维护树结构

支持操作：
- Link：连接两棵树
- Cut：切断一条边
- FindRoot：查找根节点
- PathQuery：路径查询（如路径和、路径最大值等）
- SubtreeQuery：子树查询
```

### 4.2 Python实现

```python
class LinkCutNode:
    def __init__(self, value):
        self.value = value
        self.sum = value
        self.max_val = value
        self.left = None
        self.right = None
        self.parent = None
        self.reverse = False
    
    def push(self):
        """传递翻转标记"""
        if self.reverse:
            self.left, self.right = self.right, self.left
            if self.left:
                self.left.reverse ^= True
            if self.right:
                self.right.reverse ^= True
            self.reverse = False
    
    def update(self):
        """更新聚合信息"""
        self.sum = self.value
        self.max_val = self.value
        if self.left:
            self.left.push()
            self.sum += self.left.sum
            self.max_val = max(self.max_val, self.left.max_val)
        if self.right:
            self.right.push()
            self.sum += self.right.sum
            self.max_val = max(self.max_val, self.right.max_val)

class LinkCutTree:
    def _is_root(self, x):
        """判断是否是splay树的根"""
        return not x.parent or (x.parent.left != x and x.parent.right != x)
    
    def _rotate(self, x):
        """旋转"""
        y = x.parent
        z = y.parent
        
        if y.left == x:
            y.left = x.right
            if x.right:
                x.right.parent = y
            x.right = y
        else:
            y.right = x.left
            if x.left:
                x.left.parent = y
            x.left = y
        
        y.parent = x
        x.parent = z
        
        if z:
            if z.left == y:
                z.left = x
            elif z.right == y:
                z.right = x
        
        y.update()
        x.update()
    
    def _splay(self, x):
        """伸展到根"""
        x.push()
        while not self._is_root(x):
            y = x.parent
            z = y.parent
            
            if not self._is_root(y):
                z.push()
            y.push()
            x.push()
            
            if not self._is_root(y):
                if (z.left == y) == (y.left == x):
                    self._rotate(y)
                else:
                    self._rotate(x)
            
            self._rotate(x)
    
    def access(self, x):
        """建立从根到x的preferred path"""
        last = None
        while x:
            self._splay(x)
            x.right = last
            x.update()
            last = x
            x = x.parent
        return last
    
    def make_root(self, x):
        """将x设为根"""
        self.access(x)
        self._splay(x)
        x.reverse ^= True
        x.push()
    
    def link(self, x, y):
        """连接x到y（x必须是根）"""
        self.make_root(x)
        x.parent = y
    
    def cut(self, x, y):
        """切断x和y之间的边"""
        self.make_root(x)
        self.access(y)
        self._splay(y)
        y.left.parent = None
        y.left = None
        y.update()
    
    def find_root(self, x):
        """查找x所在树的根"""
        self.access(x)
        self._splay(x)
        while x.left:
            x = x.left
            x.push()
        self._splay(x)
        return x
    
    def path_sum(self, x, y):
        """查询路径x到y的和"""
        self.make_root(x)
        self.access(y)
        self._splay(y)
        return y.sum
    
    def path_max(self, x, y):
        """查询路径x到y的最大值"""
        self.make_root(x)
        self.access(y)
        self._splay(y)
        return y.max_val
```

---

## 5. 后缀自动机（Suffix Automaton）

### 5.1 原理

```
后缀自动机（SAM）是一种用于处理字符串的高效数据结构。

核心思想：
1. 压缩表示字符串的所有子串
2. 状态代表等价类（endpos集合相同的子串）
3. 转移代表添加字符

特点：
- 空间复杂度：O(n)
- 时间复杂度：O(n)构建，O(m)查询
- 支持子串查询、最长公共子串等操作
```

### 5.2 Python实现

```python
class State:
    def __init__(self):
        self.transitions = {}
        self.link = None
        self.len = 0

class SuffixAutomaton:
    def __init__(self):
        self.size = 1
        self.last = 0
        self.states = [State()]
    
    def extend(self, c):
        """添加字符"""
        p = self.last
        curr = self.size
        self.size += 1
        self.states.append(State())
        self.states[curr].len = self.states[p].len + 1
        
        while p != -1 and c not in self.states[p].transitions:
            self.states[p].transitions[c] = curr
            p = self.states[p].link
        
        if p == -1:
            self.states[curr].link = 0
        else:
            q = self.states[p].transitions[c]
            if self.states[p].len + 1 == self.states[q].len:
                self.states[curr].link = q
            else:
                clone = self.size
                self.size += 1
                self.states.append(State())
                self.states[clone].transitions = self.states[q].transitions.copy()
                self.states[clone].link = self.states[q].link
                self.states[clone].len = self.states[p].len + 1
                
                while p != -1 and self.states[p].transitions.get(c) == q:
                    self.states[p].transitions[c] = clone
                    p = self.states[p].link
                
                self.states[q].link = clone
                self.states[curr].link = clone
        
        self.last = curr
    
    def contains(self, s):
        """判断字符串是否是子串"""
        curr = 0
        for c in s:
            if c not in self.states[curr].transitions:
                return False
            curr = self.states[curr].transitions[c]
        return True
    
    def count_distinct_substrings(self):
        """计算不同子串数量"""
        total = 0
        for i in range(1, self.size):
            total += self.states[i].len - self.states[self.states[i].link].len
        return total
```

---

## 6. 持久化数据结构

### 6.1 原理

```
持久化数据结构允许保留历史版本，同时支持查询和更新。

核心思想：
1. 每次更新时只复制受影响的节点
2. 共享未修改的部分
3. 支持对任意历史版本的查询

类型：
- 持久化线段树
- 持久化平衡树
- 持久化并查集
```

### 6.2 持久化线段树

```python
class PersistentSegmentTreeNode:
    def __init__(self, left=None, right=None, value=0):
        self.left = left
        self.right = right
        self.value = value

class PersistentSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.versions = []
        self.root = self._build(0, self.n - 1, data)
        self.versions.append(self.root)
    
    def _build(self, l, r, data):
        node = PersistentSegmentTreeNode()
        if l == r:
            node.value = data[l]
            return node
        
        mid = (l + r) // 2
        node.left = self._build(l, mid, data)
        node.right = self._build(mid + 1, r, data)
        node.value = node.left.value + node.right.value
        return node
    
    def _update(self, prev_node, l, r, idx, value):
        node = PersistentSegmentTreeNode()
        if l == r:
            node.value = value
            return node
        
        mid = (l + r) // 2
        if idx <= mid:
            node.left = self._update(prev_node.left, l, mid, idx, value)
            node.right = prev_node.right
        else:
            node.left = prev_node.left
            node.right = self._update(prev_node.right, mid + 1, r, idx, value)
        
        node.value = node.left.value + node.right.value
        return node
    
    def update(self, version_idx, idx, value):
        """基于历史版本创建新版本"""
        prev_root = self.versions[version_idx]
        new_root = self._update(prev_root, 0, self.n - 1, idx, value)
        self.versions.append(new_root)
        return len(self.versions) - 1
    
    def _query(self, node, l, r, ql, qr):
        if r < ql or l > qr:
            return 0
        if ql <= l and r <= qr:
            return node.value
        
        mid = (l + r) // 2
        return self._query(node.left, l, mid, ql, qr) + self._query(node.right, mid + 1, r, ql, qr)
    
    def query(self, version_idx, l, r):
        """查询指定版本的区间和"""
        root = self.versions[version_idx]
        return self._query(root, 0, self.n - 1, l, r)
```

---

## 7. 高级数据结构对比

### 7.1 时间复杂度对比

```
| 数据结构 | 单点更新 | 区间查询 | 特殊操作 | 主要应用 |
|---------|---------|---------|---------|---------|
| 线段树 | O(log n) | O(log n) | 区间更新 | 区间最值、区间和 |
| 树状数组 | O(log n) | O(log n) | 前缀和 | 频繁单点更新的场景 |
| 伸展树 | O(log n)* | O(log n)* | 序列维护 | 缓存友好的BST |
| Link-Cut Tree | O(log n)* | O(log n)* | 动态树操作 | 树链剖分、动态树 |
| 后缀自动机 | O(1)构建 | O(m)查询 | 子串匹配 | 字符串处理 |
| 持久化线段树 | O(log n) | O(log n) | 历史版本 | 可持久化查询 |

* 表示摊还复杂度
```

### 7.2 应用场景总结

```
- 区间查询与更新：线段树、树状数组
- 动态树维护：Link-Cut Tree
- 字符串处理：后缀自动机
- 需要历史版本：持久化数据结构
- 频繁访问优化：伸展树
```

---

## 本章小结

高级数据结构是解决复杂问题的关键工具：
1. 线段树：区间查询与更新的利器
2. 树状数组：高效的前缀和数据结构
3. 伸展树：具有局部性优化的自调整BST
4. Link-Cut Tree：动态树森林的高效维护
5. 后缀自动机：字符串处理的终极工具
6. 持久化数据结构：保留历史版本的能力

这些数据结构在算法竞赛、数据库系统、编译器设计等领域有广泛应用，是高级程序员必备的知识储备。

*最后更新：2026-06-12*
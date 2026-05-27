# 第八章：树与堆的数学

> 树是计算机科学中最重要的数据结构之一，本章深入探讨树的数学性质、各类树结构及其在算法中的应用。

---

## 目录

1. [树的基本概念](#1-树的基本概念)
2. [二叉树](#2-二叉树)
3. [二叉搜索树](#3-二叉搜索树)
4. [平衡树](#4-平衡树)
5. [B树与B+树](#5-b树与b树)
6. [堆与优先队列](#6-堆与优先队列)

---

## 1. 树的基本概念

### 1.1 树的定义

```
树是一个连通的无环图。
- 节点（Node）：存储数据的元素
- 边（Edge）：连接节点的线
- 根（Root）：没有父节点的节点
- 叶子（Leaf）：没有子节点的节点
- 路径（Path）：从一个节点到另一个节点的边序列
```

### 1.2 树的性质

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

def tree_height(root):
    """计算树的高度"""
    if not root:
        return 0
    if not root.children:
        return 1
    return 1 + max(tree_height(child) for child in root.children)

def tree_size(root):
    """计算树的节点数"""
    if not root:
        return 0
    return 1 + sum(tree_size(child) for child in root.children)
```

### 1.3 树的遍历

```python
def preorder_traversal(root):
    """前序遍历：根 -> 子树"""
    if not root:
        return []
    result = [root.value]
    for child in root.children:
        result.extend(preorder_traversal(child))
    return result

def postorder_traversal(root):
    """后序遍历：子树 -> 根"""
    if not root:
        return []
    result = []
    for child in root.children:
        result.extend(postorder_traversal(child))
    result.append(root.value)
    return result
```

---

## 2. 二叉树

### 2.1 二叉树的定义

```
二叉树是每个节点最多有两个子节点的树：
- 左子树（Left Subtree）
- 右子树（Right Subtree）
```

### 2.2 二叉树的性质

```
1. 第k层最多有2^(k-1)个节点（k≥1）
2. 高度为h的二叉树最多有2^h - 1个节点
3. n个节点的二叉树，其高度至少为log2(n+1)
```

### 2.3 完全二叉树与满二叉树

```
满二叉树：所有叶子节点都在同一层，且每个非叶子节点都有两个子节点。
完全二叉树：除最后一层外，其他层都是满的，最后一层的节点从左到右依次排列。
```

### 2.4 二叉树的遍历

```python
class BinaryTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_traversal(root):
    """中序遍历：左 -> 根 -> 右"""
    if not root:
        return []
    return inorder_traversal(root.left) + [root.value] + inorder_traversal(root.right)

def level_order_traversal(root):
    """层序遍历"""
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        result.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```

---

## 3. 二叉搜索树

### 3.1 BST的性质

```
二叉搜索树满足以下性质：
- 左子树所有节点的值 < 根节点的值
- 右子树所有节点的值 > 根节点的值
- 左右子树也都是BST
```

### 3.2 BST的操作

```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """插入节点"""
        if not self.root:
            self.root = BinaryTreeNode(value)
            return
        current = self.root
        while True:
            if value < current.value:
                if not current.left:
                    current.left = BinaryTreeNode(value)
                    break
                current = current.left
            else:
                if not current.right:
                    current.right = BinaryTreeNode(value)
                    break
                current = current.right
    
    def search(self, value):
        """查找节点"""
        current = self.root
        while current:
            if value == current.value:
                return True
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        return False
    
    def delete(self, value):
        """删除节点"""
        self.root = self._delete(self.root, value)
    
    def _delete(self, node, value):
        if not node:
            return node
        if value < node.value:
            node.left = self._delete(node.left, value)
        elif value > node.value:
            node.right = self._delete(node.right, value)
        else:
            # 找到要删除的节点
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            # 找到后继节点
            min_node = self._find_min(node.right)
            node.value = min_node.value
            node.right = self._delete(node.right, min_node.value)
        return node
    
    def _find_min(self, node):
        while node.left:
            node = node.left
        return node
```

### 3.3 BST的复杂度分析

```
平均情况（随机插入）：
- 查找：O(log n)
- 插入：O(log n)
- 删除：O(log n)

最坏情况（有序插入）：
- 查找：O(n)
- 插入：O(n)
- 删除：O(n)
```

---

## 4. 平衡树

### 4.1 AVL树

```
AVL树是一种自平衡二叉搜索树，满足：
- 任意节点的左右子树高度差不超过1
- 平衡因子 = 左子树高度 - 右子树高度 ∈ {-1, 0, 1}
```

### 4.2 AVL旋转操作

```python
class AVLNode(BinaryTreeNode):
    def __init__(self, value):
        super().__init__(value)
        self.height = 1

def get_height(node):
    return node.height if node else 0

def get_balance(node):
    return get_height(node.left) - get_height(node.right) if node else 0

def right_rotate(y):
    """右旋"""
    x = y.left
    T2 = x.right
    
    x.right = y
    y.left = T2
    
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    x.height = 1 + max(get_height(x.left), get_height(x.right))
    
    return x

def left_rotate(x):
    """左旋"""
    y = x.right
    T2 = y.left
    
    y.left = x
    x.right = T2
    
    x.height = 1 + max(get_height(x.left), get_height(x.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    
    return y

def avl_insert(node, value):
    """AVL插入"""
    if not node:
        return AVLNode(value)
    
    if value < node.value:
        node.left = avl_insert(node.left, value)
    elif value > node.value:
        node.right = avl_insert(node.right, value)
    else:
        return node
    
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    
    balance = get_balance(node)
    
    # LL情况
    if balance > 1 and value < node.left.value:
        return right_rotate(node)
    # RR情况
    if balance < -1 and value > node.right.value:
        return left_rotate(node)
    # LR情况
    if balance > 1 and value > node.left.value:
        node.left = left_rotate(node.left)
        return right_rotate(node)
    # RL情况
    if balance < -1 and value < node.right.value:
        node.right = right_rotate(node.right)
        return left_rotate(node)
    
    return node
```

### 4.3 红黑树

```
红黑树是一种自平衡二叉搜索树，满足以下性质：
1. 每个节点要么是红色，要么是黑色
2. 根节点是黑色
3. 所有叶子节点（NIL）是黑色
4. 如果一个节点是红色，则它的两个子节点都是黑色
5. 从任意节点到其每个叶子的所有路径都包含相同数目的黑色节点

红黑树的高度保证：h ≤ 2 log2(n+1)
```

---

## 5. B树与B+树

### 5.1 B树的定义

```
B树是一种平衡的多路搜索树，用于磁盘存储：
- 阶为m的B树，每个节点最多有m个子节点
- 根节点至少有2个子节点（除非是叶子）
- 非叶子节点至少有⌈m/2⌉个子节点
- 所有叶子在同一层
```

### 5.2 B树的操作

```python
class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t  # 最小度数
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, leaf=True)
        self.t = t
    
    def search(self, k):
        """查找键"""
        return self._search(self.root, k)
    
    def _search(self, node, k):
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:
            return (node, i)
        if node.leaf:
            return None
        return self._search(node.children[i], k)
    
    def insert(self, k):
        """插入键"""
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, leaf=False)
            self.root = new_root
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, k)
        else:
            self._insert_non_full(root, k)
    
    def _split_child(self, parent, i):
        """分裂子节点"""
        t = self.t
        child = parent.children[i]
        new_child = BTreeNode(t, child.leaf)
        
        parent.keys.insert(i, child.keys[t-1])
        parent.children.insert(i+1, new_child)
        
        new_child.keys = child.keys[t:]
        child.keys = child.keys[:t-1]
        
        if not child.leaf:
            new_child.children = child.children[t:]
            child.children = child.children[:t]
    
    def _insert_non_full(self, node, k):
        """向非满节点插入"""
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            while i >= 0 and k < node.keys[i]:
                node.keys[i+1] = node.keys[i]
                i -= 1
            node.keys[i+1] = k
        else:
            while i >= 0 and k < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if k > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], k)
```

### 5.3 B+树

```
B+树是B树的变体，优化了范围查询：
- 所有数据都存储在叶子节点
- 叶子节点形成链表，便于范围查询
- 非叶子节点只存储索引，不存储数据
```

### 5.4 B树与B+树的比较

```
B树：
- 优点：查询单个记录可能更快（数据可能在非叶子节点）
- 缺点：范围查询效率低

B+树：
- 优点：范围查询高效，叶子链表顺序访问
- 缺点：查询单个记录可能需要更多IO（必须到叶子节点）
```

---

## 6. 堆与优先队列

### 6.1 堆的定义

```
堆是一种完全二叉树，满足堆性质：
- 最大堆：每个父节点 ≥ 子节点
- 最小堆：每个父节点 ≤ 子节点
```

### 6.2 堆的操作

```python
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def insert(self, value):
        """插入元素"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """向上调整"""
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def extract_max(self):
        """提取最大值"""
        if not self.heap:
            return None
        max_val = self.heap[0]
        last_val = self.heap.pop()
        if self.heap:
            self.heap[0] = last_val
            self._heapify_down(0)
        return max_val
    
    def _heapify_down(self, i):
        """向下调整"""
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self._heapify_down(largest)
```

### 6.3 堆排序

```python
def heap_sort(arr):
    """堆排序"""
    heap = MaxHeap()
    for num in arr:
        heap.insert(num)
    
    result = []
    while heap.heap:
        result.insert(0, heap.extract_max())
    return result
```

### 6.4 优先队列应用

```
优先队列的典型应用：
- Dijkstra最短路径算法
- Huffman编码
- 任务调度
- 事件驱动模拟
```

---

## 本章小结

树与堆是数据结构的核心内容：
1. 树的基本性质和遍历方式
2. BST的搜索、插入、删除操作
3. AVL树和红黑树的自平衡机制
4. B树/B+树在磁盘存储中的应用
5. 堆作为优先队列的实现

这些数据结构在数据库索引、文件系统、网络路由等领域有广泛应用。
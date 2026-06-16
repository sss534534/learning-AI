# 第六章：图论与网络算法

> 图论研究图的性质和算法，广泛应用于网络分析、路径规划、社交网络等领域。

## 元数据

- **难度**: ⭐⭐
- **前置知识**: ch01-ch03
- **关联文件**: [ch05-complexity-theory.md](ch05-complexity-theory.md)
- **最后更新**: 2026-06-12

---

## 目录

1. [图的基本概念](#1-图的基本概念)
2. [图的表示](#2-图的表示)
3. [图的遍历](#3-图的遍历)
4. [最短路径算法](#4-最短路径算法)
5. [最小生成树](#5-最小生成树)
6. [网络流](#6-网络流)

---

## 1. 图的基本概念

### 1.1 图的定义

**图G = (V, E)** 由以下组成：
- V：顶点（节点）的非空集合
- E：边的集合，每条边连接两个顶点

### 1.2 图的类型

| 类型 | 特点 |
|-----|------|
| 无向图 | 边无方向，(u, v) = (v, u) |
| 有向图 | 边有方向，<u, v> ≠ <v, u> |
| 加权图 | 边有权重 |
| 简单图 | 无自环、无重边 |
| 完全图 | 任意两顶点间都有边 |
| 连通图 | 任意两点间有路径 |

### 1.3 基本术语

- **顶点度数：** 连接顶点的边数
- **路径：** 顶点序列
- **回路（环）：** 起点和终点相同的路径
- **子图：** 图的一部分

---

## 2. 图的表示

### 2.1 邻接矩阵

**适用场景：** 稠密图

**Python示例：**

```python
# 邻接矩阵表示
graph_matrix = [
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
]
```

### 2.2 邻接表

**适用场景：** 稀疏图

**Python示例：**

```python
# 邻接表表示
graph_adj = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2]
}

# 加权图的邻接表
weighted_graph = {
    0: [(1, 4), (2, 2)],
    1: [(0, 4), (2, 1), (3, 5)],
    2: [(0, 2), (1, 1), (3, 8)],
    3: [(1, 5), (2, 8)]
}
```

---

## 3. 图的遍历

### 3.1 深度优先搜索（DFS）

**算法思想：** 尽可能深入，回溯再探索

**Python实现：**

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result

# 迭代版DFS
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return result
```

**时间复杂度：** O(V + E)

### 3.2 广度优先搜索（BFS）

**算法思想：** 逐层遍历，先近后远

**Python实现：**

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

# BFS求最短路径（无权图）
def bfs_shortest_path(graph, start, end):
    if start == end:
        return [start]
    visited = {}
    queue = deque([start])
    visited[start] = None
    found = False
    while queue and not found:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)
                if neighbor == end:
                    found = True
                    break
    # 重建路径
    if end not in visited:
        return None
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = visited[current]
    return path[::-1]
```

**时间复杂度：** O(V + E)

---

## 4. 最短路径算法

### 4.1 Dijkstra算法

**适用场景：** 非负权边的单源最短路径

**Python实现：**

```python
import heapq

def dijkstra(graph, start):
    # graph: {u: [(v, weight), ...]}
    n = len(graph)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    visited = {}
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited[u] = True
        
        for v, weight in graph[u]:
            if distances[v] > distances[u] + weight:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))
    return distances

# 带路径记录的Dijkstra
def dijkstra_with_path(graph, start, end):
    n = len(graph)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    pq = [(0, start)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if u == end:
            break
        if current_dist > distances[u]:
            continue
            
        for v, weight in graph[u]:
            if distances[v] > distances[u] + weight:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                heapq.heappush(pq, (distances[v], v))
    
    # 重建路径
    if distances[end] == float('inf'):
        return None, float('inf')
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return path[::-1], distances[end]
```

**时间复杂度：** O(E + V log V)（使用优先队列）

### 4.2 Bellman-Ford算法

**适用场景：** 含负权边但无负权环的单源最短路径

**Python实现：**

```python
def bellman_ford(graph, start, V):
    distances = {node: float('inf') for node in range(V)}
    distances[start] = 0
    
    # 松弛V-1次
    for _ in range(V - 1):
        updated = False
        for u in graph:
            for v, weight in graph[u]:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    updated = True
        if not updated:
            break
    
    # 检测负权环
    has_negative_cycle = False
    for u in graph:
        for v, weight in graph[u]:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break
        if has_negative_cycle:
            break
    return distances, has_negative_cycle
```

**时间复杂度：** O(V·E)

### 4.3 Floyd-Warshall算法

**适用场景：** 所有点对的最短路径

**Python实现：**

```python
def floyd_warshall(graph_matrix):
    V = len(graph_matrix)
    dist = [row[:] for row in graph_matrix]
    
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
```

**时间复杂度：** O(V³)

---

## 5. 最小生成树

### 5.1 Prim算法

**算法思想：** 从一个顶点开始，逐步扩展

**Python实现：**

```python
import heapq

def prim(graph, start):
    V = len(graph)
    in_mst = [False] * V
    edge_weight = [float('inf')] * V
    edge_weight[start] = 0
    pq = [(0, start)]
    parent = [-1] * V
    total_weight = 0
    
    while pq:
        weight, u = heapq.heappop(pq)
        if in_mst[u]:
            continue
        in_mst[u] = True
        total_weight += weight
        
        for v, w in graph[u]:
            if not in_mst[v] and w < edge_weight[v]:
                edge_weight[v] = w
                parent[v] = u
                heapq.heappush(pq, (w, v))
    return parent, total_weight
```

**时间复杂度：** O(E + V log V)

### 5.2 Kruskal算法

**算法思想：** 按权排序，贪心选择，避免环

**Python实现（并查集）：**

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
        return True

def kruskal(edges, V):
    # edges: [(weight, u, v), ...]
    edges.sort()
    uf = UnionFind(V)
    mst = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
    return mst, total_weight
```

**时间复杂度：** O(E log E)

---

## 6. 网络流

### 6.1 Ford-Fulkerson与Edmonds-Karp算法

**Python实现（Edmonds-Karp）：**

```python
from collections import deque

def edmonds_karp(graph, source, sink):
    V = len(graph)
    residual = [row[:] for row in graph]
    max_flow = 0
    
    while True:
        # BFS找增广路径
        parent = [-1] * V
        queue = deque([source])
        parent[source] = -2
        while queue:
            u = queue.popleft()
            for v in range(V):
                if parent[v] == -1 and residual[u][v] > 0:
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break
        if parent[sink] == -1:
            break
        
        # 找最小残量
        flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            flow = min(flow, residual[u][v])
            v = u
        
        # 更新残量网络
        max_flow += flow
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= flow
            residual[v][u] += flow
            v = u
    return max_flow
```

**时间复杂度：** O(V·E²)

---

## 本章小结

图论是计算机科学的核心内容，本章要点：

1. **图的基本概念与表示方法
2. **DFS与BFS遍历算法
3. **Dijkstra、Bellman-Ford、Floyd-Warshall最短路径算法
4. **Prim与Kruskal最小生成树算法
5. **Edmonds-Karp网络流算法

**下一章（数据结构部分）：** 我们将学习集合与关系。

*最后更新：2026-06-12*


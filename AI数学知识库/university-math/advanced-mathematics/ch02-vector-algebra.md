# 第二章 向量代数与空间解析几何

## 1. 向量及其线性运算

### 1.1 向量的概念

**定义：** 既有大小又有方向的量称为向量（或矢量）。向量可以用有向线段表示，记为 $\vec{a}$ 或 $\boldsymbol{a}$。

- 向量的大小称为向量的模，记为 $|\vec{a}|$。
- 模为1的向量称为单位向量，记为 $\vec{a}^0 = \frac{\vec{a}}{|\vec{a}|}$。
- 模为0的向量称为零向量，记为 $\vec{0}$，方向任意。

### 1.2 向量的加法

**定义：** 设向量 $\vec{a}$ 和 $\vec{b}$，将 $\vec{b}$ 的起点移到 $\vec{a}$ 的终点，则从 $\vec{a}$ 的起点到 $\vec{b}$ 的终点的向量称为 $\vec{a}$ 与 $\vec{b}$ 的和，记为 $\vec{a} + \vec{b}$。

**三角形法则与平行四边形法则：**

**运算性质：**
1. 交换律：$\vec{a} + \vec{b} = \vec{b} + \vec{a}$
2. 结合律：$(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c})$
3. $\vec{a} + \vec{0} = \vec{a}$
4. $\vec{a} + (-\vec{a}) = \vec{0}$

### 1.3 向量的数乘

**定义：** 实数 $\lambda$ 与向量 $\vec{a}$ 的乘积是一个向量，记为 $\lambda\vec{a}$，满足：
- $|\lambda\vec{a}| = |\lambda||\vec{a}|$
- 当 $\lambda > 0$ 时，$\lambda\vec{a}$ 与 $\vec{a}$ 同向；当 $\lambda < 0$ 时，反向；当 $\lambda = 0$ 时，$\lambda\vec{a} = \vec{0}$

**运算性质：**
1. 结合律：$\lambda(\mu\vec{a}) = (\lambda\mu)\vec{a}$
2. 分配律：$(\lambda + \mu)\vec{a} = \lambda\vec{a} + \mu\vec{a}$，$\lambda(\vec{a} + \vec{b}) = \lambda\vec{a} + \lambda\vec{b}$

### 1.4 代码示例：向量线性运算

```python
import numpy as np

class Vector:
    def __init__(self, *components):
        self.components = np.array(components)
    
    def __add__(self, other):
        return Vector(*(self.components + other.components))
    
    def __mul__(self, scalar):
        return Vector(*(self.components * scalar))
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __str__(self):
        return f"Vector{tuple(self.components)}"
    
    @property
    def magnitude(self):
        return np.linalg.norm(self.components)
    
    @property
    def unit_vector(self):
        mag = self.magnitude
        if mag == 0:
            return Vector(*np.zeros_like(self.components))
        return Vector(*(self.components / mag))

# 示例
a = Vector(1, 2, 3)
b = Vector(4, 5, 6)
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"2a = {2 * a}")
print(f"|a| = {a.magnitude}")
print(f"a的单位向量 = {a.unit_vector}")
```

---

## 2. 向量的数量积、向量积、混合积及其几何意义

### 2.1 数量积（点积）

**定义：** 两向量 $\vec{a}$ 与 $\vec{b}$ 的数量积是一个数，等于它们的模与它们之间夹角 $\theta$ 的余弦的乘积，记为 $\vec{a} \cdot \vec{b}$：

$$\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$$

**运算性质：**
1. 交换律：$\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$
2. 分配律：$\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}$
3. 结合律：$(\lambda\vec{a}) \cdot \vec{b} = \lambda(\vec{a} \cdot \vec{b}) = \vec{a} \cdot (\lambda\vec{b})$

**几何意义：** $\vec{a} \cdot \vec{b}$ 等于 $|\vec{a}|$ 乘以 $\vec{b}$ 在 $\vec{a}$ 方向上的投影。

**垂直条件：** $\vec{a} \perp \vec{b} \iff \vec{a} \cdot \vec{b} = 0$

### 2.2 向量积（叉积）

**定义：** 两向量 $\vec{a}$ 与 $\vec{b}$ 的向量积是一个向量，记为 $\vec{a} \times \vec{b}$，满足：
- $|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta$
- 方向：垂直于 $\vec{a}$ 和 $\vec{b}$ 所确定的平面，右手定则

**运算性质：**
1. 反交换律：$\vec{a} \times \vec{b} = -\vec{b} \times \vec{a}$
2. 分配律：$\vec{a} \times (\vec{b} + \vec{c}) = \vec{a} \times \vec{b} + \vec{a} \times \vec{c}$
3. 结合律：$(\lambda\vec{a}) \times \vec{b} = \lambda(\vec{a} \times \vec{b}) = \vec{a} \times (\lambda\vec{b})$

**几何意义：** $|\vec{a} \times \vec{b}|$ 等于以 $\vec{a}$ 和 $\vec{b}$ 为邻边的平行四边形的面积。

**平行条件：** $\vec{a} \parallel \vec{b} \iff \vec{a} \times \vec{b} = \vec{0}$

### 2.3 混合积

**定义：** 三个向量 $\vec{a}, \vec{b}, \vec{c}$ 的混合积定义为 $(\vec{a} \times \vec{b}) \cdot \vec{c}$，记为 $[\vec{a}, \vec{b}, \vec{c}]$。

**性质：**
- 轮换对称性：$[\vec{a}, \vec{b}, \vec{c}] = [\vec{b}, \vec{c}, \vec{a}] = [\vec{c}, \vec{a}, \vec{b}]$
- 交换两向量变号：$[\vec{a}, \vec{b}, \vec{c}] = -[\vec{b}, \vec{a}, \vec{c}]$

**几何意义：** $|[\vec{a}, \vec{b}, \vec{c}]|$ 等于以 $\vec{a}, \vec{b}, \vec{c}$ 为棱的平行六面体的体积。

**共面条件：** 三向量共面 $\iff [\vec{a}, \vec{b}, \vec{c}] = 0$

### 2.4 代码示例：向量的乘积运算

```python
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)

def cross_product(a, b):
    return np.cross(a, b)

def scalar_triple_product(a, b, c):
    return np.dot(np.cross(a, b), c)

# 示例
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

print(f"a · b = {dot_product(a, b)}")
print(f"a × b = {cross_product(a, b)}")
print(f"[a, b, c] = {scalar_triple_product(a, b, c)}")

# 几何意义：平行四边形面积
parallelogram_area = np.linalg.norm(cross_product(a, b))
print(f"以a、b为邻边的平行四边形面积 = {parallelogram_area}")
```

---

## 3. 空间直角坐标系与向量的坐标表示

### 3.1 空间直角坐标系

在空间中取定一点 $O$，作三条互相垂直的数轴 $Ox, Oy, Oz$，它们都以 $O$ 为原点且有相同的长度单位，称为空间直角坐标系 $Oxyz$。

- 点 $M$ 的坐标为 $(x, y, z)$
- 原点 $O(0, 0, 0)$

### 3.2 向量的坐标表示

设点 $M(x, y, z)$，则位置向量 $\vec{r} = \overrightarrow{OM} = x\vec{i} + y\vec{j} + z\vec{k}$，记为 $\vec{r} = (x, y, z)$，其中 $\vec{i}, \vec{j}, \vec{k}$ 分别为 $x, y, z$ 轴正向的单位向量。

设 $\vec{a} = (a_x, a_y, a_z)$，$\vec{b} = (b_x, b_y, b_z)$，则：

**线性运算：**
- $\vec{a} + \vec{b} = (a_x + b_x, a_y + b_y, a_z + b_z)$
- $\lambda\vec{a} = (\lambda a_x, \lambda a_y, \lambda a_z)$

**数量积：**
- $\vec{a} \cdot \vec{b} = a_xb_x + a_yb_y + a_zb_z$
- $|\vec{a}| = \sqrt{a_x^2 + a_y^2 + a_z^2}$
- $\cos\theta = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|} = \frac{a_xb_x + a_yb_y + a_zb_z}{\sqrt{a_x^2 + a_y^2 + a_z^2}\sqrt{b_x^2 + b_y^2 + b_z^2}}$

**向量积：**
$$\vec{a} \times \vec{b} = \begin{vmatrix}
\vec{i} & \vec{j} & \vec{k} \\
a_x & a_y & a_z \\
b_x & b_y & b_z
\end{vmatrix} = (a_yb_z - a_zb_y, a_zb_x - a_xb_z, a_xb_y - a_yb_x)$$

**混合积：**
$$[\vec{a}, \vec{b}, \vec{c}] = \begin{vmatrix}
a_x & a_y & a_z \\
b_x & b_y & b_z \\
c_x & c_y & c_z
\end{vmatrix}$$

### 3.3 两点间距离

设两点 $M_1(x_1, y_1, z_1)$，$M_2(x_2, y_2, z_2)$，则距离：

$$|M_1M_2| = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$

### 3.4 方向角与方向余弦

向量 $\vec{a} = (a_x, a_y, a_z)$ 与 $x, y, z$ 轴的夹角 $\alpha, \beta, \gamma$ 称为方向角，$\cos\alpha, \cos\beta, \cos\gamma$ 称为方向余弦：

$$\cos\alpha = \frac{a_x}{|\vec{a}|}, \quad \cos\beta = \frac{a_y}{|\vec{a}|}, \quad \cos\gamma = \frac{a_z}{|\vec{a}|}$$

且满足：$\cos^2\alpha + \cos^2\beta + \cos^2\gamma = 1$

---

## 4. 平面及其方程、直线及其方程

### 4.1 平面的点法式方程

若平面过点 $M_0(x_0, y_0, z_0)$，且法向量为 $\vec{n} = (A, B, C)$，则平面方程为：

$$A(x - x_0) + B(y - y_0) + C(z - z_0) = 0$$

### 4.2 平面的一般式方程

$$Ax + By + Cz + D = 0$$

其中 $\vec{n} = (A, B, C)$ 为平面的法向量。

### 4.3 平面的三点式方程

过不共线三点 $M_1(x_1, y_1, z_1), M_2(x_2, y_2, z_2), M_3(x_3, y_3, z_3)$ 的平面方程：

$$\begin{vmatrix}
x - x_1 & y - y_1 & z - z_1 \\
x_2 - x_1 & y_2 - y_1 & z_2 - z_1 \\
x_3 - x_1 & y_3 - y_1 & z_3 - z_1
\end{vmatrix} = 0$$

### 4.4 两平面的夹角

两平面法向量的夹角（通常取锐角或直角）称为两平面的夹角：

$$\cos\theta = \frac{|\vec{n_1} \cdot \vec{n_2}|}{|\vec{n_1}||\vec{n_2}|} = \frac{|A_1A_2 + B_1B_2 + C_1C_2|}{\sqrt{A_1^2 + B_1^2 + C_1^2}\sqrt{A_2^2 + B_2^2 + C_2^2}}$$

- 垂直：$A_1A_2 + B_1B_2 + C_1C_2 = 0$
- 平行：$\frac{A_1}{A_2} = \frac{B_1}{B_2} = \frac{C_1}{C_2}$

### 4.5 直线的参数方程

若直线过点 $M_0(x_0, y_0, z_0)$，且方向向量为 $\vec{s} = (m, n, p)$，则参数方程为：

$$\begin{cases}
x = x_0 + mt \\
y = y_0 + nt \\
z = z_0 + pt
\end{cases}, \quad t \in \mathbb{R}$$

### 4.6 直线的对称式方程（点向式方程）

$$\frac{x - x_0}{m} = \frac{y - y_0}{n} = \frac{z - z_0}{p}$$

### 4.7 直线的一般式方程

直线作为两平面的交线：

$$\begin{cases}
A_1x + B_1y + C_1z + D_1 = 0 \\
A_2x + B_2y + C_2z + D_2 = 0
\end{cases}$$

### 4.8 两直线的夹角

两直线方向向量的夹角（通常取锐角或直角）称为两直线的夹角。

### 4.9 直线与平面的夹角

直线与平面的夹角 $\theta$ 是直线与平面中所有直线夹角中的最小者，满足：

$$\sin\theta = \frac{|\vec{s} \cdot \vec{n}|}{|\vec{s}||\vec{n}|}$$

其中 $\vec{s}$ 为直线方向向量，$\vec{n}$ 为平面法向量。

### 4.10 代码示例：平面与直线方程

```python
import numpy as np

def plane_point_normal(M0, n):
    """平面点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=0"""
    A, B, C = n
    x0, y0, z0 = M0
    D = -(A*x0 + B*y0 + C*z0)
    return (A, B, C, D)

def plane_three_points(M1, M2, M3):
    """平面三点式方程"""
    M1, M2, M3 = np.array(M1), np.array(M2), np.array(M3)
    v1 = M2 - M1
    v2 = M3 - M1
    n = np.cross(v1, v2)
    return plane_point_normal(M1, n)

def line_parametric(M0, s):
    """直线参数方程"""
    return {'point': M0, 'direction': s}

def point_on_plane(point, plane_coeffs):
    """判断点是否在平面上"""
    A, B, C, D = plane_coeffs
    x, y, z = point
    return np.isclose(A*x + B*y + C*z + D, 0)

# 示例
M0 = (1, 2, 3)
n = (2, -1, 3)
plane = plane_point_normal(M0, n)
print(f"平面方程: {plane[0]}x + {plane[1]}y + {plane[2]}z + {plane[3]} = 0")

M1, M2, M3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
plane2 = plane_three_points(M1, M2, M3)
print(f"过三点的平面方程: {plane2[0]}x + {plane2[1]}y + {plane2[2]}z + {plane2[3]} = 0")

line = line_parametric((0, 0, 0), (1, 2, 3))
print(f"直线参数方程: x={line['point'][0]}+{line['direction'][0]}t, y={line['point'][1]}+{line['direction'][1]}t, z={line['point'][2]}+{line['direction'][2]}t")
```

---

## 5. 空间曲面与曲线

### 5.1 曲面方程

如果曲面 $S$ 与三元方程 $F(x, y, z) = 0$ 满足：
- 曲面 $S$ 上任意点的坐标都满足方程
- 不在曲面 $S$ 上的点的坐标都不满足方程

则称 $F(x, y, z) = 0$ 为曲面 $S$ 的方程，曲面 $S$ 称为方程的图形。

### 5.2 球面方程

球心在 $(x_0, y_0, z_0)$，半径为 $R$ 的球面方程：

$$(x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = R^2$$

### 5.3 旋转曲面

以一条平面曲线绕其平面上一条定直线旋转一周所成的曲面称为旋转曲面。

例如，$yOz$ 平面上曲线 $f(y, z) = 0$ 绕 $z$ 轴旋转的旋转曲面方程：

$$f(\pm\sqrt{x^2 + y^2}, z) = 0$$

### 5.4 柱面

平行于定直线并沿定曲线 $C$ 移动的直线 $L$ 所形成的曲面称为柱面，$C$ 称为准线，$L$ 称为母线。

例如，方程 $x^2 + y^2 = R^2$ 在空间中表示母线平行于 $z$ 轴的圆柱面。

### 5.5 二次曲面

三元二次方程所表示的曲面称为二次曲面。

#### (1) 椭球面

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$$

#### (2) 椭圆抛物面

$$\frac{x^2}{2p} + \frac{y^2}{2q} = z \quad (p, q > 0)$$

#### (3) 双曲抛物面（马鞍面）

$$-\frac{x^2}{2p} + \frac{y^2}{2q} = z \quad (p, q > 0)$$

#### (4) 单叶双曲面

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} - \frac{z^2}{c^2} = 1$$

#### (5) 双叶双曲面

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} - \frac{z^2}{c^2} = -1$$

#### (6) 二次锥面

$$\frac{x^2}{a^2} + \frac{y^2}{b^2} - \frac{z^2}{c^2} = 0$$

### 5.6 空间曲线的方程

#### (1) 一般式方程

空间曲线作为两曲面的交线：

$$\begin{cases}
F(x, y, z) = 0 \\
G(x, y, z) = 0
\end{cases}$$

#### (2) 参数方程

$$\begin{cases}
x = x(t) \\
y = y(t) \\
z = z(t)
\end{cases}, \quad t \in [\alpha, \beta]$$

### 5.7 螺旋线参数方程示例

$$\begin{cases}
x = a\cos t \\
y = a\sin t \\
z = bt
\end{cases}, \quad t \in \mathbb{R}$$

### 5.8 代码示例：二次曲面可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ellipsoid(a=2, b=3, c=1):
    """绘制椭球面"""
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('椭球面')
    plt.show()

def plot_elliptic_paraboloid(p=2, q=2):
    """绘制椭圆抛物面"""
    x, y = np.mgrid[-5:5:100j, -5:5:100j]
    z = (x**2)/(2*p) + (y**2)/(2*q)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('椭圆抛物面')
    plt.show()

def plot_hyperbolic_paraboloid(p=2, q=2):
    """绘制双曲抛物面（马鞍面）"""
    x, y = np.mgrid[-5:5:100j, -5:5:100j]
    z = -(x**2)/(2*p) + (y**2)/(2*q)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('双曲抛物面')
    plt.show()

def plot_helix(a=1, b=0.5, t_max=10*np.pi):
    """绘制螺旋线"""
    t = np.linspace(0, t_max, 500)
    x = a * np.cos(t)
    y = a * np.sin(t)
    z = b * t
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('螺旋线')
    plt.show()

# 调用示例
if __name__ == "__main__":
    plot_ellipsoid()
    plot_elliptic_paraboloid()
    plot_hyperbolic_paraboloid()
    plot_helix()
```

---

## 6. 向量分析在AI中的应用

### 6.1 词向量（Word Embedding）

在自然语言处理（NLP）中，词向量是将词语映射为高维空间中的向量，使得语义相近的词在向量空间中距离较近。

**核心思想：**
- 将每个词 $w$ 表示为向量 $\vec{v}_w \in \mathbb{R}^d$
- 语义相似性通过向量距离或相似度衡量

### 6.2 余弦相似度

**定义：** 两向量 $\vec{a}$ 和 $\vec{b}$ 的余弦相似度定义为它们夹角的余弦值：

$$\text{sim}(\vec{a}, \vec{b}) = \cos\theta = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}$$

**性质：**
- 取值范围：$[-1, 1]$
- 1 表示完全同向，-1 表示完全反向，0 表示正交

**在词向量中的应用：** 比较两个词语义的相似程度。

### 6.3 特征空间

在机器学习中，每个样本被表示为特征空间中的一个点（向量），向量的每个维度对应一个特征。

**核心概念：**
- 样本：$\vec{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$
- 特征空间：$\mathbb{R}^d$
- 分类/聚类：在特征空间中寻找决策边界或簇

### 6.4 向量在机器学习中的其他应用

1. **线性分类器：** 决策边界为超平面 $\vec{w} \cdot \vec{x} + b = 0$
2. **主成分分析（PCA）：** 通过特征向量降维
3. **神经网络：** 权重矩阵和激活向量的运算

### 6.5 代码示例：余弦相似度与词向量

```python
import numpy as np
from collections import Counter

def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def one_hot_vector(word, vocab):
    """生成One-Hot向量"""
    vec = np.zeros(len(vocab))
    if word in vocab:
        vec[vocab.index(word)] = 1
    return vec

def bag_of_words(text, vocab):
    """生成词袋向量"""
    words = text.lower().split()
    word_counts = Counter(words)
    vec = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        vec[i] = word_counts.get(word, 0)
    return vec

# 示例：词向量与余弦相似度
vocab = ["king", "queen", "man", "woman", "prince", "princess"]

# 简化的词向量示例（实际应用中使用Word2Vec、GloVe等训练）
word_vectors = {
    "king": np.array([0.8, 0.6, 0.0, 0.0]),
    "queen": np.array([0.8, 0.5, 0.0, 0.1]),
    "man": np.array([0.0, 0.0, 0.8, 0.6]),
    "woman": np.array([0.0, 0.1, 0.8, 0.5]),
    "prince": np.array([0.7, 0.7, 0.1, 0.0]),
    "princess": np.array([0.7, 0.6, 0.1, 0.1])
}

print("词向量余弦相似度：")
print(f"king - queen: {cosine_similarity(word_vectors['king'], word_vectors['queen']):.4f}")
print(f"man - woman: {cosine_similarity(word_vectors['man'], word_vectors['woman']):.4f}")
print(f"king - man: {cosine_similarity(word_vectors['king'], word_vectors['man']):.4f}")
print(f"king - princess: {cosine_similarity(word_vectors['king'], word_vectors['princess']):.4f}")

# 文本相似度示例
text1 = "The king rules the kingdom"
text2 = "The queen governs the country"
text3 = "A man walks in the park"

full_vocab = list(set(text1.lower().split() + text2.lower().split() + text3.lower().split()))
bow1 = bag_of_words(text1, full_vocab)
bow2 = bag_of_words(text2, full_vocab)
bow3 = bag_of_words(text3, full_vocab)

print("\n文本词袋向量余弦相似度：")
print(f"text1 - text2: {cosine_similarity(bow1, bow2):.4f}")
print(f"text1 - text3: {cosine_similarity(bow1, bow3):.4f}")
print(f"text2 - text3: {cosine_similarity(bow2, bow3):.4f}")
```

---

## 本章小结

1. **向量代数基础**：掌握向量的线性运算（加法、数乘）及几何意义
2. **向量乘积**：理解数量积、向量积、混合积的定义、性质和几何意义
3. **坐标表示**：熟练运用空间直角坐标系进行向量运算
4. **空间解析几何**：掌握平面和直线的各种方程形式，以及空间曲面和曲线的表示
5. **AI应用**：理解向量分析在词向量、余弦相似度、特征空间等AI领域的核心应用

向量代数与空间解析几何是多元微积分、线性代数及机器学习的重要数学基础。

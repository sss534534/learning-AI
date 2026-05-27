# 第三章 多元函数微分学

## §3.1 多元函数的概念、极限、连续

### 3.1.1 多元函数的概念

**定义 3.1.1（多元函数）** 设 \( D \) 是 \( \mathbb{R}^n \) 的一个非空子集，称映射 \( f: D \to \mathbb{R} \) 为定义在 \( D \) 上的**n元函数**，记为
$$ z = f(x_1, x_2, \dots, x_n) \quad \text{或} \quad u = f(P) $$
其中 \( (x_1, x_2, \dots, x_n) \in D \)，点集 \( D \) 称为函数的**定义域**，记作 \( D_f \) 或 \( D \)。

**二元函数的几何意义** 二元函数 \( z = f(x, y) \) 的图形通常是空间直角坐标系中的一张曲面。

**例 3.1.1** 二元函数 \( z = \sqrt{1 - x^2 - y^2} \) 的定义域为 \( D = \{ (x, y) \mid x^2 + y^2 \leq 1 \} \)，其图形是上半球面。

### 3.1.2 多元函数的极限

**定义 3.1.2（二元函数的极限）** 设函数 \( f(P) = f(x, y) \) 在点 \( P_0(x_0, y_0) \) 的某个去心邻域内有定义。如果存在常数 \( A \)，对于任意给定的正数 \( \varepsilon > 0 \)，总存在正数 \( \delta > 0 \)，使得当点 \( P(x, y) \in \mathring{U}(P_0, \delta) \) 时，都有
$$ |f(P) - A| < \varepsilon $$
则称 \( A \) 为函数 \( f(P) \) 当 \( P \to P_0 \) 时的**极限**，记作
$$ \lim_{(x, y) \to (x_0, y_0)} f(x, y) = A \quad \text{或} \quad \lim_{P \to P_0} f(P) = A $$

**注**：二元函数的极限存在，要求点 \( P(x, y) \) 以任何方式趋近于 \( P_0(x_0, y_0) \) 时，\( f(P) \) 都趋于同一常数。

**例 3.1.2** 求极限 \( \lim_{(x, y) \to (0, 0)} \frac{xy}{\sqrt{x^2 + y^2}} \)。

**解**：令 \( (x, y) \) 沿直线 \( y = kx \) 趋近于 \( (0, 0) \)，则
$$ \lim_{\substack{(x, y) \to (0, 0) \\ y = kx}} \frac{xy}{\sqrt{x^2 + y^2}} = \lim_{x \to 0} \frac{kx^2}{|x|\sqrt{1 + k^2}} = 0 $$
但沿抛物线 \( y = x^2 \) 趋近于 \( (0, 0) \)，则
$$ \lim_{\substack{(x, y) \to (0, 0) \\ y = x^2}} \frac{xy}{\sqrt{x^2 + y^2}} = \lim_{x \to 0} \frac{x^3}{|x|\sqrt{1 + x^2}} = 0 $$
实际上，由不等式
$$ \left| \frac{xy}{\sqrt{x^2 + y^2}} \right| \leq \frac{\frac{1}{2}(x^2 + y^2)}{\sqrt{x^2 + y^2}} = \frac{1}{2}\sqrt{x^2 + y^2} \to 0 $$
故极限为 0。

### 3.1.3 多元函数的连续性

**定义 3.1.3（二元函数的连续性）** 设函数 \( f(x, y) \) 在点 \( P_0(x_0, y_0) \) 的某邻域内有定义，如果
$$ \lim_{(x, y) \to (x_0, y_0)} f(x, y) = f(x_0, y_0) $$
则称函数 \( f(x, y) \) 在点 \( P_0(x_0, y_0) \) 处**连续**。

**定理 3.1.1（有界闭区域上连续函数的性质）** 设 \( f(P) \) 在有界闭区域 \( D \) 上连续，则：
1. （有界性定理）\( f(P) \) 在 \( D \) 上有界；
2. （最值定理）\( f(P) \) 在 \( D \) 上必取得最大值和最小值；
3. （介值定理）对于介于最大值和最小值之间的任意值 \( C \)，存在 \( P \in D \) 使得 \( f(P) = C \)。

## §3.2 偏导数、高阶偏导数、全微分

### 3.2.1 偏导数

**定义 3.2.1（偏导数）** 设函数 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 的某邻域内有定义，固定 \( y = y_0 \)，得到一元函数 \( f(x, y_0) \)，如果它在 \( x = x_0 \) 处可导，则称此导数为 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 处对 \( x \) 的**偏导数**，记作
$$ \frac{\partial z}{\partial x}\bigg|_{(x_0, y_0)}, \quad \frac{\partial f}{\partial x}\bigg|_{(x_0, y_0)}, \quad z_x\bigg|_{(x_0, y_0)} \quad \text{或} \quad f_x(x_0, y_0) $$
即
$$ f_x(x_0, y_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x} $$
同理可定义对 \( y \) 的偏导数 \( f_y(x_0, y_0) \)。

**例 3.2.1** 求 \( z = x^2 + 3xy + y^2 \) 在点 \( (1, 2) \) 处的偏导数。

**解**：
$$ f_x(x, y) = 2x + 3y, \quad f_y(x, y) = 3x + 2y $$
$$ f_x(1, 2) = 2 \times 1 + 3 \times 2 = 8 $$
$$ f_y(1, 2) = 3 \times 1 + 2 \times 2 = 7 $$

### 3.2.2 高阶偏导数

**定义 3.2.2（高阶偏导数）** 设函数 \( z = f(x, y) \) 在区域 \( D \) 内有偏导函数 \( f_x(x, y) \) 和 \( f_y(x, y) \)，如果这两个偏导函数的偏导数也存在，则称它们为 \( z = f(x, y) \) 的**二阶偏导数**：
$$ \frac{\partial^2 z}{\partial x^2} = f_{xx}(x, y), \quad \frac{\partial^2 z}{\partial x \partial y} = f_{xy}(x, y) $$
$$ \frac{\partial^2 z}{\partial y \partial x} = f_{yx}(x, y), \quad \frac{\partial^2 z}{\partial y^2} = f_{yy}(x, y) $$
其中 \( f_{xy} \) 和 \( f_{yx} \) 称为**混合偏导数**。

**定理 3.2.1** 如果函数 \( z = f(x, y) \) 的两个二阶混合偏导数 \( f_{xy}(x, y) \) 和 \( f_{yx}(x, y) \) 在区域 \( D \) 内连续，则在 \( D \) 内必有
$$ f_{xy}(x, y) = f_{yx}(x, y) $$

### 3.2.3 全微分

**定义 3.2.3（全微分）** 设函数 \( z = f(x, y) \) 在点 \( (x, y) \) 的某邻域内有定义，如果函数在点 \( (x, y) \) 的全增量
$$ \Delta z = f(x + \Delta x, y + \Delta y) - f(x, y) $$
可表示为
$$ \Delta z = A \Delta x + B \Delta y + o(\rho) \quad (\rho = \sqrt{(\Delta x)^2 + (\Delta y)^2} \to 0) $$
其中 \( A, B \) 不依赖于 \( \Delta x, \Delta y \)，则称函数 \( z = f(x, y) \) 在点 \( (x, y) \) 处**可微**，而 \( A \Delta x + B \Delta y \) 称为函数在点 \( (x, y) \) 处的**全微分**，记作 \( dz \)，即
$$ dz = A dx + B dy $$

**定理 3.2.2（可微的必要条件）** 如果函数 \( z = f(x, y) \) 在点 \( (x, y) \) 处可微，则该函数在点 \( (x, y) \) 处的偏导数 \( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} \) 必存在，且
$$ dz = \frac{\partial z}{\partial x} dx + \frac{\partial z}{\partial y} dy $$

**定理 3.2.3（可微的充分条件）** 如果函数 \( z = f(x, y) \) 的偏导数 \( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} \) 在点 \( (x, y) \) 处连续，则函数在该点可微。

**例 3.2.2** 求函数 \( z = e^{xy} \) 的全微分。

**解**：
$$ \frac{\partial z}{\partial x} = y e^{xy}, \quad \frac{\partial z}{\partial y} = x e^{xy} $$
$$ dz = y e^{xy} dx + x e^{xy} dy $$

## §3.3 复合函数求导(链式法则)、隐函数求导

### 3.3.1 复合函数求导法则

**定理 3.3.1（链式法则）** 设函数 \( u = \varphi(t), v = \psi(t) \) 都在点 \( t \) 处可导，函数 \( z = f(u, v) \) 在对应点 \( (u, v) \) 处具有连续偏导数，则复合函数 \( z = f[\varphi(t), \psi(t)] \) 在点 \( t \) 处可导，且
$$ \frac{dz}{dt} = \frac{\partial z}{\partial u} \cdot \frac{du}{dt} + \frac{\partial z}{\partial v} \cdot \frac{dv}{dt} $$

**定理 3.3.2** 设 \( u = \varphi(x, y), v = \psi(x, y) \) 都在点 \( (x, y) \) 处具有对 \( x \) 及对 \( y \) 的偏导数，函数 \( z = f(u, v) \) 在对应点 \( (u, v) \) 处具有连续偏导数，则复合函数 \( z = f[\varphi(x, y), \psi(x, y)] \) 在点 \( (x, y) \) 处的两个偏导数存在，且
$$ \frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x} $$
$$ \frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial y} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial y} $$

**例 3.3.1** 设 \( z = e^u \sin v \)，而 \( u = xy, v = x + y \)，求 \( \frac{\partial z}{\partial x} \) 和 \( \frac{\partial z}{\partial y} \)。

**解**：
$$ \frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x} = e^u \sin v \cdot y + e^u \cos v \cdot 1 = e^{xy} [y \sin(x + y) + \cos(x + y)] $$
$$ \frac{\partial z}{\partial y} = e^u \sin v \cdot x + e^u \cos v \cdot 1 = e^{xy} [x \sin(x + y) + \cos(x + y)] $$

### 3.3.2 隐函数求导

**定理 3.3.3（隐函数存在定理）** 设函数 \( F(x, y) \) 在点 \( P_0(x_0, y_0) \) 的某一邻域内具有连续偏导数，且 \( F(x_0, y_0) = 0, F_y(x_0, y_0) \neq 0 \)，则方程 \( F(x, y) = 0 \) 在点 \( (x_0, y_0) \) 的某一邻域内恒能唯一确定一个单值连续且具有连续导数的函数 \( y = f(x) \)，满足 \( y_0 = f(x_0) \)，且
$$ \frac{dy}{dx} = - \frac{F_x}{F_y} $$

**定理 3.3.4（隐函数组存在定理）** 设 \( F(x, y, z) \) 在点 \( P_0(x_0, y_0, z_0) \) 的某一邻域内具有连续偏导数，且 \( F(x_0, y_0, z_0) = 0, F_z(x_0, y_0, z_0) \neq 0 \)，则方程 \( F(x, y, z) = 0 \) 在点 \( (x_0, y_0, z_0) \) 的某一邻域内恒能唯一确定一个单值连续且具有连续偏导数的函数 \( z = f(x, y) \)，满足 \( z_0 = f(x_0, y_0) \)，且
$$ \frac{\partial z}{\partial x} = - \frac{F_x}{F_z}, \quad \frac{\partial z}{\partial y} = - \frac{F_y}{F_z} $$

**例 3.3.2** 求由方程 \( e^z - xyz = 0 \) 确定的隐函数 \( z = f(x, y) \) 的偏导数。

**解**：设 \( F(x, y, z) = e^z - xyz \)，则
$$ F_x = -yz, \quad F_y = -xz, \quad F_z = e^z - xy $$
故
$$ \frac{\partial z}{\partial x} = - \frac{F_x}{F_z} = \frac{yz}{e^z - xy}, \quad \frac{\partial z}{\partial y} = \frac{xz}{e^z - xy} $$

## §3.4 方向导数、梯度、几何意义

### 3.4.1 方向导数

**定义 3.4.1（方向导数）** 设函数 \( z = f(x, y) \) 在点 \( P_0(x_0, y_0) \) 的某一邻域 \( U(P_0) \) 内有定义，\( \boldsymbol{l} \) 是从 \( P_0 \) 出发的射线，\( P(x_0 + t \cos \alpha, y_0 + t \cos \beta) \) 是 \( \boldsymbol{l} \) 上且在 \( U(P_0) \) 内的点。如果极限
$$ \lim_{t \to 0^+} \frac{f(x_0 + t \cos \alpha, y_0 + t \cos \beta) - f(x_0, y_0)}{t} $$
存在，则称此极限为函数 \( f(x, y) \) 在点 \( P_0 \) 沿方向 \( \boldsymbol{l} \) 的**方向导数**，记作 \( \frac{\partial f}{\partial \boldsymbol{l}}\bigg|_{P_0} \)。

**定理 3.4.1** 如果函数 \( f(x, y) \) 在点 \( P_0(x_0, y_0) \) 处可微，则函数在该点沿任一方向 \( \boldsymbol{l} \) 的方向导数存在，且
$$ \frac{\partial f}{\partial \boldsymbol{l}}\bigg|_{P_0} = f_x(x_0, y_0) \cos \alpha + f_y(x_0, y_0) \cos \beta $$
其中 \( \cos \alpha, \cos \beta \) 是方向 \( \boldsymbol{l} \) 的方向余弦。

### 3.4.2 梯度

**定义 3.4.2（梯度）** 设函数 \( z = f(x, y) \) 在平面区域 \( D \) 内具有一阶连续偏导数，则对于每一点 \( P(x, y) \in D \)，都可定出一个向量
$$ \text{grad} f(x, y) = \frac{\partial f}{\partial x} \boldsymbol{i} + \frac{\partial f}{\partial y} \boldsymbol{j} $$
这个向量称为函数 \( z = f(x, y) \) 在点 \( P(x, y) \) 的**梯度**，记作 \( \nabla f(x, y) \)。

**梯度的几何意义** 方向导数等于梯度在该方向上的投影，即
$$ \frac{\partial f}{\partial \boldsymbol{l}} = \text{grad} f \cdot \boldsymbol{e}_l = |\text{grad} f| \cos \theta $$
其中 \( \theta \) 是梯度与方向 \( \boldsymbol{l} \) 的夹角。当 \( \boldsymbol{l} \) 与梯度同方向时，方向导数取得最大值 \( |\text{grad} f| \)。

**例 3.4.1** 求函数 \( f(x, y, z) = x^2 + y^2 + z^2 \) 在点 \( (1, 2, 3) \) 的梯度。

**解**：
$$ f_x = 2x, \quad f_y = 2y, \quad f_z = 2z $$
$$ \text{grad} f(1, 2, 3) = (2, 4, 6) $$

### 3.4.3 几何意义

1. **曲面的切平面与法线**
设曲面 \( \Sigma \) 的方程为 \( F(x, y, z) = 0 \)，\( M_0(x_0, y_0, z_0) \) 是 \( \Sigma \) 上的点，且 \( F_x, F_y, F_z \) 在该点不同时为零。则曲面 \( \Sigma \) 在点 \( M_0 \) 处的**切平面方程**为
$$ F_x(x_0, y_0, z_0)(x - x_0) + F_y(x_0, y_0, z_0)(y - y_0) + F_z(x_0, y_0, z_0)(z - z_0) = 0 $$
**法线方程**为
$$ \frac{x - x_0}{F_x(x_0, y_0, z_0)} = \frac{y - y_0}{F_y(x_0, y_0, z_0)} = \frac{z - z_0}{F_z(x_0, y_0, z_0)} $$

2. **空间曲线的切线与法平面**
设空间曲线 \( \Gamma \) 的参数方程为 \( x = \varphi(t), y = \psi(t), z = \omega(t) \)，\( t = t_0 \) 对应点 \( M_0(x_0, y_0, z_0) \)，且 \( \varphi'(t_0), \psi'(t_0), \omega'(t_0) \) 不全为零。则曲线 \( \Gamma \) 在点 \( M_0 \) 处的**切线方程**为
$$ \frac{x - x_0}{\varphi'(t_0)} = \frac{y - y_0}{\psi'(t_0)} = \frac{z - z_0}{\omega'(t_0)} $$
**法平面方程**为
$$ \varphi'(t_0)(x - x_0) + \psi'(t_0)(y - y_0) + \omega'(t_0)(z - z_0) = 0 $$

## §3.5 多元函数的泰勒公式

**定理 3.5.1（二元函数的泰勒定理）** 设函数 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 的某一邻域内连续且有直到 \( n + 1 \) 阶的连续偏导数，\( (x_0 + h, y_0 + k) \) 为此邻域内任一点，则有
$$ f(x_0 + h, y_0 + k) = f(x_0, y_0) + \left( h \frac{\partial}{\partial x} + k \frac{\partial}{\partial y} \right) f(x_0, y_0) + \frac{1}{2!} \left( h \frac{\partial}{\partial x} + k \frac{\partial}{\partial y} \right)^2 f(x_0, y_0) + \dots + \frac{1}{n!} \left( h \frac{\partial}{\partial x} + k \frac{\partial}{\partial y} \right)^n f(x_0, y_0) + R_n $$
其中
$$ R_n = \frac{1}{(n + 1)!} \left( h \frac{\partial}{\partial x} + k \frac{\partial}{\partial y} \right)^{n + 1} f(x_0 + \theta h, y_0 + \theta k) \quad (0 < \theta < 1) $$
记号
$$ \left( h \frac{\partial}{\partial x} + k \frac{\partial}{\partial y} \right)^m f(x_0, y_0) = \sum_{p = 0}^m \binom{m}{p} h^p k^{m - p} \frac{\partial^m f}{\partial x^p \partial y^{m - p}}\bigg|_{(x_0, y_0)} $$

**例 3.5.1** 求函数 \( f(x, y) = e^{x + y} \) 在点 \( (0, 0) \) 的泰勒公式。

**解**：由于 \( f(0, 0) = 1 \)，且各阶偏导数均为 \( e^{x + y} \)，在 \( (0, 0) \) 处的值都是 1，故
$$ e^{x + y} = 1 + (x + y) + \frac{1}{2!}(x + y)^2 + \dots + \frac{1}{n!}(x + y)^n + R_n $$
$$ R_n = \frac{1}{(n + 1)!}(x + y)^{n + 1} e^{\theta(x + y)} \quad (0 < \theta < 1) $$

## §3.6 多元函数的极值

### 3.6.1 无条件极值

**定义 3.6.1（极值）** 设函数 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 的某邻域内有定义，如果对于该邻域内异于 \( (x_0, y_0) \) 的任意点 \( (x, y) \)，都有
$$ f(x, y) < f(x_0, y_0) \quad (\text{或} \quad f(x, y) > f(x_0, y_0)) $$
则称函数 \( f(x, y) \) 在点 \( (x_0, y_0) \) 处有**极大值**（或**极小值**），\( (x_0, y_0) \) 称为**极大值点**（或**极小值点**）。

**定理 3.6.1（极值的必要条件）** 设函数 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 处具有偏导数，且在该点处取得极值，则有
$$ f_x(x_0, y_0) = 0, \quad f_y(x_0, y_0) = 0 $$

**定理 3.6.2（极值的充分条件）** 设函数 \( z = f(x, y) \) 在点 \( (x_0, y_0) \) 的某邻域内连续且有一阶及二阶连续偏导数，又 \( f_x(x_0, y_0) = 0, f_y(x_0, y_0) = 0 \)，令
$$ A = f_{xx}(x_0, y_0), \quad B = f_{xy}(x_0, y_0), \quad C = f_{yy}(x_0, y_0) $$
则 \( f(x, y) \) 在 \( (x_0, y_0) \) 处：
1. 当 \( AC - B^2 > 0 \) 时，具有极值，且当 \( A < 0 \) 时为极大值，\( A > 0 \) 时为极小值；
2. 当 \( AC - B^2 < 0 \) 时，没有极值；
3. 当 \( AC - B^2 = 0 \) 时，可能有极值，也可能没有极值。

**例 3.6.1** 求函数 \( f(x, y) = x^3 - y^3 + 3x^2 + 3y^2 - 9x \) 的极值。

**解**：解方程组
$$ f_x = 3x^2 + 6x - 9 = 0, \quad f_y = -3y^2 + 6y = 0 $$
得驻点 \( (1, 0), (1, 2), (-3, 0), (-3, 2) \)。

二阶偏导数：
$$ A = f_{xx} = 6x + 6, \quad B = f_{xy} = 0, \quad C = f_{yy} = -6y + 6 $$

在点 \( (1, 0) \)：\( AC - B^2 = 12 \times 6 = 72 > 0, A > 0 \)，故有极小值 \( f(1, 0) = -5 \)；
在点 \( (1, 2) \)：\( AC - B^2 = 12 \times (-6) = -72 < 0 \)，无极值；
在点 \( (-3, 0) \)：\( AC - B^2 = -12 \times 6 = -72 < 0 \)，无极值；
在点 \( (-3, 2) \)：\( AC - B^2 = -12 \times (-6) = 72 > 0, A < 0 \)，故有极大值 \( f(-3, 2) = 31 \)。

### 3.6.2 条件极值——拉格朗日乘数法

**拉格朗日乘数法** 求函数 \( z = f(x, y) \) 在约束条件 \( \varphi(x, y) = 0 \) 下的极值，构造**拉格朗日函数**
$$ L(x, y, \lambda) = f(x, y) + \lambda \varphi(x, y) $$
其中 \( \lambda \) 为**拉格朗日乘数**。解方程组
$$ \begin{cases} L_x = f_x(x, y) + \lambda \varphi_x(x, y) = 0 \\ L_y = f_y(x, y) + \lambda \varphi_y(x, y) = 0 \\ L_\lambda = \varphi(x, y) = 0 \end{cases} $$
由方程组解出 \( x, y, \lambda \)，则 \( (x, y) \) 就是可能的极值点。

**例 3.6.2** 求表面积为 \( a^2 \) 而体积最大的长方体的体积。

**解**：设长方体的长、宽、高为 \( x, y, z \)，则体积 \( V = xyz \)，表面积 \( 2(xy + yz + zx) = a^2 \)。构造拉格朗日函数
$$ L = xyz + \lambda(2xy + 2yz + 2zx - a^2) $$
解方程组
$$ L_x = yz + 2\lambda(y + z) = 0, \quad L_y = xz + 2\lambda(x + z) = 0, \quad L_z = xy + 2\lambda(x + y) = 0, \quad 2xy + 2yz + 2zx = a^2 $$
得 \( x = y = z = \frac{a}{\sqrt{6}} \)，最大体积为 \( V = \frac{a^3}{6\sqrt{6}} \)。

## §3.7 多元函数微分学在AI中的应用

### 3.7.1 多元优化

在人工智能中，许多问题可以归结为**多元函数优化问题，即寻找目标函数 \( f(\boldsymbol{x}) \) 的最小值（或最大值），其中 \( \boldsymbol{x} = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n \)。

**梯度下降法** 是最常用的优化算法之一，其核心思想是沿着函数值下降最快的方向（即负梯度方向）迭代更新参数：
$$ \boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \alpha \nabla f(\boldsymbol{x}_k) $$
其中 \( \alpha > 0 \) 称为**学习率**。

**随机梯度下降（SGD）** 在每次迭代中只使用一个样本或一小批样本计算梯度，提高了计算效率。

### 3.7.2 神经网络反向传播

**反向传播算法** 是训练神经网络的核心方法，本质上是链式法则的巧妙应用。设神经网络的损失函数 \( L(\boldsymbol{W}, \boldsymbol{b}) \) 是关于权重 \( \boldsymbol{W} \) 和偏置 \( \boldsymbol{b} \) 的多元函数，通过链式法则逐层计算梯度：
$$ \frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}} $$
然后使用梯度下降法更新参数。

**反向传播的步骤**：
1. **前向传播**：计算网络的输出
2. **计算损失**：比较预测输出与真实标签
3. **反向传播误差**：从输出层开始，逐层向后传播误差信号
4. **更新参数**：使用梯度下降法更新权重和偏置

多元函数微分学为神经网络的训练提供了坚实的数学基础，使得我们能够高效地优化复杂的神经网络模型。
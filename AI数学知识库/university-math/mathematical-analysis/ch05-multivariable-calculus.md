# 第5章 多元函数微积分

## 1. 多元函数的极限与连续性

### 1.1 多元函数

**定义 1.1 (n元函数)** 设 \( D \subset \mathbb{R}^n \) 是一个非空点集, 若存在一个对应法则 \( f \), 使得对每个点 \( P(x_1,x_2,\dots,x_n) \in D \), 都有唯一确定的实数 \( z \) 与之对应, 则称 \( f \) 是定义在 \( D \) 上的 \( n \) 元函数, 记为
\[ z = f(x_1,x_2,\dots,x_n) \quad \text{或} \quad z = f(P) \]
其中 \( D \) 称为定义域, \( x_1,x_2,\dots,x_n \) 称为自变量, \( z \) 称为因变量.

**几何意义:** 二元函数 \( z = f(x,y) \) 表示空间直角坐标系中的一张曲面.

### 1.2 重极限

**定义 1.2 (重极限)** 设二元函数 \( f(x,y) \) 在点 \( P_0(x_0,y_0) \) 的某去心邻域内有定义, 若对任意给定的 \( \varepsilon > 0 \), 存在 \( \delta > 0 \), 使得当 \( 0 < \sqrt{(x-x_0)^2 + (y-y_0)^2} < \delta \) 时, 有
\[ |f(x,y) - A| < \varepsilon \]
则称 \( A \) 为 \( f(x,y) \) 当 \( (x,y) \to (x_0,y_0) \) 时的重极限, 记为
\[ \lim_{(x,y) \to (x_0,y_0)} f(x,y) = A \quad \text{或} \quad \lim_{P \to P_0} f(P) = A \]

**注:** 重极限存在的充要条件是: 点 \( P(x,y) \) 以任意方式趋近于 \( P_0(x_0,y_0) \) 时, \( f(x,y) \) 都趋于同一个极限.

**例 1.1** 讨论 \( \lim_{(x,y) \to (0,0)} \frac{xy}{x^2 + y^2} \) 的存在性.

令 \( y = kx \), 则
\[ \lim_{(x,y) \to (0,0)} \frac{xy}{x^2 + y^2} = \lim_{x \to 0} \frac{kx^2}{x^2 + k^2x^2} = \frac{k}{1 + k^2} \]
极限值随 \( k \) 变化而变化, 故重极限不存在.

### 1.3 累次极限

**定义 1.3 (累次极限)** 先固定 \( y \), 令 \( x \to x_0 \), 得极限 \( \lim_{x \to x_0} f(x,y) = \phi(y) \), 再令 \( y \to y_0 \), 得 \( \lim_{y \to y_0} \phi(y) = A \), 则称 \( A \) 为先 \( x \to x_0 \) 后 \( y \to y_0 \) 的累次极限, 记为
\[ \lim_{y \to y_0} \lim_{x \to x_0} f(x,y) = A \]

**定理 1.1** 若重极限 \( \lim_{(x,y) \to (x_0,y_0)} f(x,y) = A \) 存在, 且两个累次极限都存在, 则三者相等.

### 1.4 多元函数的连续性

**定义 1.4 (连续性)** 设 \( f(x,y) \) 在 \( P_0(x_0,y_0) \) 的某邻域内有定义, 若
\[ \lim_{(x,y) \to (x_0,y_0)} f(x,y) = f(x_0,y_0) \]
则称 \( f \) 在 \( P_0 \) 连续.

**性质:**
1. 连续函数的和、差、积、商(分母不为零)仍连续
2. 连续函数的复合仍连续
3. 有界闭区域上的连续函数必有最大值和最小值
4. 有界闭区域上的连续函数必一致连续

## 2. 偏导数、全微分、链式法则

### 2.1 偏导数

**定义 2.1 (偏导数)** 设 \( z = f(x,y) \) 在 \( (x_0,y_0) \) 的某邻域内有定义, 固定 \( y = y_0 \), 若一元函数 \( f(x,y_0) \) 在 \( x = x_0 \) 处可导, 则称该导数为 \( f \) 在 \( (x_0,y_0) \) 处对 \( x \) 的偏导数, 记为
\[ \frac{\partial z}{\partial x}\bigg|_{(x_0,y_0)}, \quad f_x(x_0,y_0), \quad \text{或} \quad \frac{\partial f}{\partial x}\bigg|_{(x_0,y_0)} \]
即
\[ f_x(x_0,y_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x} \]

**几何意义:** \( f_x(x_0,y_0) \) 是曲面 \( z = f(x,y) \) 与平面 \( y = y_0 \) 的交线在点 \( (x_0,y_0,f(x_0,y_0)) \) 处的切线对 \( x \) 轴的斜率.

**定理 2.1 (混合偏导数相等)** 若 \( f_{xy} \) 和 \( f_{yx} \) 在区域 \( D \) 内连续, 则在 \( D \) 内 \( f_{xy} = f_{yx} \).

### 2.2 全微分

**定义 2.2 (全微分)** 设 \( z = f(x,y) \) 在 \( (x,y) \) 的某邻域内有定义, 若全增量
\[ \Delta z = f(x + \Delta x, y + \Delta y) - f(x, y) \]
可表示为
\[ \Delta z = A\Delta x + B\Delta y + o(\rho) \quad (\rho = \sqrt{(\Delta x)^2 + (\Delta y)^2} \to 0) \]
其中 \( A, B \) 与 \( \Delta x, \Delta y \) 无关, 则称 \( f \) 在 \( (x,y) \) 处可微, 称 \( A\Delta x + B\Delta y \) 为全微分, 记为
\[ dz = A dx + B dy \]

**定理 2.2 (可微的必要条件)** 若 \( f \) 在 \( (x,y) \) 处可微, 则
1. \( f \) 在该点连续
2. 偏导数 \( f_x, f_y \) 存在, 且 \( dz = f_x dx + f_y dy \)

**定理 2.3 (可微的充分条件)** 若 \( f_x, f_y \) 在 \( (x,y) \) 处连续, 则 \( f \) 在该点可微.

### 2.3 链式法则

**定理 2.4 (链式法则)** 设 \( u = \phi(t), v = \psi(t) \) 在 \( t \) 处可导, \( z = f(u,v) \) 在对应点 \( (u,v) \) 处可微, 则复合函数 \( z = f(\phi(t), \psi(t)) \) 在 \( t \) 处可导, 且
\[ \frac{dz}{dt} = \frac{\partial z}{\partial u} \frac{du}{dt} + \frac{\partial z}{\partial v} \frac{dv}{dt} \]

**一般形式:** 设 \( z = f(u,v), u = \phi(x,y), v = \psi(x,y) \), 则
\[ \frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \frac{\partial v}{\partial x} \]
\[ \frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial z}{\partial v} \frac{\partial v}{\partial y} \]

**例 2.1** 设 \( z = e^u \sin v, u = xy, v = x + y \), 求 \( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} \).

解:
\[ \frac{\partial z}{\partial x} = e^u \sin v \cdot y + e^u \cos v \cdot 1 = e^{xy}[y \sin(x+y) + \cos(x+y)] \]
\[ \frac{\partial z}{\partial y} = e^u \sin v \cdot x + e^u \cos v \cdot 1 = e^{xy}[x \sin(x+y) + \cos(x+y)] \]

## 3. 方向导数与梯度

### 3.1 方向导数

**定义 3.1 (方向导数)** 设 \( f(x,y) \) 在点 \( P_0(x_0,y_0) \) 的某邻域内有定义, \( \mathbf{l} = (\cos\alpha, \cos\beta) \) 是单位向量, 若极限
\[ \lim_{t \to 0^+} \frac{f(x_0 + t\cos\alpha, y_0 + t\cos\beta) - f(x_0, y_0)}{t} \]
存在, 则称此极限为 \( f \) 在 \( P_0 \) 处沿方向 \( \mathbf{l} \) 的方向导数, 记为 \( \frac{\partial f}{\partial \mathbf{l}}\bigg|_{P_0} \).

**定理 3.1** 若 \( f \) 在 \( P_0 \) 处可微, 则沿任一方向 \( \mathbf{l} \) 的方向导数存在, 且
\[ \frac{\partial f}{\partial \mathbf{l}} = f_x \cos\alpha + f_y \cos\beta \]

### 3.2 梯度

**定义 3.2 (梯度)** 设 \( f(x,y) \) 具有一阶连续偏导数, 称向量
\[ \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) = \frac{\partial f}{\partial x} \mathbf{i} + \frac{\partial f}{\partial y} \mathbf{j} \]
为 \( f \) 的梯度, 记为 \( \nabla f \) 或 \( \text{grad} f \).

**性质:**
1. 方向导数等于梯度在方向 \( \mathbf{l} \) 上的投影: \( \frac{\partial f}{\partial \mathbf{l}} = \nabla f \cdot \mathbf{l} \)
2. 梯度方向是函数值增加最快的方向
3. 梯度的模是最大方向导数的值

**例 3.1** 求 \( f(x,y) = x^2 + y^2 \) 在 \( (1,2) \) 处的梯度.

解: \( \nabla f = (2x, 2y) \), 故 \( \nabla f(1,2) = (2,4) \).

## 4. 隐函数定理与逆函数定理

### 4.1 隐函数定理

**定理 4.1 (隐函数存在定理)** 设 \( F(x,y) \) 满足:
1. 在 \( P_0(x_0,y_0) \) 的某邻域内具有一阶连续偏导数
2. \( F(x_0,y_0) = 0 \)
3. \( F_y(x_0,y_0) \neq 0 \)

则方程 \( F(x,y) = 0 \) 在 \( x_0 \) 的某邻域内唯一确定一个具有连续导数的函数 \( y = f(x) \), 满足 \( y_0 = f(x_0) \), 且
\[ \frac{dy}{dx} = -\frac{F_x}{F_y} \]

**推广:** 对于 \( F(x,y,z) = 0 \), 若 \( F_z \neq 0 \), 则
\[ \frac{\partial z}{\partial x} = -\frac{F_x}{F_z}, \quad \frac{\partial z}{\partial y} = -\frac{F_y}{F_z} \]

### 4.2 逆函数定理

**定理 4.2 (逆函数定理)** 设变换 \( \begin{cases} u = u(x,y) \\ v = v(x,y) \end{cases} \) 满足:
1. 在 \( P_0(x_0,y_0) \) 的某邻域内具有一阶连续偏导数
2. \( u_0 = u(x_0,y_0), v_0 = v(x_0,y_0) \)
3. Jacobi行列式 \( J = \frac{\partial(u,v)}{\partial(x,y)} = \begin{vmatrix} u_x & u_y \\ v_x & v_y \end{vmatrix} \neq 0 \) 在 \( P_0 \) 处

则存在逆变换 \( \begin{cases} x = x(u,v) \\ y = y(u,v) \end{cases} \) 在 \( (u_0,v_0) \) 的某邻域内唯一确定, 且
\[ \frac{\partial(x,y)}{\partial(u,v)} = \frac{1}{\frac{\partial(u,v)}{\partial(x,y)}} \]

## 5. 多元函数的极值

### 5.1 无条件极值

**定义 5.1 (极值)** 设 \( f \) 在 \( P_0 \) 的某邻域内有定义, 若对该邻域内任意 \( P \neq P_0 \), 有 \( f(P) < f(P_0) \) (或 \( f(P) > f(P_0) \)), 则称 \( f(P_0) \) 为极大值 (或极小值).

**定理 5.1 (极值的必要条件)** 设 \( f \) 在 \( P_0(x_0,y_0) \) 处具有一阶偏导数, 且在 \( P_0 \) 处取得极值, 则
\[ f_x(x_0,y_0) = 0, \quad f_y(x_0,y_0) = 0 \]
称满足上式的点为驻点.

**定理 5.2 (极值的充分条件)** 设 \( f \) 在 \( P_0(x_0,y_0) \) 的某邻域内具有二阶连续偏导数, 且 \( f_x(P_0) = f_y(P_0) = 0 \). 令
\[ A = f_{xx}(P_0), \quad B = f_{xy}(P_0), \quad C = f_{yy}(P_0) \]
\[ \Delta = AC - B^2 \]
则:
1. 若 \( \Delta > 0 \), 则 \( f \) 在 \( P_0 \) 处有极值, 且 \( A > 0 \) 时极小, \( A < 0 \) 时极大
2. 若 \( \Delta < 0 \), 则 \( f \) 在 \( P_0 \) 处无极值
3. 若 \( \Delta = 0 \), 无法判定

**例 5.1** 求 \( f(x,y) = x^3 - y^3 + 3x^2 + 3y^2 - 9x \) 的极值.

解: 解方程组 \( f_x = 3x^2 + 6x - 9 = 0, f_y = -3y^2 + 6y = 0 \), 得驻点 \( (1,0), (1,2), (-3,0), (-3,2) \).

计算二阶偏导数: \( A = 6x + 6, B = 0, C = -6y + 6 \).

对 \( (1,0) \): \( \Delta = 12 \cdot 6 > 0, A > 0 \), 极小值 \( f(1,0) = -5 \)
对 \( (1,2) \): \( \Delta = 12 \cdot (-6) < 0 \), 非极值
对 \( (-3,0) \): \( \Delta = (-12) \cdot 6 < 0 \), 非极值
对 \( (-3,2) \): \( \Delta = (-12) \cdot (-6) > 0, A < 0 \), 极大值 \( f(-3,2) = 31 \)

### 5.2 条件极值与拉格朗日乘数法

**问题:** 在约束条件 \( \phi(x,y) = 0 \) 下求 \( f(x,y) \) 的极值.

**拉格朗日乘数法:** 构造拉格朗日函数
\[ L(x,y,\lambda) = f(x,y) + \lambda \phi(x,y) \]
解方程组
\[ \begin{cases} L_x = f_x + \lambda \phi_x = 0 \\ L_y = f_y + \lambda \phi_y = 0 \\ L_\lambda = \phi(x,y) = 0 \end{cases} \]

**推广:** 多个约束条件 \( \phi_1 = 0, \dots, \phi_m = 0 \) 下, 构造
\[ L = f + \lambda_1 \phi_1 + \dots + \lambda_m \phi_m \]

**例 5.2** 求表面积为 \( a^2 \) 的长方体的最大体积.

解: 设长宽高为 \( x,y,z \), 则体积 \( V = xyz \), 约束 \( 2(xy + yz + zx) = a^2 \).

构造 \( L = xyz + \lambda(2xy + 2yz + 2zx - a^2) \).

解得 \( x = y = z = \frac{a}{\sqrt{6}} \), 最大体积 \( V = \frac{a^3}{6\sqrt{6}} \).

## 6. 重积分

### 6.1 二重积分

**定义 6.1 (二重积分)** 设 \( f(x,y) \) 是有界闭区域 \( D \) 上的有界函数, 将 \( D \) 任意分成 \( n \) 个小区域 \( \Delta \sigma_1, \dots, \Delta \sigma_n \), 在每个 \( \Delta \sigma_i \) 上任取一点 \( (\xi_i, \eta_i) \), 作和 \( \sum_{i=1}^n f(\xi_i, \eta_i) \Delta \sigma_i \). 若当各小区域直径的最大值 \( \lambda \to 0 \) 时, 该和的极限存在, 则称此极限为 \( f \) 在 \( D \) 上的二重积分, 记为
\[ \iint_D f(x,y) d\sigma = \lim_{\lambda \to 0} \sum_{i=1}^n f(\xi_i, \eta_i) \Delta \sigma_i \]

**几何意义:** 当 \( f(x,y) \geq 0 \) 时, 二重积分表示以 \( z = f(x,y) \) 为顶, \( D \) 为底的曲顶柱体体积.

**性质:**
1. 线性性: \( \iint_D (kf + lg) d\sigma = k \iint_D f d\sigma + l \iint_D g d\sigma \)
2. 可加性: \( D = D_1 \cup D_2, D_1 \cap D_2 = \emptyset \), 则 \( \iint_D f d\sigma = \iint_{D_1} f d\sigma + \iint_{D_2} f d\sigma \)
3. 若在 \( D \) 上 \( f \leq g \), 则 \( \iint_D f d\sigma \leq \iint_D g d\sigma \)
4. 中值定理: 若 \( f \) 连续, 则存在 \( (\xi,\eta) \in D \), 使得 \( \iint_D f d\sigma = f(\xi,\eta) \cdot S_D \), \( S_D \) 是 \( D \) 的面积

### 6.2 二重积分的计算

**直角坐标下:**
1. \( X \)-型区域 \( D: a \leq x \leq b, \phi_1(x) \leq y \leq \phi_2(x) \)
   \[ \iint_D f(x,y) d\sigma = \int_a^b dx \int_{\phi_1(x)}^{\phi_2(x)} f(x,y) dy \]
2. \( Y \)-型区域 \( D: c \leq y \leq d, \psi_1(y) \leq x \leq \psi_2(y) \)
   \[ \iint_D f(x,y) d\sigma = \int_c^d dy \int_{\psi_1(y)}^{\psi_2(y)} f(x,y) dx \]

**极坐标下:** 令 \( x = r\cos\theta, y = r\sin\theta \), \( d\sigma = r dr d\theta \)
\[ \iint_D f(x,y) d\sigma = \iint_{D'} f(r\cos\theta, r\sin\theta) r dr d\theta \]

**例 6.1** 计算 \( \iint_D e^{-x^2 - y^2} d\sigma \), \( D: x^2 + y^2 \leq a^2 \).

解: 极坐标下 \( D': 0 \leq r \leq a, 0 \leq \theta \leq 2\pi \)
\[ \iint_D e^{-x^2 - y^2} d\sigma = \int_0^{2\pi} d\theta \int_0^a e^{-r^2} r dr = \pi(1 - e^{-a^2}) \]

### 6.3 三重积分

**定义 6.2 (三重积分)** 类似二重积分,
\[ \iiint_\Omega f(x,y,z) dV = \lim_{\lambda \to 0} \sum_{i=1}^n f(\xi_i, \eta_i, \zeta_i) \Delta V_i \]

**计算方法:**
1. 直角坐标: 先一后二或先二后一
2. 柱坐标: \( x = r\cos\theta, y = r\sin\theta, z = z \), \( dV = r dr d\theta dz \)
3. 球坐标: \( x = r\sin\varphi\cos\theta, y = r\sin\varphi\sin\theta, z = r\cos\varphi \), \( dV = r^2 \sin\varphi dr d\varphi d\theta \)

## 7. 曲线积分与曲面积分

### 7.1 曲线积分

**第一类曲线积分(对弧长):**
\[ \int_L f(x,y) ds = \lim_{\lambda \to 0} \sum_{i=1}^n f(\xi_i, \eta_i) \Delta s_i \]
计算: \( L: x = \phi(t), y = \psi(t), \alpha \leq t \leq \beta \)
\[ \int_L f(x,y) ds = \int_\alpha^\beta f(\phi(t), \psi(t)) \sqrt{\phi'^2(t) + \psi'^2(t)} dt \]

**第二类曲线积分(对坐标):**
\[ \int_L P(x,y) dx + Q(x,y) dy = \lim_{\lambda \to 0} \sum_{i=1}^n [P(\xi_i, \eta_i) \Delta x_i + Q(\xi_i, \eta_i) \Delta y_i] \]
计算:
\[ \int_L P dx + Q dy = \int_\alpha^\beta [P(\phi(t), \psi(t)) \phi'(t) + Q(\phi(t), \psi(t)) \psi'(t)] dt \]

### 7.2 格林公式

**定理 7.1 (格林公式)** 设 \( D \) 是由分段光滑的闭曲线 \( L \) 围成的闭区域, \( P, Q \) 在 \( D \) 上具有一阶连续偏导数, 则
\[ \oint_L P dx + Q dy = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) d\sigma \]
其中 \( L \) 取正向.

**推论:** 区域 \( D \) 的面积 \( A = \frac{1}{2} \oint_L x dy - y dx \).

### 7.3 曲面积分

**第一类曲面积分(对面积):**
\[ \iint_\Sigma f(x,y,z) dS = \lim_{\lambda \to 0} \sum_{i=1}^n f(\xi_i, \eta_i, \zeta_i) \Delta S_i \]
计算: \( \Sigma: z = z(x,y) \)
\[ \iint_\Sigma f dS = \iint_{D_{xy}} f(x,y,z(x,y)) \sqrt{1 + z_x^2 + z_y^2} dxdy \]

**第二类曲面积分(对坐标):**
\[ \iint_\Sigma P dydz + Q dzdx + R dxdy \]

### 7.4 高斯公式

**定理 7.2 (高斯公式)** 设空间闭区域 \( \Omega \) 由分片光滑的闭曲面 \( \Sigma \) 围成, \( P, Q, R \) 在 \( \Omega \) 上具有一阶连续偏导数, 则
\[ \oiint_\Sigma P dydz + Q dzdx + R dxdy = \iiint_\Omega \left( \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z} \right) dV \]
其中 \( \Sigma \) 取外侧.

### 7.5 斯托克斯公式

**定理 7.3 (斯托克斯公式)** 设 \( \Sigma \) 是分片光滑的有向曲面, \( \Gamma \) 是 \( \Sigma \) 的边界曲线, \( P, Q, R \) 在包含 \( \Sigma \) 的空间区域内具有一阶连续偏导数, 则
\[ \oint_\Gamma P dx + Q dy + R dz = \iint_\Sigma \begin{vmatrix} dydz & dzdx & dxdy \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix} \]

## 8. 多元函数微积分在AI中的应用

### 8.1 多元优化

**梯度下降法:** 目标是求函数 \( f(\mathbf{x}) \) 的最小值, 迭代公式
\[ \mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k) \]
其中 \( \eta > 0 \) 是学习率.

**牛顿法:** 利用二阶导数信息, 迭代公式
\[ \mathbf{x}_{k+1} = \mathbf{x}_k - H_f(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k) \]
其中 \( H_f \) 是Hessian矩阵.

**拉格朗日乘数法在SVM中的应用:** 支持向量机通过最大化间隔转化为带约束的优化问题, 利用拉格朗日对偶性求解.

### 8.2 流形理解

**流形:** 局部欧氏空间的推广, 是AI中数据常位于低维流形上.

**微分几何在深度学习:** 利用微分几何工具研究神经网络的损失曲面几何性质.

**散度与梯度在生成模型:** 信息论散度的梯度流用于生成模型训练.

**积分变换在信号处理:** 傅里叶变换、小波变换等用于特征提取.

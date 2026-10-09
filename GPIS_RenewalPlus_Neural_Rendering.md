# 平稳 GPIS 的 Renewal+ 神经渲染方案

**累计 hazard 网络、一维首达导数 Gaussian mixture，以及解析梯度与法线采样**

版本：2026-10-08。本文整理当前讨论中的方案，给出可实现的数学定义、数据生成流程、网络配置、训练目标与渲染算法。网络容量、混合分量数和步长是实验起始配置，不是已经验证的最优结果。

## 1. 目标与适用范围

目标是在不于渲染阶段显式生成 GP 实现的情况下，近似采样：

1. 射线与 GPIS 的首次碰撞距离；
2. 首达点的沿射线导数；
3. 首达点的完整梯度与法线；
4. 从已知碰撞点出发的下一次碰撞，以及有限长度射线的透射率。

采用以下分工：

- **神经网络**：近似累计 hazard，以及首达条件下的一维穿越速度分布。
- **高斯过程解析统计**：重建两个横向梯度分量。
- **Renewal+ 记忆策略**：真实碰撞后保留最近碰撞点的场值和完整梯度，丢弃更早光路的条件信息。
- **当前直线射线内的历史状态**：逐段传播，近似表达“截至当前位置尚未碰撞”造成的统计偏置。

主方案适用于足够光滑的**平稳各向同性残差核**。确定性 SDF 均值可以任意随空间变化，并以体素三线性插值表示。第 15 节扩展到固定椭圆各向异性核。

“平稳”本身不足以保证只学习一维导数就能重建完整梯度。一般平稳核仍须检查横向梯度与整条射线路径的条件独立性。本文不将该独立性推广到所有平稳核。

本文以在 $F>0$ 空域中传播、碰撞后向空域散射的射线为主。GP 场本身不必是严格满足 Eikonal 方程的距离场；只有其确定性均值来自 SDF 表示。

## 2. 随机场、符号与单位

定义随机隐式场：

$$
F(\mathbf p)=m(\mathbf p)+\epsilon(\mathbf p),
\qquad
\epsilon\sim\mathcal{GP}(0,k).
$$

主方案的协方差为：

$$
k(\mathbf p,\mathbf q)
=\sigma_f^2\rho\!\left(\frac{\|\mathbf p-\mathbf q\|}{\ell}\right).
$$

其中 $\rho(0)=1$。假设残差过程具有使用首达通量和梯度所需的光滑性，并且

$$
\beta=-\rho''(0)>0.
$$

| 符号 | 含义 |
| --- | --- |
| $\mathbf p$ | 三维空间位置 |
| $m(\mathbf p)$ | 三线性插值的确定性 SDF 均值 |
| $\epsilon(\mathbf p)$ | 零均值随机残差 |
| $\sigma_f>0$ | 场值标准差，与 $m$ 同单位 |
| $\ell>0$ | 本文核参数化中的相关长度参数 |
| $\rho(r)$ | 单位方差的径向相关函数 |
| $\beta$ | 标准化残差沿射线导数的方差 |
| $\mathbf o$ | 当前射线起点 |
| $\boldsymbol\omega$ | 单位射线方向 |
| $\mathbf r(s)=\mathbf o+s\boldsymbol\omega$ | 物理距离为 $s$ 的射线位置 |
| $L$ | 当前射线可查询的最大物理长度 |
| $\tau$ | 首次进入 $F<0$ 的物理距离 |
| $\mathbf G=\nabla F$ | 完整三维梯度 |
| $\mathbf n=\mathbf G/\|\mathbf G\|$ | 指向 $F>0$ 一侧的单位法线 |

空间变化的 $m$ 不会使残差核失去平稳性。反过来，即使先验核平稳，给定起点场值和梯度后的条件 GP 也一般不是平稳的；本方案会通过正确的起点条件处理这一点。

### 2.1 标准化射线问题

令

$$
x=\frac{s}{\ell},
\qquad
X_{\max}=\frac{L}{\ell},
\qquad
b(x)=\frac{m(\mathbf r(\ell x))}{\sigma_f},
$$

以及

$$
Z(x)=\frac{\epsilon(\mathbf r(\ell x))}{\sigma_f}.
$$

于是

$$
\frac{F(\mathbf r(\ell x))}{\sigma_f}=b(x)+Z(x),
\qquad
\operatorname{Cov}(Z(x),Z(y))=\rho(|x-y|).
$$

标准化均值导数为

$$
b'(x)=\frac{\ell}{\sigma_f}
\boldsymbol\omega^\top\nabla m(\mathbf r(\ell x)).
$$

首次进入表面时，定义正穿越速度

$$
W=-[b'(X_*)+Z'(X_*)]>0,
\qquad X_*=\frac{\tau}{\ell}.
$$

对应的物理沿射线导数为

$$
V=\boldsymbol\omega^\top\mathbf G
=-\frac{\sigma_f}{\ell}W.
$$

标准化后，同一核形状下不同的 $\sigma_f,\ell$ 可以共享网络。它们通过 $b(x)$、标准化段长及起点条件影响模型，而无需逐场景重新训练。

若 $\sigma_f=0$，应直接使用确定性 SDF 求交，不使用上述除以 $\sigma_f$ 的标准化。

## 3. 目标概率分布与 Renewal+ 条件

### 3.1 累计 hazard 与一维导数的联合模型

网络输出两个对象：

$$
H_\theta(x)=-\log T_\theta(x),
\qquad
q_\theta(w\mid X_*=x),\quad w>0.
$$

这里省略了对当前射线、均值剖面、核形状、起点条件以及历史状态的条件记号。

需要保证

$$
H_\theta(0)=0,
\qquad H'_\theta(x)\ge0,
\qquad
\int_0^\infty q_\theta(w\mid x)\,dw=1.
$$

标准化 hazard 为

$$
h_x(x)=H'_\theta(x).
$$

于是联合密度为

$$
p_\theta(X_*=x,W=w)
=e^{-H_\theta(x)}h_x(x)q_\theta(w\mid x).
$$

物理距离上的透射率与 hazard 分别是

$$
T_s(s)=e^{-H_\theta(s/\ell)},
\qquad
h_s(s)=\frac{1}{\ell}h_x(s/\ell).
$$

对于有限长度射线，未碰撞概率为

$$
P(\tau>L)=e^{-H_\theta(X_{\max})}.
$$

不要把有碰撞部分重新归一化为概率 1；未碰撞是模型必须保留的概率质量。

这里预测的是 $H=\int h\,dx=-\log T$，不是 $\int\log h\,dx$。分解为距离分布和条件导数分布，也不意味着两者独立。

### 3.2 与 GPIS 记忆模型的对应

| 记忆模型 | 在真实碰撞后保留的信息 |
| --- | --- |
| Renewal | 最近碰撞点及 $F(\mathbf p_0)=0$ |
| Renewal+ | 最近碰撞点、$F(\mathbf p_0)=0$ 和完整梯度 $\mathbf g_0$ |
| 本方案 | 以 Renewal+ 为目标，并用网络近似其条件首达分布 |

Renewal+ 不保留更早顶点及此前光路的全部场值条件。相关定义和一维／横向梯度分解可参见 [4,5]。

必须区分两种历史：

- **跨真实碰撞的历史**：按 Renewal+ 截断，只保留最新碰撞条件。
- **两次真实碰撞之间的直线历史**：逐段传播 GRU 状态，不在体素边界重置。

网络状态是对当前直线首达问题的有限维近似，不是 GP 路径历史的精确充分统计量。

### 3.3 两种起点模式

**模式 A：从已知空域出发。**

$$
F(\mathbf o)>0.
$$

用于相机射线或其他首次查询该 GPIS、且没有该 GPIS 已知碰撞梯度的射线。网络知道起点在空域，但不知道起点随机场值的具体实现。

**模式 B：从已知 GPIS 表面向外出发。**

$$
F(\mathbf p_0)=0,
\qquad
\nabla F(\mathbf p_0)=\mathbf g_0,
\qquad
\boldsymbol\omega^\top\mathbf g_0>0.
$$

用于碰撞后发出的散射射线，以及从该碰撞点发出的阴影射线。此时寻找的是起点之后的首次重新进入表面，起点的零值本身不是新碰撞。

包围盒边界、体素边界和人为计算分段都不是随机零水平面的观测。进入包围盒不意味着 $F=0$，也不自动证明 $F>0$。起点条件应由渲染问题实际已知的信息决定。

## 4. 为什么只学习一维导数即可采样三维法线

以下结论针对本文的平稳各向同性核与上述起点条件。

固定当前射线方向，取两个横向正交单位向量 $\mathbf e_1,\mathbf e_2$，并写成

$$
\mathbf E=[\mathbf e_1\ \mathbf e_2],
\qquad
\mathbf E^\top\boldsymbol\omega=\mathbf0.
$$

对任意沿射线位置 $u$，各向同性核满足

$$
\operatorname{Cov}
\left(
\mathbf E^\top\nabla\epsilon(\mathbf r(s)),
\epsilon(\mathbf r(u))
\right)=\mathbf0.
$$

因为这些量联合高斯，横向残差梯度与整条直线上的场值过程独立，也与该过程的沿射线导数独立。首达条件只涉及这一维场值过程，因此不改变横向残差的分布。

这给出模式 A 下的解析重建：

$$
\mathbf G_\tau
=V\boldsymbol\omega
+\mathbf E\left[
\mathbf E^\top\nabla m(\mathbf r(\tau))
+\sigma_g\boldsymbol\xi
\right],
$$

$$
\boldsymbol\xi\sim\mathcal N(\mathbf0,\mathbf I_2),
\qquad
\sigma_g^2=\frac{\sigma_f^2}{\ell^2}\beta.
$$

起点空域条件只约束射线上的场值，所以不破坏该独立性。模式 B 还需对横向梯度条件化于起点梯度，见第 12 节。

这里的横向分布是给定条件下的解析结果；神经网络近似的是首达距离及沿射线导数。高斯过程值—导数协方差的基础规则见 [1]。

## 5. 训练数据生成

### 5.1 只生成射线上的随机场

距离和一维导数标签可以完全由 $Z(x),Z'(x)$ 生成，不需要为每条训练射线构建三维随机场。

数据生成时，应始终区分：

- **网络可见输入**：确定性 SDF 剖面、核描述、起点模式及真正已知的条件；
- **用于生成标签的随机实现**：$Z,Z'$ 的采样值。

模式 A 中，随机实现值不能输入网络，否则训练和渲染的信息条件不一致。

### 5.2 生成三线性 SDF 剖面

从多种三维几何建立 SDF 网格，再用与渲染阶段相同的三线性插值、梯度计算和 DDA 体素遍历抽取射线。

训练几何应覆盖：

- 平面、球、圆柱、环面与高曲率区域；
- 薄层、邻近表面、凹形结构和多个可能交点；
- 掠射与近相切射线；
- 体素边界处均值导数的跳变；
- 长距离无碰撞，以及均值剖面多次靠近零水平面的情况。

在单个体素内，三线性函数限制于直线后至多为三次多项式：

$$
b_i(u)=d_{i0}+d_{i1}u+d_{i2}u^2+d_{i3}u^3,
\qquad u\in[0,1].
$$

不要把每段均值近似为常数或仅用切平面代替。网络需要看到与实际渲染一致的三次剖面。

不同几何、体素分辨率、$\sigma_f$ 与 $\ell$ 应共同生成标准化剖面，以覆盖不同的相对尺度。若网络仅训练一种核形状，它不自动具备对其他核形状的泛化能力。

### 5.3 第一版参考核：Matérn-3/2

采用

$$
\rho(r)=(1+r)e^{-r},
\qquad \beta=1.
$$

本文的 $\ell$ 对应这个公式中的尺度；与采用 $\sqrt3 r/\ell$ 的常见 Matérn 参数化换算时，需要统一长度定义。

令 $D=Z'$，则标准化过程可表示为

$$
dZ=D\,dx,
\qquad
dD=(-Z-2D)\,dx+2\,dB_x,
$$

其中 $B_x$ 是标准布朗运动。平稳状态满足

$$
(Z,D)^\top\sim\mathcal N(\mathbf0,\mathbf I_2).
$$

步长为 $\Delta$ 时，节点间精确转移为

$$
\begin{pmatrix}Z_{n+1}\\D_{n+1}\end{pmatrix}
=\mathbf A_\Delta
\begin{pmatrix}Z_n\\D_n\end{pmatrix}
+\boldsymbol\eta_n,
\qquad
\boldsymbol\eta_n\sim\mathcal N(\mathbf0,\mathbf Q_\Delta),
$$

$$
\mathbf A_\Delta=e^{-\Delta}
\begin{pmatrix}
1+\Delta&\Delta\\
-\Delta&1-\Delta
\end{pmatrix},
\qquad
\mathbf Q_\Delta=\mathbf I_2-\mathbf A_\Delta\mathbf A_\Delta^\top.
$$

这些转移在非均匀节点上也成立。使用双精度参考采样，并在极小步长下注意 $\mathbf Q_\Delta$ 的消减误差。Matérn 状态空间表示的理论依据见 [2]；这里的系数按单位场值方差、单位导数方差写出。

### 5.4 参考轨迹的两种初始化

**模式 A：已知起点在空域。**

在 Matérn 示例中采样

$$
Z(0)\sim\mathcal N(0,1)\text{ 截断至 }Z(0)>-b(0),
\qquad
D(0)\sim\mathcal N(0,1).
$$

两者独立。网络只接收 $b(0)$、核描述和模式标记，不接收采到的 $Z(0),D(0)$。

**模式 B：已知表面及完整梯度。**

设射线从 $\mathbf p_0$ 沿 $\boldsymbol\omega$ 出发，给定完整梯度 $\mathbf g_0$。初始状态为

$$
Z(0)=-\frac{m(\mathbf p_0)}{\sigma_f}=-b(0),
$$

$$
D(0)=\frac{\ell}{\sigma_f}
\left[
\boldsymbol\omega^\top\mathbf g_0
-\boldsymbol\omega^\top\nabla m(\mathbf p_0)
\right].
$$

要求初始总导数

$$
b'(0)+D(0)=\frac{\ell}{\sigma_f}\boldsymbol\omega^\top\mathbf g_0>0.
$$

这些初始状态是已知条件，可以输入网络。沿射线的首达数据不需要额外采样起点的横向梯度；横向条件在法线重建时解析使用。

模式 B 的初始向外导数应覆盖真实碰撞与 BSDF 发射产生的范围，可用参考碰撞样本生成，也可在声明的参数范围内覆盖采样。

### 5.5 其他核的通用参考采样

若核没有便捷的有限维状态表示，可联合采样有限节点上的值与导数。令 $c(d)=\rho(|d|)$、$d=x-y$，其协方差块为

$$
\operatorname{Cov}
\left[
\begin{pmatrix}Z(x)\\Z'(x)\end{pmatrix},
\begin{pmatrix}Z(y)\\Z'(y)\end{pmatrix}
\right]
=
\begin{pmatrix}
c(d)&-c'(d)\\
c'(d)&-c''(d)
\end{pmatrix}.
$$

模式 A 可先采样满足空域条件的初始状态，再用高斯条件分布生成其余节点；模式 B 直接对已知初始值与导数条件化。

高密度节点和导数观测容易造成数值病态。核近似、抖动项和有限精度均应计入参考数据误差，而不视作精确连续 GP 的性质。

### 5.6 连续首达标签与参考误差

建议先使用以下明确的数值参考流程：

1. 合并 GP 参考节点与全部体素边界。
2. 在这些节点联合采样 $Z,Z'$。
3. 用相邻节点的值和导数构造三次 Hermite 插值 $Z_h(x)$。
4. 因为每个小区间都处于一个体素内，$b(x)+Z_h(x)$ 是三次多项式。
5. 求每个小区间内的全部实根，按距离排序，取第一次向下穿越。
6. 用同一个插值函数在该根上求导，记录

$$
x_*,\qquad
w_*=-[b'(x_*)+Z_h'(x_*)]>0.
$$

模式 B 需排除起点 $x=0$ 的根，并从已知向外离开后的首个重新进入事件取标签。

不要只检查区间两端的符号：两端同号不排除内部出现两个或更多根。也不要在求根查询时重新独立采样 GP，否则得到的不是同一个实现的交点。

节点 GP 转移可以精确，但连续轨迹的 Hermite 插值及首达标签仍是数值近似。可从参考步长 $\Delta x=1/64$ 开始，再用减半步长检查首达距离、遗漏碰撞概率及导数分布的收敛。也可使用具有交点遗漏概率控制的自适应 GP 参考算法 [7]。

### 5.7 样本字段

| 字段 | 内容 |
| --- | --- |
| `mode` | 空域起点或表面起点 |
| `kernel_descriptor` | 固定核时可省略；多核时记录形状或参数 |
| `segment_features` | 沿射线逐段的三次均值剖面及段长 |
| `initial_condition` | 仅记录实际已知的初始条件 |
| `x_max` | 最大标准化长度 |
| `hit` | 是否在最大长度内碰撞 |
| `x_hit` | 首达标准化距离，仅命中样本有值 |
| `w_hit` | 首达标准化正穿越速度，仅命中样本有值 |
| `scene_id` | 按几何场景划分训练、验证与测试 |
| `reference_resolution` | 参考轨迹分辨率或误差设定 |

有限长度的截断点不应依据尚未观察的随机碰撞结果选择。未命中样本必须保留。

## 6. 网络输入与状态推进

### 6.1 渲染分段

首先用 DDA 获取体素段，再对过长或均值变化剧烈的段细分。初始设置可采用

$$
\Delta x_i\le0.25.
$$

也可限制每个子段的均值变化量；该细分规则应在训练和渲染时一致。参考数据生成步长与网络渲染步长是两种不同的尺度。

每段用以下特征准确确定三次均值剖面：

$$
\mathbf f_i=
\left[
b_i(0),\ b_i(1),\
\Delta x_i b'_i(x_i),\
\Delta x_i b'_i(x_{i+1}),\
\log\Delta x_i
\right].
$$

端点导数均取该段内部的单侧导数。四个 Hermite 数据足以确定一个三次多项式。

较大的输入动态范围可以用可逆的 `asinh` 等变换处理；直接裁剪均值或导数会丢失信息，需要作为额外近似评估。

### 6.2 配置建议

| 模块 | 起始配置 | 输出或作用 |
| --- | --- | --- |
| 段编码器 | 2 层 MLP，宽度 64，SiLU | $\mathbf e_i$ |
| 初始状态编码器 | MLP | 根据模式与初始条件生成 $\mathbf z_0$ |
| 历史状态 | GRU，隐藏维度 128 | 编码已存活的段序列 |
| 累计 hazard 头 | 2 层 MLP，宽度 64 | 4 个非负系数 |
| 导数 mixture 头 | 2 层 MLP，宽度 128 | 8 个一维混合分量的权重、均值和标准差 |

累计头输入为 $(\mathbf z_i,\mathbf e_i)$。导数头在查询段内位置 $u$ 时输入

$$
(\mathbf z_i,\mathbf e_i,u,b_i(u),b_i'(x)).
$$

这里 $b_i'(x)$ 是对标准化距离 $x$ 的导数，不是对局部坐标 $u$ 的导数。

如果整段没有碰撞，则更新

$$
\mathbf z_{i+1}
=\operatorname{GRU}(\mathbf z_i,\mathbf e_i).
$$

状态更新的语义是“在已存活条件下推进到下一段”。训练时，只沿当前样本碰撞前的前缀执行相应更新；未命中样本推进至终点。

模式 A 的初始状态不应依赖数据生成器实际采到的随机初值。模式 B 则需要显式输入已知初始值、初始导数以及模式标记。

## 7. 单调累计 hazard 头

仅将普通网络输出经过 softplus，不能保证累计 hazard 随距离单调。若隐藏状态随位置更新，仅约束网络对距离输入的偏导也不够。

本方案在每一段内固定进入状态和段特征，以显式单调多项式预测累计增量。

### 7.1 四次累计多项式

网络对第 $i$ 段输出四个实数 $a_{ij}$，并令

$$
r_{ij}=\operatorname{softplus}(a_{ij}),\qquad j=0,1,2,3.
$$

定义 Bernstein 基函数

$$
B_{j,n}(u)=\binom nj u^j(1-u)^{n-j},\qquad u\in[0,1].
$$

构造累计系数

$$
\alpha_{i0}=0,\qquad
\alpha_{i,j+1}=\alpha_{ij}+\frac{\Delta x_i}{4}r_{ij}.
$$

段内累计增量为

$$
A_i(u)=\sum_{j=0}^{4}\alpha_{ij}B_{j,4}(u).
$$

因此

$$
H_\theta(x_i+u\Delta x_i)
=H_\theta(x_i)+A_i(u),
$$

$$
h_x(x_i+u\Delta x_i)
=\frac1{\Delta x_i}\frac{dA_i}{du}
=\sum_{j=0}^{3}r_{ij}B_{j,3}(u)\ge0.
$$

整段增量是

$$
A_i(1)=\alpha_{i4}
=\frac{\Delta x_i}{4}\sum_{j=0}^{3}r_{ij}.
$$

该参数化保证累计量在段间连续、沿射线单调；hazard 允许在体素／分段边界跳变。训练似然不需要数值积分，采样时只需求单调多项式的逆。

直接建模累计强度再微分得到 hazard 的路线有神经点过程先例 [3]。这里的逐段 Bernstein 参数化是本方案的具体设计。

### 7.2 容量与精度

每段三次 hazard 是近似函数族。若局部首达密度过于尖锐，应细分段或提高多项式阶数，不能认为有限阶多项式对所有 SDF 都精确。

不要通过在所有位置添加固定的正 hazard 下限来处理数值问题；这会在长空域中引入人为消光。正性和计算稳定性应分别处理。

## 8. 一维首达导数 mixture

### 8.1 分布定义

使用正半轴上的截断 Gaussian mixture：

$$
q_\theta(w\mid x,\mathbf z)
=\sum_{j=1}^{J}\pi_j
\frac{\mathcal N(w;\mu_j,\sigma_j^2)}
{\Phi(\mu_j/\sigma_j)}
\mathbf1_{\{w>0\}},
\qquad J=8.
$$

其中 $\Phi$ 是标准正态分布函数。每个分量单独在正半轴归一化，因此分量采样权重就是 $\pi_j$。

采用

$$
\pi_j=\operatorname{softmax}(\mathbf a)_j,
\qquad
\sigma_j=\sqrt\beta\,[\operatorname{softplus}(t_j)+\varepsilon_\sigma].
$$

可将均值写为

$$
\mu_j=-b'(x)+\delta\mu_j,
$$

以确定性穿越速度为基准学习偏移。$\varepsilon_\sigma$ 用于防止方差塌缩，其设置应与参考导数标签的精度和验证结果匹配。

### 8.2 不重复加入 Rice 权重

该 mixture 直接在**首达点穿越速度**样本上训练，首达选择和穿越偏置已经包含在目标分布中。因此不应在采样或似然中再额外乘一次 $w$。

若改用显式通量加权 Gaussian mixture，则那是另一种从一开始定义的归一化分布族，需要重新定义密度与采样器，不能只在训练好的分布后补乘权重。

普通未截断 GMM 会给 $w\le0$ 分配概率；对高斯样本取绝对值产生的是折叠分布，也不是这里定义的截断分布。

### 8.3 采样与数值计算

先按 $\pi_j$ 选择分量，再从该分量截断至 $w>0$ 的正态分布采样。

理论上的逆 CDF 公式为

$$
a=\Phi(-\mu_j/\sigma_j),\qquad U\sim\operatorname{Uniform}(0,1),
$$

$$
w=\mu_j+\sigma_j\Phi^{-1}\big(a+U(1-a)\big).
$$

实现中对极端尾部应使用稳定的截断正态算法或对数尾概率形式，避免低精度下 $1-a$ 消减为零。

似然使用 log-CDF、log-softmax 与 log-sum-exp 计算。

## 9. 训练目标与训练组织

所有下述损失均在标准化距离 $x$ 与标准化穿越速度 $w$ 中计算。

### 9.1 命中样本

对参考标签 $(x_*,w_*)$，联合负对数似然为

$$
\mathcal L_{\mathrm{hit}}
=H_\theta(x_*)
-\log h_x(x_*)
-\log q_\theta(w_*\mid x_*).
$$

其中

$$
H_\theta(x_*)
=\sum_{i<i_*}A_i(1)+A_{i_*}(u_*).
$$

$i_*$ 是命中段，$u_*$ 是该段内的命中位置。直接使用首达位置与导数即可训练，无须预先估计“真实 hazard 曲线”。

### 9.2 未命中样本

对直到 $X_{\max}$ 都未命中的样本，使用

$$
\mathcal L_{\mathrm{miss}}=H_\theta(X_{\max}).
$$

未命中样本是右删失观测。只保留命中样本会改变训练目标，使模型偏向最终必然发生碰撞。

若人为重采样命中／未命中类别以平衡训练，应使用与采样方案一致的权重，或者明确承认训练分布已被改变。

### 9.3 起始超参数

| 项目 | 建议起点 |
| --- | --- |
| 第一版核 | Matérn-3/2，固定 $\rho$ |
| GRU 隐藏维度 | 128 |
| 一维 mixture 分量数 | 8 |
| 段内累计多项式阶数 | 4 |
| 首轮数据量 | $10^5$ 条随机轨迹，验证后扩到 $10^6$ 量级 |
| 同一剖面的随机实现数 | 4–16 |
| Batch | 64–128 条变长序列 |
| 优化器 | AdamW |
| 初始学习率 | $3\times10^{-4}$ |
| Weight decay | 可从 $10^{-5}$ 开始 |
| 梯度裁剪 | 范数 1 |
| 模型选择 | 验证集联合负对数似然及首达分布误差 |

这些数值需要根据参考数据质量、几何复杂度和目标误差调整。模型对任意新 SDF 的泛化能力必须通过未见几何验证，不能由网络结构直接保证。

### 9.4 训练伪代码

```text
for record in batch:
    z = initialize_from_known_start_conditions(record)
    H = 0

    for segment in prefix_until_hit_or_endpoint(record):
        features = encode_cubic_mean_and_length(segment)
        A, hazard_coefficients = cumulative_head(z, features)

        if this_segment_contains_hit(record):
            u = normalized_hit_position(record, segment)
            H += A(u)
            h = evaluate_standardized_hazard(hazard_coefficients, u)
            mixture = derivative_head(z, features, u)
            loss = H - log(h) - log_density(mixture, record.w_hit)
            break

        H += A(1)
        z = update_survival_state(z, features)

    if record.is_miss:
        loss = H

average_losses_and_backpropagate()
```

如果终点落在段内，应在数据预处理时裁剪该段，或在损失中使用相应的部分累计增量。

## 10. 渲染时的碰撞距离采样

对一条给定起点条件的射线，采样指数阈值

$$
E=-\log U,\qquad U\sim\operatorname{Uniform}(0,1).
$$

初始化历史状态 $\mathbf z_0$，逐体素段计算累计增量。

### 10.1 跳过未碰撞段

若

$$
E>A_i(1),
$$

则该段未命中，更新

$$
E\leftarrow E-A_i(1),
\qquad
\mathbf z_{i+1}=\operatorname{GRU}(\mathbf z_i,\mathbf e_i).
$$

本实现保留剩余指数阈值与当前历史，直接实现整条射线的累计分布逆变换。不要将体素边界当成新的独立起点而重置历史；只利用指数分布无记忆性重新抽取阈值、但正确保留生存条件与状态，是另一个等价实现选择。

### 10.2 在命中段求根

若

$$
E\le A_i(1),
$$

求解

$$
A_i(u_*)=E,\qquad u_*\in[0,1].
$$

使用二分法或带区间保护的 Newton 法。累计函数解析且单调；求逆的容差应按所需物理位置精度设置。

恢复碰撞位置：

$$
x_*=x_i+u_*\Delta x_i,
\qquad
s_*=\ell x_*,
\qquad
\mathbf p_*=\mathbf o+s_*\boldsymbol\omega.
$$

若走到最大长度仍未耗尽 $E$，返回未碰撞。

### 10.3 渲染伪代码

```text
function sample_first_hit(ray, known_start_condition):
    E = sample_exponential(rate=1)
    z = initialize_from_known_start_conditions(known_start_condition)

    for segment in traverse_and_subdivide_sdf_grid(ray):
        features = encode_cubic_mean_and_length(segment)
        A = cumulative_head(z, features)

        if E > A(1):
            E -= A(1)
            z = update_survival_state(z, features)
            continue

        u = invert_monotone_cumulative(A, E)
        position = segment.position(u)
        mixture = derivative_head(z, features, u)
        w = sample_positive_truncated_gaussian_mixture(mixture)
        V = -(sigma_f / ell) * w

        gradient = sample_analytic_transverse_gradient_and_combine(
            position, V, ray.direction, known_start_condition
        )
        normal = gradient / norm(gradient)
        return hit(position, gradient, normal)

    return miss
```

导数 mixture 头只在命中位置查询一次。累计头和 GRU 按遍历段数调用。实际加速程度需要相对参考 GP 求交器测量。

## 11. 空域出发时的法线采样

先从 mixture 得到 $w_*$，恢复

$$
V_*=-\frac{\sigma_f}{\ell}w_*.
$$

在碰撞位置计算三线性 SDF 的实际梯度 $\nabla m(\mathbf p_*)$。构建横向基 $\mathbf e_1,\mathbf e_2$，并采样

$$
\xi_1,\xi_2\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

完整梯度为

$$
\boxed{
\mathbf g_*
=V_*\boldsymbol\omega
+\sum_{a=1}^{2}
\left[
\mathbf e_a^\top\nabla m(\mathbf p_*)
+\sigma_g\xi_a
\right]\mathbf e_a.
}
$$

取

$$
\mathbf n_*=\frac{\mathbf g_*}{\|\mathbf g_*\|}.
$$

该重建满足

$$
\boldsymbol\omega^\top\mathbf g_*=V_*<0.
$$

不要预先归一化 $\nabla m$。三线性插值后的 SDF 梯度通常也不严格满足模长 1，而它的实际大小正是模型输入的一部分。

碰撞一般不恰好落在体素边界；数值上落在边界时，应使用与求交段一致的梯度约定。

## 12. 表面出发与完整 Renewal+ 法线条件化

设最近碰撞点是 $\mathbf p_0$，保存的完整梯度为 $\mathbf g_0$。沿新的向外方向 $\boldsymbol\omega$ 发射射线。

### 12.1 初始化下一次直线传播

使用第 5.4 节的模式 B 条件初始化网络和参考目标：

$$
Z(0)=-m(\mathbf p_0)/\sigma_f,
$$

$$
D(0)=\frac{\ell}{\sigma_f}
\boldsymbol\omega^\top
[\mathbf g_0-\nabla m(\mathbf p_0)].
$$

将此前直线传播的 GRU 状态结束，按最新碰撞条件初始化新状态。这是跨碰撞的 Renewal+ 截断；沿新射线穿过体素时则持续传递状态。

### 12.2 横向梯度仍可解析采样

令

$$
c_\perp(x)=-\frac{\rho'(x)}{\beta x},
\qquad c_\perp(0)=1.
$$

起点横向残差为

$$
a_{0,a}=\mathbf e_a^\top
[\mathbf g_0-\nabla m(\mathbf p_0)],\qquad a=1,2.
$$

在新碰撞位置 $\mathbf p_*$、标准化距离 $x_*$ 处，采样

$$
g_{\perp,a}
=\mathbf e_a^\top\nabla m(\mathbf p_*)
+c_\perp(x_*)a_{0,a}
+\sigma_g\sqrt{1-c_\perp(x_*)^2}\,\xi_a.
$$

然后重建

$$
\boxed{
\mathbf g_*=V_*\boldsymbol\omega
+g_{\perp,1}\mathbf e_1+g_{\perp,2}\mathbf e_2.
}
$$

该公式来自起点与终点横向残差梯度的二元高斯条件分布。各向同性结构保证横向过程与射线标量过程独立，因此给定首达事件和沿射线导数后仍可这样计算。

对于本文的 Matérn-3/2 核：

$$
c_\perp(x)=e^{-x}.
$$

对于 $\rho(x)=e^{-x^2/2}$ 的 squared-exponential 核：

$$
c_\perp(x)=e^{-x^2/2}.
$$

在数值实现中，仅对舍入误差导致的微小负方差做处理；不要用任意裁剪掩盖错误的核或协方差公式。

### 12.3 Renewal+ 与 Renewal Half+

完整 Renewal+ 同时保留起点梯度对沿射线导数和横向梯度的影响。

如果沿射线仍使用模式 B，但将新碰撞处的横向梯度改为第 11 节的无起点梯度条件高斯，就忽略了横向相关。这对应 [5] 中的 **Renewal Half+**，是另一种更强的近似，而不是完整 Renewal+。

即使网络只输出一维穿越速度，也要在碰撞后保存重建出的完整 $\mathbf g_*$。只保存单位法线会丢失下一条射线所需的梯度大小。

## 13. 透射率、阴影射线与路径追踪接口

### 13.1 透射率查询

无需抽取随机阈值，按与距离采样相同的起点模式和分段规则累计

$$
H_\theta(X_{\max})=\sum_i A_i(1),
$$

并返回

$$
T=e^{-H_\theta(X_{\max})}.
$$

若终点在段内，则最后一段使用部分累计增量。

### 13.2 阴影射线

从一个已知 GPIS 碰撞点发出的阴影射线，应该使用模式 B，并以该阴影方向重新计算初始沿射线导数。它不是普通的空域初始化。

若只改变了方向，起点完整梯度仍然是同一个 $\mathbf g_0$。其在新方向上的投影必须重新计算。

### 13.3 多次散射

几何采样接口返回位置、完整梯度和法线。路径追踪器据此调用材质 BSDF，生成新的向外方向，然后用最新碰撞条件重新初始化下一次直线传播。

当几何按照模型自身的概率分布采样时，它是目标模型中的随机几何变量；不应仅因它是随机量就额外除以碰撞密度。若另行设计几何重要性采样或与其他策略做 MIS，则需按实际策略构造相应 PDF 和权重。

本文没有将一次射线的模型透射率宣称为全局 GP 完整历史条件下的精确可见性。跨碰撞截断仍遵循 Renewal+。

## 14. 实现中的一致性要求

| 项目 | 要求 |
| --- | --- |
| 均值查询 | 数据生成与渲染使用同一三线性插值规则 |
| 均值导数 | 使用实际插值导数，不预先单位化 |
| 距离单位 | 明确区分 $s$ 与 $x=s/\ell$ |
| 速度单位 | 明确区分 $V$ 与 $W=-\ell V/\sigma_f$ |
| 初始信息 | 不向空域模式网络泄漏随机初值 |
| 生存历史 | 当前直线内持续推进，不在体素边界重置 |
| 跨碰撞状态 | 以最近 $F=0$ 和完整梯度重新初始化 |
| 混合分布 | 使用归一化的正半轴截断分量 |
| 穿越权重 | 直接拟合首达速度时不重复乘 Rice 权重 |
| 未碰撞样本 | 纳入似然训练与验证 |
| 未碰撞概率 | 保留 $e^{-H(X_{\max})}$，不强制所有射线命中 |
| 核形状 | 改核后更换参考采样器，或在多核训练中明确输入核描述 |

## 15. 固定椭圆各向异性核扩展

考虑

$$
k(\mathbf p,\mathbf q)
=\sigma_f^2\rho\!\left(
\sqrt{(\mathbf p-\mathbf q)^\top\mathbf M(\mathbf p-\mathbf q)}
\right),
$$

其中 $\mathbf M$ 为固定正定矩阵。沿射线定义

$$
\ell_\omega=
\frac1{\sqrt{\boldsymbol\omega^\top\mathbf M\boldsymbol\omega}}.
$$

用 $\ell_\omega$ 替代标准化中的 $\ell$ 后，射线标量过程仍具有相同核形状 $\rho$，因此可以复用一维模型。

### 15.1 空域模式的梯度重建

定义

$$
\mathbf C_\nabla=\sigma_f^2\beta\mathbf M,
\qquad
\mathbf a=
\frac{\mathbf C_\nabla\boldsymbol\omega}
{\boldsymbol\omega^\top\mathbf C_\nabla\boldsymbol\omega},
$$

$$
\mathbf C_R=
\mathbf C_\nabla-
\frac{
\mathbf C_\nabla\boldsymbol\omega
\boldsymbol\omega^\top\mathbf C_\nabla
}{
\boldsymbol\omega^\top\mathbf C_\nabla\boldsymbol\omega
}.
$$

采样到 $V$ 后，重建

$$
\mathbf g=
\nabla m+
\mathbf a[V-\boldsymbol\omega^\top\nabla m]
+\boldsymbol\xi_R,
\qquad
\boldsymbol\xi_R\sim\mathcal N(\mathbf0,\mathbf C_R).
$$

$\mathbf C_R$ 在射线方向为零，是秩至多为 2 的协方差。实现时可以在横向二维基内分解并采样，不应假设它是一个正定三维矩阵直接执行普通 Cholesky 分解。

### 15.2 表面模式的梯度重建

令 $V_0=\boldsymbol\omega^\top\mathbf g_0$，起点的回归残差为

$$
\mathbf R_0=
\mathbf g_0-\nabla m(\mathbf p_0)
-\mathbf a[V_0-\boldsymbol\omega^\top\nabla m(\mathbf p_0)].
$$

固定椭圆结构下，沿直线的回归残差与整条标量过程独立，且其起终点交叉协方差为 $c_\perp(x)\mathbf C_R$。因此

$$
\mathbf g_*=
\nabla m(\mathbf p_*)
+\mathbf a[V_*-\boldsymbol\omega^\top\nabla m(\mathbf p_*)]
+c_\perp(x_*)\mathbf R_0
+\sqrt{1-c_\perp(x_*)^2}\,\boldsymbol\xi_R.
$$

此处 $x_*=s_*/\ell_\omega$，$\boldsymbol\xi_R\sim\mathcal N(\mathbf0,\mathbf C_R)$。

这项扩展依赖固定椭圆径向结构，不能直接推广到任意平稳谱或任意空间变化的各向异性核。

## 16. 验证与误差分解

应区分以下三种误差，避免将它们混为一谈。

### 16.1 参考轨迹的数值误差

通过更细参考节点和更严格求根设置，检查：

- 首达距离 CDF；
- 有限长度未碰撞概率；
- 首达导数分布；
- 掠射与薄层区域的遗漏交点概率。

### 16.2 神经近似误差

在未见几何上评估：

- 联合负对数似然；
- 透射率校准与首达距离 CDF 误差；
- 按碰撞距离、均值高度和局部斜率分组的导数分布；
- 法线角度分布，特别是接近掠射的区域；
- 比训练序列更长的射线；
- 相同几何采用不同合理分段时的结果一致性。

有限隐藏状态与有限多项式阶数不保证对任意 SDF 剖面精确。分段一致性应作为验证指标；若差异明显，可在训练中加入不同细分下的累计概率一致性约束。

### 16.3 Renewal+ 的记忆截断误差

先与采用**同样 Renewal+ 条件**的 GP 参考渲染器比较，测量神经替代模型误差。

若再与 Global 条件模型比较，差异还包含 Renewal+ 丢弃早期光路信息的误差。不能将这部分差异全部归因于网络，也不能通过单段拟合成功宣称消除了它。

### 16.4 最小实验矩阵

| 实验 | 主要验证内容 |
| --- | --- |
| 平面均值＋不同斜率 | 距离与导数的基本尺度关系 |
| 曲面与掠射 | 首达速度尾部及法线分布 |
| 薄层与邻近表面 | 首达选择与未碰撞概率 |
| 非单调均值剖面 | 历史状态是否有用 |
| 空域起点／表面起点分别测试 | 初始化条件是否一致 |
| 保留／忽略横向起点相关 | Renewal+ 与 Renewal Half+ 的区别 |
| 长射线与不同分段 | 历史泛化与分段一致性 |
| 完整图像、相同记忆模型 | 渲染偏差、噪声与实际耗时 |

## 17. 与理论先例的关系

Schwalger 的工作从局部过零率和成对过零相关出发，近似构造带记忆的首达 hazard [6]。本方案采用相同的“首达分布需要历史修正”这一认识，但不直接使用其特定高斯动力学下的解析闭合公式。

本方案直接用首达样本训练累计 hazard 与条件导数分布，因此历史修正由网络隐式表达。横向梯度的解析部分则来自本文所限定核结构的高斯独立性。

需要分别辨认：

1. GP 值—导数统计和本核子类下的横向条件高斯公式；
2. 连续首达事件的数值参考近似；
3. 网络对首达分布的有限参数近似；
4. Renewal+ 对跨碰撞历史的截断。

## 18. 参考资料

以下文献提供相关理论和已有实现背景。本文的具体网络宽度、逐段 Bernstein 累计头、训练数据组织与组合流程属于本方案的设计建议。

1. C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*, §9.4, Derivative Observations. 2006. [章节 PDF](https://gaussianprocess.org/gpml/chapters/RW9.pdf)
2. Andrés Jordán, Susana Eyheramendy, Johannes Buchner. *State-space representation of Matérn and Damped Simple Harmonic Oscillator Gaussian processes*. 2021. [arXiv](https://arxiv.org/abs/2109.10685)
3. Takahiro Omi, Naonori Ueda, Kazuyuki Aihara. *Fully Neural Network based Model for General Temporal Point Processes*. NeurIPS 2019. [论文页面](https://papers.nips.cc/paper_files/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html)
4. Dario Seyb, Eugene d'Eon, Benedikt Bitterli, Wojciech Jarosz. *From microfacets to participating media: A unified theory of light transport with stochastic geometry*. SIGGRAPH 2024. [作者页面](https://cs.dartmouth.edu/~wjarosz/publications/seyb24from.html)
5. Kehan Xu, Benedikt Bitterli, Eugene d'Eon, Wojciech Jarosz. *Practical Gaussian Process Implicit Surfaces with Sparse Convolutions*. SIGGRAPH Asia 2025. [作者页面](https://cs.dartmouth.edu/~wjarosz/publications/xu25practical.html) · [参考实现](https://github.com/dartmouth-vcl/sparse-conv-gpis-tungsten)
6. Tilo Schwalger. *Mapping input noise to escape noise in integrate-and-fire neurons: a level-crossing approach*. Biological Cybernetics, 2021. [论文](https://link.springer.com/article/10.1007/s00422-021-00899-1)
7. Zhiqian Zhou, Dario Seyb, Shuang Zhao. *Adaptive Ray Marching for Rendering Gaussian Process Implicit Surfaces*. SIGGRAPH 2026. [作者页面](https://111116.github.io/fast-gpis-site/)

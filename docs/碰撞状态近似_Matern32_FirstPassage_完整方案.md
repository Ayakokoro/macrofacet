# 碰撞状态近似下的 Matérn 3/2 First-Passage 渲染方案

## 1. 目标与适用范围

本文定义一种面向 macrofacet 路径追踪的碰撞后自由程模型。每一条次级射线只保留最近一次真实碰撞处的状态：

$$
F(\mathbf b)=0,
\qquad
\nabla F(\mathbf b)=\mathbf g.
$$

从碰撞点 $\mathbf b$ 沿单位出射方向 $\mathbf d$ 传播时，所需透射率定义为整段无穿越概率：

$$
T_{\mathbf b}(L)
=
P\!\left[
F(\mathbf b+t\mathbf d)>0,
\ \forall t\in(0,L]
\mid
F(\mathbf b)=0,
\nabla F(\mathbf b)=\mathbf g
\right].
$$

这里计算的是 first-passage survival，而不是终点为正的概率：

$$
T_{\mathbf b}(L)
\neq
P[F(\mathbf b+L\mathbf d)>0\mid F(\mathbf b)=0,\nabla F(\mathbf b)=\mathbf g].
$$

右侧允许路径在 $(0,L)$ 内先穿过零面再返回正侧，因此不能替代真正的透射率。

第一版方案固定使用各向同性 Matérn $3/2$ kernel 和碰撞点处冻结的一阶 mean。目标是先建立一个自洽、可验证、能够用于自由程采样的模型。非线性 mean profile、各向异性 kernel 和完整历史条件化属于后续扩展。

---

## 2. 碰撞状态近似

### 2.1 保留的信息

每次真实碰撞后保留：

$$
\mathbf b,
\qquad
F(\mathbf b)=0,
\qquad
\mathbf g=\nabla F(\mathbf b).
$$

对于给定出射方向 $\mathbf d$，沿射线的一阶状态只依赖投影：

$$
g_d=\mathbf g\cdot\mathbf d.
$$

对于各向同性 GP，垂直于 $\mathbf d$ 的梯度分量不影响射线上的标量过程 $F(\mathbf b+t\mathbf d)$，所以它们不进入 survival 模型。但是完整梯度仍然必须保留，因为下一次碰撞的法线采样和反射需要它，见第 10 节。

### 2.2 丢弃的信息

模型明确丢弃：

- 当前碰撞之前的 survival 历史；
- 当前碰撞之前其他位置的场值；
- “$\mathbf b$ 是此前路径上的第一次碰撞点”这一历史条件；
- 其他 off-ray 观测；
- 更早碰撞点的场值和梯度。

因此每次真实碰撞形成一次 renewal：新射线只由当前碰撞状态初始化。这个重启只能发生在真实碰撞处，不能在一条自由程的中间位置重新条件化。

### 2.3 不允许的操作

以下做法会改变所定义的 survival 问题，应当禁止：

- 在 marching、DDA cell 边界或任意中间点重新初始化 GP；
- 用终点正概率代替整段 survival；
- 在已经传播到距离 $t$ 后丢弃累计消光，再从零开始查询长度 $\Delta$；
- 同一条自由程中途重新线性化 mean，却仍把结果解释为原条件过程的 survival。

---

## 3. 从三维 GP 约化到一维射线过程

全局场写成：

$$
F(\mathbf x)=m(\mathbf x)+Y(\mathbf x),
$$

其中 $m$ 是确定性 mean SDF，$Y$ 是零均值平稳 GP。

在碰撞点沿出射方向冻结一阶 mean：

$$
m(\mathbf b+t\mathbf d)
\approx
m_b+a_b t,
$$

其中

$$
m_b=m(\mathbf b),
\qquad
a_b=\nabla m(\mathbf b)\cdot\mathbf d.
$$

由总场碰撞条件 $F(\mathbf b)=0$ 得到残差初值：

$$
Y(0)=-m_b.
$$

由碰撞梯度得到残差方向导数：

$$
Y'(0)=g_d-a_b.
$$

因此每条次级射线对应的一维问题是：

$$
Y(0)=-m_b,
\qquad
Y'(0)=g_d-a_b,
$$

并求

$$
T_{\mathbf b}(L)
=
P\!\left[
m_b+a_b t+Y(t)>0,
\ \forall t\in(0,L]
\mid
Y(0)=-m_b,
Y'(0)=g_d-a_b
\right].
$$

对于合法的出射方向通常有

$$
g_d>0.
$$

如果 $g_d<0$，射线从零面立即进入负侧，按当前正侧 survival 定义应返回零。精确 grazing 状态 $g_d=0$ 是单独的退化边界；第一版数据域应设置一个正的最小 $g_d$，并把更小的值交给专门的极限处理，而不是混入普通训练样本。

---

## 4. Mean SDF 的三种处理方式

### 4.1 固定一阶局部 mean：第一版采用

整条自由程使用

$$
m(t)=m_b+a_b t.
$$

距离 $L$ 处的 mean SDF 已经由

$$
m_L=m_b+a_bL
$$

确定，所以不需要额外输入当前点 $m(\mathbf b+L\mathbf d)$。透射率写成：

$$
T_{\mathbf b}(L)
=
\mathcal T(L;m_b,a_b,g_d,\sigma,\ell).
$$

### 4.2 分段重新展开 mean：第一版禁止

如果每走一段就用当前 $m(x)$ 和 $\nabla m(x)\cdot\mathbf d$ 重新初始化，会把一个 first-passage survival 人为拆成多个独立过程。这不是同一个条件 GP，除非重新推导并保留前段 survival 所产生的完整条件状态。

### 4.3 使用真实非线性全局 SDF：后续扩展

严格结果依赖整段 profile：

$$
\{m(\mathbf b+t\mathbf d):0\le t\le L\}.
$$

单独的终点 SDF 不足以确定 survival。若以后需要真实 profile，应采用下列方案之一：

- 将 profile 的有限维描述作为拟合器输入；
- 沿射线直接求解时变状态空间/PDE；
- 使用保持同一条件状态的数值传播。

不能简单地在中间点重启碰撞状态模型。

---

## 5. Global frame 与 local frame

对于各向同性 Matérn kernel，以下两种计算严格等价：

1. 在 world/global frame 中条件化 $F(\mathbf b)$ 和 $\nabla F(\mathbf b)$；
2. 把 $\mathbf b$ 平移为 local origin，将 $\mathbf d$ 旋转到局部轴，再使用旋转后的梯度。

原因是各向同性 covariance 对刚体旋转不变。沿射线的 survival 只读取

$$
g_d=\mathbf g\cdot\mathbf d,
\qquad
a_b=\nabla m(\mathbf b)\cdot\mathbf d.
$$

local frame 本身不产生近似。真正的近似来自：

1. 丢弃碰撞前历史；
2. 将 mean 冻结为碰撞点的一阶展开；
3. 对未观测梯度量所做的额外统计假设；
4. 把连续 first-passage 模型替换为离线拟合器。

对于各向异性 kernel，仅旋转向量而不同时旋转 kernel metric 并不等价。后续若支持各向异性，应同时变换 metric，并使用射线有效长度：

$$
\ell_{\mathbf d}
=
\frac{1}{\sqrt{\mathbf d^T P\mathbf d}}.
$$

---

## 6. Matérn 3/2 条件过程

### 6.1 Kernel

一维射线上的 covariance 为：

$$
k(t)=\sigma^2(1+\lambda |t|)e^{-\lambda |t|},
\qquad
\lambda=\frac{\sqrt 3}{\ell}.
$$

方向导数方差为：

$$
\operatorname{Var}[Y'(0)]
=
\sigma^2\lambda^2
=
\frac{3\sigma^2}{\ell^2}.
$$

### 6.2 条件 mean

条件于

$$
Y(0)=-m_b,
\qquad
Y'(0)=g_d-a_b,
$$

总场条件 mean 为：

$$
\mu_F(t)
=
m_b+a_b t
-(1+\lambda t)e^{-\lambda t}m_b
+t e^{-\lambda t}(g_d-a_b),
\qquad t\ge0.
$$

它满足：

$$
\mu_F(0)=0,
\qquad
\mu'_F(0)=g_d.
$$

### 6.3 条件 covariance

对于 $t,s\ge0$：

$$
K_c(t,s)
=
k(|t-s|)
-\frac{k(t)k(s)}{\sigma^2}
-\frac{k'(t)k'(s)}{\sigma^2\lambda^2}.
$$

真正需要计算的是该条件高斯过程保持在正侧的概率：

$$
T_{\mathbf b}(L)
=
P[F(t)>0,\ \forall t\in(0,L]].
$$

条件 mean 和 covariance 只能给出过程定义，不能直接把 survival 简化成一个终点 Gaussian CDF。

### 6.4 状态空间形式

Matérn $3/2$ 可以写成二维 Markov SDE。令 $V=Y'$：

$$
dY=V\,dt,
$$

$$
dV=(-\lambda^2Y-2\lambda V)\,dt
+2\sigma\lambda^{3/2}\,dW_t.
$$

其平稳状态 covariance 为：

$$
\operatorname{Cov}
\begin{bmatrix}Y\\V\end{bmatrix}
=
\begin{bmatrix}
\sigma^2 & 0\\
0 & \sigma^2\lambda^2
\end{bmatrix}.
$$

初始状态不是随机抽取，而是由碰撞条件固定为：

$$
Y(0)=-m_b,
\qquad
V(0)=g_d-a_b.
$$

这个状态空间形式适合：

- 生成连续时间 first-passage Monte Carlo 数据；
- 建立二维 absorbing PDE；
- 在固定步长之间使用精确 Gaussian state transition；
- 记录 first-passage 时的纵向导数。

---

## 7. 无量纲参数化

固定 Matérn $3/2$ 后，引入：

$$
q=\frac{L}{\ell},
\qquad
\beta_0=\frac{m_b}{\sigma},
\qquad
\beta_a=\frac{a_b\ell}{\sigma},
\qquad
\beta_g=\frac{g_d\ell}{\sigma}.
$$

于是：

$$
T_{\mathbf b}(L)
=
\widehat{\mathcal T}
(q;\beta_0,\beta_a,\beta_g).
$$

拟合器权重可以固定，但输入必须随碰撞状态变化。一般不存在一条所有碰撞共享、只依赖 $L$ 的固定透射率曲线。

如果额外假设碰撞点位于 mean surface：

$$
m_b=0,
$$

可以去掉 $\beta_0$。但真实随机表面碰撞只保证 $F(\mathbf b)=0$，并不保证 $m(\mathbf b)=0$，所以第一版数据模型应保留 $\beta_0$。

如果 mean 是精确 SDF，则 $\|\nabla m\|=1$，可令 $a_b=\cos\theta$。但是

$$
g_d\approx\cos\theta
$$

还要求随机梯度模长近似固定为一，这不是 Matérn GP 自动保证的。除非明确接受这项额外近似，否则不应把模型压缩成仅依赖 $(q,\theta)$ 的二维表。

---

## 8. 拟合累计消光而不是瞬时 hazard

定义累计消光：

$$
H(t\mid\eta)=-\log T(t\mid\eta),
$$

其中

$$
\eta=(\beta_0,\beta_a,\beta_g).
$$

则：

$$
T(t\mid\eta)=e^{-H(t\mid\eta)},
$$

$$
\Sigma(t\mid\eta)=\partial_tH(t\mid\eta),
$$

$$
p_{\mathrm{FPT}}(t\mid\eta)
=
\Sigma(t\mid\eta)e^{-H(t\mid\eta)}.
$$

模型必须满足：

$$
H(0)=0,
\qquad
H(t)\ge0,
\qquad
H'(t)\ge0.
$$

### 8.1 为什么不直接拟合 hazard

Monte Carlo empirical survival

$$
\widehat T(t)
=
\frac1N\sum_{i=1}^{N}\mathbf 1[\tau_i>t]
$$

是阶梯函数。直接差分

$$
\widehat\Sigma(t)
\approx
-\frac{\log\widehat T(t+\Delta)-\log\widehat T(t)}{\Delta}
$$

会同时受到以下问题影响：

- 数值微分放大 Monte Carlo 噪声；
- survival 很小时误差被进一步放大；
- 尾部 risk set 很小；
- FPT histogram 对 bin width 很敏感；
- 真实 hazard 本身也可能非单调。

因此稳定性顺序通常是：

$$
T(t)
\longrightarrow
H(t)=-\log T(t)
\longrightarrow
\Sigma(t)=H'(t).
$$

### 8.2 单调参数化

一种可用形式是：

$$
H(q;\eta)
=
\int_0^q
\operatorname{softplus}(G(u,\eta))\,du.
$$

于是：

$$
\Sigma(t;\eta)
=
\frac1\ell
\operatorname{softplus}
\left(G\!\left(\frac t\ell,\eta\right)\right)
\ge0.
$$

可选实现包括：

- 单调 cubic spline 或 I-spline；
- 固定 $q$ knots 上的正增量表；
- 正系数基函数；
- 对正函数积分的神经网络。

第一版推荐先实现单调 spline/LUT 基线，再评估积分神经网络。这样容易检查单调性、反演和误差来源。

对于严格出射的 $g_d>0$，真实近零 hazard 通常趋向零。模型可通过 near-origin 专用基函数、显式 $q$ 因子或训练约束表达这一行为。不要让 grazing 极限主导普通状态的拟合。

---

## 9. 使用 censored FPT 样本训练

每个样本记录：

- 状态参数 $\eta_i$；
- first-passage distance $t_i$，或最大追踪距离 $L_i$；
- 是否发生穿越 $\delta_i\in\{0,1\}$；
- 对应 $\ell_i$、数值容差和随机种子。

对于发生穿越的样本：

$$
-\log p(t_i)
=
H(t_i)-\log\Sigma(t_i).
$$

对于右删失样本：

$$
-\log P(\tau>L_i)
=
H(L_i).
$$

使用无量纲 $q=t/\ell$ 且令 $H_q=\partial_qH$ 时：

$$
\Sigma_t(t)=\frac1\ell H_q(q).
$$

所以完整负对数似然为：

$$
\mathcal L
=
\sum_{i:\delta_i=1}
\left[
H(q_i;\eta_i)
-\log H_q(q_i;\eta_i)
+\log\ell_i
\right]
+
\sum_{i:\delta_i=0}
H(q_i;\eta_i).
$$

如果所有训练样本共享同一个 $\ell$，$\log\ell$ 对参数优化只是常数；多尺度数据仍应保留该项以维持正确的物理密度。

若暂时只拟合曲线，可使用：

1. Kaplan–Meier/empirical survival 得到 $\widehat T$；
2. 计算 $\widehat H=-\log\max(\widehat T,\epsilon)$；
3. 按 risk set 大小或 survival 方差加权；
4. 对 $\widehat H$ 做单调拟合；
5. 在有效 risk set 过小处截断，而不是强行拟合尾部。

直接使用逐样本 survival likelihood 通常优于先构造 noisy hazard histogram。

---

## 10. 从透射率模型闭合到完整路径追踪

只拟合 $H$ 可以查询透射率和采样下一次 first-passage distance，但完整渲染还需要下一碰撞处的梯度。必须把这两部分分开处理。

### 10.1 自由程采样

在碰撞点计算：

$$
m_b=m(\mathbf b),
\qquad
a_b=\nabla m(\mathbf b)\cdot\mathbf d,
\qquad
g_d=\mathbf g\cdot\mathbf d.
$$

构造 $\eta$，然后采样：

$$
u\sim U(0,1),
\qquad
E=-\log u.
$$

求解：

$$
H(q;\eta)=E,
\qquad
t=\ell q.
$$

若射线到 process box 出口的距离是 $L_{\rm exit}$，令

$$
q_{\rm exit}=L_{\rm exit}/\ell.
$$

当

$$
E>H(q_{\rm exit};\eta)
$$

时，本段无碰撞并从 domain 逃逸；否则用单调 root finder 反演得到碰撞距离。这个采样过程不需要显式构造 noisy 的点值 extinction，也不需要为拟合 hazard 建立 delta-tracking majorant。

### 10.2 分段透射率查询

如果已经沿同一条条件过程传播到 $t$ 且尚未穿越，则继续 $\Delta$ 的条件透射率为：

$$
T(t,t+\Delta\mid\tau>t)
=
\exp[-H(t+\Delta)+H(t)].
$$

这里的两个 $H$ 必须使用同一碰撞状态和同一次初始化，不能在 $t$ 处重新计算新的 $m_b,a_b,g_d$。

### 10.3 下一碰撞的纵向梯度

在 first passage 时，总场方向导数满足：

$$
g_d^{\rm hit}=F'(\tau)\le0.
$$

它的分布不是仅条件于 $F(\tau)=0$ 的普通 Gaussian，而是条件于“$\tau$ 是第一次穿越”的吸收边界通量分布。累计消光 $H$ 本身不能恢复这个分布。

因此数据生成时还必须记录：

$$
r
=
-\frac{\ell}{\sigma}F'(\tau)
\ge0,
$$

并另外拟合或表格化：

$$
p(r\mid q,\eta,\tau=q\ell).
$$

可以从简单的条件 quantile table、Gamma/log-normal mixture 开始，再根据误差决定是否需要条件 flow。若忽略这一步，就只能得到 transmittance，无法生成下一次反射所需的正确碰撞法线。

### 10.4 下一碰撞的横向梯度

对于各向同性 Matérn $3/2$，射线上的标量过程与横向梯度过程联合 Gaussian 且互不相关，因此横向梯度与 survival 事件独立。令：

$$
\mathbf g_\perp
=
\mathbf g-g_d\mathbf d,
$$

$$
\mathbf a_\perp
=
\nabla m(\mathbf b)-a_b\mathbf d.
$$

在 affine mean 下，距离 $t$ 处的横向梯度可直接采样为：

$$
E[\mathbf G_\perp(t)\mid\mathbf g_\perp]
=
\mathbf a_\perp
+e^{-\lambda t}(\mathbf g_\perp-\mathbf a_\perp),
$$

$$
\operatorname{Cov}[\mathbf G_\perp(t)\mid\mathbf g_\perp]
=
\sigma^2\lambda^2
(1-e^{-2\lambda t})I_\perp.
$$

把采样得到的横向分量与 $g_d^{\rm hit}\mathbf d$ 合成新的完整碰撞梯度 $\mathbf g^{\rm hit}$，即可计算 Fresnel、反射方向，并初始化下一条次级射线。

该解析分离依赖各向同性 kernel。各向异性版本必须重新处理纵向/横向耦合。

---

## 11. 主相机射线的初始化

本文定义的是从真实碰撞状态出发的 collision-to-collision 模型。相机射线从 domain 外部进入时没有

$$
F=0,\ \nabla F=\mathbf g
$$

这一初始碰撞状态，所以不能直接使用同一张 $H(q;\eta)$ 表。

第一阶段采用：

1. 主相机射线的第一次碰撞继续使用现有 `classic_global`/外部初始化方法；
2. 得到第一次真实碰撞位置和完整梯度后；
3. 所有后续次级射线使用碰撞状态 first-passage 模型。

如果以后希望主射线也使用 Matérn first-passage，需要单独建立 exterior-state 模型，例如条件于：

$$
F(0)=f_0>0,
\qquad
F'(0)=g_0,
$$

并增加 $f_0/\sigma$ 等输入。不能把外部 entry state 假装成零面碰撞状态。

---

## 12. 离线数据生成

### 12.1 首选方法：状态空间 Monte Carlo

对每个 $\eta$：

1. 从确定初值 $Y(0)=-m_b$、$V(0)=g_d-a_b$ 开始；
2. 使用 Matérn $3/2$ 线性 SDE 的精确离散 state transition；
3. 检测 $F(t)=m_b+a_bt+Y(t)$ 的第一次向下穿越；
4. 对疑似包含穿越的 interval 自适应细分；
5. 记录 $\tau$、$F'(\tau)$ 和 censor distance；
6. 用多组基础步长做 convergence study，防止漏掉步间穿越。

单纯检查固定网格采样点是否为负会高估 survival。即使状态端点是精确 Gaussian transition，first passage 仍然需要 bridge 检查或自适应细分。

### 12.2 验证方法

以下两种方法适合作为独立参考：

- 二维 absorbing PDE：状态为 $(Y,V)$ 或 $(F,F')$；
- 离散路径 Gaussian orthant probability，并随时间网格加密检查收敛。

PDE 和 Monte Carlo 至少应在一组二维参数切片上互相验证。

### 12.3 参数采样

不要一开始手工固定一个过大的四维均匀网格。推荐流程：

1. 用当前渲染器收集 $(\beta_0,\beta_a,\beta_g,q_{\rm exit})$ 的实际分布；
2. 根据实际分布确定主要训练 box；
3. 使用 Latin hypercube、Sobol 或自适应采样覆盖状态空间；
4. 对 grazing、短距离峰值和 survival 尾部加密；
5. 训练/验证/测试按状态 $\eta$ 切分，而不是只随机切分同一状态下的轨迹。

每条原始记录至少包含：

```text
kernel_id, kernel_type, sampler_type, crossing_slope_method,
sigma, ell, alpha,
beta_0, beta_a, beta_g,
max_q, base_step, training_resolution,
event_q, event, censored, crossing_slope,
seed, integrator_tolerance, minimum_step
```

---

## 13. 渲染接口建议

累计消光查询接口可设计为：

```cpp
struct CollisionStateCoordinates {
    double beta0;
    double betaMeanSlope;
    double betaCollisionSlope;
};

class CollisionStateCumulativeExtinction {
public:
    virtual ~CollisionStateCumulativeExtinction() = default;

    virtual double cumulative(double q,
                              const CollisionStateCoordinates& state) const = 0;
    virtual double derivative(double q,
                              const CollisionStateCoordinates& state) const = 0;
    virtual double inverse(double opticalDepth,
                           const CollisionStateCoordinates& state,
                           double maximumQ) const = 0;
};
```

下一碰撞梯度需要独立接口：

```cpp
class CollisionStateCrossingGradient {
public:
    virtual ~CollisionStateCrossingGradient() = default;

    virtual double sampleLongitudinalSlope(
        double q,
        const CollisionStateCoordinates& state,
        Random& rng) const = 0;
};
```

公共 `CovarianceKernel::evaluate()` 和 `KernelJet` 继续负责精确的两点 value/gradient covariance；累计消光拟合器是建立在指定 kernel 上的 transport approximation，不应塞回 kernel 类本身。

建议新增独立 transport mode，例如：

```text
global_collision_state
```

在验证完成前不要直接覆盖现有 `global_conditional`，这样可以保留 SE 解析实现作为对照。

每条 flight state 至少保存：

```text
birth_position
birth_gradient
direction
age_from_birth
beta_0
beta_a
beta_g
ell
```

所有分段查询都使用同一个 birth state 和累计 age。

---

## 14. 验证计划与验收标准

### 14.1 数学一致性

- $H(0)=0$；
- $H(q)\ge0$；
- $H_q(q)\ge0$；
- $T(q)=e^{-H(q)}\in[0,1]$；
- $p(q)=H_q(q)e^{-H(q)}\ge0$；
- 对有限 $q_{\max}$：

  $$
  \int_0^{q_{\max}}p(q)\,dq+T(q_{\max})=1.
  $$

### 14.2 数据生成收敛

- 至少三组时间分辨率；
- survival、FPT quantile 和 crossing-slope 分布随步长收敛；
- Monte Carlo 与 absorbing PDE 参数切片一致；
- 尾部只在足够 risk set 范围内报告误差。

### 14.3 坐标与尺度不变性

- global frame 与旋转后的 local frame 得到相同 survival；
- 同一组无量纲参数在不同 $\sigma,\ell$ 下曲线重合；
- 对各向同性 kernel，绕 $\mathbf d$ 旋转不改变 survival。

### 14.4 渲染一致性

- `classic_local` 和 `classic_global` 不受新 transport mode 影响；
- 固定 seed 下累计消光反演具有确定性；
- 分段查询满足

  $$
  T(0,t+\Delta)=T(0,t)T(t,t+\Delta\mid\tau>t);
  $$

- first-passage distance histogram 与离线目标一致；
- 碰撞纵向梯度始终满足向下穿越符号；
- white-furnace/单位 Fresnel 测试不存在由采样器引入的能量漂移；
- 与当前 SE `global_conditional` 在可比较参数切片上分别报告差异，而不是把差异当成数值误差隐藏。

### 14.5 建议误差指标

- survival 的最大绝对误差和加权相对误差；
- cumulative hazard 的 RMSE；
- FPT quantile error；
- censored negative log-likelihood；
- crossing-slope conditional quantile error；
- 最终图像的均值、方差和置信区间差异。

---

## 15. 分阶段实施顺序

### 阶段 A：数学与数据生成基线

1. 固定各向同性 Matérn $3/2$；
2. 固定 affine mean；
3. 实现二维状态空间精确 transition；
4. 输出 FPT/censor/crossing-slope 原始样本；
5. 完成步长收敛和 PDE 小规模验证。

### 阶段 B：累计消光拟合

1. 先实现状态网格上的单调 spline/LUT；
2. 使用 censored survival likelihood；
3. 实现 $H$、$H_q$ 和 $H^{-1}$；
4. 检查归一化、单调性和尾部误差；
5. 再决定是否需要积分神经网络。

### 阶段 C：碰撞梯度闭合

1. 拟合 first-passage 纵向 crossing slope；
2. 实现横向梯度解析条件采样；
3. 合成完整梯度并接入 conductor phase；
4. 验证梯度符号、法线分布和能量一致性。

### 阶段 D：渲染模式接入

1. 新增独立 `global_collision_state` mode；
2. 主射线先保留现有第一次碰撞方案；
3. 次级射线使用累计消光反演；
4. 真实碰撞处更新 birth state；
5. 中间位置绝不重启条件过程；
6. 输出每条路径的状态范围和 out-of-domain 诊断。

### 阶段 E：扩展

在基线通过后再考虑：

- 外部 entry-state first passage；
- 非线性 mean profile；
- 各向异性 Matérn；
- Matérn $5/2$；
- 用统一 surrogate 覆盖多个 kernel family。

---

## 16. 最终决策摘要

第一版采用以下明确约束：

1. 每条次级射线只条件于最近碰撞的 $F=0$ 和完整梯度；
2. survival 只读取 $g_d=\mathbf g\cdot\mathbf d$，但完整梯度保留给下一碰撞法线；
3. mean 在碰撞点线性化一次，并在整条自由程内保持不变；
4. kernel 固定为各向同性 Matérn $3/2$；
5. 离线生成连续 first-passage 与 crossing-slope 数据；
6. 拟合单调累计消光 $H$，不直接拟合 noisy hazard；
7. 渲染时通过 $H^{-1}(-\log u)$ 采样自由程；
8. 只有在真实碰撞处才建立新的碰撞状态；
9. 主相机射线的首次碰撞暂时沿用现有方案；
10. 在完成纵向 crossing-slope 模型前，不把该方案宣称为完整闭合的渲染 mode。

这套设计保留了真正的 first-passage transmittance，避免直接微分 Monte Carlo survival 所造成的 hazard 波动，同时清楚隔离了历史丢弃、一阶 mean、拟合近似和碰撞梯度闭合四个误差来源。

---

## 17. 当前实现状态

阶段 A 的训练数据生成器已经并入现有 `first-passage` 实验，而不是另建一套不可比较的命令。配置入口为：

```json
"initial_condition": {
  "type": "collision_state",
  "parameter_space": { ... }
},
"sampler": {
  "type": "collision_state_auto"
}
```

参考配置是 `configs/collision_state_kernels_training.json`。运行：

```powershell
build\Release\macrofacet_experiments.exe first-passage `
  --config configs\collision_state_kernels_training.json
```

已实现内容包括：

1. `explicit`、`cartesian` 和 `latin_hypercube` 三种 \((\beta_0,\beta_a,\beta_g)\) 参数设计；
2. 所有受支持 kernel 使用统一的 collision-state 配置和 CSV schema，不保留旧的 `collision_state_matern32` 类型别名；
3. Matérn \(3/2\) 从 \((y,w)=(-\beta_0,\beta_g-\beta_a)\) 启动二维精确 Gaussian transition，并使用条件 midpoint bridge、自适应 refinement 和 cubic-Hermite crossing；
4. squared exponential、Matérn \(5/2\) 和 rational quadratic 使用起点值/有限差分导数联合条件化的 circulant grid 与 cubic-Hermite crossing，并依靠多步长检查离散收敛；
5. FPT、right censor、无量纲 crossing slope、物理 crossing derivative、逐样本 seed 和数值诊断输出；
6. 多 `grid.step_sizes` 收敛数据，最小步长用 `training_resolution=1` 标识；
7. risk-set survival、density、hazard、Nelson--Aalen/product-limit 累计 hazard，以及 FPT/slope quantile 汇总；
8. `sampler_type` 和 `crossing_slope_method` 在每类输出中记录实际数值后端；
9. 可配置为固定状态跨 kernel，或固定 kernel 跨碰撞状态的 survival、hazard 和累计 hazard SVG；
10. 不同 \(\sigma,\ell\) 使用同一无量纲随机流，便于直接验证尺度不变性；
11. 旧的多 kernel `fixed_value + exact_grid_circulant` 模式保持兼容。

当前输出足以训练阶段 B 的

\[
H(q\mid\beta_0,\beta_a,\beta_g)
\]

以及阶段 C 的纵向 crossing-slope 条件模型。吸收 PDE 切片验证、累计消光拟合器本身和渲染 transport mode 仍分别属于阶段 A 的独立验证项、阶段 B 和阶段 D，尚未由数据生成器替代。

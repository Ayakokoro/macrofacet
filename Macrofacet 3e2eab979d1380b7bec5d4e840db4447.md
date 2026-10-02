# Macrofacet

# **A Radiative Transfer Framework for Spatially-Correlated Materials Theory**

## Building block

$$
\underbrace{
\frac{\partial L}{\partial s}+\omega\cdot\nabla L
}_{\text{传播}}
+
\underbrace{\Sigma(\mathbf x,\omega,s)L}_{\text{消光}}
=0
$$

$$
L(\mathbf x,\omega_o,0)
=
\int_0^\infty\!\!\int_{\Omega}
\underbrace{
B(\mathbf x,\omega_i,\omega_o,s)
}_{\text{散射算子}}
L(\mathbf x,\omega_i,s)\,d\omega_i\,ds
+Q
$$

$$

B = \Sigma\,\Lambda\,f_r

$$

- $s$ 为上一次散射事件后的传播距离
- $\Lambda$：为albedo
- $f_r$：为phase function
- $\Sigma(x,\omega,s)$：光子已经无碰撞地传播了 $s$，接下来单位距离内发生消光事件的概率是多少
- $B(x,ω_i,ω_o,s)$：发生相互作用后，有多少光被散射到方向 $ω_o$

这里暂时省略位置和方向依赖，$k$ 表示光子的起点类别scatter或者source

消光系数 $\Sigma_k(s)$，free-space pdf $p_k(s)$, transmittance $T_k(s)$ 之间的关系如下

$$
p_k(s)=-\frac{dT_k(s)}{ds}
$$

$$
\Sigma_k(s)
=-\frac{d}{ds}\ln T_k(s)
=\frac{p_k(s)}{T_k(s)}
$$

$$
\Sigma_k(s)
=
\frac{p_k(s)}
{1-\int_0^s p_k(u)\,du}
$$

实际只需要定义其中一个

# Macrofacet theory

原始论文实际上用了两个假设近似

1. 用单点的概率密度替换条件概率密度
    
    ![image.png](image.png)
    
    ![image.png](image%201.png)
    
2. 认为统计信息和出发点不相关
    
    ![image.png](image%202.png)
    

## 按照GBE的理论来说，实际上准确建模透射率的推导应该如下

先简单假设 scatter 和 source 的三个函数是一样的，所以可以不区分出发点

设

$\mathbf x_t = \mathbf x_i+t\omega_o \qquad k(\mathbf x_i) = \omega_o^Tg(\mathbf x_i)$

用 $\mathcal H$ 表示出发时保留的条件。例如保留上一交点的值和梯度

$$
\mathcal H=\{f(\mathbf x_i)=0,\ \nabla f(\mathbf x_i)=g(\mathbf x_i)\},
\qquad k(\mathbf x_i)>0.
$$

首先 $\alpha$ 事件应改为 $\alpha_t$ ：

光线从 $\mathbf x_i$ 出发，到 $\mathbf{x}_t$ 为止，整段均未被表面遮挡:

$$
\alpha_t
=
\{f(\mathbf x_i+u\omega_o) > 0,\ \forall\,0<u\le t\}
=
\{\tau>t\}

$$

- $\tau=\inf\{u>0:f(\mathbf x_i+u\omega_o) \le 0\}$：是首次碰撞距离。因此 $P(\alpha_t\mid\mathcal H)
=
T(t\mid\mathcal H)$

而原论文中的 $f(\mathbf x_t)>0$ 条件不能排除“此前碰撞过、后来又回到正侧”的 realization；

另外定义事件 $\beta_{t,\Delta t}$：在接下来长度为  $\Delta t$ 的区间内遇到表面(这和原始论文的定义一致)

$\beta_{t,\Delta t}
=
\{\exists\,u\in(t,t+\Delta t]:f(\mathbf x_i+u\omega_o) \le 0\}$

与 $\alpha_t$ 联合后，就得到严格的首次碰撞事件：

$$
\alpha_t\cap\beta_{t,\Delta t}
=
\{t<\tau\le t+\Delta t\}
$$

进行一阶展开

$$
f(\mathbf x_t +\Delta t\omega_o)
=
f(\mathbf x_t )+k(\mathbf x_t)\Delta t+o(\Delta t)
$$

因此，对已经存活到 $t$ 的路径，下一小段被表面挡住的条件为：

$$
k(\mathbf x_t)<0,\qquad
0<f(\mathbf x_t)<-k(\mathbf x_t)\Delta t

$$

于是准确的条件消光系数应写为

$$
\begin{aligned}
\Sigma(t\mid\mathcal H)
&=
\lim_{\Delta t\rightarrow0}
\frac{
P(\beta_{t,\Delta t}\mid\alpha_t,\mathcal H)
}{\Delta t}\\[2mm]
&=
\lim_{\Delta t\rightarrow0}
\frac{
P(\alpha_t\cap\beta_{t,\Delta t}\mid\mathcal H)
}{
\Delta t\,P(\alpha_t\mid\mathcal H)
}.
\end{aligned}
$$

这里省略了 $\mathbf x_i,\omega_o$ 的依赖

可以与论文中的(26)进行对比

![image.png](image%203.png)

定义尚未碰撞的值和梯度的联合密度

$$
p'(f,g;t\mid\mathcal H)\,df\,dg
=
P(f(\mathbf x_t)\in df,g(\mathbf x_t)\in dg,\alpha_t\mid\mathcal H)
$$

消光系数的分母可以写成(也就是对应的条件透射率)

$$
T(t\mid\mathcal H)
=
\int_{\mathbb R^3}\int_0^\infty
p'(f,g;t\mid\mathcal H)\,df\,dg
$$

消光系数的分子可以写成

$$
\begin{aligned}
&P(\alpha_t\cap\beta_{t,\Delta t}\mid\mathcal H)
\\
&=
\int_{\omega_o^Tg<0}
\int_0^{-(\omega_o^Tg)\Delta t}
p'(f,g;t\mid\mathcal H)\,df\,dg
+o(\Delta t)
\\
&=
\Delta t
\int_{\omega_o^Tg<0}
(-\omega_o^Tg)\,
p'(0,g;t\mid\mathcal H)\,dg
+o(\Delta t).
\end{aligned}
$$

最后得到

$$
\Sigma(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p'(0,g;t\mid\mathcal H)\,dg
}{
\displaystyle
\int_{\mathbb R^3}\int_0^\infty
p'(f,g;t\mid\mathcal H)\,df\,dg
}
$$

由此即可推出其他两个函数

也可以具体分解成端点的统计乘上整段无遮挡权重的形式

定义

$$
O(t,g\mid\mathcal H)
=
P\!\left(
\alpha_t
\mid f(\mathbf x_t) =0,g(\mathbf x_t)=g,\mathcal H
\right)
$$

$$
p'(0,g;t\mid\mathcal H)
=
p(f(\mathbf x_t)=0 ,g(\mathbf x_t)=g\mid\mathcal H)
O(t,g\mid\mathcal H).
$$

所以

$$
\Sigma(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p(f(\mathbf x_t) =0,g(\mathbf x_t)=g\mid\mathcal H)\,
O(t,g\mid\mathcal H)\,dg
}{
\displaystyle
P(\alpha_t\mid\mathcal H)
}
$$

**论文第一点假设:整段存活的信息，可以用当前点没被遮挡近似**

根据条件概率

$$
P(f(\mathbf x_t)\in df,g(\mathbf x_t)\in dg,\alpha_t\mid\mathcal H)
=
P(\alpha_t\mid \mathcal H)
P(f(\mathbf x_t)\in df,g(\mathbf x_t)\in dg\mid \alpha_t,\mathcal H)
$$

因此

$$
p'(f,g;t\mid\mathcal H)
=
P(\alpha_t\mid\mathcal H)\,
p(f,g\mid\alpha_t,\mathcal H)
$$

**第一点假设**说的是

$$
p(f,g\mid\alpha_t,\mathcal H)
\;\approx\;
p(f,g\mid f(\mathbf x_t)>0,\mathcal H)
$$

令端点联合密度为

$$
p_{\mathcal H}(f,g;t)
=
p(f(\mathbf x_t)=f,g(\mathbf x_t)=g\mid\mathcal H)
$$

$$
p'(f,g;t\mid\mathcal H)
\approx

\frac{P(\alpha_t\mid\mathcal H)}
{P(f(\mathbf x_t)>0\mid\mathcal H)}

p_{\mathcal H}(f,g;t) \qquad f>0
$$

代入原式就得到论文的结果

$$
\Sigma_1(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p_{\mathcal H}(0,g;t)\,dg
}{
\displaystyle
\int_{\mathbb R^3}\int_0^\infty
p_{\mathcal H}(f,g;t)\,df\,dg
}
$$

- 这一步仍然保留起点条件,因此它还可以依赖上一交点的梯度、传播方向和距离。

**论文第二点假设：当前点与出发点去相关**

即

$$
p_{\mathcal H}(f,g;t)
\;\approx\;
p(f(\mathbf x_t)=f,g(\mathbf x_t)=g)
$$

这里右侧仍是位置 $\mathbf x_t$ 处的局部分布，只是不再依赖起点观测值。

代入得到

$$
\Sigma_2(x_t,\omega_o)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p(f(\mathbf x_t)=0,g(\mathbf x_t)=g)\,dg
}{
P(f(\mathbf x_t)>0)
}
$$

这一步删除了起点和终点之间的条件相关性

**在这两种假设下可以用第一篇论文的理论得到和macrofacet相同的结果**

对于stationary SE 协方差

由于 $f(\mathbf x_t),g(\mathbf x_t)$ 联合高斯，零协方差意味着独立：

$\operatorname{Cov}(f(\mathbf x_t),g(\mathbf x_t))
=
\left.\nabla_y\kappa(x,y)\right|_{y=x}
=0$

$$
p(f(\mathbf x_t)=0,G(\mathbf x_t)=g)=p(f(\mathbf x_t)=0)p(g(\mathbf x_t)=g)
$$

因此消光系数进一步变成

$$
\Sigma_2(x,\omega_o)
=
\underbrace{
\frac{p(f(\mathbf x_t)=0)}{P(f(\mathbf x_t)>0)}
}_{\rho(x)}
\underbrace{
\mathbb E[(-\omega_o^Tg(\mathbf x_t))_+]
}_{\sigma_{\mathrm{proj}}(x,\omega_o)}
$$

若 $f(\mathbf x_t)\sim\mathcal N(m(\mathbf x_t),\sigma^2)$，则

$$
\rho(\mathbf x_t)
=
\frac{\phi(m(\mathbf x_t)/\sigma)}
{\sigma\,\Phi(m(\mathbf x_t)/\sigma)}
$$

在论文的模型中，这对应其式 (31) 的

$$
\sigma_t(\omega_o,t)=\rho(t)\sigma(\omega_o)
$$

## 方案

设 

$F_t = f( \mathbf x_i + t \omega_o), \qquad G_t = g( \mathbf x_i + t \omega_o), \qquad k_t = k(\mathbf x_i +t\omega_o)$

### 去掉第二点假设

设

得到

$$
p_{\mathcal H}(f,g;t)=p(F_t=f,G_t=g\mid\mathcal H)
$$

$$
\Sigma_1(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p_{\mathcal H}(0,g;t)\,dg
}{
\displaystyle
\int_{\mathbb R^3}\int_0^\infty
p_{\mathcal H}(f,g;t)\,df\,dg
}
$$

推导

### 去掉第一点和第二点假设

$$
\Sigma(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p(f(\mathbf x_t) =0,g(\mathbf x_t)=g\mid\mathcal H)\,
O(t,g\mid\mathcal H)\,dg
}{
\displaystyle
P(\alpha_t\mid\mathcal H)
}
$$

方法主要参考的是这两篇论文

Gaussian Integrals and Rice Series in
Crossing Distributions—to Compute the
Distribution of Maxima and Other Features
of Gaussian Processes

Effective computations of joint excursion times for stationary
Gaussian processes

#### Rice理论进行分解

实际上,主要的困难是 $O$ 和分母项的计算

**$O(t,g\mid\mathcal H)$**计算的是固定两端条件后，中间没有提前碰撞的概率

其定义是

$$
O(t,g\mid\mathcal H)
=
P\!\left(
\alpha_t
\mid F_t=0,G_t=g,\mathcal H
\right)
$$

可以用Rice理论来解释，先把GP限制到ray上，也就是 $F_u$  是一个 1D GP，沿 ray 的导数为：

$$
\dot F_u = \frac{dF}{du} =  G_u^T\omega_o
$$

free-space 为

$$
F_u>0
$$

从起始 surface 点离开后：

$$
F_0=0,\qquad F_u>0
$$

到了下一次 surface intersection：

$$
F_t=0
$$

对于一个非切向 hit，应当有

$$
\dot F_t<0
$$

Rice论文定义的是upcrossing，也就是由负到正的穿越，所以在这里重新定义

$$
X_u = - F_u
$$

那么 free space 变成

$$
X_u<0
$$

下一碰撞变成从负到正的 upcrossing：

$$
X_t=0,
\qquad
\dot X_t>0
$$

而

$$
\dot X_t=-\omega_o^TG_t
$$

定义

$$
v_t=-\omega_o^TG_t
$$

那么合法的 first hit就为：

$$
v_t>0
$$

先考虑 $t$ 附近发生一个 upcrossing，而且 endpoint gradient 为 $g$。

Rice crossing intensity 为

$$
W_1(t,g|\mathcal H)=v_t\,
p\!\left(
F_t=0,G_t=g
\mid\mathcal H
\right),
\qquad
v_t=-\omega_o^TG_t>0.
$$

这和

$$
\Sigma_1(t\mid\mathcal H)
=
\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^Tg)_+\,
p_{\mathcal H}(0,g;t)\,dg
}{
\displaystyle
\int_{\mathbb R^3}\int_0^\infty
p_{\mathcal H}(f,g;t)\,df\,dg
}
$$

的分子中的被积函数（也就是 $O(t,g\mid\mathcal H)$ 恒等于 1）是一样的

但是 $W_1$ 不管 $(0,t)$ 中以前是否已经有过 crossing。

所以需要定义 $W_2$：一个 earlier crossing + endpoint crossing

假设在

$$
u<t
$$

还发生过一次 upcrossing

那么 joint two-crossing Rice density 是

$$
\begin{aligned}
W_2(u,t,g|\mathcal H)
={}&
v_t
\int_0^\infty
v_u\\
&\quad
p\!\left(
X_u=0,\dot X_u=v_u,\,
F_t=0,G_t=g
\mid\mathcal H
\right)
dv_u.
\end{aligned}
$$

它的意义是：

在 $u$ 附近有一次 upcrossing，同时在 $t$ 附近又有一次具有 gradient $g$ 的 upcrossing

因此：

$$
\int_0^t
W_2(u,t,g|\mathcal H)\,du
$$

统计的就是 $t$ 处虽然撞到了 surface，但此前已经至少发生过一个 candidate crossing。这些不能被认为是 first hit，所以应该从 $W_1$ 里减掉。

以此类推

$$
\begin{aligned}
&W_3(u_1,u_2,t,g|\mathcal H)\\
&=
v_t
\int_0^\infty\int_0^\infty
v_1v_2\\
&\qquad\times
p\big(
X_{u_1}=0,\dot X_{u_1}=v_1,\\
&\qquad\qquad
X_{y_2}=0,\dot X_{u_2}=v_2,\\
&\qquad\qquad
F_t=0,G_t=g
\mid\mathcal H
\big)
dv_1dv_2.
\end{aligned}
$$

一般的 $W_{n+1}$ 有 $n$ 个 earlier crossings：

$$
0<u_1<\cdots<u_n<t
$$

那么：

$$
\begin{aligned}
&W_{n+1}
(u_1,\ldots,u_n,t,g|\mathcal H)\\
={}&
v_t
\int_0^\infty\cdots\int_0^\infty
\left(
\prod_{j=1}^n v_j
\right)\\
&\times
p\Big(
X_{u_1}=0,\dot X_{u_1}=v_1,\ldots,\\
&\qquad
X_{u_n}=0,\dot X_{u_n}=v_n,\\
&\qquad
F_t=0,G_t=g
\mid\mathcal H
\Big)
\prod_{j=1}^n dv_j.
\end{aligned}
$$

现在定义：

$$
J(t,g|\mathcal H)
$$

为第一次 surface intersection 恰好发生在 $t$，并且 gradient 为 $g$ 的 joint density/intensity。

由 inclusion–exclusion：

$$
\begin{aligned}
J(t,g|\mathcal H)
={}&
W_1(t,g|\mathcal H)\\
&-
\int_0^t
W_2(u_1,t,g|\mathcal H)\,du_1\\
&+
\int_{0<u_1<u_2<t}
W_3(u_1,u_2,t,g|\mathcal H)
\,du_1du_2\\
&-\cdots.
\end{aligned}
$$

这就是 Rice first-passage infinite series。

最后带入 $O$ 得到

$$
O(t,g|\mathcal H)
=
\frac{
J(t,g|\mathcal H)
}{
W_1(t,g|\mathcal H)
}
$$

最终

$$
\Sigma(t\mid\mathcal H)=\frac{
\displaystyle
\int_{\mathbb R^3}J(t,g\mid\mathcal H)\,dg
}{
P(\alpha_t\mid\mathcal H)}

$$

定义

$$
\mathcal N(t\mid\mathcal H)
=
\int_{\mathbb R^3}
J(t,g\mid\mathcal H)\,dg.
$$

代入 Rice expansion：

$$
\begin{aligned}
\mathcal N(t\mid\mathcal H)
=
&
\int_{\mathbb R^3}
W_1(t,g\mid\mathcal H)\,dg
\\
&-
\int_{\mathbb R^3}
\int_0^t
W_2(u_1,t,g\mid\mathcal H)
\,du_1\,dg
\\
&+
\int_{\mathbb R^3}
\int_{0<u_1<u_2<t}
W_3(u_1,u_2,t,g\mid\mathcal H)
\,du_1du_2\,dg
-\cdots.
\end{aligned}
$$

定义

$$
\overline W_n
=
\int_{\mathbb R^3}W_n\,dg,
$$

那么

$$

\begin{aligned}
\mathcal N(t\mid\mathcal H)
=
&
\overline W_1(t\mid\mathcal H)
\\
&-
\int_0^t
\overline W_2(u_1,t\mid\mathcal H)\,du_1
\\
&+
\int_{0<u_1<u_2<t}
\overline W_3(u_1,u_2,t\mid\mathcal H)\,du_1du_2
-\cdots
\end{aligned}

$$

因此

$$

\Sigma(t\mid\mathcal H)
=
\frac{
\displaystyle
\overline W_1
-
\int\overline W_2
+
\iint\overline W_3
-\cdots
}{
P(\alpha_t\mid\mathcal H)
}
$$

在 GP 定义下

$$
f(\mathbf x)\sim GP(m(x),k(x,y))
$$

那么任意有限组

$$
\begin{bmatrix}
F_{u_1}\\
\dot F_{u_1}\\
\vdots\\
F_{u_n}\\
\dot F_{u_n}\\

\end{bmatrix}
$$

都是 joint Gaussian。对应 covariance 直接从 kernel derivative 得到：

$$
\operatorname{Cov}[F_u,F_v]=k(\mathbf x_u,\mathbf x_v)
$$

$$
\operatorname{Cov}[F_u,\dot F_v]=\omega_o^T
\nabla_{\mathbf x_v}
k(\mathbf x_u,\mathbf x_v)
$$

$$
\operatorname{Cov}[\dot F_u,\dot F_v]=\omega_o^T
\nabla_{\mathbf x_u}
\nabla_{\mathbf x_v}^T
k(\mathbf x_u,\mathbf x_v)
\omega_o
$$

然后条件化：

$$
\mathcal H = \{
f(\mathbf x_i)=0,
\nabla f(\mathbf x_i)=g_i
\}
$$

以及 endpoint：

$$
F_t=0,\qquad G_t=g.
$$

仍然只是标准 Gaussian conditioning。

但是算无限和还是不现实（因为n越大，联合高斯的维度也越大）

Effect of Correlation Between Shadowing and
Shadowed Points on the Wagner and Smith
Monostatic One-Dimensional Shadowing Functions

这篇文章有对误差的估计（和准确的集合平均的对比）

![image.png](image%204.png)

![image.png](image%205.png)

对于 $\overline W_n$ 是有递推式的，主要困难在于：

1. 需要计算高维度联合高斯正值域的积分作为base case
2. 递推会消耗很多性能

#### 离散选点求多维高斯分布

$$
0<u_1<\cdots<u_N<t,
\qquad
V=(F_{u_1},\ldots,F_{u_N})^T
$$

先将 $(V,Z_t)$ 对 $\mathcal H$ 条件化，得到

$$
\begin{pmatrix}V\\Z_t\end{pmatrix}\Bigm|\mathcal H
\sim
\mathcal N\!\left[
\begin{pmatrix}\bar m_V\\\bar m_Z\end{pmatrix},
\begin{pmatrix}
\bar C_{VV}&\bar C_{VZ}\\
\bar C_{ZV}&\bar C_{ZZ}
\end{pmatrix}
\right]
$$

再对终点

$$
Z_t=z_g=\begin{pmatrix}0\\g\end{pmatrix}
$$

条件化：

$$
\mu_B(g)
=
\bar m_V+
\bar C_{VZ}\bar C_{ZZ}^{-1}(z_g-\bar m_Z)
$$

$$
C_B
=
\bar C_{VV}
-\bar C_{VZ}\bar C_{ZZ}^{-1}\bar C_{ZV}
$$

然后计算

$$
W_N(t,g)
=
P(V>0\mid Z_t=z_g,\mathcal H)
$$

这是一个 $N$ 维高斯正值域概率。令

$$
D_B=\operatorname{diag}
\big(\sqrt{(C_B)_{11}},\ldots,\sqrt{(C_B)_{NN}}\big),
\qquad
R_B=D_B^{-1}C_BD_B^{-1}
$$

则

$$
W_N(t,g)
=
\Phi_N(D_B^{-1}\mu_B(g);R_B)
$$

**但实际上这和GPIS的Volume-type遇到的first-passage-time问题是一样的,在GPIS的定义下**

GPIS论文式 (11) 定义了整段无遮挡的指示函数：

$$
I^f(0,t)=
\begin{cases}
1,& f(x_s)>0,\quad \forall s\in(0,t),\\
0,& \text{其他情况}.
\end{cases}
$$

式 (13) 定义终点条件

$$
\zeta_\delta
=
\{f(x_t)=0,\ \nabla f(x_t)=n\}
$$

然后，式 (18) 写出

$$
\mathrm T(x_t\mid\zeta)
=
\int
I^f(0,t)\,
d\gamma(f_{(x,x_t)}\mid\zeta\wedge\zeta_\delta)
=
P\!\left(f_{(x,x_t)}>0\mid\zeta\wedge\zeta_\delta\right)
$$

这是固定终点条件后，整段仍然为正的概率。

按论文写出的梯度条件，对应关系是：

- $\mathcal H \Leftrightarrow \zeta$
- $F_t=0,G_t=g \Leftrightarrow \zeta_\delta$
- $\text{整段为正} \Leftrightarrow I^f(0,t)=1$
- $O(t,g\mid\mathcal H) \Leftrightarrow T(x_t\mid\zeta)$

GPIS原论文提到：

Since the only Gaussian Markov process is the OU process, which is not smooth,there are  no Gaussian processes for which we (1) can hope to derive a non zero particle/particle free-flight density and (2) for which such a free-flight density has a known analytic form.

论文中提到的方法是用RIND数值计算出自由程分布统计，然后进行尾部拟合得到支持传统的指数介质的free-flight pdf。

RIND本身就能计算这个积分（QMC方法也可以）。同样，在低维的时候有闭式解（N=1，N=2, N=3时如果符合均值为0的高斯分布下也有）可以直接计算。

### 还有另一种方向:通过经验模型显式建模相关性

也就是在macrofacet的统计近似下显式引入相关性,参考的是A Non-Exponential Transmittance Model
for Volumetric Scene Representations.

原模型的光学厚度 

$$
\tau(t)
=
\int_0^t\rho(s)\sigma(\omega_o)\,ds
$$

那么原模型实际上是认为

$$
\mathrm{Tr}(t)=e^{-\tau(t)}
$$

经验模型的透射率可以定义为

$$
\mathrm{Tr}(x_i,\omega_o,t)
=
f(\tau(t)\big)

$$

比如 Gamma 分布

$$
\psi_\nu(\tau)
=
\frac{\nu^\nu}{\Gamma(\nu)}
\tau^{\nu-1}e^{-\nu\tau},
\qquad \nu\ge1
$$

$$
\mathbb E[\tau]=1,
\qquad
\operatorname{Var}(\tau)=\frac1\nu
$$

因此：

- $\nu=1$：原 macrofacet
- $\nu>1$：抑制很短的距离再碰撞，间距更均匀
- $\nu$  增大：碰撞距离更集中，但均值保持不变

定义

$S_\nu(\tau)
=
\int_\tau^\infty\psi_\nu(u)\,du
=
\frac{\Gamma(\nu,\nu\tau)}{\Gamma(\nu)}$

则经验模型的透射率可以定义为

$$
\mathrm{Tr}^{(\nu)}(x_i,\omega_o,t)
=
S_\nu\big(\tau(t)\big).
$$

得到

$$
\sigma_t^{(\nu)}(\omega_o,t)
=
\underbrace{\rho(t)\sigma(\omega_o)}_{\text{原局部消光}}
\underbrace{
\frac{\psi_\nu(\tau(t))}
{S_\nu(\tau(t))}
}_{\text{距离修正}}
$$

### Results

![Snipaste_2026-09-30_06-54-38.png](Snipaste_2026-09-30_06-54-38.png)

原论文

![shader_ball.png](shader_ball.png)

复现

![render_classic_local.bmp](render_classic_local.bmp)

global

![render_classic_global_directional.bmp](render_classic_global_directional.bmp)

加上条件化

![render_global_conditional_directional.bmp](render_global_conditional_directional.bmp)

一个比较简单的平面场测试

![transmittance_surface_outgoing_gx_010.svg](transmittance_surface_outgoing_gx_010.svg)

方案

1. local：引入曲率，在local计算条件化的消光系数等
2. global的情况下去掉所有假设准确计算消光系数，了解Rice分解的多少阶能足够近似，或者RIND方法中用多少个采样点能够近似得到准确的消光系数

问题

local non-stationary的问题

原论文的渲染结果在边缘会比较hard，因为pbrt本身用原有物体来做ray的剔除，我自己实现的是没用原有物体剔除的，我不太清楚具体哪个是正确的。
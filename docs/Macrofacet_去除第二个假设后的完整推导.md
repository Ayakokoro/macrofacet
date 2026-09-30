# Macrofacet：去除第二个假设后的透射率、自由程采样、NDF、相函数与估计器

本文沿 Macrofacet 理论的推导过程，保留起点与终点之间的条件相关性，推导相应的消光系数、透射率、自由程 PDF、NDF、采样算法、反射相函数及 Monte Carlo 估计器权重。原论文公式编号对应上传的 macrofacet(4).pdf 版本。[^paper]

按照附带研究笔记的划分，“去除第二个假设”具体是：**保留起点与终点之间的条件相关性，但仍用“当前点处于空侧”近似“此前整段未碰撞”。**因此，得到的是一个保留起点相关性的 Macrofacet 模型；其中高斯条件分布的计算是精确的，透射率仍保留第一个近似。[^notes]

推导后的消光系数为

$$
\boxed{
\widetilde\Sigma(t\mid H)
=
\underbrace{\frac{p(F_t=0\mid H)}{P(F_t>0\mid H)}}_{\rho_H(t)}
\underbrace{
\mathbb E[(-\omega^\mathsf TG_t)_+\mid F_t=0,H]
}_{\text{条件投影面积}}
}
$$

下文将推导与它一致的透射率、自由程 PDF、NDF 和碰撞法线 PDF。原文也指出，一旦保留起点相关性，消光系数就会依赖光线起点。[^paper]

---

先固定记号。令光线为

$$
x_t=x_i+t\omega,\qquad \|\omega\|=1,
$$

并定义

$$
F_t=f(x_t),\qquad
G_t=\nabla f(x_t),\qquad
K_t=\omega^\mathsf TG_t.
$$

约定 $f>0$ 是空侧，法线朝向空侧。起点条件为

$$
\boxed{
H=\{F_0=0,\;G_0=g_i\},
\qquad
k_i:=\omega^\mathsf Tg_i>0.
}
$$

即光线从表面出发，进入空侧。这里 $g_i$ 是**完整梯度**，包含模长。

假设 GP 足够光滑，零交叉非退化。定义下一次碰撞距离

$$
\tau=\inf\{t>0:F_t\leq 0\}.
$$

真正的透射率是

$$
T_{\mathrm{true}}(t\mid H)=P(\tau>t\mid H).
$$

本次保留的第一个近似可以明确写成

$$
\boxed{
p(F_t,G_t\mid \tau>t,H)
\approx
p(F_t,G_t\mid F_t>0,H).
}
\tag{1}
$$

而被去掉的第二个近似是

$$
p(F_t,G_t\mid H)\approx p(F_t,G_t).
$$

以下用波浪号标记由式（1）构成的输运模型。

---

首先推导消光系数。这对应原文式（23）—（29）。

记保留起点条件的端点联合 PDF 为

$$
p_H(f,g;t)
=
p(F_t=f,G_t=g\mid H),
$$

并令

$$
Q_H(t)=P(F_t>0\mid H)
=
\int_{\mathbb R^3}\int_0^\infty
p_H(f,g;t)\,df\,dg.
$$

由局部 Taylor 展开，

$$
F_{t+\Delta t}
=
F_t+K_t\Delta t+o(\Delta t).
$$

若在接下来的一小段中从空侧穿入负侧，则一阶条件为

$$
K_t<0,\qquad
0<F_t<-K_t\Delta t.
$$

因此，在式（1）的近似下，

$$
\begin{aligned}
\widetilde\Sigma(t\mid H)\Delta t
&=
\frac{1}{Q_H(t)}
\int_{\omega^\mathsf Tg<0}
\int_0^{-\omega^\mathsf Tg\,\Delta t}
p_H(f,g;t)\,df\,dg
+o(\Delta t)
\\
&=
\frac{\Delta t}{Q_H(t)}
\int_{\mathbb R^3}
(-\omega^\mathsf Tg)_+
p_H(0,g;t)\,dg
+o(\Delta t).
\end{aligned}
$$

除以 $\Delta t$ 并取极限：

$$
\boxed{
\widetilde\Sigma(t\mid H)
=
\frac{
\displaystyle\int_{\mathbb R^3}
(-\omega^\mathsf Tg)_+p_H(0,g;t)\,dg
}{
P(F_t>0\mid H)
}.
}
\tag{2}
$$

这就是**保留起点条件的原文式（29）**。接下来不再使用原文式（30）的去相关操作，而是直接计算这个条件高斯分布。

利用概率密度的分解

$$
p_H(0,g;t)
=
p(F_t=0\mid H)\,
p(G_t=g\mid F_t=0,H),
$$

式（2）成为

$$
\boxed{
\widetilde\Sigma(t\mid H)
=
\rho_H(t)B_H(t),
}
\tag{3}
$$

其中

$$
\rho_H(t)
=
\frac{p(F_t=0\mid H)}{P(F_t>0\mid H)},
$$

$$
B_H(t)
=
\mathbb E[(-K_t)_+\mid F_t=0,H].
$$

这里 $p(F_t=0\mid H)$ 表示在零点处的**密度值**。

---

现在计算所需的条件高斯分布。

设原始 GP 为

$$
f(x)\sim\mathcal{GP}(m(x),\kappa(x,y)).
$$

把场值和梯度合并为

$$
Y_t=
\begin{pmatrix}
F_t\\G_t
\end{pmatrix},
\qquad
h_i=
\begin{pmatrix}
0\\g_i
\end{pmatrix}.
$$

它的先验均值为

$$
\overline Y_t=
\begin{pmatrix}
m(x_t)\\ \nabla m(x_t)
\end{pmatrix},
$$

协方差块由核及其导数决定：

$$
\mathcal K(x,y)=
\begin{pmatrix}
\kappa(x,y)&\nabla_y^\mathsf T\kappa(x,y)\\
\nabla_x\kappa(x,y)&
\nabla_x\nabla_y^\mathsf T\kappa(x,y)
\end{pmatrix}.
$$

令 $\mathcal K_{ab}=\mathcal K(x_a,x_b)$。标准高斯条件化给出

$$
\boxed{
\mathbb E[Y_t\mid H]
=
\overline Y_t+
\mathcal K_{t0}\mathcal K_{00}^{-1}
(h_i-\overline Y_0),
}
\tag{4}
$$

$$
\boxed{
\operatorname{Cov}(Y_t\mid H)
=
\mathcal K_{tt}
-
\mathcal K_{t0}\mathcal K_{00}^{-1}\mathcal K_{0t}.
}
\tag{5}
$$

将结果记为

$$
\begin{pmatrix}F_t\\G_t\end{pmatrix}\Bigm|H
\sim
\mathcal N\left[
\begin{pmatrix}m_F\\m_G\end{pmatrix},
\begin{pmatrix}
v_F&c^\mathsf T\\
c&C_G
\end{pmatrix}
\right],
\tag{6}
$$

其中 $c=\operatorname{Cov}(G_t,F_t\mid H)$ 是三维列向量。

**保留起点条件后，通常 $c\neq0$。**因此，求零等值面上的梯度分布时，必须进一步对 $F_t=0$ 条件化：

$$
\boxed{
G_t\mid F_t=0,H
\sim\mathcal N(\bar g_t,S_t),
}
\tag{7}
$$

$$
\boxed{
\bar g_t=m_G-\frac{c}{v_F}m_F,
\qquad
S_t=C_G-\frac{cc^\mathsf T}{v_F}.
}
\tag{8}
$$

这一步同时决定消光系数和 NDF。

---

对自由程而言，只需要式（7）沿光线方向的投影：

$$
K_t\mid F_t=0,H
\sim\mathcal N(\mu_t,s_t^2),
$$

其中

$$
\boxed{
\mu_t=\omega^\mathsf T\bar g_t,
\qquad
s_t^2=\omega^\mathsf TS_t\omega.
}
\tag{9}
$$

令 $\phi,\Phi$ 分别表示标准正态 PDF、CDF。因为

$$
F_t\mid H\sim\mathcal N(m_F,v_F),
$$

所以

$$
p(F_t=0\mid H)
=
\frac{1}{\sqrt{v_F}}
\phi\left(\frac{m_F}{\sqrt{v_F}}\right),
$$

$$
P(F_t>0\mid H)
=
\Phi\left(\frac{m_F}{\sqrt{v_F}}\right).
$$

另一方面，对一维高斯 $K\sim\mathcal N(\mu,s^2)$，

$$
\begin{aligned}
\mathbb E[(-K)_+]
&=
\int_{-\infty}^0
(-k)\frac1s\phi\left(\frac{k-\mu}{s}\right)\,dk\\
&=
s\phi\left(\frac{\mu}{s}\right)
-\mu\Phi\left(-\frac{\mu}{s}\right).
\end{aligned}
$$

因此得到显式消光系数：

$$
\boxed{
\widetilde\Sigma(t\mid H)
=
\frac{
\phi\!\left(m_F(t)/\sqrt{v_F(t)}\right)
}{
\sqrt{v_F(t)}
\Phi\!\left(m_F(t)/\sqrt{v_F(t)}\right)
}
\left[
s_t\phi\left(\frac{\mu_t}{s_t}\right)
-\mu_t\Phi\left(-\frac{\mu_t}{s_t}\right)
\right].
}
\tag{10}
$$

当 $s_t=0$ 时，中括号取连续极限 $(-\mu_t)_+$。

这一步的高斯积分没有额外近似；近似仍然只来自式（1）。

---

有了消光系数，就可以沿原文式（19）—（22）推导透射率。

在该模型中，

$$
\widetilde T(t+\Delta t\mid H)
=
\widetilde T(t\mid H)
\left[1-\widetilde\Sigma(t\mid H)\Delta t
+o(\Delta t)\right].
$$

因此

$$
\frac{d\widetilde T(t\mid H)}{dt}
=
-\widetilde\Sigma(t\mid H)\widetilde T(t\mid H).
$$

由 $k_i>0$，取初值

$$
\widetilde T(0^+\mid H)=1.
$$

记

$$
\zeta_t=\frac{m_F(t)}{\sqrt{v_F(t)}},
$$

便得到完整透射率表达式：

$$
\boxed{
\widetilde T(t\mid H)
=
\exp\left[
-\int_0^t
\frac{\phi(\zeta_u)}{\sqrt{v_F(u)}\Phi(\zeta_u)}
\left\{
s_u\phi\left(\frac{\mu_u}{s_u}\right)
-\mu_u\Phi\left(-\frac{\mu_u}{s_u}\right)
\right\}
\,du
\right].
}
\tag{11}
$$

将 free-space PDF 按**自由程距离 PDF**记为 $\widetilde p_\tau$。它为

$$
\boxed{
\begin{aligned}
\widetilde p_\tau(t\mid H)
&=-\frac{d\widetilde T(t\mid H)}{dt}\\
&=
\frac{\phi(\zeta_t)}{\sqrt{v_F(t)}\Phi(\zeta_t)}
\left[
s_t\phi\left(\frac{\mu_t}{s_t}\right)
-\mu_t\Phi\left(-\frac{\mu_t}{s_t}\right)
\right]
\widetilde T(t\mid H).
\end{aligned}
}
\tag{12}
$$

**消光系数可以显式求值，但一般不能把式（11）的一维积分继续化为简单的闭式函数。**实际可数值积分，然后对累计消光

$$
A_H(t)=\int_0^t\widetilde\Sigma(u\mid H)\,du
$$

求解

$$
A_H(t)=-\log U,\qquad U\sim\mathcal U(0,1).
$$

如果在边界距离 $L$ 之前没有达到该值，则光线以概率 $\widetilde T(L\mid H)$ 穿出区域。相应地，

$$
\int_0^L\widetilde p_\tau(t\mid H)\,dt
+\widetilde T(L\mid H)=1.
$$

这里不能直接令 $\widetilde T(t)=Q_H(t)$。实际上，在当前光滑条件下，

$$
Q_H'(t)
=
p(F_t=0\mid H)\,
\mathbb E[K_t\mid F_t=0,H].
$$

这个导数包含正穿越和负穿越的**净差**；自由程需要的是首次负穿越，因此两者不是同一个量。

---

下面把条件矩对原文使用的 SE 核完全展开，便于直接实现。

设

$$
\kappa(x,y)
=
\sigma_h^2
\exp\left[-\frac12(x-y)^\mathsf TA(x-y)\right],
$$

$$
A=\operatorname{diag}(\ell_x^{-2},\ell_y^{-2},\ell_z^{-2}).
$$

定义

$$
\ell_\omega
=
(\omega^\mathsf TA\omega)^{-1/2},
\qquad
q=\frac{t}{\ell_\omega},
\qquad
R=e^{-q^2/2},
$$

以及起点观测相对于先验均值的偏移：

$$
\delta f=-m(x_i),
\qquad
\delta g=g_i-\nabla m(x_i),
\qquad
\delta k=\omega^\mathsf T\delta g.
$$

令 $d=t\omega,\;u=Ad$。由核的导数，

$$
\mathcal K_{00}
=
\mathcal K_{tt}
=
\sigma_h^2
\begin{pmatrix}
1&0\\0&A
\end{pmatrix},
$$

$$
\mathcal K_{t0}
=
\sigma_h^2R
\begin{pmatrix}
1&u^\mathsf T\\
-u&A-uu^\mathsf T
\end{pmatrix}.
\tag{13}
$$

代入式（4）—（5），得到均值：

$$
\boxed{
m_F(t)
=
m(x_t)+R(\delta f+t\delta k),
}
\tag{14}
$$

$$
\boxed{
m_G(t)
=
\nabla m(x_t)
+
R\left[
\delta g-tA\omega(\delta f+t\delta k)
\right].
}
\tag{15}
$$

以及协方差：

$$
\boxed{
v_F(t)
=
\sigma_h^2\left[1-(1+q^2)R^2\right],
}
\tag{16}
$$

$$
\boxed{
c(t)
=
\sigma_h^2R^2\,tq^2A\omega,
}
\tag{17}
$$

$$
\boxed{
C_G(t)
=
\sigma_h^2\left[
(1-R^2)A
+
R^2t^2(1-q^2)
(A\omega)(A\omega)^\mathsf T
\right].
}
\tag{18}
$$

把它们代入式（8），就得到计算 NDF 所需的完整三维参数 $\bar g_t,S_t$。

如果只计算自由程，可以仅使用五个标量矩。令

$$
m_K=\mathbb E[K_t\mid H],
\quad
v_K=\operatorname{Var}(K_t\mid H),
\quad
c_{FK}=\operatorname{Cov}(F_t,K_t\mid H).
$$

则

$$
\boxed{
m_K(t)
=
\omega^\mathsf T\nabla m(x_t)
+
R\left[
-\frac{t}{\ell_\omega^2}\delta f
+(1-q^2)\delta k
\right],
}
$$

$$
\boxed{
v_K(t)
=
\frac{\sigma_h^2}{\ell_\omega^2}
\left[1-(1-q^2+q^4)R^2\right],
}
$$

$$
\boxed{
c_{FK}(t)
=
\frac{\sigma_h^2}{\ell_\omega}q^3R^2.
}
\tag{19}
$$

然后

$$
\boxed{
\mu_t=m_K-\frac{c_{FK}}{v_F}m_F,
\qquad
s_t^2=v_K-\frac{c_{FK}^2}{v_F}.
}
\tag{20}
$$

对于原文的平面均值场 $m(x)=z$，只需代入

$$
m(x_t)=z_i+t\omega_z,\qquad
\nabla m(x_t)=e_z,
$$

$$
\delta f=-z_i,\qquad
\delta g=g_i-e_z.
$$

例如，

$$
m_F(t)
=
z_i+t\omega_z
+
R\left[-z_i+t(k_i-\omega_z)\right].
$$

当 $t\to0$ 时，这组条件协方差退化；实现中应使用极限或稳定的小量展开，不应直接在 $t=0$ 做除法。

---

接下来推导 NDF。这对应原文式（34）—（38）。

原文的关键步骤是用梯度模长做面积加权，再将梯度模长积分掉；因此出现的是 $r^3\,dr$，而不是仅有球坐标 Jacobian 的 $r^2\,dr$。[^paper]

保留起点条件后，要使用的梯度 PDF 是

$$
\boxed{
\psi_t(g)
:=
p(G_t=g\mid F_t=0,H)
=
\mathcal N_3(g;\bar g_t,S_t).
}
\tag{21}
$$

不能使用 $p(G_t\mid H)$，因为我们要描述的是零等值面上的梯度。

面积权重可以从余面积公式严格得到。对于某个 realization $f$，令

$$
S_f=\{x:f(x)=0\}.
$$

对任意法线测试函数 $a(n)$，

$$
\int_{S_f\cap V}a(n)\,dA
=
\int_V
\delta(f(x))\|\nabla f(x)\|
a\left(\frac{\nabla f(x)}{\|\nabla f(x)\|}\right)\,dx.
$$

在条件 $H$ 下取期望，位置 $x_t$ 处的面积统计包含

$$
p(F_t=0\mid H)
\int_{\mathbb R^3}
\|g\|a\left(\frac g{\|g\|}\right)\psi_t(g)\,dg.
$$

令

$$
g=rn,\qquad r>0,\quad n\in S^2,
\qquad dg=r^2\,dr\,d\omega_n,
$$

则上述积分变为

$$
p(F_t=0\mid H)
\int_{S^2}a(n)
\left[
\int_0^\infty r^3\psi_t(rn)\,dr
\right]d\omega_n.
$$

因此，沿用原文的面积加权 NDF 约定：

$$
\boxed{
D_H(n;t)
=
\int_0^\infty
r^3\mathcal N_3(rn;\bar g_t,S_t)\,dr.
}
\tag{22}
$$

它对应的真实局部面积密度为

$$
\boxed{
\mathcal A_H(n;t)
=
p(F_t=0\mid H)\,D_H(n;t).
}
\tag{23}
$$

也就是说，

$$
\mathbb E[dA(n)\mid H]
=
\mathcal A_H(n;t)\,dV\,d\omega_n.
$$

这样定义同时说明了：$D_H$ 本身不是一个积分为 1 的法线概率密度。事实上，

$$
\boxed{
\int_{S^2}D_H(n;t)\,d\omega_n
=
\mathbb E[\|G_t\|\mid F_t=0,H].
}
\tag{24}
$$

如果要得到按表面积抽样的法线 PDF，才需要用式（24）归一化。

---

式（22）的径向积分仍然可以解析计算，只是协方差现在一般不再是对角矩阵。

对给定 $n$，定义

$$
a=\frac12n^\mathsf TS_t^{-1}n,
\qquad
b=\frac12n^\mathsf TS_t^{-1}\bar g_t,
\qquad
c_\star=\frac12\bar g_t^\mathsf TS_t^{-1}\bar g_t.
$$

这里的标量 $a,b,c_\star$ 用于下面的径向积分；$c_\star$ 与式（6）中的协方差列向量 $c$ 不同。

当 $S_t$ 正定时，

$$
\mathcal N_3(rn;\bar g_t,S_t)
=
\frac{\exp(-ar^2+2br-c_\star)}
{(2\pi)^{3/2}\sqrt{\det S_t}}.
$$

所以

$$
D_H(n;t)
=
\frac{e^{-c_\star}}
{(2\pi)^{3/2}\sqrt{\det S_t}}
J_3(a,b),
$$

其中

$$
J_j(a,b)=\int_0^\infty r^j e^{-ar^2+2br}\,dr.
$$

配方得到

$$
J_0(a,b)
=
\frac{\sqrt\pi}{2\sqrt a}
e^{b^2/a}
\operatorname{erfc}\left(-\frac b{\sqrt a}\right).
\tag{25}
$$

分部积分给出

$$
J_1=\frac{1+2bJ_0}{2a},
$$

$$
J_{j+1}
=
\frac{2bJ_j+jJ_{j-1}}{2a},
\qquad j\ge1.
$$

因此

$$
J_3
=
\frac{a+b^2}{2a^3}
+
\frac{b(3a+2b^2)}{2a^3}J_0.
$$

最终得到完整的条件 NDF：

$$
\boxed{
D_H(n;t)
=
\frac{e^{-c_\star}}
{(2\pi)^{3/2}\sqrt{\det S_t}}
\left[
\frac{a+b^2}{2a^3}
+
\frac{b(3a+2b^2)}{2a^3}
\frac{\sqrt\pi}{2\sqrt a}
e^{b^2/a}
\operatorname{erfc}\left(-\frac b{\sqrt a}\right)
\right].
}
\tag{26}
$$

**这就是原文式（38）在保留起点条件后的推广。**变化集中在

$$
(0,0,1)^\mathsf T\longrightarrow\bar g_t,
\qquad
\sigma_h^2A\longrightarrow S_t.
$$

退化高斯情形，例如严格高度场极限，需要对式（22）取退化分布的极限，不能直接求奇异矩阵的逆。

---

还需要证明：这个 NDF 与前面求出的消光系数相匹配。

沿光线的条件投影面积为

$$
\begin{aligned}
\int_{S^2}(-\omega^\mathsf Tn)_+D_H(n;t)\,d\omega_n
&=
\int_{S^2}\int_0^\infty
(-\omega^\mathsf Tn)_+
r^3\psi_t(rn)\,dr\,d\omega_n\\
&=
\int_{\mathbb R^3}
(-\omega^\mathsf Tg)_+\psi_t(g)\,dg\\
&=
B_H(t).
\end{aligned}
$$

因此，

$$
\boxed{
B_H(t)
=
\int_{S^2}(-\omega^\mathsf Tn)_+D_H(n;t)\,d\omega_n
=
s_t\phi(\mu_t/s_t)
-\mu_t\Phi(-\mu_t/s_t).
}
\tag{27}
$$

消光系数于是仍保持原文的结构：

$$
\boxed{
\widetilde\Sigma(t\mid H)
=
\rho_H(t)
\int_{S^2}(-\omega^\mathsf Tn)_+D_H(n;t)\,d\omega_n.
}
\tag{28}
$$

区别在于，密度因子和 NDF 都依赖起点条件。

进一步，在这个模型中，已知距离 $t$ 发生碰撞时，梯度的采样 PDF 应为

$$
\boxed{
\widetilde p_G(g\mid\tau=t,H)
=
\frac{(-\omega^\mathsf Tg)_+\psi_t(g)}{B_H(t)}.
}
\tag{29}
$$

所以不能直接从高斯 $\psi_t$ 抽样梯度后归一化；还要考虑碰撞的投影权重。

将模长积分掉，得到对应的 VNDF，即碰撞法线 PDF：

$$
\boxed{
\widetilde p_n(n\mid\tau=t,H)
=
\frac{
(-\omega^\mathsf Tn)_+D_H(n;t)
}{
B_H(t)
}.
}
\tag{30}
$$

结合自由程 PDF，得到完整的“碰撞距离—法线”联合 PDF：

$$
\boxed{
\begin{aligned}
\widetilde p_{\tau,n}(t,n\mid H)
&=
\widetilde p_\tau(t\mid H)
\widetilde p_n(n\mid\tau=t,H)\\
&=
\widetilde T(t\mid H)\,
\rho_H(t)\,
(-\omega^\mathsf Tn)_+D_H(n;t).
\end{aligned}
}
\tag{31}
$$

对法线积分，

$$
\int_{S^2}\widetilde p_{\tau,n}(t,n\mid H)\,d\omega_n
=
\widetilde\Sigma(t\mid H)\widetilde T(t\mid H)
=
\widetilde p_\tau(t\mid H).
$$

因此，自由程和 NDF 使用的是同一套条件统计，归一化一致。

如果下一次反弹还要继续使用 $H=\{F=0,G=g\}$，则需要保留碰撞处的完整梯度；只保存归一化法线会丢失下一段条件化所需的模长信息。

---

可以检查上述公式如何退回原始 Macrofacet。

当起点影响消失，即 SE 核下 $R\to0$，并使用原文的 $m(x)=z$ 时，

$$
m_F\to z_t,\qquad v_F\to\sigma_h^2,
$$

$$
c\to0,\qquad
\bar g_t\to e_z,\qquad
S_t\to\sigma_h^2A.
$$

这里 $z_t=z_i+t\omega_z$ 表示终点的空间高度。

于是

$$
\rho_H(t)
\to
\frac{\phi(z_t/\sigma_h)}
{\sigma_h\Phi(z_t/\sigma_h)},
$$

正好恢复原文式（31）的密度因子。

同时令

$$
\alpha_j=\frac{\sqrt2\,\sigma_h}{\ell_j},
$$

则式（26）中的径向积分参数变为

$$
a=
\frac{n_x^2}{\alpha_x^2}
+\frac{n_y^2}{\alpha_y^2}
+\frac{n_z^2}{\alpha_z^2},
\qquad
b=\frac{n_z}{\alpha_z^2},
\qquad
c_\star=\frac1{\alpha_z^2},
$$

且

$$
(2\pi)^{3/2}\sqrt{\det S_t}
=
\pi^{3/2}\alpha_x\alpha_y\alpha_z.
$$

代入后恢复原文式（38）。

主要对应关系是：

| 原论文中的量 | 只去掉第二个假设后 |
| --- | --- |
| 去相关的 $p(F_t,G_t)$ | 保留 $p(F_t,G_t\mid H)$ |
| 零面梯度 $\mathcal N(e_z,\sigma_h^2A)$ | $\mathcal N(\bar g_t,S_t)$ |
| 局部密度因子 $\rho(x_t)$ | $\rho_H(t)=p(F_t=0\mid H)/P(F_t>0\mid H)$ |
| 固定统计下的 $D(n)$ | 条件 NDF $D_H(n;t)$ |
| 投影面积 $\sigma(\omega)$ | $B_H(t)$ |
| 透射率 | $\exp[-\int_0^t\rho_H(u)B_H(u)\,du]$ |

还有一个归一化性质会随之改变：

$$
\boxed{
\int_{S^2}nD_H(n;t)\,d\omega_n
=
\bar g_t.
}
\tag{32}
$$

原论文右侧是固定的 $e_z$；保留起点条件后，它应变成条件平均梯度。不能为了强行恢复原来的投影恒等式而单独重新缩放 NDF，否则会破坏式（27）与消光系数的对应关系。

---

最后明确这套推导距离真实首次碰撞还差什么。定义整段无遮挡的桥接概率

$$
\mathcal V_t(g)
=
P\!\left(
F_u>0,\ \forall\,0<u<t
\mid F_t=0,G_t=g,H
\right).
$$

真实首次碰撞的距离—梯度联合密度为

$$
\boxed{
p_{\mathrm{true}}(t,g\mid H)
=
(-\omega^\mathsf Tg)_+\,
p_H(0,g;t)\,
\mathcal V_t(g).
}
\tag{33}
$$

而本次模型给出

$$
\widetilde p(t,g\mid H)
=
(-\omega^\mathsf Tg)_+\,
p_H(0,g;t)\,
\frac{\widetilde T(t\mid H)}{Q_H(t)}.
$$

也就是说，**保留第一个近似，仍把依赖梯度的整段存活权重，替换成了一个标量存活权重。**去掉第二个假设恢复了起点与端点的高斯相关性，但没有恢复整段路径的首次通过约束。

---

以下继续推导**自由程采样算法、反射相函数和估计器权重**。这里首先沿用原文的材质设定：每个局部面片是理想镜面导体，表面法线的随机性产生宏观上的散射分布。后面再给出一般局部 BRDF 的形式。

为避免混淆射线方向与表面渲染方程的方向约定，以下使用：

- $d$：当前射线朝下一碰撞点传播的单位方向，即前文的 $\omega$。
- $v$：反射后继续追踪的单位方向。
- $n$：碰撞面朝向空侧的单位法线，因此 $d^\mathsf Tn<0$。
- $\mathcal F(c)$：导体的 Fresnel 反射率，其中 $c=-d^\mathsf Tn>0$；它与随机场值 $F_t$ 不同。

为简化书写，将式（3）、（11）、（12）记为

$$
\lambda_H(t)=\widetilde\Sigma(t\mid H)
=\rho_H(t)B_H(t),
\qquad
T_H(t)=\exp[-A_H(t)],
\qquad
p_H^{\mathrm{len}}(t)=\lambda_H(t)T_H(t),
\tag{34}
$$

其中

$$
A_H(t)=\int_0^t\lambda_H(u)\,du.
$$

以下“精确采样”均指针对这个保留第一个近似的条件模型精确采样，不意味着恢复原始 GP 的真实首次通过时间。

---

自由程采样从其累积分布开始：

$$
C_H^{\mathrm{len}}(t)
=\int_0^t p_H^{\mathrm{len}}(u)\,du
=1-T_H(t)
=1-e^{-A_H(t)}.
\tag{35}
$$

因为累计消光 $A_H(t)$ 单调不减，可以先生成一个单位指数随机变量，再反演累计消光：

$$
U\sim\mathcal U(0,1),
\qquad
E=-\log U,
\qquad
t=A_H^{-1}(E).
\tag{36}
$$

证明是

$$
P(t>s)
=P(E>A_H(s))
=e^{-A_H(s)}
=T_H(s).
$$

设当前介质段的出口或确定性边界距离为 $L$。算法必须包含“段内没有碰撞”的离散事件：

$$
\begin{cases}
E\ge A_H(L): & \text{无碰撞到达边界，概率为 }T_H(L),\\
E<A_H(L): & \text{求解 }A_H(t)=E,\quad 0<t<L.
\end{cases}
$$

不能把段内碰撞 PDF 直接重新归一化后当成完整自由程分布，否则会删除逃逸概率。

通常 $A_H$ 没有简单解析原函数，可以用数值积分配合带区间保护的 Newton 法：

$$
t_{\mathrm{new}}
=t-\frac{A_H(t)-E}{\lambda_H(t)}.
\tag{37}
$$

维护根区间 $[a,b]$：若 $A_H(t)<E$，更新 $a=t$；否则更新 $b=t$。当 Newton 步跑出区间，或 $\lambda_H(t)$ 太小而不能稳定相除时，改用二分。不要从 $\lambda_H(0)=0$ 的表面起点直接做无保护 Newton 迭代。

~~~python
def sample_length_by_inversion(H, d, L):
    E = -log(uniform_open01())
    A_L = integrate_hazard(H, d, 0, L)
    if E >= A_L:
        return Escape(L)

    a, b = 0, L
    t = 0.5 * (a + b)
    while not converged(a, b, t):
        A_t = integrate_hazard(H, d, 0, t)
        if A_t < E:
            a = t
        else:
            b = t
        rate = hazard(H, d, t)
        candidate = t - (A_t - E) / rate if rate > 0 else None
        if candidate is None or not (a < candidate < b):
            candidate = 0.5 * (a + b)
        t = candidate
    return Collision(t)
~~~

这个数学构造精确；实际的积分与求根容差会带来数值误差。若需要避免累计消光积分，可以使用后面给出的 thinning 方法。

---

若需要快速、可重复求值的数值方案，可以在距离节点上构造分段线性消光函数。

令

$$
t_j<t_{j+1},
\qquad
\Delta_j=t_{j+1}-t_j,
\qquad
b_j=\frac{\lambda_{j+1}-\lambda_j}{\Delta_j}.
$$

在第 $j$ 段使用

$$
\widehat\lambda_H(t_j+u)=\lambda_j+b_ju,
\qquad 0\le u\le\Delta_j,
$$

并累积

$$
\widehat A_{j+1}
=\widehat A_j+\frac{\lambda_j+\lambda_{j+1}}2\Delta_j.
\tag{38}
$$

抽取 $E=-\log U$。若 $E\ge\widehat A(L)$，返回逃逸；否则找到

$$
\widehat A_j\le E<\widehat A_{j+1}.
$$

令 $r=E-\widehat A_j$，则需要解

$$
\lambda_ju+\frac12b_ju^2=r.
$$

一个避免相近数相减的解为

$$
\boxed{
u=
\frac{2r}{
\lambda_j+\sqrt{\lambda_j^2+2b_jr}
},
\qquad
t=t_j+u.
}
\tag{39}
$$

零累计消光的区间不会被选中；$r=0$ 时直接返回左端点。常数消光情形自动退化为 $u=r/\lambda_j$。

这一算法精确采样的是

$$
\widehat p_H^{\mathrm{len}}(t)
=\widehat\lambda_H(t)e^{-\widehat A_H(t)},
$$

而不是未经离散化的原始 PDF。两种使用方式需要区分：

- 如果将 $\widehat\lambda_H$ 作为实际渲染模型，就在透射率与估计器中一致使用它。
- 如果只将其作为原始模型的 proposal，则要保留目标 PDF 与 proposal PDF 的比值：

$$
\boxed{
w_{\mathrm{len}}(t)
=\frac{p_H^{\mathrm{len}}(t)}
{\widehat p_H^{\mathrm{len}}(t)}
=
\frac{\lambda_H(t)}{\widehat\lambda_H(t)}
\exp[-A_H(t)+\widehat A_H(t)].
}
\tag{40}
$$

边界逃逸事件对应的权重为

$$
w_{\mathrm{esc}}
=\frac{T_H(L)}{\widehat T_H(L)}
=\exp[-A_H(L)+\widehat A_H(L)].
$$

作为 proposal 时，它必须在目标 PDF 非零的区域也非零。由此也可以看出：使用粗表采样后直接把距离权重设为 1，是在渲染粗表定义的模型。

---

第二种自由程算法是 **thinning / delta tracking**。它不要求先计算 $A_H(t)$，但需要一个有效上界

$$
M_H(t)\ge\lambda_H(t)
$$

覆盖整段目标区间。标准 null-collision 采样可用于按自由程密度抽样；这里将同一构造应用于固定起点条件 $H$ 下随距离变化的速率。[^pbrt-trans]

先用速率 $M_H$ 产生候选事件，再以概率

$$
a(t)=\frac{\lambda_H(t)}{M_H(t)}
$$

接受为真实碰撞，否则将它视为 null collision。

固定 $H$ 后，候选点是非齐次 Poisson 点过程。区间内没有被接受点的概率为

$$
\begin{aligned}
P(\text{没有真实碰撞于 }[0,t])
&=
\mathbb E_{\{s_j\}}
\left[
\prod_{s_j<t}
\left(1-\frac{\lambda_H(s_j)}{M_H(s_j)}\right)
\right]\\
&=
\exp\left[
\int_0^t
M_H(u)
\left(
1-\frac{\lambda_H(u)}{M_H(u)}-1
\right)\,du
\right]\\
&=
\exp\left[-\int_0^t\lambda_H(u)\,du\right]
=T_H(t).
\end{aligned}
\tag{41}
$$

因此，第一个被接受的事件具有所需 PDF：

$$
p_{\mathrm{accepted}}(t)=T_H(t)\lambda_H(t).
\tag{42}
$$

常数上界的伪代码为

~~~python
def sample_length_by_thinning(H, d, L, M):
    if M == 0:
        return Escape(L)
    assert M > 0
    age = 0
    while True:
        age += -log(uniform_open01()) / M
        if age >= L:
            return Escape(L)
        rate = hazard(H, d, age)
        assert rate <= M
        if uniform_open01() < rate / M:
            return Collision(age)
        # Null event: keep H and the original segment origin unchanged.
~~~

可以使用分段常数上界提高效率，但必须保证每段的上界有效。仅在若干离散位置取最大值，通常不能证明它是整个区间的上界；若没有可靠上界，使用数值累计消光反演更直接。

**null collision 不应重置起点条件，也不应把已走距离清零。**它只是采样算法的辅助事件，不是一次真实表面观测。对同一 GP 区域的数值分块边界也是如此。

若已经从原起点无碰撞走到 $t_0$，剩余距离的条件透射率应为

$$
P(\tau>t_0+u\mid\tau>t_0,H)
=
\frac{T_H(t_0+u)}{T_H(t_0)}
=
\exp\left[-\int_{t_0}^{t_0+u}\lambda_H(s)\,ds\right].
$$

它不是把当前位置当作一个新的已知碰撞点后重新计算的透射率。

---

自由程确定以后，还需要采样碰撞处的完整梯度。以下给出不依赖 NDF 数值反演的算法。

在当前距离 $t$，省略下标，记

$$
\psi(g)=\mathcal N_3(g;\bar g,S),
\qquad
\mu=d^\mathsf T\bar g,
\qquad
s^2=d^\mathsf TSd,
$$

$$
B=s\phi(\mu/s)-\mu\Phi(-\mu/s).
$$

根据前文式（29），碰撞梯度 PDF 是

$$
\boxed{
q_G(g)
=\frac{(-d^\mathsf Tg)_+\psi(g)}{B}.
}
\tag{43}
$$

令 $k=d^\mathsf Tg$。对与 $k$ 垂直的两个梯度分量积分，得到

$$
\boxed{
q_K(k)
=
\begin{cases}
\displaystyle
\frac{(-k)}{B\,s}
\phi\left(\frac{k-\mu}{s}\right),
&k<0,\\[6pt]
0,&k\ge0.
\end{cases}
}
\tag{44}
$$

它是带投影权重的一维截断高斯，不能用普通的负半轴截断高斯直接替代。

对 $k\le0$，它的 CDF 可以解析计算：

$$
\begin{aligned}
C_K(k)
&=
\frac1B\int_{-\infty}^{k}
(-u)\frac1s\phi\left(\frac{u-\mu}{s}\right)\,du\\
&=
\boxed{
\frac{
s\phi\left(\frac{k-\mu}{s}\right)
-\mu\Phi\left(\frac{k-\mu}{s}\right)
}{B}
}.
\end{aligned}
\tag{45}
$$

容易验证

$$
C_K(-\infty)=0,\qquad C_K(0)=1,
\qquad
C_K'(k)=q_K(k)\ge0.
$$

所以可以抽取 $U\sim\mathcal U(0,1)$，通过一维二分或带区间保护的 Newton 法解 $C_K(k)=U$。负方向下界可逐步向外扩张，直到包住目标概率。一般情况下不需要也没有简单的逆函数表达式。

特别地，当 $\mu=0$ 时，

$$
q_K(k)=\frac{-k}{s^2}e^{-k^2/(2s^2)},
\qquad k<0,
$$

因此有解析采样式

$$
k=-s\sqrt{-2\log U}.
\tag{46}
$$

若 $s=0$ 且 $\mu<0$，则直接取 $k=\mu$；若此时 $\mu\ge0$，则 $B=0$，该状态没有有效的负向碰撞。

得到 $k$ 后，由于投影权重仅依赖 $k$，其余梯度分量的条件分布仍是原高斯的条件分布：

$$
\boxed{
G\mid K=k,\mathrm{hit}
\sim
\mathcal N_3\left(
\bar g+\frac{Sd}{s^2}(k-\mu),
\;
S-\frac{Sdd^\mathsf TS}{s^2}
\right).
}
\tag{47}
$$

这个协方差为秩 2。可以用下面的方式采样，避免对秩 2 矩阵直接做 Cholesky 分解：

$$
\xi\sim\mathcal N_3(0,S),
$$

$$
\boxed{
g=
\bar g+\frac{Sd}{s^2}(k-\mu)
\;+\;
\xi-\frac{Sd}{s^2}(d^\mathsf T\xi).
}
\tag{48}
$$

它满足

$$
d^\mathsf Tg=k,
$$

而且均值和协方差正好对应式（47）。

~~~python
def sample_collision_gradient(mean_g, S, d):
    mu = dot(d, mean_g)
    s2 = dot(d, S @ d)
    if s2 == 0:
        assert mu < 0
        return mean_g + sample_zero_mean_gaussian(S)
    k = sample_projected_negative_gaussian(mu, sqrt(s2))
    xi = sample_zero_mean_gaussian(S)
    a = (S @ d) / s2
    g = mean_g + a * (k - mu) + xi - a * dot(d, xi)
    return g
~~~

通过 $n=g/\|g\|$ 得到的法线已经服从正确的 VNDF：

$$
q_N(n)=\frac{(-d^\mathsf Tn)_+D_H(n;t)}{B_H(t)}.
$$

因此，这种算法不需要为了采样而计算闭式 NDF 的复杂表达式；只需要条件高斯参数和一维加权高斯 CDF。极端尾部的 $B$ 与 CDF 应使用稳定的正态尾概率或对数域实现，避免相近小数相减。

---

现在推导相函数。对于理想镜面导体，局部反射关系为

$$
\boxed{
v=d-2(d^\mathsf Tn)n,
\qquad
n_h=\frac{v-d}{\|v-d\|},
\qquad
c=-d^\mathsf Tn_h>0.
}
\tag{49}
$$

除去零测度的掠射情形，这个映射将朝向入射射线的法线半球映到整个出射方向球面。其立体角 Jacobian 为

$$
d\omega_v=4c\,d\omega_n.
$$

因此，按 VNDF 采样法线并反射得到的方向 PDF 为

$$
\boxed{
q_R(v\mid d,H,t)
=\frac{q_N(n_h)}{4c}
=\frac{D_H(n_h;t)}{4B_H(t)}.
}
\tag{50}
$$

这是一个归一化的几何采样 PDF：

$$
\int_{S^2}q_R(v\mid d,H,t)\,d\omega_v=1.
$$

这个 PDF 不包含 Fresnel，也不额外乘宏法线的余弦。

原文式（7）把 Fresnel 反射率放进了被称为“phase function”的量中，并通过令散射系数等于消光系数，把反照率留在该函数内部。[^paper] 为避免与归一化 PDF 混淆，这里将这个量记为反射散射核 $\mathcal P_H$：

$$
\boxed{
\mathcal P_H(t;d,v)
=\mathcal F(c)\,q_R(v\mid d,H,t)
=
\frac{\mathcal F(c)\,D_H(n_h;t)}{4B_H(t)}.
}
\tag{51}
$$

它一般不积分为 1，而是积分为平均反射率：

$$
\bar a_H(t,d)
=\int_{S^2}\mathcal P_H(t;d,v)\,d\omega_v
=\mathbb E_{q_N}[\mathcal F(-d^\mathsf Tn)].
$$

如果需要传统的归一化相函数，可以定义

$$
\boxed{
p_{\mathrm{phase}}(v\mid d,H,t)
=\frac{\mathcal P_H(t;d,v)}{\bar a_H(t,d)},
}
\tag{52}
$$

同时使用

$$
\lambda_s(t\mid H,d)=\lambda_H(t)\bar a_H(t,d),
\qquad
\lambda_a(t\mid H,d)=\lambda_H(t)[1-\bar a_H(t,d)].
$$

两种约定给出相同的散射核：

$$
\lambda_H\mathcal P_H
=\lambda_s p_{\mathrm{phase}}.
$$

**不要同时把 Fresnel 放进 $\mathcal P_H$，又在散射系数中再乘一次同样的反射损失。**对于 RGB 或光谱 Fresnel，$\mathcal P_H$ 可以是颜色或光谱值，但采样 PDF 必须是标量；几何 PDF $q_R$ 正好满足这一要求。

在实际实现中，没有必要先数值计算 $\bar a_H$ 再采样归一化相函数。按 $q_R$ 采样，并把 Fresnel 留作权重即可。

---

保留起点梯度以后，仅有方向相函数还不足以描述下一段输运，因为下一段也需要当前梯度的模长。

将 $g=rn$ 代入式（43），有

$$
q_{r,N}(r,n)
=
\frac{(-d^\mathsf Tn)_+r^3\psi_t(rn)}{B_H(t)}.
$$

因此，在已知法线 $n$ 后，梯度模长的条件 PDF 是

$$
\boxed{
q_{r\mid N}(r\mid n,H,t)
=\frac{r^3\psi_t(rn)}{D_H(n;t)},
\qquad r>0.
}
\tag{53}
$$

经反射映射后，方向与模长的联合 PDF 为

$$
\boxed{
q_{R,r}(v,r\mid d,H,t)
=q_R(v\mid d,H,t)\,
q_{r\mid N}(r\mid n_h,H,t)
=\frac{r^3\psi_t(rn_h)}{4B_H(t)}.
}
\tag{54}
$$

由此有两种等价的状态采样方式：

1. 先用式（44）—（48）采样完整 $g$，再归一化得到 $n$ 并反射得到 $v$。
2. 先按方向 PDF 采样 $v$，计算 $n_h$，再按式（53）采样 $r$，重建 $g=rn_h$。

第一种通常更直接。第二种适合与光源方向采样等策略组合。式（53）可用其一维 CDF 数值反演，分母就是此前已解析求出的径向积分。

---

下面从输运积分推导估计器权重，而不是只凭形式判断哪些因子会抵消。

将当前保留的状态记为

$$
S_{\mathrm{path}}=(x_i,g_i,d).
$$

这里 $H$ 仍是起点场值为零、梯度为 $g_i$ 的条件。忽略体发光，令 $\mathscr L(S_{\mathrm{path}})$ 表示该条件模型下沿路径继续追踪得到的期望贡献。设 $L_b$ 是无碰撞到达确定性边界后的贡献。则

$$
\begin{aligned}
\mathscr L(S_{\mathrm{path}})
={}&T_H(L)L_b\\
&+\int_0^L
p_H^{\mathrm{len}}(t)
\int_{\mathbb R^3}
q_G(g\mid H,t)\,
\mathcal F(-d^\mathsf Tn)\,
\mathscr L(S'_{\mathrm{path}})\,dg\,dt,
\end{aligned}
\tag{55}
$$

其中

$$
n=\frac g{\|g\|},
\qquad
S'_{\mathrm{path}}=
\left(
x_i+td,\,
g,\,
d-2(d^\mathsf Tn)n
\right).
$$

这是一个带保留状态的递归积分方程；仅用位置和方向而不保存梯度，不能完整表示这里定义的后续输运。

假设采样器用 $q_{\mathrm{len}}(t)$ 产生段内碰撞，并用 $q_G^{\mathrm{prop}}(g\mid t)$ 采样完整梯度。这里 $q_{\mathrm{len}}$ 是包含碰撞分支概率的密度，其段内积分可以小于 1；余下概率是逃逸原子。

一次碰撞的 Monte Carlo 权重为

$$
\boxed{
w_{\mathrm{hit}}
=
\frac{
p_H^{\mathrm{len}}(t)\,
q_G(g\mid H,t)\,
\mathcal F(-d^\mathsf Tn)
}{
q_{\mathrm{len}}(t)\,
q_G^{\mathrm{prop}}(g\mid t)
}.
}
\tag{56}
$$

如果使用精确自由程采样和式（44）—（48）的精确梯度采样，则

$$
q_{\mathrm{len}}=p_H^{\mathrm{len}},
\qquad
q_G^{\mathrm{prop}}=q_G,
$$

所以

$$
\boxed{
w_{\mathrm{hit}}=\mathcal F(-d^\mathsf Tn).
}
\tag{57}
$$

如果当前路径吞吐量为 $\beta$，更新就是

$$
\beta\leftarrow
\beta\,\mathcal F(-d^\mathsf Tn).
$$

因此在这种匹配采样下：

- 不再额外乘一次 $T_H(t)$，因为存活概率已经包含在自由程采样中。
- 不再额外乘一次 $\lambda_H(t)$，因为碰撞密度已经包含在自由程采样中。
- 不再额外乘 NDF 或入射投影余弦，因为已经按投影加权梯度 PDF 采样。
- 不添加表面宏法线的余弦因子；局部镜面关系与方向 Jacobian 已在相函数推导中处理。

这是标准 importance sampling 的分子、分母相消。体路径追踪中相函数值除以方向采样 PDF 的更新结构，也使用同一原则。[^pbrt-integrator]

若采样器以离散概率 $Q_{\mathrm{esc}}$ 产生逃逸事件，则逃逸权重是

$$
\boxed{
w_{\mathrm{esc}}
=\frac{T_H(L)}{Q_{\mathrm{esc}}}.
}
\tag{58}
$$

使用精确自由程采样时，$Q_{\mathrm{esc}}=T_H(L)$，因此逃逸权重为 1。若主动、确定性地追踪一条 shadow ray，则情况不同：没有通过随机逃逸事件表达存活概率，需要显式计算或估计其透射率。

---

如果使用近似法线 proposal $q_N^{\mathrm{prop}}(n)$，并继续按正确的式（53）采样模长，则径向条件密度相消，权重成为

$$
\boxed{
w_{\mathrm{hit}}
=
\frac{p_H^{\mathrm{len}}(t)}{q_{\mathrm{len}}(t)}
\,
\frac{
(-d^\mathsf Tn)_+D_H(n;t)
}{
B_H(t)\,q_N^{\mathrm{prop}}(n)
}
\,
\mathcal F(-d^\mathsf Tn).
}
\tag{59}
$$

因此，使用原论文中的近似 VNDF proposal，或者使用均匀法线半球 proposal 时，不能直接把权重设为 Fresnel。

若直接使用方向 proposal $q_V^{\mathrm{prop}}(v)$，同时仍按正确的式（53）采样模长，则

$$
w_{\mathrm{hit}}
=
\frac{p_H^{\mathrm{len}}(t)}{q_{\mathrm{len}}(t)}
\,
\frac{\mathcal P_H(t;d,v)}{q_V^{\mathrm{prop}}(v)}.
$$

如果径向模长也采用不同的 proposal $q_r^{\mathrm{prop}}$，还必须乘

$$
\frac{q_{r\mid N}(r\mid n,H,t)}
{q_r^{\mathrm{prop}}(r\mid n,H,t)}.
$$

对只在段内强制生成一次碰撞的算法，有

$$
q_{\mathrm{len}}(t\mid t<L)
=\frac{p_H^{\mathrm{len}}(t)}{1-T_H(L)}.
$$

如果边界项另行计算，则碰撞分支必须携带

$$
\boxed{
w_{\mathrm{forced}}
=[1-T_H(L)]\,\mathcal F(-d^\mathsf Tn)
}
\tag{60}
$$

的权重，不能忽略强制碰撞造成的条件化。

常见策略的权重可以对照如下。表中默认距离按目标自由程分布采样，模长按正确条件 PDF 采样。

| 策略 | 方向或法线采样 | 延续权重 |
| --- | --- | --- |
| 几何散射，Fresnel 留作权重 | $q_G$、对应的 VNDF 或 $q_R$ | $\mathcal F$ |
| 近似法线 proposal | $q_N^{\mathrm{prop}}$ | $\mathcal F\,q_N/q_N^{\mathrm{prop}}$ |
| 近似方向 proposal | $q_V^{\mathrm{prop}}$ | $\mathcal P_H/q_V^{\mathrm{prop}}$ |
| 总是延续，并采归一化物理相函数 | $p_{\mathrm{phase}}$ | $\bar a_H$ |
| 标量 Fresnel 的模拟吸收 | 先采 $q_G$，再以概率 $\mathcal F$ 保留 | 保留后权重 1，吸收则终止 |
| 另加路径 Russian roulette | 在原策略基础上以概率 $p_{\mathrm{rr}}$ 保留 | 保留后额外除以 $p_{\mathrm{rr}}$ |

归一化物理相函数那一行按单波长或标量讨论；彩色材质需要明确选用的标量 proposal。对标量 Fresnel 做概率吸收之后，不能又额外乘一次相同的 Fresnel。

---

把这些步骤组合起来，就得到从已知表面碰撞出发的镜面导体随机游走：

~~~python
def trace_correlated_macrofacet(x, g, d, beta):
    while True:
        H = SurfaceCondition(position=x, value=0, gradient=g)
        L = distance_to_boundary(x, d)

        event = sample_exact_model_length(H, d, L)
        if event.is_escape:
            return beta * boundary_radiance(x + L * d, d)

        t = event.distance
        x_next = x + t * d
        mean_g, S = gradient_conditioned_on_endpoint_zero(H, d, t)
        g_next = sample_collision_gradient(mean_g, S, d)
        n = normalize(g_next)
        cosine = -dot(d, n)

        beta *= fresnel_conductor(cosine)
        d_next = d - 2 * dot(d, n) * n

        x, g, d = x_next, g_next, d_next

        if should_apply_roulette(beta):
            p_survive = choose_survival_probability(beta)
            if uniform_open01() >= p_survive:
                return 0
            beta /= p_survive
~~~

这里的 sample_exact_model_length 可以是满足数值精度的累计消光反演，也可以是使用有效上界的 thinning。若实际使用近似 proposal，则需要按式（40）、（56）或（59）加入相应权重。

新的起点满足

$$
v^\mathsf Tg
=-d^\mathsf Tg
>0,
$$

因此镜面反射后确实进入新面的空侧，可以继续使用前文的表面起点公式。

将新起点条件设为 $H'=\{f(x_{\mathrm{next}})=0,\nabla f(x_{\mathrm{next}})=g_{\mathrm{next}}\}$，并只保留这次碰撞信息，是该有限记忆模型的状态更新。若要保留更早的碰撞条件，应使用全部观测构造 GP 条件矩，采样与权重结构不变，但状态和计算量会增大。

对于相机发出的首段，不能凭空设置“起点位于零等值面”。一种明确的初始化选择是：先从相机位置处满足 $F_0>0$ 的分布中采样场值和梯度，再对这些起点观测使用相同算法。对于本文的平稳 SE 核，起点场值和梯度在先验下独立，因此可采样

$$
F_0\sim\mathcal N(m(x_0),\sigma_h^2)\ \text{截断到 }(0,\infty),
\qquad
G_0\sim\mathcal N_3(\nabla m(x_0),\sigma_h^2A).
$$

在式（14）—（20）中将 $\delta f=-m(x_i)$ 改成 $\delta f=F_0-m(x_0)$ 即可；由于起点已有正的场值，不再要求起始方向导数为正。这是一个明确指定的源端初始化模型，不应与表面起点的自由程分布混为一谈。

---

如果局部面片使用一般 BRDF，而非理想镜面，几何碰撞 PDF 仍是同一个 $q_G$。令局部 BRDF 为 $f_s(-d,v;n)$，则条件散射核为

$$
\boxed{
\mathcal P_H(t;d,v)
=
\int_{\mathbb R^3}
q_G(g\mid H,t)\,
f_s(-d,v;n)\,
(n^\mathsf Tv)_+\,
dg,
\qquad n=\frac g{\|g\|}.
}
\tag{61}
$$

理想镜面时，局部 BRDF 与余弦的乘积是带 Fresnel 的方向 delta 分布；执行该积分就恢复式（51）。

对于一般局部 BRDF，可先采样完整梯度 $g\sim q_G$，再从方向 proposal $q_{\mathrm{local}}(v\mid n,-d)$ 采样，权重为

$$
\boxed{
w_{\mathrm{hit}}
=
\frac{p_H^{\mathrm{len}}(t)}{q_{\mathrm{len}}(t)}
\,
\frac{
f_s(-d,v;n)(n^\mathsf Tv)_+
}{
q_{\mathrm{local}}(v\mid n,-d)
}.
}
\tag{62}
$$

这里假定完整梯度已按目标 $q_G$ 采样；否则还需要式（56）中的梯度密度比。

例如，若局部面片是反照率为 $a_d$ 的 Lambertian 表面，

$$
f_s=\frac{a_d}{\pi},
\qquad
q_{\mathrm{local}}=\frac{(n^\mathsf Tv)_+}{\pi},
$$

则在精确自由程采样下，延续权重就是 $a_d$。

若混合多个散射分支，分支选择概率也属于实际 proposal，必须计入分母。理想镜面的 delta 分支和连续的漫反射分支具有不同的测度，不能把二者都当作普通立体角密度直接相加。

---

若还需要直接光估计，保留完整梯度会产生一个额外依赖：反射后的 shadow-ray 透射率依赖当前梯度模长。

以下考虑按立体角定义 PDF 的非 delta 光源采样。给定刚采到的距离 $t$，在当前梯度尚未固定时，采样光源方向 $v$，其 PDF 为 $q_L(v)$，包括光源选择概率。对于理想镜面，令

$$
n_h=\frac{v-d}{\|v-d\|}.
$$

再采样

$$
r\sim q_{r\mid N}(r\mid n_h,H,t),
\qquad
g=rn_h.
$$

令 $H'_r$ 是这个新碰撞点的场值与完整梯度条件，$d_L$ 是到光源的距离。直接光贡献的目标积分为

$$
\begin{aligned}
\mathcal L_{\mathrm{dir}}(t)
=\int_{S^2}
&\mathcal P_H(t;d,v)\,
\mathbb E_{r\mid n_h}
\left[
T_{H'_r}(d_L;v)\,L_e(v)
\right]
\,d\omega_v.
\end{aligned}
\tag{63}
$$

因此，若距离已按目标自由程 PDF 采样，一次直接光估计为

$$
\boxed{
\widehat{\mathcal L}_{\mathrm{dir}}
=
\beta\,
\frac{\mathcal F(c)\,D_H(n_h;t)}
{4B_H(t)\,q_L(v)}
\,
\widehat T_{H'_r}(d_L;v)\,
L_e(v).
}
\tag{64}
$$

这里 $\beta$ 是当前反射之前的路径吞吐量；如果它已经乘过本次 Fresnel，就不能在同一贡献中再乘。$\widehat T$ 可以是显式计算的透射率，也可以是在固定 $H'_r$ 下构造的透射率估计器。若场景还含显式遮挡物，应额外检验该连接的几何可见性。

式（64）的径向条件 PDF 相消，但**不能因此不采样模长**：模长仍通过 $H'_r$ 影响 shadow-ray 透射率。

如果先前已经固定了完整梯度 $g$，局部理想镜面只能反射到唯一方向。这时不能再保持该 $g$ 不变，同时用已经边缘化法线的 $\mathcal P_H$ 对任意光源方向做 NEE。需要在当前梯度尚未固定的条件下使用式（63）—（64），或另行构造与潜在梯度状态一致的连接策略。

同样，组合不同策略做 MIS 时，必须比较同一扩展路径空间上的实际 proposal；除了方向，还要一致处理潜在梯度、碰撞或逃逸分支，以及 shadow-ray 的采样方式。仅把一个边缘相函数 PDF 代入原有代码，并不能自动保证这些状态一致。

---

上述采样和权重保证了它们对式（55）定义的条件模型一致。它们并不消除前文式（1）的存活条件近似，也不自动证明整个有限记忆随机游走与真实 GP 的多次反射路径分布相同。

特别地，保留起点条件后，不同路径通常具有不同的条件 NDF 和自由程分布。因此，不能仅凭单步方向分布归一化，就宣称全路径互易性已经成立；还需要检验反向路径的条件状态和联合密度。

---

[^paper]: Minghao Huang, Yuang Cui, Beibei Wang, Lingqi Yan. *Macrofacet Theory for Gaussian Process Statistical Surfaces*. 原文公式编号以本次上传的 macrofacet(4).pdf 为准。公开论文页面：[arXiv:2603.00280](https://arxiv.org/abs/2603.00280)；公开版本的公式编号可能不同。

[^notes]: 本次上传的研究笔记 *32c67f27-16c2-41d7-85cc-3743fa1a284b_Macrofacet.pdf*。其中将近似区分为“用单点空侧条件代替整段存活条件”和“终点与起点去相关”两步。本文只去掉第二步。

[^pbrt-trans]: Matt Pharr, Wenzel Jakob, Greg Humphreys. *Physically Based Rendering: From Theory to Implementation*, 4th ed., [Transmittance](https://pbr-book.org/4ed/Volume_Scattering/Transmittance). 这里引用其 null-collision 采样原理；对固定起点条件下的距离依赖速率所作的推导见本文式（41）—（42）。

[^pbrt-integrator]: Matt Pharr, Wenzel Jakob, Greg Humphreys. *Physically Based Rendering: From Theory to Implementation*, 4th ed., [Volume Scattering Integrators](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators). 可对照体路径追踪中相函数值与采样 PDF 的比值更新。

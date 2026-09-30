# 第一份推导核验，以及统一符号后的消光系数、NDF、VNDF 与 phase function

## 1. 结论先行

第一份 Markdown 中，从 SE kernel 的导数开始，到

$$
G_t\mid F_t=0,\mathcal H
$$

的条件均值与条件协方差为止，矩阵代数是正确的。核的一阶、二阶导数符号正确；高斯条件化中的 block covariance、Schur complement、沿光线方向的投影，以及再次条件于 $F_t=0$ 的公式也彼此一致。

但这个结论有一个重要边界：第一式中的

$$
\frac{\int(-\omega_o^\top g)_+p_{\mathcal H}(0,g;t)\,dg}
{P(F_t>0\mid\mathcal H)}
$$

是“以终点 $F_t>0$ 代替整段都未发生碰撞”这一剩余近似下的局部消光率。它不是一般平稳高斯过程的精确首次穿越 hazard。若目标是精确首次碰撞，还必须加入整段路径的存活条件。

第一份文件的主要问题是符号没有完全统一，而不是推导公式本身错误。后文先统一符号，再在不引入第二份文件中那些非必要简写的前提下推导 NDF、VNDF 和 phase function。

## 2. 统一符号

沿用第一份 Markdown 和 Macrofacet 论文的符号：

$$
\mathbf x_t=\mathbf x_i+t\omega_o,
\qquad \|\omega_o\|=1,
\qquad t>0,
$$

$$
F_0=f(\mathbf x_i),\qquad
G_0=\nabla f(\mathbf x_i),
$$

$$
F_t=f(\mathbf x_t),\qquad
G_t=\nabla f(\mathbf x_t),
$$

$$
\boxed{
\mathcal H=\{F_0=0,\;G_0=\mathbf g_i\}.
}
$$

约定 $f>0$ 为空侧，$G_t/\|G_t\|$ 指向空侧。若光线从起点表面进入空侧，还应有

$$
\omega_o^\top\mathbf g_i>0.
$$

后文使用论文中的必要符号：

- $\omega_m\in\Omega=S^2$：单位 macrofacet normal；
- $\omega_i$：镜面反射后的方向；
- $D$：NDF；
- $D_{\omega_o}$：相对于方向 $\omega_o$ 的 VNDF；
- $p(\omega_o,\omega_i)$：论文约定的 phase function；
- $\langle a,b\rangle=(a^\top b)_+$，其中 $(x)_+=\max(x,0)$。

梯度模长必须使用另一个变量 $r>0$，因为 $t$ 已经表示射线距离：

$$
\mathbf g=r\omega_m.
$$

这里 $\mathbf g$ 只是积分变量，随机梯度仍写成大写 $G_t$。

## 3. 第一份推导的逐项核验

设

$$
\kappa(\mathbf x,\mathbf y)
=\sigma_h^2
\exp\!\left[-\frac12(\mathbf x-\mathbf y)^\top
A(\mathbf x-\mathbf y)\right],
\qquad
A=\operatorname{diag}(\ell_x^{-2},\ell_y^{-2},\ell_z^{-2}).
$$

### 3.1 Kernel 导数

令 $\mathbf r=\mathbf x-\mathbf y$。由于 $A=A^\top$，

$$
\nabla_{\mathbf x}\kappa=-A\mathbf r\,\kappa,
\qquad
\nabla_{\mathbf y}\kappa=A\mathbf r\,\kappa,
$$

$$
\nabla_{\mathbf x}\nabla_{\mathbf y}^{\top}\kappa
=\kappa\left[A-A\mathbf r\mathbf r^\top A\right].
$$

这三式均正确。

代入 $\mathbf x_t-\mathbf x_i=t\omega_o$ 后，交叉协方差应统一写成

$$
\operatorname{Cov}(F_t,G_0)
=\sigma_h^2e^{-\frac12t^2\omega_o^\top A\omega_o}
t\omega_o^\top A,
$$

$$
\operatorname{Cov}(G_t,F_0)
=-\sigma_h^2e^{-\frac12t^2\omega_o^\top A\omega_o}
tA\omega_o,
$$

$$
\operatorname{Cov}(G_t,G_0)
=\sigma_h^2e^{-\frac12t^2\omega_o^\top A\omega_o}
\left[A-t^2(A\omega_o)(A\omega_o)^\top\right].
$$

两个 value-gradient covariance 的符号相反是正确的，原因是分别对 kernel 的第二个和第一个位置变量求导。

### 3.2 条件均值

由 $\mathcal H=\{F_0=0,G_0=\mathbf g_i\}$，观测残差为

$$
\begin{pmatrix}
-m(\mathbf x_i)\\
\mathbf g_i-\nabla m(\mathbf x_i)
\end{pmatrix}.
$$

第一份文件得到的两个条件均值是正确的。统一 $\omega_o$ 后为

$$
\boxed{
\begin{aligned}
\mathbb E[F_t\mid\mathcal H]
={}&m(\mathbf x_t)
+e^{-\frac12t^2\omega_o^\top A\omega_o}
\Big[-m(\mathbf x_i)
+t\omega_o^\top
(\mathbf g_i-\nabla m(\mathbf x_i))\Big].
\end{aligned}
}
$$

$$
\boxed{
\begin{aligned}
\mathbb E[G_t\mid\mathcal H]
={}&\nabla m(\mathbf x_t)
+e^{-\frac12t^2\omega_o^\top A\omega_o}
\Big[
\mathbf g_i-\nabla m(\mathbf x_i)
+tA\omega_o m(\mathbf x_i)\\
&\hspace{39mm}
-t^2A\omega_o\omega_o^\top
(\mathbf g_i-\nabla m(\mathbf x_i))
\Big].
\end{aligned}
}
$$

### 3.3 条件协方差

第一份文件的 Schur complement 结果也正确：

$$
\boxed{
\operatorname{Var}(F_t\mid\mathcal H)
=\sigma_h^2
\left\{
1-\left[1+t^2\omega_o^\top A\omega_o\right]
e^{-t^2\omega_o^\top A\omega_o}
\right\}.
}
$$

$$
\boxed{
\operatorname{Cov}(G_t,F_t\mid\mathcal H)
=\sigma_h^2t^3(\omega_o^\top A\omega_o)
e^{-t^2\omega_o^\top A\omega_o}A\omega_o.
}
$$

$$
\boxed{
\begin{aligned}
\operatorname{Var}(G_t\mid\mathcal H)
=\sigma_h^2\Big[&
(1-e^{-t^2\omega_o^\top A\omega_o})A\\
&+t^2e^{-t^2\omega_o^\top A\omega_o}
(1-t^2\omega_o^\top A\omega_o)
(A\omega_o)(A\omega_o)^\top
\Big].
\end{aligned}
}
$$

把这些 block 直接代回完整条件协方差的 Schur complement，可以逐项恢复第一份文件中的结果。

### 3.4 再条件于 $F_t=0$

这一部分同样正确。无需为它另起 $\bar g_t,S_t$ 等名字，直接保留第一份文件的符号即可：

$$
\boxed{
\begin{aligned}
\mathbb E[G_t\mid F_t=0,\mathcal H]
={}&\mathbb E[G_t\mid\mathcal H]\\
&-\frac{\operatorname{Cov}(G_t,F_t\mid\mathcal H)}
{\operatorname{Var}(F_t\mid\mathcal H)}
\mathbb E[F_t\mid\mathcal H].
\end{aligned}
}
\tag{1}
$$

$$
\boxed{
\begin{aligned}
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)
={}&\operatorname{Var}(G_t\mid\mathcal H)\\
&-\frac{
\operatorname{Cov}(G_t,F_t\mid\mathcal H)
\operatorname{Cov}(F_t,G_t\mid\mathcal H)
}{\operatorname{Var}(F_t\mid\mathcal H)}.
\end{aligned}
}
\tag{2}
$$

式（1）—（2）与第一份文件最后展开的三维公式完全等价；把上一节的四个量直接代入即可。

### 3.5 需要修正或补充说明的符号

| 第一份文件当前写法 | 问题 | 建议写法 |
| --- | --- | --- |
| $\omega_o$ 与 $\boldsymbol\omega$ 混用 | 同一条射线出现两个方向符号 | 全部统一成论文的 $\omega_o$ |
| $H$ 与 $\mathcal H$ 混用 | 条件事件名称不统一 | 全部使用 $\mathcal H$ |
| $F_i$ 与 $F_0$ 混用 | 下标含义不明确 | 射线参数起点统一用 $F_0$ |
| $\mathbf x_i$ 与下标 $0$ | 本身不矛盾，但需先定义 | 明确 $F_0=f(\mathbf x_i)$、$G_0=\nabla f(\mathbf x_i)$ |
| $p_{\mathcal H}(0,g;t)$ | 没有定义是联合条件密度 | 写明它是 $p_{F_t,G_t\mid\mathcal H}(0,\mathbf g;t)$ |
| $p(F_t=0\mid\mathcal H)$ | 连续变量在一点的概率为零 | 在密度因子中写 $p_{F_t\mid\mathcal H}(0)$ |
| $B_{\mathcal H}(t)=\mathbb E[(-\omega_o^\top g)_+\mid\cdots]$ | 期望式中误用了积分变量 | 写 $\mathbb E[(-\omega_o^\top G_t)_+\mid\cdots]$ |
| $\operatorname{Cov}(G_t\mid\mathcal H)$ | 对单个向量更常写 variance | 使用 $\operatorname{Var}(G_t\mid\mathcal H)$ |
| $F_t$ 与 Fresnel $F$ | 后续 phase function 中容易混淆 | 明确带下标的 $F_t$ 是场值；双参数 $F(-\omega_o,\omega_m)$ 是 Fresnel |

另外，第一份文件中的分母

$$
1-\left(1+t^2\omega_o^\top A\omega_o\right)
e^{-t^2\omega_o^\top A\omega_o}
$$

在 $t\to0$ 时是四阶小量。公式在 $t>0$ 时成立，但直接浮点计算会产生严重消减误差。严格的 $t=0$ 处，$F_t=0$ 与 $F_0=0$ 是同一个约束；而 $t\to0^+$ 时再要求一个相邻零点是不同的条件问题，因此不能把二者不加区分地直接代入。

## 4. 零等值面上的条件梯度密度

由第一份文件的最终结果，

$$
G_t\mid F_t=0,\mathcal H
$$

是三维高斯随机向量，其均值和协方差就是式（1）—（2）。因此其密度可直接写成

$$
\begin{aligned}
&p_{G_t\mid F_t=0,\mathcal H}(\mathbf g)\\
&=\frac{
\exp\!\left[
-\frac12
\bigl(\mathbf g-\mathbb E[G_t\mid F_t=0,\mathcal H]\bigr)^\top
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)^{-1}
\bigl(\mathbf g-\mathbb E[G_t\mid F_t=0,\mathcal H]\bigr)
\right]
}{
(2\pi)^{3/2}
\sqrt{\det\operatorname{Var}(G_t\mid F_t=0,\mathcal H)}
}.
\end{aligned}
\tag{3}
$$

这就是 NDF、VNDF 和 phase function 共同使用的唯一梯度密度。这里不能换成 $p(G_t\mid\mathcal H)$，因为 NDF 描述的是零等值面上的法线，必须额外条件于 $F_t=0$。

## 5. 消光系数 $\Sigma_1(t\mid\mathcal H)$ 的完整推导

这一节只使用前文已经固定的

$$
\mathbf x_t=\mathbf x_i+t\omega_o,
\qquad
F_t=f(\mathbf x_t),
\qquad
G_t=\nabla f(\mathbf x_t),
\qquad
\mathcal H=\{F_0=0,G_0=\mathbf g_i\}.
$$

不再引入第二份文件中的方向或条件矩简写。

### 5.1 从小区间负向穿越概率开始

在第一份文件保留的近似下，用“当前端点满足 $F_t>0$”代替“从起点到 $t$ 的整段均未碰撞”。对应的局部消光系数定义为

$$
\Sigma_1(t\mid\mathcal H)
=\lim_{\Delta t\to0^+}
\frac{1}{\Delta t}
P\!\left(
F_{t+\Delta t}<0
\mid F_t>0,\mathcal H
\right).
$$

沿光线对随机场作一阶 Taylor 展开：

$$
F_{t+\Delta t}
=F_t+\omega_o^\top G_t\,\Delta t+o(\Delta t).
$$

若光线在这段小区间内从空侧 $F_t>0$ 穿到负侧，则一阶条件为

$$
\omega_o^\top G_t<0,
\qquad
0<F_t< -\omega_o^\top G_t\,\Delta t.
$$

因此，以

$$
p_{F_t,G_t\mid\mathcal H}(f,\mathbf g;t)
$$

表示给定 $\mathcal H$ 后，位置 $\mathbf x_t$ 处场值与梯度的联合密度，有

$$
\begin{aligned}
&P\!\left(
F_{t+\Delta t}<0,F_t>0
\mid\mathcal H
\right)\\
&=\int_{\mathbb R^3}
\int_0^{(-\omega_o^\top\mathbf g)_+\Delta t}
p_{F_t,G_t\mid\mathcal H}(f,\mathbf g;t)
\,df\,d\mathbf g
+o(\Delta t).
\end{aligned}
$$

对固定的 $\mathbf g$，当 $\Delta t\to0^+$ 时，

$$
\int_0^{(-\omega_o^\top\mathbf g)_+\Delta t}
p_{F_t,G_t\mid\mathcal H}(f,\mathbf g;t)\,df
=(-\omega_o^\top\mathbf g)_+\Delta t\,
p_{F_t,G_t\mid\mathcal H}(0,\mathbf g;t)
+o(\Delta t).
$$

除以条件事件 $F_t>0$ 的概率和 $\Delta t$，再令 $\Delta t\to0^+$，得到

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
=\frac{
\displaystyle
\int_{\mathbb R^3}
(-\omega_o^\top\mathbf g)_+
p_{F_t,G_t\mid\mathcal H}(0,\mathbf g;t)
\,d\mathbf g
}{
P(F_t>0\mid\mathcal H)
}.
}
$$

这就是第一份 Markdown 开头公式的完整来源。

### 5.2 分解成密度因子与条件投影面积

联合条件密度可以分解为

$$
p_{F_t,G_t\mid\mathcal H}(0,\mathbf g;t)
=p_{F_t\mid\mathcal H}(0)
p_{G_t\mid F_t=0,\mathcal H}(\mathbf g).
$$

注意，$p_{F_t\mid\mathcal H}(0)$ 是一维连续随机变量在零点的密度值，不是事件 $F_t=0$ 的概率。

代入上式后，

$$
\begin{aligned}
\Sigma_1(t\mid\mathcal H)
={}&
\frac{p_{F_t\mid\mathcal H}(0)}
{P(F_t>0\mid\mathcal H)}\\
&\times
\int_{\mathbb R^3}
(-\omega_o^\top\mathbf g)_+
p_{G_t\mid F_t=0,\mathcal H}(\mathbf g)
\,d\mathbf g.
\end{aligned}
$$

因此，严格沿用第一份文件已经使用的两个量，

$$
\boxed{
\rho_{\mathcal H}(t)
=\frac{p_{F_t\mid\mathcal H}(0)}
{P(F_t>0\mid\mathcal H)},
}
$$

$$
\boxed{
B_{\mathcal H}(t)
=\mathbb E\!\left[
(-\omega_o^\top G_t)_+
\mid F_t=0,\mathcal H
\right],
}
$$

最终得到

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
=\rho_{\mathcal H}(t)B_{\mathcal H}(t).
}
\tag{\Sigma}
$$

### 5.3 用第一份文件已经推导出的高斯矩显式计算

沿用论文的记号，令

$$
\phi(x;\mu,v)
=\frac{1}{\sqrt{2\pi v}}
\exp\!\left[-\frac{(x-\mu)^2}{2v}\right]
$$

为均值 $\mu$、方差 $v$ 的一维高斯密度，$\Phi(x;\mu,v)$ 为对应 CDF。

由于

$$
F_t\mid\mathcal H
\sim\mathcal N\!\left(
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right),
$$

所以

$$
p_{F_t\mid\mathcal H}(0)
=\phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right),
$$

$$
P(F_t>0\mid\mathcal H)
=1-\Phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right).
$$

于是

$$
\boxed{
\rho_{\mathcal H}(t)
=\frac{
\phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right)
}{
1-\Phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right)
}.
}
$$

另一方面，

$$
\omega_o^\top G_t\mid F_t=0,\mathcal H
$$

是一维高斯变量，其均值与方差直接由式（1）—（2）投影得到：

$$
\omega_o^\top
\mathbb E[G_t\mid F_t=0,\mathcal H],
$$

$$
\omega_o^\top
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)
\omega_o.
$$

对任意 $X\sim\mathcal N(\mu,v)$，

$$
\begin{aligned}
\mathbb E[(-X)_+]
&=\int_{-\infty}^{0}(-x)\phi(x;\mu,v)\,dx\\
&=v\phi(0;\mu,v)-\mu\Phi(0;\mu,v),
\end{aligned}
$$

其中第二行使用了

$$
x\phi(x;\mu,v)
=\mu\phi(x;\mu,v)
-v\frac{\partial}{\partial x}\phi(x;\mu,v).
$$

直接代入而不再给均值和方差另起符号，得到

$$
\boxed{
\begin{aligned}
B_{\mathcal H}(t)
={}&
\Big[
\omega_o^\top
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)
\omega_o
\Big]\\
&\quad\times
\phi\!\Big(
0;
\omega_o^\top
\mathbb E[G_t\mid F_t=0,\mathcal H],
\omega_o^\top
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)
\omega_o
\Big)\\
&-
\omega_o^\top
\mathbb E[G_t\mid F_t=0,\mathcal H]\\
&\quad\times
\Phi\!\Big(
0;
\omega_o^\top
\mathbb E[G_t\mid F_t=0,\mathcal H],
\omega_o^\top
\operatorname{Var}(G_t\mid F_t=0,\mathcal H)
\omega_o
\Big).
\end{aligned}
}
$$

当投影方差为零时，上式按连续极限取

$$
\left(
-\omega_o^\top
\mathbb E[G_t\mid F_t=0,\mathcal H]
\right)_+.
$$

把第 3 节和式（1）—（2）给出的条件矩代入 $\rho_{\mathcal H}(t)$ 与 $B_{\mathcal H}(t)$，就得到完全由

$$
t,\ \omega_o,\ A,\ \sigma_h,\ m,
\ \mathbf x_i,\ \mathbf g_i
$$

决定的消光系数；不需要引入额外条件矩简写。

### 5.4 消光系数的完整闭式表达式

把第一份文件已经推导出的各个标量矩直接列出。首先，

$$
\boxed{
\begin{aligned}
\mathbb E[F_t\mid\mathcal H]
={}&m(\mathbf x_t)
+e^{-\frac12t^2\omega_o^\top A\omega_o}
\Big[
-m(\mathbf x_i)
+t\omega_o^\top
(\mathbf g_i-\nabla m(\mathbf x_i))
\Big].
\end{aligned}
}
$$

$$
\boxed{
\operatorname{Var}(F_t\mid\mathcal H)
=\sigma_h^2
\left\{
1-\left[1+t^2\omega_o^\top A\omega_o\right]
e^{-t^2\omega_o^\top A\omega_o}
\right\}.
}
$$

其次，把第一份文件中“directional derivative 的条件均值”和“再条件于 $F_t=0$”两步合并，得到

$$
\boxed{
\begin{aligned}
&\mathbb E[
\omega_o^\top G_t
\mid F_t=0,\mathcal H]
\\
={}&
\omega_o^\top\nabla m(\mathbf x_t)
\\
&+e^{-\frac12t^2\omega_o^\top A\omega_o}
\Bigg[
t(\omega_o^\top A\omega_o)m(\mathbf x_i)
\\
&\hspace{28mm}
+\left(1-t^2\omega_o^\top A\omega_o\right)
\omega_o^\top
\left(\mathbf g_i-\nabla m(\mathbf x_i)\right)
\Bigg]
\\
&-
\frac{
t^3(\omega_o^\top A\omega_o)^2
e^{-t^2\omega_o^\top A\omega_o}
}{
1-\left(1+t^2\omega_o^\top A\omega_o\right)
e^{-t^2\omega_o^\top A\omega_o}
}
\\
&\qquad\times
\Bigg\{
m(\mathbf x_t)
+e^{-\frac12t^2\omega_o^\top A\omega_o}
\left[
-m(\mathbf x_i)
+t\omega_o^\top
\left(\mathbf g_i-\nabla m(\mathbf x_i)\right)
\right]
\Bigg\}.
\end{aligned}
}
\tag{4}
$$

对应的条件方差为

$$
\boxed{
\begin{aligned}
&\operatorname{Var}(
\omega_o^\top G_t
\mid F_t=0,\mathcal H)
\\
={}&
\sigma_h^2(\omega_o^\top A\omega_o)
\Bigg\{
1-
\Big[
1-t^2\omega_o^\top A\omega_o
+t^4(\omega_o^\top A\omega_o)^2
\Big]
e^{-t^2\omega_o^\top A\omega_o}
\Bigg\}
\\
&-
\frac{
\sigma_h^2t^6(\omega_o^\top A\omega_o)^4
e^{-2t^2\omega_o^\top A\omega_o}
}{
1-\left(1+t^2\omega_o^\top A\omega_o\right)
e^{-t^2\omega_o^\top A\omega_o}
}.
\end{aligned}
}
\tag{5}
$$

于是，不含任何积分的完整消光系数为

$$
\boxed{
\begin{aligned}
\Sigma_1(t\mid\mathcal H)
={}&
\frac{
\phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right)
}{
1-\Phi\!\left(
0;
\mathbb E[F_t\mid\mathcal H],
\operatorname{Var}(F_t\mid\mathcal H)
\right)
}
\\
&\times
\Bigg\{
\operatorname{Var}(
\omega_o^\top G_t
\mid F_t=0,\mathcal H)
\\
&\qquad\times
\phi\!\Big(
0;
\mathbb E[\omega_o^\top G_t\mid F_t=0,\mathcal H],
\operatorname{Var}(\omega_o^\top G_t\mid F_t=0,\mathcal H)
\Big)
\\
&\qquad-
\mathbb E[\omega_o^\top G_t\mid F_t=0,\mathcal H]
\\
&\qquad\times
\Phi\!\Big(
0;
\mathbb E[\omega_o^\top G_t\mid F_t=0,\mathcal H],
\operatorname{Var}(\omega_o^\top G_t\mid F_t=0,\mathcal H)
\Big)
\Bigg\}.
\end{aligned}
}
\tag{6}
$$

式（6）中的四个矩均已在本节显式给出：前两个是场值的条件均值与方差，后两个分别是式（4）和式（5）。因此式（6）已经是 $t,\omega_o,A,\sigma_h,m,\mathbf x_i,\mathbf g_i$ 的闭式函数，不含数值积分，也不依赖 NDF 的数值积分。

对于论文常用的平面均值场 $m(\mathbf x)=z$，只需在上述各式中代入

$$
m(\mathbf x_i)=z_i,
\qquad
m(\mathbf x_t)=z_i+t\omega_{o,z},
\qquad
\nabla m(\mathbf x_i)=\nabla m(\mathbf x_t)=\mathbf e_z.
$$

### 5.5 Delta tracking 所需的可证明上界

式（6）能够逐点计算消光系数，但 delta tracking 需要在一个有限区间

$$
I=[t_a,t_b]
$$

上满足

$$
M_I\ge \Sigma_1(t\mid\mathcal H),
\qquad \forall t\in I.
$$

为了只在本小节讨论上界，记

$$
v_F(t)=\operatorname{Var}(F_t\mid\mathcal H),
\qquad
z(t)=\frac{\mathbb E[F_t\mid\mathcal H]}{\sqrt{v_F(t)}},
$$

$$
\mu(t)=\mathbb E[\omega_o^\top G_t\mid F_t=0,\mathcal H],
\qquad
s(t)=\sqrt{
\operatorname{Var}(\omega_o^\top G_t\mid F_t=0,\mathcal H)
}.
$$

这些只是式（4）—（6）中四个已有标量矩的简短名称。用标准高斯密度和 CDF，即 $\phi(\,\cdot\,;0,1)$ 与 $\Phi(\,\cdot\,;0,1)$，式（6）等价于

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
=
\frac{
\phi(z(t);0,1)
}{
\sqrt{v_F(t)}\,\Phi(z(t);0,1)
}
\left[
s(t)\phi\!\left(\frac{\mu(t)}{s(t)};0,1\right)
-\mu(t)\Phi\!\left(-\frac{\mu(t)}{s(t)};0,1\right)
\right].
}
\tag{7}
$$

当 $s(t)=0$ 时，中括号按连续极限取 $(-\mu(t))_+$。

假设通过带向外舍入的区间运算，在 $I$ 上得到经过证明的界

$$
v_F(t)\ge v_{F,\min}>0,
\qquad
z(t)\ge z_{\min},
$$

$$
\mu(t)\ge\mu_{\min},
\qquad
s(t)\le s_{\max}.
$$

标准高斯逆 Mills 比

$$
\frac{\phi(z;0,1)}{\Phi(z;0,1)}
$$

关于 $z$ 单调递减；同时

$$
s\phi(\mu/s;0,1)-\mu\Phi(-\mu/s;0,1)
\le
\frac{s}{\sqrt{2\pi}}+(-\mu)_+.
$$

因此下面的分段常数是一个可证明的 majorant：

$$
\boxed{
M_I
=
\frac{
\phi(z_{\min};0,1)
}{
\sqrt{v_{F,\min}}\,\Phi(z_{\min};0,1)
}
\left[
\frac{s_{\max}}{\sqrt{2\pi}}
+(-\mu_{\min})_+
\right].
}
\tag{8}
$$

这里还可以使用两个简化：

1. 由高斯条件化只会减小协方差，

   $$
   0\le s(t)^2
   \le\sigma_h^2\omega_o^\top A\omega_o,
   $$

   所以总可以安全地取

   $$
   s_{\max}=\sigma_h\sqrt{\omega_o^\top A\omega_o}.
   $$

2. 令 $u=t^2\omega_o^\top A\omega_o$，则

   $$
   v_F(t)=\sigma_h^2[1-(1+u)e^{-u}].
   $$

   因为

   $$
   \frac{d}{du}[1-(1+u)e^{-u}]=ue^{-u}\ge0,
   $$

   对不包含零点的区间 $I=[t_a,t_b]$，可以直接取

   $$
   v_{F,\min}=v_F(t_a).
   $$

$z_{\min}$ 与 $\mu_{\min}$ 一般没有统一的单调性，应使用式（4）和场值均值公式做区间求值；若区间结果太宽，就把 $I$ 二分。这样会得到一组分段常数 majorant $M_{I_j}$，通常比整段只用一个常数高效得多。

需要区分“消光系数有闭式”与“最大值的位置也有闭式”。式（7）含高斯 CDF 比值，方程 $d\Sigma_1/dt=0$ 一般没有初等函数解。因此，delta tracking 的上界仍应通过一维全局区间优化或经过证明的分段界获得，而不是期待一个通用的闭式 $\max_t\Sigma_1(t\mid\mathcal H)$。

数值实现时，令 $u=t^2\omega_o^\top A\omega_o$，分母最好计算为

$$
1-(1+u)e^{-u}
=-\operatorname{expm1}\!\left(\operatorname{log1p}(u)-u\right),
$$

并在 $u$ 很小时改用 Taylor 展开。逆 Mills 比应通过 $\log\Phi$ 或等价的正态尾函数在对数域计算。式（8）作为 majorant 时，所有区间端点必须向外舍入；普通浮点近似值即使看起来更大，也不能自动视为严格上界。

#### 包含 $t=0$ 的第一个区间

直接把 $t=0$ 放进式（8）会得到 $v_{F,\min}=0$，从而产生无穷上界；这是表示式的退化，不代表真实消光系数发散。若

$$
\omega_o^\top\mathbf g_i>0,
$$

则第一份文件的条件矩在 $t\to0^+$ 时满足

$$
\mathbb E[F_t\mid\mathcal H]
=t\omega_o^\top\mathbf g_i+O(t^2),
$$

$$
\operatorname{Var}(F_t\mid\mathcal H)
=\frac{\sigma_h^2}{2}
(\omega_o^\top A\omega_o)^2t^4
+O(t^6),
$$

$$
\mathbb E[\omega_o^\top G_t\mid F_t=0,\mathcal H]
\longrightarrow-\omega_o^\top\mathbf g_i,
\qquad
s(t)\longrightarrow0.
$$

因此

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
\sim
\frac{
\omega_o^\top\mathbf g_i
}{
\sigma_h(\omega_o^\top A\omega_o)\sqrt\pi\,t^2
}
\exp\!\left[
-\frac{(\omega_o^\top\mathbf g_i)^2}
{\sigma_h^2(\omega_o^\top A\omega_o)^2t^2}
\right]
\longrightarrow0.
}
\tag{9}
$$

实现时应把 $\Sigma_1(0\mid\mathcal H)$ 定义为这个连续极限 $0$，并对第一个很小的区间使用带余项界的 Taylor 形式或经过验证的区间算术；不要直接用两个接近数相减来计算式（4）—（5）的分母。

#### 是否存在整条无限射线的常数上界

一般不能假设存在。以 $m(\mathbf x)=z$ 为例，当 $t\to\infty$ 时，起点相关项消失。如果 $\omega_{o,z}<0$，则 $\mathbb E[F_t\mid\mathcal H]\to-\infty$，密度因子按 $-\mathbb E[F_t\mid\mathcal H]/\sigma_h^2$ 增长，而条件投影面积趋于正常数，所以 $\Sigma_1(t\mid\mathcal H)$ 通常线性增长，不存在有限的全局常数 majorant。

因此 delta tracking 应在有限介质段或到下一确定性边界的区间上构造 majorant。只在若干网格点计算 $\Sigma_1$ 后取最大值并乘经验安全系数，不能证明上界有效；一旦漏掉峰值，采样就会产生偏差。推荐对式（4）—（8）使用带向外舍入的区间 branch-and-bound，或使用能给出严格导数界的自适应分段。

### 5.6 与 NDF 的衔接

由下一节的 NDF 定义，条件投影面积也可以写成

$$
B_{\mathcal H}(t)
=\int_\Omega
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
\,d\omega_m.
$$

因此

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
=\rho_{\mathcal H}(t)
\int_\Omega
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
\,d\omega_m.
}
$$

这说明消光系数、NDF 和 VNDF 并不是三套独立假设：三者都由同一个条件梯度分布

$$
G_t\mid F_t=0,\mathcal H
$$

决定。

### 5.7 这一步仍然保留的近似

上述小区间推导对“端点条件模型”是正确的，但分母使用的是

$$
P(F_t>0\mid\mathcal H),
$$

而不是整段事件

$$
P\!\left(F_u>0,\ \forall u\in(0,t)\mid\mathcal H\right).
$$

因此，式（$\Sigma$）不能直接称为真实 GP 的精确首次碰撞 hazard。若去掉这个剩余近似，分子还会出现“给定终点场值与梯度后，整段此前均保持在空侧”的桥接存活概率；它通常依赖 $\mathbf g$，不能并入单独的标量 $\rho_{\mathcal H}(t)$。

## 6. NDF

对一个确定的 realization，零等值面的面积可由 coarea formula 写成

$$
\int_{\{\mathbf x:f(\mathbf x)=0\}}a\!\left(
\frac{\nabla f}{\|\nabla f\|}
\right)dA
=\int
\delta(f(\mathbf x))\|\nabla f(\mathbf x)\|
a\!\left(\frac{\nabla f}{\|\nabla f\|}\right)d\mathbf x.
$$

在固定位置 $\mathbf x_t$ 并条件于 $F_t=0,\mathcal H$ 后，梯度的面积权重是 $\|\mathbf g\|$。令

$$
\mathbf g=r\omega_m,
\qquad r>0,
\qquad d\mathbf g=r^2\,dr\,d\omega_m.
$$

面积权重 $r$ 与球坐标 Jacobian $r^2$ 相乘，得到 $r^3$。因此条件 NDF 为

$$
\boxed{
D(\omega_m\mid t,\omega_o,\mathcal H)
=\int_0^\infty
r^3
p_{G_t\mid F_t=0,\mathcal H}(r\omega_m)
\,dr.
}
\tag{NDF}
$$

式（3）代入上式，就已经完全用第一份 Markdown 的条件均值和条件协方差表达了 NDF；不需要再引入 $\bar g_t,S_t,a,b,c_\star,J_j$ 等中间符号。

这个 NDF 不是积分为 $1$ 的方向 PDF。它满足

$$
\int_\Omega
D(\omega_m\mid t,\omega_o,\mathcal H)\,d\omega_m
=\mathbb E[\|G_t\|\mid F_t=0,\mathcal H],
$$

以及

$$
\int_\Omega
\omega_mD(\omega_m\mid t,\omega_o,\mathcal H)\,d\omega_m
=\mathbb E[G_t\mid F_t=0,\mathcal H].
$$

广义 Macrofacet 的法线定义域是完整球面 $\Omega$，不能未经说明就缩成上半球。

## 7. NDF 与第一份文件中 $B_{\mathcal H}(t)$ 的一致性

第一份文件定义

$$
B_{\mathcal H}(t)
=\mathbb E\!\left[
(-\omega_o^\top G_t)_+
\mid F_t=0,\mathcal H
\right].
$$

将 $\mathbf g=r\omega_m$ 代入：

$$
\begin{aligned}
B_{\mathcal H}(t)
&=\int_{\mathbb R^3}
(-\omega_o^\top\mathbf g)_+
p_{G_t\mid F_t=0,\mathcal H}(\mathbf g)\,d\mathbf g\\
&=\int_\Omega
\langle-\omega_o,\omega_m\rangle
\left[
\int_0^\infty r^3
p_{G_t\mid F_t=0,\mathcal H}(r\omega_m)\,dr
\right]d\omega_m.
\end{aligned}
$$

所以

$$
\boxed{
B_{\mathcal H}(t)
=\int_\Omega
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
\,d\omega_m.
}
\tag{10}
$$

这证明了同一个条件梯度分布同时决定第一份文件的消光率和这里的 NDF。将密度记号写清后，第一份文件的消光率为

$$
\boxed{
\Sigma_1(t\mid\mathcal H)
=\frac{p_{F_t\mid\mathcal H}(0)}
{P(F_t>0\mid\mathcal H)}
B_{\mathcal H}(t).
}
\tag{11}
$$

## 8. VNDF

在位置 $\mathbf x_t$ 发生从空侧到负侧的局部穿越时，梯度 $\mathbf g$ 的权重是

$$
(-\omega_o^\top\mathbf g)_+.
$$

因此，碰撞条件下的法线密度由下式得到：

$$
\begin{aligned}
&\int_0^\infty
\frac{
(-\omega_o^\top r\omega_m)_+
p_{G_t\mid F_t=0,\mathcal H}(r\omega_m)
r^2
}{B_{\mathcal H}(t)}\,dr\\
&=\frac{
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
}{B_{\mathcal H}(t)}.
\end{aligned}
$$

故 VNDF 为

$$
\boxed{
D_{\omega_o}(\omega_m\mid t,\mathcal H)
=\frac{
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
}{B_{\mathcal H}(t)}.
}
\tag{VNDF}
$$

由式（10）立即得到

$$
\int_\Omega
D_{\omega_o}(\omega_m\mid t,\mathcal H)
\,d\omega_m=1.
$$

所以 $D$ 是面积加权 NDF，而 $D_{\omega_o}$ 才是在已发生该次碰撞时可直接采样的可见法线 PDF。

## 9. Phase function

沿用论文的方向符号，不再引入 $d,v$。理想镜面反射满足

$$
\omega_i
=\omega_o-2(\omega_o^\top\omega_m)\omega_m,
$$

因此

$$
\boxed{
\omega_m
=\frac{-\omega_o+\omega_i}
{\|-\omega_o+\omega_i\|}.
}
\tag{12}
$$

在可见法线支集上，$\omega_o^\top\omega_m<0$，并且法线立体角到反射方向立体角的 Jacobian 是

$$
d\omega_i
=4|-\omega_o^\top\omega_m|\,d\omega_m.
$$

因此，由 VNDF 采样法线再做镜面反射所得到的方向 PDF 为

$$
\begin{aligned}
\frac{
D_{\omega_o}(\omega_m\mid t,\mathcal H)
}{4|-\omega_o^\top\omega_m|}
&=\frac{
\langle-\omega_o,\omega_m\rangle
D(\omega_m\mid t,\omega_o,\mathcal H)
}{
4|-\omega_o^\top\omega_m|B_{\mathcal H}(t)
}\\
&=\boxed{
\frac{D(\omega_m\mid t,\omega_o,\mathcal H)}
{4B_{\mathcal H}(t)}
}.
\end{aligned}
\tag{13}
$$

最后，按照 Macrofacet 论文把 Fresnel 留在 phase function 中的约定，得到

$$
\boxed{
\begin{aligned}
p(\omega_o,\omega_i\mid t,\mathcal H)
&=\frac{
F(-\omega_o,\omega_m)
D_{\omega_o}(\omega_m\mid t,\mathcal H)
}{4|-\omega_o^\top\omega_m|}\\
&=\frac{
F(-\omega_o,\omega_m)
D(\omega_m\mid t,\omega_o,\mathcal H)
}{4B_{\mathcal H}(t)},\\
&\hspace{16mm}
\omega_m=\frac{-\omega_o+\omega_i}
{\|-\omega_o+\omega_i\|}.
\end{aligned}
}
\tag{Phase}
$$

这里双参数的 $F(-\omega_o,\omega_m)$ 是 Fresnel 项，不是随机场值 $F_t$。

论文令散射系数等于消光系数，并把反射损失保留在 phase function 内，所以一般有

$$
\int_\Omega
p(\omega_o,\omega_i\mid t,\mathcal H)
\,d\omega_i
=\mathbb E_{D_{\omega_o}}
[F(-\omega_o,\omega_m)]
\le 1.
$$

因此，式（Phase）通常不是积分为 $1$ 的纯方向 PDF。只有 $F=1$ 时，它才与式（13）的归一化几何方向 PDF 相同。不要同时把 Fresnel 放进 phase function，又在散射系数中重复乘一次。

## 10. 最短依赖链

统一符号后，第二份文件中相关部分可以压缩成下面四步：

$$
G_t\mid F_t=0,\mathcal H
\quad\xrightarrow{\text{式（1）—（3）}}\quad
D(\omega_m\mid t,\omega_o,\mathcal H)
$$

$$
B_{\mathcal H}(t)
=\int_\Omega
\langle-\omega_o,\omega_m\rangle D\,d\omega_m
$$

$$
\rho_{\mathcal H}(t)
=\frac{p_{F_t\mid\mathcal H}(0)}
{P(F_t>0\mid\mathcal H)},
\qquad
\Sigma_1(t\mid\mathcal H)
=\rho_{\mathcal H}(t)B_{\mathcal H}(t)
$$

$$
D_{\omega_o}
=\frac{\langle-\omega_o,\omega_m\rangle D}
{B_{\mathcal H}(t)}
$$

$$
p(\omega_o,\omega_i\mid t,\mathcal H)
=\frac{F(-\omega_o,\omega_m)D}
{4B_{\mathcal H}(t)}.
$$

这条链中没有使用第二份文件的 $d,v,\psi_t,\bar g_t,S_t,a,b,c_\star,J_j,q_R,\mathcal P_H$ 等额外简写。

## 11. 正确性边界

以上 NDF、VNDF 和 phase function 对第一份文件所定义的条件端点模型是自洽的：它们使用同一个 $G_t\mid F_t=0,\mathcal H$，VNDF 的归一化因子恰好等于 $B_{\mathcal H}(t)$，phase function 的 Jacobian 也与论文公式一致。

但仍需保留两点限定：

1. 式（11）继承了“终点在空侧代替整段存活”的近似，因此不等于一般 GP 的精确首次通过分布。
2. 条件 NDF 依赖 $t,\omega_o,\mathcal H$。反向路径具有不同的条件状态，因此单步归一化并不能单独证明完整路径互易性。

第二份文件中使用 $a,b,c_\star,J_j$ 得到的径向闭式积分在代数上是正确的，但这些量只服务于把式（NDF）写成更长的闭式，并不是建立 NDF、VNDF 或 phase function 所必需。为保持与第一份文件符号一致，这里停在可直接代入式（3）的单变量积分形式。

## 参考

- Minghao Huang, Yuang Cui, Beibei Wang, Lingqi Yan, *Macrofacet Theory for Gaussian Process Statistical Surfaces*, arXiv:2603.00280。

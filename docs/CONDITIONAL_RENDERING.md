# Global conditional rendering

## Model and data flow

The new `global_conditional` mode implements the model in
[`Macrofacet_去除第二个假设后的完整推导.md`](../Macrofacet_去除第二个假设后的完整推导.md).
Classic local and classic global remain separate modes and keep their baked
density, DDA tracking, and existing phase proposals.

Every conditional flight stores its birth position, direction, field value,
and **unnormalized** gradient. The SE kernel conditions the value and gradient
at age `t` on this observation. At a candidate collision, the gradient is
conditioned again on `F_t=0`. Its negative directional flux gives both the
extinction coefficient and the collision gradient distribution:

```text
sigma(t) = p(F_t=0 | H) / P(F_t>0 | H)
           * E[(-w.G_t)+ | F_t=0,H]
p(g | collision at t,H) proportional to (-w.g)+ p(g | F_t=0,H).
```

The first flight starts at the domain entry with a sample of `F_0>0` from the
truncated global point prior and an independent sample of its full gradient.
This is the positive exterior extension of the surface conditioning in the
derivation. Subsequent flights start with `F_0=0` and the complete gradient
sampled at the preceding collision. A collision reflects about its normalized
gradient; the path weight is conductor Fresnel and Russian roulette only.
There is no explicit transmittance sampling or multiplication.

The sampler constructs two bounds per flight: an analytic birth
envelope on `[0,h]`, and a conditional-moment interval bound on `[h,L]`.
Ordinary flights use their two constant Poisson rates. Expensive far segments
use certified interval thinning, with additional constant tracking intervals.
It accepts a candidate with `sigma(t)/majorant`. Constructing the bounds does
not sample the hazard or integrate transmittance. A failed bound or other
conditional numerical error propagates to the caller; it does not become a
black pixel sample.

## 两段保守上界

每段 flight 固定出生观测 `H` 和出生位置，以距离 `t` 为年龄。设
`a = w^T A w > 0`、`u = a t^2`、`sigma = sigma_h`。采样使用

$$
\overline\Sigma(t\mid H)=
\begin{cases}
M_{\rm near},&0\le t<h,\\
M_{\rm far},&h\le t\le L.
\end{cases}
$$

这是两种公式对应的整体上界。普通 flight 按这两个常数采样；极端上界的后段
会启用下文的区间 thinning 加速，此时实际采样可有更多常数区间。
两段使用同一个条件消光系数。切换上界时不增加 GP 观测、不重置年龄，
也不改变已经采到的完整起点梯度。到达段边界后，按下一段的上界重新采
指数距离。空碰撞也保留原来的 `H`。

### 稳定的条件统计

定义在零点连续的函数

$$
E_2(u)=\frac{e^u-1-u}{u^2}
=\sum_{n=0}^{\infty}\frac{u^n}{(n+2)!},\qquad
E_4(u)=\frac{2(\cosh u-1)-u^2}{u^4}
=\sum_{n=0}^{\infty}\frac{2u^{2n}}{(2n+4)!}.
$$

使用等价解析式

$$
v_F=\sigma^2e^{-u}u^2E_2(u),\qquad
\mu=m_K-\frac{m_F}{tE_2(u)},\qquad
s^2=\sigma^2a u^2\frac{E_4(u)}{E_2(u)}.
$$

小 `u` 用正项级数，大 `u` 使用含 `exp(-u)` 的等价式，避免指数溢出。
`E2` 取到 `n=20`，`E4` 取到 `n=12`，在 `u<=1` 的余项小于 double
舍入误差。起点统计、`A w` 和 `a` 在构造 flight 时缓存；候选点只计算
标量矩。碰撞处才构造三维梯度分布，其协方差改写为两个半正定分量之和：

$$
P=\frac{(A\omega)(A\omega)^T}{a},\qquad
S=\sigma^2\left[(1-e^{-u})(A-P)+\frac{s^2}{\sigma^2a}P\right].
$$

### 首段：用 Taylor 余项得到解析包络

先把首段限制在均值场的一个光滑区间内。NanoVDB 使用第一个出射插值单元；
仿射均值的二阶导数为零，球面均值使用到球心的最小距离给出曲率界。
若没有可用的曲率界，则报错并要求先烘焙 NanoVDB。

记起点值为 `f0`，沿出射方向的条件均值导数为 `k`，均值场二阶方向导数
绝对值上界为 `H2`。令 `df=f0-m(x0)`、`dk=w.(g0-grad m(x0))`，则在
`[0,h]` 上有

$$
|m_F(t)-f_0-kt|\le Et^2,\qquad
|m_K(t)-k|\le Dt,
$$

$$
E=\tfrac12[H_2+a(|\delta f|+h|\delta k|)],\qquad
D=H_2+a|\delta f|+\tfrac32ah|\delta k|.
$$

初始 `h` 不超过 `ell_w/2` 和首个光滑区间；表面起点还用
`k/(sigma*a)` 限制它，避免掠射时首段太宽。随后按需减半，直到：

- 表面起点：`k>0` 且 `E*h <= k/2`；
- 外部起点：`f0>0` 且 `abs(k)*h + E*h*h <= f0/2`。

这保证首段有正的均值下界，并不删除任何近起点距离。
设 `U=a*h*h`，将 `v_F=t^4 V(t)` 写为

$$
V_{\min}=\tfrac12\sigma^2a^2e^{-U},\qquad
V_{\max}=\sigma^2a^2E_2(U).
$$

利用 `E2>=1/2`、条件负通量的界
`B <= max(-mu,0)+s/sqrt(2*pi)`，得到：

- 表面起点：`m_F>=b*t`、`B<=Bmax`，其中 `b=k-E*h`；
- 外部起点：`m_F>=b`、`B<=Bcoef/t`，其中
  `b=f0-abs(k)*h-E*h*h`。

具体取 `smax` 为首段条件标准差上界，并定义

$$
C_B=(|k|+Dh)+2(|k|+Eh)+\frac{s_{\max}}{\sqrt{2\pi}}.
$$

表面起点用 `Bmax=C_B`，外部起点用 `Bcoef=2*f0+h*C_B`。
这里利用回归系数 `d(t)=1/(t*E2)<=2/t`，将起点的 `f0/t` 项单独提出。

由于正均值时 `Phi(z)>=1/2`，分别有

$$
\Sigma(t)\le C_1t^{-2}e^{-C_2/t^2}
\quad\text{或}\quad
\Sigma(t)\le C_1t^{-3}e^{-C_2/t^4},
$$

其中 `C2=b*b/(2*Vmax)`，`C1=sqrt(2/pi)*Bmax/sqrt(Vmin)`，
外部起点把 `Bmax` 换为 `Bcoef`。对于一般形式
`C1*t^(-p)*exp(-C2/t^r)`，最大值位于

$$
t_* = \min\left(h,\left(\frac{rC_2}{p}\right)^{1/r}\right).
$$

在 `t_*` 以对数形式求值即得到 `M_near`，并覆盖 `t=0` 的零极限。

### 后段：区间界合并为一个常数

`t>=h>0` 后条件值方差严格为正。内部先用区间长度
`min(t/2,ell_w/2)` 对条件矩求界，再取所有区间上界的最大值为
`M_far`。普通路径把这些内部界合并为一个常数；极端路径会复用它们加速空碰撞采样。
没有在内部区间取若干 hazard 样本后估计最大值。
在 `t<ell_w` 且当前上界乘以后段长度大于 `256` 时，将求界区间减半，
最细到 `t/16`。这使同一区间内 `t^4` 方差的变化受控，也适用于跨过首个插值单元之后。
`256` 只决定是否多花开销收紧已经有效的界，不限制上界大小或候选点数量。

NanoVDB 沿射线在插值单元边界切分。均值和方向导数在每个单元的包围盒上
都是多重仿射函数，其范围由盒角点给出；二阶方向导数用角点混合差分求界。
核均值修正项是 `exp(-a*t*t/2)` 乘以一次或二次多项式，多项式范围检查
端点和内部驻点，再做区间乘法。
另外使用条件均值的导数范围，从区间中点计算均值的第二个保守包围区间，
并与前一个区间取交。这保留了均值场与起点修正之间的抵消，防止正均值
被过松的独立区间误判成可能为负；中点只求均值，不探测消光峰值。

每个内部区间 `[l,r]` 得到 `m_F in [fmin,fmax]`、`m_K>=kmin`。
`v_F` 单调递增，而 `d(t)=1/(t*E2(a*t*t))` 单调递减，因此

$$
z_{\min}=\begin{cases}
f_{\min}/\sqrt{v_F(r)},&f_{\min}\ge0,\\
f_{\min}/\sqrt{v_F(l)},&f_{\min}<0,
\end{cases}
\qquad
\mu_{\min}=k_{\min}-\max\{d(l)f_{\max},d(r)f_{\max}\}.
$$

条件标准差使用 `s<=sigma*sqrt(a)`，小 `u` 时用
`s<=sigma*sqrt(a)*u*sqrt(2*E4(u))` 收紧。密度因子的 Mills 比率使用

$$
\frac{\phi(z)}{\Phi(z)}\le
\begin{cases}2\phi(z),&z\ge0,\\1-z,&z<0.\end{cases}
$$

将上述密度界与负通量界相乘即可得到整个内部区间的上界。
当 `mu_min>=0` 时，还使用更紧的负通量界
`B<=smax*phi(mu_min/smax)`，保留正方向条件梯度下负通量的指数抑制。
靠近起点且仍在首个光滑区间内时，使用前述 Taylor 余项界处理均值，避免
相近量相减造成的界退化。

解析不等式按连续场推导；实现为 double 运算，包含级数余项裕量、场求界
的舍入裕量和最终向上扩张。它不使用区间算术库逐条认证平台数学函数的
舍入。候选点越界会抛出 `InvalidMajorant`，不会夹紧接受概率或继续出图。

### 极小步长与极端后段的数值处理

256×256、8 spp 的 shader ball 复验暴露了小图没有覆盖的情况：在一个捕获的
flight 中，`h≈5.58e-8`，`M_far≈1.84e16`，真实消光峰值也约为 `1.25e16`。
它位于很近的插值单元交界之后。大上界既会产生大量空碰撞，也可能让一个指数步长
小于当前年龄的浮点间隔，使直接计算 `age+distance` 仍等于 `age`。

现在用两个 double 保存距离和及其舍入余量，小步长会累加到余量中。候选的 double
年龄可以暂时与前一个相同，仍执行接受/拒绝；没有强制最小步长、`nextafter` 推进
或删除起点附近的距离。真正下溢为零的距离仍作为数值错误报告。

仅补偿加法不能解决数十亿次空碰撞，因此后段同时使用以下加速：

1. 若 `M_far*(L-max(h,currentAge))<=8192`，仍用一个后段常数。
2. 否则复用构造上界时的区间界 `B<=M_far`。实际走到一个区间时，若
   `B*区间长度>64`，继续二分并用同一后段公式求保守界；发生真实碰撞后不再处理后续区间。
3. 在每个访问的区间用 `Exp(B)` 提出候选，以 `Sigma/B` 接受。

其依据是 Poisson thinning：速率 `M_far` 的过程可以拆成相互独立的 `B` 和
`M_far-B` 两个过程；后一个过程全部为空碰撞，可以直接省去。留下的真实事件强度
仍为 `B*(Sigma/B)=Sigma`。两个数字只控制计算开销，不截断消光或事件数量。
这保留了近起点/后段两种公式，但**极端情况下实际候选采样不再严格只有两个常数区间**。
整个过程仍不积分或反演透射率，也不修改出生观测。

此外，深负尾中的 `phi(z)/Phi(z)` 使用连分式计算，避免两个量级为 `-z*z/2`
的对数相减时丢掉 Mills 比率。原来的直接相减在很小条件方差下也会损失有效位。

### 开销和输出

`M_far` 覆盖整个剩余 flight，场变化大或发生掠射时仍可能产生较多空碰撞。
两段实现保证的是上界构造方式，不能保证所有场景都比空间细分更快。
NanoVDB 三线性均值在单元交界处只有值连续，方向导数可能跳变；出生后不远处
也可能出现真实的高消光窄峰。用一个常数覆盖整个后段，会把这些峰值的开销带到
更长的距离上。收紧求界区间只能减少估计余量，不能消除真实峰值带来的开销。
`render_summary.csv` 新增 `bound_intervals`、`near_candidates`、
`far_candidates`；前者统计求界用的内部区间（含收紧前的尝试），后两者统计真正的 Poisson
候选点。`resolved_config.json` 将采样器记录为
`two_segment_delta_tracking`。
`far_interval_thinning=true` 表示允许上述区间加速；CSV 的
`adaptive_majorant_flights` 统计启用次数，`rounded_candidate_steps` 统计补偿和的
高位暂时不变的候选数。

## Comparison commands

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_sphere.json --mode all --width 16 --height 16 --spp 8 --threads 4 --output outputs\two_segment_sphere_validation
build\Release\macrofacet_experiments.exe curves --config configs\render_sphere.json --rays 1024 --bins 64 --output outputs\two_segment_sphere_validation
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball_nanovdb.json --mode all --width 16 --height 16 --spp 8 --threads 4 --output outputs\two_segment_after_ball
```

The `all` render writes a BMP preview and linear PFM for each mode and
environment, plus `render_summary.csv`. `curves` uses one fixed direction and
a grid of origins. Each intersecting ray is traced once per mode. Its CSV
records the origin, direction, entry point, path length, first collision, and
escape flag. The curve CSV and SVG contain empirical survival against distance
from each ray's domain entry, clipped at domain exit. Those curves describe
first-flight transmittance; they do not include scattering after collision.

## 两段实现验证

Release 构建及 `ctest --test-dir build -C Release --output-on-failure` 通过
（1 个测试可执行文件，包含全部检查）。新增检查覆盖：

- 独立的 4×4 Gaussian 条件化计算与稳定公式的一致性；
- `t=1e-5,1e-7,1e-9` 的条件方差四阶极限；
- 表面、外部和掠射起点的对数距离扫描，上界覆盖及跨 NanoVDB 单元检查；
- 每组 2000 次 flight 的逃逸概率与独立积分的透射率比较，包含从非零年龄继续追踪；
- 单线程与多线程的图像和计数一致性。

这些数值检查用来发现实现错误；连续区间上的覆盖依据是前面的解析不等式。

以下记录来自数值修复前、严格使用两个常数的版本，保留作为历史对照。
使用固定种子 `17429`、`16×16`、`8 spp`、4 线程的单次运行，耗时来自
`render_summary.csv`，只作为小图回归记录，不代表稳定的性能基准：

| 场景 / 模式 | 白环境耗时 / s | 方向环境耗时 / s | 方向环境候选数 |
| --- | ---: | ---: | ---: |
| sphere / classic_local | 0.0064 | 0.0050 | 7,239 |
| sphere / classic_global | 0.2770 | 0.2675 | 1,466,190 |
| sphere / global_conditional | 0.1036 | 0.0942 | 158,956 |
| shader ball / classic_local | 0.0057 | 0.0042 | 21,058 |
| shader ball / classic_global | 0.0335 | 0.0324 | 399,742 |
| shader ball / global_conditional | 0.7235 | 0.7458 | 3,251,109 |

两场景的所有模式均无数值失败；conditional 均无深度截断。shader ball 的
classic_global 白环境有 3 条、方向环境有 1 条路径触及现有的 64 层深度上限，
因此这两张 classic 图仍包含该截断限制。

修改前的扫描上界 conditional 版本也保存了相同小图预算的记录：平面约
`0.135/0.147 s`，本版约 `0.082/0.093 s`；shader ball 约 `0.292/0.332 s`，
本版约 `0.724/0.746 s`（白环境 / 方向环境）。旧扫描上界缺少连续区间覆盖证明，
这些时间不能作为等价保守采样器之间的速度比较。改变候选过程也改变后续随机数消耗，
即使种子相同，路径和像素也不应逐一相等。当前两段方案在 shader ball 上仍有性能限制。

透射率验证发出 1024 条平行射线，其中 774 条与球体配置的活动域相交，三种模式复用
相同的射线几何。曲线最终逃逸比例分别为 `0.888889`、`0.881137`、`0.882429`
（local classic / global classic / global conditional）。这是首程抽样比较，
并不要求三种不同模型给出相同曲线。

产物位置：

- [球体比较图、CSV 与透射率曲线](../outputs/two_segment_sphere_validation/)
- [shader ball 比较图与计数](../outputs/two_segment_after_ball/)
- [平面两段结果](../outputs/two_segment_after_plane/)
- [旧平面记录](../outputs/two_segment_before_plane/)和[旧 shader ball 记录](../outputs/two_segment_before_ball/)

输出目录是本地实验产物；重新生成可使用上面的命令。

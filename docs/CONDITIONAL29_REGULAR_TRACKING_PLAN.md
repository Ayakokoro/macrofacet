# conditional29：适配现有框架的 regular tracking 改造方案

状态：本文保留改造前的分析与设计。regular tracking、Newton 反演、正场值首段、条件统计和验证已落地，实际接口、配置和范围见 [实现说明](CONDITIONAL29_IMPLEMENTATION.md)。依据 `Macrofacet_去除第二个假设后的完整推导.md`，文内公式编号均指该推导文档。

目标：针对保留最近起点场值与完整梯度的 conditional29 模型，使用连续累计消光反演采样自由程；采用自适应数值求积与带区间保护的 Newton 求根；距离、梯度、相函数与路径权重使用同一条件分布。

这里 regular tracking 特指 inverse optical-depth sampling，不引入 majorant、null collision 或接受/拒绝碰撞。数值积分区间只是计算分块，不是物理事件。

## 1. 当前架构及实际执行路径

```text
ExperimentMain / ExperimentConfig
  └─ RenderExperiment::runRenderExperiments
      └─ renderAnalyticScene：相机、逐像素 RNG、动态行线程调度
          └─ traceCameraPath：AABB 入口、路径吞吐量、反射、RR、环境光
              ├─ makeFlightKernel
              │   ├─ ClassicFlightKernel
              │   ├─ Conditional29FlightKernel
              │   └─ MidpointFlightKernel
              ├─ sampleTabulatedFlight（当前所有模式实际使用）
              │   └─ integrateHazard → integrateFinite / GK15
              └─ sampleCollisionGradient（相关模式）
                  └─ sampleFluxWeightedGradient
```

底层 `gpss` 提供平面/球面/常数均值场、SE 核及条件统计；`mathutility` 提供高斯概率、截断矩、PSD 条件化、积分和求根；`macrofacet` 提供 NDF、Classic 系数及导体 Fresnel。`transport` 定义单段输运，`integrator` 组织多次碰撞，`experiments` 负责配置、实验与输出。

| 现状 | 依据 | 对改造的影响 |
| --- | --- | --- |
| 已有 conditional29 hazard 与端点零值条件梯度 | `src/transport/Conditional29FlightKernel.cpp` | 保留统计模型，改进稳定性和共用计算 |
| 已保存完整起点梯度 | `FlightState`、`ConditionedRay` | 不能改成只保存单位法线 |
| 渲染使用分格累计光学厚度，格内线性反演距离 | `MacrofacetPathTracer.cpp::sampleTabulatedFlight` | 实际采样分段常数 hazard 模型；需接入连续反演 |
| 独立 `sampleFlight` 已积分连续 hazard，但通过二分求根 | `OpticalDepthSampler.cpp`、`RootFinding.h` | 可原位扩展，无需另造一套输运体系 |
| 所有 External 起点都返回 Classic kernel | `FlightKernel.cpp::makeFlightKernel` | 首段初始化需要明确策略 |
| `correlated_sampler`、`classic_sampler` 未解析为渲染选择项 | `ExperimentConfig.cpp` | 不能只改 JSON 就认为算法已切换 |
| `external_policy` 只校验字符串，工厂参数被忽略 | `ExperimentConfig.cpp`、`FlightKernel.cpp` | 需要实际存储、传递和分派 |
| `hard_depth_cap: null` 未控制实际上限，代码仍有 64 层 safety cap | `RenderConfig`、`traceCameraPath` | 必须区分完成路径与截断路径 |
| `NumericError` 被计数后转成黑色贡献 | `traceCameraPath` | 会把数值失败混入物理吸收，应改为可诊断失败 |

因此，当前 conditional29 渲染并不是 delta tracking；主要变化是从粗分格自由程升级为连续 hazard 的 regular tracking，并补齐相关状态与配置语义。

## 2. 统一的条件统计与 hazard

每段固定起点观测 $H=\{F_0=f_0,G_0=g_0\}$、出生点 $x_0$ 和传播方向 $d$。表面出生时 $f_0=0,d^Tg_0>0$；正场值出生时 $f_0>0$，方向导数可正可负。

用式（4）—（8）得到：

$$
(F_t,G_t)\mid H\sim\mathcal N\left(
\begin{bmatrix}m_F\\m_G\end{bmatrix},
\begin{bmatrix}v_F&c^T\\c&C_G\end{bmatrix}\right),
\qquad
\bar g=m_G-\frac{c}{v_F}m_F,\quad
S=C_G-\frac{cc^T}{v_F}.
$$

再令

$$
\mu=d^T\bar g,\quad s^2=d^TSd,\quad
B=s\phi(\mu/s)-\mu\Phi(-\mu/s),
$$

$$
\lambda_H(t)=\frac{\phi(m_F/\sqrt{v_F})}{\sqrt{v_F}\,\Phi(m_F/\sqrt{v_F})}B.
$$

$s=0$ 时取 $B=(-\mu)_+$。沿线距离求值采用式（14）、（16）、（19）、（20）的五个标量矩；实际碰撞时才计算式（15）、（17）、（18）所需的三维统计。

建议在 `ConditionedRay` 中集中提供标量条件零值斜率和完整条件零值梯度，避免 kernel 两条分支分别组合出不一致的条件分布。通用矩阵条件化保留为测试参考及不适用显式公式时的路径。

对于一般均值场，显式 SE 公式仍可使用，只需在 $x_0,x_t$ 求均值及梯度，不必在球面路径的每次 hazard 求值中重做动态高斯分解。现有核支持旋转后的完整 PSD precision 矩阵；用矩阵形式的 $A$，不能把实现限制为世界坐标对角矩阵。

现有 `normalLogCdf`、`negativePartMean` 可复用：

$$
\log\lambda=\log p(F_t=0\mid H)+\log B-\log P(F_t>0\mid H).
$$

`NumericPolicy` 必须显式传入并由每段对象保存；渲染开始后不改共享的全局默认策略。

## 3. 光学厚度、透射率与 Newton 的职责

光学厚度是 $A$，透射率是 $T$：

$$
A_{a,t}=\int_a^t\lambda_H(u)\,du,\qquad
T_{a,t}=e^{-A_{a,t}},\qquad
p_{a}(t)=\lambda_H(t)e^{-A_{a,t}}.
$$

其中 $a=\texttt{currentAge()}$；从出生点开始时 $a=0$。所有年龄仍相对原出生点，积分分块不会更改 $H$。

Newton 是求根方法。实施方式为：数值求积得到 $A$，Newton 反演 $A$；对于已知终点的透射率查询，只积分并指数化，不需要 Newton。这与推导式（11）、（34）—（37）一致。光学厚度与透射率的定义也可对照 [PBRT 4e: Transmittance](https://pbr-book.org/4ed/Volume_Scattering/Transmittance)。

取 $U\in(0,1)$，生成 $E=-\log(1-U)$，与文档的 $-\log U$ 同分布：

$$
E\ge A_{a,L}\Rightarrow\mathrm{Escape}(L),\qquad
E<A_{a,L}\Rightarrow A_{a,t}=E.
$$

碰撞分支与逃逸原子的质量满足

$$
\int_a^L p_a(t)\,dt+T_{a,L}=1.
$$

求根迭代使用解析导数 $A'(t)=\lambda_H(t)$：

$$
t_{n+1}=t_n-\frac{A_{a,t_n}-E}{\lambda_H(t_n)}.
$$

不需要对 hazard 再求导，也不使用有限差分估计 $A'$。

## 4. 可复用的 regular tracking 采样器

对外保留 `FlightSample`，在 `OpticalDepthSampler` 中增加带确定性指数目标的入口，方便测试以及重试时保留同一随机变量：

```cpp
FlightSample sampleRegularFlight(
    const FlightKernel&, Random&, const RegularTrackingOptions&);

FlightSample invertOpticalDepth(
    const FlightKernel&, double exponentialTarget,
    double maximumAge, const RegularTrackingOptions&);

PositiveResult integrateHazard(
    const FlightKernel&, double beginAge, double endAge,
    const NumericPolicy&);
```

`maximumAge` 默认使用域出口，也允许 fixed-flight 实验显式截到 `requestedMaximumAge`。输出 `age` 始终是绝对 flight age，不能混为距当前位置的增量。

在 `RootFinding.h` 新增可接收函数和导数的 `solveMonotoneSafeguardedNewton`。保留现有二分接口供高斯 quantile 等调用，避免让无导数的调用点一起改语义。

采样器内部维护逐段积分记录：

```text
OpticalDepthSegment { lo, hi, integral, errorEstimate, prefixDepth }
```

建议第一版直接采用以下算法，不先积分整段再反复从原点重积分：

1. 固定当前 `FlightState` 与指数目标 $E$。
2. 沿 `[currentAge, maximumAge]` 分块，调用现有自适应 GK15 积分。
3. 保存各块质量与前缀和，直到累计光学厚度超过 $E$，或到达真实域出口。
4. 若在误差控制后到出口仍未达到 $E$，返回逃逸；否则定位包含根的块 `[l,r]`。
5. 固定块左端 $l_0$ 与其前缀 $A_{a,l_0}$。Newton 中计算
   $A_{a,t}=A_{a,l_0}+\int_{l_0}^t\lambda_H(u)\,du$，复用前缀及已有积分节点。
6. 格内线性插值只用于初始猜测，最终距离必须通过连续积分残差验收。

积分初始分块同时参考相关长度 $\ell_d=(d^TAd)^{-1/2}$ 与均值变化尺度。平面可增加靠近均值零交叉的节点；球面可结合最近点/均值变化作细分。已观测正场值很小且初始斜率为负时，还要考虑近端预测交叉尺度。单次大区间 GK15 即使报告小误差，也可能遗漏狭窄峰，不能完全依赖全域的一次初始采样。

分块只影响计算效率和积分误差，不定义替代 hazard，也不产生额外随机事件。多次抽样同一 fixed flight 时可复用完成的积分分区；渲染中每次真实碰撞之后建立新缓存。不能按位置与方向缓存跨路径结果，因为 $H$ 还包含完整梯度及起点场值。

Newton 的保护规则：

- 初始猜测位于有碰撞质量的块内部；不在表面出生点 $t=0$ 直接除以零 hazard。
- 维护根区间，按累计积分残差的符号更新左右端。
- 当误差估计使残差符号不明确时先细化积分，或在已经满足整体残差容差时结束；不凭有歧义的符号排除根。
- Newton 步越界、导数不可可靠使用、计算非有限或连续迭代停滞时，回退到区间中点。回退是确定性求根保护，不是 delta tracking。
- 使用光学厚度残差与距离尺度双重准则。可采用 `abs(A_est-E)+integration_error <= tau_A`，并要求区间宽度或可靠的 Newton 距离修正达到 `tau_t`；导数近零时采用区间宽度判据。
- 根迭代耗尽与未形成 bracket 都返回明确数值失败，不得当作逃逸。

GK15 的 `absError` 是数值误差估计，不是严格区间算术上界。接近边界判定或根残差阈值时收紧积分；由独立的更高精度参考验证控制效果，不宣称机器实现严格无偏。

求积容差应按总光学厚度预算分配给各块，累计所有误差估计；不能给每个块完整的全局绝对误差预算后忽略块数。建议独立设置 optical-depth 容差和 distance 容差，因为两者量纲不同。

命中返回值：

```text
collided       = true
age            = t
logSurvival    = -A_est(a,t)
logDistancePdf = log(lambda_H(t)) - A_est(a,t)
escapeMass    = nullopt
```

逃逸返回值：

```text
collided       = false
age            = L
logSurvival    = -A_est(a,L)
logDistancePdf = nullopt
escapeMass    = exp(-A_est(a,L))
```

收敛时 $A_{a,t}\approx E$；输出实际累计积分以及残差诊断，避免用 `-E` 掩盖一次未收敛的求根。上述 PDF 是容差内的目标密度；数值反演仍有误差，不把它描述为精确符号反演。

## 5. 近起点与退化统计

表面出生、有限正相关尺度下，当 $q=t/\ell_d\to0$：

$$
v_F=\sigma_h^2\left(\tfrac12q^4-\tfrac13q^6+\tfrac18q^8+\cdots\right).
$$

现有 `varianceF <= covarianceTolerance(...)` 且均值为正就直接返回零 hazard 的做法需要替换：小方差不意味着该段碰撞质量为零，尤其是掠射起点。新的处理是：

- $t=0$ 的合法出生点单独返回极限 hazard；有限正相关尺度、严格向外的表面出生取零。
- $t>0$ 使用无量纲量、`expm1` 与级数，稳定计算 $v_F$、$c$、$C_G$ 以及再次条件化后的 $S$ 和 $s^2$。
- 稳定性不能只覆盖 $v_F$；$v_K-c_{FK}^2/v_F$ 和三维 Schur 补也有严重消减，需要统一稳定表达式。
- 标量与完整梯度的条件均值、投影方差必须一致；不得靠只修补某一个投影掩盖三维协方差问题。
- 方差小但非零时保留分布；仅容许有尺度依据的舍入级负特征值修正，不加入物理 variance floor 或固定 jitter。
- 真正退化的 PSD 统计与小正方差分开处理。若 $d^TAd=0$ 导致确定性沿线首穿，可能存在距离原子，当前连续 `FlightSample` 路径不支持；保留明确的 `UnsupportedSingularFlight` 或另做专门事件实现，不能返回伪逃逸。
- 严重尾部使用对数概率、稳定的负通量矩及 CDF；普通数值下溢与数学上的零质量应在状态中区分。

当前环境为 MSVC，`long double` 通常不能提供比 `double` 更多的有效精度。第一选择是稳定解析表达式和细化求积；若确需更高精度，应引入实际的多精度后端并报告使用情况，不能仅打开 `higher_precision_fallback` 就声称已经实现。

## 6. 碰撞梯度、NDF、相函数与权重

沿用式（43）—（48）的完整梯度采样：

$$
q_G(g\mid t,H)=\frac{(-d^Tg)_+\,\mathcal N_3(g;\bar g,S)}{B}.
$$

先采一维负通量加权斜率

$$
q_K(k)=\frac{-k}{Bs}\phi\left(\frac{k-\mu}{s}\right),\quad k<0,
$$

再采剩余的两个横向高斯分量。现有 `sampleFluxWeightedGradient` 使用与 $d$ 正交的二维基，和文档的秩二条件高斯采样等价，可以复用。普通的负半轴截断高斯不能替代该加权分布。

`sampleNegativeFluxNormal` 目前用二分；距离 Newton 是本次必需项。该一维 CDF 也可用同一个受保护 Newton 接口加速，导数为 $q_K$，但应在尾概率稳定性测试通过后接入。

条件 NDF 直接用式（22）：

$$
D_H(n;t)=\int_0^\infty r^3\mathcal N_3(rn;\bar g,S)\,dr.
$$

`GaussianNdf` 已接受任意均值、SPD 协方差，其径向截断三阶矩实现可以复用，参数必须来自本次 `hitStatistics(t)`，不能使用 Classic 的无条件统计。通用奇异协方差不能直接代入普通 NDF 密度求值；完整梯度 PSD 采样的可用范围更广。

对反射方向 $v$，取 $n_h=(v-d)/\|v-d\|$，则

$$
q_R(v)=\frac{D_H(n_h;t)}{4B},\qquad
\mathcal P_H(d,v)=\mathcal F(-d^Tn_h)q_R(v).
$$

`q_R` 是标量方向 PDF，$\mathcal P_H$ 是带 Fresnel 的 RGB 散射核，两者不能混用。建议提供接受 `HitStatistics` 的轻量条件散射求值辅助函数，用于校验和后续光源采样；当前 analog 相机路径不必在每次采样后再计算 NDF。

一次真实碰撞的流程是：

```cpp
g = sampleCollisionGradient(kernel, flight.age, rng, policy);
n = normalizedOrThrow(g);
cosine = -dot(state.direction, n); // 要求 > 0，不用 abs 掩盖方向错误
throughput *= conductorFresnel(cosine, material);
outgoing = reflectTravelDirection(state.direction, n);
state = startSurfaceFlight(hit, g, outgoing);
```

在距离与梯度匹配采样下，式（56）—（58）给出碰撞权重仅为 Fresnel，逃逸权重为 1。不重复乘 $T$、$\lambda$、NDF 或宏法线余弦。RR 存活后除以其存活概率。

完整梯度必须存入下一段，不能存成 `normalize(g)`。只有真实碰撞重置出生点和 age；积分子区间与求根试探点均不更新路径状态。

## 7. 首段初始化与配置兼容

推导的表面起点假设不适用于相机首段。方案提供两个明确可选择的初始化模型：

| 策略 | 首段状态 | 用途 |
| --- | --- | --- |
| `original_macrofacet` | 沿用无起点观测的 Classic hazard，首碰仍采完整梯度 | 保持旧场景含义，便于单独比较距离算法 |
| `sampled_exterior` | 在活跃域入口采样 $F_0>0$ 与完整 $G_0$，首段即使用 conditional29 | 文档正场值初始化在当前 AABB/域外真空框架下的适配，推荐用于完整相关流程 |

两种策略的首段距离都可用 regular tracking；使用 Classic hazard 不意味着必须使用 null tracking。

`sampled_exterior` 采用

$$
F_0\sim\mathcal N(m(x_e),\sigma_h^2)\mid F_0>0,\qquad
G_0\sim\mathcal N_3(\nabla m(x_e),\sigma_h^2A).
$$

平稳 SE 核下二者先验独立。采完后固定这些观测直到首碰/逃逸，在所有条件均值公式中使用 $\delta f=F_0-m(x_e)$，不要求 $d^TG_0>0$。截断场值通过稳定尾概率反演采样，避免在罕见正侧事件中无界拒绝采样。

这里 $x_e$ 是相机在域外时的 AABB 入口，相机在域内时是相机位置。这是明确的域入口源端模型，不能声称与“在域外相机位置对同一无限 GP 条件化”完全相同。域外已配置为真空，不应暗中加入域外 GP 条件观测。

状态层建议扩展：

```cpp
enum class BirthKind { External, Surface, ObservedExterior };
// FlightState 增加 birthValue；Surface 为 0，ObservedExterior 严格 > 0。
// hasFullGradient 继续表示是否持有完整观测。
```

`ConditionedRay` 增加一般起点观测构造，并保留当前表面构造作为兼容包装。工厂对 `ObservedExterior` 构造 Conditional29 kernel；对 `External` 使用显式选择的旧策略。

为不让一个新首段策略误改 CI 中同时运行的三个模式，建议添加 mode 专属覆盖，保留现有公共字段作为兼容默认：

```json
{
  "transport": {
    "external_policy": "original_macrofacet",
    "correlated_sampler": "optical_depth",
    "conditional29": {
      "sampler": "regular_tracking",
      "external_policy": "sampled_exterior",
      "optical_depth_solver": "safeguarded_newton"
    },
    "next_event_estimation": false,
    "roulette_start_depth": 4,
    "hard_depth_cap": null
  }
}
```

以上是建议新增的配置语义，当前解析器尚不支持该覆盖块。解析优先级为 mode 覆盖 > 公共字段 > 默认值；`optical_depth` 作为 `regular_tracking` 的兼容别名。没有覆盖且仍显式配置 `original_macrofacet` 的旧文件，保留其首段策略；本次新 conditional29 验收配置明确选择 `sampled_exterior`。

相关新增选项可放入 `Conditional29TransportConfig`，不需要复制整份 `ExperimentConfig`。`classic` 与 `midpoint` 本轮保持既有执行行为，resolved config 必须记录实际解析出的算法，避免沿用旧字符串造成误解。

`flight_table_cells` 在 conditional29 regular tracking 下只允许作为初始积分分块提示，或明确不适用；它不能再控制目标 hazard 的离散化。相应更新 `--flight-cells` 帮助信息。绝对距离容差与光学厚度容差使用不同配置项，默认值从现有 `numeric` 迁移并注明量纲。

## 8. 渲染接线、透射率查询和失败处理

```text
相机射线 / AABB 入口
  → 按策略初始化 External 或 ObservedExterior
  → makeFlightKernel(Conditional29, state, policy)
  → sampleRegularFlight：积分分块 + safeguarded Newton
      ├─ Escape：贡献 beta × 环境光
      └─ Collision：条件零值梯度 → flux 采样 → Fresnel → 镜面反射
                     → 保存完整 g，重置 age → RR → 下一段
```

conditional29 路径不再调用 `sampleTabulatedFlight`。保留按像素种子生成 RNG 的方式；积分和求根不消耗随机数，失败重试保持原始 $E$，使线程数不改变图像。

已知端点的 shadow-ray 透射率查询统一调用 `integrateHazard`，返回 `exp(-A)`。当前渲染只有环境光与 analog 路径，`next_event_estimation=false` 足以实现文档的完整随机游走。

若以后启用 NEE，应按式（63）—（64）在当前完整梯度尚未固定时，采光源方向、确定半程法线，再按式（53）采模长 $r$，使用新条件 $H'_r$ 计算 shadow-ray 透射率。不能在固定了 $g$ 后，对任意光源方向套用边缘化相函数。其模长密度为 $r^3\psi(rn)/D_H(n)$，可用数值 CDF 与受保护 Newton 反演。MIS 需在相同扩展状态空间比较 proposal，本次不把现有开关直接视为已有该实现。

数值失败策略：先在同一 $E$、同一 $H$ 上收紧积分/切换稳定表达式，预算耗尽则传播失败并报告位置、方向、age、模型和状态；不重抽随机数，不静默返回黑色，也不把失败视为逃逸。

`hard_depth_cap: null` 应表示不施加有偏的物理反射次数截断，路径终止由逃逸和 RR 决定。若保留 emergency safety cap，触发表示该次运行未通过验收，而非正常零贡献。显式有限深度的预览结果应记录截断计数。

增加每线程聚合的诊断：hazard 求值数、积分细分数、Newton 次数、二分回退数、最大累计残差、积分误差估计、失败状态分布、实际首段策略。统计缓存均归当前路径或线程所有。

## 9. 文件改动清单与实施次序

| 文件/模块 | 具体工作 |
| --- | --- |
| `include/macrofacet/mathutility/RootFinding.h` | 新增有导数的受保护 Newton；保留原二分接口 |
| `Quadrature.h` | 扩展可复用分区/误差统计能力，或由 sampler 包装已有 GK15 |
| `NumericPolicy.h` | 定义距离与累计光学厚度容差、失败与诊断语义 |
| `transport/OpticalDepthSampler.{h,cpp}` | regular tracking、确定性目标反演入口、分段前缀缓存、逃逸质量 |
| `gpss/ConditionedRay.{h,cpp}` | 一般 $F_0,G_0$ 观测、SE 显式统计、近端稳定 Schur 补 |
| `transport/Conditional29FlightKernel.{h,cpp}` | 复用统一条件统计，显式 policy，替换小正方差粗暴归零 |
| `transport/FlightState.{h,cpp}` | 正侧出生状态、birthValue、状态不变量 |
| `transport/FlightKernel.{h,cpp}` | 实际执行首段策略，区分 External 与 ObservedExterior |
| `transport/CollisionGradientSampler.cpp` | 复用现有 flux 采样，补稳定性/投影一致性检查 |
| `macrofacet/GaussianNdf` / 条件散射辅助函数 | 使用条件参数求 NDF、方向 PDF 与散射核，不混用 Classic 相函数 |
| `integrator/MacrofacetPathTracer.{h,cpp}` | conditional29 接入新 sampler、初始化分支、失败传播与诊断 |
| `experiments/ExperimentConfig.{h,cpp}` | 真正解析 sampler、mode 覆盖、首段策略、深度上限与容差 |
| `experiments/ExperimentMain.cpp` | 同步 CLI 参数语义与 resolved config |
| `experiments/FlightCurveExperiment.cpp` | 调用真实反演 sampler 生成距离样本，支持实验出口截断 |
| `experiments/RenderExperiment.cpp` | 输出新采样器/数值诊断，失败不能写成成功验收 |
| `configs/macrofacet_ci.json` 及新验证配置 | 明确 conditional29 的 regular tracking、Newton 与首段策略 |
| `tests`、`CMakeLists.txt`、文档 | 加入针对实际输运链路的验收与构建项 |

推荐顺序：先实现并验证 Newton/连续距离采样，接入旧首段策略下的 conditional29；再稳定统一条件矩并接入正侧起点；最后完善配置、实验与诊断，进行高预算渲染验收。旧首段策略的中间阶段用于隔离算法差异，最终验收覆盖两种首段模型。

## 10. 验收标准

| 层面 | 验证方法与通过条件 |
| --- | --- |
| 求根基础 | 常数 hazard 和线性非负 hazard 的解析反演；覆盖零 hazard、平台区间、根靠近出口、导数为零、越界 Newton 和迭代预算耗尽 |
| 无漏峰积分 | 窄峰、多峰和近起点交叉案例，用独立高密度/高精度参考检验，不仅依赖 GK 自报误差 |
| 条件统计 | 通用高斯条件化与 SE 显式公式在非退化区间一致；覆盖平面/球面、旋转各向异性、$q\to0$、$q\gg1$、掠射与 PSD |
| 标量/三维一致 | 条件梯度投影均值/方差等于 hazard 使用的 $\mu,s^2$；投影面积等于 $B$ |
| 距离与逃逸 | 实际采出的距离经验 CDF 对照 $1-e^{-A(t)}$；独立验证 $A(t)-E$；逃逸频率匹配 $e^{-A(L)}$ |
| 连续性 | 改初始分块数量不应产生系统性图像变化；收紧容差后距离分布和图像统计收敛 |
| 梯度分布 | 校验 $d^Tg<0$、反射后 $v^Tg>0$，一维负通量 CDF 与横向条件矩，而非只查符号 |
| NDF/相函数 | 数值验证投影 NDF 积分等于 $B$、$q_R$ 球面积分为 1、单位 Fresnel 下散射核积分为 1 |
| 正侧初始化 | $F_0>0$、截断高斯统计与梯度 PSD 统计正确；允许负起始斜率；验证首次与后续碰撞切换 |
| 配置接线 | 证明 conditional29 渲染确实调用 regular sampler；未知选项报错；resolved config 与实际行为一致 |
| 白炉 | 单位环境、单位 Fresnel 下按独立样本置信区间检查均值为 1，数值失败和紧急截断均为 0 |
| 场景回归 | 平面、球面、各向异性、掠射案例；保持 Classic/Midpoint 本轮回归行为 |
| 线程回归 | 固定 seed 在 1 线程与多线程下像素一致；不要求改算法前后逐像素相等 |

现有 `FlightCurveExperiment` 仅把指数目标放进累计光学厚度表对应的 bin，未实际生成格内碰撞距离；该直方图只能验证 bin 质量，不能发现格内线性距离采样误差。必须用 `sampleRegularFlight` / `invertOpticalDepth` 的真实距离样本替换这项验收。

`gp-reference` 继续作为另一个模型的参考，不要求 conditional29 与整段真实 GP 首穿分布完全相等。文档仍保留“以当前点正侧代替整段存活条件”的近似；本方案消除表格距离近似并控制数值误差，不声称消除模型近似或证明全路径互易性。

本轮基线：已运行现有构建的 `ctest --test-dir build -C Release --output-on-failure`，1/1 通过；这是现有测试结果，不是上述新方案已实现或已通过验证的证明。

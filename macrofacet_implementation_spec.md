# Macrofacet 原文复现与局部相关自由程实验：Codex 实现规格

版本：2026-09-18  
交付对象：接收本 Markdown、用户代码仓库及对应 Macrofacet paper 的编程 Codex。  
语言约定：正文中文，接口和伪代码使用 C++ 风格。  
本文是实现规格，不是已经实现、编译或验证过的代码。

## 0. 给实现者的任务与范围

请先阅读仓库的 AGENTS.md、构建入口、MathUtility、向量矩阵类型、随机数、介质、相函数与积分器接口。优先复用已有实现，把本文的逻辑文件映射到现有目录；不要为了匹配目录名重构整个项目。

按 A → B → C 的顺序完成：

- A：复现 Macrofacet 的 Gaussian 高度密度、Beckmann/Generalized Gaussian NDF、方向消光、conductor 散射与多次散射。
- B：保留最近真实碰撞的函数值和完整梯度，以原文式 (29) 的条件 Gaussian crossing/exterior 比率作为 hazard。
- C：在 B 的统计上增加一个中点正值约束，形成 midpoint hazard。
- 提供固定单段实验、条件 GP 数值参考、三模式渲染开关与验收结果。
- GGX heightfield 是 A 的兼容性扩展；不能直接当成 Gaussian GP 接入 B/C。

这里的“复现原文”首先指复现原文的数学模型与关键验证；没有论文原始资产、相机或代码时，不得声称逐像素复刻所有论文图片或复现作者运行时间。使用本文的解析场景完成可重复验证，并在报告中说明差异。

实现要求：

1. 不要把公式 B/C 宣称为原 GPSS 全部相关性的精确解。
2. 不要把只修改距离而不修改碰撞梯度的版本称为完整相关模型；它只能是显式命名的消融实验。
3. 不要把 TODO、返回常量或空实现视为完成。
4. 每阶段运行相应验收，修复真实问题后继续下一阶段，不需要逐阶段向用户请求确认。
5. 遇到仓库接口差异，使用适配层；只有确实缺少决定性输入时才报告阻塞。
6. 第一版数学参考实现使用 CPU/double。GPU 优化、查表、神经拟合、大规模通用 GP 求解器不属于前置要求。
7. 以下头文件可按仓库惯例拆成 .h/.cpp；职责、接口与验收保持不变。模板实现可保留在头文件。
8. 所有近似、误差容限、支持域截断、外部起点策略都必须进入配置和实验报告。

## 1. 原文、实验公式与符号

### 1.1 资料优先级

主要依据用户随附的《Macrofacet Theory for Gaussian Process Statistical Surfaces》。当前核对的附件为 macrofacet(3).pdf，15 页，文内为 August 2026 版本。公开版本可能编号不同，按公式内容识别：

- 式 (7)、(8)：conductor 散射与 VNDF。
- 式 (9)、(10)：SE covariance 与 value–gradient 联合统计。
- 式 (13)、(31)：密度与局部消光。
- 式 (29)：保留起点条件的 crossing/exterior 比率。
- 式 (30)：丢弃起点条件的独立近似。
- 式 (36)–(38)：梯度 PDF 到面积 NDF。
- 式 (40)：Beckmann VNDF 与均匀半球混合提议。
- §5：作者用 PBRT 和 null-scattering 渲染；不是要求必须移植 PBRT。

以下是我们增加的实验闭合，不是论文原有结论：

- 把式 (29) 当作年龄/起点状态相关 hazard，并积分得到自己的生存函数。
- 在式 (29) 的分子、分母同时加入一个中点正值约束。
- B/C 的完整碰撞梯度采样、有限检查点近似与一致的自由程生成方式。

外部参考：

- [Macrofacet Theory for Gaussian Process Statistical Surfaces](https://arxiv.org/abs/2603.00280)
- [GPIS 原文（Seyb 等，2024）](https://cs.dartmouth.edu/~wjarosz/publications/seyb24from-small.pdf)
- [A Radiative Transfer Framework for Spatially-Correlated Materials](https://arxiv.org/abs/1805.02651)
- [Effective computations of joint excursion times for stationary Gaussian processes](https://arxiv.org/abs/2007.14220)
- [SciPy 多维 Gaussian 数值参考接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html)

不要机械照抄排版歧义：

- covariance 的非对角条目必须使用 $κ(x,y)$，不能把方差表达式 $κ(x,x)$ 用于所有 $x,y$。
- 式 (29) 若对标量 k 积分，要先边缘化梯度的两个横向分量；不能把三维梯度 PDF 当成二维 $(F,K)$ PDF。
- 有向面积恒等式应实现为 $ ∫ n D(n) dΩ = E[G] $。原文相关式子的右侧变量记号若有歧义，以这个恒等式验证。

### 1.2 全项目方向、测度与单位

- $x$：世界空间点；长度单位统一。
- $w$：光线实际前进方向，长度为 1。
- $r(t)=x_0+t*w$；t 为从最近真实起点算起的几何距离，不是光学深度。
- $F(x)>0$：外部；F(x)<0：内部；随机 realization 一般不是严格 SDF。
- $μ(x)$：GP 均值，可取基础物体 SDF。
- $G=∇F$：完整梯度；$n=\frac{G}{|G|}$：表面法线。
- $K=w·G$：沿光线导数；从外部穿入表面时 $K<0$。
- 表面反射后 $w_out=w-2(w·n)n$；应满足 $w_out·n>0$。
- $σ$：GP 函数值标准差，不是方差。
- $ell_j$：相关长度；$α_j=√2 σ/ell_j$。
- $G$ 的协方差是 $σ²A$，其中 A 是相关长度的逆平方矩阵。
- $ρ$ 与 $hazard$ 的单位为 1/长度；投影面积因子 $Aproj$ 无量纲。
- NDF $D$ 是面积密度，通常不是积分为 1 的法线概率密度。
- VNDF $ψ$ 是命中法线的归一化概率密度。
- phaseValue 是含 Fresnel 的方向能量核；phasePdf 是实际方向采样 PDF，两者不能混用。
- 用 Aproj 表示投影面积，用 kernelPrecision 表示相关长度矩阵，避免都命名为 A。

统一单变量 Gaussian API：参数使用 mean 和 stddev。多元 Gaussian 的 covariance 存方差/协方差。

### 1.3 三种模型

设 $Z={F(x_0)=0,G(x_0)=g_0}$，只在 Surface 起点使用该等式条件。$w·g0$ 必须为正。

原文 A：
~~~text
rho(x) = phi(mu(x)/sigma) / (sigma * Phi(mu(x)/sigma))
G(x) ~ Normal(gradMu(x), sigma^2 * kernelPrecision)
Aproj(x,w) = E[max(-w·G,0)]
h_A(t) = rho(r(t)) * Aproj(r(t),w)
~~~

B / Eq29：
~~~text
(F_t,K_t) | Z is a bivariate Gaussian
U(t) = P(F_t>0 | Z)
J0(t) = p(F_t=0 | Z) * E[max(-K_t,0) | F_t=0,Z]
h_B(t) = J0(t)/U(t)
~~~
这里 p(F_t=0) 表示在 0 处的密度，不是等式事件的概率。

C / Midpoint：
~~~text
Y = F_(t/2)
U1(t) = P(Y>0,F_t>0 | Z)
J1(t) = p(F_t=0 | Z)
        * E[max(-K_t,0) * indicator(Y>0) | F_t=0,Z]
h_C(t) = J1(t)/U1(t)
~~~

三模式均由实际使用的 hazard 定义：
~~~text
tau(t) = integral_0^t h(s) ds
T_model(t) = exp(-tau(t))
p_model(t) = h(t) * T_model(t)
~~~

禁止把 U、U1、J0、J1 命名或返回为该近似模型的 transmittance/freeFlightPdf。有限检查点下，通常 J1 != -dU1/dt。

### 1.4 用户要求的第二个公式：整段首次相交 hazard

上述 C 是为了低成本实现而明确增加的中点近似；用户讨论的完整目标仍然是：

$$
A_t=\{F(x_0+s\,w)>0,\ \forall\,0<s<t\},
$$
$$
T_{\mathrm{first}}(t\mid Z)=P(A_t\mid Z),
$$
$$
\pi_t(f,k\mid Z)=p(F_t=f,K_t=k\mid Z),
\quad
V_t(k)=P(A_t\mid F_t=0,K_t=k,Z),
$$
$$
\boxed{
h_{\mathrm{first}}(t\mid Z)=
\frac{\displaystyle\int_{-\infty}^{0}(-k)\,
\pi_t(0,k\mid Z)\,V_t(k)\,dk}
{\displaystyle P(A_t\mid Z)}.
}
$$

分母是整段生存概率；不是乘上 $P(A_t)$。$π$ 是密度，$V$ 和 $T$ 是概率。在光滑、普通穿越、起点向外离开的条件下，该式是首次相交的准确表达；只保留最近交点 $Z$ 而丢弃更早路径，本身已是跨 bounce 的状态压缩。

本规格必须提供三个互相区分的结果：

1. **B：**式 (29) 的单点比率，成本低，不计算整段生存事件。
2. **C：**把 $A_t$ 用一个中点条件近似后构造的 hazard，是新的可计算闭合。
3. **F27 数值参考：**用逐步加密的整段网格，直接采样首次穿越，并用多检查点版本评估上述积分公式。它用于比较 B/C 与真正首次相交目标，不能省略后声称已经实验了第二个完整公式。

如果暂时只实现 B 与数值参考，也可以完成“比较两个公式”的第一轮实验；C 是随后研究低成本局部相关模型的具体候选。交付完整规格时仍实现 C，但报告必须保留这层区别。

对固定 $t$，在开区间内取检查点 $Y=(F_{s_1},...,F_{s_N})$。完整公式的有限网格近似为：

$$
U_N(t)=P(Y>0,F_t>0\mid Z),
$$
$$
J_N(t)=p(F_t=0\mid Z)\,
E[(-K_t)_+\mathbf1_{\{Y>0\}}\mid F_t=0,Z],
\quad h_N(t)=J_N(t)/U_N(t).
$$

$N=0$ 给出 B，$N=1$ 且 $s₁=t/2$ 给出 C。多检查点的 Gaussian 积分/Monte Carlo 在 F27 中实现。连续事件的极限、有限网格的漏检误差和估计方差分别报告；不能把有限 $N$ 的 $J_N$ 直接认作 $−U_N$'。

## 2. 文件清单与阶段

路径为逻辑建议，可映射到仓库。每个文件的职责与伪代码在后文定义。

| ID | 文件 | 阶段 |
|---|---|---|
| F01 | mathutility/NumericPolicy.h | A |
| F02 | mathutility/Quadrature.h | A参考/B/C |
| F03 | mathutility/RootFinding.h | A参考/B/C |
| F04 | mathutility/Gaussian1D.h | A |
| F05 | mathutility/GaussianMoments1D.h | A/B |
| F06 | mathutility/SmallGaussian.h | A参考/B |
| F07 | mathutility/BivariateGaussian.h | C |
| F08 | mathutility/GaussianScreenIntegral.h | C |
| F09 | gpss/MeanField.h | A |
| F10 | gpss/SquaredExponentialKernel.h | B，A可复用参数 |
| F11 | gpss/GPSSField.h | A |
| F12 | gpss/ConditionedRay.h | B/C |
| F13 | macrofacet/GaussianNdf.h | A |
| F14 | macrofacet/GgxHeightfield.h | A扩展 |
| F15 | macrofacet/ClassicCoefficients.h | A |
| F16 | macrofacet/ConductorPhase.h | A |
| F17 | transport/FlightState.h | A/B |
| F18 | transport/FlightKernel.h | A/B/C |
| F19 | transport/ClassicFlightKernel.h | A |
| F20 | transport/Conditional29FlightKernel.h | B |
| F21 | transport/MidpointFlightKernel.h | C |
| F22 | transport/OpticalDepthSampler.h | A参考/B/C |
| F23 | transport/ClassicNullTracking.h | A |
| F24 | transport/CollisionGradientSampler.h | B/C，A交叉验证 |
| F25 | integrator/MacrofacetPathTracer.cpp | A/B/C |
| F26 | experiments/FlightCurveExperiment.cpp | B/C |
| F27 | experiments/ConditionalGPReference.cpp | B/C参考 |
| F28 | experiments/RenderExperiment.cpp | A/B/C |
| F29 | experiments/ExperimentMain.cpp | 全部 |
| F30 | tests/test_gaussian_math.cpp | A/B/C |
| F31 | tests/test_gpss_statistics.cpp | B/C |
| F32 | tests/test_macrofacet_baseline.cpp | A |
| F33 | tests/test_flight_kernels.cpp | B/C |
| F34 | tests/test_sampling.cpp | A/B/C |
| F35 | configs/macrofacet_experiments.json | 全部 |
| F36 | CMakeLists.txt 或现有构建目标的增量修改 | 全部 |
| F37 | docs/IMPLEMENTATION_REPORT.md | 每阶段更新 |

## 3. F01：mathutility/NumericPolicy.h

职责：数值容限、结果状态、诊断。不得把无效计算静默变成“无碰撞”。

接口：
~~~cpp
enum class NumericStatus {
    Ok, ExactZero, Degenerate, UnderflowValueWithFiniteLog,
    NeedHigherPrecision, InvalidCovariance, InvalidInput,
    IntegrationNotConverged, RootNotBracketed
};

struct PositiveResult {
    double value;       // 可下溢为0，但logValue仍应保留
    double logValue;    // ExactZero 才是真正 -infinity
    double absError;
    NumericStatus status;
};

struct NumericPolicy {
    double relativeTolerance = 1e-7;
    double absoluteTolerance = 1e-10;
    int maxQuadratureSubdivisions = 4096;
    int maxRootIterations = 128;
    double covarianceRoundoffMultiplier = 128;
    bool allowHigherPrecisionFallback = true;
};
~~~

实现规则：

- 在接近 0 的目标上用绝对误差，在通常尺度上用相对误差；容限可配置。
- 数学上的负方差/负概率是错误；只有已证明处于舍入误差范围内的负值才可归零。
- 协方差容限按变量尺度标准化后判断，不用一个全局绝对 epsilon 混合函数值和梯度单位。
- 不使用 max(CDF,1e-6) 等修改物理统计的截断。
- 不在协方差上无条件加 jitter；如参考实验明确使用正则化，必须记录它改变了哪个模型及幅度。
- 到达积分/求根预算时返回诊断或重试更高精度，不能返回伪造的成功结果。
- 统计 t=0 的确定性极限与 t>0 的 Gaussian 计算分开。

伪代码：
~~~text
validateNonnegative(x, scale):
    if x >= 0: return x
    if abs(x) <= roundoffBound(scale): return 0 with diagnostic
    fail InvalidInput

openUniform01(rng):
    repeat u = rng.uniform01()
    until 0 < u < 1
    return u
~~~

验收：所有上层函数能传播状态；真正的零与浮点下溢在日志中可区分。

## 4. F02：mathutility/Quadrature.h

职责：确定性一维数值积分及误差估计。优先包装仓库已验证的自适应 Gauss–Kronrod 实现；没有时增加一个明确的数值后端。不能把固定少量采样点之和称作自适应积分。

接口：
~~~cpp
IntegralResult integrateFinite(f, lo, hi, policy);
IntegralResult integrateSemiInfinite(f, lower, policy);
PositiveResult integratePositiveLog(
    logf, domain, modeOrScaleHint, policy);
~~~

有限区间伪代码：
~~~text
adaptiveIntegrate(f,a,b):
    estimate both Gauss and Kronrod sums on [a,b]
    error = conservative difference/error estimate from backend
    put interval into priority queue by error
    totalValue = estimate; totalError = error
    while totalError exceeds max(absTol, relTol*abs(totalValue)):
        split interval with largest error at midpoint
        evaluate both children
        replace parent's value/error by children's totals
        if budget exhausted: IntegrationNotConverged
    return totalValue,totalError
~~~

半无限区间可用 x=lower+u/(1-u)，u∈(0,1)，Jacobian=1/(1-u)^2。负半无限区间镜像处理。积分节点不取奇异端点。

log 正值积分：
~~~text
integratePositiveLog(logf, domain, scaleHint):
    choose a finite logScale near max(logf) on domain
    integrate exp(logf(x)-logScale), including transformation Jacobian
    if scaled integral is unresolved/underflows:
        improve scale/domain transform or use higher precision
    return logValue = logScale + log(scaledIntegral)
~~~

对本文的 Gaussian×多项式×Gaussian CDF 积分，可利用 log integrand 的单峰/凹性寻找尺度。不要固定 logScale=0 后让整个尾部下溢。

验收：一维 Gaussian 质量、矩、已知多项式积分；检查误差估计随容限收紧而收敛。

## 5. F03：mathutility/RootFinding.h

职责：单调函数的有界反演；服务 Gaussian quantile、crossing-weighted Gaussian 采样和累计 optical depth 反演。

接口：
~~~cpp
RootResult solveMonotoneIncreasing(f, target, lo, hi, policy);
RootResult expandAndSolveNegativeDomain(f, target, upper, policy);
~~~

保证正确性的最小版本用二分；可用受区间保护的 Brent/Newton 加速。

~~~text
solveMonotoneIncreasing(f,target,lo,hi):
    require f(lo) <= target <= f(hi)
    loop:
        x = safeguardedInterpolationOrMidpoint(lo,hi)
        y = f(x)
        if converged in x or residual with known scale: return x
        if y < target: lo=x
        else: hi=x
    fail if iteration budget exhausted
~~~

验收：单调函数不会越界；CDF 反演不返回支持域外值；不把未 bracket 到的样本当成逃逸，逃逸只由真实边界累计 optical depth 判定。

## 6. F04：mathutility/Gaussian1D.h

职责：标准正态基础函数。使用标准库 erf/erfc 或已有可靠数值库，第一版不要求自行实现这些特殊函数。

接口：
~~~cpp
double normalPdf(double z);
double normalLogPdf(double z);
double normalCdf(double z);
double normalLogCdf(double z);
double normalSurvival(double z);
double normalLogSurvival(double z);
double normalPdfOverCdf(double z);
double normalQuantile(double u);
double normalQuantileFromLogCdf(double logU);
double sampleStandardNormal(Random& rng);
~~~

伪代码：
~~~text
normalLogPdf(z) = -0.5*z*z - 0.5*log(2*pi)
normalPdf(z) = exp(normalLogPdf(z))
normalCdf(z) = 0.5*erfc(-z/sqrt(2))
normalSurvival(z) = normalCdf(-z)
normalLogSurvival(z) = normalLogCdf(-z)

normalLogCdf(z):
    if z > 0:
        return log1p(-0.5*erfc(z/sqrt(2)))
    if ordinary erfc is accurate and nonzero:
        return log(0.5*erfc(-z/sqrt(2)))
    return vetted lower-tail log-CDF implementation
           or asymptotic expansion with truncation/error control

normalPdfOverCdf(z):
    evaluate exp(normalLogPdf(z)-normalLogCdf(z))
    use a vetted asymptotic/extended-precision path if subtraction
    of huge quadratic terms loses accuracy

normalQuantileFromLogCdf(logU):
    require logU < 0
    use vetted inverse-CDF implementation where representable
    otherwise bracket z with normalLogCdf(z) around logU
    solve monotonically in log probability

sampleStandardNormal(rng):
    return normalQuantile(openUniform01(rng))
    // Box–Muller or the host implementation is also acceptable
~~~

负尾 log Φ 的参考展开，令 x=-z>0：
~~~text
logPhi(-x) =
    -x*x/2 - log(sqrt(2*pi)) - log(x)
    + log(1 - 1/x^2 + 3/x^4 - 15/x^6 + ...)
~~~
这是渐近展开，不是任意 z 都可使用的收敛级数。只有进入安全负尾区间后使用，并在项开始增大或达到误差目标时停止。优先使用成熟后端。

验收：

- Φ(0)=1/2，φ(0)=1/sqrt(2π)，Φ(z)+Φ(-z)=1。
- 在 z∈{-40,-20,-8,-2,0,2,8,20,40} 检查 log 概率；小概率不得被 epsilon 替代。
- r(z)=φ/Φ 在 z→-∞ 时约为 -z，有限且非负。
- log-CDF 与 quantile 回代匹配，包含低尾概率。
- 所有采样的 u 严格在 (0,1)。


## 7. F05 — mathutility/GaussianMoments1D.h

**职责：**计算截断高斯矩和 crossing 所需的负部一阶矩，并采样按 crossing 速度加权的方向导数。它是三个模型共同的核心。

设 X∼N(μ,s²)，s>0，a=μ/s。定义：

$$
M_r^+(\mu,s)=E[X^r\mathbf1_{\{X>0\}}].
$$

$$
\begin{aligned}
M_0^+&=\Phi(a),\\
M_1^+&=\mu\Phi(a)+s\phi(a),\\
M_2^+&=(\mu^2+s^2)\Phi(a)+\mu s\phi(a),\\
M_3^+&=(\mu^3+3\mu s^2)\Phi(a)+(\mu^2s+2s^3)\phi(a).
\end{aligned}
$$

$$
E[(-X)_+]=M_1^+(-\mu,s).
$$

**API：**
~~~cpp
PositiveResult positiveRawMoment(int order, double mean, double stddev);
PositiveResult negativePartMean(double mean, double stddev);
double negativeFluxNormalLogPdf(double k, double mean, double stddev);
PositiveResult negativeFluxNormalCdf(double k, double mean, double stddev);
double sampleNegativeFluxNormal(double mean, double stddev, RNG &rng);
~~~

**算法：**
~~~text
positiveRawMoment(r, mu, s):
    require r in {0,1,2,3}, s >= 0
    if s == 0:
        return mu^r if mu > 0 else 0
        // for r=0 use the indicator, not pow(0,0)
    if r == 0:
        return normalCdf(mu/s) together with normalLogCdf(mu/s)
    a = mu / s
    if closed form is numerically well conditioned:
        evaluate the corresponding formula
        return both value and logValue
    otherwise:
        // a very negative tail causes cancellation in closed forms
        logIntegrand(u) = r*log(u) - (u-a)^2/2 - log(sqrt(2*pi))
        choose a positive-integrand mode:
            r > 0, a >= 0: uMode = (a + sqrt(a*a + 4*r))/2
            r > 0, a < 0:  uMode = 2*r/(sqrt(a*a + 4*r) - a)
            r == 0:       uMode = max(a, 0)
        compute I = integral over u>0 of exp(logIntegrand(u))
                    with scaled log quadrature
        return logMoment = r*log(s) + log(I)
~~~

模式计算中的平方根可用 hypot 等方法避免极端参数平方溢出。禁止通过 max(moment,epsilon) 掩盖相消。

需要采样的 crossing 分布是：

$$
q_K(k)=
\frac{(-k)\,\mathcal N(k;\mu,s^2)}
{M_1^+(-\mu,s)}\mathbf1_{\{k<0\}}.
$$

对于 k≤0，它的 CDF 为：

$$
R(k)=
\frac{s\phi(z)-\mu\Phi(z)}
{M_1^+(-\mu,s)},\qquad z=\frac{k-\mu}{s}.
$$

~~~text
sampleNegativeFluxNormal(mu, s, rng):
    if s == 0:
        if mu < 0: return mu
        otherwise: return NoCrossingMass
    u = openUniform01(rng)
    upper = 0
    expand a finite lower bound until R(lower) < u
    return monotoneRoot(R(k) - u, lower, upper)

negativeFluxNormalCdf(k, mu, s):
    if k >= 0: return 1
    evaluate numerator = s*phi(z) - mu*Phi(z)
    if cancellation or underflow makes direct evaluation unreliable:
        evaluate integral[-infinity,k] (-v)*Normal(v;mu,s^2) dv
        using positive log quadrature
    divide numerator and denominator in log space
~~~

不得把无界的 $(-k)$ 当成普通 rejection probability。$s=0$ 且 $μ≥0$ 时没有负向 crossing，不能“随便返回一个负 k”。

**验收：**矩公式与独立数值积分一致；CDF 单调且端点为 0、1；采样的经验 CDF 与 R 一致；大正 μ 的罕见负尾仍有有效 log 值。

## 8. F06 — mathutility/SmallGaussian.h

**职责：**统一小型联合高斯、线性映射、条件化和退化协方差处理。支持维度 1–8；参考 GP 网格可以复用相同原则的动态矩阵版本。

~~~cpp
template<int N> struct Gaussian {
    Vector<N> mean;
    Matrix<N,N> covariance;
};
template<int M, int N>
Gaussian<M> linearMap(const Gaussian<N>& g, const Matrix<M,N>& A,
                      const Vector<M>& offset);

Gaussian<A> conditionGaussian(
    Gaussian<A> target, Gaussian<B> observation,
    Matrix<A,B> targetObservationCovariance, Vector<B> observed);

Vector<N> sampleGaussianPSD(const Gaussian<N>& g, RNG& rng);
double logPdfGaussianSPD(const Gaussian<N>& g, Vector<N> value);
~~~

~~~text
conditionGaussian(target, observation, Cab, b):
    standardize observation dimensions by their natural scales
    remove deterministic observation components after consistency checks
    factor Cbb; prefer Cholesky for SPD
    for rank-deficient Cbb:
        identify active eigenspace using a scale-aware tolerance
        verify b-mub lies in its supported range
        solve only in that range
    alpha = solve(Cbb, b-mub)
    W = solve(Cbb, transpose(Cab))
    mean = mua + Cab * alpha
    cov = Caa - Cab * W
    symmetrize cov
    validate PSD with scale-aware tolerances
    remove only negative eigenvalues explainable by floating-point roundoff
    if accuracy is inadequate:
        retry stable analytic form or higher precision, otherwise report failure
    return Gaussian(mean,cov)

sampleGaussianPSD(g, rng):
    factor covariance in its active subspace
    draw independent standard normals only in that subspace
    return mean + factor * normals
~~~

要求：

- 不显式求矩阵逆；不通过固定 jitter 改变几何模型。
- 观测包含 F 和梯度，二者量纲可能不同，应先标准化再判断条件数。
- 零方差观测必须检查一致性；不能忽略一个不可能的观测。
- 二维平面上的退化高斯没有普通三维 Lebesgue PDF。采样可以支持 PSD，普通 log-PDF 接口只接收 SPD。
- 接近起点时的条件协方差相消，优先使用 F12 的稳定形式；禁止靠抬高方差“修复”。
- 更高精度回退可将这些小矩阵计算模板化为 long double 或宿主已有多精度类型；long double 仍不足时使用例如 Boost.Multiprecision 的高精度浮点。报告实际启用的后端，不能只放置一个无作用的配置开关。

**验收：**二维高斯的闭式条件均值/方差；线性映射后的样本矩；重复等价观测；不一致的确定观测；退化但合法的二维子空间采样。

## 9. F07 — mathutility/BivariateGaussian.h

**职责：**计算两个联合高斯变量同时为正的概率及其 log 值。中点模型的分母必须调用这里。

~~~cpp
PositiveResult standardBivariateNormalCdf(double a, double b, double correlation);
PositiveResult positiveOrthant2(Vector2 mean, Matrix2 covariance);
~~~

若两个方差均正，令 $a=μ₁/s₁，b=μ₂/s₂$，则：

$$
P(X_1>0,X_2>0)=\Phi_2(a,b;\rho).
$$

这是联合中心对称性的结果，不需要写成易相消的 $1−Φ−Φ+Φ₂$。

~~~text
positiveOrthant2(mean, cov):
    handle deterministic components first:
        deterministic value <= 0 -> exact zero
        deterministic value > 0  -> remove that condition
    compute standardized thresholds and correlation
    return standardBivariateNormalCdf(a,b,rho)

standardBivariateNormalCdf(a,b,rho):
    require -1 <= rho <= 1 within roundoff tolerance
    if rho == 0:  return Phi(a)*Phi(b), with log values added
    if rho == 1:  return Phi(min(a,b))
    if rho == -1: return max(Phi(a)+Phi(b)-1,0), evaluated stably
    use a vetted deterministic bivariate-normal algorithm if available
    otherwise evaluate:
        integral[-infinity,a] phi(x)
             * Phi((b-rho*x)/sqrt(1-rho*rho)) dx
        in scaled log space
    return value, logValue, estimated integration error
~~~

负相关接近 −1 时，$Φ(a)+Φ(b)−1$ 可改写成两个 CDF 的差，并用 log-difference 计算。内部积分用 logφ+logΦ；缩放点可由一维模式搜索获得。尾部失败必须提高精度或返回失败，不得假装独立。

**验收：**ρ=0、±1 的极限；交换两个变量不变；与高精度独立后端对照；具有极小联合概率的例子。

## 10. F08 — mathutility/GaussianScreenIntegral.h

**职责：**提供中点筛选后 crossing 通量的一维积分。

~~~cpp
PositiveResult negativeFluxTimesCdf(
    double meanK, double stddevK, double a, double b);
PositiveResult negativeFluxTimesAffineIndicator(
    double meanK, double stddevK, double intercept, double slope);
~~~

第一接口定义：

$$
I=E[(-K)_+\Phi(a+bK)],\qquad K\sim N(\mu_K,s_K^2).
$$

~~~text
negativeFluxTimesCdf(mu, s, a, b):
    if s == 0: return max(-mu,0) * Phi(a+b*mu)
    if b == 0: return negativePartMean(mu,s) * Phi(a)
    substitute k = -s*u:
        I = s * integral[0,infinity]
                    u * phi(u+mu/s) * Phi(a-b*s*u) du
    evaluate its nonnegative integrand in log space:
        log(s) + log(u) + logphi(u+mu/s) + logPhi(a-b*s*u)
    find a scale near the integrand mode; use F02
    report numerical error and status
~~~

第二接口定义：

$$
E[(-K)_+\mathbf1_{\{a+bK>0\}}].
$$

它用于中点在给定 K 后方差恰好为零的情况。先求 $k<0$ 与 $a+b_k>0$ 的交区间，再计算该区间上的负部一阶矩。区间矩可以用 F05 的累积通量之差；接近相等时改用正积分，避免相消。

**验收：**$0≤I≤E[(-K)₊]；b=0$ 的乘积恒等式；a→±∞ 的极限；与三维联合高斯直接 Monte Carlo 统计对照。不要把数值结果大幅裁剪到合法区间来掩盖积分错误。

## 11. F09 — gpss/MeanField.h

**职责：**只描述基础均值场及其解析梯度，不负责采样 GP。

~~~cpp
struct MeanJet {
    double value;
    Vector3 gradient;
};
class MeanField {
public:
    virtual MeanJet evaluate(Point3 x) const = 0;
    virtual BoundsSummary bounds(const Bounds3& domain) const = 0;
};
~~~

~~~text
PlaneMean.evaluate(x):
    require unit normal n
    return { dot(n,x)-offset, n }

SphereMean.evaluate(x):
    r = x-center
    if length(r) == 0: report NondifferentiableMeanPoint
    return { length(r)-radius, r/length(r) }

ConstantMean.evaluate(x):
    return { constantValue, (0,0,0) }
~~~

初始实验用平面。球心是球 SDF 的不可微点，应明确处理，不得以归一化零向量得到 NaN。球体的体积分中单点不可微通常是零测度问题，但接口必须有确定行为；不能让已知球心成为 GP 梯度条件观测点。

bounds 提供定义域内 $μ$ 的下界和 $||∇μ||$ 的上界，供原模型 majorant 使用。若用户自定义均值场没有可靠的 bounds，使用光学深度积分采样，不可凭稀疏采样构造声称严格的 majorant。

## 12. F10 — gpss/SquaredExponentialKernel.h

**职责：**实现固定各向异性 squared-exponential 核及函数值/梯度交叉协方差。

$$
A=R\,\mathrm{diag}(\ell_x^{-2},\ell_y^{-2},\ell_z^{-2})R^T,\quad
\delta=x-y,\quad v=A\delta,
$$
$$
\kappa(x,y)=\sigma^2e^{-\delta^TA\delta/2}.
$$

$$
\begin{aligned}
\nabla_x\kappa&=-v\kappa,\\
\nabla_y\kappa&=v\kappa,\\
\nabla_x\nabla_y^T\kappa&=(A-vv^T)\kappa.
\end{aligned}
$$

~~~cpp
struct KernelJet {
    double valueValue;
    Vector3 gradientXValueY;
    Vector3 valueXGradientY;
    Matrix3 gradientXGradientY;
};
class SquaredExponentialKernel {
    double sigma;
    Matrix3 precision; // A; not gradient covariance
public:
    KernelJet evaluate(Point3 x, Point3 y) const;
};
~~~

~~~text
evaluate(x,y):
    delta = x-y
    v = precision*delta
    k = sigma*sigma*exp(-0.5*dot(delta,v))
    return { k, -v*k, v*k, (precision-outer(v,v))*k }
~~~

要求：

- 支持某个 ℓ=∞，其 precision 为零，用于 heightfield 极限；这会产生合法退化协方差。
- 核参数和 R 在一次实验的随机场内固定。
- 不可把随位置变化的局部坐标系或 ℓ 直接代入平稳核并假定仍然正定。非平稳核是后续独立扩展。
- Gaussian NDF 的局部 α 与本核的关系是$ αᵢ=√2σ/ℓᵢ$，不是 $σ/ℓᵢ$。
- 同一点 $Cov(F,G)=0$，但不同点的函数值与梯度一般相关。

**验收：**$x/y$ 交换后的块转置关系；有限差分检查导数符号；随机位置和观测组合组成的协方差矩阵 PSD。

## 13. F11 — gpss/GPSSField.h

**职责：**组合 MeanField、核、材料和有限渲染域，提供无条件局部统计。

~~~cpp
struct PointPrior {
    double meanF, varianceF;
    Vector3 meanG;
    Matrix3 covarianceG;
    Vector3 covarianceFG;
};
struct GPSSField {
    MeanField mean;
    SquaredExponentialKernel kernel;
    Bounds3 activeDomain;
    ConductorParameters conductor;
    NdfFamily ndfFamily; // GeneralizedGaussian / BeckmannLimit / GGXBaseline
    PointPrior pointPrior(Point3 x) const;
};
~~~

~~~text
pointPrior(x):
    jet = mean.evaluate(x)
    return {
        jet.value,
        sigma^2,
        jet.gradient,
        sigma^2 * kernel.precision,
        zeroVector
    }
~~~

边界规定：

1. v1 的 B/C 实验使用一个凸的有效域，使每段在域内的传播区间明确。默认采用平面附近的有限 AABB。
2. 有效域之外消光为零，这是显式的有限域材料模型。对于无限 GP，它是截断近似，必须在报告中写出边界和截断距离。
3. 不在盒子边界自动增加一层不透明表面；外部确定几何由场景处理。
4. 若使用平面 ±3σ 的薄层，所有模型和数值参考都采用同一层；不得把域外没有命中解释为无限随机场永远不再命中。
5. 初版不支持在 B/C 中用多段不连通 shell mask 却继续把跨空洞的检查点当作有效表面。需要这项能力时，须显式修改几何事件和屏蔽条件。
6. GGXBaseline 允许独立设置 NDF 参数，但它不提供本 SE 核的高斯梯度。B/C 配置遇到 GGX 必须报错。

**验收：**局部均值/方差与核在 x=y 的导数一致；域裁剪一致；不合法核参数和 B/C+GGX 组合被拒绝。

## 14. F12 — gpss/ConditionedRay.h

**职责：**固定一次真实 bounce 的位置和完整梯度，构造后续射线所需的条件高斯。

$$
Z=\{F(x_0)=0,\ G(x_0)=g_0\},\quad x_t=x_0+t\,w.
$$

缓存四维观测 $O=(F(x₀),G(x₀))$ 的均值、协方差分解和观测值 $(0,g₀)$。$w$ 必须单位化，且外侧反射要求 $w·g₀>0$。

~~~cpp
class ConditionedRay {
public:
    ConditionedRay(const GPSSField&, Point3 x0, Vector3 g0, Vector3 w);
    Gaussian<2> endpointValueSlope(double t) const;       // (F_t,K_t)
    Gaussian<3> midpointValueSlope(double t) const;       // (Y,F_t,K_t)
    Gaussian<4> endpointValueGradient(double t) const;    // (F_t,G_t)
    Gaussian<5> midpointValueGradient(double t) const;    // (Y,F_t,G_t)
    DynamicGaussian valuesAt(const vector<double>& ages) const;
    DynamicGaussian checkpointValuesAndEndpointSlope(
        const vector<double>& interiorAges, double t) const; // (Y_1,...,Y_N,F_t,K_t)
};
~~~

~~~text
constructTarget(descriptors):
    for each target variable:
        get its prior mean from MeanField
    for every target/target and target/observation pair:
        assemble covariance from KernelJet blocks
    apply conditionGaussian(target, originObservation, crossCov, (0,g0))
    obtain K_t by the linear map K_t = dot(w,G_t)
    return the requested ordering of variables
~~~

描述符可为 Value(x) 或 GradientComponent(x,i)。协方差方向必须统一：

~~~text
Cov(F(x),F(y)) = k
Cov(F(x),G(y)) = gradientY(k), as a row
Cov(G(x),F(y)) = gradientX(k), as a column
Cov(G(x),G(y)) = mixedDerivativeXY(k)
~~~

**必须提供近起点的稳定计算。**对平面均值、固定 A，令 $d₀=μ(x₀)、mₖ=w·∇μ、a=wᵀAw、c(t)=exp(−at²/2)$，有：

$$
m_Z(t)=d_0+t m_k+c(t)[-d_0+t(w\cdot g_0-m_k)],
$$

$$
C_Z(s,t)=\sigma^2\left[e^{-a(s-t)^2/2}-c(s)c(t)(1+ast)\right].
$$

其中协方差公式实际不依赖平面均值；平面假设仅用于上面的简单均值表达式。在 a s t 很小时，可稳定写为：

$$
C_Z(s,t)=\sigma^2c(s)c(t)\,[\operatorname{expm1}(ast)-ast].
$$

expm1(r)−r 在极小 r 时也有相消，使用 $r²/2+r³/6+…$ 的受控级数。r 很大时回到原表达式，避免 0×∞。

单点还可用：

$$
\begin{aligned}
q&=at^2,\\
\operatorname{Var}(F_t\mid Z)
  &=\sigma^2[1-e^{-q}(1+q)],\\
\operatorname{Cov}(F_t,K_t\mid Z)
  &=\sigma^2a^2t^3e^{-q},\\
\operatorname{Var}(K_t\mid Z)
  &=\sigma^2a[1-e^{-q}(1-q+q^2)].
\end{aligned}
$$

~~~text
evaluateNearOrigin(t):
    use expm1 and Taylor expansions with a controlled remainder
    or reevaluate the small covariance blocks at higher precision
    preserve exact F(0)=0 and G(0)=g0
    never add a physical variance floor
~~~

起点完整梯度不可替换成单位法线；g₀=n₀ 会额外固定梯度长度并改变条件模型。严格 heightfield 等退化核要求 g₀ 落在先验支持内，否则报不一致观测。

**验收：**$m_Z(0)=0、m'_Z(0)=w·g₀$、起点观测方差为零；与通用条件化在安全距离匹配；$t→0$ 不出现负方差；在 $a>0$ 且 $t≫1/√a$ 时，起点条件对局部统计的影响消失。


## 15. F13 — macrofacet/GaussianNdf.h

**职责：**实现 Generalized Gaussian 面积 NDF、Beckmann 退化极限、投影面积和 VNDF。不要把普通梯度方向分布误当成面积 NDF。

对于 G∼N(μ_G,C_G)，定义：

$$
D(n)=\int_0^\infty r^3p_G(rn)\,dr,\qquad
A_{\mathrm{proj}}(w)=E[(-w\cdot G)_+],
$$
$$
\psi(n\mid w)=\frac{[-w\cdot n]_+D(n)}{A_{\mathrm{proj}}(w)}.
$$

r³ 中 r² 来自球坐标 Jacobian，另一个 r 来自表面积权重。

~~~cpp
class GaussianNdf {
public:
    PositiveResult evaluateD(Vector3 unitNormal) const;
    PositiveResult projectedArea(Vector3 travelDirection) const;
    double visibleNormalPdf(Vector3 unitNormal, Vector3 travelDirection) const;
};
~~~

C_G 正定时，缓存其 Cholesky L，令 $u=L⁻¹μ_G、v=L⁻¹n$：

$$
a=v\cdot v,\quad m=(u\cdot v)/a,\quad s=a^{-1/2},
\quad r_\perp=u-mv.
$$

$$
\log D(n)=
-\log(2\pi)-\tfrac12\log\det C_G-\tfrac12\log a
-\tfrac12\|r_\perp\|^2+\log M_3^+(m,s).
$$

~~~text
evaluateD(n):
    require length(n) == 1 within tolerance
    if exact heightfield parameters: use Beckmann branch below
    otherwise require SPD covariance
    u = triangularSolve(L,meanG)       // cache u
    v = triangularSolve(L,n)
    a = dot(v,v)
    m = dot(u,v)/a
    residual = u-m*v
    return exp/log of the closed form using F05

projectedArea(w):
    meanK = dot(w,meanG)
    varianceK = dot(w,covarianceG*w)
    return negativePartMean(meanK,sqrt(varianceK))

visibleNormalPdf(n,w):
    if dot(w,n) >= 0: return 0
    return (-dot(w,n))*D(n)/projectedArea(w)
~~~

**精确 heightfield 分支：**在均值平面坐标中 $G_z=1$，横向方差 $α_x²/2$、$α_y²/2$。当 $n_z>0$，

$$
D_B(n)=
\frac{\exp[-((n_x/\alpha_x)^2+(n_y/\alpha_y)^2)/n_z^2]}
{\pi\alpha_x\alpha_y n_z^4}.
$$

n_z≤0 时为零。α_x、α_y>0。三个 α 全为零属于 delta 镜面极限，不能放进普通 PDF；第一版配置明确拒绝，或调用宿主已有的确定镜面分支。不得用人为 roughness floor 偷换极限模型。

这条 Beckmann 分支要求在同一坐标系中 $μ_G=(0,0,1)、C_G=diag(α_x²/2,α_y²/2,0)。$默认退化测试使用与 kernel frame 对齐的平面均值。其他退化协方差若没有实现对应面积测度，返回 UnsupportedDegenerateNdf；不能把任意 PSD Gaussian 都套成上述 Beckmann 式。F24 的 PSD 梯度采样能力不等于 F13 已支持所有奇异 NDF 密度。

**验收：**

- $\int nD(n)d\Omega=\mu_G$，而非强制 $\int D=1$。
- 数值球面积分给出的投影面积与 F05 的解析负部矩一致。
- VNDF 的球面积分为 1。
- $α_z→0$ 的非退化结果在适当积分意义下趋于 Beckmann，$α_z=0$ 走专门分支。

## 16. F14 — macrofacet/GgxHeightfield.h

**职责：**原模型 A 的 GGX heightfield 选项。不得声称它由本项目 Gaussian SE 梯度直接导出。

$$
D_{\mathrm{GGX}}(n)=
\frac{\mathbf1_{\{n_z>0\}}}
{\pi\alpha_x\alpha_y
[(n_x/\alpha_x)^2+(n_y/\alpha_y)^2+n_z^2]^2},
$$

$$
A_{\mathrm{GGX}}(w)=
\tfrac12\left[
\sqrt{w_z^2+\alpha_x^2w_x^2+\alpha_y^2w_y^2}-w_z
\right].
$$

~~~text
projectedArea(w):
    q = alphaX^2*w.x^2 + alphaY^2*w.y^2
    root = sqrt(w.z^2+q)
    if w.z > 0:
        return q / (2*(root+w.z))  // stable difference
    return (root-w.z)/2

visibleNormalPdf(n,w):
    return max(-dot(w,n),0)*D_GGX(n)/projectedArea(w)
~~~

普通方向采样可使用宿主已有 GGX VNDF；若没有，均匀朝向入射线的半球提议配合精确权重也是正确的基准实现，效率另行报告。B/C 的配置校验拒绝该 NDF。

**验收：**投影面积的数值积分；归一化 VNDF；朝上光线、掠射光线和向下光线的方向约定。

## 17. F15 — macrofacet/ClassicCoefficients.h

**职责：**原模型的密度、投影面积、方向消光及严格 majorant。

$$
\rho(x)=
\frac{\phi(\mu(x)/\sigma)}
{\sigma\Phi(\mu(x)/\sigma)},\qquad
h_A(x,w)=\rho(x)A_{\mathrm{proj}}(x,w).
$$

~~~cpp
struct ClassicEvaluation {
    PositiveResult density;
    PositiveResult projectedArea;
    PositiveResult extinction;
};
ClassicEvaluation evaluateClassic(const GPSSField&, Point3 x, Vector3 w);
double classicMajorant(const GPSSField&, Bounds3 domain, Vector3 w);
~~~

~~~text
evaluateClassic(field,x,w):
    prior = field.pointPrior(x)
    z = prior.meanF / sigma
    logRho = normalLogPdf(z)-log(sigma)-normalLogCdf(z)
    area = selectedNdf.projectedArea(w)
    logH = logRho + area.logValue
    return both log and ordinary values

classicMajorant(field,domain,w):
    dmin = certified lower bound on mu over domain
    rhoMax = rho(dmin)
    if planar constant gradient and NDF:
        areaMax = exact projectedArea(w)
    else for Gaussian gradient:
        M = certified upper bound on norm(gradMu)
        CmaxTrace = certified upper bound on trace(Cg)
        areaMax = sqrt(M*M+CmaxTrace)
    else:
        use a certified NDF-specific directional bound
    return rhoMax*areaMax with conservative rounding
~~~

Gaussian 上界来自 Aproj≤E||G||≤√E||G||²。没有可靠 bounds 时回退到 F22。运行时若检测到 h>majorant 超出舍入误差，终止该样本并报 majorant 错误；不能简单 min(h/majorant,1)。

**验收：**在域内随机点检查上界作为辅助诊断；上界来源仍需解析或经过证明，随机检查本身不是证明。μ 很负时的 ρ 不应数值溢出或变为零。

## 18. F16 — macrofacet/ConductorPhase.h

**职责：**给原模型 A 提供含 conductor Fresnel 的方向能量核与实际采样 PDF。

对于镜面微表面，w 是入射前进方向，w' 是出射前进方向：

$$
n=\frac{w'-w}{\|w'-w\|},\quad w\cdot n<0,
$$
$$
w'=w-2(w\cdot n)n,\qquad
\left|\frac{d\Omega_n}{d\Omega_{w'}}\right|
=\frac1{4|w\cdot n|}.
$$

因此：

$$
f_{\mathrm{energy}}(w\to w')
=
\frac{F_{\mathrm{cond}}(|w\cdot n|)\,D(n)}
{4A_{\mathrm{proj}}(w)}.
$$

~~~cpp
struct PhaseSample {
    Vector3 direction;
    Spectrum energyValue;
    double samplingPdf;
    Spectrum throughputWeight;
    Vector3 sampledNormal;
};
Spectrum evaluateEnergy(Vector3 w, Vector3 wNew);
double evaluateSamplingPdf(Vector3 w, Vector3 wNew);
PhaseSample samplePhase(Vector3 w, RNG&);
~~~

Fresnel 优先调用宿主已有 conductor 实现；η 与 κ 是相对于外部介质的复折射率参数。新增实现时用复数形式或已验证实数式，并在法向入射检验：

$$
F(1)=\frac{(\eta-1)^2+\kappa^2}{(\eta+1)^2+\kappa^2}.
$$

没有宿主 Fresnel 实现时，以下复数参考算法逐 RGB 通道或逐波长使用：

~~~text
conductorFresnel(cosTheta,eta,kappa):
    c = clampToUnitIntervalForRoundoff(abs(cosTheta))
    require eta >= 0, kappa >= 0 and eta^2+kappa^2 > 0
    N = complex(eta,kappa)
    if N == 1: return 0
    if c == 0: return 1
    gamma = complexSqrt(N*N - (1-c*c))
    choose the passive-medium square-root branch
        // nonnegative real part; principal root for these eta,kappa
    rs = (c-gamma)/(c+gamma)
    rp = (N*N*c-gamma)/(N*N*c+gamma)
    return (squaredComplexMagnitude(rs)+squaredComplexMagnitude(rp))/2
~~~

允许使用更快的等价实数公式，但需与此复数参考在掠射和法向情况下对照。

**提议分布：**实现正确的均匀半球模式，并支持论文式 (40) 的混合模式：

$$
q_n(n)=R\,q_{\mathrm{BeckmannVNDF}}(n)
 +(1-R)\frac{\mathbf1_{\{-w\cdot n>0\}}}{2\pi}.
$$

R 是配置参数，本文示例 0.5 是实现选择，不是声称原文指定值。Generalized Gaussian 在下半球也可能有质量，因此保持 R<1，保证均匀分量覆盖全部可见支持。

~~~text
samplePhase(w,rng):
    R_eff = configured mixture weight
    if the host Beckmann VNDF sampler's view domain does not include -w:
        R_eff = 0
    choose component using R_eff
    sample n from that component
    qn = R_eff*qBeckmann(n) + (1-R_eff)*qUniform(n)
    psi = targetNdf.visibleNormalPdf(n,w)
    if psi == 0:
        return a zero-throughput sample
    wNew = reflectTravelDirection(w,n)
    jac = 1/(4*abs(dot(w,n)))
    return {
        wNew,
        Fresnel(abs(dot(w,n)))*psi*jac,
        qn*jac,
        Fresnel(abs(dot(w,n)))*psi/qn,
        n
    }
~~~

~~~text
evaluateSamplingPdf(w,wNew):
    if wNew == w: return 0 for the ordinary continuous branch
    n = normalize(wNew-w)
    if dot(w,n) >= 0: flip n if needed to select the incoming-facing branch
    evaluate the full mixture qn, not only the sampled component
    return qn/(4*abs(dot(w,n)))
~~~

复用宿主 Beckmann VNDF 时，必须把其“入射方向指向外部”的约定转换为 view=−w，并验证其 PDF。若仓库没有该采样器，必须提供以下无需额外论文算法的正确实现：

~~~text
sampleBeckmannVisibleNormal(w,alphaX,alphaY,rng):
    transform w into the proposal heightfield frame
    meanG = (0,0,1)
    covarianceG = diag(alphaX^2/2,alphaY^2/2,0)
    g = sampleFluxWeightedGradient(Gaussian(meanG,covarianceG),w,rng)
        // use F24's pure Gaussian routine, requiring only F05/F06
    return transformToWorld(normalize(g))

beckmannVisibleNormalPdf(n,w):
    return max(-dot(w,n),0)*D_B(n)/A_B(w)
~~~

这里采样的是带 crossing 权重的退化梯度 Gaussian，因此得到真正的 Beckmann VNDF；普通未加权梯度采样没有这个性质。可将这个纯数学 helper 在 F24 中单独声明，并用前置声明/实现文件隔离 FlightKernel，避免循环依赖。该参考采样器使用逆 CDF，可能比宿主专用 VNDF 算法慢，速度另行评估。

开发顺序：

- 第一阶段先用 R=0 完成正确的原模型复现。
- 随后用上述 flux-Gaussian 实现或宿主专用实现补齐混合采样及其 PDF；最终报告不得把 R=0 说成已实现混合提议。
- 不要用“先采普通 Gaussian 梯度再归一化”代替 VNDF。

除非已正确处理吸收 roulette，Fresnel 必须只进入 throughput 一次。phase energy 一般积分小于 1；不应无条件当成采样 PDF。

**验收：**F=1 时能量核积分为 1；实际 proposal PDF 积分为 1；均匀与混合提议对同一积分得到一致估计；白炉测试不增能。

## 19. F17 — transport/FlightState.h

**职责：**维护真实起点、传播年龄和最近交点条件。

~~~cpp
enum class BirthKind { External, Surface };
enum class ModelMode { Classic, Conditional29, Midpoint };
struct FlightState {
    BirthKind birthKind;
    Point3 birthPosition;
    Vector3 direction;
    double age;
    bool hasFullGradient;
    Vector3 birthGradient;
};
~~~

~~~text
startExternalFlight(x,w):
    return {External,x,normalize(w),0,false,unused}

startSurfaceFlight(x,g,w):
    require norm(g)>0 and dot(g,w)>0
    return {Surface,x,normalize(w),0,true,g}

startClassicCollisionFlight(x,w):
    // Valid only when subsequent propagation remains in Classic mode.
    return {Surface,x,normalize(w),0,false,unused}

onNullCollision(state,delta):
    state.age += delta
    // birthPosition, birthGradient and BirthKind stay unchanged

onRealSurfaceBounce(hitPosition,hitGradient,wNew):
    return startSurfaceFlight(hitPosition,hitGradient,wNew)
~~~

仅普通法线 n 不够构造 Surface 状态。统计起点不得因 ray epsilon 被偷偷平移；宿主几何求交偏移若必须使用，应单独保存真实统计锚点及累计距离。

External 不等于 $F(x₀)=0$。B/C 第一版明确采用 OriginalExternalPolicy：外部起点的第一段使用 A，命中后才使用最近交点条件。这是混合近似，不声称相机起点到整条路径都使用精确 GP 几何。

## 20. F18 — transport/FlightKernel.h

**职责：**让所有自由程采样器通过同一接口读取 hazard，同时保留几何诊断。

~~~cpp
struct HazardEvaluation {
    PositiveResult hazard;
    optional<double> logExteriorScreenProbability; // U / U1
    optional<double> logCrossingFlux;              // J0 / J1
};
struct HitStatistics {
    Gaussian<3> gradientGivenEndpointZero;
    optional<Gaussian<4>> midpointAndGradientGivenEndpointZero;
};
class FlightKernel {
public:
    virtual HazardEvaluation evaluate(double age) const = 0;
    virtual HitStatistics hitStatistics(double age) const = 0;
    virtual double currentAge() const = 0;
    virtual double maximumAgeInDomain() const = 0;
};
unique_ptr<FlightKernel> makeFlightKernel(
    ModelMode, const GPSSField&, const FlightState&, ExternalPolicy);
~~~

~~~text
makeFlightKernel(mode,field,state,policy):
    if mode == Classic: return ClassicFlightKernel(...)
    if field.ndfFamily == GGXBaseline: reject configuration
    if state.birthKind == External:
        require policy == OriginalExternalPolicy
        return ClassicFlightKernel(...)
    require full gradient and outward departure
    build one ConditionedRay and cache its observation factorization
    if mode == Conditional29: return Conditional29FlightKernel(...)
    if mode == Midpoint: return MidpointFlightKernel(...)
~~~

不要命名为 getTransmittance() 却返回 U 或 U₁。T_model 由采样器/光学深度积分模块计算。

## 21. F19 — transport/ClassicFlightKernel.h

**职责：**把原模型 A 接入 FlightKernel。

~~~text
evaluate(t):
    require 0 <= t <= domainExitAge
    x = birthPosition + t*direction
    return evaluateClassic(field,x,direction).extinction
           with optional diagnostic values

hitStatistics(t):
    require a Gaussian NDF family for gradient-based sampling
    prior = field.pointPrior(ray(t))
    // stationary unconditioned prior has Cov(F,G)=0
    return Gaussian(prior.meanG,prior.covarianceG)
~~~

A 忽略最近命中的起点条件；但接口仍保留真实年龄以方便共用采样代码。A 的原文相函数可以调用 F16；需要完整梯度以启动 B/C 时调用 F24。

## 22. F20 — transport/Conditional29FlightKernel.h

**职责：**完整实现式 (29) 的标量化、条件化和 log-hazard。

令 $(F,K)|Z$ 的均值为 $(m_F,m_K)$，协方差为：

$$
\begin{pmatrix}v_F&c\\c&v_K\end{pmatrix}.
$$

$$
m_{K|0}=m_K-c\,m_F/v_F,\qquad
v_{K|0}=v_K-c^2/v_F,
$$
$$
\log U=\log\Phi(m_F/\sqrt{v_F}),
$$
$$
\log J_0=
\log\mathcal N(0;m_F,v_F)+
\log M_1^+(-m_{K|0},\sqrt{v_{K|0}}).
$$

~~~text
evaluate(t):
    if t == 0 and the origin is a valid outward Surface birth:
        return exact-zero hazard
    fk = conditionedRay.endpointValueSlope(t)
    if its function-value variance is degenerate:
        use the explicit deterministic-support rules
    kGivenZero = conditionGaussian(K,F,observedF=0)
    logU = normalLogCdf(meanF/stddevF)
    logJF = normalLogPdf((0-meanF)/stddevF)-log(stddevF)
    logMoment = negativePartMean(kGivenZero.mean,kGivenZero.stddev).logValue
    logJ = logJF+logMoment
    logH = logJ-logU
    return {hazard from logH, logU, logJ}

hitStatistics(t):
    fg = conditionedRay.endpointValueGradient(t)
    return condition G on F_t=0 using F06
~~~

确定性规则：

- 统计起点 t=0 不算再次命中。
- 若沿此方向场完全确定且 $F_t>0$，没有局部 crossing，hazard=0。
- 确定性零点会产生离散碰撞质量，不能用普通连续 hazard 表示。若它在有效区间出现，走宿主确定表面分支；没有该分支则明确返回 UnsupportedSingularFlight，不能静默漏掉。
- 只要 logU 有限，即使普通 U 下溢为零，也不能把 h 设成无穷或零。
- 近起点经过两次条件化可能再次相消；需要 F12 稳定式或更高精度。

**验收：**无起点条件时退化为 A；可去相关方向上的远场趋于 A；起点向外出射时不产生 $t=0$ 自相交；每次返回 $h≥0$。

## 23. F21 — transport/MidpointFlightKernel.h

**职责：**实现一个中点的明确近似，不能将其标记为完整首次相交。

令 $Y=F_(t/2)$。分母：

$$
U_1=P(Y>0,F_t>0\mid Z).
$$

条件化 $F_t=0$ 后，设 $(Y,K)$ 的均值为 $(m_Y,m_K)$、方差为 $(v_Y,v_K)$、协方差为 $c$。若 $v_K>$0：

$$
s_r^2=v_Y-c^2/v_K,
$$
$$
a=\frac{m_Y-cm_K/v_K}{s_r},
\quad b=\frac{c/v_K}{s_r}.
$$

$$
J_1=p(F_t=0\mid Z)\,
E_{K\mid F_t=0,Z}[(-K)_+\Phi(a+bK)].
$$

~~~text
evaluate(t):
    if t == 0 and valid outward Surface birth: return zero
    yfk = conditionedRay.midpointValueSlope(t)
    logU1 = positiveOrthant2(marginal(yfk,(Y,F))).logValue
    ykGivenZero = condition (Y,K) on F=0
    if Var(K) > 0:
        residual = conditional variance of Y given K
        if residual > 0:
            derive a,b
            weightedMoment = negativeFluxTimesCdf(meanK,stdK,a,b)
        else:
            weightedMoment = negativeFluxTimesAffineIndicator(
                meanK,stdK, meanY-c*meanK/varK, c/varK)
    else:
        weightedMoment = max(-meanK,0) * P(Y>0 | F=0,Z)
        // support/deterministic checks still apply
    logJ1 = logPdfEndpointAtZero + weightedMoment.logValue
    return {exp/log(logJ1-logU1), logU1, logJ1}

hitStatistics(t):
    yfg = conditionedRay.midpointValueGradient(t)
    return condition (Y,G) on F_t=0
~~~

中点分母和分子必须使用同一个 Z、同一个 t/2、同一套协方差。若真正出现 U₁=0，不能继续假装该年龄仍有一个正常条件分布；区别它与仅普通数值下溢，并返回支持/精度诊断。

提供 debug 开关 screenCount=0，严格调用 B 计算路径，用于验证去掉检查点后完全一致。不得假定 C 一定比 B 更准确或更亮；比较结果由实验决定。


## 24. F22 — transport/OpticalDepthSampler.h

**职责：**从一个非负 hazard 生成概率一致的自由程。A 的参考实现和 B/C 的首选实现都使用它。

$$
H(t)=\int_0^t h(s)\,ds,\quad
T_{\mathrm{model}}(t)=e^{-H(t)},\quad
p_{\mathrm{model}}(t)=h(t)e^{-H(t)}.
$$

~~~cpp
struct FlightSample {
    bool collided;
    double age;
    optional<double> logSurvival; // populated by optical-depth sampling; may be lazy for null tracking
    optional<double> logDistancePdf; // continuous density, only for a collision
    optional<double> escapeMass;     // discrete boundary-escape probability
};
PositiveResult integrateHazard(const FlightKernel&, double a, double b);
FlightSample sampleFlight(const FlightKernel&, RNG&);
~~~

~~~text
sampleFlight(kernel,rng):
    a = kernel.currentAge()
    b = kernel.maximumAgeInDomain()
    E = -log1p(-openUniform01(rng))
    H_b = integrateHazard(kernel,a,b)
    if E >= H_b:
        return {false,b,-H_b,noDensity,exp(-H_b)}
    build a monotone bracket for integral_a^t h(s) ds = E on [a,b]
    t = safeguarded monotone root solve
    h = kernel.evaluate(t).hazard
    return {true,t,-E,log(h)-E,noEscapeMass}
~~~

工程要求：

- 缓存已经计算的积分区间及累计 H，避免求根时从零重复所有积分。
- 累计 H 必须单调非降。h=0 的区间允许；根定位时跳过没有光学深度增长的区间。
- 可以另提供显式的分段常数 hazard 表。每个格子 $h_j≥0$，H 在格内线性，采样可解析反演。此时报告“采样的就是该表定义的近似模型”，进行网格收敛检查。
- 禁止混合“用表采样距离、用未经同一近似的原 h 返回 PDF”，否则权重不一致。
- 普通数值溢出/求根失败不是逃逸。
- 有限域逃逸质量是 exp(−H(b))，不能把所有碰撞样本重新归一化成总质量 1。
- 起点年龄零只对真实起点成立；若继续一个已走了 a 的存活段，剩余光学深度为 $H(t)−H(a)$，不能无条件重新使用 $H(t−a)$。
- 当 $a>0$ 时，上述返回的 logSurvival、距离 PDF 和逃逸质量均条件化于已经存活到 a；age 仍是相对于原 birth 的绝对年龄。

**平面 A 的解析验收：**Aproj 不变，$d(t)=d₀+w_z t。w_z≠0$ 时：

$$
H_A(t)=
\frac{A_{\mathrm{proj}}}{w_z}
\left[
\log\Phi\!\left(\frac{d_0+w_zt}{\sigma}\right)
-\log\Phi\!\left(\frac{d_0}{\sigma}\right)
\right].
$$

w_z=0 时 H_A(t)=Aproj ρ(d₀)t。很小的 w_z 用稳定差值或极限。这个恒等式直接检验密度、方向和积分采样。

## 25. F23 — transport/ClassicNullTracking.h

**职责：**使用合法 majorant 复现原模型 A 的 null-scattering 自由程采样。先保留一个简单、可验证的 analog delta-tracking 版本。

~~~cpp
FlightSample sampleClassicNullTracking(
    const ClassicFlightKernel&, double certifiedMajorant, RNG&);
~~~

~~~text
sampleClassicNullTracking(kernel,M,rng):
    require M >= 0 and a certified bound
    t = kernel.currentAge()
    if M == 0: return boundary escape
    loop:
        t += -log1p(-openUniform01(rng))/M
        if t >= domainExitAge:
            return boundary escape
        h = kernel.evaluate(t).hazard.value
        if h > M beyond rounding allowance:
            report InvalidMajorant and abort/retry with a valid method
        if openUniform01(rng) < h/M:
            return a real collision at age t
        record a null collision
        keep the original statistical birth and keep accumulating age
~~~

delta tracking 自身不需要计算每条已采样 flight 的 T 来更新 analog throughput。若统一接口或实验输出要求 logSurvival/logDistancePdf，可用 F22 额外计算；不要为每个 null 事件都重算整个积分。

这个 majorant 仅用于 A，包括 B/C 的 External→A 第一段。B/C 的条件方差和中点分母会改变 hazard，上述上界不能直接拿来用。

**验收：**在平面 A 场景中，自由程直方图、逃逸频率与 F22/解析 H 匹配；null 事件不增加 bounce 深度，也不刷新 g₀。

## 26. F24 — transport/CollisionGradientSampler.h

**职责：**在已采样的命中年龄 t 上采样完整梯度，使散射和下一次状态更新与所选 hazard 的 crossing 统计一致。

### 26.1 A/B 的精确局部 crossing mark

A 使用无条件局部 G；B 使用 $G|F_t=0,Z$。设其 Gaussian 为 $N(μ,C)$，则需要：

$$
q_G(g)=
\frac{[-w\cdot g]_+\,\mathcal N(g;\mu,C)}
{E[(-w\cdot G)_+]}.
$$

~~~cpp
Vector3 sampleFluxWeightedGradient(Gaussian<3> gradient, Vector3 w, RNG&);
Vector3 sampleCollisionGradient(const FlightKernel&, double t, RNG&);
~~~

~~~text
sampleFluxWeightedGradient(gGaussian,w,rng):
    mk = dot(w,mu)
    cw = C*w
    vk = dot(w,cw)
    if vk > 0:
        k = sampleNegativeFluxNormal(mk,sqrt(vk),rng)
        muGivenK = mu + cw*(k-mk)/vk
        covGivenK = C - outer(cw,cw)/vk
        U = orthonormal 3x2 basis perpendicular to w
        z = sampleGaussianPSD(
                mean = transpose(U)*muGivenK,
                cov = transpose(U)*covGivenK*U)
        g = k*w + U*z
    else:
        require mk < 0
        g = sampleGaussianPSD(gGaussian,rng)
    require dot(w,g) < 0 and norm(g)>0
    return g
~~~

使用二维横向坐标可以保持 $w·g=k$，避免因三维退化分解引入投影误差。

### 26.2 C 的中点筛选 mark

C 的目标分布为：

$$
q_C(g)\propto[-w\cdot g]_+\,
p(G_t=g\mid F_t=0,Z)\,
P(Y>0\mid F_t=0,G_t=g,Z).
$$

~~~text
sampleMidpointGradient(stats,w,rng):
    // stats contains the joint (Y,G) conditioned on F_t=0,Z
    baseG = marginal(stats,G)
    attempts = 0
    loop:
        g = sampleFluxWeightedGradient(baseG,w,rng)
        yGivenG = condition Y on G=g using the same joint Gaussian
        Vg = probability(yGivenG > 0)
        attempts += 1
        if openUniform01(rng) < Vg:
            record attempts for efficiency diagnostics
            return g
        if resource budget is exhausted:
            report SamplingNotConverged; do not force acceptance
~~~

计算 hazard 时已边缘化横向梯度，所以 F21 使用 V(k)；**给定完整 g 做 rejection 时必须使用 V(g)，不能复用 V(k)**。退化的 $Y|G$ 用确定性正值判断。

理论接受率为 $J₁/J₀$，可以用来检测实现和评估成本。低接受率时可以后续设计重要性采样，但不能把拒绝上限后的普通样本伪装成正确 mark。

### 26.3 反射及权重

~~~text
g = sampleCollisionGradient(kernel,t,rng)
n = normalize(g)
wNew = w - 2*dot(w,n)*n
weight = conductorFresnel(abs(dot(w,n)))
newState = startSurfaceFlight(hitPosition,g,wNew)
~~~

这里 mark 是按目标 crossing 分布精确采样的，所以只乘 Fresnel；不再乘一次 VNDF/PDF 比率。A 使用 F16 的 proposal 模式时则必须乘其 energy/pdf。

B/C 的 External 第一段虽然使用 A hazard，也必须通过本文件采完整 g，才能启动下一段条件模型。不能在第一次命中只采 n 后令 g=n。

**验收：**K 的经验分布对照 F05；所有入射 crossing 满足 $w·g<0$；反射后 $w_{New}·g>0$；中点 rejection 的接受率匹配 $J₁/J$₀；梯度分布包含正确的横向条件变化。

## 27. F25 — integrator/MacrofacetPathTracer.cpp

**职责：**连接场景边界、flight、散射和状态，完成 A/B/C 的最小多次散射渲染。

宿主适配接口至少包括：

~~~cpp
Ray cameraRay(Pixel, RNG&);
optional<SceneHit> intersectDeterministicScene(Ray);
DomainInterval intersectActiveDomain(Ray);
Spectrum environmentEmission(Vector3 direction);
Spectrum surfaceEmission(SceneHit);
DeterministicBsdfSample sampleDeterministicBsdf(SceneHit, Vector3, RNG&);
void addSample(Pixel, Spectrum radiance);
~~~

~~~text
traceCameraPath(initialRay,mode,rng):
    L = 0
    beta = 1
    ray = initialRay
    state = startExternalFlight(ray.origin,ray.direction)
    depth = 0

    while path survives:
        find the next active-domain interval and deterministic scene intersection
        if there is no participating interval before the scene event:
            accumulate visible environment or surface emission
            if the deterministic surface scatters:
                update beta and direction with its BSDF/pdf
                state = startExternalFlight(exactSurfacePosition,newDirection)
                continue
            otherwise stop

        locate exact entry to the modeled material domain
        preserve an existing Surface anchor if merely evaluating a clipped segment
        for a new External birth, choose entry as the explicit modeled origin
        set maximum age to min(domain exit, deterministic scene intersection)

        kernel = makeFlightKernel(mode,field,state,externalPolicy)
        flight = choose legal sampler:
            Classic -> null tracking or optical-depth sampling
            Conditional29/Midpoint -> optical-depth sampling

        if flight escapes:
            continue to the actual domain boundary or deterministic scene event
            do not add a fake surface bounce at the domain boundary
            do not multiply beta by T again
            continue

        x = state.birthPosition + flight.age*state.direction
        if mode == Classic and proposalPhaseEnabled:
            ps = samplePhase(state.direction,rng)
            beta *= ps.throughputWeight
            state = startClassicCollisionFlight(x,ps.direction)
        else:
            g = sampleCollisionGradient(kernel,flight.age,rng)
            n = normalize(g)
            wNew = reflectTravelDirection(state.direction,n)
            beta *= conductorFresnel(abs(dot(state.direction,n)))
            state = startSurfaceFlight(x,g,wNew)

        ray = rayFromExactStatisticalAnchor(state)
        depth += 1
        apply unbiased Russian roulette after configured minimum depth
        stop only when absorbed, escaped to evaluated lighting, or roulette terminates
    return L
~~~

实现时可把凸域内散射循环和域外场景循环分开，但必须明确当前 ray 与统计 anchor 的差异。

**初版照明约束：**

- B/C 先实现通过路径命中环境或有面积光源获得照明的 analog 路径追踪。
- 不直接复用 A 的相函数和可见性做 B/C 的 next-event estimation。那需要给定目标方向后的 mark 积分、对应 PDF 和连接边的状态条件化，属于另一个实现任务。
- 三模式公平对比时都关闭 NEE，使用可被随机路径命中的环境/面积光；不能只用点光源然后把全黑误认为模型失效。
- 生产 A 可以复用宿主已经正确的 NEE，但独立记录这一配置。
- 若使用硬性 maxDepth，必须报告其截断偏差；优先用 Russian roulette，资源上限触发时单独计数。
- 以上 B/C 定义了带最近状态的实验传播过程；没有证明它恢复完整 GP ensemble，也没有自动证明双向互易性。不要把 camera-path 实验描述成完整物理 GP 模型的无偏渲染器。

## 28. F26 — experiments/FlightCurveExperiment.cpp

**职责：**固定同一个 Surface 起点条件，输出三种 hazard 和其实际自由程分布。先完成这项，再比较图片。

~~~cpp
void runFlightCurves(const ExperimentConfig&);
~~~

~~~text
runFlightCurves(cfg):
    field = buildField(cfg)
    state = startSurfaceFlight(cfg.x0,cfg.g0,cfg.w)
    b = min(cfg.requestedMaximumAge,domainExitAge)
    for mode in {Classic,Conditional29,Midpoint}:
        kernel = makeFlightKernel(mode,field,state,...)
        for t in a shared age grid on [0,b]:
            e = kernel.evaluate(t)
            H = integrateHazard(kernel,0,t)
            write {
                mode,t,h,logh,H,T_model=exp(-H),
                p_model=h*exp(-H),
                logScreenProbability,logCrossingFlux,
                integrationError,numericStatus,elapsedTime
            }
        draw configured independent flight samples
        write collision bins and escape count
        compare bin frequency to T_model(left)-T_model(right)
    join with the F27 reference on the same domain and age grid
~~~

输出至少包含 CSV 数据及运行配置副本/摘要。利用现有绘图脚本或一个简单 Python 后处理画 h、T、碰撞直方图；绘图只是展示，不是另一个必需的 C++ 模块。CSV 要保留 log 概率，避免图上全部显示 0 时无法诊断。

必须扫描的参数包括：出射角、σ/ℓ 比率、起点高度 d₀、起点梯度方向和长度、各向异性。不是每个组合都跑昂贵参考；先选 3–5 个预先确定的代表案例。

## 29. F27 — experiments/ConditionalGPReference.cpp

**职责：**提供独立、可逐步收敛的数值参考，并实际评估用户的整段首次相交公式。它用于研究验证，运行时不依赖此模块。

### 29.1 联合 GP 路径采样参考

~~~cpp
ReferenceResult sampleConditionalGPFirstHit(
    const ConditionedRay&, vector<double> positiveAges,
    int sampleCount, uint64_t seed);
~~~

~~~text
sampleConditionalGPFirstHit(ray,ages,M,seed):
    require 0 < ages[0] < ... < ages[N-1]
    g = ray.valuesAt(ages)
    factor its covariance once, respecting PSD and numerical rank
    for repetition in 1..M:
        f = jointly sample the whole vector from g
        first = first index j with f[j] <= 0
        if none:
            record right-censored escape at ages.back()
        else:
            record detected-hit bin j
            optionally interpolate a crossing in the first sign-change bracket
            keep interpolation distinct from an exact continuous intersection
        update the prefix-positive survival counts
    T_grid[j] = count(all f[0..j] > 0)/M
    P_grid[j] = firstHitCount[j]/M
    averageHazard[j] = -log(T_grid[j]/T_grid[j-1]) / deltaAge[j]
    attach confidence intervals and zero-count diagnostics
~~~

起点 t=0 的函数值已确定为零，不把它放入严格正值检查。起点向外导数已知，第一小区间仍可能有回穿，须靠网格收敛控制。每个网格样本必须联合采样；逐点独立高斯采样会破坏整个参考。

在相同最大距离下采用嵌套网格 Δ、Δ/2、Δ/4。有限网格可能漏掉穿入—穿出，也可能把第一次命中推迟；不能只检查端点符号便声称连续正确。可在候选区间条件加点，但对潜在“同号端点、内部双穿越”仍需整体加密验证。

如果使用同一随机路径比较嵌套网格，粗网格取细网格的子集；这样正值事件逐级收紧，可直接观察漏检。大矩阵 PSD 分解允许清除数值舍入对应的负特征值；需要额外截断小正特征值时，记录被丢弃方差及阈值收敛，不能静默称为精确参考。

### 29.2 直接评估整段公式的多检查点近似

~~~cpp
ScreenedFormulaEstimate estimateScreenedFirstPassageHazard(
    const ConditionedRay&, double t, vector<double> interiorAges,
    int sampleCount, uint64_t seed);
~~~

~~~text
estimateScreenedFirstPassageHazard(ray,t,interiorAges,M,seed):
    require every interior age lies strictly in (0,t)
    assemble the joint Gaussian (Y_1,...,Y_N,F_t,K_t) | Z
        using ConditionedRay.checkpointValuesAndEndpointSlope(interiorAges,t)

    // denominator: sample endpoint together with all interior values
    draw M joint samples of (Y,F_t) | Z
    estimate U_N = mean(indicator(all Y>0 and F_t>0))

    // numerator: equality condition is Gaussian conditioning, not rejection
    conditional = Gaussian of (Y,K_t) given F_t=0,Z
    draw M joint samples from conditional
    for each sample:
        W = max(-K_t,0) * indicator(all Y>0)
    estimate J_N = pdf(F_t=0 | Z) * mean(W)

    if denominator is statistically resolved and positive:
        estimate h_N = J_N/U_N
        attach uncertainty from independent batches or a ratio bootstrap
    else:
        report UnresolvedRareEvent; increase samples or use rare-event methods
        never replace zero observed survivors by epsilon
    return U_N,J_N,h_N,errors,sampleCounts,cost
~~~

零个检查点用 B 的解析结果验证；一个中点用 C 的确定性积分验证；随后采用 N=4、8、16、32… 比较。每个 t 的检查点可取 j*t/(N+1)，但其变化意味着有限 N 的 U_N(t) 不是一个已经证明对应 J_N 的 survival。绘制 h_N 时保留这个标记；真正的经验生存曲线来自 29.1 的固定嵌套网格。

为了降低 numerator 方差，可进一步：

~~~text
condition all checkpoint values on F_t=0 and sampled K=k
sample k from its negative-flux Gaussian q_K
estimate V_N(k) by joint conditional checkpoint samples
J_N = pF0 * negativePartMean(K|F_t=0) * average(V_N samples)
~~~

该形式直接对应原积分中的 V_t(k)。外层 k 与内层检查点的相关条件必须完整保留。若一次抽样就生成条件检查点，用 0/1 survival 指示量也是无偏的有限网格 numerator 估计；ratio 本身通常有有限样本偏差，报告这一点。

### 29.3 参考的解释范围

- 它逼近 GP 首次穿越问题，与 GPIS 的几何目标有联系；本项目研究点是 B/C 的低成本闭合，不是把联合 GP 采样重新命名为新方法。
- 返回域边界没有检测到碰撞属于删失/逃逸质量，不把碰撞样本归一化成总概率 1。
- 距离参考是必做；若要检验法线，可额外联合采样梯度并在加密交点附近条件采样。只在线性插值位置独立采一个梯度，不是有效参考。
- 不保证有限样本或有限网格已经达到精确解。网格变化应与 Monte Carlo 置信区间一起报告。

## 30. F28 — experiments/RenderExperiment.cpp

**职责：**建立不依赖作者私有资产的可重复渲染与白炉验证。

~~~text
runRenderExperiments(cfg):
    build analytic scenes:
        plane-based finite slab, Gaussian conductor
        Beckmann heightfield limit for original-model comparison
        optional sphere mean with fixed world-space Gaussian kernel
        optional constant mean in a convex box for volumetric appearance
    for supported modes and matched parameter sets:
        render with fixed camera, lighting, domain, seed scheme and sampler budget
        save linear radiance image and an optional display preview
        save elapsed time, samples, real/null collision counts,
             mean path length, numerical failures, gradient-rejection attempts
    compare only images using the same scene and lighting convention
~~~

最低要求：

- 第一张图是单位白环境、F=1 的能量验证；再使用一个可解析评估的方向变化环境观察外观差异。
- 默认使用同一组 Gaussian 参数对比 A/B/C；heightfield/GGX 只用于明确支持它的模式。
- 线性输出优先使用宿主 EXR；没有图像库时写 PFM。显示用 PNG 的 tone mapping 不进入 MSE。
- 不比较不同 exposure 后的像素误差。
- 没有作者原图数据时，不输出“与原文误差为 X”；只报告本项目的自洽验证和模式差异。
- 单次更亮/更暗不直接等于更准确。精度评价首先依赖 F26/F27 的自由程统计。

## 31. F29 — experiments/ExperimentMain.cpp

**职责：**配置解析、模式选择、随机种子、实验输出目录和失败状态。

~~~text
main(args):
    parse command in {curves,gp-reference,render,all}
    load config and apply explicit command-line overrides
    validate field parameters, covariance, domain, origins and model compatibility
    create an output directory named by experiment and reproducible seed
    write the resolved configuration
    run requested experiment stages
    write a machine-readable summary with success/failure and diagnostics
    return nonzero if a required validation or numerical computation failed
~~~

建议命令接口：

~~~text
macrofacet_experiments curves --config configs/macrofacet_experiments.json
macrofacet_experiments gp-reference --config configs/macrofacet_experiments.json
macrofacet_experiments render --config configs/macrofacet_experiments.json
macrofacet_tests
~~~

实际可执行文件名按宿主项目调整，并在 F37 写出真正运行过的命令。数学测试独立于完整渲染启动，便于先定位底层错误。


## 32. F30 — tests/test_gaussian_math.cpp

**职责：**独立验证 Gaussian 数学；不要只把同一实现换一个函数名再互相比较。

~~~text
testGaussianMath():
    compare normal log-CDF against trusted high-precision reference values
    verify CDF/quantile round trips including deep tails
    compare M0..M3 and negative-part moments against independent quadrature
    compare Gaussian conditioning against hand-derived 2-variable cases
    verify PSD sample moments in a rank-2 subspace
    check bivariate probability at rho=0,+1,-1
    compare general bivariate probability against an independent reference
    compare weighted-CDF moment with direct joint-Gaussian Monte Carlo
    verify exact deterministic branches without variance jitter
~~~

建议误差策略：

- 中央区的解析恒等式：相对误差约 1e−9；积分结果按其误差估计放宽。
- 深尾：比较 log 概率和 log 矩，避免普通值同时下溢造成“看起来一致”。
- 随机检验：使用预先给定的置信水平和样本量；固定 seed 便于复现，但不要只针对一个 seed 调容限。
- 统计检验失败时检查多重比较和抽样波动，不通过增加 epsilon 让公式通过。

## 33. F31 — tests/test_gpss_statistics.cpp

**职责：**检查核导数、观测组装和条件场。

~~~text
testGPSSStatistics():
    compare kernel analytic derivatives to central finite differences
    verify Cov(A,B) = transpose(Cov(B,A))
    verify prior same-point Cov(F,G)=0
    verify conditioning generally creates nonzero Cov(F_t,G_t)
    compare general conditioning to the plane analytic formulas
    verify observed origin value and gradient have zero conditional variance
    reject an inconsistent deterministic gradient observation
    evaluate very small t/ell and check PSD plus finite log quantities
    check distant points recover the prior when kernel correlations decay
~~~

有限差分只用于核导数的独立测试，不代替主实现中的解析导数。远场测试排除沿所有零 precision 方向传播、相关性根本不衰减的情况。

## 34. F32 — tests/test_macrofacet_baseline.cpp

**职责：**验证真正的原模型 A，包括测度、相函数和消光。

~~~text
testMacrofacetBaseline():
    numerically integrate n*D(n) over sphere; compare to meanG
    integrate max(-w.n,0)*D(n); compare to analytic projectedArea
    integrate VNDF over sphere; compare to 1
    compare generalized NDF to independent radial Gaussian integral
    compare exact Beckmann branch to alphaZ -> 0 integrated quantities
    verify GGX projected-area formula by sphere quadrature
    set Fresnel=1; integrate phase energy and proposal PDF separately
    compare uniform and mixed-proposal Monte Carlo estimates
    compare plane optical-depth integral with its analytic expression
    compare null-tracking flight CDF with analytic T
    run a small white-environment energy test
~~~

球面积分必须包含正确的 dΩ；不能把均匀 θ、φ 网格直接平均当成均匀球面。高度各向异性时需要收敛检查。

## 35. F33 — tests/test_flight_kernels.cpp

**职责：**验证 hazard 闭合及其产生的概率模型，而不是假定所有闭合等于真实首次相交。

~~~text
testFlightKernels():
    without origin conditioning, verify Eq29 reduces to classic Gaussian hazard
    with no screen, verify midpoint implementation reduces exactly to Eq29
    with one screen, compare deterministic midpoint formula to F27 Monte Carlo
    check h(t)>=0 and required log values remain finite for valid inputs
    check H(t) is nondecreasing and T_model(t) is nonincreasing
    integrate p_model on [0,b]; add exp(-H(b)); compare total to 1
    demonstrate on a nontrivial case that U(t) need not equal T_model(t)
    compare conditional-GP reference across nested grids and confidence intervals
    verify External policy dispatch and GGX incompatibility errors
~~~

不得编写“C 必须优于 B”“C 必须更暗”“相关模型必然互易”等没有被理论保证的断言。比较误差可以作为实验结果，但不是预设的通过条件。

## 36. F34 — tests/test_sampling.cpp

**职责：**验证采样分布、逃逸质量、完整梯度以及状态传递。

~~~text
testSampling():
    draw flights and compare their bins to T(left)-T(right)
    compare escape count to T(b), including the discrete escape atom
    draw flux-weighted K and compare to analytic R(k)
    draw full gradients and verify projection plus conditional transverse moments
    verify midpoint rejection acceptance agrees with J1/J0
    compare midpoint full-gradient moments against direct weighted Gaussian samples
    simulate null events; assert birth position and gradient do not change
    assert real bounce resets age and stores the newly sampled full gradient
    verify analog flight does not multiply throughput by T a second time
    verify mixture sample PDF evaluates the full mixture
~~~

直接加权 Gaussian 参考的中点 mark 测试：

~~~text
draw many (Y,G) from their joint Gaussian conditioned on F_t=0,Z
weight = max(-dot(w,G),0)*indicator(Y>0)
referenceMoment = sum(weight*chosenFunction(G))/sum(weight)
compare to the unweighted average from sampleMidpointGradient
~~~

这项测试能检测错误使用 V(k) 代替 V(g)，比仅检查方向正负更有意义。

## 37. F35 — configs/macrofacet_experiments.json

**职责：**所有决定统计模型的参数都可复现。以下为默认小实验；实现者可按宿主 JSON 规范映射，但不能偷偷改变参数含义。

~~~json
{
  "schema_version": 1,
  "seed": 17429,
  "modes": ["classic", "conditional29", "midpoint"],
  "field": {
    "mean_type": "plane",
    "plane_normal": [0.0, 0.0, 1.0],
    "plane_offset": 0.0,
    "sigma": 0.1,
    "correlation_lengths": [0.2, 0.2, 0.2],
    "kernel_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "ndf_family": "generalized_gaussian",
    "domain_min": [-2.0, -2.0, -0.3],
    "domain_max": [2.0, 2.0, 0.3],
    "outside_domain": "vacuum"
  },
  "material": {
    "type": "conductor",
    "eta_rgb": [0.2, 0.9, 1.1],
    "k_rgb": [3.9, 2.5, 2.2],
    "force_unit_fresnel_for_energy_test": false
  },
  "transport": {
    "external_policy": "original_macrofacet",
    "classic_sampler": "null_tracking",
    "correlated_sampler": "optical_depth",
    "classic_phase_proposal": "uniform",
    "beckmann_mixture_weight": 0.5,
    "next_event_estimation": false,
    "roulette_start_depth": 5,
    "hard_depth_cap": null
  },
  "fixed_flight": {
    "birth_position": [0.0, 0.0, 0.0],
    "birth_gradient": [0.0, 0.0, 1.0],
    "direction": [0.984807753012208, 0.0, 0.17364817766693033],
    "requested_maximum_age": 1.5,
    "curve_sample_count": 65,
    "flight_sample_count": 20000
  },
  "reference": {
    "path_sample_count": 4096,
    "nested_grid_intervals": [64, 128, 256],
    "formula_checkpoint_counts": [0, 1, 4, 8, 16, 32],
    "formula_age_count": 12,
    "formula_sample_count": 8192,
    "confidence_level": 0.95,
    "max_grid_points": 512
  },
  "numeric": {
    "relative_tolerance": 1e-7,
    "absolute_tolerance": 1e-10,
    "max_quadrature_subdivisions": 4096,
    "max_root_iterations": 128,
    "higher_precision_fallback": true,
    "allow_unreported_jitter": false
  },
  "render": {
    "width": 64,
    "height": 64,
    "samples_per_pixel": 256,
    "camera_position": [0.0, 0.0, 1.0],
    "camera_target": [0.0, 0.0, 0.0],
    "vertical_fov_degrees": 45.0,
    "environment": "directional_gradient",
    "linear_output_format": "host_exr_or_pfm"
  },
  "output_directory": "outputs/macrofacet_experiments"
}
~~~

参数约定：

- 默认 α_x=α_y=α_z=√2·0.1/0.2≈0.7071，从 σ、ℓ 推导，不独立输入互相矛盾的 α。
- correlation_lengths 中 null 可约定为 ∞，解析时明确实现；不能把 JSON 的非标准 Infinity 当作数字。
- eta/k 是示例 RGB 参数，不宣称对应某种测量金属；宿主若使用光谱，按其标准接口转换并记录。
- classic_phase_proposal="uniform" 使用 R=0；切换 "paper_mixture" 时才启用配置 R，并要求对应 sampler 可用。
- fixed_flight 的最大距离还要受 AABB 限制；将实际使用的 b 输出到结果。
- path_sample_count 是入门预算，不保证所有稀有事件都已解析。参考失败时标注不确定性并增加对应预算，不把未知变成零。
- 减小默认图像尺寸或样本量可用于开发，但最终报告写出实际预算和噪声。

配置校验伪代码：

~~~text
validate(cfg):
    require sigma>0
    require finite positive ell or the explicit infinity representation
    require orthonormal kernel rotation and a nonempty domain
    require unit ray direction, or normalize and record the normalization
    require dot(birthGradient,direction)>0 for a Surface fixed flight
    require observation lies in the Gaussian support
    reject conditional29/midpoint with GGX
    reject unknown external policy, unknown numerical modes and invalid tolerances
    resolve derived alpha and print it in the run summary
~~~

## 38. F36 — 构建配置

**职责：**将上述模块接入实际仓库。已有构建系统优先，不为了本文新增一个互不相干的工程。

~~~text
configureBuild():
    locate the host's math/vector/matrix/RNG/test/image facilities
    choose C++17 or the host's supported language standard
    add GPSS, macrofacet and transport sources to an appropriate library target
    expose headers through the existing include hierarchy
    link the host linear algebra and numerical special-function backends
    create macrofacet_tests or register tests in the host runner
    create macrofacet_experiments or register equivalent CLI subcommands
    keep expensive render/reference experiments outside default fast unit tests
~~~

若从空仓库实现，推荐 C++17 + 一个成熟矩阵库，Gaussian CDF 可用标准 erfc，log-CDF/逆 CDF/尾部处理按 F04 补齐。不要为了“所有数学自己实现”另写一个通用线性代数库。

依赖对应的许可证和版本沿用项目政策。构建报告必须区分“已有后端被复用”和“本次新增函数”，不能把外部库功能计作自己新实现但也不应重复造轮子。

## 39. F37 — docs/IMPLEMENTATION_REPORT.md

**职责：**实现完成后由执行 Codex 写入真实结果。本文中的示例不是报告结果，不能直接复制成已验证的事实。

~~~text
writeImplementationReport():
    record the actual paper version and equation-content mapping
    list each logical file F01..F37 and its actual repository path
    describe reused host interfaces and dependencies
    list implemented modes and unsupported configurations
    list approximations separately from numerical tolerances
    include exact build/test/experiment commands actually executed
    include test pass/fail results and resolved failures
    include fixed-flight plots/data paths and reference convergence
    include render paths, seeds, sampling budgets and timing environment
    include remaining limitations and any genuine blockers
~~~

必须回答：

1. A 是否通过 NDF、投影面积、相函数、解析平面自由程和白炉验收？
2. B 的统计是否真正来自给定 F₀、完整 G₀ 的条件 Gaussian？
3. C 是否在分子和分母使用同一个中点约束？
4. 整段公式是否由 F27 进行多检查点和联合 GP 首次相交验证？
5. 三种模式的距离 PDF 是否都与各自的积分 hazard 一致？
6. B/C 的后续反射是否使用匹配的完整梯度 mark？
7. External 起点、域截断、最近交点记忆、有限检查点分别采用什么近似？
8. 哪些结果有采样噪声、离散误差、数值积分误差？是否达到目标精度？
9. 是否实际实现论文式 (40) 的混合提议，还是目前仅使用正确但较慢的均匀提议？
10. 有没有尚未证明的互易性、完整 GP 一致性或效率结论？这些不能写成已证明。

## 40. GBE 对应关系：实现时只接入这些 building blocks

本节说明架构含义，不要求另写一个 PDE 求解器。

| GBE/非指数传输量 | 本规格中的对象 | 计算 |
|---|---|---|
| 传播年龄 | FlightState.age | 从真实 birth 累计的距离 |
| 起点类型 | BirthKind + ExternalPolicy | 外部起点与 GP 表面起点分开 |
| Σ(t) / hazard | FlightKernel.evaluate | A、B、C 各自的闭合 |
| T(t) | F22 的 exp(−H(t)) | 积分同一个 hazard |
| p(t) | h(t)T(t) | 连续碰撞密度 |
| 有限域逃逸 | T(b) | 边界上的离散概率质量 |
| 散射算子 | hazard × mark 分布 × 局部反射能量核 | F16/F24/F25 |
| 真正的整段几何参考 | F27 | 联合 GP 正值事件与 crossing 筛选 |

保留起点状态 Z 的传播方程可写为：

$$
(\partial_t+w\cdot\nabla_x)L_Z(x,w,t)
=-h_Z(t)L_Z(x,w,t),\qquad t>0.
$$

碰撞时先按当前 mark 分布采样 g，再用局部 conductor 反射获得 w'，并将新状态注入年龄零：

$$
Z'=\{F(x_{\mathrm{hit}})=0,\ G(x_{\mathrm{hit}})=g\}.
$$

这就是 F22 的 flight 与 F24 的 mark 组成一次状态转移的原因。只改 h 而继续使用未条件化的原相函数，是可以专门研究的消融，但不是这里要求的一致闭合。

对于光线段分段积分，独立指数候选只是计算工具；本模型并没有因此证明各段的底层随机表面独立。若真正把相邻段独立化，则是另外一种几何闭合，必须单独命名、验证。

## 41. 推荐完成顺序与交付门槛

| 里程碑 | 必须完成 | 通过后才能做什么 |
|---|---|---|
| M1 数学底座 | F01–F06、F09、F11 的基本局部统计；关键数学测试 | 使用这些量构造物理系数 |
| M2 原模型 A | F13–F19、F22/F23、A 的 F25；F32 验收 | 宣称完成原文核心模型复现 |
| M3 条件式 (29) | F10/F12/F20、F24 的 A/B mark、F26 | 比较带起点条件的自由程 |
| M4 整段数值参考 | F27 的两个参考路径及网格收敛 | 评价式 (29) 与完整首次相交目标的偏差 |
| M5 中点候选 | F07/F08/F21、F24 的 C mark、相应测试 | 比较一个具体低成本相关闭合 |
| M6 渲染对比与报告 | F25/F28–F29、F35–F37、全部必要验收 | 交付可复现结果 |

F14 GGX 与论文混合提议属于 A 的兼容性/效率补全；先完成正确的 Gaussian 基准，不让这两项阻塞首次相交问题的验证。若未完成，明确标为未完成的扩展，不冒充全部论文功能。

最终仓库交付物应包含：

- 可编译运行的实际实现和必要测试。
- 至少一个通过解析验证的原 Macrofacet 场景。
- 同一 Surface 起点条件下的 A/B/C 曲线及 F27 整段参考。
- 有限域逃逸概率、距离分布、梯度采样一致性验证。
- 三模式小图对比、配置、种子与真实运行命令。
- IMPLEMENTATION_REPORT.md，逐项说明完成度与限制。

## 42. 给执行 Codex 的最后说明

请将本文件作为实施规格，把用户附带的论文作为原模型依据。先检查现有工程并列出逻辑文件映射，然后按里程碑实现、验证并继续后续阶段。不要只生成接口空壳、把 U 当成 T、把 crossing 强度当成首次碰撞 PDF，或只输出一份新的计划。

遇到论文排版或记号歧义，用本文件中的测度、量纲和概率归一化验收定位问题；如发现本文某条公式与可靠推导冲突，给出具体差异并修正实现与报告，不要为了“遵守规格”保留数学错误。

本规格没有证明中点闭合必然比式 (29) 更好，也没有把完整 GP 数值参考当成低成本算法。需要实现后用同一组数据回答的是：**恢复最近碰撞信息，以及再加入少量无相交约束，分别能以什么成本改善哪些距离、逃逸和法线统计。**

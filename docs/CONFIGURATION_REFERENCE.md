# Macrofacet 实验配置参数参考

本文逐项说明 `ExperimentConfig` 及其子结构中的每个参数：JSON 键名、结构体字段、默认值、对应的数学含义与消费点。

数据流为：

```
configs/*.json  ──loadExperimentConfig──▶  ExperimentConfig  ──▶  run{Curves|ConditionalGPReference|RenderExperiments}
                                       │
                                       └──writeResolvedConfig──▶  resolved_config.json（派生量 + 实际预算）
```

结构体定义位于 `include/macrofacet/experiments/ExperimentConfig.h`，解析与写回位于 `src/experiments/ExperimentConfig.cpp`。

---

## 0. 符号与基础公式

全文反复引用以下记号。

### 0.1 统计曲面

曲面由高斯过程定义：

$$
F(x) = m(x) + G(x)
$$

其中 $G$ 是零均值高斯过程，协方差用平方指数（SE）核：

$$
k(x,y) = \sigma^2 \exp\left(-\tfrac{1}{2}(x-y)^\top P (x-y)\right),
\qquad
P = R \operatorname{diag}\left(\frac{1}{l_1^2}, \frac{1}{l_2^2}, \frac{1}{l_3^2}\right) R^\top
$$

实现中同时提供核的完整 jet（值、梯度、混合二阶导）。令 $\delta = x - y$、$v = P\delta$：

$$
k = \sigma^2 e^{-\frac{1}{2}\delta^\top v},
\qquad
\frac{\partial k}{\partial x} = -v k,
\qquad
\frac{\partial k}{\partial y} = +v k,
\qquad
\frac{\partial^2 k}{\partial x \partial y} = \left(P - v v^\top\right) k
$$

### 0.2 hazard 与透射率

整套代码的主线关系：

$$
h(t) = \frac{J(t)}{U(t)} \quad \text{（论文式 (29)）}
$$

$$
H(t) = \int_0^t h(s)\,ds \quad \text{（光学深度）},
\qquad
T(t) = \exp\left(-H(t)\right) \quad \text{（生存率 / 透射率）}
$$

由 hazard 的定义直接得到飞行距离的分布：

$$
\Pr\left[\text{age} \in [a,b]\right] = T(a) - T(b) = \int_a^b h(s) e^{-H(s)}\,ds
$$

### 0.3 条件化

`ConditionedRay` 观测到曲面通过出生点且带完整梯度：

$$
F(x_0) = 0, \qquad \nabla F(x_0) = g_0
$$

此后所有统计量（曲面命中的梯度分布、由此导出的 NDF）都是在此条件下的。

### 0.4 随机数

`Random` 提供两个均匀接口（`include/macrofacet/core/Random.h`）：

- `uniform01()`：$[0,1)$；
- `openUniform01()`：重抽直到严格落在 $(0,1)$ 内。

后者用于指数抽样，避免 $\log 0$：

$$
\text{target} = -\ln(1-u), \quad u \sim U(0,1) \;\Longrightarrow\; \text{target} \sim \mathrm{Exp}(1)
$$

---

## 1. `ExperimentConfig`（顶层）

| JSON | 字段 | 默认 | 说明 |
|---|---|---|---|
| `schema_version` | `schemaVersion` | `1` | 必须为 1，否则抛异常 |
| `seed` | `seed` | `17429` | 所有随机流的根种子 |
| `modes` | `modes` | `{Classic, Conditional29, Midpoint}` | 字符串 `classic` / `conditional29` / `midpoint` |
| `field` | `field` | — | 见 §2 |
| `fixed_flight` | `fixedFlight` | — | 见 §3 |
| `reference` | `reference` | — | 见 §4 |
| `render` | `render` | — | 见 §5 |
| `numeric` | `numeric` | — | 见 §6 |
| `output_directory` | `outputDirectory` | `outputs/macrofacet_experiments` | 可被 `--output` 覆盖 |
| `transport.beckmann_mixture_weight` | `beckmannMixtureWeight` | `0.5` | 见 §5.4 |
| `transport.classic_phase_proposal` | `classicPhaseProposal` | `uniform` | 见 §5.4 |

### 1.1 `seed` 的派生方式

同一根种子按不同用途派生，保证各实验流互不相关且可复现：

$$
\begin{aligned}
\text{曲线} &:\quad \text{seed} + \text{mode} \cdot 7919 \\
\text{GP 参考（嵌套网格）} &:\quad \text{seed} + \text{intervals} \\
\text{GP 参考（公式扫描）} &:\quad \text{seed} + 1000003 \cdot a + K \\
\text{渲染} &:\quad \text{seed} + \texttt{0x9e3779b97f4a7c15} \cdot (\text{pixelIndex}+1) + 104729 \cdot \text{mode}
\end{aligned}
$$

其中常数

$$
\texttt{0x9e3779b97f4a7c15} = 11400714819323198485 = \left\lfloor \frac{2^{64}}{\varphi} \right\rfloor
$$

为黄金比例常数。

渲染使用**逐像素独立种子**，因此多线程按行划分不会改变结果——图像与串行结果 bit-identical。

`resolved_config.json` 中记录 `"rng_stream": "per_pixel_v1"` 作为该约定的标识。

### 1.2 模型模式

| 模式 | 含义 |
|---|---|
| `classic` | 论文的去相关 Macrofacet 模型（A） |
| `conditional29` | 保留最近真实交点的 $F=0$ 与完整梯度，用式 (29) 的 crossing/exterior 比率作 hazard（B） |
| `midpoint` | 在同一条件统计中对分子分母同时加入一个中点正值筛选（C） |

---

## 2. `GPSSField`（JSON `field` 与 `material` 块）

### 2.1 `mean` — 均值场

均值场决定零水平集，也就是"表面"本身：曲面定义为 $F(x) = 0$。

| `mean_type` | 参数 | 公式 |
|---|---|---|
| `plane` | `plane_normal` $n$、`plane_offset` | $m(x) = n \cdot x - \text{offset}$（$n$ 在构造时归一化） |
| `sphere` | `sphere_center` $c$、`sphere_radius` $r$ | $m(x) = \lVert x - c \rVert - r$ |
| `constant` | `constant_value` | $m(x) = c$ |

配置中 $n = (0,0,1)$、`offset = 0`，故 $m(x) = z$：表面是 $z$ 平面，$F > 0$ 侧是外部（真空），$F < 0$ 侧是物质。

这解释了为什么需要约束飞行方向朝外：

$$
w \cdot g_0 > 0
$$

违反会在配置加载阶段直接抛异常。

### 2.2 `kernel` — 平方指数核

| JSON | 字段 | 含义 |
|---|---|---|
| `sigma` | $\sigma$ | 幅度，$\operatorname{Var}(G(x)) = \sigma^2$ |
| `correlation_lengths` | $l_1, l_2, l_3$ | 相关长度；取 `null` 表示该方向 $1/l^2 = 0$（无限相关） |
| `kernel_rotation` | $R$ | 正交右手旋转，须满足 $R^\top R = I$ 且 $\det R = 1$（容差 $10^{-10}$） |

三者共同给出：

$$
\operatorname{Cov}\left(G(x), G(y)\right) = \sigma^2 \exp\left(-\tfrac{1}{2}(x-y)^\top P (x-y)\right)
$$

配置值 $\sigma = 0.1$、$l = (0.2, 0.2, 0.2)$、$R = I$ 给出

$$
\sigma^2 = 0.01, \qquad P = 25\,I
$$

**构造期的数值处理**：精度矩阵先对称化 $\tfrac{1}{2}(P + P^\top)$，再作一次特征分解并把负特征值截断到 0。轻微非 PSD 会被静默修正，明显非 PSD 抛 `InvalidCovariance`。

#### `--sigma` 与 `--preserve-slope`

命令行覆盖 $\sigma$ 时的缩放规则：

$$
P_{\text{new}} = P_{\text{old}} \cdot \left(\frac{\sigma_{\text{old}}}{\sigma_{\text{new}}}\right)^2
$$

其依据是梯度协方差

$$
\operatorname{Cov}(\nabla G) = \sigma^2 P
$$

于是

$$
\sigma_{\text{new}}^2 P_{\text{new}}
= \sigma_{\text{new}}^2 P_{\text{old}} \left(\frac{\sigma_{\text{old}}}{\sigma_{\text{new}}}\right)^2
= \sigma_{\text{old}}^2 P_{\text{old}}
$$

即 `--preserve-slope` **恰好保持 $\sigma^2 P$ 不变**。含义是：只改变高度方差 $\sigma^2$，而把由梯度分布决定的 NDF 与坡度统计完全冻结。这是一个干净的消融开关。

### 2.3 `activeDomain`（`domain_min` / `domain_max`）

包围盒 $[\text{min}, \text{max}]$。射线求交使用 slab 法（`Bounds3::intersect`）：

$$
a_i = \frac{\text{min}_i - o_i}{d_i}, \quad
b_i = \frac{\text{max}_i - o_i}{d_i}, \quad
\text{lo} = \max_i \min(a_i, b_i), \quad
\text{hi} = \min_i \max(a_i, b_i)
$$

若 $\text{hi} < \text{lo}$ 则无交。域外固定为真空（JSON 中的 `outside_domain` 键**不被读取**）。

包围盒直接决定最大飞行距离：

$$
T_{\max}^{(\text{曲线})} = \min\left(\text{requestedMaximumAge},\ \text{kernel} \to \text{maximumAgeInDomain}()\right)
$$

$$
T_{\max}^{(\text{参考})} = \min\left(\text{requestedMaximumAge},\ \text{domain.exit}\right)
$$

配置 $[-2,-2,-0.3] \sim [2,2,0.3]$ 配合 $\sigma = 0.1$，$z$ 方向约 $\pm 3\sigma$，是刻意卡住 GP 的尾部范围。

### 2.4 `conductor`（JSON `material` 块）

| JSON | 字段 | 含义 |
|---|---|---|
| `eta_rgb` | $\eta$ | 复折射率实部 |
| `k_rgb` | $\kappa$ | 复折射率虚部（吸收） |
| `force_unit_fresnel_for_energy_test` | `forceUnitFresnel` | 置真时 Fresnel 恒为 1 |

导体 Fresnel，记 $\tilde\eta = \eta + i\kappa$、$c = |\cos\theta|$：

$$
\gamma = \sqrt{\tilde\eta^2 - \sin^2\theta}
$$

$$
r_s = \frac{c - \gamma}{c + \gamma},
\qquad
r_p = \frac{\tilde\eta^2 c - \gamma}{\tilde\eta^2 c + \gamma}
$$

$$
F(\theta) = \frac{1}{2}\left(|r_s|^2 + |r_p|^2\right)
$$

$c = 0$ 时取 $F = 1$。`forceUnitFresnel = true` 时 $F \equiv 1$，这是白炉（能量守恒）测试使用的开关。

### 2.5 `ndfFamily` 与 `ggxAlpha`

| 值 | NDF 来源 |
|---|---|
| `generalized_gaussian` | 由 $g_0$ 处的梯度高斯 $\mathcal{N}(g_0, \Sigma_G)$ 经"通量加权梯度 → NDF"映射得到（论文式 (36)–(38)） |
| `beckmann_limit` | 均值对齐 $Z$、协方差对角且 $\Sigma_{zz} = 0$ 时的闭式退化 |
| `ggx` | 独立高度场 $\mathrm{GgxHeightfield}(\alpha_x, \alpha_y)$；**仅 classic 模式支持**，其他模式会在加载期被拒绝 |

Beckmann 极限的闭式，令 $\alpha_x = \sqrt{2\Sigma_{xx}}$、$\alpha_y = \sqrt{2\Sigma_{yy}}$：

$$
D(n) = \frac{1}{\pi \alpha_x \alpha_y n_z^4}
\exp\left(
-\left(\frac{n_x}{\alpha_x}\right)^2 \frac{1}{n_z^2}
-\left(\frac{n_y}{\alpha_y}\right)^2 \frac{1}{n_z^2}
\right),
\qquad n_z > 0
$$

投影面积（G1 的分子）对 $K = w \cdot \nabla F \sim \mathcal{N}(w \cdot \mu_G,\ w^\top \Sigma_G w)$ 取负部均值：

$$
\int (w \cdot n)\, D(n)\, dn = \mathbb{E}\left[(-K)_+\right]
= \sigma_K\, \varphi\!\left(\frac{\mu_K}{\sigma_K}\right) + \mu_K\, \Phi\!\left(-\frac{\mu_K}{\sigma_K}\right)
$$

`ggx_alpha` 仅在 JSON 含该键时读取，用于 GGX 家族。

---

## 3. `FixedFlightConfig`（JSON `fixed_flight`）

这一组参数定义 A/B/C 三模型对比所用的**同一初始状态**。

| JSON | 字段 | 默认 | 含义 |
|---|---|---|---|
| `birth_position` | `birthPosition` | $(0,0,0)$ | 出生点 $x_0$，须落在 activeDomain 内（容差 $10^{-12}$） |
| `birth_gradient` | `birthGradient` | $(0,0,1)$ | $g_0 = \nabla F(x_0)$，条件化的观测量 |
| `direction` | `direction` | $(0.98481, 0, 0.17365)$ | 归一化后的 $w$ |
| `requested_maximum_age` | `requestedMaximumAge` | `1.5` | 请求的最大飞行距离 |
| `curve_sample_count` | `curveSampleCount` | `65` | 曲线采样点数 $N$，须 $\ge 2$ |
| `flight_sample_count` | `flightSampleCount` | `20000` | 直方图蒙特卡洛条数 $M$，须 $\ge 1$ |

默认方向即 $(\cos 10^\circ,\ 0,\ \sin 10^\circ)$，与 $z$ 轴夹角 $80^\circ$。

### 3.1 年龄网格

$$
t_i = T_{\max} \cdot \frac{i}{N-1}, \qquad i = 0, \dots, N-1
$$

### 3.2 逐点输出量

对每个 $t_i$ 求 hazard 并累加积分：

$$
h_i = \text{kernel.evaluate}(t_i).\text{hazard}
$$

$$
\Delta H_i = \int_{t_{i-1}}^{t_i} h(s)\,ds
\qquad \text{（自适应求积）}
$$

$$
H_i = H_{i-1} + \Delta H_i,
\qquad
T_i = e^{-H_i},
\qquad
T_{\text{model}} = h_i \cdot T_i
$$

`flight_curves.csv` 同时输出局部 exterior 筛选概率 $U_1$、crossing 通量、每步积分误差、数值状态与耗时。

### 3.3 直方图自洽性检验

抽取 $M$ 个指数变量，用累积光学深度做单调反演：

$$
\text{target} = -\ln(1-u), \quad u \sim U(0,1)
$$

$$
\text{target} \ge H_{\max} \;\Rightarrow\; \text{计为逃逸}
$$

$$
\text{target} \in \left[H_i, H_{i+1}\right) \;\Rightarrow\; \text{落入第 } i \text{ 格}
$$

期望质量由 §0.2 的恒等式给出：

$$
\mathbb{E}\left[\text{mass}_i\right] = T_i - T_{i+1} = e^{-H_i} - e^{-H_{i+1}}
$$

$$
\mathbb{E}\left[\text{escape}\right] = e^{-H_{\max}}
$$

观测值与期望值的对比即是对

$$
\Pr\left[\text{age} \in [a,b]\right] = T(a) - T(b)
$$

的数值检验。若 hazard 积分存在系统性错误，两者会分叉。

---

## 4. `ReferenceConfig`（JSON `reference`）

这一组控制 F27 数值参考——整套代码中唯一不依赖闭式公式的真值来源。

| JSON | 字段 | 默认 | 含义 |
|---|---|---|---|
| `path_sample_count` | `pathSampleCount` | `4096` | 蛮力联合采样条数 $N$ |
| `nested_grid_intervals` | `nestedGridIntervals` | $\{64,128,256\}$ | 嵌套网格分辨率 $M$ |
| `formula_checkpoint_counts` | `formulaCheckpointCounts` | $\{0,1,4,8,16,32\}$ | 检查点个数 $K$ |
| `formula_age_count` | `formulaAgeCount` | `12` | 目标年龄个数 $A$ |
| `formula_sample_count` | `formulaSampleCount` | `8192` | 公式估计的蒙特卡洛条数 $N$ |
| `confidence_level` | `confidenceLevel` | `0.95` | **解析后全项目无人使用**，见 §7 |
| `max_grid_points` | `maxGridPoints` | `512` | 过滤 $M > \text{maxGridPoints}$ 的网格 |

### 4.1 蛮力参考：`sampleConditionalGPFirstHit`

在网格 $t_i = T_{\max}(i+1)/M$ 上取条件 GP 的联合分布，一次性 Cholesky 分解后抽取 $N$ 条完整路径：

$$
\widehat{T}(t_i) = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\left\{F_j(s) > 0 \ \forall s \le t_i\right\}
$$

$$
\widehat{\text{mass}}(t_i) = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\left\{\text{首次穿零发生在 } t_i\right\}
$$

$$
\mathrm{SE}(t_i) = \sqrt{\frac{\widehat{T}(t_i)\left(1 - \widehat{T}(t_i)\right)}{N}}
\qquad \text{（二项标准误）}
$$

区间平均 hazard 由光学深度反演得到：

$$
\bar{h}(t_i) = -\frac{\ln\left(\widehat{T}(t_i) / \widehat{T}(t_{i-1})\right)}{t_i - t_{i-1}}
$$

网格 $64 \to 128 \to 256$ 加密用于观察 $\widehat{T}$ 是否收敛：网格越密，逐点条件 $F(s) > 0\ \forall s$ 才越接近真正的"整条路径不穿零"。

### 4.2 筛选公式估计：`estimateScreenedFirstPassageHazard`

构造 $(K+2)$ 维联合高斯

$$
\text{joint} = \left(s_1, \dots, s_K,\ F(t),\ F'(t)\right)
$$

检查点位置为严格内点，目标年龄按等分取：

$$
s_k = \text{age} \cdot \frac{k+1}{K+1}, \quad k = 0,\dots,K-1
\qquad
\text{age} = T_{\max} \cdot \frac{a}{A}, \quad a = 1,\dots,A
$$

#### 分母 $U_N$（exterior 概率）

对 $\left(s_1,\dots,s_K, F(t)\right)$ 的联合分布作蒙特卡洛，全部为正才算存活：

$$
U_N = \Pr\left[F(s_k) > 0\ \forall k,\ F(t) > 0\right]
$$

$$
\delta U = \sqrt{\frac{U_N\left(1 - U_N\right)}{N}}
$$

#### 分子 $J_N$（crossing 通量）

条件在 $F(t) = 0$ 上，使用 $\left(s_1,\dots,s_K, F'(t)\right) \mid F(t) = 0$：

$$
J_N = p_{F(t)}(0) \cdot \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\{\text{screening}\} \cdot \max\left(-F'_j(t),\ 0\right)
$$

$$
\delta J = p_{F(t)}(0) \cdot \sqrt{\frac{\operatorname{Var}(w)}{N}},
\qquad w = \mathbf{1}\{\text{screening}\} \cdot \left(-F'(t)\right)_+
$$

其中 $p_{F(t)}(0)$ 是 $F(t)$ 的一维正态密度在原点处的取值，$\text{screening}$ 表示全部检查点为正。当 $K = 0$ 时筛选条件为空、恒真，公式退化为纯 Rice 形式——这正是 $\{0,1,4,\dots\}$ 中 `0` 的用途。

#### 合并

$$
h_N = \frac{J_N}{U_N}
$$

$$
\frac{\delta h}{h} = \sqrt{\left(\frac{\delta J}{J}\right)^2 + \left(\frac{\delta U}{U}\right)^2}
\qquad \text{（相对误差传播）}
$$

当 $U_N = 0$（无样本通过筛选）时返回 `status = unresolved_rare_event` 且 $h = \mathrm{NaN}$——这是稀有事件未被抽中的诚实标记，而非零值。

**输出**：`screened_formula.csv` 是一张 $(\text{age}, K)$ 的二维收敛表，$K: 0 \to 32$ 应逐步逼近 §4.1 的真值。

---

## 5. `RenderConfig`（JSON `render`）

| JSON | 字段 | 默认 | 含义 |
|---|---|---|---|
| `width` / `height` | `width` / `height` | $64 \times 64$ | 分辨率 |
| `samples_per_pixel` | `samplesPerPixel` | `256` | 每像素路径数 |
| `camera_position` | `cameraPosition` | $(0,0,1)$ | 相机位置 |
| `camera_target` | `cameraTarget` | $(0,0,0)$ | 注视点 |
| `vertical_fov_degrees` | `verticalFovDegrees` | `45` | 垂直视场角（度） |
| `environment` | `environment` | `directional_gradient` | 环境光模型 |
| `flight_table_cells` | `flightTableCells` | `48` | 分段 hazard 表格数，$\ge 4$ |
| `roulette_start_depth` | `rouletteStartDepth` | `5` | 俄罗斯轮盘起始深度 |
| `thread_count` | `threadCount` | `0` | $0$ 表示硬件并发 |
| — | `safetyDepthCap` | `64` | **JSON 无法设置**，见 §7 |
| `linear_output_format` | — | — | **不被读取**，恒定输出 PFM |

### 5.1 相机

$$
\text{forward} = \operatorname{normalize}\left(\text{target} - \text{position}\right)
$$

$$
\text{scale} = \tan\left(\frac{\text{fov}}{2}\right)
\qquad \left(\text{fov} = 45^\circ \Rightarrow \text{scale} = \tan 22.5^\circ\right)
$$

对像素内抖动 $(u,v) \in (0,1)^2$：

$$
p_x = \left(2u - 1\right) \cdot \text{aspect} \cdot \text{scale},
\qquad
p_y = \left(1 - 2v\right) \cdot \text{scale}
$$

$$
d = \operatorname{normalize}\left(\text{forward} + p_x \cdot \text{right} + p_y \cdot \text{up}\right)
$$

抖动 $(u,v)$ 提供抗锯齿。正交基由 `orthonormalComplement` 对 `forward` 与 `UnitZ`（或 `UnitX`）叉乘得到。

### 5.2 环境光

$$
\text{unit\_white}:\quad L(w) = (1, 1, 1)
$$

$$
\text{directional\_gradient}:\quad \text{up} = \frac{w_z + 1}{2},
\qquad
L(w) = \left(0.15 + 0.85\,\text{up},\ 0.2 + 0.7\,\text{up},\ 0.35 + 0.55\,\text{up}\right)
$$

`unit_white` 会强制 `forceUnitFresnel = true` 以运行白炉测试：理想情况下每个像素应正好收敛到 1，任何偏差即能量泄漏，`render_summary.csv` 的各统计列用于定位泄漏来源。

### 5.3 `flightTableCells` — 分段 hazard 表

渲染不使用解析的 $T(t)$，而是把区间 $[a, b] = [\text{currentAge}, \text{maxAge}]$ 切成 $C$ 格：

$$
\text{边界}_i = a + (b - a)\frac{i}{C}, \qquad
\Delta H_i = \int_{\text{cell}_i} h(s)\,ds
$$

抽样时先取指数目标，再找累积深度首次超过目标的格子：

$$
\text{target} = -\ln(1-u),
\qquad
\text{fraction} = \frac{\text{target} - \text{acc}}{\Delta H_i}
$$

$$
\text{age} = \text{lower} + \text{fraction} \cdot \left(\text{upper} - \text{lower}\right)
$$

$$
\log p_{\text{distance}} = \log\left(\frac{\Delta H_i}{\Delta t_i}\right) - \text{target}
$$

即**返回的是分段常数 hazard**。若 $\text{target}$ 超过总深度则逃逸，返回生存率 $e^{-\text{总深度}}$。

这是明确的渲染近似：抽样与返回 PDF 使用同一张表，因此两者自洽、无偏；但表格本身的离散化误差随 $C$ 减小而增大。曲线与数学测试仍直接积分原 hazard，不经过这张表。

### 5.4 `classicPhaseProposal` 与 `beckmannMixtureWeight`

仅对 `classic` 模式生效。混合 proposal（论文式 (40)）：

$$
q(n) = (1 - w)\, q_{\text{uniform}}(n) + w\, q_{\text{beckmann}}(n)
$$

$$
q_{\text{uniform}}(n) =
\begin{cases}
\dfrac{1}{2\pi}, & -w \cdot n > 0 \\[2mm]
0, & \text{否则}
\end{cases}
$$

$$
q_{\text{beckmann}}(n) = \frac{\cos\theta\, D(n)}{\mathbb{E}\left[(-w \cdot \nabla F)_+\right]}
\qquad \text{（可见法线分布）}
$$

| 值 | 行为 |
|---|---|
| `uniform` | $w = 0$，纯均匀半球 proposal |
| `paper_mixture` | $w = \text{beckmannMixtureWeight}$，以概率 $w$ 抽取 Beckmann |
| `target_vndf` | 直接取 $q = \cos\theta\, D(n) / \mathbb{E}[(-K)_+]$；**要求非 GGX**，否则加载期报错 |

法线 PDF 转立体角需要 Jacobian：

$$
q(\omega') = \frac{q(n)}{4\,|w \cdot n|}
$$

重要性采样权重：

$$
\text{throughputWeight} = \frac{F \cdot D(n)}{q(n)}
\;\equiv\; \frac{f(\omega, \omega')}{q(\omega')}
$$

`beckmannMixtureWeight` 被约束在 $[0, 1)$ 内（构造期检查）。

### 5.5 `rouletteStartDepth` 与 `safetyDepthCap`

俄罗斯轮盘在 $\text{depth} + 1 \ge \text{rouletteStartDepth}$ 后启用：

$$
q = \operatorname{clamp}\left(\max_{\text{channel}}\left(\text{throughput}\right),\ 0.05,\ 0.95\right)
$$

$$
u \ge q \;\Rightarrow\; \text{终止（贡献 } 0\text{）},
\qquad
u < q \;\Rightarrow\; \text{throughput} \leftarrow \text{throughput} / q
$$

无偏性：

$$
\mathbb{E}\left[\text{贡献}\right] = q \cdot \frac{L}{q} + (1 - q) \cdot 0 = L
$$

`clamp` 的上下限保证既不会必然存活，也几乎不会必然终止。

`safetyDepthCap = 64` 为硬上限，超出后计入 `safetyCapTerminations` 并返回 0。该字段**无法从 JSON 配置**（见 §7）。

### 5.6 `threadCount`

$$
\text{threadCount} > 0 :\quad \text{workers} = \operatorname{clamp}\left(\text{threadCount},\ 1,\ \text{height}\right)
$$

$$
\text{threadCount} = 0 :\quad \text{workers} = \operatorname{clamp}\left(\text{hardware\_concurrency}(),\ 1,\ \text{height}\right)
$$

采用**动态行分配**（`std::atomic` 的 fetch-add），因为 correlated 模式下每行开销相差数个数量级，静态分块会让 worker 长时间空转。

由于种子只依赖像素索引，结果与线程数完全无关。

---

## 6. `NumericPolicy`（JSON `numeric`）

| JSON | 字段 | 默认 | 含义 |
|---|---|---|---|
| `relative_tolerance` | `relativeTolerance` | `1e-7` | 自适应求积的相对容差 |
| `absolute_tolerance` | `absoluteTolerance` | `1e-10` | 绝对容差 |
| `max_quadrature_subdivisions` | `maxQuadratureSubdivisions` | `4096` | 最大二分细分次数 |
| `max_root_iterations` | `maxRootIterations` | `128` | 单调反演最大迭代次数 |
| `higher_precision_fallback` | `allowHigherPrecisionFallback` | `true` | 精度兜底开关 |
| `allow_unreported_jitter` | — | — | 传 `true` **直接抛异常** |
| — | `covarianceRoundoffMultiplier` | `128` | **JSON 无法设置**，恒为 128 |

### 6.1 求积容差

采用 G7/K15 自适应求积。当子区间上两个估计满足

$$
\left|I_{K15} - I_{G7}\right| \le \max\left(\text{atol},\ \text{rtol} \cdot \left|I_{K15}\right|\right)
$$

时接受该子区间，否则继续二分，直到 `maxQuadratureSubdivisions`。因此 `rtol` 控制相对精度，`atol` 为接近零的被积函数兜底。

### 6.2 舍入容差

记 $\varepsilon$ 为双精度机器 epsilon：

$$
\varepsilon = 2.22 \times 10^{-16}
$$

$$
\text{covarianceTolerance}(\text{scale}) = 128 \cdot \varepsilon \cdot \max\left(1,\ |\text{scale}|\right)
$$

负值只要落在该容差内即视为 0，超出则抛 `InvalidInput`。该常数被精度矩阵的 PSD 检查、GaussianNdf 的对角性判定等多处复用。

### 6.3 `allow_unreported_jitter` 的设计意图

传 `true` 会抛异常，这是一个**刻意的设计声明**：PSD 条件化使用标准化后的特征空间伪逆，确定观测会检查支持一致性，不依赖给协方差加 jitter 来掩盖病态。该约定同时记录在实现报告中。

---

## 7. 解析但不生效 / 完全不读取的参数

以下条目在配置文件中存在或存在于结构体中，但不影响任何行为。请确认是有意保留还是实现遗漏。

| 位置 | 情况 |
|---|---|
| `confidence_level` | 解析进 `config.reference.confidenceLevel`，**全项目无人读取**。规格书 `macrofacet_implementation_spec.md` 第 1692 行要求 "attach confidence intervals and zero-count diagnostics"，但实现只输出 $\pm \mathrm{SE}$ 与 `unresolved_rare_event` 状态 |
| `hard_depth_cap` | JSON 中存在该键，但代码从不解析，故 `safetyDepthCap` 恒为默认值 64 |
| `covarianceRoundoffMultiplier` | 无对应 JSON 键，恒为 128 |
| `outside_domain` | 不读取，行为固定为真空 |
| `classic_sampler` | 不读取 |
| `correlated_sampler` | 不读取 |
| `next_event_estimation` | 不读取 |
| `material.type` | 不读取（只读取 eta / kappa / force_unit_fresnel） |
| `linear_output_format` | 不读取，恒定输出 PFM |
| `plane_normal` | 非单位向量会被静默归一化，不报错 |

此外，`GPSSField::pointPrior` 结构体带默认值，但从配置加载路径始终传入 `{}`，即恒为

$$
\text{meanF} = 0,\quad \text{varianceF} = 0,\quad \text{meanG} = 0,\quad \text{covarianceG} = 0
$$

真正的 $\text{meanG}$ 与 $\text{covarianceG}$ 由 `pointPrior(x)` 在运行时依据条件射线计算。

---

## 附录：参数消费点速查

| 参数 | 主要消费文件 |
|---|---|
| `seed`, `modes`, `outputDirectory` | `src/experiments/*.cpp` |
| `field.*` | `src/gpss/`, `src/macrofacet/`, `src/transport/` |
| `fixedFlight.*` | `src/experiments/FlightCurveExperiment.cpp`, `ConditionalGPReference.cpp` |
| `reference.*` | `src/experiments/ConditionalGPReference.cpp` |
| `render.*` | `src/integrator/MacrofacetPathTracer.cpp`, `src/experiments/RenderExperiment.cpp` |
| `numeric.*` | `include/macrofacet/mathutility/NumericPolicy.h` 及各求积/求根调用点 |
| `beckmannMixtureWeight`, `classicPhaseProposal` | `src/macrofacet/ConductorPhase.cpp` |

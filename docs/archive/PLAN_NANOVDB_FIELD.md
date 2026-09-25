# NanoVDB 模型场：统一输入 + 独立预计算 target

> 历史实施计划。当前实验入口已经要求 NVDB，解析场会先烘焙，三个 mode
> 均使用连续光学深度反演。实际操作见 [当前 tracing 文档](../NVDB_TRACING.md)。

目标：把原 `src/pbrt/cmd/macrofcaet_vdb_generator.cpp`（**该文件已从工作树删除**，
算法已完整转录到 §2，本文档即唯一参考）的预计算逻辑在本项目内重新实现，
产出一个 NanoVDB 文件（`density` / `alpha` / `sdf` 三个网格），并让这个文件与现有的
程序化场（`PlaneMean` / `SphereMean` / `CutawaySphereMean` / `ShaderBallMean` / `ConstantMean`）
走**同一套渲染输入接口**。全程不使用 pbrt，libigl 可用。

---

## 0. 先回答两个问题

### 0.1 渲染器为什么区分内外，能不能改成传统体渲染

现有的 GP 传输（`ClassicFlightKernel` / `Conditional29FlightKernel` / `MidpointFlightKernel`）
追踪的不是消光，而是**风险率**

```
h(t) = J(t) / U(t)          // 论文式 29，J = 命中密度，U = 存活概率
```

`mean(x) > 0` 表示外部，是因为 `U(t)` 用到均值场的符号来判定"表面还没被穿过"。
这**不是**体渲染的约定，而是统计表面模型的约定：`h(t)` 是"射线在 t 处第一次撞到随机表面"
的条件密度。

**可以改，但不能替换。** 见 0.2。

### 0.2 conditional29 为什么"不能"用 volume 模式

先纠正一个前提：**conditional29 已经在做非指数追踪了**，它不是先算 `∫h` 再取 `e^{-H}`。

`src/transport/Conditional29FlightKernel.cpp` 的 `evaluate(age)` 逐段算风险率
（`ray_.endpointValueSlope(age)` → `normalLogCdf` → `negativePartMean` → `logRho`），
再由 `src/transport/OpticalDepthSampler.cpp` 对 `∫h` 求逆采样碰撞距离。
所以"volume = 非指数追踪"这个等式不成立，两者不是一回事。

真正的分界线是：**`h(t)` 是不是一个局部量（只依赖 x）。**

| | `h(t)` 依赖什么 | 能否写成 `σ_t(x)` |
|---|---|---|
| classic / conditional29 / midpoint | 出生状态 `(f₀, g₀)`、核 `(σ, P)`、以及沿射线的 `m` | ❌ 依赖射线起点 |
| density 网格 | 逐体素值 | ✅ 是局部量 |

同一点 `x`、不同射线起点，`h` 不同——这正是条件分布的定义。要把它塞进 volume 模式，
只能丢掉"条件"，那就退化成解相关模型 A 了，渲染出来的东西不再是 conditional29。

而 `density` 网格**本来就是**逐体素的**边缘**密度（把确定性网格曲面按 `φ(d;0,σ)/Φ(3)`
抹开），它天然就是 `σ_t(x)`，volume 模式可以直接吃。

**结论**：conditional29 用 **sdf 场**就够了，不需要 σ_t。传输层实际用到的
`MeanField` 接口只有四个方法：

```
evaluate(x)              → ConditionedRay.cpp:56, 89, 152, 208
affineGradient()         → ConditionedRay.cpp:157
valueDifference(x0, t·w) → ConditionedRay.cpp:158
bounds(domain)           → ClassicCoefficients.cpp:41, OpticalDepthSampler.cpp:21
```

`NanoVdbMean` 实现这四个即可，**传输层一行不改**。`ConditionedRay.cpp:157`
本来就写好了非仿射分支：

```cpp
const double meanRemainder = field_.mean->affineGradient() ? 0.0 :
    field_.mean->valueDifference(x0_, t * w_) - t * w_.dot(originMean_.gradient);
```

解析场走短路，采样场走两次查表——这个接口设计本来就是为采样场留的。

### 0.3 那 NVDB sdf 和程序化 sdf 不是一样的吗

在渲染器眼里**完全一样**，两者都实现 `MeanField`，都提供 `m(x)` 和 `∇m(x)`。
差别全在逼近质量上，而且都可测：

| | 程序化 sdf | NVDB sdf |
|---|---|---|
| 值 | 解析精确 | 三线性，误差 `O(dx²)` |
| 梯度 | 解析（`SphereMean` 是精确单位梯度） | 中心差分，体素面之间不连续（C⁰ 而非 C¹） |
| `valueDifference` | 有的场有闭式（`SphereMean` 用弦长公式） | 两次查表 |
| `bounds()` | 可证明（`maximumGradientNorm = 1.0`，`certified = true`） | **必须实测**，不能假设 =1 |

最后一条是唯一需要动手的：`maximumGradientNorm` 直接进
`OpticalDepthSampler.cpp:21` 当安全上限，填错会静默出错。

**NVDB sdf 的价值不在"能渲染"，在于"能烘焙任意网格"**——程序化场只能给解析形状，
要渲染 PLY 模型就必须走这条路。

```
NVDB 文件 ──→ sdf 网格  ──→ NanoVdbMean ──→ 现有 GP 传输（零改动）
             alpha 网格 ──→ 逐点 roughness ──→ NDF（见 §1.4）
             density 网格 → 无渲染消费者（见 §2.5）
```

volume 模式（原 P3）**不做**。

---

## 1. 统一合约

一个"场" = **均值场**（已有接口） + **采样场**（新增）。

### 1.1 已有：`MeanField`（不动）

`include/macrofacet/gpss/MeanField.h:20-30`

```cpp
class MeanField {
  virtual MeanJet evaluate(const Point3& x) const = 0;            // 值 + 梯度
  virtual double valueDifference(const Point3&, const Vector3&) const;
  virtual BoundsSummary bounds(const Bounds3& domain) const = 0;  // 必须保守
  virtual std::optional<Vector3> affineGradient() const;
};
```

程序化场和 NVDB 场都实现它 → `sdf` 网格就是 `mean`。

### 1.2 新增：逐点材质场

`include/macrofacet/fields/ScalarField.h`（新文件）

```cpp
namespace mf {

struct ScalarBounds {
  double minimumValue = 0.0;
  double maximumValue = 0.0;
  bool certified = false;
};

// 标量场：alpha 网格 → 逐点 roughness
class ScalarField {
public:
  virtual ~ScalarField() = default;
  virtual double sample(const Point3& x) const = 0;
  virtual ScalarBounds bounds(const Bounds3& domain) const = 0;  // 必须保守（供 majorant）
};

using ScalarFieldPtr = std::shared_ptr<const ScalarField>;

} // namespace mf
```

`bounds()` 与 `MeanField::bounds` 同形，因为它在同一处被消费：`classicMajorant`
需要一个**域内最大值**（见 §1.4 D，GGX 分支）。`NanoVdbSampledField` 在装载时扫一遍
active 体素填实测值，`certified = true`。

**关于 `sdf` 网格**：它不实现新接口，直接实现 `MeanField`（§1.1）——
这就是"统一"的全部含义。程序化场和烘焙场在渲染器眼里是同一个东西。

### 1.3 `GPSSField` 扩容（向后兼容）

`include/macrofacet/gpss/GPSSField.h:25-35`

```cpp
struct GPSSField {
  MeanFieldPtr mean;
  SquaredExponentialKernel kernel;
  Bounds3 activeDomain;
  ConductorParameters conductor;
  NdfFamily ndfFamily = NdfFamily::GeneralizedGaussian;
  Vector2 ggxAlpha = Vector2(0.5, 0.5);          // 保留：全局默认

  // 新增（为空则退回原行为，现有所有测试不受影响）
  ScalarFieldPtr alphaField;    // alpha 网格 → 逐点 roughness
};
```

**`pointPrior` 不动**，另开一个访问器。理由是 `covarianceG` 今天是**一个量担两个角色**：

| | 含义 | 谁在读 | 该用哪个协方差 |
|---|---|---|---|
| **角色 1：观测/传输** | 随机场 `G` 在 `x` 处的**梯度协方差** | `ConditionedRay.cpp:62`（`g₀` 的观测协方差）、`FlightState.cpp:42`（外场梯度采样）、`ClassicCoefficients.cpp:25`（消光的投影面积） | **`σ²P`**——这是 GP 的量，与材质无关 |
| **角色 2：材质 NDF** | 着色法线分布 = 同一分布的另一种消费 | `ConductorPhase.cpp:41,49,70,87,96`、`ClassicFlightKernel.cpp:22` | `alpha(x)²I` |

两者今天恒等（都是 `σ²P`），但语义不同。**把 alpha 塞进 `pointPrior` 会连带改掉角色 1**
——也就是静默改掉三个 mode 的传输统计，正好违反下面 §1.4 A 的"传输一行不改"。
所以拆成两个：

```cpp
struct GPSSField {
  ...
  PointPrior pointPrior(const Point3& x) const;    // 传输；σ²P；一行不改
  PointPrior materialNdf(const Point3& x) const;   // 材质；alpha(x)²I；无网格时回退 σ²P
};
```

```cpp
PointPrior GPSSField::pointPrior(const Point3& x) const {          // ← 保持现状
    const MeanJet jet = mean->evaluate(x);
    const double variance = kernel.sigma() * kernel.sigma();
    return {jet.value, variance, jet.gradient, variance * kernel.precision(), Vector3::Zero()};
}

PointPrior GPSSField::materialNdf(const Point3& x) const {         // 新增：只换协方差
    PointPrior prior = pointPrior(x);
    if (alphaField) {
        const double a = alphaField->sample(x);
        prior.covarianceG = (a * a) * Matrix3::Identity();
    }
    return prior;
}
```

**这个拆分同时消掉了"射线起点在带外"的问题**：唯一会被 `x0_` 调用的访问器是
`pointPrior`（角色 1），它永远返回非奇异的 `σ²P`。alpha 只出现在角色 2，
而角色 2 只在**碰撞点/着色点**被读——那里 `ρ > 0`，必在网格内
（最远到带边那一层，即 §2.3 点 2 的残差层）。起点根本碰不到 alpha，
所以既不需要移动起点，也不需要靠 background 防奇异（§2.3 点 5）。

**换算关系（以现有代码为准，不要凭直觉写）**：`ExperimentConfig.cpp:39-63` 的
`roughnessKernel` 定下 `roughness = σ_G`：

```
gradientVariance = roughness²          // :46
correlationLength = σ / roughness      // :47
precision P = (roughness/σ)²           // :54-55
covarianceG  = σ² · P = roughness²     // :56 断言的就是这个
```

即 **`covarianceG = roughness²·I`**，`test_material_roughness.cpp:80` 也是这么断言的
（`prior.covarianceG ≈ roughness²·I`），`ExperimentConfig.cpp:330` 的
`roughness_definition` 写的就是 `Sigma_G = roughness^2 I; correlation length = sigma / roughness`。
所以逐点化就是 `a(x) = alpha(x)`，**不带 σ 因子**。

`GPSSField::validate()`（`GPSSField.cpp:15-23`，现在只查 `ggxAlpha > 0`）要加一条：
`alphaField` 存在时 `bounds(activeDomain).minimumValue > 0`。理由是
`GaussianNdf` 的构造里有一个 `Eigen::LLT<Matrix3>`（`GaussianNdf.h:21`），
`covarianceG` 奇异会在 `materialNdf` 的**着色**调用点直接炸。
（这是**下界检查**，防的是网格里出现 0/负值，不是防起点——起点不读 alpha。）

**关键确认**：`ConductorPhase` 不依赖法线方向场。它只用 `field_.ndfFamily`、`ggxAlpha`、
和 `materialNdf(position_)`（即 `∇sdf` + 逐点 roughness）。所以只要 `NanoVdbMean`
给出 `∇sdf`，相位函数与现有解析场**行为一致**。相位函数的公式一行不改——
要动的只有五处协方差的**来源**（§1.4 B/C）。

### 1.4 适配矩阵：三网格 × 三 mode

| 网格 | classic | conditional29 | midpoint | 写入范围 | background |
|---|---|---|---|---|---|
| `sdf` | ✅ | ✅ | ✅ | ±3σ 带 | **+6σ** |
| `alpha` | ✅ | ✅ | ✅ | ±3σ 带 | **α**（不是 0，见下） |
| `density` | ❌ | ❌ | ❌ | ±3σ 带 | 0（无消费者） |

**`alpha` 的语义：逐点 roughness**（已定）。生成器写进 `alpha` 网格的是
**常数** α（那个 `intrAlpha = Lerp(...)` 被注释掉了），所以逐体素性现在还没被用上，
但接口按逐体素设计。

全部消费者已逐个核对，按上面两个访问器分四类：

**A — 角色 1，继续用 `pointPrior`（`σ²P`），一行不改**
条件化的代数直接读 `kernel.precision()`：`ConditionedRay.cpp:50,204,244`、
`OpticalDepthSampler.cpp:18`、`Conditional29FlightKernel.cpp:14`；
观测模型走 `pointPrior`：`ConditionedRay.cpp:62`（起点 `g₀` 的协方差）、
`FlightState.cpp:42`（外场梯度采样）；再加上经典消光的投影面积
`ClassicCoefficients.cpp:25`。

→ **后果要写清楚**：alpha 网格调制的是**着色**的梯度协方差，不是随机场的相关长度，
也不是传输统计。用户若期待"`alpha(x)` 同时意味着相关长度 `σ/alpha(x)`"，
那需要逐点的 `P` 进条件化代数，本方案不提供。`material.roughness`（config）
与 `alpha`（网格）因此是两个独立的旋钮。

顺带一个有用的推论：`evaluateClassic` 用的全是点局部量（`ClassicCoefficients.cpp:11-36`），
所以**经典模式的 hazard 与射线起点无关**。

**B — 角色 2，改用 `materialNdf`（`alpha(x)²I`）**
`ConductorPhase.cpp:41`（`targetD`）、`:49`（`targetArea`）、`:87`（`sampleTargetVisible`）、
`ClassicFlightKernel.cpp:22`（碰撞梯度 → 着色法线）。

**C — `ConductorPhase` 的两个提议分布（必须跟随目标）**
`:70`（`sampleBeckmann`）与 `:96`（`proposalNormalPdf`）现在从**全局** `σ²·kernel.precision()`
造 Beckmann 提议，而目标（B）走 `materialNdf`。今天两者恒等；alpha 逐点后就错配——
**仍然无偏**（`:139` 的权重是 `target/qn`，是纯重要性采样而非 MIS），
但 `alpha(x)` 偏离 config 时方差上升。
→ 一并改走 `materialNdf(position_)`；无 `alphaField` 时**逐位相同**，不需要分支。

**D — GGX 分支：唯一需要"域内最大值"的地方**
GGXBaseline 下 `covarianceG` 根本不被读（`ConductorPhase.cpp:37-38,45-46`、
`ClassicCoefficients.cpp:19-20` 直接走 `GgxHeightfield(ggxAlpha)`）。
让 alpha 网格同时驱动它（`ggxAlpha := (alpha(x), alpha(x))`，标量提升为各向同性）
与 B 不冲突，两条分支各自取用。于是 A 方案（字面 `ggxAlpha`）成为 B 的特例，
GGX 也不再是"读不到 alpha 的孤岛"。`ggxAlpha` 保留为无网格时的全局回退。

**但 GGX 是 alpha 唯一会进消光的地方**（`ClassicCoefficients.cpp:20` 的投影面积
直接喂给 extinction），所以 `classicMajorant`（`:49-53`）必须取
`alphaField->bounds(domain).maximumValue`——否则 alpha 局部大于 config 值时
`areaMaximum` 偏小、不再是上界，追踪可能漏碰。**这是全部改动里唯一影响正确性的一处**；
经典/GP 家族不受影响，因为它们的消光走角色 1（A）。

**没有任何一个 mode 需要 `density`**：三个 mode 读的场完全一样，用的是同一组量：

| 量 | 来源 | 谁在用 |
|---|---|---|
| `m(x)`, `∇m(x)` | `sdf` 网格 → `NanoVdbMean` | `ConditionedRay.cpp:152, 208`、`GPSSField.cpp:8` |
| `valueDifference` | 同上（两次查表） | `ConditionedRay.cpp:158` |
| `bounds(domain)` | 同上（全域扫描） | `ClassicCoefficients.cpp:41`、`OpticalDepthSampler.cpp:21` |
| `σ` | 文件 metadata / config | 全部 |
| `P`（精度矩阵） | **config**（`correlation_lengths` 或 `material.roughness`） | 全部 |
| `alpha` | `alpha` 网格 | `ConductorPhase.cpp:36-56` |

三个 mode 的区别在**统计机制**，不在输入数据：
classic 只用边缘（`ClassicCoefficients.cpp:11-36`）；
conditional29 以出生值 `f₀` 和梯度 `g₀` 为条件；
midpoint 以中点条件 + 外部屏（`enableScreen=false` 时**直接退化成 conditional29**，
见 `MidpointFlightKernel.cpp:21, 96`）。

> **为什么三者共用同一个 `m` 是自洽的**：hazard 里的 `Φ(z)` 分母是存活项，
> 与采样分布相乘后
> `h(t)·T(t) = φ(z)/(σΦ(z)) · Φ(z) = φ(z)/σ`，
> 即 **`m(x(t)) ~ N(0, σ)`**——碰撞点就是"以均值场为中心、σ 为标准差的高斯表面"。
> 这条对三个 mode 都成立（conditional29 / midpoint 只是把 `N(0,σ)` 换成条件版本），
> 所以它们对 `m` 的需求完全一致。也解释了为什么 `±3σ` 带捕获 99.73% 的碰撞。

**所以一份 NVDB 文件三个 mode 都能跑**，`P` 是 config 侧的模型旋钮
（`sdf`/`alpha` 是几何与材质数据）。唯一跨界的量是 `σ`：在程序化场里它是模型旋钮，
在烘焙场里它是**数据的一部分**——因此两套 schema 分开，烘焙场的 σ 由文件供给
（§2.4 的 config schema 分家；回归断言在 §6 P2 验收 4）。

### 1.5 配置文件：新增 `mean_type`

`src/experiments/ExperimentConfig.cpp:125-145` 加一个分支：

```cpp
} else if (meanType == "nanovdb") {
    result.field.mean = makeNanoVdbMean(requirePath(fieldJson, "grid_file"),
                                       fieldJson.value("sigma", 0.0));  // σ 从文件读，允许覆盖
}
```

`writeResolvedConfig`（`src/experiments/ExperimentConfig.cpp:302-320`）同步加一支，
让 resolved config 能往返。

### 1.6 注册表钩子（核心库不依赖 NanoVDB）

`macrofacet` 核心库**不链接** nanovdb。新增一个小注册表：

```cpp
// include/macrofacet/fields/MeanFactory.h
using MeanFactoryFn = std::function<MeanFieldPtr(const nlohmann::json&)>;
void registerMeanFactory(const std::string& type, MeanFactoryFn fn);
MeanFieldPtr makeMeanFromJson(const nlohmann::json& fieldJson);  // 查表，未知类型报错
```

核心库内置 `plane/sphere/cutaway_sphere/shader_ball/constant`；
`macrofacet_field` 库在静态初始化时注册 `"nanovdb"`。
这样核心库保持零外部依赖，链接 `macrofacet_field` 的可执行文件才拿到 NVDB 能力。

---

## 2. 预计算规格（原 `macrofcaet_vdb_generator.cpp` 的逐句复刻）

> 原文件已从工作树删除，本节是它的唯一留存记录，也是 P1 的实现依据。
> 除 §2.3 点 2 的 `background` 外，全部逐句照搬。

### 2.1 输入 / 输出 / 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `<mesh.ply>` | 必填 | 输入网格 |
| `--sigma` | 0.05 | 表面位置标准差 |
| `--alpha` | 0.5 | 材质 NDF 参数 |
| `--x/--y/--z` | 32/32/32 | 各轴体素数 |
| `--out` | out.nvdb | 输出（新增；原来是硬编码） |
| `--sign-mode` | `winding` | 新增可选：`winding`(复刻，`√sqrD·(1−2|w|)`) / `threshold`(`sign(w<0.5)·√sqrD`，模长不被绕数误差污染；非闭合网格用) |
| `--threads` | 硬件并发数 | 新增可选：烘焙是多线程的（§3.5）；`1` 可复现单线程时序 |

### 2.2 步骤

```
1. 读 PLY  → V (n×3), F (m×3)
2. bbox = V 的 AABB
3. gap = bbox 对角线；bbox 各方向扩 3σ
4. dx = min(gap.x/xRes, gap.y/yRes, gap.z/zRes)      // 单一体素尺寸
5. 建立 igl::AABB<V,F> 与 igl::FastWindingNumberBVH(order=2)
6. 遍历体素 ijk ∈ [Floor(worldToIndex(bbox.min)), Floor(worldToIndex(bbox.max))]：
     P    = indexToWorld(ijk)
     sqrD = tree.squared_distance(V,F,P)              // 无符号距离²
     w    = igl::fast_winding_number(bvh, 2, P)       // ∈ [-1,1]
     dist = sqrt(sqrD) * (1 - 2|w|)                   // 符号翻转

     if (-3σ < dist && dist < 3σ):                    // ★ 三网格共用同一套 mask
         sdf[ijk]     = dist
         density[ijk] = Density(dist, σ, 1)
         alpha[ijk]   = α
7. background: sdf = +6σ；alpha = α（生成器的全局 --alpha）；density = 0
8. writeGrids(out.nvdb, {density, alpha, sdf})
```

即 **带内写真值，带外由 `background` 承担**。三个网格共用同一套 active mask，
与原代码一致；相对原代码只有两处改动，都是 background：

| 网格 | 原代码 | 本方案 | 理由 |
|---|---|---|---|
| `sdf` | `0` | `+6σ` | 0 的含义是"正好在表面上"→ 射线一进场就判定碰撞（§2.3 点 2） |
| `alpha` | `0` | `α` | 带外值应"像"带内值，不要拿 0 当"没有数据"；`0` 会让带边层的 roughness 偏小（§2.3 点 5） |
| `density` | `0` | `0`（不变） | 无消费者 |

**符号公式的实测精度**（P0，§6.0 ②）：`dist = √sqrD·(1 − 2|w|)` 把绕数的
**近似误差乘进了距离模长**，而不只用来定符号。对闭合网格实测：

- 内部（`w = 1.000000`）与外部（`w ≈ 0`）**精确**，模长不受影响；
- 远场（`P=(1,0,0)`，距立方体 0.5）`w = −4.46e-4` → `dist = 0.499554` 而解析值
  `0.500000`，**偏低 0.09%**。

0.09% 远小于 `dx`，且带内（`|dist| < 3σ`）离表面很近、`|w|` 更接近 0 或 1，
所以**按原代码保留**。但这条写进注释，别再以为 `(1 − 2|w|)` 只是个纯符号开关。
若将来遇到非闭合／自相交资产（`w` 落在 0.5 附近会让 `dist → 0`），
切 `--sign-mode threshold`：`dist = sign(w < 0.5) · √sqrD`（模长不再被污染）。

```
Density(x, σ, k) = k · φ(x;0,σ) / Φ(x/σ)
```
原代码用 pbrt 的 `Gaussian(x,0,σ)`（= 归一化 pdf）和手写 `0.5(1+erf(x/(σ√2)))`。
本项目直接用 `include/macrofacet/mathutility/Gaussian1D.h` 的
`normalPdf(x, 0.0, σ)` 与 `normalCdf(x/σ)`，数学上完全一致。

**坐标映射（已实测，有半体素偏移的坑）**：`setTransform(dx, origin)` 写进的是
只含对角阵的纯仿射映射，源码是

```cpp
Map::set(dx, trans, taper)   // → mat = dx·I, vec = trans
applyMap(ijk) = dx·ijk + trans        // ← 体素【角】，不是中心
```

实测 `applyMap(Vec3d(0)) == origin`、`applyMap(Vec3d(1)) == origin + dx` 逐分量成立。
所以 **`indexToWorld(ijk)` 返回的是体素下角**，NanoVDB 的 `Map` 里**没有**半体素偏移，
`Coord` 重载也不加（OpenVDB 的 `Transform::indexToWorld(Coord)` 会加，两者不一致，别混）。

于是：

| 量 | 公式 |
|---|---|
| 体素 `ijk` 占据的世界区间 | `[origin + dx·ijk, origin + dx·(ijk + 1)]` |
| 体素 `ijk` 的中心 | `origin + dx·(ijk + 0.5)`  ← 填充时用这个 |
| 世界点 `P` 的连续索引（**中心系**） | `u = (P − origin)/dx − 0.5` |
| `P` 的最近体素 | `floor(u)`；`u − floor(u)` 是插值权重 |

**`− 0.5` 不能省。** NanoVDB 的 `Map` 是角系，而体素里存的是**中心**处的场值
（OpenVDB 约定，`VdbBaker` 就是这么填的）。两套约定只差半个体素，后果是整场沿
每轴平移 `dx/2`：平坦区域看不出来，在 `|∇m| ≈ 1` 的地方就是恒定偏置 `dx/2`。

**这个坑在 P1 里真实踩到过。** `NanoVdbGridSampler.h` 最初按角系写
`u = (P − origin)/dx`，而 baker 按中心填。症状：单测对拍解析球时 sdf 恒定偏
`+0.0332`，且**在体素中心处 `viaMean ≠ stored`**——这一条就是判据。补上 `− 0.5` 后
`viaMean == stored` 逐位相等。所以验收不必依赖解析真值，用这条自洽性即可：

> 在任意体素中心采样，插值器必须原样返回该体素存的值。

这条对 `sample` 与 `sampleWithGradient` 都成立，且与 mesh 是否正确无关——它只验约定。
解析场对拍（§6 P1 验收 3）验的是另一半，两者都要有。

**`tools::build::Grid<T>` 没有 `worldToIndex`**（那是序列化后的 `NanoGrid` 上的方法），
所以 world→index 要自己按上表算，别指望 builder 给。

### 2.3 必须写进注释的保真点与 API 陷阱

1. **`dist` 是有符号的，而 `Density` 是单边公式。**
   `dist < 0`（内部）时 `Φ(x/σ) < 0.5`，密度被放大。最坏在 `dist → -3σ`：
   `φ(3)/σ / Φ(-3) ≈ 3.28/σ`，是表面峰值 `0.798/σ` 的 4 倍。
   这是**原代码的行为，不是复刻错误**。→ 保留，加 `--symmetric`（用 `|dist|`）作为可选开关。

2. **`sdf` 的 background 必须是 `+6σ`（不能是 0）。**
   原代码是 `GridBuilder(0.f)`，带外读回 0。这是本方案对 `sdf` 的唯一数值改动。

   先明确**不需要**全域 sdf。碰撞分布是

   ```
   h(t)·T(t) = φ(z)/(σΦ(z)) · Φ(z) = φ(z)/σ   ⟹   m(x(t)) ~ N(0, σ)
   ```

   碰撞点服从以均值场为中心、标准差 σ 的高斯分布，**99.73% 落在 ±3σ 内**。
   带外的值本身几乎不参与计算。

   **但它的取值仍然要害**——关键在于**起点就在 background 区**：
   `birthPosition` 是 `activeDomain` 的入射点（§5.1），
   而 `activeDomain` 是 config 盒子、比 NVDB 网格大，所以起点、以及起点到网格
   之间那一整段，读到的都是 background。所以它必须取正的大值，否则：

   - `background = 0` 的含义是"正好落在表面上" → ρ 取峰值 `0.798/σ`。
     `ConditionedRay.cpp:56-57` 在**构造时**就对 `x0_` 调 `evaluate` 与 `pointPrior`，
     于是**起点就在表面上**——三个 mode 从第一条射线起全线失效。
   - `background = +3σ` 也不够：`ρ(+3σ) = 0.0044/σ`，那一段长度 `L` 上积出
     `∫h ≈ 0.0044·L/σ`。`render_sphere`（L=0.7、σ=0.03）约 **10%** 的碰撞发生在空气里。
     `+6σ` 时 `ρ(+6σ) = 6.2e-9/σ`，外部贡献归零。

   **已知残差（接受）**：带外只有**一个** background 值，而 `sdf` 是带符号的，
   于是**紧贴带内侧**那一层（体素中心落在 `[-3σ, -3σ+dx]`）在
   `-2.9σ` 与 `+6σ` 之间插值时会**穿过 0**。后果：

   - 该层声明出的 `∫h ≈ 0.59·dx/σ`，真值是 `3.28·dx/σ` → 光学深度被低估。
   - 表现为"穿透过表面"的分量偏亮（`dx = σ/2` 时约 4 倍）。
     该分量本身只占 `Φ(-3) = 0.135%`，**净图像误差约 0.5%**。
   - 补齐内部饱和值**不能**消除它（跳变只是移到更深处，仍要穿过 0）；
     真正的修法是全域写，那会放弃稀疏性。

   取舍：保留"壳 + background"的稀疏表示（与原实现一致），把这一层作为
   **已量化、已知位置、约 0.5%** 的截断近似记录在报告里，见 §7。

3. **`maximumGradientNorm` 要按声明场实测，不要沿用 1.0。**
   `classicMajorant`（`ClassicCoefficients.cpp:55-57`）用它算面积上界。
   带内真值的 `|∇m| ≈ 1`，但带内最后一个体素与 background 之间的跳变
   （`+2.9σ → +6σ`，跨一个 `dx`）会把实测值抬到 `3σ/dx` 量级
   （`dx ≈ σ/2` 时约 6）。装载时扫一遍取实测值；它偏大只会让 majorant 变松、
   追踪变慢，不会出错。

> **与 spec 的一致性**：`macrofacet_implementation_spec.md` §13 边界规定
> 第 2 条要求"对于无限 GP，有限域是截断近似，**必须在报告中写出边界和截断距离**"，
> 第 4 条要求"若使用 ±3σ 薄层，**所有模型和数值参考都采用同一层**"。
> 本方案满足：三个 mode 共用同一套 mask、`σ` 单一来源（§2.4）、残差已量化（点 2）。
> 报告里要写明：有效域 = σ 扩展 AABB，带 = ±3σ，`background = +6σ`。

4. **`dx` 必须 ≲ `σ`，否则场是空的。**
   σ=0.01、32³、物体尺寸 2 时 `dx≈0.07 > 3σ=0.03` → 带内体素数为 **0**，静默产出空网格。
   → 新 generator 加检查：若 `dx > σ` 打印警告并给出建议分辨率
   `N = ceil(gap / (σ/2))`；`--allow-sparse` 可关掉。

5. **`alpha` 的 background 建议取 α——但这不是"必须"，起点那边的危险已经不存在了。**

   我最初把 alpha 接进 `pointPrior`，于是射线起点读到 `alpha = 0` → `covarianceG = 0`
   → `ConditionedRay` 的观测协方差奇异。**那个问题是我自己造出来的**：alpha 不该走
   `pointPrior`（角色 1 是 GP 的梯度协方差，见 §1.3；起点位置的实情见 §5.1）。
   拆成两个访问器之后，起点只读角色 1、永远是 `σ²P` ✓，所以 background 取 0 也不会奇异。

   仍然建议取 α，只剩两个次要理由：

   - 角色 2 虽然在碰撞点读，但**碰撞点可以落在带内侧那一层**（点 2 的残差层），
     那里的 `alpha` 在真值 α 与 background 之间插值；取 0 会让该处 roughness 偏小。
   - 与 `sdf` 的处理保持一致：**带外的值要"像"带内的值**，不要拿 0 当"没有数据"。

   代价为零（α 本来就是生成器参数，sidecar JSON 里已有一份），所以照取。
   将来 alpha 真正逐体素化时，"background 取带内代表值"这个约定仍然成立。

6. **NanoVDB v12 的四个 API 陷阱**（都已在 P0 实测里踩到，见 §6）：

   - **`writeGrids` 的 `VecT` 永远推导不出来**。它的签名是
     `template<typename BufferT = HostBuffer, template<typename...> class VecT = std::vector>`
     ——模板模板参数不参与推导，必须两个都显式给：

     ```cpp
     nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(
         path, handles, nanovdb::io::Codec::NONE);
     ```

     不写就是 `C2672: 未找到匹配的重载函数`，错误信息完全指不到点子上。

   - **`NANOVDB_USE_OPENVDB` / `_ZIP` / `_BLOSC` / `_CUDA` 全部默认关闭——保持关闭**。
     这正是 nanovdb 能脱离 openvdb/blosc/zlib/CUDA 独立编译的原因（§3）。
     代价是只能 `Codec::NONE`（无压缩）：实测一个 0.8 半径球、`dx=0.02`、
     ±1.6 域、只写 ±3σ 带，sdf + 常量 σ 两个网格 = **2.55 MB**。
     要压缩就得引入 blosc/zlib，收益不值得——带内体素数本来就少。

   - **MSVC 警告必须先按住**：nanovdb 头文件会产生 `C4146`（一元负号作用于无符号）、
     `C4267`（`size_t`→`uint32_t`）、`C4244`、`C4996`（`strcpy`/`strncpy`），
     以及 **`C4819`**——后者的成因是头文件里有非 ASCII 字节，而本机 MSVC 代码页是 936。
     本项目的 CMake **没有** `/utf-8`，加上它（见 §4）；再把 nanovdb 的 include
     标成 `SYSTEM`。

   - **`tools::build::Grid<T>::operator()(func, bbox, delta)` 会静默丢掉等于 background 的值**：
     `func` 的返回值若恰好等于 `background`，该体素**不激活**（`mValueMask` 保持 off），
     插值时按背景值读回。这既是稀疏性的来源（"带外写 background"不浪费，那些体素不落盘），
     也是一条要主动排查的陷阱。对这三个网格逐一确认：

     | 网格 | background | 带内值域 | 会撞上吗 |
     |---|---|---|---|
     | `sdf` | `+6σ` | `[−3σ, +3σ]`，含 0 | 不会 |
     | `density` | `0` | `[0.0044/σ, 3.28/σ]`（`Density` 在 `dist=0` 处是 `0.798/σ`，全域不为 0） | 不会 |
     | `alpha` | `α` | 常量 `α` | **会**——但只在使用 `operator()` 时；见下 |

     **`alpha` 这一行不需要特判：实现用的是 `setValue`，根本没走这条路径。**
     `setValue` → `RootNode::setValue` → `SetValue::set` **无条件**写值并置 active 位，
     不做 background 比较；只有 `operator()(func, bbox)` 里才有
     `if (v != root.mBackground) leaf->setValue(...)`。所以选 `setValue` 正是为了绕开这个陷阱：

     - 三个网格的 active mask **逐位相同**，不存在"alpha 网格整个是背景"。
       （`alpha` 若是空树，`getValue` 会返回 `background == α`，读出来**数值上碰巧也对**，
       所以这个坑不会自己暴露——只会让 mask 契约悄悄变成假的。）
     - 代价是失去了"写 background 就自动稀疏"的性质，所以带外**必须靠循环本身不写**来保证，
       而不是靠写入值等于背景。`VdbBaker` 的 band 过滤（`signedDistance` 落在 `(−band, band)`
       之外直接 `continue`）就是这条。

     因此断言应该直接写成对拍：**遍历三个网格的 active 体素数，必须相等且等于上报的
     `bandVoxels`**，再对整块 index bbox 扫一遍验证"带内 ⇒ density>0 且 alpha=α"、
     "带外 ⇒ sdf=+6σ、density=0、alpha=α"。`tests/test_nanovdb_field.cpp` 的 `testVoxelContract`
     就是这么做的，它比"带内被拒体素数为 0"更强，且不依赖 `operator()` 的行为。

     `density` 的 background 保持 0 与 `sdf` 的 `+6σ` 不同，正是 §2.5 对拍**只能在带内**做的原因。

### 2.4 σ 的落盘（你要的"随文件走"，但机制换了）

**你要的是语义**：σ 必须存在 `.nvdb` 里，而不是只活在外面的 JSON。这个语义保留。
**换掉的是机制**——`GridBlindMetaData` 这条路在 v12 走不通，且已实测确认：

- v12 的 `GridBuilder` 是 `nanovdb/tools/GridBuilder.h` 里的
  `nanovdb::tools::build::Grid<T>`，**没有** `blindDataSize` 构造参数。
- 唯一能挂 blind data 的地方是 `CreateNanoGrid<SrcGridT>::addBlindData()`，
  但它是 `preProcess()` 内部调用的私有成员，且只对两类源生效：
  openvdb 的 `PointIndexGrid`/`PointDataGrid`（需要 `NANOVDB_USE_OPENVDB`），
  以及超长 grid 名（`mSrcNodeAcc.hasLongGridName()`）。
  源是 `tools::build::Grid<float>` 时**一条也挂不上**。
- 手改 buffer 后处理（扩容 + 追加 `GridBlindMetaData` + 改
  `mBlindMetadataOffset`/`mBlindMetadataCount`）理论可行，但要连
  `writeGrids` 的 segment 尺寸和 GridChecksum 一起对上，约 40 行脆代码。
  **不值得**。

**替代方案：把 σ 写成第四个常量网格**，同一段落、同一个 `.nvdb`：

```cpp
// 没有 operator() 调用 → 树是空的，只有一个 root，文件开销约等于 0
nanovdb::tools::build::Grid<float> sig(static_cast<float>(sigma), "sigma",
                                       nanovdb::GridClass::Unknown);
sig.setTransform(dx, origin);
```

读取端与另外三个网格**走完全相同的 API**：

```cpp
auto h = nanovdb::io::readGrid<nanovdb::HostBuffer>(file, "sigma");
const double sigma = h.grid<float>()->tree().root().background();
```

这比 blind data 更好，而不是将就：

| | blind data | 常量 σ 网格 |
|---|---|---|
| 公开 API | 否，要手改 buffer | 是，全走 `createNanoGrid`/`readGrid` |
| 类型 | `GridBlindDataSemantic/Class/Type` 手填 | `float`，天然类型化 |
| 读取 | 手工 walk `blindMetaData(0)` | 与另外三个网格同一个调用 |
| 跨版本 | 元数据布局易变 | 同 grid 格式，跟主数据同生共死 |
| 校验和 | 手改后要重算 | 不用管 |

**成本**：已实测，空树常量网格与 sdf 网格一起写出后能按名读回，
`background()` 精确等于写入的 σ（见 §6 P0 实测记录）。

**同时写一份 sidecar JSON** `<out>.json`：

```json
{ "version": 1, "sigma": 0.05, "alpha": 0.5, "dx": 0.0719, "origin": [0,0,0],
  "band": 3.0, "grids": ["density","alpha","sdf","sigma"], "sign_mode": "winding",
  "background": { "sdf": 0.3, "alpha": 0.5, "density": 0.0 },
  "mesh": "knob.ply", "resolution": [32,32,32] }
```

`background` 一并落盘，理由是它**语义上属于文件而非实现细节**：
`NanoVdbMean` 在域外要返回 `sdf` 的 background（§5.1），
对拍测试要拿它做期望值。写进去就不必从 grid 里反查。

**加载优先级与一致性检查**（这条是 σ 落盘真正的收益）：

1. 读 `"sigma"` 网格 → 得到 `sigmaNvdb`；
2. 读 sidecar JSON 的 `"sigma"` → 得到 `sigmaJson`；
3. 两者都在且不一致 → **报错**，不要静默挑一个。这一条专治
   "配置改了 σ 但忘了重新烘焙"——那会让渲染结果与预期差一个统计尺度，
   而且不报错的时候几乎查不出来；
4. 只有 JSON → 用 JSON（人工编辑过的文件）；
5. 都没有 → 报错要求显式指定 σ。

**比较精度是 float32，不是 1e-9**（2026-09-24 修）。σ 网格是 float32，所以一个
经文件往返的 σ 与当初写进去的 double 相差可达**一个 float32 ulp**；而原先的容差
`1e-9 * max(1, σ)` 比这个 ulp **更窄**，于是 0.06 / 0.08 / 0.09 / 0.1 / 0.12 /
0.15 / 0.2 / 0.3 全被误判成"不一致"。后果是 `resolveSigma` 对着生成器刚写出的文件
报 `sigma disagrees between the .nvdb (0.100000) and its sidecar (0.100000)`——
**那几个 σ 上生成器自己的输出根本读不回来**（实测 18 个常用 σ 里 10 个被拒）。
现改为按 float32 比较（`sameSigma`，`NanoVdbIO.h`）：真正改过 σ 的差异远大于一个
ulp，一个也没放过。回归断言在 `testSigmaSchema`。

**config schema 分家**（2026-09-24）。σ 在两类场里性质不同，config 因此是两套：

| | 程序化场（plane / sphere / …） | 烘焙场（nanovdb） |
|---|---|---|
| σ 是什么 | 模型旋钮，与数据无关 | 数据的一部分（带 ±3σ、背景 +6σ、dx 都按它定） |
| `field.sigma` | **必填**，唯一来源 | **可省略**，省略时由文件供给 |
| 写了会怎样 | — | 与文件交叉检查；不一致**报错**，一致则按文件里的写法取值 |
| `--sigma` | 自由扫描 | 只能复述文件的 σ（float32 相等），否则**报错**要求重新烘焙 |

声明这个能力的是**场自己**：`MeanField::intrinsicSigma()`——核心库仍然不需要知道
NanoVDB 的存在，`NanoVdbMean` 覆写它，`loadExperimentConfig` 只问接口，和
`typeName()` 走的是同一条路。`correlation_lengths`（即 `P = (σ_G/σ)²`）**仍留在
config 里**：它是核参数，不是场的性质。

`writeNanoVdbMean` 仍然写 `field.sigma`，但语义从"可被覆盖的输入"变成"本轮实际用的
值"——resolution 输出里的 `field.sigma` 与 `derived.sigma` 现在都如实记录文件里那个
float32 σ（例如 `0.009999999776482582`），渲染因此可复现。

**`GridClass::LevelSet` + `background = +6σ` 正是 NanoVDB 对 level set 的惯用约定**
（窄带内写真值、带外取大的正值、值的符号表示内外）。原代码把 background 写成 0
反而偏离了这个约定。`alpha` / `density` / `sigma` 用 `GridClass::Unknown`（实测枚举值 0，
`LevelSet` 是 1）。

### 2.5 density 网格是派生量，没有独立信息

原代码两个网格写的是同一个变量：

```cpp
float d = Density(dist, intrSigma, 1.f);
densityAccessor.setValue(ijk, d);
sdfAccessor.setValue(ijk, dist);
```

逐体素成立 `density[ijk] ≡ Density(sdf[ijk], σ, 1)`。**density 网格不含任何
`sdf + σ` 之外的信息**——它就是 GP 模型边缘密度的查找表。

所以砍掉 volume 模式后，三个网格里只有 **sdf 和 alpha** 有渲染消费者。

density 仍然生成（它是文件契约的一部分，且烘焙代价可忽略），但定位改为
**交叉验证**：把烘焙的 density 网格与 `NanoVdbMean` + σ 现场算出的密度逐体素对拍，
一次验证 sdf 插值精度、σ 元数据往返、坐标映射三件事（见 §6 P2 验收 4）。

对拍**只在带内**（三网格共用的那套 active mask）进行，不要全域比：`density` 的
background 保持 0，而 `sdf` 的变成 `+6σ`，带外两者本就不该相等。这是
`density` 保持 background = 0 的唯一后果，而它没有渲染消费者，所以无所谓。

---

## 3. 依赖（全部走本地 vcpkg，全部不用 pbrt）

### 3.1 结论先行

| 依赖 | 来源 | 版本 | 状态 |
|---|---|---|---|
| **NanoVDB** | **自写 overlay port** `ports/nanovdb/`，从 openvdb 源码里只抽 `nanovdb/` 头文件 | **v12.0.1** | ✅ 已实测 |
| **libigl** | 本地 vcpkg 内置 port `libigl` | 2.6.0#2 | ✅ 已实测（含一个 port 缺陷，见 3.3） |
| **raytri.c** | 自备 shim（public domain，1 个文件） | — | ✅ 已实测 |
| **PLY 读取** | **自研**（`src/fieldgen/PlyReader.cpp`，~120 行：ascii + binary_little_endian，只要 vertex x/y/z 与 face vertex_indices） | — | 无依赖 |
| **Eigen / nlohmann-json** | 已有 | 5.0.1 / 3.12.0 | ✅ 不变 |

**不用 pbrt 这条已经满足**：下表里没有任何一项来自 pbrt。libigl 是你批准的，
nanovdb 是 Apache-2.0，raytri.c 是 public domain。

### 3.2 NanoVDB：vcpkg 里没有 port，用 overlay port 自建

**先说清楚 vcpkg 里有什么**（已查本地 `D:\vcpkg`）：

- **没有 `nanovdb` 这个 port**。`ports/` 下 `nano*` 有 17 个，没有 nanovdb；
  `versions/n-*/` 里也没有。
- 唯一沾边的是 `openvdb` port 的 `nanovdb` feature（12.0.1）。**但它用不了**：
  它的 portfile 里那段是

  ```cmake
  if (OPENVDB_BUILD_NANOVDB)
      set(NANOVDB_OPTIONS -DNANOVDB_USE_INTRINSICS=ON -DNANOVDB_USE_CUDA=ON ...)
      vcpkg_find_cuda(OUT_CUDA_TOOLKIT_ROOT cuda_toolkit_root)   # ← 硬要 CUDA
  ```

  且 `vcpkg.json` 里 `"nanovdb": { "dependencies": ["cuda"] }`。为了拿几个头文件
  去装 CUDA toolkit + boost + blosc + tbb + openexr + imath，**不成比例**。

**采用的方案**：在项目内建一个 overlay port，只把头文件抽出来。

```
ports/nanovdb/
    vcpkg.json          # name=nanovdb, version 12.0.1, license Apache-2.0
    portfile.cmake      # vcpkg_from_github(...) → file(COPY nanovdb/nanovdb → include/nanovdb)
    usage               # 可选
vcpkg-configuration.json   # { "overlay-ports": ["./ports"] }
vcpkg.json                 # dependencies 加 "nanovdb"
```

portfile 的关键一行与 `D:\vcpkg\ports\openvdb\portfile.cmake` **完全一致**：

```cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO AcademySoftwareFoundation/openvdb
    REF "v12.0.1"
    SHA512 67b859bf77c53e68116faa7915bb6a5a50a8cff10435762890e13348625e8aebdb6661b722017632471648afe31e2f9d4cd2e18456c728192bfd0accd70a40ef
)
file(COPY "${SOURCE_PATH}/nanovdb/nanovdb" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(RENAME "${CURRENT_PACKAGES_DIR}/include/nanovdb" "${CURRENT_PACKAGES_DIR}/include/nanovdb_tmp")
```

（`file(RENAME)` 一步是为了让最终布局是 `include/nanovdb/NanoVDB.h`，而不是
`include/nanovdb/nanovdb/NanoVDB.h`；具体写法在实现时调，`#include <nanovdb/NanoVDB.h>`
能解析是唯一验收标准。）

**这条零网络**：`vcpkg_from_github` 会先查 `D:\vcpkg\downloads\`，
而 `AcademySoftwareFoundation-openvdb-v12.0.1.tar.gz` **已经在那儿**，
且实测其 SHA512 与上面那行**逐字节相同**。所以这台机器上装它不会联网。

**为什么是 v12.0.1 而不是原计划记的 v10.1.0**——这是本轮最重要的一处事实更正：

- **v10.1.0 的 `nanovdb::GridBuilder<T>` 在 v12 里不存在了**。v12 的
  `nanovdb/util/GridBuilder.h` 只剩 6 行，是个 deprecated 转发壳：

  ```cpp
  #include <nanovdb/tools/GridBuilder.h>
  NANOVDB_DEPRECATED_HEADER("Include nanovdb/tools/GridBuilder.h instead.")
  ```

  真正的实现在 `nanovdb/tools/GridBuilder.h`，但它提供的**不是** `GridBuilder<T>` 类，
  而是 `nanovdb::tools::build::{RootNode, InternalNode, LeafNode, Tree, Grid, NodeManager}`。
  取 handle 的入口变成自由函数 `nanovdb::tools::createNanoGrid(grid)`。
- v10.1.0 **不是** vcpkg 里已有的版本（`openvdb` port 钉的是 12.0.1），用它要另外
  下一个 tarball，还多一份和 openvdb port 不同版本的隐患。
- v12.0.1 的头文件**就在本机**（`D:\vcpkg\buildtrees\openvdb\src\v12.0.1-*/nanovdb/nanovdb/`，
  75 个头文件、2.8 MB），API 是**读源码 + 编译运行验证**过的，不是凭记忆。

选 v12 的代价只有一个：σ 不能走 blind data（§2.4 已给出更好的替代）。值。

**纯头文件、零重依赖，已实测**：`NANOVDB_USE_OPENVDB` / `_ZIP` / `_BLOSC` / `_CUDA`
全部默认关闭，实测一个只 include `nanovdb/{NanoVDB,GridHandle,tools/GridBuilder,tools/CreateNanoGrid,io/IO}.h`
的 TU 用 `cl /std:c++17` 直接编过，**没有 openvdb、没有 blosc、没有 zlib、没有 TBB、没有 CUDA**。

### 3.3 libigl：port 能用，但 vcpkg 那份装不全

实测（隔离安装根，不动项目自己的 `installed/`）：

```
vcpkg install libigl --triplet x64-windows --x-install-root=D:/vcpkg_probe/installed
→ libigl:x64-windows@2.6.0#2  Installed in 38 s
→ CMake target: igl::igl_core（不是 igl::core）
```

**Eigen 5.0.1 的兼容性风险解除**：port 本身编过了，而且实例化也编过了——
`igl::AABB<Eigen::MatrixXd,3>::squared_distance` 与 `igl::fast_winding_number(...,2,bvh)`
在 Eigen 5.0.1 下**编译并给出正确结果**（见 §6 P0 实测记录）。
原计划里"写自己 BVH 的 200 行备选"**不需要了**。

**但 port 漏装一个文件**：`igl/raytri.c`。这条链路是

```
igl/AABB.h → AABB.cpp → ray_mesh_intersect.h → (自包含) ray_mesh_intersect.cpp
           → extern "C" { #include "raytri.c" }   ← 装完之后不存在
```

后果是 `#include <igl/AABB.h>` 直接 **fatal error C1083: 无法打开包括文件 "raytri.c"**。
libigl 源码树里这个文件是有的（`buildtrees/libigl/src/v2.6.0-*/include/igl/raytri.c`，
267 行），只是 port 没拷进 `installed/`。

**处理**：把它作为 shim 自备一份 `src/ext/igl_shim/raytri.c`，加进 include 路径。
`#include "raytri.c"` 是引号形式，MSVC 先找包含者所在目录、找不到就回落到 include 路径，
所以 shim 目录能接住。**实测有效**（§6 P0）。

好消息是**这个文件是 public domain**——它是 Tomas Möller 的 ray-triangle 代码，
libigl 自己的注释就写着 `// Alec: this file is listed as "Public Domain"`。
所以自备它没有 MPL-2.0 的传染问题，登记到 `THIRD_PARTY.md` 即可。

> 备选（更 "vcpkg-native" 但更重）：写 `ports/libigl/` overlay port 把 `raytri.c` 补进去。
> 那要连 vcpkg 的 1 个 portfile + 5 个 patch 一起接管维护，**不划算**。1 个 public domain
> 文件的 shim 更省事，也更容易在下游说清。

### 3.4 PLY 自研

原来打算复用 `src/ext/rply/`（pbrt 那份 MIT），但它随 `src/ext/` 一起删了。
本项目需要的 PLY 子集极小（顶点位置 + 三角面索引），自研比重新拉一个库更干净。

**工作树现状**：`src/pbrt/`、`src/ext/`、`.gitmodules` 都已从工作树删除。
所以 §4.1 里的 `src/ext/igl_shim/` 要新建；原来 `.gitmodules` 声明的那套子模块
（含那个 404 的 `feature/nanovdb` 分支）不再存在，不需要处理。

### 3.5 烘焙性能（实测，决定默认分辨率）

16k 三角面的球，单线程：

| 分辨率 | 体素数 | `squared_distance` | `fast_winding_number` | 合计 |
|---|---|---|---|---|
| 64³ | 262 k | 0.39 s | 0.59 s | **0.98 s** |
| 128³ | 2.1 M | 4.64 s | 10.68 s | **15.3 s** |
| 256³ | 16.8 M | 36.2 s | 57.8 s | **94.0 s** |

BVH 预计算（AABB + FWN，16k 三角面）：**0.06 s**。

两个结论：

1. **绕数比距离贵约 1.6 倍，且没有批量接口**——libigl 只提供逐点
   `fast_winding_number(bvh, order, p)`，所以循环不能靠一次大调用摊薄。
   它是瓶颈。
2. **必须多线程**。§2.3 点 4 要求 `dx ≲ σ`；`render_shader_ball_conditional` 的
   σ = 0.01、域长 ~3.2，`dx ≤ σ` 意味着一轴 **≥ 320**，即 33 M 体素
   → 单线程约 3 分钟，8 线程约 25 s。项目里已有现成的行/块并行写法可抄
   （`MacrofacetPathTracer.cpp` 的动态任务分配）。

---

## 4. 目标与构建

### 4.0 清单与配置文件

`vcpkg.json`（加一项）：

```json
{ "name": "macrofacet", "version-string": "0.1.0",
  "dependencies": ["eigen3", "nlohmann-json", "libigl", "nanovdb"] }
```

`vcpkg-configuration.json`（**新建**，overlay port 靠它生效）：

```json
{ "overlay-ports": ["./ports"] }
```

`ports/nanovdb/vcpkg.json`：

```json
{ "name": "nanovdb", "version": "12.0.1",
  "description": "NanoVDB header-only subset extracted from OpenVDB",
  "homepage": "https://github.com/AcademySoftwareFoundation/openvdb",
  "license": "Apache-2.0", "dependencies": [] }
```

`ports/nanovdb/portfile.cmake` —— 见 §3.2。装完头文件后再写一个 10 行的
`share/nanovdb/nanovdb-config.cmake` 定义 `nanovdb::nanovdb`（INTERFACE，
`target_include_directories(... SYSTEM INTERFACE "${_prefix}/include")`），
这样消费者侧就是一句标准 `find_package`，不用 `find_path` 猜路径。

### 4.1 `CMakeLists.txt` 新增

```cmake
find_package(nanovdb CONFIG REQUIRED)      # ← 我们的 overlay port
find_package(libigl CONFIG REQUIRED)       # ← 本地 vcpkg port，target 是 igl::igl_core

# raytri.c shim：vcpkg 的 libigl port 漏装了这个 public domain 文件（§3.3）
add_library(igl_raytri_shim INTERFACE)
target_include_directories(igl_raytri_shim SYSTEM INTERFACE src/ext/igl_shim)

# --- 场库：核心的扩展 ---
add_library(macrofacet_field
    src/fieldgen/PlyReader.cpp
    src/fieldgen/MeshDistanceField.cpp     # AABB + 绕数 → 带符号距离
    src/fieldgen/VdbBaker.cpp              # Density() + 三网格填充（多线程）
    src/fieldgen/NanoVdbIO.cpp             # 读写 + σ 网格 / sidecar
    src/fields/NanoVdbMean.cpp             # MeanField 实现
    src/fields/NanoVdbSampledField.cpp     # alpha 网格 → ScalarField（逐点 roughness）
    src/fields/MeanFactory.cpp             # 注册表
    src/fieldgen/FieldGeneratorMain.cpp)   # ← 独立 target 的 main
target_include_directories(macrofacet_field PUBLIC include)
target_link_libraries(macrofacet_field
    PUBLIC macrofacet nanovdb::nanovdb igl::igl_core igl_raytri_shim)
target_compile_options(macrofacet_field PRIVATE
    /W4 /permissive- /EHsc /utf-8)         # ← /utf-8 必须加，理由见下

# --- 独立可执行 target ---
add_executable(macrofacet_fieldgen src/fieldgen/FieldGeneratorMain.cpp)
target_link_libraries(macrofacet_fieldgen PRIVATE macrofacet_field)
```

`macrofacet_experiments` 与 `macrofacet_tests` 增加 `PRIVATE macrofacet_field` 即可拿到
`"nanovdb"` mean 类型。核心库 `macrofacet` 的依赖列表**一行不改**——
nanovdb 只在 `macrofacet_field` 里露头，而 `NanoVdbMean` 走 pimpl（§5.1），
所以 `include/macrofacet/` 的公共头文件里不会出现 `nanovdb::`。

**两个必须记住的编译选项**（都是实测踩出来的）：

1. **`/utf-8` 不是可选项**。nanovdb 头文件里有非 ASCII 字节，本机 MSVC 代码页是
   936，不加 `/utf-8` 就是满屏 `warning C4819: 该文件包含不能在当前代码页(936)中
   表示的字符`。核心库 `macrofacet` 现在**没有**这个选项——它是干净的，别动它；
   只给 `macrofacet_field` 加。
2. **把 nanovdb / libigl / shim 的 include 标成 `SYSTEM`**。`/W4` 之下第三方头文件
   会刷出 `C4146`（一元负号作用于无符号）、`C4267`（`size_t`→`uint32_t`）、
   `C4244`、`C4996`（`strcpy`/`strncpy`）。`nanovdb::nanovdb` 由 port 的 config
   文件标好 SYSTEM，`igl_raytri_shim` 这边我们显式标。

> 关于 `igl::igl_core` 而不是 `igl::core`：vcpkg 的提示串是
> `target_link_libraries(main PRIVATE igl::igl_stb igl::igl_core)`，
> port 自己重新命名过 target。以 `plugins`/`share/libigl` 下生成的 config 为准。

### 4.2 目录（本轮变更）

```
vcpkg.json                       改：dependencies 加 "libigl"、"nanovdb"
vcpkg-configuration.json         新建：overlay-ports = ["./ports"]
ports/nanovdb/                   新建：overlay port（vcpkg.json + portfile.cmake）
src/ext/igl_shim/raytri.c        新建：1 个 public domain 文件，补 vcpkg port 的漏装（§3.3）
include/macrofacet/fields/       新建：ScalarField.h、NanoVdbMean.h、MeanFactory.h
src/fieldgen/                    新建：generator 实现 + PLY/距离场/烘焙/IO
src/fields/                      新建：NVDB 场的接口实现
```

相对上一版计划的差别：**没有 `src/ext/nanovdb/`，也没有 `src/ext/libigl/`**。
两者都改由 vcpkg 提供（overlay port + 内置 port），不再 vendor 进源码树。
`src/ext/` 下只剩一个 267 行的 shim 文件。

**`PlyReader.cpp` 是自研**（§3.4）：需要的 PLY 子集极小（顶点位置 + 三角面索引），
不值得引入第三方。

---

## 5. 渲染器集成

### 5.1 `NanoVdbMean`（P2，核心）

```cpp
// include/macrofacet/fields/NanoVdbMean.h —— 头文件里不出现任何 nanovdb 符号
class NanoVdbMean final : public MeanField {
public:
  static MeanFieldPtr open(const std::string& gridFile, double sigmaOverride = 0.0);
  ~NanoVdbMean() override;
  MeanJet evaluate(const Point3& x) const override;   // 三线性 + 中心差分梯度
  double valueDifference(const Point3& x, const Vector3& d) const override;  // 两次查找
  BoundsSummary bounds(const Bounds3& domain) const override;
private:
  struct Impl;              // handle_ / grid_ / dx_ / sigma_ / voxelBounds_ 全在这里
  std::unique_ptr<Impl> impl_;
};
```

**为什么必须是 pimpl**：这个头文件属于高层接口（`include/macrofacet/fields/`），
而核心库 `macrofacet` 不链接 nanovdb（§1.6）。只要头文件里出现
`nanovdb::GridHandle` / `nanovdb::FloatGrid`，任何 include 它的 TU 就被迫能看到
nanovdb 头文件，§4 那条"核心库依赖列表一行不改"就守不住了。

关键的三个实现细节：

- **梯度**：中心差分 `(sdf(x+dx·eᵢ) − sdf(x−dx·eᵢ)) / (2dx)`，复用三线性采样。
  与 `SphereMean`（解析梯度）对拍时，误差应在 `O(dx²)`。
- **起点在哪里**（决定"域外行为"要多宽松）：`MacrofacetPathTracer.cpp:70` 已经把
  `birthPosition` 设成 `activeDomain ∩ ray` 的**入射点**，不是相机。但 `activeDomain`
  是 **config 声明的盒子**（`ExperimentConfig.cpp:179-181`，来自 `domain_min`/`domain_max`），
  **不是** NVDB 网格的 bbox。以 `render_sphere.json` 为例：domain ±1.6、球 r=0.8、
  σ=0.03 → 带只有 ±0.89，**起点在网格之外 0.7 处**。

  两个推论：
  1. 域外返回 `background` 是**必须**的——起点就读在那儿（上一条）。
  2. **不能靠"把起点挪进带内"来回避这个难度**：domain 边界到带边界那一段，
     对 `SphereMean` 是**真实的 sdf 斜坡**（σ=0.09 的 validation 配置下
     `∫h` 有百分之几），对 NVDB 场却是平的 `+6σ`。挪了以后两种 mean
     **不再是同一个模型**，P2 验收 1 的解析对拍就没意义了。
     正确做法是让 **alpha 根本到不了起点**（§1.3 的访问器拆分）。

- **`bounds()` 必须保守**：`minimumValue` 取域内体素最小值（装载时预计算一个
  粗粒度 min/max mip，或直接扫 band）；`maximumGradientNorm` **不能假设为 1**——
  离散 sdf 的三线性重建梯度可以超过 1（`dx` 越粗越明显）。
  装载时扫一遍所有 active 体素算 `max |central difference|`，实测值填入，
  `certified = true`。这个值直接喂给追踪的安全上限，填错会静默出错。
- **`evaluate` 必须是全定义的，不能抛异常**（这条推翻了本节早先的写法）。
  理由：起点 `x0_` 在 NVDB 网格**之外**（见上一段），而
  `ConditionedRay.cpp:56-57` 在构造时就用它调 `evaluate` 与 `pointPrior`。
  **解析场能扛住这一下**——`SphereMean` 在球外照样返回 `|x−c| − r`；
  `NanoVdbMean` 必须同样健壮。

  所以：bbox 外（以及 bbox 内的 inactive 体素）返回
  `{background, Vector3::Zero()}`，即 `sdf` 的 `+6σ` 和零梯度，而不是抛。
  于是起点处 `m = +6σ` → `z = 6` → `ρ = 6.2e-9/σ ≈ 0`，射线在真空里出发，
  这正是 `+6σ` 这个取值的第二个作用（第一个见 §2.3 点 2）。
  此外若射线整条错过物体，全程都读到 `+6σ` → 不碰撞 ✓ 语义自洽。

  实现上取 grid 的 background：`readGrid<HostBuffer>(file, "sdf")->grid<float>()
  ->tree().root().background()`（v12，已实测，§6.0 ①），**不要**硬编码 `6σ`
  ——两者只是恰好在生成时相等。

### 5.2 `"nanovdb"` mean 类型（P1）

```json
{
  "field": {
    "mean_type": "nanovdb",
    "grid_file": "outputs/fields/knob_32.nvdb",
    "sigma": 0.05,
    "correlation_lengths": [0.05, 0.05, 0.05]
  },
  "material": { "roughness": 0.5 }
}
```

`sigma` 从 NVDB blind data / sidecar 读；JSON 里显式给了就以 JSON 为准（便于做
σ 扫描实验）。`correlation_lengths` 与 `material.roughness` 的互斥逻辑
（`ExperimentConfig.cpp` 现有逻辑）原样适用。

### 5.3 volume 模式（挂起，不做）

曾评估过一条 delta tracking 通道（`σ_t` 取 density 网格、局部帧取 `∇sdf`、
出射复用 `ConductorPhase`），技术上可行，但**本方案不做**。

理由：conditional29 / classic / midpoint 追踪的是条件风险率 `h(t)`，
它依赖出生状态 `(f₀, g₀)`，不是局部量 `σ_t(x)`（§0.2）。
volume 模式能承载的只有 density 网格所代表的那条**边缘/解相关**模型，
与现有三条传输路径是并列关系而非替代关系，属于另一件事。

`density` 网格照常生成，留给将来需要时接（它的定位见 §2.5）。

---

## 6. 阶段与验收

每个阶段独立可编译、可验收、可回滚。

**P0 — 依赖打通（0.5 天）**

> **本阶段的核心内容已经做完并实测通过了**，记录见 §6.0。剩下的只是把它落进
> 项目的构建系统（写 `ports/nanovdb/`、改 `vcpkg.json`、加 shim、`cmake --build`）。

- 写 `ports/nanovdb/` overlay port（§4.0），`vcpkg.json` 加 `libigl` + `nanovdb`
- 放 `src/ext/igl_shim/raytri.c`（§3.3）
- 建一个 20 行的 TU：`#include <igl/AABB.h>` + `#include <nanovdb/NanoVDB.h>`，能编过
- 验收：`cmake --build` 通过，且**两边的版本与实测记录一致**：
  nanovdb 12.0.1、libigl 2.6.0#2、Eigen 5.0.1

### 6.0 P0 实测记录（本轮已完成）

不依赖项目构建系统，直接在临时目录把 §2 的关键链路跑通。三个探针：

**① NanoVDB 读写链路**（`cl /std:c++17 /EHsc /O2 /W3`，只给一个 `-I`）

- `tools::build::Grid<float>(bg, name, GridClass)` + `setTransform(dx, origin)`
  + `operator()(functor, CoordBBox)` + `tools::createNanoGrid()` + `io::writeGrids<>()`
  → 写出含 2 个网格的 `.nvdb`，`readGrid(file, "sdf")` 按名读回 ✓
- 半体素约定核对：`applyMap(0) → (-1.6,-1.6,-1.7)`（= origin）、
  `applyMap(1) → (-1.58,…)`（= origin + dx）→ **`applyMap` 是体素角** ✓
- 取球面附近三点，用 `origin + dx·(ijk+0.5)` 预测再读回：

  | 世界 x | 连续索引 | 预测 | 最近体素读回 | 三线性读回 |
  |---|---|---|---|---|
  | 0.79 | 119.50 | −0.0099 | **−0.009873** | +0.000125（= 两侧均值，对） |
  | 0.80 | 120.00 | +0.0101 | **+0.010123** | +0.010123（`fx=0` 退化为最近体素，对） |
  | 0.95 | 127.50 | 带外 | **+0.3**（= 6σ） | +0.3 |

  → 坐标公式、±3σ 带、`background = +6σ`、三线性，四条全对 ✓
- **空树常量网格可用**：`Grid<float>(σ, "sigma", Unknown)` 不做填充，照样序列化、
  按名读回、`background()` 精确等于 σ ✓（§2.4 的方案就靠这条）
- 文件体积：球 r=0.8、`dx=0.02`、±3σ 带，sdf + σ 两个网格 = **2.55 MB**
- 踩到的坑：`writeGrids` 的 `VecT` 推导失败（→ §2.3 点 6）、
  `tools::build::Grid` 没有 `worldToIndex`（→ §2.2）

**② libigl × Eigen 5.0.1**（`libigl:x64-windows@2.6.0#2`，38 s 装完）

- `igl::AABB<Eigen::MatrixXd,3>::squared_distance` 与
  `igl::fast_winding_number(bvh, 2, p)` **在 Eigen 5.0.1 下实例化并给出正确结果** ✓
- 单位立方体（半边长 0.5）对拍解析 SDF：

  | P | 绕数 w | 配方 dist | 解析 | |
  |---|---|---|---|---|
  | (0.6,0.6,0.6) | 0.000000 | +0.173205 | +0.173205 | ✓ |
  | (0.7,0,0) | −0.000000 | +0.200000 | +0.200000 | ✓ |
  | (0,0,0) | **+1.000000** | **−0.500000** | −0.500000 | ✓ 内部 |
  | (−0.6,0,0.1) | −0.000000 | +0.100000 | +0.100000 | ✓ |
  | (1,0,0) | −0.000446 | +0.499554 | +0.500000 | 差 0.09%，见下 |

  → §2.2 步骤 5/6 的配方**成立**；"自研 BVH"备选**不需要**了
- `|w|` 只在**极远处**偏离 0：`(1,0,0)` 处 `w = −4.46e-4`，代入
  `sqrt(sqrD)·(1−2|w|)` 把模长拉低了 0.089%。对**闭合网格**成立性没问题
  （内部 `w=1`、外部 `w=0`，实测精确到 6 位），这项误差只影响远场模长。
  但要注意它是**乘在模长上的**——见 §7 那一行
- **踩到的坑**（值得单独记一笔）：`squared_distance` 的**单点 6 参数重载**第 4 个参数是
  `up_sqr_d`（"只考虑小于它的距离"），**不是输出**。我一开始写了

  ```cpp
  double sqrD = 0.0; int I; Eigen::RowVector3d C;
  tree.squared_distance(V, F, P, sqrD, I, C);   // ← 编译通过，sqrD 被当成上界 0
  ```

  结果**整个场全 0，没有任何报错**。生成器里要用**批量形态**
  （`P` 为 `#P×3` 矩阵，`sqrD` 为 `VectorXd`），或者用返回 `Scalar` 的 5 参数单点形态，
  **不要**写这个 6 参数形式。§6 P1 验收 4 的解析对拍就是为拦这类静默错误准备的

**③ 烘焙吞吐**：见 §3.5 的表（16k 三角面，单线程，64³ = 0.98 s、256³ = 94 s）。

### 6.0b P1/P2 实测记录（本轮已完成）

**P1 — 全部验收通过。** `macrofacet_field` / `macrofacet_fieldgen` 在 `/W4 /permissive-`
下零警告；`tests/test_nanovdb_field.cpp` 覆盖验收 1/1b/2/3/4，全绿。

- 烘焙实测（球 r=0.8，σ=0.03，128³ 请求 → 自动 144³，`dx=0.0125`，32 线程）：
  带内 743988 体素（24.92%），sdf ∈ [−0.08998, +0.09000] = ±3σ，
  background = 0.18 = +6σ，`maximumGradientNorm` = 21.60，density max = 109.41，
  4.66 s + 16.70 MB。**density max = ρ(−3σ) = 3.28/σ 手算值吻合** ✓
- **验收 3 是自洽性对拍，比解析对拍更早发现问题**：在体素中心上
  `viaMean == stored` 必须逐位相等。P1 期间正是这条抓出了半体素错位
  （§2.2 那个坑）：解析对拍只给一个 ±0.033 的常数偏差，无法定位；
  自洽性对拍直接指出「烘焙对、采样错」。

**P2 — 端到端对拍（本轮实测）。** 关键数字：**渲染均值**（128×128×4spp，
`directional_gradient`）作为统计量，先测噪声底噪，再测场效应。

| 配置 | R 通道 | G 通道 | B 通道 | 相对解析球 |
|---|---|---|---|---|
| 解析 `SphereMean`（seed 1） | 0.448294 | 0.398476 | 0.470519 | — |
| 解析 `SphereMean`（seed 2） | 0.448293 | 0.398488 | 0.470532 | **1.00000 / 1.00003 / 1.00003**（噪声底噪） |
| 烘焙 `NanoVdbMean`（seed 1） | 0.447504 | 0.397929 | 0.470036 | 0.99824 / 0.99863 / 0.99897 |
| 烘焙 `NanoVdbMean`（seed 2） | 0.447580 | 0.397975 | 0.470092 | **1.00017 / 1.00011 / 1.00012**（噪声底噪） |

即：烘焙场的噪声底噪 ≈ 0.017%，解析场 ≈ 0.003%；两者差 **0.10–0.18%**，
是底噪的约 10 倍 —— 真实信号，需要归因。逐项排除：

1. **不是网格离散**：球面三角化 128 → 256 段（三角形数 ×4），
   比值从 0.99824 变成 0.99825，**完全不动** ✗
2. **是 ±3σ 带边缘**：把带放宽到 5σ、background 提到 10σ 重烘，
   比值变成 **1.00002 / 1.00006 / 1.00008** —— 落回噪声底噪之内 ✓

**结论**：`NanoVdbMean` 对解析场的复现是精确的；0.15% 的差全部来自
「带内真值 + background = +6σ」这条写入策略在 ±3σ 处留下的**均值场跳变**
（`m` 在带边从 3σ 直接跳到 6σ，而解析场是平滑过渡）。这正是 §2.3 点 2/点 3
记录、并且已经接受的那个代价 —— 现在它有了量化数字。

> **可选的取舍**：`--band 5 --background 10` 让端到端误差落到噪声以下，
> 代价是带内体素从 24.9% 涨到 34.9%（同样 `dx` 下文件更大、烘焙更慢）。
> 默认仍是 3σ/6σ；需要高保真对拍时用 5σ/10σ。

**P2 验收 5 的实现修正**：`alpha` 进消光的路径**只有 GGX 一支**。
最初的实现把高斯族的投影面积也改读了 `materialNdf` —— 那是错的：
§1.4 A 明确把 `ClassicCoefficients.cpp` 的投影面积列为角色 1（传输量，
是碰撞梯度分布而非着色 NDF），只有 D 的 GGX 分支读 alpha。
两处都用 `false &&` 短路验证过新测试确实会红：高斯族那条给出 worst ratio **2.30**，
GGX 那条给出 **96.9**（≈ (0.5/0.05)² = 100，与面积 ∝ α² 吻合），
恢复后分别绿。`classicMajorant` 的唯一正确性改动因此落在 **GGX 分支**上，
取 `alphaField->bounds(domain).maximumValue`；高斯族分支保持原样。

### 6.0c 真实模型（`scenes/shader ball.ply`）实测记录

**输入**：Blender 5.2.2 导出的 `binary_little_endian` PLY，17442 顶点 / 32996 三角形，
顶点属性为 `x y z s t`（多出的 `s`/`t` 走通用 scalar 路径被正确跳过），
面为 `list uchar uint vertex_indices`。**0 条边界边、0 条非流形边** —— 闭合流形，
所以默认的 `--sign-mode winding` 与带内剪枝都是精确的，不需要 `--no-prune-band`。

**模型结构**：**42 个互不相连的闭合分支**（χ = 84 = 2×42，每个都是亏格 0），
即"基座盘 + 主球 + 2 个环 + 36 颗小铆钉"的经典 shader ball 装配体：

| 部件 | 原生尺寸 | 说明 |
|---|---|---|
| 基座盘（2812 顶点） | 直径 0.0892 | 最宽部件，决定包围盒 |
| 主球（3074 顶点） | 直径 0.0539 → r=0.0270 | |
| 内盘（1298 顶点） | 直径 0.0628 | |
| 2 个环（各 294 顶点） | 0.0294 × 0.0055 | 细环，管径极细 |
| 36 颗铆钉（各 98 顶点） | 直径 0.0020 | 最小特征 |

**为什么必须加变换**：config 的 `sigma = 0.01` 是为**半径 1.0** 的程序化球定的。
原生模型整体只有 0.0892 宽，于是相关长度 `σ/roughness = 0.1` 比整个模型还大 12%
—— 统计面在物体上没有分辨率；而 ±3σ 带（0.06）也有模型宽度的 67%。
`macrofacet_fieldgen` 原先没有任何缩放/居中开关，任何按自己单位导出的模型
都会踩这个坑，所以本轮补了 `--fit <extent>` 与 `--center`（`MeshTransform`
落在 bake 之前，`dx`/`origin`/全部 bound 都随之推导；变换写进 sidecar，
因为它是 .nvdb 里唯一无法恢复的 bake 输入）。

**实际烘焙**：

```
macrofacet_fieldgen "scenes/shader ball.ply" --fit 2.8 --center \
    --sigma 0.01 --alpha 0.1 --x 320 --y 320 --z 320 \
    --out outputs/fields/shader_ball.nvdb
```

`--fit 2.8` 让最大尺寸 = 2.8（基座半径 1.4、主球半径 0.846），
这样模型连同 **+6σ background** 都落在 config 的 `domain [-1.6,-1.6,-1.7]..[1.6,1.6,1.5]`
之内 —— 取 3.31（对齐程序化球半径 1.0）会让基座的带**被 domain 裁掉**。
产物：`dx = 0.008045`（= 0.80σ）、357×357×329、带内 5 715 802 体素、
**144.69 MB**、32 s。自检：`sdf ∈ [-0.03, 0.0299989]`（正好 ±3σ）、
`background = 0.06`（+6σ）、`density max = 328.31`（= 3.28/σ = ρ(−3) 解析值）✓

**端到端**：`configs/render_shader_ball_nanovdb.json` 是 `render_shader_ball_conditional.json`
的 drop-in 替换（同 σ、同 domain、同相机、同 conditional29），
`correlation_lengths = [0.1,0.1,0.1]`（= σ/roughness，注意 `material.roughness`
与 `correlation_lengths` 互斥，见 `ExperimentConfig.cpp:141`）。
48×48×1 冒烟渲染通过，输出合理（`unit_white` 三通道均值 0.9796，
`directional_gradient` 均值 (0.485, 0.329, 0.352)，2211/2304 像素有能量）。

**发现 1 —— `max_quadrature_subdivisions` 必须提高**。原始 2048 直接**报错**
`integration_not_converged: initial optical-depth partition exceeds budget`。
原因在 `OpticalDepthSampler.cpp:21-24`：步长被钳到 `0.5σ / maximumGradientNorm`，
而烘焙场的 `maximumGradientNorm = (max−min)/dx`（`NanoVdbMean.cpp:24-26`），
解析场报的是真实 `|∇m| = 1`。因为 `maximumGradientNorm ≈ 6σ/dx`，
**σ 在钳位里被约掉**，剩下的步长 ≈ `dx/12`，与 σ 无关：
所需 panel 数 ≈ `12 × 射线长度 / dx`。本场 dx = 0.008 时约 8 300 > 2048，故需
16384。**粒度更细的 bake 需要更大预算**（该数随 1/dx 增长）。

**发现 2 —— 该 bound 相对物理值松了约 7.5×，直接体现为渲染成本**。
冒烟渲染 **27 ms/path**（球体那次是 0.133 ms/path），计数显示
**101 859 次 hazard 求值 / path**、6 349 个积分区间 / path —— 全部来自上述钳位。
若把 bound 换成烘焙场的物理值 `|∇sdf| = 1`，区间数降到约 850/path（≈ 7.5×），
但**这会改变消光 majorant 与积分划分，属于"改数值"的杠杆，需要你拍板**
（与 [[render-perf-pending-levers]] 里那两条同类）。按现状，
256×256×8 的满配渲染约需 **4 小时**，不建议现在就跑。

**发现 3 —— `NanoVdbMean::maximumGradientNorm()` 严格来说不是合法上界**。
它的极值只扫**活动体素**（`NanoVdbSampler.h` 的注释写明 "over the grid's active
voxels"），所以区间 [−3σ, +6σ] 那种"活动体素紧邻 background"的单元
其真实 Lipschitz 常数是 `9σ/dx`，比它报的 `6σ/dx` 大 1.5×。
本例侥幸成立（带边中心差分最大 5.6 < 7.46），而且该 bound 因为整体过松
（真实 `|∇m| ≈ 1`）在实践上被完全掩盖 —— 但它**不是构造性成立的**，
一旦有人把它收紧到物理值，`classicMajorant` 就会有漏碰风险。记录在此待定。

**P1 — generator 独立 target（2 天）**
- 实现 §2 全部逻辑 + `macrofacet_fieldgen`
- 验收：
  1. `macrofacet_fieldgen knob.ply --sigma 0.05 --x 64 --y 64 --z 64 --out knob.nvdb`
     产出文件：**三个网格共用同一套 active mask（±3σ 带内）**；
     带外读回 `sdf = +6σ`、`alpha = α`、`density = 0`（§2.2 步骤 7 的表）
  1b. `dx > σ` 时给出警告与建议分辨率（§2.3 点 4）
  2. `nanovdb::io::readGrid` 读回，抽查若干体素的 `density`，与手算 `Density()` 一致
  3. 自研的坐标公式（§2.2 的表）vs **序列化后的真 grid handle** 逐点对拍。
     注意对拍对象必须是 `io::readGrid` 拿到的 `NanoGrid`，因为
     `tools::build::Grid<T>` 身上**没有** `worldToIndex`/`indexToWorld`——
     烘焙期能用的只有自己按 `(P − origin)/dx` 算的公式（§2.2）
  4. **解析对拍**（原"对拍 pbrt 版"的替代——`src/pbrt/` 已删除，且 pbrt 不参与构建）：
     把球面三角化后走完整条流水线烘焙，逐体素与解析 `|x − c| − r` 比较。
     同时验证距离查询、绕数符号、坐标映射三件事。
     容差按网格离散误差给（球面三角化是内接多面体，`dist` 会偏大
     `O(h²/r)`，`h` 为三角形边长）

**P2 — 接入渲染（2 天）**
- `NanoVdbMean` + `NanoVdbSampledField` + 注册表 + `"nanovdb"` mean_type + `writeResolvedConfig`
- 验收：
  1. `sdf` 网格由解析球 SDF 烘焙 → `NanoVdbMean` 与 `SphereMean` 渲染同一场景，
     图像在噪声范围内一致（这条同时验证 §5.1 的梯度与 bounds）
  2. **起点回归（§2.3 点 2 / 点 5）**：起点 `x0_` 是 `activeDomain` 的入射点
     （`MacrofacetPathTracer.cpp:70`），而 `activeDomain` 是 config 盒子
     （`render_sphere` 那类配置是 ±1.6，带只有 ±1.07），所以起点落在 NVDB 网格**之外**：
     `NanoVdbMean::evaluate(x0_)` 返回 background（`+6σ`）、梯度为 0、不抛异常；
     `pointPrior(x0_).covarianceG` = `σ²P` **非奇异**（这正是 §1.3 拆分的回归点）；
     `ConditionedRay` 能正常构造（`render_sphere.json` 是 `sampled_exterior`，
     会真的走到这条路），且射线不在起点碰撞
  3. `bounds(domain)` 的扫描值对照打印，**期望值不是解析界**：
     `minimumValue ≈ -3σ`（带底），**不是 `-r`**——球心在带外，网格里根本没有
     `-r` 这个值（§2.3 点 2 已接受这一点）；`maximumGradientNorm` 也不是 1.0，
     而是被带内最后一层到 background 的跳变支配的 `3σ/dx` 量级（§2.3 点 3）。
     可断言的只有两条：`maximumGradientNorm ≥ 1.0`，且
     `≥ 遍历所有 active 体素实测的 max |中心差分|`（保守性，这条才是 majorant 的要害）
  4. **density 交叉验证**：烘焙的 `density` 网格 vs `NanoVdbMean` + σ 现场算出的
     密度，在**共用的 active mask 内**逐体素对拍（一次验证 sdf 插值精度 +
     σ 元数据往返 + 坐标映射）
  5. **alpha 路径**：`NanoVdbSampledField::bounds` 的 `maximumValue` 不小于
     采样扫描值；`validate()` 拒绝含 0 的 alpha 网格；
     §1.4 **D** 的 majorant（GGX 分支）在 `alpha(x) > config roughness` 的区域仍成立。
     另需断言**角色分离**：`pointPrior` 在有 `alphaField` 时返回的 `covarianceG`
     仍等于 `σ²P`（不是 `alpha²I`）——这条防的是实现时又把 alpha 接回传输
  6. 现有 `macrofacet_tests` 全绿（`alphaField` 为空时的回退路径必须**逐位**不变）
  7. ~~新增 `tests/test_nanovdb_mean.cpp`~~ → 实际落在 `tests/test_nanovdb_field.cpp`
     （与 P1 的用例共用同一个记忆化烘焙文件，省掉一次 4.7 s 烘焙）：
     三线性精度、域外行为、bounds 保守性、`materialNdf` 的角色分离、
     GGX majorant 的域内最大值
  8. 新增 `configs/render_sphere_nanovdb.json` + `macrofacet_fieldgen` 的
     `--out` 父目录自动创建，端到端结果见 §6.0b

**投入**：P0 半天 + P1 两天 + P2 两天。P3（volume 模式）挂起，见 §5.3。

---

## 7. 风险

| 风险 | 影响 | 处置 |
|---|---|---|
| ~~Eigen 5.0.1 vs libigl~~ | — | **已解除**（§6.0 ②：装得上、实例化得了、结果对） |
| vcpkg 的 libigl port 漏装 `igl/raytri.c` | P0 阻塞（`C1083`） | 自备 1 个 public domain shim 文件（§3.3），**已实测有效** |
| `AABB::squared_distance` 单点 6 参数重载把 `sqrD` 当成上界 | **静默产出全 0 的场**（无报错） | 用批量形态或 5 参数 Scalar 形态（§2.3 点 6、§6.0 ②）；P1 验收 4 的解析对拍兜底 |
| 绕数的近似误差乘进距离模长（`dist = √sqrD·(1−2|w|)`） | 远场模长偏低约 `2|w|`（实测 4.5e-4 → 0.09%） | 闭合网格下内部 `w=1`、外部 `w=0` 精确成立，可接受；若发现非闭合资产再切 `--sign-mode threshold`（`dist = sign(w<0.5)·√sqrD`） |
| `dist` 单边公式导致内层密度 4× | 数值/物理语义 | 保留（复刻优先），`--symmetric` 可选 |
| `tools::build::Grid::operator()` 静默丢掉等于 background 的值 | 带内出现"洞" | 生成器加断言"带内被拒体素数为 0"（§2.3 点 6 的表） |
| 带内侧一层插值穿过 0 → 光学深度低估 | 约 0.5% 图像误差（穿透过表面偏亮） | 已量化并接受（§2.3 点 2）；在报告里作为截断近似记录，符合 spec §13 边界规定第 2 条 |
| 起点在 `activeDomain` 边界（带外），逐点网格在那里只能读到 background | 若把该值喂进传输就会奇异 | **不能靠移动起点解决**（`activeDomain` 是 config 盒子，§5.2）；靠 §1.3 的访问器拆分：起点只读角色 1 |
| alpha 局部驱动 GGX 时大于 config 值 → `areaMaximum` 偏小 | **majorant 失效（正确性）** | §1.4 D 改用 `ScalarField::bounds` 的域内最大值 |
| 提议分布与目标错配（alpha 逐点后） | 无偏但方差上升 | §1.4 C 让 `sampleBeckmann`/`proposalNormalPdf` 同走 `materialNdf` |
| alpha 调制着色 NDF 但不调制传输/消光 | 语义解耦：着色法线分布 ≠ 追踪统计 | **设计决定，不做**；文档与报告必须写明，不得声称两者一致（§1.4 A） |
| ~~blind data 走非公开 API~~ | — | **已规避**：v12 下不可用，改用常量 σ 网格（§2.4），反而更稳 |
| 高分辨率下烘焙慢（256³ 单线程 94 s；`dx ≤ σ=0.01` 时约 3 分钟） | 可用性 | **必须多线程**（§3.5）；绕数是瓶颈，不能靠批量调用摊薄 |
| 现有 `outputs/macrofacet_ci` 4/6 artifact 不可复现（已知） | 无法用它做回归基线 | P2 验收用解析球自查，不依赖旧 artifact |

---

## 8. 不做的事

- 不引入任何 pbrt 头文件 / 源码到构建
- 不改 `macrofacet` 核心库的依赖列表
- 不改 `ConductorPhase` 的数学（§1.4 C 只把两处 `σ²·P` 换成等价写法
  `materialNdf(x).covarianceG`，无 `alphaField` 时逐位相同）
- 不动 `ConditionedRay` / 三个 FlightKernel / `FlightState` 的任何一行（conditional29 保持原样）
- **不改 `pointPrior` 的返回值**：它属于传输（角色 1），alpha 不进这条路（§1.3）
- **不做逐点传输核 `P(x)`**：`alpha` 网格只调制**着色**的梯度协方差，
  不改变随机场的相关长度／沿射线条件化的代数（§1.4 A）
- 不做 volume 模式，不让 conditional29 依赖 `σ_t`（§5.3）
- 不恢复 `src/pbrt/`（工作树里已删除；算法已转录进 §2，对照改用解析球，见 §6 P1 验收 4）

**依赖侧的不做**（都是这轮实测后刻意排除的选项，别再回头试）：

- **不装 `openvdb[nanovdb]`**。它硬要 `cuda` + boost + blosc + tbb + openexr + imath，
  而我们要的只是头文件（§3.2）。
- **不接管 vcpkg 的 `libigl` port**。为了补一个 `raytri.c` 去维护
  1 个 portfile + 5 个 patch 不划算；自备一个 public domain shim 文件即可（§3.3）。
- **不 vendor 第三方源码**。没有 `src/ext/nanovdb/`、没有 `src/ext/libigl/`，
  `src/ext/` 下只有那一个 267 行的 shim（§4.2）。
- **不给 `macrofacet` 核心库加 `/utf-8`**。那个选项只为 `macrofacet_field` 存在
  （nanovdb 头的非 ASCII 字节 + 本机 936 代码页，§4.1）；核心库是干净的，别顺手改它。
- **不追 v10.1.0 的 `GridBuilder`**。v12 里它已被 `tools::build::Grid` +
  `tools::createNanoGrid` 取代（§3.2），按 v12 写。

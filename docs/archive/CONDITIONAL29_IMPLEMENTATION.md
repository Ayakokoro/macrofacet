# conditional29 regular tracking 实现

> 历史实现记录。文内 Classic/Midpoint 仍使用旧分格采样器的描述已过时；
> 三模式当前都使用连续光学深度反演。见 [当前 tracing 文档](../NVDB_TRACING.md)。

2026-09-24。公式对应仓库的 `Macrofacet_去除第二个假设后的完整推导.md`。

## 渲染路径

`MacrofacetPathTracer` 的 conditional29 分支现调用 `sampleFlight`，其内部通过 `OpticalDepthSampler` 反演连续累计消光。Classic/Midpoint 仍调用原有分格距离采样器。共享的条件统计及梯度采样数值改进也会影响使用这些底层函数的模式，但不改变其目标分布。

```text
相机射线 → AABB 入口 → 首段初始化
  → 条件 hazard → 分段自适应 GK15 积分
  → safeguarded Newton 反演自由程
      → 逃逸：beta × 环境光
      → 碰撞：条件完整梯度 → Fresnel → 镜面反射 → 保存完整梯度 → RR
```

regular tracking 不构造 majorant 或 null collision。每段固定起点观测，使用式（10）的 hazard：

$$
\lambda(t)=\frac{p(F_t=0\mid H)}{P(F_t>0\mid H)}
\mathbb E[(-d^TG_t)_+\mid F_t=0,H].
$$

先采 $E=-\log(1-U)$，再解

$$
\int_a^t\lambda(u)\,du=E,
\qquad t_{new}=t-\frac{A_{a,t}-E}{\lambda(t)}.
$$

Newton 步越界或停滞时回退二分。分段累计积分只用于定位和缓存；格内线性插值仅作为初始猜测。距离结果通过连续积分残差与距离误差验收，不再由一张分段常数 hazard 表定义。

到出口仍未积满 $E$ 才返回逃逸，质量为 $\exp(-A_{a,L})$。命中结果保存实际累计积分的 `logSurvival` 和 `logDistancePdf`。当 `state.age > 0` 时从该年龄积分，仍以原出生点为距离原点。

## 配置

现有 CI 配置已显式启用：

```json
"conditional29": {
  "sampler": "regular_tracking",
  "external_policy": "sampled_exterior",
  "optical_depth_solver": "safeguarded_newton"
}
```

该对象位于 `transport` 下，优先于公共 `correlated_sampler` 和 `external_policy`。`optical_depth` 是 regular tracking 的兼容别名；其他 sampler/solver 值会报错。旧配置中的 `original_macrofacet` 首段语义保持可用。

| 选项 | 含义 |
| --- | --- |
| `sampled_exterior` | 在域入口采截断到正半轴的场值与完整先验梯度，首段即使用条件模型 |
| `original_macrofacet` | 无条件 Classic hazard 首段；距离仍用 regular tracking，首碰梯度启动下一段条件模型 |
| `numeric.absolute_tolerance` / `relative_tolerance` | 累计光学厚度残差容差 |
| `numeric.distance_absolute_tolerance` | 距离绝对容差，默认 `1e-10` |
| `numeric.distance_relative_tolerance` | 距离相对容差，默认 `1e-8` |
| `render.flight_table_cells` / `--flight-cells` | conditional29 的初始积分分块提示，不改变目标 hazard；其他模式仍为旧表格分辨率 |
| `transport.hard_depth_cap: null` | conditional29 仅通过逃逸和 RR 终止，无固定 64 层截断 |

有限 `hard_depth_cap` 是带截断的预览设置，计入 `safety_cap_terminations`。`resolved_config.json` 记录实际采样器、首段策略、上限与数值容差。不能把 Classic/Midpoint 配置里历史遗留的 sampler 字符串当成实际算法；resolved config 将其标为 `tabulated_legacy`。

正场值模型在相机位于域外时以 AABB 入口为源端，相机在域内时以相机位置为源端。它没有对域外真空位置附加 GP 观测。`FlightState::birthValue` 与 `ObservedExterior` 表示此状态，不要求初始方向导数为正。真实碰撞后回到 `Surface`、场值为零且反射方向向外。

## 数值稳定性和权重

- `ConditionedRay` 使用 SE 核显式矩；平面、球面和旋转各向异性使用同一矩阵表达式。
- 近起点用级数计算值方差与条件斜率方差；三维 Schur 补通过非负的白化分量构造。小正方差不会被直接判成零 hazard。
- 球面均值差采用范数差的稳定表达式，保留很小的切向位移产生的高度变化。
- 积分分块同时参考相关尺度、均值场尺度和近起点几何级数；条件均值过零附近增加按方差宽度分布的节点。
- 梯度采样保留式（43）的负通量权重，一维反演按分布宽度归一化并使用 Newton。极端高斯尾部 CDF 使用正项对数相加，避免消减和普通 normalizer 下溢。
- 完整梯度用于下一段；匹配采样的路径更新只有 Fresnel 和 RR 补偿，不重复乘消光、透射率或 NDF。
- `ConditionalConductorPhase` 提供条件 NDF 导出的 $q_R=D_H/(4B)$ 及 Fresnel 散射核求值。它描述当前完整梯度尚未固定时的边缘分布，不能据此对已固定镜面法线随意连接光源。

每条路径的积分缓存独立；采样后的指数目标在重试时保持不变。求根或积分失败不会被 conditional29 渲染器当成黑色或逃逸，最终失败会传播到 CLI 并包含路径信息。重试使用更严格的 double 求积；没有宣称增加 `long double` 或真实多精度后端。

`render_summary.csv` 新增 hazard 求值、积分区间、Newton 迭代、二分回退、最大光学厚度残差和积分误差估计。求积误差是 GK 的估计，算法针对条件模型在指定数值容差下实现，不声称数值上严格无偏。

## 验证和运行

```powershell
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
build\Release\macrofacet_experiments.exe render --config configs\conditional29_validation.json
build\Release\macrofacet_experiments.exe render --config configs\conditional29_sphere_validation.json
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json --output outputs\conditional29_regular_curves
```

新增测试覆盖：常数/线性消光解析反演，真实距离 CDF 与逃逸频率，零导数保护，非零起始年龄，失败传播，独立矩阵条件化参考，旋转各向异性与球面，近端方差、掠射与分块数量，正场值初始化，条件方向 PDF/NDF 归一化，极小梯度尺度、深尾采样，实际配置分派，以及线程确定性。

`FlightCurveExperiment` 的 conditional29 直方图现在使用实际采出的连续距离，不再仅统计指数目标在累计表中的 bin。

最终 Release 构建通过 CTest，测试程序报告 `checks=16449 failures=0`。已完成平面与球面 `16×16×64 spp` 的单位白炉和方向环境渲染；每次 16,384 条相机路径，均为零数值失败、零深度截断。白炉平均值为平面 `1.0002935`（像素均值标准误 `0.0002049`）、球面 `1.0000064`（标准误 `0.00000453`）；相应近似 95% 区间均包含 1。输出位于 `outputs/conditional29_validation` 和 `outputs/conditional29_sphere_validation`。

## 明确的范围

保留最近一次完整观测，仍采用文档中的单点正侧近似；不是整段真实 GP 首穿模型。NEE/MIS、一般局部 BRDF 和多精度后端不属于此次 analog 镜面导体流程。配置启用 conditional29 NEE 会明确报错。沿线精度严格为零、需要确定性距离原子的情形会报告 `UnsupportedSingularFlight`；不会伪装成连续密度或逃逸。

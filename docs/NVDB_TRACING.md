# NVDB 窄带传输

本文是当前实验程序的场输入和飞行采样说明。配置字段的完整列表见
[配置参考](CONFIGURATION_REFERENCE.md)，场景与材质参数示例见
[场景渲染](SCENE_RENDERING.md)。早期计划放在 `docs/archive/`，旧
[实现报告](IMPLEMENTATION_REPORT.md) 保留当时的结果；其旧执行路径不再作为使用说明。

## 从配置到场

`macrofacet_experiments` 在 `render`、`curves`、`gp-reference` 和 `all` 之前
统一准备 NVDB 场。Classic 和 conditional29 使用同一个场文件：

```text
解析 mean_type ──按 activeDomain 采样──┐
                                     ├── sdf + density + alpha ── NarrowBandMedium ── 两种 FlightKernel
现成 mean_type: nanovdb ──直接打开───┘
```

`plane`、`sphere`、`cutaway_sphere`、`shader_ball` 和 `constant` 是**烘焙输入**。
解析函数只在准备场和数学参考测试时求值。正式实验入口通过
`requireNanoVdbField` 检查场类型；关闭 `MACROFACET_BUILD_FIELDS` 的构建不能运行
实验程序的 tracing 命令。底层解析 `MeanField` 实现仍用于验证公式。

自动烘焙在 `field.domain_min` / `domain_max` 的整个 active domain 上采样，
并增加一个插值单元的边缘。它保留球体深处的负 SDF，不使用旧 mesh
窄带格式的正背景替代内部。文件缓存在
`<output_directory>/fields/analytic_<hash>.nvdb`；缓存键包含原始场参数、
domain、σ、体素大小和烘焙格式版本。`--sigma` 改变解析场的缓存键并触发新烘焙。
`resolved_config.json` 记录实际 `grid_file`、`voxel_size` 和 `coverage`。

默认体素边长从 `σ/2` 开始；若超过 800 万体素预算，会逐步增大并向 stderr
报告。对于细节或窄随机表面，应检查 resolved config 中的实际大小。
可以在 `field` 中显式设置正数 `bake_voxel_size`（世界单位）；超预算会报错，
不会悄悄修改指定值。

已有 `.nvdb` 文件用以下配置直接打开：

```json
{
  "field": {
    "mean_type": "nanovdb",
    "grid_file": "outputs/fields/model.nvdb",
    "use_alpha_grid": true
  }
}
```

现成 NVDB 的追踪域由 `sdf` 与 `density` 网格的空间范围自动确定，并在边缘
增加一个体素的插值余量；配置中旧的 `domain_min` / `domain_max` 会被忽略。
`resolved_config.json` 的 `derived.domain_min/max` 记录实际范围。相机坐标仍需
与模型坐标匹配。这段配置还需要 `material.ndf_family`，以及
`correlation_lengths` 或 `material.roughness` 等字段。`sigma` 从文件中的
常数 `sigma` 网格读取；显式给出的 `sigma` 必须与文件一致。`sdf` 用三线性
插值，其梯度是同一个插值函数的导数。`alpha` 可选，驱动逐点材质 NDF；
`density` 是 Classic 碰撞率的高度项。它的零背景定义传输带外的真空；
conditional29 也以它的非零插值范围限制条件 hazard，但不把静态密度值代入
条件公式。`sdf` 与核参数只为带内的投影面积和条件统计提供数据。

## Mesh 烘焙与覆盖范围

```powershell
build\Release\macrofacet_fieldgen.exe model.ply --full-domain --band 6 --sigma 0.05 --out outputs\fields\model.nvdb
```

`--full-domain` 保存烘焙盒子内的完整有符号距离，sidecar 标记
`"coverage": "full_domain"`。默认 mesh 生成方式仍是历史窄带格式，标记
`"surface_band"`；窄带外、包括深层内部，`density` 为零，因此按真空跳过。
此处的 `sdf` 正背景只供存储与带边插值使用，不把深层内部解释为真实正 SDF。
mesh 烘焙盒子是网格包围盒加 `--band × σ` 边距。`--full-domain` 可用于
需要保留完整内部场的其他实验；它按连续光学深度采样。

## 两种 mode 的飞行采样

| Mode | hazard / 碰撞状态 | 首段与反射 |
| --- | --- | --- |
| `classic` | 插值 `density` 乘方向投影面积；无出生点条件 | Classic phase proposal |
| `conditional29` | 最近一次观测 `(F₀, ∇F₀)` 下的 crossing/exterior 比率 | 完整梯度、Fresnel、镜面反射 |

窄带 Classic 使用 64³ 密度主控网格和 DDA 遍历。每格以密度插值角点的最大值
乘整条飞行方向的保守投影面积界，零上界格直接跳过；候选点以真实
`density(x) × projectedArea(x, w)` 接受。若真实 hazard 超过主控量则报错。

conditional29 和全域 Classic 调用 `sampleFlight`：取指数目标 $E=-\log(1-U)$，对各自的
hazard $h(t)$ 求累计光学深度，并用受保护 Newton 与二分回退求解

$$
\int_a^t h(u)\,du=E, \qquad T(t)=\exp\!\left(-\int_a^t h(u)\,du\right).
$$

积分用自适应 GK15。NVDB 的插值单元边界会加入分段节点；相关尺度和
`render.flight_table_cells`（CLI 为 `--flight-cells`）决定初始分段密度。
格内线性位置只作求根初值，返回的碰撞距离由连续积分残差验收。
`flight_table_cells` 是沿用旧配置键名，**不再定义分段常数 hazard 表**。
每次飞行有独立缓存；重试时保持同一个指数目标并收紧容差。

`conditional29` 在 `external_policy: "sampled_exterior"` 下，入口采样正场值
与完整先验梯度，首段便使用条件模型；窄带入口在真空中时会明确报错。
窄带输入应选 `original_macrofacet`：首段使用 Classic hazard，首次碰撞后
保存完整梯度，下一段转为条件模型。射线离开窄带时 hazard 为零；穿过真空
再进入另一段窄带时保留原出生观测和飞行 age。碰撞后梯度用于下一次条件化；匹配采样的
路径权重只乘 Fresnel 和 Russian roulette 补偿。

conditional29 的目标 hazard 为

$$
h(t)=\frac{p(F_t=0\mid H)}{P(F_t>0\mid H)}
\,\mathbb E[(-d^T G_t)_+\mid F_t=0,H].
$$

因此静态三维 `density` 网格只决定 conditional29 的传输带，无法预存这个
mode 的 hazard：它随出生观测和方向改变。`classic` 也需要逐方向的投影面积。

## 配置、运行与诊断

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json
```

配置中的 `transport.conditional29.sampler` 接受 `regular_tracking` 和旧名
`optical_depth`；求根器只支持 `safeguarded_newton`。旧
`classic_sampler` 是兼容旧配置的字段；实际 Classic sampler 由场的 coverage
决定：`surface_band` 为 `dda_null_tracking`，`full_domain` 为
`regular_tracking`。`correlated_sampler` 只作为 conditional29 sampler 的
兼容后备字段。

`numeric.absolute_tolerance` / `relative_tolerance` 控制积分与光学深度残差，
`distance_absolute_tolerance` / `distance_relative_tolerance` 控制距离误差。
`render_summary.csv` 记录 hazard 求值、求积区间、Newton 与二分次数、最大残差
及数值失败。conditional29 的最终数值失败向 CLI 传播并附带出生状态；
不把它伪装成黑色或逃逸。

模型仍使用最近一次完整观测和局部正侧近似，不能当作完整 GP 首穿真值。
启用 conditional29 的 NEE 会明确报错；确定性距离原子也不会被表示成
连续 hazard。数值求积误差是估计值，不意味着严格无偏。

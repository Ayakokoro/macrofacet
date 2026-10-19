# 固定配置的 GPIS 函数空间参考采样

`sampler.type = "gpis_function_space"` 是独立的 SE 参考后端。它直接生成已有训练接口需要的命中距离、纵向穿越速度和删失标签；现有 GRU / RNO 的输入与网络结构不变。GPIS 示例与该后端默认采用 **Tungsten 式总场值线性求交、割线速度标签**作为主要训练参考，Hermite 保留为可选对照。每个固定采样配置分别生成数据、重新训练权重。原有 circulant / Matérn 状态空间配置仍使用原后端。

## GPIS 的 renewal+ 保留什么

相邻批次之间只传递末端位置的两个观测：

```text
F(x_last), dot(grad F(x_last), omega)
```

其中 `omega` 是接下来采样的光线方向。Tungsten 的
[`GPRealNodeValues::applyMemory`](../../sparse-conv-gpis-tungsten/src/core/math/GaussianProcess.cpp)
在 `RenewalPlus` 分支中将观测向量缩为两个数；
[`FunctionSpaceGaussianProcessMedium`](../../sparse-conv-gpis-tungsten/src/core/media/FunctionSpaceGaussianProcessMedium.cpp)
用同一末端位置上的 `Derivative::None` 和 `Derivative::First` 调用 `sample_cond`。

续批时不是重新独立采样：下一批所有场值联合服从条件高斯分布。`sample_points` 是一批生成多少个点，并非保留多少个历史点。碰撞时场值为零，完整梯度参与散射；沿新方向续采样时，条件使用该梯度在新方向上的投影。renewal（没有 `+`）仅保留场值。

对归一化残差 `Z(q) = F(q)/sigma - b(q)`，`q=s/ell`，SE 核为
`rho(r)=exp(-r*r/2)`。记末端状态为 `(z,d)=(Z(0),Z'(0))`，则

```text
E[Z(r) | z,d] = rho(r) * (z + r*d)
Cov[Z(r), Z(t) | z,d] = rho(r-t) - rho(r)*rho(t)*(1+r*t)
```

这说明 `+` 通过导数保留局部走向。SE 的精确条件分布通常还依赖更早历史；renewal+ 主动丢弃这些信息，是有限记忆近似。

## 本实现的固定参数

```json
"sampler": {
  "type": "gpis_function_space",
  "history_points": 1,
  "batch_size": 8,
  "step_size": 0.015625,
  "rank_tolerance": 1e-10,
  "crossing_rule": "tungsten_linear"
}
```

| 参数 | 含义 |
| --- | --- |
| `history_points = H` | 最近 H 个网格点的残差场值，外加**最新一个点**的解析纵向导数；H、B 都限制在 1–128 |
| `batch_size = B` | 一次联合生成 B 个新点，旧末端锚点不计入 B |
| `step_size = Δq` | 无量纲固定间距，物理间距是 `ell * Δq` |
| `rank_tolerance` | 历史协方差伪逆的相对特征值截断阈值，范围 `[1e-14,1e-4]` |
| `crossing_rule` | 默认 `tungsten_linear`；对照选项 `hermite_residual_cellwise_mean` |

入口先有一个点的值和导数，历史不足 H 时逐批增长，满 H 后丢弃更早场值。`H=1` 对应 renewal+ 的批间传递状态；H>1 是这里新增的扩展。

每批在 `q_last + j*Δq`（j=1…B）联合采样场值与解析导数，批末只保留最新导数。两种求交规则共用相同的随机抽样，固定种子时可比较同一离散实现上的标签差异。线性求交在完整网格区间只使用端点场值，Hermite 还使用端点导数；本次没有将线性模式优化为仅采样场值。条件协方差在轨迹之间复用，不添加独立白噪声 jitter；SE 密集采样导致近奇异时使用上述伪逆阈值，并只裁掉条件协方差中的浮点负特征值。阈值也属于模型对应的参考配置。

网格始终从射线入口固定铺设，不随均值体素边界或几何分段重新开始。终点不必是步长的整数倍：末段在删失位置截断，仍联合生成完整批次中的残差节点，确定性均值仅在射线有效范围内求值。`grid.step_sizes` 可以省略；若提供，必须只有一个值且等于 `sampler.step_size`。不能把不同精度混进一个固定配置数据集。

支持三种原有入口约定：

- `unconditioned_first_zero`：入口场值服从未截断高斯，保留正、负两种符号并追踪首次过零。距离从真实入口计，没有入口碰撞原子。
- `positive_exterior`：条件于入口场值为正。
- `collision_state`：入口场值为零，给定向外纵向导数，追踪后续下穿。

只支持 SE 核，不支持 `fixed_value` 和 `fixed_endpoint`。A/B 数据可用于原 `renewal` 的距离/纵向速度训练；此次没有为有限记忆过程重新定义反弹后的完整三维梯度关联，也没有验证其完整多次反弹渲染分布。

## 主要训练参考：线性求交

令 `f_i = b(q_i) + Z(q_i) = F(q_i)/sigma` 为**总场值**。依次检查相邻网格点，包括批次接缝；第一对沿初始符号离开的一侧跨到另一侧的端点给出：

```text
alpha = f_i / (f_i - f_(i+1))
q_hit = q_i + alpha * (q_(i+1) - q_i)
g_secant = (f_(i+1) - f_i) / (q_(i+1) - q_i)
W = -initial_sign * g_secant > 0
```

物理有符号导数标签为 `(sigma/ell) * g_secant`。CSV 的 `crossing_slope_method` 为 `total_field_linear_secant`。不在线性残差上另加曲线均值求根，也不用解析导数替换割线速度；这两种替换都会改变 Tungsten 式标签。几何分段不会插入额外的过零检测点。

两个端点同号时，线性模式不会检测区间内隐藏的两次过零。`collision_state` 的出生零值也不计为碰撞：只登记严格正值到非正值的下穿，因此若首个新点已为负，首步内的正向离开和快速返回可能漏检。这个限制属于离散参考定义，不能据此声称它比 Hermite 更接近连续 GP 的精确首达分布。主要的 `first_hit_se` 训练使用未条件化入口，不涉及表面出生零值。

当 `q_max` 落在最后一个网格区间内部时，在该位置用两侧已采样残差值及导数作 Hermite 重建，再加有效域内的 `b(q_max)`，作为最后一段线性弦的端点。完整区间仍直接使用采样值；均值不会外推到射线外。改变删失终点可能改变最后一个不完整区间的线性弦和该区间内的标签，之前完整区间保持相同。若需纯采样端点对照，可将 `q_max` 设为 `Δq` 的整数倍。

选择 `hermite_residual_cellwise_mean` 时，保留原残差 Hermite 加精确分段均值的内部求根，速度来自该曲线在根处的导数；CSV 标记为 `analytic_derivative_hermite_cellwise_mean`。切换规则须单独采集和训练，不能只修改旧数据的元信息。

## 与 Tungsten 原版、现有采样器的差别

本实现采用 GPIS 记忆规则，并在 `tungsten_linear` 模式采用其总场线性定位与割线纵向导数公式。仍保留 macrofacet 的入口固定网格、解析导数条件和两种初始符号；未复制 Tungsten 的首点偏移、位置扰动、反弹后的跳点策略或外层对上穿路径的拒绝。末端不足一步的处理也如上单独定义。因此 H=1 表示批间状态具有相同形式，不能据此认为它完整复现 Tungsten 的路径分布。

原 SE circulant 参考一次采完整条射线的相关场值，保留全局协方差，导数来自网格差分。新后端每批根据有限历史继续采样，使用解析导数。有限记忆、批大小、步长与求交方式都可能改变首达分布，不能把已有全历史模型直接当作新过程的准确模型。

固定 H、B 时缩小 Δq，也会缩短历史覆盖距离和批次推进距离，因此不能把它直接当作向全历史 SE 过程的收敛检查。以下是切换默认规则之前的 **Hermite 历史对照**，并非线性模式测量值：2026-10-14 采用 `b(q)=0`、`q_max=4`、种子 `20261014`，每种配置 4096 条未条件化轨迹，得到末端透射率：

| 参考配置 | T(4) | Monte Carlo 标准误差 |
| --- | ---: | ---: |
| 原 circulant，Δq=1/64 | 0.1931 | 0.0062 |
| H=1，B=8，Δq=1/64 | 0.0327 | 0.0028 |
| H=8，B=8，Δq=1/64 | 0.0933 | 0.0045 |
| H=1，B=16，Δq=1/64 | 0.0564 | 0.0036 |
| H=1，B=8，Δq=1/128 | 0.0188 | 0.0021 |

结果体现了不同配置的参考分布差别；与 circulant 的比较也包含导数和求交约定的改变。原始配置、CSV 和汇总保存在本地 `build_gpis_checks/validation/`；重跑这些历史 GPIS 对照时须显式设置 `crossing_rule = "hermite_residual_cellwise_mean"`。这些数据不涉及已训练网络的精度。

## 训练、导出与配置隔离

先构建当前代码，然后从仓库根目录运行：

```powershell
cmake --build build --config Release --target macrofacet_experiments macrofacet_query_first_hit_se --parallel 4
build/Release/macrofacet_experiments.exe first-passage --config configs/first_hit_se_gpis_reference.json
python python/first_hit_se_cli.py collect --config python/configs/first_hit_se_gpis_dataset.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_gpis_gru_h16.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_gpis_rno_h16.json
python python/first_hit_se_cli.py export-model --checkpoint outputs/models/first_hit_se_gpis_linear_h1_b8_dq64_gru_h16/best.pt --output outputs/models/first_hit_se_gpis_linear_h1_b8_dq64_gru_h16/model.json
```

采集配置中的 `reference_sampler` 覆盖各来源的参考后端。示例采用独立的 affine 训练/验证/测试 profiles；相机来源也可使用同一字段。GPIS 采集不会为了凑齐射线长度而调整步长；旧后端仍沿用原有行为。原 `renewal_cli.py collect` 也支持同样的覆盖字段，但须选择 SE 核。

改变 H、B、Δq 或 `crossing_rule` 时，复制采集与训练配置，修改 `reference_sampler`，为 `output_directory`、训练 `dataset` 及模型输出目录使用新路径，然后重新采集、训练。例如 `(H=8,B=16,Δq=1/128)` 应独立于默认 `(1,8,1/64)`。对原 `renewal`，还需删除旧 `reference_steps` 或将它设置为仅含新步长的数组。当前示例输出目录都带 `gpis_linear`，与此前 Hermite 示例隔离。

完整规范化配置（包括 `crossing_rule`）写入 resolved reference、manifest、dataset、checkpoint 和导出的 JSON。打包时拒绝混合配置，并检查 CSV 的斜率标签约定；训练可声明预期配置，评估和 warm start 拒绝不匹配。此前已显式记录 Hermite 规则的 GPIS 数据和模型继续可读；只省略配置中的 `crossing_rule` 会选用新的线性默认值。缺少整个 `reference_sampler` 字段的旧数据/模型继续代表原参考后端，不能与 GPIS 标签混用。

采样器的 B 与训练的 `profiles_per_batch` / `batch_size` 是两种不同概念：后者只是优化器批大小，改变它不改变参考随机过程。网络 `hidden=16` 也不代表保留 16 个 GP 观测。采样器丢弃历史时不应重置网络隐状态；网络学习的是对随机历史积分后的首达统计。

参考步长 `sampler.step_size`、均值几何分段 `profile.maximum_step` 和 RNO 的积分步长 `model.maximum_step` 也分别控制不同对象，不要求相等。

实现不保证给定网络容量已经足够准确；应在同配置的独立测试集上检查似然、透射率曲线和速度分布。单轮小规模训练用于验证接口，不能作为收敛或渲染质量证明。

## Tungsten-compatible two-stage rendering

`gpis_function_space_tungsten` 是独立的新方法名，可分别放在
`transport.first_hit.method` 和 `transport.continuation.method` 中。
`transport.mode` 仍是 `hybrid_first_hit`。两个入口共用 `TungstenFunctionSpaceSampler`，
各自的 `sampler` 参数可以独立设置：

```json
{
  "mode": "hybrid_first_hit",
  "first_hit": {
    "method": "gpis_function_space_tungsten",
    "backend": "scalar",
    "sampler": {"sample_points": 8, "step_size": 0.01}
  },
  "continuation": {
    "method": "gpis_function_space_tungsten",
    "backend": "scalar",
    "sampler": {"sample_points": 8, "step_size": 0.01}
  }
}
```

这里 `step_size` 的单位是**世界坐标长度**，不是旧采样器的归一化 Δq。
默认值为 0.01；设为 0 表示一批覆盖剩余射线区间。`sample_points` 支持 2–128，默认 8。
旧方法 `gpis_function_space` 以及训练/数据采集参考的行为保持不变。

支持范围为本地 Tungsten 的 standard SE GP、`correlation_context="renewal+"`、
`intersect_method="gp_discrete"`、`normal_method="conditioned_gaussian"`、`skip_space=0`。
这四项可在 `sampler` 中显式写出；不支持的模式、旧的 `history_points` / `rank_tolerance`
等参数会报错。要求 scalar backend、generalized_gaussian、无 alpha grid。
不读取 checkpoint，不使用 cubic/point_linear profile；兼容首段的 `profile_mode` 若显式配置，
应为 `tungsten_direct`。

场景配置：

- [σ=0.05，两阶段 GPIS](../configs/render_first_hit_gpis_tungsten_shader_ball_sigma005.json)
- [σ=0.02，两阶段 GPIS](../configs/render_first_hit_gpis_tungsten_shader_ball_sigma002.json)

### 采样行为

- 每批先抽一个位置偏移，按原式布置 N 个点。首点为 near+0.1h，末点为 near+Nh；h 受物理步长和剩余区间约束。批次之间不额外搜索未采样的小间隙。
- 第一批调用原版正值起点采样，保留随机数消耗，但 numCondPts=0，实际批次仍无条件。
- 每批只联合采样场值。命中使用线性插值，构造交点场值及割线纵向导数，再根据当前批次保留下来的观测条件采样两个横向梯度。无命中时条件采样末端三个梯度分量，最终出域批次也消耗梯度随机数。
- `renewal+` 将状态压缩为末端场值与完整梯度在新方向上的投影。交接保留原插值场值、double 精度的 GP 观测位置和完整梯度，分别保存 float 精度的实际散射位置。
- 表面出生的第一批跳过首个符号变化区间；同一射线后续批次不重复跳过。上穿、非有限梯度或梯度模方小于原阈值 1e-7 的命中按失败终止，不作为逃逸，也不翻转场符号。
- 条件高斯使用物理单位协方差，伪逆阈值为 1e6*DBL_EPSILON*max(abs(eigenvalues))。抽样先尝试 Cholesky，失败才使用截断负特征值的平方根，不添加 jitter。
- PCG 浮点均匀随机数及成对 Box–Muller 遵循原版。两个兼容阶段共用一个 GPIS 随机流，种子取自 `event_rng.integer()`。仅后续段启用兼容方法时，种子取自 `continuation_rng.integer()`，不扰动旧 first-hit 的独立随机流。
- NanoVDB 提供独立兼容查询：模拟 identity transform、normalize_size=false、linear VDB 的 float 坐标转换、两体素边界裁剪和 OpenVDB BoxSampler 浮点插值；均值梯度采用固定世界步长 0.001 的前向差分。现有渲染和训练的均值查询不变。

首段成功才交给后续段，接口保留 Escape / Hit / RejectedUpcrossing / InvalidGradient。
首段也可以选择旧参考或 neural，再接兼容 GPIS；此时复现的是给定交点与梯度之后的条件采样，
整条路径的分布仍受首段方法影响。`renewal+` 不要求传递首段全部观测；GPIS 的 global 历史模式不在此次范围内。

### 验证与边界

`tools/generate_tungsten_function_space_fixture.py` 从指定的本地 GPIS 仓库只读提取
`intersectGP`、`makeIntersect`、`sampleGrad`、`applyMemory`、条件高斯和高斯抽样原函数，
在不链接 macrofacet 的独立程序中运行。辅助程序提供解析 SE 协方差和常量/平面均值，
禁用范围外的协方差投影、其他求解器和 CSG。
`tests/data/tungsten_function_space_reference.json` 保存源文件 SHA256、随机输入和参考批次。
测试重放输入，比较场值、交点、完整梯度、观测位置与随机数消耗；浮点结果按数值容差比较。
另外覆盖首区间跳过、截短末批、原版 OpenVDB BoxSampler、两阶段与手动多次反弹一致性、
线程确定性和配置校验。

重新生成参考数据：

```powershell
python tools/generate_tungsten_function_space_fixture.py --reference D:/projects/sparse-conv-gpis-tungsten --packages build/vcpkg_installed/x64-windows
```

兼容范围是 GPIS 介质采样。相机、场景边界求交、conductor 材质、环境和轮盘赌仍属于
macrofacet 渲染器；不宣称相同 seed 下整张图与 Tungsten 逐像素一致。两个场景的均值数据、
σ、相关长度、物理步长和边界必须一致才能进行有效图像对照。

`hybrid_summary.json` 和 `render_summary.csv` 新增 gpis_flights、gpis_continuation_flights、
gpis_batches、gpis_points、gpis_rejected_upcrossings、gpis_invalid_gradients。
可选 first-hit CSV 追加 failed、field_value、observation_x/y/z 和 gpis_seed。
上穿拒绝计入独立计数，不计为 numerical_failures。

## Global conditional 首段，GPIS 后续段（2026-10-17）

`hybrid_first_hit` 支持 `first_hit.method="global_conditional"`，可与
`continuation.method="gpis_function_space_tungsten"` 组合。完整示例：
[σ=0.02 配置](../configs/render_first_hit_global_conditional_gpis_shader_ball_sigma002.json)。

```json
"transport": {
  "mode": "hybrid_first_hit",
  "first_hit": {"method": "global_conditional", "backend": "scalar"},
  "continuation": {
    "method": "gpis_function_space_tungsten",
    "backend": "scalar",
    "sampler": {"sample_points": 8, "step_size": 0.01, "correlation_context": "renewal+"}
  },
  "roulette_start_depth": 8
}
```

首段复用 `startConditionalExterior`：在真实介质入口采样
`F ~ N(m, sigma²) | F>0`，以及独立的完整 SE 先验梯度；正值条件施加于随机总场，
并非要求 VDB 均值 `m>0`。随后固定这份入口观测，以原 `ConditionalMedium` 的
解析条件消光率和分段 delta tracking 采样首碰。距离从介质入口起算，不含相机到入口的真空段。
这里没有启用 `global_conditional_renewal_plus` 的途中观测刷新。

这是条件于正侧入口的传输，入口权重为 1，不另乘 `Phi(m/sigma)`；它与现有
无条件、保留正负入口的 first-hit 教师不同，也不同于 Tungsten 未条件化的首批采样。
首段直接查询均值场，不使用训练模型、三次剖面或 GPIS 的采样步长。
要求 scalar、SE 核、generalized_gaussian、禁用 alpha grid；首段配置中不接受
`model`、`sampler`、`profile_maximum_step` 或 `batch_size`。

命中时用 `sampleCollisionGradient` 采样通量加权的条件完整梯度，只做一次反射与 Fresnel。
传给 GPIS 的状态为碰撞观测位置、`F=0` 和未归一化完整梯度；后续从 surface-born
renewal+ 开始，不再抽取入口或首碰法线。GPIS 查询仍遵循其 float 射线约定，
条件观测位置保留 double。它只保留出生点条件，不恢复相机段的完整 GP 历史。

首段入口与 delta tracking 使用 event 随机流，碰撞梯度使用 gradient 随机流，
GPIS 的 PCG 种子来自 continuation 随机流；更换后续方法不改变首段样本。
`hybrid_summary.json` 中 `conditional_first_hit_flights` 单独统计进入域的 conditional 首段；
`conditional_flights` 是全部 conditional 航段，`continuation_flights` 不含首段。

```powershell
.\build\Release\macrofacet_experiments.exe render --config configs/render_first_hit_global_conditional_gpis_shader_ball_sigma002.json
```

## Local 首段，GPIS 后续段（2026-10-17）

`hybrid_first_hit` 支持 `first_hit.method="classic_local"`，完整示例：
[σ=0.02 配置](../configs/render_first_hit_local_gpis_shader_ball_sigma002.json)。
相机、VDB、粗糙度和后续 GPIS 参数与上面的 global_conditional 首段配置相同。

```json
"transport": {
  "mode": "hybrid_first_hit",
  "first_hit": {"method": "classic_local", "backend": "scalar"},
  "continuation": {
    "method": "gpis_function_space_tungsten",
    "backend": "scalar",
    "sampler": {"sample_points": 8, "step_size": 0.01, "correlation_context": "renewal+"}
  },
  "roulette_start_depth": 8
}
```

首段使用现有 `NarrowBandMedium` 的场景烘焙密度和 DDA delta tracking，距离从介质入口起算。
其材质固定为 `local_tangent`：非零均值梯度归一化，梯度协方差沿用配置的粗糙度。
首段独立持有 local 材质及其 projected-area majorant，后续的 `global_pointwise` 设置不会改变它。
这里不抽取入口 GP 值和梯度，也不施加入口 `F>0` 条件。

命中后从 local 材质的通量加权完整高斯梯度分布采样，相当于直接采样 local target VNDF。
首碰固定采用这个精确 proposal，`transport.classic_phase_proposal` 仍只控制 classic 后续段。
反射和 Fresnel 仅计算一次，无额外法线 PDF 权重。GPIS 接收碰撞位置、`F=0` 和未归一化的
完整 local 梯度，再按现有 surface-born renewal+ 算法处理后续段。这是 local 首碰与 GPIS
后续的混合模型；首段没有生成 GPIS 随机场历史，后续也不会恢复该历史。

首段要求 scalar、generalized_gaussian、禁用 alpha grid；不接受 `model`、`sampler`、
`profile_maximum_step` 或 `batch_size`。`profile_mode` 如显式配置，必须是 `local_direct`。
GPIS 后续段仍要求 SE 核。event、gradient、continuation 随机流分别用于首碰距离、首碰梯度、
后续采样，切换后续策略不会改变首碰样本。
`hybrid_summary.json` 的 `classic_first_hit_flights` 单独统计 local 首段；
`classic_flights` 包含全部 classic 航段，`continuation_flights` 不含首段。

```powershell
.\build\Release\macrofacet_experiments.exe render --config configs/render_first_hit_local_gpis_shader_ball_sigma002.json
```

## Conditional renewal+ 首段 0.1，GPIS 后续段（2026-10-18）

[完整配置](../configs/render_first_hit_renewal_plus_gpis_shader_ball_sigma002.json) 沿用上面的
sigma=0.02 VDB、相机、粗糙度、512×512 / 128 spp 和 GPIS 后续段设置：

```json
"transport": {
  "mode": "hybrid_first_hit",
  "first_hit": {
    "method": "global_conditional_renewal_plus",
    "backend": "scalar",
    "renewal_distance": 0.1
  },
  "continuation": {
    "method": "gpis_function_space_tungsten",
    "backend": "scalar",
    "sampler": {
      "sample_points": 8, "step_size": 0.01, "correlation_context": "renewal+",
      "intersect_method": "gp_discrete", "normal_method": "conditioned_gaussian", "skip_space": 0.0
    }
  },
  "roulette_start_depth": 8
}
```

`first_hit.renewal_distance` 是世界距离，默认 0.01，必须有限且大于零；
独立于 `transport.global_conditional_renewal_plus` 和 `continuation` 的参数。
首段的 `step_size` 会报错，避免与 GPIS 的采样点间距混淆。

相机从介质入口采样正的总场值和完整先验梯度，每个区间内复用 conditional delta tracking。
区间末端未碰撞且仍需前进时，从 `p(F,G | 当前起点观测, F>0)` 联合采样端点，
携带值与完整梯度到下一区间。总距离从原介质入口累计，最后一个区间按剩余长度截短。
这仍是只保留最近观测的正端点近似，没有条件化整段连续未碰撞事件。

命中后把碰撞位置、`F=0` 和未归一化完整梯度交给 GPIS，沿用已有的一次反射、Fresnel、
俄罗斯轮盘和 surface-born 条件化。入口、距离和正端点采样使用 event 随机流；
真实碰撞梯度使用独立 gradient 随机流，GPIS 种子来自 continuation 随机流。
当更新距离覆盖整条首段时，首碰与 `global_conditional` 的样本和权重一致。

`hybrid_summary.json` 的 `first_hit_renewal_distance` 记录实际设置，
`conditional_first_hit_renewal_segments` 单独统计首段区间数；
`global_conditional_renewal_plus_segments` 统计首段与后续段的总区间数。
首段要求 scalar、SE 核、generalized_gaussian、禁用 alpha grid；不加载神经网络或三次剖面。

```powershell
.\build\Release\macrofacet_experiments.exe render --config configs/render_first_hit_renewal_plus_gpis_shader_ball_sigma002.json
```

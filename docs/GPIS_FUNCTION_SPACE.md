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

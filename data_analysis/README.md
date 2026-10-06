# First-passage data visualization

固定距离发生首次碰撞的斜率分布使用 [固定终点条件采样](../docs/FIXED_ENDPOINT_SAMPLING.md)：

```powershell
build\Release\macrofacet_experiments.cmd first-passage --config configs\shader_ball_fixed_endpoint_se.json
if ($LASTEXITCODE -ne 0) { throw "Fixed-endpoint generation failed" }
python data_analysis\plot_fixed_endpoint.py --config data_analysis\configs\shader_ball_fixed_endpoint_se.json
```

默认固定物理距离 `0.004`，不是距离区间。Python 用“此前存活 × 负斜率”的通量权重绘制
直方图，图例报告有效样本量 ESS。配置 `bins`、`distances`、`slope_units` 可调整展示。

完整 NanoVDB mean 的 SE 示例参见 [全域 SDF 实验说明](../docs/FIRST_PASSAGE_FULL_FIELD.md)。
先运行 `configs/shader_ball_full_transmittance_se.json` 生成数据，再运行：

```powershell
python data_analysis\plot_first_passage.py --config data_analysis\configs\shader_ball_full_transmittance_se.json
```

该配置启用完整 SDF profile、透射率、hazard 和累计 hazard 图。`x_axis: "distance"`
显示场景单位距离，并把 hazard/density 转换为每场景单位；省略时仍用无量纲 q。

球面 mean 下的 survival、hazard、累计 hazard 和沿射线 SDF mean profile：

```powershell
build\Release\macrofacet_experiments.cmd first-passage --config configs\sphere_ray_transmittance_se.json
if ($LASTEXITCODE -ne 0) { throw "Sphere first-passage generation failed" }
python data_analysis\plot_first_passage.py --config data_analysis\configs\sphere_ray_transmittance_se.json
```

该配置与 `sphere_fixed_endpoint_se.json` 使用相同的球面、起点、观测梯度、方向、kernel、
距离范围和 Monte Carlo 预算，但只进行起点条件化，不固定终点为零。输出单独保存到
`outputs/sphere_ray_transmittance_se`，不重写已有固定终点斜率数据。
`survival` 是整段路径的透射率；固定终点提案的接受率不是这条 survival 曲线。
mean profile 图显示确定性的球面 SDF `m(b+t*d)`，不是起点条件化后的 GP 后验均值。

这里严格区分两层职责：

- C++ `first-passage` 实验只生成 Monte Carlo 样本、风险集统计和 provenance JSON。
- `data_analysis/` 只读取已有 CSV，选择要比较的 kernel/状态，并生成 SVG；不会重新采样 GP，也不会修改训练数据。

脚本只依赖 Python 3.10+ 标准库，不需要安装到 `python/` 的训练包中。所有相对路径都按项目根目录解析，因此命令可以从项目根目录直接运行：

```powershell
python data_analysis\plot_first_passage.py --config data_analysis\configs\first_passage_kernels.json
python data_analysis\plot_first_passage.py --config data_analysis\configs\collision_state_kernels.json
python data_analysis\plot_first_passage.py --config data_analysis\configs\collision_state_matern52_parameter_study.json
```

SE 平面 mean SDF 下，固定起点场值和梯度的单射线透射率示例：

```powershell
build\Release\macrofacet_experiments.cmd first-passage --config configs\plane_ray_transmittance_se.json
if ($LASTEXITCODE -ne 0) { throw "First-passage data generation failed" }
python data_analysis\plot_first_passage.py --config data_analysis\configs\plane_ray_transmittance_se.json
```

CMake 从选定的 Python 中自动查找 LibTorch，并为 Windows 构建生成 `.cmd`
启动脚本。该脚本仅在启动程序时临时设置 DLL 搜索路径，直接使用 Python 安装目录中的
DLL，不复制文件，也不修改系统 PATH。请使用上述 `.cmd`；直接运行 `.exe` 时仍需要
自行设置当前终端的 PATH。旧构建请先执行
`cmake --build build --config Release --parallel 4`。
上面的退出码检查会在生成失败时停止，避免随后用旧 CSV 重新绘图。

该示例设 mean 平面法线为 `(0, 0, 1)`，起点位于 mean 平面上，射线方向为
`(sqrt(3)/2, 0, 1/2)`，观测到 `F(b)=0`、`grad F(b)=(0, 0, 1.2)`。
`sigma=0.2`、`ell=1`，因此生成配置中的
`(beta_0, beta_a, beta_g)=(0, 2.5, 3)`。平面的 mean 斜率由 `beta_a`
表示，`process.mean=0` 是碰撞状态模式的配置约定。
`survival` 即整段路径的透射率，输出到
`outputs/plane_ray_transmittance_se/first_passage_survival.svg`；横轴为 `q=t/ell`。

## 配置

通用字段：

- `input_directory`：C++ 输出目录，必须包含 `first_passage_curves.csv`。
- `output_directory`：SVG 和派生直方图 CSV 的写入目录；省略时等于输入目录。
- `minimum_risk_set`：hazard 曲线保留一个 bin 所需的最小风险集。
- `plots`：启用或关闭各图。

碰撞状态输出还支持：

- `comparison: "kernels"`：固定唯一的 `state_ids` 项，对比所有 kernel。
- `comparison: "states"`：固定 `kernel_id`，按照 `state_ids` 的顺序对比状态。
- `plots.crossing_slope_density`：从 C++ 原始样本按 `(q_begin, q_end]` 分组并绘制条件斜率密度。该功能要求数据生成配置设置 `monte_carlo.write_raw_samples=true`。

斜率密度会额外写出 `first_passage_crossing_slope_density.csv`，方便核对每条直方图是否积分为 1。它是可重复生成的分析产物，不是 C++ 训练数据。

## 输出

根据配置产生：

- `first_passage_survival.svg`
- `first_passage_hazard.svg`
- `first_passage_cumulative_hazard.svg`（碰撞状态）
- `first_passage_density.svg`（可选）
- `first_passage_rice_density.svg`（固定起点实验）
- `first_passage_crossing_slope_density_q*.svg` 与对应派生 CSV（可选）


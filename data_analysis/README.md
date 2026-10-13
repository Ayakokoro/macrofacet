# First-passage data visualization

## SE 相机像素：classic_local / global_conditional / neural

[16 个固定像素的交互对比](../outputs/se_camera_transmittance_20261010/comparison.html) 复用现有 SVG 绘图、PFM 预览、相机光线检查、`curves` 与 GP `first-passage` 采样工具。点击场景图中的编号或使用下拉菜单切换像素；单光线页面支持完整光线／下降区域切换、悬停读数、相对 GP 参考的偏差、均值场及 CSV/SVG 下载。

从仓库根目录生成数据和页面（需要已构建的 Release 工具、SE h16 模型，以及用于 PFM 预览的 NumPy）：

```powershell
python tools/compare_se_camera_transmittance.py --output outputs/se_camera_transmittance_20261010 --gp-trials 65536
```

默认沿用已有 16 个固定像素，每种传统方法每条光线采样 32,768 次，neural 使用 SE 渲染配置中的模型、分段和推理后端。`--gp-trials 65536` 为每条场内光线增加各 65,536 条粗／细 GP 参考轨迹，已有传统方法、neural 数据和预览会复用。省略该参数可只生成三方法比较；如果目录中已有完整 GP 数据，绘图仍会加载它。只重新绘图可运行：

```powershell
python data_analysis/plot_transport_camera_rays.py outputs/se_camera_transmittance_20261010
```

指定其他像素时使用新的输出目录，例如 `--output outputs/se_camera_selected --pixels 256,256 144,368 256,440`；坐标从左上角以 0 开始。`--config` 可指定 SE 渲染配置，`--trials` 控制每种传统方法的样本数，`--workers` 控制同时运行的采样进程数。相同输入可恢复已完成的像素；配置、模型、场或可执行文件哈希改变时必须换输出目录。

比较固定为每个像素的第 0 次相机采样、同一 SE 场及相同有效域入口。`global_conditional` 对入口处 `F>0` 的随机场值／梯度观测取平均，与 neural 的 A 起点条件对应。`classic_local` 使用渲染器的烘焙密度与局部 NDF；条件模型使用完整 VDB 均值；neural 保留配置中的均值近似。GP 参考使用相同的 `F>0` 条件、完整 VDB 均值、条件循环嵌入网格及 Hermite 首次零穿越定位；归一化步长上限分别为 `1/64`、`1/128`，两档使用独立种子。采样曲线带逐点 95% Wilson 区间，neural 是模型直接计算的透射率。GP 是数值真值参考，置信区间只覆盖 Monte Carlo 误差，不覆盖剩余离散误差。

2026-10-10 的 16 点对比中，15 条光线进入有效域，1 条为真空。三组方法的逐光线最大差及其均值保存在 `summary.csv/json`；输入与哈希保存在 `selection.json`。中央 `(256,256)` 的 classic／neural、conditional／neural 最大差分别为 13.56／13.07 个百分点，两种传统方法之间为 1.65 个百分点。所有场内光线中，neural 与 classic／conditional 的最大差为 16.48／16.20 个百分点，均出现在 `(144,368)`；这些是固定像素上的方法差异，不是整图误差。

加入 GP 后，中央像素的 classic、conditional、neural 相对细参考最大绝对偏差分别为 3.57、2.57、12.04 个百分点；15 条场内光线的逐光线最大偏差均值分别为 1.22、0.97、8.77 个百分点，最坏值为 3.57、3.63、16.10 个百分点。GP 粗细参考的全组最大差为 0.710 个百分点，各光线粗细差均未超过该光线的双样本 95% DKW 界；这只是采样噪声下的经验一致性检查，不是离散误差的严格上界。参考的逐体素三次均值与直接 VDB 查询最大差为 `1.92e-15`。GP 参数及哈希在 `gp_selection.json`，每个像素的参考配置与运行结果在 `pixel_*/gp/`。更新后的页面保留方法间差值于 CSV/JSON，并以相对 GP 的偏差作为主图和汇总表。

## SE 首次命中后：固定起点观测的 B 段

[命中后透射率页面](../outputs/se_surface_transmittance_20261010/comparison.html) 与上述相机页面互相链接，A/B 切换保留选中的像素。仍使用原来的 16 个像素、下拉菜单、预览图标记、局部／完整曲线和悬停读数。

```powershell
cmake --build build --config Release --target macrofacet_inspect_renewal_ray --parallel 4
python tools/compare_se_camera_transmittance.py --output outputs/se_surface_transmittance_20261010 --after-first-hit --camera-suite outputs/se_camera_transmittance_20261010 --gp-trials 65536
```

`--after-first-hit` 按第 0 个相机采样的 RNG 顺序，用配置后端、batch=1 重放 neural 首次命中，再使用同一个位置 `p`、完整梯度 `g` 和镜面反射方向 `ω` 比较下一段飞行。可单独调用 `macrofacet_inspect_renewal_ray <config> <output> <x> <y> --after-first-hit` 导出这些条件。它复用渲染器的碰撞梯度重建；GPU 批大小不同可能带来浮点差异，并非整帧 wavefront 的逐位日志。

- `global_conditional` 使用 `fixed_observation`，固定 `F(p)=0`、`∇F(p)=g`。
- neural 使用表面出射 B 初始化，斜率为 `ℓ(ω·g)/σ > 0`。
- GP 使用同一观测的 `collision_state` 首达参考。当前各向同性 SE 核沿直线的标量过程只需要梯度投影 `ω·g`；粗／细步长上限为 `1/64`、`1/128`，每档 65,536 条独立轨迹。
- `classic_local` 没有起点条件化接口，保留同一位置、方向上的局部基线，不能将其标成使用了完整起点观测的条件模型。

这些曲线只条件化于单个命中点，没有加入此前整段路径的存活历史；命中来自 neural 采样，不是固定 GP 实现上的命中。`F(p)=0` 是随机场观测，均值 SDF 在该点可以不为零。横轴从命中点开始；沿用的 CSV 字段 `distance_from_entry` 此时表示离命中点的距离，`distance_from_camera` 表示从相机累计的折线路程。

这组数据有 11 条命中后的 B 光线、4 条相机采样直接逃逸、1 条未进入有效域。没有命中的像素不重采样、不生成虚构的 B 曲线；其汇总值为 `null`，页面显示“—”。中央 `(256,256)` 的 classic_local、global_conditional、neural 相对细 GP 参考的沿程最大绝对偏差分别为 **19.35、0.70、27.83 个百分点**。11 条 B 光线的逐光线最大偏差均值为 **13.46、1.11、12.94 个百分点**。GP 粗细最大差为 **0.670 个百分点**；这些数值不是整图误差，GP 仍含采样与离散误差。

`--camera-suite` 要求同一配置、模型、场和像素列表，并复用其预览。配置或可执行文件改变后请换新的输出目录；仅重绘可直接运行 `python data_analysis/plot_transport_camera_rays.py outputs/se_surface_transmittance_20261010`。每个像素的 `neural/ray.json` 和页面顶部展示命中位置、梯度、反射方向及起点斜率，`comparison_report.json` 和 `summary.json` 保留完整比较条件。

## σ=0.05、固定粗糙度的 SE 对比

[相机入射 A 段](../outputs/se_camera_transmittance_sigma_0.05_20261010/comparison.html) 和 [首次反射 B 段](../outputs/se_surface_transmittance_sigma_0.05_20261010/comparison.html) 使用 `shader_ball_full_0.05.nvdb`，保持 `material.roughness=0.5`，由烘焙文件中的 `σ≈0.05` 推导 `ℓ≈0.1414213583`。配置为 [SE h16 σ=0.05](../configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json)。沿用原来的相机、16 个像素、SE h16 模型、point_linear 输入及采样预算；A/B 页面使用新场重新渲染的 4 spp 预览，并在顶部显示场文件和参数。

```powershell
python tools/compare_se_camera_transmittance.py --config configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json --output outputs/se_camera_transmittance_sigma_0.05_20261010 --gp-trials 65536
python tools/compare_se_camera_transmittance.py --config configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json --output outputs/se_surface_transmittance_sigma_0.05_20261010 --after-first-hit --camera-suite outputs/se_camera_transmittance_sigma_0.05_20261010 --gp-trials 65536
```

B 段使用这组新参数下重新采样的首次命中位置、完整梯度和反射方向；不沿用 σ=0.02 的命中状态。`classic_local` 仍为忽略起点观测的基线，其余三条使用相同起点条件。比较结果及 GP 步长减半检查见各目录的 `summary.json`；`validation.json` 和 `browser_validation.json` 分别记录数据与交互验证。

这组有 16 条入射 A 光线、10 条首次反射 B 光线；另外 6 条相机采样直接逃逸，没有 B 曲线。相对细 GP 参考，逐光线沿程最大绝对偏差的均值如下，单位为百分点：

| 光线段 | classic_local | global_conditional | neural h16 | GP 粗细步长最大差 |
| --- | ---: | ---: | ---: | ---: |
| 入射 A（16 条） | 1.33 | 1.44 | 20.97 | 0.87 |
| 首次反射 B（10 条） | 11.31 | 0.46 | 13.06 | 0.39 |

这些是所选光线的统计，不是整图误差；GP 参考仍有采样和离散误差。新旧 NVDB 的有效域与体素间距也不同，因此与 σ=0.02 页面的差异不能仅归因于相关长度。

## 同一首次碰撞位置的条件梯度分布

[交互页面](../outputs/se_collision_gradients_sigma_0.05_20261011/comparison.html) 沿用 σ=0.05、α=0.5 的 16 个像素；固定各像素已有 neural sample 0 的首次命中距离，比较 classic_local、global_conditional、neural h16 与 GP。有 10 个有效位置，6 个没有首次命中的像素不补造目标位置。A/B 透射率页面顶部可切换到梯度分布，并保留选中的像素。

```powershell
cmake --build build --config Release --target macrofacet_inspect_collision_gradients macrofacet_experiments --parallel 4
python tools/compare_se_collision_gradients.py --camera-suite outputs/se_camera_transmittance_sigma_0.05_20261010 --output outputs/se_collision_gradients_sigma_0.05_20261011
```

每种模型采样 65,536 次，GP 每档 65,536 个固定终点提案，归一化步长分别不大于 1/64、1/128。固定的是首次碰撞位置，梯度重新采样；不是固定梯度后的退化分布，也不使用距离区间筛选。

- classic_local 复用当前材质的高斯梯度及负斜率通量采样。
- global_conditional 从入口 F>0 的观测分布采样完整值/梯度，使用 null tracking 判断是否存活到目标，权重为存活指示量乘目标 hazard。这样实现按首次碰撞密度混合入口条件，而非等权平均各条件分布。
- neural 使用同一目标位置的混合速度分布及渲染器的横向梯度重建，不再加一次通量权重。
- GP 使用项目固定终点采样器的 `positive_exterior` 分支，在终点 F=0 的提案上筛选正入口及此前存活，再施加负斜率通量权重。完整各向同性 SE 梯度的两个横向残差与直线上的场值过程独立，按对应高斯分布重建。

页面支持入射梯度、两个横向分量、模长、法线夹角和世界坐标三个分量，以及 PDF/CDF 切换、悬停读数、SVG/CSV 下载。PDF 使用覆盖所有正权重样本的共同分箱；CDF 在这些分箱边界计算。统计表列出加权均值、分位数和 ESS，GP 粗细 CDF 差同时包含离散及采样差异。物理梯度单位下的“均值场梯度”不是条件化后的随机梯度。

仅重绘：`python data_analysis/plot_collision_gradients.py outputs/se_collision_gradients_sigma_0.05_20261011`。输入、可执行文件或样本预算变化时使用新目录。

## 第二次碰撞点的条件梯度分布

[第二次碰撞交互页面](../outputs/se_second_collision_gradients_sigma_0.05_20261011/comparison.html) 复用同一 SE 场、相机和 16 个像素。原 sample 0 的 10 条 B 段全部逃逸，因此这组固定已有的第一次命中位置、完整梯度和反射方向，选取 neural 在域内发生第二次碰撞条件下的距离中位数作为共同目标。该目标是条件分位点，不冒充原样本的第二次命中；其位置、条件碰撞概率、目标 optical depth 和原逃逸状态写入 `target_selection.json`。另外 6 个没有第一次命中的像素仍无比较位置。

```powershell
cmake --build build --config Release --target macrofacet_inspect_collision_gradients --parallel 4
python -B tools/compare_se_collision_gradients.py --camera-suite outputs/se_camera_transmittance_sigma_0.05_20261010 --surface-suite outputs/se_surface_transmittance_sigma_0.05_20261010 --target-policy conditional_median --output outputs/se_second_collision_gradients_sigma_0.05_20261011
```

`--target-policy replayed` 只使用原样本实际命中；`conditional_random` 从碰撞发生条件下的距离分布随机选择目标，等价于保持第一次命中条件并对第二段距离重复采样直到命中。条件分位点通过导出的四系数 Bernstein hazard 精确积分、单调求逆获得，没有用 CSV 距离分箱代替。

global_conditional 的起点观测在本组固定，目标距离也固定，因此共同的 survival×hazard 因子在归一化后的梯度分布中抵消，直接复用生产代码的条件高斯通量采样。neural 使用 B 初始化。GP 使用已有 `collision_state` 固定终点桥，保留起点值/方向导数、终点值、区间存活筛选和负斜率通量权重。它们只使用第一次命中点的观测，没有加入此前相机路径的存活历史；classic_local 保留忽略此观测的局部基线。

GP 的完整 SE 横向梯度使用条件均值 `∇m(p₂)+ρ(g₁−∇m(p₁))` 的横向投影和方差 `(σ/ℓ)²(1−ρ²)`，其中 `ρ=exp(−(s/ℓ)²/2)`。页面保留首次与第二次梯度、A/B 透射率之间的选点导航。数据检查额外用独立 Simpson 积分核对 50% 条件分位点，并核对 GP 起点/终点约束及横向条件分布。

## GP 首达数据

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

C++ no longer requires LibTorch. Windows `.cmd` launchers and direct `.exe`
invocation are both supported. Reconfigure and rebuild older build trees before
running the commands above. The Python analysis scripts use the standard library.

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


# Renewal+ C++ 采样、法线与路径追踪

实现方案中的碰撞距离、穿越速度、完整梯度／法线采样、模型透射率和多次散射路径追踪。C++ 保留 Eigen 标量推理，并支持 LibTorch CPU/CUDA 多射线批量推理；两者加载相同 JSON 权重。核契约固定为 `rho(x)=(1+x)exp(-x)`、`beta=1`；支持空域起点 A 和已知表面向外起点 B。

## 神经渲染

已有 checkpoint 无需重新训练，先导出完整权重，再运行新增模式：

```powershell
python python/renewal_cli.py export-model --checkpoint outputs/models/renewal_matern32_h64_v2/best.pt --output outputs/models/renewal_matern32_h64_v2/renewal_model.json
cmake -S . -B build -DMACROFACET_ENABLE_TORCH=ON
cmake --build build --config Release --parallel 4
build/Release/macrofacet_experiments.exe render --config configs/render_neural_renewal_shader_ball.json --width 64 --height 64 --spp 32
```

输出包括 `render_neural_renewal_white.pfm/.bmp`、`render_neural_renewal_directional.pfm/.bmp`、`render_summary.csv` 和记录模型哈希的 `resolved_config.json`。白炉版本使用单位 Fresnel；方向环境版本使用配置中的导体 Fresnel。汇总新增 `neural_flights`、`neural_segments` 和 `neural_mixture_queries`，最后一项应等于真实碰撞数。

场景中配置：

```json
"transport": {
  "mode": "neural_renewal",
  "renewal": {
    "model": "outputs/models/renewal_matern32_h64_v2/renewal_model.json",
    "profile_maximum_step": 0.25,
    "backend": "auto",
    "batch_size": 4096
  },
  "roulette_start_depth": 8
}
```

同时使用 `field.kernel_type="matern_3_2"` 和 `material.ndf_family="generalized_gaussian"`。指定三个相等的 `correlation_lengths`，或使用 `material.roughness` 派生相关长度。固定正定椭圆各向异性也可使用：每个方向取 `ell=1/sqrt(direction^T M direction)`，梯度按方案第 15 节的条件协方差重建。空间变化的核或 alpha 网格不属于这个平稳模型；导入 NanoVDB 时设置 `use_alpha_grid=false`。

神经模式需要完整 SDF，不能把窄带外的零背景当作真实均值。程序化场景自动烘焙完整域；导入的 NanoVDB 必须标记 `coverage=full_domain`，且沿射线的插值节点完整。外部相机射线在活动域入口执行 A 初始化，离开活动域后为真空。已有三个输运模式仍可使用，`--mode all` 维持原三个模式的比较语义。

路径追踪沿用项目现有的镜面导体和环境光模型：每次神经求交采样完整梯度，以其单位法线反射，乘导体 Fresnel，保存完整梯度供下一条射线 B 初始化。几何样本不额外除以命中 PDF，也不重复乘 Rice 权重。沿直线保留 GRU 状态，碰撞换方向时重新初始化；俄罗斯轮盘和 `render.safety_depth_cap` 与现有路径追踪一致。有限深度截断仍会带来偏差，应检查汇总中的 `safety_cap_terminations`。

`RenewalMedium::surfaceTransmittance(ray, fullGradient, distance)` 提供阴影查询：新方向重新投影同一梯度并建立 B 条件，向内方向返回零。当前镜面导体路径通过 BSDF 方向到达环境光，不执行独立的环境光 NEE；阴影接口可供以后新增的非 delta 材质／显式光源调用。

## 导出与运行

### 当前点查询：`point_linear`

`transport.renewal.profile_mode` 默认为 `cubic`，也可以设为 `point_linear`。
后者只在每个实际访问段的起点查询一次均值和梯度，以局部切线代替该段的真实均值曲线。
它不做四点采样或三次系数恢复，不预先构建整条射线的剖面；碰撞后的新射线重新建立点查询。
NanoVDB 使用逐格推进，每段在下一个格子边界或 `profile_maximum_step` 处结束。
`profile_maximum_step` 的单位仍是 `x=s/ell`，不是世界距离或体素数量。

设当前点归一化均值为 `b=m/sigma`，方向导数为 `g=ell*dot(gradient,direction)/sigma`，
当前段长度为 `dx`，则送入现有网络的五个特征为：

```text
[b, b + g*dx, g*dx, g*dx, log(dx)]
```

第二个值是切线外推，不查询段尾。沿射线保留原 GRU 状态、Bernstein hazard 积分和求逆；
命中时沿用 mixture，并在实际命中点查询梯度以重建法线。起点的梯度复用首次点查询结果。
标量、LibTorch CPU 和 CUDA 后端均支持，模型透射率也沿用同一条按需生成的分段序列。
短距离透射率不构建查询范围之后的段。完整域要求保留，访问每格时在读取八个角点的同时
检查节点活跃性和有限性；不提前扫描未访问的整条射线。缺失节点仍会报错。

此模式复用现有权重，但**改变了确定性均值剖面**。切线外推在非线性格子内有误差，
相邻段重新查询也可能产生小的均值不连续；不能宣称与 `cubic` 或 GP 参考结果等价。
平面均值的切线本身精确，但分段长度不同仍可能影响学习到的离散状态更新。
减小最大步长可控制局部线性化的几何误差，不保证神经预测误差单调减小。
这是均值场的线性化，网络输出的三次 Bernstein hazard 仍保持原定义。

示例配置和命令：

```powershell
build/Release/macrofacet_experiments.exe render --config configs/render_neural_renewal_shader_ball_ply_point_linear.json --spp 4
```

示例写入 `outputs/render_neural_renewal_shader_ball_point_linear`，原 PLY 配置仍使用默认 `cubic`。
`resolved_config.json` 会显式记录 `profile_mode`。原来的三个非神经输运模式不受该设置影响。

本机 PLY 场在 64×64、4 spp、h64/CUDA 的三轮交替测量中，方向环境中位时间由
8.598 s 降为 5.145 s（约 1.67 倍速度）。该低 spp 对照的线性 RGB RMSE 为 0.06239，
包含路径分歧的采样差异，尚不能解释为纯近似偏差或 GP 真值误差。
详见[点查询验证记录](../outputs/renewal_point_linear_validation/REPORT.md)。

### 模型导出

在仓库根目录执行：

```powershell
python python/renewal_cli.py export-hazard --checkpoint outputs/models/renewal_matern32_v1/best.pt --output outputs/models/renewal_matern32_v1/hazard_model.json
cmake --build build --config Release --parallel 4
build/Release/macrofacet_experiments.exe renewal-query --config configs/renewal_matern32_surface.json --model outputs/models/renewal_matern32_v1/hazard_model.json --trials 16384 --output outputs/renewal_model_surface
```

空域模式改用 `configs/renewal_matern32_exterior.json`。命令复用 `first-passage` 配置中的均值场、射线、起点条件、核尺度、剖面分段和最大长度；也支持完整 NanoVDB 配置。它只查询神经模型，不生成新的 GP 参考轨迹。`--trials` 控制模型距离采样的统计次数，`--bins` 控制默认查询点数量。

结果 `renewal_query.json` 包含每条射线的分段特征和四个 hazard 系数、累计量、透射率、物理 hazard、采样存活频率，以及给定光学深度的求逆结果。`resolved_renewal_config.json` 保存解析后的射线和剖面配置。模型 bundle 与结果都记录原 checkpoint 的 SHA-256。

可在输入 JSON 顶层添加以下可选项（距离为无量纲 `x=s/ell`）：

```json
"renewal_query": {
  "normalized_distances": [0, 0.1, 0.5, 1, 2, 4],
  "survival_intervals": [[0, 4], [0.5, 2], [1, 1]],
  "optical_depths": [0.01, 0.1, 0.5, 1, 3, 10]
}
```

所有端点必须在配置的 `max_time` 内。缺省时查询均匀网格和整个射线区间。零长度区间返回 `T=1`、未命中；未命中距离为查询终点。

## C++ 接口

主要接口位于 [RenewalHazardModel.h](../include/macrofacet/learned/RenewalHazardModel.h) 和 [RenewalRayDistribution.h](../include/macrofacet/transport/RenewalRayDistribution.h)。示例：

```cpp
#include "macrofacet/transport/RenewalRayDistribution.h"

auto model = mf::RenewalHazardModel::load("outputs/models/renewal_matern32_v1/hazard_model.json");
const double ell = 0.1;
auto mean = mf::RayMeanProfile::affine(0.5, -0.2, 4.0, 0.25);
mf::RayStartCondition start{mf::RayStartMode::PositiveExterior, 0.0};
mf::Random rng(1234);

// 同一条射线重复查询：缓存每段系数和累计量。
auto ray = mf::RenewalRayDistribution::fromModel(model, mean, start, ell);
auto sample = ray.sample(rng);
double T = ray.transmittance(0.3);          // 从出生点存活到物理距离 0.3
double conditionalT = ray.transmittance(0.1, 0.3); // 已存活到 0.1 后继续存活
auto continued = ray.sample(rng, 0.1, 0.3);

// 单次查询：顺序推理到命中／终点即停止，不缓存整条射线。
auto streamed = mf::sampleRenewalDistance(model, mean, start, ell, rng, 0.0, 0.3);
double streamedT = mf::renewalTransmittance(model, mean, start, ell, 0.0, 0.3);
```

`RayMeanProfile` 内的距离是 `x`、均值是 `b=m/sigma`；所有 `RenewalRayDistribution` 查询和 `sample.distance` 使用物理距离 `s`。采样容差默认 `1e-9`，也以物理距离计。`sample.segment` 是命中段索引，未命中时为总段数。`hazard(s)` 返回 `h_x/ell`，`cumulativeHazard(s)` 是无量纲光学深度。

真实几何使用 `RayMeanProfile::fromField(meanField, origin, unitDirection, sigma, ell, maximumDistance, maximumStep)`。模式 B 设置 `start={RayStartMode::SurfaceOutward, ell/sigma * direction.dot(fullGradient)}`，要求该导数严格为正。传入完整梯度，不能用单位法线代替它。模型初始化自行减去射线均值的起点单侧导数。

共享 `model` 是不可变、可跨线程使用的；每条射线单独维护 GRU 状态和随机数发生器。层尺寸从 bundle 的 `model_config` 读取，Eigen 和 LibTorch 后端都支持 64／128 维隐藏状态。当前 64 维架构的 `export-hazard` 包含 50,820 个参数，`export-model` 包含全部 87,324 个参数；原 128 维架构分别为 129,476 和 174,172。hazard bundle 可用于距离／透射率查询，但渲染会明确拒绝缺少 mixture 的模型。

## 完整梯度与法线

[RenewalMedium.h](../include/macrofacet/transport/RenewalMedium.h) 的 `beginFlight` 固定整条射线的均值分段，`sample` 返回物理距离、位置、无量纲穿越速度 `W>0`、完整梯度和单位法线。不给起点梯度时为 A，传入完整梯度时为 B。命中后才查询 mixture 一次，并使用进入该段的 GRU 状态。

每个 mixture 分量分别截断到正半轴；先按权重选择分量，再采样截断 Gaussian。负均值尾部使用指数拒绝采样，直接生成截断点之上的增量，避免大数相减或近零 CDF 反演的问题。物理沿射线导数为 `V=-(sigma/ell)*W`。

各向同性 A 模式的两个横向残差独立，标准差为 `sigma/ell`。B 模式保留出生点横向梯度残差，均值衰减 `exp(-x)`，方差比例为 `1-exp(-2x)`（以 `expm1` 稳定计算）。这实现完整 Renewal+，没有丢弃起点横向相关。实际均值梯度取与命中段一致的三线性插值单侧导数；不会先归一化均值梯度或保存的碰撞梯度。

## 累计量与历史语义

四个非负系数定义段内三次 Bernstein hazard，解析积分得到四次累计多项式。距离采样只抽一次 `E=-log(U)`，逐段减去累计增量，在命中段用有界二分求逆；达到终点仍有剩余光学深度时返回未命中。透射率使用同一累计量 `T=exp(-H)`。不添加正 hazard 下限。

沿直线跨体素时保持 GRU 历史。查询终点落在段内时只积分该段的相应部分，保留原始整段网络输入。**对同一条射线做不同长度的查询时，必须复用相同的原始均值剖面和分段，不能先裁短剖面再重新输入网络。** 神经模型本身不保证任意重新分段后的结果完全相同；应沿用训练数据的体素边界和 `profile.maximum_step`（首轮训练为 `0.25`）。

`transmittance(from,to)` 和区间采样表示已从原出生点存活到 `from`，不是在 `from` 重新执行 A/B 初始化。实现直接积分区间 hazard，避免用两个可能已下溢为零的透射率相除，也避免相减很大的累计前缀。段内区间积分采用对三次多项式精确的两点 Gauss 公式；几何、累计、求逆为 double，网络权重和激活为 float32。

## 验证与边界

```powershell
python python/renewal_cli.py verify-cpp --checkpoint outputs/models/renewal_matern32_v1/best.pt --executable build/Release/macrofacet_experiments.exe --config configs/renewal_matern32_surface.json --output outputs/renewal_cpp_validation/surface
ctest --test-dir build -C Release --output-on-failure
```

`verify-cpp` 独立运行 PyTorch，逐段比较系数，并检查累计量、透射率、部分区间求逆、mixture 参数以及 Monte Carlo 存活率和速度 CDF，写出 `parity_report.json`。完整模型的 `renewal-query` 结果包含查询点的 mixture；顶层 `renewal_query.speed_sample_count` 可启用速度采样频率诊断。CMake 设置 `MACROFACET_TEST_NEURAL=ON` 后，CTest 还运行随机权重网络的跨语言 A/B 对照。纯 C++ 测试不调用 Python 推理；启用 LibTorch 构建时额外检查可用 CPU/CUDA 后端，关闭时检查标量后端和回退行为。

2026-10-08 使用已训练的 `best.pt` 检查两个示例和未见 shader ball NanoVDB 的 16 条射线，共 18 条。最大累计量绝对差 `2.06e-7`、透射率绝对差 `8.72e-8`、逆解光学深度残差 `1.75e-7`、mixture 参数绝对差 `4.77e-7`；每条 16,384 次距离采样与每个查询点 4,096 次速度采样均通过统计检查。这些数值衡量 C++ 与 PyTorch 的实现一致性，不是模型相对于 GP 的误差。

完整 shader ball 示例已以 `64x64, 32 spp, 4 threads` 渲染：两个环境各 131,072 条相机路径，均无数值失败、无深度截断，白炉平均 RGB 为 `0.999981`（开启俄罗斯轮盘）。导入完整 NanoVDB 的渲染及原三个模式的 smoke render 也已通过。三组 CTest 全部通过，其中 C++ 为 111,273 项检查、Python Renewal 为 14 项测试。结果汇总见 [validation_summary.json](../outputs/renewal_normal_validation/validation_summary.json)。

测试还覆盖正截断 Gaussian 的强负均值尾部、A/B 梯度均值和协方差、各向异性、单侧体素梯度、多次反射、白炉能量及线程确定性。神经渲染使用模型分布和 Renewal+ 记忆截断，不等同于全局 GP 完整历史参考；真实图像误差还需用相同条件的外部 GPIS 实现评估。首轮网络训练长度为 `x=2/4/8`，更长的实际路径剖面仍属于需要额外验证的泛化范围。

## 标量后端性能诊断与 SIMD 优化

以下为 Eigen 标量后端的诊断及历史优化记录。可单独构建诊断工具，分解 CPU 路径循环中的耗时，生产渲染循环不添加计时器；该工具固定使用标量推理，与配置中的批处理后端选择无关：

```powershell
cmake --build build --config Release --target macrofacet_renewal_benchmark --parallel 4
build/Release/macrofacet_renewal_benchmark.exe configs/render_neural_renewal_shader_ball.json 2 outputs/renewal_performance/profile.json
```

第二个参数是诊断 spp。工具使用单线程，跟随配置指定的环境对应的 Fresnel 和俄罗斯轮盘，排除模型加载、场烘焙及图像输出；逐段计时本身存在开销。`constructed_segments` 是预先构建的完整均值剖面段数，`evaluated_segments` 是实际推理到碰撞或离开域为止的段数。

2026-10-08，i9-13900HX、MSVC Release、Eigen SSE/SSE2，原始标量激活版本的 shader ball 单线程诊断（64×64、2 spp）：

| 部分 | 耗时占比 |
| --- | ---: |
| 段编码器、hazard MLP、GRU | 90.68% |
| 完整均值剖面构建 | 4.87% |
| 初始状态网络 | 1.31% |
| 速度 mixture 推理和采样 | 0.94% |
| hazard 积分 | 0.55% |
| 累计量求逆 | 0.10% |
| 完整梯度和法线重建 | 0.04% |
| 其他及诊断开销 | 1.51% |

此前 64×64、32 spp 的方向环境渲染有 7,478,984 次段推理、96,685 次 mixture 查询，平均每条相机路径约 57 次段推理。默认网络每段仅线性层就需要 94,784 次乘加，其中 GRU 为 73,728 次；整张图约 7090 亿次乘加，尚未计入激活、初始化和 mixture。段推理的总工作量和小矩阵执行效率是首要瓶颈；单独优化法线采样收益有限。

`RenewalHazardModel.cpp` 已将 SiLU 和 GRU 的 sigmoid/tanh 改为 Eigen 向量运算，保留稳定 sigmoid、PyTorch reset-after 约定及现有权重。首次诊断总耗时从 4.990 s 降到 4.209 s，段推理从 4.524 s 降到 3.770 s。浮点舍入会变化，不保证与旧版本逐路径完全相同。

完整渲染以 64×64、8 spp、4 线程重复三轮、交替执行优化前后程序；为减少混合核心调度差异，两个测试进程均固定到逻辑 CPU 0/2/4/6（affinity `0x55`）。白炉耗时中位数 7.053 → 6.174 s，方向环境 7.133 → 6.628 s，分别减少约 12.5% 和 7.1%。普通调度下出现约两倍的计时波动，固定 CPU 后仍有波动，因此这些数字仅描述本次测量，不能当作所有场景的稳定加速保证。完整记录包含未固定 CPU 的结果：[原始记录](../outputs/renewal_performance/comparison.json)、[固定 CPU 的对照](../outputs/renewal_performance/affinity_55/comparison.json)。

三组 CTest 和 18 条已训练模型的 A/B、NanoVDB 射线对照全部通过；最大 hazard 系数差 `1.44e-6`、透射率差 `6.70e-8`、mixture 参数差 `4.77e-7`。上述同种子渲染白炉 PFM 相同，方向环境线性 RGB RMSE 为 `2.17e-6`。这些是实现一致性检查，不是 GP 参考误差。报告：[parity/summary.json](../outputs/renewal_performance/parity/summary.json)。

另以 32×32、8 spp 白炉检查线程扩展，每种设置运行两次、不固定 CPU：1/4/8/自动线程的耗时中位数分别为 8.20/1.70/0.89/0.91 s。这表明增加射线工作线程在该测试中有效，但小图与混合核心调度的结果不能线性外推。当前示例的 `thread_count=0` 已使用自动线程数；本机为 32 个逻辑 CPU。此前与 classic_local 的对照为固定 4 线程。记录：[threads/summary.json](../outputs/renewal_performance/threads/summary.json)。

LibTorch 批处理后端现已实现，见下节。每条射线自身仍须按顺序推进 GRU。更小的 GRU 或蒸馏需要重新训练和验证。惰性构建均值剖面可以减少未使用段的工作，但按上述标量占比，单独优化该部分的收益有限。直接增大 `profile_maximum_step`、跳过远处段或调粗体素会改变网络输入、历史或均值场，不属于已经验证的等价加速。

## LibTorch CPU/CUDA 多射线推理

构建默认开启 `MACROFACET_ENABLE_TORCH=ON`，优先复用已有 `Torch_DIR`，否则从 CMake 选择的 Python 安装发现 LibTorch。不需要重新训练或改用另一种 checkpoint；完整 JSON 权重先经现有加载器验证，再上传至设备。LibTorch 代码独立编译，当前 2.12 安装所需的 C++20 不扩散到 C++17 渲染器。Windows 首次构建会把该安装的运行时 DLL 部署到可执行文件目录；运行 `.exe` 不需要启动 Python。需要完全去掉 LibTorch 依赖时配置 `-DMACROFACET_ENABLE_TORCH=OFF`。

| `transport.renewal.backend` | 行为 |
| --- | --- |
| `scalar` | 原 Eigen 逐射线 CPU 路径追踪；未指定 backend 的旧配置继续使用它 |
| `torch_cpu` | LibTorch CPU 矩阵批量推理 |
| `torch_cuda` | LibTorch CUDA 批量推理；设备或库不可用时明确报错 |
| `auto` | 优先 CUDA，其次 LibTorch CPU；未编译 LibTorch 时使用 scalar |

shader ball 示例配置选择 `auto`、`batch_size=4096`。实际选择记录于 `resolved_config.json` 的 `resolved_backend` 和 `render_summary.csv` 的 `neural_backend`。新增 `neural_initialization_batches`、`neural_segment_batches`、`neural_mixture_batches`、`neural_maximum_batch_size`；`neural_segments/neural_segment_batches` 可计算平均有效批量。

`RenewalWavefront.cpp` 用有限数量的槽位调度多条射线：初始化新飞行，批量推理每条活动射线的下一段，在 CPU 上累计 hazard／求逆，仅对命中的射线批量查询 mixture，再执行梯度重建、反射和轮盘。完成一条样本后继续该像素的下一条样本；像素结束后复用槽位。每像素保留原来的随机数流、正态分布缓存和累加顺序，不根据批大小重新播种。A/B 起点、分段、核定义及碰撞采样分布均保持原定义。

LibTorch 后端在所选设备上执行输入 `asinh` 和输出 `softplus`：初始化转换四个输入，段编码只转换前四列（第五列 `log(dx)` 保持原值），mixture 查询只转换均值与导数（`u` 保持原值）。CPU 打包原始 double，设备先执行 double `asinh` 再转 float32，避免提前转 float 使大幅值输入溢出。hazard 系数和 mixture scale 在回传前执行 `softplus(beta=1, threshold=20)`；CPU 继续执行 mixture 权重归一化、均值导数修正和 scale floor。原始 scale logits 随已有结果一起回传，在 CPU 上检查有限性，避免为小批次增加归约 kernel 或单独同步。

`torch_cuda` 在 GPU 上执行上述变换，`torch_cpu` 使用同一套 ATen 逻辑在 CPU 上执行；`scalar` 参考实现保持原样。无需重训或重新导出模型。不同设备的数学函数允许小幅浮点差异，不保证逐位相同的路径结果。

`RenewalBatchSession` 管理独立槽位状态；初始化使旧 mixture 缓存失效。`evaluate` 只推进列出的槽位一次，并缓存其**进入该段前**的状态和段编码，`mixture` 查询使用该缓存。网络权重、GRU 状态和进入段上下文常驻 GPU；CPU 每轮只传特征／槽位索引，下载少量 hazard 系数及实际命中的 mixture 参数。推理关闭 autograd，并在作用域内禁止 TF32，使用 float32；几何和累计量保持 double。不同 GEMM 实现仍可能带来浮点差异，因此不保证不同后端或批大小逐像素完全一致。

批处理渲染使用一个 CPU 调度线程及 LibTorch 设备执行；`render.thread_count` 控制原标量后端，不表示 GPU 推理线程数。场剖面、采样求逆、法线重建仍在 CPU 上。`traceCameraPath` 单射线接口和 `renewal-query` 保留 Eigen 实现。小图、小批量可能受 GPU 启动和同步开销影响；完整渲染计时包含推理会话建立和首次 CUDA 使用，且命令会依次渲染白炉和方向环境两张图。

验证工具可重放已经通过 Python 对照的 `renewal-query.json`，同时推进其中不同长度的射线，并检查 hazard、累计量、透射率和 mixture：

```powershell
cmake --build build --config Release --target macrofacet_verify_renewal_batch --parallel 4
build/Release/macrofacet_verify_renewal_batch.exe outputs/models/renewal_matern32_v1/renewal_model.json outputs/renewal_performance/parity/nanovdb_surface/cpp/renewal_query.json torch_cuda outputs/renewal_torch_validation/parity/replay.json
```

已用现有 checkpoint 检查 A/B 及 NanoVDB 共 18 条射线、335 个段：LibTorch CUDA 对 Eigen 的最大 hazard 差 `1.44e-6`、累计量差 `1.68e-7`、透射率差 `9.04e-8`、mixture 参数差 `7.16e-7`；CPU 后端也通过。详见 [parity/summary.json](../outputs/renewal_torch_validation/parity/summary.json)。这些衡量实现一致性，不是相对于 GP 的误差。CTest 另覆盖稀疏／重排批次、槽位复用、未初始化／重复槽位拒绝、命中前状态、不同批大小的图像、进度计数及白炉能量。

2026-10-08 在 RTX 4060 Laptop GPU 上以相同场、相机、种子和模型进行 64×64、8 spp 测量，交替顺序执行三轮，中位数如下。未固定 CPU，计时包含推理会话建立；首次白炉还承担 CUDA 上下文启动成本。

| 后端 | 白炉 | 方向环境 |
| --- | ---: | ---: |
| Eigen，4 线程 | 6.853 s | 7.662 s |
| Eigen，自动线程（本机 32 逻辑 CPU） | 4.363 s | 4.182 s |
| LibTorch CUDA，最多 4096 槽位 | 2.725 s | 2.365 s |

方向环境相对自动 CPU 快约 1.77 倍，相对 4 线程快约 3.24 倍。对应白炉 PFM 相同，方向环境线性 RGB RMSE `5.97e-6`，各后端数值失败和深度截断均为零。微小舍入差异可以改变个别 Monte Carlo 路径，图像差不能直接解释为模型偏差。完整参数、逐轮计时和场文件哈希见 [benchmark.json](../outputs/renewal_torch_validation/benchmark.json)。

默认 64×64、32 spp 配置也已实际完成 CUDA 渲染，两种环境各 131,072 条路径，均无数值失败和深度截断，白炉平均 RGB `0.999981`；结果位于 [full_default](../outputs/renewal_torch_validation/full_default/)。三组 CTest 通过，补充轮盘及深度上限检查后的 C++ 测试共 113,223 项通过。设置 `CUDA_VISIBLE_DEVICES=-1` 时，`auto` 正确回退至 `torch_cpu`，显式 `torch_cuda` 正确报错。

另在 `build/no_torch` 中以 `MACROFACET_ENABLE_TORCH=OFF`、`MACROFACET_BUILD_FIELDS=OFF` 完成独立构建和两组 CTest，C++ 共 108,849 项通过，确认核心渲染库不强制依赖 LibTorch。

默认 32 spp 负载另做了一次无并行构建的顺序复测：方向环境自动 CPU 为 14.718 s，CUDA 为 9.496 s（约 1.55 倍）；白炉分别为 16.587 s 和 10.307 s。这是单次复测，三轮中位数证据仍以上面的 8 spp 测量为准。完整记录见 [default_timing/summary.json](../outputs/renewal_torch_validation/default_timing/summary.json)，全部验证汇总见 [validation_summary.json](../outputs/renewal_torch_validation/validation_summary.json)。

# Renewal+ C++ 采样、法线与路径追踪

实现方案中的碰撞距离、穿越速度、完整梯度／法线采样、模型透射率和多次散射路径追踪。C++ 保留 Eigen 标量推理，并支持 LibTorch CPU/CUDA 多射线批量推理；两者加载相同 JSON 权重。支持 Matérn-3/2 的 `rho(x)=(1+x)exp(-x)`（`unit_decay`）和 SE 的 `rho(x)=exp(-x*x/2)`（`unit_length`），均为 `beta=1`；支持空域起点 A 和已知表面向外起点 B。模型导出的核与场核必须匹配。

## 单条真实相机光线的透射率对照

`macrofacet_inspect_renewal_ray` 从渲染配置重放一个像素的第 0 个相机采样，复用相机 RNG、有效域入口、`profile_mode`、原始分段和所选推理后端。当前支持完整域 NanoVDB 场景的第一段相机飞行，起点条件为 A：`F(0)>0`。它输出神经透射率、光线元数据及两份参考配置；参考计算复用已有 `first-passage` 命令，没有新增参考渲染器。

在仓库根目录运行，最后两个数字是从左上角以 0 开始的像素坐标；省略时取中央像素：

```powershell
cmake --build build --config Release --target macrofacet_inspect_renewal_ray --parallel 4
build/Release/macrofacet_inspect_renewal_ray.exe configs/render_neural_renewal_shader_ball_ply_point_linear.json outputs/renewal_camera_ray_comparison 256 256
build/Release/macrofacet_experiments.exe first-passage --config outputs/renewal_camera_ray_comparison/reference_coarse.json
build/Release/macrofacet_experiments.exe first-passage --config outputs/renewal_camera_ray_comparison/reference_fine.json
python data_analysis/plot_renewal_camera_ray.py outputs/renewal_camera_ray_comparison
```

打开输出目录的 `comparison.html`，可切换全程／下降区域并悬停读取数值。另有可单独使用的 SVG、逐点 `comparison.csv`、指标 `comparison_report.json` 和 `ray.json`。绘图复用仓库 SVG 工具，无第三方绘图库依赖；可用 `--preview-pfm <已有图像> --preview-config <对应配置>` 添加带像素标记的场景预览，此可选功能需要 NumPy，并检查相机、场、模型等配置是否匹配。

参考采用同一 VDB 均值、σ、方向相关长度和 unit-decay Matérn-3/2 核，以状态空间／桥接采样估计首达存活概率。默认每组 65,536 条 GP 轨迹，归一化粗步长不超过 `1/32`，细组将步长和最小细分步长都减半，使用独立种子。蓝色带是逐点 95% Wilson 区间，仅衡量 Monte Carlo 误差；步长复查另行报告，不构成严格无偏或收敛证明。参考沿射线的逐体素三次均值还会与直接 VDB 查询逐点核对。

神经曲线使用原始完整分段，查询距离只决定 hazard 积分的上限，不会截短剖面后重新输入网络。`point_linear` 对照包含均值近似与网络误差，不能单独归因于 GRU。CUDA 以 batch=1 重放，可能与整图批处理有微小浮点差异；工具另与 Eigen 标量结果交叉检查。曲线表示统计透射率，不是某一次 GP 实现的 0/1 可见性，也不包含后续反弹。

2026-10-09 的 [shader ball 中央像素示例](../outputs/renewal_camera_ray_comparison_20261009/comparison.html)：h64、σ=0.02，最大透射率差约 `0.1501`；该点网络 `0.7599`，细参考 `0.9100`，95% 区间约 `[0.9078, 0.9122]`。两组参考最大差 `0.00729`，CUDA／标量最大差 `4.66e-8`。这只说明该相机光线上的偏差，不能代表整图误差。

将诊断配置的 `transport.renewal.profile_mode` 改为 `cubic` 后，也可输出精确到浮点误差的 VDB 射线均值剖面。绘图支持 `--reference-directory <已有参考目录>` 复用同一条光线的 GP 参考，以及 `--compare-neural <另一模式的诊断目录>` 叠加其神经曲线；会检查场、光线、模型和查询网格的一致性。例如：

```powershell
python data_analysis/plot_renewal_camera_ray.py outputs/renewal_camera_ray_cubic_comparison_20261009 --reference-directory outputs/renewal_camera_ray_comparison_20261009 --compare-neural outputs/renewal_camera_ray_comparison_20261009
```

[同光线 cubic／point_linear 对照](../outputs/renewal_camera_ray_cubic_comparison_20261009/comparison.html) 的最大透射率误差分别为 `0.15068`／`0.15012`，两种神经预测之间最大差 `0.00333`；cubic 均值与 VDB 直接查询的最大差约 `8.6e-16`。因此这条光线的主要透射率偏差不能用局部线性近似解释；它也没有证明偏差由 GRU 架构本身造成。原渲染配置保持不变，cubic 诊断配置保存在新输出目录中。

同一条光线也可比较不同权重：`--compare-model <另一模型的诊断目录>` 要求场、起点、后端、`profile_mode` 和每段输入完全一致，图例会标记隐藏维度，报告记录两个模型的 checkpoint 哈希。例如：

```powershell
python data_analysis/plot_renewal_camera_ray.py outputs/renewal_camera_ray_h16_comparison_20261009 --reference-directory outputs/renewal_camera_ray_comparison_20261009 --compare-model outputs/renewal_camera_ray_comparison_20261009
```

2026-10-09 的 [h16／h64 对照](../outputs/renewal_camera_ray_h16_comparison_20261009/comparison.html) 保持 `point_linear`、像素 `(256,256)` 的第 0 次采样和既有 GP 参考不变。h16 最大透射率差为 `0.07216`，h64 为 `0.15012`；`T=0.5` 的入口相对距离分别为 `0.30952`／`0.27609`，GP 参考为 `0.31183`。h16 CUDA／标量最大差为 `4.23e-8`。两模型来自同一份 v2 数据和训练种子，此单射线结果不能作为隐藏维度或架构优劣的普遍结论。

同场景的 [16 个固定像素覆盖检查](../outputs/renewal_camera_rays_h16_h64_20261009/comparison.html) 增加了 15 个位置并复用中央光线，全部保持 `point_linear`。15 条光线进入有效域，另 1 条为域外真空。每条场内光线均有 65,536 条粗参考和 65,536 条细参考轨迹；13 条存在明显模型差异的光线上 h16 均优于 h64，另 2 条背景光线的两模型都接近 `T=1`。逐光线最大绝对透射率误差的均值为 h16 `0.07077`／h64 `0.14216`，最坏为 `0.12451`／`0.22197`，均出现在像素 `(256,440)`。这不是整图平均误差。

总览可点击像素编号查看单光线曲线和置信区间；`selection.json` 保存实验前的像素列表、配置及文件哈希，`summary.csv/json` 保存完整指标。脚本 `outputs/renewal_camera_rays_h16_h64_20261009/run_suite.py` 可从仓库根目录重跑／恢复，完成后运行同目录 `plot_suite.py` 生成总览。两种参考步长的最大差为 `0.00729`；CPU／CUDA 最大透射率差为 `1.22e-7`。报告另用跨光线同时 DKW 界区分 Monte Carlo 噪声，这不覆盖剩余参考离散误差。固定像素覆盖和同一场景的结果不能证明架构优劣或代表整图泛化。

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

使用与模型对应的 `field.kernel_type="matern_3_2"` 或 `"squared_exponential"`，以及 `material.ndf_family="generalized_gaussian"`。指定三个相等的 `correlation_lengths`，或使用 `material.roughness` 派生相关长度。固定正定椭圆各向异性也可使用：每个方向取 `ell=1/sqrt(direction^T M direction)`，梯度按条件协方差重建。空间变化的核或 alpha 网格不属于这个平稳模型；导入 NanoVDB 时设置 `use_alpha_grid=false`。

SE h16 模型的独立渲染配置为 `configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16.json`，使用同一 shader ball 场、相机和批处理设置。运行：

```powershell
build/Release/macrofacet_experiments.exe render --config configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16.json
```

模型加载严格检查核类型、parameterization 和 beta；渲染与 `renewal-query` 都会拒绝核不匹配。
`resolved_config.json` 记录场的 `kernel_parameterization` 和模型的 `model_kernel_type`。
相机光线检查工具生成对应核的 GP 参考配置，SE 参考使用条件循环嵌入网格。

2026-10-10 的 SE 适配验证：Release 构建和三组 CTest 全部通过，包含各向同性／旋转各向异性横向梯度的条件均值、协方差和极短距离检查。已训练 h16 模型的 12 条 A/B 光线通过 Python、Eigen、LibTorch CPU/CUDA 对照；最大透射率差 `2.66e-7`、mixture 参数差 `1.44e-6`。实际 shader ball 在三个 C++ 后端各完成 `32×32、2 spp` 渲染，数值失败和深度截断均为零；双向核不匹配均在输出前报错。相机射线 SE 参考生成与绘图链路也已通过。这些检查验证实现一致性与集成，不代表完整分辨率性能或相对 GP 的模型精度；参考图仅使用 128 条轨迹。报告见 [summary.json](../outputs/renewal_se_render_validation_20261010/summary.json)，回归日志见 [ctest.log](../outputs/renewal_se_render_validation_20261010/ctest.log)。

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

## 大光线池与统一提交

LibTorch 波前渲染把常驻光线槽数和每次推理容量分开。当前 `point_linear`
shader ball 配置沿用 h16 权重，使用：

```json
"batch_size": 8192,
"ray_pool_size": 32768,
"auxiliary_batch_minimum": 256,
"maximum_queue_delay": 4,
"max_in_flight_batches": 2
```

`ray_pool_size` 是同时保留路径、GRU 状态和命中段上下文的槽数，实际不超过像素数。
省略或设为 0 时取 `min(65536, 2*batch_size)`。每个像素仍按原顺序完成所有 spp，
保持自己的 RNG 流和累加顺序。`batch_size` 限制本次三类请求合计的不同光线数；
初始化和同次首段推理算同一条光线。可提交的 mixture 优先占用容量，其次是初始化，
最后是继续推进的段，各组内部保持 FIFO。这样不会在一整批段请求之外额外取走碰撞光线，
导致两倍批容量的光线池没有足够的另一组供 CPU/GPU 重叠。
增大池会增加 CPU 剖面和 GPU 状态内存；`cubic` 的整条剖面尤其需要留意内存。

调度器保留初始化、段推进、mixture 三条 FIFO 队列。初始化和 mixture 累积到
`auxiliary_batch_minimum`（实际夹到本次容量）后提交；未达到阈值时，最老请求等待
`maximum_queue_delay` 轮已完成提交后也可参与合批。该参数不是毫秒，也不是有积压时
每条请求的绝对等待上界；超过单批容量的积压按 FIFO 分批处理。立即刷新尾批的条件是
CPU 任务和 GPU 在途批次都已排空；否则继续积累工作线程发布的就绪光线，并按阈值或
等待轮数合批。这包括只有 mixture 待处理的最后一批，避免等待凑批造成停滞。设为 0
可禁用主动延迟。除启动和收尾等没有其他就绪工作的情况，辅助请求随段推理一起提交。

命中后的槽位冻结在命中段，直到 mixture 返回。其缓存是进入该段时的 GRU 状态与
段编码，等待期间不能继续推进、初始化或复用槽位。CPU 完成法线采样、反射和新飞行
构造之后，才把同一条路径重新加入初始化队列；不同路径的三类请求可以同批提交。

`RenewalBatchSession::submit(RenewalBatchRequests)` 先验证整批请求，再进行一次打包上传，
在 GPU 上依次处理非空的三个组，最后一次回读 hazard、mixture 和状态有效性标志。
slot ID 与原始 double 特征一起上传，asinh/softplus 仍在设备侧执行，隐藏状态留在设备上。
允许同槽“初始化＋首段”；mixture 槽必须与另外两组互斥。原 `initialize/evaluate/mixture`
接口仍可使用。各网络头仍有多个 CUDA kernel，尚未使用 CUDA Graph。

异步入口为 `submitAsync(requests)`，返回会话内的 ticket；`isReady(ticket)` 只查询完成
事件，`tryCollect(ticket)` 在未完成时返回空 optional，`collect(ticket)` 等待该批后取回。
原 `submit()` 等价于提交后立即 collect，保留同步调用方式。会话只由一个线程调用，
每个 ticket 必须且只能回收一次。请求数组在提交返回后即可修改或释放。所有涉及的槽位
在回收前保持锁定，即使设备已完成，也不能被新的初始化、段推进或 mixture 请求使用。
整批参数在提交前检查；回收时检查状态有效性，数值失败会释放 ticket 并使写入过的槽位
失效，需要重新初始化。销毁会话时会排空仍在执行的设备工作，再释放缓冲区。

每个在途批次拥有独立的 pinned 输入／输出、设备缓冲区和 CUDA 完成事件。一个专用
CUDA 计算流依次执行各批，在非阻塞 D2H 后记录事件。调用线程负责三类队列合批和派发 CPU
任务，独立推理线程专门创建／使用／销毁 LibTorch 会话，提交请求、查询事件和回收结果。
即使推理线程正在执行 ATen 调用，协调线程仍能把之前已完成的结果派发给 CPU。
CPU 工作线程持续做相机光线／新飞行初始化、碰撞判定、下一段场查询、
法线采样、反射和像素累加。每个任务最多处理 128 条光线，完成后立即发布自己的就绪列表，
不等待同一批的其他 CPU 任务。命中者进入 mixture 队列，未命中者进入下一段队列，
新飞行进入初始化队列。三类就绪请求继续统一提交和回读。

GPU 结果按提交顺序回收，但 CPU 任务可独立完成。每个槽位只能由一个任务或请求拥有；
任务通过短临界区发布结果，场查询和网络推理都不持有该锁。回读结果与请求数组由 CPU
任务共同持有，直到最后一个使用者完成，GPU 传输缓冲区则可提前用于下一批。
协调线程等待 CPU 或推理线程的完成通知。推理线程提交一批后，先等待该批的 CUDA 完成事件，
回收并发布结果，再进入下一次 LibTorch 提交。这样完成结果不会被下一次较长的 ATen 主机
调用延迟发布，CPU 能在它提交下一批时处理上一批。无请求时等待队列通知，避免短时定时
轮询在 Windows 上被系统定时粒度放大；事件等待不阻塞协调线程和 CPU 光线任务。
因此当前波前策略实际只保留一个活跃 GPU ticket，其他光线组在 CPU 处理或排队；底层会话
仍支持多个独立 GPU 批次。`max_in_flight_batches` 取 1..4，默认 2；窗口涵盖待提交、
设备执行中和完成后尚未派发的推理批次，防止生产者无限堆积请求。设为 1 只限制该窗口，
CPU 工作线程仍独立运行。光线池不足或进入尾批时会自然减少在途数量。
当前示例使用四批容量的池，给辅助队列未凑齐以及 CPU/GPU 处理中的光线预留空间。
只要还有 CPU 任务或 GPU 在途批次，就等待三类可提交请求合计达到一批不同光线，
避免刚发布一小块 CPU 结果就发起小 GPU 请求。辅助队列超龄会使该组参与合批，
不会绕过总体批量门槛；CPU 和 GPU 都已排空时刷新不足一批的尾部，保证进展。
LibTorch CPU 后端也在专用推理线程中计算，使用单批窗口，仍可与其他光线的 CPU 任务重叠；scalar 模式保持原路径。

渲染 CSV 和 profiling JSON 记录 submissions/readbacks、合并提交数、实际池容量、
辅助批次最大行数及最大等待轮数。initialization/segment/mixture batches 现在表示非空
请求组数；其总和可以大于统一提交次数。CPU 后端的 readbacks 表示逻辑返回次数，
不代表设备传输。`render.thread_count` 现在表示 CPU 光线工作线程数量，不包含协调线程和推理线程；
0 自动选择至多 8 个，并受槽位容量限制。每个像素的所有 spp 固定在一个槽位顺序完成，
因此不会并发写同一像素，统计按槽位累积后合并。异常会先停止新工作并排空 CPU 任务，
再由会话析构排空设备传输。任务完成顺序会改变批成员，GPU 浮点舍入可能改变个别路径，
即使相同配置重复运行，也不承诺训练模型的 PFM 逐字节一致。
另记录实际 GPU 最大在途批次、回收前尚未完成的批次数，以及回收结果时其他 GPU 批次尚未完成的
次数。blocking_collects 表示回收前事件尚未就绪，不表示 CPU 光线任务停止；主机 ATen 提交
本身可能比设备执行长。本策略回收优先，实际 GPU 在途峰值为 1，回收当时没有另一张 GPU
ticket，故 cpu_batches_with_gpu_pending 可以为 0；下一批 GPU 与上一批 CPU 的真实重叠
应查看 Nsight 时间线，不能用该旧计数判断异步是否生效。不能把
异步 enqueue 的主机耗时当作 GPU 执行耗时。

2026-10-10 独立 CPU 任务队列与专用推理线程的最终验证：同一 h16、`point_linear`、
512×512、1 spp，前后都用 8192 批／32768 池、8 个配置光线线程，亲和性 `0xffff`。
按旧／新／新／旧顺序各测两帧，每个进程先预热一帧；加载与预热不计时。旧版为
21.72／25.80 s，新版为 15.14／14.58 s，中位数 23.76→14.86 s，本轮耗时减少约 37%。
只有两帧／版本，旧版波动明显，未测完整 256 spp，不能承诺固定加速比。

独立 Nsight 捕获确认，CPU 初始化／推进／散射都在 8 个工作线程上运行。CPU 光线工作
与 LibTorch 主机提交重叠 6.797 s，与实际 GPU kernel 重叠 0.586 s，占 GPU kernel
执行区间并集 2.291 s 的 25.6%；这是区间交集，不是 SM 占用率，也不是线程时间之和。
当前实际 GPU ticket 峰值为 1，但下一批 GPU 与上一批 CPU 已并行。C++ 127427 项检查
通过，覆盖工作线程异常、后台推理失败、非恒定递归权重、槽位复用、尾批与 CPU/CUDA。
新版两帧及时间线均无数值错误／深度截断；旧版两帧各有 1 条路径触及原有 128 深度上限，
报告保留该记录，没有调整深度上限。原始计时、配置、图像差异与时间线见
[工作队列流水线验证](../outputs/renewal_worker_pipeline_20261010/summary_pipeline.json)。

以下为改成独立 CPU 任务队列之前的历史测量。2026-10-10 异步版本在 RTX 4060 Laptop GPU 上完成验证，仍使用 h16、sigma 0.02、
`point_linear`、8192 批／16384 池；固定 CPU 亲和性 `0xffff`。两组只改变
`max_in_flight_batches=1/2`。均先预热，计时包含会话创建，排除场／模型加载：

| 测量 | 单批（1） | 双批（2） | 观察 |
| --- | ---: | ---: | --- |
| 128×128、4 spp，三轮交替中位数 | 11.00 s | 12.73 s | 双批慢约 16% |
| 512×512、1 spp，单次对照 | 37.35 s | 29.06 s | 双批耗时少约 22% |

大图回收前事件未完成的次数由 6521 降至 185，但这只是一轮大图计时，不能作为稳定
加速倍数或高 spp 耗时预测。小图并未提速；异步也不会消除大量小 CUDA kernel 的
主机提交开销。就绪队列大小、辅助请求合批和尾批比例都会影响结果。

独立 Nsight 捕获使用 64×64、1 spp、1024 批／4096 池。将 GPU kernel 的实际执行
区间与 CPU 推进工作区间取交集、合并重叠区间后，单批为 0，双批为 **2.633 ms**；
包含协调和线程池屏障的 `cpu.advance_batch` 区间则为 3.262 ms。这证明发生了实际
执行重叠，但该小图的重叠量有限，且带插桩的时间线不用于比较整帧性能。
单批／双批分别有 2021／2210 次提交及同样次数的真实 D2H；H2D 各多 26 次模型上传。

三组 CTest 全通过，C++ 127032 项检查覆盖 CPU/CUDA、乱序回收、独立缓冲区复用、
在途槽位冲突、数值异常释放、带未完成批次的析构、1/2/4 批调度以及 CPU 工作线程数。
已训练 h16 的 12 条固定 A/B 光线在 CPU/CUDA 上均通过，最大透射率差 `2.36e-7`。
正式渲染与诊断程序的同配置 PFM 逐字节一致；同一批次深度重复运行的 PFM 也一致。
切换深度改变批形状，浮点舍入可导致 Monte Carlo 路径分歧，不能承诺两种深度逐像素
一致，也不能把图像差当作 GP 真值误差。原始记录、配置、复现脚本和时间线见
[异步验证汇总](../outputs/renewal_async_20261010/summary.json)。

同日补充的瓶颈分析保持当时 512×512、h16、8192 批／16384 池、双批配置，仅将 spp
改为 1。两次带计时运行分别为 25.84／25.38 s，主机互不重叠的阶段平均如下：

| 阶段 | 平均耗时 | 帧时间占比 |
| --- | ---: | ---: |
| CPU 打包与 LibTorch 提交 | 11.70 s | 45.7% |
| CPU 查询、推进、整理与散射 | 7.15 s | 27.9% |
| 就绪队列组批 | 3.72 s | 14.5% |
| 取回、校验和保存结果 | 2.07 s | 8.1% |
| 新飞行初始化 | 0.83 s | 3.2% |

提交阶段中，CPU 请求打包／验证约 3.12 s，后端 ATen 调用、张量管理和 CUDA 提交约
8.57 s；后者包含多种操作，不能全部解释成 kernel launch API 的耗时。结果回收中的
事件等待只有约 0.157 s。CPU 推进包含线程池派发、计算、屏障和后续整理；采样估计
场查询累计约 23.4 个线程秒，这不是可直接叠加到帧时间上的墙钟耗时。

相同配置的独立 Nsight 捕获用时 31.43 s，GPU kernel 实际执行区间合计 2.497 s
（帧区间的 7.94%，不是 SM 占用率）。共 334855 次 kernel，平均每次 7.46 微秒，
每批约 35.2 次。GRU 相关 kernel 合计 0.251 s，hazard MLP 为 0.900 s；真实 H2D／D2H
传输分别为 0.259／0.108 s。CPU 推进工作区间与 GPU kernel 的交集只有 0.0321 s，
约 96.4% 的 GPU kernel 时间发生在主机 `batch.enqueue` 区间内。

因此当前首先受主机数据准备、组批和算子提交限制，CPU 场查询与推进仍是次要大项。
同一个协调线程仍先后调用提交和 CPU 推进；GPU 异步尚未使这两部分主机工作并行。
后续优先考虑减少请求／结果的重复复制和分配、让独立提交线程与 CPU 推进重叠，
以及减少 ATen 小算子调用。仅减少 GRU 算术不会处理这些主要开销。

无计时基线在测量前为 33.02／31.12 s，测量后为 24.49／25.03 s；绝对时间存在明显
波动，不能将不同进程的时间差当成新的加速效果。以上占比来自同次计时分解；GPU 时间
来自单独捕获，不能与主机时间直接相加。未测完整 256 spp。各模式 PFM 逐字节一致。
配置、原始计时、CUDA 时间线及复现脚本见
[当前异步瓶颈报告](../outputs/renewal_async_bottleneck_20261010/summary.json)。

2026-10-10 首次合批验证（加入异步流水线之前）使用 h16、sigma 0.02、完整 NanoVDB shader ball 与方向环境，
基准进程固定到逻辑 CPU 掩码 `0xffff`，每个进程先预热一帧。旧版为 4096 批，
新版为 8192 批／16384 池；参数和模型加载不计时，会话创建计时：

| 测量 | 旧版 | 新版 | 加速 |
| --- | ---: | ---: | ---: |
| 128×128、4 spp，三轮交替中位数 | 14.17 s | 10.18 s | 1.39× |
| 512×512、1 spp，单次预热后测量 | 49.28 s | 32.99 s | 1.49× |

512 图平均段批量 4046→7056，平均初始化批量 38.1→264.4，平均 mixture 批量
23.5→157.2，结果回读 46072→9029。独立 Nsight 小图捕获确认 2416 次提交对应
2416 次实际 D2H 拷贝；H2D 扣除 26 个一次性模型权重／偏置上传后也为 2416 次。
GPU 内部仍有多次 kernel 和 D2D 操作。首轮旧版基准的亲和性设置失败，已明确排除；
有效原始记录、复现脚本和图像差异见[验证汇总](../outputs/renewal_queues_20261010/summary.json)。
小图新版三次耗时为 10.18/10.17/12.96 s，存在波动；这些结果不代表 256 spp 的完整耗时。

三组 CTest 全通过（C++ 114090 项检查），覆盖三类混合请求、延迟上下文、槽位复用、
小池／尾批、CPU 线程数及标量对照。已训练 h16 的 12 条固定 A/B 光线在 CPU/CUDA
上均通过对照，最大透射率差 `2.36e-7`；正式渲染与同配置诊断程序的 PFM 逐字节一致。
新旧调度的批形状不同，浮点舍入及后续随机采样可能分歧，不能承诺逐像素一致：本次
128 图 4 spp 的线性 RGB RMSE 为 0.02965，512 图 1 spp 为 0.04028。它们不是 GP
真值误差，也不用于判断统计偏差；核、模型权重、分段及采样公式未改动。

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

各向同性 A 模式的两个横向残差独立，标准差为 `sigma/ell`。B 模式保留出生点横向梯度残差：Matérn-3/2 的均值衰减为 `exp(-x)`、方差比例为 `1-exp(-2x)`；SE 分别为 `exp(-x*x/2)` 和 `1-exp(-x*x)`。方差均以 `expm1` 稳定计算，避免极短距离下相减归零。实际均值梯度取与命中段一致的三线性插值单侧导数；不会先归一化均值梯度或保存的碰撞梯度。

固定各向异性时令 `C=Cov(gradient)`、`a=C*w/(w^T*C*w)`、`P=I-a*w^T`。
横向残差协方差为 `P*C*P^T`，与整条直线上的场值和纵向导数独立；出生点相关系数为上述核对应的衰减。
输出梯度保持 `w.dot(gradient)=V<0`。该重建对两类平稳核都适用，不需要改变神经网络或批处理协议。

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

shader ball 示例配置选择 `auto`；`point_linear` 配置使用 `batch_size=8192`、`ray_pool_size=32768`，其他未调整的示例仍为 4096。实际选择记录于 `resolved_config.json` 的 `resolved_backend` 和 `render_summary.csv` 的 `neural_backend`。`neural_segments/neural_segment_batches` 可计算平均有效段批量，统一提交与辅助队列参数见上文。

`RenewalWavefront.cpp` 用光线池和三条请求队列调度射线：同次提交处理新飞行初始化、就绪段推理和先前命中所需的 mixture，统一回读后在 CPU 上累计 hazard／求逆并重建法线、反射和轮盘。完成一条样本后继续该像素的下一条样本；像素结束后复用槽位。每像素保留原来的随机数流、正态分布缓存和累加顺序，不根据批大小重新播种。A/B 起点、分段、核定义及碰撞采样分布均保持原定义；不同批形状的浮点舍入仍可能使个别随机路径分歧。

LibTorch 后端在所选设备上执行输入 `asinh` 和输出 `softplus`：初始化转换四个输入，段编码只转换前四列（第五列 `log(dx)` 保持原值），mixture 查询只转换均值与导数（`u` 保持原值）。CPU 打包原始 double，设备先执行 double `asinh` 再转 float32，避免提前转 float 使大幅值输入溢出。hazard 系数和 mixture scale 在回传前执行 `softplus(beta=1, threshold=20)`；CPU 继续执行 mixture 权重归一化、均值导数修正和 scale floor。原始 scale logits 随已有结果一起回传，在 CPU 上检查有限性，避免为小批次增加归约 kernel 或单独同步。

`torch_cuda` 在 GPU 上执行上述变换，`torch_cpu` 使用同一套 ATen 逻辑在 CPU 上执行；`scalar` 参考实现保持原样。无需重训或重新导出模型。不同设备的数学函数允许小幅浮点差异，不保证逐位相同的路径结果。

`RenewalBatchSession` 管理独立槽位状态；初始化使旧 mixture 缓存失效。`evaluate` 只推进列出的槽位一次，并缓存其**进入该段前**的状态和段编码，`mixture` 查询使用该缓存。网络权重、GRU 状态和进入段上下文常驻 GPU；CPU 每轮只传特征／槽位索引，下载少量 hazard 系数及实际命中的 mixture 参数。推理关闭 autograd，并在作用域内禁止 TF32，使用 float32；几何和累计量保持 double。不同 GEMM 实现仍可能带来浮点差异，因此不保证不同后端或批大小逐像素完全一致。

批处理渲染使用调用线程协调队列、一个专用线程处理 LibTorch，以及常驻 CPU 光线线程池执行初始化、hazard 积分／求逆、场查询和散射。`render.thread_count` 只表示 CPU 光线线程数：`1` 为一个工作线程，`0` 自动使用至多 8 个，并受光线池容量限制；另有协调和推理两个线程。它不修改 LibTorch 自身的推理线程设置。任务块最多 128 条光线，小批也进入工作队列，不在提交线程上计算。

每个任务独占自己的 slot、均值缓存和 RNG；图像按像素独占，统计按槽位累积。任务完成一小块就发布到三类请求队列，无整批 CPU 屏障。网络批成员可能随任务完成顺序变化，但每像素的随机数流和样本累加顺序保持。异常先等待工作线程结束，再回收设备传输。VDB 只读共享，各次查询使用独立 accessor。

`point_linear` 的逐段查询和首段构造都在工作线程执行；`cubic` 的整条均值剖面预计算也随新飞行初始化进入工作线程。两者工作量和内存占用不同，不能直接共用性能结论。下面的 2026-10-09 数据是原同步 CPU 线程池的历史测量。

2026-10-09 使用 h64、sigma=0.02 NanoVDB shader ball、CUDA、128×128、4 spp、批容量 4096，测试进程固定 P 核，三轮交错测量：修改前中位数 12.020 s，新版 1/2/4/8 线程分别为 11.016/9.792/8.894/8.555 s。8 线程相对修改前约 1.40×，相对新版单线程约 1.29×。计时排除加载与预热；生产程序不设置 affinity。图像和路径计数与修改前完全相同，h16/h64、CPU/CUDA、point_linear/cubic 的生产程序对照及全部 CTest 通过。阶段计时与完整记录见 [CPU 并行化验证报告](../outputs/renewal_cpu_parallel_20261009/REPORT.md)。

`render_summary.csv` 记录 `neural_cpu_workers`（CPU 推进可用线程数）、`neural_parallel_advance_batches` 和 `neural_serial_advance_batches`；后两者之和等于段批次数。网络和参数文件、分段及求逆精度都不因并行化改变，无需重训。`traceCameraPath` 单射线接口和 `renewal-query` 保留 Eigen 实现。小图、小批量仍可能受 GPU 启动和同步开销影响；渲染命令只输出配置选定的环境，完整计时包含推理会话建立和首次 CUDA 使用。

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

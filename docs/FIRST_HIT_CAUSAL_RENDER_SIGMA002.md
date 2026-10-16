# 残差 GRU：联合首碰采样与 GPU 批量渲染

本实现把 sigma=0.02 实验中的固定步长残差 GRU 和条件 mixture 梯度头接入实际渲染。支持 `scalar`、`torch_cpu`、`torch_cuda`；首碰完成后接已有的 CPU 续追踪。当前示例配置使用 scalar 首段与 classic_local 后续散射；下文最初的 CPU/CUDA 一致性及性能报告使用 neural_renewal 续追踪。训练文件和 checkpoint 保持不变。

- [渲染配置](../configs/render_first_hit_causal_gru_renewal_shader_ball_sigma002.json)
- [可视化验证报告](../outputs/first_hit_causal_render_sigma002_20261016/summary.html)
- [渲染数值结果](../outputs/first_hit_causal_render_sigma002_20261016/summary.json)
- [Python / C++ / CUDA 采样对照](../outputs/first_hit_causal_render_sigma002_20261016/parity/summary.json)

## 模型与单位

当前部署使用配对种子 **20261015**：距离主干 `gru_nll_20261015/best.pt`，梯度头 `mixture_gru_20261015/best.pt`。距离主干有 3,025 个参数，梯度头有 7,064 个参数，隐藏状态 16 维。此版本不包含 CDF 头导出。

入口为 activeDomain 与相机光线的交点。光线方向 ω 归一化，`q=s/ell`、`b=m/sigma`。训练内部步长固定为 **Δq=1/8**，积分网格为 **1/64**，每个积分片使用 8 点 Gauss–Legendre 求积，并在原始 cubic 边界拆开。

`profile_maximum_step` 控制场剖面构建，不是 GRU 推进步长。模型导出保存固定 GRU 步长，调用旧查询接口试图覆盖为其他步长会报错。缩小碰撞求根容差也不会改变 GRU 状态更新次数。

## 联合采样

每条光线固定使用四个开区间均匀随机数，依次用于入口分支、光学厚度、mixture 分量和分量内分位数。

1. 以 `P(k=+1)=Phi(b(0))` 抽取入口符号 k；初始化该分支的隐藏状态，抽取 `E=-log(U)`。
2. 对入口锚定的固定 q 区间，使用**进入该段前**的隐藏状态预测 log-hazard 残差 δ。该段的 hazard 为 `lambda(q)=lambda_local(k*b(q),k*b'(q))*exp(delta)`。
3. 积分该段 hazard。若未达到 E，减去整段光学厚度；仅对存活光线，用完整区间的左／中／右均值和导数 token 更新 GRU。最后一段截短至实际域出口。
4. 若该段发生碰撞，缓存当前隐藏状态。先定位命中的细积分片，再求其局部积分的反函数。求根只计算解析 local hazard，不再次调用神经网络。
5. 在命中 q、段内 phase 和缓存的入口状态上查询梯度头。先抽取 mixture 分量，再直接反演该分量在 `W>0` 上的截断正态分布。分布已经归一化，**不额外乘 W 或再做一次 Rice / NDF 权重修正**。

这样的采样实现了模型定义的 `p(k,q,W)`，包括域末端的逃逸概率。透射率查询仍积分两个符号分支：`T(q)=sum_k Phi(k*b(0))*exp(-H_k(q))`。梯度头在精确的正 recurrent 边界采用左极限状态，与训练约定一致。

## 世界坐标碰撞与法线

碰撞位置是 `x=x_entry+ell*q*omega`，采样的纵向导数是 `D=-k*sigma*W/ell`。

沿用现有的 SE 条件梯度重建：令 `mu=grad(m)`、`C=Cov(grad(F))`、`c=C*omega`、`v=omega^T*C*omega`，则

```text
g = mu + c/v * (D - omega·mu) + epsilon
Cov(epsilon) = C - c*c^T/v
n = g / ||g||
```

epsilon 在垂直于光线的二维平面内抽样。因此完整梯度严格满足 `omega·g=D`，也支持各向异性 C。实际场景使用 `sigma≈0.02`、`ell≈0.05656854`。

使用 neural_renewal 时，反射后保留完整 g 和入口符号，按已有 hybrid 逻辑变换场与梯度的侧向，再初始化条件续追踪。使用 classic_local 时，反射后的光线进入场景原有的局部密度与切向 NDF 模型。首碰 GRU 状态在这里结束；该模型没有接受过从已知表面碰撞出发的训练。

## 批量执行与性能边界

`CausalFirstHitSession` 在设备上保存每条光线的隐藏状态。活跃列表压缩后，稳定的 slot 编号仍指向原光线；命中或逃逸 slot 不再更新。mixture 只查询命中的光线，并使用留在该 slot 中的碰撞段入口状态。权重在 session 中复用，不按每段重新上传。

复用现有 `RenewalCpuExecutor` 并行构造剖面、计算局部积分／求根，以及执行法线重建和 CPU 续追踪。新模型的输入维度、初始化和 hazard 语义与旧 renewal GRU 不同，因此使用独立推理后端。

此版是同步的首碰批量实现，逐步仍有 CPU/GPU 交接，尚未融合进旧 renewal 的异步全路径 wavefront。实测中 GPU 整帧仍慢于多线程 scalar；报告中的首次运行时间含 CUDA 初始化，不能视为预热后的吞吐量。当前优先使用 scalar 做快速实验；GPU 路径用于批量一致性和后续性能优化验证。下节的预热与阶段计时进一步确定了积分、传输和调度的开销。

### CUDA 耗时诊断

新增独立诊断程序，在同一进程内连续渲染相同的 64×64、8 spp 图像，区分首次运行和预热后运行。只对生成的源码副本加计时，不修改生产采样代码，不增加逐阶段 CUDA 同步。原始数据见 [performance/summary.json](../outputs/first_hit_causal_render_sigma002_20261016/performance/summary.json)。

| 诊断方式 | 首次运行 | 预热第 1 次 | 预热第 2 次 |
| --- | ---: | ---: | ---: |
| CUDA，分阶段计时 | 22.49 s | 9.78 s | 10.03 s |
| CUDA，关闭分阶段计时 | 23.91 s | 11.26 s | 11.40 s |
| scalar，关闭分阶段计时 | 4.76 s | 4.59 s | 4.48 s |

两个 CUDA 进程的首帧均比其后两帧平均值多约 **12.6 s**。这些开销主要出现在首次初始化网络、GRU 和 mixture 的算子调用内；不能把先前的 22.48 s 全部当作稳态推理时间。字段／模型加载另计，不在此表的帧时间内。重复运行存在调度波动，因此保留两组 CUDA 测量，未把一次结果当成固定吞吐量。

分阶段计时的两次预热帧平均 9.90 s，互不重叠的主机侧耗时为：

| 阶段 | 平均耗时 | 占比 |
| --- | ---: | ---: |
| CPU local hazard 积分及碰撞求根 | 5.18 s | 52.3% |
| 网络调用，含传输、提交和等待 | 1.87 s | 18.9% |
| CPU 法线重建及后续路径追踪 | 1.98 s | 20.0% |
| CPU 均值剖面构建 | 0.47 s | 4.7% |
| 其余调度与整理 | 0.41 s | 4.1% |

这里的网络调用耗时是主机墙钟时间，不能解读为纯 GPU kernel 执行时间。

整帧累计 4,043 次批量 recurrent 推进、4,071 次读回、8,114 次特征上传和 8,082 次索引上传。配置上限是 batch 1024，但实际平均活跃 batch 只有 438。2,402 次推进的活跃数不足 512，进入现有的串行积分分支。相较 scalar 按整条光线分配给 CPU 线程，这种逐段同步、等待整批完成后再开始下一阶段的调度降低了 CPU 利用率。

因此优化优先级是：先将 local hazard 积分从逐段 CPU 等待中移出（预计算积分片或在 GPU 计算），再让不同批次的 CPU/GPU 工作重叠，并合并小算子、减少上传／读回。内部 Δq 仍保持 1/8；单纯扩大步长会改变已训练模型。

所有诊断帧的首碰计数一致，同一后端的冷／热帧像素和完全相同，并与原预览的像素和精确一致。诊断代码与模型哈希记录在结果中。可复现命令：

```powershell
cmake --build build --config Release --target macrofacet_profile_causal_first_hit -- /m:1
build/Release/macrofacet_profile_causal_first_hit.exe `
  outputs/first_hit_causal_render_sigma002_20261016/preview_torch_cuda_config.json `
  8 2 outputs/first_hit_causal_render_sigma002_20261016/performance/cuda_wall.json wall
```

把最后一个参数改为 `baseline` 可关闭分阶段计时；更换为 `preview_scalar_config.json` 可测 scalar。诊断目标仅在 LibTorch、字段库和 CUDA toolkit 可用时构建。

## 已完成验证与适用范围

- Python 解析测试覆盖零残差指数律、常数残差缩放、正半轴速度反演和短区间。
- C++ 测试覆盖原始 cubic 分段变化、短光线、活跃 slot 压缩、CPU 并行积分、三种后端、各向异性梯度重建、固定随机流和白炉守恒；完整 C++ 测试套件通过。
- 7 条剖面 × 512 次采样，与 Python 冻结模型比较；入口分支与 hit/escape 全部相同。scalar 透射率查询最大差约 `1.96e-8`，三后端归一化距离最大差约 `1.38e-5`。另检查 batch 37 与 256，以及非法 GRU 步长覆盖。
- 实际 sigma=0.02 场景的 64×64、8 spp 预览比较 32,768 条光线。完整梯度投影、法线长度和世界坐标重建均通过；首碰事件、入口分支和随机流一致。最终误差、时间、数值失败与深度上限截断数记录在报告中。
- 冻结训练协议的 10 个源码文件哈希保持不变。没有重新训练或修改训练数据。

当前这次预览（RTX 5090，CPU 8 线程，包含 CUDA 初始化）测得 scalar **3.65 s**、CUDA **22.48 s**。两者首碰数都是 **15,238**，法线分量最大差 **9.14e-6**，纵向导数约束最大误差 **1.33e-15**。两者数值失败均为 0；scalar 续追踪有 **1/32,768** 条路径达到深度上限，CUDA 为 0。完整图像 RGB RMSE 为 **0.01066**，并非逐位一致的渲染结果。上述时间是单次运行，不作为稳态吞吐量结论。

这些检查证明部署采样与现有神经模型一致，不证明它等于连续 GP 的无偏真值。当前 W 标签来自有限历史 GPIS 的 `tungsten_linear` 割线速度；横向梯度使用 SE 条件高斯假设，没有独立的三维梯度／法线监督。现有续追踪也有自己的近似。微小 float32 首碰差异可能改变后续路径的离散采样，完整渲染不要求逐位相同。报告保留深度上限截断计数，不将截断当作物理逃逸。

## classic_local 后续散射的亮噪点诊断

切换到 classic_local 时必须显式选择合适的法线提议分布。当前 Gaussian NDF 配置使用：

```json
"classic_phase_proposal": "target_vndf"
```

该字段位于 `transport` 下，与 `first_hit`、`continuation` 同级。遗漏时默认使用 `uniform`，后续每次反射的权重含有 `F * p_VNDF / q_uniform`；多次散射会放大少数路径的权重。`target_vndf` 直接采样同一个目标 VNDF，权重化为 Fresnel F。这只改变后续散射的采样提议，不改变首段网络、推进步长或目标相函数。

用户的 512×512、128 spp 渲染使用了 uniform，记录为 33,554,432 条相机路径、0 次数值失败、0 次深度上限截断；线性 PFM 最大通道值为 243.745，而环境光最大通道值为 1。原图保留在 `outputs/render_first_hit_causal_gru_local_shader_ball_sigma002`，配置修复不会追溯改变已有图像。

固定首段模型与全部场景参数，以两个独立种子 17429、51293 渲染 64×64、128 spp，只改变 `classic_phase_proposal`：

| 后续法线提议 | 估计 RGB 噪声 RMS | 两张图的最大通道值 | 平均渲染耗时 |
| --- | ---: | --- | ---: |
| uniform | 0.333741 | 7.097 / 46.903 | 43.63 s |
| target_vndf | 0.009247 | 0.832 / 0.829 | 43.50 s |

噪声 RMS 使用线性 PFM 的 `sqrt(mean((I_seed1-I_seed2)^2)/2)` 估计；这是两个种子的小图统计，uniform 的极端亮点对估计影响很大，不能视为其他场景的固定降噪倍数或相对于真值的误差。各配对种子的首碰计数完全一致，四次渲染的数值失败与深度截断均为 0。结果定位到后续提议分布造成的高方差，不验证首段模型本身的偏差。

- [对照图和报告](../outputs/first_hit_causal_local_noise_sigma002_20261016/summary.html)
- [完整统计](../outputs/first_hit_causal_local_noise_sigma002_20261016/summary.json)
- [复现实验脚本](../outputs/first_hit_causal_local_noise_sigma002_20261016/diagnose.py)

## 运行

从项目根目录执行。需要既有训练 checkpoint 和 sigma=0.02 的 full-domain NanoVDB 场；仅在选择 neural_renewal 续追踪时需要 SE renewal 模型。

```powershell
$env:PYTHONPATH = 'python'
python -B -m first_hit_se.causal_sampling `
  --backbone outputs/first_hit_causal_residual_sigma002_20261015/runs/gru_nll_20261015/best.pt `
  --gradient outputs/first_hit_causal_gradient_sigma002_20261016/runs/mixture_gru_20261015/best.pt `
  --output outputs/models/first_hit_causal_gru_sigma002/first_hit_model.json

cmake --build build --config Release --target macrofacet_experiments macrofacet_query_first_hit_se macrofacet_tests -- /m:1

python -B -m first_hit_se.causal_sampling_check
python -B -m first_hit_se.causal_render_check

build/Release/macrofacet_experiments.exe render `
  --config configs/render_first_hit_causal_gru_renewal_shader_ball_sigma002.json
```

导出会验证 gradient 训练协议及所配 backbone 的哈希和种子，拒绝随意混用主干与梯度头。当前渲染配置使用 `transport.first_hit.backend="scalar"` 的多线程 CPU 首段、classic_local 续追踪及 target_vndf 法线提议。改首段 backend 为 `torch_cuda` 可测试 CUDA；`batch_size` 只控制批量大小，不改变模型步长。`torch_cpu` / `torch_cuda` 要求启用 LibTorch，显式不可用的后端会报错，不静默回退。

## 实现入口

- [Python 导出与反演参考](../python/first_hit_se/causal_sampling.py)
- [C++ 联合采样器](../src/learned/CausalFirstHitModel.cpp)
- [CUDA / LibTorch 后端](../src/learned/CausalFirstHitTorchBackend.cpp)
- [首碰法线重建](../src/transport/FirstHitSEMedium.cpp)
- [批量 hybrid 渲染接入](../src/integrator/MacrofacetPathTracer.cpp)

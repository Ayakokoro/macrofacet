# 多视角训练与梯度分布对照：sigma=0.02

日期：2026-10-15。完成 6 组距离训练和 6 组冻结主干的梯度头训练，均使用三个配对随机种子。

后续已完成 [MLP 残差正则与透射率监督实验](FIRST_HIT_RESIDUAL_SUPERVISION_SIGMA002.md)：组合监督降低了最坏光线误差，平均误差基本持平，但部分分位数和个别光线退化，详见该轮独立报告。

更新：后续 [逐段残差 GRU](FIRST_HIT_CAUSAL_RESIDUAL_SIGMA002.md) 改善了距离拟合，其 [冻结主干的条件梯度实验](FIRST_HIT_CAUSAL_GRADIENT_SIGMA002.md) 也改善了逐光线与晚碰撞校准。当前仍保留 Mixture 为梯度主方案，CDF 作对照；新实验采用相同数据及三个配对种子。

**当前继续以 local＋residual hazard 为距离主线，梯度暂时保留截断高斯混合头。** 新增训练视角后，残差模型的平均透射率误差明显下降，但最坏光线误差增大。梯度 CDF 在本次配置下没有显示稳定的精度优势，当前实现的采样成本也更高。

- [完整 Markdown 报告](../outputs/first_hit_multiview_sigma002_20261015/summary.md)
- [图表页面](../outputs/first_hit_multiview_sigma002_20261015/summary.html)
- [结构化结果](../outputs/first_hit_multiview_sigma002_20261015/summary.json)
- [实验配置](../python/configs/first_hit_se_multiview_sigma002.json)
- 前序：[首轮设计实验](FIRST_HIT_DISTANCE_DESIGN_SIGMA002.md)、[扩展相机测试](FIRST_HIT_CAMERA_HOLDOUT_SIGMA002.md)

## 数据与比较方式

使用 `outputs/fields/shader_ball_full_0.02.nvdb`，实际 sigma≈0.02、ell≈0.05656854。
参考标签仍由 H=1、B=8、步长 1/64 的 GPIS function-space Tungsten-linear 采样器生成。

新增 8 个训练相机和 2 个验证相机，每个相机选择 36 条画面覆盖光线和 28 条剖面差异光线，共 640 条，每条采样 2048 次。选择仅使用几何剖面及原训练集，不使用测试标签。

| 划分 | 原有光线 | 新增光线 | 合计 | 参考样本 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 183 | 512 | 695 | 1,423,360 |
| 验证 | 30 | 128 | 158 | 323,584 |
| 固定测试 | 280 | 0 | 280 | 2,293,760 |

新增训练/验证相机与固定测试相机分离；1133 条物理光线不重复，4,040,704 个参考实现的随机种子不重复。原有 493 条剖面的特征、距离、命中标记、入口符号和梯度标签精确保留，旧 checkpoint 未修改。

测试集到最近训练剖面的描述距离，中位数由 0.70757 降到 0.46260，95% 分位数由 1.09918 降到 0.80335，说明训练剖面覆盖确实扩大。描述包含无量纲均值、斜率与长度；它不是概率误差的保证。

固定测试集已经在前轮用于设计诊断，本轮不参与优化或 checkpoint 选择；它不是新的盲测集。保留的原训练、验证和 24 条测试光线仍来自旧相机。这是同一场景内的比较。

## 距离结果

两种网络均使用整段均值剖面的 256 点输入、宽度 64 的 MLP、32 维编码和 64 个 log hazard 控制值，参数量均为 57,632。输出插值到 1024 个 hazard 格子。

直接方案预测 log hazard；残差方案预测对解析 local log hazard 的修正，即每个入口符号分支满足 `lambda = lambda_local * exp(delta)`。因此这里的“残差”是对概率模型的修正，并非另加一套 ResNet 架构。

三个种子为 20261015、20261016、20261017，各训练 500 epochs，每轮每条光线抽取 512 个参考样本，按验证集边缘距离 NLL 选择 checkpoint。同种子的初始参数、样本抽取顺序和优化设置一致。残差方案额外使用原始三次剖面的解析信息。

下表是逐光线最大透射率绝对误差的均值，神经网络条目平均三个种子。

| 方案 | 全部 280 条 | 画面覆盖 128 条 | 剖面差异 128 条 |
| --- | ---: | ---: | ---: |
| Local | 0.025339 | 0.021292 | 0.029040 |
| 旧 residual | 0.026686 | 0.016666 | 0.037889 |
| 新 direct | 0.074070 | 0.044302 | 0.108102 |
| 新 residual | **0.018533** | **0.012151** | **0.025622** |

新 residual 的总体误差相对 Local 降低约 26.9%，相对旧 residual 降低约 30.6%。在此前表现较弱的剖面差异组上，平均指标也已优于 Local。总体 T RMSE 为 0.009479，Local 为 0.018494；距离 NLL 分别为 0.701130、0.712595。

**平均改善同时伴随尾部退化。** 各种子最坏光线误差的均值，从旧 residual 的 0.169377 增至 0.248525，Local 为 0.145240。最突出的第 266 条测试光线，Local 误差仅 0.022938，新 residual 的三个种子为 0.248187、0.199450、0.297939；模型过早降低了后段透射率。

![失败案例](../outputs/first_hit_multiview_sigma002_20261015/distance_failures.png)

这些案例是在评估后按误差挑选的诊断图。它们说明增加视角并不能保证残差处处优于解析基线。

三个 residual 分别选择 epoch 40、15、20，后期验证 NLL 上升。新旧实验虽然均为 500 epochs，但扩充数据后每轮更新次数增加，验证集也变化，因此不能把改善完全归因于数据多样性。

## 梯度 CDF 与混合分布

每个已训练 residual 主干均冻结，分别接两个梯度头，各训练 150 epochs：

| 梯度头 | 表示 | 参数量 |
| --- | --- | ---: |
| Mixture | 8 分量正半轴截断高斯 | 8088 |
| CDF | 8 区间单调有理二次样条，变换指数基分布 | 8218 |

两者使用相同冻结编码、参考首碰位置 q、入口符号 k、局部均值及导数；隐层初始化及训练样本顺序相同。CDF 的输出分布为 `F(w)=G(1-exp(-w/s))`，通过其导数的似然训练，并支持解析逆采样。这是选择可逆 CDF 表示，不是改用经验 CDF 的逐点回归损失。

拟合对象为条件纵向速度 `W=-k*(ell/sigma)*dot(omega,g)>0`。分开训练是在隔离距离误差与条件梯度误差，不是假设两者独立；生成时仍先采首碰位置，再在该位置采梯度。完整三维法线尚未在本轮做渲染验证。

下表均为三个种子的测试均值；越小越好。

| 指标 | Mixture | CDF |
| --- | ---: | ---: |
| 条件速度 NLL | **1.257781** | 1.259126 |
| 全局 PIT KS | **0.002197** | 0.002866 |
| 逐光线 PIT KS 均值 | **0.035248** | 0.035386 |
| 联合 NLL | **1.622265** | 1.623250 |

两者精度接近，三组配对 NLL 均是 Mixture 略好，PIT 的逐种子排序则不一致。仅凭这三个种子和单场景结果，不能断言 CDF 表示普遍更差；目前没有替换现有头的证据。

整体 PIT 仍会掩盖位置差异：`q/Q>=0.75` 区间有 15,802 个测试命中样本，其 PIT KS 均值为 Mixture 0.043979、CDF 0.058855。较晚碰撞处的条件梯度仍需改进，不能由整体 KS 很小推断所有位置都已校准。

## 联合采样与成本

使用预先指定的第一种子，对每条固定测试光线生成 8192 个联合样本。两头共用距离样本，比较 15 个预设矩形上的 `P(hit, q/Q<=a, W<=b)`；速度阈值来自训练命中速度的四分位数。

平均矩形概率绝对误差为 Mixture 0.010431、CDF 0.010393，几乎一致。这里只检验有限矩形和一个种子，含参考与模型采样噪声，不能据此判定 CDF 更优。

| 计时范围 | Mixture | CDF |
| --- | ---: | ---: |
| CPU 单样本，仅逆采样 | 78.329 µs | 218.482 µs |
| CPU 单样本，梯度头＋采样 | **179.324 µs** | 394.392 µs |
| CUDA batch=4096，梯度头＋采样摊销 | **0.175 µs/样本** | 0.364 µs/样本 |

环境为 RTX 5090、PyTorch 2.12.0+cu130，CPU 单线程，7 次重复取中位数。Mixture 采用选分量＋截断高斯逆采样，没有人为加入混合 CDF 的迭代求逆。

计时不含随机数生成、场访问、剖面构建、主干编码、local 积分和距离采样。这是当前 PyTorch eager 实现的成本，不能外推为优化后的 C++/GPU 内核成本或完整渲染速度。

## 当前判断与下一轮优先级

继续研究 residual hazard，但先处理少数剖面上的过度修正；梯度保留 Mixture 作为主基线，CDF 保留作对照。当前距离失败案例的误差远大于两种梯度头的平均差异。

下一轮建议优先对比残差幅度正则或受限残差，并在独立的新训练/验证剖面中增加长距离接近零均值、反复穿零等情况。不要把上述固定测试光线直接回灌训练，也不要按其误差挑 checkpoint。此前发现的前缀一致性问题仍需单独解决，尚不能用这个整段 MLP 替代逐段 RNO/GRU。

## 验证与复现

16 项数值及回归测试通过，覆盖 CDF/PDF 导数关系、归一化、逆采样、极端尾部、距离分布和相机选择。另做了真实参考标签上的冻结主干更新与采样检查。正式运行再次确认六个梯度头训练期间主干不变、配对初始化一致，所有 checkpoint 仅按验证指标选择。

158 条验证光线的 local 1024/2048 格最大 T 差为 0.001584；同网格 8 点改 2 点求积最大差为 8.05e-7。本轮保持 1024 格、8 点求积。此前扩展测试域中的网格敏感度最高约 0.002843；这些检查及有限记忆参考标签不构成连续 GP 精度证明。

在仓库根目录运行：

```powershell
python -B python/first_hit_se_cli.py multiview-data --config python/configs/first_hit_se_multiview_sigma002.json
python -B python/first_hit_se_cli.py multiview-study --config python/configs/first_hit_se_multiview_sigma002.json
```

已完成的训练会复用报告，未完成的训练可从保存状态继续。单独重建图表与报告：

```powershell
python -B python/first_hit_se_cli.py multiview-study --config python/configs/first_hit_se_multiview_sigma002.json --phase report
```

checkpoint 位于 `outputs/first_hit_multiview_sigma002_20261015/runs/{direct,residual}_<seed>/best.pt` 和 `gradient/{mixture,cdf}_<seed>/best.pt`。

数据 SHA-256：`2fead63699474ebcde1d47169e49954994bdab270c75788a31f403f3d1e844ef`。

实现入口：[数据扩充](../python/first_hit_se/multiview_data.py)、[距离实验](../python/first_hit_se/multiview_study.py)、[梯度 CDF](../python/first_hit_se/gradient_cdf.py)、[冻结梯度训练](../python/first_hit_se/frozen_gradient_study.py)、[联合采样与计时](../python/first_hit_se/multiview_sampling.py)。

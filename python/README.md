# Renewal+ 序列数据与网络训练

实现 [方案](../GPIS_RenewalPlus_Neural_Rendering.md) 第 5–9 节：共享三次均值剖面、A/B 起点、GRU 历史状态、单调累计 hazard、正半轴截断 Gaussian mixture 和联合删失似然。固定核为 `rho(x)=(1+x)exp(-x)`，`beta=1`。旧 I-spline 包和旧 Matérn-3/2 配置不参与此流程。

## 运行

从仓库根目录执行，路径以仓库为基准。已有可用 PyTorch 时不必重新安装；GPU 训练需要匹配设备的 PyTorch CUDA 构建。C++ 多射线渲染可复用该安装的 LibTorch CPU/CUDA 库；标量 Eigen 后端也保留，详见 [C++ 渲染](../docs/RENEWAL_CPP.md)。

```powershell
python -m pip install -r python/requirements.txt
cmake --build build --config Release --parallel 4

python python/renewal_cli.py collect --config python/configs/renewal_matern32_dataset.json
python python/renewal_cli.py audit-reference --manifest outputs/renewal_dataset_v2/manifest.json --output outputs/renewal_dataset_v2/reference_audit.json
python python/renewal_cli.py prepare --manifest outputs/renewal_dataset_v2/manifest.json --output outputs/renewal_dataset_v2/dataset.pt
python python/renewal_cli.py train --config python/configs/renewal_matern32_train.json
```

可用 `train --config ... --resume` 从 `last.pt` 的下一轮继续。恢复要求配置和数据文件哈希一致；每轮使用固定的独立 shuffle 种子。相同软件、设备和数据下可复现，但不保证不同硬件版本之间逐位相同。

默认训练 16 轮，按验证集联合 NLL 保存最佳模型，连续 5 轮未改善则停止。`device` 可改为 `cpu`；新实验应使用新的输出目录。

当前默认实验使用 `sigma_range=[0.01,0.1]`，GRU 隐藏状态为 64，模型保存到 `outputs/models/renewal_matern32_h64_v2`。段编码器和两个输出头的宽度保持不变，参数量为 87,324。原始 v1 数据和 128 维模型保留在各自的旧目录；`outputs/.../collection_config.json` 是已生成数据的记录，不应修改它来声明新的采样范围。

在同一份 v2 数据上训练 128 维对照模型：

```powershell
python python/renewal_cli.py train --config python/configs/renewal_matern32_train_h128_v2.json
```

在同一份 v2 数据上进一步缩减到 16 维隐藏状态：

```powershell
python python/renewal_cli.py train --config python/configs/renewal_matern32_train_h16_v2.json
python python/renewal_cli.py export-model --checkpoint outputs/models/renewal_matern32_h16_v2/best.pt --output outputs/models/renewal_matern32_h16_v2/renewal_model.json
```

该配置保持段编码器和输出头宽度、训练种子及优化设置不变。起点编码器与隐藏状态一起缩小，参数量为 49,068；新权重保存在独立目录 `outputs/models/renewal_matern32_h16_v2`。

单独评估：

```powershell
python python/renewal_cli.py evaluate --checkpoint outputs/models/renewal_matern32_h64_v2/best.pt --dataset outputs/renewal_dataset_v2/dataset.pt --split test --device cuda --output outputs/models/renewal_matern32_h64_v2/test_metrics.json
python python/renewal_cli.py export-model --checkpoint outputs/models/renewal_matern32_h64_v2/best.pt --output outputs/models/renewal_matern32_h64_v2/renewal_model.json
python -m unittest discover -s python/tests -v
```

设置 CMake 的 `MACROFACET_TEST_NEURAL=ON` 可把 Python 测试纳入 CTest。默认关闭，避免给纯 C++ 构建增加训练依赖。

## 几何采集与参考数据

默认 24 个场景：训练 16 个、验证 4 个、测试 4 个。每组包含平面、球、切开球和 shader ball，几何参数、尺度、网格尺寸与射线独立生成。每场景每种起点 216 条剖面，每条 16 个随机实现；标准化长度为 2、4 或 8。训练共 110,592 条轨迹，验证和测试各 27,648 条。

v2 每个场景的全局 sigma 从 `[0.01,0.1]` 均匀采样，相关长度从 `[0.08,0.2]` 采样；仍使用空间恒定 sigma。v1 的 sigma 范围是 `[0.06,0.14]`，两份数据的测试 NLL 不能直接作为架构优劣比较，必须在同一测试集上评估。

新增的 `macrofacet_experiments bake-field --config ...` 仅烘焙完整 NanoVDB 和写出 resolved 配置，复用渲染器的插值基础设施。随后使用现有 `first-passage` 命令抽取均值段和采样标签。射线剖面中的导数保持段内单侧约定；跨体素不重置随机状态。

模式 B 的完整梯度按声明的范围合成：总向外标准化导数取 `[0.15,3]` 的 log-uniform 分布，横向残差为高斯。这是参数覆盖数据，尚不是由 BSDF 发射统计采集的路径分布。模式 A 仅包含空域事件，隐藏初始随机值不输入网络。

参考步长为 `1/64`、`1/128`，最小细分步长 `1/4096`。只有 `training_resolution=1` 的最细标签进入训练，较粗标签留作审计。`audit-reference` 比较 A/B 两种条件下的碰撞概率、透射率和穿越速度分位数，并给出按剖面分组的概率差标准误差。这些诊断不是连续首达误差的认证界；更严格的实验应继续检查最小步长。

`manifest.json` 保存场景、划分、源目录、几何及可执行文件哈希。采集可重跑，只有输入配置和可执行文件哈希都一致的已完成首达任务才会复用。

也可以用自定义 manifest 导入已有 C++ 参考输出：每个 source 指定 `scene_id`、`split`、`directory` 和 `geometry`。准备阶段检查单位衰减核标记、场景和几何跨划分泄漏、段的连续性、起点信息一致性、正速度、右删失和重复轨迹。旧核尺度数据会被拒绝。不能将同一几何的随机实现随机分到三个集合。

## 数据契约

`dataset.pt` 是共享剖面表和轨迹索引表，使用 `torch.load(..., weights_only=True)` 读取：

- `features`、`offsets`：变长剖面，特征顺序为 `[b0,b1,dx*db0,dx*db1,log(dx)]`。
- `initial`：`[is_surface,b0,Z0_known,D0_known]`。A 的最后两项为零并在模型内再次屏蔽；B 为 `[-b0,beta_g-b'(0+)]`。
- `sample_profile`、`hit`、`distance`、`speed`：轨迹关联和标准化标签。miss 的距离是预先确定的截断长度，速度为 NaN。
- `end_segment`、`end_u`：命中段或删失终点，训练加载器只返回到该处的前缀。
- `scene_ids`、`split`、`provenance`：划分与参考来源，均不输入网络。

几何特征用 float64 保存，防止很短的体素段在计算导数时因端点相减损失精度。进入编码器前对前四项做可逆 `asinh`，然后转为网络的 float32；log 段长保持原值。距离、速度均使用方案中的无量纲单位。

## 网络和损失

当前默认训练配置的段编码器宽 64，GRU 隐状态 64；另有 16 维缩减实验和 128 维对照配置，使用相同的数据与优化设置。C++ 的 Eigen 和 LibTorch 后端从模型文件读取尺寸。第 i 段的 hazard 只使用进入该段的状态和当前段特征；GRU 的输出用于下一段。padding、未来段、场景编号和随机实现种子不进入有效历史。

hazard 头输出 4 个 softplus 系数，解析积分得到四次 Bernstein 累计多项式。实现保证起点累计量为零、段间累计连续、非负 hazard，且没有固定正 hazard 下限。极小 hazard 的对数用稳定渐近式计算。

导数头输入当前上下文及段内 `u,b(u),b'(x)`，输出 8 个截断 Gaussian 分量。均值为 `-b'(x)+delta_mu`，尺度为 `softplus(t)+0.01`。每个分量独立在正半轴归一化；似然使用 log-CDF、log-sum-exp，并对强负均值尾部使用消减稳定的表达式。不会额外乘一次 Rice 速度权重。

命中样本使用 `H(x)-log(h(x))-log(q(w|x))`；miss 使用 `H(x_max)`。所有轨迹等权，保留原有命中／未命中比例。AdamW 学习率 `3e-4`，weight decay `1e-5`，梯度范数限制为 1。

## 产物与适用范围

`best.pt` 含网络参数、架构、核约定、选中轮次、数据哈希和验证指标。`last.pt` 还包含优化器和历史，供恢复训练。其他产物包括训练配置、设备信息、`history.json`、未训练模型的验证基线，以及验证／测试指标。

评估报告包含联合 NLL、距离 NLL、命中速度 NLL、速度 PIT 的 KS 偏差、不同长度处的透射率校准和 Brier score。分别报告 A/B 与场景的似然；测试集不参与模型选择。连续密度 NLL 可能为负，其大小只应在同一标准化数据契约下比较。

模型已支持导出到 C++，执行 GRU 推理、累计量求逆、透射率、穿越速度及完整梯度／法线采样，并通过 `neural_renewal` 模式进行多次散射路径追踪，详见 [C++ 使用说明](../docs/RENEWAL_CPP.md)。渲染使用 `export-model` 导出全部权重；`export-hazard` 仍用于只查询距离／透射率。现有三个渲染模式不读取新模型。首轮数据只覆盖上述四类几何与配置尺度；真实网格、薄层、其他拓扑及超出范围的参数，需要独立测试和补充数据后再评价泛化。

## 首轮训练结果（2026-10-08）

使用 v1 配置（sigma `[0.06,0.14]`、隐藏状态 128），在 RTX 4060 Laptop GPU 上完成 14 轮后早停，按验证集选取第 9 轮。网络有 174,172 个参数，训练和逐轮验证总计约 882 秒。

| 指标 | 结果 |
| --- | --- |
| 未训练模型验证集联合 NLL | 2.84956 |
| 最佳验证集联合 NLL | 1.09880 |
| 测试集联合 NLL | 1.09094 |
| 测试集命中速度 PIT KS | 0.01440 |
| 测试集终点平均透射率：预测／参考 | 0.46966 / 0.47153 |
| 终点平均透射率绝对差：A／B | 0.00336 / 0.00039 |

透射率差是测试集汇总校准误差，不是每条射线的误差界。最佳权重为 `outputs/models/renewal_matern32_v1/best.pt`；同目录的 `test_metrics.json` 包含 A/B 和各个未见场景的指标。三组 CTest 全部通过，其中新网络与数据管线包含 13 项测试。

## 隐藏状态缩减实验 v2（2026-10-08）

重新生成 sigma `[0.01,0.1]` 的 24 场景数据，在同一份数据上分别训练隐藏状态 64 和 128 的模型，其余架构和训练超参数相同。两者均完成 16 轮；按验证集选择的权重分别来自第 16、15 轮。段编码器仍为 64 维，hazard 头仍为 64 宽，速度头仍为 128 宽和 8 个混合分量。64 维模型的起点编码器随隐藏状态一起缩小，hazard／速度头的输入维度分别变为 128／131。

| 指标 | h128 v2 对照 | h64 v2 |
| --- | ---: | ---: |
| 参数量 | 174,172 | 87,324 |
| 最佳验证集联合 NLL | 0.91758 | 0.91627 |
| 测试集联合 NLL | 0.87801 | 0.87332 |
| 测试集距离 NLL | 0.51065 | 0.51108 |
| 测试集命中速度 PIT KS | 0.02920 | 0.02107 |
| 测试集终点平均透射率绝对差 | 0.00724 | 0.00679 |
| CUDA 方向环境渲染中位数 | 7.9378 s | 6.9499 s |
| CUDA 白炉渲染中位数 | 6.2594 s | 4.9695 s |

本次实验参数减少约 49.9%，测试集精度相近。渲染计时使用 RTX 4060 Laptop GPU、shader ball 的 sigma `0.07`、64×64、8 spp、batch 4096，三轮顺序交替执行；方向环境耗时减少约 12.4%。权重变化也会改变采样路径，且计时存在调度波动，因此这里报告的是本次完整渲染测量，不是纯矩阵计算的加速保证。训练期间 h128 进程调整过 CPU affinity，训练总时间不能用作架构性能对比。

另做了 GPU 常驻张量的 PyTorch 单段微基准（段编码、hazard、GRUCell，含 Python／算子调用开销，不含几何、传输、法线采样）：batch 4096 时 h128／h64 的中位数分别为 0.9426／0.9568 ms，未见稳定加速。它不等价于 C++ 整帧计时，也说明参数量不能直接换算成延迟。[微基准原始记录](../outputs/renewal_h64_v2_validation/network_benchmark.json) 包含各批量和全部重复测量。

另做了 GPU 常驻张量的 PyTorch 单段微基准（段编码、hazard、GRUCell，含 Python／算子调用开销，不含几何、传输、法线采样）：batch 4096 时 h128／h64 的中位数分别为 0.9426／0.9568 ms，未见稳定加速。它不等价于 C++ 整帧计时，也说明参数量不能直接换算成延迟。[微基准原始记录](../outputs/renewal_h64_v2_validation/network_benchmark.json) 包含各批量和全部重复测量。

64 维权重通过了 A/B 各 6 条射线、合计 496 个段的 Python／C++ 标量及 LibTorch CPU/CUDA 对照；CPU/CUDA 对标量的透射率最大绝对差小于 `1e-7`。14 项 Python 测试（含 C++ parity）通过，所有比较渲染及 sigma `0.01`、`0.1` 的运行检查均无数值失败或深度截断。两个 sigma 端点的渲染只检查执行，不能替代该处的参考精度评估。

默认 shader ball 配置已加载 `outputs/models/renewal_matern32_h64_v2/renewal_model.json`。本轮使用单个训练种子，不能据此断言更小网络在所有数据上更优。[完整比较](../outputs/renewal_h64_v2_validation/comparison.json)、[小模型报告](../outputs/models/renewal_matern32_h64_v2/run_report.json)、[新数据记录](../outputs/renewal_dataset_v2/collection_config.json) 包含可复核结果；旧 v1 数据与模型保持可用。

## 隐藏状态 16 维实验（2026-10-09）

在相同 v2 数据、种子和优化设置下，将 `hidden` 从 64 缩减为 16；段编码器宽 64、hazard 头宽 64、速度头宽 128 和 8 个混合分量保持不变。起点编码器与隐藏状态一起缩小，两个输出头的输入维度随之变化。完成 16 轮，按验证集联合 NLL 选中第 16 轮；训练和逐轮验证合计约 621 秒。训练进程在第 9 轮后固定到几个 P 核，因此训练耗时不用于比较架构速度。

| 指标 | h64 v2 | h16 v2 |
| --- | ---: | ---: |
| 参数量 | 87,324 | 49,068 |
| 验证集联合 NLL | 0.91627 | 0.92089 |
| 测试集联合 NLL | 0.87332 | 0.88243 |
| 测试集距离 NLL | 0.51108 | 0.51598 |
| 命中速度 PIT KS | 0.02107 | 0.02080 |
| 终点平均透射率绝对差：全部 | 0.00679 | 0.00838 |
| 终点平均透射率绝对差：A | 0.01022 | 0.01152 |
| 终点平均透射率绝对差：B | 0.00336 | 0.00523 |
| 本轮 CUDA 整帧中位数 | 15.5065 s | 15.6136 s |

参数减少 43.8%，每条射线的 GRU 状态存储减少 75%；这不等于总显存减少 75%。该次训练的测试似然与透射率校准略差，不能据单个种子认定 16 维的容量极限。透射率指标是汇总校准差，不是逐射线误差界。

本轮计时使用当前 point_linear NanoVDB 场、128×128、4 spp、batch 4096、RTX 4060 Laptop GPU，每个进程预热 1 spp 后计时，h64/h16 交替执行各三次。h64 为 16.1415、15.5065、15.1498 秒，h16 为 16.5257、13.9321、15.6136 秒；没有测出整帧加速。两模型的段查询数分别约 1529 万和 1592 万，计时包含采样路径差异、CPU 工作和 GPU 推理，不是纯网络基准。此处场景和分辨率与上面的 h128/h64 实验不同，时间不能跨表比较。

新权重通过了 A/B 各 6 条射线、合计 496 段的 Python/C++ scalar 及 LibTorch CPU/CUDA 对照；CUDA 透射率最大绝对差约 `2.36e-7`。生产渲染无数值失败或深度截断，输出与 profiling 可执行文件一致。C++ 直接从模型读取 16 维尺寸，无需重编译。

使用新权重渲染当前场景：

```powershell
.\build\Release\macrofacet_experiments.exe render --config configs/render_neural_renewal_shader_ball_ply_point_linear_h16.json
```

该独立配置保持当前场景、相机和 128×128、64 spp 设置，结果输出到 `outputs/render_neural_renewal_shader_ball_point_linear_h16`。上面的计时只用 4 spp。训练配置为 `python/configs/renewal_matern32_train_h16_v2.json`，模型为 `outputs/models/renewal_matern32_h16_v2/renewal_model.json`。完整指标、原始计时与复现脚本见 [实验报告](../outputs/renewal_h16_v2_validation/REPORT.md) 和 [模型报告](../outputs/models/renewal_matern32_h16_v2/run_report.json)。

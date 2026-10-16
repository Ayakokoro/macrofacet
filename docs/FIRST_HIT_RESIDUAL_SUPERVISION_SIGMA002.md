# MLP 残差正则与透射率监督：sigma=0.02

在现有整段剖面 MLP 上加入残差幅度正则和透射率曲线监督。沿用 [多视角实验](FIRST_HIT_MULTIVIEW_SIGMA002.md) 的数据、网络和解析 local hazard；通过三个配对种子比较原 residual、residual＋L2、residual＋L2＋T。

**9 组训练与固定测试评估已完成。组合监督将最坏光线误差的三种子均值降低约 41.1%，平均误差基本持平，但 P95 变差，过度修正尚未全面解决。** 验证集选出的权重为 eta=0.01、beta=100。

- [实验配置](../python/configs/first_hit_se_residual_supervision_sigma002.json)
- [完整报告](../outputs/first_hit_residual_supervision_sigma002_20261015/summary.md)
- [图表页面](../outputs/first_hit_residual_supervision_sigma002_20261015/summary.html)
- [结构化结果](../outputs/first_hit_residual_supervision_sigma002_20261015/summary.json)
- [验证集权重选择](../outputs/first_hit_residual_supervision_sigma002_20261015/selection.json)

## 损失定义

每个入口符号分支保持：

\[
\lambda_k(q)=\lambda_{\mathrm{local},k}(q)\exp\bigl(\delta_k(q)\bigr),
\qquad
T(q)=\sum_k\pi_k\exp\bigl(-H_k(q)\bigr).
\]

其中 `pi_k` 是解析入口符号概率，`q=s/ell` 从介质入口计距。训练目标为：

\[
\mathcal L
=\mathcal L_{\mathrm{conditional\ censored\ NLL}}
+\eta\,\underset{r}{\mathrm{mean}}\sum_k\pi_{r,k}
  \underset{j}{\mathrm{mean}}\delta_{r,k,j}^{2}
+\beta\,\underset{r,i}{\mathrm{mean}}
  \bigl(T_r(q_i)-\widehat T_r(q_i)\bigr)^2.
\]

- **残差 L2**：约束插值到 1024 格后的 log hazard 修正；按入口符号概率加权，各光线等权。这与优化器原有的参数 weight decay 是两项不同的约束。
- **透射率监督**：在 513 个均匀 `q/Q` 查询点计算 MSE，含起点与末端。经验曲线使用该训练光线全部 2048 次参考实现，分母包含逃逸样本，因此末端也监督逃逸概率。
- **原 NLL**：保留入口符号条件下的删失似然，每轮每条光线随机抽 512 个实现。曲线目标不随这一抽样改变，也不读取验证或测试标签。

经验曲线 MSE 与平均 survival Brier loss 相差一个不依赖模型的常数。保留 NLL 可同时约束事件位置及逃逸；新增曲线项直接约束 hazard 累积后得到的透射率。

两项损失复用原 forward 的输出，只在训练时增加计算。MLP 仍为 57,632 个参数、256 点剖面输入、64 个控制值、32 维编码；checkpoint 格式和推理接口保持兼容。新增损失不会增加推理时的网络调用、local 积分或采样步骤。

## 配对实验与选择规则

数据来自 `outputs/fields/shader_ball_full_0.02.nvdb`，实际 sigma≈0.02。使用已有多视角数据：695 条训练光线、158 条验证光线、280 条固定测试光线。训练/验证每条 2048 次参考实现，测试每条 8192 次。本轮没有新增标签。

参考仍为 H=1、B=8、步长 1/64 的 GPIS function-space Tungsten-linear 采样器。数据 SHA-256：

```text
2fead63699474ebcde1d47169e49954994bdab270c75788a31f403f3d1e844ef
```

每组训练 500 epochs，种子为 20261015、20261016、20261017。同种子的初始参数、样本抽取顺序、批量、优化器及学习率调度均与原 residual 配对。原 residual 的三个 checkpoint 直接复用。

权重选择流程在训练前固定：

1. 第一种子比较 `eta ∈ {0.001, 0.01, 0.1}`，此时 beta=0。
2. 固定选出的 eta，再比较 `beta ∈ {10, 100}`。
3. 每个训练按最小验证**边缘距离 NLL**选 checkpoint；候选权重按最差 10% 验证光线的平均最大透射率误差选，要求 NLL 不超过原基线 + 0.005。
4. 先写入并冻结 `selection.json`，再用另外两个种子复核选中的 L2 与 L2＋T。全部选定后评估固定测试集。

共新增 9 组训练。最终验证选择为 **eta=0.01、beta=100**。第一种子的验证最差 10% 误差均值由原 residual 的 0.060420 降到 L2＋T 的 0.056256，约下降 6.9%；两组最终候选均通过 NLL 门槛。

“最差 10%”指每个种子内，按逐光线最大 T 误差排序后取 `ceil(0.1*N)` 条的均值。选择 checkpoint 的指标与选择损失权重的指标不同，不能把最终 epoch 当作所选模型。

## 固定测试结果

下表先对每个种子计算指标，再平均三个种子；Local 为确定性基线。各项均以小为好。“最坏光线”是各个种子最大值的均值。

| 方案 | 平均最大 T 误差 | P95 | 最差 10% 均值 | 最坏光线 | T RMSE | 距离 NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | 0.025339 | 0.107070 | 0.104750 | 0.145240 | 0.018494 | 0.712595 |
| 原 residual | 0.018533 | **0.049025** | 0.065584 | 0.248525 | 0.009479 | 0.701130 |
| residual＋L2 | 0.019124 | 0.052838 | 0.068250 | 0.163305 | 0.009335 | 0.700413 |
| residual＋L2＋T | 0.018605 | 0.054377 | **0.063906** | 0.146387 | **0.008861** | **0.699851** |

仅加 L2 降低了最坏误差，但平均最大误差增加 3.2%，最差 10% 均值增加 4.1%。加入 T 监督后，相对原 residual：

- 最坏光线误差均值下降 41.1%，三个种子各自均下降；种子间标准差由 0.04021 降到 0.01740。
- 最差 10% 均值下降 2.6%，T RMSE 下降 6.5%，逃逸概率 MAE 从 0.002657 降到 0.002510。
- 平均最大误差增加 0.4%，可视为这次比较中基本持平；三个配对种子里只有一个改善这一指标。
- P95 增加 10.9%，部分原本误差较小的光线出现退化。因此不能把最坏误差改善解释为所有尾部指标均改善。

![总体与尾部比较](../outputs/first_hit_residual_supervision_sigma002_20261015/distance_comparison.png)

三个事先固定的失败案例，逐光线最大 T 误差的种子均值为：

| 测试光线索引 | Local | 原 residual | residual＋L2 | residual＋L2＋T |
| --- | ---: | ---: | ---: | ---: |
| 266 | 0.022938 | 0.248525 | 0.142212 | 0.127743 |
| 84 | 0.111995 | 0.133622 | 0.134201 | 0.121089 |
| 175 | 0.094126 | 0.133290 | 0.120159 | 0.121155 |

266 号光线的过度修正明显减轻，但仍显著差于 Local。新的明显退化案例是 15 号光线：原 residual 的 0.039956 增至 L2＋T 的 0.122245，Local 为 0.081641。该案例在评估后选出，未用于调权重。

![新方案的最大误差案例](../outputs/first_hit_residual_supervision_sigma002_20261015/new_worst_rays.png)

本轮支持把 L2＋T 保留为控制极端误差的候选方案，同时保留原 residual 作对照。当前结果还不支持直接将它视为整体更优的替代品，也不能据此判定需要更换网络架构。

原 residual 所选 epochs 为 40、15、20；L2 为 40、20、50；L2＋T 为 80、45、80。三者采用相同 checkpoint 选择规则。全部 9 组新训练及数据、原 checkpoint 身份检查均通过。

## 使用方式

在仓库根目录运行完整流程：

```powershell
python -B python/first_hit_se_cli.py residual-supervision --config python/configs/first_hit_se_residual_supervision_sigma002.json
```

可分别使用 `--phase train`、`--phase evaluate`、`--phase report`。已完成的训练在身份检查后复用，未完成的训练从保存状态继续。协议记录数据、训练源码和原基线 checkpoint 的 SHA-256；改变实现或候选配置后须使用新的输出目录。

控制器在每组训练的内部配置中写入以下字段；直接调用 `distance_study.train_one(..., mode='residual', ...)` 时也可传入：

```json
"distance_objective": {
  "residual_l2_weight": 0.01,
  "transmittance_weight": 100.0
}
```

默认两项为零，精确保留原损失与梯度。正的 `residual_l2_weight` 要求 residual 模型。原 `distance-study` CLI 会遍历 direct、residual、CDF，因此本轮请使用上面的专用 `residual-supervision` 命令，不能直接把正的残差正则权重加到旧 CLI 的混合比较配置中。

新 checkpoint 位于：

```text
outputs/first_hit_residual_supervision_sigma002_20261015/
  variants/l2_0p01_T_0/runs/residual_<seed>/best.pt
  variants/l2_0p01_T_100/runs/residual_<seed>/best.pt
```

使用原 `distance_study.load_run` 加载。`history.json` 记录原 NLL、未乘权重的 L2 与曲线 MSE、总损失和验证结果。

## 验证与边界

20 项数值与回归测试通过，包括默认损失/梯度兼容、解析正则梯度、入口符号权重、逃逸监督、曲线目标与 NLL 抽样相互独立、单次编码与梯度传播，以及验证选择规则。

```powershell
python -B -m unittest python/tests/test_distance_objective.py python/tests/test_distance_design.py python/tests/test_first_hit_hazard.py -v
```

报告生成时额外核对所有训练完成、初始化配对、最优 checkpoint 选择、测试隔离及文件哈希。评估检查透射率有限、单调递减且 T(0)=1。

另将第一种子的 L2、L2＋T checkpoint 通过原加载接口重新载入，在 3 条验证光线、每条 129 个查询点上比较批量和逐光线查询，最大 T 差分别为 2.21e-8、2.83e-8；见 [接口兼容检查](../outputs/first_hit_residual_supervision_sigma002_20261015/inference_compatibility.json)。

固定测试集已经在前轮用于设计诊断，且仍来自同一场景，不能视为新场景盲测。正则是软约束，不保证每条光线均优于 local；整段 MLP 的前缀一致性仍需单独解决。

本轮训练会改变距离编码器。之前的条件梯度头绑定旧编码器，需在新编码器上重新训练后再检查联合采样；本轮没有执行梯度头重训或完整渲染验证。

后续已完成 [轻量逐段残差 GRU 实验](FIRST_HIT_CAUSAL_RESIDUAL_SIGMA002.md)：四个方案、三个种子的对照支持历史状态改善距离精度，并由因果结构保证前缀一致性。GRU＋NLL 的平均最大 T 误差为 0.007633；相对本轮 MLP＋L2＋T 下降 59.0%。但 GRU 更新步长仍敏感，当前 CPU 整曲线查询也更慢，详见后续报告。

实现：[训练损失](../python/first_hit_se/distance_objective.py)、[训练接入](../python/first_hit_se/distance_study.py)、[实验控制](../python/first_hit_se/residual_supervision.py)、[审计与报告](../python/first_hit_se/residual_supervision_report.py)、[测试](../python/tests/test_distance_objective.py)。

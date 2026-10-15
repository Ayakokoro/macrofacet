# SE 相机 first hit：连续内部状态与 GRU 对比

本实现只学习相机光线进入介质后的首个零点：透射率、命中距离和命中处的梯度分布。
入口来自 `activeDomain` 与相机光线的交点，距离从入口重新计量。
不观测或固定入口处的随机场值、梯度；若入口 `F<0`，继续追踪首次上穿零点。
入口没有碰撞概率原子，`T(0)=1`。核固定为平稳 SE，
`rho(q)=exp(-q*q/2)`，`q=s/ell`、`b=m/sigma`。

这是独立的 `first_hit_se` 训练与 C++ 标量推理链路。原 `renewal` GRU、A/B 数据、
既有 checkpoint 和完整路径追踪器仍可使用。新模型支持相机检查工具，以及下文的
`hybrid_first_hit` CPU 标量渲染；首次碰撞后交给独立的 neural renewal 策略。
尚未接入原多次反弹的 GPU wavefront 路径追踪。新模型不能用于表面出发或反弹后的射线。

## 状态、风险与梯度

入口符号 `k=+1,-1` 的概率是 `pi_k=Phi(k*b(0))`。两个分支共享网络，分别接收
`k*b(q)`、`k*b'(q)`，并使用从入口累计的 `log1p(q)`。推理不需要知道实际入口符号。

连续状态分支是：

```text
h_k(0) = E(asinh(k*b(0)), asinh(k*b'(0)))
dh_k/dq = f(h_k, asinh(k*b), asinh(k*b'), log1p(q))
lambda_k = softplus(r(h_k, asinh(k*b), asinh(k*b'), log1p(q)))
dH_k/dq = lambda_k
T(q) = sum_k pi_k * exp(-H_k(q))
```

`h` 默认 16 维，状态与风险 MLP 宽度 64，梯度混合头宽度 128、8 个分量。
风险以无量纲距离为单位；物理风险是 `lambda/ell`。
连续动力学允许局部统计不变时，风险仍随存活历史变化。

GRU 比较分支直接复用 `renewal.model.RenewalNetwork`：5 维段特征、GRU 隐状态、
4 个正 Bernstein 风险系数、正半轴截断 Gaussian mixture。
它也使用上述双符号边缘化，因此比较的是同一个首零点事件。
它没有被改造成连续状态模型，仍然按每个几何段更新一次。

梯度头学习 `W=-k*ell/sigma*dot(omega,g)>0`，即首碰条件下的纵向穿越速度，
不额外乘一次 Rice 的速度权重。已知命中位置时，符号权重为
`pi_k*exp(-H_k)*lambda_k` 归一化，不能直接使用入口权重。
完整梯度利用平稳 SE 的解析条件分布恢复。令 `C=Cov(grad Z)`、`c=C*omega`、
`v=dot(omega,c)`、`mu=grad m(x)`，采样到物理纵向导数 `D=-k*sigma/ell*W` 后：

```text
E[g | D, first hit at x] = mu + c/v * (D-dot(omega,mu))
Cov[g | D, first hit at x] = C - c*c^T/v
```

去除纵向相关部分后的高斯梯度残差与整条直线上的场值过程独立，所以首达历史对梯度的
剩余影响已经包含在所学纵向分布内。这个结论依赖此处的平稳 SE 核。
各向异性核使用 `ell=1/sqrt(omega^T M omega)` 和完整 `C=sigma^2 M`。
保留原始梯度与法线符号；首次上穿不翻转成下降穿越。

## 分段敏感性的处理

均值继续使用 `RayMeanProfile::fromField` 的分片三次多项式；细分时精确限制原多项式。
模型不接收段长或段编号，隐藏状态与累计风险跨体素边界延续。
段长只限制 RK4 数值积分；默认内部最大步长为 `1/32` 个相关长度。
每段内有独立于输出查询位置的积分网格，输出位置通过最后一个部分 RK4 步求值，
不修改后续推进的网格状态。Python 将这些部分步批量计算。

因此连续模型的剩余分段差异来自数值求解与浮点误差，必须随步长细化检查，
不能宣称有限步长下严格相同。原 GRU 另有每段更新次数改变带来的模型差异。
`compare` 同时检查 1/2/4/8 倍精确细分、每段随机选择 1–8 倍细分，
以及 RNO 内部步长减半；报告 `max|ΔT|` 与 `max|Δ(-log T)|`。
不接受会改变原均值曲线的 `point_linear` 切线重建。

碰撞采样先按入口概率抽取分支，再一次性抽取 `E=-log(U)`；沿整个剖面累积该分支的
风险，首次达到 `E` 时求根。跨段不重抽符号或阈值，也不重置隐藏状态。

## 数据与似然

入口模式是 `unconditioned_first_zero`。原参考后端在整条射线上用无条件循环嵌入采样 SE 残差，
没有入口截断或条件化。保存首次零点距离、删失标记、入口符号和穿越速度。
符号只用于分支监督，不作为推理时已知的场实现输入。
精确的均值三次多项式与 GP 残差 Hermite 插值组合后检测首零点；GP 插值仍有离散误差。
训练用较细网格，较粗网格用于审计；网格差异和 Monte Carlo 不确定性单独记录。

固定历史的 [GPIS 函数空间后端](GPIS_FUNCTION_SPACE.md) 使用同一入口和训练接口。
其 GPIS 训练示例默认以 `tungsten_linear` 总场线性求交和割线速度为主要参考，
Hermite 为可选对照；H、采样批大小、步长或求交规则改变时分别采集、训练。
两种后端的数据不能混用，GRU / RNO 结构无需改变。

每条完整标签的训练损失为
`-log(pi_k)+H_k(d)-delta*log(lambda_k(d))-delta*log Q_k(W|d)`。
对于删失标签，入口符号提供额外的离线监督。评估时还报告真正可观测的边缘似然：
删失项为 `-log T(L)`；仅距离命中项为 `-log sum_k pi_k exp(-H_k) lambda_k`。
联合 NLL 指无量纲距离和带符号纵向速度的联合密度，不包含解析横向高斯项或物理单位的
Jacobian 常数。PIT KS 和条件速度 CDF 单独评估所学梯度头。
CDF 图使用参考命中位置和符号作混合权重，以隔离梯度头误差，不能解释为模型自身的
无条件命中速度分布。C++ 采样工具则生成模型自身的联合距离与完整梯度样本。

默认小规模数据包含 12 条真实相机光线，以及用于覆盖入口正负号的 3 个仿射剖面。
训练 11 个剖面 / 5632 条实现，验证 2 个 / 4096 条，测试 2 个 / 8192 条。
划分单位明确设为 `ray`：同一个 shader ball 场景的不同光线，不能据此声称跨场景泛化。
真实相机射线的入口均值很大，因此验证/测试没有观测到负入口；上穿语义由仿射参考和
独立对称性、采样、梯度测试覆盖。后述 v2 用独立仿射验证/测试剖面补充低入口均值；
新的独立场景仍待验证。
默认两种模型用相同种子、相同逐轮子采样、相同优化设置，训练 32 轮，每条训练剖面每轮
抽 128 条已有实现。验证与最终测试使用全部实现。RNO 为 39,545 参数、GRU 为 49,068 参数，
这不是等参数量比较。

扩展数据配置 `python/configs/first_hit_se_camera_dataset_v2.json` 保持同一 SE 核族、
入口定义和首次上/下穿语义，增加相机像素与仿射均值历史。实际训练为 152 个剖面
（120 条相机光线、32 个仿射剖面）/ 155,648 个样本，验证为 25 个 / 25,600 个样本，
测试为 26 个 / 53,248 个样本。一个验证候选像素未进入介质，记录在 manifest 的
`skipped_domain_misses` 中。训练包含 27 个全程删失剖面；仿射训练、验证、测试剖面
各自独立，也补充负入口首次上穿的标签。原来的两个测试像素继续保持留出，旧数据及权重保留。
这仍然是同场景按光线划分的实验，不能据此推断跨场景泛化能力。

`first_hit_se_{rno,gru}_h16_v2.json` 从各自 v1 权重继续训练，重置优化器，保持网络尺寸。
两者采用相同随机顺序与每剖面每轮 256 个标签、每批最多 32 个剖面的采样规则，
最多 384 轮，至少 128 轮；验证联合 NLL 连续 40 轮未累计改善 0.001 时早停。
AdamW 初始学习率为 0.001，验证指标停滞时逐级减半，最低 0.00003。
checkpoint 按实际最低验证 NLL 保存，最终测试在选定权重后进行。

`batch.py` 只缓存几何及各 RK4 阶段的输入，多条光线保留各自的原始积分网格。
隐藏状态与梯度不在步间截断，查询位置通过原来的部分 RK4 步求值。
CUDA graph 复用同一个前向/反向计算图，补齐部分使用零步长；它不改变模型或导出格式。
测试比较了逐光线与批量结果、参数梯度、连续优化更新，以及中断恢复后的权重和学习率状态。
CPU、非捕获模式仍可使用，v1 配置的默认行为保持兼容。

## 运行

从仓库根目录运行，依赖沿用 `python/requirements.txt`。
相机采集器需要 `MACROFACET_BUILD_FIELDS=ON`、完整域 NanoVDB 和配置中的场资源。
通用 `macrofacet_query_first_hit_se` 不依赖 NanoVDB。

```powershell
cmake --build build --config Release --target macrofacet_experiments macrofacet_inspect_first_hit_se macrofacet_query_first_hit_se macrofacet_tests --parallel 4
python python/first_hit_se_cli.py collect --config python/configs/first_hit_se_camera_dataset.json
python python/first_hit_se_cli.py audit-reference --manifest outputs/first_hit_se_camera_dataset_v1/manifest.json --output outputs/first_hit_se_camera_dataset_v1/reference_audit.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_rno_h16.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_gru_h16.json
python python/first_hit_se_cli.py compare --checkpoint outputs/models/first_hit_se_rno_h16_v1/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v1/best.pt --dataset outputs/first_hit_se_camera_dataset_v1/dataset.pt --output outputs/first_hit_se_comparison_20261013
```

扩展训练及前后对比：

```powershell
python python/first_hit_se_cli.py collect --config python/configs/first_hit_se_camera_dataset_v2.json
python python/first_hit_se_cli.py audit-reference --manifest outputs/first_hit_se_camera_dataset_v2/manifest.json --output outputs/first_hit_se_camera_dataset_v2/reference_audit.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_rno_h16_v2.json
python python/first_hit_se_cli.py train --config python/configs/first_hit_se_gru_h16_v2.json
python python/first_hit_se_cli.py compare --checkpoint outputs/models/first_hit_se_rno_h16_v1/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v1/best.pt --checkpoint outputs/models/first_hit_se_rno_h16_v2/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v2/best.pt --dataset outputs/first_hit_se_camera_dataset_v2/dataset.pt --output outputs/first_hit_se_comparison_v2_20261013
python python/first_hit_se_cli.py compare --checkpoint outputs/models/first_hit_se_rno_h16_v1/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v1/best.pt --checkpoint outputs/models/first_hit_se_rno_h16_v2/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v2/best.pt --dataset outputs/first_hit_se_camera_dataset_v1/dataset.pt --output outputs/first_hit_se_regression_v2_20261013
```

训练中断可给同一命令加 `--resume`；配置及数据哈希必须完全相同。
已有训练目录需恢复或改用新目录，不会静默覆盖一轮训练。
数据生成缓存检查输入和 C++ 可执行文件哈希；重新编译参考生成器后会重新采集。
`prepare` 会拒绝原 A/B 标签、不匹配的核、重复剖面和不完整/单位不一致的记录。

```powershell
python python/first_hit_se_cli.py export-model --checkpoint outputs/models/first_hit_se_rno_h16_v1/best.pt --output outputs/models/first_hit_se_rno_h16_v1/first_hit_model.json
python python/first_hit_se_cli.py export-model --checkpoint outputs/models/first_hit_se_gru_h16_v1/best.pt --output outputs/models/first_hit_se_gru_h16_v1/first_hit_model.json
python python/first_hit_se_cli.py verify-cpp --checkpoint outputs/models/first_hit_se_rno_h16_v1/best.pt --dataset outputs/first_hit_se_camera_dataset_v1/dataset.pt --output outputs/first_hit_se_cpp_rno_20261013
python python/first_hit_se_cli.py verify-cpp --checkpoint outputs/models/first_hit_se_gru_h16_v1/best.pt --dataset outputs/first_hit_se_camera_dataset_v1/dataset.pt --output outputs/first_hit_se_cpp_gru_20261013
build/Release/macrofacet_inspect_first_hit_se.exe configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json outputs/first_hit_se_camera_rno_20261013 368 368 outputs/models/first_hit_se_rno_h16_v1/first_hit_model.json 512
```

上面的旧渲染配置只提供场、相机和 SE 参数；检查器始终提取 cubic，不读取旧 GRU 权重。
输出包含入口 `ray.json`、`mean_profile.json`、可交给 `first-passage` 的 `reference.json`，
以及模型的 `neural_curve.csv`、`predictions.json`、`samples.csv`、`sample_summary.json`。
其中距离是从入口算起的物理距离，`samples.csv` 同时给出 `q,W,k,g,n`。

`compare` 生成 `comparison.html`、曲线 CSV、条件速度 CDF 和完整 JSON 指标。
`verify-cpp` 检查两个分支的累计风险、风险、命中分支概率、混合参数，以及抽样距离的 CDF。
训练/对比是研究用 Python 循环和标量 C++ 后端，尚未为整帧渲染性能优化。

```powershell
python -m unittest discover -s python/tests -v
ctest --test-dir build -C Release --output-on-failure
```

关键源码：`python/first_hit_se/model.py`、`data.py`、`train.py`、`evaluate.py`，
`src/learned/FirstHitSEModel.cpp`、`src/experiments/FirstPassageExperiment.cpp`，
以及 `src/transport/RenewalMedium.cpp` 中的 `sampleFirstHitSEGradient`。

## 首轮结果（2026-10-13）

两种模型均完成 32 轮，验证集选择第 32 轮。下面仅涵盖同场景留出的两条测试光线
（像素 `(368,368)` 和 `(32,32)`，每条 4096 个 GP 实现）。
训练数据量小、拟合尚未收敛，不能把分段一致性改善解释为拟合精度已经足够。

| 指标 | RNO | GRU |
| --- | ---: | ---: |
| 测试联合 NLL | 2.00334 | 1.76645 |
| 测试距离 NLL | 1.54207 | 1.30073 |
| 各射线最大透射率拟合误差的平均 | 0.51522 | 0.39206 |
| 8 倍精确细分：两条射线最大 `abs(ΔT)` | 2.02e-7 | 0.11850 |
| 随机细分：两条射线最大 `abs(ΔT)` | 1.18e-7 | 0.07553 |
| 命中速度 PIT KS | 0.05662 | 0.06441 |

RNO 内部步长减半的最大透射率变化为 `8.08e-8`，这一组结果已接近 float32 状态的舍入误差，
不要求误差随每次细化严格单调下降。实际训练权重的 Python/C++ 检查包含累计风险、
风险、命中分支权重与全部混合参数，所有最大绝对差小于 `5.6e-7`。
每个模型、每条测试光线另检查 512 个 C++ 距离样本与预测 CDF；相机检查器的完整梯度
满足所抽取纵向导数，最大投影误差为 `6.7e-16`。全部三组 CTest 通过，神经测试共 31 项。

[对比图](../outputs/first_hit_se_comparison_20261013/comparison.html)、
[完整指标](../outputs/first_hit_se_comparison_20261013/comparison_report.json)、
[RNO 推理检查](../outputs/first_hit_se_cpp_rno_20261013/verification.json)、
[GRU 推理检查](../outputs/first_hit_se_cpp_gru_20261013/verification.json)保留了本轮产物。
`outputs/` 按仓库规则不纳入 Git，命令和配置已保存在源码中。


## 扩展训练结果（2026-10-13）

v2 保持 h16 及同一损失形式。RNO 完成 133 轮 / 665 次新增参数更新，选择第 130 轮；
GRU 完成 151 轮 / 755 次新增参数更新，选择第 144 轮。两者均触发预设的验证平台早停，
最终学习率分别为 0.00003 和 0.00003125。RNO 最后 40 轮验证 NLL 的全范围只有 0.000614。
这些统计描述当前数据与训练规则下的平台，不证明所有剖面均已达到最佳可实现误差。

完整新测试集包含 26 个剖面 / 53,248 个样本，所有旧、新权重使用同一份测试标签：

| 指标（完整新测试集） | RNO v1 | RNO v2 | GRU v1 | GRU v2 |
| --- | ---: | ---: | ---: | ---: |
| 各剖面最大透射率误差的平均 | 0.505422 | 0.026978 | 0.458678 | 0.018984 |
| 各剖面透射率 RMSE 的平均 | 0.296322 | 0.012701 | 0.284296 | 0.007058 |
| 联合 NLL | 2.355296 | 1.268093 | 2.176663 | 1.265143 |
| 命中速度 PIT KS | 0.018285 | 0.008515 | 0.027968 | 0.007627 |
| 8 倍细分最大 `abs(ΔT)` | 3.87e-7 | 1.49e-5 | 0.182459 | 0.314384 |

RNO v2 在所有新测试剖面上的随机细分最大 `abs(ΔT)` 为 1.56e-5，
内部积分步长减半后的最大变化为 1.37e-5。完整测试集比原来的两个测试像素包含更多困难情况，
不能把原两条光线的 1e-7 量级分段差异外推到整个新测试集。
对最敏感像素 `(296,336)` 再单独细化积分步长，1/32、1/64、1/128 下的
8 倍细分最大差异依次为 1.49e-5、1.14e-6、6.01e-7，确认主要是积分误差。
训练与默认导出仍使用 1/32，以上验收统计也保持这一默认步长。

原先两条测试光线与原先 4096 条/光线的参考完全保留，前后结果如下：

| 指标（原两条测试光线） | RNO v1 | RNO v2 | GRU v1 | GRU v2 |
| --- | ---: | ---: | ---: | ---: |
| 各光线最大透射率误差的平均 | 0.515223 | 0.024736 | 0.392063 | 0.019799 |
| 联合 NLL | 2.003340 | 1.135543 | 1.766454 | 1.137216 |
| 8 倍细分最大 `abs(ΔT)` | 2.02e-7 | 1.80e-7 | 0.118498 | 0.255660 |

像素 `(32,32)` 的参考终点透射率为 1，RNO 从 0.441737 提高到 0.999919。
像素 `(368,368)` 的参考终点透射率为 0.206787，RNO 从 0.021982 提高到 0.163234，
GRU v2 为 0.212605。RNO 在这一条光线的尾部仍偏低；不能用平均指标掩盖这一残余误差。
该轮同时改变了数据量、历史覆盖范围及优化预算，没有分别量化三者各自的贡献。

203 个参考剖面的 1/64 与 1/128 网格比较均落在当前 Monte Carlo 不确定性范围内，
最大曲线差为 0.088867；这不排除剩余连续路径离散偏差。
两个 v2 模型分别在三个测试剖面上通过 Python/C++ 参数与概率检查，最大绝对差
分别为 2.41e-6、3.56e-6；每模型/每剖面另检查 512 个 C++ 碰撞样本。
完整相机梯度样本满足所抽取纵向导数，最大投影误差为 6.7e-16。

[训练汇总与学习曲线](../outputs/first_hit_se_training_v2_20261013/summary.html)、
[全部新测试剖面对比](../outputs/first_hit_se_comparison_v2_20261013/comparison.html)、
[原测试光线回归](../outputs/first_hit_se_regression_v2_20261013/comparison.html) 分别保留完整结果。

## 直接使用渲染配置中的场与相机比较（σ=0.02）

`configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16.json` 可以直接交给
`macrofacet_inspect_first_hit_se`。检查器加载 VDB、SE 核、相机、分辨率和随机种子，
从该场的 `activeDomain` 入口构建沿线均值；first-hit 权重由单独的命令行参数提供。
它不调用 `prepareRenewalModel`，所以配置中旧的 `transport.renewal.model` 不参与比较。
统计比较不要求接入完整路径追踪器；RGB 图像比较才需要将模型接入相机首次碰撞流程。
当前模型不覆盖反弹后的航段，格式也不是旧渲染器的 `macrofacet.renewal`，不能只替换旧模型路径。

该配置实际使用 `shader_ball_full_0.02.nvdb`，文件内 σ≈0.02，SE 相关长度约 0.05656854。
比较使用 VDB 三线性场的精确分片三次均值，配置中的旧 `point_linear` 近似未用于此参考。
每个指定像素取与渲染器相同种子/抖动规则下的第一条相机光线，不是整个 128 spp 像素的平均。
以下独立配置保留所选的 8 个像素，每条使用 4096 个 GP 样本，参考网格为 1/64、1/128：

```powershell
python python/first_hit_se_cli.py collect --config python/configs/first_hit_se_render_scene_sigma002_comparison.json
python python/first_hit_se_cli.py compare --checkpoint outputs/models/first_hit_se_rno_h16_v2/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v2/best.pt --dataset outputs/first_hit_se_render_scene_sigma002_dataset_20261013/dataset.pt --output outputs/first_hit_se_render_scene_sigma002_comparison_20261013 --subdivisions 1 8
```

固定 v2 权重的结果如下，原渲染 JSON 与 checkpoint 哈希均确认未变：

| 指标（8 条光线 / 32,768 个样本） | RNO v2 | GRU v2 |
| --- | ---: | ---: |
| 各光线最大透射率误差的平均 | 0.089222 | 0.075724 |
| 各光线透射率 RMSE 的平均 | 0.044291 | 0.019241 |
| 命中速度 PIT KS | 0.084531 | 0.011254 |
| 8 倍细分最大 `abs(ΔT)` | 1.71e-6 | 0.324426 |

像素 `(368,368)` 的终点透射率为 GP 0.025391、RNO 0.310719、GRU 0.029531。
本次光线最长为 21.53 个相关长度，原训练集最长为 10.76；输入历史已超出训练长度范围。
RNO 在此像素与 `(376,96)` 上有明显残余偏差。内部步长减半对所有 8 条光线的最大
透射率影响只有 5.32e-7；这一大偏差不能归于积分误差或尚未接入渲染器。
这次没有重新训练，结果是改变场参数和归一化历史后的泛化评估。

粗细 GP 网格的最大曲线差为 0.026611，均在当前 Monte Carlo 不确定性范围内。
实际 JSON 的 C++ 检查器与 Python 在 `(368,368)` 上的最大透射率差为 1.56e-7。
[场景相机光线对比](../outputs/first_hit_se_render_scene_sigma002_comparison_20261013/comparison.html)
包含像素标注、完整梯度速度 CDF、逐光线统计和文件哈希。

## 使用 shader_ball_full_0.05.nvdb 复测同一组相机光线

使用已有 `configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json`，
场文件为 `outputs/fields/shader_ball_full_0.05.nvdb`。SE 核、roughness=0.5 对应
σ≈0.05、ℓ≈0.14142136。相机和 8 个像素与上一轮相同，入口按新 VDB 的
`activeDomain` 重新计算；本组最大归一化长度为 10.183338。沿用固定的 RNO/GRU v2
权重，没有重新训练或修改渲染配置。与 σ=0.02 场景相比，相关长度及入口也发生变化，
不能把结果差异单独归因为 σ。

```powershell
python python/first_hit_se_cli.py collect --config python/configs/first_hit_se_render_scene_sigma005_comparison.json
python python/first_hit_se_cli.py audit-reference --manifest outputs/first_hit_se_render_scene_sigma005_dataset_20261013/manifest.json --output outputs/first_hit_se_render_scene_sigma005_comparison_20261013/reference_audit.json
python python/first_hit_se_cli.py compare --checkpoint outputs/models/first_hit_se_rno_h16_v2/best.pt --checkpoint outputs/models/first_hit_se_gru_h16_v2/best.pt --dataset outputs/first_hit_se_render_scene_sigma005_dataset_20261013/dataset.pt --output outputs/first_hit_se_render_scene_sigma005_comparison_20261013 --subdivisions 1 8
```

每条光线在两个参考网格上各采样 4096 个 GP 实现，指标使用最细网格的 32,768 个样本。
本组包含 4 条原训练光线、1 条原验证光线和 3 条原测试光线。均值特征和完整梯度几何
已与原数据逐一核对相同。新的 GP 实现不改变这些物理剖面的原划分，因此整组指标
不应称为独立测试误差；原测试子集为 `(32,32)`、`(368,368)`、`(376,96)`。

| 指标 | RNO v2 | GRU v2 |
| --- | ---: | ---: |
| 全部 8 条：平均最大透射率误差 | 0.035443 | 0.020799 |
| 全部 8 条：平均透射率 RMSE | 0.017211 | 0.007401 |
| 全部 8 条：命中速度 PIT KS | 0.013692 | 0.007503 |
| 全部 8 条：8 倍细分最大 `abs(ΔT)` | 1.93e-5 | 0.306052 |
| 原测试 3 条：平均最大透射率误差 | 0.044804 | 0.020254 |
| 原测试 3 条：平均透射率 RMSE | 0.027550 | 0.008402 |

像素 `(368,368)` 的终点透射率为 GP 0.213867、RNO 0.163234、GRU 0.212605；
`(376,96)` 为 GP 0.519043、RNO 0.591018、GRU 0.490899。RNO 的这两条残余
曲线偏差远大于原测试子集的积分步长减半差异（最大 7.38e-8），不是分段误差。
全部 8 条中的内部步长减半最大差异为 1.84e-5。

粗细 GP 网格的最大 CDF 差为 0.023926，8 条均在当前 DKW 和界内；这不排除剩余
连续路径离散偏差。使用已有同场景、同权重的 C++ 探针对照，`(368,368)` 上
Python/C++ 最大透射率差为 6.40e-8。梯度图固定参考首碰位置与入口符号检查纵向
速度头，不是按模型自己的首碰位置混合的无条件梯度分布。

[σ=0.05 比较页](../outputs/first_hit_se_render_scene_sigma005_comparison_20261013/comparison.html)
包含逐像素透射率、梯度头、分段检查及原划分；
[场景汇总](../outputs/first_hit_se_render_scene_sigma005_comparison_20261013/scene_summary.json)
和 [来源与哈希](../outputs/first_hit_se_render_scene_sigma005_comparison_20261013/scene_provenance.json)
记录权重、场文件和原始数据的对应关系。

## 曲线损失和状态维数对照

`python/configs/first_hit_se_ablation_v3.json` 定义四组、三个配对种子的继续训练实验：

| 组别 | 历史状态维数 | 训练目标 |
| --- | ---: | --- |
| h16_nll | 16 | 原联合似然 |
| h16_curve | 16 | 原联合似然 + `10 * mean((T - Tref)^2)` |
| h32_nll | 32 | 原联合似然 |
| h64_nll | 64 | 原联合似然 |

数据与划分完全沿用 v2；每次从同一个 h16 v2 checkpoint 开始。扩展状态时保留原坐标，
新增坐标使用随机的编码器和动态输出，并将其到原动态与两个输出头的连接初始化为零。
因此初始化时函数保持不变，而新增坐标到输出的连接仍能获得梯度。这是现有模型附近的
容量扩展对照，不等价于从头训练不同维数的全局容量比较。

三个种子为 20261021、20261022、20261023。每次固定 96 轮 / 480 次更新，
批量 32 条光线、每条每轮 256 个事件样本；相同种子使用相同的样本与批次顺序。
学习率按相同余弦日程从 3e-4 下降至 3e-5。主要结果固定使用 `last.pt`，
按验证联合 NLL 选择的 `best.pt` 另存。没有根据测试结果修改配置或选择展示种子。

曲线监督在 33 个均匀查询点（含端点）使用每条训练光线全部 1024 个 GP 标签，
不使用当轮仅抽取的 256 个事件标签重估曲线。曲线查询和事件查询共用同一次批量积分，
允许 CUDA graph 下共同反向传播，避免在 backward 前再次 replay 覆盖积分状态。

```powershell
python python/first_hit_se_cli.py ablation --config python/configs/first_hit_se_ablation_v3.json
python python/first_hit_se_cli.py ablation-evaluate --config python/configs/first_hit_se_ablation_v3.json
python python/first_hit_se_cli.py ablation-audit --config python/configs/first_hit_se_ablation_v3.json
python python/first_hit_se_cli.py ablation-report --config python/configs/first_hit_se_ablation_v3.json
```

`ablation-evaluate --watch` 可以在另一进程中依次评估已完成的训练。
重跑 `ablation` 会核对配置与数据哈希，跳过已完成训练并恢复未完成 checkpoint。
生成曲线报告需要 Matplotlib；本轮安装在输出目录的 `plot_dependencies`，未修改训练环境。
比较页和 SVG 图本身可以直接打开，无需绘图库。

完整训练、验证、测试曲线和似然分别统计；相机场景 8 条光线中原训练/验证/测试划分继续保留。
报告提供三个种子的均值、样本标准差以及相对同种子 h16 原损失组的差值。
曲线展示种子预先固定为 20261022，数值检查覆盖四条指定光线，并核对各代表模型的
三个剖面、每剖面 512 个 C++ 距离样本。

[对照比较页](../outputs/first_hit_se_ablation_20261013/comparison.html)、
[实验定义与初始化检查](../outputs/first_hit_se_ablation_20261013/experiment.json)、
[逐种子及汇总指标](../outputs/first_hit_se_ablation_20261013/summary.json) 保存本轮产物。

12 次训练均完成固定的 480 次更新。以下为三个种子的均值 ± 样本标准差，主要结果
全部使用最后一轮权重；相对改善以同预算的 h16_nll 为基准：

| 组别 | 原测试 26 条：平均最大 T 误差 | 相对改善 | 场景原测试 3 条：平均最大 T 误差 |
| --- | ---: | ---: | ---: |
| h16_nll | 0.025736 ± 0.000250 | — | 0.046094 ± 0.000552 |
| h16_curve | 0.024107 ± 0.000179 | 6.3% | 0.040687 ± 0.000602 |
| h32_nll | 0.025084 ± 0.000272 | 2.5% | 0.045393 ± 0.001366 |
| h64_nll | 0.023647 ± 0.000247 | 8.1% | 0.042233 ± 0.001240 |

各改动在完整测试集上的改善方向三个种子均一致。场景 3 条原测试光线上，曲线损失
平均降低误差 11.7%，h64 降低 8.4%；h32 的收益较小且有一个种子略微变差。
四组测试命中速度 NLL 的均值分别为 1.212362、1.212392、1.212388、1.212400，
没有出现大的梯度头退化。原 RNO v2 的完整测试曲线误差为 0.026978，原 GRU v2 为
0.018984；它们是保留的旧权重参考，不属于这次固定继续训练预算的四组。

训练目标与状态容量都能带来收益，但本次改动均未消除困难像素的尾部偏差。
由于没有新增训练剖面，也没有从头训练更大模型，本轮不能单独估计数据覆盖的贡献，
或判定 RNO 的最终容量上限。三个种子的标准差不包含参考 Monte Carlo 和场景变化的
不确定性，不能视为总体泛化误差的置信区间。

代表种子的四组模型均通过数值检查：四条指定光线上的 8 倍细分最大透射率变化为
2.20e-5，内部积分步长减半最大变化为 2.09e-5；C++/Python 最大预测差（含分布参数）
为 4.03e-6。三组 CTest 通过，新增的隐藏状态扩展、联合曲线梯度、CUDA replay 和
训练恢复检查通过。训练数据、原 RNO/GRU 权重和本轮核心训练实现的哈希均核对未变；
新增状态到风险头的连接在训练后均已学习为非零。

## 2026-10-13：冻结主干，仅训练梯度头

`python/configs/first_hit_se_gradient_head_v1.json` 使用上一轮固定代表种子的
`h16_nll_s20261022/last.pt` 作为共同起点。固定 `initial_encoder`、`dynamics`、
`hazard_head`，仅更新整个 `mixture_head`；分别比较 8、16、32 个正半轴截断高斯分量，
每组使用三个配对随机种子。梯度头的两个隐藏层仍为宽度 128。

增加分量时复制原分量并等分权重；加入最大 0.002、每个原分量内均值为零的均值偏移，
打破重复分量的对称性。因此初始密度只有很小的扰动，初始主干和透射率完全一致。
报告记录实际初始训练 NLL 和 CDF 变化；这属于从既有分布继续优化的对照，
不能视为不同分量数从头训练后的全局最优比较。

冻结主干后，在原始参考首碰位置缓存梯度头输入和带符号的均值导数。仅命中样本有
梯度监督，删失样本不参与该损失；目标是每次首碰的条件速度 NLL。每轮使用全部
73,280 个训练首碰样本，批量 4096，训练 256 轮，每组 4608 次更新。余弦学习率
从 3e-4 降至 3e-5，主结果固定取最后一轮；每四轮及第一轮按验证 NLL 保存最优权重，
报告另列验证集选择的结果。配对种子使用相同的样本顺序，未增加 GP 训练数据。

```powershell
python python/first_hit_se_cli.py gradient-head --config python/configs/first_hit_se_gradient_head_v1.json
python python/first_hit_se_cli.py gradient-head-report --config python/configs/first_hit_se_gradient_head_v1.json
python python/tests/test_first_hit_gradient_head.py
```

命令可跳过已完成模型，或从最近保存的检查点恢复，校验配置、数据和起点权重。
缓存求值与完整前向输出、梯度逐项核对；训练期间校验冻结参数哈希，结束后再检查
8 条场景光线上每条 129 个位置的透射率、累计风险和碰撞密度完全一致。

评估包括完整训练/验证/测试集的条件速度 NLL、整体和逐光线 PIT、按相对碰撞距离
分组的 PIT，以及 W≤0.1/0.25/0.5 和 W≤2/3/4 的概率校准。场景全部 8 条光线与其中
原测试集的 3 条分别统计。所有指标使用参考首碰位置和入口符号，横向梯度解析分布
保持不变；没有把这些指标解释为模型自采样碰撞位置后的完整法线分布质量。

[梯度头对照页](../outputs/first_hit_se_gradient_head_20261013/comparison.html)、
[汇总数据](../outputs/first_hit_se_gradient_head_20261013/summary.json)、
[缓存与来源校验](../outputs/first_hit_se_gradient_head_20261013/cache_manifest.json)
保存本轮产物。

9 组训练全部完成。下表为三个种子的均值，主结果固定取第 256 轮；共同起点是
此前继续训练过的 h16 模型，不是最初的 RNO v2 权重。

| 梯度头 | 训练 NLL | 独立测试 NLL | 独立测试 PIT KS |
| --- | ---: | ---: | ---: |
| 共同起点，K=8 | 1.196853 | 1.212358 | 0.005793 |
| 仅训练梯度头，K=8 | 1.192910 | 1.214034 | 0.008304 |
| 仅训练梯度头，K=16 | 1.192694 | 1.213937 | 0.008717 |
| 仅训练梯度头，K=32 | 1.192454 | 1.214042 | 0.008580 |

K=16 相对同预算 K=8 的测试 NLL 平均降低 9.67e-5，三个种子方向一致，但 PIT KS
三个种子均略差；K=32 的测试 NLL 没有进一步改善。三组训练和验证 NLL 均降低，
测试 NLL 却都高于共同起点。按验证 NLL 选择检查点后，测试 NLL 均值分别为
1.213979、1.213845、1.214061，也未优于共同起点。

σ=0.05 场景原测试 3 条光线的速度 NLL 从共同起点的 1.135345，分别变为
1.140399、1.140486、1.141929；对应 PIT KS 从 0.025738 变为
0.031553、0.031747、0.031386。该结果不支持在当前冻结主干和继续训练设置下，
把梯度误差主要归因于只有 8 个混合分量；数据覆盖、固定状态表示和其他初始化
仍未被这组对照单独区分，不应据此断言某一个因素是唯一原因。

全部模型冻结参数逐项完全相等，8 条场景光线各 129 个位置的 T、H、log hazard、
log density 最大变化均为 0。三个代表模型的 C++ 检查各覆盖 3 个剖面及每剖面
512 个距离样本，最大参数预测差 2.39e-6。新增缓存梯度、分量拆分和断点续训测试
及包含 C++ 接口的整组神经网络 CTest 通过；原 RNO/GRU 权重和训练数据哈希未变。
原模型保持可用，本次权重保存在独立目录，没有接入原渲染配置。

## 2026-10-13：训练光线覆盖与每条光线 GP 样本量对照

`python/configs/first_hit_se_data_ablation_v1.json` 保持上述共同起点、h16 状态、
K=8 梯度头、固定 SE 核和无条件 first-zero 定义，比较两个数据扩充方向。
相机来自 σ=0.05 的同一个 shader-ball 场景，距离仍从介质入口计量。

| 组别 | 相机训练光线 | 每条 GP 实现 | 相机实现总数 |
| --- | ---: | ---: | ---: |
| 重新训练的基线 | 120 | 1024 | 122,880 |
| 增加光线覆盖 | 480 | 1024 | 491,520 |
| 增加每条光线采样 | 120 | 4096 | 491,520 |

覆盖组保留原 120 条光线及全部标签，再新增 360 条光线；采样组保留每条原有的
1024 次实现，再增加 3072 次独立实现。32 条仿射剖面及其标签、原验证集与测试集
完全不变。新增像素按预设种子均匀随机排列，排除原数据所有划分使用的像素，只按
是否与介质有效域相交决定接受，不根据模型误差或 GP 标签选择。

每组仅更新梯度头，使用三个配对种子、4608 次 AdamW 更新、批量 4096、相同余弦
学习率 3e-4→3e-5。为避免扩大相机数据同时降低仿射数据的训练权重，按原训练集
首碰标签数确定两类的批内比例，并分别有放回采样。每个相同种子、相同更新步的
仿射样本序列完全相同。基线也用这一采样器重新训练，因此应优先与本轮基线比较。
采样器对训练种子、更新编号和样本类别组成的元组做 BLAKE2b 哈希，再生成各自的
随机流，避免相邻训练种子只造成批次序列平移；回归测试覆盖了这一情况。

主结果预先固定为最后一次更新；按原验证集 NLL 选出的权重另列。所有训练、验证、
测试指标均在相同的原参考标签上计算，扩充数据自身的训练损失另外保存。三种数据
设置共用一次固定扩充，三个种子只反映优化随机性，不代表新数据和场景的抽样误差。

```powershell
python python/first_hit_se_cli.py data-ablation --config python/configs/first_hit_se_data_ablation_v1.json
python python/first_hit_se_cli.py data-ablation --config python/configs/first_hit_se_data_ablation_v1.json --phase report
python python/tests/test_first_hit_data_ablation.py
```

采集、缓存和训练都支持续跑并核对来源；`--phase collect/train/report` 可以单独执行
各阶段。新增数据仍保留两档 GP 网格参考，细网格实现的随机种子也与原训练、验证、
测试样本一并检查。两档独立 Monte Carlo CDF 之差包含抽样噪声，不能解释为连续路径
离散误差的严格上界。

本轮检验的是增加数据能否改善**固定状态表示下的条件梯度头**。它不检验重新训练
状态演化后的潜在收益，也不改变透射率。横向梯度继续采用固定 SE 的解析条件分布。

9 组训练均完成 4608 次更新，下表为三个种子的最终权重均值。NLL 和 PIT KS 越低
越好；完整测试集包含 18 条相机光线和 8 条仿射剖面，因此另外列出相机子集。

| 组别 | 完整测试 NLL | 完整测试 PIT KS | 相机测试 NLL | 相机逐光线 PIT KS |
| --- | ---: | ---: | ---: | ---: |
| 共同起点，未单独微调 | 1.212358 | 0.005793 | 1.226618 | 0.033346 |
| 原数据 120 × 1024，重训基线 | 1.213884 | 0.010472 | 1.227540 | 0.042060 |
| 增加光线 480 × 1024 | 1.212844 | 0.005097 | 1.226895 | 0.040719 |
| 增加每条采样 120 × 4096 | 1.213514 | 0.011423 | 1.226597 | 0.039962 |

相对同预算基线，两组完整测试 NLL 均在三个配对种子上改善，平均差值分别为
-0.001039 和 -0.000370。增加光线的完整测试 PIT KS 平均降低 51.3%，三个种子
均改善；增加每条采样的完整测试 PIT KS 均值反而略高，仅一个种子改善。

相机子集的结论不同：增加每条采样的 NLL 和逐光线 PIT KS 三个种子均改善，
逐光线 PIT KS 均值降低 5.0%；增加光线对应均值降低 3.2%，两个种子改善。
增加每条采样使相机测试 NLL 更低，但仿射测试 NLL 略变差，故不能用混合后的
完整测试指标替代相机效果。

σ=0.05 场景原测试三条光线中只有两条有首碰标签：基线、增加光线、增加每条采样的
条件速度 NLL 分别为 1.139522、1.138185、1.138693；PIT KS 分别为
0.027715、0.035523、0.022941。增加光线在这组的 PIT 三个种子均变差，增加每条
采样三个种子均改善。场景小子集与完整测试的排序不同，应同时保留这两项证据。

加数据减轻了单独继续训练梯度头的部分退化，但没有全面超过共同起点：后者的
完整测试 NLL 和相机逐光线 PIT KS 仍优于两个扩充组。按验证 NLL 选择权重后，
两个扩充组的完整测试 NLL 为 1.212949、1.213198，也都高于共同起点；相机逐光线
PIT KS 为 0.040298、0.037741，仍高于共同起点的 0.033346。本轮不能据此判定
剩余误差只来自数据量、混合分量数或冻结状态表示中的某一个因素。

全部数据嵌套、原标签保留和冻结状态输入逐项检查通过；新增数据中的原输入与
原缓存最大差值为 0。原及新增细网格 GP 路径共 971,776 条，记录种子无重复。
九个模型的主干参数逐项相等，场景上的 T、H、log hazard、log density 最大变化
均为 0。三个代表模型各检查三个剖面及每剖面 512 个 C++ 距离样本，最大预测差
为 2.33e-6。整组神经网络 Python／C++ CTest 通过。

训练检查中发现旧草稿的相邻优化种子只使批次序列平移，已改为元组哈希并完整
重跑九组。草稿保存在 `superseded_additive_seed_draft`，不计入以上结果；修正记录
和 27,648 个优化随机流种子的唯一性检查均随报告保存。

[数据量对照页](../outputs/first_hit_se_data_ablation_20261013/comparison.html)、
[完整统计](../outputs/first_hit_se_data_ablation_20261013/summary.json)、
[数据与模型校验](../outputs/first_hit_se_data_ablation_20261013/validation.json)
保存本轮结果。原 RNO／GRU 权重与渲染配置保持原样，实验模型保存在独立目录。

## 2026-10-13：h16 状态网络与输出头联合训练

`python/configs/first_hit_se_joint_h16_v1.json` 使用上一轮每条光线增加 GP 采样的
数据集：120 条相机光线各 4096 次实现，另有原 32 条仿射剖面各 1024 次实现。
共同起点仍为 `h16_nll_s20261022/last.pt`，状态维数 16、混合分量数 8。

本轮重跑同预算冻结主干对照。两组按相同种子、相同批次训练 192 轮（960 次更新），
每轮遍历所有 152 条训练剖面，每条无放回抽取 256 条轨迹，每批最多 32 条剖面。
随机流由训练种子和轮数的 BLAKE2b 哈希确定；保存实际剖面及轨迹索引的累计哈希，
用于核对配对训练和断点恢复。三个种子共六组训练，不新增 GP 标签。

两组均使用原首达距离与梯度的联合负对数似然，按全部轨迹数归一化；删失样本
保留距离监督。冻结组只更新 `mixture_head`，缓存其参考首碰输入和恒定的距离
损失；联合组重新计算连续状态并更新 `initial_encoder`、`dynamics`、`hazard_head`
与 `mixture_head`。缓存损失及其梯度与完整前向的一致性有独立测试。

训练的 `complete_joint` 使用离线 GP 实现中的入口符号作为分支标签，包含删失
轨迹的符号。相机推理仍按入口高斯概率混合正负两个分支，不要求已知入口符号。
验证选择沿用 `joint_nll`，对删失轨迹的入口分支边缘化；两个训练模式口径相同。

学习率使用相同余弦日程 3e-4→3e-5，每八轮及首轮、末轮评估。主要结果固定为
第 192 轮，另列按原验证集联合 NLL 选择的权重。原训练／验证／测试标签以及
场景八条光线的标签用于统一评估。联合模型的状态缓存重新计算，不复用旧模型
的隐藏状态。条件梯度指标与透射率误差同时报告，按相机子集和原场景划分分开统计。

本轮按剖面采样含删失的轨迹，与上一轮 4608 次更新、每批 4096 个首碰标签的
采样不同，因此直接归因应使用本轮重跑的配对冻结组；旧实验作为历史参考。

```powershell
python python/first_hit_se_cli.py joint-state --config python/configs/first_hit_se_joint_h16_v1.json
python python/first_hit_se_cli.py joint-state --config python/configs/first_hit_se_joint_h16_v1.json --phase report
python python/tests/test_first_hit_joint_state.py
```

`--phase prepare/train/evaluate/audit/report` 可分别执行各阶段，训练自动从检查点
恢复并核对配置、输入和实现哈希。数值检查包含四条指定光线的八倍细分、内部
积分步长减半，以及代表模型的 C++ 导出和距离采样检查。

六组训练均完成 960 次更新。联合组可训练参数为 39,545，冻结组为 22,168；
两者实际网络结构和总参数量相同。下面为三个优化种子的均值，标准差与逐种子
配对结果保存在报告中。梯度指标均以参考首碰位置和入口符号为条件。

| 指标（越低越好） | 共同起点 | 仅训练梯度头 | 联合训练 |
| --- | ---: | ---: | ---: |
| 共同原训练梯度 NLL | 1.196853 | 1.195904 | 1.195119 |
| 验证梯度 NLL | 1.229375 | 1.228302 | 1.227374 |
| 相机测试梯度 NLL | 1.226618 | 1.226584 | 1.226102 |
| 相机逐光线 PIT KS | 0.033346 | 0.033005 | 0.034758 |
| 相机平均最大 T 误差 | 0.023864 | 0.023864 | 0.017798 |
| 完整测试梯度 NLL | 1.212358 | 1.213077 | 1.212631 |
| 完整测试 PIT KS | 0.005793 | 0.008372 | 0.008697 |
| 保留场景光线平均最大 T 误差 | 0.046732 | 0.046732 | 0.035912 |
| 保留场景梯度 NLL | 1.135345 | 1.136713 | 1.137899 |
| 保留场景梯度 PIT KS | 0.025738 | 0.026204 | 0.026801 |

联合训练的相机透射率误差下降 25.4%，相机梯度 NLL 下降 0.000481，均为 3/3
种子改善；相机逐光线 PIT KS 上升 5.3%，3/3 种子变差。保留场景的梯度 NLL
也是 3/3 种子变差，PIT 则有 1/3 改善。保留场景包含三条光线，其中两条有
梯度标签；完整测试集还包含八条仿射剖面，因此不应将相机和整体指标混用。

按验证联合 NLL 选择的辅助结果给出相同的相机结论：冻结／联合的梯度 NLL
为 1.226530／1.226065，逐光线 PIT 为 0.032624／0.034338，T 误差为
0.023864／0.017774。保留场景的 PIT 对权重选择较敏感，验证选择后为
0.025347／0.021971，有 2/3 种子改善，但其梯度 NLL 仍为 3/3 变差。

本轮支持在固定 h16 下通过联合微调改善透射率，条件梯度分布未获得全面改善。
联合组同时解冻了状态网络与 hazard，因此不能将收益单独归因于状态更新；
本轮也不能证明或排除 16 维状态的容量上限。三个种子的标准差仅反映优化随机性，
不包含跨场景或 GP 标签抽样的不确定性。

[联合训练对照页](../outputs/first_hit_se_joint_h16_20261013/comparison.html)、
[完整统计](../outputs/first_hit_se_joint_h16_20261013/summary.json)、
[实验配置](../python/configs/first_hit_se_joint_h16_v1.json) 保存本轮设置和结果。

配对轨迹索引累计哈希、参数更新范围和 51 条原验证／测试剖面的标签检查全部
通过。两个代表模型在四条探针光线上，精确八倍细分的最大 T 差为 2.31e-5，
积分步长减半的最大 T 差为 2.21e-5；条件梯度 CDF 的对应最大差为 2.04e-5
和 1.84e-5。每模型三个剖面、每剖面 512 次 C++ 距离采样检查通过，最大
Python／C++ 预测差为 3.79e-6。六组 CPU 复核与 CUDA 验证联合 NLL 的最大差
为 5.02e-9。整组 CTest 中的 47 项神经网络测试通过，原 RNO／GRU 权重哈希
保持不变；报告的 16 个本地引用已核对。

[配对与数值检查](../outputs/first_hit_se_joint_h16_20261013/audit.json) 和
[最终验证记录](../outputs/first_hit_se_joint_h16_20261013/validation.json) 保存详细证据。

## 单光线、多首碰距离诊断（2026-10-14）

`python/configs/first_hit_se_single_ray_v1.json` 固定 σ=0.05 shader ball 相机像素
`(376,96)`，检查当前 h16 / K8 是否能拟合一条光线上的不同首碰距离。
共同起点固定为上一轮 `joint_s20261032/last.pt`，不根据本轮测试结果挑选。
网络结构、SE 核、无入口条件化、首次上/下穿语义和原始分片三次均值保持不变。

本实验刻意以 **GP realization** 划分训练/验证/测试：三者共享一条确定的均值曲线，
独立生成整条光线上的相关 GP 实现，每个实现仅保存第一次过零的位置和梯度。
这与原先按光线划分的泛化实验不同；不能把本实验模型在该光线上的结果继续报告为
“未见光线测试”。普通数据采集器的禁止重复光线规则不变，诊断模块先分别采集，
再显式写出 `split_unit=realization` 的专用数据集。

| 划分 | GP 实现数 | 首碰梯度标签 | GP 根随机种子 |
| --- | ---: | ---: | ---: |
| 训练 | 32,768 | 16,098 | 88001401 |
| 验证 | 8,192 | 4,009 | 88001402 |
| 测试 | 32,768 | 16,083 | 88001403 |

每个划分均采集目标步长 `ell/128` 和 `ell/256`，仅使用细网格标签训练、评估。
三者的全部 realization 种子（包括两个分辨率）互不重叠，也未与原来该像素的参考
样本重叠；均值系数和完整物理几何逐项相同。该光线入口均值较高，本次未观测到
负入口实现，因此没有单独验证上穿分支。

只用训练集首碰距离的八分位点确定 8 个区间，并将两端扩展至介质入口和出口。
归一化距离 `q=s/ell` 的区间边界为：

`0, 4.374520, 4.635506, 4.874056, 5.098338, 5.341243, 5.605996, 6.018630, 9.284465`。

各测试区间分别包含 `1965,1983,2139,2054,2001,1904,2123,1914` 个首碰标签。
这些是**首碰落在距离区间内的条件分布**，不是精确固定 q 的条件 GP 桥。
模型 CDF 在每个参考首碰位置及入口符号处求值，然后在区间内平均；不能拿区间
中心的一次预测代替该平均。另对每个参考命中计算 PIT，再逐区间检查校准。

保留原模型作基线；三个优化种子 `20261041/42/43` 分别执行仅训练梯度头和联合
训练，共六组。每组 768 次更新，每次从 32,768 条训练轨迹中均匀抽取 2,048 条，
包含删失样本；每组累计使用 1,572,864 条轨迹，这不是新增的独立 GP 样本数量。
配对组的轨迹索引哈希必须一致。损失沿用完整联合 NLL，AdamW 学习率从 `3e-4`
余弦下降至 `3e-5`，梯度裁剪 1，权重衰减 `1e-5`。仅头组冻结初始编码、状态
演化与 hazard；联合组同时更新这三个模块和梯度头。

主结果固定为最后一轮，辅助结果使用独立验证集联合 NLL 最小的权重；图中固定
展示种子 `20261042`，不按测试指标选择。报告同时保存三个种子的统计、每个距离
区间的 CDF/PIT、近零速度概率、透射率，以及每个微调模型相对原模型的配对测试
梯度 NLL 差值及大样本 95% 区间。优化种子的标准差和 GP 采样误差分别解释。

```powershell
$env:PYTHONPATH = (Resolve-Path outputs/.analysis_deps).Path
python python/first_hit_se_cli.py single-ray --config python/configs/first_hit_se_single_ray_v1.json
python python/tests/test_first_hit_single_ray.py
```

也可以用 `--phase collect/train/evaluate/report` 单独执行各阶段。
上面的 `PYTHONPATH` 复用工作区已有 Matplotlib 图表依赖；采集、训练、评估只依赖
原有 PyTorch/NumPy。若环境已经安装 Matplotlib，可省略该环境变量。
[单光线报告](../outputs/first_hit_se_single_ray_20261014/comparison.html)、
[数据独立性检查](../outputs/first_hit_se_single_ray_20261014/data_audit.json)、
[参考网格检查](../outputs/first_hit_se_single_ray_20261014/reference_audit.json)
保存结果和来源。

对固定的一条光线，距离本身就可以标识其均值历史。因此即使单光线拟合成功，
也只能支持当前模型在该条件族上的拟合能力，不能单独证明 h16 学到了可迁移的
历史压缩，更不能证明任意场景中 8 个分量已经足够。

六组训练已完成，累计训练约 594 秒。下面是独立测试 realization 上的结果；
微调模型取三个种子的平均值，完整标准差和逐种子结果保存在报告中。

| 固定最后一轮 | 梯度 NLL | 整体 PIT KS | 8 区间平均 PIT KS | 最大区间 CDF 误差 | 最大 T 误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 RNO | 1.093065 | 0.026377 | 0.037195 | 0.067698 | 0.078125 |
| 单光线仅微调梯度头 | 1.089901 | 0.020266 | 0.027103 | 0.046131 | 0.078125 |
| 单光线联合微调 | 1.090124 | 0.021103 | 0.028006 | 0.049222 | 0.006236 |

两种微调的梯度 NLL 均在三个种子上改善，各自相对共同起点的配对测试 NLL 差值
大样本 95% 区间均小于零。8 区间平均 PIT KS 分别下降约 27.1% 和 24.7%；联合
训练的最大 T 误差下降约 92.0%。梯度头组与原模型的 T/H/log hazard/log density
逐项不变。联合训练对梯度的收益没有超过仅头微调。

按验证集选择的辅助权重校准更好：仅头/联合模型的梯度 NLL 分别为
`1.089144/1.089369`，整体 PIT KS 为 `0.009007/0.012048`，8 区间平均 PIT KS
为 `0.022054/0.024688`，最大区间 CDF 误差为 `0.032364/0.035491`。
这支持检查点选择对本例的梯度校准有影响；不能只用训练步数更多判断模型更好。

本轮说明当前 h16/K8 能改善同一光线不同首碰距离的条件梯度分布，且透射率可
拟合到较小误差；并没有证明梯度误差已经消失。原模型较大的局部偏差集中在较晚
首碰区间，微调后仍有残差。它支持单光线专门训练有效，但不能由此唯一定位原来的
多光线误差是数据覆盖、状态表示或优化中的哪一项。

新增 4 项分箱/独立性/条件 CDF 平均测试及 3 项联合训练回归测试通过。两个代表
模型的 C++ 参数预测和各 512 次碰撞采样检查通过，最大 Python/C++ 预测差约
`3.82e-6`。原始共同起点权重哈希保持不变。

## 2026-10-14：Tungsten 线性参考的相机专用 RNO

本轮使用 `render_neural_renewal_shader_ball_ply_point_linear_se_h16_sigma_0.05.json`
中的 VDB 和相机。场内实际参数为 `sigma=0.05000000074505806`、
`ell=0.14142135834465194`；相机位于 `(2.27,-1.09,1.42)`，目标为 `(0,0,0.35)`，
垂直视场角 25 度，分辨率 512×512。所有来源都是相机射线，按像素划分不同数据集。

固定参考为 `gpis_function_space`、H=1、B=8、Δq=1/64、rank tolerance=1e-10、
`tungsten_linear`。训练距离为从 `activeDomain` 入口计量的 `s=ell*q`，
相机到入口的距离不计入标签；保留首次上/下穿及未碰撞的删失实现。

| 划分 | 射线数 | 独立实现 | 首次命中 |
| --- | ---: | ---: | ---: |
| 训练 | 191 | 391,168 | 182,062 |
| 验证 | 32 | 65,536 | 33,272 |
| 测试 | 24 | 196,608 | 126,632 |

另有 1 条候选训练射线未进入介质。653,312 个 GP 种子无重复，入口位置回算最大误差为
`2.22e-16`。采集器支持 `collection_workers` 并行处理独立射线，默认 1；
串行与并行采集的标签、随机种子和打包顺序逐项相同，不改变参考采样配置。

RNO h16/K8 有 39,545 个参数，从随机初始化训练 200 轮，按验证联合 NLL 选择第 198 轮，
验证停滞早停，用时约 18.2 分钟。测试联合 NLL 为 `1.526339`，距离 NLL 为 `0.766365`，
纵向速度 PIT KS 为 `0.004176`。24 条测试射线的逐射线最大透射率误差平均为 `0.021413`，
最差为 `0.051823`；这些误差使用每条射线 8192 个参考实现、513 个查询位置计算。

报告显示 8 条预先选定射线的透射率，以及 3 条射线上共 9 个实际首碰代表点。
代表点由前半部分实现的 20%/50%/80% 命中距离分位附近选择；后半部分独立实现
用于两侧各 7.5% 分位窗口的梯度分布比较。纵向是 Tungsten 总场割线，横向采用
平稳 SE 条件高斯补全；图中三维参考是这一重建约定，不是额外测量的三维梯度。
窗口分布不能解释为精确固定位置的条件真值。像素 `(368,368)` 的第三个窗口仍有
`0.12053` 的速度 CDF 最大差，明显大于其参考 DKW 95% 半宽约 `0.05791`，
说明全局 PIT 较小并不保证各局部窗口都准确。

新增 3 项并发采集、独立窗口和截断混合采样测试通过；已有 4 项函数空间集成测试及
C++ 原生测试通过。最终模型对 3 条射线的 Python/C++ 预测校验通过，最大参数差约
`4.15e-6`，每条另外进行了 512 次模型碰撞采样检查。

```powershell
python python/first_hit_se_cli.py camera-study --config python/configs/first_hit_se_tungsten_camera_sigma005_20261014_study.json --phase all
```

可分阶段运行 `collect`、`train`、`report`；未完成的训练从 `last.pt` 恢复，已完成训练检查
配置与数据哈希后复用。结果见[可视化报告](../outputs/first_hit_se_tungsten_camera_sigma005_20261014/summary.html)，
配置见[相机采集](../python/configs/first_hit_se_tungsten_camera_sigma005_20261014_dataset.json)、
[RNO 训练](../python/configs/first_hit_se_tungsten_camera_sigma005_20261014_rno_h16.json)。

## 首次碰撞混合渲染（CPU 标量，2026-10-14）

`transport.mode=hybrid_first_hit` 将相机首次段与后续段分开。`first_hit.method` 可选
`neural_first_hit_se` 或 `gpis_function_space`；`continuation.method` 当前支持
`neural_renewal` 或 `classic_local`。首次段和续段后端都要求 `scalar`，不接受 `auto` 或 GPU 静默回退。
首次段使用 cubic 均值剖面，RNO 内部步长从导出模型读取；后续 renewal 的
`profile_mode` 独立配置，可使用 `point_linear`。

直接可用的场景配置：

- [RNO 首次段 + renewal](../configs/render_first_hit_rno_renewal_shader_ball_sigma005.json)
- [GPIS 首次段 + 相同 renewal](../configs/render_first_hit_gpis_renewal_shader_ball_sigma005.json)
- [RNO 首次段 + classic local](../configs/render_first_hit_rno_local_shader_ball_sigma005.json)
- [GPIS 首次段 + classic local](../configs/render_first_hit_gpis_local_shader_ball_sigma005.json)

这四份配置保持原 sigma=0.05 VDB、相机、512×512 分辨率和 128 spp。
CPU 验证可以通过 CLI 缩小渲染预算：

```powershell
cmake --build build --config Release --target macrofacet_experiments macrofacet_tests macrofacet_compare_first_hit_render --parallel 4
build/Release/macrofacet_experiments.exe render --config configs/render_first_hit_rno_renewal_shader_ball_sigma005.json --width 64 --height 64 --spp 64 --threads 8
build/Release/macrofacet_experiments.exe render --config configs/render_first_hit_gpis_renewal_shader_ball_sigma005.json --width 64 --height 64 --spp 64 --threads 8
python tools/compare_first_hit_render.py
```

最后一条命令保存两份实际运行配置，运行两种首次段、相同 renewal 续段的 64×64 / 64 spp 渲染，以及三条固定
相机射线各 4096 次/方法的采样检查，生成
[报告](../outputs/first_hit_hybrid_render_20261014/summary.html)。
可用 `--width`、`--height`、`--spp`、`--threads`、`--ray-trials`、`--output` 调整预算；
`--report-only` 从现有 CSV 重新生成图表。输出包含线性 PFM、BMP、逐样本 CSV、PNG/SVG、
统计指标和配置/模型/VDB/可执行文件哈希。模型和 VDB 仍需本地已有；脚本不会训练模型。

接入遵守以下约定：

- 仅从 `activeDomain` 入口启动一次首次段。输出 `s=ell*q`，相机到入口的距离独立保存。
  跨越体素或积分步不会重新抽取首次事件。未碰撞直接返回环境光。
- 首次反射使用模型采出的完整梯度，应用 Fresnel；不重复乘透射率、除距离 PDF，
  不再抽 classic 法线或添加 Rice 权重。
- 保存原始 `initialSign` 和梯度。renewal 续段按 `side=initialSign` 同时变换均值场和完整梯度，
  使 renewal B 初始化满足向正侧出发；核、距离单位、插值单元和梯度长度保持一致。
- `traceRenewalFromSurface` 接收位置、出射方向、完整梯度、已有路径权重及已完成碰撞数。
  第一次碰撞计入 roulette 和深度上限；RNO 隐状态不传给后续方向。
- 相机、首次事件、横向梯度与后续传播使用由 `(seed,pixel,sample,stage)` 派生的独立随机流。
  RNO/GPIS 消耗随机数的数量不同，不会改变相机射线、后续段种子或下一条样本。
- 可设置 `render.write_first_hit_samples=true` 保存采样记录；为限制内存，此诊断模式最多
  1,000,000 个相机样本。正常渲染默认关闭。渲染统计另外保存 `hybrid_summary.json`。

GPIS 对照只替换首次段，参数为 H=1、B=8、步长1/64、`tungsten_linear`。两种方法均用
纵向速度和解析 SE 横向条件高斯重建完整梯度。速度/法线的固定射线 CDF 是命中后的联合
边缘分布，包含距离与梯度两部分误差，不等价于固定碰撞点的梯度头误差。
图像比较提供有限 spp RMSE 及配对差值的采样噪声估计，不能将全部图像差值视作模型偏差。
上述对照的后续段使用旧 renewal 的表面出生模型，不保留相机段完整 GP 历史。

使用 local 续段时，`transport.continuation` 只需
`{"method":"classic_local","backend":"scalar"}`，不加载 renewal 模型。
`material.gp_model` 为 `local_tangent`，`transport.classic_phase_proposal=target_vndf`
使用已有局部 NDF 的精确可见法线采样。首次碰撞仍按模型完整梯度反射并应用一次 Fresnel；
随后从该位置和出射方向启动原有 DDA 传播，保留路径权重及已完成的碰撞数。
local 沿用场景原有 baked density 和均值场朝向（正负首次穿越均如此），不传递相机段的
GP 条件，也不因 `initialSign` 翻转局部 NDF。首次事件与后续方法使用独立随机流。

```powershell
build/Release/macrofacet_experiments.exe render --config configs/render_first_hit_rno_local_shader_ball_sigma005.json
```

可添加 `--width 32 --height 32 --spp 16 --threads 8 --output outputs/first_hit_rno_local_smoke_20261014`
作小图验证。正式配置的输出目录为 `outputs/render_first_hit_rno_local_shader_ball_sigma005`。
该小图已验证：16,384 条相机路径、7,831 次首次命中、13,824 次 local 续段采样，
数值失败与深度上限截断均为 0。新增 local 接入回归检查包含首碰随机流不变、
单位白环境能量、线程确定性、深度计数，以及与独立 classic_local 续段的逐样本权重比较；
原生 CTest 通过（207,020 个检查，0 失败）。

原生测试覆盖：入口距离、RNO 常数风险的解析命中率、GPIS 适配与直接参考样本一致、
正负穿越、完整梯度投影、负侧均值/梯度与 lazy profile、单位白环境能量、线程确定性、
续段工作量与首次随机流隔离，以及首次碰撞的深度计数和不支持配置的拒绝。

本次运行两组各 262,144 条相机路径，进入域的射线均为 261,161 条。RNO 首次命中
124,913 次（47.8299%），GPIS 命中 125,216 次（47.9459%）；上穿命中分别为 9 与 15 次。
两组均无数值失败或深度上限截断。几何/梯度检查最大误差 `1.33e-15`，PFM 与逐样本颜色
均值误差小于 `2.99e-8`（float32 输出精度）。相机方向、入口几何和四组 uint64 种子逐项一致。
线性 RGB 图像 RMSE 为 `0.0151915`，配对差值的估计 RMS 采样噪声为 `0.0147784`。
三条固定射线的速度边缘 CDF 差为 `0.02441 / 0.02795 / 0.03620`，法线余弦 CDF 差为
`0.01440 / 0.02187 / 0.02629`，每条/每方法 4096 次独立采样。
CPU 标量本次记录 RNO 约 324 秒、GPIS 约 70 秒；任务并发运行，时间用于复现记录，
不作为隔离的性能基准。原生 CTest 通过，内部 206,634 个检查、0 失败。

## 2026-10-15：固定均值剖面的条件单调样条

`architecture=spline` 复用相机专用 Tungsten-linear 数据及原有 train/validation/test 射线划分。
在每条射线的 `q/Q∈[0,1]` 上均匀取 128 或 256 个点，输入交错排列的
`asinh(k*b), asinh(k*db/dq), q/Q`，末尾附加 `log1p(Q)`。
其中 `k` 是入口符号；正负分支共享权重，符号翻转对称性由输入变换保证。
编码器为两层宽度 64 的 SiLU MLP，输出 32 维特征。
线性 CDF 头输出逃逸 logit，以及 32 或 64 个有理二次样条区间的宽度、高度、端点导数。
公式参考 [Neural Spline Flows](https://arxiv.org/abs/1906.04032)，项目内实现不引入 nflows 依赖。

对每个符号分支，`F(q)=(1-e)*G(q/Q)`，末端逃逸质量为 `e`。
命中样本使用 `-log((1-e)*G'(q/Q)/Q)`，删失样本使用 `-log(e)`；
完整训练损失还包含入口符号解析先验及命中点的条件速度混合 NLL。
先进行 20 epoch 距离预热，再联合训练。梯度头输入为编码特征、`q/Q`、`log1p(q)`、
`asinh(k*b(q))`、`asinh(k*b'(q))`；输出保留 8 分量正半轴截断高斯及残差均值形式。
训练全程没有重新生成 GPIS 标签，也不使用测试集选择模型。

C++ 抽样先按 `Phi(k*b(0))` 抽入口符号，只编码选中的分支一次，随后判断逃逸，
并解析求逆得到碰撞位置。仅命中后执行梯度 MLP；完整空间梯度沿用原 SE 条件恢复。
当前仍从 VDB 构造完整三次剖面，再在其上取固定点，故总渲染计时包含这项成本。
碰撞点局部均值与导数仍来自原三次剖面；固定采样只用于编码输入。

实现位于 `python/first_hit_se/spline.py`、`spline_train.py` 及
`src/learned/FirstHitSEModel.cpp`。训练和导出可使用：

```powershell
python python/first_hit_se_cli.py spline-train --config python/configs/first_hit_se_spline_p128_k32_sigma005.json
python python/first_hit_se_cli.py spline-study --phase all
build/Release/macrofacet_experiments.exe render --config configs/render_first_hit_spline_local_shader_ball_sigma005.json
```

四组训练配置分别为 `p128_k32`、`p256_k32`、`p128_k64`、`p256_k64`，均位于
`python/configs/first_hit_se_spline_*_sigma005.json`。重训须使用新的输出目录。
`spline-study` 是本次数据集的复现实验入口，要求四组训练均已完成，支持单独的
`--phase evaluate/benchmark/report`；性能阶段顺序运行 RNO/Spline/GPIS，避免并发负载。

验证集 joint NLL 选中 `p256_k64`。在 24 条测试射线上，RNO / Spline 最大透射率误差
分别约为 `0.05182 / 0.14935`；相同九个早中晚碰撞窗口的最大速度 CDF 误差为
`0.12053 / 0.07819`。梯度窗口的改善不能代替全部碰撞样本的 NLL 检查，且当前距离
分布精度仍落后于 RNO。128→256 个输入点、32→64 个样条区间并未带来一致改善。
[完整报告](../outputs/first_hit_spline_20261015/summary.html) 保存四组指标、三种策略计时、
透射率、梯度 CDF 和 Python/C++ 校验。数值校验同时限制 CDF、混合参数和点态 log hazard；
窄样条区间的点态密度对 float32 矩阵乘的舍入更敏感，具体差异及容差逐项保存。

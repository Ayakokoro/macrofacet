# Renewal+ 公共剖面与首达参考采样

本模块实现 `GPIS_RenewalPlus_Neural_Rendering.md` 的第一步：共享射线均值剖面、两种起点条件、Matérn-3/2 首达参考采样。它不是新的渲染模式，也不包含 GRU、累计 hazard 网络或参考渲染器。

## 核与单位

`matern_3_2` 在三维 `CovarianceKernel`、一维 `FirstPassageStationaryKernel` 和状态空间采样器中统一为

\[
\rho(x)=(1+x)e^{-x},\quad x=s/\ell,\quad
b(x)=m(\mathbf o+\ell x\omega)/\sigma,\quad \operatorname{Var}(Z')=1.
\]

各向同性梯度协方差是 `sigma^2 / ell^2 * I`。配置使用 `material.roughness=alpha` 时，继续保持 `Cov(G)=alpha^2/2*I`，此时 `ell=sqrt(2)*sigma/alpha`。

这是 Matérn-3/2 的长度约定变更。若希望保留旧版本显式长度配置的同一个物理 GP，令 `ell_new=ell_old/sqrt(3)`。旧首达数据还必须变换 `x_new=sqrt(3)*q_old`、`W_new=W_old/sqrt(3)`，不能只修改标签中的核名称。建议重新生成数据。新 resolved config 明确记录 `unit_decay`。

SE 和 Matérn-5/2 的定义不变。现有 `classic_local`、`classic_global`、`global_conditional` 保留；后者仍仅支持 SE，不会把 Matérn 静默送入 SE 条件公式。

## 公共接口

- `mathutility/CubicPolynomial.h`：Hermite 三次多项式、驻点划分、首次向下穿越。导数根将区间分成单调段，因此不会只凭两端同号就跳过内部穿越。
- `gpss/RayMeanProfile.h`：以标准化距离存储分段三次均值。`fromField()` 使用与渲染相同的 `MeanField`，在全部插值单元边界切分；`affine()` 构造仿射剖面；`subdivided()` 细分已有多项式而不改变均值。
- `gpss/Matern32Reference.h`：单位方差的值—导数状态转移、条件 bridge、初始化和首达采样。

`RayMeanSegment::features()` 返回

```text
b(0), b(1), dx*b'(begin+), dx*b'(end-), log(dx)
```

这里 `b'` 是对标准化距离的导数；`polynomial.derivative(u)` 则是对段内坐标 `u` 的导数。两端导数从该段多项式计算，避免体素面另一侧梯度污染。

NanoVDB 三线性均值沿单元内射线确实是三次多项式。球等非多项式解析均值使用分段三次近似；要使用精确的体素均值语义，应先烘焙完整 NanoVDB。公共 `fromField()` 会检查整个射线覆盖，不能将稀疏窄带的缺失值当作有效 SDF。

`profile.maximum_step` 默认 `0.25`，控制确定性剖面分段。`grid.step_sizes` 控制随机参考节点。后者可以远小于前者；改变随机参考分辨率不会改变已构造的均值多项式。解析非多项式均值还需要单独检查 `profile.maximum_step` 收敛。

## 起点模式

模式 A：`initial_condition.type="positive_exterior"`，仅已知 `F(0)>0`。采样器内部生成 `Z0>-b0` 的截断正态与独立标准正态 `D0`；这些隐藏实现值不写入网络特征。当前此入口支持 Matérn-3/2。

有空间均值时指定 `rays`，每条只有 `id/origin/direction`，不接受已知 `gradient`。仿射均值则使用：

```json
{
  "type": "positive_exterior",
  "profiles": [{"id": "flat", "beta_0": 0.0, "beta_a": 0.0}]
}
```

模式 B：沿用 `initial_condition.type="collision_state"`。空间射线保存完整、未归一化的 `gradient`，要求 `dot(direction,gradient)>0`；仿射模式继续接受三个 beta 的 `parameter_space`。初始化为

```text
Z0 = -b0
D0 = ell/sigma * dot(direction,gradient) - b'(0+)
```

体素边界不重置状态。起点零值不计为新碰撞。两种模式都保留有限长度未命中样本，命中速度 `W` 为正，物理导数为 `-sigma/ell*W`。

## 运行与输出

```powershell
cmake --build build --config Release --parallel 4
build\Release\macrofacet_experiments.cmd first-passage --config configs\renewal_matern32_exterior.json
build\Release\macrofacet_experiments.cmd first-passage --config configs\renewal_matern32_surface.json
```

两个例子不依赖外部网格文件；前者是平面空域出发，后者是球面均值上的已知表面出发。可用 `--trials 32 --threads 2 --output outputs/renewal_smoke` 做小规模检查。

沿用 `first_passage_samples.csv`、曲线和 summary。模式 A 的 `beta_g` 列为空；它不是零导数观测，起点类型由 resolved config 和段表明确标注。

新增 `first_passage_mean_segments.csv`，每个 kernel/profile 导出一次：

```text
kernel_id,state_id,start_mode,segment,x_begin,x_end,
b0,b1,dx_db0,dx_db1,log_dx,cubic_a,cubic_b,cubic_c,cubic_d
```

这张表只包含确定性几何与已知模式，不包含随机实现。它与逐轨迹样本通过 `kernel_id/state_id` 关联。`write_raw_samples=true` 才能保留联合距离／速度训练所需的逐轨迹标签。场景采集、变长序列加载、GRU 网络训练和评估已在独立 Python 包实现，见 [训练说明](../python/README.md)。

Matérn-3/2 在仿射和完整空间均值下都使用状态空间／bridge 后端。其他核的原有碰撞状态、固定正值和固定终点实验保留；非 Matérn-3/2 的 circulant 后端仍使用有限差分导数。固定终点实验也仍使用独立的 circulant 实现。

节点转移是解析高斯转移；连续首达依赖 Hermite 插值及有限 bridge 细分，仍是数值参考。`bridge_sigma_margin` 是细分准则，不是遗漏交点概率的认证界。必须比较更细步长、最小步长以及穿越速度分布。

## 已移除的旧模型

旧 I-spline Python 包、训练配置、导出脚本、C++ 模型类、模型专用测试和本地模型产物已移除。`transport.first_passage_model` 现在明确报错，避免静默忽略旧配置。

C++ 构建不再依赖 PyTorch/LibTorch；Windows `.cmd` 入口保留。`data_analysis` 下的首达可视化工具继续使用 Python 标准库。

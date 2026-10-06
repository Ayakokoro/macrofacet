# 固定终点的首次碰撞斜率采样

本模式估计给定距离发生**首次下降碰撞**时的沿射线斜率分布。不是终点单点条件高斯分布，也不是从普通 FPT 样本中筛选一个距离区间。

## 运行

```powershell
cmake --build build --config Release --parallel 4
build\Release\macrofacet_experiments.cmd first-passage --config configs\shader_ball_fixed_endpoint_se.json
if ($LASTEXITCODE -ne 0) { throw "Fixed-endpoint generation failed" }
python data_analysis\plot_fixed_endpoint.py --config data_analysis\configs\shader_ball_fixed_endpoint_se.json
```

默认复用 `shader_ball_full.nvdb`、原示例的起点和方向，比较 `g025/g100`。固定物理距离为 `0.004`，SE 的 `ell=0.002`，所以终点是 `q=2`。每组梯度、每个步长独立生成 65,536 个**提案**；最终有效样本量小于提案数。输出目录为 `outputs/shader_ball_fixed_endpoint_se`，不覆盖原透射率实验。

## 概率定义

记出生观测为 \(B=\{F(0)=0,\nabla F(0)=\mathbf g\}\)，要求正的出生方向斜率。采样整段

\[
F(t)=m(\mathbf b+t\mathbf d)+\xi(t)
\]

在 \(B,F(t_*)=0\) 下的条件 GP，终点斜率不作为观测固定。先按出生值和中心差分残差导数条件化，再做一次 Schur/Matheron 修正：

\[
\xi_{B,*}(s)=\xi_B(s)+
\frac{\operatorname{Cov}(\xi(s),\xi(t_*)\mid B)}
     {\operatorname{Var}(\xi(t_*)\mid B)}
\big[-m(t_*)-\xi_B(t_*)\big].
\]

因此改变的是整个条件路径，不是强行修改最后一个点；出生条件保持不变。实际代码在无量纲 `q=t/ell`、`X=F/sigma` 上执行这个修正。

每个提案记录 \(v_i=F_i'(t_*)\)，并检查

\[
A_i=\mathbf1\{F_i(s)>0\quad\forall s\in(0,t_*)\},\qquad
w_i=A_i\max(0,-v_i).
\]

只有带负斜率通量的 survival 筛选才能得到首次碰撞斜率分布。Python 对直方图 bin 使用

\[
\widehat p_j=\frac{\sum_{i:v_i\in j}w_i}{\Delta v\sum_iw_i}.
\]

默认横轴是正的入射斜率 \(-F'(t_*)\)。每个固定 kernel/state/distance 组内，使用无量纲权重和物理权重归一化后的密度相同。

## C++ 配置

保留原来的 `initial_condition.type: "collision_state"`；支持原 affine beta states 或完整 `process.mean_field` + physical rays。新增：

```json
"fixed_endpoint": {
  "enabled": true,
  "only": true,
  "distances": [0.004],
  "trajectories": 0
}
```

- `distances` 使用**物理场景距离**，不是 q；可给多个距离。
- `only: true` 跳过普通 FPT 数据生成。`false` 则在普通统计之外额外生成固定终点数据。
- `trajectories: 0` 复用 `monte_carlo.trajectories`；正整数可单独设置提案数。CLI `--trials` 覆盖两类预算。
- 对每个 kernel 和每个配置步长，`distance/ell` 必须落在 GP 网格节点上，且至少距起点两个步长；不支持默默四舍五入目标距离。
- 目标不能超过 `grid.max_time * ell`。NanoVDB 保留全域/射线覆盖和 sigma 一致性预检。
- 固定终点数据始终写出提案 CSV，包括被 survival 筛选排除的提案；`monte_carlo.write_raw_samples` 只控制原普通 FPT 样本。

所有 kernel 的固定终点模式统一使用条件 circulant grid，包括 Matérn 3/2；不是原 affine 模式的 Matérn state-space bridge。

## 输出

`fixed_endpoint_samples.csv`：每个提案一行，包含 kernel/state/尺度和出生 beta、目标物理距离/q、步长、`training_resolution`、trajectory 和独立 seed，以及：

| 字段 | 含义 |
|---|---|
| `start_value`, `endpoint_value` | 两个被约束为零的场值，物理单位 |
| `birth_slope_error` | 终点条件化之后，残差中心差分出生斜率观测的误差，无量纲 |
| `endpoint_slope` | 有符号的 \(d(F/\sigma)/dq\)，未截断、未加权 |
| `endpoint_derivative` | 有符号的物理 \(F'(t_*)\)，等于 `endpoint_slope * sigma / ell` |
| `survived_to_endpoint` | 开区间内此前没有碰撞，0/1 |
| `downcrossing` | 终点斜率是否为负，0/1；它本身不保证是首次碰撞 |
| `flux_weight_q` | `survived_to_endpoint * max(0,-endpoint_slope)` |
| `flux_weight` | 同一权重的物理版本，乘以 `sigma / ell` |
| `mean_profile_type` | `affine` 或 `full_field` |

`fixed_endpoint_summary.csv` 按 kernel/state/目标/步长汇总：提案数、survival 接受率、正权重数、权重和/平方和、有效样本量、加权入射斜率均值/标准差，以及**未筛选**终点斜率的样本和解析高斯均值/标准差。斜率 moments 使用无量纲斜率；加权 moments 是正入射斜率，未筛选 moments 是有符号斜率。

\[
N_{\rm eff}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

`target_field_density` 是 \(p(F(t_*)=0\mid B)\)，不包括 survival。`first_passage_density_q` 和 `first_passage_density_distance` 用端点密度乘以平均 survival-flux 权重估计首次碰撞密度，后者等于前者除以 ell。这两个字段是估计量，不是普通条件高斯终点密度。

summary 还记录两个端点最大值误差（无量纲 `F/sigma`）、出生中心差分斜率最大误差（无量纲）；没有任何正权重时，状态为 `no_positive_weight`，加权 moments 为 NaN，Python 不会伪造零概率密度图。

`resolved_first_passage_config.json` 和 `run_summary.json` 记录实际配置及采样语义。

## Python 可视化

使用独立脚本 `data_analysis/plot_fixed_endpoint.py`。配置中选择 `comparison: "states"` + `kernel_id/state_ids`，或 `comparison: "kernels"` + 唯一 state。`distances` 选择物理距离，省略则展示所有已生成目标；`bins` 控制斜率直方图。

`slope_units: "physical"` 绘制 \(-F'(t_*)\)；`dimensionless` 绘制 \(-\ell F'(t_*)/\sigma\)。可选 `slope_max` 必须覆盖所有正权重样本，否则报错，避免偷偷截断后重新归一化。脚本只使用最细分辨率提案，检查样本权重和 summary 一致，并显示正权重样本数及 ESS。

输出 `fixed_endpoint_slope_density.csv` 和 `fixed_endpoint_slope_density_d0.svg`（后续距离为 `d1` 等）。CSV 的 `target_distance` 标记每张图对应的距离。

## 误差与验证

终点零值条件化在当前离散残差 GP 上是精确的，但不是精确连续 first-passage 解。出生/终点残差导数都使用中心差分。采样多生成一个正向 residual ghost node，以估计终点中心差分；无需在终点以外查询 NanoVDB mean。终点的 mean 斜率取最后一个体素单元内的左侧导数。

沿路径同时按 GP 网格和体素面分段。每段总场是“残差 Hermite cubic + 完整单元 mean cubic”；检查端点和所有导数根，只排除规定的出生零点与终点零点，不能忽略终点附近一段 epsilon 距离。

仍须比较不同步长，检查 Monte Carlo/ESS 和 SDF 体素误差。SE 在很短距离上的端点条件方差可能数值退化，此时明确报错，不添加 jitter。长距离 survival 提案接受率可能很低；增大提案数或进一步设计重要性采样，而不是移除 survival/flux 因子。

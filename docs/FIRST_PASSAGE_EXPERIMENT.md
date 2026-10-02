# 无第一、第二假设的 first-passage Monte Carlo 实验

`macrofacet_experiments first-passage` 在一维射线上直接采样平稳 GP，并把第一次到达阈值的时间

\[
\tau=\inf\{t>0:X(t)\le b\}
\]

作为 ground truth。它不以“终点仍在正侧”代替“整段都在正侧”，也不删除起点条件。因此输出的

\[
\widehat\Sigma(t)=
\frac{\#\{\tau\in(t,t+\Delta t]\}}
     {\#\{\tau>t\}\,\Delta t}
\]

就是离散 crossing 检测精度下、不使用这两个假设的消光系数（first-passage hazard）。

## 运行

```powershell
build\Release\macrofacet_experiments.exe first-passage `
  --config configs\first_passage_kernels.json
```

快速检查可以覆盖 JSON 中的预算：

```powershell
build\Release\macrofacet_experiments.exe first-passage `
  --config configs\first_passage_kernels.json `
  --trials 2000 --bins 100 --threads 4 --output outputs\first_passage_smoke
```

这个命令不构造 3D 场、不烘焙 NanoVDB，也不运行 transport。它先回答一维 GP 的真实首次穿越统计，再为后续 3D GPSS 近似提供基准。

## 同一 CSV 中的三种消光率

`first_passage_curves.csv` 同时写出：

- `hazard_mc`：按 risk set 估计的真实 first-passage hazard，即目标 \(\Sigma(t\mid\mathcal H)\)；
- `endpoint_conditioned_hazard_sigma1`：保留 \(X(0)=a\) 的条件，但用 \(X(t)>b\) 代替整段存活，对应只保留第一个假设后的 \(\Sigma_1\)；
- `pointwise_hazard_sigma2`：进一步删除起点相关性，对应两个假设都使用时的点态 \(\Sigma_2\)；
- `rice_downcrossing_intensity`：未除以正侧占有概率的普通 Rice downcrossing intensity。它与 `pointwise_hazard_sigma2` 不是同一个归一化量。

对于常均值平稳 GP，记 \(c(t)=\operatorname{Cov}[X(t),X(0)]\)、\(q=-c''(0)\)。点态近似为

\[
\Sigma_2=
\frac{p_X(b)}{P(X>b)}\,E[(-\dot X)_+],
\qquad E[(-\dot X)_+]=\frac{\sqrt q}{\sqrt{2\pi}}.
\]

`endpoint_conditioned_hazard_sigma1` 则从
\((X_t,\dot X_t)\mid X_0=a\) 的二维条件高斯分布解析计算。两者与 `hazard_mc` 的 RMSE 及以真实 hazard RMS 归一化的 NRMSE 写入 `first_passage_summary.csv`。跨 kernel 比较近似质量时应看 NRMSE；raw RMSE 会天然偏向整体 crossing rate 较低的 kernel。

具体地，令场方差为 \(s^2=c(0)\)、\(\delta=a-m\)，则

\[
\begin{aligned}
\mu_F(t)&=m+\frac{c(t)}{s^2}\delta,&
v_F(t)&=s^2-\frac{c(t)^2}{s^2},\\
\mu_V(t)&=\frac{c'(t)}{s^2}\delta,&
v_V(t)&=q-\frac{c'(t)^2}{s^2},\\
c_{FV}(t)&=-\frac{c(t)c'(t)}{s^2}.&&
\end{aligned}
\]

在阈值 \(F_t=b\) 处再次做一次一维高斯条件化，

\[
V_t\mid F_t=b,X_0=a\sim
\mathcal N\!\left(
\mu_V+\frac{c_{FV}}{v_F}(b-\mu_F),
v_V-\frac{c_{FV}^2}{v_F}
\right).
\]

于是可直接计算

\[
\boxed{
\Sigma_1(t)=
\frac{p(F_t=b\mid X_0=a)}{P(F_t>b\mid X_0=a)}
E[(-V_t)_+\mid F_t=b,X_0=a]
}
\]

并使用 \(E[(-Z)_+]=s_Z\phi(\mu_Z/s_Z)-\mu_Z\Phi(-\mu_Z/s_Z)\)。这就是当前实现提供的低成本近似形式。它删除第二个假设，却仍然没有桥接存活因子
\(P(X(u)>b,0<u<t\mid X_t=b,\dot X_t)\)，所以必须和 `hazard_mc` 对照，而不能称为精确 first-passage hazard。

`survival` 是经验 \(P(\tau>t)\)，`cumulative_hazard_nelson_aalen` 为 \(\sum d_j/Y_j\)，`survival_from_hazard` 为其负指数。后两列用于检查

\[
-\log S(t)\simeq\int_0^t\Sigma(s)\,ds.
\]

## 起点 conditional survival 对比

`first_passage_curves.csv` 另外为每个 kernel 输出三种可与 Monte Carlo survival 对照的量：

- `start_conditioned_endpoint_survival`：解析端点概率
  \(P(X_t>b\mid X_0=a)\)；它只要求终点在正侧，允许路径此前已经穿越，因此不是 first-passage survival；
- `start_conditioned_sigma1_survival`：对起点条件消光率积分，
  \[
  S_{\Sigma_1}(t)=\exp\!\left[-\int_0^t\Sigma_1(u)\,du\right];
  \]
- `pointwise_sigma2_survival`：完全点态近似
  \(S_{\Sigma_2}(t)=\exp[-\Sigma_2t]\)。

其中直接端点概率使用

\[
X_t\mid X_0=a\sim\mathcal N\!\left(
m+\frac{c(t)}{s^2}(a-m),
s^2-\frac{c(t)^2}{s^2}
\right)
\]

解析求值。`first_passage_survival.svg` 中，同一颜色表示同一个 kernel：实线是真实 Monte Carlo first-passage survival，虚线是 \(S_{\Sigma_1}\)，点线是直接端点概率。后者对零均值平稳过程最终趋向 \(1/2\)，而真实 survival 最终趋向零，这个差异正是“当前点仍为正”不能替代“此前整段都为正”的直观表现。

`first_passage_summary.csv` 的 `endpoint_survival_rmse`、`sigma1_survival_rmse` 和 `sigma2_survival_rmse` 汇总三种近似相对于 Monte Carlo survival 的误差。

## Rice 一阶和二阶展开

`rice_series` 控制起点条件 \(X(0)=a\) 下的 Rice 展开。当前实现支持
`max_order=1` 或 `max_order=2`：

```json
"rice_series": {
  "enabled": true,
  "max_order": 2,
  "relative_tolerance": 1e-6,
  "absolute_tolerance": 1e-9,
  "max_quadrature_subdivisions": 1024,
  "covariance_roundoff_multiplier": 256.0
}
```

对有序时刻 \(0<t_1<t_2\)，程序组装

\[
(X(t_1),X(t_2),\dot X(t_1),\dot X(t_2))\mid X(0)=a
\]

的联合高斯分布，再用 Schur complement 条件化到
\(X(t_1)=X(t_2)=b\)。kernel 接口直接提供任意有符号时间差上的
\(k'(r)\) 和 \(k''(r)\)，协方差块采用

\[
\operatorname{Cov}(X_i,\dot X_j)=-k'(t_i-t_j),\qquad
\operatorname{Cov}(\dot X_i,\dot X_j)=-k''(t_i-t_j).
\]

CSV 中新增：

- `rice_w1`：一时刻向下 crossing product intensity
  \(\bar W_1(t)\)；
- `rice_w2_integral`：二时刻 product intensity 的有序积分
  \(A_2(t)=\int_0^t\bar W_2(u,t)du\)；
- `rice_density_order2`：二阶首次穿越密度
  \(f^{(2)}(t)=\bar W_1(t)-A_2(t)\)；
- `rice_survival_order1/2` 和 `rice_hazard_order1/2`：分别由对应截断密度积分得到的自洽 survival 和 hazard；
- `rice_w2_absolute_error`、`rice_w2_converged`：外层 Gauss--Kronrod 积分诊断。

`first_passage_rice_density.svg` 将 Monte Carlo density、Rice 一阶和 Rice 二阶画在同一张图中。有限阶 Rice 密度在较晚时间变负是截断失效的诊断，不会被裁剪为零。Matérn 3/2 的速度协方差在零时间差处有 cusp；在低于二阶协方差条件化可靠分辨范围的对角邻域，代码使用单侧最近可分辨值，避免把理论半正定的 Schur complement 误判成非 PSD。

## 多 kernel 与 crossing 收敛

目前支持四种平稳、均方可导 kernel：

- `squared_exponential`
- `matern_3_2`
- `matern_5_2`
- `rational_quadratic`（额外参数 `alpha`）

`variance` 是 \(c(0)\)，`length_scale` 使用各 kernel 的标准参数化。相同 `length_scale` 不代表相同梯度方差：SE 和 RQ 的 \(q=\sigma^2/\ell^2\)，Matérn 3/2 的 \(q=3\sigma^2/\ell^2\)，Matérn 5/2 的 \(q=5\sigma^2/(3\ell^2)\)。若要只比较相关形状，应调整长度尺度令 \(q\) 相同。

`grid.step_sizes` 的最小值决定实际采样网格；其他步长从同一 realization 做耦合下采样。每个步长都有独立的 curve/summary 行，因此可以检查有限网格是否漏掉“正到负再回正”的 crossing。若 0.02 与 0.01 的曲线仍明显不同，应继续加入 0.005，而不能把 0.01 当作连续过程真值。

采样使用 circulant embedding。它对指定网格产生具有目标 Toeplitz covariance 的精确高斯向量，然后通过

\[
X_c(t)=X(t)+\frac{c(t)}{c(0)}\bigl(a-X(0)\bigr)
\]

精确条件化到 `initial_condition.value`。这里的“精确”指有限网格上的联合高斯采样；连续区间 crossing 仍需用步长收敛实验控制。

`monte_carlo.thread_count=0` 使用硬件并发数；也可用 `--threads` 覆盖。每条轨迹的随机种子只由全局 seed、kernel 序号和 trajectory 序号决定，所以结果不随线程数变化。

## surviving state 与局部 closure

`state_analysis.snapshot_times` 指定记录时间。只对 \(\tau>t\) 的轨迹记录 \((X_t,\dot X_t)\)，其中导数由最细网格的中心差分估计。

- `first_passage_state_summary.csv`：各时间的 survivor 数、均值、方差、协方差及未来小窗口整体 hazard；
- `first_passage_state_hazard.csv`：在 \((x,v)\) 二维格子中估计
  \(P(\tau\le t+\Delta\mid\tau>t,X_t\in B_x,\dot X_t\in B_v)/\Delta\)；
- `first_passage_state_samples.csv`：当 `write_samples=true` 时写出每条 surviving trajectory 的状态，便于 KDE 或比较完整分布。

比较不同 `snapshot_time` 下相同 \((x,v)\) 格子的 conditional hazard，即可实验检验 \((X_t,\dot X_t)\) 是否足以近似概括历史。如果固定状态后仍有明显的时间依赖，下一步应加入滞后状态，而不是直接把 hazard 强行拟合成常数。

## 当前边界

第一版只支持 `initial_condition.type="fixed_value"` 且初值严格高于阈值。这对应建议中的 \(X(0)=a>0\) 基础实验。Macrofacet 的真实 surface birth 是 \(X(0)=0,\dot X(0)>0\) 的联合条件问题；在基础实验完成 crossing 与 hazard 收敛验证前，不应把当前结果直接称为 surface-start 的最终消光率。

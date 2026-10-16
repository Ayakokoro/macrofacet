# 当前残差 GRU 与 local 的逐光线比较

- [交互报告](../outputs/first_hit_causal_ray_comparison_sigma002_20261016/summary.html)：280 条测试光线，可按改善、退化或绝对误差排序，点击散点或相机像素选择光线。
- [完整结论与方法](../outputs/first_hit_causal_ray_comparison_sigma002_20261016/summary.md)
- [逐光线 CSV](../outputs/first_hit_causal_ray_comparison_sigma002_20261016/per_ray.csv)
- [代表光线 PNG](../outputs/first_hit_causal_ray_comparison_sigma002_20261016/representative_rays.png) / [SVG](../outputs/first_hit_causal_ray_comparison_sigma002_20261016/representative_rays.svg)

本轮检查实际部署的 **20261015** 种子，而不是多个训练种子的平均。固定 checkpoint、参考数据与内部步长，仅新增评估和可视化。

| 指标 | 解析 local | 当前 GRU |
| --- | ---: | ---: |
| 平均逐光线最大 T 误差，280 条 | 0.025339 | 0.007546 |
| 平均梯度 PIT KS，命中数 ≥1,024 的 211 条 | 0.029062 | 0.011540 |

透射率改善最大的例子为 **77、237、74**，相对 local 退化最明显的是 **245、4、221**；当前绝对误差最大的 **62、41、175** 仍优于 local。梯度较好的例子为 **232、235、233**，仍有偏差的例子为 **261、15、266**；梯度相对 local 退化明显的是 **129、17、275**。排名用于诊断，微小差异不等于统计显著。

245 的 T 退化、梯度改善，129 则相反。应分别看均值剖面、T 曲线及误差、log-hazard 残差、参考碰撞位置上的梯度 CDF/PDF 与条件 PIT。报告支持按命中位置分位数分组，避免只看整条光线的平均条件分布。

这里的 local 是残差主干所修正的解析 local hazard，保留原始均值导数和正负入口分支；梯度使用不经训练的 flux-weighted Gaussian。它不是旧实验的无历史神经头 `mixture_local`。渲染 `classic_local` 使用单位化均值梯度和烘焙 density，在理想 SDF 正侧条件下可对应，实际实现不保证完全一致。

参考是每条光线 8,192 次有限历史、有限步长 GPIS 实现。梯度变量是纵向割线速度 W，非完整三维法线真值。光线编号为零起始测试集索引，报告另列采集相机与像素坐标。

```powershell
$env:PYTHONPATH = 'python'
python -B -m first_hit_se.ray_comparison
```

仅生成报告可加 `--phase report`；数据在新的输出目录中，不改变训练协议。解析速度 CDF 已通过零均值 Rayleigh 特例、独立高斯求积、归一化、支撑域与极端方向单调性检查；部署模型、数据、缓存和训练源码哈希均已核对。

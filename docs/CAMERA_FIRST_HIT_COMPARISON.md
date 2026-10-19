# 相机首段的透射率与碰撞梯度对比

`macrofacet_compare_camera_first_hit` 直接调用生产采样器，只采样从相机出发的首段，不进行反射或后续传输。
示例配置：[compare_camera_first_hit_shader_ball_sigma002.json](../configs/compare_camera_first_hit_shader_ball_sigma002.json)。
其中 `source_config` 提供 VDB、相机、材质与数值参数；该渲染配置的首段 / 后续段选择不限制本工具比较的方法。

比较八组：

| 方法 | 步长参数 | 值（世界坐标距离） |
| --- | --- | --- |
| `classic_local` | 无 | — |
| `global_conditional` | 无 | — |
| `gpis_function_space_tungsten` | `step_size` | 1、0.1、0.01 |
| `global_conditional_renewal_plus` | `renewal_distance` | 1、0.1、0.01 |

GPIS 固定每批 8 个点，使用当前 Tungsten 兼容路径的 `renewal+`、`gp_discrete`、
`conditioned_gaussian`、`skip_space=0`。它的批次跨度为
`N * min(step_size, remaining_length / N)`，采样点还有原版的位置偏移；
因此 `step_size=1` 并不表示本场景上相邻采样点真的相隔 1。
GC renewal+ 的参数则是采样正值端点及完整梯度、更新出生条件的间隔，不能与 GPIS 步长等同。

## 运行

在仓库根目录执行：

```powershell
cmake --build build --config Release --target macrofacet_compare_camera_first_hit --parallel 4
.\build\Release\macrofacet_compare_camera_first_hit.exe configs/compare_camera_first_hit_shader_ball_sigma002.json
python tools/analyze_camera_first_hit.py --input outputs/camera_first_hit_comparison_20261017/main --pilot outputs/camera_first_hit_comparison_20261017/pilot --image outputs/render_first_hit_local_gpis_shader_ball_sigma002/render_hybrid_first_hit_directional.bmp
```

首次实验的预采样配置位于 `outputs/camera_first_hit_comparison_20261017/pilot_config.json`。
若需从头复现，先运行该配置，再运行主配置；更换 `output_directory`，避免覆盖已有样本。
`pixels` 为从零开始的相机像素坐标，每条射线固定在像素中心，所有方法共享该射线。
预采样与主采样必须使用不同 `seed`；相同像素需要出现在两者中。
预采样用于确定窗口，主采样用于估计与比较分布。

工具输出逐次样本 CSV、均值剖面、配置快照、方法与射线元数据，以及每组计数。
分析脚本使用 NumPy、SciPy、Matplotlib；兼容仓库现有 `outputs/.analysis_deps`。
生成可本地打开的 `summary.html`、PNG / SVG 图、CSV / JSON 指标及输入文件 SHA-256。

## 交互报告

`summary.html` 顶部默认显示 local、global_conditional、renewal_plus 0.1。
通过复选框任意显示 / 隐藏方法；预设可快速切换 plus 0.01、全部 plus 或全部方法。
所有交互图同步更新，无需重新采样。页面数据和脚本内嵌，可直接在浏览器本地打开，无需服务器或联网。

- 透射率重点图按当前勾选的 local / GC / plus 两两曲线最大差排序，展示前 3 条射线。
- 梯度重点图按相同距离窗口内的 CDF 最大差排序；可切换纵向、模长和两个横向分量，或改为比较所有接受命中。
- GPIS 可作为叠加参考，但不参加这些重点图的排名。排名至少需要两种 local / GC / plus 方法；梯度比较还要求两组命中数均至少 512。
- 每张图带有相对可选基准的差值图（百分点）、最大差位置与命中样本数。鼠标悬停显示数值；拖动横轴放大，双击或按钮恢复；“放大最大差附近”用于定位差异。
- “逐条查看”可选择任意射线和早 / 中 / 晚窗口。原有 PNG / SVG 及对 GPIS 的统计收在“静态导出与完整统计”中，不受复选框影响。

`pairwise_differences.json` 保存完整样本计算的两两最大差、对应位置及样本数。
排名用来定位已有样本中的差异，不能作为挑选之后的独立显著性检验。
显示曲线按共同网格抽稀，保留两两最大差的位置；显示纵值保留 9 位小数，排名指标使用原始完整精度。

## 统计口径

- 距离从 **activeDomain 入口** 开始，相机到入口的真空距离不计入。GPIS 保留渲染器的 float 相机约定，入口的微小舍入差异在校验数据中记录。
- `T(s)` 为未检测到首次交叉事件的比例。正常逃逸为无限碰撞距离；上穿拒绝计为已发生交叉，并单独记录，不当作逃逸。若有无效梯度，分析脚本停止生成无删失的透射率比较。
- 梯度图只含接受的下穿首次碰撞。记录完整未归一化 `gx,gy,gz`；展示纵向速度 `W = -dot(direction,g) * ell/sigma`、梯度模长及两个横向分量。
- 所有命中的梯度 CDF 混合了不同碰撞位置，不能单凭这张图判断同一点的梯度律是否一致。
- 相同距离窗口取独立预采样 GPIS `.01` 的命中距离 10%、50%、90% 分位点，半宽至多 `ell/4`，并避免窗口重叠。所有方法使用相同窗口。每个方法和参考均至少有 512 个命中时，才进入窗口汇总。有限宽度的窗口仍不等于精确固定空间点的条件分布。
- 静态图及原有误差表以 GPIS `.01` 为**离散参考**；交互重点图比较当前所选 local / GC / plus 方法之间的直接差值。GPIS `.01` 没有经过本实验的更细步长收敛证明，不应称为精确 GP 真值。相同随机种子不保证不同算法逐样本配对。
- 等权汇总仅代表所选射线，不代表全图平均误差。DKW 界反映蒙特卡洛不确定性，不包括离散误差或模型近似误差；完整表保留逐射线与窗口样本数。

生产实现的入口条件也不同：GC / renewal+ 显式采样 `F(entry)>0`，GPIS 第一批未施加该条件。
此外，GPIS 使用割线纵向梯度和兼容均值查询；local 使用局部切平面的通量加权梯度，
GC / renewal+ 使用对应条件场的通量加权梯度。比较结果包含这些实际实现差异。

工具与分析器校验样本编号、状态计数、有限性、距离范围、下穿方向。
整段长度不超过 1 时，GC 与 renewal+ 距离 1 使用相同随机流，样本必须逐项一致。
local 透射率还与稠密剖面的数值积分交叉检查；该积分不用于替代原始采样数据。

"""Offline interactive ray browser and exportable scientific figures."""
import json
from pathlib import Path

import numpy as np

from .camera_study import plotting, save_figure
from .ray_comparison import read_json


def example_figure(plt, output, result, curves, ids, name):
    fig, axes = plt.subplots(3, len(ids), figsize=(4.3*len(ids), 8.5), squeeze=False)
    colors = {'reference': '#182536', 'local': '#d66d18', 'gru': '#2464d5'}
    for column, i in enumerate(ids):
        r, c = result['rows'][i], curves[i]
        x = np.array(c['q'])*r['ell']
        axes[0, column].plot(x, c['b'], color=colors['reference'])
        axes[0, column].axhline(0, color='#999', lw=.7)
        axes[0, column].set(title=f'Ray {i} / {r["camera"]}\npixel {r["pixel"]}', ylabel='Mean / sigma')
        for method in colors:
            axes[1, column].plot(x, c[method+'_T'], color=colors[method], label=method, lw=1.5)
        axes[1, column].set(xlabel='Distance from entry', ylabel='Transmittance', ylim=(-.02, 1.02),
                            title=f'Max T error: local {r["T_local"]:.4f}, GRU {r["T_gru"]:.4f}')
        condition = c['conditions'][0]
        for method in colors:
            axes[2, column].plot(c['speed_grid'], condition[method+'_cdf'], color=colors[method], label=method)
        axes[2, column].set(xlabel='Longitudinal speed W', ylabel='CDF at reference hit positions', ylim=(-.02, 1.02),
                            title=f'PIT KS: local {r["gradient_local"]["pit_ks"]:.4f}, GRU {r["gradient_gru"]["pit_ks"]:.4f}')
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='lower center', ncol=3)
    fig.suptitle('Deployed seed 20261015; analytic local baseline; empirical finite-step GPIS reference')
    fig.tight_layout(rect=(0, .04, 1, .95))
    save_figure(plt, fig, output, name)


def report(output):
    result, curves = read_json(output/'summary.json'), read_json(output/'curves.json')
    rows, s = result['rows'], result['summary']
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, name in zip(axes, ('T', 'gradient')):
        subset = rows if name == 'T' else [r for r in rows if r['hits'] >= 1024]
        x = [r['T_local'] if name == 'T' else r['gradient_local']['pit_ks'] for r in subset]
        y = [r['T_gru'] if name == 'T' else r['gradient_gru']['pit_ks'] for r in subset]
        colors = ['#158877' if b <= a else '#cf641c' for a, b in zip(x, y)]
        ax.scatter(x, y, c=colors, s=18, alpha=.75)
        limit = max(x+y)*1.08
        ax.plot([0, limit], [0, limit], '--', color='#8993a1', lw=1)
        ax.set(xlabel='Analytic local error', ylabel='Deployed GRU error', xlim=(0, limit), ylim=(0, limit),
               title='Max transmittance error / all 280 rays' if name == 'T' else 'Gradient PIT KS / at least 1024 hits')
        highlights = (245, 62, 77, 129) if name == 'T' else (261, 15, 129, 17)
        for i in highlights:
            r = rows[i]
            a, b = (r['T_local'], r['T_gru']) if name == 'T' else (r['gradient_local']['pit_ks'], r['gradient_gru']['pit_ks'])
            ax.annotate(str(i), (a, b), xytext=(5, 5), textcoords='offset points', fontsize=8)
    fig.suptitle('Below diagonal: GRU improves on local; above: GRU regresses')
    fig.tight_layout()
    save_figure(plt, fig, output, 'overview')
    example_figure(plt, output, result, curves, [77, 62, 245, 129], 'representative_rays')
    example_figure(plt, output, result, curves, [261, 15, 266, 232], 'gradient_examples')

    distance_gain = s['local']['joint_distance_nll_per_trial']-s['gru']['joint_distance_nll_per_trial']
    gradient_gain = s['local']['joint_gradient_nll_per_trial']-s['gru']['joint_gradient_nll_per_trial']
    result['summary']['joint_distance_gain'] = distance_gain
    result['summary']['joint_gradient_gain'] = gradient_gain
    template = Path(__file__).with_name('ray_comparison_template.html').read_text(encoding='utf-8')
    payload = json.dumps({'report': result, 'curves': curves}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (output/'summary.html').write_text(template.replace('__REPORT_DATA__', payload.replace('</', '<\\/')), encoding='utf-8')

    example_ids = [77, 237, 62, 175, 245, 4, 221, 129, 261, 15, 266, 232]
    table = '\n'.join(f'| {i} | {rows[i]["camera"]} | {rows[i]["pixel"]} | {rows[i]["hits"]} | '
                      f'{rows[i]["T_local"]:.6f} | {rows[i]["T_gru"]:.6f} | '
                      f'{rows[i]["gradient_local"]["pit_ks"]:.6f} | {rows[i]["gradient_gru"]["pit_ks"]:.6f} |'
                      for i in example_ids)
    text = f'''# 当前残差 GRU 与解析 local：逐光线比较

[交互报告](summary.html) · [全部 280 条光线 CSV](per_ray.csv) · [结构化指标](summary.json)

当前渲染模型为 seed **{result['seed']}**，距离为 gru_nll，梯度为 mixture_gru；不是三个种子的平均网络。模型、数据、缓存和训练源码哈希均已核对。所有排名仅用于评估后诊断。

## 比较结果

| 指标 | 解析 local | 当前 GRU |
| --- | ---: | ---: |
| 280 条光线的平均最大 T 误差 | {s['local']['mean_ray_max_T_error']:.6f} | {s['gru']['mean_ray_max_T_error']:.6f} |
| 至少 1,024 次命中的 {s['gradient_rays_with_1024_hits']} 条光线，平均梯度 PIT KS | {s['local']['mean_ray_gradient_pit_ks_1024']:.6f} | {s['gru']['mean_ray_gradient_pit_ks_1024']:.6f} |
| 每次命中的条件速度 NLL | {s['local']['speed_nll_per_hit']:.6f} | {s['gru']['speed_nll_per_hit']:.6f} |

透射率误差下降 {s['T_improved_count']} 条、上升 {s['T_regressed_count']} 条；其余在 1e-9 容差内相同。命中数充足的光线中，梯度 PIT KS 改善 {s['gradient_improved_count_1024']}/{s['gradient_rays_with_1024_hits']} 条。微小差异仍可能来自参考抽样噪声，排名不等于统计显著性结论。

| 测试光线索引 | 相机 | 采集时像素 | 命中数 | T local | T GRU | 梯度 PIT local | 梯度 PIT GRU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{table}

245、4、221 的主要退化在透射率，梯度反而改善；129 的透射率大幅改善，梯度退化。261、15、266 的梯度仍有偏差，但也均优于解析 local。62、41、175 的 T 误差较大，不代表相对 local 更差。较好案例包括 T 改善很大的 77、237、74，以及梯度 PIT 较小的 232、235、233。

不能直接比较 T 最大误差与 PIT KS 的数值来判断“哪部分更重要”。以相同的联合似然分解为参照，距离／入口符号／逃逸部分的 NLL 改善为 **{distance_gain:.6f} nat/参考实现**，命中后的条件梯度部分改善为 **{gradient_gain:.6f} nat/参考实现**，两部分均有贡献。它们不是渲染图像误差的归因。

## 如何看图

1. 在散点图中点击一条光线，或用索引、排名及相机像素图选择。对角线下方表示 GRU 比 local 更好。
2. 看均值剖面 m/σ、T 曲线和带符号的 T 误差。参考 T 保留逃逸样本；灰带为经验 T 的逐点 95% Wilson 区间。
3. 看 log(λ_GRU/λ_local) 来定位残差修正发生在哪一段；正值增加 hazard，负值降低 hazard。这不是经验 hazard 真值。
4. 梯度图固定同一批参考碰撞位置 q 和入口符号 k，分别平均两种模型的条件 CDF，并与参考速度 ECDF 比较。可切换四个按命中位置分位数划分的区间。它们不是同一精确位置的真值曲线。
5. 同时看条件 PIT 校准，理想情况靠近对角线。分箱 PDF 适合查看形状，但 CDF/PIT 对样本波动更稳健。低命中光线会明确标注，不纳入默认梯度困难排名。

## local 的精确定义

本比较的解析 local 是残差主干实际修正的基线，保留原始均值导数，并混合正、负入口分支。其正速度分布是

`p_local(W | q,k) = W * φ(W-μ) / [φ(μ)+μ Φ(μ)], μ=-k*b'(q), W>0`。

这是无需训练的 flux-weighted Gaussian，不是旧实验中名为 mixture_local 的无历史神经头。

渲染 classic_local 使用 local_tangent 的单位均值梯度及烘焙 density；正侧、理想单位梯度 SDF、相同粗糙度且忽略烘焙／积分差异时，两者对应；实际代码不是逐位相同。该报告没有把解析 local 标成 classic_local 渲染结果。

参考来自有限历史 H=1、B=8、Δq=1/64 的 GPIS Tungsten-linear 标签。每条测试光线 8,192 次实现，合计 {s['reference_trials']:,} 次，其中 {s['reference_hits']:,} 次命中。梯度变量 W 是纵向割线速度；报告不提供独立三维法线真值。编号是零起始测试集索引，不能直接当作当前渲染图的像素编号。

## 复现与导出

```powershell
$env:PYTHONPATH = 'python'
python -B -m first_hit_se.ray_comparison
```

只重建报告可加 `--phase report`。PNG/SVG： [总体误差](overview.png)、[代表光线](representative_rays.png)、[梯度案例](gradient_examples.png)。
'''
    (output/'summary.md').write_text(text, encoding='utf-8')
    print(f'Interactive report and PNG/SVG figures: {output / "summary.html"}', flush=True)

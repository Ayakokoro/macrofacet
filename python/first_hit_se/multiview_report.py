"""Reproducible tables and plots for the multiview paired experiment."""
import html
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .camera_holdout import describe, nearest_distance
from .camera_study import plotting, save_figure
from .data import read_json
from .multiview_study import test_groups
from .spline import local_profile


def moments(values):
    return {'mean': float(np.mean(values)), 'seed_std': float(np.std(values))}


def gradient_groups(evaluation, groups):
    result = {}
    for group, ids in groups.items():
        result[group] = {}
        for kind in ('mixture', 'cdf'):
            scores = []
            for run in evaluation['runs']:
                if run['kind'] != kind:
                    continue
                rows = [run['metrics']['test']['per_ray'][i] for i in ids]
                hits = sum(r['hits'] for r in rows)
                scores.append({'speed_nll': sum(r['speed_nll']*r['hits'] for r in rows if r['hits'])/hits,
                    'joint_nll': float(np.mean([r['joint_nll'] for r in rows])),
                    'mean_ray_pit_ks': float(np.mean([r['pit_ks'] for r in rows if r['hits']]))})
            result[group][kind] = {key: moments([r[key] for r in scores]) for key in scores[0]}
    return result


def markdown_table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def report(spec, config, data, output):
    audit, distance, gradient, sampling, trained = (read_json(output/name) for name in (
        'data_audit.json', 'distance_evaluation.json', 'gradient_evaluation.json',
        'sampling_evaluation.json', 'distance_training.json'))
    groups = test_groups(data)
    gradient_summary = gradient_groups(gradient, groups)
    summary = {'data': audit, 'distance': distance['summary'], 'gradient': gradient_summary,
               'sampling': {k: v for k, v in sampling.items() if k != 'per_ray'},
               'gradient_overall': {kind: {metric: moments([r['metrics']['test'][metric]
                    for r in gradient['runs'] if r['kind'] == kind])
                    for metric in ('speed_nll', 'joint_nll', 'pit_ks')} for kind in ('mixture', 'cdf')}}
    summary['local_grid_audit'] = {k: v for k, v in read_json(output/'local_grid_audit.json').items() if k != 'per_ray'}
    describe_one = lambda p: describe(p['features'], p['horizon'], spec['selection']['descriptor_points'],
                                     spec['selection']['descriptor_weights'])[0]
    training = np.stack([describe_one(p) for p in data['profiles'] if p['split'] == 'train'])
    test = np.stack([describe_one(p) for p in data['profiles'] if p['split'] == 'test'])
    # Merged profile order preserves the 183 original training anchors first.
    old_distance, new_distance = nearest_distance(test, training[:183]), nearest_distance(test, training)
    summary['profile_coverage'] = {group: {name: {'median': float(np.median(values[ids])),
        'p95': float(np.quantile(values[ids], .95)), 'max': float(np.max(values[ids]))}
        for name, values in (('original_training', old_distance), ('multiview_training', new_distance))}
        for group, ids in groups.items()}
    test_raw = [p for p in data['profiles'] if p['split'] == 'test']
    residual_errors = np.array([[p['max_abs_T_error'] for p in r['metrics']['per_ray']]
                                for r in distance['results']['residual']]).mean(0)
    summary['distance_failures'] = [{'ray': int(i), 'scene_id': test_raw[i]['scene_id'],
        'source': test_raw[i]['source'], 'new_residual_mean_max_error': float(residual_errors[i]),
        'old_residual_mean_max_error': float(np.mean([r['metrics']['per_ray'][i]['max_abs_T_error']
                                                   for r in distance['results']['old_residual']])),
        'local_max_error': distance['results']['local'][0]['metrics']['per_ray'][i]['max_abs_T_error'],
        'nearest_original_training_profile': float(old_distance[i]),
        'nearest_multiview_training_profile': float(new_distance[i]),
        'profile': describe(test_raw[i]['features'], test_raw[i]['horizon'])[1]}
        for i in np.argsort(residual_errors)[-5:][::-1]]
    summary['source_sha256'] = {str(p): digest(p) for p in Path(__file__).parent.glob('*.py')}
    summary['distance_runs'] = [{k: r[k] for k in ('mode', 'seed', 'parameters', 'selected_epoch', 'training_seconds')}
                                for r in trained]
    summary['gradient_runs'] = [{k: r[k] for k in ('kind', 'seed', 'parameters', 'selected_epoch', 'training_seconds')}
                                for r in gradient['runs']]
    summary['pair_checks'] = {}
    for seed in spec['seeds']:
        pair = [r for r in trained if r['seed'] == seed]
        gp = [r for r in gradient['runs'] if r['seed'] == seed]
        summary['pair_checks'][str(seed)] = {
            'identical_distance_initialization': pair[0]['initial_parameter_sha256'] == pair[1]['initial_parameter_sha256'],
            'identical_gradient_hidden_initialization': gp[0]['common_initial_sha256'] == gp[1]['common_initial_sha256'],
            'distance_frozen': all(r['distance_model_unchanged'] for r in gp)}
        if not all(summary['pair_checks'][str(seed)].values()):
            raise AssertionError('paired-run controls failed')
    write_json(output/'summary.json', summary)
    labels = {'local': 'Local', 'old_direct': 'Old direct', 'old_residual': 'Old residual',
              'direct': 'Multiview direct', 'residual': 'Multiview residual'}
    colors = {'local': '#64748b', 'old_direct': '#fdb38c', 'old_residual': '#93c5fd',
              'direct': '#ea580c', 'residual': '#2563eb'}
    plt = plotting()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(axes, ('mean_ray_max_abs_T_error', 'rmse_T', 'distance_nll'),
                              ('Mean per-ray max |T error|', 'Transmittance RMSE', 'Distance NLL')):
        for j, mode in enumerate(labels):
            values = [distance['summary'][g][mode][key] for g in groups]
            ax.bar(np.arange(len(groups))+.15*j, [v['mean'] for v in values], width=.14,
                   yerr=[v['seed_std'] for v in values], color=colors[mode], label=labels[mode])
        ax.set_xticks(np.arange(len(groups))+.3, ['All 280', 'Original 24', 'Spatial 128', 'Diverse 128'], rotation=15)
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle('Fixed diagnostic rays; error bars: standard deviation over three training seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, 'distance_comparison')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, values, color in (('Original training', old_distance, '#ea580c'),
                                ('Multiview training', new_distance, '#2563eb')):
        ax.plot(np.sort(values), np.arange(1, len(values)+1)/len(values), color=color, label=name)
    ax.set(xlabel='Distance to nearest training profile', ylabel='Fraction of fixed test rays',
           title='Geometry-only profile coverage; all 280 fixed test rays')
    ax.legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'profile_coverage')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for kind, color in (('mixture', '#ea580c'), ('cdf', '#2563eb')):
        runs = [r for r in gradient['runs'] if r['kind'] == kind]
        histogram = np.array([np.sum([p['pit_counts'] for p in r['metrics']['test']['per_ray']], 0)/r['metrics']['test']['hits']*20 for r in runs])
        axes[0].plot((np.arange(20)+.5)/20, histogram.mean(0), color=color, label=kind)
        axes[0].fill_between((np.arange(20)+.5)/20, histogram.min(0), histogram.max(0), color=color, alpha=.15)
        binned = np.array([[b['pit_ks'] if b['pit_ks'] is not None else np.nan
                           for b in r['metrics']['test']['distance_bins']] for r in runs])
        axes[1].plot(np.arange(4), binned.mean(0), 'o-', color=color, label=kind)
        axes[1].fill_between(np.arange(4), binned.min(0), binned.max(0), color=color, alpha=.15)
        rows = [np.mean([r['metrics']['test']['per_ray'][i]['pit_ks'] for r in runs])
                for i in groups['all_280'] if runs[0]['metrics']['test']['per_ray'][i]['hits']]
        axes[2].plot(np.sort(rows), np.arange(1, len(rows)+1)/len(rows), color=color, label=kind)
    axes[0].axhline(1, ls='--', color='gray')
    axes[0].set(xlabel='Conditional PIT', ylabel='Density', title='At reference first-hit positions')
    axes[1].set_xticks(np.arange(4), ['0-.25', '.25-.5', '.5-.75', '.75-1'])
    axes[1].set(xlabel='q / horizon', ylabel='PIT KS', title='Position-dependent calibration')
    axes[2].set(xlabel='Per-ray PIT KS (seed mean)', ylabel='Fraction of hit-containing rays', title='Ray-level calibration')
    axes[0].legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'gradient_calibration')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in ('direct', 'residual'):
        curves = [read_json(output/'runs'/f'{mode}_{seed}'/'history.json') for seed in spec['seeds']]
        for j, history in enumerate(curves):
            axes[0].plot([h['epoch'] for h in history], [h['validation']['distance_nll'] for h in history],
                         color=colors[mode], alpha=.6, label=mode if j == 0 else None)
    for kind, color in (('mixture', '#ea580c'), ('cdf', '#2563eb')):
        for j, seed in enumerate(spec['seeds']):
            history = read_json(output/'gradient'/f'{kind}_{seed}'/'history.json')
            axes[1].plot([h['epoch'] for h in history], [h['validation']['speed_nll'] for h in history],
                         color=color, alpha=.6, label=kind if j == 0 else None)
    axes[0].set(xlabel='Epoch', ylabel='Validation distance NLL')
    axes[1].set(xlabel='Epoch', ylabel='Validation conditional speed NLL')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'training_curves')
    # Fixed examples: farthest geometry-selected ray from each earlier test view.
    selection = read_json(resolve(spec['fixed_holdout_directory'])/'selection.json')
    selected = selection['selected']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    cameras = list(dict.fromkeys(p['camera'] for p in selected))
    for ax, camera in zip(axes.flat, cameras):
        j = max((i for i, p in enumerate(selected) if p['camera'] == camera),
                key=lambda i: selected[i]['nearest_training_profile_distance'])
        i = len(groups['original_24'])+j
        p = test_raw[i]
        q = torch.linspace(0, p['horizon'], config['curve_points'], dtype=torch.float64)
        ref = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/len(p['hit'])
        ax.plot(q*p['geometry']['ell'], ref, 'k', lw=2, label='Reference')
        for mode in ('local', 'old_residual', 'residual'):
            curves = np.array([r['metrics']['per_ray'][i]['T'] for r in distance['results'][mode]])
            ax.plot(q*p['geometry']['ell'], curves.mean(0), color=colors[mode], label=labels[mode])
            ax.fill_between(q*p['geometry']['ell'], curves.min(0), curves.max(0), color=colors[mode], alpha=.15)
        ax.set_title(camera)
        ax.set(xlabel='Distance from entry', ylabel='Transmittance')
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(plt, fig, output, 'fixed_geometry_examples')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for column, failure in enumerate(summary['distance_failures'][:3]):
        i = failure['ray']
        p = test_raw[i]
        q = torch.linspace(0, p['horizon'], config['curve_points'], dtype=torch.float64)
        b, db = local_profile(p['features'], q)
        axes[0, column].plot(q*p['geometry']['ell'], b, color='#475569')
        axes[0, column].axhline(0, color='black', lw=.7)
        axes[0, column].set(title=f"Ray {i}\n{p['scene_id']}", ylabel='Mean / sigma')
        ref = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/len(p['hit'])
        axes[1, column].plot(q*p['geometry']['ell'], ref, 'k', lw=2, label='Reference')
        for mode in ('local', 'old_residual', 'residual'):
            curves = np.array([r['metrics']['per_ray'][i]['T'] for r in distance['results'][mode]])
            axes[1, column].plot(q*p['geometry']['ell'], curves.mean(0), color=colors[mode], label=labels[mode])
            axes[1, column].fill_between(q*p['geometry']['ell'], curves.min(0), curves.max(0), color=colors[mode], alpha=.15)
        axes[1, column].set(xlabel='Distance from entry', ylabel='Transmittance')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Failure diagnosis: three largest mean per-seed maximum T errors; selected after evaluation')
    fig.tight_layout()
    save_figure(plt, fig, output, 'distance_failures')
    distance_rows = []
    for group in groups:
        distance_rows.append([group, len(groups[group])]+[f"{distance['summary'][group][mode]['mean_ray_max_abs_T_error']['mean']:.6f}"
                                                          for mode in labels])
    dist_table = markdown_table(['组', '光线', *labels.values()], distance_rows)
    gradient_rows = [[kind]+[f"{summary['gradient_overall'][kind][metric]['mean']:.6f} ± {summary['gradient_overall'][kind][metric]['seed_std']:.6f}"
                             for metric in ('speed_nll', 'pit_ks', 'joint_nll')]
                     for kind in ('mixture', 'cdf')]
    grad_table = markdown_table(['梯度头', '条件速度 NLL', '全局 PIT KS', '联合 NLL'], gradient_rows)
    gradient_group_table = markdown_table(['组', 'Mixture 速度 NLL', 'CDF 速度 NLL', 'Mixture 逐光线 PIT KS', 'CDF 逐光线 PIT KS'],
        [[g]+[f"{gradient_summary[g][kind][metric]['mean']:.6f}"
              for metric in ('speed_nll', 'mean_ray_pit_ks') for kind in ('mixture', 'cdf')] for g in groups])
    epochs_table = markdown_table(['距离方案', '种子', '验证集所选 epoch', '参数量'],
        [[r['mode'], r['seed'], r['selected_epoch'], r['parameters']] for r in trained])
    grad_epochs_table = markdown_table(['梯度头', '种子', '验证集所选 epoch', '参数量'],
        [[r['kind'], r['seed'], r['selected_epoch'], r['parameters']] for r in gradient['runs']])
    timing_rows = [[target, kind, f"{timing['inverse_only']['microseconds_per_sample']:.3f}",
                    f"{timing['head_and_inverse']['microseconds_per_sample']:.3f}"]
                  for target, values in sampling['timings'].items() for kind, timing in values.items()]
    timing_table = markdown_table(['设备与批量', '梯度头', '仅逆采样 µs/样本', '头网络＋逆采样 µs/样本'], timing_rows)
    text = f'''# 多视角距离与条件梯度实验：sigma=0.02

数据：695 条训练光线、158 条验证光线、280 条固定测试光线；合计 4,040,704 个参考样本。
新增相机为 8 个训练视角和 2 个验证视角，均与固定测试相机分离。原有训练、验证、测试数据精确保留。
参考采样器仍为 H=1、B=8、步长 1/64 的 GPIS function-space Tungsten-linear 版本。

## 距离模型

Direct 和 local＋residual 均为 57,632 参数的整段剖面 MLP：256 输入采样点、32 维特征、64 个 log hazard 控制值、1024 个积分格子。
三个种子为 20261015、20261016、20261017；各训练 500 epochs，按验证集边缘距离 NLL 选择 checkpoint。
同种子的完整初始参数及采样顺序相同。Residual 额外使用原始三次剖面的解析 local hazard，比较的是完整混合方案。

以下为逐光线最大透射率误差的均值；神经网络条目平均三个种子。Old 为之前 183 条训练光线的固定 checkpoint。

{dist_table}

{markdown_table(['方案', '280 条距离 NLL', '280 条 T RMSE', '平均绝对逃逸概率误差'], [[labels[mode]]+[f"{distance['summary']['all_280'][mode][metric]['mean']:.6f}" for metric in ('distance_nll', 'rmse_T', 'escape_error')] for mode in labels])}

{epochs_table}

![距离误差](distance_comparison.png)
![训练剖面覆盖](profile_coverage.png)
![固定几何示例](fixed_geometry_examples.png)

平均指标改善不等于最坏情况改善。各种子最大 T 误差的均值，Local 为 {distance['summary']['all_280']['local']['worst_T_error']['mean']:.6f}，旧 residual 为 {distance['summary']['all_280']['old_residual']['worst_T_error']['mean']:.6f}，新 residual 为 {distance['summary']['all_280']['residual']['worst_T_error']['mean']:.6f}。
下面的失败案例是评估后按误差选择的诊断图，不用于训练或 checkpoint 选择。

![距离失败案例](distance_failures.png)

## 冻结距离模型后的梯度头

每个 residual checkpoint 接两个条件头，仅训练头部 150 epochs。两者使用同一冻结特征、真实首碰位置、入口符号、局部均值及导数；隐层初始化与样本顺序相同。
Mixture 使用 8 分量正半轴截断高斯；CDF 使用 8 区间单调有理二次样条 G，将指数基分布变换为 F(w)=G(1-exp(-w/s))。
CDF 头通过其导数的似然训练，具有可逆 CDF，并非用经验 CDF 的逐点回归损失。两者均拟合条件纵向速度 W，不把梯度和首碰距离视为独立变量。
这里 W=−k(ell/sigma) dot(omega,g)>0；当前实验不直接拟合三维梯度的联合 CDF。首碰位置由距离模型提供，梯度头在该位置的条件分布上采样。
CDF 与 mixture 的输出层大小、初始分布不同；这里是接近参数量的完整分布表示对照。

{grad_table}

{gradient_group_table}

{grad_epochs_table}

三个冻结主干参数逐项未变。PIT 仅在命中样本上评估；全局 PIT 可能掩盖不同光线间的偏差，因此同时保存逐光线和按 q/Q 分组结果。

![梯度校准](gradient_calibration.png)
![训练过程](training_curves.png)

## 联合采样与性能

预先指定第一种子，对 280 条测试光线各采样 8192 次。先采入口符号和距离，再条件采样 W。
比较 P(hit, q/Q≤a, W≤b) 的 15 个矩形；a 固定为 0.1、0.25、0.5、0.75、1，b 来自训练命中速度的四分位数。
两头使用相同距离样本；矩形误差同时受距离模型、梯度模型与 Monte Carlo 噪声影响。
单个矩形的两样本最坏标准差约为 {sampling['reference_and_model_worst_case_pointwise_mc_standard_deviation']:.6f}，这不是覆盖全部矩形的置信界。

{markdown_table(['梯度头', '平均矩形 CDF 绝对误差', '每光线最大矩形误差均值'], [[kind, f"{sampling['groups']['all_280'][kind]['mean_abs_joint_cdf_error']:.6f}", f"{sampling['groups']['all_280'][kind]['max_abs_joint_cdf_error']:.6f}"] for kind in ('mixture', 'cdf')])}

{timing_table}

计时环境：{sampling['torch_version']}，{sampling['gpu']}，CPU 单线程；7 次重复的中位数。Mixture 使用选分量＋截断高斯逆采样，无混合 CDF 迭代求逆。
计时使用现成条件特征与随机数，不含场访问、剖面构建、主干编码、local hazard 积分和距离采样；CUDA 数值为批量吞吐摊销。不能解释为完整渲染耗时。

## 解释范围

- 固定测试集此前已用于设计诊断，这轮不参与梯度或 checkpoint 选择；它不是全新的盲测集，也不是像素无偏样本。
- 本轮只在同一场景 sigma=0.02 上验证。新相机划分独立，但保留的原训练/验证/24 条测试仍来自同一个旧相机。
- 新旧模型都训练 500 epochs，但多视角版本每轮更新更多，验证集也扩大；不能把全部变化单独归因于数据多样性。
- 仍使用固定分辨率整段编码。前一轮已观察到前缀一致性问题；本轮结果不能证明与逐段 RNO/GRU 等价。
- 本轮保持 1024 格与 8 点求积以便比较；扩展测试域中此前 local 的 1024/2048 格最大 T 差约 0.002843。参考采样器也具有有限步长和有限记忆误差。
- 本轮 158 条验证光线的 local 网格加倍最大 T 差为 {summary['local_grid_audit']['max_grid_error']:.6f}，其中 {summary['local_grid_audit']['rays_above_previous_grid_tolerance']} 条超出旧验证集的 0.002 容差；8 点改为 2 点求积的最大差为 {summary['local_grid_audit']['max_two_point_quadrature_error']:.3g}。该检查不等于已验证残差模型可直接更换积分分辨率。
- 梯度评估对象是条件纵向速度；尚未接入完整三维法线采样与最终渲染对照。

完整数据：[summary.json](summary.json)、[距离逐光线结果](distance_evaluation.json)、[梯度逐光线结果](gradient_evaluation.json)、[联合采样与计时](sampling_evaluation.json)、[数据审计](data_audit.json)。

数据 SHA-256：`{audit['dataset_sha256']}`。
'''
    (output/'summary.md').write_text(text, encoding='utf-8')
    # An HTML gallery stays dependency-free; the complete narrative is Markdown.
    document = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>sigma=0.02 多视角实验</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px}}img{{width:100%}}pre{{white-space:pre-wrap;line-height:1.6}}</style>
<h1>sigma=0.02 多视角实验</h1><p><a href="summary.md">完整 Markdown 报告</a> · <a href="summary.json">数值汇总</a></p>
<pre>{html.escape(dist_table+'\n\n'+grad_table+'\n\n'+timing_table)}</pre>
{''.join(f'<img src="{name}.png">' for name in ('distance_comparison', 'profile_coverage', 'fixed_geometry_examples', 'distance_failures', 'gradient_calibration', 'training_curves'))}</html>'''
    (output/'summary.html').write_text(document, encoding='utf-8')
    return {'report': str(output/'summary.md'), 'summary': str(output/'summary.json')}

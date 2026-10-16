"""Reproducible plots and conditional-only summaries for the step experiment."""
import html
import math
from pathlib import Path

import numpy as np

from renewal.collect import digest, write_json
from .camera_study import plotting, save_figure
from .causal_gradient_report import LABELS, COLORS, moments
from .data import read_json
from .residual_supervision_report import table


def scores(metrics, minimum, indices=None):
    ray = metrics['per_ray'] if indices is None else [metrics['per_ray'][i] for i in indices]
    valid = sorted(r['pit_ks'] for r in ray if r['hits'] >= minimum)
    hits = sum(r['hits'] for r in ray)
    result = {'speed_nll': sum(r['speed_nll']*r['hits'] for r in ray if r['hits'])/hits,
        'mean_ray_pit_ks': float(np.mean(valid)), 'p95_ray_pit_ks': float(np.quantile(valid, .95)),
        'worst_decile_ray_pit_ks': float(np.mean(valid[-max(1, math.ceil(.1*len(valid))):])),
        'max_ray_pit_ks': max(valid), 'eligible_rays': len(valid), 'hits': hits}
    if indices is None:
        result.update({k: metrics[k] for k in
                       ('speed_nll', 'pit_ks', 'coverage_68', 'coverage_90', 'lower_tail_01', 'upper_tail_99')})
        result['late_pit_ks'] = metrics['distance_bins'][-1]['pit_ks']
        result['late_speed_nll'] = metrics['distance_bins'][-1]['speed_nll']
        result['late_hits'] = metrics['distance_bins'][-1]['hits']
    return result


def audit(spec, parent, evaluation, output):
    protocol = read_json(output/'protocol.json')
    if evaluation['protocol_sha256'] != digest(output/'protocol.json'):
        raise AssertionError('evaluation protocol changed')
    for key in ('source_sha256', 'checkpoint_sha256'):
        for path, expected in protocol[key].items():
            if digest(Path(path)) != expected:
                raise AssertionError(f'locked input changed: {path}')
    expected = {(seed, v['name'], step, 'full') for seed in parent['seeds']
                for v in parent['variants'] for step in spec['steps']}
    expected.update((spec['ablation_seed'], v['name'], step, mode) for v in parent['variants']
                    for step in spec['ablation_steps'] for mode in ('state_only', 'phase_only'))
    actual = [(r['identity']['seed'], r['identity']['method'], r['identity']['step'], r['identity']['mode'])
              for r in evaluation['records']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError('missing or duplicate step evaluations')
    for r in evaluation['records']:
        if r['identity']['protocol_sha256'] != evaluation['protocol_sha256']:
            raise AssertionError('inconsistent run identity')
    differences = [r['baseline_max_metric_difference'] for r in evaluation['records']
                   if r['identity']['step'] == spec['training_step'] and r['identity']['mode'] == 'full']
    if len(differences) != len(parent['seeds'])*len(parent['variants']) or max(differences) > spec['baseline_tolerance']:
        raise AssertionError('baseline reproduction failed')
    if (len(evaluation['weight_checks']) != len(parent['seeds'])
            or not all(r['unchanged'] for r in evaluation['weight_checks'])):
        raise AssertionError('frozen weight checks failed')
    return {'passed': True, 'full_evaluations': len(parent['seeds'])*len(parent['variants'])*len(spec['steps']),
            'diagnostic_interventions': len(expected)-len(parent['seeds'])*len(parent['variants'])*len(spec['steps']),
            'baseline_max_metric_difference': max(differences), 'unchanged_weights_and_input_files': True}


def summarize(spec, parent, evaluation):
    records = evaluation['records']
    minimum = spec['minimum_ray_hits']
    baseline = {(r['identity']['method'], r['identity']['seed']): scores(r['metrics'], minimum)
                for r in records if r['identity']['step'] == spec['training_step'] and r['identity']['mode'] == 'full'}
    rows, ablations = [], []
    for variant in parent['variants']:
        name = variant['name']
        for step in spec['steps']:
            runs = [r for r in records if r['identity']['method'] == name
                    and r['identity']['step'] == step and r['identity']['mode'] == 'full']
            values = [scores(r['metrics'], minimum) for r in runs]
            paired = [{k: v[k]-baseline[name, r['identity']['seed']][k] for k in v} for v, r in zip(values, runs)]
            groups = {group: [scores(r['metrics'], minimum, indices) for r in runs]
                      for group, indices in evaluation['groups'].items()}
            rows.append({'method': name, 'step': step, 'step_ratio': step/spec['training_step'],
                'metrics': {k: moments([v[k] for v in values]) for k in values[0]},
                'paired_delta': {k: moments([v[k] for v in paired]) for k in paired[0]},
                'groups': {group: {k: moments([v[k] for v in vs]) for k in vs[0]} for group, vs in groups.items()},
                'seeds': [{'seed': r['identity']['seed'], 'metrics': v} for r, v in zip(runs, values)]})
    for r in records:
        if r['identity']['mode'] != 'full':
            ablations.append({**r['identity'], 'metrics': scores(r['metrics'], minimum)})
    return rows, ablations


def figures(spec, parent, evaluation, rows, ablations, output):
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in zip(axes.flat,
            ('speed_nll', 'mean_ray_pit_ks', 'p95_ray_pit_ks', 'late_pit_ks'),
            ('Conditional speed NLL', 'Mean ray PIT KS', '95th percentile ray PIT KS', 'Late hits PIT KS (q/Q >= 3/4)')):
        for variant in parent['variants']:
            name = variant['name']
            selected = [r for r in rows if r['method'] == name]
            ax.errorbar([r['step_ratio'] for r in selected], [r['metrics'][key]['mean'] for r in selected],
                yerr=[r['metrics'][key]['seed_std'] for r in selected], color=COLORS[name], capsize=3,
                fmt='o--' if name == 'mixture_local' else 'o-', label=LABELS[name])
        ax.axvline(1., color='black', ls=':', lw=1.)
        ax.set(title=title, xlabel='Internal step / training step', xticks=[s/spec['training_step'] for s in spec['steps']])
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Frozen heads and GRU; fixed reference hits; three-seed mean +/- seed SD')
    fig.tight_layout()
    save_figure(plt, fig, output, 'step_accuracy')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, method in enumerate(spec['example_methods']):
        selected = [(r['identity']['step'], r['identity']['mode'], scores(r['metrics'], spec['minimum_ray_hits']))
                    for r in evaluation['records'] if r['identity']['seed'] == spec['ablation_seed']
                    and r['identity']['method'] == method]
        for col, key in enumerate(('mean_ray_pit_ks', 'late_pit_ks')):
            ax = axes[row, col]
            base = next(m[key] for step, mode, m in selected if step == spec['training_step'] and mode == 'full')
            for j, (mode, color) in enumerate((('full', '#059669'), ('state_only', '#ea580c'), ('phase_only', '#2563eb'))):
                values = [next(m[key] for st, md, m in selected if st == step and md == mode) for step in spec['ablation_steps']]
                ax.bar(np.arange(len(values))+.22*(j-1), values, width=.21, color=color, label=mode)
            ax.axhline(base, ls='--', color='black', label='Training-step baseline')
            ax.set_xticks(np.arange(len(spec['ablation_steps'])), [f'{s/spec["training_step"]:g}x' for s in spec['ablation_steps']])
            ax.set(title=LABELS[method]+' / '+key, xlabel='Internal step / training step')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('First-seed mechanism diagnostics; hybrid inputs are NOT deployment modes')
    fig.tight_layout()
    save_figure(plt, fig, output, 'step_mechanism')
    for method in spec['example_methods']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, ray in zip(axes.flat, spec['example_rays']):
            for j, step in enumerate(spec['example_steps']):
                record = next(r for r in evaluation['records'] if r['identity']['method'] == method
                    and r['identity']['seed'] == spec['example_seed'] and r['identity']['step'] == step and r['identity']['mode'] == 'full')
                example = next(e for e in record['examples'] if e['ray'] == ray)
                if j == 0:
                    ax.plot(evaluation['speed_grid'], example['reference_cdf'], color='black', lw=2, label='Reference')
                ax.plot(evaluation['speed_grid'], example['predicted_cdf'], label=f'{step/spec["training_step"]:g}x step')
            ax.set(title=f'Ray {ray}: {example["hits"]} reference hits', xlabel='Normalized speed W', ylabel='Averaged conditional CDF')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(LABELS[method]+'; first seed; SAME reference locations and entry signs')
        fig.tight_layout()
        save_figure(plt, fig, output, 'step_cdfs_'+method)


def report(spec, parent, output):
    evaluation = read_json(output/'evaluation.json')
    checks = audit(spec, parent, evaluation, output)
    rows, ablations = summarize(spec, parent, evaluation)
    figures(spec, parent, evaluation, rows, ablations, output)
    headers = ['方案', '步长 Δq', '倍率', '条件 NLL', '逐光线 PIT KS', 'P95 光线 PIT KS', '晚碰撞 PIT KS']
    entries = [[LABELS[r['method']], f'{r["step"]:g}', f'{r["step_ratio"]:g}×']+
               [f'{r["metrics"][k]["mean"]:.6f}' for k in ('speed_nll', 'mean_ray_pit_ks', 'p95_ray_pit_ks', 'late_pit_ks')]
               for r in rows]
    intervention_rows = []
    for variant in parent['variants']:
        method = variant['name']
        for step in [spec['training_step']]+spec['ablation_steps']:
            modes = ('full',) if step == spec['training_step'] else ('full', 'state_only', 'phase_only')
            for mode in modes:
                r = next(r for r in evaluation['records'] if r['identity']['method'] == method
                         and r['identity']['seed'] == spec['ablation_seed'] and r['identity']['step'] == step and r['identity']['mode'] == mode)
                value = scores(r['metrics'], spec['minimum_ray_hits'])
                intervention_rows.append([LABELS[method], f'{step/spec["training_step"]:g}×', mode]+
                    [f'{value[k]:.6f}' for k in ('speed_nll', 'mean_ray_pit_ks', 'late_pit_ks')])
    baseline = next(r for r in rows if r['method'] == 'mixture_gru' and r['step'] == spec['training_step'])
    observations = []
    for r in rows:
        if r['method'] in spec['example_methods'] and r['step'] in spec['ablation_steps']:
            base = next(b for b in rows if b['method'] == r['method'] and b['step'] == spec['training_step'])
            observations.append(f'- {LABELS[r["method"]]} 在 {r["step_ratio"]:g}× 步长下，条件 NLL 相对训练步长变化 '
                f'{r["paired_delta"]["speed_nll"]["mean"]:+.6f} nat/命中；逐光线 PIT KS 为 '
                f'{r["metrics"]["mean_ray_pit_ks"]["mean"]/base["metrics"]["mean_ray_pit_ks"]["mean"]:.2f} 倍，'
                f'晚碰撞 PIT KS 为 {r["metrics"]["late_pit_ks"]["mean"]/base["metrics"]["late_pit_ks"]["mean"]:.2f} 倍。')
    figure_names = ['step_accuracy', 'step_mechanism']+['step_cdfs_'+m for m in spec['example_methods']]
    text = f'''# 梯度头的内部步长精度测试：sigma=0.02

冻结已有的 3 个距离 GRU 与 9 个梯度头，比较内部步长；没有重新训练或根据测试集选择 checkpoint。
完整评估 {checks['full_evaluations']} 组，另外完成 {checks['diagnostic_interventions']} 组第一种子状态/位置消融。

**当前梯度头对内部 GRU 步长明显敏感，减小步长不会自动提高拟合精度。** 本轮两类 GRU 头在训练步长下的平均条件 NLL 和逐光线 PIT KS 均最佳；CDF 头同样受影响。
第一种子的诊断将主要退化定位到历史状态变化，单独改变 phase 的影响很小。当前渲染接入应保持训练时的内部 Δq=1/8，并保留跨外部分段的状态和步内进度。

## 固定条件和测试范围

- 使用上一轮相同的 {evaluation['reference_rays']} 条测试光线、{evaluation['reference_hits']:,} 次参考首碰。
- 测试对象为条件速度 `W = -k (ell/sigma) dot(omega, g) > 0`，每次固定参考位置 q、入口符号 k、参考速度 W 和原始三次均值剖面。
- `q=s/ell`，训练内部步长 Δq={spec['training_step']}；本轮为 {', '.join(str(s) for s in spec['steps'])}。
- 距离积分步长固定为 {next(iter(evaluation['backbone_configs'].values()))['integration_step']}，积分阶数固定为 {next(iter(evaluation['backbone_configs'].values()))['quadrature_order']}；参考采样器也保持不变。
- 逐光线指标要求至少 {spec['minimum_ray_hits']} 次命中，保留 {int(baseline['metrics']['eligible_rays']['mean'])} 条光线。晚碰撞定义为 q/Q≥3/4，共 {int(baseline['metrics']['late_hits']['mean']):,} 次命中。
- 表格先分别计算三个种子的指标再取平均；图中误差棒为种子标准差，不是置信区间。配对指标变化、逐种子和各光线分组见 JSON。

## 条件拟合结果

{table(headers, entries)}

所有 NLL、PIT KS 指标越小越好。PIT 检查 `F_pred(W_reference)` 的均匀性；其值包含有限参考样本噪声，不能解释为无噪声逐点真值误差。

{chr(10).join(observations)}

![步长和拟合精度](step_accuracy.png)

## 历史状态和步内位置的诊断

梯度头使用当前已完成区间的历史状态 z，以及步内进度 phase=(q-cell·Δq)/Δq。改变 Δq 会同时改变二者；局部 b(q)、b'(q)、q、入口状态保持相同。

- `full`：按照新步长重算状态和 phase，是主要步长测试。
- `state_only`：新步长状态配训练步长 phase；`phase_only`：训练步长状态配新步长 phase。
- 两种混合输入只用于识别敏感来源，不能直接用作渲染方案；非线性影响不要求可相加。
- 无历史头仍接收 phase，因此也可能随步长变化。只改状态时，它必须精确复现基线，脚本已检查。

{table(['方案', '倍率', '诊断模式', '条件 NLL', '逐光线 PIT KS', '晚碰撞 PIT KS'], intervention_rows)}

![机制诊断](step_mechanism.png)

这个来源判断只在第一种子做了消融。主测试的晚碰撞指标存在种子差异，例如半步长时有个别种子的晚碰撞 PIT 略有改善；总体结论来自三个种子的平均与逐光线结果，不代表每个条件分组都必然变差。

## 固定困难光线

光线 {', '.join(map(str, spec['example_rays']))} 来自上一轮已知案例，本轮评估前固定；未用它们选择新的模型。
以下 CDF 在相同参考位置和入口符号上平均，并与参考速度的经验 CDF 比较。这是条件分布混合后的比较，仍需结合逐光线 PIT 和 NLL。

![GRU Mixture CDF](step_cdfs_mixture_gru.png)

![GRU CDF](step_cdfs_cdf_gru.png)

## 复现与边界

```powershell
python -B python/first_hit_se_cli.py causal-gradient-step --config python/configs/first_hit_se_causal_gradient_step_sigma002.json
python -B -m unittest python/tests/test_causal_gradient_step.py -v
```

支持 `--phase evaluate/report`；逐组结果可复用。数据、原训练来源、checkpoint、新评估源码和实验配置均通过 protocol 绑定。
所有训练步长复测与上一轮条件指标的最大差值为 **{checks['baseline_max_metric_difference']:.3g}**，小于 {spec['baseline_tolerance']:g}；全程模型参数哈希与输入文件保持一致。

这是固定参考碰撞点上的梯度头测试，未重新采样距离，也未测完整渲染误差、法线或完整三维梯度。
这里改变的是内部 GRU 推进网格。外部追踪、查询或几何分段步长改变时，若保持原内部网格和 phase 定义，则不等价于本测试的内部步长变化。
测试结果本身不构成新训练步长的选择依据。要使网络适应不同内部步长，应另外做多步长训练或具有步长约束的状态更新，再用独立数据验证。
现有主干输入已经包含区间宽度，但固定步长训练仍未获得步长不变性；仅增加步长输入、固定 phase 或更换为 CDF 头都不能据此视为已解决。
'''
    summary = {'audit': checks, 'scope': read_json(output/'protocol.json')['scope'], 'rows': rows,
        'ablations': ablations, 'reference_rays': evaluation['reference_rays'], 'reference_hits': evaluation['reference_hits'],
        'report_source_sha256': digest(Path(__file__)),
        'artifact_sha256': {name: digest(output/name) for name in ('protocol.json', 'evaluation.json')}}
    write_json(output/'summary.json', summary)
    (output/'summary.md').write_text(text, encoding='utf-8')
    cells = '<thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'
    cells += ''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in entries)+'</tbody>'
    images = ''.join(f'<figure><a href="{n}.svg"><img src="{n}.png" alt="{n}"></a></figure>' for n in figure_names)
    page = '<!doctype html><meta charset="utf-8"><title>Gradient head step accuracy</title><style>body{max-width:1400px;margin:24px auto;padding:0 16px;font:16px system-ui}img{max-width:100%}pre{white-space:pre-wrap;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px 14px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}</style>'
    page += '<h1>梯度头的内部步长精度测试</h1><p><a href="../summary.html">返回梯度拟合</a> · <a href="summary.md">Markdown</a> · <a href="summary.json">JSON</a></p>'
    page += '<p>冻结模型；固定参考碰撞点；三种子平均。此处步长指内部 GRU 推进步长。</p><table>'+cells+'</table>'+images+'<pre>'+html.escape(text)+'</pre>'
    (output/'summary.html').write_text(page, encoding='utf-8')
    return {'summary': str(output/'summary.html'), 'audit': checks,
            'mean_ray_pit_ks': {v['name']: {str(r['step']): r['metrics']['mean_ray_pit_ks']['mean']
                for r in rows if r['method'] == v['name']} for v in parent['variants']}}

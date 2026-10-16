"""Reproducible report for causal residual and matched-capacity controls."""
import html
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .camera_study import plotting, save_figure
from .causal_data import query_model
from .causal_residual import CausalConfig
from .data import read_json
from .hazard import local_bin_log_rates, piecewise_distribution
from .residual_supervision_report import table
from .spline import local_profile


LABELS = {'local_mlp_grid': 'Local / old grid', 'local_causal_grid': 'Local / quadrature',
    'mlp_nll': 'MLP / NLL', 'mlp_T': 'MLP / L2+T', 'local_head_T': 'Local head / 993',
    'local_matched_T': 'Local head / 3009', 'gru_nll': 'GRU / NLL', 'gru_T': 'GRU / L2+T'}
COLORS = dict(zip(LABELS, ('#94a3b8', '#475569', '#ea580c', '#a16207', '#93c5fd', '#2563eb', '#9333ea', '#059669')))


@torch.no_grad()
def off_grid_audit(spec, data, output):
    """513 curve points coincide with old bin edges; also probe between them."""
    config = CausalConfig(**spec['model'])
    base = read_json(resolve(spec['baseline_directory'])/'training_config.json')['model']
    identity = {'dataset_sha256': spec['dataset_sha256'], 'model': spec['model'],
                'old_bins': base['bins'], 'old_quadrature': base['quadrature_order'], 'query_points': 4099}
    path = output/'off_grid_local_audit.json'
    if path.exists():
        result = read_json(path)
        if result['identity'] != identity:
            raise AssertionError('off-grid audit identity changed')
        return result
    rows = []
    for i, p in enumerate(p for p in data['profiles'] if p['split'] == 'validation'):
        f = p['features']
        Q = f[:, 4].exp().sum()
        q = torch.linspace(0, 1, 4099, dtype=torch.float64)*Q
        exact = query_model(None, f, q, config)['T']
        rates = local_bin_log_rates(f, base['bins'], base['quadrature_order'])
        H = piecewise_distribution(rates, q[None], Q)[0]
        prior = torch.special.log_ndtr(f[0, 0]*f.new_tensor([1., -1.]))
        prior -= torch.logsumexp(prior, 0)
        old = torch.logsumexp(prior[:, None]-H, 0).exp()
        rows.append({'ray': i, 'max_T_difference': float((exact-old).abs().max())})
    result = {'identity': identity, 'scope': 'validation only; local quadrature versus old piecewise-constant hazard',
              'max_T_difference': max(r['max_T_difference'] for r in rows), 'rows': rows}
    write_json(path, result)
    return result


def audit(spec, output):
    checks, run_rows = [], []
    previous = read_json(resolve(spec['supervision_directory'])/'protocol.json')
    for path, expected in previous['training_source_sha256'].items():
        if digest(Path(path)) != expected:
            raise AssertionError(f'previous baseline probability/training implementation changed: {path}')
    for directory in (output, output/'capacity_control'):
        protocol, trained = (read_json(directory/name) for name in ('protocol.json', 'training.json'))
        evaluation = read_json(directory/'evaluation.json')
        if (trained['protocol_sha256'] != digest(directory/'protocol.json') or trained['test_metrics_used']
                or evaluation['training_sha256'] != digest(directory/'training.json')):
            raise AssertionError('selection or evaluation identity failed')
        if digest(resolve(spec['dataset'])) != protocol['dataset_sha256']:
            raise AssertionError('dataset changed')
        for field in ('training_source_sha256', 'baseline_checkpoint_sha256'):
            for path, expected in protocol[field].items():
                if digest(Path(path)) != expected:
                    raise AssertionError(f'{field} changed: {path}')
        for record in trained['runs']:
            path = Path(record['checkpoint'])
            history = read_json(path.parent/'history.json')
            good = (history[-1]['epoch'] == spec['epochs'] and
                    record['selected_epoch'] == min(history, key=lambda h: h['validation']['distance_nll'])['epoch'] and
                    digest(path) == record['checkpoint_sha256'] == evaluation['checkpoint_sha256'][str(path)])
            if not good:
                raise AssertionError('run audit failed')
            run_rows.append(record)
            checks.append({'name': record['identity']['variant']['name'], 'seed': record['identity']['seed'], 'passed': True})
        for seed in spec['seeds']:
            if len({r['initial_parameter_sha256'] for r in trained['runs'] if r['identity']['seed'] == seed}) != 1:
                raise AssertionError('paired initialization failed')
    if len(checks) != 12:
        raise AssertionError('expected 12 completed runs including capacity control')
    return {'passed': True, 'previous_training_sources_unchanged': True, 'runs': checks}, run_rows


def example_figure(ids, raw, results, output, plt, name, title):
    fig, axes = plt.subplots(2, len(ids), figsize=(4.6*len(ids), 7), squeeze=False)
    for column, index in enumerate(ids):
        p = raw[index]
        q = torch.linspace(0, p['horizon'], 513, dtype=torch.float64)
        distance = q.numpy()*p['geometry']['ell']
        mean, _ = local_profile(p['features'], q)
        axes[0, column].plot(distance, mean, color='#475569')
        axes[0, column].axhline(0, color='black', lw=.7)
        axes[0, column].set(title=f'Ray {index}', ylabel='Mean / sigma')
        ref = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/len(p['hit'])
        axes[1, column].plot(distance, ref, 'k', lw=2, label='Reference')
        for method in ('local_causal_grid', 'mlp_T', 'local_matched_T', 'gru_nll', 'gru_T'):
            values = np.array([r['metrics']['per_ray'][index]['T'] for r in results[method]])
            axes[1, column].plot(distance, values.mean(0), color=COLORS[method], label=LABELS[method])
            axes[1, column].fill_between(distance, values.min(0), values.max(0), color=COLORS[method], alpha=.12)
        axes[1, column].set(xlabel='Distance from medium entry', ylabel='Transmittance', ylim=(-.02, 1.02))
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, -.01), ncol=6, fontsize=8)
    fig.suptitle(title+'; shading: range over three seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, name)


def figures(spec, output, evaluation, diagnostics, raw, runs, worst):
    plt = plotting()
    groups = list(evaluation['groups'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, title in zip(axes, ('mean_ray_max_abs_T_error', 'worst_decile_mean', 'max_abs_T_error'),
                               ('Mean per-ray max |T error|', 'Mean of worst 10% rays', 'Worst ray per seed')):
        for j, method in enumerate(LABELS):
            values = [evaluation['summary'][g][method][metric] for g in groups]
            ax.bar(np.arange(len(groups))+.1*j, [v['mean'] for v in values], width=.095,
                   yerr=[v['seed_std'] for v in values], color=COLORS[method], label=LABELS[method])
        ax.set_xticks(np.arange(len(groups))+.35, ['All 280', 'Original 24', 'Spatial 128', 'Diverse 128'], rotation=15)
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, -.02), ncol=4, fontsize=9)
    fig.suptitle('Fixed diagnostic rays; error bars: standard deviation over three training seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, 'distance_comparison')
    kinds = ('prefix_0.5', 'prefix_0.75', 'future_change', 'repartition')
    methods = [m for m in LABELS if m in diagnostics['summary'][kinds[0]]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, method in enumerate(methods):
        values = [max(1e-16, diagnostics['summary'][k][method]['max']) for k in kinds]
        ax.bar(np.arange(len(kinds))+.12*j, values, width=.115, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(np.arange(len(kinds))+.3, ['Cut at 1/2', 'Cut at 3/4', 'Change future', 'Split original cubics'])
    ax.set(yscale='log', ylabel='Maximum T difference over 280 rays', title='First seed; display floor 1e-16')
    ax.axhline(spec['diagnostics']['causal_tolerance'], color='black', ls='--', lw=.8, label='Causal tolerance')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(plt, fig, output, 'prefix_consistency')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for method, values in diagnostics['step_accuracy'].items():
        steps = sorted(map(float, values))
        axes[0].plot(steps, [values[str(s)]['mean_ray_max_abs_T_error'] for s in steps], 'o-', color=COLORS[method], label=LABELS[method])
        axes[1].plot(steps, [diagnostics['summary'][f'step_{s}'][method]['max'] for s in steps], 'o-', color=COLORS[method])
    axes[0].set(xlabel='Recurrent step in q = s / ell', ylabel='Mean per-ray max |T error|', title='First seed, no retraining')
    axes[1].set(xlabel='Recurrent step in q = s / ell', ylabel='Maximum T change from step=1/8', title='Changing the GRU step is a different operation')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(plt, fig, output, 'step_sensitivity')
    example_figure(spec['diagnostics']['fixed_failure_rays'], raw, evaluation['results'], output, plt,
                   'fixed_failures', 'Failure rays fixed before this experiment')
    example_figure(worst, raw, evaluation['results'], output, plt, 'remaining_failures', 'Largest GRU + L2+T errors; selected after evaluation')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in runs:
        name, seed = record['identity']['variant']['name'], record['identity']['seed']
        history = read_json(Path(record['checkpoint']).parent/'history.json')
        for ax, metric in zip(axes, ('distance_nll', 'mean_ray_max_abs_T_error')):
            ax.plot([h['epoch'] for h in history], [h['validation'][metric] for h in history],
                    color=COLORS[name], alpha=.6, label=LABELS[name] if seed == spec['seeds'][0] else None)
    axes[0].set(xlabel='Epoch', ylabel='Validation marginal NLL')
    axes[1].set(xlabel='Epoch', ylabel='Validation mean per-ray max |T error|')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(plt, fig, output, 'training_curves')


def report(spec, data, output):
    checks, runs = audit(spec, output)
    evaluation = read_json(output/'evaluation.json')
    capacity = read_json(output/'capacity_control'/'evaluation.json')
    diagnostics = read_json(output/'diagnostics.json')
    off_grid = off_grid_audit(spec, data, output)
    if (not diagnostics['passed'] or diagnostics['evaluation_sha256'] != digest(output/'evaluation.json')
            or diagnostics['training_sha256'] != digest(output/'training.json')
            or diagnostics['source_sha256'] != digest(Path(__file__).with_name('causal_diagnostics.py'))):
        raise AssertionError('diagnostics are missing or stale')
    if diagnostics['capacity_evaluation_sha256'] != digest(output/'capacity_control'/'evaluation.json'):
        raise AssertionError('capacity-control diagnostics are stale')
    evaluation['results']['local_matched_T'] = capacity['results']['local_matched_T']
    for group in evaluation['summary']:
        evaluation['summary'][group]['local_matched_T'] = capacity['summary'][group]['local_matched_T']
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    errors = {method: np.array([[p['max_abs_T_error'] for p in r['metrics']['per_ray']] for r in values])
              for method, values in evaluation['results'].items()}
    worst = [int(i) for i in np.argsort(errors['gru_T'].mean(0))[-3:][::-1]]
    def example(i):
        return {'ray': i, 'source': raw[i]['source'], 'scene_id': raw[i]['scene_id'],
                'mean_max_T_error': {method: float(values[:, i].mean()) for method, values in errors.items()}}
    all_scores = evaluation['summary']['all_280']
    changes = {method: {baseline: {metric: 100*(all_scores[method][metric]['mean']/all_scores[baseline][metric]['mean']-1)
        for metric in ('mean_ray_max_abs_T_error', 'worst_decile_mean', 'max_abs_T_error', 'rmse_T')}
        for baseline in ('mlp_nll', 'mlp_T', 'local_matched_T')} for method in ('gru_nll', 'gru_T')}
    validation_nll = {name: float(np.mean([r['validation']['distance_nll'] for r in runs
        if r['identity']['variant']['name'] == name])) for name in ('gru_nll', 'gru_T')}
    preferred = min(validation_nll, key=validation_nll.get)
    summary = {'audit': checks, 'runs': runs, 'distance': evaluation['summary'], 'changes_percent': changes,
        'gru_validation_nll': validation_nll, 'preferred_gru_by_mean_validation_nll': preferred,
        'fixed_failures': [example(i) for i in spec['diagnostics']['fixed_failure_rays']],
        'remaining_failures': [example(i) for i in worst], 'diagnostics': {k: v for k, v in diagnostics.items()
             if k not in ('records', 'step_per_ray', 'streaming', 'quadrature', 'local_grid_difference')},
        'streaming_max_T_difference': diagnostics['streaming']['max_T_difference'],
        'quadrature_max_T_difference': diagnostics['quadrature']['max_T_difference'],
        'local_grid_difference_max': diagnostics['local_grid_difference']['max'],
        'off_grid_local_difference_max': off_grid['max_T_difference'],
        'dataset_sha256': spec['dataset_sha256'], 'report_source_sha256': digest(Path(__file__)),
        'artifact_sha256': {str(p.relative_to(output)): digest(p) for p in (
            output/'protocol.json', output/'training.json', output/'evaluation.json', output/'diagnostics.json', output/'off_grid_local_audit.json',
            output/'capacity_control'/'protocol.json', output/'capacity_control'/'training.json', output/'capacity_control'/'evaluation.json')}}
    write_json(output/'summary.json', summary)
    figures(spec, output, evaluation, diagnostics, raw, runs, worst)
    metrics = ('mean_ray_max_abs_T_error', 'p95_ray_max_abs_T_error', 'worst_decile_mean', 'max_abs_T_error', 'rmse_T', 'distance_nll')
    result_table = table(['方案', '平均最大 T 误差', 'P95', '最差 10% 均值', '最坏光线', 'T RMSE', '距离 NLL'],
        [[LABELS[m]]+[f"{all_scores[m][k]['mean']:.6f}" for k in metrics] for m in LABELS])
    group_table = table(['组', *[LABELS[m] for m in ('mlp_T', 'local_matched_T', 'gru_nll', 'gru_T')]],
        [[g]+[f"{evaluation['summary'][g][m]['mean_ray_max_abs_T_error']['mean']:.6f}" for m in ('mlp_T', 'local_matched_T', 'gru_nll', 'gru_T')]
         for g in evaluation['groups']])
    consistency_table = table(['方案', '截断 1/2', '截断 3/4', '修改后半段', '四倍细分原剖面'],
        [[LABELS[m]]+[f"{diagnostics['summary'][kind][m]['max']:.3g}" for kind in ('prefix_0.5', 'prefix_0.75', 'future_change', 'repartition')]
         for m in LABELS if m in diagnostics['summary']['prefix_0.5']])
    examples_table = table(['光线', *[LABELS[m] for m in ('local_causal_grid', 'mlp_T', 'local_matched_T', 'gru_nll', 'gru_T')]],
        [[r['ray']]+[f"{r['mean_max_T_error'][m]:.6f}" for m in ('local_causal_grid', 'mlp_T', 'local_matched_T', 'gru_nll', 'gru_T')]
         for r in summary['fixed_failures']])
    step_table = table(['方案', 'Δq', '平均最大 T 误差', '最坏光线'],
        [[LABELS[m], step, f"{v['mean_ray_max_abs_T_error']:.6f}", f"{v['max_abs_T_error']:.6f}"]
         for m, steps in diagnostics['step_accuracy'].items() for step, v in steps.items()])
    timing_table = table(['方案', 'CPU 已准备 µs/光线', 'CPU 从三次剖面 µs/光线', 'CUDA batch128 µs/光线'],
        [[LABELS[m]]+[f"{values[k]['median_us_per_ray']:.3f}" for k in ('cpu_prepared_curve', 'cpu_from_cubics_curve', 'cuda_128_prepared_curve')]
         for m, values in diagnostics['benchmark']['timings'].items()])
    epochs_table = table(['方案', '种子', '所选 epoch', '有效参数 / 已分配参数'],
        [[r['identity']['variant']['name'], r['identity']['seed'], r['selected_epoch'], f"{r['active_parameters']} / {r['parameters']}"] for r in runs])
    findings = '\n'.join(f"- {LABELS[m]} 相对上一轮 MLP＋L2＋T：平均最大 T 误差变化 {changes[m]['mlp_T']['mean_ray_max_abs_T_error']:+.1f}%，"
        f"最差 10% 均值变化 {changes[m]['mlp_T']['worst_decile_mean']:+.1f}%，最坏误差变化 {changes[m]['mlp_T']['max_abs_T_error']:+.1f}%。"
        for m in ('gru_nll', 'gru_T'))
    text = f'''# 轻量逐段残差 GRU：sigma=0.02

完成四个方案 × 三个种子的 12 组训练，每组 500 epochs。整段 MLP 的两个对照直接复用上一轮 checkpoint。

{findings}

相同损失下，GRU＋L2＋T 相对有效参数接近的无历史头，平均最大 T 误差变化 {changes['gru_T']['local_matched_T']['mean_ray_max_abs_T_error']:+.1f}%，支持历史状态的作用。
两种 GRU 的平均验证 NLL 为：NLL 方案 {validation_nll['gru_nll']:.6f}，L2＋T 方案 {validation_nll['gru_T']:.6f}；按此规则优先保留 `{preferred}` 作为后续基线。
这里仅检查了沿用 MLP 的一组正则与 T 监督权重，不代表其他权重无效。

因果结构的前缀一致性检查通过，但直接改变 GRU 的推进步长会明显改变结果。参数更少也未带来 CPU 整条曲线查询加速；下文分别列出步长和计时结果。

## 表示与因果约束

固定 `Δq=1/8`，其中 `q=s/ell`。历史状态 16 维，段编码 16 维，残差头宽 32，总参数 3,025；原整段 MLP 为 57,632。

```text
h_0 = Initial(signed b(0), signed b'(0))
delta_i = Head(h_i, signed b(q_i), signed b'(q_i), log(1+q_i))
lambda_k(q) = lambda_local,k(q) * exp(delta_k,i), q in interval i
h_(i+1) = GRU(Encode(completed interval i), h_i)
```

当前段修正不读取段末、终点 Q、剩余长度或未来段。只有推进到下一段时，才用当前完整区间的起点/中点/终点均值与导数及区间长度更新状态。
两种入口符号共享网络，符号先验保持解析定义。残差在每个状态区间内为常数；当前区间的 local hazard 仍随均值剖面变化。
因果结构保证同一前缀不因终点和未来剖面改变。它不是通过额外前缀惩罚拟合得到的效果，也不等同于距离分布已经准确。

无历史对照在每段使用同一个入口编码，关闭历史通路。小头有效参数 993；容量匹配头宽 128、有效参数 3,009。
它们仍分配了供共同初始化对照用的 GRU/段编码参数，但这些参数不参与无历史方案的 forward，不能把已分配参数数目当成有效容量。
容量匹配对照也改变了头宽，因而不是逐个权重相同的模型；训练预算、样本次序及初始零残差保持配对。

## 数据、损失与选择

复用 sigma≈0.02 同一场景的 695 条训练、158 条验证、280 条固定测试光线；训练/验证每条 2048 次参考实现，测试每条 8192 次。
参考仍为 H=1、B=8、步长 1/64 的 Tungsten-linear 标签。本轮没有增加或重采样标签。
三个种子为 {spec['seeds']}。每轮每条训练光线抽取 512 个样本做条件删失 NLL；曲线目标使用全部训练实现及逃逸样本，在 513 个均匀 q/Q 点监督。

- `local_head_T`：无历史小头，`eta=.01, beta=100`。
- `local_matched_T`：约 3k 参数无历史头，`eta=.01, beta=100`。
- `gru_nll`：历史 GRU，仅 NLL。
- `gru_T`：历史 GRU，`NLL + .01*R_delta + 100*MSE_T`。

正则按真实区间宽度/Q 积分 delta²，再按入口符号概率加权。最后不足一个完整步长的区间不会被额外放大。
所有方案都按最低验证边缘距离 NLL 选 checkpoint；测试在全部方案确定后运行。没有根据测试表现重新调步长、损失或挑种子。
该测试集已用于前轮设计诊断，不是新的盲测集。容量匹配头是在观察首个种子的验证结果后、访问本轮测试指标之前加入的对照。

## 固定测试结果

{result_table}

所有神经网络数字均为三个种子的指标均值；“最坏光线”先逐种子取最大值再平均。图中误差条为种子间标准差，不是置信区间。

![距离结果](distance_comparison.png)

分组的平均最大 T 误差：

{group_table}

事先固定的困难光线：

{examples_table}

![固定失败案例](fixed_failures.png)

GRU＋L2＋T 当前误差最大的三条光线为 {worst}，按测试结果挑选，仅用于诊断。
175 号光线仍有明显误差和训练种子间波动；固定困难光线改善不代表每条测试光线都优于 local。参考 T 来自有限次 Monte Carlo 实现，当前比较包含参考采样误差。

![剩余失败案例](remaining_failures.png)

## 前缀、未来输入与状态延续

第一种子在全部 280 条光线上检查，下面列的是最大 T 变化。

{consistency_table}

![前缀一致性](prefix_consistency.png)

逐段调用 `advance`、保留历史状态及每个符号分支的累计 H，与批量 GRU 的最大 T 差为 {summary['streaming_max_T_difference']:.3g}。
推进时需同时保留历史状态、累计 H 和入口符号先验；每段重新初始化不能等价替代状态延续。
同一三次剖面的几何细分不会改变固定 q 推进网格；这与改变 GRU 的推进步长是两项不同的检查。
无历史因果头同样通过前缀检查，所以前缀一致性来自因果输入结构；历史状态带来的距离精度改善由上述有/无历史对照检验。

## 推进步长与积分误差

以下只改变推理步长，不重训；使用第一种子，步长诊断不用于选择模型：

{step_table}

![步长敏感性](step_sensitivity.png)

当前推理应保留训练时的固定内部步长。外部追踪分段可以不同，但需保存尚未走完的内部区间，并只在固定 q 边界更新 GRU。
这还需要后续渲染接口实现；如果希望改变 GRU 本身的更新频率，应另做多尺度训练与一致性实验。

local hazard 的解析定义与 MLP 对照相同；积分实现改为在原三次剖面边界及 1/64 网格处分裂，用 8 点求积，并对每个查询的剩余部分单独积分。
这避免了按整条光线长度重建 1024 格导致的前缀数值变化。训练前可缓存这些不依赖网络的质量及局部 log hazard，查询时仅组合累计质量和残差倍率。
在 158 条验证光线上，将积分网格减半、求积阶数提高到 16，所有四个因果方案的最大 T 差为 {summary['quadrature_max_T_difference']:.3g}。
新旧 local 网格在固定测试的 513 个查询点上的最大 T 差为 {summary['local_grid_difference_max']:.6g}。
这些点恰好落在旧 1024 格的边界，不能由此推断格内也无差异。另在全部验证光线上使用 4099 个查询点检查格内差异，最大值为 {summary['off_grid_local_difference_max']:.6g}，见 off_grid_local_audit.json。
比较 GRU 与无历史头时积分实现完全相同；比较整段 MLP 时需同时考虑这一离散差异。表格误差是在 513 个评估点上测量的，不是连续区间误差的严格上界。

## 参数量与运行成本

下面是两种入口符号的整条 513 点透射率曲线，5 次重复的中位数。CPU 单线程；CUDA batch=128 使用相同的 16 条光线各重复 8 次。

{timing_table}

“已准备”排除输入及 local 积分的构建；“从三次剖面”包含这两项，但都不含 VDB 访问、距离采样、梯度采样或渲染。逐步状态调用的计时另存于 diagnostics.json。
参数更少并不保证完整光线查询更快；GRU 需要按区间推进。当前结果尚未包含渲染中的提前终止收益。

## 训练与验证

{epochs_table}

![训练曲线](training_curves.png)

28 项数值及回归测试通过，涵盖未来梯度为零、非零残差下的任意前缀截断、后缀改变、剖面重分段、流式状态、符号对称、padding、常量解析解、密度与 T 导数、逃逸目标及旧损失兼容。
12 组训练完成、所选 epoch、配对初始化、数据和旧 checkpoint 哈希均再次核对通过。
两类训练任务部分并行运行，训练耗时不用于比较推理性能；上述微基准在训练结束后单独执行。

本轮只训练距离/透射率模型，未训练条件梯度头、实现新模型的距离采样器或接入 C++ 渲染。固定 q 推进的跨步长适用性应以实测结果为准，不能从前缀一致性推导任意步长等价。

## 复现

```powershell
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_residual_sigma002.json --phase train
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_capacity_sigma002.json --phase train
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_residual_sigma002.json --phase evaluate
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_capacity_sigma002.json --phase evaluate
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_residual_sigma002.json --phase diagnose
python -B python/first_hit_se_cli.py causal-residual --config python/configs/first_hit_se_causal_residual_sigma002.json --phase report
```

容量对照配置使用 train/evaluate 两阶段；主配置 report 汇总两目录。修改已锁定训练代码或配置应使用新输出目录。
checkpoint 使用独立格式 `first_hit_causal_residual`，由 `causal_study.load_run` 加载，不会替换原 MLP 模型。
数据 SHA-256：`{spec['dataset_sha256']}`。
'''
    (output/'summary.md').write_text(text, encoding='utf-8')
    names = ('distance_comparison', 'prefix_consistency', 'step_sensitivity', 'fixed_failures', 'remaining_failures', 'training_curves')
    images = ''.join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a></figure>' for name in names)
    page = '<!doctype html><meta charset="utf-8"><title>Causal residual GRU</title><style>body{max-width:1400px;margin:24px auto;padding:0 16px;font:16px system-ui}img{max-width:100%}pre{white-space:pre-wrap;line-height:1.6}figure{margin:24px 0}</style>'
    (output/'summary.html').write_text(page+'<h1>轻量逐段残差 GRU</h1><p><a href="summary.md">Markdown</a> · <a href="summary.json">JSON</a></p>'+images+'<pre>'+html.escape(text)+'</pre>', encoding='utf-8')
    return {'summary': str(output/'summary.md'), 'audit_passed': True, 'changes_percent': changes}

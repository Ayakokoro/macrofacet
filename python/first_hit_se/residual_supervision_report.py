"""Audit and visualize the fixed residual-supervision ablation."""
import html
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .camera_study import plotting, save_figure
from .data import read_json
from .residual_supervision import choose_candidate
from .spline import local_profile


LABELS = {'local': 'Local', 'baseline': 'Residual',
          'regularized': 'Residual + L2', 'regularized_T': 'Residual + L2 + T'}
COLORS = {'local': '#64748b', 'baseline': '#ea580c',
          'regularized': '#2563eb', 'regularized_T': '#059669'}


def table(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def audit(spec, config, output, protocol, selection, training, evaluation):
    """Fail report generation if the reported experimental controls did not hold."""
    selection_hash = digest(output/'selection.json')
    if (selection['protocol_sha256'] != digest(output/'protocol.json')
            or selection_hash != training['selection_sha256']
            or selection_hash != evaluation['selection_sha256']
            or selection['test_metrics_used']):
        raise AssertionError('selection identity or test isolation failed')
    if digest(resolve(config['dataset'])) != protocol['dataset_sha256'] or evaluation['dataset_sha256'] != protocol['dataset_sha256']:
        raise AssertionError('dataset identity changed')
    for field in ('training_source_sha256', 'baseline_checkpoint_sha256'):
        for path, expected in protocol[field].items():
            if digest(Path(path)) != expected:
                raise AssertionError(f'{field} changed: {path}')
    for method, beta_zero in (('regularized', True), ('regularized_T', False)):
        candidates = [r for r in selection['candidates'] if (r['beta'] == 0) == beta_zero]
        if choose_candidate(candidates, selection['baseline_validation'], spec['selection']) != selection[method]:
            raise AssertionError('selection differs from registered validation rule')
    records = {(r['name'], r['seed']): r for r in selection['candidates']+training['selected_runs']}
    if len(records) != training['new_training_runs']:
        raise AssertionError('incorrect number of training runs')
    checks = []
    for record in records.values():
        checkpoint = Path(record['checkpoint'])
        report = read_json(checkpoint.parent/'run_report.json')
        history = read_json(checkpoint.parent/'history.json')
        original = read_json(resolve(spec['base_directory'])/'runs'/f"residual_{record['seed']}"/'run_report.json')
        checks.append({'name': record['name'], 'seed': record['seed'],
            'complete': history[-1]['epoch'] == spec['epochs'],
            'checkpoint_unchanged': digest(checkpoint) == record['checkpoint_sha256'],
            'same_initial_parameters': report['initial_parameter_sha256'] == original['initial_parameter_sha256'],
            'test_not_evaluated_in_training': set(report['metrics']) == {'train', 'validation'},
            'minimum_validation_nll_checkpoint': report['selected_epoch'] == min(history, key=lambda h: h['validation']['distance_nll'])['epoch'],
            'correct_objective_weights': report['config']['distance_objective'] == {
                'residual_l2_weight': record['eta'], 'transmittance_weight': record['beta']}})
        if not all(v for k, v in checks[-1].items() if k not in ('name', 'seed')):
            raise AssertionError(f'paired training audit failed: {checks[-1]}')
    for path, expected in evaluation['checkpoint_sha256'].items():
        if digest(Path(path)) != expected:
            raise AssertionError('evaluated checkpoint changed')
    return {'passed': True, 'dataset_and_baseline_unchanged': True,
            'selection_frozen_before_evaluation': True, 'training_runs': checks}


def comparison_figures(evaluation, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    groups = list(evaluation['groups'])
    for ax, metric, title in zip(axes, ('mean_ray_max_abs_T_error', 'worst_decile_mean', 'max_abs_T_error'),
                              ('Mean per-ray max |T error|', 'Mean of worst 10% rays', 'Worst ray (per seed)')):
        for j, method in enumerate(LABELS):
            values = [evaluation['summary'][g][method][metric] for g in groups]
            ax.bar(np.arange(len(groups))+.19*j, [v['mean'] for v in values], width=.18,
                   yerr=[v['seed_std'] for v in values], color=COLORS[method], label=LABELS[method])
        ax.set_xticks(np.arange(len(groups))+.285, ['All 280', 'Original 24', 'Spatial 128', 'Diverse 128'], rotation=15)
        ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, -.02), ncol=4, fontsize=9)
    fig.suptitle('Fixed diagnostic test set; bars: seed mean, error bars: seed standard deviation')
    fig.tight_layout()
    save_figure(plt, fig, output, 'distance_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    errors = {}
    for method, runs in evaluation['results'].items():
        errors[method] = np.array([[p['max_abs_T_error'] for p in r['metrics']['per_ray']] for r in runs]).mean(0)
        axes[0].plot(np.sort(errors[method]), np.arange(1, len(errors[method])+1)/len(errors[method]),
                     color=COLORS[method], label=LABELS[method])
    for method in ('regularized', 'regularized_T'):
        axes[1].scatter(errors['baseline'], errors[method], s=14, alpha=.6, color=COLORS[method], label=LABELS[method])
    limit = max(v.max() for k, v in errors.items() if k != 'local')*1.04
    axes[1].plot([0, limit], [0, limit], 'k--', lw=.8)
    axes[0].set(xlabel='Per-ray max |T error|, averaged over seeds', ylabel='Fraction of test rays')
    axes[1].set(xlabel='Baseline residual error', ylabel='New residual error', xlim=(0, limit), ylim=(0, limit))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(plt, fig, output, 'ray_errors')


def example_figure(ids, raw, evaluation, config, output, plt, name, title):
    fig, axes = plt.subplots(2, len(ids), figsize=(5*len(ids), 7), squeeze=False)
    for column, index in enumerate(ids):
        p = raw[index]
        q = torch.linspace(0, p['horizon'], config['curve_points'], dtype=torch.float64)
        distance = q.numpy()*p['geometry']['ell']
        mean, _ = local_profile(p['features'], q)
        axes[0, column].plot(distance, mean, color='#475569')
        axes[0, column].axhline(0, color='black', lw=.7)
        axes[0, column].set(title=f'Ray {index}', ylabel='Mean / sigma', xlabel='Distance from medium entry')
        reference = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/len(p['hit'])
        axes[1, column].plot(distance, reference, 'k', lw=2, label='Reference')
        for method, runs in evaluation['results'].items():
            curves = np.array([r['metrics']['per_ray'][index]['T'] for r in runs])
            axes[1, column].plot(distance, curves.mean(0), color=COLORS[method], label=LABELS[method])
            axes[1, column].fill_between(distance, curves.min(0), curves.max(0), color=COLORS[method], alpha=.13)
        axes[1, column].set(xlabel='Distance from medium entry', ylabel='Transmittance', ylim=(-.02, 1.02))
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(title+'; shading: range over three seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, name)


def training_figure(spec, selection, training, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for method in ('baseline', 'regularized', 'regularized_T'):
        for j, seed in enumerate(spec['seeds']):
            if method == 'baseline':
                directory = resolve(spec['base_directory'])/'runs'/f'residual_{seed}'
                eta = beta = 0.
            else:
                record = next(r for r in training['selected_runs'] if r['name'] == selection[method]['name'] and r['seed'] == seed)
                directory = Path(record['checkpoint']).parent
                eta, beta = record['eta'], record['beta']
            history = read_json(directory/'history.json')
            x = [h['epoch'] for h in history]
            axes[0].plot(x, [h['validation']['distance_nll'] for h in history], color=COLORS[method], alpha=.6,
                         label=LABELS[method] if j == 0 else None)
            if method != 'baseline':
                axes[1].plot(x, [eta*h['train_residual_l2'] for h in history], color=COLORS[method], alpha=.6,
                             label=LABELS[method] if j == 0 else None)
            if beta:
                axes[2].plot(x, [beta*h['train_transmittance_mse'] for h in history], color=COLORS[method], alpha=.6,
                             label=LABELS[method] if j == 0 else None)
    for ax, title in zip(axes, ('Validation marginal NLL', 'Weighted residual penalty', 'Weighted transmittance MSE')):
        ax.set(xlabel='Epoch', title=title)
        ax.legend(fontsize=8)
    fig.suptitle('Auxiliary curves show training losses; checkpoints chosen by validation NLL')
    fig.tight_layout()
    save_figure(plt, fig, output, 'training_curves')


def report(spec, config, data, output):
    protocol, selection, training, evaluation = (read_json(output/name) for name in (
        'protocol.json', 'selection.json', 'training.json', 'evaluation.json'))
    checks = audit(spec, config, output, protocol, selection, training, evaluation)
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    errors = {method: np.array([[p['max_abs_T_error'] for p in run['metrics']['per_ray']] for run in runs])
              for method, runs in evaluation['results'].items()}
    worst = [int(i) for i in np.argsort(errors['regularized_T'].mean(0))[-3:][::-1]]
    fixed = protocol['fixed_failure_examples']
    def example(index):
        return {'ray': index, 'scene_id': raw[index]['scene_id'], 'source': raw[index]['source'],
            'mean_per_seed_max_T_error': {method: float(values[:, index].mean()) for method, values in errors.items()},
            'per_seed_max_T_error': {method: values[:, index].tolist() for method, values in errors.items()}}
    score = evaluation['summary']['all_280']
    changes = {method: {metric: 100*(value['mean']/score['baseline'][metric]['mean']-1)
        for metric, value in score[method].items() if metric != 'distance_nll'}
        for method in ('regularized', 'regularized_T')}
    paired = {method: {'rays_improved_on_seed_mean': int((errors[method].mean(0) < errors['baseline'].mean(0)).sum()),
        'rays_worse_than_local_on_seed_mean': int((errors[method].mean(0) > errors['local'][0]).sum()),
        'seed_changes': [{
            'seed': seed,
            'mean_ray_max_abs_T_error': float((values-errors['baseline'][j]).mean()),
            'max_abs_T_error': float(values.max()-errors['baseline'][j].max())}
            for j, (seed, values) in enumerate(zip(spec['seeds'], errors[method]))]}
        for method in ('regularized', 'regularized_T')}
    baseline_runs = [read_json(resolve(spec['base_directory'])/'runs'/f'residual_{s}'/'run_report.json') for s in spec['seeds']]
    summary = {'specification': spec, 'selection': selection, 'audit': checks,
        'groups': {group: len(indices) for group, indices in evaluation['groups'].items()},
        'distance': evaluation['summary'], 'changes_percent_vs_baseline': changes, 'paired': paired,
        'fixed_failure_examples': [example(i) for i in fixed],
        'new_worst_examples': [example(i) for i in worst],
        'new_worst_example_rule': 'largest seed-mean per-ray maximum T error of regularized_T; diagnostic only',
        'selected_runs': training['selected_runs'],
        'baseline_runs': [{k: r[k] for k in ('seed', 'parameters', 'selected_epoch', 'training_seconds')} for r in baseline_runs],
        'data': {split: {'rays': sum(p['split'] == split for p in data['profiles']),
                       'samples': sum(len(p['hit']) for p in data['profiles'] if p['split'] == split)}
                 for split in ('train', 'validation', 'test')},
        'identity': {name: digest(output/name) for name in ('protocol.json', 'selection.json', 'training.json', 'evaluation.json')},
        'dataset_sha256': protocol['dataset_sha256'], 'report_source_sha256': digest(Path(__file__))}
    write_json(output/'summary.json', summary)
    plt = plotting()
    comparison_figures(evaluation, output, plt)
    example_figure(fixed, raw, evaluation, config, output, plt, 'fixed_failures', 'Failure rays fixed before this experiment')
    example_figure(worst, raw, evaluation, config, output, plt, 'new_worst_rays', 'Largest errors after evaluation; diagnostic only')
    training_figure(spec, selection, training, output, plt)
    eta, beta = selection['regularized']['eta'], selection['regularized_T']['beta']
    pilot = selection['pilot_seed']
    selected_names = {selection[method]['name'] for method in ('regularized', 'regularized_T')}
    validation_table = table(['候选', 'eta', 'beta', 'epoch', '验证 NLL', '平均最大误差', '最差 10% 均值', 'NLL 门槛'],
        [['原 residual', 0, 0, next(r['selected_epoch'] for r in baseline_runs if r['seed'] == pilot)]+
         [f"{selection['baseline_validation'][k]:.6f}" for k in ('distance_nll', 'mean_ray_max_abs_T_error', 'worst_decile_mean')]+['基准']]+
        [[r['name']+('（选中）' if r['name'] in selected_names else ''), r['eta'], r['beta'], r['selected_epoch']]+
         [f"{r['validation'][k]:.6f}" for k in ('distance_nll', 'mean_ray_max_abs_T_error', 'worst_decile_mean')]+
         ['通过' if r['validation']['distance_nll'] <= selection['regularized']['validation_nll_limit'] else '未通过']
         for r in selection['candidates']])
    metric_names = ('mean_ray_max_abs_T_error', 'p95_ray_max_abs_T_error', 'worst_decile_mean',
                    'max_abs_T_error', 'rmse_T', 'mean_abs_escape_error', 'distance_nll')
    result_table = table(['方案', '平均最大误差', 'P95', '最差 10% 均值', '最坏光线', 'T RMSE', '逃逸 MAE', '距离 NLL'],
        [[LABELS[method]]+[f"{score[method][k]['mean']:.6f}" for k in metric_names] for method in LABELS])
    variation_table = table(['方案', '平均最大误差', '最差 10% 均值', '最坏光线'],
        [[LABELS[method]]+[f"{score[method][k]['mean']:.6f} ± {score[method][k]['seed_std']:.6f}"
         for k in ('mean_ray_max_abs_T_error', 'worst_decile_mean', 'max_abs_T_error')] for method in LABELS if method != 'local'])
    group_table = table(['组', '光线', *LABELS.values()], [[group, len(indices)]+
        [f"{evaluation['summary'][group][method]['mean_ray_max_abs_T_error']['mean']:.6f}" for method in LABELS]
        for group, indices in evaluation['groups'].items()])
    example_table = table(['固定失败光线', *LABELS.values()], [[r['ray']]+
        [f"{r['mean_per_seed_max_T_error'][method]:.6f}" for method in LABELS] for r in summary['fixed_failure_examples']])
    worst_table = table(['新方案最大误差光线', *LABELS.values()], [[r['ray']]+
        [f"{r['mean_per_seed_max_T_error'][method]:.6f}" for method in LABELS] for r in summary['new_worst_examples']])
    run_rows = [[LABELS['baseline'], r['seed'], r['selected_epoch'], r['parameters']] for r in baseline_runs]
    for method in ('regularized', 'regularized_T'):
        run_rows += [[LABELS[method], r['seed'], r['selected_epoch'], r['parameters']] for r in training['selected_runs']
                     if r['name'] == selection[method]['name']]
    run_table = table(['方案', '种子', '验证所选 epoch', '参数量'], run_rows)
    findings = '\n'.join(f"- {LABELS[method]} 相对原 residual：平均最大 T 误差变化 {changes[method]['mean_ray_max_abs_T_error']:+.1f}%，"
        f"最差 10% 均值变化 {changes[method]['worst_decile_mean']:+.1f}%，最坏光线误差变化 {changes[method]['max_abs_T_error']:+.1f}%。"
        for method in ('regularized', 'regularized_T'))
    guard = '两组最终候选均通过验证 NLL 门槛。' if all(selection[m]['nll_guard_passed'] for m in ('regularized', 'regularized_T')) else (
        '**至少一组未通过验证 NLL 门槛，按预设回退规则仅选取 NLL 最小候选用于诊断；不能将其视为已接受的改进。**')
    text = f'''# MLP 残差正则与透射率监督：sigma=0.02

使用现有多视角数据，新增 {training['new_training_runs']} 组 500-epoch 训练。冻结验证集选出的权重为 `eta={eta:g}`、`beta={beta:g}`。
原 residual、仅加 L2、L2＋透射率监督均以三个配对种子比较，原 residual 的 checkpoint 直接复用。

{findings}

L2＋T 的 P95 误差变化为 {changes['regularized_T']['p95_ray_max_abs_T_error']:+.1f}%，T RMSE 变化为 {changes['regularized_T']['rmse_T']:+.1f}%。
三个配对种子中，{sum(r['mean_ray_max_abs_T_error'] < 0 for r in paired['regularized_T']['seed_changes'])} 个种子的平均最大误差改善，
{sum(r['max_abs_T_error'] < 0 for r in paired['regularized_T']['seed_changes'])} 个种子的最坏光线误差改善。
降低最坏误差不代表所有光线均改善，应结合分位数与下方剩余失败案例判断。

{guard}

## 损失与实现

每个入口符号分支保持 `lambda_k(q) = lambda_local,k(q) * exp(delta_k(q))`：

```text
L = L_conditional_censored_NLL + eta * R_delta + beta * L_T
R_delta = mean_ray sum_k pi_k * mean_1024_bins(delta_k^2)
T(q) = sum_k pi_k * exp(-H_k(q))
L_T = mean_ray mean_513_queries((T(q) - T_reference(q))^2)
```

`pi_k` 是解析入口符号概率；残差正则作用于插值后的 log hazard 修正，不是参数 weight decay，也不是对 hazard 本身施压。
各光线按归一化距离均匀取点、等权平均。`T_reference` 由该训练光线全部 2048 次实现计算，分母包含逃逸样本，末端点直接监督逃逸概率。
NLL 每轮仍随机抽 512 次实现；曲线目标不随这次抽样变化。经验曲线 MSE 与平均 survival Brier loss 相差一个不依赖模型的常数。

原有 NLL 保留对入口符号的条件训练，并包括右删失项。透射率项监督对符号边缘化后的整条曲线。
曲线在 513 个均匀 `q/Q` 点上计算，含 0 与 Q；不是只对命中距离的条件 CDF 做归一化。
同一个 forward 的 log rates 被三项损失复用；训练多一次曲线查询，网络、checkpoint 格式和推理路径不变，参数量仍为 57,632。
缺省两项权重均为零，恢复原有损失与梯度。

## 数据与预设选择流程

- 场文件：`outputs/fields/shader_ball_full_0.02.nvdb`；实际 sigma≈0.02。
- 训练 695 条、验证 158 条，均为每条 2048 次参考实现；固定测试 280 条，每条 8192 次。
- 标签沿用 H=1、B=8、步长 1/64 的 Tungsten-linear 参考采样；本轮没有新增或替换标签。
- 三个种子：{', '.join(map(str, spec['seeds']))}；同种子初始参数与原基线相同，样本抽取顺序、优化器、500 epochs、学习率调度均保持配对。
- 第一种子先比较 `eta ∈ {spec['residual_l2_candidates']}`，固定选出的 eta，再比较 `beta ∈ {spec['transmittance_candidates']}`。
- 每个训练按最小验证**边缘距离 NLL**选 checkpoint。权重按最差 10% 验证光线的平均最大 T 误差选，并要求验证 NLL 不超过原基线 + {spec['selection']['maximum_validation_nll_increase']:g}。
- 选出的两组权重再用另外两个种子复核。`selection.json` 在复核及本轮测试评估前写入，测试指标不参与本轮权重或 checkpoint 选择。
- 这里的最差 10% 是每个种子内按逐光线最大 T 误差排序，取 `ceil(0.1*N)` 条；不是对平均曲线排序。

第一种子验证集结果：

{validation_table}

## 固定测试结果

{result_table}

表中除 Local 外均先逐种子计算指标、再平均三个种子；最坏光线列也是三个最大值的均值。
各指标以小为好。下面的 ± 为种子间总体标准差，不是置信区间。

{variation_table}

![总体与尾部比较](distance_comparison.png)

分组的逐光线最大 T 误差均值：

{group_table}

![逐光线误差分布](ray_errors.png)

## 固定失败案例与剩余误差

266、84、175 是上一轮发现的失败光线，本轮开始前已经固定。

{example_table}

![事先固定的失败光线](fixed_failures.png)

新 L2＋T 方案误差最大的三条光线为 {', '.join(map(str, worst))}，下图按本轮测试结果挑选，只用于诊断。
图中曲线平均与种子范围用于展示，表格指标没有把多个网络组合成 ensemble。

{worst_table}

![新方案的最大误差](new_worst_rays.png)

## 训练与验证

{run_table}

![训练与验证曲线](training_curves.png)

20 项数值及回归测试通过：零权重的原损失/梯度一致性、解析正则梯度、符号先验权重、含逃逸的曲线目标、抽样独立性、梯度传播、单次编码和验证选择规则均有检查。
报告生成时再次核对 {len(checks['training_runs'])} 组训练完成、初始化配对、checkpoint 选择、测试隔离、数据与旧 checkpoint 哈希；结构化结果在 [summary.json](summary.json)。
各 run 的 `history.json` 保存未乘权重的辅助损失、总损失与原 NLL；图上显示乘权重后的辅助项。

## 适用范围

这是同一场景、已在前轮用于诊断的固定测试集，不是新的盲测集；只有三个训练种子。
平均改善不保证每条光线都更好，残差正则也不是强制幅度上限。
解析 local 的网格离散误差和有限记忆参考标签误差仍存在。整段 MLP 的前缀一致性尚未解决。
本轮没有重训条件梯度头或执行完整渲染；旧梯度头绑定旧编码器，新距离模型须重新训练对应梯度头后再验证联合采样。
额外正则与曲线监督只用于训练，不增加推理网络、local 积分或采样步骤；本轮未重新测量端到端渲染时间。

## 复现

在仓库根目录运行：

```powershell
python -B python/first_hit_se_cli.py residual-supervision --config python/configs/first_hit_se_residual_supervision_sigma002.json
```

可用 `--phase train`、`--phase evaluate`、`--phase report` 分别运行；已完成训练会校验身份并复用，未完成训练从保存状态继续。
训练实现与数据在 `protocol.json` 中锁定；变更实现后应使用新的输出目录。

仅 L2 的 checkpoint：`variants/{selection['regularized']['name']}/runs/residual_<seed>/best.pt`。
L2＋T 的 checkpoint：`variants/{selection['regularized_T']['name']}/runs/residual_<seed>/best.pt`。

数据 SHA-256：`{protocol['dataset_sha256']}`。
'''
    (output/'summary.md').write_text(text, encoding='utf-8')
    names = ('distance_comparison', 'ray_errors', 'fixed_failures', 'new_worst_rays', 'training_curves')
    pictures = ''.join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a></figure>' for name in names)
    page = '<!doctype html><meta charset="utf-8"><title>Residual supervision, sigma=0.02</title>'+(
        '<style>body{max-width:1250px;margin:32px auto;font:16px system-ui;padding:0 16px}img{max-width:100%}pre{white-space:pre-wrap;line-height:1.6}figure{margin:24px 0}</style>')+(
        '<h1>MLP 残差正则与透射率监督</h1><p><a href="summary.md">Markdown 报告</a> · <a href="summary.json">结构化结果</a></p>')+pictures+'<pre>'+html.escape(text)+'</pre>'
    (output/'summary.html').write_text(page, encoding='utf-8')
    return {'summary': str(output/'summary.md'), 'eta': eta, 'beta': beta, 'audit_passed': checks['passed'],
            'changes_percent_vs_baseline': changes}

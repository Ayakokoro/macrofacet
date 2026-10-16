"""Audited fitting report for the frozen causal gradient-head experiment."""
import html
import math
from pathlib import Path

import numpy as np

from renewal.collect import digest, resolve, write_json
from .camera_study import plotting, save_figure
from .data import read_json
from .residual_supervision_report import table


LABELS = {'mlp_mixture': 'MLP / mixture', 'mlp_cdf': 'MLP / CDF',
          'mixture_local': 'No history / mixture', 'mixture_gru': 'GRU / mixture', 'cdf_gru': 'GRU / CDF'}
COLORS = dict(zip(LABELS, ('#ea580c', '#a16207', '#2563eb', '#059669', '#9333ea')))


def moments(values):
    return {'mean': float(np.mean(values)), 'seed_std': float(np.std(values))}


def audit(spec, output):
    protocol, trained, evaluation, diagnostics = (read_json(output/name) for name in
        ('protocol.json', 'training.json', 'evaluation.json', 'diagnostics.json'))
    if (trained['protocol_sha256'] != digest(output/'protocol.json') or trained['test_metrics_used']
            or evaluation['training_sha256'] != digest(output/'training.json')
            or diagnostics['training_sha256'] != digest(output/'training.json')
            or diagnostics['evaluation_sha256'] != digest(output/'evaluation.json') or not diagnostics['passed']
            or diagnostics['source_sha256'] != digest(Path(__file__).with_name('causal_gradient_diagnostics.py'))):
        raise AssertionError('training/evaluation/diagnostics identity changed')
    if digest(resolve(spec['dataset'])) != protocol['dataset_sha256']:
        raise AssertionError('dataset changed')
    for key in ('source_sha256', 'checkpoint_sha256'):
        for path, expected in protocol[key].items():
            if digest(Path(path)) != expected:
                raise AssertionError(f'locked {key} changed: {path}')
    if digest(resolve(spec['backbone_directory'])/'training.json') != protocol['backbone_training_sha256']:
        raise AssertionError('frozen distance selection changed')
    if digest(resolve(spec['baseline_directory'])/'gradient_evaluation.json') != protocol['baseline_evaluation_sha256']:
        raise AssertionError('old gradient evaluation changed')
    for record in trained['runs']:
        path = Path(record['checkpoint'])
        history = read_json(path.parent/'history.json')
        if (history[-1]['epoch'] != spec['epochs']
                or record['selected_epoch'] != min(history, key=lambda h: h['validation']['speed_nll'])['epoch']
                or digest(path) != record['checkpoint_sha256'] or digest(path) != evaluation['checkpoint_sha256'][str(path)]):
            raise AssertionError('completion or checkpoint selection failed')
    if len(trained['runs']) != len(spec['seeds'])*len(spec['variants']):
        raise AssertionError('missing training run')
    for seed in spec['seeds']:
        paired = [r for r in trained['runs'] if r['identity']['seed'] == seed]
        if (len({r['initial_hidden_sha256'] for r in paired}) != 1 or
            len({r['initial_parameter_sha256'] for r in paired if r['identity']['variant']['kind'] == 'mixture'}) != 1):
            raise AssertionError('paired head initialization failed')
    return {'passed': True, 'completed_runs': len(trained['runs']), 'frozen_distance_checkpoints_unchanged': True,
            'validation_only_selection': True, 'paired_initialization': True}, trained, evaluation, diagnostics


def summarize(spec, evaluation):
    groups, scores = evaluation['groups'], {}
    minimum = spec['diagnostics']['minimum_ray_hits']
    for group, indices in groups.items():
        scores[group] = {}
        for method, runs in evaluation['results'].items():
            rows = []
            for run in runs:
                ray = [run['metrics']['per_ray'][i] for i in indices]
                valid = [r for r in ray if r['hits'] >= minimum]
                values = sorted(r['pit_ks'] for r in valid)
                rows.append({'speed_nll': sum(r['speed_nll']*r['hits'] for r in ray if r['hits'])/sum(r['hits'] for r in ray),
                    'mean_ray_pit_ks': float(np.mean(values)), 'p95_ray_pit_ks': float(np.quantile(values, .95)),
                    'worst_decile_ray_pit_ks': float(np.mean(values[-max(1, math.ceil(.1*len(values))):])),
                    'max_ray_pit_ks': max(values), 'eligible_rays': len(valid)})
            scores[group][method] = {k: moments([r[k] for r in rows]) for k in rows[0]}
    overall = {method: {k: moments([r['metrics'][k] for r in runs]) for k in ('speed_nll', 'pit_ks', 'joint_nll')}
               for method, runs in evaluation['results'].items()}
    late = {method: moments([r['metrics']['distance_bins'][-1]['pit_ks'] for r in runs]) for method, runs in evaluation['results'].items()}
    return scores, overall, late


def example_figure(diagnostics, ids, output, plt, name, title):
    fig, axes = plt.subplots(2, len(ids), figsize=(4.5*len(ids), 7), squeeze=False)
    grid = diagnostics['speed_grid']
    for col, i in enumerate(ids):
        example = next(r for r in diagnostics['examples'] if r['ray'] == i)
        for row, condition in enumerate(example['conditions']):
            ax = axes[row, col]
            label = 'All reference hits' if row == 0 else 'Reference hits at q/Q >= 1/2'
            ax.set(title=f'Ray {i}: {label}\nn={condition["hits"]}', xlabel='Normalized crossing speed W', ylabel='Conditional CDF', ylim=(-.02, 1.02))
            if not condition['hits']:
                continue
            ax.plot(grid, condition['reference_cdf'], color='black', lw=2, label='Reference')
            for method in ('mlp_mixture', 'mixture_local', 'mixture_gru', 'cdf_gru'):
                ax.plot(grid, condition['predicted'][method], color=COLORS[method], label=LABELS[method])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, -.015), ncol=5, fontsize=8)
    fig.suptitle(title+'; predicted CDF averaged over the SAME reference locations/signs')
    fig.tight_layout()
    save_figure(plt, fig, output, name)


def figures(spec, trained, evaluation, diagnostics, scores, overall, late, output):
    plt = plotting()
    names = list(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, title, values in (
        (axes[0], 'Conditional speed NLL', [overall[m]['speed_nll'] for m in names]),
        (axes[1], f'Mean ray PIT KS (hits >= {spec["diagnostics"]["minimum_ray_hits"]})', [scores['all_280'][m]['mean_ray_pit_ks'] for m in names]),
        (axes[2], 'Late hits: q/Q >= 3/4, PIT KS', [late[m] for m in names])):
        for i, (m, v) in enumerate(zip(names, values)):
            ax.errorbar(i, v['mean'], yerr=v['seed_std'], fmt='o', color=COLORS[m], capsize=4)
        ax.set_xticks(np.arange(len(names)), [LABELS[m] for m in names], rotation=25, ha='right')
        ax.set_title(title)
    fig.suptitle('Three seeds; error bars are seed standard deviations, not confidence intervals')
    fig.tight_layout()
    save_figure(plt, fig, output, 'gradient_fit')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method in names:
        runs = evaluation['results'][method]
        counts = np.array([[r['pit_counts'] for r in run['metrics']['per_ray']] for run in runs]).sum((0, 1))
        axes[0].step((np.arange(20)+.5)/20, counts/counts.sum()*20, where='mid', color=COLORS[method], label=LABELS[method])
        values = sorted(np.mean([run['metrics']['per_ray'][i]['pit_ks'] for run in runs]) for i in range(280)
                        if runs[0]['metrics']['per_ray'][i]['hits'] >= spec['diagnostics']['minimum_ray_hits'])
        axes[1].plot(values, np.arange(1, len(values)+1)/len(values), color=COLORS[method])
        bins = np.array([[b['pit_ks'] for b in run['metrics']['distance_bins']] for run in runs])
        axes[2].errorbar(np.arange(4), bins.mean(0), yerr=bins.std(0), fmt='o-', color=COLORS[method])
    axes[0].axhline(1., ls='--', color='black', lw=.8)
    axes[0].set(xlabel='PIT = predicted CDF(reference W)', ylabel='Density', title='Global calibration')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Per-ray PIT KS, seed mean', ylabel='Fraction of eligible rays', title='Ray-level calibration')
    axes[2].set_xticks(np.arange(4), ['0–1/4', '1/4–1/2', '1/2–3/4', '3/4–1'])
    axes[2].set(xlabel='Reference hit q/Q', ylabel='PIT KS', title='Calibration by hit location')
    fig.tight_layout()
    save_figure(plt, fig, output, 'gradient_calibration')
    example_figure(diagnostics, diagnostics['fixed_rays'], output, plt, 'fixed_gradient_cdfs', 'Fixed distance-failure rays; first seed')
    example_figure(diagnostics, diagnostics['worst_rays'], output, plt, 'worst_gradient_cdfs', 'Worst GRU-mixture PIT rays selected after evaluation; first seed')
    example_figure(diagnostics, diagnostics['well_sampled_worst_rays'], output, plt, 'well_sampled_gradient_cdfs',
                   f'Worst GRU-mixture PIT rays with >= {diagnostics["well_sampled_minimum_hits"]} hits; post-evaluation diagnosis, first seed')
    fig, ax = plt.subplots(figsize=(11, 4.5))
    kinds = list(diagnostics['consistency_max'])
    for j, name in enumerate(names):
        ax.bar(np.arange(len(kinds))+.14*j, [max(1e-16, diagnostics['consistency_max'][k][name]) for k in kinds], width=.13, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(np.arange(len(kinds))+.28, kinds)
    ax.set(yscale='log', ylabel='Maximum conditional CDF change', title='First seed, both entry signs; display floor 1e-16')
    ax.axhline(spec['diagnostics']['cdf_tolerance'], ls='--', color='black', lw=.8)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(plt, fig, output, 'gradient_prefix')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for record in trained['runs']:
        name, seed = record['identity']['variant']['name'], record['identity']['seed']
        history = read_json(Path(record['checkpoint']).parent/'history.json')
        ax.plot([h['epoch'] for h in history], [h['validation']['speed_nll'] for h in history],
                color=COLORS[name], alpha=.65, label=LABELS[name] if seed == spec['seeds'][0] else None)
    ax.set(xlabel='Epoch', ylabel='Validation conditional speed NLL')
    ax.legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'gradient_training')


def report(spec, data, output):
    checks, trained, evaluation, diagnostics = audit(spec, output)
    scores, overall, late = summarize(spec, evaluation)
    validation = {v['name']: float(np.mean([r['validation']['speed_nll'] for r in trained['runs']
        if r['identity']['variant']['name'] == v['name']])) for v in spec['variants']}
    preferred = min(validation, key=validation.get)
    examples = {str(i): {'hits': evaluation['results']['mixture_gru'][0]['metrics']['per_ray'][i]['hits'],
        'mean_pit_ks': {name: float(np.mean([r['metrics']['per_ray'][i]['pit_ks'] for r in runs]))
                        for name, runs in evaluation['results'].items()}}
        for i in sorted(set(diagnostics['fixed_rays']+diagnostics['worst_rays']+diagnostics['well_sampled_worst_rays']))}
    summary = {'audit': checks, 'scope': diagnostics['scope'], 'groups': scores, 'overall': overall,
        'late_pit_ks': late, 'validation_nll': validation, 'preferred_by_validation_nll': preferred,
        'examples': examples, 'consistency_max': diagnostics['consistency_max'],
        'conditional_sampling': diagnostics['probability_summary'], 'runs': trained['runs'],
        'report_source_sha256': digest(Path(__file__)), 'artifact_sha256': {name: digest(output/name) for name in
            ('protocol.json', 'training.json', 'evaluation.json', 'diagnostics.json')}}
    step_path = output/'step_accuracy'/'summary.json'
    step_section, step_link = '', ''
    if step_path.exists():
        step = read_json(step_path)
        if not step['audit']['passed']:
            raise AssertionError('gradient step report audit failed')
        for name, expected in step['artifact_sha256'].items():
            if digest(step_path.parent/name) != expected:
                raise AssertionError('gradient step report artifact changed')
        summary['step_accuracy'] = {'summary': str(step_path), 'sha256': digest(step_path), 'audit': step['audit']}
        step_table = table(['方案', '内部步长 Δq', '条件 NLL', '逐光线 PIT KS', '晚碰撞 PIT KS'],
            [[LABELS[r['method']], r['step']]+[f"{r['metrics'][k]['mean']:.6f}"
                for k in ('speed_nll', 'mean_ray_pit_ks', 'late_pit_ks')]
                for r in step['rows'] if r['method'] in ('mixture_gru', 'cdf_gru')])
        step_section = '\n## 补充：冻结梯度头的步长测试\n\n[完整步长报告](step_accuracy/summary.html) · [Markdown](step_accuracy/summary.md)\n\n'
        step_section += '固定相同参考碰撞点，测试五种内部 GRU 步长和三个种子；另外用第一种子分别改变历史状态与步内位置。以下为三种子均值。\n\n'+step_table+'\n'
        step_link = '<p><strong>新增：</strong><a href="step_accuracy/summary.html">梯度头步长精度测试：五种步长、三个种子及状态/位置消融</a></p>'
    write_json(output/'summary.json', summary)
    figures(spec, trained, evaluation, diagnostics, scores, overall, late, output)
    result_table = table(['方案', '条件速度 NLL', '全局 PIT KS', '逐光线 PIT KS', 'P95 光线 PIT KS', '晚碰撞 PIT KS'],
        [[LABELS[m], f"{overall[m]['speed_nll']['mean']:.6f}", f"{overall[m]['pit_ks']['mean']:.6f}",
          f"{scores['all_280'][m]['mean_ray_pit_ks']['mean']:.6f}", f"{scores['all_280'][m]['p95_ray_pit_ks']['mean']:.6f}",
          f"{late[m]['mean']:.6f}"] for m in LABELS])
    group_table = table(['组', *[LABELS[m] for m in LABELS]],
        [[g]+[f"{scores[g][m]['mean_ray_pit_ks']['mean']:.6f}" for m in LABELS] for g in scores])
    fixed_table = table(['光线', '命中数', *[LABELS[m] for m in LABELS]],
        [[i, examples[str(i)]['hits']]+[f"{examples[str(i)]['mean_pit_ks'][m]:.6f}" for m in LABELS] for i in diagnostics['fixed_rays']])
    well_sampled_table = table(['光线', '命中数', *[LABELS[m] for m in LABELS]],
        [[i, examples[str(i)]['hits']]+[f"{examples[str(i)]['mean_pit_ks'][m]:.6f}" for m in LABELS] for i in diagnostics['well_sampled_worst_rays']])
    coverage_table = table(['方案', '68% 区间覆盖率', '90% 区间覆盖率', '低于 1% 分位', '高于 99% 分位'],
        [[LABELS[m]]+[f"{np.mean([r['metrics'][k] for r in evaluation['results'][m]]):.6f}"
                       for k in ('coverage_68', 'coverage_90', 'lower_tail_01', 'upper_tail_99')]
         for m in ('mixture_local', 'mixture_gru', 'cdf_gru')])
    prefix_table = table(['方案', *diagnostics['consistency_max']],
        [[LABELS[m]]+[f"{v[m]:.3g}" for v in diagnostics['consistency_max'].values()] for m in LABELS])
    sampling_table = table(['方案', '条件 CDF 阈值误差均值', '采样自身 PIT KS', '联合 NLL'],
        [[LABELS[m], f"{diagnostics['probability_summary'][m]['mean_ray_max_reference_CDF_error']:.6f}",
          f"{diagnostics['probability_summary'][m]['sample_pit_ks']:.6f}", f"{overall[m]['joint_nll']['mean']:.6f}"] for m in LABELS])
    run_table = table(['方案', '种子', '参数', '所选 epoch', '验证 NLL'],
        [[r['identity']['variant']['name'], r['identity']['seed'], r['parameters'], r['selected_epoch'],
          f"{r['validation']['speed_nll']:.6f}"] for r in trained['runs']])
    delta_local = overall['mixture_gru']['speed_nll']['mean']-overall['mixture_local']['speed_nll']['mean']
    delta_old = overall['mixture_gru']['speed_nll']['mean']-overall['mlp_mixture']['speed_nll']['mean']
    ray_change = 100*(scores['all_280']['mixture_gru']['mean_ray_pit_ks']['mean']/scores['all_280']['mlp_mixture']['mean_ray_pit_ks']['mean']-1)
    late_change = 100*(late['mixture_gru']['mean']/late['mlp_mixture']['mean']-1)
    text = f'''# 残差 GRU 条件梯度拟合：sigma=0.02

日期：2026-10-16。完成 3 个梯度方案 × 3 个种子、各 150 epochs；GRU 距离模型全部冻结。

相同容量混合头加入历史状态后，固定测试条件速度 NLL 变化 **{delta_local:+.6f} nat/命中**；相对旧整段 MLP 混合头变化 **{delta_old:+.6f}**。
三个新方案平均验证 NLL 为 {validation}，按此规则优先保留 `{preferred}`。所有 checkpoint 只按验证条件 NLL 选择，测试在九组训练全部完成后执行。
GRU 混合头相对旧 MLP 混合头的逐光线 PIT KS 均值变化 {ray_change:+.1f}%，晚碰撞 PIT KS 变化 {late_change:+.1f}%。
CDF 的总体 NLL 与逐光线 PIT 略逊、晚碰撞 PIT 略好，差异较小；当前支持保留混合头作主方案，CDF 继续作为对照。

## 拟合对象与对照

学习 `p(W | tau=q, entry sign=k, mean profile up to q)`，其中 `W=-k*(ell/sigma)*dot(omega,g)>0` 是无量纲纵向首碰梯度速度。
使用参考首碰位置和入口符号，排除距离采样误差；这不是把梯度与距离视为独立。本轮不拟合完整三维梯度或法线，也未做新模型联合距离采样及渲染验证。

数据保持不变：695 条训练光线、158 条验证光线、280 条固定测试光线；训练/验证每条 2048 次，测试每条 8192 次。
参考为 H=1、B=8、步长 1/64 的 GPIS function-space Tungsten-linear 标签，sigma≈0.02、ell≈0.05656854。测试集已经用于前轮诊断，仍是同一场景。

冻结上一轮各配对种子的 `gru_nll` checkpoint，16 维历史状态、内部步长 Δq=1/8。头输入为进入当前区间的历史状态、区间内进度、log(1+q)、带入口符号的局部均值/导数；不输入 Q 或 q/Q。
正好落在状态边界的梯度查询采用左极限状态，因此前缀在该边界结束时与延长后查询保持一致。当前区间末端不会提前进入梯度头。

- `mixture_local`：相同头，使用固定入口状态代替推进后的历史状态。
- `mixture_gru`：历史状态＋8 分量正半轴截断高斯混合。
- `cdf_gru`：历史状态＋8 区间单调有理二次 CDF，变换指数基分布。

头宽均为 64。两种 mixture 同种子初始权重完全相同；CDF 隐层初始化相同，输出初始化按分布定义不同。CDF 仍按导数的 NLL 训练，不是经验 CDF 点回归。
原 MLP 的 Mixture/CDF checkpoint 直接复用，头的 32 维输入与本轮 16 维输入不同，原头也读取完整光线；这项比较是现有完整方案对照。历史作用由同容量的两个新 mixture 头检验。
每轮每条训练光线抽 512 个实现，与旧实验的种子、抽样顺序、150 epochs 预算及优化设置一致；只有命中样本贡献梯度损失，逃逸样本没有伪造梯度标签。

## 固定测试拟合结果

{result_table}

指标先逐种子计算再平均。NLL 按命中数加权，逐光线 PIT 指标对命中数不少于 {spec['diagnostics']['minimum_ray_hits']} 的光线等权计算，避免低样本数主导最坏光线。
PIT 为网络 CDF 在参考 W 处的值，理想分布应接近均匀。全局 PIT 可因不同位置/光线偏差抵消而偏小，需结合逐光线与位置分组看。
本次符合样本数门槛的测试光线为 {int(scores['all_280']['mixture_gru']['eligible_rays']['mean'])} 条；晚碰撞定义 q/Q≥3/4，共 {evaluation['results']['mixture_gru'][0]['metrics']['distance_bins'][-1]['hits']} 个参考命中。

![拟合比较](gradient_fit.png)
![分组校准](gradient_calibration.png)

不同测试子集的逐光线 PIT KS 均值：

{group_table}

新区间覆盖率，以 .68、.90、.01、.01 为理想值：

{coverage_table}

覆盖率接近目标不代表每条条件分布已经拟合正确；样本来自有限 Monte Carlo 参考，本轮没有重新生成更高精度标签。

## 困难剖面与条件 CDF

此前固定的距离困难光线，其梯度 PIT KS 种子均值如下：

{fixed_table}

![固定光线梯度 CDF](fixed_gradient_cdfs.png)

第一种子：上行用全部参考命中，下行只用 q/Q≥1/2 的命中。预测 CDF 在完全相同的参考位置及入口符号上求平均，与该组经验 W 分布比较。
图中显示的是位置组内的条件分布平均，不是某一个精确 q 的无噪声真值。横轴范围取训练 W 的 99.5% 分位数的 1.15 倍；分布支持仍延伸到正无穷，NLL/PIT 不裁剪尾部。

按 GRU 混合头三个种子的逐光线 PIT KS 均值，误差最大的三条合格光线为 {diagnostics['worst_rays']}。这是评估后诊断选择，未用于训练或 checkpoint 选择。
它们各自只有 {[examples[str(i)]['hits'] for i in diagnostics['worst_rays']]} 次命中。PIT 经验分布也有抽样噪声，不能把这些数值全部解释为模型偏差。

![当前梯度困难光线](worst_gradient_cdfs.png)

另在评估后补充至少 {diagnostics['well_sampled_minimum_hits']} 次命中的困难光线诊断，不改动主指标的 64 次命中门槛：

{well_sampled_table}

![样本充足的困难光线](well_sampled_gradient_cdfs.png)

## 条件采样与前缀一致性

{sampling_table}

第一种子在每个参考命中位置生成一个独立 W 样本，方法间共用均匀随机数。共 {diagnostics['probability_summary']['mixture_gru']['sample_count']} 个样本/方案。
CDF 阈值来自训练 W 的 10%、25%、50%、75%、90%、99% 分位；每条光线先在这些阈值上取最大误差，再对满足命中数门槛的光线平均。
“采样自身 PIT”检查采样器是否遵循网络输出的分布，不是参考拟合指标；好采样器也能精确采样一个有偏网络。Mixture 采用选分量＋截断高斯逆采样，CDF 采用样条解析反演。
联合 NLL 在参考观测上计算：命中使用距离、入口符号与条件速度的联合密度，逃逸使用边缘逃逸概率。它包含距离改进，不能全部归因于梯度头。

对第一种子、全部 280 条光线、两个入口符号、每条 33 个位置和上述 6 个速度阈值检查。下表为条件 CDF 最大变化：
这个网格也包含低概率符号分支和非实际命中位置，属于结构检查，不是渲染平均误差。

{prefix_table}

![梯度前缀一致性](gradient_prefix.png)

三个新头的最大变化为 {diagnostics['causal_max']:.3g}，通过预设 1e-5 阈值。结构检查与拟合精度是不同问题。内部 GRU 步长仍固定为 1/8；上一轮已发现更换其步长会改变距离分布，本轮未重新测试跨步长适用性。

## 训练与复现

{run_table}

![训练曲线](gradient_training.png)

20 项数值/回归测试通过，涵盖缓存与距离状态一致、边界/任意前缀、历史消融、符号对称、冻结权重、无命中光线、CDF 密度/归一化与逆采样。
报告复核九组训练完成、最低验证 NLL 选择、同种子初始化配对、数据与新旧 checkpoint/源码哈希，原距离 checkpoint 保持不变。

```powershell
python -B python/first_hit_se_cli.py causal-gradient --config python/configs/first_hit_se_causal_gradient_sigma002.json
python -B -m unittest python/tests/test_causal_gradient.py python/tests/test_gradient_cdf.py python/tests/test_causal_residual.py -v
```

也可使用 `--phase train/evaluate/diagnose/report` 分阶段运行。输出为 `outputs/first_hit_causal_gradient_sigma002_20261016`。
新头位于 `runs/<variant>_<seed>/best.pt`，格式 `first_hit_causal_gradient`，用 `causal_gradient_study.load_head` 加载，需配合对应种子的冻结 `gru_nll` 主干。
本轮缓存仅用于固定数据拟合；实际逐段调用需要保留历史状态与区间位置。没有改动渲染配置或导出 C++ 模型。

数据 SHA-256：`{spec['dataset_sha256']}`。
'''
    text += step_section
    (output/'summary.md').write_text(text, encoding='utf-8')
    figures_list = ('gradient_fit', 'gradient_calibration', 'fixed_gradient_cdfs', 'worst_gradient_cdfs', 'well_sampled_gradient_cdfs', 'gradient_prefix', 'gradient_training')
    images = ''.join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a></figure>' for name in figures_list)
    page = '<!doctype html><meta charset="utf-8"><title>Causal gradient fitting</title><style>body{max-width:1400px;margin:24px auto;padding:0 16px;font:16px system-ui}img{max-width:100%}pre{white-space:pre-wrap;line-height:1.6}</style>'
    (output/'summary.html').write_text(page+'<h1>残差 GRU 条件梯度拟合</h1><p><a href="summary.md">Markdown</a> · <a href="summary.json">JSON</a></p>'+step_link+images+'<pre>'+html.escape(text)+'</pre>', encoding='utf-8')
    return {'summary': str(output/'summary.md'), 'audit': checks, 'preferred_by_validation_nll': preferred,
            'speed_nll': {m: v['speed_nll'] for m, v in overall.items()}}

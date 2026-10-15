"""Evaluate fixed-budget ablations without changing their training or selection."""
from pathlib import Path
import html
import math
import os
import statistics
import sys
import time

import torch

from renewal.collect import digest, resolve, write_json
from .batch import make_plan, query_many
from .data import load_data, read_json
from .train import load_model, metrics


def _curve_summary(rows):
    return {'profiles': len(rows),
            'mean_max_abs_T': sum(p['max_abs_T'] for p in rows)/len(rows),
            'mean_rmse_T': sum(p['rmse_T'] for p in rows)/len(rows),
            'worst_max_abs_T': max(p['max_abs_T'] for p in rows)}


@torch.no_grad()
def _curves(model, profiles, plans, original_splits, points=129):
    rows = []
    for begin in range(0, len(profiles), 16):
        batch = profiles[begin:begin+16]
        queries = [torch.linspace(0, p['horizon'], points, dtype=torch.float64) for p in batch]
        predicted = query_many(model, [p['features'] for p in batch], queries,
                               plans=plans[begin:begin+16], with_mixture=False)
        for p, q, prediction in zip(batch, queries, predicted):
            observed = ((~p['hit'][:, None]) | (p['distance'][:, None] > q)).double().mean(0)
            difference = prediction['T']-observed
            ray_path = Path(p['source']).parent/'ray.json'
            pixel = read_json(ray_path)['pixel'] if ray_path.exists() else None
            rows.append({'id': p['id'], 'split': p['split'], 'pixel': pixel,
                         'original_split': original_splits.get(tuple(pixel)) if pixel else None,
                         'max_abs_T': float(difference.abs().max()),
                         'rmse_T': float(difference.square().mean().sqrt()),
                         'horizon': p['horizon'], 'q': q.tolist(),
                         'T': prediction['T'].tolist(), 'GP': observed.tolist()})
    return rows


def evaluate(config_path, watch=False):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'experiment.json')
    if digest(config_path) != experiment['specification_sha256']:
        raise ValueError('experiment specification changed')
    data = load_data(resolve(spec['common']['dataset']))
    scene = load_data(resolve(spec['scene_dataset']))
    if digest(resolve(spec['common']['dataset'])) != experiment['dataset_sha256']:
        raise ValueError('training data changed')
    torch.set_num_threads(1)
    splits = {k: [p for p in data['profiles'] if p['split'] == k] for k in ('train', 'validation', 'test')}
    splits['scene'] = scene['profiles']
    plans = {k: [make_plan(p['features'], spec['common']['model']['maximum_step']) for p in ps]
             for k, ps in splits.items()}
    original_splits = {}
    for p in data['profiles']:
        ray = Path(p['source']).parent/'ray.json'
        if ray.exists():
            original_splits[tuple(read_json(ray)['pixel'])] = p['split']
    requests = [{'name': k, 'variant': k, 'seed': None, 'checkpoint': str(resolve(v)), 'baseline': True}
                for k, v in spec['baselines'].items()]
    requests += [{**r, 'checkpoint': str(Path(r['directory'])/spec['primary_checkpoint']), 'baseline': False}
                 for r in experiment['runs']]
    (root/'evaluations').mkdir(exist_ok=True)
    completed = set()
    while True:
        for request in requests:
            name = request['name']
            if name in completed:
                continue
            checkpoint = Path(request['checkpoint'])
            if not request['baseline']:
                report = checkpoint.parent/'run_report.json'
                if not report.exists() or read_json(report)['completed_epochs'] != spec['common']['epochs']:
                    continue
            output = root/'evaluations'/f'{name}.json'
            if output.exists():
                old = read_json(output)
                if old['checkpoint_sha256'] != digest(checkpoint):
                    raise ValueError(f'previous evaluation refers to another checkpoint: {name}')
                completed.add(name)
                continue
            print(f'EVALUATE {name}', flush=True)
            began = time.perf_counter()
            model, bundle = load_model(checkpoint)
            result = {'name': name, 'variant': request['variant'], 'seed': request['seed'],
                      'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                      'epoch': bundle['epoch'], 'parameters': sum(p.numel() for p in model.parameters()),
                      'model_config': model.configuration(), 'splits': {}}
            for split, ps in splits.items():
                curves = _curves(model, ps, plans[split], original_splits)
                likelihood = metrics(model, ps, batch_size=16, plans=plans[split])
                result['splits'][split] = {'curves': _curve_summary(curves), 'likelihood': likelihood,
                                           'profiles': curves}
                camera = [r for r in curves if r['pixel'] is not None]
                if camera:
                    result['splits'][split]['camera_curves'] = _curve_summary(camera)
                if split == 'scene':
                    result['splits'][split]['curves_by_original_split'] = {
                        k: _curve_summary([p for p in curves if p['original_split'] == k])
                        for k in ('train', 'validation', 'test')}
            result['evaluation_seconds'] = time.perf_counter()-began
            # Only the complete result is exposed to the report builder.
            temporary = output.with_suffix('.pending.json')
            write_json(temporary, result)
            temporary.replace(output)
            completed.add(name)
            print(f'EVALUATED {len(completed)}/{len(requests)} {name}: '
                  f'test mean-max={result["splits"]["test"]["curves"]["mean_max_abs_T"]:.6f}, '
                  f'{result["evaluation_seconds"]:.1f} seconds', flush=True)
        if len(completed) == len(requests) or not watch:
            break
        time.sleep(20)
    return {'evaluated': len(completed), 'expected': len(requests), 'output': str(root/'evaluations')}


def _flat_metrics(result):
    s = result['splits']
    return {
        'train_mean_max_T_error': s['train']['curves']['mean_max_abs_T'],
        'validation_mean_max_T_error': s['validation']['curves']['mean_max_abs_T'],
        'test_mean_max_T_error': s['test']['curves']['mean_max_abs_T'],
        'test_mean_rmse_T': s['test']['curves']['mean_rmse_T'],
        'test_camera_mean_max_T_error': s['test']['camera_curves']['mean_max_abs_T'],
        'scene_heldout_mean_max_T_error': s['scene']['curves_by_original_split']['test']['mean_max_abs_T'],
        'test_joint_nll': s['test']['likelihood']['joint_nll'],
        'test_distance_nll': s['test']['likelihood']['distance_nll'],
        'test_speed_nll': s['test']['likelihood']['speed_nll_on_hits'],
        'test_speed_pit_ks': s['test']['likelihood']['speed_pit_ks'],
    }


def _stats(values):
    return {'mean': statistics.mean(values), 'standard_deviation': statistics.stdev(values) if len(values) > 1 else None,
            'minimum': min(values), 'maximum': max(values), 'count': len(values)}


def _plots(root, experiment, results):
    # Keep the optional plotting installation and all font caches in this run.
    dependencies = root/'plot_dependencies'
    if dependencies.exists():
        sys.path.insert(0, str(dependencies))
    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    plots = root/'plots'
    plots.mkdir(exist_ok=True)
    seed = experiment['specification']['seeds'][1]
    variants = [v['name'] for v in experiment['specification']['variants']]
    names = [k for k in ('rno_v2', 'gru_v2') if k in results]
    names += [f'{v}_s{seed}' for v in variants if f'{v}_s{seed}' in results]
    colors = ['#929292', '#9b7d46', '#d45538', '#2564b5', '#2e865d', '#834ab3']
    labels = {name: name.split('_s2026')[0] for name in names}
    files = []
    if not names:
        return files
    reference = results[names[0]]['splits']['scene']['profiles']
    for index, ray in enumerate(reference):
        fig, (ax, residual) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
        q, observed = ray['q'], ray['GP']
        n, z = 4096, 1.959963984540054
        center = [(t+z*z/(2*n))/(1+z*z/n) for t in observed]
        half = [z*math.sqrt(t*(1-t)/n+z*z/(4*n*n))/(1+z*z/n) for t in observed]
        ax.fill_between(q, [c-h for c,h in zip(center,half)], [c+h for c,h in zip(center,half)],
                        color='#555555', alpha=.13, label='GP pointwise 95% MC interval')
        ax.plot(q, observed, color='#111111', linewidth=2, label='GP')
        for color, name in zip(colors, names):
            p = results[name]['splits']['scene']['profiles'][index]
            assert p['id'] == ray['id']
            ax.plot(q, p['T'], color=color, linewidth=1.5, label=labels[name])
            residual.plot(q, [a-b for a,b in zip(p['T'], observed)], color=color, linewidth=1.3)
        ax.set(title=f"Pixel {tuple(ray['pixel'])} / original {ray['original_split']} / seed {seed}",
               ylabel='Transmittance T', ylim=(-.025, 1.025))
        ax.legend(loc='best', fontsize=8, ncol=2)
        residual.axhline(0, color='#444444', linewidth=.7)
        residual.set(xlabel='q = distance from medium entry / ell', ylabel='Model - GP')
        for axis in (ax, residual):
            axis.grid(alpha=.2)
        path = plots/f"pixel_{ray['pixel'][0]}_{ray['pixel'][1]}.svg"
        fig.savefig(path)
        if ray['pixel'] == [368, 368]:
            fig.savefig(path.with_suffix('.png'), dpi=140)
        plt.close(fig)
        files.append({'path': path.relative_to(root).as_posix(), 'pixel': ray['pixel'], 'original_split': ray['original_split']})
    histories = {}
    for record in experiment['runs']:
        path = Path(record['directory'])/'history.json'
        if record['seed'] == seed and path.exists():
            histories[record['variant']] = read_json(path)
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        for i, variant in enumerate(variants):
            if variant not in histories:
                continue
            history = histories[variant]
            x = [r['epoch'] for r in history]
            axes[0].plot(x, [r['training_nll'] for r in history], label=variant, color=colors[i+2], alpha=.8)
            axes[1].plot(x, [r['validation']['joint_nll'] for r in history], label=variant, color=colors[i+2])
        for ax, title in zip(axes, ('Training likelihood (sampled per epoch)', 'Validation joint NLL')):
            ax.set(xlabel='Continuation epoch', ylabel='NLL', title=title)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.savefig(plots/'learning_curves.svg')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for axis, split, title in zip(axes, ('test', 'scene'), ('Original test set (26 profiles)', 'Scene held-out rays (3 profiles)')):
        for i, variant in enumerate(variants):
            values = []
            for j, seed_id in enumerate(experiment['specification']['seeds']):
                key = f'{variant}_s{seed_id}'
                if key not in results:
                    continue
                result = results[key]['splits'][split]
                metric = result['curves'] if split == 'test' else result['curves_by_original_split']['test']
                values.append(metric['mean_max_abs_T'])
                axis.scatter(i+(j-1)*.09, values[-1], color=colors[i+2], alpha=.55, s=22)
            if values:
                axis.errorbar(i, statistics.mean(values), yerr=statistics.stdev(values) if len(values)>1 else 0.,
                              color=colors[i+2], marker='D', capsize=5, linewidth=1.8)
        axis.set(xticks=range(len(variants)), xticklabels=variants, ylabel='Mean maximum absolute T error', title=title)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Dots: individual seeds; diamonds and bars: mean and sample standard deviation')
    fig.savefig(plots/'aggregate_errors.svg')
    plt.close(fig)
    return files


def report(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'experiment.json')
    results = {p.stem: read_json(p) for p in sorted((root/'evaluations').glob('*.json'))
               if not p.name.endswith('.pending.json')}
    flat = {k: _flat_metrics(v) for k, v in results.items()}
    aggregates = {}
    for variant in spec['variants']:
        group = [r for r in results.values() if r['variant'] == variant['name']]
        if group:
            aggregates[variant['name']] = {k: _stats([flat[r['name']][k] for r in group]) for k in flat[group[0]['name']]}
    paired = {}
    for variant in ('h16_curve', 'h32_nll', 'h64_nll'):
        pairs = [(f'{variant}_s{seed}', f'h16_nll_s{seed}') for seed in spec['seeds']
                 if f'{variant}_s{seed}' in flat and f'h16_nll_s{seed}' in flat]
        if pairs:
            paired[variant] = {}
            for key in flat[pairs[0][0]]:
                values = [flat[a][key]-flat[b][key] for a,b in pairs]
                paired[variant][key] = {**_stats(values), 'improved_seeds': sum(v < 0 for v in values),
                                        'differences': dict(zip([results[a]['seed'] for a,b in pairs], values))}
    completed = sum(record['name'] in results for record in experiment['runs'])
    summary = {'completed_runs': completed, 'expected_runs': len(experiment['runs']),
               'specification': spec, 'aggregate': aggregates, 'paired_differences_vs_h16_nll': paired,
               'individual': flat, 'representative_plot_seed': spec['seeds'][1],
               'uncertainty': 'Mean and sample standard deviation across 3 seeds, not a confidence interval.',
               'scope': experiment['scope']}
    if completed == len(experiment['runs']):
        summary['relative_improvement_percent'] = {
            name: {key: 100*(1-group[key]['mean']/aggregates['h16_nll'][key]['mean'])
                   for key in ('test_mean_max_T_error', 'scene_heldout_mean_max_T_error')}
            for name, group in aggregates.items() if name != 'h16_nll'}
    if completed == len(experiment['runs']):
        summary['relative_improvement_percent'] = {
            name: {key: 100*(1-group[key]['mean']/aggregates['h16_nll'][key]['mean'])
                   for key in ('test_mean_max_T_error', 'scene_heldout_mean_max_T_error')}
            for name, group in aggregates.items() if name != 'h16_nll'}
    audit_path = root/'numerical_audit.json'
    if audit_path.exists():
        summary['numerical_audit'] = read_json(audit_path)
    write_json(root/'summary.json', summary)
    plots = _plots(root, experiment, results)
    e = html.escape
    def value(stats):
        return f'{stats["mean"]:.6f}' + (f' ± {stats["standard_deviation"]:.6f}' if stats['standard_deviation'] is not None else ' (n=1)')
    columns = [('train_mean_max_T_error', '训练：平均最大 T 误差'),
               ('test_mean_max_T_error', '原测试 26 条：平均最大 T 误差'),
               ('test_mean_rmse_T', '原测试：平均 RMSE'),
               ('scene_heldout_mean_max_T_error', '场景原测试 3 条：平均最大 T 误差'),
               ('test_speed_nll', '原测试：梯度速度 NLL')]
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RNO 曲线损失与状态容量对照</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;color:#1e293b;max-width:1450px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#f1f5f9}img{max-width:1100px;width:100%}code{font-size:13px}.scroll{overflow:auto}small{color:#64748b}a{color:#2564b5}</style>',
        '<h1>RNO：曲线损失与历史状态容量对照</h1>',
        f'<p>已完成训练及评估 <strong>{completed}/{len(experiment["runs"])}</strong> 次；每组 3 个随机种子。以下误差均为绝对概率误差。</p>',
        '<p>固定原 v2 数据：152 条训练、25 条验证、26 条测试剖面。全部从相同 v2 函数开始继续训练，'
        'h32/h64 通过增加状态坐标并将新增坐标到原状态和输出的连接初始化为零，保持初始预测一致；新增坐标保留随机动态，可在后续训练中学习。'
        '这是现有模型附近的容量扩展实验，不是从头训练的容量极限测试。</p>',
        '<p>每次固定 96 轮 / 480 次更新，批量 32 条光线，每条每轮抽取 256 个事件标签；'
        'AdamW，学习率按相同余弦日程从 3e-4 降至 3e-5。主要结果一律使用最后一轮权重；验证 NLL 最优权重另存，不参与下面的主要比较。</p>',
        '<p>曲线组损失为原联合 NLL + 10 × MSE(T, Tref)，每条光线取 33 个均匀查询点（含端点），'
        'Tref 使用该条训练光线全部 1024 个 GP 标签。未新增 GP 数据，未使用测试标签调整训练配置。容量组保持原联合 NLL。</p>',
        '<p>± 表示 3 个种子之间的样本标准差，不是置信区间。曲线指标先对每条光线计算，再对剖面平均；'
        'GP Monte Carlo 噪声与剩余路径离散误差仍存在。场景 8 条光线中 4 条属于原训练、1 条属于原验证、3 条属于原测试；表中场景测试指标只统计最后这 3 条。</p>',
        '<h2>各组统计</h2><div class="scroll"><table><tr><th>模型 / 损失</th><th>种子数</th>'+
        ''.join(f'<th>{label}</th>' for _,label in columns)+'</tr>']
    if 'relative_improvement_percent' in summary:
        effects = summary['relative_improvement_percent']
        key = 'test_mean_max_T_error'
        conclusion = (
            '<p><strong>本轮结果：</strong>相对同预算的 h16 原损失对照，'
            f'曲线监督将完整测试集的平均最大透射率误差降低 {effects["h16_curve"][key]:.1f}%，'
            f'h32 降低 {effects["h32_nll"][key]:.1f}%，h64 降低 {effects["h64_nll"][key]:.1f}%。'
            '三个种子在完整测试集上的改善方向一致；在场景 3 条原测试光线上，曲线监督的平均收益更大。'
            '梯度速度 NLL 均约为 1.2124，困难像素的尾部偏差仍然存在。'
            '这支持训练目标与状态容量均有影响，尚不能归因于单一瓶颈；本轮没有检验增加不同光线的效果。</p>')
        content.insert(4, conclusion)
    if 'relative_improvement_percent' in summary:
        effects = summary['relative_improvement_percent']
        key = 'test_mean_max_T_error'
        conclusion = (
            '<p><strong>本轮结果：</strong>相对同预算的 h16 原损失对照，'
            f'曲线监督将完整测试集的平均最大透射率误差降低 {effects["h16_curve"][key]:.1f}%，'
            f'h32 降低 {effects["h32_nll"][key]:.1f}%，h64 降低 {effects["h64_nll"][key]:.1f}%。'
            '三个种子在完整测试集上的改善方向一致；在场景 3 条原测试光线上，曲线监督的平均收益更大。'
            '梯度速度 NLL 均约为 1.2124，困难像素的尾部偏差仍然存在。'
            '这支持训练目标与状态容量均有影响，尚不能归因于单一瓶颈；本轮没有检验增加不同光线的效果。</p>')
        content.insert(4, conclusion)
    for name in ('rno_v2', 'gru_v2'):
        if name in flat:
            content.append(f'<tr><td>{name}（原权重）</td><td>—</td>'+''.join(f'<td>{flat[name][key]:.6f}</td>' for key,_ in columns)+'</tr>')
    for variant, group in aggregates.items():
        content.append(f'<tr><td>{variant}</td><td>{group[columns[0][0]]["count"]}</td>'+''.join(f'<td>{value(group[key])}</td>' for key,_ in columns)+'</tr>')
    content.append('</table></div><img src="plots/aggregate_errors.svg" alt="Paired seed curve errors"><h2>相对同种子 h16 原损失组的变化</h2><p>负值表示误差或 NLL 降低；同时检查透射率和梯度，避免只看单项收益。</p><div class="scroll"><table><tr><th>改动</th>')
    content.append(''.join(f'<th>{label} Δ</th>' for _,label in columns)+'</tr>')
    for variant, group in paired.items():
        content.append(f'<tr><td>{variant}</td>'+''.join(f'<td>{value(group[key])}<br><small>{group[key]["improved_seeds"]}/{group[key]["count"]} 个种子改善</small></td>' for key,_ in columns)+'</tr>')
    content.append('</table></div><h2>每次训练的完整指标</h2><div class="scroll"><table><tr><th>运行</th><th>参数量</th><th>测试最大误差均值</th><th>测试联合 NLL</th><th>测试距离 NLL</th><th>测试速度 PIT KS</th></tr>')
    for record in experiment['runs']:
        name = record['name']
        if name in results:
            m = flat[name]
            content.append(f'<tr><td><a href="evaluations/{e(name)}.json">{e(name)}</a></td><td>{results[name]["parameters"]}</td>'+''.join(f'<td>{m[k]:.6f}</td>' for k in ('test_mean_max_T_error','test_joint_nll','test_distance_nll','test_speed_pit_ks'))+'</tr>')
    content.append('</table></div>')
    if 'numerical_audit' in summary:
        content.append('<h2>代表种子的数值与 C++ 检查</h2><p>分段检查覆盖固定的 4 条光线；统计的是模型自身的数值变化，不是与 GP 的拟合误差。</p><table><tr><th>模型</th><th>8 倍细分最大 ΔT</th><th>内部步长减半最大 ΔT</th><th>C++/Python 最大预测差（含分布参数）</th></tr>')
        for name, audit in summary['numerical_audit']['models'].items():
            content.append(f'<tr><td>{e(name)}</td><td>{audit["maximum_subdivision_T"]:.3g}</td><td>{audit["maximum_half_step_T"]:.3g}</td><td>{audit["maximum_cpp_python_error"]:.3g}</td></tr>')
        content.append('</table><p><a href="numerical_audit.json">完整数值检查</a></p>')
    if (root/'plots'/'learning_curves.svg').exists():
        content.append('<h2>固定展示种子的学习曲线</h2><p>展示种子预先固定为第二个种子 20261022。左图为联合似然，不含附加曲线损失；不是按测试效果选择的运行。</p><img src="plots/learning_curves.svg" alt="Training and validation likelihood">')
    content.append('<h2>σ=0.05 场景逐像素比较</h2><p>使用同一批 4096 个 GP 参考样本。阴影为点态 95% Monte Carlo 区间，未计入连续路径离散误差；下方小图显示预测与参考的差。</p>')
    for plot in plots:
        content.append(f'<p><strong>像素 {plot["pixel"]} / 原 {plot["original_split"]}</strong></p><img src="{plot["path"]}" alt="Camera first-hit transmittance">')
    content.append('<p><a href="summary.json">汇总及逐种子差值</a> · <a href="experiment.json">完整实验定义与初始化验证</a> · '
                   '<a href="implementation_sha256.json">训练实现哈希</a> · '
                   '<a href="../first_hit_se_render_scene_sigma005_comparison_20261013/comparison.html">原 σ=0.05 对比</a></p></html>')
    if (root/'validation.json').exists():
        content.insert(-1, '<p><a href="validation.json">固定预算、数据与原权重哈希、最终 checkpoint 核对</a></p>')
    if (root/'validation.json').exists():
        content.insert(-1, '<p><a href="validation.json">固定预算、数据与原权重哈希、最终 checkpoint 核对</a></p>')
    (root/'comparison.html').write_text('\n'.join(content), encoding='utf-8')
    return {'html': str(root/'comparison.html'), 'completed_runs': completed, 'aggregate': aggregates,
            'paired_differences': paired}


@torch.no_grad()
def audit(config_path):
    from renewal.profiles import subdivide_features
    from .verify_cpp import verify_cpp
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'experiment.json')
    data = load_data(resolve(spec['common']['dataset']))
    pixels = ((144, 368), (368, 368), (376, 96), (296, 336))
    probes = []
    for p in data['profiles']:
        path = Path(p['source']).parent/'ray.json'
        if path.exists() and tuple(read_json(path)['pixel']) in pixels:
            probes.append((p, read_json(path)['pixel']))
    assert len(probes) == len(pixels)
    records = {'representative_seed': spec['seeds'][1], 'pixels': pixels, 'models': {}}
    for record in experiment['runs']:
        if record['seed'] != spec['seeds'][1]:
            continue
        checkpoint = Path(record['directory'])/spec['primary_checkpoint']
        if not (checkpoint.parent/'run_report.json').exists():
            continue
        name = record['name']
        cached = root/'checks'/name/'audit.json'
        if cached.exists():
            saved = read_json(cached)
            if saved['checkpoint_sha256'] != digest(checkpoint):
                raise ValueError('audit checkpoint changed')
            records['models'][name] = saved
            continue
        torch.set_num_threads(1)
        model, _ = load_model(checkpoint)
        checks = []
        print(f'AUDIT {name}', flush=True)
        for p, pixel in probes:
            features = p['features']
            q = torch.linspace(0, p['horizon'], 65, dtype=torch.float64)
            base = model.query(features, q, with_mixture=False)
            changed = model.query(subdivide_features(features, torch.full((len(features),), 8, dtype=torch.long)),
                                  q, with_mixture=False)
            finer = model.query(features, q, with_mixture=False, maximum_step=model.config.maximum_step/2)
            checks.append({'id': p['id'], 'pixel': pixel,
                           'subdivision_max_abs_T': float((changed['T']-base['T']).abs().max()),
                           'subdivision_max_abs_H': float((changed['log_T']-base['log_T']).abs().max()),
                           'half_step_max_abs_T': float((finer['T']-base['T']).abs().max())})
        cpp = verify_cpp(checkpoint, resolve(spec['common']['dataset']),
                         resolve('build/Release/macrofacet_query_first_hit_se.exe'),
                         root/'checks'/name/'cpp', limit=3, trials=512)
        value = {'checkpoint_sha256': digest(checkpoint), 'profiles': checks,
                 'maximum_subdivision_T': max(p['subdivision_max_abs_T'] for p in checks),
                 'maximum_half_step_T': max(p['half_step_max_abs_T'] for p in checks),
                 'cpp': cpp,
                 'maximum_cpp_python_error': max(max(p['maximum_errors'].values()) for p in cpp['profiles'])}
        write_json(cached, value)
        records['models'][name] = value
        write_json(root/'numerical_audit.json', records)
    write_json(root/'numerical_audit.json', records)
    return records

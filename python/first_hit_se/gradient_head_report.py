"""Read-only aggregation and figures for frozen-backbone gradient-head runs."""
import html
from pathlib import Path
import statistics
import sys

from renewal.collect import resolve, write_json
from .data import read_json


def flat(record):
    splits = record['splits']
    result = {}
    for split in ('train', 'validation', 'test', 'scene'):
        for key in ('speed_nll_on_hits', 'speed_pit_ks'):
            result[f'{split}_{key}'] = splits[split][key]
    result['test_mean_profile_pit_ks_min64'] = splits['test']['mean_profile_pit_ks_min64']
    for key in ('speed_nll_on_hits', 'speed_pit_ks'):
        result[f'scene_heldout_{key}'] = splits['scene']['by_original_split']['test'][key]
    return result


def _summary(values):
    return {'mean': statistics.mean(values), 'standard_deviation': statistics.stdev(values) if len(values)>1 else 0.,
            'minimum': min(values), 'maximum': max(values), 'count': len(values)}


def report(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'experiment.json')
    records = {}
    for run in experiment['runs']:
        path = root/'evaluations'/f'{run["name"]}.json'
        if path.exists():
            records[run['name']] = read_json(path)
    source = read_json(root/'evaluations'/'source.json')
    individual = {name: flat(value) for name, value in records.items()}
    grouped = {str(k): [r for r in records.values() if r['components'] == k] for k in spec['components']}
    aggregate = {k: {metric: _summary([individual[r['name']][metric] for r in group])
                     for metric in flat(source)} for k, group in grouped.items() if group}
    paired = {}
    for k in spec['components'][1:]:
        pairs = [(f'k{k}_s{s}', f'k8_s{s}') for s in spec['seeds']
                 if f'k{k}_s{s}' in records and f'k8_s{s}' in records]
        if pairs:
            paired[str(k)] = {}
            for metric in flat(source):
                differences = {a.split('_s')[1]: individual[a][metric]-individual[b][metric] for a,b in pairs}
                paired[str(k)][metric] = {**_summary(list(differences.values())), 'differences': differences,
                                          'improved_seeds': sum(v < 0 for v in differences.values())}
    validation_selected = {}
    for k, group in grouped.items():
        if group:
            validation_selected[k] = {
                metric: _summary([r['validation_selected']['splits']['test'][metric] for r in group])
                for metric in ('speed_nll_on_hits', 'speed_pit_ks')}
    summary = {'completed_runs': len(records), 'expected_runs': len(experiment['runs']),
               'source': flat(source), 'aggregate': aggregate, 'paired_differences_vs_k8': paired,
               'validation_selected_test': validation_selected,
               'individual': individual, 'specification': spec,
               'backbone_unchanged_all': all(r['training']['backbone_unchanged'] for r in records.values()),
               'max_abs_T_change': max((r['frozen_prediction_max_abs_difference']['T']
                                       for r in records.values()), default=0.),
               'training_seconds': sum(r['training']['training_seconds'] for r in records.values()),
               'scope': 'Conditional speed at reference hits; transverse SE law unchanged. Primary: fixed final epoch.',
               'uncertainty': 'Sample standard deviation across training seeds; not uncertainty over GP labels or scenes.'}
    write_json(root/'summary.json', summary)
    images = _plots(root, spec, source, records)
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<title>First-hit 梯度头单独训练</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1300px;margin:30px auto;padding:0 24px;color:#1e293b}'
        'table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        'th{background:#f1f5f9}img{width:100%;max-width:1150px}code{font-size:13px}</style>',
        '<h1>固定 RNO 主干，仅训练梯度头</h1>',
        f'<p>已完成 {len(records)}/{len(experiment["runs"])} 组；混合分量 K=8/16/32，各 3 个随机种子。'
        '共同起点为上一轮 h16 NLL 对照的固定代表种子 20261022，未按本次梯度指标挑选起点。</p>',
        '<p>状态初始化、连续状态演化、hazard 全部冻结，仅更新 mixture_head。'
        '缓存参考首碰位置和入口符号对应的隐藏状态；训练只使用原训练集首碰样本。'
        'K=16/32 采用组件拆分，并加入最大 0.002 的对称均值扰动。'
        '每组完整遍历训练首碰样本 256 次，批量 4096，余弦学习率 3e-4→3e-5；主结果固定取最后一轮。</p>',
        '<p>下面的 NLL/PIT 评估条件沿射线梯度分布，数值越低越好。均值 ± 样本标准差反映三个训练种子的变化。'
        '横向梯度继续使用固定 SE 解析分布；此处未评估由模型采样碰撞位置后的完整法线分布。</p>',
        f'<p>冻结参数完全一致：{summary["backbone_unchanged_all"]}；8 条场景光线、每条 129 个查询点的最大透射率变化：'
        f'{summary["max_abs_T_change"]:.3g}。</p>',
        '<p><a href="summary.json">汇总 JSON</a> · <a href="experiment.json">实验设置</a> · '
        '<a href="cache_manifest.json">数据及缓存校验</a></p>',
        '<h2>原数据划分：独立测试结果</h2>']
    if len(records) == len(experiment['runs']) and all(
            row['test_speed_nll_on_hits']['mean'] >= flat(source)['test_speed_nll_on_hits']
            for row in aggregate.values()):
        content.insert(-1, '<p><strong>本轮单独训练梯度头没有改善独立测试 NLL。</strong>'
                       '三组的测试 NLL 均高于共同起点。增加分量的收益应与同预算 K=8 比较，'
                       '同时核对 PIT 校准；不能把训练集 NLL 的下降直接解释为泛化改善。</p>')
    keys = [('train_speed_nll_on_hits','训练 NLL'),('validation_speed_nll_on_hits','验证 NLL'),
            ('test_speed_nll_on_hits','测试 NLL'),('test_speed_pit_ks','测试 PIT KS'),
            ('test_mean_profile_pit_ks_min64','逐光线平均 PIT KS（≥64 次首碰）')]
    _table(content, keys, flat(source), aggregate)
    content.append('<h2>σ=0.05 场景光线</h2><p>8 条光线中，4 条属于原训练集、1 条属于验证集、3 条属于测试集。'
                   '独立测试列只使用后者；重新生成 GP 样本不会改变光线本身所属的数据划分。</p>')
    _table(content, [('scene_speed_nll_on_hits','全部 8 条 NLL'),('scene_speed_pit_ks','全部 8 条 PIT KS'),
                     ('scene_heldout_speed_nll_on_hits','原测试 3 条 NLL'),
                     ('scene_heldout_speed_pit_ks','原测试 3 条 PIT KS')], flat(source), aggregate)
    content.append('<h2>同种子配对：相对 K=8</h2><p>差值为新模型减 K=8，负值表示改善。</p>'
                   '<table><tr><th>K</th><th>测试 NLL 差值</th><th>NLL 改善种子</th>'
                   '<th>测试 PIT KS 差值</th><th>PIT 改善种子</th></tr>')
    for k, value in paired.items():
        a, b = value['test_speed_nll_on_hits'], value['test_speed_pit_ks']
        content.append(f'<tr><td>{k}</td><td>{a["mean"]:+.7f} ± {a["standard_deviation"]:.7f}</td>'
                       f'<td>{a["improved_seeds"]}/{a["count"]}</td>'
                       f'<td>{b["mean"]:+.6f} ± {b["standard_deviation"]:.6f}</td>'
                       f'<td>{b["improved_seeds"]}/{b["count"]}</td></tr>')
    content.append('</table><h2>按验证集 NLL 选取的检查点（辅助结果）</h2>'
                   '<p>该表用于检查最后一轮是否过拟合；所有选择只使用验证集，未按测试指标挑选轮次。</p>'
                   '<table><tr><th>K</th><th>测试 NLL</th><th>测试 PIT KS</th></tr>')
    for k, value in validation_selected.items():
        content.append(f'<tr><td>{k}</td><td>{value["speed_nll_on_hits"]["mean"]:.6f}</td>'
                       f'<td>{value["speed_pit_ks"]["mean"]:.6f}</td></tr>')
    content.append('</table><h2>近零速度与尾部概率</h2><p>网络概率在参考首碰位置和符号上取平均；'
                   'GP 列是同一批首碰标签中的经验频率。该表仍属于条件梯度头诊断，'
                   '未使用模型自己的首碰位置分布进行积分。</p>')
    for label, split, subgroup in [('完整测试集','test',None),('场景原测试 3 条','scene','test')]:
        def threshold_rows(record):
            row = record['splits'][split]
            if subgroup:
                row = row['by_original_split'][subgroup]
            return row['threshold_calibration']
        reference = threshold_rows(source)
        content.append(f'<h3>{label}</h3><table><tr><th>事件</th><th>GP</th><th>起点模型</th>'
                       +''.join(f'<th>K={k}</th>' for k, group in grouped.items() if group)+'</tr>')
        for i, value in enumerate(reference):
            tail = value['W'] >= 2
            def prob(x):
                return 1-x if tail else x
            content.append(f'<tr><td>W {">" if tail else "≤"} {value["W"]:g}</td>'
                           f'<td>{prob(value["observed_cdf"]):.6f}</td>'
                           f'<td>{prob(value["predicted_cdf"]):.6f}</td>')
            for k, group in grouped.items():
                if group:
                    values = [prob(threshold_rows(r)[i]['predicted_cdf']) for r in group]
                    content.append(f'<td>{statistics.mean(values):.6f} ± '
                                   f'{statistics.stdev(values) if len(values)>1 else 0.:.6f}</td>')
            content.append('</tr>')
        content.append('</table>')
    content.append('<h2>诊断图</h2><p>图中的代表种子固定为 20261032。PIT 曲线绘制经验 CDF 减去均匀分布 CDF，'
                   '零线表示理想校准。分组 CDF 的采样波动随首碰样本数变化。</p>')
    for filename, label in images:
        content.append(f'<p>{html.escape(label)}</p><img src="plots/{filename}" alt="{html.escape(label)}">')
    content.append('<h2>逐组结果</h2><table><tr><th>模型</th><th>轮数</th><th>更新次数</th>'
                   '<th>可训练参数</th><th>初始训练 NLL 变化</th><th>冻结参数一致</th></tr>')
    for name, r in records.items():
        t = r['training']
        content.append(f'<tr><td><a href="evaluations/{name}.json">{name}</a></td><td>{t["epochs"]}</td>'
                       f'<td>{t["optimizer_updates"]}</td><td>{t["trainable_parameters"]}</td>'
                       f'<td>{t["initialization"]["training_nll_change"]:+.9f}</td>'
                       f'<td>{t["backbone_unchanged"]}</td></tr>')
    content.append('</table><p>这些结果检验的是固定主干、既有分布拆分初始化下的继续训练。'
                   '若增加分量未改善测试分布，不能据此证明其他初始化或训练预算下的容量上限。</p>')
    if (root/'cpp_verification.json').exists():
        content.append('<p><a href="cpp_verification.json">C++ 导出与采样检查</a></p>')
    if (root/'validation.json').exists():
        content.append('<p><a href="validation.json">完整实验校验</a></p>')
    if (root/'component_diagnostics.json').exists():
        content.append('<p><a href="component_diagnostics.json">新增分量分化检查</a>：'
                       '在固定代表种子的训练首碰探针上，按原分量分组测量子分量均值跨度。</p>')
    content.append('</html>')
    (root/'comparison.html').write_text('\n'.join(content), encoding='utf-8')
    return {'completed_runs': len(records), 'output': str(root/'comparison.html'), 'aggregate': aggregate}


def _table(content, keys, source, aggregate):
    content.append('<table><tr><th>模型</th>'+''.join(f'<th>{label}</th>' for _,label in keys)+'</tr>')
    content.append('<tr><td>冻结前共同起点 K=8</td>'+''.join(f'<td>{source[k]:.6f}</td>' for k,_ in keys)+'</tr>')
    for name, row in aggregate.items():
        content.append(f'<tr><td>K={name}，仅训练梯度头</td>'+''.join(
            f'<td>{row[k]["mean"]:.6f} ± {row[k]["standard_deviation"]:.6f}</td>' for k,_ in keys)+'</tr>')
    content.append('</table>')


def _plots(root, spec, source, records):
    # Reuse the workspace-local plotting installation from the preceding
    # experiment; do not modify the training environment or install packages.
    dependency = resolve('outputs/first_hit_se_ablation_20261013/plot_dependencies')
    if dependency.exists():
        sys.path.insert(0, str(dependency))
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(root/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    directory = root/'plots'
    directory.mkdir(exist_ok=True)
    chosen = {'source': source}
    chosen.update({f'K={k}': records[name] for k in spec['components']
                   if (name := f'k{k}_s{spec["representative_seed"]}') in records})
    colors = {'source': 'black', 'K=8': '#1f77b4', 'K=16': '#ff7f0e', 'K=32': '#2ca02c'}
    images = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, r in chosen.items():
        for axis, split in zip(axes, ('test', 'scene')):
            row = r['splits'][split]
            x = np.asarray(row['pit_grid'])
            axis.plot(x, np.asarray(row['pit_ecdf'])-x, label=label, color=colors.get(label))
            axis.set(title=f'{split}: PIT calibration', xlabel='PIT', ylabel='Empirical CDF - uniform CDF')
            axis.grid(alpha=.2)
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.7); axis.legend()
    fig.savefig(directory/'pit.svg')
    fig.savefig(directory/'pit.png', dpi=140)
    plt.close(fig)
    images.append(('pit.svg','完整测试集与场景全部 8 条光线的 PIT 校准'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k in spec['components']:
        path = resolve(spec['model_directory'])/f'k{k}_s{spec["representative_seed"]}'/'history.json'
        if not path.exists():
            continue
        history = read_json(path)
        axes[0].plot([r['epoch'] for r in history], [r['training_speed_nll'] for r in history], label=f'K={k}')
        val = [r for r in history if r['validation']]
        axes[1].plot([r['epoch'] for r in val], [r['validation']['speed_nll_on_hits'] for r in val], label=f'K={k}')
    for axis, split in zip(axes, ('train', 'validation')):
        axis.axhline(source['splits'][split]['speed_nll_on_hits'], color='black', linestyle='--', label='source')
        axis.set(title=split, xlabel='epoch', ylabel='Conditional speed NLL')
        axis.grid(alpha=.2); axis.legend()
    fig.savefig(directory/'training.svg')
    fig.savefig(directory/'training.png', dpi=140)
    plt.close(fig)
    images.append(('training.svg','完整训练过程及验证集 NLL'))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for i, profile in enumerate(source['splits']['scene']['profiles']):
        axis = axes.flat[i]
        for label, r in chosen.items():
            row = r['splits']['scene']['profiles'][i]
            if row['hits']:
                x = np.asarray(r['splits']['scene']['pit_grid'])
                axis.plot(x, np.asarray(row['pit_ecdf'])-x, label=label, color=colors.get(label))
        axis.set(title=f'{profile["pixel"]}, {profile["original_split"]}, n={profile["hits"]}', xlabel='PIT', ylabel='CDF error')
        axis.axhline(0, color='black', linewidth=.7); axis.grid(alpha=.2)
    axes.flat[-1].legend()
    fig.savefig(directory/'scene_pit.svg'); plt.close(fig)
    images.append(('scene_pit.svg','场景逐光线校准；零首碰光线没有梯度标签'))
    return images

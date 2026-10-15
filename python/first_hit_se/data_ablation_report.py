"""Aggregate matched-budget data-coverage and per-ray-sampling experiments."""
import html
from pathlib import Path
import statistics
import sys
import os

from renewal.collect import resolve, write_json
from .data import read_json
from .gradient_head_report import flat as base_flat, _summary


LABELS = {'baseline': '原数据 120 × 1024', 'coverage': '增加光线 480 × 1024',
          'samples': '增加每条采样 120 × 4096'}


def flat(record):
    result = base_flat(record)
    for family in ('camera', 'affine'):
        profiles = [p for p in record['splits']['test']['profiles']
                    if (p['pixel'] is not None) == (family == 'camera') and p['hits']]
        result[f'test_{family}_speed_nll_on_hits'] = (
            sum(p['hits']*p['speed_nll_on_hits'] for p in profiles)/sum(p['hits'] for p in profiles))
        result[f'test_{family}_mean_profile_pit_ks_min64'] = statistics.mean(
            p['speed_pit_ks'] for p in profiles if p['hits'] >= 64)
    return result


def report(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'training_experiment.json')
    source = read_json(root/'evaluations'/'source.json')
    records = {}
    for r in experiment['runs']:
        path = root/'evaluations'/f'{r["name"]}.json'
        if path.exists():
            records[r['name']] = read_json(path)
    individual = {name: flat(r) for name,r in records.items()}
    grouped = {v: [r for r in records.values() if r['variant']==v] for v in LABELS}
    aggregate = {v: {key: _summary([individual[r['name']][key] for r in group]) for key in flat(source)}
                 for v,group in grouped.items() if group}
    paired = {}
    for variant in ('coverage','samples'):
        pairs = [(f'{variant}_s{s}',f'baseline_s{s}') for s in spec['seeds']
                 if f'{variant}_s{s}' in records and f'baseline_s{s}' in records]
        if pairs:
            paired[variant] = {}
            for key in flat(source):
                values = {a.split('_s')[1]: individual[a][key]-individual[b][key] for a,b in pairs}
                paired[variant][key] = {**_summary(list(values.values())), 'differences': values,
                                        'improved_seeds': sum(x<0 for x in values.values())}
    selected = {v: {key: _summary([flat(r['validation_selected'])[key] for r in group])
                    for key in flat(source)} for v,group in grouped.items() if group}
    summary = {'completed_runs':len(records), 'expected_runs':len(experiment['runs']),
               'specification':spec, 'source':flat(source), 'aggregate':aggregate,
               'paired_differences_vs_baseline':paired, 'validation_selected':selected,
               'individual':individual,
               'backbone_unchanged_all':all(r['training']['backbone_unchanged'] for r in records.values()),
               'max_abs_T_change':max((r['frozen_prediction_max_abs_difference']['T'] for r in records.values()),default=0.),
               'training_seconds':sum(r['training']['training_seconds'] for r in records.values()),
               'evaluation_scope':'Conditional W at identical ORIGINAL train/validation/test/scene reference hits.',
               'uncertainty':'Sample standard deviation across three optimization seeds; one data expansion realization.'}
    write_json(root/'summary.json',summary)
    images=plots(root,spec,source,records)
    data=read_json(root/'datasets.json')
    content=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<title>First-hit：增加光线覆盖与每条光线采样对照</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;color:#1e293b;max-width:1380px;margin:30px auto;padding:0 24px}'
        'table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        'th{background:#f1f5f9}img{width:100%;max-width:1200px}.scroll{overflow:auto}</style>',
        '<h1>增加独立光线历史，还是增加同一光线上的 GP 实现？</h1>',
        f'<p>完成 {len(records)}/{len(experiment["runs"])} 组训练。h16 RNO、8 分量梯度头和共同起点固定，'
        '仅训练梯度头。相机和 VDB 为原 σ=0.05 场景；SE 核、介质入口计距和无起点条件化的首次零点定义一致。</p>',
        '<p>覆盖组保留原 120 条光线及其标签，新增 360 条随机像素光线，各 1024 次 GP 实现；'
        '采样组保留原每条 1024 次实现，再各加 3072 次独立实现。两组相机样本总数均为 491520，'
        '32 条仿射曲线及其原标签不变。新增光线排除所有原划分像素，选择不依赖模型误差或 GP 标签。</p>',
        f'<p>每组 {spec["training"]["updates"]} 次更新，批量 {spec["training"]["batch_size"]}，'
        f'其中每批固定 {experiment["camera_per_batch"]} 个相机首碰样本和 {experiment["affine_per_batch"]} 个仿射首碰样本。'
        '两类分别有放回采样，仿射随机流在相同种子、相同更新之间完全配对。'
        '原数据基线使用相同采样器重新训练，避免与上一轮不同采样方法混比。</p>',
        '<p>主要结果固定使用最后一次更新；辅助结果只按原验证集 NLL 选择。三个种子共享同一次数据扩展，'
        '均值±标准差只表示优化随机性，不表示跨场景或新数据抽样的不确定性。</p>',
        '<p><a href="summary.json">完整汇总</a> · <a href="datasets.json">数据规模</a> · '
        '<a href="experiment.json">采集定义与哈希</a> · <a href="training_experiment.json">训练配比</a> · '
        '<a href="ray_selection.json">新增光线选择</a></p>',
        f'<p>冻结参数完全一致：{summary["backbone_unchanged_all"]}；场景 8 条光线各 129 个位置的最大透射率变化：'
        f'{summary["max_abs_T_change"]:.3g}。</p>',
        conclusions(summary),
        '<h2>共同评估集上的梯度分布</h2><p>所有模型使用相同的原训练、验证、测试标签评估。'
        '训练列也是共同的原训练集；不同扩展数据自身的训练损失仅保存在逐组 JSON，不能直接作为跨数据分布的误差比较。'
        'NLL 与 PIT KS 越小越好；这些指标以参考首碰位置和入口符号为条件。</p>']
    columns=[('train_speed_nll_on_hits','共同原训练 NLL'),('validation_speed_nll_on_hits','验证 NLL'),
             ('test_speed_nll_on_hits','测试 NLL'),('test_speed_pit_ks','测试 PIT KS'),
             ('test_mean_profile_pit_ks_min64','逐光线平均 PIT KS（≥64 首碰）')]
    test_profiles=source['splits']['test']['profiles']
    camera_count=sum(p['pixel'] is not None for p in test_profiles)
    content.append(f'<p>完整原测试集包含 {camera_count} 条相机光线和 {len(test_profiles)-camera_count} 条仿射剖面。'
                   '整体 PIT 会把不同条件下的样本合并，局部误差可能相互抵消，因此须同时看逐光线指标和下面的相机子集。</p>')
    table(content,columns,source,aggregate)
    content.append('<h2>原测试集中的相机光线与仿射剖面</h2><p>新增数据仅来自相机光线。'
                   '下面按数据类别分别统计同一批测试标签；逐光线 PIT 对至少有 64 个首碰标签的光线等权平均。</p>')
    table(content,[('test_camera_speed_nll_on_hits','相机 NLL'),
                   ('test_camera_mean_profile_pit_ks_min64','相机逐光线平均 PIT KS'),
                   ('test_affine_speed_nll_on_hits','仿射 NLL'),
                   ('test_affine_mean_profile_pit_ks_min64','仿射逐剖面平均 PIT KS')],source,aggregate)
    content.append('<h2>σ=0.05 场景保留光线</h2><p>全部 8 条包含原训练 4 条、验证 1 条和测试 3 条。'
                   '保留测试列仅统计原测试 3 条，其中零首碰光线不提供梯度标签。</p>')
    table(content,[('scene_speed_nll_on_hits','全部 8 条 NLL'),('scene_speed_pit_ks','全部 8 条 PIT KS'),
                   ('scene_heldout_speed_nll_on_hits','保留测试 NLL'),('scene_heldout_speed_pit_ks','保留测试 PIT KS')],
          source,aggregate)
    content.append('<h2>相同种子配对：相对原数据基线</h2><p>差值为扩展数据减去原数据，负值表示改善。</p>'
                   '<table><tr><th>数据扩展</th><th>测试 NLL 差值</th><th>改善种子</th>'
                   '<th>测试 PIT KS 差值</th><th>改善种子</th><th>保留场景 NLL 差值</th></tr>')
    for variant,row in paired.items():
        a,b,c=(row[k] for k in ('test_speed_nll_on_hits','test_speed_pit_ks','scene_heldout_speed_nll_on_hits'))
        content.append(f'<tr><td>{LABELS[variant]}</td><td>{a["mean"]:+.6f} ± {a["standard_deviation"]:.6f}</td>'
                       f'<td>{a["improved_seeds"]}/{a["count"]}</td><td>{b["mean"]:+.6f} ± {b["standard_deviation"]:.6f}</td>'
                       f'<td>{b["improved_seeds"]}/{b["count"]}</td><td>{c["mean"]:+.6f}</td></tr>')
    content.append('</table><h2>按验证集 NLL 选择权重的辅助结果</h2>')
    table(content,[('test_speed_nll_on_hits','测试 NLL'),('test_speed_pit_ks','测试 PIT KS'),
                   ('scene_heldout_speed_nll_on_hits','保留场景 NLL')],source,selected)
    content.append('<h2>数据采集规模</h2><table><tr><th>组别</th><th>相机光线</th><th>每条实现数</th>'
                   '<th>相机首碰标签</th><th>仿射样本</th><th>验证／测试保持原样</th></tr>')
    for variant,row in data.items():
        content.append(f'<tr><td>{LABELS[variant]}</td><td>{row["camera_rays"]}</td>'
                       f'<td>{html.escape(str(row["camera_realizations"]))}</td><td>{row["camera_hits"]}</td>'
                       f'<td>{row["affine_samples"]}</td><td>{row["heldout_unchanged"]}</td></tr>')
    content.append('</table><h2>校准与学习曲线</h2>')
    for filename,label in images:
        content.append(f'<p>{label}</p><img src="plots/{filename}" alt="{label}">')
    content.append('<h2>逐组结果</h2><table><tr><th>模型</th><th>更新次数</th><th>训练样本使用次数</th>'
                   '<th>验证选择的更新</th><th>冻结参数一致</th></tr>')
    for name,r in records.items():
        t=r['training']
        content.append(f'<tr><td><a href="evaluations/{name}.json">{name}</a></td>'
                       f'<td>{t["updates"]}</td><td>{t["examples_seen"]}</td>'
                       f'<td>{r["validation_selected"]["update"]}</td><td>{t["backbone_unchanged"]}</td></tr>')
    content.append('</table>')
    for name,label in [('reference_audit.json','新标签的粗细网格对照'),('sample_independence.json','GP 路径随机种子独立性检查'),
                       ('training_rng_validation.json','训练随机流检查'),('data_support.json','相机有效首碰标签覆盖'),
                       ('validation.json','完整实验校验'),
                       ('implementation_corrections.json','训练前检查与采样器修正记录'),
                       ('cpp_verification.json','C++ 导出与采样检查')]:
        if (root/name).exists():
            content.append(f'<p><a href="{name}">{label}</a></p>')
    content.append('<p>本轮只改变训练数据，状态表示保持冻结。即使没有改善，也不能据此排除'
                   '新数据配合重新学习状态演化的作用。横向梯度保持解析 SE 条件分布，'
                   '此处未评估模型自采样碰撞位置后的完整法线分布。</p></html>')
    (root/'comparison.html').write_text('\n'.join(content),encoding='utf-8')
    return {'completed_runs':len(records),'report':str(root/'comparison.html'),'aggregate':aggregate}


def conclusions(summary):
    if summary['completed_runs'] != summary['expected_runs']:
        return ''
    rows=summary['aggregate']
    base,coverage,samples=(rows[v] for v in ('baseline','coverage','samples'))
    mean=lambda row,key:row[key]['mean']
    source=summary['source']
    text=['<h2>本轮结果解读</h2>',
          '<p>两个扩充方向都降低了同预算重训基线的测试 NLL，但校准收益依赖评估范围。'
          '下面同时保留共同起点、重训基线和相机子集，避免把改善单个汇总指标解释为全面改善。</p>',
          f'<p>整体测试 PIT KS：基线 {mean(base,"test_speed_pit_ks"):.6f}，'
          f'增加光线 {mean(coverage,"test_speed_pit_ks"):.6f}，'
          f'增加每条采样 {mean(samples,"test_speed_pit_ks"):.6f}。'
          '该测试集同时包含相机和仿射剖面。</p>',
          f'<p>相机测试 NLL：基线 {mean(base,"test_camera_speed_nll_on_hits"):.6f}，'
          f'增加光线 {mean(coverage,"test_camera_speed_nll_on_hits"):.6f}，'
          f'增加每条采样 {mean(samples,"test_camera_speed_nll_on_hits"):.6f}。'
          f'相机逐光线 PIT KS 分别为 {mean(base,"test_camera_mean_profile_pit_ks_min64"):.6f}、'
          f'{mean(coverage,"test_camera_mean_profile_pit_ks_min64"):.6f}、'
          f'{mean(samples,"test_camera_mean_profile_pit_ks_min64"):.6f}。</p>',
          f'<p>σ=0.05 场景原测试光线的 PIT KS 分别为 {mean(base,"scene_heldout_speed_pit_ks"):.6f}、'
          f'{mean(coverage,"scene_heldout_speed_pit_ks"):.6f}、'
          f'{mean(samples,"scene_heldout_speed_pit_ks"):.6f}，与整体测试的排序不同。'
          '该组只有两条光线有梯度标签，不能替代完整相机测试集。</p>']
    if all(mean(row,'test_speed_nll_on_hits')>source['test_speed_nll_on_hits'] and
           mean(row,'test_camera_mean_profile_pit_ks_min64')>source['test_camera_mean_profile_pit_ks_min64']
           for row in (coverage,samples)):
        text.append(f'<p><strong>加数据减轻了梯度头单独继续训练的部分退化，但没有全面超过共同起点。</strong>'
                    f'共同起点的整体测试 NLL 为 {source["test_speed_nll_on_hits"]:.6f}，'
                    f'相机逐光线 PIT KS 为 {source["test_camera_mean_profile_pit_ks_min64"]:.6f}，均仍优于两组扩充模型。'
                    '本轮保持状态表示冻结，尚不能把剩余误差归因于某一个因素。</p>')
    return ''.join(text)


def table(content,columns,source,aggregate):
    content.append('<div class="scroll"><table><tr><th>模型</th>'+''.join(f'<th>{label}</th>' for _,label in columns)+'</tr>')
    base=flat(source)
    content.append('<tr><td>共同起点（未单独微调）</td>'+''.join(f'<td>{base[key]:.6f}</td>' for key,_ in columns)+'</tr>')
    for variant,row in aggregate.items():
        content.append(f'<tr><td>{LABELS[variant]}</td>'+''.join(
            f'<td>{row[key]["mean"]:.6f} ± {row[key]["standard_deviation"]:.6f}</td>' for key,_ in columns)+'</tr>')
    content.append('</table></div>')


def plots(root,spec,source,records):
    dependency=resolve('outputs/first_hit_se_ablation_20261013/plot_dependencies')
    if dependency.exists():
        sys.path.insert(0,str(dependency))
    os.environ.setdefault('MPLCONFIGDIR',str(root/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    directory=root/'plots'
    directory.mkdir(exist_ok=True)
    colors={'source':'black','baseline':'#1f77b4','coverage':'#ff7f0e','samples':'#2ca02c'}
    chosen={'source':source}
    chosen.update({v:records[name] for v in LABELS
                   if (name:=f'{v}_s{spec["representative_seed"]}') in records})
    images=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for label,r in chosen.items():
        for axis,split in zip(axes,('test','scene')):
            row=r['splits'][split]
            x=np.asarray(row['pit_grid'])
            axis.plot(x,np.asarray(row['pit_ecdf'])-x,label=label,color=colors[label])
            axis.set(title=split,xlabel='PIT',ylabel='Empirical CDF - uniform CDF')
    for axis in axes:
        axis.axhline(0,color='grey',linewidth=.7);axis.legend();axis.grid(alpha=.2)
    fig.savefig(directory/'pit.svg');fig.savefig(directory/'pit.png',dpi=140);plt.close(fig)
    images.append(('pit.svg','原测试集与场景全部光线的 PIT 校准，代表种子 20261032'))
    fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True)
    for variant in LABELS:
        histories=[]
        for seed in spec['seeds']:
            path=resolve(spec['model_directory'])/f'{variant}_s{seed}'/'history.json'
            if path.exists():
                histories.append(read_json(path))
        if not histories:
            continue
        length=min(map(len,histories))
        x=[r['update'] for r in histories[0][:length]]
        y=np.array([[r['validation']['speed_nll_on_hits'] for r in h[:length]] for h in histories])
        mean=y.mean(0)
        axis.plot(x,mean,label=variant,color=colors[variant])
        if len(y)>1:
            sd=y.std(0,ddof=1)
            axis.fill_between(x,mean-sd,mean+sd,alpha=.15,color=colors[variant])
    axis.axhline(source['splits']['validation']['speed_nll_on_hits'],color='black',linestyle='--',label='source')
    axis.set(xlabel='optimizer updates',ylabel='Common validation speed NLL')
    axis.grid(alpha=.2);axis.legend()
    fig.savefig(directory/'validation.svg');fig.savefig(directory/'validation.png',dpi=140);plt.close(fig)
    images.append(('validation.svg','相同验证集的学习曲线；阴影表示优化种子间标准差'))
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for i,p in enumerate(source['splits']['scene']['profiles']):
        axis=axes.flat[i]
        for label,r in chosen.items():
            row=r['splits']['scene']['profiles'][i]
            if row['hits']:
                x=np.asarray(r['splits']['scene']['pit_grid'])
                axis.plot(x,np.asarray(row['pit_ecdf'])-x,label=label,color=colors[label])
        axis.set(title=f'{p["pixel"]}, {p["original_split"]}, n={p["hits"]}',xlabel='PIT',ylabel='CDF error')
        axis.axhline(0,color='grey',linewidth=.7);axis.grid(alpha=.2)
    axes.flat[-1].legend()
    fig.savefig(directory/'scene_pit.svg');plt.close(fig)
    images.append(('scene_pit.svg','逐光线校准，保留原训练／验证／测试归属'))
    return images

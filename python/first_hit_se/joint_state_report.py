"""Evaluate joint-state adaptation, including gradient calibration and survival."""
import html
import os
from pathlib import Path
import statistics
import sys
import time

import torch

from renewal.collect import digest,resolve,write_json
from renewal.model import mixture_cdf
from .ablation_report import _curves,_curve_summary
from .batch import make_plan
from .data import load_data,read_json
from .data_ablation import pixel_of,profile_key
from .data_ablation_report import flat as gradient_flat
from .gradient_head import cache_profiles,evaluate_head,backbone_digest
from .train import load_model,metrics


@torch.no_grad()
def evaluate(config_path,watch=False):
    spec=read_json(config_path);root=resolve(spec['output_directory'])
    experiment=read_json(root/'experiment.json')
    torch.set_num_threads(1)
    base=load_data(resolve(spec['evaluation_dataset']))['profiles']
    groups={key:[p for p in base if p['split']==key] for key in ('train','validation','test')}
    groups['scene']=load_data(resolve(spec['scene_dataset']))['profiles']
    mapping={pixel_of(p):p['split'] for p in base if pixel_of(p)}
    plans={key:[make_plan(p['features'],experiment['source_model_config']['maximum_step']) for p in ps]
           for key,ps in groups.items()}
    cached=torch.load(resolve(spec['evaluation_contexts']),map_location='cpu',weights_only=True)
    source,_=load_model(resolve(spec['initialize_from']))
    output=root/'evaluations';output.mkdir(exist_ok=True)

    def one(checkpoint,frozen,source_result=None):
        model,bundle=load_model(checkpoint)
        if frozen and backbone_digest(model)!=backbone_digest(source):
            raise AssertionError('a frozen evaluation tried to reuse a changed state')
        result={'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'epoch':bundle['epoch'],
                'splits':{},'curve_splits':{},'likelihood':{}}
        for key,profiles in groups.items():
            context=cached['splits'][key] if frozen else cache_profiles(model,profiles,mapping)
            result['splits'][key]=evaluate_head(model,context)
            if frozen and source_result:
                result['curve_splits'][key]=source_result['curve_splits'][key]
                initial=source_result['likelihood'][key]
                row={**initial,**{k:result['splits'][key][k] for k in ('speed_nll_on_hits','speed_pit_ks')}}
                delta=(row['speed_nll_on_hits']-initial['speed_nll_on_hits'])*initial['hits']/initial['samples']
                for name in ('joint_nll','complete_joint_nll'):row[name]=initial[name]+delta
                result['likelihood'][key]=row
            else:
                curves=_curves(model,profiles,plans[key],mapping)
                record={'profiles':curves,'summary':_curve_summary(curves)}
                camera=[p for p in curves if p['pixel'] is not None]
                if camera:record['camera_summary']=_curve_summary(camera)
                if key=='scene':
                    record['by_original_split']={s:_curve_summary([p for p in curves if p['original_split']==s])
                                                 for s in ('train','validation','test')}
                result['curve_splits'][key]=record
                result['likelihood'][key]=metrics(model,profiles,batch_size=16,plans=plans[key])
            print(f'EVALUATED {Path(checkpoint).parent.name}/{Path(checkpoint).name} {key}',flush=True)
        return result

    if not (output/'source.json').exists():
        value=one(resolve(spec['initialize_from']),True)
        write_json(output/'source.json',{'name':'source',**value})
    source_result=read_json(output/'source.json')
    if source_result['checkpoint_sha256']!=experiment['sha256']['initialize_from']:
        raise ValueError('source evaluation checkpoint changed')
    while True:
        completed=0
        for run in experiment['runs']:
            directory=resolve(spec['model_directory'])/run['name'];path=output/f'{run["name"]}.json'
            if path.exists():
                if read_json(path)['checkpoint_sha256']!=digest(directory/'last.pt'):
                    raise ValueError('evaluated checkpoint changed')
                completed+=1;continue
            if not (directory/'run_report.json').exists():continue
            trained=read_json(directory/'run_report.json')
            if trained['optimizer_updates']!=experiment['updates_per_run']:continue
            print('EVALUATE',run['name'],flush=True);began=time.perf_counter()
            primary=one(directory/'last.pt',run['mode']=='head_only',source_result)
            best=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
            secondary=({**primary,'checkpoint':str(directory/'best.pt'),'checkpoint_sha256':digest(directory/'best.pt')}
                       if best['epoch']==primary['epoch'] else one(directory/'best.pt',run['mode']=='head_only',source_result))
            value={**run,**primary,'training':trained,'validation_selected':secondary,
                   'evaluation_seconds':time.perf_counter()-began}
            temporary=path.with_suffix('.pending.json');write_json(temporary,value);temporary.replace(path)
            completed+=1
        if completed==len(experiment['runs']) or not watch:break
        time.sleep(20)
    return {'completed':completed,'expected':len(experiment['runs'])}


def flat(record):
    result=gradient_flat(record)
    for key in ('train','validation','test'):
        result[f'{key}_mean_max_T_error']=record['curve_splits'][key]['summary']['mean_max_abs_T']
        result[f'{key}_distance_nll']=record['likelihood'][key]['distance_nll']
    result['test_camera_mean_max_T_error']=record['curve_splits']['test']['camera_summary']['mean_max_abs_T']
    result['scene_heldout_mean_max_T_error']=record['curve_splits']['scene']['by_original_split']['test']['mean_max_abs_T']
    return result


def conclusions(summary):
    if summary['completed_runs']!=summary['expected_runs']:
        return '<p>实验尚未全部完成；当前均值的种子数见表格，最终结论待完整配对结果。</p>'
    frozen=summary['aggregate']['head_only'];joint=summary['aggregate']['joint']
    pairs=summary['paired_joint_minus_frozen']
    t='test_camera_mean_max_T_error';n='test_camera_speed_nll_on_hits';p='test_camera_mean_profile_pit_ks_min64'
    change=100*(joint[t]['mean']/frozen[t]['mean']-1)
    text=(f'<h2>主要结果</h2><p>相机测试光线的平均最大 T 误差从 {frozen[t]["mean"]:.6f} '
          f'变为 {joint[t]["mean"]:.6f}（相对变化 {change:+.1f}%，'
          f'{pairs[t]["improved_seeds"]}/{pairs[t]["count"]} 个种子改善）。'
          f'梯度 NLL 从 {frozen[n]["mean"]:.6f} 变为 {joint[n]["mean"]:.6f}，'
          f'差值 {pairs[n]["mean"]:+.6f}；逐光线 PIT KS 从 {frozen[p]["mean"]:.6f} '
          f'变为 {joint[p]["mean"]:.6f}，{pairs[p]["improved_seeds"]}/{pairs[p]["count"]} 个种子改善。</p>')
    if change<0 and pairs[p]['mean']>=0:
        text+='<p><strong>固定 h16 的联合微调改善了透射率，条件梯度分布校准没有改善。</strong>'
        if pairs[n]['mean']<0:
            text+='梯度 NLL 的下降与 PIT 校准的变化方向不同，不能据此认为梯度分布已整体改善。'
        text+='</p>'
    return text


@torch.no_grad()
def audit(config_path):
    from renewal.profiles import subdivide_features
    from .verify_cpp import verify_cpp
    spec=read_json(config_path);root=resolve(spec['output_directory'])
    experiment=read_json(root/'experiment.json');checks={};pairs={};evaluation_consistency={}
    assert digest(config_path)==experiment['specification_sha256']
    for key,value in experiment['sha256'].items():
        if digest(resolve(spec[key]))!=value:raise AssertionError(f'changed experiment input: {key}')
    for path,value in experiment['implementation_sha256'].items():
        if digest(resolve(path))!=value:raise AssertionError(f'changed implementation: {path}')
    source,initial=load_model(resolve(spec['initialize_from']))
    base=load_data(resolve(spec['evaluation_dataset']))['profiles']
    expanded=load_data(resolve(spec['dataset']))['profiles']
    original_heldout=[p for p in base if p['split']!='train']
    expanded_heldout=[p for p in expanded if p['split']!='train']
    assert len(original_heldout)==len(expanded_heldout)
    for p,q in zip(original_heldout,expanded_heldout):
        assert profile_key(p)==profile_key(q)
        for key in ('features','distance','hit','initial_sign','speed'):
            torch.testing.assert_close(p[key],q[key],atol=0,rtol=0,equal_nan=True)
    for seed in spec['seeds']:
        records=[]
        for mode in spec['modes']:
            directory=resolve(spec['model_directory'])/f'{mode}_s{seed}'
            record=read_json(directory/'run_report.json');records.append(record)
            assert record['optimizer_updates']==experiment['updates_per_run']
            model,bundle=load_model(directory/'last.pt')
            assert model.configuration()==source.configuration()
            assert record['checkpoint_sha256']==digest(directory/'last.pt')
            evaluated=read_json(root/'evaluations'/f'{mode}_s{seed}.json')
            assert evaluated['checkpoint_sha256']==digest(directory/'last.pt')
            assert evaluated['validation_selected']['checkpoint_sha256']==digest(directory/'best.pt')
            nll_delta=max(abs(evaluated['splits'][s]['speed_nll_on_hits']-evaluated['likelihood'][s]['speed_nll_on_hits'])
                          for s in ('train','validation','test','scene'))
            validation_delta=abs(evaluated['likelihood']['validation']['joint_nll']-bundle['validation']['joint_nll'])
            assert nll_delta<1e-5 and validation_delta<1e-5
            evaluation_consistency[f'{mode}_s{seed}']={
                'cached_vs_full_speed_nll':nll_delta,'cpu_vs_training_validation_joint_nll':validation_delta}
            for block,delta in record['parameter_max_changes'].items():
                actual=max(float((value-initial['model_state'][name]).abs().max())
                           for name,value in model.state_dict().items() if name.startswith(block+'.'))
                assert abs(actual-delta)<1e-12
                if mode=='joint' or block=='mixture_head':assert actual>0
                else:assert actual==0
        for key in ('sample_trace','paths_seen','hits_seen','optimizer_updates'):
            assert len({r[key] for r in records})==1,f'paired training differs: {key}'
        pairs[str(seed)]={k:records[0][k] for k in ('sample_trace','paths_seen','hits_seen','optimizer_updates')}
    assert len({p['sample_trace'] for p in pairs.values()})==len(pairs)
    probes=[p for p in base if pixel_of(p) in ((144,368),(368,368),(376,96),(296,336))]
    assert len(probes)==4
    for mode in spec['modes']:
        name=f'{mode}_s{spec["representative_seed"]}';checkpoint=resolve(spec['model_directory'])/name/'last.pt'
        path=root/'checks'/name/'audit.json'
        if path.exists():
            value=read_json(path);assert value['checkpoint_sha256']==digest(checkpoint)
        else:
            torch.set_num_threads(1);model,_=load_model(checkpoint);rows=[]
            for p in probes:
                f=p['features'];q=torch.linspace(0,p['horizon'],65,dtype=torch.float64)
                original=model.query(f,q)
                divided=model.query(subdivide_features(f,torch.full((len(f),),8,dtype=torch.long)),q)
                fine=model.query(f,q,maximum_step=model.config.maximum_step/2)
                def cdf(prediction):
                    mix=[prediction[key] for key in ('log_weights','means','scales')]
                    return torch.stack([mixture_cdf(*mix,torch.full_like(prediction['H'],w))
                                        for w in (.1,.25,.5,1.,2.,3.,4.)])
                c0,cd,cf=map(cdf,(original,divided,fine))
                rows.append({'pixel':pixel_of(p),'subdivision_max_abs_T':float((divided['T']-original['T']).abs().max()),
                             'subdivision_max_abs_log_T':float((divided['log_T']-original['log_T']).abs().max()),
                             'half_step_max_abs_T':float((fine['T']-original['T']).abs().max()),
                             'subdivision_max_abs_speed_cdf':float((cd-c0).abs().max()),
                             'half_step_max_abs_speed_cdf':float((cf-c0).abs().max())})
            cpp=verify_cpp(checkpoint,resolve(spec['evaluation_dataset']),resolve('build/Release/macrofacet_query_first_hit_se.exe'),
                           root/'checks'/name/'cpp',3,512)
            value={'checkpoint_sha256':digest(checkpoint),'profiles':rows,'cpp':cpp,
                   'maximum_cpp_difference':max(max(p['maximum_errors'].values()) for p in cpp['profiles'])}
            write_json(path,value)
        checks[name]=value
    result={'passed':True,'heldout_labels_unchanged':True,'heldout_profiles':len(original_heldout),
            'input_hashes_unchanged':True,'paired_batches':pairs,'models':checks,
            'evaluation_consistency':evaluation_consistency}
    write_json(root/'audit.json',result)
    return result


def report(config_path):
    from .gradient_head_report import _summary
    spec=read_json(config_path);root=resolve(spec['output_directory']);experiment=read_json(root/'experiment.json')
    source=read_json(root/'evaluations/source.json');records={}
    for run in experiment['runs']:
        path=root/'evaluations'/f'{run["name"]}.json'
        if path.exists():records[run['name']]=read_json(path)
    grouped={mode:[r for r in records.values() if r['mode']==mode] for mode in spec['modes']}
    aggregates={mode:{key:_summary([flat(r)[key] for r in rows]) for key in flat(source)}
                for mode,rows in grouped.items() if rows}
    selected={mode:{key:_summary([flat(r['validation_selected'])[key] for r in rows]) for key in flat(source)}
              for mode,rows in grouped.items() if rows}
    differences={}
    for key in flat(source):
        values={str(seed):flat(records[f'joint_s{seed}'])[key]-flat(records[f'head_only_s{seed}'])[key]
                for seed in spec['seeds'] if f'joint_s{seed}' in records and f'head_only_s{seed}' in records}
        if values:differences[key]={**_summary(list(values.values())),'differences':values,'improved_seeds':sum(v<0 for v in values.values())}
    summary={'completed_runs':len(records),'expected_runs':len(experiment['runs']),'specification':spec,
             'source':flat(source),'aggregate':aggregates,'validation_selected':selected,'paired_joint_minus_frozen':differences,
             'individual':{name:flat(r) for name,r in records.items()},
             'scope':'Same data, initialization, trajectory batches, joint likelihood and updates; only trainable modules differ.',
             'uncertainty':'Three optimization seeds; one fixed scene and set of GP realizations. Gradient metrics condition on reference hits.'}
    write_json(root/'summary.json',summary)
    images=plots(root,spec,source,records)
    labels={'head_only':'h16 冻结主干','joint':'h16 联合训练'}
    def table(columns,rows=aggregates):
        output='<table><tr><th>模型</th>'+''.join(f'<th>{title}</th>' for _,title in columns)+'</tr>'
        initial=flat(source)
        output+='<tr><td>共同起点</td>'+''.join(f'<td>{initial[key]:.6f}</td>' for key,_ in columns)+'</tr>'
        for mode,values in rows.items():
            count=next(iter(values.values()))['count']
            output+=f'<tr><td>{labels[mode]}（{count} 个种子）</td>'+''.join(f'<td>{values[key]["mean"]:.6f} ± {values[key]["standard_deviation"]:.6f}</td>' for key,_ in columns)+'</tr>'
        return output+'</table>'
    content=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>h16 状态网络联合训练对照</title>',
        '<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1380px;margin:30px auto;padding:0 24px;color:#1e293b}'
        'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        'th{background:#f1f5f9}img{width:100%;max-width:1250px}</style>',
        '<h1>h16：冻结状态与联合训练状态网络、输出头</h1>',
        f'<p>完成 {len(records)}/{len(experiment["runs"])} 组。固定 K=8、状态维数 16、SE 核、σ=0.05 场景和共同起点。'
        '使用上一轮 120 条相机光线 × 4096 次 GP 实现，以及原 32 条仿射剖面；没有新增数据。</p>',
        f'<p>每组 {spec["training"]["epochs"]} 轮、{experiment["updates_per_run"]} 次更新；每轮随机遍历所有训练剖面，'
        f'每条抽取 {spec["training"]["samples_per_profile"]} 个轨迹标签，每批最多 {spec["training"]["profiles_per_batch"]} 条剖面。'
        '同种子的两组使用逐项相同的批次、联合似然、学习率和更新预算。冻结组只更新梯度头；联合组更新四个网络。'
        '删失样本参与首达距离损失。主结果固定使用最后一轮，辅助结果只按原验证集联合 NLL 选择。</p>',
        '<p>训练使用 GP 实现的入口符号作为分支标签；相机推理仍按入口高斯概率混合正负分支，'
        '不要求预先知道入口符号。验证联合 NLL 对删失轨迹的入口分支边缘化，与原流程一致。</p>',
        '<p>本轮重跑冻结对照，因为联合训练按剖面抽取含删失标签的轨迹。上一轮 4608 次、每批 4096 个命中标签的结果'
        '采用不同训练采样和预算，应作为历史参考。三种子标准差不包括场景和 GP 标签抽样误差。</p>',
        '<p><a href="summary.json">完整统计</a> · <a href="experiment.json">实验定义及来源哈希</a> · '
        '<a href="../first_hit_se_data_ablation_20261013/comparison.html">上一轮数据量实验</a></p>',
        conclusions(summary),
        '<h2>梯度分布：原相机测试子集</h2><p>NLL 与 PIT KS 越低越好。逐光线 PIT 对至少 64 个首碰标签的光线等权平均。'
        '梯度评估固定参考首碰位置及入口符号；横向梯度继续采用解析 SE 条件分布。</p>',
        table([('test_camera_speed_nll_on_hits','相机 NLL'),('test_camera_mean_profile_pit_ks_min64','相机逐光线 PIT KS'),
               ('test_speed_nll_on_hits','完整测试 NLL'),('test_speed_pit_ks','完整测试 PIT KS')]),
        '<h2>同一批标签上的训练与验证误差</h2><p>训练列使用所有模型共同的原训练标签。'
        '它与学习曲线中的随机训练批次损失口径不同，可用于检查拟合与泛化是否同时改善。</p>',
        table([('train_speed_nll_on_hits','原训练梯度 NLL'),('validation_speed_nll_on_hits','验证梯度 NLL'),
               ('test_speed_nll_on_hits','测试梯度 NLL'),('test_mean_profile_pit_ks_min64','测试逐剖面 PIT KS')]),
        '<h2>透射率</h2><p>每条剖面在 129 个共同位置与 GP 参考曲线比较，表中是逐剖面最大绝对误差的均值。'
        '联合训练改变了隐藏状态和 hazard，必须检查透射率变化。完整测试包含 18 条相机光线与 8 条仿射剖面。</p>',
        table([('test_camera_mean_max_T_error','相机平均最大 T 误差'),('test_mean_max_T_error','完整测试平均最大 T 误差'),
               ('scene_heldout_mean_max_T_error','场景保留光线 T 误差'),('test_distance_nll','测试距离 NLL')]),
        '<h2>场景原测试三条光线</h2><p>其中两条有首碰梯度标签，零命中光线只参与透射率评估。</p>',
        table([('scene_heldout_speed_nll_on_hits','条件梯度 NLL'),('scene_heldout_speed_pit_ks','条件梯度 PIT KS')]),
        '<h2>按验证联合 NLL 选择的辅助结果</h2>',
        table([('test_camera_speed_nll_on_hits','相机 NLL'),('test_camera_mean_profile_pit_ks_min64','相机逐光线 PIT KS'),
               ('test_camera_mean_max_T_error','相机 T 误差')],selected),
        '<h2>逐种子配对差值：联合减冻结</h2><p>负值表示改善。</p><table><tr><th>指标</th><th>均值 ± 标准差</th><th>改善种子</th></tr>']
    for key in ('test_camera_speed_nll_on_hits','test_camera_mean_profile_pit_ks_min64','test_camera_mean_max_T_error','scene_heldout_speed_pit_ks'):
        if key in differences:
            row=differences[key];content.append(f'<tr><td>{key}</td><td>{row["mean"]:+.6f} ± {row["standard_deviation"]:.6f}</td><td>{row["improved_seeds"]}/{row["count"]}</td></tr>')
    content.append('</table><h2>保留场景光线的近零速度与尾部概率</h2>'
                   '<p>W 是按入口符号定向并以 ℓ/σ 归一化的沿射线梯度绝对值。预测 CDF 在相同参考首碰位置上平均，'
                   'GP 列为这些标签的经验频率。</p><table><tr><th>事件</th><th>GP</th><th>共同起点</th>'
                   +''.join(f'<th>{labels[mode]}</th>' for mode in aggregates)+'</tr>')
    reference=source['splits']['scene']['by_original_split']['test']['threshold_calibration']
    for i,value in enumerate(reference):
        tail=value['W']>=2
        def probability(x):return 1-x if tail else x
        content.append(f'<tr><td>W {">" if tail else "≤"} {value["W"]:g}</td>'
                       f'<td>{probability(value["observed_cdf"]):.6f}</td>'
                       f'<td>{probability(value["predicted_cdf"]):.6f}</td>')
        for mode in aggregates:
            values=[probability(r['splits']['scene']['by_original_split']['test']['threshold_calibration'][i]['predicted_cdf'])
                    for r in grouped[mode]]
            row=_summary(values)
            content.append(f'<td>{row["mean"]:.6f} ± {row["standard_deviation"]:.6f}</td>')
        content.append('</tr>')
    content.append('</table><h2>学习曲线与场景曲线</h2>')
    for filename,label in images:content.append(f'<p>{label}</p><img src="plots/{filename}" alt="{label}">')
    content.append('<h2>逐组记录</h2><ul>')
    for name,r in records.items():
        content.append(f'<li><a href="evaluations/{name}.json">{html.escape(name)}</a>：'
                       f'{r["training"]["optimizer_updates"]} 次更新，验证选择第 {r["validation_selected"]["epoch"]} 轮。</li>')
    content.append('</ul>')
    if (root/'audit.json').exists():
        checked=read_json(root/'audit.json')
        probes=[p for r in checked['models'].values() for p in r['profiles']]
        content.append('<h2>数值与配对检查</h2><p>三个种子的配对模型使用完全相同的轨迹索引序列；'
                       '冻结模型的三个主干模块逐项不变，联合模型的四个模块均有更新。'
                       '原验证／测试标签保持不变。下面数值检查使用预选代表种子和四条指定光线。</p>')
        content.append(f'<p>精确八倍细分的最大 T 差：{max(p["subdivision_max_abs_T"] for p in probes):.3g}；'
                       f'积分步长减半的最大 T 差：{max(p["half_step_max_abs_T"] for p in probes):.3g}；'
                       f'两个代表模型的 C++ 最大预测差：{max(r["maximum_cpp_difference"] for r in checked["models"].values()):.3g}。</p>')
        if all('subdivision_max_abs_speed_cdf' in p for p in probes):
            content.append(f'<p>条件梯度 CDF 在两种入口符号、每条 65 个位置和七个速度阈值上检查：'
                           f'细分最大差 {max(p["subdivision_max_abs_speed_cdf"] for p in probes):.3g}，'
                           f'步长减半最大差 {max(p["half_step_max_abs_speed_cdf"] for p in probes):.3g}。</p>')
        content.append('<p><a href="audit.json">完整配对与数值检查</a></p>')
    if (root/'validation.json').exists():
        content.append('<p><a href="validation.json">测试结果、报告引用和原 RNO／GRU 权重校验</a></p>')
    content.append('<p>本轮比较联合微调与仅训练梯度头。联合组同时解冻状态网络和 hazard，'
                   '不能将收益单独归因于状态更新，也不能证明或排除 16 维的容量上限。</p></html>')
    (root/'comparison.html').write_text('\n'.join(content),encoding='utf-8')
    return {'completed_runs':len(records),'report':str(root/'comparison.html')}


def plots(root,spec,source,records):
    sys.path.insert(0,str(resolve('outputs/first_hit_se_ablation_20261013/plot_dependencies')))
    os.environ.setdefault('MPLCONFIGDIR',str(root/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
    output=root/'plots';output.mkdir(exist_ok=True)
    colors={'source':'black','head_only':'#1f77b4','joint':'#ff7f0e'}
    chosen={'source':source}
    chosen.update({mode:records[name] for mode in spec['modes'] if (name:=f'{mode}_s{spec["representative_seed"]}') in records})
    images=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for mode in spec['modes']:
        histories=[read_json(resolve(spec['model_directory'])/f'{mode}_s{seed}'/'history.json')
                   for seed in spec['seeds'] if (resolve(spec['model_directory'])/f'{mode}_s{seed}'/'history.json').exists()]
        if not histories:continue
        n=min(map(len,histories));x=[r['updates'] for r in histories[0][:n]]
        for axis,key in zip(axes,('joint_nll','speed_nll_on_hits')):
            y=np.array([[r['validation'][key] for r in h[:n]] for h in histories]);mean=y.mean(0)
            axis.plot(x,mean,color=colors[mode],label=mode)
            if len(y)>1:axis.fill_between(x,mean-y.std(0,ddof=1),mean+y.std(0,ddof=1),color=colors[mode],alpha=.15)
            axis.set(xlabel='Optimizer updates',ylabel='Validation '+key);axis.grid(alpha=.2)
    for axis,key in zip(axes,('joint_nll','speed_nll_on_hits')):
        axis.axhline(source['likelihood']['validation'][key],color=colors['source'],linestyle='--',label='source')
        axis.legend()
    fig.savefig(output/'learning.svg');fig.savefig(output/'learning.png',dpi=140);plt.close(fig)
    images.append(('learning.svg','验证损失；阴影是三个优化种子的标准差'))
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for i,p in enumerate(source['curve_splits']['scene']['profiles']):
        axis=axes.flat[i];axis.plot(p['q'],p['GP'],color='#777',label='GP')
        for mode,r in chosen.items():
            row=r['curve_splits']['scene']['profiles'][i]
            axis.plot(row['q'],row['T'],color=colors[mode],label=mode,linestyle='--' if mode=='source' else '-')
        axis.set(title=f'{p["pixel"]}, {p["original_split"]}',xlabel='Distance from entry / ell',ylabel='T',ylim=(-.025,1.025))
        axis.grid(alpha=.2)
    axes.flat[-1].legend();fig.savefig(output/'scene_T.svg');fig.savefig(output/'scene_T.png',dpi=130);plt.close(fig)
    images.append(('scene_T.svg','原场景光线的透射率；预选代表种子 20261032'))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for mode,r in chosen.items():
        for axis,key in zip(axes,('test','scene')):
            row=r['splits'][key];x=np.asarray(row['pit_grid'])
            axis.plot(x,np.asarray(row['pit_ecdf'])-x,color=colors[mode],label=mode)
            axis.set(title=key,xlabel='PIT',ylabel='Empirical CDF - uniform CDF');axis.grid(alpha=.2)
    for axis in axes:axis.axhline(0,color='grey',linewidth=.7);axis.legend()
    fig.savefig(output/'pit.svg');fig.savefig(output/'pit.png',dpi=140);plt.close(fig)
    images.append(('pit.svg','完整测试集与场景全部光线的条件梯度 PIT，代表种子 20261032'))
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for i,p in enumerate(source['splits']['scene']['profiles']):
        axis=axes.flat[i]
        for mode,r in chosen.items():
            row=r['splits']['scene']['profiles'][i]
            if row['hits']:
                x=np.asarray(r['splits']['scene']['pit_grid'])
                axis.plot(x,np.asarray(row['pit_ecdf'])-x,color=colors[mode],label=mode)
        axis.set(title=f'{p["pixel"]}, {p["original_split"]}, n={p["hits"]}',xlabel='PIT',ylabel='CDF error')
        axis.axhline(0,color='grey',linewidth=.7);axis.grid(alpha=.2)
    axes.flat[-1].legend();fig.savefig(output/'scene_pit.svg');fig.savefig(output/'scene_pit.png',dpi=130);plt.close(fig)
    images.append(('scene_pit.svg','场景逐光线的梯度校准；无首碰标签的光线不参与梯度评估'))
    paired_seeds=[seed for seed in spec['seeds']
                  if all(f'{mode}_s{seed}' in records for mode in spec['modes'])]
    if paired_seeds:
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for axis,group,metric,title in zip(axes,('curve_splits','splits'),('max_abs_T','speed_pit_ks'),
                                          ('Maximum T error','Gradient PIT KS')):
            rows=[]
            for p in source[group]['test']['profiles']:
                if p['pixel'] is None or (group=='splits' and p['hits']<64):continue
                values=[]
                for mode in spec['modes']:
                    values.append(statistics.mean(
                        next(q[metric] for q in records[f'{mode}_s{seed}'][group]['test']['profiles'] if q['id']==p['id'])
                        for seed in paired_seeds))
                rows.append((p['pixel'],*values))
            x=np.array([r[1] for r in rows]);y=np.array([r[2] for r in rows])
            limit=1.1*max(x.max(),y.max(),.01)
            axis.plot([0,limit],[0,limit],color='grey',linestyle='--',linewidth=1)
            axis.scatter(x,y,c=['#009e73' if b<a else '#d55e00' for a,b in zip(x,y)],s=45)
            for pixel,a,b in sorted(rows,key=lambda row:max(row[1:]),reverse=True)[:2]:
                axis.annotate(str(pixel),(a,b),xytext=(4,5),textcoords='offset points',fontsize=9)
            axis.set(title=f'{title}: {len(rows)} camera rays',xlabel='Frozen, mean over paired seeds',
                     ylabel='Joint, mean over paired seeds',xlim=(0,limit),ylim=(0,limit))
            axis.set_aspect('equal',adjustable='box');axis.grid(alpha=.2)
        fig.savefig(output/'test_scatter.svg');fig.savefig(output/'test_scatter.png',dpi=140);plt.close(fig)
        images.append(('test_scatter.svg',f'相机测试光线逐条比较，{len(paired_seeds)} 个配对种子取均值；'
                       '对角线下方表示联合训练误差更小，PIT 仅统计至少 64 次首碰的光线'))
    return images

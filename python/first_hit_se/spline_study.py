"""Evaluate trained spline variants and time the complete CPU rendering path."""
import html
from pathlib import Path
import shutil
import statistics
import subprocess

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .data import read_json, load_data
from .train import load_model
from .spline import local_profile
from .camera_study import evaluate_curves, evaluate_gradients, plotting, save_figure, pixel_of
from .verify_cpp import verify_cpp


OUT = 'outputs/first_hit_spline_20261015'
DATA = 'outputs/first_hit_se_tungsten_camera_sigma005_20261014/data/dataset.pt'
BASE = 'outputs/first_hit_se_tungsten_camera_sigma005_20261014'
VARIANTS = ('p128_k32', 'p256_k32', 'p128_k64', 'p256_k64')


def evaluate():
    torch.set_num_threads(1)
    output = resolve(OUT)
    data = load_data(resolve(DATA))
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    runs = {name: read_json(output/name/'run_report.json') for name in VARIANTS}
    baseline = read_json(resolve('outputs/models/first_hit_se_tungsten_camera_sigma005_20261014_rno_h16/run_report.json'))
    if any(r['dataset_sha256'] != digest(resolve(DATA)) for r in [baseline, *runs.values()]):
        raise ValueError('comparison requires the identical dataset and split')
    # Freeze model selection before inspecting test curves or images.
    selected = min(VARIANTS, key=lambda name: runs[name]['metrics']['validation']['joint_nll'])
    write_json(output/'selection.json', {'selected': selected, 'criterion': 'minimum validation joint NLL',
                                        'validation_scores': {n: r['metrics']['validation']['joint_nll'] for n, r in runs.items()}})
    curves = {}
    for name in VARIANTS:
        model, _ = load_model(output/name/'best.pt')
        curves[name] = evaluate_curves(model, profiles)
        print(f'evaluated {name}: maximum T error {max(r["max_abs_T_error"] for r in curves[name]):.5f}', flush=True)
    model, _ = load_model(output/selected/'best.pt')
    spec = read_json(resolve('python/configs/first_hit_se_tungsten_camera_sigma005_20261014_study.json'))
    gradients = evaluate_gradients(model, profiles, spec, output)
    # Existing plotting utility names its prediction keys rno; expose correct
    # architecture names in these new artifacts.
    for records in curves.values():
        for r in records:
            r['spline'] = r.pop('rno')
            r['terminal_spline_T'] = r.pop('terminal_rno_T')
    for r in gradients:
        r['spline_cdf'] = r.pop('rno_cdf')
        r['spline_mean_gradient'] = r.pop('rno_mean_gradient')
        path = output/(r['name']+'.npz')
        with np.load(path) as source:
            arrays = {k.replace('rno_', 'spline_'): source[k] for k in source.files}
        np.savez_compressed(path, **arrays)
    audit = []
    for p in profiles:
        q = torch.linspace(0, p['horizon'], 4097, dtype=torch.float64)
        b, db = local_profile(p['features'], q)
        for points in (128, 256):
            fixed_q = torch.linspace(0, p['horizon'], points, dtype=torch.float64)
            fixed_b, fixed_db = local_profile(p['features'], fixed_q)
            residual = np.interp(q.numpy(), fixed_q.numpy(), fixed_b.numpy())-b.numpy()
            derivative_residual = np.interp(q.numpy(), fixed_q.numpy(), fixed_db.numpy())-db.numpy()
            audit.append({'pixel': list(pixel_of(p)), 'points': points, 'normalized_spacing': p['horizon']/(points-1),
                          'linear_reconstruction_b_max_error': float(np.max(np.abs(residual))),
                          'linear_reconstruction_b_rmse': float(np.sqrt(np.mean(residual**2))),
                          'linear_reconstruction_db_max_error': float(np.max(np.abs(derivative_residual)))})
    report = {'dataset': DATA, 'dataset_sha256': digest(resolve(DATA)), 'selected': selected,
              'selection_criterion': 'minimum validation joint NLL; test is not used for selection',
              'runs': runs, 'rno_training': baseline, 'curves': curves, 'gradients': gradients, 'fixed_grid_audit': audit,
              'rno_reference': read_json(resolve(BASE)/'analysis.json'),
              'maximum_T_errors': {n: max(r['max_abs_T_error'] for r in rs) for n, rs in curves.items()},
              'mean_ray_maximum_T_errors': {n: float(np.mean([r['max_abs_T_error'] for r in rs])) for n, rs in curves.items()}}
    write_json(output/'analysis.json', report)
    shutil.copy2(output/selected/'first_hit_model.json', output/'first_hit_model.json')
    config = read_json(resolve('configs/render_first_hit_rno_local_shader_ball_sigma005.json'))
    config['transport']['first_hit']['model'] = OUT+'/first_hit_model.json'
    config['output_directory'] = 'outputs/render_first_hit_spline_local_shader_ball_sigma005'
    write_json(resolve('configs/render_first_hit_spline_local_shader_ball_sigma005.json'), config)
    verification = verify_cpp(output/selected/'best.pt', resolve(DATA),
                              resolve('build/Release/macrofacet_query_first_hit_se.exe'), output/'cpp', limit=24, trials=512)
    write_json(output/'verification.json', verification)
    return {'selected': selected, 'maximum_T_errors': report['maximum_T_errors'],
            'maximum_gradient_CDF_error': max(r['cdf_max_error'] for r in gradients)}


def benchmark():
    output = resolve(OUT)
    report = read_json(output/'analysis.json')
    results = {n: [] for n in ('rno', 'spline', 'gpis')}
    records = {n: [] for n in results}
    for name in results:
        config = read_json(resolve('configs/render_first_hit_rno_local_shader_ball_sigma005.json'))
        if name == 'spline':
            config['transport']['first_hit']['model'] = OUT+'/first_hit_model.json'
        if name == 'gpis':
            config['transport']['first_hit']['method'] = 'gpis_function_space'
            config['transport']['first_hit'].pop('model')
        config['render'].update(width=64, height=64, samples_per_pixel=16, thread_count=0)
        config['output_directory'] = OUT+'/render_'+name
        write_json(output/(name+'_config.json'), config)
    # Rotate order; no training or other CPU benchmark should run concurrently.
    for repetition, order in enumerate((('rno','spline','gpis'), ('spline','gpis','rno'), ('gpis','rno','spline'))):
        for name in order:
            with (output/f'{name}_render_{repetition}.log').open('w') as log:
                subprocess.run([str(resolve('build/Release/macrofacet_experiments.exe')), 'render', '--config',
                                str(output/(name+'_config.json'))], cwd=resolve('.'), stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = read_json(output/('render_'+name)/'hybrid_summary.json')
            if summary['numerical_failures']:
                raise RuntimeError('render has numerical failures')
            results[name].append(summary['seconds'])
            records[name].append(summary)
            write_json(output/'benchmark_partial.json', results)
            print(f'{name} render {repetition+1}: {summary["seconds"]:.4f}s', flush=True)
    medians = {n: statistics.median(v) for n, v in results.items()}
    report = {'scene': 'sigma=0.05; identical camera, seed and classic_local continuation',
              'resolution': [64, 64], 'samples_per_pixel': 16, 'threads': 'auto',
              'timing': 'renderer wall time, including profile construction; excludes initial loading',
              'selected': report['selected'], 'seconds': results, 'median_seconds': medians, 'summaries': records,
              'spline_speedup_over_optimized_rno': medians['rno']/medians['spline'],
              'spline_time_over_gpis': medians['spline']/medians['gpis']}
    write_json(output/'benchmark.json', report)
    return report


def figures_and_report():
    output = resolve(OUT)
    a = read_json(output/'analysis.json')
    benchmark_path = output/'benchmark.json'
    timing = read_json(benchmark_path) if benchmark_path.exists() else None
    plt = plotting()
    spec = read_json(resolve('python/configs/first_hit_se_tungsten_camera_sigma005_20261014_study.json'))
    curves = {tuple(r['pixel']): r for r in a['curves'][a['selected']]}
    rno = {tuple(r['pixel']): r for r in a['rno_reference']['curves']}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for axis, pixel in zip(axes.flat, spec['visualization_pixels']):
        c, old = curves[tuple(pixel)], rno[tuple(pixel)]
        s = c['distance']
        axis.fill_between(s, c['reference_low'], c['reference_high'], color='#64748b', alpha=.2)
        axis.plot(s, c['reference'], color='#334155', label='GPIS samples')
        axis.plot(old['distance'], old['rno'], color='#2563eb', linestyle='--', label='RNO')
        axis.plot(s, c['spline'], color='#d94824', label='Spline '+a['selected'])
        axis.set(title=f'Pixel {tuple(pixel)} | max error {c["max_abs_T_error"]:.3f}', xlabel='Distance from medium entry', ylabel='T', ylim=(-.02, 1.02))
    axes.flat[0].legend(fontsize=8)
    save_figure(plt, fig, output, 'transmittance')
    old_gradients = {(tuple(r['pixel']), r['window']): r for r in a['rno_reference']['gradients']}
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for axis, g in zip(axes.flat, a['gradients']):
        old = old_gradients[tuple(g['pixel']), g['window']]
        axis.plot(g['W_grid'], g['reference_cdf'], color='#334155', label=f'GPIS ({g["reference_hits"]} hits)')
        axis.plot(old['W_grid'], old['rno_cdf'], color='#2563eb', linestyle='--', label='RNO')
        axis.plot(g['W_grid'], g['spline_cdf'], color='#d94824', label='Spline')
        axis.set(title=f'{tuple(g["pixel"])} window {g["window"]} | error {g["cdf_max_error"]:.3f}', xlabel='Longitudinal crossing speed W', ylabel='CDF', ylim=(0, 1))
        axis.legend(fontsize=8)
    save_figure(plt, fig, output, 'gradient_cdf')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in VARIANTS:
        history = read_json(output/name/'history.json')
        axes[0].plot([h['epoch'] for h in history], [h['validation']['distance_nll'] for h in history], label=name)
        axes[1].plot([h['epoch'] for h in history], [h['validation']['speed_nll_on_hits'] for h in history], label=name)
    for axis, title in zip(axes, ('Validation distance NLL', 'Validation W NLL on hits')):
        axis.set(xlabel='Epoch', ylabel=title); axis.legend()
    save_figure(plt, fig, output, 'training')
    rows = []
    for name, run in [('RNO', a['rno_training']), *a['runs'].items()]:
        m = run['metrics']
        max_error = a['rno_reference']['maximum_ray_max_T_error'] if name == 'RNO' else a['maximum_T_errors'][name]
        rows.append(f'<tr><td>{name}</td><td>{run["parameters"]:,}</td><td>{run["selected_epoch"]}</td>'
                    f'<td>{m["validation"]["joint_nll"]:.5f}</td><td>{m["test"]["distance_nll"]:.5f}</td>'
                    f'<td>{m["test"]["speed_nll_on_hits"]:.5f}</td><td>{max_error:.5f}</td></tr>')
    timing_html = '<p>渲染计时尚未运行。</p>'
    if timing:
        timing_html = '<table><tr><th>第一段策略</th><th>三次渲染秒数</th><th>中位数</th></tr>'+''.join(
            f'<tr><td>{n}</td><td>{", ".join(f"{v:.3f}" for v in timing["seconds"][n])}</td><td>{v:.3f}</td></tr>'
            for n, v in timing['median_seconds'].items())+'</table>'
        timing_html += f'<p>64×64、16 spp、相同种子、后续 local；包括完整剖面构建。Spline 相对优化后 RNO 提速 {timing["spline_speedup_over_optimized_rno"]:.2f} 倍；耗时为 GPIS 的 {timing["spline_time_over_gpis"]:.2f} 倍。</p>'
        timing_html += '<div class="renders">'+''.join(f'<figure><img src="render_{n}/render_hybrid_first_hit_directional.bmp"><figcaption>{n} · 64×64 / 16 spp</figcaption></figure>' for n in timing['seconds'])+'</div>'
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定剖面单调样条首碰实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;color:#182334;background:#fafbfc}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:9px;border-bottom:1px solid #dce2e9;text-align:left}}img{{max-width:100%}}.note{{padding:16px;background:#fff2df;border-left:4px solid #df9225}}a{{color:#1757a6}}.renders{{display:flex;gap:16px}}.renders figure{{flex:1}}.renders img{{width:100%;image-rendering:pixelated}}</style>
<h1>固定均值剖面 → 单调样条 CDF → 条件梯度头</h1>
<p>复用 sigma=0.05 的相机首碰数据：191 条训练射线、32 条验证射线、24 条测试射线；没有重新采集 GPIS。距离从介质入口起算。</p>
<p>128/256 个固定位置，32/64 个有理二次样条区间；宽度 64 的编码 MLP、32 维特征、8 分量截断高斯梯度头。先按解析先验抽入口符号，然后只编码该分支；以逃逸原子和条件 CDF 采样距离，仅在命中后评估梯度头。</p>
<p>模型选择：<b>{html.escape(a['selected'])}</b>，依据验证集 joint NLL。测试射线及下方图像不参与选择。</p>
<p class="note">这是速度与精度的实验对照。固定剖面不是固定的一条均值曲线，而是每条射线在相同数量的位置采样。当前 C++ 仍先构造 VDB 的完整三次剖面再重采样；碰撞点使用原始三次剖面的局部均值和导数。梯度标签仍是 Tungsten 线性过零的纵向割线斜率。</p>
<h2>分布精度</h2><p>NLL 越小越好；测试集最大 T 误差覆盖 24 条射线，每条 513 个位置。</p>
<table><tr><th>模型</th><th>参数量</th><th>选中 epoch</th><th>验证 joint NLL</th><th>测试距离 NLL</th><th>测试梯度 W NLL</th><th>最大 T 误差</th></tr>{''.join(rows)}</table>
<h2>实际 CPU 渲染速度</h2>{timing_html}
<h2>相机射线透射率</h2><p>灰色范围为经验透射率的逐点 95% Wilson 区间。</p><img src="transmittance.png">
<h2>早、中、晚碰撞的纵向梯度 CDF</h2><p>先用每条测试射线前半样本定位窗口，再用后半样本评估。窗口内按相同碰撞位置和入口符号条件平均模型 CDF。完整梯度样本存于 gradient_*.npz。</p><img src="gradient_cdf.png">
<h2>训练曲线</h2><img src="training.png">
<p><a href="analysis.json">完整精度指标</a> · <a href="benchmark.json">渲染计时</a> · <a href="verification.json">Python/C++ 对照</a> · <a href="selection.json">模型选择</a> · <a href="first_hit_model.json">导出模型</a></p>
<p>方法依据：<a href="https://arxiv.org/abs/1906.04032">Neural Spline Flows</a>。这是该有理二次样条公式的项目内实现，没有新增 nflows 运行依赖。</p></html>'''
    (output/'summary.html').write_text(content, encoding='utf-8')
    return {'summary': str(output/'summary.html'), 'selected': a['selected']}


def run(phase):
    if phase in ('all', 'evaluate'):
        evaluate()
    if phase in ('all', 'benchmark'):
        benchmark()
    return figures_and_report()

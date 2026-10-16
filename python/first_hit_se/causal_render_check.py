"""Reproducible scalar/CUDA render smoke test and a local HTML report."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from renewal.collect import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def values(rows, keys):
    return np.array([[float(row[k]) for k in keys] for row in rows])


def report(output):
    records, summaries = {}, {}
    for backend in ('scalar', 'torch_cuda'):
        directory = output/f'preview_{backend}'
        with (directory/'first_hit_samples.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        records[backend] = rows
        summary = read_json(directory/'hybrid_summary.json')
        resolved = read_json(directory/'resolved_config.json')
        w, g = values(rows, ['ray_dx', 'ray_dy', 'ray_dz']), values(rows, ['gx', 'gy', 'gz'])
        normal = values(rows, ['nx', 'ny', 'nz'])
        hit = values(rows, ['hit'])[:, 0].astype(bool)
        ell = 1/np.sqrt(np.einsum('ni,ij,nj->n', w, np.array(resolved['derived']['kernel_metric']), w))
        sigma = resolved['derived']['sigma']
        sign, speed = values(rows, ['initial_sign'])[:, 0], values(rows, ['speed'])[:, 0]
        longitudinal = (w*g).sum(-1)+sign*sigma/ell*speed
        position = values(rows, ['x', 'y', 'z'])
        entry = values(rows, ['entry_x', 'entry_y', 'entry_z'])
        q = values(rows, ['q'])[:, 0]
        summary['longitudinal_max_abs_error'] = float(np.abs(longitudinal[hit]).max(initial=0))
        summary['normal_length_max_abs_error'] = float(np.abs(np.linalg.norm(normal[hit], axis=1)-1).max(initial=0))
        summary['position_max_abs_error'] = float(np.abs(position[hit]-entry[hit]-q[hit, None]*ell[hit, None]*w[hit]).max(initial=0))
        if max(summary[k] for k in ('longitudinal_max_abs_error', 'normal_length_max_abs_error', 'position_max_abs_error')) > 1e-10:
            raise AssertionError('physical gradient/position reconstruction failed')
        if summary['numerical_failures']:
            raise AssertionError('render numerical failure')
        for keys in (['q', 'speed'], ['gx', 'gy', 'gz'], ['radiance_r', 'radiance_g', 'radiance_b']):
            if not np.isfinite(values(rows, keys)).all():
                raise AssertionError('nonfinite render sample')
        summaries[backend] = summary
        shutil.copyfile(directory/'render_hybrid_first_hit_directional.bmp', output/f'{backend}.bmp')
    a, b = records['scalar'], records['torch_cuda']
    if len(a) != len(b):
        raise AssertionError('render sample counts differ')
    mismatches = {k: sum(x[k] != y[k] for x, y in zip(a, b))
        for k in ('pixel_x', 'pixel_y', 'sample', 'entered', 'hit', 'initial_sign',
                  'camera_seed', 'event_seed', 'gradient_seed', 'continuation_seed')}
    if any(mismatches.values()):
        raise AssertionError(mismatches)
    errors = {k: float(np.abs(values(a, [k])-values(b, [k])).max())
        for k in ('q', 'distance', 'speed', 'x', 'y', 'z', 'gx', 'gy', 'gz', 'nx', 'ny', 'nz')}
    if max(errors.values()) > 5e-4:
        raise AssertionError(errors)
    config = read_json(output/'preview_scalar_config.json')
    spp = config['render']['samples_per_pixel']
    ra, rb = (values(rows, ['radiance_r', 'radiance_g', 'radiance_b']).reshape(-1, spp, 3).mean(1)
              for rows in (a, b))
    image_error = {'rmse': float(np.sqrt(np.mean((ra-rb)**2))),
        'max_abs': float(np.max(np.abs(ra-rb))), 'mean_rgb_difference': (rb-ra).mean(0).tolist()}
    parity = read_json(output/'parity'/'summary.json')
    result = {'sampling_and_reconstruction_passed': True, 'sample_count': len(a),
        'model_sha256': digest(Path(config['transport']['first_hit']['model'])),
        'event_mismatches': mismatches, 'first_hit_max_abs_errors': errors,
        'full_image_difference': image_error, 'render_summaries': summaries,
        'parity_summary': 'parity/summary.json',
        'limitations': ['Only first-hit neural inference is batched; continuation runs on CPU.',
            'Timing includes CUDA initialization and is not a warmed throughput benchmark.',
            'The speed label is a GPIS secant; transverse gradients use the conditional SE Gaussian.',
            'Float32 first-hit differences can change later stochastic paths; full images are not bit-identical.',
            'Nonzero safety-cap terminations are recorded and must not be treated as physical escape.']}
    write_json(output/'summary.json', result)
    max_error = lambda key: max(row[key] for row in parity['rows'])
    parity_rays = len({row['ray'] for row in parity['rows']})
    parity_samples = parity['rows'][0]['samples']
    performance_html = ''
    performance_path = output/'performance'/'summary.json'
    if performance_path.exists():
        performance = read_json(performance_path)
        warm = performance['runs_seconds']['cuda_baseline'][1:]
        startup = performance['cold_extra_seconds']['cuda_baseline']
        integration = performance['warm_host_stages_seconds']['cpu_local_integration_and_root']
        performance_html = f'<p>后续分阶段诊断：首帧额外开销约 {startup:.2f} s；同一进程预热后 CUDA 基线 {min(warm):.2f}–{max(warm):.2f} s。' \
            f'计时诊断中的 CPU 积分／求根约 {integration:.2f} s，仍是主要稳态开销。' \
            '<a href="performance/summary.json">查看冷／热运行与阶段耗时</a>。</p>'
    render_rows = ''.join(f'<tr><td>{backend}</td><td>{s["first_hits"]}</td><td>{s["first_hit_escapes"]}</td>'
        f'<td>{s["numerical_failures"]}</td><td>{s["safety_cap_terminations"]}</td><td>{s["seconds"]:.3f}</td></tr>'
        for backend, s in summaries.items())
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>残差 GRU 首碰渲染验证</title>
<style>body{{font:16px/1.65 system-ui;max-width:1080px;margin:36px auto;padding:0 24px;color:#18202a}}
table{{border-collapse:collapse}}td,th{{padding:8px 16px;border:1px solid #ccd3dc}}figure{{display:inline-block;margin:20px 25px 20px 0}}
img{{width:384px;max-width:100%;image-rendering:pixelated}}code{{background:#edf2f7;padding:2px 5px}}</style>
<h1>残差 GRU 的碰撞点／法线采样与渲染</h1>
<p>sigma=0.02，SE 核，固定内部 Δq=1/8；gru_nll + mixture_gru，配对种子 20261015。
首碰在 GPU 批量推进，反射后使用现有 SE renewal 条件续追踪。</p>
<p>Python / C++ scalar / Torch CPU / CUDA：{parity_rays} 条剖面 × {parity_samples} 次采样，事件分类与入口分支一致。
scalar 查询透射率最大差 {max_error('T_max_abs_error'):.3g}；所有后端碰撞 q 最大差 {max_error('q_max_abs_error'):.3g}。
三种后端的速度最大差 {max_error('speed_max_abs_error'):.3g}。</p>
<p>实际场景 {len(a):,} 条相机光线：入口、首碰分类、分支与随机流全部一致；法线分量最大差
{max(errors[k] for k in ('nx','ny','nz')):.3g}。模型预测的一致性不等同于与完整 GPIS 渲染真值一致。</p>
<table><tr><th>首碰后端</th><th>首碰数</th><th>域内逃逸</th><th>数值失败</th><th>深度上限截断</th><th>整帧秒数</th></tr>{render_rows}</table>
<p>时间包括 CUDA 初始化和 CPU 工作；当前同步批量实现尚未取得渲染加速。局部 hazard 积分和求根在 CPU 并行执行，GPU 状态仅在完整区间存活后推进。</p>
{performance_html}
<figure><img src="scalar.bmp"><figcaption>scalar，{config['render']['width']}×{config['render']['height']}，{spp} spp</figcaption></figure>
<figure><img src="torch_cuda.bmp"><figcaption>CUDA 首碰 + CPU 续追踪，相同随机种子</figcaption></figure>
<p>整图 RGB RMSE = {image_error['rmse']:.6g}。微小的首碰位置／法线差异可能改变后续离散采样分支，整条路径不保证逐位一致；深度上限截断见上表。</p>
<p>梯度头输出纵向正速度 W，再重建满足 ω·g = −kσW/ℓ 的完整 SE 条件梯度并归一化。
本轮未重新训练，也未增加三维法线监督。CDF 头尚未导出接入。</p>
<p><a href="summary.json">渲染与物理约束检查 JSON</a> · <a href="parity/summary.json">跨语言采样检查 JSON</a>
 · <a href="../../docs/FIRST_HIT_CAUSAL_RENDER_SIGMA002.md">实现与运行说明</a></p></html>'''
    (output/'summary.html').write_text(page, encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/render_first_hit_causal_gru_renewal_shader_ball_sigma002.json')
    parser.add_argument('--output', default='outputs/first_hit_causal_render_sigma002_20261016')
    parser.add_argument('--executable', default='build/Release/macrofacet_experiments.exe')
    parser.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if not args.skip_render:
        base = read_json(Path(args.config))
        for backend in ('scalar', 'torch_cuda'):
            config = copy.deepcopy(base)
            config['render'].update(width=64, height=64, samples_per_pixel=8, write_first_hit_samples=True)
            config['transport']['first_hit']['backend'] = backend
            config['output_directory'] = str(output/f'preview_{backend}')
            path = output/f'preview_{backend}_config.json'; write_json(path, config)
            with (output/f'preview_{backend}.log').open('w') as log:
                subprocess.run([str(Path(args.executable).resolve()), 'render', '--config', str(path.resolve())],
                    cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=log, stderr=log, check=True)
            print(f'{backend} render complete', flush=True)
    report(output)


if __name__ == '__main__':
    main()

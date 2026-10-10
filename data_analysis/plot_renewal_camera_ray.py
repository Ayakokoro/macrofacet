"""Compare inspect_renewal_ray output with two first-passage reference runs.

Uses the repository's SVG plotter; no plotting package or web service is needed.
The optional PFM preview requires NumPy. Run from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import struct
import zlib
from pathlib import Path

from plot_first_passage import _read_csv, _svg_line_chart


def wilson(p: float, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.0, center-radius), min(1.0, center+radius)


def median_distance(rows: list[dict], key: str) -> float | None:
    for a, b in zip(rows, rows[1:]):
        if a[key] >= .5 >= b[key] and a[key] > b[key]:
            t = (a[key]-.5)/(a[key]-b[key])
            return a['s'] + t*(b['s']-a['s'])
    return None


def preview(pfm: Path, config_path: Path, source: dict, output: Path) -> str:
    import numpy as np
    other = json.loads(config_path.read_text(encoding='utf-8-sig'))
    for key in ('field', 'seed', 'material', 'transport'):
        if source[key] != other[key]:
            raise ValueError(f'preview config differs in {key}')
    for key in ('width', 'height', 'camera_position', 'camera_target', 'vertical_fov_degrees', 'environment'):
        if source['render'][key] != other['render'][key]:
            raise ValueError(f'preview config differs in render.{key}')
    with pfm.open('rb') as f:
        if f.readline().strip() != b'PF':
            raise ValueError('expected RGB PFM')
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        pixels = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)[::-1]
    if (w, h) != (source['render']['width'], source['render']['height']):
        raise ValueError('preview image dimensions differ')
    rgb = np.clip(pixels*abs(scale), 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
    rgb = np.rint(255*rgb).astype('uint8')
    raw = b''.join(b'\0'+row.tobytes() for row in rgb)
    def chunk(tag, data):
        return struct.pack('>I', len(data))+tag+data+struct.pack('>I', zlib.crc32(tag+data))
    output.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', w,h,8,2,0,0,0))
                       +chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))
    return f"同相机已有渲染，{other['render']['samples_per_pixel']} spp；预览按 sRGB 显示。"


def chart(output: Path, name: str, rows: list[dict], title: str, kind: str = 'T',
          mode: str = 'point_linear', compare_mode: str | None = None,
          neural_label: str | None = None) -> str:
    neural_label = neural_label or mode
    if kind == 'T':
        series = [('fine','GP reference (fine)','#1565c0',''),
                  ('coarse','GP reference (coarse)','#2e7d32','6 4'),
                  ('neural',f'Neural {neural_label}','#d64b24',''),
                  ('lo','95% pointwise CI, lower','#9ebcda','3 3'),
                  ('hi','95% pointwise CI, upper','#9ebcda','3 3')]
        limits, label = (0,1), 'Transmittance T(s)'
        if compare_mode:
            series.insert(3,('comparison_neural',f'Neural {compare_mode}','#8e44ad','5 3'))
    elif kind == 'error':
        series = [('error',f'{neural_label} - GP (fine)','#d64b24',''),
                  ('ref_delta','GP coarse - fine','#2e7d32','6 4'),
                  ('zero','Zero','#888888','3 3')]
        limits, label = None, 'Absolute difference in T'
        if compare_mode:
            series.insert(1,('comparison_error',f'{compare_mode} - GP (fine)','#8e44ad','5 3'))
    else:
        series = [('mean','VDB mean / sigma','#1565c0',''),
                  ('profile_mean',f'{mode} mean / sigma','#d64b24','6 4'),
                  ('zero','Zero level','#888888','3 3')]
        limits, label = None, 'Normalized mean b(s)'
    _svg_line_chart(output/name, title=title, x_label='Distance from field entry (scene units)',
                    y_label=label, y_limits=limits,
                    series=[dict(label=label,color=color,dash=dash,points=[(r['s'],r[key]) for r in rows])
                            for key,label,color,dash in series],
                    footer='Same camera ray, VDB, sigma, kernel and positive-exterior condition. Hover in HTML for values.')
    svg = (output/name).read_text(encoding='utf-8')
    if kind == 'T':
        lo, hi = rows[0]['s'], rows[-1]['s']
        coords = [(84+686*(r['s']-lo)/(hi-lo),410-362*r['lo']) for r in rows]
        coords += [(84+686*(r['s']-lo)/(hi-lo),410-362*r['hi']) for r in reversed(rows)]
        polygon = '<polygon fill="#1565c0" fill-opacity="0.14" points="'+ ' '.join(f'{x:.3f},{y:.3f}' for x,y in coords)+'"/>'
        svg = svg.replace('<polyline',polygon+'<polyline',1)
        (output/name).write_text(svg,encoding='utf-8')
    return svg.replace('<svg ',f'<svg class="chart" data-min="{rows[0]["s"]}" data-max="{rows[-1]["s"]}" ',1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--preview-pfm', type=Path)
    parser.add_argument('--preview-config', type=Path)
    parser.add_argument('--reference-directory', type=Path, help='Reuse coarse/fine GP runs from this directory')
    compare = parser.add_mutually_exclusive_group()
    compare.add_argument('--compare-neural', type=Path, help='Overlay another profile mode for the same ray and model')
    compare.add_argument('--compare-model', type=Path, help='Overlay another model with identical ray and profile inputs')
    args = parser.parse_args()
    p = args.directory
    meta = json.loads((p/'ray.json').read_text(encoding='utf-8'))
    mode = meta['profile_mode']
    source = json.loads((p/'render_config_snapshot.json').read_text(encoding='utf-8'))
    neural = _read_csv(p/'neural_curve.csv')
    reference_root = args.reference_directory or p
    refs = {name:_read_csv(reference_root/name/'first_passage_curves.csv') for name in ('fine','coarse')}
    comparison, compare_mode, compare_label = None, None, None
    comparison_directory = args.compare_neural or args.compare_model
    neural_label = mode
    if comparison_directory:
        other = json.loads((comparison_directory/'ray.json').read_text(encoding='utf-8'))
        same_keys = ['pixel','sample_index','pixel_seed','entry_origin','direction','entry_distance',
                     'sigma','ell','kernel','start_condition','profile_maximum_step','field']
        same_keys += ['profile_mode','backend'] if args.compare_model else ['checkpoint_sha256']
        for key in same_keys:
            if meta[key] != other[key]:
                raise ValueError(f'comparison neural ray differs in {key}')
        compare_mode = other['profile_mode']
        compare_label = compare_mode
        if args.compare_model:
            current_segments = json.loads((p/'neural_segments.json').read_text(encoding='utf-8'))
            other_segments = json.loads((comparison_directory/'neural_segments.json').read_text(encoding='utf-8'))
            if len(current_segments) != len(other_segments) or any(
                    a[key] != b[key] for a,b in zip(current_segments,other_segments)
                    for key in ('begin_x','end_x','features')):
                raise ValueError('model comparison requires identical neural segment inputs')
            def model_label(metadata):
                bundle = json.loads(Path(metadata['model']).read_text(encoding='utf-8'))
                if bundle['checkpoint_sha256'] != metadata['checkpoint_sha256']:
                    raise ValueError('model file no longer matches ray checkpoint')
                return f"h{bundle['model_config']['hidden']} / {metadata['profile_mode']}"
            neural_label,compare_label = model_label(meta),model_label(other)
            if neural_label == compare_label:
                neural_label += ' ('+meta['checkpoint_sha256'][:8]+')'
                compare_label += ' ('+other['checkpoint_sha256'][:8]+')'
        comparison = _read_csv(comparison_directory/'neural_curve.csv')
        if len(comparison) != len(neural):
            raise ValueError('comparison neural query grid size differs')
    counts = {}
    conventions = {'matern_3_2_unit_decay': ('matern_3_2', 'unit_decay'),
                   'squared_exponential_unit_length': ('squared_exponential', 'unit_length')}
    if meta['kernel'] not in conventions:
        raise ValueError('unsupported inspected kernel convention')
    kernel_type, parameterization = conventions[meta['kernel']]
    for name, ref in refs.items():
        resolved = json.loads((reference_root/name/'resolved_first_passage_config.json').read_text(encoding='utf-8'))['first_passage']
        kernel = resolved['kernels'][0]
        ray = resolved['initial_condition']['rays'][0]
        if (resolved['initial_condition']['type'] != meta['start_condition']
            or ray['origin'] != meta['entry_origin'] or ray['direction'] != meta['direction']
            or resolved['process']['mean_field']['grid_file'] != meta['field']['grid_file']
            or kernel['type'] != kernel_type or kernel['parameterization'] != parameterization
            or not math.isclose(kernel['variance'],meta['sigma']**2,rel_tol=1e-12)
            or not math.isclose(kernel['length_scale'],meta['ell'],rel_tol=1e-12)):
            raise ValueError(f'{name} reference parameters do not match inspected ray')
        if len(ref)+1 != len(neural):
            raise ValueError(f'{name} query grid size differs')
        counts[name] = int(ref[0]['at_risk'])
        if counts[name] != resolved['monte_carlo']['trajectories']:
            raise ValueError('reference sample count mismatch')
    rows = []
    for i, n in enumerate(neural):
        x, s, t = float(n['x']),float(n['distance_from_entry']),float(n['transmittance'])
        r = dict(x=x,s=s,camera_s=float(n['distance_from_camera']),neural=t,zero=0.0,
                 mean=float(n['mean_sdf'])/meta['sigma'],
                 profile_mean=float(n.get('profile_mean_sdf',n.get('linearized_mean_sdf')))/meta['sigma'])
        for name, ref in refs.items():
            if i and not math.isclose(float(ref[i-1]['q_end']),x,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('neural and reference distance grids differ')
            r[name] = float(ref[i-1]['survival']) if i else 1.0
            if not 0 <= r[name] <= 1 or (rows and r[name] > rows[-1][name]):
                raise ValueError('reference survival is invalid')
            if i:
                survivors = int(ref[i-1]['at_risk'])-int(ref[i-1]['events'])
                if abs(r[name]-survivors/counts[name]) > 1e-12:
                    raise ValueError('pointwise binomial CI requires no early censoring')
        r['lo'],r['hi'] = wilson(r['fine'],counts['fine']) if i else (1.0,1.0)
        r['error'] = t-r['fine']
        r['ref_delta'] = r['coarse']-r['fine']
        if comparison is not None:
            if not math.isclose(float(comparison[i]['x']),x,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('comparison neural distance grid differs')
            r['comparison_neural'] = float(comparison[i]['transmittance'])
            r['comparison_error'] = r['comparison_neural']-r['fine']
        rows.append(r)
    # Compare reference cellwise cubic mean against direct VDB values, independent
    # of the neural point-linear approximation.
    segments = _read_csv(reference_root/'fine/first_passage_mean_segments.csv')
    k, mean_error = 0, 0.0
    for r in rows:
        while k+1 < len(segments) and r['x'] >= float(segments[k]['x_end']):
            k += 1
        seg = segments[k]
        u = (r['x']-float(seg['x_begin']))/(float(seg['x_end'])-float(seg['x_begin']))
        b = ((float(seg['cubic_a'])*u+float(seg['cubic_b']))*u+float(seg['cubic_c']))*u+float(seg['cubic_d'])
        mean_error = max(mean_error,abs(b-r['mean'])*meta['sigma'])
    if mean_error > 1e-8:
        raise ValueError(f'reference mean differs from direct VDB: {mean_error}')
    with (p/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    worst = max(rows,key=lambda r:abs(r['error']))
    ref_difference = max(abs(r['ref_delta']) for r in rows)
    # Two independent empirical survival functions: union bound over two DKW
    # bands (each failure probability .025), valid simultaneously along the ray.
    dkw = {name:math.sqrt(math.log(80)/(2*n)) for name,n in counts.items()}
    report = dict(pixel=meta['pixel'],sample_index=0,profile_mode=mode,
                  model=meta['model'],checkpoint_sha256=meta['checkpoint_sha256'],label=neural_label,
                  reference_directory=str(reference_root),reference_trajectories=counts,
                  maximum_absolute_transmittance_difference=abs(worst['error']),
                  worst_point=worst,median_distance_neural=median_distance(rows,'neural'),
                  median_distance_reference=median_distance(rows,'fine'),
                  maximum_coarse_fine_difference=ref_difference,
                  independent_reference_dkw_95_sum=sum(dkw.values()),
                  refinement_difference_below_dkw_bound=ref_difference <= sum(dkw.values()),
                  maximum_reference_mean_vs_vdb_error=mean_error,
                  maximum_profile_mean_error=max(abs(r['mean']-r['profile_mean'])*meta['sigma'] for r in rows),
                  maximum_pointwise_wilson_half_width=max((r['hi']-r['lo'])/2 for r in rows),
                  scalar_backend_difference=meta['max_scalar_backend_transmittance_difference'],
                  note='Numerical GP first-passage reference, not exact ground truth. CI covers Monte Carlo error only; step refinement is a separate empirical check. Neural error includes configured mean approximation. One primary ray, not the whole rendered image.')
    if comparison is not None:
        report['comparison_neural'] = dict(directory=str(comparison_directory),profile_mode=compare_mode,
            kind='model' if args.compare_model else 'profile',model=other['model'],
            checkpoint_sha256=other['checkpoint_sha256'],label=compare_label,
            maximum_absolute_transmittance_difference=max(abs(r['comparison_error']) for r in rows),
            median_distance_neural=median_distance(rows,'comparison_neural'),
            maximum_difference_between_neural_modes=max(abs(r['neural']-r['comparison_neural']) for r in rows))
    (p/'comparison_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    changing = [r for r in rows if .002 < r['neural'] < .998 or .002 < r['fine'] < .998
                or .002 < r.get('comparison_neural',0) < .998]
    if changing:
        a,b = changing[0]['s'],changing[-1]['s']; pad=max((b-a)*.15,rows[1]['s'])
        zoom = [r for r in rows if a-pad <= r['s'] <= b+pad]
    else:
        zoom = rows
    labels = dict(mode=mode,compare_mode=compare_label,neural_label=neural_label)
    zoom_svg = chart(p,'transmittance_zoom.svg',zoom,'Camera ray: transmittance transition',**labels)
    full_svg = chart(p,'transmittance_full.svg',rows,'Camera ray: full field interval',**labels)
    error_svg = chart(p,'transmittance_error.svg',zoom,'Difference from numerical GP reference','error',**labels)
    mean_svg = chart(p,'mean_profile.svg',zoom,'Mean field through the transmittance transition','mean',**labels)
    image = ''
    if bool(args.preview_pfm) != bool(args.preview_config):
        raise ValueError('provide both --preview-pfm and --preview-config')
    if args.preview_pfm:
        caption = preview(args.preview_pfm,args.preview_config,source,p/'scene.png')
        px,py = meta['pixel']; w,h = meta['width'],meta['height']
        image = f'<figure><div class="preview"><img src="scene.png" alt="Shader ball render"><svg viewBox="0 0 {w} {h}"><circle cx="{px+.5}" cy="{py+.5}" r="8" fill="none" stroke="#ff335f" stroke-width="2"/><path d="M{px-15} {py+.5}h30 M{px+.5} {py-15}v30" stroke="#ff335f" stroke-width="1"/></svg></div><figcaption>红色标记：像素 ({px}, {py})，从左上角以 0 开始。{caption}</figcaption></figure>'
    explanation = ('point_linear 使用逐点值／导数构建局部线性剖面，其误差包含均值近似和网络预测两部分。'
                   if mode == 'point_linear' else
                   'cubic 从 VDB 三线性插值恢复逐体素的射线三次剖面；检查其与原场的均值差，可排除局部线性近似的影响。')
    overlay_text = ''
    if comparison is not None:
        condition = '相同光线、相同分段输入' if args.compare_model else '相同光线、相同模型'
        overlay_text = f'紫色虚线叠加{condition}的 {html.escape(compare_label)} 预测。'
    ref_link = Path(os.path.relpath(reference_root/'fine/resolved_first_passage_config.json',p)).as_posix()
    tokens = dict(PIXEL=str(tuple(meta['pixel'])),SIGMA=f"{meta['sigma']:.6g}",ELL=f"{meta['ell']:.6g}",
                  MODE=html.escape(mode),BACKEND=html.escape(meta['backend']),LABEL=html.escape(neural_label),
                  EXPLANATION=explanation,OVERLAY=overlay_text,REFLINK=html.escape(ref_link,quote=True),
                  COMPARE_JSON=json.dumps(compare_label),
                  ENTRY=f"{meta['entry_distance']:.6f}",COUNT=f"{counts['fine']:,}",
                  ERROR=f"{100*abs(worst['error']):.2f}",DELTA=f"{100*ref_difference:.2f}",
                  IMAGE=image,ZOOM=zoom_svg,FULL=full_svg,ERRORSVG=error_svg,MEAN=mean_svg,
                  DATA=json.dumps(rows,separators=(',',':')),META=html.escape(json.dumps(meta,indent=2,ensure_ascii=False)))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实相机光线：Neural / GP 透射率</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#203041;background:#f2f5f8;margin:0}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:27px;margin:0}h2{font-size:20px}section,figure{background:white;border:1px solid #dee5ec;border-radius:12px;padding:18px;margin:18px 0}p{margin:10px 0}.subtitle{color:#58697a}.stats{display:flex;gap:24px;flex-wrap:wrap}.stats strong{font-size:24px;color:#bd4426}.chart{display:block;width:100%;touch-action:pan-y}button{padding:8px 16px;border:1px solid #aaa;background:white;border-radius:6px;cursor:pointer}button.active{background:#234d79;color:white}.preview{position:relative;max-width:420px;margin:auto}.preview img{display:block;width:100%}.preview svg{position:absolute;inset:0;width:100%;height:100%}figcaption{font-size:13px;color:#637383;text-align:center}#readout{background:#edf3fa;padding:12px;font:14px/1.7 ui-monospace,monospace;min-height:50px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#1c60a4}[hidden]{display:none!important}
</style><main><h1>一条真实相机光线的透射率</h1>
<p class="subtitle">像素 @PIXEL@ · 第 0 个相机采样 · @BACKEND@ / @LABEL@<br>σ = @SIGMA@，ℓ = @ELL@。横轴 s 从 VDB 有效域入口开始；距相机 = s + @ENTRY@。</p>
<div class="stats"><div>最大透射率差<br><strong>@ERROR@ 个百分点</strong></div><div>参考步长减半后的最大差<br><strong>@DELTA@ 个百分点</strong></div><div>每组 GP 参考轨迹<br><strong>@COUNT@ 条</strong></div></div>
@IMAGE@
<section><h2>透射率 T(s)</h2><p>橙线为 @LABEL@ 网络预测，蓝线为细步长 GP 数值参考，绿虚线为粗步长复查；蓝色带为逐点 95% Wilson 置信区间。@OVERLAY@</p>
<button class="active" id="zoomButton">查看下降区域</button> <button id="fullButton">查看整条光线</button>
<div id="zoom">@ZOOM@</div><div id="full" hidden>@FULL@</div>
<div id="readout">在曲线上移动鼠标或触摸，查看距离、两条透射率和置信区间。</div>
<p>这里比较的是“沿此方向走到 s 仍未首次碰撞”的概率。每个 GP 轨迹只计第一次向下零穿越；它不是端点 F(s)&gt;0 的概率，也不是单次随机场实现中的 0/1 可见性。</p></section>
<section><h2>偏差与实际均值</h2>@ERRORSVG@@MEAN@<p>网络使用 @MODE@ 分段；GP 参考使用原 VDB 三线性场沿射线的逐体素三次剖面。@EXPLANATION@ 此比较无法单独判断网络架构的影响。</p></section>
<section><h2>比较条件</h2><p>相同相机采样、场、σ、Matérn-3/2 核及入口 F(0)&gt;0 条件。网络始终复用原始完整分段，查询终点不会重新切分网络输入。神经推理使用 @BACKEND@，批处理后端以 batch=1 重放，和整图批处理可能有浮点舍入差异。</p>
<p>“真实透射率”用已有状态空间/桥接首达采样器给出数值参考。两组独立随机种子分别使用原步长与半步长，最小细分步长也减半。置信区间仅覆盖 Monte Carlo 误差，不覆盖剩余数值离散误差。这里只检查第一段相机飞行，不能据此代表其他像素、反弹后射线或整图误差。</p>
<p><a href="transmittance_zoom.svg">下载局部 SVG</a> · <a href="transmittance_full.svg">完整 SVG</a> · <a href="comparison.csv">逐点 CSV</a> · <a href="comparison_report.json">指标 JSON</a> · <a href="@REFLINK@">参考配置</a></p><details><summary>光线与模型元数据</summary><pre>@META@</pre></details></section></main>
<script>const rows=@DATA@;const compareMode=@COMPARE_JSON@;const ns='http://www.w3.org/2000/svg';
for(const svg of document.querySelectorAll('.chart')){const line=document.createElementNS(ns,'line');for(const [k,v]of Object.entries({y1:48,y2:410,stroke:'#34495e','stroke-dasharray':'4 3','pointer-events':'none',visibility:'hidden'}))line.setAttribute(k,v);svg.append(line);svg.addEventListener('pointermove',ev=>{const pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;const x=pt.matrixTransform(svg.getScreenCTM().inverse()).x;const t=Math.max(0,Math.min(1,(x-84)/686));const s=+svg.dataset.min+t*(svg.dataset.max-svg.dataset.min);const i=Math.max(0,Math.min(rows.length-1,Math.round(s/rows.at(-1).s*(rows.length-1))));const r=rows[i];line.setAttribute('x1',84+686*t);line.setAttribute('x2',84+686*t);line.setAttribute('visibility','visible');document.getElementById('readout').textContent=`s=${r.s.toFixed(6)} | 距相机=${r.camera_s.toFixed(6)} | x=s/ℓ=${r.x.toFixed(4)}\nNeural=${r.neural.toFixed(6)} | GP=${r.fine.toFixed(6)} [${r.lo.toFixed(6)}, ${r.hi.toFixed(6)}] | 差=${(100*r.error).toFixed(3)} 个百分点`+(compareMode ? ' | '+compareMode+'='+r.comparison_neural.toFixed(6) : '');});}
for(const mode of ['zoom','full'])document.getElementById(mode+'Button').onclick=()=>{for(const other of ['zoom','full']){document.getElementById(other).hidden=other!==mode;document.getElementById(other+'Button').classList.toggle('active',other===mode);}};
</script></html>'''
    for key,value in tokens.items():
        page = page.replace('@'+key+'@',value)
    (p/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

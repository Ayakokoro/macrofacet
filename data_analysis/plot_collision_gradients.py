"""Fixed-position first-collision gradient comparison, using the repository SVG UI."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import random

from plot_first_passage import _svg_line_chart
from plot_transport_camera_rays import STYLE, read, save_csv, write

MODES = ('classic_local', 'global_conditional', 'neural', 'gp', 'gp_coarse')
COLORS = ('#1565c0', '#16856b', '#d64b24', '#232a35', '#8e6dbb')
LABELS = ('classic_local', 'global_conditional', 'neural h16', 'GP first-hit (fine)', 'GP first-hit (coarse)')
METRICS = {
    'incoming': ('沿入射方向 −ω·g', 'Incoming gradient -w.g'),
    'transverse_1': ('横向分量 e₁·g', 'Transverse gradient e1.g'),
    'transverse_2': ('横向分量 e₂·g', 'Transverse gradient e2.g'),
    'magnitude': ('梯度模长 |g|', 'Gradient magnitude |g|'),
    'normal_angle': ('法线与 −ω 的夹角（度）', 'Normal angle from -w (degrees)'),
    'gx': ('世界坐标 gₓ', 'World gradient gx'),
    'gy': ('世界坐标 gᵧ', 'World gradient gy'),
    'gz': ('世界坐标 g_z', 'World gradient gz'),
}


def weighted_summary(values, weights):
    if not values or len(values) != len(weights):
        raise ValueError('missing gradient samples')
    if not all(math.isfinite(v) and math.isfinite(w) and w > 0 for v, w in zip(values, weights)):
        raise ValueError('nonfinite gradient or nonpositive retained weight')
    total = math.fsum(weights)
    mean = math.fsum(v*w for v, w in zip(values, weights))/total
    variance = math.fsum(w*(v-mean)**2 for v, w in zip(values, weights))/total
    sorted_values = sorted(zip(values, weights))
    quantiles = []
    cumulative = 0.
    for value, weight in sorted_values:
        cumulative += weight
        while len(quantiles) < 3 and cumulative >= (.1, .5, .9)[len(quantiles)]*total:
            quantiles.append(value)
    if len(quantiles) != 3:
        raise ValueError('failed weighted quantiles')
    return dict(mean=mean, stddev=math.sqrt(variance), q10=quantiles[0], q50=quantiles[1], q90=quantiles[2],
                positive_samples=len(values), sum_weight=total,
                effective_samples=total*total/math.fsum(w*w for w in weights))


def histogram(values, weights, low, high, bins):
    if high <= low or any(v < low or v > high for v in values):
        raise ValueError('histogram range must include every sample')
    width = (high-low)/bins
    masses = [0.]*bins
    for value, weight in zip(values, weights):
        masses[min(bins-1, int((value-low)/width))] += weight
    total = math.fsum(weights)
    masses = [v/total for v in masses]
    cdf = [0.]
    for mass in masses:
        cdf.append(cdf[-1]+mass)
    if not math.isclose(cdf[-1], 1., abs_tol=1e-12):
        raise ValueError('histogram lost probability mass')
    cdf[-1] = 1.
    return dict(pdf=[v/width for v in masses], cdf=cdf)


def append_sample(data, g, weight, meta):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('invalid gradient weight')
    if weight == 0:
        return
    dot = lambda v: sum(a*b for a, b in zip(g, v))
    incoming = -dot(meta['direction'])
    if incoming <= 0:
        raise ValueError('positive-weight gradient must face the incoming ray')
    t1, t2 = dot(meta['e1']), dot(meta['e2'])
    values = (incoming, t1, t2, math.sqrt(sum(v*v for v in g)),
              math.degrees(math.atan2(math.hypot(t1, t2), incoming)), *g)
    data['weights'].append(weight)
    for key, value in zip(METRICS, values):
        data[key].append(value)


def transverse_parameters(meta):
    mean=list(meta['mean_gradient'])
    variance=meta['transverse_variance']
    if meta.get('start_condition')=='collision_state':
        q=meta['target_q']
        rho=math.exp(-.5*q*q)
        mean=[m+rho*(g-b) for m,g,b in zip(mean,meta['source_hit']['gradient'],meta['birth_mean_gradient'])]
        variance*=-math.expm1(-q*q)
        if not math.isclose(variance,meta['gp_transverse_variance'],rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('inconsistent conditioned transverse variance')
        if max(abs(a-b) for a,b in zip(mean,meta['gp_transverse_mean']))>1e-12:
            raise ValueError('inconsistent conditioned transverse mean')
    return [sum(a*b for a,b in zip(mean,meta[k])) for k in ('e1','e2')],variance


def load_samples(folder, settings):
    meta = read(folder/'target.json')
    surface=meta.get('start_condition')=='collision_state'
    data = {m: dict(weights=[], **{k: [] for k in METRICS}) for m in MODES}
    counts = {m: 0 for m in MODES}
    with (folder/'gradient_samples.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            mode = row['mode']
            counts[mode] += 1
            g = [float(row[k]) for k in ('gx', 'gy', 'gz')]
            append_sample(data[mode], g, float(row['weight']), meta)
    if any(counts[m] != settings['trials_per_transport_mode'] for m in MODES[:3]):
        raise ValueError('incomplete transport samples')
    gp_summaries = {}
    for resolution, mode in (('fine', 'gp'), ('coarse', 'gp_coarse')):
        rng = random.Random(20261011+meta['pixel'][0]*1009+meta['pixel'][1]*1000003+(resolution == 'fine')*104729)
        with (folder/f'gp_{resolution}/fixed_endpoint_summary.csv').open(newline='') as stream:
            summary, = list(csv.DictReader(stream))
        if summary['start_condition'] != ('collision_state' if surface else 'positive_exterior') or summary['status'] != 'ok':
            raise ValueError('GP endpoint sampler has no positive first-hit weights')
        if not math.isclose(float(summary['target_distance']), meta['target_distance'], abs_tol=1e-12):
            raise ValueError('GP and transport target distances differ')
        if float(summary['max_endpoint_value_error']) > 1e-10:
            raise ValueError('GP endpoint constraint failed')
        mean_t,variance=transverse_parameters(meta)
        stddev = math.sqrt(variance)
        expected_mean_slope = sum(a*b for a, b in zip(meta['mean_gradient'], meta['direction']))
        if not surface and not math.isclose(float(summary['endpoint_slope_gaussian_mean'])*meta['sigma']/meta['ell'],
                            expected_mean_slope, abs_tol=1e-7):
            raise ValueError('GP full-field endpoint derivative differs from target mean')
        if surface and (float(summary['max_start_value_error'])>1e-10 or float(summary['max_birth_slope_error'])>1e-8):
            raise ValueError('GP bridge lost its first-hit observations')
        with (folder/f'gp_{resolution}/fixed_endpoint_samples.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                counts[mode] += 1
                weight = float(row['flux_weight_q'])
                expected_weight = max(0., -float(row['endpoint_slope'])) if row['survived_to_endpoint'] == '1' else 0.
                if not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError('GP survival/flux weights differ')
                if weight <= 0:
                    continue
                if not surface and float(row['start_value']) <= 0:
                    raise ValueError('GP first-hit proposal violates positive entry')
                if surface and (abs(float(row['start_value']))>1e-10 or not math.isclose(
                    float(row['beta_g']),meta['ell']/meta['sigma']*sum(a*b for a,b in zip(meta['source_hit']['gradient'],meta['direction'])),rel_tol=1e-10)):
                    raise ValueError('GP bridge birth gradient differs from the first neural hit')
                derivative = float(row['endpoint_derivative'])
                t1, t2 = (mean+rng.gauss(0., stddev) for mean in mean_t)
                g = [derivative*w+t1*e1+t2*e2 for w, e1, e2 in zip(meta['direction'], meta['e1'], meta['e2'])]
                append_sample(data[mode], g, weight, meta)
        if counts[mode] != settings['gp_proposals_per_resolution']:
            raise ValueError('incomplete GP proposals')
        weighted = weighted_summary(data[mode]['incoming'], data[mode]['weights'])
        if not math.isclose(weighted['sum_weight'], float(summary['sum_weight_q']), rel_tol=1e-10):
            raise ValueError('GP weight sum differs from C++ summary')
        if not math.isclose(weighted['mean'], float(summary['weighted_incoming_slope_mean'])*meta['sigma']/meta['ell'], rel_tol=1e-10):
            raise ValueError('GP reconstructed longitudinal mean differs')
        gp_summaries[mode] = dict(proposals=counts[mode], effective_samples=weighted['effective_samples'],
            positive_samples=weighted['positive_samples'], step=float(summary['base_step']))
    if not math.isclose(gp_summaries['gp_coarse']['step'], 2*gp_summaries['gp']['step'], rel_tol=1e-12):
        raise ValueError('GP fine step is not half the coarse step')
    return meta, data, gp_summaries


def plot_ray(folder, settings, bins=80):
    meta, data, gp = load_samples(folder, settings)
    surface=meta.get('start_condition')=='collision_state'
    collision='第二次碰撞' if surface else '首次碰撞'
    labels=tuple(label.replace('first-hit','second-hit') if surface else label for label in LABELS)
    metrics, rows, svgs = {}, [], {}
    for key, (label, axis) in METRICS.items():
        low = min(min(data[m][key]) for m in MODES)
        high = max(max(data[m][key]) for m in MODES)
        if key in ('incoming', 'magnitude', 'normal_angle'):
            low = 0.
        if key == 'normal_angle':
            high = 90.
        width = (high-low)/bins
        edges = [low+i*width for i in range(bins+1)]
        summaries = {m: weighted_summary(data[m][key], data[m]['weights']) for m in MODES}
        curves = {m: histogram(data[m][key], data[m]['weights'], low, high, bins) for m in MODES}
        refinement = max(abs(a-b) for a, b in zip(curves['gp']['cdf'], curves['gp_coarse']['cdf']))
        metrics[key] = dict(label=label, edges=edges, curves=curves, statistics=summaries,
                            gp_coarse_fine_maximum_binned_cdf_difference=refinement)
        for i in range(bins):
            for m in MODES:
                rows.append(dict(metric=key, mode=m, bin=i, lower=edges[i], upper=edges[i+1],
                                 pdf=curves[m]['pdf'][i], cdf_at_upper=curves[m]['cdf'][i+1]))
        for kind in ('pdf', 'cdf'):
            path = folder/f'{key}_{kind}.svg'
            series=[]
            for m, color, name in zip(MODES, COLORS, labels):
                points = ([(x, curves[m]['pdf'][i]) for i in range(bins) for x in (edges[i], edges[i+1])]
                          if kind == 'pdf' else list(zip(edges, curves[m]['cdf'])))
                series.append(dict(label=name, color=color, dash='6 4' if m=='gp_coarse' else '', points=points))
            _svg_line_chart(path, title='Gradient at the fixed second-collision position' if surface else 'Gradient at the same first-collision position', x_label=axis,
                y_label='Probability density' if kind=='pdf' else 'Cumulative probability',
                series=series, y_limits=(0., 1.) if kind=='cdf' else None,
                footer='Fixed distance, no distance bin. Weighted GP survival/flux; independent coarse/fine proposals.')
            svgs[key+'_'+kind] = path.read_text(encoding='utf-8').replace('<svg ', '<svg class="chart" ', 1)
    report = dict(target=meta, gp_reference=gp, metrics=metrics,
                  note='First-hit conditional distributions, physical gradients. PDF uses shared bins covering all retained samples; CDF is evaluated at those bin edges. GP is a numerical reference with sampling and discretization error.')
    if (folder/'target_selection.json').exists():report['target_selection']=read(folder/'target_selection.json')
    write(folder/'gradient_report.json', report)
    save_csv(folder/'gradient_distributions.csv', rows)
    page = r'''<!doctype html><meta charset="utf-8"><title>@COLLISION@梯度 · @PIXEL@</title><style>@STYLE@</style>
<main><h1>像素 @PIXEL@ · 固定@COLLISION@位置的梯度分布</h1>
<section id="target"><p>σ=@SIGMA@ · ℓ=@ELL@ · SE · neural h16 / @PROFILE@</p>
<p><strong>共同条件：</strong>@BIRTH@在距离 s=@DISTANCE@ 处首次向下穿过 F=0。位置 p=@POSITION@。</p>@BIRTH_DETAILS@
<p>只固定碰撞位置，梯度重新采样。不是固定原命中的梯度，也不是筛选附近距离区间。</p>
<p class="muted">入射方向 ω=@DIRECTION@<br>横向基 e₁=@E1@；e₂=@E2@。梯度使用物理单位。</p></section>
<section><label>梯度分量 <select id="metric">@OPTIONS@</select></label>
<button id="pdfButton" class="active">概率密度 PDF</button> <button id="cdfButton">累积分布 CDF</button>
<div id="chart"></div><div id="readout" class="readout">移动鼠标读取分布；可切换梯度分量及 PDF/CDF。</div>
<p id="refinement" class="muted"></p><p><a id="svgLink">下载当前 SVG</a> · <a href="gradient_distributions.csv">分布 CSV</a> · <a href="gradient_report.json">统计与条件</a></p></section>
<section class="table"><h2>当前分量的加权统计</h2><table><thead><tr><th>模式</th><th>均值</th><th>标准差</th><th>10%</th><th>中位数</th><th>90%</th><th>正权重样本</th><th>有效样本 ESS</th></tr></thead><tbody id="stats"></tbody></table></section>
<section><h2>比较口径</h2><p>classic_local：当前材质的局部高斯梯度，按负斜率通量采样；局部切平面将均值梯度归一化，因此它的横向均值也可能与完整 SDF 不同。@CONDITIONAL_NOTE@neural：同一位置的入射速度混合分布，加上渲染器的横向梯度重建。</p>
<p>@GP_NOTE@</p>
<p class="muted">黑线为细 GP 数值参考，紫虚线为独立粗参考。直方图包含所有正权重样本；曲线间的小差异可能来自有限样本。ESS 衡量权重集中程度，不是严格置信区间；粗细步长吻合也不是连续 GP 的精确证明。</p></section></main>
<script>const DATA=@DATA@,SVGS=@SVGS@,MODES=@MODES@,LABELS=@LABELS@;let kind='pdf';
const select=document.getElementById('metric');
function draw(){const k=select.value,d=DATA.metrics[k];document.getElementById('chart').innerHTML=SVGS[k+'_'+kind];
 document.getElementById('readout').textContent=d.label+' · '+kind.toUpperCase()+'：移动鼠标读取分布。';
 document.getElementById('pdfButton').classList.toggle('active',kind==='pdf');document.getElementById('cdfButton').classList.toggle('active',kind==='cdf');
 document.getElementById('svgLink').href=k+'_'+kind+'.svg';
 document.getElementById('stats').innerHTML=MODES.map((m,i)=>{const s=d.statistics[m];return '<tr><td>'+LABELS[i]+'</td>'+['mean','stddev','q10','q50','q90','positive_samples','effective_samples'].map(x=>'<td>'+s[x].toLocaleString(undefined,{maximumFractionDigits:5})+'</td>').join('')+'</tr>'}).join('');
 document.getElementById('refinement').textContent='GP 粗细参考在分箱边界的最大 CDF 差：'+(100*d.gp_coarse_fine_maximum_binned_cdf_difference).toFixed(3)+' 个百分点；这是步长与采样差异的合计。';
 const svg=document.querySelector('.chart');svg.addEventListener('pointermove',e=>{let p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;p=p.matrixTransform(svg.getScreenCTM().inverse());const f=Math.max(0,Math.min(1,(p.x-84)/686));const n=d.edges.length-1,i=Math.min(n-1,Math.floor(f*n));const lines=[d.label+' ∈ ['+d.edges[i].toFixed(5)+', '+d.edges[i+1].toFixed(5)+']'];for(let j=0;j<MODES.length;j++){const m=MODES[j];lines.push(LABELS[j]+': '+kind.toUpperCase()+'='+d.curves[m][kind][kind==='pdf'?i:i+1].toFixed(6));}document.getElementById('readout').textContent=lines.join('\n');});}
 select.onchange=draw;document.getElementById('pdfButton').onclick=()=>{kind='pdf';draw()};document.getElementById('cdfButton').onclick=()=>{kind='cdf';draw()};draw();</script>'''
    details=''
    if surface:
        birth=meta['source_hit']
        fmt=lambda v:'('+', '.join(f'{x:.7g}' for x in v)+')'
        details=f'<p id="birthCondition">第一次命中 p₁={fmt(meta["origin"])}；F(p₁)=0；完整梯度 g₁={fmt(birth["gradient"])}。横轴距离从 p₁ 开始。</p>'
        choice=report.get('target_selection',{})
        if choice.get('policy') in ('conditional_median','conditional_random'):
            details+=f'<p id="targetPolicy">目标取 neural 在域内发生第二次碰撞条件下的 {100*choice["conditional_quantile"]:.3g}% 距离分位点；该 B 段碰撞概率为 {100*choice["hit_probability"]:.6g}%。原 sample 0 的第二段逃逸：{str(not meta["source_replayed_sample_hit"])}。</p>'
        details+='<p class="muted">GP、global_conditional 和 neural 使用同一首次命中观测；classic_local 忽略此观测。没有加入第一次命中之前的相机路径存活历史。</p>'
    tokens=dict(STYLE=STYLE, COLLISION=collision,BIRTH='从同一个首次命中位置沿反射方向出发，' if surface else '从相同入口出发，',BIRTH_DETAILS=details,
        CONDITIONAL_NOTE=('global_conditional：固定第一次命中的完整梯度，在固定第二目标处采样通量加权条件高斯；共同的存活概率和 hazard 在梯度分布归一化中抵消。' if surface else 'global_conditional：入口 F&gt;0 的完整值/梯度观测，按“存活至目标 × 目标 hazard”混合后采样碰撞梯度。'),
        GP_NOTE=('GP：同时固定起点 F(p₁)=0、完整梯度 g₁ 和终点 F(p₂)=0，筛选 p₁ 与 p₂ 之间保持正值的路径，再按负斜率通量加权。SE 横向条件均值为 ∇m(p₂)+ρ(g₁−∇m(p₁)) 的横向投影，方差为 (σ/ℓ)²(1−ρ²)，ρ=exp(−(s/ℓ)²/2)。' if surface else 'GP：固定终点 F=0，要求入口及此前路径保持 F&gt;0，并按 −ω·g 加权。各向同性 SE 下，两条横向残差梯度独立于这条直线上的场值过程，因此按相应高斯分布重建完整梯度。'),
        PIXEL=html.escape(str(meta['pixel'])), SIGMA=f"{meta['sigma']:.6g}", ELL=f"{meta['ell']:.8g}",
        PROFILE=meta['neural_profile_mode'], DISTANCE=f"{meta['target_distance']:.10g}", POSITION=str(meta['position']),
        DIRECTION=str(meta['direction']), E1=str(meta['e1']), E2=str(meta['e2']),
        OPTIONS=''.join(f'<option value="{k}">{label}</option>' for k,(label,_) in METRICS.items()),
        DATA=json.dumps(report,ensure_ascii=False), SVGS=json.dumps(svgs,ensure_ascii=False),
        MODES=json.dumps(MODES),LABELS=json.dumps(labels))
    for key,value in tokens.items(): page=page.replace('@'+key+'@',value)
    (folder/'comparison.html').write_text(page,encoding='utf-8')
    return ray_summary(report)


def ray_summary(report):
    return dict(target_distance=report['target']['target_distance'],position=report['target']['position'],
        gp_reference=report['gp_reference'],incoming=report['metrics']['incoming']['statistics'],
        gp_incoming_refinement=report['metrics']['incoming']['gp_coarse_fine_maximum_binned_cdf_difference'])


def plot_suite(out,reuse_reports=False):
    out=Path(out).resolve()
    settings=read(out/'selection.json')
    records=read(out/'results.json')
    surface=settings.get('collision_index')==2
    collision='第二次碰撞' if surface else '首次碰撞'
    markers,options,table=[],[],[]
    for r in records:
        folder=out/r['directory']
        if r['status']=='evaluated':
            r['report']=ray_summary(read(folder/'gradient_report.json')) if reuse_reports and (folder/'gradient_report.json').exists() else plot_ray(folder,settings)
            s=r['report']
            table.append('<tr><td>'+str(r['pixel'])+'</td><td>'+f"{s['target_distance']:.6f}"+'</td>'+''.join(f"<td>{s['incoming'][m]['mean']:.5f}</td>" for m in MODES[:4])+f"<td>{s['gp_reference']['gp']['effective_samples']:.0f}</td></tr>")
        else:
            reason='该像素没有首次命中，或该 B 段没有可用的第二次碰撞目标。' if surface else '原 sample 0 neural 光线没有首次命中，因此没有固定碰撞位置。未重采样或人为指定命中点。'
            (folder/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>'+STYLE+'</style><main><h1>此像素无比较位置</h1><p>'+reason+'</p></main>',encoding='utf-8')
        x,y=r['pixel']; ident=r['id']
        color='#2c5a80' if r['status']=='evaluated' else '#8a9299'
        markers.append(f'<g class="marker" data-id="{ident}"><circle cx="{x}" cy="{y}" r="11" fill="{color}" stroke="white"/><text x="{x}" y="{y+4}" text-anchor="middle" fill="white" font-size="11">{ident}</text></g>')
        options.append(f'<option value="{ident}">{ident} · ({x}, {y})'+(' — 无命中' if r['status']!='evaluated' else '')+'</option>')
    active=[r for r in records if r['status']=='evaluated']
    summary=dict(selected_pixels=len(records),active_targets=len(active),missing_targets=len(records)-len(active),rays=records,
        maximum_gp_incoming_binned_cdf_refinement=max((r['report']['gp_incoming_refinement'] for r in active),default=None))
    write(out/'summary.json',summary)
    camera=Path(settings['camera_suite'])
    nav='<a data-source-link href="'+Path(os.path.relpath(camera/'comparison.html',out)).as_posix()+'">A：相机透射率</a>'
    if (camera/'view_links.json').exists():
        surface=(camera/read(camera/'view_links.json')['surface']).resolve()
        nav+=' · <a data-source-link href="'+Path(os.path.relpath(surface,out)).as_posix()+'">B：首次反射后透射率</a>'
    if (out/'view_links.json').exists():
        for key,label in (('gradient','第一次碰撞梯度'),('gradient_second','第二次碰撞梯度')):
            links=read(out/'view_links.json')
            if key in links:nav+=' · <a data-source-link href="'+html.escape(links[key],quote=True)+'">'+label+'</a>'
    page='''<!doctype html><meta charset="utf-8"><title>SE @COLLISION@条件梯度 · σ=@SIGMA@</title><style>@STYLE@</style>
<main><h1>SE 四种模式 · @COLLISION@点的条件梯度分布</h1><p>@NAV@</p>
<p class="muted">场：@FIELD@ · σ=@SIGMA@ · ℓ=@ELL@ · α=@ALPHA@<br>@TARGET_POLICY@ 每种传统/neural 模式 @TRIALS@ 次；GP 每档 @GP_TRIALS@ 个固定终点提案。</p>
<div class="layout"><aside><figure><div class="scene"><img src="scene.png" alt="same scene preview"><svg viewBox="0 0 @WIDTH@ @HEIGHT@">@MARKERS@</svg></div><figcaption>点击编号或用下拉菜单切换像素。灰色点没有可用的碰撞比较位置。</figcaption></figure>
<section><h2>选择像素</h2><select id="pixelSelect">@OPTIONS@</select><p><button id="previous">上一个</button> <button id="next">下一个</button></p><p id="selectedInfo"></p><a id="standalone">单独打开分布</a></section>
<section><h2>条件是什么</h2><p>四种模式使用同一光线、同一碰撞距离和同一位置。</p><p>@CONDITION@</p><p class="muted">保留 16 个像素，其中 @ACTIVE@ 个有比较位置。各向同性 SE 的横向残差过程与沿射线的标量过程独立；已知的起点横向梯度会影响后续横向分布。</p></section></aside>
<iframe id="rayView" title="Selected collision gradient distribution"></iframe></div>
<section class="table"><h2>入射梯度 −ω·g 的均值</h2><table><thead><tr><th>像素</th><th>固定距离</th><th>classic_local</th><th>global_conditional</th><th>neural</th><th>GP fine</th><th>GP ESS</th></tr></thead><tbody>@TABLE@</tbody></table></section>
<p><a href="summary.json">汇总数据</a> · <a href="selection.json">配置与输入哈希</a></p></main>
<script>const R=@RECORDS@;const select=document.getElementById('pixelSelect');function choose(id){const r=R.find(r=>r.id===Number(id))||R.find(r=>r.id===5)||R[0];select.value=r.id;document.getElementById('rayView').src=r.directory+'/comparison.html';document.getElementById('standalone').href=r.directory+'/comparison.html';document.getElementById('selectedInfo').textContent='像素 ('+r.pixel.join(', ')+')：'+(r.status==='evaluated'?'同一固定@COLLISION@位置':'无比较位置');document.querySelectorAll('.marker').forEach(m=>m.classList.toggle('active',Number(m.dataset.id)===r.id));document.querySelectorAll('[data-source-link]').forEach(a=>a.hash='pixel='+r.id);history.replaceState(null,'','#pixel='+r.id);}
select.onchange=()=>choose(select.value);document.querySelectorAll('.marker').forEach(m=>m.onclick=()=>choose(m.dataset.id));document.getElementById('next').onclick=()=>choose(R[(R.findIndex(r=>r.id==select.value)+1)%R.length].id);document.getElementById('previous').onclick=()=>choose(R[(R.findIndex(r=>r.id==select.value)+R.length-1)%R.length].id);choose(new URLSearchParams(location.hash.slice(1)).get('pixel')||5);</script>'''
    target=read(out/active[0]['directory']/'target.json') if active else read(Path(settings.get('source_suite',settings['camera_suite']))/next(iter(settings['rays']))/'neural/ray.json')
    surface=settings.get('collision_index')==2
    policy=('保持第一次命中，选择发生第二次碰撞条件下的 neural 距离中位数；原 sample 0 的第二段全部逃逸。' if settings.get('target_policy')=='conditional_median' else '保持第一次命中，从发生第二次碰撞条件下的 neural 距离分布随机选择目标。' if settings.get('target_policy')=='conditional_random' else '固定每个像素已有的 neural '+collision+'距离。')
    tokens=dict(STYLE=STYLE,NAV=nav,COLLISION=collision,TARGET_POLICY=policy,
        CONDITION=('固定第一次命中位置、F=0、完整梯度及反射方向，只重采样第二次碰撞处的梯度。GP、conditional 和 neural 使用该起点观测；classic_local 为忽略起点观测的基线。不包含更早相机路径的存活历史。' if surface else '在已有首次命中位置重新采样梯度；原命中的梯度只用于 B 段透射率实验。'),
        FIELD=html.escape(Path(settings['render_config']['field']['grid_file']).name),
        SIGMA=f"{target['sigma']:.6g}",ELL=f"{target['ell']:.8g}",ALPHA=str(settings['render_config']['material']['roughness']),
        TRIALS=f"{settings['trials_per_transport_mode']:,}",GP_TRIALS=f"{settings['gp_proposals_per_resolution']:,}",
        WIDTH=str(settings['render_config']['render']['width']),HEIGHT=str(settings['render_config']['render']['height']),
        MARKERS=''.join(markers),OPTIONS=''.join(options),ACTIVE=str(len(active)),TABLE=''.join(table),RECORDS=json.dumps(records))
    for key,value in tokens.items(): page=page.replace('@'+key+'@',value)
    (out/'comparison.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    plot_suite(parser.parse_args().directory)

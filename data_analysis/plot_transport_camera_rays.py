"""Plot existing camera/transport CSVs with the repository SVG and PFM helpers."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics

from plot_first_passage import _read_csv, _svg_line_chart
from plot_renewal_camera_ray import median_distance, preview, wilson

MODES = ("classic_local", "global_conditional", "neural")
COLORS = ("#1565c0", "#16856b", "#d64b24")
PAIRS = (("classic_local", "neural"), ("global_conditional", "neural"),
         ("classic_local", "global_conditional"))
STYLE = """
body{font:15px/1.6 system-ui,sans-serif;color:#203041;background:#f2f5f8;margin:0}
main{max-width:1540px;margin:auto;padding:24px}h1{font-size:27px;margin:0}h2{font-size:19px;margin:0 0 10px}
p{margin:10px 0}.muted,figcaption{font-size:13px;color:#637383}section,figure{background:white;border:1px solid #dee5ec;border-radius:12px;padding:18px;margin:0 0 18px}
a{color:#1c60a4}button,select{padding:8px 12px;border:1px solid #b3c2d0;background:white;border-radius:6px;font:inherit;cursor:pointer}button.active{background:#234d79;color:white}
.chart{display:block;width:100%;touch-action:pan-y}.readout{background:#edf3fa;padding:12px;font:13px/1.7 ui-monospace,monospace;white-space:pre-wrap;min-height:65px}
.stats{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}.stats strong{font-size:22px}.layout{display:grid;grid-template-columns:350px minmax(0,1fr);gap:20px;align-items:start}
.scene{position:relative}.scene img{display:block;width:100%}.scene svg{position:absolute;inset:0;width:100%;height:100%}.marker{cursor:pointer}.marker.active circle{fill:#d64b24;stroke:#172f4d;stroke-width:3}
iframe{width:100%;height:1400px;border:1px solid #dee5ec;border-radius:12px;background:white}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}td,th{padding:8px;text-align:right;border-bottom:1px solid #e4eaf0}tr.selected{background:#edf3fa}.table{overflow:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}[hidden]{display:none!important}@media(max-width:1050px){.layout{grid-template-columns:1fr}.scene{max-width:450px;margin:auto}main{padding:14px}}
"""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart(folder, name, rows, kind, surface=False):
    has_gp = "gp" in rows[0]
    if kind == "T":
        specs = [(m, m, c, "") for m, c in zip(MODES, COLORS)]
        if has_gp:
            specs += [("gp", "GP first-passage (fine)", "#232a35", ""),
                      ("gp_coarse", "GP first-passage (coarse)", "#8e6dbb", "6 4")]
        label, limits = "Transmittance T(s)", (0, 1)
    elif kind == "difference":
        specs = ([(m + "_minus_gp", m + " - GP (fine)", c, "") for m, c in zip(MODES, COLORS)] if has_gp else
                 [(a + "_minus_" + b, a + " - " + b, c, "") for (a, b), c in zip(PAIRS, COLORS)])
        if has_gp:
            specs.append(("gp_refinement_delta", "GP coarse - fine", "#8e6dbb", "6 4"))
        specs.append(("zero", "Zero", "#888888", "3 3"))
        label, limits = "Difference from numerical GP reference" if has_gp else "Difference in T (not GP error)", None
    else:
        specs = [("mean", "VDB mean / sigma", COLORS[0], ""),
                 ("profile_mean", "Neural input mean / sigma", COLORS[2], "6 4"),
                 ("zero", "Zero", "#888888", "3 3")]
        label, limits = "Normalized mean b(s)", None
    path = folder / name
    _svg_line_chart(path, title={"T": "SE reflected ray: surface start" if surface else "SE camera ray: transport and GP first passage" if has_gp else "SE camera ray: three transport methods",
                    "difference": "Difference from GP first passage" if has_gp else "Pairwise method differences", "mean": "Mean field along the same ray"}[kind],
                    x_label="Distance from hit (scene units)" if surface else "Distance from field entry (scene units)", y_label=label, y_limits=limits,
                    series=[dict(label=title, color=color, dash=dash, points=[(r["s"], r[key]) for r in rows])
                            for key, title, color, dash in specs],
                    footer="Same reflected ray; classic_local ignores birth observations. Bands: 95% pointwise MC CI." if surface else "Same camera ray / SE kernel. Shaded bands: 95% pointwise Monte Carlo CI." if kind == "T" else "First camera flight only; hover in HTML to read values.")
    svg = path.read_text(encoding="utf-8")
    if kind == "T":
        low, high = rows[0]["s"], rows[-1]["s"]
        bands = []
        intervals = [("local", COLORS[0]), ("conditional", COLORS[1])]
        if has_gp:
            intervals.append(("gp", "#232a35"))
        for prefix, color in intervals:
            boundary = [(r["s"], r[prefix + "_lo"]) for r in rows]
            boundary += [(r["s"], r[prefix + "_hi"]) for r in reversed(rows)]
            points = " ".join(f"{84+686*(s-low)/(high-low):.3f},{410-362*t:.3f}" for s, t in boundary)
            bands.append(f'<polygon fill="{color}" fill-opacity="0.16" points="{points}"/>')
        svg = svg.replace("<polyline", "".join(bands) + "<polyline", 1)
        path.write_text(svg, encoding="utf-8")
    return svg.replace("<svg ", f'<svg class="chart" data-min="{rows[0]["s"]}" data-max="{rows[-1]["s"]}" ', 1)


def add_gp_rows(folder, rows, meta, settings):
    """Validate independent GP references before joining by physical distance."""
    counts, steps = {}, {}
    for resolution, key in (("coarse", "gp_coarse"), ("fine", "gp")):
        directory = folder / "gp" / resolution
        fp = read(directory / "resolved_first_passage_config.json")["first_passage"]
        kernel = fp["kernels"][0]
        ray = fp["initial_condition"]["rays"][0]
        if (fp["initial_condition"]["type"] != meta["start_condition"] or
                ray["origin"] != meta["entry_origin"] or
                any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-14) for a, b in zip(ray["direction"], meta["direction"])) or
                fp["process"]["mean_field"]["grid_file"] != meta["field"]["grid_file"] or
                kernel["type"] != "squared_exponential" or kernel["parameterization"] != "unit_length" or
                not math.isclose(kernel["variance"], meta["sigma"]**2, rel_tol=1e-12) or
                not math.isclose(kernel["length_scale"], meta["ell"], rel_tol=1e-12)):
            raise ValueError(f"{resolution} GP reference does not match camera ray")
        if meta["start_condition"] == "collision_state" and ray.get("gradient") != meta["source_hit"]["gradient"]:
            raise ValueError(f"{resolution} GP reference birth gradient differs")
        ref = _read_csv(directory / "first_passage_curves.csv")
        if len(ref) + 1 != len(rows):
            raise ValueError("GP reference query grid size differs")
        n = int(ref[0]["at_risk"])
        if n != settings["trajectories_per_resolution"] or n != fp["monte_carlo"]["trajectories"]:
            raise ValueError("GP reference sample count mismatch")
        counts[resolution] = n
        steps[resolution] = fp["grid"]["step_sizes"][0]
        for i, row in enumerate(rows):
            t = 1.0
            if i:
                r = ref[i-1]
                if not math.isclose(float(r["q_end"]), row["x"], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("GP/transport distances differ")
                if r["sampler_type"] != "conditioned_grid_circulant" or r["mean_profile_type"] != "full_field":
                    raise ValueError("expected full-field SE grid first passage")
                t = float(r["survival"])
                if abs(t - (int(r["at_risk"]) - int(r["events"])) / n) > 1e-12:
                    raise ValueError("GP confidence interval requires no early censoring")
            if not math.isfinite(t) or not 0 <= t <= (rows[i-1][key] if i else 1):
                raise ValueError("invalid GP survival")
            row[key] = t
            if resolution == "fine":
                row["gp_lo"], row["gp_hi"] = wilson(t, n) if i else (1., 1.)
                row["gp_refinement_delta"] = row["gp_coarse"] - t
                for m in MODES:
                    row[m + "_minus_gp"] = row[m] - t
    if not math.isclose(steps["coarse"], 2 * steps["fine"], rel_tol=1e-12):
        raise ValueError("expected a GP step-halving check")
    # Independent check of the cellwise cubic reference mean against direct VDB queries.
    segments = _read_csv(folder / "gp/fine/first_passage_mean_segments.csv")
    index, mean_error = 0, 0.
    for row in rows:
        while index + 1 < len(segments) and row["x"] >= float(segments[index]["x_end"]):
            index += 1
        seg = segments[index]
        u = (row["x"] - float(seg["x_begin"])) / (float(seg["x_end"]) - float(seg["x_begin"]))
        b = ((float(seg["cubic_a"]) * u + float(seg["cubic_b"])) * u + float(seg["cubic_c"])) * u + float(seg["cubic_d"])
        mean_error = max(mean_error, abs(b - row["mean"]) * meta["sigma"])
    if mean_error > 1e-8:
        raise ValueError(f"GP reference mean differs from VDB: {mean_error}")
    gap = max(abs(r["gp_refinement_delta"]) for r in rows)
    dkw = sum(math.sqrt(math.log(80) / (2 * n)) for n in counts.values())
    return dict(trajectories=counts, normalized_steps=steps, maximum_coarse_fine_difference=gap,
                maximum_reference_mean_vs_vdb_error=mean_error,
                independent_reference_dkw_95_sum=dkw, refinement_difference_below_dkw_bound=gap <= dkw,
                maximum_pointwise_wilson_half_width=max((r["gp_hi"]-r["gp_lo"])/2 for r in rows),
                note="Numerical GP first-passage reference. CI describes Monte Carlo uncertainty, not remaining discretization error; coarse/fine agreement is empirical.")


def merge_ray(folder, selection, gp_settings=None):
    meta = read(folder / "neural/ray.json")
    source = selection["render_config"]
    if read(folder / "neural/render_config_snapshot.json") != source:
        raise ValueError("neural render config does not match selection")
    resolved = read(folder / "transport/resolved_config.json")
    config = read(folder / "transport_config.json")
    assert config["field"] == source["field"] and resolved["field"]["kernel_type"] == "squared_exponential"
    assert resolved["derived"]["kernel_parameterization"] == "unit_length"
    assert math.isclose(resolved["field"]["sigma"], meta["sigma"], rel_tol=1e-12)
    ray = config["transmittance"]["rays"][0]
    assert ray["origin"] == meta["entry_origin"] and ray["direction"] == meta["direction"]
    surface = meta["start_condition"] == "collision_state"
    assert surface == (selection.get("flight") == "after_first_hit")
    birth = config["transmittance"]["conditional_birth"]
    if surface:
        hit = meta["source_hit"]
        assert birth == dict(policy="fixed_observation", value=0., gradient=hit["gradient"])
        assert ray["origin"] == hit["position"] and ray["direction"] == hit["outgoing_direction"]
        projected = sum(a*b for a, b in zip(hit["gradient"], ray["direction"]))
        assert hit["value"] == 0 and projected > 0
        assert math.isclose(projected*meta["ell"]/meta["sigma"], hit["normalized_outward_derivative"], rel_tol=1e-12)
    else:
        assert birth["policy"] == "sample_positive_exterior"
    geometry = _read_csv(folder / "transport/transmittance_rays.csv")[0]
    assert abs(float(geometry["medium_entry_distance"])) < 1e-12
    assert math.isclose(float(geometry["medium_length"]), ray["max_distance"], abs_tol=1e-12)
    neural = _read_csv(folder / "neural/neural_curve.csv")
    transport = _read_csv(folder / "transport/transmittance_curves.csv")
    groups = {m: [r for r in transport if r["mode"] == m] for m in MODES[:2]}
    assert all(len(g) == len(neural) == selection["bins"] + 1 for g in groups.values())
    rows = []
    for i, n in enumerate(neural):
        s = float(n["distance_from_entry"])
        row = dict(s=s, camera_s=float(n["distance_from_camera"]), x=float(n["x"]),
                   neural=float(n["transmittance"]), mean=float(n["mean_sdf"]) / meta["sigma"],
                   profile_mean=float(n["profile_mean_sdf"]) / meta["sigma"], zero=0.)
        for mode, prefix in zip(MODES[:2], ("local", "conditional")):
            r = groups[mode][i]
            assert math.isclose(float(r["distance"]), s, abs_tol=1e-12)
            assert int(r["trials"]) == selection["trials_per_mode"]
            row[mode] = float(r["transmittance"])
            row[prefix + "_lo"], row[prefix + "_hi"] = float(r["ci_low"]), float(r["ci_high"])
            assert 0 <= row[prefix + "_lo"] <= row[mode] + 1e-14 <= row[prefix + "_hi"] + 2e-14 <= 1 + 2e-14
        for a, b in PAIRS:
            row[a + "_minus_" + b] = row[a] - row[b]
        assert all(math.isfinite(v) for v in row.values())
        for mode in MODES:
            assert 0 <= row[mode] <= (rows[-1][mode] if rows else 1) + 1e-12
        rows.append(row)
    gp_report = add_gp_rows(folder, rows, meta, gp_settings) if gp_settings else None
    report = dict(pixel=meta["pixel"], sample_index=0, kernel=meta["kernel"], sigma=meta["sigma"], ell=meta["ell"],
                  start_condition=meta["start_condition"],
                  trials_per_sampled_mode=selection["trials_per_mode"],
                  terminal_transmittance={m: rows[-1][m] for m in MODES},
                  median_distance={m: median_distance(rows, m) for m in MODES}, pairwise={},
                  scalar_backend_difference=meta["max_scalar_backend_transmittance_difference"],
                  note="All pairwise differences are between methods, not errors relative to GP truth.")
    if surface:
        report["source_hit"] = meta["source_hit"]
        report["conditioning_scope"] = meta["conditioning_scope"]
    for a, b in PAIRS:
        key = a + "_minus_" + b
        worst = max(rows, key=lambda r: abs(r[key]))
        report["pairwise"][key] = dict(maximum_absolute_difference=abs(worst[key]), signed_at_maximum=worst[key],
                                      distance_from_entry=worst["s"], values={a: worst[a], b: worst[b]})
    if gp_report:
        report["gp_reference"] = gp_report
        report["terminal_transmittance"]["gp"] = rows[-1]["gp"]
        report["median_distance"]["gp"] = median_distance(rows, "gp")
        report["against_gp"] = {}
        for mode in MODES:
            worst = max(rows, key=lambda r: abs(r[mode + "_minus_gp"]))
            report["against_gp"][mode] = dict(maximum_absolute_difference=abs(worst[mode + "_minus_gp"]),
                signed_at_maximum=worst[mode + "_minus_gp"], distance_from_entry=worst["s"],
                method_value=worst[mode], gp_value=worst["gp"], gp_ci=[worst["gp_lo"], worst["gp_hi"]])
        report["note"] = "Errors are relative to the fine numerical GP first-passage reference, including its sampling uncertainty and remaining discretization error. Neural includes configured mean-profile approximation."
    save_csv(folder / "comparison.csv", rows)
    write(folder / "comparison_report.json", report)
    return rows, meta, report


def ray_page(folder, rows, meta, report):
    has_gp = "gp_reference" in report
    surface = meta["start_condition"] == "collision_state"
    active = [i for i, r in enumerate(rows) if any(.01 < r[m] < .99 for m in (*MODES, "gp") if m in r)]
    if active:
        lo, hi = active[0], active[-1]
        pad = max(8, (hi - lo) // 8)
        zoom = rows[max(0, lo - pad):min(len(rows), hi + pad + 1)]
    else:
        zoom = rows
    charts = {"ZOOM": chart(folder, "transmittance_zoom.svg", zoom, "T", surface),
              "FULL": chart(folder, "transmittance_full.svg", rows, "T", surface),
              "DIFFERENCE": chart(folder, "transmittance_difference.svg", zoom, "difference", surface),
              "MEAN": chart(folder, "mean_profile.svg", zoom, "mean", surface)}
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>像素 @PIXEL@ · SE 透射率</title><style>@STYLE@ main{padding:16px}h1{font-size:22px}</style>
<main><h1>像素 @PIXEL@ · 第 0 个相机采样@FLIGHT_TITLE@</h1><p class="muted">SE · σ=@SIGMA@ · ℓ=@ELL@ · neural h16 / @PROFILE@ / @BACKEND@<br>@DISTANCE_NOTE@</p>@HIT@
<section><h2>透射率 T(s)</h2><p>蓝：classic_local；绿：global_conditional；橙：neural。传统方法各采样 @TRIALS@ 次。@GP_LEGEND@阴影为逐点 95% Wilson 置信区间。</p>
<button id="zoomButton" class="active">下降区域</button> <button id="fullButton">完整光线</button>
<div id="zoom">@ZOOM@</div><div id="full" hidden>@FULL@</div><div class="readout" id="readout">在曲线上移动鼠标或触摸，读取三种方法的透射率、置信区间和距离。</div>
<p class="muted">@SURVIVAL_NOTE@</p></section>
<section><h2>@DIFFERENCE_TITLE@</h2>@DIFFERENCE@<p class="muted">@DIFFERENCE_NOTE@</p>@GP_STATS@</section>
<section><h2>沿光线的均值场</h2>@MEAN@<p class="muted">neural 使用配置中的 point_linear 分段；global_conditional 使用完整 VDB 均值；classic_local 使用渲染器的烘焙密度和局部 NDF。</p></section>
<p><a href="comparison.csv">逐点 CSV</a> · <a href="comparison_report.json">指标 JSON</a> · <a href="transmittance_full.svg">完整 SVG</a> · <a href="transmittance_zoom.svg">局部 SVG</a>@GP_LINKS@</p>
<details><summary>光线、模型和比较元数据</summary><pre>@META@</pre></details></main>
<script>const rows=@DATA@;const ns='http://www.w3.org/2000/svg';
for(const svg of document.querySelectorAll('.chart')){
 const line=document.createElementNS(ns,'line');
 for(const [k,v]of Object.entries({y1:48,y2:410,stroke:'#34495e','stroke-dasharray':'4 3','pointer-events':'none',visibility:'hidden'}))line.setAttribute(k,v);
 svg.append(line);
 svg.addEventListener('pointermove',ev=>{
  const pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;
  const x=pt.matrixTransform(svg.getScreenCTM().inverse()).x,t=Math.max(0,Math.min(1,(x-84)/686));
  const s=+svg.dataset.min+t*(svg.dataset.max-svg.dataset.min);
  const i=Math.max(0,Math.min(rows.length-1,Math.round(s/rows.at(-1).s*(rows.length-1)))),r=rows[i];
  line.setAttribute('x1',84+686*t);line.setAttribute('x2',84+686*t);line.setAttribute('visibility','visible');
  let value=`s=${r.s.toFixed(6)} | @PATH_LABEL@=${r.camera_s.toFixed(6)} | x=s/ℓ=${r.x.toFixed(4)}\nclassic_local=${r.classic_local.toFixed(6)} [${r.local_lo.toFixed(6)}, ${r.local_hi.toFixed(6)}]\nglobal_conditional=${r.global_conditional.toFixed(6)} [${r.conditional_lo.toFixed(6)}, ${r.conditional_hi.toFixed(6)}]\nneural=${r.neural.toFixed(6)}`;
  if(r.gp!==undefined)value+=`\nGP fine=${r.gp.toFixed(6)} [${r.gp_lo.toFixed(6)}, ${r.gp_hi.toFixed(6)}] | GP coarse=${r.gp_coarse.toFixed(6)}\n相对 GP: classic=${(100*r.classic_local_minus_gp).toFixed(3)} pp | conditional=${(100*r.global_conditional_minus_gp).toFixed(3)} pp | neural=${(100*r.neural_minus_gp).toFixed(3)} pp`;
  else value+=` | classic−neural=${(100*r.classic_local_minus_neural).toFixed(3)} pp | conditional−neural=${(100*r.global_conditional_minus_neural).toFixed(3)} pp`;
  document.getElementById('readout').textContent=value;
 });
}
for(const mode of ['zoom','full'])document.getElementById(mode+'Button').onclick=()=>{for(const other of ['zoom','full']){document.getElementById(other).hidden=other!==mode;document.getElementById(other+'Button').classList.toggle('active',other===mode);}};
</script></html>'''
    gp_legend, gp_stats = "", ""
    difference_title = "方法间差值"
    difference_note = "这些是方法之间的差异，未使用 GP 首达真值，不能据此判定哪种方法更准确。"
    if has_gp:
        gp = report["gp_reference"]
        gp_legend = f'黑线：GP 首达细参考；紫虚线：粗参考，各 {gp["trajectories"]["fine"]:,} 条独立轨迹。'
        difference_title = "相对 GP 首达参考的偏差"
        difference_note = "三条实线为各方法减去 GP 细参考；紫虚线为 GP 粗参考减去细参考。参考使用完整 VDB 均值和条件循环嵌入网格首达采样。置信区间只覆盖 Monte Carlo 噪声，步长减半是数值离散的经验检查；它仍是数值参考，不是解析精确真值。"
        gp_stats = '<div class="stats">' + ''.join(f'<div>{m} 最大绝对偏差<br><strong>{100*report["against_gp"][m]["maximum_absolute_difference"]:.2f} pp</strong></div>' for m in MODES) + '</div>'
        gp_stats += f'<p class="muted">GP 归一化步长：{gp["normalized_steps"]["coarse"]:.7f} / {gp["normalized_steps"]["fine"]:.7f}；粗细最大差：{100*gp["maximum_coarse_fine_difference"]:.3f} pp。</p>'
    hit_html = ""
    if surface:
        hit = meta["source_hit"]
        vector = lambda key: "(" + ", ".join(f"{v:.7f}" for v in hit[key]) + ")"
        hit_html = ('<section id="birthCondition"><h2>首次命中的共同起点条件</h2><p>' +
            f'位置 p = {vector("position")}<br>F(p) = 0；完整梯度 g = {vector("gradient")}<br>' +
            f'出射方向 ω = {vector("outgoing_direction")}<br>ω·g = {hit["outward_derivative"]:.7f}；' +
            f'归一化起点斜率 ℓ(ω·g)/σ = {hit["normalized_outward_derivative"]:.7f}。</p>' +
            '<p>global_conditional 使用固定场值与梯度；neural 使用 B 初始化；GP 条件化于相同起点观测。classic_local 不使用该观测，是同一出射光线的局部基线。</p>' +
            '<p class="muted">命中来自 SE neural 的 sample 0 重放，使用当前后端、batch=1；可能与完整 wavefront 批次有浮点差异。这里没有加入之前整段路径的存活条件。F(p)=0 是随机场观测，不要求均值 SDF 为零。</p></section>')
    tokens = dict(STYLE=STYLE, PIXEL=str(tuple(meta["pixel"])), SIGMA=f'{meta["sigma"]:.6g}', ELL=f'{meta["ell"]:.6g}',
                  FLIGHT_TITLE=" · 首次反射后的 B 段" if surface else " · 相机 A 段",
                  DISTANCE_NOTE=(f'横轴从首次命中点开始；从相机累计的折线路程 = s + {meta["entry_distance"]:.6f}。' if surface else f'横轴从有效域入口开始；距相机 = s + {meta["entry_distance"]:.6f}。'),
                  PATH_LABEL="累计路程" if surface else "距相机", HIT=hit_html,
                  SURVIVAL_NOTE="T(s) 表示从已知命中点沿反射方向走到 s，尚未再次碰撞的概率；T(0)=1。" if surface else "T(s) 表示走到 s 尚未发生首次碰撞的概率；入口之前为真空，T=1。",
                  PROFILE=html.escape(meta["profile_mode"]), BACKEND=html.escape(meta["backend"]),
                  ENTRY=f'{meta["entry_distance"]:.6f}', TRIALS=f'{report["trials_per_sampled_mode"]:,}',
                  GP_LEGEND=gp_legend, GP_STATS=gp_stats, DIFFERENCE_TITLE=difference_title, DIFFERENCE_NOTE=difference_note,
                  GP_LINKS=' · <a href="gp/reference_coarse.json">GP 粗参考配置</a> · <a href="gp/reference_fine.json">GP 细参考配置</a>' if has_gp else '',
                  META=html.escape(json.dumps(dict(ray=meta, comparison=report), indent=2, ensure_ascii=False)),
                  DATA=json.dumps(rows, separators=(",", ":"), allow_nan=False), **charts)
    for key, value in tokens.items():
        page = page.replace("@" + key + "@", value)
    (folder / "comparison.html").write_text(page, encoding="utf-8")


def plot_suite(out):
    out = Path(out).resolve()
    selection = read(out / "selection.json")
    surface = selection.get("flight") == "after_first_hit"
    records = read(out / "results.json")
    assert [r["pixel"] for r in records] == selection["pixels"]
    gp_settings = read(out / "gp_selection.json") if (out / "gp_selection.json").exists() else None
    if gp_settings:
        if [r["pixel"] for r in read(out / "gp_results.json")] != selection["pixels"]:
            raise ValueError("wait until all selected GP references are complete")
        if gp_settings["field_sha256"] != selection["field_sha256"]:
            raise ValueError("GP reference field differs from the transport field")
    reports = []
    for record in records:
        folder = out / record["directory"]
        if record["status"] == "evaluated":
            rows, meta, report = merge_ray(folder, selection, gp_settings)
            ray_page(folder, rows, meta, report)
        else:
            assert record["status"] in ("domain_miss", "camera_escape")
            report = dict(pixel=record["pixel"], terminal_transmittance={m: None if surface else 1.0 for m in MODES},
                          median_distance={m: None for m in MODES}, pairwise={})
            if gp_settings:
                report["terminal_transmittance"]["gp"] = None if surface else 1.0
                report["median_distance"]["gp"] = None
            reason = "这条相机光线未进入有效域。" if record["status"] == "domain_miss" else "这条相机光线第 0 次采样未碰撞，直接离开有效域。"
            note = "没有首次命中位置和梯度，因此没有 B 段透射率。该像素不计入命中后比较的误差汇总；未重采样来强制产生碰撞。" if surface else "classic_local、global_conditional、neural 沿整条光线均为真空透射率 T=1。有效域外不采样 GP，参考同样为 T=1。"
            (folder / "comparison.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>' + STYLE +
                f'</style><main><h1>像素 {tuple(record["pixel"])}</h1><section><p>{reason}</p><p>{note}</p></section></main></html>', encoding="utf-8")
        reports.append(dict(**record, report=report))
    evaluated = [r for r in reports if r["status"] == "evaluated"]
    summary = dict(selected_pixels=len(records), active_rays=len(evaluated),
                   domain_misses=sum(r["status"] == "domain_miss" for r in records),
                   camera_escapes=sum(r["status"] == "camera_escape" for r in records),
                   flight="after_first_hit" if surface else "camera",
                   trials_per_sampled_mode=selection["trials_per_mode"], pairs={}, rays=reports,
                   note="Maxima are over 4097 distance queries. Means average per-ray maxima over selected active rays; not image errors or GP accuracy.")
    for a, b in PAIRS:
        key = a + "_minus_" + b
        if evaluated:
            worst = max(evaluated, key=lambda r: r["report"]["pairwise"][key]["maximum_absolute_difference"])
            summary["pairs"][key] = dict(mean_per_ray_maximum=statistics.mean(r["report"]["pairwise"][key]["maximum_absolute_difference"] for r in evaluated),
                worst_pixel=worst["pixel"], **worst["report"]["pairwise"][key])
    if evaluated:
        # Union bound over the two independently sampled modes on all active rays.
        summary["familywise_95_mc_epsilon_per_curve"] = math.sqrt(math.log(4 * len(evaluated) / .05) / (2 * selection["trials_per_mode"]))
    if gp_settings and evaluated:
        summary["gp_reference"] = dict(trajectories_per_resolution=gp_settings["trajectories_per_resolution"],
            largest_coarse_fine_difference=max(r["report"]["gp_reference"]["maximum_coarse_fine_difference"] for r in evaluated),
            largest_reference_mean_vs_vdb_error=max(r["report"]["gp_reference"]["maximum_reference_mean_vs_vdb_error"] for r in evaluated),
            all_refinement_differences_below_per_ray_dkw=all(r["report"]["gp_reference"]["refinement_difference_below_dkw_bound"] for r in evaluated))
        summary["against_gp"] = {}
        for mode in MODES:
            worst = max(evaluated, key=lambda r: r["report"]["against_gp"][mode]["maximum_absolute_difference"])
            summary["against_gp"][mode] = dict(
                mean_per_ray_maximum=statistics.mean(r["report"]["against_gp"][mode]["maximum_absolute_difference"] for r in evaluated),
                worst_pixel=worst["pixel"], **worst["report"]["against_gp"][mode])
        summary["note"] = "GP errors use the fine numerical first-passage reference. Means average per-ray maxima on selected active rays, not image errors. Monte Carlo confidence does not cover remaining discretization error."
    write(out / "summary.json", summary)
    table_rows = []
    markers, options, table = [], [], []
    for r in reports:
        x, y = r["pixel"]
        i, report = r["id"], r["report"]
        missing = surface and r["status"] != "evaluated"
        options.append(f'<option value="{i}">{i} · ({x}, {y})' + (' · 无后续光线' if missing else ' · 真空' if r["status"] == "domain_miss" else '') + '</option>')
        markers.append(f'<g class="marker" data-id="{i}" role="button" tabindex="0" aria-label="像素 ({x},{y})"><title>{i}: ({x},{y})</title><circle cx="{x+.5}" cy="{y+.5}" r="11" fill="#234d79" stroke="white" stroke-width="1.5"/><text x="{x+.5}" y="{y+4.5}" text-anchor="middle" font-size="11" font-weight="bold" fill="white">{i}</text></g>')
        flat = dict(id=i, pixel_x=x, pixel_y=y, status=r["status"])
        values = []
        for a, b in PAIRS:
            key = a + "_minus_" + b
            gap = report["pairwise"].get(key, {}).get("maximum_absolute_difference", None if missing else 0.)
            flat["max_abs_" + key] = gap
            values.append('—' if gap is None else f'{100*gap:.2f}')
        for m in MODES:
            flat["terminal_" + m] = report["terminal_transmittance"][m]
            flat["median_distance_" + m] = report["median_distance"][m]
        table_modes = MODES
        if gp_settings:
            values = []
            for mode in MODES:
                error = report.get("against_gp", {}).get(mode, {}).get("maximum_absolute_difference", None if missing else 0.)
                flat["max_abs_" + mode + "_minus_gp"] = error
                values.append('—' if error is None else f'{100*error:.2f}')
            gap = report.get("gp_reference", {}).get("maximum_coarse_fine_difference", None if missing else 0.)
            flat["gp_coarse_fine_maximum_difference"] = gap
            flat["terminal_gp"] = report["terminal_transmittance"]["gp"]
            flat["median_distance_gp"] = report["median_distance"]["gp"]
            values.append('—' if gap is None else f'{100*gap:.3f}')
            table_modes = (*MODES, "gp")
        table_rows.append(flat)
        table.append(f'<tr data-id="{i}"><td><a href="{r["directory"]}/comparison.html" target="ray_view" data-id="{i}">{i}</a></td><td>({x}, {y})</td>' +
                     ''.join(f'<td>{v}</td>' for v in values) +
                     ''.join('<td>—</td>' if report["terminal_transmittance"][m] is None else f'<td>{report["terminal_transmittance"][m]:.4f}</td>' for m in table_modes) + '</tr>')
    save_csv(out / "summary.csv", table_rows)
    source = selection["render_config"]
    pfms = list((out / "preview").glob("*.pfm"))
    if len(pfms) != 1:
        raise ValueError("expected exactly one scene preview PFM")
    caption = preview(pfms[0], out / "preview_config.json", source, out / "scene.png")
    default = next((r["id"] for r in records if r["pixel"] == [source["render"]["width"] // 2, source["render"]["height"] // 2]), records[0]["id"])
    initial = next(r for r in records if r["id"] == default)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>@FIELD_NAME@ · SE @FLIGHT_TITLE@透射率</title><style>@STYLE@</style>
<main><h1>SE @FLIGHT_TITLE@透射率对比@GP_TITLE@</h1>@NAV@<p class="muted">场：@FIELD_NAME@@FIELD_PARAMETERS@<br>classic_local / global_conditional / neural h16 · @COUNT@ 个固定像素 · 同一相机第 0 次采样 · 每种传统方法每条光线 @TRIALS@ 次@GP_BUDGET@</p>
<div class="layout"><div><figure><div class="scene"><img src="scene.png" alt="SE shader ball 像素选择预览"><svg viewBox="0 0 @WIDTH@ @HEIGHT@">@MARKERS@</svg></div><figcaption>点击编号切换光线；坐标从左上角以 0 开始。@CAPTION@ 此图用于定位像素。</figcaption></figure>
<section><h2>选择像素</h2><select id="pixelSelect" aria-label="选择像素">@OPTIONS@</select><p><button id="previous">上一个</button> <button id="next">下一个</button></p><p id="selectedInfo"></p><a id="openRay" href="@INITIAL@/comparison.html" target="_blank">单独打开这条光线</a></section>
<section><h2>比较口径</h2>@CONDITION_NOTE@<p>局部模型使用渲染器的烘焙密度和局部 NDF；条件模型使用 SE 条件统计；neural 使用训练好的 SE h16 和 point_linear 输入。</p><p class="muted">@REFERENCE_NOTE@</p></section></div>
<iframe id="rayView" name="ray_view" title="所选像素的透射率曲线" src="@INITIAL@/comparison.html"></iframe></div>
<section><h2>所有选中像素</h2><p class="muted">@TABLE_NOTE@终点列为离开有效域时的透射率。点击编号查看曲线。</p><div class="table"><table><thead><tr><th>编号</th><th>像素</th>@COLUMNS@</tr></thead><tbody>@TABLE@</tbody></table></div></section>
<p><a href="summary.csv">汇总 CSV</a> · <a href="summary.json">完整报告 JSON</a> · <a href="selection.json">像素选择与输入哈希</a>@GP_LINKS@</p></main>
<script>const records=@RECORDS@;const select=document.getElementById('pixelSelect');const frame=document.getElementById('rayView');
function choose(id){const r=records.find(r=>r.id===Number(id));if(!r)return;select.value=r.id;const url=r.directory+'/comparison.html';if(frame.getAttribute('src')!==url)frame.src=url;document.getElementById('openRay').href=url;document.getElementById('selectedInfo').textContent=`像素 (${r.pixel.join(', ')})`+(@SURFACE@?(r.status==='evaluated'?' · 同一次命中后的反射光线':' · 无首次命中，无后续光线'):(r.status==='domain_miss'?' · 未进入有效域，三种 T 均为 1':' · 同一相机光线'));document.querySelectorAll('.marker').forEach(e=>e.classList.toggle('active',Number(e.dataset.id)===r.id));document.querySelectorAll('tr[data-id]').forEach(e=>e.classList.toggle('selected',Number(e.dataset.id)===r.id));for(const a of document.querySelectorAll('[data-flight-link]'))a.href=a.getAttribute('href').split('#')[0]+'#pixel='+r.id;history.replaceState(null,'','#pixel='+r.id);}
select.onchange=()=>choose(select.value);for(const e of document.querySelectorAll('[data-id]')){if(e.tagName==='TR')continue;e.onclick=ev=>{ev.preventDefault();choose(e.dataset.id)};e.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();choose(e.dataset.id)}};}
function step(delta){const i=records.findIndex(r=>r.id===Number(select.value));choose(records[(i+delta+records.length)%records.length].id)}document.getElementById('previous').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);
choose(Number(location.hash.split('=')[1])||@DEFAULT@);
</script></html>'''
    columns = '<th>|classic−neural| 最大</th><th>|conditional−neural| 最大</th><th>|classic−conditional| 最大</th><th>终点 classic</th><th>终点 conditional</th><th>终点 neural</th>'
    reference_note = "未加入 GP 首达参考，因此曲线间差异不代表某种方法的真值误差。置信带只描述采样噪声。"
    table_note = "前三列为沿整条光线的最大绝对差，单位：百分点（pp）。"
    gp_title, gp_budget = "", ""
    if gp_settings:
        columns = '<th>|classic−GP| 最大</th><th>|conditional−GP| 最大</th><th>|neural−GP| 最大</th><th>GP 粗细最大差</th><th>终点 classic</th><th>终点 conditional</th><th>终点 neural</th><th>终点 GP</th>'
        gp_title = " · GP 首达参考"
        gp_budget = f' · GP 每档 {gp_settings["trajectories_per_resolution"]:,} 条轨迹'
        reference_note = "GP 曲线统计 SE 随机场在完整 VDB 均值上首次向下穿越零点前的存活概率，同样条件于入口 F&gt;0。黑线为细参考，紫虚线为粗参考，步长上限分别为 1/128、1/64。它是数值真值参考；置信带覆盖 Monte Carlo 噪声，粗细差另用于检查离散影响。"
        table_note = "前三列为各方法相对 GP 细参考的沿程最大绝对偏差，第四列为 GP 粗细最大差，单位：百分点（pp）。"
    condition_note = '<p>曲线表示首次碰撞前的透射率。global_conditional 每次在入口采样 F&gt;0 的场值与梯度后追踪，再对这些观测取平均；neural 使用同一入口的 A 初始化。</p>'
    if surface:
        condition_note = (f'<p>{len(records)} 个固定像素中，{len(evaluated)} 个 sample 0 有首次命中；其余没有后续 B 段。</p>' +
            '<p>同一 neural 命中位置 p、完整梯度 g 和反射方向 ω。global_conditional 固定观测 F(p)=0、∇F(p)=g；neural 使用 B 初始化；GP 首达使用相同观测。</p>' +
            '<p><strong>classic_local 不使用起点观测</strong>，保留同一出射光线上的局部模型基线。没有把此前整条路径的存活历史加入条件。</p>')
        reference_note = reference_note.replace('同样条件于入口 F&gt;0', '条件于命中点 F=0 和沿出射方向的导数 ω·g；当前各向同性 SE 核沿直线的标量过程只需要这个梯度投影')
    nav = ""
    if (out / "view_links.json").exists():
        links = read(out / "view_links.json")
        nav = '<p class="flightNavigation">' + ' · '.join(
            f'<a data-flight-link="{key}" href="{html.escape(links[key], quote=True)}">{label}</a>'
            for key, label in (("camera", "A：相机入射"), ("surface", "B：首次命中后"),
                               ("gradient", "首次碰撞梯度分布"),
                               ("gradient_second", "第二次碰撞梯度分布")) if key in links) + '</p>'
    tokens = dict(STYLE=STYLE, COUNT=str(len(records)), TRIALS=f'{selection["trials_per_mode"]:,}',
                  FIELD_NAME=html.escape(Path(source["field"]["grid_file"]).name),
                  FIELD_PARAMETERS=(f' · σ={evaluated[0]["report"]["sigma"]:.6g}' if evaluated else '') +
                      (f' · ℓ={evaluated[0]["report"]["ell"]:.6g}' if evaluated and
                       all(math.isclose(r["report"]["ell"], evaluated[0]["report"]["ell"], rel_tol=1e-12) for r in evaluated) else '') +
                      (f' · α={source["material"]["roughness"]:.6g}' if 'roughness' in source['material'] else ''),
                  FLIGHT_TITLE="首次命中后" if surface else "相机光线", CONDITION_NOTE=condition_note,
                  NAV=nav, SURFACE=json.dumps(surface),
                  GP_TITLE=gp_title, GP_BUDGET=gp_budget, REFERENCE_NOTE=reference_note, TABLE_NOTE=table_note, COLUMNS=columns,
                  GP_LINKS=' · <a href="gp_selection.json">GP 参考参数与哈希</a>' if gp_settings else '',
                  WIDTH=str(source["render"]["width"]), HEIGHT=str(source["render"]["height"]),
                  MARKERS="".join(markers), OPTIONS="".join(options), TABLE="".join(table),
                  CAPTION=html.escape(caption), INITIAL=initial["directory"], DEFAULT=str(default),
                  RECORDS=json.dumps(records, separators=(",", ":")))
    for key, value in tokens.items():
        page = page.replace("@" + key + "@", value)
    (out / "comparison.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_suite(parser.parse_args().directory)
